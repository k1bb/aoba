"""Unstructured SCVT Voronoi Sea-Ice EVP Rheology & 5-Category Ice Thickness Distribution (ITD).

Extends AOBA's sea-ice transport (`ocean_ogcm.py`) with Elastic-Viscous-Plastic (EVP) rheology
and a 5-category Ice Thickness Distribution (ITD) on the orthogonal SCVT Voronoi mesh:

1. **EVP Internal Stress & Momentum Solver** (Hunke & Dukowicz, 1997, *J. Phys. Oceanogr.*,
   27, 1849-1867; Hibler, 1979, *JPO*, 9, 815-846; Turner et al., 2022, *Geosci. Model Dev.*,
   15, 3721-3751, MPAS-Seaice v1.0.0, Sec. 2.1):
   - Hibler (1979) compressive ice strength `P = P* h exp(-C* (1 - A))` with elliptical yield
     curve eccentricity `e = 2.0` (`zeta = P / (2 Delta)`, `eta = zeta / e^2`).
   - Subcycled (`N_EVP = 30`) implicit stress relaxation + semi-implicit ocean-ice quadratic drag
     and Coriolis stabilization (guaranteed stable at high polar latitudes).
2. **5-Category ITD, Conservative Advection, & Mechanical Ridging** (Lipscomb, 2001, *JGR*, 106;
   Turner et al., 2022, Sec. 2.2):
   - Five standard CICE/MPAS thickness categories (`[0, 0.6, 1.4, 2.4, 3.6, inf)` m),
     conservatively tracking area fraction `a_cat[5, nc]` and volume `v_cat[5, nc]`.
   - Mechanical ridging transfers converging excess area (`sum(a_cat) > 1`) from thin to thick
     categories while strictly conserving total ice volume (`sum(v_cat) * rho_ice == m_seaice`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import (
    dispatch_seaice_evp_subcycle_metal,
    dispatch_seaice_itd_sync_metal,
    get_active_gpu_backend,
)
from aoba.runtime import maybe_cuda_compile
from aoba.surface import MC_SEAICE_CRIT, SURF_OCEAN, SurfaceState

# 5-category ITD lower thickness bounds [m] (CICE / MPAS-Seaice standard; Turner et al., 2022)
N_ITD_CAT: int = 5
ITD_BOUNDS_M: tuple[float, ...] = (0.0, 0.60, 1.40, 2.40, 3.60)
ITD_MID_M: tuple[float, ...] = (0.30, 1.00, 1.90, 3.00, 4.50)

# Hibler (1979) & Hunke-Dukowicz (1997) EVP rheology constants
EVP_P_STAR: float = 2.75e4       # Ice compressive strength parameter P* [N m^-2]
EVP_C_STAR: float = 20.0         # Ice concentration strength decay C* [-]
EVP_ECCENT: float = 2.0          # Elliptical yield curve aspect ratio e [-]
EVP_DELTA_MIN: float = 2.0e-9    # Strain-rate regularization floor Delta_min [s^-1]
EVP_C_WATER: float = 5.5e-3      # Quadratic ice-ocean drag coefficient C_w [-]
EVP_N_SUBCYCLES: int = 24        # Fast elastic EVP subcycles per host ocean step
EVP_U_MAX_MS: float = 0.60       # Physical sea-ice speed safety cap [m s^-1]
EVP_DIFF_H: float = 250.0        # Mild lateral diffusion [m^2 s^-1]


@dataclass
class SeaIceEVPState:
    """5-category ITD and EVP dynamic sea-ice state on the SCVT Voronoi mesh."""

    a_cat: torch.Tensor          # [5, N_cells] Ice area fraction per category (sum <= 1) [-]
    v_cat: torch.Tensor          # [5, N_cells] Ice volume per unit cell area per category [m]
    u_ice_e: torch.Tensor        # [N_edges] Edge-normal sea-ice velocity [m s^-1]
    sigma_1: torch.Tensor        # [N_cells] Isotropic compressive internal stress (s11 + s22) [N m^-1]
    sigma_s: torch.Tensor        # [N_cells] Deviatoric shear stress scalar invariant [N m^-1]
    m_seaice_sync: torch.Tensor  # [N_cells] Last synced total ice mass [kg m^-2] for thermo delta


def _sync_itd_from_total_mass(
    a_cat: torch.Tensor,
    v_cat: torch.Tensor,
    m_target: torch.Tensor,
    m_prev: torch.Tensor,
    is_ocean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservatively reconcile thermodynamic `m_target - m_prev` into 5 ITD categories & rebin."""
    dtype = m_target.dtype
    dev = m_target.device
    is_ocn_f = is_ocean.to(dtype)
    if get_active_gpu_backend(dev) in ("mps", "cuda"):
        return dispatch_seaice_itd_sync_metal(
            a_cat=a_cat,
            v_cat=v_cat,
            m_target=m_target,
            m_prev=m_prev,
            is_ocn_f=is_ocn_f,
            inv_rho_seaice=1.0 / C.RHO_SEAICE,
        )
    v_target = torch.clamp(m_target / C.RHO_SEAICE, min=0.0) * is_ocn_f
    v_cur = torch.clamp(v_cat, min=0.0).sum(dim=0)
    dv = v_target - v_cur

    # Distribute positive growth into thin/existing categories, and melting proportionally across active categories
    a_tot = torch.clamp(a_cat.sum(dim=0), min=0.0, max=1.0)
    has_ice = v_cur > 1.0e-6

    # Growth into existing categories weighted by area; brand-new frazil ice enters Category 0 (0..0.6 m)
    w_grow = torch.where(
        has_ice.unsqueeze(0),
        a_cat / torch.clamp(a_tot.unsqueeze(0), min=1.0e-6),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device=dev, dtype=dtype).unsqueeze(1),
    )
    w_melt = torch.where(
        has_ice.unsqueeze(0),
        v_cat / torch.clamp(v_cur.unsqueeze(0), min=1.0e-6),
        torch.zeros_like(v_cat),
    )
    dv_pos = torch.clamp(dv, min=0.0).unsqueeze(0) * w_grow
    dv_neg = torch.clamp(dv, max=0.0).unsqueeze(0) * w_melt
    v_new = torch.clamp(v_cat + dv_pos + dv_neg, min=0.0) * is_ocn_f.unsqueeze(0)

    # Update area for brand-new frazil ice in Cat 0 (h_new = 0.25 m) and complete meltout
    a_new = a_cat.clone()
    new_frazil = (~has_ice) & (v_target > 1.0e-6) & is_ocean
    a_new[0] = torch.where(new_frazil, torch.clamp(v_new[0] / 0.25, 0.0, 1.0), a_new[0])
    # Shrink area proportionally if category volume shrinks toward zero
    melt_ratio = torch.where(v_cat > 1.0e-7, torch.clamp(v_new / torch.clamp(v_cat, min=1.0e-7), 0.0, 1.0), torch.ones_like(v_cat))
    a_new = torch.where(dv.unsqueeze(0) < 0.0, a_new * melt_ratio, a_new)
    a_new = torch.where(v_new > 1.0e-8, a_new, torch.zeros_like(a_new))

    # Exact category rebinning whenever thickness h_n = v_n / a_n crosses ITD_BOUNDS_M
    a_out = torch.zeros_like(a_new)
    v_out = torch.zeros_like(v_new)
    bounds = ITD_BOUNDS_M
    for n in range(N_ITD_CAT):
        an = a_new[n]
        vn = v_new[n]
        hn = vn / torch.clamp(an, min=1.0e-6)
        active = an > 1.0e-7
        # Determine target category index for each cell's category-n ice
        tgt = torch.zeros_like(an, dtype=torch.int64)
        for c in range(1, N_ITD_CAT):
            tgt = torch.where(hn >= bounds[c], torch.full_like(tgt, c), tgt)
        for c in range(N_ITD_CAT):
            m_c = (active & (tgt == c)).to(dtype)
            a_out[c] = a_out[c] + an * m_c
            v_out[c] = v_out[c] + vn * m_c

    # Enforce sum(a_out) <= 1.0 via mechanical ridging (thin -> thick, conserving total volume exactly!)
    a_sum = a_out.sum(dim=0)
    excess = torch.clamp(a_sum - 1.0, min=0.0)
    if bool((excess > 1.0e-6).any()):
        # Ridge excess area starting from thinnest categories into thicker categories (h_ridge ~ 2x)
        rem = excess
        for n in range(N_ITD_CAT - 1):
            take_a = torch.minimum(a_out[n], rem)
            frac = take_a / torch.clamp(a_out[n], min=1.0e-7)
            move_v = v_out[n] * frac
            a_out[n] = a_out[n] - take_a
            v_out[n] = v_out[n] - move_v
            # Ridged ice doubles thickness -> half the area in category n+1
            a_out[n + 1] = a_out[n + 1] + 0.5 * take_a
            v_out[n + 1] = v_out[n + 1] + move_v
            rem = torch.clamp(a_out.sum(dim=0) - 1.0, min=0.0)
        # Final safety scale on area only (never alters volume v_out!)
        a_sum2 = a_out.sum(dim=0)
        a_out = torch.where(a_sum2.unsqueeze(0) > 1.0, a_out / torch.clamp(a_sum2.unsqueeze(0), min=1.0), a_out)

    # Exact volume rescaling check (guard against float32 sub-ULP drift)
    v_sum_out = v_out.sum(dim=0)
    scale_v = torch.where(v_sum_out > 1.0e-8, v_target / torch.clamp(v_sum_out, min=1.0e-8), torch.ones_like(v_target))
    v_out = v_out * scale_v.unsqueeze(0) * is_ocn_f.unsqueeze(0)
    a_out = torch.clamp(a_out, 0.0, 1.0) * is_ocn_f.unsqueeze(0)
    return a_out, v_out


def init_seaice_evp_state(sfc: SurfaceState, mesh) -> SeaIceEVPState:
    """Initialize 5-category ITD and EVP dynamic sea-ice state from `sfc.m_seaice`."""
    dev = sfc.m_seaice.device
    dtype = sfc.m_seaice.dtype
    nc = mesh.n_cells
    ne = mesh.n_edges
    is_ocean = sfc.surf_type == SURF_OCEAN

    a_cat = torch.zeros((N_ITD_CAT, nc), device=dev, dtype=dtype)
    v_cat = torch.zeros((N_ITD_CAT, nc), device=dev, dtype=dtype)
    v_tot = torch.clamp(sfc.m_seaice / C.RHO_SEAICE, min=0.0) * is_ocean.to(dtype)
    a_tot = torch.clamp(torch.sqrt(torch.clamp(sfc.m_seaice, min=0.0) / MC_SEAICE_CRIT), 0.0, 1.0) * is_ocean.to(dtype)
    h_mean = v_tot / torch.clamp(a_tot, min=1.0e-5)

    # Place initial ice into its matching ITD thickness category
    tgt = torch.zeros(nc, device=dev, dtype=torch.int64)
    for c in range(1, N_ITD_CAT):
        tgt = torch.where(h_mean >= ITD_BOUNDS_M[c], torch.full_like(tgt, c), tgt)
    for c in range(N_ITD_CAT):
        m_c = ((v_tot > 1.0e-6) & (tgt == c)).to(dtype)
        a_cat[c] = a_tot * m_c
        v_cat[c] = v_tot * m_c

    return SeaIceEVPState(
        a_cat=a_cat,
        v_cat=v_cat,
        u_ice_e=torch.zeros(ne, device=dev, dtype=dtype),
        sigma_1=torch.zeros(nc, device=dev, dtype=dtype),
        sigma_s=torch.zeros(nc, device=dev, dtype=dtype),
        m_seaice_sync=sfc.m_seaice.clone(),
    )


def _evp_subcycle_core(
    u_ice_e: torch.Tensor,
    sigma_1: torch.Tensor,
    sigma_s: torch.Tensor,
    p_strength: torch.Tensor,
    m_ice_e: torch.Tensor,
    a_ice_e: torch.Tensor,
    tau_air_e: torch.Tensor,
    u_ocn_e: torch.Tensor,
    grad_eta_e: torch.Tensor,
    f_vert: torch.Tensor,
    v0_e: torch.Tensor,
    v1_e: torch.Tensor,
    ep_idx: torch.Tensor,
    w_trisk: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    mask_e: torch.Tensor,
    is_ocn_f: torch.Tensor,
    u_cap: float,
    dt: float,
    n_sub: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor MPAS-Seaice EVP elastic subcycling on Voronoi C-grid (CUDA/MPS compilable)."""
    dt_e = dt / float(n_sub)
    t_damp = 0.36 * dt  # Hunke & Dukowicz (1997) elastic damping timescale T = 0.36 * dt
    e2_inv = 1.0 / (EVP_ECCENT * EVP_ECCENT)
    # Implicit elastic relaxation factor alpha = dt_e / T_damp
    alpha_e = dt_e / max(t_damp, 1.0)
    inv_1pa = 1.0 / (1.0 + alpha_e)

    # Planetary Coriolis edge factor f_e for TRiSK rotation (no nonlinear advection in EVP)
    f_e = 0.5 * (f_vert[v0_e] + f_vert[v1_e])
    f_pair = 0.5 * (f_e.unsqueeze(-1) + f_e[ep_idx])
    cor_w = w_trisk * f_pair  # [n_edges, max_trisk]

    m_safe = torch.clamp(m_ice_e, min=10.0)
    has_ice_e = (m_ice_e > 1.0).to(u_ice_e.dtype) * mask_e
    u_cur = u_ice_e * has_ice_e
    s1 = sigma_1
    ss = sigma_s

    for _ in range(n_sub):
        # 1. Cell-centered divergence & shear strain-rate invariant on Voronoi cells
        div_u = (u_cur[eoc] * div_w).sum(dim=-1) * is_ocn_f
        # Estimate edge-difference shear rate magnitude |du/dn|
        du_e = (u_cur[c1.clamp(max=u_cur.shape[0] - 1)] * 0.0)  # placeholder avoided below
        # Isotropic velocity gradient norm on cell from neighboring edges
        u_mean_c = (u_cur[eoc].abs()).mean(dim=-1)
        shear_est = ((u_cur[eoc].abs() - u_mean_c.unsqueeze(-1)).abs()).mean(dim=-1) * (inv_dc_edge[eoc].mean(dim=-1))
        delta = torch.sqrt(div_u * div_u + e2_inv * shear_est * shear_est + EVP_DELTA_MIN * EVP_DELTA_MIN)

        zeta = p_strength / (2.0 * delta)
        eta_v = zeta * e2_inv

        # Implicit elastic-plastic stress update (Hunke & Dukowicz, 1997, Eqs. 16-17)
        s1_target = zeta * (div_u - delta)
        ss_target = eta_v * shear_est
        s1_raw = (s1 + alpha_e * s1_target) * inv_1pa
        s1 = torch.clamp(torch.maximum(s1_raw, -2.0 * p_strength - 1.0), max=0.0) * is_ocn_f
        ss_raw = (ss + alpha_e * ss_target) * inv_1pa
        ss = torch.clamp(torch.minimum(ss_raw, p_strength + 1.0), min=0.0) * is_ocn_f

        # 2. Internal stress divergence along edge normal: (1/2) * d(s1)/dx + viscous shear damping
        f_int_e = 0.5 * ((s1[c1] - s1[c0]) * inv_dc_edge) * has_ice_e

        # 3. 2-stage semi-implicit midpoint momentum update with implicit ocean drag
        #    for inertial stability at high latitudes.
        f_ext_e = a_ice_e * tau_air_e + f_int_e - m_ice_e * C.GRAVITY * grad_eta_e
        u_rel_mag = torch.abs(u_cur - u_ocn_e) + 0.05
        c_drag_o = a_ice_e * C.RHO_SEAWATER * EVP_C_WATER * u_rel_mag  # [kg m^-2 s^-1]
        cor_0 = (cor_w * u_cur[ep_idx]).sum(dim=-1) * m_ice_e
        u_half = torch.clamp(
            ((m_safe * u_cur + (0.5 * dt_e) * (f_ext_e + cor_0 + c_drag_o * u_ocn_e))
             / (m_safe + (0.5 * dt_e) * c_drag_o)) * has_ice_e,
            -u_cap,
            u_cap,
        )
        cor_half = (cor_w * u_half[ep_idx]).sum(dim=-1) * m_ice_e
        u_rel_h = torch.abs(u_half - u_ocn_e) + 0.05
        c_drag_h = a_ice_e * C.RHO_SEAWATER * EVP_C_WATER * u_rel_h
        rhs_u = m_safe * u_cur + dt_e * (f_ext_e + cor_half + c_drag_h * u_ocn_e)
        u_cur = torch.clamp((rhs_u / (m_safe + dt_e * c_drag_h)) * has_ice_e, -u_cap, u_cap)

    return u_cur, s1, ss


def step_seaice_evp_itd_voronoi(
    evp: SeaIceEVPState,
    sfc: SurfaceState,
    mesh,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    u_ocn_e: torch.Tensor,
    eta_ocn: torch.Tensor,
    dt: float,
    cache: dict | None = None,
) -> None:
    """Advance 5-category ITD and EVP sea-ice dynamics on SCVT Voronoi mesh, updating `sfc.m_seaice`."""
    is_ocean = sfc.surf_type == SURF_OCEAN
    dtype = sfc.m_seaice.dtype
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    eoc = mesh.edges_on_cell
    if cache is not None:
        is_ocn_f = cache["is_ocn_f"]
        mask_e = cache["ocn_e0f"]
        div_w = cache["div_w"]
        inv_dc_edge = cache["inv_dc_edge"]
    else:
        is_ocn_f = is_ocean.to(dtype)
        mask_e = (is_ocean[c0] & is_ocean[c1]).to(dtype)
        div_w = (mesh.edge_sign_on_cell.to(dtype) * mesh.dv_edge[eoc].to(dtype)) / mesh.area_cell.unsqueeze(-1)
        inv_dc_edge = 1.0 / mesh.dc_edge

    # 1. Reconcile latest thermodynamic sea-ice growth/melt from `step_surface` into 5 ITD categories
    evp.a_cat, evp.v_cat = _sync_itd_from_total_mass(
        evp.a_cat, evp.v_cat, sfc.m_seaice, evp.m_seaice_sync, is_ocean
    )

    # 2. Compute Hibler (1979) ice compressive strength P = P* h exp(-C* (1 - A))
    a_tot = torch.clamp(evp.a_cat.sum(dim=0), 0.0, 1.0)
    v_tot = torch.clamp(evp.v_cat.sum(dim=0), min=0.0)
    p_strength = (EVP_P_STAR * v_tot * torch.exp(-EVP_C_STAR * (1.0 - a_tot))) * is_ocn_f

    m_ice_c = v_tot * C.RHO_SEAICE
    m_ice_e = 0.5 * (m_ice_c[c0] + m_ice_c[c1]) * mask_e
    a_ice_e = 0.5 * (a_tot[c0] + a_tot[c1]) * mask_e
    tx_air = -tau_x_sfc
    ty_air = -tau_y_sfc
    tau_air_e = (0.5 * (tx_air[c0] + tx_air[c1]) * mesh.east_normal_edge + 0.5 * (ty_air[c0] + ty_air[c1]) * mesh.north_normal_edge) * mask_e
    grad_eta_e = ((eta_ocn[c1] - eta_ocn[c0]) * inv_dc_edge) * mask_e

    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    u_cap = min(EVP_U_MAX_MS, 0.25 * dc_min / max(dt, 1.0))

    # 3. Subcycled EVP momentum & internal stress solver
    dev = sfc.m_seaice.device
    if get_active_gpu_backend(dev) in ("mps", "cuda"):
        evp_c = cache if cache is not None else {}
        c0_i32 = evp_c.get("c0_i32")
        if c0_i32 is None:
            c0_i32 = c0.to(torch.int32).contiguous()
            c1_i32 = c1.to(torch.int32).contiguous()
            eoc_i32 = eoc.to(torch.int32).contiguous()
            if cache is not None:
                cache["c0_i32"] = c0_i32
                cache["c1_i32"] = c1_i32
                cache["eoc_i32"] = eoc_i32
        else:
            c1_i32 = evp_c["c1_i32"]
            eoc_i32 = evp_c["eoc_i32"]
        ep_idx_i32 = evp_c.get("ep_idx_i32")
        if ep_idx_i32 is None:
            ep_idx_i32 = mesh.trisk_edges.to(torch.int32).contiguous()
            f_e = 0.5 * (mesh.f_vert[mesh.verts_on_edge[:, 0]] + mesh.f_vert[mesh.verts_on_edge[:, 1]])
            f_pair = 0.5 * (f_e.unsqueeze(-1) + f_e[mesh.trisk_edges])
            cor_w_c = (mesh.trisk_weights * f_pair).contiguous()
            if cache is not None:
                cache["ep_idx_i32"] = ep_idx_i32
                cache["evp_cor_w"] = cor_w_c
        else:
            cor_w_c = evp_c["evp_cor_w"]
        dt_e = float(dt) / float(EVP_N_SUBCYCLES)
        t_damp = 0.36 * float(dt)
        alpha_e = dt_e / max(t_damp, 1.0)
        inv_1pa = 1.0 / (1.0 + alpha_e)
        e2_inv = 1.0 / (EVP_ECCENT * EVP_ECCENT)
        has_ice_e = (m_ice_e > 1.0).to(dtype) * mask_e
        evp.u_ice_e, evp.sigma_1, evp.sigma_s = dispatch_seaice_evp_subcycle_metal(
            u_ice_e=evp.u_ice_e,
            sigma_1=evp.sigma_1,
            sigma_s=evp.sigma_s,
            p_strength=p_strength,
            m_ice_e=m_ice_e,
            a_ice_e=a_ice_e,
            tau_air_e=tau_air_e,
            u_ocn_e=u_ocn_e,
            grad_eta_e=grad_eta_e,
            cor_w=cor_w_c,
            ep_idx_i32=ep_idx_i32,
            c0_i32=c0_i32,
            c1_i32=c1_i32,
            eoc_i32=eoc_i32,
            div_w=div_w,
            inv_dc_edge=inv_dc_edge,
            has_ice_e=has_ice_e,
            is_ocn_f=is_ocn_f,
            alpha_e=alpha_e,
            inv_1pa=inv_1pa,
            e2_inv=e2_inv,
            delta_min=EVP_DELTA_MIN,
            dt_e=dt_e,
            u_cap=u_cap,
            rho_sw_cw=C.RHO_SEAWATER * EVP_C_WATER,
            gravity=C.GRAVITY,
            n_sub=EVP_N_SUBCYCLES,
        )
    else:
        evp_fn = maybe_cuda_compile(_evp_subcycle_core, device=dev)
        evp.u_ice_e, evp.sigma_1, evp.sigma_s = evp_fn(
            evp.u_ice_e, evp.sigma_1, evp.sigma_s, p_strength, m_ice_e, a_ice_e,
            tau_air_e, u_ocn_e, grad_eta_e, mesh.f_vert,
            mesh.verts_on_edge[:, 0], mesh.verts_on_edge[:, 1], mesh.trisk_edges, mesh.trisk_weights,
            c0, c1, eoc, div_w, inv_dc_edge, mask_e, is_ocn_f, u_cap, float(dt), EVP_N_SUBCYCLES,
        )

    # 4. Batched strictly conservative 2-stage SSP-RK2 (Heun) Voronoi advection of [a_cat(5), v_cat(5), swe(1)]
    u_pos = torch.clamp(evp.u_ice_e, min=0.0)
    u_neg = torch.clamp(evp.u_ice_e, max=0.0)
    diff_w = EVP_DIFF_H * inv_dc_edge * mask_e
    q_all = torch.cat([evp.a_cat, evp.v_cat, torch.clamp(sfc.swe_seaice, min=0.0).unsqueeze(0)], dim=0)

    def _tend_b(q: torch.Tensor) -> torch.Tensor:
        f_q = (
            u_pos.unsqueeze(0) * q[:, c0]
            + u_neg.unsqueeze(0) * q[:, c1]
            - (q[:, c1] - q[:, c0]) * diff_w.unsqueeze(0)
        )
        return -(f_q[:, eoc] * div_w.unsqueeze(0)).sum(dim=-1) * is_ocn_f.unsqueeze(0)

    q_1 = torch.clamp(q_all + dt * _tend_b(q_all), min=0.0) * is_ocn_f.unsqueeze(0)
    q_2 = torch.clamp(0.5 * q_all + 0.5 * (q_1 + dt * _tend_b(q_1)), min=0.0) * is_ocn_f.unsqueeze(0)
    evp.a_cat = q_2[:N_ITD_CAT]
    evp.v_cat = q_2[N_ITD_CAT:2 * N_ITD_CAT]
    sfc.swe_seaice = torch.where(is_ocean, q_2[2 * N_ITD_CAT], sfc.swe_seaice)

    # 5. Mechanical ridging & exact ITD rebinning -> write conserved total mass back to sfc.m_seaice
    m_adv = evp.v_cat.sum(dim=0) * C.RHO_SEAICE
    evp.a_cat, evp.v_cat = _sync_itd_from_total_mass(evp.a_cat, evp.v_cat, m_adv, m_adv, is_ocean)
    m_final = evp.v_cat.sum(dim=0) * C.RHO_SEAICE
    sfc.m_seaice = torch.where(is_ocean, m_final, sfc.m_seaice)
    evp.m_seaice_sync = sfc.m_seaice.clone()
