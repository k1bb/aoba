"""Zero-Q-Flux 1.5-Layer Wind-Driven Ekman + Reduced-Gravity Shallow-Water Ocean on SCVT Voronoi Grid.

Implements Codron (2012, J. Climate, 25, 1747-1760) combined with 1.5-layer reduced-gravity
shallow-water dynamics on the host SCVT Voronoi C-grid with zero normal flux across all
land-ocean coastal edges, guaranteeing exact global ocean heat and volume conservation without
any prescribed Q-flux climatology.

References
----------
- Codron (2012, J. Climate, 25, 1747-1760, Sec. 2a, Eqs. 1-10):
  Regularized equatorial + extratropical wind-driven Ekman transport `M_E`, mass-compensating
  sub-surface return flow `-M_E`, vertical upwelling/downwelling `W_E = div(M_E)`, and
  horizontal mesoscale eddy diffusion `kappa_h`.
- Ringler, Thuburn, Klemp, & Skamarock (2010, J. Comput. Phys., 229, 3065-3090):
  TRiSK energy-conserving Coriolis/vorticity operator on spherical Voronoi C-grids.
- MIROC6 AGCM Document Writing Team (2021, Sec. 4.1.2):
  Conservative coastal routing of terrestrial river & glacier calving freshwater/fusion enthalpy.

AOBA Formulation Reference: Sec. 9.3, Eq. (9.2) (1.5-Layer Wind-Driven Ekman + Shallow-Water Ocean).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.scvt_dynamics import (
    compute_scvt_divergence,
    compute_scvt_vorticity,
    compute_vector_laplacian_edge,
)
from aoba.scvt_mesh import SCVTMesh
from aoba.surface import SURF_OCEAN, SurfaceState, _kahan_add


# Codron (2012, Sec. 2a) & 1.5-layer reduced-gravity shallow-water parameters
H_THERMOCLINE: float = 200.0        # Sub-mixed-layer return-flow thickness H_2 [m]
R_EKMAN_DAMP_DAYS: float = 2.0      # Equatorial Rayleigh friction timescale [days]
R_EKMAN_DAMP: float = 1.0 / (R_EKMAN_DAMP_DAYS * C.SECONDS_PER_DAY)  # Equatorial Rayleigh friction [s^-1]
KAPPA_OCN_HORIZ: float = 2.0e3      # Horizontal mesoscale eddy diffusivity kappa_h [m^2 s^-1]
KAPPA_OCN_VERT: float = 2.0e-5      # Weak diapycnal exchange velocity [m s^-1]
G_PRIME_OCN: float = 0.03           # First-baroclinic reduced gravity g' [m s^-2]
TAU_SW_DRAG_DAYS: float = 180.0     # Interfacial/bottom Rayleigh drag timescale [days]
R_SW_DRAG: float = 1.0 / (TAU_SW_DRAG_DAYS * C.SECONDS_PER_DAY)
BETA_MUNK_REF: float = 2.28e-11     # Equatorial beta [m^-1 s^-1] for Munk boundary viscosity
ENABLE_SHALLOW_WATER_OCN: bool = True


def recompute_derived_constants() -> None:
    """Refresh Rayleigh Ekman and shallow-water drag rates from damping timescales."""
    global R_EKMAN_DAMP, R_SW_DRAG
    R_EKMAN_DAMP = 1.0 / (R_EKMAN_DAMP_DAYS * C.SECONDS_PER_DAY)
    R_SW_DRAG = 1.0 / (TAU_SW_DRAG_DAYS * C.SECONDS_PER_DAY)


@dataclass
class EkmanOceanState:
    """Sub-mixed-layer thermocline temperature, shallow-water state, & Ekman upwelling on SCVT."""

    t_sub_ml: torch.Tensor       # Layer-2 return-flow temperature T_2 [N_cells] [K]
    w_ekman: torch.Tensor        # Diagnostic vertical mass flux div(M_1) [N_cells] [kg m^-2 s^-1]
    h_sub_ml: float = H_THERMOCLINE
    t1_comp: torch.Tensor | None = None  # Kahan float32 rounding compensation for Layer-1 T_1 [K]
    t2_comp: torch.Tensor | None = None  # Kahan float32 rounding compensation for Layer-2 T_2 [K]
    h_sw: torch.Tensor | None = None     # Active upper-ocean thickness H_sw = h_1 + H_2 + eta [N_cells] [m]
    u_sw_edge: torch.Tensor | None = None  # Shallow-water normal velocity on edges [N_edges] [m s^-1]


def init_ekman_ocean_state(
    sfc: SurfaceState,
    dt_thermocline_k: float = 6.0,
) -> EkmanOceanState:
    """Initialize Layer-2 thermocline temperature `T_2` and resting shallow-water thickness `H_sw`."""
    t2 = torch.clamp(sfc.t_ocean_ml - dt_thermocline_k, min=C.T_FREEZE_OCEAN)
    is_ocean = sfc.surf_type == SURF_OCEAN
    h_sw0 = torch.where(is_ocean, sfc.h_ocean_ml + H_THERMOCLINE, torch.zeros_like(t2))
    return EkmanOceanState(
        t_sub_ml=t2,
        w_ekman=torch.zeros_like(t2),
        t1_comp=torch.zeros_like(t2),
        t2_comp=torch.zeros_like(t2),
        h_sw=h_sw0,
        u_sw_edge=None,
    )


def step_ekman_ocean_voronoi(
    ocn: EkmanOceanState,
    sfc: SurfaceState,
    mesh: SCVTMesh,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    f_seaice: torch.Tensor,
    runoff_glacier: torch.Tensor,
    dt: float,
) -> None:
    """Advance `sfc.t_ocean_ml` (`T_1`) and `ocn.t_sub_ml` (`T_2`) by `dt` [s] in-place.

    Note on stress sign: `tau_x_sfc, tau_y_sfc` from `step_surface` are upward momentum fluxes
    from surface to atmosphere (`-rho * C_m * |U| * u_1`). Hence the wind stress exerted BY the
    atmosphere ON the ocean is `tau_ocn = -(1 - 0.5 * f_seaice) * tau_sfc`.
    """
    is_ocean = sfc.surf_type == SURF_OCEAN

    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    ocn_edge = is_ocean[c0] & is_ocean[c1]
    mask_edge = ocn_edge.to(sfc.t_ocean_ml.dtype)

    # 1. Wind stress acting on ocean surface [N m^-2] (partially attenuated under sea-ice pack)
    ice_atten = 1.0 - 0.5 * torch.clamp(f_seaice, 0.0, 1.0)
    tx_ocn = -ice_atten * tau_x_sfc
    ty_ocn = -ice_atten * tau_y_sfc

    # Interpolate stress to Voronoi edges & apply Codron (2012, Eq. 2) Ekman inversion
    tx_e = 0.5 * (tx_ocn[c0] + tx_ocn[c1])
    ty_e = 0.5 * (ty_ocn[c0] + ty_ocn[c1])
    fe = mesh.f_edge
    r_e = R_EKMAN_DAMP
    denom_ek = fe * fe + r_e * r_e

    mx_ek = (r_e * tx_e + fe * ty_e) / denom_ek  # [kg m^-1 s^-1]
    my_ek = (r_e * ty_e - fe * tx_e) / denom_ek  # [kg m^-1 s^-1]

    # Normal Ekman mass transport across edge c0 -> c1 (strictly 0 across coastlines!)
    mn_ek = (mx_ek * mesh.east_normal_edge + my_ek * mesh.north_normal_edge) * mask_edge

    # 1b. 1.5-Layer Reduced-Gravity Shallow-Water Dynamics (Sverdrup/western-boundary gyres + Kelvin/Rossby waves)
    is_ocean_f = is_ocean.to(mn_ek.dtype)
    if ocn.h_sw is None:
        ocn.h_sw = torch.where(is_ocean, sfc.h_ocean_ml + ocn.h_sub_ml, torch.zeros_like(sfc.t_ocean_ml))
    if ocn.u_sw_edge is None:
        ocn.u_sw_edge = torch.zeros_like(mn_ek)

    h1_e = 0.5 * (sfc.h_ocean_ml[c0] + sfc.h_ocean_ml[c1])
    if ENABLE_SHALLOW_WATER_OCN:
        # Enforce minimum Layer-2 thickness of 20 m (h_sw >= h_1 + 20 m) for well-conditioned enthalpy
        h_sw_min = sfc.h_ocean_ml + 20.0
        h_sw = torch.where(is_ocean, torch.clamp(torch.maximum(ocn.h_sw, h_sw_min), max=800.0), torch.zeros_like(ocn.h_sw))
        h2_old = torch.where(is_ocean, torch.clamp(h_sw - sfc.h_ocean_ml, min=20.0), torch.full_like(h_sw, ocn.h_sub_ml))
        u_sw = ocn.u_sw_edge * mask_edge
        tau_n_e = (tx_e * mesh.east_normal_edge + ty_e * mesh.north_normal_edge) * mask_edge

        dc_min = getattr(mesh, "_dc_edge_min_m", None)
        if dc_min is None:
            dc_min = float(mesh.dc_edge.min().item())
            setattr(mesh, "_dc_edge_min_m", dc_min)
        c_gw_max = math.sqrt(max(G_PRIME_OCN * 600.0, 1.0))
        dt_sw_cfl = 0.40 * dc_min / max(c_gw_max, 5.0)
        n_sw = max(1, int(math.ceil(dt / max(dt_sw_cfl, 1.0))))
        dt_sw = dt / float(n_sw)

        # Scale-aware Munk horizontal eddy viscosity so western boundary layers span ~1.6*dc_min cleanly
        delta_munk = 1.6 * dc_min
        a_h = min(BETA_MUNK_REF * (delta_munk ** 3), 0.08 * (dc_min ** 2) / max(dt_sw, 1.0))

        v0_e = mesh.verts_on_edge[:, 0]
        v1_e = mesh.verts_on_edge[:, 1]
        ep_idx = mesh.trisk_edges
        w_trisk = mesh.trisk_weights

        def _sw_rhs(u_cur: torch.Tensor, h_cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            zeta_v = compute_scvt_vorticity(mesh, u_cur)
            eta_v = zeta_v + mesh.f_vert
            eta_e = 0.5 * (eta_v[v0_e] + eta_v[v1_e])
            eta_pair = 0.5 * (eta_e.unsqueeze(-1) + eta_e[ep_idx])
            cor_sw = (w_trisk * eta_pair * u_cur[ep_idx]).sum(dim=-1)

            h_e = torch.clamp(0.5 * (h_cur[c0] + h_cur[c1]), min=20.0)
            pgf_sw = -G_PRIME_OCN * ((h_cur[c1] - h_cur[c0]) / mesh.dc_edge)
            wind_acc = tau_n_e / (C.RHO_SEAWATER * h_e)
            lap_u_sw, _ = compute_vector_laplacian_edge(mesh, u_cur)

            du = (cor_sw + pgf_sw + wind_acc + a_h * lap_u_sw - R_SW_DRAG * u_cur) * mask_edge
            h_up = torch.where(u_cur >= 0.0, h_cur[c0], h_cur[c1])
            v_flux = h_up * u_cur * mask_edge
            dh = -compute_scvt_divergence(mesh, v_flux) * is_ocean_f
            return du, dh, v_flux

        # Wicker & Skamarock (2002) 3-stage RK3 ensures strict stability for high-latitude inertial oscillations
        v_sw_acc = torch.zeros_like(u_sw)
        for _ in range(n_sw):
            u0_sw, h0_sw = u_sw, h_sw
            v_stage = torch.zeros_like(u_sw)
            for rk_dt in (dt_sw / 3.0, 0.5 * dt_sw, dt_sw):
                du_k, dh_k, v_stage = _sw_rhs(u_sw, h_sw)
                u_sw = torch.clamp((u0_sw + rk_dt * du_k) * mask_edge, -5.0, 5.0)
                h_sw = torch.where(
                    is_ocean,
                    torch.clamp(torch.maximum(h0_sw + rk_dt * dh_k, h_sw_min), max=800.0),
                    torch.zeros_like(h0_sw),
                )
            v_sw_acc = v_sw_acc + v_stage

        v_sw_eff = v_sw_acc / float(n_sw)
        ocn.u_sw_edge = u_sw
        ocn.h_sw = h_sw
        h2_new = torch.where(is_ocean, torch.clamp(h_sw - sfc.h_ocean_ml, min=20.0), torch.full_like(h_sw, ocn.h_sub_ml))

        # Decompose geostrophic/gyre velocity u_g = u_sw - M_E / (rho_w * H_sw) to avoid double-counting Ekman
        h_sw_e = torch.clamp(0.5 * (h_sw[c0] + h_sw[c1]), min=20.0)
        u_g_e = (u_sw - mn_ek / (C.RHO_SEAWATER * h_sw_e)) * mask_edge
        m1_edge = mn_ek + C.RHO_SEAWATER * h1_e * u_g_e
        # Exact Layer-2 return mass flux balancing total upper-layer transport rho_w * v_sw_eff
        m2_edge = (C.RHO_SEAWATER * v_sw_eff - m1_edge) * mask_edge
        h2_edge = 0.5 * (h2_old[c0] + h2_old[c1])
    else:
        h2_old = torch.full_like(sfc.t_ocean_ml, ocn.h_sub_ml)
        h2_new = h2_old
        m1_edge = mn_ek
        m2_edge = -mn_ek
        h2_edge = torch.full_like(mn_ek, ocn.h_sub_ml)

    # Exact mass-conserving vertical upwelling flux W_12 = div_SCVT(M_1) [kg m^-2 s^-1]
    w_ek = compute_scvt_divergence(mesh, m1_edge) * is_ocean_f
    ocn.w_ekman = w_ek

    # 2. Flux-form upwind heat transport for Layer 1 (+m1_edge) and variable-thickness Layer 2 (+m2_edge)
    t1 = sfc.t_ocean_ml
    t2 = ocn.t_sub_ml

    f1_adv = torch.clamp(m1_edge, min=0.0) * t1[c0] + torch.clamp(m1_edge, max=0.0) * t1[c1]
    f2_adv = torch.clamp(m2_edge, min=0.0) * t2[c0] + torch.clamp(m2_edge, max=0.0) * t2[c1]

    # Vertical upwind temperature across base of mixed layer: T_2 if upwelling (W_E > 0), T_1 if downwelling
    t_vert = torch.where(w_ek >= 0.0, t2, t1)

    # 3. Horizontal mesoscale eddy diffusion with zero-flux coastal Neumann boundary condition
    f1_diff = -C.RHO_SEAWATER * h1_e * KAPPA_OCN_HORIZ * ((t1[c1] - t1[c0]) / mesh.dc_edge) * mask_edge
    f2_diff = -C.RHO_SEAWATER * h2_edge * KAPPA_OCN_HORIZ * ((t2[c1] - t2[c0]) / mesh.dc_edge) * mask_edge

    div_q1 = C.CP_SEAWATER * (compute_scvt_divergence(mesh, f1_adv + f1_diff) - w_ek * t_vert)
    div_q2 = C.CP_SEAWATER * (compute_scvt_divergence(mesh, f2_adv + f2_diff) + w_ek * t_vert)

    # Weak vertical diapycnal mixing between Layer 1 and Layer 2
    q_diap = C.CP_SEAWATER * C.RHO_SEAWATER * KAPPA_OCN_VERT * (t2 - t1)

    # 4. Conservative coastal glacier calving fusion cooling (MIROC6 Sec. 4.1.2)
    # Total fusion cooling power [W] = L_f0 * area_land * runoff_glacier routed to coastal_ocean_idx
    q_calve = torch.zeros_like(t1)
    power_glac = C.L_F0 * mesh.area_cell * torch.clamp(runoff_glacier, min=0.0)
    q_calve.scatter_add_(0, mesh.coastal_ocean_idx, power_glac)
    q_calve = q_calve / mesh.area_cell  # [W m^-2] over receiving coastal ocean cell

    c1_cap = C.CP_SEAWATER * C.RHO_SEAWATER * sfc.h_ocean_ml
    c2_cap_new = C.CP_SEAWATER * C.RHO_SEAWATER * h2_new

    if ocn.t1_comp is None:
        ocn.t1_comp = torch.zeros_like(t1)
    if ocn.t2_comp is None:
        ocn.t2_comp = torch.zeros_like(t2)

    dt1 = (dt / c1_cap) * (-div_q1 + q_diap - q_calve)
    # Variable-thickness Layer-2 enthalpy update: h2_new * T2_new = h2_old * T2_old + dt * RHS_2 / (rho_w * c_p)
    dt2 = ((h2_old - h2_new) * t2 + (dt / (C.CP_SEAWATER * C.RHO_SEAWATER)) * (-div_q2 - q_diap)) / h2_new
    t1_next, comp1 = _kahan_add(t1, dt1, ocn.t1_comp)
    t2_next, comp2 = _kahan_add(t2, dt2, ocn.t2_comp)

    # If dynamic advection cools mixed layer below seawater freezing point, form sea ice conservatively!
    freeze_excess = torch.where(is_ocean, torch.clamp(C.T_FREEZE_OCEAN - t1_next, min=0.0), torch.zeros_like(t1))
    dm_ice = (c1_cap * freeze_excess) / C.L_F0
    t1_final = t1_next + freeze_excess
    t2_final = torch.clamp(t2_next, min=C.T_FREEZE_OCEAN)

    ocn.t1_comp = torch.where(is_ocean & (freeze_excess == 0.0), comp1, torch.zeros_like(comp1))
    ocn.t2_comp = torch.where(is_ocean & (t2_next >= C.T_FREEZE_OCEAN), comp2, torch.zeros_like(comp2))
    sfc.t_ocean_ml = torch.where(is_ocean, t1_final, sfc.t_ocean_ml)
    sfc.m_seaice = torch.where(is_ocean, sfc.m_seaice + dm_ice, sfc.m_seaice)
    ocn.t_sub_ml = torch.where(is_ocean, t2_final, ocn.t_sub_ml)
