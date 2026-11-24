"""Map-free, flow-dependent non-orographic gravity wave drag (convective + frontal/jet sources).

Requires **zero prescribed geographic wave-source maps**. Both gravity-wave sources are diagnosed
autonomously from the instantaneous atmospheric state and embedded 1:1 CRM diabatic heating:
1. **Frontal / Jet-Exit Source** (Hoskins, 1982; Charron & Manzini, 2002; Richter et al., 2010):
   Evaluates the 2D horizontal frontogenesis function ``F = 0.5 * D/Dt |grad_h theta|^2`` on the
   SCVT Voronoi mesh near 600 hPa; cells exceeding ``NGWD_FRONT_THRESH`` launch a Gaussian phase-
   speed spectrum aligned with the steering wind.
2. **Convective Source** (Beres, Alexander, & Holton, 2004; Beres et al., 2005; Richter et al., 2010):
   Extracts convective heating depth ``h_d``, peak heating rate ``Q_0``, launch top, and heating-
   layer mean wind ``(u_h, v_h)`` directly from the 1:1 CRM latent-heating profile ``q_conv(z)``,
   launching an intrinsic double-peaked phase-speed spectrum (``sigma_c = N_h * h_d / pi``,
   flux proportional to ``Q_0^2``) along orthogonal zonal and meridional axes.
3. **Upward Spectral Propagation & Lindzen Saturation** (Lindzen, 1981; Alexander & Dunkerton, 1999;
   Richter et al., 2010, Sec. 2c, Eqs. 4-10):
   Propagates discrete phase-speed bins ``c_m in [-c_max, +c_max]`` upward with critical-level
   absorption and Lindzen saturation ``tau_sat = eff * rho * k_h * |c_m - u|^3 / (2 N)``, using an
   **open-top radiation boundary** (unbroken flux exits the model top cleanly, ensuring stability for
   both low-top 24 km and high-top 48-80 km stratospheric/mesospheric configurations).

References
----------
- Lindzen (1981, J. Geophys. Res., 86, 9707-9714): Gravity-wave saturation & breaking.
- Hoskins (1982, Annu. Rev. Fluid Mech., 14, 131-151): 2D kinematic frontogenesis function.
- Alexander & Dunkerton (1999, J. Atmos. Sci., 56, 4167-4181): Spectral gravity-wave propagation.
- Charron & Manzini (2002, J. Atmos. Sci., 59, 923-941, Eqs. 1-7): Frontal GWD parameterization.
- Beres, Alexander, & Holton (2004, J. Atmos. Sci., 61, 324-337, Eqs. 6-12) &
  Beres et al. (2005, J. Geophys. Res., 110, D10108): Convective GWD source spectrum.
- Richter, Sassi, & Garcia (2010, J. Atmos. Sci., 67, 136-156, Sec. 2, Eqs. 1-10):
  Unified flow-dependent frontal + convective non-orographic GWD in WACCM.

AOBA Formulation Reference: Sec. 8.3, Eqs. (8.6)-(8.7) (Map-Free Frontal & Convective Non-Orographic GWD).
"""

from __future__ import annotations

import math
import torch

from aoba import constants as C
from aoba.kernels import can_use_crm_metal, dispatch_ngwd_propagate_metal
from aoba.scvt_mesh import SCVTMesh


# Flow-dependent non-orographic GWD global parameters (synchronized via `[gravity_wave]` in TOML)
NGWD_FRONT_THRESH: float = 1.25e-15    # 0.045 K^2 (100 km)^-2 h^-1 [K^2 m^-2 s^-1] (Richter et al., 2010)
NGWD_FRONT_LAUNCH_HPA: float = 600.0   # Reference steering pressure for frontogenesis [hPa]
NGWD_FRONT_TAU0: float = 1.5e-3        # Baseline frontal launch stress tau_bgnd [Pa] (Richter et al., 2010)
NGWD_FRONT_CW: float = 30.0            # Frontal Gaussian phase-speed width c_w [m s^-1]
NGWD_FRONT_EFF: float = 0.10           # Frontal intermittency efficiency effgw_front [-]

NGWD_CONV_TAU0: float = 2.0e-3         # Reference convective peak launch stress [Pa] (Beres et al., 2005)
NGWD_CONV_QREF: float = 2.0e-4         # Reference peak convective heating rate (~17.3 K d^-1) [K s^-1]
NGWD_CONV_EFF: float = 0.25            # Convective intermittency efficiency effgw_beres [-]

NGWD_WAVELENGTH_M: float = 2.0e5       # Representative horizontal wavelength lambda_h [m] (200 km)
NGWD_CMAX: float = 80.0                # Maximum discrete phase speed |c_max| [m s^-1]
NGWD_TEND_MAX: float = 1.157e-3        # Max wind tendency cap (100 m s^-1 d^-1) [m s^-2]


def compute_scvt_horizontal_gradient(
    mesh: SCVTMesh,
    field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete Gauss-Green horizontal gradient ``(df/dx, df/dy)`` on SCVT cells ``[..., N_cells]``.

    Uses ``(f_edge - f_cell)`` so any constant field has identically zero gradient on the sphere.
    """
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    f_edge = 0.5 * (field[..., c0] + field[..., c1])  # [..., N_edges]

    e_idx = mesh.edges_on_cell          # [N_cells, 6]
    grad_w = getattr(mesh, "_grad_w_xy", None)
    if grad_w is None:
        w_base = mesh.edge_sign_on_cell * mesh.dv_edge[e_idx]
        inv_a = 1.0 / torch.clamp(mesh.area_cell, min=1.0).unsqueeze(-1)
        wx = w_base * mesh.east_normal_edge[e_idx] * inv_a
        wy = w_base * mesh.north_normal_edge[e_idx] * inv_a
        grad_w = (wx, wy)
        setattr(mesh, "_grad_w_xy", grad_w)
    wx, wy = grad_w
    df_loc = f_edge[..., e_idx] - field.unsqueeze(-1)  # [..., N_cells, 6]

    df_dx = (df_loc * wx).sum(dim=-1)
    df_dy = (df_loc * wy).sum(dim=-1)
    return df_dx, df_dy


def compute_frontogenesis_2d(
    mesh: SCVTMesh,
    theta_lvl: torch.Tensor,
    u_lvl: torch.Tensor,
    v_lvl: torch.Tensor,
) -> torch.Tensor:
    """2D kinematic frontogenesis function ``F = 0.5 * D/Dt |grad_h theta|^2`` [K^2 m^-2 s^-1].

    Reference: Hoskins (1982); Charron & Manzini (2002, Eq. 1); Richter et al. (2010, Eq. 1):
        F = -(dtheta/dx)^2 * (du/dx) - (dtheta/dy)^2 * (dv/dy)
            - (dtheta/dx)*(dtheta/dy)*(dv/dx + du/dy)
    """
    dth_dx, dth_dy = compute_scvt_horizontal_gradient(mesh, theta_lvl)
    du_dx, du_dy = compute_scvt_horizontal_gradient(mesh, u_lvl)
    dv_dx, dv_dy = compute_scvt_horizontal_gradient(mesh, v_lvl)
    return -(dth_dx * dth_dx) * du_dx - (dth_dy * dth_dy) * dv_dy - (dth_dx * dth_dy) * (dv_dx + du_dy)


def _propagate_spectral_batched(
    rho_d: torch.Tensor,
    u_proj_b: torch.Tensor,
    n_int: torch.Tensor,
    dz_c: torch.Tensor,
    tau_launch_b: torch.Tensor,
    k_launch_b: torch.Tensor,
    c_bins: torch.Tensor,
    u_src_b: torch.Tensor,
    effgw_b: torch.Tensor,
    k_h: float,
    dt: float,
) -> torch.Tensor:
    """Batched multi-axis upward spectral gravity-wave propagation ``[A, Nz, N_cells]`` (sync-free).

    Simultaneously propagates all active wave axes (e.g., ``A=1`` frontal-only or ``A=3``
    frontal + zonal convective + meridional convective) across discrete phase-speed bins ``c_m``
    with critical-level absorption, Lindzen saturation, and an open-top radiation boundary.
    """
    n_axes, nz, n_cells = u_proj_b.shape
    dtype = rho_d.dtype
    device = rho_d.device
    if can_use_crm_metal(rho_d) and c_bins.numel() <= 32:
        return dispatch_ngwd_propagate_metal(
            u_proj_b=u_proj_b,
            rho_d=rho_d,
            dz_c=dz_c,
            n_int=n_int,
            tau_launch_b=tau_launch_b,
            u_src_b=u_src_b,
            k_launch_b=k_launch_b,
            effgw_b=effgw_b,
            c_bins=c_bins,
            k_h=k_h,
            dt=dt,
            tend_max=NGWD_TEND_MAX,
        )

    c_bins_b = c_bins.view(1, -1, 1)                 # [1, M, 1]
    c_rel_src = c_bins_b - u_src_b.unsqueeze(1)      # [A, M, N_cells]
    sgn_m = torch.where(c_rel_src >= 0.0, torch.ones_like(c_rel_src), -torch.ones_like(c_rel_src))
    tau_cur = torch.where(torch.abs(c_rel_src) > 2.0, torch.clamp(tau_launch_b, min=0.0), torch.zeros_like(tau_launch_b))

    # Precompute launch masks [A, Nz, 1, N_cells] outside the vertical loop (zero GPU->CPU sync)
    k_grid = torch.arange(nz, device=device).view(1, nz, 1)
    is_launch_all = (k_launch_b.unsqueeze(1) == k_grid).unsqueeze(2)     # [A, Nz, 1, N_cells]
    above_launch_all = (k_launch_b.unsqueeze(1) <= k_grid).unsqueeze(2)  # [A, Nz, 1, N_cells]

    # Precompute interface density [Nz, N_cells], interface stability [Nz, N_cells], and axis interface winds
    rho_w_all = torch.empty_like(rho_d)
    rho_w_all[:-1] = 0.5 * (rho_d[:-1] + rho_d[1:])
    rho_w_all[-1] = rho_d[-1]

    n_w_all = torch.empty_like(rho_d)
    n_w_all[:-1] = torch.clamp(n_int, min=5.0e-3)
    n_w_all[-1] = torch.clamp(n_int[-1], min=5.0e-3)

    u_w_all = torch.empty_like(u_proj_b)
    u_w_all[:, :-1] = 0.5 * (u_proj_b[:, :-1] + u_proj_b[:, 1:])
    u_w_all[:, -1] = u_proj_b[:, -1]

    inv_mass_dz = 1.0 / torch.clamp(rho_d * dz_c, min=1.0e-4)  # [Nz, N_cells]
    sat_prefac = (0.5 * k_h) * effgw_b.view(n_axes, 1, 1)      # [A, 1, 1]
    sat_base_all = sat_prefac.unsqueeze(1) * (rho_w_all / n_w_all).view(1, nz, 1, n_cells)  # [A, Nz, 1, N_cells]

    c_min_crit = 2.0
    inv_2dt = 0.5 / max(dt, 1.0)
    du_dt = torch.zeros((n_axes, nz, n_cells), device=device, dtype=dtype)
    tau_prev = torch.zeros_like(tau_cur)

    for k in range(nz):
        tau_in = torch.where(is_launch_all[:, k], tau_cur, tau_prev)
        tau_in = torch.where(above_launch_all[:, k], tau_in, torch.zeros_like(tau_in))

        c_hat = c_bins_b - u_w_all[:, k].unsqueeze(1)          # [A, M, N_cells]
        crit = (c_hat * sgn_m) <= c_min_crit
        tau_sat = sat_base_all[:, k] * (torch.abs(c_hat) ** 3)
        tau_out = torch.where(crit, torch.zeros_like(tau_in), torch.minimum(tau_in, tau_sat))

        dtau_k = torch.clamp(tau_in - tau_out, min=0.0)
        c_diff_k = c_bins_b - u_proj_b[:, k].unsqueeze(1)
        raw_acc_m = (sgn_m * dtau_k) * inv_mass_dz[k].view(1, 1, n_cells)
        max_acc_m = inv_2dt * torch.abs(c_diff_k)
        du_dt[:, k] = torch.clamp(raw_acc_m, min=-max_acc_m, max=max_acc_m).sum(dim=1)

        tau_prev = tau_out

    return torch.clamp(du_dt, min=-NGWD_TEND_MAX, max=NGWD_TEND_MAX)


# AOBA Formulation Eqs. (8.6)-(8.7)
def compute_nonorographic_gwd(
    mesh: SCVTMesh,
    rho_d: torch.Tensor,
    pres_c: torch.Tensor,
    u_cell: torch.Tensor,
    v_cell: torch.Tensor,
    theta_m: torch.Tensor,
    z_rel_c: torch.Tensor,
    dz_c: torch.Tensor,
    q_conv_c: torch.Tensor | None,
    dt: float,
    n_phase_bins: int = 17,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute total map-free non-orographic GWD acceleration ``(du_ngwd, dv_ngwd)`` ``[Nz, N_cells]``.

    Combines (in a single sync-free batched GPU pass):
    - Frontal/jet-exit gravity waves triggered by SCVT 2D frontogenesis near 600 hPa
      (Charron & Manzini, 2002; Richter et al., 2010).
    - Convectively generated gravity waves diagnosed from 1:1 CRM diabatic heating ``q_conv_c``
      (Beres et al., 2004, 2005; Richter et al., 2010).
    """
    nz, n_cells = rho_d.shape
    dtype = rho_d.dtype
    device = rho_d.device

    if nz < 4:
        z = torch.zeros_like(u_cell)
        return z, z

    # 1. Brunt-Vaisala frequency N at internal interfaces [Nz - 1, N_cells]
    dz_int = torch.clamp(z_rel_c[1:] - z_rel_c[:-1], min=10.0)
    thm_int = torch.clamp(0.5 * (theta_m[1:] + theta_m[:-1]), min=180.0)
    n2_int = torch.clamp((C.GRAVITY / thm_int) * (theta_m[1:] - theta_m[:-1]) / dz_int, 2.5e-5, 1.6e-3)
    n_int = torch.sqrt(n2_int)

    # Discrete ground-based phase-speed spectrum [-c_max, ..., +c_max]
    n_bins = max(int(n_phase_bins), 5)
    c_bins = torch.linspace(-NGWD_CMAX, NGWD_CMAX, n_bins, device=device, dtype=dtype)
    dc = float((2.0 * NGWD_CMAX) / max(n_bins - 1, 1))
    k_h = 2.0 * math.pi / max(NGWD_WAVELENGTH_M, 1.0e4)

    u_proj_list: list[torch.Tensor] = []
    tau_list: list[torch.Tensor] = []
    k_launch_list: list[torch.Tensor] = []
    u_src_list: list[torch.Tensor] = []
    eff_list: list[float] = []

    use_front = NGWD_FRONT_TAU0 > 0.0 and NGWD_FRONT_EFF > 0.0
    if use_front:
        p_target_pa = NGWD_FRONT_LAUNCH_HPA * 100.0
        k_fr_idx = torch.clamp(torch.argmin(torch.abs(pres_c - p_target_pa), dim=0), min=1, max=nz - 2)
        c_ar = torch.arange(n_cells, device=device)

        th_fr = theta_m[k_fr_idx, c_ar]
        u_fr = u_cell[k_fr_idx, c_ar]
        v_fr = v_cell[k_fr_idx, c_ar]

        ff = compute_frontogenesis_2d(mesh, th_fr, u_fr, v_fr)
        ff_ratio = (ff - NGWD_FRONT_THRESH) / max(NGWD_FRONT_THRESH, 1.0e-18)
        fr_active = torch.sigmoid(2.5 * ff_ratio) * (ff > 0.5 * NGWD_FRONT_THRESH).to(dtype)

        spd_fr = torch.sqrt(u_fr * u_fr + v_fr * v_fr + 1.0e-4)
        ux_fr = u_fr / spd_fr
        uy_fr = v_fr / spd_fr
        u_proj_fr = u_cell * ux_fr.unsqueeze(0) + v_cell * uy_fr.unsqueeze(0)

        cw = max(NGWD_FRONT_CW, 5.0)
        c_rel = c_bins.unsqueeze(1) - spd_fr.unsqueeze(0)
        gauss = torch.exp(-((c_rel / cw) ** 2))
        gauss = torch.where(torch.abs(c_rel) > 2.0, gauss, torch.zeros_like(gauss))
        norm_g = torch.clamp(gauss.sum(dim=0, keepdim=True), min=1.0e-6)
        tau_fr_m = (NGWD_FRONT_TAU0 * fr_active).unsqueeze(0) * (gauss / norm_g)

        u_proj_list.append(u_proj_fr)
        tau_list.append(tau_fr_m)
        k_launch_list.append(k_fr_idx)
        u_src_list.append(spd_fr)
        eff_list.append(NGWD_FRONT_EFF)

    use_conv = q_conv_c is not None and NGWD_CONV_TAU0 > 0.0 and NGWD_CONV_EFF > 0.0
    if use_conv:
        assert q_conv_c is not None
        q_pos = torch.clamp(q_conv_c, min=0.0)
        q_trop = torch.where(pres_c > 8000.0, q_pos, torch.zeros_like(q_pos))

        q_max, _ = q_trop.max(dim=0)
        has_conv = q_max > (0.05 * NGWD_CONV_QREF)
        in_heat = (q_trop > (0.10 * q_max.unsqueeze(0))) & has_conv.unsqueeze(0)
        h_depth = torch.clamp((dz_c * in_heat.to(dtype)).sum(dim=0), min=1000.0, max=16000.0)

        k_grid = torch.arange(nz, device=device).unsqueeze(1).expand(nz, n_cells)
        k_top_conv = torch.clamp(torch.where(in_heat, k_grid, torch.zeros_like(k_grid)).max(dim=0).values, min=1, max=nz - 2)

        w_q = q_trop * dz_c
        w_q_sum = torch.clamp(w_q.sum(dim=0), min=1.0e-12)
        u_h = (w_q * u_cell).sum(dim=0) / w_q_sum
        v_h = (w_q * v_cell).sum(dim=0) / w_q_sum
        n_c_pad = torch.cat([n_int, n_int[-1:]], dim=0)
        n_h = torch.clamp((w_q * n_c_pad).sum(dim=0) / w_q_sum, min=8.0e-3, max=2.5e-2)

        sigma_c = torch.clamp(n_h * h_depth / math.pi, min=10.0, max=60.0)
        q_amp = torch.clamp(q_max / max(NGWD_CONV_QREF, 1.0e-8), max=4.0) ** 2
        tau_scale = torch.where(has_conv, NGWD_CONV_TAU0 * q_amp, torch.zeros_like(q_max))

        for axis_u, u_src_ax in ((u_cell, u_h), (v_cell, v_h)):
            c_rel_c = c_bins.unsqueeze(1) - u_src_ax.unsqueeze(0)
            x_m = torch.abs(c_rel_c) / sigma_c.unsqueeze(0)
            spec = x_m * torch.exp(-0.5 * x_m * x_m)
            spec = torch.where(torch.abs(c_rel_c) > 2.0, spec, torch.zeros_like(spec))
            norm_s = torch.clamp(spec.sum(dim=0, keepdim=True), min=1.0e-6)
            tau_conv_m = (0.5 * tau_scale).unsqueeze(0) * (spec / norm_s) * (10.0 / max(dc, 1.0))

            u_proj_list.append(axis_u)
            tau_list.append(tau_conv_m)
            k_launch_list.append(k_top_conv)
            u_src_list.append(u_src_ax)
            eff_list.append(NGWD_CONV_EFF)

    if not u_proj_list:
        z = torch.zeros_like(u_cell)
        return z, z

    eff_t = torch.tensor(eff_list, device=device, dtype=dtype)
    acc_all = _propagate_spectral_batched(
        rho_d=rho_d,
        u_proj_b=torch.stack(u_proj_list, dim=0),
        n_int=n_int,
        dz_c=dz_c,
        tau_launch_b=torch.stack(tau_list, dim=0),
        k_launch_b=torch.stack(k_launch_list, dim=0),
        c_bins=c_bins,
        u_src_b=torch.stack(u_src_list, dim=0),
        effgw_b=eff_t,
        k_h=k_h,
        dt=dt,
    )

    du_tot = torch.zeros_like(u_cell)
    dv_tot = torch.zeros_like(v_cell)
    idx_ax = 0
    if use_front:
        du_tot = du_tot + acc_all[idx_ax] * ux_fr.unsqueeze(0)
        dv_tot = dv_tot + acc_all[idx_ax] * uy_fr.unsqueeze(0)
        idx_ax += 1
    if use_conv:
        du_tot = du_tot + acc_all[idx_ax]
        dv_tot = dv_tot + acc_all[idx_ax + 1]

    return du_tot, dv_tot
