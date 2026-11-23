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
"""

from __future__ import annotations

import math
import torch

from aoba import constants as C
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
    sgn = mesh.edge_sign_on_cell        # [N_cells, 6]
    df_loc = f_edge[..., e_idx] - field.unsqueeze(-1)  # [..., N_cells, 6]
    w_base = sgn * mesh.dv_edge[e_idx]  # [N_cells, 6]

    nx_e = mesh.east_normal_edge[e_idx]
    ny_e = mesh.north_normal_edge[e_idx]
    inv_a = 1.0 / torch.clamp(mesh.area_cell, min=1.0)

    df_dx = (df_loc * (w_base * nx_e)).sum(dim=-1) * inv_a
    df_dy = (df_loc * (w_base * ny_e)).sum(dim=-1) * inv_a
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


def _propagate_spectral_column_axis(
    rho_d: torch.Tensor,
    u_proj: torch.Tensor,
    n_int: torch.Tensor,
    dz_c: torch.Tensor,
    tau_launch_m: torch.Tensor,
    k_launch: torch.Tensor,
    c_bins: torch.Tensor,
    u_src: torch.Tensor,
    effgw: float,
    k_h: float,
    dt: float,
) -> torch.Tensor:
    """Propagate a discrete phase-speed spectrum upward along one horizontal axis.

    Implements Lindzen (1981), Alexander & Dunkerton (1999), and Richter et al. (2010, Eqs. 4-10)
    with critical-level absorption and an open-top radiation boundary (unbroken flux leaves the
    model top without artificial pile-up in the highest layer).

    Parameters
    ----------
    rho_d : [Nz, N_cells]
        Dry-air density [kg m^-3].
    u_proj : [Nz, N_cells]
        Background wind projected onto the wave propagation axis [m s^-1].
    n_int : [Nz - 1, N_cells]
        Brunt-Vaisala frequency at internal interfaces [s^-1].
    dz_c : [Nz, N_cells]
        Layer thicknesses [m].
    tau_launch_m : [M, N_cells]
        Non-negative launch momentum flux magnitude per phase-speed bin [Pa].
    k_launch : [N_cells] int64
        Launch layer index (flux enters at interface ``k_launch`` and propagates upward).
    c_bins : [M]
        Ground-based phase speeds [m s^-1].
    u_src : [N_cells]
        Background wind at launch level [m s^-1].
    effgw : float
        Intermittency efficiency factor [-].
    k_h : float
        Representative horizontal wavenumber ``2*pi / lambda_h`` [m^-1].
    dt : float
        Coupling time step [s].

    Returns
    -------
    du_dt : [Nz, N_cells]
        Axis-aligned wind acceleration [m s^-2].
    """
    nz, n_cells = rho_d.shape
    dtype = rho_d.dtype
    device = rho_d.device

    # Intrinsic phase speed sign at launch determines momentum-flux sign (+1 eastward/northward, -1 opposite)
    c_rel_src = c_bins.unsqueeze(1) - u_src.unsqueeze(0)  # [M, N_cells]
    sgn_m = torch.where(c_rel_src >= 0.0, torch.ones_like(c_rel_src), -torch.ones_like(c_rel_src))
    # Suppress zero-intrinsic-speed bins at launch
    active_m = torch.abs(c_rel_src) > 2.0
    tau_cur = torch.where(active_m, torch.clamp(tau_launch_m, min=0.0), torch.zeros_like(tau_launch_m))

    # Signed flux profile at interfaces [Nz + 1, M, N_cells] (open top: tau_signed[nz] = tau_cur at top)
    # Accumulate tendency directly layer-by-layer to avoid storing a 3D [Nz+1, M, N_cells] tensor!
    du_dt = torch.zeros((nz, n_cells), device=device, dtype=dtype)
    tau_prev = torch.zeros_like(tau_cur)  # flux entering bottom of layer k

    c_min_crit = 2.0
    sat_prefac = 0.5 * effgw * k_h

    for k in range(nz):
        # Activate launch spectrum at layer k == k_launch
        is_launch = (k_launch == k).unsqueeze(0)  # [1, N_cells]
        above_launch = (k_launch <= k).unsqueeze(0)
        tau_in = torch.where(is_launch, tau_cur, tau_prev)
        tau_in = torch.where(above_launch, tau_in, torch.zeros_like(tau_in))

        if k < nz - 1:
            rho_w = 0.5 * (rho_d[k] + rho_d[k + 1]).unsqueeze(0)  # [1, N_cells]
            u_w = 0.5 * (u_proj[k] + u_proj[k + 1]).unsqueeze(0)  # [1, N_cells]
            n_w = torch.clamp(n_int[k].unsqueeze(0), min=5.0e-3)  # [1, N_cells]
            c_hat = c_bins.unsqueeze(1) - u_w                     # [M, N_cells]
            # Critical-level filter: sign flip relative to launch or |c_hat| < c_min
            crit = (c_hat * sgn_m) <= c_min_crit
            tau_sat = sat_prefac * rho_w * (torch.abs(c_hat) ** 3) / n_w
            tau_out = torch.where(crit, torch.zeros_like(tau_in), torch.minimum(tau_in, tau_sat))
        else:
            # Open-top radiation boundary: check local saturation at top layer center, let remainder exit
            rho_t = rho_d[k].unsqueeze(0)
            u_t = u_proj[k].unsqueeze(0)
            n_t = torch.clamp(n_int[-1].unsqueeze(0), min=5.0e-3)
            c_hat = c_bins.unsqueeze(1) - u_t
            crit = (c_hat * sgn_m) <= c_min_crit
            tau_sat = sat_prefac * rho_t * (torch.abs(c_hat) ** 3) / n_t
            tau_out = torch.where(crit, torch.zeros_like(tau_in), torch.minimum(tau_in, tau_sat))

        # Momentum flux convergence in layer k: du/dt = sum_m sgn_m * (tau_in - tau_out) / (rho_d * dz)
        dtau_k = torch.clamp(tau_in - tau_out, min=0.0)
        # Prevent single-bin overshoot past its own phase speed c_m within dt
        c_diff_k = c_bins.unsqueeze(1) - u_proj[k].unsqueeze(0)
        raw_acc_m = (sgn_m * dtau_k) / torch.clamp((rho_d[k] * dz_c[k]).unsqueeze(0), min=1.0e-4)
        max_acc_m = 0.5 * torch.abs(c_diff_k) / max(dt, 1.0)
        acc_m = torch.clamp(raw_acc_m, min=-max_acc_m, max=max_acc_m)
        du_dt[k] = acc_m.sum(dim=0)

        tau_prev = tau_out

    return torch.clamp(du_dt, min=-NGWD_TEND_MAX, max=NGWD_TEND_MAX)


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

    Combines:
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

    du_tot = torch.zeros_like(u_cell)
    dv_tot = torch.zeros_like(v_cell)

    # ------------------------------------------------------------------
    # A. Frontal / Jet-Exit GWD (Charron & Manzini, 2002; Richter et al., 2010)
    # ------------------------------------------------------------------
    if NGWD_FRONT_TAU0 > 0.0 and NGWD_FRONT_EFF > 0.0:
        p_target_pa = NGWD_FRONT_LAUNCH_HPA * 100.0
        k_fr_idx = torch.argmin(torch.abs(pres_c - p_target_pa), dim=0)  # [N_cells]
        k_fr_idx = torch.clamp(k_fr_idx, min=1, max=nz - 2)
        c_ar = torch.arange(n_cells, device=device)

        th_fr = theta_m[k_fr_idx, c_ar]
        u_fr = u_cell[k_fr_idx, c_ar]
        v_fr = v_cell[k_fr_idx, c_ar]

        ff = compute_frontogenesis_2d(mesh, th_fr, u_fr, v_fr)  # [N_cells]
        # Smooth activation above threshold NGWD_FRONT_THRESH (reduces grid-point flicker)
        ff_ratio = (ff - NGWD_FRONT_THRESH) / max(NGWD_FRONT_THRESH, 1.0e-18)
        fr_active = torch.sigmoid(2.5 * ff_ratio) * (ff > 0.5 * NGWD_FRONT_THRESH).to(dtype)

        if float(fr_active.max().item()) > 1.0e-4:
            spd_fr = torch.sqrt(u_fr * u_fr + v_fr * v_fr + 1.0e-4)
            ux_fr = u_fr / spd_fr
            uy_fr = v_fr / spd_fr
            # Project full column wind onto frontal steering direction
            u_proj_fr = u_cell * ux_fr.unsqueeze(0) + v_cell * uy_fr.unsqueeze(0)
            u_src_fr = spd_fr

            # Gaussian phase-speed spectrum centered on steering wind (Richter et al., 2010, Eq. 2)
            # Normalized so sum_m tau_m ~ NGWD_FRONT_TAU0 * fr_active
            cw = max(NGWD_FRONT_CW, 5.0)
            c_rel = c_bins.unsqueeze(1) - u_src_fr.unsqueeze(0)
            gauss = torch.exp(-((c_rel / cw) ** 2))
            # Exclude c_rel near zero to avoid zero-phase-speed singularity
            gauss = torch.where(torch.abs(c_rel) > 2.0, gauss, torch.zeros_like(gauss))
            norm_g = torch.clamp(gauss.sum(dim=0, keepdim=True), min=1.0e-6)
            tau_fr_m = (NGWD_FRONT_TAU0 * fr_active).unsqueeze(0) * (gauss / norm_g)

            acc_fr = _propagate_spectral_column_axis(
                rho_d=rho_d,
                u_proj=u_proj_fr,
                n_int=n_int,
                dz_c=dz_c,
                tau_launch_m=tau_fr_m,
                k_launch=k_fr_idx,
                c_bins=c_bins,
                u_src=u_src_fr,
                effgw=NGWD_FRONT_EFF,
                k_h=k_h,
                dt=dt,
            )
            du_tot = du_tot + acc_fr * ux_fr.unsqueeze(0)
            dv_tot = dv_tot + acc_fr * uy_fr.unsqueeze(0)

    # ------------------------------------------------------------------
    # B. Convective GWD from 1:1 CRM Diabatic Heating (Beres et al., 2004, 2005)
    # ------------------------------------------------------------------
    if q_conv_c is not None and NGWD_CONV_TAU0 > 0.0 and NGWD_CONV_EFF > 0.0:
        q_pos = torch.clamp(q_conv_c, min=0.0)  # [Nz, N_cells] [K s^-1]
        # Mask strictly within troposphere (p > 80 hPa) so stratospheric radiative heating is never treated as convection
        in_trop = pres_c > 8000.0
        q_trop = torch.where(in_trop, q_pos, torch.zeros_like(q_pos))

        q_max, _ = q_trop.max(dim=0)  # [N_cells]
        has_conv = q_max > (0.05 * NGWD_CONV_QREF)
        if has_conv.any():
            # Heating layer mask: Q > 0.1 * Q_max (Beres et al., 2004, Sec. 2)
            in_heat = (q_trop > (0.10 * q_max.unsqueeze(0))) & has_conv.unsqueeze(0)
            h_depth = (dz_c * in_heat.to(dtype)).sum(dim=0)  # [N_cells] [m]
            h_depth = torch.clamp(h_depth, min=1000.0, max=16000.0)

            # Launch level = top of heating layer (clamped inside [1, nz-2])
            k_grid = torch.arange(nz, device=device).unsqueeze(1).expand(nz, n_cells)
            k_top_conv = torch.where(in_heat, k_grid, torch.zeros_like(k_grid)).max(dim=0).values
            k_top_conv = torch.clamp(k_top_conv, min=1, max=nz - 2)

            # Heating-weighted mean tropospheric wind (u_h, v_h) and stability N_h
            w_q = q_trop * dz_c
            w_q_sum = torch.clamp(w_q.sum(dim=0), min=1.0e-12)
            u_h = (w_q * u_cell).sum(dim=0) / w_q_sum
            v_h = (w_q * v_cell).sum(dim=0) / w_q_sum
            n_c_pad = torch.cat([n_int, n_int[-1:]], dim=0)
            n_h = torch.clamp((w_q * n_c_pad).sum(dim=0) / w_q_sum, min=8.0e-3, max=2.5e-2)

            # Beres et al. (2004, Fig. 6): spectral peak phase speed sigma_c = N_h * h_d / pi
            sigma_c = torch.clamp(n_h * h_depth / math.pi, min=10.0, max=60.0)  # [N_cells]
            q_amp = torch.clamp(q_max / max(NGWD_CONV_QREF, 1.0e-8), max=4.0) ** 2
            tau_scale = torch.where(has_conv, NGWD_CONV_TAU0 * q_amp, torch.zeros_like(q_max))

            # Launch along orthogonal zonal (u) and meridional (v) axes (0.5 flux share per axis)
            for axis_u, u_src_ax, is_zonal in ((u_cell, u_h, True), (v_cell, v_h, False)):
                x_m = torch.abs(c_bins.unsqueeze(1) - u_src_ax.unsqueeze(0)) / sigma_c.unsqueeze(0)
                # Double-peaked Beres spectrum: x * exp(-0.5 * x^2) peaks at |c - u_h| = sigma_c
                spec = x_m * torch.exp(-0.5 * x_m * x_m)
                spec = torch.where(torch.abs(c_bins.unsqueeze(1) - u_src_ax.unsqueeze(0)) > 2.0, spec, torch.zeros_like(spec))
                norm_s = torch.clamp(spec.sum(dim=0, keepdim=True), min=1.0e-6)
                # Scale total integrated stress per axis cleanly (~0.5 * tau_scale * dc / 10)
                tau_conv_m = (0.5 * tau_scale).unsqueeze(0) * (spec / norm_s) * (10.0 / max(dc, 1.0))

                acc_ax = _propagate_spectral_column_axis(
                    rho_d=rho_d,
                    u_proj=axis_u,
                    n_int=n_int,
                    dz_c=dz_c,
                    tau_launch_m=tau_conv_m,
                    k_launch=k_top_conv,
                    c_bins=c_bins,
                    u_src=u_src_ax,
                    effgw=NGWD_CONV_EFF,
                    k_h=k_h,
                    dt=dt,
                )
                if is_zonal:
                    du_tot = du_tot + acc_ax
                else:
                    dv_tot = dv_tot + acc_ax

    return du_tot, dv_tot
