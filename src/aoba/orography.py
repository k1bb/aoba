"""Subgrid Orographic Blocked-Flow Drag & Gravity-Wave Drag (Lott & Miller, 1997).

Driven exclusively by subgrid elevation statistics (`sigma_topo = mu`, `slope_topo = sigma`,
`aniso_topo = gamma`, `angle_topo = psi`) aggregated directly from 5-arc-min `Earth2014`
(`Hirt & Rexer, 2015`).

References
----------
- Lott & Miller (1997, Q. J. R. Meteorol. Soc., 123, 101-127, Sec. 2.2-2.3, Eqs. 7-22):
  Directional elliptic mountain drag + low-level blocked-flow wake drag below `Z_b` +
  upward wave-saturation gravity-wave drag above `Z_b`.
- Palmer, Shutts, & Swinbank (1986, Q. J. R. Meteorol. Soc., 112, 1001-1039) &
  MIROC6 AGCM Document Writing Team (2021, Sec. 3.5):
  Lindzen saturation criterion and upper-boundary momentum conservation.

AOBA Formulation Reference: Sec. 8.4, Eqs. (8.8)-(8.9) (Lott & Miller Subgrid Orographic Blocking & Mountain-Wave Drag).
"""

from __future__ import annotations

import torch

from aoba import constants as C
from aoba.kernels import can_use_crm_metal, dispatch_oro_gwd_metal


# Lott & Miller (1997, Sec. 2.4) standard dimensionless constants
HN_CRIT: float = 0.50   # Critical non-dimensional mountain height H_nc [-]
CD_BLOCK: float = 1.00  # Blocked-flow bluff-body drag coefficient C_d [-]
G_WAVE: float = 0.80    # Gravity-wave launch amplitude coefficient G [-]


def compute_orographic_drag_lott_miller(
    rho_d: torch.Tensor,
    u_cell: torch.Tensor,
    v_cell: torch.Tensor,
    theta_m: torch.Tensor,
    z_rel_c: torch.Tensor,
    dz_c: torch.Tensor,
    sigma_topo: torch.Tensor,
    slope_topo: torch.Tensor,
    aniso_topo: torch.Tensor,
    angle_topo: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cell-centered zonal & meridional accelerations `(du_oro_dt, dv_oro_dt)` `[Nz, N_cells]`.

    Uses implicit limiting on both low-level wake drag and wave-breaking tendencies so large
    host time steps (`dt <= 600 s`) remain unconditionally stable over steep mountain ranges.
    """
    if can_use_crm_metal(rho_d) and rho_d.ndim == 2:
        return dispatch_oro_gwd_metal(
            rho_d=rho_d,
            u_cell=u_cell,
            v_cell=v_cell,
            theta_m=theta_m,
            z_rel_c=z_rel_c,
            dz_c=dz_c,
            sigma_topo=sigma_topo,
            slope_topo=slope_topo,
            aniso_topo=aniso_topo,
            angle_topo=angle_topo,
            dt=dt,
            gravity=C.GRAVITY,
            g_wave=G_WAVE,
            cd_block=CD_BLOCK,
            hn_crit=HN_CRIT,
        )
    nz, n_cells = rho_d.shape
    active = (sigma_topo > 10.0) & (slope_topo > 1.0e-4)

    mu = torch.clamp(sigma_topo, min=1.0)      # [N_cells]
    slope = torch.clamp(slope_topo, min=0.0)   # [N_cells]
    gamma = torch.clamp(aniso_topo, 0.0, 1.0)  # [N_cells]
    psi = angle_topo                           # [N_cells]
    h_eff_peak = 2.0 * mu                      # Effective peak height 2*mu [m]

    # 1. Brunt-Vaisala frequency N [s^-1] at cell centers & interfaces
    dz_int = torch.clamp(z_rel_c[1:] - z_rel_c[:-1], min=10.0)
    thm_int = torch.clamp(0.5 * (theta_m[1:] + theta_m[:-1]), min=200.0)
    n2_int = torch.clamp((C.GRAVITY / thm_int) * (theta_m[1:] - theta_m[:-1]) / dz_int, 2.5e-5, 1.0e-3)
    n_int = torch.sqrt(n2_int)  # [Nz - 1, N_cells]

    n_c = torch.empty_like(theta_m)
    n_c[0] = n_int[0]
    n_c[1:-1] = 0.5 * (n_int[:-1] + n_int[1:])
    n_c[-1] = n_int[-1]

    # 2. Low-level incident flow averaged over z_rel in [0.25*mu, 2*mu] (at least lowest 2 layers)
    in_env = (z_rel_c <= torch.clamp(h_eff_peak.unsqueeze(0), min=250.0)).to(rho_d.dtype)
    in_env[:2] = 1.0
    w_env = rho_d * dz_c * in_env
    w_sum = torch.clamp(w_env.sum(dim=0), min=1.0)

    u_low = (w_env * u_cell).sum(dim=0) / w_sum
    v_low = (w_env * v_cell).sum(dim=0) / w_sum
    n_low = (w_env * n_c).sum(dim=0) / w_sum
    rho_low = (w_env * rho_d).sum(dim=0) / w_sum

    speed_low = torch.sqrt(u_low * u_low + v_low * v_low + 1.0e-4)
    ux_hat = u_low / speed_low
    uy_hat = v_low / speed_low

    # Elliptic directional factor D_dir(phi, gamma) (Lott & Miller, 1997, Eqs. 7-8)
    theta_wind = torch.atan2(v_low, u_low + 1.0e-12)
    phi = theta_wind - psi
    cos2 = torch.cos(phi) ** 2
    sin2 = torch.sin(phi) ** 2
    b_fac = 1.0 - 0.18 * gamma - 0.04 * gamma * gamma
    c_fac = 0.48 * gamma + 0.30 * gamma * gamma
    d_dir = torch.clamp(
        torch.sqrt((b_fac * cos2 + c_fac * sin2) ** 2 + ((b_fac - c_fac) ** 2) * sin2 * cos2),
        0.05,
        1.5,
    )

    # 3. Blocked-flow depth Z_b & implicit low-level wake drag (Lott & Miller, 1997, Eqs. 9-13)
    hn = n_low * h_eff_peak / torch.clamp(speed_low, min=0.5)
    z_block = torch.where(
        active & (hn > HN_CRIT),
        torch.clamp(h_eff_peak * (1.0 - HN_CRIT / torch.clamp(hn, min=HN_CRIT)), min=0.0),
        torch.zeros_like(mu),
    )

    zb_col = z_block.unsqueeze(0)
    mu_col = mu.unsqueeze(0)
    below_zb = z_rel_c < zb_col
    geom_wake = torch.where(
        below_zb,
        torch.sqrt(torch.clamp(zb_col - z_rel_c, min=0.0) / (z_rel_c + mu_col)),
        torch.zeros_like(z_rel_c),
    )
    speed_z = torch.sqrt(u_cell * u_cell + v_cell * v_cell + 1.0e-6)
    # Implicit Rayleigh damping coefficient r_blk [s^-1]: du/dt = -u * r_blk / (1 + dt * r_blk)
    r_blk = (
        CD_BLOCK
        * (slope / (2.0 * mu)).unsqueeze(0)
        * d_dir.unsqueeze(0)
        * geom_wake
        * speed_z
        * active.to(rho_d.dtype).unsqueeze(0)
    )
    eff_rate_blk = r_blk / (1.0 + dt * r_blk)
    du_blk = -eff_rate_blk * u_cell
    dv_blk = -eff_rate_blk * v_cell

    # 4. Gravity-wave launch stress at z = Z_b & Lindzen upward wave saturation (Eqs. 14-18)
    h_wave = torch.clamp(h_eff_peak - z_block, min=0.0)
    k_wave = G_WAVE * (slope / (4.0 * mu)) * d_dir * active.to(rho_d.dtype)  # [m^-1]
    tau_launch = rho_low * k_wave * n_low * speed_low * (h_wave * h_wave)     # [Pa] >= 0

    # Walk upward across interfaces m = 0..Nz (tau_gw >= 0 is stress magnitude opposing u_hat_low)
    tau_gw = torch.zeros((nz + 1, n_cells), device=rho_d.device, dtype=rho_d.dtype)
    tau_cur = tau_launch
    tau_gw[0] = tau_cur

    for k in range(1, nz):
        rho_w = 0.5 * (rho_d[k - 1] + rho_d[k])
        u_w = 0.5 * (u_cell[k - 1] + u_cell[k])
        v_w = 0.5 * (v_cell[k - 1] + v_cell[k])
        z_w_approx = 0.5 * (z_rel_c[k - 1] + z_rel_c[k])
        u_proj = torch.clamp(u_w * ux_hat + v_w * vy_hat if False else u_w * ux_hat + v_w * uy_hat, min=0.0)
        tau_sat = rho_w * k_wave * (u_proj ** 3) / (n_int[k - 1] * (HN_CRIT ** 2))
        # Below Z_b keep launch stress; above Z_b apply Lindzen saturation min(tau_cur, tau_sat)
        tau_cur = torch.where(z_w_approx <= z_block, tau_cur, torch.minimum(tau_cur, tau_sat))
        tau_gw[k] = tau_cur
    # Deposit any remaining wave flux inside top layer (tau_gw[nz] = 0 -> exact column momentum conservation)
    tau_gw[nz] = 0.0

    # Stress vector is (-tau_gw * ux_hat, -tau_gw * uy_hat); acceleration = -d(stress)/dz / rho_d
    dtau_dz = (tau_gw[:-1] - tau_gw[1:]) / (rho_d * dz_c)  # >= 0 deceleration along u_hat_low
    # Bound wave deceleration so projected wind cannot overshoot past zero in one step
    u_proj_c = torch.clamp(u_cell * ux_hat.unsqueeze(0) + v_cell * uy_hat.unsqueeze(0), min=0.0)
    decel_mag = torch.minimum(dtau_dz, 0.8 * u_proj_c / max(dt, 1.0))

    du_gwd = -decel_mag * ux_hat.unsqueeze(0)
    dv_gwd = -decel_mag * uy_hat.unsqueeze(0)

    return du_blk + du_gwd, dv_blk + dv_gwd
