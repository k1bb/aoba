"""3D Compressible Nonhydrostatic HEVI Cloud-Resolving Model (CRM), Zalesak FCT, & RCE Driver.

Implements the embedded doubly periodic 3D Arakawa C-grid / Charney-Phillips nonhydrostatic
CRM (`N_x x N_y = 8 x 8`, `dx = dy = 4 km` default) batched across `N_col` host columns for
both standalone Radiative-Convective Equilibrium (RCE) and 1:1 Superparameterization (SP).

References
----------
- Skamarock et al. (2012, Mon. Wea. Rev., 140, 3090-3105, Sec. 2-3):
  Compressible flux-form moist Euler equations using `(rho_d, Theta_m = rho_d * theta_m)`
  and 3-stage Wicker & Skamarock (2002) RK3 + Horizontally Explicit / Vertically Implicit
  (HEVI) acoustic-gravity integration.
- Klemp, Skamarock, & Dudhia (2007, Mon. Wea. Rev., 135, 2897-2913) &
  Nishizawa et al. (2025, SCALE-RM v5.5.5, Ch. 3 Eqs. 3.1-3.105):
  Off-centered implicit vertical Helmholtz tridiagonal solver on `F_z = rho_d * w`
  guaranteeing exact discrete mass and moist-potential-temperature conservation, plus
  upper-tropospheric/stratospheric Rayleigh sponge damping.
- Zalesak (1979, J. Comput. Phys., 31, 335-362, Eqs. 1-17) &
  SCALE-RM (Nishizawa et al., 2025, App. A.2 Eqs. A.20-A.35):
  Multidimensional positive-definite Flux-Corrected Transport (FCT) combining 1st-order
  monotone donor-cell upwind fluxes with 4th-order centered fluxes.
- Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, 607-623),
  Khairoutdinov, Randall, & DeMott (2005, J. Atmos. Sci., 62, 2136-2154), &
  Yamazaki & Miura (2024, J. Adv. Model. Earth Syst., 16, e2023MS004133):
  3D CRM formulation and large-scale Superparameterization tendency coupling.

AOBA Formulation Reference: Sec. 8.1 & 8.3, Eqs. (8.1)-(8.3) (Embedded Nonhydrostatic RK3-HEVI CRM & Standalone RCE).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from types import SimpleNamespace

from aoba import constants as C
from aoba.runtime import can_use_cuda_graph, run_cuda_graph_cached
from aoba.biosphere import (
    compute_optimal_soil_water_beta,
    update_daily_acclimation_smith,
    update_daily_lai_root_height,
)
from aoba.kernels import (
    can_use_crm_metal,
    dispatch_crm_fct_advect_metal,
    dispatch_crm_fct_pair_metal,
    mean_2d,
    dispatch_crm_fct_quad_metal,
    dispatch_crm_sgs_metal,
    run_crm_rk3_hevi_metal,
    solve_tridiagonal_batched,
)
from aoba.microphysics import step_microphysics, step_microphysics_two_moment
from aoba.radiation import compute_four_band_radiation
from aoba.surface import SurfaceFluxDiagnostics, SurfaceState, compute_surface_albedos, step_surface
from aoba.thermodynamics import (
    diagnose_pressure_temperature,
    rcemip_analytic_sounding,
    saturation_adjustment,
)
from aoba.turbulence import TKE_MIN, compute_smagorinsky_lilly_3d, step_mynn_level25
from aoba.vertical_grid import VerticalGrid


@dataclass
class CRMState:
    """Batched 3D nonhydrostatic CRM state on `[Nz, N_col, Ny, Nx]` Arakawa C-grid."""

    rho_d: torch.Tensor    # Dry air density [kg m^-3], shape [Nz, N_col, Ny, Nx]
    u: torch.Tensor        # Zonal velocity at x-edges (i+1/2) [m s^-1], shape [Nz, N_col, Ny, Nx]
    v: torch.Tensor        # Meridional velocity at y-edges (j+1/2) [m s^-1], shape [Nz, N_col, Ny, Nx]
    w: torch.Tensor        # Vertical velocity at z-interfaces [m s^-1], shape [Nz+1, N_col, Ny, Nx]
    theta_m: torch.Tensor  # Modified moist potential temperature [K], shape [Nz, N_col, Ny, Nx]
    q_n: torch.Tensor      # Non-precipitating water q_v + q_c + q_i [kg kg^-1], shape [Nz, N_col, Ny, Nx]
    q_p: torch.Tensor      # Precipitating water q_r + q_s + q_g [kg kg^-1], shape [Nz, N_col, Ny, Nx]
    q_v: torch.Tensor      # Diagnosed water vapor mixing ratio [kg kg^-1], shape [Nz, N_col, Ny, Nx]
    q_c: torch.Tensor      # Diagnosed cloud liquid mixing ratio [kg kg^-1], shape [Nz, N_col, Ny, Nx]
    q_i: torch.Tensor      # Diagnosed cloud ice mixing ratio [kg kg^-1], shape [Nz, N_col, Ny, Nx]
    qke: torch.Tensor      # Doubled MYNN subgrid TKE 2*e [m^2 s^-2], shape [Nz, N_col, Ny, Nx]
    dx: float = 4000.0     # Horizontal x-spacing [m]
    dy: float = 4000.0     # Horizontal y-spacing [m]
    f_coriolis: torch.Tensor | None = None  # Column Coriolis parameter [s^-1], shape [N_col]
    n_c: torch.Tensor | None = None  # 2-moment cloud droplet number mixing ratio [kg^-1], [Nz, N_col, Ny, Nx]
    n_i: torch.Tensor | None = None  # 2-moment cloud ice crystal number mixing ratio [kg^-1], [Nz, N_col, Ny, Nx]
    # Optional 6-class 2-moment precipitation hydrometeors (microphysics_scheme == "six_class_2m")
    q_r: torch.Tensor | None = None  # Prognostic rain mass mixing ratio [kg kg^-1]
    q_s: torch.Tensor | None = None  # Prognostic snow mass mixing ratio [kg kg^-1]
    q_g: torch.Tensor | None = None  # Prognostic graupel mass mixing ratio [kg kg^-1]
    n_r: torch.Tensor | None = None  # Prognostic rain number mixing ratio [kg^-1]
    n_s: torch.Tensor | None = None  # Prognostic snow number mixing ratio [kg^-1]
    n_g: torch.Tensor | None = None  # Prognostic graupel number mixing ratio [kg^-1]


@dataclass
class SPTendencies:
    """Horizontal-mean CRM physics/convective tendencies returned to host SCVT columns."""

    dtheta_m_dt: torch.Tensor   # [Nz, N_col] [K s^-1]
    dq_n_dt: torch.Tensor       # [Nz, N_col] [kg kg^-1 s^-1]
    dq_p_dt: torch.Tensor       # [Nz, N_col] [kg kg^-1 s^-1]
    du_dt: torch.Tensor         # [Nz, N_col] [m s^-2]
    dv_dt: torch.Tensor         # [Nz, N_col] [m s^-2]
    precip_rain: torch.Tensor   # Mean surface rain rate [N_col] [kg m^-2 s^-1]
    precip_snow: torch.Tensor   # Mean surface snow/ice rate [N_col] [kg m^-2 s^-1]
    precip_rain_sub: torch.Tensor | None = None  # Subcolumn rain rate [N_col, Ny, Nx] [kg m^-2 s^-1]
    precip_snow_sub: torch.Tensor | None = None  # Subcolumn snow/ice rate [N_col, Ny, Nx] [kg m^-2 s^-1]


# ---------------------------------------------------------------------------
# 1. Exact Discrete Hydrostatic Initialization
# ---------------------------------------------------------------------------
def build_discrete_hydrostatic_rcemip_crm(
    vgrid: VerticalGrid,
    n_col: int = 1,
    ny: int = 8,
    nx: int = 8,
    dx: float = 4000.0,
    dy: float = 4000.0,
    t_sfc: float = 300.0,
    p_sfc: float = 101480.0,
    f_coriolis: float = 0.0,
    pert_amplitude_k: float = 0.05,
    seed: int = 42,
) -> CRMState:
    """Initialize a discretely hydrostatic RCEMIP 3D CRM state on `vgrid.device`.

    Iteratively refines dry density `rho_d[k]` from the analytic Wing et al. (2018) sounding
    in `float64` so that discrete vertical pressure differences satisfy:

        (p[k] - p[k-1]) / dz_w[k] + 0.5 * (rho_tot[k] + rho_tot[k-1]) * g == 0

    to machine precision before casting to `vgrid.dtype`, preventing spurious startup acoustic
    waves on `float32` GPU.
    """
    dev = vgrid.zeta_c.device
    dtype = vgrid.zeta_c.dtype
    nz = vgrid.nz

    z_c64 = vgrid.zeta_c.cpu().to(dtype=torch.float64)
    z_w64 = vgrid.zeta_w.cpu().to(dtype=torch.float64)
    prof = rcemip_analytic_sounding(z_c64, z_w64, sst=t_sfc, p_sfc=p_sfc)

    temp = prof["temp_c"].clone()     # [Nz]
    qv = prof["qv_c"].clone()         # [Nz]
    gamma = C.GAMMA_D

    # Integrate discrete hydrostatic pressure from p_sfc upward
    dz_0 = float(z_c64[0] - z_w64[0])
    tv_0 = temp[0] * (1.0 + C.RV_OVER_RD * qv[0]) / (1.0 + qv[0])
    p_col = torch.empty(nz, dtype=torch.float64)
    p_col[0] = p_sfc * torch.exp(-C.GRAVITY * dz_0 / (C.R_D * tv_0))

    rho_d_col = p_col.clone()
    rho_d_col[0] = p_col[0] / (C.R_D * temp[0] * (1.0 + C.RV_OVER_RD * qv[0]))

    for k in range(1, nz):
        dz_wk = float(z_c64[k] - z_c64[k - 1])
        rho_tot_below = rho_d_col[k - 1] * (1.0 + qv[k - 1])
        # Solve p_k - p_{k-1} + 0.5 * dz_wk * g * (rho_tot_below + p_k * (1 + qv_k) / (R_d * T_k * (1 + rv/rd * qv_k))) = 0
        fac_k = (1.0 + qv[k]) / (C.R_D * temp[k] * (1.0 + C.RV_OVER_RD * qv[k]))
        p_col[k] = (p_col[k - 1] - 0.5 * dz_wk * C.GRAVITY * rho_tot_below) / (
            1.0 + 0.5 * dz_wk * C.GRAVITY * fac_k
        )
        rho_d_col[k] = p_col[k] / (C.R_D * temp[k] * (1.0 + C.RV_OVER_RD * qv[k]))

    # Exact theta_m consistent with full compressible EOS p = p_0 * (R_d * rho_d * theta_m / p_0)^gamma
    theta_m_col = (C.P_REF / (C.R_D * rho_d_col)) * torch.pow(p_col / C.P_REF, 1.0 / gamma)

    shape4 = (nz, n_col, ny, nx)
    rho_d = rho_d_col.to(device=dev, dtype=dtype).view(nz, 1, 1, 1).expand(shape4).clone()
    theta_m = theta_m_col.to(device=dev, dtype=dtype).view(nz, 1, 1, 1).expand(shape4).clone()
    q_v = qv.to(device=dev, dtype=dtype).view(nz, 1, 1, 1).expand(shape4).clone()

    if pert_amplitude_k > 0.0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        # Zero-mean random boundary-layer thermal noise in lowest 4 levels (Wing et al., 2018)
        noise = (torch.rand((4, n_col, ny, nx), generator=gen, dtype=torch.float32) - 0.5) * 2.0 * pert_amplitude_k
        noise = noise - mean_2d(noise, keepdim=True)
        theta_m[:4] = theta_m[:4] + noise.to(device=dev, dtype=dtype)

    zeros = torch.zeros(shape4, device=dev, dtype=dtype)
    w = torch.zeros((nz + 1, n_col, ny, nx), device=dev, dtype=dtype)
    qke = torch.full(shape4, 2.0 * TKE_MIN, device=dev, dtype=dtype)
    f_cor = torch.full((n_col,), f_coriolis, device=dev, dtype=dtype)

    return CRMState(
        rho_d=rho_d,
        u=zeros.clone(),
        v=zeros.clone(),
        w=w,
        theta_m=theta_m,
        q_n=q_v.clone(),
        q_p=zeros.clone(),
        q_v=q_v,
        q_c=zeros.clone(),
        q_i=zeros.clone(),
        qke=qke,
        dx=dx,
        dy=dy,
        f_coriolis=f_cor,
    )


# ---------------------------------------------------------------------------
# 2. Zalesak (1979) 3D Positive-Definite Flux-Corrected Transport (FCT)
# ---------------------------------------------------------------------------
def fct_advect_3d(
    q: torch.Tensor,
    rho_old: torch.Tensor,
    rho_new: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Advect non-negative tracer `q` over `dt` with exact conservation & positivity.

    Parameters
    ----------
    q : torch.Tensor
        Tracer mixing ratio `[Nz, N_col, Ny, Nx]`, `q >= 0`.
    rho_old, rho_new : torch.Tensor
        Dry air density before and after mass continuity step `[Nz, N_col, Ny, Nx]`, satisfying
        `rho_new == rho_old - dt * (div_x(fx) + div_y(fy) + div_z(fz))`.
    fx, fy : torch.Tensor
        Horizontal mass fluxes `[Nz, N_col, Ny, Nx]` at `(i+1/2, j)` and `(i, j+1/2)` [kg m^-2 s^-1].
    fz : torch.Tensor
        Vertical mass flux `[Nz+1, N_col, Ny, Nx]` at interfaces `k-1/2` (`fz[0] == fz[Nz] == 0`).
    dz_c : torch.Tensor
        Layer thickness `[Nz]` or `[Nz, 1, 1, 1]` [m].

    References
    ----------
    - Zalesak (1979, J. Comput. Phys., 31, Eqs. 1-17).
    - Nishizawa et al. (2025, SCALE-RM v5.5.5, App. A.2 Eqs. A.20-A.35).
    """
    nz = q.shape[0]
    if can_use_crm_metal(rho_old) and can_use_crm_metal(q):
        return dispatch_crm_fct_advect_metal(
            q.unsqueeze(0), rho_old, rho_new, fx, fy, fz, dx, dy, dz_c, dt
        )[0]
    dz = dz_c.view(nz, -1, 1, 1)
    eps = 1.0e-20

    # --- 1. Low-order 1st-order donor-cell upwind fluxes ---
    # fx[i] sits at face (i + 1/2) between cell i and cell (i + 1) mod Nx
    q_xp1 = torch.roll(q, shifts=-1, dims=-1)
    q_xm1 = torch.roll(q, shifts=1, dims=-1)
    q_xp2 = torch.roll(q, shifts=-2, dims=-1)
    fl_x = torch.clamp(fx, min=0.0) * q + torch.clamp(fx, max=0.0) * q_xp1

    # fy[j] sits at face (j + 1/2) between cell j and cell (j + 1) mod Ny
    q_yp1 = torch.roll(q, shifts=-1, dims=-2)
    q_ym1 = torch.roll(q, shifts=1, dims=-2)
    q_yp2 = torch.roll(q, shifts=-2, dims=-2)
    fl_y = torch.clamp(fy, min=0.0) * q + torch.clamp(fy, max=0.0) * q_yp1

    # fz[m] (m = 1..Nz-1) sits between cell m-1 (below) and cell m (above)
    fl_z = torch.zeros_like(fz)
    fz_int = fz[1:-1]
    fl_z[1:-1] = torch.clamp(fz_int, min=0.0) * q[:-1] + torch.clamp(fz_int, max=0.0) * q[1:]

    # Transported & diffused (monotone low-order) update
    div_fl = (
        (fl_x - torch.roll(fl_x, shifts=1, dims=-1)) / dx
        + (fl_y - torch.roll(fl_y, shifts=1, dims=-2)) / dy
        + (fl_z[1:] - fl_z[:-1]) / dz
    )
    rho_safe = torch.clamp(rho_new, min=1.0e-5)
    q_td = torch.clamp((rho_old * q - dt * div_fl) / rho_safe, min=0.0)

    # --- 2. High-order 4th-order centered horizontal & 2nd/4th-order vertical fluxes ---
    fh_x = fx * ((-q_xm1 + 7.0 * q + 7.0 * q_xp1 - q_xp2) / 12.0)
    fh_y = fy * ((-q_ym1 + 7.0 * q + 7.0 * q_yp1 - q_yp2) / 12.0)
    fh_z = torch.zeros_like(fz)
    fh_z[1:-1] = fz_int * (0.5 * (q[:-1] + q[1:]))
    if nz >= 4:
        fh_z[2:-2] = fz[2:-2] * ((-q[:-3] + 7.0 * q[1:-2] + 7.0 * q[2:-1] - q[3:]) / 12.0)

    # Anti-diffusive fluxes A = F^H - F^L
    ax = fh_x - fl_x
    ay = fh_y - fl_y
    az = fh_z - fl_z

    # --- 3. Local neighborhood bounds q_max, q_min from (q, q_td) ---
    q_hi0 = torch.maximum(q, q_td)
    q_lo0 = torch.minimum(q, q_td)

    q_max = torch.maximum(
        q_hi0,
        torch.maximum(
            torch.maximum(torch.roll(q_hi0, 1, -1), torch.roll(q_hi0, -1, -1)),
            torch.maximum(torch.roll(q_hi0, 1, -2), torch.roll(q_hi0, -1, -2)),
        ),
    )
    q_min = torch.minimum(
        q_lo0,
        torch.minimum(
            torch.minimum(torch.roll(q_lo0, 1, -1), torch.roll(q_lo0, -1, -1)),
            torch.minimum(torch.roll(q_lo0, 1, -2), torch.roll(q_lo0, -1, -2)),
        ),
    )
    # Include vertical neighbors
    q_z_up = torch.cat([q_hi0[1:], q_hi0[-1:]], dim=0)
    q_z_dn = torch.cat([q_hi0[:1], q_hi0[:-1]], dim=0)
    q_max = torch.maximum(q_max, torch.maximum(q_z_up, q_z_dn))

    q_zlo_up = torch.cat([q_lo0[1:], q_lo0[-1:]], dim=0)
    q_zlo_dn = torch.cat([q_lo0[:1], q_lo0[:-1]], dim=0)
    q_min = torch.minimum(q_min, torch.minimum(q_zlo_up, q_zlo_dn))

    # --- 4. Zalesak P^+, P^-, Q^+, Q^-, R^+, R^- (SCALE-RM Eqs. A.27-A.34) ---
    ax_im1 = torch.roll(ax, shifts=1, dims=-1)
    ay_jm1 = torch.roll(ay, shifts=1, dims=-2)

    # Net incoming anti-diffusive mass per unit volume * dt
    p_plus = dt * (
        (torch.clamp(ax_im1, min=0.0) - torch.clamp(ax, max=0.0)) / dx
        + (torch.clamp(ay_jm1, min=0.0) - torch.clamp(ay, max=0.0)) / dy
        + (torch.clamp(az[:-1], min=0.0) - torch.clamp(az[1:], max=0.0)) / dz
    )
    # Net outgoing anti-diffusive mass per unit volume * dt
    p_minus = dt * (
        (torch.clamp(ax, min=0.0) - torch.clamp(ax_im1, max=0.0)) / dx
        + (torch.clamp(ay, min=0.0) - torch.clamp(ay_jm1, max=0.0)) / dy
        + (torch.clamp(az[1:], min=0.0) - torch.clamp(az[:-1], max=0.0)) / dz
    )

    q_plus = rho_safe * torch.clamp(q_max - q_td, min=0.0)
    q_minus = rho_safe * torch.clamp(q_td - q_min, min=0.0)

    r_plus = torch.where(p_plus > eps, torch.clamp(q_plus / (p_plus + eps), 0.0, 1.0), torch.ones_like(q_td))
    r_minus = torch.where(p_minus > eps, torch.clamp(q_minus / (p_minus + eps), 0.0, 1.0), torch.ones_like(q_td))

    # Face limiters C_x, C_y, C_z in [0, 1]
    rp_xp1 = torch.roll(r_plus, shifts=-1, dims=-1)
    rm_xp1 = torch.roll(r_minus, shifts=-1, dims=-1)
    cx = torch.where(ax >= 0.0, torch.minimum(r_minus, rp_xp1), torch.minimum(r_plus, rm_xp1))

    rp_yp1 = torch.roll(r_plus, shifts=-1, dims=-2)
    rm_yp1 = torch.roll(r_minus, shifts=-1, dims=-2)
    cy = torch.where(ay >= 0.0, torch.minimum(r_minus, rp_yp1), torch.minimum(r_plus, rm_yp1))

    cz = torch.zeros_like(az)
    cz[1:-1] = torch.where(
        az[1:-1] >= 0.0,
        torch.minimum(r_minus[:-1], r_plus[1:]),
        torch.minimum(r_plus[:-1], r_minus[1:]),
    )

    ax_lim = cx * ax
    ay_lim = cy * ay
    az_lim = cz * az

    div_anti = (
        (ax_lim - torch.roll(ax_lim, shifts=1, dims=-1)) / dx
        + (ay_lim - torch.roll(ay_lim, shifts=1, dims=-2)) / dy
        + (az_lim[1:] - az_lim[:-1]) / dz
    )
    q_final = torch.clamp(q_td - (dt / rho_safe) * div_anti, min=0.0)
    return q_final


# ---------------------------------------------------------------------------
# 3. 3D Compressible Nonhydrostatic RK3-HEVI Dynamical Step
# ---------------------------------------------------------------------------
def _compute_3d_sgs_tendencies(
    rho_d: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    theta_m: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    dz_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute anisotropic 3D Smagorinsky-Lilly SGS momentum & heat tendencies (SCALE-RM Sec. 8.1.2)."""
    nz = rho_d.shape[0]
    if can_use_crm_metal(rho_d):
        dz_w_int = dz_w if dz_w.shape[0] == nz - 1 else dz_w[1:-1]
        return dispatch_crm_sgs_metal(
            rho_d, u, v, w, theta_m, dx, dy, dz_c, dz_w_int, C.GRAVITY, 0.18
        )
    dz = dz_c.view(nz, -1, 1, 1)
    dz_int = dz_w.view(nz - 1, -1, 1, 1) if dz_w.shape[0] == nz - 1 else dz_w[1:-1].view(nz - 1, -1, 1, 1)

    # Velocity gradients at cell centers for strain-rate magnitude |S|
    du_dx = (u - torch.roll(u, 1, -1)) / dx
    dv_dy = (v - torch.roll(v, 1, -2)) / dy
    dw_dz = (w[1:] - w[:-1]) / dz

    u_c = 0.5 * (u + torch.roll(u, 1, -1))
    v_c = 0.5 * (v + torch.roll(v, 1, -2))
    du_dy = 0.5 * (torch.roll(u_c, -1, -2) - torch.roll(u_c, 1, -2)) / dy
    dv_dx = 0.5 * (torch.roll(v_c, -1, -1) - torch.roll(v_c, 1, -1)) / dx

    du_dz_int = (u_c[1:] - u_c[:-1]) / dz_int
    dv_dz_int = (v_c[1:] - v_c[:-1]) / dz_int
    du_dz = torch.zeros_like(u_c)
    dv_dz = torch.zeros_like(v_c)
    du_dz[:-1] += 0.5 * du_dz_int
    du_dz[1:] += 0.5 * du_dz_int
    dv_dz[:-1] += 0.5 * dv_dz_int
    dv_dz[1:] += 0.5 * dv_dz_int

    s2 = (
        2.0 * (du_dx * du_dx + dv_dy * dv_dy + dw_dz * dw_dz)
        + (du_dy + dv_dx) ** 2
        + du_dz * du_dz
        + dv_dz * dv_dz
    )
    strain_mag = torch.sqrt(torch.clamp(s2, min=1.0e-12))

    dth_dz_int = (theta_m[1:] - theta_m[:-1]) / dz_int
    n2_int = (C.GRAVITY / torch.clamp(0.5 * (theta_m[1:] + theta_m[:-1]), min=200.0)) * dth_dz_int
    n2 = torch.zeros_like(theta_m)
    n2[:-1] += 0.5 * n2_int
    n2[1:] += 0.5 * n2_int

    nu_sgs, d_sgs = compute_smagorinsky_lilly_3d(strain_mag, n2, dx, dy, dz)

    # Horizontal eddy flux divergences (vertical PBL turbulence is handled implicitly by MYNN 2.5)
    lap_u = (torch.roll(u, -1, -1) - 2.0 * u + torch.roll(u, 1, -1)) / (dx * dx) + (
        torch.roll(u, -1, -2) - 2.0 * u + torch.roll(u, 1, -2)
    ) / (dy * dy)
    lap_v = (torch.roll(v, -1, -1) - 2.0 * v + torch.roll(v, 1, -1)) / (dx * dx) + (
        torch.roll(v, -1, -2) - 2.0 * v + torch.roll(v, 1, -2)
    ) / (dy * dy)

    w_lap = (torch.roll(w, -1, -1) - 2.0 * w + torch.roll(w, 1, -1)) / (dx * dx) + (
        torch.roll(w, -1, -2) - 2.0 * w + torch.roll(w, 1, -2)
    ) / (dy * dy)
    nu_w = torch.zeros_like(w)
    nu_w[1:-1] = 0.5 * (nu_sgs[:-1] + nu_sgs[1:])

    # Conservative horizontal scalar SGS flux for Theta_m
    d_x = 0.5 * (d_sgs + torch.roll(d_sgs, -1, -1))
    d_y = 0.5 * (d_sgs + torch.roll(d_sgs, -1, -2))
    rho_x = 0.5 * (rho_d + torch.roll(rho_d, -1, -1))
    rho_y = 0.5 * (rho_d + torch.roll(rho_d, -1, -2))
    fth_x = -rho_x * d_x * (torch.roll(theta_m, -1, -1) - theta_m) / dx
    fth_y = -rho_y * d_y * (torch.roll(theta_m, -1, -2) - theta_m) / dy
    d_Thm_sgs = -(fth_x - torch.roll(fth_x, 1, -1)) / dx - (fth_y - torch.roll(fth_y, 1, -2)) / dy

    # Add small 4th-order background horizontal numerical filter tau_div to control 2*dx grid noise
    nu_bg = 0.004 * (dx * dy) / 10.0  # ~640 m^2 s^-1 at dx = 4 km
    du_sgs = (nu_sgs + nu_bg) * lap_u
    dv_sgs = (nu_sgs + nu_bg) * lap_v
    dw_sgs = torch.zeros_like(w)
    dw_sgs[1:-1] = (nu_w[1:-1] + nu_bg) * w_lap[1:-1]

    return du_sgs, dv_sgs, dw_sgs, d_Thm_sgs


def _ws3_horizontal_advect(
    q: torch.Tensor,
    u_left: torch.Tensor,
    v_bot: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """3rd-order Wicker-Skamarock (2002) upwind-biased horizontal momentum advection.

    Evaluated in cancellation-free flux-difference form on the control-volume faces
    surrounding `q` (`u_left` on the `-x` face and `v_bot` on the `-y` face), avoiding
    finite-time shock blowup of non-conservative 2nd-order centered `-u du/dx`.
    """
    qx_m2 = torch.roll(q, 2, -1)
    qx_m1 = torch.roll(q, 1, -1)
    qx_p1 = torch.roll(q, -1, -1)
    q_x_left = (
        (-qx_m2 + 7.0 * qx_m1 + 7.0 * q - qx_p1)
        + torch.sign(u_left) * (qx_p1 - 3.0 * q + 3.0 * qx_m1 - qx_m2)
    ) / 12.0
    q_x_right = torch.roll(q_x_left, -1, -1)
    u_right = torch.roll(u_left, -1, -1)

    qy_m2 = torch.roll(q, 2, -2)
    qy_m1 = torch.roll(q, 1, -2)
    qy_p1 = torch.roll(q, -1, -2)
    q_y_bot = (
        (-qy_m2 + 7.0 * qy_m1 + 7.0 * q - qy_p1)
        + torch.sign(v_bot) * (qy_p1 - 3.0 * q + 3.0 * qy_m1 - qy_m2)
    ) / 12.0
    q_y_top = torch.roll(q_y_bot, -1, -2)
    v_top = torch.roll(v_bot, -1, -2)

    return (
        -(u_right * (q_x_right - q) - u_left * (q_x_left - q)) / dx
        - (v_top * (q_y_top - q) - v_bot * (q_y_bot - q)) / dy
    )


def _upwind_vertical_advect(
    q: torch.Tensor,
    w_int: torch.Tensor,
    dz_w_int: torch.Tensor,
) -> torch.Tensor:
    """First-order upwind vertical advection tendency `-w * dq/dz` on stretched interfaces."""
    dq_dz_int = (q[1:] - q[:-1]) / dz_w_int
    out = torch.zeros_like(q)
    out[1:] += -torch.clamp(w_int, min=0.0) * dq_dz_int
    out[:-1] += -torch.clamp(w_int, max=0.0) * dq_dz_int
    return out


def _crm_fast_acoustic_hevi_torch(
    n_ac: int,
    dt_ac: float,
    beta_hevi: float,
    gravity: float,
    dx: float,
    dy: float,
    accum_fluxes: bool,
    rho0: torch.Tensor,
    Thm0: torch.Tensor,
    u0: torch.Tensor,
    v0: torch.Tensor,
    w0: torch.Tensor,
    rho_x: torch.Tensor,
    rho_y: torch.Tensor,
    rtot_x_inv: torch.Tensor,
    rtot_y_inv: torch.Tensor,
    thm_x_eff: torch.Tensor,
    thm_y_eff: torch.Tensor,
    pres_s: torch.Tensor,
    dp_dThm_s: torch.Tensor,
    Thm_s: torch.Tensor,
    q_load: torch.Tensor,
    tend_u_slow: torch.Tensor,
    tend_v_slow: torch.Tensor,
    tend_w_slow: torch.Tensor,
    tend_Thm_slow: torch.Tensor,
    rho_dw: torch.Tensor,
    rho_tot_w: torch.Tensor,
    thm_w: torch.Tensor,
    dl: torch.Tensor,
    diag: torch.Tensor,
    du: torch.Tensor,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    sponge_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled split-explicit CRM horizontal acoustic + vertical HEVI inner loop."""
    rho_k = rho0
    Thm_k = Thm0
    u_k = u0
    v_k = v0
    w_k = w0

    fx_acc = torch.zeros_like(rho0)
    fy_acc = torch.zeros_like(rho0)
    fz_acc = torch.zeros_like(w0)
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    rho_dw_safe = torch.clamp(rho_dw, min=1.0e-5)
    rho_tot_w_safe = torch.clamp(rho_tot_w, min=1.0e-5)
    K_tri = diag.shape[0]

    for _ in range(n_ac):
        p_k = pres_s + dp_dThm_s * (Thm_k - Thm_s)
        pgf_u = -(torch.roll(p_k, -1, -1) - p_k) * inv_dx * rtot_x_inv
        pgf_v = -(torch.roll(p_k, -1, -2) - p_k) * inv_dy * rtot_y_inv

        u_next = u_k + dt_ac * (tend_u_slow + pgf_u)
        v_next = v_k + dt_ac * (tend_v_slow + pgf_v)

        fx_k = rho_x * (1.10 * u_next - 0.10 * u_k)
        fy_k = rho_y * (1.10 * v_next - 0.10 * v_k)
        fTh_x = fx_k * thm_x_eff
        fTh_y = fy_k * thm_y_eff

        div_h_rho = (fx_k - torch.roll(fx_k, 1, -1)) * inv_dx + (fy_k - torch.roll(fy_k, 1, -2)) * inv_dy
        div_h_Thm = (fTh_x - torch.roll(fTh_x, 1, -1)) * inv_dx + (fTh_y - torch.roll(fTh_y, 1, -2)) * inv_dy

        fz_old = torch.zeros_like(w_k)
        fz_old[1:-1] = rho_dw * w_k[1:-1]
        fTh_z_old = torch.zeros_like(w_k)
        fTh_z_old[1:-1] = thm_w * fz_old[1:-1]

        rho_dag = rho_k - dt_ac * (div_h_rho + (fz_old[1:] - fz_old[:-1]) / dz_c)
        Thm_dag = Thm_k + dt_ac * (-div_h_Thm - (fTh_z_old[1:] - fTh_z_old[:-1]) / dz_c + tend_Thm_slow)

        p_dag = pres_s + dp_dThm_s * (Thm_dag - Thm_s)
        p_blend = (1.0 - beta_hevi) * p_k + beta_hevi * p_dag
        rho_blend_w_t = 0.5 * (
            ((1.0 - beta_hevi) * rho_k[:-1] + beta_hevi * rho_dag[:-1]) * q_load[:-1]
            + ((1.0 - beta_hevi) * rho_k[1:] + beta_hevi * rho_dag[1:]) * q_load[1:]
        )
        nh_imb = (p_blend[1:] - p_blend[:-1]) / dz_w_int + rho_blend_w_t * gravity

        rhs_fz = dt_ac * (
            rho_dw * tend_w_slow
            - (rho_dw / rho_tot_w_safe) * nh_imb
            - sponge_w * fz_old[1:-1]
        )

        # Inlined batched Thomas solve along dim=0
        c_list: list[torch.Tensor] = []
        d_list: list[torch.Tensor] = []
        inv_b0 = 1.0 / diag[0]
        c_prev = du[0] * inv_b0
        d_prev = rhs_fz[0] * inv_b0
        c_list.append(c_prev)
        d_list.append(d_prev)
        for kk in range(1, K_tri):
            inv_den = 1.0 / (diag[kk] - dl[kk] * c_prev)
            c_prev = du[kk] * inv_den if kk < K_tri - 1 else torch.zeros_like(d_prev)
            d_prev = (rhs_fz[kk] - dl[kk] * d_prev) * inv_den
            c_list.append(c_prev)
            d_list.append(d_prev)
        x_list: list[torch.Tensor] = [d_prev] * K_tri
        x_nx = d_list[K_tri - 1]
        x_list[K_tri - 1] = x_nx
        for kk in range(K_tri - 2, -1, -1):
            x_nx = d_list[kk] - c_list[kk] * x_nx
            x_list[kk] = x_nx
        dfz = torch.stack(x_list, dim=0)

        fz_new = fz_old.clone()
        fz_new[1:-1] = fz_old[1:-1] + dfz

        rho_next = torch.clamp(
            rho_dag - beta_hevi * dt_ac * (fz_new[1:] - fz_old[1:] - (fz_new[:-1] - fz_old[:-1])) / dz_c,
            min=1.0e-5,
        )
        dfTh_z = torch.zeros_like(w_k)
        dfTh_z[1:-1] = thm_w * dfz
        Thm_next = Thm_dag - beta_hevi * dt_ac * (dfTh_z[1:] - dfTh_z[:-1]) / dz_c

        w_next = torch.zeros_like(w_k)
        w_next[1:-1] = fz_new[1:-1] / rho_dw_safe

        if accum_fluxes:
            fx_acc = fx_acc + fx_k
            fy_acc = fy_acc + fy_k
            fz_acc = fz_acc + ((1.0 - beta_hevi) * fz_old + beta_hevi * fz_new)

        rho_k = rho_next
        Thm_k = Thm_next
        u_k = u_next
        v_k = v_next
        w_k = w_next

    return rho_k, Thm_k, u_k, v_k, w_k, fx_acc, fy_acc, fz_acc


def step_crm_rk3_hevi(
    crm: CRMState,
    vgrid: VerticalGrid,
    dt_dyn: float,
    beta_hevi: float = 0.60,
    z_c_col: torch.Tensor | None = None,
    dz_c_col: torch.Tensor | None = None,
    *,
    Thm_in: torch.Tensor | None = None,
    sgs_in: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    update_theta_m: bool = True,
    thm_out: list[torch.Tensor] | None = None,
    clone_out: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance `(rho_d, u, v, w, theta_m)` over one Split-Explicit RK3-HEVI step `dt_dyn` [s].

    Slow advection, Coriolis, and 3D Smagorinsky-Lilly SGS tendencies are evaluated once per
    RK3 stage (`dt_dyn`), while horizontal sound waves and implicit vertical HEVI oscillations are
    sub-stepped at the acoustic Courant limit (`0.70 * dx / 350` for 2D C-grid where the von Neumann
    limit is `1.0`, `0.45 * min(dx, dy) / 350` for 3D; Skamarock & Klemp, 1992). On Apple Silicon
    (`mps`), the entire fast acoustic+HEVI loop dispatches to fused Metal compute kernels.

    Returns the stage-averaged mass fluxes `(fx_avg, fy_avg, fz_avg)` matching the
    continuity change `(crm.rho_d_new - crm.rho_d_old)` to machine precision for
    subsequent Zalesak 3D FCT tracer advection (`Skamarock et al., 2012`, Sec. 3c).
    """
    nz, _, ny, nx = crm.rho_d.shape
    dx = crm.dx
    dy = crm.dy
    dz_c = dz_c_col.view(nz, -1, 1, 1) if dz_c_col is not None else vgrid.dzeta_c.view(nz, 1, 1, 1)
    dz_w_int = (
        (z_c_col[1:] - z_c_col[:-1]).view(nz - 1, -1, 1, 1)
        if z_c_col is not None
        else (vgrid.zeta_c[1:] - vgrid.zeta_c[:-1]).view(nz - 1, 1, 1, 1)
    )
    sponge_w = vgrid.sponge_w[1:-1].view(nz - 1, 1, 1, 1)
    sponge_c = vgrid.sponge_c.view(nz, 1, 1, 1)

    rho0 = crm.rho_d
    u0 = crm.u
    v0 = crm.v
    w0 = crm.w
    Thm0 = Thm_in if Thm_in is not None else (rho0 * crm.theta_m)

    # Compute SGS tendencies once at start of RK3 step (or reuse caller-cached tendencies)
    if sgs_in is not None:
        du_sgs, dv_sgs, dw_sgs, dThm_sgs = sgs_in
    else:
        thm_for_sgs = (Thm0 / torch.clamp(rho0, min=1.0e-5)) if (Thm_in is not None and not update_theta_m) else crm.theta_m
        du_sgs, dv_sgs, dw_sgs, dThm_sgs = _compute_3d_sgs_tendencies(
            rho0, u0, v0, w0, thm_for_sgs, dx, dy, dz_c, dz_w_int
        )

    # Dimension-aware Coriolis handling for embedded periodic CRMs (Khairoutdinov et al., 2005; Yamazaki & Miura, 2024):
    # - 1D (1x1) & 2D (nx==1 or ny==1): disable CRM Coriolis (no cross-plane background PGF; host handles rotation).
    # - 3D (nx>1 and ny>1): perturbation-only Coriolis on u' = u - u_mean evaluated inside each RK3 stage.
    use_3d_pert_cor = (nx > 1) and (ny > 1) and (crm.f_coriolis is not None)
    f_col_3d = crm.f_coriolis.view(1, -1, 1, 1) if use_3d_pert_cor else None
    if use_3d_pert_cor and can_use_crm_metal(rho0):
        # Metal fast path uses stage-frozen slow tendency; perturbation Coriolis has |f*dt_sub| < 0.008 with SGS damping
        u_pert0 = u0 - mean_2d(u0, keepdim=True)
        v_pert0 = v0 - mean_2d(v0, keepdim=True)
        du_sgs = du_sgs + f_col_3d * v_pert0
        dv_sgs = dv_sgs - f_col_3d * u_pert0

    is_2d_crm = (ny == 1) ^ (nx == 1)
    cfl_ac = 0.70 if is_2d_crm else 0.45
    dt_ac_max = cfl_ac * min(dx, dy) / 350.0 if (ny > 1 or nx > 1) else dt_dyn
    if can_use_crm_metal(rho0):
        rho_s, Thm_s, u_s, v_s, w_s, fx_final, fy_final, fz_final = run_crm_rk3_hevi_metal(
            dt_dyn, beta_hevi, dt_ac_max,
            C.P_REF, C.R_D, C.GAMMA_D, C.GRAVITY, dx, dy,
            rho0, Thm0, u0, v0, w0, None,
            du_sgs, dv_sgs, dw_sgs, dThm_sgs,
            dz_c, dz_w_int, sponge_c, sponge_w, None,
            q_n=crm.q_n, q_p=crm.q_p,
            clone_out=clone_out,
        )
        crm.rho_d = rho_s
        crm.u = u_s
        crm.v = v_s
        crm.w = w_s
        if thm_out is not None:
            thm_out.clear()
            thm_out.append(Thm_s)
        if update_theta_m:
            crm.theta_m = Thm_s / crm.rho_d
        return fx_final, fy_final, fz_final

    q_load = 1.0 + crm.q_n + crm.q_p  # Exact moist mass loading rho_tot / rho_d

    rho_s = rho0
    u_s = u0
    v_s = v0
    w_s = w0
    Thm_s = Thm0

    # Wicker & Skamarock (2002) 3-stage RK3 sub-steps: dtau = (dt/3, dt/2, dt)
    rk_taus = (dt_dyn / 3.0, 0.5 * dt_dyn, dt_dyn)
    fx_final = torch.zeros_like(rho0)
    fy_final = torch.zeros_like(rho0)
    fz_final = torch.zeros_like(w0)

    gamma = C.GAMMA_D
    f_cor = None  # Handled above via dimension-aware 1D/2D=0 or 3D perturbation-only Coriolis

    for stage_idx, dtau in enumerate(rk_taus):
        thm_s = Thm_s / torch.clamp(rho_s, min=1.0e-5)
        pres_s = C.P_REF * torch.pow(
            torch.clamp(C.R_D * Thm_s / C.P_REF, min=1.0e-7),
            gamma,
        )
        dp_dThm_s = gamma * pres_s / torch.clamp(Thm_s, min=1.0e-3)
        rho_tot_s = rho_s * q_load

        # Stage-frozen edge densities & effective flux-form potential temperatures
        rho_x = 0.5 * (rho_s + torch.roll(rho_s, -1, -1))
        rho_y = 0.5 * (rho_s + torch.roll(rho_s, -1, -2))
        rtot_x_inv = 1.0 / torch.clamp(0.5 * (rho_tot_s + torch.roll(rho_tot_s, -1, -1)), min=1.0e-5)
        rtot_y_inv = 1.0 / torch.clamp(0.5 * (rho_tot_s + torch.roll(rho_tot_s, -1, -2)), min=1.0e-5)

        thm_xm1 = torch.roll(thm_s, 1, -1)
        thm_xp1 = torch.roll(thm_s, -1, -1)
        thm_xp2 = torch.roll(thm_s, -2, -1)
        thm_x_eff = (
            (-thm_xm1 + 7.0 * thm_s + 7.0 * thm_xp1 - thm_xp2)
            + 0.25 * torch.sign(u_s) * (thm_xp2 - 3.0 * thm_xp1 + 3.0 * thm_s - thm_xm1)
        ) / 12.0

        thm_ym1 = torch.roll(thm_s, 1, -2)
        thm_yp1 = torch.roll(thm_s, -1, -2)
        thm_yp2 = torch.roll(thm_s, -2, -2)
        thm_y_eff = (
            (-thm_ym1 + 7.0 * thm_s + 7.0 * thm_yp1 - thm_yp2)
            + 0.25 * torch.sign(v_s) * (thm_yp2 - 3.0 * thm_yp1 + 3.0 * thm_s - thm_ym1)
        ) / 12.0

        # Stage-frozen Slow Tendencies (WS3 horizontal + upwind vertical momentum advection, Coriolis, SGS, Rayleigh sponge)
        u_s_im1 = torch.roll(u_s, 1, -1)
        v_s_jm1 = torch.roll(v_s, 1, -2)
        u_c = 0.5 * (u_s + u_s_im1)
        v_c = 0.5 * (v_s + v_s_jm1)
        v_u_bot = 0.5 * (v_s_jm1 + torch.roll(v_s_jm1, -1, -1))
        u_v_left = 0.5 * (u_s_im1 + torch.roll(u_s_im1, -1, -2))

        adv_u = _ws3_horizontal_advect(u_s, u_c, v_u_bot, dx, dy)
        adv_v = _ws3_horizontal_advect(v_s, u_v_left, v_c, dx, dy)

        w_ux = 0.5 * (w_s[1:-1] + torch.roll(w_s[1:-1], -1, -1))
        w_vy = 0.5 * (w_s[1:-1] + torch.roll(w_s[1:-1], -1, -2))
        vadv_u = _upwind_vertical_advect(u_s, w_ux, dz_w_int)
        vadv_v = _upwind_vertical_advect(v_s, w_vy, dz_w_int)

        if use_3d_pert_cor and f_col_3d is not None:
            v_c_pert = v_c - mean_2d(v_c, keepdim=True)
            u_c_pert = u_c - mean_2d(u_c, keepdim=True)
            cor_u = f_col_3d * 0.5 * (v_c_pert + torch.roll(v_c_pert, -1, -1))
            cor_v = -f_col_3d * 0.5 * (u_c_pert + torch.roll(u_c_pert, -1, -2))
        else:
            cor_u = 0.0
            cor_v = 0.0

        u_anom = u_s - mean_2d(u_s, keepdim=True)
        v_anom = v_s - mean_2d(v_s, keepdim=True)

        tend_u_slow = adv_u + vadv_u + cor_u + du_sgs - sponge_c * u_anom
        tend_v_slow = adv_v + vadv_v + cor_v + dv_sgs - sponge_c * v_anom

        w_int = w_s[1:-1]
        u_w_left = 0.5 * (u_s_im1[:-1] + u_s_im1[1:])
        v_w_bot = 0.5 * (v_s_jm1[:-1] + v_s_jm1[1:])
        adv_w_h = _ws3_horizontal_advect(w_int, u_w_left, v_w_bot, dx, dy)
        dw_dz_dn = (w_int - w_s[:-2]) / dz_c[:-1]
        dw_dz_up = (w_s[2:] - w_int) / dz_c[1:]
        adv_w_v = -(torch.clamp(w_int, min=0.0) * dw_dz_dn + torch.clamp(w_int, max=0.0) * dw_dz_up)
        tend_w_slow = adv_w_h + adv_w_v + dw_sgs[1:-1]
        tend_Thm_slow = dThm_sgs

        # Acoustic substep count & stage-frozen Vertical HEVI Helmholtz tridiagonal matrix
        n_ac = max(1, int(math.ceil(dtau / max(dt_ac_max, 0.1))))
        dt_ac = dtau / float(n_ac)

        rho_dw = 0.5 * (rho_s[:-1] + rho_s[1:])
        rho_tot_w = 0.5 * (rho_tot_s[:-1] + rho_tot_s[1:])
        thm_w = 0.5 * (thm_s[:-1] + thm_s[1:])

        fac_p = ((beta_hevi * dt_ac) ** 2) * (rho_dw / (rho_tot_w * dz_w_int))
        fac_g = ((beta_hevi * dt_ac) ** 2) * (0.5 * C.GRAVITY * (rho_dw / rho_tot_w))
        c_dn = dp_dThm_s[:-1] / dz_c[:-1]
        c_up = dp_dThm_s[1:] / dz_c[1:]

        diag = torch.clamp(
            1.0
            + dt_ac * sponge_w
            + fac_p * thm_w * (c_up + c_dn)
            + fac_g * (q_load[1:] / dz_c[1:] - q_load[:-1] / dz_c[:-1]),
            min=0.5,
        )
        dl = torch.zeros_like(diag)
        du = torch.zeros_like(diag)
        dl[1:] = -fac_p[1:] * c_dn[1:] * thm_w[:-1] + fac_g[1:] * (q_load[1:-1] / dz_c[1:-1])
        du[:-1] = -fac_p[:-1] * c_up[:-1] * thm_w[1:] - fac_g[:-1] * (q_load[1:-1] / dz_c[1:-1])

        accum_stage = stage_idx == 2
        rho_s, Thm_s, u_s, v_s, w_s, fx_acc, fy_acc, fz_acc = _crm_fast_acoustic_hevi_torch(
            n_ac, dt_ac, beta_hevi, C.GRAVITY, dx, dy, accum_stage,
            rho0, Thm0, u0, v0, w0,
            rho_x, rho_y, rtot_x_inv, rtot_y_inv, thm_x_eff, thm_y_eff,
            pres_s, dp_dThm_s, Thm_s, q_load,
            tend_u_slow, tend_v_slow, tend_w_slow, tend_Thm_slow,
            rho_dw, rho_tot_w, thm_w, dl, diag, du,
            dz_c, dz_w_int, sponge_w,
        )

        if accum_stage:
            inv_nac = 1.0 / float(n_ac)
            fx_final = fx_acc * inv_nac
            fy_final = fy_acc * inv_nac
            fz_final = fz_acc * inv_nac

    crm.rho_d = rho_s
    crm.u = u_s
    crm.v = v_s
    crm.w = w_s
    if thm_out is not None:
        thm_out.clear()
        thm_out.append(Thm_s)
    if update_theta_m:
        crm.theta_m = Thm_s / crm.rho_d

    return fx_final, fy_final, fz_final


def step_crm_physics_and_dynamics(
    crm: CRMState,
    vgrid: VerticalGrid,
    dt_crm: float,
    n_dyn_substeps: int = 1,
    u_host_target: torch.Tensor | None = None,
    v_host_target: torch.Tensor | None = None,
    tau_sp_mom: float = 600.0,
    dtheta_m_ls_dt: torch.Tensor | None = None,
    dq_n_ls_dt: torch.Tensor | None = None,
    sfc_fluxes: SurfaceFluxDiagnostics | None = None,
    rad_heating_k_s: torch.Tensor | None = None,
    z_c_col: torch.Tensor | None = None,
    z_w_col: torch.Tensor | None = None,
    dz_c_col: torch.Tensor | None = None,
    microphysics_scheme: str = "one_moment",
    is_ocean_sub: torch.Tensor | None = None,
    u_sfc_sub: torch.Tensor | None = None,
    q_dust_sub: torch.Tensor | None = None,
) -> SPTendencies:
    """Advance 3D CRM by `dt_crm` [s] (RK3-HEVI + Zalesak FCT + MYNN 2.5 + SAM Microphysics).

    Returns horizontal-mean Superparameterization convective/microphysical/turbulent tendencies
    for the host SCVT grid (`Khairoutdinov et al., 2005`; `Yamazaki & Miura, 2024`).
    """
    nz, n_col, ny, nx = crm.rho_d.shape
    m_sub = n_col * ny * nx
    mp_key = str(microphysics_scheme).strip().lower()
    use_6c_mp = mp_key == "six_class_2m"
    use_2m_mp = mp_key == "two_moment"
    if use_6c_mp:
        from aoba.microphysics_6class import init_six_class_hydrometeors

        init_six_class_hydrometeors(crm)
    if use_2m_mp:
        if crm.n_c is None:
            crm.n_c = torch.zeros_like(crm.q_c)
        if crm.n_i is None:
            crm.n_i = torch.zeros_like(crm.q_i)

    if can_use_cuda_graph(crm.rho_d.device) and can_use_crm_metal(crm.rho_d):
        _opt_crm_names = ("f_coriolis", "n_c", "n_i", "q_r", "q_s", "q_g", "n_r", "n_s", "n_g")
        _core_crm_names = ("rho_d", "u", "v", "w", "theta_m", "q_n", "q_p", "q_v", "q_c", "q_i", "qke")
        _opt_crm_present = tuple(k for k in _opt_crm_names if getattr(crm, k, None) is not None)
        _opt_arg_pairs = (
            ("u_host_target", u_host_target),
            ("v_host_target", v_host_target),
            ("dtheta_m_ls_dt", dtheta_m_ls_dt),
            ("dq_n_ls_dt", dq_n_ls_dt),
            ("rad_heating_k_s", rad_heating_k_s),
            ("z_c_col", z_c_col),
            ("z_w_col", z_w_col),
            ("dz_c_col", dz_c_col),
            ("is_ocean_sub", is_ocean_sub),
            ("u_sfc_sub", u_sfc_sub),
            ("q_dust_sub", q_dust_sub),
        )
        _opt_arg_present = tuple(k for k, v in _opt_arg_pairs if v is not None)
        has_sfc = sfc_fluxes is not None

        live_in: list[torch.Tensor] = [getattr(crm, k) for k in _core_crm_names]
        live_in.extend(getattr(crm, k) for k in _opt_crm_present)
        live_in.extend(v for _, v in _opt_arg_pairs if v is not None)
        if has_sfc:
            assert sfc_fluxes is not None
            live_in.extend([
                sfc_fluxes.u_star, sfc_fluxes.buoy_flux, sfc_fluxes.tau_x,
                sfc_fluxes.tau_y, sfc_fluxes.shf, sfc_fluxes.evap,
            ])

        key = (
            "crm_step_3d",
            id(vgrid),
            tuple(tuple(t.shape) for t in live_in),
            tuple(t.dtype for t in live_in),
            _opt_crm_present,
            _opt_arg_present,
            has_sfc,
            float(dt_crm), int(n_dyn_substeps), float(tau_sp_mom), mp_key,
            float(crm.dx), float(crm.dy),
        )
        n_core = len(_core_crm_names)
        n_opt_c = len(_opt_crm_present)

        def _crm_body(ins: list[torch.Tensor]) -> list[torch.Tensor]:
            pos = 0
            c_kwargs: dict[str, object] = {}
            for k in _core_crm_names:
                c_kwargs[k] = ins[pos]
                pos += 1
            for k in _opt_crm_present:
                c_kwargs[k] = ins[pos]
                pos += 1
            tmp_crm = CRMState(dx=crm.dx, dy=crm.dy, **c_kwargs)  # type: ignore[arg-type]
            a_map: dict[str, torch.Tensor | None] = {k: None for k, _ in _opt_arg_pairs}
            for k in _opt_arg_present:
                a_map[k] = ins[pos]
                pos += 1
            sfc_in = (
                SimpleNamespace(
                    u_star=ins[pos], buoy_flux=ins[pos + 1], tau_x=ins[pos + 2],
                    tau_y=ins[pos + 3], shf=ins[pos + 4], evap=ins[pos + 5],
                )
                if has_sfc
                else None
            )
            sp_res = _step_crm_physics_and_dynamics_impl(
                crm=tmp_crm, vgrid=vgrid, dt_crm=dt_crm, n_dyn_substeps=n_dyn_substeps,
                u_host_target=a_map["u_host_target"], v_host_target=a_map["v_host_target"],
                tau_sp_mom=tau_sp_mom,
                dtheta_m_ls_dt=a_map["dtheta_m_ls_dt"], dq_n_ls_dt=a_map["dq_n_ls_dt"],
                sfc_fluxes=sfc_in,  # type: ignore[arg-type]
                rad_heating_k_s=a_map["rad_heating_k_s"],
                z_c_col=a_map["z_c_col"], z_w_col=a_map["z_w_col"], dz_c_col=a_map["dz_c_col"],
                microphysics_scheme=microphysics_scheme,
                is_ocean_sub=a_map["is_ocean_sub"], u_sfc_sub=a_map["u_sfc_sub"],
                q_dust_sub=a_map["q_dust_sub"],
            )
            out_list: list[torch.Tensor] = [getattr(tmp_crm, k) for k in _core_crm_names]
            out_list.extend(getattr(tmp_crm, k) for k in _opt_crm_present if k != "f_coriolis")
            out_list.extend([
                sp_res.dtheta_m_dt, sp_res.dq_n_dt, sp_res.dq_p_dt,
                sp_res.du_dt, sp_res.dv_dt, sp_res.precip_rain, sp_res.precip_snow,
            ])
            if sp_res.precip_rain_sub is not None and sp_res.precip_snow_sub is not None:
                out_list.extend([sp_res.precip_rain_sub, sp_res.precip_snow_sub])
            return out_list

        outs = run_cuda_graph_cached(key, live_in, _crm_body, warmup_calls=2, clone_outputs=True)
        pos = 0
        for k in _core_crm_names:
            setattr(crm, k, outs[pos])
            pos += 1
        for k in _opt_crm_present:
            if k != "f_coriolis":
                setattr(crm, k, outs[pos])
                pos += 1
        has_sub_pr = len(outs) > pos + 7
        return SPTendencies(
            dtheta_m_dt=outs[pos],
            dq_n_dt=outs[pos + 1],
            dq_p_dt=outs[pos + 2],
            du_dt=outs[pos + 3],
            dv_dt=outs[pos + 4],
            precip_rain=outs[pos + 5],
            precip_snow=outs[pos + 6],
            precip_rain_sub=(outs[pos + 7] if has_sub_pr else None),
            precip_snow_sub=(outs[pos + 8] if has_sub_pr else None),
        )

    return _step_crm_physics_and_dynamics_impl(
        crm=crm, vgrid=vgrid, dt_crm=dt_crm, n_dyn_substeps=n_dyn_substeps,
        u_host_target=u_host_target, v_host_target=v_host_target, tau_sp_mom=tau_sp_mom,
        dtheta_m_ls_dt=dtheta_m_ls_dt, dq_n_ls_dt=dq_n_ls_dt, sfc_fluxes=sfc_fluxes,
        rad_heating_k_s=rad_heating_k_s, z_c_col=z_c_col, z_w_col=z_w_col, dz_c_col=dz_c_col,
        microphysics_scheme=microphysics_scheme, is_ocean_sub=is_ocean_sub,
        u_sfc_sub=u_sfc_sub, q_dust_sub=q_dust_sub,
    )


def _step_crm_physics_and_dynamics_impl(
    crm: CRMState,
    vgrid: VerticalGrid,
    dt_crm: float,
    n_dyn_substeps: int = 1,
    u_host_target: torch.Tensor | None = None,
    v_host_target: torch.Tensor | None = None,
    tau_sp_mom: float = 600.0,
    dtheta_m_ls_dt: torch.Tensor | None = None,
    dq_n_ls_dt: torch.Tensor | None = None,
    sfc_fluxes: SurfaceFluxDiagnostics | None = None,
    rad_heating_k_s: torch.Tensor | None = None,
    z_c_col: torch.Tensor | None = None,
    z_w_col: torch.Tensor | None = None,
    dz_c_col: torch.Tensor | None = None,
    microphysics_scheme: str = "one_moment",
    is_ocean_sub: torch.Tensor | None = None,
    u_sfc_sub: torch.Tensor | None = None,
    q_dust_sub: torch.Tensor | None = None,
) -> SPTendencies:
    nz, n_col, ny, nx = crm.rho_d.shape
    m_sub = n_col * ny * nx
    mp_key = str(microphysics_scheme).strip().lower()
    use_6c_mp = mp_key == "six_class_2m"
    use_2m_mp = mp_key == "two_moment"

    # Record initial horizontal-mean profiles before CRM physics + subgrid transport
    thm_mean_0 = mean_2d(crm.theta_m)
    qn_mean_0 = mean_2d(crm.q_n)
    qp_mean_0 = mean_2d(crm.q_p)
    u_mean_0 = mean_2d(crm.u)
    v_mean_0 = mean_2d(crm.v)

    # 1. Apply optional large-scale host forcing & radiative heating
    if dtheta_m_ls_dt is not None:
        crm.theta_m = crm.theta_m + dt_crm * dtheta_m_ls_dt.view(nz, n_col, 1, 1)
    if dq_n_ls_dt is not None:
        crm.q_n = torch.clamp(crm.q_n + dt_crm * dq_n_ls_dt.view(nz, n_col, 1, 1), min=0.0)
        q_cond_ls = torch.minimum(crm.q_c + crm.q_i, crm.q_n)
        crm.q_v = torch.clamp(crm.q_n - q_cond_ls, min=0.0)
    if rad_heating_k_s is not None:
        # Convert dT/dt [K s^-1] to d(theta_m)/dt via (1 + RV_OVER_RD * q_v) / Exner
        _, exner_rad, _ = diagnose_pressure_temperature(crm.rho_d, crm.theta_m, crm.q_v)
        q_fac = 1.0 + C.RV_OVER_RD * crm.q_v
        rad_3d = rad_heating_k_s.view(nz, n_col, 1, 1) if rad_heating_k_s.ndim == 2 else rad_heating_k_s
        crm.theta_m = crm.theta_m + dt_crm * rad_3d * q_fac / exner_rad

    # Relax horizontal-mean CRM momentum toward host velocity if coupled in SP mode
    if u_host_target is not None and v_host_target is not None:
        alpha_m = min(dt_crm / max(tau_sp_mom, dt_crm), 1.0)
        du_relax = alpha_m * (u_host_target.view(nz, n_col, 1, 1) - mean_2d(crm.u, keepdim=True))
        dv_relax = alpha_m * (v_host_target.view(nz, n_col, 1, 1) - mean_2d(crm.v, keepdim=True))
        crm.u = crm.u + du_relax
        crm.v = crm.v + dv_relax

    # 2. Acoustic/nonhydrostatic RK3-HEVI + Flux-Averaged Zalesak 3D FCT (Skamarock et al., 2012, Sec. 3c)
    # Note: When ny == 1 and nx == 1 (1D column spinup mode), horizontal acoustic waves vanish identically!
    is_2d_crm = (ny == 1) ^ (nx == 1)
    if ny == 1 and nx == 1:
        n_sub = max(n_dyn_substeps, 0)
    else:
        # Split-explicit outer RK3-WS3 obeys the convective/gravity-wave CFL (~45 m/s;
        # Wicker & Skamarock, 2002 stability limit is 1.25 in 2D -> 0.675 gives dt_sub = 60 s at dx = 4 km),
        # while step_crm_rk3_hevi sub-steps horizontal acoustic waves automatically.
        cfl_adv = 0.675 if is_2d_crm else 0.45
        dt_cfl_max = min(60.0, cfl_adv * min(crm.dx, crm.dy) / 45.0)
        n_sub = max(n_dyn_substeps, int(math.ceil(dt_crm / max(dt_cfl_max, 0.1))))
    dt_sub = dt_crm / float(max(n_sub, 1))
    dz_fct = dz_c_col if dz_c_col is not None else vgrid.dzeta_c

    # Group acoustic substeps into advective super-steps (<= 60 s) via exact time-averaged mass fluxes,
    # and refresh slow 3D Smagorinsky-Lilly eddy viscosity every ~120 s (every 2 substeps at dt_sub = 60 s).
    adv_group = max(1, min(n_sub, int(math.floor(60.0 / max(dt_sub, 1.0e-3)))))
    sgs_stride = max(1, int(math.floor(120.0 / max(dt_sub, 1.0e-3)))) if is_2d_crm else 1
    use_metal_crm = can_use_crm_metal(crm.rho_d)
    if n_sub > 0:
        # Exact Galilean co-moving frame shift per column (Grabowski, 2001; Khairoutdinov et al., 2005):
        # Periodic compressible Euler + Smagorinsky SGS + FCT + perturbation Rayleigh sponge are strictly
        # invariant under a column-uniform horizontal velocity shift (U_shift, V_shift). Choosing the column
        # midpoint 0.5*(u_max + u_min) cuts peak stratospheric polar-night jet advection speed in half
        # (e.g. 87 m/s -> +-46 m/s, keeping Wicker-Skamarock RK3 CFL << 1.25) with zero extra substeps
        # and zero CPU-GPU synchronizations.
        use_galilean = (nx > 1) or (ny > 1)
        u_gal_shift: torch.Tensor | None = None
        v_gal_shift: torch.Tensor | None = None
        if use_galilean:
            if nx > 1:
                u_gal_shift = 0.5 * (
                    crm.u.amax(dim=(0, 2, 3), keepdim=True) + crm.u.amin(dim=(0, 2, 3), keepdim=True)
                )
                crm.u = crm.u - u_gal_shift
            if ny > 1:
                v_gal_shift = 0.5 * (
                    crm.v.amax(dim=(0, 2, 3), keepdim=True) + crm.v.amin(dim=(0, 2, 3), keepdim=True)
                )
                crm.v = crm.v - v_gal_shift
        dz_c_sgs = dz_c_col.view(nz, -1, 1, 1) if dz_c_col is not None else vgrid.dzeta_c.view(nz, 1, 1, 1)
        dz_w_sgs = (
            (z_c_col[1:] - z_c_col[:-1]).view(nz - 1, -1, 1, 1)
            if z_c_col is not None
            else (vgrid.zeta_c[1:] - vgrid.zeta_c[:-1]).view(nz - 1, 1, 1, 1)
        )
        Thm_cur = crm.rho_d * crm.theta_m
        thm_box: list[torch.Tensor] = []
        sgs_cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        s_idx = 0
        no_clone_pp = use_metal_crm
        while s_idx < n_sub:
            k_cnt = min(adv_group, n_sub - s_idx)
            rho_start = crm.rho_d if (k_cnt == 1 and no_clone_pp) else crm.rho_d.clone()
            if k_cnt == 1:
                if sgs_cached is None or (s_idx % sgs_stride == 0):
                    thm_sgs = crm.theta_m if s_idx == 0 else (Thm_cur / torch.clamp(crm.rho_d, min=1.0e-5))
                    sgs_cached = _compute_3d_sgs_tendencies(
                        crm.rho_d, crm.u, crm.v, crm.w, thm_sgs, crm.dx, crm.dy, dz_c_sgs, dz_w_sgs
                    )
                fx_mean, fy_mean, fz_mean = step_crm_rk3_hevi(
                    crm, vgrid, dt_sub, z_c_col=z_c_col, dz_c_col=dz_c_col,
                    Thm_in=Thm_cur, sgs_in=sgs_cached, update_theta_m=False, thm_out=thm_box,
                    clone_out=not no_clone_pp,
                )
                Thm_cur = thm_box[0]
            else:
                fx_acc: torch.Tensor | None = None
                fy_acc: torch.Tensor | None = None
                fz_acc: torch.Tensor | None = None
                for sub_i in range(k_cnt):
                    cur_s = s_idx + sub_i
                    if sgs_cached is None or (cur_s % sgs_stride == 0):
                        thm_sgs = crm.theta_m if cur_s == 0 else (Thm_cur / torch.clamp(crm.rho_d, min=1.0e-5))
                        sgs_cached = _compute_3d_sgs_tendencies(
                            crm.rho_d, crm.u, crm.v, crm.w, thm_sgs, crm.dx, crm.dy, dz_c_sgs, dz_w_sgs
                        )
                    fx, fy, fz = step_crm_rk3_hevi(
                        crm, vgrid, dt_sub, z_c_col=z_c_col, dz_c_col=dz_c_col,
                        Thm_in=Thm_cur, sgs_in=sgs_cached, update_theta_m=False, thm_out=thm_box,
                        clone_out=not no_clone_pp,
                    )
                    Thm_cur = thm_box[0]
                    if sub_i == 0:
                        fx_acc = fx.clone()
                        fz_acc = fz.clone()
                        if ny > 1:
                            fy_acc = fy.clone()
                    else:
                        assert fx_acc is not None and fz_acc is not None
                        fx_acc.add_(fx)
                        fz_acc.add_(fz)
                        if ny > 1 and fy_acc is not None:
                            fy_acc.add_(fy)
                inv_k = 1.0 / float(k_cnt)
                assert fx_acc is not None and fz_acc is not None
                fx_acc.mul_(inv_k)
                fz_acc.mul_(inv_k)
                fx_mean = fx_acc
                fz_mean = fz_acc
                if ny > 1 and fy_acc is not None:
                    fy_acc.mul_(inv_k)
                    fy_mean = fy_acc
                else:
                    fy_mean = fx_mean
            dt_adv = dt_sub * float(k_cnt)
            if use_2m_mp and crm.n_c is not None and crm.n_i is not None:
                if use_metal_crm:
                    crm.q_n, crm.q_p, crm.n_c, crm.n_i = dispatch_crm_fct_quad_metal(
                        crm.q_n, crm.q_p, crm.n_c, crm.n_i, rho_start, crm.rho_d,
                        fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv,
                    )
                else:
                    crm.q_n = fct_advect_3d(crm.q_n, rho_start, crm.rho_d, fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv)
                    crm.q_p = fct_advect_3d(crm.q_p, rho_start, crm.rho_d, fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv)
                    crm.n_c = fct_advect_3d(crm.n_c, rho_start, crm.rho_d, fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv)
                    crm.n_i = fct_advect_3d(crm.n_i, rho_start, crm.rho_d, fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv)
            elif use_metal_crm:
                crm.q_n, crm.q_p = dispatch_crm_fct_pair_metal(
                    crm.q_n, crm.q_p, rho_start, crm.rho_d,
                    fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv,
                )
            else:
                crm.q_n = fct_advect_3d(crm.q_n, rho_start, crm.rho_d, fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv)
                crm.q_p = fct_advect_3d(crm.q_p, rho_start, crm.rho_d, fx_mean, fy_mean, fz_mean, crm.dx, crm.dy, dz_fct, dt_adv)
            s_idx += k_cnt
        if no_clone_pp:
            # Detach CRM tensors from shared RK3-HEVI workspace buffers once after all substeps finish
            crm.rho_d = crm.rho_d.clone()
            crm.w = crm.w.clone()
            if u_gal_shift is None:
                crm.u = crm.u.clone()
            if v_gal_shift is None:
                crm.v = crm.v.clone()
        if u_gal_shift is not None:
            crm.u = crm.u + u_gal_shift
        if v_gal_shift is not None:
            crm.v = crm.v + v_gal_shift
        crm.theta_m = Thm_cur / crm.rho_d
        q_cond_adv = torch.minimum(crm.q_c + crm.q_i, crm.q_n)
        crm.q_v = torch.clamp(crm.q_n - q_cond_adv, min=0.0)

    # Expand column vertical metrics across `[Nz, m_sub]` if terrain-following
    zc_flat = (
        z_c_col.unsqueeze(-1).unsqueeze(-1).expand(nz, n_col, ny, nx).reshape(nz, m_sub)
        if z_c_col is not None
        else vgrid.zeta_c.view(nz, 1)
    )
    zw_flat = (
        z_w_col.unsqueeze(-1).unsqueeze(-1).expand(nz + 1, n_col, ny, nx).reshape(nz + 1, m_sub)
        if z_w_col is not None
        else vgrid.zeta_w.view(nz + 1, 1)
    )
    dz_flat = (
        dz_c_col.unsqueeze(-1).unsqueeze(-1).expand(nz, n_col, ny, nx).reshape(nz, m_sub)
        if dz_c_col is not None
        else vgrid.dzeta_c.view(nz, 1)
    )

    # 3. Implicit MYNN Level 2.5 vertical boundary-layer turbulence
    if sfc_fluxes is not None:
        # Broadcast host/column surface fluxes `[N_col]` or `[N_col, Ny, Nx]` across `[N_col * Ny * Nx]`
        def _bcast_sfc(arr: torch.Tensor) -> torch.Tensor:
            if arr.ndim == 1 and int(arr.numel()) == n_col and m_sub != n_col:
                return arr.view(n_col, 1, 1).expand(n_col, ny, nx).reshape(m_sub)
            return arr.reshape(m_sub)

        u_f, v_f, thm_f, qn_f, qke_f, _, _, qv_f, qc_f, qi_f = step_mynn_level25(
            rho_d=crm.rho_d.reshape(nz, m_sub),
            u=crm.u.reshape(nz, m_sub),
            v=crm.v.reshape(nz, m_sub),
            theta_m_c=crm.theta_m.reshape(nz, m_sub),
            q_n=crm.q_n.reshape(nz, m_sub),
            q_v=crm.q_v.reshape(nz, m_sub),
            q_c=crm.q_c.reshape(nz, m_sub),
            q_i=crm.q_i.reshape(nz, m_sub),
            qke=crm.qke.reshape(nz, m_sub),
            z_c=zc_flat,
            z_w=zw_flat,
            dz_c=dz_flat,
            u_star=_bcast_sfc(sfc_fluxes.u_star),
            sfc_buoy_flux=_bcast_sfc(sfc_fluxes.buoy_flux),
            tau_x_sfc=_bcast_sfc(sfc_fluxes.tau_x),
            tau_y_sfc=_bcast_sfc(sfc_fluxes.tau_y),
            shf_sfc=_bcast_sfc(sfc_fluxes.shf),
            evap_sfc=_bcast_sfc(sfc_fluxes.evap),
            dt=dt_crm,
            return_hydrometeors=True,
        )
        crm.u = u_f.reshape(nz, n_col, ny, nx)
        crm.v = v_f.reshape(nz, n_col, ny, nx)
        crm.theta_m = thm_f.reshape(nz, n_col, ny, nx)
        crm.q_n = qn_f.reshape(nz, n_col, ny, nx)
        crm.qke = qke_f.reshape(nz, n_col, ny, nx)
        crm.q_v = qv_f.reshape(nz, n_col, ny, nx)
        crm.q_c = qc_f.reshape(nz, n_col, ny, nx)
        crm.q_i = qi_f.reshape(nz, n_col, ny, nx)

    # 4. Mixed-phase microphysics (6-class 2M Morrison, 2-moment KK00, or 1-moment SAM KR03) & sedimentation
    if use_6c_mp and crm.q_r is not None and crm.q_s is not None and crm.q_g is not None:
        from aoba.microphysics_6class import step_microphysics_six_class_2m

        (
            crm.theta_m, crm.q_v, crm.q_c, crm.q_i,
            crm.q_r, crm.q_s, crm.q_g,
            crm.n_c, crm.n_i, crm.n_r, crm.n_s, crm.n_g,
            rain_sub, snow_sub,
        ) = step_microphysics_six_class_2m(
            rho_d=crm.rho_d,
            theta_m=crm.theta_m,
            q_v=crm.q_v,
            q_c=crm.q_c,
            q_i=crm.q_i,
            q_r=crm.q_r,
            q_s=crm.q_s,
            q_g=crm.q_g,
            n_c=crm.n_c,
            n_i=crm.n_i,
            n_r=crm.n_r,
            n_s=crm.n_s,
            n_g=crm.n_g,
            dz_c=dz_c_col if dz_c_col is not None else vgrid.dzeta_c,
            dt=dt_crm,
            is_ocean_sub=is_ocean_sub,
            u_sfc_sub=u_sfc_sub,
            q_dust_sub=q_dust_sub,
        )
        crm.q_n = crm.q_v + crm.q_c + crm.q_i
        crm.q_p = crm.q_r + crm.q_s + crm.q_g
        rain_col = mean_2d(rain_sub)
        snow_col = mean_2d(snow_sub)
    else:
        if use_2m_mp and crm.n_c is not None and crm.n_i is not None:
            w_center = 0.5 * (crm.w[:-1] + crm.w[1:])
            w_eff = (torch.clamp(w_center, min=0.0) + 0.7 * torch.sqrt(0.5 * torch.clamp(crm.qke, min=0.0))).reshape(nz, m_sub)
            mp = step_microphysics_two_moment(
                rho_d=crm.rho_d.reshape(nz, m_sub),
                theta_m_c=crm.theta_m.reshape(nz, m_sub),
                q_n=crm.q_n.reshape(nz, m_sub),
                q_p=crm.q_p.reshape(nz, m_sub),
                q_v_prev=crm.q_v.reshape(nz, m_sub),
                n_c_in=crm.n_c.reshape(nz, m_sub),
                n_i_in=crm.n_i.reshape(nz, m_sub),
                dz_c=dz_flat,
                dt=dt_crm,
                w_eff=w_eff,
                is_ocean=is_ocean_sub.reshape(m_sub) if is_ocean_sub is not None else None,
                u_sfc=u_sfc_sub.reshape(m_sub) if u_sfc_sub is not None else None,
                q_dust=q_dust_sub.reshape(nz, m_sub) if q_dust_sub is not None else None,
            )
            if mp.n_c is not None:
                crm.n_c = mp.n_c.reshape(nz, n_col, ny, nx)
            if mp.n_i is not None:
                crm.n_i = mp.n_i.reshape(nz, n_col, ny, nx)
        else:
            mp = step_microphysics(
                rho_d=crm.rho_d.reshape(nz, m_sub),
                theta_m_c=crm.theta_m.reshape(nz, m_sub),
                q_n=crm.q_n.reshape(nz, m_sub),
                q_p=crm.q_p.reshape(nz, m_sub),
                q_v_prev=crm.q_v.reshape(nz, m_sub),
                dz_c=dz_flat,
                dt=dt_crm,
            )
        crm.theta_m = mp.theta_m_c.reshape(nz, n_col, ny, nx)
        crm.q_n = mp.q_n.reshape(nz, n_col, ny, nx)
        crm.q_p = mp.q_p.reshape(nz, n_col, ny, nx)
        crm.q_v = mp.q_v.reshape(nz, n_col, ny, nx)
        crm.q_c = mp.q_c.reshape(nz, n_col, ny, nx)
        crm.q_i = mp.q_i.reshape(nz, n_col, ny, nx)

        rain_sub = mp.precip_rain.reshape(n_col, ny, nx)
        snow_sub = mp.precip_snow.reshape(n_col, ny, nx)
        rain_col = mean_2d(rain_sub)
        snow_col = mean_2d(snow_sub)

    # Diagnose net CRM physics/convective tendencies relative to pre-step + LS forcing
    thm_ls = dtheta_m_ls_dt if dtheta_m_ls_dt is not None else torch.zeros_like(thm_mean_0)
    qn_ls = dq_n_ls_dt if dq_n_ls_dt is not None else torch.zeros_like(qn_mean_0)

    dthm_crm = (mean_2d(crm.theta_m) - thm_mean_0) / dt_crm - thm_ls
    dqn_crm = (mean_2d(crm.q_n) - qn_mean_0) / dt_crm - qn_ls
    dqp_crm = (mean_2d(crm.q_p) - qp_mean_0) / dt_crm
    du_crm = (mean_2d(crm.u) - u_mean_0) / dt_crm
    dv_crm = (mean_2d(crm.v) - v_mean_0) / dt_crm

    return SPTendencies(
        dtheta_m_dt=dthm_crm,
        dq_n_dt=dqn_crm,
        dq_p_dt=dqp_crm,
        du_dt=du_crm,
        dv_dt=dv_crm,
        precip_rain=rain_col,
        precip_snow=snow_col,
        precip_rain_sub=rain_sub,
        precip_snow_sub=snow_sub,
    )


# ---------------------------------------------------------------------------
# 4. Coupled Single-Column / 3D CRM Radiative-Convective Equilibrium Driver
# ---------------------------------------------------------------------------
@dataclass
class RCEDiagnostics:
    """Instantaneous column-mean diagnostics from `step_coupled_crm_rce`."""

    olr: torch.Tensor          # Outgoing longwave radiation at TOA [N_col] [W m^-2]
    asr: torch.Tensor          # Absorbed shortwave radiation at TOA [N_col] [W m^-2]
    precip_tot: torch.Tensor   # Total surface precipitation [N_col] [kg m^-2 s^-1]
    shf: torch.Tensor          # Surface sensible heat flux [N_col] [W m^-2]
    lhf: torch.Tensor          # Surface latent heat flux [N_col] [W m^-2]
    gpp: torch.Tensor          # Canopy GPP (over land) [N_col] [umol CO2 m^-2 s^-1]
    max_w: float               # Peak 3D vertical velocity magnitude [m s^-1]


def step_coupled_crm_rce(
    crm: CRMState,
    sfc: SurfaceState,
    vgrid: VerticalGrid,
    dt_crm: float,
    step_idx: int,
    cos_zenith: torch.Tensor,
    solar_constant: float = 551.58,
    co2_ppmv: float = 400.0,
    n_dyn_substeps: int = 2,
    rad_stride: int = 1,
    cached_rad_cache: dict[str, torch.Tensor] | None = None,
) -> RCEDiagnostics:
    """Advance coupled 3D CRM + Radiation + Surface + Two-Leaf Biosphere by `dt_crm` [s].

    Follows RCEMIP (`Wing et al., 2018`: `S_0 = 551.58 W m^-2`, `cos_zenith = cos(42.05 deg) = 0.7426`
    for insolation `409.6 W m^-2` over ocean or diurnal/constant forcing over land).
    """
    nz, n_col, ny, nx = crm.rho_d.shape
    m_sub = n_col * ny * nx
    is_subgrid_sfc = sfc.surf_type.numel() == m_sub and m_sub != n_col

    def _expand_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(n_col, 1, 1).expand(n_col, ny, nx).reshape(m_sub)

    def _mean_sub(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(n_col, ny * nx).mean(dim=-1)

    # 1. Subcolumn thermodynamic & centered lowest-layer wind profiles [Nz, m_sub]
    rho_sub = crm.rho_d.reshape(nz, m_sub)
    thm_sub = crm.theta_m.reshape(nz, m_sub)
    qv_sub = crm.q_v.reshape(nz, m_sub)
    qc_sub = crm.q_c.reshape(nz, m_sub)
    qi_sub = crm.q_i.reshape(nz, m_sub)
    u0_sub = (0.5 * (crm.u[0] + torch.roll(crm.u[0], 1, dims=-1))).reshape(m_sub)
    v0_sub = (0.5 * (crm.v[0] + torch.roll(crm.v[0], 1, dims=-2))).reshape(m_sub)

    pres_sub, _, temp_sub = diagnose_pressure_temperature(rho_sub, thm_sub, qv_sub)
    dz_0 = float((vgrid.zeta_c[0] - vgrid.zeta_w[0]).item())
    pres_sfc_sub = pres_sub[0] + C.GRAVITY * rho_sub[0] * (1.0 + qv_sub[0]) * dz_0
    mu0_sub = _expand_col(cos_zenith)
    mu0_sfc = mu0_sub if is_subgrid_sfc else cos_zenith
    pres_sfc_in = pres_sfc_sub if is_subgrid_sfc else _mean_sub(pres_sfc_sub)
    temp1_in = temp_sub[0] if is_subgrid_sfc else _mean_sub(temp_sub[0])
    qv1_in = qv_sub[0] if is_subgrid_sfc else _mean_sub(qv_sub[0])
    pres1_in = pres_sub[0] if is_subgrid_sfc else _mean_sub(pres_sub[0])
    u1_in = u0_sub if is_subgrid_sfc else _mean_sub(u0_sub)
    v1_in = v0_sub if is_subgrid_sfc else _mean_sub(v0_sub)
    z_ref_rel = torch.clamp(vgrid.zeta_c[0] - sfc.bio.h_canopy.reshape(-1) * 0.65, min=5.0)

    # 2. Subcolumn surface albedos & ICA 4-band radiative transfer across all [Nz, m_sub]
    alb_vd, alb_vf, alb_nd, alb_nf, _, lai_exp, _, _ = compute_surface_albedos(
        sfc, mu0_sfc.reshape(sfc.surf_type.shape)
    )
    need_rad = (cached_rad_cache is None) or (step_idx % max(rad_stride, 1) == 0)
    if need_rad:
        t_rad_raw = torch.where(sfc.surf_type == 0, sfc.t_ocean_ml, sfc.t_canopy).reshape(-1)
        t_rad_in = t_rad_raw if is_subgrid_sfc else _expand_col(t_rad_raw)
        alb_vd_sub = alb_vd.reshape(-1) if is_subgrid_sfc else _expand_col(alb_vd)
        alb_vf_sub = alb_vf.reshape(-1) if is_subgrid_sfc else _expand_col(alb_vf)
        alb_nd_sub = alb_nd.reshape(-1) if is_subgrid_sfc else _expand_col(alb_nd)
        alb_nf_sub = alb_nf.reshape(-1) if is_subgrid_sfc else _expand_col(alb_nf)

        pres_w = torch.empty((nz + 1, m_sub), device=rho_sub.device, dtype=rho_sub.dtype)
        pres_w[0] = pres_sfc_sub
        pres_w[1:-1] = torch.sqrt(torch.clamp(pres_sub[:-1] * pres_sub[1:], min=1.0))
        pres_w[-1] = torch.clamp(0.5 * pres_sub[-1], min=10.0)

        temp_w = torch.empty((nz + 1, m_sub), device=rho_sub.device, dtype=rho_sub.dtype)
        temp_w[0] = temp_sub[0]
        temp_w[1:-1] = 0.5 * (temp_sub[:-1] + temp_sub[1:])
        temp_w[-1] = temp_sub[-1]

        s_toa = torch.full_like(mu0_sub, solar_constant)
        rad = compute_four_band_radiation(
            pres_c=pres_sub,
            pres_w=pres_w,
            temp_c=temp_sub,
            temp_w=temp_w,
            dz_c=vgrid.dzeta_c.view(nz, 1).expand(nz, m_sub),
            rho_d_c=rho_sub,
            qv_c=qv_sub,
            qc_c=qc_sub,
            qi_c=qi_sub,
            t_sfc=t_rad_in,
            albedo_vis=alb_vd_sub,
            albedo_nir=alb_nd_sub,
            mu0=mu0_sub,
            s_toa=s_toa,
            co2_ppmv=co2_ppmv,
            albedo_vis_dif=alb_vf_sub,
            albedo_nir_dif=alb_nf_sub,
        )
        dt_rad = rad.q_rad_net.reshape(nz, n_col, ny, nx)
        sw1_dir_sub, sw1_dif_sub = rad.sfc_sw1_dir_dn, rad.sfc_sw1_dif_dn
        sw2_dir_sub, sw2_dif_sub = rad.sfc_sw2_dir_dn, rad.sfc_sw2_dif_dn
        lw_dn_sub = rad.lw_flux_dn[0]
        olr_sub, asr_sub = rad.toa_olr, rad.toa_asr
        if cached_rad_cache is not None:
            cached_rad_cache["dt_rad"] = dt_rad
            cached_rad_cache["sw1_dir"] = sw1_dir_sub
            cached_rad_cache["sw1_dif"] = sw1_dif_sub
            cached_rad_cache["sw2_dir"] = sw2_dir_sub
            cached_rad_cache["sw2_dif"] = sw2_dif_sub
            cached_rad_cache["lw_dn"] = lw_dn_sub
            cached_rad_cache["olr"] = olr_sub
            cached_rad_cache["asr"] = asr_sub
    else:
        dt_rad = cached_rad_cache["dt_rad"]
        sw1_dir_sub, sw1_dif_sub = cached_rad_cache["sw1_dir"], cached_rad_cache["sw1_dif"]
        sw2_dir_sub, sw2_dif_sub = cached_rad_cache["sw2_dir"], cached_rad_cache["sw2_dif"]
        lw_dn_sub = cached_rad_cache["lw_dn"]
        olr_sub, asr_sub = cached_rad_cache["olr"], cached_rad_cache["asr"]

    sw1_dir = sw1_dir_sub if is_subgrid_sfc else _mean_sub(sw1_dir_sub)
    sw1_dif = sw1_dif_sub if is_subgrid_sfc else _mean_sub(sw1_dif_sub)
    sw2_dir = sw2_dir_sub if is_subgrid_sfc else _mean_sub(sw2_dir_sub)
    sw2_dif = sw2_dif_sub if is_subgrid_sfc else _mean_sub(sw2_dif_sub)
    lw_dn = lw_dn_sub if is_subgrid_sfc else _mean_sub(lw_dn_sub)
    olr = _mean_sub(olr_sub)
    asr = _mean_sub(asr_sub)

    rain_prev = cached_rad_cache.get("rain", torch.zeros_like(pres_sfc_in)) if cached_rad_cache is not None else torch.zeros_like(pres_sfc_in)
    snow_prev = cached_rad_cache.get("snow", torch.zeros_like(pres_sfc_in)) if cached_rad_cache is not None else torch.zeros_like(pres_sfc_in)

    # 3. Surface energy/water balance & Two-Leaf gas exchange (subgrid MAML or host mean)
    sfc_diag = step_surface(
        sfc=sfc,
        u_1=u1_in,
        v_1=v1_in,
        temp_1=temp1_in,
        qv_1=qv1_in,
        pres_1=pres1_in,
        pres_sfc=pres_sfc_in,
        z_ref_rel=z_ref_rel,
        sw1_dir_down=sw1_dir,
        sw1_dif_down=sw1_dif,
        sw2_dir_down=sw2_dir,
        sw2_dif_down=sw2_dif,
        lw_down=lw_dn,
        precip_rain=rain_prev,
        precip_snow=snow_prev,
        mu0=mu0_sfc,
        co2_ppmv=co2_ppmv,
        dt=dt_crm,
    )

    # 4. 3D CRM Dynamics + Zalesak FCT + MYNN 2.5 + SAM Microphysics
    sp_out = step_crm_physics_and_dynamics(
        crm=crm,
        vgrid=vgrid,
        dt_crm=dt_crm,
        n_dyn_substeps=n_dyn_substeps,
        sfc_fluxes=sfc_diag,
        rad_heating_k_s=dt_rad,
    )
    rain_next_in = sp_out.precip_rain_sub.reshape(m_sub) if (is_subgrid_sfc and sp_out.precip_rain_sub is not None) else sp_out.precip_rain
    snow_next_in = sp_out.precip_snow_sub.reshape(m_sub) if (is_subgrid_sfc and sp_out.precip_snow_sub is not None) else sp_out.precip_snow
    if cached_rad_cache is not None:
        cached_rad_cache["rain"] = rain_next_in
        cached_rad_cache["snow"] = snow_next_in

    # 5. Daily EEO acclimation & LAI update once per simulated solar day
    steps_per_day = max(int(round(C.SECONDS_PER_DAY / dt_crm)), 1)
    if (step_idx + 1) % steps_per_day == 0 and (sfc.surf_type == 1).any():
        is_veg = sfc.surf_type == 1
        es_1 = 611.2 * torch.exp(17.67 * (temp1_in - C.T_FREEZE) / torch.clamp(temp1_in - 29.65, min=10.0))
        e_1 = qv1_in * pres1_in / (C.EPS_WATER + (1.0 - C.EPS_WATER) * qv1_in)
        vpd_1 = torch.clamp(es_1 - e_1, min=50.0)
        update_daily_acclimation_smith(
            bio=sfc.bio,
            midday_vis_beam=sw1_dir,
            midday_vis_diff=sw1_dif,
            midday_mu0=torch.clamp(mu0_sfc, min=0.1),
            midday_temp=temp1_in + 1.5,
            midday_vpd=vpd_1,
            night_temp=temp1_in - 3.0,
            pres_sfc=pres_sfc_in,
            co2_ppmv=co2_ppmv,
            lai_exposed=lai_exp,
            is_vegetated_land=is_veg,
        )
        precip_mm_day = (rain_next_in + snow_next_in) * C.SECONDS_PER_DAY
        pet_mm_day = torch.clamp((sw1_dir + sw1_dif + sw2_dir + sw2_dif) * C.SECONDS_PER_DAY / C.L_V0, min=0.1)
        ppfd_mol_day = (4.57 * sw1_dir + 4.24 * sw1_dif) * 1.0e-6 * C.SECONDS_PER_DAY
        ws_beta = compute_optimal_soil_water_beta(sfc.w_root, sfc.bio)
        update_daily_lai_root_height(
            bio=sfc.bio,
            daily_precip_mm=precip_mm_day,
            daily_pet_mm=pet_mm_day,
            daily_ppfd_mol_m2=ppfd_mol_day,
            daily_temp_k=temp1_in,
            daily_vpd_pa=vpd_1,
            pres_sfc=pres_sfc_in,
            co2_ppmv=co2_ppmv,
            water_stress_beta=ws_beta,
            is_vegetated_land=is_veg,
        )

    shf_out = _mean_sub(sfc_diag.shf) if is_subgrid_sfc else sfc_diag.shf
    lhf_out = _mean_sub(sfc_diag.lhf) if is_subgrid_sfc else sfc_diag.lhf
    gpp_out = _mean_sub(sfc_diag.gpp) if is_subgrid_sfc else sfc_diag.gpp

    return RCEDiagnostics(
        olr=olr,
        asr=asr,
        precip_tot=sp_out.precip_rain + sp_out.precip_snow,
        shf=shf_out,
        lhf=lhf_out,
        gpp=gpp_out,
        max_w=float(crm.w.abs().max().item()),
    )
