"""Surface-layer Roughness Sublayer (RSL) similarity, MYNN Level 2.5 PBL, and 3D SGS turbulence.

Every parameterization cites its exact source equations:
1. Vegetation & Ocean Aerodynamic Roughness:
   - Raupach (1994, Boundary-Layer Meteorol., 71, 211-216, Eqs. 8 & 10-11):
     Zero-plane displacement `d(h_c, LAI)` and roughness length `z_0m(h_c, LAI)`.
   - Miller, Beljaars, & Palmer (1992, J. Climate, 5, 418-434, Eqs. 3-5) &
     SCALE-RM (Nishizawa et al., 2025, Sec. 8.6.4 Eqs. 8.360-8.362):
     Viscous smooth-flow + Charnock wave roughness lengths `z_0m, z_0h, z_0q` over ocean.
2. Monin-Obukhov Similarity + Analytical Roughness Sublayer (RSL) Bulk Transfer:
   - Harman & Finnigan (2007, Boundary-Layer Meteorol., 123, 339-363).
   - De Ridder (2010, Boundary-Layer Meteorol., 134, 257-267, Eqs. 5-22):
     Closed-form exponential-integral RSL correction `psi_star_M,H(zeta, z / z_*)`.
   - Bonan et al. (2018, Geosci. Model Dev., 11, 1467-1496) & Lee et al. (2020, GMD, 13, 521-536):
     Stability bounds and regional/global LSM coupling.
3. Mellor-Yamada-Nakanishi-Niino (MYNN) Level 2.5 Boundary-Layer Closure:
   - Nakanishi & Niino (2009, J. Meteorol. Soc. Japan, 87, 895-912) &
     SCALE-RM (Nishizawa et al., 2025, Sec. 8.2.1 Eqs. 8.96-8.176):
     Prognostic doubled TKE `q^2 = 2 * e`, Gaussian subgrid condensation (`Q_1, R, Q_l,
     beta_theta, beta_q`), three-scale master length `L` (`L_S, L_T, L_B`), Level-2/2.5
     stability functions `S_M, S_H`, and unconditionally stable implicit batched Thomas diffusion.
4. Anisotropic 3D Smagorinsky-Lilly Subgrid-Scale (SGS) Closure:
   - Lilly (1962), Scotti, Meneveau, & Lilly (1993), &
     SCALE-RM (Nishizawa et al., 2025, Sec. 8.1.2 Eqs. 8.25-8.40):
     Richardson-dependent eddy viscosity `nu_SGS` and scalar diffusivity `D_SGS` with
     grid-aspect-ratio filter length correction `f(a)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import solve_tridiagonal_batched
from aoba.thermodynamics import diagnose_pressure_temperature, qsat_mixed, saturation_adjustment

# ---------------------------------------------------------------------------
# 1. MYNN Level 2.5 Closure Constants (Nakanishi & Niino, 2009; SCALE-RM Eqs. 8.150-8.162)
# ---------------------------------------------------------------------------
MYNN_B1: float = 24.0
MYNN_B2: float = 15.0
MYNN_C2: float = 0.75
MYNN_C3: float = 0.352
MYNN_C5: float = 0.20
MYNN_GAMMA1: float = 0.235
MYNN_PR_N: float = 0.74

MYNN_A1: float = MYNN_B1 * (1.0 - 3.0 * MYNN_GAMMA1) / 6.0
MYNN_A2: float = 1.0 / (3.0 * MYNN_GAMMA1 * (MYNN_B1 ** (1.0 / 3.0)) * MYNN_PR_N)
MYNN_C1: float = MYNN_GAMMA1 - 1.0 / (3.0 * MYNN_A1 * (MYNN_B1 ** (1.0 / 3.0)))
MYNN_GAMMA2: float = (2.0 * MYNN_A1 * (3.0 - 2.0 * MYNN_C2) + MYNN_B2 * (1.0 - MYNN_C3)) / MYNN_B1
MYNN_F1: float = (
    MYNN_B1 * (MYNN_GAMMA1 - MYNN_C1)
    + 2.0 * MYNN_A1 * (3.0 - 2.0 * MYNN_C2)
    + 3.0 * MYNN_A2 * (1.0 - MYNN_C2) * (1.0 - MYNN_C5)
)
MYNN_F2: float = MYNN_B1 * (MYNN_GAMMA1 + MYNN_GAMMA2) - 3.0 * MYNN_A1 * (1.0 - MYNN_C2)
MYNN_RF1: float = MYNN_B1 * (MYNN_GAMMA1 - MYNN_C1) / MYNN_F1
MYNN_RF2: float = MYNN_B1 * MYNN_GAMMA1 / MYNN_F2
MYNN_RFC: float = MYNN_GAMMA1 / (MYNN_GAMMA1 + MYNN_GAMMA2)

TKE_MIN: float = 1.0e-4  # Minimum prognostic TKE e_min [m^2 s^-2] (q^2_min = 2e-4)

# De Ridder (2010, Eq. 22) analytical RSL parameters
RSL_MU_M: float = 2.59
RSL_MU_H: float = 0.95
RSL_NU: float = 0.50
RSL_LAMBDA: float = 1.50

# Kinematic viscosity of air [m^2 s^-1] (Miller et al., 1992; SCALE-RM Eq. 8.360)
KINEMATIC_VISC_AIR: float = 1.5e-5


# ---------------------------------------------------------------------------
# 2. Canopy & Ocean Roughness Lengths
# ---------------------------------------------------------------------------
def canopy_roughness_raupach(
    h_canopy: torch.Tensor,
    lai: torch.Tensor,
    z0_soil: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute zero-plane displacement `d`, momentum roughness `z_0m`, and scalar `z_0h`.

    Reference: Raupach (1994, Boundary-Layer Meteorol., 71, 211-216, Eqs. 8, 10, 11):
        1 - d / h = (1 - exp(-sqrt(c_d1 * LAI))) / sqrt(c_d1 * LAI)
        u_* / U_h = min(sqrt(c_S + c_R * LAI / 2), (u_* / U_h)_max)
        z_0m / h = (1 - d / h) * exp(-kappa * U_h / u_* + psi_cw)
    with `c_d1 = 7.5`, `c_S = 0.003`, `c_R = 0.3`, `(u_*/U_h)_max = 0.30`, `psi_cw = 0.193`.
    Smoothly blends to bare-ground `z0_soil` as `h_canopy -> 0` or `LAI -> 0`.
    """
    hc = torch.clamp(h_canopy, min=0.0)
    lai_pos = torch.clamp(lai, min=1.0e-4)

    x = torch.sqrt(7.5 * lai_pos)
    one_minus_d_over_h = (1.0 - torch.exp(-x)) / x
    d_disp = hc * (1.0 - one_minus_d_over_h)

    ustar_over_uh = torch.clamp(torch.sqrt(0.003 + 0.15 * lai_pos), max=0.30)
    z0m_veg = hc * one_minus_d_over_h * torch.exp(-C.VON_KARMAN / ustar_over_uh + 0.193)
    z0m = torch.maximum(z0m_veg, torch.full_like(hc, z0_soil))
    # Scalar roughness length (Garratt, 1992; De Ridder, 2010): ln(z_0m / z_0h) ≈ 2.0 -> z_0h = z_0m / e^2
    z0h = 0.135335 * z0m
    return d_disp, z0m, z0h


def ocean_roughness_miller(
    u_star: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sea-surface roughness lengths `z_0m, z_0h, z_0q` [m].

    Reference: Miller, Beljaars, & Palmer (1992, J. Climate, 5, 418-434, Eqs. 3-5) &
    SCALE-RM (Nishizawa et al., 2025, Sec. 8.6.4 Eqs. 8.360-8.362):
        z_0m = 0.11 * nu / u_* + 0.018 * u_*^2 / g
        z_0h = 0.40 * nu / u_* + 1.4e-5
        z_0q = 0.62 * nu / u_* + 1.3e-4.
    """
    us = torch.clamp(u_star, min=0.02)
    z0m = torch.clamp(0.11 * KINEMATIC_VISC_AIR / us + 0.018 * (us ** 2) / C.GRAVITY, 1.0e-6, 0.05)
    z0h = torch.clamp(0.40 * KINEMATIC_VISC_AIR / us + 1.4e-5, 1.0e-6, 0.02)
    z0q = torch.clamp(0.62 * KINEMATIC_VISC_AIR / us + 1.3e-4, 1.0e-6, 0.02)
    return z0m, z0h, z0q


# ---------------------------------------------------------------------------
# 3. Monin-Obukhov + Analytical Roughness Sublayer (RSL) Similarity
# ---------------------------------------------------------------------------
def _dyer_phi_m(zeta: torch.Tensor) -> torch.Tensor:
    """Dyer (1974) gradient stability function Phi_M(zeta) (De Ridder, 2010, Eqs. 16-17)."""
    z_unst = torch.clamp(zeta, max=0.0)
    z_stab = torch.clamp(zeta, min=0.0, max=2.0)
    return torch.where(zeta < 0.0, torch.pow(1.0 - 16.0 * z_unst, -0.25), 1.0 + 5.0 * z_stab)


def _dyer_phi_h(zeta: torch.Tensor) -> torch.Tensor:
    """Dyer (1974) gradient stability function Phi_H(zeta) (De Ridder, 2010, Eqs. 16-17)."""
    z_unst = torch.clamp(zeta, max=0.0)
    z_stab = torch.clamp(zeta, min=0.0, max=2.0)
    return torch.where(zeta < 0.0, torch.pow(1.0 - 16.0 * z_unst, -0.50), 1.0 + 5.0 * z_stab)


def _paulson_psi_m(zeta: torch.Tensor) -> torch.Tensor:
    """Paulson (1970) integrated stability function Psi_M(zeta) (De Ridder, 2010, Eqs. 18 & 20)."""
    z_unst = torch.clamp(zeta, max=0.0)
    z_stab = torch.clamp(zeta, min=0.0, max=2.0)
    x = torch.pow(1.0 - 16.0 * z_unst, 0.25)
    psi_unst = (
        2.0 * torch.log(0.5 * (1.0 + x))
        + torch.log(0.5 * (1.0 + x * x))
        - 2.0 * torch.atan(x)
        + 0.5 * math.pi
    )
    return torch.where(zeta < 0.0, psi_unst, -5.0 * z_stab)


def _paulson_psi_h(zeta: torch.Tensor) -> torch.Tensor:
    """Paulson (1970) integrated stability function Psi_H(zeta) (De Ridder, 2010, Eqs. 19-20)."""
    z_unst = torch.clamp(zeta, max=0.0)
    z_stab = torch.clamp(zeta, min=0.0, max=2.0)
    x2 = torch.pow(1.0 - 16.0 * z_unst, 0.50)
    psi_unst = 2.0 * torch.log(0.5 * (1.0 + x2))
    return torch.where(zeta < 0.0, psi_unst, -5.0 * z_stab)


def _rsl_psi_star(
    zeta: torch.Tensor,
    chi: torch.Tensor,
    mu_i: float,
    is_momentum: bool,
) -> torch.Tensor:
    """Closed-form Roughness Sublayer integral `psi_i^*(zeta, chi)` of De Ridder (2010, Eq. 22):

        psi_i^*(zeta, chi) = Phi_i((1 + nu / (mu_i * chi)) * zeta)
                             * (1 / lambda) * ln(1 + lambda / (mu_i * chi)) * exp(-mu_i * chi).
    """
    mux = torch.clamp(mu_i * chi, min=0.05)
    zeta_eff = (1.0 + RSL_NU / mux) * zeta
    phi_val = _dyer_phi_m(zeta_eff) if is_momentum else _dyer_phi_h(zeta_eff)
    e1_approx = (1.0 / RSL_LAMBDA) * torch.log1p(RSL_LAMBDA / mux) * torch.exp(-torch.clamp(mux, max=40.0))
    return phi_val * e1_approx


@dataclass
class SurfaceTransferState:
    """Bulk surface transfer coefficients and scaling parameters."""

    c_m: torch.Tensor       # Momentum bulk transfer coefficient C_M [-]
    c_h: torch.Tensor       # Sensible-heat bulk transfer coefficient C_H [-]
    c_q: torch.Tensor       # Moisture bulk transfer coefficient C_Q [-]
    u_star: torch.Tensor    # Friction velocity u_* [m s^-1]
    zeta: torch.Tensor      # Monin-Obukhov stability z_ref / L_M [-]
    wind_eff: torch.Tensor  # Gustiness-floored effective wind speed U_eff [m s^-1]


def compute_surface_transfer(
    u_1: torch.Tensor,
    v_1: torch.Tensor,
    theta_v1: torch.Tensor,
    theta_vsfc: torch.Tensor,
    z_ref: torch.Tensor,
    z0m: torch.Tensor,
    z0h: torch.Tensor,
    z0q: torch.Tensor,
    rsl_depth: torch.Tensor | None = None,
    n_iter: int = 4,
) -> SurfaceTransferState:
    """Compute Monin-Obukhov + RSL bulk transfer coefficients `C_M, C_H, C_Q`.

    Parameters
    ----------
    u_1, v_1 : torch.Tensor
        Lowest full-level wind components [m s^-1].
    theta_v1, theta_vsfc : torch.Tensor
        Lowest full-level and surface virtual potential temperatures [K].
    z_ref : torch.Tensor
        Reference height above `z_datum = z_topo + d` (typically `tilde_z_1 = 30 m`) [m].
    z0m, z0h, z0q : torch.Tensor
        Roughness lengths for momentum, heat, and moisture [m].
    rsl_depth : torch.Tensor | None
        Roughness sublayer depth above `d`, `z_* = max(2 * h_c - d, 0)` (De Ridder, 2010, Sec. 3).
        When `None` or `<= 0` (ocean / bare soil / snow), RSL correction vanishes smoothly.

    References
    ----------
    - De Ridder (2010, Boundary-Layer Meteorol., 134, Eqs. 6-7 & 22).
    - Bonan et al. (2018, Geosci. Model Dev., 11, Sec. 2.2) & Lee et al. (2020, GMD, 13, Eqs. 6-11).
    """
    z_safe = torch.clamp(z_ref, min=2.0)
    z0m_s = torch.minimum(torch.clamp(z0m, min=1.0e-6), 0.45 * z_safe)
    z0h_s = torch.minimum(torch.clamp(z0h, min=1.0e-7), 0.45 * z_safe)
    z0q_s = torch.minimum(torch.clamp(z0q, min=1.0e-7), 0.45 * z_safe)

    # Include minimum subgrid convective gustiness velocity 0.5 m s^-1 (Miller et al., 1992)
    wind = torch.sqrt(u_1 * u_1 + v_1 * v_1 + 0.25)
    theta_bar = 0.5 * (theta_v1 + theta_vsfc)
    ri_b = torch.clamp(
        C.GRAVITY * z_safe * (theta_v1 - theta_vsfc) / (torch.clamp(theta_bar, min=200.0) * wind * wind),
        -5.0,
        1.5,
    )

    has_rsl = rsl_depth is not None
    if has_rsl:
        z_star = torch.clamp(rsl_depth, min=0.0)
        active_rsl = z_star > 0.05
        chi = z_safe / torch.clamp(z_star, min=0.05)
        chi_0m = z0m_s / torch.clamp(z_star, min=0.05)
        chi_0h = z0h_s / torch.clamp(z_star, min=0.05)
        chi_0q = z0q_s / torch.clamp(z_star, min=0.05)
    else:
        active_rsl = None

    ln_zm = torch.log(z_safe / z0m_s)
    ln_zh = torch.log(z_safe / z0h_s)
    ln_zq = torch.log(z_safe / z0q_s)

    # Initial guess for zeta = z / L_M from bulk Richardson number (Bonan et al., 2018)
    zeta = torch.clamp(ri_b * ln_zm, -5.0, 2.0)

    for _ in range(n_iter):
        zeta_0m = zeta * (z0m_s / z_safe)
        zeta_0h = zeta * (z0h_s / z_safe)

        fm = ln_zm - _paulson_psi_m(zeta) + _paulson_psi_m(zeta_0m)
        fh = ln_zh - _paulson_psi_h(zeta) + _paulson_psi_h(zeta_0h)

        if has_rsl:
            # De Ridder (2010, Eqs. 6-7): integral from z_0 to z = integral_z0^inf - integral_z^inf
            # Note: integral_z0^z [Phi*(1 - phi)/z'] dz' = psi_star(z0) - psi_star(z), reducing denominator!
            rsl_m = _rsl_psi_star(zeta_0m, chi_0m, RSL_MU_M, True) - _rsl_psi_star(zeta, chi, RSL_MU_M, True)
            rsl_h = _rsl_psi_star(zeta_0h, chi_0h, RSL_MU_H, False) - _rsl_psi_star(zeta, chi, RSL_MU_H, False)
            fm = fm - torch.where(active_rsl, rsl_m, torch.zeros_like(fm))
            fh = fh - torch.where(active_rsl, rsl_h, torch.zeros_like(fh))

        fm = torch.clamp(fm, min=0.8)
        fh = torch.clamp(fh, min=0.8)
        zeta = torch.clamp(ri_b * fm * fm / fh, -5.0, 2.0)

    zeta_0m = zeta * (z0m_s / z_safe)
    zeta_0h = zeta * (z0h_s / z_safe)
    zeta_0q = zeta * (z0q_s / z_safe)

    fm = ln_zm - _paulson_psi_m(zeta) + _paulson_psi_m(zeta_0m)
    fh = ln_zh - _paulson_psi_h(zeta) + _paulson_psi_h(zeta_0h)
    fq = ln_zq - _paulson_psi_h(zeta) + _paulson_psi_h(zeta_0q)

    if has_rsl:
        rsl_m = _rsl_psi_star(zeta_0m, chi_0m, RSL_MU_M, True) - _rsl_psi_star(zeta, chi, RSL_MU_M, True)
        rsl_h = _rsl_psi_star(zeta_0h, chi_0h, RSL_MU_H, False) - _rsl_psi_star(zeta, chi, RSL_MU_H, False)
        rsl_q = _rsl_psi_star(zeta_0q, chi_0q, RSL_MU_H, False) - _rsl_psi_star(zeta, chi, RSL_MU_H, False)
        fm = fm - torch.where(active_rsl, rsl_m, torch.zeros_like(fm))
        fh = fh - torch.where(active_rsl, rsl_h, torch.zeros_like(fh))
        fq = fq - torch.where(active_rsl, rsl_q, torch.zeros_like(fq))

    fm = torch.clamp(fm, min=0.8)
    fh = torch.clamp(fh, min=0.8)
    fq = torch.clamp(fq, min=0.8)

    vk2 = C.VON_KARMAN * C.VON_KARMAN
    c_m = vk2 / (fm * fm)
    c_h = vk2 / (fm * fh)
    c_q = vk2 / (fm * fq)
    u_star = torch.sqrt(c_m) * wind

    return SurfaceTransferState(
        c_m=c_m,
        c_h=c_h,
        c_q=c_q,
        u_star=u_star,
        zeta=zeta,
        wind_eff=wind,
    )


# ---------------------------------------------------------------------------
# 4. Implicit MYNN Level 2.5 Boundary-Layer Turbulence Closure
# ---------------------------------------------------------------------------
def step_mynn_level25(
    rho_d: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    theta_m_c: torch.Tensor,
    q_n: torch.Tensor,
    q_v: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    qke: torch.Tensor,
    z_c: torch.Tensor,
    z_w: torch.Tensor,
    dz_c: torch.Tensor,
    u_star: torch.Tensor,
    sfc_buoy_flux: torch.Tensor,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    shf_sfc: torch.Tensor,
    evap_sfc: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance vertical PBL mixing and prognostic doubled TKE `qke = q^2 = 2e` via MYNN Level 2.5.

    All 3D/column fields have leading dimension `nz` (`k = 0` lowest full level, `k = nz - 1` top),
    except `z_w` which has `[nz + 1, ...]`. Surface fluxes (`tau_x_sfc`, `tau_y_sfc`, `shf_sfc` [W m^-2],
    `evap_sfc` [kg m^-2 s^-1], `sfc_buoy_flux` `= (g/theta_0) * <w'theta_v'>_sfc` [m^2 s^-3])
    are defined positive upward from the surface into layer `k = 0`.

    Returns `(u_new, v_new, theta_m_c_new, q_n_new, qke_new, k_m_int, k_h_int)`.

    References
    ----------
    - Nakanishi & Niino (2009, J. Meteorol. Soc. Japan, 87, 895-912).
    - Nishizawa et al. (2025, SCALE-RM v5.5.5, Sec. 8.2.1 Eqs. 8.96-8.176).
    """
    nz = rho_d.shape[0]
    q2 = torch.clamp(qke, min=2.0 * TKE_MIN)
    q_vel = torch.sqrt(q2)

    pres, exner, temp = diagnose_pressure_temperature(rho_d, theta_m_c, q_v)
    theta = temp / exner
    # Liquid-ice water potential temperature theta_l (SCALE-RM Eq. 8.98)
    theta_l = theta - (C.L_V0 * q_c + C.L_S0 * q_i) / (C.CP_D * exner)

    # Interior half-level interfaces k + 1/2 for k = 0 .. nz - 2 (shape [nz - 1, ...])
    dz_int = torch.clamp(z_c[1:] - z_c[:-1], min=1.0)
    z_int = z_w[1:-1] - z_w[0:1]  # Height above z_datum [nz - 1, ...]

    du_dz = (u[1:] - u[:-1]) / dz_int
    dv_dz = (v[1:] - v[:-1]) / dz_int
    dthl_dz = (theta_l[1:] - theta_l[:-1]) / dz_int
    dqn_dz = (q_n[1:] - q_n[:-1]) / dz_int
    shear2 = du_dz * du_dz + dv_dz * dv_dz + 1.0e-10

    q_int = 0.5 * (q_vel[1:] + q_vel[:-1])
    q2_int = q_int * q_int
    temp_int = 0.5 * (temp[1:] + temp[:-1])
    pres_int = 0.5 * (pres[1:] + pres[:-1])
    theta_int = 0.5 * (theta[1:] + theta[:-1])
    qn_int = 0.5 * (q_n[1:] + q_n[:-1])
    qc_int = 0.5 * (q_c[1:] + q_c[:-1]) + 0.5 * (q_i[1:] + q_i[:-1])

    # Partial condensation buoyancy coefficients beta_theta, beta_q (SCALE-RM Eqs. 8.112-8.113, 8.122-8.131)
    qsl, dqsl_dT = qsat_mixed(temp_int, pres_int)
    a_cond = 1.0 / (1.0 + (C.L_V0 / C.CP_D) * dqsl_dT)
    b_cond = (temp_int / theta_int) * dqsl_dT
    c_cond = (1.0 + C.ZVIR * qn_int - (1.0 + C.ZVIR) * qc_int) * (
        theta_int * C.L_V0 / (temp_int * C.CP_D)
    ) - (1.0 + C.ZVIR) * theta_int

    # First estimate of N^2 using grid-scale cloud fraction to evaluate master length scale L
    r_grid = (qc_int > 1.0e-6).to(rho_d.dtype)
    beta_th0 = 1.0 + C.ZVIR * qn_int - (1.0 + C.ZVIR) * qc_int - 0.5 * r_grid * a_cond * b_cond * c_cond
    beta_q0 = C.ZVIR * theta_int + 0.5 * r_grid * a_cond * c_cond
    n2_init = (C.GRAVITY / theta_int) * (beta_th0 * dthl_dz + beta_q0 * dqn_dz)

    # Master turbulent length scale 1/L = 1/L_S + 1/L_T + 1/L_B (SCALE-RM Eqs. 8.143-8.149)
    us_safe = torch.clamp(u_star, min=0.02)
    # Monin-Obukhov length L_M = -u_*^3 / (kappa * B_sfc) where B_sfc = (g/theta_0)*<w'theta_v'>_sfc
    b_sfc = sfc_buoy_flux
    b_sfc_nz = torch.where(
        torch.abs(b_sfc) < 1.0e-9,
        torch.full_like(b_sfc, 1.0e-9),
        b_sfc,
    )
    l_mo = -(us_safe ** 3) / (C.VON_KARMAN * b_sfc_nz)
    zeta_int = torch.clamp(z_int / l_mo.unsqueeze(0), -10.0, 5.0)

    ls_unst = C.VON_KARMAN * z_int * torch.pow(torch.clamp(1.0 - 100.0 * zeta_int, min=1.0), 0.20)
    ls_weak = C.VON_KARMAN * z_int / (1.0 + 2.7 * torch.clamp(zeta_int, min=0.0, max=1.0))
    ls_strong = C.VON_KARMAN * z_int / 3.7
    l_surf = torch.where(
        zeta_int < 0.0,
        ls_unst,
        torch.where(zeta_int < 1.0, ls_weak, ls_strong),
    )

    # Boundary-layer integral scale L_T = 0.23 * int(q * z dz) / int(q dz) (Eq. 8.145)
    z_rel_c = z_c - z_w[0:1]
    l_t_col = 0.23 * (q_vel * z_rel_c * dz_c).sum(dim=0) / torch.clamp((q_vel * dz_c).sum(dim=0), min=1.0e-4)
    l_t_col = torch.clamp(l_t_col, min=20.0, max=2500.0)
    l_t = l_t_col.unsqueeze(0).expand_as(q_int)

    # Convective velocity scale q_c = (B_sfc^+ * L_T)^(1/3) and buoyancy scale L_B (Eqs. 8.146, 8.149)
    q_conv = torch.pow(torch.clamp(b_sfc, min=0.0) * l_t_col, 1.0 / 3.0).unsqueeze(0)
    n_bv = torch.sqrt(torch.clamp(n2_init, min=1.0e-8))
    lb_base = q_int / n_bv
    lb_conv = (1.0 + 5.0 * torch.sqrt(q_conv / torch.clamp(l_t * n_bv, min=1.0e-6))) * lb_base
    inv_lb = torch.where(
        n2_init > 1.0e-8,
        1.0 / torch.clamp(torch.where(zeta_int < 0.0, lb_conv, lb_base), min=1.0),
        torch.zeros_like(q_int),
    )

    l_master = 1.0 / (1.0 / torch.clamp(l_surf, min=0.5) + 1.0 / l_t + inv_lb)
    l_master = torch.clamp(l_master, min=0.5, max=1500.0)

    # Subgrid Gaussian condensation variance sigma_s & cloud fraction R (SCALE-RM Eqs. 8.122-8.126)
    dq_li = dqn_dz - b_cond * dthl_dz
    sigma_s = torch.clamp(0.5 * a_cond * l_master * math.sqrt(MYNN_B2 * 0.5) * torch.abs(dq_li), min=1.0e-7)
    q1_norm = torch.clamp(a_cond * (qn_int - qsl) / (2.0 * sigma_s), -6.0, 6.0)
    r_sub = 0.5 * (1.0 + torch.erf(q1_norm / math.sqrt(2.0)))
    gauss_q1 = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * q1_norm * q1_norm)
    ql_sub = 2.0 * sigma_s * (r_sub * q1_norm + gauss_q1)
    r_tilde = torch.clamp(r_sub - (ql_sub / (2.0 * sigma_s)) * gauss_q1, 0.0, 1.0)

    beta_th = 1.0 + C.ZVIR * qn_int - (1.0 + C.ZVIR) * ql_sub - r_tilde * a_cond * b_cond * c_cond
    beta_q = C.ZVIR * theta_int + r_tilde * a_cond * c_cond
    n2 = (C.GRAVITY / theta_int) * (beta_th * dthl_dz + beta_q * dqn_dz)

    # Level-2 diagnostic equilibrium q_2, S_M2, S_H2 (SCALE-RM Eqs. 8.134-8.142)
    ri = torch.clamp(n2 / shear2, -10.0, 5.0)
    af_ratio = (MYNN_A1 * MYNN_F1) / (MYNN_A2 * MYNN_F2)
    disc = torch.clamp(
        ri * ri + 2.0 * af_ratio * (MYNN_RF1 - 2.0 * MYNN_RF2) * ri + (af_ratio * MYNN_RF1) ** 2,
        min=0.0,
    )
    rf = 0.5 * (1.0 / af_ratio) * (ri + af_ratio * MYNN_RF1 - torch.sqrt(disc))
    rf = torch.clamp(rf, max=MYNN_RFC - 1.0e-4)

    sh2 = torch.clamp(3.0 * MYNN_A2 * (MYNN_GAMMA1 + MYNN_GAMMA2) * (MYNN_RFC - rf) / (1.0 - rf), min=0.0)
    sm2 = torch.clamp(af_ratio * ((MYNN_RF1 - rf) / torch.clamp(MYNN_RF2 - rf, min=1.0e-5)) * sh2, min=0.0)
    q2_lev2 = torch.clamp(MYNN_B1 * (l_master ** 2) * sm2 * (1.0 - rf) * shear2, min=0.0)

    # Level-2.5 correction factor alpha_c and stability functions S_M, S_H (SCALE-RM Eqs. 8.110-8.121)
    alpha_c = torch.where(
        q2_lev2 > q2_int,
        q_int / torch.sqrt(torch.clamp(q2_lev2, min=1.0e-8)),
        torch.ones_like(q_int),
    )
    alpha_c2 = alpha_c * alpha_c

    g_m = torch.clamp((l_master * l_master / q2_int) * shear2, max=1.0e4)
    # Realizability lower/upper bounds on G_H = -(L^2 / q^2) * N^2 (Nakanishi & Niino, 2009)
    g_h = torch.clamp(-(l_master * l_master / q2_int) * n2, min=-4.0, max=0.028)

    phi1 = 1.0 - 3.0 * alpha_c2 * MYNN_A2 * MYNN_B2 * (1.0 - MYNN_C3) * g_h
    phi2 = 1.0 - 9.0 * alpha_c2 * MYNN_A1 * MYNN_A2 * (1.0 - MYNN_C2) * g_h
    phi3 = phi1 + 9.0 * alpha_c2 * (MYNN_A2 ** 2) * (1.0 - MYNN_C2) * (1.0 - MYNN_C5) * g_h
    phi4 = phi1 - 12.0 * alpha_c2 * MYNN_A1 * MYNN_A2 * (1.0 - MYNN_C2) * g_h
    phi5 = 6.0 * alpha_c2 * (MYNN_A1 ** 2) * g_m

    denom_25 = torch.clamp(phi2 * phi4 + phi5 * phi3, min=1.0e-5)
    s_m = torch.clamp(alpha_c * MYNN_A1 * (phi3 - 3.0 * MYNN_C1 * phi4) / denom_25, 0.0, 2.5)
    s_h = torch.clamp(alpha_c * MYNN_A2 * (phi2 + 3.0 * MYNN_C1 * phi5) / denom_25, 0.0, 3.0)

    k_m_int = l_master * q_int * s_m  # Eddy viscosity at k+1/2 [m^2 s^-1]
    k_h_int = l_master * q_int * s_h  # Eddy diffusivity at k+1/2 [m^2 s^-1]
    rho_int = 0.5 * (rho_d[1:] + rho_d[:-1])

    # Interpolate shear production, buoyancy production, and L to full levels for q^2 update (Eq. 8.163)
    prod_int = 2.0 * (k_m_int * shear2 - k_h_int * n2)
    prod_c = torch.zeros_like(q2)
    prod_c[0] = 0.5 * prod_int[0] + 2.0 * torch.clamp(
        (u_star ** 3) / (C.VON_KARMAN * torch.clamp(z_rel_c[0], min=1.0)) + sfc_buoy_flux,
        min=0.0,
    )
    prod_c[1:-1] = 0.5 * (prod_int[1:] + prod_int[:-1])
    prod_c[-1] = 0.5 * prod_int[-1]

    l_c = torch.empty_like(q2)
    l_c[0] = l_master[0]
    l_c[1:-1] = 0.5 * (l_master[1:] + l_master[:-1])
    l_c[-1] = l_master[-1]

    # Separate positive production (explicit RHS) from negative buoyancy destruction (implicit diagonal)
    prod_pos = torch.clamp(prod_c, min=0.0)
    buoy_sink_rate = torch.clamp(-prod_c, min=0.0) / q2

    # Implicit batched Thomas solve for q^2 (SCALE-RM Eqs. 8.163-8.168)
    mass_c = rho_d * dz_c
    flux_coeff_tke = dt * 3.0 * rho_int * k_m_int / dz_int

    dl_tke = torch.zeros_like(q2)
    du_tke = torch.zeros_like(q2)
    dl_tke[1:] = -flux_coeff_tke / mass_c[1:]
    du_tke[:-1] = -flux_coeff_tke / mass_c[:-1]
    d_tke = 1.0 - dl_tke - du_tke + dt * (2.0 * q_vel / (MYNN_B1 * l_c) + buoy_sink_rate)
    rhs_tke = q2 + dt * prod_pos
    qke_new = torch.clamp(solve_tridiagonal_batched(dl_tke, d_tke, du_tke, rhs_tke), min=2.0 * TKE_MIN)

    # Implicit batched Thomas solve for u, v, theta_l, q_n (SCALE-RM Eqs. 8.172-8.176)
    def _diffuse_implicit(field: torch.Tensor, k_diff: torch.Tensor, sfc_flux: torch.Tensor) -> torch.Tensor:
        fc = dt * rho_int * k_diff / dz_int
        dl = torch.zeros_like(field)
        du = torch.zeros_like(field)
        dl[1:] = -fc / mass_c[1:]
        du[:-1] = -fc / mass_c[:-1]
        diag = 1.0 - dl - du
        rhs = field.clone()
        rhs[0] = rhs[0] + dt * sfc_flux / mass_c[0]
        return solve_tridiagonal_batched(dl, diag, du, rhs)

    u_new = _diffuse_implicit(u, k_m_int, tau_x_sfc)
    v_new = _diffuse_implicit(v, k_m_int, tau_y_sfc)

    # Convert upward sensible heat flux [W m^-2] to kinematic rho_d * w'theta_l' flux [kg m^-2 K s^-1]
    cp_eff_0 = C.CP_D + C.CP_V * q_v[0]
    thl_flux_sfc = shf_sfc / (cp_eff_0 * exner[0])
    theta_l_new = _diffuse_implicit(theta_l, k_h_int, thl_flux_sfc)
    q_n_new = torch.clamp(_diffuse_implicit(q_n, k_h_int, evap_sfc), min=0.0)

    # Reconstruct updated moist potential temperature theta_m_c from diffused (theta_l_new, q_n_new)
    t_li_new = theta_l_new * exner
    temp_new, qv_new, _, _ = saturation_adjustment(t_li_new, pres, q_n_new)
    theta_m_new = (temp_new / exner) * (1.0 + C.RV_OVER_RD * qv_new)

    return u_new, v_new, theta_m_new, q_n_new, qke_new, k_m_int, k_h_int


# ---------------------------------------------------------------------------
# 5. Anisotropic 3D Smagorinsky-Lilly SGS Closure (CRM Horizontal/3D Eddy Mixing)
# ---------------------------------------------------------------------------
def scotti_aspect_factor(aspect: torch.Tensor) -> torch.Tensor:
    """Grid aspect-ratio correction `f(a)` of Scotti et al. (1993) (SCALE-RM Eqs. 8.35-8.39)."""
    a = torch.clamp(aspect, min=1.0)
    b1 = torch.atan(1.0 / a)
    b2 = 0.5 * math.pi - b1

    def _p2(z: torch.Tensor) -> torch.Tensor:
        return z * (0.98 + z * (0.073 + z * (-0.418 + 0.120 * z)))

    def _p3(z: torch.Tensor) -> torch.Tensor:
        return z * (0.976 + z * (0.188 + z * (1.169 + z * (0.755 - 0.151 * z))))

    def _p1(z: torch.Tensor) -> torch.Tensor:
        return 2.5 * _p2(z) - 1.5 * torch.pow(torch.clamp(torch.cos(z), min=0.0), 2.0 / 3.0) * torch.sin(z)

    bracket = (
        4.0 * _p1(b1) * torch.pow(a, 1.0 / 3.0)
        + 0.222 * _p2(b1) * torch.pow(a, -5.0 / 3.0)
        + 0.077 * _p3(b1) * torch.pow(a, -11.0 / 3.0)
        - 3.0 * b1
        + 4.0 * _p1(b2)
        + 0.222 * _p2(b2)
        + 0.077 * _p3(b2)
        - 3.0 * b2
    )
    return 1.736 * torch.pow(a, 1.0 / 3.0) * torch.pow(torch.clamp(bracket, min=0.1), -0.75)


def compute_smagorinsky_lilly_3d(
    strain_mag: torch.Tensor,
    n2: torch.Tensor,
    dx: float,
    dy: float,
    dz_c: torch.Tensor,
    c_s: float = 0.18,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Smagorinsky-Lilly 3D SGS eddy viscosity `nu_SGS` and diffusivity `D_SGS` [m^2 s^-1].

    Reference: Nishizawa et al. (2025, SCALE-RM v5.5.5, Sec. 8.1.2 Eqs. 8.25-8.40):
        ri = N^2 / |S|^2,  Pr_N = 0.7,  c = 16,  b = 40,  Ri_c = 0.25,  g = (1 - Pr_N) / Ri_c.
    """
    pr_n = 0.70
    ri_c = 0.25
    g_coef = (1.0 - pr_n) / ri_c

    dh = math.sqrt(dx * dy)
    aspect = torch.maximum(dh / torch.clamp(dz_c, min=1.0), dz_c / dh)
    delta_filter = torch.pow(dx * dy * torch.clamp(dz_c, min=1.0), 1.0 / 3.0)
    lam = scotti_aspect_factor(aspect) * delta_filter
    cs_lam2_s = ((c_s * lam) ** 2) * strain_mag

    ri = n2 / torch.clamp(strain_mag * strain_mag, min=1.0e-10)
    ri_unst = torch.clamp(ri, max=0.0)
    ri_stab = torch.clamp(ri, min=0.0, max=ri_c)

    nu_unst = cs_lam2_s * torch.sqrt(1.0 - 16.0 * ri_unst)
    d_unst = (cs_lam2_s / pr_n) * torch.sqrt(1.0 - 40.0 * ri_unst)

    stab_fac = torch.pow(1.0 - ri_stab / ri_c, 4.0)
    nu_stab = cs_lam2_s * stab_fac
    d_stab = (cs_lam2_s / pr_n) * stab_fac * (1.0 - g_coef * ri_stab)

    zero = torch.zeros_like(strain_mag)
    nu_sgs = torch.where(ri < 0.0, nu_unst, torch.where(ri < ri_c, nu_stab, zero))
    d_sgs = torch.where(ri < 0.0, d_unst, torch.where(ri < ri_c, d_stab, zero))
    return nu_sgs, d_sgs
