"""Two-variable diagnostic-phase bulk cloud microphysics and hydrometeor sedimentation.

Implements the complete mixed-phase single-moment microphysics of SAM:
- Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, 607-623, Appendix A Eqs. A8-A33
  and Appendix B Table of Constants):
  * Prognostic non-precipitating water mixing ratio `q_n_tot = q_v + q_c + q_i`
    (denoted `q_T` in KR03) and precipitating water mixing ratio `q_p = q_r + q_s + q_g`.
  * Temperature-ramp phase partitioning `omega_n(T)`, `omega_p(T)`, and `omega_g(T)`
    (Eqs. A8-A13).
  * Kessler cloud-water autoconversion (Eq. A30) and temperature-dependent cloud-ice
    aggregation (Eq. A31).
  * Continuous Marshall-Palmer accretion of cloud water and cloud ice by rain, snow,
    and graupel (Eqs. A27-A29).
  * Subsaturated rain evaporation and snow/graupel sublimation with ventilation factors
    (Eqs. A21-A26).
  * Cloud-ice sedimentation (Eqs. A32-A33) and precipitating hydrometeor fallout flux
    `P_m` (Eq. A19) solved via mass-conserving positive-definite upwind sub-stepping,
    partitioning surface fallout into liquid rainfall `precip_rain` [kg m^-2 s^-1] and
    solid snowfall/graupel `precip_snow` [kg m^-2 s^-1].
- Nishizawa, Tomita, & Team SCALE (2025, SCALE-RM v5.5.5, Sec. 2.6 Eq. 2.64 & Eq. 2.79)
  & Skamarock et al. (2012, Eq. 5):
  * Exact conversion of microphysical latent heating and moisture changes into the
    compressible flux variable `theta_m = theta * (1 + (R_v / R_d) * q_v)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import (
    can_use_crm_metal,
    dispatch_crm_microphysics_metal,
    solve_tridiagonal_batched,
)
from aoba.thermodynamics import (
    diagnose_pressure_temperature,
    esat_ice,
    esat_liquid,
    liquid_fraction_condensate,
    qsat_mixed,
    saturation_adjustment,
)

# ---------------------------------------------------------------------------
# Appendix B Constants of Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, p. 624)
# ---------------------------------------------------------------------------
RHO_REF_AIR: float = 1.29          # Reference air density rho_0 [kg m^-3]
RHO_RAIN: float = 1000.0           # Density of rain rho_r [kg m^-3]
RHO_SNOW: float = 100.0            # Density of snow rho_s [kg m^-3]
RHO_GRAUPEL: float = 400.0         # Density of graupel rho_g [kg m^-3]

N0_RAIN: float = 8.0e6             # Marshall-Palmer intercept for rain N_0r [m^-4]
N0_SNOW: float = 3.0e6             # Marshall-Palmer intercept for snow N_0s [m^-4]
N0_GRAUPEL: float = 4.0e6          # Marshall-Palmer intercept for graupel N_0g [m^-4]

A_FALL_RAIN: float = 842.0         # Fall-speed coefficient a_r [m^(1-b_r) s^-1]
B_FALL_RAIN: float = 0.80          # Fall-speed exponent b_r [-]
A_FALL_SNOW: float = 4.84          # Fall-speed coefficient a_s [m^(1-b_s) s^-1]
B_FALL_SNOW: float = 0.25          # Fall-speed exponent b_s [-]
A_FALL_GRAUPEL: float = 94.5       # Fall-speed coefficient a_g [m^(1-b_g) s^-1]
B_FALL_GRAUPEL: float = 0.50       # Fall-speed exponent b_g [-]

AF_RAIN: float = 0.78              # Ventilation constant a_fr [-]
BF_RAIN: float = 0.31              # Ventilation constant b_fr [-]
AF_SNOW: float = 0.65              # Ventilation constant a_fs [-]
BF_SNOW: float = 0.44              # Ventilation constant b_fs [-]
AF_GRAUPEL: float = 0.78           # Ventilation constant a_fg [-]
BF_GRAUPEL: float = 0.31           # Ventilation constant b_fg [-]

SHAPE_RAIN: float = 1.0            # Particle shape factor C_r [-]
SHAPE_SNOW: float = 2.0 / math.pi  # Particle shape factor C_s [-]
SHAPE_GRAUPEL: float = 1.0         # Particle shape factor C_g [-]

DIFF_VAPOR_0C: float = 2.210e-5    # Water-vapor diffusivity D_a at 0 degC [m^2 s^-1]
K_THERM_AIR_0C: float = 2.40e-2    # Air thermal conductivity K_a at 0 degC [J m^-1 s^-1 K^-1]
MU_AIR_0C: float = 1.717e-5        # Dynamic viscosity of air mu at 0 degC [kg m^-1 s^-1]

E_RC: float = 1.0                  # Collection efficiency of rain for cloud water [-]
E_SC: float = 1.0                  # Collection efficiency of snow for cloud water [-]
E_GC: float = 1.0                  # Collection efficiency of graupel for cloud water [-]
E_RI: float = 1.0                  # Collection efficiency of rain for cloud ice [-]
E_SI: float = 0.1                  # Collection efficiency of snow for cloud ice [-]
E_GI: float = 0.1                  # Collection efficiency of graupel for cloud ice [-]

QC_AUTOCONV_THRESH: float = 1.0e-3 # Threshold cloud liquid water q_co [kg kg^-1]
QI_AGGREG_THRESH: float = 1.0e-4   # Threshold cloud ice q_io [kg kg^-1]
RATE_AUTOCONV: float = 1.0e-3      # Autoconversion rate alpha [s^-1]
RATE_ICE_AGGREG: float = 1.0e-3    # Ice aggregation rate beta [s^-1]
VT_CLOUD_ICE: float = 0.40         # Cloud-ice terminal fall velocity V_TI [m s^-1]

# Precipitating phase-partition temperature thresholds (KR03 Appendix B, p. 624)
T0_PRECIP: float = 283.16          # T_0p above which precip is 100% rain [K]
T00_PRECIP: float = 268.16         # T_00p below which precip is 100% frozen [K]
T0_GRAUPEL: float = 283.16         # T_0g above which frozen precip is 100% graupel [K]
T00_GRAUPEL: float = 223.16        # T_00g below which frozen precip is 100% snow [K]


def _precompute_fall_coeff(a_m: float, b_m: float, rho_m: float, n0_m: float) -> float:
    """Precompute time-invariant prefactor in KR03 Eq. (A19)."""
    return (
        (a_m * math.gamma(4.0 + b_m) / 6.0)
        * ((math.pi * rho_m * n0_m) ** (-0.25 * b_m))
        * (RHO_REF_AIR ** 0.5)
    )


def _precompute_accr_coeff(
    a_m: float, b_m: float, rho_m: float, n0_m: float, eff_ml: float
) -> float:
    """Precompute time-invariant prefactor in KR03 Eq. (A29) without rho^(3+b)/4."""
    return (
        0.25
        * math.pi
        * a_m
        * n0_m
        * eff_ml
        * math.gamma(3.0 + b_m)
        * (RHO_REF_AIR ** 0.5)
        * ((math.pi * rho_m * n0_m) ** (-0.25 * (3.0 + b_m)))
    )


def _precompute_evap_coeffs(
    a_m: float, b_m: float, rho_m: float, n0_m: float, af_m: float, bf_m: float
) -> tuple[float, float]:
    """Precompute ventilation prefactors (without air density rho) in KR03 Eqs. (A25)-(A26)."""
    a_em = af_m * ((math.pi * rho_m * n0_m) ** (-0.5))
    b_em = (
        bf_m
        * ((a_m / MU_AIR_0C) ** 0.5)
        * math.gamma(0.5 * (5.0 + b_m))
        * (RHO_REF_AIR ** 0.25)
        * ((math.pi * rho_m * n0_m) ** (-0.125 * (5.0 + b_m)))
    )
    return a_em, b_em


_FALL_PREF_R = _precompute_fall_coeff(A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN)
_FALL_PREF_S = _precompute_fall_coeff(A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW)
_FALL_PREF_G = _precompute_fall_coeff(A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL)

_ACCR_RC = _precompute_accr_coeff(A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN, E_RC)
_ACCR_SC = _precompute_accr_coeff(A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW, E_SC)
_ACCR_GC = _precompute_accr_coeff(A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL, E_GC)

_ACCR_RI = _precompute_accr_coeff(A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN, E_RI)
_ACCR_SI = _precompute_accr_coeff(A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW, E_SI)
_ACCR_GI = _precompute_accr_coeff(A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL, E_GI)

_AEM_R, _BEM_R = _precompute_evap_coeffs(
    A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN, AF_RAIN, BF_RAIN
)
_AEM_S, _BEM_S = _precompute_evap_coeffs(
    A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW, AF_SNOW, BF_SNOW
)
_AEM_G, _BEM_G = _precompute_evap_coeffs(
    A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL, AF_GRAUPEL, BF_GRAUPEL
)

_MP_METAL_CONSTS: list[float] = [
    C.P_REF, C.R_D, C.R_V, C.CP_D, C.GAMMA_D, C.KAPPA_D,
    C.RV_OVER_RD, C.EPS_WATER, C.L_V0, C.L_F0, C.L_S0,
    C.E_TRIPLE, C.T_FREEZE, C.T_TRIPLE, C.T_ICE_ALL, C.T_LIQ_ALL,
    QC_AUTOCONV_THRESH, QI_AGGREG_THRESH, RATE_AUTOCONV, RATE_ICE_AGGREG,
    VT_CLOUD_ICE, T0_PRECIP, T00_PRECIP, T0_GRAUPEL, T00_GRAUPEL,
    DIFF_VAPOR_0C, K_THERM_AIR_0C,
    SHAPE_RAIN, SHAPE_SNOW, SHAPE_GRAUPEL,
    N0_RAIN, N0_SNOW, N0_GRAUPEL,
    B_FALL_RAIN, B_FALL_SNOW, B_FALL_GRAUPEL,
    _FALL_PREF_R, _FALL_PREF_S, _FALL_PREF_G,
    _ACCR_RC, _ACCR_SC, _ACCR_GC,
    _ACCR_RI, _ACCR_SI, _ACCR_GI,
    _AEM_R, _BEM_R, _AEM_S, _BEM_S, _AEM_G, _BEM_G,
]


@dataclass
class MicrophysicsResult:
    """Updated moist state and surface precipitation diagnostics after one microphysics step."""

    theta_m_c: torch.Tensor     # Updated full-level moist potential temperature [nz, ...] [K]
    q_n: torch.Tensor           # Updated non-precipitating mixing ratio q_v+q_c+q_i [nz, ...] [kg/kg]
    q_p: torch.Tensor           # Updated precipitating mixing ratio q_r+q_s+q_g [nz, ...] [kg/kg]
    temp_c: torch.Tensor        # Updated air temperature [nz, ...] [K]
    q_v: torch.Tensor           # Updated water-vapor mixing ratio [nz, ...] [kg/kg]
    q_c: torch.Tensor           # Diagnosed cloud liquid mixing ratio [nz, ...] [kg/kg]
    q_i: torch.Tensor           # Diagnosed cloud ice mixing ratio [nz, ...] [kg/kg]
    precip_rain: torch.Tensor   # Surface rainfall mass flux [kg m^-2 s^-1] [...]
    precip_snow: torch.Tensor   # Surface snowfall+graupel mass flux [kg m^-2 s^-1] [...]


def partition_precipitating_hydrometeors(
    temp: torch.Tensor,
    q_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition total precipitating water `q_p` into rain, snow, and graupel.

    Reference: Khairoutdinov & Randall (2003, Eqs. A10-A13):
        omega_p(T) = clip((T - T_00p) / (T_0p - T_00p), 0, 1)
        omega_g(T) = clip((T - T_00g) / (T_0g - T_00g), 0, 1)
        q_r = omega_p * q_p
        q_s = (1 - omega_p) * (1 - omega_g) * q_p
        q_g = (1 - omega_p) * omega_g * q_p.
    """
    qp_pos = torch.clamp(q_p, min=0.0)
    omega_p = torch.clamp((temp - T00_PRECIP) / (T0_PRECIP - T00_PRECIP), 0.0, 1.0)
    omega_g = torch.clamp((temp - T00_GRAUPEL) / (T0_GRAUPEL - T00_GRAUPEL), 0.0, 1.0)
    q_r = omega_p * qp_pos
    q_s = (1.0 - omega_p) * (1.0 - omega_g) * qp_pos
    q_g = (1.0 - omega_p) * omega_g * qp_pos
    return q_r, q_s, q_g, omega_p


def step_microphysics(
    rho_d: torch.Tensor,
    theta_m_c: torch.Tensor,
    q_n: torch.Tensor,
    q_p: torch.Tensor,
    q_v_prev: torch.Tensor,
    dz_c: torch.Tensor,
    dt: float,
) -> MicrophysicsResult:
    """Advance mixed-phase cloud microphysics and sedimentation by `dt` [s].

    All vertical profiles have leading dimension `nz` (`k = 0` lowest atmospheric
    layer adjacent to the surface, `k = nz - 1` model top).

    References
    ----------
    - Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, Eqs. A8-A33).
    - Nishizawa et al. (2025, SCALE-RM v5.5.5, Sec. 2.6 Eq. 2.64).
    """
    eps_small = 1.0e-15
    if can_use_crm_metal(rho_d) and (rho_d.ndim in (2, 4)):
        thm_new, qn_f, qp_s, t_new, qv_new, qc_new, qi_new, r_rate, s_rate = (
            dispatch_crm_microphysics_metal(
                rho_d, theta_m_c, q_n, q_p, q_v_prev, dz_c, dt, _MP_METAL_CONSTS
            )
        )
        return MicrophysicsResult(
            theta_m_c=thm_new,
            q_n=qn_f,
            q_p=qp_s,
            temp_c=t_new,
            q_v=qv_new,
            q_c=qc_new,
            q_i=qi_new,
            precip_rain=r_rate,
            precip_snow=s_rate,
        )
    qn = torch.clamp(q_n, min=0.0)
    qp = torch.clamp(q_p, min=0.0)
    qv_guess = torch.minimum(torch.clamp(q_v_prev, min=0.0), qn)

    # 1. Diagnose pressure and prior temperature, then recover liquid-ice equivalent T_li
    pres, exner, temp_old = diagnose_pressure_temperature(rho_d, theta_m_c, qv_guess)
    q_cond_old = torch.clamp(qn - qv_guess, min=0.0)
    omega_n_old = liquid_fraction_condensate(temp_old)
    t_li = temp_old - (
        omega_n_old * C.L_V0 + (1.0 - omega_n_old) * C.L_S0
    ) * q_cond_old / C.CP_D

    # First saturation adjustment before conversion rates
    temp, q_v, q_c, q_i = saturation_adjustment(t_li, pres, qn)
    q_r, q_s, q_g, omega_p_old = partition_precipitating_hydrometeors(temp, qp)

    # 2. Autoconversion (KR03 Eq. A30) & Ice Aggregation (KR03 Eq. A31)
    ice_temp_fac = torch.exp(0.025 * (temp - C.T_TRIPLE))
    dq_auto = RATE_AUTOCONV * torch.clamp(q_c - QC_AUTOCONV_THRESH, min=0.0)
    dq_aggr = RATE_ICE_AGGREG * ice_temp_fac * torch.clamp(q_i - QI_AGGREG_THRESH, min=0.0)

    # 3. Continuous Accretion of Cloud Water and Cloud Ice (KR03 Eqs. A28-A29)
    rho_r_q = torch.clamp(rho_d * q_r, min=0.0)
    rho_s_q = torch.clamp(rho_d * q_s, min=0.0)
    rho_g_q = torch.clamp(rho_d * q_g, min=0.0)

    exp_r = 0.25 * (3.0 + B_FALL_RAIN)
    exp_s = 0.25 * (3.0 + B_FALL_SNOW)
    exp_g = 0.25 * (3.0 + B_FALL_GRAUPEL)

    qr_pow = torch.pow(rho_r_q + eps_small, exp_r) - (eps_small ** exp_r)
    qs_pow = torch.pow(rho_s_q + eps_small, exp_s) - (eps_small ** exp_s)
    qg_pow = torch.pow(rho_g_q + eps_small, exp_g) - (eps_small ** exp_g)
    inv_sqrt_rho = torch.rsqrt(torch.clamp(rho_d, min=0.01))

    accr_c = inv_sqrt_rho * q_c * (_ACCR_RC * qr_pow + _ACCR_SC * qs_pow + _ACCR_GC * qg_pow)
    accr_i = (
        inv_sqrt_rho
        * ice_temp_fac
        * q_i
        * (_ACCR_RI * qr_pow + _ACCR_SI * qs_pow + _ACCR_GI * qg_pow)
    )

    # Limit condensate depletion to available q_c + q_i over dt
    total_conv_sink = (dq_auto + dq_aggr + accr_c + accr_i) * dt
    avail_cond = q_c + q_i
    conv_delta = torch.minimum(total_conv_sink, avail_cond)

    qn_mid = qn - conv_delta
    qp_mid = qp + conv_delta

    # 4. Precipitation Evaporation & Sublimation in Subsaturated Air (KR03 Eqs. A21-A26)
    qsat_mix, _ = qsat_mixed(temp, pres)
    subsat = torch.clamp(1.0 - q_v / torch.clamp(qsat_mix, min=1.0e-8), min=0.0)

    ew = esat_liquid(temp)
    ei = esat_ice(temp)
    # Thermodynamic factors A + B (KR03 Eqs. A23-A24)
    ab_liq = (C.L_V0 / (K_THERM_AIR_0C * temp)) * (
        C.L_V0 / (C.R_V * temp) - 1.0
    ) + (C.R_V * temp) / (DIFF_VAPOR_0C * torch.clamp(ew, min=1.0e-3))
    ab_ice = (C.L_S0 / (K_THERM_AIR_0C * temp)) * (
        C.L_S0 / (C.R_V * temp) - 1.0
    ) + (C.R_V * temp) / (DIFF_VAPOR_0C * torch.clamp(ei, min=1.0e-4))

    q_r_m, q_s_m, q_g_m, _ = partition_precipitating_hydrometeors(temp, qp_mid)
    rr_q = torch.clamp(rho_d * q_r_m, min=0.0)
    rs_q = torch.clamp(rho_d * q_s_m, min=0.0)
    rg_q = torch.clamp(rho_d * q_g_m, min=0.0)

    ev_exp_r = 0.125 * (5.0 + B_FALL_RAIN)
    ev_exp_s = 0.125 * (5.0 + B_FALL_SNOW)
    ev_exp_g = 0.125 * (5.0 + B_FALL_GRAUPEL)

    vent_r = _AEM_R * torch.sqrt(rr_q) + _BEM_R * (
        torch.pow(rr_q + eps_small, ev_exp_r) - (eps_small ** ev_exp_r)
    ) * (rho_d ** (-0.25))
    vent_s = _AEM_S * torch.sqrt(rs_q) + _BEM_S * (
        torch.pow(rs_q + eps_small, ev_exp_s) - (eps_small ** ev_exp_s)
    ) * (rho_d ** (-0.25))
    vent_g = _AEM_G * torch.sqrt(rg_q) + _BEM_G * (
        torch.pow(rg_q + eps_small, ev_exp_g) - (eps_small ** ev_exp_g)
    ) * (rho_d ** (-0.25))

    evap_rate_r = (2.0 * math.pi * SHAPE_RAIN * N0_RAIN / (rho_d * ab_liq)) * vent_r * subsat
    subl_rate_s = (2.0 * math.pi * SHAPE_SNOW * N0_SNOW / (rho_d * ab_ice)) * vent_s * subsat
    subl_rate_g = (2.0 * math.pi * SHAPE_GRAUPEL * N0_GRAUPEL / (rho_d * ab_ice)) * vent_g * subsat

    max_evap = torch.minimum(qp_mid, torch.clamp(qsat_mix - qn_mid, min=0.0))
    evap_delta = torch.minimum((evap_rate_r + subl_rate_s + subl_rate_g) * dt, max_evap)

    qn_after = qn_mid + evap_delta
    qp_after = qp_mid - evap_delta

    # Account for latent heat of autoconversion/accretion, phase changes & evaporation/sublimation.
    # Because T_li is defined relative to non-precipitating cloud condensate only
    # (c_pd * T_li = c_pd * T - L_eff_cloud * q_cond), moist static energy conservation
    # c_pd * T_li - L_eff_prec * q_p = const requires +L_eff_prec * (conv_delta - evap_delta)
    # plus fusion adjustment (L_eff_prec_cur - L_eff_prec_old) * q_p = L_f0 * (omega_p_old - omega_p_cur) * q_p:
    _, _, _, omega_p_cur = partition_precipitating_hydrometeors(temp, qp_after)
    l_eff_prec = omega_p_cur * C.L_V0 + (1.0 - omega_p_cur) * C.L_S0

    dt_li_micro = (
        (l_eff_prec / C.CP_D) * (conv_delta - evap_delta)
        + (C.L_F0 / C.CP_D) * (omega_p_old - omega_p_cur) * qp
    )
    t_li_after = t_li + dt_li_micro

    # 5. Unconditionally Stable Implicit Donor-Cell Sedimentation via Metal/PyTorch Batched Thomas Solver
    # (KR03 Eqs. A19 & A32-A33; SCALE-RM Sec. 6.1: exact mass conservation at arbitrary dt with zero CPU sync!)
    mass_air = torch.clamp(rho_d * dz_c, min=1.0e-4)  # Dry-air column mass per layer [kg m^-2]
    inv_sqrt_rho_sed = torch.rsqrt(torch.clamp(rho_d, min=0.01))

    f_exp_r = 1.0 + 0.25 * B_FALL_RAIN
    f_exp_s = 1.0 + 0.25 * B_FALL_SNOW
    f_exp_g = 1.0 + 0.25 * B_FALL_GRAUPEL

    q_r_s, q_s_s, q_g_s, omega_p_s = partition_precipitating_hydrometeors(temp, qp_after)
    rr = torch.clamp(rho_d * q_r_s, min=0.0)
    rs = torch.clamp(rho_d * q_s_s, min=0.0)
    rg = torch.clamp(rho_d * q_g_s, min=0.0)

    flux_r0 = _FALL_PREF_R * inv_sqrt_rho_sed * (torch.pow(rr + eps_small, f_exp_r) - (eps_small ** f_exp_r))
    flux_s0 = _FALL_PREF_S * inv_sqrt_rho_sed * (torch.pow(rs + eps_small, f_exp_s) - (eps_small ** f_exp_s))
    flux_g0 = _FALL_PREF_G * inv_sqrt_rho_sed * (torch.pow(rg + eps_small, f_exp_g) - (eps_small ** f_exp_g))
    flux_p0 = flux_r0 + flux_s0 + flux_g0

    # Effective precipitation mass-flux coefficient rate_p = rho_d * V_p [kg m^-2 s^-1]
    rate_p = torch.where(
        qp_after > 1.0e-12,
        torch.minimum(flux_p0 / torch.clamp(qp_after, min=1.0e-12), 15.0 * rho_d),
        torch.zeros_like(qp_after),
    )
    dl_sed = torch.zeros_like(qp_after)
    du_p = torch.zeros_like(qp_after)
    du_p[:-1] = -dt * rate_p[1:] / mass_air[:-1]
    diag_p = 1.0 + dt * rate_p / mass_air
    qp_sed = torch.clamp(solve_tridiagonal_batched(dl_sed, diag_p, du_p, qp_after), min=0.0)

    p_sfc_tot = rate_p[0] * qp_sed[0]  # Exact surface precipitation rate [kg m^-2 s^-1]
    rain_rate = omega_p_s[0] * p_sfc_tot
    snow_rate = (1.0 - omega_p_s[0]) * p_sfc_tot

    # Implicit cloud-ice sedimentation (KR03 Eqs. A32-A33)
    rate_qi = rho_d * VT_CLOUD_ICE
    du_qi = torch.zeros_like(q_i)
    du_qi[:-1] = -dt * rate_qi[1:] / mass_air[:-1]
    diag_qi = 1.0 + dt * rate_qi / mass_air
    qi_sed = torch.clamp(solve_tridiagonal_batched(dl_sed, diag_qi, du_qi, q_i), min=0.0)

    dqn_sed = qi_sed - q_i
    qn_final = torch.clamp(qn_after + dqn_sed, min=0.0)
    t_li_after = t_li_after - (C.L_S0 / C.CP_D) * dqn_sed
    snow_rate = snow_rate + rate_qi[0] * qi_sed[0]

    # 6. Final saturation adjustment & exact moist potential temperature reconstruction
    temp_new, qv_new, qc_new, qi_new = saturation_adjustment(t_li_after, pres, qn_final)
    theta_new = temp_new / exner
    theta_m_new = theta_new * (1.0 + C.RV_OVER_RD * qv_new)

    return MicrophysicsResult(
        theta_m_c=theta_m_new,
        q_n=qn_final,
        q_p=qp_sed,
        temp_c=temp_new,
        q_v=qv_new,
        q_c=qc_new,
        q_i=qi_new,
        precip_rain=rain_rate,
        precip_snow=snow_rate,
    )
