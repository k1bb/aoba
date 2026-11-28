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

AOBA Formulation Reference: Sec. 5.2-5.4, Eqs. (5.1)-(5.12) (SAM 1-Moment & Aerosol-Aware 2-Moment KK00).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import (
    can_use_crm_metal,
    dispatch_crm_microphysics_2m_metal,
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

QC_AUTOCONV_THRESH: float = 1.0e-3 # KR03 fallback threshold cloud liquid water q_co [kg kg^-1] (used if N_d <= 0)
QI_AGGREG_THRESH: float = 1.0e-4   # Threshold cloud ice q_io [kg kg^-1]
RATE_AUTOCONV: float = 1.0e-3      # KR03 fallback autoconversion rate alpha [s^-1]
RATE_ICE_AGGREG: float = 1.0e-3    # Ice aggregation rate beta [s^-1]
VT_CLOUD_ICE: float = 0.75         # Cloud-ice terminal fall velocity V_TI [m s^-1] (Heymsfield & Iaquinta, 2000)
CLOUD_DROPLET_CONC_CM3: float = 60.0 # Cloud droplet number conc. N_d [cm^-3] (Tomita 2008 / Berry 1968; <=0 -> KR03)

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
    CLOUD_DROPLET_CONC_CM3,
]


def recompute_derived_constants() -> None:
    """Refresh precomputed KR03 fall/accretion/ventilation coefficients and Metal constant buffer."""
    global _FALL_PREF_R, _FALL_PREF_S, _FALL_PREF_G
    global _ACCR_RC, _ACCR_SC, _ACCR_GC, _ACCR_RI, _ACCR_SI, _ACCR_GI
    global _AEM_R, _BEM_R, _AEM_S, _BEM_S, _AEM_G, _BEM_G
    _FALL_PREF_R = _precompute_fall_coeff(A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN)
    _FALL_PREF_S = _precompute_fall_coeff(A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW)
    _FALL_PREF_G = _precompute_fall_coeff(A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL)
    _ACCR_RC = _precompute_accr_coeff(A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN, E_RC)
    _ACCR_SC = _precompute_accr_coeff(A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW, E_SC)
    _ACCR_GC = _precompute_accr_coeff(A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL, E_GC)
    _ACCR_RI = _precompute_accr_coeff(A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN, E_RI)
    _ACCR_SI = _precompute_accr_coeff(A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW, E_SI)
    _ACCR_GI = _precompute_accr_coeff(A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL, E_GI)
    _AEM_R, _BEM_R = _precompute_evap_coeffs(A_FALL_RAIN, B_FALL_RAIN, RHO_RAIN, N0_RAIN, AF_RAIN, BF_RAIN)
    _AEM_S, _BEM_S = _precompute_evap_coeffs(A_FALL_SNOW, B_FALL_SNOW, RHO_SNOW, N0_SNOW, AF_SNOW, BF_SNOW)
    _AEM_G, _BEM_G = _precompute_evap_coeffs(A_FALL_GRAUPEL, B_FALL_GRAUPEL, RHO_GRAUPEL, N0_GRAUPEL, AF_GRAUPEL, BF_GRAUPEL)
    _MP_METAL_CONSTS[:] = [
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
        CLOUD_DROPLET_CONC_CM3,
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
    n_c: torch.Tensor | None = None  # Optional 2-moment cloud droplet number mixing ratio [kg^-1]
    n_i: torch.Tensor | None = None  # Optional 2-moment cloud ice crystal number mixing ratio [kg^-1]


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

    # 2. Autoconversion (Berry 1968 / Tomita 2008, SCALE-RM Eqs. 8.221-8.222 when N_d > 0;
    #    KR03 Eq. A30 threshold fallback when N_d <= 0) & Ice Aggregation (KR03 Eq. A31)
    ice_temp_fac = torch.exp(0.025 * (temp - C.T_TRIPLE))
    if CLOUD_DROPLET_CONC_CM3 > 0.0:
        n_drop = CLOUD_DROPLET_CONC_CM3
        d_drop = max(0.146 - 5.964e-2 * math.log(max(n_drop, 1.0) / 2000.0), 0.05)
        rho_qc = rho_d * q_c
        dq_auto = (16.7 * d_drop * (rho_qc * rho_qc) * q_c) / (
            5.0 * d_drop * rho_qc + 3.6e-5 * n_drop
        )
    else:
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


# ---------------------------------------------------------------------------
# 2-Moment Cloud Microphysics (Morrison et al., 2005; Khairoutdinov & Kogan, 2000;
# DeMott et al., 2010; Martin et al., 1994; Wyser, 1998)
# ---------------------------------------------------------------------------
# Map-free natural pre-industrial CCN & INP activation parameters
CCN_OCEAN_BASE_CM3: float = 50.0     # Clean pre-industrial marine CCN baseline [cm^-3] (Andreae, 2009; Miles et al., 2000)
CCN_LAND_BASE_CM3: float = 95.0      # Clean pre-industrial continental biogenic CCN [cm^-3] (Andreae, 2009)
CCN_DUST_SCALE_CM3: float = 120.0    # Additional CCN per 10 ug m^-3 mineral dust [cm^-3]
INP_DUST_ENHANCE: float = 4.0        # DeMott et al. (2010) dust immersion INP enhancement factor [-]
KK00_AUTO_COEF_LES: float = 1350.0   # Khairoutdinov & Kogan (2000, Eq. 29) 50 m LES warm-rain autoconversion
KK00_AUTO_COEF: float = 42000.0      # Reference 4 km CRM subgrid LWC-variance-enhanced KK00 (MG08 / Boutle et al., 2014)
KK00_DX_LES_M: float = 50.0          # Reference LES grid spacing [m]
KK00_SCALE_EXP: float = 0.784486     # Scale-aware power-law exponent bridging 50 m LES (1350) -> 4 km CRM (42000)


def compute_kk00_auto_coef(dx_m: float = 4000.0) -> float:
    """Scale-aware KK00 warm-rain autoconversion coefficient as a function of grid spacing `dx_m` [m].

    Smoothly bridges Khairoutdinov & Kogan (2000, Eq. 29) at 50 m LES (`1350`) to subgrid LWC-variance-enhanced
    cloud-resolving grids (Morrison & Gettelman, 2008; Boutle et al., 2014; `42000` at 4 km).
    AOBA Formulation Eqs. (5.11) & (5.13).
    """
    dx_eff = max(float(dx_m), KK00_DX_LES_M)
    c_raw = KK00_AUTO_COEF_LES * ((dx_eff / KK00_DX_LES_M) ** KK00_SCALE_EXP)
    return min(max(c_raw, KK00_AUTO_COEF_LES), 120000.0)
LIQ_DISP_KAPPA: float = 0.72         # Martin et al. (1994) droplet spectral dispersion kappa = (r_v/r_e)^3 [-]
RHO_CLOUD_ICE: float = 500.0         # Effective cloud-ice crystal bulk density [kg m^-3] (Wyser, 1998)
RE_LIQ_MIN_UM: float = 4.0           # Lower bound on prognostic liquid effective radius [um]
RE_LIQ_MAX_UM: float = 25.0          # Upper bound on prognostic liquid effective radius [um]
RE_ICE_MIN_UM: float = 15.0          # Lower bound on prognostic ice effective radius [um]
RE_ICE_MAX_UM: float = 120.0         # Upper bound on prognostic ice effective radius [um]
INP_MIN_HET_M3: float = 3.0e4        # Secondary ice multiplication floor in mixed-phase clouds [m^-3] (Field et al., 2017)


def diagnose_cloud_effective_radii(
    rho_d: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    n_c: torch.Tensor | None,
    n_i: torch.Tensor | None,
    re_liq_fallback: float = 10.0,
    re_ice_fallback: float = 48.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnose 3D cloud liquid & ice effective radii `[um]` from 2-moment `(q_c, n_c)` & `(q_i, n_i)`.

    References
    ----------
    - Martin, Johnson, & Spice (1994, J. Atmos. Sci., 51, 1823-1842):
      `r_e,liq = (3 * rho * q_c / (4 * pi * rho_w * kappa * (rho * n_c)))^(1/3)`
    - Wyser (1998, J. Climate, 11, 1793-1802) & Morrison et al. (2005):
      `r_e,ice = 1.15 * (3 * rho * q_i / (4 * pi * rho_i * (rho * n_i)))^(1/3)`
    """
    if n_c is None:
        re_l = torch.full_like(q_c, float(re_liq_fallback))
    else:
        has_l = (q_c > 1.0e-7) & (n_c > 1.0e4)
        # r_e [m] = (3 * q_c / (4 * pi * rho_w * kappa * n_c))^(1/3) (rho cancels!)
        denom_l = (4.0 * math.pi / 3.0) * RHO_RAIN * LIQ_DISP_KAPPA * torch.clamp(n_c, min=1.0e4)
        re_l_raw = torch.pow(torch.clamp(q_c, min=1.0e-9) / denom_l, 1.0 / 3.0) * 1.0e6
        re_l = torch.where(
            has_l,
            torch.clamp(re_l_raw, min=RE_LIQ_MIN_UM, max=RE_LIQ_MAX_UM),
            torch.full_like(q_c, float(re_liq_fallback)),
        )

    if n_i is None:
        re_i = torch.full_like(q_i, float(re_ice_fallback))
    else:
        has_i = (q_i > 1.0e-8) & (n_i > 1.0e1)
        denom_i = (4.0 * math.pi / 3.0) * RHO_CLOUD_ICE * torch.clamp(n_i, min=1.0e1)
        re_i_raw = 0.85 * torch.pow(torch.clamp(q_i, min=1.0e-10) / denom_i, 1.0 / 3.0) * 1.0e6
        re_i = torch.where(
            has_i,
            torch.clamp(re_i_raw, min=RE_ICE_MIN_UM, max=RE_ICE_MAX_UM),
            torch.full_like(q_i, float(re_ice_fallback)),
        )
    return re_l, re_i


def compute_natural_ccn_inp(
    rho_d: torch.Tensor,
    temp: torch.Tensor,
    w_eff: torch.Tensor | None = None,
    is_ocean: torch.Tensor | None = None,
    u_sfc: torch.Tensor | None = None,
    q_dust: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnose map-free natural activated droplet `N_act` [m^-3] and ice `N_inp` [m^-3].

    References
    ----------
    - Twomey (1959), O'Dowd et al. (1997), Andreae (2009): natural marine sea-salt/DMS
      vs continental biogenic + mineral dust CCN activation.
    - Cooper (1986) & DeMott et al. (2010, PNAS, 107, 11217-11222): temperature- and
      dust-dependent heterogeneous ice nucleation + homogeneous cirrus freezing (< 235 K).
    """
    if is_ocean is not None:
        ocn_b = is_ocean
        while ocn_b.ndim < rho_d.ndim:
            ocn_b = ocn_b.unsqueeze(0)
        u10 = torch.zeros_like(rho_d[0]) if u_sfc is None else torch.clamp(u_sfc, min=0.0, max=35.0)
        while u10.ndim < rho_d.ndim:
            u10 = u10.unsqueeze(0)
        # O'Dowd et al. (1997): wind-driven marine sea-salt enhancement
        n_ccn_ocn = torch.clamp(CCN_OCEAN_BASE_CM3 * torch.exp(0.055 * u10), min=30.0, max=140.0)
        n_ccn_land = torch.full_like(n_ccn_ocn, CCN_LAND_BASE_CM3)
        n_ccn_cm3 = torch.where(ocn_b, n_ccn_ocn, n_ccn_land)
    else:
        n_ccn_cm3 = torch.full_like(rho_d, CLOUD_DROPLET_CONC_CM3)

    if q_dust is not None:
        dust_ug_m3 = torch.clamp(rho_d * q_dust * 1.0e9, min=0.0)  # [ug m^-3]
        n_ccn_cm3 = n_ccn_cm3 + CCN_DUST_SCALE_CM3 * torch.pow(dust_ug_m3 / 10.0 + 1.0e-6, 0.5)
    else:
        dust_ug_m3 = torch.zeros_like(rho_d)

    # Updraft supersaturation activation factor (Twomey, 1959; Morrison et al., 2005)
    if w_eff is not None:
        w_pos = torch.clamp(w_eff, min=0.05, max=20.0)
        act_fac = torch.pow(w_pos / (w_pos + 0.35), 0.45)
    else:
        act_fac = 0.85
    n_act_m3 = torch.clamp(n_ccn_cm3 * act_fac, min=15.0, max=450.0) * 1.0e6  # [m^-3]

    # Cooper (1986) heterogeneous INP + DeMott et al. (2010) dust enhancement + secondary ice floor
    dt_super = torch.clamp(C.T_FREEZE - temp, min=0.0, max=40.0)
    n_cooper = 5.0 * torch.exp(0.304 * dt_super)  # [m^-3]
    dust_inp_fac = 1.0 + INP_DUST_ENHANCE * torch.pow(dust_ug_m3 / 10.0, 0.4)
    n_het = torch.clamp(n_cooper * dust_inp_fac, min=INP_MIN_HET_M3, max=5.0e5)
    # Smooth transition to homogeneous cirrus freezing (~2e5 m^-3) below 235 K
    hom_frac = torch.clamp((238.0 - temp) / 6.0, 0.0, 1.0)
    n_inp_m3 = torch.clamp((1.0 - hom_frac) * n_het + hom_frac * 2.0e5, min=INP_MIN_HET_M3, max=5.0e5)
    return n_act_m3, n_inp_m3


def _get_mp2m_metal_consts(kk00_auto_coef: float = KK00_AUTO_COEF) -> list[float]:
    return _MP_METAL_CONSTS + [
        CCN_OCEAN_BASE_CM3, CCN_LAND_BASE_CM3, CCN_DUST_SCALE_CM3, INP_DUST_ENHANCE,
        float(kk00_auto_coef), RHO_CLOUD_ICE, RE_ICE_MIN_UM, RE_ICE_MAX_UM, INP_MIN_HET_M3,
    ]


def step_microphysics_two_moment(
    rho_d: torch.Tensor,
    theta_m_c: torch.Tensor,
    q_n: torch.Tensor,
    q_p: torch.Tensor,
    q_v_prev: torch.Tensor,
    n_c_in: torch.Tensor,
    n_i_in: torch.Tensor,
    dz_c: torch.Tensor,
    dt: float,
    w_eff: torch.Tensor | None = None,
    is_ocean: torch.Tensor | None = None,
    u_sfc: torch.Tensor | None = None,
    q_dust: torch.Tensor | None = None,
    dx_m: float = 4000.0,
) -> MicrophysicsResult:
    """Advance 2-moment mixed-phase cloud microphysics `(q_n, q_p, n_c, n_i)` by `dt` [s].

    Combines exact mass-conserving saturation adjustment, scale-aware Khairoutdinov & Kogan
    (2000, Eq. 29) 2-moment warm-rain autoconversion `dq_r/dt = C_KK00(dx) * q_c^2.47 * N_c,cm^-3^(-1.79)`,
    Morrison et al. (2005) droplet/ice number activation, scavenging, & sedimentation, and
    KR03 accretion, evaporation/sublimation, and implicit donor-cell precipitation fallout.
    """
    eps_small = 1.0e-15
    kk00_coef = compute_kk00_auto_coef(dx_m)
    if can_use_crm_metal(rho_d) and (rho_d.ndim in (2, 4)):
        thm_new, qn_f, qp_s, t_new, qv_new, qc_new, qi_new, nc_f, ni_f, r_rate, s_rate = (
            dispatch_crm_microphysics_2m_metal(
                rho_d=rho_d, theta_m_c=theta_m_c, q_n=q_n, q_p=q_p, q_v_prev=q_v_prev,
                n_c_in=n_c_in, n_i_in=n_i_in, dz_c=dz_c, dt=dt,
                w_eff=w_eff, is_ocean=is_ocean, u_sfc=u_sfc, q_dust=q_dust,
                mp_consts=_get_mp2m_metal_consts(kk00_coef),
            )
        )
        return MicrophysicsResult(
            theta_m_c=thm_new, q_n=qn_f, q_p=qp_s, temp_c=t_new,
            q_v=qv_new, q_c=qc_new, q_i=qi_new,
            precip_rain=r_rate, precip_snow=s_rate, n_c=nc_f, n_i=ni_f,
        )
    qn = torch.clamp(q_n, min=0.0)
    qp = torch.clamp(q_p, min=0.0)
    nc_cur = torch.clamp(n_c_in, min=0.0)
    ni_cur = torch.clamp(n_i_in, min=0.0)
    qv_guess = torch.minimum(torch.clamp(q_v_prev, min=0.0), qn)

    # 1. Diagnose pressure, temperature, and liquid-ice equivalent temperature T_li
    pres, exner, temp_old = diagnose_pressure_temperature(rho_d, theta_m_c, qv_guess)
    q_cond_old = torch.clamp(qn - qv_guess, min=0.0)
    omega_n_old = liquid_fraction_condensate(temp_old)
    t_li = temp_old - (omega_n_old * C.L_V0 + (1.0 - omega_n_old) * C.L_S0) * q_cond_old / C.CP_D

    temp, q_v, q_c, q_i = saturation_adjustment(t_li, pres, qn)
    q_r, q_s, q_g, omega_p_old = partition_precipitating_hydrometeors(temp, qp)

    # 2. Natural CCN / INP activation (nucleation where condensate is present)
    n_act_m3, n_inp_m3 = compute_natural_ccn_inp(rho_d, temp, w_eff=w_eff, is_ocean=is_ocean, u_sfc=u_sfc, q_dust=q_dust)
    rho_safe = torch.clamp(rho_d, min=1.0e-3)
    nc_target = n_act_m3 / rho_safe
    ni_target = n_inp_m3 / rho_safe

    # Nucleate newly formed cloud droplets/crystals smoothly while respecting advected number
    nc_act = torch.where(q_c > 1.0e-7, torch.maximum(nc_cur, nc_target), torch.zeros_like(nc_cur))
    ni_act = torch.where(q_i > 1.0e-8, torch.maximum(ni_cur, ni_target), torch.zeros_like(ni_cur))
    # Cap physical droplet/ice number to realistic mean-radius bounds
    nc_cm3 = torch.clamp(rho_safe * nc_act * 1.0e-6, min=10.0, max=600.0)

    # 3. Scale-aware Khairoutdinov & Kogan (2000, Eq. 29) 2-moment autoconversion + KR03 ice aggregation
    dq_auto = torch.where(
        q_c > 1.0e-6,
        kk00_coef * torch.pow(torch.clamp(q_c, min=1.0e-9), 2.47) * torch.pow(nc_cm3, -1.79),
        torch.zeros_like(q_c),
    )
    ice_temp_fac = torch.exp(0.025 * (temp - C.T_TRIPLE))
    dq_aggr = RATE_ICE_AGGREG * ice_temp_fac * torch.clamp(q_i - QI_AGGREG_THRESH, min=0.0)

    # 4. Continuous Accretion (KR03 Eqs. A28-A29)
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
    accr_i = inv_sqrt_rho * ice_temp_fac * q_i * (_ACCR_RI * qr_pow + _ACCR_SI * qs_pow + _ACCR_GI * qg_pow)

    # Separate liquid & ice conversion caps for exact mass conservation and number scavenging
    liq_sink = torch.minimum((dq_auto + accr_c) * dt, q_c)
    ice_sink = torch.minimum((dq_aggr + accr_i) * dt, q_i)
    conv_delta = liq_sink + ice_sink

    # Deplete n_c & n_i proportionally to condensate loss (Morrison et al., 2005)
    frac_c_rem = torch.where(q_c > 1.0e-9, torch.clamp(1.0 - liq_sink / torch.clamp(q_c, min=1.0e-9), 0.0, 1.0), torch.zeros_like(q_c))
    frac_i_rem = torch.where(q_i > 1.0e-9, torch.clamp(1.0 - ice_sink / torch.clamp(q_i, min=1.0e-9), 0.0, 1.0), torch.zeros_like(q_i))
    nc_mid = nc_act * frac_c_rem
    ni_mid = ni_act * frac_i_rem

    qn_mid = qn - conv_delta
    qp_mid = qp + conv_delta

    # 5. Subsaturated precipitation evaporation & sublimation (KR03 Eqs. A21-A26)
    qsat_mix, _ = qsat_mixed(temp, pres)
    subsat = torch.clamp(1.0 - q_v / torch.clamp(qsat_mix, min=1.0e-8), min=0.0)
    ew = esat_liquid(temp)
    ei = esat_ice(temp)
    ab_liq = (C.L_V0 / (K_THERM_AIR_0C * temp)) * (C.L_V0 / (C.R_V * temp) - 1.0) + (C.R_V * temp) / (DIFF_VAPOR_0C * torch.clamp(ew, min=1.0e-3))
    ab_ice = (C.L_S0 / (K_THERM_AIR_0C * temp)) * (C.L_S0 / (C.R_V * temp) - 1.0) + (C.R_V * temp) / (DIFF_VAPOR_0C * torch.clamp(ei, min=1.0e-4))

    q_r_m, q_s_m, q_g_m, _ = partition_precipitating_hydrometeors(temp, qp_mid)
    rr_q = torch.clamp(rho_d * q_r_m, min=0.0)
    rs_q = torch.clamp(rho_d * q_s_m, min=0.0)
    rg_q = torch.clamp(rho_d * q_g_m, min=0.0)

    ev_exp_r = 0.125 * (5.0 + B_FALL_RAIN)
    ev_exp_s = 0.125 * (5.0 + B_FALL_SNOW)
    ev_exp_g = 0.125 * (5.0 + B_FALL_GRAUPEL)
    rho_m025 = torch.pow(torch.clamp(rho_d, min=0.01), -0.25)

    evap_r = (2.0 * math.pi * SHAPE_RAIN * N0_RAIN / (rho_safe * ab_liq)) * subsat * (
        _AEM_R * torch.sqrt(rr_q) + _BEM_R * rho_m025 * (torch.pow(rr_q + eps_small, ev_exp_r) - eps_small ** ev_exp_r)
    )
    subl_s = (2.0 * math.pi * SHAPE_SNOW * N0_SNOW / (rho_safe * ab_ice)) * subsat * (
        _AEM_S * torch.sqrt(rs_q) + _BEM_S * rho_m025 * (torch.pow(rs_q + eps_small, ev_exp_s) - eps_small ** ev_exp_s)
    )
    subl_g = (2.0 * math.pi * SHAPE_GRAUPEL * N0_GRAUPEL / (rho_safe * ab_ice)) * subsat * (
        _AEM_G * torch.sqrt(rg_q) + _BEM_G * rho_m025 * (torch.pow(rg_q + eps_small, ev_exp_g) - eps_small ** ev_exp_g)
    )
    max_evap = torch.minimum(qp_mid, torch.clamp(qsat_mix - q_v, min=0.0))
    evap_delta = torch.minimum((evap_r + subl_s + subl_g) * dt, max_evap)

    qn_after = qn_mid + evap_delta
    qp_after = qp_mid - evap_delta

    _, _, _, omega_p_cur = partition_precipitating_hydrometeors(temp, qp_after)
    l_eff_prec = omega_p_cur * C.L_V0 + (1.0 - omega_p_cur) * C.L_S0
    dt_li_micro = (l_eff_prec / C.CP_D) * (conv_delta - evap_delta) + (C.L_F0 / C.CP_D) * (omega_p_old - omega_p_cur) * qp
    t_li_after = t_li + dt_li_micro

    # 6. Implicit Donor-Cell Sedimentation (precipitation + cloud ice mass & number)
    mass_air = torch.clamp(rho_d * dz_c, min=1.0e-4)
    inv_sqrt_rho_sed = torch.rsqrt(torch.clamp(rho_d, min=0.01))
    f_exp_r = 1.0 + 0.25 * B_FALL_RAIN
    f_exp_s = 1.0 + 0.25 * B_FALL_SNOW
    f_exp_g = 1.0 + 0.25 * B_FALL_GRAUPEL

    q_r_s, q_s_s, q_g_s, omega_p_s = partition_precipitating_hydrometeors(temp, qp_after)
    rr = torch.clamp(rho_d * q_r_s, min=0.0)
    rs = torch.clamp(rho_d * q_s_s, min=0.0)
    rg = torch.clamp(rho_d * q_g_s, min=0.0)
    flux_p0 = (
        _FALL_PREF_R * inv_sqrt_rho_sed * (torch.pow(rr + eps_small, f_exp_r) - eps_small ** f_exp_r)
        + _FALL_PREF_S * inv_sqrt_rho_sed * (torch.pow(rs + eps_small, f_exp_s) - eps_small ** f_exp_s)
        + _FALL_PREF_G * inv_sqrt_rho_sed * (torch.pow(rg + eps_small, f_exp_g) - eps_small ** f_exp_g)
    )
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

    p_sfc_tot = rate_p[0] * qp_sed[0]
    rain_rate = omega_p_s[0] * p_sfc_tot
    snow_rate = (1.0 - omega_p_s[0]) * p_sfc_tot

    # Size-aware cloud-ice terminal velocity: larger crystals (smaller n_i/q_i) fall faster!
    _, re_i_um = diagnose_cloud_effective_radii(rho_d, q_c, q_i, nc_mid, ni_mid)
    vt_ice_eff = torch.clamp(VT_CLOUD_ICE * (re_i_um / 48.0), min=0.10, max=1.20)
    rate_qi = rho_d * vt_ice_eff
    du_qi = torch.zeros_like(q_i)
    du_qi[:-1] = -dt * rate_qi[1:] / mass_air[:-1]
    diag_qi = 1.0 + dt * rate_qi / mass_air
    qi_sed = torch.clamp(solve_tridiagonal_batched(dl_sed, diag_qi, du_qi, q_i), min=0.0)
    ni_sed = torch.clamp(solve_tridiagonal_batched(dl_sed, diag_qi, du_qi, ni_mid), min=0.0)

    dqn_sed = qi_sed - q_i
    qn_final = torch.clamp(qn_after + dqn_sed, min=0.0)
    t_li_after = t_li_after - (C.L_S0 / C.CP_D) * dqn_sed
    snow_rate = snow_rate + rate_qi[0] * qi_sed[0]

    # 7. Final saturation adjustment & consistent number concentration cleanup
    temp_new, qv_new, qc_new, qi_new = saturation_adjustment(t_li_after, pres, qn_final)
    theta_new = temp_new / exner
    theta_m_new = theta_new * (1.0 + C.RV_OVER_RD * qv_new)

    # Zero number where cloud completely evaporated/sublimated; enforce minimum floor where cloud exists
    nc_final = torch.where(qc_new > 1.0e-7, torch.clamp(nc_mid, min=10.0e6 / rho_safe, max=600.0e6 / rho_safe), torch.zeros_like(nc_mid))
    ni_final = torch.where(qi_new > 1.0e-8, torch.clamp(ni_sed, min=INP_MIN_HET_M3 / rho_safe, max=5.0e5 / rho_safe), torch.zeros_like(ni_sed))

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
        n_c=nc_final,
        n_i=ni_final,
    )
