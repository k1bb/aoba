"""Compressible moist thermodynamics, mixed-phase saturation adjustment, and RCEMIP profiles.

References
----------
- Skamarock et al. (2012, Mon. Wea. Rev., 140, 3090-3105, Eqs. 5-6):
  Modified moist potential temperature theta_m = theta * (1 + (R_v / R_d) * q_v) and
  full compressible equation of state p = p_0 * (R_d * rho_d * theta_m / p_0)^(c_pd / c_vd).
- Seidel & Yang (2024, J. Climate, 37) & Gassmann (2013, Q. J. R. Meteorol. Soc., 139):
  Exact condensate loading and vapor-buoyancy density rho = rho_d * (1 + q_n + q_p).
- Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, 607-623, Eqs. A8-A14):
  Diagnostic liquid/ice condensate ramp omega_n(T) over [253.15, 273.16] K and
  Newton-Raphson all-or-nothing saturation adjustment conserving moist enthalpy.
- Bolton (1980, Mon. Wea. Rev., 108, 1046-1053) & Murphy & Koop (2005, QJRMS, 131):
  Saturation vapor pressure over liquid water e_sat,w(T) and hexagonal ice e_sat,i(T).
- Wing et al. (2018, Geosci. Model Dev., 11, 793-813, Eqs. 2-5):
  Analytic tropical moist hydrostatic sounding for RCEMIP initialization.
"""

from __future__ import annotations

import torch

from aoba import constants as C


def liquid_fraction_condensate(temp: torch.Tensor) -> torch.Tensor:
    """Linear liquid/ice partition function omega_n(T) in [0, 1].

    Reference: Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, Eq. A13):
        omega_n(T) = clip((T - T_00n) / (T_0n - T_00n), 0, 1)
    with T_00n = 253.15 K and T_0n = 273.16 K.
    """
    return torch.clamp(
        (temp - C.T_ICE_ALL) / (C.T_LIQ_ALL - C.T_ICE_ALL),
        0.0,
        1.0,
    )


def esat_liquid(temp: torch.Tensor) -> torch.Tensor:
    """Saturation vapor pressure over planar liquid water [Pa].

    Reference: Bolton (1980, Mon. Wea. Rev., 108, Eq. 10):
        e_sw(T) = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65)).
    Clamped safely against extreme cold temperatures to prevent denominator singularity.
    """
    tc = torch.clamp(temp, min=160.0) - C.T_FREEZE
    return C.E_TRIPLE * torch.exp(17.67 * tc / (tc + 243.50))


def esat_ice(temp: torch.Tensor) -> torch.Tensor:
    """Saturation vapor pressure over hexagonal ice [Pa].

    Reference: Murray (1967, J. Appl. Meteorol., 6, 203-204) / Tetens ice formula
    adopted in standard bulk microphysics:
        e_si(T) = 611.2 * exp(21.8745584 * (T - 273.15) / (T - 7.66)).
    """
    t_safe = torch.clamp(temp, min=120.0)
    return C.E_TRIPLE * torch.exp(21.8745584 * (t_safe - C.T_FREEZE) / (t_safe - 7.66))


def qsat_mixed(temp: torch.Tensor, pres: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed-phase saturation mixing ratio q_sat(T, p) [kg/kg dry air] and d(q_sat)/dT.

    Reference: Khairoutdinov & Randall (2003, Eq. A14):
        q_sat(T, p) = omega_n(T) * q_sat,w(T, p) + (1 - omega_n(T)) * q_sat,i(T, p),
    where q_sat,x(T, p) = eps * e_sx(T) / max(p - e_sx(T), 0.01 * p).
    """
    omega = liquid_fraction_condensate(temp)
    ew = esat_liquid(temp)
    ei = esat_ice(temp)

    denom_w = torch.clamp(pres - ew, min=0.05 * pres)
    denom_i = torch.clamp(pres - ei, min=0.05 * pres)
    qsw = C.EPS_WATER * ew / denom_w
    qsi = C.EPS_WATER * ei / denom_i
    qsat = omega * qsw + (1.0 - omega) * qsi

    # Analytic temperature derivative for fast Newton-Raphson convergence
    t_w = torch.clamp(temp, min=160.0)
    t_i = torch.clamp(temp, min=120.0)
    dew_dT = ew * (17.67 * 243.50) / ((t_w - C.T_FREEZE + 243.50) ** 2)
    dei_dT = ei * (21.8745584 * (C.T_FREEZE - 7.66)) / ((t_i - 7.66) ** 2)
    dqsw_dT = C.EPS_WATER * pres * dew_dT / (denom_w ** 2)
    dqsi_dT = C.EPS_WATER * pres * dei_dT / (denom_i ** 2)

    in_ramp = ((temp > C.T_ICE_ALL) & (temp < C.T_LIQ_ALL)).to(temp.dtype)
    domega_dT = in_ramp / (C.T_LIQ_ALL - C.T_ICE_ALL)
    dqsat_dT = omega * dqsw_dT + (1.0 - omega) * dqsi_dT + domega_dT * (qsw - qsi)
    return qsat, dqsat_dT


def diagnose_pressure_temperature(
    rho_d: torch.Tensor,
    theta_m: torch.Tensor,
    q_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnose pressure p [Pa], Exner function Pi [-], and temperature T [K].

    Reference: Skamarock et al. (2012, Mon. Wea. Rev., 140, Eqs. 5-6):
        theta_m = theta * (1 + (R_v / R_d) * q_v)
        p = p_0 * (R_d * rho_d * theta_m / p_0)^(c_pd / c_vd)
        Pi = (p / p_0)^(R_d / c_pd)
        T = Pi * theta = Pi * theta_m / (1 + (R_v / R_d) * q_v)
    """
    pres = C.P_REF * torch.pow(
        torch.clamp(C.R_D * rho_d * theta_m / C.P_REF, min=1.0e-8),
        C.GAMMA_D,
    )
    exner = torch.pow(pres / C.P_REF, C.KAPPA_D)
    theta = theta_m / (1.0 + C.RV_OVER_RD * torch.clamp(q_v, min=0.0))
    temp = exner * theta
    return pres, exner, temp


def saturation_adjustment(
    temp_init: torch.Tensor,
    pres: torch.Tensor,
    q_n: torch.Tensor,
    n_iter: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-or-nothing mixed-phase saturation adjustment at constant pressure.

    Conserves liquid/ice moist enthalpy (Khairoutdinov & Randall, 2003, Eqs. A3 & A8-A14):
        h_cond = c_pd * T - L_v0 * q_c - L_s0 * q_i = const
    given initial temperature `temp_init` (corresponding to `q_c = q_i = 0`, or prior
    state enthalpy when passed as vapor-equivalent temperature `T + (L_eff/c_pd)*q_cond_old`)
    and total non-precipitating mixing ratio `q_n = q_v + q_c + q_i`.

    Parameters
    ----------
    temp_init : torch.Tensor
        Unsaturated/liquid-ice potential temperature equivalent `T_li = T - (L_v0*q_c + L_s0*q_i)/c_pd` [K].
    pres : torch.Tensor
        Air pressure [Pa].
    q_n : torch.Tensor
        Total non-precipitating water mixing ratio `q_v + q_c + q_i` [kg/kg dry air].
    n_iter : int
        Number of damped Newton-Raphson iterations (4 iterations reach < 1e-6 K accuracy).

    Returns
    -------
    temp : torch.Tensor
        Adjusted air temperature [K].
    q_v : torch.Tensor
        Adjusted water vapor mixing ratio [kg/kg].
    q_c : torch.Tensor
        Diagnosed cloud liquid water mixing ratio [kg/kg].
    q_i : torch.Tensor
        Diagnosed cloud ice mixing ratio [kg/kg].
    """
    qn_pos = torch.clamp(q_n, min=0.0)
    t_li = temp_init

    # Check if unsaturated at T = T_li (where q_cond = 0)
    qsat0, _ = qsat_mixed(t_li, pres)
    is_sat = qn_pos > qsat0

    temp = t_li.clone()
    for _ in range(n_iter):
        qsat, dqsat_dT = qsat_mixed(temp, pres)
        q_cond = torch.clamp(qn_pos - qsat, min=0.0)
        omega = liquid_fraction_condensate(temp)
        l_eff = omega * C.L_V0 + (1.0 - omega) * C.L_S0

        # Residual f(T) = T - T_li - (L_eff(T) / c_pd) * (q_n - q_sat(T)) = 0
        res = temp - t_li - (l_eff / C.CP_D) * q_cond
        in_ramp = ((temp > C.T_ICE_ALL) & (temp < C.T_LIQ_ALL)).to(temp.dtype)
        dl_dT = in_ramp * (C.L_V0 - C.L_S0) / (C.T_LIQ_ALL - C.T_ICE_ALL)
        df_dT = 1.0 + (l_eff / C.CP_D) * dqsat_dT - (dl_dT / C.CP_D) * q_cond
        step = res / torch.clamp(df_dT, min=1.0)
        temp = torch.where(is_sat, temp - step, t_li)

    qsat_final, _ = qsat_mixed(temp, pres)
    q_cond_final = torch.where(is_sat, torch.clamp(qn_pos - qsat_final, min=0.0), torch.zeros_like(qn_pos))
    q_v = qn_pos - q_cond_final
    omega_final = liquid_fraction_condensate(temp)
    q_c = omega_final * q_cond_final
    q_i = (1.0 - omega_final) * q_cond_final
    return temp, q_v, q_c, q_i


def rcemip_analytic_sounding(
    z_c: torch.Tensor,
    z_w: torch.Tensor,
    sst: float = 300.0,
    p_sfc: float = 101480.0,
) -> dict[str, torch.Tensor]:
    """Generate the exact analytic tropical moist hydrostatic sounding of Wing et al. (2018).

    Reference: Wing et al. (2018, Geosci. Model Dev., 11, 793-813, Sec. 3.2.3, Eqs. 2-5 & Table 2):
    - q0 = 12.0 g/kg (SST=295 K), 18.65 g/kg (SST=300 K), 24.0 g/kg (SST=305 K);
      interpolated smoothly via 80% surface RH for arbitrary SST.
    - z_t = 15,000 m, z_q1 = 4,000 m, z_q2 = 7,500 m, gamma = 0.0067 K m^-1, p0 = 1014.8 hPa.

    Returns dictionary with hydrostatic consistent `pres_c`, `pres_w`, `temp_c`, `temp_w`,
    `qv_c`, `rho_d_c`, `theta_m_c`, and `theta_m_w`.
    """
    z_t = 15000.0
    z_q1 = 4000.0
    z_q2 = 7500.0
    q_t = 1.0e-14  # 1e-11 g/kg in kg/kg (Wing et al., 2018, Eq. 2)
    gamma = 0.0067  # Virtual temperature lapse rate [K m^-1] (Wing et al., 2018, Eq. 3)

    # Exact Table 2 values at 295/300/305 K; 80% RH formula otherwise
    if abs(sst - 295.0) < 1e-3:
        q0 = 0.01200
    elif abs(sst - 300.0) < 1e-3:
        q0 = 0.01865
    elif abs(sst - 305.0) < 1e-3:
        q0 = 0.02400
    else:
        es0 = float(esat_liquid(torch.tensor(sst)).item())
        q0 = 0.80 * C.EPS_WATER * es0 / (p_sfc - es0)

    tv0 = sst * (1.0 + 0.608 * q0)
    tv_t = tv0 - gamma * z_t
    p_t = p_sfc * ((tv_t / tv0) ** (C.GRAVITY / (C.R_D * gamma)))

    def _eval(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trop = z <= z_t
        q_trop = q0 * torch.exp(-z / z_q1) * torch.exp(-((z / z_q2) ** 2))
        q_spec = torch.where(trop, q_trop, torch.full_like(z, q_t))
        tv = torch.where(trop, tv0 - gamma * z, torch.full_like(z, tv_t))
        temp = tv / (1.0 + 0.608 * q_spec)
        p_trop = p_sfc * torch.pow(torch.clamp((tv0 - gamma * z) / tv0, min=1e-4), C.GRAVITY / (C.R_D * gamma))
        p_strat = p_t * torch.exp(-C.GRAVITY * (z - z_t) / (C.R_D * tv_t))
        pres = torch.where(trop, p_trop, p_strat)
        # Convert specific humidity q_spec to dry-air mixing ratio r_v = q / (1 - q)
        qv_mix = q_spec / torch.clamp(1.0 - q_spec, min=0.5)
        return pres, temp, qv_mix

    pres_c, temp_c, qv_c = _eval(z_c)
    pres_w, temp_w, qv_w = _eval(z_w)

    exner_c = torch.pow(pres_c / C.P_REF, C.KAPPA_D)
    exner_w = torch.pow(pres_w / C.P_REF, C.KAPPA_D)
    theta_m_c = (temp_c / exner_c) * (1.0 + C.RV_OVER_RD * qv_c)
    theta_m_w = (temp_w / exner_w) * (1.0 + C.RV_OVER_RD * qv_w)
    rho_d_c = pres_c / (C.R_D * temp_c * (1.0 + C.RV_OVER_RD * qv_c))

    return {
        "pres_c": pres_c,
        "pres_w": pres_w,
        "temp_c": temp_c,
        "temp_w": temp_w,
        "qv_c": qv_c,
        "rho_d_c": rho_d_c,
        "theta_m_c": theta_m_c,
        "theta_m_w": theta_m_w,
    }
