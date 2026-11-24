"""Map-Free 6-Class 2-Moment Bulk Cloud Microphysics (Morrison et al., 2005; Seifert & Beheng, 2006).

Selectable via `microphysics_scheme = "six_class_2m"` (alongside `"two_moment"` and `"one_moment"`).
Separates total precipitation into three independent prognostic classes with both mass and
number mixing ratios on the embedded 3D CRM (`[Nz, N_cells, Ny, Nx]`):
- **Cloud liquid (`q_c, n_c`)** & **Cloud ice (`q_i, n_i`)** (coupled to prognostic aerosol CCN/INP)
- **Rain (`q_r, n_r`)**: Khairoutdinov & Kogan (2000) warm autoconversion & accretion, melting,
  sub-cloud evaporation with exact latent cooling, fast fall speed (`v_r ~ 3.5-8 m s^-1`).
- **Snow (`q_s, n_s`)**: Ice crystal aggregation, Wegener-Bergeron-Findeisen vapor deposition,
  moderate fall speed (`v_s ~ 0.6-1.5 m s^-1`).
- **Graupel (`q_g, n_g`)**: Supercooled droplet riming onto snow/ice (`q_c + q_s -> q_g`) and
  supercooled rain freezing (`q_r -> q_g`), fast fall speed (`v_g ~ 2.5-7 m s^-1`). Feeds true
  prognostic `q_g` directly into CRM non-inductive charging lightning (`lightning.py`)!

All phase conversions use bounded exponential/implicit rates (zero Forward Euler blowup) and
unconditionally stable implicit upwind vertical sedimentation. Total precipitation mass is
reconciled into `q_p = q_r + q_s + q_g` for seamless host SCVT coupling.

References
----------
- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005): *JAS*, 62, 1665-1691.
- Seifert, A., & Beheng, K. D. (2006): *Meteorol. Atmos. Phys.*, 92, 45-66.
- Khairoutdinov, M., & Kogan, Y. (2000): *Mon. Wea. Rev.*, 128, 229-243.
"""

from __future__ import annotations

import math
import torch

from aoba import constants as C
from aoba.kernels import can_use_crm_metal, dispatch_microphysics_6class_metal
from aoba.microphysics import (
    CCN_DUST_SCALE_CM3,
    CCN_LAND_BASE_CM3,
    CCN_OCEAN_BASE_CM3,
    INP_DUST_ENHANCE,
)
from aoba.thermodynamics import diagnose_pressure_temperature

# Physical constants for 6-class 2-moment hydrometeors (Morrison et al., 2005, Table 1)
RHO_WATER: float = 1000.0
RHO_SNOW: float = 100.0
RHO_GRAUPEL: float = 400.0
Q_EPS: float = 1.0e-12


def init_six_class_hydrometeors(crm) -> None:
    """Lazily allocate 6-class 2-moment fields (`q_r, q_s, q_g, n_c, n_i, n_r, n_s, n_g`) on `CRMState`."""
    z = torch.zeros_like(crm.q_p)
    if crm.n_c is None:
        crm.n_c = z.clone()
    if crm.n_i is None:
        crm.n_i = z.clone()
    if getattr(crm, "q_r", None) is None:
        # Partition any pre-existing q_p cleanly into rain/snow on first step
        _, _, temp = diagnose_pressure_temperature(crm.rho_d, crm.theta_m, crm.q_v)
        f_liq = torch.clamp((temp - 268.15) / 8.0, 0.0, 1.0)
        q_pos = torch.clamp(crm.q_p, min=0.0)
        crm.q_r = q_pos * f_liq
        crm.q_s = q_pos * (1.0 - f_liq) * 0.75
        crm.q_g = q_pos * (1.0 - f_liq) * 0.25
        crm.n_r = torch.where(crm.q_r > 1.0e-9, crm.q_r / (4.0e-7), z.clone())
        crm.n_s = torch.where(crm.q_s > 1.0e-9, crm.q_s / (5.0e-8), z.clone())
        crm.n_g = torch.where(crm.q_g > 1.0e-9, crm.q_g / (2.0e-7), z.clone())


def _implicit_sediment_column(
    q: torch.Tensor,
    n_num: torch.Tensor,
    rho_d: torch.Tensor,
    dz_4d: torch.Tensor,
    v_fall: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unconditionally stable top-down implicit upwind sedimentation for mass `q` and number `n`."""
    nz = q.shape[0]
    m_k = torch.clamp(rho_d * dz_4d, min=1.0e-4)
    q_out = torch.empty_like(q)
    n_out = torch.empty_like(n_num)
    flux_q = torch.zeros_like(q[0])
    flux_n = torch.zeros_like(n_num[0])
    for k in range(nz - 1, -1, -1):
        sink = dt * rho_d[k] * v_fall[k]
        denom = m_k[k] + sink
        qk = (m_k[k] * q[k] + dt * flux_q) / denom
        nk = (m_k[k] * n_num[k] + dt * flux_n) / denom
        q_out[k] = qk
        n_out[k] = nk
        flux_q = rho_d[k] * v_fall[k] * qk
        flux_n = rho_d[k] * v_fall[k] * nk
    return q_out, n_out, flux_q


def step_microphysics_six_class_2m(
    rho_d: torch.Tensor,
    theta_m: torch.Tensor,
    q_v: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    q_r: torch.Tensor,
    q_s: torch.Tensor,
    q_g: torch.Tensor,
    n_c: torch.Tensor,
    n_i: torch.Tensor,
    n_r: torch.Tensor,
    n_s: torch.Tensor,
    n_g: torch.Tensor,
    dz_c: torch.Tensor,
    dt: float,
    *,
    is_ocean_sub: torch.Tensor | None = None,
    u_sfc_sub: torch.Tensor | None = None,
    q_dust_sub: torch.Tensor | None = None,
    ccn_aero_sub: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Advance 6-class 2-moment bulk microphysics by `dt` [s].

    Returns `(theta_m, q_v, q_c, q_i, q_r, q_s, q_g, n_c, n_i, n_r, n_s, n_g, precip_rain, precip_snow)`.
    """
    if can_use_crm_metal(rho_d) and rho_d.numel() > 0:
        return dispatch_microphysics_6class_metal(
            rho_d=rho_d,
            theta_m=theta_m,
            q_v=q_v,
            q_c=q_c,
            q_i=q_i,
            q_r=q_r,
            q_s=q_s,
            q_g=q_g,
            n_c=n_c,
            n_i=n_i,
            n_r=n_r,
            n_s=n_s,
            n_g=n_g,
            dz_c=dz_c,
            dt=dt,
            is_ocean_sub=is_ocean_sub,
            u_sfc_sub=u_sfc_sub,
            q_dust_sub=q_dust_sub,
            ccn_aero_sub=ccn_aero_sub,
            ccn_ocean_cm3=CCN_OCEAN_BASE_CM3,
            ccn_land_cm3=CCN_LAND_BASE_CM3,
            ccn_dust_scale_cm3=CCN_DUST_SCALE_CM3,
            inp_dust_enhance=INP_DUST_ENHANCE,
        )

    nz, nc, ny, nx = rho_d.shape
    dtype = rho_d.dtype
    dz_4d = dz_c.view(nz, nc, 1, 1) if dz_c.ndim == 2 else dz_c.view(nz, 1, 1, 1)

    pres, pi_ex, temp = diagnose_pressure_temperature(rho_d, theta_m, q_v)
    qc = torch.clamp(q_c, min=0.0)
    qi = torch.clamp(q_i, min=0.0)
    qr = torch.clamp(q_r, min=0.0)
    qs = torch.clamp(q_s, min=0.0)
    qg = torch.clamp(q_g, min=0.0)
    qv = torch.clamp(q_v, min=1.0e-7)

    # 1. Aerosol-aware cloud droplet & ice crystal number activation (Morrison et al., 2005)
    if is_ocean_sub is not None:
        is_ocn = is_ocean_sub.view(1, nc, ny, nx).to(dtype)
    else:
        is_ocn = torch.full((1, nc, ny, nx), 0.7, device=rho_d.device, dtype=dtype)
    ccn_base_cm3 = is_ocn * CCN_OCEAN_BASE_CM3 + (1.0 - is_ocn) * CCN_LAND_BASE_CM3
    if u_sfc_sub is not None:
        ccn_base_cm3 = ccn_base_cm3 + is_ocn * 18.0 * torch.clamp(u_sfc_sub.view(1, nc, ny, nx) / 10.0, 0.0, 2.5)
    if q_dust_sub is not None:
        ccn_base_cm3 = ccn_base_cm3 + CCN_DUST_SCALE_CM3 * torch.clamp(q_dust_sub * 1.0e7, 0.0, 5.0)
    if ccn_aero_sub is not None:
        ccn_base_cm3 = ccn_base_cm3 + torch.clamp(ccn_aero_sub.view(1, nc, ny, nx) * 1.0e-6, 0.0, 250.0)
    ccn_m3 = torch.clamp(ccn_base_cm3 * 1.0e6, min=1.5e7, max=6.0e8)
    nc_target = torch.where(qc > 1.0e-7, ccn_m3 / torch.clamp(rho_d, min=0.05), torch.zeros_like(qc))
    # Smooth relaxation of n_c toward activated CCN when cloud liquid present
    nc_cur = torch.where(qc > 1.0e-7, torch.clamp(0.5 * torch.clamp(n_c, min=0.0) + 0.5 * nc_target, min=5.0e6, max=1.0e9), torch.zeros_like(qc))

    # Cooper (1986) temperature-dependent ice nucleation enhanced by dust INP
    t_sub = torch.clamp(C.T_FREEZE - temp, min=0.0, max=40.0)
    inp_fac = 1.0 + (INP_DUST_ENHANCE * torch.clamp(q_dust_sub * 1.0e7, 0.0, 5.0) if q_dust_sub is not None else 0.0)
    ni_target = torch.where(
        qi > 1.0e-8,
        torch.clamp(5.0 * torch.exp(0.304 * t_sub) * inp_fac / torch.clamp(rho_d, min=0.05), min=1.0e2, max=5.0e6),
        torch.zeros_like(qi),
    )
    ni_cur = torch.where(qi > 1.0e-8, torch.clamp(0.5 * torch.clamp(n_i, min=0.0) + 0.5 * ni_target, min=1.0e2, max=5.0e6), torch.zeros_like(qi))

    nr_cur = torch.clamp(n_r, min=0.0)
    ns_cur = torch.clamp(n_s, min=0.0)
    ng_cur = torch.clamp(n_g, min=0.0)

    # 2. Warm-rain KK00 autoconversion & accretion (q_c -> q_r)
    nc_cm3 = torch.clamp(nc_cur * rho_d * 1.0e-6, min=10.0, max=800.0)
    k_auto = 1350.0 * torch.pow(torch.clamp(qc, min=1.0e-9), 1.47) * torch.pow(nc_cm3, -1.79)
    k_accr = 67.0 * torch.pow(torch.clamp(qr, min=0.0), 1.15) * torch.pow(torch.clamp(qc, min=1.0e-9), 0.15)
    f_c2r = -torch.expm1(-torch.clamp((k_auto + k_accr) * dt, max=10.0))
    dq_c2r = torch.where(qc > 1.0e-6, f_c2r * qc, torch.zeros_like(qc))
    qc = qc - dq_c2r
    qr = qr + dq_c2r
    nr_cur = nr_cur + dq_c2r / 2.5e-7  # Mean new raindrop mass ~2.5e-7 kg (r ~ 400 um)

    # 3. Cold-cloud pathways:
    # (a) Ice crystal aggregation (q_i -> q_s)
    k_agg = 1.5e-3 * torch.exp(torch.clamp(0.025 * (temp - C.T_FREEZE), min=-3.0, max=0.0)) * torch.clamp(qi / 1.0e-4, 0.0, 5.0)
    dq_i2s = torch.where(qi > 2.0e-5, (-torch.expm1(-torch.clamp(k_agg * dt, max=5.0))) * qi, torch.zeros_like(qi))
    qi = qi - dq_i2s
    qs = qs + dq_i2s
    ns_cur = ns_cur + dq_i2s / 4.0e-8

    # (b) Riming of supercooled droplets onto snow/graupel -> GRAUPEL (Morrison et al., 2005)
    is_supercooled = (temp < C.T_FREEZE).to(dtype)
    k_rime_s = is_supercooled * 3.5 * torch.pow(torch.clamp(qs, min=0.0), 0.80)
    k_rime_g = is_supercooled * 5.0 * torch.pow(torch.clamp(qg, min=0.0), 0.80)
    dq_c2g = torch.where(
        (qc > 1.0e-5) & ((qs + qg) > 1.0e-6) & (temp < C.T_FREEZE),
        (-torch.expm1(-torch.clamp((k_rime_s + k_rime_g) * dt, max=8.0))) * qc,
        torch.zeros_like(qc),
    )
    # Heavy riming converts snow mass into graupel
    dq_s2g = torch.where(
        (qc > 2.0e-4) & (qs > 1.0e-5) & (temp < C.T_FREEZE),
        (-torch.expm1(-torch.clamp(2.5 * qc * dt * 100.0, max=5.0))) * qs,
        torch.zeros_like(qs),
    )
    qc = qc - dq_c2g
    qs = qs - dq_s2g
    qg = qg + dq_c2g + dq_s2g
    ng_cur = ng_cur + (dq_c2g + dq_s2g) / 2.0e-7

    # (c) Freezing of supercooled rain -> graupel (T < 268.15 K) with latent heat of fusion
    f_frz_rate = torch.where(temp < 268.15, 0.015 * torch.exp(torch.clamp(0.25 * (268.15 - temp), max=6.0)), torch.zeros_like(temp))
    dq_r2g = (-torch.expm1(-torch.clamp(f_frz_rate * dt, max=10.0))) * qr
    qr = qr - dq_r2g
    qg = qg + dq_r2g
    ng_cur = ng_cur + dq_r2g / 2.5e-7
    dtheta_lat = (C.L_F0 / (C.CP_D * torch.clamp(pi_ex, min=0.1))) * (dq_c2g + dq_r2g)

    # (d) Melting of snow & graupel -> rain (T > 273.15 K) with latent cooling
    t_above = torch.clamp(temp - C.T_FREEZE, min=0.0)
    f_melt_s = -torch.expm1(-torch.clamp(0.008 * t_above * dt, max=10.0))
    f_melt_g = -torch.expm1(-torch.clamp(0.004 * t_above * dt, max=10.0))
    dq_s2r = f_melt_s * qs
    dq_g2r = f_melt_g * qg
    qs = qs - dq_s2r
    qg = qg - dq_g2r
    qr = qr + dq_s2r + dq_g2r
    nr_cur = nr_cur + (dq_s2r + dq_g2r) / 3.0e-7
    dtheta_lat = dtheta_lat - (C.L_F0 / (C.CP_D * torch.clamp(pi_ex, min=0.1))) * (dq_s2r + dq_g2r)

    # (e) Sub-cloud rain evaporation in subsaturated air (cold pool generation!)
    e_sat = 611.2 * torch.exp(torch.clamp(17.67 * (temp - C.T_FREEZE) / torch.clamp(temp - 29.65, min=20.0), min=-30.0, max=15.0))
    q_sat = torch.clamp(C.EPS_WATER * e_sat / torch.clamp(pres - e_sat, min=100.0), min=1.0e-6, max=0.15)
    subsat = torch.clamp(q_sat - qv, min=0.0)
    k_evap = 0.004 * torch.pow(torch.clamp(qr * rho_d, min=0.0), 0.55)
    dq_evap = torch.minimum(
        (-torch.expm1(-torch.clamp(k_evap * dt, max=5.0))) * qr,
        0.5 * subsat,
    )
    qr = qr - dq_evap
    qv = qv + dq_evap
    dtheta_lat = dtheta_lat - (C.L_V0 / (C.CP_D * torch.clamp(pi_ex, min=0.1))) * dq_evap

    theta_new = torch.clamp(theta_m + dtheta_lat, min=150.0, max=2500.0)

    # 4. Class-specific terminal velocities [m s^-1] (air-density corrected sqrt(rho_0/rho))
    rho_fac = torch.sqrt(torch.clamp(1.2 / torch.clamp(rho_d, min=0.05), min=0.5, max=6.0))
    v_rain = torch.clamp(6.5 * torch.pow(torch.clamp(rho_d * qr * 1.0e3, min=1.0e-6), 0.14) * rho_fac, min=1.0, max=9.5)
    v_snow = torch.clamp(1.1 * torch.pow(torch.clamp(rho_d * qs * 1.0e3, min=1.0e-6), 0.10) * rho_fac, min=0.3, max=2.2)
    v_graup = torch.clamp(4.2 * torch.pow(torch.clamp(rho_d * qg * 1.0e3, min=1.0e-6), 0.18) * rho_fac, min=0.8, max=8.5)

    # 5. Unconditionally stable implicit sedimentation for Rain, Snow, & Graupel
    qr, nr_cur, pr_rain = _implicit_sediment_column(qr, nr_cur, rho_d, dz_4d, v_rain, dt)
    qs, ns_cur, pr_snow_only = _implicit_sediment_column(qs, ns_cur, rho_d, dz_4d, v_snow, dt)
    qg, ng_cur, pr_graup = _implicit_sediment_column(qg, ng_cur, rho_d, dz_4d, v_graup, dt)

    # Enforce consistent physical bounds on number concentrations
    nr_cur = torch.where(qr > 1.0e-10, torch.clamp(nr_cur, min=1.0, max=1.0e6), torch.zeros_like(nr_cur))
    ns_cur = torch.where(qs > 1.0e-10, torch.clamp(ns_cur, min=1.0, max=1.0e6), torch.zeros_like(ns_cur))
    ng_cur = torch.where(qg > 1.0e-10, torch.clamp(ng_cur, min=1.0, max=1.0e5), torch.zeros_like(ng_cur))

    return (
        theta_new, qv, qc, qi, qr, qs, qg,
        nc_cur, ni_cur, nr_cur, ns_cur, ng_cur,
        pr_rain, pr_snow_only + pr_graup,
    )
