"""Lookup-table-free four-band analytical two-stream radiative transfer solver.

Implements the unified tridiagonal multiple-scattering & thermal emission algorithm of
Toon et al. (1989) across four physical spectral bands:
- SW1 (0.25-0.69 um, UV + Visible / PAR; solar weight 0.459760):
  Background ozone profile O3(p) (Wing et al., 2018, Eq. 1), analytical UV + Chappuis O3
  absorption (Lacis & Hansen, 1974, Eqs. 8-12), molecular Rayleigh scattering, delta-Quadrature
  liquid/ice cloud scattering (Slingo, 1989, Table 1; Ebert & Curry, 1992, Table 2), and
  surface visible albedo alpha_vis. Transmitted surface flux yields PAR directly.
- SW2 (0.69-4.00 um, Near-Infrared; solar weight 0.540240):
  Pressure/temperature-scaled H2O absorption (Lacis & Hansen, 1974, Eq. 21), liquid/ice
  cloud NIR scattering & absorption (Slingo, 1989; Ebert & Curry, 1992), and surface alpha_nir.
- LW1 (Rotational-vibrational H2O + CO2 band; Planck weight 1 - beta = 0.680):
  Hemispheric-mean two-stream solver (Toon et al., 1989, Table 1 & Eqs. 25-48) with
  pressure-broadened H2O line optical depth, logarithmic CO2 optical depth (Weaver &
  Ramanathan, 1995; Koll & Cronin, 2018; Sherwood et al., 2020), and liquid/ice cloud
  infrared absorption (Stephens, 1978; Ebert & Curry, 1992, Table 3).
- LW2 (8.0-12.5 um Atmospheric Window; Planck weight beta = 0.320):
  H2O e-type self-continuum and weak line-wing absorption (Koll & Cronin, 2018; Roberts et al.,
  1976), CO2 window-edge overlap, and liquid/ice cloud window absorption (Ebert & Curry, 1992,
  Table 3 Band 2).

All vertical inputs and outputs follow AOBA's bottom-up convention:
    k = 0 is the lowest atmospheric layer (interface 0 = surface),
    k = N_z - 1 is the topmost layer (interface N_z = top of atmosphere).
Internally, arrays are flipped to Toon et al. (1989)'s top-down convention (`n = 0` at TOA)
and flipped back upon return.

AOBA Formulation Reference: Sec. 1.3 Eqs. (1.1)-(1.4) & Sec. 4.2-4.3 Eqs. (4.1)-(4.6) (Two-Stream & Four-Band).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import can_use_crm_metal, dispatch_rad_four_band_metal, solve_tridiagonal_batched


@dataclass
class RadiationOutput:
    """Radiative fluxes [W m^-2], heating rates [K s^-1], and surface PAR."""

    sw_flux_up: torch.Tensor       # Total upward SW flux at interfaces [nz + 1, ...]
    sw_flux_dn: torch.Tensor       # Total downward SW flux at interfaces [nz + 1, ...]
    lw_flux_up: torch.Tensor       # Total upward LW flux at interfaces [nz + 1, ...]
    lw_flux_dn: torch.Tensor       # Total downward LW flux at interfaces [nz + 1, ...]
    q_rad_sw: torch.Tensor         # Shortwave radiative heating rate [nz, ...] [K s^-1]
    q_rad_lw: torch.Tensor         # Longwave radiative heating rate [nz, ...] [K s^-1]
    q_rad_net: torch.Tensor        # Net radiative heating rate [nz, ...] [K s^-1]
    toa_insolation: torch.Tensor   # Incident TOA solar flux [W m^-2] [...]
    toa_asr: torch.Tensor          # Absorbed shortwave radiation at TOA [W m^-2] [...]
    toa_olr: torch.Tensor          # Outgoing longwave radiation at TOA [W m^-2] [...]
    sfc_sw_net: torch.Tensor       # Net downward SW absorbed by surface [W m^-2] [...]
    sfc_lw_net: torch.Tensor       # Net downward LW absorbed by surface (LW_dn - LW_up) [W m^-2] [...]
    sfc_par_wm2: torch.Tensor      # Downward surface PAR energy flux [W m^-2] [...]
    sfc_ppfd: torch.Tensor         # Downward surface PPFD [umol photons m^-2 s^-1] [...]
    sfc_sw1_dir_dn: torch.Tensor   # Downward surface direct-beam SW1 flux [W m^-2] [...]
    sfc_sw1_dif_dn: torch.Tensor   # Downward surface diffuse SW1 flux [W m^-2] [...]
    sfc_sw2_dir_dn: torch.Tensor   # Downward surface direct-beam SW2 flux [W m^-2] [...]
    sfc_sw2_dif_dn: torch.Tensor   # Downward surface diffuse SW2 flux [W m^-2] [...]


# ---------------------------------------------------------------------------
# 1. Solar Orbital & Zenith Geometry (Davis et al., 2017, Eqs. 2-8; MIROC6 Sec. 3.6.15)
# ---------------------------------------------------------------------------
def compute_solar_zenith(
    lat_rad: torch.Tensor,
    lon_rad: torch.Tensor,
    time_sec: float,
    solar_constant: float = C.SOLAR_CONSTANT,
    obliquity_deg: float = C.OBLIQUITY_DEG,
    eccentricity: float = C.ECCENTRICITY,
    perihelion_lon_deg: float = C.PERIHELION_LON_DEG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cosine of solar zenith angle mu_0 in [0, 1] and distance-scaled TOA S_0(t).

    References
    ----------
    - Berger (1978, J. Atmos. Sci., 35, 2362-2367) & Davis et al. (2017, GMD, 10, Eqs. 2-8).
    - MIROC6 AGCM Document Writing Team (2021, Sec. 3.6.15).
    """
    day_of_year = (time_sec / C.SECONDS_PER_DAY) % C.DAYS_PER_YEAR
    # Mean anomaly relative to vernal equinox (approx day 80.0)
    lam_m = 2.0 * math.pi * (day_of_year - 80.0) / C.DAYS_PER_YEAR
    omega_p = math.radians(perihelion_lon_deg)
    e = eccentricity
    # First-order true heliocentric longitude lambda (Davis et al., 2017, Eq. 4)
    nu_m = lam_m - omega_p
    nu = nu_m + (2.0 * e - 0.25 * e**3) * math.sin(nu_m) + 1.25 * e**2 * math.sin(2.0 * nu_m)
    lam = nu + omega_p

    eps = math.radians(obliquity_deg)
    sin_delta = math.sin(eps) * math.sin(lam)
    cos_delta = math.sqrt(max(1.0 - sin_delta * sin_delta, 0.0))

    # Earth-Sun distance factor (d_0 / d)^2 (Davis et al., 2017, Eq. 5)
    rho_rel = (1.0 + e * math.cos(nu)) / (1.0 - e * e)
    s_inst = solar_constant * (rho_rel * rho_rel)

    # Local solar hour angle H = 2*pi*(t_UTC / 86400 - 0.5) + lon_rad
    utc_frac = (time_sec % C.SECONDS_PER_DAY) / C.SECONDS_PER_DAY
    hour_angle = 2.0 * math.pi * (utc_frac - 0.5) + lon_rad

    cos_z = torch.sin(lat_rad) * sin_delta + torch.cos(lat_rad) * cos_delta * torch.cos(hour_angle)
    mu0 = torch.clamp(cos_z, min=0.0, max=1.0)
    s_toa = torch.full_like(mu0, s_inst)
    return mu0, s_toa


# ---------------------------------------------------------------------------
# 2. Background Ozone & Lacis-Hansen (1974) Solar Gas Absorptances
# ---------------------------------------------------------------------------
def background_ozone_vmr(pres_pa: torch.Tensor) -> torch.Tensor:
    """Horizontally uniform ozone volume mixing ratio [mol/mol] from pressure [Pa].

    Reference: Wing et al. (2018, Geosci. Model Dev., 11, Eq. 1):
        O3_ppmv(p) = g_1 * p_hPa^(g_2) * exp(-p_hPa / g_3)
    with g_1 = 3.6478, g_2 = 0.83209, g_3 = 11.3515 hPa.
    """
    p_hpa = torch.clamp(pres_pa * 0.01, min=1.0e-3)
    o3_ppmv = C.O3_G1 * torch.pow(p_hpa, C.O3_G2) * torch.exp(-p_hpa / C.O3_G3)
    return o3_ppmv * 1.0e-6


def _lacis_hansen_ozone_abs(x_cm_ntp: torch.Tensor) -> torch.Tensor:
    """Fraction of TOTAL solar irradiance absorbed by ozone path x [cm-NTP].

    Reference: Lacis & Hansen (1974, J. Atmos. Sci., 31, p. 122, Eqs. 8-10):
        A_oz^vis(x) = 0.02118 x / (1 + 0.042 x + 0.000323 x^2)
        A_oz^uv(x)  = 1.082 x / (1 + 138.6 x)^0.805 + 0.0658 x / (1 + (103.6 x)^3)
        A_oz(x)     = A_oz^uv(x) + A_oz^vis(x)
    """
    x = torch.clamp(x_cm_ntp, min=0.0)
    a_vis = (0.02118 * x) / (1.0 + 0.042 * x + 0.000323 * x * x)
    a_uv = (1.082 * x) / torch.pow(1.0 + 138.6 * x, 0.805) + (0.0658 * x) / (
        1.0 + torch.pow(103.6 * x, 3)
    )
    return a_uv + a_vis


def _lacis_hansen_wv_abs(y_cm: torch.Tensor) -> torch.Tensor:
    """Fraction of TOTAL solar irradiance absorbed by scaled water vapor path y [cm].

    Reference: Lacis & Hansen (1974, J. Atmos. Sci., 31, p. 126, Eq. 21):
        A_wv(y) = 2.9 y / ((1 + 141.5 y)^0.635 + 5.925 y)
    """
    y = torch.clamp(y_cm, min=0.0)
    return (2.9 * y) / (torch.pow(1.0 + 141.5 * y, 0.635) + 5.925 * y)


# ---------------------------------------------------------------------------
# 3. Cloud Optical Properties (Slingo, 1989; Ebert & Curry, 1992)
# ---------------------------------------------------------------------------
def cloud_optical_sw(
    lwp_gm2: torch.Tensor,
    iwp_gm2: torch.Tensor,
    re_liq_um: float = 10.0,
    re_ice_um: float = 48.0,
    cloud_sw_inhomog: float = 0.85,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute liquid + ice cloud optical properties (tau, omega0, g) for SW1 and SW2.

    References
    ----------
    - Slingo (1989, J. Atmos. Sci., 46, p. 1420, Eqs. 1-3 & Table 1, 4-band version; Eq. 25):
      Band 1 (SW1, 0.25-0.69 um, weight 0.459760):
        a1 = 2.817e-2, b1 = 1.305, c1 = -5.62e-8, d1 = 1.63e-7, e1 = 0.829, f1 = 2.482e-3
      Bands 2-4 solar-weighted average (SW2, 0.69-4.00 um, weight 0.540240):
        a2 = 2.456e-2, b2 = 1.400, c2 = 0.01260, d2 = 9.01e-4, e2 = 0.783, f2 = 5.01e-3
    - Ebert & Curry (1992, J. Geophys. Res., 97, p. 3833, Eqs. 6-8 & Table 2):
      Band 1 (SW1, 0.25-0.70 um, weight 0.4686):
        ai = 3.448e-3, bi = 2.431, ci = 1.0e-5, di = 0.0, ei = 0.7661, fi = 5.851e-4
      Bands 2-5 solar-weighted average (SW2, 0.70-3.50 um, weight 0.5314):
        ai = 3.448e-3, bi = 2.431, ci = 0.02437, di = 2.52e-4, ei = 0.7867, fi = 5.90e-4
    - Cahalan et al. (1994, J. Atmos. Sci., 51, 2434-2455): Subgrid cloud optical inhomogeneity
      scaling factor `cloud_sw_inhomog` (chi_SW, typically 0.70-0.78) reducing plane-parallel albedo bias.
    """
    lwp = torch.clamp(lwp_gm2, min=0.0)
    iwp = torch.clamp(iwp_gm2, min=0.0)
    chi = float(cloud_sw_inhomog)

    # SW1 liquid + ice
    tau_l1 = chi * lwp * (2.817e-2 + 1.305 / re_liq_um)
    om_l1 = min(1.0 - (-5.62e-8 + 1.63e-7 * re_liq_um), 0.999999)
    g_l1 = 0.829 + 2.482e-3 * re_liq_um

    tau_i1 = chi * iwp * (3.448e-3 + 2.431 / re_ice_um)
    om_i1 = min(1.0 - (1.0e-5 + 0.0 * re_ice_um), 0.999999)
    g_i1 = 0.7661 + 5.851e-4 * re_ice_um

    tau_1 = tau_l1 + tau_i1
    scat_1 = tau_l1 * om_l1 + tau_i1 * om_i1
    om_1 = torch.where(tau_1 > 1.0e-8, scat_1 / torch.clamp(tau_1, min=1.0e-8), torch.zeros_like(tau_1))
    g_1 = torch.where(
        scat_1 > 1.0e-8,
        (tau_l1 * om_l1 * g_l1 + tau_i1 * om_i1 * g_i1) / torch.clamp(scat_1, min=1.0e-8),
        torch.zeros_like(tau_1),
    )

    # SW2 liquid + ice
    tau_l2 = chi * lwp * (2.456e-2 + 1.400 / re_liq_um)
    om_l2 = max(min(1.0 - (0.01260 + 9.01e-4 * re_liq_um), 0.999999), 0.0)
    g_l2 = 0.783 + 5.01e-3 * re_liq_um

    tau_i2 = chi * iwp * (3.448e-3 + 2.431 / re_ice_um)
    om_i2 = max(min(1.0 - (0.02437 + 2.52e-4 * re_ice_um), 0.999999), 0.0)
    g_i2 = 0.7867 + 5.90e-4 * re_ice_um

    tau_2 = tau_l2 + tau_i2
    scat_2 = tau_l2 * om_l2 + tau_i2 * om_i2
    om_2 = torch.where(tau_2 > 1.0e-8, scat_2 / torch.clamp(tau_2, min=1.0e-8), torch.zeros_like(tau_2))
    g_2 = torch.where(
        scat_2 > 1.0e-8,
        (tau_l2 * om_l2 * g_l2 + tau_i2 * om_i2 * g_i2) / torch.clamp(scat_2, min=1.0e-8),
        torch.zeros_like(tau_2),
    )

    return {"sw1": (tau_1, om_1, g_1), "sw2": (tau_2, om_2, g_2)}


def cloud_optical_lw(
    lwp_gm2: torch.Tensor,
    iwp_gm2: torch.Tensor,
    re_ice_um: float = 48.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cloud absorption optical depths (zenith) for LW1 (non-window) and LW2 (window).

    References
    ----------
    - Stephens (1978, J. Atmos. Sci., 35, 2123-2132): Liquid water cloud LW mass absorption
      coefficient kappa_liq ≈ 0.090 m^2 g^-1 (zenith optical depth; diffusivity handled in
      Hemispheric Mean two-stream gamma_1 = 2).
    - Ebert & Curry (1992, J. Geophys. Res., 97, p. 3834, Eq. 12 & Table 3):
      Ice cloud mass absorption kappa_i = alpha_i + gamma_i / r_e [m^2 g^-1]:
        Window (8-12.5 um, Band 2): alpha_2 = 0.0016, gamma_2 = 1.166
        Non-window emission-weighted (Bands 1,3,4,5): alpha_nw = 0.00324, gamma_nw = 1.033
      (Note: Toon Hemispheric Mean multiplies zenith tau by 1/mu_1 = 2.0, vs Ebert & Curry
       beta = 1.66; we scale zenith tau by 1.66 / 2.0 = 0.83 so exp(-2 * tau_z) matches
       Ebert & Curry Eq. 9 emissivity 1 - exp(-1.66 * kappa * IWP) identically!)
    """
    lwp = torch.clamp(lwp_gm2, min=0.0)
    iwp = torch.clamp(iwp_gm2, min=0.0)
    diff_corr = 1.66 / 2.0

    kappa_liq = 0.090  # [m^2 g^-1]
    kappa_ice_nw = 0.00324 + 1.033 / re_ice_um
    kappa_ice_win = 0.00160 + 1.166 / re_ice_um

    tau_cld_lw1 = diff_corr * (kappa_liq * lwp + kappa_ice_nw * iwp)
    tau_cld_lw2 = diff_corr * (kappa_liq * lwp + kappa_ice_win * iwp)
    return tau_cld_lw1, tau_cld_lw2


# ---------------------------------------------------------------------------
# 4. Generalized Multi-Layer Two-Stream Solver (Toon et al., 1989, Eqs. 18-50)
# ---------------------------------------------------------------------------
def toon_two_stream_sw(
    tau_in: torch.Tensor,
    omega0_in: torch.Tensor,
    asym_in: torch.Tensor,
    mu0_in: torch.Tensor,
    f_toa: torch.Tensor,
    albedo_sfc: torch.Tensor,
    albedo_sfc_dif: torch.Tensor | None = None,
    return_sfc_split: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Solve multi-layer solar two-stream radiative transfer (delta-Quadrature).

    All vertical inputs here are in TOP-DOWN order: index 0 = TOA layer, index N-1 = surface layer.

    Reference: Toon et al. (1989, J. Geophys. Res., 94, Table 1, Table 2 & Eqs. 19-50);
    delta-scaling of Joseph, Wiscombe, & Weinman (1976, J. Atmos. Sci., 33, 2452-2459).

    Returns
    -------
    flux_up : torch.Tensor
        Upward shortwave flux at interfaces `[N + 1, ...]` (0 = TOA, N = surface).
    flux_dn : torch.Tensor
        Total (direct + diffuse) downward SW flux at interfaces `[N + 1, ...]`.
    """
    N = tau_in.shape[0]
    col_shape = f_toa.shape
    dtype = tau_in.dtype
    device = tau_in.device

    # Daytime mask; avoid mu0 = 0 division
    day_mask = (mu0_in > 1.0e-4) & (f_toa > 1.0e-4)
    mu0 = torch.clamp(mu0_in, min=1.0e-4)
    f_beam0 = torch.where(day_mask, f_toa, torch.zeros_like(f_toa))

    # Joseph et al. (1976) delta-scaling: f = g^2
    w0 = torch.clamp(omega0_in, min=0.0, max=0.999999)
    g0 = torch.clamp(asym_in, min=-0.99, max=0.99)
    f_del = g0 * g0
    tau = torch.clamp(tau_in * (1.0 - w0 * f_del), min=1.0e-7)
    w = (w0 * (1.0 - f_del)) / torch.clamp(1.0 - w0 * f_del, min=1.0e-7)
    w = torch.clamp(w, min=0.0, max=0.999999)
    g = (g0 - f_del) / torch.clamp(1.0 - f_del, min=1.0e-7)

    # Quadrature two-stream coefficients (Toon et al., 1989, Table 1)
    sqrt3 = math.sqrt(3.0)
    mu1 = 1.0 / sqrt3
    mu0_3d = mu0.unsqueeze(0)
    gamma1 = 0.5 * sqrt3 * (2.0 - w * (1.0 + g))
    gamma2 = 0.5 * sqrt3 * w * (1.0 - g)
    gamma3 = 0.5 * (1.0 - sqrt3 * g * mu0_3d)
    gamma4 = 1.0 - gamma3

    lam = torch.sqrt(torch.clamp(gamma1 * gamma1 - gamma2 * gamma2, min=1.0e-12))
    big_gamma = gamma2 / (gamma1 + lam)

    # Cumulative optical depth tau_c above each layer [N + 1, ...]
    tau_cum = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)
    tau_cum[1:] = torch.cumsum(tau, dim=0)

    # Avoid resonance lam == 1 / mu0 (Toon et al., 1989, p. 16290)
    inv_mu0 = 1.0 / mu0_3d
    denom_res = lam * lam - inv_mu0 * inv_mu0
    denom_res = torch.where(
        torch.abs(denom_res) < 1.0e-5,
        torch.where(denom_res >= 0, torch.full_like(denom_res, 1.0e-5), torch.full_like(denom_res, -1.0e-5)),
        denom_res,
    )

    # Direct solar beam attenuation at top (tau=0) and bottom (tau=tau_n) of each layer
    att_top = torch.exp(-torch.clamp(tau_cum[:-1] * inv_mu0, max=60.0))
    att_bot = torch.exp(-torch.clamp(tau_cum[1:] * inv_mu0, max=60.0))
    fs_3d = f_beam0.unsqueeze(0)

    c_plus_coef = w * fs_3d * ((gamma1 - inv_mu0) * gamma3 + gamma4 * gamma2) / denom_res
    c_minus_coef = w * fs_3d * ((gamma1 + inv_mu0) * gamma4 + gamma2 * gamma3) / denom_res

    cp_0 = c_plus_coef * att_top
    cp_t = c_plus_coef * att_bot
    cm_0 = c_minus_coef * att_top
    cm_t = c_minus_coef * att_bot

    exp_lt = torch.exp(-torch.clamp(lam * tau, max=50.0))
    e1 = 1.0 + big_gamma * exp_lt
    e2 = 1.0 - big_gamma * exp_lt
    e3 = big_gamma + exp_lt
    e4 = big_gamma - exp_lt

    # Assemble 2N tridiagonal matrix (Toon et al., 1989, Eqs. 39-43)
    two_N = 2 * N
    A = torch.zeros((two_N,) + col_shape, dtype=dtype, device=device)
    B = torch.zeros((two_N,) + col_shape, dtype=dtype, device=device)
    D = torch.zeros((two_N,) + col_shape, dtype=dtype, device=device)
    E = torch.zeros((two_N,) + col_shape, dtype=dtype, device=device)

    # Row 0 (l = 1 in 1-based Toon): top diffuse boundary F_1^-(0) = 0
    B[0] = e1[0]
    D[0] = -e2[0]
    E[0] = -cm_0[0]

    # Interior interfaces n = 0 .. N - 2
    if N > 1:
        # Even l = 2*(n+1) -> 0-based index 2*n + 1
        A[1 : two_N - 1 : 2] = e1[:-1] * e2[1:] - e3[:-1] * e4[1:]
        B[1 : two_N - 1 : 2] = e2[:-1] * e2[1:] - e4[:-1] * e4[1:]
        D[1 : two_N - 1 : 2] = e1[1:] * e4[1:] - e2[1:] * e3[1:]
        E[1 : two_N - 1 : 2] = (cp_0[1:] - cp_t[:-1]) * e2[1:] + (cm_t[:-1] - cm_0[1:]) * e4[1:]

        # Odd l = 2*(n+1) + 1 -> 0-based index 2*n + 2
        A[2:two_N:2] = e2[:-1] * e3[:-1] - e4[:-1] * e1[:-1]
        B[2:two_N:2] = e1[:-1] * e1[1:] - e3[:-1] * e3[1:]
        D[2:two_N:2] = e3[:-1] * e4[1:] - e1[:-1] * e2[1:]
        E[2:two_N:2] = e3[:-1] * (cp_0[1:] - cp_t[:-1]) + e1[:-1] * (cm_t[:-1] - cm_0[1:])

    # Bottom row 2N - 1 (l = 2N): surface boundary (Eq. 37 & Eq. 43)
    r_dir = torch.clamp(albedo_sfc, min=0.0, max=1.0)
    r_sfc = r_dir if albedo_sfc_dif is None else torch.clamp(albedo_sfc_dif, min=0.0, max=1.0)
    direct_sfc = mu0 * f_beam0 * att_bot[-1]
    s_sfc = r_dir * direct_sfc
    A[-1] = e1[-1] - r_sfc * e3[-1]
    B[-1] = e2[-1] - r_sfc * e4[-1]
    E[-1] = s_sfc - cp_t[-1] + r_sfc * cm_t[-1]

    Y = solve_tridiagonal_batched(A, B, D, E)
    Y1 = Y[0::2]
    Y2 = Y[1::2]

    # Recover upward and total downward fluxes at all N + 1 interfaces (Eqs. 31-32 & 50)
    flux_up = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)
    flux_dn = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)

    # Top of layer 0 (TOA interface 0)
    flux_up[0] = Y1[0] * e3[0] - Y2[0] * e4[0] + cp_0[0]
    flux_dn[0] = mu0 * f_beam0  # pure direct incident solar flux at TOA

    # Base of layers 0 .. N - 1 (interfaces 1 .. N)
    f_up_base = Y1 * e1 + Y2 * e2 + cp_t
    f_dn_diff_base = Y1 * e3 + Y2 * e4 + cm_t
    f_dn_dir_base = mu0_3d * fs_3d * att_bot

    flux_up[1:] = torch.clamp(f_up_base, min=0.0)
    flux_dn[1:] = torch.clamp(f_dn_diff_base + f_dn_dir_base, min=0.0)

    day_expand = day_mask.unsqueeze(0)
    flux_up = torch.where(day_expand, torch.clamp(flux_up, min=0.0), torch.zeros_like(flux_up))
    flux_dn = torch.where(day_expand, flux_dn, torch.zeros_like(flux_dn))
    if return_sfc_split:
        sfc_dir = torch.where(day_mask, torch.clamp(f_dn_dir_base[-1], min=0.0), torch.zeros_like(f_beam0))
        sfc_dif = torch.where(day_mask, torch.clamp(flux_dn[-1] - sfc_dir, min=0.0), torch.zeros_like(f_beam0))
        return flux_up, flux_dn, sfc_dir, sfc_dif
    return flux_up, flux_dn


def toon_two_stream_lw(
    tau_in: torch.Tensor,
    fb_int_in: torch.Tensor,
    fb_sfc: torch.Tensor,
    emiss_sfc: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve multi-layer thermal IR two-stream transfer (Hemispheric Mean, Toon et al., 1989).

    All vertical inputs here are in TOP-DOWN order:
      - `tau_in`: zenith optical thickness `[N, ...]`
      - `fb_int_in`: band blackbody flux `f_band * sigma * T_int^4` at interfaces `[N + 1, ...]`
                     (index 0 = TOA, index N = bottom air interface)
      - `fb_sfc`: surface band blackbody flux `f_band * sigma * T_sfc^4` `[...]`

    Reference: Toon et al. (1989, J. Geophys. Res., 94, Table 1 Hemispheric Mean & Eqs. 25-48).
    In the pure-absorption IR regime (omega_0 = 0), gamma_1 = 2, gamma_2 = 0, lambda = 2,
    Gamma = 0, and C_n^+(tau), C_n^-(tau) satisfy exact blackbody and optically thin limits.
    """
    N = tau_in.shape[0]
    col_shape = fb_sfc.shape
    dtype = tau_in.dtype
    device = tau_in.device

    tau = torch.clamp(tau_in, min=1.0e-6)
    # Exact analytical decoupling of Toon et al. (1989) Hemispheric Mean (omega_0 = 0):
    # Gamma = 0 => e1 = e2 = 1, e3 = -e4 = exp(-2*tau). Upward & downward streams decouple
    # cleanly except via the surface boundary U_N = emiss * fb_sfc + (1 - emiss) * D_N.
    # Rewriting cm_t - cm_0*E and cp_0 - cp_t*E in terms of w1 = 1 - (1-E)/(2*tau) and
    # w2 = (1-E)/(2*tau) - E eliminates 1/tau cancellation and keeps all weights > 0.
    two_tau = torch.clamp(2.0 * tau, max=50.0)
    exp_lt = torch.exp(-two_tau)
    one_minus_e = -torch.expm1(-two_tau)
    alpha = torch.where(
        two_tau > 1.0e-3,
        one_minus_e / two_tau,
        1.0 - 0.5 * two_tau + (1.0 / 6.0) * two_tau * two_tau,
    )
    w1 = 1.0 - alpha
    w2 = torch.clamp(alpha - exp_lt, min=0.0)

    fb0 = fb_int_in[:-1]
    fb_bot = fb_int_in[1:]
    src_dn = w2 * fb0 + w1 * fb_bot
    src_up = w1 * fb0 + w2 * fb_bot

    flux_dn = torch.empty((N + 1,) + col_shape, dtype=dtype, device=device)
    flux_up = torch.empty((N + 1,) + col_shape, dtype=dtype, device=device)
    flux_dn[0] = 0.0
    dn_cur = flux_dn[0]
    for n in range(N):
        dn_cur = dn_cur * exp_lt[n] + src_dn[n]
        flux_dn[n + 1] = dn_cur

    r_sfc = 1.0 - emiss_sfc
    up_cur = emiss_sfc * fb_sfc + r_sfc * dn_cur
    flux_up[N] = up_cur
    for n in range(N - 1, -1, -1):
        up_cur = up_cur * exp_lt[n] + src_up[n]
        flux_up[n] = up_cur

    return torch.clamp(flux_up, min=0.0), torch.clamp(flux_dn, min=0.0)


# ---------------------------------------------------------------------------
# 5. Complete Four-Band Column Radiative Transfer Driver
# ---------------------------------------------------------------------------
def compute_four_band_radiation(
    pres_c: torch.Tensor,
    pres_w: torch.Tensor,
    temp_c: torch.Tensor,
    temp_w: torch.Tensor,
    dz_c: torch.Tensor,
    rho_d_c: torch.Tensor,
    qv_c: torch.Tensor,
    qc_c: torch.Tensor,
    qi_c: torch.Tensor,
    t_sfc: torch.Tensor,
    albedo_vis: torch.Tensor,
    albedo_nir: torch.Tensor,
    mu0: torch.Tensor,
    s_toa: torch.Tensor,
    co2_ppmv: float = C.CO2_RCEMIP_PPMV,
    emiss_sfc: float | torch.Tensor = 0.98,
    re_liq_um: float = 10.0,
    re_ice_um: float = 48.0,
    cloud_sw_inhomog: float = 0.85,
    albedo_vis_dif: torch.Tensor | None = None,
    albedo_nir_dif: torch.Tensor | None = None,
    q_dust_c: torch.Tensor | None = None,
    o3_vmr_c: torch.Tensor | None = None,
    o3_lw_coupling: bool = False,
) -> RadiationOutput:
    """Evaluate 4-band (SW1, SW2, LW1, LW2) radiative fluxes and heating rates.

    All vertical array inputs (`pres_c`, `temp_c`, `dz_c`, `rho_d_c`, `qv_c`, `qc_c`, `qi_c`
    of shape `[nz, ...]` and `pres_w`, `temp_w` of shape `[nz + 1, ...]`) use AOBA's
    standard bottom-up indexing (`0 = surface`, `nz = TOA`).
    """
    if can_use_crm_metal(pres_c) and t_sfc.numel() > 0:
        from aoba.dust import (
            DUST_ASYM_SW1,
            DUST_ASYM_SW2,
            DUST_K_ABS_LW1,
            DUST_K_ABS_LW2,
            DUST_K_EXT_SW1,
            DUST_K_EXT_SW2,
            DUST_SSA_SW1,
            DUST_SSA_SW2,
        )

        consts = (
            float(C.SOLAR_FRAC_SW1),
            float(C.SOLAR_FRAC_SW2),
            float(C.LW_NONWIN_FRAC),
            float(C.LW_WINDOW_FRAC),
            float(C.STEFAN_BOLTZMANN),
            float(C.O3_G1),
            float(C.O3_G2),
            float(C.O3_G3),
            float(C.CO2_PREIND_PPMV),
            float(C.PAR_FRAC_OF_SW1),
            float(C.CP_D),
            float(C.CP_V),
            float(C.EPS_WATER),
            float(C.P_REF),
            float(C.T_FREEZE),
            float(C.GRAVITY),
            float(DUST_K_EXT_SW1),
            float(DUST_SSA_SW1),
            float(DUST_ASYM_SW1),
            float(DUST_K_EXT_SW2),
            float(DUST_SSA_SW2),
            float(DUST_ASYM_SW2),
            float(DUST_K_ABS_LW1),
            float(DUST_K_ABS_LW2),
        )
        sw_up, sw_dn, lw_up, lw_dn, q_rad_3d, sfc_out = dispatch_rad_four_band_metal(
            pres_c=pres_c,
            pres_w=pres_w,
            temp_c=temp_c,
            temp_w=temp_w,
            dz_c=dz_c,
            rho_d_c=rho_d_c,
            qv_c=qv_c,
            qc_c=qc_c,
            qi_c=qi_c,
            t_sfc=t_sfc,
            albedo_vis=albedo_vis,
            albedo_nir=albedo_nir,
            mu0=mu0,
            s_toa=s_toa,
            co2_ppmv=co2_ppmv,
            emiss_sfc=emiss_sfc,
            re_liq_um=re_liq_um,
            re_ice_um=re_ice_um,
            cloud_sw_inhomog=cloud_sw_inhomog,
            albedo_vis_dif=albedo_vis_dif,
            albedo_nir_dif=albedo_nir_dif,
            q_dust_c=q_dust_c,
            o3_vmr_c=o3_vmr_c,
            o3_lw_coupling=o3_lw_coupling,
            consts=consts,
        )
        sfc_par_wm2 = sfc_out[4]
        return RadiationOutput(
            sw_flux_up=sw_up,
            sw_flux_dn=sw_dn,
            lw_flux_up=lw_up,
            lw_flux_dn=lw_dn,
            q_rad_sw=q_rad_3d[0],
            q_rad_lw=q_rad_3d[1],
            q_rad_net=q_rad_3d[2],
            toa_insolation=sw_dn[-1],
            toa_asr=sw_dn[-1] - sw_up[-1],
            toa_olr=lw_up[-1],
            sfc_sw_net=sw_dn[0] - sw_up[0],
            sfc_lw_net=lw_dn[0] - lw_up[0],
            sfc_par_wm2=sfc_par_wm2,
            sfc_ppfd=C.UMOL_PER_JOULE_PAR * sfc_par_wm2,
            sfc_sw1_dir_dn=sfc_out[0],
            sfc_sw1_dif_dn=sfc_out[1],
            sfc_sw2_dir_dn=sfc_out[2],
            sfc_sw2_dif_dn=sfc_out[3],
        )

    # Flip vertical axis to top-down [0 = TOA, N - 1 = bottom layer] for Toon et al. (1989)
    p_c = torch.flip(pres_c, dims=[0])
    p_w = torch.flip(pres_w, dims=[0])
    t_c = torch.flip(temp_c, dims=[0])
    t_w = torch.flip(temp_w, dims=[0])
    dz = torch.flip(dz_c, dims=[0])
    rho_d = torch.flip(rho_d_c, dims=[0])
    qv = torch.clamp(torch.flip(qv_c, dims=[0]), min=0.0)
    qc = torch.clamp(torch.flip(qc_c, dims=[0]), min=0.0)
    qi = torch.clamp(torch.flip(qi_c, dims=[0]), min=0.0)
    q_dust = torch.clamp(torch.flip(q_dust_c, dims=[0]), min=0.0) if q_dust_c is not None else None

    N = p_c.shape[0]
    col_shape = t_sfc.shape
    dtype = p_c.dtype
    device = p_c.device

    # Dry-air column mass per layer [kg m^-2] and liquid/ice water paths [g m^-2]
    dm_dry = rho_d * dz
    lwp_gm2 = 1000.0 * qc * dm_dry
    iwp_gm2 = 1000.0 * qi * dm_dry

    # ------------------------------------------------------------------
    # A. Shortwave Band 1 (UV + VIS / PAR) & Band 2 (NIR)
    # ------------------------------------------------------------------
    # Slant magnification M(mu0) (Rodgers, 1967; Lacis & Hansen, 1974, Eq. 12)
    mu0_safe = torch.clamp(mu0, min=1.0e-3)
    mag_o3 = 35.0 / torch.sqrt(1224.0 * mu0_safe * mu0_safe + 1.0)

    # Layer ozone column u_O3 [cm-NTP]:
    # 1 mol O3 = 0.048 kg, molar air mass = 0.028964 kg/mol, 1 cm-NTP = 2.1415e-2 kg O3 m^-2
    o3_vmr = torch.clamp(torch.flip(o3_vmr_c, dims=[0]), min=0.0) if o3_vmr_c is not None else background_ozone_vmr(p_c)
    o3_mass_kgm2 = o3_vmr * (48.0 / 28.964) * dm_dry
    du_o3_cm = o3_mass_kgm2 / 2.1415e-2
    # Analytic overhead O3 column above model top (p in [0, p_top]) assuming O3 ~ p^g2
    # Prevents spurious Hartley-band UV heating spikes in the top layer of high-top runs
    u_o3_above = (o3_vmr[0:1] * (48.0 / 28.964) * (p_w[0:1] / C.GRAVITY)) / ((C.O3_G2 + 1.0) * 2.1415e-2)
    u_o3_cum = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)
    u_o3_cum[0:1] = u_o3_above
    u_o3_cum[1:] = u_o3_above + torch.cumsum(du_o3_cm, dim=0)

    x_o3 = u_o3_cum * mag_o3.unsqueeze(0)
    # Normalize total-solar Lacis-Hansen O3 absorptance into SW1 band fraction
    a_o3_band = torch.clamp(_lacis_hansen_ozone_abs(x_o3) / C.SOLAR_FRAC_SW1, max=0.999)
    trans_o3_ratio = (1.0 - a_o3_band[1:]) / torch.clamp(1.0 - a_o3_band[:-1], min=1.0e-5)
    dtau_o3 = -mu0_safe.unsqueeze(0) * torch.log(torch.clamp(trans_o3_ratio, min=1.0e-5, max=1.0))

    # Rayleigh molecular scattering optical depth in SW1 (~0.095 at p_s = 1013 hPa; Hansen & Travis, 1974)
    dp_pa = torch.clamp(p_w[1:] - p_w[:-1], min=1.0)
    dtau_ray = 0.095 * (dp_pa / C.P_REF)

    # Scaled water-vapor path y [cm] for SW2 (Lacis & Hansen, 1974, Eq. 21 & Fig. 13):
    # 10 kg m^-2 = 1 g cm^-2 = 1 cm precipitable water, linear p/p0 and sqrt(T0/T) scaling
    dy_wv_cm = 0.1 * qv * dm_dry * (p_c / C.P_REF) * torch.sqrt(C.T_FREEZE / torch.clamp(t_c, min=180.0))
    y_wv_cum = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)
    y_wv_cum[1:] = torch.cumsum(dy_wv_cm, dim=0)
    y_slant = y_wv_cum / mu0_safe.unsqueeze(0)
    a_wv_band = torch.clamp(_lacis_hansen_wv_abs(y_slant) / C.SOLAR_FRAC_SW2, max=0.999)
    trans_wv_ratio = (1.0 - a_wv_band[1:]) / torch.clamp(1.0 - a_wv_band[:-1], min=1.0e-5)
    dtau_wv_sw2 = -mu0_safe.unsqueeze(0) * torch.log(torch.clamp(trans_wv_ratio, min=1.0e-5, max=1.0))

    # Combine gas + Rayleigh + cloud optical properties in SW1 and SW2
    cld_sw = cloud_optical_sw(
        lwp_gm2, iwp_gm2, re_liq_um=re_liq_um, re_ice_um=re_ice_um, cloud_sw_inhomog=cloud_sw_inhomog
    )
    tau_c1, om_c1, g_c1 = cld_sw["sw1"]
    tau_c2, om_c2, g_c2 = cld_sw["sw2"]

    tau_sw1 = dtau_o3 + dtau_ray + tau_c1
    scat_sw1 = dtau_ray + tau_c1 * om_c1
    asym_scat_sw1 = tau_c1 * om_c1 * g_c1

    tau_sw2 = dtau_wv_sw2 + tau_c2
    scat_sw2 = tau_c2 * om_c2
    asym_scat_sw2 = tau_c2 * om_c2 * g_c2

    if q_dust is not None:
        from aoba.dust import (
            DUST_ASYM_SW1,
            DUST_ASYM_SW2,
            DUST_K_EXT_SW1,
            DUST_K_EXT_SW2,
            DUST_SSA_SW1,
            DUST_SSA_SW2,
        )

        dm_dust = q_dust * dm_dry
        tau_d1 = DUST_K_EXT_SW1 * dm_dust
        tau_d2 = DUST_K_EXT_SW2 * dm_dust
        tau_sw1 = tau_sw1 + tau_d1
        scat_sw1 = scat_sw1 + tau_d1 * DUST_SSA_SW1
        asym_scat_sw1 = asym_scat_sw1 + tau_d1 * DUST_SSA_SW1 * DUST_ASYM_SW1
        tau_sw2 = tau_sw2 + tau_d2
        scat_sw2 = scat_sw2 + tau_d2 * DUST_SSA_SW2
        asym_scat_sw2 = asym_scat_sw2 + tau_d2 * DUST_SSA_SW2 * DUST_ASYM_SW2

    om_sw1 = scat_sw1 / torch.clamp(tau_sw1, min=1.0e-8)
    g_sw1 = asym_scat_sw1 / torch.clamp(scat_sw1, min=1.0e-8)

    om_sw2 = scat_sw2 / torch.clamp(tau_sw2, min=1.0e-8)
    g_sw2 = asym_scat_sw2 / torch.clamp(scat_sw2, min=1.0e-8)

    sw1_up, sw1_dn, sfc_sw1_dir, sfc_sw1_dif = toon_two_stream_sw(
        tau_sw1,
        om_sw1,
        g_sw1,
        mu0,
        C.SOLAR_FRAC_SW1 * s_toa,
        albedo_vis,
        albedo_sfc_dif=albedo_vis_dif,
        return_sfc_split=True,
    )
    sw2_up, sw2_dn, sfc_sw2_dir, sfc_sw2_dif = toon_two_stream_sw(
        tau_sw2,
        om_sw2,
        g_sw2,
        mu0,
        C.SOLAR_FRAC_SW2 * s_toa,
        albedo_nir,
        albedo_sfc_dif=albedo_nir_dif,
        return_sfc_split=True,
    )

    sw_up_td = sw1_up + sw2_up
    sw_dn_td = sw1_dn + sw2_dn

    # ------------------------------------------------------------------
    # B. Longwave Band 1 (H2O + CO2) & Band 2 (8-12.5 um Window)
    # ------------------------------------------------------------------
    # Reference: Weaver & Ramanathan (1995), Koll & Cronin (2018), Sherwood et al. (2020).
    # Note: Toon Hemispheric Mean multiplies zenith tau by 2.0 along diffusive streams.
    co2_ratio = max(co2_ppmv, 1.0) / C.CO2_PREIND_PPMV
    ln_co2 = math.log(co2_ratio)
    dp2_norm = torch.clamp(p_w[1:] ** 2 - p_w[:-1] ** 2, min=0.0) / (C.P_REF ** 2)

    # LW1: Voigt/Lorentz pressure-scaled H2O rotational/vibrational bands ((p/p0)^0.55;
    # Goody, 1964; Weaver & Ramanathan, 1995) + CO2 15-um tropospheric & stratospheric
    # Voigt/Doppler core cooling (Dickinson, 1973; Andrews et al., 1987, Sec. 2.5; Sherwood et al., 2020)
    dtau_wv_lw1 = 0.30 * torch.pow(torch.clamp(p_c / C.P_REF, min=1.0e-4), 0.55) * qv * dm_dry
    strat_co2_core = (
        0.22
        * max(0.20, 1.0 + 0.35 * ln_co2)
        * torch.exp(-p_c / 10000.0)
        * torch.pow(torch.clamp(10000.0 / torch.clamp(p_c, min=50.0), min=1.0), 0.50)
    )
    dtau_co2_lw1 = max(0.02, 0.85 + 0.22 * ln_co2) * dp2_norm + (0.020 + strat_co2_core) * (dp_pa / C.P_REF)

    # LW2: 8-12.5 um atmospheric window: H2O e-type self-continuum (Roberts et al., 1976;
    # Koll & Cronin, 2018) + weak foreign continuum + CO2 15-um wing / 10-um band overlap
    e_vap = (qv * p_c) / (C.EPS_WATER + qv)
    t_cont_fac = torch.exp(torch.clamp(1800.0 * (1.0 / torch.clamp(t_c, min=190.0) - 1.0 / 296.0), max=8.0))
    dtau_wv_lw2 = (0.45 * (e_vap / C.P_REF) * t_cont_fac + 0.0025 * (p_c / C.P_REF)) * qv * dm_dry
    dtau_co2_lw2 = max(0.0, 0.015 + 0.024 * ln_co2) * dp2_norm

    tau_cld_lw1, tau_cld_lw2 = cloud_optical_lw(lwp_gm2, iwp_gm2, re_ice_um=re_ice_um)

    tau_lw1 = dtau_wv_lw1 + dtau_co2_lw1 + tau_cld_lw1
    tau_lw2 = dtau_wv_lw2 + dtau_co2_lw2 + tau_cld_lw2
    if o3_lw_coupling:
        # 9.6 um O3 band absorption inside the 8-12.5 um atmospheric window (Ramanathan & Dickinson, 1979)
        tau_lw2 = tau_lw2 + 4.5 * o3_mass_kgm2
    if q_dust is not None:
        from aoba.dust import DUST_K_ABS_LW1, DUST_K_ABS_LW2

        dm_dust = q_dust * dm_dry
        tau_lw1 = tau_lw1 + DUST_K_ABS_LW1 * dm_dust
        tau_lw2 = tau_lw2 + DUST_K_ABS_LW2 * dm_dust

    eb_int = C.STEFAN_BOLTZMANN * (torch.clamp(t_w, min=100.0) ** 4)
    eb_sfc = C.STEFAN_BOLTZMANN * (torch.clamp(t_sfc, min=100.0) ** 4)

    lw1_up, lw1_dn = toon_two_stream_lw(
        tau_lw1, C.LW_NONWIN_FRAC * eb_int, C.LW_NONWIN_FRAC * eb_sfc, emiss_sfc=emiss_sfc
    )
    lw2_up, lw2_dn = toon_two_stream_lw(
        tau_lw2, C.LW_WINDOW_FRAC * eb_int, C.LW_WINDOW_FRAC * eb_sfc, emiss_sfc=emiss_sfc
    )

    lw_up_td = lw1_up + lw2_up
    lw_dn_td = lw1_dn + lw2_dn

    # ------------------------------------------------------------------
    # C. Flip back to AOBA bottom-up indexing (k = 0 at surface, nz at TOA)
    #    and compute exact energy-conserving radiative heating rates
    # ------------------------------------------------------------------
    sw_up = torch.flip(sw_up_td, dims=[0])
    sw_dn = torch.flip(sw_dn_td, dims=[0])
    lw_up = torch.flip(lw_up_td, dims=[0])
    lw_dn = torch.flip(lw_dn_td, dims=[0])

    # Moist air heat capacity per unit area of layer k: C_layer = (c_pd + c_pv * q_v) * dm_dry [J m^-2 K^-1]
    cp_layer = (C.CP_D + C.CP_V * torch.clamp(qv_c, min=0.0)) * (rho_d_c * dz_c)

    # Net upward flux F_net_up = F_up - F_dn; convergence into layer k is -(F_net_up[k+1] - F_net_up[k])
    fnet_sw_up = sw_up - sw_dn
    fnet_lw_up = lw_up - lw_dn
    q_rad_sw = -(fnet_sw_up[1:] - fnet_sw_up[:-1]) / cp_layer
    q_rad_lw = -(fnet_lw_up[1:] - fnet_lw_up[:-1]) / cp_layer
    q_rad_net = q_rad_sw + q_rad_lw

    # Surface PAR (0.40-0.70 um portion of downward SW1 at interface k = 0)
    sw1_dn_sfc = sw1_dn[-1]
    sfc_par_wm2 = C.PAR_FRAC_OF_SW1 * sw1_dn_sfc
    sfc_ppfd = C.UMOL_PER_JOULE_PAR * sfc_par_wm2

    toa_insol = sw_dn[-1]
    toa_asr = sw_dn[-1] - sw_up[-1]
    toa_olr = lw_up[-1]
    sfc_sw_net = sw_dn[0] - sw_up[0]
    sfc_lw_net = lw_dn[0] - lw_up[0]

    return RadiationOutput(
        sw_flux_up=sw_up,
        sw_flux_dn=sw_dn,
        lw_flux_up=lw_up,
        lw_flux_dn=lw_dn,
        q_rad_sw=q_rad_sw,
        q_rad_lw=q_rad_lw,
        q_rad_net=q_rad_net,
        toa_insolation=toa_insol,
        toa_asr=toa_asr,
        toa_olr=toa_olr,
        sfc_sw_net=sfc_sw_net,
        sfc_lw_net=sfc_lw_net,
        sfc_par_wm2=sfc_par_wm2,
        sfc_ppfd=sfc_ppfd,
        sfc_sw1_dir_dn=sfc_sw1_dir,
        sfc_sw1_dif_dn=sfc_sw1_dif,
        sfc_sw2_dir_dn=sfc_sw2_dir,
        sfc_sw2_dif_dn=sfc_sw2_dif,
    )
