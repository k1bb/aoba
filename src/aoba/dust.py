"""Map-Free Global-Parameter Mineral Dust Aerosol Module (Emission, Settling, Scavenging).

Dust mobilization follows the CLM5 dust scheme (Lawrence et al., 2020, CLM5 Tech. Note, Ch. 31;
Mahowald et al., 2006), itself based on the DEAD model of Zender et al. (2003). It is driven only by
the autonomous land-surface state (exposed LAI, fractional snow cover, top-layer liquid/ice soil water,
and friction velocity) with one global tuning factor `dust_emis_factor` (default `1.5e-5`, scaled from
CLM5's `T = 5e-4` for map-free global `S = 1`), and no regional source-erodibility maps (CLM5 also
sets the erodibility `S = 1`).

References
----------
- Lawrence et al. (2020, CLM5 Tech. Note, Ch. 31, Eqs. 31.1-31.14) & Zender, Bian, & Newman (2003,
  J. Geophys. Res., 108, 4416): vertical dust flux F = T S f_m alpha Q_s sum(M), bare/vegetated
  mobilization fraction, Iversen & White (1982) threshold friction speed with Fecan et al. (1999)
  moisture factor, White (1979) saltation flux, Owen (1964) effect, and Marticorena & Bergametti
  (1995) clay-dependent sandblasting efficiency.
- Balkanski, Schulz, Claquin, & Guibert (2007, Atmos. Chem. Phys., 7, 81-95):
  Broadband shortwave & longwave mineral dust optical properties.
"""

from __future__ import annotations

import math
import torch

from aoba import constants as C
from aoba.surface import SOIL_POROSITY, SURF_LAND


# CLM5 / DEAD mobilization constants (Lawrence et al., 2020, Ch. 31; Zender et al., 2003)
DUST_VAI_MBL_THR: float = 0.3       # Leaf+stem area index above which no dust is mobilized (L+S)_t [m^2 m^-2] (Eq. 31.3)
DUST_CLAY_FRAC: float = 0.20        # Clay mass fraction of the single universal loam soil [-] (alpha uses min(clay, 0.2), Eq. 31.4)
DUST_SALT_CONST: float = 2.61       # Saltation constant c_s [-] (Eq. 31.5)
DUST_RHO_OSP: float = 2650.0        # Density of optimal saltation particles [kg m^-3] (Eq. 31.6)
DUST_D_OSP: float = 75.0e-6         # Diameter of optimal saltation particles [m] (Eq. 31.6)
DUST_RHO_SOIL_SOLID: float = 2700.0 # Soil mineral particle density for bulk density (1 - porosity) * 2700 [kg m^-3]
DUST_OWEN_COEF: float = 0.003       # Owen (1964) effect coefficient [s m^-1] (Eq. 31.12)
# Source-mode mass fractions, mass-median diameters, and geometric std devs (Table 31.1), integrated over the
# transported 0.1-10 um size range (Table 31.2) because AOBA carries a single dust tracer.
_DUST_SOURCE_MODES: tuple[tuple[float, float, float], ...] = (
    (0.036, 0.832e-6, 2.1),
    (0.957, 4.820e-6, 1.9),
    (0.007, 19.38e-6, 1.6),
)


def _transported_mass_fraction(d_min: float = 0.1e-6, d_max: float = 10.0e-6) -> float:
    """Sum of M_{i,j} over source modes i and all transport bins j spanning [d_min, d_max] (Eq. 31.14)."""
    tot = 0.0
    for m_i, d_v, s_g in _DUST_SOURCE_MODES:
        den = math.sqrt(2.0) * math.log(s_g)
        tot += 0.5 * m_i * (math.erf(math.log(d_max / d_v) / den) - math.erf(math.log(d_min / d_v) / den))
    return tot


DUST_TRANSPORTED_FRAC: float = _transported_mass_fraction()  # = 0.871


def _dry_threshold_coefficient() -> float:
    """Density-independent part of the dry saltation threshold friction speed (Eqs. 31.6, 31.10-31.11)."""
    re_t = 0.38 + 1331.0 * (100.0 * DUST_D_OSP) ** 1.56
    if re_t <= 10.0:
        re_f = 0.1291 ** 2 / (-1.0 + 1.928 * re_t ** 0.0922)
    else:
        re_f = (0.120 * (1.0 - 0.0858 * math.exp(-0.0617 * (re_t - 10.0)))) ** 2
    icf = 1.0 + 6.0e-7 / (DUST_RHO_OSP * C.GRAVITY * DUST_D_OSP ** 2.5)
    return math.sqrt(re_f * DUST_RHO_OSP * C.GRAVITY * DUST_D_OSP * icf)  # u*_t,dry = coef / sqrt(rho_atm) (~0.21 m/s)


V_SETTLE_DUST: float = 4.0e-3       # Effective Stokes settling velocity (~1.5 um radius) [m s^-1]
V_DEP_DUST: float = 6.0e-3          # Surface dry deposition velocity (>= v_settle, aerodynamic+settling) [m s^-1]
ALPHA_WET_SCAV: float = 0.50        # Precipitation washout coefficient (Zender et al., 2003; Croft et al., 2010) [m^2 kg^-1]

# Broadband optical properties (Balkanski et al., 2007)
DUST_K_EXT_SW1: float = 600.0       # VIS mass extinction efficiency [m^2 kg^-1]
DUST_SSA_SW1: float = 0.92          # VIS single-scattering albedo [-]
DUST_ASYM_SW1: float = 0.70         # VIS asymmetry factor [-]
DUST_K_EXT_SW2: float = 400.0       # NIR mass extinction efficiency [m^2 kg^-1]
DUST_SSA_SW2: float = 0.95          # NIR single-scattering albedo [-]
DUST_ASYM_SW2: float = 0.65         # NIR asymmetry factor [-]
DUST_K_ABS_LW1: float = 100.0       # LW1 mass absorption efficiency [m^2 kg^-1]
DUST_K_ABS_LW2: float = 150.0       # LW2 window mass absorption efficiency [m^2 kg^-1]


def compute_dust_emission_sub(
    surf_type: torch.Tensor,
    lai: torch.Tensor,
    f_snow: torch.Tensor,
    w_liq_soil_top: torch.Tensor,
    w_ice_soil_top: torch.Tensor,
    u_star: torch.Tensor,
    rho_atm: torch.Tensor,
    z0m: torch.Tensor,
    d_disp: torch.Tensor | None = None,
    dust_emis_factor: float = 1.5e-5,
) -> torch.Tensor:
    """Compute subcolumn mineral dust vertical emission flux `[kg m^-2 s^-1]` (CLM5 Eq. 31.1).

    `w_liq_soil_top` / `w_ice_soil_top` are top-layer *volumetric* liquid / ice soil water `[m^3 m^-3]`,
    `f_snow` the fractional snow cover `[-]`, `rho_atm` the near-surface air density `[kg m^-3]`, and
    `z0m`, `d_disp` the momentum roughness length and displacement height `[m]` used for the 10-m wind.
    """
    dtype = u_star.dtype
    is_land = (surf_type == SURF_LAND).to(dtype)

    # Mobilizable bare fraction f_m = (1 - f_sno)(1 - f_v) * liquid fraction (Eqs. 31.2-31.3; no lakes in AOBA)
    f_v = torch.clamp(torch.clamp(lai, min=0.0) / DUST_VAI_MBL_THR, 0.0, 1.0)
    w_liq = torch.clamp(w_liq_soil_top, min=0.0)
    w_ice = torch.clamp(w_ice_soil_top, min=0.0)
    liq_frac = torch.where(w_liq + w_ice > 1.0e-6, w_liq / torch.clamp(w_liq + w_ice, min=1.0e-6), torch.ones_like(w_liq))
    f_m = is_land * torch.clamp(1.0 - f_snow, 0.0, 1.0) * (1.0 - f_v) * liq_frac

    # Moisture-dependent threshold friction speed (Eqs. 31.6-31.9; Oleson et al., 2013 Eqs. 24.6-24.9).
    # CLM sets the tuning multiplier a = M_clay^-1 in w_t = a (0.17 M_clay + 0.14 M_clay^2) = 0.17 + 0.14 M_clay.
    rho_bulk = (1.0 - SOIL_POROSITY) * DUST_RHO_SOIL_SOLID
    w_grav = (w_liq + w_ice) * 1000.0 / rho_bulk
    w_thr = 0.17 + 0.14 * DUST_CLAY_FRAC
    f_w = torch.where(
        w_grav > w_thr,
        torch.sqrt(1.0 + 1.21 * torch.pow(torch.clamp(100.0 * (w_grav - w_thr), min=0.0), 0.68)),
        torch.ones_like(w_grav),
    )
    rho = torch.clamp(rho_atm, min=0.1)
    u_thr = _dry_threshold_coefficient() / torch.sqrt(rho) * f_w

    # Owen (1964) effect on the saltation friction speed using the log-law 10-m wind (Eqs. 31.12-31.13)
    u_st = torch.clamp(u_star, min=1.0e-3)
    z0 = torch.clamp(z0m, min=1.0e-5)
    d0 = torch.zeros_like(z0) if d_disp is None else torch.clamp(d_disp, min=0.0)
    z_eff = torch.clamp(10.0 - d0, min=1.5) / z0
    u10 = (u_st / C.VON_KARMAN) * torch.log(torch.clamp(z_eff, min=1.01))
    u10_thr = u_thr * u10 / u_st
    u_salt = torch.where(u10 >= u10_thr, u_st + DUST_OWEN_COEF * (u10 - u10_thr) ** 2, u_st)

    # White (1979) total horizontal saltation mass flux Q_s [kg m^-1 s^-1] (Eq. 31.5)
    ratio = u_thr / torch.clamp(u_salt, min=1.0e-3)
    q_salt = torch.where(
        ratio < 1.0,
        DUST_SALT_CONST * rho * u_salt ** 3 / C.GRAVITY * (1.0 - ratio) * (1.0 + ratio) ** 2,
        torch.zeros_like(u_salt),
    )

    # Marticorena & Bergametti (1995) sandblasting efficiency alpha [m^-1] (Eq. 31.4)
    m_clay = min(DUST_CLAY_FRAC, 0.20)
    alpha = 100.0 * math.exp((13.4 * m_clay - 6.0) * math.log(10.0))

    return float(dust_emis_factor) * f_m * alpha * q_salt * DUST_TRANSPORTED_FRAC


def step_dust_column(
    q_dust: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    z_c: torch.Tensor,
    f_dust_emit: torch.Tensor,
    precip_rate: torch.Tensor,
    u_star_cell: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Advance 3D host dust mixing ratio `[Nz, N_cells]` via emission, PBL mixing, settling & washout."""
    nz = q_dust.shape[0]
    m_k = torch.clamp(rho_d * dz_c, min=1.0e-3)
    q_cur = torch.clamp(q_dust, min=0.0).clone()

    # 1. Surface emission injection into lowest layer
    q_cur[0] = q_cur[0] + dt * torch.clamp(f_dust_emit, min=0.0) / m_k[0]

    # 2. Fast PBL turbulent mixing across lowest ~1.5 km (two-pass symmetric relaxation)
    z_rel = torch.clamp(z_c - z_c[0:1] + 0.5 * dz_c[0:1], min=10.0)
    h_pbl = 1200.0
    for k in range(min(6, nz - 1)):
        z_int = 0.5 * (z_rel[k] + z_rel[k + 1])
        k_z = torch.where(
            z_int < h_pbl,
            0.4 * torch.clamp(u_star_cell, min=0.05) * z_int * ((1.0 - z_int / h_pbl) ** 2),
            torch.zeros_like(z_int),
        )
        dz_int = torch.clamp(z_c[k + 1] - z_c[k], min=10.0)
        rho_int = 0.5 * (rho_d[k] + rho_d[k + 1])
        exch = (dt * rho_int * k_z) / dz_int
        # Implicit-like bounded exchange between layer k and k+1 conserving mass
        frac = exch / (1.0 + exch * (1.0 / m_k[k] + 1.0 / m_k[k + 1]))
        dq_flux = frac * (q_cur[k] - q_cur[k + 1])
        q_cur[k] = q_cur[k] - dq_flux / m_k[k]
        q_cur[k + 1] = q_cur[k + 1] + dq_flux / m_k[k + 1]

    # 3. Unconditionally stable top-down implicit gravitational settling + surface dry deposition
    q_out = torch.empty_like(q_cur)
    flux_above = torch.zeros_like(q_cur[0])
    for k in range(nz - 1, -1, -1):
        v_fall = V_DEP_DUST if k == 0 else V_SETTLE_DUST
        sink = dt * rho_d[k] * v_fall
        q_k = (m_k[k] * q_cur[k] + dt * flux_above) / (m_k[k] + sink)
        q_out[k] = q_k
        flux_above = rho_d[k] * v_fall * q_k

    # 4. Precipitation wet scavenging (washout)
    wash = torch.exp(-ALPHA_WET_SCAV * torch.clamp(precip_rate, min=0.0) * dt)
    q_out = q_out * wash.unsqueeze(0)
    return torch.clamp(q_out, min=0.0, max=1.0e-3)
