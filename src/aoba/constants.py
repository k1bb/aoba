"""Physical, astronomical, thermodynamic, radiative, and biogeochemical constants.

Every constant explicitly cites its published reference:
- Wing et al. (2018, Geosci. Model Dev., 11, 793-813, Table 1 & Eq. 1)
- Khairoutdinov & Randall (2003, J. Atmos. Sci., 60, 607-623, Appendix A)
- MIROC6 AGCM Document Writing Team (2021, CCSR Report No. 65, Sec. 1.3.2 & 3.6.15)
- Stocker et al. (2020, Geosci. Model Dev., 13, 1545-1581, Table 1)
- Weaver & Ramanathan (1995, J. Geophys. Res., 100, 11585-11591)
- Semtner (1976, J. Phys. Oceanogr., 6, 379-389)

AOBA Formulation Reference: Sec. 1.2 (Table of Fundamental Physical Constants).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 1. Planetary & Orbital Constants (Wing et al., 2018, Table 1; MIROC6 Sec. 1.3.2)
# ---------------------------------------------------------------------------
EARTH_RADIUS: float = 6.371e6          # Mean Earth radius R_E [m] (Wing et al., 2018, Table 1)
GRAVITY: float = 9.79764               # Mean surface gravity g [m s^-2] (Wing et al., 2018, Table 1)
OMEGA_EARTH: float = 7.292115e-5       # Planetary rotation rate Omega [rad s^-1] (MIROC6 Sec. 1.3.2)
SECONDS_PER_DAY: float = 86400.0       # Length of solar day [s]
DAYS_PER_YEAR: float = 365.2425        # Mean tropical/Gregorian year length [d] (Davis et al., 2017)
SOLAR_CONSTANT: float = 1361.0         # Present-day total solar irradiance S_0 [W m^-2] (Wing et al., 2018)
OBLIQUITY_DEG: float = 23.44           # Present-day obliquity epsilon [deg] (Davis et al., 2017, Table 2)
ECCENTRICITY: float = 0.0167           # Present-day orbital eccentricity e [-] (Davis et al., 2017, Table 2)
PERIHELION_LON_DEG: float = 283.0      # Heliocentric longitude of perihelion omega [deg] (Davis et al., 2017)

# ---------------------------------------------------------------------------
# 2. Dry-Air & Moist-Thermodynamic Constants (Wing et al., 2018, Table 1; Skamarock et al., 2012)
# ---------------------------------------------------------------------------
R_D: float = 287.04                    # Gas constant for dry air R_d [J kg^-1 K^-1]
CP_D: float = 1004.64                  # Specific heat of dry air at constant pressure c_pd [J kg^-1 K^-1]
CV_D: float = CP_D - R_D               # Specific heat of dry air at constant volume c_vd [J kg^-1 K^-1]
KAPPA_D: float = R_D / CP_D            # Poisson exponent R_d / c_pd [-]
GAMMA_D: float = CP_D / CV_D           # Heat capacity ratio c_pd / c_vd [-]

R_V: float = 461.50                    # Gas constant for water vapor R_v [J kg^-1 K^-1]
CP_V: float = 1846.0                   # Specific heat of water vapor at constant pressure c_pv [J kg^-1 K^-1]
EPS_WATER: float = R_D / R_V           # Molecular weight ratio M_v / M_d = R_d / R_v ≈ 0.62197 [-]
RV_OVER_RD: float = R_V / R_D          # R_v / R_d ≈ 1.60779 [-] (Skamarock et al., 2012, Eq. 5)
ZVIR: float = RV_OVER_RD - 1.0         # Virtual temperature coefficient (R_v/R_d - 1) ≈ 0.60779 [-]

P_REF: float = 1.0e5                   # Reference surface pressure p_0 [Pa] (Skamarock et al., 2012, Eq. 6)
T_FREEZE: float = 273.15               # Standard 0 degC in Kelvin [K]
T_TRIPLE: float = 273.16               # Water triple-point temperature T_0 [K] (Khairoutdinov & Randall, 2003, Eq. A13)
E_TRIPLE: float = 611.2                # Saturation vapor pressure at 0 degC [Pa] (Bolton, 1980)

L_V0: float = 2.501e6                  # Latent heat of vaporization at 0 degC [J kg^-1] (Wing et al., 2018, Table 1)
L_F0: float = 3.337e5                  # Latent heat of fusion at 0 degC [J kg^-1] (Wing et al., 2018, Table 1)
L_S0: float = 2.8347e6                 # Latent heat of sublimation L_v0 + L_f0 [J kg^-1]

# Mixed-phase cloud condensate temperature ramp (Khairoutdinov & Randall, 2003, Eq. A13 & App. B)
T_ICE_ALL: float = 253.15              # Temperature T_00n below which condensate is 100% ice [K]
T_LIQ_ALL: float = 273.16              # Temperature T_0n above which condensate is 100% liquid [K]

# ---------------------------------------------------------------------------
# 3. Radiative Constants (Stefan-Boltzmann, Slingo 1989, Wing 2018, Weaver & Ramanathan 1995)
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN: float = 5.670374419e-8  # Stefan-Boltzmann constant sigma [W m^-2 K^-4]
SOLAR_FRAC_SW1: float = 0.459760       # UV+VIS/PAR solar fraction (0.25-0.69 um; Slingo, 1989, Table 1)
SOLAR_FRAC_SW2: float = 0.540240       # Near-IR solar fraction (0.69-4.00 um; Slingo, 1989, Table 1)
LW_WINDOW_FRAC: float = 0.320          # 8-12.5 um window Planck weight beta (Weaver & Ramanathan, 1995; Koll & Cronin, 2018)
LW_NONWIN_FRAC: float = 1.0 - LW_WINDOW_FRAC  # H2O+CO2 band Planck weight 1 - beta = 0.680

# RCEMIP background ozone gamma distribution (Wing et al., 2018, Eq. 1)
O3_G1: float = 3.6478                  # Ozone amplitude g_1 [ppmv hPa^-g_2]
O3_G2: float = 0.83209                 # Ozone power-law exponent g_2 [-]
O3_G3: float = 11.3515                 # Ozone pressure scale g_3 [hPa]

# Well-mixed atmospheric gas defaults
CO2_PREIND_PPMV: float = 284.3         # Pre-industrial CO2 mixing ratio [ppmv]
CO2_RCEMIP_PPMV: float = 348.0         # RCEMIP standard CO2 mixing ratio [ppmv] (Wing et al., 2018, Table 2)
O2_MIXING_RATIO: float = 0.20946       # Dry-air O2 volume mixing ratio [mol mol^-1] (Stocker et al., 2020)

# PAR photon conversion (Stocker et al., 2020, Sec. 2 / SPLASH Davis et al., 2017: 2.04 umol J^-1 of total SW,
# or 4.57 umol photons per Joule of absorbed/transmitted 0.4-0.7 um PAR; McCree, 1972)
PAR_FRAC_OF_SW1: float = 0.88          # Fraction of SW1 (0.25-0.69 um) energy within 0.40-0.70 um PAR band [-]
UMOL_PER_JOULE_PAR: float = 4.57       # Quantum conversion factor [umol photons J^-1 PAR]

# ---------------------------------------------------------------------------
# 4. Ocean, Sea-Ice, and Surface Layer Constants (Codron, 2012; Semtner, 1976; Harman & Finnigan, 2007)
# ---------------------------------------------------------------------------
VON_KARMAN: float = 0.40               # von Karman constant kappa [-] (Harman & Finnigan, 2007)
RHO_SEAWATER: float = 1025.0           # Reference seawater density [kg m^-3] (Codron, 2012)
CP_SEAWATER: float = 3990.0            # Specific heat capacity of seawater [J kg^-1 K^-1] (Codron, 2012)
T_FREEZE_OCEAN: float = 271.35         # Seawater freezing point (-1.8 degC) [K] (Semtner, 1976)
RHO_SEAICE: float = 917.0              # Sea-ice density [kg m^-3] (Semtner, 1976)
K_THERM_SEAICE: float = 2.034          # Thermal conductivity of sea ice [W m^-1 K^-1] (Semtner, 1976)
K_THERM_SNOW: float = 0.3096           # Thermal conductivity of snow on ice [W m^-1 K^-1] (Semtner, 1976)

R_GAS_UNIVERSAL: float = 8.314462618   # Universal gas constant R [J mol^-1 K^-1] (Stocker et al., 2020)
PI: float = math.pi


def recompute_derived_constants() -> None:
    """Recompute derived thermodynamic/radiative constants after any primary constant update."""
    global CV_D, KAPPA_D, GAMMA_D, EPS_WATER, RV_OVER_RD, ZVIR, L_S0, LW_NONWIN_FRAC, SOLAR_FRAC_SW2
    CV_D = CP_D - R_D
    KAPPA_D = R_D / CP_D
    GAMMA_D = CP_D / CV_D
    EPS_WATER = R_D / R_V
    RV_OVER_RD = R_V / R_D
    ZVIR = RV_OVER_RD - 1.0
    L_S0 = L_V0 + L_F0
    SOLAR_FRAC_SW2 = 1.0 - SOLAR_FRAC_SW1
    LW_NONWIN_FRAC = 1.0 - LW_WINDOW_FRAC
