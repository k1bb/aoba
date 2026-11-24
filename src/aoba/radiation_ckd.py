"""Self-contained 16-Band (50 g-point) Correlated-k Two-Stream Radiative Transfer Solver.

Upgrades AOBA's radiative transfer from the legacy 4-band analytical scheme to a full
deterministic correlated-k distribution (CKD) solver spanning:
- **10 Longwave spectral bands (32 g-points, 10-3250 cm^-1)** with exact analytical
  spectral Planck integration (Widger & Woodall, 1976), MT_CKD H2O self- and
  foreign-continuum absorption (Clough et al., 1989; Mlawer et al., 2012, 2023),
  pressure/Voigt-broadened correlated-k absorption for H2O, CO2 (15 um core/wings,
  10 um hot bands, 4.3 um), O3 (9.6 um & 14.3 um), CH4 (7.66 um), and N2O (7.78, 8.6,
  17 um) resolving gas spectral overlaps, plus band-resolved liquid (Stephens, 1978;
  Hu & Stamnes, 1993), ice (Ebert & Curry, 1992, Table 3), and mineral dust
  (Balkanski et al., 2007) infrared absorption.
- **6 Shortwave spectral bands (18 g-points, 0.20-4.00 um)** covering O3 Hartley,
  Huggins, and Chappuis bands (with analytic overhead-ozone beam attenuation above the
  model top), wavelength-resolved Rayleigh lambda^-4 scattering, O2 A/B bands, H2O NIR
  absorption, and CO2 + CH4 NIR solar absorption (Etminan et al., 2016), combined with
  Slingo (1989) 4-band liquid, Ebert & Curry (1992) 5-band ice, and Balkanski et al.
  (2007) dust scattering solved via the delta-Quadrature two-stream algorithm of
  Toon et al. (1989).

Both `re_liq_um` and `re_ice_um` accept either scalar floats or per-voxel `[nz, ...]`
tensors so 2-moment prognostic cloud effective radii couple directly without overhead.

References
----------
- Toon, McKay, Ackerman, & Santhanam (1989, J. Geophys. Res., 94, 16287-16301)
- Goody, West, Chen, & Crisp (1989, J. Quant. Spectrosc. Radiat. Transfer, 42, 539-550)
- Lacis & Oinas (1991, J. Geophys. Res., 96, 9027-9063)
- Fu & Liou (1992, J. Atmos. Sci., 49, 2139-2156)
- Clough & Iacono (1995, J. Geophys. Res., 100, 16519-16535)
- Mlawer, Taubman, Brown, Iacono, & Clough (1997, J. Geophys. Res., 102, 16663-16682)
- Mlawer et al. (2012, Phil. Trans. R. Soc. A, 370, 2520-2556; 2023, JGR, 128)
- Widger & Woodall (1976, Bull. Amer. Meteorol. Soc., 57, 1217-1219)
- Slingo (1989, J. Atmos. Sci., 46, 1419-1427, Table 1)
- Ebert & Curry (1992, J. Geophys. Res., 97, 3831-3836, Tables 2-3)
- Hu & Stamnes (1993, J. Climate, 6, 728-742)
- Cahalan et al. (1994, J. Atmos. Sci., 51, 2434-2455)
- Balkanski et al. (2007, Atmos. Chem. Phys., 7, 81-95)
- Etminan, Myhre, Highwood, & Shine (2016, Geophys. Res. Lett., 43, 12614-12623)
- Pincus et al. (2020, J. Adv. Model. Earth Syst., 12, e2019MS001919, RFMIP)

AOBA Formulation Reference: Sec. 4.4, Eqs. (4.7)-(4.8) (16-Band / 50 g-pt Correlated-k Radiation).
"""

from __future__ import annotations

import math
import torch

from aoba import constants as C
from aoba.kernels import can_use_crm_metal, dispatch_rad_ckd_metal
from aoba.radiation import RadiationOutput, background_ozone_vmr, toon_two_stream_sw

# Physical constants for Planck spectral integration and molecular mass ratios
C2_PLANCK_CM_K: float = 1.438776877   # Second radiation constant h*c/k_B [cm K]
M_AIR_G: float = 28.9647              # Dry air molar mass [g mol^-1]
M_CO2_G: float = 44.0095              # CO2 molar mass [g mol^-1]
M_O3_G: float = 47.9982               # O3 molar mass [g mol^-1]
M_CH4_G: float = 16.0425              # CH4 molar mass [g mol^-1]
M_N2O_G: float = 44.0128              # N2O molar mass [g mol^-1]
M_O2_G: float = 31.9988               # O2 molar mass [g mol^-1]
LW_DIFFUSIVITY: float = 1.66          # Elsasser diffusivity factor D = 1 / mu_1 [-]

# CMIP6 1850 piControl default trace-gas concentrations (Meinshausen et al., 2017)
CH4_PREIND_PPMV: float = 0.80825      # 1850 piControl CH4 [ppmv]
N2O_PREIND_PPMV: float = 0.27302      # 1850 piControl N2O [ppmv]

# ---------------------------------------------------------------------------
# 1. Longwave 10-Band (32 g-point) Correlated-k Tables (10 - 3250 cm^-1)
# ---------------------------------------------------------------------------
N_LW_BANDS: int = 10
N_LW_GPTS: int = 32

LW_BAND_EDGES_CM: tuple[tuple[float, float], ...] = (
    (0.0, 350.0),      # B0: H2O pure rotation (0-350 cm^-1; exact telescoping lower bound)
    (350.0, 500.0),    # B1: H2O rotation wing
    (500.0, 630.0),    # B2: H2O + CO2 15 um low-nu wing + N2O 17 um
    (630.0, 700.0),    # B3: CO2 15 um core (Q/P/R branches; stratospheric cooling)
    (700.0, 820.0),    # B4: CO2 15 um high-nu wing + H2O + O3 14.3 um
    (820.0, 980.0),    # B5: Atmospheric window 1 (H2O self-cont + CO2 10.4 um hot)
    (980.0, 1080.0),   # B6: O3 9.6 um + CO2 9.4 um hot + H2O cont
    (1080.0, 1180.0),  # B7: Atmospheric window 2 (H2O cont + N2O 8.6 um)
    (1180.0, 1400.0),  # B8: CH4 7.66 um + N2O 7.78 um + H2O nu2 wing (overlap)
    (1400.0, 0.0),     # B9: H2O 6.3 um + CO2 4.3 um + N2O 4.5 um (1400-inf; exact upper bound)
)

# MT_CKD band-averaged H2O self (e-type) & foreign (p-type) continuums [m^2 kg^-1 atm^-1] at 296 K
LW_C_SELF: tuple[float, ...] = (18.0, 7.5, 2.8, 1.3, 0.72, 0.36, 0.24, 0.20, 0.38, 1.05)
LW_C_FOR: tuple[float, ...] = (0.35, 0.088, 0.025, 0.009, 0.0045, 0.0016, 0.0013, 0.0016, 0.009, 0.038)

# Spectral cloud LW mass absorption coefficients [m^2 g^-1] (zenith; multiplied by D=1.66 in solver)
# Liquid: kappa_liq(b, r_e) = a_l + b_l / r_e (Hu & Stamnes, 1993; Stephens, 1978)
LW_CLD_LIQ_A: tuple[float, ...] = (0.045, 0.070, 0.085, 0.085, 0.075, 0.055, 0.050, 0.058, 0.075, 0.080)
LW_CLD_LIQ_B: tuple[float, ...] = (0.250, 0.220, 0.180, 0.160, 0.220, 0.350, 0.380, 0.320, 0.200, 0.180)
# Ice: kappa_ice(b, r_e) = a_i + b_i / r_e (Ebert & Curry, 1992, Table 3, 5-band mapped to 10 bands)
LW_CLD_ICE_A: tuple[float, ...] = (
    0.0053, 0.0053, 0.0030, 0.0030, 0.0016, 0.0016, 0.0016, 0.0020, 0.0033, 0.0033
)
LW_CLD_ICE_B: tuple[float, ...] = (
    0.880, 0.880, 1.020, 1.020, 1.166, 1.166, 1.166, 1.120, 1.033, 1.033
)
# Mineral dust LW mass absorption [m^2 kg^-1] (Balkanski et al., 2007: 9.5 um silicate peak)
LW_DUST_K_ABS: tuple[float, ...] = (85.0, 95.0, 105.0, 100.0, 105.0, 175.0, 210.0, 165.0, 110.0, 80.0)

# 32 LW g-point rows:
# (band, weight, k_h2o, g_h2o, k_co2, g_co2, p_voigt_co2_pa, b_temp_co2_k,
#  k_o3, g_o3, k_ch4, g_ch4, k_n2o, g_n2o)
LW_GPTS: tuple[tuple[float, ...], ...] = (
    # B0: 10-350 cm^-1 (H2O pure rotation)
    (0, 0.25,    0.70, 0.85,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 0.0,   0.80, 0.0,   0.75),
    (0, 0.30,    4.80, 0.85,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 0.0,   0.80, 0.0,   0.75),
    (0, 0.27,   30.00, 0.80,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 0.0,   0.80, 0.0,   0.75),
    (0, 0.18,  190.00, 0.75,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 0.0,   0.80, 0.0,   0.75),
    # B1: 350-500 cm^-1 (H2O rotation wing)
    (1, 0.35,    0.18, 0.85,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 0.0,   0.80, 0.0,   0.75),
    (1, 0.40,    1.45, 0.85,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 0.0,   0.80, 0.0,   0.75),
    (1, 0.25,   10.50, 0.80,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 0.0,   0.80, 0.0,   0.75),
    # B2: 500-630 cm^-1 (H2O + CO2 15 um low-nu wing + N2O 17 um)
    (2, 0.30,   0.060, 0.85,  0.013,   0.82, 120.0, 1200.0, 0.0,  0.50, 0.0,   0.80,  1.45, 0.75),
    (2, 0.30,   0.280, 0.85,  0.125,   0.80, 120.0,  950.0, 0.0,  0.50, 0.0,   0.80,  8.80, 0.75),
    (2, 0.25,   1.150, 0.80,  0.980,   0.76, 140.0,  650.0, 0.0,  0.50, 0.0,   0.80, 30.00, 0.72),
    (2, 0.15,   4.000, 0.75,  6.700,   0.70, 160.0,  350.0, 0.0,  0.50, 0.0,   0.80,  0.00, 0.72),
    # B3: 630-700 cm^-1 (CO2 15 um core: Q/P/R branches + stratospheric cooling)
    (3, 0.25,   0.050, 0.85,    3.2,   0.75, 160.0,  250.0, 0.0,  0.50, 0.0,   0.80,  0.0,  0.75),
    (3, 0.30,   0.100, 0.85,   21.0,   0.70, 200.0,  100.0, 0.0,  0.50, 0.0,   0.80,  0.0,  0.75),
    (3, 0.28,   0.200, 0.80,  120.0,   0.65, 240.0,    0.0, 0.0,  0.50, 0.0,   0.80,  0.0,  0.75),
    (3, 0.17,   0.400, 0.80,  600.0,   0.58, 280.0,    0.0, 0.0,  0.50, 0.0,   0.80,  0.0,  0.75),
    # B4: 700-820 cm^-1 (CO2 15 um high-nu wing + H2O + minor O3 14.3 um)
    (4, 0.34,   0.011, 0.85,  0.0092,  0.82, 120.0, 1300.0, 0.35, 0.55, 0.0,   0.80,  0.0,  0.75),
    (4, 0.28,   0.050, 0.85,  0.0950,  0.80, 120.0, 1000.0, 1.40, 0.55, 0.0,   0.80,  0.0,  0.75),
    (4, 0.22,   0.200, 0.80,  0.8600,  0.76, 140.0,  700.0, 3.20, 0.50, 0.0,   0.80,  0.0,  0.75),
    (4, 0.16,   0.800, 0.75,  6.5000,  0.70, 160.0,  350.0, 0.60, 0.50, 0.0,   0.80,  0.0,  0.75),
    # B5: 820-980 cm^-1 (Clean Window 1: H2O cont + CO2 10.4 um hot band)
    (5, 0.65,  0.0014, 0.85,  0.0006,  0.85, 100.0, 2000.0, 0.0,  0.50, 0.0,   0.80,  0.0,  0.75),
    (5, 0.35,  0.0115, 0.85,  0.0055,  0.82, 100.0, 2000.0, 0.0,  0.50, 0.0,   0.80,  0.0,  0.75),
    # B6: 980-1080 cm^-1 (O3 9.6 um band + CO2 9.4 um hot band + H2O cont)
    (6, 0.30,  0.0014, 0.85,  0.0012,  0.82, 100.0, 2000.0,  2.2, 0.55, 0.0,   0.80,  0.0,  0.75),
    (6, 0.40,  0.0048, 0.85,  0.0055,  0.80, 100.0, 2000.0, 16.0, 0.50, 0.0,   0.80,  0.0,  0.75),
    (6, 0.30,  0.0135, 0.85,  0.0160,  0.78, 100.0, 2000.0, 78.0, 0.42, 0.0,   0.80,  0.0,  0.75),
    # B7: 1080-1180 cm^-1 (Window 2: H2O cont + N2O 8.6 um)
    (7, 0.60,  0.0024, 0.85, 0.00025,  0.85, 100.0, 1800.0, 0.0,  0.50, 0.0,   0.80,  4.0,  0.75),
    (7, 0.40,  0.0220, 0.85, 0.00120,  0.82, 100.0, 1800.0, 0.0,  0.50, 0.0,   0.80, 27.0,  0.75),
    # B8: 1180-1400 cm^-1 (CH4 7.66 um + N2O 7.78 um + H2O nu2 wing — spectral overlap)
    (8, 0.34,   0.024, 0.85,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50,   7.5, 0.82,  9.5,  0.78),
    (8, 0.38,   0.180, 0.82,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50,  54.0, 0.80, 74.0,  0.75),
    (8, 0.28,   1.200, 0.78,   0.0,    0.80, 100.0,    0.0, 0.0,  0.50, 260.0, 0.78, 355.0, 0.72),
    # B9: 1400-3250 cm^-1 (H2O 6.3 um + CO2 4.3 um + N2O 4.5 um)
    (9, 0.30,    0.38, 0.85,   0.04,   0.80, 150.0,  400.0, 0.0,  0.50,   1.5, 0.80,  5.0,  0.75),
    (9, 0.40,    4.20, 0.80,   0.50,   0.75, 150.0,  200.0, 0.0,  0.50,   8.5, 0.78, 42.0,  0.72),
    (9, 0.30,   52.00, 0.75,  15.00,   0.65, 200.0,    0.0, 0.0,  0.50,   0.0, 0.75,  0.0,  0.70),
)

# ---------------------------------------------------------------------------
# 2. Shortwave 6-Band (18 g-point) Correlated-k Tables (0.20 - 4.00 um)
# ---------------------------------------------------------------------------
N_SW_BANDS: int = 6
N_SW_GPTS: int = 18

# ASTM E-490 / Kurucz solar irradiance fractions across the 6 SW bands (sum = 1.000)
#   SW0: 0.20-0.34 um (UV: O3 Hartley+Huggins)
#   SW1: 0.34-0.69 um (VIS/PAR: O3 Chappuis + Rayleigh)
#   SW2: 0.69-1.19 um (NIR-1: H2O + O2 A/B)
#   SW3: 1.19-2.38 um (NIR-2: H2O + CO2 + CH4)
#   SW4: 2.38-3.33 um (NIR-3: H2O 2.7 + CO2 2.7 + CH4 3.3)
#   SW5: 3.33-4.00 um (NIR-4: H2O + CO2 4.3 wing + CH4)
SW_SOLAR_FRAC: tuple[float, ...] = (0.036, 0.424, 0.340, 0.163, 0.029, 0.008)
SW_RAY_TAU0: tuple[float, ...] = (0.820, 0.108, 0.015, 0.0018, 0.0002, 0.00005)
# Surface albedo mapping: SW0 & SW1 use visible albedo; SW2..SW5 use NIR albedo
SW_IS_VIS_BAND: tuple[int, ...] = (1, 1, 0, 0, 0, 0)
# Fraction of SW1 (0.34-0.69 um) falling inside 0.40-0.70 um PAR window
SW1_PAR_FRAC: float = 0.94

# Spectral cloud SW optical properties per band:
# Liquid (Slingo, 1989, Table 1, 4-band mapped to 6 bands):
#   tau = chi * LWP * (a + b / r_e), 1 - omega = c + d * r_e, g = e + f * r_e
SW_LIQ_A: tuple[float, ...] = (2.817e-2, 2.817e-2, 2.682e-2, 2.264e-2, 1.281e-2, 1.281e-2)
SW_LIQ_B: tuple[float, ...] = (1.305,    1.305,    1.346,    1.454,    1.641,    1.641)
SW_LIQ_C: tuple[float, ...] = (-5.62e-8, -5.62e-8, -6.94e-6, 4.64e-4,  0.201,    0.260)
SW_LIQ_D: tuple[float, ...] = (1.63e-7,  1.63e-7,  2.35e-5,  1.24e-3,  7.56e-3,  8.50e-3)
SW_LIQ_E: tuple[float, ...] = (0.829,    0.829,    0.794,    0.754,    0.826,    0.840)
SW_LIQ_F: tuple[float, ...] = (2.482e-3, 2.482e-3, 4.226e-3, 6.560e-3, 4.353e-3, 4.100e-3)

# Ice (Ebert & Curry, 1992, Table 2, 5-band mapped to 6 bands):
SW_ICE_A: tuple[float, ...] = (3.448e-3, 3.448e-3, 3.448e-3, 3.448e-3, 3.448e-3, 3.448e-3)
SW_ICE_B: tuple[float, ...] = (2.431,    2.431,    2.431,    2.431,    2.431,    2.431)
SW_ICE_C: tuple[float, ...] = (1.0e-5,   1.0e-5,   1.10e-4,  1.861e-2, 0.250,    0.420)
SW_ICE_D: tuple[float, ...] = (0.0,      0.0,      1.405e-5, 8.328e-4, 7.00e-4,  4.00e-4)
SW_ICE_E: tuple[float, ...] = (0.7661,   0.7661,   0.7730,   0.794,    0.865,    0.895)
SW_ICE_F: tuple[float, ...] = (5.851e-4, 5.851e-4, 5.665e-4, 7.267e-4, 3.50e-4,  2.50e-4)

# Mineral dust SW spectral optical properties (Balkanski et al., 2007)
SW_DUST_KEXT: tuple[float, ...] = (720.0, 600.0, 460.0, 350.0, 260.0, 220.0)
SW_DUST_SSA: tuple[float, ...] = (0.82,  0.92,  0.95,  0.95,  0.90,  0.86)
SW_DUST_ASYM: tuple[float, ...] = (0.73,  0.70,  0.67,  0.64,  0.60,  0.58)

# 18 SW g-point rows:
# (band, weight, k_o3, k_h2o, g_h2o, k_co2, g_co2, k_ch4, g_ch4, k_o2)
SW_GPTS: tuple[tuple[float, ...], ...] = (
    # SW0 (UV 0.20-0.34 um): O3 Hartley + Huggins
    (0, 0.45,   28.0, 0.0,     0.80, 0.0,    0.80,   0.0, 0.80, 0.0),
    (0, 0.35,  320.0, 0.0,     0.80, 0.0,    0.80,   0.0, 0.80, 0.0),
    (0, 0.20, 3400.0, 0.0,     0.80, 0.0,    0.80,   0.0, 0.80, 0.0),
    # SW1 (VIS/PAR 0.34-0.69 um): O3 Chappuis
    (1, 0.55,   0.25, 0.0,     0.80, 0.0,    0.80,   0.0, 0.80, 0.0),
    (1, 0.45,   2.10, 0.0,     0.80, 0.0,    0.80,   0.0, 0.80, 0.0),
    # SW2 (NIR-1 0.69-1.19 um): H2O + O2 A/B bands
    (2, 0.52,   0.0,  0.00018, 0.85, 0.0,    0.80,   0.0, 0.80, 0.8e-5),
    (2, 0.32,   0.0,  0.00380, 0.82, 0.0,    0.80,   0.0, 0.80, 1.0e-4),
    (2, 0.16,   0.0,  0.03900, 0.78, 0.0,    0.80,   0.0, 0.80, 6.5e-4),
    # SW3 (NIR-2 1.19-2.38 um): H2O + CO2 (1.4/1.6/2.0 um) + CH4 (1.66/2.3 um)
    (3, 0.40,   0.0,  0.00045, 0.85, 0.0008, 0.90,  0.50, 1.00, 0.0),
    (3, 0.28,   0.0,  0.01050, 0.82, 0.0140, 0.88,  5.50, 1.00, 0.0),
    (3, 0.20,   0.0,  0.13000, 0.78, 0.1200, 0.85, 32.00, 0.98, 0.0),
    (3, 0.12,   0.0,  1.35000, 0.72, 0.8000, 0.82, 130.0, 0.95, 0.0),
    # SW4 (NIR-3 2.38-3.33 um): H2O 2.7 um + CO2 2.7 um + CH4 3.3 um
    (4, 0.32,   0.0,  0.022,   0.82, 0.015,  0.90,   3.5, 1.00, 0.0),
    (4, 0.30,   0.0,  0.340,   0.78, 0.180,  0.88,  28.0, 0.98, 0.0),
    (4, 0.24,   0.0,  3.800,   0.74, 1.200,  0.85, 150.0, 0.95, 0.0),
    (4, 0.14,   0.0,  32.00,   0.68, 5.000,  0.82, 480.0, 0.92, 0.0),
    # SW5 (NIR-4 3.33-4.00 um): H2O + CO2 4.3 um wing + CH4 3.3 um wing
    (5, 0.55,   0.0,  0.035,   0.82, 0.025,  0.90,  11.0, 1.00, 0.0),
    (5, 0.45,   0.0,  0.720,   0.76, 0.450,  0.85, 100.0, 0.95, 0.0),
)


def pack_ckd_tables_flat() -> tuple[float, ...]:
    """Pack all CKD lookup arrays & physical constants into a single flat float32 tuple for Metal.

    Layout offsets (verified in `kernels_physics.py: rad_ckd_col`):
      [0..19]     : LW_BAND_EDGES_CM (10 x 2: nu1, nu2)
      [20..29]    : LW_C_SELF (10)
      [30..39]    : LW_C_FOR (10)
      [40..49]    : LW_CLD_LIQ_A (10)
      [50..59]    : LW_CLD_LIQ_B (10)
      [60..69]    : LW_CLD_ICE_A (10)
      [70..79]    : LW_CLD_ICE_B (10)
      [80..89]    : LW_DUST_K_ABS (10)
      [90..537]   : LW_GPTS (32 x 14 = 448)
      [538..543]  : SW_SOLAR_FRAC (6)
      [544..549]  : SW_RAY_TAU0 (6)
      [550..555]  : SW_LIQ_A (6)
      [556..561]  : SW_LIQ_B (6)
      [562..567]  : SW_LIQ_C (6)
      [568..573]  : SW_LIQ_D (6)
      [574..579]  : SW_LIQ_E (6)
      [580..585]  : SW_LIQ_F (6)
      [586..591]  : SW_ICE_A (6)
      [592..597]  : SW_ICE_B (6)
      [598..603]  : SW_ICE_C (6)
      [604..609]  : SW_ICE_D (6)
      [610..615]  : SW_ICE_E (6)
      [616..621]  : SW_ICE_F (6)
      [622..627]  : SW_DUST_KEXT (6)
      [628..633]  : SW_DUST_SSA (6)
      [634..639]  : SW_DUST_ASYM (6)
      [640..819]  : SW_GPTS (18 x 10 = 180)
      [820..837]  : Physical constants (18)
    Total length = 838 floats.
    """
    out: list[float] = []
    for nu1, nu2 in LW_BAND_EDGES_CM:
        out.extend((float(nu1), float(nu2)))
    out.extend(float(x) for x in LW_C_SELF)
    out.extend(float(x) for x in LW_C_FOR)
    out.extend(float(x) for x in LW_CLD_LIQ_A)
    out.extend(float(x) for x in LW_CLD_LIQ_B)
    out.extend(float(x) for x in LW_CLD_ICE_A)
    out.extend(float(x) for x in LW_CLD_ICE_B)
    out.extend(float(x) for x in LW_DUST_K_ABS)
    for row in LW_GPTS:
        out.extend(float(x) for x in row)
    for arr in (
        SW_SOLAR_FRAC, SW_RAY_TAU0,
        SW_LIQ_A, SW_LIQ_B, SW_LIQ_C, SW_LIQ_D, SW_LIQ_E, SW_LIQ_F,
        SW_ICE_A, SW_ICE_B, SW_ICE_C, SW_ICE_D, SW_ICE_E, SW_ICE_F,
        SW_DUST_KEXT, SW_DUST_SSA, SW_DUST_ASYM,
    ):
        out.extend(float(x) for x in arr)
    for row in SW_GPTS:
        out.extend(float(x) for x in row)
    out.extend((
        float(C.STEFAN_BOLTZMANN),
        float(C.O3_G1),
        float(C.O3_G2),
        float(C.O3_G3),
        float(C.CP_D),
        float(C.CP_V),
        float(C.EPS_WATER),
        float(C.P_REF),
        float(C.GRAVITY),
        float(C.O2_MIXING_RATIO),
        float(C2_PLANCK_CM_K),
        float(LW_DIFFUSIVITY),
        float(SW1_PAR_FRAC),
        float(M_CO2_G / M_AIR_G),
        float(M_O3_G / M_AIR_G),
        float(M_CH4_G / M_AIR_G),
        float(M_N2O_G / M_AIR_G),
        float(M_O2_G / M_AIR_G),
    ))
    return tuple(out)


def planck_band_fraction(nu1_cm: float, nu2_cm: float, temp_k: torch.Tensor) -> torch.Tensor:
    """Exact fractional blackbody irradiance in `[nu1_cm, nu2_cm]` via Widger & Woodall (1976).

    Convention: `nu1_cm <= 0.0` means lower bound 0 (tail=1); `nu2_cm <= 0.0` means upper bound +inf (tail=0).
    """
    inv_t = 1.0 / torch.clamp(temp_k, min=100.0)
    pref = 15.0 / (math.pi ** 4)

    def _tail_frac(nu: float, is_upper: bool) -> torch.Tensor:
        if nu <= 0.0:
            return torch.zeros_like(inv_t) if is_upper else torch.ones_like(inv_t)
        x = (C2_PLANCK_CM_K * nu) * inv_t
        acc = torch.zeros_like(inv_t)
        for n in range(1, 7):
            nx = n * x
            e = torch.exp(-torch.clamp(nx, max=60.0))
            acc = acc + e * (1.0 / float(n ** 4)) * (((nx + 3.0) * nx + 6.0) * nx + 6.0)
        return torch.clamp(pref * acc, 0.0, 1.0)

    return torch.clamp(_tail_frac(nu1_cm, False) - _tail_frac(nu2_cm, True), min=0.0)


def compute_ckd_radiation(
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
    co2_ppmv: float = C.CO2_PREIND_PPMV,
    ch4_ppmv: float = CH4_PREIND_PPMV,
    n2o_ppmv: float = N2O_PREIND_PPMV,
    emiss_sfc: float | torch.Tensor = 0.98,
    re_liq_um: float | torch.Tensor = 10.0,
    re_ice_um: float | torch.Tensor = 48.0,
    cloud_sw_inhomog: float = 0.85,
    albedo_vis_dif: torch.Tensor | None = None,
    albedo_nir_dif: torch.Tensor | None = None,
    q_dust_c: torch.Tensor | None = None,
    o3_vmr_c: torch.Tensor | None = None,
    o3_lw_coupling: bool = True,
) -> RadiationOutput:
    """Evaluate 16-band (32 LW + 18 SW g-point) correlated-k fluxes & heating rates.

    All vertical inputs use AOBA's bottom-up convention (`0 = surface`, `nz = TOA`).
    """
    if can_use_crm_metal(pres_c) and t_sfc.numel() > 0:
        sw_up, sw_dn, lw_up, lw_dn, q_rad_3d, sfc_out = dispatch_rad_ckd_metal(
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
            ch4_ppmv=ch4_ppmv,
            n2o_ppmv=n2o_ppmv,
            emiss_sfc=emiss_sfc,
            re_liq_um=re_liq_um,
            re_ice_um=re_ice_um,
            cloud_sw_inhomog=cloud_sw_inhomog,
            albedo_vis_dif=albedo_vis_dif,
            albedo_nir_dif=albedo_nir_dif,
            q_dust_c=q_dust_c,
            o3_vmr_c=o3_vmr_c,
            o3_lw_coupling=o3_lw_coupling,
            ckd_tables=pack_ckd_tables_flat(),
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

    # Flip vertical axis to top-down [0 = TOA, N - 1 = lowest layer]
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
    o3_vmr = (
        torch.clamp(torch.flip(o3_vmr_c, dims=[0]), min=0.0)
        if o3_vmr_c is not None
        else background_ozone_vmr(p_c)
    )
    if isinstance(re_liq_um, torch.Tensor):
        re_l = torch.clamp(torch.flip(re_liq_um, dims=[0]), min=3.0, max=40.0)
    else:
        re_l = torch.full_like(p_c, max(min(float(re_liq_um), 40.0), 3.0))
    if isinstance(re_ice_um, torch.Tensor):
        re_i = torch.clamp(torch.flip(re_ice_um, dims=[0]), min=10.0, max=150.0)
    else:
        re_i = torch.full_like(p_c, max(min(float(re_ice_um), 150.0), 10.0))

    N = p_c.shape[0]
    col_shape = t_sfc.shape
    dtype = p_c.dtype
    device = p_c.device

    dm_dry = rho_d * dz
    dp_frac = torch.clamp(p_w[1:] - p_w[:-1], min=1.0) / C.P_REF
    lwp_gm2 = 1000.0 * qc * dm_dry
    iwp_gm2 = 1000.0 * qi * dm_dry
    dm_dust = (q_dust * dm_dry) if q_dust is not None else None

    # Gas column masses per layer [kg_gas m^-2]
    # Stratospheric photochemical decay above 80 hPa for CH4 and N2O (WACCM/RFMIP)
    strat_falloff = torch.pow(torch.clamp(p_c / 8000.0, max=1.0), 0.38)
    u_h2o = qv * dm_dry
    u_co2 = (max(float(co2_ppmv), 0.0) * 1.0e-6 * (M_CO2_G / M_AIR_G)) * dm_dry
    u_o3 = (o3_vmr * (M_O3_G / M_AIR_G)) * dm_dry
    u_ch4 = (max(float(ch4_ppmv), 0.0) * 1.0e-6 * (M_CH4_G / M_AIR_G)) * strat_falloff * dm_dry
    u_n2o = (max(float(n2o_ppmv), 0.0) * 1.0e-6 * (M_N2O_G / M_AIR_G)) * strat_falloff * dm_dry
    u_o2 = (C.O2_MIXING_RATIO * (M_O2_G / M_AIR_G)) * dm_dry
    # Analytic overhead O3 mass above model top (prevents top-layer Hartley UV spikes)
    u_o3_above = (o3_vmr[0] * (M_O3_G / M_AIR_G) * (p_w[0] / C.GRAVITY)) / (C.O3_G2 + 1.0)

    # ------------------------------------------------------------------
    # A. Shortwave 6-Band (18 g-point) Delta-Quadrature Two-Stream
    # ------------------------------------------------------------------
    sw_up_td = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)
    sw_dn_td = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)
    sfc_sw1_dir = torch.zeros(col_shape, dtype=dtype, device=device)
    sfc_sw1_dif = torch.zeros(col_shape, dtype=dtype, device=device)
    sfc_sw2_dir = torch.zeros(col_shape, dtype=dtype, device=device)
    sfc_sw2_dif = torch.zeros(col_shape, dtype=dtype, device=device)
    sfc_par_wm2 = torch.zeros(col_shape, dtype=dtype, device=device)

    chi = float(cloud_sw_inhomog)
    mu0_safe = torch.clamp(mu0, min=1.0e-3)
    vis_dif = albedo_vis if albedo_vis_dif is None else albedo_vis_dif
    nir_dif = albedo_nir if albedo_nir_dif is None else albedo_nir_dif

    # Precompute per-band cloud + dust + Rayleigh optical properties (6 bands)
    sw_band_props: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for b in range(N_SW_BANDS):
        tau_ray = SW_RAY_TAU0[b] * dp_frac
        tau_l = chi * lwp_gm2 * (SW_LIQ_A[b] + SW_LIQ_B[b] / re_l)
        om_l = torch.clamp(1.0 - (SW_LIQ_C[b] + SW_LIQ_D[b] * re_l), 0.0, 0.999999)
        g_l = torch.clamp(SW_LIQ_E[b] + SW_LIQ_F[b] * re_l, -0.99, 0.99)

        tau_i = chi * iwp_gm2 * (SW_ICE_A[b] + SW_ICE_B[b] / re_i)
        om_i = torch.clamp(1.0 - (SW_ICE_C[b] + SW_ICE_D[b] * re_i), 0.0, 0.999999)
        g_i = torch.clamp(SW_ICE_E[b] + SW_ICE_F[b] * re_i, -0.99, 0.99)

        tau_b = tau_ray + tau_l + tau_i
        scat_b = tau_ray + tau_l * om_l + tau_i * om_i
        asym_b = tau_l * om_l * g_l + tau_i * om_i * g_i
        if dm_dust is not None:
            tau_d = SW_DUST_KEXT[b] * dm_dust
            tau_b = tau_b + tau_d
            scat_b = scat_b + tau_d * SW_DUST_SSA[b]
            asym_b = asym_b + tau_d * (SW_DUST_SSA[b] * SW_DUST_ASYM[b])
        sw_band_props.append((tau_b, scat_b, asym_b))

    p_h2o_norm = torch.clamp((p_c + 80.0) / C.P_REF, min=1.0e-4)
    p_co2_norm_sw = torch.clamp((p_c + 150.0) / C.P_REF, min=1.0e-4)
    p_ch4_norm_sw = torch.clamp((p_c + 100.0) / C.P_REF, min=1.0e-4)
    p_o2_norm = torch.pow(torch.clamp(p_c / C.P_REF, min=1.0e-4), 0.85)

    for row in SW_GPTS:
        b = int(row[0])
        w_g, ko, kh, gh, kc, gc, km, gm, ko2 = row[1:]
        tau_base, scat_base, asym_base = sw_band_props[b]
        tau_g = tau_base
        if ko > 0.0:
            tau_g = tau_g + ko * u_o3
        if kh > 0.0:
            tau_g = tau_g + (kh * torch.pow(p_h2o_norm, gh)) * u_h2o
        if kc > 0.0:
            tau_g = tau_g + (kc * torch.pow(p_co2_norm_sw, gc)) * u_co2
        if km > 0.0:
            tau_g = tau_g + (km * torch.pow(p_ch4_norm_sw, gm)) * u_ch4
        if ko2 > 0.0:
            tau_g = tau_g + (ko2 * p_o2_norm) * u_o2

        om_g = scat_base / torch.clamp(tau_g, min=1.0e-8)
        g_g = asym_base / torch.clamp(scat_base, min=1.0e-8)

        f_toa_g = (w_g * SW_SOLAR_FRAC[b]) * s_toa
        if ko > 0.0:
            f_toa_g = f_toa_g * torch.exp(-torch.clamp(ko * u_o3_above / mu0_safe, max=40.0))

        is_vis = SW_IS_VIS_BAND[b] == 1
        r_dir = albedo_vis if is_vis else albedo_nir
        r_dif = vis_dif if is_vis else nir_dif

        u_g, d_g, s_dir, s_dif = toon_two_stream_sw(
            tau_g, om_g, g_g, mu0, f_toa_g, r_dir, albedo_sfc_dif=r_dif, return_sfc_split=True
        )
        sw_up_td = sw_up_td + u_g
        sw_dn_td = sw_dn_td + d_g
        if is_vis:
            sfc_sw1_dir = sfc_sw1_dir + s_dir
            sfc_sw1_dif = sfc_sw1_dif + s_dif
            if b == 1:
                sfc_par_wm2 = sfc_par_wm2 + SW1_PAR_FRAC * (s_dir + s_dif)
        else:
            sfc_sw2_dir = sfc_sw2_dir + s_dir
            sfc_sw2_dif = sfc_sw2_dif + s_dif

    # ------------------------------------------------------------------
    # B. Longwave 10-Band (32 g-point) Linear-in-tau Absorption Two-Stream
    # ------------------------------------------------------------------
    p_atm = p_c / 101325.0
    e_atm = (qv * p_atm) / (C.EPS_WATER + qv)
    t_self = torch.exp(
        torch.clamp(1800.0 * (1.0 / torch.clamp(t_c, min=180.0) - 1.0 / 296.0), min=-3.0, max=6.0)
    )
    inv_t_diff = (1.0 / 260.0) - (1.0 / torch.clamp(t_c, min=160.0))

    eb_int = C.STEFAN_BOLTZMANN * (torch.clamp(t_w, min=100.0) ** 4)
    eb_sfc = C.STEFAN_BOLTZMANN * (torch.clamp(t_sfc, min=100.0) ** 4)

    # Precompute band Planck fluxes & band-level continuum + cloud + dust optical depths
    # Telescoping bounds (0 -> 350 -> ... -> 1400 -> inf) guarantee sum_b B_b(T) == sigma * T^4 identically.
    b_int = [planck_band_fraction(nu1, nu2, t_w) * eb_int for (nu1, nu2) in LW_BAND_EDGES_CM]
    b_sfc = [planck_band_fraction(nu1, nu2, t_sfc) * eb_sfc for (nu1, nu2) in LW_BAND_EDGES_CM]

    tau_lw_base: list[torch.Tensor] = []
    for b in range(N_LW_BANDS):
        k_liq = LW_CLD_LIQ_A[b] + LW_CLD_LIQ_B[b] / re_l
        k_ice = LW_CLD_ICE_A[b] + LW_CLD_ICE_B[b] / re_i
        tb = (LW_C_SELF[b] * e_atm * t_self + LW_C_FOR[b] * p_atm) * u_h2o + k_liq * lwp_gm2 + k_ice * iwp_gm2
        if dm_dust is not None:
            tb = tb + LW_DUST_K_ABS[b] * dm_dust
        tau_lw_base.append(tb)

    emiss_val = emiss_sfc if isinstance(emiss_sfc, torch.Tensor) else torch.full_like(t_sfc, float(emiss_sfc))
    r_sfc_lw = 1.0 - torch.clamp(emiss_val, 0.0, 1.0)

    lw_up_td = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)
    lw_dn_td = torch.zeros((N + 1,) + col_shape, dtype=dtype, device=device)

    for row in LW_GPTS:
        b = int(row[0])
        w_g, kh, gh, kc, gc, pvc, btc, ko, go, km, gm, kn, gn = row[1:]
        tau = tau_lw_base[b]
        if kh > 0.0:
            tau = tau + (kh * torch.pow(p_h2o_norm, gh)) * u_h2o
        if kc > 0.0:
            p_co2_f = torch.pow(torch.clamp((p_c + pvc) / C.P_REF, min=1.0e-4), gc)
            if btc != 0.0:
                tfac = torch.exp(torch.clamp(btc * inv_t_diff, min=-4.0, max=4.0))
                tau = tau + (kc * p_co2_f * tfac) * u_co2
            else:
                tau = tau + (kc * p_co2_f) * u_co2
        if ko > 0.0 and o3_lw_coupling:
            tau = tau + (ko * torch.pow(torch.clamp((p_c + 120.0) / C.P_REF, min=1.0e-4), go)) * u_o3
        if km > 0.0:
            tau = tau + (km * torch.pow(p_ch4_norm_sw, gm)) * u_ch4
        if kn > 0.0:
            tau = tau + (kn * torch.pow(p_ch4_norm_sw, gn)) * u_n2o

        dtau = torch.clamp(LW_DIFFUSIVITY * tau, min=1.0e-6, max=50.0)
        exp_lt = torch.exp(-dtau)
        one_minus_e = -torch.expm1(-dtau)
        alpha = torch.where(
            dtau > 1.0e-3,
            one_minus_e / dtau,
            1.0 - 0.5 * dtau + (1.0 / 6.0) * dtau * dtau,
        )
        w1 = 1.0 - alpha
        w2 = torch.clamp(alpha - exp_lt, min=0.0)

        fb0 = (w_g * b_int[b])[:-1]
        fb1 = (w_g * b_int[b])[1:]
        src_dn = w2 * fb0 + w1 * fb1
        src_up = w1 * fb0 + w2 * fb1

        cur_dn = torch.zeros(col_shape, dtype=dtype, device=device)
        for n in range(N):
            cur_dn = cur_dn * exp_lt[n] + src_dn[n]
            lw_dn_td[n + 1] = lw_dn_td[n + 1] + cur_dn

        cur_up = emiss_val * (w_g * b_sfc[b]) + r_sfc_lw * cur_dn
        lw_up_td[N] = lw_up_td[N] + cur_up
        for n in range(N - 1, -1, -1):
            cur_up = cur_up * exp_lt[n] + src_up[n]
            lw_up_td[n] = lw_up_td[n] + cur_up

    # ------------------------------------------------------------------
    # C. Flip back to AOBA bottom-up indexing & exact heating rates
    # ------------------------------------------------------------------
    sw_up = torch.flip(sw_up_td, dims=[0])
    sw_dn = torch.flip(sw_dn_td, dims=[0])
    lw_up = torch.flip(lw_up_td, dims=[0])
    lw_dn = torch.flip(lw_dn_td, dims=[0])

    cp_layer = (C.CP_D + C.CP_V * torch.clamp(qv_c, min=0.0)) * (rho_d_c * dz_c)
    fnet_sw_up = sw_up - sw_dn
    fnet_lw_up = lw_up - lw_dn
    q_rad_sw = -(fnet_sw_up[1:] - fnet_sw_up[:-1]) / cp_layer
    q_rad_lw = -(fnet_lw_up[1:] - fnet_lw_up[:-1]) / cp_layer
    q_rad_net = q_rad_sw + q_rad_lw

    return RadiationOutput(
        sw_flux_up=sw_up,
        sw_flux_dn=sw_dn,
        lw_flux_up=lw_up,
        lw_flux_dn=lw_dn,
        q_rad_sw=q_rad_sw,
        q_rad_lw=q_rad_lw,
        q_rad_net=q_rad_net,
        toa_insolation=sw_dn[-1],
        toa_asr=sw_dn[-1] - sw_up[-1],
        toa_olr=lw_up[-1],
        sfc_sw_net=sw_dn[0] - sw_up[0],
        sfc_lw_net=lw_dn[0] - lw_up[0],
        sfc_par_wm2=sfc_par_wm2,
        sfc_ppfd=C.UMOL_PER_JOULE_PAR * sfc_par_wm2,
        sfc_sw1_dir_dn=sfc_sw1_dir,
        sfc_sw1_dif_dn=sfc_sw1_dif,
        sfc_sw2_dir_dn=sfc_sw2_dir,
        sfc_sw2_dif_dn=sfc_sw2_dif,
    )
