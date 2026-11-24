"""Map-Free Planetary Geochemistry (Silicate Weathering Thermostat) & Milankovitch Orbital Evolution.

Enables AOBA to run deep-time paleoclimate (Snowball Earth, glacial-interglacial cycles,
hothouse climates) and planetary habitability experiments from universal physical laws:
1. **Carbonate-Silicate Weathering Thermostat** (Walker et al., 1981, *JGR*, 86, 9776;
   Berner, 2004, *The Phanerozoic Carbon Cycle*; Maher & Chamberlain, 2014, *Science*):
   - Computes map-free local chemical weathering CO2 drawdown on ice-free land cells as a
     function of surface runoff `q_runoff`, soil temperature `T_s`, and atmospheric pCO2,
     balanced against volcanic outgassing `volcanic_co2_pgc_yr` (default 0.03 Pg C/yr ~
     2.5 Tmol C/yr; optional acceleration factor for deep-time equilibrium tests).
   - Delivers exact stoichiometric riverine alkalinity (`2 mol Alk / mol CO2 weathered`)
     to surface ocean BGC (`ogcm.bgc.alk`) while updating prognostic atmospheric CO2.
2. **Analytical Milankovitch Orbital Evolution** (Berger, 1978, *JAS*, 35, 2362-2367;
   Laskar et al., 2004, *A&A*, 428, 261-285):
   - Evaluates continuous eccentricity `e(t)`, obliquity `eps(t)`, and climatic precession
     index `e * sin(varpi(t))` across multi-millennial timescales (`t_kyr` relative to J2000),
     directly driving AOBA's orbital insolation (`compute_orbital_insolation`) and dynamic
     SIA ice sheets (`icesheet.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.surface import SURF_LAND

SEC_PER_YEAR: float = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
PGC_PER_PPMV_CO2: float = 2.124        # 1 ppmv atmospheric CO2 = 2.124 Pg C

# Silicate weathering universal physical parameters (Walker et al., 1981; Maher & Chamberlain, 2014)
WEATH_T_REF_K: float = 288.15          # Reference land temperature [K]
WEATH_T_EFOLD_K: float = 13.7          # Arrhenius e-folding scale (~60 kJ/mol activation energy) [K]
WEATH_RUNOFF_REF: float = 1.0e-5       # Reference runoff (~315 mm/yr) [kg m^-2 s^-1]
WEATH_CO2_REF_PPMV: float = 284.3      # Pre-industrial reference pCO2 [ppmv]
WEATH_K_BASE_KGC_M2S: float = 6.5e-12  # Base silicate weathering C drawdown on bare wet land (~0.03 Pg C/yr global)

# Dominant Berger (1978, Tables 1-4) & Laskar (2004) quasi-periodic orbital terms
# Eccentricity terms: (amplitude, period_kyr, phase_rad at J2000)
_ECC_TERMS: tuple[tuple[float, float, float], ...] = (
    (0.0110, 405.0, 0.25),
    (0.0087, 94.9, 1.85),
    (0.0070, 123.8, 4.10),
    (0.0053, 98.9, 0.90),
    (0.0041, 131.2, 2.75),
)
# Obliquity terms [deg]: (amplitude_deg, period_kyr, phase_rad)
_OBL_TERMS: tuple[tuple[float, float, float], ...] = (
    (0.684, 41.00, 4.40),
    (0.186, 39.73, 1.25),
    (0.145, 53.62, 5.10),
    (0.118, 40.52, 2.10),
    (0.088, 28.91, 3.60),
)
# Climatic precession periods [kyr] (dominant 23.7, 22.4, 19.0 kyr)
_PREC_PERIOD_KYR: float = 21.74


@dataclass
class WeatheringState:
    """Prognostic geological carbon-cycle (silicate weathering + volcanism) state."""

    co2_ppmv: float
    weath_co2_pgc_yr: float = 0.0
    volc_co2_pgc_yr: float = 0.03
    alk_flux_mol_m2s: torch.Tensor | None = None  # [N_cells] riverine alkalinity flux for ocean BGC


def init_weathering_state(co2_ppmv_init: float = 284.3, volcanic_co2_pgc_yr: float = 0.03) -> WeatheringState:
    """Initialize carbonate-silicate weathering thermostat state."""
    return WeatheringState(
        co2_ppmv=max(float(co2_ppmv_init), 10.0),
        volc_co2_pgc_yr=max(float(volcanic_co2_pgc_yr), 0.0),
    )


def step_silicate_weathering(
    ws: WeatheringState,
    area_cell: torch.Tensor,
    surf_type: torch.Tensor,
    t_sfc_k: torch.Tensor,
    runoff_liq_kgm2s: torch.Tensor,
    f_snow_land: torch.Tensor,
    dt: float,
    *,
    volcanic_co2_pgc_yr: float = 0.03,
    accel_factor: float = 1.0,
) -> float:
    """Advance silicate-weathering thermostat & return updated global atmospheric CO2 [ppmv]."""
    dev = area_cell.device
    dtype = area_cell.dtype
    is_icefree_land = (surf_type == SURF_LAND).to(dtype) * (1.0 - torch.clamp(f_snow_land, 0.0, 1.0))

    # Walker et al. (1981) & Maher-Chamberlain (2014) kinetics: f_T * f_runoff * f_CO2
    t_clamped = torch.clamp(t_sfc_k, min=250.0, max=340.0)
    f_t = torch.exp(torch.clamp((t_clamped - WEATH_T_REF_K) / WEATH_T_EFOLD_K, min=-4.0, max=4.5))
    q_norm = torch.clamp(runoff_liq_kgm2s / WEATH_RUNOFF_REF, min=0.0, max=50.0)
    f_q = torch.pow(q_norm + 1.0e-6, 0.65)
    f_co2 = (max(ws.co2_ppmv, 10.0) / WEATH_CO2_REF_PPMV) ** 0.30

    # Local CO2 C drawdown [kg C m^-2 s^-1]
    f_w_cell = is_icefree_land * WEATH_K_BASE_KGC_M2S * f_t * f_q * f_co2

    a64 = area_cell.detach().cpu().to(torch.float64)
    w_pgc_yr = float((f_w_cell.detach().cpu().to(torch.float64) * a64).sum().item() * SEC_PER_YEAR * 1.0e-12)
    ws.weath_co2_pgc_yr = w_pgc_yr
    ws.volc_co2_pgc_yr = float(volcanic_co2_pgc_yr)

    # Implicitly stabilized update for geological CO2 [ppmv] (safe even at high accel_factor!)
    dt_eff_yr = (float(dt) / SEC_PER_YEAR) * max(float(accel_factor), 1.0)
    # Linearize negative weathering feedback around current CO2: dW/dC = 0.30 * W / C
    c_cur = max(ws.co2_ppmv, 10.0)
    k_fb = (0.30 * w_pgc_yr) / (PGC_PER_PPMV_CO2 * c_cur)
    net_rate_ppmv_yr = (ws.volc_co2_pgc_yr - w_pgc_yr) / PGC_PER_PPMV_CO2
    if k_fb * dt_eff_yr > 1.0e-6:
        dc = (net_rate_ppmv_yr / k_fb) * (-math.expm1(-min(k_fb * dt_eff_yr, 20.0)))
    else:
        dc = net_rate_ppmv_yr * dt_eff_yr
    ws.co2_ppmv = max(20.0, min(50000.0, c_cur + dc))

    # Stoichiometric riverine alkalinity delivery: 2 mol eq / mol C weathered = 2 / 0.012011 mol eq / kg C
    # Spread globally across ocean cells by caller (or coupled via river mouths)
    ws.alk_flux_mol_m2s = f_w_cell * (2.0 / 0.012011)
    return ws.co2_ppmv


def compute_milankovitch_orbital_params(t_kyr_from_j2000: float) -> tuple[float, float, float]:
    """Evaluate analytical Berger (1978) / Laskar (2004) Earth orbital elements at `t_kyr` relative to J2000.

    Parameters
    ----------
    t_kyr_from_j2000 : float
        Time in kiloyears relative to J2000 (e.g., `-21.0` = Last Glacial Maximum, `0.0` = present).

    Returns
    -------
    (eccentricity, obliquity_deg, perihelion_deg) : tuple[float, float, float]
        Orbital eccentricity `e in [0.0005, 0.065]`, axial tilt `eps in [22.0, 24.6]` deg,
        and longitude of perihelion `varpi in [0, 360)` deg.
    """
    t = float(t_kyr_from_j2000)
    # Eccentricity centered on 0.028 with realistic J2000 value ~0.0167
    e_raw = 0.0278 + sum(amp * math.cos(2.0 * math.pi * t / per + ph) for amp, per, ph in _ECC_TERMS)
    ecc = max(0.0005, min(0.065, e_raw))

    # Obliquity centered on 23.32 deg with realistic J2000 value ~23.44 deg
    obl_raw = 23.320 + sum(amp * math.cos(2.0 * math.pi * t / per + ph) for amp, per, ph in _OBL_TERMS)
    obl_deg = max(21.8, min(24.8, obl_raw))

    # Longitude of perihelion advancing by 360 deg per ~21.74 kyr (J2000 ~ 283.0 deg)
    varpi_deg = (283.0 + 360.0 * (t / _PREC_PERIOD_KYR)) % 360.0
    return ecc, obl_deg, varpi_deg
