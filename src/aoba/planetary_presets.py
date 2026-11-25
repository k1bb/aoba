"""Universal Planetary, Paleoclimate Snowball, Aquaplanet, & Tidally Locked Exoplanet Presets.

Exploits AOBA's 100% map-free universal physical formulation so the exact same codebase runs
modern Earth, Neoproterozoic Snowball Earth, idealized Aquaplanets (Neale & Hoskins, 2000),
and synchronously rotating terrestrial exoplanets such as TRAPPIST-1e (Turbet et al., 2018;
Fauchez et al., 2020 THAI intercomparison) via a single `planet_preset` switch.

Supported Presets (`planet_preset`):
- `"earth"` (default): Standard Earth boundary conditions (leaves all user config unchanged).
- `"aquaplanet"`: 100% global ocean, zero continents/topography, zero obliquity option, ideal
  for ITCZ / MJO / superparameterization dynamical benchmarks (Neale & Hoskins, 2000).
- `"snowball_earth"`: Faint young Sun (`S_0 = 1280 W m^-2`), cold glaciated initial state,
  silicate-weathering CO2 recovery thermostat enabled (Pierrehumbert et al., 2011).
- `"tidally_locked"`: Synchronously rotating Earth-like ocean world (`T_rot = 10 days`,
  `enable_tidal_lock = True`, fixed substellar point -> eyeball climate; Yang et al., 2013).
- `"trappist1e"`: TRAPPIST-1e habitable-zone exoplanet (`S_0 = 900 W m^-2`, `T_rot = 6.101 d`,
  `enable_tidal_lock = True`, `co2_ppmv = 1000.0`, aquaplanet surface; Fauchez et al., 2020).
- `"mars_like"`: Cold arid terrestrial planet (`S_0 = 589 W m^-2`, 100% dry bare land, low
  moisture, `omega = 7.088e-5 rad s^-1`, `obliquity = 25.19 deg`).

AOBA Formulation Reference: Sec. 11.3, Eq. (11.5) (General Planetary / Exoplanet Presets & Tidal Lock).
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any
import torch

from aoba import constants as C

VALID_PLANET_PRESETS: tuple[str, ...] = (
    "earth",
    "aquaplanet",
    "snowball_earth",
    "tidally_locked",
    "trappist1e",
    "mars_like",
)

PLANET_PRESET_SPECS: dict[str, dict[str, Any]] = {
    "earth": {},
    "aquaplanet": {
        "use_earth2014": False,
        "obliquity_deg": 0.0,
        "eccentricity": 0.0,
        "enable_river_routing": False,
        "enable_land_carbon": False,
        "enable_dust": False,
    },
    "snowball_earth": {
        "solar_constant": 1280.0,
        "co2_ppmv": 200.0,
        "enable_silicate_weathering": True,
        "enable_snow_bc_darkening": True,
    },
    "tidally_locked": {
        "use_earth2014": False,
        "enable_tidal_lock": True,
        "omega": 2.0 * math.pi / (10.0 * C.SECONDS_PER_DAY),
        "obliquity_deg": 0.0,
        "eccentricity": 0.0,
        "diurnal_cycle": False,
    },
    "trappist1e": {
        "use_earth2014": False,
        "enable_tidal_lock": True,
        "solar_constant": 900.0,
        "omega": 2.0 * math.pi / (6.101 * C.SECONDS_PER_DAY),
        "obliquity_deg": 0.0,
        "eccentricity": 0.0,
        "co2_ppmv": 1000.0,
        "diurnal_cycle": False,
        "enable_river_routing": False,
        "enable_land_carbon": False,
        "enable_dust": False,
    },
    "mars_like": {
        "use_earth2014": False,
        "solar_constant": 589.0,
        "omega": 7.0882e-5,
        "obliquity_deg": 25.19,
        "eccentricity": 0.0934,
        "co2_ppmv": 40000.0,
        "enable_ocean_bgc": False,
        "enable_river_routing": False,
        "enable_land_carbon": False,
    },
}


def resolve_planet_preset_config(cfg: Any) -> Any:
    """Return a copy of `AOBAConfig` with `planet_preset` physical defaults applied (no-op for `'earth'`)."""
    key = str(getattr(cfg, "planet_preset", "earth") or "earth").strip().lower()
    if key not in VALID_PLANET_PRESETS:
        raise ValueError(f"Unknown planet_preset '{key}'; expected one of {VALID_PLANET_PRESETS}")
    if key == "earth":
        return cfg
    spec = PLANET_PRESET_SPECS[key]
    updates = {k: v for k, v in spec.items() if hasattr(cfg, k)}
    return replace(cfg, **updates)


# AOBA Formulation Eq. (11.5)
def compute_tidally_locked_insolation(
    lat_cell: torch.Tensor,
    lon_cell: torch.Tensor,
    solar_constant: float,
    substellar_lon_deg: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute synchronous tidally locked `(mu0, s_toa, hour_angle)` with fixed substellar point `(0, lon_sub)`."""
    lon_sub = math.radians(float(substellar_lon_deg))
    dlon = lon_cell - lon_sub
    mu0 = torch.clamp(torch.cos(lat_cell) * torch.cos(dlon), min=0.0)
    s_toa = torch.full_like(mu0, float(solar_constant))
    return mu0, s_toa, dlon


def apply_planet_initial_surface(sfc: Any, preset: str) -> None:
    """Apply planetary surface mask/cryosphere initialization for `'aquaplanet'`, `'snowball_earth'`, `'mars_like'`."""
    from aoba.surface import SURF_LAND, SURF_OCEAN

    key = str(preset or "earth").strip().lower()
    if key in ("aquaplanet", "tidally_locked", "trappist1e"):
        sfc.surf_type.fill_(SURF_OCEAN)
        sfc.z_topo.zero_()
        sfc.sigma_topo.zero_()
        sfc.t_ocean_ml.fill_(288.0)
        sfc.m_seaice.zero_()
        sfc.swe_seaice.zero_()
    elif key == "snowball_earth":
        # Globally glaciated high-albedo start (3 m sea ice + 100 kg/m^2 snow cover everywhere)
        is_ocn = sfc.surf_type == SURF_OCEAN
        sfc.t_ocean_ml = torch.where(is_ocn, torch.full_like(sfc.t_ocean_ml, C.T_FREEZE_OCEAN), sfc.t_ocean_ml)
        sfc.m_seaice = torch.where(is_ocn, torch.full_like(sfc.m_seaice, 2750.0), sfc.m_seaice)
        sfc.t_seaice = torch.where(is_ocn, torch.full_like(sfc.t_seaice, 245.0), sfc.t_seaice)
        sfc.swe_seaice = torch.where(is_ocn, torch.full_like(sfc.swe_seaice, 80.0), sfc.swe_seaice)
        sfc.swe_land = torch.where(~is_ocn, torch.full_like(sfc.swe_land, 120.0), sfc.swe_land)
        sfc.t_soil.fill_(245.0)
        sfc.t_canopy.fill_(245.0)
    elif key == "mars_like":
        sfc.surf_type.fill_(SURF_LAND)
        sfc.z_topo.zero_()
        sfc.sigma_topo.fill_(150.0)
        sfc.t_soil.fill_(218.0)
        sfc.t_canopy.fill_(218.0)
        sfc.w_liq_soil.fill_(0.005)
        sfc.w_ice_soil.zero_()
        sfc.w_root.fill_(1.0)
        sfc.bio.lai.zero_()
