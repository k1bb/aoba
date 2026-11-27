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
- `"pangaea"`: Late Permian / Early Triassic (`~250 Ma`) C-shaped supercontinent enclosing the
  equatorial Tethys Ocean inside the global Panthalassa super-ocean (`S_0 = 1333 W m^-2`,
  `co2_ppmv = 1137.2`, Central Pangaean Mountains, zero prescribed biome/soil/albedo maps;
  Kutzbach & Gallimore, 1989; Parrish, 1993; Scotese, 2001).

AOBA Formulation Reference: Sec. 11.3, Eq. (11.5) (General Planetary / Exoplanet Presets & Tidal Lock).
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any
import numpy as np
import torch

from aoba import constants as C

VALID_PLANET_PRESETS: tuple[str, ...] = (
    "earth",
    "aquaplanet",
    "snowball_earth",
    "tidally_locked",
    "trappist1e",
    "mars_like",
    "pangaea",
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
    "pangaea": {
        "use_earth2014": False,
        "solar_constant": 1333.0,
        "co2_ppmv": 1137.2,
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
    elif key == "pangaea":
        # Late Permian / Early Triassic warm ice-free initial ocean + spatially uniform pioneer land
        # (100% map-free: deserts, megamonsoon forests, & polar woodlands emerge autonomously).
        is_ocn = sfc.surf_type == SURF_OCEAN
        sfc.m_seaice.zero_()
        sfc.swe_seaice.zero_()
        sfc.swe_land.zero_()
        sfc.t_ocean_ml = torch.where(is_ocn, torch.clamp(sfc.t_ocean_ml + 3.0, min=274.5, max=304.0), sfc.t_ocean_ml)
        sfc.bio.lai = torch.where(~is_ocn, torch.full_like(sfc.bio.lai, 0.35), torch.zeros_like(sfc.bio.lai))
        sfc.bio.h_canopy = torch.where(~is_ocn, torch.full_like(sfc.bio.h_canopy, 2.0), torch.zeros_like(sfc.bio.h_canopy))
        sfc.bio.f_evg = torch.where(~is_ocn, torch.full_like(sfc.bio.f_evg, 0.5), torch.zeros_like(sfc.bio.f_evg))
        sfc.bio.f_c4 = torch.where(~is_ocn, torch.full_like(sfc.bio.f_c4, 0.2), torch.zeros_like(sfc.bio.f_c4))


def build_pangaea_boundary_fields(
    xyz_cell: np.ndarray,
    lat_cell: np.ndarray,
    lon_cell: np.ndarray,
    edges_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    cells_on_edge: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct self-contained 250 Ma Late Permian / Early Triassic Pangaea elevation & bathymetry.

    Generates a classic C-shaped pole-to-pole supercontinent (~32% land fraction) comprising Laurasia,
    Gondwana, the equatorial Tethys Ocean embayment, Panthalassa super-ocean bathymetry, Central
    Pangaean Mountains (Variscan/Appalachian suture), Gondwanide cordillera, and Uralian highlands
    (Kutzbach & Gallimore, 1989; Parrish, 1993; Scotese, 2001). Supplies ONLY topography/bathymetry;
    zero vegetation, soil, albedo, or dust maps are prescribed.
    """
    from scipy.spatial import cKDTree

    n_cells = int(lat_cell.shape[0])
    lat_d = np.degrees(lat_cell).astype(np.float64)
    lon_d = np.degrees(lon_cell).astype(np.float64)  # [-180, +180]

    # Smooth analytic coastline potential phi > 0 => land, phi <= 0 => ocean
    # 1. Western supercontinental spine (Americas + West Africa + Gondwana backbone)
    spine = np.exp(-((lon_d + 22.0) / 36.0) ** 2 - ((lat_d + 6.0) / 68.0) ** 2)
    # 2. Laurasian northern arm (Euramerica + Siberia + Cathaysia north of Tethys)
    laurasia = 1.05 * np.exp(-((lon_d - 28.0) / 56.0) ** 2 - ((lat_d - 44.0) / 21.0) ** 2)
    # 3. Gondwanan southern arm (South America + Africa + India + Australia + Antarctica south of Tethys)
    gondwana = 1.15 * np.exp(-((lon_d - 24.0) / 60.0) ** 2 - ((lat_d + 52.0) / 26.0) ** 2)
    # 4. South Polar Gondwana cap
    south_cap = 0.85 * np.exp(-((lat_d + 80.0) / 16.0) ** 2) * np.exp(-((lon_d - 10.0) / 95.0) ** 2)
    # 5. Deep warm equatorial Tethys Ocean wedge opening eastward from lon ~ +5 deg
    tethys_cut = 1.35 * np.exp(-((lat_d - 2.0) / 21.0) ** 2) * (1.0 / (1.0 + np.exp(-(lon_d - 12.0) / 10.0)))
    # 6. Smooth multi-scale coastline modulation (organic coastal bays/peninsulas)
    coast_mod = (
        0.045 * np.sin(np.radians(4.0 * lon_d + 2.0 * lat_d))
        + 0.030 * np.cos(np.radians(6.0 * lat_d - 3.0 * lon_d + 25.0))
        + 0.020 * np.sin(np.radians(9.0 * lon_d - 5.0 * lat_d))
    )

    phi = np.maximum.reduce([spine, laurasia, gondwana, south_cap]) - tethys_cut + coast_mod - 0.26
    is_land = phi > 0.0
    surf_type = np.where(is_land, 1, 0).astype(np.int64)
    ocean_frac = np.where(is_land, 0.0, 1.0).astype(np.float64)
    icesheet_frac = np.zeros(n_cells, dtype=np.float64)

    # Smooth continental relief + major Permian orogenic belts (zero elevation over ocean)
    land_depth = np.clip(phi / 0.45, 0.0, 1.0)
    z_plain = 480.0 * np.sqrt(land_depth)
    # Central Pangaean Mountains (Variscan-Appalachian suture across equatorial/subtropical backbone)
    lat_cpm = 6.0 + 0.22 * (lon_d + 22.0)
    z_cpm = 2850.0 * np.exp(-((lat_d - lat_cpm) / 7.5) ** 2 - ((lon_d + 16.0) / 32.0) ** 2)
    # Gondwanide southwestern cordillera
    z_gond = 2150.0 * np.exp(-((lon_d + 44.0) / 11.0) ** 2 - ((lat_d + 42.0) / 22.0) ** 2)
    # Uralian / Angara northern highlands
    z_ural = 1750.0 * np.exp(-((lon_d - 50.0) / 12.0) ** 2 - ((lat_d - 50.0) / 15.0) ** 2)
    # Interior plateau relief undulation
    z_undul = 220.0 * land_depth * (0.5 + 0.5 * np.sin(np.radians(7.0 * lon_d)) * np.cos(np.radians(7.0 * lat_d)))

    z_topo = np.where(is_land, np.clip(z_plain + z_cpm + z_gond + z_ural + z_undul, 20.0, 4200.0), 0.0)

    # Ocean bathymetry: continental shelf (-200 m) -> abyssal Panthalassa (-4300 m) + mid-Panthalassa ridge
    ocn_dist = np.clip(-phi / 0.35, 0.0, 1.0)
    z_abyss = -200.0 - 4100.0 * np.tanh(1.8 * ocn_dist)
    dlon_ridge = ((lon_d + 145.0 + 180.0) % 360.0) - 180.0
    z_ridge = 1500.0 * np.exp(-(dlon_ridge / 14.0) ** 2) * (ocn_dist > 0.4)
    z_bed = np.where(is_land, z_topo, np.clip(z_abyss + z_ridge, -4800.0, -120.0))

    # Subgrid orography statistics consistent with relief for Lott & Miller (1997) drag
    z_orog = z_cpm + z_gond + z_ural
    sigma_topo = np.where(is_land, np.clip(45.0 + 0.16 * z_orog + 0.04 * z_topo, 20.0, 650.0), 0.0)
    slope_topo = np.where(is_land, np.clip(0.002 + 2.2e-5 * z_orog + 4.0e-6 * z_topo, 0.001, 0.08), 0.0)
    aniso_topo = np.where(is_land, np.where(z_orog > 400.0, 0.35, 0.75), 1.0)
    angle_topo = np.where(is_land, np.where(z_cpm >= np.maximum(z_gond, z_ural), 0.22, 1.35), 0.0)

    # Klemp (2011) 2-pass conservative Laplacian terrain smoothing
    z_smooth = z_topo.copy()
    for _ in range(2):
        z_next = z_smooth.copy()
        for i in range(n_cells):
            deg = int(n_edges_on_cell[i])
            nb_sum = 0.0
            for k in range(deg):
                e_id = int(edges_on_cell[i, k])
                c0, c1 = int(cells_on_edge[e_id, 0]), int(cells_on_edge[e_id, 1])
                nb = c1 if c0 == i else c0
                nb_sum += z_smooth[nb]
            z_next[i] = 0.75 * z_smooth[i] + 0.25 * (nb_sum / max(deg, 1))
        z_smooth = np.where(surf_type == 0, 0.0, z_next)

    ocn_indices = np.where(surf_type == 0)[0]
    if len(ocn_indices) > 0:
        ocn_tree = cKDTree(xyz_cell[ocn_indices])
        _, nearest_loc = ocn_tree.query(xyz_cell)
        coastal_ocean_idx = ocn_indices[nearest_loc].astype(np.int64)
    else:
        coastal_ocean_idx = np.arange(n_cells, dtype=np.int64)

    return {
        "surf_type": surf_type,
        "ocean_frac": ocean_frac,
        "icesheet_frac": icesheet_frac,
        "z_topo": z_topo,
        "z_topo_smooth": z_smooth,
        "sigma_topo": sigma_topo,
        "slope_topo": slope_topo,
        "aniso_topo": aniso_topo,
        "angle_topo": angle_topo,
        "coastal_ocean_idx": coastal_ocean_idx,
        "z_bed": z_bed,
        "h_ice_init": np.zeros(n_cells, dtype=np.float64),
        "z_min_land": np.where(is_land, np.maximum(z_topo - 0.5 * sigma_topo, 0.0), 0.0),
        "lake_basin_frac_bed": np.zeros(n_cells, dtype=np.float64),
        "lake_basin_vol_bed": np.zeros(n_cells, dtype=np.float64),
    }
