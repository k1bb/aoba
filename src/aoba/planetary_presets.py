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
- `"pangaea"`: Late Permian / Permo-Triassic Boundary (`250 Ma`) supercontinent using the
  published 6-arc-minute Scotese & Wright (2018) PALEOMAP PaleoDEM (Zenodo DOI
  `10.5281/zenodo.5460860`, auto-downloaded on demand) with `S_0 = 1333 W m^-2` and
  `co2_ppmv = 1137.2`, supplying only topography/bathymetry with zero prescribed biome,
  soil, albedo, or dust maps.

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


SCOTESE_2018_6MIN_ZIP_URL: str = (
    "https://zenodo.org/api/records/5460860/files/"
    "Scotese_Wright_2018_Maps_1-88_6minX6min_PaleoDEMS_nc.zip/content"
)
SCOTESE_2018_250MA_FILENAME: str = "scotese2018_250ma_6min.nc"


def ensure_paleomap_250ma_dem(
    paleomap_dir: str = "data/boundary/paleomap_scotese2018",
    auto_download: bool = True,
) -> Any:
    """Locate or auto-download the Scotese & Wright (2018) 6-arc-min 250 Ma PaleoDEM NetCDF.

    Uses HTTP Range extraction (~1.4 MB) from the official open-access Zenodo archive
    (DOI: 10.5281/zenodo.5460860) and caches the NetCDF under `data/boundary/paleomap_scotese2018/`.
    """
    import os
    from pathlib import Path
    import struct
    import urllib.request
    import zlib

    candidates: list[Path] = [Path(paleomap_dir) / SCOTESE_2018_250MA_FILENAME]
    env_data = os.environ.get("AOBA_DATA_DIR", "").strip()
    if env_data:
        candidates.insert(0, Path(env_data) / "boundary" / "paleomap_scotese2018" / SCOTESE_2018_250MA_FILENAME)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "data" / "boundary" / "paleomap_scotese2018" / SCOTESE_2018_250MA_FILENAME)

    for p in candidates:
        if p.is_file() and p.stat().st_size > 100_000:
            return p

    target = candidates[0]
    if not auto_download:
        raise FileNotFoundError(f"Scotese & Wright (2018) 250 Ma PaleoDEM not found at {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "AOBA-ESM/1.0"}
    req_head = urllib.request.Request(SCOTESE_2018_6MIN_ZIP_URL, method="HEAD", headers=headers)
    with urllib.request.urlopen(req_head, timeout=30) as resp:
        total = int(resp.headers["Content-Length"])
    tail_sz = min(262144, total)
    req_tail = urllib.request.Request(
        SCOTESE_2018_6MIN_ZIP_URL,
        headers={**headers, "Range": f"bytes={total - tail_sz}-{total - 1}"},
    )
    with urllib.request.urlopen(req_tail, timeout=30) as resp:
        tail = resp.read()
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("Failed to locate ZIP central directory in Scotese 2018 Zenodo archive")
    cd_size, cd_off = struct.unpack_from("<II", tail, eocd + 12)
    req_cd = urllib.request.Request(
        SCOTESE_2018_6MIN_ZIP_URL,
        headers={**headers, "Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"},
    )
    with urllib.request.urlopen(req_cd, timeout=30) as resp:
        cd = resp.read()

    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos:pos + 4] != b"PK\x01\x02":
            break
        comp_m = struct.unpack_from("<H", cd, pos + 10)[0]
        comp_sz, _ = struct.unpack_from("<II", cd, pos + 20)
        fn_len, ex_len, cm_len = struct.unpack_from("<HHH", cd, pos + 28)
        lfh_off = struct.unpack_from("<I", cd, pos + 42)[0]
        fn = cd[pos + 46:pos + 46 + fn_len].decode("utf-8", errors="replace")
        pos += 46 + fn_len + ex_len + cm_len
        if "250Ma.nc" in fn and fn.endswith(".nc"):
            req_f = urllib.request.Request(
                SCOTESE_2018_6MIN_ZIP_URL,
                headers={**headers, "Range": f"bytes={lfh_off}-{lfh_off + 30 + fn_len + 512 + comp_sz}"},
            )
            with urllib.request.urlopen(req_f, timeout=90) as resp:
                chunk = resp.read()
            lf_fn, lf_ex = struct.unpack_from("<HH", chunk, 26)
            raw = chunk[30 + lf_fn + lf_ex:30 + lf_fn + lf_ex + comp_sz]
            nc_bytes = zlib.decompress(raw, -15) if comp_m == 8 else raw
            tmp = target.with_suffix(".nc.tmp")
            tmp.write_bytes(nc_bytes)
            tmp.replace(target)
            return target
    raise RuntimeError("250Ma.nc entry not found inside Scotese & Wright (2018) 6-min Zenodo archive")


def build_pangaea_boundary_fields(
    xyz_cell: np.ndarray,
    lat_cell: np.ndarray,
    lon_cell: np.ndarray,
    edges_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    cells_on_edge: np.ndarray,
    stride: int = 2,
) -> dict[str, np.ndarray]:
    """Aggregate the published Scotese & Wright (2018) 6-arc-min 250 Ma PaleoDEM onto SCVT cells.

    Computes cos(lat)-weighted spherical-area averages of surface elevation, bathymetry, ocean
    fraction, and exact Lott & Miller (1997, Appendix A, Eqs. A1-A5) subgrid orographic gradient
    tensor `(sigma_topo, slope_topo, aniso_topo, angle_topo)`, plus 2-pass Klemp (2011) Laplacian
    smoothing and nearest coastal ocean indices. Supplies ONLY elevation and bathymetry; zero
    vegetation, soil, albedo, or dust maps are prescribed.
    """
    from scipy.spatial import cKDTree
    import xarray as xr

    nc_path = ensure_paleomap_250ma_dem(auto_download=True)
    with xr.open_dataset(nc_path) as ds:
        z_var = "z" if "z" in ds.data_vars else list(ds.data_vars)[0]
        z_raw = np.asarray(ds[z_var].values, dtype=np.float64)
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat_deg = np.asarray(ds[lat_name].values, dtype=np.float64)
        lon_deg = np.asarray(ds[lon_name].values, dtype=np.float64)

    z_raw = np.nan_to_num(z_raw, nan=0.0)
    # Drop duplicate periodic endpoint (lon = +180 == -180) if present
    if lon_deg.shape[0] > 2 and abs((lon_deg[-1] - lon_deg[0]) - 360.0) < 1.0e-3:
        lon_deg = lon_deg[:-1]
        z_raw = z_raw[:, :-1]

    # Exact Lott & Miller (1997) finite-difference terrain gradient on the native 6-arc-min grid
    dlat_rad = abs(float(lat_deg[1] - lat_deg[0])) * (math.pi / 180.0)
    dlon_rad = abs(float(lon_deg[1] - lon_deg[0])) * (math.pi / 180.0)
    dy_m = max(C.EARTH_RADIUS * dlat_rad, 100.0)
    cos_lat_full = np.maximum(np.cos(np.radians(lat_deg)), 0.05)
    dx_m_row = np.maximum(C.EARTH_RADIUS * cos_lat_full * dlon_rad, 100.0)[:, None]

    # Central finite differences (periodic in longitude, clamped at poles)
    dz_dx = (np.roll(z_raw, -1, axis=1) - np.roll(z_raw, 1, axis=1)) / (2.0 * dx_m_row)
    dz_dy = np.zeros_like(z_raw)
    sign_lat = 1.0 if lat_deg[1] > lat_deg[0] else -1.0
    dz_dy[1:-1, :] = sign_lat * (z_raw[2:, :] - z_raw[:-2, :]) / (2.0 * dy_m)
    dz_dy[0, :] = sign_lat * (z_raw[1, :] - z_raw[0, :]) / dy_m
    dz_dy[-1, :] = sign_lat * (z_raw[-1, :] - z_raw[-2, :]) / dy_m

    # Subsample stride for fast KD-tree aggregation while preserving full native gradients
    st = max(int(stride), 1)
    lat_s = lat_deg[::st]
    lon_s = lon_deg[::st]
    z_s = z_raw[::st, ::st]
    hx_s = dz_dx[::st, ::st]
    hy_s = dz_dy[::st, ::st]

    n_cells = int(xyz_cell.shape[0])
    tree = cKDTree(xyz_cell)
    w_tot = np.zeros(n_cells, dtype=np.float64)
    sum_ocn = np.zeros(n_cells, dtype=np.float64)
    sum_h = np.zeros(n_cells, dtype=np.float64)
    sum_h2 = np.zeros(n_cells, dtype=np.float64)
    sum_bed = np.zeros(n_cells, dtype=np.float64)
    sum_hx2 = np.zeros(n_cells, dtype=np.float64)
    sum_hy2 = np.zeros(n_cells, dtype=np.float64)
    sum_hxy = np.zeros(n_cells, dtype=np.float64)
    z_min_land = np.full(n_cells, 1.0e9, dtype=np.float64)

    lon_rad_s = np.radians(lon_s)
    cos_lon = np.cos(lon_rad_s)
    sin_lon = np.sin(lon_rad_s)
    n_lat_s = lat_s.shape[0]
    row_chunk = 64

    for r0 in range(0, n_lat_s, row_chunk):
        r1 = min(r0 + row_chunk, n_lat_s)
        lat_r = np.radians(lat_s[r0:r1])
        clat = np.cos(lat_r)[:, None]
        slat = np.sin(lat_r)[:, None]
        px = (clat * cos_lon[None, :]).ravel()
        py = (clat * sin_lon[None, :]).ravel()
        pz = np.broadcast_to(slat, (r1 - r0, lon_s.shape[0])).ravel()
        pts = np.column_stack((px, py, pz))
        _, cid = tree.query(pts, workers=-1)

        w = np.broadcast_to(np.maximum(clat, 1.0e-4), (r1 - r0, lon_s.shape[0])).ravel()
        zv = z_s[r0:r1].ravel()
        is_ocn_p = (zv <= 0.0).astype(np.float64)
        z_land_p = np.maximum(zv, 0.0)
        hx = np.where(zv > 0.0, hx_s[r0:r1].ravel(), 0.0)
        hy = np.where(zv > 0.0, hy_s[r0:r1].ravel(), 0.0)

        np.add.at(w_tot, cid, w)
        np.add.at(sum_ocn, cid, w * is_ocn_p)
        np.add.at(sum_h, cid, w * z_land_p)
        np.add.at(sum_h2, cid, w * z_land_p * z_land_p)
        np.add.at(sum_bed, cid, w * zv)
        np.add.at(sum_hx2, cid, w * hx * hx)
        np.add.at(sum_hy2, cid, w * hy * hy)
        np.add.at(sum_hxy, cid, w * hx * hy)

        land_mask_p = zv > 0.0
        if np.any(land_mask_p):
            np.minimum.at(z_min_land, cid[land_mask_p], z_land_p[land_mask_p])

    w_safe = np.maximum(w_tot, 1.0e-12)
    ocean_frac = np.clip(sum_ocn / w_safe, 0.0, 1.0)
    icesheet_frac = np.zeros(n_cells, dtype=np.float64)
    surf_type = np.where(ocean_frac >= 0.5, 0, 1).astype(np.int64)

    z_topo = np.where(surf_type == 0, 0.0, np.maximum(sum_h / w_safe, 1.0))
    z_bed_raw = sum_bed / w_safe
    z_bed = np.where(surf_type == 0, np.minimum(z_bed_raw, -100.0), z_topo)
    z_min_land = np.where((surf_type == 0) | (z_min_land > 1.0e8), z_topo, z_min_land)

    var_h = np.maximum(sum_h2 / w_safe - (sum_h / w_safe) ** 2, 0.0)
    sigma_topo = np.where(surf_type == 0, 0.0, np.sqrt(var_h))

    hx2 = sum_hx2 / w_safe
    hy2 = sum_hy2 / w_safe
    hxy = sum_hxy / w_safe
    k_lm = 0.5 * (hx2 + hy2)
    m_lm = 0.5 * (hx2 - hy2)
    amp_lm = np.sqrt(m_lm * m_lm + hxy * hxy)
    angle_topo = 0.5 * np.arctan2(hxy, m_lm)
    lam1 = np.maximum(k_lm + amp_lm, 1.0e-12)
    lam2 = np.maximum(k_lm - amp_lm, 0.0)
    aniso_topo = np.where(surf_type == 0, 1.0, np.sqrt(lam2 / lam1))
    slope_topo = np.where(surf_type == 0, 0.0, np.sqrt(hx2 + hy2))

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
        "z_min_land": z_min_land,
        "lake_basin_frac_bed": np.zeros(n_cells, dtype=np.float64),
        "lake_basin_vol_bed": np.zeros(n_cells, dtype=np.float64),
    }
