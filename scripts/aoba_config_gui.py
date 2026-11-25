#!/usr/bin/env python3
"""End-to-End Web GUI Workbench for AOBA: Configure, Launch, Monitor & Live 3-Globe CPU Preview.

Zero-dependency stdlib HTTP server + HTML5 Canvas workbench offering:
1. Interactive SCVT `glevel` globe + CRM subgrid scale + vertical levels visual configurator
2. Planetary/orbital/dynamic-SIA ice-sheet presets & acceleration curve preview
3. Output streams, time-mean vs instantaneous Zarr cadence, and storage estimator
4. Searchable 188-parameter TOML registry editor & live TOML export
5. Model Run Launcher + Live Process Monitor (CSV trajectories & `run.log` tail)
6. Idle-CPU background SCVT->LatLon regridder driving 3 slowly rotating Orthographic Earth globes
   for `T2` (surface air temperature), `OLR` (outgoing longwave radiation), and `LAI` (leaf area index),
   compatible with both new `--config` runs (`live_preview.npz`) and existing Zarr runs (`PID ...`).
"""

from __future__ import annotations

import argparse
import base64
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aoba.calendar import AobaCalendar  # noqa: E402
from aoba.co2_forcing import fetch_latest_co2_sources, get_co2_catalog_summary  # noqa: E402
from aoba.ozone_forcing import fetch_input4mips_ozone_climatology, get_ozone_status_summary  # noqa: E402
from aoba.param_docs import PARAM_DOCS  # noqa: E402
from aoba.params import (  # noqa: E402
    SECTION_META,
    build_configs_from_dict,
    default_config_dict,
    default_toml_path,
    dump_aoba_toml,
    load_aoba_toml,
    merge_config_dicts,
    save_aoba_toml,
)
from aoba.runtime import compute_optimal_chunks, detect_hardware_profile, get_output_dir  # noqa: E402


def _coerce_like_default(val: Any, default_val: Any) -> Any:
    """Coerce JSON payload values back into the exact Python type of `default_val`."""
    if isinstance(default_val, bool):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val)
    if isinstance(default_val, int):
        return int(round(float(val)))
    if isinstance(default_val, float):
        return float(val)
    return str(val)


def sanitize_config_dict(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge arbitrary partial dict onto baseline defaults and coerce every known key's type."""
    defaults = default_config_dict()
    merged = merge_config_dicts(defaults, raw)
    for sec, sec_defaults in defaults.items():
        for k, def_v in sec_defaults.items():
            try:
                merged[sec][k] = _coerce_like_default(merged[sec][k], def_v)
            except (TypeError, ValueError):
                merged[sec][k] = def_v
    return merged


def count_modified_keys(cfg: dict[str, dict[str, Any]]) -> int:
    """Count how many keys differ from factory defaults."""
    defaults = default_config_dict()
    n = 0
    for sec, sec_defaults in defaults.items():
        cur_sec = cfg.get(sec, {})
        for k, def_v in sec_defaults.items():
            cur_v = cur_sec.get(k, def_v)
            if isinstance(def_v, float):
                if abs(float(cur_v) - float(def_v)) > 1e-12 * max(1.0, abs(float(def_v))):
                    n += 1
            elif cur_v != def_v:
                n += 1
    return n


# =============================================================================
# CPU-Only SCVT -> Equirectangular Regridder & Live Run Monitor Engine
# =============================================================================
class CPUGlobePreviewEngine:
    """Runs strictly on CPU (`numpy`/`scipy`) to convert SCVT simulation outputs into live raster maps."""

    GRID_H = 360
    GRID_W = 720

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mesh_cache: dict[tuple[int, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._frame_cache: dict[tuple[str, str, int, float], dict[str, Any]] = {}
        d_lat = 180.0 / self.GRID_H
        d_lon = 360.0 / self.GRID_W
        lat_centers = np.linspace(90.0 - 0.5 * d_lat, -90.0 + 0.5 * d_lat, self.GRID_H, dtype=np.float64)
        lon_centers = np.linspace(-180.0 + 0.5 * d_lon, 180.0 - 0.5 * d_lon, self.GRID_W, dtype=np.float64)
        lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
        lat_r = np.deg2rad(lat2d.ravel())
        lon_r = np.deg2rad(lon2d.ravel())
        self._target_xyz = np.stack(
            [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)],
            axis=-1,
        )

    def _get_interp_weights(
        self, lat_deg: np.ndarray, lon_deg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build or fetch cached 3-neighbor spherical IDW weights + nearest index on CPU."""
        sig = (int(lat_deg.size), round(float(lat_deg[0]), 4), round(float(lon_deg[-1]), 4))
        with self._lock:
            cached = self._mesh_cache.get(sig)
        if cached is not None:
            return cached

        lat_r = np.deg2rad(lat_deg.astype(np.float64))
        lon_r = np.deg2rad(lon_deg.astype(np.float64))
        src_xyz = np.stack(
            [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)],
            axis=-1,
        )

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(src_xyz)
            dists, idxs = tree.query(self._target_xyz, k=3, workers=2)
        except Exception:
            # Pure-NumPy fallback via chordal cosine dot product
            dots = self._target_xyz @ src_xyz.T
            idxs = np.argpartition(-dots, kth=2, axis=1)[:, :3]
            row_idx = np.arange(dots.shape[0])[:, None]
            d_cos = np.clip(dots[row_idx, idxs], -1.0, 1.0)
            dists = np.sqrt(np.maximum(2.0 * (1.0 - d_cos), 0.0))

        nn1 = idxs[:, 0].astype(np.int32)
        w = 1.0 / (dists * dists + 1.0e-5)
        w /= w.sum(axis=1, keepdims=True)
        res = (idxs.astype(np.int32), w.astype(np.float32), nn1)
        with self._lock:
            self._mesh_cache[sig] = res
        return res

    @staticmethod
    def _quantize_u8(arr: np.ndarray, vmin: float, vmax: float) -> str:
        norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        u8 = np.rint(norm * 255.0).astype(np.uint8)
        return base64.b64encode(u8.tobytes()).decode("ascii")

    def regrid_fields(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        t2_c: np.ndarray,
        precip: np.ndarray,
        lai: np.ndarray,
        surf_type: np.ndarray,
        esm_range: tuple[float, float] = (0.0, 7.0),
        esm_land_only: bool = True,
        *,
        olr: np.ndarray | None = None,
        sst_c: np.ndarray | None = None,
        seaice_frac: np.ndarray | None = None,
        ocn_speed_ms: np.ndarray | None = None,
        ocn_lat_deg: np.ndarray | None = None,
        ocn_lon_deg: np.ndarray | None = None,
        ocn_wet: np.ndarray | None = None,
        ocn_is_wind: bool = False,
    ) -> dict[str, Any]:
        """Regrid irregular SCVT cell arrays onto `[180, 360]` on CPU and encode compact Uint8 rasters for 6 globes."""
        t0 = time.perf_counter()
        idxs, w, nn1 = self._get_interp_weights(lat_deg, lon_deg)

        t2_grid = (t2_c[idxs] * w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
        precip_grid = (precip[idxs] * w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
        lai_nn = np.where(surf_type == 1, lai, 0.0) if esm_land_only else lai
        lai_grid = (lai_nn[idxs] * w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
        surf_grid = surf_type[nn1].reshape(self.GRID_H, self.GRID_W).astype(np.uint8)
        olr_arr = olr if olr is not None else np.full_like(t2_c, 240.0)
        olr_grid = (olr_arr[idxs] * w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)

        # Coastline edge detection directly on CPU raster (`surf_type == 0` vs terrestrial `>= 1`)
        is_land = surf_grid >= 1
        coast = np.zeros_like(surf_grid)
        diff_x = is_land != np.roll(is_land, 1, axis=1)
        diff_y = np.zeros_like(is_land)
        diff_y[1:, :] = is_land[1:, :] != is_land[:-1, :]
        coast[(diff_x | diff_y)] = 1
        if esm_land_only:
            lai_grid = np.where(surf_grid == 1, lai_grid, 0.0)

        # Ocean globes (SST + Sea Ice, Surface Current Speed): use native fine OGCM mesh (e.g. G7 28 km) when present!
        if (
            ocn_lat_deg is not None
            and ocn_lon_deg is not None
            and sst_c is not None
            and sst_c.shape[0] == ocn_lat_deg.shape[0]
        ):
            o_idxs, o_w, o_nn1 = self._get_interp_weights(ocn_lat_deg, ocn_lon_deg)
            sst_grid = (sst_c[o_idxs] * o_w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
            ice_arr = seaice_frac if (seaice_frac is not None and seaice_frac.shape[0] == ocn_lat_deg.shape[0]) else np.zeros_like(sst_c)
            ice_grid = (ice_arr[o_idxs] * o_w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
            spd_arr = ocn_speed_ms if (ocn_speed_ms is not None and ocn_speed_ms.shape[0] == ocn_lat_deg.shape[0]) else np.zeros_like(sst_c)
            spd_grid = (spd_arr[o_idxs] * o_w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
            if ocn_wet is not None and ocn_wet.shape[0] == ocn_lat_deg.shape[0]:
                owet_grid = (ocn_wet[o_nn1] > 0).reshape(self.GRID_H, self.GRID_W).astype(np.uint8)
            else:
                owet_grid = (surf_grid == 0).astype(np.uint8)
            ocn_res_lbl = f"Native Ocean ({ocn_lat_deg.shape[0]:,} cells)"
        else:
            sst_arr = sst_c if sst_c is not None else t2_c
            sst_grid = (sst_arr[idxs] * w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
            ice_arr = seaice_frac if seaice_frac is not None else np.zeros_like(t2_c)
            ice_grid = (ice_arr[idxs] * w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
            spd_arr = ocn_speed_ms if ocn_speed_ms is not None else np.zeros_like(t2_c)
            spd_grid = (spd_arr[idxs] * w).sum(axis=1).reshape(self.GRID_H, self.GRID_W)
            owet_grid = (surf_grid == 0).astype(np.uint8)
            ocn_res_lbl = f"Host Grid ({lat_deg.shape[0]:,} cells)"

        o_land = owet_grid == 0
        ocn_coast = np.zeros_like(owet_grid)
        odx = o_land != np.roll(o_land, 1, axis=1)
        ody = np.zeros_like(o_land)
        ody[1:, :] = o_land[1:, :] != o_land[:-1, :]
        ocn_coast[(odx | ody)] = 1

        spd_max = 25.0 if ocn_is_wind else 1.2
        cpu_ms = (time.perf_counter() - t0) * 1000.0
        lo3, hi3 = float(esm_range[0]), float(esm_range[1])
        return {
            "width": self.GRID_W,
            "height": self.GRID_H,
            "t2_range": [-45.0, 40.0],
            "ir_range": [110.0, 320.0],
            "precip_range": [0.0, 25.0],
            "sst_range": [-2.0, 32.0],
            "ocn_spd_range": [0.0, spd_max],
            "ocn_is_wind": bool(ocn_is_wind),
            "ocn_res_lbl": ocn_res_lbl,
            "lai_range": [lo3, hi3],
            "esm_land_only": bool(esm_land_only),
            "t2_b64": self._quantize_u8(t2_grid, -45.0, 40.0),
            "ir_b64": self._quantize_u8(olr_grid, 110.0, 320.0),
            "precip_b64": self._quantize_u8(precip_grid, 0.0, 25.0),
            "sst_b64": self._quantize_u8(sst_grid, -2.0, 32.0),
            "seaice_b64": self._quantize_u8(ice_grid, 0.0, 1.0),
            "ocn_spd_b64": self._quantize_u8(spd_grid, 0.0, spd_max),
            "lai_b64": self._quantize_u8(lai_grid, lo3, hi3),
            "surf_b64": base64.b64encode(surf_grid.tobytes()).decode("ascii"),
            "coast_b64": base64.b64encode(coast.tobytes()).decode("ascii"),
            "ocn_wet_b64": base64.b64encode(owet_grid.tobytes()).decode("ascii"),
            "ocn_coast_b64": base64.b64encode(ocn_coast.tobytes()).decode("ascii"),
            "cpu_regrid_ms": round(cpu_ms, 2),
        }


_ESM_GLOBE_SPECS: dict[str, tuple[str, float, float, bool]] = {
    "lai": ("LAI [m² m⁻²]", 0.0, 7.0, True),
    "o3_tco_du": ("Total Ozone O₃ [DU]", 150.0, 420.0, False),
    "lightning_flash_rate": ("Lightning [fl km⁻² yr⁻¹]", 0.0, 30.0, False),
    "volc_aod": ("Volcanic AOD (550nm)", 0.0, 0.4, False),
    "f_wetland": ("Wetland/Lake Frac [-]", 0.0, 0.85, True),
    "d18o_precip_permil": ("Precip δ¹⁸O [‰]", -45.0, 2.0, False),
    "co2_sfc_ppmv": ("Surface CO₂ [ppmv]", 260.0, 460.0, False),
    "ocn_chl_sfc": ("Ocean Chl [mg m⁻³]", 0.0, 2.0, False),
    "ocn_o2_sfc": ("Ocean O₂ [mmol m⁻³]", 140.0, 360.0, False),
    "wave_hs_m": ("Wave Height H_s [m]", 0.0, 8.0, False),
}


def _compute_obs_benchmark(csv_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare active run's recent global climate means against Earth observational reference ranges."""
    refs = [
        ("ts_k", "Surface T_s [K]", "ERA5 (1981–2010)", 286.5, 289.2, 287.8),
        ("olr_wm2", "TOA OLR [W m⁻²]", "CERES EBAF", 234.0, 245.0, 239.7),
        ("asr_wm2", "TOA ASR [W m⁻²]", "CERES EBAF", 235.0, 246.0, 240.5),
        ("precip_mm_d", "Precip [mm d⁻¹]", "GPCP v2.3", 2.45, 3.05, 2.72),
        ("lai", "Land LAI [m² m⁻²]", "MODIS/GLASS", 1.20, 2.20, 1.65),
        ("seaice_mkm2", "Sea-Ice [Mkm²]", "NSIDC CDR", 17.0, 27.0, 22.0),
    ]
    rows: list[dict[str, Any]] = []
    for key, label, src, lo, hi, obs_mid in refs:
        vals = csv_data.get(key) or []
        if not vals:
            continue
        tail = [float(x) for x in vals[-min(len(vals), 365):]]
        mean_v = sum(tail) / max(len(tail), 1)
        in_band = (lo <= mean_v <= hi)
        rows.append({
            "key": key,
            "label": label,
            "obs_source": src,
            "obs_mid": obs_mid,
            "obs_range": f"{lo:g}–{hi:g}",
            "model_mean": round(mean_v, 2),
            "bias": round(mean_v - obs_mid, 2),
            "status": "ok" if in_band else "warn",
        })
    return rows


def _find_zarr_frames(out_root: Path, kind: str) -> list[dict[str, Any]]:
    """List available `(store_path, time_idx, label, sim_days)` across `single` or `yearly` Zarr stores."""
    import xarray as xr

    frames: list[dict[str, Any]] = []
    stores: list[tuple[str, Path]] = []
    single_store = out_root / f"{kind}.zarr"
    if single_store.exists():
        stores.append(("single", single_store))
    for ydir in sorted(out_root.glob("year_*")):
        p = ydir / f"{kind}.zarr"
        if p.exists():
            stores.append((ydir.name, p))

    for tag, store in stores:
        try:
            ds = xr.open_zarr(store, consolidated=False, decode_times=False)
            t_vals = ds["time"].values if "time" in ds else np.arange(ds.sizes.get("time", 0), dtype=np.float64)
            for idx, tv in enumerate(t_vals):
                frames.append({
                    "store": str(store),
                    "local_idx": int(idx),
                    "global_idx": len(frames),
                    "sim_days": float(tv),
                    "label": f"{tag} #{idx + 1} ({float(tv):.1f}d)",
                })
            ds.close()
        except Exception:
            continue
    return frames


def _load_zarr_frame_payload(
    out_root: Path,
    frame_info: dict[str, Any],
    kind: str,
    esm_field: str = "lai",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list[str]]:
    """Extract `(lat, lon, t2_c, precip, field3, surf_type, source_label, avail_esm_fields)` from a Zarr slice."""
    import xarray as xr

    store = Path(frame_info["store"])
    idx = int(frame_info["local_idx"])
    ds = xr.open_zarr(store, consolidated=False, decode_times=False)
    try:
        lat_deg = np.asarray(ds["lat_cell"].values, dtype=np.float32)
        lon_deg = np.asarray(ds["lon_cell"].values, dtype=np.float32)
        avail_esm = [k for k in _ESM_GLOBE_SPECS if k in ds and ds[k].shape[-1] == lat_deg.shape[0]]
        if "lai" not in avail_esm:
            avail_esm.insert(0, "lai")

        surf_type = None
        if "surf_type" in ds:
            surf_type = np.asarray(ds["surf_type"].values, dtype=np.int8)
        elif (out_root / "mesh.zarr").exists():
            m = xr.open_zarr(out_root / "mesh.zarr", consolidated=False)
            surf_type = np.asarray(m["surf_type"].values, dtype=np.int8)
            m.close()
        if surf_type is None:
            surf_type = np.ones_like(lat_deg, dtype=np.int8)

        t_ocn = np.asarray(ds["t_ocean_ml"].isel(time=idx).values, dtype=np.float32) if "t_ocean_ml" in ds else np.full_like(lat_deg, 288.0)
        t_can = np.asarray(ds["t_canopy"].isel(time=idx).values, dtype=np.float32) if "t_canopy" in ds else t_ocn
        t_sfc = np.where(surf_type == 0, t_ocn, t_can)

        if "temp" in ds:
            temp0 = np.asarray(ds["temp"].isel(time=idx, nz=0).values, dtype=np.float32)
            t2_k = 0.35 * t_sfc + 0.65 * temp0
        else:
            t2_k = t_sfc
        t2_c = t2_k - 273.15

        if "precip_mm_day" in ds:
            precip = np.asarray(ds["precip_mm_day"].isel(time=idx).values, dtype=np.float32)
        elif "precip_mm_day_sub" in ds:
            arr = np.asarray(ds["precip_mm_day_sub"].isel(time=idx).values, dtype=np.float32)
            precip = arr.mean(axis=tuple(range(1, arr.ndim)))
        else:
            precip = np.zeros_like(lat_deg, dtype=np.float32)

        f3_key = esm_field if (esm_field in ds and ds[esm_field].shape[-1] == lat_deg.shape[0]) else "lai"
        if f3_key != "lai":
            f3 = np.asarray(ds[f3_key].isel(time=idx).values, dtype=np.float32)
            if f3_key == "lightning_flash_rate":
                # Convert fl m^-2 s^-1 -> fl km^-2 yr^-1
                f3 = f3 * (1.0e6 * 365.2425 * 86400.0)
        elif "lai" in ds:
            f3 = np.asarray(ds["lai"].isel(time=idx).values, dtype=np.float32)
        elif "lai_sub" in ds:
            arr = np.asarray(ds["lai_sub"].isel(time=idx).values, dtype=np.float32)
            f3 = arr.mean(axis=tuple(range(1, arr.ndim)))
        else:
            f3 = np.zeros_like(lat_deg, dtype=np.float32)

        olr_arr = np.asarray(ds["olr"].isel(time=idx).values, dtype=np.float32) if "olr" in ds else np.full_like(lat_deg, 240.0)
        sst_c = t_ocn - 273.15
        if "m_seaice" in ds:
            seaice = np.clip(np.sqrt(np.maximum(np.asarray(ds["m_seaice"].isel(time=idx).values, dtype=np.float32), 0.0) / 500.0), 0.0, 1.0)
        elif "seaice_frac" in ds:
            seaice = np.clip(np.asarray(ds["seaice_frac"].isel(time=idx).values, dtype=np.float32), 0.0, 1.0)
        else:
            seaice = np.zeros_like(lat_deg, dtype=np.float32)

        return lat_deg, lon_deg, t2_c, precip, f3, surf_type, f"{kind} ({frame_info['label']})", avail_esm, olr_arr, sst_c, seaice
    finally:
        ds.close()


def _read_csv_series(out_root: Path, max_points: int = 800) -> dict[str, list[float]]:
    """Aggregate all `year_XXXX/daily_global.csv` series for real-time plotting."""
    cols = ("sim_days", "ts_k", "olr_wm2", "asr_wm2", "net_toa_wm2", "precip_mm_d", "lai", "seaice_mkm2", "wall_s_per_day")
    rows: list[list[float]] = []
    for csv_p in sorted(out_root.glob("year_*/daily_global.csv")):
        try:
            with open(csv_p, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rows.append([float(row[c]) for c in cols])
                    except (KeyError, ValueError, TypeError):
                        continue
        except OSError:
            continue

    if len(rows) > max_points:
        stride = max(len(rows) // max_points, 1)
        rows = rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride != 0 else [])

    out = {c: [] for c in cols}
    for r in rows:
        for i, c in enumerate(cols):
            out[c].append(round(r[i], 3))
    return out


def _tail_log_lines(log_path: Path, n_lines: int = 75) -> str:
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except OSError:
        return ""


def _discover_active_processes() -> list[dict[str, Any]]:
    """Inspect active OS `run_g5_30yr.py` processes cleanly without disturbing them."""
    procs: list[dict[str, Any]] = []
    try:
        res = subprocess.run(
            ["ps", "-axo", "pid,%cpu,etime,command"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        for line in res.stdout.splitlines():
            if "run_g5_30yr.py" not in line or "grep" in line:
                continue
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            pid_s, cpu_s, etime_s, cmd = parts
            m_out = re.search(r"--out-dir\s+([^\s]+)", cmd)
            out_dir = str(Path(m_out.group(1)).expanduser()) if m_out else ""
            if not out_dir:
                m_cfg = re.search(r"--config\s+([^\s]+)", cmd)
                if m_cfg:
                    cfg_p = Path(m_cfg.group(1)).expanduser()
                    if cfg_p.exists():
                        try:
                            cfg_loaded = load_aoba_toml(cfg_p)
                            raw_od = str(cfg_loaded.get("output", {}).get("out_dir", "")).strip()
                            if raw_od:
                                out_dir = str(Path(raw_od).expanduser())
                        except Exception:
                            pass
                        if not out_dir:
                            out_dir = str(cfg_p.parent)
            procs.append({
                "pid": int(pid_s),
                "cpu_pct": cpu_s,
                "etime": etime_s,
                "out_dir": out_dir,
                "command": cmd,
            })
    except Exception:
        pass
    return procs


def _discover_output_dirs(extra_dir: str = "") -> list[str]:
    """List candidate AOBA output directories containing mesh/Zarr/CSV/live preview files."""
    candidates: list[Path] = []
    base = get_output_dir(None)
    if base.exists():
        for child in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if child.is_dir():
                if (child / "mesh.zarr").exists() or (child / "live_preview.npz").exists() or list(child.glob("year_*")):
                    candidates.append(child)
    if extra_dir:
        ep = Path(extra_dir).expanduser()
        if ep not in candidates and ep.exists():
            candidates.insert(0, ep)
    return [str(p) for p in candidates[:25]]


# =============================================================================
# Managed Process Controller (Launch & Safe Stop for GUI-spawned Runs Only)
# =============================================================================
class RunManager:
    """Manages child `run_g5_30yr.py` subprocesses started from this GUI."""

    def __init__(self, preview_engine: CPUGlobePreviewEngine) -> None:
        self.engine = preview_engine
        self._lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.proc_out_dir: str = ""
        self.proc_log_path: str = ""
        self.proc_toml_path: str = ""
        self.last_watch_dir: str = ""

    def get_managed_status(self) -> dict[str, Any]:
        with self._lock:
            if self.proc is None:
                return {"running": False, "pid": None, "out_dir": self.proc_out_dir, "returncode": None}
            rc = self.proc.poll()
            return {
                "running": rc is None,
                "pid": self.proc.pid,
                "out_dir": self.proc_out_dir,
                "log_path": self.proc_log_path,
                "toml_path": self.proc_toml_path,
                "returncode": rc,
            }

    def start_run(
        self,
        cfg_dict: dict[str, dict[str, Any]],
        device: str = "auto",
        no_resume: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": False, "error": f"GUI から起動したジョブ (PID {self.proc.pid}) が既に稼働中です。"}

            clean_cfg = sanitize_config_dict(cfg_dict)
            g = int(clean_cfg["grid"]["glevel"])
            yrs = int(clean_cfg["output"]["years"])
            out_dir_raw = str(clean_cfg["output"].get("out_dir", "")).strip()
            out_dir = Path(out_dir_raw).expanduser() if out_dir_raw else get_output_dir(f"glevel{g}_{yrs}yr")
            out_dir.mkdir(parents=True, exist_ok=True)
            clean_cfg["output"]["out_dir"] = str(out_dir)

            run_toml = out_dir / "run_config.toml"
            save_aoba_toml(clean_cfg, run_toml)
            log_path = out_dir / "run.log"

            cmd = [sys.executable, "-u", str(_REPO_ROOT / "scripts" / "run_g5_30yr.py"), "--config", str(run_toml)]
            if device and device != "auto":
                cmd.extend(["--device", device])
            if no_resume:
                cmd.append("--no-resume")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(_SRC_DIR)
            log_file = open(log_path, "a", encoding="utf-8")
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(_REPO_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            log_file.close()
            self.proc_out_dir = str(out_dir)
            self.proc_log_path = str(log_path)
            self.proc_toml_path = str(run_toml)
            self.last_watch_dir = str(out_dir)
            return {
                "ok": True,
                "pid": self.proc.pid,
                "out_dir": str(out_dir),
                "log_path": str(log_path),
                "toml_path": str(run_toml),
            }

    def stop_managed_run(self) -> dict[str, Any]:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return {"ok": False, "error": "GUI 管理下の稼働中ジョブはありません（外部ジョブは保護のためここから停止できません）。"}
            pid = self.proc.pid
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
            return {"ok": True, "pid": pid}

    def build_monitor_payload(
        self,
        out_dir_str: str,
        source: str = "auto",
        frame_req: int = -1,
        esm_field: str = "lai",
    ) -> dict[str, Any]:
        active_procs = _discover_active_processes()
        managed = self.get_managed_status()
        discovered = _discover_output_dirs(self.last_watch_dir)

        chosen_str = out_dir_str.strip()
        if not chosen_str:
            if managed.get("out_dir"):
                chosen_str = managed["out_dir"]
            elif active_procs and active_procs[0].get("out_dir"):
                chosen_str = active_procs[0]["out_dir"]
            elif discovered:
                chosen_str = discovered[0]

        payload: dict[str, Any] = {
            "ok": True,
            "out_dir": chosen_str,
            "discovered_dirs": discovered,
            "active_processes": active_procs,
            "managed_run": managed,
            "csv": {},
            "log_tail": "",
            "frames": [],
            "active_frame_idx": -1,
            "raster": None,
            "meta": {},
        }
        if not chosen_str:
            return payload

        out_root = Path(chosen_str).expanduser()
        if not out_root.exists():
            payload["meta"] = {"status": "waiting_for_dir"}
            return payload

        payload["csv"] = _read_csv_series(out_root)
        payload["log_tail"] = _tail_log_lines(out_root / "run.log", n_lines=80)

        live_npz = out_root / "live_preview.npz"
        snap_frames = _find_zarr_frames(out_root, "snapshots")
        mon_frames = _find_zarr_frames(out_root, "monthly")

        chosen_mode = source
        if source == "auto":
            if live_npz.exists() and (esm_field == "lai" or (not snap_frames and not mon_frames)):
                chosen_mode = "live"
            elif snap_frames:
                chosen_mode = "snapshots"
            elif mon_frames:
                chosen_mode = "monthly"

        if chosen_mode == "live" and live_npz.exists():
            try:
                mtime = live_npz.stat().st_mtime
                cache_key = (str(out_root), "live", 0, mtime)
                cached = self.engine._frame_cache.get(cache_key)
                if cached is None:
                    with np.load(live_npz) as zf:
                        lat_deg = np.asarray(zf["lat_deg"], dtype=np.float32)
                        lon_deg = np.asarray(zf["lon_deg"], dtype=np.float32)
                        t2_c = np.asarray(zf["t2_k"], dtype=np.float32) - 273.15
                        precip = (
                            np.asarray(zf["precip_mm_day"], dtype=np.float32)
                            if "precip_mm_day" in zf
                            else np.zeros_like(t2_c)
                        )
                        olr = np.asarray(zf["olr"], dtype=np.float32) if "olr" in zf else np.full_like(t2_c, 240.0)
                        lai = np.asarray(zf["lai"], dtype=np.float32)
                        surf_type = np.asarray(zf["surf_type"], dtype=np.int8)
                        has_ocn_mesh = "ocn_lat_deg" in zf and "ocn_sst_c" in zf
                        if has_ocn_mesh:
                            ocn_lat_deg = np.asarray(zf["ocn_lat_deg"], dtype=np.float32)
                            ocn_lon_deg = np.asarray(zf["ocn_lon_deg"], dtype=np.float32)
                            ocn_wet = np.asarray(zf["ocn_wet"], dtype=np.uint8) if "ocn_wet" in zf else None
                            sst_c = np.asarray(zf["ocn_sst_c"], dtype=np.float32)
                            ocn_spd = np.asarray(zf["ocn_speed_ms"], dtype=np.float32) if "ocn_speed_ms" in zf else np.zeros_like(sst_c)
                            seaice = np.asarray(zf["ocn_seaice_frac"], dtype=np.float32) if "ocn_seaice_frac" in zf else np.zeros_like(sst_c)
                            ocn_is_wind = False
                        else:
                            ocn_lat_deg = ocn_lon_deg = ocn_wet = None
                            sst_c = np.asarray(zf["sst_c"], dtype=np.float32) if "sst_c" in zf else t2_c
                            seaice = np.asarray(zf["seaice_frac"], dtype=np.float32) if "seaice_frac" in zf else np.zeros_like(t2_c)
                            ocn_spd = np.asarray(zf["u10_ms"], dtype=np.float32) if "u10_ms" in zf else np.zeros_like(t2_c)
                            ocn_is_wind = True
                        yr = int(zf["year"])
                        day_yr = float(zf["day_in_yr"])
                        sim_days = float(zf["sim_days"])
                        wall_spd = float(zf["wall_s_per_day"])
                        date_str = str(zf["date_str"]) if "date_str" in zf else ""
                    if not date_str:
                        run_toml_p = out_root / "run_config.toml"
                        rc = load_aoba_toml(run_toml_p) if run_toml_p.exists() else {}
                        fc = rc.get("forcing", {})
                        cal = AobaCalendar(
                            start_date=str(fc.get("start_date", "2000-01-01")),
                            calendar_type=str(fc.get("calendar_type", "gregorian")),
                        )
                        date_str = cal.date_at(sim_days * 86400.0).iso_date
                    raster = self.engine.regrid_fields(
                        lat_deg, lon_deg, t2_c, precip, lai, surf_type,
                        olr=olr, sst_c=sst_c, seaice_frac=seaice, ocn_speed_ms=ocn_spd,
                        ocn_lat_deg=ocn_lat_deg, ocn_lon_deg=ocn_lon_deg, ocn_wet=ocn_wet,
                        ocn_is_wind=ocn_is_wind,
                    )
                    meta = {
                        "source": "live_preview.npz",
                        "year": yr,
                        "day_in_yr": round(day_yr, 2),
                        "sim_days": round(sim_days, 2),
                        "wall_s_per_day": round(wall_spd, 2),
                        "date_str": date_str,
                        "n_cells": int(lat_deg.size),
                    }
                    cached = {"raster": raster, "meta": meta}
                    self.engine._frame_cache[cache_key] = cached
                payload["raster"] = cached["raster"]
                payload["meta"] = cached["meta"]
                payload["frames"] = snap_frames if snap_frames else mon_frames
                payload["obs_benchmark"] = _compute_obs_benchmark(payload.get("csv", {}))
                return payload
            except Exception:
                pass

        frames = mon_frames if (chosen_mode == "monthly" and mon_frames) else (snap_frames if snap_frames else mon_frames)
        kind_used = "monthly" if (chosen_mode == "monthly" and mon_frames) else ("snapshots" if snap_frames else "monthly")
        payload["frames"] = frames
        if frames:
            idx = len(frames) - 1 if (frame_req < 0 or frame_req >= len(frames)) else int(frame_req)
            finfo = frames[idx]
            store_p = Path(finfo["store"])
            mtime = store_p.stat().st_mtime if store_p.exists() else 0.0
            cache_key = (str(out_root), kind_used, idx, mtime, esm_field)
            cached = self.engine._frame_cache.get(cache_key)
            if cached is None:
                try:
                    lat_deg, lon_deg, t2_c, precip, f3, surf_type, src_lbl, avail_esm, olr_z, sst_z, ice_z = _load_zarr_frame_payload(
                        out_root, finfo, kind_used, esm_field=esm_field
                    )
                    spec = _ESM_GLOBE_SPECS.get(esm_field, _ESM_GLOBE_SPECS["lai"])
                    raster = self.engine.regrid_fields(
                        lat_deg, lon_deg, t2_c, precip, f3, surf_type,
                        esm_range=(spec[1], spec[2]), esm_land_only=spec[3],
                        olr=olr_z, sst_c=sst_z, seaice_frac=ice_z, ocn_is_wind=True,
                    )
                    sim_d = float(finfo["sim_days"])
                    run_toml_p = out_root / "run_config.toml"
                    rc = load_aoba_toml(run_toml_p) if run_toml_p.exists() else {}
                    fc = rc.get("forcing", {})
                    cal = AobaCalendar(
                        start_date=str(fc.get("start_date", "2000-01-01")),
                        calendar_type=str(fc.get("calendar_type", "gregorian")),
                    )
                    meta = {
                        "source": src_lbl,
                        "sim_days": round(sim_d, 2),
                        "date_str": cal.date_at(sim_d * 86400.0).iso_date,
                        "n_cells": int(lat_deg.size),
                        "esm_field": esm_field,
                        "esm_label": spec[0],
                        "avail_esm_fields": avail_esm,
                    }
                    cached = {"raster": raster, "meta": meta}
                    self.engine._frame_cache[cache_key] = cached
                except Exception as exc:
                    payload["meta"] = {"source": "error", "error": str(exc)}
                    return payload
            payload["raster"] = cached["raster"]
            payload["meta"] = cached["meta"]
            payload["active_frame_idx"] = idx
        else:
            payload["meta"] = {"source": "waiting_for_first_output"}

        payload["obs_benchmark"] = _compute_obs_benchmark(payload.get("csv", {}))
        return payload


# =============================================================================
# Interactive Workbench HTML5 UI (Config + Run + Live 3 Rotating Globes)
# =============================================================================
_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<link rel="icon" type="image/png" href="/assets/aoba_icon.png" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+JP:wght@400;500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous" onload="if (typeof CURRENT_DOC_BASE !== 'undefined' && document.getElementById('doc-viewer') && document.getElementById('doc-viewer').innerHTML) loadDoc(CURRENT_DOC_BASE);"></script>
<title>AOBA Studio — 設定・実行・監視・3地球リアルタイム可視化</title>
<style>
  @font-face {
    font-family: "IBM Plex Math";
    src: url("https://cdn.jsdelivr.net/npm/@ibm/plex-math@1.1.0/fonts/complete/woff2/IBMPlexMath-Regular.woff2") format("woff2");
    font-weight: 400;
    font-style: normal;
    font-display: swap;
  }
  :root {
    --bg: #0c1119;
    --panel: #141c27;
    --panel-2: #1b2432;
    --border: #2a374b;
    --text: #e7eef8;
    --muted: #93a4bd;
    --accent: #38bdf8;
    --accent-2: #34d399;
    --warn: #fbbf24;
    --danger: #f87171;
    --sans: "IBM Plex Sans", "IBM Plex Sans JP", "Hiragino Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --math: "IBM Plex Math", "STIX Two Math", "Cambria Math", Georgia, serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    background: linear-gradient(135deg, #101929 0%, #0d1423 100%);
    border-bottom: 1px solid var(--border);
    padding: 12px 22px;
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
    position: sticky; top: 0; z-index: 30;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo-badge {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #0284c7, #10b981);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; color: #fff; font-size: 16px;
  }
  .brand h1 { margin: 0; font-size: 17px; font-weight: 700; }
  .brand p { margin: 0; font-size: 12px; color: var(--muted); }
  .header-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .pill {
    padding: 4px 10px; border-radius: 999px; font-size: 12px;
    background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
  }
  .pill strong { color: var(--accent); }
  .pill.modified strong { color: var(--warn); }
  .pill.live { border-color: rgba(16, 185, 129, 0.5); color: #6ee7b7; }
  button {
    cursor: pointer; border-radius: 7px; border: 1px solid var(--border);
    background: var(--panel-2); color: var(--text);
    padding: 7px 13px; font-size: 13px; font-weight: 600;
    transition: background 0.15s, border-color 0.15s;
  }
  button:hover { background: #243043; border-color: #3d5270; }
  button.primary { background: #0284c7; border-color: #0ea5e9; color: #fff; }
  button.primary:hover { background: #0369a1; }
  button.success { background: #059669; border-color: #10b981; color: #fff; }
  button.success:hover { background: #047857; }
  button.danger { background: #991b1b; border-color: #ef4444; color: #fee2e2; }
  button.danger:hover { background: #7f1d1d; }
  .tabs {
    display: flex; gap: 4px; padding: 9px 22px 0;
    border-bottom: 1px solid var(--border); background: #101620;
    overflow-x: auto;
  }
  .tab-btn {
    border: none; border-bottom: 2px solid transparent; border-radius: 6px 6px 0 0;
    background: transparent; color: var(--muted); padding: 9px 14px; font-size: 13px;
    white-space: nowrap;
  }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); background: var(--panel); }
  main { padding: 18px 22px 60px; max-width: 1520px; margin: 0 auto; }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }
  .grid-2 { display: grid; grid-template-columns: 1.15fr 1fr; gap: 18px; }
  .globes-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  @media (max-width: 1120px) { .grid-2, .globes-3 { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px; margin-bottom: 16px;
  }
  .card h2 { margin: 0 0 6px; font-size: 15px; color: #f1f5f9; }
  .card .sub { margin: 0 0 12px; font-size: 12.5px; color: var(--muted); }
  .field-row {
    display: grid; grid-template-columns: 1fr 165px; gap: 10px;
    align-items: center; padding: 8px 0; border-bottom: 1px solid #1e2837;
  }
  .field-row:last-child { border-bottom: none; }
  .field-label { font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
  .field-desc { font-size: 11.5px; color: var(--muted); }
  .tag-mod {
    font-size: 10px; padding: 1px 6px; border-radius: 4px;
    background: rgba(251, 191, 36, 0.16); color: var(--warn); border: 1px solid rgba(251, 191, 36, 0.35);
  }
  input[type="text"], input[type="number"], select, textarea {
    width: 100%; background: #0a0f17; border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 6px 9px;
    font-size: 13px; font-family: var(--mono);
  }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  .switch { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; font-size: 13px; }
  .switch input { width: 17px; height: 17px; accent-color: var(--accent-2); }
  canvas { display: block; width: 100%; background: #070b12; border-radius: 10px; border: 1px solid #223045; }
  .globe-card {
    background: #0d131d; border: 1px solid #26344a; border-radius: 12px;
    padding: 12px; text-align: center;
  }
  .globe-card h3 { margin: 0 0 4px; font-size: 14px; color: #e2e8f0; }
  .globe-card .g-sub { font-size: 11.5px; color: var(--muted); margin-bottom: 8px; }
  .globe-canvas { width: 100%; max-width: 310px; aspect-ratio: 1 / 1; margin: 0 auto; cursor: grab; border-radius: 10px; }
  .globe-canvas:active { cursor: grabbing; }
  .cbar {
    height: 10px; border-radius: 5px; margin: 10px 16px 4px; border: 1px solid #334155;
  }
  .cbar-labels {
    display: flex; justify-content: space-between; font-size: 11px; color: var(--muted);
    font-family: var(--mono); padding: 0 14px;
  }
  .kpi-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px; margin-top: 10px;
  }
  .kpi {
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px;
  }
  .kpi .k-title { font-size: 11px; color: var(--muted); }
  .kpi .k-val { font-size: 15px; font-weight: 700; color: var(--accent); font-family: var(--mono); }
  .preset-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; align-items: center; }
  .cmd-box {
    background: #0a0e14; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; font-family: var(--mono); font-size: 12px; color: #a5f3fc;
    overflow-x: auto; white-space: nowrap;
  }
  .registry-layout { display: grid; grid-template-columns: 260px 1fr; gap: 16px; }
  @media (max-width: 900px) { .registry-layout { grid-template-columns: 1fr; } }
  .sec-list button {
    width: 100%; text-align: left; margin-bottom: 5px; display: flex;
    justify-content: space-between; align-items: center; font-weight: 500;
  }
  .sec-list button.active { background: #0c4a6e; border-color: #38bdf8; color: #fff; }
  .toast {
    position: fixed; right: 20px; bottom: 20px; background: #064e3b;
    border: 1px solid #10b981; color: #ecfdf5; padding: 10px 16px;
    border-radius: 8px; font-size: 13px; font-weight: 600; display: none; z-index: 60;
  }
  .tree-box, .log-box {
    background: #090d14; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 14px; font-family: var(--mono); font-size: 12px; white-space: pre-wrap;
    overflow: auto; color: #cbd5e1; max-height: 290px;
  }
  .hover-banner {
    background: #101a28; border: 1px solid #2b4263; border-radius: 8px;
    padding: 8px 14px; font-family: var(--mono); font-size: 12.5px; color: #bae6fd;
    margin-bottom: 12px; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;
  }
  .docs-layout { display: grid; grid-template-columns: 280px 1fr; gap: 16px; align-items: start; }
  @media (max-width: 960px) { .docs-layout { grid-template-columns: 1fr; } }
  .doc-sidebar { position: sticky; top: 76px; max-height: calc(100vh - 100px); overflow-y: auto; }
  .toc-link {
    display: block; padding: 4px 6px; font-size: 12px; color: var(--muted);
    text-decoration: none; border-radius: 5px; cursor: pointer;
  }
  .toc-link:hover { background: var(--panel-2); color: var(--accent); }
  .toc-link.h3 { padding-left: 16px; font-size: 11.5px; }
  .doc-content { max-height: calc(100vh - 140px); overflow-y: auto; padding: 20px 26px; font-size: 14px; line-height: 1.72; }
  .doc-content h1 { font-size: 22px; margin-top: 0; border-bottom: 1px solid var(--border); padding-bottom: 8px; color: #f8fafc; }
  .doc-content h2 { font-size: 17px; margin-top: 26px; border-bottom: 1px solid #223045; padding-bottom: 5px; color: var(--accent); }
  .doc-content h3 { font-size: 15px; margin-top: 18px; color: #a7f3d0; }
  .doc-content table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 12.5px; }
  .doc-content th, .doc-content td { border: 1px solid var(--border); padding: 6px 9px; text-align: left; vertical-align: top; }
  .doc-content th { background: var(--panel-2); color: #f1f5f9; }
  .doc-content pre { background: #090d14; border: 1px solid var(--border); padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; }
  .doc-content code { background: #0d1522; padding: 1px 5px; border-radius: 4px; font-family: var(--mono); font-size: 12.5px; color: #bae6fd; }
  .doc-content pre code { background: transparent; padding: 0; }
  .doc-content blockquote { border-left: 3px solid var(--accent); margin: 10px 0; padding: 4px 12px; color: var(--muted); background: #101723; border-radius: 0 6px 6px 0; }
  .math-block {
    background: #0d1523; border: 1px solid #24344c; border-radius: 8px;
    padding: 11px 16px; margin: 12px 0; overflow-x: auto;
    font-family: var(--math);
    font-size: 16px; color: #f1f5f9;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
  }
  .math-block-eq { flex: 1; text-align: center; line-height: 1.65; font-style: normal; overflow-x: auto; overflow-y: hidden; }
  .math-tag { flex-shrink: 0; font-family: var(--mono); font-size: 12.5px; font-style: normal; color: #94a3b8; user-select: text; }
  .math-inline { font-family: var(--math); font-style: normal; color: #e0f2fe; padding: 0 1px; }
  .math-block .katex-display { margin: 0; }
  .math-block .katex { font-size: 1.04em; color: #f1f5f9; }
  .math-inline .katex { font-size: 1.01em; color: #e0f2fe; }
  .katex .text { font-family: var(--sans); }
  .math-rm { font-style: normal; }
  .math-it { font-style: italic; }
  .math-bf { font-style: normal; font-weight: 700; }
  .math-block sup, .math-block sub, .math-inline sup, .math-inline sub { font-size: 0.72em; line-height: 0; position: relative; vertical-align: baseline; }
  .math-block sup, .math-inline sup { top: -0.45em; }
  .math-block sub, .math-inline sub { bottom: -0.25em; }
  .math-frac { display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; font-size: 0.88em; margin: 0 3px; line-height: 1.15; }
  .math-frac > span:first-child { border-bottom: 1px solid #cbd5e1; padding: 0 4px 2px; }
  .math-frac > span:last-child { padding: 2px 4px 0; }
  .math-sqrt { display: inline-flex; align-items: stretch; vertical-align: middle; }
  .math-sqrt-rad { font-style: normal; font-size: 1.12em; line-height: 1.0; padding-right: 1px; }
  .math-sqrt-body { border-top: 1px solid #cbd5e1; padding: 0 3px; }
  .math-over { display: inline-block; border-top: 1px solid #cbd5e1; line-height: 1.05; padding-top: 1px; }
  .math-under { display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; line-height: 1.1; }
  .math-under > span:last-child { font-size: 0.72em; color: #94a3b8; border-top: 1px solid #64748b; padding-top: 1px; }
  .math-aligned { border-collapse: collapse; margin: 0 auto; }
  .math-aligned td { border: none !important; padding: 2px 4px !important; background: transparent !important; vertical-align: middle; }
  .math-aligned td.al-r { text-align: right; }
  .math-aligned td.al-l { text-align: left; }
  .math-cases-wrap { display: inline-flex; align-items: center; vertical-align: middle; }
  .math-cases-brace { font-style: normal; font-size: 2.1em; font-weight: 200; line-height: 0.9; margin-right: 4px; color: #cbd5e1; }
  .doc-fig { margin: 16px 0; text-align: center; }
  .doc-fig img { max-width: 100%; height: auto; border-radius: 10px; border: 1px solid var(--border); box-shadow: 0 4px 18px rgba(0,0,0,0.35); background: #f8fafc; }
  .doc-fig figcaption { font-size: 12px; color: var(--muted); margin-top: 6px; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <canvas id="aoba-header-icon" width="96" height="96"
            onclick="triggerOthelloFlip()"
            title="AOBA Othello Grid — Click to flip / クリックでオセロ反転"
            style="width:46px;height:46px;border-radius:11px;display:block;cursor:pointer;flex-shrink:0;"></canvas>
    <div>
      <h1>AOBA Studio</h1>
      <p data-i18n="brand_sub">設定 (TOML <span id="param-total">188</span>項目) → 実行 → 監視 → CPU リアルタイム 3地球可視化</p>
    </div>
  </div>
  <div class="header-actions">
    <span class="pill" id="mod-pill"><span data-i18n="mod_prefix">変更:</span> <strong id="mod-count">0</strong></span>
    <span class="pill" id="run-badge" data-i18n="run_idle">実行監視待機中</span>
    <button id="lang-btn" onclick="toggleLang()">🌐 日本語 / EN</button>
    <button onclick="resetAll()" data-i18n="btn_reset">デフォルトに戻す</button>
    <button class="success" onclick="saveTomlPrompt()" data-i18n="btn_save">TOML保存</button>
    <button class="primary" onclick="launchFromStudio()" data-i18n="btn_launch">▶ この設定で実行＆監視</button>
  </div>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('grid', this)" data-i18n="tab_grid">1. 格子・Glevel可視化</button>
  <button class="tab-btn" onclick="switchTab('forcing', this)" data-i18n="tab_forcing">2. 惑星外力・動的氷床</button>
  <button class="tab-btn" onclick="switchTab('output', this)" data-i18n="tab_output">3. 出力ファイル・間隔</button>
  <button class="tab-btn" onclick="switchTab('registry', this)" data-i18n="tab_registry">4. 全物理パラメタ (188)</button>
  <button class="tab-btn" onclick="switchTab('monitor', this)" data-i18n="tab_monitor">5. 実行・監視＆3地球プレビュー 🌍</button>
  <button class="tab-btn" onclick="switchTab('toml', this)" data-i18n="tab_toml">6. TOML・CLI</button>
  <button class="tab-btn" onclick="switchTab('docs', this)" data-i18n="tab_docs">7. ドキュメント 📖</button>
</div>

<main>
  <!-- TAB 1: GRID & VISUAL DIAGRAM -->
  <div id="tab-grid" class="tab-pane active">
    <div class="grid-2">
      <div class="card">
        <h2 data-i18n="g_canvas_title">全球正20面体 SCVT 格子 &amp; 埋め込み CRM スケール図</h2>
        <p class="sub" data-i18n="g_canvas_sub">Glevel に応じた全球 Voronoi セル分割と、埋め込まれる周期 CRM 領域・鉛直ストレッチ層をリアルタイム描画します。</p>
        <canvas id="gridCanvas" width="760" height="440"></canvas>
        <div class="kpi-strip">
          <div class="kpi"><div class="k-title" data-i18n="kpi_ncell">全球セル数 N<sub>cell</sub></div><div class="k-val" id="kpi-ncell">-</div></div>
          <div class="kpi"><div class="k-title" data-i18n="kpi_dx">ホスト平均格子幅 Δx</div><div class="k-val" id="kpi-dx">-</div></div>
          <div class="kpi"><div class="k-title" data-i18n="kpi_crmw">CRM ドメイン幅</div><div class="k-val" id="kpi-crmw">-</div></div>
          <div class="kpi"><div class="k-title" data-i18n="kpi_subcols">全球総サブカラム数</div><div class="k-val" id="kpi-subcols">-</div></div>
          <div class="kpi"><div class="k-title" data-i18n="kpi_pts">総3D格子点 (×nz)</div><div class="k-val" id="kpi-pts">-</div></div>
        </div>
      </div>
      <div class="card">
        <h2 data-i18n="g_ctrl_title">水平解像度 (SCVT) &amp; 子領域 CRM 設定</h2>
        <div class="preset-bar">
          <button onclick="applyGridPreset(3, 8, 1, 4000, 44, 720, 48000, 36000)" data-i18n="pre_g3">成層圏標準 (G3, 44層 48km)</button>
          <button onclick="applyGridPreset(4, 8, 1, 4000, 44, 720, 48000, 36000)" data-i18n="pre_g4">成層圏気候 (G4, 44層 48km)</button>
          <button onclick="applyGridPreset(5, 8, 1, 4000, 44, 720, 48000, 36000)" data-i18n="pre_g5">成層圏高解像 (G5, 44層 48km)</button>
          <button onclick="applyGridPreset(5, 8, 1, 4000, 32, 720, 24000, 17500)">高速対流圏 (G5, 32層 24km)</button>
          <button onclick="applyGridPreset(4, 8, 8, 4000, 44, 600, 48000, 36000)" data-i18n="pre_3dcrm">3D CRM (G4, 8×8, 48km)</button>
          <button onclick="autoTuneChunks()" style="border-color:var(--accent); color:var(--accent);">⚡ マシン最適チャンク (解像度維持)</button>
          <span id="hw-badge" class="pill" style="font-size:11px;"></span>
        </div>
        <div class="field-row">
          <div>
            <div class="field-label">glevel <span class="field-desc">(正20面体再帰分割レベル: 0〜6)</span></div>
            <input type="range" id="rng-glevel" min="0" max="6" step="1" oninput="setKey('grid','glevel',parseInt(this.value))" />
          </div>
          <input type="number" id="inp-grid-glevel" min="0" max="7" step="1" onchange="setKey('grid','glevel',parseInt(this.value))" />
        </div>
        <div id="grid-fields"></div>
      </div>
    </div>
  </div>

  <!-- TAB 2: FORCING & DYNAMIC ICESHEET -->
  <div id="tab-forcing" class="tab-pane">
    <div class="preset-bar">
      <button onclick="applyForcingPreset('present')" data-i18n="pre_present">現代気候 (400 ppmv, 1×Ω)</button>
      <button onclick="applyForcingPreset('full_esm')" style="border-color:var(--accent); color:var(--accent);">🌍 Full-ESM + RRTMGP</button>
      <button onclick="applyForcingPreset('half_omega')" data-i18n="pre_half_w">自転半減実験 (0.5×Ω)</button>
      <button onclick="applyForcingPreset('slow_rot')" data-i18n="pre_slow_w">超低速自転 (0.25×Ω)</button>
      <button onclick="applyForcingPreset('lgm')" data-i18n="pre_lgm">氷期実験 (185 ppmv + 動的氷床)</button>
      <button onclick="applyForcingPreset('4xco2')" data-i18n="pre_4co2">4×CO₂ 温暖化 + 動的氷床</button>
      <button onclick="applyForcingPreset('aquaplanet')">💧 Aquaplanet</button>
      <button onclick="applyForcingPreset('snowball')">❄️ Snowball Earth</button>
      <button onclick="applyForcingPreset('trappist1e')">🪐 TRAPPIST-1e (潮汐固定)</button>
    </div>
    <div class="grid-2">
      <div class="card">
        <h2 data-i18n="f_title">惑星・軌道・大気境界外力 (Forcing)</h2>
        <div id="forcing-fields"></div>
      </div>
      <div>
        <div class="card">
          <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; margin-bottom:8px;">
            <h2 style="margin:0;">大気 CO₂ 強制シナリオ ＆ 外部ソース更新</h2>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
              <button class="btn-secondary" style="padding:5px 10px; font-size:12px;" onclick="fetchLatestCO2(false)">🌐 NOAA最新取得</button>
              <button class="btn-secondary" style="padding:5px 10px; font-size:12px;" onclick="fetchLatestCO2(true)">🔄 全ソース再同期</button>
            </div>
          </div>
          <div id="co2-status-lbl" class="sub" style="margin-bottom:8px;">Loading CO2 catalog...</div>
          <canvas id="co2Canvas" width="600" height="180"></canvas>
          <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; margin-top:10px; padding-top:8px; border-top:1px solid #1e293b;">
            <span id="o3-status-lbl" class="sub" style="margin:0;">O₃ 外部境界: 未確認</span>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
              <button class="btn-secondary" style="padding:4px 9px; font-size:11px;" onclick="fetchOzoneData('CMIP6')">🌐 CMIP6 O₃取得</button>
              <button class="btn-secondary" style="padding:4px 9px; font-size:11px;" onclick="fetchOzoneData('CMIP7')">🌐 CMIP7 O₃取得</button>
            </div>
          </div>
        </div>
        <div class="card">
          <h2 data-i18n="ice_title">平衡/加速 SIA 氷床モデル設定 (Dynamic Ice Sheet)</h2>
          <div id="icesheet-fields"></div>
        </div>
        <div class="card">
          <h2 data-i18n="ice_chart_title">氷床加速係数 &amp; EMA 減衰曲線</h2>
          <canvas id="iceCanvas" width="600" height="220"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 3: OUTPUT STREAMS & CADENCE -->
  <div id="tab-output" class="tab-pane">
    <div class="grid-2">
      <div class="card">
        <h2 data-i18n="out_title">出力ファイル種類・時間平均 / 瞬時値・変数群</h2>
        <div id="output-fields"></div>
      </div>
      <div class="card">
        <h2 data-i18n="out_tree_title">出力ディレクトリ構成 &amp; 推定データ容量</h2>
        <div class="kpi-strip">
          <div class="kpi"><div class="k-title" data-i18n="est_mean">時間平均 Zarr / 年</div><div class="k-val" id="est-mean-gb">-</div></div>
          <div class="kpi"><div class="k-title" data-i18n="est_snap">瞬時値 Zarr / 年</div><div class="k-val" id="est-snap-gb">-</div></div>
          <div class="kpi"><div class="k-title" data-i18n="est_rst">リスタート / 回</div><div class="k-val" id="est-rst-gb">-</div></div>
          <div class="kpi"><div class="k-title" data-i18n="est_tot">合計 (全期間)</div><div class="k-val" id="est-total-gb">-</div></div>
        </div>
        <h3 style="margin:14px 0 6px; font-size:13px; color:var(--muted);" data-i18n="out_struct_lbl">生成されるディレクトリ構造</h3>
        <div class="tree-box" id="output-tree"></div>
      </div>
    </div>
  </div>

  <!-- TAB 4: FULL PARAMETER REGISTRY -->
  <div id="tab-registry" class="tab-pane">
    <div class="card" style="padding:12px 16px; display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
      <input type="text" id="search-box" placeholder="パラメタ名で検索 (例: v_cmax, albedo, sia, mynn...)" style="max-width:420px;" oninput="renderRegistrySection()" />
      <label class="switch"><input type="checkbox" id="filter-mod" onchange="renderRegistrySection()" /> 変更された項目のみ表示</label>
    </div>
    <div class="registry-layout">
      <div class="card sec-list" id="sec-buttons"></div>
      <div class="card">
        <h2 id="reg-sec-title">-</h2>
        <p class="sub">保存された TOML から起動時に各モジュール定数・Metal カーネル定数へ自動同期されます。</p>
        <div id="reg-fields"></div>
      </div>
    </div>
  </div>

  <!-- TAB 5: RUN, MONITOR & 3 ROTATING ORTHOGRAPHIC GLOBES -->
  <div id="tab-monitor" class="tab-pane">
    <div class="card">
      <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between;">
        <div>
          <h2 style="margin:0;" data-i18n="mon_ctrl_title">実行コントロール ＆ CPU リアルタイム監視</h2>
          <p class="sub" style="margin:2px 0 0;" data-i18n="mon_ctrl_sub">MPS GPU 上で稼働中の計算（既存の外部ジョブやこの画面から起動した実験）を邪魔せず、CPU バックグラウンド処理でリアルタイムプレビューを生成します。</p>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center;">
          <select id="launch-device" style="width:120px;">
            <option value="auto">Auto (GPU)</option>
            <option value="cuda">CUDA GPU</option>
            <option value="mps">MPS GPU</option>
            <option value="cpu">CPU</option>
          </select>
          <label class="switch" style="font-size:12px;"><input type="checkbox" id="launch-no-resume" /> <span data-i18n="lbl_no_resume">最初から (--no-resume)</span></label>
          <button class="primary" id="btn-start-run" onclick="launchFromStudio()" data-i18n="btn_run_new">▶ この設定で新規実行</button>
          <button class="danger" id="btn-stop-run" onclick="stopManagedRun()" style="display:none;" data-i18n="btn_stop">■ ジョブを停止</button>
        </div>
      </div>

      <div style="display:grid; grid-template-columns: 1.5fr 1fr 1fr 1fr; gap:10px; margin-top:12px; align-items:end;">
        <div>
          <label style="font-size:11.5px; color:var(--muted);" data-i18n="lbl_watch_dir">監視対象の出力ディレクトリ (既存実験を選択)</label>
          <select id="watch-dir-select" onchange="onWatchDirSelect(this.value)"></select>
        </div>
        <div>
          <label style="font-size:11.5px; color:var(--muted);" data-i18n="lbl_watch_src">表示ソース</label>
          <select id="watch-source" onchange="fetchMonitorOnce()">
            <option value="auto" data-i18n="opt_auto">自動 (Live優先 / Zarr)</option>
            <option value="live" data-i18n="opt_live">Live Preview (日次)</option>
            <option value="snapshots" data-i18n="opt_snap">Snapshots Zarr (瞬時値)</option>
            <option value="monthly" data-i18n="opt_mon">Monthly Zarr (月平均)</option>
          </select>
        </div>
        <div>
          <label style="font-size:11.5px; color:var(--muted);" data-i18n="lbl_watch_frame">履歴フレーム選択 (Zarr閲覧時)</label>
          <select id="watch-frame" onchange="fetchMonitorOnce()"></select>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <button onclick="fetchMonitorOnce()" style="flex:1;" data-i18n="btn_refresh">⟳ 今すぐ更新</button>
          <label class="switch" style="font-size:12px;"><input type="checkbox" id="auto-poll" checked /> <span data-i18n="lbl_autopoll">2秒自動</span></label>
        </div>
      </div>

      <div class="kpi-strip">
        <div class="kpi"><div class="k-title" data-i18n="mkpi_proc">稼働プロセス</div><div class="k-val" id="mon-proc">-</div></div>
        <div class="kpi"><div class="k-title" data-i18n="mkpi_time">表示データ時刻</div><div class="k-val" id="mon-time">-</div></div>
        <div class="kpi"><div class="k-title" data-i18n="mkpi_src">データソース</div><div class="k-val" id="mon-src" style="font-size:12px;">-</div></div>
        <div class="kpi"><div class="k-title" data-i18n="mkpi_cpu">CPU 投影処理時間</div><div class="k-val" id="mon-cpums">-</div></div>
        <div class="kpi"><div class="k-title" data-i18n="mkpi_latest">最新全球 T<sub>s</sub> / OLR / LAI</div><div class="k-val" id="mon-latest" style="font-size:12.5px;">-</div></div>
      </div>
    </div>

    <!-- 6 ROTATING ORTHOGRAPHIC GLOBES (3x2: Atmosphere + Native Fine Ocean + Biosphere) -->
    <div class="card">
      <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px;">
        <div>
          <h2 style="margin:0;" data-i18n="globes_title">6地球 リアルタイム正射図法プレビュー (T<sub>2</sub> ・ 赤外雲 OLR ・ 降水 ・ SST ・ 海流 ・ LAI)</h2>
          <p class="sub" style="margin:2px 0 0;" data-i18n="globes_sub">ドラッグで任意の緯度経度へ傾け・回転できます。6つの地球儀が同期して自転します（SST・海流は高解像海洋格子を直接描画）。</p>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
          <label class="switch" style="font-size:12.5px;"><input type="checkbox" id="globe-spin" checked /> <span data-i18n="lbl_spin">自動回転</span></label>
          <label class="switch" style="font-size:12.5px;" title="太陽直下点（昼半球）に追従して1モデル日あたり1周します"><input type="checkbox" id="globe-sync-day" checked /> <span id="lbl-sync-day">昼半球同期</span></label>
          <div style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted);">
            <span data-i18n="lbl_speed">速度</span>
            <input type="range" id="globe-speed" min="2" max="45" value="12" style="width:95px;" />
          </div>
          <label class="switch" style="font-size:12.5px;"><input type="checkbox" id="globe-grid" checked /> <span data-i18n="lbl_grid">経緯線</span></label>
          <button onclick="resetGlobeView()" style="padding:5px 10px; font-size:12px;" data-i18n="btn_reset_view">視点リセット</button>
        </div>
      </div>

      <div class="hover-banner">
        <span id="hover-coords" data-i18n="hover_hint">地球儀の上にマウスを置くと、その地点の緯度経度・T₂・赤外雲 OLR・降水・SST・海流・LAI が表示されます</span>
        <span id="view-angles" style="color:var(--muted);">Center: lat +18.0°, lon +140.0°</span>
      </div>

      <div class="globes-3">
        <div class="globe-card">
          <h3 data-i18n="g_t2_title">地表付近気温 T<sub>2</sub> [°C]</h3>
          <div class="g-sub">0.35 T<sub>sfc</sub> + 0.65 T<sub>a,z1</sub></div>
          <canvas id="globeT2" class="globe-canvas" width="300" height="300"></canvas>
          <div class="cbar" style="background:linear-gradient(90deg, #1e1b4b, #1d4ed8, #38bdf8, #fde047, #f97316, #b91c1c);"></div>
          <div class="cbar-labels"><span>-45°C</span><span>-10°C</span><span>+15°C</span><span>+40°C</span></div>
        </div>
        <div class="globe-card">
          <h3 data-i18n="g_ir_title">衛星赤外雲 OLR [W m⁻²]</h3>
          <div class="g-sub" data-i18n="g_ir_sub">ひまわり赤外風（白＝背の高い冷たい雲頂・暗＝晴天域）</div>
          <canvas id="globeIR" class="globe-canvas" width="300" height="300"></canvas>
          <div class="cbar" style="background:linear-gradient(90deg, #f8fafc, #cbd5e1, #64748b, #1e3a5f, #0b1120);"></div>
          <div class="cbar-labels"><span>110 (深対流)</span><span>180</span><span>250</span><span>320 W/m²</span></div>
        </div>
        <div class="globe-card">
          <h3 data-i18n="g_olr_title">降水量 Precip [mm d⁻¹]</h3>
          <div class="g-sub" data-i18n="g_olr_sub">ITCZ・モンスーン・前線帯のトレーサ</div>
          <canvas id="globeOLR" class="globe-canvas" width="300" height="300"></canvas>
          <div class="cbar" style="background:linear-gradient(90deg, #0f172a, #1e3a8a, #0ea5e9, #34d399, #facc15, #f97316, #ec4899);"></div>
          <div class="cbar-labels"><span>0</span><span>4</span><span>10</span><span>18</span><span>25+ mm/d</span></div>
        </div>
        <div class="globe-card">
          <h3 data-i18n="g_sst_title">海面水温 SST [°C] &amp; 海氷</h3>
          <div id="g-sst-sub" class="g-sub" data-i18n="g_sst_sub">黒潮・湾流・中規模渦・極域海氷（白）</div>
          <canvas id="globeSST" class="globe-canvas" width="300" height="300"></canvas>
          <div class="cbar" style="background:linear-gradient(90deg, #172554, #1d4ed8, #06b6d4, #22c55e, #facc15, #f97316, #b91c1c);"></div>
          <div class="cbar-labels"><span>-2°C</span><span>+6°C</span><span>+15°C</span><span>+24°C</span><span>+32°C</span></div>
        </div>
        <div class="globe-card">
          <h3 id="g-ocn-title" data-i18n="g_ocn_title">表層海流速 |u<sub>ocn</sub>| [m s⁻¹]</h3>
          <div id="g-ocn-sub" class="g-sub" data-i18n="g_ocn_sub">黒潮・メキシコ湾流・南極周極流・渦構造</div>
          <canvas id="globeOCN" class="globe-canvas" width="300" height="300"></canvas>
          <div class="cbar" style="background:linear-gradient(90deg, #040d1c, #0c2d5c, #0284c7, #22d3ee, #fde047, #f97316);"></div>
          <div id="g-ocn-cbar-labels" class="cbar-labels"><span>0.0</span><span>0.3</span><span>0.6</span><span>0.9</span><span>1.2+ m/s</span></div>
        </div>
        <div class="globe-card">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:2px;">
            <h3 id="g-esm-title" data-i18n="g_lai_title" style="margin:0;">葉面積指数 LAI [m² m⁻²]</h3>
            <select id="esm-globe-field" onchange="fetchMonitorOnce()" style="font-size:11px; padding:2px 6px; width:auto; max-width:145px;">
              <option value="lai">LAI (植生)</option>
              <option value="o3_tco_du">全オゾン O₃ [DU]</option>
              <option value="lightning_flash_rate">発雷頻度 [fl/km²/yr]</option>
              <option value="volc_aod">火山エアロゾル AOD</option>
              <option value="f_wetland">湿地・湖沼被覆率</option>
              <option value="d18o_precip_permil">降水 δ¹⁸O [‰]</option>
              <option value="co2_sfc_ppmv">地表 CO₂ [ppmv]</option>
              <option value="ocn_chl_sfc">海洋 Chl [mg/m³]</option>
              <option value="ocn_o2_sfc">海洋溶存 O₂ [mmol/m³]</option>
              <option value="wave_hs_m">有義波高 H_s [m]</option>
            </select>
          </div>
          <div id="g-esm-sub" class="g-sub" data-i18n="g_lai_sub">自律 EEO バイオーム分布（海＝濃紺・氷床＝白）</div>
          <canvas id="globeLAI" class="globe-canvas" width="300" height="300"></canvas>
          <div id="g-esm-cbar" class="cbar" style="background:linear-gradient(90deg, #92704a, #a3a332, #4d9e38, #15803d, #064e3b);"></div>
          <div id="g-esm-cbar-labels" class="cbar-labels"><span>0</span><span>2</span><span>4</span><span>7 m²/m²</span></div>
        </div>
      </div>
    </div>

    <div class="card" id="obs-benchmark-card" style="margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <h2 style="margin:0;">🛰️ 現代地球観測ベンチマーク比較 (ERA5 / CERES / GPCP / MODIS / NSIDC)</h2>
        <span class="sub" style="margin:0;">直近最大365日平均との対照</span>
      </div>
      <div id="obs-benchmark-box" class="sub">日次統計 (daily_global.csv) 待機中...</div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2 data-i18n="ts_title">全球平均時系列モニター (`daily_global.csv`)</h2>
        <canvas id="tsCanvas" width="700" height="270"></canvas>
      </div>
      <div class="card">
        <h2 data-i18n="log_title">実行ログリアルタイム表示 (`run.log`)</h2>
        <div class="log-box" id="run-log-box">...</div>
      </div>
    </div>
  </div>

  <!-- TAB 6: LIVE TOML & RUN COMMAND -->
  <div id="tab-toml" class="tab-pane">
    <div class="card">
      <h2 data-i18n="toml_cmd_title">設定ファイルを指定した手動実行コマンド</h2>
      <div class="cmd-box" id="run-cmd-box"></div>
      <div style="margin-top:10px; display:flex; gap:8px;">
        <button onclick="copyRunCmd()" data-i18n="btn_copy_cmd">コマンドをコピー</button>
        <button onclick="downloadToml()" data-i18n="btn_dl_toml">TOMLダウンロード</button>
      </div>
    </div>
    <div class="card">
      <h2 data-i18n="toml_preview_title">統合設定 TOML プレビュー</h2>
      <div style="display:flex; gap:8px; margin-bottom:10px; align-items:center;">
        <span style="font-size:13px; color:var(--muted); min-width:85px;" data-i18n="lbl_save_path">保存パス:</span>
        <input type="text" id="toml-path-input" />
        <button class="success" onclick="saveTomlDirect()" data-i18n="btn_save_direct">保存</button>
      </div>
      <textarea id="toml-preview" rows="24" readonly></textarea>
    </div>
  </div>

  <!-- TAB 7: MODEL DOCUMENTATION READER -->
  <div id="tab-docs" class="tab-pane">
    <div class="docs-layout">
      <div class="card doc-sidebar">
        <h2 data-i18n="docs_menu_title">モデルドキュメント</h2>
        <div style="display:flex; flex-direction:column; gap:6px; margin-bottom:12px;">
          <button onclick="loadDoc('formulation')" id="docbtn-formulation" class="primary" data-i18n="doc_formulation">数理定式化モデル記述書 (全12章)</button>
          <button onclick="loadDoc('configurations')" id="docbtn-configurations" data-i18n="doc_configurations">実験構成例 (Configurations)</button>
          <button onclick="loadDoc('architecture')" id="docbtn-architecture" data-i18n="doc_architecture">アーキテクチャ ＆ 根拠文献</button>
          <button onclick="loadDoc('readme')" id="docbtn-readme" data-i18n="doc_readme">概要・クイックスタート (README)</button>
        </div>
        <div style="font-size:11.5px; color:var(--muted); margin-bottom:6px;" id="doc-filename-badge">-</div>
        <hr style="border:none; border-top:1px solid var(--border); margin:10px 0;" />
        <div style="font-size:12px; font-weight:700; color:var(--muted); margin-bottom:6px;" data-i18n="docs_toc_title">目次 (クリックで移動)</div>
        <div id="doc-toc"></div>
      </div>
      <div class="card doc-content" id="doc-viewer">
        <p style="color:var(--muted);">Loading documentation...</p>
      </div>
    </div>
  </div>
</main>

<div class="toast" id="toast"></div>

<script>
let STATE = { defaults: {}, config: {}, section_meta: {}, toml_path: "", param_count: 0 };
let LANG = "ja";
let CURRENT_DOC_BASE = "formulation";
let ACTIVE_REG_SEC = "constants";
let MON = { raster: null, rawT2: null, rawOLR: null, rawLAI: null, rawSurf: null, rawCoast: null, outDir: "" };
let GLOBE_VIEW = { lat0: 18.0, lon0: 140.0, dragging: false, lastX: 0, lastY: 0, geoCacheLat: null, geoCache: null, manualOverride: false, subsolarLon: null };
let GRID_VIEW = { lon0: 138.0, lat0: 24.0, dragging: false, lastX: 0, lastY: 0, cache: {}, inflight: {} };

const I18N = {
  ja: {
    brand_sub: '設定 (TOML <span id="param-total">188</span>項目) → 実行 → 監視 → CPU リアルタイム 3地球可視化',
    mod_prefix: "変更:",
    run_idle: "実行監視待機中",
    btn_reset: "デフォルトに戻す",
    btn_save: "TOML保存",
    btn_launch: "▶ この設定で実行＆監視",
    tab_grid: "1. 格子・Glevel可視化",
    tab_forcing: "2. 惑星外力・動的氷床",
    tab_output: "3. 出力ファイル・間隔",
    tab_registry: "4. 全物理パラメタ (188)",
    tab_monitor: "5. 実行・監視＆3地球プレビュー 🌍",
    tab_toml: "6. TOML・CLI",
    tab_docs: "7. ドキュメント 📖",
    g_canvas_title: "全球正20面体 SCVT 格子 & 埋め込み CRM スケール図",
    g_canvas_sub: "Glevel に応じた全球 Voronoi セル分割と、埋め込まれる周期 CRM 領域・鉛直ストレッチ層をリアルタイム描画します。",
    kpi_ncell: "全球セル数 N<sub>cell</sub>",
    kpi_dx: "ホスト平均格子幅 Δx",
    kpi_crmw: "CRM ドメイン幅",
    kpi_subcols: "全球総サブカラム数",
    kpi_pts: "総3D格子点 (×nz)",
    g_ctrl_title: "水平解像度 (SCVT) & 子領域 CRM 設定",
    pre_g3: "標準軽量 (G3, 8×1)",
    pre_g4: "長期気候 (G4, 8×1)",
    pre_g5: "高解像度 (G5, 8×1)",
    pre_3dcrm: "3D CRM (G4, 8×8, 4km)",
    pre_present: "現代気候 (400 ppmv, 1×Ω)",
    pre_half_w: "自転半減実験 (0.5×Ω)",
    pre_slow_w: "超低速自転 (0.25×Ω)",
    pre_lgm: "氷期実験 (185 ppmv + 動的氷床)",
    pre_4co2: "4×CO₂ 温暖化 + 動的氷床",
    f_title: "惑星・軌道・大気境界外力 (Forcing)",
    ice_title: "平衡/加速 SIA 氷床モデル設定 (Dynamic Ice Sheet)",
    ice_chart_title: "氷床加速係数 & EMA 減衰曲線",
    out_title: "出力ファイル種類・時間平均 / 瞬時値・変数群",
    out_tree_title: "出力ディレクトリ構成 & 推定データ容量",
    est_mean: "時間平均 Zarr / 年",
    est_snap: "瞬時値 Zarr / 年",
    est_rst: "リスタート / 回",
    est_tot: "合計 (全期間)",
    out_struct_lbl: "生成されるディレクトリ構造",
    mon_ctrl_title: "実行コントロール ＆ CPU リアルタイム監視",
    mon_ctrl_sub: "MPS GPU 上で稼働中の計算（既存の外部ジョブやこの画面から起動した実験）を邪魔せず、CPU バックグラウンド処理でリアルタイムプレビューを生成します。",
    lbl_no_resume: "最初から (--no-resume)",
    btn_run_new: "▶ この設定で新規実行",
    btn_stop: "■ ジョブを停止",
    lbl_watch_dir: "監視対象の出力ディレクトリ (既存実験を選択)",
    lbl_watch_src: "表示ソース",
    opt_auto: "自動 (Live優先 / Zarr)",
    opt_live: "Live Preview (日次)",
    opt_snap: "Snapshots Zarr (瞬時値)",
    opt_mon: "Monthly Zarr (月平均)",
    lbl_watch_frame: "履歴フレーム選択 (Zarr閲覧時)",
    btn_refresh: "⟳ 今すぐ更新",
    lbl_autopoll: "2秒自動",
    mkpi_proc: "稼働プロセス",
    mkpi_time: "表示データ時刻",
    mkpi_src: "データソース",
    mkpi_cpu: "CPU 投影処理時間",
    mkpi_latest: "最新全球 T<sub>s</sub> / OLR / LAI",
    globes_title: "6地球 リアルタイム正射図法プレビュー (T<sub>2</sub> ・ 赤外雲 OLR ・ 降水 ・ SST ・ 海流 ・ LAI)",
    globes_sub: "ドラッグで任意の緯度経度へ傾け・回転できます。6つの地球儀が同期して自転します（SST・海流は高解像海洋格子を直接描画）。",
    lbl_spin: "自動回転",
    lbl_speed: "速度",
    lbl_grid: "経緯線",
    btn_reset_view: "視点リセット",
    hover_hint: "地球儀の上にマウスを置くと、その地点の緯度経度・T₂・降水量・LAI が表示されます",
    g_t2_title: "地表付近気温 T<sub>2</sub> [°C]",
    g_olr_title: "降水量 Precip [mm d⁻¹]",
    g_olr_sub: "ITCZ・モンスーン・前線帯のトレーサ",
    g_lai_title: "葉面積指数 LAI [m² m⁻²]",
    g_lai_sub: "自律 EEO バイオーム分布（海＝濃紺・氷床＝白）",
    ts_title: "全球平均時系列モニター (daily_global.csv)",
    log_title: "実行ログリアルタイム表示 (run.log)",
    toml_cmd_title: "設定ファイルを指定した手動実行コマンド",
    btn_copy_cmd: "コマンドをコピー",
    btn_dl_toml: "TOMLダウンロード",
    toml_preview_title: "統合設定 TOML プレビュー",
    lbl_save_path: "保存パス:",
    btn_save_direct: "保存",
    docs_menu_title: "モデルドキュメント",
    doc_formulation: "数理定式化モデル記述書 (全12章)",
    doc_architecture: "アーキテクチャ ＆ 根拠文献",
    doc_configurations: "実験構成例 (Configurations)",
    doc_readme: "概要・クイックスタート (README)",
    docs_toc_title: "目次 (クリックで移動)",
  },
  en: {
    brand_sub: 'Config (TOML <span id="param-total">188</span> keys) → Launch → Monitor → CPU Real-Time 3-Globe Preview',
    mod_prefix: "Modified:",
    run_idle: "Monitor Idle",
    btn_reset: "Reset Defaults",
    btn_save: "Save TOML",
    btn_launch: "▶ Launch & Monitor",
    tab_grid: "1. Grid & Glevel",
    tab_forcing: "2. Forcing & Ice Sheet",
    tab_output: "3. Output & Cadence",
    tab_registry: "4. All Parameters (188)",
    tab_monitor: "5. Run, Monitor & 3 Globes 🌍",
    tab_toml: "6. TOML & CLI",
    tab_docs: "7. Documentation 📖",
    g_canvas_title: "Global Icosahedral SCVT Mesh & Embedded CRM Scale Diagram",
    g_canvas_sub: "Live visualization of recursive icosahedral Voronoi cells, nested periodic CRM domain, and stretched vertical levels.",
    kpi_ncell: "Global Cells N<sub>cell</sub>",
    kpi_dx: "Host Cell Width Δx",
    kpi_crmw: "CRM Domain Width",
    kpi_subcols: "Total Subcolumns",
    kpi_pts: "Total 3D Points (×nz)",
    g_ctrl_title: "Horizontal Resolution (SCVT) & Subgrid CRM Settings",
    pre_g3: "Light Test (G3, 8×1)",
    pre_g4: "Climate Run (G4, 8×1)",
    pre_g5: "High-Res (G5, 8×1)",
    pre_3dcrm: "3D CRM (G4, 8×8, 4km)",
    pre_present: "Present Day (400 ppmv, 1×Ω)",
    pre_half_w: "Half Rotation (0.5×Ω)",
    pre_slow_w: "Slow Rotation (0.25×Ω)",
    pre_lgm: "LGM (185 ppmv + Dynamic Ice)",
    pre_4co2: "4×CO₂ + Dynamic Ice Sheet",
    f_title: "Planetary, Orbital & Atmospheric Forcing",
    ice_title: "Equilibrium / Accelerated SIA Ice Sheet (Dynamic Ice Sheet)",
    ice_chart_title: "Ice-Sheet Acceleration & EMA Weight Curve",
    out_title: "Output File Types, Averaging vs Instantaneous & Variable Groups",
    out_tree_title: "Output Directory Layout & Estimated Disk Usage",
    est_mean: "Time-Mean Zarr / yr",
    est_snap: "Snapshots Zarr / yr",
    est_rst: "Restart / checkpoint",
    est_tot: "Total Footprint",
    out_struct_lbl: "Generated Directory Structure",
    mon_ctrl_title: "Run Control & Idle-CPU Live Monitor",
    mon_ctrl_sub: "Spawns runs or attaches to existing active runs without disturbing MPS GPU compute; regrids 3-globe previews strictly on CPU.",
    lbl_no_resume: "Fresh Start (--no-resume)",
    btn_run_new: "▶ Launch This Config",
    btn_stop: "■ Stop GUI Run",
    lbl_watch_dir: "Target Output Directory (auto-discovered)",
    lbl_watch_src: "Data Source",
    opt_auto: "Auto (Live / Zarr)",
    opt_live: "Live Preview (Daily)",
    opt_snap: "Snapshots Zarr (Inst.)",
    opt_mon: "Monthly Zarr (Mean)",
    lbl_watch_frame: "History Frame (Zarr)",
    btn_refresh: "⟳ Refresh Now",
    lbl_autopoll: "Auto 5s",
    mkpi_proc: "Active Process",
    mkpi_time: "Simulated Time",
    mkpi_src: "Active Source",
    mkpi_cpu: "CPU Regrid Time",
    mkpi_latest: "Global Mean T<sub>s</sub> / OLR / LAI",
    globes_title: "Live 6-Globe Preview (T<sub>2</sub> · IR Clouds · Precip · SST · Ocean Currents · LAI)",
    globes_sub: "Drag any globe to tilt or spin synchronously; SST & currents render directly on the fine OGCM mesh.",
    lbl_spin: "Auto Spin",
    lbl_speed: "Speed",
    lbl_grid: "Graticule",
    btn_reset_view: "Reset View",
    hover_hint: "Hover over any globe to inspect latitude, longitude, surface type, T₂, Precip, and LAI",
    g_t2_title: "Surface Air Temperature T<sub>2</sub> [°C]",
    g_olr_title: "Precipitation [mm d⁻¹]",
    g_olr_sub: "ITCZ, monsoon & storm-track tracer",
    g_lai_title: "Leaf Area Index LAI [m² m⁻²]",
    g_lai_sub: "Autonomous EEO biomes (Ocean=Navy, Ice=White)",
    ts_title: "Global-Mean Time Series (daily_global.csv)",
    log_title: "Live Run Log (run.log)",
    toml_cmd_title: "CLI Execution Command with Unified TOML",
    btn_copy_cmd: "Copy Command",
    btn_dl_toml: "Download TOML",
    toml_preview_title: "Unified Configuration TOML Preview",
    lbl_save_path: "Save Path:",
    btn_save_direct: "Save",
    docs_menu_title: "Model Documentation",
    doc_formulation: "Mathematical Formulation (12 Chapters)",
    doc_architecture: "Architecture & Literature Traceability",
    doc_configurations: "Experiment Configurations",
    doc_readme: "Overview & Quick Start (README)",
    docs_toc_title: "Table of Contents",
  },
};

const PARAM_DOCS = {
  glevel: { ja: "正20面体格子の再帰分割回数 (0=12セル, 3=642, 4=2562, 5=10242)", en: "Icosahedral recursion level (N_cell = 10*4^g + 2)" },
  nz: { ja: "大気・CRM 共通の鉛直層数", en: "Shared vertical Charney-Phillips levels" },
  z_top: { ja: "モデル上端高度 [m]", en: "Model top height [m]" },
  dz_min: { ja: "最下層の鉛直格子厚 [m]", en: "Lowest layer thickness [m]" },
  z_sponge_base: { ja: "上空レーリーダンピング層の底面高度 [m]", en: "Rayleigh sponge layer base height [m]" },
  tau_sponge_min: { ja: "モデル上端での最小減衰タイムスケール [s]", en: "Minimum damping timescale at model top [s]" },
  terrain_scale_exp: { ja: "鉛直座標の地形追従減衰スケール指数 p", en: "Klemp STF vertical decay exponent p [-]" },
  crm_nx: { ja: "各ホストセル内 CRM の x方向格子数", en: "Embedded CRM x-grid points per SCVT cell" },
  crm_ny: { ja: "各ホストセル内 CRM の y方向格子数 (1=2.5Dカーテン)", en: "Embedded CRM y-grid points (1 = 2.5D curtain)" },
  crm_dx: { ja: "CRM の水平格子間隔 [m]", en: "CRM horizontal grid spacing [m]" },
  dt_host: { ja: "ホスト大気・結合ステップの時間刻み [s]", en: "Host coupling time step [s]" },
  co2_ppmv: { ja: "大気 CO₂ 濃度 [ppmv] (co2_mode='constant' 時)", en: "Atmospheric CO2 concentration [ppmv] (used when co2_mode='constant')" },
  co2_mode: { ja: "CO₂ 強制モード (constant: 固定値 / scenario: CMIP・観測・古気候 / csv: 独自CSV)", en: "CO2 forcing mode (constant / scenario / csv)" },
  co2_scenario: { ja: "CO₂ シナリオ名 (NOAA観測 / CMIP5 RCP / CMIP6 SSP / 80万年氷床コア)", en: "CO2 scenario key (NOAA Obs / CMIP5 RCP / CMIP6 SSP / 800ka Ice Core)" },
  co2_csv_path: { ja: "独自 CO₂ 時系列 CSV パス (year, co2_ppmv 列; co2_mode='csv' 時)", en: "Custom (year, co2_ppmv) CSV path when co2_mode='csv'" },
  ch4_ppmv: { ja: "大気 CH₄ 濃度 [ppmv] (1850 piControl: 0.80825, 現代: 1.803)", en: "Atmospheric CH4 concentration [ppmv] (1850 piControl: 0.80825)" },
  n2o_ppmv: { ja: "大気 N₂O 濃度 [ppmv] (1850 piControl: 0.27302, 現代: 0.332)", en: "Atmospheric N2O concentration [ppmv] (1850 piControl: 0.27302)" },
  rad_scheme: { ja: "放射伝達スキーム (ckd: 16バンド50g点 Correlated-k / four_band: 旧4バンド)", en: "Radiative transfer scheme (ckd 16-band 50 g-pt / four_band legacy)" },
  ocean_mode: { ja: "海洋モデル選択 (ogcm_3d: 3次元多層OGCM+塩分+海氷移流 / ekman_sw: 1.5層浅水エクマン)", en: "Ocean model mode (ogcm_3d 3D multi-layer OGCM / ekman_sw 1.5-layer shallow water)" },
  ocean_nz: { ja: "3D OGCM の鉛直層数 (ocean_mode='ogcm_3d' 時; 既定18層 0〜6000 m)", en: "Number of 3D OGCM vertical z-levels (default 18 covering 0-6000 m)" },
  enable_seaice_drift: { ja: "3D OGCM 時の風・海流による海氷の力学的移流 (Hibler 1979 準拠)", en: "Enable dynamic wind/current sea-ice drift in 3D OGCM mode" },
  enable_ocean_bgc: { ja: "FlexPFT 最適性海洋生態系＋OMIP/mocsy 2.0 炭酸塩化学・大気海洋CO₂交換を有効化 (既定 false)", en: "Enable FlexPFT optimality marine BGC & OMIP/mocsy 2.0 carbonate chemistry (default false)" },
  enable_river_routing: { ja: "Priority-Flood (Barnes 2014)＋運動波遅延 (mizuRoute) によるボロノイ河川ルーティングを有効化 (既定 false)", en: "Enable map-free Priority-Flood Voronoi river routing & kinematic delay (default false)" },
  enable_land_carbon: { ja: "マップフリー陸域炭素循環（木部・リター・5層凍土C・Rh）＋SPITFIRE自律森林火災を有効化 (既定 false)", en: "Enable map-free terrestrial carbon cycle (wood/litter/5-layer permafrost soil C) & autonomous SPITFIRE wildfire (default false)" },
  enable_seaice_evp_itd: { ja: "MPAS-Seaice 準拠の海氷 EVP レオロジー＋5区分氷厚分布 (ITD)・機械的リッジングを有効化 (既定 false)", en: "Enable MPAS-Seaice Voronoi EVP sea-ice rheology & 5-category ITD with mechanical ridging (default false)" },
  enable_ogcm_fct_zstar: { ja: "FESOM2 準拠の 3D OGCM 高次 Zalesak FCT トレーサー移流＋厳密 z* 体積保存座標を有効化 (既定 false)", en: "Enable FESOM2-style 3D OGCM Zalesak FCT tracer advection & exact z* vertical coordinate (default false)" },
  enable_natural_aerosols: { ja: "マップフリー自然エアロゾル（海塩・海洋 DMS 硫酸塩・森林火災 BC/OC）を有効化 (既定 false)", en: "Enable map-free natural aerosols (sea salt, marine DMS sulfate, wildfire BC/OC) (default false)" },
  microphysics_scheme: { ja: "雲微物理スキーム (two_moment: 2モーメント数濃度+粒径依存放射 / one_moment: 旧SAM 1M)", en: "Cloud microphysics scheme (two_moment prognostic Nc,Ni,re / one_moment legacy KR03)" },
  o3_mode: { ja: "オゾン分布モード (rcemip: 1D一様 / latitude_seasonal: 解析的2D緯度季節 / climatology: 外部帯状平均)", en: "Ozone mode (rcemip 1D / latitude_seasonal 2D / climatology external)" },
  o3_file_path: { ja: "外部オゾン気候値ファイル (.npz / .nc; 空欄時は data/boundary/ozone を参照)", en: "External ozone climatology path (.npz/.nc; blank uses data/boundary/ozone)" },
  o3_lw_coupling: { ja: "長波 9.6 μm 窓領域オゾン吸収を有効化", en: "Enable 9.6 um longwave O3 absorption in LW2 window" },
  enable_nonoro_gwd: { ja: "マップ不要・流れ依存の非地形性重力波 (対流 Beres 2004 + 前線 Charron & Manzini 2002)", en: "Map-free flow-dependent non-orographic GWD (convective + frontal sources)" },
  start_date: { ja: "シミュレーション開始年月日 (YYYY-MM-DD)", en: "Simulation start date (YYYY-MM-DD)" },
  end_date: { ja: "終了年月日 (YYYY-MM-DD、空欄なら years を使用)", en: "Optional end date (YYYY-MM-DD; blank uses years)" },
  calendar_type: { ja: "暦の種類 (gregorian: 閏年あり / noleap: 365日 / 360_day)", en: "Calendar type (gregorian / noleap / 360_day)" },
  insolation_mode: { ja: "日射モード (seasonal: 季節変化 / perpetual_date: 固定日射)", en: "Insolation mode (seasonal / perpetual_date)" },
  fixed_insolation_date: { ja: "固定日射実験の基準日 (例: 2000-03-20 春分, 2000-06-21 夏至)", en: "Fixed orbital date for perpetual insolation experiments" },
  diurnal_cycle: { ja: "日変化あり (OFF にすると日平均日射を一定に保つ)", en: "Enable diurnal solar cycle (OFF = daily-mean insolation)" },
  restart_monthly: { ja: "毎月末に途中再開用チェックポイント (restart_mid.pt) を保存", en: "Save mid-year restart checkpoint at the end of each month" },
  cmip7_output_enabled: { ja: "CMIP7 標準出力 (ESM-BCV 61変数・cmip7_monthly.zarr / cmip7_fx.zarr) を有効化 (スピンアップ中は OFF 推奨)", en: "Enable CMIP7 Data Request v1.2 ESM-BCV monthly Zarr stream (OFF recommended during spin-up)" },
  cmip7_include_plev19: { ja: "CMIP7 3D 大気変数を標準 19 気圧面 (plev19: 1000-1 hPa) へ補間出力", en: "Include 3D atmospheric variables interpolated onto CMIP7 plev19 (1000-1 hPa)" },
  cmip7_include_ocean_3d: { ja: "CMIP7 3D 海洋変数 (thetao, so, uo, vo, wo) を全層出力", en: "Include 3D ocean variables (thetao, so, uo, vo, wo) on ocean depth levels" },
  solar_constant: { ja: "太陽定数 S₀ [W m⁻²]", en: "Total solar irradiance S0 [W m^-2]" },
  omega: { ja: "惑星自転角速度 Ω [rad s⁻¹] (地球標準: 7.292115e-5)", en: "Planetary rotation rate Omega [rad s^-1]" },
  obliquity_deg: { ja: "地軸傾斜角 ε [deg]", en: "Axial obliquity [deg]" },
  eccentricity: { ja: "公転軌道離心率 e [-]", en: "Orbital eccentricity [-]" },
  enable_dynamic_icesheet: { ja: "動的 SIA + EMA 平衡加速氷床モデルを有効化", en: "Enable dynamic SIA + EMA equilibrium ice-sheet coupling" },
  icesheet_ema_years: { ja: "年平均 SMB の指数移動平均時定数 τ_EMA [yr]", en: "EMA smoothing timescale for annual SMB [yr]" },
  icesheet_accel_factor: { ja: "SIA 氷床力学の時間加速倍率 α [-]", en: "SIA ice-sheet dynamics time acceleration factor [-]" },
  icesheet_accel_spinup_years: { ja: "加速率を 1 へ戻す指数減衰時定数 (0 で固定)", en: "Spinup e-folding decay timescale to alpha=1 [yr] (0=constant)" },
  icesheet_isostasy_tau_yr: { ja: "アイソスタシー盤面緩和時定数 [yr]", en: "Lithospheric isostasy relaxation timescale [yr]" },
  out_dir: { ja: "出力ルートディレクトリ (空で ~/aoba_outputs/glevel{g}_{y}yr)", en: "Output root directory (empty -> ~/aoba_outputs/glevel{g}_{y}yr)" },
  years: { ja: "積分年数 [years]", en: "Integration duration [years]" },
  zarr_dir_partition: { ja: "'yearly' (年別) または 'single' (単一Zarr)", en: "'yearly' (per-year subdirs) or 'single' Zarr store" },
  monthly_mean_periods_per_year: { ja: "1年あたりの時間平均出力回数 (12=月平均)", en: "Time-averaged Zarr periods per year (12 = monthly)" },
  snapshots_periods_per_year: { ja: "1年あたりの瞬時値スナップショット回数", en: "Instantaneous snapshot writes per year" },
};

function tKey(k) { return I18N[LANG]?.[k] ?? I18N.ja[k] ?? k; }

function applyLanguage() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.getAttribute("data-i18n");
    if (I18N[LANG]?.[k] !== undefined) el.innerHTML = I18N[LANG][k];
  });
  const totalEl = document.getElementById("param-total");
  if (totalEl && STATE.param_count) totalEl.textContent = STATE.param_count;
  const sb = document.getElementById("search-box");
  if (sb) sb.placeholder = LANG === "ja" ? "パラメタ名で検索 (例: v_cmax, albedo, sia, mynn...)" : "Search parameter keys (e.g. v_cmax, albedo, sia, mynn...)";
  renderAll();
  if (document.getElementById("tab-docs")?.classList.contains("active")) loadDoc(CURRENT_DOC_BASE);
}

function toggleLang() {
  LANG = LANG === "ja" ? "en" : "ja";
  applyLanguage();
  showToast(LANG === "ja" ? "表示言語を日本語に切り替えました" : "Switched UI language to English");
}

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.style.display = "block";
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => { t.style.display = "none"; }, 2800);
}

let CO2_CATALOG = { active_source: "bundled", scenarios: [], status: {} };

async function fetchCO2Catalog() {
  try {
    const res = await fetch("/api/co2_catalog");
    CO2_CATALOG = await res.json();
    updateCO2StatusLabel();
    renderSectionContainer("forcing", "forcing-fields");
    drawCO2Canvas();
  } catch (e) { console.warn("CO2 catalog load failed:", e); }
  fetchOzoneStatus();
}

async function fetchOzoneStatus() {
  try {
    const res = await fetch("/api/ozone_status");
    const d = await res.json();
    const el = document.getElementById("o3-status-lbl");
    if (el) {
      if (d.cache_exists && d.meta && d.meta.mip_era) {
        el.textContent = `O₃ 外部境界: ${d.meta.mip_era} (${(d.meta.p_hpa_range||[]).map(x=>x.toFixed(2)).join("..")} hPa)`;
      } else {
        el.textContent = "O₃ 外部境界: 未取得 (rcemip / latitude_seasonal は不要)";
      }
    }
  } catch (e) { console.warn("O3 status failed:", e); }
}

async function fetchOzoneData(era) {
  const el = document.getElementById("o3-status-lbl");
  if (el) el.textContent = `ESGF input4MIPs (${era}) から帯状平均 O₃ を取得中...`;
  try {
    const res = await fetch("/api/ozone_fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mip_era: era }),
    });
    const d = await res.json();
    if (d.ok) {
      showToast(`${era} オゾン気候値を取得・キャッシュしました`);
      fetchOzoneStatus();
    } else {
      showToast("O₃ 取得エラー: " + (d.error || "unknown"));
    }
  } catch (e) {
    showToast("O₃ 取得に失敗しました: " + e);
  }
}

async function fetchLatestCO2(fullSync) {
  const lbl = document.getElementById("co2-status-lbl");
  if (lbl) lbl.textContent = fullSync ? "Fetching RCMIP (CMIP5/6) + NCEI Paleo + NOAA GML..." : "Fetching latest NOAA GML observations...";
  try {
    const res = await fetch("/api/co2_fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ include_rcmip: Boolean(fullSync), include_paleo: Boolean(fullSync) }),
    });
    const data = await res.json();
    if (data.ok && data.catalog) {
      CO2_CATALOG = data.catalog;
      updateCO2StatusLabel();
      renderSectionContainer("forcing", "forcing-fields");
      drawCO2Canvas();
      const gl = data.status?.sources?.noaa_gml_global;
      showToast(gl ? `最新CO₂更新完了: NOAA ${gl.latest_year}年 = ${gl.latest_ppmv} ppmv` : "CO₂データを更新しました");
    } else {
      showToast("CO₂取得エラー: " + (data.error || "unknown"));
    }
  } catch (e) {
    showToast("CO₂取得に失敗しました: " + e);
  }
}

function updateCO2StatusLabel() {
  const lbl = document.getElementById("co2-status-lbl");
  if (!lbl) return;
  const src = CO2_CATALOG.active_source === "local_cache" ? "ローカル最新キャッシュ (data/forcing/co2)" : "同梱ベースライン (src/aoba/assets)";
  const gl = CO2_CATALOG.status?.sources?.noaa_gml_global;
  const obsTxt = gl ? ` | NOAA最新: ${gl.latest_year}年 = ${gl.latest_ppmv} ppmv` : "";
  lbl.textContent = `ソース: ${src} (${(CO2_CATALOG.scenarios||[]).length} シナリオ)${obsTxt}`;
}

function drawCO2Canvas() {
  const cv = document.getElementById("co2Canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#080f1c";
  ctx.fillRect(0, 0, W, H);
  const f = STATE.config?.forcing || {};
  const mode = f.co2_mode || "constant";
  const padL = 48, padR = 16, padT = 18, padB = 28;
  const pw = W - padL - padR, ph = H - padT - padB;
  if (mode === "constant") {
    const val = Number(f.co2_ppmv || 400);
    ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(padL, padT + ph * 0.5); ctx.lineTo(W - padR, padT + ph * 0.5); ctx.stroke();
    ctx.fillStyle = "#e2e8f0"; ctx.font = "13px system-ui";
    ctx.fillText(`Constant Mode: ${val.toFixed(1)} ppmv`, padL + 12, padT + ph * 0.5 - 10);
    return;
  }
  const scn = (CO2_CATALOG.scenarios || []).find(s => s.key === f.co2_scenario);
  if (!scn || !scn.preview_years || scn.preview_years.length < 2) {
    ctx.fillStyle = "#94a3b8"; ctx.font = "12px system-ui";
    ctx.fillText(`Scenario: ${f.co2_scenario || "-"} (no preview data)`, padL, padT + 20);
    return;
  }
  const xs = scn.preview_years, ys = scn.preview_ppmv;
  const xMin = xs[0], xMax = xs[xs.length - 1];
  const yMin = Math.min(...ys) * 0.92, yMax = Math.max(...ys) * 1.05;
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yy = padT + (ph * i) / 4;
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
    const v = yMax - ((yMax - yMin) * i) / 4;
    ctx.fillStyle = "#94a3b8"; ctx.font = "10px monospace";
    ctx.fillText(v.toFixed(0), 6, yy + 3);
  }
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const px = padL + ((xs[i] - xMin) / Math.max(xMax - xMin, 1)) * pw;
    const py = padT + (1 - (ys[i] - yMin) / Math.max(yMax - yMin, 1)) * ph;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.fillStyle = "#cbd5e1"; ctx.font = "11px system-ui";
  ctx.fillText(`${scn.key} (${xMin.toFixed(0)}..${xMax.toFixed(0)})`, padL + 6, padT + 12);
  ctx.fillStyle = "#94a3b8"; ctx.font = "10px monospace";
  ctx.fillText(xMin.toFixed(0), padL, H - 8);
  ctx.fillText(xMax.toFixed(0), W - padR - 36, H - 8);
}

async function fetchInitial() {
  const res = await fetch("/api/state");
  STATE = await res.json();
  document.getElementById("param-total").textContent = STATE.param_count;
  document.getElementById("toml-path-input").value = STATE.toml_path;
  if (STATE.hardware) {
    const h = STATE.hardware;
    const gpuTxt = h.gpu_cores > 0 ? `${h.gpu_cores} GPUc` : h.backend.toUpperCase();
    const el = document.getElementById("hw-badge");
    if (el) el.textContent = `${h.chip_name} · ${gpuTxt} · ${h.total_ram_gb} GB · ${h.perf_cores} P-cores`;
  }
  renderAll();
  fetchCO2Catalog();
  const urlTab = new URLSearchParams(window.location.search).get("tab") || window.location.hash.replace(/^#/, "");
  if (urlTab && document.getElementById("tab-" + urlTab)) {
    const tabBtn = document.querySelector(`.tab-btn[onclick*="'${urlTab}'"]`);
    switchTab(urlTab, tabBtn);
  }
  fetchMonitorOnce();
}

async function autoTuneChunks() {
  const g = STATE.config.grid || {};
  const qs = new URLSearchParams({
    glevel: g.glevel ?? 3, nz: g.nz ?? 32, crm_nx: g.crm_nx ?? 8, crm_ny: g.crm_ny ?? 1,
  });
  const res = await fetch("/api/hardware/tune?" + qs.toString());
  const data = await res.json();
  if (data.ok && data.chunks) {
    STATE.config.grid.crm_col_chunk = data.chunks.crm_col_chunk;
    STATE.config.grid.rad_col_chunk = data.chunks.rad_col_chunk;
    renderAll();
    const passMsg = (data.chunks.crm_single_pass && data.chunks.rad_single_pass) ? "1パス最速処理" : "メモリ安全分割";
    showToast(`最適チャンク適用 (${passMsg}): CRM=${data.chunks.crm_col_chunk}, RAD=${data.chunks.rad_col_chunk}`);
  }
}

function switchTab(id, btn) {
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-" + id).classList.add("active");
  if (btn) btn.classList.add("active");
  if (id === "grid") drawGridCanvas();
  if (id === "forcing") { drawIceCanvas(); drawCO2Canvas(); }
  if (id === "toml") refreshTomlPreview();
  if (id === "output") updateOutputEstimates();
  if (id === "monitor") fetchMonitorOnce();
  if (id === "docs") loadDoc(CURRENT_DOC_BASE);
}

function isModified(sec, key) {
  const def = STATE.defaults[sec]?.[key];
  const cur = STATE.config[sec]?.[key];
  if (typeof def === "number") return Math.abs(Number(cur) - def) > 1e-12 * Math.max(1, Math.abs(def));
  return cur !== def;
}

function countModified() {
  let n = 0;
  for (const sec of Object.keys(STATE.defaults))
    for (const k of Object.keys(STATE.defaults[sec])) if (isModified(sec, k)) n++;
  return n;
}

function setKey(sec, key, val) {
  const def = STATE.defaults[sec]?.[key];
  if (typeof def === "boolean") val = Boolean(val);
  else if (typeof def === "number") {
    const intKeys = ["glevel","ocean_glevel","nz","ocean_nz","crm_nx","crm_ny","n_host_dyn_substeps","n_crm_dyn_substeps","rad_stride","crm_col_chunk","rad_col_chunk","earth2014_stride","years","monthly_mean_periods_per_year","snapshots_periods_per_year","restart_interval_years","land_eb_iterations"];
    val = intKeys.includes(key) ? Math.round(Number(val)) : Number(val);
    if (Number.isNaN(val)) val = def;
  }
  STATE.config[sec][key] = val;
  updateBadgesAndDiagrams();
}

function buildFieldRow(sec, key, skipKeys = []) {
  if (skipKeys.includes(key)) return "";
  const cur = STATE.config[sec][key];
  const def = STATE.defaults[sec][key];
  const mod = isModified(sec, key);
  const pd = (STATE.param_docs && STATE.param_docs[key]) || PARAM_DOCS[key];
  const desc = pd ? (pd[LANG] || pd.ja || pd.en) : ("Default: " + String(def));
  const modLbl = LANG === "ja" ? "変更" : "MOD";
  let ctrl = "";
  if (key === "zarr_dir_partition") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="yearly" ${cur==="yearly"?"selected":""}>yearly (年別 year_XXXX/ 分割・推奨)</option>
      <option value="single" ${cur==="single"?"selected":""}>single (単一 Zarr ストア)</option>
    </select>`;
  } else if (key === "calendar_type") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="gregorian" ${cur==="gregorian"?"selected":""}>gregorian (グレゴリオ暦・閏年あり)</option>
      <option value="noleap" ${cur==="noleap"?"selected":""}>noleap (365日固定・閏年なし)</option>
      <option value="360_day" ${cur==="360_day"?"selected":""}>360_day (360日理想暦・毎月30日)</option>
    </select>`;
  } else if (key === "co2_mode") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="constant" ${cur==="constant"?"selected":""}>constant (固定数値 ppmv)</option>
      <option value="scenario" ${cur==="scenario"?"selected":""}>scenario (CMIP・NOAA観測・古気候補間)</option>
      <option value="csv" ${cur==="csv"?"selected":""}>csv (独自 CSV ファイル)</option>
      <option value="prognostic" ${cur==="prognostic"?"selected":""}>prognostic (排出駆動・完全双方向予報 CO₂)</option>
    </select>`;
  } else if (key === "co2_scenario") {
    const scns = CO2_CATALOG.scenarios || [];
    if (scns.length > 0) {
      const opts = scns.map(s => `<option value="${s.key}" ${cur===s.key?"selected":""}>[${s.group}] ${LANG==="ja"?s.label_ja:s.label_en}</option>`).join("");
      ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">${opts}</select>`;
    } else {
      ctrl = `<input type="text" value="${String(cur).replace(/"/g,'&quot;')}" onchange="setKey('${sec}','${key}',this.value)" />`;
    }
  } else if (key === "rad_scheme") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="four_band" ${cur==="four_band"?"selected":""}>four_band (高速4バンド解析的2ストリーム)</option>
      <option value="ckd" ${cur==="ckd"?"selected":""}>ckd (標準16バンド 50g点 Correlated-k)</option>
      <option value="ckd_hd" ${cur==="ckd_hd"?"selected":""}>ckd_hd (高精度29バンド 111g点 MSTRN11級)</option>
      <option value="rrtmgp" ${cur==="rrtmgp"?"selected":""}>rrtmgp (公式 HITRAN / MT_CKD 4.1 240g点)</option>
    </select>`;
  } else if (key === "ocean_mode") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="ogcm_3d" ${cur==="ogcm_3d"?"selected":""}>ogcm_3d (3次元多層OGCM: 海底地形・塩分・GM90・海氷移流)</option>
      <option value="ekman_sw" ${cur==="ekman_sw"?"selected":""}>ekman_sw (高速1.5層浅水・風成エクマン海洋)</option>
    </select>`;
  } else if (key === "microphysics_scheme") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="one_moment" ${cur==="one_moment"?"selected":""}>one_moment (SAM 1モーメント固定粒径)</option>
      <option value="two_moment" ${cur==="two_moment"?"selected":""}>two_moment (標準2モーメント KK00 Nc/Ni・CCN/INP)</option>
      <option value="six_class_2m" ${cur==="six_class_2m"?"selected":""}>six_class_2m (6種2モーメント Morrison 雨・雪・霰)</option>
    </select>`;
  } else if (key === "o3_mode") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="rcemip" ${cur==="rcemip"?"selected":""}>rcemip (Wing 2018 1D一様)</option>
      <option value="latitude_seasonal" ${cur==="latitude_seasonal"?"selected":""}>latitude_seasonal (解析的 2D 緯度×季節)</option>
      <option value="climatology" ${cur==="climatology"?"selected":""}>climatology (外部 CMIP6/7 帯状平均)</option>
      <option value="prognostic" ${cur==="prognostic"?"selected":""}>prognostic (地図不要 3D 光化学・PSC 予報オゾン)</option>
    </select>`;
  } else if (key === "planet_preset") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="earth" ${cur==="earth"?"selected":""}>earth (現代／古気候地球)</option>
      <option value="aquaplanet" ${cur==="aquaplanet"?"selected":""}>aquaplanet (100% 全球海・地形なし)</option>
      <option value="snowball_earth" ${cur==="snowball_earth"?"selected":""}>snowball_earth (全球凍結スノーボールアース)</option>
      <option value="tidally_locked" ${cur==="tidally_locked"?"selected":""}>tidally_locked (同期潮汐固定惑星)</option>
      <option value="trappist1e" ${cur==="trappist1e"?"selected":""}>trappist1e (TRAPPIST-1e 系外惑星)</option>
      <option value="mars_like" ${cur==="mars_like"?"selected":""}>mars_like (乾燥寒冷・火星型惑星)</option>
    </select>`;
  } else if (key === "insolation_mode") {
    ctrl = `<select onchange="setKey('${sec}','${key}',this.value)">
      <option value="seasonal" ${cur==="seasonal"?"selected":""}>seasonal (通常の年周季節変化)</option>
      <option value="perpetual_date" ${cur==="perpetual_date"?"selected":""}>perpetual_date (日付固定日射実験)</option>
      <option value="annual_mean" ${cur==="annual_mean"?"selected":""}>annual_mean (年平均固定日射)</option>
    </select>`;
  } else if (key === "start_date" || key === "end_date" || key === "fixed_insolation_date") {
    ctrl = `<input type="date" value="${String(cur||"")}" onchange="setKey('${sec}','${key}',this.value)" />`;
  } else if (typeof def === "boolean") {
    ctrl = `<label class="switch"><input type="checkbox" ${cur?"checked":""} onchange="setKey('${sec}','${key}',this.checked)" /> <span>${cur?"ON":"OFF"}</span></label>`;
  } else if (typeof def === "number") {
    ctrl = `<input type="number" step="any" value="${cur}" onchange="setKey('${sec}','${key}',parseFloat(this.value))" />`;
  } else {
    ctrl = `<input type="text" value="${String(cur).replace(/"/g,'&quot;')}" onchange="setKey('${sec}','${key}',this.value)" />`;
  }
  return `<div class="field-row">
    <div><div class="field-label"><code>${key}</code> ${mod?`<span class="tag-mod">${modLbl}</span>`:''}</div><div class="field-desc">${desc}</div></div>
    <div>${ctrl}</div>
  </div>`;
}

function renderSectionContainer(sec, containerId, skipKeys = []) {
  const el = document.getElementById(containerId);
  if (el && STATE.config[sec]) el.innerHTML = Object.keys(STATE.config[sec]).map(k => buildFieldRow(sec, k, skipKeys)).join("");
}

function renderAll() {
  const g = STATE.config.grid?.glevel ?? 3;
  document.getElementById("rng-glevel").value = Math.min(Math.max(g, 0), 6);
  document.getElementById("inp-grid-glevel").value = g;
  renderSectionContainer("grid", "grid-fields", ["glevel"]);
  renderSectionContainer("forcing", "forcing-fields");
  renderSectionContainer("icesheet", "icesheet-fields");
  renderSectionContainer("output", "output-fields");
  renderSectionButtons();
  renderRegistrySection();
  updateBadgesAndDiagrams();
}

function updateBadgesAndDiagrams() {
  const n = countModified();
  document.getElementById("mod-count").textContent = n;
  document.getElementById("mod-pill").classList.toggle("modified", n > 0);
  const g = STATE.config.grid?.glevel ?? 3;
  document.getElementById("rng-glevel").value = Math.min(Math.max(g, 0), 6);
  document.getElementById("inp-grid-glevel").value = g;
  drawGridCanvas();
  drawIceCanvas();
  drawCO2Canvas();
  updateOutputEstimates();
  updateRunCmd();
}

function applyGridPreset(g, nx, ny, dx, nz, dt, ztop = 48000, zsp = 36000) {
  Object.assign(STATE.config.grid, { glevel: g, crm_nx: nx, crm_ny: ny, crm_dx: dx, nz: nz, dt_host: dt, z_top: ztop, z_sponge_base: zsp });
  renderAll();
  showToast(`格子プリセット G${g} (${nx}×${ny}, dx=${dx}m, nz=${nz}, top=${ztop/1000}km) を適用`);
}

function applyForcingPreset(kind) {
  const f = STATE.config.forcing, ic = STATE.config.icesheet;
  f.co2_mode = "constant";
  f.planet_preset = "earth";
  f.enable_tidal_lock = false;
  if (kind === "present") {
    f.co2_ppmv = 400.0; f.solar_constant = 1361.0; f.omega = 7.292115e-5; f.obliquity_deg = 23.44;
    ic.enable_dynamic_icesheet = false;
  } else if (kind === "full_esm") {
    f.co2_ppmv = 400.0; f.co2_mode = "prognostic"; f.enable_co2_3d_transport = true;
    f.rad_scheme = "rrtmgp"; f.microphysics_scheme = "six_class_2m"; f.o3_mode = "prognostic";
    f.enable_ocean_bgc = true; f.enable_ocean_o2_omz = true; f.enable_land_carbon = true; f.enable_land_nitrogen = true;
    f.enable_lightning = true; f.enable_fire_veg_feedback = true; f.enable_natural_aerosols = true;
    f.enable_aerosol_rad_forcing = true; f.enable_snow_bc_darkening = true; f.enable_bgc_fe_lnox_coupling = true;
    f.enable_bvoc_soa = true; f.enable_prognostic_ch4_n2o = true; f.enable_volcanic_aerosols = true;
    f.enable_solar_cycle = true; f.enable_maml_subgrid_topo = true; f.enable_maml_wetlands_lakes = true;
    f.enable_wave_coupling = true; f.enable_water_isotopes = true;
  } else if (kind === "half_omega") { f.co2_ppmv = 400.0; f.omega = 7.292115e-5 * 0.5; }
  else if (kind === "slow_rot") { f.co2_ppmv = 400.0; f.omega = 7.292115e-5 * 0.25; }
  else if (kind === "lgm") { f.co2_ppmv = 185.0; f.obliquity_deg = 22.95; f.eccentricity = 0.018994; ic.enable_dynamic_icesheet = true; ic.icesheet_accel_factor = 100.0; }
  else if (kind === "4xco2") { f.co2_ppmv = 1600.0; ic.enable_dynamic_icesheet = true; ic.icesheet_accel_factor = 50.0; }
  else if (kind === "aquaplanet") { f.planet_preset = "aquaplanet"; f.obliquity_deg = 0.0; f.eccentricity = 0.0; f.co2_ppmv = 400.0; }
  else if (kind === "snowball") { f.planet_preset = "snowball_earth"; f.solar_constant = 1280.0; f.co2_ppmv = 100.0; }
  else if (kind === "trappist1e") { f.planet_preset = "trappist1e"; f.enable_tidal_lock = true; f.solar_constant = 900.0; f.omega = 1.19e-5; f.obliquity_deg = 0.0; f.eccentricity = 0.0; }
  renderAll();
  showToast(`プリセット (${kind}) を適用しました`);
}

/* --- TAB 1 CANVAS: REAL SCVT VORONOI MESH + CRM + VERTICAL --- */
function b64ToTyped(b64, TypedArr) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TypedArr(bytes.buffer);
}

async function ensureSCVTPreviewMesh(g) {
  if (GRID_VIEW.cache[g] || GRID_VIEW.inflight[g]) return;
  GRID_VIEW.inflight[g] = true;
  try {
    const res = await fetch(`/api/scvt_mesh?glevel=${g}`);
    const data = await res.json();
    if (data && data.ok) {
      GRID_VIEW.cache[g] = {
        glevel: data.glevel,
        nCells: data.n_cells,
        nVerts: data.n_verts,
        nEdges: data.n_edges,
        xyzCell: b64ToTyped(data.xyz_cell_b64, Int16Array),
        xyzVert: b64ToTyped(data.xyz_vert_b64, Int16Array),
        vertsOnCell: b64ToTyped(data.verts_on_cell_b64, Int32Array),
        nVertsOnCell: b64ToTyped(data.n_verts_on_cell_b64, Uint8Array),
        vertsOnEdge: b64ToTyped(data.verts_on_edge_b64, Int32Array),
        surfType: b64ToTyped(data.surf_type_b64, Uint8Array),
      };
      drawGridCanvas();
    }
  } catch (e) {
    /* keep shaded ocean gradient fallback */
  } finally {
    delete GRID_VIEW.inflight[g];
  }
}

function drawGridCanvas() {
  const canvas = document.getElementById("gridCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cfgG = STATE.config.grid || {};
  const g = Math.max(0, Math.min(6, Math.round(cfgG.glevel ?? 3)));
  const nz = Math.max(4, Math.round(cfgG.nz ?? 32));
  const zTop = Number(cfgG.z_top ?? 24000);
  const dzMin = Number(cfgG.dz_min ?? 60);
  const zSponge = Number(cfgG.z_sponge_base ?? 17500);
  const crmNx = Math.max(1, Math.round(cfgG.crm_nx ?? 8));
  const crmNy = Math.max(1, Math.round(cfgG.crm_ny ?? 1));
  const crmDx = Number(cfgG.crm_dx ?? 4000);

  const R_earth_km = 6371.0;
  const nCell = 10 * Math.pow(4, g) + 2;
  const dxHostKm = Math.sqrt((4 * Math.PI * R_earth_km * R_earth_km) / nCell);
  const crmWKm = (crmNx * crmDx) / 1000.0, crmHKm = (crmNy * crmDx) / 1000.0;
  const totalSubcols = nCell * crmNx * crmNy;

  document.getElementById("kpi-ncell").textContent = nCell.toLocaleString();
  document.getElementById("kpi-dx").textContent = dxHostKm.toFixed(0) + " km";
  document.getElementById("kpi-crmw").textContent = `${crmWKm.toFixed(0)}×${crmHKm.toFixed(0)} km`;
  document.getElementById("kpi-subcols").textContent = totalSubcols.toLocaleString();
  document.getElementById("kpi-pts").textContent = (totalSubcols * nz).toLocaleString();

  const cx = 185, cy = 224, rad = 148;
  // Ocean background sphere with limb gradient
  const grad = ctx.createRadialGradient(cx - 42, cy - 48, 25, cx, cy, rad);
  grad.addColorStop(0, "#173c7a"); grad.addColorStop(0.65, "#0d2554"); grad.addColorStop(1, "#071128");
  ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.fillStyle = grad; ctx.fill();

  const mesh = GRID_VIEW.cache[g];
  let hlScreenX = cx + 18, hlScreenY = cy - 12, hlPoly = null;

  if (!mesh) {
    ensureSCVTPreviewMesh(g);
    ctx.fillStyle = "#94a3b8"; ctx.font = "12px sans-serif";
    ctx.fillText(`Loading real SCVT Glevel-${g}...`, cx - 72, cy);
  } else {
    const lonRad = (GRID_VIEW.lon0 * Math.PI) / 180.0;
    const latRad = (GRID_VIEW.lat0 * Math.PI) / 180.0;
    const cl = Math.cos(lonRad), sl = Math.sin(lonRad), cp = Math.cos(latRad), sp = Math.sin(latRad);
    const invS = 1.0 / 32767.0;

    // Project all Voronoi vertices once
    const nV = mesh.nVerts;
    const vx = new Float32Array(nV), vy = new Float32Array(nV), vz = new Float32Array(nV);
    const xyzV = mesh.xyzVert;
    for (let v = 0; v < nV; v++) {
      const x0 = xyzV[v * 3] * invS, y0 = xyzV[v * 3 + 1] * invS, z0 = xyzV[v * 3 + 2] * invS;
      // Rotate around z by -lon0 (bring lon0 to +x), then around y so +x points at viewer (z_cam)
      const xr = cl * x0 + sl * y0;
      const yr = -sl * x0 + cl * y0;
      const zr = z0;
      // Camera right = yr (east), camera up = z*cp - xr*sp (north), camera forward = xr*cp + z*sp
      vx[v] = yr;
      vy[v] = zr * cp - xr * sp;
      vz[v] = xr * cp + zr * sp;
    }

    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.clip();

    // Draw real filled Voronoi polygons colored by Earth2014 surface mask (Ocean/Land/Ice)
    const nC = mesh.nCells, xyzC = mesh.xyzCell, voc = mesh.vertsOnCell, nvoc = mesh.nVertsOnCell, st = mesh.surfType;
    let bestLandScore = -1e9, bestIdx = 0;
    for (let c = 0; c < nC; c++) {
      const x0 = xyzC[c * 3] * invS, y0 = xyzC[c * 3 + 1] * invS, z0 = xyzC[c * 3 + 2] * invS;
      const xr = cl * x0 + sl * y0, yr = -sl * x0 + cl * y0, zr = z0;
      const cx_cam = yr, cy_cam = zr * cp - xr * sp, cz_cam = xr * cp + zr * sp;
      if (cz_cam < -0.08) continue;

      // Pick a prominent mid-latitude land/host cell near front-right to zoom into CRM
      const landBonus = st[c] === 1 ? 0.5 : 0.0;
      const score = cz_cam + landBonus - 1.2 * Math.hypot(cx_cam - 0.16, cy_cam - 0.08);
      if (score > bestLandScore && nvoc[c] === 6 && cz_cam > 0.45) {
        bestLandScore = score; bestIdx = c;
      }

      const deg = nvoc[c];
      const base = c * 6;
      const shade = 0.55 + 0.45 * Math.max(0.0, cz_cam);
      const sType = st[c];
      let rC, gC, bC;
      if (sType === 0) { rC = 20 * shade; gC = 64 * shade; bC = 132 * shade; }
      else if (sType === 2) { rC = 225 * shade; gC = 240 * shade; bC = 250 * shade; }
      else { rC = 54 * shade; gC = 122 * shade; bC = 76 * shade; }
      // Highlight the 12 icosahedral pentagons subtly with warm gold tint on border/fill
      if (deg === 5) { rC = Math.min(255, rC + 28); gC = Math.min(255, gC + 18); }

      ctx.fillStyle = `rgb(${rC | 0},${gC | 0},${bC | 0})`;
      ctx.beginPath();
      for (let k = 0; k < deg; k++) {
        const vid = voc[base + k];
        const px = cx + vx[vid] * rad, py = cy - vy[vid] * rad;
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
    }

    // Stroke true Voronoi edges across the visible hemisphere
    const voe = mesh.vertsOnEdge, nE = mesh.nEdges;
    ctx.strokeStyle = g <= 2 ? "rgba(56,189,248,0.85)" : (g <= 4 ? "rgba(56,189,248,0.52)" : (g === 5 ? "rgba(125,211,252,0.34)" : "rgba(125,211,252,0.22)"));
    ctx.lineWidth = g <= 2 ? 1.5 : (g <= 4 ? 0.9 : 0.55);
    ctx.beginPath();
    for (let e = 0; e < nE; e++) {
      const a = voe[e * 2], b = voe[e * 2 + 1];
      if (vz[a] < -0.02 && vz[b] < -0.02) continue;
      ctx.moveTo(cx + vx[a] * rad, cy - vy[a] * rad);
      ctx.lineTo(cx + vx[b] * rad, cy - vy[b] * rad);
    }
    ctx.stroke();

    // Highlight selected host hexagon + collect its true vertex offsets for zoomed panel
    const degH = nvoc[bestIdx], baseH = bestIdx * 6;
    const xh0 = xyzC[bestIdx*3]*invS, yh0 = xyzC[bestIdx*3+1]*invS, zh0 = xyzC[bestIdx*3+2]*invS;
    const xrh = cl*xh0 + sl*yh0, yrh = -sl*xh0 + cl*yh0, zrh = zh0;
    hlScreenX = cx + yrh * rad;
    hlScreenY = cy - (zrh * cp - xrh * sp) * rad;
    hlPoly = [];
    ctx.beginPath();
    let maxR = 1e-6;
    for (let k = 0; k < degH; k++) {
      const vid = voc[baseH + k];
      const px = cx + vx[vid] * rad, py = cy - vy[vid] * rad;
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      const dxp = px - hlScreenX, dyp = py - hlScreenY;
      maxR = Math.max(maxR, Math.hypot(dxp, dyp));
      hlPoly.push([dxp, dyp]);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(251,191,36,0.42)"; ctx.fill();
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2.0; ctx.stroke();
    hlPoly = hlPoly.map(([dxp, dyp]) => [dxp / maxR, dyp / maxR]);
    ctx.restore();
  }

  // Globe crisp outer rim
  ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2);
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 1.6; ctx.stroke();

  ctx.fillStyle = "#e2e8f0"; ctx.font = "bold 13px sans-serif";
  ctx.fillText(LANG === "ja" ? `実 SCVT Voronoi 格子: Glevel-${g}` : `Real SCVT Voronoi Mesh: Glevel-${g}`, 18, 26);
  ctx.fillStyle = "#94a3b8"; ctx.font = "11.5px monospace";
  ctx.fillText(`N_cell=${nCell.toLocaleString()} (Δx≈${dxHostKm.toFixed(0)}km) · ドラッグ回転可`, 18, 45);

  // Connector to Zoomed Host Voronoi Cell + Nested CRM
  const zcx = 495, zcy = 205, zRad = 88;
  ctx.strokeStyle = "rgba(251,191,36,0.75)"; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(hlScreenX, hlScreenY); ctx.lineTo(zcx - zRad * 0.85, zcy - 18); ctx.stroke(); ctx.setLineDash([]);

  ctx.beginPath();
  if (hlPoly && hlPoly.length >= 5) {
    hlPoly.forEach(([ux, uy], i) => {
      const px = zcx + ux * zRad, py = zcy + uy * zRad;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
  } else {
    for (let i = 0; i < 6; i++) {
      const ang = (Math.PI/3)*i + Math.PI/6, px = zcx+zRad*Math.cos(ang), py = zcy+zRad*Math.sin(ang);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
  }
  ctx.closePath(); ctx.fillStyle = "rgba(16,185,129,0.14)"; ctx.fill(); ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2; ctx.stroke();

  const ratioX = Math.max(0.28, Math.min(0.88, crmWKm / Math.max(dxHostKm, 1)));
  const ratioY = crmNy === 1 ? 0.16 : Math.max(0.24, Math.min(0.88, crmHKm / Math.max(dxHostKm, 1)));
  const boxW = Math.round(zRad * 1.45 * ratioX), boxH = Math.round(zRad * 1.45 * ratioY);
  const bx0 = zcx - boxW/2, by0 = zcy - boxH/2;
  ctx.fillStyle = "rgba(56,189,248,0.22)"; ctx.fillRect(bx0, by0, boxW, boxH);
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 1.6; ctx.strokeRect(bx0, by0, boxW, boxH);
  ctx.strokeStyle = "rgba(56,189,248,0.5)"; ctx.lineWidth = 0.8; ctx.beginPath();
  for (let i = 1; i < Math.min(crmNx, 16); i++) { const x = bx0 + boxW*i/Math.min(crmNx,16); ctx.moveTo(x,by0); ctx.lineTo(x,by0+boxH); }
  for (let j = 1; j < Math.min(crmNy, 16); j++) { const y = by0 + boxH*j/Math.min(crmNy,16); ctx.moveTo(bx0,y); ctx.lineTo(bx0+boxW,y); }
  ctx.stroke();

  ctx.fillStyle = "#f8fafc"; ctx.font = "bold 13px sans-serif";
  ctx.fillText(LANG === "ja" ? "埋め込み CRM (MAML+ICA)" : "Embedded CRM (MAML+ICA)", 400, 26);
  ctx.fillStyle = "#38bdf8"; ctx.font = "12px monospace"; ctx.fillText(`${crmNx}×${crmNy} (dx=${(crmDx/1000).toFixed(1)} km)`, 400, 45);
  ctx.fillStyle = "#cbd5e1"; ctx.font = "11.5px monospace";
  ctx.fillText(`${LANG==="ja"?"ホスト幅":"Host"}: ~${dxHostKm.toFixed(0)} km`, 415, 330);
  ctx.fillText(`${LANG==="ja"?"CRM 域":"CRM"}: ${crmWKm.toFixed(0)}×${crmHKm.toFixed(0)} km`, 415, 348);

  const vx = 640, vw = 85, vyTop = 62, vyBot = 380, vH = vyBot - vyTop;
  const spRatio = Math.max(0, Math.min(1, 1.0 - zSponge / Math.max(zTop, 1)));
  if (spRatio > 0) { ctx.fillStyle = "rgba(251,191,36,0.16)"; ctx.fillRect(vx, vyTop, vw, vH*spRatio); }
  ctx.strokeStyle = "#475569"; ctx.lineWidth = 1.2; ctx.strokeRect(vx, vyTop, vw, vH);

  let r = 1.08;
  for (let it = 0; it < 30; it++) {
    const f = dzMin * (Math.pow(r, nz) - 1)/(r - 1) - zTop;
    const df = dzMin * ((nz*Math.pow(r, nz-1)*(r-1) - (Math.pow(r,nz)-1))/Math.pow(r-1,2));
    r = Math.max(1.001, r - f/(df || 1));
  }
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 0.9; ctx.beginPath();
  let cumZ = 0;
  for (let k = 0; k <= nz; k++) {
    const py = vyBot - (cumZ/zTop)*vH;
    ctx.moveTo(vx, py); ctx.lineTo(vx+vw, py);
    cumZ += dzMin * Math.pow(r, k);
  }
  ctx.stroke();
  ctx.fillStyle = "#f8fafc"; ctx.font = "bold 13px sans-serif"; ctx.fillText(LANG==="ja"?`鉛直層 nz=${nz}`:`Levels nz=${nz}`, 636, 26);
  ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
  ctx.fillText(`top: ${(zTop/1000).toFixed(1)} km`, 636, 45);
  ctx.fillText(`dz0: ${dzMin.toFixed(0)} m`, 636, 404);
}

function drawIceCanvas() {
  const canvas = document.getElementById("iceCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const ic = STATE.config.icesheet || {};
  const enabled = Boolean(ic.enable_dynamic_icesheet), alpha0 = Math.max(1, Number(ic.icesheet_accel_factor ?? 100));
  const tauSpin = Math.max(0, Number(ic.icesheet_accel_spinup_years ?? 0)), emaYr = Math.max(1, Number(ic.icesheet_ema_years ?? 5));
  const padL = 48, padR = 20, padT = 28, padB = 34, cw = W - padL - padR, ch = H - padT - padB;
  ctx.strokeStyle = "#334155"; ctx.strokeRect(padL, padT, cw, ch);
  if (!enabled) {
    ctx.fillStyle = "#64748b"; ctx.font = "bold 14px sans-serif"; ctx.fillText("現在: 動的氷床 OFF (固定 Earth2014 氷床)", padL+85, padT+ch/2);
    return;
  }
  const maxT = Math.max(30, tauSpin > 0 ? tauSpin*4 : 30);
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.4; ctx.beginPath();
  for (let i = 0; i <= 100; i++) {
    const t = (i/100)*maxT, aEff = tauSpin > 0 ? (1+(alpha0-1)*Math.exp(-t/tauSpin)) : alpha0;
    const x = padL+(i/100)*cw, y = padT+ch-((aEff-1)/Math.max(alpha0-1,1))*ch;
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2; ctx.setLineDash([5,4]); ctx.beginPath();
  for (let i = 0; i <= 100; i++) {
    const t = (i/100)*maxT, w = Math.exp(-t/emaYr), x = padL+(i/100)*cw, y = padT+ch-w*ch;
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#e2e8f0"; ctx.font = "12px monospace"; ctx.fillText(`α_eff(t): ×${alpha0} -> ×1 (τ_spin=${tauSpin} yr)`, padL+8, padT+18);
  ctx.fillStyle = "#34d399"; ctx.fillText(`EMA SMB 重み (τ_EMA=${emaYr} yr)`, padL+8, padT+36);
}

function updateOutputEstimates() {
  const g = STATE.config.grid || {}, o = STATE.config.output || {};
  const nCell = 10*Math.pow(4, g.glevel ?? 3)+2, nz = g.nz ?? 32, nx = g.crm_nx ?? 8, ny = g.crm_ny ?? 1, yrs = Math.max(1, o.years ?? 30);
  const bytesRec = (inc2d, inc3d, incSg) => ((inc2d ? 13*nCell*4 : 0) + (inc3d ? 5*nz*nCell*4 : 0) + (incSg ? 17*nCell*ny*nx*4 : 0)) * 0.6;
  const meanGb = o.monthly_mean_enabled ? (bytesRec(o.monthly_mean_include_2d, o.monthly_mean_include_3d, o.monthly_mean_include_subgrid_2d) * (o.monthly_mean_periods_per_year ?? 12))/1e9 : 0;
  const snapGb = o.snapshots_enabled ? (bytesRec(o.snapshots_include_2d, o.snapshots_include_3d, o.snapshots_include_subgrid_2d) * (o.snapshots_periods_per_year ?? 12))/1e9 : 0;
  const rstGb = ((7*nz*nCell + 6*nCell*nz*ny*nx + 18*nCell)*4)/1e9;
  const totGb = (meanGb+snapGb)*yrs + rstGb*Math.ceil(yrs/Math.max(1, o.restart_interval_years ?? 1));
  document.getElementById("est-mean-gb").textContent = (meanGb*1000).toFixed(1) + " MB";
  document.getElementById("est-snap-gb").textContent = (snapGb*1000).toFixed(1) + " MB";
  document.getElementById("est-rst-gb").textContent = (rstGb*1000).toFixed(1) + " MB";
  document.getElementById("est-total-gb").textContent = totGb.toFixed(2) + " GB";
  const root = o.out_dir ? o.out_dir : `~/aoba_outputs/glevel${g.glevel}_${yrs}yr`;
  document.getElementById("output-tree").textContent = `${root}/\n├── live_preview.npz   (日次 T2 / OLR / LAI ライブプレビュー)\n├── mesh.zarr          (SCVT 格子・地形)\n└── ${o.zarr_dir_partition==="single"?"monthly.zarr & snapshots.zarr":"year_0001..year_"+String(yrs).padStart(4,"0")+"/{monthly.zarr, snapshots.zarr, daily_global.csv, restart.pt}"}`;
}

function renderSectionButtons() {
  const el = document.getElementById("sec-buttons");
  el.innerHTML = Object.keys(STATE.config).map(sec => {
    const meta = STATE.section_meta[sec] || { ja: sec, en: sec };
    const lbl = meta[LANG] || meta.ja || sec;
    return `<button class="${sec===ACTIVE_REG_SEC?"active":""}" onclick="selectRegSec('${sec}')"><span>${lbl}</span><span style="font-size:11px;opacity:.75">${Object.keys(STATE.config[sec]||{}).length}</span></button>`;
  }).join("");
}
function selectRegSec(sec) { ACTIVE_REG_SEC = sec; renderSectionButtons(); renderRegistrySection(); }
function renderRegistrySection() {
  const q = (document.getElementById("search-box").value || "").trim().toLowerCase();
  const onlyMod = document.getElementById("filter-mod").checked;
  const c = document.getElementById("reg-fields");
  let rows = "";
  if (q || onlyMod) {
    document.getElementById("reg-sec-title").textContent = q
      ? (LANG === "ja" ? `検索結果: "${q}"` : `Search: "${q}"`)
      : (LANG === "ja" ? "変更済みパラメタ" : "Modified Parameters");
    for (const sec of Object.keys(STATE.config))
      for (const k of Object.keys(STATE.config[sec]))
        if ((!q || k.toLowerCase().includes(q) || sec.toLowerCase().includes(q)) && (!onlyMod || isModified(sec, k)))
          rows += buildFieldRow(sec, k);
    c.innerHTML = rows || `<p style="color:var(--muted);">${LANG==="ja"?"一致する項目がありません。":"No matching parameters."}</p>`;
  } else {
    const meta = STATE.section_meta[ACTIVE_REG_SEC] || { ja: ACTIVE_REG_SEC, en: ACTIVE_REG_SEC };
    const lbl = meta[LANG] || meta.ja || ACTIVE_REG_SEC;
    document.getElementById("reg-sec-title").textContent = `${lbl} ([${ACTIVE_REG_SEC}])`;
    c.innerHTML = Object.keys(STATE.config[ACTIVE_REG_SEC] || {}).map(k => buildFieldRow(ACTIVE_REG_SEC, k)).join("");
  }
}

function getActiveTomlPath() { return (document.getElementById("toml-path-input").value || STATE.toml_path || "config/aoba_custom.toml").trim(); }
function updateRunCmd() { document.getElementById("run-cmd-box").textContent = `PYTHONPATH=src python -u scripts/run_g5_30yr.py --config "${getActiveTomlPath()}"`; }
async function refreshTomlPreview() {
  const res = await fetch("/api/toml", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({config:STATE.config}) });
  document.getElementById("toml-preview").value = (await res.json()).toml_text; updateRunCmd();
}
async function saveTomlDirect() {
  const res = await fetch("/api/save", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({config:STATE.config, save_path:getActiveTomlPath()}) });
  const data = await res.json();
  if (data.ok) { STATE.toml_path = data.path; document.getElementById("toml-path-input").value = data.path; showToast(`保存しました: ${data.path}`); }
}
function saveTomlPrompt() {
  const next = window.prompt("保存先の TOML パスを入力:", getActiveTomlPath());
  if (next && next.trim()) { document.getElementById("toml-path-input").value = next.trim(); saveTomlDirect(); }
}
async function downloadToml() {
  const res = await fetch("/api/toml", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({config:STATE.config}) });
  const blob = new Blob([(await res.json()).toml_text], { type:"text/plain;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "aoba_config.toml"; a.click();
}
async function resetAll() {
  const res = await fetch("/api/reset", { method:"POST" });
  STATE.config = (await res.json()).config; renderAll(); refreshTomlPreview(); showToast("全パラメタを工場出荷デフォルトに戻しました");
}
function copyRunCmd() { navigator.clipboard.writeText(document.getElementById("run-cmd-box").textContent); showToast("コマンドをコピーしました"); }

/* --- EXECUTION & LIVE MONITORING --- */
async function launchFromStudio() {
  const dev = document.getElementById("launch-device").value;
  const noRes = document.getElementById("launch-no-resume").checked;
  const res = await fetch("/api/run/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config: STATE.config, device: dev, no_resume: noRes }),
  });
  const data = await res.json();
  if (!data.ok) { alert("実行エラー: " + data.error); return; }
  MON.outDir = data.out_dir;
  showToast(`ジョブを開始しました (PID ${data.pid}) -> ${data.out_dir}`);
  const monBtn = document.querySelectorAll(".tab-btn")[4];
  switchTab("monitor", monBtn);
}

async function stopManagedRun() {
  if (!confirm("この GUI から起動したジョブを停止しますか？")) return;
  const res = await fetch("/api/run/stop", { method: "POST" });
  const data = await res.json();
  if (data.ok) showToast(`PID ${data.pid} を停止しました`);
  fetchMonitorOnce();
}

function onWatchDirSelect(v) {
  MON.outDir = v;
  document.getElementById("watch-frame").innerHTML = `<option value="-1">最新フレーム</option>`;
  fetchMonitorOnce();
}

function b64ToUint8(b64) {
  const bin = atob(b64), arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}

function renderObsBenchmark(rows) {
  const box = document.getElementById("obs-benchmark-box");
  if (!box) return;
  if (!Array.isArray(rows) || rows.length === 0) {
    box.innerHTML = `<span class="sub">${LANG === "ja" ? "日次統計 (daily_global.csv) 待機中..." : "Waiting for daily_global.csv..."}</span>`;
    return;
  }
  let html = `<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:8px;">`;
  for (const r of rows) {
    const ok = r.status === "ok";
    const badgeColor = ok ? "#22c55e" : "#f59e0b";
    const sign = r.bias >= 0 ? "+" : "";
    html += `<div style="background:#0f172a; border:1px solid #1e293b; border-left:3px solid ${badgeColor}; border-radius:8px; padding:7px 9px; font-size:12px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <strong>${escHtml(r.label)}</strong>
        <span style="color:${badgeColor}; font-weight:700; font-size:11px;">${ok ? "IN BAND" : "BIAS"} (${sign}${r.bias})</span>
      </div>
      <div style="color:#cbd5e1; margin-top:2px;">Model: <b>${r.model_mean}</b> &nbsp;|&nbsp; Obs: ${r.obs_mid} (${escHtml(r.obs_range)})</div>
      <div style="color:#64748b; font-size:10.5px;">Ref: ${escHtml(r.obs_source)}</div>
    </div>`;
  }
  html += `</div>`;
  box.innerHTML = html;
}

async function fetchMonitorOnce() {
  const src = document.getElementById("watch-source")?.value || "auto";
  const frm = document.getElementById("watch-frame")?.value || "-1";
  const esmField = document.getElementById("esm-globe-field")?.value || "lai";
  const qs = new URLSearchParams({ out_dir: MON.outDir || "", source: src, frame: frm, esm_field: esmField });
  try {
    const res = await fetch("/api/monitor/snapshot?" + qs.toString());
    const data = await res.json();
    MON.outDir = data.out_dir || "";
    MON.esmField = (data.meta && data.meta.esm_field) || esmField;
    MON.esmLabel = (data.meta && data.meta.esm_label) || "LAI [m² m⁻²]";

    // Update directory picker
    const dirSel = document.getElementById("watch-dir-select");
    const dirs = data.discovered_dirs || [];
    if (MON.outDir && !dirs.includes(MON.outDir)) dirs.unshift(MON.outDir);
    dirSel.innerHTML = dirs.length
      ? dirs.map(d => `<option value="${d}" ${d===MON.outDir?"selected":""}>${d}</option>`).join("")
      : `<option value="">(出力ディレクトリ未検出)</option>`;

    // Active process badge
    const mg = data.managed_run || {}, aps = data.active_processes || [];
    const btnStart = document.getElementById("btn-start-run");
    const btnStop = document.getElementById("btn-stop-run");
    const runBadge = document.getElementById("run-badge");
    if (btnStart) btnStart.style.display = mg.running ? "none" : "inline-block";
    btnStop.style.display = mg.running ? "inline-block" : "none";
    if (mg.running) {
      document.getElementById("mon-proc").textContent = `GUI起動 PID ${mg.pid}`;
      runBadge.textContent = `稼働中 PID ${mg.pid}`; runBadge.className = "pill live";
    } else if (aps.length > 0) {
      document.getElementById("mon-proc").textContent = `外部稼働 PID ${aps[0].pid} (${aps[0].cpu_pct}%)`;
      runBadge.textContent = `外部監視中 PID ${aps[0].pid}`; runBadge.className = "pill live";
    } else {
      document.getElementById("mon-proc").textContent = "待機中 (静止閲覧)";
      runBadge.textContent = "監視待機中"; runBadge.className = "pill";
    }

    // Update frames selector
    const frmSel = document.getElementById("watch-frame");
    const frames = data.frames || [];
    if (frames.length > 0) {
      const curVal = frmSel.value;
      frmSel.innerHTML = `<option value="-1">最新 (${frames[frames.length-1].label})</option>` +
        frames.map(f => `<option value="${f.global_idx}">${f.label}</option>`).join("");
      if (Array.from(frmSel.options).some(o => o.value === curVal)) frmSel.value = curVal;
    } else {
      frmSel.innerHTML = `<option value="-1">Live / 最新</option>`;
    }

    // Metadata & KPIs
    const meta = data.meta || {};
    MON.wallSPerDay = Number(meta.wall_s_per_day || 0);
    if (meta.sim_days !== undefined && meta.sim_days !== null && !Number.isNaN(Number(meta.sim_days))) {
      const fracDay = ((Number(meta.sim_days) % 1.0) + 1.0) % 1.0;
      GLOBE_VIEW.subsolarLon = wrapLonDeg(180.0 - 360.0 * fracDay);
      const syncDay = document.getElementById("globe-sync-day")?.checked ?? true;
      if (!GLOBE_VIEW.manualOverride && syncDay && !GLOBE_VIEW.dragging) {
        GLOBE_VIEW.lon0 = GLOBE_VIEW.subsolarLon;
      }
    }
    document.getElementById("mon-src").textContent = meta.source || "-";
    document.getElementById("mon-time").textContent =
      meta.sim_days !== undefined
        ? (meta.date_str ? `${meta.date_str} (Day ${meta.sim_days})` : `Day ${meta.sim_days}`)
        : "-";
    const syncLbl = document.getElementById("lbl-sync-day");
    if (syncLbl) {
      syncLbl.textContent = MON.wallSPerDay > 0 ? `昼半球同期 (${MON.wallSPerDay.toFixed(1)}s/日)` : "昼半球同期";
    }
    document.getElementById("mon-cpums").textContent = data.raster ? `${data.raster.cpu_regrid_ms} ms (CPU)` : "-";

    // Update log & CSV chart
    document.getElementById("run-log-box").textContent = data.log_tail || "ログファイルはまだありません。";
    const logEl = document.getElementById("run-log-box"); logEl.scrollTop = logEl.scrollHeight;
    drawTimeSeries(data.csv || {});

    const csv = data.csv || {};
    if (csv.ts_k && csv.ts_k.length > 0) {
      const li = csv.ts_k.length - 1;
      document.getElementById("mon-latest").textContent = `${(csv.ts_k[li]-273.15).toFixed(1)}°C / ${csv.olr_wm2[li].toFixed(0)} / ${csv.lai[li].toFixed(2)}`;
    }

    if (data.raster) {
      MON.raster = data.raster;
      MON.rawT2 = b64ToUint8(data.raster.t2_b64);
      MON.rawIR = data.raster.ir_b64 ? b64ToUint8(data.raster.ir_b64) : null;
      MON.rawOLR = b64ToUint8(data.raster.precip_b64 || data.raster.olr_b64);
      MON.rawSST = data.raster.sst_b64 ? b64ToUint8(data.raster.sst_b64) : null;
      MON.rawSeaIce = data.raster.seaice_b64 ? b64ToUint8(data.raster.seaice_b64) : null;
      MON.rawOcnSpd = data.raster.ocn_spd_b64 ? b64ToUint8(data.raster.ocn_spd_b64) : null;
      MON.rawOcnWet = data.raster.ocn_wet_b64 ? b64ToUint8(data.raster.ocn_wet_b64) : null;
      MON.rawOcnCoast = data.raster.ocn_coast_b64 ? b64ToUint8(data.raster.ocn_coast_b64) : null;
      MON.rawLAI = b64ToUint8(data.raster.lai_b64);
      MON.rawSurf = b64ToUint8(data.raster.surf_b64);
      MON.rawCoast = b64ToUint8(data.raster.coast_b64);
      const sstSub = document.getElementById("g-sst-sub");
      if (sstSub && data.raster.ocn_res_lbl) {
        sstSub.textContent = (LANG === "ja" ? "黒潮・湾流・海氷（白） — " : "Western boundary currents & sea ice — ") + data.raster.ocn_res_lbl;
      }
      const ocnT = document.getElementById("g-ocn-title");
      const ocnS = document.getElementById("g-ocn-sub");
      const ocnL = document.getElementById("g-ocn-cbar-labels");
      if (data.raster.ocn_is_wind) {
        if (ocnT) ocnT.innerHTML = LANG === "ja" ? "地表風速 |u<sub>10</sub>| [m s⁻¹]" : "Surface Wind |u<sub>10</sub>| [m s⁻¹]";
        if (ocnS) ocnS.textContent = LANG === "ja" ? "貿易風・偏西風・低気圧の強風帯" : "Trades, westerlies & storm wind bands";
        if (ocnL) ocnL.innerHTML = "<span>0</span><span>6</span><span>12</span><span>18</span><span>25+ m/s</span>";
      } else {
        if (ocnT) ocnT.innerHTML = LANG === "ja" ? "表層海流速 |u<sub>ocn</sub>| [m s⁻¹]" : "Ocean Current |u<sub>ocn</sub>| [m s⁻¹]";
        if (ocnS) ocnS.textContent = (LANG === "ja" ? "黒潮・湾流・ACC・中規模渦 — " : "Kuroshio, Gulf Stream, ACC & eddies — ") + (data.raster.ocn_res_lbl || "");
        if (ocnL) ocnL.innerHTML = "<span>0.0</span><span>0.3</span><span>0.6</span><span>0.9</span><span>1.2+ m/s</span>";
      }
      const tEl = document.getElementById("g-esm-title");
      const sEl = document.getElementById("g-esm-sub");
      const cEl = document.getElementById("g-esm-cbar");
      const lEl = document.getElementById("g-esm-cbar-labels");
      const rng = data.raster.lai_range || [0, 7];
      if (tEl) tEl.textContent = MON.esmLabel || "LAI [m² m⁻²]";
      if (sEl) sEl.textContent = MON.esmField === "lai"
        ? (LANG === "ja" ? "自律 EEO バイオーム分布（海＝濃紺・氷床＝白）" : "Emergent EEO biome LAI (ocean=navy, ice=white)")
        : `ESM 2D Field (${MON.esmField})`;
      if (cEl) {
        cEl.style.background = MON.esmField === "lai"
          ? "linear-gradient(90deg, #92704a, #a3a332, #4d9e38, #15803d, #064e3b)"
          : "linear-gradient(90deg, #1e1b4b, #1d4ed8, #0ea5e9, #34d399, #facc15, #f97316)";
      }
      if (lEl) {
        const lo = rng[0], hi = rng[1], m1 = lo + (hi - lo) * 0.33, m2 = lo + (hi - lo) * 0.67;
        const fmt = (v) => Math.abs(hi - lo) < 2 ? v.toFixed(2) : (Math.abs(hi - lo) < 20 ? v.toFixed(1) : v.toFixed(0));
        lEl.innerHTML = `<span>${fmt(lo)}</span><span>${fmt(m1)}</span><span>${fmt(m2)}</span><span>${fmt(hi)}</span>`;
      }
    }
    renderObsBenchmark(data.obs_benchmark || []);
  } catch (e) { /* keep polling calmly */ }
}

/* --- ROTATING ORTHOGRAPHIC 3 GLOBES ENGINE --- */
function makeColorLUT(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let s0 = stops[0], s1 = stops[stops.length - 1];
    for (let k = 0; k < stops.length - 1; k++) {
      if (t >= stops[k][0] && t <= stops[k+1][0]) { s0 = stops[k]; s1 = stops[k+1]; break; }
    }
    const f = (t - s0[0]) / Math.max(s1[0] - s0[0], 1e-6);
    lut[i*3]   = Math.round(s0[1][0] + f * (s1[1][0] - s0[1][0]));
    lut[i*3+1] = Math.round(s0[1][1] + f * (s1[1][1] - s0[1][1]));
    lut[i*3+2] = Math.round(s0[1][2] + f * (s1[1][2] - s0[1][2]));
  }
  return lut;
}
const LUT_T2  = makeColorLUT([[0,[30,27,75]],[0.24,[29,78,216]],[0.44,[56,189,248]],[0.60,[253,224,71]],[0.80,[249,115,22]],[1,[185,28,28]]]);
const LUT_IR  = makeColorLUT([[0,[255,255,255]],[0.22,[226,232,240]],[0.48,[116,138,164]],[0.72,[30,58,95]],[1,[11,17,32]]]);
const LUT_OLR = makeColorLUT([[0,[15,23,42]],[0.04,[30,58,138]],[0.16,[14,165,233]],[0.40,[52,211,153]],[0.68,[250,204,21]],[0.88,[249,115,22]],[1,[236,72,153]]]);
const LUT_SST = makeColorLUT([[0,[23,37,84]],[0.22,[29,78,216]],[0.42,[6,182,212]],[0.60,[34,197,94]],[0.76,[250,204,21]],[0.89,[249,115,22]],[1,[185,28,28]]]);
const LUT_OCN = makeColorLUT([[0,[4,13,28]],[0.14,[12,45,92]],[0.34,[2,132,199]],[0.58,[34,211,238]],[0.82,[253,224,71]],[1,[249,115,22]]]);
const LUT_LAI = makeColorLUT([[0,[146,112,74]],[0.20,[163,163,50]],[0.45,[77,158,56]],[0.72,[21,128,61]],[1,[6,78,59]]]);
const LUT_ESM = makeColorLUT([[0,[30,27,75]],[0.20,[29,78,216]],[0.42,[14,165,233]],[0.65,[52,211,153]],[0.84,[250,204,21]],[1,[249,115,22]]]);

function buildOrthographicCache(lat0Deg) {
  const size = 300, cx = 150, cy = 150, R = 132;
  const lat0 = (lat0Deg * Math.PI) / 180, sinL0 = Math.sin(lat0), cosL0 = Math.cos(lat0);
  const pixIdx = [], latRow = [], dlonDeg = [], shade = [], latVal = [];
  for (let py = 0; py < size; py++) {
    const Y = (cy - py) / R;
    for (let px = 0; px < size; px++) {
      const X = (px - cx) / R, r2 = X*X + Y*Y;
      if (r2 > 1.0) continue;
      const Z = Math.sqrt(Math.max(0, 1 - r2));
      const sinLat = Math.max(-1, Math.min(1, Z*sinL0 + Y*cosL0));
      const lat = Math.asin(sinLat) * (180 / Math.PI);
      const dlon = Math.atan2(X, Z*cosL0 - Y*sinL0) * (180 / Math.PI);
      const j = Math.max(0, Math.min(359, Math.floor((90.0 - lat) * 2.0)));
      const li = 0.58 + 0.42 * Math.max(0, 0.78*Z - 0.32*X + 0.38*Y); // subtle 3D sphere shading
      pixIdx.push((py*size + px) * 4);
      latRow.push(j);
      dlonDeg.push(dlon);
      shade.push(li);
      latVal.push(lat);
    }
  }
  return { size, R, cx, cy, count: pixIdx.length, pixIdx: new Int32Array(pixIdx), latRow: new Uint16Array(latRow), dlonDeg: new Float32Array(dlonDeg), shade: new Float32Array(shade), latVal: new Float32Array(latVal) };
}

function getOrthoGeo() {
  if (GLOBE_VIEW.geoCache && Math.abs(GLOBE_VIEW.geoCacheLat - GLOBE_VIEW.lat0) < 0.05) return GLOBE_VIEW.geoCache;
  GLOBE_VIEW.geoCacheLat = GLOBE_VIEW.lat0;
  GLOBE_VIEW.geoCache = buildOrthographicCache(GLOBE_VIEW.lat0);
  return GLOBE_VIEW.geoCache;
}

function wrapLonDeg(lon) {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

function renderIdleGlobe(canvasId, kind, geo) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = geo.size;
  if (!canvas._offCv) {
    canvas._offCv = document.createElement("canvas");
    canvas._offCv.width = W; canvas._offCv.height = W;
    canvas._imgData = canvas._offCv.getContext("2d").createImageData(W, W);
  }
  const offCtx = canvas._offCv.getContext("2d");
  const img = canvas._imgData, buf = img.data;
  buf.fill(0);
  const drawGrid = document.getElementById("globe-grid")?.checked ?? true;
  const lon0 = GLOBE_VIEW.lon0;
  const baseCol = kind === "t2" ? [24, 62, 130] : (kind === "olr" ? [46, 36, 108] : [18, 76, 86]);

  for (let p = 0; p < geo.count; p++) {
    const lon = wrapLonDeg(lon0 + geo.dlonDeg[p]);
    const sh = geo.shade[p];
    let r = baseCol[0] * sh, g = baseCol[1] * sh, b = baseCol[2] * sh;
    if (drawGrid) {
      const lat = geo.latVal[p];
      if (Math.abs(lat % 30) < 0.9 || Math.abs(((lon + 360) % 30)) < 0.9) {
        r = r * 0.72 + 56; g = g * 0.72 + 95; b = b * 0.72 + 135;
      }
    }
    const o = geo.pixIdx[p];
    buf[o] = r; buf[o + 1] = g; buf[o + 2] = b; buf[o + 3] = 255;
  }
  offCtx.putImageData(img, 0, 0);

  ctx.clearRect(0, 0, W, W);
  const halo = ctx.createRadialGradient(geo.cx, geo.cy, geo.R * 0.90, geo.cx, geo.cy, geo.R * 1.10);
  halo.addColorStop(0, "rgba(56,189,248,0.28)"); halo.addColorStop(1, "rgba(56,189,248,0)");
  ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(geo.cx, geo.cy, geo.R * 1.10, 0, Math.PI * 2); ctx.fill();
  ctx.drawImage(canvas._offCv, 0, 0);
  ctx.beginPath(); ctx.arc(geo.cx, geo.cy, geo.R, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(56,189,248,0.65)"; ctx.lineWidth = 1.4; ctx.stroke();

  ctx.fillStyle = "rgba(15,23,42,0.72)";
  ctx.fillRect(geo.cx - 86, geo.cy - 13, 172, 26);
  ctx.strokeStyle = "rgba(56,189,248,0.45)";
  ctx.strokeRect(geo.cx - 86, geo.cy - 13, 172, 26);
  ctx.fillStyle = "#bae6fd"; ctx.font = "11.5px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(LANG === "ja" ? "出力待機中 (▶実行で描画)" : "Waiting for model output...", geo.cx, geo.cy + 4);
  ctx.textAlign = "left";
}

function renderOneGlobe(canvasId, kind, geo) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  if (!MON.raster) { renderIdleGlobe(canvasId, kind, geo); return; }
  const ctx = canvas.getContext("2d");
  const W = geo.size;
  if (!canvas._offCv) {
    canvas._offCv = document.createElement("canvas");
    canvas._offCv.width = W; canvas._offCv.height = W;
    canvas._imgData = canvas._offCv.getContext("2d").createImageData(W, W);
  }
  const offCtx = canvas._offCv.getContext("2d");
  const img = canvas._imgData, buf = img.data;
  buf.fill(0);
  const drawGrid = document.getElementById("globe-grid")?.checked ?? true;
  const lon0 = GLOBE_VIEW.lon0;
  const isOcnGlobe = (kind === "sst") || (kind === "ocn" && !MON.raster.ocn_is_wind);
  const srcVals = kind === "t2" ? MON.rawT2
    : (kind === "ir" ? (MON.rawIR || MON.rawOLR)
    : (kind === "olr" ? MON.rawOLR
    : (kind === "sst" ? (MON.rawSST || MON.rawT2)
    : (kind === "ocn" ? (MON.rawOcnSpd || MON.rawOLR)
    : MON.rawLAI))));
  const isLaiMode = (MON.esmField || "lai") === "lai";
  const landOnly3 = MON.raster.esm_land_only !== false;
  const lut = kind === "t2" ? LUT_T2
    : (kind === "ir" ? LUT_IR
    : (kind === "olr" ? LUT_OLR
    : (kind === "sst" ? LUT_SST
    : (kind === "ocn" ? LUT_OCN
    : (isLaiMode ? LUT_LAI : LUT_ESM)))));
  const surf = MON.rawSurf;
  const ocnWet = MON.rawOcnWet;
  const seaIce = MON.rawSeaIce;
  const coast = (isOcnGlobe && MON.rawOcnCoast) ? MON.rawOcnCoast : MON.rawCoast;

  const rH = (MON.raster && MON.raster.height) || 360;
  const rW = (MON.raster && MON.raster.width) || 720;
  const rowScale = rH / 360.0;
  const lonScale = rW / 360.0;

  for (let p = 0; p < geo.count; p++) {
    const row = Math.max(0, Math.min(rH - 1, Math.floor(geo.latRow[p] * rowScale)));
    const lon = wrapLonDeg(lon0 + geo.dlonDeg[p]);
    const col = Math.max(0, Math.min(rW - 1, Math.floor((lon + 180) * lonScale)));
    const gIdx = row * rW + col;
    const sh = geo.shade[p];

    let r, g, b;
    if (kind === "lai" && landOnly3 && surf[gIdx] === 0) {
      r = 12 * sh; g = 34 * sh; b = 74 * sh; // crisp ocean blue on land-only globe
    } else if (kind === "lai" && landOnly3 && surf[gIdx] === 2) {
      r = 228 * sh; g = 242 * sh; b = 250 * sh; // ice sheets crisp white-cyan
    } else if (isOcnGlobe && (ocnWet ? ocnWet[gIdx] === 0 : surf[gIdx] !== 0)) {
      if (surf[gIdx] === 2) {
        r = 185 * sh; g = 200 * sh; b = 215 * sh; // ice sheets soft silver on ocean globes
      } else {
        r = 34 * sh; g = 42 * sh; b = 56 * sh; // crisp dark slate land mask on SST / current globes
      }
    } else {
      const k3 = srcVals[gIdx] * 3;
      r = lut[k3] * sh; g = lut[k3+1] * sh; b = lut[k3+2] * sh;
      if (kind === "sst" && seaIce && seaIce[gIdx] > 38) {
        const fIce = Math.min(1.0, (seaIce[gIdx] - 38) / 140.0);
        r = r * (1 - fIce) + 236 * sh * fIce;
        g = g * (1 - fIce) + 248 * sh * fIce;
        b = b * (1 - fIce) + 255 * sh * fIce;
      }
    }

    if (coast[gIdx] === 1) {
      if (kind === "olr" || kind === "ir" || kind === "ocn") {
        r = r * 0.45 + 90; g = g * 0.45 + 100; b = b * 0.45 + 120;
      } else {
        r = 12; g = 18; b = 28;
      }
    }

    if (drawGrid) {
      const lat = geo.latVal[p];
      if (Math.abs(lat % 30) < 0.85 || Math.abs(((lon + 360) % 30) - 0) < 0.85) {
        r = r * 0.82 + 45; g = g * 0.82 + 65; b = b * 0.82 + 90;
      }
    }
    const o = geo.pixIdx[p];
    buf[o] = r; buf[o+1] = g; buf[o+2] = b; buf[o+3] = 255;
  }
  offCtx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, W, W);
  const halo = ctx.createRadialGradient(geo.cx, geo.cy, geo.R*0.92, geo.cx, geo.cy, geo.R*1.08);
  halo.addColorStop(0, "rgba(56,189,248,0.28)"); halo.addColorStop(1, "rgba(56,189,248,0)");
  ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(geo.cx, geo.cy, geo.R*1.08, 0, Math.PI*2); ctx.fill();
  ctx.drawImage(canvas._offCv, 0, 0);
}

let _lastAnimTs = 0;
let _lastRenderKey = "";
function animateGlobes(ts) {
  requestAnimationFrame(animateGlobes);
  const monPane = document.getElementById("tab-monitor");
  if (!monPane || !monPane.classList.contains("active")) { _lastAnimTs = ts; return; }
  const dt = Math.min((ts - (_lastAnimTs || ts)) / 1000.0, 0.1);
  _lastAnimTs = ts;

  const syncDay = document.getElementById("globe-sync-day")?.checked ?? true;
  if (!syncDay && (document.getElementById("globe-spin")?.checked ?? true) && !GLOBE_VIEW.dragging) {
    const spd = Number(document.getElementById("globe-speed")?.value ?? 12);
    GLOBE_VIEW.lon0 = wrapLonDeg(GLOBE_VIEW.lon0 + spd * dt);
  }
  const drawGrid = document.getElementById("globe-grid")?.checked ?? true;
  const rKey = `${GLOBE_VIEW.lat0.toFixed(3)}|${GLOBE_VIEW.lon0.toFixed(3)}|${MON.raster?.sim_days ?? "idle"}|${MON.raster?.step ?? ""}|${MON.esmField ?? "lai"}|${drawGrid}|${LANG}`;
  if (rKey === _lastRenderKey) return;
  _lastRenderKey = rKey;

  const geo = getOrthoGeo();
  renderOneGlobe("globeT2", "t2", geo);
  renderOneGlobe("globeIR", "ir", geo);
  renderOneGlobe("globeOLR", "olr", geo);
  renderOneGlobe("globeSST", "sst", geo);
  renderOneGlobe("globeOCN", "ocn", geo);
  renderOneGlobe("globeLAI", "lai", geo);
  document.getElementById("view-angles").textContent =
    `Center: lat ${GLOBE_VIEW.lat0>=0?"+":""}${GLOBE_VIEW.lat0.toFixed(1)}°, lon ${GLOBE_VIEW.lon0>=0?"+":""}${GLOBE_VIEW.lon0.toFixed(1)}°`;
}

function resetGlobeView() {
  GLOBE_VIEW.lat0 = 18.0;
  GLOBE_VIEW.manualOverride = false;
  GLOBE_VIEW.lon0 = GLOBE_VIEW.subsolarLon !== null ? GLOBE_VIEW.subsolarLon : 140.0;
}

function bindGlobeInteractions() {
  ["globeT2", "globeIR", "globeOLR", "globeSST", "globeOCN", "globeLAI"].forEach(id => {
    const cv = document.getElementById(id);
    if (!cv) return;
    cv.addEventListener("mousedown", e => { GLOBE_VIEW.dragging = true; GLOBE_VIEW.manualOverride = true; GLOBE_VIEW.lastX = e.clientX; GLOBE_VIEW.lastY = e.clientY; });
    cv.addEventListener("mousemove", e => {
      if (GLOBE_VIEW.dragging) {
        const dx = e.clientX - GLOBE_VIEW.lastX, dy = e.clientY - GLOBE_VIEW.lastY;
        GLOBE_VIEW.lastX = e.clientX; GLOBE_VIEW.lastY = e.clientY;
        GLOBE_VIEW.lon0 = ((GLOBE_VIEW.lon0 - dx * 0.55 + 180) % 360) - 180;
        GLOBE_VIEW.lat0 = Math.max(-80, Math.min(80, GLOBE_VIEW.lat0 + dy * 0.55));
      }
      if (!MON.raster) return;
      const rect = cv.getBoundingClientRect();
      const X = ((e.clientX - rect.left) / rect.width * 300 - 150) / 132;
      const Y = (150 - (e.clientY - rect.top) / rect.height * 300) / 132;
      if (X*X + Y*Y > 1) return;
      const Z = Math.sqrt(Math.max(0, 1 - X*X - Y*Y));
      const lat0 = GLOBE_VIEW.lat0 * Math.PI / 180;
      const lat = Math.asin(Math.max(-1, Math.min(1, Z*Math.sin(lat0) + Y*Math.cos(lat0)))) * 180 / Math.PI;
      let lon = GLOBE_VIEW.lon0 + Math.atan2(X, Z*Math.cos(lat0) - Y*Math.sin(lat0)) * 180 / Math.PI;
      lon = ((lon + 180) % 360 + 360) % 360 - 180;
      const rH = MON.raster.height || 360;
      const rW = MON.raster.width || 720;
      const row = Math.max(0, Math.min(rH - 1, Math.floor((90.0 - lat) * (rH / 180.0))));
      const col = Math.max(0, Math.min(rW - 1, Math.floor((lon + 180.0) * (rW / 360.0))));
      const idx = row * rW + col;
      const t2v = -45 + (MON.rawT2[idx]/255) * 85;
      const irv = MON.rawIR ? (110 + (MON.rawIR[idx]/255) * 210) : 240;
      const precipv = (MON.rawOLR[idx]/255) * 25.0;
      const sstv = MON.rawSST ? (-2 + (MON.rawSST[idx]/255) * 34) : t2v;
      const spdMax = (MON.raster.ocn_spd_range && MON.raster.ocn_spd_range[1]) || 1.2;
      const spdv = MON.rawOcnSpd ? ((MON.rawOcnSpd[idx]/255) * spdMax) : 0.0;
      const spdLbl = MON.raster.ocn_is_wind ? "u₁₀" : "|u_ocn|";
      const r3 = MON.raster.lai_range || [0.0, 7.0];
      const v3 = r3[0] + (MON.rawLAI[idx]/255) * (r3[1] - r3[0]);
      const lbl3 = MON.esmLabel || "LAI [m² m⁻²]";
      const stype = MON.rawSurf[idx] === 0 ? (LANG==="ja"?"海洋":"Ocean") : (MON.rawSurf[idx] === 2 ? (LANG==="ja"?"氷床":"IceSheet") : (LANG==="ja"?"陸域":"Land"));
      const ptLbl = LANG === "ja" ? "地点" : "Point";
      const sstTxt = MON.rawSurf[idx] === 0 ? ` | SST: ${sstv.toFixed(1)}°C` : "";
      document.getElementById("hover-coords").textContent =
        `${ptLbl} (${lat.toFixed(1)}°, ${lon.toFixed(1)}°, ${stype}) — T₂: ${t2v.toFixed(1)}°C | OLR: ${irv.toFixed(0)} W/m² | P: ${precipv.toFixed(1)} mm/d${sstTxt} | ${spdLbl}: ${spdv.toFixed(2)} m/s | ${lbl3}: ${v3.toFixed(2)}`;
    });
  });
  window.addEventListener("mouseup", () => { GLOBE_VIEW.dragging = false; GRID_VIEW.dragging = false; });
  const gcv = document.getElementById("gridCanvas");
  if (gcv) {
    gcv.style.cursor = "grab";
    gcv.addEventListener("mousedown", e => { GRID_VIEW.dragging = true; GRID_VIEW.lastX = e.clientX; GRID_VIEW.lastY = e.clientY; });
    gcv.addEventListener("mousemove", e => {
      if (!GRID_VIEW.dragging) return;
      const dx = e.clientX - GRID_VIEW.lastX, dy = e.clientY - GRID_VIEW.lastY;
      GRID_VIEW.lastX = e.clientX; GRID_VIEW.lastY = e.clientY;
      GRID_VIEW.lon0 = ((GRID_VIEW.lon0 - dx * 0.5 + 180) % 360) - 180;
      GRID_VIEW.lat0 = Math.max(-80, Math.min(80, GRID_VIEW.lat0 + dy * 0.5));
      drawGridCanvas();
    });
  }
}

/* --- OFFLINE MARKDOWN + RECURSIVE-DESCENT LATEX DOCUMENTATION VIEWER --- */
function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

const _TEX_SYMS = {
  alpha:"α", beta:"β", gamma:"γ", delta:"δ", epsilon:"ε", varepsilon:"ε", zeta:"ζ",
  eta:"η", theta:"θ", vartheta:"ϑ", iota:"ι", kappa:"κ", lambda:"λ", mu:"μ", nu:"ν",
  xi:"ξ", pi:"π", varpi:"ϖ", rho:"ρ", varrho:"ϱ", sigma:"σ", varsigma:"ς", tau:"τ",
  upsilon:"υ", phi:"φ", varphi:"φ", chi:"χ", psi:"ψ", omega:"ω",
  Gamma:"Γ", Delta:"Δ", Theta:"Θ", Lambda:"Λ", Xi:"Ξ", Pi:"Π", Sigma:"Σ",
  Upsilon:"Υ", Phi:"Φ", Psi:"Ψ", Omega:"Ω",
  nabla:"∇", partial:"∂", infty:"∞", ell:"ℓ", hbar:"ℏ", Re:"ℜ", Im:"ℑ",
  cdot:"·", cdotp:"·", times:"×", div:"÷", pm:"±", mp:"∓", ast:"∗", star:"⋆", circ:"°", bullet:"•",
  le:"≤", leq:"≤", ge:"≥", geq:"≥", ll:"≪", gg:"≫", ne:"≠", neq:"≠", approx:"≈",
  simeq:"≃", sim:"∼", equiv:"≡", propto:"∝", cong:"≅", doteq:"≐",
  to:"→", rightarrow:"→", leftarrow:"←", leftrightarrow:"↔", Rightarrow:"⇒", Leftarrow:"⇐",
  Leftrightarrow:"⇔", implies:"⇒", iff:"⇔", mapsto:"↦", uparrow:"↑", downarrow:"↓",
  in:"∈", notin:"∉", subset:"⊂", subseteq:"⊆", supset:"⊃", cup:"∪", cap:"∩",
  forall:"∀", exists:"∃", neg:"¬", land:"∧", lor:"∨", empty:"∅", varnothing:"∅",
  sum:"∑", prod:"∏", int:"∫", oint:"∮",
  dots:"…", ldots:"…", cdots:"⋯", vdots:"⋮", ddots:"⋱",
  langle:"⟨", rangle:"⟩", lfloor:"⌊", rfloor:"⌋", lceil:"⌈", rceil:"⌉",
  vert:"|", mid:"∣", parallel:"∥", backslash:"\\", prime:"′", dagger:"†",
  oplus:"⊕", otimes:"⊗", top:"⊤", bot:"⊥", perp:"⊥", angle:"∠",
  quad:"&emsp;", qquad:"&emsp;&emsp;", enspace:"&ensp;", thinspace:"&thinsp;"
};
const _TEX_MATH_ITALIC_GREEK = {
  alpha:"𝛼", beta:"𝛽", gamma:"𝛾", delta:"𝛿", epsilon:"𝜀", varepsilon:"𝜀", zeta:"𝜁",
  eta:"𝜂", theta:"𝜃", vartheta:"𝜗", iota:"𝜄", kappa:"𝜅", lambda:"𝜆", mu:"𝜇", nu:"𝜈",
  xi:"𝜉", pi:"𝜋", varpi:"𝜛", rho:"𝜌", varrho:"𝜚", sigma:"𝜎", varsigma:"𝜍", tau:"𝜏",
  upsilon:"𝜐", phi:"𝜙", varphi:"𝜑", chi:"𝜒", psi:"𝜓", omega:"𝜔", ell:"ℓ"
};
function _toMathItalicChar(ch) {
  const c = ch.charCodeAt(0);
  if (c >= 65 && c <= 90) return String.fromCodePoint(0x1D434 + (c - 65));
  if (c === 104) return "\u210E"; // Planck italic h
  if (c >= 97 && c <= 122) return String.fromCodePoint(0x1D44E + (c - 97));
  return escHtml(ch);
}
const _TEX_FUNCS = new Set([
  "exp","ln","log","sin","cos","tan","cot","sec","csc","arcsin","arccos","arctan",
  "sinh","cosh","tanh","max","min","sup","inf","lim","arg","det","dim","ker",
  "sgn","mod","bmod","gcd","erf","erfc","atan2","expm1","log1p","clamp","clip"
]);

function _parseLatex(src, upright = false) {
  let i = 0;
  const n = src.length;
  const skipWs = () => { while (i < n && /\s/.test(src[i])) i++; };

  function readGroup() {
    skipWs();
    if (i >= n) return "";
    if (src[i] === "{") {
      i++;
      const inner = parseSeq("}");
      if (i < n && src[i] === "}") i++;
      return inner;
    }
    return parseAtom();
  }

  function readRawGroup() {
    skipWs();
    if (i >= n) return "";
    if (src[i] === "{") {
      let depth = 0, start = i + 1;
      for (; i < n; i++) {
        if (src[i] === "\\" && i + 1 < n) { i++; continue; }
        if (src[i] === "{") depth++;
        else if (src[i] === "}") {
          depth--;
          if (depth === 0) { const s = src.slice(start, i); i++; return s; }
        }
      }
      return src.slice(start);
    }
    return src[i++];
  }

  function readOptBracket() {
    skipWs();
    if (i < n && src[i] === "[") {
      i++;
      const start = i;
      let depth = 1;
      while (i < n && depth > 0) {
        if (src[i] === "[") depth++;
        else if (src[i] === "]") depth--;
        if (depth > 0) i++;
      }
      const content = src.slice(start, i);
      if (i < n && src[i] === "]") i++;
      return content;
    }
    return null;
  }

  function parseEnv(envName) {
    const endTag = `\\end{${envName}}`;
    const endIdx = src.indexOf(endTag, i);
    const body = endIdx >= 0 ? src.slice(i, endIdx) : src.slice(i);
    i = endIdx >= 0 ? endIdx + endTag.length : n;
    const rows = body.split(/\\\\/).map(r => r.trim()).filter(r => r.length > 0);
    if (envName === "cases") {
      const trs = rows.map(r => {
        const cols = r.split("&").map(c => _parseLatex(c.trim()));
        return `<tr><td class="al-l">${cols[0] || ""}</td><td class="al-l" style="padding-left:14px !important;">${cols.slice(1).join(" ")}</td></tr>`;
      }).join("");
      return `<span class="math-cases-wrap"><span class="math-cases-brace">{</span><table class="math-aligned">${trs}</table></span>`;
    }
    const trs = rows.map(r => {
      const cols = r.split("&").map(c => _parseLatex(c.trim()));
      if (cols.length === 1) return `<tr><td colspan="2" class="al-l">${cols[0]}</td></tr>`;
      return `<tr>${cols.map((c, ci) => `<td class="${ci % 2 === 0 ? "al-r" : "al-l"}">${c}</td>`).join("")}</tr>`;
    }).join("");
    return `<table class="math-aligned">${trs}</table>`;
  }

  function parseAtom() {
    skipWs();
    if (i >= n) return "";
    const ch = src[i];
    if (ch === "{") {
      i++;
      const inner = parseSeq("}");
      if (i < n && src[i] === "}") i++;
      return inner;
    }
    if (ch === "\\") {
      i++;
      if (i >= n) return "";
      const c1 = src[i];
      if (!/[A-Za-z]/.test(c1)) {
        i++;
        if (c1 === "," || c1 === ";") return "&thinsp;";
        if (c1 === ":") return "&nbsp;";
        if (c1 === "!") return "";
        if (c1 === " ") return "&nbsp;";
        if (c1 === "\\") return "<br/>";
        if (c1 === "|") return "∥";
        if (c1 === "{" || c1 === "}" || c1 === "%" || c1 === "_" || c1 === "$" || c1 === "#") return escHtml(c1);
        return escHtml(c1);
      }
      const start = i;
      while (i < n && /[A-Za-z]/.test(src[i])) i++;
      const cmd = src.slice(start, i);

      if (cmd === "frac" || cmd === "dfrac" || cmd === "tfrac") {
        const num = readGroup();
        const den = readGroup();
        return `<span class="math-frac"><span>${num}</span><span>${den}</span></span>`;
      }
      if (cmd === "sqrt") {
        const opt = readOptBracket();
        const arg = readGroup();
        const idxHtml = opt ? `<sup style="font-size:0.7em;margin-right:-2px;">${_parseLatex(opt)}</sup>` : "";
        return `${idxHtml}<span class="math-sqrt"><span class="math-sqrt-rad">√</span><span class="math-sqrt-body">${arg}</span></span>`;
      }
      if (cmd === "mathrm" || cmd === "text" || cmd === "textrm" || cmd === "operatorname" || cmd === "rm") {
        if (cmd === "rm") return "";
        const raw = readRawGroup();
        if (cmd === "text" || cmd === "textrm") return `<span class="math-rm">${escHtml(raw)}</span>`;
        return `<span class="math-rm">${_parseLatex(raw, true)}</span>`;
      }
      if (cmd === "mathbf" || cmd === "boldsymbol" || cmd === "bm") {
        const raw = readRawGroup();
        return `<span class="math-bf">${_parseLatex(raw, true)}</span>`;
      }
      if (cmd === "mathcal" || cmd === "mathbb" || cmd === "mathit" || cmd === "mathsf") {
        return `<span class="math-rm">${readGroup()}</span>`;
      }
      if (cmd === "overline" || cmd === "bar") {
        return `<span class="math-over">${readGroup()}</span>`;
      }
      if (cmd === "hat" || cmd === "widehat") {
        return `${readGroup()}\u0302`;
      }
      if (cmd === "tilde" || cmd === "widetilde") {
        return `${readGroup()}\u0303`;
      }
      if (cmd === "vec") {
        return `<span class="math-bf">${readGroup()}</span>`;
      }
      if (cmd === "dot") return `${readGroup()}\u0307`;
      if (cmd === "ddot") return `${readGroup()}\u0308`;
      if (cmd === "underbrace" || cmd === "overbrace") {
        const body = readGroup();
        skipWs();
        let note = "";
        if (i < n && (src[i] === "_" || src[i] === "^")) {
          i++;
          note = readGroup();
        }
        return `<span class="math-under"><span>${body}</span><span>${note}</span></span>`;
      }
      if (cmd === "left" || cmd === "right" || cmd === "big" || cmd === "Big" || cmd === "bigg" || cmd === "Bigg" || cmd === "bigl" || cmd === "bigr" || cmd === "Bigl" || cmd === "Bigr") {
        skipWs();
        if (i < n && src[i] === ".") { i++; return ""; }
        if (i < n && src[i] === "\\") return parseAtom();
        if (i < n) return `<span class="math-rm">${escHtml(src[i++])}</span>`;
        return "";
      }
      if (cmd === "begin") {
        const env = readRawGroup().trim();
        return parseEnv(env);
      }
      if (cmd === "end") {
        readRawGroup();
        return "";
      }
      if (cmd === "tag") {
        readRawGroup();
        return "";
      }
      if (cmd === "displaystyle" || cmd === "textstyle" || cmd === "limits" || cmd === "nolimits" || cmd === "nonumber" || cmd === "notag") {
        return "";
      }
      if (_TEX_FUNCS.has(cmd)) {
        return `<span class="math-rm">${cmd}</span>`;
      }
      if (Object.prototype.hasOwnProperty.call(_TEX_SYMS, cmd)) {
        if (!upright && Object.prototype.hasOwnProperty.call(_TEX_MATH_ITALIC_GREEK, cmd)) {
          return _TEX_MATH_ITALIC_GREEK[cmd];
        }
        return `<span class="math-rm">${_TEX_SYMS[cmd]}</span>`;
      }
      return `<span class="math-rm">${escHtml(cmd)}</span>`;
    }
    i++;
    if (ch === "&") return "&emsp;";
    if (/[0-9.,:;()[\]=+\-/*<>|!]/.test(ch)) return `<span class="math-rm">${escHtml(ch)}</span>`;
    if (!upright && /[A-Za-z]/.test(ch)) return _toMathItalicChar(ch);
    return escHtml(ch);
  }

  function parseSeq(stopChar) {
    let out = "";
    while (i < n) {
      if (stopChar && src[i] === stopChar) break;
      if (/\s/.test(src[i])) {
        while (i < n && /\s/.test(src[i])) i++;
        if (out && !out.endsWith(" ")) out += " ";
        continue;
      }
      if (src[i] === "^" || src[i] === "_") {
        const tag = src[i++] === "^" ? "sup" : "sub";
        out += `<${tag}>${readGroup()}</${tag}>`;
        continue;
      }
      let atom = parseAtom();
      let sup = null, sub = null;
      while (true) {
        const saved = i;
        skipWs();
        if (i < n && src[i] === "^" && sup === null) { i++; sup = readGroup(); }
        else if (i < n && src[i] === "_" && sub === null) { i++; sub = readGroup(); }
        else { i = saved; break; }
      }
      if (sub !== null) atom += `<sub>${sub}</sub>`;
      if (sup !== null) atom += `<sup>${sup}</sup>`;
      out += atom;
    }
    return out;
  }

  return parseSeq(null);
}

function _renderKatexOrFallback(tex, displayMode) {
  const clean = String(tex || "").trim();
  if (typeof window !== "undefined" && window.katex && typeof window.katex.renderToString === "function") {
    try {
      return window.katex.renderToString(clean, {
        displayMode: Boolean(displayMode),
        throwOnError: false,
        strict: false,
        trust: false,
      });
    } catch (_) { /* fall through to offline renderer */ }
  }
  return _parseLatex(clean);
}

function renderLatexPretty(tex) {
  return _renderKatexOrFallback(tex, false);
}

function renderMathBlock(tex) {
  let raw = String(tex || "").trim();
  let tagText = "";
  raw = raw.replace(/\\tag\{([^{}]+)\}/g, (_, t) => { tagText = t.trim(); return ""; }).trim();
  const eqHtml = _renderKatexOrFallback(raw, true);
  const tagHtml = tagText ? `<div class="math-tag">(${escHtml(tagText)})</div>` : "";
  return `<div class="math-block"><div class="math-block-eq">${eqHtml}</div>${tagHtml}</div>`;
}

function _resolveDocAssetUrl(href) {
  const clean = href.replace(/^\.\//, "");
  if (clean.startsWith("assets/")) return "/docs/" + clean;
  if (clean.startsWith("docs/assets/")) return "/" + clean;
  if (clean.startsWith("../docs/assets/")) return "/docs/assets/" + clean.slice("../docs/assets/".length);
  return href;
}

function renderInlineMd(line) {
  const slots = [];
  const stash = (html) => { const idx = slots.length; slots.push(html); return `@@S${idx}@@`; };

  // 1. Protect inline code spans first so `$` inside backticks is never parsed as math
  let s = line.replace(/`([^`]+)`/g, (_, code) => stash(`<code>${escHtml(code)}</code>`));
  // 2. Protect inline images ![alt](href)
  s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, href) => {
    const src = _resolveDocAssetUrl(href.trim());
    return stash(`<figure class="doc-fig"><img src="${escHtml(src)}" alt="${escHtml(alt)}" loading="lazy"/>${alt ? `<figcaption>${escHtml(alt)}</figcaption>` : ""}</figure>`);
  });
  // 3. Protect inline math $...$ and \(...\)
  s = s.replace(/\\\((.+?)\\\)/g, (_, m) => stash(`<span class="math-inline">${renderLatexPretty(m)}</span>`));
  s = s.replace(/(^|[^\\])\$([^$\n]+?)\$/g, (_, pre, m) => pre + stash(`<span class="math-inline">${renderLatexPretty(m)}</span>`));
  s = s.replace(/\\\$/g, "$");

  s = escHtml(s);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, txt, href) =>
    href.startsWith("#") ? `<a href="${href}" style="color:var(--accent)">${txt}</a>` : `<span style="color:var(--accent)">${txt}</span>`
  );
  // Restore stashed HTML (may be nested once)
  for (let p = 0; p < 2; p++) {
    s = s.replace(/@@S(\d+)@@/g, (_, idx) => slots[Number(idx)]);
  }
  return s;
}

function renderMarkdownWithMath(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  let html = "", inCode = false, inMath = false, mathBuf = [], inList = false, inTable = false, toc = [];
  const flushList = () => { if (inList) { html += "</ul>"; inList = false; } };
  const flushTable = () => { if (inTable) { html += "</tbody></table>"; inTable = false; } };

  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    const trimmed = ln.trim();
    if (trimmed.startsWith("```")) {
      flushList(); flushTable();
      if (!inCode) { html += "<pre><code>"; inCode = true; } else { html += "</code></pre>"; inCode = false; }
      continue;
    }
    if (inCode) { html += escHtml(ln) + "\n"; continue; }

    if (trimmed === "$$" || trimmed === "\\[" || trimmed === "\\]") {
      flushList(); flushTable();
      if (!inMath && trimmed !== "\\]") { inMath = true; mathBuf = []; }
      else { html += renderMathBlock(mathBuf.join("\n")); inMath = false; }
      continue;
    }
    if (inMath) { mathBuf.push(ln); continue; }
    if (trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4) {
      flushList(); flushTable();
      html += renderMathBlock(trimmed.slice(2, -2)); continue;
    }

    // Standalone image line -> full textbook figure card
    const imgM = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imgM) {
      flushList(); flushTable();
      const alt = imgM[1], src = _resolveDocAssetUrl(imgM[2].trim());
      html += `<figure class="doc-fig"><img src="${escHtml(src)}" alt="${escHtml(alt)}" loading="lazy"/>${alt ? `<figcaption>${escHtml(alt)}</figcaption>` : ""}</figure>`;
      continue;
    }

    const hm = ln.match(/^(#{1,4})\s+(.+)$/);
    if (hm) {
      flushList(); flushTable();
      const lvl = hm[1].length, rawTitle = hm[2].replace(/`([^`]+)`/g, "$1").replace(/\$([^$]+)\$/g, "$1");
      const id = "sec-" + toc.length;
      if (lvl === 2 || lvl === 3) toc.push({ id, lvl, text: rawTitle });
      html += `<h${lvl} id="${id}">${renderInlineMd(hm[2])}</h${lvl}>`;
      continue;
    }
    if (trimmed === "---") { flushList(); flushTable(); html += "<hr style='border:none;border-top:1px solid var(--border);margin:16px 0;'/>"; continue; }
    if (trimmed.startsWith("|") && ln.includes("|")) {
      flushList();
      const cells = trimmed.slice(1, -1).split("|").map(c => c.trim());
      if (cells.every(c => /^:?-{2,}:?$/.test(c))) continue;
      if (!inTable) {
        inTable = true;
        html += "<table><thead><tr>" + cells.map(c => `<th>${renderInlineMd(c)}</th>`).join("") + "</tr></thead><tbody>";
      } else {
        html += "<tr>" + cells.map(c => `<td>${renderInlineMd(c)}</td>`).join("") + "</tr>";
      }
      continue;
    } else { flushTable(); }

    if (trimmed.startsWith("> ")) { flushList(); html += `<blockquote>${renderInlineMd(trimmed.slice(2))}</blockquote>`; continue; }
    const lm = ln.match(/^\s*(?:[-*]|\d+\.)\s+(.+)$/);
    if (lm) {
      if (!inList) { html += "<ul style='padding-left:20px;margin:6px 0;'>"; inList = true; }
      html += `<li>${renderInlineMd(lm[1])}</li>`;
      continue;
    } else { flushList(); }

    if (trimmed === "") { html += "<div style='height:6px'></div>"; continue; }
    html += `<p style='margin:6px 0;'>${renderInlineMd(ln)}</p>`;
  }
  flushList(); flushTable();
  return { html, toc };
}

async function loadDoc(base) {
  CURRENT_DOC_BASE = base || "formulation";
  ["formulation","configurations","architecture","readme"].forEach(b => {
    const btn = document.getElementById("docbtn-" + b);
    if (btn) btn.className = b === CURRENT_DOC_BASE ? "primary" : "";
  });
  const docKey = `${CURRENT_DOC_BASE}_${LANG}`;
  const res = await fetch("/api/docs?name=" + encodeURIComponent(docKey));
  const data = await res.json();
  if (!data.ok) { document.getElementById("doc-viewer").innerHTML = `<p>${data.error}</p>`; return; }
  document.getElementById("doc-filename-badge").textContent = data.filename;
  const rendered = renderMarkdownWithMath(data.markdown);
  const viewer = document.getElementById("doc-viewer");
  viewer.innerHTML = rendered.html;
  viewer.scrollTop = 0;
  document.getElementById("doc-toc").innerHTML = rendered.toc.map(t =>
    `<a class="toc-link ${t.lvl===3?"h3":""}" onclick="document.getElementById('${t.id}')?.scrollIntoView({behavior:'smooth'})">${escHtml(t.text)}</a>`
  ).join("");
}

/* --- LIVE TIME SERIES CHART --- */
function drawTimeSeries(csv) {
  const cv = document.getElementById("tsCanvas");
  if (!cv) return;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const days = csv.sim_days || [];
  if (days.length < 2) {
    ctx.fillStyle = "#64748b"; ctx.font = "13px sans-serif";
    ctx.fillText(LANG === "ja" ? "日次統計 (daily_global.csv) 待機中..." : "Waiting for daily_global.csv...", 36, H/2); return;
  }
  const padL = 48, padR = 16, padT = 24, padB = 28, cw = W-padL-padR, ch = H-padT-padB;
  ctx.strokeStyle = "#263347"; ctx.strokeRect(padL, padT, cw, ch);

  const plotNorm = (arr, color, vmin, vmax) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = padL + (i / (arr.length - 1)) * cw;
      const y = padT + ch - Math.max(0, Math.min(1, (arr[i] - vmin) / (vmax - vmin))) * ch;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  plotNorm(csv.ts_k, "#f87171", 275, 295);       // Red: Ts 275..295 K
  plotNorm(csv.olr_wm2, "#fbbf24", 210, 260);    // Amber: OLR 210..260 W/m2
  plotNorm(csv.precip_mm_d, "#38bdf8", 1.0, 3.5);// Cyan: Precip 1..3.5 mm/d
  plotNorm(csv.lai, "#34d399", 0.0, 2.5);        // Green: LAI 0..2.5

  const li = days.length - 1;
  ctx.font = "11.5px monospace";
  ctx.fillStyle = "#f87171"; ctx.fillText(`Ts=${csv.ts_k[li].toFixed(2)}K [275-295]`, padL+6, 16);
  ctx.fillStyle = "#fbbf24"; ctx.fillText(`OLR=${csv.olr_wm2[li].toFixed(1)} [210-260]`, padL+185, 16);
  ctx.fillStyle = "#38bdf8"; ctx.fillText(`P=${csv.precip_mm_d[li].toFixed(2)}mm/d [1-3.5]`, padL+355, 16);
  ctx.fillStyle = "#34d399"; ctx.fillText(`LAI=${csv.lai[li].toFixed(2)} [0-2.5]`, padL+530, 16);
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(`Day ${days[0].toFixed(0)}`, padL, H - 8);
  ctx.fillText(`Day ${days[li].toFixed(0)}`, W - 82, H - 8);
}

setInterval(() => {
  const auto = document.getElementById("auto-poll")?.checked;
  const active = document.getElementById("tab-monitor")?.classList.contains("active");
  if (auto && active) fetchMonitorOnce();
}, 2000);

/* --- HEADER OTHELLO VORONOI 'A' ICON ANIMATION --- */
let OTHELLO_CELLS = null;
let othelloTargetInverted = false;
let othelloFlipStartMs = -1;
let othelloNextAutoMs = performance.now() + 3400;
let othelloDirty = true;

function triggerOthelloFlip() {
  if (!OTHELLO_CELLS) return;
  const now = performance.now();
  othelloTargetInverted = !othelloTargetInverted;
  othelloFlipStartMs = now;
  othelloNextAutoMs = now + (othelloTargetInverted ? 4200 : 5400);
  othelloDirty = true;
}

function drawOthelloIconFrame(nowMs) {
  const cv = document.getElementById("aoba-header-icon");
  if (!cv) return;

  if (OTHELLO_CELLS && othelloFlipStartMs < 0 && nowMs >= othelloNextAutoMs) {
    othelloTargetInverted = !othelloTargetInverted;
    othelloFlipStartMs = nowMs;
    othelloNextAutoMs = nowMs + (othelloTargetInverted ? 4200 : 5400);
    othelloDirty = true;
  }

  const waveDur = 1550;
  const tileFlipDur = 420;
  const totalDur = waveDur + tileFlipDur;
  const isFlipping = othelloFlipStartMs >= 0 && (nowMs - othelloFlipStartMs) <= totalDur;
  if (othelloFlipStartMs >= 0 && !isFlipping) {
    othelloFlipStartMs = -1;
    othelloDirty = true;
  }
  if (!othelloDirty && !isFlipping) return;
  othelloDirty = isFlipping;

  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const S = W / 512.0;
  ctx.clearRect(0, 0, W, H);

  // Rounded dark icon badge (portable across all Safari / Chrome / Firefox versions)
  const pad = 14 * S, bw = 484 * S, r = 110 * S;
  ctx.beginPath();
  ctx.moveTo(pad + r, pad);
  ctx.arcTo(pad + bw, pad, pad + bw, pad + bw, r);
  ctx.arcTo(pad + bw, pad + bw, pad, pad + bw, r);
  ctx.arcTo(pad, pad + bw, pad, pad, r);
  ctx.arcTo(pad, pad, pad + bw, pad, r);
  ctx.closePath();
  ctx.fillStyle = "#0b1322";
  ctx.fill();
  ctx.lineWidth = 3.0 * S;
  ctx.strokeStyle = "#1e293b";
  ctx.stroke();

  if (!OTHELLO_CELLS) {
    // Crisp fallback A before JSON arrives
    ctx.beginPath();
    ctx.arc(256 * S, 256 * S, 200 * S, 0, Math.PI * 2);
    ctx.fillStyle = "#1d4ed8";
    ctx.fill();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5 * S;
    ctx.stroke();
    return;
  }

  ctx.save();
  ctx.beginPath();
  ctx.arc(256 * S, 256 * S, 200 * S, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = "#0b1322";
  ctx.fillRect(0, 0, W, H);

  const bluePal = ["#2563eb", "#1d4ed8", "#1e3a8a"];
  const greenPal = ["#22c55e", "#16a34a", "#15803d"];
  const flipAxis = -0.5585; // -32 deg diagonal Othello flip axis

  for (let i = 0; i < OTHELLO_CELLS.length; i++) {
    const cell = OTHELLO_CELLS[i]; // [cx, cy, isA, shade, wave, pts]
    const cx = cell[0] * S, cy = cell[1] * S;
    const isA = cell[2] === 1;
    const shade = cell[3];
    const wave = cell[4];
    const pts = cell[5];

    let showInv = othelloTargetInverted;
    let flipScale = 1.0;
    let rimEdge = false;

    if (isFlipping) {
      const elapsed = nowMs - othelloFlipStartMs;
      const delay = (othelloTargetInverted ? wave : (1.0 - wave)) * waveDur;
      const u = (elapsed - delay) / tileFlipDur;
      if (u <= 0) {
        showInv = !othelloTargetInverted;
      } else if (u >= 1) {
        showInv = othelloTargetInverted;
      } else {
        const th = Math.PI * (0.5 - 0.5 * Math.cos(Math.PI * u));
        const sc = Math.cos(th);
        showInv = sc < 0 ? othelloTargetInverted : !othelloTargetInverted;
        flipScale = Math.max(0.08, Math.abs(sc));
        rimEdge = Math.abs(sc) < 0.18;
      }
    }

    const useGreen = showInv ? !isA : isA;
    ctx.fillStyle = rimEdge ? "#94a3b8" : (useGreen ? greenPal[shade] : bluePal[shade]);

    ctx.save();
    if (flipScale < 0.999) {
      ctx.translate(cx, cy);
      ctx.rotate(flipAxis);
      ctx.scale(flipScale, 1.0);
      ctx.rotate(-flipAxis);
      ctx.translate(-cx, -cy);
    }
    ctx.beginPath();
    ctx.moveTo(pts[0] * S, pts[1] * S);
    for (let k = 2; k < pts.length; k += 2) {
      ctx.lineTo(pts[k] * S, pts[k + 1] * S);
    }
    ctx.closePath();
    ctx.fill();
    ctx.lineWidth = 2.0 * S;
    ctx.strokeStyle = "#0b1322";
    ctx.lineJoin = "round";
    ctx.stroke();
    ctx.restore();
  }
  ctx.restore();

  ctx.beginPath();
  ctx.arc(256 * S, 256 * S, 200 * S, 0, Math.PI * 2);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.4 * S;
  ctx.stroke();
}

async function initOthelloIcon() {
  try {
    const res = await fetch("/assets/aoba_icon.json");
    const data = await res.json();
    if (data.ok && Array.isArray(data.cells)) {
      OTHELLO_CELLS = data.cells;
      othelloDirty = true;
    }
  } catch (e) { /* keep static fallback */ }
}

function animateHeaderIcon(now) {
  drawOthelloIconFrame(now);
  requestAnimationFrame(animateHeaderIcon);
}

fetchInitial();
initOthelloIcon();
bindGlobeInteractions();
requestAnimationFrame(animateGlobes);
requestAnimationFrame(animateHeaderIcon);
</script>
</body>
</html>
"""


_SCVT_PREVIEW_MEM: dict[int, dict[str, Any]] = {}
_SCVT_NPZ_PATH = _REPO_ROOT / "config" / "scvt_preview_meshes.npz"
_OTHELLO_ICON_MEM: dict[str, Any] | None = None


def get_othello_icon_payload() -> dict[str, Any]:
    """Return compact SCVT G3 Voronoi 'A' icon cells for live Othello header animation."""
    global _OTHELLO_ICON_MEM
    if _OTHELLO_ICON_MEM is not None:
        return _OTHELLO_ICON_MEM
    if str(_REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from build_aoba_icon import build_voronoi_a_geometry  # type: ignore

    cells = build_voronoi_a_geometry(glevel=3)
    compact = [
        [c["cx"], c["cy"], 1 if c["is_a"] else 0, c["shade"], c["wave"],
         [coord for pt in c["pts"] for coord in pt]]
        for c in cells
    ]
    _OTHELLO_ICON_MEM = {"ok": True, "cells": compact}
    return _OTHELLO_ICON_MEM


def get_scvt_preview_payload(glevel: int) -> dict[str, Any]:
    """Return real precomputed SCVT Voronoi polygon mesh for `glevel` (`0..6`), with dynamic fallback."""
    g = max(0, min(6, int(glevel)))
    if g in _SCVT_PREVIEW_MEM:
        return _SCVT_PREVIEW_MEM[g]

    if _SCVT_NPZ_PATH.exists():
        with np.load(_SCVT_NPZ_PATH) as zf:
            if f"g{g}_xyz_cell" in zf:
                raw = {
                    "xyz_cell": zf[f"g{g}_xyz_cell"],
                    "xyz_vert": zf[f"g{g}_xyz_vert"],
                    "verts_on_cell": zf[f"g{g}_verts_on_cell"],
                    "n_verts_on_cell": zf[f"g{g}_n_verts_on_cell"],
                    "verts_on_edge": zf[f"g{g}_verts_on_edge"],
                    "surf_type": zf[f"g{g}_surf_type"],
                }
            else:
                raw = None
    else:
        raw = None

    if raw is None:
        if str(_REPO_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from build_scvt_preview_cache import (  # type: ignore
            _extract_voronoi_preview_level,
            load_earth2014_mask_tree,
        )

        if not hasattr(get_scvt_preview_payload, "_earth_mask"):
            get_scvt_preview_payload._earth_mask = load_earth2014_mask_tree(stride=3)  # type: ignore[attr-defined]
        e_tree, e_ocn, e_ice = get_scvt_preview_payload._earth_mask  # type: ignore[attr-defined]
        d = _extract_voronoi_preview_level(
            g, earth_tree=e_tree, earth_is_ocn=e_ocn, earth_is_ice=e_ice
        )
        raw = {
            "xyz_cell": d[f"g{g}_xyz_cell"],
            "xyz_vert": d[f"g{g}_xyz_vert"],
            "verts_on_cell": d[f"g{g}_verts_on_cell"],
            "n_verts_on_cell": d[f"g{g}_n_verts_on_cell"],
            "verts_on_edge": d[f"g{g}_verts_on_edge"],
            "surf_type": d[f"g{g}_surf_type"],
        }

    payload = {
        "ok": True,
        "glevel": g,
        "n_cells": int(raw["xyz_cell"].shape[0]),
        "n_verts": int(raw["xyz_vert"].shape[0]),
        "n_edges": int(raw["verts_on_edge"].shape[0]),
        "xyz_cell_b64": base64.b64encode(np.ascontiguousarray(raw["xyz_cell"], dtype=np.int16).tobytes()).decode("ascii"),
        "xyz_vert_b64": base64.b64encode(np.ascontiguousarray(raw["xyz_vert"], dtype=np.int16).tobytes()).decode("ascii"),
        "verts_on_cell_b64": base64.b64encode(np.ascontiguousarray(raw["verts_on_cell"], dtype=np.int32).tobytes()).decode("ascii"),
        "n_verts_on_cell_b64": base64.b64encode(np.ascontiguousarray(raw["n_verts_on_cell"], dtype=np.uint8).tobytes()).decode("ascii"),
        "verts_on_edge_b64": base64.b64encode(np.ascontiguousarray(raw["verts_on_edge"], dtype=np.int32).tobytes()).decode("ascii"),
        "surf_type_b64": base64.b64encode(np.ascontiguousarray(raw["surf_type"], dtype=np.uint8).tobytes()).decode("ascii"),
    }
    _SCVT_PREVIEW_MEM[g] = payload
    return payload


_DOC_FILES: dict[str, Path] = {
    "formulation_ja": _REPO_ROOT / "docs" / "formulation_ja.md",
    "formulation_en": _REPO_ROOT / "docs" / "formulation.md",
    "configurations_ja": _REPO_ROOT / "docs" / "configurations_ja.md",
    "configurations_en": _REPO_ROOT / "docs" / "configurations.md",
    "architecture_ja": _REPO_ROOT / "docs" / "architecture_ja.md",
    "architecture_en": _REPO_ROOT / "docs" / "architecture.md",
    "readme_ja": _REPO_ROOT / "README_ja.md",
    "readme_en": _REPO_ROOT / "README.md",
}


class ConfigGUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the full AOBA Config + Run + Monitor + 3-Globe Preview UI."""

    server_version = "AOBAStudio/2.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(body.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            raw = _HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/api/state":
            defaults = default_config_dict()
            cur = self.server.current_config  # type: ignore[attr-defined]
            self._send_json({
                "defaults": defaults,
                "config": cur,
                "section_meta": SECTION_META,
                "param_docs": PARAM_DOCS,
                "toml_path": str(self.server.toml_path),  # type: ignore[attr-defined]
                "param_count": sum(len(v) for v in defaults.values()),
                "hardware": detect_hardware_profile().to_dict(),
            })
            return

        if parsed.path == "/api/runs/discover":
            rm: RunManager = self.server.run_manager  # type: ignore[attr-defined]
            self._send_json({
                "ok": True,
                "discovered_dirs": _discover_output_dirs(rm.last_watch_dir),
                "active_processes": _discover_active_processes(),
                "managed_run": rm.get_managed_status(),
            })
            return

        if parsed.path == "/api/monitor/snapshot":
            qs = parse_qs(parsed.query)
            out_dir = qs.get("out_dir", [""])[0]
            source = qs.get("source", ["auto"])[0]
            esm_field = qs.get("esm_field", ["lai"])[0]
            try:
                frame = int(qs.get("frame", ["-1"])[0])
            except ValueError:
                frame = -1
            rm = self.server.run_manager  # type: ignore[attr-defined]
            self._send_json(rm.build_monitor_payload(out_dir, source=source, frame_req=frame, esm_field=esm_field))
            return

        if parsed.path == "/api/co2_catalog":
            self._send_json(get_co2_catalog_summary(prefer_cache=True))
            return

        if parsed.path == "/api/ozone_status":
            self._send_json(get_ozone_status_summary())
            return

        if parsed.path == "/api/scvt_mesh":
            qs = parse_qs(parsed.query)
            try:
                g = int(qs.get("glevel", ["3"])[0])
            except ValueError:
                g = 3
            self._send_json(get_scvt_preview_payload(g))
            return

        if parsed.path == "/api/hardware/tune":
            qs = parse_qs(parsed.query)
            glevel = int(qs.get("glevel", ["3"])[0])
            nz = int(qs.get("nz", ["32"])[0])
            crm_nx = int(qs.get("crm_nx", ["8"])[0])
            crm_ny = int(qs.get("crm_ny", ["1"])[0])
            hw = detect_hardware_profile()
            chunks = compute_optimal_chunks(glevel, nz, crm_nx, crm_ny, hw=hw)
            self._send_json({"ok": True, "hardware": hw.to_dict(), "chunks": chunks})
            return

        if parsed.path == "/assets/aoba_icon.png":
            icon_p = _REPO_ROOT / "docs" / "assets" / "aoba_icon.png"
            if not icon_p.exists():
                self._send_json({"error": "Icon not found"}, status=HTTPStatus.NOT_FOUND)
                return
            raw = icon_p.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/assets/aoba_icon.svg":
            svg_p = _REPO_ROOT / "docs" / "assets" / "aoba_icon.svg"
            if not svg_p.exists():
                self._send_json({"error": "SVG icon not found"}, status=HTTPStatus.NOT_FOUND)
                return
            raw = svg_p.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path.startswith("/docs/assets/"):
            fname = Path(parsed.path).name
            asset_p = (_REPO_ROOT / "docs" / "assets" / fname).resolve()
            assets_root = (_REPO_ROOT / "docs" / "assets").resolve()
            if not str(asset_p).startswith(str(assets_root)) or not asset_p.is_file():
                self._send_json({"error": "Asset not found"}, status=HTTPStatus.NOT_FOUND)
                return
            ext = asset_p.suffix.lower()
            ctype = "image/svg+xml; charset=utf-8" if ext == ".svg" else ("image/png" if ext == ".png" else "application/octet-stream")
            raw = asset_p.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/assets/aoba_icon.json":
            self._send_json(get_othello_icon_payload())
            return

        if parsed.path == "/api/docs":
            qs = parse_qs(parsed.query)
            key = qs.get("name", ["formulation_ja"])[0]
            doc_p = _DOC_FILES.get(key)
            if doc_p is None or not doc_p.exists():
                self._send_json({"ok": False, "error": f"Unknown or missing document: {key}"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({
                "ok": True,
                "key": key,
                "filename": str(doc_p.relative_to(_REPO_ROOT)),
                "markdown": doc_p.read_text(encoding="utf-8"),
            })
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except json.JSONDecodeError as exc:
            self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/toml":
            cfg = sanitize_config_dict(payload.get("config", {}))
            self._send_json({
                "ok": True,
                "toml_text": dump_aoba_toml(cfg),
                "modified_count": count_modified_keys(cfg),
            })
            return

        if parsed.path == "/api/save":
            cfg = sanitize_config_dict(payload.get("config", {}))
            save_path = Path(str(payload.get("save_path", self.server.toml_path))).expanduser()  # type: ignore[attr-defined]
            if not save_path.is_absolute():
                save_path = _REPO_ROOT / save_path
            saved = save_aoba_toml(cfg, save_path)
            self.server.current_config = cfg  # type: ignore[attr-defined]
            self.server.toml_path = saved  # type: ignore[attr-defined]
            self._send_json({
                "ok": True,
                "path": str(saved),
                "toml_text": dump_aoba_toml(cfg),
                "modified_count": count_modified_keys(cfg),
            })
            return

        if parsed.path == "/api/load":
            load_path = Path(str(payload.get("load_path", ""))).expanduser()
            if not load_path.is_absolute():
                load_path = _REPO_ROOT / load_path
            if not load_path.exists():
                self._send_json({"ok": False, "error": f"File not found: {load_path}"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                cfg = load_aoba_toml(load_path)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self.server.current_config = cfg  # type: ignore[attr-defined]
            self.server.toml_path = load_path  # type: ignore[attr-defined]
            self._send_json({"ok": True, "path": str(load_path), "config": cfg, "toml_text": dump_aoba_toml(cfg)})
            return

        if parsed.path == "/api/reset":
            cfg = default_config_dict()
            self.server.current_config = cfg  # type: ignore[attr-defined]
            self._send_json({"ok": True, "config": cfg, "toml_text": dump_aoba_toml(cfg)})
            return

        if parsed.path == "/api/co2_fetch":
            try:
                inc_rcmip = bool(payload.get("include_rcmip", False))
                inc_paleo = bool(payload.get("include_paleo", False))
                st = fetch_latest_co2_sources(include_rcmip=inc_rcmip, include_paleo=inc_paleo, write_bundled=False)
                self._send_json({"ok": True, "status": st, "catalog": get_co2_catalog_summary(prefer_cache=True)})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path == "/api/ozone_fetch":
            try:
                era = str(payload.get("mip_era", "CMIP6"))
                res = fetch_input4mips_ozone_climatology(mip_era=era)
                self._send_json(res)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path == "/api/run/start":
            rm: RunManager = self.server.run_manager  # type: ignore[attr-defined]
            cfg = sanitize_config_dict(payload.get("config", self.server.current_config))  # type: ignore[attr-defined]
            self.server.current_config = cfg  # type: ignore[attr-defined]
            device = str(payload.get("device", "auto"))
            no_resume = bool(payload.get("no_resume", False))
            res = rm.start_run(cfg, device=device, no_resume=no_resume)
            self._send_json(res, status=HTTPStatus.OK if res.get("ok") else HTTPStatus.CONFLICT)
            return

        if parsed.path == "/api/run/stop":
            rm = self.server.run_manager  # type: ignore[attr-defined]
            res = rm.stop_managed_run()
            self._send_json(res, status=HTTPStatus.OK if res.get("ok") else HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def create_gui_server(
    toml_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create and bind an interactive AOBA Studio HTTP server."""
    target = Path(toml_path).expanduser() if toml_path else default_toml_path()
    if not target.is_absolute():
        target = _REPO_ROOT / target
    cfg = load_aoba_toml(target) if target.exists() else default_config_dict()
    server = ThreadingHTTPServer((host, port), ConfigGUIHandler)
    server.toml_path = target  # type: ignore[attr-defined]
    server.current_config = cfg  # type: ignore[attr-defined]
    server.preview_engine = CPUGlobePreviewEngine()  # type: ignore[attr-defined]
    server.run_manager = RunManager(server.preview_engine)  # type: ignore[attr-defined]
    init_out_dir = str((cfg.get("output") or {}).get("out_dir", "")).strip()
    if init_out_dir:
        server.run_manager.last_watch_dir = str(Path(init_out_dir).expanduser())  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch AOBA Studio Web GUI (Config, Run, Monitor, Live 3-Globe Preview).")
    parser.add_argument("--toml", type=str, default=str(default_toml_path()), help="TOML configuration file path")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open a web browser")
    args = parser.parse_args()

    server = create_gui_server(args.toml, host=args.host, port=args.port)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"AOBA Studio running at {url}", flush=True)
    print(f"Loaded TOML: {server.toml_path}", flush=True)  # type: ignore[attr-defined]
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AOBA Studio.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
