#!/usr/bin/env python3
"""Fetch compact multi-model/observational climatology benchmarks (mean + std + 95% CI) for AOBA validation.

Downloads lightweight climatological means and confidence intervals into `data/validation/`:

1. `cmip6_picontrol_clim.zarr` + `cmip6_picontrol_summary.json`:
   Pangeo CMIP6 1850 `piControl` multi-model ensemble (up to 12 diverse ESMs, last 30 yr each,
   unified 2.0-deg grid + standard pressure/depth levels; Annual/DJF/JJA mean, inter-model std,
   95% CI of ensemble mean, and 5th-95th percentile spread).
2. `era5_clim.zarr` + `era5_summary.json`:
   Google Research ARCO-ERA5 (1.5-deg 240x121 conservative grid) for both Early baseline (1959-1978)
   and Modern WMO normal (1991-2020): Annual/DJF/JJA mean, interannual std, and 95% CI.
3. `woa23_ocean_clim.zarr` + `woa23_summary.json`:
   NOAA World Ocean Atlas 2023 1.0-deg 3D ocean temperature & salinity decadal climatology
   (analyzed mean, statistical mean, std, and 95% CI across 102 standard depth levels 0-5500 m).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
import requests
from scipy import stats
import xarray as xr

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = REPO_ROOT / "data" / "validation"

TARGET_LAT = np.linspace(-89.0, 89.0, 90, dtype=np.float32)
TARGET_LON = np.linspace(0.0, 358.0, 180, dtype=np.float32)
TARGET_PLEV_PA = np.array(
    [
        100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0, 40000.0,
        30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 7000.0, 5000.0,
        3000.0, 2000.0, 1000.0, 500.0, 100.0,
    ],
    dtype=np.float32,
)
TARGET_OCN_DEPTH_M = np.array(
    [
        25.0, 75.0, 130.0, 197.5, 282.5, 390.0, 525.0, 695.0, 910.0,
        1180.0, 1510.0, 1900.0, 2350.0, 2860.0, 3430.0, 4060.0, 4755.0, 5560.0,
    ],
    dtype=np.float32,
)

CMIP6_MODELS = [
    "MIROC6",
    "CESM2",
    "GFDL-CM4",
    "MRI-ESM2-0",
    "EC-Earth3",
    "MPI-ESM1-2-HR",
    "NorESM2-MM",
    "UKESM1-0-LL",
    "CNRM-CM6-1",
    "IPSL-CM6A-LR",
    "ACCESS-ESM1-5",
    "CanESM5",
]


def _save_zarr_v3(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=False)


def _area_weights(lat_deg: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat_deg)).clip(min=0.0)


def _global_mean_2d(arr: np.ndarray, lat_deg: np.ndarray) -> float:
    """Cosine-latitude area-weighted global mean of `[nlat, nlon]` array ignoring NaNs."""
    w = _area_weights(lat_deg)[:, None] * np.ones((1, arr.shape[1]), dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(arr[mask] * w[mask]) / np.sum(w[mask]))


def _t_ci95_halfwidth(std: np.ndarray | float, n: int) -> np.ndarray | float:
    if n <= 1:
        return np.nan if isinstance(std, float) else np.full_like(std, np.nan)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return tcrit * std / math.sqrt(float(n))


def _standardize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """Rename common CMIP/ERA5 lat-lon coords to (`lat`, `lon` in [0, 360)) and sort ascending."""
    rename = {}
    for c in da.coords:
        if c in ("latitude", "nav_lat", "y") and "lat" not in da.dims and c in da.dims:
            rename[c] = "lat"
        if c in ("longitude", "nav_lon", "x") and "lon" not in da.dims and c in da.dims:
            rename[c] = "lon"
    if rename:
        da = da.rename(rename)
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Non-rectilinear grid dims={da.dims}")
    lon = np.mod(da["lon"].values.astype(np.float64), 360.0)
    da = da.assign_coords(lon=lon).sortby("lon").sortby("lat")
    return da


def _regrid_to_2deg(da: xr.DataArray) -> xr.DataArray:
    da = _standardize_lat_lon(da)
    # Periodic longitude wrap for clean interpolation at 358 deg
    da_wrap = xr.concat([da, da.isel(lon=0).assign_coords(lon=float(da["lon"].values[0]) + 360.0)], dim="lon")
    out = da_wrap.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")
    return out


# ---------------------------------------------------------------------------
# 1. NOAA WOA23 1-Degree 3D Ocean Temperature & Salinity Climatology
# ---------------------------------------------------------------------------
def fetch_woa23(out_dir: Path) -> None:
    print("\n=== [1/3] Fetching NOAA WOA23 1-deg Ocean Climatology (Mean + Std + 95% CI) ===", flush=True)
    urls = {
        "thetao": (
            "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/1.00/woa23_decav_t00_01.nc",
            ("t_an", "t_mn", "t_sd", "t_se"),
            "degC",
        ),
        "so": (
            "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/salinity/netcdf/decav/1.00/woa23_decav_s00_01.nc",
            ("s_an", "s_mn", "s_sd", "s_se"),
            "psu",
        ),
    }

    data_vars: dict[str, tuple] = {}
    summary: dict[str, dict[str, float]] = {}
    depth_coord = lat_coord = lon_coord = None

    with tempfile.TemporaryDirectory() as tmpd:
        for short, (url, (v_an, v_mn, v_sd, v_se), unit) in urls.items():
            local_nc = Path(tmpd) / url.split("/")[-1]
            print(f"  Downloading {url.split('/')[-1]}...", flush=True)
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_nc, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            ds = xr.open_dataset(local_nc, decode_times=False)
            depth_coord = ds["depth"].values.astype(np.float32)
            lat_coord = ds["lat"].values.astype(np.float32)
            lon_coord = np.mod(ds["lon"].values.astype(np.float32), 360.0)
            order = np.argsort(lon_coord)
            lon_coord = lon_coord[order]

            an = ds[v_an].isel(time=0).values[:, :, order].astype(np.float32)
            mn = ds[v_mn].isel(time=0).values[:, :, order].astype(np.float32)
            sd = ds[v_sd].isel(time=0).values[:, :, order].astype(np.float32)
            se = ds[v_se].isel(time=0).values[:, :, order].astype(np.float32)
            ci95 = (1.96 * se).astype(np.float32)

            data_vars[f"{short}_mean"] = (("depth", "lat", "lon"), an, {"units": unit, "long_name": f"WOA23 objectively analyzed mean {short}"})
            data_vars[f"{short}_stat_mean"] = (("depth", "lat", "lon"), mn, {"units": unit})
            data_vars[f"{short}_std"] = (("depth", "lat", "lon"), sd, {"units": unit})
            data_vars[f"{short}_ci95"] = (("depth", "lat", "lon"), ci95, {"units": unit, "long_name": "95% CI half-width (1.96 * SE)"})

            # Compute depth-weighted global & depth-band summary statistics
            dz = np.empty_like(depth_coord, dtype=np.float64)
            dz[0] = 0.5 * (depth_coord[1] - depth_coord[0])
            dz[1:-1] = 0.5 * (depth_coord[2:] - depth_coord[:-2])
            dz[-1] = depth_coord[-1] - depth_coord[-2]
            w_lat = _area_weights(lat_coord)[None, :, None]
            w3d = dz[:, None, None] * w_lat * np.ones((1, 1, len(lon_coord)), dtype=np.float64)

            def _band(z_lo: float, z_hi: float) -> dict[str, float]:
                zm = (depth_coord >= z_lo) & (depth_coord < z_hi)
                a_sub = an[zm].astype(np.float64)
                sd_sub = sd[zm].astype(np.float64)
                se_sub = se[zm].astype(np.float64)
                w_sub = w3d[zm]
                ok = np.isfinite(a_sub)
                mean_v = float(np.sum(a_sub[ok] * w_sub[ok]) / np.sum(w_sub[ok]))
                ok_sd = ok & np.isfinite(sd_sub)
                rms_sd = float(np.sqrt(np.sum((sd_sub[ok_sd] ** 2) * w_sub[ok_sd]) / np.sum(w_sub[ok_sd]))) if np.any(ok_sd) else float("nan")
                ok_se = ok & np.isfinite(se_sub)
                mean_ci = float(1.96 * np.sum(se_sub[ok_se] * w_sub[ok_se]) / np.sum(w_sub[ok_se])) if np.any(ok_se) else float("nan")
                return {"mean": round(mean_v, 4), "rms_std": round(rms_sd, 4), "mean_ci95": round(mean_ci, 4)}

            summary[short] = {
                "surface_0m": _band(0.0, 1.0),
                "upper_0_300m": _band(0.0, 300.0),
                "mid_300_1500m": _band(300.0, 1500.0),
                "deep_gt1500m": _band(1500.0, 10000.0),
                "global_full_depth": _band(0.0, 10000.0),
                "units": unit,
            }
            ds.close()

    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "depth": ("depth", depth_coord, {"units": "m", "positive": "down"}),
            "lat": ("lat", lat_coord, {"units": "degrees_north"}),
            "lon": ("lon", lon_coord, {"units": "degrees_east"}),
        },
        attrs={
            "source": "NOAA World Ocean Atlas 2023 (decav, 1.0 deg)",
            "Conventions": "CF-1.11",
        },
    )
    zpath = out_dir / "woa23_ocean_clim.zarr"
    _save_zarr_v3(out_ds, zpath)
    (out_dir / "woa23_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved {zpath} | Global T={summary['thetao']['global_full_depth']['mean']:.3f} degC, S={summary['so']['global_full_depth']['mean']:.3f} psu", flush=True)


# ---------------------------------------------------------------------------
# 2. Pangeo CMIP6 piControl Multi-Model Climatology (Mean + Std + 95% CI)
# ---------------------------------------------------------------------------
def fetch_cmip6_picontrol(out_dir: Path, n_years: int = 30) -> None:
    print(f"\n=== [2/3] Fetching Pangeo CMIP6 piControl Multi-Model Ensemble (last {n_years} yr/model) ===", flush=True)
    df = pd.read_csv("https://storage.googleapis.com/cmip6/pangeo-cmip6.csv")
    pi = df[df.experiment_id == "piControl"]

    # Spec: (variable_id, table_id, is_3d, units)
    var_specs = [
        ("tas", "Amon", False, "K"),
        ("pr", "Amon", False, "kg m-2 s-1"),
        ("rlut", "Amon", False, "W m-2"),
        ("rsdt", "Amon", False, "W m-2"),
        ("rsut", "Amon", False, "W m-2"),
        ("psl", "Amon", False, "Pa"),
        ("hfls", "Amon", False, "W m-2"),
        ("hfss", "Amon", False, "W m-2"),
        ("clt", "Amon", False, "%"),
        ("uas", "Amon", False, "m s-1"),
        ("vas", "Amon", False, "m s-1"),
        ("lai", "Lmon", False, "1"),
        ("gpp", "Lmon", False, "kg m-2 s-1"),
        ("ta", "Amon", True, "K"),
        ("ua", "Amon", True, "m s-1"),
        ("va", "Amon", True, "m s-1"),
        ("hus", "Amon", True, "1"),
        ("zg", "Amon", True, "m"),
    ]

    data_vars: dict[str, tuple] = {}
    summary: dict[str, dict] = {"models_requested": CMIP6_MODELS, "n_years_per_model": n_years, "variables": {}}

    for var, table, is_3d, unit in var_specs:
        ann_stack: list[np.ndarray] = []
        djf_stack: list[np.ndarray] = []
        jja_stack: list[np.ndarray] = []
        gmean_list: list[float] = []
        used_models: list[str] = []

        sub_v = pi[(pi.variable_id == var) & (pi.table_id == table)]
        for mdl in CMIP6_MODELS:
            sub_m = sub_v[sub_v.source_id == mdl]
            if sub_m.empty:
                continue
            # Prefer gn grid, r1i1p1f1
            sub_m = sub_m.sort_values(
                by=["grid_label", "member_id", "version"],
                ascending=[True, True, False],
            )
            url = sub_m.iloc[0].zstore.replace("gs://", "https://storage.googleapis.com/")
            try:
                ds = xr.open_zarr(url, consolidated=True, use_cftime=True)
                n_mo = min(n_years * 12, ds.sizes["time"])
                da = ds[var].isel(time=slice(-n_mo, None))
                # Season means directly via month grouping
                months = da["time"].dt.month
                da_ann = da.mean("time", skipna=True).load()
                da_djf = da.where(months.isin([12, 1, 2]), drop=True).mean("time", skipna=True).load()
                da_jja = da.where(months.isin([6, 7, 8]), drop=True).mean("time", skipna=True).load()

                if is_3d:
                    pdim = "plev" if "plev" in da_ann.dims else None
                    if pdim is None:
                        continue
                    da_ann = _regrid_to_2deg(da_ann).interp(plev=TARGET_PLEV_PA, method="linear")
                    da_djf = _regrid_to_2deg(da_djf).interp(plev=TARGET_PLEV_PA, method="linear")
                    da_jja = _regrid_to_2deg(da_jja).interp(plev=TARGET_PLEV_PA, method="linear")
                    a_np = da_ann.transpose("plev", "lat", "lon").values.astype(np.float32)
                    d_np = da_djf.transpose("plev", "lat", "lon").values.astype(np.float32)
                    j_np = da_jja.transpose("plev", "lat", "lon").values.astype(np.float32)
                else:
                    da_ann = _regrid_to_2deg(da_ann)
                    da_djf = _regrid_to_2deg(da_djf)
                    da_jja = _regrid_to_2deg(da_jja)
                    a_np = da_ann.transpose("lat", "lon").values.astype(np.float32)
                    d_np = da_djf.transpose("lat", "lon").values.astype(np.float32)
                    j_np = da_jja.transpose("lat", "lon").values.astype(np.float32)
                    gmean_list.append(_global_mean_2d(a_np.astype(np.float64), TARGET_LAT))

                ann_stack.append(a_np)
                djf_stack.append(d_np)
                jja_stack.append(j_np)
                used_models.append(mdl)
            except Exception as e:
                print(f"    skip {var}/{mdl}: {e}", flush=True)
                continue

        n_m = len(used_models)
        if n_m == 0:
            print(f"  WARN: no models fetched for {var}", flush=True)
            continue

        dims = ("plev", "lat", "lon") if is_3d else ("lat", "lon")
        for tag, stack in (("ann", ann_stack), ("djf", djf_stack), ("jja", jja_stack)):
            arr = np.stack(stack, axis=0)
            mean_f = np.nanmean(arr, axis=0).astype(np.float32)
            std_f = np.nanstd(arr, axis=0, ddof=1 if n_m > 1 else 0).astype(np.float32)
            n_eff = np.sum(np.isfinite(arr), axis=0)
            ci95_f = np.where(
                n_eff > 1,
                stats.t.ppf(0.975, np.maximum(n_eff - 1, 1)) * std_f / np.sqrt(np.maximum(n_eff, 1)),
                np.nan,
            ).astype(np.float32)
            p05_f = np.nanpercentile(arr, 5, axis=0).astype(np.float32)
            p95_f = np.nanpercentile(arr, 95, axis=0).astype(np.float32)

            attrs = {"units": unit, "n_models": n_m, "models": ",".join(used_models)}
            data_vars[f"{var}_{tag}_mean"] = (dims, mean_f, attrs)
            data_vars[f"{var}_{tag}_std"] = (dims, std_f, attrs)
            data_vars[f"{var}_{tag}_ci95"] = (dims, ci95_f, {**attrs, "long_name": "95% t-CI half-width of multi-model mean"})
            if tag == "ann":
                data_vars[f"{var}_{tag}_p05"] = (dims, p05_f, attrs)
                data_vars[f"{var}_{tag}_p95"] = (dims, p95_f, attrs)

        if not is_3d and gmean_list:
            gm_arr = np.array(gmean_list, dtype=np.float64)
            gm_mean = float(np.nanmean(gm_arr))
            gm_std = float(np.nanstd(gm_arr, ddof=1)) if n_m > 1 else float("nan")
            gm_ci = float(_t_ci95_halfwidth(gm_std, n_m))
            summary["variables"][var] = {
                "n_models": n_m,
                "models": used_models,
                "units": unit,
                "global_mean": round(gm_mean, 5),
                "intermodel_std": round(gm_std, 5),
                "ci95_halfwidth": round(gm_ci, 5),
                "p05": round(float(np.nanpercentile(gm_arr, 5)), 5),
                "p95": round(float(np.nanpercentile(gm_arr, 95)), 5),
            }
            print(f"  {var:6s} ({n_m:2d} models): global_mean = {gm_mean:.4g} ± {gm_ci:.3g} (95% CI), std = {gm_std:.3g} [{unit}]", flush=True)
        else:
            summary["variables"][var] = {"n_models": n_m, "models": used_models, "units": unit, "dims": list(dims)}
            print(f"  {var:6s} ({n_m:2d} models, 3D plev19): done", flush=True)

    # Also compute global summary for rectilinear ocean surface vars (tos, sos) where available on regular gr/gn
    for ovar, otable, ounit in (("tos", "Omon", "degC"), ("sos", "Omon", "psu")):
        sub_v = pi[(pi.variable_id == ovar) & (pi.table_id == otable) & (pi.grid_label == "gr")]
        ann_stack, djf_stack, jja_stack, gmean_list, used_models = [], [], [], [], []
        for mdl in CMIP6_MODELS:
            sub_m = sub_v[sub_v.source_id == mdl]
            if sub_m.empty:
                continue
            url = sub_m.iloc[0].zstore.replace("gs://", "https://storage.googleapis.com/")
            try:
                ds = xr.open_zarr(url, consolidated=True, use_cftime=True)
                n_mo = min(n_years * 12, ds.sizes["time"])
                da = ds[ovar].isel(time=slice(-n_mo, None))
                months = da["time"].dt.month
                da_ann = _regrid_to_2deg(da.mean("time", skipna=True).load())
                da_djf = _regrid_to_2deg(da.where(months.isin([12, 1, 2]), drop=True).mean("time", skipna=True).load())
                da_jja = _regrid_to_2deg(da.where(months.isin([6, 7, 8]), drop=True).mean("time", skipna=True).load())
                a_np = da_ann.transpose("lat", "lon").values.astype(np.float32)
                ann_stack.append(a_np)
                djf_stack.append(da_djf.transpose("lat", "lon").values.astype(np.float32))
                jja_stack.append(da_jja.transpose("lat", "lon").values.astype(np.float32))
                gmean_list.append(_global_mean_2d(a_np.astype(np.float64), TARGET_LAT))
                used_models.append(mdl)
            except Exception as e:
                print(f"    skip {ovar}/{mdl}: {e}", flush=True)
        n_m = len(used_models)
        if n_m > 0:
            for tag, stack in (("ann", ann_stack), ("djf", djf_stack), ("jja", jja_stack)):
                arr = np.stack(stack, axis=0)
                mean_f = np.nanmean(arr, axis=0).astype(np.float32)
                std_f = np.nanstd(arr, axis=0, ddof=1 if n_m > 1 else 0).astype(np.float32)
                n_eff = np.sum(np.isfinite(arr), axis=0)
                ci95_f = np.where(
                    n_eff > 1,
                    stats.t.ppf(0.975, np.maximum(n_eff - 1, 1)) * std_f / np.sqrt(np.maximum(n_eff, 1)),
                    np.nan,
                ).astype(np.float32)
                attrs = {"units": ounit, "n_models": n_m, "models": ",".join(used_models)}
                data_vars[f"{ovar}_{tag}_mean"] = (("lat", "lon"), mean_f, attrs)
                data_vars[f"{ovar}_{tag}_std"] = (("lat", "lon"), std_f, attrs)
                data_vars[f"{ovar}_{tag}_ci95"] = (("lat", "lon"), ci95_f, attrs)
            gm_arr = np.array(gmean_list, dtype=np.float64)
            gm_mean = float(np.nanmean(gm_arr))
            gm_std = float(np.nanstd(gm_arr, ddof=1)) if n_m > 1 else float("nan")
            gm_ci = float(_t_ci95_halfwidth(gm_std, n_m))
            summary["variables"][ovar] = {
                "n_models": n_m,
                "models": used_models,
                "units": ounit,
                "global_mean": round(gm_mean, 4),
                "intermodel_std": round(gm_std, 4),
                "ci95_halfwidth": round(gm_ci, 4),
            }
            print(f"  {ovar:6s} ({n_m:2d} models, gr): global_mean = {gm_mean:.4g} ± {gm_ci:.3g} [{ounit}]", flush=True)

    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east"}),
            "plev": ("plev", TARGET_PLEV_PA, {"units": "Pa", "positive": "down"}),
        },
        attrs={
            "source": "Pangeo CMIP6 piControl Multi-Model Ensemble (gs://cmip6/)",
            "n_years_per_model": n_years,
            "Conventions": "CF-1.11",
        },
    )
    zpath = out_dir / "cmip6_picontrol_clim.zarr"
    _save_zarr_v3(out_ds, zpath)
    (out_dir / "cmip6_picontrol_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved {zpath}", flush=True)


# ---------------------------------------------------------------------------
# 3. Google Research ARCO-ERA5 1.5-deg Climatologies (Early 1959-1978 & Modern 1991-2020)
# ---------------------------------------------------------------------------
def fetch_arco_era5(out_dir: Path) -> None:
    print("\n=== [3/3] Fetching ARCO-ERA5 1.5-deg Climatologies (Early 1959-1978 & Modern 1991-2020) ===", flush=True)
    era_url = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
    ds = xr.open_zarr(era_url, consolidated=True)

    # Map ERA5 names -> (CMIP7 short name, scale, units, is_3d)
    # Note: total_precipitation_6hr is [m / 6h] -> * (1000 / 21600) = [kg m-2 s-1]
    vars_2d = [
        ("2m_temperature", "tas", 1.0, "K"),
        ("total_precipitation_6hr", "pr", 1000.0 / 21600.0, "kg m-2 s-1"),
        ("mean_sea_level_pressure", "psl", 1.0, "Pa"),
        ("surface_pressure", "ps", 1.0, "Pa"),
        ("total_cloud_cover", "clt", 100.0, "%"),
        ("total_column_water_vapour", "prw", 1.0, "kg m-2"),
        ("10m_u_component_of_wind", "uas", 1.0, "m s-1"),
        ("10m_v_component_of_wind", "vas", 1.0, "m s-1"),
        ("10m_wind_speed", "sfcWind", 1.0, "m s-1"),
        ("sea_surface_temperature", "tos", 1.0, "K"),
        ("sea_ice_cover", "siconc", 100.0, "%"),
    ]
    vars_3d = [
        ("temperature", "ta", 1.0, "K"),
        ("u_component_of_wind", "ua", 1.0, "m s-1"),
        ("v_component_of_wind", "va", 1.0, "m s-1"),
        ("specific_humidity", "hus", 1.0, "1"),
        ("geopotential", "zg", 1.0 / 9.80665, "m"),
    ]

    lat_e = ds["latitude"].values.astype(np.float32)
    lon_e = ds["longitude"].values.astype(np.float32)
    lev_pa = (ds["level"].values.astype(np.float32) * 100.0)
    lat_ord = np.argsort(lat_e)
    lat_sorted = lat_e[lat_ord]

    periods = {
        "early_1959_1978": (1959, 1978),
        "modern_1991_2020": (1991, 2020),
    }

    data_vars: dict[str, tuple] = {}
    summary: dict[str, dict] = {}

    # Sample every 8th 6-hr step (every 2 days, alternating 00Z/06Z/12Z/18Z via stride=5 -> 30h stride!)
    # A 30h stride (5 * 6h) samples all 4 diurnal hours equally over every 5 days while cutting I/O 5x!
    STRIDE_2D = 5
    STRIDE_3D = 15  # Every 3.75 days across all 4 diurnal hours -> ~97 samples/year -> exact seasonal means

    for ptag, (y0, y1) in periods.items():
        print(f"  Period {ptag} ({y0}-{y1})...", flush=True)
        summary[ptag] = {}
        ds_p = ds.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))

        for era_name, short, scale, unit in vars_2d:
            da = ds_p[era_name].isel(time=slice(None, None, STRIDE_2D))
            # Annual means per year for interannual std & 95% CI
            yr_ann = (da.groupby("time.year").mean("time").load() * scale).transpose("year", "latitude", "longitude").values[:, lat_ord, :]
            n_y = yr_ann.shape[0]
            mean_ann = np.nanmean(yr_ann, axis=0).astype(np.float32)
            std_ann = np.nanstd(yr_ann, axis=0, ddof=1).astype(np.float32)
            ci_ann = _t_ci95_halfwidth(std_ann, n_y).astype(np.float32)

            months = da["time"].dt.month
            mean_djf = (da.where(months.isin([12, 1, 2]), drop=True).mean("time").load() * scale).transpose("latitude", "longitude").values[lat_ord, :].astype(np.float32)
            mean_jja = (da.where(months.isin([6, 7, 8]), drop=True).mean("time").load() * scale).transpose("latitude", "longitude").values[lat_ord, :].astype(np.float32)

            attrs = {"units": unit, "period": f"{y0}-{y1}", "n_years": n_y}
            data_vars[f"{ptag}_{short}_ann_mean"] = (("lat", "lon"), mean_ann, attrs)
            data_vars[f"{ptag}_{short}_ann_std"] = (("lat", "lon"), std_ann, attrs)
            data_vars[f"{ptag}_{short}_ann_ci95"] = (("lat", "lon"), ci_ann, attrs)
            data_vars[f"{ptag}_{short}_djf_mean"] = (("lat", "lon"), mean_djf, attrs)
            data_vars[f"{ptag}_{short}_jja_mean"] = (("lat", "lon"), mean_jja, attrs)

            gm_yrs = np.array([_global_mean_2d(yr_ann[i].astype(np.float64), lat_sorted) for i in range(n_y)])
            gm_m = float(np.nanmean(gm_yrs))
            gm_s = float(np.nanstd(gm_yrs, ddof=1))
            gm_c = float(_t_ci95_halfwidth(gm_s, n_y))
            summary[ptag][short] = {
                "units": unit,
                "n_years": n_y,
                "global_mean": round(gm_m, 5),
                "interannual_std": round(gm_s, 5),
                "ci95_halfwidth": round(gm_c, 5),
            }
            print(f"    {short:8s}: {gm_m:.4g} ± {gm_c:.3g} (std={gm_s:.3g}) [{unit}]", flush=True)

        for era_name, short, scale, unit in vars_3d:
            da = ds_p[era_name].isel(time=slice(None, None, STRIDE_3D))
            yr_ann = (da.groupby("time.year").mean("time").load() * scale).transpose("year", "level", "latitude", "longitude").values[:, :, lat_ord, :]
            n_y = yr_ann.shape[0]
            mean_ann = np.nanmean(yr_ann, axis=0).astype(np.float32)
            std_ann = np.nanstd(yr_ann, axis=0, ddof=1).astype(np.float32)
            ci_ann = _t_ci95_halfwidth(std_ann, n_y).astype(np.float32)

            months = da["time"].dt.month
            mean_djf = (da.where(months.isin([12, 1, 2]), drop=True).mean("time").load() * scale).transpose("level", "latitude", "longitude").values[:, lat_ord, :].astype(np.float32)
            mean_jja = (da.where(months.isin([6, 7, 8]), drop=True).mean("time").load() * scale).transpose("level", "latitude", "longitude").values[:, lat_ord, :].astype(np.float32)

            attrs = {"units": unit, "period": f"{y0}-{y1}", "n_years": n_y}
            data_vars[f"{ptag}_{short}_ann_mean"] = (("plev", "lat", "lon"), mean_ann, attrs)
            data_vars[f"{ptag}_{short}_ann_std"] = (("plev", "lat", "lon"), std_ann, attrs)
            data_vars[f"{ptag}_{short}_ann_ci95"] = (("plev", "lat", "lon"), ci_ann, attrs)
            data_vars[f"{ptag}_{short}_djf_mean"] = (("plev", "lat", "lon"), mean_djf, attrs)
            data_vars[f"{ptag}_{short}_jja_mean"] = (("plev", "lat", "lon"), mean_jja, attrs)
            print(f"    {short:8s} (3D 13-lev): done", flush=True)

    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": ("lat", lat_sorted, {"units": "degrees_north"}),
            "lon": ("lon", lon_e, {"units": "degrees_east"}),
            "plev": ("plev", lev_pa, {"units": "Pa", "positive": "down"}),
        },
        attrs={
            "source": "Google Research ARCO-ERA5 (1.5-deg 240x121 conservative)",
            "Conventions": "CF-1.11",
        },
    )
    zpath = out_dir / "era5_clim.zarr"
    _save_zarr_v3(out_ds, zpath)
    (out_dir / "era5_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved {zpath}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch compact climatology + CI validation benchmarks for AOBA")
    parser.add_argument("--out-dir", type=str, default=str(VAL_DIR))
    parser.add_argument("--only", choices=["all", "woa23", "cmip6", "era5"], default="all")
    parser.add_argument("--cmip6-years", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "woa23"):
        fetch_woa23(out_dir)
    if args.only in ("all", "cmip6"):
        fetch_cmip6_picontrol(out_dir, n_years=args.cmip6_years)
    if args.only in ("all", "era5"):
        fetch_arco_era5(out_dir)
    print("\nALL VALIDATION BENCHMARKS READY IN:", out_dir, flush=True)


if __name__ == "__main__":
    main()
