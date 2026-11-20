#!/usr/bin/env python3
"""Long-duration AOBA climate integration runner with 1-year-per-directory Zarr v3 output & restart."""

from __future__ import annotations

import argparse
import csv
from dataclasses import is_dataclass
import os
from pathlib import Path
import time
import numpy as np
import torch
import xarray as xr

from aoba import constants as C
from aoba.coupled_model import (
    AOBAConfig,
    AOBAState,
    diagnose_model_koppen_geiger,
    init_aoba_model,
    step_aoba_model,
)
from aoba.io_zarr import (
    append_zarr,
    export_aoba_snapshot_xarray,
    export_scvt_mesh_xarray,
    save_zarr,
)
from aoba.radiation import compute_four_band_radiation
from aoba.runtime import get_device, get_output_dir
from aoba.scvt_mesh import reconstruct_cell_winds
from aoba.surface import SURF_LAND, SURF_OCEAN, compute_surface_albedos
from aoba.thermodynamics import diagnose_pressure_temperature


def _move_obj_tensors(obj, target_device: torch.device):
    """Recursively move all tensors inside nested dataclasses/dicts to `target_device`."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=target_device)
    if isinstance(obj, dict):
        return {k: _move_obj_tensors(v, target_device) for k, v in obj.items()}
    if is_dataclass(obj):
        for k, v in obj.__dict__.items():
            setattr(obj, k, _move_obj_tensors(v, target_device))
        return obj
    return obj


def _save_restart_atomic(model: AOBAState, path: Path) -> None:
    """Save complete prognostic `AOBAState` cleanly via atomic temp file + rename."""
    payload = {
        "cfg": model.cfg,
        "sim_time_sec": model.sim_time_sec,
        "step_count": model.step_count,
        "atmos": _move_obj_tensors(
            type(model.atmos)(**{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.atmos.__dict__.items()}),
            torch.device("cpu"),
        ),
        "crm": _move_obj_tensors(
            type(model.crm)(**{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.crm.__dict__.items()}),
            torch.device("cpu"),
        ),
        "sfc": {
            k: (
                {bk: bv.detach().cpu() if isinstance(bv, torch.Tensor) else bv for bk, bv in v.__dict__.items()}
                if k == "bio"
                else (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            )
            for k, v in model.sfc.__dict__.items()
        },
        "ocn": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.ocn.__dict__.items()},
        "clim": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.clim.__dict__.items()},
        "rad_cache": {k: v.detach().cpu() for k, v in model.rad_cache.items()},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_restart_into_model(model: AOBAState, path: Path, device: torch.device) -> None:
    """Restore prognostic tensors and accumulators from `restart.pt` into `model`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.sim_time_sec = float(ckpt["sim_time_sec"])
    model.step_count = int(ckpt["step_count"])
    for k, v in ckpt["atmos"].__dict__.items():
        setattr(model.atmos, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    for k, v in ckpt["crm"].__dict__.items():
        setattr(model.crm, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    for k, v in ckpt["sfc"].items():
        if k == "bio":
            for bk, bv in v.items():
                setattr(model.sfc.bio, bk, bv.to(device=device) if isinstance(bv, torch.Tensor) else bv)
        else:
            setattr(model.sfc, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    for k, v in ckpt["ocn"].items():
        setattr(model.ocn, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    for k, v in ckpt["clim"].items():
        setattr(model.clim, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    model.rad_cache = {k: v.to(device=device) for k, v in ckpt["rad_cache"].items()}


def _build_monthly_mean_xarray(model: AOBAState, sums: dict[str, torch.Tensor], count: int) -> xr.Dataset:
    """Package accumulated monthly-mean 2D & 3D fields into a CF/UGRID Zarr-compatible Dataset."""
    inv = 1.0 / float(max(count, 1))
    mesh = model.mesh
    t_days = np.array([model.sim_time_sec / C.SECONDS_PER_DAY], dtype=np.float64)
    lat_deg = np.rad2deg(mesh.lat_cell.detach().cpu().numpy())
    lon_deg = np.rad2deg(mesh.lon_cell.detach().cpu().numpy())
    kg30, kg5 = diagnose_model_koppen_geiger(model)

    data_vars = {}
    units_2d = {
        "t_ocean_ml": "K", "t_canopy": "K", "m_seaice": "kg m-2", "swe_land": "kg m-2",
        "w_root": "kg m-2", "lai": "m2 m-2", "precip_mm_day": "mm day-1",
        "olr": "W m-2", "asr": "W m-2", "evap_mm_day": "mm day-1", "gpp": "umol m-2 s-1",
    }
    units_3d = {"temp": "K", "pres": "Pa", "q_v": "kg kg-1", "u_zonal": "m s-1", "v_merid": "m s-1"}

    for name, unit in units_2d.items():
        arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
        data_vars[name] = (("time", "n_cell"), arr, {"units": unit})
    for name, unit in units_3d.items():
        arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :]
        data_vars[name] = (("time", "nz", "n_cell"), arr, {"units": unit})

    data_vars["koppen_code"] = (("time", "n_cell"), kg30.detach().cpu().numpy()[None, :])
    data_vars["koppen_major"] = (("time", "n_cell"), kg5.detach().cpu().numpy()[None, :])

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", t_days, {"units": "days since 2000-03-20 00:00:00"}),
            "nz": ("nz", model.vgrid.zeta_c.detach().cpu().numpy(), {"units": "m", "positive": "up"}),
            "lat_cell": ("n_cell", lat_deg, {"units": "degrees_north"}),
            "lon_cell": ("n_cell", lon_deg, {"units": "degrees_east"}),
        },
        attrs={
            "Conventions": "CF-1.11 UGRID-1.0",
            "glevel": model.cfg.glevel,
            "co2_ppmv": model.cfg.co2_ppmv,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-year Glevel-5 AOBA climate integration (1-year-per-dir Zarr)")
    parser.add_argument("--glevel", type=int, default=5)
    parser.add_argument("--years", type=int, default=30)
    parser.add_argument("--days-per-year", type=float, default=C.DAYS_PER_YEAR)
    parser.add_argument("--co2-ppmv", type=float, default=400.0)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else get_device()
    out_root = Path(args.out_dir).expanduser() if args.out_dir else get_output_dir(f"glevel{args.glevel}_30yr")
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = AOBAConfig(glevel=args.glevel, co2_ppmv=args.co2_ppmv)
    print(f"Initializing AOBA Glevel-{cfg.glevel} ({cfg.crm_nx}x{cfg.crm_ny}, dx={cfg.crm_dx:.0f} m, dt={cfg.dt_host:.0f} s) on {dev}...", flush=True)
    model = init_aoba_model(cfg, device=dev)

    mesh_zarr = out_root / "mesh.zarr"
    if not mesh_zarr.exists():
        save_zarr(export_scvt_mesh_xarray(model), mesh_zarr, overwrite=True)

    start_year = 1
    if not args.no_resume:
        for y in range(args.years, 0, -1):
            y_dir = out_root / f"year_{y:04d}"
            if (y_dir / "_COMPLETE").exists() and (y_dir / "restart.pt").exists():
                print(f"Resuming from completed Year {y:04d} ({y_dir / 'restart.pt'})...", flush=True)
                _load_restart_into_model(model, y_dir / "restart.pt", dev)
                start_year = y + 1
                break

    steps_per_day = max(int(round(C.SECONDS_PER_DAY / cfg.dt_host)), 1)
    steps_per_year = max(int(round(args.days_per_year * C.SECONDS_PER_DAY / cfg.dt_host)), 12)
    month_bounds = [int(round(m * steps_per_year / 12.0)) for m in range(13)]

    for year in range(start_year, args.years + 1):
        y_dir = out_root / f"year_{year:04d}"
        y_dir.mkdir(parents=True, exist_ok=True)
        monthly_zarr = y_dir / "monthly.zarr"
        snap_zarr = y_dir / "snapshots.zarr"
        csv_path = y_dir / "daily_global.csv"

        # Clean partial current-year Zarr/CSV only if starting this year fresh
        for p in (monthly_zarr, snap_zarr):
            if p.exists():
                import shutil
                shutil.rmtree(p)

        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["year", "day_in_yr", "sim_days", "ts_k", "olr_wm2", "asr_wm2", "net_toa_wm2", "precip_mm_d", "gpp_umol", "lai", "seaice_mkm2", "wall_s_per_day"])

        day_accum = {"ts": 0.0, "olr": 0.0, "asr": 0.0, "p": 0.0, "gpp": 0.0, "lai": 0.0, "ice": 0.0}
        day_cnt = 0
        m_idx = 0
        m_sums: dict[str, torch.Tensor] = {}
        m_cnt = 0
        t_day0 = time.perf_counter()

        for step_in_yr in range(1, steps_per_year + 1):
            diag = step_aoba_model(model)

            day_accum["ts"] += diag.global_mean_t_sfc_k
            day_accum["olr"] += diag.global_mean_olr_wm2
            day_accum["asr"] += diag.global_mean_asr_wm2
            day_accum["p"] += diag.global_mean_precip_mm_day
            day_accum["gpp"] += diag.land_mean_gpp_umol
            day_accum["lai"] += diag.land_mean_lai
            day_accum["ice"] += diag.seaice_area_million_km2
            day_cnt += 1

            # Sample 3D/2D monthly climatology every 6 simulated hours (or every step when steps_per_year is tiny)
            sample_stride = max(min(steps_per_day // 4, 30), 1)
            if step_in_yr % sample_stride == 0 or step_in_yr == month_bounds[m_idx + 1]:
                u_c, v_c = reconstruct_cell_winds(model.mesh, model.atmos.u_edge)
                pres_c, _, temp_c = diagnose_pressure_temperature(model.atmos.rho_d, model.atmos.theta_m, model.atmos.q_v)
                rain = model.rad_cache.get("rain", torch.zeros_like(model.sfc.t_ocean_ml))
                snow = model.rad_cache.get("snow", torch.zeros_like(model.sfc.t_ocean_ml))
                evap = model.rad_cache.get("evap", torch.zeros_like(model.sfc.t_ocean_ml))
                gpp_c = model.rad_cache.get("gpp", torch.zeros_like(model.sfc.t_ocean_ml))
                olr_c = model.rad_cache.get("olr", torch.zeros_like(model.sfc.t_ocean_ml))
                asr_c = model.rad_cache.get("asr", torch.zeros_like(model.sfc.t_ocean_ml))
                sample = {
                    "t_ocean_ml": model.sfc.t_ocean_ml,
                    "t_canopy": model.sfc.t_canopy,
                    "m_seaice": model.sfc.m_seaice,
                    "swe_land": model.sfc.swe_land,
                    "w_root": model.sfc.w_root,
                    "lai": model.sfc.bio.lai,
                    "precip_mm_day": (rain + snow) * C.SECONDS_PER_DAY,
                    "olr": olr_c,
                    "asr": asr_c,
                    "evap_mm_day": evap * C.SECONDS_PER_DAY,
                    "gpp": gpp_c,
                    "temp": temp_c,
                    "pres": pres_c,
                    "q_v": model.atmos.q_v,
                    "u_zonal": u_c,
                    "v_merid": v_c,
                }
                if m_cnt == 0:
                    m_sums = {k: v.clone() for k, v in sample.items()}
                else:
                    for k, v in sample.items():
                        m_sums[k] += v
                m_cnt += 1

            if step_in_yr % steps_per_day == 0 or step_in_yr == steps_per_year:
                if dev.type == "mps":
                    torch.mps.synchronize()
                wall_day = time.perf_counter() - t_day0
                t_day0 = time.perf_counter()
                inv_d = 1.0 / float(max(day_cnt, 1))
                day_num = step_in_yr / float(steps_per_day)
                ts_m = day_accum["ts"] * inv_d
                olr_m = day_accum["olr"] * inv_d
                asr_m = day_accum["asr"] * inv_d
                with open(csv_path, "a", newline="") as f_csv:
                    csv.writer(f_csv).writerow([
                        year, f"{day_num:.2f}", f"{diag.sim_time_days:.2f}",
                        f"{ts_m:.3f}", f"{olr_m:.2f}", f"{asr_m:.2f}", f"{asr_m - olr_m:.2f}",
                        f"{day_accum['p'] * inv_d:.3f}", f"{day_accum['gpp'] * inv_d:.3f}",
                        f"{day_accum['lai'] * inv_d:.3f}", f"{day_accum['ice'] * inv_d:.3f}", f"{wall_day:.2f}",
                    ])
                if int(round(day_num)) % 10 == 0 or step_in_yr == steps_per_year:
                    print(
                        f"[Y{year:02d} D{day_num:5.1f}] Ts={ts_m:6.2f} K | NetTOA={asr_m - olr_m:+5.2f} | "
                        f"P={day_accum['p'] * inv_d:4.2f} mm/d | LAI={day_accum['lai'] * inv_d:4.2f} | "
                        f"Ice={day_accum['ice'] * inv_d:5.2f} Mkm2 | {wall_day:5.1f} s/d",
                        flush=True,
                    )
                for k in day_accum:
                    day_accum[k] = 0.0
                day_cnt = 0

            if step_in_yr == month_bounds[m_idx + 1]:
                append_zarr(_build_monthly_mean_xarray(model, m_sums, m_cnt), monthly_zarr)
                append_zarr(export_aoba_snapshot_xarray(model), snap_zarr)
                m_sums.clear()
                m_cnt = 0
                m_idx += 1

        _save_restart_atomic(model, y_dir / "restart.pt")
        (y_dir / "_COMPLETE").touch()
        print(f"=== Completed Year {year:04d} / {args.years:04d} -> {y_dir} ===", flush=True)


if __name__ == "__main__":
    main()
