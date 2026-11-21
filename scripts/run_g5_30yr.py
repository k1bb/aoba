#!/usr/bin/env python3
"""Long-duration AOBA climate integration runner with 1-year-per-directory Zarr v3 output & restart."""

from __future__ import annotations

import argparse
import csv
from dataclasses import is_dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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
    diagnose_subgrid_2d_fields,
    export_aoba_snapshot_xarray,
    export_scvt_mesh_xarray,
    save_zarr,
)
from aoba.params import (
    OutputConfig,
    build_configs_from_dict,
    default_config_dict,
    load_aoba_toml,
)
from aoba.radiation import compute_four_band_radiation
from aoba.runtime import apply_runtime_tuning, compute_optimal_chunks, get_device, get_output_dir
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


def _serialize_surface(sfc_obj) -> dict:
    """Detach and transfer all `SurfaceState` and embedded `BiosphereState` tensors to CPU."""
    return {
        k: (
            {bk: bv.detach().cpu() if isinstance(bv, torch.Tensor) else bv for bk, bv in v.__dict__.items()}
            if k == "bio"
            else (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        )
        for k, v in sfc_obj.__dict__.items()
    }


def _restore_surface(sfc_target, payload: dict, device: torch.device) -> None:
    """Restore serialized `SurfaceState` tensors onto `device`."""
    for k, v in payload.items():
        if k == "bio":
            for bk, bv in v.items():
                setattr(sfc_target.bio, bk, bv.to(device=device) if isinstance(bv, torch.Tensor) else bv)
        else:
            setattr(sfc_target, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)


def _save_restart_atomic(model: AOBAState, path: Path) -> None:
    """Save complete prognostic `AOBAState` (including subgrid `sfc_sub`) via atomic temp file + rename."""
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
        "sfc": _serialize_surface(model.sfc),
        "sfc_sub": _serialize_surface(model.sfc_sub) if model.sfc_sub is not None else None,
        "ocn": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.ocn.__dict__.items()},
        "clim": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.clim.__dict__.items()},
        "clim_sub": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.clim_sub.__dict__.items()}
            if model.clim_sub is not None
            else None
        ),
        "icesheet": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.icesheet.__dict__.items()}
            if model.icesheet is not None
            else None
        ),
        "mesh_dynamic": {
            "z_topo": model.mesh.z_topo.detach().cpu(),
            "z_topo_smooth": model.mesh.z_topo_smooth.detach().cpu(),
            "surf_type": model.mesh.surf_type.detach().cpu(),
        },
        "rad_cache": {k: v.detach().cpu() for k, v in model.rad_cache.items()},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_restart_into_model(model: AOBAState, path: Path, device: torch.device) -> None:
    """Restore prognostic tensors (including subgrid `sfc_sub` & `clim_sub`) from `restart.pt`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.sim_time_sec = float(ckpt["sim_time_sec"])
    model.step_count = int(ckpt["step_count"])
    for k, v in ckpt["atmos"].__dict__.items():
        setattr(model.atmos, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    for k, v in ckpt["crm"].__dict__.items():
        setattr(model.crm, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    _restore_surface(model.sfc, ckpt["sfc"], device)
    if ckpt.get("sfc_sub") is not None and model.sfc_sub is not None:
        _restore_surface(model.sfc_sub, ckpt["sfc_sub"], device)
    for k, v in ckpt["ocn"].items():
        setattr(model.ocn, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    for k, v in ckpt["clim"].items():
        setattr(model.clim, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    if ckpt.get("clim_sub") is not None and model.clim_sub is not None:
        for k, v in ckpt["clim_sub"].items():
            setattr(model.clim_sub, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    if ckpt.get("icesheet") is not None and model.icesheet is not None:
        for k, v in ckpt["icesheet"].items():
            setattr(model.icesheet, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    if ckpt.get("mesh_dynamic") is not None:
        for k, v in ckpt["mesh_dynamic"].items():
            setattr(model.mesh, k, v.to(device=device))
    model.rad_cache = {k: v.to(device=device) for k, v in ckpt["rad_cache"].items()}


def _build_monthly_mean_xarray(
    model: AOBAState,
    sums: dict[str, torch.Tensor],
    count: int,
    *,
    include_2d: bool = True,
    include_3d: bool = True,
    include_subgrid_2d: bool = True,
) -> xr.Dataset:
    """Package accumulated time-mean 2D, 3D & subgrid 2D fields into a CF/UGRID Zarr Dataset."""
    inv = 1.0 / float(max(count, 1))
    mesh = model.mesh
    t_days = np.array([model.sim_time_sec / C.SECONDS_PER_DAY], dtype=np.float64)
    lat_deg = np.rad2deg(mesh.lat_cell.detach().cpu().numpy())
    lon_deg = np.rad2deg(mesh.lon_cell.detach().cpu().numpy())

    data_vars = {}
    units_2d = {
        "t_ocean_ml": "K", "t_canopy": "K", "m_seaice": "kg m-2", "swe_land": "kg m-2",
        "w_root": "kg m-2", "lai": "m2 m-2", "precip_mm_day": "mm day-1",
        "olr": "W m-2", "asr": "W m-2", "evap_mm_day": "mm day-1", "gpp": "umol m-2 s-1",
    }
    units_3d = {"temp": "K", "pres": "Pa", "q_v": "kg kg-1", "u_zonal": "m s-1", "v_merid": "m s-1"}

    if include_2d:
        kg30, kg5 = diagnose_model_koppen_geiger(model)
        for name, unit in units_2d.items():
            arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
            data_vars[name] = (("time", "n_cell"), arr, {"units": unit})
        if model.icesheet is not None:
            for name in ("h_ice", "z_topo_dyn"):
                arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
                data_vars[name] = (("time", "n_cell"), arr, {"units": "m"})
        data_vars["koppen_code"] = (("time", "n_cell"), kg30.detach().cpu().numpy()[None, :])
        data_vars["koppen_major"] = (("time", "n_cell"), kg5.detach().cpu().numpy()[None, :])
    if include_3d:
        for name, unit in units_3d.items():
            arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :]
            data_vars[name] = (("time", "nz", "n_cell"), arr, {"units": unit})
    if include_subgrid_2d:
        for name, (_, unit) in diagnose_subgrid_2d_fields(model).items():
            arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :, :]
            data_vars[name] = (("time", "n_cell", "crm_y", "crm_x"), arr, {"units": unit})

    _, _, ny, nx = model.crm.rho_d.shape

    coords: dict[str, Any] = {
        "time": ("time", t_days, {"units": "days since 2000-03-20 00:00:00"}),
        "lat_cell": ("n_cell", lat_deg, {"units": "degrees_north"}),
        "lon_cell": ("n_cell", lon_deg, {"units": "degrees_east"}),
    }
    if include_3d:
        coords["nz"] = ("nz", model.vgrid.zeta_c.detach().cpu().numpy(), {"units": "m", "positive": "up"})
    if include_subgrid_2d:
        coords["crm_y"] = ("crm_y", np.arange(ny, dtype=np.float32) * model.crm.dy, {"units": "m"})
        coords["crm_x"] = ("crm_x", np.arange(nx, dtype=np.float32) * model.crm.dx, {"units": "m"})

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "Conventions": "CF-1.11 UGRID-1.0",
            "glevel": model.cfg.glevel,
            "co2_ppmv": model.cfg.co2_ppmv,
        },
    )


def _write_live_preview_atomic(
    model: AOBAState,
    out_path: Path,
    year: int,
    day_in_yr: float,
    sim_days: float,
    wall_s_per_day: float,
) -> None:
    """Atomically write lightweight 2D live-preview fields (`T2`, `OLR`, `LAI`) for the CPU GUI."""
    mesh = model.mesh
    _, _, temp0 = diagnose_pressure_temperature(
        model.atmos.rho_d[0:1], model.atmos.theta_m[0:1], model.atmos.q_v[0:1]
    )
    t_sfc = torch.where(mesh.surf_type == SURF_OCEAN, model.sfc.t_ocean_ml, model.sfc.t_canopy)
    t2_k = 0.35 * t_sfc + 0.65 * temp0[0]
    olr = model.rad_cache.get("olr", 0.85 * C.STEFAN_BOLTZMANN * torch.clamp(t_sfc, min=150.0) ** 4)
    lai = model.sfc.bio.lai
    h_ice = model.icesheet.h_ice if model.icesheet is not None else torch.zeros_like(t_sfc)

    tmp_path = out_path.with_name(f".{out_path.name}.tmp.npz")
    np.savez(
        tmp_path,
        lat_deg=np.rad2deg(mesh.lat_cell.detach().cpu().numpy()).astype(np.float32),
        lon_deg=np.rad2deg(mesh.lon_cell.detach().cpu().numpy()).astype(np.float32),
        surf_type=mesh.surf_type.detach().cpu().numpy().astype(np.int8),
        t2_k=t2_k.detach().cpu().numpy().astype(np.float32),
        olr=olr.detach().cpu().numpy().astype(np.float32),
        lai=lai.detach().cpu().numpy().astype(np.float32),
        h_ice=h_ice.detach().cpu().numpy().astype(np.float32),
        year=np.int32(year),
        day_in_yr=np.float32(day_in_yr),
        sim_days=np.float64(sim_days),
        wall_s_per_day=np.float32(wall_s_per_day),
    )
    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-year AOBA climate integration (TOML-configurable Zarr v3 output)")
    parser.add_argument("--config", type=str, default="", help="Path to unified AOBA TOML configuration file")
    parser.add_argument("--glevel", type=int, default=None)
    parser.add_argument("--years", type=int, default=None)
    parser.add_argument("--days-per-year", type=float, default=None)
    parser.add_argument("--co2-ppmv", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dynamic-icesheet", action="store_true", default=None, help="Enable SIA + EMA-accelerated ice-sheet coupling")
    parser.add_argument("--icesheet-ema-years", type=float, default=None)
    parser.add_argument("--icesheet-accel", type=float, default=None)
    parser.add_argument("--icesheet-accel-spinup-years", type=float, default=None)
    parser.add_argument("--icesheet-isostasy-tau-yr", type=float, default=None)
    args = parser.parse_args()

    if args.config:
        cfg_dict = load_aoba_toml(args.config)
        cfg, out_cfg = build_configs_from_dict(cfg_dict, apply_modules=True)
    else:
        cfg_dict = default_config_dict()
        cfg, out_cfg = build_configs_from_dict(cfg_dict, apply_modules=False)

    if args.glevel is not None:
        cfg.glevel = args.glevel
    if args.co2_ppmv is not None:
        cfg.co2_ppmv = args.co2_ppmv
    if args.dynamic_icesheet is not None:
        cfg.enable_dynamic_icesheet = bool(args.dynamic_icesheet)
    if args.icesheet_ema_years is not None:
        cfg.icesheet_ema_years = args.icesheet_ema_years
    if args.icesheet_accel is not None:
        cfg.icesheet_accel_factor = args.icesheet_accel
    if args.icesheet_accel_spinup_years is not None:
        cfg.icesheet_accel_spinup_years = args.icesheet_accel_spinup_years
    if args.icesheet_isostasy_tau_yr is not None:
        cfg.icesheet_isostasy_tau_yr = args.icesheet_isostasy_tau_yr

    years = args.years if args.years is not None else out_cfg.years
    days_per_year = args.days_per_year if args.days_per_year is not None else out_cfg.days_per_year
    out_dir_str = args.out_dir if args.out_dir else out_cfg.out_dir

    dev = torch.device(args.device) if args.device else get_device()
    hw = apply_runtime_tuning(dev)
    opt = compute_optimal_chunks(cfg.glevel, cfg.nz, cfg.crm_nx, cfg.crm_ny, hw=hw)
    # Use at least the hardware-optimal single-pass chunk, and cap if host memory is tighter
    if opt["crm_single_pass"]:
        cfg.crm_col_chunk = max(cfg.crm_col_chunk, int(opt["crm_col_chunk"]))
    else:
        cfg.crm_col_chunk = min(cfg.crm_col_chunk, int(opt["crm_col_chunk"]))
    if opt["rad_single_pass"]:
        cfg.rad_col_chunk = max(cfg.rad_col_chunk, int(opt["rad_col_chunk"]))
    else:
        cfg.rad_col_chunk = min(cfg.rad_col_chunk, int(opt["rad_col_chunk"]))

    out_root = Path(out_dir_str).expanduser() if out_dir_str else get_output_dir(f"glevel{cfg.glevel}_{years}yr")
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Initializing AOBA Glevel-{cfg.glevel} ({cfg.crm_nx}x{cfg.crm_ny}, dx={cfg.crm_dx:.0f} m, "
        f"nz={cfg.nz}, dt={cfg.dt_host:.0f} s) on {dev} "
        f"[{hw.chip_name}, {hw.gpu_cores} GPUc, {hw.perf_cores} P-cores, {hw.total_ram_gb:.0f}GB RAM | "
        f"chunks crm={cfg.crm_col_chunk} rad={cfg.rad_col_chunk}]...",
        flush=True,
    )
    model = init_aoba_model(cfg, device=dev)

    mesh_zarr = out_root / "mesh.zarr"
    if not mesh_zarr.exists():
        save_zarr(export_scvt_mesh_xarray(model), mesh_zarr, overwrite=True)

    start_year = 1
    if not args.no_resume:
        for y in range(years, 0, -1):
            y_dir = out_root / f"year_{y:04d}"
            if (y_dir / "_COMPLETE").exists() and (y_dir / "restart.pt").exists():
                print(f"Resuming from completed Year {y:04d} ({y_dir / 'restart.pt'})...", flush=True)
                _load_restart_into_model(model, y_dir / "restart.pt", dev)
                start_year = y + 1
                break

    # Expose initial T2 / OLR / LAI + Earth2014 continents immediately for Tab 5 live globes
    _write_live_preview_atomic(
        model, out_root / "live_preview.npz", start_year, 0.0, float((start_year - 1) * days_per_year), 0.0
    )

    steps_per_day = max(int(round(C.SECONDS_PER_DAY / cfg.dt_host)), 1)
    steps_per_year = max(int(round(days_per_year * C.SECONDS_PER_DAY / cfg.dt_host)), 12)
    n_mean_periods = max(int(out_cfg.monthly_mean_periods_per_year), 1)
    mean_bounds = [int(round(m * steps_per_year / float(n_mean_periods))) for m in range(n_mean_periods + 1)]
    n_snap_periods = max(int(out_cfg.snapshots_periods_per_year), 1)
    snap_steps = {int(round(s * steps_per_year / float(n_snap_periods))) for s in range(1, n_snap_periods + 1)}
    sample_stride = max(int(round(out_cfg.monthly_mean_sample_hours * 3600.0 / cfg.dt_host)), 1)
    csv_stride_steps = max(int(out_cfg.daily_csv_interval_days) * steps_per_day, 1)

    import shutil

    single_partition = out_cfg.zarr_dir_partition == "single"
    if single_partition and start_year == 1:
        for p in (out_root / "monthly.zarr", out_root / "snapshots.zarr"):
            if p.exists():
                shutil.rmtree(p)

    for year in range(start_year, years + 1):
        y_dir = out_root / f"year_{year:04d}"
        y_dir.mkdir(parents=True, exist_ok=True)
        monthly_zarr = (out_root / "monthly.zarr") if single_partition else (y_dir / "monthly.zarr")
        snap_zarr = (out_root / "snapshots.zarr") if single_partition else (y_dir / "snapshots.zarr")
        csv_path = y_dir / "daily_global.csv"

        # Clean partial current-year Zarr only in yearly-partition mode
        if not single_partition:
            for p in (monthly_zarr, snap_zarr):
                if p.exists():
                    shutil.rmtree(p)

        if out_cfg.daily_csv_enabled:
            with open(csv_path, "w", newline="") as f_csv:
                csv.writer(f_csv).writerow([
                    "year", "day_in_yr", "sim_days", "ts_k", "olr_wm2", "asr_wm2",
                    "net_toa_wm2", "precip_mm_d", "gpp_umol", "lai", "seaice_mkm2", "wall_s_per_day",
                ])

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

            if out_cfg.monthly_mean_enabled and (
                step_in_yr % sample_stride == 0
                or (m_idx + 1 < len(mean_bounds) and step_in_yr >= mean_bounds[m_idx + 1])
            ):
                sample: dict[str, torch.Tensor] = {}
                if out_cfg.monthly_mean_include_2d:
                    rain = model.rad_cache.get("rain", torch.zeros_like(model.sfc.t_ocean_ml))
                    snow = model.rad_cache.get("snow", torch.zeros_like(model.sfc.t_ocean_ml))
                    evap = model.rad_cache.get("evap", torch.zeros_like(model.sfc.t_ocean_ml))
                    gpp_c = model.rad_cache.get("gpp", torch.zeros_like(model.sfc.t_ocean_ml))
                    olr_c = model.rad_cache.get("olr", torch.zeros_like(model.sfc.t_ocean_ml))
                    asr_c = model.rad_cache.get("asr", torch.zeros_like(model.sfc.t_ocean_ml))
                    sample.update({
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
                    })
                    if model.icesheet is not None:
                        sample["h_ice"] = model.icesheet.h_ice
                        sample["z_topo_dyn"] = model.mesh.z_topo
                if out_cfg.monthly_mean_include_3d:
                    u_c, v_c = reconstruct_cell_winds(model.mesh, model.atmos.u_edge)
                    pres_c, _, temp_c = diagnose_pressure_temperature(model.atmos.rho_d, model.atmos.theta_m, model.atmos.q_v)
                    sample.update({
                        "temp": temp_c,
                        "pres": pres_c,
                        "q_v": model.atmos.q_v,
                        "u_zonal": u_c,
                        "v_merid": v_c,
                    })
                if out_cfg.monthly_mean_include_subgrid_2d:
                    sample.update({k: val for k, (val, _) in diagnose_subgrid_2d_fields(model).items()})

                if m_cnt == 0:
                    m_sums = {k: v.clone() for k, v in sample.items()}
                else:
                    for k, v in sample.items():
                        m_sums[k] += v
                m_cnt += 1

            if step_in_yr % csv_stride_steps == 0 or step_in_yr == steps_per_year:
                if dev.type == "mps":
                    torch.mps.synchronize()
                wall_day = time.perf_counter() - t_day0
                t_day0 = time.perf_counter()
                inv_d = 1.0 / float(max(day_cnt, 1))
                day_num = step_in_yr / float(steps_per_day)
                ts_m = day_accum["ts"] * inv_d
                olr_m = day_accum["olr"] * inv_d
                asr_m = day_accum["asr"] * inv_d
                if out_cfg.daily_csv_enabled:
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
                _write_live_preview_atomic(
                    model, out_root / "live_preview.npz", year, day_num, diag.sim_time_days, wall_day
                )

            if out_cfg.monthly_mean_enabled and m_idx + 1 < len(mean_bounds) and step_in_yr >= mean_bounds[m_idx + 1]:
                if m_cnt > 0 and (
                    out_cfg.monthly_mean_include_2d
                    or out_cfg.monthly_mean_include_3d
                    or out_cfg.monthly_mean_include_subgrid_2d
                ):
                    append_zarr(
                        _build_monthly_mean_xarray(
                            model,
                            m_sums,
                            m_cnt,
                            include_2d=out_cfg.monthly_mean_include_2d,
                            include_3d=out_cfg.monthly_mean_include_3d,
                            include_subgrid_2d=out_cfg.monthly_mean_include_subgrid_2d,
                        ),
                        monthly_zarr,
                    )
                m_sums.clear()
                m_cnt = 0
                m_idx += 1

            if out_cfg.snapshots_enabled and step_in_yr in snap_steps:
                if (
                    out_cfg.snapshots_include_2d
                    or out_cfg.snapshots_include_3d
                    or out_cfg.snapshots_include_subgrid_2d
                ):
                    append_zarr(
                        export_aoba_snapshot_xarray(
                            model,
                            include_2d=out_cfg.snapshots_include_2d,
                            include_3d=out_cfg.snapshots_include_3d,
                            include_subgrid_2d=out_cfg.snapshots_include_subgrid_2d,
                        ),
                        snap_zarr,
                    )

        if year % max(int(out_cfg.restart_interval_years), 1) == 0 or year == years:
            _save_restart_atomic(model, y_dir / "restart.pt")
            (y_dir / "_COMPLETE").touch()
        print(f"=== Completed Year {year:04d} / {years:04d} -> {y_dir} ===", flush=True)


if __name__ == "__main__":
    main()
