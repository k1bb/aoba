#!/usr/bin/env python3
"""Long-duration AOBA climate integration runner with 1-year-per-directory Zarr v3 output & restart."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
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
from aoba.calendar import days_in_year
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
from aoba.cmip7_diag import CMIP7MonthlyAccumulator, export_cmip7_fx_xarray
from aoba.ocean_ogcm import diagnose_ocean_spinup_metrics
from aoba.params import (
    OutputConfig,
    build_configs_from_dict,
    default_config_dict,
    load_aoba_toml,
)
from aoba.radiation import compute_four_band_radiation
from aoba.kernels import get_active_gpu_backend
from aoba.runtime import (
    apply_runtime_tuning,
    compute_optimal_chunks,
    get_device,
    get_output_dir,
    synchronize_device,
)
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


_STATIC_SFC_KEYS: frozenset[str] = frozenset({
    "surf_type", "z_topo", "sigma_topo", "slope_topo", "aniso_topo", "angle_topo",
})


def _remap_horizontal_tensor(
    t: torch.Tensor,
    h_idx: torch.Tensor | None,
    n_old: int,
    n_sub_old: int = 1,
    n_sub_new: int = 1,
) -> torch.Tensor:
    """Remap a 1D/2D tensor whose last axis is `n_old` or `n_old * n_sub_old` onto `h_idx` (`[n_new]`)."""
    if h_idx is None and n_sub_old == n_sub_new:
        return t
    last = int(t.shape[-1])
    if n_sub_old == 1 and n_sub_new == 1:
        if last == n_old and h_idx is not None:
            return t.index_select(-1, h_idx)
        return t
    if last == n_old * n_sub_old:
        prefix = t.shape[:-1]
        t_rs = t.reshape(*prefix, n_old, n_sub_old)
        if h_idx is not None:
            t_rs = t_rs.index_select(-2, h_idx)
        if n_sub_new != n_sub_old:
            s_idx = torch.arange(n_sub_new, dtype=torch.long) % max(n_sub_old, 1)
            t_rs = t_rs.index_select(-1, s_idx)
        n_c = int(h_idx.numel()) if h_idx is not None else n_old
        return t_rs.reshape(*prefix, n_c * n_sub_new)
    return t


def _restore_surface(
    sfc_target,
    payload: dict,
    device: torch.device,
    *,
    h_idx: torch.Tensor | None = None,
    n_old: int = 0,
    n_sub_old: int = 1,
    n_sub_new: int = 1,
) -> None:
    """Restore serialized `SurfaceState` tensors onto `device` (with optional horizontal/subgrid remap)."""
    do_remap = (h_idx is not None) or (n_sub_old != n_sub_new)
    for k, v in payload.items():
        if k == "bio":
            for bk, bv in v.items():
                if isinstance(bv, torch.Tensor):
                    bv_use = _remap_horizontal_tensor(bv.cpu(), h_idx, n_old, n_sub_old, n_sub_new) if do_remap else bv
                    setattr(sfc_target.bio, bk, bv_use.to(device=device))
                else:
                    setattr(sfc_target.bio, bk, bv)
        elif do_remap and k in _STATIC_SFC_KEYS:
            # Keep active target mesh's static topography and surface-type masks intact
            continue
        elif isinstance(v, torch.Tensor):
            v_use = _remap_horizontal_tensor(v.cpu(), h_idx, n_old, n_sub_old, n_sub_new) if do_remap else v
            setattr(sfc_target, k, v_use.to(device=device))
        else:
            setattr(sfc_target, k, v)
    # Backward-compatible warm-start when resuming from pre-3-source restart checkpoints
    if "t_leaf_sun" not in payload or sfc_target.t_leaf_sun is None:
        sfc_target.t_leaf_sun = sfc_target.t_canopy.clone()
    if "t_leaf_sh" not in payload or sfc_target.t_leaf_sh is None:
        sfc_target.t_leaf_sh = sfc_target.t_canopy.clone()
    if "t_gnd_skin" not in payload or sfc_target.t_gnd_skin is None:
        sfc_target.t_gnd_skin = sfc_target.t_soil[0].clone()
    # Clamp restored soil ice to available pore space
    from aoba.surface import SOIL_POROSITY

    w_ice_cap = torch.clamp(SOIL_POROSITY - sfc_target.w_liq_soil, min=0.0)
    sfc_target.w_ice_soil = torch.clamp(sfc_target.w_ice_soil, min=0.0).minimum(w_ice_cap)
    # Backward-compatible warm-start for evergreen/deciduous leaf-habit & leaf-angle fields
    bio_payload = payload.get("bio", {}) or {}
    if "f_evg" not in bio_payload or sfc_target.bio.f_evg is None:
        from aoba.surface import SURF_LAND
        veg_f = (sfc_target.surf_type == SURF_LAND).to(sfc_target.bio.lai.dtype)
        sfc_target.bio.f_evg = 0.5 * veg_f
    if "lai_evg" not in bio_payload or sfc_target.bio.lai_evg is None:
        sfc_target.bio.lai_evg = sfc_target.bio.f_evg * sfc_target.bio.lai
    if "lai_dec" not in bio_payload or sfc_target.bio.lai_dec is None:
        sfc_target.bio.lai_dec = (1.0 - sfc_target.bio.f_evg) * sfc_target.bio.lai
    if "ema_offseason_temp" not in bio_payload or sfc_target.bio.ema_offseason_temp is None:
        sfc_target.bio.ema_offseason_temp = sfc_target.t_canopy.clone()
    if "ann_min_temp_k" not in bio_payload or sfc_target.bio.ann_min_temp_k is None:
        sfc_target.bio.ann_min_temp_k = sfc_target.t_canopy.clone()
    if "ema_tmin_k" not in bio_payload or sfc_target.bio.ema_tmin_k is None:
        sfc_target.bio.ema_tmin_k = sfc_target.t_canopy.clone()
    if "ema_ann_a0_fav" not in bio_payload or sfc_target.bio.ema_ann_a0_fav is None:
        sfc_target.bio.ema_ann_a0_fav = sfc_target.bio.ema_ann_a0.clone()
    if "ema_temp_home" not in bio_payload or sfc_target.bio.ema_temp_home is None:
        sfc_target.bio.ema_temp_home = sfc_target.bio.ema_temp_mid.clone()
    if "n_avail_fac" not in bio_payload or sfc_target.bio.n_avail_fac is None:
        sfc_target.bio.n_avail_fac = torch.ones_like(sfc_target.bio.lai)
    if "chi_leaf" not in bio_payload or sfc_target.bio.chi_leaf is None:
        sfc_target.bio.chi_leaf = torch.zeros_like(sfc_target.bio.lai)


def _inspect_nonfinite_state(model: AOBAState) -> list[str]:
    """Return human-readable diagnostics for any non-finite prognostic tensors (empty if all finite)."""
    mesh = model.mesh
    nc = int(mesh.n_cells)
    lat_deg = np.rad2deg(mesh.lat_cell.detach().cpu().numpy())
    lon_deg = np.rad2deg(mesh.lon_cell.detach().cpu().numpy())
    st_np = mesh.surf_type.detach().cpu().numpy()

    checks: list[tuple[str, torch.Tensor | None]] = [
        ("atmos.rho_d", model.atmos.rho_d),
        ("atmos.theta_m", model.atmos.theta_m),
        ("atmos.u_edge", model.atmos.u_edge),
        ("atmos.w", model.atmos.w),
        ("atmos.q_n", model.atmos.q_n),
        ("crm.rho_d", model.crm.rho_d),
        ("crm.theta_m", model.crm.theta_m),
        ("crm.q_n", model.crm.q_n),
        ("sfc.t_canopy", model.sfc.t_canopy),
        ("sfc.t_ocean_ml", model.sfc.t_ocean_ml),
        ("sfc.m_seaice", model.sfc.m_seaice),
        ("sfc.bio.lai", model.sfc.bio.lai),
    ]
    if model.sfc_sub is not None:
        checks.extend([
            ("sfc_sub.t_canopy", model.sfc_sub.t_canopy),
            ("sfc_sub.t_leaf_sun", model.sfc_sub.t_leaf_sun),
            ("sfc_sub.t_gnd_skin", model.sfc_sub.t_gnd_skin),
            ("sfc_sub.t_soil", model.sfc_sub.t_soil),
            ("sfc_sub.t_ocean_ml", model.sfc_sub.t_ocean_ml),
            ("sfc_sub.m_seaice", model.sfc_sub.m_seaice),
            ("sfc_sub.bio.lai", model.sfc_sub.bio.lai),
        ])
    if model.ogcm is not None:
        checks.extend([
            ("ogcm.temp", model.ogcm.temp),
            ("ogcm.sal", model.ogcm.sal),
            ("ogcm.u_edge", model.ogcm.u_edge),
            ("ogcm.eta", model.ogcm.eta),
            ("ogcm.u_baro", model.ogcm.u_baro),
        ])
        if model.ogcm.seaice_evp is not None:
            checks.extend([
                ("seaice_evp.a_cat", model.ogcm.seaice_evp.a_cat),
                ("seaice_evp.v_cat", model.ogcm.seaice_evp.v_cat),
                ("seaice_evp.u_ice_e", model.ogcm.seaice_evp.u_ice_e),
            ])
        if model.ogcm.bgc is not None:
            checks.extend([
                ("bgc.no3", model.ogcm.bgc.no3),
                ("bgc.dic", model.ogcm.bgc.dic),
                ("bgc.phy", model.ogcm.bgc.phy),
            ])
    if model.land_carbon is not None:
        checks.append(("land_carbon.c_leaf", model.land_carbon.c_leaf))
        if getattr(model.land_carbon, "n_min", None) is not None:
            checks.append(("land_carbon.n_min", model.land_carbon.n_min))
    if model.aerosols is not None:
        checks.append(("aerosols.q_ss", model.aerosols.q_ss))
    if model.ozone is not None:
        checks.append(("ozone.o3_vmr", model.ozone.o3_vmr))

    from aoba.coupled_model import resolve_crm_sfc_tiles

    n_sub = resolve_crm_sfc_tiles(model.cfg)
    bad_lines: list[str] = []
    for name, ten in checks:
        if ten is None or not isinstance(ten, torch.Tensor) or not ten.is_floating_point():
            continue
        bad_mask = ~torch.isfinite(ten)
        n_bad = int(bad_mask.sum().item())
        if n_bad > 0:
            loc_str = ""
            # Map first up to 5 offending cells onto (lat, lon, surf_type) when last dim matches nc or nc*n_sub
            if ten.shape[-1] == nc:
                cell_bad = bad_mask.reshape(-1, nc).any(dim=0).nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                if isinstance(cell_bad, int):
                    cell_bad = [cell_bad]
                locs = [f"c={c}(lat={lat_deg[c]:+.1f},lon={lon_deg[c]:+.1f},st={int(st_np[c])})" for c in cell_bad[:5]]
                loc_str = " | cells: " + ", ".join(locs)
            elif ten.shape[-1] == nc * n_sub:
                sub_bad = bad_mask.reshape(-1, nc, n_sub).any(dim=(0, 2)).nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                if isinstance(sub_bad, int):
                    sub_bad = [sub_bad]
                locs = [f"c={c}(lat={lat_deg[c]:+.1f},lon={lon_deg[c]:+.1f},st={int(st_np[c])})" for c in sub_bad[:5]]
                loc_str = " | cells: " + ", ".join(locs)
            bad_lines.append(f"  - {name}: {n_bad}/{ten.numel()} non-finite{loc_str}")
    return bad_lines


def _abort_if_nonfinite(
    model: AOBAState,
    year: int,
    step_in_yr: int,
    date_str: str,
    reason: str,
) -> None:
    """Dump offending tensors/cells and abort immediately without overwriting any restart checkpoint."""
    bad = _inspect_nonfinite_state(model)
    print(
        f"\n[ABORT: NON-FINITE STATE DETECTED] Y{year:04d} step_in_yr={step_in_yr} "
        f"(global_step={model.step_count}, date={date_str}) | reason={reason}",
        flush=True,
    )
    if bad:
        for line in bad:
            print(line, flush=True)
    else:
        print("  (Prognostic core tensors finite; non-finite originated in step diagnostics/radiation.)", flush=True)
    print("  [Guard] Refusing to overwrite restart_mid.pt / restart.pt. Exiting with code 2.", flush=True)
    sys.exit(2)


def _save_restart_atomic(model: AOBAState, path: Path, *, extra: dict[str, Any] | None = None) -> None:
    """Save complete prognostic `AOBAState` (including subgrid `sfc_sub`) via atomic temp file + rename."""
    bad = _inspect_nonfinite_state(model)
    if bad:
        date_s = model.calendar.date_at(model.sim_time_sec).iso_datetime
        print(
            f"\n[ABORT: REFUSING TO SAVE NON-FINITE CHECKPOINT -> {path}] "
            f"(global_step={model.step_count}, date={date_s})",
            flush=True,
        )
        for line in bad:
            print(line, flush=True)
        sys.exit(2)
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
        "ocn_sfc": _serialize_surface(model.ocn_sfc) if model.ocn_sfc is not None else None,
        "ocn": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.ocn.__dict__.items()},
        "ogcm": (
            {
                k: (
                    {bk: bv.detach().cpu() if isinstance(bv, torch.Tensor) else bv for bk, bv in v.__dict__.items()}
                    if k in ("bgc", "seaice_evp") and v is not None
                    else (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                )
                for k, v in model.ogcm.__dict__.items()
                if k != "_static_cache"
            }
            if model.ogcm is not None
            else None
        ),
        "river": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.river.__dict__.items()}
            if model.river is not None
            else None
        ),
        "lakes": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.lakes.__dict__.items()}
            if getattr(model, "lakes", None) is not None
            else None
        ),
        "explicit_lightning": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.explicit_lightning.__dict__.items()}
            if getattr(model, "explicit_lightning", None) is not None
            else None
        ),
        "lightning_distill": (
            {
                "net_state": {k: v.detach().cpu() for k, v in model.lightning_distill.net.state_dict().items()},
                "steps_seen": int(model.lightning_distill.steps_seen),
                "grad_updates": int(model.lightning_distill.grad_updates),
                "ema_loss": float(model.lightning_distill.ema_loss),
                "ema_val_corr": float(model.lightning_distill.ema_val_corr),
            }
            if getattr(model, "lightning_distill", None) is not None
            else None
        ),
        "crm_distill": (
            {
                "net_state": {k: v.detach().cpu() for k, v in model.crm_distill.net.state_dict().items()},
                "steps_seen": int(model.crm_distill.steps_seen),
                "grad_updates": int(model.crm_distill.grad_updates),
                "ema_loss": float(model.crm_distill.ema_loss),
                "ema_val_rmse_thm": float(model.crm_distill.ema_val_rmse_thm),
                "ema_val_rmse_qn": float(model.crm_distill.ema_val_rmse_qn),
            }
            if getattr(model, "crm_distill", None) is not None
            else None
        ),
        "rad_distill": (
            {
                "net_state": {k: v.detach().cpu() for k, v in model.rad_distill.net.state_dict().items()},
                "steps_seen": int(model.rad_distill.steps_seen),
                "grad_updates": int(model.rad_distill.grad_updates),
                "ema_loss": float(model.rad_distill.ema_loss),
                "ema_val_rmse_olr": float(model.rad_distill.ema_val_rmse_olr),
                "ema_val_rmse_asr": float(model.rad_distill.ema_val_rmse_asr),
            }
            if getattr(model, "rad_distill", None) is not None
            else None
        ),
        "sparse_crm_pool": (
            {
                "crm": {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in model.sparse_crm_pool.crm.__dict__.items()
                },
                "q_rad": model.sparse_crm_pool.q_rad.detach().cpu(),
                "slot_cell_ids": model.sparse_crm_pool.slot_cell_ids.detach().cpu(),
                "ny_full": int(model.sparse_crm_pool.ny_full),
                "nx_full": int(model.sparse_crm_pool.nx_full),
                "n_pool": int(model.sparse_crm_pool.n_pool),
                "q_rad_valid": bool(model.sparse_crm_pool.q_rad_valid),
            }
            if getattr(model, "sparse_crm_pool", None) is not None
            else None
        ),
        "land_carbon": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.land_carbon.__dict__.items()}
            if model.land_carbon is not None
            else None
        ),
        "aerosols": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.aerosols.__dict__.items()}
            if model.aerosols is not None
            else None
        ),
        "ozone": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.ozone.__dict__.items()}
            if model.ozone is not None
            else None
        ),
        "meso_nox": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.meso_nox.__dict__.items()}
            if getattr(model, "meso_nox", None) is not None
            else None
        ),
        "trace_gas": (
            {k: v for k, v in model.trace_gas.__dict__.items()}
            if model.trace_gas is not None
            else None
        ),
        "weathering": (
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.weathering.__dict__.items()}
            if model.weathering is not None
            else None
        ),
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
        "mesh_coords": (
            lambda u_c_s, v_c_s: {
                "atm_xyz": model.mesh.xyz_cell.detach().cpu(),
                "atm_st": model.mesh.surf_type.detach().cpu(),
                "atm_u_cell": u_c_s.detach().cpu(),
                "atm_v_cell": v_c_s.detach().cpu(),
                "ocn_xyz": (model.ocn_mesh if model.ocn_mesh is not None else model.mesh).xyz_cell.detach().cpu(),
                "ocn_z_c": model.ogcm.z_c.detach().cpu() if model.ogcm is not None else None,
                "ocn_wet0": model.ogcm.wet_c[0].detach().cpu() if model.ogcm is not None else None,
            }
        )(*__import__("aoba.scvt_mesh", fromlist=["reconstruct_cell_winds"]).reconstruct_cell_winds(model.mesh, model.atmos.u_edge)),
        "rad_cache": {
            k: v.detach().cpu()
            for k, v in model.rad_cache.items()
            if isinstance(v, torch.Tensor) and not k.startswith("_")
        },
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _interp_vertical_batch(
    z_old: torch.Tensor,
    val_old: torch.Tensor,
    z_new: torch.Tensor,
) -> torch.Tensor:
    """Vectorized 1D piecewise-linear vertical interpolation `[nz_old, N] -> [nz_new, N]`."""
    nz_old = z_old.shape[0]
    zo_t = z_old.transpose(0, 1).contiguous()   # [N, nz_old]
    vo_t = val_old.transpose(0, 1).contiguous() # [N, nz_old]
    zn_t = z_new.transpose(0, 1).contiguous()   # [N, nz_new]

    idx_hi = torch.searchsorted(zo_t, zn_t).clamp(1, nz_old - 1)
    idx_lo = idx_hi - 1
    z0 = torch.gather(zo_t, 1, idx_lo)
    z1 = torch.gather(zo_t, 1, idx_hi)
    v0 = torch.gather(vo_t, 1, idx_lo)
    v1 = torch.gather(vo_t, 1, idx_hi)
    w = ((zn_t - z0) / torch.clamp(z1 - z0, min=1.0e-3)).clamp(0.0, 1.0)
    return ((1.0 - w) * v0 + w * v1).transpose(0, 1).contiguous()


def _stratosphere_temp_extension(z_new: torch.Tensor, z_top_old: torch.Tensor, t_top_old: torch.Tensor) -> torch.Tensor:
    """Continuous U.S. Standard Atmosphere (1976) temperature extrapolation above `z_top_old`."""
    def _us76_offset(z: torch.Tensor) -> torch.Tensor:
        return torch.where(
            z <= 20000.0,
            torch.zeros_like(z),
            torch.where(
                z <= 32000.0,
                0.0010 * (z - 20000.0),
                torch.where(
                    z <= 47000.0,
                    12.0 + 0.0028 * (z - 32000.0),
                    torch.where(
                        z <= 51000.0,
                        torch.full_like(z, 54.0),
                        torch.where(
                            z <= 71000.0,
                            54.0 - 0.0028 * (z - 51000.0),
                            torch.clamp(-2.0 - 0.0020 * (z - 71000.0), min=-32.0),
                        ),
                    ),
                ),
            ),
        )

    dt_ext = _us76_offset(z_new) - _us76_offset(z_top_old)
    return torch.clamp(t_top_old + dt_ext, min=180.0, max=285.0)


def _remap_thermo_columns(
    z_old: torch.Tensor,
    rho_old: torch.Tensor,
    thm_old: torch.Tensor,
    qv_old: torch.Tensor,
    qc_old: torch.Tensor,
    qi_old: torch.Tensor,
    qp_old: torch.Tensor,
    z_new: torch.Tensor,
    *,
    rebalance_hydrostatic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hydrostatically consistent vertical remap + downward/upward extension `[nz_old, N] -> [nz_new, N]`."""
    zo = z_old.to(torch.float64)
    zn = z_new.to(torch.float64)
    p_old, _, t_old = diagnose_pressure_temperature(
        rho_old.to(torch.float64), thm_old.to(torch.float64), qv_old.to(torch.float64)
    )
    ln_p_interp = _interp_vertical_batch(zo, torch.log(torch.clamp(p_old, min=1.0e-3)), zn)
    t_interp = _interp_vertical_batch(zo, t_old, zn)
    qv_interp = torch.clamp(_interp_vertical_batch(zo, qv_old.to(torch.float64), zn), min=1.0e-7)
    qc_interp = torch.clamp(_interp_vertical_batch(zo, qc_old.to(torch.float64), zn), min=0.0)
    qi_interp = torch.clamp(_interp_vertical_batch(zo, qi_old.to(torch.float64), zn), min=0.0)
    qp_interp = torch.clamp(_interp_vertical_batch(zo, qp_old.to(torch.float64), zn), min=0.0)

    z_bot_old = zo[0:1].expand_as(zn)
    z_top_old = zo[-1:].expand_as(zn)
    below = zn < z_bot_old
    above = zn > z_top_old

    # Standard 6.5 K/km tropospheric lapse extension below coarse terrain (prevents valley shockwaves on AG4->AG6)
    t_below = torch.clamp(t_old[0:1].expand_as(zn) + 0.0065 * (z_bot_old - zn), min=180.0, max=330.0)
    t_new = torch.where(below, t_below, t_interp)
    t_new = torch.where(above, _stratosphere_temp_extension(zn, z_top_old, t_old[-1:].expand_as(zn)), t_new)

    qv_top_dry = torch.clamp(qv_old[-1:].to(torch.float64), min=1.0e-7, max=4.0e-6).expand_as(zn)
    qv_new = torch.where(above, qv_top_dry, qv_interp)
    qc_new = torch.where(above | below, torch.zeros_like(qc_interp), qc_interp)
    qi_new = torch.where(above | below, torch.zeros_like(qi_interp), qi_interp)
    qp_new = torch.where(above | below, torch.zeros_like(qp_interp), qp_interp)
    qn_new = qv_new + qc_new + qi_new

    p_new = torch.exp(ln_p_interp)
    nz_new = zn.shape[0]
    tv_new = t_new * (1.0 + (C.RV_OVER_RD - 1.0) * qv_new)

    # 1. Integrate hydrostatically DOWNWARD into valleys below z_bot_old (critical for AG4 -> AG6 mountain slopes!)
    for k in range(nz_new - 2, -1, -1):
        bl_k = below[k]
        if bl_k.any():
            dz_k = torch.clamp(zn[k + 1] - zn[k], min=1.0)
            tv_mid = 0.5 * (tv_new[k] + tv_new[k + 1])
            p_hyd_dn = p_new[k + 1] * torch.exp(C.GRAVITY * dz_k / (C.R_D * tv_mid))
            p_new[k] = torch.where(bl_k, torch.clamp(p_hyd_dn, min=1.0e-3, max=1.15e5), p_new[k])

    # 2. Integrate hydrostatically UPWARD above z_top_old (or full-column discrete balance on horizontal grid change)
    for k in range(1, nz_new):
        do_k = torch.ones_like(above[k]) if rebalance_hydrostatic else above[k]
        if do_k.any():
            dz_k = torch.clamp(zn[k] - zn[k - 1], min=1.0)
            tv_mid = 0.5 * (tv_new[k - 1] + tv_new[k])
            p_hyd = p_new[k - 1] * torch.exp(-C.GRAVITY * dz_k / (C.R_D * tv_mid))
            p_new[k] = torch.where(do_k, torch.clamp(p_hyd, min=1.0e-3), p_new[k])

    rho_new = p_new / (C.R_D * t_new * (1.0 + C.RV_OVER_RD * qv_new))
    thm_new = (C.P_REF / (C.R_D * rho_new)) * torch.pow(p_new / C.P_REF, 1.0 / C.GAMMA_D)
    return (
        rho_new.to(rho_old.dtype),
        thm_new.to(thm_old.dtype),
        qn_new.to(qv_old.dtype),
        qp_new.to(qp_old.dtype),
        qv_new.to(qv_old.dtype),
        qc_new.to(qc_old.dtype),
        qi_new.to(qi_old.dtype),
    )


def _infer_glevel_from_ncells(n_cells: int) -> int:
    """Recover icosahedral grid level `g` from `n_cells = 10 * 4^g + 2`."""
    ratio = max((int(n_cells) - 2) / 10.0, 1.0)
    return max(int(round(math.log(ratio) / math.log(4.0))), 0)


def _build_surface_aware_remap_idx(
    xyz_old: torch.Tensor,
    st_old: torch.Tensor | None,
    xyz_new: torch.Tensor,
    st_new: torch.Tensor | None,
) -> torch.Tensor:
    """Nearest-neighbor spherical remap index matching surface class (ocean/land/ice) when available."""
    from scipy.spatial import cKDTree

    xo = xyz_old.detach().cpu().numpy().astype(np.float64)
    xn = xyz_new.detach().cpu().numpy().astype(np.float64)
    tree_all = cKDTree(xo)
    _, idx_all = tree_all.query(xn, k=1)
    idx_out = np.asarray(idx_all, dtype=np.int64)

    if st_old is None or st_new is None:
        return torch.from_numpy(idx_out)

    sto = st_old.detach().cpu().numpy().astype(np.int64)
    stn = st_new.detach().cpu().numpy().astype(np.int64)
    for cls in (0, 1, 2):
        tgt_mask = stn == cls
        if not np.any(tgt_mask):
            continue
        src_ids = np.nonzero(sto == cls)[0]
        if src_ids.size == 0 and cls == 2:
            # Fallback: ice-sheet cells inherit from land cells if old grid had no ice sheets
            src_ids = np.nonzero(sto == 1)[0]
        if src_ids.size == 0:
            continue
        sub_tree = cKDTree(xo[src_ids])
        _, local_nn = sub_tree.query(xn[tgt_mask], k=1)
        idx_out[tgt_mask] = src_ids[local_nn]
    return torch.from_numpy(idx_out)


def _build_wet_ocean_remap_idx(
    xyz_old: torch.Tensor,
    wet0_old: torch.Tensor | None,
    xyz_new: torch.Tensor,
) -> torch.Tensor:
    """Nearest wet-ocean-column spherical remap index `[n_cells_new]`."""
    from scipy.spatial import cKDTree

    xo = xyz_old.detach().cpu().numpy().astype(np.float64)
    xn = xyz_new.detach().cpu().numpy().astype(np.float64)
    if wet0_old is not None:
        wet_ids = np.nonzero(wet0_old.detach().cpu().numpy() > 0.5)[0]
        if wet_ids.size > 0:
            tree_w = cKDTree(xo[wet_ids])
            _, local_nn = tree_w.query(xn, k=1)
            return torch.from_numpy(wet_ids[local_nn].astype(np.int64))
    tree_all = cKDTree(xo)
    _, idx_all = tree_all.query(xn, k=1)
    return torch.from_numpy(np.asarray(idx_all, dtype=np.int64))


def _forward_fill_wet_downward(val: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    """Propagate deepest wet-layer ocean values downward into dry bathymetry cells `[nz, N]`."""
    out = val.clone()
    nz = int(out.shape[0])
    for k in range(1, nz):
        dry = wet[k] <= 0.5
        out[k] = torch.where(dry, out[k - 1], out[k])
    return out


def _remap_ocean_3d_field(
    val_old: torch.Tensor,
    wet_old: torch.Tensor,
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    og_h_idx: torch.Tensor | None,
    wet_new: torch.Tensor,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> torch.Tensor:
    """Horizontally + vertically remap a 3D OGCM tracer `[nz_old, nc_old] -> [nz_new, nc_new]`."""
    v = val_old.cpu()
    w = wet_old.cpu()
    if og_h_idx is not None:
        v = v.index_select(1, og_h_idx)
        w = w.index_select(1, og_h_idx)
    v_fill = _forward_fill_wet_downward(v, w)
    nc_new = int(v_fill.shape[1])
    zo_2d = z_old.cpu().unsqueeze(1).expand(-1, nc_new).contiguous()
    zn_2d = z_new.cpu().unsqueeze(1).expand(-1, nc_new).contiguous()
    v_new = _interp_vertical_batch(zo_2d, v_fill, zn_2d)
    if vmin is not None or vmax is not None:
        v_new = torch.clamp(v_new, min=vmin, max=vmax)
    return v_new * (wet_new.cpu() > 0.5).to(v_new.dtype)


def _load_restart_into_model(model: AOBAState, path: Path, device: torch.device) -> dict[str, Any]:
    """Restore prognostic tensors with universal cross-grid remapping across `(glevel, nz, z_top, crm_nx/ny, ocean_glevel, ocean_nz)`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.sim_time_sec = float(ckpt["sim_time_sec"])
    model.step_count = int(ckpt["step_count"])

    ck_atmos = ckpt["atmos"]
    ck_crm = ckpt["crm"]
    mesh_coords = ckpt.get("mesh_coords") or {}

    nz_old = int(ck_atmos.rho_d.shape[0])
    nz_new = int(model.atmos.rho_d.shape[0])
    nc_old = int(ck_atmos.rho_d.shape[1])
    nc_new = int(model.atmos.rho_d.shape[1])
    _, _, ny_old, nx_old = ck_crm.rho_d.shape
    _, _, ny_new, nx_new = model.crm.rho_d.shape
    n_sub_old = int(ny_old * nx_old)
    n_sub_new = int(ny_new * nx_new)

    z_top_old_val = float(ck_atmos.z_w[-1].mean().item())
    z_top_new_val = float(model.atmos.z_w[-1].mean().item())
    needs_hremap = nc_old != nc_new
    needs_subremap = (ny_old != ny_new) or (nx_old != nx_new)
    needs_vremap = (nz_old != nz_new) or (abs(z_top_old_val - z_top_new_val) > 50.0)
    needs_atm_remap = needs_hremap or needs_vremap or needs_subremap

    # Build spherical surface-type-aware remap index for atmosphere if horizontal mesh changed
    atm_h_idx: torch.Tensor | None = None
    old_atm_xyz: torch.Tensor | None = mesh_coords.get("atm_xyz")
    old_atm_st: torch.Tensor | None = mesh_coords.get("atm_st")
    if old_atm_st is None and isinstance(ckpt.get("sfc"), dict):
        old_atm_st = ckpt["sfc"].get("surf_type")
    if old_atm_st is None and isinstance(ckpt.get("mesh_dynamic"), dict):
        old_atm_st = ckpt["mesh_dynamic"].get("surf_type")

    old_atm_mesh_rebuilt = None
    if needs_hremap:
        if old_atm_xyz is None:
            from aoba.scvt_mesh import build_scvt_mesh

            g_old = _infer_glevel_from_ncells(nc_old)
            old_atm_mesh_rebuilt = build_scvt_mesh(g_old, use_earth2014=False, device=torch.device("cpu"))
            old_atm_xyz = old_atm_mesh_rebuilt.xyz_cell.cpu()
        atm_h_idx = _build_surface_aware_remap_idx(
            old_atm_xyz, old_atm_st, model.mesh.xyz_cell.cpu(), model.mesh.surf_type.cpu()
        )

    if not needs_atm_remap:
        for k, v in ck_atmos.__dict__.items():
            setattr(model.atmos, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in ck_crm.__dict__.items():
            setattr(model.crm, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
        model.rad_cache = {
            k: v.to(device=device)
            for k, v in ckpt["rad_cache"].items()
            if isinstance(v, torch.Tensor) and not k.startswith("_")
        }
    else:
        print(
            f"  [Universal Grid Remap] Seamlessly connecting restart "
            f"(nc={nc_old}, nz={nz_old}, top={z_top_old_val/1000:.1f} km, crm={ny_old}x{nx_old}) "
            f"-> active grid (nc={nc_new}, nz={nz_new}, top={z_top_new_val/1000:.1f} km, crm={ny_new}x{nx_new})...",
            flush=True,
        )
        z_old_c = ck_atmos.z_c.cpu()
        z_new_c = model.atmos.z_c.cpu()
        z_old_w = ck_atmos.z_w.cpu()
        z_new_w = model.atmos.z_w.cpu()

        def _h_atm(t: torch.Tensor) -> torch.Tensor:
            t_cpu = t.cpu()
            return t_cpu.index_select(1, atm_h_idx) if atm_h_idx is not None else t_cpu

        z_old_c_h = _h_atm(z_old_c)
        z_old_w_h = _h_atm(z_old_w)

        # 1. Host thermodynamics (hydrostatic remap + downward/upward extension; full rebalance on horizontal change)
        rho_n, thm_n, qn_n, qp_n, qv_n, qc_n, qi_n = _remap_thermo_columns(
            z_old_c_h,
            _h_atm(ck_atmos.rho_d),
            _h_atm(ck_atmos.theta_m),
            _h_atm(ck_atmos.q_v),
            _h_atm(ck_atmos.q_c),
            _h_atm(ck_atmos.q_i),
            _h_atm(ck_atmos.q_p),
            z_new_c,
            rebalance_hydrostatic=bool(needs_hremap),
        )
        model.atmos.rho_d = rho_n.to(device=device)
        model.atmos.theta_m = thm_n.to(device=device)
        model.atmos.q_n = qn_n.to(device=device)
        model.atmos.q_p = qp_n.to(device=device)
        model.atmos.q_v = qv_n.to(device=device)
        model.atmos.q_c = qc_n.to(device=device)
        model.atmos.q_i = qi_n.to(device=device)

        # Host edge winds: either direct edge vertical remap or cell-vector remap -> edge projection
        if not needs_hremap:
            if needs_vremap:
                c0 = model.mesh.cells_on_edge[:, 0].cpu()
                c1 = model.mesh.cells_on_edge[:, 1].cpu()
                ze_old = 0.5 * (z_old_c[:, c0] + z_old_c[:, c1])
                ze_new = 0.5 * (z_new_c[:, c0] + z_new_c[:, c1])
                ue_interp = _interp_vertical_batch(ze_old, ck_atmos.u_edge.cpu(), ze_new)
                taper_e = torch.exp(-torch.clamp(ze_new - ze_old[-1:], min=0.0) / 12000.0)
                model.atmos.u_edge = (ue_interp * taper_e).to(device=device)
            else:
                model.atmos.u_edge = ck_atmos.u_edge.to(device=device)
        else:
            from aoba.scvt_mesh import build_scvt_mesh, project_cell_vectors_to_edges, reconstruct_cell_winds

            u_c_old = mesh_coords.get("atm_u_cell")
            v_c_old = mesh_coords.get("atm_v_cell")
            if u_c_old is None or v_c_old is None:
                if old_atm_mesh_rebuilt is None:
                    g_old = _infer_glevel_from_ncells(nc_old)
                    old_atm_mesh_rebuilt = build_scvt_mesh(g_old, use_earth2014=False, device=torch.device("cpu"))
                u_c_old, v_c_old = reconstruct_cell_winds(old_atm_mesh_rebuilt, ck_atmos.u_edge.cpu())
            u_c_h = u_c_old.cpu().index_select(1, atm_h_idx)
            v_c_h = v_c_old.cpu().index_select(1, atm_h_idx)
            taper_c = torch.exp(-torch.clamp(z_new_c - z_old_c_h[-1:], min=0.0) / 12000.0)
            u_c_new = (_interp_vertical_batch(z_old_c_h, u_c_h, z_new_c) * taper_c).to(device=device)
            v_c_new = (_interp_vertical_batch(z_old_c_h, v_c_h, z_new_c) * taper_c).to(device=device)
            model.atmos.u_edge = project_cell_vectors_to_edges(model.mesh, u_c_new, v_c_new)

        if needs_hremap:
            model.atmos.w = torch.zeros_like(model.atmos.w)
        else:
            w_interp = _interp_vertical_batch(z_old_w_h, _h_atm(ck_atmos.w), z_new_w)
            model.atmos.w = torch.where(z_new_w > z_old_w_h[-1:], torch.zeros_like(w_interp), w_interp).to(device=device)
        if getattr(ck_atmos, "q_dust_fine", None) is not None and getattr(ck_atmos, "q_dust_coarse", None) is not None:
            qdf_i = torch.clamp(_interp_vertical_batch(z_old_c_h, _h_atm(ck_atmos.q_dust_fine), z_new_c), min=0.0)
            qdc_i = torch.clamp(_interp_vertical_batch(z_old_c_h, _h_atm(ck_atmos.q_dust_coarse), z_new_c), min=0.0)
            model.atmos.q_dust_fine = torch.where(z_new_c > z_old_c_h[-1:], torch.zeros_like(qdf_i), qdf_i).to(device=device)
            model.atmos.q_dust_coarse = torch.where(z_new_c > z_old_c_h[-1:], torch.zeros_like(qdc_i), qdc_i).to(device=device)
            model.atmos.q_dust = model.atmos.q_dust_fine + model.atmos.q_dust_coarse
        elif getattr(ck_atmos, "q_dust", None) is not None:
            from aoba.dust import DUST_LEGACY_RESTART_FINE_FRAC

            qd_i = torch.clamp(_interp_vertical_batch(z_old_c_h, _h_atm(ck_atmos.q_dust), z_new_c), min=0.0)
            qd_m = torch.where(z_new_c > z_old_c_h[-1:], torch.zeros_like(qd_i), qd_i).to(device=device)
            model.atmos.q_dust = qd_m
            model.atmos.q_dust_fine = qd_m * DUST_LEGACY_RESTART_FINE_FRAC
            model.atmos.q_dust_coarse = qd_m * (1.0 - DUST_LEGACY_RESTART_FINE_FRAC)

        # 2. Embedded CRM columns [nz_old, nc_old, ny_old, nx_old] -> [nz_new, nc_new, ny_new, nx_new]
        m_sub_new = nc_new * n_sub_new
        s_idx = torch.arange(n_sub_new, dtype=torch.long) % max(n_sub_old, 1)

        def _hs_crm(t: torch.Tensor) -> torch.Tensor:
            # t: [nz_in, nc_old, ny_old, nx_old] -> [nz_in, nc_new * n_sub_new]
            t_flat = t.cpu().reshape(t.shape[0], nc_old, n_sub_old)
            if atm_h_idx is not None:
                t_flat = t_flat.index_select(1, atm_h_idx)
            if n_sub_new != n_sub_old:
                t_flat = t_flat.index_select(2, s_idx)
            return t_flat.reshape(t.shape[0], m_sub_new).contiguous()

        zo_sub = z_old_c_h.unsqueeze(-1).expand(nz_old, nc_new, n_sub_new).reshape(nz_old, m_sub_new)
        zn_sub = z_new_c.unsqueeze(-1).expand(nz_new, nc_new, n_sub_new).reshape(nz_new, m_sub_new)
        zow_sub = z_old_w_h.unsqueeze(-1).expand(nz_old + 1, nc_new, n_sub_new).reshape(nz_old + 1, m_sub_new)
        znw_sub = z_new_w.unsqueeze(-1).expand(nz_new + 1, nc_new, n_sub_new).reshape(nz_new + 1, m_sub_new)

        c_rho, c_thm, c_qn, c_qp, c_qv, c_qc, c_qi = _remap_thermo_columns(
            zo_sub,
            _hs_crm(ck_crm.rho_d),
            _hs_crm(ck_crm.theta_m),
            _hs_crm(ck_crm.q_v),
            _hs_crm(ck_crm.q_c),
            _hs_crm(ck_crm.q_i),
            _hs_crm(ck_crm.q_p),
            zn_sub,
            rebalance_hydrostatic=bool(needs_hremap),
        )
        taper_sub = torch.exp(-torch.clamp(zn_sub - zo_sub[-1:], min=0.0) / 12000.0)
        c_u = _interp_vertical_batch(zo_sub, _hs_crm(ck_crm.u), zn_sub) * taper_sub
        c_v = _interp_vertical_batch(zo_sub, _hs_crm(ck_crm.v), zn_sub) * taper_sub
        if needs_hremap:
            c_w = torch.zeros_like(znw_sub, dtype=ck_crm.w.dtype)
        else:
            c_w_i = _interp_vertical_batch(zow_sub, _hs_crm(ck_crm.w), znw_sub)
            c_w = torch.where(znw_sub > zow_sub[-1:], torch.zeros_like(c_w_i), c_w_i)
        c_qke_i = _interp_vertical_batch(zo_sub, _hs_crm(ck_crm.qke), zn_sub)
        c_qke = torch.where(zn_sub > zo_sub[-1:], torch.full_like(c_qke_i, 2.0e-4), torch.clamp(c_qke_i, min=2.0e-4))

        sh4 = (nz_new, nc_new, ny_new, nx_new)
        model.crm.rho_d = c_rho.reshape(sh4).to(device=device)
        model.crm.theta_m = c_thm.reshape(sh4).to(device=device)
        model.crm.q_n = c_qn.reshape(sh4).to(device=device)
        model.crm.q_p = c_qp.reshape(sh4).to(device=device)
        model.crm.q_v = c_qv.reshape(sh4).to(device=device)
        model.crm.q_c = c_qc.reshape(sh4).to(device=device)
        model.crm.q_i = c_qi.reshape(sh4).to(device=device)
        model.crm.u = c_u.reshape(sh4).to(device=device)
        model.crm.v = c_v.reshape(sh4).to(device=device)
        model.crm.w = c_w.reshape(nz_new + 1, nc_new, ny_new, nx_new).to(device=device)
        model.crm.qke = c_qke.reshape(sh4).to(device=device)
        _mp_scheme = str(model.cfg.microphysics_scheme)
        _keep_opt = (
            {"n_c", "n_i"} if _mp_scheme == "two_moment"
            else ({"n_c", "n_i", "q_r", "q_s", "q_g", "n_r", "n_s", "n_g"} if _mp_scheme == "six_class_2m" else set())
        )
        for opt_field in ("n_c", "n_i", "q_r", "q_s", "q_g", "n_r", "n_s", "n_g"):
            old_val = getattr(ck_crm, opt_field, None)
            if isinstance(old_val, torch.Tensor) and (opt_field in _keep_opt or getattr(model.crm, opt_field, None) is not None):
                rm = torch.clamp(_interp_vertical_batch(zo_sub, _hs_crm(old_val), zn_sub), min=0.0)
                rm = torch.where(zn_sub > zo_sub[-1:], torch.zeros_like(rm), rm)
                setattr(model.crm, opt_field, rm.reshape(sh4).to(device=device))
            elif getattr(model.crm, opt_field, None) is not None:
                setattr(model.crm, opt_field, torch.zeros(sh4, device=device, dtype=model.crm.rho_d.dtype))
            else:
                setattr(model.crm, opt_field, None)

        # Force fresh radiation evaluation on the first step with the remapped grid
        model.rad_cache = {}
        stride = max(int(model.cfg.rad_stride), 1)
        model.step_count = (model.step_count // stride) * stride
        if needs_vremap:
            ckpt["_vertical_remapped"] = True
        if needs_hremap:
            ckpt["_horizontal_remapped"] = True

    # Honor active run_config grid spacing (`crm_dx`) when switching spacing on resume
    if getattr(model, "sparse_crm_pool", None) is None:
        model.crm.dx = float(model.cfg.crm_dx)
        model.crm.dy = float(model.cfg.crm_dx)
    else:
        model.sparse_crm_pool.crm.dx = float(model.cfg.crm_dx)
        model.sparse_crm_pool.crm.dy = float(model.cfg.crm_dx)

    # 3. Surface states (`sfc`, `sfc_sub`, `ocn_sfc`)
    from aoba.coupled_model import resolve_crm_sfc_tiles

    n_sfc_new = resolve_crm_sfc_tiles(model.cfg)
    n_sfc_old = n_sub_old
    _restore_surface(model.sfc, ckpt["sfc"], device, h_idx=atm_h_idx, n_old=nc_old)
    if model.sfc_sub is not None:
        if ckpt.get("sfc_sub") is not None:
            ck_sfc_sub = ckpt["sfc_sub"]
            n_sfc_old = max(int(ck_sfc_sub["t_canopy"].shape[-1]) // max(nc_old, 1), 1)
            _restore_surface(
                model.sfc_sub,
                ck_sfc_sub,
                device,
                h_idx=atm_h_idx,
                n_old=nc_old,
                n_sub_old=n_sfc_old,
                n_sub_new=n_sfc_new,
            )
        else:
            _restore_surface(
                model.sfc_sub,
                ckpt["sfc"],
                device,
                h_idx=atm_h_idx,
                n_old=nc_old,
                n_sub_old=1,
                n_sub_new=n_sfc_new,
            )
    needs_sfc_remap = (n_sfc_old != n_sfc_new)

    # Build ocean horizontal remap index when ocean mesh changed (or transitioning from unified -> high-res ocean)
    ocn_mesh_act = model.ocn_mesh if model.ocn_mesh is not None else model.mesh
    ocn_sfc_act = model.ocn_sfc if model.ocn_sfc is not None else model.sfc
    ocn_nc_new = int(ocn_mesh_act.n_cells)
    ck_ocn_sfc = ckpt.get("ocn_sfc")
    if model.ocn_sfc is not None:
        src_sfc = ck_ocn_sfc if ck_ocn_sfc is not None else ckpt["sfc"]
        src_n_ocn = int(src_sfc["t_ocean_ml"].shape[-1])
        if src_n_ocn == ocn_nc_new:
            _restore_surface(model.ocn_sfc, src_sfc, device)
        else:
            src_xyz = mesh_coords.get("ocn_xyz") if ck_ocn_sfc is not None else old_atm_xyz
            if src_xyz is None or int(src_xyz.shape[0]) != src_n_ocn:
                from aoba.scvt_mesh import build_scvt_mesh

                src_xyz = build_scvt_mesh(_infer_glevel_from_ncells(src_n_ocn), use_earth2014=False, device=torch.device("cpu")).xyz_cell.cpu()
            ocn_sfc_h_idx = _build_surface_aware_remap_idx(
                src_xyz, src_sfc.get("surf_type"), ocn_mesh_act.xyz_cell.cpu(), ocn_mesh_act.surf_type.cpu()
            )
            _restore_surface(model.ocn_sfc, src_sfc, device, h_idx=ocn_sfc_h_idx, n_old=src_n_ocn)

    # 4. 1.5-layer Ekman-SW ocean state (`model.ocn`)
    if not needs_hremap:
        for k, v in ckpt["ocn"].items():
            setattr(model.ocn, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    else:
        for k, v in ckpt["ocn"].items():
            if not isinstance(v, torch.Tensor):
                setattr(model.ocn, k, v)
            elif v.ndim >= 1 and int(v.shape[-1]) == nc_old:
                setattr(model.ocn, k, _remap_horizontal_tensor(v.cpu(), atm_h_idx, nc_old).to(device=device))
            elif k == "u_sw_edge" and hasattr(model.ocn, k) and isinstance(getattr(model.ocn, k), torch.Tensor):
                setattr(model.ocn, k, torch.zeros_like(getattr(model.ocn, k)))

    # 5. 3D OGCM (+ BGC + EVP-ITD sea ice) with universal (ocean_glevel, ocean_nz) remapping
    if model.ogcm is not None:
        ck_og = ckpt.get("ogcm")
        ck_og_nc = int(ck_og["eta"].shape[0]) if (ck_og is not None and "eta" in ck_og and isinstance(ck_og["eta"], torch.Tensor)) else -1
        ck_og_nz = int(ck_og.get("nz_ocn", -1)) if ck_og is not None else -1
        if ck_og is not None and ck_og_nz == model.ogcm.nz_ocn and ck_og_nc == ocn_nc_new:
            for k, v in ck_og.items():
                if k == "_static_cache":
                    continue
                if k in ("bgc", "seaice_evp"):
                    sub_target = getattr(model.ogcm, k, None)
                    if v is not None and sub_target is not None:
                        sub_items = v.items() if isinstance(v, dict) else v.__dict__.items()
                        for bk, bv in sub_items:
                            setattr(sub_target, bk, bv.to(device=device) if isinstance(bv, torch.Tensor) else bv)
                    continue
                setattr(model.ogcm, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
        elif ck_og is not None and "temp" in ck_og and "sal" in ck_og and ck_og_nc > 0 and ck_og_nz > 0:
            print(
                f"  [Ocean OGCM Remap] Remapping 3D OGCM (nc={ck_og_nc}, nz={ck_og_nz}) "
                f"-> active grid (nc={ocn_nc_new}, nz={model.ogcm.nz_ocn})...",
                flush=True,
            )
            ocn_xyz_old = mesh_coords.get("ocn_xyz")
            if ocn_xyz_old is None or int(ocn_xyz_old.shape[0]) != ck_og_nc:
                from aoba.scvt_mesh import build_scvt_mesh

                ocn_xyz_old = build_scvt_mesh(_infer_glevel_from_ncells(ck_og_nc), use_earth2014=False, device=torch.device("cpu")).xyz_cell.cpu()
            wet_old = ck_og.get("wet_c")
            if wet_old is None:
                wet_old = torch.ones_like(ck_og["temp"])
            wet0_old = mesh_coords.get("ocn_wet0")
            if wet0_old is None:
                wet0_old = wet_old[0]
            og_h_idx = (
                _build_wet_ocean_remap_idx(ocn_xyz_old, wet0_old, ocn_mesh_act.xyz_cell.cpu())
                if ck_og_nc != ocn_nc_new
                else None
            )
            z_old_ocn = ck_og.get("z_c")
            if z_old_ocn is None:
                z_old_ocn = mesh_coords.get("ocn_z_c")
            if z_old_ocn is None:
                z_old_ocn = torch.linspace(10.0, 4500.0, ck_og_nz)
            z_new_ocn = model.ogcm.z_c.cpu()

            model.ogcm.temp = _remap_ocean_3d_field(
                ck_og["temp"], wet_old, z_old_ocn, z_new_ocn, og_h_idx, model.ogcm.wet_c, vmin=271.15, vmax=315.0
            ).to(device=device)
            model.ogcm.sal = _remap_ocean_3d_field(
                ck_og["sal"], wet_old, z_old_ocn, z_new_ocn, og_h_idx, model.ogcm.wet_c, vmin=5.0, vmax=45.0
            ).to(device=device)
            # Zero currents, SSH, and Kahan residuals for clean geostrophic spinup on the new bathymetry
            model.ogcm.u_edge.zero_()
            model.ogcm.u_baro.zero_()
            model.ogcm.eta.zero_()
            model.ogcm.w_vel.zero_()
            model.ogcm.t_comp.zero_()
            model.ogcm.s_comp.zero_()
            if isinstance(model.ogcm.m_seaice_prev, torch.Tensor):
                model.ogcm.m_seaice_prev = ocn_sfc_act.m_seaice.clone()
            # Keep surface ocean mixed-layer temperature synchronized with remapped top OGCM layer
            wet0_mask = model.ogcm.wet_c[0] > 0.5
            ocn_sfc_act.t_ocean_ml = torch.where(wet0_mask, model.ogcm.temp[0], ocn_sfc_act.t_ocean_ml)

            # Remap 3D Ocean BGC tracers when present in both checkpoint and active model
            ck_bgc = ck_og.get("bgc")
            if ck_bgc is not None and model.ogcm.bgc is not None:
                bgc_items = ck_bgc.items() if isinstance(ck_bgc, dict) else ck_bgc.__dict__.items()
                for bk, bv in bgc_items:
                    if not isinstance(bv, torch.Tensor) or not hasattr(model.ogcm.bgc, bk):
                        continue
                    tgt = getattr(model.ogcm.bgc, bk)
                    if bk.endswith("_comp") and isinstance(tgt, torch.Tensor):
                        tgt.zero_()
                    elif bv.ndim == 2 and int(bv.shape[0]) == ck_og_nz and int(bv.shape[1]) == ck_og_nc:
                        rm_bgc = _remap_ocean_3d_field(
                            bv, wet_old, z_old_ocn, z_new_ocn, og_h_idx, model.ogcm.wet_c, vmin=0.0
                        )
                        setattr(model.ogcm.bgc, bk, rm_bgc.to(device=device))
                    elif bv.ndim == 1 and int(bv.shape[0]) == ck_og_nc:
                        rm_1d = bv.cpu().index_select(0, og_h_idx) if og_h_idx is not None else bv.cpu()
                        setattr(model.ogcm.bgc, bk, (rm_1d * wet0_mask.cpu().to(rm_1d.dtype)).to(device=device))

            # Re-sync EVP-ITD sea-ice categories cleanly from the already-remapped `ocn_sfc_act.m_seaice`
            if bool(model.cfg.enable_seaice_evp_itd):
                from aoba.seaice_evp import init_seaice_evp_state

                model.ogcm.seaice_evp = init_seaice_evp_state(ocn_sfc_act, ocn_mesh_act)
        else:
            from aoba.ocean_ogcm import init_ogcm_state

            model.ogcm = init_ogcm_state(ocn_sfc_act, ocn_mesh_act, nz_ocn=model.cfg.ocean_nz)
            if bool(model.cfg.enable_ocean_bgc):
                from aoba.ocean_bgc import init_ocean_bgc_state

                model.ogcm.bgc = init_ocean_bgc_state(model.ogcm.wet_c, model.ogcm.h_c, model.ogcm.z_c, ocn_mesh_act.area_cell)
            if bool(model.cfg.enable_seaice_evp_itd):
                from aoba.seaice_evp import init_seaice_evp_state

                model.ogcm.seaice_evp = init_seaice_evp_state(ocn_sfc_act, ocn_mesh_act)

    # 6. River routing state (keep new mesh static drainage network if horizontal grid changed)
    _STATIC_RIVER_KEYS = {"downstream_idx", "reach_len_m", "channel_slope", "flow_vel_ms", "drain_area_m2", "is_land_river"}
    if ckpt.get("river") is not None and model.river is not None:
        ck_riv = ckpt["river"]
        if not needs_hremap:
            for k, v in ck_riv.items():
                if hasattr(model.river, k):
                    setattr(model.river, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
        else:
            if isinstance(ck_riv.get("storage_kg"), torch.Tensor) and int(ck_riv["storage_kg"].shape[0]) == nc_old:
                st_new = _remap_horizontal_tensor(ck_riv["storage_kg"].cpu(), atm_h_idx, nc_old).to(device=device)
                model.river.storage_kg = st_new * model.river.is_land_river.to(st_new.dtype)
                model.river.storage_comp.zero_()

    # 7. Dynamic lakes state (exact copy on same grid; fill-fraction-conserving remap on grid change)
    _STATIC_LAKE_KEYS = {"f_cap", "h_cap", "v_cap_m3", "d_pf_m", "is_basin"}
    if ckpt.get("lakes") is not None and getattr(model, "lakes", None) is not None:
        from aoba.lakes import update_lake_hypsometry

        ck_lk = ckpt["lakes"]
        if not needs_hremap:
            for k, v in ck_lk.items():
                if hasattr(model.lakes, k):
                    setattr(model.lakes, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
        else:
            v_old = ck_lk.get("v_lake_m3")
            vcap_old = ck_lk.get("v_cap_m3")
            if isinstance(v_old, torch.Tensor) and isinstance(vcap_old, torch.Tensor) and int(v_old.shape[0]) == nc_old:
                phi_old = torch.where(vcap_old > 1.0, (v_old / torch.clamp(vcap_old, min=1.0)).clamp(0.0, 1.0), torch.zeros_like(v_old))
                phi_new = _remap_horizontal_tensor(phi_old.cpu(), atm_h_idx, nc_old).to(device=device)
                model.lakes.v_lake_m3 = phi_new * model.lakes.v_cap_m3
                model.lakes.v_lake_comp.zero_()
            for k in ("t_ml", "t_bot", "h_ml", "h_ice", "swe_ice", "t_ice"):
                v = ck_lk.get(k)
                if isinstance(v, torch.Tensor) and int(v.shape[0]) == nc_old and hasattr(model.lakes, k):
                    setattr(model.lakes, k, _remap_horizontal_tensor(v.cpu(), atm_h_idx, nc_old).to(device=device))
            model.lakes.h_ml = torch.minimum(model.lakes.h_ml, torch.clamp(model.lakes.h_cap, min=1.0))
        model.lakes.v_lake_m3 = torch.minimum(torch.clamp(model.lakes.v_lake_m3, min=0.0), model.lakes.v_cap_m3)
        update_lake_hypsometry(model.lakes, model.mesh.area_cell)

    # 7b. Explicit bulk lightning charge state & optional OSSP lightning distillation head
    if ckpt.get("explicit_lightning") is not None and getattr(model, "explicit_lightning", None) is not None:
        ck_lt = ckpt["explicit_lightning"]
        q_old = ck_lt.get("q_chg")
        if isinstance(q_old, torch.Tensor) and tuple(q_old.shape) == tuple(model.explicit_lightning.q_chg.shape):
            for k, v in ck_lt.items():
                if hasattr(model.explicit_lightning, k):
                    setattr(model.explicit_lightning, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    if ckpt.get("lightning_distill") is not None and bool(getattr(model.cfg, "enable_lightning", False)) and bool(getattr(model.cfg, "enable_crm_self_distill", False)):
        from aoba.lightning_explicit import init_lightning_self_distill_state

        ck_ld = ckpt["lightning_distill"]
        if getattr(model, "lightning_distill", None) is None:
            model.lightning_distill = init_lightning_self_distill_state(device=device)
        if isinstance(ck_ld.get("net_state"), dict):
            model.lightning_distill.net.load_state_dict(
                {k: v.to(device=device) for k, v in ck_ld["net_state"].items()}, strict=False
            )
        model.lightning_distill.steps_seen = int(ck_ld.get("steps_seen", 0))
        model.lightning_distill.grad_updates = int(ck_ld.get("grad_updates", 0))
        model.lightning_distill.ema_loss = float(ck_ld.get("ema_loss", 0.0))
        model.lightning_distill.ema_val_corr = float(ck_ld.get("ema_val_corr", 0.0))

    # 8. Land carbon & nitrogen pools (supports both cell-level and subgrid pools)
    if ckpt.get("land_carbon") is not None and model.land_carbon is not None:
        ck_lc = ckpt["land_carbon"]
        ck_c_leaf = ck_lc.get("c_leaf")
        if isinstance(ck_c_leaf, torch.Tensor):
            lc_old_n = int(ck_c_leaf.shape[-1])
            lc_new_n = int(model.land_carbon.c_leaf.shape[-1])
            if lc_old_n == lc_new_n and not needs_hremap and not needs_subremap:
                for k, v in ck_lc.items():
                    if hasattr(model.land_carbon, k):
                        setattr(model.land_carbon, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
            else:
                lc_sub_old = n_sfc_old if lc_old_n == nc_old * n_sfc_old else 1
                lc_sub_new = n_sfc_new if lc_new_n == nc_new * n_sfc_new else 1
                for k, v in ck_lc.items():
                    if not hasattr(model.land_carbon, k):
                        continue
                    if not isinstance(v, torch.Tensor):
                        setattr(model.land_carbon, k, v)
                    elif k.endswith("_comp"):
                        tgt = getattr(model.land_carbon, k)
                        if isinstance(tgt, torch.Tensor):
                            tgt.zero_()
                    elif v.ndim >= 1 and int(v.shape[-1]) == lc_old_n:
                        rm = _remap_horizontal_tensor(v.cpu(), atm_h_idx, nc_old, lc_sub_old, lc_sub_new)
                        setattr(model.land_carbon, k, rm.to(device=device))

    # 9. Natural aerosols & prognostic ozone (horizontal + vertical remap)
    if ckpt.get("aerosols") is not None and model.aerosols is not None:
        z_oc_a = z_old_c_h if needs_atm_remap else ck_atmos.z_c.cpu()
        z_nc_a = model.atmos.z_c.cpu()
        for k, v in ckpt["aerosols"].items():
            if not hasattr(model.aerosols, k):
                continue
            if not isinstance(v, torch.Tensor):
                setattr(model.aerosols, k, v)
            elif v.ndim == 2 and int(v.shape[0]) == nz_old and int(v.shape[1]) == nc_old:
                vh = v.cpu().index_select(1, atm_h_idx) if atm_h_idx is not None else v.cpu()
                if needs_hremap or needs_vremap:
                    remapped = torch.clamp(_interp_vertical_batch(z_oc_a, vh, z_nc_a), min=0.0)
                    remapped = torch.where(z_nc_a > z_oc_a[-1:], torch.zeros_like(remapped), remapped)
                    setattr(model.aerosols, k, remapped.to(device=device))
                else:
                    setattr(model.aerosols, k, vh.to(device=device))
            elif v.ndim == 1 and int(v.shape[0]) == nc_old:
                vh = v.cpu().index_select(0, atm_h_idx) if atm_h_idx is not None else v.cpu()
                setattr(model.aerosols, k, vh.to(device=device))

    if ckpt.get("ozone") is not None and model.ozone is not None:
        z_oc_o = z_old_c_h if needs_atm_remap else ck_atmos.z_c.cpu()
        z_nc_o = model.atmos.z_c.cpu()
        for k, v in ckpt["ozone"].items():
            if not hasattr(model.ozone, k):
                continue
            if not isinstance(v, torch.Tensor):
                setattr(model.ozone, k, v)
            elif v.ndim == 2 and int(v.shape[0]) == nz_old and int(v.shape[1]) == nc_old:
                vh = v.cpu().index_select(1, atm_h_idx) if atm_h_idx is not None else v.cpu()
                if needs_hremap or needs_vremap:
                    remapped = torch.clamp(_interp_vertical_batch(z_oc_o, vh, z_nc_o), min=1.0e-10)
                    setattr(model.ozone, k, remapped.to(device=device))
                else:
                    setattr(model.ozone, k, vh.to(device=device))
            elif v.ndim == 1 and int(v.shape[0]) == nc_old:
                vh = v.cpu().index_select(0, atm_h_idx) if atm_h_idx is not None else v.cpu()
                setattr(model.ozone, k, vh.to(device=device))

    if ckpt.get("meso_nox") is not None and (bool(model.cfg.enable_red_sprites) or float(model.cfg.z_top) >= 68000.0):
        from aoba.mesosphere import init_prognostic_nox_state
        from aoba.thermodynamics import diagnose_pressure_temperature

        if model.meso_nox is None:
            p_nx_r, _, _ = diagnose_pressure_temperature(model.atmos.rho_d, model.atmos.theta_m, model.atmos.q_v)
            model.meso_nox = init_prognostic_nox_state(p_nx_r, n2o_ppmv=model.current_n2o_ppmv())
        z_oc_n = z_old_c_h if needs_atm_remap else ck_atmos.z_c.cpu()
        z_nc_n = model.atmos.z_c.cpu()
        for k, v in ckpt["meso_nox"].items():
            if not hasattr(model.meso_nox, k):
                continue
            if not isinstance(v, torch.Tensor):
                setattr(model.meso_nox, k, v)
            elif v.ndim == 2 and int(v.shape[0]) == nz_old and int(v.shape[1]) == nc_old:
                vh = v.cpu().index_select(1, atm_h_idx) if atm_h_idx is not None else v.cpu()
                if needs_hremap or needs_vremap:
                    remapped = torch.clamp(_interp_vertical_batch(z_oc_n, vh, z_nc_n), min=1.0e-12, max=2.0e-6)
                    setattr(model.meso_nox, k, remapped.to(device=device))
                else:
                    setattr(model.meso_nox, k, vh.to(device=device))
            elif v.ndim == 1 and int(v.shape[0]) == nc_old:
                vh = v.cpu().index_select(0, atm_h_idx) if atm_h_idx is not None else v.cpu()
                setattr(model.meso_nox, k, vh.to(device=device))

    if ckpt.get("trace_gas") is not None and bool(model.cfg.enable_prognostic_ch4_n2o):
        from aoba.trace_gas_cycle import init_trace_gas_cycle_state

        if model.trace_gas is None:
            model.trace_gas = init_trace_gas_cycle_state()
        for k, v in ckpt["trace_gas"].items():
            if hasattr(model.trace_gas, k):
                setattr(model.trace_gas, k, v)
    if ckpt.get("weathering") is not None and bool(model.cfg.enable_silicate_weathering):
        from aoba.planetary_geochem import init_weathering_state

        if model.weathering is None:
            model.weathering = init_weathering_state()
        for k, v in ckpt["weathering"].items():
            if hasattr(model.weathering, k):
                setattr(model.weathering, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)

    # 10. Climate accumulators (`clim`, `clim_sub`)
    for k, v in ckpt["clim"].items():
        if not isinstance(v, torch.Tensor):
            setattr(model.clim, k, v)
        elif needs_hremap and v.ndim >= 1 and int(v.shape[-1]) == nc_old:
            setattr(model.clim, k, _remap_horizontal_tensor(v.cpu(), atm_h_idx, nc_old).to(device=device))
        else:
            setattr(model.clim, k, v.to(device=device))
    if model.clim_sub is not None:
        ck_cs = ckpt.get("clim_sub")
        if ck_cs is not None:
            for k, v in ck_cs.items():
                if not isinstance(v, torch.Tensor):
                    setattr(model.clim_sub, k, v)
                elif (needs_hremap or needs_sfc_remap) and v.ndim >= 1 and int(v.shape[-1]) == nc_old * n_sfc_old:
                    setattr(model.clim_sub, k, _remap_horizontal_tensor(v.cpu(), atm_h_idx, nc_old, n_sfc_old, n_sfc_new).to(device=device))
                else:
                    setattr(model.clim_sub, k, v.to(device=device))

    # 11. Dynamic ice sheet (`icesheet`) & dynamic mesh fields
    _STATIC_ICE_KEYS = {"z_bed", "z_bed_unloaded"}
    if ckpt.get("icesheet") is not None and model.icesheet is not None:
        for k, v in ckpt["icesheet"].items():
            if not hasattr(model.icesheet, k):
                continue
            if needs_hremap and k in _STATIC_ICE_KEYS:
                continue
            if not isinstance(v, torch.Tensor):
                setattr(model.icesheet, k, v)
            elif needs_hremap and v.ndim >= 1 and int(v.shape[-1]) == nc_old:
                setattr(model.icesheet, k, _remap_horizontal_tensor(v.cpu(), atm_h_idx, nc_old).to(device=device))
            else:
                setattr(model.icesheet, k, v.to(device=device))
    if ckpt.get("mesh_dynamic") is not None and not needs_hremap:
        for k, v in ckpt["mesh_dynamic"].items():
            setattr(model.mesh, k, v.to(device=device))
    if ckpt.get("crm_distill") is not None and getattr(model, "crm_distill", None) is not None:
        ck_cd = ckpt["crm_distill"]
        try:
            model.crm_distill.net.load_state_dict(
                {k: v.to(device=device) for k, v in ck_cd["net_state"].items()}, strict=True
            )
            model.crm_distill.steps_seen = int(ck_cd.get("steps_seen", model.crm_distill.warmup_steps))
            model.crm_distill.grad_updates = int(ck_cd.get("grad_updates", 0))
            model.crm_distill.ema_loss = float(ck_cd.get("ema_loss", 1.0))
            model.crm_distill.ema_val_rmse_thm = float(ck_cd.get("ema_val_rmse_thm", 1.0))
            model.crm_distill.ema_val_rmse_qn = float(ck_cd.get("ema_val_rmse_qn", 1.0e-3))
        except Exception:
            # Architecture width changed: start student fresh with warmup automatically
            pass
    if ckpt.get("rad_distill") is not None and getattr(model, "rad_distill", None) is not None:
        ck_rd = ckpt["rad_distill"]
        try:
            model.rad_distill.net.load_state_dict(
                {k: v.to(device=device) for k, v in ck_rd["net_state"].items()}, strict=True
            )
            model.rad_distill.steps_seen = int(ck_rd.get("steps_seen", model.rad_distill.warmup_steps))
            model.rad_distill.grad_updates = int(ck_rd.get("grad_updates", 0))
            model.rad_distill.ema_loss = float(ck_rd.get("ema_loss", 1.0))
            model.rad_distill.ema_val_rmse_olr = float(ck_rd.get("ema_val_rmse_olr", 5.0))
            model.rad_distill.ema_val_rmse_asr = float(ck_rd.get("ema_val_rmse_asr", 5.0))
        except Exception:
            pass
    if getattr(model, "sparse_crm_pool", None) is not None:
        pool_dst = model.sparse_crm_pool
        ck_sp = ckpt.get("sparse_crm_pool")
        restored_pool = False
        if (
            ck_sp is not None
            and not needs_hremap
            and not needs_vremap
            and int(ck_sp.get("ny_full", -1)) == int(pool_dst.ny_full)
            and int(ck_sp.get("nx_full", -1)) == int(pool_dst.nx_full)
            and int(ck_sp.get("n_pool", -1)) == int(pool_dst.n_pool)
        ):
            try:
                for k, v in ck_sp["crm"].items():
                    if isinstance(v, torch.Tensor):
                        setattr(pool_dst.crm, k, v.to(device=device))
                    else:
                        setattr(pool_dst.crm, k, v)
                pool_dst.q_rad = ck_sp["q_rad"].to(device=device)
                pool_dst.slot_cell_ids = ck_sp["slot_cell_ids"].to(device=device, dtype=torch.int64)
                pool_dst.q_rad_valid = bool(ck_sp.get("q_rad_valid", False))
                restored_pool = True
            except Exception:
                restored_pool = False
        if not restored_pool:
            # Cross-grid / resolution remap or legacy checkpoint: repopulate slots cleanly from remapped tile CRM
            pool_dst.slot_cell_ids.fill_(-1)
            pool_dst.q_rad_valid = False
    return ckpt


_SUBGRID_2D_UNITS: dict[str, str] = {
    "lwp_sub": "kg m-2", "iwp_sub": "kg m-2", "w_max_sub": "m s-1",
    "precip_mm_day_sub": "mm day-1", "sw1_dir_sub": "W m-2", "sw1_dif_sub": "W m-2",
    "sw_dn_sub": "W m-2", "lw_dn_sub": "W m-2", "olr_sub": "W m-2", "shf_sub": "W m-2",
    "evap_mm_day_sub": "mm day-1", "gpp_sub": "umol m-2 s-1", "t_sfc_sub": "K",
    "m_seaice_sub": "kg m-2", "swe_land_sub": "kg m-2", "w_root_sub": "kg m-2", "lai_sub": "m2 m-2",
}


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
        # Complete-optimality EEO vegetation & cloud/sea-ice diagnostics for spin-up & paper verification
        "h_canopy": "m", "f_evg": "1", "f_c4": "1", "s_root_cap": "kg m-2",
        "vcmax25": "umol m-2 s-1", "chi_leaf": "1",
        "clt": "%", "lwp": "kg m-2", "iwp": "kg m-2", "prw": "kg m-2",
        "siconc": "%", "sithick": "m",
    }
    units_2d_ocn = {
        "ocn_eta": "m",
        "ocn_sss": "psu",
        "ocn_h_bl": "m",
        "tau_x_sfc": "N m-2",
        "tau_y_sfc": "N m-2",
        "runoff_total": "kg m-2 s-1",
        "runoff_glacier": "kg m-2 s-1",
        "precip_snow_mm_day": "mm day-1",
        "sw1_dir_sfc": "W m-2",
        "sw1_dif_sfc": "W m-2",
        "sw2_dir_sfc": "W m-2",
        "sw2_dif_sfc": "W m-2",
        "lw_dn_sfc": "W m-2",
    }
    units_3d = {"temp": "K", "pres": "Pa", "q_v": "kg kg-1", "u_zonal": "m s-1", "v_merid": "m s-1"}
    units_3d_ocn = {
        "ocn_temp": "K",
        "ocn_sal": "psu",
        "ocn_u_zonal": "m s-1",
        "ocn_v_merid": "m s-1",
        "ocn_w_vel": "m s-1",
    }

    if include_2d:
        kg30, kg5 = diagnose_model_koppen_geiger(model)
        for name, unit in units_2d.items():
            if name in sums:
                arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
                data_vars[name] = (("time", "n_cell"), arr, {"units": unit})
        if model.ozone is not None and "o3_tco_du" in sums:
            data_vars["o3_tco_du"] = (("time", "n_cell"), (sums["o3_tco_du"] * inv).detach().cpu().numpy()[None, :], {"units": "DU"})
        if model.ogcm is not None:
            ocn_cell_dim = "n_cell_ocn" if model.ocn_mesh is not None else "n_cell"
            for name, unit in units_2d_ocn.items():
                if name in sums:
                    arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
                    c_dim = ocn_cell_dim if name in ("ocn_eta", "ocn_sss", "ocn_h_bl") else "n_cell"
                    data_vars[name] = (("time", c_dim), arr, {"units": unit})
        if model.icesheet is not None:
            for name in ("h_ice", "z_topo_dyn"):
                if name in sums:
                    arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
                    data_vars[name] = (("time", "n_cell"), arr, {"units": "m"})
        data_vars["koppen_code"] = (("time", "n_cell"), kg30.detach().cpu().numpy()[None, :])
        data_vars["koppen_major"] = (("time", "n_cell"), kg5.detach().cpu().numpy()[None, :])
        if getattr(model, "crm_distill", None) is not None or getattr(model, "rad_distill", None) is not None:
            from aoba.crm_self_distill import consume_distill_shadow_maps

            sh_maps = consume_distill_shadow_maps(
                getattr(model, "crm_distill", None),
                getattr(model, "rad_distill", None),
                reset=False,
            )
            _sh_units = {
                "shadow_atm_dthm_k": "K",
                "shadow_atm_dprec_mmd": "mm day-1",
                "shadow_atm_unc": "1",
                "shadow_land_dthm0_k": "K",
                "shadow_land_dprec_mmd": "mm day-1",
                "shadow_rad_dolr_wm2": "W m-2",
                "shadow_rad_bias_olr_wm2": "W m-2",
                "shadow_land_dlwd_wm2": "W m-2",
                "shadow_land_dswd_wm2": "W m-2",
                "shadow_teacher_visits": "1",
            }
            for sk, sv in sh_maps.items():
                data_vars[sk] = (
                    ("time", "n_cell"),
                    sv.detach().cpu().numpy().astype(np.float32)[None, :],
                    {"units": _sh_units.get(sk, "1")},
                )
    if include_3d:
        for name, unit in units_3d.items():
            if name in sums:
                arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :]
                data_vars[name] = (("time", "nz", "n_cell"), arr, {"units": unit})
        if model.ozone is not None and "o3_vmr" in sums:
            data_vars["o3_vmr"] = (("time", "nz", "n_cell"), (sums["o3_vmr"] * inv).detach().cpu().numpy()[None, :, :], {"units": "mol mol-1"})
        if model.ogcm is not None:
            ocn_cell_dim = "n_cell_ocn" if model.ocn_mesh is not None else "n_cell"
            for name, unit in units_3d_ocn.items():
                if name in sums:
                    arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :]
                    data_vars[name] = (("time", "nz_ocn", ocn_cell_dim), arr, {"units": unit})
    if include_subgrid_2d:
        for name, unit in _SUBGRID_2D_UNITS.items():
            if name in sums:
                arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :, :]
                data_vars[name] = (("time", "n_cell", "crm_y", "crm_x"), arr, {"units": unit})

    _, _, ny, nx = model.crm.rho_d.shape

    coords: dict[str, Any] = {
        "time": ("time", t_days, {"units": "days since 2000-03-20 00:00:00"}),
        "lat_cell": ("n_cell", lat_deg, {"units": "degrees_north"}),
        "lon_cell": ("n_cell", lon_deg, {"units": "degrees_east"}),
    }
    if model.ocn_mesh is not None:
        coords["lat_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lat_cell.detach().cpu().numpy()), {"units": "degrees_north"})
        coords["lon_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lon_cell.detach().cpu().numpy()), {"units": "degrees_east"})
    if include_3d:
        coords["nz"] = ("nz", model.vgrid.zeta_c.detach().cpu().numpy(), {"units": "m", "positive": "up"})
        if model.ogcm is not None:
            coords["nz_ocn"] = ("nz_ocn", model.ogcm.z_c.detach().cpu().numpy(), {"units": "m", "positive": "down"})
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
    t2_k_override: torch.Tensor | None = None,
    precip_mm_day_override: torch.Tensor | None = None,
    olr_override: torch.Tensor | None = None,
) -> None:
    """Atomically write lightweight 2D live-preview fields (`T2`, `Precip`, `LAI`, `OLR`) for the CPU GUI."""
    mesh = model.mesh
    if t2_k_override is None:
        _, _, temp0 = diagnose_pressure_temperature(
            model.atmos.rho_d[0:1], model.atmos.theta_m[0:1], model.atmos.q_v[0:1]
        )
        t_sfc = torch.where(mesh.surf_type == SURF_OCEAN, model.sfc.t_ocean_ml, model.sfc.t_canopy)
        t2_k = 0.35 * t_sfc + 0.65 * temp0[0]
    else:
        t2_k = t2_k_override
    if precip_mm_day_override is None:
        rain = model.rad_cache.get("rain", torch.zeros_like(t2_k))
        snow = model.rad_cache.get("snow", torch.zeros_like(t2_k))
        precip_mm_day = (rain + snow) * C.SECONDS_PER_DAY
    else:
        precip_mm_day = precip_mm_day_override
    if olr_override is None:
        t_sfc_olr = torch.where(mesh.surf_type == SURF_OCEAN, model.sfc.t_ocean_ml, model.sfc.t_canopy)
        olr = model.rad_cache.get("olr", 0.85 * C.STEFAN_BOLTZMANN * torch.clamp(t_sfc_olr, min=150.0) ** 4)
    else:
        olr = olr_override
    lai = model.sfc.bio.lai
    h_ice = model.icesheet.h_ice if model.icesheet is not None else torch.zeros_like(t2_k)
    sst_c_host = (model.sfc.t_ocean_ml - 273.15)
    seaice_host = torch.clamp(torch.sqrt(torch.clamp(model.sfc.m_seaice, min=0.0) / 500.0), 0.0, 1.0)
    swe_land = torch.clamp(model.sfc.swe_land, min=0.0)

    # Column LWP & IWP [g m^-2] for Himawari True-Color RGB cloud reflectance
    dz_c4 = model.atmos.dz_c.unsqueeze(-1).unsqueeze(-1)
    lwp_gm2 = (model.crm.rho_d * torch.clamp(model.crm.q_c, min=0.0) * dz_c4).sum(dim=0).mean(dim=(-2, -1)) * 1.0e3
    iwp_gm2 = (model.crm.rho_d * torch.clamp(model.crm.q_i, min=0.0) * dz_c4).sum(dim=0).mean(dim=(-2, -1)) * 1.0e3

    # Lowest-layer wind vector (u10_u, u10_v) & speed [m s^-1] + significant wave height H_s [m]
    u10_u_t, u10_v_t = reconstruct_cell_winds(mesh, model.atmos.u_edge[0:1])
    u10_u = u10_u_t[0]
    u10_v = u10_v_t[0]
    u10_ms = torch.sqrt(torch.clamp(u10_u * u10_u + u10_v * u10_v, min=0.0))
    if getattr(model, "ocean_waves", None) is not None and getattr(model.ocean_waves, "hs", None) is not None:
        hs_raw = model.ocean_waves.hs
        if hs_raw.shape[0] == mesh.n_cells:
            wave_hs_m = torch.clamp(hs_raw, min=0.0, max=15.0)
        else:
            wave_hs_m = torch.clamp(0.0246 * u10_ms * u10_ms, min=0.0, max=12.0) * (mesh.surf_type == SURF_OCEAN).to(u10_ms.dtype)
    else:
        wave_hs_m = torch.clamp(0.0246 * u10_ms * u10_ms, min=0.0, max=12.0) * (mesh.surf_type == SURF_OCEAN).to(u10_ms.dtype)

    # Exact solar declination & subsolar longitude for Himawari solar incidence angle
    cal_dt = model.calendar.date_at(model.sim_time_sec)
    obliq_rad = math.radians(float(model.cfg.obliquity_deg))
    doy_now = float(cal_dt.doy)
    yr_days = max(float(cal_dt.days_in_year), 1.0)
    orb_ang = 2.0 * math.pi * ((doy_now - 80.0) / yr_days)
    solar_decl_deg = math.degrees(math.asin(max(-1.0, min(1.0, math.sin(obliq_rad) * math.sin(orb_ang)))))
    frac_day = ((model.sim_time_sec / C.SECONDS_PER_DAY) % 1.0 + 1.0) % 1.0
    subsolar_lon_deg = ((180.0 - 360.0 * frac_day + 180.0) % 360.0) - 180.0

    date_str = cal_dt.iso_datetime[:16]
    coords_np = getattr(mesh, "_live_preview_coords", None)
    if coords_np is None or len(coords_np) < 4:
        coords_np = (
            np.rad2deg(mesh.lat_cell.detach().cpu().numpy()).astype(np.float32),
            np.rad2deg(mesh.lon_cell.detach().cpu().numpy()).astype(np.float32),
            mesh.surf_type.detach().cpu().numpy().astype(np.int8),
            mesh.z_topo.detach().cpu().numpy().astype(np.float32),
        )
        mesh._live_preview_coords = coords_np
    lat_deg_np, lon_deg_np, surf_type_np, z_topo_np = coords_np

    extra_arrays: dict[str, Any] = {
        "sst_c": sst_c_host.detach().cpu().numpy().astype(np.float32),
        "seaice_frac": seaice_host.detach().cpu().numpy().astype(np.float32),
        "u10_ms": u10_ms.detach().cpu().numpy().astype(np.float32),
        "u10_u": u10_u.detach().cpu().numpy().astype(np.float32),
        "u10_v": u10_v.detach().cpu().numpy().astype(np.float32),
        "lwp_gm2": lwp_gm2.detach().cpu().numpy().astype(np.float32),
        "iwp_gm2": iwp_gm2.detach().cpu().numpy().astype(np.float32),
        "swe_land": swe_land.detach().cpu().numpy().astype(np.float32),
        "z_topo": z_topo_np,
        "wave_hs_m": wave_hs_m.detach().cpu().numpy().astype(np.float32),
        "solar_decl_deg": np.float32(solar_decl_deg),
        "subsolar_lon_deg": np.float32(subsolar_lon_deg),
    }
    if model.ogcm is not None:
        omesh = model.ocn_mesh if model.ocn_mesh is not None else mesh
        ocn_coords = getattr(omesh, "_live_preview_ocn_coords", None)
        if ocn_coords is None or len(ocn_coords) < 4:
            ocn_coords = (
                np.rad2deg(omesh.lat_cell.detach().cpu().numpy()).astype(np.float32),
                np.rad2deg(omesh.lon_cell.detach().cpu().numpy()).astype(np.float32),
                model.ogcm.wet_c[0].detach().cpu().numpy().astype(np.uint8),
                model.ogcm.h_col.detach().cpu().numpy().astype(np.float32),
            )
            omesh._live_preview_ocn_coords = ocn_coords
        olat_np, olon_np, owet_np, odepth_np = ocn_coords
        osst_c = (model.ogcm.temp[0] - 273.15).detach().cpu().numpy().astype(np.float32)
        ou_c_t, ov_c_t = reconstruct_cell_winds(omesh, model.ogcm.u_edge[0:1])
        ou_c = ou_c_t[0]
        ov_c = ov_c_t[0]
        ospd = torch.sqrt(torch.clamp(ou_c * ou_c + ov_c * ov_c, min=0.0)).detach().cpu().numpy().astype(np.float32)
        osfc = model.ocn_sfc if model.ocn_sfc is not None else model.sfc
        oice = torch.clamp(torch.sqrt(torch.clamp(osfc.m_seaice, min=0.0) / 500.0), 0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        extra_arrays["ocn_lat_deg"] = olat_np
        extra_arrays["ocn_lon_deg"] = olon_np
        extra_arrays["ocn_wet"] = owet_np
        extra_arrays["ocn_depth_m"] = odepth_np
        extra_arrays["ocn_sst_c"] = osst_c
        extra_arrays["ocn_speed_ms"] = ospd
        extra_arrays["ocn_u_ms"] = ou_c.detach().cpu().numpy().astype(np.float32)
        extra_arrays["ocn_v_ms"] = ov_c.detach().cpu().numpy().astype(np.float32)
        extra_arrays["ocn_seaice_frac"] = oice

    tmp_path = out_path.with_name(f".{out_path.name}.tmp.npz")
    np.savez(
        tmp_path,
        lat_deg=lat_deg_np,
        lon_deg=lon_deg_np,
        surf_type=surf_type_np,
        t2_k=t2_k.detach().cpu().numpy().astype(np.float32),
        precip_mm_day=precip_mm_day.detach().cpu().numpy().astype(np.float32),
        olr=olr.detach().cpu().numpy().astype(np.float32),
        lai=lai.detach().cpu().numpy().astype(np.float32),
        h_ice=h_ice.detach().cpu().numpy().astype(np.float32),
        year=np.int32(year),
        day_in_yr=np.float32(day_in_yr),
        sim_days=np.float64(sim_days),
        wall_s_per_day=np.float32(wall_s_per_day),
        date_str=np.array(date_str),
        **extra_arrays,
    )
    os.replace(tmp_path, out_path)


def _ensure_csv_header_upgraded(csv_path: Path, expected_header: list[str]) -> None:
    """Safely upgrade an existing CSV file's header if new diagnostic columns were appended."""
    if not csv_path.exists():
        return
    try:
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        return
    if not rows:
        return
    old_hdr = rows[0]
    if old_hdr == expected_header:
        return
    if len(old_hdr) < len(expected_header) and expected_header[:len(old_hdr)] == old_hdr:
        pad = ["nan"] * (len(expected_header) - len(old_hdr))
        tmp_p = csv_path.with_name(f".{csv_path.name}.upgrade.tmp")
        with open(tmp_p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(expected_header)
            for r in rows[1:]:
                w.writerow(r + pad)
        os.replace(tmp_p, csv_path)


def _git_version_info() -> tuple[str, bool]:
    """Return `(short_sha, is_dirty)` of the AOBA repository without failing if git is absent."""
    repo = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        st = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        return sha, bool(st)
    except Exception:
        return "unknown", False


def _serialize_applied_config(cfg: AOBAConfig, out_cfg: OutputConfig) -> dict[str, Any]:
    """Build a JSON-serializable canonical snapshot of all model + output parameters."""
    flat: dict[str, Any] = {}
    for k, v in asdict(cfg).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[k] = v
        else:
            flat[k] = str(v)
    for k, v in asdict(out_cfg).items():
        if k == "out_dir":
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[f"output.{k}"] = v
        else:
            flat[f"output.{k}"] = str(v)
    return flat


def _record_tuning_history(
    out_root: Path,
    cfg: AOBAConfig,
    out_cfg: OutputConfig,
    *,
    sim_year: int,
    sim_date: str,
    sim_days: float,
    step_count: int,
    source_ckpt: str = "",
    prev_ckpt_cfg: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Automatically log initial configuration, code commits, and any parameter changes over spin-up.

    Writes both machine-readable `tuning_history.jsonl` and publication-ready `tuning_history.md`
    (with exact simulated year, date, step, git commit, parameter diffs, and tuning rationale).
    """
    cur_cfg = _serialize_applied_config(cfg, out_cfg)
    git_sha, git_dirty = _git_version_info()
    snap_path = out_root / "last_applied_config.json"
    jsonl_path = out_root / "tuning_history.jsonl"
    md_path = out_root / "tuning_history.md"

    prev_cfg: dict[str, Any] | None = None
    prev_sha: str | None = None
    event = "INIT"
    if snap_path.exists():
        try:
            prev_rec = json.loads(snap_path.read_text(encoding="utf-8"))
            prev_cfg = prev_rec.get("config")
            prev_sha = prev_rec.get("git_commit")
            event = "RESUME"
        except (OSError, json.JSONDecodeError):
            prev_cfg = None
    elif prev_ckpt_cfg is not None:
        prev_cfg = prev_ckpt_cfg
        event = "PHASE_TRANSITION"

    diffs: dict[str, dict[str, Any]] = {}
    if prev_cfg is not None:
        all_keys = sorted(set(prev_cfg.keys()) | set(cur_cfg.keys()))
        for k in all_keys:
            v_old = prev_cfg.get(k)
            v_new = cur_cfg.get(k)
            if v_old != v_new:
                diffs[k] = {"old": v_old, "new": v_new}
        if diffs and event == "RESUME":
            event = "CONFIG_CHANGE"
        elif prev_sha is not None and prev_sha != git_sha and event == "RESUME":
            event = "CODE_CHANGE"

    should_log = (event != "RESUME") or bool(diffs) or bool(note.strip()) or not jsonl_path.exists()
    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if should_log:
        entry: dict[str, Any] = {
            "timestamp_utc": ts_utc,
            "event": event,
            "sim_year": int(sim_year),
            "sim_date": str(sim_date),
            "sim_days": round(float(sim_days), 3),
            "step": int(step_count),
            "git_commit": git_sha,
            "git_dirty": bool(git_dirty),
            "source_ckpt": str(source_ckpt),
            "note": str(note.strip()),
            "changes": diffs,
        }
        if event == "INIT":
            entry["initial_summary"] = {
                "glevel": cfg.glevel,
                "ocean_glevel": cfg.ocean_glevel,
                "nz": cfg.nz,
                "ocean_nz": cfg.ocean_nz,
                "crm_nx": cfg.crm_nx,
                "crm_ny": cfg.crm_ny,
                "crm_dx": cfg.crm_dx,
                "crm_sfc_tiles": cfg.crm_sfc_tiles,
                "enable_crm_self_distill": cfg.enable_crm_self_distill,
                "enable_rad_self_distill": cfg.enable_rad_self_distill,
                "crm_distill_stride": cfg.crm_distill_stride,
                "co2_mode": cfg.co2_mode,
                "co2_scenario": cfg.co2_scenario,
                "co2_ppmv": cfg.co2_ppmv,
                "rad_scheme": cfg.rad_scheme,
                "microphysics_scheme": cfg.microphysics_scheme,
            }
        with open(jsonl_path, "a", encoding="utf-8") as f_j:
            f_j.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if not md_path.exists():
            md_path.write_text(
                "# AOBA Spin-Up & Experiment Tuning History\n\n"
                "| UTC Time | Event | Sim Year | Sim Date | Step | Git | Changes (`old -> new`) | Note |\n"
                "| :--- | :--- | ---: | :--- | ---: | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )
        if diffs:
            chg_str = "; ".join(f"`{k}`: `{v['old']}` -> `{v['new']}`" for k, v in diffs.items())
        elif event == "INIT":
            chg_str = (
                f"Initial run: G{cfg.glevel}z{cfg.nz} / CRM {cfg.crm_ny}x{cfg.crm_nx}@{cfg.crm_dx:.0f}m / "
                f"OG{cfg.ocean_glevel or cfg.glevel}z{cfg.ocean_nz} / CO2={cfg.co2_mode}:{cfg.co2_scenario if cfg.co2_mode=='scenario' else cfg.co2_ppmv}"
            )
        else:
            chg_str = "(no parameter diff)"
        git_lbl = f"`{git_sha}{'*' if git_dirty else ''}`"
        note_clean = note.strip().replace("|", "/").replace("\n", " ") or (f"from `{Path(source_ckpt).name}`" if source_ckpt else "-")
        with open(md_path, "a", encoding="utf-8") as f_m:
            f_m.write(
                f"| {ts_utc} | {event} | {sim_year} | {sim_date} | {step_count} | {git_lbl} | {chg_str} | {note_clean} |\n"
            )
        print(f"[TUNING LOG] Recorded {event} at Y{sim_year:04d} ({sim_date}, step {step_count}): {len(diffs)} param diff(s)", flush=True)

    snap_tmp = out_root / ".last_applied_config.json.tmp"
    snap_tmp.write_text(
        json.dumps(
            {
                "timestamp_utc": ts_utc,
                "git_commit": git_sha,
                "git_dirty": bool(git_dirty),
                "sim_year": int(sim_year),
                "sim_date": str(sim_date),
                "step": int(step_count),
                "config": cur_cfg,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(snap_tmp, snap_path)
    return cur_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-year AOBA climate integration (TOML-configurable Zarr v3 output)")
    parser.add_argument("--config", type=str, default="", help="Path to unified AOBA TOML configuration file")
    parser.add_argument("--glevel", type=int, default=None)
    parser.add_argument("--years", type=int, default=None)
    parser.add_argument("--days-per-year", type=float, default=None)
    parser.add_argument("--co2-ppmv", type=float, default=None)
    parser.add_argument("--co2-mode", type=str, default=None, help="constant, scenario, or csv")
    parser.add_argument("--co2-scenario", type=str, default=None, help="CO2 scenario key (e.g. obs_historical_noaa, cmip6_ssp245)")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--init-from-restart",
        type=str,
        default="",
        help="Optional external restart.pt / restart_mid.pt to warm-start from (supports vertical grid remap)",
    )
    parser.add_argument("--dynamic-icesheet", action="store_true", default=None, help="Enable SIA + EMA-accelerated ice-sheet coupling")
    parser.add_argument("--icesheet-ema-years", type=float, default=None)
    parser.add_argument("--icesheet-accel", type=float, default=None)
    parser.add_argument("--icesheet-accel-spinup-years", type=float, default=None)
    parser.add_argument("--icesheet-isostasy-tau-yr", type=float, default=None)
    parser.add_argument("--cmip7-output", action="store_true", default=None, help="Enable CMIP7 Data Request v1.2 ESM-BCV monthly Zarr stream")
    parser.add_argument("--enable-ocean-bgc", action="store_true", default=None, help="Enable FlexPFT marine BGC & OMIP/mocsy 2.0 carbonate chemistry")
    parser.add_argument("--tuning-note", type=str, default="", help="Optional provenance note recorded in tuning_history.md / tuning_history.jsonl")
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
        if args.co2_mode is None:
            cfg.co2_mode = "constant"
    if args.co2_mode is not None:
        cfg.co2_mode = args.co2_mode
    if args.co2_scenario is not None:
        cfg.co2_scenario = args.co2_scenario
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
    if args.cmip7_output is not None:
        out_cfg.cmip7_output_enabled = bool(args.cmip7_output)
    if args.enable_ocean_bgc is not None:
        cfg.enable_ocean_bgc = bool(args.enable_ocean_bgc)

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

    gpu_kern = get_active_gpu_backend(dev)
    print(
        f"Initializing AOBA Glevel-{cfg.glevel} ({cfg.crm_nx}x{cfg.crm_ny}, dx={cfg.crm_dx:.0f} m, "
        f"nz={cfg.nz}, dt={cfg.dt_host:.0f} s, CO2={cfg.co2_mode}:{cfg.co2_scenario if cfg.co2_mode == 'scenario' else cfg.co2_ppmv}) on {dev} "
        f"[{hw.chip_name}, {hw.gpu_cores} GPUc/SMs, {hw.device_mem_gb:.1f}GB VRAM, kernels={gpu_kern} | "
        f"chunks crm={cfg.crm_col_chunk} rad={cfg.rad_col_chunk}]...",
        flush=True,
    )
    model = init_aoba_model(cfg, device=dev)

    mesh_zarr = out_root / "mesh.zarr"
    if not mesh_zarr.exists():
        save_zarr(export_scvt_mesh_xarray(model), mesh_zarr, overwrite=True)

    cmip7_enabled = bool(getattr(out_cfg, "cmip7_output_enabled", False))
    cmip7_acc: CMIP7MonthlyAccumulator | None = None
    if cmip7_enabled:
        cmip7_fx_zarr = out_root / "cmip7_fx.zarr"
        if not cmip7_fx_zarr.exists():
            save_zarr(export_cmip7_fx_xarray(model), cmip7_fx_zarr, overwrite=True)
        cmip7_acc = CMIP7MonthlyAccumulator(
            include_plev19=bool(getattr(out_cfg, "cmip7_include_plev19", True)),
            include_ocean_3d=bool(getattr(out_cfg, "cmip7_include_ocean_3d", True)),
        )

    start_year = 1
    resume_step_in_yr = 0
    resume_m_idx = 0
    resume_m_cnt = 0
    resume_m_sums: dict[str, torch.Tensor] = {}
    resume_mon_accum: dict[str, float] | None = None
    resume_mon_cnt = 0
    prev_ohc_j: float | None = None
    prev_ohc_time_sec: float | None = None
    resumed_existing = False
    loaded_ckpt_path = ""
    loaded_ckpt_cfg: dict[str, Any] | None = None
    if not args.no_resume:
        for y in range(years, 0, -1):
            y_dir = out_root / f"year_{y:04d}"
            if (y_dir / "_COMPLETE").exists() and (y_dir / "restart.pt").exists():
                print(f"Resuming from completed Year {y:04d} ({y_dir / 'restart.pt'})...", flush=True)
                ckpt_yr = _load_restart_into_model(model, y_dir / "restart.pt", dev)
                loaded_ckpt_path = str(y_dir / "restart.pt")
                loaded_ckpt_cfg = ckpt_yr.get("applied_config", None)
                prev_ohc_j = ckpt_yr.get("prev_ohc_j", None)
                prev_ohc_time_sec = ckpt_yr.get("prev_ohc_time_sec", None)
                start_year = y + 1
                resumed_existing = True
                break
        for y in range(years, start_year - 1, -1):
            mid_pt = out_root / f"year_{y:04d}" / "restart_mid.pt"
            if mid_pt.exists():
                print(f"Resuming mid-year from {mid_pt}...", flush=True)
                ckpt_mid = _load_restart_into_model(model, mid_pt, dev)
                loaded_ckpt_path = str(mid_pt)
                loaded_ckpt_cfg = ckpt_mid.get("applied_config", None)
                start_year = y
                resumed_existing = True
                prev_ohc_j = ckpt_mid.get("prev_ohc_j", None)
                prev_ohc_time_sec = ckpt_mid.get("prev_ohc_time_sec", None)
                if not ckpt_mid.get("_vertical_remapped", False):
                    resume_step_in_yr = int(ckpt_mid.get("resume_step_in_yr", 0))
                    resume_m_idx = int(ckpt_mid.get("resume_m_idx", 0))
                    resume_m_cnt = int(ckpt_mid.get("resume_m_cnt", 0))
                    raw_msums = ckpt_mid.get("resume_m_sums", {}) or {}
                    resume_m_sums = {k: v.to(device=dev) for k, v in raw_msums.items()}
                    resume_mon_accum = ckpt_mid.get("resume_mon_accum", None)
                    resume_mon_cnt = int(ckpt_mid.get("resume_mon_cnt", 0))
                    if cmip7_acc is not None and ckpt_mid.get("resume_cmip7_state") is not None:
                        cmip7_acc.load_state_dict(ckpt_mid["resume_cmip7_state"], dev)
                break

    if not resumed_existing and args.init_from_restart:
        init_pt = Path(args.init_from_restart).expanduser()
        if not init_pt.exists():
            raise FileNotFoundError(f"--init-from-restart file not found: {init_pt}")
        print(f"Warm-starting from external restart checkpoint: {init_pt}...", flush=True)
        ckpt_ext = _load_restart_into_model(model, init_pt, dev)
        loaded_ckpt_path = str(init_pt)
        loaded_ckpt_cfg = ckpt_ext.get("applied_config", None)
        elapsed_yr = int(math.floor(model.sim_time_sec / (C.SECONDS_PER_DAY * days_per_year) + 1.0e-6))
        start_year = max(1, min(years, elapsed_yr + 1))
        print(
            f"  Continuing at date {model.calendar.date_at(model.sim_time_sec).iso_datetime} "
            f"(start_year={start_year:04d}/{years:04d})",
            flush=True,
        )

    # Automatic spin-up tuning & provenance history log (`tuning_history.jsonl` + `tuning_history.md`)
    applied_cfg_snapshot = _record_tuning_history(
        out_root,
        cfg,
        out_cfg,
        sim_year=start_year,
        sim_date=model.calendar.date_at(model.sim_time_sec).iso_date,
        sim_days=model.sim_time_sec / C.SECONDS_PER_DAY,
        step_count=model.step_count,
        source_ckpt=loaded_ckpt_path,
        prev_ckpt_cfg=loaded_ckpt_cfg,
        note=args.tuning_note,
    )

    # Continuous ocean + coupled spin-up monitor CSV across all years & restarts
    spinup_csv = out_root / "spinup_monitor.csv"
    _SPINUP_HEADER = [
        "year", "month", "date", "sim_days",
        "ts_k", "olr_wm2", "asr_wm2", "net_toa_wm2", "ohc_flux_wm2",
        "precip_mm_d", "gpp_umol", "lai", "seaice_mkm2", "seaice_nh_mkm2", "seaice_sh_mkm2",
        "sst_global_k", "sss_global_psu",
        "t_ocn_global_k", "t_ocn_0_300m_k", "t_ocn_300_1500m_k", "t_ocn_deep_k",
        "s_ocn_global_psu", "ohc_zj", "ocn_ke_jm2", "ocn_eta_rms_m",
        "ocn_hbl_mean_m", "ocn_hbl_max_m", "amoc_26n_sv", "amoc_max_sv", "acc_drake_sv",
        # Extended Letter-paper spin-up, EEO vegetation, sea-ice extent/vol, MOC/OHT & OSSP shadow columns
        "co2_ppmv", "clt_pct", "lwp_gm2", "iwp_gm2", "prw_kgm2",
        "gpp_pgc_yr", "lai_land_mean", "hcanopy_land_m", "fevg_land", "fc4_land",
        "seaice_ext_nh_mkm2", "seaice_ext_sh_mkm2", "seaice_vol_nh_km3", "seaice_vol_sh_km3",
        "amoc_26n_depth_m", "aabw_30s_sv", "oht_atl_26n_pw",
        "land_c_veg_pgc", "land_c_soil_pgc",
        "shadow_atm_dthm_k", "shadow_rad_dolr_wm2", "shadow_land_dthm0_k", "shadow_land_dlwd_wm2",
    ]
    a_earth_m2 = float(model.mesh.area_cell.detach().cpu().to(torch.float64).sum().item())
    if spinup_csv.exists() and (resumed_existing or start_year > 1 or resume_step_in_yr > 0):
        _ensure_csv_header_upgraded(spinup_csv, _SPINUP_HEADER)
    if not spinup_csv.exists() or (start_year == 1 and resume_step_in_yr == 0 and not resumed_existing):
        with open(spinup_csv, "w", newline="") as f_sp:
            csv.writer(f_sp).writerow(_SPINUP_HEADER)
            if model.ogcm is not None:
                ocn_mesh_0 = model.ocn_mesh if model.ocn_mesh is not None else model.mesh
                ocn_sfc_0 = model.ocn_sfc if model.ocn_sfc is not None else model.sfc
                m0 = diagnose_ocean_spinup_metrics(model.ogcm, ocn_sfc_0, ocn_mesh_0)
                prev_ohc_j = m0.get("ohc_j", None)
                prev_ohc_time_sec = model.sim_time_sec
                date0 = model.calendar.date_at(model.sim_time_sec).iso_date
                csv.writer(f_sp).writerow([
                    0, 0, date0, f"{model.sim_time_sec / C.SECONDS_PER_DAY:.2f}",
                    "nan", "nan", "nan", "nan", "0.000",
                    "nan", "nan", "nan",
                    f"{m0.get('seaice_nh_mkm2', 0.0) + m0.get('seaice_sh_mkm2', 0.0):.3f}",
                    f"{m0.get('seaice_nh_mkm2', float('nan')):.3f}",
                    f"{m0.get('seaice_sh_mkm2', float('nan')):.3f}",
                    f"{m0.get('sst_global_k', float('nan')):.4f}",
                    f"{m0.get('sss_global_psu', float('nan')):.4f}",
                    f"{m0.get('t_ocn_global_k', float('nan')):.5f}",
                    f"{m0.get('t_ocn_0_300m_k', float('nan')):.5f}",
                    f"{m0.get('t_ocn_300_1500m_k', float('nan')):.5f}",
                    f"{m0.get('t_ocn_deep_k', float('nan')):.5f}",
                    f"{m0.get('s_ocn_global_psu', float('nan')):.5f}",
                    f"{m0.get('ohc_zj', float('nan')):.4f}",
                    f"{m0.get('ocn_ke_jm2', float('nan')):.2f}",
                    f"{m0.get('ocn_eta_rms_m', float('nan')):.4f}",
                    f"{m0.get('ocn_hbl_mean_m', float('nan')):.2f}",
                    f"{m0.get('ocn_hbl_max_m', float('nan')):.1f}",
                    f"{m0.get('amoc_26n_sv', float('nan')):.2f}",
                    f"{m0.get('amoc_max_sv', float('nan')):.2f}",
                    f"{m0.get('acc_drake_sv', float('nan')):.2f}",
                    f"{model.current_co2_ppmv():.2f}",
                    "nan", "nan", "nan", "nan",
                    "nan", "nan", "nan", "nan", "nan",
                    f"{m0.get('seaice_ext_nh_mkm2', float('nan')):.3f}",
                    f"{m0.get('seaice_ext_sh_mkm2', float('nan')):.3f}",
                    f"{m0.get('seaice_vol_nh_km3', float('nan')):.3f}",
                    f"{m0.get('seaice_vol_sh_km3', float('nan')):.3f}",
                    f"{m0.get('amoc_26n_depth_m', float('nan')):.1f}",
                    f"{m0.get('aabw_30s_sv', float('nan')):.2f}",
                    f"{m0.get('oht_atl_26n_pw', float('nan')):.3f}",
                    "nan", "nan",
                    "nan", "nan", "nan", "nan",
                ])
    elif prev_ohc_j is None and model.ogcm is not None:
        ocn_mesh_r = model.ocn_mesh if model.ocn_mesh is not None else model.mesh
        ocn_sfc_r = model.ocn_sfc if model.ocn_sfc is not None else model.sfc
        m_res = diagnose_ocean_spinup_metrics(model.ogcm, ocn_sfc_r, ocn_mesh_r)
        prev_ohc_j = m_res.get("ohc_j", None)
        prev_ohc_time_sec = model.sim_time_sec

    # Continuous pre-update teacher-student shadow divergence CSV (Host Atmosphere + Land Surface)
    from aoba.crm_self_distill import CRM_SHADOW_KEYS, RAD_SHADOW_KEYS, read_distill_shadow_metrics

    shadow_enabled = bool(cfg.enable_distill_shadow_diag) and (
        bool(cfg.enable_crm_self_distill) or bool(cfg.enable_rad_self_distill)
    )
    shadow_csv = (out_root / "distill_shadow.csv") if shadow_enabled else None
    _SHADOW_HEADER = [
        "year", "date", "step", "sim_days", "co2_ppmv",
        "crm_teacher_frac", "rad_teacher_frac",
        *CRM_SHADOW_KEYS,
        *RAD_SHADOW_KEYS,
    ]
    shadow_stride_steps = max(int(cfg.rad_stride), 1)
    last_shadow_metrics: dict[str, float] = {}
    if shadow_csv is not None and (
        not shadow_csv.exists() or (start_year == 1 and resume_step_in_yr == 0 and not resumed_existing)
    ):
        with open(shadow_csv, "w", newline="") as f_sh:
            csv.writer(f_sh).writerow(_SHADOW_HEADER)

    # Expose initial T2 / OLR / LAI + Earth2014 continents immediately for Tab 5 live globes
    _write_live_preview_atomic(
        model, out_root / "live_preview.npz", start_year, 0.0, float((start_year - 1) * days_per_year), 0.0
    )

    steps_per_day = max(int(round(C.SECONDS_PER_DAY / cfg.dt_host)), 1)
    end_days_limit = model.calendar.days_until(getattr(out_cfg, "end_date", ""))
    max_total_steps = (end_days_limit * steps_per_day) if (end_days_limit is not None and args.years is None) else None
    if end_days_limit is not None and args.years is None:
        # Ensure enough outer year iterations to cover end_date
        years = max(years, int(math.ceil(end_days_limit / 360.0)) + 1)

    sample_stride = max(int(round(out_cfg.monthly_mean_sample_hours * 3600.0 / cfg.dt_host)), 1)
    csv_stride_steps = max(int(out_cfg.daily_csv_interval_days) * steps_per_day, 1)
    preview_stride_steps = max(int(round(3600.0 / cfg.dt_host)), 1)

    import shutil

    single_partition = out_cfg.zarr_dir_partition == "single"
    if single_partition and start_year == 1:
        for p in (out_root / "monthly.zarr", out_root / "snapshots.zarr"):
            if p.exists():
                shutil.rmtree(p)

    stop_by_end_date = False
    for year in range(start_year, years + 1):
        cal_yr_days = days_in_year(model.calendar.start_year + year - 1, model.calendar.calendar_type)
        if args.days_per_year is not None:
            eff_days_per_year = float(args.days_per_year)
        elif abs(float(out_cfg.days_per_year) - C.DAYS_PER_YEAR) > 1.0e-3:
            eff_days_per_year = float(out_cfg.days_per_year)
        else:
            eff_days_per_year = float(cal_yr_days)
        steps_per_year = max(int(round(eff_days_per_year * C.SECONDS_PER_DAY / cfg.dt_host)), 12)
        n_mean_periods = max(int(out_cfg.monthly_mean_periods_per_year), 1)
        mean_bounds = [int(round(m * steps_per_year / float(n_mean_periods))) for m in range(n_mean_periods + 1)]
        n_snap_periods = max(int(out_cfg.snapshots_periods_per_year), 1)
        snap_steps = {int(round(s * steps_per_year / float(n_snap_periods))) for s in range(1, n_snap_periods + 1)}

        y_dir = out_root / f"year_{year:04d}"
        y_dir.mkdir(parents=True, exist_ok=True)
        monthly_zarr = (out_root / "monthly.zarr") if single_partition else (y_dir / "monthly.zarr")
        snap_zarr = (out_root / "snapshots.zarr") if single_partition else (y_dir / "snapshots.zarr")
        cmip7_zarr = (out_root / "cmip7_monthly.zarr") if single_partition else (y_dir / "cmip7_monthly.zarr")
        csv_path = y_dir / "daily_global.csv"

        is_mid_resume = (year == start_year and resume_step_in_yr > 0)

        # Clean partial current-year Zarr only in yearly-partition mode when starting year fresh
        if not single_partition and not is_mid_resume:
            for p in (monthly_zarr, snap_zarr, cmip7_zarr):
                if p.exists():
                    shutil.rmtree(p)

        if out_cfg.daily_csv_enabled and not (is_mid_resume and csv_path.exists()):
            with open(csv_path, "w", newline="") as f_csv:
                csv.writer(f_csv).writerow([
                    "year", "date", "day_in_yr", "sim_days", "ts_k", "olr_wm2", "asr_wm2",
                    "net_toa_wm2", "precip_mm_d", "gpp_umol", "lai", "seaice_mkm2", "wall_s_per_day",
                ])

        day_accum = {"ts": 0.0, "olr": 0.0, "asr": 0.0, "p": 0.0, "gpp": 0.0, "lai": 0.0, "ice": 0.0}
        day_cnt = 0
        mon_accum = (
            dict(resume_mon_accum)
            if (is_mid_resume and resume_mon_accum is not None)
            else {"ts": 0.0, "olr": 0.0, "asr": 0.0, "p": 0.0, "gpp": 0.0, "lai": 0.0, "ice": 0.0}
        )
        mon_cnt = resume_mon_cnt if is_mid_resume else 0
        mon_esm_sums: dict[str, torch.Tensor] = {}
        mon_esm_cnt = 0
        hr_t2_sum: torch.Tensor | None = None
        hr_precip_sum: torch.Tensor | None = None
        hr_olr_sum: torch.Tensor | None = None
        hr_cnt = 0
        t_hr0 = time.perf_counter()
        m_idx = resume_m_idx if is_mid_resume else 0
        m_sums: dict[str, torch.Tensor] = resume_m_sums if is_mid_resume else {}
        m_cnt = resume_m_cnt if is_mid_resume else 0
        step_start = (resume_step_in_yr + 1) if is_mid_resume else 1
        resume_step_in_yr = 0
        resume_m_idx = 0
        resume_m_cnt = 0
        resume_m_sums = {}
        resume_mon_accum = None
        resume_mon_cnt = 0
        t_day0 = time.perf_counter()
        prev_cal_month = model.calendar.date_at(model.sim_time_sec).month

        for step_in_yr in range(step_start, steps_per_year + 1):
            is_preview_step = (step_in_yr % preview_stride_steps == 0)
            is_monthly_end = (
                out_cfg.monthly_mean_enabled
                and m_idx + 1 < len(mean_bounds)
                and step_in_yr >= mean_bounds[m_idx + 1]
            )
            is_monthly_sample = out_cfg.monthly_mean_enabled and (
                (step_in_yr % sample_stride == 0) or is_monthly_end
            )
            is_snap_step = out_cfg.snapshots_enabled and (step_in_yr in snap_steps)
            next_cal_month = model.calendar.date_at(model.sim_time_sec + cfg.dt_host).month
            is_month_restart = (
                getattr(out_cfg, "restart_monthly", True)
                and (next_cal_month != prev_cal_month)
                and (step_in_yr < steps_per_year)
            )
            is_year_end = (step_in_yr == steps_per_year) or (
                max_total_steps is not None and (model.step_count + 1) >= max_total_steps
            )
            is_sync_step = is_preview_step or is_monthly_sample or is_snap_step or is_month_restart or is_year_end

            diag = step_aoba_model(
                model,
                compute_diagnostics=bool(out_cfg.daily_csv_enabled),
                sync_host_surface=is_sync_step,
            )

            # Zero-overhead immediate NaN/Inf guard on already-synced scalar diagnostics
            if bool(out_cfg.daily_csv_enabled):
                _dvals = (
                    ("Ts", diag.global_mean_t_sfc_k),
                    ("OLR", diag.global_mean_olr_wm2),
                    ("ASR", diag.global_mean_asr_wm2),
                    ("P", diag.global_mean_precip_mm_day),
                    ("GPP", diag.land_mean_gpp_umol),
                    ("LAI", diag.land_mean_lai),
                    ("Ice", diag.seaice_area_million_km2),
                )
                _bad_d = [f"{k}={v}" for k, v in _dvals if not math.isfinite(v)]
                if _bad_d:
                    _abort_if_nonfinite(
                        model, year, step_in_yr, diag.date_str,
                        reason="non-finite step diagnostics: " + ", ".join(_bad_d),
                    )

            day_accum["ts"] += diag.global_mean_t_sfc_k
            day_accum["olr"] += diag.global_mean_olr_wm2
            day_accum["asr"] += diag.global_mean_asr_wm2
            day_accum["p"] += diag.global_mean_precip_mm_day
            day_accum["gpp"] += diag.land_mean_gpp_umol
            day_accum["lai"] += diag.land_mean_lai
            day_accum["ice"] += diag.seaice_area_million_km2
            day_cnt += 1

            mon_accum["ts"] += diag.global_mean_t_sfc_k
            mon_accum["olr"] += diag.global_mean_olr_wm2
            mon_accum["asr"] += diag.global_mean_asr_wm2
            mon_accum["p"] += diag.global_mean_precip_mm_day
            mon_accum["gpp"] += diag.land_mean_gpp_umol
            mon_accum["lai"] += diag.land_mean_lai
            mon_accum["ice"] += diag.seaice_area_million_km2
            mon_cnt += 1
            # Zero-sync on-device monthly accumulators for intermittent ESM fluxes (MAML fires, lightning, BNF, O3 dep)
            _esm_step: dict[str, torch.Tensor] = {}
            if model.land_carbon is not None:
                _lc = model.land_carbon
                _esm_step["lc_fire_c"] = _lc.fire_c_kgc_m2s
                _esm_step["lc_ba_s"] = _lc.burned_frac_s
                if _lc.bnf_kgn_m2s is not None:
                    _esm_step["lc_bnf"] = _lc.bnf_kgn_m2s
                if _lc.n_leach_kgn_m2s is not None:
                    _esm_step["lc_leach"] = _lc.n_leach_kgn_m2s
            if model.last_lightning is not None:
                _esm_step["lgt_tot"] = model.last_lightning.flash_tot_cell
                _esm_step["lgt_cg"] = model.last_lightning.flash_cg_cell
                _esm_step["lgt_lnox"] = model.last_lightning.n_dep_lnox_cell
            if model.ozone is not None:
                _esm_step["oz_dep"] = model.ozone.o3_drydep_kgm2s
            if _esm_step:
                if mon_esm_cnt == 0:
                    mon_esm_sums = {k: v.detach().clone() for k, v in _esm_step.items()}
                else:
                    for k, v in _esm_step.items():
                        if k in mon_esm_sums:
                            mon_esm_sums[k] += v.detach()
                        else:
                            mon_esm_sums[k] = v.detach().clone()
                mon_esm_cnt += 1
            if cmip7_acc is not None:
                cmip7_acc.update_step_extrema(
                    model,
                    end_of_day=(step_in_yr % steps_per_day == 0 or step_in_yr == steps_per_year),
                )

            # Accumulate 1-hour-mean 2D preview fields on device
            _, _, temp0_p = diagnose_pressure_temperature(
                model.atmos.rho_d[0:1], model.atmos.theta_m[0:1], model.atmos.q_v[0:1]
            )
            if is_sync_step or model.sfc_sub is None:
                t_sfc_p = torch.where(model.mesh.surf_type == SURF_OCEAN, model.sfc.t_ocean_ml, model.sfc.t_canopy)
            else:
                from aoba.coupled_model import resolve_crm_sfc_tiles

                ss = model.sfc_sub
                n_sub = resolve_crm_sfc_tiles(model.cfg)
                t_sfc_p = torch.where(ss.surf_type == SURF_OCEAN, ss.t_ocean_ml, ss.t_canopy).reshape(model.mesh.n_cells, n_sub).mean(dim=-1)
            t2_inst = 0.35 * t_sfc_p + 0.65 * temp0_p[0]
            rain_p = model.rad_cache.get("rain", torch.zeros_like(t2_inst))
            snow_p = model.rad_cache.get("snow", torch.zeros_like(t2_inst))
            precip_inst = (rain_p + snow_p) * C.SECONDS_PER_DAY
            olr_inst = model.rad_cache.get("olr", 0.85 * C.STEFAN_BOLTZMANN * torch.clamp(t_sfc_p, min=150.0) ** 4)
            if hr_t2_sum is None:
                hr_t2_sum = t2_inst.clone()
                hr_precip_sum = precip_inst.clone()
                hr_olr_sum = olr_inst.clone()
            else:
                hr_t2_sum += t2_inst
                hr_precip_sum += precip_inst
                hr_olr_sum += olr_inst
            hr_cnt += 1

            if step_in_yr % preview_stride_steps == 0 and hr_cnt > 0:
                wall_hr = time.perf_counter() - t_hr0
                t_hr0 = time.perf_counter()
                _write_live_preview_atomic(
                    model,
                    out_root / "live_preview.npz",
                    year,
                    float(step_in_yr) / float(steps_per_day),
                    model.sim_time_sec / C.SECONDS_PER_DAY,
                    wall_hr * 24.0,
                    t2_k_override=hr_t2_sum / float(hr_cnt),
                    precip_mm_day_override=hr_precip_sum / float(hr_cnt),
                    olr_override=hr_olr_sum / float(hr_cnt),
                )
                hr_t2_sum = None
                hr_precip_sum = None
                hr_olr_sum = None
                hr_cnt = 0

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
                    bio_m = model.sfc.bio
                    fc4_m = torch.clamp(bio_m.f_c4, 0.0, 1.0)
                    fevg_m = torch.clamp(bio_m.f_evg, 0.0, 1.0) if bio_m.f_evg is not None else torch.zeros_like(fc4_m)
                    vcmax_m = (1.0 - fc4_m) * bio_m.vcmax25_c3 + fc4_m * bio_m.vcmax25_c4
                    chi_m = bio_m.chi_leaf if bio_m.chi_leaf is not None else torch.zeros_like(fc4_m)
                    dz_4m = model.atmos.dz_c[:, :, None, None]
                    lwp_sub_m = (model.crm.rho_d * model.crm.q_c * dz_4m).sum(dim=0)
                    iwp_sub_m = (model.crm.rho_d * model.crm.q_i * dz_4m).sum(dim=0)
                    nc_m = model.mesh.n_cells
                    clt_m = (((lwp_sub_m + iwp_sub_m) > 1.0e-4).to(olr_c.dtype).reshape(nc_m, -1).mean(dim=-1)) * 100.0
                    lwp_m = lwp_sub_m.reshape(nc_m, -1).mean(dim=-1)
                    iwp_m = iwp_sub_m.reshape(nc_m, -1).mean(dim=-1)
                    prw_m = (model.atmos.rho_d * model.atmos.q_v * model.atmos.dz_c).sum(dim=0)
                    f_ice_m = torch.clamp(torch.sqrt(torch.clamp(model.sfc.m_seaice, min=0.0) / 450.0), 0.0, 1.0)
                    sithick_m = torch.where(
                        f_ice_m > 1.0e-3,
                        (torch.clamp(model.sfc.m_seaice, min=0.0) / C.RHO_SEAICE) / torch.clamp(f_ice_m, min=1.0e-3),
                        torch.zeros_like(f_ice_m),
                    )
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
                        "h_canopy": bio_m.h_canopy,
                        "f_evg": fevg_m,
                        "f_c4": fc4_m,
                        "s_root_cap": bio_m.s_root_cap,
                        "vcmax25": vcmax_m,
                        "chi_leaf": chi_m,
                        "clt": clt_m,
                        "lwp": lwp_m,
                        "iwp": iwp_m,
                        "prw": prw_m,
                        "siconc": f_ice_m * 100.0,
                        "sithick": sithick_m,
                    })
                    if model.ogcm is not None:
                        og = model.ogcm
                        sample["ocn_eta"] = og.eta
                        sample["ocn_sss"] = og.sal[0]
                        if og.h_bl is not None:
                            sample["ocn_h_bl"] = og.h_bl
                        sample["precip_snow_mm_day"] = snow * C.SECONDS_PER_DAY
                        sample["sw1_dir_sfc"] = model.rad_cache.get("sw1_dir", torch.zeros_like(model.sfc.t_ocean_ml))
                        sample["sw1_dif_sfc"] = model.rad_cache.get("sw1_dif", torch.zeros_like(model.sfc.t_ocean_ml))
                        sample["sw2_dir_sfc"] = model.rad_cache.get("sw2_dir", torch.zeros_like(model.sfc.t_ocean_ml))
                        sample["sw2_dif_sfc"] = model.rad_cache.get("sw2_dif", torch.zeros_like(model.sfc.t_ocean_ml))
                        sample["lw_dn_sfc"] = model.rad_cache.get("lw_dn", torch.zeros_like(model.sfc.t_ocean_ml))
                        sd_sub = model.last_sfc_diag_sub
                        if sd_sub is not None:
                            from aoba.coupled_model import resolve_crm_sfc_tiles

                            n_sub_c = resolve_crm_sfc_tiles(model.cfg)
                            nc_c = model.mesh.n_cells
                            sample["tau_x_sfc"] = sd_sub.tau_x.reshape(nc_c, n_sub_c).mean(dim=-1)
                            sample["tau_y_sfc"] = sd_sub.tau_y.reshape(nc_c, n_sub_c).mean(dim=-1)
                            sample["runoff_total"] = sd_sub.runoff_total.reshape(nc_c, n_sub_c).mean(dim=-1)
                            sample["runoff_glacier"] = sd_sub.runoff_ice.reshape(nc_c, n_sub_c).mean(dim=-1)
                    if model.ozone is not None:
                        sample["o3_tco_du"] = model.ozone.tco_du
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
                    if model.ozone is not None:
                        sample["o3_vmr"] = model.ozone.o3_vmr
                    if model.ogcm is not None:
                        og = model.ogcm
                        ocn_mesh_z = model.ocn_mesh if model.ocn_mesh is not None else model.mesh
                        uo_c, vo_c = reconstruct_cell_winds(ocn_mesh_z, og.u_edge * og.wet_e.to(og.u_edge.dtype))
                        wo_c = 0.5 * (og.w_vel[:-1] + og.w_vel[1:])
                        sample.update({
                            "ocn_temp": og.temp,
                            "ocn_sal": og.sal,
                            "ocn_u_zonal": uo_c,
                            "ocn_v_merid": vo_c,
                            "ocn_w_vel": wo_c,
                        })
                if out_cfg.monthly_mean_include_subgrid_2d:
                    sample.update({k: val for k, (val, _) in diagnose_subgrid_2d_fields(model).items()})

                if m_cnt == 0:
                    m_sums = {k: v.clone() for k, v in sample.items()}
                else:
                    for k, v in sample.items():
                        if k in m_sums:
                            m_sums[k] += v
                        else:
                            m_sums[k] = v.clone()
                m_cnt += 1
                if cmip7_acc is not None:
                    cmip7_acc.sample(model)

            if shadow_csv is not None and (
                step_in_yr % shadow_stride_steps == 0
                or step_in_yr == 1
                or step_in_yr == steps_per_year
            ):
                sm = read_distill_shadow_metrics(model.crm_distill, model.rad_distill)
                if sm:
                    last_shadow_metrics = sm
                    co2_sh = model.current_co2_ppmv()
                    sh_row = [
                        year,
                        diag.date_str,
                        int(model.step_count),
                        f"{diag.sim_time_days:.4f}",
                        f"{co2_sh:.2f}",
                        f"{sm.get('crm_teacher_frac', float('nan')):.4f}",
                        f"{sm.get('rad_teacher_frac', float('nan')):.4f}",
                    ]
                    for k_sh in (*CRM_SHADOW_KEYS, *RAD_SHADOW_KEYS):
                        v_sh = sm.get(k_sh, float("nan"))
                        sh_row.append(f"{v_sh:.5g}" if math.isfinite(v_sh) else "nan")
                    with open(shadow_csv, "a", newline="") as f_sh:
                        csv.writer(f_sh).writerow(sh_row)

            if step_in_yr % csv_stride_steps == 0 or step_in_yr == steps_per_year:
                synchronize_device(dev)
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
                            year, diag.date_str, f"{day_num:.2f}", f"{diag.sim_time_days:.2f}",
                            f"{ts_m:.3f}", f"{olr_m:.2f}", f"{asr_m:.2f}", f"{asr_m - olr_m:.2f}",
                            f"{day_accum['p'] * inv_d:.3f}", f"{day_accum['gpp'] * inv_d:.3f}",
                            f"{day_accum['lai'] * inv_d:.3f}", f"{day_accum['ice'] * inv_d:.3f}", f"{wall_day:.2f}",
                        ])
                if int(round(day_num)) % 10 == 0 or step_in_yr == steps_per_year:
                    print(
                        f"[{diag.date_str} | Y{year:02d} D{day_num:5.1f}] Ts={ts_m:6.2f} K | NetTOA={asr_m - olr_m:+5.2f} | "
                        f"P={day_accum['p'] * inv_d:4.2f} mm/d | LAI={day_accum['lai'] * inv_d:4.2f} | "
                        f"Ice={day_accum['ice'] * inv_d:5.2f} Mkm2 | {wall_day:5.1f} s/d",
                        flush=True,
                    )
                    if last_shadow_metrics:
                        sm_p = last_shadow_metrics
                        co2_p = model.current_co2_ppmv()
                        print(
                            f"  [SHADOW ATMOS | CO2={co2_p:6.1f} ppmv] "
                            f"dTh={sm_p.get('atm_rmse_thm_k', float('nan')):.3f}K "
                            f"dQn={sm_p.get('atm_rmse_qn_gkg', float('nan')):.3f}g/kg "
                            f"dP={sm_p.get('atm_rmse_prec_mmd', float('nan')):.2f}mm/d(r={sm_p.get('atm_corr_prec', float('nan')):.2f}) | "
                            f"dOLR={sm_p.get('rad_rmse_olr_wm2', float('nan')):.2f}({sm_p.get('rad_bias_olr_wm2', float('nan')):+.2f}) "
                            f"dASR={sm_p.get('rad_rmse_asr_wm2', float('nan')):.2f} "
                            f"dQrad={sm_p.get('rad_rmse_qrad_kd', float('nan')):.2f}K/d | "
                            f"unc={sm_p.get('atm_unc_mean', float('nan')):.3f}",
                            flush=True,
                        )
                        print(
                            f"  [SHADOW LAND  | tiles={int(sm_p.get('sfc_n_land_tiles', 0))}] "
                            f"dTh0={sm_p.get('sfc_rmse_thm0_k', float('nan')):.3f}K "
                            f"dQv0={sm_p.get('sfc_rmse_qv0_gkg', float('nan')):.3f}g/kg "
                            f"dU0={sm_p.get('sfc_rmse_wind0_ms', float('nan')):.2f}m/s "
                            f"dP={sm_p.get('sfc_rmse_prec_mmd', float('nan')):.2f}mm/d(r_an={sm_p.get('sfc_corr_prec_anom', float('nan')):.2f}) | "
                            f"dSWdn={sm_p.get('sfc_rad_rmse_swdn_wm2', float('nan')):.1f} "
                            f"dPAR={sm_p.get('sfc_rad_rmse_par_wm2', float('nan')):.1f} "
                            f"dLWdn={sm_p.get('sfc_rad_rmse_lwd_wm2', float('nan')):.2f}({sm_p.get('sfc_rad_bias_lwd_wm2', float('nan')):+.2f}) W/m2",
                            flush=True,
                        )
                for k in day_accum:
                    day_accum[k] = 0.0
                day_cnt = 0

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
                # Log monthly coupled + 3D OGCM spin-up monitoring metrics
                inv_m = 1.0 / float(max(mon_cnt, 1))
                m_ts = mon_accum["ts"] * inv_m
                m_olr = mon_accum["olr"] * inv_m
                m_asr = mon_accum["asr"] * inv_m
                m_net_toa = m_asr - m_olr
                m_p = mon_accum["p"] * inv_m
                m_gpp = mon_accum["gpp"] * inv_m
                m_lai = mon_accum["lai"] * inv_m
                m_ice = mon_accum["ice"] * inv_m
                ocn_mesh_d = model.ocn_mesh if model.ocn_mesh is not None else model.mesh
                ocn_sfc_d = model.ocn_sfc if model.ocn_sfc is not None else model.sfc
                ocn_m = diagnose_ocean_spinup_metrics(model.ogcm, ocn_sfc_d, ocn_mesh_d) if model.ogcm is not None else {}
                ohc_now = ocn_m.get("ohc_j", float("nan"))
                if prev_ohc_j is not None and prev_ohc_time_sec is not None and math.isfinite(ohc_now):
                    dt_mon = max(model.sim_time_sec - prev_ohc_time_sec, 1.0)
                    ohc_flux_wm2 = (ohc_now - prev_ohc_j) / (dt_mon * a_earth_m2)
                else:
                    ohc_flux_wm2 = float("nan")
                if math.isfinite(ohc_now):
                    prev_ohc_j = ohc_now
                    prev_ohc_time_sec = model.sim_time_sec
                mon_num = m_idx + 1
                # Monthly global cloud, EEO vegetation, land-carbon, and shadow-divergence summaries
                a_h64 = model.mesh.area_cell.detach().cpu().to(torch.float64)
                a_h_tot = max(float(a_h64.sum().item()), 1.0)
                is_lnd_64 = (model.mesh.surf_type.detach().cpu() == SURF_LAND).to(torch.float64)
                a_lnd_tot = max(float((a_h64 * is_lnd_64).sum().item()), 1.0)
                def _gmean(key: str) -> float:
                    if m_cnt <= 0 or key not in m_sums:
                        return float("nan")
                    return float(((m_sums[key].detach().cpu().double() / float(m_cnt)) * a_h64).sum().item() / a_h_tot)
                def _lmean(t_dev: torch.Tensor) -> float:
                    return float((t_dev.detach().cpu().double() * a_h64 * is_lnd_64).sum().item() / a_lnd_tot)
                clt_mon = _gmean("clt")
                lwp_mon = _gmean("lwp") * 1.0e3
                iwp_mon = _gmean("iwp") * 1.0e3
                prw_mon = _gmean("prw")
                bio_now = model.sfc.bio
                lai_lnd = _lmean(bio_now.lai)
                hc_lnd = _lmean(bio_now.h_canopy)
                fevg_lnd = _lmean(torch.clamp(bio_now.f_evg, 0.0, 1.0)) if bio_now.f_evg is not None else float("nan")
                fc4_lnd = _lmean(torch.clamp(bio_now.f_c4, 0.0, 1.0))
                # Global terrestrial GPP [PgC/yr] from land-only monthly mean gpp [umol CO2 m^-2 s^-1]
                gpp_pgc_yr = m_gpp * 12.011e-9 * a_lnd_tot * (C.DAYS_PER_YEAR * C.SECONDS_PER_DAY) * 1.0e-12
                lc_veg_pgc = float("nan")
                lc_soil_pgc = float("nan")
                if model.land_carbon is not None:
                    from aoba.land_carbon import diagnose_land_carbon_metrics

                    _lc_sp = diagnose_land_carbon_metrics(model.land_carbon, model.mesh.area_cell, model.mesh.surf_type)
                    lc_veg_pgc = float(_lc_sp.get("c_veg_pgc", float("nan")))
                    lc_soil_pgc = float(_lc_sp.get("c_soil_pgc", float("nan")))
                with open(spinup_csv, "a", newline="") as f_sp:
                    csv.writer(f_sp).writerow([
                        year, mon_num, diag.date_str, f"{diag.sim_time_days:.2f}",
                        f"{m_ts:.3f}", f"{m_olr:.2f}", f"{m_asr:.2f}", f"{m_net_toa:.3f}", f"{ohc_flux_wm2:.3f}",
                        f"{m_p:.3f}", f"{m_gpp:.3f}", f"{m_lai:.3f}", f"{m_ice:.3f}",
                        f"{ocn_m.get('seaice_nh_mkm2', float('nan')):.3f}",
                        f"{ocn_m.get('seaice_sh_mkm2', float('nan')):.3f}",
                        f"{ocn_m.get('sst_global_k', float('nan')):.4f}",
                        f"{ocn_m.get('sss_global_psu', float('nan')):.4f}",
                        f"{ocn_m.get('t_ocn_global_k', float('nan')):.5f}",
                        f"{ocn_m.get('t_ocn_0_300m_k', float('nan')):.5f}",
                        f"{ocn_m.get('t_ocn_300_1500m_k', float('nan')):.5f}",
                        f"{ocn_m.get('t_ocn_deep_k', float('nan')):.5f}",
                        f"{ocn_m.get('s_ocn_global_psu', float('nan')):.5f}",
                        f"{ocn_m.get('ohc_zj', float('nan')):.4f}",
                        f"{ocn_m.get('ocn_ke_jm2', float('nan')):.2f}",
                        f"{ocn_m.get('ocn_eta_rms_m', float('nan')):.4f}",
                        f"{ocn_m.get('ocn_hbl_mean_m', float('nan')):.2f}",
                        f"{ocn_m.get('ocn_hbl_max_m', float('nan')):.1f}",
                        f"{ocn_m.get('amoc_26n_sv', float('nan')):.2f}",
                        f"{ocn_m.get('amoc_max_sv', float('nan')):.2f}",
                        f"{ocn_m.get('acc_drake_sv', float('nan')):.2f}",
                        f"{model.current_co2_ppmv():.2f}",
                        f"{clt_mon:.2f}", f"{lwp_mon:.2f}", f"{iwp_mon:.2f}", f"{prw_mon:.2f}",
                        f"{gpp_pgc_yr:.2f}", f"{lai_lnd:.3f}", f"{hc_lnd:.2f}", f"{fevg_lnd:.3f}", f"{fc4_lnd:.3f}",
                        f"{ocn_m.get('seaice_ext_nh_mkm2', float('nan')):.3f}",
                        f"{ocn_m.get('seaice_ext_sh_mkm2', float('nan')):.3f}",
                        f"{ocn_m.get('seaice_vol_nh_km3', float('nan')):.3f}",
                        f"{ocn_m.get('seaice_vol_sh_km3', float('nan')):.3f}",
                        f"{ocn_m.get('amoc_26n_depth_m', float('nan')):.1f}",
                        f"{ocn_m.get('aabw_30s_sv', float('nan')):.2f}",
                        f"{ocn_m.get('oht_atl_26n_pw', float('nan')):.3f}",
                        f"{lc_veg_pgc:.1f}", f"{lc_soil_pgc:.1f}",
                        f"{last_shadow_metrics.get('atm_rmse_thm_k', float('nan')):.4f}",
                        f"{last_shadow_metrics.get('rad_rmse_olr_wm2', float('nan')):.3f}",
                        f"{last_shadow_metrics.get('sfc_rmse_thm0_k', float('nan')):.4f}",
                        f"{last_shadow_metrics.get('sfc_rad_rmse_lwd_wm2', float('nan')):.3f}",
                    ])
                if ocn_m:
                    print(
                        f"[OCEAN Y{year:03d} M{mon_num:02d} | {diag.date_str}] "
                        f"NetTOA={m_net_toa:+5.2f} dOHC={ohc_flux_wm2:+5.2f} W/m2 | "
                        f"Tocn(glob/up/mid/deep)={ocn_m['t_ocn_global_k']:.3f}/{ocn_m['t_ocn_0_300m_k']:.2f}/"
                        f"{ocn_m['t_ocn_300_1500m_k']:.2f}/{ocn_m['t_ocn_deep_k']:.3f} K | "
                        f"S={ocn_m['s_ocn_global_psu']:.4f} psu | "
                        f"AMOC(26N/max)={ocn_m['amoc_26n_sv']:.1f}/{ocn_m['amoc_max_sv']:.1f} Sv | "
                        f"ACC={ocn_m['acc_drake_sv']:.1f} Sv | KE={ocn_m['ocn_ke_jm2']:.0f} J/m2 | "
                        f"MLD={ocn_m['ocn_hbl_mean_m']:.0f}m(max {ocn_m['ocn_hbl_max_m']:.0f}m) | "
                        f"Ice(NH/SH)={ocn_m['seaice_nh_mkm2']:.1f}/{ocn_m['seaice_sh_mkm2']:.1f} Mkm2",
                        flush=True,
                    )
                if model.ogcm is not None and model.ogcm.bgc is not None:
                    from aoba.ocean_bgc import diagnose_ocean_bgc_metrics

                    bgc_m = diagnose_ocean_bgc_metrics(model.ogcm.bgc, model.ogcm, ocn_mesh_d)
                    if bgc_m:
                        print(
                            f"[FlexPFT Y{year:03d} M{mon_num:02d} | {diag.date_str}] "
                            f"NPP={bgc_m['ocean_npp_pgc_yr']:.1f} PgC/yr | "
                            f"EPC100={bgc_m['ocean_epc100_pgc_yr']:.1f} PgC/yr | "
                            f"fgCO2={bgc_m['ocean_fgco2_pgc_yr']:+.2f} PgC/yr | "
                            f"Chl={bgc_m['ocean_chl_sfc_mg_m3']:.3f} mg/m3 | "
                            f"pCO2={bgc_m['ocean_spco2_uatm']:.1f} uatm | pH={bgc_m['ocean_ph_sfc']:.3f}",
                            flush=True,
                        )
                esm_parts: list[str] = []
                inv_esm = 1.0 / float(max(mon_esm_cnt, 1))
                sec_yr = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
                a_host_64 = model.mesh.area_cell.detach().cpu().to(torch.float64)
                if model.river is not None:
                    q_riv_sv = float((model.river.mouth_flux_kgm2s.detach().cpu().double() * a_host_64).sum().item()) * 1.0e-9
                    esm_parts.append(f"River={q_riv_sv:.3f} Sv")
                if model.land_carbon is not None:
                    from aoba.land_carbon import diagnose_land_carbon_metrics

                    lc_m = diagnose_land_carbon_metrics(model.land_carbon, model.mesh.area_cell, model.mesh.surf_type)
                    # Prefer true monthly-integrated fluxes for intermittent discrete MAML fires & N fluxes
                    fire_pg_yr = lc_m["land_fire_pgc_yr"]
                    ba_mha_yr = lc_m["burned_area_mha_yr"]
                    bnf_tgn_yr = lc_m.get("bnf_tgn_yr", 0.0)
                    leach_tgn_yr = lc_m.get("n_leach_tgn_yr", 0.0)
                    if mon_esm_cnt > 0 and "lc_fire_c" in mon_esm_sums:
                        m_lc = mon_esm_sums["lc_fire_c"].shape[0]
                        nc_h = a_host_64.shape[0]
                        if m_lc != nc_h and (m_lc % nc_h == 0):
                            n_s = m_lc // nc_h
                            a_lc = (a_host_64 / float(n_s)).unsqueeze(1).expand(nc_h, n_s).reshape(m_lc)
                        else:
                            a_lc = a_host_64
                        fire_pg_yr = float((mon_esm_sums["lc_fire_c"].detach().cpu().double() * inv_esm * a_lc).sum().item() * sec_yr * 1.0e-12)
                        ba_mha_yr = float((mon_esm_sums["lc_ba_s"].detach().cpu().double() * inv_esm * a_lc).sum().item() * sec_yr * 1.0e-10)
                        if "lc_bnf" in mon_esm_sums:
                            bnf_tgn_yr = float((mon_esm_sums["lc_bnf"].detach().cpu().double() * inv_esm * a_lc).sum().item() * sec_yr * 1.0e-9)
                        if "lc_leach" in mon_esm_sums:
                            leach_tgn_yr = float((mon_esm_sums["lc_leach"].detach().cpu().double() * inv_esm * a_lc).sum().item() * sec_yr * 1.0e-9)
                    esm_parts.append(
                        f"LandC={lc_m['c_land_total_pgc']:.0f} PgC "
                        f"(Veg={lc_m['c_veg_pgc']:.0f}, Soil={lc_m['c_soil_pgc']:.0f}, "
                        f"Fire={fire_pg_yr:.2f} PgC/yr, BA={ba_mha_yr:.1f} Mha/yr)"
                    )
                    if "n_min_pgn" in lc_m:
                        esm_parts.append(
                            f"LandN(Nmin={lc_m['n_min_pgn']:.3f} PgN, BNF={bnf_tgn_yr:.1f} TgN/yr, "
                            f"Leach={leach_tgn_yr:.1f}, phiN={lc_m.get('phi_n_mean', 1.0):.2f}, "
                            f"froot={lc_m.get('f_root_mean', 0.35):.2f})"
                        )
                if model.last_lightning is not None:
                    if mon_esm_cnt > 0 and "lgt_tot" in mon_esm_sums:
                        ft_m = mon_esm_sums["lgt_tot"].detach().cpu().double() * inv_esm
                        fcg_m = mon_esm_sums["lgt_cg"].detach().cpu().double() * inv_esm
                        lnox_m = mon_esm_sums["lgt_lnox"].detach().cpu().double() * inv_esm
                        fl_tot_s = float((ft_m * a_host_64).sum().item())
                        fl_cg_s = float((fcg_m * a_host_64).sum().item())
                        cg_pct = 100.0 * fl_cg_s / max(fl_tot_s, 1.0e-12)
                        lnox_tg = float((lnox_m * a_host_64).sum().item() * sec_yr * 1.0e-9)
                        st_cpu = model.mesh.surf_type.detach().cpu()
                        m_l = st_cpu == SURF_LAND
                        m_o = st_cpu == SURF_OCEAN
                        l_den = float((ft_m[m_l] * a_host_64[m_l]).sum().item()) / max(float(a_host_64[m_l].sum().item()), 1.0)
                        o_den = float((ft_m[m_o] * a_host_64[m_o]).sum().item()) / max(float(a_host_64[m_o].sum().item()), 1.0)
                        lo_r = l_den / max(o_den, 1.0e-30)
                        esm_parts.append(
                            f"Light={fl_tot_s:.1f} fl/s (CG={cg_pct:.0f}%, L/O={lo_r:.1f}x, LNOx={lnox_tg:.2f} TgN/yr)"
                        )
                    else:
                        from aoba.lightning import diagnose_lightning_metrics

                        lm = diagnose_lightning_metrics(model.last_lightning, model.mesh.area_cell, model.mesh.surf_type)
                        lo_r = lm.get("land_ocean_flash_ratio", 0.0)
                        esm_parts.append(
                            f"Light={lm['flash_tot_per_s']:.1f} fl/s "
                            f"(CG={lm['cg_frac_pct']:.0f}%, L/O={lo_r:.1f}x, LNOx={lm['lnox_tgn_yr']:.2f} TgN/yr)"
                        )
                if model.ozone is not None:
                    from aoba.ozone_chemistry import diagnose_ozone_metrics

                    oz_m = diagnose_ozone_metrics(model.ozone, model.mesh.area_cell, model.mesh.lat_cell)
                    dep_tg = oz_m["o3_drydep_tg_yr"]
                    if mon_esm_cnt > 0 and "oz_dep" in mon_esm_sums:
                        dep_tg = float((mon_esm_sums["oz_dep"].detach().cpu().double() * inv_esm * a_host_64).sum().item() * sec_yr * 1.0e-9)
                    esm_parts.append(
                        f"O3={oz_m['o3_tco_du']:.0f} DU (tr={oz_m['o3_tco_trop_du']:.0f}/hi={oz_m['o3_tco_high_du']:.0f}, "
                        f"sfc={oz_m['o3_sfc_ppbv']:.1f} ppbv, dep={dep_tg:.0f} Tg/yr)"
                    )
                if model.trace_gas is not None:
                    tg = model.trace_gas
                    esm_parts.append(f"GHG(CH4={tg.ch4_ppmv:.3f}, N2O={tg.n2o_ppmv:.3f} ppmv)")
                if model.weathering is not None:
                    ws = model.weathering
                    esm_parts.append(f"Weath(CO2={ws.co2_ppmv:.1f} ppmv, W={ws.weath_co2_pgc_yr:.3f} PgC/yr)")
                if model.ogcm is not None and model.ogcm.seaice_evp is not None:
                    evp_s = model.ogcm.seaice_evp
                    v_ice_kkm3 = float((evp_s.v_cat.sum(dim=0).detach().cpu().double() * ocn_mesh_d.area_cell.detach().cpu().double()).sum().item()) * 1.0e-12
                    u_ice_max = float(evp_s.u_ice_e.abs().max().item())
                    esm_parts.append(f"EVP_Ice={v_ice_kkm3:.1f}e3 km3 (max|u|={u_ice_max:.2f} m/s)")
                if model.aerosols is not None:
                    a_w = a_host_64
                    a_tot_m2 = float(a_w.sum().item())
                    aod_nat = float(((model.aerosols.aod_ss + model.aerosols.aod_so4 + model.aerosols.aod_bc + model.aerosols.aod_oc).detach().cpu().double() * a_w).sum().item()) / a_tot_m2
                    ccn_mean = float((model.aerosols.ccn_aero_m3.detach().cpu().double() * a_w).sum().item()) * 1.0e-6 / a_tot_m2
                    if model.atmos.q_dust is not None:
                        from aoba.dust import DUST_K_EXT_SW1, DUST_K_EXT_SW1_COARSE, DUST_K_EXT_SW1_FINE
                        if model.atmos.q_dust_fine is not None and model.atmos.q_dust_coarse is not None:
                            dm_d = (model.atmos.rho_d * model.atmos.dz_c).detach().cpu().double()
                            col_f = (dm_d * torch.clamp(model.atmos.q_dust_fine, min=0.0).detach().cpu().double()).sum(dim=0)
                            col_c = (dm_d * torch.clamp(model.atmos.q_dust_coarse, min=0.0).detach().cpu().double()).sum(dim=0)
                            aod_dust = float(((DUST_K_EXT_SW1_FINE * col_f + DUST_K_EXT_SW1_COARSE * col_c) * a_w).sum().item()) / a_tot_m2
                        else:
                            col_dust = (model.atmos.rho_d * model.atmos.dz_c * torch.clamp(model.atmos.q_dust, min=0.0)).sum(dim=0).detach().cpu().double()
                            aod_dust = float((DUST_K_EXT_SW1 * col_dust * a_w).sum().item()) / a_tot_m2
                        esm_parts.append(f"Aero(AOD_nat={aod_nat:.4f}, AOD_dust={aod_dust:.4f}, CCN={ccn_mean:.1f} cm^-3)")
                    else:
                        esm_parts.append(f"Aero(AOD={aod_nat:.4f}, CCN={ccn_mean:.1f} cm^-3)")
                if getattr(model, "meso_nox", None) is not None or getattr(model, "last_red_sprites", None) is not None:
                    a_w_m = a_host_64
                    a_tot_m = float(a_w_m.sum().item())
                    m_bits = []
                    if getattr(model, "meso_nox", None) is not None:
                        mn = model.meso_nox
                        t_mp_min = float(mn.t_mesopause_k.detach().cpu().double().min().item())
                        nox_m = float((mn.nox_meso_ppbv.detach().cpu().double() * a_w_m).sum().item()) / a_tot_m
                        nlc_pct = float((mn.nlc_frac.detach().cpu().double() * a_w_m).sum().item()) / a_tot_m * 100.0
                        m_bits.append(f"Tmp_min={t_mp_min:.1f}K, NOx={nox_m:.1f}ppbv, NLC={nlc_pct:.2f}%")
                    if getattr(model, "last_red_sprites", None) is not None:
                        sp_o = model.last_red_sprites
                        sp_min = float((sp_o.sprite_rate_cell.detach().cpu().double() * a_w_m).sum().item()) * 60.0
                        el_min = float((sp_o.elves_rate_cell.detach().cpu().double() * a_w_m).sum().item()) * 60.0 if getattr(sp_o, "elves_rate_cell", None) is not None else 0.0
                        bj_min = float((sp_o.blue_jet_rate_cell.detach().cpu().double() * a_w_m).sum().item()) * 60.0 if getattr(sp_o, "blue_jet_rate_cell", None) is not None else 0.0
                        gj_hr = float((sp_o.gigantic_jet_rate_cell.detach().cpu().double() * a_w_m).sum().item()) * 3600.0 if getattr(sp_o, "gigantic_jet_rate_cell", None) is not None else 0.0
                        m_bits.append(f"TLE(Sp={sp_min:.2f}/m, Elv={el_min:.1f}/m, BJ={bj_min:.2f}/m, GJ={gj_hr:.2f}/h)")
                    esm_parts.append(f"Meso({', '.join(m_bits)})")
                if esm_parts:
                    print(f"[ESM Y{year:03d} M{mon_num:02d} | {diag.date_str}] " + " | ".join(esm_parts), flush=True)
                for k in mon_accum:
                    mon_accum[k] = 0.0
                mon_cnt = 0
                mon_esm_sums.clear()
                mon_esm_cnt = 0
                if cmip7_acc is not None and cmip7_acc.sample_count > 0:
                    append_zarr(cmip7_acc.to_xarray(model), cmip7_zarr)
                    cmip7_acc.reset()
                if getattr(model, "crm_distill", None) is not None or getattr(model, "rad_distill", None) is not None:
                    from aoba.crm_self_distill import consume_distill_shadow_maps

                    consume_distill_shadow_maps(
                        getattr(model, "crm_distill", None),
                        getattr(model, "rad_distill", None),
                        reset=True,
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

            cur_cal_month = model.calendar.date_at(model.sim_time_sec).month
            if getattr(out_cfg, "restart_monthly", True) and cur_cal_month != prev_cal_month and step_in_yr < steps_per_year:
                _save_restart_atomic(
                    model,
                    y_dir / "restart_mid.pt",
                    extra={
                        "resume_step_in_yr": step_in_yr,
                        "resume_m_idx": m_idx,
                        "resume_m_cnt": m_cnt,
                        "resume_m_sums": {k: v.detach().cpu() for k, v in m_sums.items()},
                        "resume_mon_accum": dict(mon_accum),
                        "resume_mon_cnt": mon_cnt,
                        "prev_ohc_j": prev_ohc_j,
                        "prev_ohc_time_sec": prev_ohc_time_sec,
                        "resume_cmip7_state": cmip7_acc.state_dict_cpu() if cmip7_acc is not None else None,
                        "applied_config": applied_cfg_snapshot,
                    },
                )
                print(f"[Saved mid-year restart at {diag.date_str} (step {step_in_yr}/{steps_per_year})]", flush=True)
                stop_flag = out_root / "STOP_AFTER_MONTH"
                if stop_flag.exists():
                    stop_flag.unlink(missing_ok=True)
                    print(f"[Graceful STOP_AFTER_MONTH requested; exiting cleanly after saving {y_dir / 'restart_mid.pt'}]", flush=True)
                    return
            prev_cal_month = cur_cal_month

            if max_total_steps is not None and model.step_count >= max_total_steps:
                stop_by_end_date = True
                break

        if year % max(int(out_cfg.restart_interval_years), 1) == 0 or year == years or stop_by_end_date:
            _save_restart_atomic(
                model,
                y_dir / "restart.pt",
                extra={
                    "prev_ohc_j": prev_ohc_j,
                    "prev_ohc_time_sec": prev_ohc_time_sec,
                    "applied_config": applied_cfg_snapshot,
                },
            )
            (y_dir / "_COMPLETE").touch()
            (y_dir / "restart_mid.pt").unlink(missing_ok=True)
        print(f"=== Completed Year {year:04d} / {years:04d} -> {y_dir} ===", flush=True)
        stop_flag = out_root / "STOP_AFTER_MONTH"
        if stop_flag.exists():
            stop_flag.unlink(missing_ok=True)
            print(f"[Graceful STOP_AFTER_MONTH requested; exiting cleanly after completing Year {year:04d}]", flush=True)
            return
        if stop_by_end_date:
            print(f"Reached target end_date ({out_cfg.end_date}) at {model.calendar.date_at(model.sim_time_sec).iso_date}.", flush=True)
            break


if __name__ == "__main__":
    main()
