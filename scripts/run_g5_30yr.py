#!/usr/bin/env python3
"""Long-duration AOBA climate integration runner with 1-year-per-directory Zarr v3 output & restart."""

from __future__ import annotations

import argparse
import csv
from dataclasses import is_dataclass
import math
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


def _restore_surface(sfc_target, payload: dict, device: torch.device) -> None:
    """Restore serialized `SurfaceState` tensors onto `device`."""
    for k, v in payload.items():
        if k == "bio":
            for bk, bv in v.items():
                setattr(sfc_target.bio, bk, bv.to(device=device) if isinstance(bv, torch.Tensor) else bv)
        else:
            setattr(sfc_target, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
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
    # Backward-compatible warm-start for evergreen/deciduous leaf-habit fields
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
    if model.aerosols is not None:
        checks.append(("aerosols.q_ss", model.aerosols.q_ss))

    n_sub = max(int(model.cfg.crm_ny * model.cfg.crm_nx), 1)
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
                        torch.clamp(54.0 - 0.0024 * (z - 51000.0), min=-28.0),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hydrostatically consistent vertical remap + stratospheric extension `[nz_old, N] -> [nz_new, N]`."""
    zo = z_old.to(torch.float64)
    zn = z_new.to(torch.float64)
    p_old, _, t_old = diagnose_pressure_temperature(
        rho_old.to(torch.float64), thm_old.to(torch.float64), qv_old.to(torch.float64)
    )
    ln_p_interp = _interp_vertical_batch(zo, torch.log(torch.clamp(p_old, min=1.0)), zn)
    t_interp = _interp_vertical_batch(zo, t_old, zn)
    qv_interp = torch.clamp(_interp_vertical_batch(zo, qv_old.to(torch.float64), zn), min=1.0e-7)
    qc_interp = torch.clamp(_interp_vertical_batch(zo, qc_old.to(torch.float64), zn), min=0.0)
    qi_interp = torch.clamp(_interp_vertical_batch(zo, qi_old.to(torch.float64), zn), min=0.0)
    qp_interp = torch.clamp(_interp_vertical_batch(zo, qp_old.to(torch.float64), zn), min=0.0)

    z_top_old = zo[-1:].expand_as(zn)
    above = zn > z_top_old
    t_new = torch.where(above, _stratosphere_temp_extension(zn, z_top_old, t_old[-1:].expand_as(zn)), t_interp)
    qv_top_dry = torch.clamp(qv_old[-1:].to(torch.float64), min=1.0e-7, max=4.0e-6).expand_as(zn)
    qv_new = torch.where(above, qv_top_dry, qv_interp)
    qc_new = torch.where(above, torch.zeros_like(qc_interp), qc_interp)
    qi_new = torch.where(above, torch.zeros_like(qi_interp), qi_interp)
    qp_new = torch.where(above, torch.zeros_like(qp_interp), qp_interp)
    qn_new = qv_new + qc_new + qi_new

    p_new = torch.exp(ln_p_interp)
    nz_new = zn.shape[0]
    for k in range(1, nz_new):
        ab_k = above[k]
        if ab_k.any():
            dz_k = torch.clamp(zn[k] - zn[k - 1], min=1.0)
            t_mid = 0.5 * (t_new[k - 1] + t_new[k])
            p_hyd = p_new[k - 1] * torch.exp(-C.GRAVITY * dz_k / (C.R_D * t_mid))
            p_new[k] = torch.where(ab_k, torch.clamp(p_hyd, min=1.0), p_new[k])

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


def _load_restart_into_model(model: AOBAState, path: Path, device: torch.device) -> dict[str, Any]:
    """Restore prognostic tensors (including vertical remap when `nz` / `z_top` changes) from `restart.pt`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.sim_time_sec = float(ckpt["sim_time_sec"])
    model.step_count = int(ckpt["step_count"])

    ck_atmos = ckpt["atmos"]
    ck_crm = ckpt["crm"]
    nz_old = int(ck_atmos.rho_d.shape[0])
    nz_new = int(model.atmos.rho_d.shape[0])
    z_top_old_val = float(ck_atmos.z_w[-1].mean().item())
    z_top_new_val = float(model.atmos.z_w[-1].mean().item())
    needs_vremap = (nz_old != nz_new) or (abs(z_top_old_val - z_top_new_val) > 50.0)

    if not needs_vremap:
        for k, v in ck_atmos.__dict__.items():
            setattr(model.atmos, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in ck_crm.__dict__.items():
            setattr(model.crm, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
        model.rad_cache = {k: v.to(device=device) for k, v in ckpt["rad_cache"].items()}
    else:
        print(
            f"  [Vertical Remap] Seamlessly connecting restart ({nz_old} layers, top={z_top_old_val/1000:.1f} km) "
            f"-> active grid ({nz_new} layers, top={z_top_new_val/1000:.1f} km)...",
            flush=True,
        )
        z_old_c = ck_atmos.z_c.cpu()
        z_new_c = model.atmos.z_c.cpu()
        z_old_w = ck_atmos.z_w.cpu()
        z_new_w = model.atmos.z_w.cpu()

        # 1. Host thermodynamics (hydrostatic stratospheric extension above z_old_top)
        rho_n, thm_n, qn_n, qp_n, qv_n, qc_n, qi_n = _remap_thermo_columns(
            z_old_c, ck_atmos.rho_d, ck_atmos.theta_m, ck_atmos.q_v, ck_atmos.q_c, ck_atmos.q_i, ck_atmos.q_p, z_new_c
        )
        model.atmos.rho_d = rho_n.to(device=device)
        model.atmos.theta_m = thm_n.to(device=device)
        model.atmos.q_n = qn_n.to(device=device)
        model.atmos.q_p = qp_n.to(device=device)
        model.atmos.q_v = qv_n.to(device=device)
        model.atmos.q_c = qc_n.to(device=device)
        model.atmos.q_i = qi_n.to(device=device)

        # Host edge winds: interpolate below z_old_top, smooth exponential taper aloft
        c0 = model.mesh.cells_on_edge[:, 0].cpu()
        c1 = model.mesh.cells_on_edge[:, 1].cpu()
        ze_old = 0.5 * (z_old_c[:, c0] + z_old_c[:, c1])
        ze_new = 0.5 * (z_new_c[:, c0] + z_new_c[:, c1])
        ue_interp = _interp_vertical_batch(ze_old, ck_atmos.u_edge, ze_new)
        taper_e = torch.exp(-torch.clamp(ze_new - ze_old[-1:], min=0.0) / 12000.0)
        model.atmos.u_edge = (ue_interp * taper_e).to(device=device)

        w_interp = _interp_vertical_batch(z_old_w, ck_atmos.w, z_new_w)
        model.atmos.w = torch.where(z_new_w > z_old_w[-1:], torch.zeros_like(w_interp), w_interp).to(device=device)
        if getattr(ck_atmos, "q_dust", None) is not None and model.atmos.q_dust is not None:
            qd_i = torch.clamp(_interp_vertical_batch(z_old_c, ck_atmos.q_dust, z_new_c), min=0.0)
            model.atmos.q_dust = torch.where(z_new_c > z_old_c[-1:], torch.zeros_like(qd_i), qd_i).to(device=device)

        # 2. Embedded 1:1 CRM columns [nz, nc, ny, nx] -> flatten to [nz, nc*ny*nx]
        _, nc, ny, nx = ck_crm.rho_d.shape
        n_sub = ny * nx
        m_sub = nc * n_sub
        zo_sub = z_old_c.unsqueeze(-1).expand(nz_old, nc, n_sub).reshape(nz_old, m_sub)
        zn_sub = z_new_c.unsqueeze(-1).expand(nz_new, nc, n_sub).reshape(nz_new, m_sub)
        zow_sub = z_old_w.unsqueeze(-1).expand(nz_old + 1, nc, n_sub).reshape(nz_old + 1, m_sub)
        znw_sub = z_new_w.unsqueeze(-1).expand(nz_new + 1, nc, n_sub).reshape(nz_new + 1, m_sub)

        c_rho, c_thm, c_qn, c_qp, c_qv, c_qc, c_qi = _remap_thermo_columns(
            zo_sub,
            ck_crm.rho_d.reshape(nz_old, m_sub),
            ck_crm.theta_m.reshape(nz_old, m_sub),
            ck_crm.q_v.reshape(nz_old, m_sub),
            ck_crm.q_c.reshape(nz_old, m_sub),
            ck_crm.q_i.reshape(nz_old, m_sub),
            ck_crm.q_p.reshape(nz_old, m_sub),
            zn_sub,
        )
        taper_sub = torch.exp(-torch.clamp(zn_sub - zo_sub[-1:], min=0.0) / 12000.0)
        c_u = _interp_vertical_batch(zo_sub, ck_crm.u.reshape(nz_old, m_sub), zn_sub) * taper_sub
        c_v = _interp_vertical_batch(zo_sub, ck_crm.v.reshape(nz_old, m_sub), zn_sub) * taper_sub
        c_w_i = _interp_vertical_batch(zow_sub, ck_crm.w.reshape(nz_old + 1, m_sub), znw_sub)
        c_w = torch.where(znw_sub > zow_sub[-1:], torch.zeros_like(c_w_i), c_w_i)
        c_qke_i = _interp_vertical_batch(zo_sub, ck_crm.qke.reshape(nz_old, m_sub), zn_sub)
        c_qke = torch.where(zn_sub > zo_sub[-1:], torch.full_like(c_qke_i, 2.0e-4), torch.clamp(c_qke_i, min=2.0e-4))

        sh4 = (nz_new, nc, ny, nx)
        model.crm.rho_d = c_rho.reshape(sh4).to(device=device)
        model.crm.theta_m = c_thm.reshape(sh4).to(device=device)
        model.crm.q_n = c_qn.reshape(sh4).to(device=device)
        model.crm.q_p = c_qp.reshape(sh4).to(device=device)
        model.crm.q_v = c_qv.reshape(sh4).to(device=device)
        model.crm.q_c = c_qc.reshape(sh4).to(device=device)
        model.crm.q_i = c_qi.reshape(sh4).to(device=device)
        model.crm.u = c_u.reshape(sh4).to(device=device)
        model.crm.v = c_v.reshape(sh4).to(device=device)
        model.crm.w = c_w.reshape(nz_new + 1, nc, ny, nx).to(device=device)
        model.crm.qke = c_qke.reshape(sh4).to(device=device)
        model.crm.n_c = None
        model.crm.n_i = None

        # Force fresh radiation evaluation on the first step with the new vertical grid
        model.rad_cache = {}
        stride = max(int(model.cfg.rad_stride), 1)
        model.step_count = (model.step_count // stride) * stride
        ckpt["_vertical_remapped"] = True

    # Honor active run_config grid spacing (`crm_dx`) when switching spacing on resume
    model.crm.dx = float(model.cfg.crm_dx)
    model.crm.dy = float(model.cfg.crm_dx)
    _restore_surface(model.sfc, ckpt["sfc"], device)
    if ckpt.get("sfc_sub") is not None and model.sfc_sub is not None:
        _restore_surface(model.sfc_sub, ckpt["sfc_sub"], device)
    for k, v in ckpt["ocn"].items():
        setattr(model.ocn, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    if model.ogcm is not None:
        if ckpt.get("ogcm") is not None and int(ckpt["ogcm"].get("nz_ocn", -1)) == model.ogcm.nz_ocn:
            for k, v in ckpt["ogcm"].items():
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
        else:
            from aoba.ocean_ogcm import init_ogcm_state

            model.ogcm = init_ogcm_state(model.sfc, model.mesh, nz_ocn=model.cfg.ocean_nz)
            if bool(model.cfg.enable_ocean_bgc):
                from aoba.ocean_bgc import init_ocean_bgc_state

                model.ogcm.bgc = init_ocean_bgc_state(model.ogcm.wet_c, model.ogcm.h_c, model.ogcm.z_c, model.mesh.area_cell)
            if bool(model.cfg.enable_seaice_evp_itd):
                from aoba.seaice_evp import init_seaice_evp_state

                model.ogcm.seaice_evp = init_seaice_evp_state(model.sfc, model.mesh)
    if ckpt.get("river") is not None and model.river is not None:
        for k, v in ckpt["river"].items():
            setattr(model.river, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    if ckpt.get("land_carbon") is not None and model.land_carbon is not None:
        for k, v in ckpt["land_carbon"].items():
            setattr(model.land_carbon, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
    if ckpt.get("aerosols") is not None and model.aerosols is not None:
        for k, v in ckpt["aerosols"].items():
            if isinstance(v, torch.Tensor) and needs_vremap and v.ndim == 2 and v.shape[0] == nz_old:
                remapped = torch.clamp(_interp_vertical_batch(z_old_c, v.cpu(), z_new_c), min=0.0)
                remapped = torch.where(z_new_c > z_old_c[-1:], torch.zeros_like(remapped), remapped)
                setattr(model.aerosols, k, remapped.to(device=device))
            else:
                setattr(model.aerosols, k, v.to(device=device) if isinstance(v, torch.Tensor) else v)
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
        if model.ogcm is not None:
            for name, unit in units_2d_ocn.items():
                if name in sums:
                    arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
                    data_vars[name] = (("time", "n_cell"), arr, {"units": unit})
        if model.icesheet is not None:
            for name in ("h_ice", "z_topo_dyn"):
                if name in sums:
                    arr = (sums[name] * inv).detach().cpu().numpy()[None, :]
                    data_vars[name] = (("time", "n_cell"), arr, {"units": "m"})
        data_vars["koppen_code"] = (("time", "n_cell"), kg30.detach().cpu().numpy()[None, :])
        data_vars["koppen_major"] = (("time", "n_cell"), kg5.detach().cpu().numpy()[None, :])
    if include_3d:
        for name, unit in units_3d.items():
            if name in sums:
                arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :]
                data_vars[name] = (("time", "nz", "n_cell"), arr, {"units": unit})
        if model.ogcm is not None:
            for name, unit in units_3d_ocn.items():
                if name in sums:
                    arr = (sums[name] * inv).detach().cpu().numpy()[None, :, :]
                    data_vars[name] = (("time", "nz_ocn", "n_cell"), arr, {"units": unit})
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

    date_str = model.calendar.date_at(model.sim_time_sec).iso_datetime[:16]
    coords_np = getattr(mesh, "_live_preview_coords", None)
    if coords_np is None:
        coords_np = (
            np.rad2deg(mesh.lat_cell.detach().cpu().numpy()).astype(np.float32),
            np.rad2deg(mesh.lon_cell.detach().cpu().numpy()).astype(np.float32),
            mesh.surf_type.detach().cpu().numpy().astype(np.int8),
        )
        mesh._live_preview_coords = coords_np
    lat_deg_np, lon_deg_np, surf_type_np = coords_np
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
    )
    os.replace(tmp_path, out_path)


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
    if not args.no_resume:
        for y in range(years, 0, -1):
            y_dir = out_root / f"year_{y:04d}"
            if (y_dir / "_COMPLETE").exists() and (y_dir / "restart.pt").exists():
                print(f"Resuming from completed Year {y:04d} ({y_dir / 'restart.pt'})...", flush=True)
                ckpt_yr = _load_restart_into_model(model, y_dir / "restart.pt", dev)
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
    ]
    a_earth_m2 = float(model.mesh.area_cell.detach().cpu().to(torch.float64).sum().item())
    if not spinup_csv.exists() or (start_year == 1 and resume_step_in_yr == 0 and not resumed_existing):
        with open(spinup_csv, "w", newline="") as f_sp:
            csv.writer(f_sp).writerow(_SPINUP_HEADER)
            if model.ogcm is not None:
                m0 = diagnose_ocean_spinup_metrics(model.ogcm, model.sfc, model.mesh)
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
                ])
    elif prev_ohc_j is None and model.ogcm is not None:
        m_res = diagnose_ocean_spinup_metrics(model.ogcm, model.sfc, model.mesh)
        prev_ohc_j = m_res.get("ohc_j", None)
        prev_ohc_time_sec = model.sim_time_sec

    if not resumed_existing and args.init_from_restart:
        init_pt = Path(args.init_from_restart).expanduser()
        if not init_pt.exists():
            raise FileNotFoundError(f"--init-from-restart file not found: {init_pt}")
        print(f"Warm-starting from external restart checkpoint: {init_pt}...", flush=True)
        _load_restart_into_model(model, init_pt, dev)
        elapsed_yr = int(math.floor(model.sim_time_sec / (C.SECONDS_PER_DAY * days_per_year) + 1.0e-6))
        start_year = max(1, min(years, elapsed_yr + 1))
        print(
            f"  Continuing at date {model.calendar.date_at(model.sim_time_sec).iso_datetime} "
            f"(start_year={start_year:04d}/{years:04d})",
            flush=True,
        )

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
                ss = model.sfc_sub
                n_sub = model.cfg.crm_ny * model.cfg.crm_nx
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
                            n_sub_c = model.cfg.crm_ny * model.cfg.crm_nx
                            nc_c = model.mesh.n_cells
                            sample["tau_x_sfc"] = sd_sub.tau_x.reshape(nc_c, n_sub_c).mean(dim=-1)
                            sample["tau_y_sfc"] = sd_sub.tau_y.reshape(nc_c, n_sub_c).mean(dim=-1)
                            sample["runoff_total"] = sd_sub.runoff_total.reshape(nc_c, n_sub_c).mean(dim=-1)
                            sample["runoff_glacier"] = sd_sub.runoff_ice.reshape(nc_c, n_sub_c).mean(dim=-1)
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
                    if model.ogcm is not None:
                        og = model.ogcm
                        uo_c, vo_c = reconstruct_cell_winds(model.mesh, og.u_edge * og.wet_e.to(og.u_edge.dtype))
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
                ocn_m = diagnose_ocean_spinup_metrics(model.ogcm, model.sfc, model.mesh) if model.ogcm is not None else {}
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

                    bgc_m = diagnose_ocean_bgc_metrics(model.ogcm.bgc, model.ogcm, model.mesh)
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
                if model.river is not None:
                    q_riv_sv = float((model.river.mouth_flux_kgm2s.detach().cpu().double() * model.mesh.area_cell.detach().cpu().double()).sum().item()) * 1.0e-9
                    esm_parts.append(f"River={q_riv_sv:.3f} Sv")
                if model.land_carbon is not None:
                    from aoba.land_carbon import diagnose_land_carbon_metrics

                    lc_m = diagnose_land_carbon_metrics(model.land_carbon, model.mesh.area_cell)
                    esm_parts.append(
                        f"LandC={lc_m['c_land_total_pgc']:.0f} PgC "
                        f"(Veg={lc_m['c_veg_pgc']:.0f}, Soil={lc_m['c_soil_pgc']:.0f}, "
                        f"Fire={lc_m['land_fire_pgc_yr']:.2f} PgC/yr)"
                    )
                if model.ogcm is not None and model.ogcm.seaice_evp is not None:
                    evp_s = model.ogcm.seaice_evp
                    v_ice_kkm3 = float((evp_s.v_cat.sum(dim=0).detach().cpu().double() * model.mesh.area_cell.detach().cpu().double()).sum().item()) * 1.0e-12
                    u_ice_max = float(evp_s.u_ice_e.abs().max().item())
                    esm_parts.append(f"EVP_Ice={v_ice_kkm3:.1f}e3 km3 (max|u|={u_ice_max:.2f} m/s)")
                if model.aerosols is not None:
                    a_w = model.mesh.area_cell.detach().cpu().double()
                    a_tot_m2 = float(a_w.sum().item())
                    aod_nat = float(((model.aerosols.aod_ss + model.aerosols.aod_so4 + model.aerosols.aod_bc + model.aerosols.aod_oc).detach().cpu().double() * a_w).sum().item()) / a_tot_m2
                    ccn_mean = float((model.aerosols.ccn_aero_m3.detach().cpu().double() * a_w).sum().item()) * 1.0e-6 / a_tot_m2
                    if model.atmos.q_dust is not None:
                        from aoba.dust import DUST_K_EXT_SW1
                        col_dust = (model.atmos.rho_d * model.atmos.dz_c * torch.clamp(model.atmos.q_dust, min=0.0)).sum(dim=0).detach().cpu().double()
                        aod_dust = float((DUST_K_EXT_SW1 * col_dust * a_w).sum().item()) / a_tot_m2
                        esm_parts.append(f"Aero(AOD_nat={aod_nat:.4f}, AOD_dust={aod_dust:.4f}, CCN={ccn_mean:.1f} cm^-3)")
                    else:
                        esm_parts.append(f"Aero(AOD={aod_nat:.4f}, CCN={ccn_mean:.1f} cm^-3)")
                if esm_parts:
                    print(f"[ESM Y{year:03d} M{mon_num:02d} | {diag.date_str}] " + " | ".join(esm_parts), flush=True)
                for k in mon_accum:
                    mon_accum[k] = 0.0
                mon_cnt = 0
                if cmip7_acc is not None and cmip7_acc.sample_count > 0:
                    append_zarr(cmip7_acc.to_xarray(model), cmip7_zarr)
                    cmip7_acc.reset()
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
