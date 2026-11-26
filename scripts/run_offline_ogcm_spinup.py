#!/usr/bin/env python3
"""Phase 2 Sandwich Spin-Up: Self-Forced Offline 3D OGCM + Sea-Ice Deep-Ocean Equilibration.

Drives AOBA's exact 3D SCVT Voronoi OGCM (`step_ogcm_voronoi`) and fractional-lead ocean/sea-ice
surface energy balance (`_ocean_seaice_post_core` + `compute_surface_transfer`) using ONLY
the periodic 12-month seasonal surface forcing climatology harvested from AOBA's own Phase 1
coupled integration (`year_XXXX/monthly.zarr`).

No external geographic maps, no prescribed SST restoring, and no ocean Q-fluxes are used.
Because the atmospheric CRM, radiation, and land biosphere are bypassed and `dt_ocn` can be
increased (e.g. 3600 s), deep-ocean spin-up runs ~100x faster than full coupled mode
(~500-700 sim-years/day on A100 or ~150-250 sim-years/day on Apple Silicon MPS).

Upon completion (or reaching `--stop-deep-drift`), writes an equilibrated Phase-3 coupled
restart checkpoint (`restart_phase3_init.pt`) ready for `scripts/run_g5_30yr.py --init-from-restart`.
"""

from __future__ import annotations

import argparse
import csv
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
from aoba.coupled_model import AOBAConfig, compute_orbital_insolation, init_aoba_model
from aoba.ocean_ogcm import (
    diagnose_ocean_spinup_metrics,
    step_ogcm_voronoi,
)
from aoba.runtime import apply_runtime_tuning, get_device, maybe_cuda_compile, synchronize_device
from aoba.surface import (
    ALB_SEAICE_NIR,
    ALB_SEAICE_VIS,
    ALB_SNOW_VIS_NEW,
    ALB_SNOW_NIR_NEW,
    MC_SEAICE_CRIT,
    SURF_OCEAN,
    _ocean_seaice_post_core,
    _surface_ocean_pre_transfer_core,
    compute_ocean_albedo_briegleb,
)
from aoba.turbulence import compute_surface_transfer


def _load_self_forcing_climatology(
    coupled_dir: Path,
    year_start: int,
    year_end: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Average `year_XXXX/monthly.zarr` over `[year_start..year_end]` into a `[12, n_cell]` climatology."""
    sums: dict[str, np.ndarray] = {}
    n_yrs = 0
    req_2d = (
        "tau_x_sfc", "tau_y_sfc", "runoff_total", "runoff_glacier",
        "precip_mm_day", "precip_snow_mm_day", "sw1_dir_sfc", "sw1_dif_sfc",
        "sw2_dir_sfc", "sw2_dif_sfc", "lw_dn_sfc",
    )
    req_3d_bot = ("temp", "pres", "q_v", "u_zonal", "v_merid")

    for y in range(year_start, year_end + 1):
        zpath = coupled_dir / f"year_{y:04d}" / "monthly.zarr"
        if not zpath.exists():
            continue
        ds = xr.open_zarr(zpath, consolidated=False)
        if ds.sizes.get("time", 0) < 12:
            ds.close()
            continue
        for k in req_2d:
            if k not in ds:
                raise KeyError(f"Missing self-forcing variable '{k}' in {zpath}. Re-run Phase 1 years with updated run_g5_30yr.py.")
            arr = ds[k].isel(time=slice(0, 12)).values.astype(np.float64)
            sums[k] = arr if n_yrs == 0 else (sums[k] + arr)
        for k in req_3d_bot:
            arr = ds[k].isel(time=slice(0, 12), nz=0).values.astype(np.float64)
            sums[f"{k}_0"] = arr if n_yrs == 0 else (sums[f"{k}_0"] + arr)
        ds.close()
        n_yrs += 1

    if n_yrs == 0:
        raise FileNotFoundError(f"No complete 12-month monthly.zarr found in {coupled_dir} for years {year_start}..{year_end}")

    print(f"Loaded AOBA Phase-1 self-forcing climatology from {n_yrs} years ({year_start}..{year_end}).", flush=True)
    inv = 1.0 / float(n_yrs)
    clim = {k: torch.tensor(v * inv, device=device, dtype=torch.float32) for k, v in sums.items()}
    # Derived rain & snow fluxes [kg m^-2 s^-1]
    p_tot = torch.clamp(clim["precip_mm_day"] / C.SECONDS_PER_DAY, min=0.0)
    p_sn = torch.clamp(clim["precip_snow_mm_day"] / C.SECONDS_PER_DAY, min=0.0).minimum(p_tot)
    clim["precip_rain"] = torch.clamp(p_tot - p_sn, min=0.0)
    clim["precip_snow"] = p_sn
    return clim


def _interp_periodic_month(clim: dict[str, torch.Tensor], frac_yr: float) -> dict[str, torch.Tensor]:
    """Smoothly linearly interpolate a periodic 12-month `[12, nc]` climatology at `frac_yr in [0, 1)`."""
    pos = (frac_yr % 1.0) * 12.0 - 0.5
    m0 = int(math.floor(pos)) % 12
    m1 = (m0 + 1) % 12
    w1 = pos - math.floor(pos)
    w0 = 1.0 - w1
    return {k: (w0 * v[m0] + w1 * v[m1]) for k, v in clim.items()}


def _step_offline_ocean_surface(
    sfc,
    mesh,
    frc: dict[str, torch.Tensor],
    mu0: torch.Tensor,
    z_ref_rel: torch.Tensor,
    dt_ocn: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run AOBA's exact ocean + fractional-lead sea-ice energy balance using self-forcing climatology."""
    dev = sfc.t_ocean_ml.device
    is_ocean = sfc.surf_type == SURF_OCEAN

    # Ocean + sea-ice surface albedos (exact equations from `compute_surface_albedos`)
    f_ice = torch.where(
        is_ocean,
        torch.clamp(torch.sqrt(torch.clamp(sfc.m_seaice, min=0.0) / MC_SEAICE_CRIT), 0.0, 1.0),
        torch.zeros_like(mu0),
    )
    ocn_dir, ocn_dif = compute_ocean_albedo_briegleb(mu0)
    f_sn_ice = torch.tanh(torch.clamp(sfc.swe_seaice, min=0.0) / 15.0)
    ice_vis = (1.0 - f_sn_ice) * ALB_SEAICE_VIS + f_sn_ice * ALB_SNOW_VIS_NEW
    ice_nir = (1.0 - f_sn_ice) * ALB_SEAICE_NIR + f_sn_ice * ALB_SNOW_NIR_NEW
    alb_vd = (1.0 - f_ice) * ocn_dir + f_ice * ice_vis
    alb_vf = (1.0 - f_ice) * ocn_dif + f_ice * ice_vis
    alb_nd = (1.0 - f_ice) * ocn_dir + f_ice * ice_nir
    alb_nf = (1.0 - f_ice) * ocn_dif + f_ice * ice_nir

    sw1_dir = torch.clamp(frc["sw1_dir_sfc"], min=0.0)
    sw1_dif = torch.clamp(frc["sw1_dif_sfc"], min=0.0)
    sw2_dir = torch.clamp(frc["sw2_dir_sfc"], min=0.0)
    sw2_dif = torch.clamp(frc["sw2_dif_sfc"], min=0.0)
    lw_dn = torch.clamp(frc["lw_dn_sfc"], min=50.0)

    sw_abs = (
        sw1_dir * (1.0 - alb_vd)
        + sw1_dif * (1.0 - alb_vf)
        + sw2_dir * (1.0 - alb_nd)
        + sw2_dif * (1.0 - alb_nf)
    )
    sw_abs_ocn = (1.0 - f_ice) * sw_abs

    # Lowest-level atmospheric state from AOBA Phase 1 climatology
    temp_1 = torch.clamp(frc["temp_0"], min=180.0)
    pres_1 = torch.clamp(frc["pres_0"], min=40000.0)
    qv_1 = torch.clamp(frc["q_v_0"], min=1.0e-7)
    u_1 = frc["u_zonal_0"]
    v_1 = frc["v_merid_0"]
    rho_sfc = pres_1 / (C.R_D * temp_1 * (1.0 + C.RV_OVER_RD * qv_1))
    pres_sfc = pres_1 + C.GRAVITY * rho_sfc * (1.0 + qv_1) * z_ref_rel
    exner_1 = torch.pow(pres_1 / C.P_REF, C.KAPPA_D)
    exner_s = torch.pow(pres_sfc / C.P_REF, C.KAPPA_D)
    theta_1 = temp_1 / exner_1
    theta_v1 = theta_1 * (1.0 + C.ZVIR * qv_1)
    t_air_sfc = theta_1 * exner_s
    cp_air = C.CP_D * (1.0 + 0.84 * qv_1)

    ocn_pre_fn = maybe_cuda_compile(_surface_ocean_pre_transfer_core, device=dev)
    z0m_sea, z0h_sea, z0q_sea, qsat_ow, qsat_si, theta_vsfc_o = ocn_pre_fn(
        sfc.u_star_ocean, f_ice, sfc.t_ocean_ml, sfc.t_seaice, pres_sfc, exner_s,
    )
    tr_ocn = compute_surface_transfer(
        u_1, v_1, theta_v1, theta_vsfc_o, z_ref_rel,
        z0m_sea, z0h_sea, z0q_sea, rsl_depth=None,
    )
    ga_h_o = tr_ocn.c_h * tr_ocn.wind_eff
    ga_q_o = tr_ocn.c_q * tr_ocn.wind_eff

    comp0 = sfc.t_ocean_ml_comp if sfc.t_ocean_ml_comp is not None else torch.zeros_like(sfc.t_ocean_ml)
    ocn_post_fn = maybe_cuda_compile(_ocean_seaice_post_core, device=dev)
    t_ml_next, comp_final, m_si_next, t_si_new, swe_si_next, shf_o, evap_o, lhf_o, lw_emit_o = ocn_post_fn(
        sfc.t_ocean_ml, comp0, sfc.h_ocean_ml, sfc.m_seaice, sfc.t_seaice, sfc.swe_seaice,
        f_ice, qsat_ow, qsat_si, ga_h_o, ga_q_o, rho_sfc, cp_air, t_air_sfc, qv_1,
        sw_abs, sw1_dir, sw1_dif, sw2_dir, sw2_dif, lw_dn,
        frc["precip_snow"], mu0, dt_ocn,
    )

    sfc.t_ocean_ml = torch.where(is_ocean, t_ml_next, sfc.t_ocean_ml)
    sfc.t_ocean_ml_comp = torch.where(is_ocean, comp_final, comp0)
    sfc.u_star_ocean = torch.where(is_ocean, tr_ocn.u_star, sfc.u_star_ocean)
    sfc.m_seaice = torch.where(is_ocean, m_si_next, torch.zeros_like(m_si_next))
    sfc.t_seaice = torch.where(is_ocean, t_si_new, sfc.t_seaice)
    sfc.swe_seaice = torch.where(is_ocean, swe_si_next, torch.zeros_like(swe_si_next))

    # Net surface heat flux into ocean+ice [W m^-2] for monitoring
    q_net_o = sw_abs + (lw_dn - lw_emit_o) - shf_o - lhf_o
    return f_ice, evap_o, sw_abs_ocn, q_net_o, tr_ocn.u_star


def _serialize_ogcm(ogcm_state) -> dict[str, Any]:
    """Serialize 3D OGCM state (including nested BGC & EVP-ITD dataclasses) matching `run_g5_30yr.py`."""
    return {
        k: (
            {bk: bv.detach().cpu() if isinstance(bv, torch.Tensor) else bv for bk, bv in v.__dict__.items()}
            if k in ("bgc", "seaice_evp") and v is not None
            else (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        )
        for k, v in ogcm_state.__dict__.items()
        if not k.startswith("_")
    }


def _restore_ogcm(ogcm_target, ck_og: dict[str, Any], dev: torch.device) -> None:
    """Restore 3D OGCM state (including nested BGC & EVP-ITD dicts) onto `dev`."""
    for k, v in ck_og.items():
        if k.startswith("_"):
            continue
        if k in ("bgc", "seaice_evp"):
            sub_target = getattr(ogcm_target, k, None)
            if v is not None and sub_target is not None:
                sub_items = v.items() if isinstance(v, dict) else v.__dict__.items()
                for bk, bv in sub_items:
                    setattr(sub_target, bk, bv.to(device=dev) if isinstance(bv, torch.Tensor) else bv)
            continue
        setattr(ogcm_target, k, v.to(device=dev) if isinstance(v, torch.Tensor) else v)


def inject_offline_ocean_into_restart(
    base_restart_path: Path,
    ogcm_state,
    ocn_sfc_state,
    out_restart_path: Path,
    ocn_coupler=None,
) -> None:
    """Inject equilibrated 3D OGCM (+ BGC + EVP-ITD) & sea-ice state cleanly into a coupled `restart.pt`."""
    ckpt = torch.load(base_restart_path, map_location="cpu", weights_only=False)
    ckpt["ogcm"] = _serialize_ogcm(ogcm_state)

    ocn_keys = ("t_ocean_ml", "t_ocean_ml_comp", "u_star_ocean", "m_seaice", "t_seaice", "swe_seaice")
    if ocn_coupler is not None and ckpt.get("ocn_sfc") is not None:
        # Independent fine-ocean grid: update `ocn_sfc` directly and project conservatively onto host `sfc`
        for k in ocn_keys:
            val = getattr(ocn_sfc_state, k, None)
            if val is not None:
                ckpt["ocn_sfc"][k] = val.detach().cpu()
        from aoba.ocean_coupler import ocn_to_atm_mean

        is_ocn_host = (ckpt["sfc"]["surf_type"].detach().cpu() == SURF_OCEAN)
        for k in ocn_keys:
            val = getattr(ocn_sfc_state, k, None)
            if val is not None and k in ckpt["sfc"]:
                proj = ocn_to_atm_mean(ocn_coupler, val).detach().cpu()
                ckpt["sfc"][k] = torch.where(is_ocn_host, proj, ckpt["sfc"][k])
        host_src = ckpt["sfc"]
    else:
        is_ocn_host = (ocn_sfc_state.surf_type.detach().cpu() == SURF_OCEAN)
        for k in ocn_keys:
            val = getattr(ocn_sfc_state, k, None)
            if val is not None:
                ckpt["sfc"][k] = val.detach().cpu()
        host_src = ckpt["sfc"]

    # Synchronize subgrid ocean tiles (`sfc_sub`) identically to host cell ocean state
    if ckpt.get("sfc_sub") is not None:
        nc = int(is_ocn_host.shape[0])
        m_sub = int(ckpt["sfc_sub"]["t_ocean_ml"].shape[0])
        n_sub = m_sub // nc
        is_ocn_sub = is_ocn_host.unsqueeze(1).expand(nc, n_sub).reshape(m_sub)
        for k in ocn_keys:
            val = host_src.get(k)
            if val is not None and k in ckpt["sfc_sub"]:
                bcast = val.detach().cpu().unsqueeze(1).expand(nc, n_sub).reshape(m_sub)
                ckpt["sfc_sub"][k] = torch.where(is_ocn_sub, bcast, ckpt["sfc_sub"][k])

    # Clear cached OHC reference so Phase 3 starts a fresh dOHC/dt baseline
    ckpt.pop("prev_ohc_j", None)
    ckpt.pop("prev_ohc_time_sec", None)
    tmp = out_restart_path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, out_restart_path)
    print(f"Injected equilibrated 3D OGCM state into Phase-3 coupled restart: {out_restart_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 Self-Forced Offline 3D OGCM Deep-Ocean Spin-Up")
    parser.add_argument("--coupled-dir", type=str, required=True, help="Phase-1 coupled output dir (contains year_XXXX/)")
    parser.add_argument("--base-restart", type=str, default="", help="Phase-1 restart.pt (default: latest year_XXXX/restart.pt)")
    parser.add_argument("--forcing-year-start", type=int, default=21)
    parser.add_argument("--forcing-year-end", type=int, default=30)
    parser.add_argument("--years", type=int, default=1000, help="Maximum offline ocean surface years")
    parser.add_argument("--dt-ocn", type=float, default=3600.0, help="Offline ocean timestep [s] (default 3600 s)")
    parser.add_argument("--deep-accel", type=float, default=1.0, help="Bryan (1984) deep-ocean (>1500m) tracer acceleration factor (e.g. 6.0)")
    parser.add_argument("--bgc-stride", type=int, default=1, help="Call FlexPFT BGC every N ocean steps (e.g. 4 for 4h BGC cadence)")
    parser.add_argument("--sync-tail-years", type=int, default=20, help="Final synchronous (1x) polish years after deep-accel convergence")
    parser.add_argument("--stop-deep-drift", type=float, default=0.01, help="Auto-stop when deep (>1500m) drift < value [K/deep-century]")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for Phase-2 offline spin-up")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    deep_accel_req = max(float(args.deep_accel), 1.0)
    bgc_stride = max(int(args.bgc_stride), 1)
    sync_tail_yr = max(int(args.sync_tail_years), 0) if deep_accel_req > 1.0001 else 0

    coupled_dir = Path(args.coupled_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.base_restart:
        base_restart = Path(args.base_restart).expanduser().resolve()
    else:
        cands = sorted(coupled_dir.glob("year_*/restart.pt"))
        if not cands:
            raise FileNotFoundError(f"No year_*/restart.pt found in {coupled_dir}")
        base_restart = cands[-1]

    dev = torch.device(args.device) if args.device else get_device()
    hw = apply_runtime_tuning(dev)
    print(f"=== AOBA Phase-2 Self-Forced Offline 3D OGCM Spin-Up on {dev} [{hw.chip_name}] ===", flush=True)
    print(f"  Base restart : {base_restart}", flush=True)
    print(
        f"  Forcing years: {args.forcing_year_start}..{args.forcing_year_end} | dt_ocn={args.dt_ocn:.0f} s | "
        f"max={args.years} yr | deep_accel={deep_accel_req:.1f}x | bgc_stride={bgc_stride} | sync_tail={sync_tail_yr} yr",
        flush=True,
    )

    ckpt = torch.load(base_restart, map_location="cpu", weights_only=False)
    cfg: AOBAConfig = ckpt["cfg"]
    model = init_aoba_model(cfg, device=dev)
    assert model.ogcm is not None

    ocn_mesh = model.ocn_mesh if model.ocn_mesh is not None else model.mesh
    ocn_sfc = model.ocn_sfc if model.ocn_sfc is not None else model.sfc
    ocn_coupler = model.ocn_coupler
    has_fine_ocn = ocn_coupler is not None
    ocn_keys = ("t_ocean_ml", "t_ocean_ml_comp", "u_star_ocean", "m_seaice", "t_seaice", "swe_seaice")

    # Restore ocean & surface state from Phase-1 restart (or resume Phase-2 checkpoint if present)
    off_ckpt_path = out_dir / "offline_ogcm_checkpoint.pt"
    start_year = 1
    if off_ckpt_path.exists():
        print(f"Resuming Phase-2 offline ocean spin-up from {off_ckpt_path}...", flush=True)
        ock = torch.load(off_ckpt_path, map_location="cpu", weights_only=False)
        start_year = int(ock["year"]) + 1
        _restore_ogcm(model.ogcm, ock["ogcm"], dev)
        for k, v in ock["sfc"].items():
            if v is not None:
                setattr(ocn_sfc, k, v.to(device=dev) if isinstance(v, torch.Tensor) else v)
    else:
        _restore_ogcm(model.ogcm, ckpt["ogcm"], dev)
        sfc_src = ckpt["ocn_sfc"] if (has_fine_ocn and ckpt.get("ocn_sfc") is not None) else ckpt["sfc"]
        for k in ocn_keys:
            if k in sfc_src and sfc_src[k] is not None:
                setattr(ocn_sfc, k, sfc_src[k].to(device=dev))

    clim_atm = _load_self_forcing_climatology(coupled_dir, args.forcing_year_start, args.forcing_year_end, dev)
    if has_fine_ocn:
        assert ocn_coupler is not None
        from aoba.ocean_coupler import atm_to_ocn_conservative_flux, atm_to_ocn_intensive

        print(
            f"Remapping 12-month self-forcing climatology from host (nc={model.mesh.n_cells}) "
            f"-> fine ocean (nc={ocn_mesh.n_cells})...",
            flush=True,
        )
        clim = {}
        for k, v in clim_atm.items():
            if k in ("runoff_total", "runoff_glacier"):
                clim[k] = torch.stack(
                    [atm_to_ocn_conservative_flux(ocn_coupler, v[m], ocean_only_source=False) for m in range(12)],
                    dim=0,
                )
            elif k in ("precip_rain", "precip_snow"):
                clim[k] = torch.stack(
                    [atm_to_ocn_intensive(ocn_coupler, v[m], ocean_only_source=False) for m in range(12)],
                    dim=0,
                )
            else:
                clim[k] = torch.stack(
                    [atm_to_ocn_intensive(ocn_coupler, v[m], ocean_only_source=True) for m in range(12)],
                    dim=0,
                )
        z_ref_rel = atm_to_ocn_intensive(
            ocn_coupler,
            torch.clamp(model.atmos.z_c[0] - model.atmos.z_w[0], min=10.0),
            ocean_only_source=False,
        )
    else:
        clim = clim_atm
        z_ref_rel = torch.clamp(model.atmos.z_c[0] - model.atmos.z_w[0], min=10.0)

    a_earth_m2 = float(ocn_mesh.area_cell.detach().cpu().to(torch.float64).sum().item())
    yr_sec = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    steps_per_year = max(int(round(yr_sec / args.dt_ocn)), 12)
    dt_step = yr_sec / float(steps_per_year)

    has_bgc = model.ogcm.bgc is not None
    if has_bgc:
        from aoba.ocean_bgc import diagnose_ocean_bgc_metrics

    csv_path = out_dir / "offline_spinup_monitor.csv"
    header = [
        "off_year", "t_ocn_global_k", "t_ocn_0_300m_k", "t_ocn_300_1500m_k", "t_ocn_deep_k",
        "sst_global_k", "s_ocn_global_psu", "sss_global_psu", "ohc_zj", "ohc_flux_wm2",
        "ocn_ke_jm2", "ocn_eta_rms_m", "ocn_hbl_mean_m", "ocn_hbl_max_m",
        "amoc_26n_sv", "amoc_max_sv", "acc_drake_sv", "seaice_nh_mkm2", "seaice_sh_mkm2",
        "deep_drift_k_per_century", "wall_s_per_yr",
    ]
    if has_bgc:
        header += ["ocean_npp_pgc_yr", "ocean_fgco2_pgc_yr", "ocean_o2_global_mmol_m3", "ocean_omz_vol_pct"]

    if start_year == 1 or not csv_path.exists():
        m0 = diagnose_ocean_spinup_metrics(model.ogcm, ocn_sfc, ocn_mesh)
        row0 = [
            0, f"{m0['t_ocn_global_k']:.5f}", f"{m0['t_ocn_0_300m_k']:.5f}",
            f"{m0['t_ocn_300_1500m_k']:.5f}", f"{m0['t_ocn_deep_k']:.5f}",
            f"{m0['sst_global_k']:.4f}", f"{m0['s_ocn_global_psu']:.5f}", f"{m0['sss_global_psu']:.4f}",
            f"{m0['ohc_zj']:.4f}", "0.000", f"{m0['ocn_ke_jm2']:.2f}", f"{m0['ocn_eta_rms_m']:.4f}",
            f"{m0['ocn_hbl_mean_m']:.2f}", f"{m0['ocn_hbl_max_m']:.1f}",
            f"{m0['amoc_26n_sv']:.2f}", f"{m0['amoc_max_sv']:.2f}", f"{m0['acc_drake_sv']:.2f}",
            f"{m0['seaice_nh_mkm2']:.3f}", f"{m0['seaice_sh_mkm2']:.3f}", "nan", "0.00",
        ]
        if has_bgc:
            mb0 = diagnose_ocean_bgc_metrics(model.ogcm.bgc, model.ogcm, ocn_mesh)
            row0 += [
                f"{mb0.get('ocean_npp_pgc_yr', 0.0):.2f}",
                f"{mb0.get('ocean_fgco2_pgc_yr', 0.0):.3f}",
                f"{mb0.get('ocean_o2_global_mmol_m3', 0.0):.2f}",
                f"{mb0.get('ocean_omz_vol_pct', 0.0):.3f}",
            ]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(row0)
        prev_ohc_j = m0["ohc_j"]
        deep_hist = [m0["t_ocn_deep_k"]]
    else:
        m_cur = diagnose_ocean_spinup_metrics(model.ogcm, ocn_sfc, ocn_mesh)
        prev_ohc_j = m_cur["ohc_j"]
        deep_hist = [m_cur["t_ocn_deep_k"]]

    in_sync_tail = False
    sync_tail_rem = sync_tail_yr
    deep_eq_yr = 0.0
    drift_win = max(10, int(round(50.0 / deep_accel_req)))

    for yr in range(start_year, args.years + sync_tail_yr + 1):
        active_accel = 1.0 if in_sync_tail else deep_accel_req
        t0 = time.perf_counter()
        for step_i in range(steps_per_year):
            frac_yr = (step_i + 0.5) / float(steps_per_year)
            t_sec_yr = frac_yr * yr_sec
            frc = _interp_periodic_month(clim, frac_yr)
            mu0, _, _ = compute_orbital_insolation(
                t_sec_yr, ocn_mesh.lat_cell, ocn_mesh.lon_cell,
                cfg.solar_constant, cfg.obliquity_deg, cfg.eccentricity, cfg.perihelion_deg,
                t_diurnal_sec=t_sec_yr, diurnal_cycle=False,
            )
            f_ice, evap_o, sw_abs_ocn, _, _ = _step_offline_ocean_surface(
                ocn_sfc, ocn_mesh, frc, mu0, z_ref_rel, dt_step
            )
            u10_o = torch.sqrt(frc["u_zonal_0"] ** 2 + frc["v_merid_0"] ** 2 + 1.0e-6)
            do_bgc_now = ((step_i + 1) % bgc_stride == 0) or (step_i + 1 == steps_per_year)
            bgc_scale_now = float(bgc_stride) if ((step_i + 1) % bgc_stride == 0) else float((step_i + 1) % bgc_stride)
            step_ogcm_voronoi(
                ogcm=model.ogcm,
                sfc=ocn_sfc,
                mesh=ocn_mesh,
                tau_x_sfc=frc["tau_x_sfc"],
                tau_y_sfc=frc["tau_y_sfc"],
                f_seaice=f_ice,
                evap_cell=evap_o,
                precip_cell=frc["precip_rain"] + frc["precip_snow"],
                runoff_total=frc["runoff_total"],
                runoff_glacier=frc["runoff_glacier"],
                dt=dt_step,
                enable_seaice_drift=bool(cfg.enable_seaice_drift),
                sw_net_ocn=sw_abs_ocn,
                u10_ocn=u10_o,
                co2_ppmv=float(cfg.co2_ppmv),
                enable_fct_zstar=bool(getattr(cfg, "enable_ogcm_fct_zstar", True)),
                enable_scale_aware_gm=bool(getattr(cfg, "enable_ogcm_scale_aware_gm", True)),
                enable_submeso=bool(getattr(cfg, "enable_ogcm_submeso", True)),
                enable_tidal_mixing=bool(getattr(cfg, "enable_ogcm_tidal_mixing", True)),
                enable_o2_omz=bool(getattr(cfg, "enable_ocean_o2_omz", False)),
                deep_tracer_accel=active_accel,
                step_bgc=do_bgc_now,
                dt_bgc_scale=bgc_scale_now,
            )
        synchronize_device(dev)
        wall_yr = time.perf_counter() - t0
        deep_eq_yr += active_accel
        m = diagnose_ocean_spinup_metrics(model.ogcm, ocn_sfc, ocn_mesh)
        ohc_flux = (m["ohc_j"] - prev_ohc_j) / (yr_sec * a_earth_m2)
        prev_ohc_j = m["ohc_j"]
        deep_hist.append(m["t_ocn_deep_k"])
        if len(deep_hist) > drift_win:
            deep_drift = (deep_hist[-1] - deep_hist[-1 - drift_win]) * (100.0 / (drift_win * max(active_accel, 1.0)))
        else:
            deep_drift = float("nan")

        row = [
            yr, f"{m['t_ocn_global_k']:.5f}", f"{m['t_ocn_0_300m_k']:.5f}",
            f"{m['t_ocn_300_1500m_k']:.5f}", f"{m['t_ocn_deep_k']:.5f}",
            f"{m['sst_global_k']:.4f}", f"{m['s_ocn_global_psu']:.5f}", f"{m['sss_global_psu']:.4f}",
            f"{m['ohc_zj']:.4f}", f"{ohc_flux:.3f}", f"{m['ocn_ke_jm2']:.2f}", f"{m['ocn_eta_rms_m']:.4f}",
            f"{m['ocn_hbl_mean_m']:.2f}", f"{m['ocn_hbl_max_m']:.1f}",
            f"{m['amoc_26n_sv']:.2f}", f"{m['amoc_max_sv']:.2f}", f"{m['acc_drake_sv']:.2f}",
            f"{m['seaice_nh_mkm2']:.3f}", f"{m['seaice_sh_mkm2']:.3f}",
            f"{deep_drift:.4f}" if math.isfinite(deep_drift) else "nan",
            f"{wall_yr:.2f}",
        ]
        bgc_str = ""
        if has_bgc:
            mb = diagnose_ocean_bgc_metrics(model.ogcm.bgc, model.ogcm, ocn_mesh)
            row += [
                f"{mb.get('ocean_npp_pgc_yr', 0.0):.2f}",
                f"{mb.get('ocean_fgco2_pgc_yr', 0.0):.3f}",
                f"{mb.get('ocean_o2_global_mmol_m3', 0.0):.2f}",
                f"{mb.get('ocean_omz_vol_pct', 0.0):.3f}",
            ]
            bgc_str = (
                f" | NPP={mb.get('ocean_npp_pgc_yr', 0.0):.1f} PgC/y"
                f" fgCO2={mb.get('ocean_fgco2_pgc_yr', 0.0):+.2f}"
                f" O2={mb.get('ocean_o2_global_mmol_m3', 0.0):.1f}"
            )
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if yr == 1 or yr % 10 == 0 or yr == args.years or in_sync_tail:
            tag = "SYNC-TAIL" if in_sync_tail else f"ACCEL {active_accel:.1f}x"
            print(
                f"[OFFLINE OGCM Y{yr:04d} ({tag}, deep~{deep_eq_yr:.0f}yr)] dOHC={ohc_flux:+5.2f} W/m2 | "
                f"Tocn(glob/up/mid/deep)={m['t_ocn_global_k']:.3f}/{m['t_ocn_0_300m_k']:.2f}/"
                f"{m['t_ocn_300_1500m_k']:.2f}/{m['t_ocn_deep_k']:.3f} K | S={m['s_ocn_global_psu']:.4f} | "
                f"AMOC={m['amoc_26n_sv']:.1f} Sv | ACC={m['acc_drake_sv']:.1f} Sv{bgc_str} | "
                f"deep_drift={deep_drift:+.4f} K/100yr | {wall_yr:.1f} s/yr",
                flush=True,
            )

        converged_now = (
            (not in_sync_tail)
            and math.isfinite(deep_drift)
            and abs(deep_drift) < args.stop_deep_drift
            and deep_eq_yr >= 200.0
        )
        reached_max_accel = (not in_sync_tail) and (yr >= args.years)

        if yr % 25 == 0 or converged_now or reached_max_accel or (in_sync_tail and sync_tail_rem <= 1):
            tmp_ck = off_ckpt_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "year": yr,
                    "deep_eq_yr": deep_eq_yr,
                    "ogcm": _serialize_ogcm(model.ogcm),
                    "sfc": {k: getattr(ocn_sfc, k).detach().cpu() for k in ocn_keys if getattr(ocn_sfc, k, None) is not None},
                },
                tmp_ck,
            )
            os.replace(tmp_ck, off_ckpt_path)

        if in_sync_tail:
            sync_tail_rem -= 1
            if sync_tail_rem <= 0:
                print(f"=== Phase-2 synchronous polish tail complete at Year {yr} (deep~{deep_eq_yr:.0f} yr)! ===", flush=True)
                break
        elif converged_now or reached_max_accel:
            reason = (
                f"deep equilibrium met (|drift|={abs(deep_drift):.4f} < {args.stop_deep_drift} K/100yr)"
                if converged_now
                else f"reached max accelerated years ({args.years} surf yr, deep~{deep_eq_yr:.0f} yr)"
            )
            if sync_tail_yr > 0:
                print(f"=== {reason} at Year {yr}; starting {sync_tail_yr}-yr synchronous (1x) polish tail... ===", flush=True)
                in_sync_tail = True
                sync_tail_rem = sync_tail_yr
            else:
                print(f"=== {reason} at Year {yr}! ===", flush=True)
                break

    inject_offline_ocean_into_restart(
        base_restart_path=base_restart,
        ogcm_state=model.ogcm,
        ocn_sfc_state=ocn_sfc,
        out_restart_path=out_dir / "restart_phase3_init.pt",
        ocn_coupler=ocn_coupler,
    )


if __name__ == "__main__":
    main()
