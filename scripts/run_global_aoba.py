#!/usr/bin/env python3
"""Run the full global 1:1 Superparameterized AOBA Earth System Model (Zarr v3 output)."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.io_zarr import append_zarr, export_aoba_snapshot_xarray
from aoba.runtime import get_device, get_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Global 1:1 Superparameterized AOBA runner")
    parser.add_argument("--glevel", type=int, default=3, help="SCVT mesh level (3->642, 4->2562, 5->10242)")
    parser.add_argument("--days", type=float, default=5.0, help="Simulation duration [days]")
    parser.add_argument("--dt-host", type=float, default=180.0, help="Coupled host/SP step [s]")
    parser.add_argument("--co2-ppmv", type=float, default=400.0, help="Atmospheric CO2 concentration [ppmv]")
    parser.add_argument("--out-hours", type=float, default=6.0, help="History output interval [hours]")
    parser.add_argument("--out", type=str, default="", help="Output Zarr v3 store path")
    parser.add_argument("--device", type=str, default="", help="Compute device (cpu or mps)")
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else get_device()
    cfg = AOBAConfig(
        glevel=args.glevel,
        dt_host=args.dt_host,
        co2_ppmv=args.co2_ppmv,
    )
    print(f"Initializing AOBA (Glevel-{cfg.glevel}, CO2={cfg.co2_ppmv} ppmv) on device={dev}...")
    model = init_aoba_model(cfg, device=dev)

    n_steps = max(int(round(args.days * 86400.0 / cfg.dt_host)), 1)
    out_stride = max(int(round(args.out_hours * 3600.0 / cfg.dt_host)), 1)
    out_path = Path(args.out) if args.out else (
        get_output_dir() / f"aoba_global_g{cfg.glevel}_co2_{int(cfg.co2_ppmv)}.zarr"
    )

    for step in range(n_steps):
        diag = step_aoba_model(model)
        if (step + 1) % out_stride == 0 or step == n_steps - 1:
            ds = export_aoba_snapshot_xarray(model)
            append_zarr(ds, out_path)
            print(
                f"[AOBA {dev}] day={diag.sim_time_days:7.3f} | Ts={diag.global_mean_t_sfc_k:6.2f} K | "
                f"OLR={diag.global_mean_olr_wm2:5.1f} | ASR={diag.global_mean_asr_wm2:5.1f} | "
                f"P={diag.global_mean_precip_mm_day:4.2f} mm/d | GPP={diag.land_mean_gpp_umol:4.2f} | "
                f"LAI={diag.land_mean_lai:4.2f} | Ice={diag.seaice_area_million_km2:5.2f} Mkm2"
            )


if __name__ == "__main__":
    main()
