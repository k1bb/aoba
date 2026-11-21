#!/usr/bin/env python3
"""Run a coupled 3D Nonhydrostatic CRM Radiative-Convective Equilibrium (RCE) simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import torch
import xarray as xr

from aoba.crm import build_discrete_hydrostatic_rcemip_crm, step_coupled_crm_rce
from aoba.io_zarr import append_zarr
from aoba.runtime import get_device, get_output_dir
from aoba.surface import SURF_LAND, SURF_OCEAN, init_surface_state
from aoba.vertical_grid import build_vertical_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="AOBA 3D CRM RCE runner (Zarr v3 output)")
    parser.add_argument("--days", type=float, default=1.0, help="Simulation duration [days]")
    parser.add_argument("--dt", type=float, default=20.0, help="CRM step [s]")
    parser.add_argument("--co2-ppmv", type=float, default=400.0, help="Atmospheric CO2 [ppmv]")
    parser.add_argument("--surface", choices=["ocean", "land"], default="ocean")
    parser.add_argument("--out", type=str, default="", help="Output Zarr v3 store path")
    parser.add_argument("--device", type=str, default="", help="cpu or mps")
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else get_device()
    vgrid = build_vertical_grid(device=dev, dtype=torch.float32)
    crm = build_discrete_hydrostatic_rcemip_crm(vgrid, n_col=1, pert_amplitude_k=0.10)

    stype = SURF_LAND if args.surface == "land" else SURF_OCEAN
    sfc = init_surface_state(torch.tensor([stype], device=dev), t_init=300.0)
    if stype == SURF_LAND:
        sfc.bio.lai.fill_(3.5)
        sfc.bio.h_canopy.fill_(20.0)

    mu0 = torch.tensor([0.7426], device=dev)
    rad_cache: dict[str, torch.Tensor] = {}
    n_steps = max(int(round(args.days * 86400.0 / args.dt)), 1)
    out_stride = max(int(round(3600.0 / args.dt)), 1)
    out_path = Path(args.out) if args.out else (get_output_dir() / f"rce_{args.surface}.zarr")

    for step in range(n_steps):
        diag = step_coupled_crm_rce(
            crm=crm,
            sfc=sfc,
            vgrid=vgrid,
            dt_crm=args.dt,
            step_idx=step,
            cos_zenith=mu0,
            co2_ppmv=args.co2_ppmv,
            rad_stride=6,
            cached_rad_cache=rad_cache,
        )
        if (step + 1) % out_stride == 0 or step == n_steps - 1:
            t_day = (step + 1) * args.dt / 86400.0
            ds = xr.Dataset(
                data_vars={
                    "olr": (("time",), np.array([float(diag.olr[0].item())], dtype=np.float32)),
                    "asr": (("time",), np.array([float(diag.asr[0].item())], dtype=np.float32)),
                    "precip_mm_day": (("time",), np.array([float(diag.precip_tot[0].item()) * 86400.0], dtype=np.float32)),
                    "gpp": (("time",), np.array([float(diag.gpp[0].item())], dtype=np.float32)),
                    "max_w": (("time",), np.array([diag.max_w], dtype=np.float32)),
                },
                coords={"time": ("time", np.array([t_day], dtype=np.float64), {"units": "days"})},
            )
            append_zarr(ds, out_path)
            print(
                f"[RCE {dev}] day={t_day:6.3f} | OLR={diag.olr[0].item():6.1f} W/m2 | "
                f"P={diag.precip_tot[0].item()*86400.0:5.2f} mm/d | GPP={diag.gpp[0].item():5.2f} | max|w|={diag.max_w:5.2f} m/s"
            )


if __name__ == "__main__":
    main()
