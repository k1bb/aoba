#!/usr/bin/env python3
"""Run the Held & Suarez (1994) global SCVT Voronoi nonhydrostatic dynamical-core benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import xarray as xr

from aoba.io_zarr import append_zarr
from aoba.runtime import get_device, get_output_dir
from aoba.scvt_dynamics import compute_held_suarez_forcing, init_global_hydrostatic_atmosphere, step_scvt_rk3_hevi
from aoba.scvt_mesh import build_scvt_mesh, reconstruct_cell_winds
from aoba.thermodynamics import diagnose_pressure_temperature
from aoba.vertical_grid import build_vertical_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="AOBA Held-Suarez SCVT Voronoi benchmark")
    parser.add_argument("--glevel", type=int, default=3, help="SCVT mesh level (3 -> 642 cells)")
    parser.add_argument("--days", type=float, default=5.0, help="Duration [days]")
    parser.add_argument("--dt", type=float, default=150.0, help="Dynamical step [s]")
    parser.add_argument("--out", type=str, default="", help="Output Zarr v3 store path")
    parser.add_argument("--device", type=str, default="", help="cpu or mps")
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else get_device()
    mesh = build_scvt_mesh(glevel=args.glevel, n_lloyd_iter=10, use_earth2014=False, device=dev)
    vgrid = build_vertical_grid(device=dev)
    state = init_global_hydrostatic_atmosphere(mesh, vgrid, moist=False, baroclinic_pert=True)

    n_steps = max(int(round(args.days * 86400.0 / args.dt)), 1)
    out_stride = max(int(round(21600.0 / args.dt)), 1)
    out_path = Path(args.out) if args.out else (get_output_dir() / f"held_suarez_g{args.glevel}.zarr")

    for step in range(n_steps):
        du_hs, dth_hs = compute_held_suarez_forcing(state, mesh)
        step_scvt_rk3_hevi(state, mesh, vgrid, dt_dyn=args.dt, du_phys_edge=du_hs, dtheta_m_phys=dth_hs)

        if (step + 1) % out_stride == 0 or step == n_steps - 1:
            t_day = (step + 1) * args.dt / 86400.0
            uc, vc = reconstruct_cell_winds(mesh, state.u_edge)
            _, _, temp = diagnose_pressure_temperature(state.rho_d, state.theta_m, state.q_v)
            ds = xr.Dataset(
                data_vars={
                    "u_zonal": (("time", "nz", "n_cell"), uc.detach().cpu().numpy()[None, :, :]),
                    "v_merid": (("time", "nz", "n_cell"), vc.detach().cpu().numpy()[None, :, :]),
                    "temp": (("time", "nz", "n_cell"), temp.detach().cpu().numpy()[None, :, :]),
                },
                coords={
                    "time": ("time", np.array([t_day], dtype=np.float64), {"units": "days"}),
                    "nz": ("nz", vgrid.zeta_c.detach().cpu().numpy()),
                    "lat_cell": ("n_cell", np.rad2deg(mesh.lat_cell.detach().cpu().numpy())),
                    "lon_cell": ("n_cell", np.rad2deg(mesh.lon_cell.detach().cpu().numpy())),
                },
            )
            append_zarr(ds, out_path)
            print(f"[Held-Suarez {dev}] day={t_day:6.2f} | max|u|={uc.abs().max().item():6.2f} m/s")


if __name__ == "__main__":
    main()
