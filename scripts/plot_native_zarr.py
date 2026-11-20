#!/usr/bin/env python3
"""Plot AOBA native SCVT Voronoi grid directly with Cartopy (no lat-lon interpolation)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np

from aoba.io_zarr import load_zarr
from plot_demo_results import build_voronoi_polygons_platecarree


def main() -> None:
    parser = argparse.ArgumentParser(description="Native SCVT Voronoi Cartopy plotter for AOBA Zarr stores")
    parser.add_argument("--mesh", type=str, required=True, help="Path to mesh.zarr")
    parser.add_argument("--zarr", type=str, required=True, help="Path to monthly.zarr or snapshots.zarr")
    parser.add_argument("--var", type=str, default="u_zonal", help="Variable name to plot")
    parser.add_argument("--time-idx", type=int, default=-1, help="Time index")
    parser.add_argument("--lev-idx", type=int, default=14, help="Vertical level index if 3D (default ~upper troposphere)")
    parser.add_argument("--sub-stat", choices=["mean", "std", "min", "max", "idx"], default="mean", help="Reduction across CRM subcolumns if plotting *_sub")
    parser.add_argument("--crm-y", type=int, default=0, help="CRM y-index when --sub-stat=idx")
    parser.add_argument("--crm-x", type=int, default=0, help="CRM x-index when --sub-stat=idx")
    parser.add_argument("--out", type=str, default="native_map.png", help="Output PNG path")
    args = parser.parse_args()

    ds_m = load_zarr(args.mesh)
    ds_d = load_zarr(args.zarr)

    da = ds_d[args.var].isel(time=args.time_idx)
    if "nz" in da.dims:
        da = da.isel(nz=args.lev_idx)
    if "crm_y" in da.dims and "crm_x" in da.dims:
        if args.sub_stat == "idx":
            da = da.isel(crm_y=args.crm_y, crm_x=args.crm_x)
        else:
            da = getattr(da, args.sub_stat)(dim=("crm_y", "crm_x"))
    vals = da.values

    polys, owners = build_voronoi_polygons_platecarree(
        ds_m["lat_cell"].values,
        ds_m["lon_cell"].values,
        ds_m["verts_on_cell"].values,
        ds_m["n_edges_on_cell"].values,
        ds_m["lat_vert_deg"].values,
        ds_m["lon_vert_deg"].values,
    )

    fig = plt.figure(figsize=(11, 5.4), dpi=180)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    coll = PolyCollection(polys, transform=ccrs.PlateCarree(), cmap="RdBu_r" if "u_" in args.var else "viridis", edgecolor="none")
    coll.set_array(vals[owners])
    ax.add_collection(coll)
    ax.coastlines(linewidth=0.6, color="#222222")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#444444")
    plt.colorbar(coll, ax=ax, orientation="horizontal", pad=0.05, shrink=0.72, label=args.var)
    t_day = float(ds_d["time"].isel(time=args.time_idx).values)
    ax.set_title(f"AOBA Native SCVT Grid — {args.var} (day {t_day:.1f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
