#!/usr/bin/env python3
"""Standalone post-processing utility: Convert AOBA Zarr v3 store to CF/UGRID NetCDF4 (.nc).

Keeps AOBA's runtime I/O strictly single-format (Zarr v3) while enabling seamless export
to NetCDF4 for external tools such as `cdo`, `ncdump`, and Panoply.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xarray as xr


def export_zarr_to_netcdf(zarr_path: str | Path, nc_path: str | Path) -> Path:
    """Read an AOBA Zarr v3 store and export it as a compressed NetCDF4 file."""
    src = Path(zarr_path).expanduser()
    dst = Path(nc_path).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(src, consolidated=False)
    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds.data_vars
    }
    ds.to_netcdf(dst, format="NETCDF4", engine="netcdf4", encoding=encoding)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AOBA Zarr v3 store to NetCDF4 (.nc).")
    parser.add_argument("input_zarr", type=Path, help="Input .zarr directory path")
    parser.add_argument("output_nc", type=Path, help="Output .nc file path")
    args = parser.parse_args()
    out = export_zarr_to_netcdf(args.input_zarr, args.output_nc)
    print(f"Exported {args.input_zarr} -> {out}")


if __name__ == "__main__":
    main()
