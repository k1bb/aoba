"""Fetch and cache monthly zonal-mean ozone climatology from ESGF input4MIPs for AOBA.

Downloads a CMIP6 (``UReading-CCMI-1-0``) or CMIP7 (``FZJ-CMIP-ozone-1-1``) monthly ``vmro3``
NetCDF file from LLNL ESGF, averages over longitude if needed, and saves a compact
``(12, n_plev, n_lat)`` climatology NPZ to ``data/boundary/ozone/o3_zonal_climatology.npz``
(gitignored).

Usage:
    /Users/andrew/miniforge/envs/myenv/bin/python scripts/fetch_ozone_data.py --mip CMIP6
    /Users/andrew/miniforge/envs/myenv/bin/python scripts/fetch_ozone_data.py --mip CMIP7
    /Users/andrew/miniforge/envs/myenv/bin/python scripts/fetch_ozone_data.py --status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aoba.ozone_forcing import fetch_input4mips_ozone_climatology, get_ozone_status_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch monthly zonal-mean ozone climatology from ESGF input4MIPs")
    parser.add_argument("--mip", type=str, default="CMIP6", choices=["CMIP6", "CMIP7"], help="input4MIPs source era")
    parser.add_argument("--out", type=str, default="", help="Optional output .npz path")
    parser.add_argument("--status", action="store_true", help="Print cached ozone status and exit")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(get_ozone_status_summary(), indent=2))
        return

    print(f"Fetching {args.mip} input4MIPs vmro3 from LLNL ESGF...", flush=True)
    res = fetch_input4mips_ozone_climatology(mip_era=args.mip, out_path=args.out or None)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
