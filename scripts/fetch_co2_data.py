"""Fetch and update external atmospheric CO2 forcing datasets for AOBA.

Downloads latest observations and historical/paleo records into ``data/forcing/co2/``:
- NOAA GML Global Marine Surface Annual Mean (1979-present)
- NOAA GML Mauna Loa Annual Mean (1959-present)
- NCEI / Bereiter et al. (2015) Antarctic Ice Core 800 kyr Composite
- RCMIP v5.1.0 (CMIP5 RCPs & CMIP6 Historical/SSPs, 1750-2500)

Usage:
    /Users/andrew/miniforge/envs/myenv/bin/python scripts/fetch_co2_data.py
    /Users/andrew/miniforge/envs/myenv/bin/python scripts/fetch_co2_data.py --noaa-only
    /Users/andrew/miniforge/envs/myenv/bin/python scripts/fetch_co2_data.py --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aoba.co2_forcing import fetch_latest_co2_sources, get_co2_catalog_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latest CO2 forcing data from NOAA GML, NCEI Paleo, and RCMIP")
    parser.add_argument("--noaa-only", action="store_true", help="Only refresh NOAA GML latest observations (fast)")
    parser.add_argument("--skip-rcmip", action="store_true", help="Skip downloading RCMIP v5.1.0 (use bundled CMIP5/6)")
    parser.add_argument("--skip-paleo", action="store_true", help="Skip downloading NCEI 800ka ice-core composite")
    parser.add_argument("--write-bundled", action="store_true", help="Also refresh src/aoba/assets/co2_scenarios.csv")
    parser.add_argument("--list", action="store_true", help="List available CO2 scenarios and exit without downloading")
    args = parser.parse_args()

    if args.list:
        summary = get_co2_catalog_summary(prefer_cache=True)
        print(f"Active CO2 source: {summary['active_source']} (cache_exists={summary['cache_exists']})")
        for s in summary["scenarios"]:
            print(
                f"  {s['key']:24s} [{s['group']:16s}] "
                f"{s['min_year']:10.1f} .. {s['max_year']:7.1f} "
                f"({s['first_ppmv']:.1f} -> {s['last_ppmv']:.1f} ppmv, n={s['points']})"
            )
        return

    include_rcmip = not (args.noaa_only or args.skip_rcmip)
    include_paleo = not (args.noaa_only or args.skip_paleo)
    print(f"Fetching CO2 datasets (rcmip={include_rcmip}, paleo={include_paleo}, noaa=True)...", flush=True)
    status = fetch_latest_co2_sources(
        include_rcmip=include_rcmip,
        include_paleo=include_paleo,
        write_bundled=args.write_bundled,
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
