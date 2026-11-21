#!/usr/bin/env python3
"""All-in-one AOBA bootstrap script for fresh clones & new machines.

Performs end-to-end setup:
  1. Verifies Python >= 3.10, required scientific libraries (`torch`, `numpy`, `scipy`,
     `xarray`, `zarr`, `netcdf4`, `matplotlib`), and GPU backend (`mps` / `cuda` / `cpu`).
  2. Verifies or downloads 5-arc-min Earth2014 topography / ice-sheet / bedrock binaries
     (`SUR2014`, `ICE2014`, `TBI2014`, `BED2014`) into `data/boundary/earth2014_5min/`.
  3. Runs the AOBA verification test suites.
  4. Executes a brief Earth2014-coupled global `Glevel-3` smoke test run verifying Zarr v3 I/O.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


REQUIRED_MODULES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy>=1.24"),
    ("torch", "torch>=2.2"),
    ("scipy", "scipy>=1.10"),
    ("xarray", "xarray>=2025.1"),
    ("zarr", "zarr>=3.0"),
    ("netCDF4", "netcdf4>=1.6"),
    ("matplotlib", "matplotlib>=3.7"),
)
OPTIONAL_MODULES: tuple[tuple[str, str], ...] = (
    ("cartopy", "cartopy"),
)


def check_dependencies(*, install_missing: bool = False, editable: bool = False) -> None:
    """Inspect Python, required libraries, and GPU backend availability."""
    print("=== [Step 1/4] Checking Python & Dependencies ===", flush=True)
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python >= 3.10 is required (found {sys.version.split()[0]}).")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)

    missing_pkgs: list[str] = []
    for mod_name, pip_spec in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {mod_name:<12s} {ver}", flush=True)
        except ImportError:
            print(f"  [MISSING] {mod_name:<12s} ({pip_spec})", flush=True)
            missing_pkgs.append(pip_spec)

    for mod_name, pip_spec in OPTIONAL_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            print(f"  [OK] {mod_name:<12s} {getattr(mod, '__version__', 'installed')} (optional)", flush=True)
        except ImportError:
            print(f"  [OPT] {mod_name:<12s} not installed (optional for Cartopy static plots)", flush=True)

    if missing_pkgs:
        if install_missing:
            print(f"Installing missing packages: {missing_pkgs}", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", *missing_pkgs], check=True)
        else:
            raise RuntimeError(
                "Missing required libraries: "
                + ", ".join(missing_pkgs)
                + "\nRe-run with `--install-deps` or run: pip install "
                + " ".join(f'"{p}"' for p in missing_pkgs)
            )

    if editable:
        print("Installing `aoba` in editable mode (`pip install --no-build-isolation --no-deps -e .`)...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."],
            cwd=str(_REPO_ROOT),
            check=True,
        )

    import torch

    if torch.cuda.is_available():
        backend = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        backend = "mps (Apple Silicon GPU)"
    else:
        backend = "cpu"
    print(f"Compute backend: {backend}", flush=True)


def fetch_earth2014(*, prefer_wget: bool = False, include_ret: bool = False) -> Path:
    """Verify and download required Earth2014 5-arc-min boundary binaries."""
    print("\n=== [Step 2/4] Verifying / Downloading Earth2014 Boundary Data ===", flush=True)
    from aoba.runtime import EARTH2014_REQUIRED_LAYERS, ensure_earth2014_data

    layers = EARTH2014_REQUIRED_LAYERS + (("RET2014",) if include_ret else ())
    target_dir = ensure_earth2014_data(
        None, layers=layers, auto_download=True, prefer_wget=prefer_wget, verbose=True
    )
    for layer in layers:
        fp = target_dir / f"Earth2014.{layer}.5min.geod.bin"
        print(f"  [OK] {fp.name} ({fp.stat().st_size:,} bytes)", flush=True)

    scvt_cache = _REPO_ROOT / "config" / "scvt_preview_meshes.npz"
    if not scvt_cache.exists():
        scripts_dir = _REPO_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from build_scvt_preview_cache import build_preview_archive

        print("Precomputing SCVT G0..G6 GUI preview cache...", flush=True)
        build_preview_archive(scvt_cache, max_glevel=6)
    return target_dir


def run_test_suites() -> None:
    """Run all AOBA unit & integration verification tests."""
    print("\n=== [Step 3/4] Running AOBA Verification Test Suites ===", flush=True)
    tests = [
        "tests/test_phase1_foundation.py",
        "tests/test_phase1_physics_biosphere.py",
        "tests/test_phase1_crm_rce.py",
        "tests/test_phase2_phase3_global.py",
        "tests/test_params_and_gui.py",
    ]
    for t in tests:
        print(f"--- Running {t} ---", flush=True)
        subprocess.run([sys.executable, str(_REPO_ROOT / t)], cwd=str(_REPO_ROOT), check=True)


def run_global_smoke_test(*, device: str = "") -> None:
    """Run a brief coupled Earth2014 global Glevel-3 smoke integration into a temp directory."""
    print("\n=== [Step 4/4] Running Global Earth2014 Glevel-3 Coupled Smoke Run ===", flush=True)
    import xarray as xr

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_global_aoba.py"),
        "--glevel", "3",
        "--days", "0.25",
        "--dt-host", "720",
        "--out-hours", "6.0",
    ]
    if device:
        cmd.extend(["--device", device])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_zarr = Path(tmpdir) / "smoke_g3.zarr"
        cmd.extend(["--out", str(out_zarr)])
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)
        ds = xr.open_zarr(out_zarr, consolidated=False)
        print(
            f"  [OK] Smoke Zarr verified: n_cell={ds.sizes['n_cell']}, time={ds.sizes['time']}, vars={len(ds.data_vars)}",
            flush=True,
        )
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap AOBA: deps -> Earth2014 download -> tests -> smoke run.")
    parser.add_argument("--install-deps", action="store_true", help="Install missing PyPI dependencies via pip")
    parser.add_argument("--editable", action="store_true", help="Run `pip install --no-build-isolation --no-deps -e .`")
    parser.add_argument("--wget", action="store_true", help="Prefer external `wget` CLI if available when downloading Earth2014")
    parser.add_argument("--include-ret", action="store_true", help="Also download optional Earth2014.RET2014.5min.geod.bin")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit/integration test suites")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip Earth2014 global smoke test run")
    parser.add_argument("--device", type=str, default="", help="Device for final smoke test run (cpu, mps, cuda)")
    args = parser.parse_args()

    check_dependencies(install_missing=args.install_deps, editable=args.editable)
    fetch_earth2014(prefer_wget=args.wget, include_ret=args.include_ret)
    if not args.skip_tests:
        run_test_suites()
    if not args.skip_smoke:
        run_global_smoke_test(device=args.device)

    print("\n=== AOBA Setup Complete! Ready to run. ===", flush=True)
    print(f"Launch AOBA Studio GUI : {sys.executable} scripts/aoba_config_gui.py", flush=True)
    print(f"Run Multi-Year Model   : {sys.executable} -u scripts/run_g5_30yr.py --config config/aoba_default.toml", flush=True)


if __name__ == "__main__":
    main()
