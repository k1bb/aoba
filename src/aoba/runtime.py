"""Hardware backend selection, data directory discovery, and output directory management.

References
----------
- Repository I/O specification: Zarr v3 native output in ~/aoba_outputs; local protected
  boundary/validation datasets in <repo>/data (Hirt & Rexer, 2015; Beck et al., 2018).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import torch


EARTH2014_BASE_URL = "https://ddfe.curtin.edu.au/models/Earth2014/data_5min/topo_grids"
EARTH2014_5MIN_BYTES = 2160 * 4320 * 2  # 18,662,400 bytes (>i2 global 5-arc-min grid)
EARTH2014_REQUIRED_LAYERS = ("SUR2014", "ICE2014", "TBI2014", "BED2014")


def get_device(preferred: str | None = None) -> torch.device:
    """Return the optimal Float32 compute device (CUDA -> MPS -> CPU)."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_output_dir(run_name: str | None = None) -> Path:
    """Return local SSD output path outside Google Drive sync by default."""
    base = Path(os.environ.get("AOBA_OUTPUT_DIR", Path.home() / "aoba_outputs")).expanduser()
    target = base / run_name if run_name else base
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_data_dir() -> Path:
    """Return path to local boundary and validation datasets."""
    if "AOBA_DATA_DIR" in os.environ:
        return Path(os.environ["AOBA_DATA_DIR"]).expanduser()
    return Path(__file__).resolve().parents[2] / "data"


def resolve_earth2014_dir(earth2014_dir: str | Path | None = None) -> Path:
    """Resolve `earth2014_dir` cleanly relative to repo root / `AOBA_DATA_DIR`."""
    default_dir = get_data_dir() / "boundary" / "earth2014_5min"
    if earth2014_dir is None or str(earth2014_dir).strip() == "":
        return default_dir
    p = Path(earth2014_dir).expanduser()
    if p.is_absolute() or p.exists():
        return p
    repo_rel = Path(__file__).resolve().parents[2] / p
    return repo_rel if (repo_rel.exists() or str(p) == "data/boundary/earth2014_5min") else default_dir


def _download_file(url: str, dst: Path, *, prefer_wget: bool = False) -> None:
    """Download `url` to `dst` atomically via `.part` temp file (`wget`, `curl`, or `urllib`)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.part")
    if tmp.exists():
        tmp.unlink()

    if prefer_wget and shutil.which("wget"):
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(tmp), url], check=True)
    else:
        try:
            with urllib.request.urlopen(url, timeout=60.0) as resp, open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception:
            if shutil.which("wget"):
                subprocess.run(["wget", "-q", "-O", str(tmp), url], check=True)
            elif shutil.which("curl"):
                subprocess.run(["curl", "-fsSL", "-o", str(tmp), url], check=True)
            else:
                raise

    if tmp.stat().st_size != EARTH2014_5MIN_BYTES:
        got = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded size mismatch for {dst.name}: got {got} bytes, expected {EARTH2014_5MIN_BYTES}")
    os.replace(tmp, dst)


def ensure_earth2014_data(
    earth2014_dir: str | Path | None = None,
    *,
    layers: tuple[str, ...] = EARTH2014_REQUIRED_LAYERS,
    auto_download: bool = True,
    prefer_wget: bool = False,
    verbose: bool = True,
) -> Path:
    """Verify Earth2014 5-arc-min binary grids and auto-download missing layers on first use."""
    target_dir = resolve_earth2014_dir(earth2014_dir)
    missing: list[str] = []
    for layer in layers:
        f = target_dir / f"Earth2014.{layer}.5min.geod.bin"
        if not (f.exists() and f.stat().st_size == EARTH2014_5MIN_BYTES):
            missing.append(layer)

    if not missing:
        return target_dir
    if not auto_download or os.environ.get("AOBA_NO_AUTO_DOWNLOAD") == "1":
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    for layer in missing:
        fname = f"Earth2014.{layer}.5min.geod.bin"
        url = f"{EARTH2014_BASE_URL}/{fname}"
        dst = target_dir / fname
        if verbose:
            print(f"[AOBA Setup] Downloading {fname} (17.8 MB) from {EARTH2014_BASE_URL} ...", flush=True)
        _download_file(url, dst, prefer_wget=prefer_wget)
        if verbose:
            print(f"[AOBA Setup] Saved {dst}", flush=True)
    return target_dir
