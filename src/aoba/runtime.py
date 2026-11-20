"""Hardware backend selection, data directory discovery, and output directory management.

References
----------
- Repository I/O specification: Zarr v3 native output in ~/aoba_outputs; local protected
  boundary/validation datasets in <repo>/data (Hirt & Rexer, 2015; Beck et al., 2018).
"""

from __future__ import annotations

import os
from pathlib import Path
import torch


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
