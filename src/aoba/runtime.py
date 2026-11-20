"""Hardware backend selection and cloud-safe output directory management."""

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
