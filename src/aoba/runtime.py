"""Hardware backend selection, data directory discovery, and output directory management.

References
----------
- Repository I/O specification: Zarr v3 native output in ~/aoba_outputs; local protected
  boundary/validation datasets in <repo>/data (Hirt & Rexer, 2015; Beck et al., 2018).
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
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


def synchronize_device(device: torch.device | str | None = None) -> None:
    """Synchronize asynchronous GPU streams cleanly on CUDA or Apple Silicon MPS."""
    dev = torch.device(device) if device is not None else get_device()
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


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


@dataclass(frozen=True)
class HardwareProfile:
    """Detected host CPU/GPU/memory specification for runtime auto-tuning."""

    chip_name: str
    backend: str
    gpu_cores: int
    perf_cores: int
    total_cores: int
    total_ram_gb: float
    device_mem_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


def _sysctl_str(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _sysctl_int(key: str, default: int = 0) -> int:
    s = _sysctl_str(key)
    try:
        return int(s) if s else default
    except ValueError:
        return default


def _detect_apple_gpu_cores() -> tuple[str, int]:
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        m_model = re.search(r'"model"\s*=\s*"([^"]+)"', out)
        m_cores = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
        model = m_model.group(1) if m_model else ""
        cores = int(m_cores.group(1)) if m_cores else 0
        return model, cores
    except Exception:
        return "", 0


@lru_cache(maxsize=1)
def detect_hardware_profile() -> HardwareProfile:
    """Detect CPU, GPU, and memory capacity on macOS (Apple Silicon), CUDA, or generic CPU."""
    total_cores = os.cpu_count() or 1
    perf_cores = total_cores
    total_ram_gb = 8.0
    chip_name = platform.processor() or "CPU"
    gpu_cores = 0

    if platform.system() == "Darwin":
        brand = _sysctl_str("machdep.cpu.brand_string")
        if brand:
            chip_name = brand
        p0 = _sysctl_int("hw.perflevel0.physicalcpu", 0)
        if p0 > 0:
            perf_cores = p0
        mem_bytes = _sysctl_int("hw.memsize", 0)
        if mem_bytes > 0:
            total_ram_gb = mem_bytes / (1024.0 ** 3)
        agx_name, agx_cores = _detect_apple_gpu_cores()
        if agx_name and chip_name in ("arm", "CPU", ""):
            chip_name = agx_name
        gpu_cores = agx_cores
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_ram_gb = kb / (1024.0 ** 2)
                        break
        except Exception:
            pass

    if torch.cuda.is_available():
        backend = "cuda"
        props = torch.cuda.get_device_properties(0)
        chip_name = props.name
        gpu_cores = int(props.multi_processor_count)
        device_mem_gb = props.total_memory / (1024.0 ** 3)
    elif torch.backends.mps.is_available():
        backend = "mps"
        try:
            rec = torch.mps.recommended_max_memory()
            device_mem_gb = rec / (1024.0 ** 3) if rec > 0 else total_ram_gb * 0.70
        except Exception:
            device_mem_gb = total_ram_gb * 0.70
    else:
        backend = "cpu"
        device_mem_gb = total_ram_gb * 0.70

    return HardwareProfile(
        chip_name=chip_name,
        backend=backend,
        gpu_cores=gpu_cores,
        perf_cores=perf_cores,
        total_cores=total_cores,
        total_ram_gb=round(total_ram_gb, 1),
        device_mem_gb=round(device_mem_gb, 1),
    )


def compute_optimal_chunks(
    glevel: int,
    nz: int,
    crm_nx: int,
    crm_ny: int,
    hw: HardwareProfile | None = None,
) -> dict[str, int | bool]:
    """Compute fastest memory-safe `crm_col_chunk` and `rad_col_chunk` for any grid resolution."""
    if hw is None:
        hw = detect_hardware_profile()

    n_cells = 10 * (4 ** max(int(glevel), 0)) + 2
    n_sub = max(int(crm_nx), 1) * max(int(crm_ny), 1)
    m_sub = n_cells * n_sub
    nz_safe = max(int(nz), 1)

    # Keep peak working buffers within ~45% of available device memory to prevent Unified Memory swap
    budget_bytes = max(hw.device_mem_gb * 0.45, 1.0) * (1024.0 ** 3)

    # Approximate Float32 working memory: ~180 * nz * n_sub bytes/host-cell for CRM, ~240 * nz bytes/subcol for radiation
    crm_bytes_per_cell = 180.0 * nz_safe * n_sub
    rad_bytes_per_col = 240.0 * nz_safe

    max_crm = max(512, int(budget_bytes / crm_bytes_per_cell))
    max_rad = max(2048, int(budget_bytes / rad_bytes_per_col))

    def _round_down_1024(v: int, lo: int) -> int:
        return max(lo, (v // 1024) * 1024)

    if max_crm >= n_cells:
        crm_chunk = max(16384, ((n_cells + 1023) // 1024) * 1024)
        crm_single_pass = True
    else:
        crm_chunk = _round_down_1024(max_crm, 512)
        crm_single_pass = False

    if max_rad >= m_sub:
        rad_chunk = max(32768, ((m_sub + 1023) // 1024) * 1024)
        rad_single_pass = True
    else:
        rad_chunk = _round_down_1024(max_rad, 2048)
        rad_single_pass = False

    return {
        "crm_col_chunk": int(crm_chunk),
        "rad_col_chunk": int(rad_chunk),
        "crm_single_pass": crm_single_pass,
        "rad_single_pass": rad_single_pass,
        "n_cells": int(n_cells),
        "m_sub": int(m_sub),
    }


_CUDA_COMPILED_FN_CACHE: dict[int, object] = {}


def maybe_cuda_compile(fn, *, device: torch.device | str | None = None):
    """Return a lazily `torch.compile`-fused version of pure-tensor `fn` on CUDA (no-op on CPU/MPS).

    Disabled when `AOBA_DISABLE_TORCH_COMPILE=1`. Falls back cleanly to eager `fn` if Inductor/Triton
    is unavailable on the host system.
    """
    if device is not None:
        dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    else:
        dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_type != "cuda" or os.environ.get("AOBA_DISABLE_TORCH_COMPILE") == "1":
        return fn
    fid = id(fn)
    cached = _CUDA_COMPILED_FN_CACHE.get(fid)
    if cached is not None:
        return cached
    try:
        compiled = torch.compile(fn, dynamic=False)
    except Exception:
        compiled = fn
    _CUDA_COMPILED_FN_CACHE[fid] = compiled
    return compiled


def apply_runtime_tuning(device: torch.device | str | None = None) -> HardwareProfile:
    """Apply thread-count and backend tuning matched to the host hardware."""
    hw = detect_hardware_profile()
    # Pin CPU intra-op threads to performance cores (avoids E-core straggler slowdown on Apple Silicon)
    try:
        torch.set_num_threads(max(hw.perf_cores, 1))
    except Exception:
        pass
    if str(device or hw.backend) == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return hw
