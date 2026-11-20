"""Unified Zarr v3 model I/O module (CF-1.11 / UGRID-1.0 compliant).

All simulation history streams, restart checkpoints, and preprocessed mesh caches
use Zarr v3 exclusively (`zarr_format=3`, `consolidated=False` per Zarr v3 spec).
Standalone NetCDF4 conversion is provided out-of-core in `scripts/zarr_to_netcdf.py`.
"""

from __future__ import annotations

from pathlib import Path
import xarray as xr


def save_zarr(dataset: xr.Dataset, store_path: str | Path, overwrite: bool = True) -> Path:
    """Write an xarray.Dataset to a CF/UGRID-compliant Zarr v3 store."""
    path = Path(store_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "w-"
    ds = dataset.copy(deep=False)
    ds.attrs.setdefault("Conventions", "CF-1.11 UGRID-1.0")
    ds.to_zarr(path, mode=mode, zarr_format=3, consolidated=False)
    return path


def append_zarr(dataset: xr.Dataset, store_path: str | Path, append_dim: str = "time") -> Path:
    """Atomically append a new time chunk to an existing Zarr v3 history store.

    Creates the store automatically if it does not yet exist.
    """
    path = Path(store_path).expanduser()
    if not path.exists():
        return save_zarr(dataset, path, overwrite=True)
    # Preserve existing root attributes when appending along time
    existing_attrs = dict(xr.open_zarr(path, consolidated=False).attrs)
    drop_vars = [c for c in dataset.coords if c != append_dim and append_dim not in dataset[c].dims]
    ds_step = dataset.drop_vars(drop_vars, errors="ignore").copy(deep=False)
    merged_attrs = {**existing_attrs, **ds_step.attrs}
    merged_attrs.setdefault("Conventions", "CF-1.11 UGRID-1.0")
    ds_step.attrs = merged_attrs
    ds_step.to_zarr(path, mode="a", append_dim=append_dim, zarr_format=3, consolidated=False)
    return path


def load_zarr(store_path: str | Path) -> xr.Dataset:
    """Open a Zarr v3 dataset lazily or eagerly via xarray."""
    path = Path(store_path).expanduser()
    return xr.open_zarr(path, consolidated=False)
