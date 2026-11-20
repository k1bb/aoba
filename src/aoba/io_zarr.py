"""Unified Zarr v3 model I/O module (CF-1.11 / UGRID-1.0 compliant).

All simulation history streams, restart checkpoints, and preprocessed mesh caches
use Zarr v3 exclusively (`zarr_format=3`, `consolidated=False` per Zarr v3 spec).
Standalone NetCDF4 conversion is provided out-of-core in `scripts/zarr_to_netcdf.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from aoba.coupled_model import AOBAState


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


def export_aoba_snapshot_xarray(model: "AOBAState") -> xr.Dataset:
    """Package the current global `AOBAState` into a CF-1.11 / UGRID-1.0 `xarray.Dataset`."""
    from aoba import constants as C
    from aoba.coupled_model import diagnose_model_koppen_geiger
    from aoba.scvt_mesh import reconstruct_cell_winds
    from aoba.thermodynamics import diagnose_pressure_temperature

    mesh = model.mesh
    atmos = model.atmos
    sfc = model.sfc

    u_c, v_c = reconstruct_cell_winds(mesh, atmos.u_edge)
    pres_c, _, temp_c = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
    kg30, kg5 = diagnose_model_koppen_geiger(model)

    t_days = np.array([model.sim_time_sec / C.SECONDS_PER_DAY], dtype=np.float64)
    lat_deg = np.rad2deg(mesh.lat_cell.detach().cpu().numpy())
    lon_deg = np.rad2deg(mesh.lon_cell.detach().cpu().numpy())

    def _c1(t):
        return t.detach().cpu().numpy()[None, :]

    def _c2(t):
        # [nz, n_cell] -> [1, nz, n_cell]
        return t.detach().cpu().numpy()[None, :, :]

    ds = xr.Dataset(
        data_vars={
            "mesh2d": xr.DataArray(
                np.int32(0),
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "face_dimension": "n_cell",
                    "edge_dimension": "n_edge",
                    "face_coordinates": "lon_cell lat_cell",
                },
            ),
            "area_cell": (("n_cell",), mesh.area_cell.detach().cpu().numpy(), {"units": "m2"}),
            "surf_type": (("n_cell",), mesh.surf_type.detach().cpu().numpy(), {"long_name": "0=Ocean, 1=Land, 2=IceSheet"}),
            "z_topo": (("n_cell",), mesh.z_topo.detach().cpu().numpy(), {"units": "m"}),
            "t_ocean_ml": (("time", "n_cell"), _c1(sfc.t_ocean_ml), {"units": "K"}),
            "t_canopy": (("time", "n_cell"), _c1(sfc.t_canopy), {"units": "K"}),
            "m_seaice": (("time", "n_cell"), _c1(sfc.m_seaice), {"units": "kg m-2"}),
            "swe_land": (("time", "n_cell"), _c1(sfc.swe_land), {"units": "kg m-2"}),
            "w_root": (("time", "n_cell"), _c1(sfc.w_root), {"units": "kg m-2"}),
            "lai": (("time", "n_cell"), _c1(sfc.bio.lai), {"units": "m2 m-2"}),
            "h_canopy": (("time", "n_cell"), _c1(sfc.bio.h_canopy), {"units": "m"}),
            "vcmax25_c3": (("time", "n_cell"), _c1(sfc.bio.vcmax25_c3), {"units": "umol m-2 s-1"}),
            "f_c4": (("time", "n_cell"), _c1(sfc.bio.f_c4), {"units": "1"}),
            "koppen_code": (("time", "n_cell"), _c1(kg30), {"long_name": "Beck et al. (2018) 1..30 class"}),
            "koppen_major": (("time", "n_cell"), _c1(kg5), {"long_name": "1=A, 2=B, 3=C, 4=D, 5=E"}),
            "temp": (("time", "nz", "n_cell"), _c2(temp_c), {"units": "K"}),
            "pres": (("time", "nz", "n_cell"), _c2(pres_c), {"units": "Pa"}),
            "q_v": (("time", "nz", "n_cell"), _c2(atmos.q_v), {"units": "kg kg-1"}),
            "u_zonal": (("time", "nz", "n_cell"), _c2(u_c), {"units": "m s-1"}),
            "v_merid": (("time", "nz", "n_cell"), _c2(v_c), {"units": "m s-1"}),
        },
        coords={
            "time": ("time", t_days, {"units": "days since 2000-03-20 00:00:00"}),
            "nz": ("nz", model.vgrid.zeta_c.detach().cpu().numpy(), {"units": "m", "positive": "up"}),
            "lat_cell": ("n_cell", lat_deg, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon_cell": ("n_cell", lon_deg, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.11 UGRID-1.0",
            "model": "AOBA (Autonomous Optimality Biosphere-Atmosphere)",
            "glevel": model.cfg.glevel,
            "co2_ppmv": model.cfg.co2_ppmv,
        },
    )
    return ds
