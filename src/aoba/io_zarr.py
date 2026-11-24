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
    existing_ds = xr.open_zarr(path, consolidated=False)
    existing_attrs = dict(existing_ds.attrs)
    drop_vars = [c for c in dataset.coords if c != append_dim and append_dim not in dataset[c].dims]
    # Keep schema compatibility when appending into an older Zarr store lacking newly added variables
    extra_vars = [v for v in dataset.data_vars if v not in existing_ds.data_vars]
    ds_step = dataset.drop_vars(drop_vars + extra_vars, errors="ignore").copy(deep=False)
    merged_attrs = {**existing_attrs, **ds_step.attrs}
    merged_attrs.setdefault("Conventions", "CF-1.11 UGRID-1.0")
    ds_step.attrs = merged_attrs
    ds_step.to_zarr(path, mode="a", append_dim=append_dim, zarr_format=3, consolidated=False)
    return path


def load_zarr(store_path: str | Path) -> xr.Dataset:
    """Open a Zarr v3 dataset lazily or eagerly via xarray."""
    path = Path(store_path).expanduser()
    return xr.open_zarr(path, consolidated=False)


def diagnose_subgrid_2d_fields(model: "AOBAState") -> dict[str, tuple]:
    """Compute column-integrated CRM, ICA radiative, and MAML surface 2D subgrid fields `[n_cell, ny, nx]`."""
    from aoba import constants as C
    from aoba.surface import SURF_OCEAN

    crm = model.crm
    nz, nc, ny, nx = crm.rho_d.shape
    m_sub = nc * ny * nx
    dz_4 = model.atmos.dz_c[:, :, None, None]

    def _rsub(t):
        return t.reshape(nc, ny, nx)

    rc = model.rad_cache
    z_sub = _rsub(crm.rho_d[0].new_zeros(m_sub))
    rain_s = _rsub(rc["rain_sub"]) if "rain_sub" in rc else z_sub
    snow_s = _rsub(rc["snow_sub"]) if "snow_sub" in rc else z_sub
    sw1_dir_s = _rsub(rc["sw1_dir_sub"]) if "sw1_dir_sub" in rc else z_sub
    sw1_dif_s = _rsub(rc["sw1_dif_sub"]) if "sw1_dif_sub" in rc else z_sub
    sw2_dir_s = _rsub(rc["sw2_dir_sub"]) if "sw2_dir_sub" in rc else z_sub
    sw2_dif_s = _rsub(rc["sw2_dif_sub"]) if "sw2_dif_sub" in rc else z_sub
    lw_dn_s = _rsub(rc["lw_dn_sub"]) if "lw_dn_sub" in rc else z_sub
    olr_s = _rsub(rc["olr_sub"]) if "olr_sub" in rc else z_sub
    shf_s = _rsub(rc["shf_sub"]) if "shf_sub" in rc else z_sub
    evap_s = _rsub(rc["evap_sub"]) if "evap_sub" in rc else z_sub
    gpp_s = _rsub(rc["gpp_sub"]) if "gpp_sub" in rc else z_sub

    fields: dict[str, tuple] = {
        "lwp_sub": ((crm.rho_d * crm.q_c * dz_4).sum(dim=0), "kg m-2"),
        "iwp_sub": ((crm.rho_d * crm.q_i * dz_4).sum(dim=0), "kg m-2"),
        "w_max_sub": (crm.w.max(dim=0).values, "m s-1"),
        "precip_mm_day_sub": ((rain_s + snow_s) * C.SECONDS_PER_DAY, "mm day-1"),
        "sw1_dir_sub": (sw1_dir_s, "W m-2"),
        "sw1_dif_sub": (sw1_dif_s, "W m-2"),
        "sw_dn_sub": (sw1_dir_s + sw1_dif_s + sw2_dir_s + sw2_dif_s, "W m-2"),
        "lw_dn_sub": (lw_dn_s, "W m-2"),
        "olr_sub": (olr_s, "W m-2"),
        "shf_sub": (shf_s, "W m-2"),
        "evap_mm_day_sub": (evap_s * C.SECONDS_PER_DAY, "mm day-1"),
        "gpp_sub": (gpp_s, "umol m-2 s-1"),
    }

    if model.sfc_sub is not None and model.sfc_sub.t_canopy.numel() == m_sub:
        ss = model.sfc_sub
        t_sfc_s = _rsub(ss.t_ocean_ml.where(ss.surf_type == SURF_OCEAN, ss.t_canopy))
        fields["t_sfc_sub"] = (t_sfc_s, "K")
        fields["m_seaice_sub"] = (_rsub(ss.m_seaice), "kg m-2")
        fields["swe_land_sub"] = (_rsub(ss.swe_land), "kg m-2")
        fields["w_root_sub"] = (_rsub(ss.w_root), "kg m-2")
        fields["lai_sub"] = (_rsub(ss.bio.lai), "m2 m-2")

    return fields


def export_aoba_snapshot_xarray(
    model: "AOBAState",
    include_2d: bool = True,
    include_3d: bool = True,
    include_subgrid_2d: bool = True,
) -> xr.Dataset:
    """Package the current global `AOBAState` into a CF-1.11 / UGRID-1.0 `xarray.Dataset`."""
    from aoba import constants as C
    from aoba.coupled_model import diagnose_model_koppen_geiger
    from aoba.scvt_mesh import reconstruct_cell_winds
    from aoba.thermodynamics import diagnose_pressure_temperature

    mesh = model.mesh
    atmos = model.atmos
    sfc = model.sfc

    t_days = np.array([model.sim_time_sec / C.SECONDS_PER_DAY], dtype=np.float64)
    lat_deg = np.rad2deg(mesh.lat_cell.detach().cpu().numpy())
    lon_deg = np.rad2deg(mesh.lon_cell.detach().cpu().numpy())

    def _c1(t):
        return t.detach().cpu().numpy()[None, :]

    def _c2(t):
        # [nz, n_cell] -> [1, nz, n_cell]
        return t.detach().cpu().numpy()[None, :, :]

    def _c3(t):
        # [n_cell, ny, nx] -> [1, n_cell, ny, nx]
        return t.detach().cpu().numpy()[None, :, :, :]

    _, _, ny, nx = model.crm.rho_d.shape

    data_vars: dict[str, Any] = {
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
    }
    if include_2d:
        kg30, kg5 = diagnose_model_koppen_geiger(model)
        data_vars.update({
            "t_ocean_ml": (("time", "n_cell"), _c1(sfc.t_ocean_ml), {"units": "K"}),
            "t_canopy": (("time", "n_cell"), _c1(sfc.t_canopy), {"units": "K"}),
            "m_seaice": (("time", "n_cell"), _c1(sfc.m_seaice), {"units": "kg m-2"}),
            "swe_land": (("time", "n_cell"), _c1(sfc.swe_land), {"units": "kg m-2"}),
            "w_root": (("time", "n_cell"), _c1(sfc.w_root), {"units": "kg m-2"}),
            "lai": (("time", "n_cell"), _c1(sfc.bio.lai), {"units": "m2 m-2"}),
            "h_canopy": (("time", "n_cell"), _c1(sfc.bio.h_canopy), {"units": "m"}),
            "vcmax25_c3": (("time", "n_cell"), _c1(sfc.bio.vcmax25_c3), {"units": "umol m-2 s-1"}),
            "f_c4": (("time", "n_cell"), _c1(sfc.bio.f_c4), {"units": "1"}),
            "f_evg": (("time", "n_cell"), _c1(sfc.bio.f_evg if sfc.bio.f_evg is not None else sfc.bio.f_c4 * 0.0), {"units": "1"}),
            "koppen_code": (("time", "n_cell"), _c1(kg30), {"long_name": "Beck et al. (2018) 1..30 class"}),
            "koppen_major": (("time", "n_cell"), _c1(kg5), {"long_name": "1=A, 2=B, 3=C, 4=D, 5=E"}),
        })
        if model.ozone is not None:
            data_vars["o3_tco_du"] = (("time", "n_cell"), _c1(model.ozone.tco_du), {"units": "DU"})
        if model.icesheet is not None:
            data_vars["h_ice"] = (("time", "n_cell"), _c1(model.icesheet.h_ice), {"units": "m"})
            data_vars["z_topo_dyn"] = (("time", "n_cell"), _c1(mesh.z_topo), {"units": "m"})
        if model.ogcm is not None:
            og = model.ogcm
            ocn_dim = "n_cell_ocn" if model.ocn_mesh is not None else "n_cell"
            data_vars["ocn_eta"] = (("time", ocn_dim), _c1(og.eta), {"units": "m"})
            data_vars["ocn_sss"] = (("time", ocn_dim), _c1(og.sal[0]), {"units": "psu"})
            if og.h_bl is not None:
                data_vars["ocn_h_bl"] = (("time", ocn_dim), _c1(og.h_bl), {"units": "m"})
    if include_3d:
        u_c, v_c = reconstruct_cell_winds(mesh, atmos.u_edge)
        pres_c, _, temp_c = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
        data_vars.update({
            "temp": (("time", "nz", "n_cell"), _c2(temp_c), {"units": "K"}),
            "pres": (("time", "nz", "n_cell"), _c2(pres_c), {"units": "Pa"}),
            "q_v": (("time", "nz", "n_cell"), _c2(atmos.q_v), {"units": "kg kg-1"}),
            "u_zonal": (("time", "nz", "n_cell"), _c2(u_c), {"units": "m s-1"}),
            "v_merid": (("time", "nz", "n_cell"), _c2(v_c), {"units": "m s-1"}),
        })
        if model.ozone is not None:
            data_vars["o3_vmr"] = (("time", "nz", "n_cell"), _c2(model.ozone.o3_vmr), {"units": "mol mol-1"})
        if model.ogcm is not None:
            og = model.ogcm
            ocn_mesh_z = model.ocn_mesh if model.ocn_mesh is not None else mesh
            ocn_dim = "n_cell_ocn" if model.ocn_mesh is not None else "n_cell"
            uo_c, vo_c = reconstruct_cell_winds(ocn_mesh_z, og.u_edge * og.wet_e.to(og.u_edge.dtype))
            wo_c = 0.5 * (og.w_vel[:-1] + og.w_vel[1:])
            data_vars.update({
                "ocn_temp": (("time", "nz_ocn", ocn_dim), _c2(og.temp), {"units": "K"}),
                "ocn_sal": (("time", "nz_ocn", ocn_dim), _c2(og.sal), {"units": "psu"}),
                "ocn_u_zonal": (("time", "nz_ocn", ocn_dim), _c2(uo_c), {"units": "m s-1"}),
                "ocn_v_merid": (("time", "nz_ocn", ocn_dim), _c2(vo_c), {"units": "m s-1"}),
                "ocn_w_vel": (("time", "nz_ocn", ocn_dim), _c2(wo_c), {"units": "m s-1"}),
            })
    if include_subgrid_2d:
        for k, (val, unit) in diagnose_subgrid_2d_fields(model).items():
            data_vars[k] = (("time", "n_cell", "crm_y", "crm_x"), _c3(val), {"units": unit})

    coords: dict[str, Any] = {
        "time": ("time", t_days, {"units": "days since 2000-03-20 00:00:00"}),
        "lat_cell": ("n_cell", lat_deg, {"units": "degrees_north", "standard_name": "latitude"}),
        "lon_cell": ("n_cell", lon_deg, {"units": "degrees_east", "standard_name": "longitude"}),
    }
    if model.ocn_mesh is not None:
        coords["lat_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lat_cell.detach().cpu().numpy()), {"units": "degrees_north"})
        coords["lon_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lon_cell.detach().cpu().numpy()), {"units": "degrees_east"})
    if include_3d:
        coords["nz"] = ("nz", model.vgrid.zeta_c.detach().cpu().numpy(), {"units": "m", "positive": "up"})
        if model.ogcm is not None:
            coords["nz_ocn"] = ("nz_ocn", model.ogcm.z_c.detach().cpu().numpy(), {"units": "m", "positive": "down"})
    if include_subgrid_2d:
        coords["crm_y"] = ("crm_y", np.arange(ny, dtype=np.float32) * model.crm.dy, {"units": "m"})
        coords["crm_x"] = ("crm_x", np.arange(nx, dtype=np.float32) * model.crm.dx, {"units": "m"})

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "Conventions": "CF-1.11 UGRID-1.0",
            "model": "AOBA (Autonomous Optimality Biosphere-Atmosphere)",
            "glevel": model.cfg.glevel,
            "co2_ppmv": model.cfg.co2_ppmv,
        },
    )
    return ds


def export_scvt_mesh_xarray(model: "AOBAState") -> xr.Dataset:
    """Export UGRID-1.0 SCVT Voronoi polygon topology for direct native-grid Cartopy plotting."""
    import torch

    mesh = model.mesh
    ev = mesh.edges_on_vert
    ce_on_v = mesh.cells_on_edge[ev]
    xyz_sum = mesh.xyz_cell[ce_on_v].sum(dim=(1, 2))
    xyz_vert = xyz_sum / torch.clamp(torch.norm(xyz_sum, dim=-1, keepdim=True), min=1.0e-12)
    lat_vert_deg = torch.rad2deg(torch.asin(torch.clamp(xyz_vert[:, 2], -1.0, 1.0))).detach().cpu().numpy()
    lon_vert_deg = torch.rad2deg(torch.atan2(xyz_vert[:, 1], xyz_vert[:, 0])).detach().cpu().numpy()

    e_cell = mesh.edges_on_cell
    sgn_cell = mesh.edge_sign_on_cell
    verts_on_cell = torch.where(
        sgn_cell >= 0.0,
        mesh.verts_on_edge[e_cell, 0],
        mesh.verts_on_edge[e_cell, 1],
    ).detach().cpu().numpy()

    data_vars: dict[str, Any] = {
        "area_cell": (("n_cell",), mesh.area_cell.detach().cpu().numpy(), {"units": "m2"}),
        "surf_type": (("n_cell",), mesh.surf_type.detach().cpu().numpy(), {"long_name": "0=Ocean, 1=Land, 2=IceSheet"}),
        "z_topo": (("n_cell",), mesh.z_topo.detach().cpu().numpy(), {"units": "m"}),
        "verts_on_cell": (("n_cell", "max_n_vert"), verts_on_cell),
        "n_edges_on_cell": (("n_cell",), mesh.n_edges_on_cell.detach().cpu().numpy()),
    }
    coords: dict[str, Any] = {
        "lat_cell": ("n_cell", np.rad2deg(mesh.lat_cell.detach().cpu().numpy()), {"units": "degrees_north"}),
        "lon_cell": ("n_cell", np.rad2deg(mesh.lon_cell.detach().cpu().numpy()), {"units": "degrees_east"}),
        "lat_vert_deg": ("n_vert", lat_vert_deg, {"units": "degrees_north"}),
        "lon_vert_deg": ("n_vert", lon_vert_deg, {"units": "degrees_east"}),
        "zeta_c": ("nz", model.vgrid.zeta_c.detach().cpu().numpy(), {"units": "m"}),
        "zeta_w": ("nz_w", model.vgrid.zeta_w.detach().cpu().numpy(), {"units": "m"}),
    }
    if mesh.z_bed is not None:
        data_vars["z_bed"] = (("n_cell",), mesh.z_bed.detach().cpu().numpy(), {"units": "m"})
    if model.ogcm is not None:
        og = model.ogcm
        ocn_dim = "n_cell_ocn" if model.ocn_mesh is not None else "n_cell"
        data_vars["ocn_h_c"] = (("nz_ocn", ocn_dim), og.h_c.detach().cpu().numpy(), {"units": "m"})
        data_vars["ocn_h_col"] = ((ocn_dim,), og.h_col.detach().cpu().numpy(), {"units": "m"})
        data_vars["ocn_wet_c"] = (("nz_ocn", ocn_dim), og.wet_c.detach().cpu().numpy().astype(np.int8))
        coords["nz_ocn"] = ("nz_ocn", og.z_c.detach().cpu().numpy(), {"units": "m", "positive": "down"})
        coords["nz_ocn_w"] = ("nz_ocn_w", og.z_w.detach().cpu().numpy(), {"units": "m", "positive": "down"})
        if model.ocn_mesh is not None:
            data_vars["area_cell_ocn"] = (("n_cell_ocn",), model.ocn_mesh.area_cell.detach().cpu().numpy(), {"units": "m2"})
            coords["lat_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lat_cell.detach().cpu().numpy()), {"units": "degrees_north"})
            coords["lon_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lon_cell.detach().cpu().numpy()), {"units": "degrees_east"})

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "Conventions": "CF-1.11 UGRID-1.0",
            "glevel": model.cfg.glevel,
        },
    )
