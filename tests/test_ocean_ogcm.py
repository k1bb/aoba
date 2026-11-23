"""Verification tests for AOBA's 3D multi-layer SCVT OGCM and dynamic sea-ice drift.

Validates:
1. Exact global enthalpy and salt conservation over wind-driven multi-step integration
   with GM bolus transport, Redi diffusion, barotropic free surface, and vertical mixing.
2. Exact constant-field preservation (uniform T & S remain constant under wind stress).
3. Nonlinear EOS (Roquet et al., 2015) cabbeling / haline contraction and convective
   adjustment mixing dense cold/salty polar surface water into deep layers.
4. Conservative dynamic sea-ice drift with zero coastline leakage.
5. Selectable `ocean_mode="ogcm_3d"` vs `ocean_mode="ekman_sw"` in `step_aoba_model`
   on both CPU and Apple Silicon `mps`.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.ocean_ogcm import (
    S_REF_PSU,
    compute_seawater_density_anomaly,
    init_ogcm_state,
    step_ogcm_voronoi,
    step_seaice_drift_voronoi,
)
from aoba.scvt_mesh import build_scvt_mesh
from aoba.surface import SURF_LAND, SURF_OCEAN, init_surface_state


def _make_basin_mesh_and_sfc(device: str = "cpu"):
    mesh = build_scvt_mesh(glevel=2, device=device)
    # Create realistic continent + mid-ocean ridge bathymetry for zero-flux coastal checks
    lat = mesh.lat_cell
    lon = mesh.lon_cell
    is_land = (lat > -0.3) & (lat < 0.7) & (lon > -0.6) & (lon < 0.6)
    mesh.surf_type = torch.where(is_land, torch.full_like(mesh.surf_type, SURF_LAND), torch.full_like(mesh.surf_type, SURF_OCEAN))
    mesh.z_bed = torch.where(
        is_land,
        torch.full((mesh.n_cells,), 300.0, device=device),
        -3500.0 + 1500.0 * torch.cos(2.0 * lon),  # 2000..5000 m submarine ridges
    )
    sfc = init_surface_state(mesh.surf_type, t_init=290.0, h_ml_ocean=50.0)
    # Meridional SST gradient (300 K equator -> 274 K poles)
    sfc.t_ocean_ml = torch.clamp(274.0 + 26.0 * (torch.cos(lat) ** 2), min=C.T_FREEZE_OCEAN + 0.5)
    return mesh, sfc


def _total_heat_and_salt(ogcm, mesh) -> tuple[float, float]:
    h_eff = ogcm.h_c.clone()
    h_eff[0] = torch.where(ogcm.wet_c[0], torch.clamp(ogcm.h_c[0] + ogcm.eta, min=10.0), torch.zeros_like(ogcm.eta))
    vol = h_eff * mesh.area_cell.unsqueeze(0)
    heat = float((vol * ogcm.temp).sum().item())
    salt = float((vol * ogcm.sal).sum().item())
    return heat, salt


def test_ogcm_constant_preservation_and_global_conservation() -> None:
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    ogcm = init_ogcm_state(sfc, mesh, nz_ocn=10)
    zeros = torch.zeros(mesh.n_cells)
    # Realistic zonal trade-wind & westerly stress profile
    lat = mesh.lat_cell
    tau_x = 0.12 * torch.cos(3.0 * lat)  # atmospheric upward stress -> -tau_x acts on ocean
    tau_y = 0.03 * torch.sin(2.0 * lat)

    # 1. Exact constant-field preservation under strong wind forcing
    ogcm_const = init_ogcm_state(sfc, mesh, nz_ocn=10)
    ogcm_const.temp = torch.where(ogcm_const.wet_c, torch.full_like(ogcm_const.temp, 286.0), ogcm_const.temp)
    ogcm_const.sal = torch.where(ogcm_const.wet_c, torch.full_like(ogcm_const.sal, 34.7), ogcm_const.sal)
    sfc_const = init_surface_state(mesh.surf_type, t_init=286.0, h_ml_ocean=50.0)
    for _ in range(10):
        step_ogcm_voronoi(
            ogcm_const, sfc_const, mesh, tau_x, tau_y, zeros,
            zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False,
        )
    t_err = float((ogcm_const.temp[ogcm_const.wet_c] - 286.0).abs().max().item())
    s_err = float((ogcm_const.sal[ogcm_const.wet_c] - 34.7).abs().max().item())
    assert t_err < 1.0e-3, f"Constant T drift={t_err}"
    assert s_err < 1.0e-3, f"Constant S drift={s_err}"

    # 2. Exact global heat & salt conservation with 3D stratification, GM bolus, Redi & wind stress
    # Inject a salty tropical anomaly & cold polar anomaly (without surface freezing)
    ogcm.sal[0] = torch.where(ogcm.wet_c[0], 34.7 + 1.0 * torch.cos(lat), ogcm.sal[0])
    h0, s0 = _total_heat_and_salt(ogcm, mesh)
    for _ in range(15):
        step_ogcm_voronoi(
            ogcm, sfc, mesh, tau_x, tau_y, zeros,
            zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False,
        )
    h1, s1 = _total_heat_and_salt(ogcm, mesh)
    rel_h = abs(h1 - h0) / abs(h0)
    rel_s = abs(s1 - s0) / abs(s0)
    assert rel_h < 2.0e-5, f"Heat conservation rel_err={rel_h:.3e}"
    assert rel_s < 2.0e-5, f"Salt conservation rel_err={rel_s:.3e}"
    assert float(ogcm.u_edge.abs().max().item()) > 1.0e-3
    assert torch.isfinite(ogcm.temp).all() and torch.isfinite(ogcm.sal).all()


def test_nonlinear_eos_and_polar_convective_adjustment() -> None:
    # Haline contraction > thermal expansion at freezing polar T (Roquet et al. 2015)
    t_cold = torch.tensor([272.0])
    t_warm = torch.tensor([295.0])
    s_lo = torch.tensor([34.0])
    s_hi = torch.tensor([35.0])
    z0 = torch.tensor([100.0])
    assert float(compute_seawater_density_anomaly(t_cold, s_hi, z0)) > float(compute_seawater_density_anomaly(t_cold, s_lo, z0))
    assert float(compute_seawater_density_anomaly(t_cold, s_lo, z0)) > float(compute_seawater_density_anomaly(t_warm, s_lo, z0))

    # Brine-driven surface densification triggers convective adjustment to deeper layers
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    ogcm = init_ogcm_state(sfc, mesh, nz_ocn=10)
    # Create unstable heavy/salty surface cap at ocean cell 0
    ocn_idx = int(torch.nonzero(ogcm.wet_c[0], as_tuple=False)[0].item())
    ogcm.temp[:, ocn_idx] = 276.0
    ogcm.sal[:, ocn_idx] = S_REF_PSU
    sfc.t_ocean_ml[ocn_idx] = 273.0
    ogcm.sal[0, ocn_idx] = 36.5  # dense brine cap -> drho/dz < 0
    s_deep_before = float(ogcm.sal[1, ocn_idx].item())
    zeros = torch.zeros(mesh.n_cells)
    step_ogcm_voronoi(ogcm, sfc, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False)
    s_deep_after = float(ogcm.sal[1, ocn_idx].item())
    assert s_deep_after > s_deep_before + 0.05, f"Convective adjustment failed: {s_deep_before} -> {s_deep_after}"


def test_dynamic_seaice_drift_conservation() -> None:
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    is_ocean = sfc.surf_type == SURF_OCEAN
    # Place polar sea ice + snow pack on ocean cells with |lat| > 55 deg
    polar = is_ocean & (mesh.lat_cell.abs() > 0.95)
    sfc.m_seaice = torch.where(polar, torch.full_like(sfc.m_seaice, 1200.0), torch.zeros_like(sfc.m_seaice))
    sfc.swe_seaice = torch.where(polar, torch.full_like(sfc.swe_seaice, 150.0), torch.zeros_like(sfc.swe_seaice))

    m0 = float((sfc.m_seaice * mesh.area_cell).sum().item())
    s0 = float((sfc.swe_seaice * mesh.area_cell).sum().item())
    tau_x = torch.full((mesh.n_cells,), -0.15)
    tau_y = torch.full((mesh.n_cells,), 0.08)
    u_ocn = torch.full((mesh.n_edges,), 0.05)

    for _ in range(10):
        step_seaice_drift_voronoi(sfc, mesh, tau_x, tau_y, u_ocn, dt=720.0)

    m1 = float((sfc.m_seaice * mesh.area_cell).sum().item())
    s1 = float((sfc.swe_seaice * mesh.area_cell).sum().item())
    assert abs(m1 - m0) / m0 < 1.0e-5, f"Sea-ice mass drift: {m0} -> {m1}"
    assert abs(s1 - s0) / s0 < 1.0e-5, f"Sea-ice snow drift: {s0} -> {s1}"
    assert float(sfc.m_seaice[~is_ocean].abs().max().item()) == 0.0


def test_selectable_ocean_modes_cpu_and_mps() -> None:
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        for mode in ("ogcm_3d", "ekman_sw"):
            cfg = AOBAConfig(
                glevel=1, nz=12, z_top=24000.0, crm_nx=4, crm_ny=1,
                use_earth2014=False, ocean_mode=mode, ocean_nz=8,
                rad_scheme="ckd", enable_nonoro_gwd=False,
            )
            m = init_aoba_model(cfg, device=dev)
            if mode == "ogcm_3d":
                assert m.ogcm is not None and m.ogcm.nz_ocn == 8
            else:
                assert m.ogcm is None
            for _ in range(3):
                d = step_aoba_model(m)
            assert torch.isfinite(torch.tensor(d.global_mean_t_sfc_k))
            if mode == "ogcm_3d":
                assert torch.isfinite(m.ogcm.temp).all() and torch.isfinite(m.ogcm.sal).all()
