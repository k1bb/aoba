#!/usr/bin/env python3
"""Unit and integration tests for A4 FESOM2-style 3D OGCM Zalesak FCT Advection & Exact z* Coord."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.ocean_ogcm import step_ogcm_voronoi


def _zstar_global_means(ogcm, mesh) -> tuple[float, float]:
    area = mesh.area_cell.detach().cpu().double()
    h_c = ogcm.h_c.detach().cpu().double()
    h_col = torch.clamp(ogcm.h_col.detach().cpu().double(), min=1.0)
    eta = ogcm.eta.detach().cpu().double()
    scale = torch.clamp(1.0 + eta / h_col, min=0.5, max=1.5)
    vol = (h_c * scale.unsqueeze(0)) * area.unsqueeze(0)
    v_tot = float(vol.sum().item())
    t64 = ogcm.temp.detach().cpu().double() + ogcm.t_comp.detach().cpu().double()
    s64 = ogcm.sal.detach().cpu().double() + ogcm.s_comp.detach().cpu().double()
    return float((vol * t64).sum().item() / v_tot), float((vol * s64).sum().item() / v_tot)


def test_fct_zstar_conservation_and_monotonicity() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = AOBAConfig(
        glevel=3, nz=18, dt_host=720.0, crm_nx=1, crm_ny=1,
        ocean_mode="ogcm_3d", enable_ogcm_fct_zstar=True,
    )
    model = init_aoba_model(cfg, device=dev)
    ogcm, sfc, mesh = model.ogcm, model.sfc, model.mesh
    assert ogcm is not None

    t0_glb, s0_glb = _zstar_global_means(ogcm, mesh)
    t_min_0 = float(ogcm.temp[ogcm.wet_c].min().item())
    t_max_0 = float(ogcm.temp[ogcm.wet_c].max().item())

    lat = mesh.lat_cell
    tau_x = -(0.12 * (-torch.cos(3.0 * lat)))
    z0 = torch.zeros_like(lat)
    for _ in range(10 * 24):
        step_ogcm_voronoi(
            ogcm, sfc, mesh, tau_x, z0, z0, z0, z0, z0, z0,
            dt=3600.0, enable_seaice_drift=False, enable_fct_zstar=True,
        )

    t1_glb, s1_glb = _zstar_global_means(ogcm, mesh)
    t_min_1 = float(ogcm.temp[ogcm.wet_c].min().item())
    t_max_1 = float(ogcm.temp[ogcm.wet_c].max().item())

    dT_yr = (t1_glb - t0_glb) * (365.0 / 10.0)
    dS_yr = (s1_glb - s0_glb) * (365.0 / 10.0)
    assert abs(dT_yr) < 1.0e-3, f"FCT z* temperature drift too high: {dT_yr:.3e} K/yr"
    assert abs(dS_yr) < 1.0e-3, f"FCT z* salinity drift too high: {dS_yr:.3e} psu/yr"
    # Monotonicity check: FCT must NOT create spurious overshoots/undershoots beyond initial T range
    assert t_min_1 >= t_min_0 - 0.05 and t_max_1 <= t_max_0 + 0.05, (
        f"FCT monotonicity violated: [{t_min_1:.2f}, {t_max_1:.2f}] vs init [{t_min_0:.2f}, {t_max_0:.2f}]"
    )
    print(
        f"  [PASS] 10-day A4 FCT+z* on {dev}: dT/yr={dT_yr:+.2e} K/yr, dS/yr={dS_yr:+.2e} psu/yr, "
        f"monotone T in [{t_min_1:.2f}, {t_max_1:.2f}] K"
    )

    # Coupled step_aoba_model smoke test
    step_aoba_model(model)
    assert torch.isfinite(ogcm.temp).all() and torch.isfinite(ogcm.sal).all()
    print(f"  [PASS] Coupled step_aoba_model with enable_ogcm_fct_zstar=True on {dev}")


if __name__ == "__main__":
    print("Running A4 OGCM Zalesak FCT & Exact z* tests...")
    test_fct_zstar_conservation_and_monotonicity()
    print("ALL A4 OGCM FCT+z* TESTS PASSED!")
