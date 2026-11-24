#!/usr/bin/env python3
"""Unit and integration tests for A2 Map-Free Terrestrial Carbon Cycle & Autonomous Wildfire."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.land_carbon import diagnose_land_carbon_metrics, init_land_carbon_state, step_land_carbon_and_fire
from aoba.surface import SURF_LAND


def test_exact_carbon_conservation_and_permafrost_and_fire() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 3 idealized land columns: [0] tropical wet forest, [1] dry savanna (fire-prone), [2] frozen permafrost tundra
    surf_type = torch.tensor([SURF_LAND, SURF_LAND, SURF_LAND], device=dev, dtype=torch.int64)
    lai = torch.tensor([5.0, 2.0, 1.0], device=dev)
    hc = torch.tensor([25.0, 3.0, 0.5], device=dev)
    area = torch.tensor([1.0e12, 1.0e12, 1.0e12], device=dev)
    lc = init_land_carbon_state(surf_type, lai, hc, device=dev)

    t_soil = torch.tensor([
        [300.0, 305.0, 265.0],
        [299.0, 303.0, 264.0],
        [298.0, 300.0, 263.0],
        [297.0, 298.0, 262.0],
        [296.0, 296.0, 261.0],
    ], device=dev)
    w_liq = torch.tensor([
        [15.0, 4.0, 0.5],
        [45.0, 12.0, 1.0],
        [120.0, 35.0, 2.0],
        [240.0, 70.0, 4.0],
        [480.0, 140.0, 8.0],
    ], device=dev)
    w_ice = torch.tensor([
        [0.0, 0.0, 18.0],
        [0.0, 0.0, 50.0],
        [0.0, 0.0, 130.0],
        [0.0, 0.0, 260.0],
        [0.0, 0.0, 500.0],
    ], device=dev)
    gpp = torch.tensor([12.0, 4.0, 1.0], device=dev)
    rleaf = torch.tensor([1.0, 0.4, 0.1], device=dev)
    f_snow = torch.tensor([0.0, 0.0, 0.9], device=dev)
    t_sfc = torch.tensor([300.0, 308.0, 260.0], device=dev)
    vpd = torch.tensor([600.0, 3200.0, 100.0], device=dev)
    u10 = torch.tensor([3.0, 7.0, 4.0], device=dev)
    precip = torch.tensor([8.0e-5, 0.0, 1.0e-5], device=dev)

    m0 = diagnose_land_carbon_metrics(lc, area)
    dt = 3600.0
    cum_nbp_pgc = 0.0
    for _ in range(30 * 24):
        step_land_carbon_and_fire(
            lc, surf_type, gpp, rleaf, lai, hc, t_soil, w_liq, w_ice, f_snow, t_sfc, vpd, u10, precip, dt
        )
        cum_nbp_pgc += float((lc.nbp_kgc_m2s.detach().cpu().double() * area.cpu().double()).sum().item() * dt * 1.0e-12)
    m1 = diagnose_land_carbon_metrics(lc, area)

    # 1. Exact carbon conservation: d(C_total) == integral(NBP dt)
    dc_pgc = m1["c_land_total_pgc"] - m0["c_land_total_pgc"]
    rel_err = abs(dc_pgc - cum_nbp_pgc) / max(abs(m0["c_land_total_pgc"]), 1.0)
    assert rel_err < 1.0e-6, f"Land carbon conservation error: dC={dc_pgc:.6e} vs int(NBP)={cum_nbp_pgc:.6e} (rel={rel_err:.2e})"

    # 2. Permafrost stabilization: frozen tundra Rh must be << warm tropical forest Rh
    rh_cpu = lc.rh_kgc_m2s.detach().cpu()
    assert float(rh_cpu[2]) < 0.05 * float(rh_cpu[0]), f"Permafrost failed to suppress Rh: {rh_cpu.tolist()}"

    # 3. Autonomous wildfire: dry hot savanna must burn & emit BC/OC while wet rainforest & snowy tundra stay zero
    fire_cpu = lc.fire_c_kgc_m2s.detach().cpu()
    assert float(fire_cpu[1]) > 0.0 and float(lc.fire_bc_kg_m2s.detach().cpu()[1]) > 0.0
    assert float(fire_cpu[0]) == 0.0 and float(fire_cpu[2]) == 0.0
    print(f"  [PASS] 30-day A2 carbon conservation on {dev}: rel_err={rel_err:.2e}, Rh_tundra/Rh_trop={float(rh_cpu[2]/rh_cpu[0]):.4f}, savanna fire ACTIVE")

    # 4. Coupled step_aoba_model smoke test
    cfg = AOBAConfig(glevel=3, nz=18, dt_host=720.0, crm_nx=1, crm_ny=1, ocean_mode="ogcm_3d", enable_land_carbon=True)
    model = init_aoba_model(cfg, device=dev)
    step_aoba_model(model)
    mc = diagnose_land_carbon_metrics(model.land_carbon, model.mesh.area_cell)
    assert math.isfinite(mc["c_land_total_pgc"]) and mc["c_land_total_pgc"] > 100.0
    print(f"  [PASS] Coupled A2 on {dev}: C_land={mc['c_land_total_pgc']:.1f} PgC (veg={mc['c_veg_pgc']:.1f}, soil={mc['c_soil_pgc']:.1f})")


if __name__ == "__main__":
    print("Running A2 Map-Free Terrestrial Carbon & Autonomous Wildfire tests...")
    test_exact_carbon_conservation_and_permafrost_and_fire()
    print("ALL A2 LAND CARBON & FIRE TESTS PASSED!")
