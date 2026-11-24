#!/usr/bin/env python3
"""Unit and integration tests for A5 Map-Free Natural Aerosols (Sea Salt, Marine DMS SO4, Wildfire BC/OC)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.aerosols import compute_seasalt_and_dms_emissions, init_natural_aerosol_state, step_natural_aerosols_column
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.surface import SURF_LAND, SURF_OCEAN


def test_natural_aerosols_physics_and_full_group_a_coupled() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Verify Jaegle (2011) sea-salt SST/wind scaling & Kloster (2006) DMS->SO4 Chl coupling
    st = torch.tensor([SURF_OCEAN, SURF_OCEAN, SURF_LAND], device=dev, dtype=torch.int64)
    sst = torch.tensor([298.15, 275.15, 290.0], device=dev)  # Warm tropics vs cold subpolar
    u10 = torch.tensor([10.0, 10.0, 10.0], device=dev)
    f_ice = torch.tensor([0.0, 0.0, 0.0], device=dev)
    chl = torch.tensor([0.1, 2.5, 0.0], device=dev)  # Subpolar bloom has much higher Chl -> higher DMS
    emis_ss, emis_so4 = compute_seasalt_and_dms_emissions(st, sst, u10, f_ice, chl_sfc_mg_m3=chl)
    assert float(emis_ss[0]) > float(emis_ss[1]) > 0.0, "Jaegle (2011): warmer SST must emit more sea salt"
    assert float(emis_so4[1]) > float(emis_so4[0]) > 0.0, "High-Chl subpolar ocean must emit more DMS->SO4"
    assert float(emis_ss[2]) == 0.0 and float(emis_so4[2]) == 0.0, "Land must have zero marine emissions"
    print("  [PASS] Jaegle (2011) sea-salt SST scaling & Kloster (2006) FlexPFT-Chl DMS coupling")

    # 2. Full Group A (A1+A2+A3+A4+A5+A6 ALL ON simultaneously!) coupled test on MPS!
    cfg = AOBAConfig(
        glevel=3, nz=18, dt_host=720.0, crm_nx=1, crm_ny=1,
        ocean_mode="ogcm_3d",
        enable_river_routing=True,
        enable_land_carbon=True,
        enable_seaice_evp_itd=True,
        enable_ogcm_fct_zstar=True,
        enable_natural_aerosols=True,
        enable_ocean_bgc=True,
    )
    model = init_aoba_model(cfg, device=dev)
    for _ in range(5):
        step_aoba_model(model)

    aero = model.aerosols
    assert aero is not None
    assert torch.isfinite(aero.q_ss).all() and (aero.q_ss >= 0.0).all()
    assert torch.isfinite(aero.q_so4).all() and (aero.q_so4 >= 0.0).all()
    assert torch.isfinite(aero.q_bc).all() and torch.isfinite(aero.q_oc).all()
    assert float(aero.aod_ss.max()) > 0.0 and float(aero.aod_so4.max()) > 0.0
    print(
        f"  [PASS] ALL Group A (A1+A2+A3+A4+A5+A6) simultaneously coupled on {dev}: "
        f"max AOD_ss={float(aero.aod_ss.max()):.4f}, max AOD_so4={float(aero.aod_so4.max()):.5f}, "
        f"max CCN={float(aero.ccn_aero_m3.max())*1e-6:.1f} cm^-3"
    )


if __name__ == "__main__":
    print("Running A5 Natural Aerosols & Full Group A (A1-A6) Integration tests...")
    test_natural_aerosols_physics_and_full_group_a_coupled()
    print("ALL A5 NATURAL AEROSOLS & FULL GROUP A TESTS PASSED!")
