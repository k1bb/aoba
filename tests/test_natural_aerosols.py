#!/usr/bin/env python3
"""Unit and integration tests for A5 Map-Free Natural Aerosols (Sea Salt, Marine DMS SO4, Wildfire BC/OC)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.aerosols import (
    compute_aerosol_radiative_optics,
    compute_fire_plume_top_m,
    compute_hygroscopic_growth,
    compute_seasalt_and_dms_emissions,
    init_natural_aerosol_state,
    step_natural_aerosols_column,
)
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.dust import DUST_COARSE_EMIS_FRAC, DUST_FINE_EMIS_FRAC, step_dust_two_mode_column
from aoba.surface import SURF_LAND, SURF_OCEAN


def test_hygroscopic_optics_fire_plume_and_two_mode_dust() -> None:
    """Verify f(RH) optics, Byram fire plume lofting (CPU vs MPS parity), and 2-mode fine/coarse dust."""
    nz, nc = 16, 4
    dz = torch.full((nz, nc), 250.0, dtype=torch.float32)  # 0..4000 m
    z_c = torch.cumsum(dz, dim=0) - 0.5 * dz
    rho = torch.exp(-z_c / 8500.0) * 1.2

    # 1. Hygroscopic optical growth f(RH): wet SS/SO4 extinction >> dry, BC unchanged, dry absorption invariant
    aero = init_natural_aerosol_state(nz, nc, torch.device("cpu"))
    aero.q_ss.fill_(1.0e-8)
    aero.q_so4.fill_(1.0e-8)
    aero.q_bc.fill_(2.0e-9)
    aero.q_oc.fill_(1.0e-8)
    rh_dry = torch.zeros((nz, nc), dtype=torch.float32)
    rh_wet = torch.full((nz, nc), 0.85, dtype=torch.float32)
    tau_d, scat_d, asym_d, lw_d = compute_aerosol_radiative_optics(aero, rho, dz, rh_3d=rh_dry)
    tau_w, scat_w, asym_w, lw_w = compute_aerosol_radiative_optics(aero, rho, dz, rh_3d=rh_wet)
    assert float(tau_w.sum()) > 1.7 * float(tau_d.sum()), "Expected strong hygroscopic SW extinction growth at RH=85%"
    abs_diff = float(((tau_w - scat_w) - (tau_d - scat_d)).abs().max())
    assert abs_diff < 1.0e-6, f"Water uptake must not alter dry absorption: diff={abs_diff}"
    f_ss, f_so4, f_bc, f_oc = compute_hygroscopic_growth(rh_wet)
    assert float(f_ss[0, 0]) > 2.5 and float(f_so4[0, 0]) > 2.0
    assert float(f_ss[0, 0]) > float(f_so4[0, 0]) > float(f_oc[0, 0]) > float(f_bc[0, 0]) == 1.0

    # 2. Wildfire buoyant smoke plume height & CPU-vs-MPS fused kernel parity
    ib = torch.tensor([0.0, 5.0, 1600.0, 10000.0], dtype=torch.float32)  # none, weak, savanna, crown fire
    zp = compute_fire_plume_top_m(ib)
    assert float(zp[0]) == 0.0 and float(zp[1]) == 0.0
    assert 1000.0 < float(zp[2]) < 1400.0 and 2300.0 < float(zp[3]) < 2700.0

    st = torch.tensor([SURF_OCEAN, SURF_LAND, SURF_LAND, SURF_LAND], dtype=torch.int64)
    sst = torch.full((nc,), 295.0)
    u10 = torch.tensor([12.0, 4.0, 8.0, 10.0])
    fice = torch.zeros(nc)
    pr = torch.zeros(nc)
    fbc = torch.tensor([0.0, 1.0e-9, 1.0e-9, 1.0e-9])
    foc = torch.tensor([0.0, 5.0e-9, 5.0e-9, 5.0e-9])
    fsoa = torch.tensor([0.0, 2.0e-9, 2.0e-9, 2.0e-9])

    a_cpu = init_natural_aerosol_state(nz, nc, torch.device("cpu"))
    step_natural_aerosols_column(
        a_cpu, rho, dz, st, sst, u10, fice, pr, dt=600.0,
        fire_bc_kg_m2s=fbc, fire_oc_kg_m2s=foc, soa_oc_kg_m2s=fsoa,
        fire_intensity_kw_m=ib, rh_3d=rh_wet,
    )
    # Crown fire (col 3, z_plume=2500m) must inject BC into layer k=8 (z~2125 m), unlike weak fire (col 1)
    assert float(a_cpu.q_bc[8, 3]) > 1.0e-12 and float(a_cpu.q_bc[8, 1]) == 0.0
    assert float(a_cpu.q_oc[8, 3]) > 1.0e-12 and float(a_cpu.q_oc[8, 1]) == 0.0

    if torch.backends.mps.is_available():
        mps = torch.device("mps")
        a_mps = init_natural_aerosol_state(nz, nc, mps)
        step_natural_aerosols_column(
            a_mps, rho.to(mps), dz.to(mps), st.to(mps), sst.to(mps), u10.to(mps), fice.to(mps), pr.to(mps), dt=600.0,
            fire_bc_kg_m2s=fbc.to(mps), fire_oc_kg_m2s=foc.to(mps), soa_oc_kg_m2s=fsoa.to(mps),
            fire_intensity_kw_m=ib.to(mps), rh_3d=rh_wet.to(mps),
        )
        rel_bc = float((a_cpu.q_bc - a_mps.q_bc.cpu()).abs().max() / torch.clamp(a_cpu.q_bc.abs().max(), min=1e-20))
        rel_oc = float((a_cpu.q_oc - a_mps.q_oc.cpu()).abs().max() / torch.clamp(a_cpu.q_oc.abs().max(), min=1e-20))
        rel_aod = float((a_cpu.aod_ss - a_mps.aod_ss.cpu()).abs().max() / torch.clamp(a_cpu.aod_ss.abs().max(), min=1e-20))
        assert rel_bc < 1.0e-5 and rel_oc < 1.0e-5 and rel_aod < 1.0e-5, (
            f"CPU vs MPS aerosol parity drift: rel_bc={rel_bc:.2e}, rel_oc={rel_oc:.2e}, rel_aod={rel_aod:.2e}"
        )

    # 3. 2-Mode Mineral Dust (fine 0.1-2.5 um vs coarse 2.5-10 um): fine mode persists far longer than coarse
    assert abs((DUST_FINE_EMIS_FRAC + DUST_COARSE_EMIS_FRAC) - 1.0) < 1.0e-6
    qf = torch.zeros((nz, nc), dtype=torch.float32)
    qc = torch.zeros((nz, nc), dtype=torch.float32)
    qf[:4] = 1.0e-7  # Initial 1 km PBL dust layer
    qc[:4] = 1.0e-7
    f_zero = torch.zeros(nc)
    u_st = torch.full((nc,), 0.3)
    for _ in range(96):  # 48 h settling with zero emission
        qf, qc, qt = step_dust_two_mode_column(qf, qc, rho, dz, z_c, f_zero, pr, u_st, dt=1800.0)
    assert float((rho * dz * qf).sum()) > 2.0 * float((rho * dz * qc).sum()), (
        "Fine dust (0.1-2.5 um) must persist longer in suspension than coarse dust (2.5-10 um)"
    )
    assert float((qt - (qf + qc)).abs().max()) < 1.0e-7
    print("  [PASS] Hygroscopic f(RH), Byram fire plume lofting (CPU/MPS parity), & 2-mode fine/coarse dust")


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
    test_hygroscopic_optics_fire_plume_and_two_mode_dust()
    test_natural_aerosols_physics_and_full_group_a_coupled()
    print("ALL A5 NATURAL AEROSOLS & FULL GROUP A TESTS PASSED!")
