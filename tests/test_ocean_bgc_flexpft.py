#!/usr/bin/env python3
"""Unit and integration tests for FlexPFT marine biogeochemistry & OMIP/mocsy 2.0 carbonate solver."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.ocean_bgc import (
    _lambert_w0_exp,
    compute_flexpft_rates,
    diagnose_ocean_bgc_metrics,
    init_ocean_bgc_state,
    solve_surface_carbonate_mocsy,
)
from aoba.ocean_ogcm import step_ogcm_voronoi


def test_lambert_w0_exp_overflow_free() -> None:
    ys = torch.tensor([-20.0, -5.0, 0.0, 0.5, 1.0, 2.0, 10.0, 80.0, 500.0], dtype=torch.float32)
    ws = _lambert_w0_exp(ys)
    assert torch.isfinite(ws).all() and (ws > 0.0).all()
    resid = (ws + torch.log(ws) - ys).abs()
    assert float(resid.max().item()) < 2.0e-4, f"Lambert W0 residual too large: {resid}"
    print("  [PASS] Lambert W0(exp(y)) overflow-free float32 accuracy")


def test_flexpft_photoacclimation_and_stoichiometry() -> None:
    # 1. Photoacclimation: dim subsurface light (15 W/m^2) should have HIGHER Chl:C than bright surface (200 W/m^2)
    no3 = torch.tensor([[10.0, 10.0, 10.0, 0.15, 20.0]], dtype=torch.float32)
    par = torch.tensor([[0.0, 15.0, 200.0, 100.0, 100.0]], dtype=torch.float32)
    temp = torch.full_like(no3, 293.15)
    wet = torch.ones_like(no3)
    mu, q, theta_hat, f_v, _ = compute_flexpft_rates(no3, par, temp, wet)

    assert float(theta_hat[0, 0]) == 0.0 and float(mu[0, 0]) == 0.0, "Dark layer must have zero growth/Chl"
    assert float(theta_hat[0, 1]) > float(theta_hat[0, 2]) > 0.0, (
        f"Expected higher Chl:C in dim light (SCM): dim={theta_hat[0,1]:.3f}, bright={theta_hat[0,2]:.3f}"
    )
    # 2. Flexible C:N stoichiometry: oligotrophic (0.15 mmol/m^3) C:N > nutrient-replete (20 mmol/m^3) C:N
    cn_oligo = 1.0 / float(q[0, 3])
    cn_replete = 1.0 / float(q[0, 4])
    assert cn_oligo > cn_replete + 2.0, f"Expected higher C:N under N limitation: {cn_oligo:.2f} vs {cn_replete:.2f}"
    assert 4.0 < cn_replete < 11.0 and 10.0 < cn_oligo < 35.0
    print(f"  [PASS] FlexPFT photoacclimation (theta_hat dim={theta_hat[0,1]:.3f} > bright={theta_hat[0,2]:.3f}) & C:N (oligo={cn_oligo:.1f} > replete={cn_replete:.1f})")


def test_mocsy_carbonate_benchmark() -> None:
    # Standard subtropical surface seawater: T=25 C, S=35 psu, DIC=2050 mmol/m^3 (~2000 umol/kg), Alk=2350 mmol/m^3 (~2293 umol/kg)
    dic = torch.tensor([2050.0, 2050.0], dtype=torch.float32)
    alk = torch.tensor([2350.0, 2350.0], dtype=torch.float32)
    temp = torch.tensor([298.15, 288.15], dtype=torch.float32)  # 25 C vs 15 C
    sal = torch.tensor([35.0, 35.0], dtype=torch.float32)
    wet = torch.ones_like(dic)

    pco2, ph, co2_star = solve_surface_carbonate_mocsy(dic, alk, temp, sal, wet)
    assert torch.isfinite(pco2).all() and torch.isfinite(ph).all()
    assert 7.95 < float(ph[0]) < 8.20, f"Unexpected pH at 25C: {ph[0]:.3f}"
    assert 280.0 < float(pco2[0]) < 460.0, f"Unexpected pCO2 at 25C: {pco2[0]:.1f} uatm"
    # Takahashi thermodynamic rule: +10 C warming raises pCO2 by ~40-60% (~4.2%/C)
    ratio = float(pco2[0] / pco2[1])
    assert 1.35 < ratio < 1.65, f"Unexpected +10C pCO2 ratio: {ratio:.3f}"
    print(f"  [PASS] mocsy 2.0 carbonate equilibrium: 25C pH={ph[0]:.3f}, pCO2={pco2[0]:.1f} uatm (+10C factor={ratio:.2f}x)")


def test_3d_bgc_conservation_and_coupled_mps() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = AOBAConfig(
        glevel=3, nz=18, dt_host=720.0, crm_nx=1, crm_ny=1,
        ocean_mode="ogcm_3d", enable_ocean_bgc=True,
    )
    model = init_aoba_model(cfg, device=dev)
    assert model.ogcm is not None and model.ogcm.bgc is not None
    ogcm, sfc, mesh = model.ogcm, model.sfc, model.mesh
    bgc = ogcm.bgc

    # 1. Exact 3D Nitrogen conservation under 5 days of 3D OGCM dynamics + FlexPFT + sinking (zero air-sea flux)
    m0 = diagnose_ocean_bgc_metrics(bgc, ogcm, mesh)
    lat = mesh.lat_cell
    tau_x = -(0.10 * (-torch.cos(3.0 * lat)))
    z0 = torch.zeros_like(lat)
    par = 120.0 * torch.clamp(torch.cos(lat), min=0.0)
    f_ice = torch.zeros_like(lat)

    for _ in range(5 * 24):
        step_ogcm_voronoi(
            ogcm, sfc, mesh, tau_x, z0, f_ice, z0, z0, z0, z0,
            dt=3600.0, enable_seaice_drift=False, sw_par_ocn=par, u10_ocn=z0, co2_ppmv=284.3,
        )
    m1 = diagnose_ocean_bgc_metrics(bgc, ogcm, mesh)
    rel_n_err = abs(m1["ocean_tot_n_mmol"] - m0["ocean_tot_n_mmol"]) / max(abs(m0["ocean_tot_n_mmol"]), 1.0)
    assert rel_n_err < 1.0e-5, f"Total ocean N conservation drift too large: {rel_n_err:.3e}"
    assert m1["ocean_npp_pgc_yr"] > 5.0 and m1["ocean_chl_sfc_mg_m3"] > 0.01
    print(
        f"  [PASS] 5-day 3D FlexPFT on {dev}: N rel_err={rel_n_err:.2e}, "
        f"NPP={m1['ocean_npp_pgc_yr']:.1f} PgC/yr, EPC100={m1['ocean_epc100_pgc_yr']:.1f} PgC/yr, "
        f"Chl={m1['ocean_chl_sfc_mg_m3']:.3f} mg/m3, pH={m1['ocean_ph_sfc']:.3f}, pCO2={m1['ocean_spco2_uatm']:.1f} uatm"
    )

    # 2. Full coupled step_aoba_model with enable_ocean_bgc=True & overhead comparison
    cfg_on = AOBAConfig(glevel=3, nz=24, dt_host=720.0, crm_nx=4, crm_ny=1, rad_scheme="ckd", ocean_mode="ogcm_3d", enable_ocean_bgc=True)
    cfg_off = AOBAConfig(glevel=3, nz=24, dt_host=720.0, crm_nx=4, crm_ny=1, rad_scheme="ckd", ocean_mode="ogcm_3d", enable_ocean_bgc=False)
    m_on = init_aoba_model(cfg_on, device=dev)
    m_off = init_aoba_model(cfg_off, device=dev)
    # Warmup
    step_aoba_model(m_on)
    step_aoba_model(m_off)
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        step_aoba_model(m_off)
    if dev.type == "mps":
        torch.mps.synchronize()
    t_off = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(5):
        step_aoba_model(m_on)
    if dev.type == "mps":
        torch.mps.synchronize()
    t_on = time.perf_counter() - t0
    diag_bgc = diagnose_ocean_bgc_metrics(m_on.ogcm.bgc, m_on.ogcm, m_on.mesh)
    assert math.isfinite(diag_bgc["ocean_fgco2_pgc_yr"]) and math.isfinite(diag_bgc["ocean_npp_pgc_yr"])
    overhead_pct = (t_on / max(t_off, 1e-6) - 1.0) * 100.0
    print(
        f"  [PASS] Coupled FlexPFT on {dev}: fgCO2={diag_bgc['ocean_fgco2_pgc_yr']:+.2f} PgC/yr, "
        f"NPP={diag_bgc['ocean_npp_pgc_yr']:.1f} PgC/yr, overhead={overhead_pct:+.1f}% ({t_off/5*1000:.1f} -> {t_on/5*1000:.1f} ms/step)"
    )


if __name__ == "__main__":
    print("Running FlexPFT Marine BGC & OMIP/mocsy 2.0 tests...")
    test_lambert_w0_exp_overflow_free()
    test_flexpft_photoacclimation_and_stoichiometry()
    test_mocsy_carbonate_benchmark()
    test_3d_bgc_conservation_and_coupled_mps()
    print("ALL FLEXPFT MARINE BGC TESTS PASSED!")
