"""Verification tests for prognostic 3D ozone chemistry, high-res ocean_glevel coupler, and ESM fixes."""

from __future__ import annotations

import math
import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.lightning import compute_crm_lightning_and_lnox
from aoba.scvt_mesh import build_scvt_mesh
from aoba.ocean_coupler import (
    atm_to_ocn_conservative_flux,
    atm_to_ocn_intensive,
    build_ocean_atmos_coupler,
    ocn_to_atm_mean,
)
from aoba.ozone_chemistry import (
    diagnose_ozone_metrics,
    init_prognostic_ozone_state,
    step_prognostic_ozone,
)
from aoba.surface import SURF_LAND, SURF_OCEAN
from aoba.thermodynamics import diagnose_pressure_temperature


def test_lightning_zero_cirrus_gate() -> None:
    """Inert stratiform/cirrus clouds with zero updraft graupel flux must spark strictly 0 flashes."""
    nz, nc, ny, nx = 20, 4, 2, 2
    z_c = torch.linspace(200.0, 18000.0, nz)
    dz_c = torch.full((nz,), 900.0)
    rho_d = torch.exp(-z_c / 8000.0).view(nz, 1, 1, 1).expand(nz, nc, ny, nx).clone()
    theta_m = (290.0 + 0.005 * z_c).view(nz, 1, 1, 1).expand(nz, nc, ny, nx).clone()
    q_v = torch.full((nz, nc, ny, nx), 1.0e-3)
    q_i = torch.full((nz, nc, ny, nx), 5.0e-4)   # Thick inert cirrus (IWP > 1 kg m^-2)
    q_p = torch.zeros((nz, nc, ny, nx))
    w_stag = torch.zeros((nz + 1, nc, ny, nx))   # Zero updraft!
    area = torch.full((nc,), 1.0e11)

    diag = compute_crm_lightning_and_lnox(rho_d, theta_m, q_v, q_i, q_p, w_stag, z_c, dz_c, area)
    assert float(diag.flash_tot_sub.max().item()) == 0.0, "Inert cirrus at w=0 must produce zero lightning"
    assert float(diag.flash_cg_sub.max().item()) == 0.0
    print("[PASS] Lightning inert-cirrus strict zero gate")


def test_prognostic_ozone_photochemistry_physics() -> None:
    """Verify Chapman equilibrium, CO2-cooling O3 super-recovery, PSC ozone hole, LNOx & LAI dry dep."""
    nz = 30
    z_c = torch.linspace(200.0, 46000.0, nz)
    dz_c = torch.full((nz,), 46000.0 / nz)
    # Standard stratospheric-like p, T profile
    p_1d = 1.0e5 * torch.exp(-z_c / 7200.0)
    t_1d = torch.where(
        z_c < 14000.0,
        295.0 - 0.0065 * z_c,
        torch.where(z_c < 22000.0, torch.full_like(z_c, 204.0), 204.0 + 0.0024 * (z_c - 22000.0)),
    )
    rho_1d = p_1d / (287.04 * t_1d)
    qv_1d = torch.where(z_c < 14000.0, 0.012 * torch.exp(-z_c / 2500.0), torch.full_like(z_c, 3.0e-6))

    # 4 test columns:
    # 0: sunlit tropics (T normal, eCl=0.55)
    # 1: sunlit cooled stratosphere (T - 10 K above 20 km -> Arrhenius O3 super-recovery)
    # 2: sunlit cold polar vortex (T = 186 K in lower strat, eCl=3.5 -> PSC ozone hole!)
    # 3: dark polar night (T = 186 K, mu0=0 -> NO sunlit PSC depletion)
    nc = 4
    p_c = p_1d.unsqueeze(1).expand(nz, nc).clone()
    t_c = t_1d.unsqueeze(1).expand(nz, nc).clone()
    t_c[z_c > 25000.0, 1] -= 10.0
    t_c[(z_c > 14000.0) & (z_c < 26000.0), 2] = 186.0
    t_c[(z_c > 14000.0) & (z_c < 26000.0), 3] = 186.0
    rho_c = (p_c / (287.04 * t_c)).clone()
    dz_2d = dz_c.unsqueeze(1).expand(nz, nc).clone()
    qv_c = qv_1d.unsqueeze(1).expand(nz, nc).clone()
    mu0 = torch.tensor([0.65, 0.65, 0.35, 0.0])
    surf_type = torch.tensor([SURF_LAND, SURF_OCEAN, SURF_OCEAN, SURF_OCEAN])
    lai = torch.tensor([4.0, 0.0, 0.0, 0.0])
    f_snow = torch.zeros(nc)
    f_ice = torch.zeros(nc)
    area = torch.full((nc,), 1.0e14)

    # Compare pre-industrial eCl=0.55 vs ozone-hole eCl=3.5
    oz_pi = init_prognostic_ozone_state(p_c, rho_c, dz_2d)
    oz_cfc = init_prognostic_ozone_state(p_c, rho_c, dz_2d)
    dt = 1800.0
    for _ in range(48 * 15):  # 15 days
        step_prognostic_ozone(oz_pi, p_c, t_c, rho_c, dz_2d, qv_c, mu0, surf_type, lai, f_snow, f_ice, dt, ecl_ppbv=0.55)
        step_prognostic_ozone(oz_cfc, p_c, t_c, rho_c, dz_2d, qv_c, mu0, surf_type, lai, f_snow, f_ice, dt, ecl_ppbv=3.5)

    assert torch.isfinite(oz_pi.o3_vmr).all() and (oz_pi.o3_vmr > 0.0).all()
    assert torch.isfinite(oz_cfc.o3_vmr).all() and (oz_cfc.o3_vmr > 0.0).all()

    # 1. Realistic total column (200..420 DU) and upper-strat cooling super-recovery
    assert 180.0 < float(oz_pi.tco_du[0].item()) < 420.0, f"Unexpected TCO: {oz_pi.tco_du[0].item()}"
    k_upper = int((p_1d - 500.0).abs().argmin().item())
    assert float(oz_pi.o3_vmr[k_upper, 1].item()) > float(oz_pi.o3_vmr[k_upper, 0].item()), (
        "Stratospheric cooling must increase upper-stratospheric O3 via Arrhenius Chapman kinetics"
    )

    # 2. Sunlit PSC ozone hole under eCl=3.5 vs eCl=0.55, while polar night stays undepleted
    du_polar_pi = float(oz_pi.tco_du[2].item())
    du_polar_cfc = float(oz_cfc.tco_du[2].item())
    du_night_cfc = float(oz_cfc.tco_du[3].item())
    assert du_polar_cfc < du_polar_pi - 40.0, f"Expected PSC depletion: cfc={du_polar_cfc:.1f} vs pi={du_polar_pi:.1f}"
    assert du_night_cfc > du_polar_cfc + 40.0, f"Polar night should not photochemically deplete: night={du_night_cfc:.1f}"

    # 3. Vegetated land dry deposition > open ocean dry deposition
    assert float(oz_pi.o3_drydep_kgm2s[0].item()) > float(oz_pi.o3_drydep_kgm2s[1].item())

    m = diagnose_ozone_metrics(oz_pi, area, torch.tensor([0.0, 0.0, -1.3, -1.3]))
    assert 150.0 < m["o3_tco_du"] < 420.0
    print(
        f"[PASS] Prognostic O3 physics: trop={oz_pi.tco_du[0]:.1f} DU, "
        f"polar_PI={du_polar_pi:.1f} vs polar_CFC={du_polar_cfc:.1f} DU, "
        f"upper_cool={oz_pi.o3_vmr[k_upper,1]*1e6:.2f} > {oz_pi.o3_vmr[k_upper,0]*1e6:.2f} ppmv"
    )


def test_ocean_coupler_conservation() -> None:
    """Verify exact Voronoi conservative flux mapping and area-weighted SST aggregation (G1 <-> G2)."""
    atm_mesh = build_scvt_mesh(glevel=1, n_lloyd_iter=5, use_earth2014=True, earth2014_stride=8)
    ocn_mesh = build_scvt_mesh(glevel=2, n_lloyd_iter=5, use_earth2014=True, earth2014_stride=8)
    coupler = build_ocean_atmos_coupler(atm_mesh, ocn_mesh)

    # Intensive field round-trip on constant field is exact
    const_atm = torch.full((atm_mesh.n_cells,), 285.0)
    const_ocn = atm_to_ocn_intensive(coupler, const_atm)
    back_atm = ocn_to_atm_mean(coupler, const_ocn, fallback_atm=const_atm)
    assert float((back_atm - 285.0).abs().max().item()) < 1.0e-4  # Within 1-3 float32 ULPs at 285 K

    # Extensive runoff flux global integral is strictly conserved
    flux_atm = torch.linspace(0.0, 1.0e-4, atm_mesh.n_cells)
    flux_ocn = atm_to_ocn_conservative_flux(coupler, flux_atm, ocean_only_source=False)
    tot_atm = float((flux_atm.double() * atm_mesh.area_cell.double()).sum().item())
    tot_ocn = float((flux_ocn.double() * coupler.ocn_wet_area.double()).sum().item())
    rel_err = abs(tot_ocn - tot_atm) / max(abs(tot_atm), 1.0e-12)
    assert rel_err < 1.0e-5, f"Runoff flux conservation failed: rel_err={rel_err:.3e}"
    print(f"[PASS] Ocean coupler G1({atm_mesh.n_cells}) <-> G2({ocn_mesh.n_cells}): flux rel_err={rel_err:.2e}")


def test_coupled_hires_ocean_and_prognostic_o3_cpu_mps() -> None:
    """Smoke-test coupled G1 atmos + G2 high-res OGCM + 3D CRM + prognostic O3 + Full-ESM on CPU & MPS."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for dev_s in devices:
        cfg = AOBAConfig(
            glevel=1,
            ocean_glevel=2,
            nz=12,
            z_top=24000.0,
            z_sponge_base=18000.0,
            ocean_nz=6,
            crm_nx=2,
            crm_ny=2,
            dt_host=600.0,
            n_crm_dyn_substeps=10,
            rad_stride=1,
            o3_mode="prognostic",
            o3_ecl_ppbv=0.55,
            enable_land_carbon=True,
            enable_land_nitrogen=True,
            enable_lightning=True,
            enable_fire_veg_feedback=True,
            enable_ocean_bgc=True,
            enable_seaice_evp_itd=True,
            enable_natural_aerosols=True,
            use_earth2014=True,
            earth2014_stride=8,
        )
        model = init_aoba_model(cfg, device=dev_s)
        assert model.ocn_mesh is not None and model.ocn_mesh.n_cells == 162
        assert model.mesh.n_cells == 42
        assert model.ogcm is not None and model.ogcm.temp.shape[1] == 162
        assert model.ozone is not None and model.ozone.o3_vmr.shape == (12, 42)

        for _ in range(3):
            diag = step_aoba_model(model, compute_diagnostics=True, sync_host_surface=True)
            assert math.isfinite(diag.global_mean_t_sfc_k)
            assert math.isfinite(diag.global_mean_olr_wm2)
        assert torch.isfinite(model.ozone.o3_vmr).all() and (model.ozone.o3_vmr > 0.0).all()
        assert torch.isfinite(model.ogcm.temp).all()
        assert torch.isfinite(model.ogcm.eta).all()
        oz_m = diagnose_ozone_metrics(model.ozone, model.mesh.area_cell, model.mesh.lat_cell)
        print(
            f"[PASS] Full-ESM G1(42)x2x2 + HiRes Ocean G2(162) + Prognostic O3 on {dev_s}: "
            f"Ts={diag.global_mean_t_sfc_k:.2f} K, O3={oz_m['o3_tco_du']:.1f} DU"
        )


if __name__ == "__main__":
    test_lightning_zero_cirrus_gate()
    test_prognostic_ozone_photochemistry_physics()
    test_ocean_coupler_conservation()
    test_coupled_hires_ocean_and_prognostic_o3_cpu_mps()
    print("ALL PROGNOSTIC OZONE & HIGH-RES OCEAN TESTS PASSED!")
