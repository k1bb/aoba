"""Comprehensive verification suite for all 7 Universal ESM toggleable modules & current-relative wind."""

from __future__ import annotations

import math
import torch

from aoba.aerosols import compute_aerosol_radiative_optics, compute_bvoc_soa_emission, init_natural_aerosol_state
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.planetary_geochem import (
    compute_milankovitch_orbital_params,
    init_weathering_state,
    step_silicate_weathering,
)
from aoba.radiation_hd import N_LW_GPTS_HD, N_SW_GPTS_HD
from aoba.surface import SURF_LAND, SURF_OCEAN, compute_surface_albedos, init_surface_state, step_snow_impurity_darkening
from aoba.trace_gas_cycle import init_trace_gas_cycle_state, step_trace_gas_cycle


def test_snicar_snow_bc_darkening_and_bvoc_soa() -> None:
    """Verify BC/dust snow deposition darkens visible snow albedo and BVOC->SOA responds to LAI & T."""
    st = torch.tensor([SURF_LAND, SURF_LAND, SURF_OCEAN])
    sfc = init_surface_state(st, z_topo=torch.zeros(3), t_init=270.0)
    sfc.swe_land = torch.tensor([50.0, 50.0, 0.0])
    sfc.swe_max_season = torch.tensor([50.0, 50.0, 0.0])
    sfc.bio.lai = torch.tensor([0.1, 0.1, 0.0])
    sfc.bio.h_canopy = torch.tensor([0.1, 0.1, 0.0])
    mu0 = torch.tensor([0.6, 0.6, 0.6])
    a_clean = compute_surface_albedos(sfc, mu0)[1].clone()  # alb_vis_dif

    # Deposit BC onto cell 0 only, leave cell 1 clean
    bc_dep = torch.tensor([1.0e-10, 0.0, 0.0])
    for _ in range(100):
        step_snow_impurity_darkening(sfc, bc_dep, None, torch.zeros(3), 3600.0)
    a_dirty = compute_surface_albedos(sfc, mu0)[1]
    assert float(a_dirty[0].item()) < float(a_clean[0].item()) - 0.15
    assert abs(float(a_dirty[1].item()) - float(a_clean[1].item())) < 1.0e-5

    # BVOC->SOA: tropical forest >> cold desert >> ocean (0)
    soa = compute_bvoc_soa_emission(
        torch.tensor([SURF_LAND, SURF_LAND, SURF_OCEAN]),
        torch.tensor([5.0, 0.1, 0.0]),
        torch.tensor([302.0, 275.0, 300.0]),
        torch.tensor([250.0, 100.0, 250.0]),
    )
    assert float(soa[0].item()) > 20.0 * float(soa[1].item()) and float(soa[2].item()) == 0.0
    print(f"[PASS] SNICAR snow darkening ({a_clean[0]:.3f}->{a_dirty[0]:.3f}) & BVOC-SOA ({soa[0]:.2e} kg/m2/s)")


def test_trace_gas_and_planetary_geochem() -> None:
    """Verify CH4/N2O cycles, silicate weathering negative CO2 feedback, and Milankovitch orbital bounds."""
    tg = init_trace_gas_cycle_state(0.808, 0.273)
    nc = 4
    area = torch.full((nc,), 5.1e14 / nc)
    st = torch.full((nc,), SURF_LAND)
    t_s = torch.full((nc,), 290.0)
    w_top = torch.full((nc,), 0.40)  # Wet anoxic soil
    rh = torch.full((nc,), 2.0e-8)
    for _ in range(10):
        step_trace_gas_cycle(tg, area, st, t_s, w_top, 86400.0 * 30.0, rh_kgc_m2s=rh, mean_tco_du=295.0)
    assert 0.1 < tg.ch4_ppmv < 10.0 and 0.1 < tg.n2o_ppmv < 1.5 and tg.ch4_emis_wetland_tgyr > 0.0

    # Silicate weathering thermostat: hot high-CO2 planet draws down CO2; cold snowball accumulates volcanic CO2
    ws_hot = init_weathering_state(1200.0, volcanic_co2_pgc_yr=0.03)
    ws_cold = init_weathering_state(280.0, volcanic_co2_pgc_yr=0.03)
    step_silicate_weathering(
        ws_hot, area, st, torch.full((nc,), 308.0), torch.full((nc,), 5.0e-5), torch.zeros(nc),
        86400.0 * 365.0, volcanic_co2_pgc_yr=0.03, accel_factor=1000.0,
    )
    step_silicate_weathering(
        ws_cold, area, st, torch.full((nc,), 260.0), torch.zeros(nc), torch.ones(nc),
        86400.0 * 365.0, volcanic_co2_pgc_yr=0.03, accel_factor=1000.0,
    )
    assert ws_hot.co2_ppmv < 1200.0 and ws_cold.co2_ppmv > 280.0

    # Milankovitch orbital elements at J2000 (0 kyr) and LGM (-21 kyr)
    e0, obl0, p0 = compute_milankovitch_orbital_params(0.0)
    e21, obl21, p21 = compute_milankovitch_orbital_params(-21.0)
    assert 0.0005 <= e0 <= 0.065 and 21.8 <= obl0 <= 24.8 and 0.0 <= p0 < 360.0
    assert abs(obl0 - obl21) > 0.2
    print(
        f"[PASS] TraceGas (CH4={tg.ch4_ppmv:.3f}, N2O={tg.n2o_ppmv:.3f}), "
        f"Weathering (hot {1200}->{ws_hot.co2_ppmv:.1f}, cold {280}->{ws_cold.co2_ppmv:.1f}), "
        f"Milankovitch (obl 0kyr={obl0:.2f} vs -21kyr={obl21:.2f} deg)"
    )


def test_universal_all_on_and_current_rel_wind_gating() -> None:
    """Verify ALL 7 switches ON (six_class_2m + ckd_hd + 3D TICA + O3 + CH4/N2O + weathering + Milankovitch + HiRes Ocean + rel-wind) on CPU & MPS."""
    assert N_LW_GPTS_HD == 72 and N_SW_GPTS_HD == 39 and (N_LW_GPTS_HD + N_SW_GPTS_HD) == 111

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
            rad_scheme="ckd_hd",
            enable_rad_3d_tica=True,
            microphysics_scheme="six_class_2m",
            o3_mode="prognostic",
            enable_land_carbon=True,
            enable_land_nitrogen=True,
            enable_lightning=True,
            enable_fire_veg_feedback=True,
            enable_ocean_bgc=True,
            enable_seaice_evp_itd=True,
            enable_natural_aerosols=True,
            enable_aerosol_rad_forcing=True,
            enable_snow_bc_darkening=True,
            enable_bgc_fe_lnox_coupling=True,
            enable_bvoc_soa=True,
            enable_prognostic_ch4_n2o=True,
            enable_current_rel_wind=True,
            enable_ogcm_scale_aware_gm=True,
            enable_ogcm_submeso=True,
            enable_ogcm_tidal_mixing=True,
            enable_silicate_weathering=True,
            weathering_accel_factor=100.0,
            enable_milankovitch_orbit=True,
            milankovitch_start_kyr=-21.0,
            use_earth2014=True,
            earth2014_stride=8,
        )
        m = init_aoba_model(cfg, device=dev_s)
        for _ in range(2):
            diag = step_aoba_model(m, compute_diagnostics=True, sync_host_surface=True)
            assert math.isfinite(diag.global_mean_t_sfc_k)
            assert math.isfinite(diag.global_mean_olr_wm2)
        assert m.crm.q_r is not None and m.crm.q_s is not None and m.crm.q_g is not None
        assert m.trace_gas is not None and m.weathering is not None and m.ozone is not None
        assert m.ogcm is not None and m.ogcm.bgc is not None and m.ogcm.bgc.fe_sfc is not None
        assert m.sfc_sub is not None and m.sfc_sub.bc_snow is not None
        print(
            f"[PASS] Universal ALL-ON ESM ({dev_s}): 6-class 2M + 111g CKD-HD + 3D TICA + O3 + "
            f"CH4/N2O + Weathering + Milankovitch + G2 Ocean + RelWind | Ts={diag.global_mean_t_sfc_k:.2f} K"
        )

    # Verify current-relative wind gating: affects 3D CRM (ny=2, nx=2), does NOT affect 2D SP (ny=1, nx=4)!
    def _run_tau_diff(ny: int, nx: int) -> float:
        base_kw = dict(
            glevel=1, nz=10, z_top=20000.0, z_sponge_base=15000.0, ocean_nz=4,
            crm_nx=nx, crm_ny=ny, dt_host=600.0, n_crm_dyn_substeps=6, rad_stride=1,
            rad_scheme="four_band", enable_land_carbon=False, enable_ocean_bgc=False,
            use_earth2014=True, earth2014_stride=8,
        )
        m_on = init_aoba_model(AOBAConfig(**base_kw, enable_current_rel_wind=True), device="cpu")
        m_off = init_aoba_model(AOBAConfig(**base_kw, enable_current_rel_wind=False), device="cpu")
        # Inject strong 1 m/s ocean surface current on both models
        assert m_on.ogcm is not None and m_off.ogcm is not None
        m_on.ogcm.u_edge[0] = 1.0 * m_on.ogcm.wet_e[0].float()
        m_off.ogcm.u_edge[0] = 1.0 * m_off.ogcm.wet_e[0].float()
        step_aoba_model(m_on, compute_diagnostics=False, sync_host_surface=True)
        step_aoba_model(m_off, compute_diagnostics=False, sync_host_surface=True)
        assert m_on.last_sfc_diag_sub is not None and m_off.last_sfc_diag_sub is not None
        return float((m_on.last_sfc_diag_sub.tau_x - m_off.last_sfc_diag_sub.tau_x).abs().max().item())

    diff_3d = _run_tau_diff(ny=2, nx=2)
    diff_2d = _run_tau_diff(ny=1, nx=4)
    assert diff_3d > 1.0e-4, f"Expected relative-wind effect in 3D CRM, got {diff_3d}"
    assert diff_2d == 0.0, f"Relative-wind must stay OFF in 2D SP (ny=1), got {diff_2d}"
    print(f"[PASS] Ocean-current relative wind gating: 3D CRM max|d_tau|={diff_3d:.4f} N/m^2, 2D SP={diff_2d:.4f}")


if __name__ == "__main__":
    test_snicar_snow_bc_darkening_and_bvoc_soa()
    test_trace_gas_and_planetary_geochem()
    test_universal_all_on_and_current_rel_wind_gating()
    print("ALL UNIVERSAL ESM SUITE TESTS PASSED!")
