"""Verification tests for Batch 2 MAML x Ocean coupling:
1. MAML subgrid orography & microclimate downscaling (exact zero parent-cell mean dz, lapse-rate T contrast).
2. MAML dynamic wetlands/floodplains/lakes (flat alluvial basin floods > steep mountain ridge, CH4 coupling).
3. Coupled ocean waves H_s/T_p, Stokes-drift Langmuir KPP factor, & hurricane sea-spray enthalpy flux.
4. Full coupled Batch 2 integration on CPU and MPS.
"""

from __future__ import annotations

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.maml_land_subgrid import (
    diagnose_maml_wetland_area_mkm2,
    downscale_maml_microclimate,
    init_maml_subgrid_land_state,
    step_maml_wetlands_lakes,
)
from aoba.ocean_waves import (
    compute_sea_spray_enthalpy_fluxes,
    diagnose_ocean_wave_metrics,
    init_ocean_wave_state,
    step_ocean_waves,
)
from aoba.surface import SOIL_POROSITY, SURF_LAND


def test_maml_subgrid_topo_and_wetlands() -> None:
    nc, n_sub = 4, 16
    m = nc * n_sub
    st = torch.full((m,), SURF_LAND, dtype=torch.long)
    # Cell 0: flat alluvial floodplain (sigma=10 m); Cell 1: alpine ridge (sigma=600 m)
    sig = torch.full((nc, n_sub), 100.0)
    sig[0] = 10.0
    sig[1] = 600.0
    sig = sig.reshape(m)

    ml = init_maml_subgrid_land_state(st, sig, nc, n_sub)
    # Exact zero parent-cell mean elevation offset
    dz_means = ml.dz_sub_m.reshape(nc, n_sub).mean(dim=1).abs().max().item()
    assert dz_means < 1.0e-4, f"Non-zero parent elevation bias: {dz_means}"

    t_in = torch.full((m,), 285.0)
    p_in = torch.full((m,), 1.0e5)
    sw_in = torch.full((m,), 400.0)
    t_dn, p_dn, sw_dn = downscale_maml_microclimate(ml, t_in, p_in, sw_in)
    # Alpine cell 1 must span > 10 K between valley floor and summit tile!
    t_span_alpine = float(t_dn[16:32].max() - t_dn[16:32].min())
    assert t_span_alpine > 10.0, f"Alpine lapse-rate span too small: {t_span_alpine} K"

    # Wetlands: flat wet basin (cell 0) must flood much more than steep mountain (cell 1)
    w_liq = torch.full((m,), 0.92 * SOIL_POROSITY)
    t_soil = torch.full((m,), 290.0)
    area = torch.full((nc,), 1.0e12)
    q_riv = torch.tensor([8000.0, 50.0, 100.0, 100.0])
    for _ in range(30):
        step_maml_wetlands_lakes(ml, st, sig, w_liq, t_soil, area, nc, n_sub, 86400.0, river_discharge_cell=q_riv)
    fw_flat = float(ml.f_wetland[0:16].mean())
    fw_mtn = float(ml.f_wetland[16:32].mean())
    a_mkm2 = diagnose_maml_wetland_area_mkm2(ml, area, nc, n_sub)
    assert fw_flat > 4.0 * fw_mtn and fw_flat > 0.35 and a_mkm2 > 0.1
    print(f"[PASS] MAML Topo/Wetlands: alpine dT={t_span_alpine:.1f} K (mean_err={dz_means:.1e}), f_wet(flat)={fw_flat:.2f} vs f_wet(mtn)={fw_mtn:.2f}")


def test_ocean_waves_langmuir_and_spray() -> None:
    n = 4
    w = init_ocean_wave_state(n, torch.device("cpu"))
    u10 = torch.tensor([3.0, 10.0, 22.0, 40.0])  # calm, moderate, gale, hurricane
    fice = torch.zeros(n)
    is_ocn = torch.ones(n, dtype=torch.bool)
    for _ in range(40):
        step_ocean_waves(w, u10, fice, is_ocn, 3600.0)
    assert w.hs_m[3] > w.hs_m[2] > w.hs_m[1] > w.hs_m[0]
    assert float(w.langmuir_fac[2]) > 1.05
    diag = diagnose_ocean_wave_metrics(w, is_ocn, fice)

    d_lh, d_sh = compute_sea_spray_enthalpy_fluxes(
        u10,
        torch.full((n,), 301.0),
        torch.full((n,), 299.0),
        torch.full((n,), 0.015),
        torch.full((n,), 1.0e5),
        torch.ones(n),
    )
    assert float(d_lh[0]) == 0.0 and float(d_lh[1]) == 0.0 and float(d_lh[3]) > 100.0

    # Voronoi horizontal swell propagation: storm belt (|lat| > 35 deg) radiates swell into calm equator (U10 = 0)
    from aoba.scvt_mesh import build_scvt_mesh

    mesh = build_scvt_mesh(2, use_earth2014=False, device=torch.device("cpu"))
    nc = mesh.n_cells
    lat_deg = mesh.lat_cell * (180.0 / 3.141592653589793)
    is_eq = lat_deg.abs() < 15.0
    is_storm = lat_deg.abs() > 35.0
    is_ocn_m = torch.ones(nc, dtype=torch.bool)
    fice_m = torch.zeros(nc)
    u10_m = torch.where(is_storm, torch.full((nc,), 20.0), torch.zeros(nc))

    w_prop = init_ocean_wave_state(nc, torch.device("cpu"))
    w_prop.hs_m.zero_()
    w_local = init_ocean_wave_state(nc, torch.device("cpu"))
    w_local.hs_m.zero_()
    for _ in range(96):  # 48 hours
        step_ocean_waves(w_prop, u10_m, fice_m, is_ocn_m, 1800.0, mesh=mesh)
        step_ocean_waves(w_local, u10_m, fice_m, is_ocn_m, 1800.0, mesh=None)
    hs_eq_prop = float(w_prop.hs_m[is_eq].mean())
    hs_eq_local = float(w_local.hs_m[is_eq].mean())
    assert hs_eq_prop > 0.5 and hs_eq_local < 1.0e-5, (
        f"Expected Voronoi swell pool in calm tropics: prop={hs_eq_prop:.2f} m vs local={hs_eq_local:.4f} m"
    )
    print(
        f"[PASS] Ocean Waves: Hs={[round(float(x),2) for x in w.hs_m]} m, La_fac={diag['wave_langmuir_mean']:.2f}, "
        f"Spray LH(40m/s)={float(d_lh[3]):.1f} W/m^2, Tropical Swell={hs_eq_prop:.2f} m"
    )


def test_coupled_batch2_cpu_mps() -> None:
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        cfg = AOBAConfig(
            glevel=1,
            ocean_glevel=2,
            crm_ny=2,
            crm_nx=2,
            enable_maml_subgrid_topo=True,
            enable_maml_wetlands_lakes=True,
            enable_wave_coupling=True,
            enable_river_routing=True,
            enable_land_carbon=True,
            enable_prognostic_ch4_n2o=True,
        )
        m = init_aoba_model(cfg, device=dev)
        d1 = step_aoba_model(m)
        d2 = step_aoba_model(m)
        assert m.maml_land is not None and m.ocean_waves is not None
        assert torch.isfinite(m.maml_land.f_wetland).all() and torch.isfinite(m.ocean_waves.hs_m).all()
        print(
            f"[PASS] Coupled Batch 2 ({dev}): Ts={d2.global_mean_t_sfc_k:.2f} K, "
            f"Hs_max={float(m.ocean_waves.hs_m.max()):.2f} m, La_max={float(m.ocean_waves.langmuir_fac.max()):.2f}, "
            f"max_f_wet={float(m.maml_land.f_wetland.max()):.3f}"
        )


if __name__ == "__main__":
    test_maml_subgrid_topo_and_wetlands()
    test_ocean_waves_langmuir_and_spray()
    test_coupled_batch2_cpu_mps()
    print("ALL BATCH 2 MAML & WAVE COUPLING TESTS PASSED!")
