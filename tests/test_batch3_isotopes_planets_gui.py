#!/usr/bin/env python3
"""Verification tests for Batch 3 Universal ESM modules:
1. Prognostic Stable Water Isotopes (H2^18O & HDO) with Majoube/Merlivat-Nief/Jouzel-Merlivat
   fractionation, Craig-Gordon marine evaporation, exact Rayleigh distillation, & ice-core archiving.
2. Universal Planetary / Paleo / Exoplanet Presets (aquaplanet, snowball_earth, trappist1e, mars_like)
   and synchronous tidally locked substellar insolation.
3. Coupled Full-ESM ALL-ON integration on CPU + MPS parity & Zarr v3 export of new 2D ESM fields.
4. AOBA Studio Full-ESM 3D globe variable selector (`esm_field`), 1-click presets, and
   Observational Earth-System Benchmark Panel (`obs_benchmark`).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from aoba import (
    AOBAConfig,
    default_toml_path,
    export_aoba_snapshot_xarray,
    init_aoba_model,
    step_aoba_model,
)
from aoba.planetary_presets import (
    VALID_PLANET_PRESETS,
    compute_tidally_locked_insolation,
    resolve_planet_preset_config,
)
from aoba.surface import SURF_LAND, SURF_OCEAN
from aoba.water_isotopes import (
    equilibrium_fractionation_factors,
    init_water_isotope_state,
    step_water_isotopes,
)
from aoba_config_gui import create_gui_server


def test_water_isotope_physics() -> None:
    """Verify Majoube (1971) / Merlivat-Nief (1967) fractionation, Craig-Gordon, Rayleigh & snow archive."""
    temps = torch.tensor([293.15, 273.15, 243.15], dtype=torch.float32)
    a18, ad, a18_l, ad_l = equilibrium_fractionation_factors(temps)

    # At 20 C (293.15 K), Majoube (1971): alpha_l^18 ~ 1.0098, alpha_l^D ~ 1.085
    assert 1.0090 < float(a18_l[0]) < 1.0105, f"Unexpected alpha_l^18(20C) = {float(a18_l[0])}"
    assert 1.080 < float(ad_l[0]) < 1.092, f"Unexpected alpha_l^D(20C) = {float(ad_l[0])}"
    # Colder temperatures produce stronger fractionation (alpha increases as T drops)
    assert float(a18[2]) > float(a18[1]) > float(a18[0])
    assert float(ad[2]) > float(ad[1]) > float(ad[0])

    # Column test: cell 0 = warm tropical ocean (evaporating), cell 1 = midlat raining, cell 2 = polar snowing
    nz, nc = 6, 3
    q_v = torch.full((nz, nc), 0.008, dtype=torch.float32)
    q_v[:, 2] = 0.0015
    temp_c = torch.tensor(
        [
            [298.0, 282.0, 250.0],
            [290.0, 275.0, 246.0],
            [282.0, 268.0, 242.0],
            [272.0, 260.0, 238.0],
            [260.0, 250.0, 234.0],
            [248.0, 240.0, 230.0],
        ],
        dtype=torch.float32,
    )
    rho_d = torch.full((nz, nc), 1.0, dtype=torch.float32)
    dz_c = torch.full((nz, nc), 1000.0, dtype=torch.float32)
    surf_type = torch.tensor([SURF_OCEAN, SURF_LAND, SURF_LAND], dtype=torch.int32)
    t_sfc = torch.tensor([300.0, 285.0, 248.0], dtype=torch.float32)
    evap = torch.tensor([5.0e-5, 0.0, 0.0], dtype=torch.float32)
    rain = torch.tensor([0.0, 4.0e-4, 0.0], dtype=torch.float32)
    snow = torch.tensor([0.0, 0.0, 2.0e-4], dtype=torch.float32)

    iso = init_water_isotope_state(q_v)
    r18_init_rain = float(iso.q_v_18o[0, 1] / q_v[0, 1])

    for _ in range(10):
        step_water_isotopes(
            iso, q_v, temp_c, rho_d, dz_c, surf_type, t_sfc, evap, rain, snow, dt=1800.0
        )

    # Rayleigh rainout must deplete residual vapor in precipitating cell 1
    r18_after_rain = float(iso.q_v_18o[0, 1] / q_v[0, 1])
    assert r18_after_rain < r18_init_rain, f"Expected Rayleigh depletion: {r18_after_rain} vs {r18_init_rain}"
    # Ocean marine vapor near -15..-5 permil VSMOW
    d18_marine = (float(iso.q_v_18o[0, 0] / q_v[0, 0]) - 1.0) * 1000.0
    assert -18.0 < d18_marine < -4.0, f"Unexpected marine vapor d18O = {d18_marine} permil"
    # Polar snowfall archive must accumulate snow weight and remain finite
    assert float(iso.snow_iso_weight[2]) > 0.0
    assert torch.isfinite(iso.d18o_snow_permil).all()
    assert torch.isfinite(iso.dexcess_permil).all()


def test_planetary_presets_and_tidal_lock() -> None:
    """Verify aquaplanet, snowball_earth, trappist1e, mars_like presets and substellar insolation."""
    assert set(VALID_PLANET_PRESETS) == {
        "earth", "aquaplanet", "snowball_earth", "tidally_locked", "trappist1e", "mars_like"
    }

    # Substellar insolation geometry
    lat = torch.tensor([0.0, 0.0, math.pi / 3.0, 0.0], dtype=torch.float32)
    lon = torch.tensor([0.0, math.pi, 0.0, math.pi / 2.0], dtype=torch.float32)
    mu0, s_toa, _ = compute_tidally_locked_insolation(lat, lon, solar_constant=900.0, substellar_lon_deg=0.0)
    assert abs(float(mu0[0]) - 1.0) < 1e-6    # substellar point
    assert abs(float(mu0[1]) - 0.0) < 1e-6    # antistellar permanent night
    assert abs(float(mu0[2]) - 0.5) < 1e-5    # 60 deg N on substellar meridian
    assert abs(float(mu0[3]) - 0.0) < 1e-5    # terminator
    assert float(s_toa[0]) == 900.0

    # Aquaplanet initialization: 100% ocean, zero topography
    cfg_aqua = AOBAConfig(glevel=1, nz=12, crm_nx=2, crm_ny=1, planet_preset="aquaplanet")
    m_aqua = init_aoba_model(cfg_aqua, device="cpu")
    assert bool((m_aqua.sfc.surf_type == SURF_OCEAN).all())
    assert float(m_aqua.sfc.z_topo.abs().max()) == 0.0

    # Snowball Earth: glaciated surface + reduced solar constant
    cfg_snow = AOBAConfig(glevel=1, nz=12, crm_nx=2, crm_ny=1, use_earth2014=False, planet_preset="snowball_earth")
    m_snow = init_aoba_model(cfg_snow, device="cpu")
    assert m_snow.cfg.solar_constant == 1280.0
    assert m_snow.cfg.enable_silicate_weathering is True
    assert float(m_snow.sfc.swe_land.max()) >= 80.0 or float(m_snow.sfc.m_seaice.max()) >= 2000.0

    # TRAPPIST-1e: tidally locked ocean planet runs cleanly for 2 steps
    cfg_t1e = AOBAConfig(glevel=1, nz=12, crm_nx=2, crm_ny=1, planet_preset="trappist1e")
    m_t1e = init_aoba_model(cfg_t1e, device="cpu")
    assert m_t1e.cfg.enable_tidal_lock is True
    assert abs(m_t1e.cfg.solar_constant - 900.0) < 1e-6
    for _ in range(2):
        step_aoba_model(m_t1e)
    assert torch.isfinite(m_t1e.sfc.t_ocean_ml).all()
    assert torch.isfinite(m_t1e.atmos.theta_m).all()

    # Mars-like: 100% arid land
    cfg_mars = resolve_planet_preset_config(AOBAConfig(glevel=1, nz=12, crm_nx=2, crm_ny=1, planet_preset="mars_like"))
    assert cfg_mars.solar_constant == 589.0
    m_mars = init_aoba_model(cfg_mars, device="cpu")
    assert bool((m_mars.sfc.surf_type == SURF_LAND).all())


def test_coupled_batch3_cpu_mps_and_gui_esm_monitor() -> None:
    """Verify Batch 3 ALL-ON coupled integration (CPU + MPS), Zarr ESM export, and AOBA Studio monitor."""
    cfg = AOBAConfig(
        glevel=1,
        nz=12,
        crm_nx=2,
        crm_ny=1,
        use_earth2014=False,
        co2_mode="prognostic",
        enable_co2_3d_transport=True,
        o3_mode="prognostic",
        enable_ocean_bgc=True,
        enable_ocean_o2_omz=True,
        enable_volcanic_aerosols=True,
        volcanic_eruption_year=0.0,
        volcanic_so2_tg=15.0,
        enable_solar_cycle=True,
        enable_maml_subgrid_topo=True,
        enable_maml_wetlands_lakes=True,
        enable_wave_coupling=True,
        enable_water_isotopes=True,
        enable_lightning=True,
    )
    m_cpu = init_aoba_model(cfg, device="cpu")
    for _ in range(3):
        step_aoba_model(m_cpu)
    assert m_cpu.water_isotopes is not None

    assert torch.isfinite(m_cpu.water_isotopes.q_v_18o).all()
    assert torch.isfinite(m_cpu.water_isotopes.q_v_hdo).all()
    assert torch.isfinite(m_cpu.water_isotopes.d18o_precip_permil).all()

    # Verify MPS parity if Apple Silicon MPS is available
    if torch.backends.mps.is_available():
        m_mps = init_aoba_model(cfg, device="mps")
        for _ in range(3):
            step_aoba_model(m_mps)
        d18_diff = float(
            (m_cpu.water_isotopes.d18o_precip_permil - m_mps.water_isotopes.d18o_precip_permil.cpu()).abs().max()
        )
        assert math.isfinite(d18_diff) and d18_diff < 2.0, f"CPU/MPS d18O divergence: {d18_diff}"

    # Export Zarr v3 snapshot and check all new Full-ESM variables
    ds = export_aoba_snapshot_xarray(m_cpu, include_2d=True, include_3d=False, include_subgrid_2d=False)
    for var in (
        "o3_tco_du", "o3_sfc_ppbv", "lightning_flash_rate", "volc_aod",
        "f_wetland", "d18o_precip_permil", "d18o_snow_permil",
        "co2_sfc_ppmv", "ocn_chl_sfc", "ocn_o2_sfc", "ocn_omz_frac", "wave_hs_m",
    ):
        assert var in ds, f"Missing Full-ESM Zarr variable: {var}"
        assert np.isfinite(ds[var].values).all(), f"Non-finite values in {var}"

    # Verify AOBA Studio HTTP server serves ESM 3D globe selector, 1-click presets & obs_benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir)
        snap_store = out_root / "snapshots.zarr"
        ds.to_zarr(snap_store, mode="w")

        # Write a synthetic year_0001/daily_global.csv matching run_g5_30yr.py so obs_benchmark renders
        y1_dir = out_root / "year_0001"
        y1_dir.mkdir(parents=True, exist_ok=True)
        csv_path = y1_dir / "daily_global.csv"
        csv_path.write_text(
            "sim_days,ts_k,olr_wm2,asr_wm2,net_toa_wm2,precip_mm_d,lai,seaice_mkm2,wall_s_per_day\n"
            "1.0,287.9,239.5,240.2,0.7,2.75,1.62,21.8,12.0\n"
            "2.0,288.1,240.1,240.6,0.5,2.78,1.65,22.1,12.1\n",
            encoding="utf-8",
        )

        server = create_gui_server(default_toml_path(), host="127.0.0.1", port=0)
        port = server.server_address[1]
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            base = f"http://127.0.0.1:{port}"
            html = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
            assert "esm-globe-field" in html
            assert "obs-benchmark-box" in html
            assert "applyForcingPreset('full_esm')" in html
            assert "applyForcingPreset('trappist1e')" in html

            for field in ("o3_tco_du", "d18o_precip_permil", "wave_hs_m", "f_wetland"):
                url = (
                    f"{base}/api/monitor/snapshot?"
                    f"out_dir={urllib.request.quote(str(out_root))}&source=snapshots&esm_field={field}"
                )
                snap = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))
                assert snap["ok"]
                assert snap["meta"]["esm_field"] == field
                assert snap["raster"] is not None
                assert len(snap["raster"]["lai_b64"]) > 0
                assert len(snap["obs_benchmark"]) == 6
                assert all(r["status"] == "ok" for r in snap["obs_benchmark"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    test_water_isotope_physics()
    test_planetary_presets_and_tidal_lock()
    test_coupled_batch3_cpu_mps_and_gui_esm_monitor()
    print("ALL BATCH 3 WATER ISOTOPES, PLANETARY PRESETS & AOBA STUDIO TESTS PASSED!")
