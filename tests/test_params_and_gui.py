#!/usr/bin/env python3
"""Unit and integration tests for unified TOML registry, OutputConfig, and GUI server."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import xarray as xr

from aoba import (
    AOBAConfig,
    OutputConfig,
    build_configs_from_dict,
    constants as C,
    default_config_dict,
    default_toml_path,
    export_aoba_snapshot_xarray,
    init_aoba_model,
    load_aoba_toml,
    reset_default_parameters,
    save_aoba_toml,
)
from aoba import icesheet as _ice
from aoba import microphysics as _mp
from aoba import turbulence as _turb

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from aoba_config_gui import create_gui_server  # noqa: E402


def test_toml_roundtrip_and_module_sync() -> None:
    toml_p = default_toml_path()
    assert toml_p.exists(), f"Missing default TOML: {toml_p}"

    defaults = default_config_dict()
    loaded = load_aoba_toml(toml_p)
    assert len(defaults) == 14
    assert sum(len(v) for v in defaults.values()) == 260
    assert set(loaded.keys()) == set(defaults.keys())
    for sec in defaults:
        assert set(loaded[sec].keys()) == set(defaults[sec].keys()), f"Key mismatch in [{sec}]"

    try:
        custom = default_config_dict()
        custom["constants"]["cp_d"] = 1010.0
        custom["turbulence"]["tke_min"] = 2.5e-4
        custom["microphysics"]["qc_autoconv_thresh"] = 2.0e-3
        custom["microphysics"]["cloud_droplet_conc_cm3"] = 80.0
        custom["forcing"]["cloud_sw_inhomog"] = 0.78
        custom["icesheet_physics"]["rho_ice"] = 900.0
        custom["forcing"]["omega"] = C.OMEGA_EARTH * 0.5

        cfg, out_cfg = build_configs_from_dict(custom, apply_modules=True)
        assert isinstance(cfg, AOBAConfig)
        assert isinstance(out_cfg, OutputConfig)
        assert abs(cfg.omega - C.OMEGA_EARTH * 0.5) < 1e-12
        assert abs(cfg.cloud_sw_inhomog - 0.78) < 1e-12
        assert abs(C.CP_D - 1010.0) < 1e-9
        assert abs(C.CV_D - (1010.0 - C.R_D)) < 1e-9
        assert abs(_turb.TKE_MIN - 2.5e-4) < 1e-12
        assert abs(_turb._MYNN_METAL_CONSTS[17] - 2.5e-4) < 1e-12
        assert abs(_mp.QC_AUTOCONV_THRESH - 2.0e-3) < 1e-12
        assert abs(_mp._MP_METAL_CONSTS[16] - 2.0e-3) < 1e-12
        assert abs(_mp.CLOUD_DROPLET_CONC_CM3 - 80.0) < 1e-9
        assert abs(_mp._MP_METAL_CONSTS[51] - 80.0) < 1e-9
        assert abs(_ice.RHO_ICE - 900.0) < 1e-9
        assert abs(_ice.ISOSTASY_RATIO - 900.0 / _ice.RHO_MANTLE) < 1e-9
    finally:
        reset_default_parameters()

    assert abs(C.CP_D - 1004.64) < 1e-6
    assert abs(_turb.TKE_MIN - 1.0e-4) < 1e-12
    assert abs(_ice.RHO_ICE - 917.0) < 1e-9


def test_snapshot_variable_filtering() -> None:
    cfg = AOBAConfig(glevel=1, nz=8, crm_nx=2, crm_ny=1, use_earth2014=False)
    model = init_aoba_model(cfg, device="cpu")

    ds_no3d = export_aoba_snapshot_xarray(model, include_2d=True, include_3d=False, include_subgrid_2d=True)
    assert "t_canopy" in ds_no3d and "lai_sub" in ds_no3d
    assert "temp" not in ds_no3d and "nz" not in ds_no3d.coords

    ds_only2d = export_aoba_snapshot_xarray(model, include_2d=True, include_3d=False, include_subgrid_2d=False)
    assert "t_canopy" in ds_only2d
    assert "temp" not in ds_only2d and "lai_sub" not in ds_only2d


def test_gui_server_endpoints() -> None:
    server = create_gui_server(default_toml_path(), host="127.0.0.1", port=0)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
        assert "gridCanvas" in html and "globeT2" in html and "globeOLR" in html and "globeLAI" in html
        assert "renderIdleGlobe" in html
        assert "lang-btn" in html and "toggleLang" in html and "tab-docs" in html and "renderMarkdownWithMath" in html
        assert "/assets/aoba_icon.png" in html and "aoba-header-icon" in html
        icon_bytes = urllib.request.urlopen(f"{base}/assets/aoba_icon.png").read()
        assert icon_bytes[:4] == b"\x89PNG"
        svg_txt = urllib.request.urlopen(f"{base}/assets/aoba_icon.svg").read().decode("utf-8")
        assert "<svg" in svg_txt and "othello-tile" in svg_txt
        icon_json = json.loads(urllib.request.urlopen(f"{base}/assets/aoba_icon.json").read().decode("utf-8"))
        assert icon_json["ok"] and len(icon_json["cells"]) == 354
        assert sum(c[2] for c in icon_json["cells"]) == 39

        for doc_k in ("formulation_ja", "formulation_en", "architecture_ja", "architecture_en", "readme_ja", "readme_en"):
            doc_resp = json.loads(urllib.request.urlopen(f"{base}/api/docs?name={doc_k}").read().decode("utf-8"))
            assert doc_resp["ok"] and len(doc_resp["markdown"]) > 200

        for g_lvl, exp_cells, exp_edges in ((0, 12, 30), (3, 642, 1920), (5, 10242, 30720)):
            m_resp = json.loads(urllib.request.urlopen(f"{base}/api/scvt_mesh?glevel={g_lvl}").read().decode("utf-8"))
            assert m_resp["ok"] and m_resp["n_cells"] == exp_cells and m_resp["n_edges"] == exp_edges
            assert len(m_resp["verts_on_cell_b64"]) > 0 and len(m_resp["surf_type_b64"]) > 0

        state = json.loads(urllib.request.urlopen(f"{base}/api/state").read().decode("utf-8"))
        assert state["param_count"] == 260
        assert "grid" in state["config"]

        co2_cat = json.loads(urllib.request.urlopen(f"{base}/api/co2_catalog").read().decode("utf-8"))
        assert len(co2_cat["scenarios"]) >= 20

        o3_st = json.loads(urllib.request.urlopen(f"{base}/api/ozone_status").read().decode("utf-8"))
        assert "rcemip" in o3_st["modes"] and "latitude_seasonal" in o3_st["modes"]

        disc = json.loads(urllib.request.urlopen(f"{base}/api/runs/discover").read().decode("utf-8"))
        assert disc["ok"]

        # Modify glevel & omega via /api/toml and save to temp file via /api/save
        mod_cfg = state["config"]
        mod_cfg["grid"]["glevel"] = 4
        mod_cfg["forcing"]["omega"] = C.OMEGA_EARTH * 0.5
        req_toml = urllib.request.Request(
            f"{base}/api/toml",
            data=json.dumps({"config": mod_cfg}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        toml_resp = json.loads(urllib.request.urlopen(req_toml).read().decode("utf-8"))
        assert toml_resp["ok"] and toml_resp["modified_count"] == 2
        assert "glevel = 4" in toml_resp["toml_text"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_toml = tmp / "custom_half_rot.toml"
            req_save = urllib.request.Request(
                f"{base}/api/save",
                data=json.dumps({"config": mod_cfg, "save_path": str(out_toml)}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            save_resp = json.loads(urllib.request.urlopen(req_save).read().decode("utf-8"))
            assert save_resp["ok"] and out_toml.exists()
            reloaded = load_aoba_toml(out_toml)
            assert reloaded["grid"]["glevel"] == 4
            assert abs(reloaded["forcing"]["omega"] - C.OMEGA_EARTH * 0.5) < 1e-12

            # Launch micro CPU run via /api/run/start, wait for completion, and query /api/monitor/snapshot
            run_out = tmp / "gui_run_out"
            run_cfg = default_config_dict()
            run_cfg["grid"]["glevel"] = 0
            run_cfg["grid"]["nz"] = 8
            run_cfg["grid"]["crm_nx"] = 2
            run_cfg["grid"]["crm_ny"] = 1
            run_cfg["grid"]["dt_host"] = 3600.0
            run_cfg["grid"]["n_crm_dyn_substeps"] = 2
            run_cfg["grid"]["rad_stride"] = 1
            run_cfg["forcing"]["use_earth2014"] = False
            run_cfg["output"]["out_dir"] = str(run_out)
            run_cfg["output"]["years"] = 1
            run_cfg["output"]["days_per_year"] = 0.5

            req_start = urllib.request.Request(
                f"{base}/api/run/start",
                data=json.dumps({"config": run_cfg, "device": "cpu", "no_resume": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start_resp = json.loads(urllib.request.urlopen(req_start).read().decode("utf-8"))
            assert start_resp["ok"] and start_resp["pid"] > 0

            import time
            deadline = time.time() + 40.0
            while time.time() < deadline:
                st = json.loads(urllib.request.urlopen(f"{base}/api/runs/discover").read().decode("utf-8"))
                if not st["managed_run"]["running"]:
                    break
                time.sleep(0.4)
            assert not st["managed_run"]["running"] and st["managed_run"]["returncode"] == 0

            mon = json.loads(
                urllib.request.urlopen(
                    f"{base}/api/monitor/snapshot?out_dir={run_out}&source=live"
                ).read().decode("utf-8")
            )
            assert mon["ok"] and mon["raster"] is not None
            assert mon["raster"]["width"] == 360 and mon["raster"]["height"] == 180
            assert len(mon["raster"]["t2_b64"]) > 1000
            assert len(mon["raster"]["precip_b64"]) > 1000
            assert len(mon["raster"]["lai_b64"]) > 1000
            assert mon["meta"]["source"] == "live_preview.npz"
            assert "Completed Year 0001" in mon["log_tail"]
    finally:
        server.shutdown()
        server.server_close()
        th.join(timeout=2.0)


def test_runner_cli_with_custom_toml_cpu() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        out_dir = tmp / "run_out"
        cfg = default_config_dict()
        cfg["grid"]["glevel"] = 0
        cfg["grid"]["nz"] = 8
        cfg["grid"]["crm_nx"] = 2
        cfg["grid"]["crm_ny"] = 1
        cfg["grid"]["crm_dx"] = 8000.0
        cfg["grid"]["dt_host"] = 3600.0
        cfg["grid"]["n_crm_dyn_substeps"] = 2
        cfg["grid"]["rad_stride"] = 1
        cfg["forcing"]["use_earth2014"] = False
        cfg["icesheet"]["enable_dynamic_icesheet"] = True
        cfg["output"]["out_dir"] = str(out_dir)
        cfg["output"]["years"] = 1
        cfg["output"]["days_per_year"] = 0.5  # 12 host steps
        cfg["output"]["zarr_dir_partition"] = "single"
        cfg["output"]["monthly_mean_periods_per_year"] = 2
        cfg["output"]["monthly_mean_sample_hours"] = 3.0
        cfg["output"]["monthly_mean_include_3d"] = False
        cfg["output"]["snapshots_periods_per_year"] = 2
        cfg["output"]["snapshots_include_subgrid_2d"] = False

        cfg_path = tmp / "test_cfg.toml"
        save_aoba_toml(cfg, cfg_path)

        proc = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "run_g5_30yr.py"),
                "--config",
                str(cfg_path),
                "--device",
                "cpu",
                "--no-resume",
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Completed Year 0001 / 0001" in proc.stdout

        assert (out_dir / "mesh.zarr").exists()
        assert (out_dir / "monthly.zarr").exists()
        assert (out_dir / "snapshots.zarr").exists()
        assert (out_dir / "year_0001" / "_COMPLETE").exists()
        assert (out_dir / "year_0001" / "restart.pt").exists()

        ds_m = xr.open_zarr(out_dir / "monthly.zarr", consolidated=False)
        assert ds_m.sizes["time"] == 2
        assert "t_canopy" in ds_m and "lai_sub" in ds_m and "h_ice" in ds_m
        assert "temp" not in ds_m

        ds_s = xr.open_zarr(out_dir / "snapshots.zarr", consolidated=False)
        assert ds_s.sizes["time"] == 2
        assert "temp" in ds_s and "h_ice" in ds_s
        assert "lai_sub" not in ds_s


def test_co2_forcing_modes_and_calendar() -> None:
    from aoba.calendar import AobaCalendar
    from aoba.co2_forcing import resolve_co2_series

    # 1. Constant mode
    s_const = resolve_co2_series(co2_mode="constant", co2_ppmv=315.0)
    assert not s_const.is_time_varying
    assert abs(s_const.evaluate(2025.0) - 315.0) < 1e-9

    # 2. Bundled/cached scenarios: CMIP5, CMIP6, NOAA Obs, and 800ka Paleo
    s_ssp = resolve_co2_series(co2_mode="scenario", co2_scenario="cmip6_ssp585")
    assert s_ssp.is_time_varying
    assert abs(s_ssp.evaluate(1850.0) - 284.317) < 0.05
    assert s_ssp.evaluate(2100.0) > 1100.0

    s_rcp = resolve_co2_series(co2_mode="scenario", co2_scenario="cmip5_rcp45")
    assert abs(s_rcp.evaluate(2050.0) - 486.5) < 1.0

    s_obs = resolve_co2_series(co2_mode="scenario", co2_scenario="obs_historical_noaa")
    assert s_obs.evaluate(2025.0) > 420.0

    s_paleo = resolve_co2_series(co2_mode="scenario", co2_scenario="paleo_antarctica_800ka")
    assert 170.0 < s_paleo.evaluate(-19050.0) < 220.0

    # 3. Negative BCE calendar parsing + decimal year + coupled model integration
    cal = AobaCalendar(start_date="-19050-03-20", calendar_type="noleap")
    assert cal.start_year == -19050 and abs(cal.decimal_year_at(0.0) - (-19050.0 + 78.0 / 365.0)) < 1e-3

    cfg = AOBAConfig(
        glevel=1, nz=8, crm_nx=2, crm_ny=1, use_earth2014=False,
        co2_mode="scenario", co2_scenario="cmip6_ssp585", start_date="2050-01-01",
    )
    model = init_aoba_model(cfg, device="cpu")
    assert abs(model.current_co2_ppmv() - s_ssp.evaluate(2050.0)) < 0.1

    # 4. Custom CSV mode
    with tempfile.TemporaryDirectory() as tmpdir:
        cpath = Path(tmpdir) / "custom_co2.csv"
        cpath.write_text("year,co2_ppmv\n2000,370.0\n2010,390.0\n", encoding="utf-8")
        s_csv = resolve_co2_series(co2_mode="csv", co2_csv_path=str(cpath))
        assert abs(s_csv.evaluate(2005.0) - 380.0) < 1e-9


def test_ozone_forcing_modes_and_high_top() -> None:
    import numpy as np
    import torch
    from aoba.ozone_forcing import resolve_ozone_provider
    from aoba.radiation import compute_four_band_radiation

    # 1. latitude_seasonal mode: poleward column increase + high-top stratosphere decay (down to 0.01 hPa)
    p_hpa = torch.tensor([1000.0, 500.0, 200.0, 50.0, 10.0, 1.0, 0.05, 0.01], dtype=torch.float32)
    pres_pa = (p_hpa * 100.0).unsqueeze(1).expand(-1, 3)  # [8, 3]
    lat_rad = torch.deg2rad(torch.tensor([0.0, 55.0, -55.0], dtype=torch.float32))

    prov_ls = resolve_ozone_provider("latitude_seasonal")
    vmr_apr = prov_ls.compute_vmr(pres_pa, lat_rad=lat_rad, doy=90.0)
    assert vmr_apr.shape == (8, 3)
    assert torch.isfinite(vmr_apr).all() and (vmr_apr >= 0.0).all()
    # High-top mesosphere (0.01 hPa) should decay well below the ~10 hPa peak
    assert vmr_apr[7, 0] < 0.2 * vmr_apr[4, 0]
    # NH spring (DOY 90) higher at 55N than 55S
    assert vmr_apr[3, 1] > vmr_apr[3, 2]

    # 2. External climatology mode with high-top extrapolation above table top (0.3 hPa -> 0.01 hPa)
    with tempfile.TemporaryDirectory() as tmpdir:
        o3_npz = Path(tmpdir) / "test_o3.npz"
        p_tbl = np.array([0.3, 1.0, 10.0, 50.0, 200.0, 1000.0], dtype=np.float32)
        lat_tbl = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
        vmr_tbl = np.zeros((12, len(p_tbl), len(lat_tbl)), dtype=np.float32)
        vmr_tbl[:, 0, :] = 1.5e-6
        vmr_tbl[:, 1, :] = 6.0e-6
        vmr_tbl[:, 2, :] = 9.0e-6
        vmr_tbl[:, 3, :] = 2.0e-6
        vmr_tbl[:, 4, :] = 2.0e-7
        vmr_tbl[:, 5, :] = 3.0e-8
        np.savez(o3_npz, lat_deg=lat_tbl, p_hpa=p_tbl, vmr=vmr_tbl)

        prov_ext = resolve_ozone_provider("climatology", str(o3_npz))
        vmr_ext = prov_ext.compute_vmr(pres_pa, lat_rad=lat_rad, doy=15.0)
        assert vmr_ext.shape == (8, 3) and torch.isfinite(vmr_ext).all()
        assert vmr_ext[7, 0] < vmr_ext[5, 0]  # Decays smoothly above 0.3 hPa table top

    # 3. 9.6 um LW coupling reduces TOA OLR slightly
    nz = 8
    p_w = torch.linspace(100000.0, 100.0, nz + 1)
    p_c = 0.5 * (p_w[:-1] + p_w[1:])
    t_c = torch.linspace(290.0, 220.0, nz)
    t_w = torch.linspace(295.0, 220.0, nz + 1)
    dz = torch.full((nz,), 2000.0)
    rho = p_c / (287.04 * t_c)
    z0 = torch.zeros((nz, 1))
    rad_off = compute_four_band_radiation(
        p_c[:, None], p_w[:, None], t_c[:, None], t_w[:, None], dz[:, None], rho[:, None],
        z0 + 1e-3, z0, z0, torch.tensor([295.0]), torch.tensor([0.1]), torch.tensor([0.2]),
        torch.tensor([0.5]), torch.tensor([1361.0]), o3_lw_coupling=False,
    )
    rad_on = compute_four_band_radiation(
        p_c[:, None], p_w[:, None], t_c[:, None], t_w[:, None], dz[:, None], rho[:, None],
        z0 + 1e-3, z0, z0, torch.tensor([295.0]), torch.tensor([0.1]), torch.tensor([0.2]),
        torch.tensor([0.5]), torch.tensor([1361.0]), o3_lw_coupling=True,
    )
    assert rad_on.toa_olr.item() < rad_off.toa_olr.item()


def test_nonorographic_gwd_and_stratosphere_high_top() -> None:
    import torch
    from aoba.coupled_model import step_aoba_model
    from aoba.gravity_wave import compute_frontogenesis_2d, compute_nonorographic_gwd
    from aoba.scvt_mesh import build_scvt_mesh
    from aoba.thermodynamics import diagnose_pressure_temperature

    # 1. Frontogenesis operator: zero for uniform field, positive in confluent thermal gradient
    mesh = build_scvt_mesh(glevel=2, n_lloyd_iter=4, use_earth2014=False, device="cpu")
    th_const = torch.full((mesh.n_cells,), 300.0)
    u_zero = torch.zeros((mesh.n_cells,))
    ff_zero = compute_frontogenesis_2d(mesh, th_const, u_zero, u_zero)
    assert float(ff_zero.abs().max().item()) < 1e-18

    # Synthetic midlatitude baroclinic frontal zone with meridional confluence
    th_grad = 300.0 - 20.0 * torch.tanh(10.0 * (mesh.lat_cell - 0.785))
    v_conf = -20.0 * torch.tanh(10.0 * (mesh.lat_cell - 0.785))
    ff_front = compute_frontogenesis_2d(mesh, th_grad, u_zero, v_conf)
    assert float(ff_front.max().item()) > 1.25e-15

    # 2. High-top stratospheric coupled integration (48 km, nz=44, map-free NGWD + 2D O3 + 9.6 um LW)
    cfg = AOBAConfig(
        glevel=1,
        nz=44,
        z_top=48000.0,
        z_sponge_base=36000.0,
        crm_nx=2,
        crm_ny=1,
        use_earth2014=False,
        enable_nonoro_gwd=True,
        o3_mode="latitude_seasonal",
        o3_lw_coupling=True,
        dt_host=600.0,
        n_crm_dyn_substeps=15,
        rad_stride=1,
    )
    model = init_aoba_model(cfg, device="cpu")
    p0, _, t0 = diagnose_pressure_temperature(model.atmos.rho_d, model.atmos.theta_m, model.atmos.q_v)
    # Initial stratopause near 46 km should be ~0.5-1.5 hPa and ~255-275 K
    assert 50.0 < float(p0[-1].mean().item()) < 200.0
    assert 245.0 < float(t0[-1].mean().item()) < 280.0

    # Inject realistic convective heating and shear wind to verify Beres + Lindzen saturation in stratosphere
    q_conv = torch.zeros_like(model.atmos.rho_d)
    trop_mask = (model.atmos.z_c > 2000.0) & (model.atmos.z_c < 12000.0)
    q_conv = torch.where(trop_mask, torch.full_like(q_conv, 3.0e-4), q_conv)
    u_shear = 45.0 * torch.tanh((model.atmos.z_c - 15000.0) / 10000.0)
    du_gwd, dv_gwd = compute_nonorographic_gwd(
        mesh=model.mesh,
        rho_d=model.atmos.rho_d,
        pres_c=p0,
        u_cell=u_shear,
        v_cell=torch.zeros_like(u_shear),
        theta_m=model.atmos.theta_m,
        z_rel_c=model.atmos.z_c - model.atmos.z_w[0:1],
        dz_c=model.atmos.dz_c,
        q_conv_c=q_conv,
        dt=600.0,
    )
    assert torch.isfinite(du_gwd).all() and torch.isfinite(dv_gwd).all()
    # Stratospheric breaking above 20 km should decelerate the strong positive westerly jet
    strat_idx = model.vgrid.zeta_c > 20000.0
    assert float(du_gwd[strat_idx].abs().max().item()) > 1.0e-6

    # Run 6 coupled steps across the full 48 km stratosphere domain
    for _ in range(6):
        diag = step_aoba_model(model)
        assert 150.0 < diag.global_mean_t_sfc_k < 340.0
    p1, _, t1 = diagnose_pressure_temperature(model.atmos.rho_d, model.atmos.theta_m, model.atmos.q_v)
    assert torch.isfinite(t1).all() and float(t1[-1].min().item()) > 180.0 and float(t1[-1].max().item()) < 310.0

    # 3. Seamless vertical-remap restart: low-top (nz=16, 24 km) -> high-top (nz=44, 48 km)
    from run_g5_30yr import _load_restart_into_model, _save_restart_atomic
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_low = AOBAConfig(
            glevel=1, nz=16, z_top=24000.0, z_sponge_base=17500.0,
            crm_nx=2, crm_ny=1, use_earth2014=False, dt_host=600.0, n_crm_dyn_substeps=10,
        )
        m_low = init_aoba_model(cfg_low, device="cpu")
        step_aoba_model(m_low)
        ckpt_p = Path(tmpdir) / "low_top_restart.pt"
        _save_restart_atomic(m_low, ckpt_p)

        m_high = init_aoba_model(cfg, device="cpu")
        ck = _load_restart_into_model(m_high, ckpt_p, torch.device("cpu"))
        assert ck.get("_vertical_remapped", False) is True
        assert m_high.atmos.rho_d.shape[0] == 44 and m_high.crm.rho_d.shape[0] == 44
        for _ in range(2):
            d_hi = step_aoba_model(m_high)
            assert 150.0 < d_hi.global_mean_t_sfc_k < 340.0
        _, _, t_hi = diagnose_pressure_temperature(m_high.atmos.rho_d, m_high.atmos.theta_m, m_high.atmos.q_v)
        assert torch.isfinite(t_hi).all() and float(t_hi[-1].min().item()) > 180.0 and float(t_hi[-1].max().item()) < 310.0


if __name__ == "__main__":
    test_toml_roundtrip_and_module_sync()
    test_snapshot_variable_filtering()
    test_gui_server_endpoints()
    test_co2_forcing_modes_and_calendar()
    test_ozone_forcing_modes_and_high_top()
    test_nonorographic_gwd_and_stratosphere_high_top()
    test_runner_cli_with_custom_toml_cpu()
    print("ALL PARAMS, OUTPUT & GUI TESTS PASSED.")
