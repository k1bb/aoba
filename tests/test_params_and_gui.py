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
    assert len(defaults) == 13
    assert sum(len(v) for v in defaults.values()) == 216
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
        assert state["param_count"] == 216
        assert "grid" in state["config"]

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


if __name__ == "__main__":
    test_toml_roundtrip_and_module_sync()
    test_snapshot_variable_filtering()
    test_gui_server_endpoints()
    test_runner_cli_with_custom_toml_cpu()
    print("ALL PARAMS, OUTPUT & GUI TESTS PASSED.")
