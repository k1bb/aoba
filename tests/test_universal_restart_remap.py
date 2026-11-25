"""Tests for universal cross-grid restart remapping in `scripts/run_g5_30yr.py`."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.thermodynamics import diagnose_pressure_temperature

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_g5_30yr", _REPO_ROOT / "scripts" / "run_g5_30yr.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_universal_cross_grid_restart_remap_and_legacy_fallback(tmp_path: Path) -> None:
    """Simultaneously change glevel, nz, z_top, crm_nx, ocean_glevel, and ocean_nz across a restart."""
    runner = _load_runner_module()
    device = torch.device("cpu")

    # Source model: coarse G1, nz=12, z_top=36 km, crm_nx=2, unified G1 ocean nz=6
    cfg_src = AOBAConfig(
        glevel=1,
        nz=12,
        z_top=36000.0,
        dz_min=150.0,
        z_sponge_base=28000.0,
        crm_nx=2,
        crm_ny=1,
        dt_host=300.0,
        n_host_dyn_substeps=2,
        n_crm_dyn_substeps=4,
        rad_stride=1,
        rad_scheme="four_band",
        microphysics_scheme="two_moment",
        o3_mode="prognostic",
        ocean_mode="ogcm_3d",
        ocean_glevel=1,
        ocean_nz=6,
        enable_ocean_bgc=True,
        enable_seaice_evp_itd=True,
        enable_river_routing=True,
        enable_dynamic_lakes=True,
        enable_land_carbon=True,
        enable_natural_aerosols=True,
        use_earth2014=False,
    )
    src = init_aoba_model(cfg_src, device=device)
    # Stamp distinct physical temperatures on land vs ocean to verify zero coastal contamination
    land_mask_src = src.sfc.surf_type == 1
    ocn_mask_src = src.sfc.surf_type == 0
    if land_mask_src.any():
        src.sfc.t_canopy[land_mask_src] = 295.0
    if ocn_mask_src.any():
        src.sfc.t_ocean_ml[ocn_mask_src] = 285.0
        if src.ogcm is not None:
            src.ogcm.temp[:, ocn_mask_src] = 282.0 * src.ogcm.wet_c[:, ocn_mask_src]
            src.ogcm.sal[:, ocn_mask_src] = 35.1 * src.ogcm.wet_c[:, ocn_mask_src]

    for _ in range(2):
        step_aoba_model(src)

    ckpt_path = tmp_path / "restart.pt"
    runner._save_restart_atomic(src, ckpt_path)

    # Target model: finer G2, nz=18, z_top=65 km, crm_nx=4, higher-res G3 ocean nz=10
    cfg_dst = AOBAConfig(
        glevel=2,
        nz=18,
        z_top=65000.0,
        dz_min=150.0,
        z_sponge_base=52000.0,
        crm_nx=4,
        crm_ny=1,
        dt_host=300.0,
        n_host_dyn_substeps=2,
        n_crm_dyn_substeps=4,
        rad_stride=1,
        rad_scheme="four_band",
        microphysics_scheme="two_moment",
        o3_mode="prognostic",
        ocean_mode="ogcm_3d",
        ocean_glevel=3,
        ocean_nz=10,
        enable_ocean_bgc=True,
        enable_seaice_evp_itd=True,
        enable_river_routing=True,
        enable_dynamic_lakes=True,
        enable_land_carbon=True,
        enable_natural_aerosols=True,
        use_earth2014=False,
    )
    dst = init_aoba_model(cfg_dst, device=device)
    meta = runner._load_restart_into_model(dst, ckpt_path, device)
    assert meta.get("_vertical_remapped") is True
    assert meta.get("_horizontal_remapped") is True

    # Verify thermodynamic stratospheric extension (36 km -> 65 km)
    p_dst, _, t_dst = diagnose_pressure_temperature(dst.atmos.rho_d, dst.atmos.theta_m, dst.atmos.q_v)
    assert torch.isfinite(p_dst).all() and (p_dst > 0.0).all()
    assert torch.isfinite(t_dst).all() and float(t_dst.min()) >= 170.0 and float(t_dst.max()) <= 330.0
    assert dst.ozone is not None and torch.isfinite(dst.ozone.o3_vmr).all() and (dst.ozone.o3_vmr > 0.0).all()

    # Verify OGCM wet-cell remap & bathymetry masking (dry cells == 0, wet cells finite & physical)
    assert dst.ogcm is not None
    wet = dst.ogcm.wet_c > 0.5
    dry = ~wet
    assert torch.isfinite(dst.ogcm.temp).all() and torch.isfinite(dst.ogcm.sal).all()
    if wet.any():
        assert float(dst.ogcm.temp[wet].min()) >= 271.0
        assert float(dst.ogcm.sal[wet].min()) >= 5.0
    if dry.any():
        assert float(dst.ogcm.temp[dry].abs().max()) == 0.0
    assert dst.ogcm.bgc is not None and torch.isfinite(dst.ogcm.bgc.dic).all()
    assert dst.ogcm.seaice_evp is not None and torch.isfinite(dst.ogcm.seaice_evp.a_cat).all()

    # Verify dynamic lakes bounds & hypsometry
    assert dst.lakes is not None
    assert torch.isfinite(dst.lakes.v_lake_m3).all()
    assert bool((dst.lakes.v_lake_m3 <= dst.lakes.v_cap_m3 + 1.0e-3).all())

    # Advance 2 full coupled steps on the remapped target grid and check zero non-finite states
    for _ in range(2):
        diag = step_aoba_model(dst)
        assert torch.isfinite(torch.tensor(diag.global_mean_t_sfc_k))
    assert runner._inspect_nonfinite_state(dst) == []

    # Backward-compatibility: strip `mesh_coords` (legacy checkpoint) and verify remap still succeeds
    legacy_path = tmp_path / "restart_legacy.pt"
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw.pop("mesh_coords", None)
    torch.save(raw, legacy_path)

    dst_legacy = init_aoba_model(cfg_dst, device=device)
    runner._load_restart_into_model(dst_legacy, legacy_path, device)
    diag_leg = step_aoba_model(dst_legacy)
    assert torch.isfinite(torch.tensor(diag_leg.global_mean_t_sfc_k))
    assert runner._inspect_nonfinite_state(dst_legacy) == []


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as _d:
        test_universal_cross_grid_restart_remap_and_legacy_fallback(Path(_d))
    print("PASS: test_universal_cross_grid_restart_remap_and_legacy_fallback")
