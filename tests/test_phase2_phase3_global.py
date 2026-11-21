"""Phase 2 & Phase 3 Global Tests: SCVT Voronoi Core, Earth2014, Orography, Ekman Ocean, Köppen, & Full AOBA."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.icesheet import IceSheetState, init_icesheet_state, step_sia_icesheet_scvt
from aoba.io_zarr import append_zarr, export_aoba_snapshot_xarray, load_zarr
from aoba.koppen import KOPPEN_NAMES, classify_koppen_geiger
from aoba.ocean_ekman import init_ekman_ocean_state, step_ekman_ocean_voronoi
from aoba.orography import compute_orographic_drag_lott_miller
from aoba.scvt_dynamics import (
    compute_held_suarez_forcing,
    compute_scvt_divergence,
    fct_advect_scvt,
    init_global_hydrostatic_atmosphere,
    step_scvt_rk3_hevi,
)
from aoba.scvt_mesh import build_scvt_mesh
from aoba.surface import SURF_ICESHEET, SURF_OCEAN, init_surface_state
from aoba.vertical_grid import build_vertical_grid


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def test_scvt_mesh_trisk_and_earth2014(device: str) -> None:
    """Verify spherical Voronoi areas, Gauss divergence theorem, TRiSK weights, & Earth2014 masks."""
    mesh = build_scvt_mesh(
        glevel=3, n_lloyd_iter=8, use_earth2014=True, earth2014_stride=6, device=device
    )
    sphere_area = 4.0 * math.pi * (C.EARTH_RADIUS ** 2)
    rel_area_err = abs(float(mesh.area_cell.sum().item()) - sphere_area) / sphere_area
    assert rel_area_err < 1.0e-5

    # Earth2014 global ocean fraction ~ 0.703, ice sheets present, Himalayas > 3800 m
    ocn_frac_global = float(((mesh.ocean_frac * mesh.area_cell).sum() / mesh.area_cell.sum()).item())
    assert 0.68 < ocn_frac_global < 0.73
    assert int((mesh.surf_type == SURF_ICESHEET).sum().item()) >= 10
    assert float(mesh.z_topo.max().item()) > 3800.0

    # Exact discrete Gauss divergence theorem: sum_i A_i * div_i == 0
    flux_rand = torch.randn(mesh.n_edges, device=device)
    div_rand = compute_scvt_divergence(mesh, flux_rand)
    global_div_int = float((mesh.area_cell * div_rand).sum().item()) / sphere_area
    assert abs(global_div_int) < 1.0e-6

    # TRiSK solid-body rotation accuracy & energy-conserving skew-symmetry
    u_east = 25.0 * torch.cos(mesh.lat_edge)
    u_norm = u_east * mesh.east_normal_edge
    u_tan_exact = -u_east * mesh.north_normal_edge
    u_tan_trisk = (mesh.trisk_weights * u_norm[mesh.trisk_edges]).sum(dim=-1)
    err_rms = float((u_tan_trisk - u_tan_exact).pow(2).mean().sqrt().item())
    assert err_rms < 0.25

    ke_work = float((mesh.area_edge * flux_rand * (mesh.trisk_weights * flux_rand[mesh.trisk_edges]).sum(dim=-1)).sum().item())
    assert abs(ke_work) / float((mesh.area_edge * flux_rand * flux_rand).sum().item()) < 1.0e-6


def test_orography_and_scvt_dynamics_conservation(device: str) -> None:
    """Verify Lott & Miller (1997) mountain drag, global SCVT mass conservation, FCT, & Held-Suarez."""
    mesh = build_scvt_mesh(glevel=2, n_lloyd_iter=6, use_earth2014=True, earth2014_stride=8, device=device)
    vgrid = build_vertical_grid(device=device)
    state = init_global_hydrostatic_atmosphere(mesh, vgrid, moist=True, baroclinic_pert=True)

    # 1. Lott & Miller mountain drag opposes wind over mountains & vanishes over ocean
    u_test = torch.full_like(state.rho_d, 15.0)
    v_test = torch.zeros_like(state.rho_d)
    z_rel = state.z_c - state.z_w[0:1]
    du_oro, dv_oro = compute_orographic_drag_lott_miller(
        state.rho_d, u_test, v_test, state.theta_m, z_rel, state.dz_c,
        mesh.sigma_topo, mesh.slope_topo, mesh.aniso_topo, mesh.angle_topo, dt=300.0,
    )
    is_ocn = mesh.surf_type == SURF_OCEAN
    assert float(du_oro[:, is_ocn].abs().max().item()) == 0.0
    has_mtn = (mesh.sigma_topo > 100.0) & (mesh.slope_topo > 1.0e-3)
    col_drag_x = (state.rho_d * state.dz_c * du_oro).sum(dim=0)
    assert (col_drag_x[has_mtn] < 0.0).all()

    # 2. Global dry-air mass conservation & Voronoi FCT tracer conservation over Earth2014 terrain
    area = mesh.area_cell.unsqueeze(0)
    mass_0 = (area * state.rho_d * state.dz_c).sum()
    q_tracer = torch.zeros_like(state.q_n)
    q_tracer[2:8, :30] = 5.0e-3
    tmass_0 = (area * state.rho_d * state.dz_c * q_tracer).sum()

    for _ in range(5):
        du_hs, dth_hs = compute_held_suarez_forcing(state, mesh)
        m_old = (state.rho_d * state.dz_c).clone()
        uh_e, om_z = step_scvt_rk3_hevi(
            state, mesh, vgrid, dt_dyn=90.0, du_phys_edge=du_hs, dtheta_m_phys=dth_hs
        )
        m_new = state.rho_d * state.dz_c
        q_tracer = fct_advect_scvt(mesh, q_tracer, m_old, m_new, uh_e, om_z, dt=90.0)

    mass_1 = (area * state.rho_d * state.dz_c).sum()
    tmass_1 = (area * state.rho_d * state.dz_c * q_tracer).sum()
    assert torch.allclose(mass_1, mass_0, rtol=5.0e-5)
    assert torch.allclose(tmass_1, tmass_0, rtol=1.0e-4)
    assert (q_tracer >= 0.0).all()
    assert not torch.isnan(state.u_edge).any()


def test_ekman_ocean_upwelling_and_heat_conservation(device: str) -> None:
    """Verify Codron (2012) equatorial Ekman upwelling & exact global 2-layer ocean heat conservation."""
    mesh = build_scvt_mesh(glevel=3, n_lloyd_iter=6, use_earth2014=True, earth2014_stride=8, device=device)
    sfc = init_surface_state(mesh.surf_type, t_init=295.0)
    sfc.t_ocean_ml = 301.0 - 25.0 * (torch.sin(mesh.lat_cell) ** 2)
    ocn = init_ekman_ocean_state(sfc, dt_thermocline_k=8.0)

    is_ocn = (mesh.surf_type == SURF_OCEAN).to(mesh.area_cell.dtype)
    c1 = C.CP_SEAWATER * C.RHO_SEAWATER * sfc.h_ocean_ml
    c2 = C.CP_SEAWATER * C.RHO_SEAWATER * ocn.h_sub_ml

    # Measure enthalpy relative to T_FREEZE_OCEAN for float32 precision
    def _ocean_heat() -> torch.Tensor:
        e_cell = c1 * (sfc.t_ocean_ml - C.T_FREEZE_OCEAN) + c2 * (ocn.t_sub_ml - C.T_FREEZE_OCEAN)
        return (mesh.area_cell * is_ocn * e_cell).sum()

    h_before = _ocean_heat()

    # Impose tropical easterly trade winds (u_1 < 0 -> upward sfc momentum flux tau_x_sfc = +0.10 N/m^2)
    # and midlatitude westerlies (u_1 > 0 -> tau_x_sfc = -0.12 N/m^2)
    lat_deg = torch.rad2deg(mesh.lat_cell)
    tau_x_sfc = torch.where(lat_deg.abs() < 28.0, torch.full_like(lat_deg, 0.10), torch.full_like(lat_deg, -0.12))
    tau_y_sfc = torch.zeros_like(lat_deg)
    zeros = torch.zeros_like(lat_deg)

    for _ in range(10):
        step_ekman_ocean_voronoi(ocn, sfc, mesh, tau_x_sfc, tau_y_sfc, zeros, zeros, dt=3600.0)

    h_after = _ocean_heat()
    assert torch.allclose(h_after, h_before, rtol=2.0e-4)

    # Equatorial ocean band (|lat| < 8 deg) must exhibit net Ekman upwelling (W_E > 0)!
    eq_ocn = (mesh.surf_type == SURF_OCEAN) & (lat_deg.abs() < 8.0)
    subtrop_ocn = (mesh.surf_type == SURF_OCEAN) & (lat_deg.abs() > 18.0) & (lat_deg.abs() < 35.0)
    assert float(ocn.w_ekman[eq_ocn].mean().item()) > 0.0
    assert float(ocn.w_ekman[subtrop_ocn].mean().item()) < 0.0

    # Verify 1.5-layer shallow-water state: finite, coastal normal velocity == 0, volume conserved
    assert ocn.h_sw is not None and ocn.u_sw_edge is not None
    assert not torch.isnan(ocn.h_sw).any() and not torch.isnan(ocn.u_sw_edge).any()
    c0, c1 = mesh.cells_on_edge[:, 0], mesh.cells_on_edge[:, 1]
    ocn_edge = (mesh.surf_type[c0] == SURF_OCEAN) & (mesh.surf_type[c1] == SURF_OCEAN)
    assert float(ocn.u_sw_edge[~ocn_edge].abs().max().item()) == 0.0
    assert float(ocn.u_sw_edge[ocn_edge].abs().max().item()) > 0.0
    vol_sw = float((mesh.area_cell * is_ocn * ocn.h_sw).sum().item())
    vol_init = float((mesh.area_cell * is_ocn * (sfc.h_ocean_ml + ocn.h_sub_ml)).sum().item())
    assert abs(vol_sw - vol_init) / vol_init < 1.0e-5
    # Subtropical gyre convergence should thicken h_sw relative to equatorial divergence
    assert float(ocn.h_sw[subtrop_ocn].mean().item()) > float(ocn.h_sw[eq_ocn].mean().item())


def test_koppen_geiger_beck_2018_table1(device: str) -> None:
    """Verify exact Beck et al. (2018, Table 1) 30-class classification across canonical stations."""
    # 9 stations: Singapore(Af), Sahara(BWh), Rome(Csa), Tokyo(Cfa), London(Cfb),
    #             Moscow(Dfb), Yakutsk(Dfd), Utqiagvik/Barrow(ET), DomeC(EF, E-before-B!), CapeTown(Csb, SH!)
    t_mon = torch.tensor(
        [
            # Singapore (Af): all months >= 26 C, all P >= 120 mm
            [26.5, 27.0, 27.5, 28.0, 28.0, 28.0, 27.8, 27.7, 27.5, 27.4, 26.8, 26.5],
            # Sahara (BWh): T_ann > 25 C, hyper-arid P ~ 2 mm/mo
            [14.0, 17.0, 22.0, 27.0, 32.0, 36.0, 38.0, 37.0, 33.0, 27.0, 20.0, 15.0],
            # Rome (Csa): NH Mediterranean dry hot summer (T_hot = 25 >= 22, P_sdry = 15 < 40 & < P_wwet/3)
            [8.0, 9.0, 11.0, 14.0, 18.0, 22.0, 25.0, 25.0, 21.0, 17.0, 12.0, 9.0],
            # Tokyo (Cfa): 0 < T_cold = 5.2 < 18, T_hot = 26.4 >= 22, humid all year
            [5.2, 5.7, 8.7, 13.9, 18.2, 21.4, 25.0, 26.4, 22.8, 17.5, 12.1, 7.6],
            # London (Cfb): 0 < T_cold = 4.5 < 18, T_hot = 18.5 < 22, >= 4 months > 10 C
            [4.5, 5.0, 7.0, 9.5, 13.0, 16.0, 18.5, 18.0, 15.0, 11.5, 7.5, 5.0],
            # Moscow (Dfb): T_cold = -7.5 <= 0, T_hot = 19.2 < 22, >= 4 months > 10 C, humid
            [-7.5, -6.7, -1.0, 6.7, 13.2, 17.0, 19.2, 17.0, 11.3, 5.6, -1.2, -5.2],
            # Yakutsk (Dfd): T_cold = -40.0 < -38.0, 3 months > 10 C
            [-40.0, -34.0, -20.0, -4.0, 8.0, 16.0, 19.5, 15.0, 6.0, -8.0, -27.0, -38.0],
            # Barrow (ET): 0 <= T_hot = 4.7 < 10 C
            [-25.0, -26.0, -24.0, -16.0, -5.0, 2.0, 4.7, 3.8, -0.5, -8.0, -17.0, -23.0],
            # Dome C Antarctica (EF): T_hot = -28 < 0 C, hyper-dry P = 2 mm/yr -> MUST be EF, NOT BWk!
            [-55.0, -48.0, -58.0, -64.0, -65.0, -65.0, -64.0, -63.0, -60.0, -52.0, -38.0, -28.0],
            # Cape Town (Csb, Southern Hemisphere lat < 0): dry warm Dec-Feb summer!
            [20.5, 20.8, 19.5, 17.0, 14.5, 12.8, 12.2, 12.8, 14.2, 16.0, 18.0, 19.8],
        ],
        dtype=torch.float32,
        device=device,
    ).T.contiguous()  # [12, 10]

    p_mon = torch.tensor(
        [
            [220.0, 160.0, 180.0, 170.0, 170.0, 150.0, 150.0, 160.0, 160.0, 190.0, 250.0, 280.0],  # Singapore
            [2.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 2.0],                          # Sahara
            [80.0, 75.0, 65.0, 65.0, 50.0, 30.0, 15.0, 25.0, 70.0, 110.0, 115.0, 95.0],            # Rome
            [52.0, 56.0, 117.0, 124.0, 138.0, 168.0, 154.0, 168.0, 210.0, 198.0, 93.0, 51.0],      # Tokyo
            [55.0, 40.0, 42.0, 44.0, 50.0, 45.0, 45.0, 50.0, 50.0, 68.0, 60.0, 55.0],              # London
            [46.0, 36.0, 33.0, 38.0, 52.0, 84.0, 90.0, 80.0, 67.0, 66.0, 60.0, 53.0],              # Moscow
            [9.0, 8.0, 7.0, 8.0, 20.0, 35.0, 40.0, 38.0, 30.0, 18.0, 15.0, 11.0],                  # Yakutsk
            [5.0, 4.0, 4.0, 5.0, 5.0, 8.0, 22.0, 25.0, 18.0, 12.0, 7.0, 5.0],                      # Barrow
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],                          # Dome C
            [15.0, 16.0, 20.0, 41.0, 69.0, 93.0, 82.0, 77.0, 40.0, 30.0, 16.0, 17.0],              # Cape Town (SH!)
        ],
        dtype=torch.float32,
        device=device,
    ).T.contiguous()

    lats = torch.deg2rad(torch.tensor([1.3, 24.0, 41.9, 35.7, 51.5, 55.8, 62.0, 71.3, -75.1, -33.9], device=device))
    codes, _ = classify_koppen_geiger(t_mon, p_mon, lats)
    names = [KOPPEN_NAMES[int(c)] for c in codes.cpu().tolist()]
    expected = ["Af", "BWh", "Csa", "Cfa", "Cfb", "Dfb", "Dfd", "ET", "EF", "Csb"]
    assert names == expected, f"Expected {expected}, got {names}"


def test_full_global_coupled_aoba_and_zarr_v3_io(device: str) -> None:
    """Verify end-to-end 1:1 Superparameterized Global AOBA coupling & UGRID Zarr v3 output."""
    cfg = AOBAConfig(
        glevel=2,               # 162 Voronoi cells x (8x8x32 CRM) = 331,776 3D cells per step!
        dt_host=120.0,
        n_host_dyn_substeps=2,
        n_crm_dyn_substeps=4,
        rad_stride=2,
        earth2014_stride=8,
        co2_ppmv=400.0,
    )
    model = init_aoba_model(cfg, device=device)

    for _ in range(3):
        diag = step_aoba_model(model)

    assert 275.0 < diag.global_mean_t_sfc_k < 305.0
    assert 150.0 < diag.global_mean_olr_wm2 < 350.0
    assert diag.global_mean_asr_wm2 > 150.0
    assert diag.land_mean_gpp_umol > 0.5
    assert diag.seaice_area_million_km2 > 0.0
    assert not torch.isnan(model.atmos.theta_m).any()
    assert not torch.isnan(model.atmos.u_edge).any()
    assert model.sfc_sub is not None
    nc = model.mesh.n_cells
    n_sub = cfg.crm_ny * cfg.crm_nx
    assert model.sfc_sub.t_canopy.numel() == nc * n_sub
    # True subgrid MAML + ICA radiation: CRM perturbations induce subcolumn surface & radiative spread
    t_can_sub_spread = float(model.sfc_sub.t_canopy.reshape(nc, n_sub).std(dim=-1).max().item())
    assert t_can_sub_spread > 1.0e-5
    assert torch.allclose(model.sfc.t_canopy, model.sfc_sub.t_canopy.reshape(nc, n_sub).mean(dim=-1), atol=1.0e-5)
    assert model.rad_cache["q_rad_sub"].shape == (cfg.nz, nc * n_sub)

    # Export to CF-1.11 / UGRID-1.0 Zarr v3 and verify round-trip reload
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = Path(tmpdir) / "test_aoba_global.zarr"
        ds_snap = export_aoba_snapshot_xarray(model)
        append_zarr(ds_snap, zpath)
        ds_reloaded = load_zarr(zpath)
        assert ds_reloaded.attrs["Conventions"] == "CF-1.11 UGRID-1.0"
        assert ds_reloaded.sizes["n_cell"] == 162
        assert ds_reloaded.sizes["nz"] == 32
        assert ds_reloaded.sizes["crm_y"] == cfg.crm_ny
        assert ds_reloaded.sizes["crm_x"] == cfg.crm_nx
        assert "koppen_code" in ds_reloaded.data_vars
        assert "lwp_sub" in ds_reloaded.data_vars
        assert "t_sfc_sub" in ds_reloaded.data_vars
        assert "sw1_dir_sub" in ds_reloaded.data_vars


def test_dynamic_sia_icesheet_and_vertical_remap(device: str) -> None:
    """Verify SIA ice volume conservation, EMA SMB + accelerated transition, and vertical remapping."""
    cfg = AOBAConfig(
        glevel=2,
        crm_nx=4,
        crm_ny=1,
        crm_dx=16000.0,
        dt_host=120.0,
        n_host_dyn_substeps=2,
        n_crm_dyn_substeps=2,
        rad_stride=2,
        earth2014_stride=8,
        enable_dynamic_icesheet=True,
        icesheet_ema_years=2.0,
        icesheet_accel_factor=200.0,
        icesheet_accel_spinup_years=10.0,
        icesheet_isostasy_tau_yr=500.0,
    )
    model = init_aoba_model(cfg, device=device)
    assert model.icesheet is not None
    assert model.mesh.z_bed is not None and model.mesh.h_ice_init is not None

    # 1. Closed-sphere SIA flow conservation test (zero ocean calving, zero SMB)
    dry_mesh = model.mesh.to(device)
    dry_mesh.ocean_frac = torch.zeros_like(dry_mesh.ocean_frac)
    dry_ice = IceSheetState(
        h_ice=2500.0 * torch.exp(-((0.5 * math.pi - dry_mesh.lat_cell) / 0.4) ** 2),
        z_bed=torch.zeros_like(dry_mesh.z_topo),
        z_bed_unloaded=torch.zeros_like(dry_mesh.z_topo),
        smb_ema_kgm2_s=torch.zeros_like(dry_mesh.z_topo),
        smb_year_accum_kgm2=torch.zeros_like(dry_mesh.z_topo),
        year_elapsed_sec=C.DAYS_PER_YEAR * C.SECONDS_PER_DAY,
    )
    vol_0 = float((dry_mesh.area_cell * dry_ice.h_ice).sum().item())
    peak_0 = float(dry_ice.h_ice.max().item())
    step_sia_icesheet_scvt(dry_ice, dry_mesh, ema_years=1.0, accel_factor=500.0, isostasy_tau_yr=0.0)
    vol_1 = float((dry_mesh.area_cell * dry_ice.h_ice).sum().item())
    assert abs(vol_1 - vol_0) / max(vol_0, 1.0) < 1.0e-5
    assert (dry_ice.h_ice >= 0.0).all()
    assert float(dry_ice.h_ice.max().item()) < peak_0

    # 2. Full coupled melt/growth transition + Klemp STF height update + barometric remap stability
    ice_cells = torch.where(model.mesh.surf_type == SURF_ICESHEET)[0]
    land_cells = torch.where((model.mesh.surf_type == 1) & (model.mesh.ocean_frac < 0.5))[0]
    assert ice_cells.numel() > 0 and land_cells.numel() > 0
    melt_idx = int(ice_cells[0].item())
    grow_idx = int(land_cells[0].item())

    yr_sec = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    model.icesheet.year_elapsed_sec = yr_sec - cfg.dt_host
    # Impose strong cumulative ablation at melt_idx and accumulation at grow_idx
    model.icesheet.smb_year_accum_kgm2[melt_idx] = -2.0e6
    model.icesheet.smb_year_accum_kgm2[grow_idx] = 2.0e5
    z_w0_before = model.atmos.z_w[0].clone()

    diag = step_aoba_model(model)
    assert model.icesheet.n_updates == 1
    assert int(model.mesh.surf_type[melt_idx].item()) == 1  # Deglaciated -> Land!
    assert int(model.mesh.surf_type[grow_idx].item()) == SURF_ICESHEET  # Glaciated -> IceSheet!
    assert torch.allclose(model.atmos.z_w[0], model.mesh.z_topo_smooth, atol=1.0e-4)
    assert float((model.atmos.z_w[0] - z_w0_before).abs().max().item()) > 1.0

    # Continue stepping coupled physics/dynamics after remapping and verify Zarr snapshot export
    for _ in range(2):
        diag = step_aoba_model(model)
    assert not torch.isnan(model.atmos.theta_m).any()
    assert not torch.isnan(model.crm.theta_m).any()
    assert 270.0 < diag.global_mean_t_sfc_k < 310.0
    ds_ice = export_aoba_snapshot_xarray(model)
    assert "h_ice" in ds_ice.data_vars and "z_topo_dyn" in ds_ice.data_vars


if __name__ == "__main__":
    for dev in _available_devices():
        print(f"Running Phase 2 & Phase 3 Global AOBA tests on device={dev}...")
        test_scvt_mesh_trisk_and_earth2014(dev)
        test_orography_and_scvt_dynamics_conservation(dev)
        test_ekman_ocean_upwelling_and_heat_conservation(dev)
        test_koppen_geiger_beck_2018_table1(dev)
        test_full_global_coupled_aoba_and_zarr_v3_io(dev)
        test_dynamic_sia_icesheet_and_vertical_remap(dev)
    print("ALL PHASE 2 & PHASE 3 GLOBAL AOBA TESTS PASSED!")
