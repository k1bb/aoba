"""Verification tests for AOBA's 3D multi-layer SCVT OGCM and dynamic sea-ice drift.

Validates:
1. Exact global enthalpy and salt conservation over wind-driven multi-step integration
   with GM bolus transport, Redi diffusion, barotropic free surface, and vertical mixing.
2. Exact constant-field preservation (uniform T & S remain constant under wind stress).
3. Nonlinear EOS (Roquet et al., 2015) cabbeling / haline contraction and convective
   adjustment mixing dense cold/salty polar surface water into deep layers.
4. Conservative dynamic sea-ice drift with zero coastline leakage.
5. Selectable `ocean_mode="ogcm_3d"` vs `ocean_mode="ekman_sw"` in `step_aoba_model`
   on both CPU and Apple Silicon `mps`.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.ocean_ogcm import (
    S_REF_PSU,
    compute_seawater_density_anomaly,
    init_ogcm_state,
    step_ogcm_voronoi,
    step_seaice_drift_voronoi,
)
from aoba.scvt_mesh import build_scvt_mesh
from aoba.surface import SURF_LAND, SURF_OCEAN, init_surface_state


def _make_basin_mesh_and_sfc(device: str = "cpu"):
    mesh = build_scvt_mesh(glevel=2, device=device)
    # Create realistic continent + mid-ocean ridge bathymetry for zero-flux coastal checks
    lat = mesh.lat_cell
    lon = mesh.lon_cell
    is_land = (lat > -0.3) & (lat < 0.7) & (lon > -0.6) & (lon < 0.6)
    mesh.surf_type = torch.where(is_land, torch.full_like(mesh.surf_type, SURF_LAND), torch.full_like(mesh.surf_type, SURF_OCEAN))
    mesh.z_bed = torch.where(
        is_land,
        torch.full((mesh.n_cells,), 300.0, device=device),
        -3500.0 + 1500.0 * torch.cos(2.0 * lon),  # 2000..5000 m submarine ridges
    )
    sfc = init_surface_state(mesh.surf_type, t_init=290.0, h_ml_ocean=50.0)
    # Meridional SST gradient (300 K equator -> 274 K poles)
    sfc.t_ocean_ml = torch.clamp(274.0 + 26.0 * (torch.cos(lat) ** 2), min=C.T_FREEZE_OCEAN + 0.5)
    return mesh, sfc


def _total_heat_and_salt(ogcm, mesh) -> tuple[float, float]:
    h_eff = ogcm.h_c.clone()
    h_eff[0] = torch.where(ogcm.wet_c[0], torch.clamp(ogcm.h_c[0] + ogcm.eta, min=10.0), torch.zeros_like(ogcm.eta))
    vol = h_eff * mesh.area_cell.unsqueeze(0)
    heat = float((vol * ogcm.temp).sum().item())
    salt = float((vol * ogcm.sal).sum().item())
    return heat, salt


def test_ogcm_constant_preservation_and_global_conservation() -> None:
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    ogcm = init_ogcm_state(sfc, mesh, nz_ocn=10)
    zeros = torch.zeros(mesh.n_cells)
    # Realistic zonal trade-wind & westerly stress profile
    lat = mesh.lat_cell
    tau_x = 0.12 * torch.cos(3.0 * lat)  # atmospheric upward stress -> -tau_x acts on ocean
    tau_y = 0.03 * torch.sin(2.0 * lat)

    # 1. Exact constant-field preservation under strong wind forcing
    ogcm_const = init_ogcm_state(sfc, mesh, nz_ocn=10)
    ogcm_const.temp = torch.where(ogcm_const.wet_c, torch.full_like(ogcm_const.temp, 286.0), ogcm_const.temp)
    ogcm_const.sal = torch.where(ogcm_const.wet_c, torch.full_like(ogcm_const.sal, 34.7), ogcm_const.sal)
    sfc_const = init_surface_state(mesh.surf_type, t_init=286.0, h_ml_ocean=50.0)
    for _ in range(10):
        step_ogcm_voronoi(
            ogcm_const, sfc_const, mesh, tau_x, tau_y, zeros,
            zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False,
        )
    t_err = float((ogcm_const.temp[ogcm_const.wet_c] - 286.0).abs().max().item())
    s_err = float((ogcm_const.sal[ogcm_const.wet_c] - 34.7).abs().max().item())
    assert t_err < 1.0e-3, f"Constant T drift={t_err}"
    assert s_err < 1.0e-3, f"Constant S drift={s_err}"

    # 2. Exact global heat & salt conservation with 3D stratification, GM bolus, Redi & wind stress
    # Inject a salty tropical anomaly & cold polar anomaly (without surface freezing)
    ogcm.sal[0] = torch.where(ogcm.wet_c[0], 34.7 + 1.0 * torch.cos(lat), ogcm.sal[0])
    h0, s0 = _total_heat_and_salt(ogcm, mesh)
    for _ in range(15):
        step_ogcm_voronoi(
            ogcm, sfc, mesh, tau_x, tau_y, zeros,
            zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False,
        )
    h1, s1 = _total_heat_and_salt(ogcm, mesh)
    rel_h = abs(h1 - h0) / abs(h0)
    rel_s = abs(s1 - s0) / abs(s0)
    assert rel_h < 2.0e-5, f"Heat conservation rel_err={rel_h:.3e}"
    assert rel_s < 2.0e-5, f"Salt conservation rel_err={rel_s:.3e}"
    assert float(ogcm.u_edge.abs().max().item()) > 1.0e-3
    assert torch.isfinite(ogcm.temp).all() and torch.isfinite(ogcm.sal).all()


def test_nonlinear_eos_and_polar_convective_adjustment() -> None:
    # Haline contraction > thermal expansion at freezing polar T (Roquet et al. 2015)
    t_cold = torch.tensor([272.0])
    t_warm = torch.tensor([295.0])
    s_lo = torch.tensor([34.0])
    s_hi = torch.tensor([35.0])
    z0 = torch.tensor([100.0])
    assert float(compute_seawater_density_anomaly(t_cold, s_hi, z0)) > float(compute_seawater_density_anomaly(t_cold, s_lo, z0))
    assert float(compute_seawater_density_anomaly(t_cold, s_lo, z0)) > float(compute_seawater_density_anomaly(t_warm, s_lo, z0))

    # Brine-driven surface densification triggers convective adjustment to deeper layers
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    ogcm = init_ogcm_state(sfc, mesh, nz_ocn=10)
    # Create unstable heavy/salty surface cap at ocean cell 0
    ocn_idx = int(torch.nonzero(ogcm.wet_c[0], as_tuple=False)[0].item())
    ogcm.temp[:, ocn_idx] = 276.0
    ogcm.sal[:, ocn_idx] = S_REF_PSU
    sfc.t_ocean_ml[ocn_idx] = 273.0
    ogcm.sal[0, ocn_idx] = 36.5  # dense brine cap -> drho/dz < 0
    s_deep_before = float(ogcm.sal[1, ocn_idx].item())
    zeros = torch.zeros(mesh.n_cells)
    step_ogcm_voronoi(ogcm, sfc, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False)
    s_deep_after = float(ogcm.sal[1, ocn_idx].item())
    assert s_deep_after > s_deep_before + 0.05, f"Convective adjustment failed: {s_deep_before} -> {s_deep_after}"


def test_dynamic_seaice_drift_conservation() -> None:
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    is_ocean = sfc.surf_type == SURF_OCEAN
    # Place polar sea ice + snow pack on ocean cells with |lat| > 55 deg
    polar = is_ocean & (mesh.lat_cell.abs() > 0.95)
    sfc.m_seaice = torch.where(polar, torch.full_like(sfc.m_seaice, 1200.0), torch.zeros_like(sfc.m_seaice))
    sfc.swe_seaice = torch.where(polar, torch.full_like(sfc.swe_seaice, 150.0), torch.zeros_like(sfc.swe_seaice))

    m0 = float((sfc.m_seaice * mesh.area_cell).sum().item())
    s0 = float((sfc.swe_seaice * mesh.area_cell).sum().item())
    tau_x = torch.full((mesh.n_cells,), -0.15)
    tau_y = torch.full((mesh.n_cells,), 0.08)
    u_ocn = torch.full((mesh.n_edges,), 0.05)

    for _ in range(10):
        step_seaice_drift_voronoi(sfc, mesh, tau_x, tau_y, u_ocn, dt=720.0)

    m1 = float((sfc.m_seaice * mesh.area_cell).sum().item())
    s1 = float((sfc.swe_seaice * mesh.area_cell).sum().item())
    assert abs(m1 - m0) / m0 < 1.0e-5, f"Sea-ice mass drift: {m0} -> {m1}"
    assert abs(s1 - s0) / s0 < 1.0e-5, f"Sea-ice snow drift: {s0} -> {s1}"
    assert float(sfc.m_seaice[~is_ocean].abs().max().item()) == 0.0


def test_selectable_ocean_modes_cpu_and_mps() -> None:
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        for mode in ("ogcm_3d", "ekman_sw"):
            cfg = AOBAConfig(
                glevel=1, nz=12, z_top=24000.0, crm_nx=4, crm_ny=1,
                use_earth2014=False, ocean_mode=mode, ocean_nz=8,
                rad_scheme="ckd", enable_nonoro_gwd=False,
            )
            m = init_aoba_model(cfg, device=dev)
            if mode == "ogcm_3d":
                assert m.ogcm is not None and m.ogcm.nz_ocn == 8
            else:
                assert m.ogcm is None
            for _ in range(3):
                d = step_aoba_model(m)
            assert torch.isfinite(torch.tensor(d.global_mean_t_sfc_k))
            if mode == "ogcm_3d":
                assert torch.isfinite(m.ogcm.temp).all() and torch.isfinite(m.ogcm.sal).all()
                assert m.ogcm.h_bl is not None and torch.isfinite(m.ogcm.h_bl).all()
                assert m.ogcm.kv_last is not None and torch.isfinite(m.ogcm.kv_last).all()


def test_kpp_boundary_layer_and_nonlocal_convection_conservation() -> None:
    """Verify KPP boundary-layer deepening under cooling/wind and exact column heat conservation."""
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    ogcm = init_ogcm_state(sfc, mesh, nz_ocn=12)
    is_ocn = ogcm.wet_c[0]
    zeros = torch.zeros(mesh.n_cells)

    # Set weak upper stratification so strong wind + cooling deepens KPP h_BL beyond Layer 0 (50 m)
    ogcm.temp = torch.where(
        ogcm.wet_c,
        286.0 - 0.002 * ogcm.z_c.unsqueeze(1),
        ogcm.temp,
    )
    sfc.t_ocean_ml = ogcm.temp[0].clone() - 0.15  # Strong surface cooling (B_0 < 0 -> convective!)
    tau_x = torch.full((mesh.n_cells,), -0.25)
    tau_y = torch.zeros(mesh.n_cells)

    # Record heat after surface cooling is applied to Layer 0
    ogcm_check = init_ogcm_state(sfc, mesh, nz_ocn=12)
    ogcm_check.temp = ogcm.temp.clone()
    ogcm_check.temp[0] = torch.where(is_ocn, sfc.t_ocean_ml, ogcm_check.temp[0])
    h0, s0 = _total_heat_and_salt(ogcm_check, mesh)

    step_ogcm_voronoi(
        ogcm, sfc, mesh, tau_x, tau_y, zeros, zeros, zeros, zeros, zeros,
        dt=720.0, enable_seaice_drift=False,
    )
    h1, s1 = _total_heat_and_salt(ogcm, mesh)
    assert abs(h1 - h0) / abs(h0) < 2.0e-5
    assert abs(s1 - s0) / abs(s0) < 2.0e-5
    assert ogcm.h_bl is not None
    # Convective cooling + strong wind must deepen KPP boundary layer well beyond 50 m
    assert float(ogcm.h_bl[is_ocn].mean().item()) > 100.0
    assert ogcm.kv_last is not None
    assert float(ogcm.kv_last[0, is_ocn].mean().item()) > 5.0e-3


def test_richardson_shear_and_topographic_tidal_mixing() -> None:
    """Verify Ri-dependent interior shear mixing & Simmons (2004) topographic internal-tide mixing."""
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    # Build two identical oceans: one with strong vertical shear, one at rest
    ogcm_rest = init_ogcm_state(sfc, mesh, nz_ocn=12)
    ogcm_shear = init_ogcm_state(sfc, mesh, nz_ocn=12)
    # Inject strong upper-thermocline vertical velocity shear (equatorial undercurrent style)
    ogcm_shear.u_edge[2] = 0.60 * ogcm_shear.wet_e[2].to(torch.float32)
    ogcm_shear.u_edge[3] = -0.60 * ogcm_shear.wet_e[3].to(torch.float32)

    zeros = torch.zeros(mesh.n_cells)
    sfc_rest = init_surface_state(mesh.surf_type, t_init=290.0)
    sfc_rest.t_ocean_ml = ogcm_rest.temp[0].clone()
    sfc_shear = init_surface_state(mesh.surf_type, t_init=290.0)
    sfc_shear.t_ocean_ml = ogcm_shear.temp[0].clone()
    step_ogcm_voronoi(ogcm_rest, sfc_rest, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False)
    step_ogcm_voronoi(ogcm_shear, sfc_shear, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros, dt=720.0, enable_seaice_drift=False)

    assert ogcm_rest.kv_last is not None and ogcm_shear.kv_last is not None
    wet23 = ogcm_shear.wet_c[2] & ogcm_shear.wet_c[3]
    kv_rest_23 = float(ogcm_rest.kv_last[2, wet23].mean().item())
    kv_shear_23 = float(ogcm_shear.kv_last[2, wet23].mean().item())
    assert kv_shear_23 > kv_rest_23 * 10.0, f"Shear mixing too weak: rest={kv_rest_23:.2e}, shear={kv_shear_23:.2e}"

    # Topographic internal-tide mixing: steep ridge bathymetry >> flat abyssal plain near seafloor
    mesh_flat = build_scvt_mesh(glevel=2, device="cpu")
    mesh_flat.surf_type = torch.full((mesh_flat.n_cells,), SURF_OCEAN, dtype=torch.int8)
    mesh_flat.z_bed = torch.full((mesh_flat.n_cells,), -4000.0)
    sfc_flat = init_surface_state(mesh_flat.surf_type, t_init=290.0)
    ogcm_flat = init_ogcm_state(sfc_flat, mesh_flat, nz_ocn=12)

    mesh_ridge = build_scvt_mesh(glevel=2, device="cpu")
    mesh_ridge.surf_type = torch.full((mesh_ridge.n_cells,), SURF_OCEAN, dtype=torch.int8)
    mesh_ridge.z_bed = -3800.0 + 1600.0 * torch.sin(6.0 * mesh_ridge.lon_cell)  # Steep submarine ridges
    sfc_ridge = init_surface_state(mesh_ridge.surf_type, t_init=290.0)
    ogcm_ridge = init_ogcm_state(sfc_ridge, mesh_ridge, nz_ocn=12)
    # Give ridge mesh G5-scale bathymetric roughness (0.008) matching Earth2014 Mid-Atlantic Ridge p90
    ogcm_ridge.bathym_slope = torch.clamp(ogcm_ridge.bathym_slope * 8.0, max=0.012)

    z0 = torch.zeros(mesh_flat.n_cells)
    step_ogcm_voronoi(ogcm_flat, sfc_flat, mesh_flat, z0, z0, z0, z0, z0, z0, z0, dt=720.0, enable_seaice_drift=False)
    step_ogcm_voronoi(ogcm_ridge, sfc_ridge, mesh_ridge, z0, z0, z0, z0, z0, z0, z0, dt=720.0, enable_seaice_drift=False)

    assert ogcm_flat.kv_last is not None and ogcm_ridge.kv_last is not None
    wet_deep = ogcm_ridge.wet_c[7] & ogcm_ridge.wet_c[8]
    kv_flat_deep = float(ogcm_flat.kv_last[7, wet_deep].mean().item())
    kv_ridge_deep = float(ogcm_ridge.kv_last[7, wet_deep].mean().item())
    assert kv_ridge_deep > kv_flat_deep * 5.0, f"Tidal mixing failed: flat={kv_flat_deep:.2e}, ridge={kv_ridge_deep:.2e}"


def test_shortwave_penetration_subsurface_warming_and_conservation() -> None:
    """Verify Jerlov Type I SW penetration warms subsurface Layer 1 while strictly conserving column heat."""
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    ogcm_dark = init_ogcm_state(sfc, mesh, nz_ocn=10)
    ogcm_sun = init_ogcm_state(sfc, mesh, nz_ocn=10)
    # Horizontally uniform stably stratified column well above freezing (279..295 K)
    t_prof = 295.0 - 0.003 * ogcm_sun.z_c.unsqueeze(1)
    ogcm_dark.temp = torch.where(ogcm_dark.wet_c, t_prof, ogcm_dark.temp)
    ogcm_sun.temp = ogcm_dark.temp.clone()
    sfc_dark = init_surface_state(mesh.surf_type, t_init=295.0, h_ml_ocean=50.0)
    sfc_dark.t_ocean_ml = ogcm_dark.temp[0].clone()
    sfc_sun = init_surface_state(mesh.surf_type, t_init=295.0, h_ml_ocean=50.0)
    sfc_sun.t_ocean_ml = ogcm_sun.temp[0].clone()
    h0, _ = _total_heat_and_salt(ogcm_sun, mesh)

    zeros = torch.zeros(mesh.n_cells)
    sw_net = torch.full((mesh.n_cells,), 300.0)  # 300 W/m^2 open-water solar absorption
    step_ogcm_voronoi(
        ogcm_dark, sfc_dark, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
        dt=3600.0, enable_seaice_drift=False, sw_net_ocn=None,
    )
    step_ogcm_voronoi(
        ogcm_sun, sfc_sun, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
        dt=3600.0, enable_seaice_drift=False, sw_net_ocn=sw_net,
    )
    h1, _ = _total_heat_and_salt(ogcm_sun, mesh)
    assert abs(h1 - h0) / abs(h0) < 1.0e-6, f"Penetrative SW broke enthalpy conservation: {h0} -> {h1}"
    # Layer 1 received transmitted solar radiation (> dark control) while Layer 0 transferred it (< dark control)
    wet1 = ogcm_sun.wet_c[1]
    is_ocn = ogcm_sun.wet_c[0]
    assert float((ogcm_sun.temp[1, wet1] - ogcm_dark.temp[1, wet1]).mean().item()) > 1.0e-4
    assert float((ogcm_sun.temp[0, is_ocn] - ogcm_dark.temp[0, is_ocn]).mean().item()) < -1.0e-4


def test_full_redi_suppresses_veronis_spurious_mixing() -> None:
    """Verify full rotated Redi tensor suppresses spurious cross-isopycnal drift vs z-level diffusion."""
    import aoba.ocean_ogcm as ogcm_mod

    mesh = build_scvt_mesh(glevel=2, device="cpu")
    mesh.surf_type = torch.full((mesh.n_cells,), SURF_OCEAN, dtype=torch.int8)
    mesh.z_bed = torch.full((mesh.n_cells,), -4000.0)
    sfc = init_surface_state(mesh.surf_type, t_init=285.0)
    lat = mesh.lat_cell

    def _run_front(full_redi: bool) -> tuple[float, float]:
        old_flag = ogcm_mod.ENABLE_FULL_REDI
        old_gm = ogcm_mod.KAPPA_GM
        old_redi = ogcm_mod.KAPPA_REDI_HORIZ
        try:
            ogcm_mod.ENABLE_FULL_REDI = full_redi
            ogcm_mod.KAPPA_GM = 0.0  # Isolate pure Redi tensor from GM bolus
            ogcm_mod.KAPPA_REDI_HORIZ = 1.0e5  # Strong diffusivity to separate Redi from weak baroclinic advection
            ogcm = init_ogcm_state(sfc, mesh, nz_ocn=10)
            # Slanted isopycnal front: T(z, lat) with uniform slope well below GM_SLOPE_MAX
            ogcm.temp = 292.0 - 0.004 * ogcm.z_c.unsqueeze(1) + 4.0 * torch.cos(lat).unsqueeze(0)
            sfc_loc = init_surface_state(mesh.surf_type, t_init=285.0)
            sfc_loc.t_ocean_ml = ogcm.temp[0].clone()
            h_init, s_init = _total_heat_and_salt(ogcm, mesh)
            var0 = float(ogcm.temp[4].var().item())
            zeros = torch.zeros(mesh.n_cells)
            for _ in range(8):
                ogcm.u_edge.zero_()
                ogcm.u_baro.zero_()
                step_ogcm_voronoi(ogcm, sfc_loc, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros, dt=1800.0, enable_seaice_drift=False)
            h_end, s_end = _total_heat_and_salt(ogcm, mesh)
            assert abs(h_end - h_init) / abs(h_init) < 2.0e-5
            assert abs(s_end - s_init) / abs(s_init) < 2.0e-5
            var1 = float(ogcm.temp[4].var().item())
            return abs(var1 - var0), float(ogcm.kv_last[4].mean().item())
        finally:
            ogcm_mod.ENABLE_FULL_REDI = old_flag
            ogcm_mod.KAPPA_GM = old_gm
            ogcm_mod.KAPPA_REDI_HORIZ = old_redi

    drift_redi, kv_redi = _run_front(True)
    drift_zlev, kv_zlev = _run_front(False)
    # Full Redi cuts spurious Veronis smearing nearly in half even alongside baroclinic advection
    assert drift_redi < drift_zlev * 0.60, f"Full Redi did not suppress Veronis drift: redi={drift_redi:.4e}, zlev={drift_zlev:.4e}"
    assert kv_redi > kv_zlev  # Implicit vertical Redi S^2 diagonal term active


def test_ogcm_zero_forcing_float32_conservation() -> None:
    """Verify multi-step float32 abyssal vertical mixing + Jerlov SW cause < 1e-4 K/yr & < 1e-5 psu/yr drift."""
    mesh, sfc = _make_basin_mesh_and_sfc("cpu")
    ogcm = init_ogcm_state(sfc, mesh, nz_ocn=18)
    zeros = torch.zeros(mesh.n_cells)
    sw_const = torch.full((mesh.n_cells,), 200.0)

    def _mean_ts() -> tuple[float, float]:
        heff = ogcm.h_c.clone()
        heff[0] = torch.where(ogcm.wet_c[0], torch.clamp(ogcm.h_c[0] + ogcm.eta, min=10.0), torch.zeros_like(ogcm.eta))
        vol = (heff * mesh.area_cell.unsqueeze(0)).double()
        vsum = vol.sum().item()
        t_mean = (vol * (ogcm.temp.double() + ogcm.t_comp.double())).sum().item() / vsum
        s_mean = (vol * (ogcm.sal.double() + ogcm.s_comp.double())).sum().item() / vsum
        return t_mean, s_mean

    t0, s0 = _mean_ts()
    n_steps = 120  # 1 full simulated day at dt = 720 s
    for _ in range(n_steps):
        step_ogcm_voronoi(
            ogcm, sfc, mesh, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
            dt=720.0, enable_seaice_drift=False, sw_net_ocn=sw_const,
        )
    t1, s1 = _mean_ts()
    dt_per_yr = abs(t1 - t0) * 365.0
    ds_per_yr = abs(s1 - s0) * 365.0
    assert dt_per_yr < 1.0e-4, f"Float32 OGCM enthalpy drift too large: {dt_per_yr:.3e} K/yr"
    assert ds_per_yr < 1.0e-5, f"Float32 OGCM salinity drift too large: {ds_per_yr:.3e} psu/yr"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
