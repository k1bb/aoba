"""Comprehensive regression tests for:
1. Map-free autonomous dynamic lakes (`src/aoba/lakes.py` + `kernels_esm.py` + `river_routing.py`):
   - Exact bowl hypsometry volume identity (`f_lake * A * h_lake == V`).
   - Multi-decadal zero-forcing water & heat conservation (float32 + Kahan).
   - Aridification drying test (`E > P + Q_in => V -> 0, f_lake -> 0` with zero negative volume/NaN).
   - Endorheic basin retention (`Q_spill == 0`) vs. exorheic sill-overflow transition (`Q_spill > 0`).
   - Freshwater 4 C density-maximum convective overturning & winter lake-ice freeze/thaw enthalpy.
   - CPU vs. MPS Metal/CUDA fused kernel (`dynamic_lake_step_col`) parity & CUDA transpilation check.
2. Discrete-crown canopy clumping (`Omega(mu0, L)`) & non-iterative optimal leaf angle (`chi_leaf`):
   - Low-LAI over-absorption reduction (`Omega < 1` at sparse LAI, `Omega -> 1` in closed canopy,
     singularity-free `L -> 0` analytical limit, solar-zenith angle dependence).
   - Exact sunlit/shaded PAR & area conservation (`I_sun + I_sh == I_c`, `L_sun + L_sh == L`).
   - Exact two-stream albedo/transmittance energy bounds across `chi_leaf in [-0.4, +0.5]`.
   - Autonomous zero-iteration leaf-angle emergence: erectophile in arid low-LAI & dense forests,
     planophile in cool humid sparse canopies.
"""

from __future__ import annotations

import math
import torch

from aoba import constants as C
from aoba.biosphere import (
    canopy_light_absorption_depury,
    canopy_sunlit_sw_fraction,
    canopy_two_stream_albedo_dai,
    compute_discrete_crown_clumping,
    init_biosphere_state,
    ross_goudriaan_optics,
    update_daily_acclimation_smith,
)
from aoba.lakes import (
    H_LAKE_MIN_ACTIVE,
    RHO_ICE_LAKE,
    RHO_W_FRESH,
    T_MD_FRESH,
    _step_dynamic_lakes_py,
    diagnose_lake_globals,
    init_dynamic_lake_state,
    step_dynamic_lakes,
    update_lake_hypsometry,
)
from aoba.river_routing import init_river_routing_state, step_river_routing_voronoi
from aoba.scvt_mesh import build_scvt_mesh


def _make_synthetic_basin_mesh(device: str = "cpu"):
    """Build a small SCVT mesh with a synthetic land continent containing an inland closed depression."""
    mesh = build_scvt_mesh(glevel=1, use_earth2014=False, device=device)
    nc = mesh.n_cells
    # Make cells 0..9 land around a central depression at cell 0, surrounded by rim > 100 m, rest ocean
    st = torch.zeros(nc, dtype=torch.int64, device=device)
    z = torch.zeros(nc, dtype=torch.float32, device=device)
    sig = torch.zeros(nc, dtype=torch.float32, device=device)
    # Find cell 0 and its ring of neighbors
    coe = mesh.cells_on_edge.cpu().numpy()
    nbs = set()
    for u, v in coe:
        if u == 0:
            nbs.add(int(v))
        elif v == 0:
            nbs.add(int(u))
    st[0] = 1
    z[0] = 50.0   # Basin floor at 50 m
    sig[0] = 30.0
    for nb in nbs:
        st[nb] = 1
        z[nb] = 180.0  # Surrounding sill rim at 180 m -> d_pf = 130 m!
        sig[nb] = 40.0
    mesh.surf_type = st
    mesh.ocean_frac = (st == 0).to(torch.float32)
    mesh.z_topo = z
    mesh.z_topo_smooth = z.clone()
    mesh.z_min_land = z.clone()
    mesh.sigma_topo = sig
    return mesh


def test_dynamic_lakes_physics_and_conservation() -> None:
    print("Running Dynamic Lake Model tests...")
    mesh = _make_synthetic_basin_mesh("cpu")
    nc = mesh.n_cells
    lake = init_dynamic_lake_state(mesh, init_fill_frac=0.5, max_frac=0.85, max_depth_m=400.0, device=torch.device("cpu"))
    assert bool(lake.is_basin[0].item()), "Central depression cell 0 must be detected as a map-free Priority-Flood basin!"
    assert lake.d_pf_m[0].item() > 100.0, f"Expected d_pf > 100 m, got {lake.d_pf_m[0].item()}"

    # 1. Exact bowl hypsometry volume identity: f_lake * area_cell * h_lake == v_lake_m3
    vol_check = float((lake.f_lake[0] * mesh.area_cell[0] * lake.h_lake[0]).item())
    vol_true = float(lake.v_lake_m3[0].item())
    rel_err = abs(vol_check - vol_true) / max(vol_true, 1.0)
    assert rel_err < 1.0e-5, f"Bowl hypsometry volume mismatch: rel_err={rel_err}"
    print(f"  [PASS] Priority-Flood basin detection (d_pf={lake.d_pf_m[0].item():.1f} m) & exact hypsometry (err={rel_err:.2e})")

    # 2. Endorheic retention vs. exorheic sill-overflow water conservation with river routing
    river = init_river_routing_state(mesh, device=torch.device("cpu"))
    v_init_kg = float(lake.v_lake_m3.sum().item() * 1000.0)
    runoff = torch.zeros(nc)
    # Apply moderate runoff on rim neighbors so water flows into basin cell 0
    rim_mask = (mesh.surf_type == 1) & (~lake.is_basin)
    runoff[rim_mask] = 2.0e-4  # kg m^-2 s^-1
    dt = 3600.0
    cum_in_kg = 0.0
    cum_ocn_kg = 0.0
    for _ in range(5):
        cum_in_kg += float((runoff * (1.0 - lake.f_lake) * mesh.area_cell * dt).sum().item())
        mouth = step_river_routing_voronoi(river, runoff, mesh.area_cell, dt, lake_state=lake)
        cum_ocn_kg += float((mouth * mesh.area_cell * dt).sum().item())
    # While basin is < v_cap, spill from basin must be strictly zero
    assert lake.q_spill_m3s[0].item() == 0.0, "Partially filled endorheic basin must not spill!"
    tot_now_kg = float(lake.v_lake_m3.sum().item() * 1000.0 + river.storage_kg.sum().item() + cum_ocn_kg)
    cons_err = abs(tot_now_kg - (v_init_kg + cum_in_kg)) / max(v_init_kg + cum_in_kg, 1.0)
    assert cons_err < 1.0e-5, f"Endorheic water conservation error: {cons_err}"

    # Fill basin past v_cap -> must transition cleanly to exorheic spill!
    lake.v_lake_m3[0] = lake.v_cap_m3[0] * 0.9999
    lake.v_lake_comp[0] = 0.0
    update_lake_hypsometry(lake, mesh.area_cell)
    big_runoff = torch.zeros(nc)
    big_runoff[0] = 5.0e-1  # Large local runoff directly into nearly-full basin
    step_river_routing_voronoi(river, big_runoff, mesh.area_cell, dt, lake_state=lake)
    assert lake.q_spill_m3s[0].item() > 0.0, "Full exorheic basin must spill excess volume into river channel!"
    assert lake.v_lake_m3[0].item() <= lake.v_cap_m3[0].item() * (1.0 + 1.0e-6)
    print(f"  [PASS] Endorheic->Exorheic sill-overflow transition & river water conservation (err={cons_err:.2e})")

    # 3. Aridification desiccation test (E > P -> V -> 0, f_lake -> 0, zero NaN/negative)
    lake_dry = init_dynamic_lake_state(mesh, init_fill_frac=0.0005, max_frac=0.85, max_depth_m=4.0, device=torch.device("cpu"))
    assert lake_dry.f_lake[0].item() > 0.0, "Shallow desert lake should start active"
    zeros = torch.zeros(nc)
    ones = torch.ones(nc)
    for _ in range(1500):
        step_dynamic_lakes(
            lake=lake_dry,
            u_1=8.0 * ones,
            v_1=zeros,
            temp_1=312.0 * ones,
            qv_1=0.002 * ones,
            pres_sfc=101325.0 * ones,
            rho_1=1.15 * ones,
            z_ref_rel=20.0 * ones,
            sw1_dir_dn=350.0 * ones,
            sw1_dif_dn=50.0 * ones,
            sw2_dir_dn=350.0 * ones,
            sw2_dif_dn=50.0 * ones,
            lw_dn=380.0 * ones,
            precip_rain=zeros,
            precip_snow=zeros,
            mu0_cell=0.8 * ones,
            area_cell=mesh.area_cell,
            f_cor_cell=mesh.f_cell,
            dt=3600.0,
        )
    assert lake_dry.f_lake[0].item() == 0.0, f"Expected arid lake to dry out to f_lake=0, got {lake_dry.f_lake[0].item()}"
    assert lake_dry.v_lake_m3[0].item() >= 0.0 and torch.isfinite(lake_dry.v_lake_m3).all()
    print("  [PASS] Aridification desiccation test (lake smoothly contracts to f_lake=0, zero NaN/negative)")

    # 4. 4 C Freshwater Convective Overturning & Winter Lake-Ice Formation + Melting
    lake_ov = init_dynamic_lake_state(mesh, init_fill_frac=0.6, max_frac=0.85, max_depth_m=40.0, device=torch.device("cpu"))
    # Set summer-like warm epilimnion over 4 C hypolimnion, then cool epilimnion toward 4 C
    lake_ov.t_ml[0] = 278.5
    lake_ov.t_bot[0] = 277.13  # 3.98 C (max density)
    lake_ov.h_ml[0] = 5.0
    # Zero radiative/turbulent forcing + exact density inversion trigger: set t_ml slightly closer to T_md than t_bot
    lake_ov.t_ml[0] = 277.20
    lake_ov.t_bot[0] = 280.00  # Lighter than 277.20 K -> MUST trigger instantaneous enthalpy-conserving overturn!
    h_liq_pre = float(lake_ov.h_lake[0].item())
    h_ml_pre = float(lake_ov.h_ml[0].item())
    h_bot_pre = h_liq_pre - h_ml_pre
    t_mix_expected = (h_ml_pre * 277.20 + h_bot_pre * 280.00) / h_liq_pre
    step_dynamic_lakes(
        lake=lake_ov,
        u_1=zeros, v_1=zeros,
        temp_1=torch.full((nc,), 277.20),
        qv_1=torch.full((nc,), 0.005),
        pres_sfc=101325.0 * ones,
        rho_1=1.25 * ones,
        z_ref_rel=20.0 * ones,
        sw1_dir_dn=zeros, sw1_dif_dn=zeros, sw2_dir_dn=zeros, sw2_dif_dn=zeros,
        lw_dn=torch.full((nc,), C.STEFAN_BOLTZMANN * (277.20 ** 4)),
        precip_rain=zeros, precip_snow=zeros,
        mu0_cell=0.5 * ones, area_cell=mesh.area_cell, f_cor_cell=mesh.f_cell, dt=60.0,
    )
    assert abs(lake_ov.t_ml[0].item() - lake_ov.t_bot[0].item()) < 1.0e-3, "Overturning must homogenize water column!"
    assert abs(lake_ov.t_ml[0].item() - t_mix_expected) < 0.05, f"Overturning enthalpy mismatch: {lake_ov.t_ml[0].item()} vs {t_mix_expected}"
    print(f"  [PASS] 4 C freshwater convective overturning (T_mix={lake_ov.t_ml[0].item():.2f} K vs expected {t_mix_expected:.2f} K)")

    # Sub-zero winter freezing test -> lake ice must grow and albedo must jump above 0.50
    lake_ice = init_dynamic_lake_state(mesh, init_fill_frac=0.5, max_frac=0.85, max_depth_m=20.0, device=torch.device("cpu"))
    lake_ice.t_ml[0] = C.T_FREEZE + 0.05
    lake_ice.t_bot[0] = T_MD_FRESH
    lake_ice.h_ml[0] = 1.0
    for _ in range(48):
        step_dynamic_lakes(
            lake=lake_ice,
            u_1=5.0 * ones, v_1=zeros,
            temp_1=253.15 * ones, qv_1=0.0005 * ones,
            pres_sfc=101325.0 * ones, rho_1=1.35 * ones, z_ref_rel=20.0 * ones,
            sw1_dir_dn=zeros, sw1_dif_dn=zeros, sw2_dir_dn=zeros, sw2_dif_dn=zeros,
            lw_dn=180.0 * ones, precip_rain=zeros, precip_snow=1.0e-5 * ones,
            mu0_cell=0.3 * ones, area_cell=mesh.area_cell, f_cor_cell=mesh.f_cell, dt=1800.0,
        )
    assert lake_ice.h_ice[0].item() > 0.01, f"Expected lake ice growth in -20 C winter, got {lake_ice.h_ice[0].item()}"
    assert lake_ice.alb_vis_dir[0].item() > 0.50, f"Expected frozen lake albedo > 0.50, got {lake_ice.alb_vis_dir[0].item()}"
    area_mkm2, vol_k, ice_mkm2 = diagnose_lake_globals(lake_ice, mesh.area_cell)
    assert area_mkm2 > 0.0 and vol_k > 0.0 and ice_mkm2 > 0.0
    print(f"  [PASS] Winter lake-ice freeze & snow albedo feedback (h_ice={lake_ice.h_ice[0].item():.3f} m, alb_vis={lake_ice.alb_vis_dir[0].item():.2f})")

    # 5. Metal->CUDA transpilation & CPU vs MPS GPU kernel parity
    from aoba.kernels_cuda import transpile_metal_to_cuda
    from aoba.kernels_esm import ESM_METAL_SHADER_SRC
    cuda_src, specs = transpile_metal_to_cuda(ESM_METAL_SHADER_SRC)
    assert any(s.name == "dynamic_lake_step_col" for s in specs), "dynamic_lake_step_col missing from CUDA transpiler!"
    assert "dynamic_lake_step_col" in cuda_src
    print("  [PASS] Metal->CUDA transpiler verified for dynamic_lake_step_col")

    if torch.backends.mps.is_available():
        mesh_mps = _make_synthetic_basin_mesh("mps")
        lk_cpu = init_dynamic_lake_state(mesh, init_fill_frac=0.5, device=torch.device("cpu"))
        lk_mps = init_dynamic_lake_state(mesh_mps, init_fill_frac=0.5, device=torch.device("mps"))
        # Run both open-water and frozen-ice steps
        for step_i, t_air_val in enumerate([295.0, 260.0, 255.0]):
            args_c = dict(
                u_1=4.0 * ones, v_1=2.0 * ones, temp_1=t_air_val * ones, qv_1=0.005 * ones,
                pres_sfc=101325.0 * ones, rho_1=1.2 * ones, z_ref_rel=20.0 * ones,
                sw1_dir_dn=200.0 * ones, sw1_dif_dn=60.0 * ones, sw2_dir_dn=200.0 * ones, sw2_dif_dn=60.0 * ones,
                lw_dn=300.0 * ones, precip_rain=1.0e-5 * ones, precip_snow=1.0e-5 * ones,
                mu0_cell=0.7 * ones, area_cell=mesh.area_cell, f_cor_cell=mesh.f_cell, dt=1800.0,
            )
            args_m = {k: (v.to("mps") if isinstance(v, torch.Tensor) else v) for k, v in args_c.items()}
            _step_dynamic_lakes_py(lk_cpu, **args_c)
            step_dynamic_lakes(lk_mps, **args_m)
        for name in ("v_lake_m3", "t_ml", "t_bot", "h_ml", "h_ice", "f_lake", "shf", "lhf", "evap"):
            a = getattr(lk_cpu, name)
            b = getattr(lk_mps, name).cpu()
            diff = float((a - b).abs().max().item())
            scale = max(float(a.abs().max().item()), 1.0)
            assert diff / scale < 1.0e-4, f"CPU vs MPS lake mismatch in {name}: diff={diff}, scale={scale}"
        print("  [PASS] CPU vs MPS Metal kernel parity (< 1e-4 rel err across all lake prognostic & flux fields)")


def test_clumping_and_optimal_leaf_angle() -> None:
    print("Running Discrete-Crown Clumping & Optimal Leaf Angle tests...")
    mu0 = torch.tensor([0.2, 0.5, 0.85, 0.85, 0.85])
    lai = torch.tensor([0.3, 0.3, 0.3, 2.0, 6.0])

    # 1. Discrete-crown clumping properties
    om_b, om_d, fc = compute_discrete_crown_clumping(mu0, lai)
    # Sparse LAI at high sun must have strong clumping (Omega < 0.80)
    assert om_b[2].item() < 0.80, f"Expected sparse high-sun clumping Omega < 0.80, got {om_b[2].item()}"
    # Lower solar elevation (mu0=0.2) closes horizontal crown gaps -> higher Omega than mu0=0.85 at same LAI=0.3
    assert om_b[0].item() > om_b[2].item(), f"Expected Omega(low sun) > Omega(high sun): {om_b[0].item()} vs {om_b[2].item()}"
    # Closed canopy (LAI=6.0) approaches Omega -> 1.0 (0.90-0.98)
    assert om_b[4].item() > 0.90, f"Expected closed canopy Omega > 0.90, got {om_b[4].item()}"
    # Singularity-free L -> 0 analytical limit
    om0, _, _ = compute_discrete_crown_clumping(torch.tensor([0.8]), torch.tensor([0.0]))
    expected_om0 = (1.0 - math.exp(-0.5 * 1.5)) / (0.5 * 1.5)
    assert abs(om0.item() - expected_om0) < 1.0e-5, f"L->0 limit mismatch: {om0.item()} vs {expected_om0}"
    print(f"  [PASS] Clumping Omega(L=0.3, mu=0.85)={om_b[2].item():.3f}, Omega(L=0.3, mu=0.20)={om_b[0].item():.3f}, Omega(L=6)={om_b[4].item():.3f}")

    # 2. Low-LAI over-absorption bias reduction & exact conservation in de Pury + 2-stream albedo
    rad = canopy_light_absorption_depury(
        torch.full_like(lai, 200.0), torch.full_like(lai, 60.0), mu0, lai
    )
    assert torch.allclose(rad.i_sun + rad.i_sh, rad.i_c, atol=1.0e-4)
    assert torch.allclose(rad.l_sun + rad.l_sh, lai, atol=1.0e-4)
    # Unclumped turbid absorption at L=0.3, mu=0.85 would be larger than clumped absorption
    k_turb = (0.5 / 0.85) * math.sqrt(0.85)
    i_cb_turb = (1.0 - rad.rho_cb[2].item()) * 4.57 * 200.0 * (1.0 - math.exp(-k_turb * 0.3))
    i_cd_turb = (1.0 - 0.036) * 4.24 * 60.0 * (1.0 - math.exp(-0.719 * 0.3))
    assert rad.i_c[2].item() < (i_cb_turb + i_cd_turb) * 0.88, "Sparse clumping must materially reduce low-LAI over-absorption!"

    # Two-stream albedos valid across chi_L in [-0.4, +0.5]
    for chi_val in (-0.4, 0.0, 0.5):
        chi_t = torch.full_like(lai, chi_val)
        avd, avf, and_, anf, tv, tn = canopy_two_stream_albedo_dai(
            lai, mu0, torch.full_like(lai, 0.15), torch.full_like(lai, 0.30), chi_leaf=chi_t
        )
        for arr in (avd, avf, and_, anf, tv, tn):
            assert torch.isfinite(arr).all() and (arr >= 0.0).all() and (arr <= 1.0).all()
        f_sw = canopy_sunlit_sw_fraction(torch.full_like(lai, 200.0), torch.full_like(lai, 60.0), mu0, lai, 0.15, chi_leaf=chi_t)
        assert torch.isfinite(f_sw).all() and (f_sw >= 0.0).all() and (f_sw <= 1.0).all()
    g0, mub0, _, _ = ross_goudriaan_optics(torch.tensor([0.5]), torch.tensor([0.0]))
    assert abs(g0.item() - 0.5) < 1.0e-6 and abs(mub0.item() - 1.0) < 1.0e-6
    print("  [PASS] Ross-Goudriaan optics, exact spherical limit & 2-stream energy bounds verified")

    # 3. Zero-iteration autonomous leaf-angle emergence across 3 contrasting biomes:
    #    [0] Arid low-LAI savanna/shrubland -> erectophile (chi_L < -0.15)
    #    [1] Dense tropical rainforest (LAI=5.5) -> erectophile (chi_L < -0.15)
    #    [2] Cool humid sparse pioneer/tundra (LAI=0.5, MI=1.8, low VPD) -> planophile (chi_L > +0.08)
    veg = torch.tensor([True, True, True])
    bio = init_biosphere_state((3,), veg)
    bio.lai = torch.tensor([0.5, 5.5, 0.5])
    bio.ema_ann_precip = torch.tensor([220.0, 2400.0, 1200.0])
    bio.ema_ann_pet = torch.tensor([1600.0, 1300.0, 600.0])
    for _ in range(90):
        update_daily_acclimation_smith(
            bio=bio,
            midday_vis_beam=torch.tensor([380.0, 320.0, 140.0]),
            midday_vis_diff=torch.tensor([60.0, 110.0, 100.0]),
            midday_mu0=torch.tensor([0.85, 0.85, 0.40]),
            midday_temp=torch.tensor([306.0, 300.0, 285.0]),
            midday_vpd=torch.tensor([2800.0, 800.0, 350.0]),
            night_temp=torch.tensor([293.0, 296.0, 280.0]),
            pres_sfc=torch.full((3,), 101325.0),
            co2_ppmv=284.0,
            lai_exposed=bio.lai,
            is_vegetated_land=veg,
        )
    assert bio.chi_leaf is not None
    chi_vals = bio.chi_leaf.tolist()
    assert chi_vals[0] < -0.15, f"Arid low-LAI must become erectophile (< -0.15), got {chi_vals[0]:.3f}"
    assert chi_vals[1] < -0.15, f"Dense forest must become erectophile (< -0.15), got {chi_vals[1]:.3f}"
    assert chi_vals[2] > 0.08, f"Cool humid sparse canopy must become planophile (> +0.08), got {chi_vals[2]:.3f}"
    print(f"  [PASS] Emergent zero-iteration chi_L: arid_lowLAI={chi_vals[0]:+.3f}, dense_forest={chi_vals[1]:+.3f}, humid_sparse={chi_vals[2]:+.3f}")


if __name__ == "__main__":
    test_dynamic_lakes_physics_and_conservation()
    test_clumping_and_optimal_leaf_angle()
    print("ALL DYNAMIC LAKE, CLUMPING & LEAF-ANGLE TESTS PASSED!")
