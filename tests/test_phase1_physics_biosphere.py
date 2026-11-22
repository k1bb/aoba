"""Unit and physical verification tests for microphysics, turbulence, biosphere, and surface."""

from __future__ import annotations

import math
from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba import constants as C
from aoba.biosphere import (
    COST_C_SMITH,
    THETA_NRH,
    allocate_two_leaf_capacity,
    canopy_light_absorption_depury,
    compute_two_leaf_gas_exchange,
    init_biosphere_state,
    lambert_w0,
    non_rectangular_hyperbola,
    phi_psii_temperature,
    smith_omega,
    update_daily_acclimation_smith,
    update_daily_lai_root_height,
)
from aoba.microphysics import step_microphysics
from aoba.surface import (
    SURF_ICESHEET,
    SURF_LAND,
    SURF_OCEAN,
    compute_ocean_albedo_briegleb,
    compute_surface_albedos,
    init_surface_state,
    step_multilayer_soil_thermodynamics,
    step_surface,
)
from aoba.thermodynamics import rcemip_analytic_sounding
from aoba.turbulence import (
    canopy_roughness_raupach,
    compute_smagorinsky_lilly_3d,
    compute_surface_transfer,
    ocean_roughness_miller,
    step_mynn_level25,
)
from aoba.vertical_grid import build_vertical_grid, compute_stf_heights


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def test_microphysics_water_conservation_and_phase_partition(device: str) -> None:
    """Verify total column water conservation and warm-rain vs cold-snow fallout."""
    vgrid = build_vertical_grid(nz=32, device=device)
    z_datum = torch.zeros(2, device=device)
    z_w, z_c, dz_c = compute_stf_heights(vgrid, z_datum)
    snd = rcemip_analytic_sounding(z_c, z_w, sst=300.0)

    rho_d = snd["rho_d_c"].clone()
    theta_m = snd["theta_m_c"].clone()
    qv0 = snd["qv_c"].clone()
    qn = qv0.clone()
    qp = torch.zeros_like(qn)

    # Column 0: inject warm low-level supersaturation (z ~ 1.5 km, T > 285 K) -> liquid rain
    qn[5:9, 0] = qn[5:9, 0] + 4.0e-3
    # Column 1: inject cold upper-tropospheric supersaturation (z ~ 9-11 km, T < 245 K) -> snow
    qn[20:24, 1] = qn[20:24, 1] + 1.5e-3

    dt = 15.0
    init_col_water = (rho_d * (qn + qp) * dz_c).sum(dim=0)
    total_precip = torch.zeros(2, device=device)

    qv_cur = qv0.clone()
    for _ in range(20):
        res = step_microphysics(rho_d, theta_m, qn, qp, qv_cur, dz_c, dt=dt)
        theta_m, qn, qp, qv_cur = res.theta_m_c, res.q_n, res.q_p, res.q_v
        total_precip = total_precip + (res.precip_rain + res.precip_snow) * dt

    final_col_water = (rho_d * (qn + qp) * dz_c).sum(dim=0) + total_precip
    rel_err = torch.abs(final_col_water - init_col_water) / init_col_water
    assert float(rel_err.max().item()) < 1.0e-5
    assert float(res.precip_rain[0].item()) > 0.0
    assert float(res.precip_snow[0].item()) == 0.0


def test_rsl_and_mynn25_conservation(device: str) -> None:
    """Verify De Ridder (2010) RSL enhancement and MYNN Level 2.5 exact flux conservation."""
    # 1. RSL vs non-RSL transfer over a 25 m forest canopy (LAI = 5)
    hc = torch.tensor([25.0], device=device)
    lai = torch.tensor([5.0], device=device)
    d, z0m, z0h = canopy_roughness_raupach(hc, lai)
    z_ref = torch.tensor([30.0], device=device)
    u1 = torch.tensor([4.0], device=device)
    v1 = torch.tensor([0.0], device=device)
    thv1 = torch.tensor([299.0], device=device)
    thvs = torch.tensor([301.0], device=device)

    tr_no_rsl = compute_surface_transfer(u1, v1, thv1, thvs, z_ref, z0m, z0h, z0h, rsl_depth=None)
    tr_rsl = compute_surface_transfer(u1, v1, thv1, thvs, z_ref, z0m, z0h, z0h, rsl_depth=2.0 * hc - d)
    assert float(tr_rsl.c_m.item()) > float(tr_no_rsl.c_m.item())
    assert float(tr_rsl.c_h.item()) > float(tr_no_rsl.c_h.item())

    # 2. Ocean smooth-viscous + Charnock roughness minimum at moderate u_*
    z0m_weak, _, _ = ocean_roughness_miller(torch.tensor([0.03], device=device))
    z0m_mod, _, _ = ocean_roughness_miller(torch.tensor([0.12], device=device))
    z0m_gale, _, _ = ocean_roughness_miller(torch.tensor([1.00], device=device))
    assert float(z0m_weak.item()) > float(z0m_mod.item())
    assert float(z0m_gale.item()) > float(z0m_mod.item())

    # 3. MYNN Level 2.5 column moisture & momentum conservation on batch of 16 columns
    vgrid = build_vertical_grid(nz=32, device=device)
    z_w, z_c, dz_c = compute_stf_heights(vgrid, torch.zeros(16, device=device))
    snd = rcemip_analytic_sounding(z_c, z_w, sst=300.0)

    rho_d = snd["rho_d_c"]
    u = torch.full_like(rho_d, 6.0)
    v = torch.full_like(rho_d, -2.0)
    thm = snd["theta_m_c"]
    qn = snd["qv_c"]
    qc = torch.zeros_like(qn)
    qi = torch.zeros_like(qn)
    qke = torch.full_like(qn, 0.10)

    ustar = torch.full((16,), 0.35, device=device)
    b_flux = torch.full((16,), 2.0e-3, device=device)
    taux = torch.full((16,), -0.12, device=device)
    tauy = torch.full((16,), 0.04, device=device)
    shf = torch.full((16,), 60.0, device=device)
    evap = torch.full((16,), 5.0e-5, device=device)
    dt = 15.0

    m0_u = (rho_d * u * dz_c).sum(dim=0)
    m0_q = (rho_d * qn * dz_c).sum(dim=0)

    u_n, v_n, thm_n, qn_n, qke_n, km, kh = step_mynn_level25(
        rho_d, u, v, thm, qn, qn, qc, qi, qke, z_c, z_w, dz_c,
        ustar, b_flux, taux, tauy, shf, evap, dt=dt,
    )
    m1_u = (rho_d * u_n * dz_c).sum(dim=0)
    m1_q = (rho_d * qn_n * dz_c).sum(dim=0)

    assert torch.allclose(m1_u, m0_u + taux * dt, rtol=1e-5, atol=1e-4)
    assert torch.allclose(m1_q, m0_q + evap * dt, rtol=1e-5, atol=1e-5)
    assert float(qke_n[0].mean().item()) > float(qke[0].mean().item())
    assert not torch.isnan(thm_n).any()

    # 4. 3D Smagorinsky-Lilly non-negativity and critical Richardson cutoff
    strain = torch.tensor([0.02, 0.02, 0.02], device=device)
    n2_vals = torch.tensor([-1.0e-4, 4.0e-5, 5.0e-4], device=device)  # Ri < 0, 0 < Ri < 0.25, Ri > 0.25
    nu_sgs, d_sgs = compute_smagorinsky_lilly_3d(strain, n2_vals, 4000.0, 4000.0, torch.full_like(strain, 200.0))
    assert float(nu_sgs[0].item()) > float(nu_sgs[1].item()) > 0.0
    assert float(nu_sgs[2].item()) == 0.0
    assert float(d_sgs[2].item()) == 0.0


def test_smith_nrh_pmodel_and_two_leaf_allocation(device: str) -> None:
    """Verify Smith et al. (2019) non-rectangular-hyperbola optimality and Two-Leaf allocation."""
    # 1. Exact first-order optimality condition d(A_j)/d(J_max) == c across m in (4c, 0.95)
    m_vals = torch.tensor([0.23, 0.30, 0.3629, 0.55, 0.78, 0.92], device=device)
    omega, omega_star = smith_omega(m_vals)
    assert (omega > 0.0).all()
    assert not torch.isnan(omega).any()

    # Finite-difference check of d(A_j)/d(J_max) at J_max = phi * I * omega
    phi_i = torch.full_like(m_vals, 200.0)
    jmax_opt = phi_i * omega
    eps = 0.2
    aj_plus = 0.25 * m_vals * non_rectangular_hyperbola(phi_i, jmax_opt + eps, THETA_NRH)
    aj_minus = 0.25 * m_vals * non_rectangular_hyperbola(phi_i, jmax_opt - eps, THETA_NRH)
    daj_djmax = (aj_plus - aj_minus) / (2.0 * eps)
    assert torch.allclose(daj_djmax, torch.full_like(m_vals, COST_C_SMITH), atol=1.0e-4)

    # Below m <= 4c, investment ceases cleanly
    om_zero, _ = smith_omega(torch.tensor([0.10, 4.0 * COST_C_SMITH], device=device))
    assert (om_zero == 0.0).all()

    # 2. Exact Two-Leaf conservation X_sun + X_sh == X and higher sunlit specific capacity
    vis_b = torch.tensor([260.0, 180.0], device=device)
    vis_d = torch.tensor([60.0, 90.0], device=device)
    mu0_mid = torch.tensor([0.85, 0.70], device=device)
    lai = torch.tensor([4.5, 2.0], device=device)
    acclim_rad = canopy_light_absorption_depury(vis_b, vis_d, mu0_mid, lai)
    inst_rad = canopy_light_absorption_depury(vis_b, vis_d, torch.tensor([0.60, 0.45], device=device), lai)

    x_tot = torch.tensor([120.0, 75.0], device=device)
    x_sun, x_sh = allocate_two_leaf_capacity(x_tot, acclim_rad, inst_rad.k_db, inst_rad.l_sun, lai)
    assert torch.allclose(x_sun + x_sh, x_tot, atol=1.0e-4)
    # Specific capacity per unit leaf area must be higher in sunlit leaves than shaded leaves!
    assert ((x_sun / inst_rad.l_sun) > (x_sh / inst_rad.l_sh)).all()

    # 3. Elevated CO2 acclimation down-regulates C3 V_cmax25 & C4 V_pmax25 and lowers F_4
    is_veg = torch.tensor([True, True], device=device)
    bio_400 = init_biosphere_state((2,), is_veg, device=device)
    bio_800 = init_biosphere_state((2,), is_veg, device=device)
    t_mid = torch.tensor([301.15, 293.15], device=device)
    t_ngt = torch.tensor([294.15, 285.15], device=device)
    vpd = torch.tensor([1600.0, 1000.0], device=device)
    p_s = torch.tensor([101325.0, 101325.0], device=device)

    for _ in range(60):
        update_daily_acclimation_smith(
            bio_400, vis_b, vis_d, mu0_mid, t_mid, vpd, t_ngt, p_s, 400.0, lai, is_veg, alpha_ema=0.20
        )
        update_daily_acclimation_smith(
            bio_800, vis_b, vis_d, mu0_mid, t_mid, vpd, t_ngt, p_s, 800.0, lai, is_veg, alpha_ema=0.20
        )

    assert (bio_800.vcmax25_c3 < bio_400.vcmax25_c3).all()
    assert (bio_800.vpmax25_c4 < bio_400.vpmax25_c4).all()
    assert (bio_800.f_c4 < bio_400.f_c4).all()

    # 4. Lambert W_0 accuracy and climate-driven prognostic LAI contrast (Tropical Wet vs Desert)
    z_test = torch.tensor([-0.36, -0.20, -0.05, 0.0, 1.5], device=device)
    w_val = lambert_w0(z_test)
    assert torch.allclose(w_val * torch.exp(w_val), z_test, atol=2.0e-6)

    bio_clim = init_biosphere_state((2,), is_veg, device=device)
    for _ in range(730):
        update_daily_lai_root_height(
            bio_clim,
            daily_precip_mm=torch.tensor([8.0, 0.05], device=device),   # Wet tropics vs hyper-arid desert
            daily_pet_mm=torch.tensor([4.5, 7.0], device=device),
            daily_ppfd_mol_m2=torch.tensor([42.0, 50.0], device=device),
            daily_temp_k=torch.tensor([299.15, 303.15], device=device),
            daily_vpd_pa=torch.tensor([900.0, 3500.0], device=device),
            pres_sfc=p_s,
            co2_ppmv=400.0,
            water_stress_beta=torch.tensor([1.0, 0.08], device=device),
            is_vegetated_land=is_veg,
        )
    assert float(bio_clim.lai[0].item()) > 3.0
    assert float(bio_clim.lai[1].item()) < 0.3
    assert float(bio_clim.h_canopy[0].item()) > 15.0
    assert float(bio_clim.h_canopy[1].item()) < 2.0

    # 5. Autonomous evergreen vs. deciduous leaf-habit emergence across 5 seasonal climate regimes
    #    (Kikuzawa, 1991; Sakai & Larcher, 1987; Givnish, 2002; Wang et al., 2023):
    #    [0: Humid Tropics, 1: Tropical Dry/Savanna, 2: Cool Temperate, 3: Boreal Taiga, 4: East Siberia Larix]
    veg5 = torch.ones(5, dtype=torch.bool, device=device)
    p5 = torch.full((5,), 101325.0, device=device)
    bio_evg = init_biosphere_state((5,), veg5, device=device)
    t_mean_c = torch.tensor([25.0, 26.0, 9.0, -4.0, -14.0], device=device)
    t_amp_c = torch.tensor([1.5, 4.0, 14.0, 19.0, 31.0], device=device)
    ppfd_mean = torch.tensor([42.0, 44.0, 26.0, 18.0, 17.0], device=device)
    ppfd_amp = torch.tensor([4.0, 8.0, 16.0, 16.0, 16.0], device=device)
    p_ann_mm = torch.tensor([2200.0, 950.0, 900.0, 520.0, 380.0], device=device)
    pet_ann_mm = torch.tensor([1350.0, 1450.0, 720.0, 420.0, 360.0], device=device)
    lai_winter = None
    lai_summer = None
    for day in range(5 * 365):
        doy = day % 365
        # Peak summer at doy=182, mid-winter at doy=0
        c_seas = -math.cos(2.0 * math.pi * doy / 365.0)
        t_day = C.T_FREEZE + t_mean_c + t_amp_c * c_seas
        t_min_day = t_day - torch.tensor([3.0, 5.0, 6.0, 6.0, 6.0], device=device)
        ppfd_day = torch.clamp(ppfd_mean + ppfd_amp * c_seas, min=0.5)
        # Tropical dry savanna (col 1) has a 5-month warm dry season when c_seas < -0.1
        is_dry_sav = (c_seas < -0.1)
        p_day = p_ann_mm / 365.0
        p_day = p_day.clone()
        p_day[1] = 0.2 if is_dry_sav else (p_ann_mm[1] / 210.0)
        pet_day = torch.clamp((pet_ann_mm / 365.0) * (1.0 + 0.6 * c_seas), min=0.05)
        vpd_day = torch.tensor([700.0, 2400.0 if is_dry_sav else 900.0, 800.0, 500.0, 500.0], device=device)
        beta_day = torch.tensor([1.0, 0.12 if is_dry_sav else 0.95, 0.90, 0.90, 0.85], device=device)
        update_daily_lai_root_height(
            bio_evg, p_day, pet_day, ppfd_day, t_day, vpd_day, p5, 400.0,
            water_stress_beta=beta_day, is_vegetated_land=veg5, daily_t_min_k=t_min_day,
        )
        if day == 4 * 365 + 15:
            lai_winter = bio_evg.lai.clone()
        elif day == 4 * 365 + 185:
            lai_summer = bio_evg.lai.clone()

    fe = bio_evg.f_evg
    assert fe is not None and lai_winter is not None and lai_summer is not None
    # Humid Tropics -> evergreen broadleaf; Boreal Taiga -> evergreen needleleaf with winter needles retained!
    assert float(fe[0].item()) > 0.70 and float(lai_winter[0].item()) > 3.0
    assert float(fe[3].item()) > 0.55 and float(lai_winter[3].item()) > 1.2
    # Tropical Dry Savanna & Cool Temperate -> lower evergreen share than Humid Tropics & Boreal Taiga
    assert float(fe[1].item()) < float(fe[0].item()) - 0.15
    assert float(fe[2].item()) < float(fe[3].item()) - 0.10
    # Extreme East Siberia (-51 C winter min < -40 C frost threshold) -> deciduous larch sheds needles in winter
    assert float(fe[4].item()) < 0.20 and float(lai_winter[4].item()) < 0.50
    assert float(lai_summer[4].item()) > float(lai_winter[4].item()) + 0.50


def test_surface_soil_permafrost_snow_burial_and_seaice(device: str) -> None:
    """Verify 5-layer soil freeze-thaw enthalpy conservation, snow burial, and fractional sea ice."""
    # 1. Exact enthalpy conservation across 0 degC soil freezing (zero-curtain test)
    t_soil = torch.full((5, 2), 273.65, device=device)  # +0.5 degC initial soil
    w_liq = torch.full((5, 2), 0.30, device=device)
    w_ice = torch.zeros((5, 2), device=device)
    dz = torch.tensor([0.05, 0.15, 0.40, 0.80, 1.60], device=device).unsqueeze(1)

    def _enthalpy(ts: torch.Tensor, wl: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        mw = 1000.0 * wl
        mi = 917.0 * wi
        c_v = 1.26e6 + 4186.0 * mw + 2106.0 * mi
        return ((c_v * (ts - C.T_FREEZE) - C.L_F0 * mi) * dz).sum(dim=0)

    h_init = _enthalpy(t_soil, w_liq, w_ice)
    g_cool = torch.tensor([-80.0, 40.0], device=device)  # Strong cooling on col 0, warming on col 1
    dt = 600.0
    for _ in range(30):
        t_soil, w_liq, w_ice = step_multilayer_soil_thermodynamics(t_soil, w_liq, w_ice, g_cool, dt=dt)

    h_final = _enthalpy(t_soil, w_liq, w_ice)
    expected_dh = g_cool * (30.0 * dt)
    assert torch.allclose(h_final - h_init, expected_dh, rtol=5.0e-4, atol=250.0)
    # Column 0 must have formed soil ice and remain pinned near 273.15 K (zero-curtain!)
    assert float(w_ice[0, 0].item()) > 0.01
    assert abs(float(t_soil[0, 0].item()) - C.T_FREEZE) < 0.5

    # 2. Snow-burial masking: tall taiga vs short tundra under 120 kg/m^2 snow (~0.48 m depth)
    stypes = torch.tensor([SURF_LAND, SURF_LAND, SURF_OCEAN, SURF_ICESHEET], device=device)
    sfc = init_surface_state(stypes, t_init=268.0)
    sfc.bio.lai = torch.tensor([4.0, 1.5, 0.0, 0.0], device=device)
    sfc.bio.h_canopy = torch.tensor([18.0, 0.25, 0.0, 0.0], device=device)  # Taiga vs Tundra
    sfc.swe_land = torch.tensor([120.0, 120.0, 0.0, 1000.0], device=device)
    sfc.swe_max_season = sfc.swe_land.clone()

    mu0 = torch.tensor([0.5, 0.5, 0.15, 0.5], device=device)
    alb_vd, _, _, _, h_exp, lai_exp, a_sn, _ = compute_surface_albedos(sfc, mu0)
    # Short tundra (col 1) is buried within snow-covered fraction -> tiny exposed LAI & high snow albedo
    assert float(lai_exp[1].item()) < 0.05
    assert float(lai_exp[0].item()) > 3.5
    assert float(alb_vd[1].item()) > 0.75
    assert float(alb_vd[0].item()) < 0.25

    # 3. Ocean zenith albedo & sea-ice growth under polar winter cooling
    ocn_low_sun, _ = compute_ocean_albedo_briegleb(torch.tensor([0.10], device=device))
    ocn_high_sun, _ = compute_ocean_albedo_briegleb(torch.tensor([0.95], device=device))
    assert float(ocn_low_sun.item()) > 0.20
    assert float(ocn_high_sun.item()) < 0.04

    sfc.t_ocean_ml[2] = C.T_FREEZE_OCEAN + 0.02
    u1 = torch.full((4,), 5.0, device=device)
    v1 = torch.zeros(4, device=device)
    t1 = torch.tensor([265.0, 265.0, 250.0, 245.0], device=device)  # Cold polar air over ocean col 2
    qv1 = torch.full((4,), 5.0e-4, device=device)
    p1 = torch.full((4,), 100500.0, device=device)
    ps = torch.full((4,), 101325.0, device=device)
    zref = torch.full((4,), 30.0, device=device)
    zeros = torch.zeros(4, device=device)
    lw_dn = torch.full((4,), 160.0, device=device)

    for _ in range(30):
        diag = step_surface(
            sfc, u1, v1, t1, qv1, p1, ps, zref,
            zeros, zeros, zeros, zeros, lw_dn, zeros, zeros, zeros, 400.0, dt=900.0,
        )
    assert float(sfc.m_seaice[2].item()) > 0.0
    assert 0.0 < float(diag.f_seaice[2].item()) < 1.0
    assert not torch.isnan(diag.shf).any()


if __name__ == "__main__":
    for dev in _available_devices():
        print(f"Running Phase 1 Physics/Turbulence/Biosphere/Surface tests on device={dev}...")
        test_microphysics_water_conservation_and_phase_partition(dev)
        test_rsl_and_mynn25_conservation(dev)
        test_smith_nrh_pmodel_and_two_leaf_allocation(dev)
        test_surface_soil_permafrost_snow_burial_and_seaice(dev)
    print("ALL PHASE 1 PHYSICS, TURBULENCE, BIOSPHERE & SURFACE TESTS PASSED!")

