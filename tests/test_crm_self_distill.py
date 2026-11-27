"""Tests for online active-learning & self-distilling 2D CRM (`aoba.crm_self_distill`)."""
from __future__ import annotations

import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.crm_self_distill import (
    SubgridSpatialHead,
    apply_exact_conservation_closure,
    apply_exact_rad_conservation_closure,
    apply_subgrid_coldpool_outflow,
    apply_subgrid_surface_microclimate,
    compute_periodic_spatial_stats,
    compute_subgrid_structure_features,
    reconstruct_subgrid_heterogeneity,
)


def test_exact_column_conservation_closure() -> None:
    """Verify exact water-mass, moist-enthalpy, and horizontal-momentum closure to float32 precision."""
    torch.manual_seed(0)
    nz, n_col = 32, 64
    dt = 600.0

    raw_prof = torch.randn(n_col, 8, nz) * 2.0
    raw_scal = torch.randn(n_col, 3) * 2.0

    rho_d = torch.linspace(1.2, 0.05, nz).unsqueeze(1).expand(nz, n_col).clone()
    dz_c = torch.linspace(120.0, 2000.0, nz).unsqueeze(1).expand(nz, n_col).clone()
    exner = torch.linspace(1.0, 0.35, nz).unsqueeze(1).expand(nz, n_col).clone()
    temp_sfc = torch.linspace(265.0, 302.0, n_col)
    thm_post_ls = torch.linspace(290.0, 650.0, nz).unsqueeze(1).expand(nz, n_col).clone()
    qn_post_ls = torch.linspace(1.5e-2, 1.0e-5, nz).unsqueeze(1).expand(nz, n_col).clone()
    qp_pre = torch.full((nz, n_col), 2.0e-5)
    qc_pre = torch.minimum(torch.full((nz, n_col), 5.0e-5), 0.25 * qn_post_ls)
    qi_pre = torch.minimum(torch.full((nz, n_col), 2.0e-5), 0.15 * qn_post_ls)
    u_pre = torch.randn(nz, n_col) * 10.0
    v_pre = torch.randn(nz, n_col) * 5.0
    shf = torch.randn(n_col) * 80.0
    evap = torch.rand(n_col) * 1.0e-4
    tau_x = torch.randn(n_col) * 0.15
    tau_y = torch.randn(n_col) * 0.10

    out = apply_exact_conservation_closure(
        raw_prof, raw_scal, rho_d, dz_c, exner, temp_sfc, thm_post_ls, qn_post_ls,
        qp_pre, qc_pre, qi_pre, u_pre, v_pre, shf, evap, tau_x, tau_y, dt,
    )
    dm = rho_d * dz_c

    # 1. Exact water mass conservation
    w_pre = (dm * (qn_post_ls + qp_pre)).sum(dim=0) + evap * dt
    w_post = (dm * (out["q_n"] + out["q_p"])).sum(dim=0) + (out["precip_rain"] + out["precip_snow"]) * dt
    rel_w_err = float(((w_post - w_pre).abs() / torch.clamp(w_pre, min=1.0)).max())
    assert rel_w_err < 1.0e-5, f"Water conservation error too large: {rel_w_err}"

    # 2. Exact moist sensible-enthalpy conservation in true temperature space T = theta_m*exner/(1 + RV/RD * qv)
    qv_pre = torch.clamp(qn_post_ls - qc_pre - qi_pre, min=0.0)
    t_post_ls = (thm_post_ls * exner) / (1.0 + C.RV_OVER_RD * qv_pre)
    t_new = (out["theta_m"] * exner) / (1.0 + C.RV_OVER_RD * out["q_v"])
    d_m_cond = (dm * ((out["q_c"] + out["q_i"] + out["q_p"]) - (qc_pre + qi_pre + qp_pre))).sum(dim=0)
    d_m_ice = (dm * (out["q_i"] - qi_pre)).sum(dim=0)
    dh_expected = (
        shf * dt
        + C.L_V0 * ((out["precip_rain"] + out["precip_snow"]) * dt + d_m_cond)
        + C.L_F0 * (out["precip_snow"] * dt + d_m_ice)
    )
    dh_actual = C.CP_D * (dm * (t_new - t_post_ls)).sum(dim=0)
    h_col = C.CP_D * (dm * t_post_ls).sum(dim=0)
    rel_h_col = float(((dh_actual - dh_expected).abs() / torch.clamp(h_col, min=1.0e8)).max())
    rel_h_err = float(((dh_actual - dh_expected).abs() / torch.clamp(dh_expected.abs(), min=1.0e6)).max())
    assert rel_h_col < 1.0e-6 and rel_h_err < 2.0e-4, f"Enthalpy conservation error too large: rel_col={rel_h_col}, rel_dh={rel_h_err}"

    # 2b. Exact stratosphere/mesosphere invariance (exner <= 0.50)
    strat = exner <= 0.50
    assert bool(strat.any())
    assert float((out["theta_m"][strat] - thm_post_ls[strat]).abs().max()) == 0.0
    assert float((out["q_n"][strat] - qn_post_ls[strat]).abs().max()) == 0.0
    assert float((out["q_v"][strat] - qv_pre[strat]).abs().max()) == 0.0

    # 3. Exact horizontal momentum impulse conservation
    du_imp = (dm * (out["u"] - u_pre)).sum(dim=0)
    dv_imp = (dm * (out["v"] - v_pre)).sum(dim=0)
    assert float((du_imp - tau_x * dt).abs().max()) < 1.0e-2
    assert float((dv_imp - tau_y * dt).abs().max()) < 1.0e-2

    # 4. Strict non-negativity & condensate partition consistency
    for k in ("q_n", "q_p", "q_v", "q_c", "q_i", "precip_rain", "precip_snow"):
        assert bool((out[k] >= 0.0).all()), f"Negative value in {k}"
    assert float((out["q_v"] + out["q_c"] + out["q_i"] - out["q_n"]).abs().max()) < 1.0e-6

    # 5. Exact mass-conserving subcolumn heterogeneity reconstruction (clear vs cloudy & dry vs rainy)
    f_cld_t = torch.full((nz, n_col), 0.25)
    f_prec_t = torch.full((n_col,), 0.25)
    cids = torch.arange(n_col, dtype=torch.int64)
    qn4, qp4, qv4, qc4, qi4, rain3, snow3 = reconstruct_subgrid_heterogeneity(
        out["q_v"], out["q_c"], out["q_i"], out["q_p"], f_cld_t,
        out["precip_rain"], out["precip_snow"], f_prec_t, cids, ny=1, nx=8, step_count=5,
    )
    assert float((qc4.mean(dim=(-2, -1)) - out["q_c"]).abs().max()) < 1.0e-6
    assert float((qi4.mean(dim=(-2, -1)) - out["q_i"]).abs().max()) < 1.0e-6
    assert float((qn4.mean(dim=(-2, -1)) - out["q_n"]).abs().max()) < 1.0e-6
    assert float((rain3.mean(dim=(-2, -1)) - out["precip_rain"]).abs().max()) < 1.0e-6
    # Verify 6 of 8 subcolumns are genuinely cloud-free whenever f_cld == 0.25 and below the 6 g/kg ceiling
    unclamped = (out["q_c"] + out["q_i"]) <= 0.25 * 0.006
    assert bool(unclamped.any())
    clear_frac = float((qc4 == 0.0).to(torch.float32)[unclamped].mean())
    assert clear_frac >= 0.74, f"Expected >=75% clear-sky subcolumns at f_cld=0.25, got {clear_frac:.3f}"


def test_2d_and_3d_spatial_clustering_descriptors() -> None:
    """Verify Moran's I, L_patch, and rho(SHF, LHF) distinguish clustered vs scattered 2D & 3D patches."""
    # 1. 2D (1 x 8) wildfire patch: 2 burned tiles out of 8, identical mean & std
    f_clump_2d = torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    f_scatt_2d = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    f_unif_2d = torch.ones((1, 1, 8)) * 0.25
    fields_2d = torch.cat([f_clump_2d, f_scatt_2d, f_unif_2d], dim=0)
    std2, mi2, lp2 = compute_periodic_spatial_stats(fields_2d, dx=4000.0)
    assert abs(float(std2[0] - std2[1])) < 1.0e-6, "Mean/std should match between clumped & scattered"
    assert float(mi2[0]) > 0.25, f"Expected positive Moran's I for clumped 2D patch, got {float(mi2[0])}"
    assert float(mi2[1]) < -0.25, f"Expected negative Moran's I for scattered 2D patch, got {float(mi2[1])}"
    assert float(mi2[2]) == 0.0 and float(lp2[2]) == 0.0
    assert float(lp2[0]) > float(lp2[1]) * 1.3, f"Expected larger L_patch for clumped 2D ({float(lp2[0]):.2f} vs {float(lp2[1]):.2f})"

    # 2. 3D (4 x 4) CRM domain: 2x2 contiguous forest block vs 4x4 checkerboard
    f_clump_3d = torch.zeros((1, 4, 4))
    f_clump_3d[0, :2, :2] = 1.0
    f_scatt_3d = torch.tensor([[[((r + c) % 2) for c in range(4)] for r in range(4)]], dtype=torch.float32)
    fields_3d = torch.cat([f_clump_3d, f_scatt_3d], dim=0)
    _, mi3, lp3 = compute_periodic_spatial_stats(fields_3d, dx=2000.0, dy=2000.0)
    assert float(mi3[0]) > 0.25, f"Expected positive Moran's I for 3D block, got {float(mi3[0])}"
    assert float(mi3[1]) < -0.95, f"Expected Moran's I ~ -1 for 3D checkerboard, got {float(mi3[1])}"
    assert float(lp3[0]) > float(lp3[1]) * 1.5

    # 3. Bowen-ratio (wildfire vs forest: rho=-1) vs cloud-shadowing (rho=+1)
    shf_bowen = torch.tensor([[[200.0, 200.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]]])
    lhf_bowen = torch.tensor([[[10.0, 10.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]]])
    shf_shad = torch.tensor([[[120.0, 120.0, 120.0, 120.0, 30.0, 30.0, 30.0, 30.0]]])
    lhf_shad = torch.tensor([[[160.0, 160.0, 160.0, 160.0, 40.0, 40.0, 40.0, 40.0]]])
    shf_both = torch.cat([shf_bowen, shf_shad], dim=0)
    lhf_both = torch.cat([lhf_bowen, lhf_shad], dim=0)
    trad_both = torch.full_like(shf_both, 295.0)
    alb_both = torch.full_like(shf_both, 0.15)
    qn_both = torch.full((4, 2, 1, 8), 0.01)
    w_both = torch.zeros((5, 2, 1, 8))
    struct = compute_subgrid_structure_features(shf_both, lhf_both, trad_both, alb_both, qn_both, w_both, dx=4000.0)
    assert struct.shape == (2, 16)
    assert float(struct[0, 6]) < -0.99, f"Expected rho(SHF,LHF) ~ -1 for wildfire/forest, got {float(struct[0, 6])}"
    assert float(struct[1, 6]) > 0.99, f"Expected rho(SHF,LHF) ~ +1 for cloud shadow, got {float(struct[1, 6])}"


def test_learned_spatial_placement_forest_locking_and_microclimate() -> None:
    """Verify SubgridSpatialHead learns forest-locking rain placement & microclimate preserves conservation."""
    torch.manual_seed(42)
    head = SubgridSpatialHead(col_hidden_dim=32, c_hid=16)
    opt = torch.optim.Adam(head.parameters(), lr=1.0e-2)

    # Synthetic forest-locking dataset: 4x4 3D CRM with random moist-forest patches (+LHF', -SHF')
    # Teacher places rain strictly on the forest patch
    b = 32
    h_col = torch.zeros((b, 32))
    forest = (torch.rand((b, 4, 4)) > 0.65).float()
    # Ensure at least 1 forest pixel per sample
    forest[:, 0, 0] = 1.0
    shf_a = -(forest - forest.mean(dim=(-2, -1), keepdim=True)) * 1.5
    lhf_a = (forest - forest.mean(dim=(-2, -1), keepdim=True)) * 1.5
    z_a = torch.zeros_like(shf_a)
    sub_maps = torch.stack([shf_a, lhf_a, z_a, z_a, z_a, z_a], dim=1)  # [B, 6, 4, 4]
    p_tgt = forest / forest.sum(dim=(-2, -1), keepdim=True)

    for _ in range(80):
        sc, sp, dt_cp, dq_cp, ug_cp = head(h_col, sub_maps)
        loss = -(p_tgt.reshape(b, -1) * torch.log_softmax(sp.reshape(b, -1), dim=-1)).sum(dim=-1).mean()
        loss = loss - (p_tgt.reshape(b, -1) * torch.log_softmax(sc.reshape(b, -1), dim=-1)).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sc_eval, sp_eval, dt_cp_eval, dq_cp_eval, ug_cp_eval = head(h_col, sub_maps)
        assert float(dt_cp_eval.mean(dim=(-2, -1)).abs().max()) < 1.0e-5
        assert float(dq_cp_eval.mean(dim=(-2, -1)).abs().max()) < 1.0e-7
        assert bool((ug_cp_eval >= 0.0).all())
        nz = 4
        qv = torch.full((nz, b), 0.01)
        qc = torch.full((nz, b), 2.0e-4)
        qi = torch.zeros((nz, b))
        qp = torch.zeros((nz, b))
        f_cld = forest.mean(dim=(-2, -1)).unsqueeze(0).expand(nz, b)
        f_prec = forest.mean(dim=(-2, -1))
        rain_col = torch.full((b,), 1.0e-4)
        snow_col = torch.zeros((b,))
        cids = torch.arange(b, dtype=torch.int64)
        _, _, _, qc4, _, rain3, _ = reconstruct_subgrid_heterogeneity(
            qv, qc, qi, qp, f_cld, rain_col, snow_col, f_prec, cids, ny=4, nx=4,
            score_cld=sc_eval, score_prec=sp_eval,
        )
        # Exact mass conservation AND >95% of rain lands directly on the moist forest tiles!
        assert float((rain3.mean(dim=(-2, -1)) - rain_col).abs().max()) < 1.0e-8
        assert float((qc4.mean(dim=(-2, -1)) - qc).abs().max()) < 1.0e-7
        rain_on_forest_frac = float((rain3 * forest).sum() / rain3.sum())
        assert rain_on_forest_frac > 0.95, f"Expected >95% rain locked onto forest, got {rain_on_forest_frac:.3f}"

        # Verify radial density-current outflow preserves exact cell-mean momentum and matches gust EKE
        u_mean0 = torch.full((b,), 2.0)
        v_mean0 = torch.full((b,), -1.5)
        dthm_test = -3.0 * (forest - forest.mean(dim=(-2, -1), keepdim=True))
        ug_test = 1.0 + 4.0 * forest
        u0_cp, v0_cp = apply_subgrid_coldpool_outflow(u_mean0, v_mean0, dthm_test, ug_test, cids, step_count=12)
        assert float((u0_cp.mean(dim=(-2, -1)) - u_mean0).abs().max()) < 1.0e-5
        assert float((v0_cp.mean(dim=(-2, -1)) - v_mean0).abs().max()) < 1.0e-5
        eke_out = ((u0_cp - u_mean0[:, None, None]) ** 2 + (v0_cp - v_mean0[:, None, None]) ** 2).mean(dim=(-2, -1))
        assert bool((eke_out > 1.0).all()), "Cold-pool outflow must inject non-zero gust kinetic energy"

    # Verify zero-mean surface microclimate warms burned tiles and moistens forest tiles
    thm0 = torch.full((2, 1, 8), 300.0)
    qv0 = torch.full((2, 1, 8), 0.012)
    rho0 = torch.full((2,), 1.18)
    dz0 = torch.full((2,), 120.0)
    ex0 = torch.full((2,), 1.0)
    shf3 = torch.tensor([[[250.0, 250.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]]] * 2)
    evap3 = torch.tensor([[[5.0e-6, 5.0e-6, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4]]] * 2)
    thm0_new, qv0_new = apply_subgrid_surface_microclimate(thm0, qv0, rho0, dz0, ex0, shf3, evap3, dt=1800.0)
    assert float((thm0_new.mean(dim=(-2, -1)) - 300.0).abs().max()) < 1.0e-5
    assert float((qv0_new.mean(dim=(-2, -1)) - 0.012).abs().max()) < 1.0e-7
    assert float(thm0_new[0, 0, 0] - thm0_new[0, 0, 4]) > 0.5, "Burned tile air must be warmer than forest tile air"
    assert float(qv0_new[0, 0, 4] - qv0_new[0, 0, 0]) > 1.0e-4, "Forest tile air must be moister than burned tile air"


def test_exact_radiation_conservation_closure() -> None:
    """Verify exact TOA-atmosphere-surface radiative energy closure, zero night SW, and wildfire albedo sensitivity."""
    torch.manual_seed(3)
    nz, m_col = 28, 64
    raw_qrad = torch.randn(m_col, nz) * 3.0
    raw_scal = torch.randn(m_col, 7) * 1.5
    cp_layer = torch.linspace(1.5e5, 8.0e3, nz).unsqueeze(1).expand(nz, m_col).clone()
    mu0 = torch.cat([torch.zeros(32), torch.linspace(0.05, 1.0, 32)])
    s_inc = mu0 * 1361.0
    alb_vd = torch.full((m_col,), 0.18)
    alb_vf = torch.full((m_col,), 0.18)
    alb_nd = torch.full((m_col,), 0.28)
    alb_nf = torch.full((m_col,), 0.28)
    emiss = torch.full((m_col,), 0.97)
    t_sfc = torch.linspace(240.0, 310.0, m_col)

    out = apply_exact_rad_conservation_closure(
        raw_qrad, raw_scal, cp_layer, s_inc, alb_vd, alb_vf, alb_nd, alb_nf, emiss, t_sfc
    )
    # 1. Exact zero night-side SW
    for k in ("sw1_dir", "sw1_dif", "sw2_dir", "sw2_dif", "asr"):
        assert float(out[k][:32].abs().max()) == 0.0, f"Night-side {k} must be identically 0.0"

    # 2. Physical SW bounds on day side
    sw_dn_tot = out["sw1_dir"] + out["sw1_dif"] + out["sw2_dir"] + out["sw2_dif"]
    assert bool((sw_dn_tot <= s_inc + 1.0e-4).all())
    assert bool((out["asr"] <= s_inc + 1.0e-4).all())

    # 3. Exact TOA - sfc == column atmospheric radiative heating integral
    f_sw_sfc = (
        (1.0 - alb_vd) * out["sw1_dir"] + (1.0 - alb_vf) * out["sw1_dif"]
        + (1.0 - alb_nd) * out["sw2_dir"] + (1.0 - alb_nf) * out["sw2_dif"]
    )
    f_lw_sfc = emiss * (out["lw_dn"] - C.STEFAN_BOLTZMANN * (t_sfc ** 4))
    f_atm_expected = (out["asr"] - out["olr"]) - (f_sw_sfc + f_lw_sfc)
    f_atm_actual = (cp_layer * out["q_rad"]).sum(dim=0)
    rel_err = float(((f_atm_actual - f_atm_expected).abs() / torch.clamp(f_atm_expected.abs(), min=50.0)).max())
    assert rel_err < 1.0e-5, f"Radiative column energy conservation error: {rel_err}"

    # 4. Immediate wildfire albedo darkening sensitivity on day-side tile
    alb_vd_dark = alb_vd.clone()
    alb_vd_dark[48] = 0.05  # charred black soil
    out_dark = apply_exact_rad_conservation_closure(
        raw_qrad, raw_scal, cp_layer, s_inc, alb_vd_dark, alb_vf, alb_nd, alb_nf, emiss, t_sfc
    )
    assert float(out_dark["asr"][48] - out["asr"][48]) > 1.0, "Darkening surface albedo must immediately raise ASR"


def test_coupled_self_distill_vs_full_crm() -> None:
    """Compare coupled AOBA with paired 1/13 CRM + ICA-Radiation self-distillation against 100% genuine physics."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_kw = dict(
        glevel=2,
        nz=20,
        z_top=36000.0,
        z_sponge_base=28000.0,
        crm_nx=4,
        crm_ny=1,
        dt_host=600.0,
        n_host_dyn_substeps=1,
        n_crm_dyn_substeps=4,
        rad_stride=2,
        rad_scheme="four_band",
        microphysics_scheme="two_moment",
        ocean_mode="ekman_sw",
        use_earth2014=False,
        enable_dust=False,
        enable_natural_aerosols=False,
        enable_land_carbon=False,
        enable_ocean_bgc=False,
        enable_river_routing=False,
    )
    torch.manual_seed(7)
    m_ctrl = init_aoba_model(
        AOBAConfig(**base_kw, enable_crm_self_distill=False, enable_rad_self_distill=False), device=dev
    )
    torch.manual_seed(7)
    m_dist = init_aoba_model(
        AOBAConfig(
            **base_kw,
            enable_crm_self_distill=True,
            enable_rad_self_distill=True,
            crm_distill_stride=13,
            crm_distill_warmup_steps=12,
            crm_distill_hidden_dim=64,
            crm_distill_replay_size=4096,
        ),
        device=dev,
    )

    # Run 12 warmup steps (6 rad steps) + 12 active paired CRM+Rad self-distillation steps
    for _ in range(24):
        d_ctrl = step_aoba_model(m_ctrl)
        d_dist = step_aoba_model(m_dist)

    assert m_dist.crm_distill is not None
    assert m_dist.rad_distill is not None
    assert m_dist.crm_distill.steps_seen == 24
    assert m_dist.rad_distill.steps_seen == 12
    assert m_dist.crm_distill.grad_updates > 0
    assert m_dist.rad_distill.grad_updates > 0
    assert m_dist.crm_distill.teacher_frac_last < 0.15
    assert m_dist.rad_distill.teacher_frac_last < 0.15
    assert torch.isfinite(m_dist.atmos.theta_m).all()
    assert torch.isfinite(m_dist.atmos.q_n).all()
    assert abs(d_ctrl.global_mean_t_sfc_k - d_dist.global_mean_t_sfc_k) < 1.5
    assert abs(d_ctrl.global_mean_olr_wm2 - d_dist.global_mean_olr_wm2) < 15.0
    assert abs(d_ctrl.global_mean_asr_wm2 - d_dist.global_mean_asr_wm2) < 20.0
    print(
        f"  [PASS] Coupled G2 24-step paired CRM+Rad parity ({dev}): "
        f"Ctrl Ts={d_ctrl.global_mean_t_sfc_k:.2f}K OLR={d_ctrl.global_mean_olr_wm2:.1f} ASR={d_ctrl.global_mean_asr_wm2:.1f} | "
        f"Distill Ts={d_dist.global_mean_t_sfc_k:.2f}K OLR={d_dist.global_mean_olr_wm2:.1f} ASR={d_dist.global_mean_asr_wm2:.1f} | "
        f"crm_tf={m_dist.crm_distill.teacher_frac_last:.3f} rad_tf={m_dist.rad_distill.teacher_frac_last:.3f}"
    )


def test_rcelarge_decoupled_surface_tiles_and_persistent_anchor() -> None:
    """Verify RCE_large (nx=64, 256 km) with crm_sfc_tiles=8 and 50% persistent anchor cohort."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(13)
    cfg = AOBAConfig(
        glevel=2,
        nz=20,
        z_top=36000.0,
        z_sponge_base=28000.0,
        crm_nx=64,
        crm_ny=1,
        crm_dx=4000.0,
        crm_sfc_tiles=8,
        dt_host=600.0,
        n_host_dyn_substeps=1,
        n_crm_dyn_substeps=4,
        rad_stride=2,
        rad_scheme="four_band",
        microphysics_scheme="two_moment",
        ocean_mode="ekman_sw",
        use_earth2014=False,
        enable_dust=False,
        enable_natural_aerosols=False,
        enable_land_carbon=False,
        enable_ocean_bgc=False,
        enable_river_routing=False,
        enable_crm_self_distill=True,
        enable_rad_self_distill=True,
        crm_distill_stride=7,
        crm_distill_warmup_steps=4,
        crm_distill_dwell_steps=64,
        crm_distill_hidden_dim=48,
        crm_distill_replay_size=2048,
    )
    m = init_aoba_model(cfg, device=dev)
    assert m.sfc_sub is not None
    assert m.sfc_sub.t_canopy.shape == (m.mesh.n_cells * 8,)
    assert m.crm.theta_m.shape == (20, m.mesh.n_cells, 1, 64)

    # Verify persistent anchor cohort (cohort_id == 0) remains continuously in teacher_idx across dwell windows
    from aoba.crm_self_distill import select_teacher_indices

    for _ in range(10):
        d = step_aoba_model(m)

    assert m.crm_distill is not None and m.rad_distill is not None
    assert torch.isfinite(m.atmos.theta_m).all()
    assert torch.isfinite(m.crm.theta_m).all()
    m.crm_distill.steps_seen = 10
    t1, _ = select_teacher_indices(m.crm_distill)
    m.crm_distill.steps_seen = 10 + 64
    t2, _ = select_teacher_indices(m.crm_distill)
    anchor_all = set(torch.nonzero(m.crm_distill.cohort_id == 0, as_tuple=False).squeeze(-1).tolist())
    n_teach = max(1, (m.crm_distill.n_cells + m.crm_distill.stride - 1) // m.crm_distill.stride)
    n_anchor = min(len(anchor_all), max(n_teach // 2, 1))
    shared_anchors = anchor_all & set(t1.tolist()) & set(t2.tolist())
    assert n_anchor >= 1
    assert len(shared_anchors) >= n_anchor
    print(
        f"  [PASS] RCE_large (nx=64, sfc_tiles=8, dwell=64 persistent anchor) ({dev}): "
        f"Ts={d.global_mean_t_sfc_k:.2f}K OLR={d.global_mean_olr_wm2:.1f} ASR={d.global_mean_asr_wm2:.1f}"
    )


def test_2d_isotropic_surface_tile_grid_and_kronecker_ops() -> None:
    """Verify general 2D isotropic tile factorization and exact 2D Kronecker block avg/expand."""
    from aoba.coupled_model import resolve_crm_sfc_tile_grid
    from aoba.crm_self_distill import block_avg_2d, block_expand_2d

    cases = [
        (16, 64, 16, (2, 8)),
        (16, 16, 16, (4, 4)),
        (64, 64, 16, (4, 4)),
        (12, 36, 12, (2, 6)),
        (15, 45, 15, (3, 5)),
        (1, 64, 8, (1, 8)),
        (16, 64, 0, (16, 64)),
    ]
    for ny, nx, req, exp in cases:
        got = resolve_crm_sfc_tile_grid(ny, nx, req)
        assert got == exp, f"tile_grid({ny}, {nx}, {req}) -> {got}, expected {exp}"
        ny_t, nx_t = got
        x_tile = torch.randn(4, 3, ny_t, nx_t)
        x_full = block_expand_2d(x_tile, ny_t, nx_t, ny, nx)
        assert x_full.shape == (4, 3, ny, nx)
        x_back = block_avg_2d(x_full, ny, nx, ny_t, nx_t)
        assert torch.allclose(x_back, x_tile, atol=1.0e-6)
        # Verify exact cell-mean preservation on arbitrary 2D field
        f_full = torch.randn(4, 3, ny, nx)
        f_tile = block_avg_2d(f_full, ny, nx, ny_t, nx_t)
        assert torch.allclose(f_tile.mean(dim=(-2, -1)), f_full.mean(dim=(-2, -1)), atol=1.0e-6)


def test_sparse_3d_crm_pool_slot_preservation_and_coupled_tica() -> None:
    """Verify teacher-only SparseCRMPool across arbitrary 3D shapes, slot preservation, 3D TICA & O3."""
    from aoba.coupled_model import resolve_crm_sfc_tile_grid
    from aoba.crm_self_distill import sync_sparse_crm_pool

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(19)
    for (ny, nx, dx, tiles, rad_sd) in [
        (16, 64, 4000.0, 16, True),
        (12, 36, 3000.0, 12, False),
    ]:
        cfg = AOBAConfig(
            glevel=1,
            nz=16,
            z_top=30000.0,
            z_sponge_base=24000.0,
            crm_ny=ny,
            crm_nx=nx,
            crm_dx=dx,
            crm_sfc_tiles=tiles,
            dt_host=360.0,
            n_host_dyn_substeps=1,
            n_crm_dyn_substeps=4,
            rad_stride=2,
            rad_scheme="four_band",
            enable_rad_3d_tica=True,
            o3_mode="prognostic",
            microphysics_scheme="two_moment",
            ocean_mode="ekman_sw",
            use_earth2014=False,
            enable_dust=False,
            enable_natural_aerosols=False,
            enable_land_carbon=False,
            enable_ocean_bgc=False,
            enable_river_routing=False,
            enable_crm_self_distill=True,
            enable_rad_self_distill=rad_sd,
            crm_distill_stride=8,
            crm_distill_warmup_steps=4,
            crm_distill_dwell_steps=48,
            crm_distill_hidden_dim=48,
            crm_distill_replay_size=2048,
        )
        m = init_aoba_model(cfg, device=dev)
        ny_t, nx_t = resolve_crm_sfc_tile_grid(ny, nx, tiles)
        assert m.sparse_crm_pool is not None
        assert m.crm.rho_d.shape == (16, m.mesh.n_cells, ny_t, nx_t)
        assert m.sparse_crm_pool.crm.rho_d.shape == (16, m.sparse_crm_pool.n_pool, ny, nx)

        for _ in range(6):
            d = step_aoba_model(m)

        assert torch.isfinite(m.atmos.theta_m).all()
        assert torch.isfinite(m.sparse_crm_pool.crm.theta_m).all()
        assert m.sparse_crm_pool.q_rad_valid

        # Verify slot preservation: cell staying in t_idx retains its exact slot & untouched 3D state
        pool = m.sparse_crm_pool
        cur_ids = pool.slot_cell_ids.clone()
        keep_cell = int(cur_ids[0].item())
        sentinel = pool.crm.theta_m[:, 0].clone()
        # Rotate only the last slot to a different host cell
        new_ids = cur_ids.clone()
        used = set(cur_ids.tolist())
        replacement = next(c for c in range(m.mesh.n_cells) if c not in used)
        new_ids[-1] = replacement
        ordered, n_act = sync_sparse_crm_pool(
            pool, new_ids, m.crm, m.atmos.rho_d, m.mesh.f_cell, step_seed=99
        )
        assert n_act == pool.n_pool
        assert int(ordered[0].item()) == keep_cell
        assert torch.equal(pool.crm.theta_m[:, 0], sentinel)
        assert int(ordered[-1].item()) == replacement
        print(
            f"  [PASS] SparseCRMPool 3D {ny}x{nx}@{dx:.0f}m (tiles={ny_t}x{nx_t}, rad_sd={rad_sd}, 3D-TICA+O3) ({dev}): "
            f"Ts={d.global_mean_t_sfc_k:.2f}K OLR={d.global_mean_olr_wm2:.1f}"
        )


def test_ogcm_gpu_fused_parity_and_async_ocean_stride() -> None:
    """Verify fused GPU 3D OGCM CPU-vs-GPU parity and conservative ocean_stride=3 with six_class_2m."""
    from aoba.ocean_ogcm import init_ogcm_state, step_ogcm_voronoi
    from aoba.scvt_mesh import build_scvt_mesh
    from aoba.surface import init_surface_state

    has_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    gpu_dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    def _run_ogcm(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        mesh = build_scvt_mesh(glevel=2, n_lloyd_iter=4, use_earth2014=False, device=dev)
        is_land = (mesh.lat_cell > 0.2) & (mesh.lat_cell < 0.8) & (mesh.lon_cell > 0.0) & (mesh.lon_cell < 1.2)
        mesh.surf_type = torch.where(is_land, torch.ones_like(mesh.surf_type), torch.zeros_like(mesh.surf_type))
        mesh.z_topo = torch.where(is_land, torch.full_like(mesh.z_topo, 500.0), torch.full_like(mesh.z_topo, -4000.0))
        sfc = init_surface_state(mesh.surf_type, z_topo=mesh.z_topo, t_init=290.0, h_ml_ocean=15.0)
        sfc.t_ocean_ml = 295.0 - 20.0 * (torch.sin(mesh.lat_cell) ** 2)
        ogcm = init_ogcm_state(sfc, mesh, nz_ocn=12)
        ogcm.u_edge = 0.2 * torch.sin(mesh.lat_cell[mesh.cells_on_edge[:, 0]]).unsqueeze(0).expand_as(ogcm.u_edge).clone() * ogcm.wet_e.float()
        tau_x = 0.08 * torch.cos(3.0 * mesh.lat_cell)
        tau_y = 0.03 * torch.sin(2.0 * mesh.lon_cell)
        z0 = torch.zeros_like(tau_x)
        evap = torch.full_like(tau_x, 3.0e-5)
        precip = torch.full_like(tau_x, 3.0e-5)
        sw_net = torch.clamp(200.0 * torch.cos(mesh.lat_cell), min=0.0)
        for _ in range(3):
            step_ogcm_voronoi(
                ogcm=ogcm, sfc=sfc, mesh=mesh,
                tau_x_sfc=tau_x, tau_y_sfc=tau_y, f_seaice=z0,
                evap_cell=evap, precip_cell=precip, runoff_total=z0, runoff_glacier=z0,
                dt=1440.0, sw_net_ocn=sw_net,
                enable_fct_zstar=True, enable_scale_aware_gm=True,
                enable_submeso=True, enable_tidal_mixing=True,
            )
        return ogcm.temp.cpu(), ogcm.sal.cpu(), ogcm.u_edge.cpu()

    if has_gpu:
        t_c, s_c, u_c = _run_ogcm(torch.device("cpu"))
        t_g, s_g, u_g = _run_ogcm(gpu_dev)
        rel_t = float((t_c - t_g).abs().max() / t_c.abs().max())
        err_s = float((s_c - s_g).abs().max())
        err_u = float((u_c - u_g).abs().max())
        assert rel_t < 1.0e-5 and err_s < 1.0e-4 and err_u < 1.0e-4, (
            f"OGCM GPU parity failed: rel_T={rel_t:.2e}, dS={err_s:.2e}, du={err_u:.2e}"
        )

    # Coupled conservative ocean_stride=3 with 3D 8x8 + six_class_2m + SparseCRMPool
    cfg = AOBAConfig(
        glevel=1, nz=16, z_top=30000.0, z_sponge_base=24000.0,
        crm_ny=8, crm_nx=8, crm_dx=4000.0, crm_sfc_tiles=16,
        dt_host=720.0, n_host_dyn_substeps=1, n_crm_dyn_substeps=4,
        rad_stride=2, rad_scheme="four_band", enable_rad_3d_tica=True,
        o3_mode="prognostic", microphysics_scheme="six_class_2m",
        ocean_mode="ogcm_3d", ocean_nz=8, ocean_stride=3,
        use_earth2014=False, enable_dust=False, enable_natural_aerosols=False,
        enable_land_carbon=False, enable_ocean_bgc=False, enable_river_routing=False,
        enable_crm_self_distill=True, enable_rad_self_distill=True,
        crm_distill_stride=8, crm_distill_warmup_steps=3, crm_distill_dwell_steps=48,
        crm_distill_hidden_dim=48, crm_distill_replay_size=2048,
    )
    m = init_aoba_model(cfg, device=gpu_dev)
    for _ in range(6):
        d = step_aoba_model(m)
    assert int(m.ocn_flux_accum.get("count", 0)) == 0
    assert torch.isfinite(m.atmos.theta_m).all()
    assert m.ogcm is not None and torch.isfinite(m.ogcm.temp).all()
    assert m.sparse_crm_pool is not None and m.sparse_crm_pool.crm.q_r is not None
    print(
        f"  [PASS] Fused GPU OGCM parity + conservative ocean_stride=3 + 3D 8x8 six_class_2m ({gpu_dev}): "
        f"Ts={d.global_mean_t_sfc_k:.2f}K OLR={d.global_mean_olr_wm2:.1f}"
    )


def test_shadow_divergence_abrupt_4xco2_atmos_and_land() -> None:
    """Verify pre-update shadow evaluation detects an abrupt-4xCO2 spike and subsequent online
    relaxation for BOTH Host Atmosphere and Land Surface coupling fidelity, plus spatial maps,
    fine-ocean CMIP7/MOC/EEO outputs, and automatic tuning-history logging."""
    import json
    import math
    import tempfile
    from pathlib import Path

    from aoba.cmip7_diag import CMIP7MonthlyAccumulator, export_cmip7_fx_xarray
    from aoba.crm_self_distill import (
        CRM_SHADOW_KEYS,
        RAD_SHADOW_KEYS,
        consume_distill_shadow_maps,
        read_distill_shadow_metrics,
    )
    from aoba.surface import SURF_LAND

    torch.manual_seed(17)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    cfg = AOBAConfig(
        glevel=2, nz=18, z_top=30000.0, z_sponge_base=24000.0,
        crm_ny=1, crm_nx=4, crm_dx=4000.0, crm_sfc_tiles=4,
        dt_host=720.0, n_host_dyn_substeps=1, n_crm_dyn_substeps=4,
        rad_stride=1, rad_scheme="ckd", o3_mode="rcemip",
        co2_mode="constant", co2_ppmv=284.3,
        ocean_mode="ekman_sw", use_earth2014=False,
        enable_dust=False, enable_natural_aerosols=False,
        enable_land_carbon=False, enable_ocean_bgc=False, enable_river_routing=False,
        enable_crm_self_distill=True, enable_rad_self_distill=True,
        enable_distill_shadow_diag=True,
        crm_distill_stride=4, crm_distill_warmup_steps=8, crm_distill_dwell_steps=64,
        crm_distill_hidden_dim=64, crm_distill_replay_size=2048,
    )
    model = init_aoba_model(cfg, device=dev)
    # Assign a deterministic ~40% continental landmass so land-tile shadow metrics are exercised
    is_land = (model.mesh.lat_cell.abs() < 0.85) & (model.mesh.lon_cell > -0.5) & (model.mesh.lon_cell < 2.2)
    model.mesh.surf_type = torch.where(is_land, torch.full_like(model.mesh.surf_type, SURF_LAND), model.mesh.surf_type)
    model.sfc.surf_type = model.mesh.surf_type.clone()
    if model.sfc_sub is not None:
        model.sfc_sub.surf_type = model.mesh.surf_type.unsqueeze(1).expand(-1, 4).reshape(-1).clone()

    # Phase 1: piControl (284.3 ppmv) spin-in so CRM & radiation students converge
    for _ in range(36):
        step_aoba_model(model)
    pre = read_distill_shadow_metrics(model.crm_distill, model.rad_distill)
    for k in (*CRM_SHADOW_KEYS, *RAD_SHADOW_KEYS):
        assert k in pre and math.isfinite(pre[k]), f"Missing/non-finite shadow metric {k}: {pre.get(k)}"
    assert pre["sfc_n_land_tiles"] > 0.0 and pre["sfc_rad_n_land"] > 0.0, (
        f"Expected active land tiles in shadow diagnostics, got {pre['sfc_n_land_tiles']} / {pre['sfc_rad_n_land']}"
    )

    # Phase 2: Abrupt 4xCO2 shock (284.3 -> 1137.27 ppmv) — measure pre-update OOD jump on the very next step
    model.cfg.co2_ppmv = 1137.27
    step_aoba_model(model)
    shock = read_distill_shadow_metrics(model.crm_distill, model.rad_distill)
    d_olr_bias = abs(shock["rad_bias_olr_wm2"] - pre["rad_bias_olr_wm2"])
    d_lwd_bias = abs(shock["sfc_rad_bias_lwd_wm2"] - pre["sfc_rad_bias_lwd_wm2"])
    assert shock["rad_rmse_olr_wm2"] > pre["rad_rmse_olr_wm2"] + 1.5 or d_olr_bias > 3.0, (
        f"Host Atmos OLR shadow did not react to abrupt 4xCO2: pre={pre['rad_rmse_olr_wm2']:.2f}({pre['rad_bias_olr_wm2']:+.2f}), "
        f"shock={shock['rad_rmse_olr_wm2']:.2f}({shock['rad_bias_olr_wm2']:+.2f})"
    )
    assert shock["sfc_rad_rmse_lwd_wm2"] > pre["sfc_rad_rmse_lwd_wm2"] + 1.5 or d_lwd_bias > 2.0, (
        f"Land Surface LW_dn shadow did not react to abrupt 4xCO2: pre={pre['sfc_rad_rmse_lwd_wm2']:.2f}({pre['sfc_rad_bias_lwd_wm2']:+.2f}), "
        f"shock={shock['sfc_rad_rmse_lwd_wm2']:.2f}({shock['sfc_rad_bias_lwd_wm2']:+.2f})"
    )

    # Phase 3: Online adaptation under 4xCO2 — verify both Host Atmosphere and Land Surface divergence settle
    post_hist: list[dict[str, float]] = []
    for _ in range(28):
        step_aoba_model(model)
        post_hist.append(read_distill_shadow_metrics(model.crm_distill, model.rad_distill))
    post = post_hist[-1]
    post_olr_avg = sum(h["rad_rmse_olr_wm2"] for h in post_hist[-4:]) / 4.0
    post_olr_bias_avg = sum(abs(h["rad_bias_olr_wm2"]) for h in post_hist[-4:]) / 4.0
    post_lwd_avg = sum(h["sfc_rad_rmse_lwd_wm2"] for h in post_hist[-4:]) / 4.0
    assert post_olr_avg < max(shock["rad_rmse_olr_wm2"], pre["rad_rmse_olr_wm2"]) and post_olr_bias_avg < abs(shock["rad_bias_olr_wm2"]), (
        f"Host Atmos OLR shadow divergence did not relax after 4xCO2: shock={shock['rad_rmse_olr_wm2']:.2f}({shock['rad_bias_olr_wm2']:+.2f}), "
        f"post_avg={post_olr_avg:.2f}(|bias|={post_olr_bias_avg:.2f})"
    )
    assert post_lwd_avg < max(shock["sfc_rad_rmse_lwd_wm2"], pre["sfc_rad_rmse_lwd_wm2"]), (
        f"Land Surface LW_dn shadow RMSE did not settle after 4xCO2: shock={shock['sfc_rad_rmse_lwd_wm2']:.2f}, post_avg={post_lwd_avg:.2f}"
    )

    # Verify per-cell spatial shadow divergence maps
    s_maps = consume_distill_shadow_maps(model.crm_distill, model.rad_distill, reset=True)
    for req_k in (
        "shadow_atm_dthm_k", "shadow_atm_dprec_mmd", "shadow_atm_unc",
        "shadow_land_dthm0_k", "shadow_rad_dolr_wm2", "shadow_land_dlwd_wm2", "shadow_teacher_visits",
    ):
        assert req_k in s_maps and s_maps[req_k].shape == (model.mesh.n_cells,), f"Bad spatial shadow map {req_k}"
    assert int((s_maps["shadow_teacher_visits"] > 0).sum().item()) > 0
    assert torch.isfinite(s_maps["shadow_land_dlwd_wm2"]).any()

    # Verify CMIP7 + MOC/OHT + EEO + spatial shadow export with independent fine ocean (G1 atmos + G2 ocean)
    cfg_fo = AOBAConfig(
        glevel=1, ocean_glevel=2, nz=12, z_top=30000.0, z_sponge_base=24000.0,
        crm_ny=1, crm_nx=8, crm_dx=4000.0, crm_sfc_tiles=4,
        dt_host=720.0, n_host_dyn_substeps=1, n_crm_dyn_substeps=2,
        rad_stride=1, rad_scheme="four_band", o3_mode="rcemip",
        ocean_mode="ogcm_3d", ocean_nz=8, ocean_stride=1, use_earth2014=False,
        enable_dust=False, enable_natural_aerosols=False,
        enable_land_carbon=False, enable_ocean_bgc=False, enable_river_routing=False,
        enable_crm_self_distill=True, enable_rad_self_distill=True,
        enable_distill_shadow_diag=True,
        crm_distill_stride=4, crm_distill_warmup_steps=2, crm_distill_dwell_steps=4,
        crm_distill_hidden_dim=32, crm_distill_replay_size=512,
    )
    m_fo = init_aoba_model(cfg_fo, device=dev)
    acc = CMIP7MonthlyAccumulator(include_plev19=True, include_ocean_3d=True)
    for _ in range(3):
        step_aoba_model(m_fo)
        acc.update_step_extrema(m_fo, end_of_day=True)
        acc.sample(m_fo)
    ds_fx = export_cmip7_fx_xarray(m_fo)
    ds_mon = acc.to_xarray(m_fo)
    assert "areacello" in ds_fx and ds_fx["deptho"].dims == ("n_cell_ocn",)
    assert ds_mon["tos"].dims == ("time", "n_cell_ocn") and ds_mon["thetao"].dims == ("time", "lev", "n_cell_ocn")
    assert ds_mon["siconc"].dims == ("time", "n_cell_ocn")
    for req_v in ("msftmz_atl", "msftmz_glob", "oht_atl", "oht_glob", "hcanopy", "fevg", "fc4", "fapar", "chi_leaf", "shadow_atm_dthm_k", "shadow_rad_dolr_wm2"):
        assert req_v in ds_mon, f"Missing Letter-paper CMIP7 variable: {req_v}"

    # Verify automatic tuning-history logging (INIT -> CONFIG_CHANGE)
    import importlib.util

    run_script = Path(__file__).resolve().parents[1] / "scripts" / "run_g5_30yr.py"
    spec = importlib.util.spec_from_file_location("_aoba_run_mod", str(run_script))
    assert spec is not None and spec.loader is not None
    run_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_mod)
    from aoba.params import OutputConfig

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        out_c = OutputConfig()
        run_mod._record_tuning_history(td_p, cfg_fo, out_c, sim_year=1, sim_date="2000-03-20", sim_days=0.0, step_count=0, note="initial spinup")
        cfg_fo2 = AOBAConfig(**{**cfg_fo.__dict__, "co2_ppmv": 1137.27, "glevel": 2})
        run_mod._record_tuning_history(td_p, cfg_fo2, out_c, sim_year=1001, sim_date="3000-03-20", sim_days=365250.0, step_count=43830000, note="abrupt 4xCO2")
        lines = [json.loads(l) for l in (td_p / "tuning_history.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2 and lines[0]["event"] == "INIT" and lines[1]["event"] == "CONFIG_CHANGE"
        assert lines[1]["changes"]["co2_ppmv"] == {"old": 284.3, "new": 1137.27}
        assert lines[1]["changes"]["glevel"] == {"old": 1, "new": 2}
        assert "abrupt 4xCO2" in (td_p / "tuning_history.md").read_text(encoding="utf-8")

    print(
        f"  [PASS] Shadow divergence abrupt-4xCO2 + fine-ocean CMIP7/MOC/EEO + tuning log ({dev}): "
        f"Host OLR RMSE {pre['rad_rmse_olr_wm2']:.2f} -> {shock['rad_rmse_olr_wm2']:.2f} -> {post['rad_rmse_olr_wm2']:.2f} W/m2 "
        f"(bias {pre['rad_bias_olr_wm2']:+.2f} -> {shock['rad_bias_olr_wm2']:+.2f} -> {post['rad_bias_olr_wm2']:+.2f}) | "
        f"Land LWdn RMSE {pre['sfc_rad_rmse_lwd_wm2']:.2f} -> {shock['sfc_rad_rmse_lwd_wm2']:.2f} -> {post['sfc_rad_rmse_lwd_wm2']:.2f} W/m2 | "
        f"Host dTh={post['atm_rmse_thm_k']:.3f}K Land dTh0={post['sfc_rmse_thm0_k']:.3f}K"
    )


def test_restart_grid_remap_ossp_reset() -> None:
    """Verify `_load_restart_into_model` resets OSSP students & q_p on vertical-grid remap,
    resets warmup schedule on horizontal remap, and preserves OSSP verbatim on same-grid resume."""
    import importlib.util
    import tempfile
    from pathlib import Path

    dev = torch.device("cpu")
    run_script = Path(__file__).resolve().parents[1] / "scripts" / "run_g5_30yr.py"
    spec = importlib.util.spec_from_file_location("_aoba_run_mod_remap", str(run_script))
    assert spec is not None and spec.loader is not None
    run_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_mod)

    base_kw = dict(
        glevel=1, nz=16, z_top=30000.0, z_sponge_base=24000.0,
        crm_ny=1, crm_nx=8, crm_dx=4000.0, dt_host=600.0,
        n_host_dyn_substeps=1, n_crm_dyn_substeps=2, rad_stride=2,
        rad_scheme="four_band", o3_mode="rcemip", microphysics_scheme="one_moment",
        ocean_mode="ekman_sw", use_earth2014=False,
        enable_dust=False, enable_natural_aerosols=False,
        enable_land_carbon=False, enable_ocean_bgc=False, enable_river_routing=False,
        enable_crm_self_distill=True, enable_rad_self_distill=True,
        crm_distill_stride=4, crm_distill_warmup_steps=4, crm_distill_dwell_steps=4,
        crm_distill_hidden_dim=32, crm_distill_replay_size=256,
    )
    m_src = init_aoba_model(AOBAConfig(**base_kw), device=dev)
    m_src.crm_distill.steps_seen = 12345
    m_src.crm_distill.ema_val_rmse_thm = 0.08
    m_src.rad_distill.steps_seen = 6789
    m_src.rad_distill.ema_val_rmse_olr = 1.5
    m_src.atmos.q_p.fill_(5.0e-4)
    m_src.crm.q_p.fill_(5.0e-4)
    with torch.no_grad():
        for p in m_src.crm_distill.net.parameters():
            p.fill_(0.123)
            break
    src_first_w = next(iter(m_src.crm_distill.net.parameters())).detach().clone()

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "src_restart.pt"
        run_mod._save_restart_atomic(m_src, ckpt_path)

        # 1. Exact same grid -> preserves weights & steps_seen verbatim
        m_same = init_aoba_model(AOBAConfig(**base_kw), device=dev)
        run_mod._load_restart_into_model(m_same, ckpt_path, dev)
        assert m_same.crm_distill.steps_seen == 12345
        assert m_same.rad_distill.steps_seen == 6789
        assert torch.allclose(next(iter(m_same.crm_distill.net.parameters())), src_first_w)
        assert float(m_same.atmos.q_p.max()) > 1.0e-4

        # 2. Horizontal-only remap (glevel 1 -> 2, same 30 km z_top) -> warm weights, reset warmup schedule
        m_h = init_aoba_model(AOBAConfig(**{**base_kw, "glevel": 2}), device=dev)
        run_mod._load_restart_into_model(m_h, ckpt_path, dev)
        assert m_h.crm_distill.steps_seen == 0
        assert m_h.crm_distill.ema_val_rmse_thm >= 1.9
        assert m_h.rad_distill.steps_seen == 0
        assert torch.allclose(next(iter(m_h.crm_distill.net.parameters())), src_first_w)

        # 3. Vertical-grid remap (30 km -> 45 km) -> fresh weights, steps_seen=0, zeroed q_p
        m_v = init_aoba_model(AOBAConfig(**{**base_kw, "z_top": 45000.0, "z_sponge_base": 36000.0}), device=dev)
        init_v_w = next(iter(m_v.crm_distill.net.parameters())).detach().clone()
        run_mod._load_restart_into_model(m_v, ckpt_path, dev)
        assert m_v.crm_distill.steps_seen == 0
        assert m_v.rad_distill.steps_seen == 0
        assert torch.allclose(next(iter(m_v.crm_distill.net.parameters())), init_v_w)
        assert float(m_v.atmos.q_p.abs().max()) == 0.0
        assert float(m_v.crm.q_p.abs().max()) == 0.0

    print("  [PASS] Restart grid-remap OSSP reset (same-grid preserve / h-remap re-warmup / v-remap full reset & q_p zero)")


if __name__ == "__main__":
    test_exact_column_conservation_closure()
    print("  [PASS] Exact column conservation closure (water, enthalpy, momentum, non-negativity)")
    test_2d_and_3d_spatial_clustering_descriptors()
    print("  [PASS] 2D & 3D periodic spatial clustering descriptors (Moran's I, L_patch, rho(SHF, LHF))")
    test_learned_spatial_placement_forest_locking_and_microclimate()
    print("  [PASS] Learned SubgridSpatialHead forest-locking placement & zero-mean microclimate")
    test_exact_radiation_conservation_closure()
    print("  [PASS] Exact per-subcolumn radiative energy closure, zero night SW, wildfire albedo response")
    test_coupled_self_distill_vs_full_crm()
    test_rcelarge_decoupled_surface_tiles_and_persistent_anchor()
    test_2d_isotropic_surface_tile_grid_and_kronecker_ops()
    print("  [PASS] General 2D isotropic surface-tile grid & exact Kronecker block avg/expand")
    test_sparse_3d_crm_pool_slot_preservation_and_coupled_tica()
    test_ogcm_gpu_fused_parity_and_async_ocean_stride()
    test_shadow_divergence_abrupt_4xco2_atmos_and_land()
    test_restart_grid_remap_ossp_reset()
    print("ALL CRM + RADIATION SELF-DISTILLATION TESTS PASSED!")
