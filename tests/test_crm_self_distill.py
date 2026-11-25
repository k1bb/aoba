"""Tests for online active-learning & self-distilling 2D CRM (`aoba.crm_self_distill`)."""
from __future__ import annotations

import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.crm_self_distill import (
    SubgridSpatialHead,
    apply_exact_conservation_closure,
    apply_exact_rad_conservation_closure,
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
    qc_pre = torch.full((nz, n_col), 5.0e-5)
    qi_pre = torch.full((nz, n_col), 2.0e-5)
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

    # 2. Exact moist enthalpy conservation
    d_m_cond = (dm * ((out["q_c"] + out["q_i"] + out["q_p"]) - (qc_pre + qi_pre + qp_pre))).sum(dim=0)
    d_m_ice = (dm * (out["q_i"] - qi_pre)).sum(dim=0)
    dh_expected = (
        shf * dt
        + C.L_V0 * ((out["precip_rain"] + out["precip_snow"]) * dt + d_m_cond)
        + C.L_F0 * (out["precip_snow"] * dt + d_m_ice)
    )
    dh_actual = C.CP_D * (dm * exner * (out["theta_m"] - thm_post_ls)).sum(dim=0)
    rel_h_err = float(((dh_actual - dh_expected).abs() / torch.clamp(dh_expected.abs(), min=1.0e5)).max())
    assert rel_h_err < 1.0e-4, f"Enthalpy conservation error too large: {rel_h_err}"

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
    # Verify 6 of 8 subcolumns are genuinely cloud-free when f_cld == 0.25
    clear_frac = float((qc4 == 0.0).to(torch.float32).mean())
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
        sc, sp = head(h_col, sub_maps)
        loss = -(p_tgt.reshape(b, -1) * torch.log_softmax(sp.reshape(b, -1), dim=-1)).sum(dim=-1).mean()
        loss = loss - (p_tgt.reshape(b, -1) * torch.log_softmax(sc.reshape(b, -1), dim=-1)).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sc_eval, sp_eval = head(h_col, sub_maps)
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
    anchors = set(torch.nonzero(m.crm_distill.cohort_id == 0, as_tuple=False).squeeze(-1).tolist())
    assert len(anchors.intersection(set(t1.tolist()))) >= 1
    assert anchors.intersection(set(t1.tolist())) == anchors.intersection(set(t2.tolist()))
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
    print("ALL CRM + RADIATION SELF-DISTILLATION TESTS PASSED!")
