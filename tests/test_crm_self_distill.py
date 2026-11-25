"""Tests for online active-learning & self-distilling 2D CRM (`aoba.crm_self_distill`)."""
from __future__ import annotations

import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.crm_self_distill import apply_exact_conservation_closure


def test_exact_column_conservation_closure() -> None:
    """Verify exact water-mass, moist-enthalpy, and horizontal-momentum closure to float32 precision."""
    torch.manual_seed(0)
    nz, n_col = 32, 64
    dt = 600.0

    raw_prof = torch.randn(n_col, 7, nz) * 2.0
    raw_scal = torch.randn(n_col, 2) * 2.0

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


def test_coupled_self_distill_vs_full_crm() -> None:
    """Compare coupled AOBA with 1/13 self-distilling CRM against 100% genuine 2D CRM."""
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
    m_ctrl = init_aoba_model(AOBAConfig(**base_kw, enable_crm_self_distill=False), device=dev)
    torch.manual_seed(7)
    m_dist = init_aoba_model(
        AOBAConfig(
            **base_kw,
            enable_crm_self_distill=True,
            crm_distill_stride=13,
            crm_distill_warmup_steps=6,
            crm_distill_hidden_dim=48,
            crm_distill_replay_size=4096,
        ),
        device=dev,
    )

    # Run 6 warmup steps + 12 active self-distillation steps
    for _ in range(18):
        d_ctrl = step_aoba_model(m_ctrl)
        d_dist = step_aoba_model(m_dist)

    assert m_dist.crm_distill is not None
    assert m_dist.crm_distill.steps_seen == 18
    assert m_dist.crm_distill.grad_updates > 0
    assert m_dist.crm_distill.teacher_frac_last < 0.15
    assert torch.isfinite(m_dist.atmos.theta_m).all()
    assert torch.isfinite(m_dist.atmos.q_n).all()
    assert abs(d_ctrl.global_mean_t_sfc_k - d_dist.global_mean_t_sfc_k) < 1.5
    assert abs(d_ctrl.global_mean_olr_wm2 - d_dist.global_mean_olr_wm2) < 15.0
    print(
        f"  [PASS] Coupled G2 18-step parity ({dev}): "
        f"Ctrl Ts={d_ctrl.global_mean_t_sfc_k:.2f}K OLR={d_ctrl.global_mean_olr_wm2:.1f} | "
        f"Distill Ts={d_dist.global_mean_t_sfc_k:.2f}K OLR={d_dist.global_mean_olr_wm2:.1f} | "
        f"teacher_frac={m_dist.crm_distill.teacher_frac_last:.3f} grad_updates={m_dist.crm_distill.grad_updates}"
    )


if __name__ == "__main__":
    test_exact_column_conservation_closure()
    print("  [PASS] Exact column conservation closure (water, enthalpy, momentum, non-negativity)")
    test_coupled_self_distill_vs_full_crm()
    print("ALL CRM SELF-DISTILLATION TESTS PASSED!")
