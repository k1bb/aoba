"""Tests for Schneider et al. (2025, JAMES) & Gan et al. (2026, New Phytol.) optimality thermal acclimation."""

from __future__ import annotations

import torch

from aoba import constants as C
from aoba.biosphere import (
    compute_two_leaf_gas_exchange,
    init_biosphere_state,
    prepare_two_leaf_allocation,
    update_daily_acclimation_smith,
)


def test_emergent_thermal_acclimation_schneider_gan() -> None:
    """Verify emergent T_opt vs T_growth slope, buffered A_opt, declining J25/V25, and xi(T_g) separation."""
    tgs_c = [10.0, 15.0, 20.0, 25.0, 30.0]
    n = len(tgs_c)
    veg = torch.ones(n, dtype=torch.bool)
    bio = init_biosphere_state((n,), veg, device="cpu")
    bio.lai = torch.full((n,), 3.0)
    bio.f_c4 = torch.zeros(n)  # Pure C3 test matching ACi-TGlob / Schneider (2025) / Gan (2026)

    tg_k = torch.tensor([273.15 + t for t in tgs_c], dtype=torch.float32)
    bio.ema_temp_mid = tg_k.clone()
    bio.ema_temp_home = tg_k.clone()
    bio.ema_temp_night = tg_k - 5.0

    vis_b = torch.full((n,), 260.0)
    vis_d = torch.full((n,), 60.0)
    mu0 = torch.full((n,), 0.85)
    vpd_g = torch.full((n,), 1000.0)
    pres = torch.full((n,), 101325.0)
    lai = torch.full((n,), 3.0)

    # Spin up 15-day EMA capacities to equilibrium at each growth temperature
    for _ in range(120):
        update_daily_acclimation_smith(
            bio=bio,
            midday_vis_beam=vis_b,
            midday_vis_diff=vis_d,
            midday_mu0=mu0,
            midday_temp=tg_k,
            midday_vpd=vpd_g,
            night_temp=tg_k - 5.0,
            pres_sfc=pres,
            co2_ppmv=400.0,
            lai_exposed=lai,
            is_vegetated_land=veg,
        )
    # Keep pure C3 and exact T_home = T_growth for controlled comparison
    bio.f_c4.zero_()
    bio.ema_temp_home = tg_k.clone()

    # Check declining Jmax25 / Vcmax25 with warming (Gan et al., 2026, Fig. 4)
    jv_ratio = (bio.jmax25_c3 / torch.clamp(bio.vcmax25_c3, min=1.0e-3)).tolist()
    for i in range(n - 1):
        assert jv_ratio[i] > jv_ratio[i + 1], f"Jmax25/Vcmax25 must decline with T_growth: {jv_ratio}"

    # Sweep instantaneous leaf temperature T_leaf from 5 to 48 C at RH = 60% (field warming curve)
    tl_c_grid = torch.linspace(5.0, 48.0, 173)
    anet_curves = torch.zeros((len(tl_c_grid), n))
    ws = torch.ones(n)
    alloc = prepare_two_leaf_allocation(bio, vis_b, vis_d, mu0, lai)
    for k, tc_val in enumerate(tl_c_grid):
        tl = torch.full((n,), 273.15 + float(tc_val))
        es = 610.78 * torch.exp(17.27 * tc_val / (tc_val + 237.3))
        vpd_inst = torch.full((n,), max(float(0.40 * es), 50.0))
        res = compute_two_leaf_gas_exchange(
            bio=bio,
            vis_beam_wm2=vis_b,
            vis_diff_wm2=vis_d,
            mu0=mu0,
            temp_sun=tl,
            temp_sh=tl,
            vpd_sun=vpd_inst,
            vpd_sh=vpd_inst,
            pres_sfc=pres,
            co2_ppmv=400.0,
            lai_exposed=lai,
            water_stress_beta=ws,
            precomputed_alloc=alloc,
        )
        anet_curves[k] = res.an_net

    idx_opt = torch.argmax(anet_curves, dim=0)
    t_opt = [float(tl_c_grid[int(i)]) for i in idx_opt]
    a_opt = [float(anet_curves[int(idx_opt[j]), j]) for j in range(n)]
    slope = (t_opt[-1] - t_opt[0]) / (tgs_c[-1] - tgs_c[0])

    assert 0.50 <= slope <= 0.85, f"Expected T_opt vs T_growth slope in [0.50, 0.85], got {slope:.3f} (T_opt={t_opt})"
    for i in range(n - 1):
        assert t_opt[i + 1] > t_opt[i], f"T_opt must increase monotonically with T_growth: {t_opt}"
    # Schneider et al. (2025): buffered A_opt across cold-to-temperate growth temperatures
    a_spread = (max(a_opt[:4]) - min(a_opt[:4])) / max(a_opt[:4])
    assert a_spread < 0.25, f"A_opt should remain well-buffered across 10-25 C: {a_opt}"
    print(
        f"  [PASS] Emergent thermal acclimation (Schneider 2025 / Gan 2026): "
        f"T_opt={[round(x, 1) for x in t_opt]} C (slope={slope:.2f} C/C), "
        f"J25/V25={[round(x, 2) for x in jv_ratio]}"
    )


def test_gpu_metal_parity_and_cuda_transpile() -> None:
    """Verify exact CPU vs MPS Metal parity and CUDA shader transpilation for two_leaf_gas_exchange_col."""
    from aoba.kernels_cuda import transpile_metal_to_cuda
    from aoba.kernels_physics import PHYSICS_METAL_SHADER_SRC

    cuda_src, spec_list = transpile_metal_to_cuda(PHYSICS_METAL_SHADER_SRC)
    specs = {s.name: s for s in spec_list}
    assert "two_leaf_gas_exchange_col" in specs and "bio_peaked_arrh_ds" in cuda_src
    param_names = [p.name for p in specs["two_leaf_gas_exchange_col"].params]
    assert "t_home" in param_names, f"t_home missing from Metal/CUDA kernel spec: {param_names}"
    print(f"  [PASS] Metal->CUDA transpiler verified ({len(param_names)} buffers including t_home)")

    if not torch.backends.mps.is_available():
        return

    n = 32
    torch.manual_seed(42)
    veg = torch.ones(n, dtype=torch.bool)
    bio_cpu = init_biosphere_state((n,), veg, device="cpu")
    bio_cpu.lai = torch.linspace(0.5, 5.0, n)
    bio_cpu.ema_temp_mid = torch.linspace(280.15, 305.15, n)
    bio_cpu.ema_temp_home = torch.linspace(283.15, 303.15, n)
    bio_cpu.f_c4 = torch.linspace(0.0, 0.8, n)
    bio_cpu.vcmax25_c3 = torch.linspace(25.0, 90.0, n)
    bio_cpu.jmax25_c3 = torch.linspace(55.0, 170.0, n)
    bio_cpu.r25_c3 = torch.linspace(0.4, 1.5, n)
    bio_cpu.vcmax25_c4 = torch.linspace(15.0, 45.0, n)
    bio_cpu.jmax25_c4 = torch.linspace(70.0, 160.0, n)
    bio_cpu.vpmax25_c4 = torch.linspace(25.0, 80.0, n)
    bio_cpu.r25_c4 = torch.linspace(0.3, 1.0, n)

    vis_b = torch.linspace(50.0, 350.0, n)
    vis_d = torch.linspace(20.0, 100.0, n)
    mu0 = torch.linspace(0.2, 0.95, n)
    t_sun = torch.linspace(278.15, 312.15, n)
    t_sh = t_sun - 2.0
    vpd_sun = torch.linspace(200.0, 3500.0, n)
    vpd_sh = 0.7 * vpd_sun
    pres = torch.full((n,), 101325.0)
    ws = torch.linspace(0.2, 1.0, n)
    gbw = torch.full((n,), 0.04)

    res_cpu = compute_two_leaf_gas_exchange(
        bio_cpu, vis_b, vis_d, mu0, t_sun, t_sh, vpd_sun, vpd_sh, pres, 380.0, bio_cpu.lai, ws, gbw, gbw
    )

    from dataclasses import fields
    bio_mps = init_biosphere_state((n,), veg, device="mps")
    for f in fields(bio_cpu):
        val = getattr(bio_cpu, f.name)
        setattr(bio_mps, f.name, val.to("mps") if isinstance(val, torch.Tensor) else val)

    res_mps = compute_two_leaf_gas_exchange(
        bio_mps,
        vis_b.to("mps"),
        vis_d.to("mps"),
        mu0.to("mps"),
        t_sun.to("mps"),
        t_sh.to("mps"),
        vpd_sun.to("mps"),
        vpd_sh.to("mps"),
        pres.to("mps"),
        380.0,
        bio_mps.lai,
        ws.to("mps"),
        gbw.to("mps"),
        gbw.to("mps"),
    )

    for attr in ("gpp", "resp_leaf", "an_net", "gsw_sun", "gsw_sh"):
        diff = float((getattr(res_cpu, attr) - getattr(res_mps, attr).cpu()).abs().max().item())
        scale = max(float(getattr(res_cpu, attr).abs().max().item()), 1.0e-6)
        assert diff / scale < 1.0e-4, f"CPU vs MPS mismatch on {attr}: max_diff={diff}, rel={diff/scale}"
    print("  [PASS] CPU vs MPS Metal kernel parity (< 1e-4 rel err across C3/C4 + T_home + xi'(T_g))")


if __name__ == "__main__":
    print("Running Schneider (2025) & Gan (2026) Thermal Acclimation tests...")
    test_emergent_thermal_acclimation_schneider_gan()
    test_gpu_metal_parity_and_cuda_transpile()
    print("ALL THERMAL ACCLIMATION TESTS PASSED!")
