"""Verification suite for AOBA Full-ESM: CN-model Nitrogen, CRM Lightning/LNOx, & MAML Patch SPITFIRE."""

from __future__ import annotations

import torch

from aoba import constants as C
from aoba.biosphere import init_biosphere_state
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.land_carbon import (
    compute_bnf_carbon_cost,
    diagnose_land_carbon_metrics,
    init_land_carbon_state,
    step_land_carbon_and_fire,
)
from aoba.lightning import compute_crm_lightning_and_lnox, diagnose_lightning_metrics
from aoba.surface import SURF_LAND, SURF_OCEAN


def test_bnf_temperature_optimality() -> None:
    """Verify Houlton (2008) / Fisher (2010, FUN) BNF cost peaks in efficiency near 25 C and shuts down at 0 C."""
    t_c = torch.tensor([0.0, 10.0, 25.15, 40.0])
    cost = compute_bnf_carbon_cost(t_c)
    assert 5.5 < cost[2].item() < 7.0, f"Unexpected 25 C BNF cost: {cost[2].item()}"
    assert cost[0].item() > 150.0, f"Arctic BNF cost should exceed 150 kgC/kgN, got {cost[0].item()}"
    assert cost[2].item() < cost[1].item() < cost[0].item()
    assert cost[2].item() < cost[3].item()
    print(f"[PASS] FUN/Houlton BNF cost: 0C={cost[0]:.1f}, 10C={cost[1]:.1f}, 25C={cost[2]:.2f}, 40C={cost[3]:.1f} kgC/kgN")


def test_exact_cn_conservation_and_functional_balance() -> None:
    """Verify exact float32 Kahan C & N mass conservation and functional-balance root allocation shift."""
    n = 2  # Cell 0: N-replete tropical; Cell 1: N-starved cool boreal
    st = torch.tensor([SURF_LAND, SURF_LAND], dtype=torch.int64)
    lai = torch.tensor([4.0, 2.0])
    hc = torch.tensor([25.0, 12.0])
    lc = init_land_carbon_state(st, lai, hc)
    # Starve Cell 1 of mineral N to trigger functional-balance root allocation shift
    lc.n_min[0] = 0.080
    lc.n_min[1] = 0.0002

    def _tot_c() -> torch.Tensor:
        return (
            (lc.c_leaf + lc.c_leaf_comp)
            + (lc.c_root + lc.c_root_comp)
            + (lc.c_wood + lc.c_wood_comp)
            + (lc.c_lit_fast + lc.c_lit_fast_comp)
            + (lc.c_lit_slow + lc.c_lit_slow_comp)
            + (lc.c_soil + lc.c_soil_comp).sum(dim=0)
        ).double()

    def _tot_n() -> torch.Tensor:
        return (
            (lc.n_leaf + lc.n_leaf_comp)
            + (lc.n_root + lc.n_root_comp)
            + (lc.n_wood + lc.n_wood_comp)
            + (lc.n_lit_fast + lc.n_lit_fast_comp)
            + (lc.n_lit_slow + lc.n_lit_slow_comp)
            + (lc.n_soil + lc.n_soil_comp).sum(dim=0)
            + (lc.n_min + lc.n_min_comp)
        ).double()

    c0 = _tot_c()
    n0 = _tot_n()
    c_flux_sum = torch.zeros(n, dtype=torch.float64)
    n_flux_sum = torch.zeros(n, dtype=torch.float64)

    t_soil = torch.tensor([[298.15, 276.15]] * 5)
    w_liq = torch.tensor([[0.28, 0.25]] * 5)
    w_ice = torch.zeros(5, n)
    f_snow = torch.zeros(n)
    t_sfc = torch.tensor([299.0, 278.0])
    vpd = torch.tensor([800.0, 400.0])
    u10 = torch.tensor([3.0, 4.0])
    precip = torch.tensor([5.0e-5, 2.0e-5])
    gpp = torch.tensor([12.0, 6.0])
    rleaf = torch.tensor([1.0, 0.5])
    vc25 = torch.tensor([55.0, 35.0])
    fevg = torch.tensor([0.9, 0.7])
    transp = torch.tensor([4.0e-5, 1.5e-5])
    ndep = torch.tensor([2.0e-11, 0.5e-11])
    cg = torch.tensor([1.0e-13, 1.0e-14])
    dt = 3600.0

    for _ in range(360):
        step_land_carbon_and_fire(
            lc, st, gpp, rleaf, lai, hc, t_soil, w_liq, w_ice, f_snow, t_sfc, vpd, u10, precip, dt,
            enable_nitrogen=True, vcmax25_eff=vc25, f_evg=fevg, transp_kgm2s=transp,
            n_dep_kgn_m2s=ndep, flash_cg_m2s=cg,
        )
        c_flux_sum += lc.nbp_kgc_m2s.double() * dt
        n_flux_sum += (
            lc.n_dep_kgn_m2s + lc.bnf_kgn_m2s - lc.n_leach_kgn_m2s - lc.n_denit_kgn_m2s - lc.n_fire_kgn_m2s
        ).double() * dt

    c_err = ((_tot_c() - c0) - c_flux_sum).abs().max().item()
    n_err = ((_tot_n() - n0) - n_flux_sum).abs().max().item()
    assert c_err < 1.0e-5, f"Carbon conservation violation: {c_err:.3e}"
    assert n_err < 1.0e-6, f"Nitrogen conservation violation: {n_err:.3e}"
    assert lc.phi_n[1].item() < lc.phi_n[0].item() - 0.15, f"Expected N limitation on Cell 1: {lc.phi_n.tolist()}"
    assert lc.f_root_alloc[1].item() > lc.f_root_alloc[0].item() + 0.10, (
        f"Expected functional-balance root shift: {lc.f_root_alloc.tolist()}"
    )
    print(
        f"[PASS] CN conservation (dC={c_err:.2e}, dN={n_err:.2e}) & functional balance "
        f"(phiN={lc.phi_n[0]:.2f} vs {lc.phi_n[1]:.2f}, f_root={lc.f_root_alloc[0]:.2f} vs {lc.f_root_alloc[1]:.2f})"
    )


def test_crm_lightning_and_lnox_conservation() -> None:
    """Verify McCaul CRM lightning responds to updraft graupel, CG fraction follows Price & Rind, and LNOx conserves."""
    nz, nc, ny, nx = 32, 4, 2, 2
    z_c = torch.linspace(200.0, 20000.0, nz)
    dz_c = torch.full((nz,), z_c[1].item() - z_c[0].item())
    # Linear temperature profile: 300 K at surface -> 200 K at 15 km -> 210 K above
    temp = torch.clamp(300.0 - 0.0067 * z_c, min=200.0)
    pres = 101325.0 * torch.exp(-z_c / 8000.0)
    rho_d = (pres / (C.R_D * temp)).view(nz, 1, 1, 1).expand(nz, nc, ny, nx).clone()
    exner = (pres / C.P_REF) ** (C.R_D / C.CP_D)
    theta_m = (temp / exner).view(nz, 1, 1, 1).expand(nz, nc, ny, nx).clone()
    q_v = torch.full_like(rho_d, 1.0e-4)
    q_i = torch.zeros_like(rho_d)
    q_p = torch.zeros_like(rho_d)
    w_stag = torch.zeros(nz + 1, nc, ny, nx)

    # Cell 0: strong deep continental-like storm (w=8 m/s, graupel at -15 C) on subcolumn (0, 0)
    k_mid = (temp - 258.15).abs().argmin().item()
    w_stag[k_mid - 2 : k_mid + 4, 0, 0, 0] = 8.0
    q_p[k_mid - 2 : k_mid + 4, 0, 0, 0] = 3.0e-3
    q_i[k_mid : k_mid + 8, 0, 0, 0] = 1.0e-3
    # Cell 1: weak maritime-like storm (w=1.5 m/s, low graupel)
    w_stag[k_mid - 1 : k_mid + 2, 1, 0, 0] = 1.5
    q_p[k_mid - 1 : k_mid + 2, 1, 0, 0] = 4.0e-4
    q_i[k_mid : k_mid + 3, 1, 0, 0] = 2.0e-4

    area = torch.full((nc,), 1.0e11)
    diag = compute_crm_lightning_and_lnox(rho_d, theta_m, q_v, q_i, q_p, w_stag, z_c, dz_c, area)
    assert diag.flash_tot_sub[0, 0, 0].item() > 10.0 * diag.flash_tot_sub[1, 0, 0].item()
    assert diag.flash_tot_sub[0, 0, 0].item() > 0.0 and diag.flash_tot_sub[2, 0, 0].item() == 0.0
    assert 0.05 <= diag.f_cg_sub[0, 0, 0].item() <= 0.50

    # Exact global LNOx mass conservation: integral(n_dep_lnox) == integral(3.5 * flash_tot)
    prod = (3.5 * diag.flash_tot_cell.double() * area.double()).sum().item()
    dep = (diag.n_dep_lnox_cell.double() * area.double()).sum().item()
    rel = abs(prod - dep) / max(prod, 1.0e-30)
    assert rel < 1.0e-6, f"LNOx deposition conservation error: {rel:.3e}"
    # Background deposition reaches non-storm cell 2 too!
    assert diag.n_dep_lnox_cell[2].item() > 0.0
    m = diagnose_lightning_metrics(diag, area, torch.tensor([SURF_LAND, SURF_OCEAN, SURF_LAND, SURF_OCEAN]))
    print(f"[PASS] CRM Lightning & LNOx: fl={m['flash_tot_per_s']:.3f}/s, CG={m['cg_frac_pct']:.1f}%, LNOx_err={rel:.2e}")


def test_maml_patch_wildfire_mosaic() -> None:
    """Verify MAML discrete-patch SPITFIRE burns a subset of 4x4 patches and opens canopy mosaic."""
    nc, ny, nx = 1, 4, 4
    m_sub = nc * ny * nx
    st = torch.full((m_sub,), SURF_LAND, dtype=torch.int64)
    veg = torch.ones(m_sub, dtype=torch.bool)
    bio = init_biosphere_state((m_sub,), veg)
    bio.lai[:] = 3.5
    bio.lai_evg[:] = 2.0
    bio.lai_dec[:] = 1.5
    bio.h_canopy[:] = 10.0

    lc = init_land_carbon_state(st, bio.lai, bio.h_canopy, maml_n_sub=ny * nx)
    lc.fire_di[:] = 3500.0  # Dry season
    lc.c_lit_fast[:] = 0.80
    lc.c_lit_slow[:] = 0.60

    t_soil = torch.full((5, m_sub), 302.0)
    w_liq = torch.full((5, m_sub), 0.08)
    w_ice = torch.zeros(5, m_sub)
    f_snow = torch.zeros(m_sub)
    t_sfc = torch.full((m_sub,), 306.0)
    vpd = torch.full((m_sub,), 3200.0)
    u10 = torch.full((m_sub,), 7.0)
    precip = torch.zeros(m_sub)
    gpp = torch.full((m_sub,), 4.0)
    rleaf = torch.full((m_sub,), 0.6)
    # Dry lightning strike across savanna cell
    cg = torch.full((m_sub,), 2.0e-12)

    burned_any = torch.zeros(m_sub, dtype=torch.bool)
    for _ in range(30):
        step_land_carbon_and_fire(
            lc, st, gpp, rleaf, bio.lai, bio.h_canopy, t_soil, w_liq, w_ice, f_snow, t_sfc, vpd, u10, precip,
            dt=86400.0, enable_nitrogen=True, flash_cg_m2s=cg, bio=bio,
            enable_fire_veg_feedback=True, maml_shape=(nc, ny, nx), crm_dx_m=4000.0,
        )
        burned_any |= (lc.patch_burned_flag > 0.5)

    n_burnt = int(burned_any.sum().item())
    hc_spread = (bio.h_canopy.max() - bio.h_canopy.min()).item()
    lai_spread = (bio.lai.max() - bio.lai.min()).item()
    assert 0 < n_burnt < m_sub, f"Expected partial MAML patch mosaic, got {n_burnt}/{m_sub} burnt"
    assert hc_spread > 1.5 and lai_spread > 1.0, f"Expected fire-created canopy heterogeneity: dh={hc_spread:.2f}, dLAI={lai_spread:.2f}"
    print(f"[PASS] MAML 4x4 fire patch mosaic: {n_burnt}/{m_sub} burnt, h_c=[{bio.h_canopy.min():.1f}, {bio.h_canopy.max():.1f}] m, LAI=[{bio.lai.min():.2f}, {bio.lai.max():.2f}]")


def test_3d_crm_coupled_cpu_mps_parity() -> None:
    """Verify full-ESM Glevel-1 3D CRM (2x2) runs cleanly on both CPU and MPS with tight parity."""
    def _run(dev: str) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        cfg = AOBAConfig(
            glevel=1, nz=24, z_top=24000.0, z_sponge_base=18000.0,
            crm_ny=2, crm_nx=2, dt_host=600.0, n_crm_dyn_substeps=6, rad_stride=2,
            use_earth2014=False,
            ocean_mode="ogcm_3d", enable_ocean_bgc=True, enable_land_carbon=True,
            enable_land_nitrogen=True, enable_lightning=True, enable_fire_veg_feedback=True,
            enable_natural_aerosols=True, enable_dust=True,
        )
        m = init_aoba_model(cfg, device=dev)
        for _ in range(4):
            step_aoba_model(m)
        assert m.last_lightning is not None
        assert m.land_carbon is not None and m.land_carbon.n_min is not None
        return {
            "u": m.atmos.u_edge.detach().cpu(),
            "thm": m.atmos.theta_m.detach().cpu(),
            "crm_u": m.crm.u.detach().cpu(),
            "n_min": m.land_carbon.n_min.detach().cpu(),
            "fl": m.last_lightning.flash_tot_cell.detach().cpu(),
        }

    c = _run("cpu")
    if torch.backends.mps.is_available():
        g = _run("mps")
        for k in c:
            assert torch.isfinite(c[k]).all().item() and torch.isfinite(g[k]).all().item()
            rel = (c[k] - g[k]).abs().max().item() / max(c[k].abs().max().item(), 1.0e-12)
            print(f"  3D CRM Full-ESM CPU vs MPS {k:6s}: rel={rel:.3e}")
            assert rel < 5.0e-3, f"CPU vs MPS mismatch in {k}: rel={rel:.3e}"
    print("[PASS] 3D CRM Full-ESM coupled step (CPU + MPS)")


def test_spitfire_cured_grass_slope_and_gpu_parity() -> None:
    """Verify cured C4 savanna grass fuel, Rothermel slope factor phi_s, and CPU-vs-GPU fire parity."""
    # 3 savanna/forest land cells: 0=flat forest (f_c4=0, slope=0), 1=flat C4 savanna (f_c4=0.9, slope=0), 2=steep C4 savanna (f_c4=0.9, slope=0.5)
    n = 3
    st = torch.full((n,), SURF_LAND, dtype=torch.int64)
    lai = torch.tensor([2.5, 2.5, 2.5])
    hc = torch.tensor([10.0, 1.0, 1.0])
    f_c4 = torch.tensor([0.0, 0.9, 0.9])
    slope = torch.tensor([0.0, 0.0, 0.5])

    def _run_case(dev: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = torch.device(dev)
        lc = init_land_carbon_state(st.to(d), lai.to(d), hc.to(d), device=d)
        # Thin litter (0.09 kg C/m^2) where forest barely sustains fire, while cured C4 grass burns strongly
        lc.c_lit_fast.fill_(0.07)
        lc.c_lit_slow.fill_(0.06)
        lc.c_leaf.fill_(0.25)
        lc.fire_di.fill_(2500.0)
        t_soil = torch.full((5, n), 302.0, device=d)
        w_liq = torch.full((5, n), 0.08, device=d)
        w_ice = torch.zeros((5, n), device=d)
        f_snow = torch.zeros((n,), device=d)
        t_sfc = torch.full((n,), 306.0, device=d)
        vpd = torch.full((n,), 3000.0, device=d)
        u10 = torch.full((n,), 5.0, device=d)
        precip = torch.zeros((n,), device=d)
        gpp = torch.full((n,), 4.0, device=d)
        rleaf = torch.full((n,), 0.5, device=d)
        cg = torch.full((n,), 1.0e-11, device=d)
        step_land_carbon_and_fire(
            lc, st.to(d), gpp, rleaf, lai.to(d), hc.to(d), t_soil, w_liq, w_ice, f_snow, t_sfc, vpd, u10, precip,
            dt=86400.0, enable_nitrogen=True, f_c4=f_c4.to(d), slope_tan=slope.to(d), flash_cg_m2s=cg,
        )
        return lc.fire_intensity_kw_m.detach().cpu(), lc.fire_c_kgc_m2s.detach().cpu()

    ib_c, fc_c = _run_case("cpu")
    assert ib_c[1].item() > 2.0 * ib_c[0].item(), f"Expected cured C4 grass to boost fireline intensity: {ib_c.tolist()}"
    assert ib_c[2].item() > ib_c[1].item() * 1.10, f"Expected slope phi_s to accelerate fireline intensity: {ib_c.tolist()}"
    assert fc_c[1].item() > fc_c[0].item() and fc_c[2].item() > fc_c[1].item()

    if torch.backends.mps.is_available():
        ib_g, fc_g = _run_case("mps")
        rel_ib = float((ib_c - ib_g).abs().max() / ib_c.abs().max().clamp(min=1e-6))
        rel_fc = float((fc_c - fc_g).abs().max() / fc_c.abs().max().clamp(min=1e-18))
        assert rel_ib < 1.0e-4 and rel_fc < 1.0e-4, f"SPITFIRE CPU vs MPS mismatch: ib={rel_ib:.3e}, fc={rel_fc:.3e}"
        print(
            f"[PASS] SPITFIRE 1h cured-grass + 100h litter + slope phi_s (CPU vs MPS): "
            f"I_B={[round(v, 1) for v in ib_c.tolist()]} kW/m (rel_ib={rel_ib:.2e}, rel_fc={rel_fc:.2e})"
        )
    else:
        print(f"[PASS] SPITFIRE 1h cured-grass + 100h litter + slope phi_s: I_B={[round(v, 1) for v in ib_c.tolist()]} kW/m")


if __name__ == "__main__":
    test_bnf_temperature_optimality()
    test_exact_cn_conservation_and_functional_balance()
    test_crm_lightning_and_lnox_conservation()
    test_maml_patch_wildfire_mosaic()
    test_spitfire_cured_grass_slope_and_gpu_parity()
    test_3d_crm_coupled_cpu_mps_parity()
    print("\nALL CN-MODEL, CRM LIGHTNING, AND MAML SPITFIRE TESTS PASSED!")
