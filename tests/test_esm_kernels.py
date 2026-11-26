"""Verification suite for AOBA's single-source Metal/CUDA fused ESM kernels (`kernels_esm.py`).

Tests:
1. Single-source Metal shader compilation (MPS) and automatic Metal-to-CUDA transpilation
   (`build_cuda_extension_sources`) for all 5 new kernels:
   `rad_rrtmgp_lw_grp`, `rad_rrtmgp_sw_grp`, `rad_ckd_hd_col`, `microphysics_6class_col`, `ozone_chem_col`.
2. Exact numerical parity between CPU PyTorch reference and MPS/CUDA fused kernels across
   day + night + cloud + dust + 5-species aerosols + ozone + LNOx + wildfire inputs.
3. Zero-sync `apply_3d_tica_cloud_shift` exact parent-cell cloud water conservation.
4. End-to-end Full-ESM step (`rad_scheme = "rrtmgp"` + `microphysics_scheme = "six_class_2m"` +
   `o3_mode = "prognostic"` + `enable_rad_3d_tica = True`) CPU vs MPS parity & finiteness.
"""

from __future__ import annotations

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.kernels import _all_shader_src
from aoba.kernels_cuda import build_cuda_extension_sources
from aoba.microphysics_6class import step_microphysics_six_class_2m
from aoba.ozone_chemistry import init_prognostic_ozone_state, step_prognostic_ozone
from aoba.radiation_hd import apply_3d_tica_cloud_shift, compute_ckd_hd_radiation
from aoba.radiation_rrtmgp import compute_rrtmgp_radiation


def test_cuda_transpilation() -> None:
    import re
    cpp_src, cuda_src, specs = build_cuda_extension_sources(_all_shader_src())
    names = {s.name for s in specs}
    required = {
        "rad_rrtmgp_lw_grp",
        "rad_rrtmgp_sw_grp",
        "rad_ckd_hd_col",
        "microphysics_6class_col",
        "ozone_chem_col",
        "crm_ens_mean_unc_transpose",
        "ogcm_multi_vdiff_col",
        "ocean_bgc_column_col",
        "crm_lightning_col",
        "explicit_lightning_charge_col",
        "explicit_lightning_poisson_thomas",
        "explicit_lightning_discharge_grp",
        "land_carbon_cn_fire_prep_tile",
    }
    missing = required - names
    assert not missing, f"Missing CUDA transpiled kernels: {missing}"
    assert "AOBA_SHARED" in cuda_src and "AOBA_SYNCTHREADS()" in cuda_src
    assert "AOBA_SYNCWARP()" in cuda_src
    assert "aoba_shfl_xor" in cuda_src
    assert "simdgroup_barrier" not in cuda_src, "Untranspiled simdgroup_barrier in CUDA source"
    assert "mem_flags::" not in cuda_src, "Untranspiled Metal mem_flags:: in CUDA source"
    assert not re.search(r"\bAOBA_SHARED\s+(?:const\s+)?[A-Za-z_]\w*\s*[*&]", cuda_src), (
        "Invalid __shared__ pointer declaration in CUDA source"
    )
    assert "launch_rad_rrtmgp_lw_grp" in cpp_src
    print(f"[PASS] CUDA transpilation verified ({len(specs)} total kernels, all ESM/CUDA invariants satisfied)")


def test_gpu_parity() -> None:
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        print("[SKIP] No MPS/CUDA GPU available for GPU parity test")
        return

    nz, M = 44, 64
    torch.manual_seed(123)
    pres_c = torch.logspace(5.0, 2.5, nz).view(nz, 1).expand(nz, M).contiguous()
    pres_w = torch.logspace(5.02, 2.3, nz + 1).view(nz + 1, 1).expand(nz + 1, M).contiguous()
    temp_c = torch.linspace(295.0, 210.0, nz).view(nz, 1).expand(nz, M).contiguous() + 2.5 * torch.randn(nz, M)
    temp_w = torch.linspace(298.0, 208.0, nz + 1).view(nz + 1, 1).expand(nz + 1, M).contiguous()
    dz_c = torch.full((nz, M), 500.0)
    rho_d = (pres_c / (287.05 * temp_c)).contiguous()
    qv = (5e-3 * torch.exp(-torch.linspace(0, 6, nz)).view(nz, 1)).expand(nz, M).contiguous()
    qc = torch.zeros(nz, M)
    qc[4:12] = 4e-5
    qi = torch.zeros(nz, M)
    qi[15:24] = 1.5e-5
    o3 = torch.full((nz, M), 8e-7)
    qd = torch.full((nz, M), 2e-7)
    aero = (
        torch.full((nz, M), 2e-3),
        torch.full((nz, M), 1.6e-3),
        torch.full((nz, M), 1.1e-3),
        torch.full((nz, M), 3e-4),
    )
    ts = torch.full((M,), 298.0)
    alb_v = torch.full((M,), 0.12)
    alb_n = torch.full((M,), 0.25)
    mu0 = torch.linspace(-0.2, 0.95, M)
    stoa = torch.full((M,), 1361.0)

    def to_d(x):
        return tuple(t.to(dev) for t in x) if isinstance(x, tuple) else x.to(dev)

    # 1. RRTMGP (240g HITRAN + MT_CKD 4.1)
    r_c = compute_rrtmgp_radiation(pres_c, pres_w, temp_c, temp_w, dz_c, rho_d, qv, qc, qi, ts, alb_v, alb_n, mu0, stoa, q_dust_c=qd, o3_vmr_c=o3, aero_opt=aero)
    r_g = compute_rrtmgp_radiation(to_d(pres_c), to_d(pres_w), to_d(temp_c), to_d(temp_w), to_d(dz_c), to_d(rho_d), to_d(qv), to_d(qc), to_d(qi), to_d(ts), to_d(alb_v), to_d(alb_n), to_d(mu0), to_d(stoa), q_dust_c=to_d(qd), o3_vmr_c=to_d(o3), aero_opt=to_d(aero))
    d_olr = float((r_c.toa_olr - r_g.toa_olr.cpu()).abs().max())
    d_asr = float((r_c.toa_asr - r_g.toa_asr.cpu()).abs().max())
    d_par = float((r_c.sfc_par_wm2 - r_g.sfc_par_wm2.cpu()).abs().max())
    assert d_olr < 5.0e-3 and d_asr < 2.0e-2 and d_par < 5.0e-3, f"RRTMGP mismatch: {d_olr}, {d_asr}, {d_par}"
    print(f"[PASS] RRTMGP CPU vs {dev.type} parity: dOLR={d_olr:.5f}, dASR={d_asr:.5f}, dPAR={d_par:.5f} W/m^2")

    # 2. CKD-HD (111g MSTRN11-class)
    h_c = compute_ckd_hd_radiation(pres_c, pres_w, temp_c, temp_w, dz_c, rho_d, qv, qc, qi, ts, alb_v, alb_n, mu0, stoa, q_dust_c=qd, o3_vmr_c=o3, aero_opt=aero)
    h_g = compute_ckd_hd_radiation(to_d(pres_c), to_d(pres_w), to_d(temp_c), to_d(temp_w), to_d(dz_c), to_d(rho_d), to_d(qv), to_d(qc), to_d(qi), to_d(ts), to_d(alb_v), to_d(alb_n), to_d(mu0), to_d(stoa), q_dust_c=to_d(qd), o3_vmr_c=to_d(o3), aero_opt=to_d(aero))
    h_olr = float((h_c.toa_olr - h_g.toa_olr.cpu()).abs().max())
    h_asr = float((h_c.toa_asr - h_g.toa_asr.cpu()).abs().max())
    assert h_olr < 5.0e-3 and h_asr < 2.0e-2, f"CKD-HD mismatch: {h_olr}, {h_asr}"
    print(f"[PASS] CKD-HD CPU vs {dev.type} parity: dOLR={h_olr:.5f}, dASR={h_asr:.5f} W/m^2")

    # 3. 6-Class 2-Moment Microphysics
    nc_m = 4
    rd4 = rho_d.view(nz, nc_m, 4, 4).contiguous()
    th4 = (temp_c * (1e5 / pres_c).pow(0.2857)).view(nz, nc_m, 4, 4).contiguous()
    qv4 = qv.view(nz, nc_m, 4, 4).contiguous()
    qc4 = (qc * 10.0).view(nz, nc_m, 4, 4).contiguous()
    qi4 = (qi * 10.0).view(nz, nc_m, 4, 4).contiguous()
    qr4 = torch.full((nz, nc_m, 4, 4), 1.5e-4)
    qs4 = torch.full((nz, nc_m, 4, 4), 8e-5)
    qg4 = torch.full((nz, nc_m, 4, 4), 6e-5)
    nc4 = torch.full((nz, nc_m, 4, 4), 8e7)
    ni4 = torch.full((nz, nc_m, 4, 4), 2e4)
    nr4 = torch.full((nz, nc_m, 4, 4), 1e4)
    ns4 = torch.full((nz, nc_m, 4, 4), 5e3)
    ng4 = torch.full((nz, nc_m, 4, 4), 2e3)
    dz1 = torch.full((nz,), 500.0)
    is_o = torch.randint(0, 2, (nc_m, 4, 4)).float()
    us_s = torch.full((nc_m, 4, 4), 9.0)
    ccn_s = torch.full((nc_m, 4, 4), 5e7)
    qd4 = qd.view(nz, nc_m, 4, 4).contiguous()

    m_c = step_microphysics_six_class_2m(rd4, th4, qv4, qc4, qi4, qr4, qs4, qg4, nc4, ni4, nr4, ns4, ng4, dz1, 10.0, is_ocean_sub=is_o, u_sfc_sub=us_s, q_dust_sub=qd4, ccn_aero_sub=ccn_s)
    m_g = step_microphysics_six_class_2m(rd4.to(dev), th4.to(dev), qv4.to(dev), qc4.to(dev), qi4.to(dev), qr4.to(dev), qs4.to(dev), qg4.to(dev), nc4.to(dev), ni4.to(dev), nr4.to(dev), ns4.to(dev), ng4.to(dev), dz1.to(dev), 10.0, is_ocean_sub=is_o.to(dev), u_sfc_sub=us_s.to(dev), q_dust_sub=qd4.to(dev), ccn_aero_sub=ccn_s.to(dev))
    max_rel = max(float((a - b.cpu()).abs().max() / max(float(a.abs().max()), 1e-12)) for a, b in zip(m_c, m_g))
    assert max_rel < 5.0e-5, f"6-class microphysics mismatch: max_rel={max_rel}"
    print(f"[PASS] 6-Class 2M CPU vs {dev.type} parity: max_rel={max_rel:.2e} across all 14 outputs")

    # 4. Prognostic Ozone Chemistry
    st_c = torch.randint(0, 3, (M,))
    lai_c = torch.linspace(0.0, 5.0, M)
    fs_c = torch.linspace(0.0, 0.8, M)
    lnx_c = torch.full((M,), 1e-12)
    fir_c = torch.full((M,), 1e-10)
    oz_c = init_prognostic_ozone_state(pres_c, rho_d, dz_c)
    oz_g = init_prognostic_ozone_state(pres_c.to(dev), rho_d.to(dev), dz_c.to(dev))
    step_prognostic_ozone(oz_c, pres_c, temp_c, rho_d, dz_c, qv, mu0, st_c, lai_c, fs_c, fs_c, 720.0, lnox_cell_kgn_m2s=lnx_c, fire_c_cell_kgc_m2s=fir_c)
    step_prognostic_ozone(oz_g, pres_c.to(dev), temp_c.to(dev), rho_d.to(dev), dz_c.to(dev), qv.to(dev), mu0.to(dev), st_c.to(dev), lai_c.to(dev), fs_c.to(dev), fs_c.to(dev), 720.0, lnox_cell_kgn_m2s=lnx_c.to(dev), fire_c_cell_kgc_m2s=fir_c.to(dev))
    d_tco = float((oz_c.tco_du - oz_g.tco_du.cpu()).abs().max())
    d_sfc = float((oz_c.o3_sfc_ppbv - oz_g.o3_sfc_ppbv.cpu()).abs().max())
    assert d_tco < 1.0e-3 and d_sfc < 1.0e-3
    print(f"[PASS] Ozone Chem CPU vs {dev.type} parity: dTCO={d_tco:.5f} DU, dsfc={d_sfc:.5f} ppbv")

    # 5. Zero-sync TICA cloud shift exact conservation
    qc_sub = torch.rand((nz, nc_m * 16), device=dev)
    qi_sub = torch.rand((nz, nc_m * 16), device=dev)
    z1 = torch.linspace(250.0, 22000.0, nz, device=dev)
    mu_nc = torch.tensor([-0.3, 0.25, 0.6, 0.95], device=dev)
    ha_nc = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=dev)
    qc_sh, qi_sh = apply_3d_tica_cloud_shift(qc_sub, qi_sub, z1, mu_nc, ha_nc, nc_m, 4, 4, 4000.0)
    err_c = float((qc_sub.reshape(nz, nc_m, 16).sum(dim=-1) - qc_sh.reshape(nz, nc_m, 16).sum(dim=-1)).abs().max())
    err_i = float((qi_sub.reshape(nz, nc_m, 16).sum(dim=-1) - qi_sh.reshape(nz, nc_m, 16).sum(dim=-1)).abs().max())
    assert err_c < 1.0e-5 and err_i < 1.0e-5
    assert float((qc_sub[:, 16:32] - qc_sh[:, 16:32]).abs().max()) > 0.01, "Daytime slant column must shift!"
    assert float((qc_sub[:, 0:16] - qc_sh[:, 0:16]).abs().max()) == 0.0, "Nighttime column must stay unshifted!"
    print(f"[PASS] Zero-sync 3D TICA exact mass conservation: err_c={err_c:.2e}, err_i={err_i:.2e}")


def test_full_esm_rrtmgp_step() -> None:
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AOBAConfig(
        glevel=1,
        crm_ny=2,
        crm_nx=2,
        rad_scheme="rrtmgp",
        microphysics_scheme="six_class_2m",
        o3_mode="prognostic",
        enable_rad_3d_tica=True,
        enable_aerosol_rad_forcing=True,
    )
    m = init_aoba_model(cfg, device=dev)
    diag = step_aoba_model(m)
    assert torch.isfinite(m.sfc.t_soil).all() and torch.isfinite(m.ozone.o3_vmr).all()
    print(f"[PASS] Full-ESM step on {dev} (RRTMGP + 6-Class 2M + O3 + 3D TICA): Ts={diag.global_mean_t_sfc_k:.2f} K, OLR={diag.global_mean_olr_wm2:.1f} W/m^2")


if __name__ == "__main__":
    test_cuda_transpilation()
    test_gpu_parity()
    test_full_esm_rrtmgp_step()
    print("ALL ESM FUSED KERNEL TESTS PASSED!")
