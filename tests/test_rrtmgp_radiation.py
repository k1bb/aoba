"""Verification & RFMIP-style benchmark of official HITRAN/MT_CKD-4.1 RRTMGP vs CKD / CKD-HD."""

from __future__ import annotations

import math
import time
import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.radiation import background_ozone_vmr, compute_four_band_radiation
from aoba.radiation_ckd import compute_ckd_radiation
from aoba.radiation_hd import compute_ckd_hd_radiation
from aoba.radiation_rrtmgp import compute_rrtmgp_radiation


def _make_standard_44l_profile(n_col: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Construct a standard 44-layer clear-sky mid-latitude summer atmosphere (0..48 km)."""
    nz = 44
    z_w = torch.linspace(0.0, 48000.0, nz + 1, device=device)
    z_c = 0.5 * (z_w[:-1] + z_w[1:])
    dz_c = z_w[1:] - z_w[:-1]
    # Standard mid-latitude summer T(z): 294 K surface, 215 K tropopause @ 13 km, 270 K stratopause @ 48 km
    t_1d = torch.where(
        z_c < 13000.0,
        294.0 - 0.0061 * z_c,
        torch.where(z_c < 20000.0, torch.full_like(z_c, 214.7), 214.7 + 0.0020 * (z_c - 20000.0)),
    )
    t_w_1d = torch.empty(nz + 1, device=device)
    t_w_1d[0] = 294.2
    t_w_1d[1:-1] = 0.5 * (t_1d[:-1] + t_1d[1:])
    t_w_1d[-1] = t_1d[-1]
    # Hydrostatic pressure integration
    p_w_1d = torch.empty(nz + 1, device=device)
    p_w_1d[0] = 101325.0
    for k in range(nz):
        p_w_1d[k + 1] = p_w_1d[k] * torch.exp(-C.GRAVITY * dz_c[k] / (C.R_D * t_1d[k]))
    p_c_1d = torch.sqrt(p_w_1d[:-1] * p_w_1d[1:])
    rho_1d = p_c_1d / (C.R_D * t_1d)
    # Realistic MLS water vapor (~12 g/kg surface, 3 ppmv stratosphere)
    qv_1d = torch.where(z_c < 13000.0, 0.012 * torch.exp(-z_c / 2400.0), torch.full_like(z_c, 3.0e-6))
    o3_1d = background_ozone_vmr(p_c_1d)

    def _ex(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).expand(-1, n_col).contiguous()

    return dict(
        pres_c=_ex(p_c_1d),
        pres_w=_ex(p_w_1d),
        temp_c=_ex(t_1d),
        temp_w=_ex(t_w_1d),
        dz_c=_ex(dz_c),
        rho_d_c=_ex(rho_1d),
        qv_c=_ex(qv_1d),
        qc_c=torch.zeros((nz, n_col), device=device),
        qi_c=torch.zeros((nz, n_col), device=device),
        t_sfc=torch.full((n_col,), 294.2, device=device),
        albedo_vis=torch.full((n_col,), 0.15, device=device),
        albedo_nir=torch.full((n_col,), 0.25, device=device),
        mu0=torch.full((n_col,), 0.50, device=device),
        s_toa=torch.full((n_col,), 1361.0, device=device),
        o3_vmr_c=_ex(o3_1d),
    )


def test_rrtmgp_accuracy_and_rfmip_forcings() -> None:
    """Verify RRTMGP HITRAN accuracy on clear-sky fluxes, GHG forcings (CO2/CH4/N2O), and CPU/MPS parity."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prof = _make_standard_44l_profile(4, dev)

    def _run(fn, co2=284.3, ch4=0.80825, n2o=0.27302):
        kw = dict(prof, co2_ppmv=co2, o3_lw_coupling=True)
        if fn is not compute_four_band_radiation:
            kw["ch4_ppmv"] = ch4
            kw["n2o_ppmv"] = n2o
        return fn(**kw)

    print("=== Clear-Sky 44-Layer Standard Atmosphere Comparison ===")
    results = {}
    for name, fn in (
        ("four_band", compute_four_band_radiation),
        ("ckd (50g)", compute_ckd_radiation),
        ("ckd_hd (111g)", compute_ckd_hd_radiation),
        ("rrtmgp (240g HITRAN)", compute_rrtmgp_radiation),
    ):
        r0 = _run(fn, co2=284.3, ch4=0.80825, n2o=0.27302)
        r_2c = _run(fn, co2=568.6)
        r_4c = _run(fn, co2=1137.2)
        r_2m = _run(fn, ch4=1.6165)
        r_2n = _run(fn, n2o=0.54604)

        olr = float(r0.toa_olr[0].item())
        asr = float(r0.toa_asr[0].item())
        lw_dn_s = float(r0.lw_flux_dn[0, 0].item())
        sw_dn_s = float(r0.sw_flux_dn[0, 0].item())
        irf_2c = float((r0.toa_olr[0] - r_2c.toa_olr[0] + r_2c.toa_asr[0] - r0.toa_asr[0]).item())
        irf_4c = float((r0.toa_olr[0] - r_4c.toa_olr[0] + r_4c.toa_asr[0] - r0.toa_asr[0]).item())
        irf_2m = float((r0.toa_olr[0] - r_2m.toa_olr[0] + r_2m.toa_asr[0] - r0.toa_asr[0]).item())
        irf_2n = float((r0.toa_olr[0] - r_2n.toa_olr[0] + r_2n.toa_asr[0] - r0.toa_asr[0]).item())
        # Stratospheric cooling rate at ~10 hPa (~31 km, k ~ 28) [K/day]
        k10 = int((prof["pres_c"][:, 0] - 1000.0).abs().argmin().item())
        q_lw_10 = float(r0.q_rad_lw[k10, 0].item()) * 86400.0
        q_sw_10 = float(r0.q_rad_sw[k10, 0].item()) * 86400.0
        results[name] = (olr, asr, lw_dn_s, irf_2c, irf_4c, irf_2m, irf_2n)
        print(
            f"  {name:22s}: OLR={olr:6.2f} ASR={asr:6.2f} LWdn_s={lw_dn_s:6.2f} SWdn_s={sw_dn_s:6.2f} | "
            f"IRF 2xCO2={irf_2c:+5.2f} 4xCO2={irf_4c:+5.2f} 2xCH4={irf_2m:+4.2f} 2xN2O={irf_2n:+4.2f} W/m2 | "
            f"10hPa Q_lw={q_lw_10:+5.2f} Q_sw={q_sw_10:+5.2f} K/d"
        )

    olr_r, asr_r, lwdn_r, f2c, f4c, f2m, f2n = results["rrtmgp (240g HITRAN)"]
    # Physical RFMIP/textbook bounds for mid-latitude summer clear-sky atmosphere
    assert 260.0 < olr_r < 305.0, f"Unexpected RRTMGP OLR: {olr_r}"
    assert 450.0 < asr_r < 560.0, f"Unexpected RRTMGP ASR: {asr_r}"
    assert 300.0 < lwdn_r < 390.0, f"Unexpected RRTMGP LWdn_sfc: {lwdn_r}"
    assert 2.2 < f2c < 4.5, f"Unexpected RRTMGP 2xCO2 IRF: {f2c}"
    assert 4.8 < f4c < 9.0, f"Unexpected RRTMGP 4xCO2 IRF: {f4c}"
    assert 0.25 < f2m < 1.2, f"Unexpected RRTMGP 2xCH4 IRF: {f2m}"
    assert 0.40 < f2n < 1.8, f"Unexpected RRTMGP 2xN2O IRF: {f2n}"

    # CPU vs MPS parity
    if torch.backends.mps.is_available():
        p_cpu = _make_standard_44l_profile(2, torch.device("cpu"))
        p_mps = _make_standard_44l_profile(2, torch.device("mps"))
        rc = compute_rrtmgp_radiation(**p_cpu, o3_lw_coupling=True)
        rm = compute_rrtmgp_radiation(**p_mps, o3_lw_coupling=True)
        diff_olr = abs(float(rc.toa_olr[0].item()) - float(rm.toa_olr[0].item()))
        diff_asr = abs(float(rc.toa_asr[0].item()) - float(rm.toa_asr[0].item()))
        assert diff_olr < 0.05 and diff_asr < 0.05, f"CPU/MPS mismatch: dOLR={diff_olr}, dASR={diff_asr}"
        print(f"[PASS] RRTMGP CPU vs MPS parity: dOLR={diff_olr:.4f} W/m2, dASR={diff_asr:.4f} W/m2")


def test_rrtmgp_speed_and_coupled_smoke() -> None:
    """Measure MPS speed on 1024 columns x 44 layers & smoke-test coupled AOBA with rad_scheme='rrtmgp'."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prof_b = _make_standard_44l_profile(1024, dev)
    # Warmup
    _ = compute_ckd_radiation(**prof_b, o3_lw_coupling=True)
    _ = compute_rrtmgp_radiation(**prof_b, o3_lw_coupling=True)
    if dev.type == "mps":
        torch.mps.synchronize()
    for name, fn in (("ckd (50g Metal)", compute_ckd_radiation), ("rrtmgp (240g HITRAN)", compute_rrtmgp_radiation)):
        t0 = time.perf_counter()
        _ = fn(**prof_b, o3_lw_coupling=True)
        if dev.type == "mps":
            torch.mps.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"[BENCH] {name:22s} (1024 cols x 44L on {dev.type}): {ms:6.1f} ms")

    cfg = AOBAConfig(
        glevel=1, nz=14, z_top=32000.0, z_sponge_base=24000.0, ocean_nz=4,
        crm_nx=2, crm_ny=2, dt_host=600.0, n_crm_dyn_substeps=6, rad_stride=1,
        rad_scheme="rrtmgp", o3_mode="prognostic",
        enable_land_carbon=True, enable_ocean_bgc=False,
        use_earth2014=True, earth2014_stride=8,
    )
    m = init_aoba_model(cfg, device=str(dev))
    for _ in range(2):
        d = step_aoba_model(m, compute_diagnostics=True, sync_host_surface=True)
        assert math.isfinite(d.global_mean_olr_wm2) and math.isfinite(d.global_mean_asr_wm2)
    print(f"[PASS] Coupled AOBA + rad_scheme='rrtmgp' on {dev.type}: Ts={d.global_mean_t_sfc_k:.2f} K, OLR={d.global_mean_olr_wm2:.1f} W/m2")


if __name__ == "__main__":
    test_rrtmgp_accuracy_and_rfmip_forcings()
    test_rrtmgp_speed_and_coupled_smoke()
    print("ALL RRTMGP HITRAN RADIATION TESTS PASSED!")
