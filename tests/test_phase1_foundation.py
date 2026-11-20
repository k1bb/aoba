"""Unit, conservation, and physical-limit tests for Phase 1 foundation & 4-band radiation."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aoba import (
    append_zarr,
    build_vertical_grid,
    compute_four_band_radiation,
    compute_solar_zenith,
    compute_stf_heights,
    constants as C,
    diagnose_pressure_temperature,
    get_device,
    liquid_fraction_condensate,
    load_zarr,
    qsat_mixed,
    rcemip_analytic_sounding,
    saturation_adjustment,
    save_zarr,
    solve_tridiagonal_batched,
)
from aoba.kernels import solve_tridiagonal_pytorch
from aoba.radiation import toon_two_stream_lw, toon_two_stream_sw


def test_tridiagonal_solver_cpu_and_mps():
    """Verify batched Thomas solver against dense linear solve and check CPU-MPS parity."""
    torch.manual_seed(42)
    for K in [32, 64]:
        M = 128
        dl = -0.3 * torch.rand(K, M, dtype=torch.float32)
        du = -0.3 * torch.rand(K, M, dtype=torch.float32)
        d = 1.5 + torch.rand(K, M, dtype=torch.float32)
        dl[0] = 0.0
        du[-1] = 0.0
        rhs = torch.randn(K, M, dtype=torch.float32)

        x_cpu = solve_tridiagonal_batched(dl, d, du, rhs)

        # Reconstruct residual dl*x_{k-1} + d*x_k + du*x_{k+1} - rhs
        res = d * x_cpu - rhs
        res[1:] += dl[1:] * x_cpu[:-1]
        res[:-1] += du[:-1] * x_cpu[1:]
        assert float(res.abs().max().item()) < 2.0e-5

        if torch.backends.mps.is_available():
            dev = torch.device("mps")
            x_mps = solve_tridiagonal_batched(dl.to(dev), d.to(dev), du.to(dev), rhs.to(dev))
            x_ref = solve_tridiagonal_pytorch(dl.to(dev), d.to(dev), du.to(dev), rhs.to(dev))
            assert float((x_mps - x_ref).abs().max().item()) < 2.0e-5
            assert float((x_mps.cpu() - x_cpu).abs().max().item()) < 2.0e-5


def test_vertical_grid_and_canopy_datum_ordering():
    """Verify Charney-Phillips grid spacing, Klemp (2011) STF positivity, and canopy datum."""
    vgrid = build_vertical_grid(nz=32, z_top=24000.0, dz_min=60.0)
    assert vgrid.zeta_w.shape == (33,)
    assert vgrid.zeta_c.shape == (32,)
    assert abs(float(vgrid.dzeta_c[0].item()) - 60.0) < 1.0e-4
    assert abs(float(vgrid.zeta_w[-1].item()) - 24000.0) < 1.0e-2
    assert torch.all(vgrid.dzeta_c > 0.0)

    # Test extreme topography from Dead Sea (-415 m) to Tibetan/Himalayan plateau (6500 m)
    # plus tall tropical forest canopy (h_c = 45 m, d = 30 m => h_c - d = 15 m < tilde_z_1 = 30 m)
    z_topo = torch.tensor([-415.0, 0.0, 2500.0, 6500.0])
    h_c = torch.tensor([2.0, 45.0, 20.0, 0.0])
    d_disp = (2.0 / 3.0) * h_c
    z_datum = z_topo + d_disp

    z_w, z_c, dz_c = compute_stf_heights(vgrid, z_datum)
    assert torch.all(dz_c > 10.0)
    tilde_z1 = z_c[0] - z_datum
    assert torch.all(tilde_z1 > (h_c - d_disp))


def test_moist_thermodynamics_and_saturation_adjustment():
    """Verify state equation inversion and moist enthalpy conservation in saturation adjustment."""
    rho_d = torch.tensor([1.15, 0.75, 0.35], dtype=torch.float32)
    qv = torch.tensor([0.015, 0.004, 0.0002], dtype=torch.float32)
    theta = torch.tensor([300.0, 315.0, 335.0], dtype=torch.float32)
    theta_m = theta * (1.0 + C.RV_OVER_RD * qv)

    pres, exner, temp = diagnose_pressure_temperature(rho_d, theta_m, qv)
    assert torch.allclose(temp, exner * theta, atol=1.0e-4)

    # Mixed-phase ramp bounds (Khairoutdinov & Randall, 2003, Eq. A13)
    temps_test = torch.tensor([240.0, 253.15, 263.155, 273.16, 290.0])
    omega = liquid_fraction_condensate(temps_test)
    assert float(omega[0].item()) == 0.0
    assert float(omega[1].item()) == 0.0
    assert abs(float(omega[2].item()) - 0.5) < 1.0e-4
    assert float(omega[3].item()) == 1.0
    assert float(omega[4].item()) == 1.0

    # Test supersaturated warm, mixed-phase, and ice parcels
    t_li = torch.tensor([290.0, 263.0, 240.0], dtype=torch.float32)
    p_pa = torch.tensor([90000.0, 55000.0, 30000.0], dtype=torch.float32)
    qn = torch.tensor([0.022, 0.006, 0.0015], dtype=torch.float32)

    t_adj, qv_adj, qc_adj, qi_adj = saturation_adjustment(t_li, p_pa, qn, n_iter=5)

    # 1. Total non-precipitating water conserved exactly
    assert torch.allclose(qv_adj + qc_adj + qi_adj, qn, atol=1.0e-7)
    # 2. Liquid-ice enthalpy conserved: c_pd * T_adj - L_v0 * q_c - L_s0 * q_i == c_pd * T_li
    t_li_recovered = t_adj - (C.L_V0 * qc_adj + C.L_S0 * qi_adj) / C.CP_D
    assert torch.allclose(t_li_recovered, t_li, atol=1.0e-3)
    # 3. Saturated parcels satisfy q_v == q_sat(T_adj, p)
    qsat_adj, _ = qsat_mixed(t_adj, p_pa)
    assert torch.allclose(qv_adj, qsat_adj, atol=1.0e-5)


def test_toon_two_stream_exact_limiting_cases():
    """Verify Toon et al. (1989) Table 2 exact analytical limits for SW and LW."""
    N = 16
    # 1. SW conservative scattering limit (omega_0 -> 1): zero atmospheric absorption!
    tau = torch.full((N, 2), 0.8)
    om = torch.full((N, 2), 0.999999)
    g = torch.full((N, 2), 0.80)
    mu0 = torch.tensor([0.7, 0.4])
    f_toa = torch.tensor([500.0, 500.0])
    alb = torch.tensor([0.2, 0.8])
    f_up, f_dn = toon_two_stream_sw(tau, om, g, mu0, f_toa, alb)
    net_toa = f_dn[0] - f_up[0]
    net_sfc = f_dn[-1] - f_up[-1]
    assert torch.allclose(net_toa, net_sfc, atol=5.0e-2)

    # 2. LW isothermal optically thick blackbody limit: F_up(TOA) -> f_band * sigma * T^4
    t_iso = 290.0
    fb_val = C.STEFAN_BOLTZMANN * (t_iso ** 4)
    tau_thick = torch.full((N, 1), 5.0)
    fb_int = torch.full((N + 1, 1), fb_val)
    fb_sfc = torch.full((1,), fb_val)
    lw_up, lw_dn = toon_two_stream_lw(tau_thick, fb_int, fb_sfc, emiss_sfc=1.0)
    assert abs(float(lw_up[0, 0].item()) - fb_val) < 0.05


def test_four_band_radiation_benchmarks_and_energy_conservation():
    """Verify energy budget closure, 2xCO2 forcing, OLR slope, and cloud radiative effects."""
    dev = get_device()
    vgrid = build_vertical_grid(nz=32, z_top=24000.0, dz_min=60.0, device=dev)
    z_w, z_c, dz_c = compute_stf_heights(vgrid, torch.zeros((1,), device=dev))
    snd = rcemip_analytic_sounding(z_c, z_w, sst=300.0)

    mu0 = torch.full((1,), math.cos(math.radians(42.05)), device=dev)
    s_toa = torch.full((1,), 551.58, device=dev)
    alb = torch.full((1,), 0.07, device=dev)
    zeros = torch.zeros_like(z_c)
    tsfc = torch.full((1,), 300.0, device=dev)

    rad_1x = compute_four_band_radiation(
        snd["pres_c"], snd["pres_w"], snd["temp_c"], snd["temp_w"], dz_c, snd["rho_d_c"],
        snd["qv_c"], zeros, zeros, tsfc, alb, alb, mu0, s_toa, co2_ppmv=284.3, emiss_sfc=1.0,
    )
    rad_2x = compute_four_band_radiation(
        snd["pres_c"], snd["pres_w"], snd["temp_c"], snd["temp_w"], dz_c, snd["rho_d_c"],
        snd["qv_c"], zeros, zeros, tsfc, alb, alb, mu0, s_toa, co2_ppmv=568.6, emiss_sfc=1.0,
    )

    # 1. Column radiative energy conservation: vertically integrated Q_rad == (TOA_net - SFC_net)
    cp_layer = (C.CP_D + C.CP_V * snd["qv_c"]) * (snd["rho_d_c"] * dz_c)
    col_heating_wm2 = (rad_1x.q_rad_net * cp_layer).sum(dim=0)
    toa_net_in = rad_1x.toa_asr - rad_1x.toa_olr
    sfc_net_in = rad_1x.sfc_sw_net + rad_1x.sfc_lw_net
    assert torch.allclose(col_heating_wm2, toa_net_in - sfc_net_in, atol=5.0e-2)

    # 2. Instantaneous 2xCO2 radiative forcing within Sherwood et al. (2020) physical range
    forcing_2x = float((rad_1x.toa_olr - rad_2x.toa_olr).item())
    assert 3.2 <= forcing_2x <= 4.5

    # 3. Positive surface PPFD in realistic tropical mean range
    ppfd = float(rad_1x.sfc_ppfd.item())
    assert 500.0 <= ppfd <= 900.0

    # 4. Cloud radiative effects: low liquid stratocumulus cools via SW albedo;
    #    high tropical ice anvil reduces OLR strongly (greenhouse warming)
    qc_low = torch.zeros_like(z_c)
    qc_low[4] = 0.0008  # low marine cloud around ~500 m
    rad_low_cld = compute_four_band_radiation(
        snd["pres_c"], snd["pres_w"], snd["temp_c"], snd["temp_w"], dz_c, snd["rho_d_c"],
        snd["qv_c"], qc_low, zeros, tsfc, alb, alb, mu0, s_toa, co2_ppmv=348.0, emiss_sfc=1.0,
    )
    assert float(rad_low_cld.toa_asr.item()) < float(rad_1x.toa_asr.item()) - 30.0

    qi_high = torch.zeros_like(z_c)
    qi_high[25] = 0.00015  # high ice anvil around ~11.7 km
    rad_high_cld = compute_four_band_radiation(
        snd["pres_c"], snd["pres_w"], snd["temp_c"], snd["temp_w"], dz_c, snd["rho_d_c"],
        snd["qv_c"], zeros, qi_high, tsfc, alb, alb, mu0, s_toa, co2_ppmv=348.0, emiss_sfc=1.0,
    )
    assert float(rad_high_cld.toa_olr.item()) < float(rad_1x.toa_olr.item()) - 25.0


def test_zarr_v3_io_roundtrip_and_append():
    """Verify native Zarr v3 creation, atomic time-chunk append, and reload."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "history.zarr"
        ds0 = xr.Dataset(
            {"olr": (("time", "nCells"), np.full((2, 16), 285.0, dtype=np.float32))},
            coords={"time": [0.0, 900.0], "latCell": ("nCells", np.zeros(16, dtype=np.float32))},
        )
        save_zarr(ds0, store)
        ds1 = xr.Dataset(
            {"olr": (("time", "nCells"), np.full((1, 16), 286.5, dtype=np.float32))},
            coords={"time": [1800.0]},
        )
        append_zarr(ds1, store, append_dim="time")
        loaded = load_zarr(store)
        assert loaded.attrs["Conventions"] == "CF-1.11 UGRID-1.0"
        assert loaded.sizes["time"] == 3
        assert np.allclose(loaded["olr"][:, 0].values, [285.0, 285.0, 286.5])


if __name__ == "__main__":
    test_tridiagonal_solver_cpu_and_mps()
    test_vertical_grid_and_canopy_datum_ordering()
    test_moist_thermodynamics_and_saturation_adjustment()
    test_toon_two_stream_exact_limiting_cases()
    test_four_band_radiation_benchmarks_and_energy_conservation()
    test_zarr_v3_io_roundtrip_and_append()
    print("ALL PHASE 1 FOUNDATION & RADIATION TESTS PASSED!")
