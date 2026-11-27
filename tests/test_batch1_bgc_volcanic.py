"""Verification tests for Batch 1 ESM closed loops:
1. Emissions-driven interactive atmospheric CO2 (global mass conservation + 3D SCVT FCT).
2. Prognostic 3D Ocean dissolved O2, Garcia-Gordon solubility, OMZ hypoxia, & denitrification -> N2O.
3. Interactive stratospheric volcanic SO2 -> H2SO4 aerosols & 11-yr Schwabe solar cycle.
4. Coupled ALL-ON Batch 1 AOBA integration on CPU and MPS.
"""

from __future__ import annotations

import math
import torch

from aoba import constants as C
from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.ocean_bgc import compute_o2_saturation_mmol_m3
from aoba.prognostic_co2 import PGC_PER_PPMV, init_prognostic_co2_state, step_prognostic_co2
from aoba.scvt_mesh import build_scvt_mesh
from aoba.volcanic_solar import (
    compute_solar_cycle_modulation,
    compute_volcanic_rad_optics_and_sad,
    init_volcanic_solar_state,
    step_volcanic_aerosols,
)


def test_prognostic_co2_mass_conservation() -> None:
    nc, nz = 42, 16
    area = torch.full((nc,), 4.0 * math.pi * (6.371e6 ** 2) / nc)
    rho = torch.ones((nz, nc))
    dz = torch.full((nz, nc), 500.0)
    lat = torch.linspace(-1.4, 1.4, nc)

    st = init_prognostic_co2_state(284.3, nz=nz, n_cells=nc, enable_3d=True)
    dt_yr = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    # Inject 10 Pg C/yr for 1 year with zero land/ocean uptake -> exact +10/2.124 ppmv
    step_prognostic_co2(st, area, dt_yr, anthro_pgc_yr=10.0, rho_d=rho, dz_c=dz, lat_cell=lat)
    expected = 284.3 + 10.0 / PGC_PER_PPMV
    assert abs(st.co2_ppmv_glob - expected) < 1.0e-6, f"CO2 drift: {st.co2_ppmv_glob} vs {expected}"
    # Verify 3D VMR global mass mean is locked to co2_ppmv_glob
    mean_3d = float(st.co2_vmr.mean().item() * 1.0e6)
    assert abs(mean_3d - expected) < 1.0e-3, f"3D CO2 mean {mean_3d} vs {expected}"
    print(f"[PASS] Prognostic CO2 exact budget: 284.3 -> {st.co2_ppmv_glob:.4f} ppmv (3D mean={mean_3d:.4f})")


def test_ocean_o2_solubility_and_omz() -> None:
    t_cold = torch.tensor([273.15])
    t_warm = torch.tensor([298.15])
    sal = torch.tensor([35.0])
    o2_c = float(compute_o2_saturation_mmol_m3(t_cold, sal).item())
    o2_w = float(compute_o2_saturation_mmol_m3(t_warm, sal).item())
    assert 340.0 < o2_c < 370.0 and 200.0 < o2_w < 225.0 and o2_c > o2_w
    print(f"[PASS] Garcia-Gordon O2 solubility: 0C={o2_c:.1f}, 25C={o2_w:.1f} mmol O2/m^3")


def test_volcanic_bateman_and_solar_cycle() -> None:
    mesh = build_scvt_mesh(1, device=torch.device("cpu"))
    vs = init_volcanic_solar_state(mesh.n_cells, torch.device("cpu"))
    # Pinatubo eruption at year 0.0: 17 Tg SO2 at 15 deg N
    dt = 86400.0
    for day in range(120):
        step_volcanic_aerosols(vs, mesh, float(day), dt, eruption_year=0.0, eruption_so2_tg=17.0, eruption_lat_deg=15.0)
    assert vs.eruption_fired
    assert vs.burden_so2_tg < 1.0, f"SO2 should mostly oxidize by day 120, got {vs.burden_so2_tg}"
    assert 15.0 < vs.burden_h2so4_tg < 26.0, f"Unexpected H2SO4 burden {vs.burden_h2so4_tg} Tg"
    assert 0.08 < vs.aod_volc_mean < 0.22, f"Unexpected Pinatubo global AOD {vs.aod_volc_mean}"

    pres = torch.logspace(5.0, 2.5, 44).view(44, 1).expand(44, mesh.n_cells).contiguous()
    dz = torch.full((44, mesh.n_cells), 500.0)
    (tsw, ssw, asw, tlw), sad = compute_volcanic_rad_optics_and_sad(vs, pres, dz)
    assert float(tsw.sum(dim=0).mean()) > 0.08 and float(tlw.sum(dim=0).mean()) > 0.003
    assert float(sad.max()) > 5.0

    tsi_max, uv_max = compute_solar_cycle_modulation(2.75 * C.DAYS_PER_YEAR, amp_wm2=1.0, period_yr=11.0)
    tsi_min, uv_min = compute_solar_cycle_modulation(8.25 * C.DAYS_PER_YEAR, amp_wm2=1.0, period_yr=11.0)
    assert tsi_max > 0.45 and tsi_min < -0.45 and uv_max > 1.01 and uv_min < 0.99
    print(f"[PASS] Pinatubo day 120: SO2={vs.burden_so2_tg:.2f} Tg, H2SO4={vs.burden_h2so4_tg:.2f} Tg, AOD={vs.aod_volc_mean:.3f}, maxSAD={float(sad.max()):.1f} um2/cm3 | Solar TSI=[{tsi_min:+.2f},{tsi_max:+.2f}] W/m2")


def test_coupled_batch1_cpu_mps() -> None:
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        cfg = AOBAConfig(
            glevel=1,
            ocean_glevel=2,
            crm_ny=2,
            crm_nx=2,
            co2_mode="prognostic",
            enable_co2_3d_transport=True,
            co2_anthro_pgc_yr=10.0,
            enable_ocean_bgc=True,
            enable_ocean_o2_omz=True,
            enable_land_carbon=True,
            enable_prognostic_ch4_n2o=True,
            enable_volcanic_aerosols=True,
            volcanic_eruption_year=0.0,
            volcanic_so2_tg=17.0,
            enable_solar_cycle=True,
            o3_mode="prognostic",
        )
        m = init_aoba_model(cfg, device=dev)
        # Seed a suboxic layer to exercise OMZ denitrification path directly
        m.ogcm.bgc.o2 = None  # let step lazily initialize first
        d1 = step_aoba_model(m)
        assert m.ogcm.bgc.o2 is not None
        m.ogcm.bgc.o2[3] = 5.0 * m.ogcm.wet_c[3].to(m.ogcm.bgc.o2.dtype)
        m.ogcm.bgc.det_c[3] = 2.0 * m.ogcm.wet_c[3].to(m.ogcm.bgc.det_c.dtype)
        d2 = step_aoba_model(m)
        assert torch.isfinite(m.prognostic_co2.co2_vmr).all()
        assert torch.isfinite(m.ogcm.bgc.o2).all()
        assert float(m.ogcm.bgc.denit_moln_m2s.max()) > 0.0, "OMZ denitrification must activate when O2 < 20!"
        assert m.volcanic_solar.eruption_fired and m.volcanic_solar.burden_so2_tg > 10.0
        print(
            f"[PASS] Coupled Batch 1 ({dev}): Ts={d2.global_mean_t_sfc_k:.2f} K, "
            f"CO2={m.current_co2_ppmv():.4f} ppmv, O2_mean={float(m.ogcm.bgc.o2[m.ogcm.wet_c].mean()):.1f} mmol/m3, "
            f"Volc_SO2={m.volcanic_solar.burden_so2_tg:.2f} Tg, O3={float(m.ozone.tco_du.mean()):.1f} DU"
        )


def test_ocean_bgc_stride_zstar_conservation() -> None:
    """Verify exact z* telescoping mass conservation and parity for `bgc_stride=2` vs `1`."""
    from aoba.ocean_bgc import init_ocean_bgc_state
    from aoba.ocean_ogcm import init_ogcm_state, step_ogcm_voronoi
    from aoba.surface import SURF_OCEAN, init_surface_state

    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev_s in devs:
        dev = torch.device(dev_s)
        mesh = build_scvt_mesh(1, device=dev)
        nc = mesh.n_cells
        dt = 2160.0

        def _run(stride_k: int) -> tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
            st_ocn = torch.full((nc,), SURF_OCEAN, device=dev, dtype=torch.int64)
            sfc = init_surface_state(st_ocn, h_ml_ocean=50.0)
            ogcm = init_ogcm_state(sfc, mesh, nz_ocn=12)
            ogcm.bgc = init_ocean_bgc_state(ogcm.wet_c, ogcm.h_c, ogcm.z_c, mesh.area_cell)
            # Non-uniform forcing to exercise 3D horizontal/vertical z* stretching & advection
            lat = mesh.lat_cell
            lon = mesh.lon_cell
            tau_x = 0.08 * torch.cos(2.0 * lat)
            tau_y = 0.03 * torch.sin(lon)
            f_ice = torch.zeros(nc, device=dev)
            evap = torch.zeros(nc, device=dev)
            prec = torch.zeros(nc, device=dev)
            runoff = torch.zeros(nc, device=dev)
            sw_net = torch.clamp(220.0 * torch.cos(lat), min=0.0)
            par_in = 0.43 * sw_net
            u10 = torch.full((nc,), 7.0, device=dev)

            def _inv() -> tuple[float, float]:
                h_c64 = ogcm.h_c.detach().cpu().to(torch.float64)
                eta64 = ogcm.eta.detach().cpu().to(torch.float64)
                area64 = mesh.area_cell.detach().cpu().to(torch.float64)
                h_col = torch.clamp(h_c64.sum(dim=0), min=1.0)
                sc = torch.clamp(1.0 + eta64 / h_col, min=0.5, max=1.5)
                vol = h_c64 * sc.unsqueeze(0) * area64.unsqueeze(0)
                b = ogcm.bgc
                n_tot = (
                    (b.no3.detach().cpu().to(torch.float64) + b.no3_comp.detach().cpu().to(torch.float64))
                    + b.phy.detach().cpu().to(torch.float64)
                    + b.zoo.detach().cpu().to(torch.float64)
                    + b.det_n.detach().cpu().to(torch.float64)
                )
                alk_tot = b.alk.detach().cpu().to(torch.float64) + b.alk_comp.detach().cpu().to(torch.float64)
                return float((vol * n_tot).sum().item()), float((vol * alk_tot).sum().item())

            n0, alk0 = _inv()
            for step_i in range(6):
                # Modulate wind stress each step so consecutive steps have distinct eta/h*
                fac = 1.0 + 0.15 * math.sin(0.9 * step_i)
                step_ogcm_voronoi(
                    ogcm=ogcm, sfc=sfc, mesh=mesh,
                    tau_x_sfc=tau_x * fac, tau_y_sfc=tau_y * fac, f_seaice=f_ice,
                    evap_cell=evap, precip_cell=prec, runoff_total=runoff, runoff_glacier=runoff,
                    dt=dt, enable_seaice_drift=False, sw_net_ocn=sw_net, sw_par_ocn=par_in,
                    u10_ocn=u10, co2_ppmv=284.3, enable_fct_zstar=True, enable_o2_omz=False,
                    bgc_stride=stride_k,
                )
            n1, alk1 = _inv()
            if stride_k > 1:
                assert getattr(ogcm, "_bgc_stride_phase", -1) == 0
            assert torch.isfinite(ogcm.bgc.no3).all() and torch.isfinite(ogcm.bgc.dic).all()
            return n0, n1, alk0, alk1, ogcm.bgc.no3.clone(), ogcm.bgc.dic.clone()

        n0_1, n1_1, a0_1, a1_1, no3_1, dic_1 = _run(1)
        n0_2, n1_2, a0_2, a1_2, no3_2, dic_2 = _run(2)
        rel_n_2 = abs(n1_2 - n0_2) / max(abs(n0_2), 1.0)
        rel_a_2 = abs(a1_2 - a0_2) / max(abs(a0_2), 1.0)
        rel_dic_diff = float((dic_2 - dic_1).abs().mean().item() / max(float(dic_1.abs().mean().item()), 1.0))
        rel_no3_diff = float((no3_2 - no3_1).abs().mean().item() / max(float(no3_1.abs().mean().item()), 1.0))
        assert rel_n_2 < 1.0e-4, f"[{dev_s}] BGC stride=2 N conservation drift {rel_n_2:.2e}"
        assert rel_a_2 < 1.0e-4, f"[{dev_s}] BGC stride=2 ALK conservation drift {rel_a_2:.2e}"
        assert rel_dic_diff < 1.0e-3, f"[{dev_s}] BGC stride=2 vs 1 DIC diff {rel_dic_diff:.2e}"
        assert rel_no3_diff < 5.0e-3, f"[{dev_s}] BGC stride=2 vs 1 NO3 diff {rel_no3_diff:.2e}"
        print(
            f"[PASS] Ocean BGC stride=2 z* conservation ({dev_s}): "
            f"dN/N={rel_n_2:.2e}, dALK/ALK={rel_a_2:.2e}, DIC_diff={rel_dic_diff:.2e}, NO3_diff={rel_no3_diff:.2e}"
        )


if __name__ == "__main__":
    test_prognostic_co2_mass_conservation()
    test_ocean_o2_solubility_and_omz()
    test_volcanic_bateman_and_solar_cycle()
    test_coupled_batch1_cpu_mps()
    test_ocean_bgc_stride_zstar_conservation()
    print("ALL BATCH 1 CLOSED-LOOP ESM TESTS PASSED!")
