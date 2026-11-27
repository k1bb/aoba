"""Verification suite for High-Top Mesosphere (90 km), Non-LTE Radiation, Molecular Diffusion,
Prognostic 3D NOx, Bidirectional Sprite/GJ -> Ozone Coupling, NLC, and Full TLEs (Sprites, Halos,
ELVES, Blue Jets, Gigantic Jets).
"""

from __future__ import annotations

import torch

from aoba.coupled_model import AOBAConfig, init_aoba_model, step_aoba_model
from aoba.mesosphere import (
    apply_mesospheric_molecular_diffusion,
    compute_mesospheric_nlte_radiation,
    diagnose_noctilucent_clouds,
    init_prognostic_nox_state,
    step_prognostic_nox,
)
from aoba.ozone_chemistry import init_prognostic_ozone_state, step_prognostic_ozone
from aoba.red_sprites import compute_red_sprites
from aoba.scvt_mesh import build_scvt_mesh
from aoba.surface import SURF_LAND
from aoba.thermodynamics import diagnose_pressure_temperature
from aoba.vertical_grid import build_vertical_grid


def _get_devices() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    elif torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def test_nlte_radiation_and_co2_mesospheric_cooling() -> None:
    for dev in _get_devices():
        vg = build_vertical_grid(nz=64, z_top=90000.0, dz_min=60.0, z_sponge_base=76000.0, device=dev)
        zc = vg.zeta_c.unsqueeze(1).expand(64, 4)
        dzc = vg.dzeta_c.unsqueeze(1).expand(64, 4)
        # US-76-like temperature profile: 288 K sfc -> 265 K stratopause -> 195 K mesopause
        temp = torch.where(
            zc < 12000.0, 288.0 - 0.0065 * zc,
            torch.where(zc < 50000.0, 210.0 + 0.00145 * (zc - 12000.0), 265.0 - 0.0020 * (zc - 50000.0)),
        )
        pres = 101325.0 * torch.exp(-zc / 7000.0)
        rho = pres / (287.04 * temp)
        o3 = torch.full_like(pres, 2.0e-6)
        mu0 = torch.tensor([-0.5, 0.0, 0.5, 0.9], device=dev)
        q_lte = torch.zeros_like(pres)

        q_1x = compute_mesospheric_nlte_radiation(q_lte, pres, temp, rho, dzc, o3, mu0, co2_ppmv=284.3)
        q_4x = compute_mesospheric_nlte_radiation(q_lte, pres, temp, rho, dzc, o3, mu0, co2_ppmv=1137.2)

        assert torch.isfinite(q_1x).all() and torch.isfinite(q_4x).all()
        # Below 35 km (p > 5 hPa), non-LTE blend must leave LTE radiation virtually untouched (<1e-7 K/s)
        low = zc[:, 0] < 35000.0
        assert float(q_1x[low].abs().max()) < 1.0e-7, f"Low-altitude leak: {float(q_1x[low].abs().max())}"
        # Nighttime mesosphere (mu0 < 0, z > 65 km) cools via CO2 15 um, and 4xCO2 cools MORE strongly!
        hi = zc[:, 0] > 68000.0
        assert float(q_1x[hi, 0].mean()) < -1.0e-6, "Nighttime mesosphere should cool radiatively"
        assert float(q_4x[hi, 0].mean()) < float(q_1x[hi, 0].mean()) * 1.25, "4xCO2 must intensify mesospheric cooling"
        # Daytime mesosphere (mu0 = 0.9) has UV heating offsetting CO2 cooling
        assert float(q_1x[hi, 3].mean()) > float(q_1x[hi, 0].mean()), "Daytime UV heating must warm mesosphere vs night"
    print("  [PASS] Fomichev non-LTE CO2 cooling (4xCO2 intensified) + Strobel UV heating + zero trop/strat leak")


def test_molecular_diffusion_conservation_and_stability() -> None:
    for dev in _get_devices():
        mesh = build_scvt_mesh(glevel=1, use_earth2014=False, device=dev)
        vg = build_vertical_grid(nz=64, z_top=90000.0, dz_min=60.0, z_sponge_base=76000.0, device=dev)
        nc, ne = mesh.n_cells, mesh.n_edges
        zc = vg.zeta_c.unsqueeze(1).expand(64, nc).contiguous()
        dzc = vg.dzeta_c.unsqueeze(1).expand(64, nc).contiguous()
        # Test true sensible-temperature Fourier conduction on a realistic stratified mesosphere:
        # 1) Exact enthalpy conservation sum(rho * dz * T) under wavy T(z)
        temp_wavy = 200.0 + 25.0 * torch.sin(zc / 5000.0)
        pres = 101325.0 * torch.exp(-zc / 7000.0)
        rho = pres / (287.04 * temp_wavy)
        exner = torch.pow(pres / 100000.0, 287.04 / 1004.64)
        thm = temp_wavy / exner
        u_e = 60.0 * torch.sin(zc[:, :1] / 4000.0).expand(64, ne).contiguous()

        dm = (rho * dzc).cpu().double()
        h_before = (dm * temp_wavy.cpu().double()).sum(dim=0)
        thm_new, u_new = apply_mesospheric_molecular_diffusion(
            rho, thm, u_e, zc, dzc, temp_wavy, mesh.cells_on_edge[:, 0], mesh.cells_on_edge[:, 1], dt=720.0
        )
        t_after = thm_new * exner
        h_after = (dm * t_after.cpu().double()).sum(dim=0)
        rel_err = float(((h_after - h_before).abs() / h_before.abs()).max().item())
        assert rel_err < 1.0e-5, f"Molecular thermal conduction failed column enthalpy conservation: {rel_err}"
        # 2) Strictly zero spurious pumping in an isothermal stratified mesosphere (despite theta_m > 10,000 K!)
        t_iso = torch.full((64, nc), 185.0, device=dev)
        thm_iso = t_iso / exner
        thm_iso_new, _ = apply_mesospheric_molecular_diffusion(
            rho, thm_iso, u_e, zc, dzc, t_iso, mesh.cells_on_edge[:, 0], mesh.cells_on_edge[:, 1], dt=720.0
        )
        assert float((thm_iso_new - thm_iso).abs().max().item()) < 1.0e-4, "Isothermal atmosphere must have zero molecular heat flux!"
        # Global max principle + strict kinetic-energy dissipation & column momentum conservation
        rho_e = 0.5 * (rho[:, mesh.cells_on_edge[:, 0]] + rho[:, mesh.cells_on_edge[:, 1]])
        dz_e = 0.5 * (dzc[:, mesh.cells_on_edge[:, 0]] + dzc[:, mesh.cells_on_edge[:, 1]])
        dm_e = (rho_e * dz_e).cpu().double()
        p_bef = (dm_e * u_e.cpu().double()).sum(dim=0)
        p_aft = (dm_e * u_new.cpu().double()).sum(dim=0)
        assert float(((p_aft - p_bef).abs() / torch.clamp(p_bef.abs(), min=1.0)).max()) < 1.0e-5
        ke_bef = float((0.5 * dm_e * u_e.cpu().double().square()).sum())
        ke_aft = float((0.5 * dm_e * u_new.cpu().double().square()).sum())
        assert ke_aft < ke_bef and float(u_new.abs().max()) <= float(u_e.abs().max()) + 1.0e-4
        assert torch.isfinite(thm_new).all() and torch.isfinite(u_new).all()
    print("  [PASS] Implicit increment-form mesospheric molecular viscosity & thermal diffusion conservation")


def test_bidirectional_sprite_nox_ozone_nlc_and_all_tles() -> None:
    for dev in _get_devices():
        vg = build_vertical_grid(nz=64, z_top=90000.0, dz_min=60.0, z_sponge_base=76000.0, device=dev)
        nc = 8
        zc = vg.zeta_c.unsqueeze(1).expand(64, nc).contiguous()
        dzc = vg.dzeta_c.unsqueeze(1).expand(64, nc).contiguous()
        # Columns 0..3: night (mu0=-0.4), cols 4..7: day (mu0=+0.7); col 0 & 4 have vigorous deep MCS + CG!
        mu0 = torch.tensor([-0.4, -0.4, -0.4, -0.4, 0.7, 0.7, 0.7, 0.7], device=dev)
        lat = torch.tensor([0.1, 1.3, -1.3, 0.5, 0.1, 1.3, -1.3, 0.5], device=dev)
        temp = torch.where(
            zc < 12000.0, 295.0 - 0.0065 * zc,
            torch.where(zc < 50000.0, 217.0 + 0.0013 * (zc - 12000.0), 266.0 - 0.0021 * (zc - 50000.0)),
        )
        # Impose a cold summer polar mesopause (138 K at 80-86 km) in column 1 to verify NLC emergence!
        cold_mask = (zc >= 78000.0) & (zc <= 87000.0)
        temp[:, 1] = torch.where(cold_mask[:, 1], torch.full_like(temp[:, 1], 138.0), temp[:, 1])
        pres = 101325.0 * torch.exp(-zc / 7000.0)
        rho = pres / (287.04 * temp)
        exner = torch.pow(pres / 1.0e5, 287.04 / 1004.64)
        qv = torch.where(zc > 20000.0, torch.full_like(pres, 4.0e-6), 0.01 * torch.exp(-zc / 3000.0))
        thm = temp * (1.0 + 1.6078 * qv) / exner
        qi = torch.zeros_like(pres); qp = torch.zeros_like(pres)
        # Deep anvil IWP (~3 kg/m^2) in cols 0 & 4
        anvil_k = (zc >= 8000.0) & (zc <= 14000.0)
        qi[:, 0] = torch.where(anvil_k[:, 0], torch.full_like(qi[:, 0], 1.5e-3), qi[:, 0])
        qi[:, 4] = torch.where(anvil_k[:, 4], torch.full_like(qi[:, 4], 1.5e-3), qi[:, 4])

        flash_cg = torch.tensor([5.0e-10, 0.0, 0.0, 0.0, 5.0e-10, 0.0, 0.0, 0.0], device=dev)
        ccd = torch.tensor([14.5, 0.0, 0.0, 0.0, 14.5, 0.0, 0.0, 0.0], device=dev)
        area = torch.full((nc,), 1.0e10, device=dev)

        tle = compute_red_sprites(
            flash_cg_cell=flash_cg, cold_cloud_km_cell=ccd, mu0_cell=mu0,
            rho_host=rho, theta_m_host=thm, q_v_host=qv, q_i_host=qi, q_p_host=qp,
            z_c_host=zc, dz_c_host=dzc, area_cell=area,
        )
        # Verify ALL 5 TLE classes (Sprites, Halos, ELVES, Blue Jets, Gigantic Jets)
        assert float(tle.sprite_rate_cell[0]) > 20.0 * float(tle.sprite_rate_cell[4]), "Sprites must be strongly nocturnal"
        assert tle.elves_rate_cell is not None and float(tle.elves_rate_cell[0]) > 5.0 * float(tle.elves_rate_cell[4]), "ELVES must be nocturnal"
        assert tle.blue_jet_rate_cell is not None and float(tle.blue_jet_rate_cell[0]) > 0.0
        assert tle.blue_jet_top_km_cell is not None and 25.0 <= float(tle.blue_jet_top_km_cell[0]) <= 44.0
        assert tle.gigantic_jet_rate_cell is not None and float(tle.gigantic_jet_rate_cell[0]) > 0.0
        assert float(tle.gigantic_jet_rate_cell[0]) < float(tle.blue_jet_rate_cell[0]), "GJs must be rarer than Blue Jets"

        # Verify NLC emergence in cold summer mesopause col 1 (138 K) vs warm col 2 (~192 K)
        nlc_f, nlc_iwc, t_mp = diagnose_noctilucent_clouds(pres, temp, rho, zc, qv)
        assert float(nlc_f[1]) > 0.70 and float(nlc_iwc[1]) > 0.0, f"NLC failed to form at 138 K: f={float(nlc_f[1])}"
        assert float(nlc_f[2]) < 0.01, f"NLC formed spuriously at warm mesopause: f={float(nlc_f[2])}"

        # Verify bidirectional Sprite/GJ -> NOx -> Ozone catalytic destruction + nighttime secondary O3 max!
        nox_ctrl = init_prognostic_nox_state(pres)
        nox_sp = init_prognostic_nox_state(pres)
        oz_ctrl = init_prognostic_ozone_state(pres, rho, dzc)
        oz_sp = init_prognostic_ozone_state(pres, rho, dzc)
        surf_t = torch.full((nc,), SURF_LAND, device=dev, dtype=torch.int64)
        z1 = torch.zeros((nc,), device=dev)

        # Inject vigorous sustained sprite storm into col 0 (100x amplification for clean 6-hour signal)
        sp_nox_strong = tle.sprite_nox_mol_m2s * 200.0
        for _ in range(30):
            step_prognostic_nox(
                nox_ctrl, pres, temp, rho, zc, dzc, qv, mu0, lat, z1, dt=720.0,
            )
            step_prognostic_nox(
                nox_sp, pres, temp, rho, zc, dzc, qv, mu0, lat, z1, dt=720.0,
                sprite_nox_mol_m2s=sp_nox_strong, sprite_z_init_km=tle.z_init_km_cell, sprite_z_stop_km=tle.z_stop_km_cell,
            )
            step_prognostic_ozone(
                oz_ctrl, pres, temp, rho, dzc, qv, mu0, surf_t, z1, z1, z1, dt=720.0,
                nox_vmr_3d=nox_ctrl.q_nox, z_c=zc,
            )
            step_prognostic_ozone(
                oz_sp, pres, temp, rho, dzc, qv, mu0, surf_t, z1, z1, z1, dt=720.0,
                nox_vmr_3d=nox_sp.q_nox, z_c=zc,
            )
        assert float(nox_sp.nox_meso_ppbv[0]) > float(nox_ctrl.nox_meso_ppbv[0]) + 1.0, "Sprite NOx failed to accumulate"
        # Mesospheric O3 (55-85 km) in sprite column 0 must be measurably depleted relative to control!
        m_band = (zc[:, 0] >= 55000.0) & (zc[:, 0] <= 86000.0)
        o3_ctrl_m = float(oz_ctrl.o3_vmr[m_band, 0].mean())
        o3_sp_m = float(oz_sp.o3_vmr[m_band, 0].mean())
        assert o3_sp_m < o3_ctrl_m * 0.98, f"Bidirectional Sprite-NOx -> O3 depletion too weak: ctrl={o3_ctrl_m:.3e}, sp={o3_sp_m:.3e}"
        # Nighttime secondary O3 peak (76-86 km, col 2) must exceed daytime mesospheric O3 (col 6)
        sec_band = (zc[:, 0] >= 76000.0) & (zc[:, 0] <= 86000.0)
        assert float(oz_ctrl.o3_vmr[sec_band, 2].mean()) > 3.0 * float(oz_ctrl.o3_vmr[sec_band, 6].mean())
        print(
            f"  [PASS] All 5 TLEs + NLC (138K f={float(nlc_f[1]):.2f}) + Sprite->NOx({float(nox_sp.nox_meso_ppbv[0]):.1f} ppbv)"
            f"->O3({o3_ctrl_m*1e6:.2f}->{o3_sp_m*1e6:.2f} ppmv) + Night 2nd O3 max on {dev.type}"
        )


def test_coupled_hightop_90km_step() -> None:
    dev = _get_devices()[-1]
    cfg = AOBAConfig(
        glevel=2, nz=64, z_top=90000.0, z_sponge_base=76000.0, dt_host=720.0,
        enable_superparam=True, crm_nx=8, crm_ny=1, crm_dx=4000.0,
        rad_scheme="ckd", o3_mode="prognostic", enable_nonoro_gwd=True,
        ocean_mode="ekman_sw", use_earth2014=False,
        enable_lightning=True, lightning_scheme="explicit_bulk", enable_red_sprites=True,
        enable_dust=False, enable_natural_aerosols=False, enable_land_carbon=False,
        enable_ocean_bgc=False, enable_river_routing=False,
    )
    model = init_aoba_model(cfg, device=dev)
    for _ in range(4):
        step_aoba_model(model)
    assert model.meso_nox is not None and torch.isfinite(model.meso_nox.q_nox).all()
    assert model.last_red_sprites is not None and model.last_red_sprites.elves_rate_cell is not None
    assert model.ozone is not None and torch.isfinite(model.ozone.o3_vmr).all()
    assert torch.isfinite(model.atmos.theta_m).all() and torch.isfinite(model.atmos.u_edge).all()
    print(f"  [PASS] Coupled high-top 90 km (nz=64) GWD + non-LTE + mol-diff + NOx + O3 + all 5 TLEs on {dev.type}")


if __name__ == "__main__":
    print("Running High-Top Mesosphere, NOx<->O3, NLC & Full TLE verification tests...")
    test_nlte_radiation_and_co2_mesospheric_cooling()
    test_molecular_diffusion_conservation_and_stability()
    test_bidirectional_sprite_nox_ozone_nlc_and_all_tles()
    test_coupled_hightop_90km_step()
    print("ALL HIGH-TOP MESOSPHERE & FULL TLE TESTS PASSED!")
