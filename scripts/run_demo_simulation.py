#!/usr/bin/env python3
"""Run a multi-season / full-annual global AOBA test simulation + 1:1 3D SP-CRM showcase.

Protocol (zero external PFT, LAI, soil/albedo, or ocean Q-flux maps):
1. Phase A — 60-Day Coupled Spinup (Nov 1 -> Dec 31 of Year 0):
   Spins up the global SCVT nonhydrostatic general circulation (jets, Hadley/Ferrel cells,
   baroclinic storm tracks), cloud/water-vapor fields, active soil layers, snowpack, sea ice,
   Ekman surface ocean, and 15-day EMA Smith/Scott-Smith photosynthetic traits to Jan 1 00:00.
2. Phase B — Full 365-Day Seasonal Cycle (Jan 1 -> Dec 31 of Year 1):
   Integrates an uninterrupted 12-month seasonal cycle at dt_host = 1800 s, recording:
   - Daily global radiative & hydrological time series
   - Clean 12-month [12, N_cells] T_m [degC] & P_m [mm/month] for exact 30-class Köppen-Geiger
     classification (Beck et al., 2018, 2023)
   - Annual, DJF, and JJA 3D atmospheric general circulation & 2D surface/ocean fields
   - Daily cell-level meteorology (P, PET, PPFD, T, VPD, T_min, sw1_dir, sw1_dif, mu0_noon)
     to equilibrate the slow 1-year EEO structural pools (Cai et al., 2025; Zhou et al., 2025;
     Kempes et al., 2011; Gao et al., 2014) strictly from AOBA's own endogenous annual climate
   - Monthly CF-1.11 / UGRID-1.0 Zarr v3 snapshots in ~/aoba_outputs/
3. Phase C — 30-Day Coupled Post-Equilibration Verification (Jan 1 -> Jan 30 of Year 2):
   Continues full two-way coupled atmosphere-ocean-biosphere integration with the equilibrated
   biome structure (LAI, h_c, S_R, F_4, Vcmax25) feeding back onto surface roughness (Raupach,
   1994), 2-stream canopy albedo (Dai et al., 2004), and transpiration.
4. Phase D — Active 1:1 3D Nonhydrostatic SP-CRM Showcase (8x8x32 on Apple Silicon MPS GPU):
   Upgrades all 642 SCVT columns to explicit 3D cloud-resolving grids (1,314,816 grid points,
   dx = 4 km, dt = 180 s) on MPS GPU to resolve 3D convective updraft plumes & cloud towers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys
import time

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from aoba import constants as C
from aoba.biosphere import (
    BETA_C3,
    DHA_GAMMA,
    DHA_KC_C3,
    DHA_KO_C3,
    GAMMA_STAR25_C3,
    KC25_C3,
    KO25_C3,
    THETA_NRH,
    _arrhenius,
    compute_two_leaf_gas_exchange,
    phi_psii_temperature,
    smith_omega,
    update_daily_acclimation_smith,
    update_daily_lai_root_height,
    water_viscosity_rel,
)
from aoba.coupled_model import (
    AOBAConfig,
    broadcast_host_surface_to_subgrid,
    compute_orbital_insolation,
    diagnose_model_koppen_geiger,
    init_aoba_model,
    step_aoba_model,
)
from aoba.crm import CRMState
from aoba.io_zarr import append_zarr, export_aoba_snapshot_xarray
from aoba.runtime import get_output_dir
from aoba.scvt_mesh import reconstruct_cell_winds
from aoba.surface import SURF_LAND, SURF_OCEAN, compute_surface_albedos
from aoba.thermodynamics import diagnose_pressure_temperature
from aoba.turbulence import TKE_MIN


def _compute_daily_a0_mol(
    daily_ppfd_mol: torch.Tensor,
    daily_temp_k: torch.Tensor,
    daily_vpd_pa: torch.Tensor,
    pres_sfc: torch.Tensor,
    co2_ppmv: float,
    ws_beta: torch.Tensor,
) -> torch.Tensor:
    """Evaluate daily potential C3 canopy assimilation A_0 [mol C m^-2 d^-1] if fAPAR = 1."""
    ca_pa = (co2_ppmv * 1.0e-6) * pres_sfc
    oi_pa = C.O2_MIXING_RATIO * pres_sfc
    eta_s = water_viscosity_rel(daily_temp_k)
    gamma3 = _arrhenius(GAMMA_STAR25_C3, DHA_GAMMA, daily_temp_k) * (pres_sfc / 101325.0)
    km3 = _arrhenius(KC25_C3, DHA_KC_C3, daily_temp_k) * (
        1.0 + oi_pa / _arrhenius(KO25_C3, DHA_KO_C3, daily_temp_k)
    )
    d_s = torch.clamp(daily_vpd_pa, min=50.0)
    xi3 = torch.sqrt(BETA_C3 * (km3 + gamma3) / (1.6 * eta_s))
    chi3 = torch.clamp(gamma3 / ca_pa + (1.0 - gamma3 / ca_pa) * (xi3 / (xi3 + torch.sqrt(d_s))), 0.05, 0.95)
    ci3 = chi3 * ca_pa
    m_c3 = torch.clamp((ci3 - gamma3) / torch.clamp(ci3 + 2.0 * gamma3, min=0.1), min=0.0)
    _, omega_star3 = smith_omega(m_c3)
    lue_c3 = phi_psii_temperature(daily_temp_k) * m_c3 * omega_star3 / (8.0 * THETA_NRH)
    cold_gate = torch.sigmoid(0.5 * (daily_temp_k - C.T_FREEZE))
    return lue_c3 * torch.clamp(daily_ppfd_mol, min=0.0) * cold_gate * torch.clamp(ws_beta, 0.05, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spinup-days", type=int, default=60)
    parser.add_argument("--annual-days", type=int, default=365)
    parser.add_argument("--post-days", type=int, default=30)
    parser.add_argument("--sp-steps", type=int, default=40)
    args = parser.parse_args()

    torch.set_num_threads(2)
    out_dir = get_output_dir()
    zarr_path = out_dir / "aoba_g3_demo.zarr"
    diag_path = out_dir / "aoba_g3_demo_diagnostics.pt"
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    dt_host = 1800.0  # 48 steps/day (proven unconditionally stable over steep Earth2014 terrain)
    steps_per_day = int(round(C.SECONDS_PER_DAY / dt_host))
    spinup_days = args.spinup_days
    annual_days = args.annual_days
    post_days = args.post_days
    total_days = spinup_days + annual_days + post_days

    cfg = AOBAConfig(
        glevel=3,
        nz=32,
        dt_host=dt_host,
        n_host_dyn_substeps=2,
        rad_stride=3,
        enable_superparam=True,
        crm_nx=1,
        crm_ny=1,
        n_crm_dyn_substeps=0,
        crm_col_chunk=642,
        co2_ppmv=400.0,
    )

    print("=== Initializing Global AOBA (Glevel-3 SCVT, 642 cells, Nz=32, Earth2014, CO2=400 ppmv) ===", flush=True)
    t_wall0 = time.time()
    model = init_aoba_model(cfg, device="cpu", dtype=torch.float32)
    # Start 60 days before Jan 1 (Nov 2 of Year 0) so Day 60 lands cleanly on Jan 1 00:00
    yr_sec = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    model.sim_time_sec = (C.DAYS_PER_YEAR - float(spinup_days)) * C.SECONDS_PER_DAY

    nc = model.mesh.n_cells
    nz = cfg.nz

    ts_day, ts_tsfc, ts_asr, ts_olr, ts_precip, ts_lai, ts_gpp, ts_seaice = (
        [], [], [], [], [], [], [], []
    )

    def _make_acc():
        return {
            "count": 0,
            "u": torch.zeros((nz, nc)),
            "v": torch.zeros((nz, nc)),
            "w": torch.zeros((nz, nc)),
            "temp": torch.zeros((nz, nc)),
            "qv": torch.zeros((nz, nc)),
            "qcond": torch.zeros((nz, nc)),
            "pres": torch.zeros((nz, nc)),
            "t_sfc": torch.zeros(nc),
            "t_air": torch.zeros(nc),
            "precip": torch.zeros(nc),
            "evap": torch.zeros(nc),
            "olr": torch.zeros(nc),
            "asr": torch.zeros(nc),
            "f_seaice": torch.zeros(nc),
            "m_snow": torch.zeros(nc),
            "sst": torch.zeros(nc),
            "w_ekman": torch.zeros(nc),
            "alb_sw": torch.zeros(nc),
        }

    acc_ann = _make_acc()
    acc_djf = _make_acc()
    acc_jja = _make_acc()

    # Store daily meteorological drivers over the 365-day annual cycle to spin up slow multi-year
    # EEO structural pools strictly from AOBA's own endogenous 365-day simulated climate!
    ann_daily_drivers = {
        "p_mm": torch.zeros((annual_days, nc)),
        "pet_mm": torch.zeros((annual_days, nc)),
        "ppfd_mol": torch.zeros((annual_days, nc)),
        "temp_k": torch.zeros((annual_days, nc)),
        "vpd_pa": torch.zeros((annual_days, nc)),
        "t_min_k": torch.zeros((annual_days, nc)),
        "psfc": torch.zeros((annual_days, nc)),
        "ws_beta": torch.zeros((annual_days, nc)),
        "m_snow": torch.zeros((annual_days, nc)),
        "noon_vb": torch.zeros((annual_days, nc)),
        "noon_vd": torch.zeros((annual_days, nc)),
        "noon_mu0": torch.zeros((annual_days, nc)),
        "noon_t": torch.zeros((annual_days, nc)),
        "noon_vpd": torch.zeros((annual_days, nc)),
    }

    for day_idx in range(total_days):
        # Reset 12-month Köppen-Geiger accumulators right at Jan 1 00:00 after 60-day spinup!
        if day_idx == spinup_days:
            print(">>> Spinup complete (60 days): starting clean 365-day seasonal cycle (Jan 1 -> Dec 31)!", flush=True)
            model.clim.month_temp_sum_c.zero_()
            model.clim.month_precip_mm.zero_()
            model.clim.month_steps.zero_()
            model.clim.month_days_accum.zero_()

        d_tsfc = d_asr = d_olr = d_prec = d_lai = d_gpp = d_ice = 0.0

        day_p = torch.zeros(nc)
        day_pet = torch.zeros(nc)
        day_ppfd = torch.zeros(nc)
        day_t = torch.zeros(nc)
        day_vpd = torch.zeros(nc)
        day_tmin = torch.full((nc,), 400.0)
        noon_vb = torch.zeros(nc)
        noon_vd = torch.zeros(nc)
        noon_mu0 = torch.full((nc,), 0.2)
        noon_t = torch.zeros(nc)
        noon_vpd = torch.full((nc,), 500.0)

        in_annual_window = spinup_days <= day_idx < (spinup_days + annual_days)

        for s in range(steps_per_day):
            t_before = model.sim_time_sec
            diag = step_aoba_model(model)
            d_tsfc += diag.global_mean_t_sfc_k / steps_per_day
            d_asr += diag.global_mean_asr_wm2 / steps_per_day
            d_olr += diag.global_mean_olr_wm2 / steps_per_day
            d_prec += diag.global_mean_precip_mm_day / steps_per_day
            d_lai += diag.land_mean_lai / steps_per_day
            d_gpp += diag.land_mean_gpp_umol / steps_per_day
            d_ice += diag.seaice_area_million_km2 / steps_per_day

            if in_annual_window:
                rain_s = model.rad_cache["rain"]
                snow_s = model.rad_cache["snow"]
                ptot_s = rain_s + snow_s
                sw_tot_s = (
                    model.rad_cache["sw1_dir"]
                    + model.rad_cache["sw1_dif"]
                    + model.rad_cache["sw2_dir"]
                    + model.rad_cache["sw2_dif"]
                )
                pres_c, _, temp_c = diagnose_pressure_temperature(
                    model.atmos.rho_d, model.atmos.theta_m, model.atmos.q_v
                )
                t_air = temp_c[0]
                es_1 = 611.2 * torch.exp(17.67 * (t_air - C.T_FREEZE) / torch.clamp(t_air - 29.65, min=10.0))
                e_1 = model.atmos.q_v[0] * pres_c[0] / (C.EPS_WATER + (1.0 - C.EPS_WATER) * model.atmos.q_v[0])
                vpd_1 = torch.clamp(es_1 - e_1, min=50.0)

                day_p += ptot_s * dt_host
                day_pet += torch.clamp(sw_tot_s, min=0.0) * dt_host / C.L_V0
                day_ppfd += (
                    4.57 * model.rad_cache["sw1_dir"] + 4.24 * model.rad_cache["sw1_dif"]
                ) * 1.0e-6 * dt_host
                day_t += t_air / steps_per_day
                day_vpd += vpd_1 / steps_per_day
                day_tmin = torch.minimum(day_tmin, t_air)

                # Track local solar noon peak insolation per cell
                mu0_s, _, _ = compute_orbital_insolation(
                    t_before + 0.5 * dt_host,
                    model.mesh.lat_cell,
                    model.mesh.lon_cell,
                    cfg.solar_constant,
                    cfg.obliquity_deg,
                    cfg.eccentricity,
                    cfg.perihelion_deg,
                )
                higher_sun = mu0_s > noon_mu0
                noon_mu0 = torch.where(higher_sun, mu0_s, noon_mu0)
                noon_vb = torch.where(higher_sun, model.rad_cache["sw1_dir"], noon_vb)
                noon_vd = torch.where(higher_sun, model.rad_cache["sw1_dif"], noon_vd)
                noon_t = torch.where(higher_sun, t_air, noon_t)
                noon_vpd = torch.where(higher_sun, vpd_1, noon_vpd)

                # Sample 6-hourly (4x/day) for diurnal-unbiased 3D & 2D seasonal means
                if s % 12 == 0:
                    u_c, v_c = reconstruct_cell_winds(model.mesh, model.atmos.u_edge)
                    w_c = 0.5 * (model.atmos.w[:-1] + model.atmos.w[1:])
                    qcond = model.atmos.q_c + model.atmos.q_i
                    t_sfc_map = torch.where(
                        model.sfc.surf_type == SURF_OCEAN, model.sfc.t_ocean_ml, model.sfc.t_canopy
                    )
                    f_ice = torch.where(
                        model.sfc.surf_type == SURF_OCEAN,
                        torch.clamp(model.sfc.m_seaice / 300.0, 0.0, 0.995),
                        torch.zeros_like(t_sfc_map),
                    )
                    alb_vd, alb_vf, alb_nd, alb_nf, _, _, _, _ = compute_surface_albedos(
                        model.sfc, torch.full((nc,), 0.5)
                    )
                    alb_broad = 0.5 * (0.5 * (alb_vd + alb_vf) + 0.5 * (alb_nd + alb_nf))

                    # Surface evaporation proxy from bulk latent flux or moisture tendency
                    m_idx = min(int(((t_before % yr_sec) / yr_sec) * 12.0), 11)
                    targets = [acc_ann]
                    if m_idx in (11, 0, 1):
                        targets.append(acc_djf)
                    elif m_idx in (5, 6, 7):
                        targets.append(acc_jja)

                    for acc in targets:
                        acc["count"] += 1
                        acc["u"] += u_c
                        acc["v"] += v_c
                        acc["w"] += w_c
                        acc["temp"] += temp_c
                        acc["qv"] += model.atmos.q_v
                        acc["qcond"] += qcond
                        acc["pres"] += pres_c
                        acc["t_sfc"] += t_sfc_map
                        acc["t_air"] += t_air
                        acc["precip"] += ptot_s * C.SECONDS_PER_DAY
                        acc["evap"] += model.rad_cache["evap"] * C.SECONDS_PER_DAY
                        acc["olr"] += model.rad_cache["olr"]
                        acc["asr"] += model.rad_cache["asr"]
                        acc["f_seaice"] += f_ice
                        acc["m_snow"] += model.sfc.swe_land
                        acc["sst"] += model.sfc.t_ocean_ml
                        acc["w_ekman"] += model.ocn.w_ekman
                        acc["alb_sw"] += alb_broad

        if not math.isfinite(d_tsfc):
            raise RuntimeError(f"Non-finite T_sfc encountered on day {day_idx + 1}!")

        if in_annual_window:
            k_yr = day_idx - spinup_days
            pres_c, _, _ = diagnose_pressure_temperature(
                model.atmos.rho_d, model.atmos.theta_m, model.atmos.q_v
            )
            dz0 = model.atmos.z_c[0] - model.atmos.z_w[0]
            psfc = pres_c[0] + C.GRAVITY * model.atmos.rho_d[0] * (1.0 + model.atmos.q_v[0]) * dz0
            ws_b = torch.clamp(
                model.sfc.w_root / (0.5 * torch.clamp(model.sfc.bio.s_root_cap, min=20.0)), 0.0, 1.0
            )
            noon_t_safe = torch.where(noon_t > 100.0, noon_t, day_t)

            ann_daily_drivers["p_mm"][k_yr] = day_p
            ann_daily_drivers["pet_mm"][k_yr] = day_pet
            ann_daily_drivers["ppfd_mol"][k_yr] = day_ppfd
            ann_daily_drivers["temp_k"][k_yr] = day_t
            ann_daily_drivers["vpd_pa"][k_yr] = day_vpd
            ann_daily_drivers["t_min_k"][k_yr] = day_tmin
            ann_daily_drivers["psfc"][k_yr] = psfc
            ann_daily_drivers["ws_beta"][k_yr] = ws_b
            ann_daily_drivers["m_snow"][k_yr] = model.sfc.swe_land
            ann_daily_drivers["noon_vb"][k_yr] = noon_vb
            ann_daily_drivers["noon_vd"][k_yr] = noon_vd
            ann_daily_drivers["noon_mu0"][k_yr] = noon_mu0
            ann_daily_drivers["noon_t"][k_yr] = noon_t_safe
            ann_daily_drivers["noon_vpd"][k_yr] = noon_vpd

            if (k_yr + 1) % 30 == 0 or k_yr == annual_days - 1:
                ds = export_aoba_snapshot_xarray(model)
                append_zarr(ds, zarr_path)

            # Freeze clean 365-day Köppen-Geiger climate right at Dec 31 of Year 1!
            if k_yr == annual_days - 1:
                kg_code30, kg_major5 = diagnose_model_koppen_geiger(model)
                steps_safe = torch.clamp(model.clim.month_steps, min=1.0)
                days_safe = torch.clamp(model.clim.month_days_accum, min=1.0e-3)
                t_mon_c = (model.clim.month_temp_sum_c / steps_safe).clone()
                p_mon_mm = (model.clim.month_precip_mm * ((C.DAYS_PER_YEAR / 12.0) / days_safe)).clone()

                # Spin up slow multi-year EEO structural pools (5 cycles over AOBA's own 365-day
                # endogenous climate — zero external data!) so 1-yr e-folding pools reach <1% residual
                print(">>> Equilibrating slow multi-year EEO biosphere pools over endogenous 365-day AOBA climate...", flush=True)
                veg_mask = model.sfc.surf_type == SURF_LAND
                bio = model.sfc.bio
                lai_seasonal = torch.zeros((annual_days, nc))
                gpp_seasonal = torch.zeros((annual_days, nc))
                f4_seasonal = torch.zeros((annual_days, nc))
                vc3_seasonal = torch.zeros((annual_days, nc))

                for cyc in range(5):
                    bio.cum_deficit.zero_()
                    for kd in range(annual_days):
                        lai_exp_d = bio.lai * torch.clamp(1.0 - ann_daily_drivers["m_snow"][kd] / 50.0, 0.0, 1.0)
                        update_daily_acclimation_smith(
                            bio=bio,
                            midday_vis_beam=ann_daily_drivers["noon_vb"][kd],
                            midday_vis_diff=ann_daily_drivers["noon_vd"][kd],
                            midday_mu0=ann_daily_drivers["noon_mu0"][kd],
                            midday_temp=ann_daily_drivers["noon_t"][kd],
                            midday_vpd=ann_daily_drivers["noon_vpd"][kd],
                            night_temp=ann_daily_drivers["t_min_k"][kd],
                            pres_sfc=ann_daily_drivers["psfc"][kd],
                            co2_ppmv=cfg.co2_ppmv,
                            lai_exposed=lai_exp_d,
                            is_vegetated_land=veg_mask,
                        )
                        update_daily_lai_root_height(
                            bio=bio,
                            daily_precip_mm=ann_daily_drivers["p_mm"][kd],
                            daily_pet_mm=torch.clamp(ann_daily_drivers["pet_mm"][kd], min=0.05),
                            daily_ppfd_mol_m2=ann_daily_drivers["ppfd_mol"][kd],
                            daily_temp_k=ann_daily_drivers["temp_k"][kd],
                            daily_vpd_pa=ann_daily_drivers["vpd_pa"][kd],
                            pres_sfc=ann_daily_drivers["psfc"][kd],
                            co2_ppmv=cfg.co2_ppmv,
                            water_stress_beta=ann_daily_drivers["ws_beta"][kd],
                            is_vegetated_land=veg_mask,
                        )
                        if cyc == 4:
                            lai_seasonal[kd] = bio.lai
                            f4_seasonal[kd] = bio.f_c4
                            vc3_seasonal[kd] = bio.vcmax25_c3
                            # Evaluate midday GPP -> daily mean GPP [umol m^-2 s^-1]
                            flx_m = compute_two_leaf_gas_exchange(
                                bio=bio,
                                vis_beam_wm2=ann_daily_drivers["noon_vb"][kd],
                                vis_diff_wm2=ann_daily_drivers["noon_vd"][kd],
                                mu0=ann_daily_drivers["noon_mu0"][kd],
                                temp_sun=ann_daily_drivers["noon_t"][kd],
                                temp_sh=ann_daily_drivers["noon_t"][kd],
                                vpd_sun=ann_daily_drivers["noon_vpd"][kd],
                                vpd_sh=ann_daily_drivers["noon_vpd"][kd],
                                pres_sfc=ann_daily_drivers["psfc"][kd],
                                co2_ppmv=cfg.co2_ppmv,
                                lai_exposed=lai_exp_d,
                                water_stress_beta=ann_daily_drivers["ws_beta"][kd],
                            )
                            # Diurnal integral of cos-bell insolation over daylength ~ 0.32 * midday peak
                            gpp_seasonal[kd] = torch.where(veg_mask, 0.32 * flx_m.gpp, torch.zeros_like(flx_m.gpp))

                print(">>> Starting 30-day two-way coupled atmosphere-biosphere verification pass!", flush=True)

        ts_day.append(float(day_idx - spinup_days + 1))
        ts_tsfc.append(d_tsfc)
        ts_asr.append(d_asr)
        ts_olr.append(d_olr)
        ts_precip.append(d_prec)
        ts_lai.append(d_lai)
        ts_gpp.append(d_gpp)
        ts_seaice.append(d_ice)

        if (day_idx + 1) % 30 == 0 or day_idx in (spinup_days - 1, spinup_days + annual_days - 1, total_days - 1):
            elapsed = time.time() - t_wall0
            rate = elapsed / (day_idx + 1)
            rem = rate * (total_days - (day_idx + 1))
            print(
                f"[StepDay {day_idx+1:3d}/{total_days} | RelDay {day_idx-spinup_days+1:+4d}] "
                f"Ts={d_tsfc:6.2f} K | ASR={d_asr:5.1f} | OLR={d_olr:5.1f} | "
                f"P={d_prec:4.2f} mm/d | LAI={d_lai:4.2f} | GPP={d_gpp:4.2f} | Ice={d_ice:5.2f} Mkm2 | "
                f"{elapsed:5.0f}s elapsed (~{rem:4.0f}s left)",
                flush=True,
            )

    def _norm_acc(acc):
        cnt = max(acc["count"], 1)
        return {k: (v / cnt if isinstance(v, torch.Tensor) else v) for k, v in acc.items()}

    ann_norm = _norm_acc(acc_ann)
    djf_norm = _norm_acc(acc_djf)
    jja_norm = _norm_acc(acc_jja)

    # -----------------------------------------------------------------------
    # Phase D: Active 1:1 3D Nonhydrostatic SP-CRM Showcase (8x8x32 on MPS GPU)
    # -----------------------------------------------------------------------
    dev_sp = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== Phase D: Running 1:1 Embedded 8x8x32 3D SP-CRM Showcase on {dev_sp} (dt=180s, 40 steps = 2h) ===", flush=True)
    t_sp0 = time.time()
    cfg_sp = AOBAConfig(
        glevel=3,
        nz=32,
        crm_nx=8,
        crm_ny=8,
        crm_dx=4000.0,
        dt_host=180.0,
        n_host_dyn_substeps=1,
        n_crm_dyn_substeps=6,
        rad_stride=4,
        crm_col_chunk=256,
        co2_ppmv=400.0,
        enable_superparam=True,
    )
    model_sp = init_aoba_model(cfg_sp, device=dev_sp, dtype=torch.float32)
    model_sp.sim_time_sec = model.sim_time_sec

    for attr in ("rho_d", "u_edge", "w", "theta_m", "q_n", "q_p", "q_v", "q_c", "q_i"):
        setattr(model_sp.atmos, attr, getattr(model.atmos, attr).to(dev_sp).clone())
    for attr in model.sfc.__dataclass_fields__:
        if attr == "bio":
            continue
        val = getattr(model.sfc, attr)
        if isinstance(val, torch.Tensor):
            setattr(model_sp.sfc, attr, val.to(dev_sp).clone())
    for attr in model.sfc.bio.__dataclass_fields__:
        val = getattr(model.sfc.bio, attr)
        if isinstance(val, torch.Tensor):
            setattr(model_sp.sfc.bio, attr, val.to(dev_sp).clone())
    broadcast_host_surface_to_subgrid(model_sp)
    model_sp.ocn.t_sub_ml = model.ocn.t_sub_ml.to(dev_sp).clone()
    model_sp.ocn.w_ekman = model.ocn.w_ekman.to(dev_sp).clone()

    u_sp0, v_sp0 = reconstruct_cell_winds(model_sp.mesh, model_sp.atmos.u_edge)
    shape4 = (nz, nc, 8, 8)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2026)
    pert = (torch.rand((3, nc, 8, 8), generator=gen) - 0.5) * 0.35
    pert = (pert - pert.mean(dim=(-2, -1), keepdim=True)).to(dev_sp)

    thm4 = model_sp.atmos.theta_m.view(nz, nc, 1, 1).expand(shape4).clone()
    thm4[:3] = thm4[:3] + pert
    model_sp.crm = CRMState(
        rho_d=model_sp.atmos.rho_d.view(nz, nc, 1, 1).expand(shape4).clone(),
        u=u_sp0.view(nz, nc, 1, 1).expand(shape4).clone(),
        v=v_sp0.view(nz, nc, 1, 1).expand(shape4).clone(),
        w=torch.zeros((nz + 1, nc, 8, 8), device=dev_sp),
        theta_m=thm4,
        q_n=model_sp.atmos.q_n.view(nz, nc, 1, 1).expand(shape4).clone(),
        q_p=model_sp.atmos.q_p.view(nz, nc, 1, 1).expand(shape4).clone(),
        q_v=model_sp.atmos.q_v.view(nz, nc, 1, 1).expand(shape4).clone(),
        q_c=model_sp.atmos.q_c.view(nz, nc, 1, 1).expand(shape4).clone(),
        q_i=model_sp.atmos.q_i.view(nz, nc, 1, 1).expand(shape4).clone(),
        qke=torch.full(shape4, 2.0 * TKE_MIN, device=dev_sp),
        dx=4000.0,
        dy=4000.0,
        f_coriolis=model_sp.mesh.f_cell.clone(),
    )

    for k_sp in range(args.sp_steps):
        diag_sp = step_aoba_model(model_sp)
        if (k_sp + 1) % max(args.sp_steps // 4, 1) == 0:
            w_max_sp = float(model_sp.crm.w.abs().max().item())
            print(
                f"[3D SP-CRM step {k_sp+1:2d}/{args.sp_steps}] Ts={diag_sp.global_mean_t_sfc_k:6.2f} K | "
                f"P={diag_sp.global_mean_precip_mm_day:4.2f} mm/d | max|w_crm|={w_max_sp:5.2f} m/s | "
                f"{time.time()-t_sp0:.1f}s",
                flush=True,
            )

    if annual_days >= 365:
        lai_djf = torch.cat([lai_seasonal[:59], lai_seasonal[334:]], dim=0).mean(dim=0).cpu()
        lai_jja = lai_seasonal[151:243].mean(dim=0).cpu()
        gpp_jja = gpp_seasonal[151:243].mean(dim=0).cpu()
    else:
        lai_djf = lai_seasonal[: max(annual_days // 2, 1)].mean(dim=0).cpu()
        lai_jja = lai_seasonal[-max(annual_days // 2, 1) :].mean(dim=0).cpu()
        gpp_jja = gpp_seasonal[-max(annual_days // 2, 1) :].mean(dim=0).cpu()

    # Reconstruct spherical Voronoi vertex coordinates & per-cell vertex indices from SCVTMesh topology
    ev = model.mesh.edges_on_vert                                  # [N_verts, 3]
    ce_on_v = model.mesh.cells_on_edge[ev]                         # [N_verts, 3, 2]
    xyz_sum = model.mesh.xyz_cell[ce_on_v].sum(dim=(1, 2))         # Each of the 3 cells around v appears twice
    xyz_vert = xyz_sum / torch.clamp(torch.norm(xyz_sum, dim=-1, keepdim=True), min=1.0e-12)
    lat_vert_deg = torch.rad2deg(torch.asin(torch.clamp(xyz_vert[:, 2], -1.0, 1.0))).cpu()
    lon_vert_deg = torch.rad2deg(torch.atan2(xyz_vert[:, 1], xyz_vert[:, 0])).cpu()

    e_cell = model.mesh.edges_on_cell                              # [N_cells, 6]
    sgn_cell = model.mesh.edge_sign_on_cell                        # [N_cells, 6]
    verts_on_cell = torch.where(
        sgn_cell >= 0.0,
        model.mesh.verts_on_edge[e_cell, 0],
        model.mesh.verts_on_edge[e_cell, 1],
    ).cpu()

    payload = {
        "lat_deg": torch.rad2deg(model.mesh.lat_cell).cpu(),
        "lon_deg": torch.rad2deg(model.mesh.lon_cell).cpu(),
        "area_cell": model.mesh.area_cell.cpu(),
        "surf_type": model.sfc.surf_type.cpu(),
        "z_topo": model.mesh.z_topo.cpu(),
        "sigma_topo": model.mesh.sigma_topo.cpu(),
        "verts_on_cell": verts_on_cell,
        "n_edges_on_cell": model.mesh.n_edges_on_cell.cpu(),
        "lat_vert_deg": lat_vert_deg,
        "lon_vert_deg": lon_vert_deg,
        "zeta_c": model.vgrid.zeta_c.cpu(),
        "zeta_w": model.vgrid.zeta_w.cpu(),
        "z_c": model.atmos.z_c.cpu(),
        "ts": {
            "day": torch.tensor(ts_day),
            "t_sfc": torch.tensor(ts_tsfc),
            "asr": torch.tensor(ts_asr),
            "olr": torch.tensor(ts_olr),
            "precip": torch.tensor(ts_precip),
            "lai": torch.tensor(ts_lai),
            "gpp": torch.tensor(ts_gpp),
            "seaice": torch.tensor(ts_seaice),
        },
        "ann": ann_norm,
        "djf": djf_norm,
        "jja": jja_norm,
        "monthly": {
            "t_mon_c": t_mon_c.cpu(),
            "p_mon_mm": p_mon_mm.cpu(),
        },
        "biosphere": {
            "lai_ann_mean": lai_seasonal.mean(dim=0).cpu(),
            "lai_peak": lai_seasonal.max(dim=0).values.cpu(),
            "lai_djf": lai_djf,
            "lai_jja": lai_jja,
            "gpp_ann_mean": gpp_seasonal.mean(dim=0).cpu(),
            "gpp_jja": gpp_jja,
            "f_c4_ann": f4_seasonal.mean(dim=0).cpu(),
            "vcmax25_c3_ann": vc3_seasonal.mean(dim=0).cpu(),
            "vcmax25_c3_peak": vc3_seasonal.max(dim=0).values.cpu(),
            "h_canopy": model.sfc.bio.h_canopy.cpu(),
            "s_root_cap": model.sfc.bio.s_root_cap.cpu(),
            "ema_ann_precip": model.sfc.bio.ema_ann_precip.cpu(),
            "ema_ann_pet": model.sfc.bio.ema_ann_pet.cpu(),
        },
        "koppen": {
            "code_30": kg_code30.cpu(),
            "major_5": kg_major5.cpu(),
        },
        "sp_crm": {
            "w": model_sp.crm.w.detach().cpu(),
            "q_cond": (model_sp.crm.q_c + model_sp.crm.q_i).detach().cpu(),
            "q_p": model_sp.crm.q_p.detach().cpu(),
            "theta_m_anom": (
                model_sp.crm.theta_m - model_sp.crm.theta_m.mean(dim=(-2, -1), keepdim=True)
            ).detach().cpu(),
        },
    }
    torch.save(payload, diag_path)
    print(f"\nSaved diagnostics bundle -> {diag_path}")
    print(f"Saved CF-1.11 / UGRID-1.0 Zarr v3 store -> {zarr_path}")
    print(f"Total simulation wall-clock time: {(time.time() - t_wall0)/60.0:.2f} min")


if __name__ == "__main__":
    main()
