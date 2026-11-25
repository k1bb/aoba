"""Map-Free Prognostic Atmospheric CH4 & N2O Cycles with OH & Stratospheric Photolysis Feedback.

Closes the natural non-CO2 greenhouse-gas biogeochemical loops completely map-free:
1. **Prognostic CH4 (`ch4_ppmv_prog`)**:
   - Sources: (i) anoxic wetland & permafrost-thaw soil methanogenesis driven directly by
     AOBA's 5-layer soil heterotrophic respiration under warm waterlogged soils
     (Walter & Heimann, 2000, *GBC*, 14, 745; Riley et al., 2011, *BG*, 8, 1925), plus
     (ii) pyrogenic wildfire CH4 (`0.006 kg CH4 / kg C burned`; Andreae & Merlet, 2001).
   - Sinks: (i) tropospheric OH oxidation with temperature, humidity, overhead-O3 UV, and
     CH4-OH self-feedback (`d ln[OH] / d ln[CH4] = -0.32`, Prather et al., 2012, *GRL*;
     Holmes et al., 2013, *ACP*), plus (ii) aerated soil microbial methanotrophy (~30 Tg/yr).
2. **Prognostic N2O (`n2o_ppmv_prog`)**:
   - Sources: (i) terrestrial microbial denitrification & nitrification coupled directly to
     `land_carbon.n_denit_kgn_m2s` & soil mineralization (Butterbach-Bahl et al., 2013,
     *Phil. Trans. R. Soc. B*), plus (ii) marine subsurface remineralization/denitrification
     coupled to `ogcm.bgc.epc100` export production (Battaglia & Joos, 2018, *BG*, 15, 3551).
   - Sink: stratospheric UV photolysis & O(1D) destruction (`tau_N2O ~ 116 yr`; Prather et al.,
     2015), modulated by overhead stratospheric ozone self-shielding.

Uses float64 global atmospheric burden integration and exact exponential relaxation
(zero Forward Euler, unconditionally stable for multi-millennial runs).

AOBA Formulation Reference: Sec. 10.6, Eq. (10.7) (Prognostic Atmospheric CH4 & N2O Cycles).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.surface import SOIL_POROSITY, SURF_LAND

SEC_PER_YEAR: float = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
# Conversion factors between atmospheric mixing ratio [ppmv] and global mass burden [Tg]
# Total dry air mass ~ 5.135e18 kg = 1.7728e20 mol -> 1 ppmv = 1.7728e14 mol
TG_CH4_PER_PPMV: float = 2844.0        # 1.7728e14 mol * 16.0425 g/mol = 2844 Tg CH4 / ppmv
TG_N2O_N_PER_PPMV: float = 4966.0      # 1.7728e14 mol * 28.0134 g N/mol = 4966 Tg N / ppmv

# Physical source/sink constants (map-free universal global parameters)
CH4_PI_REF_PPMV: float = 0.80825       # 1850 piControl reference [ppmv]
N2O_PI_REF_PPMV: float = 0.27302       # 1850 piControl reference [ppmv]
CH4_TAU_OH_REF_YR: float = 9.5         # Pre-industrial tropospheric OH lifetime [yr] (Prather et al., 2012)
CH4_OH_SELF_EXP: float = -0.32         # CH4-OH self-feedback exponent d ln[OH]/d ln[CH4] [-]
CH4_SOIL_SINK_YR: float = 1.0 / 120.0  # Aerated upland soil methanotrophy sink rate [yr^-1]
CH4_FIRE_EF: float = 0.006             # Wildfire emission factor [kg CH4 / kg C burned]
CH4_WETLAND_FRAC: float = 0.045        # Fraction of anoxic waterlogged soil Rh emitted as CH4-C (16/12 mass ratio)
CH4_BG_GEO_TGYR: float = 20.0          # Natural geological/marine/termite background [Tg CH4 yr^-1]

N2O_TAU_STRAT_REF_YR: float = 116.0    # Stratospheric photolysis lifetime [yr] (Prather et al., 2015)
N2O_DENIT_YIELD: float = 0.025         # Fraction of soil denitrification N released as N2O-N [-]
N2O_RH_NITRIF_YIELD: float = 3.5e-4    # Soil nitrification N2O-N per kg soil Rh-C [kg N / kg C]
N2O_OCEAN_EXPORT_YIELD: float = 3.5e-4 # Marine subsurface N2O-N per kg exported POC (Battaglia & Joos, 2018)


@dataclass
class TraceGasCycleState:
    """Prognostic well-mixed atmospheric CH4 & N2O state [ppmv] and global flux diagnostics."""

    ch4_ppmv: float
    n2o_ppmv: float
    # Instantaneous global flux diagnostics [Tg CH4/yr] and [Tg N/yr]
    ch4_emis_wetland_tgyr: float = 0.0
    ch4_emis_fire_tgyr: float = 0.0
    ch4_sink_oh_tgyr: float = 0.0
    n2o_emis_land_tgn_yr: float = 0.0
    n2o_emis_ocean_tgn_yr: float = 0.0
    n2o_sink_strat_tgn_yr: float = 0.0


def init_trace_gas_cycle_state(
    ch4_ppmv_init: float = CH4_PI_REF_PPMV,
    n2o_ppmv_init: float = N2O_PI_REF_PPMV,
) -> TraceGasCycleState:
    """Initialize prognostic CH4 & N2O state from configured baseline concentrations."""
    return TraceGasCycleState(
        ch4_ppmv=max(float(ch4_ppmv_init), 0.05),
        n2o_ppmv=max(float(n2o_ppmv_init), 0.05),
    )


def step_trace_gas_cycle(
    tg: TraceGasCycleState,
    area_cell: torch.Tensor,
    surf_type: torch.Tensor,
    t_sfc_k: torch.Tensor,
    w_liq_soil_top: torch.Tensor,
    dt: float,
    *,
    rh_kgc_m2s: torch.Tensor | None = None,
    fire_c_kgc_m2s: torch.Tensor | None = None,
    n_denit_kgn_m2s: torch.Tensor | None = None,
    ocean_epc100_molc_m2s: torch.Tensor | None = None,
    ocean_denit_moln_m2s: torch.Tensor | None = None,
    ocean_area_cell: torch.Tensor | None = None,
    f_wetland_sub: torch.Tensor | None = None,
    mean_tco_du: float = 300.0,
) -> None:
    """Advance global prognostic CH4 & N2O mixing ratios via exact exponential integration (float64)."""
    dt_yr = float(dt) / SEC_PER_YEAR
    a_host = area_cell.detach().cpu().to(torch.float64)
    a_tot = max(float(a_host.sum().item()), 1.0)
    t_glob = float((t_sfc_k.detach().cpu().to(torch.float64) * a_host).sum().item() / a_tot)

    def _area_for(t: torch.Tensor) -> torch.Tensor:
        nc = a_host.shape[0]
        m = t.shape[0]
        if m != nc and (m % nc == 0):
            ns = m // nc
            return (a_host / float(ns)).unsqueeze(1).expand(nc, ns).reshape(m)
        return a_host

    # 1. CH4 Emissions [Tg CH4 yr^-1]
    emis_wet_tg = 0.0
    if rh_kgc_m2s is not None:
        rh64 = torch.clamp(rh_kgc_m2s.detach().cpu().to(torch.float64), min=0.0)
        a_lc = _area_for(rh64)
        sat = torch.clamp(w_liq_soil_top.detach().cpu().to(torch.float64) / SOIL_POROSITY, 0.0, 1.0)
        if sat.shape[0] != rh64.shape[0]:
            sat = sat.unsqueeze(1).expand(a_host.shape[0], rh64.shape[0] // a_host.shape[0]).reshape(-1)
        anoxic = torch.clamp((sat - 0.65) / 0.35, 0.0, 1.0) ** 2
        if f_wetland_sub is not None:
            fw64 = torch.clamp(f_wetland_sub.detach().cpu().to(torch.float64), 0.0, 1.0)
            if fw64.shape[0] != rh64.shape[0]:
                fw64 = fw64.unsqueeze(1).expand(a_host.shape[0], rh64.shape[0] // a_host.shape[0]).reshape(-1)
            anoxic = torch.clamp(0.5 * anoxic + 0.5 * fw64, 0.0, 1.0)
        # Convert kg C m^-2 s^-1 -> Tg CH4 yr^-1 (16.0425 / 12.011 = 1.3356)
        emis_wet_tg = float((rh64 * anoxic * a_lc).sum().item() * SEC_PER_YEAR * 1.0e-9 * CH4_WETLAND_FRAC * 1.3356)
    else:
        # Fallback temperature-sensitive natural wetland emission (~175 Tg CH4/yr at 287 K)
        emis_wet_tg = 175.0 * math.exp(0.06 * max(-15.0, min(15.0, t_glob - 287.0)))

    emis_fire_tg = 0.0
    if fire_c_kgc_m2s is not None:
        fc64 = torch.clamp(fire_c_kgc_m2s.detach().cpu().to(torch.float64), min=0.0)
        emis_fire_tg = float((fc64 * _area_for(fc64)).sum().item() * SEC_PER_YEAR * 1.0e-9 * CH4_FIRE_EF)

    e_ch4_tot = emis_wet_tg + emis_fire_tg + CH4_BG_GEO_TGYR

    # Tropospheric OH sink rate [yr^-1]: temperature Q10 + UV/O3 + CH4-OH self-feedback
    t_oh_fac = math.exp(0.032 * max(-20.0, min(20.0, t_glob - 287.0)))
    uv_oh_fac = (300.0 / max(float(mean_tco_du), 120.0)) ** 0.35
    ch4_oh_fac = (max(tg.ch4_ppmv, 0.05) / CH4_PI_REF_PPMV) ** CH4_OH_SELF_EXP
    k_ch4_oh = (1.0 / CH4_TAU_OH_REF_YR) * t_oh_fac * uv_oh_fac * ch4_oh_fac
    k_ch4_tot = k_ch4_oh + CH4_SOIL_SINK_YR

    # Exact exponential update for d(C)/dt = E/M - k*C
    p_ch4 = e_ch4_tot / TG_CH4_PER_PPMV
    c_eq = p_ch4 / max(k_ch4_tot, 1.0e-9)
    ch4_new = tg.ch4_ppmv + (c_eq - tg.ch4_ppmv) * (-math.expm1(-k_ch4_tot * dt_yr))
    tg.ch4_ppmv = max(0.05, min(25.0, ch4_new))
    tg.ch4_emis_wetland_tgyr = emis_wet_tg
    tg.ch4_emis_fire_tgyr = emis_fire_tg
    tg.ch4_sink_oh_tgyr = k_ch4_oh * tg.ch4_ppmv * TG_CH4_PER_PPMV

    # 2. N2O Emissions [Tg N yr^-1]
    emis_n2o_land = 0.0
    if n_denit_kgn_m2s is not None:
        dn64 = torch.clamp(n_denit_kgn_m2s.detach().cpu().to(torch.float64), min=0.0)
        emis_n2o_land += float((dn64 * _area_for(dn64)).sum().item() * SEC_PER_YEAR * 1.0e-9 * N2O_DENIT_YIELD)
    if rh_kgc_m2s is not None:
        rh64 = torch.clamp(rh_kgc_m2s.detach().cpu().to(torch.float64), min=0.0)
        emis_n2o_land += float((rh64 * _area_for(rh64)).sum().item() * SEC_PER_YEAR * 1.0e-9 * N2O_RH_NITRIF_YIELD)
    if n_denit_kgn_m2s is None and rh_kgc_m2s is None:
        emis_n2o_land = 6.5 * math.exp(0.04 * max(-15.0, min(15.0, t_glob - 287.0)))

    emis_n2o_ocn = 0.0
    if ocean_epc100_molc_m2s is not None and ocean_area_cell is not None:
        epc64 = torch.clamp(ocean_epc100_molc_m2s.detach().cpu().to(torch.float64), min=0.0)
        ao64 = ocean_area_cell.detach().cpu().to(torch.float64)
        # mol C m^-2 s^-1 * 0.012011 kg C / mol -> Tg C/yr * yield -> Tg N/yr
        emis_n2o_ocn = float((epc64 * ao64).sum().item() * 0.012011 * SEC_PER_YEAR * 1.0e-9 * N2O_OCEAN_EXPORT_YIELD)
        if ocean_denit_moln_m2s is not None:
            # OMZ anaerobic denitrification N2O hotspot (~3% N2O yield; Battaglia & Joos, 2018)
            od64 = torch.clamp(ocean_denit_moln_m2s.detach().cpu().to(torch.float64), min=0.0)
            emis_n2o_ocn += float((od64 * ao64).sum().item() * 0.014007 * SEC_PER_YEAR * 1.0e-9 * 0.030)
    else:
        emis_n2o_ocn = 3.8

    e_n2o_tot = emis_n2o_land + emis_n2o_ocn
    uv_n2o_fac = (300.0 / max(float(mean_tco_du), 120.0)) ** 0.25
    k_n2o = (1.0 / N2O_TAU_STRAT_REF_YR) * uv_n2o_fac
    p_n2o = e_n2o_tot / TG_N2O_N_PER_PPMV
    n2o_eq = p_n2o / max(k_n2o, 1.0e-9)
    n2o_new = tg.n2o_ppmv + (n2o_eq - tg.n2o_ppmv) * (-math.expm1(-k_n2o * dt_yr))
    tg.n2o_ppmv = max(0.05, min(2.0, n2o_new))
    tg.n2o_emis_land_tgn_yr = emis_n2o_land
    tg.n2o_emis_ocean_tgn_yr = emis_n2o_ocn
    tg.n2o_sink_strat_tgn_yr = k_n2o * tg.n2o_ppmv * TG_N2O_N_PER_PPMV
