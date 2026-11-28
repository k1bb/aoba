"""FlexPFT Optimality-Based 3D Marine Biogeochemistry & OMIP/mocsy 2.0 Carbonate Chemistry.

Couples seamlessly to AOBA's 3D SCVT Voronoi OGCM (`ocean_ogcm.py`) as a map-free
marine carbon and ecosystem module (`enable_ocean_bgc = true` by default).
Mirrors AOBA's Eco-Evolutionary Optimality (EEO) terrestrial biosphere (`biosphere.py`)
in the ocean: phytoplankton instantaneously optimize their intracellular resource
allocation (light-harvesting Chl:C vs. nutrient-uptake machinery vs. carbon fixation)
and cellular stoichiometry (variable C:N and Chl:C ratios) without requiring rigid
fixed-Redfield PFT lookup tables or costly intracellular quota tracers.

Prognostic 3D Tracers (`[N_zo, N_cells]`, float32 + Neumaier/Kahan compensation):
1. `no3`   : Dissolved inorganic nitrogen NO3- [mmol N m^-3]
2. `phy`   : Phytoplankton nitrogen biomass P [mmol N m^-3] (flexible C:N = 1/Q, Chl:N = theta_hat/Q)
3. `zoo`   : Zooplankton nitrogen biomass Z [mmol N m^-3]
4. `det_n` : Sinking particulate organic nitrogen detritus D_N [mmol N m^-3]
5. `det_c` : Sinking particulate organic carbon detritus D_C [mmol C m^-3]
6. `dic`   : Dissolved inorganic carbon C_T [mmol C m^-3]
7. `alk`   : Total alkalinity A_T [mmol eq m^-3]

Key Physical & Biogeochemical Processes:
1. **FlexPFT Optimality Growth & Photoacclimation** (Pahlow, Dietze, & Oschlies, 2013,
   *MEPS*, 473, 1-5; Pahlow & Oschlies, 2013, *MEPS*, 489, 1-16; Smith et al., 2016,
   *J. Plankton Res.*, 38, 977-992; Sasai, Smith et al., 2022, *Biogeosciences*, 19,
   4865-4882, Eqs. 2-7 & Table 1):
   - Overflow-free float32 Lambert W0 solver (`_lambert_w0_exp`) for optimal chloroplast
     Chl:C ratio `theta_hat(I, T)`, naturally reproducing deep Subsurface Chlorophyll
     Maxima (SCM) in oligotrophic subtropical gyres.
   - Optimal cellular nitrogen quota `Q(N, I, T)` and nutrient-uptake allocation `f_V`,
     producing realistic flexible C:N (~5.5 in light-limited deep layers to ~18 in
     oligotrophic gyres) and feeding variable C:N export into `det_c`.
2. **Conservative NPZD Food Web, Implicit Detritus Sinking & CaCO3 Pump**:
   - Holling Type III grazing, linear/quadratic mortality, remineralization, and
     unconditionally stable implicit vertical sinking of `det_n` and `det_c` with
     complete benthic remineralization at the seafloor (strictly closed ocean box).
   - Strictly conserves total ocean nitrogen (`no3 + phy + zoo + det_n`), total carbon
     (`dic + det_c + C_org`, zero air-sea flux), and alkalinity (`alk + no3`).
3. **OMIP / mocsy 2.0 Surface Carbonate Chemistry & Air-Sea CO2 Exchange** (Orr &
   Epitalon, 2015, *GMD*, 8, 485-499; Orr et al., 2017, *GMD*, 10, 2169-2199, Sec. 2.5
   & Tables 1-2; Lueker et al., 2000; Dickson, 1990; Millero, 1995; Weiss & Price, 1980;
   Wanninkhof, 2014):
   - Scaled float32-safe SolveSAPHE-style Newton-Raphson pH / [H+] solver on the total
     pH scale (avoids float32 underflow of [H+]^2 ~ 10^-16).
   - Exact Wanninkhof (2014) CO2 Schmidt polynomial & quadratic gas-transfer velocity
     attenuated by sea-ice cover `(1 - f_seaice)`.

AOBA Formulation Reference: Sec. 9.8, Eqs. (9.13)-(9.15) (FlexPFT Marine BGC, OMIP/mocsy 2.0 Carbonate & O2/OMZ Denitrification).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import (
    dispatch_ocean_bgc_column_metal,
    get_active_gpu_backend,
    solve_tridiagonal_batched,
)
from aoba.runtime import maybe_cuda_compile
from aoba.surface import _kahan_add

# ---------------------------------------------------------------------------
# 1. FlexPFT Optimality & Ecosystem Constants (Sasai et al., 2022, Table 1; Smith et al., 2016)
# ---------------------------------------------------------------------------
FLEXPFT_MU_MAX_D: float = 1.4          # Community-mean potential max growth rate mu_max [d^-1] (Sasai et al., 2022)
FLEXPFT_V0_HAT_D: float = 1.0          # Potential max N uptake rate V_hat_0 [mol N (mol C)^-1 d^-1]
FLEXPFT_A0_HAT: float = 1.0            # Potential max N affinity A_hat_0 [m^3 (mmol N)^-1 d^-1]
FLEXPFT_ALPHA_I: float = 2.0           # Chl-specific initial P-I slope [m^2 W^-1 (g Chl)^-1 mol C d^-1]
FLEXPFT_Q_S: float = 0.0195            # Structural minimum N:C cell quota Q_s = Q_0 / 2 [mol N (mol C)^-1]
FLEXPFT_ZETA_N: float = 0.60           # Respiratory cost of inorganic N assimilation [mol C (mol N)^-1]
FLEXPFT_ZETA_CHL: float = 0.80         # Biosynthetic cost of chlorophyll [mol C (g Chl)^-1] (Pahlow et al., 2013)
FLEXPFT_RM_CHL_D: float = 0.05         # Chlorophyll maintenance respiration rate R_M^Chl [d^-1]
FLEXPFT_THETA_MAX: float = 1.0         # Physiological cap on chloroplast Chl:C [g Chl (mol C)^-1]
FLEXPFT_EA_J_MOL: float = 4.8e4        # Arrhenius activation energy E_a (Q10 ~ 2.0) [J mol^-1]
FLEXPFT_T_REF_K: float = 293.15        # Reference temperature T_ref = 20 C [K]
R_GAS_J_MOL_K: float = 8.314462618     # Universal gas constant R [J mol^-1 K^-1]

# Underwater PAR attenuation (Morel, 1988; Sasai et al., 2022)
K_PAR_WATER: float = 0.04              # Pure seawater PAR attenuation coefficient [m^-1]
K_PAR_CHL: float = 0.03                # Chlorophyll self-shading attenuation [m^2 (mg Chl)^-1]
PAR_FRAC_SW: float = 0.43              # PAR fraction of net surface shortwave if PAR not supplied [-]

# NPZD food-web & detritus export parameters (Aumont et al., 2015; Sasai et al., 2022)
ZOO_G_MAX_D: float = 1.0               # Max zooplankton grazing rate at T_ref [d^-1]
ZOO_K_GRAZE: float = 0.5               # Holling Type III half-saturation biomass [mmol N m^-3]
ZOO_ASSIM_EFF: float = 0.70            # Zooplankton N assimilation efficiency [-]
ZOO_EXCR_D: float = 0.05               # Zooplankton linear excretion to NO3/DIC [d^-1]
ZOO_MORT_QUAD_D: float = 0.15          # Zooplankton quadratic mortality to detritus [(mmol N m^-3)^-1 d^-1]
PHY_MORT_LIN_D: float = 0.03           # Phytoplankton linear mortality [d^-1]
PHY_MORT_QUAD_D: float = 0.05          # Phytoplankton quadratic aggregation mortality [(mmol N m^-3)^-1 d^-1]
DET_REMIN_D: float = 0.12              # Detritus remineralization rate at T_ref (z* ~ 125 m) [d^-1]
DET_W_SINK_M_D: float = 15.0           # Particulate detritus sinking speed [m d^-1]
R_CN_ZOO: float = 106.0 / 16.0         # Zooplankton C:N Redfield stoichiometry (6.625 mol C / mol N)
RAIN_RATIO_CACO3: float = 0.07         # CaCO3 : POC export rain ratio (Aumont et al., 2015; Orr et al., 2017)

# OMIP / Wanninkhof (2014) air-sea CO2 gas exchange constants (Orr et al., 2017, Sec. 2.5 & Tables 1-2)
WANNINKHOF_A_MS: float = 6.97e-7       # Quadratic transfer coefficient for m s^-1 [m s^-1 / (m s^-1)^2]
SC_CO2_A: float = 2116.8
SC_CO2_B: float = -136.25
SC_CO2_C: float = 4.7353
SC_CO2_D: float = -0.092307
SC_CO2_E: float = 0.0007555

# Weiss & Price (1980) CO2 solubility function phi_CO2^0 [mol L^-1 atm^-1] (Orr et al., 2017, Table 2)
WEISS_A1: float = -160.7333
WEISS_A2: float = 215.4152
WEISS_A3: float = 89.8920
WEISS_A4: float = -1.47759
WEISS_B1: float = 0.029941
WEISS_B2: float = -0.027455
WEISS_B3: float = 0.0053407

# Global reference initial BGC inventories (WOA/GLODAP global means; strictly map-free)
NO3_GLOBAL_MEAN: float = 30.5          # [mmol N m^-3]
DIC_GLOBAL_MEAN: float = 2255.0        # [mmol C m^-3]
ALK_GLOBAL_MEAN: float = 2365.0        # [mmol eq m^-3]


@dataclass
class OceanBGCState:
    """3D FlexPFT marine biogeochemistry & carbonate chemistry state on the SCVT OGCM mesh."""

    # 7 prognostic 3D tracers [N_zo, N_cells]
    no3: torch.Tensor                  # Dissolved nitrate NO3- [mmol N m^-3]
    phy: torch.Tensor                  # Phytoplankton N biomass [mmol N m^-3]
    zoo: torch.Tensor                  # Zooplankton N biomass [mmol N m^-3]
    det_n: torch.Tensor                # Sinking PON detritus [mmol N m^-3]
    det_c: torch.Tensor                # Sinking POC detritus [mmol C m^-3]
    dic: torch.Tensor                  # Dissolved inorganic carbon C_T [mmol C m^-3]
    alk: torch.Tensor                  # Total alkalinity A_T [mmol eq m^-3]

    # Neumaier/Kahan float32 rounding compensation buffers for slow large-reservoir tracers
    no3_comp: torch.Tensor
    dic_comp: torch.Tensor
    alk_comp: torch.Tensor

    # Instantaneous FlexPFT & OMIP diagnostics (zero-copy for Zarr / CMIP7 diagnostics)
    chl: torch.Tensor                  # 3D Chlorophyll-a concentration [mg Chl m^-3]
    npp_c: torch.Tensor                # 3D Net Primary Production [mmol C m^-3 s^-1]
    c_to_n_phy: torch.Tensor           # 3D Flexible phytoplankton C:N ratio [mol C (mol N)^-1]
    fgco2: torch.Tensor                # 2D Air-sea CO2 flux (positive into ocean) [mol C m^-2 s^-1]
    spco2_uatm: torch.Tensor           # 2D Surface ocean pCO2 [uatm]
    ph_sfc: torch.Tensor               # 2D Surface ocean pH (total scale) [-]
    intpp: torch.Tensor                # 2D Vertically integrated NPP [mol C m^-2 s^-1]
    epc100: torch.Tensor               # 2D Sinking POC export flux across ~100 m [mol C m^-2 s^-1]
    fe_sfc: torch.Tensor | None = None # 2D Surface dissolved Fe [nM = umol Fe m^-3] (dust fertilization)
    # Optional prognostic 3D dissolved oxygen (O2), OMZ, & marine denitrification (enable_ocean_o2_omz)
    o2: torch.Tensor | None = None             # [N_zo, N_cells] dissolved O2 [mmol O2 m^-3]
    o2_comp: torch.Tensor | None = None        # [N_zo, N_cells] Neumaier/Kahan compensation
    fgo2: torch.Tensor | None = None           # [N_cells] air-sea O2 flux (pos into ocean) [mol O2 m^-2 s^-1]
    denit_moln_m2s: torch.Tensor | None = None # [N_cells] column anaerobic denitrification [mol N m^-2 s^-1]
    omz_col_frac: torch.Tensor | None = None   # [N_cells] fraction of water column with O2 < 20 mmol m^-3
    n2fix_moln_m2s: torch.Tensor | None = None # [N_cells] surface diazotrophic N2 fixation [mol N m^-2 s^-1]


O2_GLOBAL_MEAN: float = 205.0          # Reference mean ocean dissolved O2 [mmol m^-3]
R_O2_TO_C: float = 138.0 / 106.0       # Anderson & Sarmiento (1994) photosynthetic/remin O2:C ratio [-]
O2_SUBOXIC_MMOL: float = 20.0          # OMZ hypoxia threshold [mmol O2 m^-3] (Paulmier & Ruiz-Pino, 2009)
O2_HALF_SAT_MMOL: float = 5.0          # Monod half-saturation for aerobic remineralization [mmol m^-3]
R_DENIT_N_PER_O2: float = 0.8          # Stoichiometric NO3 consumed per O2-equivalent in denitrification [-]
N2FIX_MAX_MMOL_M3_D: float = 0.012     # Warm-oligotrophic diazotroph N2 fixation rate (~120 Tg N/yr globally; Monteiro et al., 2011)
N2FIX_KN_INHIB: float = 1.5            # NO3 half-inhibition for N2 fixation [mmol N m^-3]


def compute_o2_saturation_mmol_m3(temp_k: torch.Tensor, sal_psu: torch.Tensor) -> torch.Tensor:
    """Exact seawater O2 saturation concentration [mmol m^-3] (Garcia & Gordon, 1992, L&O 37, Table 1)."""
    tc = torch.clamp(temp_k - 273.15, min=-2.0, max=40.0)
    ts = torch.log((298.15 - tc) / (273.15 + tc))
    s = torch.clamp(sal_psu, min=0.0, max=45.0)
    ln_c = (
        5.80871 + 3.20291 * ts + 4.17887 * (ts ** 2) + 5.10006 * (ts ** 3)
        - 9.86643e-2 * (ts ** 4) + 3.80369 * (ts ** 5)
        + s * (-7.01577e-3 - 7.70028e-3 * ts - 1.13864e-2 * (ts ** 2) - 9.51519e-3 * (ts ** 3))
        - 2.75915e-7 * (s ** 2)
    )
    # umol kg^-1 * (rho_sw / 1000) -> mmol m^-3
    return torch.clamp(torch.exp(torch.clamp(ln_c, min=4.0, max=6.5)) * (C.RHO_SEAWATER * 1.0e-3), min=120.0, max=450.0)


def _step_ocean_o2_omz_column(
    bgc: OceanBGCState,
    ogcm,
    u10_ocn: torch.Tensor,
    f_seaice: torch.Tensor,
    h0_new: torch.Tensor,
    wet_cf: torch.Tensor,
    dt: float,
) -> None:
    """Advance 3D dissolved O2 air-sea exchange, O2 photosynthesis/remineralization, & OMZ denitrification."""
    assert bgc.o2 is not None and bgc.o2_comp is not None
    t_c = torch.clamp(ogcm.temp - 273.15, min=-2.0, max=40.0)
    wet0 = ogcm.wet_c[0].to(bgc.o2.dtype)

    # 1. Surface air-sea O2 gas exchange (Wanninkhof, 2014 Sc_O2 + exact exponential relaxation)
    o2_sat = compute_o2_saturation_mmol_m3(ogcm.temp[0], ogcm.sal[0])
    sc_o2 = torch.clamp(1920.4 - 135.6 * t_c[0] + 5.2122 * (t_c[0] ** 2) - 0.10939 * (t_c[0] ** 3) + 0.00093777 * (t_c[0] ** 4), min=150.0, max=2500.0)
    # k_w [m s^-1] = 0.251 [cm/h / (m/s)^2] * (0.01/3600) * u10^2 * sqrt(660/Sc)
    kw_o2 = (0.251 * (0.01 / 3600.0)) * (torch.clamp(u10_ocn, min=0.0, max=40.0) ** 2) * torch.sqrt(660.0 / sc_o2) * (1.0 - torch.clamp(f_seaice, 0.0, 1.0)) * wet0
    h0_safe = torch.clamp(h0_new, min=10.0)
    k_ex = kw_o2 / h0_safe
    f_ex = -torch.expm1(-torch.clamp(k_ex * float(dt), min=0.0, max=20.0))
    do2_sfc = (o2_sat - bgc.o2[0]) * f_ex * wet0
    # Air-sea O2 flux [mol O2 m^-2 s^-1] (mmol m^-3 * m * 1e-3 / s)
    bgc.fgo2 = (do2_sfc * h0_safe * 1.0e-3) / max(float(dt), 1.0)

    # 2. 3D biological O2 production (NPP) & aerobic/suboxic remineralization + OMZ denitrification
    o2_cur = torch.clamp(bgc.o2, min=0.0)
    o2_prod = (R_O2_TO_C * torch.clamp(bgc.npp_c, min=0.0)) * float(dt) * wet_cf

    # Temperature-dependent detrital remineralization rate [s^-1] matching FlexPFT (0.06 d^-1 at 20 C, Q10 ~ 2)
    f_t_rem = torch.exp(torch.clamp(0.0693 * (t_c - 20.0), min=-2.0, max=2.0))
    rem_c = (0.06 / C.SECONDS_PER_DAY) * f_t_rem * torch.clamp(bgc.det_c, min=0.0) * wet_cf  # mmol C m^-3 s^-1
    o2_dem_rate = R_O2_TO_C * rem_c  # mmol O2 m^-3 s^-1

    # Smooth Monod transition from aerobic respiration to anaerobic denitrification below ~20 mmol m^-3
    f_ox = o2_cur / (o2_cur + O2_HALF_SAT_MMOL)
    # Exact bounded exponential O2 consumption (strictly O2 >= 0)
    k_o2_loss = (o2_dem_rate * f_ox) / torch.clamp(o2_cur, min=1.0e-4)
    do2_cons = o2_cur * (-torch.expm1(-torch.clamp(k_o2_loss * float(dt), min=0.0, max=20.0)))

    do2_tot = o2_prod - do2_cons
    do2_tot[0] = do2_tot[0] + do2_sfc
    o2_new, o2_comp_new = _kahan_add(bgc.o2, do2_tot * wet_cf, bgc.o2_comp)
    bgc.o2 = torch.clamp(o2_new, min=0.0, max=500.0) * wet_cf
    bgc.o2_comp = o2_comp_new * wet_cf

    # Anaerobic denitrification in suboxic OMZs (O2 < 20 mmol m^-3): consumes NO3-, produces N2O!
    omz_gate = torch.clamp((O2_SUBOXIC_MMOL - bgc.o2) / O2_SUBOXIC_MMOL, 0.0, 1.0) * wet_cf
    no3_pos = torch.clamp(bgc.no3, min=0.0)
    denit_rate = R_DENIT_N_PER_O2 * o2_dem_rate * (1.0 - f_ox) * omz_gate  # mmol N m^-3 s^-1
    k_denit = denit_rate / torch.clamp(no3_pos, min=1.0e-4)
    dno3_denit = no3_pos * (-torch.expm1(-torch.clamp(k_denit * float(dt), min=0.0, max=10.0)))
    no3_new, no3_comp_new = _kahan_add(bgc.no3, -dno3_denit * wet_cf, bgc.no3_comp)
    bgc.no3 = torch.clamp(no3_new, min=0.0) * wet_cf
    bgc.no3_comp = no3_comp_new * wet_cf

    h_eff = ogcm.h_c.clone()
    h_eff[0] = h0_safe * wet0
    # Column-integrated marine denitrification [mol N m^-2 s^-1]
    bgc.denit_moln_m2s = ((dno3_denit * h_eff).sum(dim=0) * 1.0e-3) / max(float(dt), 1.0)
    h_tot = torch.clamp((h_eff * wet_cf).sum(dim=0), min=1.0)
    bgc.omz_col_frac = ((bgc.o2 < O2_SUBOXIC_MMOL).to(bgc.o2.dtype) * h_eff * wet_cf).sum(dim=0) / h_tot

    # 3. Surface Marine Diazotrophic N2 Fixation (Monteiro et al., 2011; Landolfi et al., 2017):
    #    Warm (>20 C), low-NO3, Fe-modulated surface N2 fixation balancing OMZ denitrification
    t0_f = t_c[0]
    f_t_fix = torch.clamp((t0_f - 18.0) / 10.0, 0.0, 1.0) ** 2
    f_n_inhib = N2FIX_KN_INHIB / (N2FIX_KN_INHIB + torch.clamp(bgc.no3[0], min=0.0))
    f_fe_fix = (
        torch.clamp(bgc.fe_sfc, min=0.0) / (torch.clamp(bgc.fe_sfc, min=0.0) + 0.20)
        if bgc.fe_sfc is not None
        else torch.full_like(t0_f, 0.65)
    )
    f_ice_open = 1.0 - torch.clamp(f_seaice, 0.0, 1.0)
    dno3_fix = (N2FIX_MAX_MMOL_M3_D * (float(dt) / C.SECONDS_PER_DAY)) * f_t_fix * f_n_inhib * f_fe_fix * f_ice_open * wet0
    no3_0_f, no3_0_fc = _kahan_add(bgc.no3[0], dno3_fix, bgc.no3_comp[0])
    bgc.no3[0] = no3_0_f * wet0
    bgc.no3_comp[0] = no3_0_fc * wet0
    # N2 -> NO3- nitrification decreases alkalinity by 1 eq / mol NO3-
    alk_0_f, alk_0_fc = _kahan_add(bgc.alk[0], -dno3_fix, bgc.alk_comp[0])
    bgc.alk[0] = alk_0_f * wet0
    bgc.alk_comp[0] = alk_0_fc * wet0
    bgc.n2fix_moln_m2s = (dno3_fix * h0_safe * 1.0e-3) / max(float(dt), 1.0)


def apply_atmospheric_bgc_deposition(
    bgc: OceanBGCState,
    ogcm,
    dust_dep_kgm2s: torch.Tensor | None,
    lnox_dep_kgn_m2s: torch.Tensor | None,
    alk_weath_mol_m2s: torch.Tensor | None,
    dt: float,
    *,
    bc_dep_kgm2s: torch.Tensor | None = None,
    fire_n_dep_kgn_m2s: torch.Tensor | None = None,
    river_n_kgn_m2s: torch.Tensor | None = None,
    river_c_kgc_m2s: torch.Tensor | None = None,
    river_fe_mol_m2s: torch.Tensor | None = None,
) -> None:
    """Couple atmospheric (dust + pyrogenic BC Fe, lightning + wildfire N) and riverine (N, C, Alk, Fe)
    inputs into surface ocean FlexPFT BGC via Kahan compensated summation.

    References: Jickells et al. (2005, Science), Moore et al. (2004, GBC), Ito et al. (2019, Sci. Adv.),
    Seitzinger et al. (2002), Regnier et al. (2013), Raiswell & Canfield (2012).
    """
    wet0 = ogcm.wet_c[0].to(bgc.no3.dtype)
    h0 = torch.clamp(ogcm.h_c[0], min=10.0)
    if bgc.fe_sfc is None:
        bgc.fe_sfc = torch.where(ogcm.wet_c[0], torch.full_like(h0, 0.35), torch.zeros_like(h0))
    # 1. Surface dissolved Fe [nM = umol m^-3]: mineral dust + pyrogenic BC soluble Fe + coastal river Fe
    fe_bg = 0.25
    tau_fe = 90.0 * C.SECONDS_PER_DAY
    k_fe = 1.0 / tau_fe
    fe_in_nm_s = torch.zeros_like(h0)
    if dust_dep_kgm2s is not None:
        # 1 kg dust -> 0.035 * 0.02 = 7e-4 kg Fe = 1.253e4 umol Fe -> / h0 [m] = nM
        fe_in_nm_s = fe_in_nm_s + (torch.clamp(dust_dep_kgm2s, min=0.0) * 1.253e4) / h0
    if bc_dep_kgm2s is not None:
        # Pyrogenic wildfire aerosol soluble Fe (~4% Fe/BC mass * 15% solubility -> 1.07e5 umol Fe / kg BC; Ito et al., 2019)
        fe_in_nm_s = fe_in_nm_s + (torch.clamp(bc_dep_kgm2s, min=0.0) * 1.07e5) / h0
    if river_fe_mol_m2s is not None:
        # Coastal riverine bioavailable Fe [mol Fe m^-2 s^-1] -> * 1e6 [umol] / h0 [m] = nM s^-1
        fe_in_nm_s = fe_in_nm_s + (torch.clamp(river_fe_mol_m2s, min=0.0) * 1.0e6) / h0
    fe_eq = fe_bg + fe_in_nm_s * tau_fe
    fe_new = bgc.fe_sfc + (fe_eq - bgc.fe_sfc) * (-math.expm1(-k_fe * float(dt)))
    bgc.fe_sfc = torch.clamp(fe_new, min=0.05, max=6.0) * wet0

    # 2. Atmospheric (LNOx + wildfire) & riverine N delivery -> surface NO3 [mmol N m^-3] via Kahan summation
    n_tot_kgn = None
    for n_src in (lnox_dep_kgn_m2s, fire_n_dep_kgn_m2s, river_n_kgn_m2s):
        if n_src is not None:
            v_pos = torch.clamp(n_src, min=0.0)
            n_tot_kgn = v_pos if n_tot_kgn is None else (n_tot_kgn + v_pos)
    if n_tot_kgn is not None:
        dno3 = ((n_tot_kgn * float(dt) / 0.014007) * 1.0e3 / h0) * wet0
        bgc.no3[0], bgc.no3_comp[0] = _kahan_add(bgc.no3[0], dno3, bgc.no3_comp[0])

    # 3. Riverine fluvial carbon (DIC+DOC) delivery -> surface DIC [mmol C m^-3] via Kahan summation
    if river_c_kgc_m2s is not None:
        ddic = ((torch.clamp(river_c_kgc_m2s, min=0.0) * float(dt) / 0.012011) * 1.0e3 / h0) * wet0
        bgc.dic[0], bgc.dic_comp[0] = _kahan_add(bgc.dic[0], ddic, bgc.dic_comp[0])

    # 4. Riverine / silicate-weathering alkalinity -> surface Alk [mmol eq m^-3] via Kahan summation
    if alk_weath_mol_m2s is not None:
        dalk = ((torch.clamp(alk_weath_mol_m2s, min=0.0) * float(dt)) * 1.0e3 / h0) * wet0
        bgc.alk[0], bgc.alk_comp[0] = _kahan_add(bgc.alk[0], dalk, bgc.alk_comp[0])


def init_ocean_bgc_state(
    wet_c: torch.Tensor,
    h_c: torch.Tensor,
    z_c: torch.Tensor,
    area_cell: torch.Tensor,
) -> OceanBGCState:
    """Initialize map-free 3D FlexPFT BGC state with analytic nutricline and carbonate profiles."""
    dev = h_c.device
    dtype = h_c.dtype
    nzo, nc = h_c.shape
    wet_f = wet_c.to(dtype)
    vol = h_c * area_cell.to(device=dev, dtype=dtype).unsqueeze(0)
    v_sum = torch.clamp(vol.sum(), min=1.0)

    # Smooth map-free vertical profiles normalized to WOA/GLODAP global means (no geographic maps)
    z_2d = z_c.to(device=dev, dtype=dtype).unsqueeze(1)
    nut_shape = 1.0 - torch.exp(-z_2d / 400.0)

    no3_raw = 0.5 + 33.5 * nut_shape
    no3_shift = NO3_GLOBAL_MEAN - (vol * no3_raw).sum() / v_sum
    no3 = torch.where(wet_c, torch.clamp(no3_raw + no3_shift, min=0.2), torch.zeros_like(no3_raw))

    dic_raw = 2020.0 + 265.0 * nut_shape
    dic_shift = DIC_GLOBAL_MEAN - (vol * dic_raw).sum() / v_sum
    dic = torch.where(wet_c, dic_raw + dic_shift, torch.zeros_like(dic_raw))

    alk_raw = 2305.0 + 75.0 * nut_shape
    alk_shift = ALK_GLOBAL_MEAN - (vol * alk_raw).sum() / v_sum
    alk = torch.where(wet_c, alk_raw + alk_shift, torch.zeros_like(alk_raw))

    # Small uniform euphotic seed biomass so FlexPFT blooms autonomously where light & nutrients allow
    euph = torch.exp(-z_2d / 120.0)
    phy = (0.08 * euph) * wet_f
    zoo = (0.04 * euph) * wet_f
    det_n = (0.02 * euph) * wet_f
    det_c = (R_CN_ZOO * 0.02 * euph) * wet_f

    z3 = torch.zeros((nzo, nc), device=dev, dtype=dtype)
    z2 = torch.zeros(nc, device=dev, dtype=dtype)
    return OceanBGCState(
        no3=no3,
        phy=phy,
        zoo=zoo,
        det_n=det_n,
        det_c=det_c,
        dic=dic,
        alk=alk,
        no3_comp=z3.clone(),
        dic_comp=z3.clone(),
        alk_comp=z3.clone(),
        chl=z3.clone(),
        npp_c=z3.clone(),
        c_to_n_phy=torch.full((nzo, nc), R_CN_ZOO, device=dev, dtype=dtype) * wet_f,
        fgco2=z2.clone(),
        spco2_uatm=torch.full((nc,), 284.3, device=dev, dtype=dtype) * wet_f[0],
        ph_sfc=torch.full((nc,), 8.15, device=dev, dtype=dtype) * wet_f[0],
        intpp=z2.clone(),
        epc100=z2.clone(),
    )


def _lambert_w0_exp(y: torch.Tensor) -> torch.Tensor:
    """Overflow-free float32 principal Lambert branch `w = W0(exp(y))` solving `w + ln(w) = y`.

    Uses Iacono & Boyd (2017) initial guess followed by 3 Newton/Halley updates on
    `g(w) = w + ln(w) - y = 0`. Never evaluates `exp(y)` for `y > 1`, preventing float32
    overflow at arbitrarily high irradiance while achieving ~1e-7 relative precision.
    """
    y_hi = torch.clamp(y, min=1.0001)
    ly = torch.log(y_hi)
    w_hi = y_hi - ly + ly / y_hi
    ey_lo = torch.exp(torch.clamp(y, min=-30.0, max=1.0))
    w_lo = ey_lo / (1.0 + ey_lo)
    w = torch.where(y > 1.0, w_hi, w_lo)
    w = torch.clamp(w, min=1.0e-12)
    for _ in range(3):
        w = torch.clamp(w * (1.0 + y - torch.log(w)) / (1.0 + w), min=1.0e-12)
    return w


def compute_flexpft_rates(
    no3: torch.Tensor,
    par_wm2: torch.Tensor,
    temp_k: torch.Tensor,
    wet_cf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FlexPFT optimal growth rate `mu_fl` [d^-1], quota `Q`, `theta_hat`, `f_v`, `f_temp`.

    Implements the instantaneous acclimation (IA) optimality theory of Pahlow, Dietze, &
    Oschlies (2013, MEPS, 473), Smith et al. (2016, J. Plankton Res., 38), and Sasai,
    Smith et al. (2022, Biogeosciences, 19, Eqs. 2-7 & Table 1).
    """
    t_safe = torch.clamp(temp_k, min=270.0, max=315.0)
    # Arrhenius temperature scaling F(T) (Sasai et al., 2022, Eq. 3, Q10 ~ 2.0)
    f_temp = torch.exp(-(FLEXPFT_EA_J_MOL / R_GAS_J_MOL_K) * (1.0 / t_safe - 1.0 / FLEXPFT_T_REF_K))
    v0 = FLEXPFT_MU_MAX_D * f_temp  # [d^-1]

    # Optimal chloroplast Chl:C ratio theta_hat(I, T) (Pahlow et al., 2013; Smith et al., 2016)
    # Threshold irradiance I_0 = zeta_Chl * R_M^Chl / alpha_I below which Chl cost exceeds gain.
    i_pos = torch.clamp(par_wm2, min=0.0)
    i_thresh = (FLEXPFT_ZETA_CHL * FLEXPFT_RM_CHL_D) / FLEXPFT_ALPHA_I
    above_i0 = i_pos > i_thresh
    i_safe = torch.clamp(i_pos, min=i_thresh + 1.0e-6)
    x_light = (FLEXPFT_ALPHA_I * i_safe) / (FLEXPFT_ZETA_CHL * v0)
    r_maint = (FLEXPFT_RM_CHL_D * f_temp) / v0  # = R_M^Chl / mu_max (temperature-coherent)
    y_arg = torch.log(1.0 + r_maint) + 1.0 + x_light
    w_val = _lambert_w0_exp(y_arg)
    theta_raw = (1.0 / FLEXPFT_ZETA_CHL) + (v0 / (FLEXPFT_ALPHA_I * i_safe)) * (1.0 - w_val)
    theta_hat = torch.where(
        above_i0,
        torch.clamp(theta_raw, min=0.0, max=FLEXPFT_THETA_MAX),
        torch.zeros_like(i_pos),
    ) * wet_cf

    # Light limitation S(I, T) = 1 - exp(-alpha * theta_hat * I / (mu_max * F(T))) (Eq. 2)
    # Using -expm1 for float32 accuracy as I -> 0.
    arg_s = torch.clamp((FLEXPFT_ALPHA_I * theta_hat * i_pos) / v0, min=0.0, max=40.0)
    s_light = -torch.expm1(-arg_s)
    v_light = v0 * s_light  # Light-limited carbon fixation rate [d^-1]

    # Potential nutrient uptake rate V_hat^N(N, T) (Pahlow, 2005; Sasai et al., 2022, Eq. 6)
    n_pos = torch.clamp(no3, min=0.0)
    v0_n = FLEXPFT_V0_HAT_D * f_temp
    a0_n = FLEXPFT_A0_HAT * f_temp
    ratio_va = v0_n / torch.clamp(a0_n, min=1.0e-6)  # = V0 / A0 [mmol N m^-3]
    v_hat_n = (v0_n * n_pos) / (n_pos + ratio_va + 2.0 * torch.sqrt(ratio_va * n_pos + 1.0e-12) + 1.0e-12)

    # Optimal cellular N:C quota Q(N, I, T) (Sasai et al., 2022, Eq. 5) & allocation f_V (Eq. 7)
    # Exact balanced-growth optimality: Q* = Q_s * (1 + sqrt(1 + 1 / (Q_s * (V_L / V_hat^N + zeta^N))))
    v_n_safe = torch.clamp(v_hat_n, min=1.0e-9)
    a_ratio = v_light / v_n_safe
    b_cost = a_ratio + FLEXPFT_ZETA_N
    gamma_q = torch.clamp(FLEXPFT_Q_S * b_cost, min=1.0e-5)
    q_quota = FLEXPFT_Q_S * (1.0 + torch.sqrt(1.0 + 1.0 / gamma_q))
    q_quota = torch.clamp(q_quota, min=FLEXPFT_Q_S * 1.05, max=0.25)

    # Optimal allocation f_V ensuring exact balanced growth mu = f_V * V_hat^N / Q
    v_l_safe = torch.clamp(v_light, min=1.0e-9)
    f_v = (1.0 - FLEXPFT_Q_S / q_quota) / (1.0 + (v_hat_n / v_l_safe) * (FLEXPFT_ZETA_N + 1.0 / q_quota))
    f_v = torch.where((v_light > 1.0e-9) & (n_pos > 1.0e-9), torch.clamp(f_v, 0.0, 0.99), torch.zeros_like(f_v))

    mu_fl = ((f_v * v_hat_n) / q_quota) * wet_cf
    return mu_fl, q_quota, theta_hat, f_v, f_temp


def solve_surface_carbonate_mocsy(
    dic_mmol_m3: torch.Tensor,
    alk_mmol_m3: torch.Tensor,
    temp_k: torch.Tensor,
    sal_psu: torch.Tensor,
    wet0_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve surface seawater carbonate equilibrium (`pCO2 [uatm]`, `pH_T`, `[CO2*] [mol m^-3]`).

    Implements OMIP / mocsy 2.0 equilibrium constants on the total pH scale (Orr & Epitalon,
    2015, GMD, 8, 485-499; Orr et al., 2017, GMD, 10, 2169-2199):
    - `K0`: Weiss (1974) / Weiss & Price (1980) CO2 solubility
    - `K1, K2`: Lueker et al. (2000) carbonic acid dissociation constants
    - `K_B`: Dickson (1990) boric acid dissociation constant + Uppström (1974) total boron
    - `K_W`: Millero (1995) water self-ionization constant

    Uses a scaled variable `x = [H+] * 1e8` (order unity) inside 6 bracketed Newton-Raphson
    iterations so float32 never suffers underflow on `[H+]^2 ~ 10^-16`.
    """
    t_k = torch.clamp(temp_k, min=271.15, max=315.0)
    s = torch.clamp(sal_psu, min=5.0, max=45.0)
    inv_t = 1.0 / t_k
    ln_t = torch.log(t_k)
    sqrt_s = torch.sqrt(s)

    # Convert [mmol m^-3] -> [mol kg^-1] (dividing by 1000 * rho_sw)
    inv_rho_1000 = 1.0 / (1000.0 * C.RHO_SEAWATER)
    ct = torch.clamp(dic_mmol_m3 * inv_rho_1000, min=1.0e-5, max=5.0e-3)
    at = torch.clamp(alk_mmol_m3 * inv_rho_1000, min=1.0e-5, max=5.0e-3)

    # Total boron B_T [mol kg^-1] (Uppström, 1974)
    bt = 4.157e-4 * (s / 35.0)

    # Lueker et al. (2000) pK1, pK2 (total pH scale)
    pk1 = 3633.86 * inv_t - 61.2172 + 9.67770 * ln_t - 0.011555 * s + 1.152e-4 * s * s
    pk2 = 471.78 * inv_t + 25.9290 - 3.16967 * ln_t - 0.01781 * s + 1.122e-4 * s * s

    # Dickson (1990) ln(K_B) (total pH scale)
    ln_kb = (
        (-8966.90 - 2890.53 * sqrt_s - 77.942 * s + 1.728 * s * sqrt_s - 0.0996 * s * s) * inv_t
        + (148.0248 + 137.1942 * sqrt_s + 1.62142 * s)
        + (-24.4344 - 25.085 * sqrt_s - 0.2474 * s) * ln_t
        + 0.053105 * sqrt_s * t_k
    )

    # Millero (1995) ln(K_W)
    ln_kw = (
        148.9652 - 13847.26 * inv_t - 23.6521 * ln_t
        + (-5.977 + 118.67 * inv_t + 1.0495 * ln_t) * sqrt_s
        - 0.01615 * s
    )

    # Scale all constants by 1e8 (K1s = K1 * 1e8, K2s = K2 * 1e8, KBs = KB * 1e8, KWs = KW * 1e16)
    # so x = [H+] * 1e8 is O(1) and all quadratic products stay well inside float32 normal range!
    ln10 = math.log(10.0)
    k1s = torch.exp((8.0 - pk1) * ln10)
    k2s = torch.exp((8.0 - pk2) * ln10)
    kbs = torch.exp(ln_kb + 8.0 * ln10)
    kws = torch.exp(ln_kw + 16.0 * ln10)

    # Bracketed Newton-Raphson for f(x) = TA_calc(x) - AT = 0, x in [1e-3, 1e4] (pH 4..11)
    # Initial guess pH = 8.1 -> x = 0.7943
    x = torch.full_like(ct, 0.794328)
    k12s = k1s * k2s
    for _ in range(7):
        denom_c = x * x + k1s * x + k12s
        inv_dc = 1.0 / torch.clamp(denom_c, min=1.0e-12)
        carb_alk = ct * (k1s * x + 2.0 * k12s) * inv_dc
        bor_alk = bt * kbs / (x + kbs)
        wat_alk = (kws / x - x) * 1.0e-8
        f_val = carb_alk + bor_alk + wat_alk - at

        # Exact analytic derivative df/dx < 0 (monotone decreasing in x = [H+]*1e8)
        dcarb_dx = ct * (k1s * denom_c - (k1s * x + 2.0 * k12s) * (2.0 * x + k1s)) * (inv_dc * inv_dc)
        dbor_dx = -bt * kbs / ((x + kbs) ** 2)
        dwat_dx = (-kws / (x * x) - 1.0) * 1.0e-8
        df_dx = torch.clamp(dcarb_dx + dbor_dx + dwat_dx, max=-1.0e-12)

        step = torch.clamp(f_val / df_dx, min=-0.5 * x, max=0.5 * x)
        x = torch.clamp(x - step, min=1.0e-3, max=1.0e4)

    # Dissolved [CO2*] [mol kg^-1] -> [mol m^-3]
    denom_c = x * x + k1s * x + k12s
    co2_star_mol_kg = ct * (x * x) / torch.clamp(denom_c, min=1.0e-12)
    co2_star_mol_m3 = co2_star_mol_kg * C.RHO_SEAWATER

    # Weiss (1974) CO2 solubility K0 [mol kg^-1 atm^-1] (Orr & Epitalon, 2015)
    t100 = t_k * 0.01
    ln_k0 = (
        -60.2409 + 93.4517 / t100 + 23.3585 * torch.log(t100)
        + s * (0.023517 - 0.023656 * t100 + 0.0047036 * t100 * t100)
    )
    k0_mol_kg_atm = torch.exp(ln_k0)

    pco2_uatm = torch.clamp((co2_star_mol_kg / torch.clamp(k0_mol_kg_atm, min=1.0e-6)) * 1.0e6, 20.0, 4000.0) * wet0_f
    ph_total = (8.0 - torch.log10(x)) * wet0_f
    return pco2_uatm, ph_total, co2_star_mol_m3 * wet0_f


def _bgc_ecosystem_core(
    no3: torch.Tensor,
    phy: torch.Tensor,
    zoo: torch.Tensor,
    det_n: torch.Tensor,
    det_c: torch.Tensor,
    dic: torch.Tensor,
    alk: torch.Tensor,
    no3_comp: torch.Tensor,
    dic_comp: torch.Tensor,
    alk_comp: torch.Tensor,
    temp_k: torch.Tensor,
    sal_psu: torch.Tensor,
    sw_par_sfc: torch.Tensor,
    u10_ocn: torch.Tensor,
    f_seaice: torch.Tensor,
    h_c: torch.Tensor,
    h_c_safe: torch.Tensor,
    h0_new: torch.Tensor,
    wet_c: torch.Tensor,
    wet_cf: torch.Tensor,
    is_deepest_c: torch.Tensor,
    co2_ppmv: float,
    dt: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pure-tensor FlexPFT biological reactions, implicit sinking, and air-sea CO2 exchange."""
    nzo = no3.shape[0]
    dt_day = dt / C.SECONDS_PER_DAY
    inv_dt = 1.0 / max(dt, 1.0)

    # 1. Underwater PAR profile with chlorophyll self-shading (previous Chl estimate from phy)
    #    Use conservative initial Chl estimate for self-shading, then compute exact FlexPFT Chl.
    chl_est = torch.clamp(phy * 1.59, min=0.0, max=30.0) * wet_cf
    kd = (K_PAR_WATER + K_PAR_CHL * chl_est) * wet_cf
    tau_par = kd * h_c
    tau_top = torch.cumsum(tau_par, dim=0) - tau_par
    i_sfc = torch.clamp(sw_par_sfc, min=0.0) * (1.0 - torch.clamp(f_seaice, 0.0, 1.0)) * wet_cf[0]
    i_top = i_sfc.unsqueeze(0) * torch.exp(-torch.clamp(tau_top, max=40.0))
    # Exact layer-mean PAR = I_top * (1 - exp(-tau)) / tau with Taylor expansion for tau -> 0
    att_fac = torch.where(
        tau_par > 1.0e-4,
        -torch.expm1(-torch.clamp(tau_par, max=40.0)) / torch.clamp(tau_par, min=1.0e-4),
        1.0 - 0.5 * tau_par + (1.0 / 6.0) * tau_par * tau_par,
    )
    par_layer = i_top * att_fac * wet_cf

    # 2. FlexPFT Optimal Growth, Quota Q, Photoacclimation theta_hat, & Variable C:N
    mu_fl, q_quota, theta_hat, _, f_temp = compute_flexpft_rates(no3, par_layer, temp_k, wet_cf)
    c_to_n = 1.0 / q_quota  # Variable phytoplankton C:N [mol C (mol N)^-1]

    # 3. Strictly Conservative NPZD + Flexible Carbon Reactions
    p_pos = torch.clamp(phy, min=0.0) * wet_cf
    z_pos = torch.clamp(zoo, min=0.0) * wet_cf
    n_pos = torch.clamp(no3, min=0.0) * wet_cf
    dn_pos = torch.clamp(det_n, min=0.0) * wet_cf
    dc_pos = torch.clamp(det_c, min=0.0) * wet_cf

    # Phytoplankton N uptake (capped to 50% of available NO3 per step for positivity)
    dn_up = torch.minimum(mu_fl * p_pos * dt_day, 0.5 * n_pos) * wet_cf
    dc_up = dn_up * c_to_n  # Flexible primary production [mmol C m^-3]

    # Grazing (Holling Type III) & phytoplankton mortality (capped to available P)
    p2 = p_pos * p_pos
    g_rate = (ZOO_G_MAX_D * f_temp) * (p2 / (p2 + ZOO_K_GRAZE * ZOO_K_GRAZE)) * z_pos
    m_p_rate = (PHY_MORT_LIN_D * f_temp) * p_pos + PHY_MORT_QUAD_D * p2
    tot_p_loss = (g_rate + m_p_rate) * dt_day
    scale_p = torch.where(tot_p_loss > 0.8 * p_pos, (0.8 * p_pos) / torch.clamp(tot_p_loss, min=1.0e-12), torch.ones_like(p_pos))
    dn_graze = (g_rate * dt_day) * scale_p * wet_cf
    dn_pmort = (m_p_rate * dt_day) * scale_p * wet_cf

    # Zooplankton excretion & quadratic mortality (capped to available Z)
    z_excr_r = (ZOO_EXCR_D * f_temp) * z_pos
    z_mort_r = ZOO_MORT_QUAD_D * z_pos * z_pos
    tot_z_loss = (z_excr_r + z_mort_r) * dt_day
    scale_z = torch.where(tot_z_loss > 0.8 * z_pos, (0.8 * z_pos) / torch.clamp(tot_z_loss, min=1.0e-12), torch.ones_like(z_pos))
    dn_zexcr = (z_excr_r * dt_day) * scale_z * wet_cf
    dn_zmort = (z_mort_r * dt_day) * scale_z * wet_cf

    # Detritus remineralization via exact exponential decay (temperature-dependent; same rate for PON & POC)
    f_remin = -torch.expm1(-torch.clamp(DET_REMIN_D * f_temp * dt_day, min=0.0, max=20.0)) * wet_cf
    dn_remin = f_remin * dn_pos
    dc_remin = f_remin * dc_pos

    # Update nitrogen pools (EXACT sum(dno3 + dphy + dzoo + ddet_n) == 0!)
    d_no3 = (-dn_up + dn_zexcr + dn_remin) * wet_cf
    d_phy = (dn_up - dn_graze - dn_pmort) * wet_cf
    d_zoo = (ZOO_ASSIM_EFF * dn_graze - dn_zexcr - dn_zmort) * wet_cf
    d_det_n = (dn_pmort + (1.0 - ZOO_ASSIM_EFF) * dn_graze + dn_zmort - dn_remin) * wet_cf

    # Update carbon & alkalinity pools with exact stoichiometry closure:
    # - Phytoplankton growth fixes dc_up = dn_up * (1/Q)
    # - Phytoplankton mortality transfers dn_pmort * (1/Q) to det_c
    # - Grazing transfers (1 - eff) * dn_graze * (1/Q) to det_c, eff * dn_graze * R_CN_ZOO to zoo,
    #   and respires excess C = eff * dn_graze * (1/Q - R_CN_ZOO) directly to DIC!
    # - Zooplankton excretion respires dn_zexcr * R_CN_ZOO to DIC; mortality transfers dn_zmort * R_CN_ZOO to det_c
    # - Optional CaCO3 rain ratio (export & remineralization balanced via alkalinity: 2 eq / mol CaCO3)
    dc_pmort = dn_pmort * c_to_n
    dc_fecal = (1.0 - ZOO_ASSIM_EFF) * dn_graze * c_to_n
    dc_zoo_resp = ZOO_ASSIM_EFF * dn_graze * (c_to_n - R_CN_ZOO) + dn_zexcr * R_CN_ZOO
    dc_zmort = dn_zmort * R_CN_ZOO

    d_det_c = (dc_pmort + dc_fecal + dc_zmort - dc_remin) * wet_cf
    # Net DIC reaction balances organic production vs respiration + remineralization
    # (plus net CaCO3 precipitation coupled to POC export and dissolution coupled to remineralization)
    dc_caco3_prod = RAIN_RATIO_CACO3 * (dc_pmort + dc_fecal)
    dc_caco3_diss = RAIN_RATIO_CACO3 * dc_remin
    d_dic = (-dc_up + dc_zoo_resp + dc_remin - dc_caco3_prod + dc_caco3_diss) * wet_cf
    # Alkalinity: +1 eq per mol NO3 assimilated, -1 eq per mol NO3 remineralized/excreted, -2 eq per mol CaCO3
    d_alk = (-d_no3 - 2.0 * (dc_caco3_prod - dc_caco3_diss)) * wet_cf

    no3_new, no3_c = _kahan_add(no3, d_no3, no3_comp)
    phy_new = torch.clamp(phy + d_phy, min=0.0) * wet_cf
    zoo_new = torch.clamp(zoo + d_zoo, min=0.0) * wet_cf
    det_n_mid = torch.clamp(det_n + d_det_n, min=0.0) * wet_cf
    det_c_mid = torch.clamp(det_c + d_det_c, min=0.0) * wet_cf
    dic_new, dic_c = _kahan_add(dic, d_dic, dic_comp)
    alk_new, alk_c = _kahan_add(alk, d_alk, alk_comp)

    # 4. Unconditionally Stable Implicit Vertical Detritus Sinking + Benthic Remineralization
    #    Downward upwind implicit sweep from k = 0 -> nzo-1:
    #    (h_k + w_s * dt) * D_k^{n+1} = h_k * D_k^n + w_s * dt * D_{k-1}^{n+1}
    #    Flux reaching seafloor (bottom wet layer) remineralizes immediately into bottom NO3/DIC/Alk
    #    so zero nitrogen, carbon, or alkalinity ever leaks out the seafloor.
    w_sink_ms = DET_W_SINK_M_D / C.SECONDS_PER_DAY
    h_eff = torch.cat([h0_new.unsqueeze(0), h_c_safe[1:]], dim=0)
    w_dt = w_sink_ms * dt

    dn_list: list[torch.Tensor] = []
    dc_list: list[torch.Tensor] = []
    f_in_n = torch.zeros_like(det_n_mid[0])
    f_in_c = torch.zeros_like(det_c_mid[0])
    benth_n_list: list[torch.Tensor] = []
    benth_c_list: list[torch.Tensor] = []
    epc_100 = torch.zeros_like(det_c_mid[0])

    for k in range(nzo):
        wk = wet_cf[k]
        hk = h_eff[k]
        is_bot_k = is_deepest_c[k].to(no3.dtype)
        # Active sinking out of layer k (into k+1 if wet below, else into benthic remineralization)
        denom = hk + w_dt
        dn_k = ((hk * det_n_mid[k] + f_in_n) / denom) * wk
        dc_k = ((hk * det_c_mid[k] + f_in_c) / denom) * wk
        f_out_n = (w_dt * dn_k) * wk
        f_out_c = (w_dt * dc_k) * wk

        # Record sinking POC flux leaving Layer 1 (~100 m depth) for CMIP7 `epc100` [mol C m^-2 s^-1]
        if k == 1:
            epc_100 = (f_out_c * 1.0e-3) * inv_dt

        # If this is the deepest wet cell, remineralize bottom sinking flux directly in-place
        benth_n_list.append((f_out_n / hk) * is_bot_k)
        benth_c_list.append((f_out_c / hk) * is_bot_k)
        f_in_n = f_out_n * (1.0 - is_bot_k)
        f_in_c = f_out_c * (1.0 - is_bot_k)
        dn_list.append(dn_k)
        dc_list.append(dc_k)

    det_n_new = torch.stack(dn_list, dim=0)
    det_c_new = torch.stack(dc_list, dim=0)
    benth_n = torch.stack(benth_n_list, dim=0)
    benth_c = torch.stack(benth_c_list, dim=0)
    no3_new, no3_c = _kahan_add(no3_new, benth_n, no3_c)
    dic_new, dic_c = _kahan_add(dic_new, benth_c, dic_c)
    alk_new, alk_c = _kahan_add(alk_new, -benth_n, alk_c)

    # 5. OMIP / mocsy 2.0 Surface Carbonate Chemistry & Wanninkhof (2014) Air-Sea CO2 Flux
    wet0_f = wet_cf[0]
    spco2_uatm, ph_sfc, co2_star_mol_m3 = solve_surface_carbonate_mocsy(
        dic_new[0], alk_new[0], temp_k[0], sal_psu[0], wet0_f
    )
    # Weiss & Price (1980) combined solubility phi_CO2^0 [mol L^-1 atm^-1] -> *1000 [mol m^-3 atm^-1]
    t0_k = torch.clamp(temp_k[0], min=271.15, max=315.0)
    t0_c = t0_k - 273.15
    s0 = torch.clamp(sal_psu[0], min=5.0, max=45.0)
    t100 = t0_k * 0.01
    ln_phi0 = (
        WEISS_A1 + WEISS_A2 / t100 + WEISS_A3 * torch.log(t100) + WEISS_A4 * t100 * t100
        + s0 * (WEISS_B1 + WEISS_B2 * t100 + WEISS_B3 * t100 * t100)
    )
    phi0_mol_m3_atm = 1000.0 * torch.exp(ln_phi0)
    co2_sat_mol_m3 = phi0_mol_m3_atm * (co2_ppmv * 1.0e-6)  # At Pa = 1 atm

    # Wanninkhof (2014) CO2 Schmidt number & gas-transfer velocity k_w [m s^-1] (Orr et al., 2017, Eq. 11)
    tc_c = torch.clamp(t0_c, min=-2.0, max=40.0)
    sc_co2 = SC_CO2_A + tc_c * (SC_CO2_B + tc_c * (SC_CO2_C + tc_c * (SC_CO2_D + tc_c * SC_CO2_E)))
    open_ocn = 1.0 - torch.clamp(f_seaice, 0.0, 1.0)
    kw_ms = open_ocn * WANNINKHOF_A_MS * (u10_ocn * u10_ocn) * torch.sqrt(660.0 / torch.clamp(sc_co2, min=50.0))

    # Air-sea CO2 flux [mol C m^-2 s^-1], positive INTO ocean (CMIP7 `fgco2` sign convention)
    fgco2 = (kw_ms * (co2_sat_mol_m3 - co2_star_mol_m3)) * wet0_f
    # Convert flux [mol C m^-2 s^-1] -> Layer-0 DIC tendency [mmol C m^-3] (* 1000 * dt / h0_new)
    ddic_gas_0 = (fgco2 * 1000.0 * dt / torch.clamp(h0_new, min=10.0)) * wet0_f
    dic0_g, dic0_gc = _kahan_add(dic_new[0], ddic_gas_0, dic_c[0])
    dic_new = torch.cat([dic0_g.unsqueeze(0), dic_new[1:]], dim=0)
    dic_c = torch.cat([dic0_gc.unsqueeze(0), dic_c[1:]], dim=0)

    # 6. FlexPFT Diagnostics: Chl [mg m^-3] = P * (theta_hat / Q) (Sasai et al., 2022, Sec. 2.2)
    chl_diag = (phy_new * (theta_hat / q_quota)) * wet_cf
    npp_c_diag = (dc_up * inv_dt) * wet_cf  # [mmol C m^-3 s^-1]
    intpp_diag = ((npp_c_diag * h_eff).sum(dim=0) * 1.0e-3) * wet0_f  # [mol C m^-2 s^-1]

    return (
        no3_new, phy_new, zoo_new, det_n_new, det_c_new, dic_new, alk_new,
        no3_c, dic_c, alk_c,
        chl_diag, npp_c_diag, c_to_n * wet_cf, fgco2, spco2_uatm, ph_sfc, intpp_diag, epc_100,
    )


def step_ocean_bgc_flexpft(
    bgc: OceanBGCState,
    ogcm,
    mesh,
    h_u_edge: torch.Tensor,
    sx_int: torch.Tensor,
    w_vel: torch.Tensor,
    h0_new: torch.Tensor,
    dh0_eff: torch.Tensor,
    a_t: torch.Tensor | None,
    b_t: torch.Tensor | None,
    d_t: torch.Tensor | None,
    c_dif: torch.Tensor,
    h_eff: torch.Tensor,
    cache: dict,
    sw_par_sfc: torch.Tensor,
    u10_ocn: torch.Tensor,
    f_seaice: torch.Tensor,
    co2_ppmv: float,
    dt: float,
    enable_full_redi: bool = True,
    enable_o2_omz: bool = False,
    *,
    h_old_z: torch.Tensor | None = None,
    h_new_z: torch.Tensor | None = None,
    w_star: torch.Tensor | None = None,
    enable_fct_zstar: bool = False,
    c0_i32: torch.Tensor | None = None,
    c1_i32: torch.Tensor | None = None,
    eoc_i32: torch.Tensor | None = None,
    active_w_c: torch.Tensor | None = None,
    use_gpu_ogcm: bool = False,
) -> None:
    """Advance 3D FlexPFT marine BGC tracers by `dt` [s]: transport + vertical mixing + FlexPFT + CO2 (+ O2/OMZ)."""
    from aoba.ocean_ogcm import KAPPA_REDI_HORIZ, _ogcm_advect_redi_batch
    from aoba.kernels import dispatch_ogcm_fct_zstar_metal, dispatch_ogcm_multi_vdiff_metal

    dev = bgc.dic.device
    dtype = bgc.dic.dtype
    nzo, nc = bgc.dic.shape
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    eoc = mesh.edges_on_cell
    wet_c = ogcm.wet_c
    wet_cf = cache["wet_cf"]
    wet_ef = cache["wet_ef"]
    wet_int_cf = cache["wet_int_cf"]
    wet_int_ef = cache["wet_int_ef"]
    h_e = ogcm.h_e
    h_c_safe = cache["h_c_safe"]
    dz_c_int = cache["dz_c_int"]
    inv_dc_edge = cache["inv_dc_edge"]
    div_w = cache["div_w"]
    w_vor_edge = cache["w_vor_edge"]
    active_eoc = cache["active_eoc"]
    inv_area_c = cache["inv_area_c"]

    # Lazily allocate 3D dissolved O2 when enable_o2_omz is active (100% map-free vertical profile)
    if enable_o2_omz and bgc.o2 is None:
        z_prof = torch.linspace(260.0, 175.0, nzo, device=dev, dtype=dtype).view(nzo, 1)
        bgc.o2 = z_prof.expand(nzo, nc).contiguous() * wet_cf
        bgc.o2_comp = torch.zeros((nzo, nc), device=dev, dtype=dtype)
        bgc.fgo2 = torch.zeros((nc,), device=dev, dtype=dtype)
        bgc.denit_moln_m2s = torch.zeros((nc,), device=dev, dtype=dtype)
        bgc.omz_col_frac = torch.zeros((nc,), device=dev, dtype=dtype)
        bgc.n2fix_moln_m2s = torch.zeros((nc,), device=dev, dtype=dtype)

    use_o2 = enable_o2_omz and bgc.o2 is not None and bgc.o2_comp is not None
    nt = 8 if use_o2 else 7

    # 1. Conservative 3D Advection + Full Rotated Redi Diffusion
    ref_key = f"bgc_ref_b_{nt}"
    bgc_ref = cache.get(ref_key)
    if bgc_ref is None:
        ref_vals = [NO3_GLOBAL_MEAN, 0.0, 0.0, 0.0, 0.0, DIC_GLOBAL_MEAN, ALK_GLOBAL_MEAN]
        if use_o2:
            ref_vals.append(O2_GLOBAL_MEAN)
        bgc_ref = torch.tensor(ref_vals, device=dev, dtype=dtype).view(nt, 1, 1)
        cache[ref_key] = bgc_ref
    zero_c = cache.get("bgc_zero_c")
    if zero_c is None or zero_c.shape != (nzo, nc):
        zero_c = torch.zeros((nzo, nc), device=dev, dtype=dtype)
        cache["bgc_zero_c"] = zero_c

    q_list = [bgc.no3, bgc.phy, bgc.zoo, bgc.det_n, bgc.det_c, bgc.dic, bgc.alk]
    c_list = [bgc.no3_comp, zero_c, zero_c, zero_c, zero_c, bgc.dic_comp, bgc.alk_comp]
    if use_o2:
        q_list.append(bgc.o2)
        c_list.append(bgc.o2_comp)
    qt = torch.stack(q_list, dim=0)
    ct = torch.stack(c_list, dim=0)

    on_gpu_f32 = get_active_gpu_backend(dev) in ("mps", "cuda") and dtype == torch.float32
    use_gpu_fct = (
        use_gpu_ogcm
        and on_gpu_f32
        and enable_fct_zstar
        and h_old_z is not None
        and h_new_z is not None
        and w_star is not None
        and c0_i32 is not None
        and c1_i32 is not None
        and eoc_i32 is not None
        and active_w_c is not None
    )
    if use_gpu_fct:
        # Process tracers in B=5 chunks when GPU VRAM allows (zero-advective-edge-buffer FCT
        # halves workspace footprint so 7-9 BGC tracers complete in 2 calls instead of 3).
        if dev.type == "cuda":
            _vram_gb = float(torch.cuda.get_device_properties(dev).total_memory) / 1.0e9
            bgc_batch = 5 if (_vram_gb >= 20.0 or nc * nzo <= 3_500_000) else 2
        elif dev.type == "mps":
            _mps_gb = float(torch.mps.recommended_max_working_set_size()) / 1.0e9 if hasattr(torch.mps, "recommended_max_working_set_size") else 0.0
            bgc_batch = 5 if (_mps_gb >= 20.0 or nc * nzo <= 3_500_000) else 2
        else:
            bgc_batch = 5 if (nc * nzo <= 3_500_000) else 2
        znl_k = f"bgc_zero_nl_{bgc_batch}"
        zero_nl_b = cache.get(znl_k)
        if zero_nl_b is None or zero_nl_b.shape != (bgc_batch, nzo - 1, nc):
            zero_nl_b = torch.zeros((bgc_batch, nzo - 1, nc), device=dev, dtype=dtype)
            cache[znl_k] = zero_nl_b
        for i0 in range(0, nt, bgc_batch):
            i1 = min(i0 + bgc_batch, nt)
            bi = i1 - i0
            dispatch_ogcm_fct_zstar_metal(
                q=qt[i0:i1],
                q_comp=ct[i0:i1],
                f_nl_q=zero_nl_b[:bi],
                h_u_edge=h_u_edge,
                sx_int=sx_int,
                w_star=w_star,
                h_e=h_e,
                h_old_z=h_old_z,
                h_new_z=h_new_z,
                dz_c_int=dz_c_int,
                inv_dc_edge=inv_dc_edge,
                wet_cf=wet_cf,
                wet_ef=wet_ef,
                wet_int_cf=wet_int_cf,
                wet_int_ef=wet_int_ef,
                c0_i32=c0_i32,
                c1_i32=c1_i32,
                eoc_i32=eoc_i32,
                div_w=div_w,
                w_vor_edge=w_vor_edge,
                active_w=active_w_c,
                q_ref=bgc_ref[i0:i1],
                dt=dt,
                kappa_redi=float(KAPPA_REDI_HORIZ),
                enable_full_redi=bool(enable_full_redi),
                in_place=True,
            )
    else:
        znl_key = f"bgc_zero_nl_{nt}"
        zero_nl = cache.get(znl_key)
        if zero_nl is None or zero_nl.shape != (nt, nzo - 1, nc):
            zero_nl = torch.zeros((nt, nzo - 1, nc), device=dev, dtype=dtype)
            cache[znl_key] = zero_nl
        adv_b_fn = maybe_cuda_compile(_ogcm_advect_redi_batch, device=dev)
        qt, ct = adv_b_fn(
            qt, ct, zero_nl, h_u_edge, sx_int, w_vel, h_e, h_c_safe, h0_new, dh0_eff,
            dz_c_int, inv_dc_edge, wet_cf, wet_ef, wet_int_cf, wet_int_ef,
            c0, c1, eoc, div_w, w_vor_edge, active_eoc, inv_area_c,
            bgc_ref, dt, enable_full_redi,
        )

    # 2. Strictly Conservative Flux-Increment Implicit Vertical Mixing
    if on_gpu_f32 and 2 <= nzo <= 64:
        # Factor shared tridiagonal ONCE per ocean column across all B tracers (zero temp 3D buffers)
        clamp_mask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | ((1 << 7) if use_o2 else 0)
        qt, ct = dispatch_ogcm_multi_vdiff_metal(
            q=qt, q_comp=ct, c_dif=c_dif, h_eff=h_eff, wet_cf=wet_cf, clamp_nonneg_mask=clamp_mask
        )
        bgc.no3, bgc.no3_comp = qt[0], ct[0]
        bgc.phy = qt[1]
        bgc.zoo = qt[2]
        bgc.det_n = qt[3]
        bgc.det_c = qt[4]
        bgc.dic, bgc.dic_comp = qt[5], ct[5]
        bgc.alk, bgc.alk_comp = qt[6], ct[6]
        if use_o2:
            bgc.o2, bgc.o2_comp = qt[7], ct[7]
    else:
        if a_t is None or b_t is None or d_t is None:
            z_row_1 = torch.zeros((1, nc), device=dev, dtype=dtype)
            a_t = torch.cat([z_row_1, -c_dif / h_eff[1:]], dim=0)
            d_t = torch.cat([-c_dif / h_eff[:-1], z_row_1], dim=0)
            b_t = 1.0 - a_t - d_t
        z_row = torch.zeros((nt, 1, nc), device=dev, dtype=dtype)
        q_diff = qt[:, 1:] - qt[:, :-1]
        c_dif_b = c_dif.unsqueeze(0)
        h_eff_b = h_eff.unsqueeze(0)
        f_exp = c_dif_b * q_diff
        rhs = (torch.cat([f_exp, z_row], dim=1) - torch.cat([z_row, f_exp], dim=1)) / h_eff_b
        rhs_flat = rhs.permute(1, 0, 2).reshape(nzo, nt * nc)
        a_b = a_t.unsqueeze(1).expand(nzo, nt, nc).reshape(nzo, nt * nc)
        b_b = b_t.unsqueeze(1).expand(nzo, nt, nc).reshape(nzo, nt * nc)
        d_b = d_t.unsqueeze(1).expand(nzo, nt, nc).reshape(nzo, nt * nc)
        dq_imp = solve_tridiagonal_batched(a_b, b_b, d_b, rhs_flat).view(nzo, nt, nc).permute(1, 0, 2)
        f_imp = c_dif_b * (q_diff + (dq_imp[:, 1:] - dq_imp[:, :-1]))
        dq_cons = ((torch.cat([f_imp, z_row], dim=1) - torch.cat([z_row, f_imp], dim=1)) / h_eff_b) * wet_cf.unsqueeze(0)

        qt_k, ct_k = _kahan_add(qt, dq_cons, ct)
        bgc.no3, bgc.no3_comp = qt_k[0], ct_k[0]
        bgc.phy = torch.clamp(qt[1] + dq_cons[1], min=0.0) * wet_cf
        bgc.zoo = torch.clamp(qt[2] + dq_cons[2], min=0.0) * wet_cf
        bgc.det_n = torch.clamp(qt[3] + dq_cons[3], min=0.0) * wet_cf
        bgc.det_c = torch.clamp(qt[4] + dq_cons[4], min=0.0) * wet_cf
        bgc.dic, bgc.dic_comp = qt_k[5], ct_k[5]
        bgc.alk, bgc.alk_comp = qt_k[6], ct_k[6]
        if use_o2:
            bgc.o2, bgc.o2_comp = torch.clamp(qt_k[7], min=0.0) * wet_cf, ct_k[7] * wet_cf

    # 3. FlexPFT Ecosystem Reactions, Implicit Sinking, O2/OMZ+Diazotrophy, & OMIP/mocsy Air-Sea CO2 Exchange
    sw_par_eff = sw_par_sfc
    if bgc.fe_sfc is not None:
        fe_lim = torch.clamp((bgc.fe_sfc / (bgc.fe_sfc + 0.20)) / 0.75, min=0.25, max=1.35)
        sw_par_eff = sw_par_sfc * fe_lim

    if on_gpu_f32:
        (
            bgc.no3, bgc.phy, bgc.zoo, bgc.det_n, bgc.det_c, bgc.dic, bgc.alk,
            bgc.no3_comp, bgc.dic_comp, bgc.alk_comp,
            bgc.chl, bgc.npp_c, bgc.c_to_n_phy, bgc.fgco2, bgc.spco2_uatm, bgc.ph_sfc, bgc.intpp, bgc.epc100,
            o2_out, o2_c_out, fgo2_out, denit_out, omz_out, n2fix_out,
        ) = dispatch_ocean_bgc_column_metal(
            no3=bgc.no3, phy=bgc.phy, zoo=bgc.zoo, det_n=bgc.det_n, det_c=bgc.det_c, dic=bgc.dic, alk=bgc.alk,
            no3_comp=bgc.no3_comp, dic_comp=bgc.dic_comp, alk_comp=bgc.alk_comp,
            temp_k=ogcm.temp, sal_psu=ogcm.sal, h_c=ogcm.h_c, wet_cf=wet_cf,
            h0_new=h0_new, sw_par_sfc=sw_par_eff, u10_ocn=u10_ocn, f_seaice=f_seaice,
            co2_ppmv=float(co2_ppmv), dt=float(dt), rho_sw=float(C.RHO_SEAWATER),
            o2=bgc.o2 if use_o2 else None,
            o2_comp=bgc.o2_comp if use_o2 else None,
            fe_sfc=bgc.fe_sfc,
            in_place=True,
        )
        if use_o2:
            bgc.o2 = o2_out
            bgc.o2_comp = o2_c_out
            bgc.fgo2 = fgo2_out
            bgc.denit_moln_m2s = denit_out
            bgc.omz_col_frac = omz_out
            bgc.n2fix_moln_m2s = n2fix_out
        return

    # Deepest wet cell mask (CPU fallback)
    is_deepest_c = cache.get("is_deepest_c")
    if is_deepest_c is None:
        is_deepest_c = torch.zeros_like(wet_c)
        if wet_c.shape[0] > 1:
            is_deepest_c[:-1] = wet_c[:-1] & (~wet_c[1:])
        is_deepest_c[-1] = wet_c[-1]
        cache["is_deepest_c"] = is_deepest_c

    eco_fn = maybe_cuda_compile(_bgc_ecosystem_core, device=dev)
    (
        bgc.no3, bgc.phy, bgc.zoo, bgc.det_n, bgc.det_c, bgc.dic, bgc.alk,
        bgc.no3_comp, bgc.dic_comp, bgc.alk_comp,
        bgc.chl, bgc.npp_c, bgc.c_to_n_phy, bgc.fgco2, bgc.spco2_uatm, bgc.ph_sfc, bgc.intpp, bgc.epc100,
    ) = eco_fn(
        bgc.no3, bgc.phy, bgc.zoo, bgc.det_n, bgc.det_c, bgc.dic, bgc.alk,
        bgc.no3_comp, bgc.dic_comp, bgc.alk_comp,
        ogcm.temp, ogcm.sal, sw_par_eff, u10_ocn, f_seaice,
        ogcm.h_c, h_c_safe, h0_new, wet_c, wet_cf, is_deepest_c,
        float(co2_ppmv), float(dt),
    )
    if use_o2:
        _step_ocean_o2_omz_column(bgc, ogcm, u10_ocn, f_seaice, h0_new, wet_cf, dt)


def diagnose_ocean_bgc_metrics(bgc: OceanBGCState, ogcm, mesh) -> dict[str, float]:
    """Compute float64-accumulated global marine BGC metrics (NPP, CO2 uptake, Chl, pH, pCO2, inventories)."""
    area = mesh.area_cell.detach().cpu().to(torch.float64)
    wet0 = ogcm.wet_c[0].detach().cpu()
    area_ocn = torch.where(wet0, area, torch.zeros_like(area))
    a_ocn = float(area_ocn.sum().item())
    if a_ocn <= 0.0:
        return {}

    h_eff = ogcm.h_c.detach().cpu().to(torch.float64).clone()
    h_eff[0] = torch.where(wet0, torch.clamp(h_eff[0] + ogcm.eta.detach().cpu().to(torch.float64), min=10.0), torch.zeros_like(area))
    vol = h_eff * area.unsqueeze(0)
    v_tot = float(vol.sum().item())

    no3_64 = bgc.no3.detach().cpu().to(torch.float64) + bgc.no3_comp.detach().cpu().to(torch.float64)
    phy_64 = bgc.phy.detach().cpu().to(torch.float64)
    zoo_64 = bgc.zoo.detach().cpu().to(torch.float64)
    det_n64 = bgc.det_n.detach().cpu().to(torch.float64)
    det_c64 = bgc.det_c.detach().cpu().to(torch.float64)
    dic_64 = bgc.dic.detach().cpu().to(torch.float64) + bgc.dic_comp.detach().cpu().to(torch.float64)
    alk_64 = bgc.alk.detach().cpu().to(torch.float64) + bgc.alk_comp.detach().cpu().to(torch.float64)

    # Convert [mol C s^-1] -> [Pg C yr^-1] (12.011 g/mol * 1e-15 Pg/g * seconds/yr)
    mol_s_to_pgc_yr = 12.011 * 1.0e-15 * C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    intpp_pgc_yr = float((area_ocn * bgc.intpp.detach().cpu().to(torch.float64)).sum().item() * mol_s_to_pgc_yr)
    fgco2_pgc_yr = float((area_ocn * bgc.fgco2.detach().cpu().to(torch.float64)).sum().item() * mol_s_to_pgc_yr)
    epc100_pgc_yr = float((area_ocn * bgc.epc100.detach().cpu().to(torch.float64)).sum().item() * mol_s_to_pgc_yr)

    chl_sfc = float((area_ocn * bgc.chl[0].detach().cpu().to(torch.float64)).sum().item() / a_ocn)
    spco2 = float((area_ocn * bgc.spco2_uatm.detach().cpu().to(torch.float64)).sum().item() / a_ocn)
    ph_sfc = float((area_ocn * bgc.ph_sfc.detach().cpu().to(torch.float64)).sum().item() / a_ocn)

    tot_n_mmol = float((vol * (no3_64 + phy_64 + zoo_64 + det_n64)).sum().item())
    tot_dic_pgc = float((vol * dic_64).sum().item() * 12.011e-18)
    tot_alk_meq = float((vol * alk_64).sum().item())

    out = {
        "ocean_npp_pgc_yr": intpp_pgc_yr,
        "ocean_fgco2_pgc_yr": fgco2_pgc_yr,
        "ocean_epc100_pgc_yr": epc100_pgc_yr,
        "ocean_chl_sfc_mg_m3": chl_sfc,
        "ocean_spco2_uatm": spco2,
        "ocean_ph_sfc": ph_sfc,
        "ocean_no3_global_mmol_m3": float((vol * no3_64).sum().item() / max(v_tot, 1.0)),
        "ocean_dic_global_mmol_m3": float((vol * dic_64).sum().item() / max(v_tot, 1.0)),
        "ocean_alk_global_mmol_m3": float((vol * alk_64).sum().item() / max(v_tot, 1.0)),
        "ocean_tot_n_mmol": tot_n_mmol,
        "ocean_tot_dic_pgc": tot_dic_pgc,
        "ocean_tot_alk_meq": tot_alk_meq,
    }
    if bgc.o2 is not None and bgc.o2_comp is not None:
        o2_64 = bgc.o2.detach().cpu().to(torch.float64) + bgc.o2_comp.detach().cpu().to(torch.float64)
        out["ocean_o2_global_mmol_m3"] = float((vol * o2_64).sum().item() / max(v_tot, 1.0))
        omz_vol = float((vol * (o2_64 < O2_SUBOXIC_MMOL).to(torch.float64)).sum().item())
        out["ocean_omz_vol_pct"] = 100.0 * omz_vol / max(v_tot, 1.0)
        if bgc.denit_moln_m2s is not None:
            den64 = bgc.denit_moln_m2s.detach().cpu().to(torch.float64)
            out["ocean_denit_tgn_yr"] = float((area_ocn * den64).sum().item() * 0.014007 * C.DAYS_PER_YEAR * C.SECONDS_PER_DAY * 1.0e-9)
    return out
