"""Map-Free Terrestrial Carbon Cycle, Vertically Resolved Soil/Permafrost C, & Autonomous Wildfire.

Extends AOBA's Eco-Evolutionary Optimality (EEO) terrestrial biosphere (`biosphere.py`)
into a fully closed land carbon cycle without prescribed PFT or biome maps (`enable_land_carbon = false`
by default):

1. **Dynamic Vegetation & Litter Pools + Autotrophic/Heterotrophic Respiration** (Clark et al.,
   2011, *Geosci. Model Dev.*, 4, 701-722, TRIFFID/JULES):
   - Prognostic `c_leaf`, `c_wood`, `c_root`, `c_lit_fast` (metabolic), `c_lit_slow` (structural).
   - Map-free allocation driven directly by emergent EEO canopy height `h_canopy` (short
     grasslands allocate to leaf/root; tall forests allocate strongly to wood).
2. **Vertically Resolved 5-Layer Soil Organic Carbon & Permafrost Stabilization** (Koven et al.,
   2013, *Biogeosciences*, 10, 7109-7131, CLM4.5bgc):
   - Matches AOBA's 5 soil thermal/hydrology layers (`0.05, 0.15, 0.40, 0.80, 1.60 m`) with
     depth-attenuated intrinsic decay, Q10 temperature response, moisture limitation, frozen-soil
     permafrost suppression when `T_soil < 273.15 K`, and conservative vertical cryoturbation/bioturbation.
3. **Process-Based Autonomous Wildfire & Pyrogenic Emissions** (Thonicke et al., 2010,
   *Biogeosciences*, 7, 1991-2011, SPITFIRE):
   - Prognostic Nesterov dryness index, fuel moisture / fire danger index (FDI), convective
     lightning ignition proxy, wind-assisted spread, combustion of litter & live wood/foliage,
     and pyrogenic CO2, black carbon (BC), and organic carbon (OC) aerosol emission fluxes
     (coupling directly to `aerosols.py`).

Strictly conserves total land carbon:
    d(C_leaf + C_wood + C_root + C_lit_fast + C_lit_slow + sum_k C_soil_k) / dt
        == GPP - R_a - R_h - F_fire == -NBP
using Neumaier/Kahan float32 compensated summation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.surface import N_SOIL_LAYERS, SOIL_LAYER_DEPTHS, SURF_LAND, _kahan_add

# Unit conversion: 1 umol CO2 m^-2 s^-1 -> kg C m^-2 s^-1 (12.011e-9 kg C / umol)
UMOL_TO_KG_C: float = 12.011e-9
SEC_PER_YEAR: float = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY

# Pool turnover times & respiration parameters (Clark et al., 2011, Table 4)
TAU_LEAF_YR: float = 1.5           # Leaf turnover timescale [yr]
TAU_ROOT_YR: float = 2.0           # Fine-root turnover timescale [yr]
TAU_WOOD_YR: float = 40.0          # Woody stem/coarse-root turnover timescale [yr]
TAU_LIT_FAST_YR: float = 0.7       # Fast metabolic litter decay timescale at 25 C [yr]
TAU_LIT_SLOW_YR: float = 4.0       # Structural woody litter decay timescale at 25 C [yr]
TAU_SOIL_BASE_YR: float = 35.0     # Base surface soil organic C turnover timescale at 25 C [yr]
RESP_GROWTH_FRAC: float = 0.25     # Growth respiration fraction of (GPP - R_maint) [-]
R_MAINT_WOOD_YR: float = 0.012     # Woody sapwood/stem maintenance respiration rate at 25 C [yr^-1]
R_MAINT_ROOT_YR: float = 0.25      # Fine-root maintenance respiration rate at 25 C [yr^-1]
HUMIF_FRAC_FAST: float = 0.30      # Fraction of decomposed fast litter entering soil C (rest -> R_h) [-]
HUMIF_FRAC_SLOW: float = 0.45      # Fraction of decomposed slow litter entering soil C (rest -> R_h) [-]

# Vertically resolved soil C & permafrost parameters (Koven et al., 2013, Sec. 2.2)
SOIL_Z_TAU_M: float = 0.50         # e-folding depth for intrinsic soil decay rate attenuation [m]
SOIL_Q10: float = 2.0              # Heterotrophic respiration Q10 [-]
SOIL_FROZEN_SUPPRESS: float = 0.05 # Decomposition multiplier when fully frozen (permafrost lock) [-]
SOIL_CRYO_DIFF_M2_YR: float = 5.0e-4  # Vertical bioturbation / cryoturbation diffusivity [m^2 yr^-1]

# Autonomous SPITFIRE wildfire parameters (Thonicke et al., 2010)
FIRE_DI_MAX: float = 5000.0        # Nesterov dryness index saturation cap [K^2]
FIRE_DI_K: float = 6.0e-4          # Fire danger index FDI = 1 - exp(-k * DI) [K^-2]
FIRE_FUEL_MIN_KGC: float = 0.15    # Minimum litter fuel load required to sustain fire [kg C m^-2]
FIRE_MAX_RATE_YR: float = 0.35     # Maximum annual fractional burned area under extreme drought [yr^-1]
FIRE_COMBUST_LIT: float = 0.70     # Litter combustion completeness [-]
FIRE_COMBUST_LEAF: float = 0.50    # Canopy leaf combustion completeness during fire [-]
FIRE_COMBUST_WOOD: float = 0.15    # Woody stem combustion completeness (crown/surface fire) [-]
FIRE_EF_BC: float = 0.0012         # Black carbon emission factor [kg BC / kg C burned] (Andreae & Merlet, 2001)
FIRE_EF_OC: float = 0.0110         # Organic carbon aerosol emission factor [kg OC / kg C burned]


@dataclass
class LandCarbonState:
    """Prognostic map-free terrestrial carbon pools, permafrost soil C, & wildfire state."""

    c_leaf: torch.Tensor           # [N_cells] Leaf carbon [kg C m^-2]
    c_wood: torch.Tensor           # [N_cells] Woody stem + coarse-root carbon [kg C m^-2]
    c_root: torch.Tensor           # [N_cells] Fine-root carbon [kg C m^-2]
    c_lit_fast: torch.Tensor       # [N_cells] Fast (leaf+root) litter carbon [kg C m^-2]
    c_lit_slow: torch.Tensor       # [N_cells] Slow (woody) litter carbon [kg C m^-2]
    c_soil: torch.Tensor           # [5, N_cells] Vertically resolved soil C per layer [kg C m^-2]

    # Neumaier/Kahan compensation buffers for multi-decadal float32 conservation
    c_leaf_comp: torch.Tensor
    c_wood_comp: torch.Tensor
    c_root_comp: torch.Tensor
    c_lit_fast_comp: torch.Tensor
    c_lit_slow_comp: torch.Tensor
    c_soil_comp: torch.Tensor

    # Prognostic SPITFIRE Nesterov dryness index [K^2]
    fire_di: torch.Tensor

    # Instantaneous carbon & pyrogenic aerosol diagnostics [kg m^-2 s^-1]
    gpp_kgc_m2s: torch.Tensor      # Gross primary production [kg C m^-2 s^-1]
    ra_kgc_m2s: torch.Tensor       # Autotrophic respiration [kg C m^-2 s^-1]
    npp_kgc_m2s: torch.Tensor      # Net primary production (GPP - Ra) [kg C m^-2 s^-1]
    rh_kgc_m2s: torch.Tensor       # Heterotrophic respiration (litter + 5-layer soil) [kg C m^-2 s^-1]
    fire_c_kgc_m2s: torch.Tensor   # Wildfire pyrogenic CO2-C flux [kg C m^-2 s^-1]
    nbp_kgc_m2s: torch.Tensor      # Net biome production (GPP - Ra - Rh - Fire; + = land sink)
    fire_bc_kg_m2s: torch.Tensor   # Wildfire black carbon emission flux [kg BC m^-2 s^-1]
    fire_oc_kg_m2s: torch.Tensor   # Wildfire organic carbon emission flux [kg OC m^-2 s^-1]
    burned_frac_s: torch.Tensor    # Instantaneous fractional burned area rate [s^-1]


def init_land_carbon_state(
    surf_type: torch.Tensor,
    lai_init: torch.Tensor,
    h_canopy_init: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> LandCarbonState:
    """Initialize map-free terrestrial carbon pools purely from initial EEO LAI & canopy height."""
    dev = device if device is not None else surf_type.device
    nc = surf_type.shape[0]
    is_land = (surf_type.to(dev) == SURF_LAND).to(dtype)
    lai = torch.clamp(lai_init.to(dev, dtype=dtype), min=0.0) * is_land
    hc = torch.clamp(h_canopy_init.to(dev, dtype=dtype), min=0.0) * is_land

    # Map-free allometric initialization (no biome/PFT maps)
    c_leaf = (0.06 * lai) * is_land
    c_root = (0.08 * lai + 0.05 * hc) * is_land
    c_wood = (0.45 * torch.pow(hc, 1.3)) * is_land
    c_lit_fast = (0.25 * lai) * is_land
    c_lit_slow = (0.15 * c_wood) * is_land

    # 5-layer vertically distributed initial soil carbon (~8 kg C m^-2 vegetated land average)
    dz = torch.tensor(SOIL_LAYER_DEPTHS, device=dev, dtype=dtype)
    z_mid = torch.cumsum(dz, dim=0) - 0.5 * dz
    w_z = torch.exp(-z_mid / 0.65) * dz
    w_z = w_z / w_z.sum()
    soil_tot = (1.5 + 2.5 * torch.clamp(lai, max=6.0)) * is_land
    c_soil = w_z.unsqueeze(1) * soil_tot.unsqueeze(0)

    z1 = torch.zeros(nc, device=dev, dtype=dtype)
    return LandCarbonState(
        c_leaf=c_leaf,
        c_wood=c_wood,
        c_root=c_root,
        c_lit_fast=c_lit_fast,
        c_lit_slow=c_lit_slow,
        c_soil=c_soil,
        c_leaf_comp=z1.clone(),
        c_wood_comp=z1.clone(),
        c_root_comp=z1.clone(),
        c_lit_fast_comp=z1.clone(),
        c_lit_slow_comp=z1.clone(),
        c_soil_comp=torch.zeros_like(c_soil),
        fire_di=z1.clone(),
        gpp_kgc_m2s=z1.clone(),
        ra_kgc_m2s=z1.clone(),
        npp_kgc_m2s=z1.clone(),
        rh_kgc_m2s=z1.clone(),
        fire_c_kgc_m2s=z1.clone(),
        nbp_kgc_m2s=z1.clone(),
        fire_bc_kg_m2s=z1.clone(),
        fire_oc_kg_m2s=z1.clone(),
        burned_frac_s=z1.clone(),
    )


def step_land_carbon_and_fire(
    lc: LandCarbonState,
    surf_type: torch.Tensor,
    gpp_umol: torch.Tensor,
    resp_leaf_umol: torch.Tensor,
    lai: torch.Tensor,
    h_canopy: torch.Tensor,
    t_soil: torch.Tensor,
    w_liq_soil: torch.Tensor,
    w_ice_soil: torch.Tensor,
    f_snow: torch.Tensor,
    t_sfc_k: torch.Tensor,
    vpd_pa: torch.Tensor,
    u10_ms: torch.Tensor,
    precip_kgm2s: torch.Tensor,
    dt: float,
) -> None:
    """Advance terrestrial carbon pools, 5-layer soil/permafrost C, & SPITFIRE wildfire by `dt` [s]."""
    dtype = lc.c_leaf.dtype
    dev = lc.c_leaf.device
    is_land = (surf_type == SURF_LAND).to(dtype)
    dt_yr = dt / SEC_PER_YEAR
    inv_dt = 1.0 / max(dt, 1.0)

    # 1. GPP & Autotrophic Respiration (Leaf + Stem + Root maintenance + Growth)
    gpp = torch.clamp(gpp_umol, min=0.0) * UMOL_TO_KG_C * is_land
    r_leaf = torch.clamp(resp_leaf_umol, min=0.0) * UMOL_TO_KG_C * is_land

    t0_c = torch.clamp(t_soil[0] - 273.15, min=-40.0, max=50.0)
    q10_veg = torch.pow(torch.full_like(t0_c, SOIL_Q10), 0.1 * (t0_c - 25.0))
    r_wood_m = (lc.c_wood * (R_MAINT_WOOD_YR / SEC_PER_YEAR) * q10_veg) * is_land
    r_root_m = (lc.c_root * (R_MAINT_ROOT_YR / SEC_PER_YEAR) * q10_veg) * is_land
    r_maint = r_leaf + r_wood_m + r_root_m

    # Growth respiration applies only to positive net carbon surplus (Clark et al., 2011)
    surplus = torch.clamp(gpp - r_maint, min=0.0)
    r_growth = RESP_GROWTH_FRAC * surplus
    ra = r_maint + r_growth
    npp = gpp - ra  # Can be slightly negative at night; drawn safely from c_leaf/c_root without violating conservation

    # 2. Map-Free Allocation Guided by Emergent EEO Canopy Height h_canopy
    #    Short canopies (grasslands, h < 2 m) -> f_wood ~ 0.05; tall forests (h ~ 25 m) -> f_wood ~ 0.45
    hc_pos = torch.clamp(h_canopy, min=0.0)
    f_wood = 0.05 + 0.42 * (hc_pos / (hc_pos + 4.0))
    f_leaf = 0.55 * (1.0 - f_wood)
    f_root = 1.0 - f_wood - f_leaf

    d_npp = npp * dt
    # If NPP < 0 (nocturnal maintenance respiration), draw proportionally from available live pools
    live_tot = torch.clamp(lc.c_leaf + lc.c_wood + lc.c_root, min=1.0e-8)
    neg_draw = torch.clamp(-d_npp, min=0.0)
    neg_capped = torch.minimum(neg_draw, 0.5 * live_tot)
    # Any maintenance respiration exceeding live-pool cap is reduced identically so carbon conservation is EXACT!
    ra_eff = ra - (neg_draw - neg_capped) * inv_dt
    npp_eff = gpp - ra_eff

    pos_in = torch.clamp(d_npp, min=0.0)
    in_leaf = f_leaf * pos_in - neg_capped * (lc.c_leaf / live_tot)
    in_wood = f_wood * pos_in - neg_capped * (lc.c_wood / live_tot)
    in_root = f_root * pos_in - neg_capped * (lc.c_root / live_tot)

    # 3. Tissue Turnover -> Litter Pools (exact exponential fractions)
    f_turn_l = -math.expm1(-dt_yr / TAU_LEAF_YR)
    f_turn_r = -math.expm1(-dt_yr / TAU_ROOT_YR)
    f_turn_w = -math.expm1(-dt_yr / TAU_WOOD_YR)

    c_leaf_m, leaf_comp = _kahan_add(lc.c_leaf, in_leaf, lc.c_leaf_comp)
    c_root_m, root_comp = _kahan_add(lc.c_root, in_root, lc.c_root_comp)
    c_wood_m, wood_comp = _kahan_add(lc.c_wood, in_wood, lc.c_wood_comp)

    turn_l = f_turn_l * torch.clamp(c_leaf_m, min=0.0) * is_land
    turn_r = f_turn_r * torch.clamp(c_root_m, min=0.0) * is_land
    turn_w = f_turn_w * torch.clamp(c_wood_m, min=0.0) * is_land

    c_leaf_1, leaf_comp = _kahan_add(c_leaf_m, -turn_l, leaf_comp)
    c_root_1, root_comp = _kahan_add(c_root_m, -turn_r, root_comp)
    c_wood_1, wood_comp = _kahan_add(c_wood_m, -turn_w, wood_comp)

    c_lit_fast_m, lf_comp = _kahan_add(lc.c_lit_fast, turn_l + turn_r, lc.c_lit_fast_comp)
    c_lit_slow_m, ls_comp = _kahan_add(lc.c_lit_slow, turn_w, lc.c_lit_slow_comp)

    # 4. Litter & Vertically Resolved 5-Layer Soil / Permafrost Decomposition (Koven et al., 2013)
    dz = torch.tensor(SOIL_LAYER_DEPTHS, device=dev, dtype=dtype).unsqueeze(1)  # [5, 1]
    z_mid = (torch.cumsum(dz, dim=0) - 0.5 * dz)  # [5, 1]

    # Temperature scalar (Q10) & permafrost frozen-soil suppression
    t_s_c = torch.clamp(t_soil - 273.15, min=-45.0, max=50.0)
    f_t_soil = torch.pow(torch.full_like(t_s_c, SOIL_Q10), 0.1 * (t_s_c - 25.0))
    w_tot = torch.clamp(w_liq_soil + w_ice_soil, min=1.0e-6)
    f_ice_soil = torch.clamp(w_ice_soil / w_tot, 0.0, 1.0)
    f_freeze = (1.0 - f_ice_soil) + SOIL_FROZEN_SUPPRESS * f_ice_soil

    # Soil moisture scalar: peaks at moderate wetness, drops in desert drought & anoxic saturation
    # Approximate relative pore wetness from water mass / (0.45 * 1000 * dz)
    sat_frac = torch.clamp(w_liq_soil / (450.0 * dz), 0.0, 1.0)
    f_moist = torch.clamp(0.15 + 0.85 * torch.sin(math.pi * torch.clamp(sat_frac, 0.0, 0.95)), 0.10, 1.0)
    env_soil = f_t_soil * f_freeze * f_moist * is_land.unsqueeze(0)

    # Litter decomposition (governed by Layer-0 soil environment)
    env0 = env_soil[0]
    k_lf = -torch.expm1(-torch.clamp((dt_yr / TAU_LIT_FAST_YR) * env0, max=20.0))
    k_ls = -torch.expm1(-torch.clamp((dt_yr / TAU_LIT_SLOW_YR) * env0, max=20.0))
    dec_lf = k_lf * torch.clamp(c_lit_fast_m, min=0.0)
    dec_ls = k_ls * torch.clamp(c_lit_slow_m, min=0.0)
    c_lit_fast_1, lf_comp = _kahan_add(c_lit_fast_m, -dec_lf, lf_comp)
    c_lit_slow_1, ls_comp = _kahan_add(c_lit_slow_m, -dec_ls, ls_comp)

    rh_lit_kg = (1.0 - HUMIF_FRAC_FAST) * dec_lf + (1.0 - HUMIF_FRAC_SLOW) * dec_ls
    humif_in = HUMIF_FRAC_FAST * dec_lf + HUMIF_FRAC_SLOW * dec_ls  # Enters top soil layers

    # Distribute humified litter into top 2 soil layers (70% L0, 30% L1)
    d_soil_in = torch.zeros_like(lc.c_soil)
    d_soil_in[0] = 0.70 * humif_in
    d_soil_in[1] = 0.30 * humif_in

    # Depth-attenuated soil decomposition (Koven et al., 2013, Eq. 6)
    depth_fac = torch.exp(-z_mid / SOIL_Z_TAU_M)
    k_soil = -torch.expm1(-torch.clamp((dt_yr / TAU_SOIL_BASE_YR) * depth_fac * env_soil, max=20.0))
    c_soil_m, soil_comp = _kahan_add(lc.c_soil, d_soil_in, lc.c_soil_comp)
    dec_soil = k_soil * torch.clamp(c_soil_m, min=0.0)
    c_soil_1, soil_comp = _kahan_add(c_soil_m, -dec_soil, soil_comp)
    rh_soil_kg = dec_soil.sum(dim=0)

    # Conservative vertical cryoturbation / bioturbation mixing across the 5 soil layers
    # (flux on C concentration C_k / dz_k; zero-flux top & bottom boundaries -> exact column conservation!)
    c_conc = torch.clamp(c_soil_1, min=0.0) / dz
    dz_int = 0.5 * (dz[:-1] + dz[1:])
    f_mix = -(SOIL_CRYO_DIFF_M2_YR * dt_yr) * (c_conc[1:] - c_conc[:-1]) / dz_int
    # Stability cap so mixing never removes >25% of a layer in one step
    f_cap = 0.25 * torch.minimum(torch.clamp(c_soil_1[:-1], min=0.0), torch.clamp(c_soil_1[1:], min=0.0))
    f_mix = torch.clamp(f_mix, -f_cap, f_cap)
    z_row = torch.zeros_like(c_soil_1[0:1])
    d_mix = (torch.cat([z_row, f_mix], dim=0) - torch.cat([f_mix, z_row], dim=0)) * is_land.unsqueeze(0)
    c_soil_new, soil_comp = _kahan_add(c_soil_1, d_mix, soil_comp)

    # 5. Autonomous Map-Free Wildfire (SPITFIRE; Thonicke et al., 2010)
    #    Update Nesterov dryness index (resets on rain > 3 mm/day or snow cover > 0.2)
    t_air_c = torch.clamp(t_sfc_k - 273.15, min=-40.0, max=55.0)
    # Approximate dewpoint depression from VPD via Clausius-Clapeyron slope (~200 Pa/K at 25 C)
    dp_dep = torch.clamp(vpd_pa / 160.0, 0.0, 35.0)
    di_inc = torch.clamp(t_air_c, min=0.0) * dp_dep * (dt / C.SECONDS_PER_DAY)
    wet_reset = (precip_kgm2s * C.SECONDS_PER_DAY > 3.0) | (f_snow > 0.20)
    di_new = torch.where(wet_reset, torch.zeros_like(lc.fire_di), torch.clamp(lc.fire_di + di_inc, 0.0, FIRE_DI_MAX)) * is_land
    lc.fire_di = di_new

    fdi = -torch.expm1(-FIRE_DI_K * di_new)
    fuel_lit = c_lit_fast_1 + c_lit_slow_1
    f_fuel = torch.clamp((fuel_lit - FIRE_FUEL_MIN_KGC) / 0.50, 0.0, 1.0)
    f_wind = torch.clamp(0.30 + 0.14 * u10_ms, 0.30, 1.50)
    burn_rate_s = (FIRE_MAX_RATE_YR / SEC_PER_YEAR) * fdi * f_fuel * f_wind * is_land
    f_burn = -torch.expm1(-torch.clamp(burn_rate_s * dt, min=0.0, max=0.25))

    burn_lf = (f_burn * FIRE_COMBUST_LIT) * torch.clamp(c_lit_fast_1, min=0.0)
    burn_ls = (f_burn * FIRE_COMBUST_LIT) * torch.clamp(c_lit_slow_1, min=0.0)
    burn_leaf = (f_burn * FIRE_COMBUST_LEAF) * torch.clamp(c_leaf_1, min=0.0)
    burn_wood = (f_burn * FIRE_COMBUST_WOOD) * torch.clamp(c_wood_1, min=0.0)

    c_lit_fast_new, lf_comp = _kahan_add(c_lit_fast_1, -burn_lf, lf_comp)
    c_lit_slow_new, ls_comp = _kahan_add(c_lit_slow_1, -burn_ls, ls_comp)
    c_leaf_new, leaf_comp = _kahan_add(c_leaf_1, -burn_leaf, leaf_comp)
    c_wood_new, wood_comp = _kahan_add(c_wood_1, -burn_wood, wood_comp)

    fire_c_kg = burn_lf + burn_ls + burn_leaf + burn_wood
    fire_c_flux = fire_c_kg * inv_dt
    rh_flux = (rh_lit_kg + rh_soil_kg) * inv_dt

    # Commit updated state & diagnostics
    lc.c_leaf = c_leaf_new * is_land
    lc.c_wood = c_wood_new * is_land
    lc.c_root = c_root_1 * is_land
    lc.c_lit_fast = c_lit_fast_new * is_land
    lc.c_lit_slow = c_lit_slow_new * is_land
    lc.c_soil = c_soil_new * is_land
    lc.c_leaf_comp = leaf_comp * is_land
    lc.c_wood_comp = wood_comp * is_land
    lc.c_root_comp = root_comp * is_land
    lc.c_lit_fast_comp = lf_comp * is_land
    lc.c_lit_slow_comp = ls_comp * is_land
    lc.c_soil_comp = soil_comp * is_land.unsqueeze(0)

    lc.gpp_kgc_m2s = gpp
    lc.ra_kgc_m2s = ra_eff
    lc.npp_kgc_m2s = npp_eff
    lc.rh_kgc_m2s = rh_flux
    lc.fire_c_kgc_m2s = fire_c_flux
    lc.nbp_kgc_m2s = npp_eff - rh_flux - fire_c_flux
    lc.fire_bc_kg_m2s = FIRE_EF_BC * fire_c_flux
    lc.fire_oc_kg_m2s = FIRE_EF_OC * fire_c_flux
    lc.burned_frac_s = f_burn * inv_dt


def diagnose_land_carbon_metrics(lc: LandCarbonState, area_cell: torch.Tensor) -> dict[str, float]:
    """Compute global terrestrial carbon stocks [Pg C] and fluxes [Pg C yr^-1] in float64."""
    area = area_cell.detach().cpu().to(torch.float64)
    kg_to_pg = 1.0e-12
    kg_s_to_pg_yr = 1.0e-12 * SEC_PER_YEAR

    def _stock(t: torch.Tensor) -> float:
        return float((t.detach().cpu().to(torch.float64) * area).sum().item() * kg_to_pg)

    def _flux(t: torch.Tensor) -> float:
        return float((t.detach().cpu().to(torch.float64) * area).sum().item() * kg_s_to_pg_yr)

    c_leaf_64 = lc.c_leaf.detach().cpu().to(torch.float64) + lc.c_leaf_comp.detach().cpu().to(torch.float64)
    c_root_64 = lc.c_root.detach().cpu().to(torch.float64) + lc.c_root_comp.detach().cpu().to(torch.float64)
    c_wood_64 = lc.c_wood.detach().cpu().to(torch.float64) + lc.c_wood_comp.detach().cpu().to(torch.float64)
    c_lf_64 = lc.c_lit_fast.detach().cpu().to(torch.float64) + lc.c_lit_fast_comp.detach().cpu().to(torch.float64)
    c_ls_64 = lc.c_lit_slow.detach().cpu().to(torch.float64) + lc.c_lit_slow_comp.detach().cpu().to(torch.float64)
    c_soil_64 = (lc.c_soil.detach().cpu().to(torch.float64) + lc.c_soil_comp.detach().cpu().to(torch.float64)).sum(dim=0)
    c_veg = float(((c_leaf_64 + c_root_64 + c_wood_64) * area).sum().item() * kg_to_pg)
    c_lit = float(((c_lf_64 + c_ls_64) * area).sum().item() * kg_to_pg)
    c_soil = float((c_soil_64 * area).sum().item() * kg_to_pg)
    return {
        "c_veg_pgc": c_veg,
        "c_litter_pgc": c_lit,
        "c_soil_pgc": c_soil,
        "c_land_total_pgc": c_veg + c_lit + c_soil,
        "land_gpp_pgc_yr": _flux(lc.gpp_kgc_m2s),
        "land_npp_pgc_yr": _flux(lc.npp_kgc_m2s),
        "land_rh_pgc_yr": _flux(lc.rh_kgc_m2s),
        "land_fire_pgc_yr": _flux(lc.fire_c_kgc_m2s),
        "land_nbp_pgc_yr": _flux(lc.nbp_kgc_m2s),
    }
