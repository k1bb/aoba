"""Map-Free EEO Terrestrial Carbon-Nitrogen (CN) Cycle, Permafrost Soil C/N, & MAML SPITFIRE Wildfire.

Extends AOBA's Eco-Evolutionary Optimality (EEO) terrestrial biosphere (`biosphere.py`)
into a fully closed, map-free coupled terrestrial carbon-nitrogen-fire system driven
purely by universal global parameters and 1:1 MAML subgrid patches (`[N_cells * Ny * Nx]`
or host `[N_cells]`):

1. **Stocker & Prentice (2024) CN-Model + FUN/Houlton Optimal Nitrogen Acquisition**:
   - Optimal leaf N demand driven by P-model acclimated `Vcmax25` and emergent leaf mass
     per area `LMA(f_evg, f_cold)` (Dong et al., 2022, *J. Ecol.*, 110, 2585-2602;
     Stocker & Prentice, 2024, *bioRxiv* 10.1101/2024.04.25.591063;
     Stocker et al., 2025, *New Phytologist*, 245, 49-68).
   - First-principles soil solute-transport N uptake (transpiration mass flow +
     moisture/fine-root-limited diffusion) plus temperature-dependent symbiotic biological
     nitrogen fixation (BNF; Houlton et al., 2008, *Nature*, 454, 327-330; Fisher et al.,
     2010, *Global Biogeochem. Cycles*, 24, GB1014, FUN Eq. 6), coupled with natural
     CRM lightning `LNOx` atmospheric deposition (`lightning.py`), leaching, anaerobic
     denitrification, and pyrogenic N volatilization.
   - Functional-balance dynamic carbon allocation: under nitrogen scarcity (`phi_N < 1`),
     plants shift assimilated carbon from aboveground tissues into fine roots (`f_root`)
     and gently elevate effective foliage carbon cost `z_eff` in `biosphere.py`.
2. **Vertically Resolved 5-Layer Soil Organic C & N + Permafrost Stabilization** (Koven et al.,
   2013, *Biogeosciences*, 10, 7109-7131, CLM4.5bgc):
   - Matches AOBA's 5 soil thermal/hydrology layers (`0.05, 0.15, 0.40, 0.80, 1.60 m`) with
     depth-attenuated decay, Q10 response, moisture limitation, frozen-soil permafrost
     suppression, organic N mineralization/immobilization, and conservative cryoturbation.
3. **Process-Based MAML Patch SPITFIRE Wildfire** (Thonicke et al., 2010, *Biogeosciences*,
   7, 1991-2011; Rothermel, 1972; Byram, 1959; Van Wagner, 1973):
   - Driven by explicit CRM cloud-to-ground lightning strikes (`lightning.py`), Nesterov
     dryness index, dead-fuel moisture damping, Rothermel forward/backward rate of spread,
     wind-elliptical burned area, Byram fireline intensity with 50 kW m^-1 self-extinction,
     scorch height, and canopy-height-dependent bark resistance.
   - In MAML subgrid mode (`maml_shape = (nc, ny, nx)`), burns discrete dry subcolumns and
     spreads to adjacent dry 2D CRM patches, directly resetting burnt-patch EEO `lai` and
     `h_canopy` to enable emergent subgrid forest-savanna patch mosaics.

Strictly conserves total land carbon and nitrogen via Neumaier/Kahan float32 summation:
    d(C_total) / dt == GPP - R_a_eff - R_h - F_fire_C == NBP
    d(N_total) / dt == F_dep + F_BNF - F_leach - F_denit - F_fire_N

AOBA Formulation Reference: Sec. 7.8, Eqs. (7.41)-(7.45) (Stocker-Prentice C/N, LES LMA/Lifespan, FUN BNF, 5-Layer Permafrost & MAML SPITFIRE Wildfire).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.biosphere import BiosphereState
from aoba.surface import N_SOIL_LAYERS, SOIL_LAYER_DEPTHS, SOIL_POROSITY, SURF_LAND, _kahan_add

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
HUMIF_FRAC_FAST: float = 0.14      # Fraction of decomposed fast litter entering soil C (rest -> R_h & net N min) [-]
HUMIF_FRAC_SLOW: float = 0.18      # Fraction of decomposed slow litter entering soil C (rest -> R_h & net N min) [-]

# Vertically resolved soil C/N & permafrost parameters (Koven et al., 2013, Sec. 2.2)
SOIL_Z_TAU_M: float = 0.50         # e-folding depth for intrinsic soil decay rate attenuation [m]
SOIL_Q10: float = 2.0              # Heterotrophic respiration Q10 [-]
SOIL_FROZEN_SUPPRESS: float = 0.05 # Decomposition multiplier when fully frozen (permafrost lock) [-]
SOIL_CRYO_DIFF_M2_YR: float = 5.0e-4  # Vertical bioturbation / cryoturbation diffusivity [m^2 yr^-1]

# Stocker & Prentice (2024) CN-model & FUN/Houlton (2008, 2010) Nitrogen parameters (100% map-free)
N_RUBISCO_SLOPE: float = 1.76e-5   # Leaf photosynthetic N per Vcmax25 [kg N m^-2 / (umol m^-2 s^-1)] (Dong et al., 2022)
N_STRUCT_SLOPE: float = 0.016      # Structural leaf N per leaf C mass [kg N (kg C)^-1] (Dong et al., 2022)
CN_LEAF_MIN: float = 15.0          # Minimum physiological leaf C:N ratio [kg C (kg N)^-1]
CN_LEAF_MAX: float = 65.0          # Maximum physiological leaf C:N ratio [kg C (kg N)^-1]
CN_ROOT_TARGET: float = 45.0       # Fine-root C:N stoichiometry [kg C (kg N)^-1]
CN_WOOD_TARGET: float = 250.0      # Woody stem C:N stoichiometry [kg C (kg N)^-1]
CN_SOIL_TARGET: float = 14.0       # Humified soil organic matter target C:N [kg C (kg N)^-1]
N_RETRANS_LEAF: float = 0.50       # Foliar N resorption efficiency during senescence [-]
N_RETRANS_ROOT: float = 0.25       # Fine-root N resorption efficiency during turnover [-]
N_UPTAKE_K_YR: float = 45.0        # Maximum diffusive mineral-N uptake rate constant (~8-day root labile-N turnover) [yr^-1]
N_ROOT_HALF_SAT: float = 0.25      # Fine-root C half-saturation for diffusive N uptake [kg C m^-2]
BNF_COST_S: float = 5.0            # FUN/Houlton BNF base carbon cost factor [kg C (kg N)^-1] (Fisher et al., 2010)
BNF_COST_A: float = -3.62          # Houlton et al. (2008) nitrogenase temperature curve intercept [-]
BNF_COST_B: float = 0.27           # Houlton et al. (2008) nitrogenase temperature slope [C^-1]
BNF_COST_C: float = 25.15          # Houlton et al. (2008) nitrogenase optimum temperature [C]
BNF_MAX_NPP_FRAC: float = 0.022    # Max fraction of positive NPP allocatable to symbiotic BNF (~60-120 Tg N/yr; Fisher 2010) [-]
BNF_ASYM_MAX_YR: float = 2.0e-4    # Asymbiotic free-living BNF max rate at 25 C [kg N m^-2 yr^-1] (~15-25 Tg N/yr)
N_LEACH_RATE_YR: float = 0.60      # Mineral-N leaching rate constant under wet soil drainage [yr^-1]
N_DENIT_RATE_YR: float = 0.80      # Anaerobic denitrification rate constant under waterlogging [yr^-1]
N_MIN_INIT: float = 0.0025         # Map-free initial soil mineral N stock (~2.5 g N m^-2, ~0.3 Pg N globally) [kg N m^-2]
PHI_N_EMA_DAYS: float = 15.0       # Smoothing timescale for N sufficiency factor phi_N [days]

# Process-based SPITFIRE wildfire parameters (Thonicke et al., 2010; Rothermel, 1972; Byram, 1959)
FIRE_DI_MAX: float = 5000.0        # Nesterov dryness index saturation cap [K^2]
FIRE_DI_K: float = 3.0e-4          # Fire danger index FDI = 1 - exp(-k * DI) [K^-2]
FIRE_FUEL_MIN_KGC: float = 0.12    # Minimum litter fuel load required to sustain fire [kg C m^-2]
FIRE_IGNIT_EFF: float = 0.04       # CG lightning ignition efficiency (Thonicke et al., 2010, Sec. 2.2) [-]
FIRE_BG_CG_FLASH_M2S: float = 1.5e-14  # Fallback global mean CG flash density when lightning module disabled [fl m^-2 s^-1]
FIRE_ROS_BASE_M_MIN: float = 3.5   # Baseline Rothermel dry-fuel spread velocity [m min^-1]
FIRE_HEAT_CONTENT_KJ: float = 18000.0  # Low heat content of dry biomass fuel [kJ kg^-1] (Thonicke et al., 2010)
FIRE_EXTINCT_KW_M: float = 50.0    # Minimum Byram fireline intensity for self-sustaining spread [kW m^-1]
FIRE_SCORCH_COEF: float = 0.11     # Van Wagner (1973) / Thonicke et al. (2010) scorch-height factor [m (kW/m)^-2/3]
FIRE_MAX_RATE_YR: float = 1.50     # Peak dry-season instantaneous fractional burned-area cap (~0.25/yr over 2-mo dry season) [yr^-1]
FIRE_COMBUST_LIT: float = 0.70     # Litter combustion completeness [-]
FIRE_COMBUST_LEAF: float = 0.55    # Canopy leaf combustion completeness during crown/surface fire [-]
FIRE_COMBUST_WOOD: float = 0.10    # Woody stem combustion completeness weighted by crown mortality [-]
FIRE_N_VOL_FRAC: float = 0.85      # Fraction of combusted biomass N volatilized (rest returns to n_min as ash) [-]
FIRE_EF_BC: float = 0.0012         # Black carbon emission factor [kg BC / kg C burned] (Andreae & Merlet, 2001)
FIRE_EF_OC: float = 0.0110         # Organic carbon aerosol emission factor [kg OC / kg C burned]

_LC_SOIL_DZ_CACHE: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}


@dataclass
class LandCarbonState:
    """Prognostic map-free terrestrial carbon & nitrogen pools, permafrost soil C/N, & MAML SPITFIRE state."""

    # Carbon pools [M] and [5, M] (M = N_cells or MAML N_cells*Ny*Nx) [kg C m^-2]
    c_leaf: torch.Tensor
    c_wood: torch.Tensor
    c_root: torch.Tensor
    c_lit_fast: torch.Tensor
    c_lit_slow: torch.Tensor
    c_soil: torch.Tensor

    # Neumaier/Kahan compensation buffers for multi-decadal float32 carbon conservation
    c_leaf_comp: torch.Tensor
    c_wood_comp: torch.Tensor
    c_root_comp: torch.Tensor
    c_lit_fast_comp: torch.Tensor
    c_lit_slow_comp: torch.Tensor
    c_soil_comp: torch.Tensor

    # Prognostic SPITFIRE Nesterov dryness index [K^2]
    fire_di: torch.Tensor

    # Instantaneous carbon & pyrogenic aerosol diagnostics [kg m^-2 s^-1]
    gpp_kgc_m2s: torch.Tensor
    ra_kgc_m2s: torch.Tensor
    npp_kgc_m2s: torch.Tensor
    rh_kgc_m2s: torch.Tensor
    fire_c_kgc_m2s: torch.Tensor
    nbp_kgc_m2s: torch.Tensor
    fire_bc_kg_m2s: torch.Tensor
    fire_oc_kg_m2s: torch.Tensor
    burned_frac_s: torch.Tensor

    # Prognostic Nitrogen pools [kg N m^-2] + Neumaier/Kahan compensators (Stocker & Prentice, 2024)
    n_leaf: torch.Tensor | None = None
    n_wood: torch.Tensor | None = None
    n_root: torch.Tensor | None = None
    n_lit_fast: torch.Tensor | None = None
    n_lit_slow: torch.Tensor | None = None
    n_soil: torch.Tensor | None = None        # [5, M]
    n_min: torch.Tensor | None = None         # [M] Soil mineral N (NH4+ + NO3-) [kg N m^-2]

    n_leaf_comp: torch.Tensor | None = None
    n_wood_comp: torch.Tensor | None = None
    n_root_comp: torch.Tensor | None = None
    n_lit_fast_comp: torch.Tensor | None = None
    n_lit_slow_comp: torch.Tensor | None = None
    n_soil_comp: torch.Tensor | None = None
    n_min_comp: torch.Tensor | None = None

    # Nitrogen cycle & functional-balance diagnostics
    phi_n: torch.Tensor | None = None             # 15-day EMA nitrogen sufficiency in [0.15, 1.0] [-]
    f_root_alloc: torch.Tensor | None = None      # Emergent root C allocation fraction [-]
    bnf_kgn_m2s: torch.Tensor | None = None       # Symbiotic + asymbiotic BNF flux [kg N m^-2 s^-1]
    n_uptake_kgn_m2s: torch.Tensor | None = None  # Soil mineral-N root uptake flux [kg N m^-2 s^-1]
    n_dep_kgn_m2s: torch.Tensor | None = None     # Natural LNOx atmospheric N deposition [kg N m^-2 s^-1]
    n_leach_kgn_m2s: torch.Tensor | None = None   # Soil drainage mineral-N leaching [kg N m^-2 s^-1]
    n_denit_kgn_m2s: torch.Tensor | None = None   # Anaerobic denitrification loss [kg N m^-2 s^-1]
    n_fire_kgn_m2s: torch.Tensor | None = None    # Pyrogenic N volatilization loss [kg N m^-2 s^-1]

    # MAML discrete-patch fire hazard accumulator & physical SPITFIRE diagnostics
    fire_accum: torch.Tensor | None = None        # Staggered patch burn-hazard accumulator in [0, 1) [-]
    fire_intensity_kw_m: torch.Tensor | None = None  # Byram fireline intensity I_B [kW m^-1]
    fire_scorch_m: torch.Tensor | None = None     # Van Wagner / SPITFIRE scorch height [m]
    fire_mort_frac: torch.Tensor | None = None    # Woody crown/cambial mortality fraction [-]
    patch_burned_flag: torch.Tensor | None = None # 1.0 on patches that burned during the last step, else 0.0


def init_land_carbon_state(
    surf_type: torch.Tensor,
    lai_init: torch.Tensor,
    h_canopy_init: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    maml_n_sub: int = 1,
) -> LandCarbonState:
    """Initialize map-free terrestrial carbon & nitrogen pools purely from initial EEO LAI & canopy height."""
    dev = device if device is not None else surf_type.device
    m_tot = surf_type.shape[0]
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

    # Stoichiometrically consistent map-free initial nitrogen pools [kg N m^-2]
    n_leaf = (c_leaf / 30.0) * is_land
    n_root = (c_root / CN_ROOT_TARGET) * is_land
    n_wood = (c_wood / CN_WOOD_TARGET) * is_land
    n_lit_fast = (c_lit_fast / 35.0) * is_land
    n_lit_slow = (c_lit_slow / 150.0) * is_land
    n_soil = (c_soil / CN_SOIL_TARGET) * is_land.unsqueeze(0)
    n_min = (N_MIN_INIT * is_land)

    # Deterministic staggered initial phase across MAML subcolumns so discrete patch fires
    # sample the continuous burn-rate distribution smoothly without synchronized grid-cell flashes
    if maml_n_sub > 1 and (m_tot % maml_n_sub == 0):
        nc = m_tot // maml_n_sub
        # Golden-ratio quasi-random phase in [0, 1) across subcolumns & host cells
        idx_sub = torch.arange(maml_n_sub, device=dev, dtype=dtype).view(1, maml_n_sub)
        idx_c = torch.arange(nc, device=dev, dtype=dtype).view(nc, 1)
        fire_accum = torch.remainder(
            (idx_sub + 0.5) / float(maml_n_sub) + 0.61803398875 * idx_c,
            1.0,
        ).reshape(m_tot) * is_land
    else:
        fire_accum = torch.zeros(m_tot, device=dev, dtype=dtype)

    z1 = torch.zeros(m_tot, device=dev, dtype=dtype)
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
        n_leaf=n_leaf,
        n_wood=n_wood,
        n_root=n_root,
        n_lit_fast=n_lit_fast,
        n_lit_slow=n_lit_slow,
        n_soil=n_soil,
        n_min=n_min,
        n_leaf_comp=z1.clone(),
        n_wood_comp=z1.clone(),
        n_root_comp=z1.clone(),
        n_lit_fast_comp=z1.clone(),
        n_lit_slow_comp=z1.clone(),
        n_soil_comp=torch.zeros_like(n_soil),
        n_min_comp=z1.clone(),
        phi_n=torch.ones(m_tot, device=dev, dtype=dtype),
        f_root_alloc=torch.full((m_tot,), 0.35, device=dev, dtype=dtype),
        bnf_kgn_m2s=z1.clone(),
        n_uptake_kgn_m2s=z1.clone(),
        n_dep_kgn_m2s=z1.clone(),
        n_leach_kgn_m2s=z1.clone(),
        n_denit_kgn_m2s=z1.clone(),
        n_fire_kgn_m2s=z1.clone(),
        fire_accum=fire_accum,
        fire_intensity_kw_m=z1.clone(),
        fire_scorch_m=z1.clone(),
        fire_mort_frac=z1.clone(),
        patch_burned_flag=z1.clone(),
    )


def _ensure_nitrogen_fields(lc: LandCarbonState, is_land: torch.Tensor) -> None:
    """Lazily initialize nitrogen & fire-patch fields for backward compatibility with older checkpoints."""
    if lc.n_leaf is not None and lc.fire_accum is not None:
        return
    z1 = torch.zeros_like(lc.c_leaf)
    if lc.n_leaf is None:
        lc.n_leaf = (lc.c_leaf / 30.0) * is_land
        lc.n_root = (lc.c_root / CN_ROOT_TARGET) * is_land
        lc.n_wood = (lc.c_wood / CN_WOOD_TARGET) * is_land
        lc.n_lit_fast = (lc.c_lit_fast / 35.0) * is_land
        lc.n_lit_slow = (lc.c_lit_slow / 150.0) * is_land
        lc.n_soil = (lc.c_soil / CN_SOIL_TARGET) * is_land.unsqueeze(0)
        lc.n_min = N_MIN_INIT * is_land
        lc.n_leaf_comp = z1.clone()
        lc.n_root_comp = z1.clone()
        lc.n_wood_comp = z1.clone()
        lc.n_lit_fast_comp = z1.clone()
        lc.n_lit_slow_comp = z1.clone()
        lc.n_soil_comp = torch.zeros_like(lc.c_soil)
        lc.n_min_comp = z1.clone()
        lc.phi_n = torch.ones_like(lc.c_leaf)
        lc.f_root_alloc = torch.full_like(lc.c_leaf, 0.35)
        lc.bnf_kgn_m2s = z1.clone()
        lc.n_uptake_kgn_m2s = z1.clone()
        lc.n_dep_kgn_m2s = z1.clone()
        lc.n_leach_kgn_m2s = z1.clone()
        lc.n_denit_kgn_m2s = z1.clone()
        lc.n_fire_kgn_m2s = z1.clone()
    if lc.fire_accum is None:
        lc.fire_accum = z1.clone()
        lc.fire_intensity_kw_m = z1.clone()
        lc.fire_scorch_m = z1.clone()
        lc.fire_mort_frac = z1.clone()
        lc.patch_burned_flag = z1.clone()


def compute_bnf_carbon_cost(t_soil_c: torch.Tensor) -> torch.Tensor:
    """Houlton et al. (2008) / Fisher et al. (2010, FUN Eq. 6) BNF carbon cost [kg C (kg N)^-1].

    Peaks in efficiency at ~25.15 degC (~6.26 kg C / kg N) and increases steeply in cold
    boreal/tundra soils (~187 kg C / kg N at 0 degC), shutting down symbiotic BNF map-free.
    """
    tc = torch.clamp(t_soil_c, min=-20.0, max=50.0)
    g_t = BNF_COST_A + BNF_COST_B * tc * (1.0 - 0.5 * tc / BNF_COST_C)
    eff = torch.clamp(torch.exp(torch.clamp(g_t, min=-10.0, max=2.0)), min=1.0e-3)
    return BNF_COST_S / eff


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
    *,
    enable_nitrogen: bool = True,
    vcmax25_eff: torch.Tensor | None = None,
    f_evg: torch.Tensor | None = None,
    f_c4: torch.Tensor | None = None,
    slope_tan: torch.Tensor | None = None,
    transp_kgm2s: torch.Tensor | None = None,
    n_dep_kgn_m2s: torch.Tensor | None = None,
    flash_cg_m2s: torch.Tensor | None = None,
    bio: BiosphereState | None = None,
    enable_fire_veg_feedback: bool = True,
    maml_shape: tuple[int, int, int] | None = None,
    crm_dx_m: float = 4000.0,
) -> None:
    """Advance terrestrial C-N pools, 5-layer soil/permafrost C & N, & MAML SPITFIRE wildfire by `dt` [s]."""
    dtype = lc.c_leaf.dtype
    dev = lc.c_leaf.device
    is_land = (surf_type == SURF_LAND).to(dtype)
    _ensure_nitrogen_fields(lc, is_land)
    dt_yr = dt / SEC_PER_YEAR
    inv_dt = 1.0 / max(dt, 1.0)

    from aoba.kernels import dispatch_land_carbon_cn_fire_metal, get_active_gpu_backend

    if get_active_gpu_backend(dev) in ("mps", "cuda") and dtype == torch.float32:
        use_maml_patch = (maml_shape is not None) and (maml_shape[1] * maml_shape[2] > 1)
        cg_raw = (
            torch.clamp(flash_cg_m2s, min=0.0)
            if flash_cg_m2s is not None
            else torch.full_like(lc.c_leaf, FIRE_BG_CG_FLASH_M2S)
        )
        if use_maml_patch and flash_cg_m2s is not None:
            assert maml_shape is not None
            nc_cg, ny_cg, nx_cg = maml_shape
            cg_3d = cg_raw.view(nc_cg, ny_cg, nx_cg)
            cg_cell_mean = (
                cg_3d.reshape(nc_cg, ny_cg * nx_cg)
                .mean(dim=-1, keepdim=True)
                .expand(nc_cg, ny_cg * nx_cg)
                .reshape(-1)
            )
            cg_rate = 0.50 * cg_raw + 0.50 * cg_cell_mean
        else:
            cg_rate = cg_raw

        vc25_in = vcmax25_eff if vcmax25_eff is not None else torch.full_like(lc.c_leaf, 45.0)
        fevg_in = f_evg if f_evg is not None else torch.full_like(lc.c_leaf, 0.5)
        fc4_in = torch.clamp(f_c4, 0.0, 1.0) if f_c4 is not None else torch.zeros_like(lc.c_leaf)
        slope_in = torch.clamp(slope_tan, 0.0, 0.75) if slope_tan is not None else torch.zeros_like(lc.c_leaf)
        transp_in = transp_kgm2s if transp_kgm2s is not None else torch.zeros_like(lc.c_leaf)
        n_dep_in = n_dep_kgn_m2s if n_dep_kgn_m2s is not None else torch.zeros_like(lc.c_leaf)

        f_in = torch.stack(
            [
                gpp_umol, resp_leaf_umol, h_canopy, f_snow, t_sfc_k, vpd_pa,
                u10_ms, precip_kgm2s, vc25_in, fevg_in, transp_in, n_dep_in, cg_rate,
                fc4_in, slope_in,
            ],
            dim=0,
        ).contiguous()
        c_1d_io = torch.stack(
            [
                lc.c_leaf, lc.c_root, lc.c_wood, lc.c_lit_fast, lc.c_lit_slow,
                lc.c_leaf_comp, lc.c_root_comp, lc.c_wood_comp, lc.c_lit_fast_comp, lc.c_lit_slow_comp,
            ],
            dim=0,
        ).contiguous()
        c_soil_io = lc.c_soil.contiguous()
        c_soil_comp_io = lc.c_soil_comp.contiguous()

        if enable_nitrogen:
            assert lc.n_leaf is not None and lc.n_root is not None and lc.n_wood is not None
            assert lc.n_lit_fast is not None and lc.n_lit_slow is not None and lc.n_min is not None
            assert lc.n_soil is not None and lc.n_soil_comp is not None
            assert lc.n_leaf_comp is not None and lc.n_root_comp is not None and lc.n_wood_comp is not None
            assert lc.n_lit_fast_comp is not None and lc.n_lit_slow_comp is not None and lc.n_min_comp is not None
            n_1d_io = torch.stack(
                [
                    lc.n_leaf, lc.n_root, lc.n_wood, lc.n_lit_fast, lc.n_lit_slow, lc.n_min,
                    lc.n_leaf_comp, lc.n_root_comp, lc.n_wood_comp,
                    lc.n_lit_fast_comp, lc.n_lit_slow_comp, lc.n_min_comp,
                ],
                dim=0,
            ).contiguous()
            n_soil_io = lc.n_soil.contiguous()
            n_soil_comp_io = lc.n_soil_comp.contiguous()
        else:
            dummy_n = torch.empty((1,), device=dev, dtype=dtype)
            n_1d_io = dummy_n
            n_soil_io = dummy_n
            n_soil_comp_io = dummy_n

        phi_in = lc.phi_n if lc.phi_n is not None else torch.ones_like(lc.c_leaf)
        fa_in = lc.fire_accum if lc.fire_accum is not None else torch.zeros_like(lc.c_leaf)
        fr_in = lc.f_root_alloc if lc.f_root_alloc is not None else torch.full_like(lc.c_leaf, 0.35)
        aux_io = torch.stack([phi_in, lc.fire_di, fa_in, fr_in], dim=0).contiguous()

        fire_out, n_diag_out, f_burn, fire_cn = dispatch_land_carbon_cn_fire_metal(
            surf_type_i32=surf_type.to(torch.int32).contiguous(),
            f_in=f_in,
            t_soil=t_soil.contiguous(),
            w_liq=w_liq_soil.contiguous(),
            w_ice=w_ice_soil.contiguous(),
            c_1d_io=c_1d_io,
            c_soil_io=c_soil_io,
            c_soil_comp_io=c_soil_comp_io,
            n_1d_io=n_1d_io,
            n_soil_io=n_soil_io,
            n_soil_comp_io=n_soil_comp_io,
            aux_io=aux_io,
            enable_n=bool(enable_nitrogen),
            use_maml=bool(use_maml_patch),
            maml_shape=maml_shape,
            dt=float(dt),
            sec_per_yr=float(SEC_PER_YEAR),
            crm_dx_m=float(crm_dx_m),
            soil_porosity=float(SOIL_POROSITY),
        )

        # Unpack updated pools & diagnostics
        lc.c_leaf = c_1d_io[0]
        lc.c_root = c_1d_io[1]
        lc.c_wood = c_1d_io[2]
        lc.c_lit_fast = c_1d_io[3]
        lc.c_lit_slow = c_1d_io[4]
        lc.c_leaf_comp = c_1d_io[5]
        lc.c_root_comp = c_1d_io[6]
        lc.c_wood_comp = c_1d_io[7]
        lc.c_lit_fast_comp = c_1d_io[8]
        lc.c_lit_slow_comp = c_1d_io[9]
        lc.c_soil = c_soil_io
        lc.c_soil_comp = c_soil_comp_io

        lc.phi_n = aux_io[0]
        lc.fire_di = aux_io[1]
        if use_maml_patch:
            lc.fire_accum = aux_io[2]
            lc.patch_burned_flag = f_burn
        else:
            lc.patch_burned_flag = (f_burn > 1.0e-6).to(dtype)
        lc.f_root_alloc = aux_io[3]

        lc.fire_intensity_kw_m = fire_out[3]
        lc.fire_scorch_m = fire_out[4]
        p_mort = fire_out[5]
        lc.fire_mort_frac = p_mort

        gpp = fire_out[7]
        ra_eff = fire_out[8]
        npp_eff = gpp - ra_eff
        rh_flux = fire_out[9]
        fire_c_flux = fire_cn[0]

        if enable_nitrogen:
            lc.n_leaf = n_1d_io[0]
            lc.n_root = n_1d_io[1]
            lc.n_wood = n_1d_io[2]
            lc.n_lit_fast = n_1d_io[3]
            lc.n_lit_slow = n_1d_io[4]
            lc.n_min = n_1d_io[5]
            lc.n_leaf_comp = n_1d_io[6]
            lc.n_root_comp = n_1d_io[7]
            lc.n_wood_comp = n_1d_io[8]
            lc.n_lit_fast_comp = n_1d_io[9]
            lc.n_lit_slow_comp = n_1d_io[10]
            lc.n_min_comp = n_1d_io[11]
            lc.n_soil = n_soil_io
            lc.n_soil_comp = n_soil_comp_io
            lc.bnf_kgn_m2s = n_diag_out[0]
            lc.n_uptake_kgn_m2s = n_diag_out[1]
            lc.n_dep_kgn_m2s = n_diag_out[2]
            lc.n_leach_kgn_m2s = n_diag_out[3]
            lc.n_denit_kgn_m2s = n_diag_out[4]
            lc.n_fire_kgn_m2s = fire_cn[1]
            if bio is not None:
                bio.n_avail_fac = lc.phi_n.clone()

        if enable_fire_veg_feedback and bio is not None:
            lai_drop = torch.clamp(f_burn * (0.35 + 0.55 * p_mort), 0.0, 0.92)
            hc_drop = torch.clamp(f_burn * (0.65 * p_mort), 0.0, 0.85)
            is_land_b = surf_type == SURF_LAND
            if bio.lai_evg is not None and bio.lai_dec is not None:
                bio.lai_evg = torch.where(is_land_b, bio.lai_evg * (1.0 - lai_drop), bio.lai_evg)
                bio.lai_dec = torch.where(is_land_b, bio.lai_dec * (1.0 - lai_drop), bio.lai_dec)
                bio.lai = torch.where(is_land_b, torch.clamp(bio.lai_evg + bio.lai_dec, 0.0, 8.0), bio.lai)
            else:
                bio.lai = torch.where(is_land_b, torch.clamp(bio.lai * (1.0 - lai_drop), 0.0, 8.0), bio.lai)
            bio.h_canopy = torch.where(is_land_b, torch.clamp(bio.h_canopy * (1.0 - hc_drop), min=0.10), bio.h_canopy)

        lc.gpp_kgc_m2s = gpp
        lc.ra_kgc_m2s = ra_eff
        lc.npp_kgc_m2s = npp_eff
        lc.rh_kgc_m2s = rh_flux
        lc.fire_c_kgc_m2s = fire_c_flux
        lc.nbp_kgc_m2s = npp_eff - rh_flux - fire_c_flux
        lc.fire_bc_kg_m2s = FIRE_EF_BC * fire_c_flux
        lc.fire_oc_kg_m2s = FIRE_EF_OC * fire_c_flux
        lc.burned_frac_s = f_burn * inv_dt
        return

    # 1. GPP & Autotrophic Respiration (Leaf + Stem + Root maintenance + Growth)
    gpp = torch.clamp(gpp_umol, min=0.0) * UMOL_TO_KG_C * is_land
    r_leaf = torch.clamp(resp_leaf_umol, min=0.0) * UMOL_TO_KG_C * is_land

    t0_c = torch.clamp(t_soil[0] - C.T_FREEZE, min=-40.0, max=50.0)
    q10_veg = torch.pow(torch.full_like(t0_c, SOIL_Q10), 0.1 * (t0_c - 25.0))
    r_wood_m = (lc.c_wood * (R_MAINT_WOOD_YR / SEC_PER_YEAR) * q10_veg) * is_land
    r_root_m = (lc.c_root * (R_MAINT_ROOT_YR / SEC_PER_YEAR) * q10_veg) * is_land
    r_maint = r_leaf + r_wood_m + r_root_m

    # Growth respiration applies only to positive net carbon surplus (Clark et al., 2011)
    surplus = torch.clamp(gpp - r_maint, min=0.0)
    r_growth = RESP_GROWTH_FRAC * surplus
    ra = r_maint + r_growth
    npp = gpp - ra

    # 2. Functional-Balance Carbon Allocation (Stocker & Prentice, 2024; height + nitrogen sufficiency)
    hc_pos = torch.clamp(h_canopy, min=0.0)
    f_wood_base = 0.05 + 0.42 * (hc_pos / (hc_pos + 4.0))
    f_root_base = 0.45 * (1.0 - f_wood_base)
    phi_prev = torch.clamp(lc.phi_n, 0.15, 1.0) if (enable_nitrogen and lc.phi_n is not None) else torch.ones_like(hc_pos)
    if enable_nitrogen:
        # Under N scarcity (phi_N < 1), shift C allocation dynamically into fine roots to forage for soil N
        f_root = torch.clamp(f_root_base + 0.35 * (1.0 - phi_prev), 0.15, 0.72)
        rem_abg = 1.0 - f_root
        abg_base = torch.clamp(1.0 - f_root_base, min=0.10)
        f_wood = rem_abg * (f_wood_base / abg_base)
        f_leaf = rem_abg - f_wood
    else:
        f_wood = f_wood_base
        f_leaf = 0.55 * (1.0 - f_wood)
        f_root = 1.0 - f_wood - f_leaf
    lc.f_root_alloc = f_root

    d_npp = npp * dt
    live_tot = torch.clamp(lc.c_leaf + lc.c_wood + lc.c_root, min=1.0e-8)
    neg_draw = torch.clamp(-d_npp, min=0.0)
    neg_capped = torch.minimum(neg_draw, 0.5 * live_tot)
    ra_eff = ra - (neg_draw - neg_capped) * inv_dt
    pos_in = torch.clamp(d_npp, min=0.0)

    # 3. Tissue Turnover (exact exponential fractions) & Leaf/Root N Retranslocation
    # Under N scarcity (phi_N < 1), thicker high-LMA leaves extend leaf lifespan and reduce leaf N turnover (Wright et al., 2004 LES)
    tau_leaf_eff = TAU_LEAF_YR * (1.0 + 0.80 * (1.0 - phi_prev))
    f_turn_l = -torch.expm1(-dt_yr / tau_leaf_eff)
    f_turn_r = -math.expm1(-dt_yr / TAU_ROOT_YR)
    f_turn_w = -math.expm1(-dt_yr / TAU_WOOD_YR)

    # Compute topsoil environment once for soil C/N decomposition and N uptake/BNF
    dz_key = (dev.type, dev.index or 0, dtype)
    dz = _LC_SOIL_DZ_CACHE.get(dz_key)
    if dz is None:
        dz = torch.tensor(SOIL_LAYER_DEPTHS, device=dev, dtype=dtype).unsqueeze(1)  # [5, 1]
        _LC_SOIL_DZ_CACHE[dz_key] = dz
    z_mid = torch.cumsum(dz, dim=0) - 0.5 * dz                                 # [5, 1]
    t_s_c = torch.clamp(t_soil - C.T_FREEZE, min=-45.0, max=50.0)
    f_t_soil = torch.pow(torch.full_like(t_s_c, SOIL_Q10), 0.1 * (t_s_c - 25.0))
    w_tot = torch.clamp(w_liq_soil + w_ice_soil, min=1.0e-6)
    f_ice_soil = torch.clamp(w_ice_soil / w_tot, 0.0, 1.0)
    f_freeze = (1.0 - f_ice_soil) + SOIL_FROZEN_SUPPRESS * f_ice_soil
    sat_frac = torch.clamp(w_liq_soil / SOIL_POROSITY, 0.0, 1.0)
    f_moist = torch.clamp(0.15 + 0.85 * torch.sin(math.pi * torch.clamp(sat_frac, 0.0, 0.95)), 0.10, 1.0)
    env_soil = f_t_soil * f_freeze * f_moist * is_land.unsqueeze(0)

    # 3b. Stocker CN-Model Optimal Leaf N Demand, Soil Solute-Transport N Uptake & FUN/Houlton BNF
    c_bnf_spent = torch.zeros_like(pos_in)
    if enable_nitrogen:
        assert lc.n_leaf is not None and lc.n_root is not None and lc.n_wood is not None
        assert lc.n_lit_fast is not None and lc.n_lit_slow is not None
        assert lc.n_soil is not None and lc.n_min is not None
        assert lc.n_leaf_comp is not None and lc.n_root_comp is not None and lc.n_wood_comp is not None
        assert lc.n_lit_fast_comp is not None and lc.n_lit_slow_comp is not None
        assert lc.n_soil_comp is not None and lc.n_min_comp is not None

        # Optimal leaf C:N from Dong et al. (2022), Wright et al. (2004 LES) & Stocker & Prentice (2024):
        # N_area = n_rub*Vcmax25 + n_struct*LMA, with LMA increasing under evergreen habit, cold, or N scarcity
        fevg_v = torch.clamp(f_evg, 0.0, 1.0) if f_evg is not None else torch.full_like(hc_pos, 0.5)
        f_cold_n = torch.sigmoid((-t0_c) / 5.0)
        lma_kgc = 0.045 + 0.055 * fevg_v + 0.025 * f_cold_n + 0.035 * (1.0 - phi_prev)
        vc25 = torch.clamp(vcmax25_eff, min=5.0, max=200.0) if vcmax25_eff is not None else torch.full_like(hc_pos, 45.0)
        n_area_opt = N_RUBISCO_SLOPE * vc25 + N_STRUCT_SLOPE * lma_kgc
        cn_leaf_opt = torch.clamp(lma_kgc / torch.clamp(n_area_opt, min=1.0e-4), CN_LEAF_MIN, CN_LEAF_MAX)

        # Tissue N turnover & internal resorption (retranslocation) before new growth allocation
        nturn_l_tot = f_turn_l * torch.clamp(lc.n_leaf, min=0.0) * is_land
        nturn_r_tot = f_turn_r * torch.clamp(lc.n_root, min=0.0) * is_land
        nturn_w_tot = f_turn_w * torch.clamp(lc.n_wood, min=0.0) * is_land
        n_retrans = N_RETRANS_LEAF * nturn_l_tot + N_RETRANS_ROOT * nturn_r_tot
        nturn_l_lit = nturn_l_tot - N_RETRANS_LEAF * nturn_l_tot
        nturn_r_lit = nturn_r_tot - N_RETRANS_ROOT * nturn_r_tot

        # Gross & net plant N demand for potential new C allocation `pos_in`
        d_n_gross = pos_in * (
            f_leaf / cn_leaf_opt + f_root / CN_ROOT_TARGET + f_wood / CN_WOOD_TARGET
        ) * is_land
        d_n_net = torch.clamp(d_n_gross - n_retrans, min=0.0)

        # First-principles soil solute-transport N uptake (Stocker & Prentice, 2024):
        # (a) transpiration mass flow + (b) fine-root diffusive uptake
        n_min_avail = torch.clamp(lc.n_min, min=0.0) * is_land
        # Root-zone liquid water column [kg m^-2] (layers 0..2 = top 0.60 m)
        w_root_col = torch.clamp(
            (w_liq_soil[:3] * dz[:3] * 1000.0).sum(dim=0),
            min=15.0,
        )
        e_tr = torch.clamp(transp_kgm2s, min=0.0) if transp_kgm2s is not None else torch.zeros_like(pos_in)
        f_mass = -torch.expm1(-torch.clamp((e_tr * dt) / w_root_col, max=0.5))
        u_mass = f_mass * n_min_avail

        root_eff = torch.clamp(lc.c_root, min=0.0) / (torch.clamp(lc.c_root, min=0.0) + N_ROOT_HALF_SAT)
        k_diff = N_UPTAKE_K_YR * f_moist[0] * f_freeze[0] * root_eff
        f_diff = -torch.expm1(-torch.clamp(k_diff * dt_yr, max=20.0))
        u_diff = f_diff * torch.clamp(n_min_avail - u_mass, min=0.0)
        u_soil_pot = u_mass + u_diff
        u_soil = torch.minimum(d_n_net, u_soil_pot)

        # Optimal Symbiotic BNF (Fisher et al., 2010 FUN; Houlton et al., 2008)
        d_n_rem = torch.clamp(d_n_net - u_soil, min=0.0)
        cost_fix = compute_bnf_carbon_cost(t0_c)  # [kg C / kg N]
        c_bnf_cap = BNF_MAX_NPP_FRAC * pos_in * is_land
        bnf_sym = torch.minimum(d_n_rem, c_bnf_cap / cost_fix)
        c_bnf_spent = bnf_sym * cost_fix  # Paid out of positive NPP surplus -> added to R_a (exact C conservation!)

        # Asymbiotic free-living microbial BNF entering soil mineral N directly
        f_t_bnf = torch.clamp(5.0 / cost_fix, 0.0, 1.0)
        bnf_asym = (BNF_ASYM_MAX_YR * dt_yr) * f_t_bnf * f_moist[0] * f_freeze[0] * is_land

        # Update N sufficiency EMA phi_N in [0.15, 1.0] (1.0 when N demand is zero/nighttime)
        n_supply_tot = n_retrans + u_soil + bnf_sym
        has_demand = d_n_gross > 1.0e-12
        phi_inst = torch.where(
            has_demand,
            torch.clamp(n_supply_tot / torch.clamp(d_n_gross, min=1.0e-12), 0.15, 1.0),
            phi_prev,
        )
        alpha_phi = -math.expm1(-dt / (PHI_N_EMA_DAYS * C.SECONDS_PER_DAY))
        phi_new = torch.where(is_land > 0.5, (1.0 - alpha_phi) * phi_prev + alpha_phi * phi_inst, torch.ones_like(phi_prev))
        lc.phi_n = torch.clamp(phi_new, 0.15, 1.0)
        if bio is not None:
            bio.n_avail_fac = lc.phi_n.clone()

        # Allocate acquired plant N proportionally across (leaf, root, wood) demands (exact N conservation!)
        w_nl = f_leaf / cn_leaf_opt
        w_nr = f_root / CN_ROOT_TARGET
        w_nw = f_wood / CN_WOOD_TARGET
        w_nsum = torch.clamp(w_nl + w_nr + w_nw, min=1.0e-8)
        n_in_leaf = n_supply_tot * (w_nl / w_nsum)
        n_in_root = n_supply_tot * (w_nr / w_nsum)
        n_in_wood = n_supply_tot - n_in_leaf - n_in_root

        # Subtract C spent on symbiotic BNF from pos_in and add to autotrophic respiration ra_eff!
        pos_in = torch.clamp(pos_in - c_bnf_spent, min=0.0)
        ra_eff = ra_eff + c_bnf_spent * inv_dt

    npp_eff = gpp - ra_eff
    in_leaf = f_leaf * pos_in - neg_capped * (lc.c_leaf / live_tot)
    in_wood = f_wood * pos_in - neg_capped * (lc.c_wood / live_tot)
    in_root = f_root * pos_in - neg_capped * (lc.c_root / live_tot)

    # Apply carbon growth & turnover with Neumaier/Kahan summation
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

    # 4. Litter & Vertically Resolved 5-Layer Soil / Permafrost C & N Decomposition (Koven et al., 2013)
    env0 = env_soil[0]
    k_lf = -torch.expm1(-torch.clamp((dt_yr / TAU_LIT_FAST_YR) * env0, max=20.0))
    k_ls = -torch.expm1(-torch.clamp((dt_yr / TAU_LIT_SLOW_YR) * env0, max=20.0))
    dec_lf = k_lf * torch.clamp(c_lit_fast_m, min=0.0)
    dec_ls = k_ls * torch.clamp(c_lit_slow_m, min=0.0)
    c_lit_fast_1, lf_comp = _kahan_add(c_lit_fast_m, -dec_lf, lf_comp)
    c_lit_slow_1, ls_comp = _kahan_add(c_lit_slow_m, -dec_ls, ls_comp)

    rh_lit_kg = (1.0 - HUMIF_FRAC_FAST) * dec_lf + (1.0 - HUMIF_FRAC_SLOW) * dec_ls
    humif_in = HUMIF_FRAC_FAST * dec_lf + HUMIF_FRAC_SLOW * dec_ls

    d_soil_in = torch.zeros_like(lc.c_soil)
    d_soil_in[0] = 0.70 * humif_in
    d_soil_in[1] = 0.30 * humif_in

    depth_fac = torch.exp(-z_mid / SOIL_Z_TAU_M)
    k_soil = -torch.expm1(-torch.clamp((dt_yr / TAU_SOIL_BASE_YR) * depth_fac * env_soil, max=20.0))
    c_soil_m, soil_comp = _kahan_add(lc.c_soil, d_soil_in, lc.c_soil_comp)
    dec_soil = k_soil * torch.clamp(c_soil_m, min=0.0)
    c_soil_1, soil_comp = _kahan_add(c_soil_m, -dec_soil, soil_comp)
    rh_soil_kg = dec_soil.sum(dim=0)

    # Conservative vertical cryoturbation / bioturbation mixing across the 5 soil layers
    c_conc = torch.clamp(c_soil_1, min=0.0) / dz
    dz_int = 0.5 * (dz[:-1] + dz[1:])
    f_mix = -(SOIL_CRYO_DIFF_M2_YR * dt_yr) * (c_conc[1:] - c_conc[:-1]) / dz_int
    f_cap = 0.25 * torch.minimum(torch.clamp(c_soil_1[:-1], min=0.0), torch.clamp(c_soil_1[1:], min=0.0))
    f_mix = torch.clamp(f_mix, -f_cap, f_cap)
    z_row = torch.zeros_like(c_soil_1[0:1])
    d_mix = (torch.cat([z_row, f_mix], dim=0) - torch.cat([f_mix, z_row], dim=0)) * is_land.unsqueeze(0)
    c_soil_new, soil_comp = _kahan_add(c_soil_1, d_mix, soil_comp)

    # Advance soil & litter Nitrogen pools in lockstep (exact conservation!)
    if enable_nitrogen:
        n_leaf_1, nl_comp = _kahan_add(lc.n_leaf, n_in_leaf - nturn_l_tot, lc.n_leaf_comp)
        n_root_1, nr_comp = _kahan_add(lc.n_root, n_in_root - nturn_r_tot, lc.n_root_comp)
        n_wood_1, nw_comp = _kahan_add(lc.n_wood, n_in_wood - nturn_w_tot, lc.n_wood_comp)

        n_lf_m, nlf_comp = _kahan_add(lc.n_lit_fast, nturn_l_lit + nturn_r_lit, lc.n_lit_fast_comp)
        n_ls_m, nls_comp = _kahan_add(lc.n_lit_slow, nturn_w_tot, lc.n_lit_slow_comp)
        dec_nlf = k_lf * torch.clamp(n_lf_m, min=0.0)
        dec_nls = k_ls * torch.clamp(n_ls_m, min=0.0)
        n_lf_1, nlf_comp = _kahan_add(n_lf_m, -dec_nlf, nlf_comp)
        n_ls_1, nls_comp = _kahan_add(n_ls_m, -dec_nls, nls_comp)

        # Litter N released -> humified soil organic N (targeting CN_SOIL_TARGET) + mineral N net flux
        n_lit_rel = dec_nlf + dec_nls
        n_humif_target = humif_in / CN_SOIL_TARGET
        # If litter N < target, immobilize from available n_min (after uptake) without going negative
        n_min_post_up = torch.clamp(lc.n_min - u_soil, min=0.0) * is_land
        n_immob_need = torch.clamp(n_humif_target - n_lit_rel, min=0.0)
        n_immob = torch.minimum(n_immob_need, 0.5 * n_min_post_up)
        n_humif_actual = torch.where(n_lit_rel >= n_humif_target, n_humif_target, n_lit_rel + n_immob)
        n_min_from_lit = n_lit_rel - n_humif_actual  # Positive = mineralization, negative = -n_immob

        d_nsoil_in = torch.zeros_like(lc.n_soil)
        d_nsoil_in[0] = 0.70 * n_humif_actual
        d_nsoil_in[1] = 0.30 * n_humif_actual

        n_soil_m, nsoil_comp = _kahan_add(lc.n_soil, d_nsoil_in, lc.n_soil_comp)
        dec_nsoil = k_soil * torch.clamp(n_soil_m, min=0.0)
        n_soil_1, nsoil_comp = _kahan_add(n_soil_m, -dec_nsoil, nsoil_comp)
        n_min_from_soil = dec_nsoil.sum(dim=0)

        # Conservative cryoturbation mixing of 5-layer soil N
        n_conc = torch.clamp(n_soil_1, min=0.0) / dz
        fn_mix = -(SOIL_CRYO_DIFF_M2_YR * dt_yr) * (n_conc[1:] - n_conc[:-1]) / dz_int
        fn_cap = 0.25 * torch.minimum(torch.clamp(n_soil_1[:-1], min=0.0), torch.clamp(n_soil_1[1:], min=0.0))
        fn_mix = torch.clamp(fn_mix, -fn_cap, fn_cap)
        dn_mix = (torch.cat([z_row, fn_mix], dim=0) - torch.cat([fn_mix, z_row], dim=0)) * is_land.unsqueeze(0)
        n_soil_new, nsoil_comp = _kahan_add(n_soil_1, dn_mix, nsoil_comp)

        # Atmospheric LNOx deposition + asymbiotic BNF + mineralization - uptake -> mineral N pool
        n_dep_in = (torch.clamp(n_dep_kgn_m2s, min=0.0) * dt * is_land) if n_dep_kgn_m2s is not None else torch.zeros_like(pos_in)
        dn_min_bio = n_dep_in + bnf_asym + n_min_from_lit + n_min_from_soil - u_soil
        n_min_1, nmin_comp = _kahan_add(lc.n_min, dn_min_bio, lc.n_min_comp)

        # Leaching (wet soil drainage) & anaerobic denitrification (waterlogged warm soils)
        n_min_pos = torch.clamp(n_min_1, min=0.0) * is_land
        wet_leach = torch.clamp((sat_frac[0] - 0.35) / 0.65, 0.0, 1.0) * f_freeze[0]
        wet_denit = torch.clamp((sat_frac[0] - 0.65) / 0.35, 0.0, 1.0) * f_t_soil[0] * f_freeze[0]
        k_leach = N_LEACH_RATE_YR * wet_leach
        k_denit = N_DENIT_RATE_YR * wet_denit
        k_loss_tot = k_leach + k_denit
        f_loss = -torch.expm1(-torch.clamp(k_loss_tot * dt_yr, max=20.0))
        n_loss_tot = f_loss * n_min_pos
        n_leach = torch.where(k_loss_tot > 1.0e-12, n_loss_tot * (k_leach / torch.clamp(k_loss_tot, min=1.0e-12)), torch.zeros_like(n_loss_tot))
        n_denit = n_loss_tot - n_leach
        n_min_2, nmin_comp = _kahan_add(n_min_1, -n_loss_tot, nmin_comp)

    # 5. Process-Based MAML SPITFIRE Wildfire (Thonicke et al., 2010; Rothermel, 1972; Byram, 1959)
    t_air_c = torch.clamp(t_sfc_k - C.T_FREEZE, min=-40.0, max=55.0)
    dp_dep = torch.clamp(vpd_pa / 160.0, 0.0, 35.0)
    di_inc = torch.clamp(t_air_c, min=0.0) * dp_dep * (dt / C.SECONDS_PER_DAY)
    wet_reset = (precip_kgm2s * C.SECONDS_PER_DAY > 3.0) | (f_snow > 0.20)
    di_new = torch.where(wet_reset, torch.zeros_like(lc.fire_di), torch.clamp(lc.fire_di + di_inc, 0.0, FIRE_DI_MAX)) * is_land
    lc.fire_di = di_new

    fdi = -torch.expm1(-FIRE_DI_K * di_new)
    fc4_pos = torch.clamp(f_c4, 0.0, 1.0) if f_c4 is not None else torch.zeros_like(fdi)
    slope_pos = torch.clamp(slope_tan, 0.0, 0.75) if slope_tan is not None else torch.zeros_like(fdi)

    # Multi-class fuel bed (Thonicke et al., 2010): 1h fine litter + dry-cured C4 grass vs. 100h coarse woody litter
    f_cure = torch.clamp(0.25 + 0.75 * fdi, 0.25, 1.0)
    c_grass_cured = f_cure * fc4_pos * torch.clamp(c_leaf_1, min=0.0)
    c_fuel_1h = torch.clamp(c_lit_fast_1, min=0.0) + c_grass_cured
    c_fuel_100h = torch.clamp(c_lit_slow_1, min=0.0)
    fuel_tot = c_fuel_1h + c_fuel_100h
    w_1h = c_fuel_1h / torch.clamp(fuel_tot, min=1.0e-6)
    fuel_surf = c_fuel_1h + 0.65 * c_fuel_100h
    f_fuel = torch.clamp((fuel_surf - 0.08) / 0.38, 0.0, 1.0)

    # Differential Rothermel (1972) fuel moisture damping (1h fine/grass dries faster than 100h woody litter)
    r_m_1h = torch.clamp(1.0 - fdi * (1.0 - 0.30 * sat_frac[0]), 0.04, 1.0)
    r_m_100h = torch.clamp(1.0 - 0.72 * fdi * (1.0 - 0.55 * sat_frac[0]), 0.08, 1.0)
    r_moist = w_1h * r_m_1h + (1.0 - w_1h) * r_m_100h
    eta_m = torch.clamp(1.0 - 2.59 * r_moist + 5.11 * (r_moist ** 2) - 3.52 * (r_moist ** 3), 0.0, 1.0)
    eta_m_100h = torch.clamp(1.0 - 2.59 * r_m_100h + 5.11 * (r_m_100h ** 2) - 3.52 * (r_m_100h ** 3), 0.0, 1.0)

    # Fuel-bed baseline ROS (fast cured savanna grass ~8.5 m/min vs. forest litter ~3.2 m/min), wind & slope (Rothermel Eq. 51)
    grass_ros_w = torch.clamp(0.65 * fc4_pos + 0.35 * w_1h * (1.0 - torch.tanh(hc_pos / 4.0)), 0.0, 1.0)
    ros_base = 3.2 + 5.3 * grass_ros_w
    u_fwd_ms = torch.clamp(u10_ms, min=0.0) * (1.0 - 0.45 * torch.tanh(hc_pos / 8.0))
    u_fwd_mmin = u_fwd_ms * 60.0
    phi_w = 0.05 * torch.pow(torch.clamp(u_fwd_mmin, max=1200.0), 0.85)
    phi_s = 3.5 * (slope_pos * slope_pos)
    ros_f = torch.clamp(ros_base * eta_m * (1.0 + phi_w + phi_s) * f_fuel, 0.0, 60.0)
    ros_b = ros_f * torch.exp(-0.012 * torch.clamp(u_fwd_mmin, max=500.0))

    # Wind-dependent ellipse length-to-breadth ratio L_B & daily fire duration t_fire (Thonicke et al., 2010)
    l_b = 1.0 + 8.729 * torch.pow(1.0 - torch.exp(-0.03 * torch.clamp(u_fwd_ms, max=30.0)), 2.155)
    t_fire_min = 241.0 / (1.0 + 240.0 * torch.exp(-11.06 * fdi))
    d_major_m = (ros_f + ros_b) * t_fire_min
    a_fire_m2 = torch.clamp((0.25 * math.pi / l_b) * (d_major_m * d_major_m), max=crm_dx_m * crm_dx_m)

    # Byram (1959) fireline intensity I_B [kW m^-1] & self-extinction threshold (50 kW m^-1)
    # Factor 2.0 converts kg C -> kg dry biomass; includes cured 1h grass/fine litter + moisture-damped 100h coarse litter
    fc_dm = 2.0 * (0.80 * c_fuel_1h + 0.35 * eta_m_100h * c_fuel_100h)
    i_byram = FIRE_HEAT_CONTENT_KJ * fc_dm * (ros_f / 60.0)
    can_sustain = (i_byram >= FIRE_EXTINCT_KW_M).to(dtype) * is_land

    # Van Wagner (1973) / Thonicke et al. (2010) scorch height SH & bark-thickness woody mortality
    sh_m = FIRE_SCORCH_COEF * torch.pow(torch.clamp(i_byram, min=0.0), 2.0 / 3.0) * can_sustain
    f_ck = torch.clamp((sh_m - 0.35 * hc_pos) / torch.clamp(0.65 * hc_pos, min=0.5), 0.0, 1.0)
    p_crown = 0.85 * (f_ck * f_ck * f_ck)
    # Emergent bark thickness increases allometrically with tree height (tall trees survive mild surface fires)
    d_bark_cm = 0.8 + 1.8 * torch.pow(hc_pos / 10.0, 0.6)
    tau_l_min = 1.5 + 0.002 * torch.clamp(i_byram, max=5000.0)
    tau_c_min = 2.9 * (d_bark_cm * d_bark_cm)
    r_camb = tau_l_min / torch.clamp(tau_c_min, min=0.5)
    p_camb = torch.clamp((r_camb - 0.22) / 1.78, 0.0, 1.0)
    p_mort = torch.clamp(1.0 - (1.0 - p_crown) * (1.0 - p_camb), 0.0, 1.0) * can_sustain

    # Cloud-to-ground lightning ignitions [ignitions m^-2] & continuous SPITFIRE burn probability
    # In MAML subgrid mode, thunderstorms emit anvil/gust-front CG strikes ("dry lightning") across
    # adjacent dry subcolumns of the same parent grid cell, not solely inside the wet rain shaft
    # where precip > 3 mm/d resets FDI to zero. Blending 50% local + 50% parent-cell mean strictly
    # conserves total parent-cell CG strikes while enabling realistic dry-lightning ignition.
    use_maml_patch = (maml_shape is not None) and (maml_shape[1] * maml_shape[2] > 1)
    cg_raw = torch.clamp(flash_cg_m2s, min=0.0) if flash_cg_m2s is not None else torch.full_like(fdi, FIRE_BG_CG_FLASH_M2S)
    if use_maml_patch and flash_cg_m2s is not None:
        assert maml_shape is not None
        nc_cg, ny_cg, nx_cg = maml_shape
        cg_3d = cg_raw.view(nc_cg, ny_cg, nx_cg)
        cg_cell_mean = cg_3d.reshape(nc_cg, ny_cg * nx_cg).mean(dim=-1, keepdim=True).expand(nc_cg, ny_cg * nx_cg).reshape(-1)
        cg_rate = 0.50 * cg_raw + 0.50 * cg_cell_mean
    else:
        cg_rate = cg_raw
    n_ig = FIRE_IGNIT_EFF * cg_rate * fdi * dt
    rate_raw_s = torch.minimum(n_ig * a_fire_m2 * inv_dt, torch.full_like(fdi, FIRE_MAX_RATE_YR / SEC_PER_YEAR)) * can_sustain
    p_burn_cont = -torch.expm1(-torch.clamp(rate_raw_s * dt, min=0.0, max=0.25))

    # Discrete MAML patch burning + 2D neighbor fire spread when running on subcolumns (nc, ny, nx)
    if use_maml_patch:
        assert maml_shape is not None
        nc_m, ny_m, nx_m = maml_shape
        assert lc.fire_accum is not None
        acc_new = (lc.fire_accum + p_burn_cont) * is_land
        did_burn = (acc_new >= 1.0) & (can_sustain > 0.5)
        # 2D periodic CRM neighbor spread when forward spread reaches adjacent dry subcolumns
        spread_reach = (d_major_m >= 0.40 * crm_dx_m) & did_burn
        if ny_m > 1 or nx_m > 1:
            sr_3d = spread_reach.view(nc_m, ny_m, nx_m)
            nb_fire = torch.roll(sr_3d, 1, dims=-1) | torch.roll(sr_3d, -1, dims=-1)
            if ny_m > 1:
                nb_fire = nb_fire | torch.roll(sr_3d, 1, dims=-2) | torch.roll(sr_3d, -1, dims=-2)
            dry_nb = nb_fire.reshape(-1) & (can_sustain > 0.5) & (fdi > 0.25) & (acc_new >= 0.50)
            did_burn = did_burn | dry_nb
        # Reset hazard accumulator on burnt patches (retain fractional remainder only when unburnt)
        lc.fire_accum = torch.where(did_burn, torch.zeros_like(acc_new), torch.clamp(acc_new, 0.0, 0.9999))
        f_burn = did_burn.to(dtype)
        lc.patch_burned_flag = f_burn
    else:
        f_burn = p_burn_cont
        lc.patch_burned_flag = (f_burn > 1.0e-6).to(dtype)

    lc.fire_intensity_kw_m = i_byram * can_sustain
    lc.fire_scorch_m = sh_m
    lc.fire_mort_frac = p_mort

    # Combust fine (1h) litter, coarse (100h) litter, foliage, and mortality-weighted wood (exact C & N conservation!)
    wood_comb_eff = FIRE_COMBUST_WOOD * (0.25 + 0.75 * p_mort)
    leaf_comb_eff = FIRE_COMBUST_LEAF * (0.30 + 0.70 * p_mort)
    burn_lf = (f_burn * 0.75) * torch.clamp(c_lit_fast_1, min=0.0)
    burn_ls = (f_burn * 0.40) * torch.clamp(c_lit_slow_1, min=0.0)
    burn_leaf = (f_burn * leaf_comb_eff) * torch.clamp(c_leaf_1, min=0.0)
    burn_wood = (f_burn * wood_comb_eff) * torch.clamp(c_wood_1, min=0.0)

    c_lit_fast_new, lf_comp = _kahan_add(c_lit_fast_1, -burn_lf, lf_comp)
    c_lit_slow_new, ls_comp = _kahan_add(c_lit_slow_1, -burn_ls, ls_comp)
    c_leaf_new, leaf_comp = _kahan_add(c_leaf_1, -burn_leaf, leaf_comp)
    c_wood_new, wood_comp = _kahan_add(c_wood_1, -burn_wood, wood_comp)

    fire_c_kg = burn_lf + burn_ls + burn_leaf + burn_wood
    fire_c_flux = fire_c_kg * inv_dt
    rh_flux = (rh_lit_kg + rh_soil_kg) * inv_dt

    # Direct EEO vegetation structure feedback on burnt MAML patches (forest-savanna bistability!)
    if enable_fire_veg_feedback and bio is not None:
        # On burnt patches, crown mortality reduces LAI & canopy height, opening the understory to C4 grass!
        lai_drop = torch.clamp(f_burn * (0.35 + 0.55 * p_mort), 0.0, 0.92)
        hc_drop = torch.clamp(f_burn * (0.65 * p_mort), 0.0, 0.85)
        if bio.lai_evg is not None and bio.lai_dec is not None:
            bio.lai_evg = torch.where(is_land > 0.5, bio.lai_evg * (1.0 - lai_drop), bio.lai_evg)
            bio.lai_dec = torch.where(is_land > 0.5, bio.lai_dec * (1.0 - lai_drop), bio.lai_dec)
            bio.lai = torch.where(is_land > 0.5, torch.clamp(bio.lai_evg + bio.lai_dec, 0.0, 8.0), bio.lai)
        else:
            bio.lai = torch.where(is_land > 0.5, torch.clamp(bio.lai * (1.0 - lai_drop), 0.0, 8.0), bio.lai)
        bio.h_canopy = torch.where(
            is_land > 0.5,
            torch.clamp(bio.h_canopy * (1.0 - hc_drop), min=0.10),
            bio.h_canopy,
        )

    # Pyrogenic nitrogen volatilization + post-fire ash N return to soil mineral pool
    if enable_nitrogen:
        burn_nlf = (f_burn * 0.75) * torch.clamp(n_lf_1, min=0.0)
        burn_nls = (f_burn * 0.40) * torch.clamp(n_ls_1, min=0.0)
        burn_nleaf = (f_burn * leaf_comb_eff) * torch.clamp(n_leaf_1, min=0.0)
        burn_nwood = (f_burn * wood_comb_eff) * torch.clamp(n_wood_1, min=0.0)

        n_lf_new, nlf_comp = _kahan_add(n_lf_1, -burn_nlf, nlf_comp)
        n_ls_new, nls_comp = _kahan_add(n_ls_1, -burn_nls, nls_comp)
        n_leaf_new, nl_comp = _kahan_add(n_leaf_1, -burn_nleaf, nl_comp)
        n_wood_new, nw_comp = _kahan_add(n_wood_1, -burn_nwood, nw_comp)

        burn_n_tot = burn_nlf + burn_nls + burn_nleaf + burn_nwood
        n_fire_vol = FIRE_N_VOL_FRAC * burn_n_tot
        n_fire_ash = burn_n_tot - n_fire_vol
        n_min_new, nmin_comp = _kahan_add(n_min_2, n_fire_ash, nmin_comp)

        lc.n_leaf = n_leaf_new * is_land
        lc.n_root = n_root_1 * is_land
        lc.n_wood = n_wood_new * is_land
        lc.n_lit_fast = n_lf_new * is_land
        lc.n_lit_slow = n_ls_new * is_land
        lc.n_soil = n_soil_new * is_land.unsqueeze(0)
        lc.n_min = n_min_new * is_land

        lc.n_leaf_comp = nl_comp * is_land
        lc.n_root_comp = nr_comp * is_land
        lc.n_wood_comp = nw_comp * is_land
        lc.n_lit_fast_comp = nlf_comp * is_land
        lc.n_lit_slow_comp = nls_comp * is_land
        lc.n_soil_comp = nsoil_comp * is_land.unsqueeze(0)
        lc.n_min_comp = nmin_comp * is_land

        lc.bnf_kgn_m2s = (bnf_sym + bnf_asym) * inv_dt
        lc.n_uptake_kgn_m2s = u_soil * inv_dt
        lc.n_dep_kgn_m2s = n_dep_in * inv_dt
        lc.n_leach_kgn_m2s = n_leach * inv_dt
        lc.n_denit_kgn_m2s = n_denit * inv_dt
        lc.n_fire_kgn_m2s = n_fire_vol * inv_dt

    # Commit updated carbon state & diagnostics
    lc.c_leaf = c_leaf_new * is_land
    lc.c_wood = c_wood_new * is_land
    lc.c_root = c_root_1 * is_land
    lc.c_lit_fast = c_lit_fast_new * is_land
    lc.c_lit_slow = c_lit_slow_new * is_land
    lc.c_soil = c_soil_new * is_land.unsqueeze(0)
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


def diagnose_land_carbon_metrics(
    lc: LandCarbonState,
    area_cell: torch.Tensor,
    surf_type: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute global terrestrial C [Pg C, Pg C/yr] and N [Pg N, Tg N/yr] stocks & fluxes in float64.

    Handles both host-cell `[N_cells]` and MAML subcolumn `[N_cells * Ny * Nx]` shapes automatically.
    """
    area_c = area_cell.detach().cpu().to(torch.float64)
    nc = area_c.shape[0]
    m_tot = lc.c_leaf.shape[0]
    if m_tot != nc and (m_tot % nc == 0):
        n_sub = m_tot // nc
        area = (area_c / float(n_sub)).unsqueeze(1).expand(nc, n_sub).reshape(m_tot)
    else:
        area = area_c

    kg_to_pg = 1.0e-12
    kg_s_to_pg_yr = 1.0e-12 * SEC_PER_YEAR
    kg_s_to_tgn_yr = 1.0e-9 * SEC_PER_YEAR

    def _flux_pg(t: torch.Tensor) -> float:
        return float((t.detach().cpu().to(torch.float64) * area).sum().item() * kg_s_to_pg_yr)

    def _flux_tgn(t: torch.Tensor) -> float:
        return float((t.detach().cpu().to(torch.float64) * area).sum().item() * kg_s_to_tgn_yr)

    c_leaf_64 = lc.c_leaf.detach().cpu().to(torch.float64) + lc.c_leaf_comp.detach().cpu().to(torch.float64)
    c_root_64 = lc.c_root.detach().cpu().to(torch.float64) + lc.c_root_comp.detach().cpu().to(torch.float64)
    c_wood_64 = lc.c_wood.detach().cpu().to(torch.float64) + lc.c_wood_comp.detach().cpu().to(torch.float64)
    c_lf_64 = lc.c_lit_fast.detach().cpu().to(torch.float64) + lc.c_lit_fast_comp.detach().cpu().to(torch.float64)
    c_ls_64 = lc.c_lit_slow.detach().cpu().to(torch.float64) + lc.c_lit_slow_comp.detach().cpu().to(torch.float64)
    c_soil_64 = (lc.c_soil.detach().cpu().to(torch.float64) + lc.c_soil_comp.detach().cpu().to(torch.float64)).sum(dim=0)
    c_veg = float(((c_leaf_64 + c_root_64 + c_wood_64) * area).sum().item() * kg_to_pg)
    c_lit = float(((c_lf_64 + c_ls_64) * area).sum().item() * kg_to_pg)
    c_soil = float((c_soil_64 * area).sum().item() * kg_to_pg)

    out = {
        "c_veg_pgc": c_veg,
        "c_litter_pgc": c_lit,
        "c_soil_pgc": c_soil,
        "c_land_total_pgc": c_veg + c_lit + c_soil,
        "land_gpp_pgc_yr": _flux_pg(lc.gpp_kgc_m2s),
        "land_npp_pgc_yr": _flux_pg(lc.npp_kgc_m2s),
        "land_rh_pgc_yr": _flux_pg(lc.rh_kgc_m2s),
        "land_fire_pgc_yr": _flux_pg(lc.fire_c_kgc_m2s),
        "land_nbp_pgc_yr": _flux_pg(lc.nbp_kgc_m2s),
        # Burned area [Mha yr^-1] (1 Mha = 1e10 m^2)
        "burned_area_mha_yr": float((lc.burned_frac_s.detach().cpu().to(torch.float64) * area).sum().item() * SEC_PER_YEAR * 1.0e-10),
    }

    if lc.n_leaf is not None and lc.n_min is not None:
        nl = lc.n_leaf.detach().cpu().to(torch.float64) + lc.n_leaf_comp.detach().cpu().to(torch.float64)
        nr = lc.n_root.detach().cpu().to(torch.float64) + lc.n_root_comp.detach().cpu().to(torch.float64)
        nw = lc.n_wood.detach().cpu().to(torch.float64) + lc.n_wood_comp.detach().cpu().to(torch.float64)
        nlf = lc.n_lit_fast.detach().cpu().to(torch.float64) + lc.n_lit_fast_comp.detach().cpu().to(torch.float64)
        nls = lc.n_lit_slow.detach().cpu().to(torch.float64) + lc.n_lit_slow_comp.detach().cpu().to(torch.float64)
        ns = (lc.n_soil.detach().cpu().to(torch.float64) + lc.n_soil_comp.detach().cpu().to(torch.float64)).sum(dim=0)
        nm = lc.n_min.detach().cpu().to(torch.float64) + lc.n_min_comp.detach().cpu().to(torch.float64)
        n_veg = float(((nl + nr + nw) * area).sum().item() * kg_to_pg)
        n_lit = float(((nlf + nls) * area).sum().item() * kg_to_pg)
        n_soil = float((ns * area).sum().item() * kg_to_pg)
        n_min = float((nm * area).sum().item() * kg_to_pg)
        out.update(
            {
                "n_veg_pgn": n_veg,
                "n_soil_pgn": n_soil,
                "n_min_pgn": n_min,
                "n_land_total_pgn": n_veg + n_lit + n_soil + n_min,
                "bnf_tgn_yr": _flux_tgn(lc.bnf_kgn_m2s),
                "n_uptake_tgn_yr": _flux_tgn(lc.n_uptake_kgn_m2s),
                "n_dep_tgn_yr": _flux_tgn(lc.n_dep_kgn_m2s),
                "n_leach_tgn_yr": _flux_tgn(lc.n_leach_kgn_m2s),
                "n_denit_tgn_yr": _flux_tgn(lc.n_denit_kgn_m2s),
                "n_fire_tgn_yr": _flux_tgn(lc.n_fire_kgn_m2s),
            }
        )
        if lc.phi_n is not None and lc.f_root_alloc is not None:
            if surf_type is not None:
                st_c = surf_type.detach().cpu()
                if st_c.shape[0] != m_tot and (m_tot % st_c.shape[0] == 0):
                    st_m = st_c.unsqueeze(1).expand(nc, m_tot // nc).reshape(m_tot)
                else:
                    st_m = st_c
                w_l = area * (st_m == SURF_LAND).to(torch.float64)
            else:
                w_l = area * (c_leaf_64 + c_soil_64 > 1.0e-6).to(torch.float64)
            wl_sum = max(float(w_l.sum().item()), 1.0)
            out["phi_n_mean"] = float((lc.phi_n.detach().cpu().to(torch.float64) * w_l).sum().item()) / wl_sum
            out["f_root_mean"] = float((lc.f_root_alloc.detach().cpu().to(torch.float64) * w_l).sum().item()) / wl_sum
    return out
