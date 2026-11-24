"""Unified Land-Snow-Ice-Ocean surface coupling, multi-layer freeze-thaw soil, and sea ice.

Implements every surface physical process identified in our MIROC6 / MATSIRO6 / SCALE-RM audit
with zero external PFT, soil-texture, or albedo lookup maps:

1. Fractional Snow Cover, Snow Burial, Aging Snow Albedo, & Glacier Runoff Cap:
   - Swenson & Lawrence (2012, J. Geophys. Res., 117, D21107, Eqs. 3-6) &
     Nitta et al. (2014, J. Climate, 27, 3318-3330):
     Subgrid topographic/hysteretic fractional snow-covered area `A_sn`.
   - MATSIRO6 Document Writing Team (2021, CCSR Report No. 66, Ch. 2 Eqs. 2.1-2.3) &
     Abolafia-Rosenzweig et al. (2021, JAMES, 13, e2021MS002665):
     Snow-burial masking of canopy height `h_exp` and leaf area `L_exp`.
   - Yang et al. (1997, J. Climate, 10, 353-373) & MATSIRO6 (2021, Eq. 3.1, Sec. 8.6 Eq. 8.44,
     & Sec. 8.8.1 Eqs. 8.58-8.62):
     Two-band (VIS/NIR) prognostic snow aging, solar-zenith dependence, and `1000 kg m^-2`
     glacier runoff cap.
2. Map-Free Two-Band Albedo & Three-Component Evapotranspiration:
   - Oleson et al. (2013, CLM4.5 Tech. Note, Sec. 3.2) & Lee & Khairoutdinov (2015, Eq. 21):
     Moisture-dependent bare-soil VIS & NIR albedo (bright dry desert sand -> dark wet soil).
   - Dai, Dickinson, & Wang (2004, J. Climate, 17, 2281-2299, Eqs. 5-9 & App. A):
     Two-Big-Leaf canopy radiative transfer and sunlit/shaded foliage energy balance.
   - Sakaguchi & Zeng (2009, J. Geophys. Res., 114, D01107), Lee & Khairoutdinov (2015,
     SLM Eqs. 5-16, 24-28 & App. A Eqs. A8-A14), & MATSIRO6 (2021, Sec. 5.4 & Ch. 7):
     Partitioning of evapotranspiration into sunlit/shaded transpiration, wet-canopy
     interception evaporation/sublimation, and stability-corrected bare-soil evaporation.
    - Lawrence et al. (2020, CLM5 Tech. Note, Eqs. 5.72-5.81, 5.109-5.113, 5.139, 7.2-7.23) &
      Swenson & Lawrence (2014): liquid-only tanh canopy interception, (W_can/W_max)^(2/3) wetted
      fraction, water-limited wet-canopy conductance, and dry-surface-layer soil resistance.
3. Five-Layer Soil Heat Conduction & Freeze-Thaw Permafrost Phase Change:
   - Nishizawa, Tomita, & Team SCALE (2025, SCALE-RM v5.5.5, Sec. 8.7.1 Eqs. 8.384-8.418),
     MATSIRO6 (2021, Ch. 10 Eqs. 10.1-10.27), & Clapp & Hornberger (1978, WRR, 14, 601-604):
     Implicit batched Thomas soil thermal diffusion down to 3.0 m depth with moisture-dependent
     thermal conductivity and exact enthalpy-conserving 0 degC liquid-ice phase change.
4. Ocean Surface Albedo, Slab Mixed Layer, & Fractional-Lead Thermodynamic Sea Ice:
   - Briegleb et al. (1986, J. Climate Appl. Meteorol., 25, 214-226, Eq. 11) &
     MIROC6 (2021, Sec. 3.8): Solar-zenith-dependent open-ocean direct/diffuse albedo.
   - Miller, Beljaars, & Palmer (1992, J. Climate, 5, 418-434): Viscous + Charnock roughness.
   - Semtner (1976, J. Phys. Oceanogr., 6, 379-389), Codron (2012, Clim. Dyn., 38, 379-389),
     & SCALE-RM (2025, Sec. 8.6 Eqs. 8.328-8.358):
     Slab ocean mixed layer with snowfall fusion cooling and fractional-lead sea ice `f_i`.

AOBA Formulation Reference: Sec. 7.3, Eqs. (7.7)-(7.10) (3-Source Energy Balance, Snow/SNICAR & 5-Layer Soil).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.biosphere import (
    BiosphereState,
    LEAF_SCAT_NIR,
    LEAF_SCAT_VIS,
    canopy_sunlit_sw_fraction,
    canopy_two_stream_albedo_dai,
    compute_optimal_soil_water_beta,
    compute_two_leaf_gas_exchange,
    init_biosphere_state,
    prepare_two_leaf_allocation,
)
from aoba.kernels import (
    can_use_elem_metal,
    dispatch_surface_albedos_metal,
    solve_tridiagonal_batched,
)
from aoba.runtime import maybe_cuda_compile
from aoba.thermodynamics import esat_ice, esat_liquid, qsat_mixed
from aoba.turbulence import (
    canopy_roughness_raupach,
    compute_surface_transfer,
    ocean_roughness_miller,
)

# ---------------------------------------------------------------------------
# 1. Universal Soil, Snow, & Sea-Ice Constants (MATSIRO6, SCALE-RM Sec. 8.6-8.7, CLM4.5)
# ---------------------------------------------------------------------------
SOIL_LAYER_DEPTHS: tuple[float, ...] = (0.05, 0.15, 0.40, 0.80, 1.60)  # 5 layers = 3.00 m total [m]
N_SOIL_LAYERS: int = len(SOIL_LAYER_DEPTHS)

# Universal mineral soil baseline (Clapp & Hornberger, 1978 medium loam; SCALE-RM Sec. 8.7.1)
RHO_WATER: float = 1000.0          # Liquid freshwater density [kg m^-3]
RHO_ICE_SOLID: float = 917.0       # Solid ice density [kg m^-3]
CP_LIQ_WATER: float = 4186.0       # Specific heat of liquid water c_l [J kg^-1 K^-1]
CP_ICE_SOLID: float = 2106.0       # Specific heat of solid ice c_i [J kg^-1 K^-1]
CS_DRY_SOIL: float = 1.26e6        # Volumetric heat capacity of dry mineral soil matrix C_s [J m^-3 K^-1]
KAPPA_DRY_SOIL: float = 0.30       # Dry mineral soil thermal conductivity [W m^-1 K^-1] (MATSIRO6 Eq. 10.4)
SOIL_POROSITY: float = 0.43        # Volumetric saturation porosity w_sat [m^3 m^-3] (Clapp & Hornberger, 1978)
PSI_SAT_SOIL: float = -0.20        # Saturated matric head psi_s [m] (Clapp & Hornberger, 1978)
B_CLAPP_SOIL: float = 5.39         # Clapp & Hornberger (1978) retention exponent b [-]
K_SAT_SOIL: float = 7.0e-6         # Saturated hydraulic conductivity K_sat [m s^-1] (Clapp & Hornberger, 1978, Table 2 loam)
ICE_IMPEDANCE_OMEGA: float = 6.0   # Frozen-soil hydraulic impedance exponent Omega [-] (Swenson et al., 2012; CLM5 Eq. 7.89)

# Canopy hydrology (CLM5 Tech. Note Sec. 7.1, Eqs. 7.4, 7.12, 7.23; Dickinson et al., 1993)
WC_MAX_PER_LAI: float = 0.10       # Maximum liquid canopy storage per unit LAI p_liq [kg m^-2] (CLM5 Eq. 7.12)
ALPHA_INTERCEPT_LIQ: float = 1.0   # Fraction of a leaf collecting rain alpha_liq in f_pi = alpha tanh(L) (CLM5 Eq. 7.4)
F_WET_MAX: float = 1.0             # Upper bound on wetted canopy fraction f_wet (CLM5 Eq. 7.23 caps at 1)

# Land skin energy-balance solver: number of fixed Newton passes (1 = single linearized pass; CLM5 Sec. 5.3 iterates to convergence)
# Three passes remove >95% of the single-pass error at +4% cost (10 passes: +27% cost for no visible gain in coupled runs).
LAND_EB_ITERATIONS: int = 3

# Dry-surface-layer (DSL) bare-soil evaporation resistance (Swenson & Lawrence, 2014; CLM5 Eqs. 5.72-5.81)
DSL_DMAX_MM: float = 15.0          # Maximum DSL thickness D_max [mm] (CLM5 Eq. 5.77)
DSL_THETA_INIT_FRAC: float = 0.8   # DSL initiation moisture theta_init as a fraction of porosity [-] (secondary-source value)
PSI_AIR_MM: float = 1.0e7          # Air-dry matric potential Psi_air [mm] (CLM5 Eq. 5.78)

# Snow & albedo constants (MATSIRO6 Ch. 8; Yang et al., 1997)
RHO_SNOWPACK: float = 250.0        # Representative bulk snowpack density [kg m^-3]
SWE_GLACIER_MAX: float = 1000.0    # Maximum SWE before glacier calving/runoff [kg m^-2] (MATSIRO6 Eq. 8.44)
TAU_SNOW_AGE: float = 1.0e6        # Snow aging timescale tau_age [s] (Yang et al., 1997; MATSIRO6 Eq. 8.60)
ALB_SNOW_VIS_NEW: float = 0.90     # Fresh snow visible albedo [-] (MATSIRO6 Sec. 8.8.1)
ALB_SNOW_NIR_NEW: float = 0.70     # Fresh snow near-IR albedo [-] (MATSIRO6 Sec. 8.8.1)
ALB_SNOW_VIS_OLD: float = 0.65     # Aged snow visible albedo [-] (MATSIRO6 Sec. 8.8.1)
ALB_SNOW_NIR_OLD: float = 0.35     # Aged snow near-IR albedo [-] (Yang et al., 1997)
EMISSIVITY_LAND: float = 0.98      # Thermal IR emissivity of vegetated/bare/snow land [-]
EMISSIVITY_OCEAN: float = 0.985    # Thermal IR emissivity of seawater/sea ice [-]

# Fractional-lead sea-ice constants (Semtner, 1976; SCALE-RM Sec. 8.6.2 Eq. 8.342)
MC_SEAICE_CRIT: float = 450.0      # Critical sea-ice mass for 100% ice cover (~0.49 m) [kg m^-2]
ALB_SEAICE_VIS: float = 0.72       # Bare sea-ice visible albedo [-] (Semtner, 1976; Ebert & Curry, 1993)
ALB_SEAICE_NIR: float = 0.42       # Bare sea-ice near-IR albedo [-]

# Surface type codes derived algebraically from Earth2014 (Hirt & Rexer, 2015)
SURF_OCEAN: int = 0                # Open ocean / shelf water (SUR - TBI > 0)
SURF_LAND: int = 1                 # Non-glaciated continental land (SUR == TBI, ICE == 0)
SURF_ICESHEET: int = 2             # Grounded continental ice sheet (SUR == TBI, ICE > 0)


# ---------------------------------------------------------------------------
# 2. Prognostic Surface State & Initialization
# ---------------------------------------------------------------------------
@dataclass
class SurfaceState:
    """Complete prognostic surface state across arbitrary tile grid shape `[...]`."""

    surf_type: torch.Tensor       # Integer mask: 0=ocean, 1=land, 2=grounded ice sheet `[...]`
    z_topo: torch.Tensor          # Surface elevation above sea level [m] `[...]`
    sigma_topo: torch.Tensor      # Subgrid topographic std-dev [m] `[...]`

    # Land & ice-sheet states
    t_soil: torch.Tensor          # 5-layer soil/firn temperatures `[N_SOIL, ...]` [K]
    w_liq_soil: torch.Tensor      # 5-layer volumetric liquid water `[N_SOIL, ...]` [m^3 m^-3]
    w_ice_soil: torch.Tensor      # 5-layer volumetric soil ice `[N_SOIL, ...]` [m^3 m^-3]
    w_root: torch.Tensor          # Prognostic root-zone available water bucket `W` [kg m^-2]
    w_canopy: torch.Tensor        # Intercepted canopy water/snow mass `W_c` [kg m^-2]
    t_canopy: torch.Tensor        # Effective foliage temperature [K]
    swe_land: torch.Tensor        # Land/glacier snow water equivalent `S_n` [kg m^-2]
    swe_max_season: torch.Tensor  # Seasonal peak SWE for melt hysteresis (Swenson & Lawrence, 2012) [kg m^-2]
    snow_age: torch.Tensor        # Nondimensional snow age `A_g` (Yang et al., 1997) [-]

    # Ocean & sea-ice states
    t_ocean_ml: torch.Tensor      # Ocean mixed-layer temperature `T_ml` [K]
    h_ocean_ml: torch.Tensor      # Ocean mixed-layer depth `H_ml` [m]
    u_star_ocean: torch.Tensor    # Previous-step ocean friction velocity `u_*` [m s^-1]
    m_seaice: torch.Tensor        # Sea-ice mass per unit cell area `M_i` [kg m^-2]
    t_seaice: torch.Tensor        # Sea-ice skin/interior temperature `T_i` [K]
    swe_seaice: torch.Tensor      # Snow water equivalent on sea ice [kg m^-2]

    # Embedded EEO biosphere state
    bio: BiosphereState
    t_ocean_ml_comp: torch.Tensor | None = None  # Kahan float32 rounding compensation for T_ml [K]
    # Optional 3-source warm-start temperatures (backward-compatible with older restart checkpoints)
    t_leaf_sun: torch.Tensor | None = None       # Sunlit leaf equilibrium temperature [K]
    t_leaf_sh: torch.Tensor | None = None        # Shaded leaf equilibrium temperature [K]
    t_gnd_skin: torch.Tensor | None = None       # Ground/snow skin temperature under canopy [K]
    # Optional SNICAR snow/sea-ice surface light-absorbing impurities (Flanner et al., 2007, 2009)
    bc_snow: torch.Tensor | None = None          # Top-2cm snow/ice black carbon load [kg BC m^-2]
    dust_snow: torch.Tensor | None = None        # Top-2cm snow/ice mineral dust load [kg dust m^-2]


def step_snow_impurity_darkening(
    sfc: SurfaceState,
    bc_dep_kgm2s: torch.Tensor | None,
    dust_dep_kgm2s: torch.Tensor | None,
    snow_fall_kgm2s: torch.Tensor,
    dt: float,
) -> None:
    """Update top-layer snow/sea-ice BC & dust impurities via exact exponential burial/scavenging (Flanner et al., 2007)."""
    has_cover = (sfc.swe_land > 0.5) | (sfc.swe_seaice > 0.5) | (sfc.m_seaice > 10.0)
    if sfc.bc_snow is None:
        sfc.bc_snow = torch.zeros_like(sfc.swe_land)
    if sfc.dust_snow is None:
        sfc.dust_snow = torch.zeros_like(sfc.swe_land)
    # Burial by fresh snow (15 kg m^-2 fresh SWE buries surface layer) + 45-day melt/sublimation washout
    k_bury = torch.clamp(snow_fall_kgm2s, min=0.0) / 15.0 + 1.0 / (45.0 * C.SECONDS_PER_DAY)
    decay = torch.exp(-torch.clamp(k_bury * dt, max=20.0))
    int_fac = torch.where(k_bury > 1.0e-12, -torch.expm1(-torch.clamp(k_bury * dt, max=20.0)) / torch.clamp(k_bury, min=1.0e-12), torch.full_like(k_bury, float(dt)))
    bc_in = torch.clamp(bc_dep_kgm2s, min=0.0) if bc_dep_kgm2s is not None else torch.zeros_like(sfc.swe_land)
    du_in = torch.clamp(dust_dep_kgm2s, min=0.0) if dust_dep_kgm2s is not None else torch.zeros_like(sfc.swe_land)
    sfc.bc_snow = torch.where(has_cover, torch.clamp(sfc.bc_snow * decay + bc_in * int_fac, 0.0, 5.0e-5), torch.zeros_like(sfc.bc_snow))
    sfc.dust_snow = torch.where(has_cover, torch.clamp(sfc.dust_snow * decay + du_in * int_fac, 0.0, 5.0e-2), torch.zeros_like(sfc.dust_snow))


def _kahan_add(val: torch.Tensor, delta: torch.Tensor, comp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate small `delta` onto `val` (~280 K) in float32 without swamping loss."""
    y = delta + comp
    t = val + y
    new_comp = torch.where(val.abs() >= y.abs(), (val - t) + y, (y - t) + val)
    return t, new_comp


@dataclass
class SurfaceFluxDiagnostics:
    """Atmospheric lower-boundary fluxes and radiative properties returned by `step_surface`."""

    shf: torch.Tensor             # Upward sensible heat flux `H` [W m^-2]
    lhf: torch.Tensor             # Upward latent heat flux `lambda * E` [W m^-2]
    evap: torch.Tensor            # Upward water-vapor mass flux `E` [kg m^-2 s^-1]
    tau_x: torch.Tensor           # Upward momentum flux (opposing wind: `-rho C_M U u_1`) [N m^-2]
    tau_y: torch.Tensor           # Upward meridional momentum flux [N m^-2]
    u_star: torch.Tensor          # Surface friction velocity `u_*` [m s^-1]
    buoy_flux: torch.Tensor       # Kinematic surface buoyancy flux `(g/theta_0) <w'theta_v'>` [m^2 s^-3]

    alb_vis_dir: torch.Tensor     # Surface direct-beam visible (SW1) albedo [-]
    alb_vis_dif: torch.Tensor     # Surface diffuse visible (SW1) albedo [-]
    alb_nir_dir: torch.Tensor     # Surface direct-beam near-IR (SW2) albedo [-]
    alb_nir_dif: torch.Tensor     # Surface diffuse near-IR (SW2) albedo [-]
    t_rad_sfc: torch.Tensor       # Effective radiative skin temperature `(LW_emit / sigma)^(1/4)` [K]
    emiss_sfc: torch.Tensor       # Surface longwave emissivity [-]
    d_disp: torch.Tensor          # Aerodynamic zero-plane displacement `d` [m]
    z0m: torch.Tensor             # Momentum roughness length `z_0m` [m]

    runoff_total: torch.Tensor    # Liquid + glacier runoff mass flux to return to ocean [kg m^-2 s^-1]
    runoff_ice: torch.Tensor      # Solid glacier calving component (for fusion enthalpy closure) [kg m^-2 s^-1]
    gpp: torch.Tensor             # Ecosystem gross primary production [umol CO2 m^-2 s^-1]
    f_seaice: torch.Tensor        # Diagnosed fractional sea-ice area `f_i` [-]
    f_snow_land: torch.Tensor     # Diagnosed fractional snow cover on land `A_sn` [-]
    smb_ice_flux: torch.Tensor | None = None  # Climatic snow/ice surface mass balance flux [kg m^-2 s^-1]
    resp_leaf: torch.Tensor | None = None     # Canopy leaf dark respiration R_d [umol CO2 m^-2 s^-1]
    an_net: torch.Tensor | None = None        # Canopy net CO2 assimilation (GPP - R_d) [umol CO2 m^-2 s^-1]
    transp: torch.Tensor | None = None        # Canopy transpiration water flux [kg m^-2 s^-1]


def init_surface_state(
    surf_type: torch.Tensor,
    z_topo: torch.Tensor | None = None,
    sigma_topo: torch.Tensor | None = None,
    t_init: float = 295.0,
    h_ml_ocean: float = 50.0,
) -> SurfaceState:
    """Initialize a consistent `SurfaceState` across tile shape `surf_type.shape`."""
    dev = surf_type.device
    shape = surf_type.shape
    f_dtype = torch.float32
    st = surf_type.to(dtype=torch.int64)

    z_t = torch.zeros(shape, device=dev, dtype=f_dtype) if z_topo is None else z_topo.to(dev, f_dtype)
    sig_t = torch.full(shape, 50.0, device=dev, dtype=f_dtype) if sigma_topo is None else sigma_topo.to(dev, f_dtype)

    is_veg = st == SURF_LAND
    is_ice = st == SURF_ICESHEET
    bio = init_biosphere_state(shape, is_veg, device=dev, dtype=f_dtype)

    t_land0 = torch.where(is_ice, torch.full(shape, 255.0, device=dev, dtype=f_dtype), torch.full(shape, t_init, device=dev, dtype=f_dtype))
    t_soil = t_land0.unsqueeze(0).repeat((N_SOIL_LAYERS,) + (1,) * len(shape))
    w_liq = torch.where(
        is_ice.unsqueeze(0),
        torch.zeros_like(t_soil),
        torch.full_like(t_soil, 0.25),
    )
    w_ice = torch.where(
        is_ice.unsqueeze(0),
        torch.full_like(t_soil, SOIL_POROSITY),
        torch.zeros_like(t_soil),
    )

    swe0 = torch.where(is_ice, torch.full(shape, SWE_GLACIER_MAX, device=dev, dtype=f_dtype), torch.zeros(shape, device=dev, dtype=f_dtype))

    w_root0, _, _ = compute_root_zone_weights_and_water(w_liq, bio.s_root_cap)
    return SurfaceState(
        surf_type=st,
        z_topo=z_t,
        sigma_topo=sig_t,
        t_soil=t_soil,
        w_liq_soil=w_liq,
        w_ice_soil=w_ice,
        w_root=torch.where(is_veg, w_root0, torch.zeros(shape, device=dev, dtype=f_dtype)),
        w_canopy=torch.zeros(shape, device=dev, dtype=f_dtype),
        t_canopy=t_land0.clone(),
        swe_land=swe0.clone(),
        swe_max_season=torch.maximum(swe0, torch.full_like(swe0, 10.0)),
        snow_age=torch.zeros(shape, device=dev, dtype=f_dtype),
        t_ocean_ml=torch.full(shape, max(t_init, C.T_FREEZE_OCEAN + 0.5), device=dev, dtype=f_dtype),
        h_ocean_ml=torch.full(shape, h_ml_ocean, device=dev, dtype=f_dtype),
        u_star_ocean=torch.full(shape, 0.15, device=dev, dtype=f_dtype),
        m_seaice=torch.zeros(shape, device=dev, dtype=f_dtype),
        t_seaice=torch.full(shape, C.T_FREEZE_OCEAN, device=dev, dtype=f_dtype),
        swe_seaice=torch.zeros(shape, device=dev, dtype=f_dtype),
        bio=bio,
        t_ocean_ml_comp=torch.zeros(shape, device=dev, dtype=f_dtype),
        t_leaf_sun=t_land0.clone(),
        t_leaf_sh=t_land0.clone(),
        t_gnd_skin=t_land0.clone(),
    )


def _solve_3x3_batched(
    a00: torch.Tensor,
    a01: torch.Tensor,
    a02: torch.Tensor,
    a10: torch.Tensor,
    a11: torch.Tensor,
    a12: torch.Tensor,
    a20: torch.Tensor,
    a21: torch.Tensor,
    a22: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form batched 3x3 linear solve via cofactors for the 3-source land energy-balance Jacobian."""
    c00 = a11 * a22 - a12 * a21
    c01 = a12 * a20 - a10 * a22
    c02 = a10 * a21 - a11 * a20
    det = a00 * c00 + a01 * c01 + a02 * c02
    inv_det = 1.0 / torch.where(det.abs() < 1.0e-12, torch.full_like(det, 1.0e-12), det)

    c10 = a02 * a21 - a01 * a22
    c11 = a00 * a22 - a02 * a20
    c12 = a01 * a20 - a00 * a21
    c20 = a01 * a12 - a02 * a11
    c21 = a02 * a10 - a00 * a12
    c22 = a00 * a11 - a01 * a10

    x0 = (c00 * b0 + c10 * b1 + c20 * b2) * inv_det
    x1 = (c01 * b0 + c11 * b1 + c21 * b2) * inv_det
    x2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det
    return x0, x1, x2


# ---------------------------------------------------------------------------
# 3. Albedo & Snow-Cover Helper Functions
# ---------------------------------------------------------------------------
def compute_snow_fraction_swenson(
    swe: torch.Tensor,
    swe_max: torch.Tensor,
    sigma_topo: torch.Tensor,
    is_icesheet: torch.Tensor,
) -> torch.Tensor:
    """Fractional snow-covered area `A_sn` in `[0, 1]` with melt hysteresis & topography.

    References
    ----------
    - Swenson & Lawrence (2012, J. Geophys. Res., 117, D21107, Eqs. 4-6):
      Accumulation `tanh` curve and ablation power-law depletion `1 - (acos(...) / pi)^N_melt`
      with `N_melt = 200 / max(10, sigma_topo)`.
    - MATSIRO6 (2021, Sec. 8.1): `A_sn = 1` over permanent grounded ice sheets.
    """
    sn = torch.clamp(swe, min=0.0)
    sn_max = torch.maximum(swe_max, sn + 1.0e-4)
    n_melt = 200.0 / torch.clamp(sigma_topo, min=10.0, max=1000.0)

    # Accumulation limit when sn ~ sn_max; smooth depletion curve during melt (sn < sn_max)
    ratio = torch.clamp(sn / torch.clamp(sn_max, min=1.0), 0.0, 1.0)
    # Swenson & Lawrence (2012, Eq. 6): f_sno = 1 - (acos(2 * ratio - 1) / pi)^N_melt,
    # blended with accumulation cap tanh(sn / (2.5 * 20)) so very thin 1 mm dustings don't hit 100%
    f_acc = torch.tanh(sn / 25.0)
    theta_acos = torch.acos(torch.clamp(2.0 * ratio - 1.0, -1.0 + 1.0e-6, 1.0 - 1.0e-6)) / math.pi
    f_abl = 1.0 - torch.pow(torch.clamp(theta_acos, 0.0, 1.0), n_melt)
    a_sn = torch.minimum(f_acc, f_abl)
    a_sn = torch.where(sn <= 1.0e-3, torch.zeros_like(a_sn), a_sn)
    return torch.where(is_icesheet, torch.ones_like(a_sn), torch.clamp(a_sn, 0.0, 1.0))


def compute_snow_albedo_yang(
    snow_age: torch.Tensor,
    mu0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-band (VIS/NIR) direct and diffuse snow albedos with aging & zenith angle.

    References
    ----------
    - Yang et al. (1997, J. Climate, 10, 353-373) & MATSIRO6 (2021, Eq. 3.1 & Eqs. 8.58-8.60):
        f_alb = A_g / (1 + A_g)
        alpha_hat_b = alpha_b,new + f_alb * (alpha_b,old - alpha_b,new)
        alpha_b(mu) = alpha_hat_b + (1 - alpha_hat_b) * 0.4 * (1 - mu)^5,
      with `mu = mu_0` for direct beam and `mu = cos(50 deg) = 0.64279` for diffuse sky.
    """
    f_alb = torch.clamp(snow_age / (1.0 + torch.clamp(snow_age, min=0.0)), 0.0, 0.999)
    vis0 = ALB_SNOW_VIS_NEW + f_alb * (ALB_SNOW_VIS_OLD - ALB_SNOW_VIS_NEW)
    nir0 = ALB_SNOW_NIR_NEW + f_alb * (ALB_SNOW_NIR_OLD - ALB_SNOW_NIR_NEW)

    zen_dir = 0.4 * torch.pow(1.0 - torch.clamp(mu0, 0.0, 1.0), 5.0)
    zen_dif = 0.4 * ((1.0 - math.cos(math.radians(50.0))) ** 5)

    vis_dir = torch.clamp(vis0 + (1.0 - vis0) * zen_dir, 0.05, 0.98)
    vis_dif = torch.clamp(vis0 + (1.0 - vis0) * zen_dif, 0.05, 0.98)
    nir_dir = torch.clamp(nir0 + (1.0 - nir0) * zen_dir, 0.05, 0.95)
    nir_dif = torch.clamp(nir0 + (1.0 - nir0) * zen_dif, 0.05, 0.95)
    return vis_dir, vis_dif, nir_dir, nir_dif


def compute_ocean_albedo_briegleb(mu0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solar-zenith-dependent open-ocean direct and diffuse shortwave albedos.

    References
    ----------
    - Briegleb, Minnis, Ramanathan, & Harrison (1986, J. Climate Appl. Meteorol., 25,
      214-226, Eq. 11):
        alpha_dir(mu_0) = 0.026 / (mu_0^1.7 + 0.065) + 0.15 * (mu_0 - 0.1) * (mu_0 - 0.5) * (mu_0 - 1.0)
    - MIROC6 (2021, Sec. 3.8.1.1 Eq. 3.476): diffuse sea-surface albedo `alpha_dif = 0.06`.
    """
    mu = torch.clamp(mu0, 0.01, 1.0)
    alb_dir = 0.026 / (torch.pow(mu, 1.7) + 0.065) + 0.15 * (mu - 0.1) * (mu - 0.5) * (mu - 1.0)
    alb_dir = torch.clamp(alb_dir, 0.02, 0.40)
    alb_dif = torch.full_like(alb_dir, 0.06)
    return alb_dir, alb_dif


def compute_bare_soil_albedo_moisture(w_liq_top: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map-free moisture-dependent bare-soil VIS & NIR albedo (Oleson et al., 2013, CLM4.5 Sec. 3.2).

    Dry hyper-arid sand/mineral soil reaches `(VIS = 0.28, NIR = 0.50)` (sustaining subtropical
    desert radiative cooling; Charney, 1975), darkening smoothly with volumetric topsoil water
    `Delta = max(0.11 - 0.40 * w_liq_top, 0)` down to wet soil `(VIS = 0.14, NIR = 0.28)`.
    """
    wetness = torch.clamp(w_liq_top / SOIL_POROSITY, 0.0, 1.0)
    # Continuous smooth reduction from dry desert sand to moist soil
    alb_vis = 0.14 + 0.14 * torch.pow(1.0 - wetness, 1.5)
    alb_nir = 0.28 + 0.22 * torch.pow(1.0 - wetness, 1.5)
    return alb_vis, alb_nir


def compute_surface_albedos(
    sfc: SurfaceState,
    mu0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnose 2-band direct/diffuse albedos, exposed canopy `(h_exp, lai_exp)`, `A_sn`, and `f_ice`."""
    if can_use_elem_metal(mu0) and mu0.shape == sfc.surf_type.shape:
        if getattr(sfc, "_surf_type_i32_src", None) is not sfc.surf_type:
            sfc._surf_type_i32 = sfc.surf_type.to(torch.int32)
            sfc._surf_type_i32_src = sfc.surf_type
        out_8 = dispatch_surface_albedos_metal(
            surf_type_i32=sfc._surf_type_i32,
            swe_land=sfc.swe_land,
            swe_max=sfc.swe_max_season,
            sigma_topo=sfc.sigma_topo,
            snow_age=sfc.snow_age,
            w_liq_top=sfc.w_liq_soil[0],
            h_canopy=sfc.bio.h_canopy,
            lai=sfc.bio.lai,
            m_seaice=sfc.m_seaice,
            swe_seaice=sfc.swe_seaice,
            mu0=mu0,
            consts=(
                float(ALB_SNOW_VIS_NEW),
                float(ALB_SNOW_NIR_NEW),
                float(ALB_SNOW_VIS_OLD),
                float(ALB_SNOW_NIR_OLD),
                float(RHO_SNOWPACK),
                float(SOIL_POROSITY),
                float(MC_SEAICE_CRIT),
                float(ALB_SEAICE_VIS),
                float(ALB_SEAICE_NIR),
                float(LEAF_SCAT_VIS),
                float(LEAF_SCAT_NIR),
            ),
        )
        alb_vis_dir, alb_vis_dif, alb_nir_dir, alb_nir_dif = out_8[0], out_8[1], out_8[2], out_8[3]
        h_exp, lai_exp, a_sn, f_ice = out_8[4], out_8[5], out_8[6], out_8[7]
        if sfc.bc_snow is not None or sfc.dust_snow is not None:
            bc_m = torch.clamp(sfc.bc_snow, min=0.0) if sfc.bc_snow is not None else torch.zeros_like(mu0)
            du_m = torch.clamp(sfc.dust_snow, min=0.0) if sfc.dust_snow is not None else torch.zeros_like(mu0)
            # SNICAR visible darkening bounded to <= 45% reduction on snow/sea-ice fraction
            d_frac = torch.clamp(180.0 * torch.sqrt(bc_m) + 3.5 * torch.sqrt(du_m), 0.0, 0.45)
            f_cov = torch.where(sfc.surf_type == SURF_OCEAN, f_ice, a_sn)
            mult = 1.0 - f_cov * d_frac
            alb_vis_dir = alb_vis_dir * mult
            alb_vis_dif = alb_vis_dif * mult
        return (alb_vis_dir, alb_vis_dif, alb_nir_dir, alb_nir_dif, h_exp, lai_exp, a_sn, f_ice)

    is_ocean = sfc.surf_type == SURF_OCEAN
    is_ice = sfc.surf_type == SURF_ICESHEET

    # 1. Land & ice-sheet snow cover and snow-burial masking (MATSIRO6 Eqs. 2.1-2.3)
    a_sn = compute_snow_fraction_swenson(sfc.swe_land, sfc.swe_max_season, sfc.sigma_topo, is_ice)
    snow_depth = sfc.swe_land / RHO_SNOWPACK
    h0 = sfc.bio.h_canopy
    h_buried = torch.clamp(h0 - snow_depth, min=0.0)
    lai_buried = sfc.bio.lai * (h_buried / torch.clamp(h0, min=0.05))
    h_exp = (1.0 - a_sn) * h0 + a_sn * h_buried
    lai_exp = (1.0 - a_sn) * sfc.bio.lai + a_sn * lai_buried

    sn_vis_dir, sn_vis_dif, sn_nir_dir, sn_nir_dif = compute_snow_albedo_yang(sfc.snow_age, mu0)
    soil_vis, soil_nir = compute_bare_soil_albedo_moisture(sfc.w_liq_soil[0])

    gnd_vis_dir = (1.0 - a_sn) * soil_vis + a_sn * sn_vis_dir
    gnd_vis_dif = (1.0 - a_sn) * soil_vis + a_sn * sn_vis_dif
    gnd_nir_dir = (1.0 - a_sn) * soil_nir + a_sn * sn_nir_dir
    gnd_nir_dif = (1.0 - a_sn) * soil_nir + a_sn * sn_nir_dif

    l_vis_dir, _, l_nir_dir, _, _, _ = canopy_two_stream_albedo_dai(lai_exp, mu0, gnd_vis_dir, gnd_nir_dir)
    _, l_vis_dif, _, l_nir_dif, _, _ = canopy_two_stream_albedo_dai(lai_exp, mu0, gnd_vis_dif, gnd_nir_dif)

    # 2. Ocean + fractional-lead sea ice (SCALE-RM Eq. 8.342; Briegleb et al., 1986)
    f_ice = torch.where(
        is_ocean,
        torch.clamp(torch.sqrt(torch.clamp(sfc.m_seaice, min=0.0) / MC_SEAICE_CRIT), 0.0, 1.0),
        torch.zeros_like(mu0),
    )
    ocn_dir, ocn_dif = compute_ocean_albedo_briegleb(mu0)
    f_sn_ice = torch.tanh(torch.clamp(sfc.swe_seaice, min=0.0) / 15.0)
    ice_vis = (1.0 - f_sn_ice) * ALB_SEAICE_VIS + f_sn_ice * ALB_SNOW_VIS_NEW
    ice_nir = (1.0 - f_sn_ice) * ALB_SEAICE_NIR + f_sn_ice * ALB_SNOW_NIR_NEW

    o_vis_dir = (1.0 - f_ice) * ocn_dir + f_ice * ice_vis
    o_vis_dif = (1.0 - f_ice) * ocn_dif + f_ice * ice_vis
    o_nir_dir = (1.0 - f_ice) * ocn_dir + f_ice * ice_nir
    o_nir_dif = (1.0 - f_ice) * ocn_dif + f_ice * ice_nir

    alb_vis_dir = torch.where(is_ocean, o_vis_dir, l_vis_dir)
    alb_vis_dif = torch.where(is_ocean, o_vis_dif, l_vis_dif)
    alb_nir_dir = torch.where(is_ocean, o_nir_dir, l_nir_dir)
    alb_nir_dif = torch.where(is_ocean, o_nir_dif, l_nir_dif)

    if sfc.bc_snow is not None or sfc.dust_snow is not None:
        bc_m = torch.clamp(sfc.bc_snow, min=0.0) if sfc.bc_snow is not None else torch.zeros_like(mu0)
        du_m = torch.clamp(sfc.dust_snow, min=0.0) if sfc.dust_snow is not None else torch.zeros_like(mu0)
        d_frac = torch.clamp(180.0 * torch.sqrt(bc_m) + 3.5 * torch.sqrt(du_m), 0.0, 0.45)
        f_cov = torch.where(is_ocean, f_ice, a_sn)
        mult = 1.0 - f_cov * d_frac
        alb_vis_dir = alb_vis_dir * mult
        alb_vis_dif = alb_vis_dif * mult

    return alb_vis_dir, alb_vis_dif, alb_nir_dir, alb_nir_dif, h_exp, lai_exp, a_sn, f_ice


# ---------------------------------------------------------------------------
# 4. Five-Layer Implicit Soil Heat Conduction & Freeze-Thaw Phase Change
# ---------------------------------------------------------------------------
_SOIL_GEOM_CACHE: dict[tuple[str, int, torch.dtype, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_soil_geom_tensors(
    dev: torch.device,
    dtype: torch.dtype,
    col_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached 5-layer soil `(dz, z_top, dz_int)` geometry tensors on `(dev, dtype)`."""
    key = (dev.type, dev.index or 0, dtype, col_shape)
    cached = _SOIL_GEOM_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SOIL_GEOM_CACHE) >= 32:
        _SOIL_GEOM_CACHE.pop(next(iter(_SOIL_GEOM_CACHE)))
    view_shape = (N_SOIL_LAYERS,) + (1,) * len(col_shape)
    dz = torch.tensor(SOIL_LAYER_DEPTHS, device=dev, dtype=dtype).view(view_shape)
    z_bot_vals = []
    z_mid_vals = []
    acc = 0.0
    for d in SOIL_LAYER_DEPTHS:
        z_mid_vals.append(acc + 0.5 * d)
        acc += d
        z_bot_vals.append(acc)
    z_top = torch.tensor([0.0] + z_bot_vals[:-1], device=dev, dtype=dtype).view(view_shape)
    dz_int = torch.tensor(
        [z_mid_vals[k + 1] - z_mid_vals[k] for k in range(N_SOIL_LAYERS - 1)],
        device=dev,
        dtype=dtype,
    ).view((N_SOIL_LAYERS - 1,) + (1,) * len(col_shape))
    cached = (dz, z_top, dz_int)
    _SOIL_GEOM_CACHE[key] = cached
    return cached


def _thomas_5layer_pure(
    dl: torch.Tensor,
    diag: torch.Tensor,
    du: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Pure-tensor 5-layer Thomas solver (compiles with zero graph breaks on CUDA Inductor)."""
    c = []
    d = []
    inv0 = 1.0 / diag[0]
    c.append(du[0] * inv0)
    d.append(rhs[0] * inv0)
    for k in range(1, 5):
        inv_k = 1.0 / (diag[k] - dl[k] * c[k - 1])
        c.append(du[k] * inv_k if k < 4 else torch.zeros_like(inv_k))
        d.append((rhs[k] - dl[k] * d[k - 1]) * inv_k)
    x = [d[4]] * 5
    for k in range(3, -1, -1):
        x[k] = d[k] - c[k] * x[k + 1]
    return torch.stack(x, dim=0)


def _soil_thermo_core(
    t_soil: torch.Tensor,
    w_liq: torch.Tensor,
    w_ice: torch.Tensor,
    g_ground_down: torch.Tensor,
    dz: torch.Tensor,
    dt: float,
    use_pure_thomas: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mw = RHO_WATER * torch.clamp(w_liq, min=0.0)
    mi = RHO_ICE_SOLID * torch.clamp(w_ice, min=0.0)
    m_tot = mw + mi

    t0 = C.T_FREEZE
    c_vol = CS_DRY_SOIL + CP_LIQ_WATER * mw + CP_ICE_SOLID * mi
    u_int0 = c_vol * (t_soil - t0) - C.L_F0 * mi
    u_top = u_int0[0] + (dt / dz[0]) * g_ground_down
    u_int = torch.cat([u_top.unsqueeze(0), u_int0[1:]], dim=0)

    mi_raw = -u_int / C.L_F0
    mi_new = torch.minimum(torch.clamp(mi_raw, min=0.0), m_tot)
    mw_new = torch.clamp(m_tot - mi_new, min=0.0)

    w_liq_new = mw_new / RHO_WATER
    w_ice_new = mi_new / RHO_ICE_SOLID
    c_vol_new = CS_DRY_SOIL + CP_LIQ_WATER * mw_new + CP_ICE_SOLID * mi_new
    dt_mid = (u_int + C.L_F0 * mi_new) / c_vol_new

    w_tot_vol = w_liq_new + w_ice_new
    kappa = KAPPA_DRY_SOIL * (1.0 + 5.5 * torch.tanh(w_tot_vol / 0.25)) + 0.8 * w_ice_new

    dz_int = 0.5 * (dz[1:] + dz[:-1])
    k_int = 0.5 * (kappa[1:] + kappa[:-1])
    cond_int = dt * k_int / dz_int

    cap_layer = c_vol_new * dz
    z0 = torch.zeros_like(dt_mid[0:1])
    dl = torch.cat([z0, -cond_int / cap_layer[1:]], dim=0)
    du = torch.cat([-cond_int / cap_layer[:-1], z0], dim=0)
    diag = 1.0 - dl - du
    if use_pure_thomas and dt_mid.shape[0] == 5:
        t_final = t0 + _thomas_5layer_pure(dl, diag, du, dt_mid)
    else:
        t_final = t0 + solve_tridiagonal_batched(dl, diag, du, dt_mid)
    return t_final, w_liq_new, w_ice_new


def step_multilayer_soil_thermodynamics(
    t_soil: torch.Tensor,
    w_liq: torch.Tensor,
    w_ice: torch.Tensor,
    g_ground_down: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance 5-layer soil temperature and liquid/ice phase change over `dt` [s].

    Implements the exact internal-energy conserving operator-split scheme of
    Nishizawa et al. (2025, SCALE-RM v5.5.5, Sec. 8.7.1 Eqs. 8.384-8.418) and
    MATSIRO6 (2021, Ch. 10 Eqs. 10.1-10.27):
    1. Apply downward ground heat flux `g_ground_down` [W m^-2] into top layer `k = 0`.
    2. Compute exact enthalpy-conserving 0 degC liquid-ice phase change (zero-curtain effect).
    3. Solve implicit Fourier heat conduction across the 5 layers via `solve_tridiagonal_batched`
       with moisture-dependent thermal conductivity `kappa(w)` (MATSIRO6 Eq. 10.4).
    """
    dev = t_soil.device
    dtype = t_soil.dtype
    col_shape = t_soil.shape[1:]
    dz, _, _ = _get_soil_geom_tensors(dev, dtype, col_shape)
    use_pure = (dev.type == "cuda") and (t_soil.shape[0] == 5)
    fn = maybe_cuda_compile(_soil_thermo_core, device=dev) if use_pure else _soil_thermo_core
    return fn(t_soil, w_liq, w_ice, g_ground_down, dz, dt, use_pure)


# ---------------------------------------------------------------------------
# 4b. Five-Layer Implicit Richards Soil Hydrology & Dynamic Rooting Depth
# ---------------------------------------------------------------------------
def compute_root_zone_weights_and_water(
    w_liq_soil: torch.Tensor,
    s_root_cap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnose vertically integrated root-zone plant-available water `W_avail` and layer uptake weights.

    Maps the Gao et al. (2014) optimal root-zone storage capacity `S_R` [kg m^-2] onto an effective
    dynamic rooting depth `z_root` [m] over AOBA's 5-layer soil grid using Clapp & Hornberger (1978)
    field capacity (`psi_fc = -3.3 m`) and wilting point (`psi_wp = -150 m`), and integrates liquid
    plant-available water vertically for the Holtzman et al. (2024, Sec. 2.2.1 Eq. 1) stomatal
    water-stress optimality (`beta_opt`):
        z_root = clip(S_R / [rho_w * (theta_fc - theta_wp)], 0.15, Z_tot),
        W_avail = sum_k rho_w * dz_k * alpha_root_k * max(theta_liq_k - theta_wp, 0).

    Frozen layers naturally reduce `theta_liq_k`, eliminating any ad-hoc unfrozen fraction multiplier.
    Returns `(w_avail [kg m^-2], w_uptake_weights [N_SOIL, ...], alpha_root [N_SOIL, ...])`.
    """
    dev = w_liq_soil.device
    dtype = w_liq_soil.dtype
    col_shape = w_liq_soil.shape[1:]

    dz, z_top, _ = _get_soil_geom_tensors(dev, dtype, col_shape)
    z_tot = sum(SOIL_LAYER_DEPTHS)

    theta_fc = SOIL_POROSITY * (abs(PSI_SAT_SOIL) / 3.3) ** (1.0 / B_CLAPP_SOIL)
    theta_wp = SOIL_POROSITY * (abs(PSI_SAT_SOIL) / 150.0) ** (1.0 / B_CLAPP_SOIL)
    dtheta_paw = max(theta_fc - theta_wp, 0.05)

    z_root = torch.clamp(
        torch.clamp(s_root_cap, min=20.0) / (RHO_WATER * dtheta_paw),
        min=0.15,
        max=z_tot,
    )
    alpha_root = torch.clamp((z_root.unsqueeze(0) - z_top) / dz, 0.0, 1.0)

    avail_layer = RHO_WATER * dz * alpha_root * torch.clamp(w_liq_soil - theta_wp, min=0.0)
    w_avail = avail_layer.sum(dim=0)

    root_depth_w = alpha_root * dz
    root_depth_w = root_depth_w / torch.clamp(root_depth_w.sum(dim=0, keepdim=True), min=1.0e-6)
    w_up = torch.where(
        w_avail.unsqueeze(0) > 1.0e-6,
        avail_layer / torch.clamp(w_avail.unsqueeze(0), min=1.0e-6),
        root_depth_w,
    )
    return w_avail, w_up, alpha_root


def _soil_hydro_core(
    w_liq: torch.Tensor,
    w_ice: torch.Tensor,
    rain_thru: torch.Tensor,
    melt_tot: torch.Tensor,
    e_soil: torch.Tensor,
    e_trans: torch.Tensor,
    dew_ground: torch.Tensor,
    s_root_cap: torch.Tensor,
    is_ice: torch.Tensor,
    dz: torch.Tensor,
    z_top: torch.Tensor,
    dz_int: torch.Tensor,
    dt: float,
    use_pure_thomas: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_l = torch.clamp(w_liq, min=0.0)
    w_i = torch.clamp(w_ice, min=0.0)
    theta_max = torch.clamp(SOIL_POROSITY - w_i, min=0.0)

    ice_frac = torch.clamp(w_i / SOIL_POROSITY, 0.0, 1.0)
    f_ice = torch.pow(10.0, -ICE_IMPEDANCE_OMEGA * ice_frac)

    s_min = 0.02
    s_eff = torch.clamp(w_l / SOIL_POROSITY, s_min, 1.0)
    th_safe = torch.clamp(w_l, min=s_min * SOIL_POROSITY)
    psi = torch.clamp(PSI_SAT_SOIL * torch.pow(s_eff, -B_CLAPP_SOIL), min=-1.0e5)
    dpsi_dth = torch.clamp(-B_CLAPP_SOIL * psi / th_safe, max=1.0e6)

    exp_k = 2.0 * B_CLAPP_SOIL + 3.0
    s_int = 0.5 * (s_eff[:-1] + s_eff[1:])
    f_ice_int = torch.minimum(f_ice[:-1], f_ice[1:])
    k_int = f_ice_int * K_SAT_SOIL * torch.pow(s_int, exp_k)
    q_int = k_int * (1.0 + (psi[:-1] - psi[1:]) / dz_int)

    k_bot = f_ice[-1] * K_SAT_SOIL * torch.pow(s_eff[-1], exp_k)
    dk_bot_dth = exp_k * k_bot / th_safe[-1]

    q_liq_in = torch.clamp(rain_thru + melt_tot + dew_ground, min=0.0)
    s_top_tot = torch.clamp((w_l[0] + w_i[0]) / SOIL_POROSITY, 0.0, 1.0)
    f_imp = 1.0 - f_ice[0]
    f_sat = f_imp + (1.0 - f_imp) * torch.pow(s_top_tot, 4.0)
    q_infil_cap = RHO_WATER * f_ice[0] * K_SAT_SOIL * torch.clamp(
        1.0 + (PSI_SAT_SOIL - psi[0]) / (0.5 * SOIL_LAYER_DEPTHS[0]), min=1.0, max=1.0e4
    )
    q_infil = torch.minimum((1.0 - f_sat) * q_liq_in, q_infil_cap)
    q_infil = torch.where(is_ice, torch.zeros_like(q_infil), q_infil)
    sfc_runoff_0 = torch.where(is_ice, q_liq_in, q_liq_in - q_infil)

    # Inline root-zone uptake weights (same as compute_root_zone_weights_and_water)
    z_tot = sum(SOIL_LAYER_DEPTHS)
    theta_fc = SOIL_POROSITY * (abs(PSI_SAT_SOIL) / 3.3) ** (1.0 / B_CLAPP_SOIL)
    theta_wp = SOIL_POROSITY * (abs(PSI_SAT_SOIL) / 150.0) ** (1.0 / B_CLAPP_SOIL)
    dtheta_paw = max(theta_fc - theta_wp, 0.05)
    z_root = torch.clamp(torch.clamp(s_root_cap, min=20.0) / (RHO_WATER * dtheta_paw), min=0.15, max=z_tot)
    alpha_root = torch.clamp((z_root.unsqueeze(0) - z_top) / dz, 0.0, 1.0)
    avail_layer = RHO_WATER * dz * alpha_root * torch.clamp(w_l - theta_wp, min=0.0)
    w_avail = avail_layer.sum(dim=0)
    root_depth_w = alpha_root * dz
    root_depth_w = root_depth_w / torch.clamp(root_depth_w.sum(dim=0, keepdim=True), min=1.0e-6)
    w_up = torch.where(
        w_avail.unsqueeze(0) > 1.0e-6,
        avail_layer / torch.clamp(w_avail.unsqueeze(0), min=1.0e-6),
        root_depth_w,
    )

    s_trans = (torch.clamp(e_trans, min=0.0) / RHO_WATER).unsqueeze(0) * w_up
    s_trans = torch.where(is_ice.unsqueeze(0), torch.zeros_like(s_trans), s_trans)

    a_int = (k_int / dz_int) * dpsi_dth[:-1]
    c_int = (k_int / dz_int) * dpsi_dth[1:]
    z0 = torch.zeros_like(w_l[0:1])
    dl = torch.cat([z0, -a_int], dim=0)
    du = torch.cat([-c_int, z0], dim=0)
    diag = (dz / dt).expand_as(w_l) + torch.cat([a_int, z0], dim=0) + torch.cat([z0, c_int], dim=0)
    diag = torch.cat([diag[:-1], (diag[-1] + dk_bot_dth).unsqueeze(0)], dim=0)

    e_soil_ms = torch.where(is_ice, torch.zeros_like(e_soil), torch.clamp(e_soil, min=0.0)) / RHO_WATER
    rhs0 = (q_infil / RHO_WATER - e_soil_ms) - q_int[0] - s_trans[0]
    rhs_mid = q_int[:-1] - q_int[1:] - s_trans[1:-1]
    rhs_bot = q_int[-1] - k_bot - s_trans[-1]
    rhs = torch.cat([rhs0.unsqueeze(0), rhs_mid, rhs_bot.unsqueeze(0)], dim=0)

    if use_pure_thomas and w_l.shape[0] == 5:
        dtheta = _thomas_5layer_pure(dl, diag, du, rhs)
    else:
        dtheta = solve_tridiagonal_batched(dl, diag, du, rhs)
    theta_new_raw = w_l + dtheta

    q_drain_raw = k_bot + dk_bot_dth * dtheta[-1]
    q_drain_neg = torch.clamp(-q_drain_raw, min=0.0)
    q_drain = torch.clamp(q_drain_raw, min=0.0)
    th_layers = [theta_new_raw[k] for k in range(N_SOIL_LAYERS)]
    th_layers[-1] = th_layers[-1] - (dt / SOIL_LAYER_DEPTHS[-1]) * q_drain_neg

    excess_mass = torch.zeros_like(w_l[0])
    for k in range(N_SOIL_LAYERS - 1, -1, -1):
        th_k = th_layers[k] + excess_mass / (RHO_WATER * SOIL_LAYER_DEPTHS[k])
        over = torch.clamp(th_k - theta_max[k], min=0.0)
        th_layers[k] = th_k - over
        excess_mass = over * (RHO_WATER * SOIL_LAYER_DEPTHS[k])
    sfc_runoff = sfc_runoff_0 + excess_mass / dt

    deficit_mass = torch.zeros_like(w_l[0])
    for k in range(N_SOIL_LAYERS):
        th_k = th_layers[k] - deficit_mass / (RHO_WATER * SOIL_LAYER_DEPTHS[k])
        under = torch.clamp(-th_k, min=0.0)
        th_layers[k] = th_k + under
        deficit_mass = under * (RHO_WATER * SOIL_LAYER_DEPTHS[k])

    theta_new = torch.stack(th_layers, dim=0)
    q_drain_mass = RHO_WATER * q_drain * dt
    reduce_drain = torch.minimum(deficit_mass, q_drain_mass)
    q_drain_mass = q_drain_mass - reduce_drain
    sub_runoff = torch.where(is_ice, torch.zeros_like(q_drain_mass), q_drain_mass / dt)
    w_liq_next = torch.where(is_ice.unsqueeze(0), w_liq, theta_new)
    return w_liq_next, sfc_runoff, sub_runoff


def step_multilayer_soil_hydrology(
    w_liq: torch.Tensor,
    w_ice: torch.Tensor,
    rain_thru: torch.Tensor,
    melt_tot: torch.Tensor,
    e_soil: torch.Tensor,
    e_trans: torch.Tensor,
    dew_ground: torch.Tensor,
    s_root_cap: torch.Tensor,
    is_ice: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance 5-layer volumetric liquid soil water via linearized implicit Richards equation.

    Implements map-free unsaturated vertical water transport, frozen-soil hydraulic impedance,
    saturation/infiltration-excess surface runoff, root-zone transpiration uptake, and free bottom
    gravitational drainage on AOBA's 5-layer soil grid with exact mass conservation:
    - Clapp & Hornberger (1978, WRR, 14, 601-604): retention `psi(s) = psi_s s^-b` and
      hydraulic conductivity `K(s) = K_sat s^(2b+3)`.
    - Swenson, Lawrence, & Lee (2012, J. Geophys. Res., 117, F02022) & Lawrence et al. (2020,
      CLM5 Tech. Note, Sec. 7.3-7.5, Eqs. 7.89-7.129): ice impedance `f_ice = 10^(-Omega * f_i)`,
      midpoint-saturation interface conductivity `K_{k+1/2}`, implicit tridiagonal Richards solver,
      and free gravitational drainage `q_drain = K_bot`.

    Returns `(w_liq_next [N_SOIL, ...], sfc_runoff [kg m^-2 s^-1], sub_runoff [kg m^-2 s^-1])`.
    """
    dev = w_liq.device
    dtype = w_liq.dtype
    col_shape = w_liq.shape[1:]
    dz, z_top, dz_int = _get_soil_geom_tensors(dev, dtype, col_shape)
    use_pure = (dev.type == "cuda") and (w_liq.shape[0] == 5)
    fn = maybe_cuda_compile(_soil_hydro_core, device=dev) if use_pure else _soil_hydro_core
    return fn(
        w_liq, w_ice, rain_thru, melt_tot, e_soil, e_trans, dew_ground,
        s_root_cap, is_ice, dz, z_top, dz_int, dt, use_pure,
    )


def _surface_land_pre_newton_core(
    sw1_dir_down: torch.Tensor,
    sw1_dif_down: torch.Tensor,
    sw2_dir_down: torch.Tensor,
    sw2_dif_down: torch.Tensor,
    alb_vis_dir: torch.Tensor,
    alb_vis_dif: torch.Tensor,
    alb_nir_dir: torch.Tensor,
    alb_nir_dif: torch.Tensor,
    h_exp: torch.Tensor,
    lai_exp: torch.Tensor,
    a_sn: torch.Tensor,
    temp_1: torch.Tensor,
    qv_1: torch.Tensor,
    pres_1: torch.Tensor,
    pres_sfc: torch.Tensor,
    mu0: torch.Tensor,
    l_sun: torch.Tensor,
    l_sh: torch.Tensor,
    k_db_inst: torch.Tensor,
    snow_age: torch.Tensor,
    w_liq_soil0: torch.Tensor,
    w_ice_soil0: torch.Tensor,
    t_soil0: torch.Tensor,
    w_canopy: torch.Tensor,
    swe_land: torch.Tensor,
):
    """Pure-tensor terrestrial pre-Newton setup (albedo absorption, 3-source SW, soil DSL, RSL)."""
    sw_abs = (
        sw1_dir_down * (1.0 - alb_vis_dir)
        + sw1_dif_down * (1.0 - alb_vis_dif)
        + sw2_dir_down * (1.0 - alb_nir_dir)
        + sw2_dif_down * (1.0 - alb_nir_dif)
    )
    exner_1 = torch.pow(pres_1 / C.P_REF, C.KAPPA_D)
    exner_s = torch.pow(pres_sfc / C.P_REF, C.KAPPA_D)
    theta_1 = temp_1 / exner_1
    theta_v1 = theta_1 * (1.0 + C.ZVIR * qv_1)
    rho_sfc = pres_sfc / (C.R_D * temp_1 * (1.0 + C.ZVIR * qv_1))
    cp_air = C.CP_D + C.CP_V * qv_1

    d_land, z0m_land, _ = canopy_roughness_raupach(h_exp, lai_exp, z0_soil=0.01)
    z0m_terr = torch.where(lai_exp < 0.05, (1.0 - a_sn) * z0m_land + a_sn * 0.002, z0m_land)
    has_veg = lai_exp >= 0.01
    f_veg_rad = torch.where(has_veg, 1.0 - torch.exp(-lai_exp), torch.zeros_like(lai_exp))
    z0h_terr = torch.where(
        has_veg,
        (1.0 - f_veg_rad) * (0.135335 * z0m_terr) + f_veg_rad * z0m_terr,
        0.135335 * z0m_terr,
    )
    rsl_depth = torch.clamp(2.0 * h_exp - d_land, min=0.0)

    has_sun = has_veg & (l_sun > 1.0e-4)
    has_sh = has_veg & (l_sh > 1.0e-4)
    f_sun_area = torch.where(has_veg, torch.clamp(l_sun / torch.clamp(lai_exp, min=1.0e-4), 0.0, 1.0), torch.zeros_like(lai_exp))
    f_sh_area = torch.where(has_veg, 1.0 - f_sun_area, torch.zeros_like(lai_exp))

    sw_tot_vis = sw1_dir_down * (1.0 - alb_vis_dir) + sw1_dif_down * (1.0 - alb_vis_dif)
    sw_tot_nir = sw2_dir_down * (1.0 - alb_nir_dir) + sw2_dif_down * (1.0 - alb_nir_dif)
    sn_vis_d, sn_vis_f, sn_nir_d, sn_nir_f = compute_snow_albedo_yang(snow_age, mu0)
    soil_v, soil_n = compute_bare_soil_albedo_moisture(w_liq_soil0)
    ag_vis_d = (1.0 - a_sn) * soil_v + a_sn * sn_vis_d
    ag_vis_f = (1.0 - a_sn) * soil_v + a_sn * sn_vis_f
    ag_nir_d = (1.0 - a_sn) * soil_n + a_sn * sn_nir_d
    ag_nir_f = (1.0 - a_sn) * soil_n + a_sn * sn_nir_f

    kb_vis_p = k_db_inst * math.sqrt(1.0 - LEAF_SCAT_VIS)
    kd_vis_p = 0.78 * math.sqrt(1.0 - LEAF_SCAT_VIS)
    kb_nir_p = k_db_inst * math.sqrt(1.0 - LEAF_SCAT_NIR)
    kd_nir_p = 0.78 * math.sqrt(1.0 - LEAF_SCAT_NIR)
    tr_vis_d = torch.exp(-torch.clamp(kb_vis_p * lai_exp, max=40.0))
    tr_vis_f = torch.exp(-torch.clamp(kd_vis_p * lai_exp, max=40.0))
    tr_nir_d = torch.exp(-torch.clamp(kb_nir_p * lai_exp, max=40.0))
    tr_nir_f = torch.exp(-torch.clamp(kd_nir_p * lai_exp, max=40.0))

    sw_gnd_vis = torch.minimum(
        sw_tot_vis,
        tr_vis_d * sw1_dir_down * (1.0 - ag_vis_d) + tr_vis_f * sw1_dif_down * (1.0 - ag_vis_f),
    )
    sw_gnd_nir = torch.minimum(
        sw_tot_nir,
        tr_nir_d * sw2_dir_down * (1.0 - ag_nir_d) + tr_nir_f * sw2_dif_down * (1.0 - ag_nir_f),
    )
    sw_can_vis = torch.where(has_veg, torch.clamp(sw_tot_vis - sw_gnd_vis, min=0.0), torch.zeros_like(sw_tot_vis))
    sw_can_nir = torch.where(has_veg, torch.clamp(sw_tot_nir - sw_gnd_nir, min=0.0), torch.zeros_like(sw_tot_nir))
    f_sun_vis = canopy_sunlit_sw_fraction(sw1_dir_down, sw1_dif_down, mu0, lai_exp, LEAF_SCAT_VIS)
    f_sun_nir = canopy_sunlit_sw_fraction(sw2_dir_down, sw2_dif_down, mu0, lai_exp, LEAF_SCAT_NIR)
    sw_abs_sun = torch.where(has_sun, f_sun_vis * sw_can_vis + f_sun_nir * sw_can_nir, torch.zeros_like(sw_abs))
    sw_abs_sh = torch.where(has_sh, (sw_can_vis + sw_can_nir) - sw_abs_sun, torch.zeros_like(sw_abs))
    sw_abs_gnd = sw_abs - sw_abs_sun - sw_abs_sh

    wc_cap = WC_MAX_PER_LAI * lai_exp
    f_wet = torch.where(
        has_veg,
        torch.clamp(torch.pow(torch.clamp(w_canopy, min=0.0) / torch.clamp(wc_cap, min=1.0e-4), 2.0 / 3.0), 0.0, F_WET_MAX),
        torch.zeros_like(lai_exp),
    )
    sat_top = torch.clamp((w_liq_soil0 + w_ice_soil0) / SOIL_POROSITY, 0.01, 1.0)
    psi_top = PSI_SAT_SOIL * torch.pow(sat_top, -B_CLAPP_SOIL)
    h_soil_base = torch.exp(torch.clamp(psi_top * C.GRAVITY / (C.R_V * t_soil0), -10.0, 0.0))

    theta_liq_top = torch.clamp(w_liq_soil0, min=0.0, max=SOIL_POROSITY)
    theta_air = SOIL_POROSITY * (abs(PSI_SAT_SOIL) * 1.0e3 / PSI_AIR_MM) ** (1.0 / B_CLAPP_SOIL)
    theta_init = DSL_THETA_INIT_FRAC * SOIL_POROSITY
    d_sl = DSL_DMAX_MM * 1.0e-3 * torch.clamp((theta_init - theta_liq_top) / max(theta_init - theta_air, 1.0e-3), min=0.0, max=13.0)
    phi_air = max(SOIL_POROSITY - theta_air, 1.0e-3)
    tau_soil = (phi_air ** 2) * (phi_air / SOIL_POROSITY) ** (3.0 / B_CLAPP_SOIL)
    d_vap = 2.12e-5 * torch.pow(t_soil0 / C.T_FREEZE, 1.75)
    r_soil = d_sl / (d_vap * tau_soil)

    dz_top_half = 0.5 * SOIL_LAYER_DEPTHS[0]
    k_top = KAPPA_DRY_SOIL * (1.0 + 5.5 * torch.tanh((w_liq_soil0 + w_ice_soil0) / 0.25))
    r_snow_therm = a_sn * (0.5 * torch.clamp(swe_land / RHO_SNOWPACK, max=1.0) / C.K_THERM_SNOW)
    g_cond_coeff = 1.0 / (dz_top_half / k_top + r_snow_therm)

    beta_rsl = torch.clamp(torch.sqrt(0.003 + 0.15 * torch.clamp(lai_exp, min=1.0e-4)), 0.12, 0.30)
    alpha_u_half = 0.5 * torch.clamp(0.20 / (2.0 * beta_rsl * beta_rsl), 0.30, 1.20)
    t_air_sfc = theta_1 * exner_s
    has_snow = swe_land > 1.0e-3
    eps_g = EMISSIVITY_LAND
    lw_refl_c = 1.0 + (1.0 - f_veg_rad) * (1.0 - eps_g)
    lw_emit_c_coef = 2.0 - f_veg_rad * (1.0 - eps_g)
    return (
        sw_abs, exner_s, theta_1, theta_v1, rho_sfc, cp_air,
        d_land, z0m_terr, z0h_terr, rsl_depth, has_veg, f_veg_rad,
        has_sun, has_sh, f_sun_area, f_sh_area,
        sw_abs_sun, sw_abs_sh, sw_abs_gnd, wc_cap, f_wet,
        h_soil_base, r_soil, g_cond_coeff, beta_rsl, alpha_u_half,
        t_air_sfc, has_snow, lw_refl_c, lw_emit_c_coef,
    )


def _surface_land_newton_pre_bio_core(
    t_sun_cur: torch.Tensor,
    t_sh_cur: torch.Tensor,
    t_gnd_cur: torch.Tensor,
    pres_sfc: torch.Tensor,
    q_af_in: torch.Tensor,
    c_h_land: torch.Tensor,
    wind_eff_land: torch.Tensor,
    u_star_land: torch.Tensor,
    beta_rsl: torch.Tensor,
    alpha_u_half: torch.Tensor,
    lai_exp: torch.Tensor,
    k_db_inst: torch.Tensor,
    h_exp: torch.Tensor,
    z0m_terr: torch.Tensor,
    has_veg: torch.Tensor,
    has_sun: torch.Tensor,
    has_sh: torch.Tensor,
    f_veg_rad: torch.Tensor,
    f_sun_area: torch.Tensor,
    f_sh_area: torch.Tensor,
):
    """Pure-tensor leaf boundary-layer conductances, r_d_uc, qsat, and leaf-surface VPD."""
    ga_land = c_h_land * wind_eff_land
    u_can_top = torch.minimum(torch.clamp(u_star_land / beta_rsl, min=0.10), wind_eff_land)
    gb_top = 0.01 * torch.sqrt(u_can_top / 0.04)
    gb_tot_can = gb_top * (1.0 - torch.exp(-torch.clamp(alpha_u_half * lai_exp, max=30.0))) / alpha_u_half
    gb_sun = torch.where(
        has_sun,
        torch.clamp(
            gb_top * (1.0 - torch.exp(-torch.clamp((k_db_inst + alpha_u_half) * lai_exp, max=30.0)))
            / (k_db_inst + alpha_u_half),
            min=1.0e-4,
        ),
        torch.zeros_like(lai_exp),
    )
    gb_sh = torch.where(has_sh, torch.clamp(gb_tot_can - gb_sun, min=1.0e-4), torch.zeros_like(lai_exp))

    t_fol_avg = torch.where(has_veg, f_sun_area * t_sun_cur + f_sh_area * t_sh_cur, t_gnd_cur)
    stab_uc = C.GRAVITY * h_exp * (t_fol_avg - t_gnd_cur) / (torch.clamp(t_gnd_cur, min=180.0) * (u_star_land * u_star_land + 1.0e-3))
    c_dense = torch.where(stab_uc > 0.0, 0.004 / (1.0 + 0.5 * torch.clamp(stab_uc, max=10.0)), torch.full_like(stab_uc, 0.004))
    c_bare = C.VON_KARMAN / (0.13 * torch.pow(torch.clamp(u_star_land * z0m_terr / 1.5e-5, min=1.0), 0.45))
    c_uv = (1.0 - f_veg_rad) * c_bare + f_veg_rad * c_dense
    r_d_uc = 1.0 / torch.clamp(c_uv * u_star_land, min=1.0e-3)

    qsat_sun, dq_sun = qsat_mixed(t_sun_cur, pres_sfc)
    qsat_sh, dq_sh = qsat_mixed(t_sh_cur, pres_sfc)
    qsat_gnd, dq_gnd = qsat_mixed(t_gnd_cur, pres_sfc)

    q_af_vpd = torch.clamp(q_af_in, min=1.0e-7)
    e_af_pa = q_af_vpd * pres_sfc / (C.EPS_WATER + (1.0 - C.EPS_WATER) * q_af_vpd)
    vpd_sun_af = torch.clamp(esat_liquid(t_sun_cur) - e_af_pa, min=20.0)
    vpd_sh_af = torch.clamp(esat_liquid(t_sh_cur) - e_af_pa, min=20.0)
    return (
        ga_land, gb_sun, gb_sh, r_d_uc,
        qsat_sun, dq_sun, qsat_sh, dq_sh, qsat_gnd, dq_gnd,
        q_af_vpd, vpd_sun_af, vpd_sh_af,
    )


def _surface_ocean_pre_transfer_core(
    u_star_ocean: torch.Tensor,
    f_ice: torch.Tensor,
    t_ocean_ml: torch.Tensor,
    t_seaice: torch.Tensor,
    pres_sfc: torch.Tensor,
    exner_s: torch.Tensor,
):
    """Pure-tensor ocean/sea-ice roughness and surface virtual potential temperature."""
    z0m_ocn, z0h_ocn, z0q_ocn = ocean_roughness_miller(u_star_ocean)
    z0m_sea = (1.0 - f_ice) * z0m_ocn + f_ice * 0.001
    z0h_sea = (1.0 - f_ice) * z0h_ocn + f_ice * 0.0002
    z0q_sea = (1.0 - f_ice) * z0q_ocn + f_ice * 0.0002
    esat_ow = esat_liquid(t_ocean_ml)
    esat_si = esat_ice(t_seaice)
    qsat_ow = 0.98 * C.EPS_WATER * esat_ow / torch.clamp(pres_sfc - esat_ow, min=0.1 * pres_sfc)
    qsat_si = C.EPS_WATER * esat_si / torch.clamp(pres_sfc - esat_si, min=0.1 * pres_sfc)
    t_skin_sea = (1.0 - f_ice) * t_ocean_ml + f_ice * t_seaice
    q_skin_sea = (1.0 - f_ice) * qsat_ow + f_ice * qsat_si
    theta_vsfc_o = (t_skin_sea / exner_s) * (1.0 + C.ZVIR * q_skin_sea)
    return z0m_sea, z0h_sea, z0q_sea, qsat_ow, qsat_si, theta_vsfc_o


def _land_newton_jacobian_core(
    t_sun_cur: torch.Tensor,
    t_sh_cur: torch.Tensor,
    t_gnd_cur: torch.Tensor,
    t_soil0: torch.Tensor,
    t_air_sfc: torch.Tensor,
    qv_1: torch.Tensor,
    q_af_vpd: torch.Tensor,
    qsat_sun: torch.Tensor,
    dq_sun: torch.Tensor,
    qsat_sh: torch.Tensor,
    dq_sh: torch.Tensor,
    qsat_gnd: torch.Tensor,
    dq_gnd: torch.Tensor,
    ga_land: torch.Tensor,
    gb_sun: torch.Tensor,
    gb_sh: torch.Tensor,
    gsw_sun: torch.Tensor,
    gsw_sh: torch.Tensor,
    r_d_uc: torch.Tensor,
    r_soil: torch.Tensor,
    h_soil_base: torch.Tensor,
    a_sn: torch.Tensor,
    f_wet: torch.Tensor,
    w_canopy: torch.Tensor,
    rho_sfc: torch.Tensor,
    cp_air: torch.Tensor,
    has_veg: torch.Tensor,
    has_sun: torch.Tensor,
    has_sh: torch.Tensor,
    has_snow: torch.Tensor,
    f_veg_rad: torch.Tensor,
    f_sun_area: torch.Tensor,
    f_sh_area: torch.Tensor,
    lw_refl_c: torch.Tensor,
    lw_emit_c_coef: torch.Tensor,
    sw_abs_sun: torch.Tensor,
    sw_abs_sh: torch.Tensor,
    sw_abs_gnd: torch.Tensor,
    lw_down: torch.Tensor,
    g_cond_coeff: torch.Tensor,
    prev_dt_sun: torch.Tensor,
    prev_dt_sh: torch.Tensor,
    prev_dt_gnd: torch.Tensor,
    dt: float,
    apply_damping: bool,
):
    """Pure-tensor 3-source Newton Jacobian assembly & 3x3 solve (CUDA-compilable)."""
    e_pot_wet = rho_sfc * (
        gb_sun * (qsat_sun - q_af_vpd) + gb_sh * (qsat_sh - q_af_vpd)
    )
    dew_on_leaf = (e_pot_wet <= 0.0) & has_veg
    f_wet_eff = torch.where(
        e_pot_wet > 1.0e-12,
        torch.minimum(f_wet, torch.clamp(w_canopy, min=0.0) / (dt * torch.clamp(e_pot_wet, min=1.0e-12))),
        f_wet,
    )
    f_wet_eff = torch.where(dew_on_leaf, torch.ones_like(f_wet_eff), f_wet_eff)

    g_trans_sun = torch.where(
        dew_on_leaf | (~has_sun),
        torch.zeros_like(f_wet),
        (1.0 - f_wet) * (gb_sun * gsw_sun) / torch.clamp(gb_sun + gsw_sun, min=1.0e-6),
    )
    g_trans_sh = torch.where(
        dew_on_leaf | (~has_sh),
        torch.zeros_like(f_wet),
        (1.0 - f_wet) * (gb_sh * gsw_sh) / torch.clamp(gb_sh + gsw_sh, min=1.0e-6),
    )
    g_inter_sun = torch.where(has_sun, f_wet_eff * gb_sun, torch.zeros_like(f_wet))
    g_inter_sh = torch.where(has_sh, f_wet_eff * gb_sh, torch.zeros_like(f_wet))

    gw_0 = g_trans_sun + g_inter_sun
    gw_1 = g_trans_sh + g_inter_sh

    q_soil_ref = torch.where(has_veg, q_af_vpd, qv_1)
    dry_soil_cap = (qsat_gnd > q_soil_ref) & (q_soil_ref > h_soil_base * qsat_gnd)
    h_soil = torch.where(
        dry_soil_cap,
        q_soil_ref / torch.clamp(qsat_gnd, min=1.0e-8),
        h_soil_base,
    )
    g_soil_snowfree = (1.0 - a_sn) / torch.clamp(r_d_uc + r_soil, min=1.0e-4)
    g_snow_subl = a_sn / torch.clamp(r_d_uc, min=1.0e-4)
    gw_2 = g_soil_snowfree + g_snow_subl
    h_gnd = torch.where(
        gw_2 > 1.0e-8,
        (g_snow_subl + g_soil_snowfree * h_soil) / torch.clamp(gw_2, min=1.0e-8),
        h_soil,
    )
    dhq_gnd_dT = torch.where(
        gw_2 > 1.0e-8,
        (g_snow_subl + torch.where(dry_soil_cap, torch.zeros_like(g_soil_snowfree), g_soil_snowfree * h_soil))
        / torch.clamp(gw_2, min=1.0e-8),
        torch.where(dry_soil_cap, torch.zeros_like(h_soil), h_soil),
    ) * dq_gnd

    gh_0 = 2.0 * gb_sun
    gh_1 = 2.0 * gb_sh
    gh_2 = 1.0 / r_d_uc

    gh_sum = ga_land + gh_0 + gh_1 + gh_2
    t_af = torch.where(
        has_veg,
        (ga_land * t_air_sfc + gh_0 * t_sun_cur + gh_1 * t_sh_cur + gh_2 * t_gnd_cur) / gh_sum,
        t_gnd_cur,
    )
    gw_sum = ga_land + gw_0 + gw_1 + gw_2
    q_af = (ga_land * qv_1 + gw_0 * qsat_sun + gw_1 * qsat_sh + gw_2 * h_gnd * qsat_gnd) / gw_sum

    wh_0 = torch.where(has_veg, gh_0 / gh_sum, torch.zeros_like(gh_0))
    wh_1 = torch.where(has_veg, gh_1 / gh_sum, torch.zeros_like(gh_1))
    wh_2 = torch.where(has_veg, gh_2 / gh_sum, torch.ones_like(gh_2))
    ww_0 = gw_0 / gw_sum
    ww_1 = gw_1 / gw_sum
    ww_2 = gw_2 / gw_sum

    l_vap_sun = torch.where(t_sun_cur < C.T_FREEZE, C.L_S0, C.L_V0)
    l_vap_sh = torch.where(t_sh_cur < C.T_FREEZE, C.L_S0, C.L_V0)
    l_vap_gnd = torch.where((a_sn > 0.5) | (t_gnd_cur < C.T_FREEZE), C.L_S0, C.L_V0)

    eps_g = EMISSIVITY_LAND
    sig = C.STEFAN_BOLTZMANN
    t0_3 = t_sun_cur * t_sun_cur * t_sun_cur
    t1_3 = t_sh_cur * t_sh_cur * t_sh_cur
    t2_3 = t_gnd_cur * t_gnd_cur * t_gnd_cur
    t0_4 = t0_3 * t_sun_cur
    t1_4 = t1_3 * t_sh_cur
    t2_4 = t2_3 * t_gnd_cur

    dc_sun = f_veg_rad * f_sun_area
    dc_sh = f_veg_rad * f_sh_area
    r_lw_0 = dc_sun * (lw_refl_c * lw_down + eps_g * sig * t2_4 - lw_emit_c_coef * sig * t0_4)
    r_lw_1 = dc_sh * (lw_refl_c * lw_down + eps_g * sig * t2_4 - lw_emit_c_coef * sig * t1_4)
    r_lw_2 = eps_g * (1.0 - f_veg_rad) * lw_down + eps_g * sig * (dc_sun * t0_4 + dc_sh * t1_4) - eps_g * sig * t2_4

    rho_cp = rho_sfc * cp_air
    h_0 = rho_cp * gh_0 * (t_sun_cur - t_af)
    h_1 = rho_cp * gh_1 * (t_sh_cur - t_af)
    h_2 = torch.where(has_veg, rho_cp * gh_2 * (t_gnd_cur - t_af), rho_cp * ga_land * (t_gnd_cur - t_air_sfc))

    e_0 = rho_sfc * gw_0 * (qsat_sun - q_af)
    e_1 = rho_sfc * gw_1 * (qsat_sh - q_af)
    e_2 = torch.maximum(rho_sfc * gw_2 * (h_gnd * qsat_gnd - q_af), -rho_sfc * ga_land * qv_1)

    g0_down = g_cond_coeff * (t_gnd_cur - t_soil0)
    b0 = sw_abs_sun + r_lw_0 - h_0 - l_vap_sun * e_0
    b1 = sw_abs_sh + r_lw_1 - h_1 - l_vap_sh * e_1
    b2 = sw_abs_gnd + r_lw_2 - h_2 - l_vap_gnd * e_2 - g0_down

    k_lw_0 = 4.0 * dc_sun * sig
    k_lw_1 = 4.0 * dc_sh * sig
    lw_g_3 = 4.0 * eps_g * sig * t2_3

    j00 = rho_cp * gh_0 * (1.0 - wh_0) + l_vap_sun * rho_sfc * gw_0 * dq_sun * (1.0 - ww_0) + k_lw_0 * lw_emit_c_coef * t0_3
    j01 = -rho_cp * gh_0 * wh_1 - l_vap_sun * rho_sfc * gw_0 * ww_1 * dq_sh
    j02 = -rho_cp * gh_0 * wh_2 - l_vap_sun * rho_sfc * gw_0 * ww_2 * dhq_gnd_dT - dc_sun * lw_g_3

    j10 = -rho_cp * gh_1 * wh_0 - l_vap_sh * rho_sfc * gw_1 * ww_0 * dq_sun
    j11 = rho_cp * gh_1 * (1.0 - wh_1) + l_vap_sh * rho_sfc * gw_1 * dq_sh * (1.0 - ww_1) + k_lw_1 * lw_emit_c_coef * t1_3
    j12 = -rho_cp * gh_1 * wh_2 - l_vap_sh * rho_sfc * gw_1 * ww_2 * dhq_gnd_dT - dc_sh * lw_g_3

    dh2_dt2 = torch.where(has_veg, rho_cp * gh_2 * (1.0 - wh_2), rho_cp * ga_land)
    j20 = torch.where(has_veg, -rho_cp * gh_2 * wh_0 - l_vap_gnd * rho_sfc * gw_2 * ww_0 * dq_sun - eps_g * k_lw_0 * t0_3, torch.zeros_like(j00))
    j21 = torch.where(has_veg, -rho_cp * gh_2 * wh_1 - l_vap_gnd * rho_sfc * gw_2 * ww_1 * dq_sh - eps_g * k_lw_1 * t1_3, torch.zeros_like(j00))
    j22 = dh2_dt2 + l_vap_gnd * rho_sfc * gw_2 * dhq_gnd_dT * (1.0 - ww_2) + g_cond_coeff + lw_g_3

    one_t = torch.ones_like(j00)
    zero_t = torch.zeros_like(j00)
    t_pin_sun = torch.where(has_sh, t_sh_cur, t_gnd_cur)
    t_pin_sh = torch.where(has_sun, t_sun_cur, t_gnd_cur)

    j00_s = torch.where(has_sun, torch.clamp(j00, min=1.0), one_t)
    j01_s = torch.where(has_sun, j01, zero_t)
    j02_s = torch.where(has_sun, j02, zero_t)
    b0_s = torch.where(has_sun, b0, t_pin_sun - t_sun_cur)

    j10_s = torch.where(has_sh, j10, zero_t)
    j11_s = torch.where(has_sh, torch.clamp(j11, min=1.0), one_t)
    j12_s = torch.where(has_sh, j12, zero_t)
    b1_s = torch.where(has_sh, b1, t_pin_sh - t_sh_cur)

    j22_s = torch.clamp(j22, min=1.0)
    dt0_raw, dt1_raw, dt2_raw = _solve_3x3_batched(
        j00_s, j01_s, j02_s, j10_s, j11_s, j12_s, j20, j21, j22_s, b0_s, b1_s, b2
    )
    dt0 = torch.clamp(dt0_raw, -10.0, 10.0)
    dt1 = torch.clamp(dt1_raw, -10.0, 10.0)
    dt2 = torch.clamp(dt2_raw, -10.0, 10.0)
    if apply_damping:
        dt0 = torch.where(dt0 * prev_dt_sun < 0.0, 0.5 * dt0, dt0)
        dt1 = torch.where(dt1 * prev_dt_sh < 0.0, 0.5 * dt1, dt1)
        dt2 = torch.where(dt2 * prev_dt_gnd < 0.0, 0.5 * dt2, dt2)

    t_sun_new = torch.clamp(t_sun_cur + dt0, min=150.0, max=370.0)
    t_sh_new = torch.clamp(t_sh_cur + dt1, min=150.0, max=370.0)
    t_gnd_new = torch.clamp(t_gnd_cur + dt2, min=150.0, max=370.0)

    melt_excess_t = torch.where(has_snow & (t_gnd_new > C.T_FREEZE), t_gnd_new - C.T_FREEZE, zero_t)
    t_gnd_final = t_gnd_new - melt_excess_t
    dt2_fin = t_gnd_final - t_gnd_cur
    q_melt_avail = melt_excess_t * j22_s
    return (
        t_sun_new, t_sh_new, t_gnd_new, t_gnd_final, dt0, dt1, dt2, dt2_fin, q_melt_avail,
        q_af, wh_0, wh_1, wh_2, ww_0, ww_1, ww_2, gh_0, gh_1, gh_2, rho_cp,
        h_0, h_1, h_2, g_trans_sun, g_trans_sh, g_inter_sun, g_inter_sh,
        g_soil_snowfree, g_snow_subl, h_soil, dry_soil_cap, dhq_gnd_dT, j22_s,
    )


def _land_post_newton_core(
    t_sun_cur: torch.Tensor,
    t_sh_cur: torch.Tensor,
    t_sun_new: torch.Tensor,
    t_sh_new: torch.Tensor,
    t_gnd_new: torch.Tensor,
    t_gnd_final: torch.Tensor,
    dt2_fin: torch.Tensor,
    q_melt_avail: torch.Tensor,
    q_af: torch.Tensor,
    wh_0: torch.Tensor,
    wh_1: torch.Tensor,
    wh_2: torch.Tensor,
    ww_0: torch.Tensor,
    ww_1: torch.Tensor,
    ww_2: torch.Tensor,
    gh_0: torch.Tensor,
    gh_1: torch.Tensor,
    gh_2: torch.Tensor,
    ga_land: torch.Tensor,
    rho_sfc: torch.Tensor,
    rho_cp: torch.Tensor,
    h_0: torch.Tensor,
    h_1: torch.Tensor,
    h_2: torch.Tensor,
    g_trans_sun: torch.Tensor,
    g_trans_sh: torch.Tensor,
    g_inter_sun: torch.Tensor,
    g_inter_sh: torch.Tensor,
    g_soil_snowfree: torch.Tensor,
    g_snow_subl: torch.Tensor,
    qsat_sun: torch.Tensor,
    dq_sun: torch.Tensor,
    qsat_sh: torch.Tensor,
    dq_sh: torch.Tensor,
    qsat_gnd: torch.Tensor,
    dq_gnd: torch.Tensor,
    h_soil: torch.Tensor,
    dry_soil_cap: torch.Tensor,
    dhq_gnd_dT: torch.Tensor,
    j22_s: torch.Tensor,
    qv_1: torch.Tensor,
    a_sn: torch.Tensor,
    has_veg: torch.Tensor,
    has_sun: torch.Tensor,
    has_sh: torch.Tensor,
    has_snow: torch.Tensor,
    is_terrestrial: torch.Tensor,
    is_ice: torch.Tensor,
    f_veg_rad: torch.Tensor,
    f_sun_area: torch.Tensor,
    f_sh_area: torch.Tensor,
    lw_refl_c: torch.Tensor,
    sw_abs: torch.Tensor,
    lw_down: torch.Tensor,
    lai_exp: torch.Tensor,
    wc_cap: torch.Tensor,
    w_canopy: torch.Tensor,
    w_avail_root: torch.Tensor,
    s_root_cap: torch.Tensor,
    water_stress: torch.Tensor,
    swe_land: torch.Tensor,
    swe_max_season: torch.Tensor,
    snow_age: torch.Tensor,
    w_liq_soil0: torch.Tensor,
    t_soil0: torch.Tensor,
    precip_rain: torch.Tensor,
    precip_snow: torch.Tensor,
    dt: float,
):
    """Pure-tensor post-Newton land fluxes, canopy water, and snowpack update (CUDA-compilable)."""
    eps_g = EMISSIVITY_LAND
    sig = C.STEFAN_BOLTZMANN
    t_sun_final = torch.where(has_sun, t_sun_new, torch.where(has_sh, t_sh_new, t_gnd_final))
    t_sh_final = torch.where(has_sh, t_sh_new, torch.where(has_sun, t_sun_new, t_gnd_final))
    dt0_fin = t_sun_final - t_sun_cur
    dt1_fin = t_sh_final - t_sh_cur

    dt_af_fin = wh_0 * dt0_fin + wh_1 * dt1_fin + wh_2 * dt2_fin
    dq_af_fin = ww_0 * dq_sun * dt0_fin + ww_1 * dq_sh * dt1_fin + ww_2 * dhq_gnd_dT * dt2_fin

    shf_0 = h_0 + rho_cp * gh_0 * (dt0_fin - dt_af_fin)
    shf_1 = h_1 + rho_cp * gh_1 * (dt1_fin - dt_af_fin)
    shf_2 = torch.where(
        has_veg,
        h_2 + rho_cp * gh_2 * (dt2_fin - dt_af_fin),
        h_2 + rho_cp * ga_land * dt2_fin,
    )
    shf_l = shf_0 + shf_1 + shf_2

    dq0_fin = (qsat_sun + dq_sun * dt0_fin) - (q_af + dq_af_fin)
    dq1_fin = (qsat_sh + dq_sh * dt1_fin) - (q_af + dq_af_fin)
    dq2_soil_fin = torch.where(
        dry_soil_cap,
        torch.zeros_like(qsat_gnd),
        h_soil * (qsat_gnd + dq_gnd * dt2_fin) - (q_af + dq_af_fin),
    )
    dq2_snow_fin = (qsat_gnd + dq_gnd * dt2_fin) - (q_af + dq_af_fin)

    e_trans_raw = rho_sfc * (g_trans_sun * dq0_fin + g_trans_sh * dq1_fin)
    e_inter_raw_signed = rho_sfc * (g_inter_sun * dq0_fin + g_inter_sh * dq1_fin)
    e_soil_raw_signed = rho_sfc * g_soil_snowfree * dq2_soil_fin
    e_snow_raw_signed = rho_sfc * g_snow_subl * dq2_snow_fin

    e_inter_pos = torch.clamp(e_inter_raw_signed, min=0.0)
    e_trans_pos = torch.clamp(e_trans_raw, min=0.0)
    e_soil_pos = torch.clamp(e_soil_raw_signed, min=0.0)
    e_snow_pos = torch.clamp(e_snow_raw_signed, min=0.0)
    dew_leaf_neg = torch.clamp(e_inter_raw_signed, max=0.0) + torch.clamp(e_trans_raw, max=0.0)
    dew_gnd_neg = torch.clamp(e_soil_raw_signed, max=0.0) + torch.clamp(e_snow_raw_signed, max=0.0)
    dew_neg = torch.maximum(dew_leaf_neg + dew_gnd_neg, -rho_sfc * ga_land * qv_1)

    l_vap_land = torch.where((a_sn > 0.5) | (t_gnd_final < C.T_FREEZE), C.L_S0, C.L_V0)
    e_inter = torch.minimum(e_inter_pos, torch.where(is_terrestrial, w_canopy / dt, e_inter_pos))
    shf_l = shf_l + l_vap_land * (e_inter_pos - e_inter)
    sw_max_rate = (1.05 / 3600.0) * water_stress * (s_root_cap / 150.0)
    e_trans = torch.minimum(
        torch.minimum(e_trans_pos, sw_max_rate),
        torch.where(is_terrestrial, w_avail_root / dt, e_trans_pos),
    )
    e_snow = torch.minimum(e_snow_pos, torch.where(is_terrestrial, swe_land / dt, e_snow_pos))
    max_soil_evap = RHO_WATER * SOIL_LAYER_DEPTHS[0] * torch.clamp(w_liq_soil0 - 0.02, min=0.0) / dt
    e_soil = torch.minimum(e_soil_pos, torch.where(is_terrestrial, max_soil_evap, e_soil_pos))

    evap_l = e_inter + e_trans + e_snow + e_soil + dew_neg
    lhf_l = l_vap_land * evap_l

    tg4 = (t_gnd_final * t_gnd_final) ** 2
    ts4 = (t_sun_final * t_sun_final) ** 2
    tsh4 = (t_sh_final * t_sh_final) ** 2
    lw_emit_tot_l = (
        (1.0 - f_veg_rad) * eps_g * sig * tg4
        + f_veg_rad * lw_refl_c * sig * (f_sun_area * ts4 + f_sh_area * tsh4)
    )
    lw_up_l = ((1.0 - f_veg_rad) ** 2) * (1.0 - eps_g) * lw_down + lw_emit_tot_l
    t_skin_final = torch.pow(torch.clamp(lw_emit_tot_l / (eps_g * sig), min=1.0e4), 0.25)
    g_into_soil = sw_abs + lw_down - lw_up_l - shf_l - lhf_l - q_melt_avail

    f_int = torch.clamp(ALPHA_INTERCEPT_LIQ * torch.tanh(lai_exp), 0.0, 1.0)
    wc_mid = w_canopy + dt * (f_int * precip_rain - e_inter - dew_leaf_neg)
    drip = torch.clamp(wc_mid - wc_cap, min=0.0) / dt
    wc_next = torch.clamp(wc_mid - drip * dt, min=0.0)
    rain_thru = (1.0 - f_int) * precip_rain + drip
    snow_thru = precip_snow

    swe_mid = torch.clamp(swe_land + dt * (snow_thru - e_snow), min=0.0)
    melt_rate = torch.minimum(q_melt_avail / C.L_F0, swe_mid / dt)
    warm_ground_melt = torch.where((t_soil0 > C.T_FREEZE + 1.0) & (swe_mid < 2.0), swe_mid / dt, torch.zeros_like(swe_mid))
    melt_tot = torch.minimum(melt_rate + warm_ground_melt, swe_mid / dt)
    g_into_soil = g_into_soil - C.L_F0 * (melt_tot - melt_rate)
    swe_after_melt = torch.clamp(swe_mid - melt_tot * dt, min=0.0)

    glacier_runoff = torch.clamp(swe_after_melt - SWE_GLACIER_MAX, min=0.0) / dt
    swe_next = swe_after_melt - glacier_runoff * dt
    bare_ice_melt_pot = torch.where(
        (~has_snow) & (t_gnd_new > C.T_FREEZE),
        (t_gnd_new - C.T_FREEZE) * j22_s / C.L_F0,
        torch.zeros_like(t_gnd_new),
    )
    smb_ice_land = torch.where(
        is_terrestrial,
        snow_thru - e_snow - melt_tot - bare_ice_melt_pot,
        torch.zeros_like(snow_thru),
    )

    swe_max_next = torch.where(
        swe_next <= 1.0e-2,
        torch.zeros_like(swe_next),
        torch.maximum(swe_max_season, swe_next),
    )
    t_sn = torch.clamp(t_skin_final, max=C.T_FREEZE)
    f_age = torch.exp(5000.0 * (1.0 / C.T_FREEZE - 1.0 / torch.clamp(t_sn, min=200.0)))
    r_dirt = torch.where(is_ice, torch.full_like(f_age, 0.01), torch.full_like(f_age, 0.30))
    age_inc = (f_age + torch.pow(f_age, 10.0) + r_dirt) * (dt / TAU_SNOW_AGE)
    fresh_refresh = torch.clamp(snow_thru * dt / 10.0, 0.0, 1.0)
    snow_age_next = torch.where(
        swe_next <= 1.0e-3,
        torch.zeros_like(snow_age),
        (1.0 - fresh_refresh) * (snow_age + age_inc),
    )
    dew_ground = -dew_gnd_neg
    return (
        t_sun_final, t_sh_final, t_skin_final, shf_l, lhf_l, evap_l,
        e_soil, e_trans, g_into_soil, wc_next, rain_thru, melt_tot,
        dew_ground, glacier_runoff, swe_next, smb_ice_land, swe_max_next, snow_age_next,
    )


def _ocean_seaice_post_core(
    t_ocean_ml: torch.Tensor,
    t_ocean_ml_comp: torch.Tensor,
    h_ocean_ml: torch.Tensor,
    m_seaice: torch.Tensor,
    t_seaice: torch.Tensor,
    swe_seaice: torch.Tensor,
    f_ice: torch.Tensor,
    qsat_ow: torch.Tensor,
    qsat_si: torch.Tensor,
    ga_h_o: torch.Tensor,
    ga_q_o: torch.Tensor,
    rho_sfc: torch.Tensor,
    cp_air: torch.Tensor,
    t_air_sfc: torch.Tensor,
    qv_1: torch.Tensor,
    sw_abs: torch.Tensor,
    sw1_dir_down: torch.Tensor,
    sw1_dif_down: torch.Tensor,
    sw2_dir_down: torch.Tensor,
    sw2_dif_down: torch.Tensor,
    lw_down: torch.Tensor,
    precip_snow: torch.Tensor,
    mu0: torch.Tensor,
    dt: float,
):
    """Pure-tensor open-water, sea-ice skin, and Kahan mixed-layer update (CUDA-compilable)."""
    shf_ow = rho_sfc * cp_air * ga_h_o * (t_ocean_ml - t_air_sfc)
    evap_ow = rho_sfc * ga_q_o * (qsat_ow - qv_1)
    lhf_ow = C.L_V0 * evap_ow
    t_ml_4 = (t_ocean_ml * t_ocean_ml) ** 2
    lw_up_ow = EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * t_ml_4 + (1.0 - EMISSIVITY_OCEAN) * lw_down
    alb_ocn_dir = compute_ocean_albedo_briegleb(mu0)[0]
    sw_abs_ow = (
        (sw1_dir_down + sw2_dir_down) * (1.0 - alb_ocn_dir)
        + (sw1_dif_down + sw2_dif_down) * 0.94
    )
    g_down_ow = sw_abs_ow + lw_down - lw_up_ow - shf_ow - lhf_ow - C.L_F0 * precip_snow

    h_ice_eff = torch.clamp(m_seaice / (C.RHO_SEAICE * torch.clamp(f_ice, min=0.05)), min=0.05)
    h_sn_ice = swe_seaice / RHO_SNOWPACK
    cond_ice = 1.0 / (0.5 * h_ice_eff / C.K_THERM_SEAICE + h_sn_ice / C.K_THERM_SNOW)

    shf_si0 = rho_sfc * cp_air * ga_h_o * (t_seaice - t_air_sfc)
    subl_si0 = rho_sfc * ga_q_o * (qsat_si - qv_1)
    t_si_3 = t_seaice * t_seaice * t_seaice
    lw_up_si0 = EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * (t_si_3 * t_seaice) + (1.0 - EMISSIVITY_OCEAN) * lw_down
    g_cond_si0 = cond_ice * (t_seaice - C.T_FREEZE_OCEAN)
    dqsi_dT = qsat_si * (21.8745584 * (C.T_FREEZE - 7.66)) / (torch.clamp(t_seaice, min=180.0) - 7.66) ** 2
    denom_si = (
        rho_sfc * cp_air * ga_h_o
        + C.L_S0 * rho_sfc * ga_q_o * dqsi_dT
        + cond_ice
        + 4.0 * EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * t_si_3
    )
    # Exact sea-ice tile absorbed SW using snow-covered sea-ice VIS/NIR albedos (matches compute_surface_albedos
    # so (1 - f_ice) * sw_abs_ow + f_ice * sw_abs_si == sw_abs and open-water SW never heats sea-ice floes!)
    f_sn_ice = torch.tanh(torch.clamp(swe_seaice, min=0.0) / 15.0)
    ice_vis = (1.0 - f_sn_ice) * ALB_SEAICE_VIS + f_sn_ice * ALB_SNOW_VIS_NEW
    ice_nir = (1.0 - f_sn_ice) * ALB_SEAICE_NIR + f_sn_ice * ALB_SNOW_NIR_NEW
    sw_abs_si = (sw1_dir_down + sw1_dif_down) * (1.0 - ice_vis) + (sw2_dir_down + sw2_dif_down) * (1.0 - ice_nir)

    res_si = sw_abs_si + lw_down - lw_up_si0 - shf_si0 - C.L_S0 * subl_si0 - g_cond_si0
    t_si_raw = torch.clamp(t_seaice + torch.clamp(res_si / denom_si, -10.0, 10.0), min=170.0, max=310.0)
    si_melt_excess = torch.clamp(t_si_raw - C.T_FREEZE, min=0.0)
    t_si_new = torch.where(f_ice > 1.0e-4, t_si_raw - si_melt_excess, torch.full_like(t_si_raw, C.T_FREEZE_OCEAN))
    dt_si = t_si_new - t_seaice

    shf_si = shf_si0 + rho_sfc * cp_air * ga_h_o * dt_si
    subl_si = subl_si0 + rho_sfc * ga_q_o * dqsi_dT * dt_si
    lhf_si = C.L_S0 * subl_si
    g_oi_up = cond_ice * (C.T_FREEZE_OCEAN - t_si_new)

    swe_si_mid = torch.clamp(swe_seaice + f_ice * dt * (precip_snow - torch.clamp(subl_si, min=0.0)), min=0.0)
    q_top_melt = f_ice * si_melt_excess * denom_si
    sn_si_melt = torch.minimum(q_top_melt * dt / C.L_F0, swe_si_mid)
    swe_si_next = swe_si_mid - sn_si_melt
    q_rem_melt = torch.clamp(q_top_melt * dt - C.L_F0 * sn_si_melt, min=0.0)

    cw_ml = C.RHO_SEAWATER * C.CP_SEAWATER * h_ocean_ml
    dt_ml_flux = (dt / cw_ml) * ((1.0 - f_ice) * g_down_ow - f_ice * g_oi_up)
    t_ml_prov, comp1 = _kahan_add(t_ocean_ml, dt_ml_flux, t_ocean_ml_comp)

    e_freeze = cw_ml * torch.clamp(C.T_FREEZE_OCEAN - t_ml_prov, min=0.0)
    e_basal_melt = cw_ml * torch.clamp(t_ml_prov - C.T_FREEZE_OCEAN, min=0.0)
    m_freeze = e_freeze / C.L_F0
    m_melt = torch.minimum((e_basal_melt + q_rem_melt) / C.L_F0, m_seaice + m_freeze)
    m_si_next = torch.clamp(m_seaice + m_freeze - m_melt, min=0.0)
    dt_ml_lat = (C.L_F0 * m_freeze - C.L_F0 * torch.minimum(e_basal_melt / C.L_F0, m_seaice)) / cw_ml
    t_ml_next, comp2 = _kahan_add(t_ml_prov, dt_ml_lat, comp1)
    comp_final = torch.where((e_freeze > 0.0) | (m_melt > 0.0), torch.zeros_like(comp2), comp2)

    shf_o = (1.0 - f_ice) * shf_ow + f_ice * shf_si
    evap_o = (1.0 - f_ice) * evap_ow + f_ice * subl_si
    lhf_o = (1.0 - f_ice) * lhf_ow + f_ice * lhf_si
    lw_emit_o = EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * (
        (1.0 - f_ice) * t_ml_4 + f_ice * ((t_si_new * t_si_new) ** 2)
    )
    return t_ml_next, comp_final, m_si_next, t_si_new, swe_si_next, shf_o, evap_o, lhf_o, lw_emit_o


# ---------------------------------------------------------------------------
# 5. Unified Land / Ice-Sheet / Ocean+Sea-Ice Time-Stepping Driver
# ---------------------------------------------------------------------------
def _surface_part_c_merge_core(
    is_terrestrial: torch.Tensor,
    is_land: torch.Tensor,
    is_ocean: torch.Tensor,
    t_soil_new: torch.Tensor,
    w_liq_new: torch.Tensor,
    w_ice_new: torch.Tensor,
    w_root_next: torch.Tensor,
    wc_next: torch.Tensor,
    t_skin_final: torch.Tensor,
    t_sun_final: torch.Tensor,
    t_sh_final: torch.Tensor,
    t_gnd_final: torch.Tensor,
    swe_next: torch.Tensor,
    swe_max_next: torch.Tensor,
    snow_age_next: torch.Tensor,
    t_ml_next: torch.Tensor,
    comp_final: torch.Tensor,
    comp0: torch.Tensor,
    m_si_next: torch.Tensor,
    t_si_new: torch.Tensor,
    swe_si_next: torch.Tensor,
    sfc_t_soil: torch.Tensor,
    sfc_w_liq_soil: torch.Tensor,
    sfc_w_ice_soil: torch.Tensor,
    sfc_w_root: torch.Tensor,
    sfc_t_canopy: torch.Tensor,
    sfc_t_ocean_ml: torch.Tensor,
    sfc_u_star_ocean: torch.Tensor,
    sfc_t_seaice: torch.Tensor,
    shf_o: torch.Tensor,
    shf_l: torch.Tensor,
    lhf_o: torch.Tensor,
    lhf_l: torch.Tensor,
    evap_o: torch.Tensor,
    evap_l: torch.Tensor,
    lw_emit_o: torch.Tensor,
    cm_ocn: torch.Tensor,
    cm_land: torch.Tensor,
    wind_ocn: torch.Tensor,
    wind_land: torch.Tensor,
    ustar_ocn: torch.Tensor,
    ustar_land: torch.Tensor,
    rho_sfc: torch.Tensor,
    cp_air: torch.Tensor,
    exner_s: torch.Tensor,
    theta_1: torch.Tensor,
    theta_v1: torch.Tensor,
    qv_1: torch.Tensor,
    u_1: torch.Tensor,
    v_1: torch.Tensor,
    d_land: torch.Tensor,
    z0m_sea: torch.Tensor,
    z0m_terr: torch.Tensor,
    liq_runoff: torch.Tensor,
    glacier_runoff: torch.Tensor,
    two_leaf_gpp: torch.Tensor,
    two_leaf_resp: torch.Tensor,
    two_leaf_an: torch.Tensor,
    e_trans: torch.Tensor,
    a_sn: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Pure-tensor Part C state commit & merged flux diagnostics (CUDA-compilable)."""
    terr_2d = is_terrestrial.unsqueeze(0)
    t_soil_out = torch.where(terr_2d, t_soil_new, sfc_t_soil)
    w_liq_out = torch.where(terr_2d, w_liq_new, sfc_w_liq_soil)
    w_ice_out = torch.where(terr_2d, w_ice_new, sfc_w_ice_soil)
    w_root_out = torch.where(is_land, w_root_next, sfc_w_root)
    z0 = torch.zeros_like(wc_next)
    wc_out = torch.where(is_land, wc_next, z0)
    t_can_out = torch.where(is_terrestrial, t_skin_final, sfc_t_canopy)
    t_sun_out = torch.where(is_terrestrial, t_sun_final, t_can_out)
    t_sh_out = torch.where(is_terrestrial, t_sh_final, t_can_out)
    t_gnd_out = torch.where(is_terrestrial, t_gnd_final, t_can_out)
    swe_out = torch.where(is_terrestrial, swe_next, z0)
    swe_max_out = torch.where(is_terrestrial, swe_max_next, z0)
    snow_age_out = torch.where(is_terrestrial, snow_age_next, z0)

    t_ml_out = torch.where(is_ocean, t_ml_next, sfc_t_ocean_ml)
    comp_out = torch.where(is_ocean, comp_final, comp0)
    ustar_o_out = torch.where(is_ocean, ustar_ocn, sfc_u_star_ocean)
    m_si_out = torch.where(is_ocean, m_si_next, z0)
    t_si_out = torch.where(is_ocean, t_si_new, sfc_t_seaice)
    swe_si_out = torch.where(is_ocean, swe_si_next, z0)

    shf = torch.where(is_ocean, shf_o, shf_l)
    lhf = torch.where(is_ocean, lhf_o, lhf_l)
    evap = torch.where(is_ocean, evap_o, evap_l)

    cm_all = torch.where(is_ocean, cm_ocn, cm_land)
    wind_all = torch.where(is_ocean, wind_ocn, wind_land)
    ustar_all = torch.where(is_ocean, ustar_ocn, ustar_land)
    tau_x = -rho_sfc * cm_all * wind_all * u_1
    tau_y = -rho_sfc * cm_all * wind_all * v_1

    w_th_sfc = shf / (rho_sfc * cp_air * exner_s)
    w_qv_sfc = evap / rho_sfc
    buoy_flux = (C.GRAVITY / theta_v1) * (w_th_sfc * (1.0 + C.ZVIR * qv_1) + C.ZVIR * theta_1 * w_qv_sfc)

    emiss = torch.where(is_ocean, torch.full_like(shf, EMISSIVITY_OCEAN), torch.full_like(shf, EMISSIVITY_LAND))
    lw_emit_l = EMISSIVITY_LAND * C.STEFAN_BOLTZMANN * (t_skin_final ** 4)
    t_rad = torch.pow(torch.where(is_ocean, lw_emit_o, lw_emit_l) / (emiss * C.STEFAN_BOLTZMANN), 0.25)

    d_out = torch.where(is_ocean, z0, d_land)
    z0m_out = torch.where(is_ocean, z0m_sea, z0m_terr)
    runoff_tot = torch.where(is_terrestrial, liq_runoff + glacier_runoff, z0)
    runoff_glac = torch.where(is_terrestrial, glacier_runoff, z0)
    gpp_out = torch.where(is_land, two_leaf_gpp, z0)
    resp_out = torch.where(is_land, two_leaf_resp, z0)
    an_out = torch.where(is_land, two_leaf_an, z0)
    transp_out = torch.where(is_land, e_trans, z0)
    f_snow_land = torch.where(is_terrestrial, a_sn, z0)

    return (
        t_soil_out, w_liq_out, w_ice_out, w_root_out, wc_out,
        t_can_out, t_sun_out, t_sh_out, t_gnd_out, swe_out, swe_max_out, snow_age_out,
        t_ml_out, comp_out, ustar_o_out, m_si_out, t_si_out, swe_si_out,
        shf, lhf, evap, tau_x, tau_y, ustar_all, buoy_flux, emiss, t_rad,
        d_out, z0m_out, runoff_tot, runoff_glac, gpp_out, resp_out, an_out, transp_out, f_snow_land,
    )


def step_surface(
    sfc: SurfaceState,
    u_1: torch.Tensor,
    v_1: torch.Tensor,
    temp_1: torch.Tensor,
    qv_1: torch.Tensor,
    pres_1: torch.Tensor,
    pres_sfc: torch.Tensor,
    z_ref_rel: torch.Tensor,
    sw1_dir_down: torch.Tensor,
    sw1_dif_down: torch.Tensor,
    sw2_dir_down: torch.Tensor,
    sw2_dif_down: torch.Tensor,
    lw_down: torch.Tensor,
    precip_rain: torch.Tensor,
    precip_snow: torch.Tensor,
    mu0: torch.Tensor,
    co2_ppmv: float,
    dt: float,
    precomputed_albedos: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> SurfaceFluxDiagnostics:
    """Advance land/snow/soil and ocean/sea-ice states by `dt` [s] and compute lower BCs."""
    is_ocean = sfc.surf_type == SURF_OCEAN
    is_land = sfc.surf_type == SURF_LAND
    is_ice = sfc.surf_type == SURF_ICESHEET
    is_terrestrial = ~is_ocean

    # 1. Surface albedos, exposed canopy geometry, snow fraction A_sn, and sea-ice fraction f_ice
    (
        alb_vis_dir,
        alb_vis_dif,
        alb_nir_dir,
        alb_nir_dif,
        h_exp,
        lai_exp,
        a_sn,
        f_ice,
    ) = precomputed_albedos if precomputed_albedos is not None else compute_surface_albedos(sfc, mu0)

    # Vertically integrated root-zone plant-available liquid water & optimal stress beta (Holtzman et al., 2024; Gao et al., 2014)
    w_avail_root, _, _ = compute_root_zone_weights_and_water(sfc.w_liq_soil, sfc.bio.s_root_cap)
    water_stress = compute_optimal_soil_water_beta(w_avail_root, sfc.bio)

    # Precompute temperature-independent sunlit/shaded PAR & enzyme allocation once before Newton iterations
    vis_beam = C.PAR_FRAC_OF_SW1 * sw1_dir_down
    vis_diff = C.PAR_FRAC_OF_SW1 * sw1_dif_down
    alloc_2l = prepare_two_leaf_allocation(sfc.bio, vis_beam, vis_diff, mu0, lai_exp)
    l_sun = alloc_2l.inst_rad.l_sun
    l_sh = alloc_2l.inst_rad.l_sh
    k_db_inst = alloc_2l.inst_rad.k_db

    # =========================================================================
    # PART A: TERRESTRIAL TILES (3-Source Sunlit/Shaded/Ground + RSL + P-Model)
    # =========================================================================
    t_soil0 = sfc.t_soil[0]
    pre_land_fn = maybe_cuda_compile(_surface_land_pre_newton_core, device=u_1.device)
    (
        sw_abs, exner_s, theta_1, theta_v1, rho_sfc, cp_air,
        d_land, z0m_terr, z0h_terr, rsl_depth, has_veg, f_veg_rad,
        has_sun, has_sh, f_sun_area, f_sh_area,
        sw_abs_sun, sw_abs_sh, sw_abs_gnd, wc_cap, f_wet,
        h_soil_base, r_soil, g_cond_coeff, beta_rsl, alpha_u_half,
        t_air_sfc, has_snow, lw_refl_c, lw_emit_c_coef,
    ) = pre_land_fn(
        sw1_dir_down, sw1_dif_down, sw2_dir_down, sw2_dif_down,
        alb_vis_dir, alb_vis_dif, alb_nir_dir, alb_nir_dif,
        h_exp, lai_exp, a_sn, temp_1, qv_1, pres_1, pres_sfc, mu0,
        l_sun, l_sh, k_db_inst, sfc.snow_age, sfc.w_liq_soil[0],
        sfc.w_ice_soil[0], t_soil0, sfc.w_canopy, sfc.swe_land,
    )

    # Warm-start sunlit leaf, shaded leaf, and ground skin temperatures (backward-compatible with older restarts)
    t_gnd_cur = sfc.t_gnd_skin.clone() if sfc.t_gnd_skin is not None else t_soil0.clone()
    t_sun_cur = sfc.t_leaf_sun.clone() if sfc.t_leaf_sun is not None else sfc.t_canopy.clone()
    t_sh_cur = sfc.t_leaf_sh.clone() if sfc.t_leaf_sh is not None else sfc.t_canopy.clone()
    t_gnd_cur = torch.where(has_veg, t_gnd_cur, sfc.t_canopy)

    # Coupled 3-source (sunlit leaf, shaded leaf, ground) + canopy-air + stomatal Newton iteration
    # (Dai et al., 2004, Sec. 2b-c; Bonan et al., 2018, Eqs. 8-13, 21-24; CLM5 Sec. 5.2-5.3)
    n_eb_iter = max(int(LAND_EB_ITERATIONS), 1)
    prev_dt_sun = torch.zeros_like(t_sun_cur)
    prev_dt_sh = torch.zeros_like(t_sh_cur)
    prev_dt_gnd = torch.zeros_like(t_gnd_cur)
    pre_bio_fn = maybe_cuda_compile(_surface_land_newton_pre_bio_core, device=u_1.device)
    jac_fn = maybe_cuda_compile(_land_newton_jacobian_core, device=u_1.device)

    for _eb_it in range(n_eb_iter):
        # 1. Surface-layer RSL aerodynamic transfer g_a evaluated at composite radiative/canopy-air temperature
        t_eff_mo = torch.where(
            has_veg,
            f_veg_rad * (f_sun_area * t_sun_cur + f_sh_area * t_sh_cur) + (1.0 - f_veg_rad) * t_gnd_cur,
            t_gnd_cur,
        )
        qsat_mo, _ = qsat_mixed(t_eff_mo, pres_sfc)
        theta_vsfc_l = (t_eff_mo / exner_s) * (1.0 + C.ZVIR * qsat_mo)
        tr_land = compute_surface_transfer(
            u_1, v_1, theta_v1, theta_vsfc_l, z_ref_rel,
            z0m_terr, z0h_terr, z0h_terr, rsl_depth=rsl_depth,
        )

        # 2-4. RSL leaf boundary-layer conductances, under-canopy r_d, qsat, & leaf-surface VPD
        q_af_in = qv_1 if _eb_it == 0 else q_af
        (
            ga_land, gb_sun, gb_sh, r_d_uc,
            qsat_sun, dq_sun, qsat_sh, dq_sh, qsat_gnd, dq_gnd,
            q_af_vpd, vpd_sun_af, vpd_sh_af,
        ) = pre_bio_fn(
            t_sun_cur, t_sh_cur, t_gnd_cur, pres_sfc, q_af_in,
            tr_land.c_h, tr_land.wind_eff, tr_land.u_star,
            beta_rsl, alpha_u_half, lai_exp, k_db_inst, h_exp, z0m_terr,
            has_veg, has_sun, has_sh, f_veg_rad, f_sun_area, f_sh_area,
        )

        # 5. Iterative leaf-surface (c_s, D_s) P-model photosynthesis & stomatal conductance
        two_leaf = compute_two_leaf_gas_exchange(
            sfc.bio, vis_beam, vis_diff, mu0, t_sun_cur, t_sh_cur, vpd_sun_af, vpd_sh_af,
            pres_sfc, co2_ppmv, lai_exp, water_stress,
            gbw_sun=torch.clamp(gb_sun, min=1.0e-4),
            gbw_sh=torch.clamp(gb_sh, min=1.0e-4),
            precomputed_alloc=alloc_2l,
        )

        # 6-7. Coupled CLM5 wet-canopy limitation, 3-source conductances, Jacobian & 3x3 solve
        (
            t_sun_new, t_sh_new, t_gnd_new, t_gnd_final, dt0, dt1, dt2, dt2_fin, q_melt_avail,
            q_af, wh_0, wh_1, wh_2, ww_0, ww_1, ww_2, gh_0, gh_1, gh_2, rho_cp,
            h_0, h_1, h_2, g_trans_sun, g_trans_sh, g_inter_sun, g_inter_sh,
            g_soil_snowfree, g_snow_subl, h_soil, dry_soil_cap, dhq_gnd_dT, j22_s,
        ) = jac_fn(
            t_sun_cur, t_sh_cur, t_gnd_cur, t_soil0, t_air_sfc, qv_1, q_af_vpd,
            qsat_sun, dq_sun, qsat_sh, dq_sh, qsat_gnd, dq_gnd,
            ga_land, gb_sun, gb_sh, two_leaf.gsw_sun, two_leaf.gsw_sh,
            r_d_uc, r_soil, h_soil_base, a_sn, f_wet, sfc.w_canopy,
            rho_sfc, cp_air, has_veg, has_sun, has_sh, has_snow,
            f_veg_rad, f_sun_area, f_sh_area, lw_refl_c, lw_emit_c_coef,
            sw_abs_sun, sw_abs_sh, sw_abs_gnd, lw_down, g_cond_coeff,
            prev_dt_sun, prev_dt_sh, prev_dt_gnd, dt, (_eb_it > 0),
        )
        prev_dt_sun, prev_dt_sh, prev_dt_gnd = dt0, dt1, dt2

        if _eb_it < n_eb_iter - 1:
            t_sun_cur = torch.where(has_sun, t_sun_new, torch.where(has_sh, t_sh_new, t_gnd_final))
            t_sh_cur = torch.where(has_sh, t_sh_new, torch.where(has_sun, t_sun_new, t_gnd_final))
            t_gnd_cur = t_gnd_final

    # Final converged 3-source temperatures, fluxes, canopy water, and snowpack
    post_land_fn = maybe_cuda_compile(_land_post_newton_core, device=u_1.device)
    (
        t_sun_final, t_sh_final, t_skin_final, shf_l, lhf_l, evap_l,
        e_soil, e_trans, g_into_soil, wc_next, rain_thru, melt_tot,
        dew_ground, glacier_runoff, swe_next, smb_ice_land, swe_max_next, snow_age_next,
    ) = post_land_fn(
        t_sun_cur, t_sh_cur, t_sun_new, t_sh_new, t_gnd_new, t_gnd_final, dt2_fin,
        q_melt_avail, q_af, wh_0, wh_1, wh_2, ww_0, ww_1, ww_2, gh_0, gh_1, gh_2,
        ga_land, rho_sfc, rho_cp, h_0, h_1, h_2, g_trans_sun, g_trans_sh,
        g_inter_sun, g_inter_sh, g_soil_snowfree, g_snow_subl, qsat_sun, dq_sun,
        qsat_sh, dq_sh, qsat_gnd, dq_gnd, h_soil, dry_soil_cap, dhq_gnd_dT, j22_s,
        qv_1, a_sn, has_veg, has_sun, has_sh, has_snow, is_terrestrial, is_ice,
        f_veg_rad, f_sun_area, f_sh_area, lw_refl_c, sw_abs, lw_down, lai_exp,
        wc_cap, sfc.w_canopy, w_avail_root, sfc.bio.s_root_cap, water_stress,
        sfc.swe_land, sfc.swe_max_season, sfc.snow_age, sfc.w_liq_soil[0],
        sfc.t_soil[0], precip_rain, precip_snow, dt,
    )

    # Advance 5-layer implicit Richards soil hydrology with dynamic rooting depth z_root(S_R)
    # (Clapp & Hornberger, 1978; Swenson et al., 2012; CLM5 Sec. 7.3-7.5; Holtzman et al., 2024)
    w_liq_hydro, sfc_runoff, sub_runoff = step_multilayer_soil_hydrology(
        sfc.w_liq_soil, sfc.w_ice_soil, rain_thru, melt_tot, e_soil, e_trans, dew_ground,
        sfc.bio.s_root_cap, is_ice, dt,
    )
    liq_runoff = sfc_runoff + sub_runoff

    # Advance 5-layer freeze-thaw soil heat conduction
    t_soil_new, w_liq_new, w_ice_new = step_multilayer_soil_thermodynamics(
        sfc.t_soil, w_liq_hydro, sfc.w_ice_soil, g_into_soil, dt
    )
    # Synchronize diagnostic root-zone plant-available water W_avail after freeze-thaw update
    w_root_next, _, _ = compute_root_zone_weights_and_water(w_liq_new, sfc.bio.s_root_cap)

    # =========================================================================
    # PART B: OCEAN + FRACTIONAL-LEAD SEA ICE (SCALE-RM Sec. 8.6 & MIROC6 Sec. 3.8)
    # =========================================================================
    ocn_pre_fn = maybe_cuda_compile(_surface_ocean_pre_transfer_core, device=u_1.device)
    z0m_sea, z0h_sea, z0q_sea, qsat_ow, qsat_si, theta_vsfc_o = ocn_pre_fn(
        sfc.u_star_ocean, f_ice, sfc.t_ocean_ml, sfc.t_seaice, pres_sfc, exner_s,
    )

    tr_ocn = compute_surface_transfer(
        u_1, v_1, theta_v1, theta_vsfc_o, z_ref_rel,
        z0m_sea, z0h_sea, z0q_sea, rsl_depth=None,
    )
    ga_h_o = tr_ocn.c_h * tr_ocn.wind_eff
    ga_q_o = tr_ocn.c_q * tr_ocn.wind_eff

    comp0 = sfc.t_ocean_ml_comp if sfc.t_ocean_ml_comp is not None else torch.zeros_like(sfc.t_ocean_ml)
    ocn_post_fn = maybe_cuda_compile(_ocean_seaice_post_core, device=u_1.device)
    t_ml_next, comp_final, m_si_next, t_si_new, swe_si_next, shf_o, evap_o, lhf_o, lw_emit_o = ocn_post_fn(
        sfc.t_ocean_ml, comp0, sfc.h_ocean_ml, sfc.m_seaice, sfc.t_seaice, sfc.swe_seaice,
        f_ice, qsat_ow, qsat_si, ga_h_o, ga_q_o, rho_sfc, cp_air, t_air_sfc, qv_1,
        sw_abs, sw1_dir_down, sw1_dif_down, sw2_dir_down, sw2_dif_down, lw_down,
        precip_snow, mu0, dt,
    )

    # =========================================================================
    # PART C: COMMIT PROGNOSTIC UPDATES BY SURFACE TYPE & MERGE DIAGNOSTICS
    # =========================================================================
    merge_fn = maybe_cuda_compile(_surface_part_c_merge_core, device=u_1.device)
    (
        sfc.t_soil, sfc.w_liq_soil, sfc.w_ice_soil, sfc.w_root, sfc.w_canopy,
        sfc.t_canopy, sfc.t_leaf_sun, sfc.t_leaf_sh, sfc.t_gnd_skin,
        sfc.swe_land, sfc.swe_max_season, sfc.snow_age,
        sfc.t_ocean_ml, sfc.t_ocean_ml_comp, sfc.u_star_ocean,
        sfc.m_seaice, sfc.t_seaice, sfc.swe_seaice,
        shf, lhf, evap, tau_x, tau_y, ustar_all, buoy_flux, emiss, t_rad,
        d_out, z0m_out, runoff_tot, runoff_glac, gpp_out, resp_out, an_out, transp_out, f_snow_land,
    ) = merge_fn(
        is_terrestrial, is_land, is_ocean,
        t_soil_new, w_liq_new, w_ice_new, w_root_next, wc_next,
        t_skin_final, t_sun_final, t_sh_final, t_gnd_final,
        swe_next, swe_max_next, snow_age_next,
        t_ml_next, comp_final, comp0, m_si_next, t_si_new, swe_si_next,
        sfc.t_soil, sfc.w_liq_soil, sfc.w_ice_soil, sfc.w_root, sfc.t_canopy,
        sfc.t_ocean_ml, sfc.u_star_ocean, sfc.t_seaice,
        shf_o, shf_l, lhf_o, lhf_l, evap_o, evap_l, lw_emit_o,
        tr_ocn.c_m, tr_land.c_m, tr_ocn.wind_eff, tr_land.wind_eff,
        tr_ocn.u_star, tr_land.u_star,
        rho_sfc, cp_air, exner_s, theta_1, theta_v1, qv_1, u_1, v_1,
        d_land, z0m_sea, z0m_terr, liq_runoff, glacier_runoff,
        two_leaf.gpp, two_leaf.resp_leaf, two_leaf.an_net, e_trans, a_sn,
    )

    return SurfaceFluxDiagnostics(
        shf=shf,
        lhf=lhf,
        evap=evap,
        tau_x=tau_x,
        tau_y=tau_y,
        u_star=ustar_all,
        buoy_flux=buoy_flux,
        alb_vis_dir=alb_vis_dir,
        alb_vis_dif=alb_vis_dif,
        alb_nir_dir=alb_nir_dir,
        alb_nir_dif=alb_nir_dif,
        t_rad_sfc=t_rad,
        emiss_sfc=emiss,
        d_disp=d_out,
        z0m=z0m_out,
        runoff_total=runoff_tot,
        runoff_ice=runoff_glac,
        gpp=gpp_out,
        f_seaice=f_ice,
        f_snow_land=f_snow_land,
        smb_ice_flux=smb_ice_land,
        resp_leaf=resp_out,
        an_net=an_out,
        transp=transp_out,
    )
