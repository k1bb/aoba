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
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.biosphere import (
    BiosphereState,
    canopy_two_stream_albedo_dai,
    compute_optimal_soil_water_beta,
    compute_two_leaf_gas_exchange,
    init_biosphere_state,
)
from aoba.kernels import solve_tridiagonal_batched
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
MC_SEAICE_CRIT: float = 300.0      # Critical sea-ice mass for 100% ice cover (~0.33 m) [kg m^-2]
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
    )


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

    return alb_vis_dir, alb_vis_dif, alb_nir_dir, alb_nir_dif, h_exp, lai_exp, a_sn, f_ice


# ---------------------------------------------------------------------------
# 4. Five-Layer Implicit Soil Heat Conduction & Freeze-Thaw Phase Change
# ---------------------------------------------------------------------------
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
    dz = torch.tensor(SOIL_LAYER_DEPTHS, device=dev, dtype=dtype).view((N_SOIL_LAYERS,) + (1,) * len(col_shape))

    mw = RHO_WATER * torch.clamp(w_liq, min=0.0)        # Liquid mass density [kg m^-3]
    mi = RHO_ICE_SOLID * torch.clamp(w_ice, min=0.0)    # Ice mass density [kg m^-3]
    m_tot = mw + mi

    # Volumetric heat capacity C_L [J m^-3 K^-1] and internal energy U [J m^-3] relative to T_0 = 273.15 K
    # (SCALE-RM Eqs. 8.390-8.395; measuring relative to T_0 preserves full float32 precision on MPS GPU!)
    t0 = C.T_FREEZE
    c_vol = CS_DRY_SOIL + CP_LIQ_WATER * mw + CP_ICE_SOLID * mi
    u_int = c_vol * (t_soil - t0) - C.L_F0 * mi
    u_int = u_int.clone()
    u_int[0] = u_int[0] + (dt / dz[0]) * g_ground_down

    # Exact 0 degC phase-change partition at T = T_0 (U_all_liq = 0, U_all_ice = -L_f0 * M_tot):
    mi_raw = -u_int / C.L_F0
    mi_new = torch.minimum(torch.clamp(mi_raw, min=0.0), m_tot)
    mw_new = torch.clamp(m_tot - mi_new, min=0.0)

    w_liq_new = mw_new / RHO_WATER
    w_ice_new = mi_new / RHO_ICE_SOLID
    c_vol_new = CS_DRY_SOIL + CP_LIQ_WATER * mw_new + CP_ICE_SOLID * mi_new
    dt_mid = (u_int + C.L_F0 * mi_new) / c_vol_new

    # Moisture-dependent thermal conductivity kappa [W m^-1 K^-1] (MATSIRO6 Eq. 10.4 + ice boost)
    w_tot_vol = w_liq_new + w_ice_new
    kappa = KAPPA_DRY_SOIL * (1.0 + 5.5 * torch.tanh(w_tot_vol / 0.25)) + 0.8 * w_ice_new

    # Implicit tridiagonal heat conduction in Celsius anomaly (T - T_0) for optimal float32 precision
    dz_int = 0.5 * (dz[1:] + dz[:-1])
    k_int = 0.5 * (kappa[1:] + kappa[:-1])
    cond_int = dt * k_int / dz_int

    cap_layer = c_vol_new * dz
    dl = torch.zeros_like(dt_mid)
    du = torch.zeros_like(dt_mid)
    dl[1:] = -cond_int / cap_layer[1:]
    du[:-1] = -cond_int / cap_layer[:-1]
    diag = 1.0 - dl - du
    t_final = t0 + solve_tridiagonal_batched(dl, diag, du, dt_mid)

    return t_final, w_liq_new, w_ice_new


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
    view_shape = (N_SOIL_LAYERS,) + (1,) * len(col_shape)

    dz = torch.tensor(SOIL_LAYER_DEPTHS, device=dev, dtype=dtype).view(view_shape)
    z_bot_vals = []
    acc = 0.0
    for d in SOIL_LAYER_DEPTHS:
        acc += d
        z_bot_vals.append(acc)
    z_tot = acc
    z_top = torch.tensor([0.0] + z_bot_vals[:-1], device=dev, dtype=dtype).view(view_shape)

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
    view_shape = (N_SOIL_LAYERS,) + (1,) * len(col_shape)

    dz = torch.tensor(SOIL_LAYER_DEPTHS, device=dev, dtype=dtype).view(view_shape)
    z_mid_vals = []
    acc = 0.0
    for d in SOIL_LAYER_DEPTHS:
        z_mid_vals.append(acc + 0.5 * d)
        acc += d
    dz_int = torch.tensor(
        [z_mid_vals[k + 1] - z_mid_vals[k] for k in range(N_SOIL_LAYERS - 1)],
        device=dev,
        dtype=dtype,
    ).view((N_SOIL_LAYERS - 1,) + (1,) * len(col_shape))

    w_l = torch.clamp(w_liq, min=0.0)
    w_i = torch.clamp(w_ice, min=0.0)
    theta_max = torch.clamp(SOIL_POROSITY - w_i, min=0.0)

    # Ice impedance factor f_ice = 10^(-Omega * (w_ice / phi)) (Swenson et al., 2012; CLM5 Eq. 7.89)
    ice_frac = torch.clamp(w_i / SOIL_POROSITY, 0.0, 1.0)
    f_ice = torch.pow(10.0, -ICE_IMPEDANCE_OMEGA * ice_frac)

    # Hydraulic head psi_k [m], derivative dpsi/dtheta > 0, and conductivities (Clapp & Hornberger, 1978)
    s_min = 0.02
    s_eff = torch.clamp(w_l / SOIL_POROSITY, s_min, 1.0)
    th_safe = torch.clamp(w_l, min=s_min * SOIL_POROSITY)
    psi = torch.clamp(PSI_SAT_SOIL * torch.pow(s_eff, -B_CLAPP_SOIL), min=-1.0e5)
    dpsi_dth = torch.clamp(-B_CLAPP_SOIL * psi / th_safe, max=1.0e6)

    exp_k = 2.0 * B_CLAPP_SOIL + 3.0
    s_int = 0.5 * (s_eff[:-1] + s_eff[1:])
    f_ice_int = torch.minimum(f_ice[:-1], f_ice[1:])
    k_int = f_ice_int * K_SAT_SOIL * torch.pow(s_int, exp_k)  # [N_SOIL-1, ...] [m s^-1]

    # Explicit Darcy flux q_{k+1/2} [m s^-1], positive downward (CLM5 Eq. 7.91)
    q_int = k_int * (1.0 + (psi[:-1] - psi[1:]) / dz_int)

    # Bottom free gravitational drainage q_bot = K_bot (CLM5 Sec. 7.5)
    k_bot = f_ice[-1] * K_SAT_SOIL * torch.pow(s_eff[-1], exp_k)
    dk_bot_dth = exp_k * k_bot / th_safe[-1]

    # Map-free surface infiltration & saturation/frozen/Hortonian surface runoff [kg m^-2 s^-1]
    q_liq_in = torch.clamp(rain_thru + melt_tot + dew_ground, min=0.0)
    s_top_tot = torch.clamp((w_l[0] + w_i[0]) / SOIL_POROSITY, 0.0, 1.0)
    f_imp = 1.0 - f_ice[0]
    f_sat = f_imp + (1.0 - f_imp) * torch.pow(s_top_tot, 4.0)
    q_infil_cap = RHO_WATER * f_ice[0] * K_SAT_SOIL * torch.clamp(
        1.0 + (PSI_SAT_SOIL - psi[0]) / z_mid_vals[0], min=1.0, max=1.0e4
    )
    q_infil = torch.minimum((1.0 - f_sat) * q_liq_in, q_infil_cap)
    q_infil = torch.where(is_ice, torch.zeros_like(q_infil), q_infil)
    sfc_runoff_0 = torch.where(is_ice, q_liq_in, q_liq_in - q_infil)

    # Root-zone transpiration sink weights across layers (Holtzman et al., 2024; Gao et al., 2014)
    _, w_up, _ = compute_root_zone_weights_and_water(w_l, s_root_cap)
    s_trans = (torch.clamp(e_trans, min=0.0) / RHO_WATER).unsqueeze(0) * w_up  # [N_SOIL, ...] [m s^-1]
    s_trans = torch.where(is_ice.unsqueeze(0), torch.zeros_like(s_trans), s_trans)

    # Linearized implicit tridiagonal Richards matrix (strictly diagonally dominant M-matrix; CLM5 Eqs. 7.115-7.129)
    a_int = (k_int / dz_int) * dpsi_dth[:-1]  # dq_{k+1/2}/dtheta_k >= 0
    c_int = (k_int / dz_int) * dpsi_dth[1:]   # -dq_{k+1/2}/dtheta_{k+1} >= 0

    dl = torch.zeros_like(w_l)
    du = torch.zeros_like(w_l)
    diag = (dz / dt).expand_as(w_l).clone()
    dl[1:] = -a_int
    du[:-1] = -c_int
    diag[:-1] = diag[:-1] + a_int
    diag[1:] = diag[1:] + c_int
    diag[-1] = diag[-1] + dk_bot_dth

    rhs = torch.zeros_like(w_l)
    e_soil_ms = torch.where(is_ice, torch.zeros_like(e_soil), torch.clamp(e_soil, min=0.0)) / RHO_WATER
    rhs[0] = (q_infil / RHO_WATER - e_soil_ms) - q_int[0] - s_trans[0]
    rhs[1:-1] = q_int[:-1] - q_int[1:] - s_trans[1:-1]
    rhs[-1] = q_int[-1] - k_bot - s_trans[-1]

    dtheta = solve_tridiagonal_batched(dl, diag, du, rhs)
    theta_new = w_l + dtheta

    # Linearized bottom drainage (exact mass closure if raw q_drain < 0)
    q_drain_raw = k_bot + dk_bot_dth * dtheta[-1]
    q_drain_neg = torch.clamp(-q_drain_raw, min=0.0)
    q_drain = torch.clamp(q_drain_raw, min=0.0)
    theta_new = theta_new.clone()
    theta_new[-1] = theta_new[-1] - (dt / SOIL_LAYER_DEPTHS[-1]) * q_drain_neg

    # Exact mass-conserving bounds: (1) bottom-up supersaturation sweep -> sfc_runoff
    excess_mass = torch.zeros_like(w_l[0])
    for k in range(N_SOIL_LAYERS - 1, -1, -1):
        th_k = theta_new[k] + excess_mass / (RHO_WATER * SOIL_LAYER_DEPTHS[k])
        over = torch.clamp(th_k - theta_max[k], min=0.0)
        theta_new[k] = th_k - over
        excess_mass = over * (RHO_WATER * SOIL_LAYER_DEPTHS[k])
    sfc_runoff = sfc_runoff_0 + excess_mass / dt

    # (2) Top-down non-negativity guard borrowing from neighbor / bottom drainage
    deficit_mass = torch.zeros_like(w_l[0])
    for k in range(N_SOIL_LAYERS):
        th_k = theta_new[k] - deficit_mass / (RHO_WATER * SOIL_LAYER_DEPTHS[k])
        under = torch.clamp(-th_k, min=0.0)
        theta_new[k] = th_k + under
        deficit_mass = under * (RHO_WATER * SOIL_LAYER_DEPTHS[k])

    q_drain_mass = RHO_WATER * q_drain * dt
    reduce_drain = torch.minimum(deficit_mass, q_drain_mass)
    q_drain_mass = q_drain_mass - reduce_drain
    sub_runoff = torch.where(is_ice, torch.zeros_like(q_drain_mass), q_drain_mass / dt)
    w_liq_next = torch.where(is_ice.unsqueeze(0), w_liq, theta_new)
    return w_liq_next, sfc_runoff, sub_runoff


# ---------------------------------------------------------------------------
# 5. Unified Land / Ice-Sheet / Ocean+Sea-Ice Time-Stepping Driver
# ---------------------------------------------------------------------------
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
    ) = compute_surface_albedos(sfc, mu0)

    # Absorbed net shortwave [W m^-2]
    sw_abs = (
        sw1_dir_down * (1.0 - alb_vis_dir)
        + sw1_dif_down * (1.0 - alb_vis_dif)
        + sw2_dir_down * (1.0 - alb_nir_dir)
        + sw2_dif_down * (1.0 - alb_nir_dif)
    )

    # Reference-level potential temperature and virtual potential temperature
    exner_1 = torch.pow(pres_1 / C.P_REF, C.KAPPA_D)
    exner_s = torch.pow(pres_sfc / C.P_REF, C.KAPPA_D)
    theta_1 = temp_1 / exner_1
    theta_v1 = theta_1 * (1.0 + C.ZVIR * qv_1)
    rho_sfc = pres_sfc / (C.R_D * temp_1 * (1.0 + C.ZVIR * qv_1))
    cp_air = C.CP_D + C.CP_V * qv_1

    # =========================================================================
    # PART A: TERRESTRIAL TILES (Vegetated/Bare Land & Grounded Ice Sheets)
    # =========================================================================
    d_land, z0m_land, z0h_land = canopy_roughness_raupach(h_exp, lai_exp, z0_soil=0.01)
    # Smooth snow roughness when snow-covered and canopy buried
    z0m_snow = torch.full_like(z0m_land, 0.002)
    z0m_terr = torch.where(lai_exp < 0.05, (1.0 - a_sn) * z0m_land + a_sn * z0m_snow, z0m_land)
    z0h_terr = 0.135335 * z0m_terr
    rsl_depth = torch.clamp(2.0 * h_exp - d_land, min=0.0)

    # Effective composite land skin temperature prior to step
    f_veg_rad = 1.0 - torch.exp(-lai_exp)
    t_gnd = sfc.t_soil[0]
    t_skin_land = f_veg_rad * sfc.t_canopy + (1.0 - f_veg_rad) * t_gnd
    qsat_skin_l, dqsat_dT_l = qsat_mixed(t_skin_land, pres_sfc)
    theta_vsfc_l = (t_skin_land / exner_s) * (1.0 + C.ZVIR * qsat_skin_l)

    tr_land = compute_surface_transfer(
        u_1, v_1, theta_v1, theta_vsfc_l, z_ref_rel,
        z0m_terr, z0h_terr, z0h_terr, rsl_depth=rsl_depth,
    )
    ga_land = tr_land.c_h * tr_land.wind_eff  # Aerodynamic conductance 1 / r_a [m s^-1]

    # Vertically integrated root-zone plant-available liquid water & optimal stress beta (Holtzman et al., 2024; Gao et al., 2014)
    w_avail_root, _, _ = compute_root_zone_weights_and_water(sfc.w_liq_soil, sfc.bio.s_root_cap)
    water_stress = compute_optimal_soil_water_beta(w_avail_root, sfc.bio)

    # Sunlit/shaded leaf temperatures: sunlit leaves are slightly warmer under strong direct beam (Dai et al., 2004)
    gb_leaf = torch.clamp(0.01 * torch.sqrt(tr_land.u_star / 0.04) * torch.clamp(lai_exp, min=0.1), min=1.0e-3)
    dt_sun_est = torch.clamp(0.35 * sw1_dir_down / (rho_sfc * cp_air * (gb_leaf + ga_land) + 20.0), 0.0, 5.0)
    t_sun = sfc.t_canopy + 0.5 * dt_sun_est
    t_sh = sfc.t_canopy - 0.2 * dt_sun_est

    ea_pa = qv_1 * pres_sfc / (C.EPS_WATER + (1.0 - C.EPS_WATER) * qv_1)
    vpd_sun = torch.clamp(esat_liquid(t_sun) - ea_pa, min=20.0)
    vpd_sh = torch.clamp(esat_liquid(t_sh) - ea_pa, min=20.0)

    vis_beam = C.PAR_FRAC_OF_SW1 * sw1_dir_down
    vis_diff = C.PAR_FRAC_OF_SW1 * sw1_dif_down
    two_leaf = compute_two_leaf_gas_exchange(
        sfc.bio, vis_beam, vis_diff, mu0, t_sun, t_sh, vpd_sun, vpd_sh,
        pres_sfc, co2_ppmv, lai_exp, water_stress,
    )
    gs_canopy = two_leaf.gsw_sun + two_leaf.gsw_sh  # Canopy stomatal conductance [m s^-1]

    # Wetted canopy fraction f_wet = [W_can / (p_liq L)]^(2/3) <= f_wet_max (CLM5 Eq. 7.23; Dickinson et al., 1993)
    wc_cap = WC_MAX_PER_LAI * lai_exp
    f_wet = torch.where(
        lai_exp >= 0.01,
        torch.clamp(
            torch.pow(torch.clamp(sfc.w_canopy, min=0.0) / torch.clamp(wc_cap, min=1.0e-4), 2.0 / 3.0),
            0.0,
            F_WET_MAX,
        ),
        torch.zeros_like(lai_exp),
    )

    # Bare-soil evaporation resistance via the CLM5 dry-surface-layer scheme (Swenson & Lawrence, 2014)
    # Surface wetness s_1 from liquid + ice (CLM5 Eq. 5.75) and Philip (1957) soil-air humidity factor (Eq. 5.73)
    sat_top = torch.clamp((sfc.w_liq_soil[0] + sfc.w_ice_soil[0]) / SOIL_POROSITY, 0.01, 1.0)
    psi_top = PSI_SAT_SOIL * torch.pow(sat_top, -B_CLAPP_SOIL)  # Matric head [m]
    h_soil_base = torch.exp(torch.clamp(psi_top * C.GRAVITY / (C.R_V * t_gnd), -10.0, 0.0))

    # Dry-surface-layer thickness D_SL, air-dry moisture theta_air, tortuosity tau, and vapor diffusivity D_v (Eqs. 5.77-5.81)
    theta_liq_top = torch.clamp(sfc.w_liq_soil[0], min=0.0, max=SOIL_POROSITY)
    theta_air = SOIL_POROSITY * (abs(PSI_SAT_SOIL) * 1.0e3 / PSI_AIR_MM) ** (1.0 / B_CLAPP_SOIL)
    theta_init = DSL_THETA_INIT_FRAC * SOIL_POROSITY
    d_sl = DSL_DMAX_MM * 1.0e-3 * torch.clamp(
        (theta_init - theta_liq_top) / max(theta_init - theta_air, 1.0e-3), min=0.0, max=13.0
    )
    phi_air = max(SOIL_POROSITY - theta_air, 1.0e-3)
    tau_soil = (phi_air ** 2) * (phi_air / SOIL_POROSITY) ** (3.0 / B_CLAPP_SOIL)
    d_vap = 2.12e-5 * torch.pow(t_gnd / C.T_FREEZE, 1.75)
    r_soil = d_sl / (d_vap * tau_soil)  # [s m^-1]

    # Fixed-count Newton iteration of the composite-skin energy balance (CLM5 Sec. 5.3 iterates to convergence):
    # transfer coefficients, leaf/under-canopy conductances and the linearized balance are re-evaluated at each iterate.
    # Stomatal conductance (the two-leaf solve above) stays at its start-of-step value.
    # LAND_EB_ITERATIONS = 1 reproduces a single linearized pass about the previous-step skin temperature (default 3).
    n_eb_iter = max(int(LAND_EB_ITERATIONS), 1)
    for _eb_it in range(n_eb_iter):
        if _eb_it > 0:
            qsat_skin_l, dqsat_dT_l = qsat_mixed(t_skin_land, pres_sfc)
            theta_vsfc_l = (t_skin_land / exner_s) * (1.0 + C.ZVIR * qsat_skin_l)
            tr_land = compute_surface_transfer(
                u_1, v_1, theta_v1, theta_vsfc_l, z_ref_rel,
                z0m_terr, z0h_terr, z0h_terr, rsl_depth=rsl_depth,
            )
            ga_land = tr_land.c_h * tr_land.wind_eff
            gb_leaf = torch.clamp(0.01 * torch.sqrt(tr_land.u_star / 0.04) * torch.clamp(lai_exp, min=0.1), min=1.0e-3)

        # CLM5 Sec. 5.2: do not condense atmospheric vapor into very dry soil (q_soil = q_atm when q_soil < q_atm < q_sat)
        h_soil = torch.where(
            (qsat_skin_l > qv_1) & (qv_1 > h_soil_base * qsat_skin_l),
            qv_1 / torch.clamp(qsat_skin_l, min=1.0e-8),
            h_soil_base,
        )

        # Under-canopy stability-corrected aerodynamic resistance r_d (Sakaguchi & Zeng, 2009; SLM Eqs. A10-A14)
        stab_uc = C.GRAVITY * h_exp * (sfc.t_canopy - t_gnd) / (t_gnd * (tr_land.u_star ** 2 + 1.0e-3))
        c_dense = torch.where(stab_uc > 0.0, 0.004 / (1.0 + 0.5 * torch.clamp(stab_uc, max=10.0)), torch.full_like(stab_uc, 0.004))
        c_bare = C.VON_KARMAN / (0.13 * torch.pow(torch.clamp(tr_land.u_star * z0m_terr / 1.5e-5, min=1.0), 0.45))
        c_uv = (1.0 - f_veg_rad) * c_bare + f_veg_rad * c_dense
        r_d_uc = 1.0 / torch.clamp(c_uv * tr_land.u_star, min=1.0e-3)

        # Conductances for the 3 ET pathways [m s^-1]:
        # CLM5 Eqs. 5.109-5.113: transpiration acts on the dry foliage fraction (1 - f_wet), while the wet-canopy term
        # is limited by available canopy water, f_wet' = min(f_wet, W_can / (dt * E_pot)); dew (E_pot <= 0) wets the whole leaf.
        qleaf_sat, _ = qsat_mixed(sfc.t_canopy, pres_sfc)
        e_pot_wet = rho_sfc * gb_leaf * (qleaf_sat - qv_1)  # Potential flux of a fully wetted canopy [kg m^-2 s^-1]
        dew_on_leaf = (e_pot_wet <= 0.0) & (lai_exp >= 0.01)
        f_wet_eff = torch.where(
            e_pot_wet > 1.0e-12,
            torch.minimum(f_wet, torch.clamp(sfc.w_canopy, min=0.0) / (dt * torch.clamp(e_pot_wet, min=1.0e-12))),
            f_wet,
        )
        f_wet_eff = torch.where(dew_on_leaf, torch.ones_like(f_wet_eff), f_wet_eff)

        # 1. Stomatal transpiration: series of gb_leaf and gs_canopy on dry foliage (1 - f_wet); none while dew wets the leaf
        g_trans = torch.where(
            dew_on_leaf,
            torch.zeros_like(f_wet),
            (1.0 - f_wet) * (gb_leaf * gs_canopy) / torch.clamp(gb_leaf + gs_canopy, min=1.0e-6),
        )
        # 2. Wet-canopy interception evaporation: boundary conductance gb_leaf on the water-limited wet fraction
        g_inter = f_wet_eff * gb_leaf
        # 3. Ground evaporation (snow-free soil + snow sublimation on A_sn; CLM5 Eq. 5.110).
        # Atmospheric resistance 1/g_a is applied once below in series with g_surf_vap.
        g_soil_snowfree = (1.0 - a_sn) / torch.clamp(r_d_uc + r_soil, min=1.0e-4)
        g_snow_subl = a_sn / torch.clamp(r_d_uc, min=1.0e-4)

        g_surf_vap = g_trans + g_inter + g_soil_snowfree + g_snow_subl
        # Total series conductance with atmospheric reference layer r_a
        g_evap_eff = (ga_land * g_surf_vap) / torch.clamp(ga_land + g_surf_vap, min=1.0e-6)
        h_eff_sfc = torch.where(
            g_surf_vap > 1.0e-6,
            (g_trans + g_inter + g_snow_subl + g_soil_snowfree * h_soil) / torch.clamp(g_surf_vap, min=1.0e-6),
            h_soil,
        )

        l_vap_land = torch.where((a_sn > 0.5) | (t_skin_land < C.T_FREEZE), C.L_S0, C.L_V0)

        # Implicit linearized surface energy balance for Delta_T_skin (MATSIRO6 Sec. 6.3)
        # R_n(T + dT) - H(T + dT) - L * E(T + dT) - G_cond(T + dT) = 0
        lw_net0 = EMISSIVITY_LAND * (lw_down - C.STEFAN_BOLTZMANN * (t_skin_land ** 4))
        rn0 = sw_abs + lw_net0
        shf0_l = rho_sfc * cp_air * ga_land * (t_skin_land - theta_1 * exner_s)
        evap0_l = rho_sfc * g_evap_eff * (h_eff_sfc * qsat_skin_l - qv_1)
        # Bound upward evaporation by available water stores
        evap0_l = torch.maximum(evap0_l, -rho_sfc * ga_land * qv_1)

        dz_top_half = 0.5 * SOIL_LAYER_DEPTHS[0]
        k_top = KAPPA_DRY_SOIL * (1.0 + 5.5 * torch.tanh((sfc.w_liq_soil[0] + sfc.w_ice_soil[0]) / 0.25))
        # Thermal resistance of snowpack above topsoil when A_sn > 0 (units: m^2 K W^-1)
        snow_depth = sfc.swe_land / RHO_SNOWPACK
        r_snow_therm = a_sn * (0.5 * torch.clamp(snow_depth, max=1.0) / C.K_THERM_SNOW)
        g_cond_coeff = 1.0 / (dz_top_half / k_top + r_snow_therm)
        g0_down = g_cond_coeff * (t_skin_land - sfc.t_soil[0])

        drn_dT = -4.0 * EMISSIVITY_LAND * C.STEFAN_BOLTZMANN * (t_skin_land ** 3)
        dshf_dT = rho_sfc * cp_air * ga_land
        dlhf_dT = l_vap_land * rho_sfc * g_evap_eff * h_eff_sfc * dqsat_dT_l
        dg_dT = g_cond_coeff

        denom_eb = dshf_dT + dlhf_dT + dg_dT - drn_dT
        dt_skin = torch.clamp((rn0 - shf0_l - l_vap_land * evap0_l - g0_down) / torch.clamp(denom_eb, min=1.0), -10.0, 10.0)
        t_skin_new = t_skin_land + dt_skin

        # Snowmelt cap at T_FREEZE when snow is present (MATSIRO6 Sec. 6.3.3)
        has_snow = sfc.swe_land > 1.0e-3
        melt_excess_t = torch.where(has_snow & (t_skin_new > C.T_FREEZE), t_skin_new - C.T_FREEZE, torch.zeros_like(t_skin_new))
        t_skin_final = t_skin_new - melt_excess_t
        dt_final = t_skin_final - t_skin_land
        q_melt_avail = melt_excess_t * denom_eb  # Energy [W m^-2] available to melt snow

        if _eb_it < n_eb_iter - 1:
            t_skin_land = t_skin_final  # Next Newton iterate

    shf_l = shf0_l + dshf_dT * dt_final
    evap_raw_l = evap0_l + rho_sfc * g_evap_eff * h_eff_sfc * dqsat_dT_l * dt_final

    # Partition total upward terrestrial water flux across interception, transpiration, snow, and soil
    w_tot_g = torch.clamp(g_surf_vap, min=1.0e-8)
    evap_pos = torch.clamp(evap_raw_l, min=0.0)
    dew_neg = torch.clamp(evap_raw_l, max=0.0)

    # Cap wet-canopy evaporation by W_can / dt and transpiration by SPLASH supply S_w (Davis et al., 2017);
    # the latent-heat shortfall of the wet-canopy cap is returned as sensible heat (CLM5 Eq. 5.139)
    e_inter_raw = evap_pos * (g_inter / w_tot_g)
    e_inter = torch.minimum(e_inter_raw, torch.where(is_terrestrial, sfc.w_canopy / dt, evap_pos))
    shf_l = shf_l + l_vap_land * (e_inter_raw - e_inter)
    sw_max_rate = (1.05 / 3600.0) * water_stress * (sfc.bio.s_root_cap / 150.0)  # Max supply ~1.05 mm/hr [kg m^-2 s^-1]
    e_trans = torch.minimum(
        torch.minimum(evap_pos * (g_trans / w_tot_g), sw_max_rate),
        torch.where(is_terrestrial, w_avail_root / dt, evap_pos),
    )
    e_snow = torch.minimum(evap_pos * (g_snow_subl / w_tot_g), torch.where(is_terrestrial, sfc.swe_land / dt, evap_pos))
    max_soil_evap = RHO_WATER * SOIL_LAYER_DEPTHS[0] * torch.clamp(sfc.w_liq_soil[0] - 0.02, min=0.0) / dt
    e_soil = torch.minimum(evap_pos * (g_soil_snowfree / w_tot_g), torch.where(is_terrestrial, max_soil_evap, evap_pos))

    evap_l = e_inter + e_trans + e_snow + e_soil + dew_neg
    lhf_l = l_vap_land * evap_l
    lw_up_l = EMISSIVITY_LAND * C.STEFAN_BOLTZMANN * (t_skin_final ** 4) + (1.0 - EMISSIVITY_LAND) * lw_down
    g_into_soil = sw_abs + lw_down - lw_up_l - shf_l - lhf_l - q_melt_avail

    # Canopy water balance: liquid interception f_pi = alpha_liq * tanh(L), throughfall, and drip (CLM5 Eqs. 7.2-7.10, 7.17)
    # Snowfall is not stored in the canopy (no canopy-snow reservoir) and falls directly onto the snowpack.
    f_int = torch.clamp(ALPHA_INTERCEPT_LIQ * torch.tanh(lai_exp), 0.0, 1.0)
    wc_mid = sfc.w_canopy + dt * (f_int * precip_rain - e_inter - torch.where(f_veg_rad > 0.1, 0.5 * dew_neg, torch.zeros_like(dew_neg)))
    drip = torch.clamp(wc_mid - wc_cap, min=0.0) / dt
    wc_next = torch.clamp(wc_mid - drip * dt, min=0.0)
    rain_thru = (1.0 - f_int) * precip_rain + drip
    snow_thru = precip_snow

    # Update snowpack SWE, melt, snow aging, and glacier calving cap (MATSIRO6 Ch. 8)
    swe_mid = torch.clamp(sfc.swe_land + dt * (snow_thru - e_snow), min=0.0)
    melt_rate = torch.minimum(q_melt_avail / C.L_F0, swe_mid / dt)
    # Immediate melt of thin snow falling on warm unfrozen ground
    warm_ground_melt = torch.where((sfc.t_soil[0] > C.T_FREEZE + 1.0) & (swe_mid < 2.0), swe_mid / dt, torch.zeros_like(swe_mid))
    melt_tot = torch.minimum(melt_rate + warm_ground_melt, swe_mid / dt)
    g_into_soil = g_into_soil - C.L_F0 * (melt_tot - melt_rate)
    swe_after_melt = torch.clamp(swe_mid - melt_tot * dt, min=0.0)

    # Glacier calving/runoff cap at 1000 kg m^-2 (MATSIRO6 Sec. 8.6 Eq. 8.44)
    glacier_runoff = torch.clamp(swe_after_melt - SWE_GLACIER_MAX, min=0.0) / dt
    swe_next = swe_after_melt - glacier_runoff * dt
    bare_ice_melt_pot = torch.where(
        (~has_snow) & (t_skin_new > C.T_FREEZE),
        (t_skin_new - C.T_FREEZE) * denom_eb / C.L_F0,
        torch.zeros_like(t_skin_new),
    )
    smb_ice_land = torch.where(
        is_terrestrial,
        snow_thru - e_snow - melt_tot - bare_ice_melt_pot,
        torch.zeros_like(snow_thru),
    )

    # Seasonal peak SWE hysteresis update (Swenson & Lawrence, 2012)
    swe_max_next = torch.where(
        swe_next <= 1.0e-2,
        torch.zeros_like(swe_next),
        torch.maximum(sfc.swe_max_season, swe_next),
    )
    # Snow aging update (Yang et al., 1997; MATSIRO6 Eqs. 8.60-8.61; fresh 10 mm snow resets age)
    t_sn = torch.clamp(t_skin_final, max=C.T_FREEZE)
    f_age = torch.exp(5000.0 * (1.0 / C.T_FREEZE - 1.0 / torch.clamp(t_sn, min=200.0)))
    r_dirt = torch.where(is_ice, torch.full_like(f_age, 0.01), torch.full_like(f_age, 0.30))
    age_inc = (f_age + torch.pow(f_age, 10.0) + r_dirt) * (dt / TAU_SNOW_AGE)
    fresh_refresh = torch.clamp(snow_thru * dt / 10.0, 0.0, 1.0)
    snow_age_next = torch.where(
        swe_next <= 1.0e-3,
        torch.zeros_like(sfc.snow_age),
        (1.0 - fresh_refresh) * (sfc.snow_age + age_inc),
    )

    # Advance 5-layer implicit Richards soil hydrology with dynamic rooting depth z_root(S_R)
    # (Clapp & Hornberger, 1978; Swenson et al., 2012; CLM5 Sec. 7.3-7.5; Holtzman et al., 2024)
    dew_ground = -(1.0 - 0.5 * (f_veg_rad > 0.1).to(dt_final.dtype)) * dew_neg  # Positive downward dew [kg m^-2 s^-1]
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
    z0m_ocn, z0h_ocn, z0q_ocn = ocean_roughness_miller(sfc.u_star_ocean)
    z0m_ice = torch.full_like(z0m_ocn, 0.001)
    z0m_sea = (1.0 - f_ice) * z0m_ocn + f_ice * z0m_ice
    z0h_sea = (1.0 - f_ice) * z0h_ocn + f_ice * (0.2 * z0m_ice)
    z0q_sea = (1.0 - f_ice) * z0q_ocn + f_ice * (0.2 * z0m_ice)

    # Open-water saturation specific humidity (0.98 salinity reduction; MIROC6) & sea-ice saturation
    qsat_ow = 0.98 * C.EPS_WATER * esat_liquid(sfc.t_ocean_ml) / torch.clamp(pres_sfc - esat_liquid(sfc.t_ocean_ml), min=0.1 * pres_sfc)
    qsat_si = C.EPS_WATER * esat_ice(sfc.t_seaice) / torch.clamp(pres_sfc - esat_ice(sfc.t_seaice), min=0.1 * pres_sfc)
    t_skin_sea = (1.0 - f_ice) * sfc.t_ocean_ml + f_ice * sfc.t_seaice
    q_skin_sea = (1.0 - f_ice) * qsat_ow + f_ice * qsat_si
    theta_vsfc_o = (t_skin_sea / exner_s) * (1.0 + C.ZVIR * q_skin_sea)

    tr_ocn = compute_surface_transfer(
        u_1, v_1, theta_v1, theta_vsfc_o, z_ref_rel,
        z0m_sea, z0h_sea, z0q_sea, rsl_depth=None,
    )
    ga_h_o = tr_ocn.c_h * tr_ocn.wind_eff
    ga_q_o = tr_ocn.c_q * tr_ocn.wind_eff

    # Separate fluxes over open water and sea ice (MIROC6 Sec. 3.8.4 & SCALE-RM Sec. 8.6.2)
    shf_ow = rho_sfc * cp_air * ga_h_o * (sfc.t_ocean_ml - theta_1 * exner_s)
    evap_ow = rho_sfc * ga_q_o * (qsat_ow - qv_1)
    lhf_ow = C.L_V0 * evap_ow
    lw_up_ow = EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * (sfc.t_ocean_ml ** 4) + (1.0 - EMISSIVITY_OCEAN) * lw_down
    sw_abs_ow = (
        sw1_dir_down * (1.0 - compute_ocean_albedo_briegleb(mu0)[0])
        + sw1_dif_down * 0.94
        + sw2_dir_down * (1.0 - compute_ocean_albedo_briegleb(mu0)[0])
        + sw2_dif_down * 0.94
    )
    # Net downward heat into open water including snowfall fusion cooling -L_f0 * F_snow (SCALE-RM Eq. 8.332)
    g_down_ow = sw_abs_ow + lw_down - lw_up_ow - shf_ow - lhf_ow - C.L_F0 * precip_snow

    # Sea-ice skin & conductive flux G_oi (Semtner, 1976; SCALE-RM Eqs. 8.340-8.358)
    h_ice_eff = torch.clamp(sfc.m_seaice / (C.RHO_SEAICE * torch.clamp(f_ice, min=0.05)), min=0.05)
    h_sn_ice = sfc.swe_seaice / RHO_SNOWPACK
    cond_ice = 1.0 / (0.5 * h_ice_eff / C.K_THERM_SEAICE + h_sn_ice / C.K_THERM_SNOW)

    shf_si0 = rho_sfc * cp_air * ga_h_o * (sfc.t_seaice - theta_1 * exner_s)
    subl_si0 = rho_sfc * ga_q_o * (qsat_si - qv_1)
    lw_up_si0 = EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * (sfc.t_seaice ** 4) + (1.0 - EMISSIVITY_OCEAN) * lw_down
    g_cond_si0 = cond_ice * (sfc.t_seaice - C.T_FREEZE_OCEAN)
    # Linearized sea-ice skin update
    dqsi_dT = qsat_si * (21.8745584 * (C.T_FREEZE - 7.66)) / (torch.clamp(sfc.t_seaice, min=180.0) - 7.66) ** 2
    denom_si = (
        rho_sfc * cp_air * ga_h_o
        + C.L_S0 * rho_sfc * ga_q_o * dqsi_dT
        + cond_ice
        + 4.0 * EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * (sfc.t_seaice ** 3)
    )
    res_si = sw_abs + lw_down - lw_up_si0 - shf_si0 - C.L_S0 * subl_si0 - g_cond_si0
    t_si_raw = sfc.t_seaice + torch.clamp(res_si / denom_si, -10.0, 10.0)
    si_melt_excess = torch.clamp(t_si_raw - C.T_FREEZE, min=0.0)
    t_si_new = torch.where(f_ice > 1.0e-4, t_si_raw - si_melt_excess, torch.full_like(t_si_raw, C.T_FREEZE_OCEAN))
    dt_si = t_si_new - sfc.t_seaice

    shf_si = shf_si0 + rho_sfc * cp_air * ga_h_o * dt_si
    subl_si = subl_si0 + rho_sfc * ga_q_o * dqsi_dT * dt_si
    lhf_si = C.L_S0 * subl_si
    g_oi_up = cond_ice * (C.T_FREEZE_OCEAN - t_si_new)  # Upward conductive heat from ocean to ice

    # Snow on sea ice + top/basal sea-ice mass update (SCALE-RM Eqs. 8.345-8.358)
    swe_si_mid = torch.clamp(sfc.swe_seaice + f_ice * dt * (precip_snow - torch.clamp(subl_si, min=0.0)), min=0.0)
    q_top_melt = f_ice * si_melt_excess * denom_si
    sn_si_melt = torch.minimum(q_top_melt * dt / C.L_F0, swe_si_mid)
    swe_si_next = swe_si_mid - sn_si_melt
    q_rem_melt = torch.clamp(q_top_melt * dt - C.L_F0 * sn_si_melt, min=0.0)

    # Update ocean mixed-layer temperature T_ocean_ml before basal freeze/melt check
    cw_ml = C.RHO_SEAWATER * C.CP_SEAWATER * sfc.h_ocean_ml
    dt_ml_flux = (dt / cw_ml) * ((1.0 - f_ice) * g_down_ow - f_ice * g_oi_up)
    comp0 = sfc.t_ocean_ml_comp if sfc.t_ocean_ml_comp is not None else torch.zeros_like(sfc.t_ocean_ml)
    t_ml_prov, comp1 = _kahan_add(sfc.t_ocean_ml, dt_ml_flux, comp0)

    # If T_ml_prov < T_FREEZE_OCEAN: form new sea ice; if T_ml_prov > T_FREEZE_OCEAN and M_i > 0: basal melt
    e_freeze = cw_ml * torch.clamp(C.T_FREEZE_OCEAN - t_ml_prov, min=0.0)
    e_basal_melt = cw_ml * torch.clamp(t_ml_prov - C.T_FREEZE_OCEAN, min=0.0)
    m_freeze = e_freeze / C.L_F0
    m_melt = torch.minimum((e_basal_melt + q_rem_melt) / C.L_F0, sfc.m_seaice + m_freeze)

    m_si_next = torch.clamp(sfc.m_seaice + m_freeze - m_melt, min=0.0)
    # Net enthalpy adjustment on mixed layer after freeze/melt
    dt_ml_lat = (C.L_F0 * m_freeze - C.L_F0 * torch.minimum(e_basal_melt / C.L_F0, sfc.m_seaice)) / cw_ml
    t_ml_next, comp2 = _kahan_add(t_ml_prov, dt_ml_lat, comp1)
    comp_final = torch.where((e_freeze > 0.0) | (m_melt > 0.0), torch.zeros_like(comp2), comp2)

    shf_o = (1.0 - f_ice) * shf_ow + f_ice * shf_si
    evap_o = (1.0 - f_ice) * evap_ow + f_ice * subl_si
    lhf_o = (1.0 - f_ice) * lhf_ow + f_ice * lhf_si
    lw_emit_o = EMISSIVITY_OCEAN * C.STEFAN_BOLTZMANN * (
        (1.0 - f_ice) * (sfc.t_ocean_ml ** 4) + f_ice * (t_si_new ** 4)
    )

    # =========================================================================
    # PART C: COMMIT PROGNOSTIC UPDATES BY SURFACE TYPE & MERGE DIAGNOSTICS
    # =========================================================================
    sfc.t_soil = torch.where(is_terrestrial.unsqueeze(0), t_soil_new, sfc.t_soil)
    sfc.w_liq_soil = torch.where(is_terrestrial.unsqueeze(0), w_liq_new, sfc.w_liq_soil)
    sfc.w_ice_soil = torch.where(is_terrestrial.unsqueeze(0), w_ice_new, sfc.w_ice_soil)
    sfc.w_root = torch.where(is_land, w_root_next, sfc.w_root)
    sfc.w_canopy = torch.where(is_land, wc_next, torch.zeros_like(wc_next))
    sfc.t_canopy = torch.where(is_terrestrial, t_skin_final, sfc.t_canopy)
    sfc.swe_land = torch.where(is_terrestrial, swe_next, torch.zeros_like(swe_next))
    sfc.swe_max_season = torch.where(is_terrestrial, swe_max_next, torch.zeros_like(swe_max_next))
    sfc.snow_age = torch.where(is_terrestrial, snow_age_next, torch.zeros_like(snow_age_next))

    sfc.t_ocean_ml = torch.where(is_ocean, t_ml_next, sfc.t_ocean_ml)
    sfc.t_ocean_ml_comp = torch.where(is_ocean, comp_final, comp0)
    sfc.u_star_ocean = torch.where(is_ocean, tr_ocn.u_star, sfc.u_star_ocean)
    sfc.m_seaice = torch.where(is_ocean, m_si_next, torch.zeros_like(m_si_next))
    sfc.t_seaice = torch.where(is_ocean, t_si_new, sfc.t_seaice)
    sfc.swe_seaice = torch.where(is_ocean, swe_si_next, torch.zeros_like(swe_si_next))

    shf = torch.where(is_ocean, shf_o, shf_l)
    lhf = torch.where(is_ocean, lhf_o, lhf_l)
    evap = torch.where(is_ocean, evap_o, evap_l)

    cm_all = torch.where(is_ocean, tr_ocn.c_m, tr_land.c_m)
    wind_all = torch.where(is_ocean, tr_ocn.wind_eff, tr_land.wind_eff)
    ustar_all = torch.where(is_ocean, tr_ocn.u_star, tr_land.u_star)
    tau_x = -rho_sfc * cm_all * wind_all * u_1
    tau_y = -rho_sfc * cm_all * wind_all * v_1

    # Kinematic surface buoyancy flux B_sfc = (g / theta_v1) * [H / (rho c_p Exner) + 0.608 * theta_1 * E / rho]
    w_th_sfc = shf / (rho_sfc * cp_air * exner_s)
    w_qv_sfc = evap / rho_sfc
    buoy_flux = (C.GRAVITY / theta_v1) * (w_th_sfc * (1.0 + C.ZVIR * qv_1) + C.ZVIR * theta_1 * w_qv_sfc)

    emiss = torch.where(is_ocean, torch.full_like(shf, EMISSIVITY_OCEAN), torch.full_like(shf, EMISSIVITY_LAND))
    lw_emit_l = EMISSIVITY_LAND * C.STEFAN_BOLTZMANN * (t_skin_final ** 4)
    t_rad = torch.pow(
        torch.where(is_ocean, lw_emit_o, lw_emit_l) / (emiss * C.STEFAN_BOLTZMANN),
        0.25,
    )

    d_out = torch.where(is_ocean, torch.zeros_like(d_land), d_land)
    z0m_out = torch.where(is_ocean, z0m_sea, z0m_terr)
    runoff_tot = torch.where(is_terrestrial, liq_runoff + glacier_runoff, torch.zeros_like(liq_runoff))
    runoff_glac = torch.where(is_terrestrial, glacier_runoff, torch.zeros_like(glacier_runoff))
    gpp_out = torch.where(is_land, two_leaf.gpp, torch.zeros_like(two_leaf.gpp))

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
        f_snow_land=torch.where(is_terrestrial, a_sn, torch.zeros_like(a_sn)),
        smb_ice_flux=smb_ice_land,
    )
