"""Global 1:1 Superparameterized AOBA Coupled Earth System Model Driver.

Couples all AOBA subsystems on CPU or Apple Silicon `mps` GPU using **only** minimal external
physical boundary conditions (`Earth2014` geometry/elevation, planetary rotation `Omega`,
orbital insolation `S_0`, and well-mixed `CO2` / `O2`):
1. Global SCVT Voronoi C-Grid + Charney-Phillips Nonhydrostatic RK3-HEVI Host (`scvt_dynamics.py`)
2. Lott & Miller (1997) Subgrid Orographic Drag (`orography.py`)
3. 1:1 Embedded 3D Nonhydrostatic HEVI CRM + Zalesak FCT + Smagorinsky-Lilly SGS (`crm.py`)
4. Implicit MYNN Level 2.5 PBL + Analytical Roughness Sublayer (RSL) Turbulence (`turbulence.py`)
5. SAM 1-Moment Mixed-Phase Cloud Microphysics & Sedimentation (`microphysics.py`)
6. 4-Band Two-Stream Window/Non-Window Radiative Transfer (`radiation.py`)
7. Two-Leaf Non-Rectangular-Hyperbola Smith/Scott-Smith EEO Biosphere + Prognostic LAI (`biosphere.py`)
8. 5-Layer Freeze-Thaw Permafrost Soil, Snow-Burial Albedo, & Glacier Calving (`surface.py`)
9. Zero-Q-Flux 1.5-Layer Wind-Driven Voronoi Ekman Ocean & Fractional-Lead Sea Ice (`ocean_ekman.py`)
10. Emergent 30-Class Köppen-Geiger Climate Diagnostics (`koppen.py`)

AOBA Formulation Reference: Sec. 1.1 Eqs. (1.1)-(1.4), Sec. 8.6 Eq. (8.13) & Sec. 9.5 Eq. (9.8) (Coupled Time-Stepping, SP/GCRM & Current-Relative Coupling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import torch

from aoba import constants as C
from aoba.calendar import AobaCalendar
from aoba.co2_forcing import CO2ForcingSeries, resolve_co2_series
from aoba.ozone_forcing import OzoneForcingProvider, resolve_ozone_provider
from aoba.biosphere import (
    BiosphereState,
    compute_optimal_soil_water_beta,
    update_daily_acclimation_smith,
    update_daily_lai_root_height,
)
from aoba.crm import CRMState, SPTendencies, step_crm_physics_and_dynamics
from aoba.dust import compute_dust_emission_sub, step_dust_column
from aoba.gravity_wave import compute_nonorographic_gwd
from aoba.icesheet import (
    H_ICE_THRESHOLD_M,
    IceSheetState,
    init_icesheet_state,
    remap_atmosphere_and_crm_vertical,
    step_sia_icesheet_scvt,
)
from aoba.kernels import can_use_elem_metal, dispatch_crm_student_tile_update_metal, mean_2d
from aoba.koppen import classify_koppen_geiger
from aoba.microphysics import diagnose_cloud_effective_radii
from aoba.ocean_ekman import EkmanOceanState, init_ekman_ocean_state, step_ekman_ocean_voronoi
from aoba.ocean_ogcm import OceanOGCMState, init_ogcm_state, step_ogcm_voronoi
from aoba.orography import compute_orographic_drag_lott_miller
from aoba.radiation import compute_four_band_radiation
from aoba.radiation_ckd import CH4_PREIND_PPMV, N2O_PREIND_PPMV, compute_ckd_radiation
from aoba.runtime import maybe_cuda_compile
from aoba.scvt_dynamics import (
    GlobalAtmosState,
    apply_scvt_storm_smagorinsky,
    fct_advect_scvt,
    fct_advect_scvt_batch,
    init_global_hydrostatic_atmosphere,
    step_scvt_rk3_hevi,
)
from aoba.scvt_mesh import (
    SCVTMesh,
    build_scvt_mesh,
    project_cell_vectors_to_edges,
    reconstruct_cell_winds,
)
from aoba.surface import (
    EMISSIVITY_LAND,
    EMISSIVITY_OCEAN,
    SURF_ICESHEET,
    SURF_LAND,
    SURF_OCEAN,
    SWE_GLACIER_MAX,
    SurfaceFluxDiagnostics,
    SurfaceState,
    compute_surface_albedos,
    init_surface_state,
    step_surface,
)
from aoba.thermodynamics import diagnose_pressure_temperature
from aoba.turbulence import TKE_MIN, step_mynn_level25
from aoba.vertical_grid import VerticalGrid, build_vertical_grid


@dataclass
class AOBAConfig:
    """Configuration for a global 1:1 Superparameterized AOBA simulation."""

    glevel: int = 5                          # SCVT mesh level (default 5 -> 10242 cells; 3 -> 642; 4 -> 2562)
    nz: int = 44                             # Shared Charney-Phillips vertical levels (troposphere + full stratosphere)
    z_top: float = 48000.0                   # Model top height [m] (~0.85 hPa stratopause; 24000 m for low-top mode)
    dz_min: float = 60.0                     # Lowest layer thickness [m]
    z_sponge_base: float = 36000.0           # Rayleigh sponge base height [m] (upper stratosphere 36-48 km)
    tau_sponge_min: float = 300.0            # Minimum sponge damping timescale [s]
    terrain_scale_exp: float = 1.5           # Klemp STF attenuation exponent p [-]
    h_ml_ocean: float = 50.0                 # Ocean mixed-layer depth [m]
    ocean_mode: str = "ogcm_3d"              # Ocean model: "ogcm_3d" (3D multi-layer OGCM) or "ekman_sw" (1.5-layer shallow-water)
    ocean_glevel: int = 0                    # Independent ocean SCVT glevel (0 = share host glevel; >0 = e.g. G6/G7/G8 high-res OGCM)
    ocean_nz: int = 18                       # Vertical z-levels when ocean_mode="ogcm_3d" (0..6000 m)
    ocean_stride: int = 1                    # Conservative async 3D OGCM coupling interval [host steps] (1=every step, 2-3=faster)
    enable_seaice_drift: bool = True         # Enable dynamic wind/current sea-ice drift in 3D OGCM mode
    enable_ocean_bgc: bool = True            # FlexPFT marine biogeochemistry & OMIP/mocsy 2.0 carbonate chemistry
    enable_river_routing: bool = True        # Map-free Priority-Flood Voronoi river routing
    enable_land_carbon: bool = True          # Map-free terrestrial carbon cycle, permafrost soil C, & SPITFIRE wildfire
    enable_land_nitrogen: bool = True        # Stocker & Prentice (2024) CN-model & FUN/Houlton BNF nitrogen cycle
    enable_lightning: bool = True            # Map-free CRM non-inductive charging lightning & LNOx deposition
    enable_fire_veg_feedback: bool = True    # MAML burnt-patch crown-fire reset of EEO LAI & canopy height
    enable_seaice_evp_itd: bool = True       # MPAS-Seaice Voronoi EVP rheology + 5-category ITD
    enable_ogcm_fct_zstar: bool = True       # FESOM2-style 3D Zalesak FCT advection + exact z* coordinate
    enable_ogcm_scale_aware_gm: bool = True  # Hallberg (2013) resolution-aware GM eddy scale function
    enable_ogcm_submeso: bool = True         # Fox-Kemper et al. (2008) mixed-layer submesoscale eddy restratification
    enable_ogcm_tidal_mixing: bool = True    # St. Laurent et al. (2002) bathymetric roughness internal tidal mixing
    enable_current_rel_wind: bool = True     # Air-sea fluxes using wind relative to ocean current (active when 3D CRM or SP off)
    enable_natural_aerosols: bool = True     # Map-free sea-salt, marine DMS sulfate, & wildfire BC/OC aerosols
    enable_aerosol_rad_forcing: bool = True  # Direct SW+LW radiative forcing of BC/OC/SO4/sea-salt aerosols
    enable_snow_bc_darkening: bool = True    # SNICAR snow & sea-ice visible albedo darkening by BC & dust deposition
    enable_bgc_fe_lnox_coupling: bool = True # Dust Fe ocean fertilization & LNOx marine NO3 deposition into FlexPFT
    enable_bvoc_soa: bool = True             # EEO forest BVOC (isoprene/monoterpene) -> SOA -> CCN feedback
    enable_prognostic_ch4_n2o: bool = False  # Prognostic natural CH4 (wetland/fire/OH) & N2O (soil/ocean/UV) cycles
    enable_silicate_weathering: bool = False # Carbonate-silicate weathering CO2 thermostat & riverine alkalinity
    enable_co2_3d_transport: bool = False    # 3D SCVT FCT CO2 transport (Keeling Curve) when co2_mode="prognostic"
    co2_anthro_pgc_yr: float = 0.0           # Prescribed anthropogenic CO2 emission rate [Pg C yr^-1] (co2_mode="prognostic")
    enable_ocean_o2_omz: bool = False        # Prognostic 3D dissolved O2, OMZ hypoxia, & marine anaerobic denitrification
    enable_volcanic_aerosols: bool = False   # Stratospheric SO2 -> H2SO4 volcanic/SAI aerosols coupled to radiation & O3
    volcanic_eruption_year: float = -1.0     # Simulation year to trigger discrete eruption pulse (< 0 disables pulse)
    volcanic_so2_tg: float = 0.0             # Discrete eruption SO2 mass [Tg SO2] (e.g. 17.0 for Mt. Pinatubo)
    volcanic_eruption_lat_deg: float = 15.0  # Eruption latitude [deg N]
    volcanic_eruption_width_deg: float = 15.0 # Initial meridional plume Gaussian width [deg]
    sai_so2_tg_yr: float = 0.0               # Continuous stratospheric SO2 injection rate [Tg SO2 yr^-1] (geoengineering)
    sai_lat_band_deg: float = 20.0           # Meridional band width for continuous SAI injection [deg]
    enable_solar_cycle: bool = False         # 11-year Schwabe solar cycle modulating TSI & stratospheric UV photolysis
    solar_cycle_amp_wm2: float = 1.0         # Peak-to-trough TSI solar cycle amplitude [W m^-2] (Lean et al., 2005)
    solar_cycle_period_yr: float = 11.0      # Solar cycle period [yr]
    solar_cycle_phase_yr: float = 0.0        # Solar cycle reference phase year [yr]
    enable_maml_subgrid_topo: bool = False   # MAML subgrid elevation lapse-rate & slope microclimate downscaling
    enable_maml_wetlands_lakes: bool = False # Dynamic MAML inundated wetlands/floodplains/lakes -> CH4 & evaporation
    enable_dynamic_lakes: bool = False       # Map-free dynamic lakes (Earth2014/Priority-Flood basins + 2-layer FLake + ice)
    lake_init_fill_frac: float = 0.50        # Uniform initial topographic basin fill fraction in [0, 1] [-]
    lake_max_frac: float = 0.85              # Maximum lake area fraction of a land Voronoi cell [-]
    lake_max_depth_m: float = 400.0          # Cap on full-spill mean basin depth [m]
    lake_extinction_m1: float = 0.50         # Freshwater Beer-Lambert solar extinction coefficient [m^-1]
    enable_wave_coupling: bool = False       # Prognostic ocean waves H_s/T_p, Stokes Langmuir KPP mixing, & sea-spray flux
    enable_water_isotopes: bool = False      # Prognostic 3D H2^18O & HDO stable water isotopes + ice-core d18O archive
    planet_preset: str = "earth"             # "earth", "aquaplanet", "snowball_earth", "tidally_locked", "trappist1e", "mars_like"
    enable_tidal_lock: bool = False          # Synchronous tidally locked insolation (fixed substellar point)
    substellar_lon_deg: float = 0.0          # Substellar longitude [deg E] when enable_tidal_lock=True
    volcanic_co2_pgc_yr: float = 0.03        # Volcanic CO2 outgassing for silicate-weathering mode [Pg C yr^-1]
    weathering_accel_factor: float = 1.0     # Optional geological acceleration factor for weathering mode [-]
    enable_milankovitch_orbit: bool = False  # Analytical Berger (1978) / Laskar (2004) Milankovitch orbital evolution
    milankovitch_start_kyr: float = 0.0      # Starting orbital epoch relative to J2000 [kyr] (e.g. -21.0 for LGM)
    milankovitch_accel_factor: float = 1.0   # Optional orbital acceleration multiplier for paleoclimate spin-up [-]
    re_liq_um: float = 10.0                  # Fallback/1M cloud liquid droplet effective radius [um]
    re_ice_um: float = 48.0                  # Fallback/1M cloud ice crystal effective radius [um]
    cloud_sw_inhomog: float = 0.62           # Subgrid SW cloud optical inhomogeneity factor chi_SW (Cahalan et al., 1994) [-]
    microphysics_scheme: str = "two_moment"  # "two_moment" (KK00), "six_class_2m" (Morrison r/s/g), or "one_moment"
    crm_nx: int = 8                          # Embedded CRM x-grid points per SCVT cell (default 2D 8x1 = 32 km curtain)
    crm_ny: int = 1                          # Embedded CRM y-grid points (1 = SP-CAM 2.5D curtain for Mac climate runs)
    crm_dx: float = 4000.0                   # Embedded CRM horizontal spacing [m] (4 km resolves convective cores cleanly)
    dt_host: float = 720.0                   # Coupled host / physics / SP time step [s] (12 min climate coupling)
    n_host_dyn_substeps: int = 1             # Minimum host RK3-HEVI substeps per dt_host (auto-increased if CFL requires)
    n_crm_dyn_substeps: int = 12             # Embedded 2D CRM RK3-HEVI substeps per dt_host (12 x 60 s = 720 s at dx=4 km)
    rad_stride: int = 5                      # Radiation evaluation stride (steps; 5 x 720 s = 3600 s = 1 hr)
    crm_col_chunk: int = 16384               # Column batch chunk size for CRM evaluation (Unified Memory single-pass)
    rad_col_chunk: int = 32768               # Subcolumn chunk size for ICA radiative transfer (caps MPS peak buffer)
    co2_ppmv: float = 284.3                  # Well-mixed atmospheric CO2 [ppmv] (default CMIP6 1850 piControl)
    co2_mode: str = "constant"               # "constant", "scenario", or "csv"
    co2_scenario: str = "obs_historical_noaa"  # Scenario key when co2_mode="scenario"
    co2_csv_path: str = ""                   # Custom (year, co2_ppmv) CSV path when co2_mode="csv"
    ch4_ppmv: float = CH4_PREIND_PPMV        # Well-mixed atmospheric CH4 [ppmv] (1850 piControl = 0.80825)
    n2o_ppmv: float = N2O_PREIND_PPMV        # Well-mixed atmospheric N2O [ppmv] (1850 piControl = 0.27302)
    rad_scheme: str = "ckd"                  # "ckd" (16b), "ckd_hd" (29b/111g), "rrtmgp" (HITRAN 240g), or "four_band"
    enable_rad_3d_tica: bool = False         # 3D CRM tilted solar beam cloud shift (Várnai & Davies, 1999 TICA)
    o3_mode: str = "latitude_seasonal"       # "rcemip", "latitude_seasonal", "climatology", or "prognostic" (3D map-free O3)
    o3_file_path: str = ""                   # Optional external zonal/3D ozone (.npz/.nc) when o3_mode="climatology"
    o3_lw_coupling: bool = True              # Enable 9.6 um O3 absorption in LW2 atmospheric window
    o3_ecl_ppbv: float = 0.55                # Equivalent stratospheric Cl [ppbv] for PSC loss when o3_mode="prognostic" (0.55=1850, 3.5=2000)
    enable_nonoro_gwd: bool = True           # Map-free flow-dependent convective + frontal non-orographic GWD
    solar_constant: float = C.SOLAR_CONSTANT # Total solar irradiance S_0 [W m^-2]
    omega: float = C.OMEGA_EARTH             # Planetary rotation rate [rad s^-1]
    obliquity_deg: float = C.OBLIQUITY_DEG   # Axial tilt [deg]
    eccentricity: float = C.ECCENTRICITY     # Orbital eccentricity [-]
    perihelion_deg: float = C.PERIHELION_LON_DEG
    start_date: str = "2000-03-20"           # Calendar start date (YYYY-MM-DD; default Vernal Equinox)
    calendar_type: str = "gregorian"         # "gregorian" (leap years), "noleap" (365d), or "360_day"
    insolation_mode: str = "seasonal"        # "seasonal" or "perpetual_date"
    fixed_insolation_date: str = "2000-03-20"  # Fixed orbital date for perpetual insolation experiments
    diurnal_cycle: bool = True               # True: diurnal solar cycle; False: daily-mean insolation
    use_earth2014: bool = True
    earth2014_dir: str = "data/boundary/earth2014_5min"
    earth2014_stride: int = 2
    enable_superparam: bool = True           # 1:1 embedded 3D CRM in every SCVT cell
    crm_sfc_tiles: int = 0                   # Surface tiles/cell for large CRMs (0 = 1:1 with crm_ny*crm_nx; e.g. 8 for RCE_large)
    enable_crm_self_distill: bool = False    # Online self-distilling / active-learning 2D/3D CRM acceleration
    enable_rad_self_distill: bool = False    # Online per-subcolumn ICA radiation self-distillation paired with CRM
    enable_distill_shadow_diag: bool = True  # Pre-update shadow evaluation on teachers (Host Atmos + Land Tiles)
    crm_distill_stride: int = 13             # Teacher column stride (1 in `stride` columns run genuine CRM/Rad/step)
    crm_distill_warmup_steps: int = 24       # Initial 100%-CRM warmup steps before activating student inference
    crm_distill_dwell_steps: int = 12        # Continuous teacher dwell steps (>=48 enables 50% persistent RCEMIP anchor)
    crm_distill_hidden_dim: int = 192        # Hidden width of pure-GEMM residual column MLP student
    crm_distill_replay_size: int = 65536     # GPU-resident stratified replay buffer capacity [columns]
    enable_dust: bool = True                 # Map-free global-parameter mineral dust aerosol + radiative coupling
    dust_emis_factor: float = 2.5e-6         # Global dust emission tuning factor T (map-free S=1 CLM5 Eq. 31.1; AeroCom AOD ~ 0.03) [-]
    enable_dynamic_icesheet: bool = False    # Dynamic SIA + EMA-accelerated ice sheet & vertical grid update
    icesheet_ema_years: float = 5.0          # EMA smoothing timescale for annual SMB [years]
    icesheet_accel_factor: float = 100.0     # Initial ice-sheet spinup acceleration multiplier [-]
    icesheet_accel_spinup_years: float = 0.0 # Optional e-folding climate timescale for accel decay to 1.0 [yr] (0=const)
    icesheet_isostasy_tau_yr: float = 1000.0 # Lithospheric isostasy relaxation timescale in ice-years [yr] (0=instant)


@dataclass
class ClimateAccumulator:
    """Running daily & 12-month accumulators for EEO phenology and Köppen-Geiger classification."""

    # Daily accumulators reset every 86,400 s
    day_precip_mm: torch.Tensor
    day_pet_mm: torch.Tensor
    day_ppfd_mol: torch.Tensor
    day_temp_sum: torch.Tensor
    day_vpd_sum: torch.Tensor
    day_t_min: torch.Tensor
    day_steps: int = 0

    # 12-month climatology accumulators [12, N_cells]
    month_temp_sum_c: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    month_precip_mm: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    month_steps: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    month_days_accum: torch.Tensor = field(default_factory=lambda: torch.empty(0))


@dataclass
class AOBAState:
    """Complete prognostic state of the coupled AOBA Earth System Model."""

    cfg: AOBAConfig
    mesh: SCVTMesh
    vgrid: VerticalGrid
    atmos: GlobalAtmosState
    crm: CRMState
    sfc: SurfaceState
    ocn: EkmanOceanState
    clim: ClimateAccumulator
    ogcm: OceanOGCMState | None = None
    ocn_mesh: SCVTMesh | None = None
    ocn_sfc: SurfaceState | None = None
    ocn_coupler: object | None = None
    sfc_sub: SurfaceState | None = None
    clim_sub: ClimateAccumulator | None = None
    icesheet: IceSheetState | None = None
    river: object | None = None
    lakes: object | None = None
    land_carbon: object | None = None
    aerosols: object | None = None
    ozone: object | None = None
    trace_gas: object | None = None
    weathering: object | None = None
    prognostic_co2: object | None = None
    volcanic_solar: object | None = None
    maml_land: object | None = None
    ocean_waves: object | None = None
    water_isotopes: object | None = None
    sim_time_sec: float = 0.0
    step_count: int = 0
    rad_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    calendar: AobaCalendar = field(default_factory=AobaCalendar)
    co2_series: CO2ForcingSeries | None = None
    o3_provider: OzoneForcingProvider | None = None
    q_conv_diag: torch.Tensor | None = None  # Previous-step 1:1 CRM convective heating [Nz, N_cells] [K s^-1]
    last_sfc_diag_sub: SurfaceFluxDiagnostics | None = None  # Zero-copy ref for optional CMIP7 diagnostics
    last_s_toa: torch.Tensor | None = None                   # Zero-copy ref for TOA incident SW (`rsdt`)
    last_lightning: object | None = None                     # Latest CRM LightningDiagnostics (CG strikes + LNOx)
    crm_distill: object | None = None                        # Optional online self-distilling 2D/3D CRM controller
    rad_distill: object | None = None                        # Optional online per-subcolumn ICA radiation controller
    sparse_crm_pool: object | None = None                    # Teacher-only full-3D CRM pool (when n_sub < n_crm in 3D)
    ocn_flux_accum: dict[str, object] = field(default_factory=dict)  # Conservative flux accumulator for ocean_stride > 1

    def current_co2_ppmv(self, sim_time_sec: float | None = None) -> float:
        """Return active atmospheric CO2 [ppmv] at ``sim_time_sec`` (or ``self.sim_time_sec``)."""
        if str(self.cfg.co2_mode).strip().lower() == "prognostic" and self.prognostic_co2 is not None:
            return float(self.prognostic_co2.co2_ppmv_glob)
        if bool(self.cfg.enable_silicate_weathering) and self.weathering is not None:
            return float(self.weathering.co2_ppmv)
        if self.co2_series is None or not self.co2_series.is_time_varying:
            return float(self.cfg.co2_ppmv)
        t_sec = self.sim_time_sec if sim_time_sec is None else float(sim_time_sec)
        return self.co2_series.evaluate(self.calendar.decimal_year_at(t_sec))

    def current_ch4_ppmv(self) -> float:
        """Return active atmospheric CH4 [ppmv] (prognostic if enabled, else configured constant)."""
        if bool(self.cfg.enable_prognostic_ch4_n2o) and self.trace_gas is not None:
            return float(self.trace_gas.ch4_ppmv)
        return float(self.cfg.ch4_ppmv)

    def current_n2o_ppmv(self) -> float:
        """Return active atmospheric N2O [ppmv] (prognostic if enabled, else configured constant)."""
        if bool(self.cfg.enable_prognostic_ch4_n2o) and self.trace_gas is not None:
            return float(self.trace_gas.n2o_ppmv)
        return float(self.cfg.n2o_ppmv)


def resolve_crm_sfc_tile_grid(ny: int, nx: int, req: int) -> tuple[int, int]:
    """Factorize `(ny, nx)` into an isotropic 2D surface-tile grid `(ny_t, nx_t)` with `ny_t|ny, nx_t|nx`.

    Picks the exact divisor pair with `ny_t * nx_t == req` (or largest exact divisor `<= req` when
    `req` does not divide `ny * nx`) that minimizes physical tile aspect-ratio distortion
    `|log((ny / ny_t) / (nx / nx_t))|`. Examples:
    - `(16, 64), req=16` -> `(2, 8)` (`8x8` CRM blocks = exact square `32x32 km` tiles at 4 km)
    - `(16, 16), req=16` -> `(4, 4)` (`4x4` CRM blocks)
    - `(64, 64), req=16` -> `(4, 4)` (`16x16` CRM blocks)
    - `(12, 36), req=12` -> `(2, 6)` (`6x6` CRM blocks)
    - `(1, 64),  req=8`  -> `(1, 8)` (1D exact compatibility for 2D `RCE_large`)
    """
    ny_i = max(int(ny), 1)
    nx_i = max(int(nx), 1)
    n_crm = ny_i * nx_i
    req_i = int(req)
    if req_i <= 0 or req_i >= n_crm:
        return ny_i, nx_i
    div_y = [d for d in range(1, ny_i + 1) if ny_i % d == 0]
    div_x = [d for d in range(1, nx_i + 1) if nx_i % d == 0]
    best_pair = (1, 1)
    best_key: tuple[int, float, float] | None = None
    target_y = ny_i * math.sqrt(float(req_i) / float(n_crm))
    for dy_t in div_y:
        for dx_t in div_x:
            prod = dy_t * dx_t
            if prod > req_i:
                continue
            by = ny_i / float(dy_t)
            bx = nx_i / float(dx_t)
            aspect_err = abs(math.log(by / bx))
            key = (-prod, aspect_err, abs(float(dy_t) - target_y))
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (dy_t, dx_t)
    return best_pair


def resolve_crm_sfc_tiles(cfg: AOBAConfig) -> int:
    """Return surface-tile count `n_sfc = ny_t * nx_t` per host cell (`<= crm_ny * crm_nx`)."""
    ny_t, nx_t = resolve_crm_sfc_tile_grid(cfg.crm_ny, cfg.crm_nx, getattr(cfg, "crm_sfc_tiles", 0))
    return ny_t * nx_t


def use_sparse_crm_pool(cfg: AOBAConfig, nc: int) -> bool:
    """Return True when full-resolution CRM state should live in a teacher-only `SparseCRMPool`.

    Activates whenever superparameterization + CRM self-distillation are enabled with coarser
    surface tiles (`ny_t * nx_t < crm_ny * crm_nx`) and either a 3D CRM (`crm_ny > 1 and crm_nx > 1`)
    or a very large global CRM (`nc * crm_ny * crm_nx > 1_000_000`). Keeps 2D `1x8` and `1x64`
    dense paths unchanged.
    """
    if not (bool(cfg.enable_superparam) and bool(cfg.enable_crm_self_distill)):
        return False
    ny, nx = max(int(cfg.crm_ny), 1), max(int(cfg.crm_nx), 1)
    n_crm = ny * nx
    ny_t, nx_t = resolve_crm_sfc_tile_grid(ny, nx, getattr(cfg, "crm_sfc_tiles", 0))
    if ny_t * nx_t >= n_crm:
        return False
    return (ny > 1 and nx > 1) or (int(nc) * n_crm > 1_000_000)


def _crm_to_sfc_tile(
    x: torch.Tensor,
    nc: int,
    n_sfc: int,
    n_crm: int,
    *,
    ny: int = 1,
    nx: int = 0,
    ny_t: int = 1,
    nx_t: int = 0,
) -> torch.Tensor:
    """2D Kronecker block-average `[..., nc * n_crm]` (or `[..., nc, ny, nx]`) onto `[..., nc * n_sfc]`."""
    if n_sfc == n_crm:
        if x.ndim >= 1 and x.shape[-1] == nc * n_crm:
            return x
        return x.reshape(*x.shape[:-3], nc * n_sfc)
    nx_eff = nx if nx > 0 else (n_crm // max(ny, 1))
    nxt_eff = nx_t if nx_t > 0 else (n_sfc // max(ny_t, 1))
    by = ny // max(ny_t, 1)
    bx = nx_eff // max(nxt_eff, 1)
    if x.ndim >= 1 and x.shape[-1] == nc * n_crm:
        lead = x.shape[:-1]
    else:
        lead = x.shape[:-3]
    return (
        x.reshape(*lead, nc, ny_t, by, nxt_eff, bx)
        .mean(dim=(-3, -1))
        .reshape(*lead, nc * n_sfc)
    )


def _sfc_tile_to_crm(
    x: torch.Tensor,
    nc: int,
    n_sfc: int,
    n_crm: int,
    *,
    ny: int = 1,
    nx: int = 0,
    ny_t: int = 1,
    nx_t: int = 0,
) -> torch.Tensor:
    """2D Kronecker tile-expand `[..., nc * n_sfc]` onto `[..., nc * n_crm]`."""
    if n_sfc == n_crm:
        return x
    nx_eff = nx if nx > 0 else (n_crm // max(ny, 1))
    nxt_eff = nx_t if nx_t > 0 else (n_sfc // max(ny_t, 1))
    by = ny // max(ny_t, 1)
    bx = nx_eff // max(nxt_eff, 1)
    lead = x.shape[:-1]
    return (
        x.reshape(*lead, nc, ny_t, 1, nxt_eff, 1)
        .expand(*lead, nc, ny_t, by, nxt_eff, bx)
        .reshape(*lead, nc * n_crm)
    )


def _cell_to_sub(x: torch.Tensor, nc: int, n_sub: int) -> torch.Tensor:
    """Expand host `[..., nc]` tensor to flattened subcolumns `[..., nc * n_sub]`."""
    return x.unsqueeze(-1).expand(*x.shape, n_sub).reshape(*x.shape[:-1], nc * n_sub)


def _sub_to_cell(x: torch.Tensor, nc: int, n_sub: int) -> torch.Tensor:
    """Average flattened subcolumns `[..., nc * n_sub]` onto host cells `[..., nc]`."""
    return x.reshape(*x.shape[:-1], nc, n_sub).mean(dim=-1)


def _sync_host_surface_from_subgrid(
    sfc_host: SurfaceState,
    sfc_sub: SurfaceState,
    nc: int,
    n_sub: int,
) -> None:
    """Mirror subcolumn MAML means (`sfc_sub`) back to host `[nc]` summary state (`sfc_host`)."""
    for attr in sfc_host.__dataclass_fields__:
        if attr == "bio":
            continue
        val = getattr(sfc_sub, attr)
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            setattr(sfc_host, attr, _sub_to_cell(val, nc, n_sub))
    for attr in sfc_host.bio.__dataclass_fields__:
        val = getattr(sfc_sub.bio, attr)
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            setattr(sfc_host.bio, attr, _sub_to_cell(val, nc, n_sub))


def broadcast_host_surface_to_subgrid(model: AOBAState) -> None:
    """Broadcast host `[nc]` `model.sfc` & `bio` across all surface tiles in `model.sfc_sub`."""
    if model.sfc_sub is None:
        return
    nc = model.mesh.n_cells
    n_sub = resolve_crm_sfc_tiles(model.cfg)
    for attr in model.sfc.__dataclass_fields__:
        if attr == "bio":
            continue
        val = getattr(model.sfc, attr)
        if isinstance(val, torch.Tensor):
            setattr(model.sfc_sub, attr, _cell_to_sub(val, nc, n_sub).clone())
    for attr in model.sfc.bio.__dataclass_fields__:
        val = getattr(model.sfc.bio, attr)
        if isinstance(val, torch.Tensor):
            setattr(model.sfc_sub.bio, attr, _cell_to_sub(val, nc, n_sub).clone())


def _compute_four_band_radiation_chunked(
    pres_c: torch.Tensor,
    pres_w: torch.Tensor,
    temp_c: torch.Tensor,
    temp_w: torch.Tensor,
    dz_c: torch.Tensor,
    rho_d_c: torch.Tensor,
    qv_c: torch.Tensor,
    qc_c: torch.Tensor,
    qi_c: torch.Tensor,
    t_sfc: torch.Tensor,
    albedo_vis: torch.Tensor,
    albedo_nir: torch.Tensor,
    mu0: torch.Tensor,
    s_toa: torch.Tensor,
    co2_ppmv: float,
    emiss_sfc: torch.Tensor,
    albedo_vis_dif: torch.Tensor,
    albedo_nir_dif: torch.Tensor,
    col_chunk: int,
    re_liq_um: float | torch.Tensor = 10.0,
    re_ice_um: float | torch.Tensor = 48.0,
    cloud_sw_inhomog: float = 0.85,
    q_dust_c: torch.Tensor | None = None,
    o3_vmr_c: torch.Tensor | None = None,
    o3_lw_coupling: bool = False,
    rad_scheme: str = "ckd",
    ch4_ppmv: float = CH4_PREIND_PPMV,
    n2o_ppmv: float = N2O_PREIND_PPMV,
    aero_opt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate 1D ICA radiative transfer across `[nz, m_sub]`, chunking when `m_sub > col_chunk`."""
    nz, m_sub = pres_c.shape
    chunk = max(col_chunk, 1)
    rkey = str(rad_scheme).strip().lower()
    use_rrtmgp = rkey == "rrtmgp"
    use_ckd_hd = rkey == "ckd_hd"
    use_ckd = rkey == "ckd"

    def _eval_slice(sl: slice | None):
        if sl is None:
            pc, pw, tc, tw, dz, rh = pres_c, pres_w, temp_c, temp_w, dz_c, rho_d_c
            qv, qc, qi, ts = qv_c, qc_c, qi_c, t_sfc
            av, an, m0, st, em = albedo_vis, albedo_nir, mu0, s_toa, emiss_sfc
            avf, anf = albedo_vis_dif, albedo_nir_dif
            qd, o3 = q_dust_c, o3_vmr_c
            rel = re_liq_um
            rei = re_ice_um
            ao = aero_opt
        else:
            pc, pw, tc, tw, dz, rh = pres_c[:, sl], pres_w[:, sl], temp_c[:, sl], temp_w[:, sl], dz_c[:, sl], rho_d_c[:, sl]
            qv, qc, qi, ts = qv_c[:, sl], qc_c[:, sl], qi_c[:, sl], t_sfc[sl]
            av, an, m0, st, em = albedo_vis[sl], albedo_nir[sl], mu0[sl], s_toa[sl], emiss_sfc[sl]
            avf, anf = albedo_vis_dif[sl], albedo_nir_dif[sl]
            qd = q_dust_c[:, sl] if q_dust_c is not None else None
            o3 = o3_vmr_c[:, sl] if o3_vmr_c is not None else None
            rel = re_liq_um[:, sl] if isinstance(re_liq_um, torch.Tensor) else re_liq_um
            rei = re_ice_um[:, sl] if isinstance(re_ice_um, torch.Tensor) else re_ice_um
            ao = tuple(t[:, sl] for t in aero_opt) if aero_opt is not None else None

        if use_rrtmgp:
            from aoba.radiation_rrtmgp import compute_rrtmgp_radiation

            return compute_rrtmgp_radiation(
                pres_c=pc, pres_w=pw, temp_c=tc, temp_w=tw, dz_c=dz, rho_d_c=rh,
                qv_c=qv, qc_c=qc, qi_c=qi, t_sfc=ts, albedo_vis=av, albedo_nir=an,
                mu0=m0, s_toa=st, co2_ppmv=co2_ppmv, ch4_ppmv=ch4_ppmv, n2o_ppmv=n2o_ppmv,
                emiss_sfc=em, re_liq_um=rel, re_ice_um=rei, cloud_sw_inhomog=cloud_sw_inhomog,
                albedo_vis_dif=avf, albedo_nir_dif=anf, q_dust_c=qd, o3_vmr_c=o3,
                o3_lw_coupling=o3_lw_coupling, aero_opt=ao,
            )
        if use_ckd_hd:
            from aoba.radiation_hd import compute_ckd_hd_radiation

            return compute_ckd_hd_radiation(
                pres_c=pc, pres_w=pw, temp_c=tc, temp_w=tw, dz_c=dz, rho_d_c=rh,
                qv_c=qv, qc_c=qc, qi_c=qi, t_sfc=ts, albedo_vis=av, albedo_nir=an,
                mu0=m0, s_toa=st, co2_ppmv=co2_ppmv, ch4_ppmv=ch4_ppmv, n2o_ppmv=n2o_ppmv,
                emiss_sfc=em, re_liq_um=rel, re_ice_um=rei, cloud_sw_inhomog=cloud_sw_inhomog,
                albedo_vis_dif=avf, albedo_nir_dif=anf, q_dust_c=qd, o3_vmr_c=o3,
                o3_lw_coupling=o3_lw_coupling, aero_opt=ao,
            )
        if use_ckd:
            base = compute_ckd_radiation(
                pres_c=pc, pres_w=pw, temp_c=tc, temp_w=tw, dz_c=dz, rho_d_c=rh,
                qv_c=qv, qc_c=qc, qi_c=qi, t_sfc=ts, albedo_vis=av, albedo_nir=an,
                mu0=m0, s_toa=st, co2_ppmv=co2_ppmv, ch4_ppmv=ch4_ppmv, n2o_ppmv=n2o_ppmv,
                emiss_sfc=em, re_liq_um=rel, re_ice_um=rei, cloud_sw_inhomog=cloud_sw_inhomog,
                albedo_vis_dif=avf, albedo_nir_dif=anf, q_dust_c=qd, o3_vmr_c=o3,
                o3_lw_coupling=o3_lw_coupling,
            )
        else:
            rel_f = float(rel.mean().item()) if isinstance(rel, torch.Tensor) else float(rel)
            rei_f = float(rei.mean().item()) if isinstance(rei, torch.Tensor) else float(rei)
            base = compute_four_band_radiation(
                pres_c=pc, pres_w=pw, temp_c=tc, temp_w=tw, dz_c=dz, rho_d_c=rh,
                qv_c=qv, qc_c=qc, qi_c=qi, t_sfc=ts, albedo_vis=av, albedo_nir=an,
                mu0=m0, s_toa=st, co2_ppmv=co2_ppmv, emiss_sfc=em,
                re_liq_um=rel_f, re_ice_um=rei_f, cloud_sw_inhomog=cloud_sw_inhomog,
                albedo_vis_dif=avf, albedo_nir_dif=anf, q_dust_c=qd, o3_vmr_c=o3,
                o3_lw_coupling=o3_lw_coupling,
            )
        if ao is not None:
            from aoba.radiation_hd import apply_aerosol_rad_delta

            return apply_aerosol_rad_delta(
                base, pc, tc, tw, dz, rh, ts, av, avf, m0, st, ao, emiss_sfc=em
            )
        return base

    if chunk >= m_sub:
        rad = _eval_slice(None)
        return (
            rad.q_rad_net,
            rad.sfc_sw1_dir_dn,
            rad.sfc_sw1_dif_dn,
            rad.sfc_sw2_dir_dn,
            rad.sfc_sw2_dif_dn,
            rad.lw_flux_dn[0],
            rad.toa_olr,
            rad.toa_asr,
        )

    dev, dt = pres_c.device, pres_c.dtype
    q_rad = torch.empty((nz, m_sub), device=dev, dtype=dt)
    sw1_dir = torch.empty(m_sub, device=dev, dtype=dt)
    sw1_dif = torch.empty(m_sub, device=dev, dtype=dt)
    sw2_dir = torch.empty(m_sub, device=dev, dtype=dt)
    sw2_dif = torch.empty(m_sub, device=dev, dtype=dt)
    lw_dn = torch.empty(m_sub, device=dev, dtype=dt)
    olr = torch.empty(m_sub, device=dev, dtype=dt)
    asr = torch.empty(m_sub, device=dev, dtype=dt)

    for i0 in range(0, m_sub, chunk):
        i1 = min(i0 + chunk, m_sub)
        sl = slice(i0, i1)
        rad = _eval_slice(sl)
        q_rad[:, sl] = rad.q_rad_net
        sw1_dir[sl] = rad.sfc_sw1_dir_dn
        sw1_dif[sl] = rad.sfc_sw1_dif_dn
        sw2_dir[sl] = rad.sfc_sw2_dir_dn
        sw2_dif[sl] = rad.sfc_sw2_dif_dn
        lw_dn[sl] = rad.lw_flux_dn[0]
        olr[sl] = rad.toa_olr
        asr[sl] = rad.toa_asr

    return q_rad, sw1_dir, sw1_dif, sw2_dir, sw2_dif, lw_dn, olr, asr


# ---------------------------------------------------------------------------
# 1. Orbital Solar Geometry (Berger, 1978; Davis et al., 2017, Eqs. 2-9)
# ---------------------------------------------------------------------------
def compute_orbital_insolation(
    t_sec: float,
    lat_rad: torch.Tensor,
    lon_rad: torch.Tensor,
    solar_constant: float = C.SOLAR_CONSTANT,
    obliquity_deg: float = C.OBLIQUITY_DEG,
    eccentricity: float = C.ECCENTRICITY,
    perihelion_deg: float = C.PERIHELION_LON_DEG,
    t_diurnal_sec: float | None = None,
    diurnal_cycle: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute instantaneous `(cos_zenith, s_toa_eff, hour_angle_rad)` from orbital elements.

    `t_sec = 0` corresponds to Vernal Equinox (`lambda_s = 0`) at `00:00 UTC`.
    """
    yr_sec = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    frac_yr = (t_sec % yr_sec) / yr_sec
    eps_rad = math.radians(obliquity_deg)
    omega_rad = math.radians(perihelion_deg)
    e = eccentricity

    # Mean anomaly M relative to perihelion (Vernal Equinox lambda = 0 -> true anomaly nu_0 = -omega)
    nu0 = -omega_rad
    e_anom0 = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(0.5 * nu0))
    m0 = e_anom0 - e * math.sin(e_anom0)
    m_anom = m0 + 2.0 * math.pi * frac_yr

    # 3-term equation of center for true anomaly nu (Berger, 1978)
    nu = (
        m_anom
        + (2.0 * e - 0.25 * (e ** 3)) * math.sin(m_anom)
        + 1.25 * (e * e) * math.sin(2.0 * m_anom)
        + (13.0 / 12.0) * (e ** 3) * math.sin(3.0 * m_anom)
    )
    lam_solar = nu + omega_rad
    dist_fac = ((1.0 + e * math.cos(nu)) / (1.0 - e * e)) ** 2
    sin_dec = math.sin(eps_rad) * math.sin(lam_solar)
    cos_dec = math.sqrt(max(1.0 - sin_dec * sin_dec, 0.0))

    # Local solar hour angle h in [-pi, pi]: h == 0 at local solar noon!
    t_diur = t_sec if t_diurnal_sec is None else t_diurnal_sec
    frac_day = (t_diur % C.SECONDS_PER_DAY) / C.SECONDS_PER_DAY
    h_raw = 2.0 * math.pi * frac_day + lon_rad - math.pi
    hour_angle = torch.atan2(torch.sin(h_raw), torch.cos(h_raw))

    if diurnal_cycle:
        mu0_raw = torch.sin(lat_rad) * sin_dec + torch.cos(lat_rad) * cos_dec * torch.cos(hour_angle)
        mu0 = torch.clamp(mu0_raw, min=0.0)
    else:
        tan_lat = torch.tan(torch.clamp(lat_rad, -1.55, 1.55))
        tan_dec = sin_dec / max(cos_dec, 1.0e-6)
        h0 = torch.acos(torch.clamp(-tan_lat * tan_dec, -1.0, 1.0))
        mu0_mean = (h0 * torch.sin(lat_rad) * sin_dec + torch.cos(lat_rad) * cos_dec * torch.sin(h0)) / math.pi
        mu0 = torch.clamp(mu0_mean, min=0.0)
    s_toa = torch.full_like(mu0, solar_constant * dist_fac)
    return mu0, s_toa, hour_angle


# ---------------------------------------------------------------------------
# 2. Model Initialization
# ---------------------------------------------------------------------------
def init_aoba_model(
    cfg: AOBAConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AOBAState:
    """Initialize global SCVT mesh, Earth2014 boundaries, atmosphere, 1:1 CRM, land, & Ekman ocean."""
    if cfg is None:
        cfg = AOBAConfig()
    if str(cfg.planet_preset).strip().lower() != "earth":
        from aoba.planetary_presets import resolve_planet_preset_config

        cfg = resolve_planet_preset_config(cfg)
    dev = torch.device(device)

    mesh = build_scvt_mesh(
        glevel=cfg.glevel,
        n_lloyd_iter=10,
        use_earth2014=cfg.use_earth2014,
        earth2014_dir=cfg.earth2014_dir,
        earth2014_stride=cfg.earth2014_stride,
        omega=cfg.omega,
        device=dev,
        dtype=dtype,
    )
    vgrid = build_vertical_grid(
        nz=cfg.nz,
        z_top=cfg.z_top,
        dz_min=cfg.dz_min,
        z_sponge_base=cfg.z_sponge_base,
        tau_sponge_min=cfg.tau_sponge_min,
        terrain_scale_exp=cfg.terrain_scale_exp,
        device=dev,
        dtype=dtype,
    )
    atmos = init_global_hydrostatic_atmosphere(mesh, vgrid, moist=True, baroclinic_pert=True)
    if cfg.enable_dust:
        atmos.q_dust = torch.zeros_like(atmos.q_n)

    nz = cfg.nz
    nc = mesh.n_cells
    ny_full, nx_full = cfg.crm_ny, cfg.crm_nx
    n_crm_full = ny_full * nx_full
    ny_t, nx_t = resolve_crm_sfc_tile_grid(ny_full, nx_full, getattr(cfg, "crm_sfc_tiles", 0))
    n_sub = ny_t * nx_t
    m_sub = nc * n_sub
    use_pool = use_sparse_crm_pool(cfg, nc)
    ny, nx = (ny_t, nx_t) if use_pool else (ny_full, nx_full)
    n_crm = ny * nx
    m_crm = nc * n_crm
    shape4 = (nz, nc, ny, nx)

    # Initialize host-mean surface summary `[nc]` & subgrid MAML surface `[nc * ny * nx]`
    t_sfc_lat = 301.0 - 40.0 * (torch.sin(mesh.lat_cell) ** 2) - 0.0065 * torch.clamp(mesh.z_topo, min=0.0)
    sfc = init_surface_state(
        mesh.surf_type, z_topo=mesh.z_topo, t_init=293.15, sigma_topo=mesh.sigma_topo, h_ml_ocean=cfg.h_ml_ocean,
    )
    t_ocn_raw_h = 301.0 - 34.0 * (torch.sin(mesh.lat_cell) ** 2)
    sfc.t_ocean_ml = torch.where(
        mesh.surf_type == SURF_OCEAN,
        torch.clamp(t_ocn_raw_h, min=C.T_FREEZE_OCEAN + 0.2),
        sfc.t_ocean_ml,
    )
    sfc.t_canopy = t_sfc_lat.clone()
    sfc.t_soil = t_sfc_lat.unsqueeze(0).expand_as(sfc.t_soil).clone()
    polar_cold = (mesh.surf_type == SURF_OCEAN) & (t_ocn_raw_h < C.T_FREEZE_OCEAN)
    sfc.m_seaice = torch.where(
        polar_cold,
        torch.clamp((C.T_FREEZE_OCEAN - t_ocn_raw_h) * 450.0, min=200.0, max=1400.0),
        torch.zeros_like(t_ocn_raw_h),
    )
    if str(cfg.planet_preset).strip().lower() != "earth":
        from aoba.planetary_presets import apply_planet_initial_surface

        apply_planet_initial_surface(sfc, cfg.planet_preset)
        mesh.surf_type = sfc.surf_type.clone()
        mesh.ocean_frac = (sfc.surf_type == SURF_OCEAN).to(mesh.ocean_frac.dtype)
        mesh.z_topo = sfc.z_topo.clone()
        mesh.sigma_topo = sfc.sigma_topo.clone()

    ocn = init_ekman_ocean_state(sfc)
    ocn_mesh: SCVTMesh | None = None
    ocn_sfc: SurfaceState | None = None
    ocn_coupler: object | None = None
    use_indep_ocn = (
        str(cfg.ocean_mode).strip().lower() == "ogcm_3d"
        and int(cfg.ocean_glevel) > 0
        and int(cfg.ocean_glevel) != int(cfg.glevel)
    )
    if use_indep_ocn:
        from aoba.ocean_coupler import build_ocean_atmos_coupler, ocn_to_atm_mean

        ocn_mesh = build_scvt_mesh(
            glevel=int(cfg.ocean_glevel),
            n_lloyd_iter=10,
            use_earth2014=cfg.use_earth2014,
            earth2014_dir=cfg.earth2014_dir,
            earth2014_stride=cfg.earth2014_stride,
            omega=cfg.omega,
            device=dev,
            dtype=dtype,
        )
        ocn_coupler = build_ocean_atmos_coupler(mesh, ocn_mesh, device=dev)
        t_ocn_lat = 301.0 - 40.0 * (torch.sin(ocn_mesh.lat_cell) ** 2) - 0.0065 * torch.clamp(ocn_mesh.z_topo, min=0.0)
        ocn_sfc = init_surface_state(
            ocn_mesh.surf_type, z_topo=ocn_mesh.z_topo, t_init=293.15,
            sigma_topo=ocn_mesh.sigma_topo, h_ml_ocean=cfg.h_ml_ocean,
        )
        t_ocn_raw_f = 301.0 - 34.0 * (torch.sin(ocn_mesh.lat_cell) ** 2)
        ocn_sfc.t_ocean_ml = torch.where(
            ocn_mesh.surf_type == SURF_OCEAN,
            torch.clamp(t_ocn_raw_f, min=C.T_FREEZE_OCEAN + 0.2),
            ocn_sfc.t_ocean_ml,
        )
        polar_ocn = (ocn_mesh.surf_type == SURF_OCEAN) & (t_ocn_raw_f < C.T_FREEZE_OCEAN)
        ocn_sfc.m_seaice = torch.where(
            polar_ocn,
            torch.clamp((C.T_FREEZE_OCEAN - t_ocn_raw_f) * 450.0, min=200.0, max=1400.0),
            torch.zeros_like(t_ocn_raw_f),
        )
        ogcm = init_ogcm_state(ocn_sfc, ocn_mesh, nz_ocn=cfg.ocean_nz)
        if bool(cfg.enable_ocean_bgc):
            from aoba.ocean_bgc import init_ocean_bgc_state

            ogcm.bgc = init_ocean_bgc_state(ogcm.wet_c, ogcm.h_c, ogcm.z_c, ocn_mesh.area_cell)
        if bool(cfg.enable_seaice_evp_itd):
            from aoba.seaice_evp import init_seaice_evp_state

            ogcm.seaice_evp = init_seaice_evp_state(ocn_sfc, ocn_mesh)
        # Synchronize initial host SST & sea ice with exact area-weighted fine-ocean mean
        sfc.t_ocean_ml = ocn_to_atm_mean(ocn_coupler, ocn_sfc.t_ocean_ml, fallback_atm=sfc.t_ocean_ml)
        sfc.m_seaice = ocn_to_atm_mean(ocn_coupler, ocn_sfc.m_seaice, fallback_atm=sfc.m_seaice)
        sfc.swe_seaice = ocn_to_atm_mean(ocn_coupler, ocn_sfc.swe_seaice, fallback_atm=sfc.swe_seaice)
        sfc.h_ocean_ml = ocn_to_atm_mean(ocn_coupler, ocn_sfc.h_ocean_ml, fallback_atm=sfc.h_ocean_ml)
    else:
        ogcm = (
            init_ogcm_state(sfc, mesh, nz_ocn=cfg.ocean_nz)
            if str(cfg.ocean_mode).strip().lower() == "ogcm_3d"
            else None
        )
        if ogcm is not None and bool(cfg.enable_ocean_bgc):
            from aoba.ocean_bgc import init_ocean_bgc_state

            ogcm.bgc = init_ocean_bgc_state(ogcm.wet_c, ogcm.h_c, ogcm.z_c, mesh.area_cell)
        if ogcm is not None and bool(cfg.enable_seaice_evp_itd):
            from aoba.seaice_evp import init_seaice_evp_state

            ogcm.seaice_evp = init_seaice_evp_state(sfc, mesh)

    sfc_sub = init_surface_state(
        _cell_to_sub(mesh.surf_type, nc, n_sub),
        z_topo=_cell_to_sub(mesh.z_topo, nc, n_sub),
        sigma_topo=_cell_to_sub(mesh.sigma_topo, nc, n_sub),
        t_init=293.15,
        h_ml_ocean=cfg.h_ml_ocean,
    )

    # Initialize 1:1 embedded 3D CRM state `[Nz, N_cells, Ny, Nx]` synchronized to host columns
    u_init, v_init = reconstruct_cell_winds(mesh, atmos.u_edge)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    noise = (torch.rand((3, nc, ny, nx), generator=gen, dtype=torch.float32) - 0.5) * 0.10
    noise = (noise - mean_2d(noise, keepdim=True)).to(device=dev, dtype=dtype)

    thm_crm = atmos.theta_m.view(nz, nc, 1, 1).expand(shape4).clone()
    thm_crm[:3] = thm_crm[:3] + noise

    crm = CRMState(
        rho_d=atmos.rho_d.view(nz, nc, 1, 1).expand(shape4).clone(),
        u=u_init.view(nz, nc, 1, 1).expand(shape4).clone(),
        v=v_init.view(nz, nc, 1, 1).expand(shape4).clone(),
        w=torch.zeros((nz + 1, nc, ny, nx), device=dev, dtype=dtype),
        theta_m=thm_crm,
        q_n=atmos.q_n.view(nz, nc, 1, 1).expand(shape4).clone(),
        q_p=torch.zeros(shape4, device=dev, dtype=dtype),
        q_v=atmos.q_v.view(nz, nc, 1, 1).expand(shape4).clone(),
        q_c=torch.zeros(shape4, device=dev, dtype=dtype),
        q_i=torch.zeros(shape4, device=dev, dtype=dtype),
        qke=torch.full(shape4, 2.0 * TKE_MIN, device=dev, dtype=dtype),
        dx=float(cfg.crm_dx) * float(nx_full // nx) if use_pool else float(cfg.crm_dx),
        dy=float(cfg.crm_dx) * float(ny_full // ny) if use_pool else float(cfg.crm_dx),
        f_coriolis=mesh.f_cell.clone(),
    )

    def _make_clim(n_pts: int, include_monthly: bool) -> ClimateAccumulator:
        z = torch.zeros(n_pts, device=dev, dtype=dtype)
        if include_monthly:
            return ClimateAccumulator(
                day_precip_mm=z.clone(),
                day_pet_mm=z.clone(),
                day_ppfd_mol=z.clone(),
                day_temp_sum=z.clone(),
                day_vpd_sum=z.clone(),
                day_t_min=torch.full_like(z, 400.0),
                day_steps=0,
                month_temp_sum_c=torch.zeros((12, n_pts), device=dev, dtype=dtype),
                month_precip_mm=torch.zeros((12, n_pts), device=dev, dtype=dtype),
                month_steps=torch.zeros((12, 1), device=dev, dtype=dtype),
                month_days_accum=torch.zeros((12, 1), device=dev, dtype=dtype),
            )
        return ClimateAccumulator(
            day_precip_mm=z.clone(),
            day_pet_mm=z.clone(),
            day_ppfd_mol=z.clone(),
            day_temp_sum=z.clone(),
            day_vpd_sum=z.clone(),
            day_t_min=torch.full_like(z, 400.0),
            day_steps=0,
        )

    ozone_state = None
    if str(cfg.o3_mode).strip().lower() == "prognostic":
        from aoba.ozone_chemistry import init_prognostic_ozone_state

        pres_init_oz, _, _ = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
        ozone_state = init_prognostic_ozone_state(pres_init_oz, atmos.rho_d, atmos.dz_c)

    model = AOBAState(
        cfg=cfg,
        mesh=mesh,
        vgrid=vgrid,
        atmos=atmos,
        crm=crm,
        sfc=sfc,
        ocn=ocn,
        ogcm=ogcm,
        ocn_mesh=ocn_mesh,
        ocn_sfc=ocn_sfc,
        ocn_coupler=ocn_coupler,
        clim=_make_clim(nc, include_monthly=True),
        sfc_sub=sfc_sub,
        clim_sub=_make_clim(m_sub, include_monthly=False),
        icesheet=init_icesheet_state(mesh) if cfg.enable_dynamic_icesheet else None,
        river=(
            __import__("aoba.river_routing", fromlist=["init_river_routing_state"]).init_river_routing_state(mesh, device=dev)
            if bool(cfg.enable_river_routing)
            else None
        ),
        lakes=(
            __import__("aoba.lakes", fromlist=["init_dynamic_lake_state"]).init_dynamic_lake_state(
                mesh,
                init_fill_frac=float(cfg.lake_init_fill_frac),
                max_frac=float(cfg.lake_max_frac),
                max_depth_m=float(cfg.lake_max_depth_m),
                device=dev,
                dtype=atmos.q_n.dtype,
            )
            if bool(cfg.enable_dynamic_lakes)
            else None
        ),
        land_carbon=None,
        aerosols=(
            __import__("aoba.aerosols", fromlist=["init_natural_aerosol_state"]).init_natural_aerosol_state(
                nz, nc, device=dev, dtype=atmos.q_n.dtype
            )
            if bool(cfg.enable_natural_aerosols)
            else None
        ),
        ozone=ozone_state,
        calendar=AobaCalendar(
            start_date=cfg.start_date,
            calendar_type=cfg.calendar_type,
            insolation_mode=cfg.insolation_mode,
            fixed_insolation_date=cfg.fixed_insolation_date,
            diurnal_cycle=cfg.diurnal_cycle,
            seconds_per_day=C.SECONDS_PER_DAY,
        ),
        co2_series=resolve_co2_series(
            co2_mode=cfg.co2_mode,
            co2_ppmv=cfg.co2_ppmv,
            co2_scenario=cfg.co2_scenario,
            co2_csv_path=cfg.co2_csv_path,
        ),
        o3_provider=resolve_ozone_provider(
            o3_mode=cfg.o3_mode,
            o3_file_path=cfg.o3_file_path,
        ),
    )
    broadcast_host_surface_to_subgrid(model)
    if bool(cfg.enable_land_carbon):
        from aoba.land_carbon import init_land_carbon_state

        model.land_carbon = init_land_carbon_state(
            model.sfc_sub.surf_type,
            model.sfc_sub.bio.lai,
            model.sfc_sub.bio.h_canopy,
            device=dev,
            maml_n_sub=n_sub,
        )
    if bool(cfg.enable_superparam) and bool(cfg.enable_crm_self_distill):
        from aoba.crm_self_distill import init_crm_self_distill_state, init_sparse_crm_pool

        model.crm_distill = init_crm_self_distill_state(
            n_cells=nc,
            nz=nz,
            device=dev,
            stride=int(cfg.crm_distill_stride),
            warmup_steps=int(cfg.crm_distill_warmup_steps),
            dwell_steps=int(cfg.crm_distill_dwell_steps),
            hidden_dim=int(cfg.crm_distill_hidden_dim),
            replay_size=int(cfg.crm_distill_replay_size),
            ny=ny_t,
            nx=nx_t,
        )
        if use_pool:
            st_eff = max(int(cfg.crm_distill_stride), 2)
            n_pool = max(1, (nc + st_eff - 1) // st_eff)
            model.sparse_crm_pool = init_sparse_crm_pool(
                n_pool=n_pool,
                nz=nz,
                ny_full=ny_full,
                nx_full=nx_full,
                ny_tile=ny_t,
                nx_tile=nx_t,
                dx=float(cfg.crm_dx),
                device=dev,
                dtype=dtype,
            )
    if bool(cfg.enable_rad_self_distill):
        from aoba.crm_self_distill import init_rad_self_distill_state

        r_stride = max(int(cfg.rad_stride), 1)
        model.rad_distill = init_rad_self_distill_state(
            n_cells=nc,
            nz=nz,
            device=dev,
            stride=int(cfg.crm_distill_stride),
            warmup_steps=max(int(cfg.crm_distill_warmup_steps) // r_stride, 6),
            dwell_steps=max(int(cfg.crm_distill_dwell_steps) // r_stride, 4),
            hidden_dim=min(max(int(cfg.crm_distill_hidden_dim) * 2 // 3, 64), 192),
            replay_size=int(cfg.crm_distill_replay_size),
        )
    return model


def _apply_dynamic_icesheet_update(model: AOBAState) -> None:
    """Advance 2D SIA ice sheet using annual EMA SMB, flip Land<->IceSheet masks, and remap vertical grid."""
    ice = model.icesheet
    sfc_sub = model.sfc_sub
    assert ice is not None and sfc_sub is not None
    cfg = model.cfg
    mesh = model.mesh
    nc = mesh.n_cells
    n_sub = resolve_crm_sfc_tiles(cfg)

    sim_yr = model.sim_time_sec / (C.DAYS_PER_YEAR * C.SECONDS_PER_DAY)
    if cfg.icesheet_accel_spinup_years > 0.0:
        accel_eff = 1.0 + (cfg.icesheet_accel_factor - 1.0) * math.exp(-sim_yr / cfg.icesheet_accel_spinup_years)
    else:
        accel_eff = cfg.icesheet_accel_factor

    ice_diag = step_sia_icesheet_scvt(
        ice=ice,
        mesh=mesh,
        ema_years=cfg.icesheet_ema_years,
        accel_factor=accel_eff,
        isostasy_tau_yr=cfg.icesheet_isostasy_tau_yr,
    )

    is_ocean = mesh.ocean_frac >= 0.5
    new_surf_type = torch.where(
        is_ocean,
        torch.zeros_like(mesh.surf_type),
        torch.where(
            ice.h_ice >= H_ICE_THRESHOLD_M,
            torch.full_like(mesh.surf_type, SURF_ICESHEET),
            torch.full_like(mesh.surf_type, SURF_LAND),
        ),
    )
    newly_glaciated_cell = (mesh.surf_type != SURF_ICESHEET) & (new_surf_type == SURF_ICESHEET)
    newly_deglaciated_cell = (mesh.surf_type == SURF_ICESHEET) & (new_surf_type == SURF_LAND)

    old_z_topo = mesh.z_topo
    mesh.surf_type = new_surf_type
    mesh.z_topo = ice_diag["z_topo_new"]
    mesh.z_topo_smooth = ice_diag["z_topo_smooth_new"]
    if mesh.z_min_land is not None:
        mesh.z_min_land = mesh.z_min_land + (mesh.z_topo - old_z_topo)
    model.sfc.surf_type = new_surf_type
    model.sfc.z_topo = mesh.z_topo

    new_surf_sub = _cell_to_sub(new_surf_type, nc, n_sub)
    sfc_sub.surf_type = new_surf_sub
    sfc_sub.z_topo = _cell_to_sub(mesh.z_topo, nc, n_sub)

    if newly_glaciated_cell.any():
        gl_sub = _cell_to_sub(newly_glaciated_cell, nc, n_sub)
        sfc_sub.swe_land = torch.where(gl_sub, torch.full_like(sfc_sub.swe_land, SWE_GLACIER_MAX), sfc_sub.swe_land)
        sfc_sub.swe_max_season = torch.where(gl_sub, torch.full_like(sfc_sub.swe_max_season, SWE_GLACIER_MAX), sfc_sub.swe_max_season)
        sfc_sub.bio.lai = torch.where(gl_sub, torch.zeros_like(sfc_sub.bio.lai), sfc_sub.bio.lai)
        if sfc_sub.bio.lai_evg is not None:
            sfc_sub.bio.lai_evg = torch.where(gl_sub, torch.zeros_like(sfc_sub.bio.lai_evg), sfc_sub.bio.lai_evg)
        if sfc_sub.bio.lai_dec is not None:
            sfc_sub.bio.lai_dec = torch.where(gl_sub, torch.zeros_like(sfc_sub.bio.lai_dec), sfc_sub.bio.lai_dec)
        if sfc_sub.bio.f_evg is not None:
            sfc_sub.bio.f_evg = torch.where(gl_sub, torch.zeros_like(sfc_sub.bio.f_evg), sfc_sub.bio.f_evg)
        sfc_sub.bio.h_canopy = torch.where(gl_sub, torch.zeros_like(sfc_sub.bio.h_canopy), sfc_sub.bio.h_canopy)

    if newly_deglaciated_cell.any():
        dg_sub = _cell_to_sub(newly_deglaciated_cell, nc, n_sub)
        sfc_sub.swe_land = torch.where(dg_sub, torch.zeros_like(sfc_sub.swe_land), sfc_sub.swe_land)
        sfc_sub.swe_max_season = torch.where(dg_sub, torch.full_like(sfc_sub.swe_max_season, 10.0), sfc_sub.swe_max_season)
        sfc_sub.w_liq_soil = torch.where(dg_sub.unsqueeze(0), torch.full_like(sfc_sub.w_liq_soil, 0.25), sfc_sub.w_liq_soil)
        sfc_sub.w_ice_soil = torch.where(dg_sub.unsqueeze(0), torch.zeros_like(sfc_sub.w_ice_soil), sfc_sub.w_ice_soil)
        from aoba.surface import compute_root_zone_weights_and_water
        w_dg_avail, _, _ = compute_root_zone_weights_and_water(sfc_sub.w_liq_soil, sfc_sub.bio.s_root_cap)
        sfc_sub.w_root = torch.where(dg_sub, w_dg_avail, sfc_sub.w_root)
        sfc_sub.bio.lai = torch.where(dg_sub, torch.full_like(sfc_sub.bio.lai, 0.25), sfc_sub.bio.lai)
        if sfc_sub.bio.lai_evg is not None:
            sfc_sub.bio.lai_evg = torch.where(dg_sub, torch.full_like(sfc_sub.bio.lai_evg, 0.125), sfc_sub.bio.lai_evg)
        if sfc_sub.bio.lai_dec is not None:
            sfc_sub.bio.lai_dec = torch.where(dg_sub, torch.full_like(sfc_sub.bio.lai_dec, 0.125), sfc_sub.bio.lai_dec)
        if sfc_sub.bio.f_evg is not None:
            sfc_sub.bio.f_evg = torch.where(dg_sub, torch.full_like(sfc_sub.bio.f_evg, 0.5), sfc_sub.bio.f_evg)
        sfc_sub.bio.h_canopy = torch.where(dg_sub, torch.full_like(sfc_sub.bio.h_canopy, 0.5), sfc_sub.bio.h_canopy)

    # Keep active ice-sheet cells replenished to glacier cap since net mass change was transferred to ice.h_ice
    is_ice_sub = new_surf_sub == SURF_ICESHEET
    if is_ice_sub.any():
        sfc_sub.swe_land = torch.where(is_ice_sub, torch.full_like(sfc_sub.swe_land, SWE_GLACIER_MAX), sfc_sub.swe_land)
        sfc_sub.swe_max_season = torch.where(is_ice_sub, torch.full_like(sfc_sub.swe_max_season, SWE_GLACIER_MAX), sfc_sub.swe_max_season)

    remap_atmosphere_and_crm_vertical(model.atmos, model.crm, mesh, model.vgrid, mesh.z_topo_smooth)
    # Recompute Priority-Flood drainage & topographic lake basins after ice-sheet/isostasy topography shift
    if model.river is not None:
        from aoba.river_routing import init_river_routing_state
        old_stor = model.river.storage_kg
        old_comp = model.river.storage_comp
        model.river = init_river_routing_state(mesh, device=mesh.z_topo.device, dtype=old_stor.dtype)
        model.river.storage_kg = old_stor * model.river.is_land_river.to(old_stor.dtype)
        model.river.storage_comp = old_comp * model.river.is_land_river.to(old_stor.dtype)
    if model.lakes is not None:
        from aoba.lakes import rebuild_lake_basins_from_topography
        spill_m3 = rebuild_lake_basins_from_topography(
            model.lakes,
            mesh,
            max_frac=float(cfg.lake_max_frac),
            max_depth_m=float(cfg.lake_max_depth_m),
        )
        if model.river is not None:
            from aoba.surface import _kahan_add
            model.river.storage_kg, model.river.storage_comp = _kahan_add(
                model.river.storage_kg, spill_m3 * 1000.0, model.river.storage_comp
            )


# ---------------------------------------------------------------------------
# 3. Coupled Global Time Step (`step_aoba_model`)
# ---------------------------------------------------------------------------
@dataclass
class GlobalStepDiagnostics:
    """Global area-weighted summary diagnostics from one coupled `step_aoba_model` call."""

    sim_time_days: float
    global_mean_t_sfc_k: float
    global_mean_olr_wm2: float
    global_mean_asr_wm2: float
    global_mean_precip_mm_day: float
    land_mean_gpp_umol: float
    land_mean_lai: float
    seaice_area_million_km2: float
    date_str: str = ""


_ACCLIM_BIO_ATTRS: tuple[str, ...] = (
    "ema_vis_beam", "ema_vis_diff", "ema_mu0_mid", "ema_temp_mid",
    "ema_vpd_mid", "ema_temp_night", "ema_temp_home", "vcmax25_c3", "jmax25_c3", "r25_c3",
    "vcmax25_c4", "jmax25_c4", "vpmax25_c4", "r25_c4", "f_c4",
)


def _coupled_post_crm_accum_core(
    rho_post_0: torch.Tensor,
    thm_post_0: torch.Tensor,
    qv_post_0: torch.Tensor,
    dz0_sub: torch.Tensor,
    rain_new_sub: torch.Tensor,
    snow_new_sub: torch.Tensor,
    sw1_dir_sub: torch.Tensor,
    sw1_dif_sub: torch.Tensor,
    sw2_dir_sub: torch.Tensor,
    sw2_dif_sub: torch.Tensor,
    lw_dn_sub: torch.Tensor,
    alb_vis_dir: torch.Tensor,
    alb_vis_dif: torch.Tensor,
    alb_nir_dir: torch.Tensor,
    alb_nir_dif: torch.Tensor,
    emiss_sfc: torch.Tensor,
    t_rad_sfc: torch.Tensor,
    day_precip_mm: torch.Tensor,
    day_pet_mm: torch.Tensor,
    day_ppfd_mol: torch.Tensor,
    day_temp_sum: torch.Tensor,
    day_vpd_sum: torch.Tensor,
    day_t_min: torch.Tensor,
    dt: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pure-tensor post-CRM lowest-level meteorology, Priestley-Taylor PET & daily accumulators."""
    pres_post_0, _, t_air_sub = diagnose_pressure_temperature(rho_post_0, thm_post_0, qv_post_0)
    pres_sfc_post_sub = pres_post_0 + C.GRAVITY * rho_post_0 * (1.0 + qv_post_0) * dz0_sub
    t_denom = torch.clamp(t_air_sub - 29.65, min=10.0)
    es_sub = 611.2 * torch.exp(17.67 * (t_air_sub - C.T_FREEZE) / t_denom)
    e_sub = qv_post_0 * pres_post_0 / (C.EPS_WATER + (1.0 - C.EPS_WATER) * qv_post_0)
    vpd_sub = torch.clamp(es_sub - e_sub, min=50.0)

    precip_tot_sub = rain_new_sub + snow_new_sub
    sw_net_sub = (
        (1.0 - alb_vis_dir) * sw1_dir_sub
        + (1.0 - alb_vis_dif) * sw1_dif_sub
        + (1.0 - alb_nir_dir) * sw2_dir_sub
        + (1.0 - alb_nir_dif) * sw2_dif_sub
    )
    lw_net_sub = emiss_sfc * (lw_dn_sub - C.STEFAN_BOLTZMANN * (t_rad_sfc ** 4))
    rn_pos_sub = torch.clamp(sw_net_sub + lw_net_sub, min=0.0)
    des_dt_sub = es_sub * (17.67 * 243.5) / (t_denom * t_denom)
    gamma_psy_sub = (C.CP_D * pres_sfc_post_sub) / (C.EPS_WATER * C.L_V0)
    pet_pt_sub = 1.26 * (des_dt_sub / (des_dt_sub + gamma_psy_sub)) * rn_pos_sub / C.L_V0

    day_precip_new = day_precip_mm + precip_tot_sub * dt
    day_pet_new = day_pet_mm + pet_pt_sub * dt
    day_ppfd_new = day_ppfd_mol + (4.57 * sw1_dir_sub + 4.24 * sw1_dif_sub) * 1.0e-6 * dt
    day_temp_new = day_temp_sum + t_air_sub
    day_vpd_new = day_vpd_sum + vpd_sub
    day_t_min_new = torch.minimum(day_t_min, t_air_sub)
    return (
        t_air_sub, pres_sfc_post_sub, vpd_sub,
        day_precip_new, day_pet_new, day_ppfd_new,
        day_temp_new, day_vpd_new, day_t_min_new,
    )


# AOBA Formulation Eqs. (8.13), (9.8)
def step_aoba_model(
    model: AOBAState,
    compute_diagnostics: bool = True,
    sync_host_surface: bool = True,
) -> GlobalStepDiagnostics:
    """Advance the global 1:1 Superparameterized AOBA Earth System Model by `cfg.dt_host` [s]."""
    assert model.sfc_sub is not None and model.clim_sub is not None
    cfg = model.cfg
    mesh = model.mesh
    vgrid = model.vgrid
    atmos = model.atmos
    crm = model.crm
    sfc = model.sfc
    sfc_sub = model.sfc_sub
    clim = model.clim
    clim_sub = model.clim_sub

    dt = cfg.dt_host
    nz, nc, ny, nx = crm.rho_d.shape
    ny_full, nx_full = max(int(cfg.crm_ny), 1), max(int(cfg.crm_nx), 1)
    n_crm_full = ny_full * nx_full
    ny_t, nx_t = resolve_crm_sfc_tile_grid(ny_full, nx_full, getattr(cfg, "crm_sfc_tiles", 0))
    n_crm = ny * nx
    m_crm = nc * n_crm
    n_sub = ny_t * nx_t
    m_sub = nc * n_sub
    pool = getattr(model, "sparse_crm_pool", None)
    has_pool = pool is not None

    def _c2s(x: torch.Tensor, n_cells: int = nc) -> torch.Tensor:
        return _crm_to_sfc_tile(x, n_cells, n_sub, n_crm, ny=ny, nx=nx, ny_t=ny_t, nx_t=nx_t)

    def _s2c(x: torch.Tensor, n_cells: int = nc) -> torch.Tensor:
        return _sfc_tile_to_crm(x, n_cells, n_sub, n_crm, ny=ny, nx=nx, ny_t=ny_t, nx_t=nx_t)

    def _f2s(x: torch.Tensor, n_cells: int) -> torch.Tensor:
        return _crm_to_sfc_tile(x, n_cells, n_sub, n_crm_full, ny=ny_full, nx=nx_full, ny_t=ny_t, nx_t=nx_t)

    def _s2f(x: torch.Tensor, n_cells: int) -> torch.Tensor:
        return _sfc_tile_to_crm(x, n_cells, n_sub, n_crm_full, ny=ny_full, nx=nx_full, ny_t=ny_t, nx_t=nx_t)

    # Pre-synchronize teacher-only SparseCRMPool once per step so both full-3D ICA/TICA radiation
    # and full-3D RK3-HEVI CRM dynamics operate on the exact same slot-preserving teacher cohort.
    t_pool_idx: torch.Tensor | None = None
    s_pool_idx: torch.Tensor | None = None
    n_pool_act = 0
    if has_pool:
        from aoba.crm_self_distill import (
            init_crm_self_distill_state,
            select_teacher_indices,
            sync_sparse_crm_pool,
        )

        if model.crm_distill is None:
            model.crm_distill = init_crm_self_distill_state(
                n_cells=nc,
                nz=nz,
                device=atmos.rho_d.device,
                stride=int(cfg.crm_distill_stride),
                warmup_steps=int(cfg.crm_distill_warmup_steps),
                dwell_steps=int(cfg.crm_distill_dwell_steps),
                hidden_dim=int(cfg.crm_distill_hidden_dim),
                replay_size=int(cfg.crm_distill_replay_size),
                ny=ny_t,
                nx=nx_t,
            )
        pool_sel = select_teacher_indices(model.crm_distill, allow_sparse_warmup=True)
        assert pool_sel is not None
        t_raw_pool, s_pool_idx = pool_sel
        prev_qrad_4d = (
            model.rad_cache["q_rad_sub"].reshape(nz, nc, ny_t, nx_t)
            if ("q_rad_sub" in model.rad_cache and model.rad_cache["q_rad_sub"].shape[-1] == m_sub)
            else None
        )
        t_pool_idx, n_pool_act = sync_sparse_crm_pool(
            pool, t_raw_pool, crm, atmos.rho_d, mesh.f_cell,
            tile_q_rad_4d=prev_qrad_4d, step_seed=model.step_count,
        )

    # 1. Orbital solar geometry & zero-sync CPU local solar noon detection
    yr_sec_trop = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    _, t_diur_prev = model.calendar.orbital_times_sec(model.sim_time_sec, yr_sec_trop)
    t_seas_next, t_diur_next = model.calendar.orbital_times_sec(model.sim_time_sec + dt, yr_sec_trop)
    lon_cpu = getattr(mesh, "_lon_cell_cpu", None)
    if lon_cpu is None:
        lon_cpu = mesh.lon_cell.detach().cpu()
        setattr(mesh, "_lon_cell_cpu", lon_cpu)
    frac_prev = (t_diur_prev % C.SECONDS_PER_DAY) / C.SECONDS_PER_DAY
    frac_next = (t_diur_next % C.SECONDS_PER_DAY) / C.SECONDS_PER_DAY
    h_raw_prev = 2.0 * math.pi * frac_prev + lon_cpu - math.pi
    h_raw_next = 2.0 * math.pi * frac_next + lon_cpu - math.pi
    h_prev_cpu = torch.atan2(torch.sin(h_raw_prev), torch.cos(h_raw_prev))
    h_next_cpu = torch.atan2(torch.sin(h_raw_next), torch.cos(h_raw_next))
    crossed_noon_cpu = (h_prev_cpu < 0.0) & (h_next_cpu >= 0.0) & ((h_next_cpu - h_prev_cpu) < math.pi)
    # Optional analytical Milankovitch orbital evolution (Berger, 1978; Laskar et al., 2004)
    obl_now, ecc_now, peri_now = cfg.obliquity_deg, cfg.eccentricity, cfg.perihelion_deg
    if bool(cfg.enable_milankovitch_orbit):
        from aoba.planetary_geochem import compute_milankovitch_orbital_params

        sim_yr_now = model.sim_time_sec / yr_sec_trop
        t_kyr = float(cfg.milankovitch_start_kyr) + (sim_yr_now * 1.0e-3) * max(float(cfg.milankovitch_accel_factor), 1.0)
        ecc_now, obl_now, peri_now = compute_milankovitch_orbital_params(t_kyr)
    # Optional 11-year Schwabe solar cycle modulation of TSI & stratospheric UV
    s0_eff = float(cfg.solar_constant)
    uv_j_o2_scale = 1.0
    if bool(cfg.enable_solar_cycle) or bool(cfg.enable_volcanic_aerosols):
        from aoba.volcanic_solar import compute_solar_cycle_modulation, init_volcanic_solar_state

        if model.volcanic_solar is None:
            model.volcanic_solar = init_volcanic_solar_state(nc, atmos.rho_d.device, atmos.rho_d.dtype)
        if bool(cfg.enable_solar_cycle):
            tsi_d, uv_j_o2_scale = compute_solar_cycle_modulation(
                model.sim_time_sec / C.SECONDS_PER_DAY,
                amp_wm2=cfg.solar_cycle_amp_wm2,
                period_yr=cfg.solar_cycle_period_yr,
                phase_yr=cfg.solar_cycle_phase_yr,
            )
            model.volcanic_solar.tsi_delta_wm2 = tsi_d
            model.volcanic_solar.uv_j_o2_scale = uv_j_o2_scale
            s0_eff = max(100.0, s0_eff + tsi_d)
    if bool(cfg.enable_tidal_lock):
        from aoba.planetary_presets import compute_tidally_locked_insolation

        mu0, s_toa, hour_angle_cell = compute_tidally_locked_insolation(
            mesh.lat_cell, mesh.lon_cell, s0_eff, cfg.substellar_lon_deg
        )
    else:
        mu0, s_toa, hour_angle_cell = compute_orbital_insolation(
            t_seas_next, mesh.lat_cell, mesh.lon_cell,
            s0_eff, obl_now, ecc_now, peri_now,
            t_diurnal_sec=t_diur_next, diurnal_cycle=cfg.diurnal_cycle,
        )
    mu0_sub = _cell_to_sub(mu0, nc, n_sub)
    s_toa_sub = _cell_to_sub(s_toa, nc, n_sub)

    # 2. Subcolumn CRM lowest-layer meteorology & host cell reference heights
    u_crm_0 = (0.5 * (crm.u[0] + torch.roll(crm.u[0], 1, dims=-1))).reshape(m_crm)
    v_crm_0 = (0.5 * (crm.v[0] + torch.roll(crm.v[0], 1, dims=-2))).reshape(m_crm)
    u_sub_0 = _c2s(u_crm_0)
    v_sub_0 = _c2s(v_crm_0)
    rho_crm_flat = crm.rho_d.reshape(nz, m_crm)
    thm_crm_flat = crm.theta_m.reshape(nz, m_crm)
    qv_crm_flat = crm.q_v.reshape(nz, m_crm)
    qc_crm_flat = crm.q_c.reshape(nz, m_crm)
    qi_crm_flat = crm.q_i.reshape(nz, m_crm)

    rho_sub_0 = _c2s(rho_crm_flat[0])
    thm_sub_0 = _c2s(thm_crm_flat[0])
    qv_sub_0 = _c2s(qv_crm_flat[0])

    need_rad = (not model.rad_cache) or ("q_rad_sub" not in model.rad_cache) or (model.step_count % max(cfg.rad_stride, 1) == 0)
    pres_sub_0, _, temp_sub_0 = diagnose_pressure_temperature(rho_sub_0, thm_sub_0, qv_sub_0)
    dz0_col = atmos.z_c[0] - atmos.z_w[0]
    dz0_sub = _cell_to_sub(dz0_col, nc, n_sub)
    pres_sfc_sub = pres_sub_0 + C.GRAVITY * rho_sub_0 * (1.0 + qv_sub_0) * dz0_sub
    z_rel_c = atmos.z_c - atmos.z_w[0:1]
    z_ref_rel_sub = torch.clamp(_cell_to_sub(z_rel_c[0], nc, n_sub) - 0.65 * sfc_sub.bio.h_canopy, min=5.0)

    # 3. Subcolumn surface albedos & ICA 4-band radiative transfer across all `[Nz, m_sub]`
    _pre_alb_sub = compute_surface_albedos(sfc_sub, mu0_sub)
    alb_vd, alb_vf, alb_nd, alb_nf, _, lai_exp_sub, _, f_ice_sub = _pre_alb_sub
    _lake_f_sub_rad = None
    _lake_trad_sub = None
    if bool(cfg.enable_dynamic_lakes):
        from aoba.lakes import EMISS_LAKE, compute_lake_surface_albedos, init_dynamic_lake_state
        if model.lakes is None:
            model.lakes = init_dynamic_lake_state(
                mesh,
                init_fill_frac=float(cfg.lake_init_fill_frac),
                max_frac=float(cfg.lake_max_frac),
                max_depth_m=float(cfg.lake_max_depth_m),
                device=temp_sub_0.device,
                dtype=temp_sub_0.dtype,
            )
        l_avd, l_avf, l_and, l_anf, l_trad = compute_lake_surface_albedos(model.lakes, mu0)
        _lake_f_sub_rad = torch.clamp(_cell_to_sub(model.lakes.f_lake, nc, n_sub), 0.0, 1.0)
        _lake_trad_sub = _cell_to_sub(l_trad, nc, n_sub)
        omf_l = 1.0 - _lake_f_sub_rad
        alb_vd = omf_l * alb_vd + _lake_f_sub_rad * _cell_to_sub(l_avd, nc, n_sub)
        alb_vf = omf_l * alb_vf + _lake_f_sub_rad * _cell_to_sub(l_avf, nc, n_sub)
        alb_nd = omf_l * alb_nd + _lake_f_sub_rad * _cell_to_sub(l_and, nc, n_sub)
        alb_nf = omf_l * alb_nf + _lake_f_sub_rad * _cell_to_sub(l_anf, nc, n_sub)
    if need_rad:
        t_rad_ocn = ((1.0 - f_ice_sub) * (sfc_sub.t_ocean_ml ** 4) + f_ice_sub * (torch.clamp(sfc_sub.t_seaice, min=150.0) ** 4)) ** 0.25
        t_rad_land_eff = sfc_sub.t_canopy
        if _lake_f_sub_rad is not None and _lake_trad_sub is not None:
            t_rad_land_eff = ((1.0 - _lake_f_sub_rad) * (torch.clamp(t_rad_land_eff, min=150.0) ** 4) + _lake_f_sub_rad * (torch.clamp(_lake_trad_sub, min=150.0) ** 4)) ** 0.25
        t_rad_in = torch.where(sfc_sub.surf_type == SURF_OCEAN, t_rad_ocn, t_rad_land_eff)
        emiss_land_eff = torch.full_like(t_rad_in, EMISSIVITY_LAND)
        if _lake_f_sub_rad is not None:
            emiss_land_eff = (1.0 - _lake_f_sub_rad) * emiss_land_eff + _lake_f_sub_rad * EMISS_LAKE
        emiss_in = torch.where(
            sfc_sub.surf_type == SURF_OCEAN,
            torch.full_like(t_rad_in, EMISSIVITY_OCEAN),
            emiss_land_eff,
        )

        # Prepare host-cell [nz, nc] aerosol, dust, and ozone fields once before expanding only needed columns
        if cfg.enable_dust and atmos.q_dust is None:
            atmos.q_dust = torch.zeros_like(atmos.q_n)
        q_dust_cell = atmos.q_dust if (cfg.enable_dust and atmos.q_dust is not None) else None
        aero_opt_cell = None
        if bool(cfg.enable_aerosol_rad_forcing) and bool(cfg.enable_natural_aerosols) and model.aerosols is not None:
            from aoba.aerosols import compute_aerosol_radiative_optics

            aero_opt_cell = compute_aerosol_radiative_optics(model.aerosols, atmos.rho_d, atmos.dz_c)
        if bool(cfg.enable_volcanic_aerosols) and model.volcanic_solar is not None:
            from aoba.volcanic_solar import compute_volcanic_rad_optics_and_sad

            pres_host_v, _, _ = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
            v_cell, _ = compute_volcanic_rad_optics_and_sad(model.volcanic_solar, pres_host_v, atmos.dz_c)
            aero_opt_cell = v_cell if aero_opt_cell is None else tuple(a + b for a, b in zip(aero_opt_cell, v_cell))

        o3_prog_cell = None
        o3_use_provider = False
        doy_now_rad = 0.0
        if str(cfg.o3_mode).strip().lower() == "prognostic":
            if model.ozone is None:
                from aoba.ozone_chemistry import init_prognostic_ozone_state

                pres_host_oz, _, _ = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
                model.ozone = init_prognostic_ozone_state(pres_host_oz, atmos.rho_d, atmos.dz_c)
            o3_prog_cell = model.ozone.o3_vmr
        elif model.o3_provider is not None and model.o3_provider.mode != "rcemip":
            o3_use_provider = True
            doy_now_rad = float(model.calendar.date_at(model.sim_time_sec).doy)

        co2_rad_now = model.current_co2_ppmv()
        ch4_rad_now = model.current_ch4_ppmv()
        n2o_rad_now = model.current_n2o_ppmv()
        use_rad_distill = bool(cfg.enable_rad_self_distill)
        if use_rad_distill and model.rad_distill is None:
            from aoba.crm_self_distill import init_rad_self_distill_state

            r_stride = max(int(cfg.rad_stride), 1)
            model.rad_distill = init_rad_self_distill_state(
                n_cells=nc,
                nz=nz,
                device=rho_crm_flat.device,
                stride=int(cfg.crm_distill_stride),
                warmup_steps=max(int(cfg.crm_distill_warmup_steps) // r_stride, 6),
                dwell_steps=max(int(cfg.crm_distill_dwell_steps) // r_stride, 4),
                hidden_dim=min(max(int(cfg.crm_distill_hidden_dim) * 2 // 3, 64), 192),
                replay_size=int(cfg.crm_distill_replay_size),
            )
        rad_sel = None
        if use_rad_distill and model.rad_distill is not None:
            from aoba.crm_self_distill import (
                evaluate_rad_shadow_on_teachers,
                extract_rad_distill_features,
                pack_rad_teacher_context_and_targets,
                predict_rad_student_columns,
                select_rad_teacher_indices,
                train_rad_student_online,
            )

            sparse_warmup_3d = (ny > 1 and nx > 1 and m_crm > 100_000)
            rad_sel = select_rad_teacher_indices(
                model.rad_distill, crm_state=model.crm_distill, allow_sparse_warmup=sparse_warmup_3d,
            )

        use_2m_re = (
            str(cfg.microphysics_scheme).strip().lower() in ("two_moment", "six_class_2m")
            and crm.n_c is not None
            and crm.n_i is not None
        )

        if has_pool:
            assert t_pool_idx is not None and s_pool_idx is not None
            dev_r = rho_crm_flat.device
            n_tr = n_pool_act
            n_sr = int(s_pool_idx.numel())
            sfc_off = torch.arange(n_sub, device=dev_r, dtype=torch.int64).unsqueeze(0)
            t_sfc = (t_pool_idx.unsqueeze(1) * n_sub + sfc_off).reshape(-1)
            s_sfc = (s_pool_idx.unsqueeze(1) * n_sub + sfc_off).reshape(-1)

            # 3p-a. Active teachers: 100% genuine full-3D `(ny_full, nx_full)` ICA / 3D-TICA radiation
            m_tp = n_tr * n_crm_full
            rho_t = pool.crm.rho_d[:, :n_tr].reshape(nz, m_tp)
            thm_t = pool.crm.theta_m[:, :n_tr].reshape(nz, m_tp)
            qv_t = pool.crm.q_v[:, :n_tr].reshape(nz, m_tp)
            qc_t = pool.crm.q_c[:, :n_tr].reshape(nz, m_tp)
            qi_t = pool.crm.q_i[:, :n_tr].reshape(nz, m_tp)
            pres_t, _, temp_t = diagnose_pressure_temperature(rho_t, thm_t, qv_t)
            dz_t = _cell_to_sub(atmos.dz_c[:, t_pool_idx], n_tr, n_crm_full)
            pres_sfc_t = pres_t[0] + C.GRAVITY * rho_t[0] * (1.0 + qv_t[0]) * dz_t[0]
            pres_w_t = torch.empty((nz + 1, m_tp), device=dev_r, dtype=pres_t.dtype)
            pres_w_t[0] = pres_sfc_t
            pres_w_t[1:-1] = torch.sqrt(torch.clamp(pres_t[:-1] * pres_t[1:], min=1.0))
            pres_w_t[-1] = torch.clamp(0.5 * pres_t[-1], min=0.1)
            temp_w_t = torch.empty((nz + 1, m_tp), device=dev_r, dtype=temp_t.dtype)
            temp_w_t[0] = temp_t[0]
            temp_w_t[1:-1] = 0.5 * (temp_t[:-1] + temp_t[1:])
            temp_w_t[-1] = temp_t[-1]

            if bool(cfg.enable_rad_3d_tica) and n_crm_full > 1:
                from aoba.radiation_hd import apply_3d_tica_cloud_shift

                qc_t, qi_t = apply_3d_tica_cloud_shift(
                    qc_t, qi_t, z_rel_c[:, t_pool_idx], mu0[t_pool_idx], hour_angle_cell[t_pool_idx],
                    n_tr, ny_full, nx_full, float(cfg.crm_dx),
                )
            use_2m_re_p = (
                str(cfg.microphysics_scheme).strip().lower() in ("two_moment", "six_class_2m")
                and getattr(pool.crm, "n_c", None) is not None
                and getattr(pool.crm, "n_i", None) is not None
            )
            if use_2m_re_p:
                re_l_t, re_i_t = diagnose_cloud_effective_radii(
                    rho_t, qc_t, qi_t,
                    pool.crm.n_c[:, :n_tr].reshape(nz, m_tp),
                    pool.crm.n_i[:, :n_tr].reshape(nz, m_tp),
                    re_liq_fallback=cfg.re_liq_um, re_ice_fallback=cfg.re_ice_um,
                )
            else:
                re_l_t, re_i_t = cfg.re_liq_um, cfg.re_ice_um

            t_rad_t = _s2f(t_rad_in[t_sfc], n_tr)
            emiss_t = _s2f(emiss_in[t_sfc], n_tr)
            alb_vd_t = _s2f(alb_vd[t_sfc], n_tr)
            alb_vf_t = _s2f(alb_vf[t_sfc], n_tr)
            alb_nd_t = _s2f(alb_nd[t_sfc], n_tr)
            alb_nf_t = _s2f(alb_nf[t_sfc], n_tr)
            mu0_t = _cell_to_sub(mu0[t_pool_idx], n_tr, n_crm_full)
            s_toa_t = _cell_to_sub(s_toa[t_pool_idx], n_tr, n_crm_full)
            qd_t = _cell_to_sub(q_dust_cell[:, t_pool_idx], n_tr, n_crm_full) if q_dust_cell is not None else None
            ao_t = tuple(_cell_to_sub(a[:, t_pool_idx], n_tr, n_crm_full) for a in aero_opt_cell) if aero_opt_cell is not None else None
            if o3_prog_cell is not None:
                o3_t = _cell_to_sub(o3_prog_cell[:, t_pool_idx], n_tr, n_crm_full)
            elif o3_use_provider:
                o3_t = model.o3_provider.compute_vmr(
                    pres_t, lat_rad=_cell_to_sub(mesh.lat_cell[t_pool_idx], n_tr, n_crm_full), doy=doy_now_rad
                )
            else:
                o3_t = None

            q_rad_t, sw1d_t, sw1f_t, sw2d_t, sw2f_t, lwd_t, olr_t, asr_t = (
                _compute_four_band_radiation_chunked(
                    pres_c=pres_t, pres_w=pres_w_t, temp_c=temp_t, temp_w=temp_w_t,
                    dz_c=dz_t, rho_d_c=rho_t, qv_c=qv_t, qc_c=qc_t, qi_c=qi_t,
                    t_sfc=t_rad_t, albedo_vis=alb_vd_t, albedo_nir=alb_nd_t,
                    mu0=mu0_t, s_toa=s_toa_t, co2_ppmv=co2_rad_now, emiss_sfc=emiss_t,
                    albedo_vis_dif=alb_vf_t, albedo_nir_dif=alb_nf_t, col_chunk=cfg.rad_col_chunk,
                    re_liq_um=re_l_t, re_ice_um=re_i_t, cloud_sw_inhomog=cfg.cloud_sw_inhomog,
                    q_dust_c=qd_t, o3_vmr_c=o3_t, o3_lw_coupling=cfg.o3_lw_coupling,
                    rad_scheme=cfg.rad_scheme, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now, aero_opt=ao_t,
                )
            )
            pool.q_rad[:, :n_tr] = q_rad_t.reshape(nz, n_tr, ny_full, nx_full)
            pool.q_rad_valid = True

            q_rad_sub = torch.empty((nz, m_sub), device=dev_r, dtype=rho_crm_flat.dtype)
            sw1_dir_sub = torch.empty_like(t_rad_in)
            sw1_dif_sub = torch.empty_like(t_rad_in)
            sw2_dir_sub = torch.empty_like(t_rad_in)
            sw2_dif_sub = torch.empty_like(t_rad_in)
            lw_dn_sub = torch.empty_like(t_rad_in)
            olr_sub = torch.empty_like(t_rad_in)
            asr_sub = torch.empty_like(t_rad_in)

            # 3p-b. Student cells at tile resolution `(ny_t, nx_t)`: RadSelfDistillNet or genuine tile ICA
            if use_rad_distill and model.rad_distill is not None:
                from aoba.crm_self_distill import (
                    evaluate_rad_shadow_on_teachers,
                    extract_rad_distill_features,
                    pack_rad_teacher_context_and_targets,
                    predict_rad_student_columns,
                    train_rad_student_online,
                )

                if m_tp > 6144:
                    tr_s = torch.randperm(m_tp, device=dev_r)[:6144]
                else:
                    tr_s = slice(None)
                x_rt, cp_rt_raw, s_inc_rt = extract_rad_distill_features(
                    pres_t[:, tr_s], temp_t[:, tr_s], dz_t[:, tr_s], rho_t[:, tr_s],
                    qv_t[:, tr_s], qc_t[:, tr_s], qi_t[:, tr_s],
                    t_rad_t[tr_s], alb_vd_t[tr_s], alb_vf_t[tr_s], alb_nd_t[tr_s], alb_nf_t[tr_s],
                    emiss_t[tr_s], mu0_t[tr_s], s_toa_t[tr_s],
                    co2_ppmv=co2_rad_now, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now,
                    o3_vmr_c=(o3_t[:, tr_s] if o3_t is not None else None),
                    q_dust_c=(qd_t[:, tr_s] if qd_t is not None else None),
                    tau_aero_sw_c=(ao_t[0][:, tr_s] if ao_t is not None else None),
                )
                if bool(cfg.enable_distill_shadow_diag):
                    land_mask_rt = _s2f((sfc_sub.surf_type[t_sfc] == SURF_LAND).to(t_rad_t.dtype), n_tr)[tr_s]
                    t_cell_sub_rt = t_pool_idx.unsqueeze(1).expand(n_tr, n_crm_full).reshape(-1)[tr_s]
                    evaluate_rad_shadow_on_teachers(
                        model.rad_distill, x_rt, cp_rt_raw, s_inc_rt,
                        alb_vd_t[tr_s], alb_vf_t[tr_s], alb_nd_t[tr_s], alb_nf_t[tr_s],
                        emiss_t[tr_s], t_rad_t[tr_s],
                        q_rad_t=q_rad_t[:, tr_s], sw1d_t=sw1d_t[tr_s], sw1f_t=sw1f_t[tr_s],
                        sw2d_t=sw2d_t[tr_s], sw2f_t=sw2f_t[tr_s], lwd_t=lwd_t[tr_s],
                        olr_t=olr_t[tr_s], asr_t=asr_t[tr_s],
                        land_mask_sub=land_mask_rt,
                        t_cell_idx_sub=t_cell_sub_rt,
                        step_count=model.step_count,
                    )
                cp_rt, cs_rt, tqr_rt, ts_rt = pack_rad_teacher_context_and_targets(
                    cp_rt_raw, s_inc_rt,
                    alb_vd_t[tr_s], alb_vf_t[tr_s], alb_nd_t[tr_s], alb_nf_t[tr_s], emiss_t[tr_s], t_rad_t[tr_s],
                    q_rad_t[:, tr_s], sw1d_t[tr_s], sw1f_t[tr_s], sw2d_t[tr_s], sw2f_t[tr_s],
                    lwd_t[tr_s], olr_t[tr_s], asr_t[tr_s],
                )
                n_rgrad = 4 if model.rad_distill.steps_seen < model.rad_distill.warmup_steps else 2
                train_rad_student_online(
                    model.rad_distill, x_rt, cp_rt, cs_rt, tqr_rt, ts_rt, n_grad_steps=n_rgrad
                )

                # Chunked zero-gather student inference over contiguous cell slices (<=32,768 subcols/chunk):
                # eliminates >2 GB of one-shot full-grid temporary tensors & gathers/scatters while
                # staying bit-exact (ICA subcolumns are strictly independent; teacher cells overwritten below).
                chunk_cells = max(1, 32768 // max(n_sub, 1))
                # Balance chunk size slightly upward when tail is tiny (e.g. G6 40962 -> 10x4097 instead of 10x4096+2)
                n_chunks_r = max(1, (nc + chunk_cells - 1) // chunk_cells)
                if n_chunks_r > 1 and (nc % chunk_cells) <= max(64, chunk_cells // 16):
                    chunk_cells = (nc + n_chunks_r - 2) // (n_chunks_r - 1)
                for c0 in range(0, nc, chunk_cells):
                    c1 = min(c0 + chunk_cells, nc)
                    n_ch = c1 - c0
                    s0 = c0 * n_sub
                    s1 = c1 * n_sub
                    rho_c = rho_crm_flat[:, s0:s1].contiguous()
                    thm_c = thm_crm_flat[:, s0:s1].contiguous()
                    qv_c = qv_crm_flat[:, s0:s1].contiguous()
                    qc_c = qc_crm_flat[:, s0:s1].contiguous()
                    qi_c = qi_crm_flat[:, s0:s1].contiguous()
                    pres_c, _, temp_c = diagnose_pressure_temperature(rho_c, thm_c, qv_c)
                    dz_c_sl = atmos.dz_c[:, c0:c1]
                    qd_c_sl = q_dust_cell[:, c0:c1] if q_dust_cell is not None else None
                    tau_c_sl = aero_opt_cell[0][:, c0:c1] if aero_opt_cell is not None else None
                    if o3_prog_cell is not None:
                        o3_c_sl = o3_prog_cell[:, c0:c1]
                    elif o3_use_provider:
                        o3_c_sl = model.o3_provider.compute_vmr(
                            pres_c, lat_rad=_cell_to_sub(mesh.lat_cell[c0:c1], n_ch, n_sub), doy=doy_now_rad
                        )
                    else:
                        o3_c_sl = None
                    x_rc, cp_rc, s_inc_rc = extract_rad_distill_features(
                        pres_c, temp_c, dz_c_sl, rho_c, qv_c, qc_c, qi_c,
                        t_rad_in[s0:s1], alb_vd[s0:s1], alb_vf[s0:s1], alb_nd[s0:s1], alb_nf[s0:s1],
                        emiss_in[s0:s1], mu0_sub[s0:s1], s_toa_sub[s0:s1],
                        co2_ppmv=co2_rad_now, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now,
                        o3_vmr_c=o3_c_sl, q_dust_c=qd_c_sl, tau_aero_sw_c=tau_c_sl,
                        cell_sub=n_sub,
                    )
                    rad_stu_out, rad_stu_unc = predict_rad_student_columns(
                        model.rad_distill, x_rc, cp_rc, s_inc_rc,
                        alb_vd[s0:s1], alb_vf[s0:s1], alb_nd[s0:s1], alb_nf[s0:s1],
                        emiss_in[s0:s1], t_rad_in[s0:s1],
                    )
                    q_rad_sub[:, s0:s1] = rad_stu_out["q_rad"]
                    sw1_dir_sub[s0:s1] = rad_stu_out["sw1_dir"]
                    sw1_dif_sub[s0:s1] = rad_stu_out["sw1_dif"]
                    sw2_dir_sub[s0:s1] = rad_stu_out["sw2_dir"]
                    sw2_dif_sub[s0:s1] = rad_stu_out["sw2_dif"]
                    lw_dn_sub[s0:s1] = rad_stu_out["lw_dn"]
                    olr_sub[s0:s1] = rad_stu_out["olr"]
                    asr_sub[s0:s1] = rad_stu_out["asr"]
                    model.rad_distill.last_uncertainty[c0:c1] = rad_stu_unc.reshape(n_ch, n_sub).mean(dim=-1)

                # Overwrite active teacher cells with genuine full-3D ICA / 3D-TICA radiation outputs
                q_rad_sub[:, t_sfc] = _f2s(q_rad_t, n_tr)
                sw1_dir_sub[t_sfc] = _f2s(sw1d_t, n_tr)
                sw1_dif_sub[t_sfc] = _f2s(sw1f_t, n_tr)
                sw2_dir_sub[t_sfc] = _f2s(sw2d_t, n_tr)
                sw2_dif_sub[t_sfc] = _f2s(sw2f_t, n_tr)
                lw_dn_sub[t_sfc] = _f2s(lwd_t, n_tr)
                olr_sub[t_sfc] = _f2s(olr_t, n_tr)
                asr_sub[t_sfc] = _f2s(asr_t, n_tr)
                model.rad_distill.last_uncertainty[t_pool_idx] = 0.0
                model.rad_distill.teacher_frac_last = float(n_tr) / float(max(nc, 1))
                model.rad_distill.steps_seen += 1
            else:
                q_rad_sub[:, t_sfc] = _f2s(q_rad_t, n_tr)
                sw1_dir_sub[t_sfc] = _f2s(sw1d_t, n_tr)
                sw1_dif_sub[t_sfc] = _f2s(sw1f_t, n_tr)
                sw2_dir_sub[t_sfc] = _f2s(sw2d_t, n_tr)
                sw2_dif_sub[t_sfc] = _f2s(sw2f_t, n_tr)
                lw_dn_sub[t_sfc] = _f2s(lwd_t, n_tr)
                olr_sub[t_sfc] = _f2s(olr_t, n_tr)
                asr_sub[t_sfc] = _f2s(asr_t, n_tr)

                rho_s = rho_crm_flat[:, s_sfc]
                thm_s = thm_crm_flat[:, s_sfc]
                qv_s = qv_crm_flat[:, s_sfc]
                qc_s = qc_crm_flat[:, s_sfc]
                qi_s = qi_crm_flat[:, s_sfc]
                pres_s, _, temp_s = diagnose_pressure_temperature(rho_s, thm_s, qv_s)
                dz_s = _cell_to_sub(atmos.dz_c[:, s_pool_idx], n_sr, n_sub)
                qd_s = _cell_to_sub(q_dust_cell[:, s_pool_idx], n_sr, n_sub) if q_dust_cell is not None else None
                ao_s = tuple(_cell_to_sub(a[:, s_pool_idx], n_sr, n_sub) for a in aero_opt_cell) if aero_opt_cell is not None else None
                if o3_prog_cell is not None:
                    o3_s = _cell_to_sub(o3_prog_cell[:, s_pool_idx], n_sr, n_sub)
                elif o3_use_provider:
                    o3_s = model.o3_provider.compute_vmr(
                        pres_s, lat_rad=_cell_to_sub(mesh.lat_cell[s_pool_idx], n_sr, n_sub), doy=doy_now_rad
                    )
                else:
                    o3_s = None
                pres_sfc_s = pres_s[0] + C.GRAVITY * rho_s[0] * (1.0 + qv_s[0]) * dz_s[0]
                pres_w_s = torch.empty((nz + 1, n_sr * n_sub), device=dev_r, dtype=pres_s.dtype)
                pres_w_s[0] = pres_sfc_s
                pres_w_s[1:-1] = torch.sqrt(torch.clamp(pres_s[:-1] * pres_s[1:], min=1.0))
                pres_w_s[-1] = torch.clamp(0.5 * pres_s[-1], min=0.1)
                temp_w_s = torch.empty((nz + 1, n_sr * n_sub), device=dev_r, dtype=temp_s.dtype)
                temp_w_s[0] = temp_s[0]
                temp_w_s[1:-1] = 0.5 * (temp_s[:-1] + temp_s[1:])
                temp_w_s[-1] = temp_s[-1]
                qr_s, s1d_s, s1f_s, s2d_s, s2f_s, lwd_s, olr_s, asr_s = (
                    _compute_four_band_radiation_chunked(
                        pres_c=pres_s, pres_w=pres_w_s, temp_c=temp_s, temp_w=temp_w_s,
                        dz_c=dz_s, rho_d_c=rho_s, qv_c=qv_s, qc_c=qc_s, qi_c=qi_s,
                        t_sfc=t_rad_in[s_sfc], albedo_vis=alb_vd[s_sfc], albedo_nir=alb_nd[s_sfc],
                        mu0=mu0_sub[s_sfc], s_toa=s_toa_sub[s_sfc], co2_ppmv=co2_rad_now, emiss_sfc=emiss_in[s_sfc],
                        albedo_vis_dif=alb_vf[s_sfc], albedo_nir_dif=alb_nf[s_sfc], col_chunk=cfg.rad_col_chunk,
                        re_liq_um=cfg.re_liq_um, re_ice_um=cfg.re_ice_um, cloud_sw_inhomog=cfg.cloud_sw_inhomog,
                        q_dust_c=qd_s, o3_vmr_c=o3_s, o3_lw_coupling=cfg.o3_lw_coupling,
                        rad_scheme=cfg.rad_scheme, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now, aero_opt=ao_s,
                    )
                )
                q_rad_sub[:, s_sfc] = qr_s
                sw1_dir_sub[s_sfc] = s1d_s
                sw1_dif_sub[s_sfc] = s1f_s
                sw2_dir_sub[s_sfc] = s2d_s
                sw2_dif_sub[s_sfc] = s2f_s
                lw_dn_sub[s_sfc] = lwd_s
                olr_sub[s_sfc] = olr_s
                asr_sub[s_sfc] = asr_s
        elif rad_sel is not None and model.rad_distill is not None:
            t_ridx, s_ridx = rad_sel
            n_tr = int(t_ridx.numel())
            n_sr = int(s_ridx.numel())
            dev_r = rho_crm_flat.device
            crm_off = torch.arange(n_crm, device=dev_r, dtype=torch.int64).unsqueeze(0)
            sfc_off = torch.arange(n_sub, device=dev_r, dtype=torch.int64).unsqueeze(0)
            t_crm = (t_ridx.unsqueeze(1) * n_crm + crm_off).reshape(-1)
            s_crm = (s_ridx.unsqueeze(1) * n_crm + crm_off).reshape(-1)
            t_sfc = (t_ridx.unsqueeze(1) * n_sub + sfc_off).reshape(-1)
            s_sfc = (s_ridx.unsqueeze(1) * n_sub + sfc_off).reshape(-1)

            # 3a. Teacher cells: 100% genuine full-n_crm ICA radiation (resolving full RCE_large self-aggregation!)
            rho_t = rho_crm_flat[:, t_crm]
            thm_t = thm_crm_flat[:, t_crm]
            qv_t = qv_crm_flat[:, t_crm]
            qc_t = qc_crm_flat[:, t_crm]
            qi_t = qi_crm_flat[:, t_crm]
            pres_t, _, temp_t = diagnose_pressure_temperature(rho_t, thm_t, qv_t)
            dz_t = _cell_to_sub(atmos.dz_c[:, t_ridx], n_tr, n_crm)
            pres_sfc_t = pres_t[0] + C.GRAVITY * rho_t[0] * (1.0 + qv_t[0]) * dz_t[0]
            pres_w_t = torch.empty((nz + 1, n_tr * n_crm), device=dev_r, dtype=pres_t.dtype)
            pres_w_t[0] = pres_sfc_t
            pres_w_t[1:-1] = torch.sqrt(torch.clamp(pres_t[:-1] * pres_t[1:], min=1.0))
            pres_w_t[-1] = torch.clamp(0.5 * pres_t[-1], min=0.1)
            temp_w_t = torch.empty((nz + 1, n_tr * n_crm), device=dev_r, dtype=temp_t.dtype)
            temp_w_t[0] = temp_t[0]
            temp_w_t[1:-1] = 0.5 * (temp_t[:-1] + temp_t[1:])
            temp_w_t[-1] = temp_t[-1]

            if bool(cfg.enable_rad_3d_tica) and n_crm > 1:
                from aoba.radiation_hd import apply_3d_tica_cloud_shift

                qc_t, qi_t = apply_3d_tica_cloud_shift(
                    qc_t, qi_t, z_rel_c[:, t_ridx], mu0[t_ridx], hour_angle_cell[t_ridx],
                    n_tr, ny, nx, float(cfg.crm_dx),
                )
            if use_2m_re:
                re_l_t, re_i_t = diagnose_cloud_effective_radii(
                    rho_t, qc_t, qi_t,
                    crm.n_c[:, t_ridx].reshape(nz, n_tr * n_crm),
                    crm.n_i[:, t_ridx].reshape(nz, n_tr * n_crm),
                    re_liq_fallback=cfg.re_liq_um, re_ice_fallback=cfg.re_ice_um,
                )
            else:
                re_l_t, re_i_t = cfg.re_liq_um, cfg.re_ice_um

            t_rad_t = _s2c(t_rad_in[t_sfc], n_tr)
            emiss_t = _s2c(emiss_in[t_sfc], n_tr)
            alb_vd_t = _s2c(alb_vd[t_sfc], n_tr)
            alb_vf_t = _s2c(alb_vf[t_sfc], n_tr)
            alb_nd_t = _s2c(alb_nd[t_sfc], n_tr)
            alb_nf_t = _s2c(alb_nf[t_sfc], n_tr)
            mu0_t = _cell_to_sub(mu0[t_ridx], n_tr, n_crm)
            s_toa_t = _cell_to_sub(s_toa[t_ridx], n_tr, n_crm)
            qd_t = _cell_to_sub(q_dust_cell[:, t_ridx], n_tr, n_crm) if q_dust_cell is not None else None
            ao_t = tuple(_cell_to_sub(a[:, t_ridx], n_tr, n_crm) for a in aero_opt_cell) if aero_opt_cell is not None else None
            if o3_prog_cell is not None:
                o3_t = _cell_to_sub(o3_prog_cell[:, t_ridx], n_tr, n_crm)
            elif o3_use_provider:
                o3_t = model.o3_provider.compute_vmr(
                    pres_t, lat_rad=_cell_to_sub(mesh.lat_cell[t_ridx], n_tr, n_crm), doy=doy_now_rad
                )
            else:
                o3_t = None

            q_rad_t, sw1d_t, sw1f_t, sw2d_t, sw2f_t, lwd_t, olr_t, asr_t = (
                _compute_four_band_radiation_chunked(
                    pres_c=pres_t, pres_w=pres_w_t, temp_c=temp_t, temp_w=temp_w_t,
                    dz_c=dz_t, rho_d_c=rho_t, qv_c=qv_t, qc_c=qc_t, qi_c=qi_t,
                    t_sfc=t_rad_t, albedo_vis=alb_vd_t, albedo_nir=alb_nd_t,
                    mu0=mu0_t, s_toa=s_toa_t, co2_ppmv=co2_rad_now, emiss_sfc=emiss_t,
                    albedo_vis_dif=alb_vf_t, albedo_nir_dif=alb_nf_t, col_chunk=cfg.rad_col_chunk,
                    re_liq_um=re_l_t, re_ice_um=re_i_t, cloud_sw_inhomog=cfg.cloud_sw_inhomog,
                    q_dust_c=qd_t, o3_vmr_c=o3_t, o3_lw_coupling=cfg.o3_lw_coupling,
                    rad_scheme=cfg.rad_scheme, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now, aero_opt=ao_t,
                )
            )
            # Train radiation student online from stratified teacher subcolumns (capped at 6144 for speed)
            m_tr_tot = n_tr * n_crm
            if m_tr_tot > 6144:
                tr_s = torch.randperm(m_tr_tot, device=dev_r)[:6144]
            else:
                tr_s = slice(None)
            x_rt, cp_rt_raw, s_inc_rt = extract_rad_distill_features(
                pres_t[:, tr_s], temp_t[:, tr_s], dz_t[:, tr_s], rho_t[:, tr_s],
                qv_t[:, tr_s], qc_t[:, tr_s], qi_t[:, tr_s],
                t_rad_t[tr_s], alb_vd_t[tr_s], alb_vf_t[tr_s], alb_nd_t[tr_s], alb_nf_t[tr_s],
                emiss_t[tr_s], mu0_t[tr_s], s_toa_t[tr_s],
                co2_ppmv=co2_rad_now, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now,
                o3_vmr_c=(o3_t[:, tr_s] if o3_t is not None else None),
                q_dust_c=(qd_t[:, tr_s] if qd_t is not None else None),
                tau_aero_sw_c=(ao_t[0][:, tr_s] if ao_t is not None else None),
            )
            cp_rt, cs_rt, tqr_rt, ts_rt = pack_rad_teacher_context_and_targets(
                cp_rt_raw, s_inc_rt,
                alb_vd_t[tr_s], alb_vf_t[tr_s], alb_nd_t[tr_s], alb_nf_t[tr_s], emiss_t[tr_s], t_rad_t[tr_s],
                q_rad_t[:, tr_s], sw1d_t[tr_s], sw1f_t[tr_s], sw2d_t[tr_s], sw2f_t[tr_s],
                lwd_t[tr_s], olr_t[tr_s], asr_t[tr_s],
            )
            if bool(cfg.enable_distill_shadow_diag):
                land_mask_rt = _s2c((sfc_sub.surf_type[t_sfc] == SURF_LAND).to(t_rad_t.dtype), n_tr)[tr_s]
                t_cell_sub_rt = t_ridx.unsqueeze(1).expand(n_tr, n_crm).reshape(-1)[tr_s]
                evaluate_rad_shadow_on_teachers(
                    model.rad_distill, x_rt, cp_rt_raw, s_inc_rt,
                    alb_vd_t[tr_s], alb_vf_t[tr_s], alb_nd_t[tr_s], alb_nf_t[tr_s],
                    emiss_t[tr_s], t_rad_t[tr_s],
                    q_rad_t=q_rad_t[:, tr_s], sw1d_t=sw1d_t[tr_s], sw1f_t=sw1f_t[tr_s],
                    sw2d_t=sw2d_t[tr_s], sw2f_t=sw2f_t[tr_s], lwd_t=lwd_t[tr_s],
                    olr_t=olr_t[tr_s], asr_t=asr_t[tr_s],
                    land_mask_sub=land_mask_rt,
                    t_cell_idx_sub=t_cell_sub_rt,
                    step_count=model.step_count,
                )
            n_rgrad_d = 4 if model.rad_distill.steps_seen < model.rad_distill.warmup_steps else 2
            train_rad_student_online(
                model.rad_distill, x_rt, cp_rt, cs_rt, tqr_rt, ts_rt, n_grad_steps=n_rgrad_d
            )

            # 3b. Student cells: fast per-surface-tile (`n_sub`) RadSelfDistillNet inference
            if n_sub == n_crm:
                # Chunked zero-gather/zero-scatter contiguous path for 1:1 MAML+ICA grids
                q_rad_sub = torch.empty((nz, m_crm), device=dev_r, dtype=rho_crm_flat.dtype)
                sw1_dir_sub = torch.empty_like(t_rad_in)
                sw1_dif_sub = torch.empty_like(t_rad_in)
                sw2_dir_sub = torch.empty_like(t_rad_in)
                sw2_dif_sub = torch.empty_like(t_rad_in)
                lw_dn_sub = torch.empty_like(t_rad_in)
                olr_sub = torch.empty_like(t_rad_in)
                asr_sub = torch.empty_like(t_rad_in)
                chunk_cells = max(1, 32768 // max(n_sub, 1))
                # Balance chunk size slightly upward when tail is tiny (e.g. G6 40962 -> 10x4097 instead of 10x4096+2)
                n_chunks_r = max(1, (nc + chunk_cells - 1) // chunk_cells)
                if n_chunks_r > 1 and (nc % chunk_cells) <= max(64, chunk_cells // 16):
                    chunk_cells = (nc + n_chunks_r - 2) // (n_chunks_r - 1)
                for c0 in range(0, nc, chunk_cells):
                    c1 = min(c0 + chunk_cells, nc)
                    n_ch = c1 - c0
                    s0 = c0 * n_sub
                    s1 = c1 * n_sub
                    rho_c = rho_crm_flat[:, s0:s1].contiguous()
                    thm_c = thm_crm_flat[:, s0:s1].contiguous()
                    qv_c = qv_crm_flat[:, s0:s1].contiguous()
                    qc_c = qc_crm_flat[:, s0:s1].contiguous()
                    qi_c = qi_crm_flat[:, s0:s1].contiguous()
                    pres_c, _, temp_c = diagnose_pressure_temperature(rho_c, thm_c, qv_c)
                    dz_c_sl = atmos.dz_c[:, c0:c1]
                    qd_c_sl = q_dust_cell[:, c0:c1] if q_dust_cell is not None else None
                    tau_c_sl = aero_opt_cell[0][:, c0:c1] if aero_opt_cell is not None else None
                    if o3_prog_cell is not None:
                        o3_c_sl = o3_prog_cell[:, c0:c1]
                    elif o3_use_provider:
                        o3_c_sl = model.o3_provider.compute_vmr(
                            pres_c, lat_rad=_cell_to_sub(mesh.lat_cell[c0:c1], n_ch, n_sub), doy=doy_now_rad
                        )
                    else:
                        o3_c_sl = None
                    x_rc, cp_rc, s_inc_rc = extract_rad_distill_features(
                        pres_c, temp_c, dz_c_sl, rho_c, qv_c, qc_c, qi_c,
                        t_rad_in[s0:s1], alb_vd[s0:s1], alb_vf[s0:s1], alb_nd[s0:s1], alb_nf[s0:s1],
                        emiss_in[s0:s1], mu0_sub[s0:s1], s_toa_sub[s0:s1],
                        co2_ppmv=co2_rad_now, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now,
                        o3_vmr_c=o3_c_sl, q_dust_c=qd_c_sl, tau_aero_sw_c=tau_c_sl,
                        cell_sub=n_sub,
                    )
                    rad_stu_out, rad_stu_unc = predict_rad_student_columns(
                        model.rad_distill, x_rc, cp_rc, s_inc_rc,
                        alb_vd[s0:s1], alb_vf[s0:s1], alb_nd[s0:s1], alb_nf[s0:s1],
                        emiss_in[s0:s1], t_rad_in[s0:s1],
                    )
                    q_rad_sub[:, s0:s1] = rad_stu_out["q_rad"]
                    sw1_dir_sub[s0:s1] = rad_stu_out["sw1_dir"]
                    sw1_dif_sub[s0:s1] = rad_stu_out["sw1_dif"]
                    sw2_dir_sub[s0:s1] = rad_stu_out["sw2_dir"]
                    sw2_dif_sub[s0:s1] = rad_stu_out["sw2_dif"]
                    lw_dn_sub[s0:s1] = rad_stu_out["lw_dn"]
                    olr_sub[s0:s1] = rad_stu_out["olr"]
                    asr_sub[s0:s1] = rad_stu_out["asr"]
                    model.rad_distill.last_uncertainty[c0:c1] = rad_stu_unc.reshape(n_ch, n_sub).mean(dim=-1)

                q_rad_sub.index_copy_(1, t_crm, q_rad_t)
                sw1_dir_sub.index_copy_(0, t_crm, sw1d_t)
                sw1_dif_sub.index_copy_(0, t_crm, sw1f_t)
                sw2_dir_sub.index_copy_(0, t_crm, sw2d_t)
                sw2_dif_sub.index_copy_(0, t_crm, sw2f_t)
                lw_dn_sub.index_copy_(0, t_crm, lwd_t)
                olr_sub.index_copy_(0, t_crm, olr_t)
                asr_sub.index_copy_(0, t_crm, asr_t)

                model.rad_distill.last_uncertainty[t_ridx] = 0.0
                model.rad_distill.teacher_frac_last = float(n_tr) / float(max(nc, 1))
                model.rad_distill.steps_seen += 1
            else:
                rho_s = _c2s(rho_crm_flat[:, s_crm], n_sr)
                thm_s = _c2s(thm_crm_flat[:, s_crm], n_sr)
                qv_s = _c2s(qv_crm_flat[:, s_crm], n_sr)
                qc_s = _c2s(qc_crm_flat[:, s_crm], n_sr)
                qi_s = _c2s(qi_crm_flat[:, s_crm], n_sr)
                pres_s, _, temp_s = diagnose_pressure_temperature(rho_s, thm_s, qv_s)
                dz_s = _cell_to_sub(atmos.dz_c[:, s_ridx], n_sr, n_sub)
                qd_s = _cell_to_sub(q_dust_cell[:, s_ridx], n_sr, n_sub) if q_dust_cell is not None else None
                tau_aero_s = _cell_to_sub(aero_opt_cell[0][:, s_ridx], n_sr, n_sub) if aero_opt_cell is not None else None
                if o3_prog_cell is not None:
                    o3_s = _cell_to_sub(o3_prog_cell[:, s_ridx], n_sr, n_sub)
                elif o3_use_provider:
                    o3_s = model.o3_provider.compute_vmr(
                        pres_s, lat_rad=_cell_to_sub(mesh.lat_cell[s_ridx], n_sr, n_sub), doy=doy_now_rad
                    )
                else:
                    o3_s = None
                x_rs, cp_rs, s_inc_rs = extract_rad_distill_features(
                    pres_s, temp_s, dz_s, rho_s, qv_s, qc_s, qi_s,
                    t_rad_in[s_sfc], alb_vd[s_sfc], alb_vf[s_sfc], alb_nd[s_sfc], alb_nf[s_sfc],
                    emiss_in[s_sfc], mu0_sub[s_sfc], s_toa_sub[s_sfc],
                    co2_ppmv=co2_rad_now, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now,
                    o3_vmr_c=o3_s, q_dust_c=qd_s, tau_aero_sw_c=tau_aero_s,
                )
                rad_stu_out, rad_stu_unc = predict_rad_student_columns(
                    model.rad_distill, x_rs, cp_rs, s_inc_rs,
                    alb_vd[s_sfc], alb_vf[s_sfc], alb_nd[s_sfc], alb_nf[s_sfc], emiss_in[s_sfc], t_rad_in[s_sfc],
                )

                q_rad_sub = torch.empty((nz, m_crm), device=dev_r, dtype=rho_crm_flat.dtype)
                sw1_dir_sub = torch.empty_like(t_rad_in)
                sw1_dif_sub = torch.empty_like(t_rad_in)
                sw2_dir_sub = torch.empty_like(t_rad_in)
                sw2_dif_sub = torch.empty_like(t_rad_in)
                lw_dn_sub = torch.empty_like(t_rad_in)
                olr_sub = torch.empty_like(t_rad_in)
                asr_sub = torch.empty_like(t_rad_in)

                q_rad_sub[:, t_crm] = q_rad_t
                sw1_dir_sub[t_sfc] = _c2s(sw1d_t, n_tr)
                sw1_dif_sub[t_sfc] = _c2s(sw1f_t, n_tr)
                sw2_dir_sub[t_sfc] = _c2s(sw2d_t, n_tr)
                sw2_dif_sub[t_sfc] = _c2s(sw2f_t, n_tr)
                lw_dn_sub[t_sfc] = _c2s(lwd_t, n_tr)
                olr_sub[t_sfc] = _c2s(olr_t, n_tr)
                asr_sub[t_sfc] = _c2s(asr_t, n_tr)

                q_rad_sub[:, s_crm] = _s2c(rad_stu_out["q_rad"], n_sr)
                sw1_dir_sub[s_sfc] = rad_stu_out["sw1_dir"]
                sw1_dif_sub[s_sfc] = rad_stu_out["sw1_dif"]
                sw2_dir_sub[s_sfc] = rad_stu_out["sw2_dir"]
                sw2_dif_sub[s_sfc] = rad_stu_out["sw2_dif"]
                lw_dn_sub[s_sfc] = rad_stu_out["lw_dn"]
                olr_sub[s_sfc] = rad_stu_out["olr"]
                asr_sub[s_sfc] = rad_stu_out["asr"]

                model.rad_distill.last_uncertainty.zero_()
                model.rad_distill.last_uncertainty[s_ridx] = rad_stu_unc.reshape(n_sr, n_sub).mean(dim=-1)
                model.rad_distill.teacher_frac_last = float(n_tr) / float(max(nc, 1))
                model.rad_distill.steps_seen += 1
        else:
            # Full-physics / warmup ICA radiation across `[nz, m_crm]`
            pres_crm, _, temp_crm = diagnose_pressure_temperature(rho_crm_flat, thm_crm_flat, qv_crm_flat)
            dz_rad_crm = _cell_to_sub(atmos.dz_c, nc, n_crm)
            pres_sfc_crm = pres_crm[0] + C.GRAVITY * rho_crm_flat[0] * (1.0 + qv_crm_flat[0]) * dz_rad_crm[0]
            pres_w_crm = torch.empty((nz + 1, m_crm), device=pres_crm.device, dtype=pres_crm.dtype)
            pres_w_crm[0] = pres_sfc_crm
            pres_w_crm[1:-1] = torch.sqrt(torch.clamp(pres_crm[:-1] * pres_crm[1:], min=1.0))
            pres_w_crm[-1] = torch.clamp(0.5 * pres_crm[-1], min=0.1)
            temp_w_crm = torch.empty((nz + 1, m_crm), device=temp_crm.device, dtype=temp_crm.dtype)
            temp_w_crm[0] = temp_crm[0]
            temp_w_crm[1:-1] = 0.5 * (temp_crm[:-1] + temp_crm[1:])
            temp_w_crm[-1] = temp_crm[-1]

            qc_rad, qi_rad = qc_crm_flat, qi_crm_flat
            if bool(cfg.enable_rad_3d_tica) and n_crm > 1:
                from aoba.radiation_hd import apply_3d_tica_cloud_shift

                qc_rad, qi_rad = apply_3d_tica_cloud_shift(
                    qc_crm_flat, qi_crm_flat, z_rel_c, mu0, hour_angle_cell, nc, ny, nx, float(cfg.crm_dx)
                )
            if use_2m_re:
                re_liq_rad, re_ice_rad = diagnose_cloud_effective_radii(
                    rho_crm_flat, qc_crm_flat, qi_crm_flat,
                    crm.n_c.reshape(nz, m_crm), crm.n_i.reshape(nz, m_crm),
                    re_liq_fallback=cfg.re_liq_um, re_ice_fallback=cfg.re_ice_um,
                )
            else:
                re_liq_rad, re_ice_rad = cfg.re_liq_um, cfg.re_ice_um

            t_rad_crm = _s2c(t_rad_in)
            emiss_crm = _s2c(emiss_in)
            alb_vd_crm = _s2c(alb_vd)
            alb_vf_crm = _s2c(alb_vf)
            alb_nd_crm = _s2c(alb_nd)
            alb_nf_crm = _s2c(alb_nf)
            mu0_crm = _cell_to_sub(mu0, nc, n_crm)
            s_toa_crm = _cell_to_sub(s_toa, nc, n_crm)
            q_dust_rad = _cell_to_sub(q_dust_cell, nc, n_crm) if q_dust_cell is not None else None
            aero_opt_rad = tuple(_cell_to_sub(a, nc, n_crm) for a in aero_opt_cell) if aero_opt_cell is not None else None
            if o3_prog_cell is not None:
                o3_vmr_rad = _cell_to_sub(o3_prog_cell, nc, n_crm)
            elif o3_use_provider:
                o3_vmr_rad = model.o3_provider.compute_vmr(
                    pres_crm, lat_rad=_cell_to_sub(mesh.lat_cell, nc, n_crm), doy=doy_now_rad
                )
            else:
                o3_vmr_rad = None

            q_rad_sub, sw1d_c, sw1f_c, sw2d_c, sw2f_c, lwd_c, olr_c, asr_c = (
                _compute_four_band_radiation_chunked(
                    pres_c=pres_crm, pres_w=pres_w_crm, temp_c=temp_crm, temp_w=temp_w_crm,
                    dz_c=dz_rad_crm, rho_d_c=rho_crm_flat, qv_c=qv_crm_flat, qc_c=qc_rad, qi_c=qi_rad,
                    t_sfc=t_rad_crm, albedo_vis=alb_vd_crm, albedo_nir=alb_nd_crm,
                    mu0=mu0_crm, s_toa=s_toa_crm, co2_ppmv=co2_rad_now, emiss_sfc=emiss_crm,
                    albedo_vis_dif=alb_vf_crm, albedo_nir_dif=alb_nf_crm, col_chunk=cfg.rad_col_chunk,
                    re_liq_um=re_liq_rad, re_ice_um=re_ice_rad, cloud_sw_inhomog=cfg.cloud_sw_inhomog,
                    q_dust_c=q_dust_rad, o3_vmr_c=o3_vmr_rad, o3_lw_coupling=cfg.o3_lw_coupling,
                    rad_scheme=cfg.rad_scheme, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now, aero_opt=aero_opt_rad,
                )
            )
            if use_rad_distill and model.rad_distill is not None:
                n_rsamp = min(m_crm, 4096)
                rw_idx = (
                    torch.randperm(m_crm, device=pres_crm.device)[:n_rsamp]
                    if n_rsamp < m_crm
                    else torch.arange(m_crm, device=pres_crm.device)
                )
                x_rw, cp_rw_raw, s_inc_rw = extract_rad_distill_features(
                    pres_crm[:, rw_idx], temp_crm[:, rw_idx], dz_rad_crm[:, rw_idx], rho_crm_flat[:, rw_idx],
                    qv_crm_flat[:, rw_idx], qc_rad[:, rw_idx], qi_rad[:, rw_idx],
                    t_rad_crm[rw_idx], alb_vd_crm[rw_idx], alb_vf_crm[rw_idx], alb_nd_crm[rw_idx], alb_nf_crm[rw_idx],
                    emiss_crm[rw_idx], mu0_crm[rw_idx], s_toa_crm[rw_idx],
                    co2_ppmv=co2_rad_now, ch4_ppmv=ch4_rad_now, n2o_ppmv=n2o_rad_now,
                    o3_vmr_c=(o3_vmr_rad[:, rw_idx] if o3_vmr_rad is not None else None),
                    q_dust_c=(q_dust_rad[:, rw_idx] if q_dust_rad is not None else None),
                    tau_aero_sw_c=(aero_opt_rad[0][:, rw_idx] if aero_opt_rad is not None else None),
                )
                cp_rw, cs_rw, tqr_rw, ts_rw = pack_rad_teacher_context_and_targets(
                    cp_rw_raw, s_inc_rw,
                    alb_vd_crm[rw_idx], alb_vf_crm[rw_idx], alb_nd_crm[rw_idx], alb_nf_crm[rw_idx],
                    emiss_crm[rw_idx], t_rad_crm[rw_idx],
                    q_rad_sub[:, rw_idx], sw1d_c[rw_idx], sw1f_c[rw_idx],
                    sw2d_c[rw_idx], sw2f_c[rw_idx], lwd_c[rw_idx], olr_c[rw_idx], asr_c[rw_idx],
                )
                if bool(cfg.enable_distill_shadow_diag):
                    land_mask_rw = _s2c((sfc_sub.surf_type == SURF_LAND).to(t_rad_crm.dtype))[rw_idx]
                    t_cell_sub_rw = torch.div(rw_idx, max(n_crm, 1), rounding_mode="floor")
                    evaluate_rad_shadow_on_teachers(
                        model.rad_distill, x_rw, cp_rw_raw, s_inc_rw,
                        alb_vd_crm[rw_idx], alb_vf_crm[rw_idx], alb_nd_crm[rw_idx], alb_nf_crm[rw_idx],
                        emiss_crm[rw_idx], t_rad_crm[rw_idx],
                        q_rad_t=q_rad_sub[:, rw_idx], sw1d_t=sw1d_c[rw_idx], sw1f_t=sw1f_c[rw_idx],
                        sw2d_t=sw2d_c[rw_idx], sw2f_t=sw2f_c[rw_idx], lwd_t=lwd_c[rw_idx],
                        olr_t=olr_c[rw_idx], asr_t=asr_c[rw_idx],
                        land_mask_sub=land_mask_rw,
                        t_cell_idx_sub=t_cell_sub_rw,
                        step_count=model.step_count,
                    )
                train_rad_student_online(model.rad_distill, x_rw, cp_rw, cs_rw, tqr_rw, ts_rw)
                model.rad_distill.teacher_frac_last = 1.0
                model.rad_distill.steps_seen += 1

            sw1_dir_sub = _c2s(sw1d_c)
            sw1_dif_sub = _c2s(sw1f_c)
            sw2_dir_sub = _c2s(sw2d_c)
            sw2_dif_sub = _c2s(sw2f_c)
            lw_dn_sub = _c2s(lwd_c)
            olr_sub = _c2s(olr_c)
            asr_sub = _c2s(asr_c)
        model.rad_cache["q_rad_sub"] = q_rad_sub
        model.rad_cache["sw1_dir_sub"] = sw1_dir_sub
        model.rad_cache["sw1_dif_sub"] = sw1_dif_sub
        model.rad_cache["sw2_dir_sub"] = sw2_dir_sub
        model.rad_cache["sw2_dif_sub"] = sw2_dif_sub
        model.rad_cache["lw_dn_sub"] = lw_dn_sub
        model.rad_cache["olr_sub"] = olr_sub
        model.rad_cache["asr_sub"] = asr_sub
        # Host cell-mean convenience diagnostics
        model.rad_cache["sw1_dir"] = _sub_to_cell(sw1_dir_sub, nc, n_sub)
        model.rad_cache["sw1_dif"] = _sub_to_cell(sw1_dif_sub, nc, n_sub)
        model.rad_cache["sw2_dir"] = _sub_to_cell(sw2_dir_sub, nc, n_sub)
        model.rad_cache["sw2_dif"] = _sub_to_cell(sw2_dif_sub, nc, n_sub)
        model.rad_cache["lw_dn"] = _sub_to_cell(lw_dn_sub, nc, n_sub)
        model.rad_cache["olr"] = _sub_to_cell(olr_sub, nc, n_sub)
        model.rad_cache["asr"] = _sub_to_cell(asr_sub, nc, n_sub)

    q_rad_sub = model.rad_cache["q_rad_sub"]
    sw1_dir_sub = model.rad_cache["sw1_dir_sub"]
    sw1_dif_sub = model.rad_cache["sw1_dif_sub"]
    sw2_dir_sub = model.rad_cache["sw2_dir_sub"]
    sw2_dif_sub = model.rad_cache["sw2_dif_sub"]
    lw_dn_sub = model.rad_cache["lw_dn_sub"]
    olr = model.rad_cache["olr"]
    asr = model.rad_cache["asr"]

    rain_prev_sub = model.rad_cache.get("rain_sub", torch.zeros_like(pres_sfc_sub))
    snow_prev_sub = model.rad_cache.get("snow_sub", torch.zeros_like(pres_sfc_sub))

    # Snapshot pre-surface ocean ML/ice state to extract exact atmospheric surface tendencies when ocean_glevel != glevel
    has_indep_ocn = model.ocn_coupler is not None and model.ocn_mesh is not None and model.ocn_sfc is not None
    if has_indep_ocn:
        t_ml_pre_sfc = _sub_to_cell(sfc_sub.t_ocean_ml, nc, n_sub)
        m_ice_pre_sfc = _sub_to_cell(sfc_sub.m_seaice, nc, n_sub)
        swe_ice_pre_sfc = _sub_to_cell(sfc_sub.swe_seaice, nc, n_sub)

    # Ocean-current relative air-sea wind (Renault et al., 2016, 2019; Pacanowski, 1987):
    # Strictly active ONLY when CRM is 3D (crm_ny > 1 and crm_nx > 1) OR superparameterization is disabled!
    use_rel_wind = (
        bool(cfg.enable_current_rel_wind)
        and model.ogcm is not None
        and ((not bool(cfg.enable_superparam)) or (ny_full > 1 and nx_full > 1))
    )
    u_ocn_sub = torch.zeros_like(u_sub_0)
    v_ocn_sub = torch.zeros_like(v_sub_0)
    if use_rel_wind:
        assert model.ogcm is not None
        ocn_m_rw = model.ocn_mesh if has_indep_ocn else mesh
        assert ocn_m_rw is not None
        uo0, vo0 = reconstruct_cell_winds(ocn_m_rw, (model.ogcm.u_edge[0:1] * model.ogcm.wet_e[0:1].to(model.ogcm.u_edge.dtype)))
        uo_c, vo_c = uo0[0], vo0[0]
        if has_indep_ocn:
            from aoba.ocean_coupler import ocn_to_atm_mean

            z_atm = torch.zeros(nc, device=uo_c.device, dtype=uo_c.dtype)
            uo_c = ocn_to_atm_mean(model.ocn_coupler, uo_c, fallback_atm=z_atm)
            vo_c = ocn_to_atm_mean(model.ocn_coupler, vo_c, fallback_atm=z_atm)
        is_ocn_rw = (sfc_sub.surf_type == SURF_OCEAN)
        u_ocn_sub = torch.where(is_ocn_rw, _cell_to_sub(uo_c, nc, n_sub), u_ocn_sub)
        v_ocn_sub = torch.where(is_ocn_rw, _cell_to_sub(vo_c, nc, n_sub), v_ocn_sub)

    # Optional MAML subgrid orographic microclimate downscaling (lapse rate + terrain aspect)
    temp_sfc_in = temp_sub_0
    pres_sfc_in = pres_sfc_sub
    sw1_dir_in = sw1_dir_sub
    sw2_dir_in = sw2_dir_sub
    if bool(cfg.enable_maml_subgrid_topo) or bool(cfg.enable_maml_wetlands_lakes):
        from aoba.maml_land_subgrid import downscale_maml_microclimate, init_maml_subgrid_land_state

        if model.maml_land is None:
            model.maml_land = init_maml_subgrid_land_state(sfc_sub.surf_type, sfc_sub.sigma_topo, nc, n_sub)
        if bool(cfg.enable_maml_subgrid_topo):
            temp_sfc_in, pres_sfc_in, sw1_dir_in = downscale_maml_microclimate(
                model.maml_land, temp_sub_0, pres_sfc_sub, sw1_dir_sub
            )
            sw2_dir_in = torch.clamp(sw2_dir_sub * model.maml_land.slope_sw_fac, min=0.0)

    # 4. Subgrid MAML Land/Snow/Soil/Two-Leaf Biosphere & Ocean/Sea-Ice Balance + Host Voronoi Ekman Ocean
    sfc_diag_sub = step_surface(
        sfc=sfc_sub,
        u_1=u_sub_0 - u_ocn_sub,
        v_1=v_sub_0 - v_ocn_sub,
        temp_1=temp_sfc_in,
        qv_1=qv_sub_0,
        pres_1=pres_sub_0,
        pres_sfc=pres_sfc_in,
        z_ref_rel=z_ref_rel_sub,
        sw1_dir_down=sw1_dir_in,
        sw1_dif_down=sw1_dif_sub,
        sw2_dir_down=sw2_dir_in,
        sw2_dif_down=sw2_dif_sub,
        lw_down=lw_dn_sub,
        precip_rain=rain_prev_sub,
        precip_snow=snow_prev_sub,
        mu0=mu0_sub,
        co2_ppmv=model.current_co2_ppmv(),
        dt=dt,
        precomputed_albedos=_pre_alb_sub,
    )

    # Optional dynamic MAML wetland/floodplain/lake evaporation boost & coupled wave/sea-spray fluxes
    if bool(cfg.enable_maml_wetlands_lakes) and model.maml_land is not None:
        from aoba.maml_land_subgrid import step_maml_wetlands_lakes

        q_riv_c = getattr(model.river, "discharge_m3s", None) if model.river is not None else None
        step_maml_wetlands_lakes(
            model.maml_land, sfc_sub.surf_type, sfc_sub.sigma_topo,
            sfc_sub.w_liq_soil[0], sfc_sub.t_soil[0], mesh.area_cell, nc, n_sub, dt,
            river_discharge_cell=q_riv_c,
        )
        # Inundated wetland/lake tiles have enhanced open-water evaporation
        wet_boost = 1.0 + 0.35 * model.maml_land.f_wetland
        sfc_diag_sub.evap = sfc_diag_sub.evap * wet_boost
        sfc_diag_sub.lhf = sfc_diag_sub.lhf * wet_boost

    # Optional map-free dynamic lakes (2-layer FLake + 4 C overturning + lake ice) tiled conservatively with land
    if bool(cfg.enable_dynamic_lakes) and model.lakes is not None:
        from aoba.lakes import step_dynamic_lakes

        f_lake_pre_sub = torch.clamp(_cell_to_sub(model.lakes.f_lake, nc, n_sub), 0.0, 1.0)
        step_dynamic_lakes(
            lake=model.lakes,
            u_1=_sub_to_cell(u_sub_0, nc, n_sub),
            v_1=_sub_to_cell(v_sub_0, nc, n_sub),
            temp_1=_sub_to_cell(temp_sub_0, nc, n_sub),
            qv_1=_sub_to_cell(qv_sub_0, nc, n_sub),
            pres_sfc=_sub_to_cell(pres_sfc_sub, nc, n_sub),
            rho_1=_sub_to_cell(rho_sub_0 * (1.0 + qv_sub_0), nc, n_sub),
            z_ref_rel=_sub_to_cell(z_ref_rel_sub, nc, n_sub),
            sw1_dir_dn=_sub_to_cell(sw1_dir_sub, nc, n_sub),
            sw1_dif_dn=_sub_to_cell(sw1_dif_sub, nc, n_sub),
            sw2_dir_dn=_sub_to_cell(sw2_dir_sub, nc, n_sub),
            sw2_dif_dn=_sub_to_cell(sw2_dif_sub, nc, n_sub),
            lw_dn=_sub_to_cell(lw_dn_sub, nc, n_sub),
            precip_rain=_sub_to_cell(rain_prev_sub, nc, n_sub),
            precip_snow=_sub_to_cell(snow_prev_sub, nc, n_sub),
            mu0_cell=mu0,
            area_cell=mesh.area_cell,
            f_cor_cell=mesh.f_cell,
            dt=dt,
            k_ext_m1=float(cfg.lake_extinction_m1),
        )
        omf_l = 1.0 - f_lake_pre_sub
        sfc_diag_sub.shf = omf_l * sfc_diag_sub.shf + f_lake_pre_sub * _cell_to_sub(model.lakes.shf, nc, n_sub)
        sfc_diag_sub.lhf = omf_l * sfc_diag_sub.lhf + f_lake_pre_sub * _cell_to_sub(model.lakes.lhf, nc, n_sub)
        sfc_diag_sub.evap = omf_l * sfc_diag_sub.evap + f_lake_pre_sub * _cell_to_sub(model.lakes.evap, nc, n_sub)
        sfc_diag_sub.tau_x = omf_l * sfc_diag_sub.tau_x + f_lake_pre_sub * _cell_to_sub(model.lakes.tau_x, nc, n_sub)
        sfc_diag_sub.tau_y = omf_l * sfc_diag_sub.tau_y + f_lake_pre_sub * _cell_to_sub(model.lakes.tau_y, nc, n_sub)
        sfc_diag_sub.t_rad_sfc = (
            omf_l * (torch.clamp(sfc_diag_sub.t_rad_sfc, min=150.0) ** 4)
            + f_lake_pre_sub * (torch.clamp(_cell_to_sub(model.lakes.t_rad, nc, n_sub), min=150.0) ** 4)
        ) ** 0.25
        sfc_diag_sub.alb_vis_dir = omf_l * sfc_diag_sub.alb_vis_dir + f_lake_pre_sub * _cell_to_sub(model.lakes.alb_vis_dir, nc, n_sub)
        sfc_diag_sub.alb_vis_dif = omf_l * sfc_diag_sub.alb_vis_dif + f_lake_pre_sub * _cell_to_sub(model.lakes.alb_vis_dif, nc, n_sub)
        sfc_diag_sub.alb_nir_dir = omf_l * sfc_diag_sub.alb_nir_dir + f_lake_pre_sub * _cell_to_sub(model.lakes.alb_nir_dir, nc, n_sub)
        sfc_diag_sub.alb_nir_dif = omf_l * sfc_diag_sub.alb_nir_dif + f_lake_pre_sub * _cell_to_sub(model.lakes.alb_nir_dif, nc, n_sub)

    langmuir_ocn = None
    if bool(cfg.enable_wave_coupling):
        from aoba.ocean_waves import compute_sea_spray_enthalpy_fluxes, init_ocean_wave_state, step_ocean_waves

        is_ocn_wave_sub = (sfc_sub.surf_type == SURF_OCEAN)
        open_ocn_sub = torch.clamp(1.0 - torch.clamp(sfc_diag_sub.f_seaice, 0.0, 1.0), 0.0, 1.0) * is_ocn_wave_sub.to(temp_sub_0.dtype)
        u10_rel_sub = torch.hypot(u_sub_0 - u_ocn_sub, v_sub_0 - v_ocn_sub)
        d_lh_sp, d_sh_sp = compute_sea_spray_enthalpy_fluxes(
            u10_rel_sub, sfc_sub.t_ocean_ml, temp_sub_0, qv_sub_0, pres_sfc_sub, open_ocn_sub
        )
        sfc_diag_sub.lhf = sfc_diag_sub.lhf + d_lh_sp
        sfc_diag_sub.shf = sfc_diag_sub.shf + d_sh_sp
        sfc_diag_sub.evap = sfc_diag_sub.evap + d_lh_sp / C.L_V0

        active_w_mesh = model.ocn_mesh if (str(cfg.ocean_mode).strip().lower() == "ogcm_3d" and has_indep_ocn) else mesh
        active_w_sfc = model.ocn_sfc if (str(cfg.ocean_mode).strip().lower() == "ogcm_3d" and has_indep_ocn) else sfc
        assert active_w_mesh is not None and active_w_sfc is not None
        if model.ocean_waves is None:
            model.ocean_waves = init_ocean_wave_state(active_w_mesh.n_cells, temp_sub_0.device, temp_sub_0.dtype)
        u10_w_cell = _sub_to_cell(u10_rel_sub, nc, n_sub)
        fice_w_cell = _sub_to_cell(sfc_diag_sub.f_seaice, nc, n_sub)
        if active_w_mesh is not mesh and model.ocn_coupler is not None:
            from aoba.ocean_coupler import atm_to_ocn_intensive

            u10_w_in = atm_to_ocn_intensive(model.ocn_coupler, u10_w_cell)
            fice_w_in = atm_to_ocn_intensive(model.ocn_coupler, fice_w_cell)
        else:
            u10_w_in = u10_w_cell
            fice_w_in = fice_w_cell
        step_ocean_waves(model.ocean_waves, u10_w_in, fice_w_in, active_w_sfc.surf_type == SURF_OCEAN, dt)
        langmuir_ocn = model.ocean_waves.langmuir_fac
        # Apply wave-age sea-surface drag modulation onto open-ocean subcolumns
        drag_c = model.ocean_waves.drag_mod
        if active_w_mesh is not mesh and model.ocn_coupler is not None:
            from aoba.ocean_coupler import ocn_to_atm_mean

            drag_c = ocn_to_atm_mean(model.ocn_coupler, drag_c, fallback_atm=torch.ones(nc, device=drag_c.device, dtype=drag_c.dtype))
        drag_sub = 1.0 + open_ocn_sub * (_cell_to_sub(drag_c, nc, n_sub) - 1.0)
        sfc_diag_sub.tau_x = sfc_diag_sub.tau_x * drag_sub
        sfc_diag_sub.tau_y = sfc_diag_sub.tau_y * drag_sub

    is_ocn_sub = sfc_sub.surf_type == SURF_OCEAN
    t1_cell_before = _sub_to_cell(sfc_sub.t_ocean_ml, nc, n_sub)
    m_ice_cell_before = _sub_to_cell(sfc_sub.m_seaice, nc, n_sub)
    swe_ice_cell_before = _sub_to_cell(sfc_sub.swe_seaice, nc, n_sub)
    sfc.t_ocean_ml = t1_cell_before.clone()
    sfc.m_seaice = m_ice_cell_before.clone()
    sfc.swe_seaice = swe_ice_cell_before.clone()
    sfc.h_ocean_ml = _sub_to_cell(sfc_sub.h_ocean_ml, nc, n_sub)
    if str(cfg.ocean_mode).strip().lower() == "ogcm_3d":
        active_ocn_mesh = model.ocn_mesh if has_indep_ocn else mesh
        active_ocn_sfc = model.ocn_sfc if has_indep_ocn else sfc
        assert active_ocn_mesh is not None and active_ocn_sfc is not None
        if model.ogcm is None:
            model.ogcm = init_ogcm_state(active_ocn_sfc, active_ocn_mesh, nz_ocn=cfg.ocean_nz)
        if bool(cfg.enable_ocean_bgc) and model.ogcm.bgc is None:
            from aoba.ocean_bgc import init_ocean_bgc_state

            model.ogcm.bgc = init_ocean_bgc_state(model.ogcm.wet_c, model.ogcm.h_c, model.ogcm.z_c, active_ocn_mesh.area_cell)
        if bool(cfg.enable_seaice_evp_itd) and model.ogcm.seaice_evp is None:
            from aoba.seaice_evp import init_seaice_evp_state

            model.ogcm.seaice_evp = init_seaice_evp_state(active_ocn_sfc, active_ocn_mesh)
        sw_abs_ocn_sub = (1.0 - torch.clamp(sfc_diag_sub.f_seaice, 0.0, 1.0)) * (
            sw1_dir_sub * (1.0 - sfc_diag_sub.alb_vis_dir)
            + sw1_dif_sub * (1.0 - sfc_diag_sub.alb_vis_dif)
            + sw2_dir_sub * (1.0 - sfc_diag_sub.alb_nir_dir)
            + sw2_dif_sub * (1.0 - sfc_diag_sub.alb_nir_dif)
        )
        sw_par_ocn = (
            _sub_to_cell(C.PAR_FRAC_OF_SW1 * (sw1_dir_sub * (1.0 - sfc_diag_sub.alb_vis_dir) + sw1_dif_sub * (1.0 - sfc_diag_sub.alb_vis_dif)), nc, n_sub)
            if model.ogcm.bgc is not None
            else None
        )
        u10_ocn = (
            _sub_to_cell(torch.hypot(u_sub_0, v_sub_0), nc, n_sub)
            if model.ogcm.bgc is not None
            else None
        )
        runoff_tot_cell = _sub_to_cell(sfc_diag_sub.runoff_total, nc, n_sub)
        runoff_ice_cell = _sub_to_cell(sfc_diag_sub.runoff_ice, nc, n_sub)
        river_mouth_flux = None
        if bool(cfg.enable_river_routing):
            from aoba.river_routing import init_river_routing_state, step_river_routing_voronoi

            if model.river is None:
                model.river = init_river_routing_state(mesh, device=runoff_tot_cell.device)
            runoff_liq_cell = torch.clamp(runoff_tot_cell - runoff_ice_cell, min=0.0)
            river_mouth_flux = step_river_routing_voronoi(
                model.river,
                runoff_liq_cell,
                mesh.area_cell,
                dt,
                lake_state=model.lakes if bool(cfg.enable_dynamic_lakes) else None,
            )

        tau_x_ocn = _sub_to_cell(sfc_diag_sub.tau_x, nc, n_sub)
        tau_y_ocn = _sub_to_cell(sfc_diag_sub.tau_y, nc, n_sub)
        f_ice_ocn = _sub_to_cell(sfc_diag_sub.f_seaice, nc, n_sub)
        evap_ocn = _sub_to_cell(sfc_diag_sub.evap, nc, n_sub)
        precip_ocn = _sub_to_cell(rain_prev_sub + snow_prev_sub, nc, n_sub)
        sw_net_ocn = _sub_to_cell(sw_abs_ocn_sub, nc, n_sub)
        runoff_tot_ocn = runoff_tot_cell
        runoff_ice_ocn = runoff_ice_cell
        river_mouth_ocn = river_mouth_flux

        if has_indep_ocn:
            from aoba.ocean_coupler import (
                apply_atm_surface_increments_to_ocn_sfc,
                atm_to_ocn_conservative_flux,
                atm_to_ocn_intensive,
                ocn_to_atm_mean,
            )

            coupler = model.ocn_coupler
            apply_atm_surface_increments_to_ocn_sfc(
                coupler,
                active_ocn_sfc,
                t1_cell_before - t_ml_pre_sfc,
                m_ice_cell_before - m_ice_pre_sfc,
                swe_ice_cell_before - swe_ice_pre_sfc,
                t_ml_atm_ref=t_ml_pre_sfc,
                dt=dt,
            )

        ocn_stride = max(int(getattr(cfg, "ocean_stride", 1)), 1)
        if ocn_stride > 1:
            acc = model.ocn_flux_accum
            cur_cnt = int(acc.get("count", 0))
            flux_bundle: dict[str, torch.Tensor | None] = {
                "tau_x": tau_x_ocn,
                "tau_y": tau_y_ocn,
                "f_ice": f_ice_ocn,
                "evap": evap_ocn,
                "precip": precip_ocn,
                "sw_net": sw_net_ocn,
                "runoff_tot": runoff_tot_ocn,
                "runoff_ice": runoff_ice_ocn,
                "river_mouth": river_mouth_ocn,
                "sw_par": sw_par_ocn,
                "u10": u10_ocn,
                "langmuir": langmuir_ocn,
            }
            if cur_cnt == 0:
                for fk, fv in flux_bundle.items():
                    acc[fk] = fv.clone() if fv is not None else None
            else:
                for fk, fv in flux_bundle.items():
                    if fv is not None:
                        prev_v = acc.get(fk)
                        acc[fk] = (prev_v + fv) if isinstance(prev_v, torch.Tensor) else fv.clone()
            cur_cnt += 1
            acc["count"] = cur_cnt
            do_step_ogcm = cur_cnt >= ocn_stride
            if do_step_ogcm:
                inv_cnt = 1.0 / float(cur_cnt)
                tau_x_ocn = acc["tau_x"] * inv_cnt  # type: ignore[operator]
                tau_y_ocn = acc["tau_y"] * inv_cnt  # type: ignore[operator]
                f_ice_ocn = acc["f_ice"] * inv_cnt  # type: ignore[operator]
                evap_ocn = acc["evap"] * inv_cnt    # type: ignore[operator]
                precip_ocn = acc["precip"] * inv_cnt  # type: ignore[operator]
                sw_net_ocn = acc["sw_net"] * inv_cnt  # type: ignore[operator]
                runoff_tot_ocn = acc["runoff_tot"] * inv_cnt  # type: ignore[operator]
                runoff_ice_ocn = acc["runoff_ice"] * inv_cnt  # type: ignore[operator]
                river_mouth_ocn = (acc["river_mouth"] * inv_cnt) if isinstance(acc.get("river_mouth"), torch.Tensor) else None
                sw_par_ocn = (acc["sw_par"] * inv_cnt) if isinstance(acc.get("sw_par"), torch.Tensor) else None
                u10_ocn = (acc["u10"] * inv_cnt) if isinstance(acc.get("u10"), torch.Tensor) else None
                langmuir_use = (acc["langmuir"] * inv_cnt) if isinstance(acc.get("langmuir"), torch.Tensor) else None
                dt_ocn = dt * float(cur_cnt)
                acc.clear()
                acc["count"] = 0
            else:
                dt_ocn = dt
                langmuir_use = langmuir_ocn
        else:
            do_step_ogcm = True
            dt_ocn = dt
            langmuir_use = langmuir_ocn

        if do_step_ogcm:
            if has_indep_ocn:
                tau_x_ocn = atm_to_ocn_intensive(coupler, tau_x_ocn)
                tau_y_ocn = atm_to_ocn_intensive(coupler, tau_y_ocn)
                f_ice_ocn = atm_to_ocn_intensive(coupler, f_ice_ocn)
                evap_ocn = atm_to_ocn_intensive(coupler, evap_ocn)
                precip_ocn = atm_to_ocn_intensive(coupler, precip_ocn, ocean_only_source=False)
                sw_net_ocn = atm_to_ocn_intensive(coupler, sw_net_ocn)
                if sw_par_ocn is not None:
                    sw_par_ocn = atm_to_ocn_intensive(coupler, sw_par_ocn)
                if u10_ocn is not None:
                    u10_ocn = atm_to_ocn_intensive(coupler, u10_ocn)
                runoff_tot_ocn = atm_to_ocn_conservative_flux(coupler, runoff_tot_ocn, ocean_only_source=False)
                runoff_ice_ocn = atm_to_ocn_conservative_flux(coupler, runoff_ice_ocn, ocean_only_source=False)
                if river_mouth_ocn is not None:
                    river_mouth_ocn = atm_to_ocn_conservative_flux(coupler, river_mouth_ocn, ocean_only_source=True)

            step_ogcm_voronoi(
                ogcm=model.ogcm,
                sfc=active_ocn_sfc,
                mesh=active_ocn_mesh,
                tau_x_sfc=tau_x_ocn,
                tau_y_sfc=tau_y_ocn,
                f_seaice=f_ice_ocn,
                evap_cell=evap_ocn,
                precip_cell=precip_ocn,
                runoff_total=runoff_tot_ocn,
                runoff_glacier=runoff_ice_ocn,
                dt=dt_ocn,
                enable_seaice_drift=bool(cfg.enable_seaice_drift),
                sw_net_ocn=sw_net_ocn,
                sw_par_ocn=sw_par_ocn,
                u10_ocn=u10_ocn,
                co2_ppmv=model.current_co2_ppmv(),
                river_mouth_flux=river_mouth_ocn,
                enable_fct_zstar=bool(cfg.enable_ogcm_fct_zstar),
                enable_scale_aware_gm=bool(cfg.enable_ogcm_scale_aware_gm),
                enable_submeso=bool(cfg.enable_ogcm_submeso),
                enable_tidal_mixing=bool(cfg.enable_ogcm_tidal_mixing),
                enable_o2_omz=bool(cfg.enable_ocean_o2_omz),
                langmuir_fac=langmuir_use,
            )
            if has_indep_ocn:
                sfc.t_ocean_ml = ocn_to_atm_mean(model.ocn_coupler, active_ocn_sfc.t_ocean_ml, fallback_atm=sfc.t_ocean_ml)
                sfc.m_seaice = ocn_to_atm_mean(model.ocn_coupler, active_ocn_sfc.m_seaice, fallback_atm=sfc.m_seaice)
                sfc.swe_seaice = ocn_to_atm_mean(model.ocn_coupler, active_ocn_sfc.swe_seaice, fallback_atm=sfc.swe_seaice)
                sfc.h_ocean_ml = ocn_to_atm_mean(model.ocn_coupler, active_ocn_sfc.h_ocean_ml, fallback_atm=sfc.h_ocean_ml)
    else:
        if bool(cfg.enable_river_routing) and bool(cfg.enable_dynamic_lakes) and model.lakes is not None:
            from aoba.river_routing import init_river_routing_state, step_river_routing_voronoi

            runoff_tot_cell = _sub_to_cell(sfc_diag_sub.runoff_total, nc, n_sub)
            runoff_ice_cell = _sub_to_cell(sfc_diag_sub.runoff_ice, nc, n_sub)
            if model.river is None:
                model.river = init_river_routing_state(mesh, device=runoff_tot_cell.device)
            step_river_routing_voronoi(
                model.river,
                torch.clamp(runoff_tot_cell - runoff_ice_cell, min=0.0),
                mesh.area_cell,
                dt,
                lake_state=model.lakes,
            )
        step_ekman_ocean_voronoi(
            ocn=model.ocn,
            sfc=sfc,
            mesh=mesh,
            tau_x_sfc=_sub_to_cell(sfc_diag_sub.tau_x, nc, n_sub),
            tau_y_sfc=_sub_to_cell(sfc_diag_sub.tau_y, nc, n_sub),
            f_seaice=_sub_to_cell(sfc_diag_sub.f_seaice, nc, n_sub),
            runoff_glacier=_sub_to_cell(sfc_diag_sub.runoff_ice, nc, n_sub),
            dt=dt,
        )
    dt1_sub = _cell_to_sub(sfc.t_ocean_ml - t1_cell_before, nc, n_sub)
    dm_ice_sub = _cell_to_sub(sfc.m_seaice - m_ice_cell_before, nc, n_sub)
    dswe_ice_sub = _cell_to_sub(sfc.swe_seaice - swe_ice_cell_before, nc, n_sub)
    t1_sub_next = sfc_sub.t_ocean_ml + dt1_sub
    freeze_def = torch.where(is_ocn_sub, torch.clamp(C.T_FREEZE_OCEAN - t1_sub_next, min=0.0), torch.zeros_like(t1_sub_next))
    c1_sub = C.CP_SEAWATER * C.RHO_SEAWATER * sfc_sub.h_ocean_ml
    sfc_sub.t_ocean_ml = torch.where(is_ocn_sub, t1_sub_next + freeze_def, sfc_sub.t_ocean_ml)
    sfc_sub.m_seaice = torch.where(
        is_ocn_sub,
        torch.clamp(sfc_sub.m_seaice + dm_ice_sub + (c1_sub * freeze_def) / C.L_F0, min=0.0),
        sfc_sub.m_seaice,
    )
    sfc_sub.swe_seaice = torch.where(
        is_ocn_sub,
        torch.clamp(sfc_sub.swe_seaice + dswe_ice_sub, min=0.0),
        sfc_sub.swe_seaice,
    )

    # 5. Subgrid Orographic Drag (Lott & Miller, 1997) + Map-Free Non-Orographic GWD
    #    (Convective: Beres et al., 2004, 2005; Frontal: Charron & Manzini, 2002; Richter et al., 2010)
    u_cell, v_cell = reconstruct_cell_winds(mesh, atmos.u_edge)
    du_oro_c, dv_oro_c = compute_orographic_drag_lott_miller(
        rho_d=atmos.rho_d,
        u_cell=u_cell,
        v_cell=v_cell,
        theta_m=atmos.theta_m,
        z_rel_c=z_rel_c,
        dz_c=atmos.dz_c,
        sigma_topo=mesh.sigma_topo,
        slope_topo=mesh.slope_topo,
        aniso_topo=mesh.aniso_topo,
        angle_topo=mesh.angle_topo,
        dt=dt,
    )
    if cfg.enable_nonoro_gwd:
        pres_host_gwd, _, _ = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
        du_ngwd_c, dv_ngwd_c = compute_nonorographic_gwd(
            mesh=mesh,
            rho_d=atmos.rho_d,
            pres_c=pres_host_gwd,
            u_cell=u_cell,
            v_cell=v_cell,
            theta_m=atmos.theta_m,
            z_rel_c=z_rel_c,
            dz_c=atmos.dz_c,
            q_conv_c=model.q_conv_diag,
            dt=dt,
        )
        du_oro_c = du_oro_c + du_ngwd_c
        dv_oro_c = dv_oro_c + dv_ngwd_c
    du_oro_edge = project_cell_vectors_to_edges(mesh, du_oro_c, dv_oro_c)

    # 6. Host SCVT Nonhydrostatic RK3-HEVI Dynamical Core & Voronoi FCT
    thm_before_dyn = atmos.theta_m.clone()
    qn_before_dyn = atmos.q_n.clone()

    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    dt_adv_host = min(900.0, 0.40 * dc_min / 100.0)
    n_hdyn = max(cfg.n_host_dyn_substeps, int(math.ceil(dt / max(dt_adv_host, 1.0))))
    dt_hdyn = dt / float(n_hdyn)

    m_old_host = (atmos.rho_d * atmos.dz_c).clone()
    uh_acc = torch.zeros_like(atmos.u_edge)
    om_acc = torch.zeros_like(atmos.w)
    for _ in range(n_hdyn):
        uh_e, om_z = step_scvt_rk3_hevi(
            state=atmos,
            mesh=mesh,
            vgrid=vgrid,
            dt_dyn=dt_hdyn,
            du_phys_edge=du_oro_edge,
        )
        uh_acc = uh_acc + uh_e
        om_acc = om_acc + om_z

    inv_nh = 1.0 / float(n_hdyn)
    m_new_host = atmos.rho_d * atmos.dz_c
    uh_mean = uh_acc * inv_nh
    om_mean = om_acc * inv_nh
    adv_list = [atmos.q_n, atmos.q_p, atmos.q_c, atmos.q_i]
    has_dust_adv = bool(cfg.enable_dust and atmos.q_dust is not None)
    if has_dust_adv:
        adv_list.append(atmos.q_dust)
    has_aero_adv = bool(cfg.enable_natural_aerosols and model.aerosols is not None)
    if has_aero_adv:
        adv_list.extend([model.aerosols.q_ss, model.aerosols.q_so4, model.aerosols.q_bc, model.aerosols.q_oc])
    has_o3_adv = bool(str(cfg.o3_mode).strip().lower() == "prognostic" and model.ozone is not None)
    if has_o3_adv:
        adv_list.append(model.ozone.o3_vmr)
    has_co2_adv = bool(
        str(cfg.co2_mode).strip().lower() == "prognostic"
        and bool(cfg.enable_co2_3d_transport)
        and model.prognostic_co2 is not None
        and model.prognostic_co2.co2_vmr is not None
    )
    if has_co2_adv:
        adv_list.append(model.prognostic_co2.co2_vmr)
    has_iso_adv = bool(bool(cfg.enable_water_isotopes) and model.water_isotopes is not None)
    if has_iso_adv:
        adv_list.extend([model.water_isotopes.q_v_18o, model.water_isotopes.q_v_hdo])
    q_adv = fct_advect_scvt_batch(
        mesh, torch.stack(adv_list, dim=0),
        m_old_host, m_new_host, uh_mean, om_mean, dt,
    )
    atmos.q_n = q_adv[0]
    atmos.q_p = q_adv[1]
    apply_scvt_storm_smagorinsky(atmos, mesh, dt)
    qc_adv = torch.clamp(q_adv[2], min=0.0)
    qi_adv = torch.clamp(q_adv[3], min=0.0)
    qci_adv = qc_adv + qi_adv
    sc_adv = torch.where(qci_adv > atmos.q_n, atmos.q_n / torch.clamp(qci_adv, min=1.0e-12), torch.ones_like(qci_adv))
    atmos.q_c = qc_adv * sc_adv
    atmos.q_i = qi_adv * sc_adv
    atmos.q_v = torch.clamp(atmos.q_n - atmos.q_c - atmos.q_i, min=0.0)
    idx_q = 4
    if has_dust_adv:
        atmos.q_dust = q_adv[idx_q]
        idx_q += 1
    if has_aero_adv:
        model.aerosols.q_ss = q_adv[idx_q]
        model.aerosols.q_so4 = q_adv[idx_q + 1]
        model.aerosols.q_bc = q_adv[idx_q + 2]
        model.aerosols.q_oc = q_adv[idx_q + 3]
        idx_q += 4
    if has_o3_adv:
        model.ozone.o3_vmr = torch.clamp(q_adv[idx_q], min=1.0e-10, max=4.0e-5)
        idx_q += 1
    if has_co2_adv:
        model.prognostic_co2.co2_vmr = torch.clamp(q_adv[idx_q], min=1.0e-5, max=0.05)
        idx_q += 1
    if has_iso_adv:
        model.water_isotopes.q_v_18o = torch.clamp(q_adv[idx_q], min=0.0)
        model.water_isotopes.q_v_hdo = torch.clamp(q_adv[idx_q + 1], min=0.0)
        idx_q += 2

    dthm_ls = (atmos.theta_m - thm_before_dyn) / dt
    dqn_ls = (atmos.q_n - qn_before_dyn) / dt
    u_host_new, v_host_new = reconstruct_cell_winds(mesh, atmos.u_edge)

    # Align subcolumn drag with post-dynamics lowest-layer flow shift while retaining subgrid gusts & current feedback
    u_sub_post = (u_sub_0 - u_ocn_sub) + _cell_to_sub(u_host_new[0] - u_cell[0], nc, n_sub)
    v_sub_post = (v_sub_0 - v_ocn_sub) + _cell_to_sub(v_host_new[0] - v_cell[0], nc, n_sub)
    old_spd_sub = torch.sqrt((u_sub_0 - u_ocn_sub) ** 2 + (v_sub_0 - v_ocn_sub) ** 2 + 1.0e-4)
    tau_mag_sub = torch.sqrt(sfc_diag_sub.tau_x * sfc_diag_sub.tau_x + sfc_diag_sub.tau_y * sfc_diag_sub.tau_y)
    drag_c_sub = tau_mag_sub / old_spd_sub
    sfc_diag_sub.tau_x = -drag_c_sub * u_sub_post
    sfc_diag_sub.tau_y = -drag_c_sub * v_sub_post

    # 7. Embedded 2D/3D CRM Physics & Convective Transport with Subgrid Surface Tiles & ICA Radiation
    rain_new_sub = torch.zeros_like(pres_sfc_sub)
    snow_new_sub = torch.zeros_like(pres_sfc_sub)
    q_rad_4d = q_rad_sub.reshape(nz, nc, ny, nx)

    by_full = ny_full // max(ny_t, 1)
    bx_full = nx_full // max(nx_t, 1)
    by_crm = ny // max(ny_t, 1)
    bx_crm = nx // max(nx_t, 1)

    def _tile3d_to_crm3d(t: torch.Tensor, *, to_full: bool = False) -> torch.Tensor:
        ny_o, nx_o = (ny_full, nx_full) if to_full else (ny, nx)
        by_o, bx_o = (by_full, bx_full) if to_full else (by_crm, bx_crm)
        if ny_o == ny_t and nx_o == nx_t:
            return t
        return (
            t.reshape(t.shape[0], ny_t, 1, nx_t, 1)
            .expand(t.shape[0], ny_t, by_o, nx_t, bx_o)
            .reshape(t.shape[0], ny_o, nx_o)
            .contiguous()
        )

    def _crm3d_to_tile3d(t: torch.Tensor, *, from_full: bool = False) -> torch.Tensor:
        ny_i, nx_i = (ny_full, nx_full) if from_full else (ny, nx)
        by_i, bx_i = (by_full, bx_full) if from_full else (by_crm, bx_crm)
        if ny_i == ny_t and nx_i == nx_t:
            return t
        return t.reshape(t.shape[0], ny_t, by_i, nx_t, bx_i).mean(dim=(-3, -1))

    def _crm4d_to_tile4d(t: torch.Tensor, *, from_full: bool = False) -> torch.Tensor:
        ny_i, nx_i = (ny_full, nx_full) if from_full else (ny, nx)
        by_i, bx_i = (by_full, bx_full) if from_full else (by_crm, bx_crm)
        if ny_i == ny_t and nx_i == nx_t:
            return t
        return t.reshape(t.shape[0], t.shape[1], ny_t, by_i, nx_t, bx_i).mean(dim=(-3, -1))

    def _tile4d_to_crm4d(t: torch.Tensor, *, to_full: bool = False) -> torch.Tensor:
        ny_o, nx_o = (ny_full, nx_full) if to_full else (ny, nx)
        by_o, bx_o = (by_full, bx_full) if to_full else (by_crm, bx_crm)
        if ny_o == ny_t and nx_o == nx_t:
            return t
        return (
            t.reshape(t.shape[0], t.shape[1], ny_t, 1, nx_t, 1)
            .expand(t.shape[0], t.shape[1], ny_t, by_o, nx_t, bx_o)
            .reshape(t.shape[0], t.shape[1], ny_o, nx_o)
        )

    if cfg.enable_superparam:
        thm_before_crm = atmos.theta_m.clone()
        du_crm_c = torch.zeros_like(u_host_new)
        dv_crm_c = torch.zeros_like(v_host_new)
        sfc_diag_tile = {k: v.reshape(nc, ny_t, nx_t) for k, v in sfc_diag_sub.__dict__.items()}
        is_ocn_tile = is_ocn_sub.reshape(nc, ny_t, nx_t)
        u_sfc_tile = torch.hypot(u_sub_post, v_sub_post).reshape(nc, ny_t, nx_t)
        q_dust_4d = (
            atmos.q_dust.unsqueeze(-1).unsqueeze(-1).expand(nz, nc, ny, nx)
            if (cfg.enable_dust and atmos.q_dust is not None)
            else None
        )

        chunk = max(cfg.crm_col_chunk, 1)
        use_distill = bool(cfg.enable_crm_self_distill)
        if use_distill and model.crm_distill is None:
            from aoba.crm_self_distill import init_crm_self_distill_state

            model.crm_distill = init_crm_self_distill_state(
                n_cells=nc,
                nz=nz,
                device=atmos.rho_d.device,
                stride=int(cfg.crm_distill_stride),
                warmup_steps=int(cfg.crm_distill_warmup_steps),
                dwell_steps=int(cfg.crm_distill_dwell_steps),
                hidden_dim=int(cfg.crm_distill_hidden_dim),
                replay_size=int(cfg.crm_distill_replay_size),
                ny=ny_t,
                nx=nx_t,
            )
        distill_sel = None
        sub_maps_all_sd = None
        if use_distill and model.crm_distill is not None:
            from aoba.crm_self_distill import (
                apply_subgrid_coldpool_outflow,
                apply_subgrid_surface_microclimate,
                compute_pool_org_invariants,
                compute_subgrid_structure_features,
                evaluate_crm_shadow_on_teachers,
                extract_crm_distill_features,
                extract_spatial_targets,
                extract_subgrid_spatial_maps,
                get_convective_memory_features,
                pack_teacher_context_and_targets,
                predict_student_columns,
                reconstruct_subgrid_heterogeneity,
                select_teacher_indices,
                train_student_online,
                update_convective_memory,
            )

            q_rad_cell_pre = _sub_to_cell(q_rad_sub, nc, n_crm)
            shf_cell_pre = _sub_to_cell(sfc_diag_sub.shf, nc, n_sub)
            lhf_cell_pre = _sub_to_cell(sfc_diag_sub.lhf, nc, n_sub)
            evap_cell_pre = _sub_to_cell(sfc_diag_sub.evap, nc, n_sub)
            buoy_cell_pre_sd = _sub_to_cell(sfc_diag_sub.buoy_flux, nc, n_sub)
            ustar_cell_pre = _sub_to_cell(sfc_diag_sub.u_star, nc, n_sub)
            tx_cell_pre = _sub_to_cell(sfc_diag_sub.tau_x, nc, n_sub)
            ty_cell_pre = _sub_to_cell(sfc_diag_sub.tau_y, nc, n_sub)
            is_ocn_cell_pre = (sfc.surf_type == SURF_OCEAN)
            rho_pre_sd = atmos.rho_d.clone()
            thm_pre_sd = atmos.theta_m.clone()
            qn_pre_sd = atmos.q_n.clone()
            qp_pre_sd = atmos.q_p.clone()
            qv_pre_sd = atmos.q_v.clone()
            qc_pre_sd = atmos.q_c.clone()
            qi_pre_sd = atmos.q_i.clone()
            qke_pre_sd = mean_2d(crm.qke).clone()
            dx_sfc_eff = float(cfg.crm_dx) * float(bx_full)
            dy_sfc_eff = float(cfg.crm_dx) * float(by_full)
            subgrid_struct_sd = compute_subgrid_structure_features(
                sfc_diag_tile["shf"],
                sfc_diag_tile["lhf"],
                sfc_diag_tile["t_rad_sfc"],
                sfc_diag_tile["alb_vis_dir"],
                crm.q_n,
                crm.w,
                dx=(dx_sfc_eff if has_pool else float(cfg.crm_dx)),
                dy=(dy_sfc_eff if has_pool else float(cfg.crm_dx)),
                dx_sfc=dx_sfc_eff,
                dy_sfc=dy_sfc_eff,
            )
            if has_pool and t_pool_idx is not None:
                # Overwrite active teacher atmospheric organization invariants with true full-3D `(ny_full, nx_full)` stats
                subgrid_struct_sd[t_pool_idx, 12:16] = compute_pool_org_invariants(
                    pool.crm.q_n[:, :n_pool_act],
                    pool.crm.w[:, :n_pool_act],
                    dx=float(cfg.crm_dx),
                    dy=float(cfg.crm_dx),
                )
            if n_sub > 1:
                sub_maps_all_sd = extract_subgrid_spatial_maps(
                    sfc_diag_tile["shf"],
                    sfc_diag_tile["lhf"],
                    sfc_diag_tile["t_rad_sfc"],
                    sfc_diag_tile["alb_vis_dir"],
                    _crm4d_to_tile4d(crm.q_n),
                    _crm4d_to_tile4d(crm.w),
                    thm0_3d=_crm3d_to_tile3d(crm.theta_m[0]),
                    qv0_3d=_crm3d_to_tile3d(crm.q_v[0]),
                )
            conv_mem_sd = get_convective_memory_features(model.crm_distill, buoy_cell_pre_sd)
            x_all_sd, exner_all_sd, temp_all_sd = extract_crm_distill_features(
                rho_pre_sd, thm_pre_sd, qv_pre_sd, qc_pre_sd, qi_pre_sd, qp_pre_sd,
                u_host_new, v_host_new, atmos.dz_c, dthm_ls, dqn_ls, q_rad_cell_pre,
                shf_cell_pre, lhf_cell_pre, ustar_cell_pre, tx_cell_pre, ty_cell_pre,
                is_ocn_cell_pre, mesh.f_cell, dt,
                subgrid_struct=subgrid_struct_sd,
                conv_memory=conv_mem_sd,
            )
            thm_post_ls_sd = thm_pre_sd + dt * q_rad_cell_pre * (1.0 + C.RV_OVER_RD * qv_pre_sd) / torch.clamp(exner_all_sd, min=1.0e-3)
            sparse_warmup_crm = (ny > 1 and nx > 1 and m_crm > 100_000)
            if has_pool and t_pool_idx is not None and s_pool_idx is not None:
                distill_sel = (t_pool_idx, s_pool_idx)
            else:
                distill_sel = select_teacher_indices(model.crm_distill, allow_sparse_warmup=sparse_warmup_crm)

        if distill_sel is not None and model.crm_distill is not None:
            t_idx, s_idx = distill_sel
            rain_3d = rain_new_sub.reshape(nc, ny_t, nx_t)
            snow_3d = snow_new_sub.reshape(nc, ny_t, nx_t)

            # Operator-split OSSP baseline: advance stiff 1D vertical MYNN Level 2.5 PBL mixing and
            # surface flux injection analytically on all host columns (`[nz, nc]`, single fused tridiagonal
            # kernel) so surrogate columns preserve exact boundary-layer physics and unconditional vertical
            # stability while the neural surrogate emulates purely 3D convective transport and microphysics.
            buoy_cell_pre = _sub_to_cell(sfc_diag_sub.buoy_flux, nc, n_sub)
            qke_host_in = mean_2d(crm.qke)
            (
                u_mynn_sd, v_mynn_sd, thm_mynn_sd, qn_mynn_sd, qke_mynn_sd,
                _, _, qv_mynn_sd, qc_mynn_sd, qi_mynn_sd,
            ) = step_mynn_level25(
                rho_d=rho_pre_sd,
                u=u_host_new,
                v=v_host_new,
                theta_m_c=thm_post_ls_sd,
                q_n=qn_pre_sd,
                q_v=qv_pre_sd,
                q_c=qc_pre_sd,
                q_i=qi_pre_sd,
                qke=qke_host_in,
                z_c=atmos.z_c,
                z_w=atmos.z_w,
                dz_c=atmos.dz_c,
                u_star=ustar_cell_pre,
                sfc_buoy_flux=buoy_cell_pre,
                tau_x_sfc=tx_cell_pre,
                tau_y_sfc=ty_cell_pre,
                shf_sfc=shf_cell_pre,
                evap_sfc=evap_cell_pre,
                dt=dt,
                return_hydrometeors=True,
                do_sat_adj=False,
            )
            _, exner_mynn_sd, temp_mynn_sd = diagnose_pressure_temperature(rho_pre_sd, thm_mynn_sd, qv_mynn_sd)
            zero_sfc_sd = torch.zeros_like(shf_cell_pre)

            # 7a-i. Execute genuine 2D/3D CRM strictly on the active teacher subset (`1 / stride` columns)
            if has_pool:
                n_t = n_pool_act
                if n_t == pool.n_pool:
                    sub_crm_t = pool.crm
                else:
                    sub_crm_t = CRMState(
                        rho_d=pool.crm.rho_d[:, :n_t],
                        u=pool.crm.u[:, :n_t],
                        v=pool.crm.v[:, :n_t],
                        w=pool.crm.w[:, :n_t],
                        theta_m=pool.crm.theta_m[:, :n_t],
                        q_n=pool.crm.q_n[:, :n_t],
                        q_p=pool.crm.q_p[:, :n_t],
                        q_v=pool.crm.q_v[:, :n_t],
                        q_c=pool.crm.q_c[:, :n_t],
                        q_i=pool.crm.q_i[:, :n_t],
                        qke=pool.crm.qke[:, :n_t],
                        dx=float(cfg.crm_dx),
                        dy=float(cfg.crm_dx),
                        f_coriolis=mesh.f_cell[t_idx],
                        n_c=pool.crm.n_c[:, :n_t] if pool.crm.n_c is not None else None,
                        n_i=pool.crm.n_i[:, :n_t] if pool.crm.n_i is not None else None,
                    )
                drho_t = (atmos.rho_d[:, t_idx] - mean_2d(sub_crm_t.rho_d)).unsqueeze(-1).unsqueeze(-1)
                sub_crm_t.rho_d = torch.clamp(sub_crm_t.rho_d + drho_t, min=1.0e-5)
                sub_crm_t.f_coriolis = mesh.f_cell[t_idx]
                # Only expand the 6 SurfaceFluxDiagnostics fields read by `step_crm_physics_and_dynamics`
                _crm_sfc_keys = ("shf", "evap", "tau_x", "tau_y", "u_star", "buoy_flux")
                sub_sfc_t = SurfaceFluxDiagnostics(
                    **{
                        k: (_tile3d_to_crm3d(v.index_select(0, t_idx), to_full=True) if k in _crm_sfc_keys else v.index_select(0, t_idx))
                        for k, v in sfc_diag_tile.items()
                    }
                )
                q_rad_t_in = (
                    pool.q_rad[:, :n_t]
                    if pool.q_rad_valid
                    else _tile4d_to_crm4d(q_rad_4d[:, t_idx], to_full=True)
                )
                q_dust_t_in = (
                    atmos.q_dust[:, t_idx].unsqueeze(-1).unsqueeze(-1).expand(nz, n_t, ny_full, nx_full)
                    if (cfg.enable_dust and atmos.q_dust is not None)
                    else None
                )
                sp_t = step_crm_physics_and_dynamics(
                    crm=sub_crm_t,
                    vgrid=vgrid,
                    dt_crm=dt,
                    n_dyn_substeps=cfg.n_crm_dyn_substeps,
                    u_host_target=u_host_new[:, t_idx],
                    v_host_target=v_host_new[:, t_idx],
                    tau_sp_mom=dt,
                    dtheta_m_ls_dt=dthm_ls[:, t_idx],
                    dq_n_ls_dt=dqn_ls[:, t_idx],
                    sfc_fluxes=sub_sfc_t,
                    rad_heating_k_s=q_rad_t_in,
                    z_c_col=atmos.z_c[:, t_idx],
                    z_w_col=atmos.z_w[:, t_idx],
                    dz_c_col=atmos.dz_c[:, t_idx],
                    microphysics_scheme=cfg.microphysics_scheme,
                    is_ocean_sub=_tile3d_to_crm3d(is_ocn_tile[t_idx], to_full=True),
                    u_sfc_sub=_tile3d_to_crm3d(u_sfc_tile[t_idx], to_full=True),
                    q_dust_sub=q_dust_t_in,
                )
                if n_t < pool.n_pool:
                    pool.crm.rho_d[:, :n_t] = sub_crm_t.rho_d
                    pool.crm.u[:, :n_t] = sub_crm_t.u
                    pool.crm.v[:, :n_t] = sub_crm_t.v
                    pool.crm.w[:, :n_t] = sub_crm_t.w
                    pool.crm.theta_m[:, :n_t] = sub_crm_t.theta_m
                    pool.crm.q_n[:, :n_t] = sub_crm_t.q_n
                    pool.crm.q_p[:, :n_t] = sub_crm_t.q_p
                    pool.crm.q_v[:, :n_t] = sub_crm_t.q_v
                    pool.crm.q_c[:, :n_t] = sub_crm_t.q_c
                    pool.crm.q_i[:, :n_t] = sub_crm_t.q_i
                    pool.crm.qke[:, :n_t] = sub_crm_t.qke
                for opt_attr in ("n_c", "n_i", "q_r", "q_s", "q_g", "n_r", "n_s", "n_g"):
                    src_f = getattr(sub_crm_t, opt_attr, None)
                    if src_f is not None:
                        dst_f = getattr(crm, opt_attr, None)
                        if dst_f is None:
                            dst_f = torch.zeros_like(crm.q_c)
                            setattr(crm, opt_attr, dst_f)
                        dst_f.index_copy_(1, t_idx, _crm4d_to_tile4d(src_f, from_full=True))
                        if n_t < pool.n_pool:
                            p_dst = getattr(pool.crm, opt_attr, None)
                            if p_dst is None:
                                p_dst = torch.zeros_like(pool.crm.q_c)
                                setattr(pool.crm, opt_attr, p_dst)
                            p_dst[:, :n_t] = src_f
            else:
                rho_crm_t = crm.rho_d[:, t_idx]
                drho_t = (atmos.rho_d[:, t_idx] - mean_2d(rho_crm_t)).unsqueeze(-1).unsqueeze(-1)
                sub_crm_t = CRMState(
                    rho_d=torch.clamp(rho_crm_t + drho_t, min=1.0e-5),
                    u=crm.u[:, t_idx],
                    v=crm.v[:, t_idx],
                    w=crm.w[:, t_idx],
                    theta_m=crm.theta_m[:, t_idx],
                    q_n=crm.q_n[:, t_idx],
                    q_p=crm.q_p[:, t_idx],
                    q_v=crm.q_v[:, t_idx],
                    q_c=crm.q_c[:, t_idx],
                    q_i=crm.q_i[:, t_idx],
                    qke=crm.qke[:, t_idx],
                    dx=cfg.crm_dx,
                    dy=cfg.crm_dx,
                    f_coriolis=mesh.f_cell[t_idx],
                    n_c=crm.n_c[:, t_idx] if crm.n_c is not None else None,
                    n_i=crm.n_i[:, t_idx] if crm.n_i is not None else None,
                    q_r=crm.q_r[:, t_idx] if getattr(crm, "q_r", None) is not None else None,
                    q_s=crm.q_s[:, t_idx] if getattr(crm, "q_s", None) is not None else None,
                    q_g=crm.q_g[:, t_idx] if getattr(crm, "q_g", None) is not None else None,
                    n_r=crm.n_r[:, t_idx] if getattr(crm, "n_r", None) is not None else None,
                    n_s=crm.n_s[:, t_idx] if getattr(crm, "n_s", None) is not None else None,
                    n_g=crm.n_g[:, t_idx] if getattr(crm, "n_g", None) is not None else None,
                )
                sub_sfc_t = SurfaceFluxDiagnostics(**{k: _tile3d_to_crm3d(v[t_idx]) for k, v in sfc_diag_tile.items()})
                sp_t = step_crm_physics_and_dynamics(
                    crm=sub_crm_t,
                    vgrid=vgrid,
                    dt_crm=dt,
                    n_dyn_substeps=cfg.n_crm_dyn_substeps,
                    u_host_target=u_host_new[:, t_idx],
                    v_host_target=v_host_new[:, t_idx],
                    tau_sp_mom=dt,
                    dtheta_m_ls_dt=dthm_ls[:, t_idx],
                    dq_n_ls_dt=dqn_ls[:, t_idx],
                    sfc_fluxes=sub_sfc_t,
                    rad_heating_k_s=q_rad_4d[:, t_idx],
                    z_c_col=atmos.z_c[:, t_idx],
                    z_w_col=atmos.z_w[:, t_idx],
                    dz_c_col=atmos.dz_c[:, t_idx],
                    microphysics_scheme=cfg.microphysics_scheme,
                    is_ocean_sub=_tile3d_to_crm3d(is_ocn_tile[t_idx]),
                    u_sfc_sub=_tile3d_to_crm3d(u_sfc_tile[t_idx]),
                    q_dust_sub=q_dust_4d[:, t_idx] if q_dust_4d is not None else None,
                )
                for opt_attr in ("n_c", "n_i", "q_r", "q_s", "q_g", "n_r", "n_s", "n_g"):
                    src_f = getattr(sub_crm_t, opt_attr, None)
                    if src_f is not None:
                        dst_f = getattr(crm, opt_attr, None)
                        if dst_f is None:
                            dst_f = torch.zeros_like(crm.q_c)
                            setattr(crm, opt_attr, dst_f)
                        dst_f.index_copy_(1, t_idx, src_f)

            rho_t_mean = mean_2d(sub_crm_t.rho_d)
            inv_rho_t = 1.0 / torch.clamp(rho_t_mean, min=1.0e-5)
            thm_t_post = mean_2d(sub_crm_t.rho_d * sub_crm_t.theta_m) * inv_rho_t
            qn_t_post = mean_2d(sub_crm_t.rho_d * sub_crm_t.q_n) * inv_rho_t
            qp_t_post = mean_2d(sub_crm_t.rho_d * sub_crm_t.q_p) * inv_rho_t
            qv_t_post = mean_2d(sub_crm_t.rho_d * sub_crm_t.q_v) * inv_rho_t
            qc_t_post = mean_2d(sub_crm_t.rho_d * sub_crm_t.q_c) * inv_rho_t
            qi_t_post = mean_2d(sub_crm_t.rho_d * sub_crm_t.q_i) * inv_rho_t
            u_t_post = mean_2d(sub_crm_t.u)
            v_t_post = mean_2d(sub_crm_t.v)

            if sp_t.precip_rain_sub is not None and sp_t.precip_snow_sub is not None:
                rain_t_tile = _crm3d_to_tile3d(sp_t.precip_rain_sub, from_full=has_pool)
                snow_t_tile = _crm3d_to_tile3d(sp_t.precip_snow_sub, from_full=has_pool)
            else:
                rain_t_tile = sp_t.precip_rain.unsqueeze(-1).unsqueeze(-1).expand(-1, ny_t, nx_t)
                snow_t_tile = sp_t.precip_snow.unsqueeze(-1).unsqueeze(-1).expand(-1, ny_t, nx_t)

            # 7a-ii. Online self-distillation update from fresh teacher CRM + GPU replay buffer
            f_cld_t = mean_2d(((sub_crm_t.q_c + sub_crm_t.q_i) > 1.0e-6).to(atmos.q_n.dtype))
            if sp_t.precip_rain_sub is not None and sp_t.precip_snow_sub is not None:
                f_prec_t = mean_2d(((sp_t.precip_rain_sub + sp_t.precip_snow_sub) > 1.0e-7).to(atmos.q_n.dtype))
            else:
                f_prec_t = ((sp_t.precip_rain + sp_t.precip_snow) > 1.0e-7).to(atmos.q_n.dtype)
            ctx_p_t, ctx_s_t, tgt_p_t, tgt_s_t = pack_teacher_context_and_targets(
                rho_pre_sd[:, t_idx], atmos.dz_c[:, t_idx], exner_mynn_sd[:, t_idx], temp_mynn_sd[0, t_idx],
                thm_mynn_sd[:, t_idx], qn_mynn_sd[:, t_idx], qp_pre_sd[:, t_idx],
                qc_mynn_sd[:, t_idx], qi_mynn_sd[:, t_idx], u_mynn_sd[:, t_idx], v_mynn_sd[:, t_idx],
                zero_sfc_sd[t_idx], zero_sfc_sd[t_idx], zero_sfc_sd[t_idx], zero_sfc_sd[t_idx],
                thm_t_post, qn_t_post, qp_t_post, qc_t_post, qi_t_post, u_t_post, v_t_post,
                sp_t.precip_rain, sp_t.precip_snow, f_cld_t, f_prec_t,
            )
            thm0_t_tile = _crm3d_to_tile3d(sub_crm_t.theta_m[0], from_full=has_pool)
            qv0_t_tile = _crm3d_to_tile3d(sub_crm_t.q_v[0], from_full=has_pool)
            u0_t_tile = _crm3d_to_tile3d(sub_crm_t.u[0], from_full=has_pool)
            v0_t_tile = _crm3d_to_tile3d(sub_crm_t.v[0], from_full=has_pool)
            tgt_maps_t, has_map_t = (
                extract_spatial_targets(
                    _crm4d_to_tile4d(sub_crm_t.q_c, from_full=has_pool),
                    _crm4d_to_tile4d(sub_crm_t.q_i, from_full=has_pool),
                    rain_t_tile, snow_t_tile,
                    rho_pre_sd[:, t_idx] * atmos.dz_c[:, t_idx],
                    thm0_3d=thm0_t_tile,
                    qv0_3d=qv0_t_tile,
                    u0_3d=u0_t_tile,
                    v0_3d=v0_t_tile,
                )
                if sub_maps_all_sd is not None
                else (None, None)
            )
            if bool(cfg.enable_distill_shadow_diag):
                wind0_t_tile = _crm3d_to_tile3d(
                    torch.hypot(sub_crm_t.u[0], sub_crm_t.v[0]), from_full=has_pool
                )
                land_mask_t_tile = (sfc_sub.surf_type.reshape(nc, ny_t, nx_t)[t_idx] == SURF_LAND)
                evaluate_crm_shadow_on_teachers(
                    model.crm_distill, x_all_sd[t_idx],
                    rho_pre_sd[:, t_idx], atmos.dz_c[:, t_idx], exner_mynn_sd[:, t_idx], temp_mynn_sd[0, t_idx],
                    thm_mynn_sd[:, t_idx], qn_mynn_sd[:, t_idx], qp_pre_sd[:, t_idx],
                    qc_mynn_sd[:, t_idx], qi_mynn_sd[:, t_idx], u_mynn_sd[:, t_idx], v_mynn_sd[:, t_idx],
                    zero_sfc_sd[t_idx], zero_sfc_sd[t_idx], zero_sfc_sd[t_idx], zero_sfc_sd[t_idx],
                    dt,
                    thm_t_post=thm_t_post, qn_t_post=qn_t_post, qc_t_post=qc_t_post, qi_t_post=qi_t_post,
                    u_t_post=u_t_post, v_t_post=v_t_post,
                    rain_t_cell=sp_t.precip_rain, snow_t_cell=sp_t.precip_snow, f_cld_t=f_cld_t,
                    sub_maps_t=(sub_maps_all_sd[t_idx] if sub_maps_all_sd is not None else None),
                    rain_t_tile=rain_t_tile, snow_t_tile=snow_t_tile,
                    thm0_t_tile=thm0_t_tile, qv0_t_tile=qv0_t_tile, wind0_t_tile=wind0_t_tile,
                    land_mask_tile=land_mask_t_tile,
                    shf_tile=sfc_diag_tile["shf"][t_idx], evap_tile=sfc_diag_tile["evap"][t_idx],
                    t_cell_idx=t_idx,
                    step_count=model.step_count,
                )
            n_cgrad = 4 if ((has_pool or sparse_warmup_crm) and model.crm_distill.steps_seen < model.crm_distill.warmup_steps) else 2
            train_student_online(
                model.crm_distill, x_all_sd[t_idx], ctx_p_t, ctx_s_t, tgt_p_t, tgt_s_t, dt,
                n_grad_steps=n_cgrad,
                sub_maps_fresh=(sub_maps_all_sd[t_idx] if sub_maps_all_sd is not None else None),
                tgt_maps_fresh=tgt_maps_t,
                tgt_has_map_fresh=has_map_t,
            )

            # 7a-iii. Conservation-constrained student inference & tile update.
            # When teachers are sparse (`t_idx` << `nc`), updating all `nc` columns contiguously
            # and overwriting only the tiny `t_idx` teacher slice eliminates ~30 full-grid fancy-index
            # gathers/scatters while yielding mathematically identical teacher & student states.
            use_contig_stu = int(t_idx.numel()) * 4 < nc
            if use_contig_stu:
                stu_out, stu_unc = predict_student_columns(
                    model.crm_distill,
                    x_all_sd,
                    rho_pre_sd, atmos.dz_c, exner_mynn_sd, temp_mynn_sd[0],
                    thm_mynn_sd, qn_mynn_sd, qp_pre_sd,
                    qc_mynn_sd, qi_mynn_sd, u_mynn_sd, v_mynn_sd,
                    zero_sfc_sd, zero_sfc_sd, zero_sfc_sd, zero_sfc_sd,
                    dt,
                    sub_maps_stu=sub_maps_all_sd,
                )
                atmos.theta_m = stu_out["theta_m"]
                atmos.q_n = stu_out["q_n"]
                atmos.q_p = stu_out["q_p"]
                atmos.q_v = stu_out["q_v"]
                atmos.q_c = stu_out["q_c"]
                atmos.q_i = stu_out["q_i"]
                du_crm_c = (stu_out["u"] - u_host_new) / dt
                dv_crm_c = (stu_out["v"] - v_host_new) / dt

                if (
                    can_use_elem_metal(atmos.rho_d)
                    and 1 < (ny_t * nx_t) <= 64
                    and tuple(crm.rho_d.shape[-2:]) == (ny_t, nx_t)
                ):
                    if not crm.rho_d.is_contiguous():
                        crm.rho_d = crm.rho_d.contiguous()
                    if not crm.theta_m.is_contiguous():
                        crm.theta_m = crm.theta_m.contiguous()
                    if not crm.q_n.is_contiguous():
                        crm.q_n = crm.q_n.contiguous()
                    if not crm.q_p.is_contiguous():
                        crm.q_p = crm.q_p.contiguous()
                    if not crm.q_v.is_contiguous():
                        crm.q_v = crm.q_v.contiguous()
                    if not crm.q_c.is_contiguous():
                        crm.q_c = crm.q_c.contiguous()
                    if not crm.q_i.is_contiguous():
                        crm.q_i = crm.q_i.contiguous()
                    if not crm.u.is_contiguous():
                        crm.u = crm.u.contiguous()
                    if not crm.v.is_contiguous():
                        crm.v = crm.v.contiguous()
                    dispatch_crm_student_tile_update_metal(
                        atmos_rho=atmos.rho_d,
                        stu_thm=stu_out["theta_m"],
                        stu_qv=stu_out["q_v"],
                        stu_qc=stu_out["q_c"],
                        stu_qi=stu_out["q_i"],
                        stu_qp=stu_out["q_p"],
                        stu_fcld=stu_out["f_cld"],
                        stu_u=stu_out["u"],
                        stu_v=stu_out["v"],
                        stu_rain=stu_out["precip_rain"],
                        stu_snow=stu_out["precip_snow"],
                        stu_fprec=stu_out["f_prec"],
                        score_cld=stu_out.get("score_cld"),
                        score_prec=stu_out.get("score_prec"),
                        rho_pre_0=rho_pre_sd[0],
                        dz_0=atmos.dz_c[0],
                        exner_0=exner_mynn_sd[0],
                        shf_tile=sfc_diag_tile["shf"],
                        evap_tile=sfc_diag_tile["evap"],
                        dthm_cp=stu_out.get("dthm_cp"),
                        dqv_cp=stu_out.get("dqv_cp"),
                        u_gust=stu_out.get("u_gust"),
                        crm_rho=crm.rho_d,
                        crm_thm=crm.theta_m,
                        crm_qn=crm.q_n,
                        crm_qp=crm.q_p,
                        crm_qv=crm.q_v,
                        crm_qc=crm.q_c,
                        crm_qi=crm.q_i,
                        crm_u=crm.u,
                        crm_v=crm.v,
                        rain_3d=rain_3d,
                        snow_3d=snow_3d,
                        ny_t=ny_t,
                        nx_t=nx_t,
                        step_count=model.step_count,
                        dt=dt,
                    )
                    crm.w.mul_(0.85)
                    crm.qke.copy_(qke_mynn_sd.unsqueeze(-1).unsqueeze(-1))
                else:
                    all_ids = torch.arange(nc, device=atmos.rho_d.device, dtype=torch.int64)
                    qn_4s, qp_4s, qv_4s, qc_4s, qi_4s, rain_3s, snow_3s = reconstruct_subgrid_heterogeneity(
                        stu_out["q_v"], stu_out["q_c"], stu_out["q_i"], stu_out["q_p"], stu_out["f_cld"],
                        stu_out["precip_rain"], stu_out["precip_snow"], stu_out["f_prec"],
                        all_ids, ny_t, nx_t, step_count=model.step_count,
                        score_cld=stu_out.get("score_cld"),
                        score_prec=stu_out.get("score_prec"),
                    )
                    rain_3d = rain_3s
                    snow_3d = snow_3s
                    crm.rho_d.copy_(atmos.rho_d.unsqueeze(-1).unsqueeze(-1))
                    crm_thm_tile = _crm4d_to_tile4d(crm.theta_m)
                    thm_t_new = stu_out["theta_m"].unsqueeze(-1).unsqueeze(-1) + 0.85 * (
                        crm_thm_tile - mean_2d(crm_thm_tile, keepdim=True)
                    )
                    thm_0_mc, qv_0_mc = apply_subgrid_surface_microclimate(
                        thm_t_new[0], qv_4s[0],
                        rho_pre_sd[0], atmos.dz_c[0], exner_all_sd[0],
                        sfc_diag_tile["shf"], sfc_diag_tile["evap"], dt,
                        dthm_cp=stu_out.get("dthm_cp"),
                        dqv_cp=stu_out.get("dqv_cp"),
                    )
                    thm_t_new[0] = thm_0_mc
                    qv_4s[0] = qv_0_mc
                    qn_4s[0] = qv_0_mc + qc_4s[0] + qi_4s[0]
                    crm.theta_m = _tile4d_to_crm4d(thm_t_new)
                    crm.q_n = _tile4d_to_crm4d(qn_4s)
                    crm.q_p = _tile4d_to_crm4d(qp_4s)
                    crm.q_v = _tile4d_to_crm4d(qv_4s)
                    crm.q_c = _tile4d_to_crm4d(qc_4s)
                    crm.q_i = _tile4d_to_crm4d(qi_4s)
                    crm.u = stu_out["u"].unsqueeze(-1).unsqueeze(-1) + 0.85 * (crm.u - mean_2d(crm.u, keepdim=True))
                    crm.v = stu_out["v"].unsqueeze(-1).unsqueeze(-1) + 0.85 * (crm.v - mean_2d(crm.v, keepdim=True))
                    if stu_out.get("dthm_cp") is not None and stu_out.get("u_gust") is not None:
                        u0_cp, v0_cp = apply_subgrid_coldpool_outflow(
                            stu_out["u"][0], stu_out["v"][0],
                            stu_out["dthm_cp"], stu_out["u_gust"],
                            all_ids, step_count=model.step_count,
                        )
                        crm.u[0] = _tile3d_to_crm3d(u0_cp)
                        crm.v[0] = _tile3d_to_crm3d(v0_cp)
                    crm.w.mul_(0.85)
                    crm.qke.copy_(qke_mynn_sd.unsqueeze(-1).unsqueeze(-1))

                model.crm_distill.last_uncertainty = stu_unc
                model.crm_distill.last_uncertainty[t_idx] = 0.0
            else:
                stu_out, stu_unc = predict_student_columns(
                    model.crm_distill,
                    x_all_sd[s_idx],
                    rho_pre_sd[:, s_idx], atmos.dz_c[:, s_idx], exner_mynn_sd[:, s_idx], temp_mynn_sd[0, s_idx],
                    thm_mynn_sd[:, s_idx], qn_mynn_sd[:, s_idx], qp_pre_sd[:, s_idx],
                    qc_mynn_sd[:, s_idx], qi_mynn_sd[:, s_idx], u_mynn_sd[:, s_idx], v_mynn_sd[:, s_idx],
                    zero_sfc_sd[s_idx], zero_sfc_sd[s_idx], zero_sfc_sd[s_idx], zero_sfc_sd[s_idx],
                    dt,
                    sub_maps_stu=(sub_maps_all_sd[s_idx] if sub_maps_all_sd is not None else None),
                )
                atmos.theta_m[:, s_idx] = stu_out["theta_m"]
                atmos.q_n[:, s_idx] = stu_out["q_n"]
                atmos.q_p[:, s_idx] = stu_out["q_p"]
                atmos.q_v[:, s_idx] = stu_out["q_v"]
                atmos.q_c[:, s_idx] = stu_out["q_c"]
                atmos.q_i[:, s_idx] = stu_out["q_i"]
                du_crm_c[:, s_idx] = (stu_out["u"] - u_host_new[:, s_idx]) / dt
                dv_crm_c[:, s_idx] = (stu_out["v"] - v_host_new[:, s_idx]) / dt

                qn_4s, qp_4s, qv_4s, qc_4s, qi_4s, rain_3s, snow_3s = reconstruct_subgrid_heterogeneity(
                    stu_out["q_v"], stu_out["q_c"], stu_out["q_i"], stu_out["q_p"], stu_out["f_cld"],
                    stu_out["precip_rain"], stu_out["precip_snow"], stu_out["f_prec"],
                    s_idx, ny_t, nx_t, step_count=model.step_count,
                    score_cld=stu_out.get("score_cld"),
                    score_prec=stu_out.get("score_prec"),
                )
                rain_3d[s_idx] = rain_3s
                snow_3d[s_idx] = snow_3s
                crm.rho_d[:, s_idx] = atmos.rho_d[:, s_idx].unsqueeze(-1).unsqueeze(-1)
                crm_thm_tile_s = _crm4d_to_tile4d(crm.theta_m[:, s_idx])
                thm_t_new = stu_out["theta_m"].unsqueeze(-1).unsqueeze(-1) + 0.85 * (
                    crm_thm_tile_s - mean_2d(crm_thm_tile_s, keepdim=True)
                )
                thm_0_mc, qv_0_mc = apply_subgrid_surface_microclimate(
                    thm_t_new[0], qv_4s[0],
                    rho_pre_sd[0, s_idx], atmos.dz_c[0, s_idx], exner_all_sd[0, s_idx],
                    sfc_diag_tile["shf"][s_idx], sfc_diag_tile["evap"][s_idx], dt,
                    dthm_cp=stu_out.get("dthm_cp"),
                    dqv_cp=stu_out.get("dqv_cp"),
                )
                thm_t_new[0] = thm_0_mc
                qv_4s[0] = qv_0_mc
                qn_4s[0] = qv_0_mc + qc_4s[0] + qi_4s[0]
                crm.theta_m[:, s_idx] = _tile4d_to_crm4d(thm_t_new)
                crm.q_n[:, s_idx] = _tile4d_to_crm4d(qn_4s)
                crm.q_p[:, s_idx] = _tile4d_to_crm4d(qp_4s)
                crm.q_v[:, s_idx] = _tile4d_to_crm4d(qv_4s)
                crm.q_c[:, s_idx] = _tile4d_to_crm4d(qc_4s)
                crm.q_i[:, s_idx] = _tile4d_to_crm4d(qi_4s)
                crm_u_s = crm.u[:, s_idx]
                crm_v_s = crm.v[:, s_idx]
                crm_u_upd = stu_out["u"].unsqueeze(-1).unsqueeze(-1) + 0.85 * (crm_u_s - mean_2d(crm_u_s, keepdim=True))
                crm_v_upd = stu_out["v"].unsqueeze(-1).unsqueeze(-1) + 0.85 * (crm_v_s - mean_2d(crm_v_s, keepdim=True))
                if stu_out.get("dthm_cp") is not None and stu_out.get("u_gust") is not None:
                    u0_cp, v0_cp = apply_subgrid_coldpool_outflow(
                        stu_out["u"][0], stu_out["v"][0],
                        stu_out["dthm_cp"], stu_out["u_gust"],
                        s_idx, step_count=model.step_count,
                    )
                    crm_u_upd[0] = _tile3d_to_crm3d(u0_cp)
                    crm_v_upd[0] = _tile3d_to_crm3d(v0_cp)
                crm.u[:, s_idx] = crm_u_upd
                crm.v[:, s_idx] = crm_v_upd
                crm.w[:, s_idx] = 0.85 * crm.w[:, s_idx]
                crm.qke[:, s_idx] = qke_mynn_sd[:, s_idx].unsqueeze(-1).unsqueeze(-1)

                model.crm_distill.last_uncertainty.zero_()
                model.crm_distill.last_uncertainty[s_idx] = stu_unc

            # Commit exact genuine teacher state onto the tiny `t_idx` columns.
            # Keep global host `atmos.rho_d` governed uniformly by `step_scvt_rk3_hevi` across
            # teacher & student cells so sparse teacher columns do not create isolated hydrostatic
            # density dipoles against surrounding student cells.
            atmos.theta_m[:, t_idx] = thm_t_post
            atmos.q_n[:, t_idx] = qn_t_post
            atmos.q_p[:, t_idx] = qp_t_post
            atmos.q_v[:, t_idx] = qv_t_post
            atmos.q_c[:, t_idx] = qc_t_post
            atmos.q_i[:, t_idx] = qi_t_post
            du_crm_c[:, t_idx] = (u_t_post - u_host_new[:, t_idx]) / dt
            dv_crm_c[:, t_idx] = (v_t_post - v_host_new[:, t_idx]) / dt
            rain_3d[t_idx] = rain_t_tile
            snow_3d[t_idx] = snow_t_tile
            rain_new_sub = rain_3d.reshape(m_sub)
            snow_new_sub = snow_3d.reshape(m_sub)
            if has_pool:
                crm.rho_d.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.rho_d, from_full=True))
                crm.u.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.u, from_full=True))
                crm.v.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.v, from_full=True))
                crm.w.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.w, from_full=True))
                crm.theta_m.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.theta_m, from_full=True))
                crm.q_n.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.q_n, from_full=True))
                crm.q_p.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.q_p, from_full=True))
                crm.q_v.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.q_v, from_full=True))
                crm.q_c.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.q_c, from_full=True))
                crm.q_i.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.q_i, from_full=True))
                crm.qke.index_copy_(1, t_idx, _crm4d_to_tile4d(sub_crm_t.qke, from_full=True))
            else:
                crm.rho_d.index_copy_(1, t_idx, sub_crm_t.rho_d)
                crm.u.index_copy_(1, t_idx, sub_crm_t.u)
                crm.v.index_copy_(1, t_idx, sub_crm_t.v)
                crm.w.index_copy_(1, t_idx, sub_crm_t.w)
                crm.theta_m.index_copy_(1, t_idx, sub_crm_t.theta_m)
                crm.q_n.index_copy_(1, t_idx, sub_crm_t.q_n)
                crm.q_p.index_copy_(1, t_idx, sub_crm_t.q_p)
                crm.q_v.index_copy_(1, t_idx, sub_crm_t.q_v)
                crm.q_c.index_copy_(1, t_idx, sub_crm_t.q_c)
                crm.q_i.index_copy_(1, t_idx, sub_crm_t.q_i)
                crm.qke.index_copy_(1, t_idx, sub_crm_t.qke)

            model.crm_distill.teacher_frac_last = float(t_idx.numel()) / float(max(nc, 1))
            model.crm_distill.steps_seen += 1
        elif chunk >= nc and n_sub == n_crm:
            drho_mean = (atmos.rho_d - mean_2d(crm.rho_d)).unsqueeze(-1).unsqueeze(-1)
            crm.rho_d = torch.clamp(crm.rho_d + drho_mean, min=1.0e-5)
            crm.f_coriolis = mesh.f_cell
            sp = step_crm_physics_and_dynamics(
                crm=crm,
                vgrid=vgrid,
                dt_crm=dt,
                n_dyn_substeps=cfg.n_crm_dyn_substeps,
                u_host_target=u_host_new,
                v_host_target=v_host_new,
                tau_sp_mom=dt,
                dtheta_m_ls_dt=dthm_ls,
                dq_n_ls_dt=dqn_ls,
                sfc_fluxes=SurfaceFluxDiagnostics(**sfc_diag_tile),
                rad_heating_k_s=q_rad_4d,
                z_c_col=atmos.z_c,
                z_w_col=atmos.z_w,
                dz_c_col=atmos.dz_c,
                microphysics_scheme=cfg.microphysics_scheme,
                is_ocean_sub=is_ocn_tile,
                u_sfc_sub=u_sfc_tile,
                q_dust_sub=q_dust_4d,
            )
            rho_crm_mean = mean_2d(crm.rho_d)
            inv_rho_crm = 1.0 / torch.clamp(rho_crm_mean, min=1.0e-5)
            atmos.rho_d = rho_crm_mean
            atmos.theta_m = mean_2d(crm.rho_d * crm.theta_m) * inv_rho_crm
            atmos.q_n = mean_2d(crm.rho_d * crm.q_n) * inv_rho_crm
            atmos.q_p = mean_2d(crm.rho_d * crm.q_p) * inv_rho_crm
            atmos.q_v = mean_2d(crm.rho_d * crm.q_v) * inv_rho_crm
            atmos.q_c = mean_2d(crm.rho_d * crm.q_c) * inv_rho_crm
            atmos.q_i = mean_2d(crm.rho_d * crm.q_i) * inv_rho_crm
            du_crm_c = (mean_2d(crm.u) - u_host_new) / dt
            dv_crm_c = (mean_2d(crm.v) - v_host_new) / dt
            if sp.precip_rain_sub is not None and sp.precip_snow_sub is not None:
                rain_new_sub = sp.precip_rain_sub.reshape(m_sub)
                snow_new_sub = sp.precip_snow_sub.reshape(m_sub)
            else:
                rain_new_sub = _cell_to_sub(sp.precip_rain, nc, n_sub)
                snow_new_sub = _cell_to_sub(sp.precip_snow, nc, n_sub)
        else:
            rain_3d = rain_new_sub.reshape(nc, ny_t, nx_t)
            snow_3d = snow_new_sub.reshape(nc, ny_t, nx_t)
            eff_chunk = min(chunk, max(1, 32768 // max(n_crm, 1))) if n_sub < n_crm else chunk
            for i0 in range(0, nc, eff_chunk):
                i1 = min(i0 + eff_chunk, nc)
                sl = slice(i0, i1)
                rho_crm_sl = crm.rho_d[:, sl]
                drho_mean = (atmos.rho_d[:, sl] - mean_2d(rho_crm_sl)).unsqueeze(-1).unsqueeze(-1)
                sub_crm = CRMState(
                    rho_d=torch.clamp(rho_crm_sl + drho_mean, min=1.0e-5),
                    u=crm.u[:, sl],
                    v=crm.v[:, sl],
                    w=crm.w[:, sl],
                    theta_m=crm.theta_m[:, sl],
                    q_n=crm.q_n[:, sl],
                    q_p=crm.q_p[:, sl],
                    q_v=crm.q_v[:, sl],
                    q_c=crm.q_c[:, sl],
                    q_i=crm.q_i[:, sl],
                    qke=crm.qke[:, sl],
                    dx=cfg.crm_dx,
                    dy=cfg.crm_dx,
                    f_coriolis=mesh.f_cell[sl],
                    n_c=crm.n_c[:, sl] if crm.n_c is not None else None,
                    n_i=crm.n_i[:, sl] if crm.n_i is not None else None,
                )
                sub_sfc = SurfaceFluxDiagnostics(**{k: _tile3d_to_crm3d(v[sl]) for k, v in sfc_diag_tile.items()})
                sp = step_crm_physics_and_dynamics(
                    crm=sub_crm,
                    vgrid=vgrid,
                    dt_crm=dt,
                    n_dyn_substeps=cfg.n_crm_dyn_substeps,
                    u_host_target=u_host_new[:, sl],
                    v_host_target=v_host_new[:, sl],
                    tau_sp_mom=dt,
                    dtheta_m_ls_dt=dthm_ls[:, sl],
                    dq_n_ls_dt=dqn_ls[:, sl],
                    sfc_fluxes=sub_sfc,
                    rad_heating_k_s=q_rad_4d[:, sl],
                    z_c_col=atmos.z_c[:, sl],
                    z_w_col=atmos.z_w[:, sl],
                    dz_c_col=atmos.dz_c[:, sl],
                    microphysics_scheme=cfg.microphysics_scheme,
                    is_ocean_sub=_tile3d_to_crm3d(is_ocn_tile[sl]),
                    u_sfc_sub=_tile3d_to_crm3d(u_sfc_tile[sl]),
                    q_dust_sub=q_dust_4d[:, sl] if q_dust_4d is not None else None,
                )
                crm.rho_d[:, sl] = sub_crm.rho_d
                crm.u[:, sl] = sub_crm.u
                crm.v[:, sl] = sub_crm.v
                crm.w[:, sl] = sub_crm.w
                crm.theta_m[:, sl] = sub_crm.theta_m
                crm.q_n[:, sl] = sub_crm.q_n
                crm.q_p[:, sl] = sub_crm.q_p
                crm.q_v[:, sl] = sub_crm.q_v
                crm.q_c[:, sl] = sub_crm.q_c
                crm.q_i[:, sl] = sub_crm.q_i
                crm.qke[:, sl] = sub_crm.qke
                if sub_crm.n_c is not None:
                    if crm.n_c is None:
                        crm.n_c = torch.zeros_like(crm.q_c)
                    crm.n_c[:, sl] = sub_crm.n_c
                if sub_crm.n_i is not None:
                    if crm.n_i is None:
                        crm.n_i = torch.zeros_like(crm.q_i)
                    crm.n_i[:, sl] = sub_crm.n_i

                rho_crm_mean = mean_2d(sub_crm.rho_d)
                inv_rho_crm = 1.0 / torch.clamp(rho_crm_mean, min=1.0e-5)
                atmos.rho_d[:, sl] = rho_crm_mean
                atmos.theta_m[:, sl] = mean_2d(sub_crm.rho_d * sub_crm.theta_m) * inv_rho_crm
                atmos.q_n[:, sl] = mean_2d(sub_crm.rho_d * sub_crm.q_n) * inv_rho_crm
                atmos.q_p[:, sl] = mean_2d(sub_crm.rho_d * sub_crm.q_p) * inv_rho_crm
                atmos.q_v[:, sl] = mean_2d(sub_crm.rho_d * sub_crm.q_v) * inv_rho_crm
                atmos.q_c[:, sl] = mean_2d(sub_crm.rho_d * sub_crm.q_c) * inv_rho_crm
                atmos.q_i[:, sl] = mean_2d(sub_crm.rho_d * sub_crm.q_i) * inv_rho_crm

                du_crm_c[:, sl] = (mean_2d(sub_crm.u) - u_host_new[:, sl]) / dt
                dv_crm_c[:, sl] = (mean_2d(sub_crm.v) - v_host_new[:, sl]) / dt
                if sp.precip_rain_sub is not None and sp.precip_snow_sub is not None:
                    rain_3d[sl] = _crm3d_to_tile3d(sp.precip_rain_sub)
                    snow_3d[sl] = _crm3d_to_tile3d(sp.precip_snow_sub)
                else:
                    rain_3d[sl] = sp.precip_rain.unsqueeze(-1).unsqueeze(-1)
                    snow_3d[sl] = sp.precip_snow.unsqueeze(-1).unsqueeze(-1)

        if use_distill and distill_sel is None and model.crm_distill is not None:
            # Warmup phase: 100% genuine CRM ran above; collect stratified teacher pairs and pre-train student
            rain_cell_w = _sub_to_cell(rain_new_sub, nc, n_sub)
            snow_cell_w = _sub_to_cell(snow_new_sub, nc, n_sub)
            n_samp = min(nc, 512)
            if n_samp < nc:
                w_idx = torch.randperm(nc, device=atmos.rho_d.device)[:n_samp]
            else:
                w_idx = torch.arange(nc, device=atmos.rho_d.device)
            u_post_w = u_host_new[:, w_idx] + dt * du_crm_c[:, w_idx]
            v_post_w = v_host_new[:, w_idx] + dt * dv_crm_c[:, w_idx]
            f_cld_w = mean_2d(((crm.q_c[:, w_idx] + crm.q_i[:, w_idx]) > 1.0e-6).to(atmos.q_n.dtype))
            f_prec_w = mean_2d(
                ((rain_new_sub.reshape(nc, ny_t, nx_t)[w_idx] + snow_new_sub.reshape(nc, ny_t, nx_t)[w_idx]) > 1.0e-7).to(atmos.q_n.dtype)
            )
            # Advance exact same operator-split host 1D MYNN baseline (do_sat_adj=False) on warmup samples
            # so warmup replay targets are 100% consistent with post-warmup student inference (zero double-mixing).
            buoy_cell_w = _sub_to_cell(sfc_diag_sub.buoy_flux, nc, n_sub)[w_idx]
            (
                u_mynn_w, v_mynn_w, thm_mynn_w, qn_mynn_w, _,
                _, _, qv_mynn_w, qc_mynn_w, qi_mynn_w,
            ) = step_mynn_level25(
                rho_d=rho_pre_sd[:, w_idx],
                u=u_host_new[:, w_idx],
                v=v_host_new[:, w_idx],
                theta_m_c=thm_post_ls_sd[:, w_idx],
                q_n=qn_pre_sd[:, w_idx],
                q_v=qv_pre_sd[:, w_idx],
                q_c=qc_pre_sd[:, w_idx],
                q_i=qi_pre_sd[:, w_idx],
                qke=qke_pre_sd[:, w_idx],
                z_c=atmos.z_c[:, w_idx],
                z_w=atmos.z_w[:, w_idx],
                dz_c=atmos.dz_c[:, w_idx],
                u_star=ustar_cell_pre[w_idx],
                sfc_buoy_flux=buoy_cell_w,
                tau_x_sfc=tx_cell_pre[w_idx],
                tau_y_sfc=ty_cell_pre[w_idx],
                shf_sfc=shf_cell_pre[w_idx],
                evap_sfc=evap_cell_pre[w_idx],
                dt=dt,
                return_hydrometeors=True,
                do_sat_adj=False,
            )
            _, exner_mynn_w, temp_mynn_w = diagnose_pressure_temperature(rho_pre_sd[:, w_idx], thm_mynn_w, qv_mynn_w)
            zero_sfc_w = torch.zeros_like(shf_cell_pre[w_idx])
            ctx_p_w, ctx_s_w, tgt_p_w, tgt_s_w = pack_teacher_context_and_targets(
                rho_pre_sd[:, w_idx], atmos.dz_c[:, w_idx], exner_mynn_w, temp_mynn_w[0],
                thm_mynn_w, qn_mynn_w, qp_pre_sd[:, w_idx],
                qc_mynn_w, qi_mynn_w, u_mynn_w, v_mynn_w,
                zero_sfc_w, zero_sfc_w, zero_sfc_w, zero_sfc_w,
                atmos.theta_m[:, w_idx], atmos.q_n[:, w_idx], atmos.q_p[:, w_idx],
                atmos.q_c[:, w_idx], atmos.q_i[:, w_idx], u_post_w, v_post_w,
                rain_cell_w[w_idx], snow_cell_w[w_idx], f_cld_w, f_prec_w,
            )
            rain_w_tile = rain_new_sub.reshape(nc, ny_t, nx_t)[w_idx]
            snow_w_tile = snow_new_sub.reshape(nc, ny_t, nx_t)[w_idx]
            thm0_w_tile = _crm3d_to_tile3d(crm.theta_m[0, w_idx])
            qv0_w_tile = _crm3d_to_tile3d(crm.q_v[0, w_idx])
            u0_w_tile = _crm3d_to_tile3d(crm.u[0, w_idx])
            v0_w_tile = _crm3d_to_tile3d(crm.v[0, w_idx])
            tgt_maps_w, has_map_w = (
                extract_spatial_targets(
                    _crm4d_to_tile4d(crm.q_c[:, w_idx]), _crm4d_to_tile4d(crm.q_i[:, w_idx]),
                    rain_w_tile,
                    snow_w_tile,
                    rho_pre_sd[:, w_idx] * atmos.dz_c[:, w_idx],
                    thm0_3d=thm0_w_tile,
                    qv0_3d=qv0_w_tile,
                    u0_3d=u0_w_tile,
                    v0_3d=v0_w_tile,
                )
                if sub_maps_all_sd is not None
                else (None, None)
            )
            if bool(cfg.enable_distill_shadow_diag):
                wind0_w_tile = _crm3d_to_tile3d(torch.hypot(crm.u[0, w_idx], crm.v[0, w_idx]))
                land_mask_w_tile = (sfc_sub.surf_type.reshape(nc, ny_t, nx_t)[w_idx] == SURF_LAND)
                evaluate_crm_shadow_on_teachers(
                    model.crm_distill, x_all_sd[w_idx],
                    rho_pre_sd[:, w_idx], atmos.dz_c[:, w_idx], exner_mynn_w, temp_mynn_w[0],
                    thm_mynn_w, qn_mynn_w, qp_pre_sd[:, w_idx],
                    qc_mynn_w, qi_mynn_w, u_mynn_w, v_mynn_w,
                    zero_sfc_w, zero_sfc_w, zero_sfc_w, zero_sfc_w,
                    dt,
                    thm_t_post=atmos.theta_m[:, w_idx], qn_t_post=atmos.q_n[:, w_idx],
                    qc_t_post=atmos.q_c[:, w_idx], qi_t_post=atmos.q_i[:, w_idx],
                    u_t_post=u_post_w, v_t_post=v_post_w,
                    rain_t_cell=rain_cell_w[w_idx], snow_t_cell=snow_cell_w[w_idx], f_cld_t=f_cld_w,
                    sub_maps_t=(sub_maps_all_sd[w_idx] if sub_maps_all_sd is not None else None),
                    rain_t_tile=rain_w_tile, snow_t_tile=snow_w_tile,
                    thm0_t_tile=thm0_w_tile, qv0_t_tile=qv0_w_tile, wind0_t_tile=wind0_w_tile,
                    land_mask_tile=land_mask_w_tile,
                    shf_tile=sfc_diag_tile["shf"][w_idx], evap_tile=sfc_diag_tile["evap"][w_idx],
                    t_cell_idx=w_idx,
                    step_count=model.step_count,
                )
            train_student_online(
                model.crm_distill, x_all_sd[w_idx], ctx_p_w, ctx_s_w, tgt_p_w, tgt_s_w, dt,
                sub_maps_fresh=(sub_maps_all_sd[w_idx] if sub_maps_all_sd is not None else None),
                tgt_maps_fresh=tgt_maps_w,
                tgt_has_map_fresh=has_map_w,
            )
            model.crm_distill.teacher_frac_last = 1.0
            model.crm_distill.steps_seen += 1

        du_sp_edge = project_cell_vectors_to_edges(mesh, du_crm_c, dv_crm_c)
        atmos.u_edge = atmos.u_edge + dt * du_sp_edge
        _, pi_post_crm, _ = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
        q_rad_cell = _sub_to_cell(q_rad_sub, nc, n_crm)
        model.q_conv_diag = torch.clamp(((atmos.theta_m - thm_before_crm) / dt) * pi_post_crm - q_rad_cell, min=0.0)

    rain_new = _sub_to_cell(rain_new_sub, nc, n_sub)
    snow_new = _sub_to_cell(snow_new_sub, nc, n_sub)
    if bool(cfg.enable_superparam) and use_distill and model.crm_distill is not None:
        update_convective_memory(
            model.crm_distill,
            (rain_new + snow_new) * C.SECONDS_PER_DAY,
            buoy_cell_pre_sd,
            dt,
        )

    # 7b. Map-free CRM Non-Inductive Charging Lightning & Natural LNOx Deposition
    light_diag = None
    if bool(cfg.enable_lightning) and bool(cfg.enable_superparam):
        from aoba.lightning import compute_crm_lightning_and_lnox

        light_diag = compute_crm_lightning_and_lnox(
            rho_d=crm.rho_d,
            theta_m=crm.theta_m,
            q_v=crm.q_v,
            q_i=crm.q_i,
            q_p=crm.q_p,
            w_stag=crm.w,
            z_c=atmos.z_c,
            dz_c=atmos.dz_c,
            area_cell=mesh.area_cell,
            q_g_prog=getattr(crm, "q_g", None),
            q_s_prog=getattr(crm, "q_s", None),
        )
        model.last_lightning = light_diag
    f_dust_dep_cell = None
    if cfg.enable_dust:
        if atmos.q_dust is None:
            atmos.q_dust = torch.zeros_like(atmos.q_n)
        f_dust_sub = compute_dust_emission_sub(
            surf_type=sfc_sub.surf_type,
            lai=lai_exp_sub,
            f_snow=sfc_diag_sub.f_snow_land,
            w_liq_soil_top=sfc_sub.w_liq_soil[0],
            w_ice_soil_top=sfc_sub.w_ice_soil[0],
            u_star=sfc_diag_sub.u_star,
            rho_atm=rho_sub_0 * (1.0 + qv_sub_0),
            z0m=sfc_diag_sub.z0m,
            d_disp=sfc_diag_sub.d_disp,
            dust_emis_factor=cfg.dust_emis_factor,
        )
        if bool(cfg.enable_dynamic_lakes) and model.lakes is not None:
            # Active lakes suppress mineral dust; desiccated basins (f_lake -> 0) naturally expose dust sources
            f_dust_sub = f_dust_sub * torch.clamp(1.0 - _cell_to_sub(model.lakes.f_lake, nc, n_sub), 0.0, 1.0)
        f_dust_emit_c = _sub_to_cell(f_dust_sub, nc, n_sub)
        dm_dust_c = torch.clamp(atmos.rho_d * atmos.dz_c, min=0.0)
        dust_col_pre = (dm_dust_c * torch.clamp(atmos.q_dust, min=0.0)).sum(dim=0)
        atmos.q_dust = step_dust_column(
            q_dust=atmos.q_dust,
            rho_d=atmos.rho_d,
            dz_c=atmos.dz_c,
            z_c=atmos.z_c,
            f_dust_emit=f_dust_emit_c,
            precip_rate=rain_new + snow_new,
            u_star_cell=_sub_to_cell(sfc_diag_sub.u_star, nc, n_sub),
            dt=dt,
        )
        dust_col_post = (dm_dust_c * torch.clamp(atmos.q_dust, min=0.0)).sum(dim=0)
        f_dust_dep_cell = torch.clamp((dust_col_pre + dt * f_dust_emit_c - dust_col_post) / max(dt, 1.0), min=0.0)
    evap_cell = _sub_to_cell(sfc_diag_sub.evap, nc, n_sub)
    shf_cell = _sub_to_cell(sfc_diag_sub.shf, nc, n_sub)
    gpp_cell = _sub_to_cell(sfc_diag_sub.gpp, nc, n_sub)
    t_rad_cell = _sub_to_cell(sfc_diag_sub.t_rad_sfc, nc, n_sub)
    f_seaice_cell = _sub_to_cell(sfc_diag_sub.f_seaice, nc, n_sub)

    model.rad_cache["rain_sub"] = rain_new_sub
    model.rad_cache["snow_sub"] = snow_new_sub
    model.rad_cache["evap_sub"] = sfc_diag_sub.evap
    model.rad_cache["shf_sub"] = sfc_diag_sub.shf
    model.rad_cache["gpp_sub"] = sfc_diag_sub.gpp
    model.rad_cache["rain"] = rain_new
    model.rad_cache["snow"] = snow_new
    model.rad_cache["evap"] = evap_cell
    model.rad_cache["shf"] = shf_cell
    model.rad_cache["gpp"] = gpp_cell
    model.last_sfc_diag_sub = sfc_diag_sub
    model.last_s_toa = s_toa

    # 8. Subgrid Solar Noon EEO Acclimation, Daily LAI/Root Phenology, Host Sync, & Monthly Köppen
    rho_post_0 = _c2s(crm.rho_d[0].reshape(m_crm))
    thm_post_0 = _c2s(crm.theta_m[0].reshape(m_crm))
    qv_post_0 = _c2s(crm.q_v[0].reshape(m_crm))
    post_accum_fn = maybe_cuda_compile(_coupled_post_crm_accum_core, device=rho_post_0.device)
    (
        t_air_sub, pres_sfc_post_sub, vpd_sub,
        day_precip_new, day_pet_new, day_ppfd_new,
        day_temp_new, day_vpd_new, day_t_min_new,
    ) = post_accum_fn(
        rho_post_0, thm_post_0, qv_post_0, dz0_sub,
        rain_new_sub, snow_new_sub,
        sw1_dir_sub, sw1_dif_sub, sw2_dir_sub, sw2_dif_sub, lw_dn_sub,
        sfc_diag_sub.alb_vis_dir, sfc_diag_sub.alb_vis_dif,
        sfc_diag_sub.alb_nir_dir, sfc_diag_sub.alb_nir_dif,
        sfc_diag_sub.emiss_sfc, sfc_diag_sub.t_rad_sfc,
        clim_sub.day_precip_mm, clim_sub.day_pet_mm, clim_sub.day_ppfd_mol,
        clim_sub.day_temp_sum, clim_sub.day_vpd_sum, clim_sub.day_t_min, dt,
    )

    # Cache CPU land-cell mask once so ocean-only longitude bands skip solar-noon snapshots cleanly
    land_cell_cpu = getattr(model, "_land_cell_cpu_cache", None)
    if land_cell_cpu is None or int(land_cell_cpu.numel()) != nc:
        land_cell_cpu = (sfc_sub.surf_type.reshape(nc, n_sub) == SURF_LAND).any(dim=-1).cpu()
        model._land_cell_cpu_cache = land_cell_cpu  # type: ignore[attr-defined]
    crossed_noon_land_cpu = crossed_noon_cpu & land_cell_cpu
    if bool(crossed_noon_land_cpu.any()):
        noon_buf = getattr(clim_sub, "_noon_buf", None)
        noon_seen = getattr(clim_sub, "_noon_seen", None)
        if noon_buf is None or int(noon_buf.shape[1]) != m_sub or noon_buf.device != rho_post_0.device:
            noon_buf = torch.zeros((8, m_sub), device=rho_post_0.device, dtype=rho_post_0.dtype)
            noon_seen = torch.zeros(m_sub, device=rho_post_0.device, dtype=torch.bool)
            clim_sub._noon_buf = noon_buf    # type: ignore[attr-defined]
            clim_sub._noon_seen = noon_seen  # type: ignore[attr-defined]
        noon_cells = torch.nonzero(crossed_noon_land_cpu, as_tuple=False).squeeze(-1).to(device=rho_post_0.device)
        noon_sub_idx = (
            noon_cells.unsqueeze(-1) * n_sub + torch.arange(n_sub, device=rho_post_0.device)
        ).reshape(-1)
        t_air_n = t_air_sub.index_select(0, noon_sub_idx)
        snap = torch.stack([
            sw1_dir_sub.index_select(0, noon_sub_idx),
            sw1_dif_sub.index_select(0, noon_sub_idx),
            torch.clamp(mu0_sub.index_select(0, noon_sub_idx), min=0.05),
            t_air_n,
            vpd_sub.index_select(0, noon_sub_idx),
            torch.minimum(clim_sub.day_t_min.index_select(0, noon_sub_idx), t_air_n - 2.0),
            pres_sfc_post_sub.index_select(0, noon_sub_idx),
            lai_exp_sub.index_select(0, noon_sub_idx),
        ], dim=0)
        noon_buf.index_copy_(1, noon_sub_idx, snap)
        noon_seen.index_fill_(0, noon_sub_idx, True)

    clim_sub.day_precip_mm = day_precip_new
    clim_sub.day_pet_mm = day_pet_new
    clim_sub.day_ppfd_mol = day_ppfd_new
    clim_sub.day_temp_sum = day_temp_new
    clim_sub.day_vpd_sum = day_vpd_new
    clim_sub.day_t_min = day_t_min_new
    clim_sub.day_steps += 1

    # Host monthly Köppen-Geiger accumulators (calendar month index 0..11)
    precip_tot = rain_new + snow_new
    t_air = _sub_to_cell(t_air_sub, nc, n_sub)
    if bool(cfg.enable_land_carbon):
        from aoba.land_carbon import init_land_carbon_state, step_land_carbon_and_fire

        if model.land_carbon is None:
            model.land_carbon = init_land_carbon_state(
                sfc_sub.surf_type, sfc_sub.bio.lai, sfc_sub.bio.h_canopy, device=t_air.device, maml_n_sub=n_sub
            )
        is_maml_lc = int(model.land_carbon.c_leaf.shape[0]) == m_sub
        if is_maml_lc:
            resp_leaf_sub = (
                sfc_diag_sub.resp_leaf
                if sfc_diag_sub.resp_leaf is not None
                else torch.zeros_like(sfc_diag_sub.gpp)
            )
            f4_sub = torch.clamp(sfc_sub.bio.f_c4, 0.0, 1.0)
            vc25_eff_sub = (1.0 - f4_sub) * sfc_sub.bio.vcmax25_c3 + f4_sub * sfc_sub.bio.vcmax25_c4
            step_land_carbon_and_fire(
                lc=model.land_carbon,
                surf_type=sfc_sub.surf_type,
                gpp_umol=sfc_diag_sub.gpp,
                resp_leaf_umol=resp_leaf_sub,
                lai=sfc_sub.bio.lai,
                h_canopy=sfc_sub.bio.h_canopy,
                t_soil=sfc_sub.t_soil,
                w_liq_soil=sfc_sub.w_liq_soil,
                w_ice_soil=sfc_sub.w_ice_soil,
                f_snow=sfc_diag_sub.f_snow_land,
                t_sfc_k=t_air_sub,
                vpd_pa=vpd_sub,
                u10_ms=torch.hypot(u_sub_0, v_sub_0),
                precip_kgm2s=rain_new_sub + snow_new_sub,
                dt=dt,
                enable_nitrogen=bool(cfg.enable_land_nitrogen),
                vcmax25_eff=vc25_eff_sub,
                f_evg=sfc_sub.bio.f_evg,
                transp_kgm2s=torch.clamp(sfc_diag_sub.evap, min=0.0),
                n_dep_kgn_m2s=_c2s(light_diag.n_dep_lnox_sub.reshape(m_crm)) if light_diag is not None else None,
                flash_cg_m2s=_c2s(light_diag.flash_cg_sub.reshape(m_crm)) if light_diag is not None else None,
                bio=sfc_sub.bio,
                enable_fire_veg_feedback=bool(cfg.enable_fire_veg_feedback),
                maml_shape=(nc, ny_t, nx_t),
                crm_dx_m=float(cfg.crm_dx) * math.sqrt(float(by_full * bx_full)),
            )
            # Mirror updated MAML LAI / canopy height / N availability back to host summary
            sfc.bio.lai = _sub_to_cell(sfc_sub.bio.lai, nc, n_sub)
            sfc.bio.h_canopy = _sub_to_cell(sfc_sub.bio.h_canopy, nc, n_sub)
            if sfc_sub.bio.lai_evg is not None:
                sfc.bio.lai_evg = _sub_to_cell(sfc_sub.bio.lai_evg, nc, n_sub)
            if sfc_sub.bio.lai_dec is not None:
                sfc.bio.lai_dec = _sub_to_cell(sfc_sub.bio.lai_dec, nc, n_sub)
            if sfc_sub.bio.n_avail_fac is not None:
                sfc.bio.n_avail_fac = _sub_to_cell(sfc_sub.bio.n_avail_fac, nc, n_sub)
        else:
            lai_c = _sub_to_cell(sfc_sub.bio.lai, nc, n_sub)
            hc_c = _sub_to_cell(sfc_sub.bio.h_canopy, nc, n_sub)
            resp_leaf_c = (
                _sub_to_cell(sfc_diag_sub.resp_leaf, nc, n_sub)
                if sfc_diag_sub.resp_leaf is not None
                else torch.zeros_like(gpp_cell)
            )
            f4_c = torch.clamp(_sub_to_cell(sfc_sub.bio.f_c4, nc, n_sub), 0.0, 1.0)
            vc25_eff_c = (1.0 - f4_c) * _sub_to_cell(sfc_sub.bio.vcmax25_c3, nc, n_sub) + f4_c * _sub_to_cell(sfc_sub.bio.vcmax25_c4, nc, n_sub)
            step_land_carbon_and_fire(
                lc=model.land_carbon,
                surf_type=sfc.surf_type,
                gpp_umol=gpp_cell,
                resp_leaf_umol=resp_leaf_c,
                lai=lai_c,
                h_canopy=hc_c,
                t_soil=_sub_to_cell(sfc_sub.t_soil, nc, n_sub),
                w_liq_soil=_sub_to_cell(sfc_sub.w_liq_soil, nc, n_sub),
                w_ice_soil=_sub_to_cell(sfc_sub.w_ice_soil, nc, n_sub),
                f_snow=_sub_to_cell(sfc_diag_sub.f_snow_land, nc, n_sub),
                t_sfc_k=t_air,
                vpd_pa=_sub_to_cell(vpd_sub, nc, n_sub),
                u10_ms=_sub_to_cell(torch.hypot(u_sub_0, v_sub_0), nc, n_sub),
                precip_kgm2s=precip_tot,
                dt=dt,
                enable_nitrogen=bool(cfg.enable_land_nitrogen),
                vcmax25_eff=vc25_eff_c,
                f_evg=_sub_to_cell(sfc_sub.bio.f_evg, nc, n_sub) if sfc_sub.bio.f_evg is not None else None,
                transp_kgm2s=torch.clamp(evap_cell, min=0.0),
                n_dep_kgn_m2s=light_diag.n_dep_lnox_cell if light_diag is not None else None,
                flash_cg_m2s=light_diag.flash_cg_cell if light_diag is not None else None,
                bio=None,
                enable_fire_veg_feedback=False,
                maml_shape=None,
                crm_dx_m=float(cfg.crm_dx),
            )
    if bool(cfg.enable_natural_aerosols):
        from aoba.aerosols import init_natural_aerosol_state, step_natural_aerosols_column

        if model.aerosols is None:
            model.aerosols = init_natural_aerosol_state(nz, nc, device=atmos.q_n.device, dtype=atmos.q_n.dtype)
        chl_sfc = (
            model.ogcm.bgc.chl[0]
            if (model.ogcm is not None and model.ogcm.bgc is not None)
            else None
        )
        if chl_sfc is not None and has_indep_ocn:
            from aoba.ocean_coupler import ocn_to_atm_mean

            chl_sfc = ocn_to_atm_mean(model.ocn_coupler, chl_sfc)
        if model.land_carbon is not None:
            if int(model.land_carbon.fire_bc_kg_m2s.shape[0]) == m_sub:
                fire_bc = _sub_to_cell(model.land_carbon.fire_bc_kg_m2s, nc, n_sub)
                fire_oc = _sub_to_cell(model.land_carbon.fire_oc_kg_m2s, nc, n_sub)
            else:
                fire_bc = model.land_carbon.fire_bc_kg_m2s
                fire_oc = model.land_carbon.fire_oc_kg_m2s
        else:
            fire_bc = None
            fire_oc = None
        soa_oc_cell = None
        if bool(cfg.enable_bvoc_soa):
            from aoba.aerosols import compute_bvoc_soa_emission

            sw_par_cell = C.PAR_FRAC_OF_SW1 * (
                model.rad_cache.get("sw1_dir", torch.zeros_like(t_air))
                + model.rad_cache.get("sw1_dif", torch.zeros_like(t_air))
            )
            soa_oc_cell = compute_bvoc_soa_emission(sfc.surf_type, sfc.bio.lai, sfc.t_canopy, sw_par_cell)
        step_natural_aerosols_column(
            aero=model.aerosols,
            rho_d=atmos.rho_d,
            dz_c=atmos.dz_c,
            surf_type=sfc.surf_type,
            sst_k=sfc.t_ocean_ml,
            u10_ms=_sub_to_cell(torch.hypot(u_sub_0, v_sub_0), nc, n_sub),
            f_seaice=f_seaice_cell,
            precip_rate=precip_tot,
            dt=dt,
            chl_sfc_mg_m3=chl_sfc,
            fire_bc_kg_m2s=fire_bc,
            fire_oc_kg_m2s=fire_oc,
            soa_oc_kg_m2s=soa_oc_cell,
            u_star_cell=_sub_to_cell(sfc_diag_sub.u_star, nc, n_sub),
        )
    # 8b-2. SNICAR Snow/Sea-Ice BC + Dust Albedo Darkening (Flanner et al., 2007, 2009)
    if bool(cfg.enable_snow_bc_darkening):
        from aoba.surface import step_snow_impurity_darkening

        bc_dep_c = model.aerosols.dep_bc_kgm2s if (model.aerosols is not None and getattr(model.aerosols, "dep_bc_kgm2s", None) is not None) else None
        step_snow_impurity_darkening(
            sfc_sub,
            bc_dep_kgm2s=_cell_to_sub(bc_dep_c, nc, n_sub) if bc_dep_c is not None else None,
            dust_dep_kgm2s=_cell_to_sub(f_dust_dep_cell, nc, n_sub) if f_dust_dep_cell is not None else None,
            snow_fall_kgm2s=snow_new_sub,
            dt=dt,
        )
    # 8b-3. Silicate Weathering Thermostat (Walker et al., 1981) -> Prognostic CO2 & Riverine Alkalinity
    alk_weath_cell = None
    if bool(cfg.enable_silicate_weathering):
        from aoba.planetary_geochem import init_weathering_state, step_silicate_weathering

        if model.weathering is None:
            model.weathering = init_weathering_state(model.current_co2_ppmv(), cfg.volcanic_co2_pgc_yr)
        runoff_liq_w = torch.clamp(_sub_to_cell(sfc_diag_sub.runoff_total - sfc_diag_sub.runoff_ice, nc, n_sub), min=0.0)
        step_silicate_weathering(
            ws=model.weathering,
            area_cell=mesh.area_cell,
            surf_type=sfc.surf_type,
            t_sfc_k=t_air,
            runoff_liq_kgm2s=runoff_liq_w,
            f_snow_land=_sub_to_cell(sfc_diag_sub.f_snow_land, nc, n_sub),
            dt=dt,
            volcanic_co2_pgc_yr=cfg.volcanic_co2_pgc_yr,
            accel_factor=cfg.weathering_accel_factor,
        )
        alk_weath_cell = model.weathering.alk_flux_mol_m2s
    # 8b-4. Land-Atmosphere-Ocean BGC Deposition (Dust Fe + LNOx NO3 + Weathering Alk -> FlexPFT)
    if (
        (bool(cfg.enable_bgc_fe_lnox_coupling) or alk_weath_cell is not None)
        and model.ogcm is not None
        and model.ogcm.bgc is not None
    ):
        from aoba.ocean_bgc import apply_atmospheric_bgc_deposition

        du_bgc = f_dust_dep_cell if bool(cfg.enable_bgc_fe_lnox_coupling) else None
        ln_bgc = (light_diag.n_dep_lnox_cell if light_diag is not None else None) if bool(cfg.enable_bgc_fe_lnox_coupling) else None
        alk_bgc = alk_weath_cell
        if has_indep_ocn:
            from aoba.ocean_coupler import atm_to_ocn_conservative_flux, atm_to_ocn_intensive

            if du_bgc is not None:
                du_bgc = atm_to_ocn_intensive(model.ocn_coupler, du_bgc)
            if ln_bgc is not None:
                ln_bgc = atm_to_ocn_intensive(model.ocn_coupler, ln_bgc)
            if alk_bgc is not None:
                alk_bgc = atm_to_ocn_conservative_flux(model.ocn_coupler, alk_bgc, ocean_only_source=False)
        elif alk_bgc is not None:
            # Spread land weathering alkalinity evenly onto wet ocean surface cells
            w0 = model.ogcm.wet_c[0].to(alk_bgc.dtype)
            a_ocn_sum = torch.clamp((mesh.area_cell * w0).sum(), min=1.0)
            alk_tot = (alk_bgc * mesh.area_cell).sum()
            alk_bgc = (alk_tot / a_ocn_sum) * w0
        apply_atmospheric_bgc_deposition(model.ogcm.bgc, model.ogcm, du_bgc, ln_bgc, alk_bgc, dt)
    # 8b-5. Prognostic Natural CH4 & N2O Cycles (Wetland/Permafrost + Fire + Denit/Marine-OMZ + OH/UV)
    if bool(cfg.enable_prognostic_ch4_n2o):
        from aoba.trace_gas_cycle import init_trace_gas_cycle_state, step_trace_gas_cycle

        if model.trace_gas is None:
            model.trace_gas = init_trace_gas_cycle_state(cfg.ch4_ppmv, cfg.n2o_ppmv)
        mean_tco_now = float(model.ozone.tco_du.mean().item()) if model.ozone is not None else 300.0
        ocn_m_tg = model.ocn_mesh if has_indep_ocn else mesh
        step_trace_gas_cycle(
            tg=model.trace_gas,
            area_cell=mesh.area_cell,
            surf_type=sfc.surf_type,
            t_sfc_k=t_air,
            w_liq_soil_top=sfc_sub.w_liq_soil[0],
            dt=dt,
            rh_kgc_m2s=model.land_carbon.rh_kgc_m2s if model.land_carbon is not None else None,
            fire_c_kgc_m2s=model.land_carbon.fire_c_kgc_m2s if model.land_carbon is not None else None,
            n_denit_kgn_m2s=getattr(model.land_carbon, "n_denit_kgn_m2s", None) if model.land_carbon is not None else None,
            ocean_epc100_molc_m2s=model.ogcm.bgc.epc100 if (model.ogcm is not None and model.ogcm.bgc is not None) else None,
            ocean_denit_moln_m2s=getattr(model.ogcm.bgc, "denit_moln_m2s", None) if (model.ogcm is not None and model.ogcm.bgc is not None) else None,
            ocean_area_cell=ocn_m_tg.area_cell if ocn_m_tg is not None else None,
            f_wetland_sub=model.maml_land.f_wetland if (bool(cfg.enable_maml_wetlands_lakes) and model.maml_land is not None) else None,
            mean_tco_du=mean_tco_now,
        )
    # 8b-6. Stratospheric Volcanic / SAI Aerosols (SO2 -> H2SO4)
    volc_sad_3d = None
    if bool(cfg.enable_volcanic_aerosols) and model.volcanic_solar is not None:
        from aoba.volcanic_solar import compute_volcanic_rad_optics_and_sad, step_volcanic_aerosols

        step_volcanic_aerosols(
            vs=model.volcanic_solar,
            mesh=mesh,
            sim_time_days=model.sim_time_sec / C.SECONDS_PER_DAY,
            dt=dt,
            eruption_year=cfg.volcanic_eruption_year,
            eruption_so2_tg=cfg.volcanic_so2_tg,
            eruption_lat_deg=cfg.volcanic_eruption_lat_deg,
            eruption_width_deg=cfg.volcanic_eruption_width_deg,
            sai_so2_tg_yr=cfg.sai_so2_tg_yr,
            sai_lat_band_deg=cfg.sai_lat_band_deg,
        )
        pres_host_vs, _, _ = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
        _, volc_sad_3d = compute_volcanic_rad_optics_and_sad(model.volcanic_solar, pres_host_vs, atmos.dz_c)
    # 8b-7. Emissions-Driven Interactive Prognostic Atmospheric CO2 (Closed Land+Ocean+Weath+Anthro Budget)
    if str(cfg.co2_mode).strip().lower() == "prognostic":
        from aoba.prognostic_co2 import init_prognostic_co2_state, step_prognostic_co2

        if model.prognostic_co2 is None:
            model.prognostic_co2 = init_prognostic_co2_state(
                co2_ppmv_init=cfg.co2_ppmv,
                nz=nz,
                n_cells=nc,
                enable_3d=bool(cfg.enable_co2_3d_transport),
                device=atmos.rho_d.device,
                dtype=atmos.rho_d.dtype,
            )
        ocn_m_c = model.ocn_mesh if has_indep_ocn else mesh
        fg_ocn_c = model.ogcm.bgc.fgco2 if (model.ogcm is not None and model.ogcm.bgc is not None) else None
        fg_host_c = None
        if fg_ocn_c is not None and has_indep_ocn and model.ocn_coupler is not None:
            from aoba.ocean_coupler import ocn_to_atm_mean

            fg_host_c = ocn_to_atm_mean(model.ocn_coupler, fg_ocn_c)
        weath_sink = float(model.weathering.weathering_pgc_yr) if (bool(cfg.enable_silicate_weathering) and model.weathering is not None) else 0.0
        volc_src = float(cfg.volcanic_co2_pgc_yr) if bool(cfg.enable_silicate_weathering) else 0.0
        step_prognostic_co2(
            state=model.prognostic_co2,
            area_cell=mesh.area_cell,
            dt=dt,
            land_nbp_kgc_m2s=model.land_carbon.nbp_kgc_m2s if model.land_carbon is not None else None,
            ocean_fgco2_molc_m2s=fg_ocn_c,
            ocean_area_cell=ocn_m_c.area_cell if ocn_m_c is not None else None,
            ocean_fgco2_host_molc_m2s=fg_host_c,
            anthro_pgc_yr=float(cfg.co2_anthro_pgc_yr),
            volcanic_pgc_yr=volc_src,
            weathering_pgc_yr=weath_sink,
            rho_d=atmos.rho_d,
            dz_c=atmos.dz_c,
            lat_cell=mesh.lat_cell,
        )
    if str(cfg.o3_mode).strip().lower() == "prognostic":
        from aoba.ozone_chemistry import init_prognostic_ozone_state, step_prognostic_ozone

        pres_host_oz, _, temp_host_oz = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
        if model.ozone is None:
            model.ozone = init_prognostic_ozone_state(pres_host_oz, atmos.rho_d, atmos.dz_c)
        lnox_prod_cell = None
        if light_diag is not None:
            from aoba.lightning import LNOX_KG_N_PER_FLASH

            lnox_prod_cell = light_diag.flash_tot_cell * LNOX_KG_N_PER_FLASH
        fire_c_cell = None
        if model.land_carbon is not None:
            if int(model.land_carbon.fire_c_kgc_m2s.shape[0]) == m_sub:
                fire_c_cell = _sub_to_cell(model.land_carbon.fire_c_kgc_m2s, nc, n_sub)
            else:
                fire_c_cell = model.land_carbon.fire_c_kgc_m2s
        step_prognostic_ozone(
            oz=model.ozone,
            pres_c=pres_host_oz,
            temp_c=temp_host_oz,
            rho_d=atmos.rho_d,
            dz_c=atmos.dz_c,
            q_v=atmos.q_v,
            mu0=mu0,
            surf_type=sfc.surf_type,
            lai_cell=sfc.bio.lai,
            f_snow_cell=_sub_to_cell(sfc_diag_sub.f_snow_land, nc, n_sub),
            f_seaice_cell=f_seaice_cell,
            dt=dt,
            n2o_ppmv=model.current_n2o_ppmv(),
            ch4_ppmv=model.current_ch4_ppmv(),
            ecl_ppbv=cfg.o3_ecl_ppbv,
            lnox_cell_kgn_m2s=lnox_prod_cell,
            fire_c_cell_kgc_m2s=fire_c_cell,
            uv_j_o2_scale=uv_j_o2_scale,
            volc_sad_um2_cm3=volc_sad_3d,
        )
    # 8b-8. Stable Water Isotopes (H2^18O, HDO) Fractionation & Ice-Core Archiving
    if bool(cfg.enable_water_isotopes):
        from aoba.water_isotopes import init_water_isotope_state, step_water_isotopes

        if model.water_isotopes is None:
            model.water_isotopes = init_water_isotope_state(atmos.q_v)
        _, _, temp_host_iso = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
        step_water_isotopes(
            iso=model.water_isotopes,
            q_v=atmos.q_v,
            temp_c=temp_host_iso,
            rho_d=atmos.rho_d,
            dz_c=atmos.dz_c,
            surf_type=sfc.surf_type,
            t_sfc=t_rad_cell,
            evap_cell=evap_cell,
            rain_cell=rain_new,
            snow_cell=snow_new,
            dt=dt,
        )
    month_idx = model.calendar.date_at(model.sim_time_sec).month - 1
    clim.month_temp_sum_c[month_idx] += (t_air - C.T_FREEZE)
    clim.month_precip_mm[month_idx] += precip_tot * dt
    clim.month_steps[month_idx] += 1.0
    clim.month_days_accum[month_idx] += dt / C.SECONDS_PER_DAY

    steps_per_day = max(int(round(C.SECONDS_PER_DAY / dt)), 1)
    if clim_sub.day_steps >= steps_per_day:
        # Apply daily Smith/Scott EEO enzyme acclimation in ONE vectorized batch using exact local solar-noon snapshots
        noon_buf = getattr(clim_sub, "_noon_buf", None)
        noon_seen = getattr(clim_sub, "_noon_seen", None)
        if noon_buf is not None and noon_seen is not None:
            noon_land_idx = torch.nonzero(noon_seen & (sfc_sub.surf_type == SURF_LAND), as_tuple=False).squeeze(-1)
            if int(noon_land_idx.numel()) > 0:
                _acclim_read_attrs = (
                    "chi_leaf", "ema_ann_pet", "ema_ann_precip", "ema_mu0_mid",
                    "ema_temp_home", "ema_temp_mid", "ema_temp_night",
                    "ema_vis_beam", "ema_vis_diff", "ema_vpd_mid", "f_c4",
                    "jmax25_c3", "jmax25_c4", "lai", "r25_c3", "r25_c4",
                    "vcmax25_c3", "vcmax25_c4", "vpmax25_c4",
                )
                bio_d = dict(sfc_sub.bio.__dict__)
                for attr in _acclim_read_attrs:
                    val = bio_d.get(attr)
                    if isinstance(val, torch.Tensor):
                        bio_d[attr] = val.index_select(0, noon_land_idx)
                sub_bio = BiosphereState(**bio_d)
                nb = noon_buf.index_select(1, noon_land_idx)
                update_daily_acclimation_smith(
                    bio=sub_bio,
                    midday_vis_beam=nb[0],
                    midday_vis_diff=nb[1],
                    midday_mu0=nb[2],
                    midday_temp=nb[3],
                    midday_vpd=nb[4],
                    night_temp=nb[5],
                    pres_sfc=nb[6],
                    co2_ppmv=model.current_co2_ppmv(),
                    lai_exposed=nb[7],
                    is_vegetated_land=torch.ones(int(noon_land_idx.numel()), device=nb.device, dtype=torch.bool),
                )
                for a in _ACCLIM_BIO_ATTRS:
                    getattr(sfc_sub.bio, a).index_copy_(0, noon_land_idx, getattr(sub_bio, a))
            noon_seen.zero_()
        ws_beta_sub = compute_optimal_soil_water_beta(sfc_sub.w_root, sfc_sub.bio)
        daily_t_mean_sub = clim_sub.day_temp_sum / float(clim_sub.day_steps)
        daily_t_min_sub = torch.where(clim_sub.day_t_min < 399.0, clim_sub.day_t_min, daily_t_mean_sub)
        update_daily_lai_root_height(
            bio=sfc_sub.bio,
            daily_precip_mm=clim_sub.day_precip_mm,
            daily_pet_mm=torch.clamp(clim_sub.day_pet_mm, min=0.05),
            daily_ppfd_mol_m2=clim_sub.day_ppfd_mol,
            daily_temp_k=daily_t_mean_sub,
            daily_vpd_pa=clim_sub.day_vpd_sum / float(clim_sub.day_steps),
            pres_sfc=pres_sfc_post_sub,
            co2_ppmv=model.current_co2_ppmv(),
            water_stress_beta=ws_beta_sub,
            is_vegetated_land=(sfc_sub.surf_type == SURF_LAND),
            daily_t_min_k=daily_t_min_sub,
        )
        clim_sub.day_precip_mm.zero_()
        clim_sub.day_pet_mm.zero_()
        clim_sub.day_ppfd_mol.zero_()
        clim_sub.day_temp_sum.zero_()
        clim_sub.day_vpd_sum.zero_()
        clim_sub.day_t_min.fill_(400.0)
        clim_sub.day_steps = 0

    if sync_host_surface:
        _sync_host_surface_from_subgrid(sfc, sfc_sub, nc, n_sub)

    model.sim_time_sec += dt
    model.step_count += 1

    # Dynamic / EMA-accelerated 2D SIA continental ice sheet & conservative vertical coordinate update
    if model.icesheet is not None and sfc_diag_sub.smb_ice_flux is not None:
        model.icesheet.smb_year_accum_kgm2 = model.icesheet.smb_year_accum_kgm2 + _sub_to_cell(sfc_diag_sub.smb_ice_flux, nc, n_sub) * dt
        model.icesheet.year_elapsed_sec += dt
        if model.icesheet.year_elapsed_sec >= C.DAYS_PER_YEAR * C.SECONDS_PER_DAY - 0.5 * dt:
            _apply_dynamic_icesheet_update(model)
            _sync_host_surface_from_subgrid(sfc, sfc_sub, nc, n_sub)

    if not compute_diagnostics:
        return GlobalStepDiagnostics(
            sim_time_days=model.sim_time_sec / C.SECONDS_PER_DAY,
            global_mean_t_sfc_k=0.0,
            global_mean_olr_wm2=0.0,
            global_mean_asr_wm2=0.0,
            global_mean_precip_mm_day=0.0,
            land_mean_gpp_umol=0.0,
            land_mean_lai=0.0,
            seaice_area_million_km2=0.0,
            date_str=model.calendar.date_at(model.sim_time_sec).iso_date,
        )

    # Compute area-weighted global diagnostics with a single batched GPU->CPU sync
    area = mesh.area_cell
    area_tot = area.sum()
    land_mask = (sfc.surf_type == SURF_LAND).to(area.dtype)
    land_area = torch.clamp((area * land_mask).sum(), min=1.0)
    lai_cell = sfc.bio.lai if sync_host_surface else _sub_to_cell(sfc_sub.bio.lai, nc, n_sub)

    diag_vec = torch.stack([
        (area * t_rad_cell).sum() / area_tot,
        (area * olr).sum() / area_tot,
        (area * asr).sum() / area_tot,
        (area * precip_tot * C.SECONDS_PER_DAY).sum() / area_tot,
        (area * land_mask * gpp_cell).sum() / land_area,
        (area * land_mask * lai_cell).sum() / land_area,
        (area * f_seaice_cell).sum() * 1.0e-12,
    ]).cpu().tolist()
    t_sfc_mean, olr_mean, asr_mean, p_mm_day, gpp_land, lai_land, seaice_mkm2 = diag_vec

    return GlobalStepDiagnostics(
        sim_time_days=model.sim_time_sec / C.SECONDS_PER_DAY,
        global_mean_t_sfc_k=t_sfc_mean,
        global_mean_olr_wm2=olr_mean,
        global_mean_asr_wm2=asr_mean,
        global_mean_precip_mm_day=p_mm_day,
        land_mean_gpp_umol=gpp_land,
        land_mean_lai=lai_land,
        seaice_area_million_km2=seaice_mkm2,
        date_str=model.calendar.date_at(model.sim_time_sec).iso_date,
    )


def diagnose_model_koppen_geiger(model: AOBAState) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnose emergent 30-class & 5-major-group Köppen-Geiger maps from accumulated monthly means."""
    clim = model.clim
    steps_safe = torch.clamp(clim.month_steps, min=1.0)
    days_safe = torch.clamp(clim.month_days_accum, min=1.0e-3)
    t_mon_c = clim.month_temp_sum_c / steps_safe
    # Normalize accumulated precipitation to standard 30.437-day month equivalent [mm month^-1]
    p_mon_mm = clim.month_precip_mm * ((C.DAYS_PER_YEAR / 12.0) / days_safe)
    return classify_koppen_geiger(t_mon_c, p_mon_mm, model.mesh.lat_cell)
