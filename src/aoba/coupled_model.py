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
    compute_optimal_soil_water_beta,
    update_daily_acclimation_smith,
    update_daily_lai_root_height,
)
from aoba.crm import CRMState, SPTendencies, step_crm_physics_and_dynamics
from aoba.dust import compute_dust_emission_sub, step_dust_column
from aoba.icesheet import (
    H_ICE_THRESHOLD_M,
    IceSheetState,
    init_icesheet_state,
    remap_atmosphere_and_crm_vertical,
    step_sia_icesheet_scvt,
)
from aoba.koppen import classify_koppen_geiger
from aoba.ocean_ekman import EkmanOceanState, init_ekman_ocean_state, step_ekman_ocean_voronoi
from aoba.orography import compute_orographic_drag_lott_miller
from aoba.radiation import compute_four_band_radiation
from aoba.scvt_dynamics import (
    GlobalAtmosState,
    fct_advect_scvt,
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
from aoba.turbulence import TKE_MIN
from aoba.vertical_grid import VerticalGrid, build_vertical_grid


@dataclass
class AOBAConfig:
    """Configuration for a global 1:1 Superparameterized AOBA simulation."""

    glevel: int = 3                          # SCVT mesh level (3 -> 642 cells; 4 -> 2562; 5 -> 10242)
    nz: int = 32                             # Shared Charney-Phillips vertical levels
    z_top: float = 24000.0                   # Model top height [m]
    dz_min: float = 60.0                     # Lowest layer thickness [m]
    z_sponge_base: float = 17500.0           # Rayleigh sponge base height [m]
    tau_sponge_min: float = 300.0            # Minimum sponge damping timescale [s]
    terrain_scale_exp: float = 1.5           # Klemp STF attenuation exponent p [-]
    h_ml_ocean: float = 50.0                 # Ocean mixed-layer depth [m]
    re_liq_um: float = 10.0                  # Cloud liquid droplet effective radius [um]
    re_ice_um: float = 48.0                  # Cloud ice crystal effective radius [um]
    cloud_sw_inhomog: float = 0.85           # Subgrid SW cloud optical inhomogeneity factor chi_SW (Cahalan et al., 1994) [-]
    crm_nx: int = 8                          # Embedded CRM x-grid points per SCVT cell (default 2D 8x1 = 32 km curtain)
    crm_ny: int = 1                          # Embedded CRM y-grid points (1 = SP-CAM 2.5D curtain for Mac climate runs)
    crm_dx: float = 4000.0                   # Embedded CRM horizontal spacing [m] (4 km resolves convective cores cleanly)
    dt_host: float = 720.0                   # Coupled host / physics / SP time step [s] (12 min climate coupling)
    n_host_dyn_substeps: int = 1             # Minimum host RK3-HEVI substeps per dt_host (auto-increased if CFL requires)
    n_crm_dyn_substeps: int = 18             # Embedded CRM RK3-HEVI substeps per dt_host (18 x 40 s = 720 s at dx=4 km)
    rad_stride: int = 5                      # Radiation evaluation stride (steps; 5 x 720 s = 3600 s = 1 hr)
    crm_col_chunk: int = 16384               # Column batch chunk size for CRM evaluation (Unified Memory single-pass)
    rad_col_chunk: int = 32768               # Subcolumn chunk size for ICA radiative transfer (caps MPS peak buffer)
    co2_ppmv: float = 400.0                  # Well-mixed atmospheric CO2 concentration [ppmv] (used when co2_mode="constant")
    co2_mode: str = "constant"               # "constant", "scenario", or "csv"
    co2_scenario: str = "obs_historical_noaa"  # Scenario key when co2_mode="scenario"
    co2_csv_path: str = ""                   # Custom (year, co2_ppmv) CSV path when co2_mode="csv"
    o3_mode: str = "rcemip"                  # "rcemip" (1D), "latitude_seasonal" (2D analytical), or "climatology"
    o3_file_path: str = ""                   # Optional external zonal/3D ozone (.npz/.nc) when o3_mode="climatology"
    o3_lw_coupling: bool = False             # Enable 9.6 um O3 absorption in LW2 atmospheric window
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
    enable_dust: bool = True                 # Map-free global-parameter mineral dust aerosol + radiative coupling
    dust_emis_factor: float = 1.5e-5         # Global dust emission tuning factor T (map-free S=1 CLM5 Eq. 31.1) [-]
    enable_dynamic_icesheet: bool = False    # Opt-in SIA + EMA-accelerated ice sheet & vertical grid update
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
    sfc_sub: SurfaceState | None = None
    clim_sub: ClimateAccumulator | None = None
    icesheet: IceSheetState | None = None
    sim_time_sec: float = 0.0
    step_count: int = 0
    rad_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    calendar: AobaCalendar = field(default_factory=AobaCalendar)
    co2_series: CO2ForcingSeries | None = None
    o3_provider: OzoneForcingProvider | None = None

    def current_co2_ppmv(self, sim_time_sec: float | None = None) -> float:
        """Return active atmospheric CO2 [ppmv] at ``sim_time_sec`` (or ``self.sim_time_sec``)."""
        if self.co2_series is None or not self.co2_series.is_time_varying:
            return float(self.cfg.co2_ppmv)
        t_sec = self.sim_time_sec if sim_time_sec is None else float(sim_time_sec)
        return self.co2_series.evaluate(self.calendar.decimal_year_at(t_sec))


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
    """Broadcast host `[nc]` `model.sfc` & `bio` across all CRM subcolumns in `model.sfc_sub`."""
    if model.sfc_sub is None:
        return
    nc = model.mesh.n_cells
    n_sub = model.cfg.crm_ny * model.cfg.crm_nx
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
    re_liq_um: float = 10.0,
    re_ice_um: float = 48.0,
    cloud_sw_inhomog: float = 0.85,
    q_dust_c: torch.Tensor | None = None,
    o3_vmr_c: torch.Tensor | None = None,
    o3_lw_coupling: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate 1D ICA 4-band radiative transfer across `[nz, m_sub]`, chunking when `m_sub > col_chunk`."""
    nz, m_sub = pres_c.shape
    chunk = max(col_chunk, 1)
    if chunk >= m_sub:
        rad = compute_four_band_radiation(
            pres_c=pres_c,
            pres_w=pres_w,
            temp_c=temp_c,
            temp_w=temp_w,
            dz_c=dz_c,
            rho_d_c=rho_d_c,
            qv_c=qv_c,
            qc_c=qc_c,
            qi_c=qi_c,
            t_sfc=t_sfc,
            albedo_vis=albedo_vis,
            albedo_nir=albedo_nir,
            mu0=mu0,
            s_toa=s_toa,
            co2_ppmv=co2_ppmv,
            emiss_sfc=emiss_sfc,
            re_liq_um=re_liq_um,
            re_ice_um=re_ice_um,
            cloud_sw_inhomog=cloud_sw_inhomog,
            albedo_vis_dif=albedo_vis_dif,
            albedo_nir_dif=albedo_nir_dif,
            q_dust_c=q_dust_c,
            o3_vmr_c=o3_vmr_c,
            o3_lw_coupling=o3_lw_coupling,
        )
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
        rad = compute_four_band_radiation(
            pres_c=pres_c[:, sl],
            pres_w=pres_w[:, sl],
            temp_c=temp_c[:, sl],
            temp_w=temp_w[:, sl],
            dz_c=dz_c[:, sl],
            rho_d_c=rho_d_c[:, sl],
            qv_c=qv_c[:, sl],
            qc_c=qc_c[:, sl],
            qi_c=qi_c[:, sl],
            t_sfc=t_sfc[sl],
            albedo_vis=albedo_vis[sl],
            albedo_nir=albedo_nir[sl],
            mu0=mu0[sl],
            s_toa=s_toa[sl],
            co2_ppmv=co2_ppmv,
            emiss_sfc=emiss_sfc[sl],
            re_liq_um=re_liq_um,
            re_ice_um=re_ice_um,
            cloud_sw_inhomog=cloud_sw_inhomog,
            albedo_vis_dif=albedo_vis_dif[sl],
            albedo_nir_dif=albedo_nir_dif[sl],
            q_dust_c=q_dust_c[:, sl] if q_dust_c is not None else None,
            o3_vmr_c=o3_vmr_c[:, sl] if o3_vmr_c is not None else None,
            o3_lw_coupling=o3_lw_coupling,
        )
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
    ny, nx = cfg.crm_ny, cfg.crm_nx
    n_sub = ny * nx
    m_sub = nc * n_sub
    shape4 = (nz, nc, ny, nx)

    # Initialize host-mean surface summary `[nc]` & subgrid MAML surface `[nc * ny * nx]`
    t_sfc_lat = 301.0 - 40.0 * (torch.sin(mesh.lat_cell) ** 2) - 0.0065 * torch.clamp(mesh.z_topo, min=0.0)
    sfc = init_surface_state(
        mesh.surf_type, z_topo=mesh.z_topo, t_init=293.15, sigma_topo=mesh.sigma_topo, h_ml_ocean=cfg.h_ml_ocean,
    )
    sfc.t_ocean_ml = torch.where(
        mesh.surf_type == SURF_OCEAN,
        torch.clamp(301.0 - 34.0 * (torch.sin(mesh.lat_cell) ** 2), min=C.T_FREEZE_OCEAN + 0.2),
        sfc.t_ocean_ml,
    )
    sfc.t_canopy = t_sfc_lat.clone()
    sfc.t_soil = t_sfc_lat.unsqueeze(0).expand_as(sfc.t_soil).clone()
    polar_cold = (mesh.surf_type == SURF_OCEAN) & (t_sfc_lat < C.T_FREEZE_OCEAN)
    sfc.m_seaice = torch.where(polar_cold, torch.full_like(t_sfc_lat, 900.0), torch.zeros_like(t_sfc_lat))

    ocn = init_ekman_ocean_state(sfc)

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
    noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)).to(device=dev, dtype=dtype)

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
        dx=cfg.crm_dx,
        dy=cfg.crm_dx,
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

    model = AOBAState(
        cfg=cfg,
        mesh=mesh,
        vgrid=vgrid,
        atmos=atmos,
        crm=crm,
        sfc=sfc,
        ocn=ocn,
        clim=_make_clim(nc, include_monthly=True),
        sfc_sub=sfc_sub,
        clim_sub=_make_clim(m_sub, include_monthly=False),
        icesheet=init_icesheet_state(mesh) if cfg.enable_dynamic_icesheet else None,
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
    return model


def _apply_dynamic_icesheet_update(model: AOBAState) -> None:
    """Advance 2D SIA ice sheet using annual EMA SMB, flip Land<->IceSheet masks, and remap vertical grid."""
    ice = model.icesheet
    sfc_sub = model.sfc_sub
    assert ice is not None and sfc_sub is not None
    cfg = model.cfg
    mesh = model.mesh
    nc = mesh.n_cells
    n_sub = cfg.crm_ny * cfg.crm_nx

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

    mesh.surf_type = new_surf_type
    mesh.z_topo = ice_diag["z_topo_new"]
    mesh.z_topo_smooth = ice_diag["z_topo_smooth_new"]
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
    "ema_vpd_mid", "ema_temp_night", "vcmax25_c3", "jmax25_c3", "r25_c3",
    "vcmax25_c4", "jmax25_c4", "vpmax25_c4", "r25_c4", "f_c4",
)


def step_aoba_model(model: AOBAState) -> GlobalStepDiagnostics:
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
    n_sub = ny * nx
    m_sub = nc * n_sub

    # 1. Orbital solar geometry & local solar noon crossings
    yr_sec_trop = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    t_seas_prev, t_diur_prev = model.calendar.orbital_times_sec(model.sim_time_sec, yr_sec_trop)
    t_seas_next, t_diur_next = model.calendar.orbital_times_sec(model.sim_time_sec + dt, yr_sec_trop)
    _, _, h_prev = compute_orbital_insolation(
        t_seas_prev, mesh.lat_cell, mesh.lon_cell,
        cfg.solar_constant, cfg.obliquity_deg, cfg.eccentricity, cfg.perihelion_deg,
        t_diurnal_sec=t_diur_prev, diurnal_cycle=cfg.diurnal_cycle,
    )
    mu0, s_toa, h_next = compute_orbital_insolation(
        t_seas_next, mesh.lat_cell, mesh.lon_cell,
        cfg.solar_constant, cfg.obliquity_deg, cfg.eccentricity, cfg.perihelion_deg,
        t_diurnal_sec=t_diur_next, diurnal_cycle=cfg.diurnal_cycle,
    )
    # Local solar noon crossed when h_prev < 0 <= h_next (excluding midnight wrap around +/-pi)
    crossed_noon = (h_prev < 0.0) & (h_next >= 0.0) & ((h_next - h_prev) < math.pi)
    mu0_sub = _cell_to_sub(mu0, nc, n_sub)
    s_toa_sub = _cell_to_sub(s_toa, nc, n_sub)
    crossed_noon_sub = _cell_to_sub(crossed_noon, nc, n_sub)

    # 2. Subcolumn CRM lowest-layer meteorology & host cell reference heights
    u_sub_0 = (0.5 * (crm.u[0] + torch.roll(crm.u[0], 1, dims=-1))).reshape(m_sub)
    v_sub_0 = (0.5 * (crm.v[0] + torch.roll(crm.v[0], 1, dims=-2))).reshape(m_sub)
    rho_sub = crm.rho_d.reshape(nz, m_sub)
    thm_sub = crm.theta_m.reshape(nz, m_sub)
    qv_sub = crm.q_v.reshape(nz, m_sub)
    qc_sub = crm.q_c.reshape(nz, m_sub)
    qi_sub = crm.q_i.reshape(nz, m_sub)

    pres_sub, _, temp_sub = diagnose_pressure_temperature(rho_sub, thm_sub, qv_sub)
    dz0_col = atmos.z_c[0] - atmos.z_w[0]
    dz0_sub = _cell_to_sub(dz0_col, nc, n_sub)
    pres_sfc_sub = pres_sub[0] + C.GRAVITY * rho_sub[0] * (1.0 + qv_sub[0]) * dz0_sub
    z_rel_c = atmos.z_c - atmos.z_w[0:1]
    z_ref_rel_sub = torch.clamp(_cell_to_sub(z_rel_c[0], nc, n_sub) - 0.65 * sfc_sub.bio.h_canopy, min=5.0)

    # 3. Subcolumn surface albedos & ICA 4-band radiative transfer across all `[Nz, m_sub]`
    alb_vd, alb_vf, alb_nd, alb_nf, _, lai_exp_sub, _, f_ice_sub = compute_surface_albedos(sfc_sub, mu0_sub)
    if (not model.rad_cache) or ("q_rad_sub" not in model.rad_cache) or (model.step_count % max(cfg.rad_stride, 1) == 0):
        t_rad_ocn = ((1.0 - f_ice_sub) * (sfc_sub.t_ocean_ml ** 4) + f_ice_sub * (torch.clamp(sfc_sub.t_seaice, min=150.0) ** 4)) ** 0.25
        t_rad_in = torch.where(sfc_sub.surf_type == SURF_OCEAN, t_rad_ocn, sfc_sub.t_canopy)
        emiss_in = torch.where(
            sfc_sub.surf_type == SURF_OCEAN,
            torch.full_like(t_rad_in, EMISSIVITY_OCEAN),
            torch.full_like(t_rad_in, EMISSIVITY_LAND),
        )
        pres_w_sub = torch.empty((nz + 1, m_sub), device=pres_sub.device, dtype=pres_sub.dtype)
        pres_w_sub[0] = pres_sfc_sub
        pres_w_sub[1:-1] = torch.sqrt(torch.clamp(pres_sub[:-1] * pres_sub[1:], min=1.0))
        pres_w_sub[-1] = torch.clamp(0.5 * pres_sub[-1], min=10.0)

        temp_w_sub = torch.empty((nz + 1, m_sub), device=temp_sub.device, dtype=temp_sub.dtype)
        temp_w_sub[0] = temp_sub[0]
        temp_w_sub[1:-1] = 0.5 * (temp_sub[:-1] + temp_sub[1:])
        temp_w_sub[-1] = temp_sub[-1]

        if cfg.enable_dust and atmos.q_dust is None:
            atmos.q_dust = torch.zeros_like(atmos.q_n)
        q_dust_rad = _cell_to_sub(atmos.q_dust, nc, n_sub) if (cfg.enable_dust and atmos.q_dust is not None) else None
        o3_vmr_rad = None
        if model.o3_provider is not None and model.o3_provider.mode != "rcemip":
            doy_now = float(model.calendar.date_at(model.sim_time_sec).doy)
            o3_vmr_rad = model.o3_provider.compute_vmr(
                pres_sub,
                lat_rad=_cell_to_sub(mesh.lat_cell, nc, n_sub),
                doy=doy_now,
            )
        q_rad_sub, sw1_dir_sub, sw1_dif_sub, sw2_dir_sub, sw2_dif_sub, lw_dn_sub, olr_sub, asr_sub = (
            _compute_four_band_radiation_chunked(
                pres_c=pres_sub,
                pres_w=pres_w_sub,
                temp_c=temp_sub,
                temp_w=temp_w_sub,
                dz_c=_cell_to_sub(atmos.dz_c, nc, n_sub),
                rho_d_c=rho_sub,
                qv_c=qv_sub,
                qc_c=qc_sub,
                qi_c=qi_sub,
                t_sfc=t_rad_in,
                albedo_vis=alb_vd,
                albedo_nir=alb_nd,
                mu0=mu0_sub,
                s_toa=s_toa_sub,
                co2_ppmv=model.current_co2_ppmv(),
                emiss_sfc=emiss_in,
                albedo_vis_dif=alb_vf,
                albedo_nir_dif=alb_nf,
                col_chunk=cfg.rad_col_chunk,
                re_liq_um=cfg.re_liq_um,
                re_ice_um=cfg.re_ice_um,
                cloud_sw_inhomog=cfg.cloud_sw_inhomog,
                q_dust_c=q_dust_rad,
                o3_vmr_c=o3_vmr_rad,
                o3_lw_coupling=cfg.o3_lw_coupling,
            )
        )
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

    # 4. Subgrid MAML Land/Snow/Soil/Two-Leaf Biosphere & Ocean/Sea-Ice Balance + Host Voronoi Ekman Ocean
    sfc_diag_sub = step_surface(
        sfc=sfc_sub,
        u_1=u_sub_0,
        v_1=v_sub_0,
        temp_1=temp_sub[0],
        qv_1=qv_sub[0],
        pres_1=pres_sub[0],
        pres_sfc=pres_sfc_sub,
        z_ref_rel=z_ref_rel_sub,
        sw1_dir_down=sw1_dir_sub,
        sw1_dif_down=sw1_dif_sub,
        sw2_dir_down=sw2_dir_sub,
        sw2_dif_down=sw2_dif_sub,
        lw_down=lw_dn_sub,
        precip_rain=rain_prev_sub,
        precip_snow=snow_prev_sub,
        mu0=mu0_sub,
        co2_ppmv=model.current_co2_ppmv(),
        dt=dt,
    )

    is_ocn_sub = sfc_sub.surf_type == SURF_OCEAN
    t1_cell_before = _sub_to_cell(sfc_sub.t_ocean_ml, nc, n_sub)
    m_ice_cell_before = _sub_to_cell(sfc_sub.m_seaice, nc, n_sub)
    sfc.t_ocean_ml = t1_cell_before.clone()
    sfc.m_seaice = m_ice_cell_before.clone()
    sfc.h_ocean_ml = _sub_to_cell(sfc_sub.h_ocean_ml, nc, n_sub)
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
    t1_sub_next = sfc_sub.t_ocean_ml + dt1_sub
    freeze_def = torch.where(is_ocn_sub, torch.clamp(C.T_FREEZE_OCEAN - t1_sub_next, min=0.0), torch.zeros_like(t1_sub_next))
    c1_sub = C.CP_SEAWATER * C.RHO_SEAWATER * sfc_sub.h_ocean_ml
    sfc_sub.t_ocean_ml = torch.where(is_ocn_sub, t1_sub_next + freeze_def, sfc_sub.t_ocean_ml)
    sfc_sub.m_seaice = torch.where(
        is_ocn_sub,
        torch.clamp(sfc_sub.m_seaice + dm_ice_sub + (c1_sub * freeze_def) / C.L_F0, min=0.0),
        sfc_sub.m_seaice,
    )

    # 5. Lott & Miller (1997) Subgrid Orographic Drag
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
    du_oro_edge = project_cell_vectors_to_edges(mesh, du_oro_c, dv_oro_c)

    # 6. Host SCVT Nonhydrostatic RK3-HEVI Dynamical Core & Voronoi FCT
    thm_before_dyn = atmos.theta_m.clone()
    qn_before_dyn = atmos.q_n.clone()

    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    dt_adv_host = min(900.0, 0.40 * dc_min / 80.0)
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
    atmos.q_n = fct_advect_scvt(mesh, atmos.q_n, m_old_host, m_new_host, uh_acc * inv_nh, om_acc * inv_nh, dt)
    atmos.q_p = fct_advect_scvt(mesh, atmos.q_p, m_old_host, m_new_host, uh_acc * inv_nh, om_acc * inv_nh, dt)
    if cfg.enable_dust and atmos.q_dust is not None:
        atmos.q_dust = fct_advect_scvt(mesh, atmos.q_dust, m_old_host, m_new_host, uh_acc * inv_nh, om_acc * inv_nh, dt)

    dthm_ls = (atmos.theta_m - thm_before_dyn) / dt
    dqn_ls = (atmos.q_n - qn_before_dyn) / dt
    u_host_new, v_host_new = reconstruct_cell_winds(mesh, atmos.u_edge)

    # Align subcolumn drag with post-dynamics lowest-layer flow shift while retaining subgrid gusts
    u_sub_post = u_sub_0 + _cell_to_sub(u_host_new[0] - u_cell[0], nc, n_sub)
    v_sub_post = v_sub_0 + _cell_to_sub(v_host_new[0] - v_cell[0], nc, n_sub)
    old_spd_sub = torch.sqrt(u_sub_0 * u_sub_0 + v_sub_0 * v_sub_0 + 1.0e-4)
    tau_mag_sub = torch.sqrt(sfc_diag_sub.tau_x * sfc_diag_sub.tau_x + sfc_diag_sub.tau_y * sfc_diag_sub.tau_y)
    drag_c_sub = tau_mag_sub / old_spd_sub
    sfc_diag_sub.tau_x = -drag_c_sub * u_sub_post
    sfc_diag_sub.tau_y = -drag_c_sub * v_sub_post

    # 7. 1:1 Embedded 3D CRM Physics & Convective Transport with Subgrid Surface & ICA Radiation
    rain_new_sub = torch.zeros_like(pres_sfc_sub)
    snow_new_sub = torch.zeros_like(pres_sfc_sub)
    q_rad_4d = q_rad_sub.reshape(nz, nc, ny, nx)

    if cfg.enable_superparam:
        du_crm_c = torch.zeros_like(u_host_new)
        dv_crm_c = torch.zeros_like(v_host_new)
        sfc_diag_3d = {k: v.reshape(nc, ny, nx) for k, v in sfc_diag_sub.__dict__.items()}

        chunk = max(cfg.crm_col_chunk, 1)
        if chunk >= nc:
            drho_mean = (atmos.rho_d - crm.rho_d.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
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
                sfc_fluxes=SurfaceFluxDiagnostics(**sfc_diag_3d),
                rad_heating_k_s=q_rad_4d,
                z_c_col=atmos.z_c,
                z_w_col=atmos.z_w,
                dz_c_col=atmos.dz_c,
            )
            atmos.theta_m = crm.theta_m.mean(dim=(-2, -1))
            atmos.q_n = crm.q_n.mean(dim=(-2, -1))
            atmos.q_p = crm.q_p.mean(dim=(-2, -1))
            atmos.q_v = crm.q_v.mean(dim=(-2, -1))
            atmos.q_c = crm.q_c.mean(dim=(-2, -1))
            atmos.q_i = crm.q_i.mean(dim=(-2, -1))
            du_crm_c = (crm.u.mean(dim=(-2, -1)) - u_host_new) / dt
            dv_crm_c = (crm.v.mean(dim=(-2, -1)) - v_host_new) / dt
            if sp.precip_rain_sub is not None and sp.precip_snow_sub is not None:
                rain_new_sub = sp.precip_rain_sub.reshape(m_sub)
                snow_new_sub = sp.precip_snow_sub.reshape(m_sub)
            else:
                rain_new_sub = _cell_to_sub(sp.precip_rain, nc, n_sub)
                snow_new_sub = _cell_to_sub(sp.precip_snow, nc, n_sub)
        else:
            rain_3d = rain_new_sub.reshape(nc, ny, nx)
            snow_3d = snow_new_sub.reshape(nc, ny, nx)
            for i0 in range(0, nc, chunk):
                i1 = min(i0 + chunk, nc)
                sl = slice(i0, i1)
                rho_crm_sl = crm.rho_d[:, sl]
                drho_mean = (atmos.rho_d[:, sl] - rho_crm_sl.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
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
                )
                sub_sfc = SurfaceFluxDiagnostics(**{k: v[sl] for k, v in sfc_diag_3d.items()})
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

                atmos.theta_m[:, sl] = sub_crm.theta_m.mean(dim=(-2, -1))
                atmos.q_n[:, sl] = sub_crm.q_n.mean(dim=(-2, -1))
                atmos.q_p[:, sl] = sub_crm.q_p.mean(dim=(-2, -1))
                atmos.q_v[:, sl] = sub_crm.q_v.mean(dim=(-2, -1))
                atmos.q_c[:, sl] = sub_crm.q_c.mean(dim=(-2, -1))
                atmos.q_i[:, sl] = sub_crm.q_i.mean(dim=(-2, -1))

                du_crm_c[:, sl] = (sub_crm.u.mean(dim=(-2, -1)) - u_host_new[:, sl]) / dt
                dv_crm_c[:, sl] = (sub_crm.v.mean(dim=(-2, -1)) - v_host_new[:, sl]) / dt
                if sp.precip_rain_sub is not None and sp.precip_snow_sub is not None:
                    rain_3d[sl] = sp.precip_rain_sub
                    snow_3d[sl] = sp.precip_snow_sub
                else:
                    rain_3d[sl] = sp.precip_rain.unsqueeze(-1).unsqueeze(-1)
                    snow_3d[sl] = sp.precip_snow.unsqueeze(-1).unsqueeze(-1)

        du_sp_edge = project_cell_vectors_to_edges(mesh, du_crm_c, dv_crm_c)
        atmos.u_edge = atmos.u_edge + dt * du_sp_edge

    rain_new = _sub_to_cell(rain_new_sub, nc, n_sub)
    snow_new = _sub_to_cell(snow_new_sub, nc, n_sub)
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
            rho_atm=rho_sub[0] * (1.0 + qv_sub[0]),
            z0m=sfc_diag_sub.z0m,
            d_disp=sfc_diag_sub.d_disp,
            dust_emis_factor=cfg.dust_emis_factor,
        )
        atmos.q_dust = step_dust_column(
            q_dust=atmos.q_dust,
            rho_d=atmos.rho_d,
            dz_c=atmos.dz_c,
            z_c=atmos.z_c,
            f_dust_emit=_sub_to_cell(f_dust_sub, nc, n_sub),
            precip_rate=rain_new + snow_new,
            u_star_cell=_sub_to_cell(sfc_diag_sub.u_star, nc, n_sub),
            dt=dt,
        )
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

    # 8. Subgrid Solar Noon EEO Acclimation, Daily LAI/Root Phenology, Host Sync, & Monthly Köppen
    rho_post_sub = crm.rho_d.reshape(nz, m_sub)
    thm_post_sub = crm.theta_m.reshape(nz, m_sub)
    qv_post_sub = crm.q_v.reshape(nz, m_sub)
    pres_post_sub, _, temp_post_sub = diagnose_pressure_temperature(rho_post_sub, thm_post_sub, qv_post_sub)
    t_air_sub = temp_post_sub[0]
    pres_sfc_post_sub = pres_post_sub[0] + C.GRAVITY * rho_post_sub[0] * (1.0 + qv_post_sub[0]) * dz0_sub
    es_sub = 611.2 * torch.exp(17.67 * (t_air_sub - C.T_FREEZE) / torch.clamp(t_air_sub - 29.65, min=10.0))
    e_sub = qv_post_sub[0] * pres_post_sub[0] / (C.EPS_WATER + (1.0 - C.EPS_WATER) * qv_post_sub[0])
    vpd_sub = torch.clamp(es_sub - e_sub, min=50.0)

    is_veg_noon_sub = crossed_noon_sub & (sfc_sub.surf_type == SURF_LAND)
    if is_veg_noon_sub.any():
        saved_bio = {a: getattr(sfc_sub.bio, a).clone() for a in _ACCLIM_BIO_ATTRS}
        update_daily_acclimation_smith(
            bio=sfc_sub.bio,
            midday_vis_beam=sw1_dir_sub,
            midday_vis_diff=sw1_dif_sub,
            midday_mu0=torch.clamp(mu0_sub, min=0.05),
            midday_temp=t_air_sub,
            midday_vpd=vpd_sub,
            night_temp=torch.minimum(clim_sub.day_t_min, t_air_sub - 2.0),
            pres_sfc=pres_sfc_post_sub,
            co2_ppmv=model.current_co2_ppmv(),
            lai_exposed=lai_exp_sub,
            is_vegetated_land=(sfc_sub.surf_type == SURF_LAND),
        )
        for a, old_val in saved_bio.items():
            setattr(sfc_sub.bio, a, torch.where(is_veg_noon_sub, getattr(sfc_sub.bio, a), old_val))

    precip_tot_sub = rain_new_sub + snow_new_sub
    clim_sub.day_precip_mm = clim_sub.day_precip_mm + precip_tot_sub * dt
    # Priestley & Taylor (1972, Mon. Wea. Rev., 100, 81-92) potential evapotranspiration [kg m^-2 s^-1]:
    # PET = 1.26 * [s / (s + gamma)] * max(R_n, 0) / L_v0 (consistent with Cai et al., 2025; Zhou et al., 2025)
    sw_net_sub = (
        (1.0 - sfc_diag_sub.alb_vis_dir) * sw1_dir_sub
        + (1.0 - sfc_diag_sub.alb_vis_dif) * sw1_dif_sub
        + (1.0 - sfc_diag_sub.alb_nir_dir) * sw2_dir_sub
        + (1.0 - sfc_diag_sub.alb_nir_dif) * sw2_dif_sub
    )
    lw_net_sub = sfc_diag_sub.emiss_sfc * (lw_dn_sub - C.STEFAN_BOLTZMANN * (sfc_diag_sub.t_rad_sfc ** 4))
    rn_pos_sub = torch.clamp(sw_net_sub + lw_net_sub, min=0.0)
    t_denom_pt = torch.clamp(t_air_sub - 29.65, min=10.0)
    des_dt_sub = es_sub * (17.67 * 243.5) / (t_denom_pt * t_denom_pt)
    gamma_psy_sub = (C.CP_D * pres_sfc_post_sub) / (C.EPS_WATER * C.L_V0)
    pet_pt_sub = 1.26 * (des_dt_sub / (des_dt_sub + gamma_psy_sub)) * rn_pos_sub / C.L_V0
    clim_sub.day_pet_mm = clim_sub.day_pet_mm + pet_pt_sub * dt
    clim_sub.day_ppfd_mol = clim_sub.day_ppfd_mol + (4.57 * sw1_dir_sub + 4.24 * sw1_dif_sub) * 1.0e-6 * dt
    clim_sub.day_temp_sum = clim_sub.day_temp_sum + t_air_sub
    clim_sub.day_vpd_sum = clim_sub.day_vpd_sum + vpd_sub
    clim_sub.day_t_min = torch.minimum(clim_sub.day_t_min, t_air_sub)
    clim_sub.day_steps += 1

    # Host monthly Köppen-Geiger accumulators (calendar month index 0..11)
    precip_tot = rain_new + snow_new
    t_air = _sub_to_cell(t_air_sub, nc, n_sub)
    month_idx = model.calendar.date_at(model.sim_time_sec).month - 1
    clim.month_temp_sum_c[month_idx] += (t_air - C.T_FREEZE)
    clim.month_precip_mm[month_idx] += precip_tot * dt
    clim.month_steps[month_idx] += 1.0
    clim.month_days_accum[month_idx] += dt / C.SECONDS_PER_DAY

    steps_per_day = max(int(round(C.SECONDS_PER_DAY / dt)), 1)
    if clim_sub.day_steps >= steps_per_day:
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

    # Compute area-weighted global diagnostics
    area = mesh.area_cell
    area_tot = area.sum()
    land_mask = (sfc.surf_type == SURF_LAND).to(area.dtype)
    land_area = torch.clamp((area * land_mask).sum(), min=1.0)

    t_sfc_mean = float(((area * t_rad_cell).sum() / area_tot).item())
    olr_mean = float(((area * olr).sum() / area_tot).item())
    asr_mean = float(((area * asr).sum() / area_tot).item())
    p_mm_day = float(((area * precip_tot * C.SECONDS_PER_DAY).sum() / area_tot).item())
    gpp_land = float(((area * land_mask * gpp_cell).sum() / land_area).item())
    lai_land = float(((area * land_mask * sfc.bio.lai).sum() / land_area).item())
    seaice_mkm2 = float(((area * f_seaice_cell).sum() * 1.0e-12).item())

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
