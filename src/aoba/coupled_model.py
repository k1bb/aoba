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
from aoba.biosphere import update_daily_acclimation_smith, update_daily_lai_root_height
from aoba.crm import CRMState, SPTendencies, step_crm_physics_and_dynamics
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
from aoba.surface import SURF_LAND, SURF_OCEAN, SurfaceState, compute_surface_albedos, init_surface_state, step_surface
from aoba.thermodynamics import diagnose_pressure_temperature
from aoba.turbulence import TKE_MIN
from aoba.vertical_grid import VerticalGrid, build_vertical_grid


@dataclass
class AOBAConfig:
    """Configuration for a global 1:1 Superparameterized AOBA simulation."""

    glevel: int = 3                          # SCVT mesh level (3 -> 642 cells; 4 -> 2562; 5 -> 10242)
    nz: int = 32                             # Shared Charney-Phillips vertical levels
    crm_nx: int = 16                         # Embedded CRM x-grid points per SCVT cell (default 2D 16x1 = 64 km curtain)
    crm_ny: int = 1                          # Embedded CRM y-grid points (1 = SP-CAM 2.5D curtain for Mac climate runs)
    crm_dx: float = 4000.0                   # Embedded CRM horizontal spacing [m]
    dt_host: float = 180.0                   # Coupled host / physics / SP time step [s]
    n_host_dyn_substeps: int = 2             # Host RK3-HEVI acoustic/gravity substeps per dt_host
    n_crm_dyn_substeps: int = 6              # Embedded CRM RK3-HEVI substeps per dt_host
    rad_stride: int = 4                      # Radiation evaluation stride (steps)
    crm_col_chunk: int = 16384               # Column batch chunk size for CRM evaluation (Unified Memory single-pass)
    co2_ppmv: float = 400.0                  # Well-mixed atmospheric CO2 concentration [ppmv]
    solar_constant: float = C.SOLAR_CONSTANT # Total solar irradiance S_0 [W m^-2]
    omega: float = C.OMEGA_EARTH             # Planetary rotation rate [rad s^-1]
    obliquity_deg: float = C.OBLIQUITY_DEG   # Axial tilt [deg]
    eccentricity: float = C.ECCENTRICITY     # Orbital eccentricity [-]
    perihelion_deg: float = C.PERIHELION_LON_DEG
    use_earth2014: bool = True
    earth2014_dir: str = "data/boundary/earth2014_5min"
    earth2014_stride: int = 2
    enable_superparam: bool = True           # 1:1 embedded 3D CRM in every SCVT cell


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
    sim_time_sec: float = 0.0
    step_count: int = 0
    rad_cache: dict[str, torch.Tensor] = field(default_factory=dict)


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
    frac_day = (t_sec % C.SECONDS_PER_DAY) / C.SECONDS_PER_DAY
    h_raw = 2.0 * math.pi * frac_day + lon_rad - math.pi
    hour_angle = torch.atan2(torch.sin(h_raw), torch.cos(h_raw))

    mu0_raw = torch.sin(lat_rad) * sin_dec + torch.cos(lat_rad) * cos_dec * torch.cos(hour_angle)
    mu0 = torch.clamp(mu0_raw, min=0.0)
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
    vgrid = build_vertical_grid(nz=cfg.nz, device=dev, dtype=dtype)
    atmos = init_global_hydrostatic_atmosphere(mesh, vgrid, moist=True, baroclinic_pert=True)

    # Initialize surface temperature consistent with latitude & elevation so spinup is smooth
    t_sfc_lat = 301.0 - 40.0 * (torch.sin(mesh.lat_cell) ** 2) - 0.0065 * torch.clamp(mesh.z_topo, min=0.0)
    sfc = init_surface_state(mesh.surf_type, t_init=293.15, sigma_topo=mesh.sigma_topo)
    sfc.t_ocean_ml = torch.where(
        mesh.surf_type == SURF_OCEAN,
        torch.clamp(301.0 - 34.0 * (torch.sin(mesh.lat_cell) ** 2), min=C.T_FREEZE_OCEAN + 0.2),
        sfc.t_ocean_ml,
    )
    sfc.t_canopy = t_sfc_lat.clone()
    sfc.t_soil = t_sfc_lat.unsqueeze(0).expand_as(sfc.t_soil).clone()
    # Seed initial sea ice in polar oceans where T_lat < T_freeze_ocean
    polar_cold = (mesh.surf_type == SURF_OCEAN) & (t_sfc_lat < C.T_FREEZE_OCEAN)
    sfc.m_seaice = torch.where(polar_cold, torch.full_like(t_sfc_lat, 900.0), torch.zeros_like(t_sfc_lat))

    ocn = init_ekman_ocean_state(sfc)

    # Initialize 1:1 embedded 3D CRM state `[Nz, N_cells, Ny, Nx]` synchronized to host columns
    nz = cfg.nz
    nc = mesh.n_cells
    ny, nx = cfg.crm_ny, cfg.crm_nx
    shape4 = (nz, nc, ny, nx)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    noise = (torch.rand((3, nc, ny, nx), generator=gen, dtype=torch.float32) - 0.5) * 0.10
    noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)).to(device=dev, dtype=dtype)

    thm_crm = atmos.theta_m.view(nz, nc, 1, 1).expand(shape4).clone()
    thm_crm[:3] = thm_crm[:3] + noise

    crm = CRMState(
        rho_d=atmos.rho_d.view(nz, nc, 1, 1).expand(shape4).clone(),
        u=torch.zeros(shape4, device=dev, dtype=dtype),
        v=torch.zeros(shape4, device=dev, dtype=dtype),
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

    z1 = torch.zeros(nc, device=dev, dtype=dtype)
    clim = ClimateAccumulator(
        day_precip_mm=z1.clone(),
        day_pet_mm=z1.clone(),
        day_ppfd_mol=z1.clone(),
        day_temp_sum=z1.clone(),
        day_vpd_sum=z1.clone(),
        day_t_min=torch.full_like(z1, 400.0),
        day_steps=0,
        month_temp_sum_c=torch.zeros((12, nc), device=dev, dtype=dtype),
        month_precip_mm=torch.zeros((12, nc), device=dev, dtype=dtype),
        month_steps=torch.zeros((12, 1), device=dev, dtype=dtype),
        month_days_accum=torch.zeros((12, 1), device=dev, dtype=dtype),
    )

    return AOBAState(
        cfg=cfg,
        mesh=mesh,
        vgrid=vgrid,
        atmos=atmos,
        crm=crm,
        sfc=sfc,
        ocn=ocn,
        clim=clim,
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


def step_aoba_model(model: AOBAState) -> GlobalStepDiagnostics:
    """Advance the global 1:1 Superparameterized AOBA Earth System Model by `cfg.dt_host` [s]."""
    cfg = model.cfg
    mesh = model.mesh
    vgrid = model.vgrid
    atmos = model.atmos
    sfc = model.sfc
    dt = cfg.dt_host
    nz = cfg.nz
    nc = mesh.n_cells

    # 1. Orbital solar geometry at midpoint of step & detect local solar noon crossings
    _, _, h_prev = compute_orbital_insolation(
        model.sim_time_sec, mesh.lat_cell, mesh.lon_cell,
        cfg.solar_constant, cfg.obliquity_deg, cfg.eccentricity, cfg.perihelion_deg,
    )
    mu0, s_toa, h_next = compute_orbital_insolation(
        model.sim_time_sec + dt, mesh.lat_cell, mesh.lon_cell,
        cfg.solar_constant, cfg.obliquity_deg, cfg.eccentricity, cfg.perihelion_deg,
    )
    # Local solar noon crossed when h_prev < 0 <= h_next (excluding midnight wrap around +/-pi)
    crossed_noon = (h_prev < 0.0) & (h_next >= 0.0) & ((h_next - h_prev) < math.pi)

    # 2. Diagnose host cell winds, pressure, temperature
    u_cell, v_cell = reconstruct_cell_winds(mesh, atmos.u_edge)
    pres_c, _, temp_c = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
    dz0 = atmos.z_c[0] - atmos.z_w[0]
    pres_sfc = pres_c[0] + C.GRAVITY * atmos.rho_d[0] * (1.0 + atmos.q_v[0]) * dz0
    z_rel_c = atmos.z_c - atmos.z_w[0:1]
    z_ref_rel = torch.clamp(z_rel_c[0] - 0.65 * sfc.bio.h_canopy, min=5.0)

    # 3. Surface albedos & 4-band radiative transfer
    alb_vd, alb_vf, alb_nd, alb_nf, _, lai_exp, _, _ = compute_surface_albedos(sfc, mu0)
    if (not model.rad_cache) or (model.step_count % max(cfg.rad_stride, 1) == 0):
        t_rad_in = torch.where(sfc.surf_type == SURF_OCEAN, sfc.t_ocean_ml, sfc.t_canopy)
        pres_w = torch.empty((nz + 1, nc), device=pres_c.device, dtype=pres_c.dtype)
        pres_w[0] = pres_sfc
        pres_w[1:-1] = torch.sqrt(torch.clamp(pres_c[:-1] * pres_c[1:], min=1.0))
        pres_w[-1] = torch.clamp(0.5 * pres_c[-1], min=10.0)

        temp_w = torch.empty((nz + 1, nc), device=temp_c.device, dtype=temp_c.dtype)
        temp_w[0] = temp_c[0]
        temp_w[1:-1] = 0.5 * (temp_c[:-1] + temp_c[1:])
        temp_w[-1] = temp_c[-1]

        rad = compute_four_band_radiation(
            pres_c=pres_c,
            pres_w=pres_w,
            temp_c=temp_c,
            temp_w=temp_w,
            dz_c=atmos.dz_c,
            rho_d_c=atmos.rho_d,
            qv_c=atmos.q_v,
            qc_c=atmos.q_c,
            qi_c=atmos.q_i,
            t_sfc=t_rad_in,
            albedo_vis=alb_vd,
            albedo_nir=alb_nd,
            mu0=mu0,
            s_toa=s_toa,
            co2_ppmv=cfg.co2_ppmv,
            albedo_vis_dif=alb_vf,
            albedo_nir_dif=alb_nf,
        )
        model.rad_cache["q_rad"] = rad.q_rad_net
        model.rad_cache["sw1_dir"] = rad.sfc_sw1_dir_dn
        model.rad_cache["sw1_dif"] = rad.sfc_sw1_dif_dn
        model.rad_cache["sw2_dir"] = rad.sfc_sw2_dir_dn
        model.rad_cache["sw2_dif"] = rad.sfc_sw2_dif_dn
        model.rad_cache["lw_dn"] = rad.lw_flux_dn[0]
        model.rad_cache["olr"] = rad.toa_olr
        model.rad_cache["asr"] = rad.toa_asr

    q_rad = model.rad_cache["q_rad"]
    sw1_dir = model.rad_cache["sw1_dir"]
    sw1_dif = model.rad_cache["sw1_dif"]
    sw2_dir = model.rad_cache["sw2_dir"]
    sw2_dif = model.rad_cache["sw2_dif"]
    lw_dn = model.rad_cache["lw_dn"]
    olr = model.rad_cache["olr"]
    asr = model.rad_cache["asr"]

    rain_prev = model.rad_cache.get("rain", torch.zeros_like(pres_sfc))
    snow_prev = model.rad_cache.get("snow", torch.zeros_like(pres_sfc))

    # 4. Land/Snow/Soil/Two-Leaf Biosphere & Ocean/Sea-Ice Surface Balance + Voronoi Ekman Ocean
    sfc_diag = step_surface(
        sfc=sfc,
        u_1=u_cell[0],
        v_1=v_cell[0],
        temp_1=temp_c[0],
        qv_1=atmos.q_v[0],
        pres_1=pres_c[0],
        pres_sfc=pres_sfc,
        z_ref_rel=z_ref_rel,
        sw1_dir_down=sw1_dir,
        sw1_dif_down=sw1_dif,
        sw2_dir_down=sw2_dir,
        sw2_dif_down=sw2_dif,
        lw_down=lw_dn,
        precip_rain=rain_prev,
        precip_snow=snow_prev,
        mu0=mu0,
        co2_ppmv=cfg.co2_ppmv,
        dt=dt,
    )
    step_ekman_ocean_voronoi(
        ocn=model.ocn,
        sfc=sfc,
        mesh=mesh,
        tau_x_sfc=sfc_diag.tau_x,
        tau_y_sfc=sfc_diag.tau_y,
        f_seaice=sfc_diag.f_seaice,
        runoff_glacier=sfc_diag.runoff_ice,
        dt=dt,
    )

    # 5. Lott & Miller (1997) Subgrid Orographic Drag
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

    # Host RK3-HEVI performs internal MPAS split-explicit acoustic sub-stepping automatically;
    # enforce synoptic/gravity-wave RK3 Courant limit dt_hdyn <= min(900 s, 0.40 * dc_min / 80)
    dt_adv_host = min(900.0, 0.40 * float(mesh.dc_edge.min().item()) / 80.0)
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

    dthm_ls = (atmos.theta_m - thm_before_dyn) / dt
    dqn_ls = (atmos.q_n - qn_before_dyn) / dt
    u_host_new, v_host_new = reconstruct_cell_winds(mesh, atmos.u_edge)

    # Align surface momentum drag coefficient with post-dynamics lowest-layer wind so drag strictly opposes flow
    old_spd0 = torch.sqrt(u_cell[0] * u_cell[0] + v_cell[0] * v_cell[0] + 1.0e-4)
    tau_mag0 = torch.sqrt(sfc_diag.tau_x * sfc_diag.tau_x + sfc_diag.tau_y * sfc_diag.tau_y)
    drag_c0 = tau_mag0 / old_spd0
    sfc_diag.tau_x = -drag_c0 * u_host_new[0]
    sfc_diag.tau_y = -drag_c0 * v_host_new[0]

    # 7. 1:1 Embedded 3D CRM Physics & Convective Transport Across All SCVT Columns
    rain_new = torch.zeros_like(pres_sfc)
    snow_new = torch.zeros_like(pres_sfc)

    if cfg.enable_superparam:
        du_crm_c = torch.zeros_like(u_host_new)
        dv_crm_c = torch.zeros_like(v_host_new)

        chunk = max(cfg.crm_col_chunk, 1)
        if chunk >= nc:
            drho_mean = (atmos.rho_d - model.crm.rho_d.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
            model.crm.rho_d = torch.clamp(model.crm.rho_d + drho_mean, min=1.0e-5)
            model.crm.f_coriolis = mesh.f_cell
            sp = step_crm_physics_and_dynamics(
                crm=model.crm,
                vgrid=vgrid,
                dt_crm=dt,
                n_dyn_substeps=cfg.n_crm_dyn_substeps,
                u_host_target=u_host_new,
                v_host_target=v_host_new,
                tau_sp_mom=dt,
                dtheta_m_ls_dt=dthm_ls,
                dq_n_ls_dt=dqn_ls,
                sfc_fluxes=sfc_diag,
                rad_heating_k_s=q_rad,
                z_c_col=atmos.z_c,
                z_w_col=atmos.z_w,
                dz_c_col=atmos.dz_c,
            )
            atmos.theta_m = model.crm.theta_m.mean(dim=(-2, -1))
            atmos.q_n = model.crm.q_n.mean(dim=(-2, -1))
            atmos.q_p = model.crm.q_p.mean(dim=(-2, -1))
            atmos.q_v = model.crm.q_v.mean(dim=(-2, -1))
            atmos.q_c = model.crm.q_c.mean(dim=(-2, -1))
            atmos.q_i = model.crm.q_i.mean(dim=(-2, -1))
            du_crm_c = (model.crm.u.mean(dim=(-2, -1)) - u_host_new) / dt
            dv_crm_c = (model.crm.v.mean(dim=(-2, -1)) - v_host_new) / dt
            rain_new = sp.precip_rain
            snow_new = sp.precip_snow
        else:
            for i0 in range(0, nc, chunk):
                i1 = min(i0 + chunk, nc)
                sl = slice(i0, i1)
                # Synchronize horizontal-mean CRM dry density with host column while keeping 3D buoyant perturbations
                rho_crm_sl = model.crm.rho_d[:, sl]
                drho_mean = (atmos.rho_d[:, sl] - rho_crm_sl.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
                sub_crm = CRMState(
                    rho_d=torch.clamp(rho_crm_sl + drho_mean, min=1.0e-5),
                    u=model.crm.u[:, sl],
                    v=model.crm.v[:, sl],
                    w=model.crm.w[:, sl],
                    theta_m=model.crm.theta_m[:, sl],
                    q_n=model.crm.q_n[:, sl],
                    q_p=model.crm.q_p[:, sl],
                    q_v=model.crm.q_v[:, sl],
                    q_c=model.crm.q_c[:, sl],
                    q_i=model.crm.q_i[:, sl],
                    qke=model.crm.qke[:, sl],
                    dx=cfg.crm_dx,
                    dy=cfg.crm_dx,
                    f_coriolis=mesh.f_cell[sl],
                )
                # Slice surface fluxes for chunk
                sub_sfc = type(sfc_diag)(
                    **{k: v[sl] for k, v in sfc_diag.__dict__.items()}
                )
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
                    rad_heating_k_s=q_rad[:, sl],
                    z_c_col=atmos.z_c[:, sl],
                    z_w_col=atmos.z_w[:, sl],
                    dz_c_col=atmos.dz_c[:, sl],
                )
                # Write updated 3D subgrid CRM arrays back & couple horizontal-mean state to host!
                model.crm.rho_d[:, sl] = sub_crm.rho_d
                model.crm.u[:, sl] = sub_crm.u
                model.crm.v[:, sl] = sub_crm.v
                model.crm.w[:, sl] = sub_crm.w
                model.crm.theta_m[:, sl] = sub_crm.theta_m
                model.crm.q_n[:, sl] = sub_crm.q_n
                model.crm.q_p[:, sl] = sub_crm.q_p
                model.crm.q_v[:, sl] = sub_crm.q_v
                model.crm.q_c[:, sl] = sub_crm.q_c
                model.crm.q_i[:, sl] = sub_crm.q_i
                model.crm.qke[:, sl] = sub_crm.qke

                atmos.theta_m[:, sl] = sub_crm.theta_m.mean(dim=(-2, -1))
                atmos.q_n[:, sl] = sub_crm.q_n.mean(dim=(-2, -1))
                atmos.q_p[:, sl] = sub_crm.q_p.mean(dim=(-2, -1))
                atmos.q_v[:, sl] = sub_crm.q_v.mean(dim=(-2, -1))
                atmos.q_c[:, sl] = sub_crm.q_c.mean(dim=(-2, -1))
                atmos.q_i[:, sl] = sub_crm.q_i.mean(dim=(-2, -1))

                # Net momentum tendency from surface stress + MYNN PBL + convective momentum transport
                du_crm_c[:, sl] = (sub_crm.u.mean(dim=(-2, -1)) - u_host_new[:, sl]) / dt
                dv_crm_c[:, sl] = (sub_crm.v.mean(dim=(-2, -1)) - v_host_new[:, sl]) / dt
                rain_new[sl] = sp.precip_rain
                snow_new[sl] = sp.precip_snow

        du_sp_edge = project_cell_vectors_to_edges(mesh, du_crm_c, dv_crm_c)
        atmos.u_edge = atmos.u_edge + dt * du_sp_edge

    model.rad_cache["rain"] = rain_new
    model.rad_cache["snow"] = snow_new
    model.rad_cache["evap"] = sfc_diag.evap
    model.rad_cache["shf"] = sfc_diag.shf
    model.rad_cache["gpp"] = sfc_diag.gpp

    # 8. Local Solar Noon Smith/Scott-Smith EEO Acclimation + Daily LAI + Monthly Köppen Accumulators
    _, _, temp_after = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
    t_air = temp_after[0]
    es_1 = 611.2 * torch.exp(17.67 * (t_air - C.T_FREEZE) / torch.clamp(t_air - 29.65, min=10.0))
    e_1 = atmos.q_v[0] * pres_c[0] / (C.EPS_WATER + (1.0 - C.EPS_WATER) * atmos.q_v[0])
    vpd_1 = torch.clamp(es_1 - e_1, min=50.0)

    is_veg_noon = crossed_noon & (sfc.surf_type == SURF_LAND)
    if is_veg_noon.any():
        # Update EMA midday downward shortwave -> de Pury light absorption -> Two-Leaf enzyme pools ONLY on cells crossing local noon!
        idx_noon = torch.where(is_veg_noon)[0]
        # Construct temporary sliced view or masked update
        bio_old_vb = sfc.bio.ema_vis_beam.clone()
        bio_old_vd = sfc.bio.ema_vis_diff.clone()
        bio_old_mu = sfc.bio.ema_mu0_mid.clone()
        bio_old_tm = sfc.bio.ema_temp_mid.clone()
        bio_old_vp = sfc.bio.ema_vpd_mid.clone()
        bio_old_tn = sfc.bio.ema_temp_night.clone()
        vc3_0, jm3_0, r3_0 = sfc.bio.vcmax25_c3.clone(), sfc.bio.jmax25_c3.clone(), sfc.bio.r25_c3.clone()
        vc4_0, jm4_0, vp4_0, r4_0 = sfc.bio.vcmax25_c4.clone(), sfc.bio.jmax25_c4.clone(), sfc.bio.vpmax25_c4.clone(), sfc.bio.r25_c4.clone()
        f4_0 = sfc.bio.f_c4.clone()

        update_daily_acclimation_smith(
            bio=sfc.bio,
            midday_vis_beam=sw1_dir,
            midday_vis_diff=sw1_dif,
            midday_mu0=torch.clamp(mu0, min=0.05),
            midday_temp=t_air,
            midday_vpd=vpd_1,
            night_temp=torch.minimum(model.clim.day_t_min, t_air - 2.0),
            pres_sfc=pres_sfc,
            co2_ppmv=cfg.co2_ppmv,
            lai_exposed=lai_exp,
            is_vegetated_land=(sfc.surf_type == SURF_LAND),
        )
        # Keep unchanged values on columns NOT crossing local solar noon this step
        sfc.bio.ema_vis_beam = torch.where(is_veg_noon, sfc.bio.ema_vis_beam, bio_old_vb)
        sfc.bio.ema_vis_diff = torch.where(is_veg_noon, sfc.bio.ema_vis_diff, bio_old_vd)
        sfc.bio.ema_mu0_mid = torch.where(is_veg_noon, sfc.bio.ema_mu0_mid, bio_old_mu)
        sfc.bio.ema_temp_mid = torch.where(is_veg_noon, sfc.bio.ema_temp_mid, bio_old_tm)
        sfc.bio.ema_vpd_mid = torch.where(is_veg_noon, sfc.bio.ema_vpd_mid, bio_old_vp)
        sfc.bio.ema_temp_night = torch.where(is_veg_noon, sfc.bio.ema_temp_night, bio_old_tn)
        sfc.bio.vcmax25_c3 = torch.where(is_veg_noon, sfc.bio.vcmax25_c3, vc3_0)
        sfc.bio.jmax25_c3 = torch.where(is_veg_noon, sfc.bio.jmax25_c3, jm3_0)
        sfc.bio.r25_c3 = torch.where(is_veg_noon, sfc.bio.r25_c3, r3_0)
        sfc.bio.vcmax25_c4 = torch.where(is_veg_noon, sfc.bio.vcmax25_c4, vc4_0)
        sfc.bio.jmax25_c4 = torch.where(is_veg_noon, sfc.bio.jmax25_c4, jm4_0)
        sfc.bio.vpmax25_c4 = torch.where(is_veg_noon, sfc.bio.vpmax25_c4, vp4_0)
        sfc.bio.r25_c4 = torch.where(is_veg_noon, sfc.bio.r25_c4, r4_0)
        sfc.bio.f_c4 = torch.where(is_veg_noon, sfc.bio.f_c4, f4_0)

    # Update daily & monthly climate accumulators
    precip_tot = rain_new + snow_new
    clim = model.clim
    clim.day_precip_mm = clim.day_precip_mm + precip_tot * dt
    clim.day_pet_mm = clim.day_pet_mm + torch.clamp(sw1_dir + sw1_dif + sw2_dir + sw2_dif, min=0.0) * dt / C.L_V0
    clim.day_ppfd_mol = clim.day_ppfd_mol + (4.57 * sw1_dir + 4.24 * sw1_dif) * 1.0e-6 * dt
    clim.day_temp_sum = clim.day_temp_sum + t_air
    clim.day_vpd_sum = clim.day_vpd_sum + vpd_1
    clim.day_t_min = torch.minimum(clim.day_t_min, t_air)
    clim.day_steps += 1

    # Calendar month index 0..11 within tropical year
    yr_sec = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
    month_idx = min(int(((model.sim_time_sec % yr_sec) / yr_sec) * 12.0), 11)
    clim.month_temp_sum_c[month_idx] += (t_air - C.T_FREEZE)
    clim.month_precip_mm[month_idx] += precip_tot * dt
    clim.month_steps[month_idx] += 1.0
    clim.month_days_accum[month_idx] += dt / C.SECONDS_PER_DAY

    steps_per_day = max(int(round(C.SECONDS_PER_DAY / dt)), 1)
    if clim.day_steps >= steps_per_day:
        ws_beta = torch.clamp(sfc.w_root / (0.5 * torch.clamp(sfc.bio.s_root_cap, min=20.0)), 0.0, 1.0)
        update_daily_lai_root_height(
            bio=sfc.bio,
            daily_precip_mm=clim.day_precip_mm,
            daily_pet_mm=torch.clamp(clim.day_pet_mm, min=0.05),
            daily_ppfd_mol_m2=clim.day_ppfd_mol,
            daily_temp_k=clim.day_temp_sum / float(clim.day_steps),
            daily_vpd_pa=clim.day_vpd_sum / float(clim.day_steps),
            pres_sfc=pres_sfc,
            co2_ppmv=cfg.co2_ppmv,
            water_stress_beta=ws_beta,
            is_vegetated_land=(sfc.surf_type == SURF_LAND),
        )
        clim.day_precip_mm.zero_()
        clim.day_pet_mm.zero_()
        clim.day_ppfd_mol.zero_()
        clim.day_temp_sum.zero_()
        clim.day_vpd_sum.zero_()
        clim.day_t_min.fill_(400.0)
        clim.day_steps = 0

    model.sim_time_sec += dt
    model.step_count += 1

    # Compute area-weighted global diagnostics
    area = mesh.area_cell
    area_tot = area.sum()
    land_mask = (sfc.surf_type == SURF_LAND).to(area.dtype)
    land_area = torch.clamp((area * land_mask).sum(), min=1.0)

    t_sfc_mean = float(((area * sfc_diag.t_rad_sfc).sum() / area_tot).item())
    olr_mean = float(((area * olr).sum() / area_tot).item())
    asr_mean = float(((area * asr).sum() / area_tot).item())
    p_mm_day = float(((area * precip_tot * C.SECONDS_PER_DAY).sum() / area_tot).item())
    gpp_land = float(((area * land_mask * sfc_diag.gpp).sum() / land_area).item())
    lai_land = float(((area * land_mask * sfc.bio.lai).sum() / land_area).item())
    seaice_mkm2 = float(((area * sfc_diag.f_seaice).sum() * 1.0e-12).item())

    return GlobalStepDiagnostics(
        sim_time_days=model.sim_time_sec / C.SECONDS_PER_DAY,
        global_mean_t_sfc_k=t_sfc_mean,
        global_mean_olr_wm2=olr_mean,
        global_mean_asr_wm2=asr_mean,
        global_mean_precip_mm_day=p_mm_day,
        land_mean_gpp_umol=gpp_land,
        land_mean_lai=lai_land,
        seaice_area_million_km2=seaice_mkm2,
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
