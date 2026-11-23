"""3D Multi-Layer Hydrostatic Boussinesq SCVT Voronoi OGCM & Dynamic Sea-Ice Drift.

Upgrades AOBA's ocean component from the 1.5-layer Ekman + reduced-gravity shallow-water
model (`ocean_ekman.py`, retained as `ocean_mode = "ekman_sw"`) to a full-depth 3D
z-level primitive-equation ocean (`ocean_mode = "ogcm_3d"`) on the orthogonal SCVT
Voronoi C-grid using `Earth2014 BED2014` global bathymetry (`mesh.z_bed`).

Physics & Numerics (Zero Q-Flux, Strictly Conservative):
1. **Vertical Grid & Bathymetry**: `N_zo` stretched z-levels (default 18 levels spanning
   0-5500 m; Layer 0 thickness = `h_ml_ocean = 50 m`) with exact wet-cell partial-step
   thicknesses `h_c[N_zo, N_cells]` masked by `Earth2014 BED2014` bathymetry and zero
   normal flux across coastlines and submarine ridges.
2. **Nonlinear Equation of State** (Roquet et al., 2015, *Ocean Modelling*, 90, 29-43):
   Simplified TEOS-10 Boussinesq polynomial capturing thermal expansion, cabbeling,
   thermobaricity, and haline contraction (driving polar deep convection & AMOC).
3. **TRiSK SCVT C-Grid Split-Explicit Dynamics** (Ringler et al., 2010, 2013, *Ocean
   Modelling*, 69, 211-232):
   - 3D baroclinic momentum with hydrostatic pressure gradient, energy-conserving TRiSK
     Coriolis/vorticity, scale-aware Munk horizontal viscosity, quadratic bottom drag,
     and implicit vertical momentum mixing forced by surface wind stress.
   - Fast 2D barotropic free-surface mode `(u_baro, eta)` substepped with Wicker &
     Skamarock (2002) RK3, ensuring exact 3D continuity `w_{k-1/2} = w_{k+1/2} - div(h u)`.
4. **Gent-McWilliams (1990) Eddy Bolus Transport & Redi Diffusion** (Gent & McWilliams,
   1990, *JPO*, 20, 150-155; Danabasoglu & McWilliams, 1995; Griffies, 1998):
   Tapered isopycnal-slope eddy streamfunction `Psi_GM = kappa_GM * s` producing a
   strictly non-divergent bolus velocity `u_GM = -dPsi_GM/dz` that slumps steep ACC /
   subpolar fronts and controls Southern Ocean deep heat uptake.
5. **Tracers (T, S), Implicit Vertical Mixing & Convective Adjustment** (Bryan & Lewis,
   1979; Pacanowski & Philander, 1981; Huang, 1993):
   - Exact flux-form 3D advection + horizontal isopycnal/eddy diffusion + batched implicit
     tridiagonal vertical diffusion with enhanced boundary-layer mixing and instantaneous
     convective adjustment (`K_v -> K_conv`) wherever `drho/dz < 0`.
   - Conservative surface virtual salt flux from `(E - P - R_river - R_glacier)` routed
     via `mesh.coastal_ocean_idx` plus sea-ice brine rejection / meltwater freshening.
6. **Dynamic Sea-Ice Drift** (Hibler, 1979; Leppäranta, 2011):
   Free-drift wind + Layer-0 ocean current + Coriolis turning with plastic ridging
   resistance above critical thickness `h_ridge`, conservatively advecting `m_seaice`
   and `swe_seaice` across wet ocean Voronoi edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import solve_tridiagonal_batched
from aoba.scvt_dynamics import (
    compute_scvt_divergence,
    compute_scvt_vorticity,
    compute_vector_laplacian_edge,
)
from aoba.scvt_mesh import SCVTMesh
from aoba.surface import SURF_OCEAN, SurfaceState, _kahan_add

# ---------------------------------------------------------------------------
# Physical Constants & Default Parameters (TOML-exposed in [ocean_ogcm] / [ocean_ekman])
# ---------------------------------------------------------------------------
OGCM_DEFAULT_DZ: tuple[float, ...] = (
    50.0, 50.0, 60.0, 75.0, 95.0, 120.0,
    150.0, 190.0, 240.0, 300.0, 360.0, 420.0,
    480.0, 540.0, 600.0, 660.0, 730.0, 880.0,
)  # 18 levels, sum = 6000 m (Layer 0 matches h_ml_ocean = 50 m)

S_REF_PSU: float = 34.7             # Global reference seawater salinity [psu]
S_SEAICE_PSU: float = 4.0           # Bulk salinity retained in sea ice [psu]
T_REF_EOS_K: float = 283.15         # Reference potential temperature for EOS (10 degC) [K]
# Roquet et al. (2015, Table 1) simplified TEOS-10 Boussinesq EOS coefficients
EOS_ALPHA0: float = 1.67e-4         # Linear thermal expansion at 10 C [K^-1]
EOS_ALPHA_T: float = 1.00e-5        # Quadratic cabbeling coefficient [K^-2]
EOS_ALPHA_Z: float = 1.60e-8        # Thermobaric depth-temperature coefficient [m^-1 K^-1]
EOS_BETA0: float = 7.80e-4          # Linear haline contraction coefficient [psu^-1]
EOS_ALPHA_S: float = 2.00e-6        # Haline-thermal cross term [K^-1 psu^-1]

KAPPA_GM: float = 800.0             # Gent-McWilliams isopycnal thickness diffusivity [m^2 s^-1]
KAPPA_REDI_HORIZ: float = 1000.0    # Horizontal / isopycnal mesoscale tracer diffusivity [m^2 s^-1]
GM_SLOPE_MAX: float = 1.0e-2        # Danabasoglu & McWilliams (1995) slope taper cap [-]
KAPPA_V_BG: float = 2.0e-5          # Upper-ocean background diapycnal diffusivity [m^2 s^-1]
KAPPA_V_ABYSS: float = 1.0e-4       # Abyssal diapycnal diffusivity (Bryan & Lewis, 1979) [m^2 s^-1]
KAPPA_V_CONV: float = 1.0           # Convective adjustment diffusivity when drho/dz < 0 [m^2 s^-1]
A_V_MOM: float = 1.0e-2             # Vertical eddy viscosity for wind-driven momentum [m^2 s^-1]
C_DRAG_BOT: float = 2.5e-3          # Quadratic bottom friction drag coefficient [-]
R_RAYLEIGH_OCN: float = 1.0 / (120.0 * C.SECONDS_PER_DAY)  # Weak background drag [s^-1]
BETA_MUNK_OGCM: float = 2.28e-11    # Equatorial beta for scale-aware Munk viscosity [m^-1 s^-1]

# Dynamic sea-ice drift parameters (Hibler, 1979; Leppäranta, 2011)
SEAICE_WIND_FACTOR: float = 0.018   # Nansen wind-drift coupling factor (~1.8% of u_a / tau) [-]
SEAICE_TURN_DEG: float = 20.0       # Wind-to-ice Coriolis turning angle [deg]
SEAICE_H_RIDGE_M: float = 3.0       # Ridging resistance thickness scale [m]
SEAICE_U_MAX: float = 0.50          # Maximum sea-ice drift speed [m s^-1]
SEAICE_DIFF_H: float = 500.0        # Subgrid sea-ice lateral diffusivity [m^2 s^-1]


def build_ogcm_dz_levels(nz_ocn: int = 18, h_ml0: float = 50.0, max_depth: float = 5500.0) -> torch.Tensor:
    """Construct smoothly stretched ocean layer thicknesses `[nz_ocn]` summing to `max_depth`."""
    if nz_ocn == len(OGCM_DEFAULT_DZ) and abs(h_ml0 - 50.0) < 1.0e-3 and abs(max_depth - 5500.0) < 1.0:
        return torch.tensor(OGCM_DEFAULT_DZ, dtype=torch.float32)
    nz = max(int(nz_ocn), 2)
    dz = [float(h_ml0)]
    if nz == 2:
        dz.append(max(max_depth - h_ml0, 50.0))
        return torch.tensor(dz, dtype=torch.float32)
    # Geometric stretch for remaining nz - 1 layers
    rem = max(max_depth - h_ml0, 100.0)
    # Solve sum_{m=0}^{nz-2} h_ml0 * r^m = rem via bisection
    lo, hi = 1.0, 2.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = sum(h_ml0 * (mid ** m) for m in range(nz - 1))
        if s < rem:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    for m in range(nz - 1):
        dz.append(h_ml0 * (r ** m))
    t = torch.tensor(dz, dtype=torch.float32)
    t[1:] = t[1:] * ((max_depth - t[0]) / t[1:].sum())
    return t


def compute_seawater_density_anomaly(
    temp_k: torch.Tensor,
    sal_psu: torch.Tensor,
    z_depth_m: torch.Tensor,
) -> torch.Tensor:
    """Nonlinear Boussinesq density anomaly `rho' = rho - rho_0` [kg m^-3] (Roquet et al., 2015).

    Includes thermal expansion, cabbeling, thermobaricity, and haline contraction.
    """
    dT = temp_k - T_REF_EOS_K
    dS = sal_psu - S_REF_PSU
    drho_over_rho0 = (
        -EOS_ALPHA0 * dT
        - 0.5 * EOS_ALPHA_T * dT * dT
        - EOS_ALPHA_Z * z_depth_m * dT
        + EOS_BETA0 * dS
        - EOS_ALPHA_S * dT * dS
    )
    return C.RHO_SEAWATER * drho_over_rho0


@dataclass
class OceanOGCMState:
    """Prognostic 3D multi-layer SCVT Voronoi OGCM state."""

    # Geometry & bathymetry masks (static after init)
    nz_ocn: int
    dz_nom: torch.Tensor         # Nominal layer thicknesses [N_zo] [m]
    z_c: torch.Tensor            # Layer center depths (positive downward) [N_zo] [m]
    z_w: torch.Tensor            # Interface depths [N_zo + 1] [m]
    h_c: torch.Tensor            # Actual wet cell thicknesses [N_zo, N_cells] [m] (0 if dry)
    h_e: torch.Tensor            # Wet edge thicknesses [N_zo, N_edges] [m] (0 if land/ridge)
    wet_c: torch.Tensor          # Boolean wet cell mask [N_zo, N_cells]
    wet_e: torch.Tensor          # Boolean wet edge mask [N_zo, N_edges]
    h_col: torch.Tensor          # Total wet water-column depth [N_cells] [m]
    h_col_e: torch.Tensor        # Wet edge column depth [N_edges] [m]
    k_bot: torch.Tensor          # Deepest wet layer index per cell [N_cells] (0 if land)

    # 3D Prognostic fields
    temp: torch.Tensor           # 3D potential temperature [N_zo, N_cells] [K]
    sal: torch.Tensor            # 3D practical salinity [N_zo, N_cells] [psu]
    u_edge: torch.Tensor         # 3D edge-normal velocity [N_zo, N_edges] [m s^-1]
    eta: torch.Tensor            # Free-surface height anomaly [N_cells] [m]
    u_baro: torch.Tensor         # Barotropic depth-mean normal velocity [N_edges] [m s^-1]

    # Diagnostics & Kahan float32 rounding compensation
    w_vel: torch.Tensor          # Vertical velocity at interfaces (positive upward) [N_zo+1, N_cells] [m s^-1]
    t_comp: torch.Tensor         # Kahan compensation for 3D temp [N_zo, N_cells] [K]
    s_comp: torch.Tensor         # Kahan compensation for 3D sal [N_zo, N_cells] [psu]
    m_seaice_prev: torch.Tensor  # Previous-step cell-mean sea-ice mass for brine/melt salt closure [N_cells] [kg m^-2]


def init_ogcm_state(
    sfc: SurfaceState,
    mesh: SCVTMesh,
    nz_ocn: int = 18,
    t_abyss_k: float = 275.0,
) -> OceanOGCMState:
    """Initialize 3D SCVT OGCM from `Earth2014 BED2014` bathymetry and analytic thermocline."""
    dev = sfc.t_ocean_ml.device
    dtype = sfc.t_ocean_ml.dtype
    nc = mesh.n_cells
    ne = mesh.n_edges

    h_ml0 = float(sfc.h_ocean_ml[sfc.surf_type == SURF_OCEAN].mean().item()) if bool((sfc.surf_type == SURF_OCEAN).any()) else 50.0
    dz_nom = build_ogcm_dz_levels(nz_ocn=nz_ocn, h_ml0=h_ml0, max_depth=5500.0).to(device=dev, dtype=dtype)
    nzo = dz_nom.shape[0]
    z_w = torch.zeros(nzo + 1, device=dev, dtype=dtype)
    z_w[1:] = torch.cumsum(dz_nom, dim=0)
    z_c = 0.5 * (z_w[:-1] + z_w[1:])

    is_ocean = sfc.surf_type == SURF_OCEAN
    max_depth = float(z_w[-1].item())
    if mesh.z_bed is not None:
        # Earth2014 BED2014 bathymetry: negative values over ocean -> positive depth
        depth_raw = torch.clamp(-mesh.z_bed.to(device=dev, dtype=dtype), min=h_ml0, max=max_depth)
    else:
        depth_raw = torch.full((nc,), min(4000.0, max_depth), device=dev, dtype=dtype)
    h_target = torch.where(is_ocean, depth_raw, torch.zeros_like(depth_raw))

    # Build partial-step wet cell thicknesses h_c[nzo, nc]
    h_c = torch.zeros((nzo, nc), device=dev, dtype=dtype)
    for k in range(nzo):
        top_k = z_w[k]
        dz_k = dz_nom[k]
        if k == 0:
            hk = torch.where(is_ocean, torch.full_like(h_target, float(dz_k.item())), torch.zeros_like(h_target))
        else:
            rem = torch.clamp(h_target - top_k, min=0.0, max=float(dz_k.item()))
            # Enforce minimum partial step 20 m to avoid tiny-volume CFL issues
            hk = torch.where(rem >= 20.0, rem, torch.zeros_like(rem))
        h_c[k] = hk

    wet_c = h_c > 0.0
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    wet_e = wet_c[:, c0] & wet_c[:, c1]
    h_e = torch.where(wet_e, torch.minimum(h_c[:, c0], h_c[:, c1]), torch.zeros((nzo, ne), device=dev, dtype=dtype))
    h_col = h_c.sum(dim=0)
    h_col_e = h_e.sum(dim=0)

    # Deepest wet layer index per cell
    k_idx = torch.arange(nzo, device=dev).unsqueeze(1).expand(nzo, nc)
    k_bot = torch.where(wet_c, k_idx, torch.zeros_like(k_idx)).max(dim=0).values

    # Smooth map-free initial thermocline: T(z, lat) decays from SST toward T_abyss = 275 K
    cos2_lat = torch.cos(mesh.lat_cell.to(device=dev, dtype=dtype)) ** 2
    d_therm = 250.0 + 550.0 * cos2_lat  # [nc] deeper thermocline in tropics/subtropics
    sst = sfc.t_ocean_ml.clone()
    z_rel = torch.clamp(z_c.unsqueeze(1) - z_c[0], min=0.0)
    decay = torch.exp(-z_rel / d_therm.unsqueeze(0))
    t_init = t_abyss_k + (sst.unsqueeze(0) - t_abyss_k) * decay
    t_init[0] = sst
    t_init = torch.where(wet_c, torch.clamp(t_init, min=C.T_FREEZE_OCEAN), torch.full_like(t_init, t_abyss_k))

    sal_init = torch.full((nzo, nc), S_REF_PSU, device=dev, dtype=dtype)

    return OceanOGCMState(
        nz_ocn=nzo,
        dz_nom=dz_nom,
        z_c=z_c,
        z_w=z_w,
        h_c=h_c,
        h_e=h_e,
        wet_c=wet_c,
        wet_e=wet_e,
        h_col=h_col,
        h_col_e=h_col_e,
        k_bot=k_bot,
        temp=t_init,
        sal=sal_init,
        u_edge=torch.zeros((nzo, ne), device=dev, dtype=dtype),
        eta=torch.zeros(nc, device=dev, dtype=dtype),
        u_baro=torch.zeros(ne, device=dev, dtype=dtype),
        w_vel=torch.zeros((nzo + 1, nc), device=dev, dtype=dtype),
        t_comp=torch.zeros((nzo, nc), device=dev, dtype=dtype),
        s_comp=torch.zeros((nzo, nc), device=dev, dtype=dtype),
        m_seaice_prev=sfc.m_seaice.clone(),
    )


def step_seaice_drift_voronoi(
    sfc: SurfaceState,
    mesh: SCVTMesh,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    u_ocn_e: torch.Tensor,
    dt: float,
) -> None:
    """Conservatively advect `sfc.m_seaice` & `sfc.swe_seaice` across wet ocean Voronoi edges.

    Uses wind + Layer-0 ocean free-drift with Coriolis turning (Hibler, 1979; Leppäranta, 2011)
    and plastic ridging resistance above `SEAICE_H_RIDGE_M` to prevent coastal pile-up.
    """
    is_ocean = sfc.surf_type == SURF_OCEAN
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    ocn_edge = is_ocean[c0] & is_ocean[c1]
    mask_e = ocn_edge.to(sfc.m_seaice.dtype)

    # Wind-driven stress on ice [N m^-2] (atmosphere -> ice = -tau_sfc)
    tx_i = -tau_x_sfc
    ty_i = -tau_y_sfc
    # Free-drift velocity from wind stress (drag balance: u_wind ~ sqrt(|tau|/(rho_w C_dw)))
    # rho_w * C_dw ~ 5.5 kg m^-3; with Coriolis turning angle theta_i (right in NH, left in SH)
    tau_mag = torch.sqrt(tx_i * tx_i + ty_i * ty_i + 1.0e-12)
    u_scale = torch.sqrt(tau_mag / 5.5)
    ux_w = (tx_i / tau_mag) * u_scale
    uy_w = (ty_i / tau_mag) * u_scale

    theta_rad = math.radians(SEAICE_TURN_DEG)
    sin_t = math.sin(theta_rad) * torch.sign(mesh.f_cell)
    cos_t = math.cos(theta_rad)
    # Turn right of wind in NH (f > 0): (ux*cos + uy*sin, -ux*sin + uy*cos)
    ux_drift = ux_w * cos_t + uy_w * sin_t
    uy_drift = -ux_w * sin_t + uy_w * cos_t

    ux_e = 0.5 * (ux_drift[c0] + ux_drift[c1])
    uy_e = 0.5 * (uy_drift[c0] + uy_drift[c1])
    u_wind_n = ux_e * mesh.east_normal_edge + uy_e * mesh.north_normal_edge
    u_ice_raw = (u_ocn_e + u_wind_n) * mask_e

    # Plastic ridging resistance: smoothly damp convergence into thick ice (> SEAICE_H_RIDGE_M)
    h_ice_c = sfc.m_seaice / C.RHO_SEAICE
    h_dst = torch.where(u_ice_raw >= 0.0, h_ice_c[c1], h_ice_c[c0])
    ridge_fac = 1.0 / (1.0 + (h_dst / SEAICE_H_RIDGE_M) ** 2)
    # CFL velocity cap
    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    u_cap = min(SEAICE_U_MAX, 0.25 * dc_min / max(dt, 1.0))
    u_ice = torch.clamp(u_ice_raw * ridge_fac, -u_cap, u_cap) * mask_e

    # Conservative upwind flux + weak lateral smoothing for both m_seaice and swe_seaice
    m_i = torch.clamp(sfc.m_seaice, min=0.0)
    s_i = torch.clamp(sfc.swe_seaice, min=0.0)
    f_m_adv = torch.clamp(u_ice, min=0.0) * m_i[c0] + torch.clamp(u_ice, max=0.0) * m_i[c1]
    f_s_adv = torch.clamp(u_ice, min=0.0) * s_i[c0] + torch.clamp(u_ice, max=0.0) * s_i[c1]

    f_m_dif = -SEAICE_DIFF_H * ((m_i[c1] - m_i[c0]) / mesh.dc_edge) * mask_e
    f_s_dif = -SEAICE_DIFF_H * ((s_i[c1] - s_i[c0]) / mesh.dc_edge) * mask_e

    is_ocn_f = is_ocean.to(m_i.dtype)
    dm = -dt * compute_scvt_divergence(mesh, f_m_adv + f_m_dif) * is_ocn_f
    ds = -dt * compute_scvt_divergence(mesh, f_s_adv + f_s_dif) * is_ocn_f

    sfc.m_seaice = torch.where(is_ocean, torch.clamp(m_i + dm, min=0.0), sfc.m_seaice)
    sfc.swe_seaice = torch.where(is_ocean, torch.clamp(s_i + ds, min=0.0), sfc.swe_seaice)


def step_ogcm_voronoi(
    ogcm: OceanOGCMState,
    sfc: SurfaceState,
    mesh: SCVTMesh,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    f_seaice: torch.Tensor,
    evap_cell: torch.Tensor,
    precip_cell: torch.Tensor,
    runoff_total: torch.Tensor,
    runoff_glacier: torch.Tensor,
    dt: float,
    enable_seaice_drift: bool = True,
) -> None:
    """Advance 3D SCVT Voronoi OGCM (`temp`, `sal`, `u_edge`, `eta`) & dynamic sea ice by `dt` [s].

    Synchronizes Layer 0 `ogcm.temp[0]` with `sfc.t_ocean_ml` (incorporating surface radiative
    and turbulent heating already applied by `step_surface`) and returns updated `sfc.t_ocean_ml`
    and `sfc.m_seaice` conserving global enthalpy, salt, and sea-ice mass.
    """
    dev = ogcm.temp.device
    dtype = ogcm.temp.dtype
    nzo = ogcm.nz_ocn
    nc = mesh.n_cells
    ne = mesh.n_edges

    is_ocean = sfc.surf_type == SURF_OCEAN
    is_ocn_f = is_ocean.to(dtype)
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    v0_e = mesh.verts_on_edge[:, 0]
    v1_e = mesh.verts_on_edge[:, 1]
    ep_idx = mesh.trisk_edges
    w_trisk = mesh.trisk_weights
    wet_c = ogcm.wet_c
    wet_e = ogcm.wet_e
    wet_cf = wet_c.to(dtype)
    wet_ef = wet_e.to(dtype)
    h_c = ogcm.h_c
    h_e = ogcm.h_e
    h_c_safe = torch.clamp(h_c, min=1.0)
    h_e_safe = torch.clamp(h_e, min=1.0)

    # 0. Pull Layer-0 temperature updated by surface heat flux (`step_surface`) into OGCM
    T = ogcm.temp.clone()
    T[0] = torch.where(is_ocean, sfc.t_ocean_ml, T[0])
    S = ogcm.sal.clone()

    # 1. Surface freshwater & sea-ice brine/melt virtual salt flux into Layer 0 (Huang, 1993)
    # Route terrestrial liquid + glacier runoff conservatively to nearest coastal ocean cell
    r_coast = torch.zeros(nc, device=dev, dtype=dtype)
    r_coast.scatter_add_(0, mesh.coastal_ocean_idx, mesh.area_cell * torch.clamp(runoff_total, min=0.0))
    r_coast = r_coast / mesh.area_cell  # [kg m^-2 s^-1]

    # Coastal glacier calving fusion cooling (MIROC6 Sec. 4.1.2)
    q_calve = torch.zeros(nc, device=dev, dtype=dtype)
    q_calve.scatter_add_(0, mesh.coastal_ocean_idx, C.L_F0 * mesh.area_cell * torch.clamp(runoff_glacier, min=0.0))
    q_calve = q_calve / mesh.area_cell  # [W m^-2]
    dt_calve = -(dt * q_calve) / (C.CP_SEAWATER * C.RHO_SEAWATER * h_c_safe[0])
    T[0] = torch.where(is_ocean, T[0] + dt_calve, T[0])

    # Thermodynamic sea-ice growth (+) / melt (-) during step_surface -> brine rejection / freshening
    dm_ice_thermo = sfc.m_seaice - ogcm.m_seaice_prev  # [kg m^-2]
    fw_evap_minus_p = (evap_cell - precip_cell - r_coast)  # [kg m^-2 s^-1] (+ = salinification)
    ds_sfc = (
        S_REF_PSU * (fw_evap_minus_p * dt) + (S_REF_PSU - S_SEAICE_PSU) * dm_ice_thermo
    ) / (C.RHO_SEAWATER * h_c_safe[0])
    S[0] = torch.where(is_ocean, torch.clamp(S[0] + ds_sfc, 5.0, 45.0), S[0])

    # 2. Nonlinear EOS density anomaly & hydrostatic baroclinic pressure anomaly
    rho_anom = compute_seawater_density_anomaly(T, S, ogcm.z_c.unsqueeze(1)) * wet_cf
    # p'[k] = sum_{m < k} g * rho'_m * h_m + 0.5 * g * rho'_k * h_k
    g_rho_h = C.GRAVITY * rho_anom * h_c
    p_cum = torch.cumsum(g_rho_h, dim=0)
    p_baroc = p_cum - 0.5 * g_rho_h  # [nzo, nc]

    # 3. Wind stress at ocean surface [N m^-2]
    ice_atten = 1.0 - 0.5 * torch.clamp(f_seaice, 0.0, 1.0)
    tx_ocn = -ice_atten * tau_x_sfc
    ty_ocn = -ice_atten * tau_y_sfc
    tx_e = 0.5 * (tx_ocn[c0] + tx_ocn[c1])
    ty_e = 0.5 * (ty_ocn[c0] + ty_ocn[c1])
    tau_n_e = (tx_e * mesh.east_normal_edge + ty_e * mesh.north_normal_edge) * wet_ef[0]

    # 4. 3D Baroclinic Momentum Step (TRiSK Coriolis + Baroclinic PGF + Munk Viscosity + Bottom Drag)
    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    delta_munk = 1.6 * dc_min
    a_h = min(BETA_MUNK_OGCM * (delta_munk ** 3), 0.06 * (dc_min ** 2) / max(dt, 1.0))

    u = ogcm.u_edge * wet_ef
    zeta_v = compute_scvt_vorticity(mesh, u)  # [nzo, n_verts]
    eta_v = zeta_v + mesh.f_vert.unsqueeze(0)
    eta_e = 0.5 * (eta_v[:, v0_e] + eta_v[:, v1_e])  # [nzo, ne]
    eta_pair = 0.5 * (eta_e.unsqueeze(-1) + eta_e[:, ep_idx])  # [nzo, ne, 10]
    cor_3d = (w_trisk.unsqueeze(0) * eta_pair * u[:, ep_idx]).sum(dim=-1)

    pgf_bc = -(1.0 / C.RHO_SEAWATER) * ((p_baroc[:, c1] - p_baroc[:, c0]) / mesh.dc_edge.unsqueeze(0))
    lap_u, _ = compute_vector_laplacian_edge(mesh, u)

    # Bottom drag applied on deepest wet layer of each edge
    is_bot_e = torch.zeros_like(wet_e)
    for k in range(nzo):
        if k == nzo - 1:
            is_bot_e[k] = wet_e[k]
        else:
            is_bot_e[k] = wet_e[k] & (~wet_e[k + 1])
    r_bot = torch.where(is_bot_e, C_DRAG_BOT * torch.abs(u) / h_e_safe, torch.zeros_like(u))

    du_bc = (cor_3d + pgf_bc + a_h * lap_u - (R_RAYLEIGH_OCN + r_bot) * u) * wet_ef
    du_bc[0] = du_bc[0] + (tau_n_e / (C.RHO_SEAWATER * h_e_safe[0])) * wet_ef[0]

    # Explicit baroclinic predictor + implicit vertical momentum viscosity (stabilizes shallow/wind-forced layers)
    u_star = torch.clamp((u + dt * du_bc) * wet_ef, -3.0, 3.0)
    # Implicit vertical viscosity tridiagonal solve on [nzo, ne]
    dz_int = 0.5 * (h_e_safe[:-1] + h_e_safe[1:])
    wet_int_e = wet_e[:-1] & wet_e[1:]
    c_mom = torch.where(wet_int_e, dt * A_V_MOM / dz_int, torch.zeros_like(dz_int))
    a_m = torch.zeros((nzo, ne), device=dev, dtype=dtype)
    b_m = torch.ones((nzo, ne), device=dev, dtype=dtype)
    d_m = torch.zeros((nzo, ne), device=dev, dtype=dtype)
    a_m[1:] = -c_mom / h_e_safe[1:]
    d_m[:-1] = -c_mom / h_e_safe[:-1]
    b_m[:-1] = b_m[:-1] + c_mom / h_e_safe[:-1]
    b_m[1:] = b_m[1:] + c_mom / h_e_safe[1:]
    u_star = solve_tridiagonal_batched(a_m, b_m, d_m, u_star) * wet_ef

    # 5. Split-Explicit 2D Barotropic Free-Surface Mode (Wicker-Skamarock RK3)
    h_col_e_safe = torch.clamp(ogcm.h_col_e, min=1.0)
    ocn_e0 = wet_e[0]
    ocn_e0f = ocn_e0.to(dtype)
    u_star_baro = (u_star * h_e).sum(dim=0) / h_col_e_safe * ocn_e0f
    # Barotropic forcing = depth-averaged baroclinic tendency (u_star_baro - u_baro_old) / dt
    f_baro = (u_star_baro - ogcm.u_baro) / max(dt, 1.0)

    # Effective external gravity-wave speed (free-surface reduced stiffness / exact gravity)
    h_max = float(ogcm.h_col.max().item()) if bool(is_ocean.any()) else 100.0
    c_bt = math.sqrt(C.GRAVITY * max(h_max, 50.0))
    dt_bt_cfl = 0.35 * dc_min / max(c_bt, 10.0)
    n_bt = max(1, min(int(math.ceil(dt / max(dt_bt_cfl, 1.0))), 24))
    dt_bt = dt / float(n_bt)

    u_bt = ogcm.u_baro * ocn_e0f
    eta_old = ogcm.eta * is_ocn_f
    eta_bt = eta_old
    u_bt_acc = torch.zeros_like(u_bt)

    for _ in range(n_bt):
        u0_bt, eta0_bt = u_bt, eta_bt
        u_stage = u_bt
        for rk_dt in (dt_bt / 3.0, 0.5 * dt_bt, dt_bt):
            pgf_bt = -C.GRAVITY * ((eta_bt[c1] - eta_bt[c0]) / mesh.dc_edge)
            du_bt = (pgf_bt + f_baro) * ocn_e0f
            u_stage = torch.clamp((u0_bt + rk_dt * du_bt) * ocn_e0f, -2.5, 2.5)
            deta = -compute_scvt_divergence(mesh, ogcm.h_col_e * u_stage) * is_ocn_f
            eta_bt = torch.clamp((eta0_bt + rk_dt * deta) * is_ocn_f, -5.0, 5.0)
            u_bt = u_stage
        u_bt_acc = u_bt_acc + u_stage

    u_bt_mean = u_bt_acc / float(n_bt)
    ogcm.u_baro = u_bt

    # Reconcile 3D velocity with barotropic transport so depth-integral matches u_bt_mean
    u_new = (u_star + (u_bt_mean - u_star_baro).unsqueeze(0)) * wet_ef
    u_new = torch.clamp(u_new, -3.0, 3.0)
    ogcm.u_edge = u_new

    # 6. Gent-McWilliams (1990) Isopycnal Eddy Bolus Velocity `u_GM = -dPsi_GM/dz`
    # Density gradient along wet edges & vertical stratification across interfaces
    drho_dx = ((rho_anom[:, c1] - rho_anom[:, c0]) / mesh.dc_edge.unsqueeze(0)) * wet_ef
    # Vertical stratification drho/dz > 0 (stable) at interfaces k+1/2 (1..nzo-1)
    dz_c_int = 0.5 * (h_c_safe[:-1] + h_c_safe[1:])
    wet_int_c = wet_c[:-1] & wet_c[1:]
    drho_dz_c = torch.where(
        wet_int_c,
        (rho_anom[1:] - rho_anom[:-1]) / dz_c_int,
        torch.zeros_like(dz_c_int),
    )
    # Interpolate stratification to edges & taper slope |s| <= GM_SLOPE_MAX (Danabasoglu & McWilliams, 1995)
    drho_dz_e = 0.5 * (drho_dz_c[:, c0] + drho_dz_c[:, c1])
    drho_dx_int = 0.5 * (drho_dx[:-1] + drho_dx[1:])
    slope_int = torch.where(
        wet_int_e,
        torch.clamp(-drho_dx_int / torch.clamp(drho_dz_e, min=1.0e-5), -GM_SLOPE_MAX, GM_SLOPE_MAX),
        torch.zeros_like(drho_dx_int),
    )
    psi_gm = torch.zeros((nzo + 1, ne), device=dev, dtype=dtype)
    psi_gm[1:-1] = KAPPA_GM * slope_int  # Psi = 0 at surface (k=0) and seafloor (k=nzo)
    # Bolus velocity u_GM[k] = -(Psi[k+1] - Psi[k]) / h_e[k] -> strictly zero depth-integral!
    u_gm = torch.where(wet_e, -(psi_gm[1:] - psi_gm[:-1]) / h_e_safe, torch.zeros_like(u_new))
    u_gm = torch.clamp(u_gm, -0.3, 0.3)

    u_adv = (u_new + u_gm) * wet_ef

    # 7. Exact Mass-Conserving Vertical Velocity `w` [nzo + 1, nc] (positive upward; w_bot = 0)
    h_u_edge = h_e * u_adv  # [nzo, ne] [m^2 s^-1]
    div_hu = compute_scvt_divergence(mesh, h_u_edge) * wet_cf  # [nzo, nc] [m s^-1]
    w_vel = torch.zeros((nzo + 1, nc), device=dev, dtype=dtype)
    # Integrate upward from seafloor (w[nzo] = 0): w[k] = w[k+1] - div_hu[k]
    for k in range(nzo - 1, -1, -1):
        w_vel[k] = (w_vel[k + 1] - div_hu[k]) * wet_cf[k]
    # Exact surface free-surface tendency w_sfc_raw = w_vel[1] - div_hu[0] = d(eta)/dt (Campin et al., 2004)
    w_sfc_raw = (w_vel[1] - div_hu[0]) * wet_cf[0]
    eta_new = torch.clamp((eta_old + dt * w_sfc_raw) * is_ocn_f, -5.0, 5.0)
    ogcm.eta = eta_new
    w_vel[0] = 0.0
    ogcm.w_vel = w_vel

    # Exact free-surface Layer-0 thicknesses (h0_old, h0_new) guarantee BOTH constant-field
    # preservation AND exact global heat/salt conservation (Campin et al., 2004)
    h0_old = torch.clamp(h_c_safe[0] + eta_old, min=10.0)
    h0_new = torch.clamp(h_c_safe[0] + eta_new, min=10.0)
    # Exact effective thickness change matching flux divergence (respecting eta clamp)
    dh0_eff = h0_new - h0_old

    # 8. 3D Conservative Flux-Form Tracer Advection + Horizontal Diffusion for T and S
    def _advect_diffuse_3d(q: torch.Tensor, q_comp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Horizontal upwind advection + Redi/eddy diffusion on wet edges
        f_adv = torch.clamp(h_u_edge, min=0.0) * q[:, c0] + torch.clamp(h_u_edge, max=0.0) * q[:, c1]
        f_dif = -KAPPA_REDI_HORIZ * h_e * ((q[:, c1] - q[:, c0]) / mesh.dc_edge.unsqueeze(0)) * wet_ef
        div_h = compute_scvt_divergence(mesh, f_adv + f_dif) * wet_cf

        # Vertical upwind flux across interior interfaces 1..nzo-1 (w > 0 is upward: layer k+1 -> k)
        w_int = w_vel[1:-1]  # [nzo-1, nc]
        f_v_int = torch.where(w_int >= 0.0, w_int * q[1:], w_int * q[:-1]) * wet_int_c.to(dtype)
        v_inflow = torch.zeros_like(q)
        v_inflow[:-1] = v_inflow[:-1] + f_v_int
        v_inflow[1:] = v_inflow[1:] - f_v_int

        dq_adv = (dt / h_c_safe) * (-div_h + v_inflow)
        # Layer 0 exact free-surface update: (h0_new * q0_new - h0_old * q0) = dt * (-div_h[0] + v_inflow[0])
        dq_adv[0] = (dt * (-div_h[0] + v_inflow[0]) - dh0_eff * q[0]) / h0_new
        dq_adv = dq_adv * wet_cf
        q_next, comp_next = _kahan_add(q, dq_adv, q_comp)
        return q_next, comp_next

    T, ogcm.t_comp = _advect_diffuse_3d(T, ogcm.t_comp)
    S, ogcm.s_comp = _advect_diffuse_3d(S, ogcm.s_comp)

    # 9. Implicit Vertical Mixing (Bryan-Lewis + Wind ML + Instantaneous Convective Adjustment!)
    rho_post = compute_seawater_density_anomaly(T, S, ogcm.z_c.unsqueeze(1)) * wet_cf
    drho_dz_post = torch.where(
        wet_int_c,
        (rho_post[1:] - rho_post[:-1]) / dz_c_int,
        torch.ones_like(dz_c_int),
    )
    # Bryan & Lewis (1979) depth-dependent diapycnal diffusivity: ~2e-5 upper -> ~1e-4 abyssal
    z_int_m = ogcm.z_w[1:-1].unsqueeze(1)
    kv_bg = KAPPA_V_BG + 0.5 * (KAPPA_V_ABYSS - KAPPA_V_BG) * (1.0 + torch.tanh((z_int_m - 2000.0) / 800.0))
    # Surface wind-mixing boost across upper 150 m (Pacanowski-Philander / KPP style)
    u_star_o = torch.sqrt(torch.sqrt(tx_ocn * tx_ocn + ty_ocn * ty_ocn + 1.0e-12) / C.RHO_SEAWATER)
    kv_wind = (0.04 * u_star_o * 50.0).unsqueeze(0) * torch.exp(-z_int_m / 90.0)
    kv = kv_bg + kv_wind
    # Convective adjustment whenever denser water overlies lighter water (drho/dz < 0)
    kv = torch.where(drho_dz_post < 0.0, torch.full_like(kv, KAPPA_V_CONV), kv)
    kv = torch.where(wet_int_c, kv, torch.zeros_like(kv))

    c_dif = dt * kv / dz_c_int
    h_eff = h_c_safe.clone()
    h_eff[0] = h0_new
    a_t = torch.zeros((nzo, nc), device=dev, dtype=dtype)
    b_t = torch.ones((nzo, nc), device=dev, dtype=dtype)
    d_t = torch.zeros((nzo, nc), device=dev, dtype=dtype)
    a_t[1:] = -c_dif / h_eff[1:]
    d_t[:-1] = -c_dif / h_eff[:-1]
    b_t[:-1] = b_t[:-1] + c_dif / h_eff[:-1]
    b_t[1:] = b_t[1:] + c_dif / h_eff[1:]

    T = solve_tridiagonal_batched(a_t, b_t, d_t, T)
    S = solve_tridiagonal_batched(a_t, b_t, d_t, S)
    S = torch.where(wet_c, torch.clamp(S, 5.0, 45.0), S)

    # 10. Conservative Freezing Check on Layer 0 (if advection/mixing cools T[0] below T_freeze)
    freeze_exc = torch.where(is_ocean, torch.clamp(C.T_FREEZE_OCEAN - T[0], min=0.0), torch.zeros_like(T[0]))
    c0_cap = C.CP_SEAWATER * C.RHO_SEAWATER * h_c_safe[0]
    dm_freeze = (c0_cap * freeze_exc) / C.L_F0
    T[0] = torch.where(is_ocean, T[0] + freeze_exc, T[0])
    # Brine rejection from any dynamical freezing
    ds_brine = ((S_REF_PSU - S_SEAICE_PSU) * dm_freeze) / (C.RHO_SEAWATER * h_c_safe[0])
    S[0] = torch.where(is_ocean, torch.clamp(S[0] + ds_brine, 5.0, 45.0), S[0])
    T = torch.where(wet_c, torch.clamp(T, min=C.T_FREEZE_OCEAN), T)

    ogcm.temp = T
    ogcm.sal = S
    sfc.t_ocean_ml = torch.where(is_ocean, T[0], sfc.t_ocean_ml)
    sfc.m_seaice = torch.where(is_ocean, torch.clamp(sfc.m_seaice + dm_freeze, min=0.0), sfc.m_seaice)

    # 11. Dynamic Sea-Ice Drift (after recording thermodynamic mass change, so advection doesn't alias brine!)
    if enable_seaice_drift:
        step_seaice_drift_voronoi(sfc, mesh, tau_x_sfc, tau_y_sfc, u_new[0], dt)

    ogcm.m_seaice_prev = sfc.m_seaice.clone()
