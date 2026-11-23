"""3D Multi-Layer Hydrostatic Boussinesq SCVT Voronoi OGCM & Dynamic Sea-Ice Drift.

Upgrades AOBA's ocean component from the 1.5-layer Ekman + reduced-gravity shallow-water
model (`ocean_ekman.py`, retained as `ocean_mode = "ekman_sw"`) to a full-depth 3D
z-level primitive-equation ocean (`ocean_mode = "ogcm_3d"`) on the orthogonal SCVT
Voronoi C-grid using `Earth2014 BED2014` global bathymetry (`mesh.z_bed`).

Physics & Numerics (Zero Q-Flux, Strictly Conservative):
1. **Vertical Grid & Bathymetry**: `N_zo` stretched z-levels (default 18 levels spanning
   0-6000 m; Layer 0 thickness = `h_ml_ocean`) with exact wet-cell partial-step
   thicknesses `h_c[N_zo, N_cells]` masked by `Earth2014 BED2014` bathymetry and zero
   normal flux across coastlines and submarine ridges.
2. **Nonlinear Equation of State** (Roquet et al., 2015, *Ocean Modelling*, 90, 29-43):
   Simplified TEOS-10 Boussinesq polynomial capturing thermal expansion, cabbeling,
   thermobaricity, and haline contraction (driving polar deep convection & AMOC).
3. **TRiSK SCVT C-Grid Split-Explicit Dynamics** (Ringler et al., 2010, 2013, *Ocean
   Modelling*, 69, 211-232):
   - 3D baroclinic momentum with hydrostatic pressure gradient, energy-conserving TRiSK
     Coriolis/vorticity, scale-aware Munk horizontal viscosity, quadratic bottom drag,
     and implicit KPP + shear-dependent vertical momentum viscosity.
   - Fast 2D barotropic free-surface mode `(u_baro, eta)` substepped with Wicker &
     Skamarock (2002) RK3, ensuring exact 3D continuity `w_{k-1/2} = w_{k+1/2} - div(h u)`.
4. **Gent-McWilliams (1990) Eddy Bolus & Full Rotated Redi (1982) Isopycnal Tensor**
   (Gent & McWilliams, 1990, *JPO*, 20, 150-155; Redi, 1982, *JPO*, 12, 1154-1158;
   Danabasoglu & McWilliams, 1995; Griffies et al., 1998, *JPO*, 28, 805-820;
   Ferrari et al., 2008):
   - Boundary-layer-tapered neutral slope `S_x = -drho_dx / drho_dz` yields a strictly
     non-divergent GM bolus velocity `u_GM = -dPsi_GM/dz` and the full small-slope Redi
     rotated diffusion tensor `F_Redi = -kappa_Redi [[1, S_x], [S_x, S_x^2]] grad(q)`,
     eliminating spurious z-level diapycnal (Veronis) mixing while folding the diagonal
     `kappa_Redi * |S|^2` term into the unconditionally stable implicit vertical solver.
5. **KPP Boundary Layer, Shear Instability, Topographic Internal-Tide Mixing & Penetrative
   Shortwave Absorption** (Large, McWilliams, & Doney, 1994, *Rev. Geophys.*, 32, 363-403;
   Pacanowski & Philander, 1981; Paulson & Simpson, 1977, *JPO*, 7, 952-956;
   St. Laurent et al., 2002; Simmons et al., 2004, *Ocean Modelling*, 6, 245-263; Huang, 1993):
   - Diagnoses KPP boundary-layer depth `h_BL` via bulk Richardson number `Ri_b <= 0.30`,
     cubic KPP diffusivity profile `h_BL * w_s * G(sigma)`, and conservative non-local
     convective counter-gradient transport `gamma_T, gamma_S` under surface buoyancy loss.
   - Interior gradient Richardson-number shear mixing (`Ri_g < 0.70`, Large et al., 1994, Eq. 28).
   - Bottom-intensified topographic internal-tide deep mixing driven by `Earth2014 BED2014`
     bathymetric slope `|grad H|` (Simmons et al., 2004) over Bryan-Lewis background.
   - Conservative Jerlov Type I two-exponential shortwave solar penetration redistributing
     transmitted solar radiation below Layer 0 without perturbing total column enthalpy.
   - Instantaneous implicit convective adjustment (`K_v -> K_conv`) wherever `drho/dz < 0`.
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
from aoba.scvt_mesh import SCVTMesh, reconstruct_cell_winds
from aoba.surface import SURF_OCEAN, SurfaceState, _kahan_add

# ---------------------------------------------------------------------------
# Physical Constants & Default Parameters (TOML-exposed in [ocean_ogcm])
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

# Mesoscale eddy parameterization (GM90 + Full Rotated Redi 1982)
KAPPA_GM: float = 800.0             # Gent-McWilliams isopycnal thickness diffusivity [m^2 s^-1]
KAPPA_REDI_HORIZ: float = 1000.0    # Redi isopycnal tracer diffusivity [m^2 s^-1]
GM_SLOPE_MAX: float = 1.0e-2        # Danabasoglu & McWilliams (1995) neutral slope cap [-]
ENABLE_FULL_REDI: bool = True       # Full rotated Redi tensor (False reverts to z-level diffusion)

# Vertical mixing: KPP (Large et al., 1994) + Shear + Simmons (2004) Internal Tide + Background
KPP_RI_CRIT: float = 0.30           # KPP critical bulk Richardson number (Large et al., 1994) [-]
KPP_CS_NONLOCAL: float = 6.33       # KPP non-local convective transport coefficient C_s [-]
KPP_KV_MAX: float = 0.50            # Upper cap on KPP boundary-layer diffusivity [m^2 s^-1]
SHEAR_RI_0: float = 0.70            # Critical gradient Ri for interior shear mixing (Large et al., 1994) [-]
SHEAR_NU_0: float = 5.0e-3          # Peak interior shear diffusivity/viscosity at Ri -> 0 [m^2 s^-1]
TIDAL_E0_WM2: float = 8.0e-3        # Reference internal-tide energy flux over ridge slope (Simmons 2004) [W m^-2]
TIDAL_SLOPE_REF: float = 4.0e-3     # Reference topographic slope |grad H| for tidal drag [-]
TIDAL_ZETA_M: float = 500.0         # Vertical e-folding decay scale above seafloor [m]
TIDAL_KV_MAX: float = 5.0e-3        # Upper cap on abyssal topographic tidal diffusivity [m^2 s^-1]

KAPPA_V_BG: float = 1.5e-5          # Upper-ocean background diapycnal diffusivity [m^2 s^-1]
KAPPA_V_ABYSS: float = 3.0e-5       # Background abyssal floor (supplemented by tidal mixing) [m^2 s^-1]
KAPPA_V_CONV: float = 1.0           # Convective adjustment diffusivity when drho/dz < 0 [m^2 s^-1]
A_V_MOM: float = 1.0e-3             # Background interior vertical momentum viscosity [m^2 s^-1]
C_DRAG_BOT: float = 2.5e-3          # Quadratic bottom friction drag coefficient [-]
R_RAYLEIGH_OCN: float = 1.0 / (120.0 * C.SECONDS_PER_DAY)  # Weak background drag [s^-1]
BETA_MUNK_OGCM: float = 2.28e-11    # Equatorial beta for scale-aware Munk viscosity [m^-1 s^-1]

# Penetrative shortwave solar absorption (Paulson & Simpson, 1977, Jerlov Type I clear open ocean)
SW_PEN_R: float = 0.58              # Red/IR rapidly attenuated fraction [-]
SW_PEN_ZETA1_M: float = 0.35        # Attenuation depth of red/IR band [m]
SW_PEN_ZETA2_M: float = 23.0        # Attenuation depth of blue-green penetrating band [m]

# Dynamic sea-ice drift parameters (Hibler, 1979; Leppäranta, 2011)
SEAICE_WIND_FACTOR: float = 0.018   # Nansen wind-drift coupling factor (~1.8% of u_a / tau) [-]
SEAICE_TURN_DEG: float = 20.0       # Wind-to-ice Coriolis turning angle [deg]
SEAICE_H_RIDGE_M: float = 3.0       # Ridging resistance thickness scale [m]
SEAICE_U_MAX: float = 0.50          # Maximum sea-ice drift speed [m s^-1]
SEAICE_DIFF_H: float = 500.0        # Subgrid sea-ice lateral diffusivity [m^2 s^-1]


def build_ogcm_dz_levels(nz_ocn: int = 18, h_ml0: float = 50.0, max_depth: float = 5500.0) -> torch.Tensor:
    """Construct stretched z-level nominal thicknesses `[nz_ocn]` [m] starting at `h_ml0`."""
    if nz_ocn == len(OGCM_DEFAULT_DZ) and abs(h_ml0 - 50.0) < 1.0e-3 and max_depth >= 5500.0:
        return torch.tensor(OGCM_DEFAULT_DZ, dtype=torch.float32)
    nz = max(int(nz_ocn), 4)
    # Geometric stretch from h_ml0 to span max_depth
    r_lo, r_hi = 1.001, 2.0
    for _ in range(40):
        r_mid = 0.5 * (r_lo + r_hi)
        tot = h_ml0 * (r_mid ** nz - 1.0) / (r_mid - 1.0)
        if tot < max_depth:
            r_lo = r_mid
        else:
            r_hi = r_mid
    r = 0.5 * (r_lo + r_hi)
    dz = [h_ml0 * (r ** k) for k in range(nz)]
    dz[0] = float(h_ml0)
    return torch.tensor(dz, dtype=torch.float32)


def compute_seawater_density_anomaly(
    temp_k: torch.Tensor,
    sal_psu: torch.Tensor,
    depth_m: torch.Tensor,
) -> torch.Tensor:
    """Roquet et al. (2015, Table 1) simplified nonlinear TEOS-10 Boussinesq density anomaly [kg m^-3].

    Computes `rho' = rho(T, S, z) - rho_0` capturing:
    - Thermal expansion (`alpha_0`),
    - Nonlinear cabbeling (`alpha_T * dT^2`, crucial for polar dense-water formation),
    - Thermobaricity (`alpha_z * z * dT`, deep-water compressibility),
    - Haline contraction (`beta_0 * dS`) and thermohaline cross-term (`alpha_S * dT * dS`).
    """
    dt = temp_k - T_REF_EOS_K
    ds = sal_psu - S_REF_PSU
    alpha_eff = EOS_ALPHA0 + EOS_ALPHA_T * dt + EOS_ALPHA_Z * depth_m + EOS_ALPHA_S * ds
    return C.RHO_SEAWATER * (-alpha_eff * dt + EOS_BETA0 * ds)


def compute_shortwave_transmission(z_m: torch.Tensor) -> torch.Tensor:
    """Paulson & Simpson (1977) Jerlov Type I downward shortwave transmission fraction `I(z) in [0, 1]`."""
    z_pos = torch.clamp(z_m, min=0.0)
    return SW_PEN_R * torch.exp(-z_pos / SW_PEN_ZETA1_M) + (1.0 - SW_PEN_R) * torch.exp(-z_pos / SW_PEN_ZETA2_M)


@dataclass
class OceanOGCMState:
    """3D SCVT Voronoi Boussinesq OGCM state (strictly conservative, zero Q-flux)."""

    nz_ocn: int                  # Number of z-levels (default 18)
    dz_nom: torch.Tensor         # Nominal layer thicknesses [N_zo] [m]
    z_c: torch.Tensor            # Nominal layer-center depths [N_zo] [m]
    z_w: torch.Tensor            # Nominal interface depths [N_zo+1] [m]
    h_c: torch.Tensor            # Wet partial-step cell thicknesses [N_zo, N_cells] [m] (0 on dry/land)
    h_e: torch.Tensor            # Wet edge thicknesses [N_zo, N_edges] [m]
    wet_c: torch.Tensor          # Boolean wet cell mask [N_zo, N_cells]
    wet_e: torch.Tensor          # Boolean wet edge mask [N_zo, N_edges]
    h_col: torch.Tensor          # Total wet water-column depth [N_cells] [m]
    h_col_e: torch.Tensor        # Total wet water-column depth on edges [N_edges] [m]
    k_bot: torch.Tensor          # Deepest wet layer index per cell [N_cells] (int64)

    # 3D Prognostic fields
    temp: torch.Tensor           # 3D potential temperature [N_zo, N_cells] [K]
    sal: torch.Tensor            # 3D practical salinity [N_zo, N_cells] [psu]
    u_edge: torch.Tensor         # 3D edge-normal velocity [N_zo, N_edges] [m s^-1]
    eta: torch.Tensor            # Free-surface height anomaly [N_cells] [m]
    u_baro: torch.Tensor         # Barotropic depth-mean normal velocity [N_edges] [m s^-1]

    # Diagnostics, topographic mixing geometry & Kahan float32 rounding compensation
    w_vel: torch.Tensor          # Vertical velocity at interfaces (positive upward) [N_zo+1, N_cells] [m s^-1]
    t_comp: torch.Tensor         # Kahan compensation for 3D temp [N_zo, N_cells] [K]
    s_comp: torch.Tensor         # Kahan compensation for 3D sal [N_zo, N_cells] [psu]
    m_seaice_prev: torch.Tensor  # Previous-step cell-mean sea-ice mass for brine/melt salt closure [N_cells] [kg m^-2]
    bathym_slope: torch.Tensor | None = None  # Bathymetric slope `|grad H|` for internal-tide mixing [N_cells] [-]
    h_bl: torch.Tensor | None = None          # Diagnosed KPP surface boundary-layer depth [N_cells] [m]
    kv_last: torch.Tensor | None = None       # Last diagnosed vertical tracer diffusivity [N_zo-1, N_cells] [m^2 s^-1]


def _compute_bathymetric_slope(h_col: torch.Tensor, wet_c0: torch.Tensor, mesh: SCVTMesh) -> torch.Tensor:
    """Compute isotropic Voronoi bathymetric slope `|grad H|` [N_cells] for Simmons (2004) tidal mixing."""
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    ocn_e = (wet_c0[c0] & wet_c0[c1]).to(h_col.dtype)
    dh_dx = ((h_col[c1] - h_col[c0]) / mesh.dc_edge) * ocn_e  # [n_edges]
    # Voronoi cell isotropic gradient norm^2: sum_{e in i} (0.5 * dc_e * dv_e / A_i) * (dh/dx_e)^2
    w_e = 0.5 * mesh.dc_edge * mesh.dv_edge
    active_c = (mesh.edge_sign_on_cell != 0.0).to(h_col.dtype)
    eoc = mesh.edges_on_cell
    grad2 = ((w_e * dh_dx * dh_dx)[eoc] * active_c).sum(dim=-1) / mesh.area_cell
    return torch.sqrt(torch.clamp(grad2, min=0.0)) * wet_c0.to(h_col.dtype)


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
    bathym_slope = _compute_bathymetric_slope(h_col, wet_c[0], mesh)

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
        bathym_slope=bathym_slope,
        h_bl=torch.where(is_ocean, h_c[0], torch.zeros(nc, device=dev, dtype=dtype)),
        kv_last=torch.zeros((nzo - 1, nc), device=dev, dtype=dtype),
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


def _diagnose_kpp_and_vertical_mixing(
    T: torch.Tensor,
    S: torch.Tensor,
    rho_anom: torch.Tensor,
    u_cell: torch.Tensor,
    v_cell: torch.Tensor,
    u_star_o: torch.Tensor,
    b0_sfc: torch.Tensor,
    wt_sfc: torch.Tensor,
    ws_sfc: torch.Tensor,
    ogcm: OceanOGCMState,
    mesh: SCVTMesh,
    is_ocean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnose KPP boundary layer, non-local convective fluxes, Ri shear mixing, and tidal mixing.

    References
    ----------
    - Large, McWilliams, & Doney (1994, Rev. Geophys., 32, Eqs. 19-28): KPP bulk Richardson
      boundary-layer depth `h_BL`, cubic shape `G(sigma)`, non-local convective transport
      `gamma_T, gamma_S`, and interior gradient-Ri shear instability mixing.
    - St. Laurent, Simmons, & Jayne (2002) & Simmons et al. (2004, Ocean Modelling, 6, Eqs. 3-6):
      Bottom-intensified topographic internal-tide diapycnal diffusivity from `|grad H|`.

    Returns
    -------
    kv_total : [N_zo-1, N_cells]  Vertical tracer diffusivity (excl. Redi S^2) [m^2 s^-1]
    av_total : [N_zo-1, N_cells]  Vertical momentum viscosity [m^2 s^-1]
    h_bl     : [N_cells]          Diagnosed KPP boundary-layer depth [m]
    f_nl_t   : [N_zo-1, N_cells]  Upward KPP non-local heat flux at interior interfaces [K m s^-1]
    f_nl_s   : [N_zo-1, N_cells]  Upward KPP non-local salt flux at interior interfaces [psu m s^-1]
    """
    dtype = T.dtype
    nzo = ogcm.nz_ocn
    wet_c = ogcm.wet_c
    wet_int_c = wet_c[:-1] & wet_c[1:]
    h_c_safe = torch.clamp(ogcm.h_c, min=1.0)
    dz_c_int = 0.5 * (h_c_safe[:-1] + h_c_safe[1:])  # [nzo-1, nc]
    z_c_2d = ogcm.z_c.unsqueeze(1)                   # [nzo, 1]
    z_int_m = ogcm.z_w[1:-1].unsqueeze(1)            # [nzo-1, 1]
    is_ocn_f = is_ocean.to(dtype)

    # 1. Stratification N^2 [s^-2] and resolved vertical velocity shear S^2 [s^-2] at interior interfaces
    drho_dz = torch.where(
        wet_int_c,
        (rho_anom[1:] - rho_anom[:-1]) / dz_c_int,
        torch.zeros_like(dz_c_int),
    )
    n2_int = (C.GRAVITY / C.RHO_SEAWATER) * torch.clamp(drho_dz, min=0.0)
    du_dz = (u_cell[1:] - u_cell[:-1]) / dz_c_int
    dv_dz = (v_cell[1:] - v_cell[:-1]) / dz_c_int
    s2_int = (du_dz * du_dz + dv_dz * dv_dz) * wet_int_c.to(dtype)

    # 2. KPP Boundary-Layer Depth h_BL via Bulk Richardson Number (Large et al., 1994, Eqs. 21-24)
    # Buoyancy difference relative to Layer 0: db_k = (g / rho_0) * (rho_k - rho_0) (>0 when stable)
    db_k = (C.GRAVITY / C.RHO_SEAWATER) * (rho_anom - rho_anom[0:1])  # [nzo, nc]
    dv2_k = (u_cell - u_cell[0:1]) ** 2 + (v_cell - v_cell[0:1]) ** 2  # [nzo, nc]
    # Unresolved turbulent velocity scale V_t^2 (Large et al., 1994, Eq. 23)
    n2_cell = torch.zeros_like(T)
    n2_cell[0] = n2_int[0]
    n2_cell[1:-1] = 0.5 * (n2_int[:-1] + n2_int[1:])
    n2_cell[-1] = n2_int[-1]
    w_star_est = torch.pow(torch.clamp(-b0_sfc, min=0.0) * torch.clamp(ogcm.h_col, min=50.0, max=500.0), 1.0 / 3.0)
    w_s0 = C.VON_KARMAN * torch.pow(u_star_o ** 3 + 2.0 * (w_star_est ** 3) + 1.0e-12, 1.0 / 3.0)
    vt2_k = 0.08 * torch.sqrt(torch.clamp(n2_cell, min=0.0)) * z_c_2d * w_s0.unsqueeze(0)
    ri_b = (db_k * z_c_2d) / (dv2_k + vt2_k + 1.0e-6)

    # Deepest contiguous wet layer where Ri_b < KPP_RI_CRIT (or unstable db_k <= 0)
    in_bl = wet_c & ((ri_b < KPP_RI_CRIT) | (db_k <= 0.0))
    # Enforce contiguity downward from Layer 0 so deep inversions don't detach h_BL
    in_bl_contig = torch.cumprod(in_bl.to(torch.int32), dim=0).to(torch.bool)
    in_bl_contig[0] = wet_c[0]
    h_bl_ri = (ogcm.h_c * in_bl_contig.to(dtype)).sum(dim=0)

    # Under stabilizing surface buoyancy forcing (B_0 > 0), cap by Ekman & Monin-Obukhov scales (Large et al., 1994, Eq. 24)
    f_abs = torch.clamp(torch.abs(mesh.f_cell.to(dtype)), min=1.0e-5)
    h_ek = 0.7 * u_star_o / f_abs
    h_mo = (u_star_o ** 3) / (C.VON_KARMAN * torch.clamp(b0_sfc, min=1.0e-12))
    h_stab_cap = torch.minimum(h_ek, h_mo)
    h_bl = torch.where(b0_sfc > 1.0e-11, torch.minimum(h_bl_ri, torch.maximum(h_stab_cap, ogcm.h_c[0])), h_bl_ri)
    h_bl = torch.clamp(h_bl, min=ogcm.h_c[0], max=torch.clamp(ogcm.h_col, min=1.0)) * is_ocn_f

    # 3. KPP Cubic Boundary-Layer Diffusivity K_v,KPP(sigma) & Non-Local Fluxes (Large et al., 1994, Eqs. 19-20)
    sigma = z_int_m / torch.clamp(h_bl.unsqueeze(0), min=1.0)  # [nzo-1, nc]
    inside_bl_int = wet_int_c & (sigma > 0.0) & (sigma < 1.0)
    sig_c = torch.clamp(sigma, 0.0, 1.0)
    g_shape = sig_c * (1.0 - sig_c) ** 2  # G(0)=0, G(1)=0, peak 4/27 at sigma=1/3

    # Turbulent velocity scale w_s(sigma) (Large et al., 1994, Eq. 13 / convective scaling)
    conv_scale = 16.0 * C.VON_KARMAN * sig_c * torch.clamp(-b0_sfc, min=0.0).unsqueeze(0) * h_bl.unsqueeze(0)
    w_s_int = C.VON_KARMAN * torch.pow(u_star_o.unsqueeze(0) ** 3 + conv_scale + 1.0e-12, 1.0 / 3.0)
    kv_kpp = torch.where(
        inside_bl_int,
        torch.clamp(h_bl.unsqueeze(0) * w_s_int * g_shape, min=0.0, max=KPP_KV_MAX),
        torch.zeros_like(z_int_m.expand_as(n2_int)),
    )

    # Non-local convective transport active ONLY under surface buoyancy loss (B_0 < 0, Large et al., 1994, Eq. 20)
    # Surface cooling (wt_sfc < 0) drives sinking cold plumes -> positive upward heat flux (-C_s * G * wt_sfc > 0);
    # brine rejection (ws_sfc > 0) drives sinking salty plumes -> negative upward salt flux (-C_s * G * ws_sfc < 0).
    is_conv_sfc = (b0_sfc < -1.0e-11).unsqueeze(0) & inside_bl_int
    nl_weight = torch.where(is_conv_sfc, KPP_CS_NONLOCAL * g_shape, torch.zeros_like(g_shape))
    f_nl_t = -nl_weight * wt_sfc.unsqueeze(0)
    f_nl_s = -nl_weight * ws_sfc.unsqueeze(0)

    # 4. Interior Gradient-Richardson Shear Instability Mixing (Large et al., 1994, Eq. 28; Pacanowski & Philander, 1981)
    ri_g = n2_int / (s2_int + 1.0e-10)
    ri_ratio = torch.clamp(ri_g / SHEAR_RI_0, 0.0, 1.0)
    kv_shear = torch.where(
        wet_int_c,
        SHEAR_NU_0 * ((1.0 - ri_ratio * ri_ratio) ** 3),
        torch.zeros_like(n2_int),
    )

    # 5. Topographic Internal-Tide Abyssal Mixing (St. Laurent et al., 2002; Simmons et al., 2004, Eqs. 3-6)
    slope_b = ogcm.bathym_slope
    if slope_b is None:
        slope_b = _compute_bathymetric_slope(ogcm.h_col, wet_c[0], mesh)
        ogcm.bathym_slope = slope_b
    # Energy flux scales with bathymetric roughness (|grad H| / s_ref)^2 (capped at 16x over steep ridges)
    e_tide = TIDAL_E0_WM2 * (torch.clamp(slope_b / TIDAL_SLOPE_REF, 0.0, 4.0) ** 2)  # [W m^-2]
    # Vertical structure decaying exponentially upward from local seafloor H_col with scale zeta = 500 m
    hab = torch.clamp(ogcm.h_col.unsqueeze(0) - z_int_m, min=0.0)  # height above seafloor [m]
    norm_f = TIDAL_ZETA_M * torch.clamp(1.0 - torch.exp(-ogcm.h_col.unsqueeze(0) / TIDAL_ZETA_M), min=1.0e-2)
    f_z_tide = torch.exp(-hab / TIDAL_ZETA_M) / norm_f
    # Gamma * q_loc = 0.20 * (1/3) = 0.0667 (St. Laurent et al., 2002)
    kv_tide = (0.0667 * e_tide.unsqueeze(0) * f_z_tide) / (C.RHO_SEAWATER * torch.clamp(n2_int, min=1.0e-7))
    kv_tide = torch.where(wet_int_c, torch.clamp(kv_tide, min=0.0, max=TIDAL_KV_MAX), torch.zeros_like(n2_int))

    # 6. Background Bryan & Lewis (1979) floor + Convective Adjustment whenever drho/dz < 0
    kv_bg = KAPPA_V_BG + 0.5 * (KAPPA_V_ABYSS - KAPPA_V_BG) * (1.0 + torch.tanh((z_int_m - 2000.0) / 800.0))
    kv_total = kv_bg + kv_tide + kv_kpp + kv_shear
    kv_total = torch.where(drho_dz < 0.0, torch.maximum(kv_total, torch.full_like(kv_total, KAPPA_V_CONV)), kv_total)
    kv_total = torch.where(wet_int_c, kv_total, torch.zeros_like(kv_total))

    # Momentum eddy viscosity A_v (background + KPP + shear; Prandtl ~ 1 in BL, >= A_V_MOM in interior)
    av_total = torch.where(wet_int_c, A_V_MOM + kv_kpp + kv_shear, torch.zeros_like(kv_total))
    return kv_total, av_total, h_bl, f_nl_t, f_nl_s


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
    sw_net_ocn: torch.Tensor | None = None,
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
    wet_int_c = wet_c[:-1] & wet_c[1:]
    wet_int_e = wet_e[:-1] & wet_e[1:]
    dz_c_int = 0.5 * (h_c_safe[:-1] + h_c_safe[1:])

    # 0. Record previous Layer-0 state for KPP surface buoyancy forcing diagnosis, then pull surface update
    t0_prev = ogcm.temp[0].clone()
    s0_prev = ogcm.sal[0].clone()
    T = ogcm.temp.clone()
    T[0] = torch.where(is_ocean, sfc.t_ocean_ml, T[0])
    S = ogcm.sal.clone()

    # 1. Surface freshwater, glacier calving cooling & sea-ice brine/melt virtual salt flux (Huang, 1993)
    r_coast = torch.zeros(nc, device=dev, dtype=dtype)
    r_coast.scatter_add_(0, mesh.coastal_ocean_idx, mesh.area_cell * torch.clamp(runoff_total, min=0.0))
    r_coast = r_coast / mesh.area_cell  # [kg m^-2 s^-1]

    q_calve = torch.zeros(nc, device=dev, dtype=dtype)
    q_calve.scatter_add_(0, mesh.coastal_ocean_idx, C.L_F0 * mesh.area_cell * torch.clamp(runoff_glacier, min=0.0))
    q_calve = q_calve / mesh.area_cell  # [W m^-2]
    dt_calve = -(dt * q_calve) / (C.CP_SEAWATER * C.RHO_SEAWATER * h_c_safe[0])
    T[0] = torch.where(is_ocean, T[0] + dt_calve, T[0])

    dm_ice_thermo = sfc.m_seaice - ogcm.m_seaice_prev  # [kg m^-2]
    fw_evap_minus_p = evap_cell - precip_cell - r_coast  # [kg m^-2 s^-1] (+ = salinification)
    ds_sfc = (
        S_REF_PSU * (fw_evap_minus_p * dt) + (S_REF_PSU - S_SEAICE_PSU) * dm_ice_thermo
    ) / (C.RHO_SEAWATER * h_c_safe[0])
    S[0] = torch.where(is_ocean, torch.clamp(S[0] + ds_sfc, 5.0, 45.0), S[0])

    # 1b. Conservative Jerlov Type I Shortwave Penetration below Layer 0 (Paulson & Simpson, 1977)
    # Redistributes `I(z_w[1]) * sw_net_ocn` from Layer 0 into underlying wet layers -> exact column enthalpy conservation!
    if sw_net_ocn is not None and nzo >= 2:
        sw_pos = torch.clamp(sw_net_ocn, min=0.0) * is_ocn_f
        trans_w = compute_shortwave_transmission(ogcm.z_w)  # [nzo + 1]
        # Fraction absorbed in each subsurface wet layer k >= 1
        frac_sub = torch.zeros((nzo, nc), device=dev, dtype=dtype)
        for k in range(1, nzo):
            # If layer k is deepest wet layer, absorb any remaining light reaching seafloor
            is_deepest = wet_c[k] & (~wet_c[k + 1] if k + 1 < nzo else torch.ones_like(wet_c[k]))
            f_k = torch.where(is_deepest, trans_w[k], trans_w[k] - trans_w[k + 1])
            frac_sub[k] = torch.where(wet_c[k], f_k, torch.zeros_like(sw_pos))
        frac_out_0 = frac_sub[1:].sum(dim=0)  # Exact sum transferred out of Layer 0
        q_to_dt = (dt / (C.CP_SEAWATER * C.RHO_SEAWATER)) * sw_pos
        T[0] = torch.where(is_ocean, T[0] - (q_to_dt * frac_out_0) / h_c_safe[0], T[0])
        T[1:] = torch.where(wet_c[1:], T[1:] + (q_to_dt.unsqueeze(0) * frac_sub[1:]) / h_c_safe[1:], T[1:])

    # Diagnose exact surface kinematic fluxes & buoyancy flux B_0 [m^2 s^-3] for KPP
    inv_dt = 1.0 / max(dt, 1.0)
    wt_sfc = (T[0] - t0_prev) * h_c_safe[0] * inv_dt * is_ocn_f  # [K m s^-1] (<0 = surface cooling)
    ws_sfc = (S[0] - s0_prev) * h_c_safe[0] * inv_dt * is_ocn_f  # [psu m s^-1] (>0 = brine/evap salinification)
    rho0_prev = compute_seawater_density_anomaly(t0_prev, s0_prev, ogcm.z_c[0])
    rho0_post = compute_seawater_density_anomaly(T[0], S[0], ogcm.z_c[0])
    b0_sfc = -(C.GRAVITY / C.RHO_SEAWATER) * ((rho0_post - rho0_prev) * h_c_safe[0] * inv_dt) * is_ocn_f

    # 2. Nonlinear EOS density anomaly & hydrostatic baroclinic pressure anomaly
    rho_anom = compute_seawater_density_anomaly(T, S, ogcm.z_c.unsqueeze(1)) * wet_cf
    g_rho_h = C.GRAVITY * rho_anom * h_c
    p_cum = torch.cumsum(g_rho_h, dim=0)
    p_baroc = p_cum - 0.5 * g_rho_h  # [nzo, nc]

    # 3. Wind stress at ocean surface [N m^-2] & ocean friction velocity u_*o
    ice_atten = 1.0 - 0.5 * torch.clamp(f_seaice, 0.0, 1.0)
    tx_ocn = -ice_atten * tau_x_sfc
    ty_ocn = -ice_atten * tau_y_sfc
    u_star_o = torch.sqrt(torch.sqrt(tx_ocn * tx_ocn + ty_ocn * ty_ocn + 1.0e-12) / C.RHO_SEAWATER)
    tx_e = 0.5 * (tx_ocn[c0] + tx_ocn[c1])
    ty_e = 0.5 * (ty_ocn[c0] + ty_ocn[c1])
    tau_n_e = (tx_e * mesh.east_normal_edge + ty_e * mesh.north_normal_edge) * wet_ef[0]

    # 3b. Reconstruct 3D cell-centered ocean currents & diagnose KPP / shear / tidal mixing
    u = ogcm.u_edge * wet_ef
    u_cell, v_cell = reconstruct_cell_winds(mesh, u)
    kv_diag, av_diag, h_bl, f_nl_t, f_nl_s = _diagnose_kpp_and_vertical_mixing(
        T=T, S=S, rho_anom=rho_anom, u_cell=u_cell, v_cell=v_cell,
        u_star_o=u_star_o, b0_sfc=b0_sfc, wt_sfc=wt_sfc, ws_sfc=ws_sfc,
        ogcm=ogcm, mesh=mesh, is_ocean=is_ocean,
    )
    ogcm.h_bl = h_bl

    # 4. 3D Baroclinic Momentum Step (TRiSK Coriolis + Baroclinic PGF + Munk Viscosity + Bottom Drag)
    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    delta_munk = 1.6 * dc_min
    a_h = min(BETA_MUNK_OGCM * (delta_munk ** 3), 0.06 * (dc_min ** 2) / max(dt, 1.0))

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
        is_bot_e[k] = wet_e[k] if k == nzo - 1 else (wet_e[k] & (~wet_e[k + 1]))
    r_bot = torch.where(is_bot_e, C_DRAG_BOT * torch.abs(u) / h_e_safe, torch.zeros_like(u))

    du_bc = (cor_3d + pgf_bc + a_h * lap_u - (R_RAYLEIGH_OCN + r_bot) * u) * wet_ef
    du_bc[0] = du_bc[0] + (tau_n_e / (C.RHO_SEAWATER * h_e_safe[0])) * wet_ef[0]

    # Explicit baroclinic predictor + implicit KPP/shear vertical momentum viscosity
    u_star = torch.clamp((u + dt * du_bc) * wet_ef, -3.0, 3.0)
    dz_int_e = 0.5 * (h_e_safe[:-1] + h_e_safe[1:])
    av_e = 0.5 * (av_diag[:, c0] + av_diag[:, c1])
    c_mom = torch.where(wet_int_e, dt * av_e / dz_int_e, torch.zeros_like(dz_int_e))
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
    f_baro = (u_star_baro - ogcm.u_baro) / max(dt, 1.0)

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

    u_new = (u_star + (u_bt_mean - u_star_baro).unsqueeze(0)) * wet_ef
    u_new = torch.clamp(u_new, -3.0, 3.0)
    ogcm.u_edge = u_new

    # 6. Gent-McWilliams (1990) Eddy Bolus & Neutral Isopycnal Slope (Boundary-Layer Tapered)
    drho_dx = ((rho_anom[:, c1] - rho_anom[:, c0]) / mesh.dc_edge.unsqueeze(0)) * wet_ef
    drho_dz_c = torch.where(
        wet_int_c,
        (rho_anom[1:] - rho_anom[:-1]) / dz_c_int,
        torch.zeros_like(dz_c_int),
    )
    drho_dz_e = 0.5 * (drho_dz_c[:, c0] + drho_dz_c[:, c1])
    drho_dx_int = 0.5 * (drho_dx[:-1] + drho_dx[1:])
    slope_raw = torch.where(
        wet_int_e,
        torch.clamp(-drho_dx_int / torch.clamp(drho_dz_e, min=1.0e-5), -GM_SLOPE_MAX, GM_SLOPE_MAX),
        torch.zeros_like(drho_dx_int),
    )
    # Ferrari et al. (2008) / Danabasoglu & McWilliams (1995) linear taper inside KPP boundary layer
    h_bl_e = torch.clamp(0.5 * (h_bl[c0] + h_bl[c1]), min=1.0).unsqueeze(0)
    taper_bl = torch.clamp(ogcm.z_w[1:-1].unsqueeze(1) / h_bl_e, 0.0, 1.0)
    sx_int = slope_raw * taper_bl  # [nzo-1, ne] neutral slope S_x = dz_iso/dx (z downward)

    psi_gm = torch.zeros((nzo + 1, ne), device=dev, dtype=dtype)
    psi_gm[1:-1] = KAPPA_GM * sx_int  # Psi = 0 at surface and seafloor -> exact zero depth-integral
    u_gm = torch.where(wet_e, -(psi_gm[1:] - psi_gm[:-1]) / h_e_safe, torch.zeros_like(u_new))
    u_gm = torch.clamp(u_gm, -0.3, 0.3)

    u_adv = (u_new + u_gm) * wet_ef

    # 7. Exact Mass-Conserving Vertical Velocity `w` [nzo + 1, nc] (positive upward; w_bot = 0)
    h_u_edge = h_e * u_adv
    div_hu = compute_scvt_divergence(mesh, h_u_edge) * wet_cf
    w_vel = torch.zeros((nzo + 1, nc), device=dev, dtype=dtype)
    for k in range(nzo - 1, -1, -1):
        w_vel[k] = (w_vel[k + 1] - div_hu[k]) * wet_cf[k]
    w_sfc_raw = (w_vel[1] - div_hu[0]) * wet_cf[0]
    eta_new = torch.clamp((eta_old + dt * w_sfc_raw) * is_ocn_f, -5.0, 5.0)
    ogcm.eta = eta_new
    w_vel[0] = 0.0
    ogcm.w_vel = w_vel

    h0_old = torch.clamp(h_c_safe[0] + eta_old, min=10.0)
    h0_new = torch.clamp(h_c_safe[0] + eta_new, min=10.0)
    dh0_eff = h0_new - h0_old

    # Precompute Voronoi cell geometric weights for exact 2D Redi tensor projection
    eoc = mesh.edges_on_cell  # [nc, 6]
    w_vor_edge = 0.5 * mesh.dc_edge * mesh.dv_edge  # [ne]
    active_eoc = (mesh.edge_sign_on_cell != 0.0).to(dtype)
    inv_area_c = 1.0 / mesh.area_cell  # [nc]

    if ENABLE_FULL_REDI:
        # Diagonal vertical Redi diffusivity K_v,Redi = kappa_Redi * |S|^2 (folded into implicit tridiagonal!)
        s2_edge_w = (sx_int * sx_int) * w_vor_edge.unsqueeze(0)  # [nzo-1, ne]
        s2_cell = (s2_edge_w[:, eoc] * active_eoc.unsqueeze(0)).sum(dim=-1) * inv_area_c.unsqueeze(0)
        kv_redi = torch.where(wet_int_c, KAPPA_REDI_HORIZ * s2_cell, torch.zeros_like(s2_cell))
    else:
        kv_redi = torch.zeros_like(kv_diag)

    # 8. 3D Conservative Flux-Form Tracer Advection + Full Rotated Redi Diffusion + KPP Non-Local Flux
    def _advect_diffuse_3d(
        q: torch.Tensor,
        q_comp: torch.Tensor,
        f_nl_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dq_dx = ((q[:, c1] - q[:, c0]) / mesh.dc_edge.unsqueeze(0)) * wet_ef  # [nzo, ne]
        f_adv = torch.clamp(h_u_edge, min=0.0) * q[:, c0] + torch.clamp(h_u_edge, max=0.0) * q[:, c1]

        if ENABLE_FULL_REDI:
            # Griffies et al. (1998) symmetric triad Redi cross-terms (exact zero flux when q = rho!)
            dq_dz_c = torch.where(wet_int_c, (q[1:] - q[:-1]) / dz_c_int, torch.zeros_like(dz_c_int))
            dq_dz_e = 0.5 * (dq_dz_c[:, c0] + dq_dz_c[:, c1]) * wet_int_e.to(dtype)
            sx_dqdz_int = sx_int * dq_dz_e  # [nzo-1, ne]
            # Average adjacent upper/lower interface cross-terms onto layer k
            sx_dqdz_lay = torch.zeros_like(dq_dx)
            sx_dqdz_lay[:-1] = sx_dqdz_lay[:-1] + 0.5 * sx_dqdz_int
            sx_dqdz_lay[1:] = sx_dqdz_lay[1:] + 0.5 * sx_dqdz_int
            f_dif = -KAPPA_REDI_HORIZ * h_e * (dq_dx + sx_dqdz_lay) * wet_ef

            # Upward vertical Redi cross-flux: +kappa_Redi * (S . grad_h q) at interior interfaces
            dq_dx_int = 0.5 * (dq_dx[:-1] + dq_dx[1:]) * wet_int_e.to(dtype)
            s_dqdx_ew = (sx_int * dq_dx_int) * w_vor_edge.unsqueeze(0)  # [nzo-1, ne]
            s_dot_gradq = (s_dqdx_ew[:, eoc] * active_eoc.unsqueeze(0)).sum(dim=-1) * inv_area_c.unsqueeze(0)
            f_v_redi_cross = torch.where(wet_int_c, KAPPA_REDI_HORIZ * s_dot_gradq, torch.zeros_like(s_dot_gradq))
        else:
            f_dif = -KAPPA_REDI_HORIZ * h_e * dq_dx * wet_ef
            f_v_redi_cross = torch.zeros_like(f_nl_q)

        div_h = compute_scvt_divergence(mesh, f_adv + f_dif) * wet_cf

        # Vertical upwind advection + Redi cross-flux + KPP non-local convective flux (all interior -> exact conservation!)
        w_int = w_vel[1:-1]
        f_v_adv = torch.where(w_int >= 0.0, w_int * q[1:], w_int * q[:-1]) * wet_int_c.to(dtype)
        # Soft limiter on non-local flux to avoid single-step overshoot while preserving exact flux form
        h_min_int = torch.minimum(h_c_safe[:-1], h_c_safe[1:])
        q_span = torch.abs(q[1:] - q[:-1]) + 0.5
        f_nl_cap = 0.35 * h_min_int * q_span / max(dt, 1.0)
        f_nl_lim = torch.clamp(f_nl_q, -f_nl_cap, f_nl_cap) * wet_int_c.to(dtype)

        f_v_int = f_v_adv + f_v_redi_cross + f_nl_lim
        v_inflow = torch.zeros_like(q)
        v_inflow[:-1] = v_inflow[:-1] + f_v_int
        v_inflow[1:] = v_inflow[1:] - f_v_int

        dq_adv = (dt / h_c_safe) * (-div_h + v_inflow)
        dq_adv[0] = (dt * (-div_h[0] + v_inflow[0]) - dh0_eff * q[0]) / h0_new
        dq_adv = dq_adv * wet_cf
        return _kahan_add(q, dq_adv, q_comp)

    T, ogcm.t_comp = _advect_diffuse_3d(T, ogcm.t_comp, f_nl_t)
    S, ogcm.s_comp = _advect_diffuse_3d(S, ogcm.s_comp, f_nl_s)

    # 9. Implicit Vertical Mixing (KPP + Shear + Tidal + Background + Implicit Redi S^2 + Post-Advection Convective Adj.)
    rho_post = compute_seawater_density_anomaly(T, S, ogcm.z_c.unsqueeze(1)) * wet_cf
    drho_dz_post = torch.where(
        wet_int_c,
        (rho_post[1:] - rho_post[:-1]) / dz_c_int,
        torch.ones_like(dz_c_int),
    )
    kv = kv_diag + kv_redi
    kv = torch.where(drho_dz_post < 0.0, torch.maximum(kv, torch.full_like(kv, KAPPA_V_CONV)), kv)
    kv = torch.where(wet_int_c, kv, torch.zeros_like(kv))
    ogcm.kv_last = kv

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
