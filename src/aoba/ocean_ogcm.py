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

AOBA Formulation Reference: Sec. 9.4, Eqs. (9.3)-(9.7) (3D Multi-Layer SCVT OGCM, FCT-z*, GM90/Redi, KPP & Tidal Mixing).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import (
    dispatch_ogcm_baroclinic_f_fixed_metal,
    dispatch_ogcm_baroclinic_rk3_update_metal,
    dispatch_ogcm_barotropic_rk3_metal,
    dispatch_ogcm_fct_zstar_metal,
    dispatch_ogcm_gm_bolus_edge_metal,
    dispatch_ogcm_kpp_vdiff_metal,
    dispatch_ogcm_mom_vdiff_metal,
    dispatch_ogcm_pre_kpp_metal,
    dispatch_ogcm_tracer_vdiff_freeze_metal,
    dispatch_ogcm_trisk_coriolis_wet_metal,
    dispatch_ogcm_vec_lap_wet_metal,
    dispatch_scvt_trisk_rot_metal,
    dispatch_scvt_weighted_edge_sum_metal,
    get_active_gpu_backend,
    solve_tridiagonal_batched,
)
from aoba.runtime import can_use_cuda_graph, maybe_cuda_compile, run_cuda_graph_cached
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
    bgc: object | None = None                 # Optional FlexPFT OceanBGCState (when enable_ocean_bgc=True)
    seaice_evp: object | None = None          # SeaIceEVPState (when enable_seaice_evp_itd=True)


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
    t_abyss_k: float = 274.5,
) -> OceanOGCMState:
    """Initialize 3D SCVT OGCM from `Earth2014 BED2014` bathymetry and analytic thermocline/halocline."""
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
            dz_f = float(dz_k.item())
            rem = torch.clamp(h_target - top_k, min=0.0, max=dz_f)
            # Enforce minimum partial step (35% of nominal dz, capped at 20 m) to prevent
            # vanishing bottom slivers while keeping thin upper-ocean levels (< 20 m) wet.
            min_partial = min(20.0, 0.35 * dz_f)
            hk = torch.where(rem >= min_partial, rem, torch.zeros_like(rem))
        h_c[k] = hk

    # Enforce strict monotonic vertical wetting (a wet layer can never sit beneath a dry layer)
    for k in range(1, nzo):
        h_c[k] = torch.where(h_c[k - 1] > 0.0, h_c[k], torch.zeros_like(h_c[k]))

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

    # Smooth map-free initial thermocline & polar halocline:
    # T(z, lat) decays from SST toward WOA23-scale abyss T_abyss = 274.5 K (+1.35 C) with
    # realistic pycnocline e-folding depth (460 m tropics -> 200 m poles, matching WOA23 global
    # mean T ~ 3.5 C and avoiding excessive initial baroclinic AMOC overshoot), while a localized
    # polar halocline (-0.22 psu, sin^8(lat) poleward of ~65 deg, globally normalized to exact
    # S_REF_PSU = 34.7 psu) stabilizes polar shelves without reversing the Southern Ocean ACC density front.
    lat_c = mesh.lat_cell.to(device=dev, dtype=dtype)
    cos2_lat = torch.cos(lat_c) ** 2
    sin2_lat = torch.sin(lat_c) ** 2
    sin8_lat = (sin2_lat * sin2_lat) * (sin2_lat * sin2_lat)
    d_therm = 200.0 + 260.0 * cos2_lat  # [nc] realistic WOA23-scale pycnocline depth
    sst = sfc.t_ocean_ml.clone()
    z_rel = torch.clamp(z_c.unsqueeze(1) - z_c[0], min=0.0)
    decay = torch.exp(-z_rel / d_therm.unsqueeze(0))
    t_init = t_abyss_k + (sst.unsqueeze(0) - t_abyss_k) * decay
    t_init[0] = sst
    t_init = torch.where(wet_c, torch.clamp(t_init, min=C.T_FREEZE_OCEAN), torch.full_like(t_init, t_abyss_k))

    sal_raw = S_REF_PSU - 0.22 * sin8_lat.unsqueeze(0) * torch.exp(-z_rel / 200.0)
    vol_init = h_c * mesh.area_cell.to(device=dev, dtype=dtype).unsqueeze(0)
    v_sum = torch.clamp(vol_init.sum(), min=1.0)
    sal_shift = S_REF_PSU - (vol_init * sal_raw).sum() / v_sum
    sal_init = torch.where(wet_c, sal_raw + sal_shift, torch.full_like(sal_raw, S_REF_PSU))
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


def _ogcm_seaice_drift_core(
    m_seaice: torch.Tensor,
    swe_seaice: torch.Tensor,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    u_ocn_e: torch.Tensor,
    is_ocean: torch.Tensor,
    is_ocn_f: torch.Tensor,
    mask_e: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    f_cell: torch.Tensor,
    east_normal_edge: torch.Tensor,
    north_normal_edge: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    u_cap: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-tensor conservative Voronoi sea-ice free-drift + plastic ridging step (CUDA-compilable)."""
    tx_i = -tau_x_sfc
    ty_i = -tau_y_sfc
    tau_mag = torch.sqrt(tx_i * tx_i + ty_i * ty_i + 1.0e-12)
    u_scale = torch.sqrt(tau_mag / 5.5)
    ux_w = (tx_i / tau_mag) * u_scale
    uy_w = (ty_i / tau_mag) * u_scale

    theta_rad = math.radians(SEAICE_TURN_DEG)
    sin_t = math.sin(theta_rad) * torch.sign(f_cell)
    cos_t = math.cos(theta_rad)
    ux_drift = ux_w * cos_t + uy_w * sin_t
    uy_drift = -ux_w * sin_t + uy_w * cos_t

    ux_e = 0.5 * (ux_drift[c0] + ux_drift[c1])
    uy_e = 0.5 * (uy_drift[c0] + uy_drift[c1])
    u_wind_n = ux_e * east_normal_edge + uy_e * north_normal_edge
    u_ice_raw = (u_ocn_e + u_wind_n) * mask_e

    h_ice_c = m_seaice / C.RHO_SEAICE
    h_dst = torch.where(u_ice_raw >= 0.0, h_ice_c[c1], h_ice_c[c0])
    ridge_fac = 1.0 / (1.0 + (h_dst / SEAICE_H_RIDGE_M) ** 2)
    u_ice = torch.clamp(u_ice_raw * ridge_fac, -u_cap, u_cap) * mask_e

    m_i = torch.clamp(m_seaice, min=0.0)
    s_i = torch.clamp(swe_seaice, min=0.0)
    u_pos = torch.clamp(u_ice, min=0.0)
    u_neg = torch.clamp(u_ice, max=0.0)
    f_m_tot = u_pos * m_i[c0] + u_neg * m_i[c1] - SEAICE_DIFF_H * ((m_i[c1] - m_i[c0]) * inv_dc_edge) * mask_e
    f_s_tot = u_pos * s_i[c0] + u_neg * s_i[c1] - SEAICE_DIFF_H * ((s_i[c1] - s_i[c0]) * inv_dc_edge) * mask_e

    dm = -dt * (f_m_tot[eoc] * div_w).sum(dim=-1) * is_ocn_f
    ds = -dt * (f_s_tot[eoc] * div_w).sum(dim=-1) * is_ocn_f

    m_new = torch.where(is_ocean, torch.clamp(m_i + dm, min=0.0), m_seaice)
    s_new = torch.where(is_ocean, torch.clamp(s_i + ds, min=0.0), swe_seaice)
    return m_new, s_new


def step_seaice_drift_voronoi(
    sfc: SurfaceState,
    mesh: SCVTMesh,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    u_ocn_e: torch.Tensor,
    dt: float,
    cache: dict | None = None,
) -> None:
    """Conservatively advect `sfc.m_seaice` & `sfc.swe_seaice` across wet ocean Voronoi edges.

    Uses wind + Layer-0 ocean free-drift with Coriolis turning (Hibler, 1979; Leppäranta, 2011)
    and plastic ridging resistance above `SEAICE_H_RIDGE_M` to prevent coastal pile-up.
    """
    is_ocean = sfc.surf_type == SURF_OCEAN
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    dtype = sfc.m_seaice.dtype
    if cache is not None:
        is_ocn_f = cache["is_ocn_f"]
        mask_e = cache["ocn_e0f"]
        div_w = cache["div_w"]
        inv_dc_edge = cache["inv_dc_edge"]
    else:
        is_ocn_f = is_ocean.to(dtype)
        mask_e = (is_ocean[c0] & is_ocean[c1]).to(dtype)
        div_w = (mesh.edge_sign_on_cell.to(dtype) * mesh.dv_edge[mesh.edges_on_cell].to(dtype)) / mesh.area_cell.unsqueeze(-1)
        inv_dc_edge = 1.0 / mesh.dc_edge

    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    u_cap = min(SEAICE_U_MAX, 0.25 * dc_min / max(dt, 1.0))

    drift_fn = maybe_cuda_compile(_ogcm_seaice_drift_core, device=sfc.m_seaice.device)
    sfc.m_seaice, sfc.swe_seaice = drift_fn(
        sfc.m_seaice, sfc.swe_seaice, tau_x_sfc, tau_y_sfc, u_ocn_e,
        is_ocean, is_ocn_f, mask_e, c0, c1, mesh.edges_on_cell, div_w,
        mesh.f_cell, mesh.east_normal_edge, mesh.north_normal_edge, inv_dc_edge,
        u_cap, dt,
    )


def _get_ogcm_static_cache(ogcm: OceanOGCMState, mesh: SCVTMesh, is_ocean: torch.Tensor) -> dict:
    """Lazily precompute static bathymetry, mask, tidal-base, and Jerlov-SW arrays on device."""
    cache = getattr(ogcm, "_static_cache", None)
    dev = ogcm.temp.device
    dtype = ogcm.temp.dtype
    if (
        cache is not None
        and cache.get("device") == dev
        and cache.get("dtype") == dtype
        and cache.get("bathym_slope_id") == id(ogcm.bathym_slope)
    ):
        return cache

    nzo = ogcm.nz_ocn
    nc = mesh.n_cells
    wet_c = ogcm.wet_c
    wet_e = ogcm.wet_e
    wet_cf = wet_c.to(dtype)
    wet_ef = wet_e.to(dtype)
    h_c_safe = torch.clamp(ogcm.h_c, min=1.0)
    h_e_safe = torch.clamp(ogcm.h_e, min=1.0)
    wet_int_c = wet_c[:-1] & wet_c[1:]
    wet_int_e = wet_e[:-1] & wet_e[1:]
    wet_int_cf = wet_int_c.to(dtype)
    wet_int_ef = wet_int_e.to(dtype)
    dz_c_int = 0.5 * (h_c_safe[:-1] + h_c_safe[1:])
    dz_int_e = 0.5 * (h_e_safe[:-1] + h_e_safe[1:])
    h_col_safe = torch.clamp(ogcm.h_col, min=1.0)
    h_col_e_safe = torch.clamp(ogcm.h_col_e, min=1.0)
    ocn_e0f = wet_e[0].to(dtype)
    is_ocn_f = is_ocean.to(dtype)

    # Deepest wet edge layer mask (vectorized, no per-step loop)
    is_bot_e = torch.zeros_like(wet_e)
    if nzo > 1:
        is_bot_e[:-1] = wet_e[:-1] & (~wet_e[1:])
    is_bot_e[-1] = wet_e[-1]

    # Static Jerlov Type I subsurface absorption fractions
    frac_sub = torch.zeros((nzo, nc), device=dev, dtype=dtype)
    if nzo >= 2:
        trans_w = compute_shortwave_transmission(ogcm.z_w)
        is_deepest_c = torch.zeros_like(wet_c)
        is_deepest_c[:-1] = wet_c[:-1] & (~wet_c[1:])
        is_deepest_c[-1] = wet_c[-1]
        d_trans = trans_w[:-1] - trans_w[1:]
        f_raw = torch.where(is_deepest_c, trans_w[:-1].unsqueeze(1), d_trans.unsqueeze(1))
        frac_sub[1:] = torch.where(wet_c[1:], f_raw[1:], torch.zeros_like(f_raw[1:]))
    frac_out_0 = frac_sub[1:].sum(dim=0)

    # Static Bryan-Lewis background diffusivity & Simmons (2004) tidal energy numerator
    z_int_m = ogcm.z_w[1:-1].unsqueeze(1)
    kv_bg = KAPPA_V_BG + 0.5 * (KAPPA_V_ABYSS - KAPPA_V_BG) * (1.0 + torch.tanh((z_int_m - 2000.0) / 800.0))

    slope_b = ogcm.bathym_slope
    if slope_b is None:
        slope_b = _compute_bathymetric_slope(ogcm.h_col, wet_c[0], mesh)
        ogcm.bathym_slope = slope_b
    # Combine resolved bathymetric slope with subgrid roughness sigma_topo (St. Laurent et al., 2002)
    rough_fac = 1.0 + 0.5 * torch.clamp(mesh.sigma_topo.to(dtype) / 250.0, 0.0, 4.0)
    e_tide = TIDAL_E0_WM2 * (torch.clamp(slope_b / TIDAL_SLOPE_REF, 0.0, 4.0) ** 2) * rough_fac
    hab = torch.clamp(ogcm.h_col.unsqueeze(0) - z_int_m, min=0.0)
    norm_f = TIDAL_ZETA_M * torch.clamp(1.0 - torch.exp(-ogcm.h_col.unsqueeze(0) / TIDAL_ZETA_M), min=1.0e-2)
    f_z_tide = torch.exp(-hab / TIDAL_ZETA_M) / norm_f
    kv_tide_num = (0.0667 * e_tide.unsqueeze(0) * f_z_tide) / C.RHO_SEAWATER

    # Hallberg (2013, Ocean Modelling, 72, 92-103) resolution-aware GM scale factor on edges:
    # R_GM = 1 / (1 + (L_d / dx)^2) using 1st-baroclinic gravity wave speed c1 ~ 2.2 m s^-1
    f_edge = 0.5 * (mesh.f_cell[mesh.cells_on_edge[:, 0]] + mesh.f_cell[mesh.cells_on_edge[:, 1]]).to(dtype)
    beta_e = (2.0 * C.OMEGA_EARTH / C.EARTH_RADIUS)
    c1_bc = 2.2
    f_eff_ld = torch.maximum(torch.abs(f_edge), torch.full_like(f_edge, math.sqrt(2.0 * beta_e * c1_bc)))
    ld_edge = c1_bc / torch.clamp(f_eff_ld, min=1.0e-6)
    gm_scale_e = 1.0 / (1.0 + (ld_edge / torch.clamp(mesh.dc_edge.to(dtype), min=1.0e3)) ** 2)
    # Fox-Kemper et al. (2008, 2011) submesoscale scaling factor on edges: C_e * min(dx/L_f, 8.0) / sqrt(f^2 + tau^-2)
    tau_sub_inv = 1.0 / (5.0 * C.SECONDS_PER_DAY)
    f_sub_e = torch.sqrt(f_edge * f_edge + tau_sub_inv * tau_sub_inv)
    fk_coef_e = 0.06 * torch.clamp(mesh.dc_edge.to(dtype) / 500.0, min=1.0, max=8.0) / f_sub_e

    # Static Voronoi Redi geometric weights & scalar h_max (eliminates per-step CPU-GPU sync!)
    w_vor_edge = 0.5 * mesh.dc_edge * mesh.dv_edge
    active_eoc = (mesh.edge_sign_on_cell != 0.0).to(dtype)
    inv_area_c = 1.0 / mesh.area_cell
    active_w_c = (active_eoc * inv_area_c.unsqueeze(-1)).contiguous()
    div_w = (mesh.edge_sign_on_cell.to(dtype) * mesh.dv_edge[mesh.edges_on_cell].to(dtype)) * inv_area_c.unsqueeze(-1)
    vort_w = (
        mesh.edge_sign_on_vert.to(dtype) * mesh.dc_edge[mesh.edges_on_vert].to(dtype)
    ) / mesh.area_vert.to(dtype).unsqueeze(-1)
    h_max = float(ogcm.h_col.max().item()) if bool(is_ocean.any()) else 100.0
    f_abs = torch.clamp(torch.abs(mesh.f_cell.to(dtype)), min=1.0e-5)
    h_col_kpp = torch.clamp(ogcm.h_col, min=50.0, max=500.0)

    cache = {
        "device": dev,
        "dtype": dtype,
        "bathym_slope_id": id(ogcm.bathym_slope),
        "wet_cf": wet_cf,
        "wet_ef": wet_ef,
        "h_c_safe": h_c_safe,
        "h_e_safe": h_e_safe,
        "wet_int_c": wet_int_c,
        "wet_int_e": wet_int_e,
        "wet_int_cf": wet_int_cf,
        "wet_int_ef": wet_int_ef,
        "dz_c_int": dz_c_int,
        "dz_int_e": dz_int_e,
        "h_col_safe": h_col_safe,
        "h_col_e_safe": h_col_e_safe,
        "ocn_e0f": ocn_e0f,
        "is_ocn_f": is_ocn_f,
        "is_bot_e": is_bot_e,
        "frac_sub": frac_sub,
        "frac_out_0": frac_out_0,
        "kv_bg": kv_bg,
        "kv_tide_num": kv_tide_num,
        "gm_scale_e": gm_scale_e,
        "fk_coef_e": fk_coef_e,
        "w_vor_edge": w_vor_edge,
        "active_eoc": active_eoc,
        "inv_area_c": inv_area_c,
        "active_w_c": active_w_c,
        "div_w": div_w,
        "vort_w": vort_w,
        "h_max": h_max,
        "f_abs": f_abs,
        "h_col_kpp": h_col_kpp,
        "inv_dc_edge": 1.0 / mesh.dc_edge,
    }
    setattr(ogcm, "_static_cache", cache)
    return cache


def _ogcm_kpp_mixing_core(
    T: torch.Tensor,
    rho_anom: torch.Tensor,
    u_cell: torch.Tensor,
    v_cell: torch.Tensor,
    u_star_o: torch.Tensor,
    b0_sfc: torch.Tensor,
    wt_sfc: torch.Tensor,
    ws_sfc: torch.Tensor,
    wet_c: torch.Tensor,
    wet_int_c: torch.Tensor,
    wet_int_cf: torch.Tensor,
    h_c: torch.Tensor,
    dz_c_int: torch.Tensor,
    z_c_2d: torch.Tensor,
    z_int_m: torch.Tensor,
    is_ocn_f: torch.Tensor,
    h_col_kpp: torch.Tensor,
    h_col_safe: torch.Tensor,
    f_abs: torch.Tensor,
    kv_bg: torch.Tensor,
    kv_tide_num: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor KPP boundary layer, shear, tidal, and convective diffusivity diagnosis (CUDA-compilable)."""
    dtype = T.dtype
    drho_dz = ((rho_anom[1:] - rho_anom[:-1]) / dz_c_int) * wet_int_cf
    n2_int = (C.GRAVITY / C.RHO_SEAWATER) * torch.clamp(drho_dz, min=0.0)
    du_dz = (u_cell[1:] - u_cell[:-1]) / dz_c_int
    dv_dz = (v_cell[1:] - v_cell[:-1]) / dz_c_int
    s2_int = (du_dz * du_dz + dv_dz * dv_dz) * wet_int_cf

    db_k = (C.GRAVITY / C.RHO_SEAWATER) * (rho_anom - rho_anom[0:1])
    dv2_k = (u_cell - u_cell[0:1]) ** 2 + (v_cell - v_cell[0:1]) ** 2
    n2_cell = torch.cat([n2_int[0:1], 0.5 * (n2_int[:-1] + n2_int[1:]), n2_int[-1:]], dim=0)
    w_star_est = torch.pow(torch.clamp(-b0_sfc, min=0.0) * h_col_kpp, 1.0 / 3.0)
    w_s0 = C.VON_KARMAN * torch.pow(u_star_o ** 3 + 2.0 * (w_star_est ** 3) + 1.0e-12, 1.0 / 3.0)
    vt2_k = 0.08 * torch.sqrt(torch.clamp(n2_cell, min=0.0)) * z_c_2d * w_s0.unsqueeze(0)
    ri_b = (db_k * z_c_2d) / (dv2_k + vt2_k + 1.0e-6)

    in_bl = wet_c & ((ri_b < KPP_RI_CRIT) | (db_k <= 0.0))
    in_bl_contig = torch.cumprod(in_bl.to(torch.int32), dim=0)
    in_bl_contig = torch.cat([wet_c[0:1].to(torch.int32), in_bl_contig[1:]], dim=0)
    h_bl_ri = (h_c * in_bl_contig.to(dtype)).sum(dim=0)

    h_ek = 0.7 * u_star_o / f_abs
    h_mo = (u_star_o ** 3) / (C.VON_KARMAN * torch.clamp(b0_sfc, min=1.0e-12))
    h_stab_cap = torch.minimum(h_ek, h_mo)
    h_bl = torch.where(b0_sfc > 1.0e-11, torch.minimum(h_bl_ri, torch.maximum(h_stab_cap, h_c[0])), h_bl_ri)
    h_bl = torch.clamp(h_bl, min=h_c[0], max=h_col_safe) * is_ocn_f

    sigma = z_int_m / torch.clamp(h_bl.unsqueeze(0), min=1.0)
    inside_bl_int = wet_int_c & (sigma > 0.0) & (sigma < 1.0)
    sig_c = torch.clamp(sigma, 0.0, 1.0)
    g_shape = sig_c * ((1.0 - sig_c) ** 2)

    conv_scale = 16.0 * C.VON_KARMAN * sig_c * torch.clamp(-b0_sfc, min=0.0).unsqueeze(0) * h_bl.unsqueeze(0)
    w_s_int = C.VON_KARMAN * torch.pow(u_star_o.unsqueeze(0) ** 3 + conv_scale + 1.0e-12, 1.0 / 3.0)
    kv_kpp = torch.clamp(h_bl.unsqueeze(0) * w_s_int * g_shape, min=0.0, max=KPP_KV_MAX) * inside_bl_int.to(dtype)

    is_conv_sfc = ((b0_sfc < -1.0e-11).unsqueeze(0) & inside_bl_int).to(dtype)
    nl_weight = (KPP_CS_NONLOCAL * g_shape) * is_conv_sfc
    f_nl_t = -nl_weight * wt_sfc.unsqueeze(0)
    f_nl_s = -nl_weight * ws_sfc.unsqueeze(0)

    ri_g = n2_int / (s2_int + 1.0e-10)
    ri_ratio = torch.clamp(ri_g / SHEAR_RI_0, 0.0, 1.0)
    one_m_r2 = 1.0 - ri_ratio * ri_ratio
    kv_shear = (SHEAR_NU_0 * (one_m_r2 * one_m_r2 * one_m_r2)) * wet_int_cf

    kv_tide = torch.clamp(kv_tide_num / torch.clamp(n2_int, min=1.0e-7), min=0.0, max=TIDAL_KV_MAX) * wet_int_cf

    kv_total = kv_bg + kv_tide + kv_kpp + kv_shear
    kv_total = torch.where(drho_dz < 0.0, torch.clamp(kv_total, min=KAPPA_V_CONV), kv_total) * wet_int_cf
    av_total = (A_V_MOM + kv_kpp + kv_shear) * wet_int_cf
    return kv_total, av_total, h_bl, f_nl_t, f_nl_s


def _ogcm_pre_kpp_core(
    ogcm_temp: torch.Tensor,
    ogcm_sal: torch.Tensor,
    ogcm_t_comp: torch.Tensor,
    ogcm_s_comp: torch.Tensor,
    sfc_t_ocean_ml: torch.Tensor,
    sfc_m_seaice: torch.Tensor,
    ogcm_m_seaice_prev: torch.Tensor,
    r_coast: torch.Tensor,
    q_calve: torch.Tensor,
    evap_cell: torch.Tensor,
    precip_cell: torch.Tensor,
    sw_net_ocn: torch.Tensor,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    f_seaice: torch.Tensor,
    is_ocean: torch.Tensor,
    is_ocn_f: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_ef: torch.Tensor,
    h_c: torch.Tensor,
    h_c_safe: torch.Tensor,
    dz_nom_2d: torch.Tensor,
    z_c_2d: torch.Tensor,
    frac_out_0: torch.Tensor,
    frac_sub: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    east_normal_edge: torch.Tensor,
    north_normal_edge: torch.Tensor,
    dt: float,
    apply_sw: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pure-tensor OGCM surface buoyancy, calving, salinity, Jerlov SW, EOS & stress prep (CUDA-compilable)."""
    t0_prev = ogcm_temp[0]
    s0_prev = ogcm_sal[0]
    t0_in = torch.where(is_ocean, sfc_t_ocean_ml, t0_prev)

    dt_calve = (-(dt * q_calve) / (C.CP_SEAWATER * C.RHO_SEAWATER * h_c_safe[0])) * is_ocn_f
    t0_calv, tc0_calv = _kahan_add(t0_in, dt_calve, ogcm_t_comp[0])
    t0_cur = torch.where(is_ocean, t0_calv, t0_in)
    tc0_cur = torch.where(is_ocean, tc0_calv, ogcm_t_comp[0])

    dm_ice_thermo = sfc_m_seaice - ogcm_m_seaice_prev
    fw_evap_minus_p = evap_cell - precip_cell - r_coast
    ds_sfc = (
        (S_REF_PSU * (fw_evap_minus_p * dt) + (S_REF_PSU - S_SEAICE_PSU) * dm_ice_thermo)
        / (C.RHO_SEAWATER * h_c_safe[0])
    ) * is_ocn_f
    s0_next, sc0_next = _kahan_add(s0_prev, ds_sfc, ogcm_s_comp[0])
    s0_cur = torch.where(is_ocean, torch.clamp(s0_next, 5.0, 45.0), s0_prev)
    sc0_cur = torch.where(is_ocean, sc0_next, ogcm_s_comp[0])

    T = torch.cat([t0_cur.unsqueeze(0), ogcm_temp[1:]], dim=0)
    S = torch.cat([s0_cur.unsqueeze(0), ogcm_sal[1:]], dim=0)
    t_comp = torch.cat([tc0_cur.unsqueeze(0), ogcm_t_comp[1:]], dim=0)
    s_comp = torch.cat([sc0_cur.unsqueeze(0), ogcm_s_comp[1:]], dim=0)

    if apply_sw:
        sw_pos = torch.clamp(sw_net_ocn, min=0.0) * is_ocn_f
        q_to_dt = (dt / (C.CP_SEAWATER * C.RHO_SEAWATER)) * sw_pos
        dt_sw_0 = -(q_to_dt * frac_out_0) / h_c_safe[0]
        dt_sw_sub = (q_to_dt.unsqueeze(0) * frac_sub[1:]) / h_c_safe[1:]
        dt_sw = torch.cat([dt_sw_0.unsqueeze(0), dt_sw_sub], dim=0) * wet_cf
        T, t_comp = _kahan_add(T, dt_sw, t_comp)

    inv_dt = 1.0 / max(dt, 1.0)
    wt_sfc = (T[0] - t0_prev) * h_c_safe[0] * inv_dt * is_ocn_f
    ws_sfc = (S[0] - s0_prev) * h_c_safe[0] * inv_dt * is_ocn_f
    rho0_prev = compute_seawater_density_anomaly(t0_prev, s0_prev, z_c_2d[0, 0])
    rho0_post = compute_seawater_density_anomaly(T[0], S[0], z_c_2d[0, 0])
    b0_sfc = -(C.GRAVITY / C.RHO_SEAWATER) * ((rho0_post - rho0_prev) * h_c_safe[0] * inv_dt) * is_ocn_f

    # Evaluate baroclinic hydrostatic pressure at fixed nominal z-level centers `z_c[k]`
    # (Pacanowski & Gnanadesikan, 1998): guarantees exact zero PGF in a resting stratified ocean
    # over partial-step topography (avoids spurious topographic pressure gradients where h_c differs).
    rho_anom = compute_seawater_density_anomaly(T, S, z_c_2d) * wet_cf
    g_rho_dz = C.GRAVITY * rho_anom * dz_nom_2d
    p_cum = torch.cumsum(g_rho_dz, dim=0)
    p_baroc = p_cum - 0.5 * g_rho_dz

    ice_atten = 1.0 - 0.5 * torch.clamp(f_seaice, 0.0, 1.0)
    tx_ocn = -ice_atten * tau_x_sfc
    ty_ocn = -ice_atten * tau_y_sfc
    u_star_o = torch.sqrt(torch.sqrt(tx_ocn * tx_ocn + ty_ocn * ty_ocn + 1.0e-12) / C.RHO_SEAWATER)
    tx_e = 0.5 * (tx_ocn[c0] + tx_ocn[c1])
    ty_e = 0.5 * (ty_ocn[c0] + ty_ocn[c1])
    tau_n_e = (tx_e * east_normal_edge + ty_e * north_normal_edge) * wet_ef[0]
    return T, S, t_comp, s_comp, wt_sfc, ws_sfc, b0_sfc, rho_anom, p_baroc, u_star_o, tau_n_e


def _ogcm_baroclinic_rk3_core(
    u: torch.Tensor,
    lap_u: torch.Tensor,
    p_baroc: torch.Tensor,
    av_diag: torch.Tensor,
    tau_n_e: torch.Tensor,
    f_vert: torch.Tensor,
    eov: torch.Tensor,
    vort_w: torch.Tensor,
    v0_e: torch.Tensor,
    v1_e: torch.Tensor,
    ep_idx: torch.Tensor,
    w_trisk: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    is_bot_e: torch.Tensor,
    h_e_safe: torch.Tensor,
    dz_int_e: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_ef: torch.Tensor,
    a_h: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-stage Wicker-Skamarock (2002) RK3 TRiSK Coriolis + baroclinic PGF + semi-implicit drag + tridiag.

    Three-stage WS-RK3 is strictly stable along the imaginary axis up to `|f dt| <= sqrt(3) ~ 1.73`,
    ensuring inertial-oscillation stability across both coupled (`dt = 720 s`) and offline (`dt = 3600 s`)
    timesteps at polar latitudes.
    """
    pgf_bc = -(1.0 / C.RHO_SEAWATER) * ((p_baroc[:, c1] - p_baroc[:, c0]) * inv_dc_edge.unsqueeze(0))
    f_fixed = (pgf_bc + a_h * lap_u) * wet_ef
    f_fixed_0 = f_fixed[0] + (tau_n_e / (C.RHO_SEAWATER * h_e_safe[0])) * wet_ef[0]
    f_fixed = torch.cat([f_fixed_0.unsqueeze(0), f_fixed[1:]], dim=0)

    u_s = u
    vort_w_u = vort_w.unsqueeze(0)
    f_vert_u = f_vert.unsqueeze(0)
    w_trisk_u = w_trisk.unsqueeze(0)
    for rk_dt in (dt / 3.0, 0.5 * dt, dt):
        zeta_v = (u_s[:, eov] * vort_w_u).sum(dim=-1)
        eta_v = zeta_v + f_vert_u
        eta_e = 0.5 * (eta_v[:, v0_e] + eta_v[:, v1_e])
        eta_pair = 0.5 * (eta_e.unsqueeze(-1) + eta_e[:, ep_idx])
        cor_3d = (w_trisk_u * eta_pair * u_s[:, ep_idx]).sum(dim=-1)
        r_drag = R_RAYLEIGH_OCN + torch.where(is_bot_e, C_DRAG_BOT * torch.abs(u_s) / h_e_safe, torch.zeros_like(u_s))
        u_s = torch.clamp(((u + rk_dt * (cor_3d + f_fixed)) / (1.0 + rk_dt * r_drag)) * wet_ef, -3.0, 3.0)
    u_star = u_s

    av_e = 0.5 * (av_diag[:, c0] + av_diag[:, c1])
    c_mom = (dt * av_e / dz_int_e) * wet_int_ef
    z_row = torch.zeros_like(u[0:1])
    a_m = torch.cat([z_row, -c_mom / h_e_safe[1:]], dim=0)
    d_m = torch.cat([-c_mom / h_e_safe[:-1], z_row], dim=0)
    b_m = 1.0 - a_m - d_m
    return u_star, a_m, b_m, d_m


def _ogcm_barotropic_rk3_core(
    u_star_raw: torch.Tensor,
    u_prev_baro: torch.Tensor,
    u_baro_prev: torch.Tensor,
    eta_old: torch.Tensor,
    g_inv_dc: torch.Tensor,
    h_e: torch.Tensor,
    h_col_e: torch.Tensor,
    h_col_e_safe: torch.Tensor,
    wet_ef: torch.Tensor,
    ocn_e0f: torch.Tensor,
    is_ocn_f: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    n_bt: int,
    dt_bt: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor barotropic mode projection + 2D Wicker-Skamarock RK3 loop (CUDA-compilable)."""
    u_star = u_star_raw * wet_ef
    u_star_baro = (u_star * h_e).sum(dim=0) / h_col_e_safe * ocn_e0f
    # Depth-averaged slow tendency plus implicit fast-fluctuation damping `-r_pert*(u_stage - u_prev_baro)`
    # with `r_pert = 1/dt` (eliminates split-explicit boxcar-averaging resonance at w*dt ~ 3.5 while
    # introducing identically zero drag on steady geostrophic currents; Shchepetkin & McWilliams, 2005).
    r_bt_pert = 1.0 / max(dt, 1.0)
    f_baro = (u_star_baro - u_prev_baro) / max(dt, 1.0) + r_bt_pert * u_prev_baro
    u_bt = u_baro_prev * ocn_e0f
    eta_bt = eta_old
    u_bt_acc = torch.zeros_like(u_bt)
    dt1 = dt_bt / 3.0
    dt2 = 0.5 * dt_bt
    dt3 = dt_bt
    for _ in range(n_bt):
        u0_bt = u_bt
        eta0_bt = eta_bt
        for rk_dt in (dt1, dt2, dt3):
            inv_damp = 1.0 / (1.0 + rk_dt * r_bt_pert)
            pgf_bt = g_inv_dc * (eta_bt[c1] - eta_bt[c0])
            u_stage = torch.clamp(((u0_bt + rk_dt * (pgf_bt + f_baro)) * inv_damp) * ocn_e0f, -2.5, 2.5)
            flux_bt = h_col_e * u_stage
            deta = -(flux_bt[eoc] * div_w).sum(dim=-1) * is_ocn_f
            eta_bt = torch.clamp((eta0_bt + rk_dt * deta) * is_ocn_f, -5.0, 5.0)
            u_bt = u_stage
        u_bt_acc = u_bt_acc + u_bt
    return u_star, u_star_baro, u_bt, u_bt_acc / float(n_bt)


def _ogcm_gm_and_w_core(
    u_star: torch.Tensor,
    u_bt_mean: torch.Tensor,
    u_star_baro: torch.Tensor,
    rho_anom: torch.Tensor,
    h_bl: torch.Tensor,
    eta_old: torch.Tensor,
    h_e: torch.Tensor,
    h_e_safe: torch.Tensor,
    h_c_safe: torch.Tensor,
    dz_c_int: torch.Tensor,
    z_int_m: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_cf: torch.Tensor,
    wet_int_ef: torch.Tensor,
    is_ocn_f: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    w_vor_edge: torch.Tensor,
    active_eoc: torch.Tensor,
    inv_area_c: torch.Tensor,
    dt: float,
    enable_full_redi: bool,
    gm_scale_e: torch.Tensor | None = None,
    fk_coef_e: torch.Tensor | None = None,
    enable_submeso: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pure-tensor GM90 + Hallberg scale-aware + Fox-Kemper submeso bolus + continuity w + eta."""
    u_new = torch.clamp((u_star + (u_bt_mean - u_star_baro).unsqueeze(0)) * wet_ef, -4.0, 4.0)

    drho_dx = ((rho_anom[:, c1] - rho_anom[:, c0]) * inv_dc_edge.unsqueeze(0)) * wet_ef
    drho_dz_c = ((rho_anom[1:] - rho_anom[:-1]) / dz_c_int) * wet_int_cf
    drho_dz_e = 0.5 * (drho_dz_c[:, c0] + drho_dz_c[:, c1])
    drho_dx_int = 0.5 * (drho_dx[:-1] + drho_dx[1:])
    slope_raw = torch.clamp(-drho_dx_int / torch.clamp(drho_dz_e, min=1.0e-5), -GM_SLOPE_MAX, GM_SLOPE_MAX) * wet_int_ef
    h_bl_e = torch.clamp(0.5 * (h_bl[c0] + h_bl[c1]), min=1.0).unsqueeze(0)
    taper_bl = torch.clamp(z_int_m / h_bl_e, 0.0, 1.0)
    sx_int = slope_raw * taper_bl

    kappa_gm_eff = (KAPPA_GM * gm_scale_e.unsqueeze(0)) if gm_scale_e is not None else KAPPA_GM
    psi_int = kappa_gm_eff * sx_int
    if enable_submeso and fk_coef_e is not None:
        # Fox-Kemper et al. (2008, JPO, Eq. 5) mixed-layer baroclinic eddy overturning streamfunction
        db_dx_ml = -(C.GRAVITY / C.RHO_SEAWATER) * drho_dx[0:1]
        z_norm = z_int_m / torch.clamp(h_bl_e, min=10.0)
        in_ml = (z_norm > 0.0) & (z_norm < 1.0)
        xi = 2.0 * torch.clamp(z_norm, 0.0, 1.0) - 1.0
        mu_fk = torch.where(in_ml, (1.0 - xi * xi) * (1.0 + (5.0 / 21.0) * xi * xi), torch.zeros_like(z_norm))
        psi_mle = torch.clamp(
            fk_coef_e.unsqueeze(0) * (h_bl_e * h_bl_e) * db_dx_ml * mu_fk,
            min=-2.5,
            max=2.5,
        ) * wet_int_ef
        psi_int = psi_int + psi_mle
    # Bound bolus streamfunction before vertical differencing so |u_gm| <= 0.30 m s^-1 AND
    # depth-integrated bolus transport sum_k(h_e * u_gm) == 0 identically (zero spurious eta pumping).
    psi_cap = 0.15 * torch.minimum(h_e_safe[:-1], h_e_safe[1:])
    psi_int = torch.clamp(psi_int, min=-psi_cap, max=psi_cap) * wet_int_ef
    z_e = torch.zeros_like(u_new[0:1])
    dpsi = torch.cat([psi_int, z_e], dim=0) - torch.cat([z_e, psi_int], dim=0)
    u_gm = (-dpsi / h_e_safe) * wet_ef

    h_u_edge = h_e * ((u_new + u_gm) * wet_ef)
    div_hu = (h_u_edge[:, eoc] * div_w.unsqueeze(0)).sum(dim=-1) * wet_cf
    w_up = -torch.flip(torch.cumsum(torch.flip(div_hu, dims=[0]), dim=0), dims=[0]) * wet_cf
    z_c = torch.zeros_like(div_hu[0:1])
    w_vel = torch.cat([z_c, w_up[1:], z_c], dim=0)
    eta_new = torch.clamp((eta_old + dt * w_up[0]) * is_ocn_f, -5.0, 5.0)

    h0_old = torch.clamp(h_c_safe[0] + eta_old, min=10.0)
    h0_new = torch.clamp(h_c_safe[0] + eta_new, min=10.0)
    dh0_eff = h0_new - h0_old

    if enable_full_redi:
        s2_edge_w = (sx_int * sx_int) * w_vor_edge.unsqueeze(0)
        s2_cell = (s2_edge_w[:, eoc] * active_eoc.unsqueeze(0)).sum(dim=-1) * inv_area_c.unsqueeze(0)
        kv_redi = (KAPPA_REDI_HORIZ * s2_cell) * wet_int_cf
    else:
        kv_redi = torch.zeros_like(wet_int_cf)

    return u_new, sx_int, h_u_edge, w_vel, eta_new, h0_new, dh0_eff, kv_redi


def _ogcm_vdiff_prep_core(
    T: torch.Tensor,
    S: torch.Tensor,
    kv_diag: torch.Tensor,
    kv_redi: torch.Tensor,
    z_c_2d: torch.Tensor,
    dz_c_int: torch.Tensor,
    h_c_safe: torch.Tensor,
    h0_new: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_int_c: torch.Tensor,
    wet_int_cf: torch.Tensor,
    dt: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pure-tensor post-advection stability check + conservative flux-increment tridiagonal assembly."""
    rho_post = compute_seawater_density_anomaly(T, S, z_c_2d) * wet_cf
    drho_dz_post = torch.where(
        wet_int_c,
        (rho_post[1:] - rho_post[:-1]) / dz_c_int,
        torch.ones_like(dz_c_int),
    )
    kv = kv_diag + kv_redi
    kv = torch.where(drho_dz_post < 0.0, torch.clamp(kv, min=KAPPA_V_CONV), kv) * wet_int_cf

    c_dif = (dt * kv / dz_c_int) * wet_int_cf
    h_eff = torch.cat([h0_new.unsqueeze(0), h_c_safe[1:]], dim=0)
    z_row = torch.zeros_like(T[0:1])
    a_t = torch.cat([z_row, -c_dif / h_eff[1:]], dim=0)
    d_t = torch.cat([-c_dif / h_eff[:-1], z_row], dim=0)
    b_t = 1.0 - a_t - d_t

    t_diff = T[1:] - T[:-1]
    f_exp_t = c_dif * t_diff
    rhs_t = (torch.cat([f_exp_t, z_row], dim=0) - torch.cat([z_row, f_exp_t], dim=0)) / h_eff

    s_diff = S[1:] - S[:-1]
    f_exp_s = c_dif * s_diff
    rhs_s = (torch.cat([f_exp_s, z_row], dim=0) - torch.cat([z_row, f_exp_s], dim=0)) / h_eff

    return kv, c_dif, h_eff, a_t, b_t, d_t, t_diff, s_diff, rhs_t, rhs_s


def _ogcm_vdiff_post_freeze_core(
    T: torch.Tensor,
    t_comp: torch.Tensor,
    S: torch.Tensor,
    s_comp: torch.Tensor,
    t_diff: torch.Tensor,
    s_diff: torch.Tensor,
    dt_imp: torch.Tensor,
    ds_imp: torch.Tensor,
    c_dif: torch.Tensor,
    h_eff: torch.Tensor,
    h_c_safe: torch.Tensor,
    wet_c: torch.Tensor,
    wet_cf: torch.Tensor,
    is_ocean: torch.Tensor,
    is_ocn_f: torch.Tensor,
    sfc_m_seaice: torch.Tensor,
    sfc_t_ocean_ml: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor telescoping implicit flux update + Kahan summation + conservative Layer-0 freezing."""
    z_row = torch.zeros_like(T[0:1])
    f_imp_t = c_dif * (t_diff + (dt_imp[1:] - dt_imp[:-1]))
    dt_cons = ((torch.cat([f_imp_t, z_row], dim=0) - torch.cat([z_row, f_imp_t], dim=0)) / h_eff) * wet_cf
    T, t_comp = _kahan_add(T, dt_cons, t_comp)

    f_imp_s = c_dif * (s_diff + (ds_imp[1:] - ds_imp[:-1]))
    ds_cons = ((torch.cat([f_imp_s, z_row], dim=0) - torch.cat([z_row, f_imp_s], dim=0)) / h_eff) * wet_cf
    S, s_comp = _kahan_add(S, ds_cons, s_comp)
    S = torch.where(wet_c, torch.clamp(S, 5.0, 45.0), S)

    freeze_exc = torch.where(is_ocean, torch.clamp(C.T_FREEZE_OCEAN - T[0], min=0.0), torch.zeros_like(T[0]))
    c0_cap = C.CP_SEAWATER * C.RHO_SEAWATER * h_c_safe[0]
    dm_freeze = (c0_cap * freeze_exc) / C.L_F0
    t0_frz = torch.where(is_ocean, T[0] + freeze_exc, T[0])
    ds_brine = ((S_REF_PSU - S_SEAICE_PSU) * dm_freeze) / (C.RHO_SEAWATER * h_c_safe[0])
    s0_br, sc0_br = _kahan_add(S[0], ds_brine * is_ocn_f, s_comp[0])
    s0_frz = torch.where(is_ocean, torch.clamp(s0_br, 5.0, 45.0), S[0])
    sc0_frz = torch.where(is_ocean, sc0_br, s_comp[0])

    T_out = torch.cat([t0_frz.unsqueeze(0), T[1:]], dim=0)
    T_out = torch.where(wet_c, torch.clamp(T_out, min=C.T_FREEZE_OCEAN), T_out)
    S_out = torch.cat([s0_frz.unsqueeze(0), S[1:]], dim=0)
    sc_out = torch.cat([sc0_frz.unsqueeze(0), s_comp[1:]], dim=0)

    t_ml_out = torch.where(is_ocean, T_out[0], sfc_t_ocean_ml)
    m_ice_out = torch.where(is_ocean, torch.clamp(sfc_m_seaice + dm_freeze, min=0.0), sfc_m_seaice)
    return T_out, t_comp, S_out, sc_out, t_ml_out, m_ice_out


def _ogcm_advect_fct_zstar_core(
    q: torch.Tensor,
    q_comp: torch.Tensor,
    f_nl_q: torch.Tensor,
    h_u_edge: torch.Tensor,
    sx_int: torch.Tensor,
    w_star: torch.Tensor,
    h_e: torch.Tensor,
    h_old_z: torch.Tensor,
    h_new_z: torch.Tensor,
    dz_c_int: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_cf: torch.Tensor,
    wet_int_ef: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    w_vor_edge: torch.Tensor,
    active_eoc: torch.Tensor,
    inv_area_c: torch.Tensor,
    q_ref: float,
    dt: float,
    enable_full_redi: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FESOM2-style 3D Zalesak FCT tracer advection + exact z* stretching + Redi + Kahan.

    References
    ----------
    - Danilov et al. (2017, Geosci. Model Dev., 10, 765-789, FESOM2, Sec. 3.2 & 3.4)
    - Zalesak (1979, J. Comput. Phys., 31, 335-362)
    - Adcroft & Campin (2004, Ocean Modelling, 7, 269-284, z* vertical coordinate)
    """
    qp = (q - q_ref) * wet_cf
    h_old_safe = torch.clamp(h_old_z, min=1.0)
    h_new_safe = torch.clamp(h_new_z, min=1.0)
    dh_z = h_new_z - h_old_z

    # 1. Low-order monotone upwind fluxes (horizontal & vertical)
    fl_h = (torch.clamp(h_u_edge, min=0.0) * qp[:, c0] + torch.clamp(h_u_edge, max=0.0) * qp[:, c1]) * wet_ef
    w_int = w_star[1:-1]
    fl_v = torch.where(w_int >= 0.0, w_int * qp[1:], w_int * qp[:-1]) * wet_int_cf

    div_fl_h = (fl_h[:, eoc] * div_w.unsqueeze(0)).sum(dim=-1) * wet_cf
    z_cell = torch.zeros_like(qp[0:1])
    vin_fl = torch.cat([fl_v, z_cell], dim=0) - torch.cat([z_cell, fl_v], dim=0)
    # Exact z* continuity-consistent low-order transported-and-diffused field q_td
    qp_td = ((h_old_safe * qp + dt * (-div_fl_h + vin_fl)) / h_new_safe) * wet_cf

    # 2. High-order 2nd-order centered fluxes & anti-diffusive fluxes A = F^H - F^L
    fh_h = (0.5 * h_u_edge * (qp[:, c0] + qp[:, c1])) * wet_ef
    fh_v = (0.5 * w_int * (qp[:-1] + qp[1:])) * wet_int_cf
    ah_e = (fh_h - fl_h) * wet_ef
    av_int = (fh_v - fl_v) * wet_int_cf

    # 3. Local Voronoi + vertical neighborhood extrema (q_min, q_max) from (qp, qp_td)
    q_hi0 = torch.maximum(qp, qp_td)
    q_lo0 = torch.minimum(qp, qp_td)
    # Gather neighbor extrema across edges using wet-masked sentinel
    big_val = 1.0e6
    e_hi = torch.where(wet_ef > 0.5, torch.maximum(q_hi0[:, c0], q_hi0[:, c1]), torch.full_like(ah_e, -big_val))
    e_lo = torch.where(wet_ef > 0.5, torch.minimum(q_lo0[:, c0], q_lo0[:, c1]), torch.full_like(ah_e, +big_val))
    q_max = torch.maximum(q_hi0, e_hi[:, eoc].max(dim=-1).values)
    q_min = torch.minimum(q_lo0, e_lo[:, eoc].min(dim=-1).values)
    # Include wet vertical neighbors (k-1, k+1)
    q_up_hi = torch.cat([torch.where(wet_int_cf > 0.5, q_hi0[1:], q_hi0[:-1]), q_hi0[-1:]], dim=0)
    q_dn_hi = torch.cat([q_hi0[:1], torch.where(wet_int_cf > 0.5, q_hi0[:-1], q_hi0[1:])], dim=0)
    q_max = torch.maximum(q_max, torch.maximum(q_up_hi, q_dn_hi))
    q_up_lo = torch.cat([torch.where(wet_int_cf > 0.5, q_lo0[1:], q_lo0[:-1]), q_lo0[-1:]], dim=0)
    q_dn_lo = torch.cat([q_lo0[:1], torch.where(wet_int_cf > 0.5, q_lo0[:-1], q_lo0[1:])], dim=0)
    q_min = torch.minimum(q_min, torch.minimum(q_up_lo, q_dn_lo))

    # 4. Zalesak (1979) P^+, P^-, Q^+, Q^-, R^+, R^- and face limiters C_h, C_v in [0, 1]
    # Note: div_w > 0 means edge points OUT of cell i, so incoming anti-diffusive flux is -ah_e * div_w > 0
    a_in_h = -(ah_e[:, eoc] * div_w.unsqueeze(0))
    p_plus_h = torch.clamp(a_in_h, min=0.0).sum(dim=-1)
    p_minus_h = torch.clamp(-a_in_h, min=0.0).sum(dim=-1)
    # Vertical: av_int > 0 flows UPWARD from k+1 into k (into k, out of k+1)
    p_plus_v = torch.cat([torch.clamp(av_int, min=0.0), z_cell], dim=0) + torch.cat([z_cell, torch.clamp(-av_int, min=0.0)], dim=0)
    p_minus_v = torch.cat([torch.clamp(-av_int, min=0.0), z_cell], dim=0) + torch.cat([z_cell, torch.clamp(av_int, min=0.0)], dim=0)

    eps = 1.0e-12
    p_plus = dt * (p_plus_h + p_plus_v) * wet_cf
    p_minus = dt * (p_minus_h + p_minus_v) * wet_cf
    q_plus = h_new_safe * torch.clamp(q_max - qp_td, min=0.0)
    q_minus = h_new_safe * torch.clamp(qp_td - q_min, min=0.0)
    r_plus = torch.where(p_plus > eps, torch.clamp(q_plus / (p_plus + eps), 0.0, 1.0), torch.ones_like(qp))
    r_minus = torch.where(p_minus > eps, torch.clamp(q_minus / (p_minus + eps), 0.0, 1.0), torch.ones_like(qp))

    # Edge limiter: ah_e >= 0 flows c0 -> c1 (out of c0, into c1)
    c_lim_h = torch.where(
        ah_e >= 0.0,
        torch.minimum(r_minus[:, c0], r_plus[:, c1]),
        torch.minimum(r_plus[:, c0], r_minus[:, c1]),
    ) * wet_ef
    # Vertical limiter: av_int >= 0 flows k+1 -> k (out of k+1, into k)
    c_lim_v = torch.where(
        av_int >= 0.0,
        torch.minimum(r_minus[1:], r_plus[:-1]),
        torch.minimum(r_plus[1:], r_minus[:-1]),
    ) * wet_int_cf

    f_adv_h = fl_h + c_lim_h * ah_e
    f_adv_v = fl_v + c_lim_v * av_int

    # 5. Full Rotated Redi Diffusion & KPP Non-Local Flux (identical to standard path)
    dq_dx = ((qp[:, c1] - qp[:, c0]) * inv_dc_edge.unsqueeze(0)) * wet_ef
    if enable_full_redi:
        dq_dz_c = ((qp[1:] - qp[:-1]) / dz_c_int) * wet_int_cf
        dq_dz_e = 0.5 * (dq_dz_c[:, c0] + dq_dz_c[:, c1]) * wet_int_ef
        sx_dqdz_int = sx_int * dq_dz_e
        z_row = torch.zeros_like(dq_dx[0:1])
        sx_dqdz_lay = 0.5 * (torch.cat([sx_dqdz_int, z_row], dim=0) + torch.cat([z_row, sx_dqdz_int], dim=0))
        f_dif = -KAPPA_REDI_HORIZ * h_e * (dq_dx + sx_dqdz_lay) * wet_ef

        dq_dx_int = 0.5 * (dq_dx[:-1] + dq_dx[1:]) * wet_int_ef
        s_dqdx_ew = (sx_int * dq_dx_int) * w_vor_edge.unsqueeze(0)
        s_dot_gradq = (s_dqdx_ew[:, eoc] * active_eoc.unsqueeze(0)).sum(dim=-1) * inv_area_c.unsqueeze(0)
        f_v_redi_cross = (KAPPA_REDI_HORIZ * s_dot_gradq) * wet_int_cf
    else:
        f_dif = -KAPPA_REDI_HORIZ * h_e * dq_dx * wet_ef
        f_v_redi_cross = torch.zeros_like(f_nl_q)

    f_tot_h = f_adv_h + f_dif
    div_h = (f_tot_h[:, eoc] * div_w.unsqueeze(0)).sum(dim=-1) * wet_cf

    h_min_int = torch.minimum(h_new_safe[:-1], h_new_safe[1:])
    q_span = torch.abs(qp[1:] - qp[:-1]) + 0.5
    f_nl_cap = 0.35 * h_min_int * q_span / max(dt, 1.0)
    f_nl_lim = torch.clamp(f_nl_q, -f_nl_cap, f_nl_cap) * wet_int_cf

    f_v_tot = f_adv_v + f_v_redi_cross + f_nl_lim
    v_inflow = torch.cat([f_v_tot, z_cell], dim=0) - torch.cat([z_cell, f_v_tot], dim=0)

    # Exact z* flux-form update: h_new * q_new = h_old * q_old + dt * (-div_h + v_inflow)
    # -> dq = (dt * (-div_h + v_inflow) - dh_z * qp) / h_new_safe (exact constancy & conservation!)
    dq_fct = ((dt * (-div_h + v_inflow) - dh_z * qp) / h_new_safe) * wet_cf
    return _kahan_add(q, dq_fct, q_comp)


def _ogcm_advect_redi_core(
    q: torch.Tensor,
    q_comp: torch.Tensor,
    f_nl_q: torch.Tensor,
    h_u_edge: torch.Tensor,
    sx_int: torch.Tensor,
    w_vel: torch.Tensor,
    h_e: torch.Tensor,
    h_c_safe: torch.Tensor,
    h0_new: torch.Tensor,
    dh0_eff: torch.Tensor,
    dz_c_int: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_cf: torch.Tensor,
    wet_int_ef: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    w_vor_edge: torch.Tensor,
    active_eoc: torch.Tensor,
    inv_area_c: torch.Tensor,
    q_ref: float,
    dt: float,
    enable_full_redi: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-tensor 3D anomaly-form tracer advection + full rotated Redi + KPP non-local flux + Kahan."""
    qp = q - q_ref
    dq_dx = ((qp[:, c1] - qp[:, c0]) * inv_dc_edge.unsqueeze(0)) * wet_ef
    f_adv = torch.clamp(h_u_edge, min=0.0) * qp[:, c0] + torch.clamp(h_u_edge, max=0.0) * qp[:, c1]

    if enable_full_redi:
        dq_dz_c = ((qp[1:] - qp[:-1]) / dz_c_int) * wet_int_cf
        dq_dz_e = 0.5 * (dq_dz_c[:, c0] + dq_dz_c[:, c1]) * wet_int_ef
        sx_dqdz_int = sx_int * dq_dz_e
        z_row = torch.zeros_like(dq_dx[0:1])
        sx_dqdz_lay = 0.5 * (torch.cat([sx_dqdz_int, z_row], dim=0) + torch.cat([z_row, sx_dqdz_int], dim=0))
        f_dif = -KAPPA_REDI_HORIZ * h_e * (dq_dx + sx_dqdz_lay) * wet_ef

        dq_dx_int = 0.5 * (dq_dx[:-1] + dq_dx[1:]) * wet_int_ef
        s_dqdx_ew = (sx_int * dq_dx_int) * w_vor_edge.unsqueeze(0)
        s_dot_gradq = (s_dqdx_ew[:, eoc] * active_eoc.unsqueeze(0)).sum(dim=-1) * inv_area_c.unsqueeze(0)
        f_v_redi_cross = (KAPPA_REDI_HORIZ * s_dot_gradq) * wet_int_cf
    else:
        f_dif = -KAPPA_REDI_HORIZ * h_e * dq_dx * wet_ef
        f_v_redi_cross = torch.zeros_like(f_nl_q)

    f_tot = f_adv + f_dif
    div_h = (f_tot[:, eoc] * div_w.unsqueeze(0)).sum(dim=-1) * wet_cf

    w_int = w_vel[1:-1]
    f_v_adv = torch.where(w_int >= 0.0, w_int * qp[1:], w_int * qp[:-1]) * wet_int_cf
    h_min_int = torch.minimum(h_c_safe[:-1], h_c_safe[1:])
    q_span = torch.abs(qp[1:] - qp[:-1]) + 0.5
    f_nl_cap = 0.35 * h_min_int * q_span / max(dt, 1.0)
    f_nl_lim = torch.clamp(f_nl_q, -f_nl_cap, f_nl_cap) * wet_int_cf

    f_v_int = f_v_adv + f_v_redi_cross + f_nl_lim
    z_cell = torch.zeros_like(q[0:1])
    v_inflow = torch.cat([f_v_int, z_cell], dim=0) - torch.cat([z_cell, f_v_int], dim=0)

    dq_sub = (dt / h_c_safe[1:]) * (-div_h[1:] + v_inflow[1:])
    dq_0 = (dt * (-div_h[0] + v_inflow[0]) - dh0_eff * qp[0]) / h0_new
    dq_adv = torch.cat([dq_0.unsqueeze(0), dq_sub], dim=0) * wet_cf
    return _kahan_add(q, dq_adv, q_comp)


def _ogcm_advect_fct_zstar_batch(
    q: torch.Tensor,
    q_comp: torch.Tensor,
    f_nl_q: torch.Tensor,
    h_u_edge: torch.Tensor,
    sx_int: torch.Tensor,
    w_star: torch.Tensor,
    h_e: torch.Tensor,
    h_old_z: torch.Tensor,
    h_new_z: torch.Tensor,
    dz_c_int: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_cf: torch.Tensor,
    wet_int_ef: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    w_vor_edge: torch.Tensor,
    active_eoc: torch.Tensor,
    inv_area_c: torch.Tensor,
    q_ref: torch.Tensor,
    dt: float,
    enable_full_redi: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched `[B, nzo, nc]` FESOM2-style 3D Zalesak FCT + exact z* + Redi + Kahan."""
    wet_cf_b = wet_cf.unsqueeze(0)
    wet_ef_b = wet_ef.unsqueeze(0)
    wet_ic_b = wet_int_cf.unsqueeze(0)
    wet_ie_b = wet_int_ef.unsqueeze(0)

    qp = (q - q_ref) * wet_cf_b
    h_old_safe = torch.clamp(h_old_z, min=1.0).unsqueeze(0)
    h_new_safe = torch.clamp(h_new_z, min=1.0).unsqueeze(0)
    dh_z = (h_new_z - h_old_z).unsqueeze(0)
    hu_b = h_u_edge.unsqueeze(0)
    w_int = w_star[1:-1].unsqueeze(0)

    fl_h = (torch.clamp(hu_b, min=0.0) * qp[:, :, c0] + torch.clamp(hu_b, max=0.0) * qp[:, :, c1]) * wet_ef_b
    fl_v = torch.where(w_int >= 0.0, w_int * qp[:, 1:], w_int * qp[:, :-1]) * wet_ic_b

    div_w_b = div_w.unsqueeze(0).unsqueeze(0)
    div_fl_h = (fl_h[:, :, eoc] * div_w_b).sum(dim=-1) * wet_cf_b
    z_cell = torch.zeros_like(qp[:, 0:1])
    vin_fl = torch.cat([fl_v, z_cell], dim=1) - torch.cat([z_cell, fl_v], dim=1)
    qp_td = ((h_old_safe * qp + dt * (-div_fl_h + vin_fl)) / h_new_safe) * wet_cf_b

    fh_h = (0.5 * hu_b * (qp[:, :, c0] + qp[:, :, c1])) * wet_ef_b
    fh_v = (0.5 * w_int * (qp[:, :-1] + qp[:, 1:])) * wet_ic_b
    ah_e = (fh_h - fl_h) * wet_ef_b
    av_int = (fh_v - fl_v) * wet_ic_b

    q_hi0 = torch.maximum(qp, qp_td)
    q_lo0 = torch.minimum(qp, qp_td)
    big_val = 1.0e6
    e_hi = torch.where(wet_ef_b > 0.5, torch.maximum(q_hi0[:, :, c0], q_hi0[:, :, c1]), torch.full_like(ah_e, -big_val))
    e_lo = torch.where(wet_ef_b > 0.5, torch.minimum(q_lo0[:, :, c0], q_lo0[:, :, c1]), torch.full_like(ah_e, +big_val))
    q_max = torch.maximum(q_hi0, e_hi[:, :, eoc].max(dim=-1).values)
    q_min = torch.minimum(q_lo0, e_lo[:, :, eoc].min(dim=-1).values)

    q_up_hi = torch.cat([torch.where(wet_ic_b > 0.5, q_hi0[:, 1:], q_hi0[:, :-1]), q_hi0[:, -1:]], dim=1)
    q_dn_hi = torch.cat([q_hi0[:, :1], torch.where(wet_ic_b > 0.5, q_hi0[:, :-1], q_hi0[:, 1:])], dim=1)
    q_max = torch.maximum(q_max, torch.maximum(q_up_hi, q_dn_hi))
    q_up_lo = torch.cat([torch.where(wet_ic_b > 0.5, q_lo0[:, 1:], q_lo0[:, :-1]), q_lo0[:, -1:]], dim=1)
    q_dn_lo = torch.cat([q_lo0[:, :1], torch.where(wet_ic_b > 0.5, q_lo0[:, :-1], q_lo0[:, 1:])], dim=1)
    q_min = torch.minimum(q_min, torch.minimum(q_up_lo, q_dn_lo))

    a_in_h = -(ah_e[:, :, eoc] * div_w_b)
    p_plus_h = torch.clamp(a_in_h, min=0.0).sum(dim=-1)
    p_minus_h = torch.clamp(-a_in_h, min=0.0).sum(dim=-1)
    p_plus_v = torch.cat([torch.clamp(av_int, min=0.0), z_cell], dim=1) + torch.cat([z_cell, torch.clamp(-av_int, min=0.0)], dim=1)
    p_minus_v = torch.cat([torch.clamp(-av_int, min=0.0), z_cell], dim=1) + torch.cat([z_cell, torch.clamp(av_int, min=0.0)], dim=1)

    eps = 1.0e-12
    p_plus = dt * (p_plus_h + p_plus_v) * wet_cf_b
    p_minus = dt * (p_minus_h + p_minus_v) * wet_cf_b
    q_plus = h_new_safe * torch.clamp(q_max - qp_td, min=0.0)
    q_minus = h_new_safe * torch.clamp(qp_td - q_min, min=0.0)
    r_plus = torch.where(p_plus > eps, torch.clamp(q_plus / (p_plus + eps), 0.0, 1.0), torch.ones_like(qp))
    r_minus = torch.where(p_minus > eps, torch.clamp(q_minus / (p_minus + eps), 0.0, 1.0), torch.ones_like(qp))

    c_lim_h = torch.where(
        ah_e >= 0.0,
        torch.minimum(r_minus[:, :, c0], r_plus[:, :, c1]),
        torch.minimum(r_plus[:, :, c0], r_minus[:, :, c1]),
    ) * wet_ef_b
    c_lim_v = torch.where(
        av_int >= 0.0,
        torch.minimum(r_minus[:, 1:], r_plus[:, :-1]),
        torch.minimum(r_plus[:, 1:], r_minus[:, :-1]),
    ) * wet_ic_b

    f_adv_h = fl_h + c_lim_h * ah_e
    f_adv_v = fl_v + c_lim_v * av_int

    dq_dx = ((qp[:, :, c1] - qp[:, :, c0]) * inv_dc_edge.unsqueeze(0).unsqueeze(0)) * wet_ef_b
    if enable_full_redi:
        dq_dz_c = ((qp[:, 1:] - qp[:, :-1]) / dz_c_int.unsqueeze(0)) * wet_ic_b
        dq_dz_e = 0.5 * (dq_dz_c[:, :, c0] + dq_dz_c[:, :, c1]) * wet_ie_b
        sx_dqdz_int = sx_int.unsqueeze(0) * dq_dz_e
        z_row = torch.zeros_like(dq_dx[:, 0:1])
        sx_dqdz_lay = 0.5 * (torch.cat([sx_dqdz_int, z_row], dim=1) + torch.cat([z_row, sx_dqdz_int], dim=1))
        f_dif = -KAPPA_REDI_HORIZ * h_e.unsqueeze(0) * (dq_dx + sx_dqdz_lay) * wet_ef_b

        dq_dx_int = 0.5 * (dq_dx[:, :-1] + dq_dx[:, 1:]) * wet_ie_b
        s_dqdx_ew = (sx_int.unsqueeze(0) * dq_dx_int) * w_vor_edge.unsqueeze(0).unsqueeze(0)
        s_dot_gradq = (s_dqdx_ew[:, :, eoc] * active_eoc.unsqueeze(0).unsqueeze(0)).sum(dim=-1) * inv_area_c.unsqueeze(0).unsqueeze(0)
        f_v_redi_cross = (KAPPA_REDI_HORIZ * s_dot_gradq) * wet_ic_b
    else:
        f_dif = -KAPPA_REDI_HORIZ * h_e.unsqueeze(0) * dq_dx * wet_ef_b
        f_v_redi_cross = torch.zeros_like(f_nl_q)

    f_tot_h = f_adv_h + f_dif
    div_h = (f_tot_h[:, :, eoc] * div_w_b).sum(dim=-1) * wet_cf_b

    h_min_int = torch.minimum(h_new_safe[:, :-1], h_new_safe[:, 1:])
    q_span = torch.abs(qp[:, 1:] - qp[:, :-1]) + 0.5
    f_nl_cap = 0.35 * h_min_int * q_span / max(dt, 1.0)
    f_nl_lim = torch.clamp(f_nl_q, -f_nl_cap, f_nl_cap) * wet_ic_b

    f_v_tot = f_adv_v + f_v_redi_cross + f_nl_lim
    v_inflow = torch.cat([f_v_tot, z_cell], dim=1) - torch.cat([z_cell, f_v_tot], dim=1)

    dq_fct = ((dt * (-div_h + v_inflow) - dh_z * qp) / h_new_safe) * wet_cf_b
    return _kahan_add(q, dq_fct, q_comp)


def _ogcm_advect_redi_batch(
    q: torch.Tensor,
    q_comp: torch.Tensor,
    f_nl_q: torch.Tensor,
    h_u_edge: torch.Tensor,
    sx_int: torch.Tensor,
    w_vel: torch.Tensor,
    h_e: torch.Tensor,
    h_c_safe: torch.Tensor,
    h0_new: torch.Tensor,
    dh0_eff: torch.Tensor,
    dz_c_int: torch.Tensor,
    inv_dc_edge: torch.Tensor,
    wet_cf: torch.Tensor,
    wet_ef: torch.Tensor,
    wet_int_cf: torch.Tensor,
    wet_int_ef: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    eoc: torch.Tensor,
    div_w: torch.Tensor,
    w_vor_edge: torch.Tensor,
    active_eoc: torch.Tensor,
    inv_area_c: torch.Tensor,
    q_ref: torch.Tensor,
    dt: float,
    enable_full_redi: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched `[B, nzo, nc]` 3D anomaly-form tracer advection + full rotated Redi + KPP non-local + Kahan."""
    wet_cf_b = wet_cf.unsqueeze(0)
    wet_ef_b = wet_ef.unsqueeze(0)
    wet_ic_b = wet_int_cf.unsqueeze(0)
    wet_ie_b = wet_int_ef.unsqueeze(0)
    hu_b = h_u_edge.unsqueeze(0)

    qp = q - q_ref
    dq_dx = ((qp[:, :, c1] - qp[:, :, c0]) * inv_dc_edge.unsqueeze(0).unsqueeze(0)) * wet_ef_b
    f_adv = torch.clamp(hu_b, min=0.0) * qp[:, :, c0] + torch.clamp(hu_b, max=0.0) * qp[:, :, c1]

    if enable_full_redi:
        dq_dz_c = ((qp[:, 1:] - qp[:, :-1]) / dz_c_int.unsqueeze(0)) * wet_ic_b
        dq_dz_e = 0.5 * (dq_dz_c[:, :, c0] + dq_dz_c[:, :, c1]) * wet_ie_b
        sx_dqdz_int = sx_int.unsqueeze(0) * dq_dz_e
        z_row = torch.zeros_like(dq_dx[:, 0:1])
        sx_dqdz_lay = 0.5 * (torch.cat([sx_dqdz_int, z_row], dim=1) + torch.cat([z_row, sx_dqdz_int], dim=1))
        f_dif = -KAPPA_REDI_HORIZ * h_e.unsqueeze(0) * (dq_dx + sx_dqdz_lay) * wet_ef_b

        dq_dx_int = 0.5 * (dq_dx[:, :-1] + dq_dx[:, 1:]) * wet_ie_b
        s_dqdx_ew = (sx_int.unsqueeze(0) * dq_dx_int) * w_vor_edge.unsqueeze(0).unsqueeze(0)
        s_dot_gradq = (s_dqdx_ew[:, :, eoc] * active_eoc.unsqueeze(0).unsqueeze(0)).sum(dim=-1) * inv_area_c.unsqueeze(0).unsqueeze(0)
        f_v_redi_cross = (KAPPA_REDI_HORIZ * s_dot_gradq) * wet_ic_b
    else:
        f_dif = -KAPPA_REDI_HORIZ * h_e.unsqueeze(0) * dq_dx * wet_ef_b
        f_v_redi_cross = torch.zeros_like(f_nl_q)

    f_tot = f_adv + f_dif
    div_h = (f_tot[:, :, eoc] * div_w.unsqueeze(0).unsqueeze(0)).sum(dim=-1) * wet_cf_b

    w_int = w_vel[1:-1].unsqueeze(0)
    f_v_adv = torch.where(w_int >= 0.0, w_int * qp[:, 1:], w_int * qp[:, :-1]) * wet_ic_b
    h_min_int = torch.minimum(h_c_safe[:-1], h_c_safe[1:]).unsqueeze(0)
    q_span = torch.abs(qp[:, 1:] - qp[:, :-1]) + 0.5
    f_nl_cap = 0.35 * h_min_int * q_span / max(dt, 1.0)
    f_nl_lim = torch.clamp(f_nl_q, -f_nl_cap, f_nl_cap) * wet_ic_b

    f_v_int = f_v_adv + f_v_redi_cross + f_nl_lim
    z_cell = torch.zeros_like(q[:, 0:1])
    v_inflow = torch.cat([f_v_int, z_cell], dim=1) - torch.cat([z_cell, f_v_int], dim=1)

    dq_sub = (dt / h_c_safe[1:].unsqueeze(0)) * (-div_h[:, 1:] + v_inflow[:, 1:])
    dq_0 = (dt * (-div_h[:, 0] + v_inflow[:, 0]) - dh0_eff.unsqueeze(0) * qp[:, 0]) / h0_new.unsqueeze(0)
    dq_adv = torch.cat([dq_0.unsqueeze(1), dq_sub], dim=1) * wet_cf_b
    return _kahan_add(q, dq_adv, q_comp)


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
    cache: dict | None = None,
    enable_tidal_mixing: bool = True,
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
    if cache is None:
        cache = _get_ogcm_static_cache(ogcm, mesh, is_ocean)
    kv_tide_in = cache["kv_tide_num"] if enable_tidal_mixing else torch.zeros_like(cache["kv_tide_num"])
    if (
        get_active_gpu_backend(T.device) in ("mps", "cuda")
        and T.dtype == torch.float32
        and 2 <= int(ogcm.nz_ocn) <= 64
        and not T.requires_grad
    ):
        return dispatch_ogcm_kpp_vdiff_metal(
            rho_anom=rho_anom, u_cell=u_cell, v_cell=v_cell,
            u_star_o=u_star_o, b0_sfc=b0_sfc, wt_sfc=wt_sfc, ws_sfc=ws_sfc,
            wet_cf=cache["wet_cf"], wet_int_cf=cache["wet_int_cf"],
            h_c=ogcm.h_c, dz_c_int=cache["dz_c_int"],
            z_c=ogcm.z_c, z_int_m=ogcm.z_w[1:-1],
            h_col_kpp=cache["h_col_kpp"], h_col_safe=cache["h_col_safe"],
            f_abs=cache["f_abs"], kv_bg=cache["kv_bg"], kv_tide_num=kv_tide_in,
            kpp_ri_crit=KPP_RI_CRIT, kpp_kv_max=KPP_KV_MAX, kpp_cs_nl=KPP_CS_NONLOCAL,
            shear_ri_0=SHEAR_RI_0, shear_nu_0=SHEAR_NU_0,
            tidal_kv_max=TIDAL_KV_MAX, kappa_v_conv=KAPPA_V_CONV, a_v_mom=A_V_MOM,
        )
    kpp_fn = maybe_cuda_compile(_ogcm_kpp_mixing_core, device=T.device)
    return kpp_fn(
        T, rho_anom, u_cell, v_cell, u_star_o, b0_sfc, wt_sfc, ws_sfc,
        ogcm.wet_c, cache["wet_int_c"], cache["wet_int_cf"], ogcm.h_c,
        cache["dz_c_int"], ogcm.z_c.unsqueeze(1), ogcm.z_w[1:-1].unsqueeze(1),
        cache["is_ocn_f"], cache["h_col_kpp"], cache["h_col_safe"],
        cache["f_abs"], cache["kv_bg"], kv_tide_in,
    )


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
    sw_par_ocn: torch.Tensor | None = None,
    u10_ocn: torch.Tensor | None = None,
    co2_ppmv: float = 284.3,
    river_mouth_flux: torch.Tensor | None = None,
    enable_fct_zstar: bool = False,
    enable_scale_aware_gm: bool = True,
    enable_submeso: bool = True,
    enable_tidal_mixing: bool = True,
    enable_o2_omz: bool = False,
    langmuir_fac: torch.Tensor | None = None,
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
    cache = _get_ogcm_static_cache(ogcm, mesh, is_ocean)

    if can_use_cuda_graph(dev) and ogcm.bgc is None:
        has_evp = ogcm.seaice_evp is not None
        has_sw = sw_net_ocn is not None
        has_riv = river_mouth_flux is not None
        has_lang = langmuir_fac is not None

        live_in: list[torch.Tensor] = [
            ogcm.temp, ogcm.sal, ogcm.t_comp, ogcm.s_comp,
            ogcm.u_edge, ogcm.u_baro, ogcm.eta, ogcm.m_seaice_prev,
            sfc.t_ocean_ml, sfc.m_seaice, sfc.swe_seaice,
            tau_x_sfc, tau_y_sfc, f_seaice, evap_cell, precip_cell,
            runoff_total, runoff_glacier,
        ]
        if has_sw:
            assert sw_net_ocn is not None
            live_in.append(sw_net_ocn)
        if has_riv:
            assert river_mouth_flux is not None
            live_in.append(river_mouth_flux)
        if has_lang:
            assert langmuir_fac is not None
            live_in.append(langmuir_fac)
        if has_evp:
            evp0 = ogcm.seaice_evp
            live_in.extend([
                evp0.a_cat, evp0.v_cat, evp0.u_ice_e,
                evp0.sigma_1, evp0.sigma_s, evp0.m_seaice_sync,
            ])

        g_key = (
            "ogcm_voronoi",
            nzo, nc, ne,
            int(mesh.cells_on_edge.data_ptr()),
            str(dev), str(dtype),
            bool(enable_seaice_drift), bool(has_evp),
            bool(has_sw), bool(has_riv), bool(has_lang),
            bool(enable_fct_zstar), bool(enable_scale_aware_gm),
            bool(enable_submeso), bool(enable_tidal_mixing),
            round(float(dt), 6),
        )

        def _cg_body(inp: list[torch.Tensor]) -> list[torch.Tensor]:
            idx = 18
            sw_i = inp[idx] if has_sw else None
            idx += int(has_sw)
            riv_i = inp[idx] if has_riv else None
            idx += int(has_riv)
            lang_i = inp[idx] if has_lang else None
            idx += int(has_lang)

            evp_v = None
            if has_evp:
                from aoba.seaice_evp import SeaIceEVPState

                evp_v = SeaIceEVPState(
                    a_cat=inp[idx],
                    v_cat=inp[idx + 1],
                    u_ice_e=inp[idx + 2],
                    sigma_1=inp[idx + 3],
                    sigma_s=inp[idx + 4],
                    m_seaice_sync=inp[idx + 5],
                )
            ogcm_d = dict(ogcm.__dict__)
            ogcm_d["temp"] = inp[0]
            ogcm_d["sal"] = inp[1]
            ogcm_d["t_comp"] = inp[2]
            ogcm_d["s_comp"] = inp[3]
            ogcm_d["u_edge"] = inp[4]
            ogcm_d["u_baro"] = inp[5]
            ogcm_d["eta"] = inp[6]
            ogcm_d["m_seaice_prev"] = inp[7]
            ogcm_d["seaice_evp"] = evp_v
            ogcm_v = OceanOGCMState(**{k: v for k, v in ogcm_d.items() if k in OceanOGCMState.__dataclass_fields__})
            ogcm_v._static_cache = getattr(ogcm, "_static_cache", None)  # type: ignore[attr-defined]

            sfc_d = dict(sfc.__dict__)
            sfc_d["t_ocean_ml"] = inp[8]
            sfc_d["m_seaice"] = inp[9]
            sfc_d["swe_seaice"] = inp[10]
            sfc_v = SurfaceState(**{k: v for k, v in sfc_d.items() if k in SurfaceState.__dataclass_fields__})

            _step_ogcm_voronoi_impl(
                ogcm_v, sfc_v, mesh, cache, is_ocean,
                inp[11], inp[12], inp[13], inp[14], inp[15], inp[16], inp[17], dt,
                enable_seaice_drift=enable_seaice_drift,
                sw_net_ocn=sw_i, sw_par_ocn=None, u10_ocn=None, co2_ppmv=co2_ppmv,
                river_mouth_flux=riv_i,
                enable_fct_zstar=enable_fct_zstar,
                enable_scale_aware_gm=enable_scale_aware_gm,
                enable_submeso=enable_submeso,
                enable_tidal_mixing=enable_tidal_mixing,
                enable_o2_omz=enable_o2_omz,
                langmuir_fac=lang_i,
            )
            res = [
                ogcm_v.temp, ogcm_v.sal, ogcm_v.t_comp, ogcm_v.s_comp,
                ogcm_v.u_edge, ogcm_v.u_baro, ogcm_v.eta, ogcm_v.m_seaice_prev,
                ogcm_v.h_bl, ogcm_v.w_vel, ogcm_v.kv_last,
                sfc_v.t_ocean_ml, sfc_v.m_seaice, sfc_v.swe_seaice,
            ]
            if has_evp and evp_v is not None:
                res.extend([
                    evp_v.a_cat, evp_v.v_cat, evp_v.u_ice_e,
                    evp_v.sigma_1, evp_v.sigma_s, evp_v.m_seaice_sync,
                ])
            return res

        out = run_cuda_graph_cached(g_key, live_in, _cg_body, warmup_calls=2, clone_outputs=False)
        ogcm.temp.copy_(out[0])
        ogcm.sal.copy_(out[1])
        ogcm.t_comp.copy_(out[2])
        ogcm.s_comp.copy_(out[3])
        ogcm.u_edge.copy_(out[4])
        ogcm.u_baro.copy_(out[5])
        ogcm.eta.copy_(out[6])
        ogcm.m_seaice_prev.copy_(out[7])
        ogcm.h_bl = out[8].clone()
        ogcm.w_vel = out[9].clone()
        ogcm.kv_last = out[10].clone()
        sfc.t_ocean_ml.copy_(out[11])
        sfc.m_seaice.copy_(out[12])
        sfc.swe_seaice.copy_(out[13])
        if has_evp and ogcm.seaice_evp is not None:
            ogcm.seaice_evp.a_cat.copy_(out[14])
            ogcm.seaice_evp.v_cat.copy_(out[15])
            ogcm.seaice_evp.u_ice_e.copy_(out[16])
            ogcm.seaice_evp.sigma_1.copy_(out[17])
            ogcm.seaice_evp.sigma_s.copy_(out[18])
            ogcm.seaice_evp.m_seaice_sync.copy_(out[19])
        return

    _step_ogcm_voronoi_impl(
        ogcm, sfc, mesh, cache, is_ocean,
        tau_x_sfc, tau_y_sfc, f_seaice, evap_cell, precip_cell,
        runoff_total, runoff_glacier, dt,
        enable_seaice_drift=enable_seaice_drift,
        sw_net_ocn=sw_net_ocn, sw_par_ocn=sw_par_ocn, u10_ocn=u10_ocn, co2_ppmv=co2_ppmv,
        river_mouth_flux=river_mouth_flux,
        enable_fct_zstar=enable_fct_zstar,
        enable_scale_aware_gm=enable_scale_aware_gm,
        enable_submeso=enable_submeso,
        enable_tidal_mixing=enable_tidal_mixing,
        enable_o2_omz=enable_o2_omz,
        langmuir_fac=langmuir_fac,
    )


def _step_ogcm_voronoi_impl(
    ogcm: OceanOGCMState,
    sfc: SurfaceState,
    mesh: SCVTMesh,
    cache: dict,
    is_ocean: torch.Tensor,
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
    sw_par_ocn: torch.Tensor | None = None,
    u10_ocn: torch.Tensor | None = None,
    co2_ppmv: float = 284.3,
    river_mouth_flux: torch.Tensor | None = None,
    enable_fct_zstar: bool = False,
    enable_scale_aware_gm: bool = True,
    enable_submeso: bool = True,
    enable_tidal_mixing: bool = True,
    enable_o2_omz: bool = False,
    langmuir_fac: torch.Tensor | None = None,
) -> None:
    """Core 3D Voronoi OGCM + sea-ice advancement body (shared across CUDA Graph & MPS/CPU)."""
    dev = ogcm.temp.device
    dtype = ogcm.temp.dtype
    nzo = ogcm.nz_ocn
    nc = mesh.n_cells
    ne = mesh.n_edges
    is_ocn_f = cache["is_ocn_f"]
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    v0_e = mesh.verts_on_edge[:, 0]
    v1_e = mesh.verts_on_edge[:, 1]
    ep_idx = mesh.trisk_edges
    w_trisk = mesh.trisk_weights
    wet_c = ogcm.wet_c
    wet_e = ogcm.wet_e
    wet_cf = cache["wet_cf"]
    wet_ef = cache["wet_ef"]
    h_c = ogcm.h_c
    h_e = ogcm.h_e
    h_c_safe = cache["h_c_safe"]
    h_e_safe = cache["h_e_safe"]
    wet_int_c = cache["wet_int_c"]
    wet_int_e = cache["wet_int_e"]
    wet_int_cf = cache["wet_int_cf"]
    wet_int_ef = cache["wet_int_ef"]
    dz_c_int = cache["dz_c_int"]
    inv_dc_edge = cache["inv_dc_edge"]

    # 0-3. Surface buoyancy, calving, salinity, Jerlov SW penetration, Roquet EOS & wind stress
    if river_mouth_flux is not None:
        # Priority-Flood Voronoi river routing delivers liquid runoff to ocean river mouths;
        # add coastal glacier calving meltwater via coastal_ocean_idx so total freshwater + latent heat closes.
        r_calve = torch.zeros(nc, device=dev, dtype=dtype)
        r_calve.scatter_add_(0, mesh.coastal_ocean_idx, mesh.area_cell * torch.clamp(runoff_glacier, min=0.0))
        r_coast = torch.clamp(river_mouth_flux, min=0.0) + r_calve * cache["inv_area_c"]
    else:
        r_coast = torch.zeros(nc, device=dev, dtype=dtype)
        r_coast.scatter_add_(0, mesh.coastal_ocean_idx, mesh.area_cell * torch.clamp(runoff_total, min=0.0))
        r_coast = r_coast * cache["inv_area_c"]  # [kg m^-2 s^-1]

    q_calve = torch.zeros(nc, device=dev, dtype=dtype)
    q_calve.scatter_add_(0, mesh.coastal_ocean_idx, C.L_F0 * mesh.area_cell * torch.clamp(runoff_glacier, min=0.0))
    q_calve = q_calve * cache["inv_area_c"]  # [W m^-2]

    apply_sw = (sw_net_ocn is not None) and (nzo >= 2)
    sw_in = sw_net_ocn if sw_net_ocn is not None else torch.zeros_like(sfc.t_ocean_ml)
    z_c_2d = ogcm.z_c.unsqueeze(1)
    z_int_m = ogcm.z_w[1:-1].unsqueeze(1)

    use_gpu_ogcm = get_active_gpu_backend(dev) in ("mps", "cuda") and ogcm.temp.dtype == torch.float32
    if use_gpu_ogcm:
        T, S, t_comp, s_comp, wt_sfc, ws_sfc, b0_sfc, rho_anom, p_baroc = dispatch_ogcm_pre_kpp_metal(
            T=ogcm.temp, S=ogcm.sal, t_comp=ogcm.t_comp, s_comp=ogcm.s_comp,
            sfc_t_ocean_ml=sfc.t_ocean_ml, sfc_m_seaice=sfc.m_seaice, m_seaice_prev=ogcm.m_seaice_prev,
            r_coast=r_coast, q_calve=q_calve, evap_cell=evap_cell, precip_cell=precip_cell,
            sw_net_ocn=sw_in, wet_cf=wet_cf, h_c_safe=h_c_safe, dz_nom=ogcm.dz_nom, z_c=ogcm.z_c,
            frac_out_0=cache["frac_out_0"], frac_sub=cache["frac_sub"],
            dt=dt, apply_sw=apply_sw,
            t_ref_eos=T_REF_EOS_K, s_ref_psu=S_REF_PSU, s_seaice_psu=S_SEAICE_PSU,
            eos_a0=EOS_ALPHA0, eos_at=EOS_ALPHA_T, eos_az=EOS_ALPHA_Z, eos_as=EOS_ALPHA_S,
            eos_b0=EOS_BETA0,
        )
        ice_atten = 1.0 - 0.5 * torch.clamp(f_seaice, 0.0, 1.0)
        tx_ocn = -ice_atten * tau_x_sfc
        ty_ocn = -ice_atten * tau_y_sfc
        u_star_o = torch.sqrt(torch.sqrt(tx_ocn * tx_ocn + ty_ocn * ty_ocn + 1.0e-12) / C.RHO_SEAWATER)
        tx_e = 0.5 * (tx_ocn[c0] + tx_ocn[c1])
        ty_e = 0.5 * (ty_ocn[c0] + ty_ocn[c1])
        tau_n_e = (tx_e * mesh.east_normal_edge + ty_e * mesh.north_normal_edge) * wet_ef[0]
    else:
        pre_kpp_fn = maybe_cuda_compile(_ogcm_pre_kpp_core, device=dev)
        (
            T, S, t_comp, s_comp, wt_sfc, ws_sfc, b0_sfc,
            rho_anom, p_baroc, u_star_o, tau_n_e,
        ) = pre_kpp_fn(
            ogcm.temp, ogcm.sal, ogcm.t_comp, ogcm.s_comp,
            sfc.t_ocean_ml, sfc.m_seaice, ogcm.m_seaice_prev,
            r_coast, q_calve, evap_cell, precip_cell, sw_in,
            tau_x_sfc, tau_y_sfc, f_seaice,
            is_ocean, is_ocn_f, wet_cf, wet_ef, h_c, h_c_safe, ogcm.dz_nom.unsqueeze(1), z_c_2d,
            cache["frac_out_0"], cache["frac_sub"], c0, c1,
            mesh.east_normal_edge, mesh.north_normal_edge, dt, apply_sw,
        )

    # 3b. Reconstruct 3D cell-centered ocean currents & diagnose KPP / shear / tidal mixing
    u = ogcm.u_edge * wet_ef
    u_cell, v_cell = reconstruct_cell_winds(mesh, u)
    kv_diag, av_diag, h_bl, f_nl_t, f_nl_s = _diagnose_kpp_and_vertical_mixing(
        T=T, S=S, rho_anom=rho_anom, u_cell=u_cell, v_cell=v_cell,
        u_star_o=u_star_o, b0_sfc=b0_sfc, wt_sfc=wt_sfc, ws_sfc=ws_sfc,
        ogcm=ogcm, mesh=mesh, is_ocean=is_ocean, cache=cache,
        enable_tidal_mixing=enable_tidal_mixing,
    )
    if langmuir_fac is not None:
        in_bl = (z_int_m < torch.clamp(h_bl.unsqueeze(0), min=1.0)).to(dtype)
        l_scale = 1.0 + in_bl * (torch.clamp(langmuir_fac.to(device=dev, dtype=dtype), 1.0, 2.2).unsqueeze(0) - 1.0)
        kv_diag = torch.clamp(kv_diag * l_scale, max=KPP_KV_MAX)
        av_diag = torch.clamp(av_diag * l_scale, max=KPP_KV_MAX)
    ogcm.h_bl = h_bl

    # 4. 3D Baroclinic Momentum Step (WS-RK3 TRiSK Coriolis + Baroclinic PGF + Bernoulli -grad K
    #    + Scale-Aware Biharmonic Hyperviscosity + Residual Munk Floor + Semi-Implicit Drag;
    #    AOBA Formulation Eq. (9.6); Ringler et al., 2010, 2013; Griffies & Hallberg, 2000).
    dc_min = getattr(mesh, "_dc_edge_min_m", None)
    if dc_min is None:
        dc_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dc_min)
    delta_m4 = 0.65 * dc_min
    nu_4 = min(BETA_MUNK_OGCM * (delta_m4 ** 5), 0.0010 * (dc_min ** 4) / max(dt, 1.0))
    delta_m2 = 0.25 * dc_min
    a_h = min(BETA_MUNK_OGCM * (delta_m2 ** 3), 0.04 * (dc_min ** 2) / max(dt, 1.0))

    # Symmetric wet-masked biharmonic operator D = wet * del^2 * wet guarantees strict energy
    # dissipation <u, -nu_4 * D^2 u> = -nu_4 ||D u||^2 <= 0 at all interior and coastal wet edges.
    use_gpu_ogcm = get_active_gpu_backend(dev) in ("mps", "cuda") and u.dtype == torch.float32
    w_vor_edge = cache["w_vor_edge"]
    active_w_c = cache["active_w_c"]

    if use_gpu_ogcm:
        wet_vf = cache.get("wet_vf")
        if wet_vf is None:
            wet_vf = (wet_ef[:, mesh.edges_on_vert].sum(dim=-1) > 0.0).to(dtype).contiguous()
            cache["wet_vf"] = wet_vf
        ws_bc = cache.get("ws_bc")
        if ws_bc is None:
            nv = int(mesh.n_verts)
            ws_bc = {
                "div": torch.zeros((nzo, nc), device=dev, dtype=dtype),
                "zeta": torch.zeros((nzo, nv), device=dev, dtype=dtype),
                "eta": torch.zeros((nzo, ne), device=dev, dtype=dtype),
                "lap2": torch.zeros((nzo, ne), device=dev, dtype=dtype),
                "lap4": torch.zeros((nzo, ne), device=dev, dtype=dtype),
                "f_fixed": torch.zeros((nzo, ne), device=dev, dtype=dtype),
                "cor": torch.zeros((nzo, ne), device=dev, dtype=dtype),
                "u_a": torch.zeros((nzo, ne), device=dev, dtype=dtype),
                "u_b": torch.zeros((nzo, ne), device=dev, dtype=dtype),
            }
            cache["ws_bc"] = ws_bc
        lap2_u_wet = dispatch_ogcm_vec_lap_wet_metal(
            mesh, u_edge=u, wet_cf=wet_cf, wet_vf=wet_vf, wet_ef=wet_ef,
            div_buf=ws_bc["div"], zeta_buf=ws_bc["zeta"], lap_out=ws_bc["lap2"],
        )
        lap4_u = dispatch_ogcm_vec_lap_wet_metal(
            mesh, u_edge=lap2_u_wet, wet_cf=wet_cf, wet_vf=wet_vf, wet_ef=wet_ef,
            div_buf=ws_bc["div"], zeta_buf=ws_bc["zeta"], lap_out=ws_bc["lap4"],
        )
        w_vor_u2 = w_vor_edge.unsqueeze(0) * (u * u)
        ke_c = 0.5 * dispatch_scvt_weighted_edge_sum_metal(mesh, w_vor_u2, active_w_c) * wet_cf
        p_bern = (p_baroc + C.RHO_SEAWATER * ke_c) * wet_cf

        # Fused GPU baroclinic RK3 + wet-skipping TRiSK Coriolis + register-assembled momentum Thomas solve
        f_fixed = dispatch_ogcm_baroclinic_f_fixed_metal(
            mesh,
            p_baroc=p_bern, lap2_u=lap2_u_wet, lap4_u=lap4_u, tau_n_e=tau_n_e,
            h_e0_safe=h_e_safe[0], wet_ef=wet_ef, a_h=a_h, nu_4=nu_4, out=ws_bc["f_fixed"],
        )
        is_bot_f = cache.get("is_bot_ef")
        if is_bot_f is None:
            is_bot_f = (cache["is_bot_e"].to(u.dtype) * (C_DRAG_BOT / h_e_safe)).contiguous()
            cache["is_bot_ef"] = is_bot_f
        u_s = u
        u_out_seq = (ws_bc["u_a"], ws_bc["u_b"], ws_bc["u_a"])
        for rk_dt, u_dst in zip((dt / 3.0, 0.5 * dt, dt), u_out_seq):
            cor_3d = dispatch_ogcm_trisk_coriolis_wet_metal(
                mesh, u_edge=u_s, wet_vf=wet_vf, wet_ef=wet_ef,
                zeta_buf=ws_bc["zeta"], eta_buf=ws_bc["eta"], rot_out=ws_bc["cor"],
                pre_zeroed=True,
            )
            u_s = dispatch_ogcm_baroclinic_rk3_update_metal(
                u0=u, u_s=u_s, cor_3d=cor_3d, f_fixed=f_fixed,
                is_bot_f=is_bot_f, wet_ef=wet_ef,
                rk_dt=rk_dt, r_rayleigh=R_RAYLEIGH_OCN, out=u_dst,
            )
        u_star = u_s
        if 1 <= nzo <= 64 and u_star.dtype == torch.float32:
            u_star_raw = dispatch_ogcm_mom_vdiff_metal(
                mesh,
                u_star=u_star, av_diag=av_diag, h_e_safe=h_e_safe,
                dz_int_e=cache["dz_int_e"], wet_ef=wet_ef, wet_int_ef=wet_int_ef,
                dt=dt,
            )
        else:
            av_e = 0.5 * (av_diag[:, c0] + av_diag[:, c1])
            c_mom = (dt * av_e / cache["dz_int_e"]) * wet_int_ef
            z_row = torch.zeros_like(u[0:1])
            a_m = torch.cat([z_row, -c_mom / h_e_safe[1:]], dim=0)
            d_m = torch.cat([-c_mom / h_e_safe[:-1], z_row], dim=0)
            b_m = 1.0 - a_m - d_m
            u_star_raw = solve_tridiagonal_batched(a_m, b_m, d_m, u_star)
    else:
        lap2_u, _ = compute_vector_laplacian_edge(mesh, u)
        lap2_u_wet = lap2_u * wet_ef
        lap4_u, _ = compute_vector_laplacian_edge(mesh, lap2_u_wet)
        visc_h_edge = (a_h * lap2_u_wet - nu_4 * lap4_u) * wet_ef
        w_vor_u2 = w_vor_edge.unsqueeze(0) * (u * u)
        ke_c = (
            0.5
            * (w_vor_u2[:, mesh.edges_on_cell] * cache["active_eoc"].unsqueeze(0)).sum(dim=-1)
            * cache["inv_area_c"].unsqueeze(0)
            * wet_cf
        )
        p_bern = (p_baroc + C.RHO_SEAWATER * ke_c) * wet_cf
        bc_rk3_fn = maybe_cuda_compile(_ogcm_baroclinic_rk3_core, device=dev)
        u_star, a_m, b_m, d_m = bc_rk3_fn(
            u, visc_h_edge, p_bern, av_diag, tau_n_e,
            mesh.f_vert, mesh.edges_on_vert, cache["vort_w"],
            v0_e, v1_e, ep_idx, w_trisk, c0, c1, inv_dc_edge,
            cache["is_bot_e"], h_e_safe, cache["dz_int_e"], wet_ef, wet_int_ef, 1.0, dt,
        )
        u_star_raw = solve_tridiagonal_batched(a_m, b_m, d_m, u_star)

    # 5. Split-Explicit 2D Barotropic Free-Surface Mode (Wicker-Skamarock RK3)
    h_col_e_safe = cache["h_col_e_safe"]
    ocn_e0f = cache["ocn_e0f"]
    h_max = cache["h_max"]
    c_bt = math.sqrt(C.GRAVITY * max(h_max, 50.0))
    dt_bt_cfl = 0.35 * dc_min / max(c_bt, 10.0)
    n_bt = max(1, min(int(math.ceil(dt / max(dt_bt_cfl, 1.0))), 192))
    dt_bt = dt / float(n_bt)

    u_prev_baro = (u * h_e).sum(dim=0) / h_col_e_safe * ocn_e0f
    eta_old = ogcm.eta * is_ocn_f
    g_inv_dc = -C.GRAVITY * inv_dc_edge
    eoc = mesh.edges_on_cell  # [nc, 6]
    div_w = cache["div_w"]

    if get_active_gpu_backend(dev) in ("mps", "cuda"):
        c0_i32 = cache.get("c0_i32")
        if c0_i32 is None:
            c0_i32 = c0.to(torch.int32).contiguous()
            c1_i32 = c1.to(torch.int32).contiguous()
            eoc_i32 = eoc.to(torch.int32).contiguous()
            cache["c0_i32"] = c0_i32
            cache["c1_i32"] = c1_i32
            cache["eoc_i32"] = eoc_i32
        else:
            c1_i32 = cache["c1_i32"]
            eoc_i32 = cache["eoc_i32"]
        u_star = u_star_raw * wet_ef
        u_star_baro = (u_star * h_e).sum(dim=0) / h_col_e_safe * ocn_e0f
        r_bt_pert = 1.0 / max(dt, 1.0)
        f_baro = (u_star_baro - u_prev_baro) / max(dt, 1.0) + r_bt_pert * u_prev_baro
        u_bt, u_bt_mean = dispatch_ogcm_barotropic_rk3_metal(
            u_bt_init=ogcm.u_baro * ocn_e0f,
            eta_init=eta_old,
            f_baro=f_baro,
            g_inv_dc=g_inv_dc,
            h_col_e=ogcm.h_col_e,
            ocn_e0f=ocn_e0f,
            is_ocn_f=is_ocn_f,
            c0_i32=c0_i32,
            c1_i32=c1_i32,
            eoc_i32=eoc_i32,
            div_w=div_w,
            n_bt=n_bt,
            dt_bt=dt_bt,
        )
    else:
        baro_fn = maybe_cuda_compile(_ogcm_barotropic_rk3_core, device=dev)
        u_star, u_star_baro, u_bt, u_bt_mean = baro_fn(
            u_star_raw, u_prev_baro, ogcm.u_baro, eta_old, g_inv_dc, h_e, ogcm.h_col_e, h_col_e_safe,
            wet_ef, ocn_e0f, is_ocn_f, c0, c1, eoc, div_w, n_bt, dt_bt, dt,
        )
    # Time-filtered barotropic mode matches u_new depth mean (Ringler et al., 2013; Shchepetkin & McWilliams, 2005)
    ogcm.u_baro = u_bt_mean

    # 6-7. GM90 Eddy Bolus, Mass-Conserving Vertical Velocity `w`, Free Surface `eta` & Redi Kv
    w_vor_edge = cache["w_vor_edge"]
    active_eoc = cache["active_eoc"]
    inv_area_c = cache["inv_area_c"]
    active_w_c = cache.get("active_w_c")
    if active_w_c is None:
        active_w_c = (active_eoc * inv_area_c.unsqueeze(-1)).contiguous()
        cache["active_w_c"] = active_w_c
    if use_gpu_ogcm:
        u_new, sx_int, h_u_edge, s2_edge_w = dispatch_ogcm_gm_bolus_edge_metal(
            mesh,
            w_vor_edge=w_vor_edge, u_star=u_star, u_bt_mean=u_bt_mean, u_star_baro=u_star_baro,
            rho_anom=rho_anom, dz_c_int=dz_c_int, wet_int_cf=wet_int_cf,
            h_bl=h_bl, h_e=h_e, h_e_safe=h_e_safe, z_int_m=ogcm.z_w[1:-1],
            wet_ef=wet_ef, wet_int_ef=wet_int_ef,
            gm_scale_e=cache["gm_scale_e"] if enable_scale_aware_gm else None,
            fk_coef_e=cache["fk_coef_e"] if enable_submeso else None,
            kappa_gm=KAPPA_GM, gm_slope_max=GM_SLOPE_MAX,
            enable_scale_aware_gm=enable_scale_aware_gm,
            enable_submeso=enable_submeso,
            enable_full_redi=bool(ENABLE_FULL_REDI),
        )
        # Fused Metal/CUDA SCVT divergence (zero [nzo, nc, 6] allocation!)
        div_hu = compute_scvt_divergence(mesh, h_u_edge) * wet_cf
        w_up = -torch.flip(torch.cumsum(torch.flip(div_hu, dims=[0]), dim=0), dims=[0]) * wet_cf
        z_c = torch.zeros_like(div_hu[0:1])
        w_vel = torch.cat([z_c, w_up[1:], z_c], dim=0)
        eta_new = torch.clamp((eta_old + dt * w_up[0]) * is_ocn_f, -5.0, 5.0)
        h0_old = torch.clamp(h_c_safe[0] + eta_old, min=10.0)
        h0_new = torch.clamp(h_c_safe[0] + eta_new, min=10.0)
        dh0_eff = h0_new - h0_old
        if ENABLE_FULL_REDI:
            s2_cell = dispatch_scvt_weighted_edge_sum_metal(mesh, s2_edge_w, active_w_c)
            kv_redi = (KAPPA_REDI_HORIZ * s2_cell) * wet_int_cf
        else:
            kv_redi = torch.zeros_like(wet_int_cf)
    else:
        gm_w_fn = maybe_cuda_compile(_ogcm_gm_and_w_core, device=dev)
        u_new, sx_int, h_u_edge, w_vel, eta_new, h0_new, dh0_eff, kv_redi = gm_w_fn(
            u_star, u_bt_mean, u_star_baro, rho_anom, h_bl, eta_old,
            h_e, h_e_safe, h_c_safe, dz_c_int, z_int_m, inv_dc_edge,
            wet_cf, wet_ef, wet_int_cf, wet_int_ef, is_ocn_f,
            c0, c1, eoc, div_w, w_vor_edge, active_eoc, inv_area_c,
            dt, bool(ENABLE_FULL_REDI),
            cache["gm_scale_e"] if enable_scale_aware_gm else None,
            cache["fk_coef_e"] if enable_submeso else None,
            bool(enable_submeso),
        )
    ogcm.u_edge = u_new
    ogcm.eta = eta_new
    ogcm.w_vel = w_vel

    # 8. 3D Conservative Flux-Form Tracer Advection (Anomaly Form q' = q - q_ref to minimize float32 cancellation)
    #    + Full Rotated Redi Diffusion + KPP Non-Local Flux + Neumaier/Kahan Compensated Summation
    if enable_fct_zstar:
        # FESOM2-style exact z* proportional column stretching h*_k = h_c * (1 + eta / H) + Zalesak FCT
        # (Danilov et al., 2017; Adcroft & Campin, 2004). Exact continuity:
        # (h*_new - h*_old)/dt + div_h(h_u_edge) - (w*_below - w*_above) == 0 at EVERY layer k!
        h_col_s = cache["h_col_safe"]
        scale_old = torch.clamp(1.0 + eta_old / h_col_s, min=0.5, max=1.5)
        scale_new = torch.clamp(1.0 + eta_new / h_col_s, min=0.5, max=1.5)
        h_old_z = h_c * scale_old.unsqueeze(0)
        h_new_z = h_c * scale_new.unsqueeze(0)
        # Static vertical fraction below each interior interface (cached once!)
        frac_below = cache.get("frac_below")
        if frac_below is None:
            h_below = torch.flip(torch.cumsum(torch.flip(h_c, dims=[0]), dim=0), dims=[0])
            frac_below = ((h_below[1:] / h_col_s.unsqueeze(0)) * wet_int_cf).contiguous()
            cache["frac_below"] = frac_below
        deta_dt_eff = ((eta_new - eta_old) / max(dt, 1.0)) * is_ocn_f
        w_star_int = (w_vel[1:-1] - frac_below * deta_dt_eff.unsqueeze(0)) * wet_int_cf
        z_w0 = torch.zeros_like(w_vel[0:1])
        w_star = torch.cat([z_w0, w_star_int, z_w0], dim=0)

        ts_ref = cache.get("ts_ref_b")
        if ts_ref is None:
            ts_ref = torch.tensor([T_REF_EOS_K, S_REF_PSU], device=dev, dtype=T.dtype).view(2, 1, 1)
            cache["ts_ref_b"] = ts_ref
        if use_gpu_ogcm:
            ts_q, ts_c = dispatch_ogcm_fct_zstar_metal(
                q=torch.stack([T, S], dim=0),
                q_comp=torch.stack([t_comp, s_comp], dim=0),
                f_nl_q=torch.stack([f_nl_t, f_nl_s], dim=0),
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
                q_ref=ts_ref,
                dt=dt,
                kappa_redi=float(KAPPA_REDI_HORIZ),
                enable_full_redi=bool(ENABLE_FULL_REDI),
            )
        else:
            fct_b_fn = maybe_cuda_compile(_ogcm_advect_fct_zstar_batch, device=dev)
            ts_q, ts_c = fct_b_fn(
                torch.stack([T, S], dim=0),
                torch.stack([t_comp, s_comp], dim=0),
                torch.stack([f_nl_t, f_nl_s], dim=0),
                h_u_edge, sx_int, w_star, h_e, h_old_z, h_new_z,
                dz_c_int, inv_dc_edge, wet_cf, wet_ef, wet_int_cf, wet_int_ef,
                c0, c1, eoc, div_w, w_vor_edge, active_eoc, inv_area_c,
                ts_ref, dt, bool(ENABLE_FULL_REDI),
            )
        T, S = ts_q[0], ts_q[1]
        t_comp, s_comp = ts_c[0], ts_c[1]
    else:
        ts_ref = cache.get("ts_ref_b")
        if ts_ref is None:
            ts_ref = torch.tensor([T_REF_EOS_K, S_REF_PSU], device=dev, dtype=T.dtype).view(2, 1, 1)
            cache["ts_ref_b"] = ts_ref
        adv_b_fn = maybe_cuda_compile(_ogcm_advect_redi_batch, device=dev)
        ts_q, ts_c = adv_b_fn(
            torch.stack([T, S], dim=0),
            torch.stack([t_comp, s_comp], dim=0),
            torch.stack([f_nl_t, f_nl_s], dim=0),
            h_u_edge, sx_int, w_vel, h_e, h_c_safe, h0_new, dh0_eff,
            dz_c_int, inv_dc_edge, wet_cf, wet_ef, wet_int_cf, wet_int_ef,
            c0, c1, eoc, div_w, w_vor_edge, active_eoc, inv_area_c,
            ts_ref, dt, bool(ENABLE_FULL_REDI),
        )
        T, S = ts_q[0], ts_q[1]
        t_comp, s_comp = ts_c[0], ts_c[1]

    # 9-10. Strictly Conservative Flux-Increment Implicit Vertical Mixing (Float32-Safe!) & Freezing Check
    #    Solving `(I + L) dq = -L q` and reconstructing telescoping fluxes `F_imp = c_dif * d(q + dq)`
    #    prevents float32 diagonal underflow (`1.0f + c_dif/h == 1.0f` in the abyssal ocean) from
    #    creating spurious heat/salt, while `_kahan_add` retains sub-ULP deep-ocean tendencies.
    h_c_vdiff = torch.clamp(h_new_z, min=1.0) if enable_fct_zstar else h_c_safe
    h0_vdiff = h_c_vdiff[0] if enable_fct_zstar else h0_new
    a_t: torch.Tensor | None = None
    b_t: torch.Tensor | None = None
    d_t: torch.Tensor | None = None
    c_dif: torch.Tensor | None = None
    h_eff: torch.Tensor | None = None
    if use_gpu_ogcm and 2 <= nzo <= 64 and T.dtype == torch.float32:
        T, t_comp, S, s_comp, kv, sfc.t_ocean_ml, sfc.m_seaice = dispatch_ogcm_tracer_vdiff_freeze_metal(
            T=T, t_comp=t_comp, S=S, s_comp=s_comp,
            kv_diag=kv_diag, kv_redi=kv_redi, z_c=ogcm.z_c,
            dz_c_int=dz_c_int, h_c_safe=h_c_vdiff, h0_new=h0_vdiff, h0_frz=h_c_safe[0],
            wet_cf=wet_cf, wet_int_cf=wet_int_cf,
            sfc_m_seaice=sfc.m_seaice, sfc_t_ocean_ml=sfc.t_ocean_ml,
            dt=dt, t_ref_eos=T_REF_EOS_K, s_ref_psu=S_REF_PSU, s_seaice_psu=S_SEAICE_PSU,
            eos_a0=EOS_ALPHA0, eos_at=EOS_ALPHA_T, eos_az=EOS_ALPHA_Z, eos_as=EOS_ALPHA_S,
            eos_b0=EOS_BETA0, kappa_v_conv=KAPPA_V_CONV,
        )
        ogcm.kv_last = kv
    else:
        vdiff_prep_fn = maybe_cuda_compile(_ogcm_vdiff_prep_core, device=dev)
        kv, c_dif, h_eff, a_t, b_t, d_t, t_diff, s_diff, rhs_t, rhs_s = vdiff_prep_fn(
            T, S, kv_diag, kv_redi, z_c_2d, dz_c_int, h_c_vdiff, h0_vdiff,
            wet_cf, wet_int_c, wet_int_cf, dt,
        )
        ogcm.kv_last = kv

        nc = T.shape[1]
        dts_imp = solve_tridiagonal_batched(
            torch.cat([a_t, a_t], dim=1),
            torch.cat([b_t, b_t], dim=1),
            torch.cat([d_t, d_t], dim=1),
            torch.cat([rhs_t, rhs_s], dim=1),
        )
        dt_imp, ds_imp = dts_imp[:, :nc], dts_imp[:, nc:]

        vdiff_post_fn = maybe_cuda_compile(_ogcm_vdiff_post_freeze_core, device=dev)
        T, t_comp, S, s_comp, sfc.t_ocean_ml, sfc.m_seaice = vdiff_post_fn(
            T, t_comp, S, s_comp, t_diff, s_diff, dt_imp, ds_imp,
            c_dif, h_eff, h_c_safe, wet_c, wet_cf, is_ocean, is_ocn_f,
            sfc.m_seaice, sfc.t_ocean_ml,
        )
    ogcm.temp = T
    ogcm.sal = S
    ogcm.t_comp = t_comp
    ogcm.s_comp = s_comp

    # 10b. FlexPFT Marine Biogeochemistry & OMIP/mocsy 2.0 Carbonate Chemistry
    if ogcm.bgc is not None:
        from aoba.ocean_bgc import PAR_FRAC_SW, step_ocean_bgc_flexpft

        if a_t is None:
            c_dif = (dt * kv / dz_c_int) * wet_int_cf
            h_eff = torch.cat([h0_vdiff.unsqueeze(0), h_c_vdiff[1:]], dim=0)
            z_row = torch.zeros_like(T[0:1])
            a_t = torch.cat([z_row, -c_dif / h_eff[1:]], dim=0)
            d_t = torch.cat([-c_dif / h_eff[:-1], z_row], dim=0)
            b_t = 1.0 - a_t - d_t

        par_in = (
            sw_par_ocn
            if sw_par_ocn is not None
            else (PAR_FRAC_SW * torch.clamp(sw_net_ocn, min=0.0) if sw_net_ocn is not None else torch.zeros_like(sfc.t_ocean_ml))
        )
        if u10_ocn is not None:
            u10_in = u10_ocn
        else:
            # Fallback 10 m wind speed estimate from surface wind stress magnitude |tau| = rho_a * C_d * u10^2
            tau_mag = torch.sqrt(tau_x_sfc * tau_x_sfc + tau_y_sfc * tau_y_sfc + 1.0e-12)
            u10_in = torch.sqrt(tau_mag / (1.225 * 1.3e-3))
        step_ocean_bgc_flexpft(
            bgc=ogcm.bgc,
            ogcm=ogcm,
            mesh=mesh,
            h_u_edge=h_u_edge,
            sx_int=sx_int,
            w_vel=w_vel,
            h0_new=h0_new,
            dh0_eff=dh0_eff,
            a_t=a_t,
            b_t=b_t,
            d_t=d_t,
            c_dif=c_dif,
            h_eff=h_eff,
            cache=cache,
            sw_par_sfc=par_in,
            u10_ocn=u10_in,
            f_seaice=f_seaice,
            co2_ppmv=co2_ppmv,
            dt=dt,
            enable_full_redi=bool(ENABLE_FULL_REDI),
            enable_o2_omz=bool(enable_o2_omz),
        )

    # 11. Dynamic Sea-Ice Drift (after recording thermodynamic mass change, so advection doesn't alias brine!)
    if enable_seaice_drift:
        if ogcm.seaice_evp is not None:
            from aoba.seaice_evp import step_seaice_evp_itd_voronoi

            step_seaice_evp_itd_voronoi(
                ogcm.seaice_evp, sfc, mesh, tau_x_sfc, tau_y_sfc, u_new[0], ogcm.eta, dt, cache=cache
            )
        else:
            step_seaice_drift_voronoi(sfc, mesh, tau_x_sfc, tau_y_sfc, u_new[0], dt, cache=cache)

    ogcm.m_seaice_prev = sfc.m_seaice.clone()


def diagnose_ocean_spinup_metrics(
    ogcm: OceanOGCMState,
    sfc: SurfaceState,
    mesh: SCVTMesh,
) -> dict[str, float]:
    """Compute float64-accumulated 3D OGCM spin-up diagnostics (T/S by depth, OHC, KE, AMOC, ACC, sea ice).

    Purely passive post-step monitoring diagnostics; does not modify model state or physics.
    """
    from aoba.surface import MC_SEAICE_CRIT

    # Reconstruct 3D cell winds on device, then transfer compact [nzo, nc] arrays to CPU float64
    # for exact global integral accumulation across all backends (CUDA, MPS, CPU).
    u_c_dev, v_c_dev = reconstruct_cell_winds(mesh, ogcm.u_edge * ogcm.wet_e.to(ogcm.u_edge.dtype))
    u64 = u_c_dev.detach().cpu().to(torch.float64)
    v64 = v_c_dev.detach().cpu().to(torch.float64)

    area = mesh.area_cell.detach().cpu().to(torch.float64)
    wet_c = ogcm.wet_c.detach().cpu()
    wet0 = wet_c[0]
    area_ocn = torch.where(wet0, area, torch.zeros_like(area))
    a_ocn_tot = float(area_ocn.sum().item())
    if a_ocn_tot <= 0.0:
        return {}

    h_c64 = ogcm.h_c.detach().cpu().to(torch.float64)
    eta64 = ogcm.eta.detach().cpu().to(torch.float64)
    h_eff = h_c64.clone()
    h_eff[0] = torch.where(
        wet0,
        torch.clamp(h_c64[0] + eta64, min=10.0),
        torch.zeros_like(area),
    )
    vol = h_eff * area.unsqueeze(0)  # [nzo, nc] [m^3]
    v_tot = float(vol.sum().item())

    t64 = ogcm.temp.detach().cpu().to(torch.float64) + ogcm.t_comp.detach().cpu().to(torch.float64)
    s64 = ogcm.sal.detach().cpu().to(torch.float64) + ogcm.s_comp.detach().cpu().to(torch.float64)

    t_global = float((vol * t64).sum().item() / max(v_tot, 1.0))
    s_global = float((vol * s64).sum().item() / max(v_tot, 1.0))
    sst_global = float((area_ocn * t64[0]).sum().item() / a_ocn_tot)
    sss_global = float((area_ocn * s64[0]).sum().item() / a_ocn_tot)

    zc = ogcm.z_c.detach().cpu().to(torch.float64)
    def _band_mean(mask_z: torch.Tensor) -> float:
        if not bool(mask_z.any()):
            return float("nan")
        vb = vol[mask_z]
        vsum = float(vb.sum().item())
        if vsum <= 0.0:
            return float("nan")
        return float((vb * t64[mask_z]).sum().item() / vsum)

    t_upper = _band_mean(zc < 300.0)
    t_mid = _band_mean((zc >= 300.0) & (zc < 1500.0))
    t_deep = _band_mean(zc >= 1500.0)

    ohc_j = float(C.RHO_SEAWATER * C.CP_SEAWATER * (vol * (t64 - C.T_FREEZE)).sum().item())
    ohc_zj = ohc_j * 1.0e-21

    ke_jm2 = float(0.5 * C.RHO_SEAWATER * (vol * (u64 * u64 + v64 * v64)).sum().item() / a_ocn_tot)
    eta_rms = math.sqrt(float((area_ocn * eta64 * eta64).sum().item() / a_ocn_tot))

    if ogcm.h_bl is not None:
        hbl64 = ogcm.h_bl.detach().cpu().to(torch.float64)
        hbl_mean = float((area_ocn * hbl64).sum().item() / a_ocn_tot)
        hbl_max = float(torch.where(wet0, hbl64, torch.zeros_like(hbl64)).max().item())
    else:
        hbl_mean = float("nan")
        hbl_max = float("nan")

    # Hemispheric sea-ice areas [10^6 km^2]
    m_si64 = sfc.m_seaice.detach().cpu().to(torch.float64)
    f_ice = torch.clamp(torch.sqrt(torch.clamp(m_si64, min=0.0) / MC_SEAICE_CRIT), 0.0, 1.0)
    lat_rad = mesh.lat_cell.detach().cpu().to(torch.float64)
    lon_deg = torch.rad2deg(mesh.lon_cell.detach().cpu().to(torch.float64))
    lat_deg = torch.rad2deg(lat_rad)
    ice_nh = float((area_ocn * f_ice * (lat_rad >= 0.0).to(torch.float64)).sum().item() * 1.0e-12)
    ice_sh = float((area_ocn * f_ice * (lat_rad < 0.0).to(torch.float64)).sum().item() * 1.0e-12)

    # Meridional overturning streamfunction (AMOC in Atlantic sector, barotropic-mean removed) [Sv]
    def _amoc_at_lat(lat_c: float, half_w: float = 2.5) -> float:
        lon_hi = -10.0 if lat_c < 35.0 else 20.0
        band = wet0 & (lat_deg >= (lat_c - half_w)) & (lat_deg <= (lat_c + half_w)) & (lon_deg >= -82.0) & (lon_deg <= lon_hi)
        if not bool(band.any()):
            return 0.0
        dlat_m = C.EARTH_RADIUS * math.radians(2.0 * half_w)
        # Transport per layer [m^3 s^-1]: sum_{i in band} v_k * h_k * area_i / dlat_m
        w_sec = area[band].unsqueeze(0) / dlat_m
        t_layer = (v64[:, band] * h_eff[:, band] * w_sec).sum(dim=1)
        a_layer = (h_eff[:, band] * w_sec).sum(dim=1)
        a_tot = float(a_layer.sum().item())
        if a_tot <= 0.0:
            return 0.0
        t_oc = t_layer - (t_layer.sum() / a_tot) * a_layer
        psi_sv = torch.cumsum(t_oc, dim=0) * 1.0e-6
        return float(psi_sv.max().item())

    amoc_26n = _amoc_at_lat(26.5, half_w=2.5)
    amoc_max = max(_amoc_at_lat(lat_c, half_w=2.5) for lat_c in (24.0, 26.5, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0))

    # Antarctic Circumpolar Current (ACC) eastward volume transport through Drake Passage [Sv]
    drake = wet0 & (lon_deg >= -72.0) & (lon_deg <= -58.0) & (lat_deg >= -66.0) & (lat_deg <= -54.0)
    if bool(drake.any()):
        dlon_m = C.EARTH_RADIUS * torch.clamp(torch.cos(lat_rad[drake]), min=0.1) * math.radians(14.0)
        acc_drake = float((u64[:, drake] * h_eff[:, drake] * (area[drake] / dlon_m).unsqueeze(0)).sum().item() * 1.0e-6)
    else:
        acc_drake = 0.0

    return {
        "t_ocn_global_k": t_global,
        "t_ocn_0_300m_k": t_upper,
        "t_ocn_300_1500m_k": t_mid,
        "t_ocn_deep_k": t_deep,
        "sst_global_k": sst_global,
        "s_ocn_global_psu": s_global,
        "sss_global_psu": sss_global,
        "ohc_j": ohc_j,
        "ohc_zj": ohc_zj,
        "ocn_ke_jm2": ke_jm2,
        "ocn_eta_rms_m": eta_rms,
        "ocn_hbl_mean_m": hbl_mean,
        "ocn_hbl_max_m": hbl_max,
        "amoc_26n_sv": amoc_26n,
        "amoc_max_sv": amoc_max,
        "acc_drake_sv": acc_drake,
        "seaice_nh_mkm2": ice_nh,
        "seaice_sh_mkm2": ice_sh,
    }
