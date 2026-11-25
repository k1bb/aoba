"""Map-Free Autonomous Dynamic Lake Model (Topographic Depressions + 2-Layer FLake Thermodynamics).

Represents inland lakes completely without prescribed geographic lake-mask or depth lookup maps:
topographic basin capacities `(f_cap, h_cap, v_cap_m3)` emerge purely from solid-Earth `Earth2014`
bedrock depressions (`SUR2014 > BED2014` on non-ocean, non-ice terrain) combined with valley-floor
Priority-Flood closed depressions (`Barnes, Lehman, & Mulla, 2014`, computed in `river_routing.py`).

Key physical processes:
1. **Dynamic Bowl Hypsometry & Endorheic/Exorheic Transition**:
   Prognostic basin water volume `v_lake_m3` (with float32 + Kahan compensation) maps conservatively
   to lake area fraction `f_lake = f_cap * (V / V_cap)^(2/3)` and mean depth
   `h_lake = h_cap * (V / V_cap)^(1/3)` (`f_lake * area_cell * h_lake == V` identically).
   Under arid climates (`E > P + Q_in`), lakes contract and desiccate smoothly to `f_lake = 0`,
   exposing bare/dust-emitting land; under wet climates, basins fill to `v_cap_m3` and spill excess
   water directly into the Priority-Flood Voronoi river network.
2. **Two-Layer FLake-Style Thermodynamics & 4 C Overturning (Mironov, 2008; Hostetler & Bartlein, 1990)**:
   Prognostic mixed-layer (`t_ml`, `h_ml`) and hypolimnion (`t_bot`) temperatures with Beer-Lambert
   shortwave penetration, exact freshwater density-maximum equation of state
   `rho_f(T) = rho_w * [1 - 6.8e-6 * (T - 277.13)^2]`, instantaneous enthalpy-conserving spring/autumn
   convective overturning, wind-buoyancy mixed-layer entrainment/detrainment, and exact analytical
   exponential thermocline diffusion.
3. **Prognostic Lake Ice, Snow-on-Ice & Albedo Feedback**:
   Implicit surface energy balance over open water and conductive snow+ice cover, exact latent-heat
   freezing/melting, Fresnel open-water vs. ice/snow albedos, and fused single-launch Metal/CUDA GPU
   execution (`dynamic_lake_step_col` in `kernels_esm.py`).

AOBA Formulation Reference: Sec. 7.10, Eqs. (7.47)-(7.52) (Map-Free Dynamic Lakes & 2-Layer Thermodynamics).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.river_routing import compute_priority_flood_depressions
from aoba.surface import SURF_LAND, _kahan_add

# Physical constants for freshwater lakes (Mironov, 2008; Hostetler & Bartlein, 1990)
RHO_W_FRESH: float = 1000.0         # Freshwater density [kg m^-3]
CP_W_FRESH: float = 4186.0          # Freshwater specific heat capacity [J kg^-1 K^-1]
RHO_ICE_LAKE: float = 917.0         # Freshwater lake-ice density [kg m^-3]
CP_ICE_LAKE: float = 2100.0         # Lake-ice specific heat capacity [J kg^-1 K^-1]
K_ICE_LAKE: float = 2.2             # Thermal conductivity of lake ice [W m^-1 K^-1]
K_SNOW_LAKE: float = 0.25           # Thermal conductivity of snow on lake ice [W m^-1 K^-1]
RHO_SNOW_LAKE: float = 300.0        # Bulk density of snow on lake ice [kg m^-3]
EMISS_LAKE: float = 0.985           # Broadband longwave emissivity of lake water/ice [-]
T_MD_FRESH: float = 277.13          # Temperature of maximum freshwater density (3.98 C) [K]
ALPHA_RHO_FRESH: float = 6.8e-6     # Quadratic thermal expansion coefficient around T_md [K^-2]
Z0_LAKE_OPEN: float = 1.0e-4        # Open-water aerodynamic roughness length [m]
Z0_LAKE_ICE: float = 1.0e-3         # Frozen lake-ice aerodynamic roughness length [m]
H_LAKE_MIN_ACTIVE: float = 0.05     # Minimum mean depth for active lake tiling [m]
H_ML_MIN: float = 1.0               # Minimum mixed-layer depth when liquid depth permits [m]
KZ_THERMOCLINE: float = 2.0e-5      # Background thermocline eddy diffusivity [m^2 s^-1]
GAMMA_W_ICE: float = 5.0e-5         # Under-ice turbulent heat transfer velocity [m s^-1]
HYPS_EXP_PF: float = 1.5            # Parabolic bowl volume factor V = (1/p) * A_cap * d_pf [-]
SIGMA_PF_SCALE_M: float = 50.0      # Topographic relief scale for Priority-Flood basin fraction [m]


@dataclass
class DynamicLakeState:
    """Prognostic basin storage, 2-layer thermodynamics, lake ice, and surface diagnostics `[N_cells]`."""

    # Map-free geometric basin capacity (from Earth2014 bedrock + Priority-Flood depressions)
    f_cap: torch.Tensor             # [N_cells] maximum geometric lake area fraction in [0, max_frac] [-]
    h_cap: torch.Tensor             # [N_cells] full-basin mean depth V_cap / (f_cap * A_cell) [m]
    v_cap_m3: torch.Tensor          # [N_cells] maximum basin water-equivalent capacity [m^3]
    d_pf_m: torch.Tensor            # [N_cells] Priority-Flood closed-depression sill depth [m]
    is_basin: torch.Tensor          # [N_cells] bool (True where v_cap_m3 > 0 on land)

    # Prognostic hydrological & thermodynamic state
    v_lake_m3: torch.Tensor         # [N_cells] liquid + lake-ice water-equivalent volume [m^3]
    v_lake_comp: torch.Tensor       # [N_cells] Kahan compensation for v_lake_m3 [m^3]
    t_ml: torch.Tensor              # [N_cells] mixed-layer (epilimnion) temperature [K]
    t_bot: torch.Tensor             # [N_cells] deep-layer (hypolimnion) temperature [K]
    h_ml: torch.Tensor              # [N_cells] mixed-layer depth [m]
    h_ice: torch.Tensor             # [N_cells] lake-ice thickness [m]
    swe_ice: torch.Tensor           # [N_cells] snow water equivalent on lake ice [kg m^-2]
    t_ice: torch.Tensor             # [N_cells] lake-ice/snow skin temperature [K]

    # Diagnosed geometry, fluxes, and hydrology
    f_lake: torch.Tensor            # [N_cells] active lake area fraction of cell [-]
    h_lake: torch.Tensor            # [N_cells] active lake mean depth [m]
    f_ice: torch.Tensor             # [N_cells] frozen indicator (0 or 1 over active lake) [-]
    t_rad: torch.Tensor             # [N_cells] lake radiative skin temperature [K]
    alb_vis_dir: torch.Tensor       # [N_cells] lake visible direct albedo [-]
    alb_vis_dif: torch.Tensor       # [N_cells] lake visible diffuse albedo [-]
    alb_nir_dir: torch.Tensor       # [N_cells] lake NIR direct albedo [-]
    alb_nir_dif: torch.Tensor       # [N_cells] lake NIR diffuse albedo [-]
    shf: torch.Tensor               # [N_cells] lake sensible heat flux [W m^-2]
    lhf: torch.Tensor               # [N_cells] lake latent heat flux [W m^-2]
    evap: torch.Tensor              # [N_cells] lake evaporation/sublimation mass flux [kg m^-2 s^-1]
    tau_x: torch.Tensor             # [N_cells] zonal momentum stress over lake [N m^-2]
    tau_y: torch.Tensor             # [N_cells] meridional momentum stress over lake [N m^-2]
    q_inflow_m3s: torch.Tensor      # [N_cells] river + local runoff inflow into basin [m^3 s^-1]
    q_spill_m3s: torch.Tensor       # [N_cells] sill overflow into downstream river [m^3 s^-1]


def _derive_basin_capacities(
    mesh,
    *,
    max_frac: float = 0.85,
    max_depth_m: float = 400.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive map-free basin capacities `(f_cap, h_cap, v_cap_m3, d_pf_m, is_basin)` from mesh geometry."""
    dev = device if device is not None else mesh.z_topo.device
    is_land = (mesh.surf_type.to(dev) == SURF_LAND)
    is_land_f = is_land.to(dtype)
    area_c = mesh.area_cell.to(device=dev, dtype=dtype)
    sigma_c = torch.clamp(mesh.sigma_topo.to(device=dev, dtype=dtype), min=0.0)

    # 1. Macro-scale closed depressions via Priority-Flood on valley-floor elevation
    z_valley = getattr(mesh, "z_min_land", None)
    if z_valley is None:
        z_valley = mesh.z_topo
    d_pf_np = compute_priority_flood_depressions(z_valley, mesh.surf_type, mesh.cells_on_edge)
    d_pf = torch.as_tensor(d_pf_np, device=dev, dtype=dtype) * is_land_f

    f_pf = float(max_frac) * d_pf / (d_pf + torch.clamp(sigma_c, min=SIGMA_PF_SCALE_M))
    f_pf = torch.where(d_pf > 0.5, torch.clamp(f_pf, 0.0, float(max_frac)), torch.zeros_like(f_pf))
    v_pf = (f_pf * area_c) * torch.clamp(d_pf, min=0.0, max=float(max_depth_m)) / HYPS_EXP_PF

    # 2. Subgrid solid-Earth bedrock depressions from Earth2014 (SUR2014 - BED2014 on non-ocean/ice pixels)
    f_bed_raw = getattr(mesh, "lake_basin_frac_bed", None)
    v_bed_depth = getattr(mesh, "lake_basin_vol_bed", None)
    if f_bed_raw is not None and v_bed_depth is not None:
        f_bed = torch.clamp(f_bed_raw.to(device=dev, dtype=dtype), 0.0, float(max_frac)) * is_land_f
        v_bed = torch.clamp(v_bed_depth.to(device=dev, dtype=dtype), min=0.0) * area_c * is_land_f
    else:
        f_bed = torch.zeros_like(f_pf)
        v_bed = torch.zeros_like(v_pf)

    # 3. Combine bedrock & Priority-Flood basins cleanly; enforce exact V_cap = f_cap * A_cell * h_cap
    f_cap = torch.clamp(torch.maximum(f_bed, f_pf), 0.0, float(max_frac)) * is_land_f
    v_raw = torch.maximum(v_bed, v_pf) * is_land_f
    has_cap = is_land & (f_cap > 1.0e-4) & (v_raw > 1.0)
    f_cap = torch.where(has_cap, f_cap, torch.zeros_like(f_cap))
    h_cap = torch.where(
        has_cap,
        torch.clamp(v_raw / torch.clamp(f_cap * area_c, min=1.0), min=1.0, max=float(max_depth_m)),
        torch.zeros_like(f_cap),
    )
    v_cap = f_cap * area_c * h_cap
    return f_cap, h_cap, v_cap, d_pf, has_cap


def update_lake_hypsometry(lake: DynamicLakeState, area_cell: torch.Tensor) -> None:
    """Update active lake fraction `f_lake` and mean depth `h_lake` from prognostic volume `v_lake_m3`."""
    v_cap_safe = torch.clamp(lake.v_cap_m3, min=1.0)
    phi = torch.where(lake.is_basin, torch.clamp(lake.v_lake_m3 / v_cap_safe, 0.0, 1.0), torch.zeros_like(lake.v_lake_m3))
    # Bowl hypsometry: f_lake = f_cap * phi^(2/3), h_lake = h_cap * phi^(1/3) -> f_lake * A * h_lake == V exactly
    phi_13 = torch.pow(torch.clamp(phi, min=0.0), 1.0 / 3.0)
    f_raw = lake.f_cap * (phi_13 * phi_13)
    h_raw = lake.h_cap * phi_13
    active = lake.is_basin & (h_raw >= H_LAKE_MIN_ACTIVE) & (f_raw > 1.0e-6)
    lake.f_lake = torch.where(active, f_raw, torch.zeros_like(f_raw))
    lake.h_lake = torch.where(active, h_raw, torch.zeros_like(h_raw))
    lake.f_ice = torch.where(active & (lake.h_ice > 1.0e-4), torch.ones_like(f_raw), torch.zeros_like(f_raw))


def init_dynamic_lake_state(
    mesh,
    *,
    init_fill_frac: float = 0.5,
    max_frac: float = 0.85,
    max_depth_m: float = 400.0,
    init_temp_k: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> DynamicLakeState:
    """Initialize map-free dynamic lake state purely from SCVT mesh & Earth2014 bedrock geometry."""
    dev = device if device is not None else mesh.z_topo.device
    f_cap, h_cap, v_cap, d_pf, is_basin = _derive_basin_capacities(
        mesh, max_frac=max_frac, max_depth_m=max_depth_m, device=dev, dtype=dtype
    )
    nc = mesh.n_cells
    z0 = torch.zeros(nc, device=dev, dtype=dtype)
    fill0 = float(min(max(init_fill_frac, 0.0), 1.0))
    v0 = v_cap * fill0

    if init_temp_k is not None:
        t0 = torch.clamp(init_temp_k.to(device=dev, dtype=dtype), min=C.T_FREEZE + 1.0, max=310.0)
    else:
        # Smooth latitude-dependent initial temperature guess (cold poles -> warm tropics, >= 4 C)
        lat_rad = mesh.lat_cell.to(device=dev, dtype=dtype)
        t0 = torch.clamp(277.15 + 22.0 * torch.cos(lat_rad) ** 2, min=C.T_FREEZE + 1.0, max=305.0)

    t_bot0 = torch.clamp(torch.minimum(t0, torch.full_like(t0, 281.15)), min=T_MD_FRESH)
    h_ml0 = torch.clamp(0.4 * h_cap * (fill0 ** (1.0 / 3.0) if fill0 > 0.0 else 0.0), min=H_ML_MIN, max=20.0)

    state = DynamicLakeState(
        f_cap=f_cap,
        h_cap=h_cap,
        v_cap_m3=v_cap,
        d_pf_m=d_pf,
        is_basin=is_basin,
        v_lake_m3=v0,
        v_lake_comp=z0.clone(),
        t_ml=t0.clone(),
        t_bot=t_bot0,
        h_ml=h_ml0,
        h_ice=z0.clone(),
        swe_ice=z0.clone(),
        t_ice=torch.full((nc,), C.T_FREEZE, device=dev, dtype=dtype),
        f_lake=z0.clone(),
        h_lake=z0.clone(),
        f_ice=z0.clone(),
        t_rad=t0.clone(),
        alb_vis_dir=torch.full((nc,), 0.06, device=dev, dtype=dtype),
        alb_vis_dif=torch.full((nc,), 0.06, device=dev, dtype=dtype),
        alb_nir_dir=torch.full((nc,), 0.06, device=dev, dtype=dtype),
        alb_nir_dif=torch.full((nc,), 0.06, device=dev, dtype=dtype),
        shf=z0.clone(),
        lhf=z0.clone(),
        evap=z0.clone(),
        tau_x=z0.clone(),
        tau_y=z0.clone(),
        q_inflow_m3s=z0.clone(),
        q_spill_m3s=z0.clone(),
    )
    update_lake_hypsometry(state, mesh.area_cell.to(device=dev, dtype=dtype))
    return state


def rebuild_lake_basins_from_topography(
    lake: DynamicLakeState,
    mesh,
    *,
    max_frac: float = 0.85,
    max_depth_m: float = 400.0,
) -> torch.Tensor:
    """Recompute topographic basins after dynamic ice-sheet / isostasy updates; return spilled volume [m^3]."""
    dev = lake.v_lake_m3.device
    dtype = lake.v_lake_m3.dtype
    f_cap, h_cap, v_cap, d_pf, is_basin = _derive_basin_capacities(
        mesh, max_frac=max_frac, max_depth_m=max_depth_m, device=dev, dtype=dtype
    )
    spill_m3 = torch.clamp(lake.v_lake_m3 - v_cap, min=0.0)
    lake.f_cap = f_cap
    lake.h_cap = h_cap
    lake.v_cap_m3 = v_cap
    lake.d_pf_m = d_pf
    lake.is_basin = is_basin
    lake.v_lake_m3 = torch.minimum(torch.clamp(lake.v_lake_m3, min=0.0), v_cap)
    lake.v_lake_comp = torch.where(is_basin, lake.v_lake_comp, torch.zeros_like(lake.v_lake_comp))
    update_lake_hypsometry(lake, mesh.area_cell.to(device=dev, dtype=dtype))
    return spill_m3


def compute_lake_surface_albedos(
    lake: DynamicLakeState,
    mu0_cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute lake visible/NIR direct & diffuse albedos and radiative skin temperature `[N_cells]`.

    Uses Briegleb & Ramanathan (1982) zenith-dependent Fresnel albedo for open freshwater and
    smooth snow-on-lake-ice albedo transition when frozen (`h_ice > 1e-4 m`).
    """
    mu_safe = torch.clamp(mu0_cell, min=0.03, max=1.0)
    # Open freshwater Fresnel direct albedo (Eq. 9.1) + diffuse albedo 0.06
    alb_open_dir = torch.clamp(0.026 / (torch.pow(mu_safe, 1.7) + 0.065) + 0.15 * (mu_safe - 0.1) * (mu_safe - 0.5) * (mu_safe - 1.0), 0.04, 0.45)
    alb_open_dif = torch.full_like(alb_open_dir, 0.06)

    # Bare lake-ice albedo (0.55 VIS / 0.30 NIR) blended smoothly toward snow-on-ice (0.82 VIS / 0.52 NIR)
    f_snow_ice = lake.swe_ice / (lake.swe_ice + 20.0)
    alb_ice_vis = 0.55 + (0.82 - 0.55) * f_snow_ice
    alb_ice_nir = 0.30 + (0.52 - 0.30) * f_snow_ice

    is_frozen = (lake.h_ice > 1.0e-4) & (lake.f_lake > 0.0)
    alb_vd = torch.where(is_frozen, alb_ice_vis, alb_open_dir)
    alb_vf = torch.where(is_frozen, alb_ice_vis, alb_open_dif)
    alb_nd = torch.where(is_frozen, alb_ice_nir, alb_open_dir)
    alb_nf = torch.where(is_frozen, alb_ice_nir, alb_open_dif)
    t_rad = torch.where(is_frozen, torch.clamp(lake.t_ice, min=180.0, max=C.T_FREEZE), torch.clamp(lake.t_ml, min=C.T_FREEZE, max=330.0))

    lake.alb_vis_dir = alb_vd
    lake.alb_vis_dif = alb_vf
    lake.alb_nir_dir = alb_nd
    lake.alb_nir_dif = alb_nf
    lake.t_rad = t_rad
    return alb_vd, alb_vf, alb_nd, alb_nf, t_rad


def _qsat_and_dqsat(temp_k: torch.Tensor, pres_pa: torch.Tensor, *, over_ice: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnus-Tetens saturation specific humidity and d(qsat)/dT over freshwater or ice."""
    t_c = torch.clamp(temp_k - C.T_FREEZE, min=-90.0, max=60.0)
    if over_ice:
        a, b = 21.875, 265.5
        lat_h = C.L_S0
    else:
        a, b = 17.269, 237.3
        lat_h = C.L_V0
    es = 610.94 * torch.exp(a * t_c / (t_c + b))
    p_eff = torch.clamp(pres_pa - (1.0 - C.EPS_WATER) * es, min=100.0)
    qs = torch.clamp(C.EPS_WATER * es / p_eff, min=1.0e-8, max=0.5)
    dqs_dt = qs * lat_h / (C.R_V * torch.clamp(temp_k, min=180.0) ** 2)
    return qs, dqs_dt


def _step_dynamic_lakes_py(
    lake: DynamicLakeState,
    u_1: torch.Tensor,
    v_1: torch.Tensor,
    temp_1: torch.Tensor,
    qv_1: torch.Tensor,
    pres_sfc: torch.Tensor,
    rho_1: torch.Tensor,
    z_ref_rel: torch.Tensor,
    sw1_dir_dn: torch.Tensor,
    sw1_dif_dn: torch.Tensor,
    sw2_dir_dn: torch.Tensor,
    sw2_dif_dn: torch.Tensor,
    lw_dn: torch.Tensor,
    precip_rain: torch.Tensor,
    precip_snow: torch.Tensor,
    mu0_cell: torch.Tensor,
    area_cell: torch.Tensor,
    f_cor_cell: torch.Tensor,
    dt: float,
    *,
    k_ext_m1: float = 0.5,
) -> None:
    """Vectorized PyTorch reference implementation of AOBA's map-free 2-layer dynamic lake step."""
    # 1. Refresh hypsometry & albedos from current basin volume (including any river inflow from previous step)
    update_lake_hypsometry(lake, area_cell)
    alb_vd, alb_vf, alb_nd, alb_nf, _ = compute_lake_surface_albedos(lake, mu0_cell)

    f_lake = lake.f_lake
    h_lake = lake.h_lake
    active = (f_lake > 1.0e-6) & (h_lake >= H_LAKE_MIN_ACTIVE) & lake.is_basin
    active_f = active.to(temp_1.dtype)
    area_lake = torch.clamp(f_lake * area_cell, min=1.0)

    # 2. Absorbed broadband shortwave [W m^-2]
    sw_abs = (
        sw1_dir_dn * (1.0 - alb_vd)
        + sw1_dif_dn * (1.0 - alb_vf)
        + sw2_dir_dn * (1.0 - alb_nd)
        + sw2_dif_dn * (1.0 - alb_nf)
    )
    sw_abs = torch.clamp(sw_abs, min=0.0)

    # 3. Bulk aerodynamic transfer coefficient over open water vs. frozen ice
    is_frozen = active & (lake.h_ice > 1.0e-4)
    z0 = torch.where(is_frozen, torch.full_like(z_ref_rel, Z0_LAKE_ICE), torch.full_like(z_ref_rel, Z0_LAKE_OPEN))
    z_r = torch.clamp(z_ref_rel, min=2.0)
    cd_neut = (C.VON_KARMAN / torch.log(z_r / z0)) ** 2
    u_eff = torch.sqrt(u_1 * u_1 + v_1 * v_1 + 0.25)
    t_skin_prev = torch.where(is_frozen, lake.t_ice, lake.t_ml)
    ri_b = C.GRAVITY * z_r * (temp_1 - t_skin_prev) / (torch.clamp(temp_1, min=180.0) * u_eff * u_eff)
    stab = torch.where(
        ri_b >= 0.0,
        1.0 / (1.0 + 5.0 * torch.clamp(ri_b, max=2.0)),
        torch.clamp(1.0 - 2.0 * torch.clamp(ri_b, min=-2.0), max=2.0),
    )
    cd = torch.clamp(cd_neut * stab, min=2.0e-4, max=5.0e-3)
    ga = torch.clamp(rho_1, min=0.1) * cd * u_eff
    tau_x = -ga * u_1 * active_f
    tau_y = -ga * v_1 * active_f

    # 4. Liquid column geometry before thermodynamics
    ice_eq_h = (RHO_ICE_LAKE / RHO_W_FRESH) * lake.h_ice
    h_liq = torch.clamp(h_lake - ice_eq_h, min=0.0)
    h_ml = torch.clamp(lake.h_ml, min=torch.minimum(torch.full_like(h_liq, H_ML_MIN), h_liq), max=torch.clamp(h_liq, min=0.05))
    h_bot = torch.clamp(h_liq - h_ml, min=0.0)

    # --- Branch A: Open water (active & not frozen) ---
    qs_w, dqs_w = _qsat_and_dqsat(lake.t_ml, pres_sfc, over_ice=False)
    f_sw_ml = torch.where(h_bot > 1.0e-3, -torch.expm1(-float(k_ext_m1) * h_ml), torch.ones_like(h_ml))
    sw_ml = f_sw_ml * sw_abs
    sw_bot = (1.0 - f_sw_ml) * sw_abs

    t0_w = lake.t_ml
    f_lw_up_w = EMISS_LAKE * C.STEFAN_BOLTZMANN * (t0_w ** 4)
    f_sh_w0 = C.CP_D * ga * (t0_w - temp_1)
    f_lh_w0 = C.L_V0 * ga * (qs_w - qv_1)
    # Snow falling onto open water melts immediately, absorbing latent heat of fusion
    f_net_w0 = sw_ml + EMISS_LAKE * lw_dn - f_lw_up_w - f_sh_w0 - f_lh_w0 - C.L_F0 * torch.clamp(precip_snow, min=0.0)
    df_dt_w = 4.0 * EMISS_LAKE * C.STEFAN_BOLTZMANN * (t0_w ** 3) + C.CP_D * ga + C.L_V0 * ga * dqs_w
    c_ml = RHO_W_FRESH * CP_W_FRESH * torch.clamp(h_ml, min=H_LAKE_MIN_ACTIVE)
    dt_ml_w = (f_net_w0 * dt) / (c_ml + dt * df_dt_w)
    t_ml_w = t0_w + dt_ml_w

    shf_w = C.CP_D * ga * (t_ml_w - temp_1)
    evap_w_raw = ga * (qs_w + dqs_w * dt_ml_w - qv_1)
    # Strictly cap evaporation at available water so basin volume never becomes negative
    evap_w_max = (RHO_W_FRESH * h_lake) / max(dt, 1.0) + torch.clamp(precip_rain + precip_snow, min=0.0)
    evap_w = torch.minimum(evap_w_raw, evap_w_max)
    lhf_w = C.L_V0 * evap_w
    dv_w_m3 = ((torch.clamp(precip_rain + precip_snow, min=0.0) - evap_w) * area_lake * dt) / RHO_W_FRESH

    # Bottom-layer Beer-Lambert SW heating
    t_bot_w = torch.where(
        h_bot > 1.0e-3,
        lake.t_bot + (sw_bot * dt) / (RHO_W_FRESH * CP_W_FRESH * torch.clamp(h_bot, min=1.0e-3)),
        t_ml_w,
    )

    # Surface freezing when open-water mixed layer cools below T_freeze
    freeze_def = torch.clamp(C.T_FREEZE - t_ml_w, min=0.0)
    dh_ice_new = torch.minimum((c_ml * freeze_def) / (RHO_ICE_LAKE * C.L_F0), (RHO_W_FRESH / RHO_ICE_LAKE) * h_lake)
    t_ml_w = torch.clamp(t_ml_w, min=C.T_FREEZE, max=330.0)
    t_bot_w = torch.clamp(t_bot_w, min=C.T_FREEZE, max=330.0)

    # --- Branch B: Ice-covered lake (active & frozen) ---
    swe_i = lake.swe_ice + torch.clamp(precip_snow, min=0.0) * dt
    h_ice_i = lake.h_ice
    h_snow_i = swe_i / RHO_SNOW_LAKE
    k_eff = 1.0 / (torch.clamp(h_ice_i, min=1.0e-3) / K_ICE_LAKE + h_snow_i / K_SNOW_LAKE)

    t0_i = torch.clamp(lake.t_ice, min=180.0, max=C.T_FREEZE)
    qs_i, dqs_i = _qsat_and_dqsat(t0_i, pres_sfc, over_ice=True)
    f_net_i0 = (
        sw_abs
        + EMISS_LAKE * lw_dn
        - EMISS_LAKE * C.STEFAN_BOLTZMANN * (t0_i ** 4)
        - C.CP_D * ga * (t0_i - temp_1)
        - C.L_S0 * ga * (qs_i - qv_1)
        + k_eff * (C.T_FREEZE - t0_i)
    )
    df_dt_i = 4.0 * EMISS_LAKE * C.STEFAN_BOLTZMANN * (t0_i ** 3) + C.CP_D * ga + C.L_S0 * ga * dqs_i + k_eff
    c_skin_i = RHO_ICE_LAKE * CP_ICE_LAKE * torch.clamp(h_ice_i, min=0.01, max=0.10)
    dt_ice = (f_net_i0 * dt) / (c_skin_i + dt * df_dt_i)
    t_ice_star = t0_i + dt_ice

    # Excess surface energy above T_freeze melts snow first, then lake ice
    e_melt_sfc = torch.clamp(t_ice_star - C.T_FREEZE, min=0.0) * (c_skin_i + dt * df_dt_i)
    t_ice_new = torch.clamp(t_ice_star, min=180.0, max=C.T_FREEZE)

    shf_i = C.CP_D * ga * (t_ice_new - temp_1)
    evap_i_raw = ga * (qs_i + dqs_i * (t_ice_new - t0_i) - qv_1)
    # Clamp sublimation to available snow + ice mass (deposition when evap_i_raw < 0 adds to snow)
    avail_si_kgm2 = swe_i + RHO_ICE_LAKE * h_ice_i
    evap_i = torch.minimum(evap_i_raw, avail_si_kgm2 / max(dt, 1.0))
    lhf_i = C.L_S0 * evap_i

    sub_kgm2 = evap_i * dt
    # Deposition (<0) goes to swe; sublimation (>0) removes snow first then ice
    dm_snow_sub = torch.where(sub_kgm2 >= 0.0, torch.minimum(swe_i, sub_kgm2), sub_kgm2)
    swe_i = torch.clamp(swe_i - dm_snow_sub, min=0.0)
    dm_ice_sub = torch.clamp(sub_kgm2 - dm_snow_sub, min=0.0)
    dh_ice_sub = torch.minimum(h_ice_i, dm_ice_sub / RHO_ICE_LAKE)
    h_ice_i = torch.clamp(h_ice_i - dh_ice_sub, min=0.0)

    # Surface melting: snow melts into liquid lake volume (v_lake_m3); ice melts in-place (constant v_lake_m3)
    dm_snow_melt = torch.minimum(swe_i, e_melt_sfc / C.L_F0)
    swe_i = torch.clamp(swe_i - dm_snow_melt, min=0.0)
    e_rem = torch.clamp(e_melt_sfc - dm_snow_melt * C.L_F0, min=0.0)
    dh_ice_melt_sfc = torch.minimum(h_ice_i, e_rem / (RHO_ICE_LAKE * C.L_F0))
    h_ice_i = torch.clamp(h_ice_i - dh_ice_melt_sfc, min=0.0)

    # Basal conductive freezing minus under-ice warm water melting (exact exponential cooling of T_ml -> T_f)
    f_cond_base = k_eff * (C.T_FREEZE - t_ice_new)
    r_under = (GAMMA_W_ICE / torch.clamp(h_ml, min=H_LAKE_MIN_ACTIVE)) * dt
    fac_under = -torch.expm1(-torch.clamp(r_under, min=0.0, max=40.0))
    dt_ml_under = torch.clamp(lake.t_ml - C.T_FREEZE, min=0.0) * fac_under
    e_under = c_ml * dt_ml_under
    t_ml_i = torch.clamp(lake.t_ml - dt_ml_under, min=C.T_FREEZE)
    dh_ice_base = (f_cond_base * dt - e_under) / (RHO_ICE_LAKE * C.L_F0)
    max_ice_h = (RHO_W_FRESH / RHO_ICE_LAKE) * h_lake
    h_ice_i = torch.minimum(torch.clamp(h_ice_i + dh_ice_base, min=0.0), max_ice_h)

    # If all lake ice melted, dump any residual snow on ice into liquid lake volume cleanly
    all_melted = h_ice_i <= 1.0e-4
    dm_snow_dump = torch.where(all_melted, swe_i, torch.zeros_like(swe_i))
    swe_i = torch.where(all_melted, torch.zeros_like(swe_i), swe_i)
    h_ice_i = torch.where(all_melted, torch.zeros_like(h_ice_i), h_ice_i)

    # Net liquid+ice water-equivalent volume change on ice-covered lake: rain + melted/dumped snow - ice sublimation
    dv_i_m3 = ((torch.clamp(precip_rain, min=0.0) * dt + dm_snow_melt + dm_snow_dump - dh_ice_sub * RHO_ICE_LAKE) * area_lake) / RHO_W_FRESH

    # --- Combine Open-Water and Ice-Covered Branches ---
    dv_m3 = torch.where(active, torch.where(is_frozen, dv_i_m3, dv_w_m3), torch.zeros_like(dv_w_m3))
    v_new, v_comp_new = _kahan_add(lake.v_lake_m3, dv_m3, lake.v_lake_comp)
    lake.v_lake_m3 = torch.where(lake.is_basin, torch.clamp(v_new, min=0.0), torch.zeros_like(v_new))
    lake.v_lake_comp = torch.where(lake.is_basin, v_comp_new, torch.zeros_like(v_comp_new))

    t_ml_next = torch.where(active, torch.where(is_frozen, t_ml_i, t_ml_w), torch.clamp(temp_1, min=C.T_FREEZE, max=320.0))
    t_bot_next = torch.where(active, torch.where(is_frozen, lake.t_bot, t_bot_w), t_ml_next)
    h_ice_next = torch.where(active, torch.where(is_frozen, h_ice_i, dh_ice_new), torch.zeros_like(h_ice_i))
    swe_next = torch.where(active, torch.where(is_frozen, swe_i, torch.zeros_like(swe_i)), torch.zeros_like(swe_i))
    t_ice_next = torch.where(active, torch.where(is_frozen, t_ice_new, torch.full_like(t_ice_new, C.T_FREEZE)), torch.clamp(temp_1, min=180.0, max=C.T_FREEZE))

    lake.shf = torch.where(active, torch.where(is_frozen, shf_i, shf_w), torch.zeros_like(shf_w))
    lake.lhf = torch.where(active, torch.where(is_frozen, lhf_i, lhf_w), torch.zeros_like(lhf_w))
    lake.evap = torch.where(active, torch.where(is_frozen, evap_i, evap_w), torch.zeros_like(evap_w))
    lake.tau_x = tau_x
    lake.tau_y = tau_y

    # 5. Exact 2-layer vertical thermodynamics: 4 C freshwater convective overturning + entrainment + diffusion
    update_lake_hypsometry(lake, area_cell)
    active2 = (lake.f_lake > 1.0e-6) & (lake.h_lake >= H_LAKE_MIN_ACTIVE) & lake.is_basin
    # Clean desiccation: if lake contracted below active threshold, zero out ice/snow cleanly
    h_ice_next = torch.where(active2, torch.minimum(h_ice_next, (RHO_W_FRESH / RHO_ICE_LAKE) * lake.h_lake), torch.zeros_like(h_ice_next))
    swe_next = torch.where(active2, swe_next, torch.zeros_like(swe_next))

    h_liq2 = torch.clamp(lake.h_lake - (RHO_ICE_LAKE / RHO_W_FRESH) * h_ice_next, min=0.0)
    h_ml_min2 = torch.minimum(torch.full_like(h_liq2, H_ML_MIN), h_liq2)
    h_ml2 = torch.clamp(h_ml, min=h_ml_min2, max=h_liq2)
    h_bot2 = torch.clamp(h_liq2 - h_ml2, min=0.0)

    # Freshwater quadratic EOS around T_md = 277.13 K (3.98 C)
    rho_ml = RHO_W_FRESH * (1.0 - ALPHA_RHO_FRESH * (t_ml_next - T_MD_FRESH) ** 2)
    rho_bot = RHO_W_FRESH * (1.0 - ALPHA_RHO_FRESH * (t_bot_next - T_MD_FRESH) ** 2)

    # 5a. Instantaneous enthalpy-conserving convective overturning whenever surface water is denser
    overturn = active2 & (h_bot2 > 1.0e-3) & (rho_ml >= rho_bot)
    t_mix = (h_ml2 * t_ml_next + h_bot2 * t_bot_next) / torch.clamp(h_liq2, min=1.0e-3)
    t_ml2 = torch.where(overturn, t_mix, t_ml_next)
    t_bot2 = torch.where(overturn, t_mix, t_bot_next)
    h_ml2 = torch.where(overturn, h_liq2, h_ml2)
    h_bot2 = torch.clamp(h_liq2 - h_ml2, min=0.0)

    # 5b. Wind-driven mixed-layer entrainment / detrainment (exact enthalpy conservation)
    strat = active2 & (h_liq2 > H_ML_MIN + 1.0e-3) & (~overturn) & (h_ice_next <= 1.0e-4)
    drho = torch.clamp(rho_bot - rho_ml, min=0.0)
    n_bv = torch.sqrt((C.GRAVITY / RHO_W_FRESH) * drho / torch.clamp(h_liq2, min=1.0))
    u_star_w = torch.sqrt(torch.clamp(rho_1, min=0.1) / RHO_W_FRESH) * torch.sqrt(cd) * u_eff
    h_eq = torch.clamp((0.25 * u_star_w) / (torch.abs(f_cor_cell) + 0.15 * n_bv + 1.0e-4), min=h_ml_min2, max=h_liq2)
    tau_mix = C.SECONDS_PER_DAY
    rel_mix = -math.expm1(-float(dt) / tau_mix)
    h_ml_new = torch.where(strat, h_ml2 + (h_eq - h_ml2) * rel_mix, h_ml2)
    h_ml_new = torch.clamp(h_ml_new, min=h_ml_min2, max=h_liq2)
    dh_ml = h_ml_new - h_ml2
    h_bot_new = torch.clamp(h_liq2 - h_ml_new, min=0.0)

    # Entrainment (dh > 0) mixes bottom water into ML; detrainment (dh < 0) moves ML water into bottom
    t_ml_ent = (h_ml2 * t_ml2 + torch.clamp(dh_ml, min=0.0) * t_bot2) / torch.clamp(h_ml_new, min=1.0e-3)
    t_bot_det = (h_bot2 * t_bot2 + torch.clamp(-dh_ml, min=0.0) * t_ml2) / torch.clamp(h_bot_new, min=1.0e-3)
    t_ml2 = torch.where(strat & (dh_ml > 0.0), t_ml_ent, t_ml2)
    t_bot2 = torch.where(strat & (dh_ml < 0.0) & (h_bot_new > 1.0e-3), t_bot_det, t_bot2)
    h_ml2 = h_ml_new
    h_bot2 = h_bot_new

    # 5c. Exact analytical 2-layer exponential thermocline diffusion (exact heat conservation)
    has_two = active2 & (h_ml2 > 1.0e-3) & (h_bot2 > 1.0e-3)
    h_harm = (h_ml2 * h_bot2) / torch.clamp(h_liq2, min=1.0e-3)
    rate_diff = KZ_THERMOCLINE / torch.clamp(h_harm * 0.5 * h_liq2, min=1.0e-2)
    f_diff = -torch.expm1(-torch.clamp(rate_diff * dt, min=0.0, max=40.0))
    d_t = t_ml2 - t_bot2
    t_ml_fin = torch.where(has_two, t_ml2 - d_t * f_diff * (h_bot2 / torch.clamp(h_liq2, min=1.0e-3)), t_ml2)
    t_bot_fin = torch.where(has_two, t_bot2 + d_t * f_diff * (h_ml2 / torch.clamp(h_liq2, min=1.0e-3)), t_ml_fin)

    lake.t_ml = torch.clamp(t_ml_fin, min=C.T_FREEZE, max=330.0)
    lake.t_bot = torch.clamp(t_bot_fin, min=C.T_FREEZE, max=330.0)
    lake.h_ml = h_ml2
    lake.h_ice = h_ice_next
    lake.swe_ice = swe_next
    lake.t_ice = t_ice_next
    lake.f_ice = torch.where(active2 & (h_ice_next > 1.0e-4), torch.ones_like(h_ice_next), torch.zeros_like(h_ice_next))
    lake.t_rad = torch.where(lake.f_ice > 0.5, lake.t_ice, lake.t_ml)


def step_dynamic_lakes(
    lake: DynamicLakeState,
    u_1: torch.Tensor,
    v_1: torch.Tensor,
    temp_1: torch.Tensor,
    qv_1: torch.Tensor,
    pres_sfc: torch.Tensor,
    rho_1: torch.Tensor,
    z_ref_rel: torch.Tensor,
    sw1_dir_dn: torch.Tensor,
    sw1_dif_dn: torch.Tensor,
    sw2_dir_dn: torch.Tensor,
    sw2_dif_dn: torch.Tensor,
    lw_dn: torch.Tensor,
    precip_rain: torch.Tensor,
    precip_snow: torch.Tensor,
    mu0_cell: torch.Tensor,
    area_cell: torch.Tensor,
    f_cor_cell: torch.Tensor,
    dt: float,
    *,
    k_ext_m1: float = 0.5,
) -> None:
    """Advance map-free dynamic lakes by `dt` [s] (dispatches fused Metal/CUDA kernel on GPU)."""
    if temp_1.device.type in ("mps", "cuda") and temp_1.dtype == torch.float32:
        from aoba.kernels import can_use_elem_metal, dispatch_dynamic_lake_step_metal

        if can_use_elem_metal(temp_1):
            dispatch_dynamic_lake_step_metal(
                lake=lake,
                u_1=u_1,
                v_1=v_1,
                temp_1=temp_1,
                qv_1=qv_1,
                pres_sfc=pres_sfc,
                rho_1=rho_1,
                z_ref_rel=z_ref_rel,
                sw1_dir_dn=sw1_dir_dn,
                sw1_dif_dn=sw1_dif_dn,
                sw2_dir_dn=sw2_dir_dn,
                sw2_dif_dn=sw2_dif_dn,
                lw_dn=lw_dn,
                precip_rain=precip_rain,
                precip_snow=precip_snow,
                mu0_cell=mu0_cell,
                area_cell=area_cell,
                f_cor_cell=f_cor_cell,
                dt=dt,
                k_ext_m1=k_ext_m1,
            )
            return

    _step_dynamic_lakes_py(
        lake=lake,
        u_1=u_1,
        v_1=v_1,
        temp_1=temp_1,
        qv_1=qv_1,
        pres_sfc=pres_sfc,
        rho_1=rho_1,
        z_ref_rel=z_ref_rel,
        sw1_dir_dn=sw1_dir_dn,
        sw1_dif_dn=sw1_dif_dn,
        sw2_dir_dn=sw2_dir_dn,
        sw2_dif_dn=sw2_dif_dn,
        lw_dn=lw_dn,
        precip_rain=precip_rain,
        precip_snow=precip_snow,
        mu0_cell=mu0_cell,
        area_cell=area_cell,
        f_cor_cell=f_cor_cell,
        dt=dt,
        k_ext_m1=k_ext_m1,
    )


def diagnose_lake_globals(lake: DynamicLakeState, area_cell: torch.Tensor) -> tuple[float, float, float]:
    """Compute float64 global `(lake_area_mkm2, lake_vol_1e3km3, lake_ice_area_mkm2)` (lazy diagnostic)."""
    a64 = area_cell.detach().cpu().to(torch.float64)
    fl64 = lake.f_lake.detach().cpu().to(torch.float64)
    fi64 = lake.f_ice.detach().cpu().to(torch.float64)
    vl64 = lake.v_lake_m3.detach().cpu().to(torch.float64)
    area_mkm2 = float((fl64 * a64).sum().item() * 1.0e-12)
    vol_1e3km3 = float(vl64.sum().item() * 1.0e-12)
    ice_mkm2 = float((fl64 * fi64 * a64).sum().item() * 1.0e-12)
    return area_mkm2, vol_1e3km3, ice_mkm2
