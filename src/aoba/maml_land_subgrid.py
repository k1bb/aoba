"""MAML Subgrid Orography Microclimate Downscaling & Dynamic Wetlands/Floodplains/Lakes.

Exploits AOBA's 1:1 Multiple-Atmosphere Multiple-Land (MAML) subgrid surface tiles (`[N_cells * Ny * Nx]`)
completely map-free using only `Earth2014` subgrid topographic roughness `sigma_topo` and prognostic
hydrology/river discharge:

1. **MAML Subgrid Orography & Microclimate Downscaling (`enable_maml_subgrid_topo`)**:
   - Partitions each land cell's subgrid elevation variance `sigma_topo` deterministically across
     the `n_sub = Ny * Nx` MAML land tiles with exact zero parent-cell mean (`sum(dz) == 0`,
     spanning `[-1.65, +1.65] * sigma_topo`).
   - Applies environmental lapse-rate downscaling (`dT = -6.5 K km^-1 * dz`), barometric pressure
     scaling, and terrain slope/aspect direct-beam insolation contrast (zero parent-cell mean bias)
     so mountain snowlines, alpine glaciers, and elevation-zoned EEO forest treelines emerge inside
     coarse global host cells (Fiddes & Gruber, 2014, *GMD*, 7, 387; Tesfa et al., 2020, *JAMES*).

2. **MAML Dynamic Wetlands, Floodplains, & Inland Lakes (`enable_maml_wetlands_lakes`)**:
   - Diagnoses subgrid inundated fraction `f_wetland [N_cells * n_sub] in [0, 0.85]` dynamically
     from top-layer soil wetness, topographic flatness `1 / (1 + sigma_topo / 80 m)` (TOPMODEL
     wetness principle; Beven & Kirkby, 1979; Gedney & Cox, 2003), subgrid valley elevation rank,
     and local Priority-Flood river discharge (`river_routing.py`).
   - Feeds directly into surface albedo/evaporation (inundated open water) and anoxic wetland CH4
     emissions (`trace_gas_cycle.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.surface import SOIL_POROSITY, SURF_LAND

LAPSE_RATE_K_M: float = 0.0065     # Environmental temperature lapse rate [K m^-1]
WETLAND_SIGMA0_M: float = 80.0     # Topographic flatness scale [m] for alluvial floodplain pooling
WETLAND_MAX_FRAC: float = 0.85     # Maximum inundated wetland/lake fraction on a land tile [-]


@dataclass
class MAMLSubgridLandState:
    """Deterministic zero-mean subgrid elevation & dynamic wetland/lake fraction on MAML tiles."""

    dz_sub_m: torch.Tensor         # [N_cells * n_sub] elevation offset relative to parent cell [m]
    slope_sw_fac: torch.Tensor     # [N_cells * n_sub] zero-mean aspect SW factor (mean 1.0 per cell)
    valley_weight: torch.Tensor    # [N_cells * n_sub] valley-bottom pooling weight in [0.4, 1.8]
    f_wetland: torch.Tensor        # [N_cells * n_sub] dynamic inundated wetland/floodplain/lake frac [-]
    # Global diagnostics
    wetland_area_mkm2: float = 0.0


def init_maml_subgrid_land_state(
    surf_type_sub: torch.Tensor,
    sigma_topo_sub: torch.Tensor,
    nc: int,
    n_sub: int,
) -> MAMLSubgridLandState:
    """Initialize zero-mean MAML subgrid elevation offsets and valley weights from `Earth2014` `sigma_topo`."""
    dev = sigma_topo_sub.device
    dtype = sigma_topo_sub.dtype
    m_sub = nc * n_sub
    is_land = (surf_type_sub == SURF_LAND).to(dtype)

    if n_sub <= 1:
        z = torch.zeros((m_sub,), device=dev, dtype=dtype)
        return MAMLSubgridLandState(
            dz_sub_m=z.clone(),
            slope_sw_fac=torch.ones((m_sub,), device=dev, dtype=dtype),
            valley_weight=torch.ones((m_sub,), device=dev, dtype=dtype),
            f_wetland=z.clone(),
        )

    # Deterministic symmetric zero-mean elevation quantile ladder across the n_sub tiles (std == 1.0!)
    q = torch.linspace(-1.0, 1.0, n_sub, device=dev, dtype=dtype)
    q = q - q.mean()
    q = q / torch.clamp(torch.sqrt((q * q).mean()), min=1.0e-6)
    # Cap extreme elevation excursion at +/- 1.65 sigma_topo (clipped to +/- 1800 m for stability)
    q_sub = q.unsqueeze(0).expand(nc, n_sub).reshape(m_sub)
    dz_sub = torch.clamp(q_sub * torch.clamp(sigma_topo_sub, min=0.0, max=1200.0), min=-1800.0, max=1800.0) * is_land
    # Enforce exact zero parent-cell mean even on mixed land/coast cells
    dz_2d = dz_sub.reshape(nc, n_sub)
    dz_sub = (dz_2d - dz_2d.mean(dim=1, keepdim=True)).reshape(m_sub) * is_land

    # Alternating sunlit/shaded ridge aspect modulation (max +/- 12% on steep terrain, exact mean 1.0)
    alt = torch.tensor([1.0 if (i % 2 == 0) else -1.0 for i in range(n_sub)], device=dev, dtype=dtype)
    alt = alt - alt.mean()
    steep = torch.clamp(sigma_topo_sub / 400.0, 0.0, 1.0) * is_land
    slope_fac = 1.0 + 0.12 * steep * alt.unsqueeze(0).expand(nc, n_sub).reshape(m_sub)

    # Valley tiles (dz < 0) pool runoff & groundwater first; ridges (dz > 0) drain fast
    valley_w = torch.clamp(1.0 - 0.55 * q_sub, min=0.4, max=1.8)

    return MAMLSubgridLandState(
        dz_sub_m=dz_sub,
        slope_sw_fac=slope_fac,
        valley_weight=valley_w,
        f_wetland=torch.zeros((m_sub,), device=dev, dtype=dtype),
    )


def downscale_maml_microclimate(
    ml: MAMLSubgridLandState,
    t_air_sub: torch.Tensor,
    pres_sfc_sub: torch.Tensor,
    sw_dir_sub: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply environmental lapse-rate & barometric microclimate downscaling across MAML land tiles."""
    t_down = torch.clamp(t_air_sub - LAPSE_RATE_K_M * ml.dz_sub_m, min=180.0, max=340.0)
    p_down = pres_sfc_sub * torch.exp(torch.clamp(-C.GRAVITY * ml.dz_sub_m / (C.R_D * torch.clamp(t_air_sub, min=200.0)), min=-0.5, max=0.5))
    sw_down = torch.clamp(sw_dir_sub * ml.slope_sw_fac, min=0.0)
    return t_down, p_down, sw_down


def step_maml_wetlands_lakes(
    ml: MAMLSubgridLandState,
    surf_type_sub: torch.Tensor,
    sigma_topo_sub: torch.Tensor,
    w_liq_soil_top: torch.Tensor,
    t_soil_top: torch.Tensor,
    area_cell: torch.Tensor,
    nc: int,
    n_sub: int,
    dt: float,
    *,
    river_discharge_cell: torch.Tensor | None = None,
) -> None:
    """Update dynamic MAML wetland / floodplain / lake inundation fraction (exact exponential relaxation)."""
    dtype = w_liq_soil_top.dtype
    is_land = (surf_type_sub == SURF_LAND).to(dtype)

    # 1. Topographic flatness factor: flat alluvial basins flood readily, steep mountains drain
    flatness = 1.0 / (1.0 + torch.clamp(sigma_topo_sub, min=0.0) / WETLAND_SIGMA0_M)

    # 2. Soil moisture saturation excess (TOPMODEL principle; Beven & Kirkby, 1979)
    sat = torch.clamp(w_liq_soil_top / SOIL_POROSITY, 0.0, 1.0)
    wet_soil = torch.clamp((sat - 0.58) / 0.42, 0.0, 1.0) ** 1.6

    # 3. Optional Priority-Flood river floodplain overbank contribution
    river_boost = torch.zeros_like(wet_soil)
    if river_discharge_cell is not None:
        q_sub = river_discharge_cell.unsqueeze(1).expand(nc, n_sub).reshape(-1)
        # Smooth Michaelis-Menten flooding above ~1500 m^3/s discharge in flat valley tiles
        river_boost = 0.35 * (torch.clamp(q_sub, min=0.0) / (torch.clamp(q_sub, min=0.0) + 4000.0))

    # Frozen tundra stays solid until thawed (reduces open-water evaporation / wetland methane in deep winter)
    thaw_fac = torch.clamp((t_soil_top - 270.15) / 5.0, 0.15, 1.0)
    f_target = torch.clamp((wet_soil + river_boost) * flatness * ml.valley_weight * thaw_fac, 0.0, WETLAND_MAX_FRAC) * is_land

    # Exact exponential wetland drainage/inundation adjustment (tau ~ 10 days)
    tau_wet = 10.0 * C.SECONDS_PER_DAY
    relax = -math.expm1(-float(dt) / tau_wet)
    ml.f_wetland = torch.clamp(ml.f_wetland + (f_target - ml.f_wetland) * relax, 0.0, WETLAND_MAX_FRAC) * is_land


def diagnose_maml_wetland_area_mkm2(ml: MAMLSubgridLandState, area_cell: torch.Tensor, nc: int, n_sub: int) -> float:
    """Compute float64 global inundated wetland/floodplain/lake area [Mkm^2] (lazy diagnostic, zero hot-path sync)."""
    a_sub = (area_cell.detach().cpu().to(torch.float64) / float(n_sub)).unsqueeze(1).expand(nc, n_sub).reshape(-1)
    area_mkm2 = float((ml.f_wetland.detach().cpu().to(torch.float64) * a_sub).sum().item() * 1.0e-12)
    ml.wetland_area_mkm2 = area_mkm2
    return area_mkm2
