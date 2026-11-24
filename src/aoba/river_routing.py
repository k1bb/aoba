"""Map-Free SCVT Voronoi River Routing (Priority-Flood Network + Kinematic Storage Delay).

Autonomously derives a globally acyclic river drainage network directly from `Earth2014`
surface topography on AOBA's SCVT Voronoi mesh using the optimal Priority-Flood algorithm
(Barnes, Lehman, & Mulla, 2014, *Computers & Geosciences*, 62, 117-127) — requiring ZERO
prescribed river-direction lookup maps so continental/paleoclimate/ice-sheet topography
changes automatically reorganize drainage basins.

Routes terrestrial surface/subsurface runoff (`runoff_total - runoff_ice`) down the
emergent Voronoi river network with slope-dependent kinematic wave velocity and exact
exponential-reservoir time integration (Mizukami et al., 2016, *Geosci. Model Dev.*, 9,
2223-2238; Miller et al., 1994, *J. Climate*, 7, 914-928), delivering focused freshwater
plumes (Amazon, Congo, Yangtze, Siberian Arctic rivers) to coastal ocean Voronoi cells
with exact float32 + Kahan mass conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import numpy as np
import torch

from aoba.surface import SURF_OCEAN, _kahan_add

# Kinematic river velocity parameters (Miller et al., 1994; Mizukami et al., 2016)
RIVER_V_REF_MS: float = 0.50       # Reference river flow velocity at S_ref [m s^-1]
RIVER_SLOPE_REF: float = 1.0e-3    # Reference topographic slope S_ref [-]
RIVER_V_MIN_MS: float = 0.15       # Minimum lowland/floodplain flow velocity [m s^-1]
RIVER_V_MAX_MS: float = 2.00       # Maximum steep mountain river velocity [m s^-1]
RIVER_SLOPE_MIN: float = 1.0e-4    # Minimum effective slope across filled flats [-]
RIVER_SLOPE_MAX: float = 1.0e-1    # Maximum effective slope cap [-]


@dataclass
class RiverRoutingState:
    """SCVT Voronoi river network topology (from Priority-Flood) and channel water storage."""

    downstream_idx: torch.Tensor   # [N_cells] int64 downstream neighbor index (self for ocean)
    reach_len_m: torch.Tensor      # [N_cells] float32 Voronoi reach length dc_edge [m]
    channel_slope: torch.Tensor    # [N_cells] float32 filled-DEM channel slope [-]
    flow_vel_ms: torch.Tensor      # [N_cells] float32 kinematic flow velocity [m s^-1]
    drain_area_m2: torch.Tensor    # [N_cells] float32 upstream catchment drainage area [m^2]
    is_land_river: torch.Tensor    # [N_cells] bool (True on non-ocean cells)
    storage_kg: torch.Tensor       # [N_cells] float32 prognostic liquid river water storage [kg]
    storage_comp: torch.Tensor     # [N_cells] float32 Kahan compensation for storage_kg [kg]
    discharge_m3s: torch.Tensor    # [N_cells] float32 instantaneous river discharge [m^3 s^-1]
    mouth_flux_kgm2s: torch.Tensor # [N_cells] float32 freshwater discharge into ocean cells [kg m^-2 s^-1]


def build_priority_flood_network(
    z_topo: torch.Tensor,
    surf_type: torch.Tensor,
    cells_on_edge: torch.Tensor,
    dc_edge: torch.Tensor,
    area_cell: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build globally acyclic Voronoi river network via Priority-Flood (Barnes et al., 2014).

    Floods inward from all ocean cells (`surf_type == SURF_OCEAN`) using a binary min-heap.
    Every non-ocean Voronoi cell receives a unique downstream neighbor guaranteed to reach
    the ocean with zero cycles and zero endorheic sinks.

    Returns `(downstream_idx, reach_len_m, channel_slope, drain_area_m2)` as NumPy arrays.
    """
    z_np = z_topo.detach().cpu().numpy().astype(np.float64)
    st_np = surf_type.detach().cpu().numpy().astype(np.int64)
    coe_np = cells_on_edge.detach().cpu().numpy().astype(np.int64)
    dc_np = dc_edge.detach().cpu().numpy().astype(np.float64)
    area_np = area_cell.detach().cpu().numpy().astype(np.float64)
    nc = z_np.shape[0]

    # Build adjacency list: adj[u] = list of (neighbor_v, dc_uv)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(nc)]
    for e in range(coe_np.shape[0]):
        u, v = int(coe_np[e, 0]), int(coe_np[e, 1])
        d = float(dc_np[e])
        adj[u].append((v, d))
        adj[v].append((u, d))

    visited = np.zeros(nc, dtype=bool)
    downstream = np.arange(nc, dtype=np.int64)
    reach_len = np.full(nc, float(np.median(dc_np)), dtype=np.float64)
    z_filled = z_np.copy()

    heap: list[tuple[float, int]] = []
    # Seed Priority-Flood with all ocean cells at elevation 0.0
    ocean_cells = np.nonzero(st_np == SURF_OCEAN)[0]
    if ocean_cells.size == 0:
        # Pure aquaplanet/land-world fallback: drain toward global elevation minimum
        min_idx = int(np.argmin(z_np))
        ocean_cells = np.array([min_idx], dtype=np.int64)

    for c in ocean_cells:
        ci = int(c)
        visited[ci] = True
        z_filled[ci] = 0.0
        heapq.heappush(heap, (0.0, ci))

    order: list[int] = []
    eps_z = 1.0e-3  # Tiny positive tie-breaking gradient across filled depressions [m]
    while heap:
        elev_u, u = heapq.heappop(heap)
        order.append(u)
        for v, d_uv in adj[u]:
            if not visited[v]:
                visited[v] = True
                downstream[v] = u
                reach_len[v] = max(d_uv, 1.0)
                elev_v = max(float(z_np[v]), elev_u + eps_z)
                z_filled[v] = elev_v
                heapq.heappush(heap, (elev_v, v))

    # Channel slope along filled DEM
    dz = np.maximum(z_filled - z_filled[downstream], 0.0)
    slope = np.clip(dz / np.maximum(reach_len, 1.0), RIVER_SLOPE_MIN, RIVER_SLOPE_MAX)
    slope[st_np == SURF_OCEAN] = RIVER_SLOPE_MIN

    # Accumulate upstream catchment drainage area in reverse topological order (upstream -> mouth)
    drain_area = area_np.copy()
    for u in reversed(order):
        if st_np[u] != SURF_OCEAN:
            ds = int(downstream[u])
            if ds != u:
                drain_area[ds] += drain_area[u]

    return downstream, reach_len, slope, drain_area


def init_river_routing_state(mesh, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> RiverRoutingState:
    """Initialize Priority-Flood Voronoi river routing state from `mesh` geometry."""
    dev = device if device is not None else mesh.z_topo.device
    ds_np, len_np, slp_np, area_np = build_priority_flood_network(
        mesh.z_topo, mesh.surf_type, mesh.cells_on_edge, mesh.dc_edge, mesh.area_cell
    )
    ds = torch.as_tensor(ds_np, device=dev, dtype=torch.int64)
    rlen = torch.as_tensor(len_np, device=dev, dtype=dtype)
    slp = torch.as_tensor(slp_np, device=dev, dtype=dtype)
    darea = torch.as_tensor(area_np, device=dev, dtype=dtype)
    is_land = (mesh.surf_type.to(dev) != SURF_OCEAN)

    # Slope-dependent kinematic wave velocity (Miller et al., 1994; Mizukami et al., 2016)
    vel = torch.clamp(RIVER_V_REF_MS * torch.sqrt(slp / RIVER_SLOPE_REF), RIVER_V_MIN_MS, RIVER_V_MAX_MS)
    nc = mesh.n_cells
    z0 = torch.zeros(nc, device=dev, dtype=dtype)
    return RiverRoutingState(
        downstream_idx=ds,
        reach_len_m=rlen,
        channel_slope=slp,
        flow_vel_ms=vel,
        drain_area_m2=darea,
        is_land_river=is_land,
        storage_kg=z0.clone(),
        storage_comp=z0.clone(),
        discharge_m3s=z0.clone(),
        mouth_flux_kgm2s=z0.clone(),
    )


def step_river_routing_voronoi(
    river: RiverRoutingState,
    runoff_liquid_kgm2s: torch.Tensor,
    area_cell: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Advance Voronoi river channel storage by `dt` [s] and return ocean mouth flux [kg m^-2 s^-1].

    Uses exact exponential linear-reservoir outflow `f = -expm1(-v/L * dt)` per Voronoi cell,
    which is unconditionally positive and strictly conserves total liquid water mass
    (channel storage + ocean river-mouth delivery == cumulative terrestrial runoff).
    """
    is_land_f = river.is_land_river.to(runoff_liquid_kgm2s.dtype)
    is_ocn_f = 1.0 - is_land_f

    # 1. Add local terrestrial liquid runoff [kg] into river channel storage
    inflow_local_kg = torch.clamp(runoff_liquid_kgm2s, min=0.0) * area_cell * dt * is_land_f
    stor_mid, comp_mid = _kahan_add(river.storage_kg, inflow_local_kg, river.storage_comp)
    stor_pos = torch.clamp(stor_mid, min=0.0) * is_land_f

    # 2. Exact unconditional exponential outflow fraction f_out = 1 - exp(-(v / L) * dt)
    k_rate = river.flow_vel_ms / torch.clamp(river.reach_len_m, min=1.0)
    f_out = -torch.expm1(-torch.clamp(k_rate * dt, min=0.0, max=40.0))
    outflow_kg = (stor_pos * f_out) * is_land_f

    # 3. Scatter outflow to downstream Voronoi cell (land -> channel storage; ocean -> river mouth)
    inflow_up_kg = torch.zeros_like(outflow_kg)
    inflow_up_kg.scatter_add_(0, river.downstream_idx, outflow_kg)

    net_land_delta_kg = (inflow_up_kg * is_land_f) - outflow_kg
    stor_new, comp_new = _kahan_add(stor_mid, net_land_delta_kg, comp_mid)
    river.storage_kg = torch.clamp(stor_new, min=0.0) * is_land_f
    river.storage_comp = comp_new * is_land_f

    inv_dt = 1.0 / max(dt, 1.0)
    river.discharge_m3s = (outflow_kg * inv_dt) * 1.0e-3  # rho_w = 1000 kg m^-3 -> m^3 s^-1
    mouth_flux = (inflow_up_kg * is_ocn_f) / torch.clamp(area_cell * dt, min=1.0)
    river.mouth_flux_kgm2s = mouth_flux
    return mouth_flux
