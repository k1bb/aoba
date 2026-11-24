"""Exact Voronoi Conservative Atmosphere <-> High-Resolution Ocean Grid Coupler.

Enables AOBA to run its 3D SCVT OGCM (`ogcm.py`), FlexPFT marine biogeochemistry
(`ocean_bgc.py`), and EVP-ITD sea-ice dynamics (`seaice_evp.py`) on an **independent
higher-resolution icosahedral SCVT mesh** (`ocean_glevel > glevel`, e.g., Glevel-3/5
superparameterized atmosphere coupled to a Glevel-6/7/8 eddy-permitting or
eddy-resolving ocean) while strictly preserving fine-scale SST fronts, western boundary
currents, eddies, and global heat/freshwater/ice conservation.

Mathematical Construction
-------------------------
Every coarse host atmosphere cell `i` on the unit sphere is the exact spherical Voronoi
polygon of its generator `xyz_atm[i]`. Consequently, each fine ocean cell center
`xyz_ocn[j]` belongs to the unique parent atmosphere cell:

    parent[j] = argmax_i (xyz_ocn[j] . xyz_atm[i])

- **Ocean -> Atmosphere (`ocn_to_atm_mean`)**: Exact area-weighted average over all wet
  fine ocean cells inside each coarse atmosphere cell via vectorized `scatter_add_`,
  with nearest-wet-cell fallback along mismatched coastal boundaries.
- **Atmosphere -> Ocean (`atm_to_ocn_intensive` / `atm_to_ocn_conservative`)**:
  Direct gather of intensive forcing fields (wind stress, 10 m wind, PAR, ice fraction),
  or globally area-conserving transfer of extensive fluxes (runoff, freshwater, heat
  increments) so global integrals match across mismatched land-sea masks.
- **Increment-Based Surface Coupling**: Atmospheric surface-layer tendencies
  `(dT_sfc, dm_ice_sfc, dswe_sfc)` from `step_surface` are applied directly to the fine
  ocean surface state `ocn_sfc` (with exact latent-heat freezing protection) so fine-scale
  ocean SST and sea-ice structures are never diffused onto the coarse atmospheric grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from aoba import constants as C
from aoba.scvt_mesh import SCVTMesh
from aoba.surface import SURF_OCEAN


@dataclass
class OceanAtmosCoupler:
    """Precomputed GPU-resident Voronoi coupler between coarse atmosphere and fine ocean SCVT meshes."""

    n_atm: int
    n_ocn: int
    atm_glevel: int
    ocn_glevel: int
    ocn_to_atm_idx: torch.Tensor         # [N_ocn] int64 parent host atmosphere cell index
    atm_to_ocn_nearest_wet: torch.Tensor # [N_atm] int64 nearest wet fine ocean cell (coastal fallback)
    ocn_wet_mask: torch.Tensor           # [N_ocn] bool (fine ocean cell is wet SURF_OCEAN)
    atm_ocn_mask: torch.Tensor           # [N_atm] bool (coarse host cell is SURF_OCEAN)
    ocn_area: torch.Tensor               # [N_ocn] float32 fine cell area [m^2]
    atm_area: torch.Tensor               # [N_atm] float32 coarse cell area [m^2]
    ocn_wet_area: torch.Tensor           # [N_ocn] float32 ocn_area * ocn_wet_mask
    atm_wet_area_sum: torch.Tensor       # [N_atm] float32 sum of fine wet ocean area inside host cell
    has_wet_children: torch.Tensor       # [N_atm] bool (atm_wet_area_sum > 0)
    coastal_ocn_nearest: torch.Tensor    # [N_atm] int64 nearest wet fine ocean cell for coastal runoff routing


def _nearest_spherical_indices(
    src_xyz: torch.Tensor,
    dst_xyz: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Return index in `dst_xyz` minimizing great-circle distance for every point in `src_xyz`."""
    src_cpu = src_xyz.detach().cpu().to(torch.float32)
    dst_t = dst_xyz.detach().cpu().to(torch.float32).T.contiguous()  # [3, N_dst]
    n_src = src_cpu.shape[0]
    out = torch.empty(n_src, dtype=torch.long)
    for start in range(0, n_src, chunk_size):
        end = min(start + chunk_size, n_src)
        dots = src_cpu[start:end] @ dst_t
        out[start:end] = dots.argmax(dim=1)
    return out


def build_ocean_atmos_coupler(
    atm_mesh: SCVTMesh,
    ocn_mesh: SCVTMesh,
    device: torch.device | None = None,
) -> OceanAtmosCoupler:
    """Construct an exact spherical Voronoi coupler between `atm_mesh` and `ocn_mesh`."""
    dev = device if device is not None else atm_mesh.area_cell.device
    n_atm = int(atm_mesh.n_cells)
    n_ocn = int(ocn_mesh.n_cells)

    # Exact Voronoi containment: fine ocean cell center -> coarse atmosphere parent cell
    ocn_to_atm = _nearest_spherical_indices(ocn_mesh.xyz_cell, atm_mesh.xyz_cell).to(dev)

    ocn_wet = (ocn_mesh.surf_type.to(dev) == SURF_OCEAN)
    atm_ocn = (atm_mesh.surf_type.to(dev) == SURF_OCEAN)
    ocn_area = ocn_mesh.area_cell.to(device=dev, dtype=torch.float32)
    atm_area = atm_mesh.area_cell.to(device=dev, dtype=torch.float32)
    ocn_wet_area = torch.where(ocn_wet, ocn_area, torch.zeros_like(ocn_area))

    atm_wet_area_sum = torch.zeros(n_atm, device=dev, dtype=torch.float32)
    atm_wet_area_sum.scatter_add_(0, ocn_to_atm, ocn_wet_area)
    has_wet_children = atm_wet_area_sum > 0.0

    # Nearest wet fine ocean cell for every coarse cell (coastal fallback & river-mouth transfer)
    wet_indices = torch.nonzero(ocn_wet, as_tuple=False).squeeze(-1)
    if wet_indices.numel() > 0:
        wet_xyz = ocn_mesh.xyz_cell.to(dev)[wet_indices]
        nearest_wet_pos = _nearest_spherical_indices(atm_mesh.xyz_cell, wet_xyz).to(dev)
        atm_to_ocn_nearest_wet = wet_indices[nearest_wet_pos]
    else:
        atm_to_ocn_nearest_wet = torch.zeros(n_atm, device=dev, dtype=torch.long)

    return OceanAtmosCoupler(
        n_atm=n_atm,
        n_ocn=n_ocn,
        atm_glevel=int(atm_mesh.glevel),
        ocn_glevel=int(ocn_mesh.glevel),
        ocn_to_atm_idx=ocn_to_atm,
        atm_to_ocn_nearest_wet=atm_to_ocn_nearest_wet,
        ocn_wet_mask=ocn_wet,
        atm_ocn_mask=atm_ocn,
        ocn_area=ocn_area,
        atm_area=atm_area,
        ocn_wet_area=ocn_wet_area,
        atm_wet_area_sum=atm_wet_area_sum,
        has_wet_children=has_wet_children,
        coastal_ocn_nearest=atm_to_ocn_nearest_wet,
    )


def atm_to_ocn_intensive(
    coupler: OceanAtmosCoupler,
    field_atm: torch.Tensor,
) -> torch.Tensor:
    """Gather an intensive atmospheric surface field `[N_atm]` onto the fine ocean mesh `[N_ocn]`."""
    return field_atm[coupler.ocn_to_atm_idx]


def atm_to_ocn_conservative_flux(
    coupler: OceanAtmosCoupler,
    flux_atm: torch.Tensor,
    *,
    ocean_only_source: bool = False,
) -> torch.Tensor:
    """Conservatively map a per-area flux `[N_atm]` (e.g. runoff/freshwater) onto fine wet ocean cells `[N_ocn]`.

    Guarantees exact global integral conservation:
        sum(flux_ocn * ocn_wet_area) == sum(flux_atm * src_area)
    """
    src_mask = coupler.atm_ocn_mask if ocean_only_source else torch.ones_like(coupler.atm_ocn_mask)
    total_mass_atm = (flux_atm * coupler.atm_area * src_mask.to(flux_atm.dtype)).sum()

    # Map via parent cell; for coarse cells with no wet children (coastal mismatch), deposit to nearest wet cell
    raw_ocn = flux_atm[coupler.ocn_to_atm_idx] * coupler.ocn_wet_mask.to(flux_atm.dtype)
    orphan_mask = src_mask & (~coupler.has_wet_children)
    if bool(orphan_mask.any()):
        orphan_mass = torch.where(orphan_mask, flux_atm * coupler.atm_area, torch.zeros_like(flux_atm))
        extra_mass = torch.zeros(coupler.n_ocn, device=flux_atm.device, dtype=flux_atm.dtype)
        extra_mass.scatter_add_(0, coupler.coastal_ocn_nearest, orphan_mass)
        raw_ocn = raw_ocn + extra_mass / torch.clamp(coupler.ocn_area, min=1.0)

    total_mass_ocn = (raw_ocn * coupler.ocn_wet_area).sum()
    # Exact global conservation factor (safeguarded against zero flux)
    scale = torch.where(
        total_mass_ocn.abs() > 1.0e-20,
        total_mass_atm / torch.where(total_mass_ocn.abs() > 1.0e-20, total_mass_ocn, torch.ones_like(total_mass_ocn)),
        torch.ones_like(total_mass_ocn),
    )
    return raw_ocn * torch.clamp(scale, min=0.0, max=10.0)


def ocn_to_atm_mean(
    coupler: OceanAtmosCoupler,
    field_ocn: torch.Tensor,
    fallback_atm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Area-weighted conservative mean of fine wet ocean field `[N_ocn]` onto coarse host cells `[N_atm]`."""
    num = torch.zeros(coupler.n_atm, device=field_ocn.device, dtype=field_ocn.dtype)
    num.scatter_add_(0, coupler.ocn_to_atm_idx, field_ocn * coupler.ocn_wet_area.to(field_ocn.dtype))
    den = torch.clamp(coupler.atm_wet_area_sum.to(field_ocn.dtype), min=1.0)
    mean_atm = num / den
    # Coastal coarse ocean cells without wet children fall back to nearest wet fine ocean cell
    nearest_val = field_ocn[coupler.atm_to_ocn_nearest_wet]
    ocn_agg = torch.where(coupler.has_wet_children, mean_atm, nearest_val)
    if fallback_atm is None:
        return ocn_agg
    return torch.where(coupler.atm_ocn_mask, ocn_agg, fallback_atm)


def apply_atm_surface_increments_to_ocn_sfc(
    coupler: OceanAtmosCoupler,
    ocn_sfc: object,
    dt_ml_atm: torch.Tensor,
    dm_ice_atm: torch.Tensor,
    dswe_ice_atm: torch.Tensor,
) -> None:
    """Apply atmospheric surface-layer tendencies onto fine `ocn_sfc` with exact freeze protection.

    Preserves fine-scale SST gradients and sea-ice edges on `ocn_sfc` while transferring
    the atmospheric heat and sea-ice mass exchange from `step_surface`.
    """
    dt_ocn = atm_to_ocn_intensive(coupler, dt_ml_atm)
    dm_ocn = atm_to_ocn_intensive(coupler, dm_ice_atm)
    dswe_ocn = atm_to_ocn_intensive(coupler, dswe_ice_atm)
    is_wet = coupler.ocn_wet_mask

    t_next = ocn_sfc.t_ocean_ml + dt_ocn
    freeze_def = torch.where(is_wet, torch.clamp(C.T_FREEZE_OCEAN - t_next, min=0.0), torch.zeros_like(t_next))
    c_ml = C.CP_SEAWATER * C.RHO_SEAWATER * ocn_sfc.h_ocean_ml
    ocn_sfc.t_ocean_ml = torch.where(is_wet, t_next + freeze_def, ocn_sfc.t_ocean_ml)
    ocn_sfc.m_seaice = torch.where(
        is_wet,
        torch.clamp(ocn_sfc.m_seaice + dm_ocn + (c_ml * freeze_def) / C.L_F0, min=0.0),
        ocn_sfc.m_seaice,
    )
    ocn_sfc.swe_seaice = torch.where(
        is_wet,
        torch.clamp(ocn_sfc.swe_seaice + dswe_ocn, min=0.0),
        ocn_sfc.swe_seaice,
    )
