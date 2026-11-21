"""Zero-Q-Flux 1.5-Layer Wind-Driven Ekman Ocean & Coastal Runoff Routing on SCVT Voronoi Grid.

Implements Codron (2012, J. Climate, 25, 1747-1760) directly on the host SCVT Voronoi C-grid
with zero normal flux across all land-ocean coastal edges, guaranteeing exact global ocean heat
conservation without any prescribed Q-flux climatology.

References
----------
- Codron (2012, J. Climate, 25, 1747-1760, Sec. 2a, Eqs. 1-10):
  Regularized equatorial + extratropical wind-driven Ekman transport `M_E`, mass-compensating
  sub-surface return flow `-M_E`, vertical upwelling/downwelling `W_E = div(M_E)`, and
  horizontal mesoscale eddy diffusion `kappa_h`.
- MIROC6 AGCM Document Writing Team (2021, Sec. 4.1.2):
  Conservative coastal routing of terrestrial river & glacier calving freshwater/fusion enthalpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch

from aoba import constants as C
from aoba.scvt_dynamics import compute_scvt_divergence
from aoba.scvt_mesh import SCVTMesh
from aoba.surface import SURF_OCEAN, SurfaceState


# Codron (2012, Sec. 2a) standard physical parameters
H_THERMOCLINE: float = 200.0        # Sub-mixed-layer return-flow thickness H_2 [m]
R_EKMAN_DAMP_DAYS: float = 2.0      # Equatorial Rayleigh friction timescale [days]
R_EKMAN_DAMP: float = 1.0 / (R_EKMAN_DAMP_DAYS * C.SECONDS_PER_DAY)  # Equatorial Rayleigh friction [s^-1]
KAPPA_OCN_HORIZ: float = 2.0e3      # Horizontal mesoscale eddy diffusivity kappa_h [m^2 s^-1]
KAPPA_OCN_VERT: float = 2.0e-5      # Weak diapycnal exchange velocity [m s^-1]


def recompute_derived_constants() -> None:
    """Refresh Rayleigh Ekman friction rate from damping timescale."""
    global R_EKMAN_DAMP
    R_EKMAN_DAMP = 1.0 / (R_EKMAN_DAMP_DAYS * C.SECONDS_PER_DAY)


@dataclass
class EkmanOceanState:
    """Sub-mixed-layer thermocline temperature & diagnostic upwelling on SCVT cells."""

    t_sub_ml: torch.Tensor       # Layer-2 return-flow temperature T_2 [N_cells] [K]
    w_ekman: torch.Tensor        # Diagnostic Ekman upwelling mass flux div(M_E) [N_cells] [kg m^-2 s^-1]
    h_sub_ml: float = H_THERMOCLINE
    t1_comp: torch.Tensor | None = None  # Kahan float32 rounding compensation for Layer-1 T_1 [K]
    t2_comp: torch.Tensor | None = None  # Kahan float32 rounding compensation for Layer-2 T_2 [K]


def _kahan_add(val: torch.Tensor, delta: torch.Tensor, comp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate small `delta` onto `val` (~280 K) in float32 without swamping loss."""
    y = delta + comp
    t = val + y
    new_comp = torch.where(val.abs() >= y.abs(), (val - t) + y, (y - t) + val)
    return t, new_comp


def init_ekman_ocean_state(
    sfc: SurfaceState,
    dt_thermocline_k: float = 6.0,
) -> EkmanOceanState:
    """Initialize Layer-2 thermocline temperature `T_2 = max(T_freeze_ocn, T_1 - dt_thermocline)`."""
    t2 = torch.clamp(sfc.t_ocean_ml - dt_thermocline_k, min=C.T_FREEZE_OCEAN)
    return EkmanOceanState(
        t_sub_ml=t2,
        w_ekman=torch.zeros_like(t2),
        t1_comp=torch.zeros_like(t2),
        t2_comp=torch.zeros_like(t2),
    )


def step_ekman_ocean_voronoi(
    ocn: EkmanOceanState,
    sfc: SurfaceState,
    mesh: SCVTMesh,
    tau_x_sfc: torch.Tensor,
    tau_y_sfc: torch.Tensor,
    f_seaice: torch.Tensor,
    runoff_glacier: torch.Tensor,
    dt: float,
) -> None:
    """Advance `sfc.t_ocean_ml` (`T_1`) and `ocn.t_sub_ml` (`T_2`) by `dt` [s] in-place.

    Note on stress sign: `tau_x_sfc, tau_y_sfc` from `step_surface` are upward momentum fluxes
    from surface to atmosphere (`-rho * C_m * |U| * u_1`). Hence the wind stress exerted BY the
    atmosphere ON the ocean is `tau_ocn = -(1 - 0.5 * f_seaice) * tau_sfc`.
    """
    is_ocean = sfc.surf_type == SURF_OCEAN
    if not is_ocean.any():
        return

    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    ocn_edge = is_ocean[c0] & is_ocean[c1]
    mask_edge = ocn_edge.to(sfc.t_ocean_ml.dtype)

    # 1. Wind stress acting on ocean surface [N m^-2] (partially attenuated under sea-ice pack)
    ice_atten = 1.0 - 0.5 * torch.clamp(f_seaice, 0.0, 1.0)
    tx_ocn = -ice_atten * tau_x_sfc
    ty_ocn = -ice_atten * tau_y_sfc

    # Interpolate stress to Voronoi edges & apply Codron (2012, Eq. 2) Ekman inversion
    tx_e = 0.5 * (tx_ocn[c0] + tx_ocn[c1])
    ty_e = 0.5 * (ty_ocn[c0] + ty_ocn[c1])
    fe = mesh.f_edge
    r_e = R_EKMAN_DAMP
    denom_ek = fe * fe + r_e * r_e

    mx_ek = (r_e * tx_e + fe * ty_e) / denom_ek  # [kg m^-1 s^-1]
    my_ek = (r_e * ty_e - fe * tx_e) / denom_ek  # [kg m^-1 s^-1]

    # Normal Ekman mass transport across edge c0 -> c1 (strictly 0 across coastlines!)
    mn_ek = (mx_ek * mesh.east_normal_edge + my_ek * mesh.north_normal_edge) * mask_edge

    # Exact mass-conserving vertical Ekman upwelling flux W_E = div_SCVT(M_E) [kg m^-2 s^-1]
    w_ek = compute_scvt_divergence(mesh, mn_ek) * is_ocean.to(mn_ek.dtype)
    ocn.w_ekman = w_ek

    # 2. Flux-form upwind heat transport for Layer 1 (+mn_ek) and Layer 2 (-mn_ek)
    t1 = sfc.t_ocean_ml
    t2 = ocn.t_sub_ml

    f1_adv = torch.clamp(mn_ek, min=0.0) * t1[c0] + torch.clamp(mn_ek, max=0.0) * t1[c1]
    # Layer 2 return transport is mn_2 = -mn_ek:
    mn_ret = -mn_ek
    f2_adv = torch.clamp(mn_ret, min=0.0) * t2[c0] + torch.clamp(mn_ret, max=0.0) * t2[c1]

    # Vertical upwind temperature across base of mixed layer: T_2 if upwelling (W_E > 0), T_1 if downwelling
    t_vert = torch.where(w_ek >= 0.0, t2, t1)

    # 3. Horizontal mesoscale eddy diffusion with zero-flux coastal Neumann boundary condition
    h1_e = 0.5 * (sfc.h_ocean_ml[c0] + sfc.h_ocean_ml[c1])
    f1_diff = -C.RHO_SEAWATER * h1_e * KAPPA_OCN_HORIZ * ((t1[c1] - t1[c0]) / mesh.dc_edge) * mask_edge
    f2_diff = -C.RHO_SEAWATER * ocn.h_sub_ml * KAPPA_OCN_HORIZ * ((t2[c1] - t2[c0]) / mesh.dc_edge) * mask_edge

    div_q1 = C.CP_SEAWATER * (compute_scvt_divergence(mesh, f1_adv + f1_diff) - w_ek * t_vert)
    div_q2 = C.CP_SEAWATER * (compute_scvt_divergence(mesh, f2_adv + f2_diff) + w_ek * t_vert)

    # Weak vertical diapycnal mixing between Layer 1 and Layer 2
    q_diap = C.CP_SEAWATER * C.RHO_SEAWATER * KAPPA_OCN_VERT * (t2 - t1)

    # 4. Conservative coastal glacier calving fusion cooling (MIROC6 Sec. 4.1.2)
    q_calve = torch.zeros_like(t1)
    if (runoff_glacier > 0.0).any():
        # Total fusion cooling power [W] = L_f0 * area_land * runoff_glacier routed to coastal_ocean_idx
        power_glac = C.L_F0 * mesh.area_cell * torch.clamp(runoff_glacier, min=0.0)
        q_calve.scatter_add_(0, mesh.coastal_ocean_idx, power_glac)
        q_calve = q_calve / mesh.area_cell  # [W m^-2] over receiving coastal ocean cell

    c1_cap = C.CP_SEAWATER * C.RHO_SEAWATER * sfc.h_ocean_ml
    c2_cap = C.CP_SEAWATER * C.RHO_SEAWATER * ocn.h_sub_ml

    if ocn.t1_comp is None:
        ocn.t1_comp = torch.zeros_like(t1)
    if ocn.t2_comp is None:
        ocn.t2_comp = torch.zeros_like(t2)

    dt1 = (dt / c1_cap) * (-div_q1 + q_diap - q_calve)
    dt2 = (dt / c2_cap) * (-div_q2 - q_diap)
    t1_next, comp1 = _kahan_add(t1, dt1, ocn.t1_comp)
    t2_next, comp2 = _kahan_add(t2, dt2, ocn.t2_comp)

    # If dynamic advection cools mixed layer below seawater freezing point, form sea ice conservatively!
    freeze_excess = torch.where(is_ocean, torch.clamp(C.T_FREEZE_OCEAN - t1_next, min=0.0), torch.zeros_like(t1))
    dm_ice = (c1_cap * freeze_excess) / C.L_F0
    t1_final = t1_next + freeze_excess
    t2_final = torch.clamp(t2_next, min=C.T_FREEZE_OCEAN)

    ocn.t1_comp = torch.where(is_ocean & (freeze_excess == 0.0), comp1, torch.zeros_like(comp1))
    ocn.t2_comp = torch.where(is_ocean & (t2_next >= C.T_FREEZE_OCEAN), comp2, torch.zeros_like(comp2))
    sfc.t_ocean_ml = torch.where(is_ocean, t1_final, sfc.t_ocean_ml)
    sfc.m_seaice = torch.where(is_ocean, sfc.m_seaice + dm_ice, sfc.m_seaice)
    ocn.t_sub_ml = torch.where(is_ocean, t2_final, ocn.t_sub_ml)
