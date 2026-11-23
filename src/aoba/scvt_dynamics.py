"""Global SCVT Voronoi C-Grid + Charney-Phillips Nonhydrostatic RK3-HEVI Dynamical Core.

Solves the fully compressible flux-form moist nonhydrostatic Euler equations on the global
icosahedral SCVT orthogonal C-grid with Klemp (2011) terrain-following coordinates, exact
energy-conserving TRiSK rotational/Coriolis operators, Zalesak positive-definite Voronoi FCT
tracer transport, and Held & Suarez (1994) benchmark forcing.

References
----------
- Ringler, Thuburn, Klemp, & Skamarock (2010, J. Comput. Phys., 229, 3065-3090, Eqs. 17-26):
  Vector-invariant kinetic energy, dual-triangle vorticity, and energy-conserving nonlinear
  TRiSK Coriolis/vorticity flux on arbitrary spherical Voronoi C-grids.
- Skamarock, Klemp, Duda, Fowler, Park, & Ringler (2012, Mon. Wea. Rev., 140, 3090-3105):
  MPAS-Atmosphere nonhydrostatic RK3-HEVI formulation in `(rho_d, Theta_m)` variables with
  divergence damping and 4th-order horizontal hyperdiffusion.
- Gassmann (2013, Q. J. R. Meteorol. Soc., 139, 152-175):
  ICON-IAP moist nonhydrostatic Voronoi C-grid energetics and condensate loading.
- Klemp (2011, Mon. Wea. Rev., 139, 2163-2177):
  Smoothed terrain-following (STF) vertical coordinate and horizontal pressure-gradient metric.
- Held & Suarez (1994, Bull. Amer. Meteor. Soc., 75, 1825-1830, Eqs. 1-4):
  Dry global dynamical-core benchmark Newtonian relaxation and boundary-layer Rayleigh drag.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import (
    can_use_crm_metal,
    dispatch_scvt_div_metal,
    dispatch_scvt_fast_acoustic_metal,
    dispatch_scvt_scalar_lap_metal,
    dispatch_scvt_trisk_rot_metal,
    dispatch_scvt_vec_lap_metal,
    dispatch_scvt_vorticity_metal,
    solve_tridiagonal_batched,
)
from aoba.scvt_mesh import SCVTMesh, project_cell_vectors_to_edges, reconstruct_cell_winds
from aoba.thermodynamics import diagnose_pressure_temperature, qsat_mixed, rcemip_analytic_sounding
from aoba.vertical_grid import VerticalGrid, compute_stf_heights


@dataclass
class GlobalAtmosState:
    """Prognostic 3D atmospheric state on the global SCVT Voronoi C-grid."""

    rho_d: torch.Tensor      # Dry air density [Nz, N_cells] [kg m^-3]
    u_edge: torch.Tensor     # Normal horizontal velocity at Voronoi edges [Nz, N_edges] [m s^-1]
    w: torch.Tensor          # Physical vertical velocity at interfaces [Nz+1, N_cells] [m s^-1]
    theta_m: torch.Tensor    # Modified moist potential temperature [Nz, N_cells] [K]
    q_n: torch.Tensor        # Non-precipitating water [Nz, N_cells] [kg kg^-1]
    q_p: torch.Tensor        # Precipitating water [Nz, N_cells] [kg kg^-1]
    q_v: torch.Tensor        # Diagnosed water vapor [Nz, N_cells] [kg kg^-1]
    q_c: torch.Tensor        # Diagnosed cloud liquid [Nz, N_cells] [kg kg^-1]
    q_i: torch.Tensor        # Diagnosed cloud ice [Nz, N_cells] [kg kg^-1]
    z_w: torch.Tensor        # Physical interface heights [Nz+1, N_cells] [m]
    z_c: torch.Tensor        # Physical layer-center heights [Nz, N_cells] [m]
    dz_c: torch.Tensor       # Physical layer thicknesses [Nz, N_cells] [m]
    q_dust: torch.Tensor | None = None  # Prognostic mineral dust mixing ratio [Nz, N_cells] [kg kg^-1]


# ---------------------------------------------------------------------------
# 1. Discrete Hydrostatic Global Initialization Over Arbitrary Earth2014 Terrain
# ---------------------------------------------------------------------------
def init_global_hydrostatic_atmosphere(
    mesh: SCVTMesh,
    vgrid: VerticalGrid,
    t_eq_sfc: float = 300.0,
    dt_pole: float = 35.0,
    p_sl: float = 101325.0,
    moist: bool = True,
    baroclinic_pert: bool = True,
) -> GlobalAtmosState:
    """Initialize a discretely hydrostatic global atmosphere over `mesh.z_topo_smooth`.

    Integrates hydrostatic pressure upward from terrain `z_w[0] = z_topo_smooth` in `float64`
    so `(p[k] - p[k-1]) / dz_w[k] + 0.5 * (rho_tot[k] + rho_tot[k-1]) * g == 0` holds in every
    column before casting to target `dtype`/`device`.
    """
    dev = mesh.lat_cell.device
    dtype = mesh.lat_cell.dtype
    nz = vgrid.nz
    n_cells = mesh.n_cells
    n_edges = mesh.n_edges

    z_w, z_c, dz_c = compute_stf_heights(vgrid, mesh.z_topo_smooth)

    zc64 = z_c.cpu().to(torch.float64)
    zw64 = z_w.cpu().to(torch.float64)
    lat64 = mesh.lat_cell.cpu().to(torch.float64)
    lon64 = mesh.lon_cell.cpu().to(torch.float64)

    # Smooth meridional surface temperature & moisture profile
    sin2_lat = torch.sin(lat64) ** 2
    t_sl = t_eq_sfc - dt_pole * sin2_lat  # [N_cells] at z = 0
    lapse = 0.0065
    t_strat = 208.0

    temp64 = torch.maximum(
        t_sl.unsqueeze(0) - lapse * zc64,
        torch.full_like(zc64, t_strat),
    )
    # U.S. Standard Atmosphere (1976) stratospheric inversion & mesospheric lapse above 20 km
    # Keeps z <= 20 km identical while giving realistic static stability N^2 & stratopause (~262 K) up to 48-80 km
    strat_dt = torch.where(
        zc64 <= 20000.0,
        torch.zeros_like(zc64),
        torch.where(
            zc64 <= 32000.0,
            0.0010 * (zc64 - 20000.0),
            torch.where(
                zc64 <= 47000.0,
                12.0 + 0.0028 * (zc64 - 32000.0),
                torch.where(
                    zc64 <= 51000.0,
                    torch.full_like(zc64, 54.0),
                    torch.clamp(54.0 - 0.0024 * (zc64 - 51000.0), min=-28.0),
                ),
            ),
        ),
    )
    temp64 = temp64 + strat_dt
    if baroclinic_pert:
        # Small localized synoptic temperature wave in midlatitudes to seed baroclinic eddies
        wave = 1.0 * torch.sin(3.0 * lon64) * (torch.sin(2.0 * lat64) ** 2)
        env_z = torch.exp(-((zc64 - 4000.0) / 3500.0) ** 2)
        temp64 = temp64 + env_z * wave.unsqueeze(0)

    # Exact barometric surface pressure for piecewise-linear lapse rate
    p_sfc64 = p_sl * torch.pow(torch.clamp(1.0 - lapse * zw64[0] / t_sl, min=0.2), C.GRAVITY / (C.R_D * lapse))

    if moist:
        # Sub-saturated mixed-phase relative humidity profile (65% at surface -> 15% aloft; zero startup ice flash!)
        p_approx = p_sl * torch.pow(torch.clamp(1.0 - lapse * zc64 / t_sl.unsqueeze(0), min=0.05), C.GRAVITY / (C.R_D * lapse))
        rh_prof = 0.65 * torch.exp(-((zc64 - zw64[0:1]) / 4500.0) ** 1.5)
        qv64 = torch.clamp(rh_prof * qsat_mixed(temp64, p_approx)[0], min=1.0e-7)
    else:
        qv64 = torch.zeros_like(zc64)

    p64 = torch.empty_like(zc64)
    rho64 = torch.empty_like(zc64)

    dz0 = zc64[0] - zw64[0]
    tv0 = temp64[0] * (1.0 + C.RV_OVER_RD * qv64[0]) / (1.0 + qv64[0])
    p64[0] = p_sfc64 * torch.exp(-C.GRAVITY * dz0 / (C.R_D * tv0))
    rho64[0] = p64[0] / (C.R_D * temp64[0] * (1.0 + C.RV_OVER_RD * qv64[0]))

    for k in range(1, nz):
        dz_wk = zc64[k] - zc64[k - 1]
        rtot_below = rho64[k - 1] * (1.0 + qv64[k - 1])
        fac_k = (1.0 + qv64[k]) / (C.R_D * temp64[k] * (1.0 + C.RV_OVER_RD * qv64[k]))
        p64[k] = (p64[k - 1] - 0.5 * dz_wk * C.GRAVITY * rtot_below) / (1.0 + 0.5 * dz_wk * C.GRAVITY * fac_k)
        rho64[k] = p64[k] / (C.R_D * temp64[k] * (1.0 + C.RV_OVER_RD * qv64[k]))

    thm64 = (C.P_REF / (C.R_D * rho64)) * torch.pow(p64 / C.P_REF, 1.0 / C.GAMMA_D)

    rho_d = rho64.to(device=dev, dtype=dtype)
    theta_m = thm64.to(device=dev, dtype=dtype)
    qv = qv64.to(device=dev, dtype=dtype)

    return GlobalAtmosState(
        rho_d=rho_d,
        u_edge=torch.zeros((nz, n_edges), device=dev, dtype=dtype),
        w=torch.zeros((nz + 1, n_cells), device=dev, dtype=dtype),
        theta_m=theta_m,
        q_n=qv.clone(),
        q_p=torch.zeros_like(qv),
        q_v=qv,
        q_c=torch.zeros_like(qv),
        q_i=torch.zeros_like(qv),
        z_w=z_w,
        z_c=z_c,
        dz_c=dz_c,
    )


# ---------------------------------------------------------------------------
# 2. Horizontal Discrete Calculus Operators on SCVT Voronoi C-Grid
# ---------------------------------------------------------------------------
def compute_scvt_divergence(
    mesh: SCVTMesh,
    flux_edge: torch.Tensor,
) -> torch.Tensor:
    """Discrete Gauss divergence `(1 / A_i) * sum_{e in cell i} s_{i,e} * l_e * flux_edge[..., e]`.

    Satisfies exact spherical integral conservation: `sum_i A_i * div_i == 0`.
    """
    if flux_edge.ndim == 2 and can_use_crm_metal(flux_edge):
        return dispatch_scvt_div_metal(mesh, flux_edge)
    le_flux = flux_edge * mesh.dv_edge  # [..., n_edges]
    gathered = le_flux[..., mesh.edges_on_cell]  # [..., n_cells, 6]
    return (gathered * mesh.edge_sign_on_cell).sum(dim=-1) / mesh.area_cell


def compute_scvt_vorticity(
    mesh: SCVTMesh,
    u_edge: torch.Tensor,
) -> torch.Tensor:
    """Discrete Stokes curl `(1 / A_v) * sum_{e in vert v} s_{v,e} * dc_e * u_edge[..., e]`."""
    if u_edge.ndim == 2 and can_use_crm_metal(u_edge):
        return dispatch_scvt_vorticity_metal(mesh, u_edge)
    dc_u = u_edge * mesh.dc_edge  # [..., n_edges]
    ev = mesh.edges_on_vert       # [n_verts, 3]
    return (dc_u[..., ev] * mesh.edge_sign_on_vert).sum(dim=-1) / mesh.area_vert


def compute_vector_laplacian_edge(
    mesh: SCVTMesh,
    u_edge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute vector Laplacian `grad(div u) - curl(zeta)` and pure grad-div on edges `[..., n_edges]`."""
    if u_edge.ndim == 2 and can_use_crm_metal(u_edge):
        return dispatch_scvt_vec_lap_metal(mesh, u_edge)
    div_c = compute_scvt_divergence(mesh, u_edge)      # [..., n_cells]
    zeta_v = compute_scvt_vorticity(mesh, u_edge)      # [..., n_verts]

    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    grad_div = (div_c[..., c1] - div_c[..., c0]) / mesh.dc_edge

    # Along edge e, tangential vector t_e goes v0 -> v1, with n_e x t_e = +r_e:
    # vector identity del^2 u = grad(div u) - r x grad(zeta) => n_e . (r x grad(zeta)) = -t_e . grad(zeta)!
    #Wait: n_e . (-r x grad(zeta)) = +(n_e x r) . grad(zeta) = -t_e . grad(zeta) = -(zeta[v1] - zeta[v0]) / dv_edge!
    v0 = mesh.verts_on_edge[:, 0]
    v1 = mesh.verts_on_edge[:, 1]
    rot_curl = -(zeta_v[..., v1] - zeta_v[..., v0]) / torch.clamp(mesh.dv_edge, min=100.0)

    return grad_div + rot_curl, grad_div


def compute_scalar_laplacian_cell(
    mesh: SCVTMesh,
    phi_cell: torch.Tensor,
) -> torch.Tensor:
    """Compute conservative cell-centered scalar Laplacian `div(grad(phi))` `[..., n_cells]`."""
    if phi_cell.ndim == 2 and can_use_crm_metal(phi_cell):
        return dispatch_scvt_scalar_lap_metal(mesh, phi_cell)
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    grad_e = (phi_cell[..., c1] - phi_cell[..., c0]) / mesh.dc_edge
    return compute_scvt_divergence(mesh, grad_e)


# ---------------------------------------------------------------------------
# 3. Positive-Definite Zalesak FCT Tracer Advection on SCVT Voronoi Grid
# ---------------------------------------------------------------------------
def fct_advect_scvt(
    mesh: SCVTMesh,
    q: torch.Tensor,
    mass_old: torch.Tensor,
    mass_new: torch.Tensor,
    uh_edge: torch.Tensor,
    omega_z: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Advect non-negative scalar `q` `[Nz, N_cells]` with exact mass conservation & positivity.

    Parameters
    ----------
    mass_old, mass_new : torch.Tensor
        Coupled dry-air layer mass `rho_d * dz_c` `[Nz, N_cells]` [kg m^-2], satisfying
        `mass_new == mass_old - dt * (div_scvt(uh_edge) + (omega_z[1:] - omega_z[:-1]))`.
    uh_edge : torch.Tensor
        Horizontal layer dry-mass flux `[Nz, N_edges]` [kg m^-1 s^-1] (normal `c0 -> c1`).
    omega_z : torch.Tensor
        Contravariant vertical dry-mass flux `[Nz+1, N_cells]` [kg m^-2 s^-1].
    """
    nz, n_cells = q.shape
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    eps = 1.0e-20

    q_c0 = q[:, c0]
    q_c1 = q[:, c1]

    # 1. Low-order 1st-order donor-cell upwind fluxes
    fl_h = torch.clamp(uh_edge, min=0.0) * q_c0 + torch.clamp(uh_edge, max=0.0) * q_c1
    fl_z = torch.zeros_like(omega_z)
    om_int = omega_z[1:-1]
    fl_z[1:-1] = torch.clamp(om_int, min=0.0) * q[:-1] + torch.clamp(om_int, max=0.0) * q[1:]

    div_fl = compute_scvt_divergence(mesh, fl_h) + (fl_z[1:] - fl_z[:-1])
    m_safe = torch.clamp(mass_new, min=1.0e-4)
    q_td = torch.clamp((mass_old * q - dt * div_fl) / m_safe, min=0.0)

    # 2. Centered high-order fluxes & anti-diffusive fluxes A = F^H - F^L
    fh_h = uh_edge * (0.5 * (q_c0 + q_c1))
    fh_z = torch.zeros_like(omega_z)
    fh_z[1:-1] = om_int * (0.5 * (q[:-1] + q[1:]))
    if nz >= 4:
        fh_z[2:-2] = omega_z[2:-2] * ((-q[:-3] + 7.0 * q[1:-2] + 7.0 * q[2:-1] - q[3:]) / 12.0)

    ah = fh_h - fl_h
    az = fh_z - fl_z

    # 3. Local neighborhood extrema across cell + Voronoi edge neighbors + vertical neighbors
    q_hi0 = torch.maximum(q, q_td)
    q_lo0 = torch.minimum(q, q_td)

    # Neighbor cell indices across each cell's 6 edges
    e_cell = mesh.edges_on_cell  # [N_cells, 6]
    nb_c0 = mesh.cells_on_edge[e_cell, 0]
    nb_c1 = mesh.cells_on_edge[e_cell, 1]
    cell_id = torch.arange(n_cells, device=q.device).unsqueeze(1)
    nb_idx = torch.where(nb_c0 == cell_id, nb_c1, nb_c0)
    valid_nb = mesh.edge_sign_on_cell != 0.0  # [N_cells, 6]
    nb_idx_safe = torch.where(valid_nb, nb_idx, cell_id.expand_as(nb_idx))

    q_max_h = q_hi0[:, nb_idx_safe].max(dim=-1).values
    q_min_h = q_lo0[:, nb_idx_safe].min(dim=-1).values

    q_z_up = torch.cat([q_hi0[1:], q_hi0[-1:]], dim=0)
    q_z_dn = torch.cat([q_hi0[:1], q_hi0[:-1]], dim=0)
    q_max = torch.maximum(torch.maximum(q_hi0, q_max_h), torch.maximum(q_z_up, q_z_dn))

    q_zlo_up = torch.cat([q_lo0[1:], q_lo0[-1:]], dim=0)
    q_zlo_dn = torch.cat([q_lo0[:1], q_lo0[:-1]], dim=0)
    q_min = torch.minimum(torch.minimum(q_lo0, q_min_h), torch.minimum(q_zlo_up, q_zlo_dn))

    # 4. Zalesak incoming/outgoing anti-diffusive sums
    # For cell i and edge a with outward sign s_{i,a}: flux OUT of cell i is s_{i,a} * l_e * ah_e
    le_ah = ah * mesh.dv_edge  # [Nz, N_edges]
    out_edge = le_ah[:, e_cell] * mesh.edge_sign_on_cell  # [Nz, N_cells, 6]: >0 outgoing, <0 incoming

    p_plus_h = (-torch.clamp(out_edge, max=0.0)).sum(dim=-1) / mesh.area_cell
    p_minus_h = torch.clamp(out_edge, min=0.0).sum(dim=-1) / mesh.area_cell

    p_plus = dt * (p_plus_h + torch.clamp(az[:-1], min=0.0) - torch.clamp(az[1:], max=0.0))
    p_minus = dt * (p_minus_h + torch.clamp(az[1:], min=0.0) - torch.clamp(az[:-1], max=0.0))

    q_plus = m_safe * torch.clamp(q_max - q_td, min=0.0)
    q_minus = m_safe * torch.clamp(q_td - q_min, min=0.0)

    r_plus = torch.where(p_plus > eps, torch.clamp(q_plus / (p_plus + eps), 0.0, 1.0), torch.ones_like(q_td))
    r_minus = torch.where(p_minus > eps, torch.clamp(q_minus / (p_minus + eps), 0.0, 1.0), torch.ones_like(q_td))

    ch = torch.where(
        ah >= 0.0,
        torch.minimum(r_minus[:, c0], r_plus[:, c1]),
        torch.minimum(r_plus[:, c0], r_minus[:, c1]),
    )
    cz = torch.zeros_like(az)
    cz[1:-1] = torch.where(
        az[1:-1] >= 0.0,
        torch.minimum(r_minus[:-1], r_plus[1:]),
        torch.minimum(r_plus[:-1], r_minus[1:]),
    )

    ah_lim = ch * ah
    az_lim = cz * az
    div_anti = compute_scvt_divergence(mesh, ah_lim) + (az_lim[1:] - az_lim[:-1])
    return torch.clamp(q_td - (dt / m_safe) * div_anti, min=0.0)


# ---------------------------------------------------------------------------
# 4. Global Nonhydrostatic RK3-HEVI Step on SCVT Voronoi C-Grid
# ---------------------------------------------------------------------------
def _scvt_fast_acoustic_hevi_cpu_gpu(
    n_ac: int,
    dt_ac: float,
    beta_hevi: float,
    gravity: float,
    accum_fluxes: bool,
    m0: torch.Tensor,
    Thm0: torch.Tensor,
    u0: torch.Tensor,
    w0: torch.Tensor,
    udotz0: torch.Tensor,
    pres_s: torch.Tensor,
    dp_dThm_s: torch.Tensor,
    Thm_s: torch.Tensor,
    fac_z0: torch.Tensor,
    fac_z1: torch.Tensor,
    rtot_e: torch.Tensor,
    dc_e: torch.Tensor,
    tend_u_slow: torch.Tensor,
    tend_Thm_slow: torch.Tensor,
    sponge_c: torch.Tensor,
    sponge_w: torch.Tensor,
    area_e: torch.Tensor,
    zw_slope_e: torch.Tensor,
    valid_ce: torch.Tensor,
    area_c: torch.Tensor,
    m_edge: torch.Tensor,
    thm_e: torch.Tensor,
    rho_dw: torch.Tensor,
    rtot_w: torch.Tensor,
    thm_w: torch.Tensor,
    dz_c: torch.Tensor,
    dz_w_int: torch.Tensor,
    q_load: torch.Tensor,
    dl: torch.Tensor,
    diag: torch.Tensor,
    du: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    dv_edge: torch.Tensor,
    edges_on_cell: torch.Tensor,
    edge_sign_on_cell: torch.Tensor,
    area_cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled MPAS split-explicit horizontal acoustic + vertical HEVI loop."""
    m_k = m0
    Thm_k = Thm0
    u_k = u0
    w_k = w0
    udotz_k = udotz0

    uh_accum = torch.zeros_like(u0)
    om_accum = torch.zeros_like(w0)
    rho_dw_safe = torch.clamp(rho_dw, min=1.0e-5)

    for _ in range(n_ac):
        p_k = pres_s + dp_dThm_s * (Thm_k - Thm_s)
        pgf_e = -(p_k[:, c1] * fac_z1 - p_k[:, c0] * fac_z0) / (rtot_e * dc_e)

        u_next = u_k + dt_ac * (tend_u_slow + pgf_e - sponge_c * u_k)
        u_int_next = 0.5 * (u_next[:-1] + u_next[1:])
        uz_next = area_e * u_int_next * zw_slope_e
        udotz_next = (uz_next[:, edges_on_cell] * valid_ce).sum(dim=-1) / area_c

        uh_k = m_edge * (1.10 * u_next - 0.10 * u_k)
        fTh_h_k = uh_k * thm_e
        div_h_m = (((uh_k * dv_edge)[:, edges_on_cell]) * edge_sign_on_cell).sum(dim=-1) / area_cell
        div_h_Thm = (((fTh_h_k * dv_edge)[:, edges_on_cell]) * edge_sign_on_cell).sum(dim=-1) / area_cell

        om_old = torch.zeros_like(w_k)
        om_old[1:-1] = rho_dw * (w_k[1:-1] - udotz_k)

        m_dag = m_k - dt_ac * (div_h_m + (om_old[1:] - om_old[:-1]))
        fTh_z_old = torch.zeros_like(w_k)
        fTh_z_old[1:-1] = thm_w * om_old[1:-1]
        Thm_dag = Thm_k + dt_ac * (-div_h_Thm - (fTh_z_old[1:] - fTh_z_old[:-1]) + tend_Thm_slow)

        p_dag = pres_s + dp_dThm_s * (Thm_dag - Thm_s)
        rho_k = m_k / dz_c
        rho_dag = m_dag / dz_c
        rtot_k_w = 0.5 * (rho_k[:-1] * q_load[:-1] + rho_k[1:] * q_load[1:])
        rtot_dag_w = 0.5 * (rho_dag[:-1] * q_load[:-1] + rho_dag[1:] * q_load[1:])

        p_blend = (1.0 - beta_hevi) * p_k + beta_hevi * p_dag
        rtot_blend_w = (1.0 - beta_hevi) * rtot_k_w + beta_hevi * rtot_dag_w
        nh_imbal = (p_blend[1:] - p_blend[:-1]) / dz_w_int + rtot_blend_w * gravity

        fz_phys_k = rho_dw * w_k[1:-1]
        rhs_om = (
            -rho_dw * (udotz_next - udotz_k)
            + dt_ac * (-(rho_dw / rtot_w) * nh_imbal - sponge_w * fz_phys_k)
        )
        # Inlined batched Thomas tridiagonal solve along dim=0
        K_tri = diag.shape[0]
        c_list: list[torch.Tensor] = []
        d_list: list[torch.Tensor] = []
        inv_b0 = 1.0 / diag[0]
        c_prev = du[0] * inv_b0
        d_prev = rhs_om[0] * inv_b0
        c_list.append(c_prev)
        d_list.append(d_prev)
        for kk in range(1, K_tri):
            inv_den = 1.0 / (diag[kk] - dl[kk] * c_prev)
            c_prev = du[kk] * inv_den if kk < K_tri - 1 else torch.zeros_like(d_prev)
            d_prev = (rhs_om[kk] - dl[kk] * d_prev) * inv_den
            c_list.append(c_prev)
            d_list.append(d_prev)
        x_list: list[torch.Tensor] = [d_prev] * K_tri
        x_nx = d_list[K_tri - 1]
        x_list[K_tri - 1] = x_nx
        for kk in range(K_tri - 2, -1, -1):
            x_nx = d_list[kk] - c_list[kk] * x_nx
            x_list[kk] = x_nx
        dom = torch.stack(x_list, dim=0)

        om_new = om_old.clone()
        om_new[1:-1] = om_old[1:-1] + dom

        m_next = torch.clamp(
            m_dag - beta_hevi * dt_ac * (om_new[1:] - om_old[1:] - (om_new[:-1] - om_old[:-1])),
            min=1.0e-3,
        )
        dfTh_z = torch.zeros_like(w_k)
        dfTh_z[1:-1] = thm_w * dom
        Thm_next = Thm_dag - beta_hevi * dt_ac * (dfTh_z[1:] - dfTh_z[:-1])

        w_next = torch.zeros_like(w_k)
        w_next[1:-1] = udotz_next + om_new[1:-1] / rho_dw_safe

        if accum_fluxes:
            uh_accum = uh_accum + uh_k
            om_accum = om_accum + ((1.0 - beta_hevi) * om_old + beta_hevi * om_new)

        m_k = m_next
        Thm_k = Thm_next
        u_k = u_next
        w_k = w_next
        udotz_k = udotz_next

    return m_k, Thm_k, u_k, w_k, uh_accum, om_accum


def step_scvt_rk3_hevi(
    state: GlobalAtmosState,
    mesh: SCVTMesh,
    vgrid: VerticalGrid,
    dt_dyn: float,
    beta_hevi: float = 0.60,
    du_phys_edge: torch.Tensor | None = None,
    dtheta_m_phys: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance global SCVT nonhydrostatic state `(rho_d, u_edge, w, theta_m)` over `dt_dyn` [s].

    Returns `(uh_final, omega_z_final)` matching `(mass_new - mass_old)` for Zalesak FCT.
    """
    nz = vgrid.nz
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    dc_e = mesh.dc_edge.unsqueeze(0)          # [1, N_edges]
    area_e = mesh.area_edge.unsqueeze(0)      # [1, N_edges]
    area_c = mesh.area_cell.unsqueeze(0)      # [1, N_cells]

    dz_c = state.dz_c                         # [Nz, N_cells]
    dz_c_e = 0.5 * (dz_c[:, c0] + dz_c[:, c1])
    dz_w_int = state.z_c[1:] - state.z_c[:-1] # [Nz-1, N_cells]
    dz_w_e = 0.5 * (dz_w_int[:, c0] + dz_w_int[:, c1])
    slope_z_e = (state.z_c[:, c1] - state.z_c[:, c0]) / dc_e  # Terrain coordinate slope (dz/dx)_zeta

    sponge_w = vgrid.sponge_w[1:-1].unsqueeze(1)
    sponge_c = vgrid.sponge_c.unsqueeze(1)

    # Scale-aware horizontal divergence damping & del^4 hyperviscosity (Skamarock et al., 2012, Sec. 3d)
    # Note: on hexagonal Voronoi grids dv_edge ~ dc_edge / sqrt(3), so max(|lambda(lap4)|) ~ 480 / dc_min^4;
    # setting nu_4 = 0.0002 * dc_min^4 / dt_dyn keeps |dt_dyn * nu_4 * lambda_max| < 0.10 unconditionally!
    dx_min = getattr(mesh, "_dc_edge_min_m", None)
    if dx_min is None:
        dx_min = float(mesh.dc_edge.min().item())
        setattr(mesh, "_dc_edge_min_m", dx_min)
    nu_div = 0.02 * (dx_min ** 2) / max(dt_dyn, 1.0)
    nu_4 = 0.0002 * (dx_min ** 4) / max(dt_dyn, 1.0)

    lap2_u0, grad_div_u0 = compute_vector_laplacian_edge(mesh, state.u_edge)
    lap4_u0, _ = compute_vector_laplacian_edge(mesh, lap2_u0)
    visc_u = nu_div * grad_div_u0 - nu_4 * lap4_u0

    lap2_th0 = compute_scalar_laplacian_cell(mesh, state.theta_m)
    lap4_th0 = compute_scalar_laplacian_cell(mesh, lap2_th0)
    visc_Thm = -(0.5 * nu_4) * (state.rho_d * dz_c) * lap4_th0

    m0 = state.rho_d * dz_c                   # Coupled dry mass [Nz, N_cells] [kg m^-2]
    Thm0 = m0 * state.theta_m                 # Coupled Theta_m [Nz, N_cells] [kg K m^-2]
    u0 = state.u_edge
    w0 = state.w
    q_load = 1.0 + state.q_n + state.q_p

    m_s = m0
    Thm_s = Thm0
    u_s = u0
    w_s = w0

    uh_final = torch.zeros_like(u0)
    om_final = torch.zeros_like(w0)
    gamma = C.GAMMA_D

    du_ext = du_phys_edge if du_phys_edge is not None else torch.zeros_like(u0)
    dThm_ext = (m0 * dtheta_m_phys) if dtheta_m_phys is not None else torch.zeros_like(Thm0)

    # Precompute stage-invariant edge/cell stencil views
    zw_slope_e = (state.z_w[1:-1, c1] - state.z_w[1:-1, c0]) / dc_e
    valid_ce = (mesh.edge_sign_on_cell != 0.0).to(u0.dtype)
    v0_e = mesh.verts_on_edge[:, 0]
    v1_e = mesh.verts_on_edge[:, 1]
    ep_idx = mesh.trisk_edges
    w_trisk = mesh.trisk_weights.unsqueeze(0)
    # Terrain-following acoustic Courant limit over steep Earth2014 Himalayas/Antarctica (dt_ac <= ~350 s on Glevel-3)
    dt_ac_max = max(0.16 * dx_min / 350.0, 1.0)

    rk_taus = (dt_dyn / 3.0, 0.5 * dt_dyn, dt_dyn)
    for stage_idx, dtau in enumerate(rk_taus):
        rho_s = m_s / dz_c
        thm_s = Thm_s / torch.clamp(m_s, min=1.0e-4)
        rho_thm_vol = Thm_s / dz_c
        pres_s = C.P_REF * torch.pow(
            torch.clamp(C.R_D * rho_thm_vol / C.P_REF, min=1.0e-7),
            gamma,
        )
        dp_dThm_s = gamma * pres_s / torch.clamp(Thm_s, min=1.0e-3)
        rtot_s = rho_s * q_load

        # Stage-frozen edge scalars & Zängl (2012) / Lin (1997) midpoint-altitude log-hydrostatic PGF factors
        m_edge = 0.5 * (m_s[:, c0] + m_s[:, c1])
        thm_e = 0.5 * (thm_s[:, c0] + thm_s[:, c1]) - 0.15 * torch.sign(u_s) * (thm_s[:, c1] - thm_s[:, c0])
        rtot_e = torch.clamp(0.5 * (rtot_s[:, c0] + rtot_s[:, c1]), min=1.0e-5)
        dz_10 = state.z_c[:, c1] - state.z_c[:, c0]
        fac_z0 = torch.exp(-0.5 * (rtot_s[:, c0] * C.GRAVITY / torch.clamp(pres_s[:, c0], min=10.0)) * dz_10)
        fac_z1 = torch.exp(+0.5 * (rtot_s[:, c1] * C.GRAVITY / torch.clamp(pres_s[:, c1], min=10.0)) * dz_10)

        # --- Slow Vector-Invariant Momentum RHS Evaluated Once per RK3 Stage ---
        zeta_v = compute_scvt_vorticity(mesh, u_s)
        if can_use_crm_metal(u_s):
            rot_acc = dispatch_scvt_trisk_rot_metal(mesh, zeta_v, u_s)
        else:
            eta_v = zeta_v + mesh.f_vert.unsqueeze(0)
            eta_e = 0.5 * (eta_v[:, v0_e] + eta_v[:, v1_e])
            eta_pair = 0.5 * (eta_e.unsqueeze(-1) + eta_e[:, ep_idx])
            rot_acc = (w_trisk * eta_pair * u_s[:, ep_idx]).sum(dim=-1)

        ke_edge = area_e * (u_s * u_s)
        ke_cell = 0.5 * (ke_edge[:, mesh.edges_on_cell] * valid_ce).sum(dim=-1) / area_c
        grad_ke = (ke_cell[:, c1] - ke_cell[:, c0]) / dc_e

        u_int_e = 0.5 * (u_s[:-1] + u_s[1:])
        uz_edge = area_e * u_int_e * zw_slope_e
        u_dot_gradz_s = (uz_edge[:, mesh.edges_on_cell] * valid_ce).sum(dim=-1) / area_c

        w_tilde_s = w_s[1:-1] - u_dot_gradz_s
        wt_e = 0.5 * (w_tilde_s[:, c0] + w_tilde_s[:, c1])
        du_dz_int = (u_s[1:] - u_s[:-1]) / dz_w_e
        vadv_u = torch.zeros_like(u_s)
        vadv_u[:-1] += -0.5 * wt_e * du_dz_int
        vadv_u[1:] += -0.5 * wt_e * du_dz_int

        tend_u_slow = rot_acc - grad_ke + vadv_u + visc_u + du_ext
        tend_Thm_slow = visc_Thm + dThm_ext

        # Stage-frozen vertical HEVI Helmholtz tridiagonal coefficients for step dt_ac
        n_ac = max(1, int(math.ceil(dtau / dt_ac_max)))
        dt_ac = dtau / float(n_ac)

        rho_dw = 0.5 * (rho_s[:-1] + rho_s[1:])
        rtot_w = torch.clamp(0.5 * (rtot_s[:-1] + rtot_s[1:]), min=1.0e-5)
        thm_w = 0.5 * (thm_s[:-1] + thm_s[1:])

        fac_p = ((beta_hevi * dt_ac) ** 2) * (rho_dw / (rtot_w * dz_w_int))
        fac_g = ((beta_hevi * dt_ac) ** 2) * (0.5 * C.GRAVITY * (rho_dw / rtot_w))
        c_dn = dp_dThm_s[:-1]
        c_up = dp_dThm_s[1:]

        diag = torch.clamp(
            1.0
            + dt_ac * sponge_w
            + fac_p * thm_w * (c_up + c_dn)
            + fac_g * (q_load[1:] / dz_c[1:] - q_load[:-1] / dz_c[:-1]),
            min=0.5,
        )
        dl = torch.zeros_like(diag)
        du = torch.zeros_like(diag)
        dl[1:] = -fac_p[1:] * c_dn[1:] * thm_w[:-1] + fac_g[1:] * (q_load[1:-1] / dz_c[1:-1])
        du[:-1] = -fac_p[:-1] * c_up[:-1] * thm_w[1:] - fac_g[:-1] * (q_load[1:-1] / dz_c[1:-1])

        # --- Fast Horizontal Acoustic + Vertical Implicit HEVI Loop from U^0 over n_ac steps of dt_ac ---
        udotz0 = u_dot_gradz_s if stage_idx == 0 else (
            ((area_e * 0.5 * (u0[:-1] + u0[1:]) * zw_slope_e)[:, mesh.edges_on_cell] * valid_ce).sum(dim=-1) / area_c
        )

        if u0.device.type != "mps":
            m_s, Thm_s, u_s, w_s, uh_accum, om_accum = _scvt_fast_acoustic_hevi_cpu_gpu(
                n_ac=n_ac,
                dt_ac=dt_ac,
                beta_hevi=beta_hevi,
                gravity=C.GRAVITY,
                accum_fluxes=(stage_idx == 2),
                m0=m0,
                Thm0=Thm0,
                u0=u0,
                w0=w0,
                udotz0=udotz0,
                pres_s=pres_s,
                dp_dThm_s=dp_dThm_s,
                Thm_s=Thm_s,
                fac_z0=fac_z0,
                fac_z1=fac_z1,
                rtot_e=rtot_e,
                dc_e=dc_e,
                tend_u_slow=tend_u_slow,
                tend_Thm_slow=tend_Thm_slow,
                sponge_c=sponge_c,
                sponge_w=sponge_w,
                area_e=area_e,
                zw_slope_e=zw_slope_e,
                valid_ce=valid_ce,
                area_c=area_c,
                m_edge=m_edge,
                thm_e=thm_e,
                rho_dw=rho_dw,
                rtot_w=rtot_w,
                thm_w=thm_w,
                dz_c=dz_c,
                dz_w_int=dz_w_int,
                q_load=q_load,
                dl=dl,
                diag=diag,
                du=du,
                c0=c0,
                c1=c1,
                dv_edge=mesh.dv_edge,
                edges_on_cell=mesh.edges_on_cell,
                edge_sign_on_cell=mesh.edge_sign_on_cell,
                area_cell=mesh.area_cell,
            )
        elif can_use_crm_metal(m0):
            m_s, Thm_s, u_s, w_s, uh_accum, om_accum = dispatch_scvt_fast_acoustic_metal(
                n_ac=n_ac,
                dt_ac=dt_ac,
                beta_hevi=beta_hevi,
                gravity=C.GRAVITY,
                accum_fluxes=(stage_idx == 2),
                m0=m0,
                Thm0=Thm0,
                u0=u0,
                w0=w0,
                udotz0=udotz0,
                pres_s=pres_s,
                dp_dThm_s=dp_dThm_s,
                Thm_ref=Thm_s,
                fac_z0=fac_z0,
                fac_z1=fac_z1,
                rtot_e=rtot_e,
                dc_e=dc_e,
                tend_u_slow=tend_u_slow,
                tend_Thm_slow=tend_Thm_slow,
                sponge_c=sponge_c,
                sponge_w=sponge_w,
                area_e=area_e,
                zw_slope_e=zw_slope_e,
                dv_edge=mesh.dv_edge,
                area_cell=mesh.area_cell,
                edges_on_cell=mesh.edges_on_cell,
                edge_sign_on_cell=mesh.edge_sign_on_cell,
                c0=c0,
                c1=c1,
                m_edge=m_edge,
                thm_e=thm_e,
                rho_dw=rho_dw,
                rtot_w=rtot_w,
                thm_w=thm_w,
                dz_c=dz_c,
                dz_w_int=dz_w_int,
                q_load=q_load,
                dl=dl,
                diag=diag,
                du=du,
            )
        else:
            m_k = m0
            Thm_k = Thm0
            u_k = u0
            w_k = w0
            udotz_k = udotz0
            uh_accum = torch.zeros_like(u0)
            om_accum = torch.zeros_like(w0)

            for _ in range(n_ac):
                p_k = pres_s + dp_dThm_s * (Thm_k - Thm_s)
                pgf_e = -(p_k[:, c1] * fac_z1 - p_k[:, c0] * fac_z0) / (rtot_e * dc_e)

                u_next = u_k + dt_ac * (tend_u_slow + pgf_e - sponge_c * u_k)
                u_int_next = 0.5 * (u_next[:-1] + u_next[1:])
                uz_next = area_e * u_int_next * zw_slope_e
                udotz_next = (uz_next[:, mesh.edges_on_cell] * valid_ce).sum(dim=-1) / area_c

                uh_k = m_edge * (1.10 * u_next - 0.10 * u_k)
                fTh_h_k = uh_k * thm_e
                div_h_m = compute_scvt_divergence(mesh, uh_k)
                div_h_Thm = compute_scvt_divergence(mesh, fTh_h_k)

                om_old = torch.zeros_like(w_k)
                om_old[1:-1] = rho_dw * (w_k[1:-1] - udotz_k)

                m_dag = m_k - dt_ac * (div_h_m + (om_old[1:] - om_old[:-1]))
                fTh_z_old = torch.zeros_like(w_k)
                fTh_z_old[1:-1] = thm_w * om_old[1:-1]
                Thm_dag = Thm_k + dt_ac * (-div_h_Thm - (fTh_z_old[1:] - fTh_z_old[:-1]) + tend_Thm_slow)

                p_dag = pres_s + dp_dThm_s * (Thm_dag - Thm_s)
                rho_k = m_k / dz_c
                rho_dag = m_dag / dz_c
                rtot_k_w = 0.5 * (rho_k[:-1] * q_load[:-1] + rho_k[1:] * q_load[1:])
                rtot_dag_w = 0.5 * (rho_dag[:-1] * q_load[:-1] + rho_dag[1:] * q_load[1:])

                p_blend = (1.0 - beta_hevi) * p_k + beta_hevi * p_dag
                rtot_blend_w = (1.0 - beta_hevi) * rtot_k_w + beta_hevi * rtot_dag_w
                nh_imbal = (p_blend[1:] - p_blend[:-1]) / dz_w_int + rtot_blend_w * C.GRAVITY

                fz_phys_k = rho_dw * w_k[1:-1]
                rhs_om = (
                    -rho_dw * (udotz_next - udotz_k)
                    + dt_ac * (-(rho_dw / rtot_w) * nh_imbal - sponge_w * fz_phys_k)
                )
                dom = solve_tridiagonal_batched(dl, diag, du, rhs_om)

                om_new = om_old.clone()
                om_new[1:-1] = om_old[1:-1] + dom

                m_next = torch.clamp(
                    m_dag - beta_hevi * dt_ac * (om_new[1:] - om_old[1:] - (om_new[:-1] - om_old[:-1])),
                    min=1.0e-3,
                )
                dfTh_z = torch.zeros_like(w_k)
                dfTh_z[1:-1] = thm_w * dom
                Thm_next = Thm_dag - beta_hevi * dt_ac * (dfTh_z[1:] - dfTh_z[:-1])

                w_next = torch.zeros_like(w_k)
                w_next[1:-1] = udotz_next + om_new[1:-1] / torch.clamp(rho_dw, min=1.0e-5)

                if stage_idx == 2:
                    uh_accum = uh_accum + uh_k
                    om_accum = om_accum + ((1.0 - beta_hevi) * om_old + beta_hevi * om_new)

                m_k = m_next
                Thm_k = Thm_next
                u_k = u_next
                w_k = w_next
                udotz_k = udotz_next

            m_s = m_k
            Thm_s = Thm_k
            u_s = u_k
            w_s = w_k

        if stage_idx == 2:
            inv_nac = 1.0 / float(n_ac)
            uh_final = uh_accum * inv_nac
            om_final = om_accum * inv_nac

    state.rho_d = m_s / dz_c
    state.u_edge = u_s
    state.w = w_s
    state.theta_m = Thm_s / m_s

    return uh_final, om_final


# ---------------------------------------------------------------------------
# 5. Held & Suarez (1994) Global Benchmark Forcing
# ---------------------------------------------------------------------------
def compute_held_suarez_forcing(
    state: GlobalAtmosState,
    mesh: SCVTMesh,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Held & Suarez (1994, BAMS, 75, Eqs. 1-4) `(du_edge_dt, dtheta_m_dt)`."""
    pres, exner, temp = diagnose_pressure_temperature(state.rho_d, state.theta_m, state.q_v)
    dz0 = state.z_c[0] - state.z_w[0]
    p_sfc = pres[0] + C.GRAVITY * state.rho_d[0] * dz0
    sigma = torch.clamp(pres / p_sfc.unsqueeze(0), 0.0, 1.0)

    # Boundary-layer Rayleigh drag below sigma_b = 0.7 (k_f = 1 day^-1)
    kf = 1.0 / C.SECONDS_PER_DAY
    kv_cell = kf * torch.clamp((sigma - 0.7) / (1.0 - 0.7), min=0.0)
    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    kv_edge = 0.5 * (kv_cell[:, c0] + kv_cell[:, c1])
    du_hs = -kv_edge * state.u_edge

    # Newtonian thermal relaxation toward T_eq(lat, p) (k_a = 1/40 d^-1, k_s = 1/4 d^-1)
    lat = mesh.lat_cell.unsqueeze(0)
    sin2 = torch.sin(lat) ** 2
    cos2 = torch.cos(lat) ** 2
    cos4 = cos2 * cos2

    ka = 1.0 / (40.0 * C.SECONDS_PER_DAY)
    ks = 1.0 / (4.0 * C.SECONDS_PER_DAY)
    kt = ka + (ks - ka) * torch.clamp((sigma - 0.7) / (1.0 - 0.7), min=0.0) * cos4

    p_norm = torch.clamp(pres / C.P_REF, min=1.0e-5)
    theta_eq_raw = 315.0 - 60.0 * sin2 - 10.0 * torch.log(p_norm) * cos2
    t_eq = torch.clamp(theta_eq_raw * exner, min=200.0)
    theta_eq = t_eq / exner

    dtheta_hs = -kt * (state.theta_m - theta_eq)
    return du_hs, dtheta_hs
