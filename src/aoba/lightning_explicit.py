"""Explicit Bulk Cloud Electrification & Lightning Discharge Model for AOBA (SCALE-LT Class).

Prognoses volumetric electric charge densities across five condensed hydrometeor species
(cloud water, rain, cloud ice, snow, and graupel) inside the embedded 2D/3D CRM, computes
non-inductive riming electrification with temperature–LWC polarity reversal (Takahashi, 1978;
Saunders & Peck, 1998; Sato et al., 2019), advects and sediments charges conservatively,
solves the exact 2D/3D periodic-horizontal / Dirichlet-vertical Poisson equation for electric
potential via batched 2D FFT + Thomas tridiagonal elimination, executes dielectric-breakdown
neutralization discharges (Fierro et al., 2013; Sato et al., 2019), classifies IC vs. CG
flashes, and outputs mass-conserving natural LNOx deposition (Schumann & Huntrieser, 2007).

Zero geographic or land/ocean lookup maps are used; all charging structures (dipoles, tripoles,
lower positive charge centers, and IC/CG ratios) emerge directly from CRM microphysics and
updraft dynamics. Fully batched on GPU (MPS / CUDA) and CPU with zero per-cell or per-layer
Python loops.

References
----------
- Takahashi, T. (1978): "Riming electrification as a charge generation mechanism in
  thunderstorms", J. Atmos. Sci., 35(8), 1536-1548.
- Chiu, C.-S. (1978): "Numerical study of cloud electrification in an axisymmetric,
  time-dependent cloud model", J. Geophys. Res., 83(C10), 5025-5049.
- Saunders, C. P. R., & Peck, S. L. (1998): "Laboratory studies of the influence of the rime
  accretion rate on charge transfer during crystal/graupel collisions", J. Geophys. Res.,
  103(D12), 13949-13956.
- Rakov, V. A., & Uman, M. A. (2003): *Lightning: Physics and Effects*, Cambridge Univ. Press.
- Dwyer, J. R. (2003): "A fundamental limit on electric fields in air", Geophys. Res. Lett.,
  30(20), 2055.
- Mansell, E. R., MacGorman, D. R., Ziegler, C. L., & Straka, J. M. (2005): "Charge structure
  and lightning sensitivity in a simulated multicell thunderstorm", J. Geophys. Res., 110, D12101.
- Schumann, U., & Huntrieser, H. (2007): "The global lightning-induced nitrogen oxides source",
  Atmos. Chem. Phys., 7, 3823-3907.
- Fierro, A. O., Mansell, E. R., MacGorman, D. R., & Ziegler, C. L. (2013): "The implementation
  of an explicit charging and discharge lightning scheme within the WRF-ARW model", Mon. Wea. Rev.,
  141(7), 2390-2409.
- Sato, Y., Hayashi, S., & Hashimoto, A. (2019): "Sensitivity of explicit bulk lightning model
  to cloud microphysics schemes in SCALE-RM", J. Geophys. Res. Atmospheres, 124(17-18), 10200-10218.

AOBA Formulation Reference: Sec. 5.6.2, Eqs. (5.19)-(5.23) (Explicit Bulk Cloud Electrification & Discharge).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import solve_tridiagonal_batched
from aoba.lightning import (
    LNOX_KG_N_PER_FLASH,
    LNOX_LOCAL_DEP_FRAC,
    LightningDiagnostics,
)
from aoba.microphysics import (
    RHO_GRAUPEL,
    RHO_RAIN,
    RHO_SNOW,
    T0_GRAUPEL,
    T00_GRAUPEL,
    T0_PRECIP,
    T00_PRECIP,
)
from aoba.thermodynamics import diagnose_pressure_temperature


# Physical constants for explicit bulk electrification & discharge
EPSILON_0: float = 8.854187817e-12      # Vacuum permittivity [F m^-1]
RHO_AIR_REF: float = 1.225              # Reference sea-level air density [kg m^-3]
E_CRIT_SEA_LEVEL: float = 1.2e5         # Dwyer (2003) / Sato et al. (2019) breakeven threshold at sea level [V m^-1]
E_DISCHARGE_EXTENT_FRAC: float = 0.30   # Channel propagation threshold fraction of E_crit (Fierro et al., 2013) [-]
DISCHARGE_NEUT_FRAC: float = 0.30       # Fraction of excess charge neutralized per discharge pass (Fierro et al., 2013) [-]
Q_CHARGE_PER_FLASH_C: float = 15.0      # Typical charge neutralized per lightning flash (Rakov & Uman, 2003) [C flash^-1]
DQ_MAX_TAKAHASHI_C: float = 2.5e-14     # Peak charge separated per rebounding graupel-ice collision (25 fC; Takahashi, 1978) [C]
REBOUND_FRAC_GI: float = 0.90           # Non-sticking rebound fraction for graupel-ice/snow collisions [-]
TAU_ION_RELAX_S: float = 600.0          # Fair-weather atmospheric ionic conductivity relaxation timescale (Chiu, 1978) [s]
Z_CG_CHANNEL_MAX_M: float = 3500.0      # Maximum AGL height of lower breakdown core to connect to ground (CG) [m]
RHO_CLOUD_ICE: float = 500.0            # Effective bulk crystal density for cloud ice diameter diagnostic [kg m^-3]
Q_CHG_CLAMP_C_M3: float = 5.0e-8        # Physical upper bound on per-species space charge density (50 nC m^-3) [C m^-3]


@dataclass
class ExplicitLightningState:
    """Prognostic 3D hydrometeor charge densities and electric-field diagnostics on the CRM grid.

    Species axis (`dim=0`) order of `q_chg` (`[5, Nz, N_cells, Ny, Nx]` [C m^-3]):
      0: cloud water (`q_c`)
      1: rain (`q_r`)
      2: cloud ice (`q_i`)
      3: snow (`q_s`)
      4: graupel (`q_g`)
    """

    q_chg: torch.Tensor                  # [5, Nz, N_cells, Ny, Nx] volumetric charge density [C m^-3]
    phi_e: torch.Tensor                  # [Nz, N_cells, Ny, Nx] electric potential [V]
    e_mag: torch.Tensor                  # [Nz, N_cells, Ny, Nx] electric field magnitude |E| [V m^-1]
    e_z: torch.Tensor                    # [Nz, N_cells, Ny, Nx] vertical electric field E_z [V m^-1]
    last_flash_ic_s: float = 0.0         # Latest global IC flash rate [s^-1]
    last_flash_cg_s: float = 0.0         # Latest global CG flash rate [s^-1]
    last_e_max_kv_m: float = 0.0         # Latest global peak electric field [kV m^-1]


def init_explicit_lightning_state(
    nz: int,
    n_cells: int,
    ny: int,
    nx: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ExplicitLightningState:
    """Allocate zero-initialized neutral `ExplicitLightningState` matching the CRM grid."""
    dev = torch.device(device)
    z4 = torch.zeros((nz, n_cells, ny, nx), device=dev, dtype=dtype)
    return ExplicitLightningState(
        q_chg=torch.zeros((5, nz, n_cells, ny, nx), device=dev, dtype=dtype),
        phi_e=z4.clone(),
        e_mag=z4.clone(),
        e_z=z4.clone(),
    )


# AOBA Formulation Eq. (5.19)
def takahashi_charge_per_collision(
    temp_k: torch.Tensor,
    lwc_g_m3: torch.Tensor,
) -> torch.Tensor:
    """Smooth differentiable Takahashi (1978) / Sato et al. (2019) charge transferred to graupel [C].

    Returns positive values (`graupel +`, `ice/snow -`) in warm/high-LWC wet-riming zones
    (forming lower positive charge centers) and negative values (`graupel -`, `ice/snow +`)
    in colder/moderate-LWC dry-riming zones (forming the main mid-level negative / upper-level
    positive thunderstorm dipole). Vanishes smoothly outside the supercooled mixed-phase zone
    (-40 C < T < 0 C, LWC > 0).
    """
    t_c = temp_k - C.T_FREEZE
    lwc_pos = torch.clamp(lwc_g_m3, min=0.0)
    # Charge-reversal temperature T_rev(LWC) [degC]: ~ -8 C at low LWC, dropping to ~ -19 C at high LWC
    t_rev_c = -10.0 - 10.0 * torch.tanh(0.65 * (lwc_pos - 0.55))
    polarity = torch.tanh((t_c - t_rev_c) / 4.0)
    lwc_fac = lwc_pos / (lwc_pos + 0.15)
    # Smooth mixed-phase envelope active strictly below 0 C and above -40 C
    t_gate = torch.sigmoid((t_c + 40.0) / 3.0) * torch.sigmoid(-t_c / 1.5) * (t_c < 0.0).to(temp_k.dtype)
    return DQ_MAX_TAKAHASHI_C * polarity * lwc_fac * t_gate


# AOBA Formulation Eq. (5.21)
def solve_electric_potential_poisson(
    rho_q_tot: torch.Tensor,
    dz_c: torch.Tensor,
    dx: float,
    dy: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve exact 2D/3D periodic-horizontal Dirichlet-vertical Poisson equation `nabla^2 Phi = -rho_q / eps_0`.

    Uses a single non-iterative batched 2D real FFT (`rfft2`) in `(y, x)` coupled to the exact
    stretched-grid 1D tridiagonal Thomas solver (`solve_tridiagonal_batched`) in `z` with
    grounded lower (`Phi(0) = 0`) and upper (`Phi(z_top) = 0`) Dirichlet boundaries.

    Returns
    -------
    phi_e : Tensor `[Nz, N_cells, Ny, Nx]` [V]
    e_x, e_y, e_z : Tensors `[Nz, N_cells, Ny, Nx]` [V m^-1]
    """
    nz, nc, ny, nx = rho_q_tot.shape
    dev = rho_q_tot.device
    dtype = rho_q_tot.dtype
    dx_m = max(float(dx), 1.0)
    dy_m = max(float(dx if dy is None else dy), 1.0)

    # Vertical layer thicknesses [Nz, N_cells_or_1] -> [Nz, nc]
    if dz_c.ndim == 2:
        dz2 = dz_c.to(device=dev, dtype=dtype)
    else:
        dz2 = dz_c.to(device=dev, dtype=dtype).view(nz, 1).expand(nz, nc)
    dz2 = torch.clamp(dz2, min=1.0)

    # Interface distances between adjacent layer centers (and half-layer to Dirichlet boundaries)
    dz_dn = torch.empty_like(dz2)
    dz_up = torch.empty_like(dz2)
    dz_dn[0] = 0.5 * dz2[0]
    if nz > 1:
        dz_mid = 0.5 * (dz2[:-1] + dz2[1:])
        dz_dn[1:] = dz_mid
        dz_up[:-1] = dz_mid
    dz_up[-1] = 0.5 * dz2[-1]

    c_dn = 1.0 / dz_dn  # [Nz, nc]
    c_up = 1.0 / dz_up  # [Nz, nc]

    # Exact discrete periodic Laplacian eigenvalues lambda_{m,n} on [Ny, Nx//2 + 1]
    n_kx = nx // 2 + 1
    kx_idx = torch.arange(n_kx, device=dev, dtype=dtype)
    ky_idx = torch.arange(ny, device=dev, dtype=dtype)
    lam_x = (4.0 / (dx_m * dx_m)) * torch.sin(math.pi * kx_idx / float(max(nx, 1))) ** 2
    lam_y = (4.0 / (dy_m * dy_m)) * torch.sin(math.pi * ky_idx / float(max(ny, 1))) ** 2 if ny > 1 else torch.zeros_like(ky_idx)
    lam_h = (lam_y.view(ny, 1) + lam_x.view(1, n_kx)).reshape(1, 1, ny * n_kx)  # [1, 1, M_k]
    m_k = ny * n_kx

    # 2D real FFT of charge density along (y, x): [Nz, nc, ny, n_kx]
    rhs_phys = rho_q_tot * (dz2.view(nz, nc, 1, 1) / EPSILON_0)
    rhs_hat = torch.fft.rfft2(rhs_phys, dim=(-2, -1))  # complex64

    # Pack real and imaginary parts into one batched real tridiagonal solve [Nz, 2 * nc * m_k]
    rhs_ri = torch.stack([rhs_hat.real, rhs_hat.imag], dim=1).reshape(nz, 2 * nc * m_k)

    # Build real tridiagonal coefficients: -c_dn*Phi_{k-1} + (c_dn + c_up + lam*dz)*Phi_k - c_up*Phi_{k+1} = rhs
    a_col = -c_dn.clone()
    a_col[0] = 0.0
    d_col = -c_up.clone()
    d_col[-1] = 0.0
    b_base = (c_dn + c_up).unsqueeze(-1) + dz2.unsqueeze(-1) * lam_h  # [Nz, nc, m_k]

    # Expand across real/imag pair (factor 2)
    a_tri = a_col.unsqueeze(-1).expand(nz, nc, m_k).unsqueeze(1).expand(nz, 2, nc, m_k).reshape(nz, 2 * nc * m_k).contiguous()
    b_tri = b_base.unsqueeze(1).expand(nz, 2, nc, m_k).reshape(nz, 2 * nc * m_k).contiguous()
    d_tri = d_col.unsqueeze(-1).expand(nz, nc, m_k).unsqueeze(1).expand(nz, 2, nc, m_k).reshape(nz, 2 * nc * m_k).contiguous()

    phi_ri = solve_tridiagonal_batched(a_tri, b_tri, d_tri, rhs_ri.contiguous()).view(nz, 2, nc, ny, n_kx)
    phi_hat = torch.complex(phi_ri[:, 0], phi_ri[:, 1])
    phi_e = torch.fft.irfft2(phi_hat, s=(ny, nx), dim=(-2, -1)).to(dtype)

    # 3D Electric Field E = -nabla Phi
    if nx > 1:
        e_x = -(torch.roll(phi_e, shifts=-1, dims=-1) - torch.roll(phi_e, shifts=1, dims=-1)) / (2.0 * dx_m)
    else:
        e_x = torch.zeros_like(phi_e)
    if ny > 1:
        e_y = -(torch.roll(phi_e, shifts=-1, dims=-2) - torch.roll(phi_e, shifts=1, dims=-2)) / (2.0 * dy_m)
    else:
        e_y = torch.zeros_like(phi_e)

    # Vertical E_z = -dPhi/dz respecting Dirichlet Phi(0) = 0 and Phi(z_top) = 0
    e_z = torch.empty_like(phi_e)
    if nz == 1:
        e_z[0] = 0.0
    else:
        # Interior centered derivative via interface fluxes
        flux_up = (phi_e[1:] - phi_e[:-1]) / dz_up[:-1].view(nz - 1, nc, 1, 1)
        flux_bot = phi_e[0] / dz_dn[0].view(nc, 1, 1)      # (Phi_0 - 0) / (dz_0 / 2)
        flux_top = -phi_e[-1] / dz_up[-1].view(nc, 1, 1)   # (0 - Phi_{nz-1}) / (dz_{nz-1} / 2)
        e_z[0] = -0.5 * (flux_bot + flux_up[0])
        if nz > 2:
            e_z[1:-1] = -0.5 * (flux_up[:-1] + flux_up[1:])
        e_z[-1] = -0.5 * (flux_up[-1] + flux_top)

    return phi_e, e_x, e_y, e_z


def _advect_sediment_charges(
    q_chg: torch.Tensor,          # [5, Nz, N_cells, Ny, Nx]
    u_stag: torch.Tensor | None,  # [Nz, N_cells, Ny, Nx]
    v_stag: torch.Tensor | None,  # [Nz, N_cells, Ny, Nx]
    w_stag: torch.Tensor,         # [Nz+1, N_cells, Ny, Nx]
    vt_5: torch.Tensor,           # [5, Nz, N_cells, Ny, Nx] downward fall speeds >= 0 [m s^-1]
    dz_c: torch.Tensor,           # [Nz, N_cells] or [Nz]
    dx: float,
    dy: float,
    dt: float,
) -> torch.Tensor:
    """Conservatively transport 5 hydrometeor charges via CFL-safe horizontal donor-cell & implicit vertical sweep."""
    ns, nz, nc, ny, nx = q_chg.shape
    dev = q_chg.device
    dtype = q_chg.dtype
    q_out = q_chg

    # 1. Conservative periodic horizontal donor-cell advection (CFL-capped to <= 0.45 per axis)
    if u_stag is not None and nx > 1:
        u_face = u_stag.unsqueeze(0)  # face i+1/2 [1, Nz, nc, ny, nx]
        c_max_x = 0.45 * dx / max(dt, 1.0e-3)
        u_eff = torch.clamp(u_face, -c_max_x, c_max_x)
        q_xp1 = torch.roll(q_out, shifts=-1, dims=-1)
        fx = torch.clamp(u_eff, min=0.0) * q_out + torch.clamp(u_eff, max=0.0) * q_xp1
        q_out = q_out - (dt / dx) * (fx - torch.roll(fx, shifts=1, dims=-1))
    if v_stag is not None and ny > 1:
        v_face = v_stag.unsqueeze(0)
        c_max_y = 0.45 * dy / max(dt, 1.0e-3)
        v_eff = torch.clamp(v_face, -c_max_y, c_max_y)
        q_yp1 = torch.roll(q_out, shifts=-1, dims=-2)
        fy = torch.clamp(v_eff, min=0.0) * q_out + torch.clamp(v_eff, max=0.0) * q_yp1
        q_out = q_out - (dt / dy) * (fy - torch.roll(fy, shifts=1, dims=-2))

    # 2. Unconditionally stable implicit vertical advection + gravitational sedimentation
    #    Net vertical velocity at interfaces k+1/2 (k = 0..nz-2): w_net = w_stag[1:-1] - 0.5*(vt[k] + vt[k+1])
    if dz_c.ndim == 2:
        dz_5 = dz_c.to(device=dev, dtype=dtype).view(1, nz, nc, 1, 1).expand(ns, nz, nc, ny, nx)
    else:
        dz_5 = dz_c.to(device=dev, dtype=dtype).view(1, nz, 1, 1, 1).expand(ns, nz, nc, ny, nx)
    dz_5 = torch.clamp(dz_5, min=1.0)

    if nz > 1:
        w_int = w_stag[1:-1].unsqueeze(0) - 0.5 * (vt_5[:, :-1] + vt_5[:, 1:])  # [5, nz-1, nc, ny, nx]
        w_pos = torch.clamp(w_int, min=0.0)
        w_neg = torch.clamp(w_int, max=0.0)  # <= 0 (downward)
        # Precipitation charge flux reaching ground at k = 0 (open bottom boundary for falling hydrometeors)
        v_bot = torch.clamp(vt_5[:, 0], min=0.0)  # [5, nc, ny, nx]

        z_top_row = torch.zeros_like(q_out[:, 0:1])
        # Continuity: dz_k * q_k^{n+1} + dt*(F_{k+1/2} - F_{k-1/2}) = dz_k * q_k^n
        # where F_{k+1/2} = w_pos * q_k + w_neg * q_{k+1}, F_{-1/2} = -v_bot * q_0, F_{nz-1/2} = 0
        dl = torch.cat([z_top_row, -dt * w_pos], dim=1)  # coefficient of q_{k-1}
        du = torch.cat([dt * w_neg, z_top_row], dim=1)   # coefficient of q_{k+1}
        out_up = torch.cat([dt * w_pos, z_top_row], dim=1)
        out_dn = torch.cat([dt * v_bot.unsqueeze(1), -dt * w_neg], dim=1)
        diag = dz_5 + out_up + out_dn
        rhs = dz_5 * q_out

        n_col_tot = ns * nc * ny * nx
        q_new = solve_tridiagonal_batched(
            dl.permute(1, 0, 2, 3, 4).reshape(nz, n_col_tot).contiguous(),
            diag.permute(1, 0, 2, 3, 4).reshape(nz, n_col_tot).contiguous(),
            du.permute(1, 0, 2, 3, 4).reshape(nz, n_col_tot).contiguous(),
            rhs.permute(1, 0, 2, 3, 4).reshape(nz, n_col_tot).contiguous(),
        ).view(nz, ns, nc, ny, nx).permute(1, 0, 2, 3, 4).contiguous()
        return q_new
    return q_out


# AOBA Formulation Eqs. (5.19)-(5.23)
def step_explicit_bulk_lightning_and_lnox(
    state: ExplicitLightningState,
    rho_d: torch.Tensor,
    theta_m: torch.Tensor,
    q_v: torch.Tensor,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    q_p: torch.Tensor,
    w_stag: torch.Tensor,
    z_c: torch.Tensor,
    dz_c: torch.Tensor,
    area_cell: torch.Tensor,
    dx: float,
    dt: float,
    dy: float | None = None,
    u_stag: torch.Tensor | None = None,
    v_stag: torch.Tensor | None = None,
    q_r_prog: torch.Tensor | None = None,
    q_s_prog: torch.Tensor | None = None,
    q_g_prog: torch.Tensor | None = None,
    n_i_prog: torch.Tensor | None = None,
    n_s_prog: torch.Tensor | None = None,
    n_g_prog: torch.Tensor | None = None,
) -> LightningDiagnostics:
    """Advance SCALE-LT-class explicit bulk hydrometeor electrification, Poisson field, & discharges.

    Works seamlessly with `six_class_2m` (using prognostic `q_r, q_s, q_g, n_i, n_s, n_g`),
    `two_moment`, and `one_moment` microphysics, on both 2D (`ny = 1`) and 3D (`ny > 1`) CRMs.
    Returns a standard `LightningDiagnostics` object compatible with AOBA wildfire, ozone, and
    biogeochemical nitrogen/iron coupling.
    """
    nz, nc, ny, nx = rho_d.shape
    dev = rho_d.device
    dtype = rho_d.dtype
    dx_m = max(float(dx), 1.0)
    dy_m = max(float(dx if dy is None else dy), 1.0)
    dt_s = max(float(dt), 1.0)

    _, _, temp = diagnose_pressure_temperature(rho_d, theta_m, q_v)
    rho_pos = torch.clamp(rho_d, min=1.0e-4)

    # 1. Resolve 5 condensed hydrometeor mixing ratios (using true 6-class prognostic fields when active)
    q_c_pos = torch.clamp(q_c, min=0.0)
    q_i_pos = torch.clamp(q_i, min=0.0)
    if q_g_prog is not None:
        q_r = torch.clamp(q_r_prog, min=0.0) if q_r_prog is not None else torch.zeros_like(q_c_pos)
        q_s = torch.clamp(q_s_prog, min=0.0) if q_s_prog is not None else torch.zeros_like(q_c_pos)
        q_g = torch.clamp(q_g_prog, min=0.0)
    else:
        f_ice_p = torch.clamp((T0_PRECIP - temp) / max(T0_PRECIP - T00_PRECIP, 1.0), 0.0, 1.0)
        f_graup = torch.clamp((temp - T00_GRAUPEL) / max(T0_GRAUPEL - T00_GRAUPEL, 1.0), 0.0, 1.0)
        q_p_pos = torch.clamp(q_p, min=0.0)
        q_r = (1.0 - f_ice_p) * q_p_pos
        q_p_ice = f_ice_p * q_p_pos
        q_g = q_p_ice * f_graup
        q_s = q_p_ice * (1.0 - f_graup)

    # 2. Estimate bulk number concentrations [m^-3], mean diameters [m], and fall speeds [m s^-1]
    #    (using prognostic 2-moment number concentrations when available, else standard bulk relations)
    m_r = rho_pos * q_r
    m_i = rho_pos * q_i_pos
    m_s = rho_pos * q_s
    m_g = rho_pos * q_g

    rho_fac = torch.pow(RHO_AIR_REF / rho_pos, 0.5)

    # Cloud ice
    if n_i_prog is not None:
        n_i_v = torch.clamp(rho_pos * n_i_prog, min=1.0)
    else:
        n_i_v = torch.clamp(1.0e5 * torch.pow(torch.clamp(m_i * 1.0e3, min=0.0), 0.5), min=1.0)
    d_i = torch.clamp(torch.pow(6.0 * m_i / (math.pi * RHO_CLOUD_ICE * n_i_v) + 1.0e-20, 1.0 / 3.0), 2.0e-5, 5.0e-4)
    vt_i = torch.where(q_i_pos > 1.0e-7, torch.clamp(0.45 * rho_fac, 0.05, 1.5), torch.zeros_like(q_i_pos))

    # Snow
    if n_s_prog is not None:
        n_s_v = torch.clamp(rho_pos * n_s_prog, min=1.0)
    else:
        n_s_v = torch.clamp(3.0e6 * torch.pow(torch.clamp(m_s / (math.pi * RHO_SNOW * 3.0e6) + 1.0e-24, min=1.0e-24), 0.25), min=1.0)
    d_s = torch.clamp(torch.pow(6.0 * m_s / (math.pi * RHO_SNOW * n_s_v) + 1.0e-20, 1.0 / 3.0), 1.0e-4, 6.0e-3)
    vt_s = torch.where(q_s > 1.0e-7, torch.clamp(4.84 * torch.pow(d_s, 0.25) * rho_fac, 0.2, 3.0), torch.zeros_like(q_s))

    # Graupel
    if n_g_prog is not None:
        n_g_v = torch.clamp(rho_pos * n_g_prog, min=1.0)
    else:
        n_g_v = torch.clamp(4.0e6 * torch.pow(torch.clamp(m_g / (math.pi * RHO_GRAUPEL * 4.0e6) + 1.0e-24, min=1.0e-24), 0.25), min=1.0)
    d_g = torch.clamp(torch.pow(6.0 * m_g / (math.pi * RHO_GRAUPEL * n_g_v) + 1.0e-20, 1.0 / 3.0), 2.0e-4, 1.5e-2)
    vt_g = torch.where(q_g > 1.0e-7, torch.clamp(94.5 * torch.pow(d_g, 0.5) * rho_fac, 0.5, 12.0), torch.zeros_like(q_g))

    # Rain
    d_r = torch.clamp(torch.pow(m_r / (math.pi * RHO_RAIN * 8.0e6) + 1.0e-24, 0.25), 1.0e-4, 5.0e-3)
    vt_r = torch.where(q_r > 1.0e-7, torch.clamp(842.0 * torch.pow(d_r, 0.8) * rho_fac, 0.5, 10.0), torch.zeros_like(q_r))
    vt_c = torch.zeros_like(q_c_pos)

    # 3. Non-inductive riming electrification (Takahashi, 1978; Sato et al., 2019, Sec. 2.1)
    #    Strict zero-sum transfer: d(rho_q_g) + d(rho_q_i) + d(rho_q_s) == 0 identically!
    lwc_g_m3 = 1.0e3 * rho_pos * q_c_pos
    dq_coll = takahashi_charge_per_collision(temp, lwc_g_m3)  # [C per collision]

    active_g = (q_g > 1.0e-6).to(dtype)
    active_i = (q_i_pos > 1.0e-7).to(dtype)
    active_s = (q_s > 1.0e-6).to(dtype)

    # Gravitational sweep-out collision kernels [collisions m^-3 s^-1]
    r_gi = REBOUND_FRAC_GI * (0.25 * math.pi) * ((d_g + d_i) ** 2) * torch.abs(vt_g - vt_i) * n_g_v * n_i_v * active_g * active_i
    r_gs = REBOUND_FRAC_GI * (0.25 * math.pi) * ((d_g + d_s) ** 2) * torch.abs(vt_g - vt_s) * n_g_v * n_s_v * active_g * active_s

    dq_i_sep = torch.clamp(-dt_s * r_gi * dq_coll, -Q_CHG_CLAMP_C_M3, Q_CHG_CLAMP_C_M3)
    dq_s_sep = torch.clamp(-dt_s * r_gs * dq_coll, -Q_CHG_CLAMP_C_M3, Q_CHG_CLAMP_C_M3)
    dq_g_sep = -(dq_i_sep + dq_s_sep)  # exact conservation of zero net charge creation!

    q_chg = state.q_chg.clone()
    q_chg[2] = q_chg[2] + dq_i_sep
    q_chg[3] = q_chg[3] + dq_s_sep
    q_chg[4] = q_chg[4] + dq_g_sep

    # 4. Conservative 3D transport & implicit gravitational sedimentation separating opposite charges
    vt_5 = torch.stack([vt_c, vt_r, vt_i, vt_s, vt_g], dim=0)
    q_chg = _advect_sediment_charges(q_chg, u_stag, v_stag, w_stag, vt_5, dz_c, dx_m, dy_m, dt_s)

    # 5. Melting / complete-evaporation transfer & exact exponential ionic relaxation (Chiu, 1978)
    #    Melting graupel/snow above 0 C transfers charge into rain; absent hydrometeors neutralize
    melt_frac = torch.clamp((temp - C.T_FREEZE) / 3.0, 0.0, 1.0) * (1.0 - math.exp(-dt_s / 180.0))
    q_chg[1] = q_chg[1] + melt_frac * (q_chg[3] + q_chg[4])
    q_chg[3] = q_chg[3] * (1.0 - melt_frac)
    q_chg[4] = q_chg[4] * (1.0 - melt_frac)

    q_mass_5 = torch.stack([q_c_pos, q_r, q_i_pos, q_s, q_g], dim=0)
    has_hydro = (q_mass_5 > 1.0e-8).to(dtype)
    decay_ion = math.exp(-dt_s / TAU_ION_RELAX_S)
    decay_dry = math.exp(-dt_s / 60.0)
    q_chg = torch.clamp(q_chg * (has_hydro * decay_ion + (1.0 - has_hydro) * decay_dry), -Q_CHG_CLAMP_C_M3, Q_CHG_CLAMP_C_M3)

    # 6. Solve exact 2D/3D Poisson equation for electric potential & 3D electric field
    rho_q_tot = q_chg.sum(dim=0)
    phi_e, e_x, e_y, e_z = solve_electric_potential_poisson(rho_q_tot, dz_c, dx_m, dy_m)
    e_mag = torch.sqrt(e_x * e_x + e_y * e_y + e_z * e_z + 1.0e-12)

    # 7. Dielectric breakdown discharge & IC/CG classification (Fierro et al., 2013; Sato et al., 2019)
    e_crit = E_CRIT_SEA_LEVEL * torch.clamp(rho_pos / RHO_AIR_REF, min=0.05)
    e_ratio = e_mag / e_crit
    # Trigger per host CRM domain [N_cells]
    domain_max_ratio = e_ratio.amax(dim=(0, 2, 3))  # [nc]
    triggered = (domain_max_ratio >= 1.0).view(1, nc, 1, 1)

    # Discharge region within triggered domains where |E| >= E_DISCHARGE_EXTENT_FRAC * E_crit
    in_channel = triggered & (e_ratio >= E_DISCHARGE_EXTENT_FRAC)

    if z_c.ndim == 2:
        zc_4d = z_c.to(device=dev, dtype=dtype).view(nz, nc, 1, 1)
        dz_4d = dz_c.to(device=dev, dtype=dtype).view(nz, nc, 1, 1)
    else:
        zc_4d = z_c.to(device=dev, dtype=dtype).view(nz, 1, 1, 1)
        dz_4d = dz_c.to(device=dev, dtype=dtype).view(nz, 1, 1, 1)
    z_agl = zc_4d - zc_4d[0:1] + 0.5 * dz_4d[0:1]

    # CG discharge criterion: breakdown channel reaches below Z_CG_CHANNEL_MAX_M AGL or strong surface E_z
    low_channel = in_channel & (z_agl <= Z_CG_CHANNEL_MAX_M)
    sfc_ez_ratio = torch.abs(e_z[0]) / torch.clamp(e_crit[0], min=1.0)
    is_cg_sub = (low_channel.any(dim=0)) | (triggered.squeeze(0) & (sfc_ez_ratio >= 0.25))  # [nc, ny, nx]
    is_cg_dom = is_cg_sub.any(dim=(-2, -1)).view(1, nc, 1, 1)

    # Neutralize fraction DISCHARGE_NEUT_FRAC of space charge inside discharge channel
    # Excess scaling increases smoothly with over-threshold ratio so severe storms discharge multiple flashes/step
    neut_eff = torch.clamp(DISCHARGE_NEUT_FRAC * torch.clamp(domain_max_ratio, min=1.0, max=3.0).view(1, nc, 1, 1), max=0.80)
    chan_f = in_channel.to(dtype)
    dq_neut_5 = -neut_eff.unsqueeze(0) * chan_f.unsqueeze(0) * q_chg  # [5, nz, nc, ny, nx]

    # For pure IC discharges (not grounded), remove channel-volume-mean charge imbalance so IC strictly conserves sum(rho_q) == 0
    cell_vol = dz_4d * (dx_m * dy_m)
    dq_neut_tot = dq_neut_5.sum(dim=0)  # [nz, nc, ny, nx]
    chan_vol = (chan_f * cell_vol).sum(dim=(0, 2, 3), keepdim=True)  # [1, nc, 1, 1]
    ic_only = triggered & (~is_cg_dom) & (chan_vol > 1.0)
    imbal_mean = (dq_neut_tot * cell_vol).sum(dim=(0, 2, 3), keepdim=True) / torch.clamp(chan_vol, min=1.0)
    # Distribute exact zero-sum balancing adjustment proportionally to |q_chg| across active channel species
    abs_q_sum = torch.clamp(q_chg.abs().sum(dim=0, keepdim=True), min=1.0e-18)
    w_spec = q_chg.abs() / abs_q_sum
    dq_neut_5 = torch.where(
        ic_only.unsqueeze(0) & in_channel.unsqueeze(0),
        dq_neut_5 - w_spec * imbal_mean.unsqueeze(0),
        dq_neut_5,
    )
    q_chg = q_chg + dq_neut_5

    # Diagnose subcolumn flash-rate density [flashes m^-2 s^-1] from neutralized charge dipole magnitude
    # Each bipolar flash neutralizes +Q_flash and -Q_flash -> integral |dq_neut| dV = 2 * Q_flash * N_flash
    dq_abs_col_c = 0.5 * (dq_neut_5.sum(dim=0).abs() * dz_4d).sum(dim=0) * (dx_m * dy_m)  # [nc, ny, nx] [C]
    n_flashes_sub = dq_abs_col_c / Q_CHARGE_PER_FLASH_C
    flash_tot_sub = n_flashes_sub / (dx_m * dy_m * dt_s)  # [flashes m^-2 s^-1]
    f_cg_sub = torch.where(
        flash_tot_sub > 0.0,
        is_cg_sub.to(dtype),
        torch.zeros_like(flash_tot_sub),
    )
    flash_cg_sub = flash_tot_sub * f_cg_sub

    # Cold-cloud depth diagnostic for LightningDiagnostics parity
    is_warm = temp >= C.T_FREEZE
    z_freeze = torch.where(is_warm, zc_4d, torch.zeros_like(temp)).amax(dim=0)
    has_ice = (q_i_pos + q_s + q_g) >= 1.0e-5
    z_ice_top = torch.where(has_ice, zc_4d, torch.zeros_like(temp)).amax(dim=0)
    ccd_km = torch.clamp((z_ice_top - z_freeze) * 1.0e-3, min=0.0)

    # Update prognostic state in place
    state.q_chg = q_chg
    state.phi_e = phi_e
    state.e_mag = e_mag
    state.e_z = e_z

    flash_tot_cell = flash_tot_sub.reshape(nc, -1).mean(dim=-1)
    flash_cg_cell = flash_cg_sub.reshape(nc, -1).mean(dim=-1)

    # 8. Schumann & Huntrieser (2007) natural LNOx production & mass-conserving N deposition
    lnox_local_sub = LNOX_KG_N_PER_FLASH * flash_tot_sub
    lnox_cell = lnox_local_sub.reshape(nc, -1).mean(dim=-1)
    area_w = area_cell.to(device=dev, dtype=dtype)
    area_sum = torch.clamp(area_w.sum(), min=1.0)
    lnox_global_mean = (lnox_cell * area_w).sum() / area_sum

    n_dep_lnox_sub = (
        LNOX_LOCAL_DEP_FRAC * lnox_local_sub + (1.0 - LNOX_LOCAL_DEP_FRAC) * lnox_global_mean
    )
    n_dep_lnox_cell = n_dep_lnox_sub.reshape(nc, -1).mean(dim=-1)

    return LightningDiagnostics(
        flash_tot_sub=flash_tot_sub,
        flash_cg_sub=flash_cg_sub,
        f_cg_sub=f_cg_sub,
        cold_cloud_km_sub=ccd_km,
        flash_tot_cell=flash_tot_cell,
        flash_cg_cell=flash_cg_cell,
        n_dep_lnox_sub=n_dep_lnox_sub,
        n_dep_lnox_cell=n_dep_lnox_cell,
    )
