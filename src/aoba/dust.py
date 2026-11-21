"""Map-Free Global-Parameter Mineral Dust Aerosol Module (Emission, Settling, Scavenging).

Computes mineral dust emission solely from autonomous land surface state (LAI, snow cover,
top-layer soil moisture, and friction velocity `u_star`) using a single global scaling constant
`c_dust_emiss` [kg s^2 m^-5], without any prescribed regional dust-source maps.

References
----------
- Marticorena & Bergametti (1995, J. Geophys. Res., 100, 16415-16430):
  Threshold friction velocity and horizontal-to-vertical saltation-sandblasting flux.
- Fécan, Marticorena, & Bergametti (1999, Ann. Geophys., 17, 149-157):
  Soil-moisture inhibition of wind erosion.
- Balkanski, Schulz, Claquin, & Guibert (2007, Atmos. Chem. Phys., 7, 81-95):
  Broadband shortwave & longwave mineral dust optical properties.
"""

from __future__ import annotations

import torch

from aoba import constants as C
from aoba.surface import SURF_LAND


# Standard physical constants for mineral dust (no regional tuning)
U_STAR_THRESH_DUST: float = 0.25    # Dry saltation threshold friction velocity [m s^-1]
THETA_MOIST_INHIB: float = 0.20     # Top-soil volumetric moisture inhibiting saltation [m^3 m^-3]
V_SETTLE_DUST: float = 4.0e-3       # Effective Stokes settling velocity (~1.5 um radius) [m s^-1]
V_DEP_DUST: float = 2.0e-3          # Surface dry deposition velocity [m s^-1]
ALPHA_WET_SCAV: float = 0.10        # Precipitation washout coefficient [m^2 kg^-1]

# Broadband optical properties (Balkanski et al., 2007)
DUST_K_EXT_SW1: float = 600.0       # VIS mass extinction efficiency [m^2 kg^-1]
DUST_SSA_SW1: float = 0.92          # VIS single-scattering albedo [-]
DUST_ASYM_SW1: float = 0.70         # VIS asymmetry factor [-]
DUST_K_EXT_SW2: float = 400.0       # NIR mass extinction efficiency [m^2 kg^-1]
DUST_SSA_SW2: float = 0.95          # NIR single-scattering albedo [-]
DUST_ASYM_SW2: float = 0.65         # NIR asymmetry factor [-]
DUST_K_ABS_LW1: float = 100.0       # LW1 mass absorption efficiency [m^2 kg^-1]
DUST_K_ABS_LW2: float = 150.0       # LW2 window mass absorption efficiency [m^2 kg^-1]


def compute_dust_emission_sub(
    surf_type: torch.Tensor,
    lai: torch.Tensor,
    swe_land: torch.Tensor,
    w_liq_soil_top: torch.Tensor,
    u_star: torch.Tensor,
    c_dust_emiss: float = 1.5e-9,
    k_beer_lai: float = 0.5,
    dz_soil_top: float = 0.05,
) -> torch.Tensor:
    """Compute subcolumn mineral dust vertical emission flux `[kg m^-2 s^-1]` without source maps."""
    is_land = (surf_type == SURF_LAND).to(u_star.dtype)
    f_bare = torch.exp(-k_beer_lai * torch.clamp(lai, min=0.0))
    snow_free = (swe_land < 1.0).to(u_star.dtype)
    theta_top = torch.clamp(w_liq_soil_top / (1000.0 * dz_soil_top), min=0.0)
    dry_fac = torch.clamp(1.0 - theta_top / THETA_MOIST_INHIB, min=0.0, max=1.0)

    u_safe = torch.clamp(u_star, min=U_STAR_THRESH_DUST)
    salt_fac = torch.clamp(1.0 - (U_STAR_THRESH_DUST / u_safe) ** 2, min=0.0)
    return float(c_dust_emiss) * is_land * f_bare * snow_free * dry_fac * (u_star ** 3) * salt_fac


def step_dust_column(
    q_dust: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    z_c: torch.Tensor,
    f_dust_emit: torch.Tensor,
    precip_rate: torch.Tensor,
    u_star_cell: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Advance 3D host dust mixing ratio `[Nz, N_cells]` via emission, PBL mixing, settling & washout."""
    nz = q_dust.shape[0]
    m_k = torch.clamp(rho_d * dz_c, min=1.0e-3)
    q_cur = torch.clamp(q_dust, min=0.0).clone()

    # 1. Surface emission injection into lowest layer
    q_cur[0] = q_cur[0] + dt * torch.clamp(f_dust_emit, min=0.0) / m_k[0]

    # 2. Fast PBL turbulent mixing across lowest ~1.5 km (two-pass symmetric relaxation)
    z_rel = torch.clamp(z_c - z_c[0:1] + 0.5 * dz_c[0:1], min=10.0)
    h_pbl = 1200.0
    for k in range(min(6, nz - 1)):
        z_int = 0.5 * (z_rel[k] + z_rel[k + 1])
        k_z = torch.where(
            z_int < h_pbl,
            0.4 * torch.clamp(u_star_cell, min=0.05) * z_int * ((1.0 - z_int / h_pbl) ** 2),
            torch.zeros_like(z_int),
        )
        dz_int = torch.clamp(z_c[k + 1] - z_c[k], min=10.0)
        rho_int = 0.5 * (rho_d[k] + rho_d[k + 1])
        exch = (dt * rho_int * k_z) / dz_int
        # Implicit-like bounded exchange between layer k and k+1 conserving mass
        frac = exch / (1.0 + exch * (1.0 / m_k[k] + 1.0 / m_k[k + 1]))
        dq_flux = frac * (q_cur[k] - q_cur[k + 1])
        q_cur[k] = q_cur[k] - dq_flux / m_k[k]
        q_cur[k + 1] = q_cur[k + 1] + dq_flux / m_k[k + 1]

    # 3. Unconditionally stable top-down implicit gravitational settling + surface dry deposition
    q_out = torch.empty_like(q_cur)
    flux_above = torch.zeros_like(q_cur[0])
    for k in range(nz - 1, -1, -1):
        v_fall = V_DEP_DUST if k == 0 else V_SETTLE_DUST
        sink = dt * rho_d[k] * v_fall
        q_k = (m_k[k] * q_cur[k] + dt * flux_above) / (m_k[k] + sink)
        q_out[k] = q_k
        flux_above = rho_d[k] * v_fall * q_k

    # 4. Precipitation wet scavenging (washout)
    wash = torch.exp(-ALPHA_WET_SCAV * torch.clamp(precip_rate, min=0.0) * dt)
    q_out = q_out * wash.unsqueeze(0)
    return torch.clamp(q_out, min=0.0, max=1.0e-3)
