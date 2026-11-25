"""Map-Free Embedded-CRM Non-Inductive Charging Lightning & LNOx Deposition Module.

Diagnoses subcolumn (`[N_cells, Ny, Nx]`) total lightning flash-rate density,
cloud-to-ground (CG) strike rate for wildfire ignition, and natural lightning-produced
nitrogen-oxide (`LNOx`) atmospheric nitrogen deposition directly from explicit CRM
updrafts, graupel/ice hydrometeors, and cold-cloud depth with ZERO geographic or
land/ocean lookup maps.

References
----------
- McCaul, E. W., Goodman, S. J., LaCasse, K. M., & Cecil, D. J. (2009):
  "Forecasting lightning threat using cloud-resolving model simulations",
  Weather and Forecasting, 24(3), 709-729, Eqs. (1)-(3) (F1 graupel flux at -15 C,
  F2 vertically integrated ice path, F3 = 0.95*F1 + 0.05*F2 blended flash-rate proxy).
- Price, C., & Rind, D. (1993):
  "What determines the cloud-to-ground lightning fraction in thunderstorms?",
  Geophysical Research Letters, 20(6), 463-466, Eq. (1) (cold-cloud depth polynomial
  Z = 0.021 H^4 - 0.648 H^3 + 7.493 H^2 - 36.54 H + 63.09, f_CG = 1 / (1 + Z)).
- Schumann, U., & Huntrieser, H. (2007):
  "The global lightning-induced nitrogen oxides source",
  Atmospheric Chemistry and Physics, 7, 3823-3907, Sec. 4 & Table 20
  (~44 +/- 5 flashes s^-1 globally; 250 mol NOx = 3.5 kg N per flash -> ~5 Tg N yr^-1).

AOBA Formulation Reference: Sec. 5.6, Eqs. (5.16)-(5.18) (CRM Non-Inductive Charging Lightning & LNOx).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from aoba import constants as C
from aoba.microphysics import T0_GRAUPEL, T00_GRAUPEL, T0_PRECIP, T00_PRECIP
from aoba.thermodynamics import diagnose_pressure_temperature


# Universal global parameters (ZERO land/ocean or geographic maps)
LIGHTNING_CAL_FACTOR: float = 3.6       # Single global scaling factor aligning CRM F3 to ~44 fl s^-1 [-]
LIGHTNING_W1_WEIGHT: float = 0.95       # McCaul et al. (2009, Eq. 3) weight on updraft graupel flux F1 [-]
LIGHTNING_W2_WEIGHT: float = 0.05       # Weight on vertically integrated total ice path F2 [-]
LIGHTNING_F1_COEF: float = 0.042        # McCaul et al. (2009, Eq. 1) graupel-flux slope [fl (5 min)^-1 km^-2 / (g m^-2 s^-1)]
LIGHTNING_F2_COEF: float = 0.20         # McCaul et al. (2009, Eq. 2) total-ice-path slope [fl (5 min)^-1 km^-2 / (kg m^-2)]
LIGHTNING_UPDRAFT_EXP: float = 1.25     # Superlinear non-inductive riming-velocity charging exponent (Saunders et al., 1991) [-]
LIGHTNING_CORE_HALF_SAT: float = 1.0    # Half-saturation graupel flux gating F2 so inert cirrus at w=0 sparks strictly 0 [g m^-2 s^-1]
LIGHTNING_T_CHARGE_K: float = 258.15    # Center of non-inductive charging zone (-15 degC) [K]
LIGHTNING_T_SIGMA_K: float = 7.0        # Gaussian half-width of charging zone (-22..-8 degC) [K]
LIGHTNING_ICE_TOP_THRESH: float = 1.0e-5  # Ice mixing ratio defining convective cloud top [kg kg^-1]
LNOX_KG_N_PER_FLASH: float = 3.5        # 250 mol N / flash (Schumann & Huntrieser, 2007) [kg N flash^-1]
LNOX_LOCAL_DEP_FRAC: float = 0.50       # Fraction of LNOx deposited locally via convective washout; remainder well-mixed [-]


@dataclass
class LightningDiagnostics:
    """Subcolumn & host-cell lightning and natural LNOx nitrogen deposition diagnostics."""

    flash_tot_sub: torch.Tensor      # Total (IC + CG) flash-rate density [N_cells, Ny, Nx] [flashes m^-2 s^-1]
    flash_cg_sub: torch.Tensor       # Cloud-to-ground strike density [N_cells, Ny, Nx] [flashes m^-2 s^-1]
    f_cg_sub: torch.Tensor           # Cloud-to-ground fraction [N_cells, Ny, Nx] [-]
    cold_cloud_km_sub: torch.Tensor  # Cold-cloud depth (Z_top - Z_0C) [N_cells, Ny, Nx] [km]
    flash_tot_cell: torch.Tensor     # Host-cell mean total flash-rate density [N_cells] [flashes m^-2 s^-1]
    flash_cg_cell: torch.Tensor      # Host-cell mean CG strike density [N_cells] [flashes m^-2 s^-1]
    n_dep_lnox_sub: torch.Tensor     # Natural atmospheric N deposition from LNOx [N_cells, Ny, Nx] [kg N m^-2 s^-1]
    n_dep_lnox_cell: torch.Tensor    # Host-cell mean LNOx N deposition [N_cells] [kg N m^-2 s^-1]


# AOBA Formulation Eqs. (5.16)-(5.18)
def compute_crm_lightning_and_lnox(
    rho_d: torch.Tensor,
    theta_m: torch.Tensor,
    q_v: torch.Tensor,
    q_i: torch.Tensor,
    q_p: torch.Tensor,
    w_stag: torch.Tensor,
    z_c: torch.Tensor,
    dz_c: torch.Tensor,
    area_cell: torch.Tensor,
    cal_factor: float | None = None,
    q_g_prog: torch.Tensor | None = None,
    q_s_prog: torch.Tensor | None = None,
) -> LightningDiagnostics:
    """Diagnose subcolumn lightning flash rate, CG fraction, and mass-conserving LNOx deposition.

    Parameters
    ----------
    rho_d, theta_m, q_v, q_i, q_p : Tensor `[Nz, N_cells, Ny, Nx]`
        Embedded CRM dry air density, moist potential temperature, vapor, cloud ice, and precip.
    w_stag : Tensor `[Nz + 1, N_cells, Ny, Nx]`
        Staggered CRM vertical velocity `[m s^-1]`.
    z_c, dz_c : Tensor `[Nz, N_cells]` or `[Nz]`
        Column layer center heights and thicknesses `[m]`.
    area_cell : Tensor `[N_cells]`
        SCVT host cell area `[m^2]` for exact area-weighted global LNOx conservation.
    """
    nz, nc, ny, nx = rho_d.shape
    dev = rho_d.device
    dtype = rho_d.dtype

    _, _, temp = diagnose_pressure_temperature(rho_d, theta_m, q_v)
    w_c = 0.5 * (w_stag[:-1] + w_stag[1:])
    w_up = torch.clamp(w_c, min=0.0)

    # Use true prognostic graupel/snow when 6-class 2-moment microphysics is active; else KR03 split
    q_i_pos = torch.clamp(q_i, min=0.0)
    if q_g_prog is not None:
        q_g = torch.clamp(q_g_prog, min=0.0)
        q_s = torch.clamp(q_s_prog, min=0.0) if q_s_prog is not None else torch.zeros_like(q_g)
        q_p_ice = q_g + q_s
    else:
        f_ice_p = torch.clamp((T0_PRECIP - temp) / max(T0_PRECIP - T00_PRECIP, 1.0), 0.0, 1.0)
        f_graup = torch.clamp((temp - T00_GRAUPEL) / max(T0_GRAUPEL - T00_GRAUPEL, 1.0), 0.0, 1.0)
        q_p_pos = torch.clamp(q_p, min=0.0)
        q_p_ice = f_ice_p * q_p_pos
        q_g = q_p_ice * f_graup

    # Broadcast vertical metrics to [Nz, N_cells, 1, 1]
    if z_c.ndim == 2:
        zc_4d = z_c.view(nz, nc, 1, 1)
        dz_4d = dz_c.view(nz, nc, 1, 1)
    else:
        zc_4d = z_c.view(nz, 1, 1, 1)
        dz_4d = dz_c.view(nz, 1, 1, 1)

    # 1. McCaul et al. (2009, Eqs. 1-3) non-inductive charging proxies (purely physical, map-free)
    #    F1: upward graupel mass flux [g m^-2 s^-1] weighted around the -15 degC charging zone
    t_wt = torch.exp(-(((temp - LIGHTNING_T_CHARGE_K) / LIGHTNING_T_SIGMA_K) ** 2))
    gfx_lin = t_wt * rho_d * w_up * (1000.0 * q_g)
    gfx_max = gfx_lin.amax(dim=0)  # [N_cells, Ny, Nx]
    gfx_chg = (t_wt * rho_d * torch.pow(w_up, LIGHTNING_UPDRAFT_EXP) * (1000.0 * q_g)).amax(dim=0)
    f1 = LIGHTNING_F1_COEF * gfx_chg  # [N_cells, Ny, Nx]

    #    F2: vertically integrated total ice-phase hydrometeor mass [kg m^-2]
    iwp_tot = (rho_d * dz_4d * (q_i_pos + q_p_ice)).sum(dim=0)
    #    Strict zero-at-zero quadratic Hill gate on active mixed-phase graupel flux so inert cirrus
    #    and weak stratiform clouds (gfx == 0) spark strictly zero lightning
    has_active_core = (gfx_max / (gfx_max + LIGHTNING_CORE_HALF_SAT)) ** 2
    f2 = LIGHTNING_F2_COEF * iwp_tot * has_active_core

    c_scale = LIGHTNING_CAL_FACTOR if cal_factor is None else float(cal_factor)
    # Convert McCaul units [flashes (5 min)^-1 km^-2] -> SI [flashes m^-2 s^-1]
    inv_5min_km2 = 1.0 / (300.0 * 1.0e6)
    flash_tot_sub = torch.clamp(
        c_scale * (LIGHTNING_W1_WEIGHT * f1 + LIGHTNING_W2_WEIGHT * f2) * inv_5min_km2,
        min=0.0,
    )

    # 2. Price & Rind (1993, Eq. 1) cold-cloud depth (CCD) IC/CG partitioning
    is_warm = temp >= C.T_FREEZE
    z_freeze = torch.where(is_warm, zc_4d, torch.zeros_like(temp)).amax(dim=0)
    has_ice = (q_i_pos + q_p_ice) >= LIGHTNING_ICE_TOP_THRESH
    z_ice_top = torch.where(has_ice, zc_4d, torch.zeros_like(temp)).amax(dim=0)
    ccd_km_raw = torch.clamp((z_ice_top - z_freeze) * 1.0e-3, min=0.0)
    h_poly = torch.clamp(ccd_km_raw, min=5.5, max=14.0)
    h2 = h_poly * h_poly
    h3 = h2 * h_poly
    h4 = h2 * h2
    z_ic_cg = torch.clamp(
        0.021 * h4 - 0.648 * h3 + 7.493 * h2 - 36.54 * h_poly + 63.09,
        min=1.0,
        max=19.0,
    )
    f_cg_sub = 1.0 / (1.0 + z_ic_cg)
    flash_cg_sub = flash_tot_sub * f_cg_sub

    flash_tot_cell = flash_tot_sub.reshape(flash_tot_sub.shape[0], -1).mean(dim=-1)
    flash_cg_cell = flash_cg_sub.reshape(flash_cg_sub.shape[0], -1).mean(dim=-1)

    # 3. Schumann & Huntrieser (2007) natural LNOx production & mass-conserving N deposition
    #    50% deposited locally with the storm, 50% distributed globally as background atmospheric NOx
    lnox_local_sub = LNOX_KG_N_PER_FLASH * flash_tot_sub
    lnox_cell = lnox_local_sub.reshape(lnox_local_sub.shape[0], -1).mean(dim=-1)
    area_w = area_cell.to(device=dev, dtype=dtype)
    area_sum = torch.clamp(area_w.sum(), min=1.0)
    lnox_global_mean = (lnox_cell * area_w).sum() / area_sum

    n_dep_lnox_sub = (
        LNOX_LOCAL_DEP_FRAC * lnox_local_sub + (1.0 - LNOX_LOCAL_DEP_FRAC) * lnox_global_mean
    )
    n_dep_lnox_cell = n_dep_lnox_sub.reshape(n_dep_lnox_sub.shape[0], -1).mean(dim=-1)

    return LightningDiagnostics(
        flash_tot_sub=flash_tot_sub,
        flash_cg_sub=flash_cg_sub,
        f_cg_sub=f_cg_sub,
        cold_cloud_km_sub=ccd_km_raw,
        flash_tot_cell=flash_tot_cell,
        flash_cg_cell=flash_cg_cell,
        n_dep_lnox_sub=n_dep_lnox_sub,
        n_dep_lnox_cell=n_dep_lnox_cell,
    )


def diagnose_lightning_metrics(
    diag: LightningDiagnostics,
    area_cell: torch.Tensor,
    surf_type: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute global float64 lightning flash rate [s^-1], CG %, land/ocean ratio, and LNOx [Tg N yr^-1]."""
    area = area_cell.detach().cpu().to(torch.float64)
    ft = diag.flash_tot_cell.detach().cpu().to(torch.float64)
    fcg = diag.flash_cg_cell.detach().cpu().to(torch.float64)
    ndep = diag.n_dep_lnox_cell.detach().cpu().to(torch.float64)

    fl_tot_s = float((ft * area).sum().item())
    fl_cg_s = float((fcg * area).sum().item())
    cg_pct = 100.0 * fl_cg_s / max(fl_tot_s, 1.0e-12)
    lnox_tgn_yr = float((ndep * area).sum().item() * (C.DAYS_PER_YEAR * C.SECONDS_PER_DAY) * 1.0e-9)

    out = {
        "flash_tot_per_s": fl_tot_s,
        "flash_cg_per_s": fl_cg_s,
        "cg_frac_pct": cg_pct,
        "lnox_tgn_yr": lnox_tgn_yr,
    }
    if surf_type is not None:
        st = surf_type.detach().cpu()
        is_land = (st == 1)
        is_ocn = (st == 0)
        a_land = float(area[is_land].sum().item())
        a_ocn = float(area[is_ocn].sum().item())
        r_land = float((ft[is_land] * area[is_land]).sum().item()) / max(a_land, 1.0)
        r_ocn = float((ft[is_ocn] * area[is_ocn]).sum().item()) / max(a_ocn, 1.0)
        out["land_ocean_flash_ratio"] = r_land / max(r_ocn, 1.0e-18)
    return out
