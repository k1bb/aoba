"""Online Active-Learning & Self-Distilling 2D CRM Acceleration Module for AOBA.

Accelerates Superparameterized (SP) integrations on both Apple Silicon (MPS) and NVIDIA
(CUDA) GPUs by running AOBA's genuine nonhydrostatic 2D CRM on a rotating spatially
stratified teacher subset (`1 / stride` columns per step, default `1 / 13` with continuous
convective-memory dwell + epistemic uncertainty-guided active fallback) while a lightweight,
architecture-constrained 1D-ResNet student emulator predicts subgrid convective, turbulent,
and microphysical tendencies on the remaining columns via dense GPU matrix operations.

Key Scientific & Numerical Guarantees
-------------------------------------
1. **Map-Free Universal Input Features**: Uses strictly local thermodynamic, dynamical,
   radiative, and surface-flux profiles (zero geographic `lat`/`lon` or lookup-table inputs)
   so online adaptation generalizes across present-day, $4\\times\\mathrm{CO}_2$, paleoclimate,
   and exoplanet regimes.
2. **Exact Algebraic Column Conservation (Beucler et al., 2021)**:
   - Total column water mass ($\\sum_k \\Delta m_k (q_n + q_p) + P_{\\mathrm{tot}} = W_{\\mathrm{pre}} + E\\Delta t$)
     is closed analytically to float32 machine precision with strict non-negativity of
     $q_v, q_c, q_i, q_p, P_{\\mathrm{rain}}, P_{\\mathrm{snow}}$.
   - Column moist enthalpy change ($c_p \\sum_k \\Delta m_k \\Pi_k \\Delta\\theta_{m,k}$) matches
     exact surface sensible heat flux plus net latent heating from phase changes and
     precipitation to machine precision.
   - Column horizontal momentum change ($\\sum_k \\Delta m_k \\Delta u_k, \\Delta v_k$) matches
     exact surface wind-stress impulse $(\\tau_x \\Delta t, \\tau_y \\Delta t)$ to machine precision.
3. **Unbroken Convective Memory & Zero Catastrophic Forgetting**:
   - Each rotating teacher cohort remains on the genuine 2D CRM for a continuous dwell window
     (`dwell_steps = 12`) so embedded cloud/cold-pool memory stays intact.
   - A GPU-resident dual-zone replay buffer (recency ring + long-horizon reservoir) prevents
     seasonal/latitudinal forgetting without CPU–GPU stalls.
   - A 2-head epistemic ensemble routes high-uncertainty / out-of-distribution columns to the
     genuine 2D CRM teacher automatically.

References
----------
- Beucler, T., Pritchard, M., Rasp, S., Ott, J., Baldi, P., & Gentine, P. (2021).
  Enforcing analytic constraints in neural networks emulating physical systems.
  *Physical Review Letters*, 126(9), 098302.
- Yu, S., Hannah, W., Peng, L., et al. (2023/2025). ClimSim / ClimSim-Online: A large
  multi-scale dataset and framework for hybrid ML-physics climate emulation.
  *NeurIPS 2023* / *JMLR 2025*.
- Kochkov, D., Yuval, J., Langmore, I., et al. (2024). Neural general circulation models
  for weather and climate. *Nature*, 632, 1060-1066.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoba import constants as C
from aoba.thermodynamics import diagnose_pressure_temperature, qsat_mixed_val


# ---------------------------------------------------------------------------
# 1. Universal Map-Free Feature Extraction & Exact Physical Conservation
# ---------------------------------------------------------------------------
_N_PROF_IN = 14      # 14 vertical profile channels per level
_N_STRUCT_IN = 16    # 2D/3D periodic surface heterogeneity + atmospheric organization invariants
_N_SCAL_IN = 7 + _N_STRUCT_IN  # 7 cell-mean boundary forcings + 16 spatial-clustering invariants = 23
_N_PROF_OUT = 8      # (d_temp_raw, d_qn_raw, d_qp_raw, logit_fc, logit_fi, d_u_raw, d_v_raw, logit_fcld)
_N_SCAL_OUT = 3      # (raw_ptot_frac, logit_fsnow, logit_fprec)


def compute_periodic_spatial_stats(
    field_3d: torch.Tensor,  # [N, ny, nx]
    dx: float,
    dy: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute translation/rotation-invariant 2D/3D periodic spatial clustering descriptors.

    Works seamlessly for both 2D (`ny == 1, nx >= 2`) and 3D (`ny >= 2, nx >= 2`) periodic
    CRM domains, capturing:
      1. `std`: subgrid fluctuation amplitude $\\sigma_\\phi = \\sqrt{\\mathrm{Var}(\\phi)}$.
      2. `moran_i`: periodic nearest-neighbor Moran's $I \\in [-1, +1]$ ($+1$ for coherent
         contiguous patches/clusters; $-1$ for fine scattered/checkerboard patches; $0$ for
         horizontally uniform fields).
      3. `l_patch_km`: characteristic physical patch length scale [km]
         $L_{\\mathrm{patch}} = \\sqrt{2\\,\\mathrm{Var}(\\phi) / (\\langle|\\nabla \\phi'|^2\\rangle_{\\mathrm{iso}} + \\epsilon)}$
         (Patton et al., 2005 control length for thermally driven mesoscale land breezes).
    """
    n_col, ny, nx = field_3d.shape
    dev = field_3d.device
    dtype = field_3d.dtype
    if ny * nx <= 1:
        z = torch.zeros(n_col, device=dev, dtype=dtype)
        return z, z, z

    dx_m = max(float(dx), 1.0)
    dy_m = max(float(dx if dy is None else dy), 1.0)
    anom = field_3d - field_3d.mean(dim=(-2, -1), keepdim=True)
    var = (anom * anom).mean(dim=(-2, -1))
    std = torch.sqrt(torch.clamp(var, min=0.0))
    active = var > 1.0e-12

    cov_acc = torch.zeros_like(var)
    grad_acc = torch.zeros_like(var)
    n_axes = 0
    if nx > 1:
        ax_next = torch.roll(anom, shifts=-1, dims=-1)
        cov_acc = cov_acc + (anom * ax_next).mean(dim=(-2, -1))
        grad_acc = grad_acc + ((ax_next - anom) ** 2).mean(dim=(-2, -1)) / (dx_m * dx_m)
        n_axes += 1
    if ny > 1:
        ay_next = torch.roll(anom, shifts=-1, dims=-2)
        cov_acc = cov_acc + (anom * ay_next).mean(dim=(-2, -1))
        grad_acc = grad_acc + ((ay_next - anom) ** 2).mean(dim=(-2, -1)) / (dy_m * dy_m)
        n_axes += 1

    inv_axes = 1.0 / float(max(n_axes, 1))
    cov_nn = cov_acc * inv_axes
    grad_iso = grad_acc * inv_axes

    moran_i = torch.where(
        active,
        torch.clamp(cov_nn / torch.clamp(var, min=1.0e-12), min=-1.0, max=1.0),
        torch.zeros_like(var),
    )
    l_patch_m = torch.where(
        active & (grad_iso > 1.0e-24),
        torch.sqrt(torch.clamp(2.0 * var / torch.clamp(grad_iso, min=1.0e-24), min=0.0)),
        torch.zeros_like(var),
    )
    l_patch_km = l_patch_m * 1.0e-3
    return std, moran_i, l_patch_km


def compute_subgrid_structure_features(
    shf_3d: torch.Tensor,       # [N, ny, nx] sensible heat flux [W m^-2]
    lhf_3d: torch.Tensor,       # [N, ny, nx] latent heat flux [W m^-2]
    t_rad_3d: torch.Tensor,     # [N, ny, nx] radiative surface temperature [K]
    alb_vis_3d: torch.Tensor,   # [N, ny, nx] visible direct albedo [-]
    qn_4d: torch.Tensor,        # [nz, N, ny, nx] total non-precipitating water [kg kg^-1]
    w_4d: torch.Tensor,         # [nz+1, N, ny, nx] vertical velocity [m s^-1]
    dx: float,
    dy: float | None = None,
) -> torch.Tensor:
    """Build universal `[N, 16]` dimensionless 2D/3D surface-heterogeneity & organization features.

    Captures both amplitude and spatial arrangement (clustered vs scattered patches, physical
    patch length scale $L_{\\mathrm{patch}}$) of surface fluxes/albedo/temperature and convective
    moisture, plus $\\rho(\\mathrm{SHF}, \\mathrm{LHF}) \\in [-1, +1]$ to separate land-state
    Bowen-ratio contrast (e.g., wildfire bare patch vs forest: $\\rho \\approx -1$) from cloud
    shadowing ($\\rho \\approx +1$).
    """
    shf_std, shf_mi, shf_lp = compute_periodic_spatial_stats(shf_3d, dx, dy)
    lhf_std, lhf_mi, lhf_lp = compute_periodic_spatial_stats(lhf_3d, dx, dy)
    tr_std, tr_mi, tr_lp = compute_periodic_spatial_stats(t_rad_3d, dx, dy)
    alb_std, alb_mi, _ = compute_periodic_spatial_stats(alb_vis_3d, dx, dy)

    # Spatial cross-correlation rho(SHF, LHF) in [-1, +1]
    a_shf = shf_3d - shf_3d.mean(dim=(-2, -1), keepdim=True)
    a_lhf = lhf_3d - lhf_3d.mean(dim=(-2, -1), keepdim=True)
    cov_sl = (a_shf * a_lhf).mean(dim=(-2, -1))
    denom_sl = shf_std * lhf_std
    rho_shf_lhf = torch.where(
        denom_sl > 1.0e-4,
        torch.clamp(cov_sl / torch.clamp(denom_sl, min=1.0e-4), min=-1.0, max=1.0),
        torch.zeros_like(cov_sl),
    )

    # Column-mean moisture field [N, ny, nx] for RCEMIP convective self-aggregation clustering
    qn_col_3d = qn_4d.mean(dim=0)
    qn_std, qn_mi, qn_lp = compute_periodic_spatial_stats(qn_col_3d, dx, dy)
    w_rms = torch.sqrt(torch.clamp((w_4d * w_4d).mean(dim=(0, 2, 3)), min=0.0))

    return torch.stack([
        torch.clamp(shf_std * 0.01, 0.0, 5.0),
        shf_mi,
        torch.log1p(torch.clamp(shf_lp, 0.0, 200.0)),
        torch.clamp(lhf_std * 0.01, 0.0, 5.0),
        lhf_mi,
        torch.log1p(torch.clamp(lhf_lp, 0.0, 200.0)),
        rho_shf_lhf,
        torch.clamp(tr_std * 0.2, 0.0, 5.0),
        tr_mi,
        torch.log1p(torch.clamp(tr_lp, 0.0, 200.0)),
        torch.clamp(alb_std * 5.0, 0.0, 5.0),
        alb_mi,
        torch.log1p(torch.clamp(qn_std, min=0.0) * 1.0e3),
        qn_mi,
        torch.log1p(torch.clamp(qn_lp, 0.0, 200.0)),
        torch.clamp(w_rms, 0.0, 10.0),
    ], dim=1).contiguous()


def extract_crm_distill_features(
    rho_d: torch.Tensor,         # [nz, N]
    theta_m: torch.Tensor,       # [nz, N]
    q_v: torch.Tensor,           # [nz, N]
    q_c: torch.Tensor,           # [nz, N]
    q_i: torch.Tensor,           # [nz, N]
    q_p: torch.Tensor,           # [nz, N]
    u: torch.Tensor,             # [nz, N]
    v: torch.Tensor,             # [nz, N]
    dz_c: torch.Tensor,          # [nz, N]
    dthm_ls: torch.Tensor,       # [nz, N]
    dqn_ls: torch.Tensor,        # [nz, N]
    q_rad: torch.Tensor,         # [nz, N]
    shf: torch.Tensor,           # [N]
    lhf: torch.Tensor,           # [N]
    u_star: torch.Tensor,        # [N]
    tau_x: torch.Tensor,         # [N]
    tau_y: torch.Tensor,         # [N]
    is_ocean: torch.Tensor,      # [N]
    f_cor: torch.Tensor,         # [N]
    dt: float,
    *,
    subgrid_struct: torch.Tensor | None = None,  # [N, 16] 2D/3D surface heterogeneity & org invariants
    sub_org_qn_std: torch.Tensor | None = None,  # legacy fallback when subgrid_struct is omitted
    sub_org_w_rms: torch.Tensor | None = None,   # legacy fallback when subgrid_struct is omitted
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flat dimensionless map-free input `[N, 14*nz + 23]` plus `(exner, temp)` `[nz, N]`."""
    pres, exner, temp = diagnose_pressure_temperature(rho_d, theta_m, q_v)
    qs = torch.clamp(qsat_mixed_val(temp, pres), min=1.0e-7, max=0.2)
    rh = torch.clamp(q_v / qs, min=0.0, max=2.0)

    nz, n_col = rho_d.shape
    prof = torch.stack([
        (temp - 250.0) * 0.02,
        (torch.log(torch.clamp(pres, min=1.0)) - 10.0) * 0.4,
        (theta_m - 320.0) * 0.0125,
        torch.log1p(torch.clamp(q_v, min=0.0) * 1.0e3),
        torch.log1p(torch.clamp(q_c, min=0.0) * 1.0e4),
        torch.log1p(torch.clamp(q_i, min=0.0) * 1.0e4),
        torch.log1p(torch.clamp(q_p, min=0.0) * 1.0e4),
        rh,
        torch.clamp(u * (1.0 / 30.0), -4.0, 4.0),
        torch.clamp(v * (1.0 / 30.0), -4.0, 4.0),
        torch.clamp(dthm_ls * dt, -10.0, 10.0),
        torch.clamp(dqn_ls * dt * 1.0e3, -10.0, 10.0),
        torch.clamp(q_rad * dt * 2.0, -10.0, 10.0),
        torch.log(torch.clamp(dz_c, min=10.0) * 2.0e-3),
    ], dim=0)  # [14, nz, N]

    if subgrid_struct is None:
        struct_mat = torch.zeros((n_col, _N_STRUCT_IN), device=rho_d.device, dtype=rho_d.dtype)
        if sub_org_qn_std is not None:
            struct_mat[:, 12] = torch.log1p(torch.clamp(sub_org_qn_std, min=0.0) * 1.0e3)
        if sub_org_w_rms is not None:
            struct_mat[:, 15] = torch.clamp(sub_org_w_rms, 0.0, 10.0)
    else:
        struct_mat = subgrid_struct.to(device=rho_d.device, dtype=rho_d.dtype)

    sfc_base = torch.stack([
        torch.clamp(shf * 5.0e-3, -5.0, 5.0),
        torch.clamp(lhf * 5.0e-3, -5.0, 5.0),
        torch.clamp(u_star * 2.0, 0.0, 5.0),
        torch.clamp(tau_x * 5.0, -5.0, 5.0),
        torch.clamp(tau_y * 5.0, -5.0, 5.0),
        is_ocean.to(rho_d.dtype),
        torch.clamp(f_cor.abs() * 1.0e4, 0.0, 5.0),
    ], dim=1)  # [N, 7]

    # Flatten to [N, 14*nz + 23] for 100% dense BLAS GEMM / Tensor-Core execution
    prof_flat = prof.permute(2, 0, 1).reshape(n_col, _N_PROF_IN * nz)
    x = torch.cat([prof_flat, sfc_base, struct_mat], dim=1).contiguous()
    return x, exner, temp


def apply_exact_conservation_closure(
    raw_prof: torch.Tensor,      # [N, 7, nz]
    raw_scal: torch.Tensor,      # [N, 2]
    rho_d: torch.Tensor,         # [nz, N]
    dz_c: torch.Tensor,          # [nz, N]
    exner: torch.Tensor,         # [nz, N]
    temp_sfc: torch.Tensor,      # [N] lowest-layer temperature [K]
    thm_post_ls: torch.Tensor,   # [nz, N] theta_m AFTER exact large-scale + radiative forcing
    qn_post_ls: torch.Tensor,    # [nz, N] q_n AFTER exact large-scale forcing
    qp_pre: torch.Tensor,        # [nz, N]
    qc_pre: torch.Tensor,        # [nz, N]
    qi_pre: torch.Tensor,        # [nz, N]
    u_pre: torch.Tensor,         # [nz, N]
    v_pre: torch.Tensor,         # [nz, N]
    shf: torch.Tensor,           # [N] surface sensible heat flux [W m^-2]
    evap: torch.Tensor,          # [N] surface evaporation flux [kg m^-2 s^-1]
    tau_x: torch.Tensor,         # [N] surface zonal stress [N m^-2]
    tau_y: torch.Tensor,         # [N] surface meridional stress [N m^-2]
    dt: float,
) -> dict[str, torch.Tensor]:
    """Project raw NN outputs onto exact column water, moist-enthalpy, and momentum manifolds.

    Every output satisfies:
      1. `sum_k dm_k (q_n_new + q_p_new) + P_tot * dt == sum_k dm_k (q_n_post_ls + q_p_pre) + evap * dt`
      2. `c_p * sum_k dm_k * exner_k * (thm_new - thm_post_ls) == shf*dt + L_v*(P_tot*dt + dM_cond) + L_f*(P_snow*dt + dM_ice)`
      3. `sum_k dm_k * (u_new - u_pre) == tau_x * dt`, `sum_k dm_k * (v_new - v_pre) == tau_y * dt`
      4. `q_v >= 0, q_c >= 0, q_i >= 0, q_p >= 0, P_rain >= 0, P_snow >= 0`
    """
    # Convert [N, 8, nz] -> [nz, N] per field (contiguous for MPS backward kernels)
    rp = raw_prof.permute(1, 2, 0).contiguous()  # [8, nz, N]
    # Predict bounded temperature residual dT [K] -> d_theta_m = dT / exner so troposphere & stratosphere share O(1 K) scale
    d_temp_raw = torch.tanh(rp[0] * 0.2) * 5.0
    d_thm_raw = d_temp_raw / torch.clamp(exner, min=0.05)
    d_qn_raw = torch.tanh(rp[1] * 0.2) * 5.0e-3      # bounded [kg kg^-1] per step
    d_qp_raw = torch.tanh(rp[2] * 0.2) * 2.0e-3
    f_cond_logit = rp[3]
    f_ice_logit = rp[4]
    d_u_raw = torch.tanh(rp[5] * 0.2) * 8.0          # bounded [m s^-1] per step
    d_v_raw = torch.tanh(rp[6] * 0.2) * 8.0
    f_cld_pred = torch.sigmoid(rp[7] - 1.0)          # subgrid cloud area fraction in (0, 1)
    f_prec_pred = torch.sigmoid(raw_scal[:, 2] - 1.0)  # subgrid precipitating area fraction in (0, 1)

    dm = torch.clamp(rho_d * dz_c, min=1.0e-6)       # [nz, N] kg m^-2
    dm_sum = torch.clamp(dm.sum(dim=0), min=1.0e-3)  # [N]

    # --- 1. Exact Total Column Water Mass Conservation ---
    w_pre = (dm * (torch.clamp(qn_post_ls, min=0.0) + torch.clamp(qp_pre, min=0.0))).sum(dim=0)
    w_avail = torch.clamp(w_pre + evap * dt, min=1.0e-8)  # [N] kg m^-2

    # Predicted total precipitation mass [kg m^-2] bounded to physical column fraction <= 0.35
    p_frac = 0.35 * torch.sigmoid(raw_scal[:, 0] - 4.0)
    p_tot_mass = torch.clamp(p_frac * w_avail, min=0.0)
    # Suppress spurious precipitation when total column water is negligible
    p_tot_mass = torch.where(w_avail > 0.05, p_tot_mass, torch.zeros_like(p_tot_mass))

    # Apply surface evaporation into lowest layer before non-negative rescaling
    qn_base = torch.clamp(qn_post_ls, min=0.0).clone()
    qn_base[0] = torch.clamp(qn_base[0] + (evap * dt) / torch.clamp(dm[0], min=1.0e-4), min=0.0)
    qn_prov = torch.clamp(qn_base + d_qn_raw, min=0.0)
    qp_prov = torch.clamp(qp_pre + d_qp_raw, min=0.0)

    w_prov = (dm * (qn_prov + qp_prov)).sum(dim=0)
    w_target = torch.clamp(w_avail - p_tot_mass, min=0.0)
    scale_w = w_target / torch.clamp(w_prov, min=1.0e-10)
    # Guard extreme rescaling when column is nearly bone-dry
    zero_prov = w_prov <= 1.0e-10
    qn_new = torch.where(zero_prov.unsqueeze(0), qn_base, qn_prov * scale_w.unsqueeze(0))
    qp_new = torch.where(zero_prov.unsqueeze(0), qp_pre, qp_prov * scale_w.unsqueeze(0))
    # Recompute exact residual precipitation if zero_prov triggered
    w_post_check = (dm * (qn_new + qp_new)).sum(dim=0)
    p_tot_mass = torch.clamp(w_avail - w_post_check, min=0.0)

    # Rain / snow partition (zero snow above 278 K near surface)
    f_snow = torch.sigmoid(raw_scal[:, 1]) * torch.clamp((278.0 - temp_sfc) / 6.0, min=0.0, max=1.0)
    p_snow_rate = (p_tot_mass * f_snow) / dt
    p_rain_rate = (p_tot_mass * (1.0 - f_snow)) / dt

    # Partition q_n into (q_v, q_c, q_i) strictly preserving q_v + q_c + q_i == q_n.
    # Dead-zone threshold guarantees exact zero condensate in clear sky (eliminating spurious thin-cloud ASR bias).
    f_cond = torch.clamp(F.relu(torch.sigmoid(f_cond_logit - 1.5) - 0.18) * 1.16, 0.0, 0.95)
    f_ice = torch.sigmoid(f_ice_logit)
    q_cond = qn_new * f_cond
    qc_new = q_cond * (1.0 - f_ice)
    qi_new = q_cond * f_ice
    qv_new = torch.clamp(qn_new - qc_new - qi_new, min=0.0)

    # --- 2. Exact Column Moist Enthalpy / Static Energy Conservation ---
    # Net condensate + precipitation latent heating over the CRM step [J m^-2]
    d_m_cond = (dm * ((qc_new + qi_new + qp_new) - (qc_pre + qi_pre + qp_pre))).sum(dim=0)
    d_m_ice = (dm * (qi_new - qi_pre)).sum(dim=0)
    dh_exact = (
        shf * dt
        + C.L_V0 * (p_tot_mass + d_m_cond)
        + C.L_F0 * (p_tot_mass * f_snow + d_m_ice)
    )
    dh_raw = C.CP_D * (dm * exner * d_thm_raw).sum(dim=0)
    denom_h = torch.clamp(C.CP_D * (dm * exner * exner).sum(dim=0), min=1.0)
    d_thm_cons = d_thm_raw + ((dh_exact - dh_raw) / denom_h).unsqueeze(0) * exner
    thm_new = torch.clamp(thm_post_ls + d_thm_cons, min=150.0, max=10000.0)

    # --- 3. Exact Column Horizontal Momentum Conservation ---
    du_imp_raw = (dm * d_u_raw).sum(dim=0)
    dv_imp_raw = (dm * d_v_raw).sum(dim=0)
    du_cons = d_u_raw + ((tau_x * dt - du_imp_raw) / dm_sum).unsqueeze(0)
    dv_cons = d_v_raw + ((tau_y * dt - dv_imp_raw) / dm_sum).unsqueeze(0)
    u_new = u_pre + du_cons
    v_new = v_pre + dv_cons

    return {
        "theta_m": thm_new,
        "q_n": qn_new,
        "q_p": qp_new,
        "q_v": qv_new,
        "q_c": qc_new,
        "q_i": qi_new,
        "u": u_new,
        "v": v_new,
        "precip_rain": p_rain_rate,
        "precip_snow": p_snow_rate,
        "f_cld": f_cld_pred,
        "f_prec": f_prec_pred,
    }


def extract_subgrid_spatial_maps(
    shf_3d: torch.Tensor,       # [N, ny, nx] sensible heat flux [W m^-2]
    lhf_3d: torch.Tensor,       # [N, ny, nx] latent heat flux [W m^-2]
    t_rad_3d: torch.Tensor,     # [N, ny, nx] radiative surface temperature [K]
    alb_vis_3d: torch.Tensor,   # [N, ny, nx] visible direct albedo [-]
    qn_4d: torch.Tensor,        # [nz, N, ny, nx] total non-precipitating water [kg kg^-1]
    w_4d: torch.Tensor,         # [nz+1, N, ny, nx] vertical velocity [m s^-1]
) -> torch.Tensor:
    """Build `[N, 6, ny, nx]` dimensionless periodic spatial anomaly maps for `SubgridSpatialHead`.

    Provides the exact 2D/3D spatial arrangement of surface heat/moisture/temperature/albedo
    patches alongside pre-step convective moisture & updraft memory so the periodic spatial
    head can learn forest-locking RCE, forest-grassland vegetation breezes (including wind-
    shifted upwind/downwind asymmetry), and convective self-aggregation directly from the
    genuine 2D/3D CRM teacher.
    """
    shf_a = torch.clamp((shf_3d - shf_3d.mean(dim=(-2, -1), keepdim=True)) * 0.01, -5.0, 5.0)
    lhf_a = torch.clamp((lhf_3d - lhf_3d.mean(dim=(-2, -1), keepdim=True)) * 0.01, -5.0, 5.0)
    tr_a = torch.clamp((t_rad_3d - t_rad_3d.mean(dim=(-2, -1), keepdim=True)) * 0.2, -5.0, 5.0)
    alb_a = torch.clamp((alb_vis_3d - alb_vis_3d.mean(dim=(-2, -1), keepdim=True)) * 5.0, -5.0, 5.0)
    qn_col = qn_4d.mean(dim=0)
    qn_a = torch.clamp((qn_col - qn_col.mean(dim=(-2, -1), keepdim=True)) * 1.0e3, -5.0, 5.0)
    w_col = torch.clamp(w_4d.mean(dim=0), -5.0, 5.0)
    return torch.stack([shf_a, lhf_a, tr_a, alb_a, qn_a, w_col], dim=1).contiguous()


def _periodic_pad2d(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Manual MPS/CUDA/CPU-safe periodic 2D padding on `[B, C, ny, nx]` (supports `ny == 1`)."""
    if pad <= 0:
        return x
    ny, nx = int(x.shape[-2]), int(x.shape[-1])
    if nx >= pad:
        x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
    else:
        idx_x = (torch.arange(-pad, nx + pad, device=x.device) % nx)
        x = x.index_select(-1, idx_x)
    if ny >= pad:
        x = torch.cat([x[..., -pad:, :], x, x[..., :pad, :]], dim=-2)
    else:
        idx_y = (torch.arange(-pad, ny + pad, device=x.device) % ny)
        x = x.index_select(-2, idx_y)
    return x


def _scores_to_ranks(
    scores_flat: torch.Tensor,   # [N_s, n_sub]
    tie_break: torch.Tensor,     # [N_s, n_sub] small deterministic tie-breaker in [0, 1e-4]
) -> torch.Tensor:
    """Convert `[N_s, n_sub]` spatial preference scores to descending rank `0 .. n_sub-1`."""
    n_s, n_sub = scores_flat.shape
    order = torch.argsort(scores_flat + tie_break, dim=-1, descending=True)
    ranks = torch.empty_like(scores_flat)
    src = torch.arange(n_sub, device=scores_flat.device, dtype=scores_flat.dtype).unsqueeze(0).expand(n_s, n_sub)
    ranks.scatter_(1, order, src)
    return ranks


def reconstruct_subgrid_heterogeneity(
    q_v_col: torch.Tensor,       # [nz, N_s]
    q_c_col: torch.Tensor,       # [nz, N_s]
    q_i_col: torch.Tensor,       # [nz, N_s]
    q_p_col: torch.Tensor,       # [nz, N_s]
    f_cld: torch.Tensor,         # [nz, N_s] predicted subgrid cloud fraction in [0, 1]
    rain_col: torch.Tensor,      # [N_s]
    snow_col: torch.Tensor,      # [N_s]
    f_prec: torch.Tensor,        # [N_s] predicted subgrid precipitating area fraction in [0, 1]
    cell_ids: torch.Tensor,      # [N_s] int64 cell indices for uniform-surface tie-breaking
    ny: int,
    nx: int,
    step_count: int = 0,
    *,
    score_cld: torch.Tensor | None = None,   # [N_s, ny, nx] learned cloud spatial preference
    score_prec: torch.Tensor | None = None,  # [N_s, ny, nx] learned precip spatial preference
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct exact mass-conserving clear/cloudy & dry/rainy subcolumns `[nz, N_s, ny, nx]`.

    Uses the learned periodic spatial scores (`score_cld`, `score_prec`) predicted by
    `SubgridSpatialHead` directly from the `(ny, nx)` surface & moisture anomaly maps so
    convection and rain lock onto moist forests, forest-grassland vegetation-breeze fronts,
    warm patches, or pre-existing moist clusters exactly as learned from the genuine CRM
    teacher, while maintaining exact cell-mean mass conservation and genuine clear-sky zeros.
    """
    n_sub = max(int(ny * nx), 1)
    nz, n_s = q_c_col.shape
    dev = q_c_col.device
    dtype = q_c_col.dtype

    if n_sub == 1:
        q_v_4 = q_v_col.view(nz, n_s, 1, 1)
        q_c_4 = q_c_col.view(nz, n_s, 1, 1)
        q_i_4 = q_i_col.view(nz, n_s, 1, 1)
        q_p_4 = q_p_col.view(nz, n_s, 1, 1)
        q_n_4 = q_v_4 + q_c_4 + q_i_4
        return q_n_4, q_p_4, q_v_4, q_c_4, q_i_4, rain_col.view(n_s, 1, 1), snow_col.view(n_s, 1, 1)

    sub_idx = torch.arange(n_sub, device=dev, dtype=torch.int64).unsqueeze(0)  # [1, n_sub]
    shift = ((cell_ids + (int(step_count) // 12)) % n_sub).unsqueeze(1)        # [N_s, 1]
    rank_fallback = ((sub_idx + shift) % n_sub).to(dtype)                      # [N_s, n_sub]
    tie_break = (1.0 - rank_fallback / float(n_sub)) * 1.0e-4

    if score_cld is not None:
        rank_cld = _scores_to_ranks(score_cld.reshape(n_s, n_sub).to(dtype), tie_break)
    else:
        rank_cld = rank_fallback

    if score_prec is not None:
        rank_prec = _scores_to_ranks(score_prec.reshape(n_s, n_sub).to(dtype), tie_break)
    else:
        rank_prec = rank_cld

    # Cloud subcolumn multipliers m_cld [nz, N_s, n_sub] with exact mean == 1.0
    f_min = 1.0 / float(n_sub)
    f_c_eff = torch.clamp(f_cld, min=f_min, max=1.0)                           # [nz, N_s]
    w_cld = torch.clamp(f_c_eff.unsqueeze(-1) * float(n_sub) - rank_cld.unsqueeze(0), min=0.0, max=1.0)
    m_cld = w_cld / torch.clamp(w_cld.mean(dim=-1, keepdim=True), min=1.0e-6)  # exact mean == 1.0

    q_c_4 = (q_c_col.unsqueeze(-1) * m_cld).view(nz, n_s, ny, nx)
    q_i_4 = (q_i_col.unsqueeze(-1) * m_cld).view(nz, n_s, ny, nx)
    q_p_4 = (q_p_col.unsqueeze(-1) * m_cld).view(nz, n_s, ny, nx)
    q_v_4 = q_v_col.unsqueeze(-1).expand(nz, n_s, n_sub).reshape(nz, n_s, ny, nx).contiguous()
    q_n_4 = q_v_4 + q_c_4 + q_i_4

    # Precipitation subcolumn multipliers with exact mean == 1.0
    f_p_eff = torch.clamp(f_prec, min=f_min, max=1.0)                          # [N_s]
    w_pr = torch.clamp(f_p_eff.unsqueeze(-1) * float(n_sub) - rank_prec, min=0.0, max=1.0)
    m_pr = w_pr / torch.clamp(w_pr.mean(dim=-1, keepdim=True), min=1.0e-6)     # exact mean == 1.0
    rain_3 = (rain_col.unsqueeze(-1) * m_pr).view(n_s, ny, nx)
    snow_3 = (snow_col.unsqueeze(-1) * m_pr).view(n_s, ny, nx)
    return q_n_4, q_p_4, q_v_4, q_c_4, q_i_4, rain_3, snow_3


def apply_subgrid_surface_microclimate(
    crm_thm_0: torch.Tensor,   # [N_s, ny, nx] lowest-layer theta_m (updated with cell mean + damped anom)
    crm_qv_0: torch.Tensor,    # [N_s, ny, nx] lowest-layer q_v
    rho_0: torch.Tensor,       # [N_s] lowest-layer dry air density [kg m^-3]
    dz_0: torch.Tensor,        # [N_s] lowest-layer thickness [m]
    exner_0: torch.Tensor,     # [N_s] lowest-layer Exner function [-]
    shf_3d: torch.Tensor,      # [N_s, ny, nx] subgrid sensible heat flux [W m^-2]
    evap_3d: torch.Tensor,     # [N_s, ny, nx] subgrid evaporation [kg m^-2 s^-1]
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject exact zero-mean near-surface `(theta_m', q_v')` microclimate from tile `(SHF', evap')`.

    Ensures air immediately above warm/dry grassland or wildfire-burned tiles becomes warmer
    and drier while air above cool transpiring forest tiles stays cooler and moister, with
    zero perturbation to the cell-mean enthalpy or water vapor budget.
    """
    if int(shf_3d.shape[-2] * shf_3d.shape[-1]) <= 1:
        return crm_thm_0, crm_qv_0
    dm_0 = torch.clamp(rho_0 * dz_0, min=1.0).unsqueeze(-1).unsqueeze(-1)
    ex_0 = torch.clamp(exner_0, min=0.1).unsqueeze(-1).unsqueeze(-1)
    shf_a = shf_3d - shf_3d.mean(dim=(-2, -1), keepdim=True)
    evap_a = evap_3d - evap_3d.mean(dim=(-2, -1), keepdim=True)

    d_thm = torch.clamp((shf_a * dt) / (C.CP_D * dm_0 * ex_0), -3.0, 3.0)
    d_thm = d_thm - d_thm.mean(dim=(-2, -1), keepdim=True)
    thm_out = crm_thm_0 + d_thm

    qv_mean = crm_qv_0.mean(dim=(-2, -1), keepdim=True)
    d_qv = torch.clamp((evap_a * dt) / dm_0, -2.0e-3, 2.0e-3)
    d_qv = d_qv - d_qv.mean(dim=(-2, -1), keepdim=True)
    qv_cand = torch.clamp(crm_qv_0 + d_qv, min=0.0)
    qv_cand_mean = qv_cand.mean(dim=(-2, -1), keepdim=True)
    qv_out = torch.where(
        qv_cand_mean > 1.0e-10,
        qv_cand * (qv_mean / torch.clamp(qv_cand_mean, min=1.0e-10)),
        crm_qv_0,
    )
    return thm_out, qv_out


# ---------------------------------------------------------------------------
# 2. Pure-GEMM Residual Column MLP + FiLM-Modulated Periodic 2D Spatial Head
# ---------------------------------------------------------------------------
_N_SPATIAL_IN = 6  # (shf', lhf', t_rad', alb', qn_col', w_col)


class SubgridSpatialHead(nn.Module):
    """Size-invariant FiLM-modulated periodic 2D ConvNet predicting `(score_cld, score_prec)` on `(ny, nx)`.

    Conditioned on the column MLP latent state `h_col` (encoding ambient wind vector `(u, v)`,
    stratification, patch length scale `L_patch`, and Moran's `I`), this head works identically
    on 2D (`1 x nx`) and 3D (`ny x nx`) periodic CRM grids to learn forest-locking RCE,
    wind-shifted forest-grassland vegetation breezes, and convective cluster persistence directly
    from the genuine CRM teacher.
    """

    def __init__(self, col_hidden_dim: int, c_hid: int = 16) -> None:
        super().__init__()
        self.c_hid = int(c_hid)
        self.film = nn.Linear(int(col_hidden_dim), 2 * self.c_hid)
        self.conv1 = nn.Conv2d(_N_SPATIAL_IN, self.c_hid, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=0)
        self.conv_out = nn.Conv2d(self.c_hid, 2, kernel_size=1, padding=0)
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, h_col: torch.Tensor, sub_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(score_cld [B, ny, nx], score_prec [B, ny, nx])`."""
        gb = self.film(h_col)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        s = F.silu(self.conv1(_periodic_pad2d(sub_maps, 1)))
        s = F.silu(s * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta)
        s = s + F.silu(self.conv2(_periodic_pad2d(s, 1)))
        logits = self.conv_out(s)  # [B, 2, ny, nx]
        return logits[:, 0], logits[:, 1]


class CRMSelfDistillNet(nn.Module):
    """100% dense-GEMM residual column emulator + FiLM-modulated periodic 2D spatial head.

    Directly couples every vertical level from surface boundary layer (`k = 0`) to
    stratopause (`k = nz - 1`) in a single dense linear projection while exposing
    2 independent output heads for epistemic uncertainty-guided active learning and a
    periodic 2D spatial head for subgrid cloud & precipitation placement.
    """

    def __init__(self, nz: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.nz = int(nz)
        h = max(int(hidden_dim), 32)
        self.hidden_dim = h
        self.d_in = _N_PROF_IN * self.nz + _N_SCAL_IN
        self.d_out = _N_PROF_OUT * self.nz + _N_SCAL_OUT

        self.in_proj = nn.Linear(self.d_in, h)
        self.norm1 = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.norm2 = nn.LayerNorm(h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, h)
        self.head_a = nn.Linear(h, self.d_out)
        self.head_b = nn.Linear(h, self.d_out)
        self.spatial_head = SubgridSpatialHead(col_hidden_dim=h, c_hid=16)

        # Initialize output heads near zero so early predictions stay conservatively close to zero residual
        for head in (self.head_a, self.head_b):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        h = h + self.fc2(F.silu(self.fc1(self.norm1(h))))
        h = h + self.fc4(F.silu(self.fc3(self.norm2(h))))
        return h

    def forward_with_latent(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(prof_a, scal_a, prof_b, scal_b, h_latent)`."""
        b = int(x.shape[0])
        h = self.encode(x)
        ya = self.head_a(h)
        yb = self.head_b(h)
        n_p = _N_PROF_OUT * self.nz
        return (
            ya[:, :n_p].reshape(b, _N_PROF_OUT, self.nz),
            ya[:, n_p:],
            yb[:, :n_p].reshape(b, _N_PROF_OUT, self.nz),
            yb[:, n_p:],
            h,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(prof_a [B,8,nz], scal_a [B,3], prof_b [B,8,nz], scal_b [B,3])`."""
        pa, sa, pb, sb, _ = self.forward_with_latent(x)
        return pa, sa, pb, sb


# ---------------------------------------------------------------------------
# 3. GPU-Resident Dual-Zone Replay Buffer (Recency Ring + Long-Horizon Reservoir)
# ---------------------------------------------------------------------------
def extract_spatial_targets(
    qc_4d: torch.Tensor,      # [nz, B, ny, nx] post-CRM liquid cloud mixing ratio
    qi_4d: torch.Tensor,      # [nz, B, ny, nx] post-CRM ice cloud mixing ratio
    rain_3d: torch.Tensor,    # [B, ny, nx] post-CRM rain rate [kg m^-2 s^-1]
    snow_3d: torch.Tensor,    # [B, ny, nx] post-CRM snow rate [kg m^-2 s^-1]
    dm_col: torch.Tensor,     # [nz, B] dry air mass per layer [kg m^-2]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract normalized `[B, 2, ny, nx]` teacher cloud/precip spatial probability maps and `[B, 2]` active masks."""
    b, ny, nx = rain_3d.shape
    n_sub = max(int(ny * nx), 1)
    unif = 1.0 / float(n_sub)

    cwp = ((qc_4d + qi_4d) * dm_col.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)  # [B, ny, nx] kg m^-2
    cwp_sum = cwp.sum(dim=(-2, -1), keepdim=True)
    has_cld = (cwp_sum.squeeze(-1).squeeze(-1) > 1.0e-4).to(cwp.dtype)
    p_cld = torch.where(
        cwp_sum > 1.0e-4,
        cwp / torch.clamp(cwp_sum, min=1.0e-8),
        torch.full_like(cwp, unif),
    )

    pr_tot = torch.clamp(rain_3d + snow_3d, min=0.0)
    pr_sum = pr_tot.sum(dim=(-2, -1), keepdim=True)
    has_pr = (pr_sum.squeeze(-1).squeeze(-1) > 1.0e-7).to(pr_tot.dtype)
    p_pr = torch.where(
        pr_sum > 1.0e-7,
        pr_tot / torch.clamp(pr_sum, min=1.0e-10),
        torch.full_like(pr_tot, unif),
    )

    tgt_maps = torch.stack([p_cld, p_pr], dim=1).contiguous()   # [B, 2, ny, nx]
    has_map = torch.stack([has_cld, has_pr], dim=1).contiguous()  # [B, 2]
    return tgt_maps, has_map


def _resize_spatial_for_buffer(t: torch.Tensor, ny_b: int, nx_b: int, *, is_prob: bool = False) -> torch.Tensor:
    """Optionally pool very large 3D CRM maps (`ny > 32` or `nx > 32`) to keep replay VRAM bounded."""
    if int(t.shape[-2]) == ny_b and int(t.shape[-1]) == nx_b:
        return t
    out = F.adaptive_avg_pool2d(t.float(), (ny_b, nx_b))
    if is_prob:
        out = out / torch.clamp(out.sum(dim=(-2, -1), keepdim=True), min=1.0e-8)
    return out


class GPUReplayBuffer:
    """Zero-CPU-sync GPU ring + reservoir replay buffer preventing seasonal forgetting."""

    def __init__(
        self,
        capacity: int,
        nz: int,
        device: torch.device,
        *,
        ny: int = 1,
        nx: int = 1,
    ) -> None:
        self.capacity = max(int(capacity), 1024)
        self.nz = int(nz)
        self.d_in = _N_PROF_IN * self.nz + _N_SCAL_IN
        self.device = device
        self.ny_b = min(max(int(ny), 1), 32)
        self.nx_b = min(max(int(nx), 1), 32)
        self.has_spatial = (self.ny_b * self.nx_b) > 1
        self.ring_cap = self.capacity // 2
        self.res_cap = self.capacity - self.ring_cap
        self.ring_size = 0
        self.res_size = 0
        self.write_ptr = 0
        self.total_seen = 0

        # Compact float16 flat features + float32 conservation budgets/targets (~1.3 KB/col)
        self.x = torch.zeros((self.capacity, self.d_in), device=device, dtype=torch.float16)
        # Physics context needed to run exact conservation layer during training:
        # [rho_d, dz_c, exner, thm_post_ls, qn_post_ls, qp_pre, qc_pre, qi_pre, u_pre, v_pre] -> 10 profiles
        self.ctx_p = torch.zeros((self.capacity, 10, self.nz), device=device, dtype=torch.float32)
        # Scalar context: [temp_sfc, shf, evap, tau_x, tau_y] -> 5 scalars
        self.ctx_s = torch.zeros((self.capacity, 5), device=device, dtype=torch.float32)
        # Teacher targets: [thm_tgt, qn_tgt, qp_tgt, qc_tgt, qi_tgt, u_tgt, v_tgt, f_cld_tgt] -> 8 profiles
        self.tgt_p = torch.zeros((self.capacity, 8, self.nz), device=device, dtype=torch.float32)
        # Teacher scalar targets: [precip_rain, precip_snow, f_prec_tgt] -> 3 scalars
        self.tgt_s = torch.zeros((self.capacity, 3), device=device, dtype=torch.float32)
        # Optional 2D/3D spatial maps for SubgridSpatialHead (~8 MB for 65536 x 1 x 8)
        if self.has_spatial:
            self.sub_maps = torch.zeros(
                (self.capacity, _N_SPATIAL_IN, self.ny_b, self.nx_b), device=device, dtype=torch.float16
            )
            self.tgt_maps = torch.zeros(
                (self.capacity, 2, self.ny_b, self.nx_b), device=device, dtype=torch.float16
            )
            self.tgt_has_map = torch.zeros((self.capacity, 2), device=device, dtype=torch.float32)
        else:
            self.sub_maps = None
            self.tgt_maps = None
            self.tgt_has_map = None

    @property
    def size(self) -> int:
        return self.ring_size + self.res_size

    @torch.no_grad()
    def push(
        self,
        x: torch.Tensor,          # [B, d_in]
        ctx_p: torch.Tensor,      # [B, 10, nz]
        ctx_s: torch.Tensor,      # [B, 5]
        tgt_p: torch.Tensor,      # [B, 8, nz]
        tgt_s: torch.Tensor,      # [B, 3]
        sub_maps: torch.Tensor | None = None,     # [B, 6, ny, nx]
        tgt_maps: torch.Tensor | None = None,     # [B, 2, ny, nx]
        tgt_has_map: torch.Tensor | None = None,  # [B, 2]
    ) -> None:
        b = int(x.shape[0])
        if b == 0:
            return
        if b > self.ring_cap:
            perm = torch.randperm(b, device=self.device)[: self.ring_cap]
            x, ctx_p, ctx_s, tgt_p, tgt_s = x[perm], ctx_p[perm], ctx_s[perm], tgt_p[perm], tgt_s[perm]
            if sub_maps is not None:
                sub_maps = sub_maps[perm]
            if tgt_maps is not None:
                tgt_maps = tgt_maps[perm]
            if tgt_has_map is not None:
                tgt_has_map = tgt_has_map[perm]
            b = int(x.shape[0])

        # Clamp float16 features safely away from FP16 overflow (max 65504)
        x_f16 = torch.clamp(x, -100.0, 100.0).to(torch.float16)
        use_sp = (
            self.has_spatial
            and self.sub_maps is not None
            and self.tgt_maps is not None
            and self.tgt_has_map is not None
            and sub_maps is not None
            and tgt_maps is not None
            and tgt_has_map is not None
        )
        if use_sp:
            sm_f16 = torch.clamp(
                _resize_spatial_for_buffer(sub_maps, self.ny_b, self.nx_b, is_prob=False), -10.0, 10.0
            ).to(torch.float16)
            tm_f16 = torch.clamp(
                _resize_spatial_for_buffer(tgt_maps, self.ny_b, self.nx_b, is_prob=True), 0.0, 1.0
            ).to(torch.float16)

        # 1. Insert 75% into rolling recency ring [0, ring_cap)
        n_ring = max(1, (3 * b) // 4)
        idx_ring = (torch.arange(n_ring, device=self.device) + self.write_ptr) % self.ring_cap
        self.x[idx_ring] = x_f16[:n_ring]
        self.ctx_p[idx_ring] = ctx_p[:n_ring]
        self.ctx_s[idx_ring] = ctx_s[:n_ring]
        self.tgt_p[idx_ring] = tgt_p[:n_ring]
        self.tgt_s[idx_ring] = tgt_s[:n_ring]
        if use_sp:
            self.sub_maps[idx_ring] = sm_f16[:n_ring]
            self.tgt_maps[idx_ring] = tm_f16[:n_ring]
            self.tgt_has_map[idx_ring] = tgt_has_map[:n_ring]
        self.write_ptr = (self.write_ptr + n_ring) % self.ring_cap
        self.ring_size = min(self.ring_cap, self.ring_size + n_ring)

        # 2. Insert 25% into long-horizon uniform reservoir [ring_cap, capacity)
        n_res = b - n_ring
        if n_res > 0 and self.res_cap > 0:
            if self.res_size < self.res_cap:
                take = min(n_res, self.res_cap - self.res_size)
                idx_r = torch.arange(self.res_size, self.res_size + take, device=self.device) + self.ring_cap
                self.x[idx_r] = x_f16[n_ring : n_ring + take]
                self.ctx_p[idx_r] = ctx_p[n_ring : n_ring + take]
                self.ctx_s[idx_r] = ctx_s[n_ring : n_ring + take]
                self.tgt_p[idx_r] = tgt_p[n_ring : n_ring + take]
                self.tgt_s[idx_r] = tgt_s[n_ring : n_ring + take]
                if use_sp:
                    self.sub_maps[idx_r] = sm_f16[n_ring : n_ring + take]
                    self.tgt_maps[idx_r] = tm_f16[n_ring : n_ring + take]
                    self.tgt_has_map[idx_r] = tgt_has_map[n_ring : n_ring + take]
                self.res_size += take
            else:
                idx_r = torch.randint(self.ring_cap, self.capacity, (n_res,), device=self.device)
                self.x[idx_r] = x_f16[n_ring:]
                self.ctx_p[idx_r] = ctx_p[n_ring:]
                self.ctx_s[idx_r] = ctx_s[n_ring:]
                self.tgt_p[idx_r] = tgt_p[n_ring:]
                self.tgt_s[idx_r] = tgt_s[n_ring:]
                if use_sp:
                    self.sub_maps[idx_r] = sm_f16[n_ring:]
                    self.tgt_maps[idx_r] = tm_f16[n_ring:]
                    self.tgt_has_map[idx_r] = tgt_has_map[n_ring:]

        self.total_seen += b

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
    ]:
        n = min(int(batch_size), self.size)
        if self.res_size > 0 and self.ring_size > 0:
            n_ring_s = n // 2
            n_res_s = n - n_ring_s
            i1 = torch.randint(0, self.ring_size, (n_ring_s,), device=self.device)
            i2 = torch.randint(self.ring_cap, self.ring_cap + self.res_size, (n_res_s,), device=self.device)
            idx = torch.cat([i1, i2], dim=0)
        elif self.ring_size > 0:
            idx = torch.randint(0, self.ring_size, (n,), device=self.device)
        else:
            idx = torch.randint(self.ring_cap, self.ring_cap + self.res_size, (n,), device=self.device)
        sm = self.sub_maps[idx].to(torch.float32) if self.sub_maps is not None else None
        tm = self.tgt_maps[idx].to(torch.float32) if self.tgt_maps is not None else None
        hm = self.tgt_has_map[idx] if self.tgt_has_map is not None else None
        return (
            self.x[idx].to(torch.float32),
            self.ctx_p[idx],
            self.ctx_s[idx],
            self.tgt_p[idx],
            self.tgt_s[idx],
            sm,
            tm,
            hm,
        )


# ---------------------------------------------------------------------------
# 4. Self-Distillation Controller State & Online Training / Inference Step
# ---------------------------------------------------------------------------
@dataclass
class CRMSelfDistillState:
    """Prognostic state for online active-learning 2D/3D CRM self-distillation."""

    net: CRMSelfDistillNet
    optimizer: torch.optim.Optimizer
    replay: GPUReplayBuffer
    stride: int
    warmup_steps: int
    dwell_steps: int
    n_cells: int
    cohort_id: torch.Tensor             # [n_cells] int64 spatially stratified cohort in 0..stride-1
    last_uncertainty: torch.Tensor      # [n_cells] float32 epistemic spread from previous step
    steps_seen: int = 0
    grad_updates: int = 0
    ema_loss: float = 1.0
    ema_val_rmse_thm: float = 1.0
    ema_val_rmse_qn: float = 1.0e-3
    teacher_frac_last: float = 1.0


def init_crm_self_distill_state(
    n_cells: int,
    nz: int,
    device: torch.device,
    *,
    stride: int = 13,
    warmup_steps: int = 24,
    dwell_steps: int = 12,
    hidden_dim: int = 192,
    replay_size: int = 65536,
    lr: float = 1.5e-3,
    ny: int = 1,
    nx: int = 8,
) -> CRMSelfDistillState:
    """Initialize online self-distilling 2D/3D CRM student network, optimizer, and GPU replay buffer."""
    st = max(int(stride), 2)
    net = CRMSelfDistillNet(nz=nz, hidden_dim=hidden_dim).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1.0e-4)
    replay = GPUReplayBuffer(capacity=replay_size, nz=nz, device=device, ny=ny, nx=nx)
    # Quasi-uniform spatial cohort hash across all latitudes/longitudes (coprime Knuth multiplier)
    cell_ids = torch.arange(n_cells, device=device, dtype=torch.int64)
    cohort_id = ((cell_ids * 2654435761) % st).to(torch.int64)
    return CRMSelfDistillState(
        net=net,
        optimizer=opt,
        replay=replay,
        stride=st,
        warmup_steps=max(int(warmup_steps), 1),
        dwell_steps=max(int(dwell_steps), 1),
        n_cells=int(n_cells),
        cohort_id=cohort_id,
        last_uncertainty=torch.zeros(n_cells, device=device, dtype=torch.float32),
    )


def select_teacher_indices(state: CRMSelfDistillState) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return `(teacher_idx, student_idx)` or `None` during warmup (100% genuine 2D/3D CRM).

    Selects a fixed-size teacher set of `n_teach = ceil(n_cells / stride)` columns:
    - 85% from the current spatially stratified rotating cohort (unbroken convective memory
      across `dwell_steps` continuous steps),
    - 15% from the highest epistemic-uncertainty columns globally (active OOD fallback).
    """
    if state.steps_seen < state.warmup_steps:
        return None
    nc = state.n_cells
    n_teach = max(1, (nc + state.stride - 1) // state.stride)
    if n_teach >= nc:
        return None

    active_cohort = ((state.steps_seen - state.warmup_steps) // state.dwell_steps) % state.stride
    rot_mask = state.cohort_id == active_cohort
    rot_idx = torch.nonzero(rot_mask, as_tuple=False).squeeze(-1)

    n_ood = max(1, n_teach // 6) if n_teach >= 12 else 0
    n_rot_keep = min(int(rot_idx.numel()), max(n_teach - n_ood, 1))
    rot_sel = rot_idx[:n_rot_keep]

    n_rem = n_teach - n_rot_keep
    if n_rem > 0:
        # Mask out already-selected rotating cohort and pick highest-uncertainty OOD columns
        unc = state.last_uncertainty.clone()
        unc[rot_sel] = -1.0e9
        _, ood_sel = torch.topk(unc, k=n_rem, largest=True, sorted=False)
        teacher_idx = torch.cat([rot_sel, ood_sel], dim=0)
    else:
        teacher_idx = rot_sel

    teach_mask = torch.zeros(nc, device=teacher_idx.device, dtype=torch.bool)
    teach_mask[teacher_idx] = True
    student_idx = torch.nonzero(~teach_mask, as_tuple=False).squeeze(-1)
    return teacher_idx, student_idx


def _compute_distill_loss(
    pred_a: dict[str, torch.Tensor],
    pred_b: dict[str, torch.Tensor],
    tgt_p: torch.Tensor,   # [B, 8, nz]
    tgt_s: torch.Tensor,   # [B, 3]
    exner: torch.Tensor,   # [nz, B]
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mass-and-temperature-scaled multi-variable loss across both epistemic heads."""
    tp = tgt_p.permute(1, 2, 0).contiguous()  # [8, nz, B]
    thm_t, qn_t, qp_t, qc_t, qi_t, u_t, v_t, fcld_t = tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], tp[6], tp[7]
    t_tgt = thm_t * exner  # Temperature space [K] ensures uniform troposphere/stratosphere weighting
    pr_t = tgt_s[:, 0]
    ps_t = tgt_s[:, 1]
    fprec_t = tgt_s[:, 2]

    def _head_loss(pr: dict[str, torch.Tensor]) -> torch.Tensor:
        l_temp = F.smooth_l1_loss(pr["theta_m"] * exner, t_tgt)
        l_qn = F.smooth_l1_loss(pr["q_n"] * 1.0e3, qn_t * 1.0e3)
        l_qp = F.smooth_l1_loss(pr["q_p"] * 2.0e3, qp_t * 2.0e3)
        l_qc = F.smooth_l1_loss(pr["q_c"] * 5.0e3, qc_t * 5.0e3)
        l_qi = F.smooth_l1_loss(pr["q_i"] * 5.0e3, qi_t * 5.0e3)
        l_u = F.smooth_l1_loss(pr["u"] * 0.2, u_t * 0.2)
        l_v = F.smooth_l1_loss(pr["v"] * 0.2, v_t * 0.2)
        l_fcld = F.smooth_l1_loss(pr["f_cld"], fcld_t)
        l_fprec = F.smooth_l1_loss(pr["f_prec"], fprec_t)
        l_pr = F.smooth_l1_loss(
            torch.log1p(torch.clamp(pr["precip_rain"] * 86400.0, min=0.0)),
            torch.log1p(torch.clamp(pr_t * 86400.0, min=0.0)),
        )
        l_ps = F.smooth_l1_loss(
            torch.log1p(torch.clamp(pr["precip_snow"] * 86400.0, min=0.0)),
            torch.log1p(torch.clamp(ps_t * 86400.0, min=0.0)),
        )
        return (
            2.0 * l_temp + 2.5 * l_qn + 0.8 * l_qp + 1.5 * (l_qc + l_qi)
            + 0.5 * (l_u + l_v) + 1.2 * (l_pr + l_ps) + 1.8 * l_fcld + 0.8 * l_fprec
        )

    loss = 0.5 * (_head_loss(pred_a) + _head_loss(pred_b))
    with torch.no_grad():
        rmse_t = torch.sqrt(F.mse_loss(0.5 * (pred_a["theta_m"] + pred_b["theta_m"]) * exner, t_tgt))
        rmse_qn = torch.sqrt(F.mse_loss(0.5 * (pred_a["q_n"] + pred_b["q_n"]), qn_t))
    return loss, rmse_t, rmse_qn


def _compute_spatial_placement_loss(
    score_cld: torch.Tensor,   # [B, ny, nx]
    score_prec: torch.Tensor,  # [B, ny, nx]
    tgt_maps: torch.Tensor,    # [B, 2, ny, nx] normalized probability targets
    has_map: torch.Tensor,     # [B, 2] active masks (has_cld, has_prec)
) -> torch.Tensor:
    """Cross-entropy spatial placement loss teaching where clouds & rain form on `(ny, nx)`."""
    b = int(score_cld.shape[0])
    sc_flat = score_cld.reshape(b, -1)
    sp_flat = score_prec.reshape(b, -1)
    tc_flat = tgt_maps[:, 0].reshape(b, -1)
    tp_flat = tgt_maps[:, 1].reshape(b, -1)

    ce_cld = -(tc_flat * F.log_softmax(sc_flat, dim=-1)).sum(dim=-1)
    ce_pr = -(tp_flat * F.log_softmax(sp_flat, dim=-1)).sum(dim=-1)

    m_cld = has_map[:, 0]
    m_pr = has_map[:, 1]
    loss_cld = (ce_cld * m_cld).sum() / torch.clamp(m_cld.sum(), min=1.0)
    loss_pr = (ce_pr * m_pr).sum() / torch.clamp(m_pr.sum(), min=1.0)
    return 0.5 * (loss_cld + loss_pr)


def train_student_online(
    state: CRMSelfDistillState,
    x_fresh: torch.Tensor,       # [B_t, d_in]
    ctx_p_fresh: torch.Tensor,   # [B_t, 10, nz]
    ctx_s_fresh: torch.Tensor,   # [B_t, 5]
    tgt_p_fresh: torch.Tensor,   # [B_t, 8, nz]
    tgt_s_fresh: torch.Tensor,   # [B_t, 3]
    dt: float,
    *,
    sub_maps_fresh: torch.Tensor | None = None,     # [B_t, 6, ny, nx]
    tgt_maps_fresh: torch.Tensor | None = None,     # [B_t, 2, ny, nx]
    tgt_has_map_fresh: torch.Tensor | None = None,  # [B_t, 2]
    n_grad_steps: int = 2,
    batch_size: int = 512,
) -> None:
    """Push fresh teacher CRM pairs to GPU replay buffer and execute online AdamW updates."""
    state.replay.push(
        x_fresh, ctx_p_fresh, ctx_s_fresh, tgt_p_fresh, tgt_s_fresh,
        sub_maps=sub_maps_fresh, tgt_maps=tgt_maps_fresh, tgt_has_map=tgt_has_map_fresh,
    )
    if state.replay.size < 64:
        return

    state.net.train()
    n_steps = max(int(n_grad_steps), 1)
    # Use extra gradient steps during warmup or when OOD surge raises validation RMSE
    if state.steps_seen < state.warmup_steps or state.ema_val_rmse_thm > 2.5:
        n_steps = max(n_steps, 4)

    b_fresh = int(x_fresh.shape[0])
    half_b = min(batch_size // 2, b_fresh)
    use_sp = (
        state.replay.has_spatial
        and sub_maps_fresh is not None
        and tgt_maps_fresh is not None
        and tgt_has_map_fresh is not None
    )
    if use_sp:
        sm_fresh_b = _resize_spatial_for_buffer(sub_maps_fresh, state.replay.ny_b, state.replay.nx_b, is_prob=False)
        tm_fresh_b = _resize_spatial_for_buffer(tgt_maps_fresh, state.replay.ny_b, state.replay.nx_b, is_prob=True)
    else:
        sm_fresh_b = None
        tm_fresh_b = None

    last_loss = None
    last_rthm = None
    last_rqn = None
    for _ in range(n_steps):
        # Mix 50% fresh teacher columns + 50% long-horizon replay columns
        f_idx = torch.randint(0, b_fresh, (half_b,), device=x_fresh.device)
        xr, cpr, csr, tpr, tsr, smr, tmr, hmr = state.replay.sample(batch_size - half_b)
        xb = torch.cat([x_fresh[f_idx], xr], dim=0)
        cpb = torch.cat([ctx_p_fresh[f_idx], cpr], dim=0)
        csb = torch.cat([ctx_s_fresh[f_idx], csr], dim=0)
        tpb = torch.cat([tgt_p_fresh[f_idx], tpr], dim=0)
        tsb = torch.cat([tgt_s_fresh[f_idx], tsr], dim=0)

        pa_raw, sa_raw, pb_raw, sb_raw, h_lat = state.net.forward_with_latent(xb)
        # Unpack physics context [B, 10, nz] -> [nz, B]
        cp_t = cpb.permute(1, 2, 0).contiguous()
        rho_b, dz_b, ex_b = cp_t[0], cp_t[1], cp_t[2]
        thm_ls_b, qn_ls_b = cp_t[3], cp_t[4]
        qp_b, qc_b, qi_b, u_b, v_b = cp_t[5], cp_t[6], cp_t[7], cp_t[8], cp_t[9]
        tsfc_b, shf_b, evap_b, tx_b, ty_b = csb[:, 0], csb[:, 1], csb[:, 2], csb[:, 3], csb[:, 4]

        pred_a = apply_exact_conservation_closure(
            pa_raw, sa_raw, rho_b, dz_b, ex_b, tsfc_b, thm_ls_b, qn_ls_b,
            qp_b, qc_b, qi_b, u_b, v_b, shf_b, evap_b, tx_b, ty_b, dt,
        )
        pred_b = apply_exact_conservation_closure(
            pb_raw, sb_raw, rho_b, dz_b, ex_b, tsfc_b, thm_ls_b, qn_ls_b,
            qp_b, qc_b, qi_b, u_b, v_b, shf_b, evap_b, tx_b, ty_b, dt,
        )
        loss, r_thm, r_qn = _compute_distill_loss(pred_a, pred_b, tpb, tsb, ex_b, dt)

        if use_sp and sm_fresh_b is not None and tm_fresh_b is not None and smr is not None and tmr is not None and hmr is not None:
            smb = torch.cat([sm_fresh_b[f_idx], smr], dim=0)
            tmb = torch.cat([tm_fresh_b[f_idx], tmr], dim=0)
            hmb = torch.cat([tgt_has_map_fresh[f_idx], hmr], dim=0)
            sc_pred, sp_pred = state.net.spatial_head(h_lat, smb)
            loss_sp = _compute_spatial_placement_loss(sc_pred, sp_pred, tmb, hmb)
            loss = loss + 0.6 * loss_sp

        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.net.parameters(), max_norm=2.0)
        state.optimizer.step()
        state.grad_updates += 1
        last_loss, last_rthm, last_rqn = loss, r_thm, r_qn

    # Update scalar telemetry only every 16 steps to avoid per-step CPU-GPU sync stalls!
    if last_loss is not None and (state.steps_seen % 16 == 0 or state.steps_seen == state.warmup_steps - 1):
        lv = float(last_loss.detach().item())
        rt = float(last_rthm.detach().item())
        rq = float(last_rqn.detach().item())
        alpha = 0.2
        state.ema_loss = (1.0 - alpha) * state.ema_loss + alpha * lv
        state.ema_val_rmse_thm = (1.0 - alpha) * state.ema_val_rmse_thm + alpha * rt
        state.ema_val_rmse_qn = (1.0 - alpha) * state.ema_val_rmse_qn + alpha * rq


@torch.no_grad()
def predict_student_columns(
    state: CRMSelfDistillState,
    x_stu: torch.Tensor,         # [N_s, d_in]
    rho_d: torch.Tensor,         # [nz, N_s]
    dz_c: torch.Tensor,          # [nz, N_s]
    exner: torch.Tensor,         # [nz, N_s]
    temp_sfc: torch.Tensor,      # [N_s]
    thm_post_ls: torch.Tensor,   # [nz, N_s]
    qn_post_ls: torch.Tensor,    # [nz, N_s]
    qp_pre: torch.Tensor,        # [nz, N_s]
    qc_pre: torch.Tensor,        # [nz, N_s]
    qi_pre: torch.Tensor,        # [nz, N_s]
    u_pre: torch.Tensor,         # [nz, N_s]
    v_pre: torch.Tensor,         # [nz, N_s]
    shf: torch.Tensor,           # [N_s]
    evap: torch.Tensor,          # [N_s]
    tau_x: torch.Tensor,         # [N_s]
    tau_y: torch.Tensor,         # [N_s]
    dt: float,
    *,
    sub_maps_stu: torch.Tensor | None = None,  # [N_s, 6, ny, nx] optional spatial maps for placement
    chunk_size: int = 16384,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run batched student inference with exact conservation closure, spatial scores, and epistemic uncertainty."""
    state.net.eval()
    n_s = int(x_stu.shape[0])
    nz = int(rho_d.shape[0])
    dev = x_stu.device

    pa_all = torch.empty((n_s, _N_PROF_OUT, nz), device=dev, dtype=torch.float32)
    sa_all = torch.empty((n_s, _N_SCAL_OUT), device=dev, dtype=torch.float32)
    pb_all = torch.empty((n_s, _N_PROF_OUT, nz), device=dev, dtype=torch.float32)
    sb_all = torch.empty((n_s, _N_SCAL_OUT), device=dev, dtype=torch.float32)
    want_sp = sub_maps_stu is not None and int(sub_maps_stu.shape[-2] * sub_maps_stu.shape[-1]) > 1
    if want_sp:
        ny_s, nx_s = int(sub_maps_stu.shape[-2]), int(sub_maps_stu.shape[-1])
        sc_all = torch.empty((n_s, ny_s, nx_s), device=dev, dtype=torch.float32)
        sp_all = torch.empty((n_s, ny_s, nx_s), device=dev, dtype=torch.float32)
    else:
        sc_all = None
        sp_all = None

    # Use FP16 autocast on MPS and BF16 on CUDA for peak Matrix/Tensor Core throughput!
    use_cuda_amp = (dev.type == "cuda")
    use_mps_amp = (dev.type == "mps")
    for i0 in range(0, n_s, chunk_size):
        i1 = min(i0 + chunk_size, n_s)
        if use_cuda_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pa, sa, pb, sb, h_lat = state.net.forward_with_latent(x_stu[i0:i1])
                if want_sp and sub_maps_stu is not None:
                    sc_c, sp_c = state.net.spatial_head(h_lat, sub_maps_stu[i0:i1])
        elif use_mps_amp:
            with torch.amp.autocast("mps", dtype=torch.float16):
                pa, sa, pb, sb, h_lat = state.net.forward_with_latent(x_stu[i0:i1])
                if want_sp and sub_maps_stu is not None:
                    sc_c, sp_c = state.net.spatial_head(h_lat, sub_maps_stu[i0:i1])
        else:
            pa, sa, pb, sb, h_lat = state.net.forward_with_latent(x_stu[i0:i1])
            if want_sp and sub_maps_stu is not None:
                sc_c, sp_c = state.net.spatial_head(h_lat, sub_maps_stu[i0:i1])
        pa_all[i0:i1] = pa.float()
        sa_all[i0:i1] = sa.float()
        pb_all[i0:i1] = pb.float()
        sb_all[i0:i1] = sb.float()
        if want_sp and sc_all is not None and sp_all is not None:
            sc_all[i0:i1] = sc_c.float()
            sp_all[i0:i1] = sp_c.float()

    # Ensemble-mean raw prediction -> single exact conservation projection
    p_mean = 0.5 * (pa_all + pb_all)
    s_mean = 0.5 * (sa_all + sb_all)
    out = apply_exact_conservation_closure(
        p_mean, s_mean, rho_d, dz_c, exner, temp_sfc, thm_post_ls, qn_post_ls,
        qp_pre, qc_pre, qi_pre, u_pre, v_pre, shf, evap, tau_x, tau_y, dt,
    )
    if sc_all is not None and sp_all is not None:
        out["score_cld"] = sc_all
        out["score_prec"] = sp_all
    # Epistemic disagreement score per column [N_s] for active OOD teacher routing
    unc = (pa_all - pb_all).abs().mean(dim=(1, 2)) + (sa_all - sb_all).abs().mean(dim=1)
    return out, unc


def pack_teacher_context_and_targets(
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    exner: torch.Tensor,
    temp_sfc: torch.Tensor,
    thm_post_ls: torch.Tensor,
    qn_post_ls: torch.Tensor,
    qp_pre: torch.Tensor,
    qc_pre: torch.Tensor,
    qi_pre: torch.Tensor,
    u_pre: torch.Tensor,
    v_pre: torch.Tensor,
    shf: torch.Tensor,
    evap: torch.Tensor,
    tau_x: torch.Tensor,
    tau_y: torch.Tensor,
    thm_post: torch.Tensor,
    qn_post: torch.Tensor,
    qp_post: torch.Tensor,
    qc_post: torch.Tensor,
    qi_post: torch.Tensor,
    u_post: torch.Tensor,
    v_post: torch.Tensor,
    precip_rain: torch.Tensor,
    precip_snow: torch.Tensor,
    f_cld_tgt: torch.Tensor,
    f_prec_tgt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack teacher CRM physics context and post-CRM targets (including subgrid `f_cld`, `f_prec`)."""
    ctx_p = torch.stack([
        rho_d, dz_c, exner, thm_post_ls, qn_post_ls, qp_pre, qc_pre, qi_pre, u_pre, v_pre,
    ], dim=0).permute(2, 0, 1).contiguous()
    ctx_s = torch.stack([temp_sfc, shf, evap, tau_x, tau_y], dim=1).contiguous()
    tgt_p = torch.stack([
        thm_post, qn_post, qp_post, qc_post, qi_post, u_post, v_post, f_cld_tgt,
    ], dim=0).permute(2, 0, 1).contiguous()
    tgt_s = torch.stack([precip_rain, precip_snow, f_prec_tgt], dim=1).contiguous()
    return ctx_p, ctx_s, tgt_p, tgt_s


# ---------------------------------------------------------------------------
# 5. Paired Online Per-Subcolumn ICA Radiation Self-Distillation
# ---------------------------------------------------------------------------
_N_RAD_PROF_IN = 9    # (temp, ln_p, qv, qc, qi, dm_dry, o3, q_dust, tau_aero_sw)
_N_RAD_SCAL_IN = 13   # (mu0, s_inc, day_mask, alb_vd, alb_vf, alb_nd, alb_nf, emiss, t_sfc, co2, ch4, n2o, cwp_tot)
_N_RAD_SCAL_OUT = 7   # (sw1_dir_f, sw1_dif_f, sw2_dir_f, sw2_dif_f, sw_atm_f, lw_dn_raw, olr_raw)


def extract_rad_distill_features(
    pres_c: torch.Tensor,       # [nz, M]
    temp_c: torch.Tensor,       # [nz, M]
    dz_c: torch.Tensor,         # [nz, M]
    rho_d_c: torch.Tensor,      # [nz, M]
    qv_c: torch.Tensor,         # [nz, M]
    qc_c: torch.Tensor,         # [nz, M]
    qi_c: torch.Tensor,         # [nz, M]
    t_sfc: torch.Tensor,        # [M]
    alb_vd: torch.Tensor,       # [M]
    alb_vf: torch.Tensor,       # [M]
    alb_nd: torch.Tensor,       # [M]
    alb_nf: torch.Tensor,       # [M]
    emiss_sfc: torch.Tensor,    # [M]
    mu0: torch.Tensor,          # [M]
    s_toa: torch.Tensor,        # [M]
    co2_ppmv: float,
    ch4_ppmv: float = 0.80825,
    n2o_ppmv: float = 0.27302,
    *,
    o3_vmr_c: torch.Tensor | None = None,
    q_dust_c: torch.Tensor | None = None,
    tau_aero_sw_c: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flat dimensionless per-subcolumn ICA radiation features `[M, 9*nz + 13]` plus `(cp_layer, s_inc)`."""
    nz, m_col = pres_c.shape
    dev = pres_c.device
    dtype = pres_c.dtype

    dm_dry = torch.clamp(rho_d_c * dz_c, min=1.0e-6)
    cp_layer = (C.CP_D + C.CP_V * torch.clamp(qv_c, min=0.0)) * dm_dry  # [nz, M] J m^-2 K^-1
    s_inc = torch.clamp(mu0, min=0.0, max=1.0) * torch.clamp(s_toa, min=0.0)  # [M] W m^-2
    cwp_tot = ((torch.clamp(qc_c, min=0.0) + torch.clamp(qi_c, min=0.0)) * dm_dry).sum(dim=0)  # [M] kg m^-2

    o3_ch = torch.log1p(torch.clamp(o3_vmr_c, min=0.0) * 1.0e6) if o3_vmr_c is not None else torch.zeros_like(pres_c)
    dust_ch = torch.log1p(torch.clamp(q_dust_c, min=0.0) * 1.0e6) if q_dust_c is not None else torch.zeros_like(pres_c)
    aero_ch = torch.log1p(torch.clamp(tau_aero_sw_c, min=0.0) * 10.0) if tau_aero_sw_c is not None else torch.zeros_like(pres_c)

    prof = torch.stack([
        (temp_c - 250.0) * 0.02,
        (torch.log(torch.clamp(pres_c, min=1.0)) - 10.0) * 0.4,
        torch.log1p(torch.clamp(qv_c, min=0.0) * 1.0e3),
        torch.log1p(torch.clamp(qc_c, min=0.0) * 1.0e4),
        torch.log1p(torch.clamp(qi_c, min=0.0) * 1.0e4),
        torch.log1p(dm_dry * 1.0e-3),
        o3_ch,
        dust_ch,
        aero_ch,
    ], dim=0)  # [9, nz, M]

    co2_f = math.log(max(float(co2_ppmv), 1.0) / 284.317)
    ch4_f = math.log(max(float(ch4_ppmv), 0.01) / 0.80825)
    n2o_f = math.log(max(float(n2o_ppmv), 0.01) / 0.27302)
    day_mask = (mu0 > 1.0e-4).to(dtype)

    scal = torch.stack([
        torch.clamp(mu0, 0.0, 1.0),
        torch.clamp(s_inc * (1.0 / 1361.0), 0.0, 4.0),
        day_mask,
        torch.clamp(alb_vd, 0.0, 1.0),
        torch.clamp(alb_vf, 0.0, 1.0),
        torch.clamp(alb_nd, 0.0, 1.0),
        torch.clamp(alb_nf, 0.0, 1.0),
        torch.clamp(emiss_sfc, 0.0, 1.0),
        torch.clamp((t_sfc - 280.0) * 0.03, -6.0, 6.0),
        torch.full((m_col,), co2_f, device=dev, dtype=dtype),
        torch.full((m_col,), ch4_f, device=dev, dtype=dtype),
        torch.full((m_col,), n2o_f, device=dev, dtype=dtype),
        torch.log1p(torch.clamp(cwp_tot, min=0.0) * 10.0),
    ], dim=1)  # [M, 13]

    prof_flat = prof.permute(2, 0, 1).reshape(m_col, _N_RAD_PROF_IN * nz)
    x = torch.cat([prof_flat, scal], dim=1).contiguous()
    return x, cp_layer, s_inc


def apply_exact_rad_conservation_closure(
    raw_qrad: torch.Tensor,    # [M, nz]
    raw_scal: torch.Tensor,    # [M, 7]
    cp_layer: torch.Tensor,    # [nz, M] moist layer heat capacity [J m^-2 K^-1]
    s_inc: torch.Tensor,       # [M] incident TOA solar flux mu0 * s_toa [W m^-2]
    alb_vd: torch.Tensor,      # [M]
    alb_vf: torch.Tensor,      # [M]
    alb_nd: torch.Tensor,      # [M]
    alb_nf: torch.Tensor,      # [M]
    emiss_sfc: torch.Tensor,   # [M]
    t_sfc: torch.Tensor,       # [M]
) -> dict[str, torch.Tensor]:
    """Project raw radiation NN outputs onto the exact per-subcolumn radiative energy manifold.

    Guarantees:
      1. Exact zero shortwave fluxes (`sw1_dir, sw1_dif, sw2_dir, sw2_dif, asr == 0`) on night side.
      2. Bounded physical transmittances (`sw_dn_tot <= s_inc`, `f_sw_sfc_net <= asr <= s_inc`).
      3. Direct analytical response to each subcolumn's true surface albedos `(alb_vd..alb_nf)`
         and surface emissivity/temperature `(emiss_sfc, t_sfc)`.
      4. Exact TOA-atmosphere-surface column radiative energy closure to float32 machine precision:
         `sum_k cp_layer_k * q_rad_k == (asr - olr) - (f_sw_sfc_net + f_lw_sfc_net)`.
    """
    # Bounded SW surface transmittance fractions
    s1d_f = torch.sigmoid(raw_scal[:, 0] - 1.5)
    s1f_f = torch.sigmoid(raw_scal[:, 1] - 2.0)
    s2d_f = torch.sigmoid(raw_scal[:, 2] - 1.5)
    s2f_f = torch.sigmoid(raw_scal[:, 3] - 2.0)
    s_sum = s1d_f + s1f_f + s2d_f + s2f_f
    s_norm = 1.0 / torch.clamp(s_sum, min=1.0)

    sw1_dir = s_inc * (s1d_f * s_norm)
    sw1_dif = s_inc * (s1f_f * s_norm)
    sw2_dir = s_inc * (s2d_f * s_norm)
    sw2_dif = s_inc * (s2f_f * s_norm)

    f_sw_sfc_net = (
        (1.0 - alb_vd) * sw1_dir
        + (1.0 - alb_vf) * sw1_dif
        + (1.0 - alb_nd) * sw2_dir
        + (1.0 - alb_nf) * sw2_dif
    )
    f_atm_sw = torch.sigmoid(raw_scal[:, 4] - 1.5)
    asr = f_sw_sfc_net + f_atm_sw * torch.clamp(s_inc - f_sw_sfc_net, min=0.0)

    # Climate-invariant LW fluxes scaled around surface Planck emission sigma * T_sfc^4
    eb_sfc = C.STEFAN_BOLTZMANN * (torch.clamp(t_sfc, min=120.0) ** 4)
    lw_dn = torch.clamp(F.softplus(raw_scal[:, 5] + 0.8) * 0.65 * eb_sfc, min=5.0, max=850.0)
    olr = torch.clamp(F.softplus(raw_scal[:, 6] + 0.5) * 0.65 * eb_sfc, min=15.0, max=750.0)
    f_lw_sfc_net = emiss_sfc * (lw_dn - eb_sfc)

    # Atmospheric radiative heating rate [nz, M] [K s^-1] with exact column flux-divergence closure
    q_rad_raw = torch.tanh(raw_qrad.transpose(0, 1).contiguous() * 0.05) * (120.0 / C.SECONDS_PER_DAY)
    f_atm_exact = (asr - olr) - (f_sw_sfc_net + f_lw_sfc_net)
    f_atm_raw = (cp_layer * q_rad_raw).sum(dim=0)
    cp_tot = torch.clamp(cp_layer.sum(dim=0), min=1.0)
    q_rad_cons = q_rad_raw + ((f_atm_exact - f_atm_raw) / cp_tot).unsqueeze(0)

    return {
        "q_rad": q_rad_cons,
        "sw1_dir": sw1_dir,
        "sw1_dif": sw1_dif,
        "sw2_dir": sw2_dir,
        "sw2_dif": sw2_dif,
        "lw_dn": lw_dn,
        "olr": olr,
        "asr": asr,
    }


class RadSelfDistillNet(nn.Module):
    """Pure-GEMM 2-head epistemic residual column MLP for online ICA radiation self-distillation."""

    def __init__(self, nz: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.nz = int(nz)
        h = max(int(hidden_dim), 32)
        self.hidden_dim = h
        self.d_in = _N_RAD_PROF_IN * self.nz + _N_RAD_SCAL_IN
        self.d_out = self.nz + _N_RAD_SCAL_OUT

        self.in_proj = nn.Linear(self.d_in, h)
        self.norm1 = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.norm2 = nn.LayerNorm(h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, h)
        self.head_a = nn.Linear(h, self.d_out)
        self.head_b = nn.Linear(h, self.d_out)

        for head in (self.head_a, self.head_b):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(qrad_a [B, nz], scal_a [B, 7], qrad_b [B, nz], scal_b [B, 7])`."""
        h = F.silu(self.in_proj(x))
        h = h + self.fc2(F.silu(self.fc1(self.norm1(h))))
        h = h + self.fc4(F.silu(self.fc3(self.norm2(h))))
        ya = self.head_a(h)
        yb = self.head_b(h)
        return ya[:, : self.nz], ya[:, self.nz :], yb[:, : self.nz], yb[:, self.nz :]


class GPURadReplayBuffer:
    """Zero-CPU-sync GPU ring + reservoir replay buffer for online radiation self-distillation."""

    def __init__(self, capacity: int, nz: int, device: torch.device) -> None:
        self.capacity = max(int(capacity), 1024)
        self.nz = int(nz)
        self.d_in = _N_RAD_PROF_IN * self.nz + _N_RAD_SCAL_IN
        self.device = device
        self.ring_cap = self.capacity // 2
        self.res_cap = self.capacity - self.ring_cap
        self.ring_size = 0
        self.res_size = 0
        self.write_ptr = 0
        self.total_seen = 0

        self.x = torch.zeros((self.capacity, self.d_in), device=device, dtype=torch.float16)
        self.cp_layer = torch.zeros((self.capacity, self.nz), device=device, dtype=torch.float32)
        # Scalar context: [s_inc, alb_vd, alb_vf, alb_nd, alb_nf, emiss_sfc, t_sfc]
        self.ctx_s = torch.zeros((self.capacity, 7), device=device, dtype=torch.float32)
        self.tgt_qrad = torch.zeros((self.capacity, self.nz), device=device, dtype=torch.float32)
        # Scalar flux targets: [sw1_dir, sw1_dif, sw2_dir, sw2_dif, lw_dn, olr, asr]
        self.tgt_s = torch.zeros((self.capacity, 7), device=device, dtype=torch.float32)

    @property
    def size(self) -> int:
        return self.ring_size + self.res_size

    @torch.no_grad()
    def push(
        self,
        x: torch.Tensor,          # [B, d_in]
        cp_layer: torch.Tensor,   # [B, nz]
        ctx_s: torch.Tensor,      # [B, 7]
        tgt_qrad: torch.Tensor,   # [B, nz]
        tgt_s: torch.Tensor,      # [B, 7]
    ) -> None:
        b = int(x.shape[0])
        if b == 0:
            return
        if b > self.ring_cap:
            perm = torch.randperm(b, device=self.device)[: self.ring_cap]
            x, cp_layer, ctx_s, tgt_qrad, tgt_s = x[perm], cp_layer[perm], ctx_s[perm], tgt_qrad[perm], tgt_s[perm]
            b = int(x.shape[0])

        x_f16 = torch.clamp(x, -100.0, 100.0).to(torch.float16)
        n_ring = max(1, (3 * b) // 4)
        idx_ring = (torch.arange(n_ring, device=self.device) + self.write_ptr) % self.ring_cap
        self.x[idx_ring] = x_f16[:n_ring]
        self.cp_layer[idx_ring] = cp_layer[:n_ring]
        self.ctx_s[idx_ring] = ctx_s[:n_ring]
        self.tgt_qrad[idx_ring] = tgt_qrad[:n_ring]
        self.tgt_s[idx_ring] = tgt_s[:n_ring]
        self.write_ptr = (self.write_ptr + n_ring) % self.ring_cap
        self.ring_size = min(self.ring_cap, self.ring_size + n_ring)

        n_res = b - n_ring
        if n_res > 0 and self.res_cap > 0:
            if self.res_size < self.res_cap:
                take = min(n_res, self.res_cap - self.res_size)
                idx_r = torch.arange(self.res_size, self.res_size + take, device=self.device) + self.ring_cap
                self.x[idx_r] = x_f16[n_ring : n_ring + take]
                self.cp_layer[idx_r] = cp_layer[n_ring : n_ring + take]
                self.ctx_s[idx_r] = ctx_s[n_ring : n_ring + take]
                self.tgt_qrad[idx_r] = tgt_qrad[n_ring : n_ring + take]
                self.tgt_s[idx_r] = tgt_s[n_ring : n_ring + take]
                self.res_size += take
            else:
                idx_r = torch.randint(self.ring_cap, self.capacity, (n_res,), device=self.device)
                self.x[idx_r] = x_f16[n_ring:]
                self.cp_layer[idx_r] = cp_layer[n_ring:]
                self.ctx_s[idx_r] = ctx_s[n_ring:]
                self.tgt_qrad[idx_r] = tgt_qrad[n_ring:]
                self.tgt_s[idx_r] = tgt_s[n_ring:]

        self.total_seen += b

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(int(batch_size), self.size)
        if self.res_size > 0 and self.ring_size > 0:
            n_ring_s = n // 2
            n_res_s = n - n_ring_s
            i1 = torch.randint(0, self.ring_size, (n_ring_s,), device=self.device)
            i2 = torch.randint(self.ring_cap, self.ring_cap + self.res_size, (n_res_s,), device=self.device)
            idx = torch.cat([i1, i2], dim=0)
        elif self.ring_size > 0:
            idx = torch.randint(0, self.ring_size, (n,), device=self.device)
        else:
            idx = torch.randint(self.ring_cap, self.ring_cap + self.res_size, (n,), device=self.device)
        return (
            self.x[idx].to(torch.float32),
            self.cp_layer[idx],
            self.ctx_s[idx],
            self.tgt_qrad[idx],
            self.tgt_s[idx],
        )


@dataclass
class RadSelfDistillState:
    """Prognostic state for online per-subcolumn ICA radiation self-distillation."""

    net: RadSelfDistillNet
    optimizer: torch.optim.Optimizer
    replay: GPURadReplayBuffer
    stride: int
    warmup_steps: int
    dwell_steps: int
    n_cells: int
    cohort_id: torch.Tensor             # [n_cells] int64 spatially stratified cohort in 0..stride-1
    last_uncertainty: torch.Tensor      # [n_cells] float32 epistemic spread per host cell
    steps_seen: int = 0
    grad_updates: int = 0
    ema_loss: float = 1.0
    ema_val_rmse_olr: float = 5.0
    ema_val_rmse_asr: float = 5.0
    teacher_frac_last: float = 1.0


def init_rad_self_distill_state(
    n_cells: int,
    nz: int,
    device: torch.device,
    *,
    stride: int = 13,
    warmup_steps: int = 12,
    dwell_steps: int = 6,
    hidden_dim: int = 128,
    replay_size: int = 65536,
    lr: float = 2.0e-3,
) -> RadSelfDistillState:
    """Initialize online ICA radiation student network, optimizer, and GPU replay buffer."""
    st = max(int(stride), 2)
    net = RadSelfDistillNet(nz=nz, hidden_dim=hidden_dim).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1.0e-4)
    replay = GPURadReplayBuffer(capacity=replay_size, nz=nz, device=device)
    cell_ids = torch.arange(n_cells, device=device, dtype=torch.int64)
    cohort_id = ((cell_ids * 2654435761) % st).to(torch.int64)
    return RadSelfDistillState(
        net=net,
        optimizer=opt,
        replay=replay,
        stride=st,
        warmup_steps=max(int(warmup_steps), 1),
        dwell_steps=max(int(dwell_steps), 1),
        n_cells=int(n_cells),
        cohort_id=cohort_id,
        last_uncertainty=torch.zeros(n_cells, device=device, dtype=torch.float32),
    )


def select_rad_teacher_indices(
    state: RadSelfDistillState,
    *,
    crm_state: CRMSelfDistillState | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select host-cell `(teacher_idx, student_idx)` for ICA radiation, synced with CRM when active."""
    if state.steps_seen < state.warmup_steps:
        return None
    if crm_state is not None and crm_state.stride == state.stride and crm_state.steps_seen >= crm_state.warmup_steps:
        # Synchronize radiation teachers with active CRM teacher cohort + combined epistemic OOD
        nc = state.n_cells
        n_teach = max(1, (nc + state.stride - 1) // state.stride)
        if n_teach >= nc:
            return None
        active_cohort = ((crm_state.steps_seen - crm_state.warmup_steps) // crm_state.dwell_steps) % crm_state.stride
        rot_idx = torch.nonzero(crm_state.cohort_id == active_cohort, as_tuple=False).squeeze(-1)
        n_ood = max(1, n_teach // 6) if n_teach >= 12 else 0
        n_rot_keep = min(int(rot_idx.numel()), max(n_teach - n_ood, 1))
        rot_sel = rot_idx[:n_rot_keep]
        n_rem = n_teach - n_rot_keep
        if n_rem > 0:
            unc = state.last_uncertainty + 0.5 * crm_state.last_uncertainty
            unc[rot_sel] = -1.0e9
            _, ood_sel = torch.topk(unc, k=n_rem, largest=True, sorted=False)
            teacher_idx = torch.cat([rot_sel, ood_sel], dim=0)
        else:
            teacher_idx = rot_sel
        teach_mask = torch.zeros(nc, device=teacher_idx.device, dtype=torch.bool)
        teach_mask[teacher_idx] = True
        student_idx = torch.nonzero(~teach_mask, as_tuple=False).squeeze(-1)
        return teacher_idx, student_idx

    nc = state.n_cells
    n_teach = max(1, (nc + state.stride - 1) // state.stride)
    if n_teach >= nc:
        return None
    active_cohort = ((state.steps_seen - state.warmup_steps) // state.dwell_steps) % state.stride
    rot_idx = torch.nonzero(state.cohort_id == active_cohort, as_tuple=False).squeeze(-1)
    n_ood = max(1, n_teach // 6) if n_teach >= 12 else 0
    n_rot_keep = min(int(rot_idx.numel()), max(n_teach - n_ood, 1))
    rot_sel = rot_idx[:n_rot_keep]
    n_rem = n_teach - n_rot_keep
    if n_rem > 0:
        unc = state.last_uncertainty.clone()
        unc[rot_sel] = -1.0e9
        _, ood_sel = torch.topk(unc, k=n_rem, largest=True, sorted=False)
        teacher_idx = torch.cat([rot_sel, ood_sel], dim=0)
    else:
        teacher_idx = rot_sel
    teach_mask = torch.zeros(nc, device=teacher_idx.device, dtype=torch.bool)
    teach_mask[teacher_idx] = True
    student_idx = torch.nonzero(~teach_mask, as_tuple=False).squeeze(-1)
    return teacher_idx, student_idx


def _compute_rad_distill_loss(
    pred_a: dict[str, torch.Tensor],
    pred_b: dict[str, torch.Tensor],
    tgt_qrad: torch.Tensor,   # [nz, B] [K s^-1]
    tgt_s: torch.Tensor,      # [B, 7] [W m^-2]
    cp_layer: torch.Tensor,   # [nz, B]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-scale heating-rate + TOA/surface flux loss across both epistemic heads."""
    qr_t_kd = tgt_qrad * C.SECONDS_PER_DAY
    cp_w = cp_layer / torch.clamp(cp_layer.sum(dim=0, keepdim=True), min=1.0)
    s_scale = 0.01  # 100 W m^-2 -> 1.0

    def _head_loss(pr: dict[str, torch.Tensor]) -> torch.Tensor:
        qr_p_kd = pr["q_rad"] * C.SECONDS_PER_DAY
        l_qr_unif = F.smooth_l1_loss(qr_p_kd, qr_t_kd, beta=0.5)
        l_qr_mass = (cp_w * F.smooth_l1_loss(qr_p_kd, qr_t_kd, beta=0.25, reduction="none")).sum(dim=0).mean()
        l_s1d = F.smooth_l1_loss(pr["sw1_dir"] * s_scale, tgt_s[:, 0] * s_scale, beta=0.05)
        l_s1f = F.smooth_l1_loss(pr["sw1_dif"] * s_scale, tgt_s[:, 1] * s_scale, beta=0.05)
        l_s2d = F.smooth_l1_loss(pr["sw2_dir"] * s_scale, tgt_s[:, 2] * s_scale, beta=0.05)
        l_s2f = F.smooth_l1_loss(pr["sw2_dif"] * s_scale, tgt_s[:, 3] * s_scale, beta=0.05)
        l_lwd = F.smooth_l1_loss(pr["lw_dn"] * s_scale, tgt_s[:, 4] * s_scale, beta=0.05)
        l_olr = F.smooth_l1_loss(pr["olr"] * s_scale, tgt_s[:, 5] * s_scale, beta=0.05)
        l_asr = F.smooth_l1_loss(pr["asr"] * s_scale, tgt_s[:, 6] * s_scale, beta=0.05)
        return (
            1.5 * l_qr_unif + 2.5 * l_qr_mass
            + 2.0 * (l_s1d + l_s1f + l_s2d + l_s2f)
            + 3.0 * (l_lwd + l_olr + l_asr)
        )

    loss = 0.5 * (_head_loss(pred_a) + _head_loss(pred_b))
    with torch.no_grad():
        olr_m = 0.5 * (pred_a["olr"] + pred_b["olr"])
        asr_m = 0.5 * (pred_a["asr"] + pred_b["asr"])
        rmse_olr = torch.sqrt(F.mse_loss(olr_m, tgt_s[:, 5]))
        rmse_asr = torch.sqrt(F.mse_loss(asr_m, tgt_s[:, 6]))
    return loss, rmse_olr, rmse_asr


def pack_rad_teacher_context_and_targets(
    cp_layer: torch.Tensor,    # [nz, B]
    s_inc: torch.Tensor,       # [B]
    alb_vd: torch.Tensor,      # [B]
    alb_vf: torch.Tensor,      # [B]
    alb_nd: torch.Tensor,      # [B]
    alb_nf: torch.Tensor,      # [B]
    emiss_sfc: torch.Tensor,   # [B]
    t_sfc: torch.Tensor,       # [B]
    q_rad: torch.Tensor,       # [nz, B]
    sw1_dir: torch.Tensor,     # [B]
    sw1_dif: torch.Tensor,     # [B]
    sw2_dir: torch.Tensor,     # [B]
    sw2_dif: torch.Tensor,     # [B]
    lw_dn: torch.Tensor,       # [B]
    olr: torch.Tensor,         # [B]
    asr: torch.Tensor,         # [B]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack teacher ICA radiation closure context and ground-truth targets for GPU replay."""
    cp_b = cp_layer.transpose(0, 1).contiguous()
    ctx_s = torch.stack([s_inc, alb_vd, alb_vf, alb_nd, alb_nf, emiss_sfc, t_sfc], dim=1).contiguous()
    tgt_qr = q_rad.transpose(0, 1).contiguous()
    tgt_s = torch.stack([sw1_dir, sw1_dif, sw2_dir, sw2_dif, lw_dn, olr, asr], dim=1).contiguous()
    return cp_b, ctx_s, tgt_qr, tgt_s


def train_rad_student_online(
    state: RadSelfDistillState,
    x_fresh: torch.Tensor,        # [B_t, d_in]
    cp_fresh: torch.Tensor,       # [B_t, nz]
    ctx_s_fresh: torch.Tensor,    # [B_t, 7]
    tgt_qr_fresh: torch.Tensor,   # [B_t, nz]
    tgt_s_fresh: torch.Tensor,    # [B_t, 7]
    *,
    n_grad_steps: int = 3,
    batch_size: int = 512,
) -> None:
    """Push fresh teacher ICA radiation pairs to GPU replay buffer and run online AdamW updates."""
    state.replay.push(x_fresh, cp_fresh, ctx_s_fresh, tgt_qr_fresh, tgt_s_fresh)
    if state.replay.size < 64:
        return

    state.net.train()
    n_steps = max(int(n_grad_steps), 1)
    if state.steps_seen < state.warmup_steps or state.ema_val_rmse_olr > 12.0 or state.ema_val_rmse_asr > 18.0:
        n_steps = max(n_steps, 6)

    b_fresh = int(x_fresh.shape[0])
    half_b = min(batch_size // 2, b_fresh)

    last_loss = None
    last_olr = None
    last_asr = None
    for _ in range(n_steps):
        f_idx = torch.randint(0, b_fresh, (half_b,), device=x_fresh.device)
        xr, cpr, csr, tqr, tsr = state.replay.sample(batch_size - half_b)
        xb = torch.cat([x_fresh[f_idx], xr], dim=0)
        cpb = torch.cat([cp_fresh[f_idx], cpr], dim=0).transpose(0, 1).contiguous()  # [nz, B]
        csb = torch.cat([ctx_s_fresh[f_idx], csr], dim=0)
        tqb = torch.cat([tgt_qr_fresh[f_idx], tqr], dim=0).transpose(0, 1).contiguous()  # [nz, B]
        tsb = torch.cat([tgt_s_fresh[f_idx], tsr], dim=0)

        qa_raw, sa_raw, qb_raw, sb_raw = state.net(xb)
        pred_a = apply_exact_rad_conservation_closure(
            qa_raw, sa_raw, cpb, csb[:, 0], csb[:, 1], csb[:, 2], csb[:, 3], csb[:, 4], csb[:, 5], csb[:, 6]
        )
        pred_b = apply_exact_rad_conservation_closure(
            qb_raw, sb_raw, cpb, csb[:, 0], csb[:, 1], csb[:, 2], csb[:, 3], csb[:, 4], csb[:, 5], csb[:, 6]
        )
        loss, r_olr, r_asr = _compute_rad_distill_loss(pred_a, pred_b, tqb, tsb, cpb)

        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.net.parameters(), max_norm=2.0)
        state.optimizer.step()
        state.grad_updates += 1
        last_loss, last_olr, last_asr = loss, r_olr, r_asr

    if last_loss is not None and (state.steps_seen % 8 == 0 or state.steps_seen == state.warmup_steps - 1):
        lv = float(last_loss.detach().item())
        ro = float(last_olr.detach().item())
        ra = float(last_asr.detach().item())
        alpha = 0.2
        state.ema_loss = (1.0 - alpha) * state.ema_loss + alpha * lv
        state.ema_val_rmse_olr = (1.0 - alpha) * state.ema_val_rmse_olr + alpha * ro
        state.ema_val_rmse_asr = (1.0 - alpha) * state.ema_val_rmse_asr + alpha * ra


@torch.no_grad()
def predict_rad_student_columns(
    state: RadSelfDistillState,
    x_stu: torch.Tensor,        # [M_s, d_in]
    cp_layer: torch.Tensor,     # [nz, M_s]
    s_inc: torch.Tensor,        # [M_s]
    alb_vd: torch.Tensor,       # [M_s]
    alb_vf: torch.Tensor,       # [M_s]
    alb_nd: torch.Tensor,       # [M_s]
    alb_nf: torch.Tensor,       # [M_s]
    emiss_sfc: torch.Tensor,    # [M_s]
    t_sfc: torch.Tensor,        # [M_s]
    chunk_size: int = 32768,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run batched FP16/BF16 radiation student inference with exact per-subcolumn energy closure."""
    state.net.eval()
    m_s = int(x_stu.shape[0])
    nz = int(cp_layer.shape[0])
    dev = x_stu.device

    qa_all = torch.empty((m_s, nz), device=dev, dtype=torch.float32)
    sa_all = torch.empty((m_s, _N_RAD_SCAL_OUT), device=dev, dtype=torch.float32)
    qb_all = torch.empty((m_s, nz), device=dev, dtype=torch.float32)
    sb_all = torch.empty((m_s, _N_RAD_SCAL_OUT), device=dev, dtype=torch.float32)

    use_cuda_amp = (dev.type == "cuda")
    use_mps_amp = (dev.type == "mps")
    for i0 in range(0, m_s, chunk_size):
        i1 = min(i0 + chunk_size, m_s)
        if use_cuda_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                qa, sa, qb, sb = state.net(x_stu[i0:i1])
        elif use_mps_amp:
            with torch.amp.autocast("mps", dtype=torch.float16):
                qa, sa, qb, sb = state.net(x_stu[i0:i1])
        else:
            qa, sa, qb, sb = state.net(x_stu[i0:i1])
        qa_all[i0:i1] = qa.float()
        sa_all[i0:i1] = sa.float()
        qb_all[i0:i1] = qb.float()
        sb_all[i0:i1] = sb.float()

    q_mean = 0.5 * (qa_all + qb_all)
    s_mean = 0.5 * (sa_all + sb_all)
    out = apply_exact_rad_conservation_closure(
        q_mean, s_mean, cp_layer, s_inc, alb_vd, alb_vf, alb_nd, alb_nf, emiss_sfc, t_sfc
    )
    unc = (qa_all - qb_all).abs().mean(dim=1) + (sa_all - sb_all).abs().mean(dim=1)
    return out, unc
