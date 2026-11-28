"""Prognostic Stable Water Isotopes (H2^18O and HDO) for Paleoclimate Ice-Core Comparison.

Tracks VSMOW-normalized heavy-water vapor mixing ratios `q_v_18o` and `q_v_hdo` `[Nz, N_cells]`
advected conservatively by AOBA's fused SCVT FCT solver alongside atmospheric moisture `q_n`:
    R_i = q_v_i / q_v,   delta_i [permil] = (R_i - 1) * 1000   (relative to VSMOW = 1.0)

Physical Parameterizations & References
---------------------------------------
1. **Liquid-Vapor Equilibrium Fractionation** (Majoube, 1971, *J. Chim. Phys.*, 68, 1423-1436):
   - ln(alpha_l^18) = 1137/T^2 - 0.4156/T - 0.0020667
   - ln(alpha_l^D)  = 24844/T^2 - 76.248/T + 0.052612
2. **Ice-Vapor Equilibrium & Supersaturation Kinetic Fractionation**
   (Majoube, 1971; Merlivat & Nief, 1967, *Tellus*, 19, 122; Jouzel & Merlivat, 1984, *JGR*, 89):
   - ln(alpha_i^18) = 11.839/T - 0.028224
   - ln(alpha_i^D)  = 16288/T^2 - 0.0934
   - Kinetic supersaturation damping `S_i = 1.0 - 0.003 * (T - 273.15)` below freezing reproduces
     Antarctic/Greenland d-excess (`d = dD - 8 * d18O`).
3. **Craig & Gordon (1965) / Merlivat & Jouzel (1979) Marine Evaporation Fractionation**:
   - Ocean surface isotopic ratio `R_ocn = 1.0` (0 permil VSMOW) with kinetic diffusivity
     factors (`k_18 = 0.006`, `k_D = 0.0053`), producing realistic marine vapor `~ -11 permil`.
4. **Exact Rayleigh Condensation Distillation & Snow/Ice-Core Archiving**:
   - Uses exact exponential/log Rayleigh fractionation `R_new = R_old * (q_new / q_old)^(alpha-1)`
     (zero Forward Euler, unconditionally positivity- and ratio-bounded) and archives snowfall
     `d18o_snow_permil` & `dd_snow_permil` for direct proxy comparison with ice cores.

AOBA Formulation Reference: Sec. 10.7, Eqs. (10.9)-(10.11) (Prognostic 3D H2^18O/HDO, Craig-Gordon & Rayleigh).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from aoba import constants as C
from aoba.runtime import maybe_cuda_compile
from aoba.surface import SURF_OCEAN

# Kinetic evaporation factors over open ocean (Merlivat & Jouzel, 1979, JGR, 84, 5029)
K_KIN_18O_SMOOTH: float = 0.0060   # Smooth regime (U_10 < 7 m s^-1)
K_KIN_18O: float = 0.0060          # Backward-compatible alias
K_KIN_HDO: float = 0.00528         # 0.88 * K_KIN_18O_SMOOTH (Merlivat, 1978)
U10_ROUGH_SWITCH_MS: float = 7.0   # Aerodynamic smooth-to-rough ocean transition [m s^-1]
K_KIN_18O_SOIL: float = 0.0120     # Bare-soil diffusive boundary-layer kinetic factor (Mathieu & Bariac, 1996)
K_KIN_D_OVER_18: float = 0.88      # Merlivat (1978) ratio k_D / k_18
# Ratio diffusivity ratios D_light / D_heavy for snow kinetic supersaturation (Jouzel & Merlivat, 1984)
DIFF_RATIO_18O: float = 1.0285
DIFF_RATIO_HDO: float = 1.0251
# Below-cloud raindrop diffusive kinetic ratios (D_light / D_heavy)^0.58 (Stewart, 1975, JGR, 80, 1133)
KAPPA_DROP_18O: float = 1.0285 ** 0.58   # = 1.01645
KAPPA_DROP_HDO: float = 1.0251 ** 0.58   # = 1.01449
RAIN_REEVAP_PREF_KGM2S: float = 1.5e-4   # Reference rain intensity (~13 mm/day) for sub-cloud re-evaporation (Risi et al., 2008)


@dataclass
class WaterIsotopeState:
    """Prognostic 3D atmospheric H2^18O / HDO vapor fields & precipitation/ice-core archives."""

    q_v_18o: torch.Tensor              # [Nz, N_cells] VSMOW-normalized H2^18O vapor [kg kg^-1]
    q_v_hdo: torch.Tensor              # [Nz, N_cells] VSMOW-normalized HDO vapor [kg kg^-1]
    d18o_precip_permil: torch.Tensor   # [N_cells] instantaneous precipitation d18O [permil VSMOW]
    dd_precip_permil: torch.Tensor     # [N_cells] instantaneous precipitation dD [permil VSMOW]
    dexcess_permil: torch.Tensor       # [N_cells] precipitation deuterium excess d = dD - 8*d18O [permil]
    d18o_snow_permil: torch.Tensor     # [N_cells] accumulated snowpack/ice-core d18O archive [permil]
    dd_snow_permil: torch.Tensor       # [N_cells] accumulated snowpack/ice-core dD archive [permil]
    snow_iso_weight: torch.Tensor      # [N_cells] cumulative snowfall mass weight [kg m^-2]


# AOBA Formulation Eqs. (10.9)-(10.10)
def equilibrium_fractionation_factors(
    temp_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(alpha_18_eff, alpha_d_eff, alpha_liq_18, alpha_liq_d)` at temperature `temp_k` [K].

    Blends liquid (Majoube, 1971) and ice (Merlivat & Nief, 1967 + Jouzel & Merlivat, 1984 kinetic)
    across the mixed-phase window `[253.15, 273.15] K`.
    """
    t = torch.clamp(temp_k, min=190.0, max=330.0)
    inv_t = 1.0 / t
    inv_t2 = inv_t * inv_t

    a_l18 = torch.exp(1137.0 * inv_t2 - 0.4156 * inv_t - 0.0020667)
    a_ld = torch.exp(24844.0 * inv_t2 - 76.248 * inv_t + 0.052612)

    a_i18_eq = torch.exp(11.839 * inv_t - 0.028224)
    a_id_eq = torch.exp(16288.0 * inv_t2 - 0.0934)

    # Jouzel & Merlivat (1984) supersaturation kinetic factor for ice crystal deposition
    s_ice = 1.0 + 0.003 * torch.clamp(C.T_FREEZE - t, min=0.0, max=60.0)
    a_i18 = a_i18_eq * (s_ice / (1.0 + a_i18_eq * (s_ice - 1.0) * DIFF_RATIO_18O))
    a_id = a_id_eq * (s_ice / (1.0 + a_id_eq * (s_ice - 1.0) * DIFF_RATIO_HDO))

    f_liq = torch.clamp((t - C.T_ICE_ALL) / (C.T_FREEZE - C.T_ICE_ALL), 0.0, 1.0)
    a_18_eff = f_liq * a_l18 + (1.0 - f_liq) * a_i18
    a_d_eff = f_liq * a_ld + (1.0 - f_liq) * a_id
    return a_18_eff, a_d_eff, a_l18, a_ld


def init_water_isotope_state(q_v: torch.Tensor) -> WaterIsotopeState:
    """Initialize 3D H2^18O / HDO vapor near marine equilibrium (`-12 permil` d18O, `-86 permil` dD)."""
    qv_pos = torch.clamp(q_v, min=0.0)
    nc = q_v.shape[1]
    dev = q_v.device
    dtype = q_v.dtype
    z = torch.zeros((nc,), device=dev, dtype=dtype)
    return WaterIsotopeState(
        q_v_18o=qv_pos * 0.988,   # -12 permil VSMOW
        q_v_hdo=qv_pos * 0.914,   # -86 permil VSMOW (d-excess = +10 permil)
        d18o_precip_permil=torch.full((nc,), -5.0, device=dev, dtype=dtype),
        dd_precip_permil=torch.full((nc,), -30.0, device=dev, dtype=dtype),
        dexcess_permil=torch.full((nc,), 10.0, device=dev, dtype=dtype),
        d18o_snow_permil=torch.full((nc,), -20.0, device=dev, dtype=dtype),
        dd_snow_permil=torch.full((nc,), -150.0, device=dev, dtype=dtype),
        snow_iso_weight=z.clone(),
    )


def _step_water_isotopes_core(
    q_v_18o: torch.Tensor,
    q_v_hdo: torch.Tensor,
    d18o_precip: torch.Tensor,
    dd_precip: torch.Tensor,
    d18o_snow: torch.Tensor,
    dd_snow: torch.Tensor,
    snow_iso_weight: torch.Tensor,
    q_v: torch.Tensor,
    temp_c: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    surf_type: torch.Tensor,
    t_sfc: torch.Tensor,
    evap_cell: torch.Tensor,
    rain_cell: torch.Tensor,
    snow_cell: torch.Tensor,
    dt: float,
    rh_sfc: torch.Tensor | None,
    u10_cell: torch.Tensor | None,
    lai_cell: torch.Tensor | None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pure-tensor water isotope update (fuses into Triton kernels via `maybe_cuda_compile` on CUDA)."""
    dtype = q_v.dtype
    qv_pos = torch.clamp(q_v, min=1.0e-8)
    dm = torch.clamp(rho_d * dz_c, min=1.0e-4)

    r18 = torch.clamp(q_v_18o / qv_pos, min=0.90, max=1.03)
    rd = torch.clamp(q_v_hdo / qv_pos, min=0.35, max=1.15)

    a18_eff, ad_eff, a18_l, ad_l = equilibrium_fractionation_factors(temp_c)

    # 1. Surface evaporation fractionation in lowest layer k = 0 (Craig & Gordon, 1965; Merlivat & Jouzel, 1979)
    is_ocn = (surf_type == SURF_OCEAN).to(dtype)
    _, _, a18_sfc, ad_sfc = equilibrium_fractionation_factors(t_sfc)
    h_rel = torch.clamp(rh_sfc, 0.25, 0.94) if rh_sfc is not None else torch.full_like(t_sfc, 0.78)

    if u10_cell is not None:
        u10_pos = torch.clamp(u10_cell, min=0.0, max=50.0)
        k18_rough = torch.clamp(0.000285 * u10_pos + 0.00082, min=0.0028, max=0.012)
        k18_ocn = torch.where(u10_pos < U10_ROUGH_SWITCH_MS, torch.full_like(u10_pos, K_KIN_18O_SMOOTH), k18_rough)
    else:
        k18_ocn = torch.full_like(t_sfc, K_KIN_18O_SMOOTH)
    kd_ocn = K_KIN_D_OVER_18 * k18_ocn

    one_m_h = torch.clamp(1.0 - h_rel, min=0.06)
    r18_evap_ocn = torch.clamp(((1.0 / a18_sfc - h_rel * r18[0]) / one_m_h) * (1.0 - k18_ocn), 0.94, 1.01)
    rd_evap_ocn = torch.clamp(((1.0 / ad_sfc - h_rel * rd[0]) / one_m_h) * (1.0 - kd_ocn), 0.72, 1.05)

    r18_soil = torch.clamp(1.0 + d18o_precip * 1.0e-3, 0.92, 1.02)
    rd_soil = torch.clamp(1.0 + dd_precip * 1.0e-3, 0.45, 1.08)
    if lai_cell is not None:
        f_bare = torch.exp(-0.5 * torch.clamp(lai_cell, min=0.0))
        r18_bare = torch.clamp(((r18_soil / a18_sfc - h_rel * r18[0]) / one_m_h) * (1.0 - K_KIN_18O_SOIL), 0.90, 1.02)
        rd_bare = torch.clamp(
            ((rd_soil / ad_sfc - h_rel * rd[0]) / one_m_h) * (1.0 - K_KIN_D_OVER_18 * K_KIN_18O_SOIL), 0.40, 1.08
        )
        r18_land = (1.0 - f_bare) * r18_soil + f_bare * r18_bare
        rd_land = (1.0 - f_bare) * rd_soil + f_bare * rd_bare
    else:
        r18_land = r18_soil
        rd_land = rd_soil

    r18_evap = is_ocn * r18_evap_ocn + (1.0 - is_ocn) * r18_land
    rd_evap = is_ocn * rd_evap_ocn + (1.0 - is_ocn) * rd_land

    dq_ev = (torch.clamp(evap_cell, min=0.0) * float(dt)) / dm[0]
    f_ev = dq_ev / (qv_pos[0] + dq_ev)
    r18_0 = r18[0] + (r18_evap - r18[0]) * f_ev
    rd_0 = rd[0] + (rd_evap - rd[0]) * f_ev
    r18 = torch.cat([r18_0.unsqueeze(0), r18[1:]], dim=0)
    rd = torch.cat([rd_0.unsqueeze(0), rd[1:]], dim=0)

    # 2. Exact Rayleigh condensation distillation across precipitating columns
    rain_pos = torch.clamp(rain_cell, min=0.0)
    snow_pos = torch.clamp(snow_cell, min=0.0)
    precip_tot = rain_pos + snow_pos
    cwv = torch.clamp((qv_pos * dm).sum(dim=0), min=1.0e-3)
    f_rem = torch.exp(-torch.clamp((precip_tot * float(dt)) / cwv, min=0.0, max=2.0))

    w_cond = (qv_pos * dm) / cwv.unsqueeze(0)
    r18_cond_col = (w_cond * r18 * a18_eff).sum(dim=0)
    rd_cond_col = (w_cond * rd * ad_eff).sum(dim=0)

    ln_f = torch.log(torch.clamp(f_rem, min=0.05, max=1.0)).unsqueeze(0)
    r18 = torch.clamp(r18 * torch.exp((a18_eff - 1.0) * ln_f), min=0.90, max=1.03)
    rd = torch.clamp(rd * torch.exp((ad_eff - 1.0) * ln_f), min=0.35, max=1.15)

    # 2b. Below-cloud raindrop partial re-evaporation & diffusive exchange (Stewart, 1975; Risi et al., 2008)
    h_sub = torch.clamp(rh_sfc, 0.25, 0.98) if rh_sfc is not None else torch.full_like(t_sfc, 0.78)
    one_m_hs = torch.clamp(1.0 - h_sub, min=0.02)
    f_rain_share = torch.where(precip_tot > 1.0e-12, rain_pos / torch.clamp(precip_tot, min=1.0e-12), torch.zeros_like(rain_pos))
    f_reevap = f_rain_share * torch.clamp(
        0.35 * one_m_hs / (1.0 + rain_pos / RAIN_REEVAP_PREF_KGM2S), min=0.0, max=0.45
    )
    f_drop = torch.clamp(1.0 - f_reevap, min=0.55, max=1.0)
    ln_fdrop = torch.log(f_drop)

    ak18 = a18_l[0] * KAPPA_DROP_18O
    akd = ad_l[0] * KAPPA_DROP_HDO
    den18 = torch.clamp(1.0 - ak18 * one_m_hs, min=0.10)
    dend = torch.clamp(1.0 - akd * one_m_hs, min=0.10)
    gamma18 = (a18_l[0] * h_sub) / den18
    gammad = (ad_l[0] * h_sub) / dend
    beta18 = den18 / (ak18 * one_m_hs)
    betad = dend / (akd * one_m_hs)

    pow18 = torch.exp(torch.clamp(beta18 * ln_fdrop, min=-10.0, max=0.0))
    powd = torch.exp(torch.clamp(betad * ln_fdrop, min=-10.0, max=0.0))
    r18_drop = torch.clamp(gamma18 * r18[0] + (r18_cond_col - gamma18 * r18[0]) * pow18, min=0.90, max=1.03)
    rd_drop = torch.clamp(gammad * rd[0] + (rd_cond_col - gammad * rd[0]) * powd, min=0.35, max=1.15)

    r18_re = torch.where(
        f_reevap > 1.0e-5,
        torch.clamp((r18_cond_col - f_drop * r18_drop) / torch.clamp(f_reevap, min=1.0e-5), min=0.90, max=1.03),
        r18[0],
    )
    rd_re = torch.where(
        f_reevap > 1.0e-5,
        torch.clamp((rd_cond_col - f_drop * rd_drop) / torch.clamp(f_reevap, min=1.0e-5), min=0.35, max=1.15),
        rd[0],
    )
    dq_re = (f_reevap * rain_pos * float(dt)) / dm[0]
    f_mix_re = dq_re / (qv_pos[0] + dq_re)
    r18_0_post = torch.clamp(r18[0] + (r18_re - r18[0]) * f_mix_re, min=0.90, max=1.03)
    rd_0_post = torch.clamp(rd[0] + (rd_re - rd[0]) * f_mix_re, min=0.35, max=1.15)
    r18 = torch.cat([r18_0_post.unsqueeze(0), r18[1:]], dim=0)
    rd = torch.cat([rd_0_post.unsqueeze(0), rd[1:]], dim=0)

    r18_sfc_p = f_rain_share * r18_drop + (1.0 - f_rain_share) * r18_cond_col
    rd_sfc_p = f_rain_share * rd_drop + (1.0 - f_rain_share) * rd_cond_col

    has_p = precip_tot > 1.0e-8
    d18_now = (r18_sfc_p - 1.0) * 1000.0
    dd_now = (rd_sfc_p - 1.0) * 1000.0
    d18o_precip_new = torch.where(has_p, d18_now, d18o_precip)
    dd_precip_new = torch.where(has_p, dd_now, dd_precip)
    dexcess_new = dd_precip_new - 8.0 * d18o_precip_new

    d18_sn_now = (r18_cond_col - 1.0) * 1000.0
    dd_sn_now = (rd_cond_col - 1.0) * 1000.0
    dsnow = snow_pos * float(dt)
    w_old = torch.clamp(snow_iso_weight, min=0.0, max=500.0)
    w_new = w_old + dsnow
    alpha_sn = torch.where(w_new > 1.0e-9, dsnow / torch.clamp(w_new, min=1.0e-9), torch.zeros_like(dsnow))
    d18o_snow_new = d18o_snow + (d18_sn_now - d18o_snow) * alpha_sn
    dd_snow_new = dd_snow + (dd_sn_now - dd_snow) * alpha_sn
    snow_w_new = torch.clamp(w_new, max=500.0)

    return (
        r18 * qv_pos,
        rd * qv_pos,
        d18o_precip_new,
        dd_precip_new,
        dexcess_new,
        d18o_snow_new,
        dd_snow_new,
        snow_w_new,
    )


# AOBA Formulation Eqs. (10.10)-(10.12)
def step_water_isotopes(
    iso: WaterIsotopeState,
    q_v: torch.Tensor,            # [Nz, N_cells] updated atmospheric vapor [kg kg^-1]
    temp_c: torch.Tensor,         # [Nz, N_cells] temperature [K]
    rho_d: torch.Tensor,          # [Nz, N_cells] dry air density [kg m^-3]
    dz_c: torch.Tensor,           # [Nz, N_cells] layer thickness [m]
    surf_type: torch.Tensor,      # [N_cells] surface type (SURF_OCEAN = 0)
    t_sfc: torch.Tensor,          # [N_cells] surface temperature [K]
    evap_cell: torch.Tensor,      # [N_cells] surface evaporation [kg m^-2 s^-1]
    rain_cell: torch.Tensor,      # [N_cells] surface rain rate [kg m^-2 s^-1]
    snow_cell: torch.Tensor,      # [N_cells] surface snow rate [kg m^-2 s^-1]
    dt: float,
    *,
    rh_sfc: torch.Tensor | None = None,
    u10_cell: torch.Tensor | None = None,
    lai_cell: torch.Tensor | None = None,
) -> None:
    """Advance isotopic fractionation (Craig-Gordon wind-regime + bare-soil + Rayleigh + Stewart raindrop re-evap).

    Zero Forward Euler, strictly ratio-bounded (`R_18 in [0.90, 1.03]`, `R_D in [0.35, 1.15]`),
    zero Python vertical loops or CPU-GPU syncs; uses fused Metal/CUDA column kernel when available.
    """
    from aoba.kernels import can_use_crm_metal, dispatch_water_isotopes_metal

    if can_use_crm_metal(q_v) and q_v.numel() > 0:
        (
            iso.q_v_18o,
            iso.q_v_hdo,
            iso.d18o_precip_permil,
            iso.dd_precip_permil,
            iso.dexcess_permil,
            iso.d18o_snow_permil,
            iso.dd_snow_permil,
            iso.snow_iso_weight,
        ) = dispatch_water_isotopes_metal(
            q_v_18o=iso.q_v_18o, q_v_hdo=iso.q_v_hdo,
            d18o_precip=iso.d18o_precip_permil, dd_precip=iso.dd_precip_permil,
            d18o_snow=iso.d18o_snow_permil, dd_snow=iso.dd_snow_permil,
            snow_iso_weight=iso.snow_iso_weight,
            q_v=q_v, temp_c=temp_c, rho_d=rho_d, dz_c=dz_c,
            surf_type=surf_type, t_sfc=t_sfc, evap_cell=evap_cell,
            rain_cell=rain_cell, snow_cell=snow_cell, dt=float(dt),
            rh_sfc=rh_sfc, u10_cell=u10_cell, lai_cell=lai_cell,
            surf_ocean_id=int(SURF_OCEAN),
            kappa_drop_18o=float(KAPPA_DROP_18O),
            kappa_drop_hdo=float(KAPPA_DROP_HDO),
        )
        return
    fn = maybe_cuda_compile(_step_water_isotopes_core, device=q_v.device)
    (
        iso.q_v_18o,
        iso.q_v_hdo,
        iso.d18o_precip_permil,
        iso.dd_precip_permil,
        iso.dexcess_permil,
        iso.d18o_snow_permil,
        iso.dd_snow_permil,
        iso.snow_iso_weight,
    ) = fn(
        iso.q_v_18o, iso.q_v_hdo, iso.d18o_precip_permil, iso.dd_precip_permil,
        iso.d18o_snow_permil, iso.dd_snow_permil, iso.snow_iso_weight,
        q_v, temp_c, rho_d, dz_c, surf_type, t_sfc, evap_cell, rain_cell, snow_cell,
        float(dt), rh_sfc, u10_cell, lai_cell,
    )
