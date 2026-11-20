"""PFT-free Two-Leaf C3/C4 Eco-Evolutionary Optimality (EEO) biosphere module.

Implements a fully parameter-table-free terrestrial biosphere combining:
1. Canopy Radiative Transfer & Closed-Form Two-Leaf Capacity Allocation:
   - McCree (1972) beam/diffuse PAR photon conversion & de Pury & Farquhar (1997,
     Plant Cell Environ., 20, 537-557) sunlit/shaded PPFD absorption.
   - Optimal nitrogen/enzyme vertical allocation proportional to EMA-smoothed midday
     downward irradiance absorbed profile `tilde_I_l(L) / tilde_I_c` integrated over
     instantaneous `f_sun(L) = exp(-k_db * L)` (Field, 1983; Farquhar, 1989;
     de Pury & Farquhar, 1997; Hashimoto, 2026 ATAMI Sec. 2.2.5 Eq. 2.24 & App. A.1.4
     Eqs. A1.23-A1.25).
   - Dai, Dickinson, & Wang (2004, J. Climate, 17, 2281-2299, Appendix A Eqs. A1-A6):
     Singularity-free two-stream canopy VIS & NIR albedos and ground transmission.
2. Non-Rectangular-Hyperbola EEO Acclimation of C3 & C4 Photosynthesis & Leaf Respiration:
   - Smith et al. (2019, Ecology Letters, 22, 506-517, Eqs. 1-20):
     Non-rectangular hyperbola `theta * J^2 - (phi * I + J_max) * J + phi * I * J_max = 0`
     (`theta = 0.85`, marginal cost `c = 0.05336251`) yielding closed-form `(omega, omega^*)`
     and optimal C3 `J_max` and `V_cmax` combined with Kattge & Knorr (2007) peaked Arrhenius.
   - Scott & Smith (2022, JAMES, 14, e2021MS002470, Eqs. 1-26 & Table 1):
     Analogous non-rectangular-hyperbola least-cost + three-enzyme coordination for C4
     photosynthesis (`beta_C4 = 166`, `phi_L = 0.20`, `g_bs = 0.003 mol m^-2 s^-1`),
     yielding optimal C4 `J_max`, `V_pmax`, and `V_cmax`.
   - Heskel et al. (2016, PNAS, 113, 3832-3837) & Ren et al. (2024, New Phytol., 241, 578-591;
     2025, JAMES, 17, e2024MS004599, Eqs. 13-14):
     Macromolecular rate theory (MMRT) leaf dark respiration coupled to acclimated `V_cmax`
     (`R_25 = b_resp * V_cmax,opt / f_r(T_night)`, `b_resp = 0.018`).
   - Medlyn et al. (2011, Global Change Biol., 17, 2134-2144) & Prentice et al. (2014,
     Ecol. Lett., 17, 82-91) & Mengoli et al. (2022, JAMES, 14, e2021MS002767):
     Dual-timescale EMA acclimation and fast sub-daily optimal stomatal conductance.
   - Luo et al. (2024, Nat. Commun., 15, 1219) & Lavergne et al. (2026, Commun. Earth Environ.,
     7, 93): Autonomous C4 fractional abundance `F_4 in [0, 1]` from C4 vs C3 optimality advantage.
3. Prognostic EEO Leaf Area Index, Drought-Bridging Root Zone, & Allometric Canopy Height:
   - Cai et al. (2025, Commun. Earth Environ., 6, 19, Eqs. 1-2 & 12) &
     Zhou et al. (2025, Global Change Biol., 31, e70125, Eqs. 6 & 13-20):
     Annual peak `fAPAR_max` (`f_0 = 0.65 * exp(...)`, `z_cost = 13.86 mol m^-2 yr^-1`) and
     Lambert-`W_0` steady-state leaf area `L_s` with EMA phenology lag `alpha = 0.067`.
   - Gao et al. (2014, Geophys. Res. Lett., 41, 7916-7923) & Davis et al. (2017, GMD, 10, 689-708):
     Optimal root-zone water storage capacity `S_R` and supply limit `S_w`.
   - Kempes, West, Crowell, & Girvan (2011, PLoS ONE, 6, e20551, Eqs. 2-5):
     Metabolic vs water-limited maximum canopy height `h_c`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C

# ---------------------------------------------------------------------------
# 1. Universal EEO Constants (Smith et al., 2019; Scott & Smith, 2022; Cai et al., 2025; Zhou et al., 2025)
# ---------------------------------------------------------------------------
THETA_NRH: float = 0.85            # Non-rectangular hyperbola curvature theta [-] (Smith et al., 2019, Eq. 9)
COST_C_SMITH: float = 0.05336251   # Marginal cost of J_max investment c [-] (Smith et al., 2019, Eq. 11)
PHI_PSII_REF_C3: float = 0.257     # Realized PSII quantum yield at reference T [mol mol^-1] (Smith et al., 2019)

BETA_C3: float = 146.0             # Carboxylation/transpiration cost ratio for C3 [-] (Smith et al., 2019, Eq. 2)
BETA_C4: float = 166.0             # Carboxylation/transpiration cost ratio for C4 [-] (Scott & Smith, 2022, Eq. 2)

# C3 Bernacchi et al. (2001) & Kattge & Knorr (2007) parameters (Smith et al., 2019, Eqs. 3 & 20; Scott & Smith Table 1)
GAMMA_STAR25_C3: float = 4.332     # C3 photorespiratory compensation point at 25 degC & 1 atm [Pa]
KC25_C3: float = 41.03             # C3 Rubisco K_c at 25 degC [Pa]
KO25_C3: float = 28210.0           # C3 Rubisco K_o at 25 degC [Pa]
DHA_GAMMA: float = 37830.0         # Activation energy for Gamma^* [J mol^-1]
DHA_KC_C3: float = 79430.0         # Activation energy for C3 K_c [J mol^-1]
DHA_KO_C3: float = 36380.0         # Activation energy for C3 K_o [J mol^-1]

# C4 Boyd et al. (2015) & von Caemmerer (2000) parameters (Scott & Smith, 2022, Table 1)
GAMMA_STAR25_C4: float = 2.60      # C4 bundle-sheath Gamma^* at 25 degC [Pa]
KP25_UMOL_C4: float = 60.5         # C4 PEPC K_p at 25 degC [umol mol^-1]
DHA_KP_C4: float = 27200.0         # Activation energy for C4 K_p [J mol^-1]
KC25_C4: float = 121.0             # C4 Rubisco K_c at 25 degC [Pa]
DHA_KC_C4: float = 64200.0         # Activation energy for C4 K_c [J mol^-1]
KO25_C4: float = 29200.0           # C4 Rubisco K_o at 25 degC [Pa]
DHA_KO_C4: float = 10500.0         # Activation energy for C4 K_o [J mol^-1]
PHI_LEAK_C4: float = 0.20          # Bundle-sheath CO2 leakiness phi_L [-] (Scott & Smith, 2022, Eq. 14)
GBS_C4: float = 0.003              # Bundle-sheath conductance g_bs [mol m^-2 s^-1] (Scott & Smith, 2022, Table 1)

# Peaked Arrhenius enzyme kinetics (Kattge & Knorr, 2007; Smith et al., 2019, Eq. 20)
HA_VCMAX: float = 71513.0          # V_cmax activation energy [J mol^-1]
HA_JMAX: float = 49884.0           # J_max activation energy [J mol^-1]
HA_VPMAX: float = 55000.0          # V_pmax activation energy [J mol^-1] (Massad et al., 2007)
HD_ENZYME: float = 200000.0        # High-temperature deactivation energy [J mol^-1]

# Leaf dark respiration coupling (Heskel et al., 2016; Ren et al., 2024, 2025, Eq. 13)
B_RESP_REN: float = 0.018          #Universal R_leaf(T_night) / V_cmax,opt ratio [-] (Ren et al., 2025, Eq. 13)

# Canopy radiative transfer constants (McCree, 1972; de Pury & Farquhar, 1997; Dai et al., 2004)
LEAF_SCAT_VIS: float = 0.15        # Leaf scattering coefficient in VIS/PAR (sigma = r_f + t_f) [-]
LEAF_SCAT_NIR: float = 0.80        # Leaf scattering coefficient in NIR [-] (Sellers et al., 1986; Dai et al., 2004)
KD_PRIME_PAR: float = 0.719        # Diffuse PAR attenuation coefficient k_d' [-] (de Pury & Farquhar, 1997)
RHO_CD_PAR: float = 0.036          # Diffuse canopy-top PAR reflectance rho_cd [-] (de Pury & Farquhar, 1997)

# Prognostic EEO LAI constants (Cai et al., 2025, Eq. 12; Zhou et al., 2025, Eqs. 6, 10, 15-20)
Z_FOLIAGE_COST: float = 12.227     # Canopy foliage carbon cost z [mol C m^-2 yr^-1] (Zhou et al., 2025, Eq. 10)
K_BEER_LAI: float = 0.50           # Canopy extinction coefficient k [-] (Cai et al., 2025; Zhou et al., 2025)
SIGMA_ZHOU: float = 0.771          # Seasonal LAI-GPP shape factor sigma [-] (Zhou et al., 2025, Eq. 20)
ALPHA_EMA_ACCLIM: float = 0.067    # Daily EMA weight (~15-day e-folding memory) [-] (Mengoli et al., 2022; Zhou et al., 2025)


# ---------------------------------------------------------------------------
# 2. Temperature & Water-Viscosity Functions
# ---------------------------------------------------------------------------
def water_viscosity_rel(temp_k: torch.Tensor) -> torch.Tensor:
    """Viscosity of liquid water relative to 25 degC `eta^* = eta(T) / eta(298.15 K)` [-].

    Reference: Vogel equation adopted in Prentice et al. (2014) & Stocker et al. (2020, Eq. 13):
        eta(T) = exp(-3.719 + 580 / (T - 138)).
    """
    t_safe = torch.clamp(temp_k, min=253.15, max=333.15)
    eta_t = torch.exp(-3.719 + 580.0 / (t_safe - 138.0))
    eta_25 = math.exp(-3.719 + 580.0 / (298.15 - 138.0))
    return eta_t / eta_25


def phi_psii_temperature(temp_k: torch.Tensor) -> torch.Tensor:
    """Temperature-dependent realized quantum yield of PSII `phi_PSII(T)` [mol e^- mol^-1 photon].

    Combines the Bernacchi et al. (2003) quadratic temperature dependence (Scott & Smith, 2022,
    Eq. 9; Stocker et al., 2020, Eq. 18) scaled to Smith et al. (2019) `0.257` at 20 degC:
        phi_raw(T_C) = 0.352 + 0.022 * T_C - 0.00034 * T_C^2.
    """
    tc = torch.clamp(temp_k - C.T_FREEZE, -5.0, 50.0)
    phi_bern = torch.clamp(0.352 + 0.022 * tc - 3.4e-4 * tc * tc, min=0.02)
    phi_bern_20 = 0.352 + 0.022 * 20.0 - 3.4e-4 * 400.0  # = 0.656
    return PHI_PSII_REF_C3 * (phi_bern / phi_bern_20)


def _arrhenius(val25: float, dha: float, temp_k: torch.Tensor) -> torch.Tensor:
    """Standard Arrhenius temperature scaling relative to 298.15 K (Bernacchi et al., 2001)."""
    t_s = torch.clamp(temp_k, min=240.0, max=335.0)
    return val25 * torch.exp(dha * (t_s - 298.15) / (298.15 * C.R_GAS_UNIVERSAL * t_s))


def peaked_arrhenius(
    temp_k: torch.Tensor,
    t_growth_k: torch.Tensor,
    ha: float,
    is_jmax: bool = False,
) -> torch.Tensor:
    """Peaked Arrhenius thermal response relative to 298.15 K with entropy acclimation.

    Reference: Kattge & Knorr (2007, Plant Cell Environ., 30, 1176-1190) &
    Smith et al. (2019, Ecology Letters, 22, Eq. 20):
        Delta_S_V = 668.39 - 1.07 * T_growth_C
        Delta_S_J = 659.70 - 0.75 * T_growth_C.
    """
    t0 = 298.15
    t_s = torch.clamp(temp_k, min=250.0, max=330.0)
    tg_c = torch.clamp(t_growth_k - C.T_FREEZE, 0.0, 40.0)
    ds = (659.70 - 0.75 * tg_c) if is_jmax else (668.39 - 1.07 * tg_c)

    num = 1.0 + torch.exp((t0 * ds - HD_ENZYME) / (C.R_GAS_UNIVERSAL * t0))
    den = 1.0 + torch.exp((t_s * ds - HD_ENZYME) / (C.R_GAS_UNIVERSAL * t_s))
    return torch.exp(ha * (t_s - t0) / (C.R_GAS_UNIVERSAL * t0 * t_s)) * (num / den)


def heskel_respiration_scale(temp_k: torch.Tensor) -> torch.Tensor:
    """Universal MMRT leaf dark respiration temperature response `f_r(T) / f_r(298.15 K)` [-].

    Reference: Heskel et al. (2016, PNAS, 113, 3832-3837, Eq. 3) &
    Ren et al. (2025, JAMES, 17, Eqs. 13-14):
        ln R(T_C) = a + 0.1012 * T_C - 0.0005 * T_C^2,
    normalized to unity at 25 degC (`0.1012 * 25 - 0.0005 * 625 = 2.2175`).
    """
    tc = torch.clamp(temp_k - C.T_FREEZE, -20.0, 55.0)
    return torch.exp((0.1012 * tc - 0.0005 * tc * tc) - 2.2175)


def non_rectangular_hyperbola(
    i_psii: torch.Tensor,
    j_max: torch.Tensor,
    theta: float = THETA_NRH,
) -> torch.Tensor:
    """Smaller root of the Farquhar & Wong (1984) / Smith et al. (2019, Eq. 9) hyperbola:

        theta * J^2 - (i_psii + J_max) * J + i_psii * J_max = 0.
    """
    b = i_psii + j_max
    disc = torch.clamp(b * b - 4.0 * theta * i_psii * j_max, min=0.0)
    return (b - torch.sqrt(disc)) / (2.0 * theta)


def smith_omega(m_fac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form non-rectangular-hyperbola optimality factors `(omega, omega^*)`.

    Reference: Smith et al. (2019, Ecology Letters, 22, Eqs. 14-18) &
    Scott & Smith (2022, JAMES, 14, Eqs. 7-8):
        omega + (1 - 2*theta) = (1 - 2*theta*(4c/m)) * sqrt((1 + omega)^2 - 4*theta*omega)
    ->  omega = -(1 - 2*theta) + sign(1 - 2*theta*(4c/m))
                * sqrt((1 - theta) * (1 / ((4c/m) * (1 - theta * 4c/m)) - 4*theta))
        omega^* = 1 + omega - sqrt((1 + omega)^2 - 4 * theta * omega).
    Using `sign(1 - 2*theta*(4c/m))` ensures exact C^infty continuity across `m = 8*theta*c`
    and `omega -> 0` as `m -> 4*c^+`, vanishing identically for `m <= 4*c`.
    """
    four_c = 4.0 * COST_C_SMITH
    valid = m_fac > four_c
    m_safe = torch.where(valid, m_fac, torch.ones_like(m_fac))
    x = four_c / m_safe
    denom = torch.clamp(x * (1.0 - THETA_NRH * x), min=1.0e-8)
    rad = torch.clamp((1.0 - THETA_NRH) * (1.0 / denom - 4.0 * THETA_NRH), min=0.0)
    branch_sign = torch.where(1.0 - 2.0 * THETA_NRH * x >= 0.0, 1.0, -1.0)
    omega_raw = -(1.0 - 2.0 * THETA_NRH) + branch_sign * torch.sqrt(rad)
    omega = torch.where(valid, torch.clamp(omega_raw, min=0.0), torch.zeros_like(m_fac))
    disc_star = torch.clamp((1.0 + omega) ** 2 - 4.0 * THETA_NRH * omega, min=0.0)
    omega_star = torch.where(valid, torch.clamp(1.0 + omega - torch.sqrt(disc_star), min=0.0), torch.zeros_like(m_fac))
    return omega, omega_star


# ---------------------------------------------------------------------------
# 3. Canopy Radiative Transfer & Closed-Form Two-Leaf Allocation
# ---------------------------------------------------------------------------
@dataclass
class CanopyRadiationProfile:
    """Two-stream sunlit/shaded PPFD absorption profile (de Pury & Farquhar, 1997)."""

    i_b0: torch.Tensor       # Incident beam PPFD [umol m^-2 s^-1]
    i_d0: torch.Tensor       # Incident diffuse PPFD [umol m^-2 s^-1]
    k_db: torch.Tensor       # Direct-beam extinction coefficient 0.5 / cos(theta_z) [-]
    k_b_prime: torch.Tensor  # Scattered-beam attenuation rate k_db * sqrt(1 - sigma) [-]
    rho_cb: torch.Tensor      # Beam canopy-top reflectance [-]
    l_sun: torch.Tensor      # Sunlit leaf area index [m^2 m^-2]
    l_sh: torch.Tensor       # Shaded leaf area index [m^2 m^-2]
    i_c: torch.Tensor        # Whole-canopy absorbed PPFD [umol m^-2 s^-1]
    i_sun: torch.Tensor      # Sunlit-fraction absorbed PPFD [umol m^-2 s^-1]
    i_sh: torch.Tensor       # Shaded-fraction absorbed PPFD [umol m^-2 s^-1]


def canopy_light_absorption_depury(
    vis_beam_wm2: torch.Tensor,
    vis_diff_wm2: torch.Tensor,
    mu0: torch.Tensor,
    lai: torch.Tensor,
) -> CanopyRadiationProfile:
    """Compute whole-canopy, sunlit, and shaded absorbed PPFD [umol photon m^-2 s^-1].

    Implements McCree (1972) & de Pury & Farquhar (1997, Plant Cell Environ., 20, 537-557),
    matching Section A.1.1-A.1.3 (Eqs. A1.1-A1.22) of Hashimoto (2026, ATAMI).
    Note: `vis_beam_wm2` and `vis_diff_wm2` are visible PAR (0.4-0.7 um) fluxes [W m^-2].
    """
    # Eqs. A1.1-A1.2 in umol photon m^-2 s^-1
    i_b0 = 4.57 * torch.clamp(vis_beam_wm2, min=0.0)
    i_d0 = 4.24 * torch.clamp(vis_diff_wm2, min=0.0)

    lc = torch.clamp(lai, min=0.0)
    has_veg = lc >= 0.01

    # Eqs. A1.3-A1.6
    k_db = 0.5 / torch.clamp(mu0, min=0.01)
    sqrt_1ms = math.sqrt(1.0 - LEAF_SCAT_VIS)
    k_b_prime = k_db * sqrt_1ms
    rho_h = (1.0 - sqrt_1ms) / (1.0 + sqrt_1ms)
    rho_cb = 2.0 * rho_h * k_db / (1.0 + k_db)

    # Eqs. A1.11-A1.13: Canopy-integrated beam, diffuse, and total absorbed PPFD
    i_cb = (1.0 - rho_cb) * i_b0 * (1.0 - torch.exp(-torch.clamp(k_b_prime * lc, max=50.0)))
    i_cd = (1.0 - RHO_CD_PAR) * i_d0 * (1.0 - torch.exp(-torch.clamp(KD_PRIME_PAR * lc, max=50.0)))
    i_c = torch.where(has_veg, i_cb + i_cd, torch.zeros_like(lc))

    # Eq. A1.15: Sunlit and shaded leaf area indices
    k_lc = k_db * lc
    l_sun_raw = torch.where(
        k_lc < 1.0e-4,
        lc * (1.0 - 0.5 * k_lc),
        (1.0 - torch.exp(-torch.clamp(k_lc, max=50.0))) / k_db,
    )
    l_sun = torch.where(has_veg, torch.clamp(l_sun_raw, min=0.0), torch.zeros_like(lc))
    l_sh = torch.where(has_veg, torch.clamp(lc - l_sun, min=0.0), torch.zeros_like(lc))

    # Eqs. A1.17, A1.19, A1.20: Sunlit direct-beam, scattered-beam, and diffuse absorption
    i_sun_db = i_b0 * (1.0 - LEAF_SCAT_VIS) * (1.0 - torch.exp(-torch.clamp(k_lc, max=50.0)))
    i_sun_sb = i_b0 * torch.clamp(
        (1.0 - rho_cb)
        * k_b_prime
        * (1.0 - torch.exp(-torch.clamp((k_b_prime + k_db) * lc, max=50.0)))
        / (k_b_prime + k_db)
        - 0.5 * (1.0 - LEAF_SCAT_VIS) * (1.0 - torch.exp(-torch.clamp(2.0 * k_lc, max=50.0))),
        min=0.0,
    )
    i_sun_d = (
        i_d0
        * (1.0 - RHO_CD_PAR)
        * KD_PRIME_PAR
        * (1.0 - torch.exp(-torch.clamp((KD_PRIME_PAR + k_db) * lc, max=50.0)))
        / (KD_PRIME_PAR + k_db)
    )

    i_sun = torch.where(has_veg, torch.minimum(i_sun_db + i_sun_sb + i_sun_d, i_c), torch.zeros_like(lc))
    i_sh = torch.where(has_veg, torch.clamp(i_c - i_sun, min=0.0), torch.zeros_like(lc))

    return CanopyRadiationProfile(
        i_b0=i_b0,
        i_d0=i_d0,
        k_db=k_db,
        k_b_prime=k_b_prime,
        rho_cb=rho_cb,
        l_sun=l_sun,
        l_sh=l_sh,
        i_c=i_c,
        i_sun=i_sun,
        i_sh=i_sh,
    )


def allocate_two_leaf_capacity(
    x_total: torch.Tensor,
    acclim_rad: CanopyRadiationProfile,
    k_db_inst: torch.Tensor,
    l_sun_inst: torch.Tensor,
    lai: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition whole-canopy capacity `x_total` into `(x_sun, x_sh)` in closed form.

    Implements Section 2.2.5 Eq. (2.24) & Appendix A.1.4 Eqs. (A1.23)-(A1.25) of
    Hashimoto (2026, ATAMI), following Field (1983), Farquhar (1989), and
    de Pury & Farquhar (1997):
        X_sun = (X / tilde_I_c) * [
            (1 - tilde_rho_cb) * tilde_k_b' * tilde_I_b0 / (tilde_k_b' + k_db)
                * {1 - exp(-(tilde_k_b' + k_db) * L_c)}
          + (1 - rho_cd) * k_d' * tilde_I_d0 / (k_d' + k_db)
                * {1 - exp(-(k_d' + k_db) * L_c)}
        ]
        X_sh = max(1e-6, X - X_sun),
    where `acclim_rad` was computed by applying `canopy_light_absorption_depury` to the
    EMA-averaged midday downward shortwave irradiances at the current `lai`.
    """
    lc = torch.clamp(lai, min=0.0)
    has_veg = lc >= 0.01
    has_light = acclim_rad.i_c > 1.0e-2  # 1e-2 umol m^-2 s^-1

    term_b = (
        (1.0 - acclim_rad.rho_cb)
        * acclim_rad.k_b_prime
        * acclim_rad.i_b0
        * (1.0 - torch.exp(-torch.clamp((acclim_rad.k_b_prime + k_db_inst) * lc, max=50.0)))
        / (acclim_rad.k_b_prime + k_db_inst)
    )
    term_d = (
        (1.0 - RHO_CD_PAR)
        * KD_PRIME_PAR
        * acclim_rad.i_d0
        * (1.0 - torch.exp(-torch.clamp((KD_PRIME_PAR + k_db_inst) * lc, max=50.0)))
        / (KD_PRIME_PAR + k_db_inst)
    )
    x_sun_opt = x_total * (term_b + term_d) / torch.clamp(acclim_rad.i_c, min=1.0e-2)
    x_sun_area = x_total * (l_sun_inst / torch.clamp(lc, min=1.0e-4))

    x_sun = torch.where(
        has_veg,
        torch.minimum(torch.where(has_light, x_sun_opt, x_sun_area), x_total),
        torch.zeros_like(x_total),
    )
    x_sh = torch.where(
        has_veg,
        torch.clamp(x_total - x_sun, min=1.0e-6),
        torch.zeros_like(x_total),
    )
    return x_sun, x_sh


def canopy_two_stream_albedo_dai(
    lai: torch.Tensor,
    mu0: torch.Tensor,
    alb_ground_vis: torch.Tensor,
    alb_ground_nir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Singularity-free Sellers (1985) / Dai, Dickinson, & Wang (2004, App. A) 2-stream canopy albedos.

    Assumes spherical leaf angle distribution (`chi_L = 0 -> phi_1 = 0.5, phi_2 = 0,
    G(mu) = 0.5, mu_bar = 1.0` via Dai et al., 2004, Eq. A4c), avoiding all angle singularities,
    and smoothly Regularizes `sigma = mu_bar^2 * K^2 - (b^2 - c^2) -> 0` (Dai et al., 2004, Eqs. A5-A6).

    Returns `(alb_dir_vis, alb_dif_vis, alb_dir_nir, alb_dif_nir, trans_vis, trans_nir)`.
    """
    lc = torch.clamp(lai, min=0.0)
    mu_s = torch.clamp(mu0, min=0.02)
    k_opt = 0.5 / mu_s  # Direct-beam optical depth per LAI (G=0.5)
    mu_bar = 1.0        # Exact spherical limit of Eq. (A4c): 1 / (2 * 0.5) = 1.0

    def _band_solve(omega: float, alb_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Spherical leaves with equal leaf reflectance & transmittance (r_l = t_l = omega / 2):
        # diffuse upscatter beta = 0.5, direct upscatter beta_0 = 0.5 * (1 + mu_bar * K) / (1 + mu_bar * K) ->
        # Dickinson (1983) / Sellers (1985): beta_0 = ((1 + mu_bar * K) * a_s) / (omega * mu_bar * K)
        beta = 0.5
        # Single-scattering albedo of spherical leaves (Sellers, 1985, Eq. 15)
        tmp_as = 0.5 * omega * (0.5 / (0.5 + mu_s)) * (
            1.0 - (mu_s / (0.5 + mu_s)) * torch.log((0.5 + mu_s) / mu_s)
        )
        beta_0 = torch.clamp(((1.0 + mu_bar * k_opt) * tmp_as) / (omega * mu_bar * k_opt), 0.05, 0.95)

        b = 1.0 - (1.0 - beta) * omega
        c = omega * beta
        h = math.sqrt(b * b - c * c) / mu_bar
        p1 = b + mu_bar * h
        p2 = b - mu_bar * h

        s1 = torch.exp(-torch.clamp(h * lc, max=40.0))
        s2 = torch.exp(-torch.clamp(k_opt * lc, max=40.0))

        # Diffuse incident upward reflectance I_dif^up(0) and ground transmission
        d1 = p1 * (1.0 - alb_g * p2 / c) / s1 - p2 * (1.0 - alb_g * p1 / c) * s1
        d1_safe = torch.where(torch.abs(d1) < 1.0e-7, torch.full_like(d1, 1.0e-7), d1)
        h7 = c * (1.0 - alb_g * p2 / c) / (d1_safe * s1)
        h8 = -c * (1.0 - alb_g * p1 / c) * s1 / d1_safe
        alb_dif = torch.clamp(h7 + h8, 0.0, 1.0)
        trans_dif = torch.clamp((h7 * p1 / c) * s1 + (h8 * p2 / c) / s1, 0.0, 1.0)

        # Direct beam: avoid Dai et al. (2004, App. A) sigma = 0 singularity smoothly
        sigma_raw = (mu_bar * k_opt) ** 2 - (b * b - c * c)
        # Perturb k_opt by tiny 1e-4 relative offset if |sigma| < 1e-4 (smooth C^inf match to Eqs. A5-A6)
        k_reg = torch.where(torch.abs(sigma_raw) < 1.0e-4, k_opt + 2.0e-4, k_opt)
        sigma = (mu_bar * k_reg) ** 2 - (b * b - c * c)

        d_coef = omega * mu_bar * k_reg * beta_0
        f_coef = omega * mu_bar * k_reg * (1.0 - beta_0)
        u1 = b - c / torch.clamp(alb_g, min=1.0e-4)
        u2 = b - c * alb_g
        u3 = f_coef + c * alb_g

        h1 = -d_coef * (b - mu_bar * k_reg) - c * f_coef
        h4 = -f_coef * (b + mu_bar * k_reg) - c * d_coef

        d2 = (u1 + mu_bar * h) / s1 - (u1 - mu_bar * h) * s1
        d2_safe = torch.where(torch.abs(d2) < 1.0e-7, torch.full_like(d2, 1.0e-7), d2)

        h2 = (
            (d_coef - h1 * (b + mu_bar * k_reg) / sigma) * (u1 - mu_bar * h) * s1
            - p2 * (d_coef - c - (h1 / sigma) * (u1 + mu_bar * k_reg)) * s2
        ) / d2_safe
        h3 = (
            -(d_coef - h1 * (b + mu_bar * k_reg) / sigma) * (u1 + mu_bar * h) / s1
            + p1 * (d_coef - c - (h1 / sigma) * (u1 + mu_bar * k_reg)) * s2
        ) / d2_safe

        alb_dir = torch.clamp(h1 / sigma + h2 + h3, 0.0, 1.0)
        trans_dir = torch.clamp(
            s2 + (h4 / sigma) * s2
            + ((h2 * (u2 + mu_bar * h)) * s1 + (h3 * (u2 - mu_bar * h)) / s1) / c,
            0.0,
            1.0,
        )
        # Effective canopy transmission (beam+diffuse average) to forest floor
        trans_eff = torch.where(lc < 1.0e-3, torch.ones_like(lc), 0.5 * (trans_dir + trans_dif))
        alb_dir_out = torch.where(lc < 1.0e-3, alb_g, alb_dir)
        alb_dif_out = torch.where(lc < 1.0e-3, alb_g, alb_dif)
        return alb_dir_out, alb_dif_out, trans_eff

    vis_dir, vis_dif, tr_vis = _band_solve(LEAF_SCAT_VIS, alb_ground_vis)
    nir_dir, nir_dif, tr_nir = _band_solve(LEAF_SCAT_NIR, alb_ground_nir)
    return vis_dir, vis_dif, nir_dir, nir_dif, tr_vis, tr_nir


# ---------------------------------------------------------------------------
# 4. Biosphere Prognostic State, Daily EMA Acclimation, & Sub-Daily Two-Leaf Fluxes
# ---------------------------------------------------------------------------
@dataclass
class BiosphereState:
    """Prognostic EEO biosphere state per subgrid land tile (shape `[...]`)."""

    # EMA-smoothed midday & nighttime forcing history
    ema_vis_beam: torch.Tensor   # EMA midday incident visible beam irradiance [W m^-2]
    ema_vis_diff: torch.Tensor   # EMA midday incident visible diffuse irradiance [W m^-2]
    ema_mu0_mid: torch.Tensor    # EMA midday solar zenith cosine [-]
    ema_temp_mid: torch.Tensor   # EMA midday canopy/air temperature [K]
    ema_vpd_mid: torch.Tensor    # EMA midday vapor pressure deficit [Pa]
    ema_temp_night: torch.Tensor # EMA nighttime temperature [K]

    # Whole-canopy 25 degC acclimated biochemical pools [umol m^-2 s^-1]
    vcmax25_c3: torch.Tensor
    jmax25_c3: torch.Tensor
    r25_c3: torch.Tensor

    vcmax25_c4: torch.Tensor
    jmax25_c4: torch.Tensor
    vpmax25_c4: torch.Tensor
    r25_c4: torch.Tensor

    # Structural & hydrological ecosystem traits
    f_c4: torch.Tensor           # Autonomous C4 fraction in [0, 1] [-]
    lai: torch.Tensor            # Prognostic snow-free leaf area index [m^2 m^-2]
    h_canopy: torch.Tensor       # Prognostic potential canopy height [m]
    s_root_cap: torch.Tensor     # Optimal drought-bridging root-zone capacity S_R [kg m^-2 = mm]

    # Running annual hydroclimatic accumulators for EEO LAI, S_R, and h_c
    ema_ann_precip: torch.Tensor # Running mean annual precipitation [kg m^-2 yr^-1]
    ema_ann_pet: torch.Tensor    # Running mean annual potential ET [kg m^-2 yr^-1]
    ema_ann_a0: torch.Tensor     # Running mean annual potential GPP A_0 [mol C m^-2 yr^-1]
    ema_gsl_frac: torch.Tensor   # Running fraction of year with T > 0 degC [-]
    cum_deficit: torch.Tensor    # Running seasonal water deficit tracker [kg m^-2]


def init_biosphere_state(
    shape: tuple[int, ...],
    is_vegetated_land: torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BiosphereState:
    """Initialize neutral PFT-free biosphere state (`LAI = 0` on ocean/ice sheets)."""
    dev = torch.device(device)
    veg = is_vegetated_land.to(device=dev, dtype=dtype)

    def _f(val: float) -> torch.Tensor:
        return torch.full(shape, val, device=dev, dtype=dtype)

    return BiosphereState(
        ema_vis_beam=_f(220.0),
        ema_vis_diff=_f(80.0),
        ema_mu0_mid=_f(0.75),
        ema_temp_mid=_f(293.15),
        ema_vpd_mid=_f(1000.0),
        ema_temp_night=_f(286.15),
        vcmax25_c3=_f(45.0) * veg,
        jmax25_c3=_f(90.0) * veg,
        r25_c3=_f(0.80) * veg,
        vcmax25_c4=_f(25.0) * veg,
        jmax25_c4=_f(110.0) * veg,
        vpmax25_c4=_f(35.0) * veg,
        r25_c4=_f(0.50) * veg,
        f_c4=_f(0.15) * veg,
        lai=_f(2.0) * veg,
        h_canopy=_f(6.0) * veg,
        s_root_cap=_f(150.0) * veg + _f(20.0) * (1.0 - veg),
        ema_ann_precip=_f(900.0),
        ema_ann_pet=_f(1100.0),
        ema_ann_a0=_f(180.0),
        ema_gsl_frac=_f(0.65),
        cum_deficit=_f(0.0),
    )


def update_daily_acclimation_smith(
    bio: BiosphereState,
    midday_vis_beam: torch.Tensor,
    midday_vis_diff: torch.Tensor,
    midday_mu0: torch.Tensor,
    midday_temp: torch.Tensor,
    midday_vpd: torch.Tensor,
    night_temp: torch.Tensor,
    pres_sfc: torch.Tensor,
    co2_ppmv: float,
    lai_exposed: torch.Tensor,
    is_vegetated_land: torch.Tensor,
    alpha_ema: float = ALPHA_EMA_ACCLIM,
) -> None:
    """Update EMA forcing & Smith et al. (2019) / Scott & Smith (2022) acclimated capacities in-place.

    Follows the exact two-step acclimation light workflow:
    1. Smooth midday incident downward shortwave `(VIS_b, VIS_d)` and `mu_0` via EMA.
    2. Run `canopy_light_absorption_depury` on the EMA-smoothed incident radiation at
       current `lai_exposed` to obtain `tilde_I_c` for non-rectangular-hyperbola optimality.
    """
    veg = is_vegetated_land

    # 1. Update EMA of midday downward shortwave, solar geometry, midday T/VPD, and night T
    bio.ema_vis_beam = (1.0 - alpha_ema) * bio.ema_vis_beam + alpha_ema * torch.clamp(midday_vis_beam, min=0.0)
    bio.ema_vis_diff = (1.0 - alpha_ema) * bio.ema_vis_diff + alpha_ema * torch.clamp(midday_vis_diff, min=0.0)
    bio.ema_mu0_mid = (1.0 - alpha_ema) * bio.ema_mu0_mid + alpha_ema * torch.clamp(midday_mu0, min=0.05)
    bio.ema_temp_mid = (1.0 - alpha_ema) * bio.ema_temp_mid + alpha_ema * midday_temp
    bio.ema_vpd_mid = (1.0 - alpha_ema) * bio.ema_vpd_mid + alpha_ema * torch.clamp(midday_vpd, min=50.0)
    bio.ema_temp_night = (1.0 - alpha_ema) * bio.ema_temp_night + alpha_ema * night_temp

    # 2. Compute acclimation canopy light absorption from EMA downward SW at current exposed LAI
    acclim_rad = canopy_light_absorption_depury(
        bio.ema_vis_beam, bio.ema_vis_diff, bio.ema_mu0_mid, lai_exposed
    )
    i_abs = acclim_rad.i_c  # Whole-canopy absorbed PPFD [umol m^-2 s^-1]

    tg = bio.ema_temp_mid
    d_pa = bio.ema_vpd_mid
    ca_pa = (co2_ppmv * 1.0e-6) * pres_sfc
    oi_pa = C.O2_MIXING_RATIO * pres_sfc
    eta_star = water_viscosity_rel(tg)
    phi_psii = phi_psii_temperature(tg)

    # 3. C3 Non-Rectangular-Hyperbola Optimality (Smith et al., 2019, Eqs. 1-20)
    gamma_c3 = _arrhenius(GAMMA_STAR25_C3, DHA_GAMMA, tg) * (pres_sfc / 101325.0)
    kc_c3 = _arrhenius(KC25_C3, DHA_KC_C3, tg)
    ko_c3 = _arrhenius(KO25_C3, DHA_KO_C3, tg)
    km_c3 = kc_c3 * (1.0 + oi_pa / ko_c3)

    xi_c3 = torch.sqrt(BETA_C3 * (km_c3 + gamma_c3) / (1.6 * eta_star))
    chi_c3 = torch.clamp(
        gamma_c3 / ca_pa + (1.0 - gamma_c3 / ca_pa) * (xi_c3 / (xi_c3 + torch.sqrt(d_pa))),
        0.05,
        0.95,
    )
    ci_c3 = chi_c3 * ca_pa
    m_c3 = torch.clamp((ci_c3 - gamma_c3) / torch.clamp(ci_c3 + 2.0 * gamma_c3, min=0.1), min=0.0)
    mc_c3 = torch.clamp((ci_c3 - gamma_c3) / torch.clamp(ci_c3 + km_c3, min=0.5), min=1.0e-4)

    omega_c3, omega_star_c3 = smith_omega(m_c3)
    jmax_opt_c3 = phi_psii * i_abs * omega_c3                                # Smith et al. (2019, Eq. 15)
    aj_opt_c3 = phi_psii * i_abs * m_c3 * omega_star_c3 / (8.0 * THETA_NRH)  # Smith et al. (2019, Eq. 17)
    vcmax_opt_c3 = torch.where(m_c3 > 4.0 * COST_C_SMITH, aj_opt_c3 / mc_c3, torch.zeros_like(i_abs))

    fv_c3 = torch.clamp(peaked_arrhenius(tg, tg, HA_VCMAX, is_jmax=False), min=0.05)
    fj_c3 = torch.clamp(peaked_arrhenius(tg, tg, HA_JMAX, is_jmax=True), min=0.05)
    fr_night = torch.clamp(heskel_respiration_scale(bio.ema_temp_night), min=0.05)

    lai_scale = torch.clamp(lai_exposed, min=1.0)
    vc25_c3_tgt = torch.minimum(torch.clamp(vcmax_opt_c3 / fv_c3, min=0.0), 300.0 * lai_scale)
    jm25_c3_tgt = torch.minimum(torch.clamp(jmax_opt_c3 / fj_c3, min=0.0), 700.0 * lai_scale)
    r25_c3_tgt = torch.minimum(torch.clamp(B_RESP_REN * vcmax_opt_c3 / fr_night, min=0.0), 10.0 * lai_scale)

    # 4. C4 Non-Rectangular-Hyperbola Optimality (Scott & Smith, 2022, Eqs. 1-25; von Caemmerer & Furbank, 1999)
    gamma_c4 = _arrhenius(GAMMA_STAR25_C4, DHA_GAMMA, tg) * (pres_sfc / 101325.0)
    kp_pa_c4 = _arrhenius(KP25_UMOL_C4 * 1.0e-6, DHA_KP_C4, tg) * pres_sfc
    kc_c4 = _arrhenius(KC25_C4, DHA_KC_C4, tg)
    ko_c4 = _arrhenius(KO25_C4, DHA_KO_C4, tg)
    km_c4 = kc_c4 * (1.0 + oi_pa / ko_c4)

    xi_c4 = torch.sqrt(BETA_C4 * kp_pa_c4 / (1.6 * eta_star))
    chi_m_c4 = torch.clamp(xi_c4 / (xi_c4 + torch.sqrt(d_pa)), 0.05, 0.92)
    cm_c4 = chi_m_c4 * ca_pa

    # C4 RuBP/PEP regeneration requires 5 mol e^- per mol CO2 vs 4 in C3 (4/5 = 0.80; von Caemmerer & Furbank, 1999)
    # and operates at bundle-sheath partial pressure C_bs >> Gamma^* (Scott & Smith, 2022, Eqs. 6 & 22)
    chill_c4 = torch.sigmoid(0.35 * ((tg - C.T_FREEZE) - 12.0))
    gbs_canopy = GBS_C4 * torch.clamp(lai_exposed, min=0.1)
    cbs_c4 = cm_c4 + 250.0  # Initial bundle-sheath estimate [Pa], refined by 2 fixed-point steps
    for _ in range(2):
        m_c4 = 0.80 * chill_c4 * torch.clamp((cbs_c4 - gamma_c4) / torch.clamp(cbs_c4 + 2.0 * gamma_c4, min=0.1), min=0.0)
        omega_c4, omega_star_c4 = smith_omega(m_c4)
        al_opt_c4 = phi_psii * i_abs * m_c4 * omega_star_c4 / (8.0 * THETA_NRH)  # Scott & Smith (2022, Eq. 5)
        leak_c4 = (PHI_LEAK_C4 / (1.0 - PHI_LEAK_C4)) * al_opt_c4                # Scott & Smith (2022, Eq. 15)
        cbs_c4 = cm_c4 + (leak_c4 * 1.0e-6 / gbs_canopy) * pres_sfc              # Scott & Smith (2022, Eq. 22)

    jmax_opt_c4 = phi_psii * i_abs * omega_c4                                    # Scott & Smith (2022, Eq. 25)
    vpmax_opt_c4 = (al_opt_c4 + leak_c4) * (kp_pa_c4 + cm_c4) / torch.clamp(cm_c4, min=0.5)  # Eq. 16
    vcmax_opt_c4 = al_opt_c4 * (cbs_c4 + km_c4) / torch.clamp(cbs_c4 - gamma_c4, min=1.0)    # Eq. 24

    fp_c4 = torch.clamp(peaked_arrhenius(tg, tg, HA_VPMAX, is_jmax=False), min=0.05)
    vc25_c4_tgt = torch.minimum(torch.clamp(vcmax_opt_c4 / fv_c3, min=0.0), 300.0 * lai_scale)
    jm25_c4_tgt = torch.minimum(torch.clamp(jmax_opt_c4 / fj_c3, min=0.0), 700.0 * lai_scale)
    vp25_c4_tgt = torch.minimum(torch.clamp(vpmax_opt_c4 / fp_c4, min=0.0), 400.0 * lai_scale)
    r25_c4_tgt = torch.minimum(torch.clamp(B_RESP_REN * vcmax_opt_c4 / fr_night, min=0.0), 10.0 * lai_scale)

    # Update capacity pools with EMA filter (Mengoli et al., 2022; Ren et al., 2025)
    bio.vcmax25_c3 = torch.where(veg, (1.0 - alpha_ema) * bio.vcmax25_c3 + alpha_ema * vc25_c3_tgt, torch.zeros_like(tg))
    bio.jmax25_c3 = torch.where(veg, (1.0 - alpha_ema) * bio.jmax25_c3 + alpha_ema * jm25_c3_tgt, torch.zeros_like(tg))
    bio.r25_c3 = torch.where(veg, (1.0 - alpha_ema) * bio.r25_c3 + alpha_ema * r25_c3_tgt, torch.zeros_like(tg))

    bio.vcmax25_c4 = torch.where(veg, (1.0 - alpha_ema) * bio.vcmax25_c4 + alpha_ema * vc25_c4_tgt, torch.zeros_like(tg))
    bio.jmax25_c4 = torch.where(veg, (1.0 - alpha_ema) * bio.jmax25_c4 + alpha_ema * jm25_c4_tgt, torch.zeros_like(tg))
    bio.vpmax25_c4 = torch.where(veg, (1.0 - alpha_ema) * bio.vpmax25_c4 + alpha_ema * vp25_c4_tgt, torch.zeros_like(tg))
    bio.r25_c4 = torch.where(veg, (1.0 - alpha_ema) * bio.r25_c4 + alpha_ema * r25_c4_tgt, torch.zeros_like(tg))

    # 5. Autonomous C4 Dominance Fraction F_4 (Scott & Smith, 2022, Eq. 26; Luo et al., 2024; Lavergne et al., 2026)
    # C4 grasses outcompete C3 in warm, high-light open/semi-open canopies (f_open = exp(-0.5 * LAI))
    # whenever net C4 assimilation exceeds net C3 assimilation (Delta_A = A_n,C4 - A_n,C3 > 0).
    an_c3_mid = aj_opt_c3 - B_RESP_REN * vcmax_opt_c3 * (heskel_respiration_scale(tg) / fr_night)
    an_c4_mid = al_opt_c4 - B_RESP_REN * vcmax_opt_c4 * (heskel_respiration_scale(tg) / fr_night)
    adv_c4 = (an_c4_mid - an_c3_mid) / torch.clamp(torch.abs(an_c3_mid) + 1.0, min=1.0)
    # Tree-shade exclusion of C4 understory in closed-canopy forests (Luo et al., 2024, Methods)
    herb_openness = torch.clamp(1.15 - 0.18 * bio.lai, 0.02, 1.0)
    f4_target = herb_openness * torch.sigmoid(8.0 * adv_c4)
    # Slow ecological turnover (~60-day smoothing)
    alpha_f4 = 0.25 * alpha_ema
    bio.f_c4 = torch.where(veg, (1.0 - alpha_f4) * bio.f_c4 + alpha_f4 * f4_target, torch.zeros_like(tg))


@dataclass
class TwoLeafFluxResult:
    """Instantaneous Two-Leaf carbon assimilation and stomatal conductances."""

    gpp: torch.Tensor       # Canopy gross primary production [umol CO2 m^-2 s^-1]
    resp_leaf: torch.Tensor # Canopy leaf dark respiration [umol CO2 m^-2 s^-1]
    an_net: torch.Tensor    # Canopy net assimilation GPP - R_leaf [umol CO2 m^-2 s^-1]
    gsw_sun: torch.Tensor   # Sunlit stomatal conductance to H2O [m s^-1]
    gsw_sh: torch.Tensor    # Shaded stomatal conductance to H2O [m s^-1]
    l_sun: torch.Tensor     # Instantaneous sunlit LAI [m^2 m^-2]
    l_sh: torch.Tensor      # Instantaneous shaded LAI [m^2 m^-2]


def compute_two_leaf_gas_exchange(
    bio: BiosphereState,
    vis_beam_wm2: torch.Tensor,
    vis_diff_wm2: torch.Tensor,
    mu0: torch.Tensor,
    temp_sun: torch.Tensor,
    temp_sh: torch.Tensor,
    vpd_sun: torch.Tensor,
    vpd_sh: torch.Tensor,
    pres_sfc: torch.Tensor,
    co2_ppmv: float,
    lai_exposed: torch.Tensor,
    water_stress_beta: torch.Tensor,
) -> TwoLeafFluxResult:
    """Evaluate sub-daily sunlit and shaded C3/C4 photosynthesis and stomatal conductance.

    1. Recomputes the acclimation light profile `acclim_rad` from `bio.ema_vis_beam,
       bio.ema_vis_diff, bio.ema_mu0_mid` at `lai_exposed`.
    2. Allocates `V_cmax25, J_max25, V_pmax25, R_25` to sunlit and shaded pools via
       Eqs. (A1.24)-(A1.25) of Hashimoto (2026, ATAMI).
    3. Evaluates non-rectangular-hyperbola `J` (Smith et al., 2019, Eq. 9), hyperbolic
       co-limitation, MMRT respiration (Heskel et al., 2016), and Medlyn/Prentice/Scott-Smith
       optimal stomatal conductance in kinematic units [m s^-1] (Dai et al., 2004, Eq. 45).
    """
    acclim_rad = canopy_light_absorption_depury(
        bio.ema_vis_beam, bio.ema_vis_diff, bio.ema_mu0_mid, lai_exposed
    )
    inst_rad = canopy_light_absorption_depury(vis_beam_wm2, vis_diff_wm2, mu0, lai_exposed)

    k_db = inst_rad.k_db
    l_sun = inst_rad.l_sun
    l_sh = inst_rad.l_sh
    has_veg = lai_exposed >= 0.01

    # Allocate C3 and C4 enzyme pools into sunlit and shaded leaves
    vc25_c3_sun, vc25_c3_sh = allocate_two_leaf_capacity(bio.vcmax25_c3, acclim_rad, k_db, l_sun, lai_exposed)
    jm25_c3_sun, jm25_c3_sh = allocate_two_leaf_capacity(bio.jmax25_c3, acclim_rad, k_db, l_sun, lai_exposed)
    r25_c3_sun, r25_c3_sh = allocate_two_leaf_capacity(bio.r25_c3, acclim_rad, k_db, l_sun, lai_exposed)

    vc25_c4_sun, vc25_c4_sh = allocate_two_leaf_capacity(bio.vcmax25_c4, acclim_rad, k_db, l_sun, lai_exposed)
    jm25_c4_sun, jm25_c4_sh = allocate_two_leaf_capacity(bio.jmax25_c4, acclim_rad, k_db, l_sun, lai_exposed)
    vp25_c4_sun, vp25_c4_sh = allocate_two_leaf_capacity(bio.vpmax25_c4, acclim_rad, k_db, l_sun, lai_exposed)
    r25_c4_sun, r25_c4_sh = allocate_two_leaf_capacity(bio.r25_c4, acclim_rad, k_db, l_sun, lai_exposed)

    ca_pa = (co2_ppmv * 1.0e-6) * pres_sfc
    ca_mol = co2_ppmv * 1.0e-6
    oi_pa = C.O2_MIXING_RATIO * pres_sfc
    tg = bio.ema_temp_mid
    ws = torch.clamp(water_stress_beta, 0.0, 1.0)

    def _eval_fraction(
        i_par: torch.Tensor,
        tl: torch.Tensor,
        d_pa: torch.Tensor,
        vc25_3: torch.Tensor,
        jm25_3: torch.Tensor,
        r25_3: torch.Tensor,
        vc25_4: torch.Tensor,
        jm25_4: torch.Tensor,
        vp25_4: torch.Tensor,
        r25_4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_s = torch.clamp(d_pa, min=50.0)
        eta_s = water_viscosity_rel(tl)
        phi_t = phi_psii_temperature(tl)
        fv = peaked_arrhenius(tl, tg, HA_VCMAX, is_jmax=False)
        fj = peaked_arrhenius(tl, tg, HA_JMAX, is_jmax=True)
        fp = peaked_arrhenius(tl, tg, HA_VPMAX, is_jmax=False)
        fr = heskel_respiration_scale(tl)

        # --- C3 leaf fraction ---
        gamma3 = _arrhenius(GAMMA_STAR25_C3, DHA_GAMMA, tl) * (pres_sfc / 101325.0)
        km3 = _arrhenius(KC25_C3, DHA_KC_C3, tl) * (1.0 + oi_pa / _arrhenius(KO25_C3, DHA_KO_C3, tl))
        xi3 = torch.sqrt(BETA_C3 * (km3 + gamma3) / (1.6 * eta_s))
        chi3 = torch.clamp(gamma3 / ca_pa + (1.0 - gamma3 / ca_pa) * (xi3 / (xi3 + torch.sqrt(d_s))), 0.05, 0.95)
        ci3 = chi3 * ca_pa

        vc3 = ws * vc25_3 * fv
        jm3 = ws * jm25_3 * fj
        j_3 = non_rectangular_hyperbola(phi_t * i_par, jm3, THETA_NRH)
        ac3 = vc3 * torch.clamp((ci3 - gamma3) / (ci3 + km3), min=0.0)
        aj3 = 0.25 * j_3 * torch.clamp((ci3 - gamma3) / (ci3 + 2.0 * gamma3), min=0.0)
        # Smooth hyperbolic co-limitation (theta_cj = 0.98)
        b_cj3 = ac3 + aj3
        gpp3 = (b_cj3 - torch.sqrt(torch.clamp(b_cj3 * b_cj3 - 4.0 * 0.98 * ac3 * aj3, min=0.0))) / (2.0 * 0.98)
        rd3 = r25_3 * fr
        an3_pos = torch.clamp(gpp3 - rd3, min=0.0)
        # Optimal stomatal conductance [mol m^-2 s^-1]: g_sw = 1.6 * A_n / (c_a_mol * (1 - chi))
        # (Prentice et al., 2014; Medlyn et al., 2011; Scott & Smith, 2022)
        gsw3_mol = 1.6 * (an3_pos * 1.0e-6) / torch.clamp(ca_mol * (1.0 - chi3), min=1.0e-6)

        # --- C4 leaf fraction ---
        gamma4 = _arrhenius(GAMMA_STAR25_C4, DHA_GAMMA, tl) * (pres_sfc / 101325.0)
        kp4 = _arrhenius(KP25_UMOL_C4 * 1.0e-6, DHA_KP_C4, tl) * pres_sfc
        km4 = _arrhenius(KC25_C4, DHA_KC_C4, tl) * (1.0 + oi_pa / _arrhenius(KO25_C4, DHA_KO_C4, tl))
        xi4 = torch.sqrt(BETA_C4 * kp4 / (1.6 * eta_s))
        chi4 = torch.clamp(xi4 / (xi4 + torch.sqrt(d_s)), 0.05, 0.92)
        cm4 = chi4 * ca_pa

        vc4 = ws * vc25_4 * fv
        jm4 = ws * jm25_4 * fj
        vp4 = ws * vp25_4 * fp
        j_4 = non_rectangular_hyperbola(phi_t * i_par, jm4, THETA_NRH)
        al4 = 0.20 * j_4 * torch.sigmoid(0.35 * ((tl - C.T_FREEZE) - 12.0))  # 5 mol e^- / mol CO2 (von Caemmerer, 2000)
        ap4 = (1.0 - PHI_LEAK_C4) * vp4 * cm4 / torch.clamp(cm4 + kp4, min=0.5)
        ac4 = vc4  # Bundle-sheath CO2-saturated Rubisco limit
        b_cj4 = ac4 + al4
        acj4 = (b_cj4 - torch.sqrt(torch.clamp(b_cj4 * b_cj4 - 4.0 * 0.85 * ac4 * al4, min=0.0))) / (2.0 * 0.85)
        b_p4 = acj4 + ap4
        gpp4 = (b_p4 - torch.sqrt(torch.clamp(b_p4 * b_p4 - 4.0 * 0.85 * acj4 * ap4, min=0.0))) / (2.0 * 0.85)
        rd4 = r25_4 * fr
        an4_pos = torch.clamp(gpp4 - rd4, min=0.0)
        gsw4_mol = 1.6 * (an4_pos * 1.0e-6) / torch.clamp(ca_mol * (1.0 - chi4), min=1.0e-6)

        # Weighted by autonomous C4 fraction F_4
        f4 = bio.f_c4
        gpp_mix = (1.0 - f4) * gpp3 + f4 * gpp4
        rd_mix = (1.0 - f4) * rd3 + f4 * rd4
        gsw_mol = (1.0 - f4) * gsw3_mol + f4 * gsw4_mol + 0.002 * ws * (l_sun + l_sh > 0.01).to(tl.dtype)
        # Convert mol m^-2 s^-1 to kinematic m s^-1 via ideal gas molar volume R * T / P (Dai et al., 2004, Eq. 45)
        gsw_ms = gsw_mol * (C.R_GAS_UNIVERSAL * tl / pres_sfc)
        return gpp_mix, rd_mix, gsw_ms

    gpp_sun, rd_sun, gsw_sun_raw = _eval_fraction(
        inst_rad.i_sun,
        temp_sun,
        vpd_sun,
        vc25_c3_sun,
        jm25_c3_sun,
        r25_c3_sun,
        vc25_c4_sun,
        jm25_c4_sun,
        vp25_c4_sun,
        r25_c4_sun,
    )
    gpp_sh, rd_sh, gsw_sh_raw = _eval_fraction(
        inst_rad.i_sh,
        temp_sh,
        vpd_sh,
        vc25_c3_sh,
        jm25_c3_sh,
        r25_c3_sh,
        vc25_c4_sh,
        jm25_c4_sh,
        vp25_c4_sh,
        r25_c4_sh,
    )

    zero = torch.zeros_like(lai_exposed)
    gpp_tot = torch.where(has_veg, gpp_sun + gpp_sh, zero)
    rd_tot = torch.where(has_veg, rd_sun + rd_sh, zero)
    gsw_sun = torch.where(has_veg & (l_sun > 1.0e-4), gsw_sun_raw, zero)
    gsw_sh = torch.where(has_veg & (l_sh > 1.0e-4), gsw_sh_raw, zero)

    return TwoLeafFluxResult(
        gpp=gpp_tot,
        resp_leaf=rd_tot,
        an_net=gpp_tot - rd_tot,
        gsw_sun=gsw_sun,
        gsw_sh=gsw_sh,
        l_sun=l_sun,
        l_sh=l_sh,
    )


# ---------------------------------------------------------------------------
# 5. Principal-Branch Lambert W_0 & Daily Prognostic EEO LAI / S_R / h_c
# ---------------------------------------------------------------------------
def lambert_w0(x: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    """Principal branch `W_0(x)` on `x >= -1 / e` via Halley's third-order iteration.

    Used in Zhou et al. (2025, Global Change Biol., 31, Eq. 15) for `x = -k * mu * exp(-k * mu)`
    which always lies in `[-1 / e, 0]`. Note that for `u = k * mu`:
        -u * exp(-u) has two real branches when u > 0:
        - If 0 <= u <= 1: W_0(-u * exp(-u)) = -u  ->  mu + W_0 / k = 0 (canopy cannot sustain itself!)
        - If u > 1:       W_0(-u * exp(-u)) in (-1, 0) > -u  ->  L_s = mu + W_0 / k > 0!
    """
    inv_e = 1.0 / math.e
    x_clamped = torch.clamp(x, min=-inv_e + 1.0e-7, max=100.0)
    # Accurate initial guess near branch point -1/e and for positive x
    p = torch.sqrt(torch.clamp(2.0 * (math.e * x_clamped + 1.0), min=1.0e-12))
    w = torch.where(
        x_clamped < 0.0,
        -1.0 + p * (1.0 - p / 3.0),
        torch.log1p(torch.clamp(x_clamped, min=0.0)),
    )
    for _ in range(n_iter):
        ew = torch.exp(w)
        wew = w * ew
        f = wew - x_clamped
        wp1 = torch.clamp(w + 1.0, min=1.0e-5)
        denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1)
        w = w - f / torch.clamp(denom, min=1.0e-12)
    return w


def update_daily_lai_root_height(
    bio: BiosphereState,
    daily_precip_mm: torch.Tensor,
    daily_pet_mm: torch.Tensor,
    daily_ppfd_mol_m2: torch.Tensor,
    daily_temp_k: torch.Tensor,
    daily_vpd_pa: torch.Tensor,
    pres_sfc: torch.Tensor,
    co2_ppmv: float,
    water_stress_beta: torch.Tensor,
    is_vegetated_land: torch.Tensor,
) -> None:
    """Update prognostic EEO LAI, root-zone storage `S_R`, and canopy height `h_c` once per day.

    References
    ----------
    - Cai et al. (2025, Commun. Earth Environ., 6, 19, Eqs. 1-2 & 12):
      Energy-limited `fAPAR_c` and water-limited `fAPAR_w` combined via log-sum-exp (`K = 10`).
    - Zhou et al. (2025, Global Change Biol., 31, e70125, Eqs. 6 & 15-20):
      Aridity-dependent `f_0(AI)` and Lambert-`W_0` steady-state `L_s` with `alpha = 0.067` EMA.
    - Gao et al. (2014, Geophys. Res. Lett., 41, 7916-7923):
      Cumulative dry-season water deficit `(PET * fAPAR_max - P)^+` sizing `S_R`.
    - Kempes et al. (2011, PLoS ONE, 6, e20551, Eqs. 2-5):
      Allometric vascular balance between available water flow `Q_p` and evaporative/metabolic scaling.
    """
    veg = is_vegetated_land
    alpha_yr = 1.0 / 365.2425  # 1-year e-folding memory for annual hydroclimatic totals

    # 1. Potential daily GPP A_0 [mol C m^-2 d^-1] if fAPAR = 1 (Smith et al., 2019; Zhou et al., 2025, Eq. 1)
    ca_pa = (co2_ppmv * 1.0e-6) * pres_sfc
    oi_pa = C.O2_MIXING_RATIO * pres_sfc
    eta_s = water_viscosity_rel(daily_temp_k)
    gamma3 = _arrhenius(GAMMA_STAR25_C3, DHA_GAMMA, daily_temp_k) * (pres_sfc / 101325.0)
    km3 = _arrhenius(KC25_C3, DHA_KC_C3, daily_temp_k) * (
        1.0 + oi_pa / _arrhenius(KO25_C3, DHA_KO_C3, daily_temp_k)
    )
    d_s = torch.clamp(daily_vpd_pa, min=50.0)
    xi3 = torch.sqrt(BETA_C3 * (km3 + gamma3) / (1.6 * eta_s))
    chi3 = torch.clamp(gamma3 / ca_pa + (1.0 - gamma3 / ca_pa) * (xi3 / (xi3 + torch.sqrt(d_s))), 0.05, 0.95)
    ci3 = chi3 * ca_pa
    m_c3 = torch.clamp((ci3 - gamma3) / torch.clamp(ci3 + 2.0 * gamma3, min=0.1), min=0.0)
    _, omega_star3 = smith_omega(m_c3)

    lue_c3 = phi_psii_temperature(daily_temp_k) * m_c3 * omega_star3 / (8.0 * THETA_NRH)
    # Apply thermal growing-season gate below 0 degC and root-zone water supply constraint
    cold_gate = torch.sigmoid(0.5 * (daily_temp_k - C.T_FREEZE))
    a0_day = lue_c3 * torch.clamp(daily_ppfd_mol_m2, min=0.0) * cold_gate * torch.clamp(water_stress_beta, 0.05, 1.0)

    # Update annual running accumulators
    bio.ema_ann_precip = (1.0 - alpha_yr) * bio.ema_ann_precip + alpha_yr * (daily_precip_mm * 365.2425)
    bio.ema_ann_pet = (1.0 - alpha_yr) * bio.ema_ann_pet + alpha_yr * (daily_pet_mm * 365.2425)
    bio.ema_ann_a0 = (1.0 - alpha_yr) * bio.ema_ann_a0 + alpha_yr * (a0_day * 365.2425)
    above_zero = (daily_temp_k > C.T_FREEZE).to(daily_temp_k.dtype)
    bio.ema_gsl_frac = (1.0 - alpha_yr) * bio.ema_gsl_frac + alpha_yr * above_zero

    # 2. Annual maximum greenness fAPAR_max & LAI_max (Cai et al., 2025, Eqs. 1-2 & 12; Zhou et al., 2025, Eq. 6)
    # Per Zhou et al. (2025, Sec. 2.1.2), f_0 = 0.65 in energy-limited climates (AI <= 1.9) and declines
    # as AI increases beyond 1.9 into water-limited regimes.
    ai = torch.clamp(bio.ema_ann_pet / torch.clamp(bio.ema_ann_precip, min=10.0), 0.05, 50.0)
    ai_arid = torch.clamp(ai, min=1.9)
    f0 = 0.65 * torch.exp(-0.604169 * (torch.log(ai_arid / 1.9) ** 2))  # Zhou et al. (2025, Eq. 6)
    p_mol_yr = bio.ema_ann_precip * (1000.0 / 18.01528)            # 1 kg H2O = 55.508 mol H2O
    a0_yr = torch.clamp(bio.ema_ann_a0, min=1.0e-2)

    fapar_w = f0 * p_mol_yr * ca_pa * (1.0 - chi3) / torch.clamp(1.6 * d_s * a0_yr, min=1.0)
    fapar_c = 1.0 - Z_FOLIAGE_COST / (K_BEER_LAI * a0_yr)
    # Smooth minimum via log-sum-exp with K = 10 (Cai et al., 2025, Eq. 12)
    k_lse = 10.0
    m_pair = torch.minimum(fapar_w, fapar_c)
    fapar_smooth = m_pair - (1.0 / k_lse) * torch.log(
        torch.exp(-k_lse * (fapar_w - m_pair)) + torch.exp(-k_lse * (fapar_c - m_pair))
    )
    fapar_max = torch.clamp(fapar_smooth, 0.0, 0.96)
    lai_max = -torch.log(1.0 - fapar_max) / K_BEER_LAI

    # 3. Daily steady-state LAI via Lambert W_0 & EMA phenology (Zhou et al., 2025, Eqs. 15, 16, 20)
    gsl_days = torch.clamp(bio.ema_gsl_frac * 365.2425, min=15.0)
    m_alloc = SIGMA_ZHOU * gsl_days * lai_max / torch.clamp(a0_yr * fapar_max, min=0.1)
    mu_lai = m_alloc * a0_day
    u_lai = K_BEER_LAI * mu_lai
    arg_w = -u_lai * torch.exp(-torch.clamp(u_lai, max=40.0))
    ls_raw = mu_lai + lambert_w0(arg_w) / K_BEER_LAI
    l_steady = torch.clamp(torch.minimum(ls_raw, lai_max), min=0.0)

    lai_new = (1.0 - ALPHA_EMA_ACCLIM) * bio.lai + ALPHA_EMA_ACCLIM * l_steady
    bio.lai = torch.where(veg, torch.clamp(lai_new, 0.0, 8.0), torch.zeros_like(lai_new))

    # 4. Optimal drought-bridging root-zone storage S_R [kg m^-2] (Gao et al., 2014)
    daily_demand = fapar_max * daily_pet_mm
    bio.cum_deficit = torch.clamp(bio.cum_deficit + (daily_demand - daily_precip_mm), 0.0, 800.0)
    sr_target = torch.clamp(50.0 + 1.35 * bio.cum_deficit, 50.0, 650.0)
    # Slow structural root adaptation (~1-year memory; rapid upward ratchet during dry spells)
    alpha_sr = torch.where(sr_target > bio.s_root_cap, 10.0 * alpha_yr, alpha_yr)
    bio.s_root_cap = torch.where(
        veg,
        (1.0 - alpha_sr) * bio.s_root_cap + alpha_sr * sr_target,
        torch.full_like(sr_target, 20.0),
    )

    # 5. Potential canopy height h_c [m] from vascular water-energy scaling (Kempes et al., 2011, Eqs. 2-5)
    # In Kempes et al. (2011), maximum sustainable height scales positively with available water
    # capture (P / PET) and annual photosynthetic energy surplus (fAPAR_max * A_0):
    # yields ~30-40 m in humid tropical forests (Af), ~15-22 m in temperate/boreal forests (Cf/Df),
    # ~2-6 m in savannas/shrublands (Aw/BS), and < 0.5 m in tundra/deserts (ET/BW).
    moisture_idx = torch.clamp(bio.ema_ann_precip / torch.clamp(bio.ema_ann_pet, min=50.0), 0.0, 2.5)
    prod_idx = torch.clamp((fapar_max * a0_yr) / 180.0, 0.0, 2.0)
    hc_target = torch.clamp(32.0 * torch.pow(moisture_idx, 0.85) * torch.pow(prod_idx, 0.75), 0.05, 40.0)
    bio.h_canopy = torch.where(
        veg,
        (1.0 - alpha_yr) * bio.h_canopy + alpha_yr * hc_target,
        torch.zeros_like(hc_target),
    )
