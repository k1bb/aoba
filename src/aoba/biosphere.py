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

AOBA Formulation Reference: Sec. 7.2-7.5, Eqs. (7.1)-(7.31) (Ross-Goudriaan Optimal Leaf Angle, Discrete-Crown Clumping, Two-Leaf C3/C4 P-Model, Liebig Co-Limitation, Lambert-W0 LAI, LES Habit, S_R & h_c).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import can_use_elem_metal, dispatch_two_leaf_gas_exchange_metal

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

# Peaked Arrhenius enzyme kinetics (Kumarathunge et al., 2019; Schneider et al., 2025, Table 1 & Eq. 7)
HA_VCMAX: float = 71513.0          # Reference V_cmax activation energy [J mol^-1]
HA_VCMAX_BASE: float = 42600.0     # V_cmax activation energy intercept at 0 degC [J mol^-1]
HA_VCMAX_SLOPE: float = 1140.0     # V_cmax activation energy slope dH_a,V/dT_g [J mol^-1 K^-1]
HA_JMAX: float = 40710.0           # J_max activation energy [J mol^-1] (Kumarathunge et al., 2019)
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
Z_FOLIAGE_COST: float = 12.227     # Deciduous annual foliage carbon cost z [mol C m^-2 yr^-1] (Zhou et al., 2025, Eq. 10)
K_BEER_LAI: float = 0.50           # Canopy extinction coefficient k [-] (Cai et al., 2025; Zhou et al., 2025)
SIGMA_ZHOU: float = 0.771          # Seasonal LAI-GPP shape factor sigma [-] (Zhou et al., 2025, Eq. 20)
ALPHA_EMA_ACCLIM: float = 0.067    # Daily EMA weight (~15-day e-folding memory) [-] (Mengoli et al., 2022; Zhou et al., 2025)

# Evergreen vs. Deciduous leaf-habit EEO constants (Kikuzawa, 1991, Eqs. 1-5; Givnish, 2002;
# Sakai & Larcher, 1987; Wang et al., 2023, Science Advances, 9, eadd5667)
EVG_COST_RATIO_BASE: float = 1.35  # Warm evergreen broadleaf construction-cost ratio C_evg/C_dec [-]
EVG_COST_RATIO_COLD: float = 0.55  # Additional frost-hardening LMA construction-cost ratio below 0 degC [-]
EVG_COLD_EFF_PENALTY: float = 0.14 # Photosynthetic efficiency penalty for cold-hardened evergreen needles [-]
LEAF_AGING_B_YR: float = 15.0      # Kikuzawa (1991) hypothetical zero-gain evergreen leaf age b [yr]
DECID_FLUSH_DAYS: float = 25.0     # Deciduous seasonal canopy-flush photosynthetic lag [days]
EVG_OFFSEASON_RESP25: float = 32.0 # Evergreen off-season maintenance/stress cost at 25 degC [mol C m^-2 yr^-1]
EVG_FROST_CRIT_K: float = 233.15   # Deep supercooling winter frost-damage threshold (-40 degC) [K]
EVG_FROST_COST: float = 18.0       # Winter frost-damage carbon penalty below T_frost [mol C m^-2 yr^-1]


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
    t_home_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Peaked Arrhenius thermal response relative to 298.15 K with full enzyme acclimation.

    Reference: Kumarathunge et al. (2019, New Phytol., 222, 768-784, Table 3) &
    Schneider, Gessler, & Stocker (2025, JAMES, 17, e2024MS004789, Table 1 & Eq. 7):
        H_a,V(T_g)           = 42600 + 1140 * T_growth_C  [J mol^-1]
        Delta_S_V(T_g)       = 645.13 - 0.38 * T_growth_C [J mol^-1 K^-1]
        H_a,J                = 40710                      [J mol^-1]
        Delta_S_J(T_g,T_hom) = 658.77 - 0.84 * T_home_C - 0.52 * (T_growth_C - T_home_C).
    """
    t0 = 298.15
    t_s = torch.clamp(temp_k, min=250.0, max=330.0)
    tg_c = torch.clamp(t_growth_k - C.T_FREEZE, 0.0, 40.0)
    if is_jmax:
        th_c = torch.clamp((t_home_k if t_home_k is not None else t_growth_k) - C.T_FREEZE, 0.0, 40.0)
        ds = 658.77 - 0.84 * th_c - 0.52 * (tg_c - th_c)
        ha_eff: float | torch.Tensor = ha
    else:
        ds = 645.13 - 0.38 * tg_c
        ha_eff = (HA_VCMAX_BASE + HA_VCMAX_SLOPE * tg_c) if ha == HA_VCMAX else ha

    num = 1.0 + torch.exp((t0 * ds - HD_ENZYME) / (C.R_GAS_UNIVERSAL * t0))
    den = 1.0 + torch.exp((t_s * ds - HD_ENZYME) / (C.R_GAS_UNIVERSAL * t_s))
    return torch.exp(ha_eff * (t_s - t0) / (C.R_GAS_UNIVERSAL * t0 * t_s)) * (num / den)


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
    kd_prime: torch.Tensor | None = None    # Clumped diffuse PAR extinction coefficient [m^2 m^-2]
    omega_beam: torch.Tensor | None = None  # Direct-beam discrete-crown clumping index Omega(mu0, LAI) [-]


# Universal discrete-crown packing scale L_{c,0} [m^2 m^-2] (Li & Strahler, 1992; Ni-Meister et al., 2010)
CROWN_LAI_MIN: float = 1.50
CHI_LEAF_MIN: float = -0.40
CHI_LEAF_MAX: float = 0.50


def ross_goudriaan_optics(
    mu0: torch.Tensor,
    chi_leaf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form Ross (1975) / Goudriaan (1977) / Dai et al. (2004, Eq. A4) leaf projection optics.

    Maps leaf orientation index `chi_L in [-0.4, +0.5]` (`0`: spherical, `<0`: erectophile,
    `>0`: planophile) to `(G(mu0), mu_bar, phi_1, phi_2)` with smooth singularity-free Taylor
    behavior as `phi_2 -> 0` (reducing identically to `G = 0.5, mu_bar = 1.0` at `chi_L = 0`).
    """
    mu_s = torch.clamp(mu0, min=0.01, max=1.0)
    if chi_leaf is None:
        phi1 = torch.full_like(mu_s, 0.5)
        phi2 = torch.zeros_like(mu_s)
        g_mu = torch.full_like(mu_s, 0.5)
        mu_bar = torch.ones_like(mu_s)
        return g_mu, mu_bar, phi1, phi2

    chi = torch.clamp(chi_leaf, min=CHI_LEAF_MIN, max=CHI_LEAF_MAX)
    phi1 = torch.clamp(0.5 - 0.633 * chi - 0.330 * chi * chi, min=0.05, max=0.95)
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    g_mu = torch.clamp(phi1 + phi2 * mu_s, min=0.05, max=1.2)

    # Dai et al. (2004, Eq. A4c): mu_bar = (1/phi2) * [1 - (phi1/phi2)*ln(1 + phi2/phi1)]
    # Written via u = phi2/phi1 -> (1/phi1) * (u - ln(1+u)) / u^2 with 3-term Taylor as |u| < 1e-3
    u = torch.clamp(phi2 / phi1, min=-0.95, max=20.0)
    u_safe = torch.where(torch.abs(u) < 1.0e-3, torch.ones_like(u), u)
    mu_exact = (1.0 / phi1) * (u_safe - torch.log1p(u_safe)) / (u_safe * u_safe)
    mu_taylor = (1.0 / phi1) * (0.5 - u * (1.0 / 3.0 - 0.25 * u))
    mu_bar = torch.clamp(torch.where(torch.abs(u) < 1.0e-3, mu_taylor, mu_exact), min=0.4, max=2.5)
    return g_mu, mu_bar, phi1, phi2


def compute_discrete_crown_clumping(
    mu0: torch.Tensor,
    lai: torch.Tensor,
    chi_leaf: torch.Tensor | None = None,
    l_crown_min: float = CROWN_LAI_MIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map-free geometric-optical discrete-crown clumping `(omega_beam, omega_diff, f_crown)`.

    Follows Li & Strahler (1992), Nilson (1999), Ni-Meister et al. (2010), and Braghiere et al.
    (2019, GMD) for Poisson-distributed finite crowns of minimum internal leaf density `L_{c,0}`:
        f_c(mu) = 1 - exp(-L / (L_{c,0} * max(mu, 0.15))),
        P_gap(mu) = (1 - f_c(mu)) + f_c(mu) * exp(-k_0(mu) * L / f_c(mu)),
        Omega(mu, L) = -ln(P_gap(mu)) / (k_0(mu) * L).
    Eliminates the low-LAI over-absorption bias of infinite turbid-medium canopies (`Omega < 1`
    in sparse savannas/shrublands) while smoothly converging to `Omega -> 1` in closed forests
    and remaining singularity-free down to `L -> 0` via the exact analytic limit
    `Omega(L -> 0) = (1 - exp(-G * L_{c,0})) / (G * L_{c,0})`.
    """
    lc = torch.clamp(lai, min=0.0)
    mu_s = torch.clamp(mu0, min=0.01, max=1.0)
    g_mu, mu_bar, _, _ = ross_goudriaan_optics(mu_s, chi_leaf)
    k_db0 = g_mu / mu_s

    def _omega_for(k0: torch.Tensor, mu_proj: torch.Tensor, g_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_p = torch.clamp(mu_proj, min=0.15, max=1.0)
        lam_c = lc / (float(l_crown_min) * mu_p)
        f_c = -torch.expm1(-torch.clamp(lam_c, max=40.0))
        # Exact L -> 0 limit: (1 - exp(-G * L_{c,0})) / (G * L_{c,0})
        gl0 = torch.clamp(g_proj * float(l_crown_min), min=1.0e-4)
        om_lim0 = -torch.expm1(-gl0) / gl0

        tau_tot = k0 * lc
        f_c_safe = torch.clamp(f_c, min=1.0e-6)
        tau_crown = torch.clamp(tau_tot / f_c_safe, max=40.0)
        # P_gap = 1 - f_c * (1 - exp(-tau_crown)); compute -ln(P_gap) via -log1p(-abs_frac)
        abs_frac = torch.clamp(f_c * (-torch.expm1(-tau_crown)), min=0.0, max=1.0 - 1.0e-7)
        neg_ln_pgap = -torch.log1p(-abs_frac)
        om_raw = neg_ln_pgap / torch.clamp(tau_tot, min=1.0e-6)
        om = torch.where(
            lc < 1.0e-3,
            om_lim0,
            torch.where((f_c > 0.9999) | (tau_tot > 25.0), torch.ones_like(om_raw), torch.clamp(om_raw, min=0.35, max=1.0)),
        )
        return om, f_c

    omega_beam, f_crown = _omega_for(k_db0, mu_s, g_mu)
    # Hemispherical diffuse clumping evaluated at effective diffuse cosine mu_d = min(0.5 / mu_bar, 1) -> 1/(2*G_d) ~ 0.55
    mu_d = torch.clamp(0.55 * mu_bar, min=0.25, max=0.90)
    k_d0 = 1.0 / mu_bar
    g_d = k_d0 * mu_d
    omega_diff, _ = _omega_for(k_d0, mu_d, g_d)
    return omega_beam, omega_diff, f_crown


def canopy_light_absorption_depury(
    vis_beam_wm2: torch.Tensor,
    vis_diff_wm2: torch.Tensor,
    mu0: torch.Tensor,
    lai: torch.Tensor,
    chi_leaf: torch.Tensor | None = None,
) -> CanopyRadiationProfile:
    """Compute whole-canopy, sunlit, and shaded absorbed PPFD [umol photon m^-2 s^-1].

    Extends McCree (1972) & de Pury & Farquhar (1997, Plant Cell Environ., 20, 537-557) with
    Ross-Goudriaan leaf orientation (`chi_leaf`) and map-free geometric-optical discrete-crown
    clumping (`Omega(mu0, L)`; Li & Strahler, 1992; Ni-Meister et al., 2010; Braghiere et al.,
    2019), eliminating the sparse-canopy over-absorption bias while preserving exact energy
    conservation (`I_sun + I_sh == I_c`, `L_sun + L_sh == L`).
    Note: `vis_beam_wm2` and `vis_diff_wm2` are visible PAR (0.4-0.7 um) fluxes [W m^-2].
    """
    # Eqs. A1.1-A1.2 in umol photon m^-2 s^-1
    i_b0 = 4.57 * torch.clamp(vis_beam_wm2, min=0.0)
    i_d0 = 4.24 * torch.clamp(vis_diff_wm2, min=0.0)

    lc = torch.clamp(lai, min=0.0)
    has_veg = lc >= 0.01

    # Ross-Goudriaan projection G(mu0; chi_L) and discrete-crown clumping Omega(mu0, L)
    mu_s = torch.clamp(mu0, min=0.01, max=1.0)
    g_mu, mu_bar, _, _ = ross_goudriaan_optics(mu_s, chi_leaf)
    omega_b, omega_d, _ = compute_discrete_crown_clumping(mu_s, lc, chi_leaf)

    k_db = omega_b * (g_mu / mu_s)
    sqrt_1ms = math.sqrt(1.0 - LEAF_SCAT_VIS)
    k_b_prime = k_db * sqrt_1ms
    kd_prime = omega_d * (KD_PRIME_PAR / mu_bar)
    rho_h = (1.0 - sqrt_1ms) / (1.0 + sqrt_1ms)
    rho_cb = 2.0 * rho_h * k_db / (1.0 + k_db)

    # Eqs. A1.11-A1.13: Canopy-integrated beam, diffuse, and total absorbed PPFD
    i_cb = (1.0 - rho_cb) * i_b0 * (1.0 - torch.exp(-torch.clamp(k_b_prime * lc, max=50.0)))
    i_cd = (1.0 - RHO_CD_PAR) * i_d0 * (1.0 - torch.exp(-torch.clamp(kd_prime * lc, max=50.0)))
    i_c = torch.where(has_veg, i_cb + i_cd, torch.zeros_like(lc))

    # Eq. A1.15: Sunlit and shaded leaf area indices
    k_lc = k_db * lc
    l_sun_raw = torch.where(
        k_lc < 1.0e-4,
        lc * (1.0 - 0.5 * k_lc),
        (1.0 - torch.exp(-torch.clamp(k_lc, max=50.0))) / torch.clamp(k_db, min=1.0e-4),
    )
    l_sun = torch.where(has_veg, torch.minimum(torch.clamp(l_sun_raw, min=0.0), lc), torch.zeros_like(lc))
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
        * kd_prime
        * (1.0 - torch.exp(-torch.clamp((kd_prime + k_db) * lc, max=50.0)))
        / (kd_prime + k_db)
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
        kd_prime=kd_prime,
        omega_beam=omega_b,
    )


def canopy_sunlit_sw_fraction(
    sw_beam_wm2: torch.Tensor,
    sw_diff_wm2: torch.Tensor,
    mu0: torch.Tensor,
    lai: torch.Tensor,
    leaf_scat: float,
    chi_leaf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of canopy-absorbed broadband SW absorbed by sunlit foliage (clumped Ross-Goudriaan).

    Applies the exact analytic beam + scattered-beam + diffuse two-stream integrals for arbitrary
    leaf scattering coefficient `leaf_scat` (VIS = 0.15, NIR = 0.80) using Ross-Goudriaan leaf
    orientation `chi_leaf` and discrete-crown clumping `Omega(mu0, LAI)`.
    """
    lc = torch.clamp(lai, min=0.0)
    mu_s = torch.clamp(mu0, min=0.01, max=1.0)
    g_mu, mu_bar, _, _ = ross_goudriaan_optics(mu_s, chi_leaf)
    omega_b, omega_d, _ = compute_discrete_crown_clumping(mu_s, lc, chi_leaf)

    k_db = omega_b * (g_mu / mu_s)
    sqrt_1ms = math.sqrt(max(1.0 - leaf_scat, 1.0e-4))
    k_b_prime = k_db * sqrt_1ms
    k_d_prime = omega_d * ((KD_PRIME_PAR / math.sqrt(1.0 - LEAF_SCAT_VIS)) / mu_bar) * sqrt_1ms
    rho_h = (1.0 - sqrt_1ms) / (1.0 + sqrt_1ms)
    rho_cb = 2.0 * rho_h * k_db / (1.0 + k_db)
    rho_cd = 2.0 * rho_h * 0.78 / 1.78

    ib = torch.clamp(sw_beam_wm2, min=0.0)
    id_ = torch.clamp(sw_diff_wm2, min=0.0)
    k_lc = k_db * lc

    q_cb = (1.0 - rho_cb) * ib * (1.0 - torch.exp(-torch.clamp(k_b_prime * lc, max=50.0)))
    q_cd = (1.0 - rho_cd) * id_ * (1.0 - torch.exp(-torch.clamp(k_d_prime * lc, max=50.0)))
    q_c = q_cb + q_cd

    q_sun_db = ib * (1.0 - leaf_scat) * (1.0 - torch.exp(-torch.clamp(k_lc, max=50.0)))
    q_sun_sb = ib * torch.clamp(
        (1.0 - rho_cb) * k_b_prime * (1.0 - torch.exp(-torch.clamp((k_b_prime + k_db) * lc, max=50.0)))
        / (k_b_prime + k_db)
        - 0.5 * (1.0 - leaf_scat) * (1.0 - torch.exp(-torch.clamp(2.0 * k_lc, max=50.0))),
        min=0.0,
    )
    q_sun_d = (
        id_ * (1.0 - rho_cd) * k_d_prime
        * (1.0 - torch.exp(-torch.clamp((k_d_prime + k_db) * lc, max=50.0)))
        / (k_d_prime + k_db)
    )
    q_sun = torch.minimum(q_sun_db + q_sun_sb + q_sun_d, q_c)
    f_sun_area = torch.where(
        k_lc < 1.0e-4,
        torch.ones_like(lc),
        (1.0 - torch.exp(-torch.clamp(k_lc, max=50.0))) / torch.clamp(k_lc, min=1.0e-4),
    )
    return torch.where(q_c > 1.0e-4, torch.clamp(q_sun / torch.clamp(q_c, min=1.0e-4), 0.0, 1.0), f_sun_area)


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
    kd_a = acclim_rad.kd_prime if acclim_rad.kd_prime is not None else KD_PRIME_PAR
    term_d = (
        (1.0 - RHO_CD_PAR)
        * kd_a
        * acclim_rad.i_d0
        * (1.0 - torch.exp(-torch.clamp((kd_a + k_db_inst) * lc, max=50.0)))
        / (kd_a + k_db_inst)
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
    chi_leaf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Singularity-free Sellers (1985) / Dai, Dickinson, & Wang (2004, App. A) 2-stream canopy albedos.

    Supports Ross-Goudriaan leaf orientation (`chi_leaf in [-0.4, +0.5]`, Dai et al., 2004, Eq. A4)
    and geometric-optical discrete-crown clumping (`Omega_b(mu0, L), Omega_d(L)`; Ni-Meister et al.,
    2010; Braghiere et al., 2019), avoiding all angle/denominator singularities (`sigma -> 0`) while
    strictly conserving radiative energy (`albedo + transmittance + absorption == 1`).

    Returns `(alb_dir_vis, alb_dif_vis, alb_dir_nir, alb_dif_nir, trans_vis, trans_nir)`.
    """
    lc = torch.clamp(lai, min=0.0)
    mu_s = torch.clamp(mu0, min=0.02, max=1.0)
    g_mu, mu_bar, phi1, phi2 = ross_goudriaan_optics(mu_s, chi_leaf)
    omega_b, omega_d, _ = compute_discrete_crown_clumping(mu_s, lc, chi_leaf)

    k_opt = g_mu / mu_s        # Unclumped direct-beam optical depth per effective LAI
    lc_dir = omega_b * lc      # Clumped effective direct-beam LAI (Nilson, 1971; Braghiere et al., 2019)
    lc_dif = omega_d * lc      # Clumped effective diffuse LAI

    def _band_solve(omega: float, alb_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Equal leaf reflectance & transmittance (r_l = t_l = omega / 2): exact diffuse upscatter beta = 0.5
        beta = 0.5
        # Exact Sellers (1985, Eq. 15) / CLM5 (Eq. 3.16) single-scattering albedo for general Ross-Goudriaan G(mu)
        b_ss = torch.clamp(g_mu + mu_s * phi2, min=1.0e-4)
        a_ss = torch.clamp(mu_s * phi1, min=1.0e-5)
        tmp_as = 0.5 * omega * (g_mu / b_ss) * (1.0 - (a_ss / b_ss) * torch.log1p(b_ss / a_ss))
        beta_0 = torch.clamp(((1.0 + mu_bar * k_opt) * tmp_as) / torch.clamp(omega * mu_bar * k_opt, min=1.0e-5), 0.05, 0.95)

        b = 1.0 - (1.0 - beta) * omega
        c = omega * beta
        h = math.sqrt(b * b - c * c) / mu_bar
        p1 = b + mu_bar * h
        p2 = b - mu_bar * h

        # Diffuse stream uses clumped diffuse optical depth lc_dif
        s1_d = torch.exp(-torch.clamp(h * lc_dif, max=40.0))
        d1 = p1 * (1.0 - alb_g * p2 / c) / s1_d - p2 * (1.0 - alb_g * p1 / c) * s1_d
        d1_safe = torch.where(torch.abs(d1) < 1.0e-7, torch.full_like(d1, 1.0e-7), d1)
        h7 = c * (1.0 - alb_g * p2 / c) / (d1_safe * s1_d)
        h8 = -c * (1.0 - alb_g * p1 / c) * s1_d / d1_safe
        alb_dif = torch.clamp(h7 + h8, 0.0, 1.0)
        trans_dif = torch.clamp((h7 * p1 / c) * s1_d + (h8 * p2 / c) / s1_d, 0.0, 1.0)

        # Direct-beam stream uses clumped direct optical depth lc_dir
        s1 = torch.exp(-torch.clamp(h * lc_dir, max=40.0))
        s2 = torch.exp(-torch.clamp(k_opt * lc_dir, max=40.0))
        sigma_raw = (mu_bar * k_opt) ** 2 - (b * b - c * c)
        k_reg = torch.where(torch.abs(sigma_raw) < 1.0e-4, k_opt + 2.0e-4, k_opt)
        sigma = (mu_bar * k_reg) ** 2 - (b * b - c * c)

        d_coef = omega * mu_bar * k_reg * beta_0
        f_coef = omega * mu_bar * k_reg * (1.0 - beta_0)
        u1 = b - c / torch.clamp(alb_g, min=1.0e-4)
        u2 = b - c * alb_g

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
    dry_spell_days: torch.Tensor | None = None    # Consecutive growing-season dry days (< 0.5 mm/d) [days]
    ann_max_dry_days: torch.Tensor | None = None  # Running annual maximum drydown duration [days]
    ema_dmax_days: torch.Tensor | None = None     # Climatological growing-season D_max (Holtzman et al., 2024) [days]
    ema_pet_day: torch.Tensor | None = None       # 15-day EMA daily potential ET [mm d^-1]
    # Evergreen vs. deciduous leaf-habit optimality state (Kikuzawa, 1991; Givnish, 2002; Wang et al., 2023)
    f_evg: torch.Tensor | None = None             # Autonomous evergreen foliage fraction in [0, 1] [-]
    lai_evg: torch.Tensor | None = None           # Evergreen physical leaf area component [m^2 m^-2]
    lai_dec: torch.Tensor | None = None           # Deciduous physical leaf area component [m^2 m^-2]
    ema_offseason_temp: torch.Tensor | None = None # Running mean off-season temperature T_off [K]
    ann_min_temp_k: torch.Tensor | None = None    # Running annual minimum daily temperature [K]
    ema_tmin_k: torch.Tensor | None = None        # Climatological annual minimum temperature T_min,yr [K]
    ema_ann_a0_fav: torch.Tensor | None = None    # Moisture-gated annual potential GPP for leaf-habit profit [mol C m^-2 yr^-1]
    # Multi-year emergent thermal-home adaptation temperature T_home (Kumarathunge et al., 2019; Schneider et al., 2025)
    ema_temp_home: torch.Tensor | None = None     # Slow (~3-yr) climatological warm-season adaptation temperature [K]
    # CN-model nitrogen sufficiency factor phi_N in [0.15, 1.0] (Stocker & Prentice, 2024; 1.0 = N-replete / N-off)
    n_avail_fac: torch.Tensor | None = None       # 15-day EMA nitrogen availability factor [-]
    # Non-iterative optimal Ross-Goudriaan leaf orientation index chi_L in [-0.4, +0.5] (0=spherical, <0=erect, >0=flat)
    chi_leaf: torch.Tensor | None = None          # 30-day acclimated autonomous leaf inclination index [-]


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
        dry_spell_days=_f(0.0),
        ann_max_dry_days=_f(15.0),
        ema_dmax_days=_f(20.0),
        ema_pet_day=_f(3.0),
        f_evg=_f(0.5) * veg,
        lai_evg=_f(1.0) * veg,
        lai_dec=_f(1.0) * veg,
        ema_offseason_temp=_f(280.15),
        ann_min_temp_k=_f(278.15),
        ema_tmin_k=_f(278.15),
        ema_ann_a0_fav=_f(180.0),
        ema_temp_home=_f(293.15),
        n_avail_fac=_f(1.0),
        chi_leaf=_f(0.0),
    )


def compute_optimal_soil_water_beta(
    w_root: torch.Tensor,
    bio: BiosphereState,
    unfrozen_frac: torch.Tensor | None = None,
) -> torch.Tensor:
    """Non-empirical time-integrated stomatal soil-moisture stress `beta_opt` in `[0, 1]`.

    Combines Holtzman et al. (2024, Eqs. 2, 7 & Fig. 4) and Lu, Duursma, & Medlyn (2016):
        tau = 0.85 * D_max + 6.59  [days],
        beta_opt = clip(sqrt(max(w_eff - w_c, 0) / max(tau * E_pot, 0.1 * S_R)), 0, 1).
    """
    s_cap = torch.clamp(bio.s_root_cap, min=20.0)
    w_eff = w_root * unfrozen_frac if unfrozen_frac is not None else w_root
    w_c = 0.05 * s_cap
    dmax = bio.ema_dmax_days if bio.ema_dmax_days is not None else torch.full_like(w_root, 20.0)
    tau_days = 0.85 * torch.clamp(dmax, min=1.0, max=200.0) + 6.59
    pet_d = torch.clamp(
        bio.ema_pet_day if bio.ema_pet_day is not None else torch.full_like(w_root, 3.0),
        min=0.2,
    )
    w_demand = torch.maximum(tau_days * pet_d, 0.10 * s_cap)
    return torch.clamp(torch.sqrt(torch.clamp(w_eff - w_c, min=0.0) / w_demand), 0.0, 1.0)


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
    # Multi-year map-free climatological thermal-home adaptation T_home (Kumarathunge et al., 2019; Schneider et al., 2025)
    alpha_home = 1.0 / (3.0 * 365.2425)
    thome_prev = bio.ema_temp_home if bio.ema_temp_home is not None else bio.ema_temp_mid.clone()
    is_warm_grow = midday_temp > C.T_FREEZE
    bio.ema_temp_home = torch.where(
        veg & is_warm_grow,
        (1.0 - alpha_home) * thome_prev + alpha_home * midday_temp,
        thome_prev,
    )

    # 1b. Non-iterative closed-form optimal Ross-Goudriaan leaf orientation chi_L in [-0.40, +0.50]
    #     Balances canopy light distribution (erect at high LAI, flat at humid low LAI; Monsi & Saeki,
    #     1953; Hikosaka & Hirose, 1997) against aridity/VPD/insolation heat-avoidance erectophily
    #     (Ehleringer & Werk, 1986; Falster & Westoby, 2003; Pisek et al., 2013) with 30-day relaxation.
    chi_light = 0.25 * torch.tanh((2.5 - torch.clamp(bio.lai, min=0.0)) / 2.0)
    mi_ann = bio.ema_ann_precip / torch.clamp(bio.ema_ann_pet, min=10.0)
    s_dry = torch.clamp((1.2 - mi_ann) / 1.2, 0.0, 1.0)
    s_vpd = bio.ema_vpd_mid / (bio.ema_vpd_mid + 1500.0)
    s_sun = torch.clamp((bio.ema_mu0_mid - 0.35) / 0.55, 0.0, 1.0)
    chi_stress = -0.55 * (0.60 * s_dry + 0.40 * s_vpd) * (0.40 + 0.60 * s_sun)
    chi_target = torch.clamp(chi_light + chi_stress, CHI_LEAF_MIN, CHI_LEAF_MAX)
    chi_prev = bio.chi_leaf if bio.chi_leaf is not None else torch.zeros_like(chi_target)
    alpha_chi = 0.5 * alpha_ema  # ~30-day leaf-orientation acclimation timescale
    bio.chi_leaf = torch.where(veg, (1.0 - alpha_chi) * chi_prev + alpha_chi * chi_target, torch.zeros_like(chi_target))

    # 2. Compute acclimation canopy light absorption from EMA downward SW at current exposed LAI
    acclim_rad = canopy_light_absorption_depury(
        bio.ema_vis_beam, bio.ema_vis_diff, bio.ema_mu0_mid, lai_exposed, chi_leaf=bio.chi_leaf
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

    thome = bio.ema_temp_home if bio.ema_temp_home is not None else tg
    fv_c3 = torch.clamp(peaked_arrhenius(tg, tg, HA_VCMAX, is_jmax=False, t_home_k=thome), min=0.05)
    fj_c3 = torch.clamp(peaked_arrhenius(tg, tg, HA_JMAX, is_jmax=True, t_home_k=thome), min=0.05)
    fr_night = torch.clamp(heskel_respiration_scale(bio.ema_temp_night), min=0.05)

    lai_scale = torch.clamp(lai_exposed, min=1.0)
    vc25_c3_tgt = torch.minimum(torch.clamp(vcmax_opt_c3 / fv_c3, min=0.0), 300.0 * lai_scale)
    jm25_c3_tgt = torch.minimum(torch.clamp(jmax_opt_c3 / fj_c3, min=0.0), 700.0 * lai_scale)
    r25_c3_tgt = torch.minimum(torch.clamp(B_RESP_REN * vcmax_opt_c3 / fr_night, min=0.0), 10.0 * lai_scale)

    # 4. C4 Non-Rectangular-Hyperbola Optimality (Scott & Smith, 2022, Eqs. 1-25; von Caemmerer & Furbank, 1999)
    gamma_c4 = _arrhenius(GAMMA_STAR25_C4, DHA_GAMMA, tg) * (pres_sfc / 101325.0)
    kp_pa_c4 = _arrhenius(KP25_UMOL_C4 * 1.0e-6 * 101325.0, DHA_KP_C4, tg)
    kc_c4 = _arrhenius(KC25_C4, DHA_KC_C4, tg)
    ko_c4 = _arrhenius(KO25_C4, DHA_KO_C4, tg)
    km_c4 = kc_c4 * (1.0 + oi_pa / ko_c4)

    xi_c4 = torch.sqrt(BETA_C4 * kp_pa_c4 / (1.6 * eta_star))
    chi_m_c4 = torch.clamp(xi_c4 / (xi_c4 + torch.sqrt(d_pa)), 0.05, 0.92)
    cm_c4 = chi_m_c4 * ca_pa

    # C4 RuBP/PEP regeneration requires 5 mol e^- per mol CO2 (4/5 = 0.80) AND pays the bundle-sheath
    # CO2 leakage ATP overhead (1 - phi_leak, phi_leak = 0.20 -> effective quantum-yield ratio 0.64;
    # von Caemmerer, 2000; Scott & Smith, 2022, Eqs. 6 & 22; Ehleringer et al., 1997 crossover ~23-25 C at 284 ppmv).
    c4_qy_eff = 0.80 * (1.0 - PHI_LEAK_C4)
    chill_c4 = torch.sigmoid(0.35 * ((tg - C.T_FREEZE) - 12.0))
    gbs_canopy = GBS_C4 * torch.clamp(lai_exposed, min=0.1)
    cbs_c4 = cm_c4 + 250.0  # Initial bundle-sheath estimate [Pa], refined by 2 fixed-point steps
    for _ in range(2):
        m_c4 = c4_qy_eff * chill_c4 * torch.clamp((cbs_c4 - gamma_c4) / torch.clamp(cbs_c4 + 2.0 * gamma_c4, min=0.1), min=0.0)
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
    # Beer-Lambert canopy gap transmittance excluding C4 understory beneath closed forests (Luo et al., 2024; Still et al., 2003)
    herb_openness = torch.exp(-K_BEER_LAI * torch.clamp(bio.lai, min=0.0))
    f4_target = herb_openness * torch.sigmoid(8.0 * adv_c4)
    # Slow ecological turnover (~60-day smoothing)
    alpha_f4 = 0.25 * alpha_ema
    bio.f_c4 = torch.where(veg, (1.0 - alpha_f4) * bio.f_c4 + alpha_f4 * f4_target, torch.zeros_like(tg))


@dataclass
class TwoLeafAllocation:
    """Temperature-independent sunlit/shaded radiative & biochemical capacity allocation."""

    inst_rad: CanopyRadiationProfile
    vc25_c3_sun: torch.Tensor
    vc25_c3_sh: torch.Tensor
    jm25_c3_sun: torch.Tensor
    jm25_c3_sh: torch.Tensor
    r25_c3_sun: torch.Tensor
    r25_c3_sh: torch.Tensor
    vc25_c4_sun: torch.Tensor
    vc25_c4_sh: torch.Tensor
    jm25_c4_sun: torch.Tensor
    jm25_c4_sh: torch.Tensor
    vp25_c4_sun: torch.Tensor
    vp25_c4_sh: torch.Tensor
    r25_c4_sun: torch.Tensor
    r25_c4_sh: torch.Tensor
    sun_frac: torch.Tensor | None = None
    bio: BiosphereState | None = None


def prepare_two_leaf_allocation(
    bio: BiosphereState,
    vis_beam_wm2: torch.Tensor,
    vis_diff_wm2: torch.Tensor,
    mu0: torch.Tensor,
    lai_exposed: torch.Tensor,
) -> TwoLeafAllocation:
    """Precompute sunlit/shaded PAR profiles and enzyme capacities once per timestep before EB iterations."""
    acclim_rad = canopy_light_absorption_depury(
        bio.ema_vis_beam, bio.ema_vis_diff, bio.ema_mu0_mid, lai_exposed, chi_leaf=bio.chi_leaf
    )
    inst_rad = canopy_light_absorption_depury(vis_beam_wm2, vis_diff_wm2, mu0, lai_exposed, chi_leaf=bio.chi_leaf)
    k_db = inst_rad.k_db
    l_sun = inst_rad.l_sun
    kd_acclim = acclim_rad.kd_prime if acclim_rad.kd_prime is not None else KD_PRIME_PAR

    lc = torch.clamp(lai_exposed, min=0.0)
    has_veg = lc >= 0.01
    has_light = acclim_rad.i_c > 1.0e-2
    term_b = (
        (1.0 - acclim_rad.rho_cb)
        * acclim_rad.k_b_prime
        * acclim_rad.i_b0
        * (1.0 - torch.exp(-torch.clamp((acclim_rad.k_b_prime + k_db) * lc, max=50.0)))
        / (acclim_rad.k_b_prime + k_db)
    )
    term_d = (
        (1.0 - RHO_CD_PAR)
        * kd_acclim
        * acclim_rad.i_d0
        * (1.0 - torch.exp(-torch.clamp((kd_acclim + k_db) * lc, max=50.0)))
        / (kd_acclim + k_db)
    )
    sun_frac_opt = (term_b + term_d) / torch.clamp(acclim_rad.i_c, min=1.0e-2)
    sun_frac_area = l_sun / torch.clamp(lc, min=1.0e-4)
    sun_frac = torch.clamp(torch.where(has_light, sun_frac_opt, sun_frac_area), max=1.0)
    zero = torch.zeros_like(bio.vcmax25_c3)

    if can_use_elem_metal(lai_exposed):
        # On MPS Metal, two_leaf_gas_exchange_col partitions from bio.* and sun_frac in registers,
        # avoiding 14 full-grid multiply/clamp/where launches here.
        return TwoLeafAllocation(
            inst_rad=inst_rad,
            vc25_c3_sun=zero,
            vc25_c3_sh=zero,
            jm25_c3_sun=zero,
            jm25_c3_sh=zero,
            r25_c3_sun=zero,
            r25_c3_sh=zero,
            vc25_c4_sun=zero,
            vc25_c4_sh=zero,
            jm25_c4_sun=zero,
            jm25_c4_sh=zero,
            vp25_c4_sun=zero,
            vp25_c4_sh=zero,
            r25_c4_sun=zero,
            r25_c4_sh=zero,
            sun_frac=sun_frac,
            bio=bio,
        )

    def _split(x_tot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_s = torch.where(has_veg, x_tot * sun_frac, zero)
        x_h = torch.where(has_veg, torch.clamp(x_tot - x_s, min=1.0e-6), zero)
        return x_s, x_h

    vc25_c3_sun, vc25_c3_sh = _split(bio.vcmax25_c3)
    jm25_c3_sun, jm25_c3_sh = _split(bio.jmax25_c3)
    r25_c3_sun, r25_c3_sh = _split(bio.r25_c3)

    vc25_c4_sun, vc25_c4_sh = _split(bio.vcmax25_c4)
    jm25_c4_sun, jm25_c4_sh = _split(bio.jmax25_c4)
    vp25_c4_sun, vp25_c4_sh = _split(bio.vpmax25_c4)
    r25_c4_sun, r25_c4_sh = _split(bio.r25_c4)

    return TwoLeafAllocation(
        inst_rad=inst_rad,
        vc25_c3_sun=vc25_c3_sun,
        vc25_c3_sh=vc25_c3_sh,
        jm25_c3_sun=jm25_c3_sun,
        jm25_c3_sh=jm25_c3_sh,
        r25_c3_sun=r25_c3_sun,
        r25_c3_sh=r25_c3_sh,
        vc25_c4_sun=vc25_c4_sun,
        vc25_c4_sh=vc25_c4_sh,
        jm25_c4_sun=jm25_c4_sun,
        jm25_c4_sh=jm25_c4_sh,
        vp25_c4_sun=vp25_c4_sun,
        vp25_c4_sh=vp25_c4_sh,
        r25_c4_sun=r25_c4_sun,
        r25_c4_sh=r25_c4_sh,
        sun_frac=sun_frac,
        bio=bio,
    )


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
    gbw_sun: torch.Tensor | None = None,
    gbw_sh: torch.Tensor | None = None,
    precomputed_alloc: TwoLeafAllocation | None = None,
) -> TwoLeafFluxResult:
    """Evaluate sub-daily sunlit and shaded C3/C4 photosynthesis and stomatal conductance.

    1. Uses `precomputed_alloc` (or computes `prepare_two_leaf_allocation`) for sunlit/shaded
       PAR absorption and enzyme pools (Hashimoto, 2026, ATAMI, Eqs. A1.24-A1.25).
    2. Evaluates non-rectangular-hyperbola `J` (Smith et al., 2019, Eq. 9), exact Liebig
       minimum co-limitation (`min(Ac, Aj)` for C3, `min(Ac, Al, Ap)` for C4; Farquhar et al.,
       1980; von Caemmerer, 2000; preserving 1.000 coordination parity with Smith/Scott-Smith
       acclimation while two-leaf sunlit/shaded partitioning and subgrid MAML heterogeneity
       provide physical canopy-scale smoothness), MMRT respiration (Heskel et al., 2016), and
       Medlyn/Prentice/Scott-Smith optimal stomatal conductance in kinematic units [m s^-1]
       (Dai et al., 2004, Eq. 45).
    3. When finite leaf boundary-layer conductances `gbw_sun, gbw_sh` [m s^-1] are provided,
       couples the P-model to the leaf-surface microclimate `(c_s, D_s)` via 2 fixed-point passes:
         c_s = c_af - A_n * P / g_bc   (g_bc = g_bw / 1.37; Dai et al., 2004, Eq. 11),
         D_s = [g_bw / (g_bw + g_sw)] * D_leaf_af  (Bonan et al., 2018; Wang et al., 2017),
       so the returned `gsw_sun, gsw_sh` are exact stomatal conductances consistent with
       series `(g_b^-1 + g_s^-1)^-1` in the leaf energy balance without double-counting.
    """
    alloc = (
        precomputed_alloc
        if precomputed_alloc is not None
        else prepare_two_leaf_allocation(bio, vis_beam_wm2, vis_diff_wm2, mu0, lai_exposed)
    )
    inst_rad = alloc.inst_rad
    l_sun = inst_rad.l_sun
    l_sh = inst_rad.l_sh
    thome = bio.ema_temp_home if bio.ema_temp_home is not None else bio.ema_temp_mid
    if (
        can_use_elem_metal(lai_exposed)
        and alloc.sun_frac is not None
        and (alloc.bio is bio)
        and (gbw_sun is None or not gbw_sun.requires_grad)
        and (gbw_sh is None or not gbw_sh.requires_grad)
    ):
        gas_consts = (
            float(C.R_GAS_UNIVERSAL),
            float(C.T_FREEZE),
            float(C.O2_MIXING_RATIO),
            float(THETA_NRH),
            float(PHI_PSII_REF_C3),
            float(BETA_C3),
            float(BETA_C4),
            float(GAMMA_STAR25_C3),
            float(KC25_C3),
            float(KO25_C3),
            float(DHA_GAMMA),
            float(DHA_KC_C3),
            float(DHA_KO_C3),
            float(KP25_UMOL_C4),
            float(DHA_KP_C4),
            float(PHI_LEAK_C4),
            float(HA_VCMAX_BASE),
            float(HA_JMAX),
            float(HA_VPMAX),
            float(HD_ENZYME),
            float(HA_VCMAX_SLOPE),
        )
        out_5 = dispatch_two_leaf_gas_exchange_metal(
            i_sun=inst_rad.i_sun,
            i_sh=inst_rad.i_sh,
            l_sun=l_sun,
            l_sh=l_sh,
            sun_frac=alloc.sun_frac,
            vc25_c3=bio.vcmax25_c3,
            jm25_c3=bio.jmax25_c3,
            r25_c3=bio.r25_c3,
            vc25_c4=bio.vcmax25_c4,
            jm25_c4=bio.jmax25_c4,
            vp25_c4=bio.vpmax25_c4,
            r25_c4=bio.r25_c4,
            tg_mid=bio.ema_temp_mid,
            t_home=thome,
            f_c4=bio.f_c4,
            t_sun=temp_sun,
            t_sh=temp_sh,
            vpd_sun=vpd_sun,
            vpd_sh=vpd_sh,
            pres_sfc=pres_sfc,
            lai_exp=lai_exposed,
            ws_beta=water_stress_beta,
            gbw_sun=gbw_sun,
            gbw_sh=gbw_sh,
            co2_ppmv=co2_ppmv,
            consts=gas_consts,
        )
        return TwoLeafFluxResult(
            gpp=out_5[0],
            resp_leaf=out_5[1],
            an_net=out_5[2],
            gsw_sun=out_5[3],
            gsw_sh=out_5[4],
            l_sun=l_sun,
            l_sh=l_sh,
        )

    has_veg = lai_exposed >= 0.01

    ca_pa = (co2_ppmv * 1.0e-6) * pres_sfc
    oi_pa = C.O2_MIXING_RATIO * pres_sfc
    tg = bio.ema_temp_mid
    ws = torch.clamp(water_stress_beta, 0.0, 1.0)
    f4 = bio.f_c4

    # Acclimated stomatal sensitivity xi'(T_growth) & thermally acclimated quantum yield phi(T_growth)
    # held constant over the diurnal cycle (Schneider et al., 2025, Eqs. 11-12; Gan et al., 2026, Eq. 18;
    # Mengoli et al., 2022) so instantaneous leaf-temperature spikes elicit realistic stomatal VPD closure
    # and emergent T_opt vs. T_growth scaling without PFT tables.
    eta_g = water_viscosity_rel(tg)
    phi_g = phi_psii_temperature(tg)
    gamma3_g = _arrhenius(GAMMA_STAR25_C3, DHA_GAMMA, tg) * (pres_sfc / 101325.0)
    km3_g = _arrhenius(KC25_C3, DHA_KC_C3, tg) * (1.0 + oi_pa / _arrhenius(KO25_C3, DHA_KO_C3, tg))
    xi3_accl = torch.sqrt(BETA_C3 * (km3_g + gamma3_g) / (1.6 * eta_g))
    kp4_g = _arrhenius(KP25_UMOL_C4 * 1.0e-6 * 101325.0, DHA_KP_C4, tg)
    xi4_accl = torch.sqrt(BETA_C4 * kp4_g / (1.6 * eta_g))

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
        gbw_ms: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_af = torch.clamp(d_pa, min=20.0)
        fv = peaked_arrhenius(tl, tg, HA_VCMAX, is_jmax=False, t_home_k=thome)
        fj = peaked_arrhenius(tl, tg, HA_JMAX, is_jmax=True, t_home_k=thome)
        fp = peaked_arrhenius(tl, tg, HA_VPMAX, is_jmax=False, t_home_k=thome)
        fr = heskel_respiration_scale(tl)

        # Temperature-only kinetic and electron-transport terms (invariant across (c_s, D_s) passes)
        gamma3 = _arrhenius(GAMMA_STAR25_C3, DHA_GAMMA, tl) * (pres_sfc / 101325.0)
        km3 = _arrhenius(KC25_C3, DHA_KC_C3, tl) * (1.0 + oi_pa / _arrhenius(KO25_C3, DHA_KO_C3, tl))
        xi3 = xi3_accl
        vc3 = ws * vc25_3 * fv
        jm3 = ws * jm25_3 * fj
        j_3 = non_rectangular_hyperbola(phi_g * i_par, jm3, THETA_NRH)
        rd3 = r25_3 * fr

        kp4 = _arrhenius(KP25_UMOL_C4 * 1.0e-6 * 101325.0, DHA_KP_C4, tl)
        xi4 = xi4_accl
        vc4 = ws * vc25_4 * fv
        jm4 = ws * jm25_4 * fj
        vp4 = ws * vp25_4 * fp
        j_4 = non_rectangular_hyperbola(phi_g * i_par, jm4, THETA_NRH)
        al4 = 0.20 * (1.0 - PHI_LEAK_C4) * j_4 * torch.sigmoid(0.35 * ((tl - C.T_FREEZE) - 12.0))  # 5 mol e^- + BS leakage (von Caemmerer, 2000)
        ac4 = vc4  # Bundle-sheath CO2-saturated Rubisco limit
        acj4 = torch.minimum(ac4, al4)  # Exact Liebig minimum (no artificial coordination attenuation)
        rd4 = r25_4 * fr

        g0_mol = 0.002 * ws * (l_sun + l_sh > 0.01).to(tl.dtype)
        molar_vol = C.R_GAS_UNIVERSAL * tl / pres_sfc  # [m^3 mol^-1]
        if gbw_ms is not None:
            gbw_mol = torch.clamp(gbw_ms / molar_vol, min=1.0e-4)
            gbc_mol = gbw_mol / 1.37  # Diffusivity ratio (D_H2O/D_CO2)^(2/3) = 1.37 (Dai et al., 2004, Eq. 11)
            n_bl_passes = 2
        else:
            gbw_mol = None
            gbc_mol = None
            n_bl_passes = 1

        cs_pa = ca_pa
        d_s = torch.clamp(d_af, min=50.0)
        gpp_mix = torch.zeros_like(tl)
        rd_mix = (1.0 - f4) * rd3 + f4 * rd4
        gsw_mol = g0_mol

        for _bl in range(n_bl_passes):
            cs_mol = torch.clamp(cs_pa / pres_sfc, min=1.0e-5)

            # --- C3 leaf fraction at leaf surface (c_s, D_s) ---
            chi3 = torch.clamp(gamma3 / cs_pa + (1.0 - gamma3 / cs_pa) * (xi3 / (xi3 + torch.sqrt(d_s))), 0.05, 0.95)
            ci3 = chi3 * cs_pa
            ac3 = vc3 * torch.clamp((ci3 - gamma3) / (ci3 + km3), min=0.0)
            aj3 = 0.25 * j_3 * torch.clamp((ci3 - gamma3) / (ci3 + 2.0 * gamma3), min=0.0)
            gpp3 = torch.minimum(ac3, aj3)
            an3_pos = torch.clamp(gpp3 - rd3, min=0.0)
            gsw3_mol = 1.6 * (an3_pos * 1.0e-6) / torch.clamp(cs_mol * (1.0 - chi3), min=1.0e-6)

            # --- C4 leaf fraction at leaf surface (c_s, D_s) ---
            chi4 = torch.clamp(xi4 / (xi4 + torch.sqrt(d_s)), 0.05, 0.92)
            cm4 = chi4 * cs_pa
            ap4 = (1.0 - PHI_LEAK_C4) * vp4 * cm4 / torch.clamp(cm4 + kp4, min=0.5)
            gpp4 = torch.minimum(acj4, ap4)
            an4_pos = torch.clamp(gpp4 - rd4, min=0.0)
            gsw4_mol = 1.6 * (an4_pos * 1.0e-6) / torch.clamp(cs_mol * (1.0 - chi4), min=1.0e-6)

            gpp_mix = (1.0 - f4) * gpp3 + f4 * gpp4
            gsw_mol = (1.0 - f4) * gsw3_mol + f4 * gsw4_mol + g0_mol

            if gbw_mol is not None and _bl < n_bl_passes - 1:
                an_mix_mol = torch.clamp(gpp_mix - rd_mix, min=0.0) * 1.0e-6
                cs_pa = torch.clamp(ca_pa - (an_mix_mol / gbc_mol) * pres_sfc, min=0.35 * ca_pa, max=ca_pa)
                d_s = torch.clamp((gbw_mol / (gbw_mol + gsw_mol)) * d_af, min=20.0)

        # Convert mol m^-2 s^-1 to kinematic m s^-1 via ideal gas molar volume R * T / P (Dai et al., 2004, Eq. 45)
        gsw_ms = gsw_mol * molar_vol
        return gpp_mix, rd_mix, gsw_ms

    gpp_sun, rd_sun, gsw_sun_raw = _eval_fraction(
        inst_rad.i_sun,
        temp_sun,
        vpd_sun,
        alloc.vc25_c3_sun,
        alloc.jm25_c3_sun,
        alloc.r25_c3_sun,
        alloc.vc25_c4_sun,
        alloc.jm25_c4_sun,
        alloc.vp25_c4_sun,
        alloc.r25_c4_sun,
        gbw_sun,
    )
    gpp_sh, rd_sh, gsw_sh_raw = _eval_fraction(
        inst_rad.i_sh,
        temp_sh,
        vpd_sh,
        alloc.vc25_c3_sh,
        alloc.jm25_c3_sh,
        alloc.r25_c3_sh,
        alloc.vc25_c4_sh,
        alloc.jm25_c4_sh,
        alloc.vp25_c4_sh,
        alloc.r25_c4_sh,
        gbw_sh,
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
    water_stress_beta: torch.Tensor | None = None,
    is_vegetated_land: torch.Tensor | None = None,
    daily_t_min_k: torch.Tensor | None = None,
) -> None:
    """Update prognostic EEO LAI, evergreen/deciduous habit `f_evg`, `S_R`, and canopy height `h_c` once per day.

    References
    ----------
    - Cai et al. (2025, Commun. Earth Environ., 6, 19, Eqs. 1-2 & 12):
      Energy-limited `fAPAR_c` and water-limited `fAPAR_w` combined via log-sum-exp (`K = 10`).
    - Zhou et al. (2025, Global Change Biol., 31, e70125, Eqs. 6 & 15-20):
      Aridity-dependent `f_0(AI)` and Lambert-`W_0` steady-state `L_s` with `alpha = 0.067` EMA.
    - Kikuzawa (1991, Am. Nat., 138, 1250-1263), Givnish (2002, Silva Fennica, 36, 703-743),
      Sakai & Larcher (1987), & Wang et al. (2023, Science Advances, 9, eadd5667):
      Autonomous evergreen vs. deciduous leaf-habit dominance `f_evg` from leaf-longevity carbon
      profit optimization with temperature-dependent off-season respiration and winter frost damage.
    - Gao et al. (2014, Geophys. Res. Lett., 41, 7916-7923):
      Cumulative dry-season water deficit `(PET * fAPAR_max - P)^+` sizing `S_R`.
    - Kempes et al. (2011, PLoS ONE, 6, e20551, Eqs. 2-5):
      Allometric vascular balance between available water flow `Q_p` and evaporative/metabolic scaling.
    """
    veg = is_vegetated_land if is_vegetated_land is not None else torch.ones_like(daily_temp_k, dtype=torch.bool)
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
    # Potential daily GPP A_0 with universal vascular-growth cold sink gate (Körner, 2012; ~5 degC base)
    cold_gate = torch.sigmoid(0.5 * ((daily_temp_k - C.T_FREEZE) - 5.0))
    a0_day = lue_c3 * torch.clamp(daily_ppfd_mol_m2, min=0.0) * cold_gate

    # Update growing-season drydown length D_max & daily PET for Holtzman et al. (2024) tau
    is_dry = (daily_precip_mm < 0.5) & (daily_temp_k > C.T_FREEZE)
    dry_cur = bio.dry_spell_days if bio.dry_spell_days is not None else torch.zeros_like(daily_precip_mm)
    ann_max = bio.ann_max_dry_days if bio.ann_max_dry_days is not None else torch.full_like(daily_precip_mm, 15.0)
    ema_dmax = bio.ema_dmax_days if bio.ema_dmax_days is not None else torch.full_like(daily_precip_mm, 20.0)
    ema_pet = bio.ema_pet_day if bio.ema_pet_day is not None else torch.full_like(daily_precip_mm, 3.0)

    dry_next = torch.where(is_dry, dry_cur + 1.0, torch.zeros_like(dry_cur))
    ann_max_next = torch.maximum(ann_max * (1.0 - alpha_yr), dry_next)
    bio.dry_spell_days = torch.where(veg, dry_next, torch.zeros_like(dry_next))
    bio.ann_max_dry_days = torch.where(veg, ann_max_next, torch.zeros_like(ann_max_next))
    bio.ema_dmax_days = torch.where(
        veg,
        (1.0 - alpha_yr) * ema_dmax + alpha_yr * torch.clamp(ann_max_next, min=1.0, max=200.0),
        torch.full_like(ema_dmax, 20.0),
    )
    bio.ema_pet_day = torch.where(
        veg,
        (1.0 - ALPHA_EMA_ACCLIM) * ema_pet + ALPHA_EMA_ACCLIM * torch.clamp(daily_pet_mm, min=0.1, max=20.0),
        torch.full_like(ema_pet, 3.0),
    )

    # Update annual running accumulators (favorable season includes both thermal & non-drought conditions)
    bio.ema_ann_precip = (1.0 - alpha_yr) * bio.ema_ann_precip + alpha_yr * (daily_precip_mm * 365.2425)
    bio.ema_ann_pet = (1.0 - alpha_yr) * bio.ema_ann_pet + alpha_yr * (daily_pet_mm * 365.2425)
    bio.ema_ann_a0 = (1.0 - alpha_yr) * bio.ema_ann_a0 + alpha_yr * (a0_day * 365.2425)
    ws_b = water_stress_beta if water_stress_beta is not None else torch.ones_like(daily_temp_k)
    a0_day_fav = a0_day * torch.clamp(ws_b, 0.0, 1.0)
    a0_fav_prev = bio.ema_ann_a0_fav if bio.ema_ann_a0_fav is not None else bio.ema_ann_a0.clone()
    bio.ema_ann_a0_fav = (1.0 - alpha_yr) * a0_fav_prev + alpha_yr * (a0_day_fav * 365.2425)
    is_fav = (daily_temp_k > C.T_FREEZE + 5.0) & (ws_b > 0.20)
    bio.ema_gsl_frac = (1.0 - alpha_yr) * bio.ema_gsl_frac + alpha_yr * is_fav.to(daily_temp_k.dtype)

    # Track off-season temperature T_off and climatological annual minimum temperature T_min,yr
    t_min_in = daily_t_min_k if daily_t_min_k is not None else daily_temp_k
    t_off_prev = bio.ema_offseason_temp if bio.ema_offseason_temp is not None else daily_temp_k.clone()
    alpha_off = torch.where(~is_fav, torch.full_like(daily_temp_k, 1.0 / 30.0), torch.full_like(daily_temp_k, alpha_yr))
    bio.ema_offseason_temp = (1.0 - alpha_off) * t_off_prev + alpha_off * daily_temp_k

    ann_tmin = bio.ann_min_temp_k if bio.ann_min_temp_k is not None else t_min_in.clone()
    ema_tmin = bio.ema_tmin_k if bio.ema_tmin_k is not None else t_min_in.clone()
    ann_tmin_next = torch.minimum(ann_tmin + 0.25 * alpha_yr * torch.clamp(t_min_in - ann_tmin, min=0.0), t_min_in)
    ema_tmin_next = (1.0 - 2.0 * alpha_yr) * ema_tmin + (2.0 * alpha_yr) * ann_tmin_next
    bio.ann_min_temp_k = ann_tmin_next
    bio.ema_tmin_k = ema_tmin_next

    # 2. Annual hydroclimatic water limit fAPAR_w & baseline canopy assimilation a_leaf
    # Per Zhou et al. (2025, Sec. 2.1.2), f_0 = 0.65 in energy-limited climates (AI <= 1.9) and declines
    # as AI increases beyond 1.9 into water-limited regimes.
    ai = torch.clamp(bio.ema_ann_pet / torch.clamp(bio.ema_ann_precip, min=10.0), 0.05, 50.0)
    ai_arid = torch.clamp(ai, min=1.9)
    f0 = 0.65 * torch.exp(-0.604169 * (torch.log(ai_arid / 1.9) ** 2))  # Zhou et al. (2025, Eq. 6)
    p_mol_yr = bio.ema_ann_precip * (1000.0 / 18.01528)            # 1 kg H2O = 55.508 mol H2O
    a0_yr = torch.clamp(bio.ema_ann_a0, min=1.0e-2)

    k_lse = 10.0

    def _lse_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mp = torch.minimum(x, y)
        return mp - (1.0 / k_lse) * torch.log(torch.exp(-k_lse * (x - mp)) + torch.exp(-k_lse * (y - mp)))

    fapar_w = f0 * p_mol_yr * ca_pa * (1.0 - chi3) / torch.clamp(1.6 * d_s * a0_yr, min=1.0)
    fapar_c0 = 1.0 - Z_FOLIAGE_COST / (K_BEER_LAI * a0_yr)
    fapar_max0 = torch.clamp(_lse_min(fapar_w, fapar_c0), 0.0, 0.96)
    lai_max0 = -torch.log(1.0 - fapar_max0) / K_BEER_LAI

    # 2b. Autonomous Evergreen vs. Deciduous Leaf-Habit Optimality (Kikuzawa, 1991; Givnish, 2002;
    #     Sakai & Larcher, 1987; Wang et al., 2023, Science Advances, 9, eadd5667)
    f_gsl = torch.clamp(bio.ema_gsl_frac, 0.05, 1.0)
    a0_fav_yr = torch.clamp(bio.ema_ann_a0_fav, min=1.0e-2)
    a_leaf = torch.clamp(a0_fav_yr * fapar_max0 / torch.clamp(lai_max0, min=0.50), min=1.0)
    # Cold-hardening & nutrient-conservation structural LMA investment & efficiency trade-off
    # (Wright et al., 2004 LES; Givnish, 2002; Kikuzawa, 1991; Wang et al., 2023):
    # Under N scarcity (phi_N < 1), plants invest in thicker, higher-LMA leaves that extend
    # optimal evergreen lifespan tau_evg and penalize annual deciduous leaf-N turnover loss.
    f_cold = torch.sigmoid((C.T_FREEZE - ema_tmin_next) / 5.0)
    n_fac = torch.clamp(bio.n_avail_fac, 0.25, 1.0) if bio.n_avail_fac is not None else None
    n_scarcity = (1.0 - n_fac) if n_fac is not None else torch.zeros_like(f_cold)
    c_dec = Z_FOLIAGE_COST
    c_dec_eff = c_dec * (1.0 + 0.80 * n_scarcity)
    c_evg = (EVG_COST_RATIO_BASE + EVG_COST_RATIO_COLD * f_cold + 0.80 * n_scarcity) * c_dec
    eta_evg = torch.clamp(1.0 - EVG_COLD_EFF_PENALTY * f_cold, min=0.40)
    # Kikuzawa (1991, Eq. 5) optimal evergreen leaf lifespan tau_evg [yr] (lengthens under cold or N scarcity)
    tau_evg = torch.clamp(torch.sqrt(2.0 * LEAF_AGING_B_YR * c_evg / torch.clamp(eta_evg * a_leaf, min=1.0)), 1.0, 8.0)
    # Temperature-dependent off-season maintenance/stress cost (Q10 ~ 2.0, Heskel et al., 2016)
    f_resp_off = torch.exp(torch.clamp(0.0693 * (bio.ema_offseason_temp - 298.15), min=-6.0, max=3.0))
    # Extreme continental winter frost-damage penalty (Sakai & Larcher, 1987; Givnish, 2002; Larix belt)
    frost_pen = EVG_FROST_COST * torch.sigmoid((EVG_FROST_CRIT_K - ema_tmin_next) / 4.0)

    b_dec = max(LEAF_AGING_B_YR / 2.5, 1.0)
    flush_frac = torch.clamp(1.0 - DECID_FLUSH_DAYS / torch.clamp(365.2425 * f_gsl, min=15.0), min=0.0)
    g_dec = flush_frac * (1.0 - f_gsl / (2.0 * b_dec)) * a_leaf - c_dec_eff
    g_evg = (
        eta_evg * (1.0 - tau_evg / (2.0 * LEAF_AGING_B_YR)) * a_leaf
        - (c_evg / tau_evg)
        - EVG_OFFSEASON_RESP25 * f_resp_off * (1.0 - f_gsl)
        - frost_pen
    )
    f_evg_target = torch.sigmoid(4.0 * (g_evg - g_dec) / max(c_dec, 1.0))
    f_evg_prev = bio.f_evg if bio.f_evg is not None else (0.5 * veg.to(daily_temp_k.dtype))
    f_evg_new = torch.where(veg, (1.0 - alpha_yr) * f_evg_prev + alpha_yr * f_evg_target, torch.zeros_like(f_evg_target))
    bio.f_evg = torch.clamp(f_evg_new, 0.0, 1.0)

    # Effective amortized annual foliage cost z_eff lowers carbon barrier for long-lived evergreen canopies,
    # floored at 0.55 * c_dec, elevated in permafrost/alpine short-GSL soils (Wang et al., 2023; Körner, 2012),
    # and modulated by CN-model nitrogen scarcity (Stocker & Prentice, 2024; n_cost_mult = 1.0 when N-replete).
    cold_soil_cost = 1.0 + 1.5 * torch.sigmoid(20.0 * (0.28 - f_gsl))
    n_cost_mult = 1.0 + 0.80 * n_scarcity
    z_eff = (
        torch.clamp((1.0 - bio.f_evg) * c_dec + bio.f_evg * (c_evg / tau_evg), min=0.55 * c_dec)
        * cold_soil_cost
        * n_cost_mult
    )
    fapar_c = 1.0 - z_eff / (K_BEER_LAI * a0_yr)
    fapar_max = torch.clamp(_lse_min(fapar_w, fapar_c), 0.0, 0.96)
    lai_max = -torch.log(1.0 - fapar_max) / K_BEER_LAI

    # 3. Dual-timescale prognostic LAI: evergreen persistence + Zhou et al. (2025) deciduous phenology
    gsl_days = torch.clamp(f_gsl * 365.2425, min=15.0)
    m_alloc = SIGMA_ZHOU * gsl_days * lai_max / torch.clamp(a0_fav_yr * fapar_max, min=0.1)
    mu_lai = m_alloc * a0_day_fav
    u_lai = K_BEER_LAI * mu_lai
    arg_w = -u_lai * torch.exp(-torch.clamp(u_lai, max=40.0))
    ls_raw = mu_lai + lambert_w0(arg_w) / K_BEER_LAI
    l_steady = torch.clamp(torch.minimum(ls_raw, lai_max), min=0.0)

    lai_evg_prev = bio.lai_evg if bio.lai_evg is not None else (bio.f_evg * bio.lai)
    lai_dec_prev = bio.lai_dec if bio.lai_dec is not None else ((1.0 - bio.f_evg) * bio.lai)

    lai_evg_target = bio.f_evg * lai_max
    alpha_evg_turn = 1.0 / (365.2425 * tau_evg)
    # Multi-year evergreen retention in winter; faster crown build-up on growth, drought dieback above lai_max,
    # and acute winter frost-kill mortality when T_min falls below EVG_FROST_CRIT_K (-40 degC)
    alpha_frost = 0.025 * torch.sigmoid((EVG_FROST_CRIT_K - t_min_in) / 4.0)
    alpha_evg_up = torch.clamp(4.0 * alpha_evg_turn, max=ALPHA_EMA_ACCLIM)
    alpha_evg_down = torch.maximum(alpha_evg_turn, alpha_frost)
    alpha_evg_eff = torch.where(
        lai_evg_prev > lai_max,
        torch.clamp(alpha_evg_down, min=0.015),
        torch.where(lai_evg_target > lai_evg_prev, alpha_evg_up, alpha_evg_down),
    )
    lai_evg_new = (1.0 - alpha_evg_eff) * lai_evg_prev + alpha_evg_eff * lai_evg_target
    lai_dec_target = (1.0 - bio.f_evg) * l_steady
    lai_dec_new = (1.0 - ALPHA_EMA_ACCLIM) * lai_dec_prev + ALPHA_EMA_ACCLIM * lai_dec_target

    bio.lai_evg = torch.where(veg, torch.clamp(lai_evg_new, 0.0, 8.0), torch.zeros_like(lai_evg_new))
    bio.lai_dec = torch.where(veg, torch.clamp(lai_dec_new, 0.0, 8.0), torch.zeros_like(lai_dec_new))
    bio.lai = torch.where(veg, torch.clamp(bio.lai_evg + bio.lai_dec, 0.0, 8.0), torch.zeros_like(lai_max))

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
    #    modulated by universal alpine/arctic cambial thermal treeline limitation (Körner, 2012, Ch. 3;
    #    Paulsen & Körner, 2014: >= ~94 days above 5 degC for woody tree-stem xylogenesis).
    #    Cap moisture_idx at 1.2 so cold low-PET bogs/plateaus cannot spuriously grow tall forests.
    moisture_idx = torch.clamp(bio.ema_ann_precip / torch.clamp(bio.ema_ann_pet, min=50.0), 0.0, 1.2)
    prod_idx = torch.clamp((fapar_max * a0_yr) / 180.0, 0.0, 2.0)
    # Growing-season length (> 5 C) thermal treeline gate (Körner, 2012: treeline at ~90-100 days -> f_gsl ~ 0.30)
    treeline_gsl = torch.sigmoid(30.0 * (f_gsl - 0.30))
    hc_target = torch.clamp(
        34.0 * torch.pow(moisture_idx, 0.85) * torch.pow(prod_idx, 0.75) * treeline_gsl,
        0.05,
        40.0,
    )
    bio.h_canopy = torch.where(
        veg,
        (1.0 - alpha_yr) * bio.h_canopy + alpha_yr * hc_target,
        torch.zeros_like(hc_target),
    )
