"""First-Principles Quasi-Electrostatic Red Sprite & Sprite-Halo Model for AOBA (TLE Module).

Computes mesospheric (38-90 km) Transient Luminous Events (TLEs: sprite halos, prompt downward
positive/negative sprite streamers, sub-breakdown associative-detachment delayed sprites,
N2 1st/2nd Positive optical emissions with altitude-dependent collisional quenching, and
mesospheric NOx plasma-chemistry production) driven directly by AOBA's cloud-to-ground (CG)
lightning discharges and Wait-Spies solar-zenith-angle D-region ionospheric conductivity.

Zero geographic or land/ocean lookup maps are used; the nighttime occurrence preference,
mesoscale convective system (MCS) and winter-storm +CG hotspots, red-head (~65-85 km) to
blue-tendril (~40-55 km) optical transition, and downward stopping altitudes emerge purely from
electrodynamics, Townsend density similarity, and atmospheric thermodynamics.

Fully vectorized on GPU (MPS / CUDA) and CPU with zero per-cell or per-level Python loops.

References
----------
- Pasko, V. P., Inan, U. S., Bell, T. F., & Taranenko, Y. N. (1997): "Sprites produced by
  quasi-electrostatic heating and ionization in the lower ionosphere", J. Geophys. Res. Space
  Physics, 102(A3), 4529-4561.
- Sentman, D. D., Stenbaek-Nielsen, H. C., McHarg, M. G., & Morrill, J. S. (2008): "Plasma
  chemistry of sprite streamers", J. Geophys. Res. Atmospheres, 113, D11112 (and Correction, 2008).
- Luque, A., & Ebert, U. (2009): "Emergence of sprite streamers from screening-ionization waves
  in the lower ionosphere", Nature Geoscience, 2(11), 757-760.
- Ebert, U., Nijdam, S., Li, C., Luque, A., Briels, T., & van Veldhuizen, E. (2010): "Review of
  recent results on streamer discharges and discussion of their relevance for sprites and
  lightning", J. Geophys. Res. Space Physics, 115, A00E43.
- Luque, A., & Gordillo-Vázquez, F. J. (2012): "Mesospheric electric breakdown and delayed
  sprite ignition caused by electron detachment", Nature Geoscience, 5(1), 22-25.
- Qin, J., Pasko, V. P., McHarg, M. G., & Stenbaek-Nielsen, H. C. (2014): "Investigation of
  streamer initiation in sprite-halo events", Nat. Commun. / Penn State CSSL.

AOBA Formulation Reference: Sec. 5.6.3, Eqs. (5.24)-(5.27) (Quasi-Electrostatic Red Sprites & Halos).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.runtime import maybe_cuda_compile


# Fundamental physical constants for mesospheric electrodynamics & optical emission
EPSILON_0: float = 8.854187817e-12       # Vacuum permittivity [F m^-1]
Q_ELECTRON: float = 1.602176634e-19      # Elementary charge [C]
M_AIR_KG: float = 4.810e-26              # Mean dry-air molecular mass (~28.97 amu) [kg]
N_AIR_STP: float = 2.688e25              # Loschmidt air number density at STP [m^-3]
E_K_STP: float = 3.2e6                   # Conventional breakdown field at STP (120 Td; Pasko 1997) [V m^-1]
E_ST_POS_STP: float = 4.4e5              # Positive streamer stability field at STP (Ebert et al., 2010) [V m^-1]
E_ST_NEG_STP: float = 1.25e6             # Negative streamer stability field at STP (Ebert et al., 2010) [V m^-1]
E_DELAY_FRAC: float = 0.70               # Associative detachment sub-breakdown threshold fraction (Luque & GV 2012) [-]
E_HALO_FRAC: float = 0.50                # Diffuse sprite-halo excitation threshold fraction (Pasko 1997; Qin 2014) [-]
TAU_DETACH_MIN_S: float = 1.5e-3         # Minimum dielectric relaxation time for delayed O- + N2 detachment at 74-80 km [s]
TAU_DISCHARGE_S: float = 1.0e-3          # Characteristic impulse CG charge removal timescale (Pasko 1997) [s]
MU_E_STP: float = 0.045                  # Effective low-field electron mobility scaled at STP [m^2 V^-1 s^-1]

# Wait & Spies (1964) / Pasko et al. (1997) D-region ionosphere parameters
H_PRIME_NIGHT_KM: float = 85.0           # Nighttime Wait-Spies reference reflection height [km]
H_PRIME_DAY_KM: float = 70.0             # Daytime Wait-Spies reference reflection height [km]
BETA_NIGHT_KM1: float = 0.50             # Nighttime exponential electron-density slope [km^-1]
BETA_DAY_KM1: float = 0.30               # Daytime exponential electron-density slope [km^-1]

# Optical quenching constants (Pasko et al., 1997, Sec. 2.4; Vallance Jones, 1974)
# N2 1st Positive (1PN2, crimson red 650-750 nm): Einstein A = 1.7e5 s^-1, quenched below ~53 km
A_EINSTEIN_1PN2: float = 1.7e5           # [s^-1]
KQ_N2_1PN2: float = 1.0e-17              # Quenching rate coefficient with N2 [m^3 s^-1]
# N2 2nd Positive (2PN2, violet/blue 337 nm): Einstein A = 2.0e7 s^-1, quenched below ~30 km
A_EINSTEIN_2PN2: float = 2.0e7           # [s^-1]
KQ_O2_2PN2: float = 3.0e-16              # Quenching rate coefficient with O2 [m^3 s^-1]
F_N2_AIR: float = 0.7808                 # Atmospheric N2 mole fraction [-]
F_O2_AIR: float = 0.2095                 # Atmospheric O2 mole fraction [-]

# Mesospheric plasma chemistry yield (Sentman et al., 2008; Enell et al., 2008; Winkler & Notholt, 2015)
NOX_MOLECULES_PER_SPRITE: float = 5.0e22        # Typical NOx production per sprite event [molecules sprite^-1]
NOX_MOLECULES_PER_GIGANTIC_JET: float = 5.0e23  # Gigantic jet cloud-to-ionosphere NOx yield [molecules GJ^-1]

# ELVES (Inan et al., 1991; Barrington-Leigh & Inan, 1999; Chen et al., 2008) &
# Blue Jet / Gigantic Jet (Pasko et al., 2002; Krehbiel et al., 2008) physical constants
MU_0_VAC: float = 4.0e-7 * math.pi              # Vacuum permeability [H m^-1]
V_RETURN_STROKE_MS: float = 1.5e8               # Characteristic CG return-stroke velocity [m s^-1]
E_ELVES_FRAC: float = 0.45                      # EMP heating/N2-1PN2 threshold fraction of E_k at D-region base [-]
ELVES_RING_RANGE_M: float = 1.5e5               # Peak EMP radiation slant distance R (~55 deg zenith angle) [m]
ELVES_SIN_THETA: float = 0.82                   # Radiation dipole sin(theta) at peak EMP ring [-]
I_PEAK_SIGMA_LN: float = 0.72                   # Log-normal spread of CG peak current I_peak (Orville & Huffines 2001) [-]

# Standard 26-level mesospheric evaluation grid (38..88 km, dz = 2 km)
N_MESO_LEVELS: int = 26
Z_MESO_MIN_KM: float = 38.0
Z_MESO_DZ_KM: float = 2.0


@dataclass
class RedSpriteDiagnostics:
    """Global host-cell diagnostics for mesospheric TLEs (sprites, halos, ELVES, blue/gigantic jets) & NOx."""

    sprite_rate_cell: torch.Tensor         # [N_cells] total sprite rate (prompt + delayed) [sprites m^-2 s^-1]
    sprite_prompt_rate_cell: torch.Tensor  # [N_cells] prompt breakdown sprite rate (E >= E_k) [sprites m^-2 s^-1]
    sprite_delayed_rate_cell: torch.Tensor # [N_cells] delayed detachment sprite rate (0.7 E_k <= E < E_k) [sprites m^-2 s^-1]
    halo_rate_cell: torch.Tensor           # [N_cells] sprite halo rate (E >= 0.5 E_k) [halos m^-2 s^-1]
    cmc_median_ckm_cell: torch.Tensor      # [N_cells] conditional median impulse CG charge moment change [C km]
    e_over_ek_max_cell: torch.Tensor       # [N_cells] peak mesospheric E_peak / E_k for 99th-pct CG flash [-]
    red_1pn2_kr_cell: torch.Tensor         # [N_cells] peak N2 1st Positive (crimson red) brightness [kR]
    blue_2pn2_kr_cell: torch.Tensor        # [N_cells] peak N2 2nd Positive (blue tendril) brightness [kR]
    z_init_km_cell: torch.Tensor           # [N_cells] sprite initiation altitude [km]
    z_stop_km_cell: torch.Tensor           # [N_cells] downward streamer stopping altitude [km]
    sprite_nox_mol_m2s: torch.Tensor       # [N_cells] column mesospheric NOx production (sprites + GJs) [molecules m^-2 s^-1]
    elves_rate_cell: torch.Tensor | None = None          # [N_cells] EMP ELVES occurrence rate [elves m^-2 s^-1]
    blue_jet_rate_cell: torch.Tensor | None = None       # [N_cells] stratospheric Blue Jet rate [jets m^-2 s^-1]
    gigantic_jet_rate_cell: torch.Tensor | None = None   # [N_cells] cloud-to-ionosphere Gigantic Jet rate [jets m^-2 s^-1]
    blue_jet_top_km_cell: torch.Tensor | None = None     # [N_cells] representative Blue Jet terminal altitude [km]


def _dielectric_screening_factor(r_tau: torch.Tensor) -> torch.Tensor:
    """Exact peak-field dielectric screening factor S(r) = exp(-ln(r) / (r - 1)) for r = tau_sigma / tau_f.

    Solves dE/dt + E/tau_sigma = (E_QE / tau_f) * exp(-t / tau_f) analytically without singularity at r = 1
    (where S(1) = e^-1), approaching 1 at night (r >> 1) and r -> 0 in daytime D-region (r << 1).
    """
    r_safe = torch.clamp(r_tau, min=1.0e-6, max=1.0e6)
    u = torch.log(r_safe)
    # Near u = 0, u / expm1(u) = 1 - u/2 + u^2/12
    small = u.abs() < 1.0e-3
    ratio = torch.where(
        small,
        1.0 - 0.5 * u + (u * u) / 12.0,
        u / torch.where(small, torch.ones_like(u), torch.expm1(u)),
    )
    return torch.exp(-ratio)


# AOBA Formulation Eq. (5.24)
def compute_mesospheric_conductivity_and_density(
    z_meso_km: torch.Tensor,     # [Nz_m, 1] [km]
    mu0_cell: torch.Tensor,      # [N_cells] cosine of solar zenith angle [-]
    rho_top: torch.Tensor,       # [N_cells] air density at top model layer [kg m^-3]
    z_top_km: torch.Tensor,      # [N_cells] altitude of top model layer [km]
    temp_top: torch.Tensor,      # [N_cells] temperature at top model layer [K]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mesospheric air number density N(z) [m^-3], conductivity sigma(z) [S m^-1], & tau_sigma [s].

    Uses smooth solar-zenith-angle Wait & Spies (1964) / Pasko et al. (1997) D-region electron
    profiles plus ambient ion conductivity, and hydrostatic continuation from AOBA's upper levels.
    """
    # Smooth twilight transition between nighttime (mu0 <= -0.10, solar zenith > 96 deg) & daytime (mu0 >= 0.10)
    day_fac = torch.clamp((mu0_cell + 0.10) / 0.20, 0.0, 1.0).unsqueeze(0)  # [1, nc]
    h_prime = H_PRIME_NIGHT_KM * (1.0 - day_fac) + H_PRIME_DAY_KM * day_fac
    beta = BETA_NIGHT_KM1 * (1.0 - day_fac) + BETA_DAY_KM1 * day_fac

    # Hydrostatic mesospheric continuation incorporating mesospheric cooling (~265 K at 50 km -> ~195 K at 85 km)
    # Effective mesospheric scale height H_meso = R_d * T_eff / g (~6.1-6.4 km, matching US-76)
    z_top_b = z_top_km.unsqueeze(0)  # [1, nc]
    dz_above = z_meso_km - z_top_b
    t_eff = torch.clamp(temp_top.unsqueeze(0) - 1.8 * torch.clamp(z_meso_km - 50.0, min=0.0, max=35.0), min=185.0, max=265.0)
    t_mean = 0.5 * (torch.clamp(temp_top.unsqueeze(0), min=190.0, max=265.0) + t_eff)
    h_scale_km = (C.R_D * t_mean / C.GRAVITY) * 1.0e-3  # [Nz_m, nc]
    rho_meso = torch.clamp(rho_top.unsqueeze(0), min=1.0e-8) * torch.exp(-dz_above / h_scale_km)
    n_air = torch.clamp(rho_meso / M_AIR_KG, min=1.0e17, max=N_AIR_STP)  # [Nz_m, nc]

    # Wait & Spies (1964) D-region electron density Ne(z) [m^-3]
    # Ne(z) = 1.43e13 * exp(-0.15 * h') * exp((beta - 0.15) * (z - h'))
    arg_ne = torch.clamp((beta - 0.15) * (z_meso_km - h_prime), min=-40.0, max=25.0)
    n_e = 1.43e13 * torch.exp(-0.15 * h_prime + arg_ne)
    n_e = torch.clamp(n_e, min=1.0e3, max=1.0e11)

    # Electron mobility scales inversely with neutral air density (Townsend similarity)
    mu_e = MU_E_STP * (N_AIR_STP / n_air)
    sigma_e = Q_ELECTRON * n_e * mu_e
    # Ambient stratospheric/mesospheric cosmic-ray ion conductivity (Pasko et al., 1997, Fig. 6b):
    # follows Dejnakarintra & Park (1974) exp(z/6 km) up to 55 km, rolling over above 55 km so electrons dominate >65 km
    z_ion_eff = torch.clamp(z_meso_km, max=55.0) / 6.0 + torch.clamp(z_meso_km - 55.0, min=0.0) / 16.0
    sigma_ion = 5.0e-14 * torch.exp(z_ion_eff)
    sigma_tot = sigma_e + sigma_ion
    tau_sigma = EPSILON_0 / torch.clamp(sigma_tot, min=1.0e-16)
    return n_air, sigma_tot, tau_sigma


# AOBA Formulation Eqs. (5.25)-(5.27)
def compute_sprite_vertical_profile(
    cmc_ckm: float,
    mu0: float = -0.3,
    z_q_km: float = 10.0,
    positive_cg: bool = True,
    tau_f_s: float = TAU_DISCHARGE_S,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Evaluate 1D vertical mesospheric (38-88 km) profiles for a single CG flash of given CMC [C km].

    Returns dictionary with altitude grid `z_km`, `e_over_ek`, `tau_sigma_s`, `red_1pn2_kr`,
    `blue_2pn2_kr`, `z_init_km`, `z_stop_km`, `is_halo`, `is_prompt_sprite`, `is_delayed_sprite`.
    """
    dev = torch.device(device)
    z_km = (Z_MESO_MIN_KM + Z_MESO_DZ_KM * torch.arange(N_MESO_LEVELS, device=dev, dtype=torch.float32)).unsqueeze(1)
    mu0_t = torch.tensor([float(mu0)], device=dev, dtype=torch.float32)
    # Standard US-76 reference anchor at 40 km: rho ~ 4.0e-3 kg m^-3, T ~ 250 K
    rho_40 = torch.tensor([4.0e-3], device=dev, dtype=torch.float32)
    z_40 = torch.tensor([40.0], device=dev, dtype=torch.float32)
    t_40 = torch.tensor([250.0], device=dev, dtype=torch.float32)

    n_air, sigma_tot, tau_sigma = compute_mesospheric_conductivity_and_density(z_km, mu0_t, rho_40, z_40, t_40)
    z_m = z_km * 1.0e3
    zq_m = max(float(z_q_km), 3.0) * 1.0e3
    dm_q_cm = abs(float(cmc_ckm)) * 1.0e3  # [C m]

    # Exact conducting-ground image dipole QE field E_QE(z) [V m^-1]
    e_qe = (dm_q_cm / (math.pi * EPSILON_0)) * (z_m / torch.clamp(z_m * z_m - zq_m * zq_m, min=1.0e6) ** 2)
    screen = _dielectric_screening_factor(tau_sigma / max(float(tau_f_s), 1.0e-4))
    e_peak = e_qe * screen

    n_ratio = n_air / N_AIR_STP
    e_k = E_K_STP * n_ratio
    e_over_ek = e_peak / torch.clamp(e_k, min=1.0e-6)

    # Breakdown, delayed detachment (Luque & GV 2012), and halo thresholds
    prompt_mask = e_over_ek >= 1.0
    delay_mask = (e_over_ek >= E_DELAY_FRAC) & (~prompt_mask) & (tau_sigma >= TAU_DETACH_MIN_S)
    sprite_init_mask = prompt_mask | delay_mask
    halo_mask = e_over_ek >= E_HALO_FRAC

    # Streamer downward propagation stopping altitude via Ebert et al. (2010) stability field E_st^+/-
    e_st_stp = E_ST_POS_STP if positive_cg else E_ST_NEG_STP
    e_st = e_st_stp * n_ratio
    prop_ok = e_peak >= e_st

    has_sprite = sprite_init_mask.any(dim=0)
    k_peak = e_over_ek.argmax(dim=0)
    z_init_km = torch.where(has_sprite, z_km[k_peak, 0], torch.zeros_like(mu0_t))
    # Lowest altitude below initiation where E_peak >= E_st (downward positive/negative streamer stopping height)
    below_init = z_km <= z_init_km.unsqueeze(0)
    active_streamer = has_sprite.unsqueeze(0) & below_init & prop_ok
    z_stop_km = torch.where(
        active_streamer.any(dim=0),
        torch.where(active_streamer, z_km, torch.full_like(z_km, 100.0)).amin(dim=0),
        z_init_km,
    )

    # Pasko et al. (1997) N2 1PN2 (red) & 2PN2 (blue) optical brightness [kR] with collisional quenching
    # Quenching factor Q_band(z) = A_band / (A_band + k_q * N_quench)
    q_fac_1p = A_EINSTEIN_1PN2 / (A_EINSTEIN_1PN2 + KQ_N2_1PN2 * F_N2_AIR * n_air)
    q_fac_2p = A_EINSTEIN_2PN2 / (A_EINSTEIN_2PN2 + KQ_O2_2PN2 * F_O2_AIR * n_air)
    # Excitation rate rises steeply with reduced electric field E / E_k (Pasko et al., 1997, Fig. 3)
    x_exc = torch.clamp(e_over_ek, min=0.0, max=4.0)
    exc_1p = torch.where(x_exc > 0.20, torch.exp(-2.8 / torch.clamp(x_exc, min=0.20)), torch.zeros_like(x_exc))
    exc_2p = torch.where(x_exc > 0.25, torch.exp(-4.2 / torch.clamp(x_exc, min=0.25)), torch.zeros_like(x_exc))
    # Streamer head enhancement boosts localized emission where streamers propagate
    streamer_boost = torch.where(active_streamer, torch.full_like(x_exc, 12.0), torch.ones_like(x_exc))
    red_1pn2_kr = 4.5e3 * streamer_boost * exc_1p * q_fac_1p
    blue_2pn2_kr = 1.8e3 * streamer_boost * exc_2p * q_fac_2p

    return {
        "z_km": z_km[:, 0],
        "e_over_ek": e_over_ek[:, 0],
        "tau_sigma_s": tau_sigma[:, 0],
        "sigma_sm": sigma_tot[:, 0],
        "red_1pn2_kr": red_1pn2_kr[:, 0],
        "blue_2pn2_kr": blue_2pn2_kr[:, 0],
        "z_init_km": z_init_km[0],
        "z_stop_km": z_stop_km[0],
        "is_halo": halo_mask.any(dim=0)[0],
        "is_prompt_sprite": prompt_mask.any(dim=0)[0],
        "is_delayed_sprite": (delay_mask.any(dim=0) & (~prompt_mask.any(dim=0)))[0],
    }


def _compute_red_sprites_core(
    flash_cg_cell: torch.Tensor,
    cold_cloud_km_cell: torch.Tensor,
    mu0_cell: torch.Tensor,
    rho_host: torch.Tensor,
    temp_top: torch.Tensor,
    q_i_host: torch.Tensor,
    q_p_host: torch.Tensor,
    z_c_host: torch.Tensor,
    dz_c_host: torch.Tensor,
    z_q_km_in: torch.Tensor,
    f_pos_cg_explicit: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    dev = rho_host.device
    dtype = rho_host.dtype
    nz = int(rho_host.shape[0])
    z_meso_km = (
        Z_MESO_MIN_KM + Z_MESO_DZ_KM * torch.arange(N_MESO_LEVELS, device=dev, dtype=dtype)
    ).unsqueeze(1)  # [26, 1]
    z_c_km = z_c_host * 1.0e-3  # [Nz, nc]
    rho_top = rho_host[-1]
    z_top_km = z_c_km[-1]

    n_air_ext, _, _ = compute_mesospheric_conductivity_and_density(
        z_meso_km, mu0_cell, rho_top, z_top_km, temp_top
    )
    z_ref_1d = z_c_km[:, 0].contiguous()
    idx_hi = torch.clamp(torch.searchsorted(z_ref_1d, z_meso_km[:, 0]), min=1, max=max(nz - 1, 1))
    idx_lo = idx_hi - 1
    z_lo = z_c_km[idx_lo]
    z_hi = z_c_km[idx_hi]
    w_lin = torch.clamp((z_meso_km - z_lo) / torch.clamp(z_hi - z_lo, min=1.0e-3), 0.0, 1.0)
    log_rho_res = (1.0 - w_lin) * torch.log(torch.clamp(rho_host[idx_lo], min=1.0e-9)) + w_lin * torch.log(
        torch.clamp(rho_host[idx_hi], min=1.0e-9)
    )
    n_air_res = torch.clamp(torch.exp(log_rho_res) / M_AIR_KG, min=1.0e17, max=N_AIR_STP)
    inside_model = z_meso_km <= z_top_km.unsqueeze(0)
    n_air = torch.where(inside_model, n_air_res, n_air_ext)

    day_fac = torch.clamp((mu0_cell + 0.10) / 0.20, 0.0, 1.0).unsqueeze(0)
    h_prime = H_PRIME_NIGHT_KM * (1.0 - day_fac) + H_PRIME_DAY_KM * day_fac
    beta = BETA_NIGHT_KM1 * (1.0 - day_fac) + BETA_DAY_KM1 * day_fac
    n_e = torch.clamp(
        1.43e13 * torch.exp(-0.15 * h_prime + torch.clamp((beta - 0.15) * (z_meso_km - h_prime), min=-40.0, max=25.0)),
        min=1.0e3,
        max=1.0e11,
    )
    mu_e = MU_E_STP * (N_AIR_STP / n_air)
    z_ion_eff = torch.clamp(z_meso_km, max=55.0) / 6.0 + torch.clamp(z_meso_km - 55.0, min=0.0) / 16.0
    sigma_tot = Q_ELECTRON * n_e * mu_e + 5.0e-14 * torch.exp(z_ion_eff)
    tau_sigma = EPSILON_0 / torch.clamp(sigma_tot, min=1.0e-16)
    n_ratio = n_air / N_AIR_STP
    e_k = E_K_STP * n_ratio

    dm = torch.clamp(rho_host * dz_c_host, min=0.0)
    iwp_cell = (dm * torch.clamp(q_i_host + q_p_host, min=0.0)).sum(dim=0)
    z_q_km = z_q_km_in

    z_m = z_meso_km * 1.0e3
    zq_m = (z_q_km * 1.0e3).unsqueeze(0)
    e_qe_unit = (1.0e3 / (math.pi * EPSILON_0)) * (z_m / torch.clamp(z_m * z_m - zq_m * zq_m, min=1.0e6) ** 2)
    screen = _dielectric_screening_factor(tau_sigma / TAU_DISCHARGE_S)
    e_peak_unit = torch.clamp(e_qe_unit * screen, min=1.0e-12)

    cmc_req_prompt_z = e_k / e_peak_unit
    cmc_req_halo_z = (E_HALO_FRAC * e_k) / e_peak_unit
    can_detach = tau_sigma >= TAU_DETACH_MIN_S
    cmc_req_delay_z = torch.where(can_detach, (E_DELAY_FRAC * e_k) / e_peak_unit, cmc_req_prompt_z)

    cmc_crit_prompt, _ = cmc_req_prompt_z.min(dim=0)
    cmc_crit_delay, k_init_delay = cmc_req_delay_z.min(dim=0)
    cmc_crit_halo, _ = cmc_req_halo_z.min(dim=0)

    iwp_fac = iwp_cell / (iwp_cell + 0.25)
    ccd_fac = torch.clamp((cold_cloud_km_cell - 6.0) / 7.0, 0.0, 1.0)
    anvil_fac = torch.clamp(0.5 * iwp_fac + 0.5 * ccd_fac, 0.0, 1.0)
    cmc_med = 22.0 + 48.0 * anvil_fac
    sigma_ln = 1.15
    inv_s2 = 1.0 / (sigma_ln * math.sqrt(2.0))

    f_pos_cg = 0.10 + 0.15 * anvil_fac
    f_pos_cg = torch.where(f_pos_cg_explicit >= 0.0, 0.5 * f_pos_cg + 0.5 * f_pos_cg_explicit, f_pos_cg)
    p_prompt = 0.5 * torch.erfc(torch.log(torch.clamp(cmc_crit_prompt / cmc_med, min=1.0e-4)) * inv_s2)
    p_any_sprite = 0.5 * torch.erfc(torch.log(torch.clamp(cmc_crit_delay / cmc_med, min=1.0e-4)) * inv_s2)
    p_delay = torch.clamp(p_any_sprite - p_prompt, min=0.0)
    p_halo = 0.5 * torch.erfc(torch.log(torch.clamp(cmc_crit_halo / cmc_med, min=1.0e-4)) * inv_s2)

    eff_pol_wt = f_pos_cg + 0.12 * (1.0 - f_pos_cg)
    cg_pos = torch.clamp(flash_cg_cell, min=0.0)
    sprite_prompt_rate = cg_pos * eff_pol_wt * p_prompt
    sprite_delayed_rate = cg_pos * eff_pol_wt * p_delay
    sprite_rate = sprite_prompt_rate + sprite_delayed_rate
    halo_rate = cg_pos * p_halo

    cmc_99 = cmc_med * math.exp(2.3263 * sigma_ln)
    e_over_ek_99 = (e_peak_unit * cmc_99.unsqueeze(0)) / torch.clamp(e_k, min=1.0e-6)
    e_over_ek_max = e_over_ek_99.amax(dim=0)

    has_active = sprite_rate > 1.0e-18
    z_init_km = torch.where(has_active, z_meso_km[k_init_delay, 0], torch.zeros_like(cmc_med))

    e_peak_99 = e_peak_unit * cmc_99.unsqueeze(0)
    e_st_pos = E_ST_POS_STP * n_ratio
    below_init = z_meso_km <= z_init_km.unsqueeze(0)
    in_streamer = has_active.unsqueeze(0) & below_init & (e_peak_99 >= e_st_pos)
    z_stop_km = torch.where(
        in_streamer.any(dim=0),
        torch.where(in_streamer, z_meso_km, torch.full_like(e_peak_99, 100.0)).amin(dim=0),
        z_init_km,
    )

    q_fac_1p = A_EINSTEIN_1PN2 / (A_EINSTEIN_1PN2 + KQ_N2_1PN2 * F_N2_AIR * n_air)
    q_fac_2p = A_EINSTEIN_2PN2 / (A_EINSTEIN_2PN2 + KQ_O2_2PN2 * F_O2_AIR * n_air)
    x_exc = torch.clamp(e_over_ek_99, min=0.0, max=4.0)
    exc_1p = torch.where(x_exc > 0.20, torch.exp(-2.8 / torch.clamp(x_exc, min=0.20)), torch.zeros_like(x_exc))
    exc_2p = torch.where(x_exc > 0.25, torch.exp(-4.2 / torch.clamp(x_exc, min=0.25)), torch.zeros_like(x_exc))
    boost = torch.where(in_streamer, torch.full_like(x_exc, 12.0), torch.ones_like(x_exc))
    red_1pn2_kr = torch.where(has_active, (4.5e3 * boost * exc_1p * q_fac_1p).amax(dim=0), torch.zeros_like(cmc_med))
    blue_2pn2_kr = torch.where(has_active, (1.8e3 * boost * exc_2p * q_fac_2p).amax(dim=0), torch.zeros_like(cmc_med))

    e_k_ion = e_k[-1]
    e_emp_thresh = E_ELVES_FRAC * e_k_ion
    emp_day_atten = torch.exp(-3.2 * day_fac[0])
    e_emp_per_a = (MU_0_VAC * V_RETURN_STROKE_MS / (2.0 * math.pi * ELVES_RING_RANGE_M)) * ELVES_SIN_THETA * emp_day_atten
    i_crit_ka = (e_emp_thresh / torch.clamp(e_emp_per_a, min=1.0e-12)) * 1.0e-3
    i_med_ka = 24.0 + 18.0 * anvil_fac
    inv_si2 = 1.0 / (I_PEAK_SIGMA_LN * math.sqrt(2.0))
    p_elves = 0.5 * torch.erfc(torch.log(torch.clamp(i_crit_ka / i_med_ka, min=1.0e-4)) * inv_si2)
    elves_rate = cg_pos * p_elves

    z_cld_km = torch.clamp(4.0 + 0.85 * torch.clamp(cold_cloud_km_cell, min=0.0, max=16.0), 4.0, 17.5)
    n_ratio_cld = torch.exp(-z_cld_km / 7.2)
    overshoot_gate = torch.sigmoid((z_cld_km - 11.5) / 1.2) * anvil_fac
    e_cld_eff = (cmc_med * 1.0e3 / (2.0 * math.pi * EPSILON_0 * (3.5e3 ** 3))) * overshoot_gate
    e_st_cld = E_ST_POS_STP * n_ratio_cld
    bj_ratio = e_cld_eff / torch.clamp(e_st_cld, min=1.0)
    p_blue_jet = 0.008 * overshoot_gate * 0.5 * torch.erfc(
        torch.log(torch.clamp(1.8 / torch.clamp(bj_ratio, min=1.0e-4), min=1.0e-4)) * inv_s2
    )
    blue_jet_rate = cg_pos * p_blue_jet
    blue_jet_top_km = torch.where(
        blue_jet_rate > 1.0e-20,
        torch.clamp(z_cld_km + 14.0 + 10.0 * torch.tanh(bj_ratio), 24.0, 44.0),
        torch.zeros_like(z_cld_km),
    )

    gj_cloud_gate = torch.sigmoid((z_cld_km - 14.2) / 0.85) * (1.0 - 0.85 * day_fac[0])
    p_gigantic_jet = p_blue_jet * 0.14 * gj_cloud_gate * torch.clamp(p_halo / 0.05, 0.0, 1.5)
    gigantic_jet_rate = cg_pos * p_gigantic_jet

    len_scale = torch.clamp((z_init_km - z_stop_km) / 25.0, min=0.3, max=2.0)
    sprite_nox_mol_m2s = (
        sprite_rate * NOX_MOLECULES_PER_SPRITE * len_scale
        + gigantic_jet_rate * NOX_MOLECULES_PER_GIGANTIC_JET
    )
    return (
        sprite_rate, sprite_prompt_rate, sprite_delayed_rate, halo_rate, cmc_med,
        e_over_ek_max, red_1pn2_kr, blue_2pn2_kr, z_init_km, z_stop_km,
        sprite_nox_mol_m2s, elves_rate, blue_jet_rate, gigantic_jet_rate, blue_jet_top_km,
    )


# AOBA Formulation Eqs. (5.24)-(5.27)
def compute_red_sprites(
    *,
    flash_cg_cell: torch.Tensor,          # [N_cells] CG flash rate density [flashes m^-2 s^-1]
    cold_cloud_km_cell: torch.Tensor,     # [N_cells] cold-cloud thickness [km]
    mu0_cell: torch.Tensor,               # [N_cells] cosine of solar zenith angle [-]
    rho_host: torch.Tensor,               # [Nz, N_cells] dry air density [kg m^-3]
    theta_m_host: torch.Tensor,           # [Nz, N_cells] moist potential temperature [K]
    q_v_host: torch.Tensor,               # [Nz, N_cells] water vapor [kg kg^-1]
    q_i_host: torch.Tensor,               # [Nz, N_cells] cloud ice [kg kg^-1]
    q_p_host: torch.Tensor,               # [Nz, N_cells] precipitating hydrometeors [kg kg^-1]
    z_c_host: torch.Tensor,               # [Nz, N_cells] layer center altitude [m]
    dz_c_host: torch.Tensor,              # [Nz, N_cells] layer thickness [m]
    area_cell: torch.Tensor,              # [N_cells] Voronoi cell area [m^2]
    explicit_state: object | None = None, # Optional ExplicitLightningState for direct charge-height diagnostics
) -> RedSpriteDiagnostics:
    """Compute global map-free mesospheric red sprite, sprite-halo, optical, & NOx diagnostics."""
    from aoba.thermodynamics import diagnose_pressure_temperature

    nz, nc = rho_host.shape
    # Only top-level temperature is needed for above-model hydrostatic continuation (64x less work than full 3D!)
    _, _, temp_top_2d = diagnose_pressure_temperature(rho_host[-1:], theta_m_host[-1:], q_v_host[-1:])
    temp_top = temp_top_2d[0]

    z_q_km = torch.clamp(5.0 + 0.45 * torch.clamp(cold_cloud_km_cell, min=0.0, max=14.0), 5.5, 11.5)
    f_pos_cg_explicit = torch.full_like(z_q_km, -1.0)
    if explicit_state is not None and hasattr(explicit_state, "q_chg"):
        q_chg_t = explicit_state.q_chg  # [5, Nz, N_cols_lt, Ny, Nx]
        if int(q_chg_t.shape[2]) == nc and int(q_chg_t.shape[1]) == nz:
            rho_q_col = q_chg_t.sum(dim=0).mean(dim=(-2, -1))
            q_abs = rho_q_col.abs() * dz_c_host
            q_abs_sum = q_abs.sum(dim=0)
            has_chg = q_abs_sum > 1.0e-8
            z_q_exp = (q_abs * (z_c_host * 1.0e-3)).sum(dim=0) / torch.clamp(q_abs_sum, min=1.0e-12)
            z_q_km = torch.where(has_chg, torch.clamp(z_q_exp, 4.5, 12.5), z_q_km)
            q_pos_sum = (torch.clamp(rho_q_col, min=0.0) * dz_c_host).sum(dim=0)
            pos_frac_3d = torch.clamp(q_pos_sum / torch.clamp(q_abs_sum, min=1.0e-12), 0.05, 0.50)
            f_pos_cg_explicit = torch.where(has_chg, 0.5 * pos_frac_3d, f_pos_cg_explicit)

    from aoba.kernels import can_use_elem_metal, dispatch_red_sprites_metal

    if can_use_elem_metal(rho_host):
        (
            sprite_rate, sprite_prompt_rate, sprite_delayed_rate, halo_rate, cmc_med,
            e_over_ek_max, red_1pn2_kr, blue_2pn2_kr, z_init_km, z_stop_km,
            sprite_nox_mol_m2s, elves_rate, blue_jet_rate, gigantic_jet_rate, blue_jet_top_km,
        ) = dispatch_red_sprites_metal(
            flash_cg_cell=flash_cg_cell,
            cold_cloud_km_cell=cold_cloud_km_cell,
            mu0_cell=mu0_cell,
            temp_top=temp_top,
            z_q_km_in=z_q_km,
            f_pos_cg_explicit=f_pos_cg_explicit,
            rho_host=rho_host,
            q_i_host=q_i_host,
            q_p_host=q_p_host,
            z_c_host=z_c_host,
            dz_c_host=dz_c_host,
        )
    else:
        core_fn = maybe_cuda_compile(_compute_red_sprites_core, device=rho_host.device)
        (
            sprite_rate, sprite_prompt_rate, sprite_delayed_rate, halo_rate, cmc_med,
            e_over_ek_max, red_1pn2_kr, blue_2pn2_kr, z_init_km, z_stop_km,
            sprite_nox_mol_m2s, elves_rate, blue_jet_rate, gigantic_jet_rate, blue_jet_top_km,
        ) = core_fn(
            flash_cg_cell, cold_cloud_km_cell, mu0_cell,
            rho_host, temp_top, q_i_host, q_p_host, z_c_host, dz_c_host,
            z_q_km, f_pos_cg_explicit,
        )

    return RedSpriteDiagnostics(
        sprite_rate_cell=sprite_rate,
        sprite_prompt_rate_cell=sprite_prompt_rate,
        sprite_delayed_rate_cell=sprite_delayed_rate,
        halo_rate_cell=halo_rate,
        cmc_median_ckm_cell=cmc_med,
        e_over_ek_max_cell=e_over_ek_max,
        red_1pn2_kr_cell=red_1pn2_kr,
        blue_2pn2_kr_cell=blue_2pn2_kr,
        z_init_km_cell=z_init_km,
        z_stop_km_cell=z_stop_km,
        sprite_nox_mol_m2s=sprite_nox_mol_m2s,
        elves_rate_cell=elves_rate,
        blue_jet_rate_cell=blue_jet_rate,
        gigantic_jet_rate_cell=gigantic_jet_rate,
        blue_jet_top_km_cell=blue_jet_top_km,
    )
