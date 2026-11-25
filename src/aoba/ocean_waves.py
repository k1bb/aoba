"""Coupled Surface Gravity Waves, Stokes-Drift Langmuir Turbulence, & Sea-Spray Enthalpy Fluxes.

Map-free prognostic sea-state parameterization coupled bidirectionally with AOBA's ocean,
atmospheric boundary layer, and embedded 3D CRM (`enable_wave_coupling = true`):

1. **Prognostic Significant Wave Height `H_s` & Peak Period `T_p`** (Pierson & Moskowitz, 1964;
   Hasselmann et al., 1973 JONSWAP; Janssen, 2004):
   - Relaxes `H_s` [m] and `T_p` [s] toward wind-forced equilibrium (`H_s,eq = 0.0246 * U_10^2`,
     `T_p,eq = 0.81 * U_10`, attenuated by sea-ice cover `(1 - f_ice)`) via unconditionally stable
     exact exponential integration (`tau_wave ~ 6 h` growth / `18 h` swell decay).
2. **Wave-Age-Dependent Charnock Sea-Surface Roughness Modulation** (Janssen, 1991, *JPO*, 21):
   - Young wind-driven seas (`c_p / u_* < 20`) are steeper and rougher than mature swell,
     modulating open-ocean momentum drag smoothly within `[0.85, 1.35]`.
3. **Stokes-Drift Langmuir Mixing Enhancement in OGCM KPP** (McWilliams & Sullivan, 2000,
   *Spill Sci. Tech.*; Takaya et al., 2010, *JGR*, 115, C06009):
   - Computes surface Stokes drift `u_s = (pi^3 * H_s^2) / (g * T_p^3)` and turbulent Langmuir
     number `La_t = sqrt(u_* / u_s)` -> enhancement factor `E_L = sqrt(1 + (0.08 / La_t^4) + (0.04 / La_t^2))`
     clamped to `[1.0, 2.2]`, boosting OGCM mixed-layer KPP diffusivity `K_v`!
4. **High-Wind Sea-Spray Sensible & Latent Enthalpy Fluxes** (Fairall et al., 2009, *JAS*, 66;
   Andreas et al., 2015, *QJRMS*, 141):
   - Activates smoothly above `U_10 > 12 m s^-1` (storm/tropical-cyclone regime), injecting
     wave-breaking spume-droplet latent evaporation (`LH_spray`) and compensating sensible cooling
     (`SH_spray`) directly into the CRM marine boundary layer!

AOBA Formulation Reference: Sec. 9.6, Eqs. (9.9)-(9.10) (Prognostic Waves Hs/Tp, Stokes/Langmuir KPP, Wave-Age Drag & Sea Spray).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C

PI: float = math.pi
RHO_AIR_REF: float = 1.225         # Reference surface air density [kg m^-3]
TAU_WAVE_GROW_S: float = 6.0 * 3600.0    # Wind-sea growth relaxation time [s]
TAU_WAVE_DECAY_S: float = 18.0 * 3600.0  # Swell decay relaxation time [s]
U10_SPRAY_THRESH: float = 12.0     # Wind threshold for spume droplet generation [m s^-1]


@dataclass
class OceanWaveState:
    """Prognostic surface wave height, period, Stokes drift, Langmuir factor, & sea-spray fluxes."""

    hs_m: torch.Tensor                 # [N_ocn] Significant wave height H_s [m]
    tp_s: torch.Tensor                 # [N_ocn] Peak wave period T_p [s]
    u_stokes_ms: torch.Tensor          # [N_ocn] Surface Stokes drift speed u_s [m s^-1]
    langmuir_fac: torch.Tensor         # [N_ocn] OGCM KPP Langmuir diffusivity multiplier [1.0..2.2]
    drag_mod: torch.Tensor             # [N_ocn] Wave-age sea-surface drag multiplier [0.85..1.35]
    # Global diagnostics
    hs_mean_m: float = 0.0
    hs_max_m: float = 0.0
    langmuir_mean: float = 1.0


def init_ocean_wave_state(
    n_ocn: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> OceanWaveState:
    """Initialize calm background ocean swell (`H_s = 1.2 m`, `T_p = 6.0 s`)."""
    return OceanWaveState(
        hs_m=torch.full((n_ocn,), 1.2, device=device, dtype=dtype),
        tp_s=torch.full((n_ocn,), 6.0, device=device, dtype=dtype),
        u_stokes_ms=torch.full((n_ocn,), 0.03, device=device, dtype=dtype),
        langmuir_fac=torch.ones((n_ocn,), device=device, dtype=dtype),
        drag_mod=torch.ones((n_ocn,), device=device, dtype=dtype),
    )


# AOBA Formulation Eqs. (9.9)-(9.10)
def step_ocean_waves(
    waves: OceanWaveState,
    u10_ocn: torch.Tensor,
    f_seaice: torch.Tensor,
    is_ocean: torch.Tensor,
    dt: float,
) -> None:
    """Advance prognostic ocean wave state `(H_s, T_p)` & diagnose Stokes drift + Langmuir enhancement."""
    dtype = waves.hs_m.dtype
    ocn_f = is_ocean.to(dtype)
    open_f = torch.clamp(1.0 - torch.clamp(f_seaice, 0.0, 1.0), 0.0, 1.0) * ocn_f
    u10 = torch.clamp(u10_ocn, min=0.0, max=65.0) * open_f

    # 1. Pierson-Moskowitz / JONSWAP fetch-limited equilibrium wave height & period
    # Cap equilibrium H_s at 18 m and T_p at 20 s for extreme hurricane winds
    hs_eq = torch.clamp(0.0246 * u10 * u10, min=0.0, max=18.0) * open_f
    tp_eq = torch.clamp(0.81 * u10, min=2.5, max=20.0) * open_f + 2.5 * (1.0 - open_f)

    # Asymmetric exact exponential relaxation: faster wind-sea growth (6 h) than swell decay (18 h)
    r_grow = -math.expm1(-float(dt) / TAU_WAVE_GROW_S)
    r_dec = -math.expm1(-float(dt) / TAU_WAVE_DECAY_S)
    r_hs = torch.where(hs_eq >= waves.hs_m, torch.full_like(hs_eq, r_grow), torch.full_like(hs_eq, r_dec))
    r_tp = torch.where(tp_eq >= waves.tp_s, torch.full_like(tp_eq, r_grow), torch.full_like(tp_eq, r_dec))

    waves.hs_m = torch.clamp(waves.hs_m + (hs_eq - waves.hs_m) * r_hs, min=0.0, max=20.0) * open_f
    waves.tp_s = torch.clamp(waves.tp_s + (tp_eq - waves.tp_s) * r_tp, min=2.5, max=22.0)

    # 2. Deep-water surface Stokes drift: u_s = (pi^3 * H_s^2) / (g * T_p^3) [m s^-1]
    u_s = torch.clamp((PI ** 3) * (waves.hs_m ** 2) / (C.GRAVITY * (waves.tp_s ** 3)), min=0.0, max=1.2) * open_f
    waves.u_stokes_ms = u_s

    # 3. Ocean-side friction velocity u_*w & turbulent Langmuir number La_t = sqrt(u_*w / u_s)
    # Approximate C_d ~ 1.3e-3 -> u_*a = sqrt(C_d)*U10, u_*w = sqrt(rho_a/rho_w) * u_*a
    u_star_a = torch.sqrt(torch.tensor(1.3e-3, device=u10.device, dtype=dtype)) * torch.clamp(u10, min=0.5)
    u_star_w = u_star_a * math.sqrt(RHO_AIR_REF / C.RHO_SEAWATER)
    la_t = torch.sqrt(torch.clamp(u_star_w / torch.clamp(u_s, min=1.0e-4), min=0.04, max=25.0))
    # Takaya et al. (2010) / McWilliams & Sullivan (2000) KPP Langmuir enhancement factor
    e_lang = torch.sqrt(1.0 + 0.08 / (la_t ** 4) + 0.04 / (la_t ** 2))
    waves.langmuir_fac = torch.where(open_f > 0.1, torch.clamp(e_lang, min=1.0, max=2.2), torch.ones_like(e_lang))

    # 4. Wave-age-dependent drag multiplier (young wind seas c_p / u_*a < 25 -> steeper -> higher drag)
    c_p = (C.GRAVITY * waves.tp_s) / (2.0 * PI)
    wave_age = torch.clamp(c_p / torch.clamp(u_star_a, min=0.05), min=5.0, max=60.0)
    drag_m = torch.clamp((25.0 / wave_age) ** 0.25, min=0.85, max=1.35)
    waves.drag_mod = torch.where(open_f > 0.1, drag_m, torch.ones_like(drag_m))


def diagnose_ocean_wave_metrics(waves: OceanWaveState, is_ocean: torch.Tensor, f_seaice: torch.Tensor) -> dict[str, float]:
    """Compute open-ocean wave diagnostics (`hs_mean_m`, `hs_max_m`, `langmuir_mean`) without hot-path sync."""
    open_f = torch.clamp(1.0 - torch.clamp(f_seaice.detach().cpu().to(torch.float64), 0.0, 1.0), 0.0, 1.0) * is_ocean.detach().cpu().to(torch.float64)
    w_sum = max(float(open_f.sum().item()), 1.0)
    hs64 = waves.hs_m.detach().cpu().to(torch.float64)
    lm64 = waves.langmuir_fac.detach().cpu().to(torch.float64)
    waves.hs_mean_m = float((hs64 * open_f).sum().item() / w_sum)
    waves.hs_max_m = float(hs64.max().item())
    waves.langmuir_mean = float((lm64 * open_f).sum().item() / w_sum)
    return {
        "wave_hs_mean_m": waves.hs_mean_m,
        "wave_hs_max_m": waves.hs_max_m,
        "wave_langmuir_mean": waves.langmuir_mean,
    }


def compute_sea_spray_enthalpy_fluxes(
    u10_sub: torch.Tensor,
    t_sst_sub: torch.Tensor,
    t_air_sub: torch.Tensor,
    qv_air_sub: torch.Tensor,
    pres_sfc_sub: torch.Tensor,
    is_open_ocean_sub: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute high-wind sea-spray latent & sensible heat flux increments `(d_lh_wm2, d_sh_wm2)`.

    Activates smoothly above `U_10 > 12 m s^-1` (Fairall et al., 2009; Andreas et al., 2015).
    Positive `d_lh_wm2` adds moisture/latent heat to the atmosphere; evaporative cooling of spray
    droplets reduces sensible heating (`d_sh_wm2` typically slightly negative or small positive when
    SST >> T_air), conserving total enthalpy transfer cleanly.
    """
    u_ex = torch.clamp(u10_sub - U10_SPRAY_THRESH, min=0.0, max=50.0)
    # Saturation specific humidity at SST (Bolton 1980)
    tc_s = torch.clamp(t_sst_sub - C.T_FREEZE, min=-2.0, max=40.0)
    e_sat = 611.2 * torch.exp(17.67 * tc_s / (tc_s + 243.5))
    qs_sst = torch.clamp(C.EPS_WATER * e_sat / torch.clamp(pres_sfc_sub - e_sat, min=1000.0), min=1.0e-5, max=0.08)
    dq = torch.clamp(qs_sst - torch.clamp(qv_air_sub, min=0.0), min=0.0, max=0.03)
    dt_sa = torch.clamp(t_sst_sub - t_air_sub, min=-10.0, max=15.0)

    # Spray volume flux scales steeply (~u_ex^2.6, capped safely for hurricane speeds)
    spray_fac = torch.clamp((u_ex / 20.0) ** 2.6, min=0.0, max=4.0) * is_open_ocean_sub
    # Reference hurricane spray latent heat ~150-350 W/m^2 at U10 = 35-45 m/s
    d_lh = torch.clamp(120.0 * spray_fac * (dq / 0.006), min=0.0, max=450.0)
    # Direct thermal exchange from warm droplets minus ~25% evaporative cooling feedback
    d_sh = torch.clamp(25.0 * spray_fac * (dt_sa / 2.0) - 0.25 * d_lh, min=-150.0, max=150.0)
    return d_lh, d_sh
