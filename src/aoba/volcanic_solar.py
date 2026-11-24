"""Interactive Stratospheric Volcanic Aerosols (SO2 -> H2SO4) & 11-Year Solar Cycle Module.

Provides two map-free external/natural climate forcing mechanisms toggleable via `config/aoba_default.toml`:

1. **Prognostic Stratospheric Volcanic / SAI Sulfate Aerosols (`enable_volcanic_aerosols`)**:
   - Evolves column-integrated stratospheric sulfur dioxide `so2_strat [N_cells]` [kg S m^-2] and
     sulfate aerosol `h2so4_strat [N_cells]` [kg H2SO4 m^-2] on the SCVT sphere using exact
     analytical 2-stage Bateman exponential kinetics (`SO2 -> H2SO4` with `tau_ox = 35 days`,
     gravitational settling & stratosphere-troposphere removal with `tau_set = 1.1 yr`;
     Stenchikov et al., 1998, *JGR*, 103, 13837; Toohey et al., 2016, *GMD*, 9, 4049).
   - Supports BOTH discrete Pinatubo-class volcanic eruption pulses (`volcanic_eruption_year`,
     `volcanic_so2_tg`, `volcanic_eruption_lat_deg`) AND continuous stratospheric aerosol injection
     (`sai_so2_tg_yr`, `sai_lat_band_deg`).
   - Spreads plumes horizontally across the sphere via conservative SCVT Laplacian mixing
     (representing Brewer-Dobson meridional dispersion), maps vertically onto `15-100 hPa`
     (centered at `~40 hPa`) to produce SW scattering + LW IR absorption (`aero_opt` delta)
     AND stratospheric sulfate surface area density `sad_um2_cm3` for heterogeneous ozone depletion!

2. **11-Year Schwabe Solar Activity Cycle (`enable_solar_cycle`)**:
   - Modulates total solar irradiance (`s_toa` by `+/- solar_cycle_amp_wm2`, default `1.0 W m^-2`,
     Lean et al., 2005, *Solar Phys.*, 230, 27) AND stratospheric UV O2 photolysis `J_O2`
     (`+/- 3%` UV modulation, Gray et al., 2010, *Rev. Geophys.*, 48, RG4001).

AOBA Formulation Reference: Sec. 10.4, Eqs. (10.6)-(10.8) (2-Stage Bateman Volcanic SO2->H2SO4 & 11-yr Solar Cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C

SEC_PER_YEAR: float = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
TAU_SO2_OX_S: float = 35.0 * C.SECONDS_PER_DAY     # SO2 + OH -> H2SO4 e-folding time [s]
TAU_H2SO4_SET_S: float = 1.10 * SEC_PER_YEAR       # Stratospheric aerosol removal e-folding time [s]
M_H2SO4_OVER_SO2: float = 98.08 / 64.066           # Mass conversion SO2 -> pure H2SO4 (~1.531)
# Wet H2SO4/H2O (75 wt%) stratospheric aerosol optical properties (Stenchikov et al., 1998)
K_EXT_SW_M2_KG: float = 3400.0                     # 550 nm mass extinction [m^2 / kg H2SO4]
SSA_SW_SULFATE: float = 0.998                      # Nearly pure scattering in SW
ASYM_SW_SULFATE: float = 0.72                      # Accumulation-mode r_eff ~ 0.4-0.5 um
K_ABS_LW_M2_KG: float = 135.0                      # Broadband LW IR absorption [m^2 / kg H2SO4]
# Surface area density conversion: ~1.8e4 um^2 cm^-3 per (kg H2SO4 m^-3) -> Pinatubo peak ~30-40 um^2/cm^3
SAD_PER_KG_M3: float = 1.8e9


@dataclass
class VolcanicSolarState:
    """Prognostic stratospheric SO2 / H2SO4 aerosol columns and solar-cycle phase state."""

    so2_strat_kgm2: torch.Tensor       # [N_cells] stratospheric SO2 mass column [kg SO2 m^-2]
    h2so4_strat_kgm2: torch.Tensor     # [N_cells] stratospheric H2SO4 aerosol column [kg H2SO4 m^-2]
    eruption_fired: bool = False       # Ensures configured single eruption pulse fires exactly once
    # Diagnostics
    aod_volc_mean: float = 0.0         # Global-mean stratospheric volcanic 550 nm AOD [-]
    burden_so2_tg: float = 0.0         # Global stratospheric SO2 burden [Tg SO2]
    burden_h2so4_tg: float = 0.0       # Global stratospheric H2SO4 burden [Tg H2SO4]
    tsi_delta_wm2: float = 0.0         # Instantaneous solar-cycle TSI anomaly [W m^-2]
    uv_j_o2_scale: float = 1.0         # Instantaneous UV photolysis scaling factor for ozone [-]


def init_volcanic_solar_state(
    n_cells: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> VolcanicSolarState:
    """Initialize clean background stratospheric volcanic/solar state (zero initial plume)."""
    z = torch.zeros((n_cells,), device=device, dtype=dtype)
    return VolcanicSolarState(so2_strat_kgm2=z.clone(), h2so4_strat_kgm2=z.clone())


# AOBA Formulation Eq. (10.8)
def compute_solar_cycle_modulation(
    sim_time_days: float,
    amp_wm2: float = 1.0,
    period_yr: float = 11.0,
    phase_yr: float = 0.0,
) -> tuple[float, float]:
    """Return `(tsi_delta_wm2, uv_j_o2_scale)` for the Schwabe solar cycle (Lean et al., 2005)."""
    t_yr = float(sim_time_days) / C.DAYS_PER_YEAR
    phase = 2.0 * math.pi * ((t_yr - float(phase_yr)) / max(float(period_yr), 0.1))
    s = math.sin(phase)
    tsi_delta = 0.5 * float(amp_wm2) * s
    # Stratospheric UV (200-242 nm Herzberg) varies ~3% peak-to-trough per 1 W/m^2 TSI
    uv_scale = max(0.85, min(1.15, 1.0 + 0.015 * float(amp_wm2) * s))
    return tsi_delta, uv_scale


# AOBA Formulation Eqs. (10.6)-(10.7)
def step_volcanic_aerosols(
    vs: VolcanicSolarState,
    mesh,
    sim_time_days: float,
    dt: float,
    *,
    eruption_year: float = -1.0,
    eruption_so2_tg: float = 0.0,
    eruption_lat_deg: float = 15.0,
    eruption_width_deg: float = 15.0,
    sai_so2_tg_yr: float = 0.0,
    sai_lat_band_deg: float = 20.0,
) -> None:
    """Advance stratospheric SO2 -> H2SO4 oxidation, settling, and horizontal Brewer-Dobson spreading.

    Uses exact 2-stage Bateman analytical exponential integration (zero Forward Euler!).
    """
    dev = vs.so2_strat_kgm2.device
    dtype = vs.so2_strat_kgm2.dtype
    lat_deg = torch.rad2deg(mesh.lat_cell.to(device=dev, dtype=dtype))
    area = mesh.area_cell.to(device=dev, dtype=dtype)

    # 1. Optional discrete volcanic eruption pulse (e.g. Pinatubo: 17 Tg SO2 at 15 deg N)
    sim_yr = float(sim_time_days) / C.DAYS_PER_YEAR
    if (not vs.eruption_fired) and float(eruption_year) >= 0.0 and float(eruption_so2_tg) > 0.0 and sim_yr >= float(eruption_year):
        w_lat = torch.exp(-0.5 * ((lat_deg - float(eruption_lat_deg)) / max(float(eruption_width_deg), 5.0)) ** 2)
        denom = torch.clamp((w_lat * area).sum(), min=1.0)
        # 1 Tg = 1e9 kg SO2
        vs.so2_strat_kgm2 = vs.so2_strat_kgm2 + (float(eruption_so2_tg) * 1.0e9 / denom) * w_lat
        vs.eruption_fired = True

    # 2. Optional continuous stratospheric aerosol injection (SAI geoengineering / persistent volcanism)
    if float(sai_so2_tg_yr) > 0.0:
        w_sai = torch.exp(-0.5 * (lat_deg / max(float(sai_lat_band_deg), 5.0)) ** 2)
        denom = torch.clamp((w_sai * area).sum(), min=1.0)
        kg_step = float(sai_so2_tg_yr) * 1.0e9 * (float(dt) / SEC_PER_YEAR)
        vs.so2_strat_kgm2 = vs.so2_strat_kgm2 + (kg_step / denom) * w_sai

    # 3. Exact 2-stage Bateman analytical solution for:
    #    dS/dt = -k1 * S,   dA/dt = alpha * k1 * S - k2 * A
    k1 = 1.0 / TAU_SO2_OX_S
    k2 = 1.0 / TAU_H2SO4_SET_S
    e1 = math.exp(-k1 * float(dt))
    e2 = math.exp(-k2 * float(dt))
    s0 = torch.clamp(vs.so2_strat_kgm2, min=0.0)
    a0 = torch.clamp(vs.h2so4_strat_kgm2, min=0.0)
    s_new = s0 * e1
    # Exact Bateman transfer factor (cancellation-free since k1 >> k2)
    bateman = M_H2SO4_OVER_SO2 * (k1 / (k1 - k2)) * (e2 - e1)
    a_new = a0 * e2 + s0 * bateman

    # 4. Meridional Brewer-Dobson horizontal eddy diffusion (K_h ~ 6e5 m^2/s, stability-bounded)
    dx_mean = getattr(mesh, "_volc_dx_mean", None)
    if dx_mean is None:
        dx_mean = float(mesh.dc_edge.mean().item())
        mesh._volc_dx_mean = dx_mean
    k_h = min(6.0e5, 0.08 * dx_mean * dx_mean / max(float(dt), 1.0))
    if k_h > 0.0:
        from aoba.scvt_dynamics import compute_scalar_laplacian_cell

        lap_a = compute_scalar_laplacian_cell(mesh, a_new.unsqueeze(0)).squeeze(0)
        a_new = torch.clamp(a_new + (k_h * float(dt)) * lap_a, min=0.0)

    vs.so2_strat_kgm2 = s_new
    vs.h2so4_strat_kgm2 = a_new

    a64 = area.detach().cpu().to(torch.float64)
    a_sum = max(float(a64.sum().item()), 1.0)
    s_tot = float((s_new.detach().cpu().to(torch.float64) * a64).sum().item())
    h_tot = float((a_new.detach().cpu().to(torch.float64) * a64).sum().item())
    vs.burden_so2_tg = s_tot * 1.0e-9
    vs.burden_h2so4_tg = h_tot * 1.0e-9
    vs.aod_volc_mean = (h_tot / a_sum) * K_EXT_SW_M2_KG


def compute_volcanic_rad_optics_and_sad(
    vs: VolcanicSolarState,
    pres_c: torch.Tensor,  # [Nz, N_cells] Pa
    dz_c: torch.Tensor,    # [Nz, N_cells] m
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Map stratospheric H2SO4 column onto 15-100 hPa vertical Gaussian profile -> `(aero_opt, sad_um2_cm3)`.

    Returns:
    - `aero_opt = (tau_sw, scat_sw, asym_sw, tau_lw)` `[Nz, N_cells]` ready to add into radiation.
    - `sad_um2_cm3 [Nz, N_cells]` stratospheric sulfate aerosol surface area density [um^2 cm^-3]
      for heterogeneous ozone destruction in `ozone_chemistry.py`.
    """
    p_hpa = torch.clamp(pres_c * 0.01, min=0.5)
    # Log-pressure Gaussian plume centered at 40 hPa (sigma = 0.55 in ln p -> spans ~18-90 hPa)
    w_z = torch.exp(-0.5 * ((torch.log(p_hpa / 40.0)) / 0.55) ** 2)
    w_norm = w_z / torch.clamp(w_z.sum(dim=0, keepdim=True), min=1.0e-12)
    dm_h2so4 = vs.h2so4_strat_kgm2.unsqueeze(0) * w_norm  # [Nz, N_cells] kg H2SO4 m^-2 per layer

    tau_sw = dm_h2so4 * K_EXT_SW_M2_KG
    scat_sw = tau_sw * SSA_SW_SULFATE
    asym_sw = scat_sw * ASYM_SW_SULFATE
    tau_lw = dm_h2so4 * K_ABS_LW_M2_KG

    conc_kg_m3 = dm_h2so4 / torch.clamp(dz_c, min=50.0)
    sad_um2_cm3 = torch.clamp(conc_kg_m3 * SAD_PER_KG_M3, max=150.0)
    return (tau_sw, scat_sw, asym_sw, tau_lw), sad_um2_cm3
