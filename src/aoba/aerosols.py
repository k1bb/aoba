"""Map-Free Natural Aerosols: Sea Salt, Marine DMS Sulfate, and Wildfire Carbonaceous Smoke (BC/OC).

Extends AOBA's prognostic mineral dust (`dust.py`) with 4 fully interactive, map-free natural
aerosol mass mixing ratios `[Nz, N_cells]` [kg kg^-1] (`enable_natural_aerosols = false` by default):

1. **Sea-Salt Aerosol (`q_ss`)** (Jaeglé et al., 2011, *Atmos. Chem. Phys.*, 11, 3137-3157,
   Eq. 4; Gong, 2003):
   - SST-modulated whitecap emission `F_ss ~ (1 - f_ice) * f(SST) * u_10^3.41` over open ocean.
2. **Marine DMS-Derived Sulfate (`q_so4`)** (Kloster et al., 2006, *Biogeosciences*, 3, 29-51;
   Simó & Dachs, 2002; Wanninkhof, 2014 / Orr et al., 2017, Table 1 DMS Schmidt polynomial):
   - Seawater DMS diagnosed from FlexPFT surface chlorophyll (`ogcm.bgc.chl[0]` when active,
     or SST/insolation open-ocean proxy otherwise), transferred via wind-driven air-sea gas
     exchange and oxidized to hygroscopic nss-SO4 (marine stratocumulus CCN feedback!).
3. **Wildfire Black Carbon (`q_bc`) & Organic Carbon (`q_oc`)** (Thonicke et al., 2010,
   *Biogeosciences*, 7, 1991-2011):
   - Directly coupled to autonomous wildfire pyrogenic emission fluxes (`land_carbon.fire_bc_kg_m2s`,
     `land_carbon.fire_oc_kg_m2s`).

All 4 tracers are advected conservatively by host SCVT Zalesak FCT (`fct_advect_scvt_batch`),
subject to unconditionally stable implicit gravitational settling, exponential wet scavenging,
and dry deposition, and diagnose both 550 nm Aerosol Optical Depth (AOD) and hygroscopic CCN
number concentration for 2-moment cloud microphysics.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from aoba import constants as C
from aoba.kernels import dispatch_natural_aerosol_4sp_metal, get_active_gpu_backend
from aoba.surface import SURF_OCEAN

# Sea-salt emission & removal (Jaeglé et al., 2011, Eq. 4)
SEASALT_C0: float = 2.5e-13        # Base mass flux prefactor [kg m^-2 s^-1 (m/s)^-3.41] (~4000 Tg/yr global)
SEASALT_V_SETTLE: float = 0.004    # Effective gravitational settling speed [m s^-1]
SEASALT_V_DRY: float = 0.005       # Surface dry deposition velocity [m s^-1]
SEASALT_WET_SCAV: float = 1200.0   # Wet scavenging coefficient [m^2 kg^-1]

# Marine DMS -> SO4 emission & oxidation (Kloster et al., 2006; Orr et al., 2017 Table 1)
DMS_SC_A: float = 2855.7
DMS_SC_B: float = -177.63
DMS_SC_C: float = 6.0438
DMS_SC_D: float = -0.11645
DMS_SC_E: float = 0.00094743
DMS_KW_COEFF: float = 6.97e-7      # Wanninkhof (2014) transfer prefactor [m s^-1 / (m/s)^2]
DMS_TO_SO4_YIELD: float = 0.75     # Molar yield of nss-SO4 from oxidized DMS [-]
M_SO4_KG_MOL: float = 0.09606      # Molar mass of SO4^2- [kg mol^-1]
SO4_V_SETTLE: float = 0.0004       # Fine-mode sulfate settling speed [m s^-1]
SO4_V_DRY: float = 0.0015          # Sulfate dry deposition velocity [m s^-1]
SO4_WET_SCAV: float = 950.0        # Hygroscopic sulfate wet scavenging [m^2 kg^-1]

# Wildfire BC & OC removal (fine carbonaceous smoke)
BC_V_SETTLE: float = 0.0002        # Fine BC settling speed [m s^-1]
BC_V_DRY: float = 0.0010           # BC dry deposition velocity [m s^-1]
BC_WET_SCAV: float = 500.0         # Hydrophobic->aged BC wet scavenging [m^2 kg^-1]
OC_WET_SCAV: float = 750.0         # Partially hydrophilic OC wet scavenging [m^2 kg^-1]

# 550 nm Mass Extinction Efficiencies [m^2 kg^-1] (Chin et al., 2002; Kinne et al., 2006)
MEE_SS_550: float = 2500.0         # Sea salt (hygroscopic scattering, SSA ~ 0.99)
MEE_SO4_550: float = 6000.0        # Sulfate (hygroscopic scattering, SSA ~ 0.99)
MEE_BC_550: float = 10000.0        # Black carbon (strongly absorbing, SSA ~ 0.25)
MEE_OC_550: float = 4000.0         # Organic carbon (SSA ~ 0.92)


@dataclass
class NaturalAerosolState:
    """Prognostic 3D natural aerosol mass mixing ratios `[Nz, N_cells]` [kg kg^-1] & AOD diagnostics."""

    q_ss: torch.Tensor             # Sea salt [kg kg^-1]
    q_so4: torch.Tensor            # Marine DMS-derived nss-sulfate [kg kg^-1]
    q_bc: torch.Tensor             # Wildfire black carbon [kg kg^-1]
    q_oc: torch.Tensor             # Wildfire organic carbon [kg kg^-1]

    # 2D Diagnostics [N_cells]
    aod_ss: torch.Tensor           # 550 nm Sea-salt AOD [-]
    aod_so4: torch.Tensor          # 550 nm Sulfate AOD [-]
    aod_bc: torch.Tensor           # 550 nm Black-carbon AOD [-]
    aod_oc: torch.Tensor           # 550 nm Organic-carbon AOD [-]
    emis_ss_kgm2s: torch.Tensor    # Sea-salt emission flux [kg m^-2 s^-1]
    emis_so4_kgm2s: torch.Tensor   # DMS-derived SO4 source flux [kg m^-2 s^-1]
    ccn_aero_m3: torch.Tensor      # Lowest-layer hygroscopic CCN proxy from SS + SO4 + OC [m^-3]


def init_natural_aerosol_state(
    nz: int,
    n_cells: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> NaturalAerosolState:
    """Initialize zero/clean 3D natural aerosol state."""
    z3 = torch.zeros((nz, n_cells), device=device, dtype=dtype)
    z2 = torch.zeros(n_cells, device=device, dtype=dtype)
    return NaturalAerosolState(
        q_ss=z3.clone(),
        q_so4=z3.clone(),
        q_bc=z3.clone(),
        q_oc=z3.clone(),
        aod_ss=z2.clone(),
        aod_so4=z2.clone(),
        aod_bc=z2.clone(),
        aod_oc=z2.clone(),
        emis_ss_kgm2s=z2.clone(),
        emis_so4_kgm2s=z2.clone(),
        ccn_aero_m3=z2.clone(),
    )


def compute_seasalt_and_dms_emissions(
    surf_type: torch.Tensor,
    sst_k: torch.Tensor,
    u10_ms: torch.Tensor,
    f_seaice: torch.Tensor,
    chl_sfc_mg_m3: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute map-free sea-salt (Jaeglé et al., 2011) & marine DMS->SO4 (Kloster et al., 2006) fluxes."""
    is_ocn = (surf_type == SURF_OCEAN).to(sst_k.dtype)
    open_ocn = is_ocn * (1.0 - torch.clamp(f_seaice, 0.0, 1.0))
    tc = torch.clamp(sst_k - 273.15, min=-2.0, max=35.0)
    u_safe = torch.clamp(u10_ms, min=0.0, max=40.0)

    # 1. Jaeglé et al. (2011, Eq. 4) SST-dependent sea-salt source [kg m^-2 s^-1]
    f_sst = torch.clamp(0.3 + tc * (0.1 + tc * (-0.0076 + 0.00021 * tc)), min=0.1, max=4.0)
    emis_ss = open_ocn * SEASALT_C0 * f_sst * torch.pow(u_safe + 1.0e-4, 3.41)

    # 2. Marine DMS concentration [nmol L^-1 = umol m^-3] -> Wanninkhof/Kloster air-sea flux -> SO4
    if chl_sfc_mg_m3 is not None:
        # Simó & Dachs (2002) biological DMS scaling from FlexPFT Chl [mg m^-3]
        dms_nm = torch.clamp(1.2 + 2.8 * torch.log1p(torch.clamp(chl_sfc_mg_m3, min=0.0)), min=0.5, max=12.0)
    else:
        # Map-free open-ocean baseline DMS (~2.5 nM warm subtropics, ~3.5 nM productive mid/high-lat)
        dms_nm = 2.0 + 1.5 * torch.exp(-((tc - 10.0) / 12.0) ** 2)

    sc_dms = DMS_SC_A + tc * (DMS_SC_B + tc * (DMS_SC_C + tc * (DMS_SC_D + tc * DMS_SC_E)))
    kw_dms = DMS_KW_COEFF * (u_safe * u_safe) * torch.sqrt(660.0 / torch.clamp(sc_dms, min=50.0))
    # DMS_w [umol m^-3] * 1e-6 [mol/umol] * kw [m/s] * yield * M_SO4 [kg/mol] -> [kg SO4 m^-2 s^-1]
    emis_so4 = open_ocn * (dms_nm * 1.0e-6) * kw_dms * DMS_TO_SO4_YIELD * M_SO4_KG_MOL
    return emis_ss, emis_so4


def _step_single_aerosol_column(
    q: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    emis_sfc: torch.Tensor,
    precip_rate: torch.Tensor,
    v_settle: float,
    v_dry: float,
    alpha_wet: float,
    dt: float,
) -> torch.Tensor:
    """Strictly positive implicit emission + settling + dry deposition + exponential wet scavenging."""
    nz = q.shape[0]
    m_k = torch.clamp(rho_d * dz_c, min=1.0e-5)
    q_cur = torch.clamp(q, min=0.0).clone()

    # Inject surface emission into lowest atmospheric layer k = 0
    q_cur[0] = q_cur[0] + (dt * torch.clamp(emis_sfc, min=0.0)) / m_k[0]

    # Top-down implicit gravitational settling + surface dry deposition (unconditionally stable)
    flux_above = torch.zeros_like(q_cur[0])
    out_layers: list[torch.Tensor] = [torch.empty(0)] * nz
    for k in range(nz - 1, -1, -1):
        v_bot = (v_settle + v_dry) if k == 0 else v_settle
        sink = dt * rho_d[k] * v_bot
        q_k = (m_k[k] * q_cur[k] + dt * flux_above) / (m_k[k] + sink)
        flux_above = rho_d[k] * v_bot * q_k
        out_layers[k] = q_k
    q_post = torch.stack(out_layers, dim=0)

    # Exponential precipitation wet scavenging
    wash = torch.exp(-torch.clamp(alpha_wet * torch.clamp(precip_rate, min=0.0) * dt, max=20.0))
    return torch.clamp(q_post * wash.unsqueeze(0), min=0.0, max=1.0e-3)


def step_natural_aerosols_column(
    aero: NaturalAerosolState,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    surf_type: torch.Tensor,
    sst_k: torch.Tensor,
    u10_ms: torch.Tensor,
    f_seaice: torch.Tensor,
    precip_rate: torch.Tensor,
    dt: float,
    chl_sfc_mg_m3: torch.Tensor | None = None,
    fire_bc_kg_m2s: torch.Tensor | None = None,
    fire_oc_kg_m2s: torch.Tensor | None = None,
) -> None:
    """Advance natural aerosol columns (emission, settling, wet scavenging) & update AOD/CCN."""
    emis_ss, emis_so4 = compute_seasalt_and_dms_emissions(
        surf_type, sst_k, u10_ms, f_seaice, chl_sfc_mg_m3=chl_sfc_mg_m3
    )
    emis_bc = fire_bc_kg_m2s if fire_bc_kg_m2s is not None else torch.zeros_like(emis_ss)
    emis_oc = fire_oc_kg_m2s if fire_oc_kg_m2s is not None else torch.zeros_like(emis_ss)

    if get_active_gpu_backend(aero.q_ss.device) in ("mps", "cuda"):
        q_in = torch.stack([aero.q_ss, aero.q_so4, aero.q_bc, aero.q_oc], dim=0)
        emis_4 = torch.stack([emis_ss, emis_so4, emis_bc, emis_oc], dim=0)
        q_out, aod_out = dispatch_natural_aerosol_4sp_metal(
            q_in=q_in,
            rho_d=rho_d,
            dz_c=dz_c,
            emis_4=emis_4,
            precip_rate=precip_rate,
            v_settle_4=(SEASALT_V_SETTLE, SO4_V_SETTLE, BC_V_SETTLE, BC_V_SETTLE),
            v_dry_4=(SEASALT_V_DRY, SO4_V_DRY, BC_V_DRY, BC_V_DRY),
            alpha_wet_4=(SEASALT_WET_SCAV, SO4_WET_SCAV, BC_WET_SCAV, OC_WET_SCAV),
            mee_4=(MEE_SS_550, MEE_SO4_550, MEE_BC_550, MEE_OC_550),
            dt=float(dt),
        )
        aero.q_ss, aero.q_so4, aero.q_bc, aero.q_oc = q_out[0], q_out[1], q_out[2], q_out[3]
        aero.aod_ss, aero.aod_so4, aero.aod_bc, aero.aod_oc = aod_out[0], aod_out[1], aod_out[2], aod_out[3]
    else:
        aero.q_ss = _step_single_aerosol_column(
            aero.q_ss, rho_d, dz_c, emis_ss, precip_rate, SEASALT_V_SETTLE, SEASALT_V_DRY, SEASALT_WET_SCAV, dt
        )
        aero.q_so4 = _step_single_aerosol_column(
            aero.q_so4, rho_d, dz_c, emis_so4, precip_rate, SO4_V_SETTLE, SO4_V_DRY, SO4_WET_SCAV, dt
        )
        aero.q_bc = _step_single_aerosol_column(
            aero.q_bc, rho_d, dz_c, emis_bc, precip_rate, BC_V_SETTLE, BC_V_DRY, BC_WET_SCAV, dt
        )
        aero.q_oc = _step_single_aerosol_column(
            aero.q_oc, rho_d, dz_c, emis_oc, precip_rate, BC_V_SETTLE, BC_V_DRY, OC_WET_SCAV, dt
        )
        # Diagnose 550 nm column AOD [-]
        mass_c = rho_d * dz_c
        aero.aod_ss = MEE_SS_550 * (mass_c * aero.q_ss).sum(dim=0)
        aero.aod_so4 = MEE_SO4_550 * (mass_c * aero.q_so4).sum(dim=0)
        aero.aod_bc = MEE_BC_550 * (mass_c * aero.q_bc).sum(dim=0)
        aero.aod_oc = MEE_OC_550 * (mass_c * aero.q_oc).sum(dim=0)

    aero.emis_ss_kgm2s = emis_ss
    aero.emis_so4_kgm2s = emis_so4

    # Convert lowest-layer mass concentration [kg m^-3] -> CCN number [m^-3] (typical modal particle masses)
    c_ss_0 = rho_d[0] * aero.q_ss[0]
    c_so4_0 = rho_d[0] * aero.q_so4[0]
    c_oc_0 = rho_d[0] * aero.q_oc[0]
    aero.ccn_aero_m3 = torch.clamp(
        c_ss_0 / 5.0e-15 + c_so4_0 / 2.0e-17 + 0.5 * c_oc_0 / 3.0e-17,
        min=0.0,
        max=5.0e9,
    )
