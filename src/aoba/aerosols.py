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
SEASALT_WET_SCAV: float = 0.80     # Below-cloud wet scavenging [m^2 kg^-1] (Croft et al., 2010; matches dust.py)

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
SO4_WET_SCAV: float = 0.60         # Hygroscopic sulfate wet scavenging [m^2 kg^-1]

# Wildfire BC & OC removal (fine carbonaceous smoke)
BC_V_SETTLE: float = 0.0002        # Fine BC settling speed [m s^-1]
BC_V_DRY: float = 0.0010           # BC dry deposition velocity [m s^-1]
BC_WET_SCAV: float = 0.30          # Hydrophobic->aged BC wet scavenging [m^2 kg^-1]
OC_WET_SCAV: float = 0.50          # Partially hydrophilic OC wet scavenging [m^2 kg^-1]

# 550 nm Mass Extinction Efficiencies [m^2 kg^-1] (Chin et al., 2002; Kinne et al., 2006)
MEE_SS_550: float = 2500.0         # Sea salt (hygroscopic scattering, SSA ~ 0.99)
MEE_SO4_550: float = 6000.0        # Sulfate (hygroscopic scattering, SSA ~ 0.99)
MEE_BC_550: float = 10000.0        # Black carbon (strongly absorbing, SSA ~ 0.25)
MEE_OC_550: float = 4000.0         # Organic carbon / SOA (SSA ~ 0.92)

# Single-scattering albedo, asymmetry g, & LW absorption [m^2 kg^-1] (Bond & Bergstrom, 2006; Kinne et al., 2006)
SSA_SS_SW: float = 0.99
SSA_SO4_SW: float = 0.99
SSA_BC_SW: float = 0.25
SSA_OC_SW: float = 0.92
G_SS_SW: float = 0.76
G_SO4_SW: float = 0.70
G_BC_SW: float = 0.35
G_OC_SW: float = 0.70
LW_K_ABS_SS: float = 80.0
LW_K_ABS_SO4: float = 120.0
LW_K_ABS_BC: float = 450.0
LW_K_ABS_OC: float = 180.0

# Biogenic VOC (isoprene + monoterpene) -> SOA yield (Guenther et al., 2012 MEGAN; Tunved et al., 2006)
BVOC_SOA_BASE_KGM2S: float = 3.2e-11  # Base SOA mass source at 30 C, full canopy, bright PAR (~35-55 Tg SOA/yr)


@dataclass
class NaturalAerosolState:
    """Prognostic 3D natural aerosol mass mixing ratios `[Nz, N_cells]` [kg kg^-1] & AOD diagnostics."""

    q_ss: torch.Tensor             # Sea salt [kg kg^-1]
    q_so4: torch.Tensor            # Marine DMS-derived nss-sulfate [kg kg^-1]
    q_bc: torch.Tensor             # Wildfire black carbon [kg kg^-1]
    q_oc: torch.Tensor             # Wildfire organic carbon + biogenic SOA [kg kg^-1]

    # 2D Diagnostics [N_cells]
    aod_ss: torch.Tensor           # 550 nm Sea-salt AOD [-]
    aod_so4: torch.Tensor          # 550 nm Sulfate AOD [-]
    aod_bc: torch.Tensor           # 550 nm Black-carbon AOD [-]
    aod_oc: torch.Tensor           # 550 nm Organic-carbon / SOA AOD [-]
    emis_ss_kgm2s: torch.Tensor    # Sea-salt emission flux [kg m^-2 s^-1]
    emis_so4_kgm2s: torch.Tensor   # DMS-derived SO4 source flux [kg m^-2 s^-1]
    ccn_aero_m3: torch.Tensor      # Lowest-layer hygroscopic CCN proxy from SS + SO4 + OC [m^-3]
    emis_soa_kgm2s: torch.Tensor | None = None  # Biogenic forest BVOC -> SOA mass flux [kg m^-2 s^-1]
    dep_bc_kgm2s: torch.Tensor | None = None    # Total BC wet+dry deposition flux to surface [kg m^-2 s^-1]


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
        emis_soa_kgm2s=z2.clone(),
        dep_bc_kgm2s=z2.clone(),
    )


def compute_bvoc_soa_emission(
    surf_type: torch.Tensor,
    lai_cell: torch.Tensor,
    t_canopy_k: torch.Tensor,
    sw_par_wm2: torch.Tensor,
) -> torch.Tensor:
    """Map-free biogenic terpene/isoprene -> SOA mass source (Guenther et al., 2012 MEGAN; Tunved et al., 2006)."""
    from aoba.surface import SURF_LAND

    is_land = (surf_type == SURF_LAND).to(lai_cell.dtype)
    f_canopy = 1.0 - torch.exp(-0.5 * torch.clamp(lai_cell, min=0.0))
    # Monoterpene/isoprene Arrhenius response centered at 303.15 K (clamped to avoid extremes)
    t_c = torch.clamp(t_canopy_k - 273.15, min=-20.0, max=48.0)
    f_temp = torch.exp(torch.clamp(0.09 * (t_c - 30.0), min=-5.0, max=2.0))
    # Light-dependent (isoprene) + dark-pool (monoterpene) blend
    par_pos = torch.clamp(sw_par_wm2, min=0.0)
    f_light = 0.35 + 0.65 * (par_pos / (par_pos + 120.0))
    return is_land * BVOC_SOA_BASE_KGM2S * f_canopy * f_temp * f_light


def compute_aerosol_radiative_optics(
    aero: NaturalAerosolState,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 3D `[Nz, N_cells]` SW extinction, scattering, asymmetry*scat, & LW absorption for radiation."""
    dm = torch.clamp(rho_d * dz_c, min=0.0)
    t_ss = dm * (MEE_SS_550 * torch.clamp(aero.q_ss, min=0.0))
    t_so4 = dm * (MEE_SO4_550 * torch.clamp(aero.q_so4, min=0.0))
    t_bc = dm * (MEE_BC_550 * torch.clamp(aero.q_bc, min=0.0))
    t_oc = dm * (MEE_OC_550 * torch.clamp(aero.q_oc, min=0.0))
    tau_sw = t_ss + t_so4 + t_bc + t_oc
    scat_sw = SSA_SS_SW * t_ss + SSA_SO4_SW * t_so4 + SSA_BC_SW * t_bc + SSA_OC_SW * t_oc
    asym_sw = (
        G_SS_SW * SSA_SS_SW * t_ss
        + G_SO4_SW * SSA_SO4_SW * t_so4
        + G_BC_SW * SSA_BC_SW * t_bc
        + G_OC_SW * SSA_OC_SW * t_oc
    )
    tau_lw = dm * (
        LW_K_ABS_SS * torch.clamp(aero.q_ss, min=0.0)
        + LW_K_ABS_SO4 * torch.clamp(aero.q_so4, min=0.0)
        + LW_K_ABS_BC * torch.clamp(aero.q_bc, min=0.0)
        + LW_K_ABS_OC * torch.clamp(aero.q_oc, min=0.0)
    )
    return tau_sw, scat_sw, asym_sw, tau_lw


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
    u_star_cell: torch.Tensor,
    v_settle: float,
    v_dry: float,
    alpha_wet: float,
    dt: float,
) -> torch.Tensor:
    """Strictly positive implicit emission + PBL mixing + settling + dry dep + wet scavenging."""
    nz = q.shape[0]
    m_k = torch.clamp(rho_d * dz_c, min=1.0e-5)
    q_cur = torch.clamp(q, min=0.0).clone()

    # 1. Inject surface emission into lowest atmospheric layer k = 0
    q_cur[0] = q_cur[0] + (dt * torch.clamp(emis_sfc, min=0.0)) / m_k[0]

    # 2. Mass-conserving PBL turbulent mixing across lowest ~1.2 km (matches dust.py)
    n_mix = min(6, nz - 1)
    if n_mix > 0:
        h_pbl = 1200.0
        u_st = torch.clamp(u_star_cell, min=0.05)
        z_rel = [0.5 * dz_c[0]]
        for k in range(1, n_mix + 1):
            z_rel.append(z_rel[-1] + 0.5 * (dz_c[k - 1] + dz_c[k]))
        for k in range(n_mix):
            z_int = 0.5 * (z_rel[k] + z_rel[k + 1])
            k_z = torch.where(
                z_int < h_pbl,
                0.4 * u_st * z_int * ((1.0 - z_int / h_pbl) ** 2),
                torch.zeros_like(z_int),
            )
            dz_int = torch.clamp(0.5 * (dz_c[k] + dz_c[k + 1]), min=10.0)
            rho_int = 0.5 * (rho_d[k] + rho_d[k + 1])
            exch = (dt * rho_int * k_z) / dz_int
            frac = exch / (1.0 + exch * (1.0 / m_k[k] + 1.0 / m_k[k + 1]))
            dq_flux = frac * (q_cur[k] - q_cur[k + 1])
            q_cur[k] = q_cur[k] - dq_flux / m_k[k]
            q_cur[k + 1] = q_cur[k + 1] + dq_flux / m_k[k + 1]

    # 3. Top-down implicit gravitational settling + surface dry deposition (unconditionally stable)
    flux_above = torch.zeros_like(q_cur[0])
    out_layers: list[torch.Tensor] = [torch.empty(0)] * nz
    for k in range(nz - 1, -1, -1):
        v_bot = (v_settle + v_dry) if k == 0 else v_settle
        sink = dt * rho_d[k] * v_bot
        q_k = (m_k[k] * q_cur[k] + dt * flux_above) / (m_k[k] + sink)
        flux_above = rho_d[k] * v_bot * q_k
        out_layers[k] = q_k
    q_post = torch.stack(out_layers, dim=0)

    # 4. Exponential precipitation wet scavenging
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
    soa_oc_kg_m2s: torch.Tensor | None = None,
    u_star_cell: torch.Tensor | None = None,
) -> None:
    """Advance natural aerosol columns (emission, PBL mixing, settling, wet scavenging) & update AOD/CCN."""
    emis_ss, emis_so4 = compute_seasalt_and_dms_emissions(
        surf_type, sst_k, u10_ms, f_seaice, chl_sfc_mg_m3=chl_sfc_mg_m3
    )
    emis_bc = fire_bc_kg_m2s if fire_bc_kg_m2s is not None else torch.zeros_like(emis_ss)
    emis_oc = fire_oc_kg_m2s if fire_oc_kg_m2s is not None else torch.zeros_like(emis_ss)
    if soa_oc_kg_m2s is not None:
        emis_oc = emis_oc + torch.clamp(soa_oc_kg_m2s, min=0.0)
        aero.emis_soa_kgm2s = torch.clamp(soa_oc_kg_m2s, min=0.0)
    else:
        aero.emis_soa_kgm2s = torch.zeros_like(emis_ss)
    mass_c_pre = torch.clamp(rho_d * dz_c, min=0.0)
    bc_col_before = (mass_c_pre * torch.clamp(aero.q_bc, min=0.0)).sum(dim=0)
    u_st = (
        torch.clamp(u_star_cell, min=0.05)
        if u_star_cell is not None
        else torch.clamp(0.035 * torch.clamp(u10_ms, min=0.0), min=0.05)
    )

    if get_active_gpu_backend(aero.q_ss.device) in ("mps", "cuda"):
        q_in = torch.stack([aero.q_ss, aero.q_so4, aero.q_bc, aero.q_oc], dim=0)
        emis_4 = torch.stack([emis_ss, emis_so4, emis_bc, emis_oc], dim=0)
        q_out, aod_out = dispatch_natural_aerosol_4sp_metal(
            q_in=q_in,
            rho_d=rho_d,
            dz_c=dz_c,
            emis_4=emis_4,
            precip_rate=precip_rate,
            u_star_cell=u_st,
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
            aero.q_ss, rho_d, dz_c, emis_ss, precip_rate, u_st, SEASALT_V_SETTLE, SEASALT_V_DRY, SEASALT_WET_SCAV, dt
        )
        aero.q_so4 = _step_single_aerosol_column(
            aero.q_so4, rho_d, dz_c, emis_so4, precip_rate, u_st, SO4_V_SETTLE, SO4_V_DRY, SO4_WET_SCAV, dt
        )
        aero.q_bc = _step_single_aerosol_column(
            aero.q_bc, rho_d, dz_c, emis_bc, precip_rate, u_st, BC_V_SETTLE, BC_V_DRY, BC_WET_SCAV, dt
        )
        aero.q_oc = _step_single_aerosol_column(
            aero.q_oc, rho_d, dz_c, emis_oc, precip_rate, u_st, BC_V_SETTLE, BC_V_DRY, OC_WET_SCAV, dt
        )
        # Diagnose 550 nm column AOD [-]
        mass_c = rho_d * dz_c
        aero.aod_ss = MEE_SS_550 * (mass_c * aero.q_ss).sum(dim=0)
        aero.aod_so4 = MEE_SO4_550 * (mass_c * aero.q_so4).sum(dim=0)
        aero.aod_bc = MEE_BC_550 * (mass_c * aero.q_bc).sum(dim=0)
        aero.aod_oc = MEE_OC_550 * (mass_c * aero.q_oc).sum(dim=0)

    aero.emis_ss_kgm2s = emis_ss
    aero.emis_so4_kgm2s = emis_so4
    # Exact column-conserved BC wet+dry deposition flux [kg BC m^-2 s^-1] for snow/sea-ice darkening
    bc_col_after = (mass_c_pre * torch.clamp(aero.q_bc, min=0.0)).sum(dim=0)
    aero.dep_bc_kgm2s = torch.clamp((bc_col_before + float(dt) * emis_bc - bc_col_after) / max(float(dt), 1.0), min=0.0)

    # Convert lowest-layer mass concentration [kg m^-3] -> CCN number [m^-3] (typical modal particle masses)
    c_ss_0 = rho_d[0] * aero.q_ss[0]
    c_so4_0 = rho_d[0] * aero.q_so4[0]
    c_oc_0 = rho_d[0] * aero.q_oc[0]
    aero.ccn_aero_m3 = torch.clamp(
        c_ss_0 / 5.0e-15 + c_so4_0 / 2.0e-17 + 0.5 * c_oc_0 / 3.0e-17,
        min=0.0,
        max=5.0e9,
    )
