"""Map-Free Prognostic 3D Atmospheric Ozone (O3) Photochemistry & Dry Deposition Module.

Eliminates prescribed ozone climatology maps by solving prognostic 3D ozone volume mixing
ratio `o3_vmr [Nz, N_cells]` [mol mol^-1] directly from universal physical laws, coupled
bidirectionally with:
- Host SCVT 3D Zalesak FCT advection (`fct_advect_scvt_batch` -> emergent Brewer-Dobson
  circulation, polar vortex transport, and stratosphere-troposphere exchange),
- AOBA's shortwave UV/Chappuis + 9.6 um longwave radiation (`radiation.py`),
- Embedded CRM lightning `LNOx` production (`lightning.py`) & wildfire precursors (`land_carbon.py`),
- Prognostic stratospheric water vapor (`q_v`), global `n2o_ppmv` / `ch4_ppmv` / `o3_ecl_ppbv`,
- Emergent EEO vegetation `LAI`, snow/sea-ice cover, and surface type for dry deposition.

Physical Formulation & References
---------------------------------
1. **Chapman (1930) Odd-Oxygen Photochemistry with Overhead UV Self-Shielding**
   (Brasseur & Solomon, 2005, *Aeronomy of the Middle Atmosphere*, Ch. 4-5;
   Burkholder et al., 2019, JPL Evaluation No. 19):
   - 2-band O2 photolysis (Herzberg continuum 200-242 nm + Schumann-Runge 175-205 nm) and
     2-band O3 photolysis (Hartley UV + Huggins/Chappuis) along spherical-shell slant paths
     `amf = 35 / sqrt(1224 * mu0^2 + 1)`.
   - Temperature-dependent Arrhenius recombination `k2(T) = 6.0e-34 * (300/T)^2.4` and
     destruction `k3(T) = 8.0e-12 * exp(-2060/T)` -> automatic stratospheric cooling /
     ozone super-recovery feedback under elevated CO2!
2. **Stratospheric Catalytic NOx (from N2O) & HOx (from prognostic H2O) Loss Cycles**
   (Brasseur & Solomon, 2005, Sec. 5.3-5.4; McLinden et al., 2000, *JGR*, 105, 14653).
3. **Polar Stratospheric Cloud (PSC) Heterogeneous Halocarbon Loss**
   (Cariolle & Teyssèdre, 2007, *Atmos. Chem. Phys.*, 7, 2183-2196, Eq. 3; Solomon, 1999):
   - Activates autonomously whenever `T < 195 K` in sunlit stratospheric air (`mu0 > 0`),
     scaled quadratically by global equivalent effective stratospheric chlorine `o3_ecl_ppbv`
     (`~0.55 ppbv` natural pre-industrial; `~3.5 ppbv` late-20th-century CFC peak).
4. **Tropospheric O3 Production (CH4 + CRM LNOx + Wildfire) & Wet O(1D)+H2O Destruction**
   (Stevenson et al., 2006, *JGR*, 111, D08301; Schumann & Huntrieser, 2007, *ACP*, 7, 3823).
5. **Surface Dry Deposition Coupled to Emergent EEO Vegetation**
   (Wesely, 1989, *Atmos. Environ.*, 23, 1293-1304; Hardacre et al., 2015, *ACP*, 15, 6419).
6. **Unconditionally Stable Exact Riccati-Exponential Integration (Zero Forward Euler)**:
   Solves `d(chi)/dt = P - L1*chi - L2*chi^2` using the cancellation-free positive equilibrium
   root `chi_eq = 2*P / (L1 + sqrt(L1^2 + 4*P*L2))` and exact exponential relaxation
   `chi_new = chi + (chi_eq - chi) * (-expm1(-L_eff * dt))` with `L_eff = L1 + L2*(chi + chi_eq)`.

AOBA Formulation Reference: Sec. 10.2, Eq. (10.2) (Prognostic 3D Atmospheric Ozone Photochemistry).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from aoba import constants as C
from aoba.kernels import can_use_crm_metal, dispatch_ozone_chem_metal
from aoba.radiation import background_ozone_vmr
from aoba.surface import SURF_LAND, SURF_OCEAN

# Physical constants & molecular properties
K_BOLTZMANN: float = 1.380649e-23        # [J K^-1]
AVOGADRO: float = 6.02214076e23          # [mol^-1]
M_AIR_KG_MOLEC: float = 28.9647e-3 / AVOGADRO
M_O3_OVER_AIR: float = 48.0 / 28.9647    # Molar mass ratio M_O3 / M_air [-]
O2_VMR: float = 0.20946                  # Dry-air O2 volume mixing ratio [-]
DU_MOLEC_CM2: float = 2.6867e16          # 1 Dobson Unit [molecules cm^-2]
SEC_PER_YEAR: float = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY

# Photolysis parameters (Brasseur & Solomon, 2005, Ch. 4-5)
J_O2_HZ_0: float = 1.55e-9               # Herzberg continuum (200-242 nm) unattenuated rate [s^-1]
SIGMA_O2_HZ: float = 5.0e-24             # O2 Herzberg absorption cross-section [cm^2]
SIGMA_O3_HZ: float = 7.5e-19             # O3 shielding cross-section in Herzberg window [cm^2]
J_O2_SR_0: float = 4.2e-10               # Schumann-Runge (175-205 nm) effective rate [s^-1]
SIGMA_O2_SR: float = 2.5e-22             # O2 Schumann-Runge cross-section [cm^2]
SIGMA_O3_SR: float = 4.5e-19             # O3 shielding cross-section in SR window [cm^2]

J_O3_HT_0: float = 8.5e-3                # Hartley band (<310 nm) O3 photolysis [s^-1]
SIGMA_O3_HT: float = 2.6e-18             # Effective Hartley shielding cross-section [cm^2]
J_O3_CH_0: float = 4.5e-4                # Huggins + Chappuis visible/near-UV O3 photolysis [s^-1]
SIGMA_O3_CH: float = 1.0e-20             # Weak Chappuis cross-section [cm^2]
J_O1D_HT_0: float = 3.0e-4               # Stratospheric Hartley O(1D) yield rate [s^-1]
SIGMA_O1D_HT: float = 1.8e-18            # Hartley O(1D) shielding cross-section [cm^2]
J_O1D_UVB_0: float = 4.5e-5              # Tropospheric UV-B (305-320 nm) O(1D) rate [s^-1]
SIGMA_O1D_UVB: float = 2.8e-19           # UV-B O3 cross-section (allows tropical surface UV-B) [cm^2]

# Catalytic cycle & tropospheric parameters (universal global constants, zero geographic maps)
N2O_PI_PPMV: float = 0.273               # Pre-industrial N2O reference [ppmv]
CH4_PI_PPMV: float = 0.808               # Pre-industrial CH4 reference [ppmv]
ECL_REF_PPBV: float = 3.5                # Late-20th-century CFC peak equivalent Cl reference [ppbv]
PSC_TEMP_K: float = 195.0                # NAT/ice polar stratospheric cloud threshold temperature [K]
PSC_LOSS_RATE_S: float = 1.0 / (12.0 * C.SECONDS_PER_DAY)  # Sunlit PSC catalytic loss at eCl = 3.5 ppbv [s^-1]

TROP_BG_PROD_S: float = 1.8e-14          # Clean pre-industrial background tropospheric P_O3 at mu0=1 (~20-35 ppbv) [mol/mol s^-1]
TROP_HO2_SINK_S: float = 3.5e-7          # Background tropospheric HO2 + O3 loss (~33-day ceiling lifetime) [s^-1]
LNOX_O3_PROD_EFF: float = 10.0           # Net O3 molecules produced per LNOx N molecule in free troposphere [-]
FIRE_O3_PROD_EFF: float = 0.04           # Net O3 mol per mol wildfire C (via CO/NOx/VOC precursors) [-]

# Surface dry deposition velocities [m s^-1] (Wesely, 1989; Hardacre et al., 2015)
VD_O3_OCEAN: float = 0.0005              # Open ocean & sea-ice uptake [m s^-1]
VD_O3_BARE_SNOW: float = 0.0008          # Bare soil / snow-covered land [m s^-1]
VD_O3_CANOPY_CUTIC: float = 0.0018       # Cuticular + soil uptake beneath canopy [m s^-1]
VD_O3_CANOPY_STOM: float = 0.0042        # Daytime stomatal O3 uptake scaling with (1 - exp(-0.5*LAI)) [m s^-1]


@dataclass
class PrognosticOzoneState:
    """Prognostic 3D atmospheric ozone VMR `[Nz, N_cells]` [mol mol^-1] & column diagnostics."""

    o3_vmr: torch.Tensor                 # [Nz, N_cells] volume mixing ratio [mol mol^-1]
    tco_du: torch.Tensor                 # [N_cells] total column ozone [Dobson Units]
    o3_sfc_ppbv: torch.Tensor            # [N_cells] lowest-layer ozone [ppbv]
    o3_drydep_kgm2s: torch.Tensor        # [N_cells] surface O3 dry-deposition sink [kg O3 m^-2 s^-1]
    o3_chem_net_kgm2s: torch.Tensor      # [N_cells] column-integrated net chemical tendency [kg O3 m^-2 s^-1]


def init_prognostic_ozone_state(
    pres_c: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
) -> PrognosticOzoneState:
    """Initialize 3D ozone horizontally uniformly from Wing et al. (2018) O3(p) (100% map-free)."""
    o3_vmr = torch.clamp(background_ozone_vmr(pres_c), min=1.0e-9, max=2.5e-5).to(dtype=pres_c.dtype, device=pres_c.device)
    dm = rho_d * dz_c
    dN_air = (dm / M_AIR_KG_MOLEC) * 1.0e-4
    tco_du = (o3_vmr * dN_air).sum(dim=0) / DU_MOLEC_CM2
    z2 = torch.zeros_like(tco_du)
    return PrognosticOzoneState(
        o3_vmr=o3_vmr,
        tco_du=tco_du,
        o3_sfc_ppbv=o3_vmr[0] * 1.0e9,
        o3_drydep_kgm2s=z2.clone(),
        o3_chem_net_kgm2s=z2.clone(),
    )


# AOBA Formulation Eq. (10.2)
def step_prognostic_ozone(
    oz: PrognosticOzoneState,
    pres_c: torch.Tensor,
    temp_c: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    q_v: torch.Tensor,
    mu0: torch.Tensor,
    surf_type: torch.Tensor,
    lai_cell: torch.Tensor,
    f_snow_cell: torch.Tensor,
    f_seaice_cell: torch.Tensor,
    dt: float,
    *,
    n2o_ppmv: float = N2O_PI_PPMV,
    ch4_ppmv: float = CH4_PI_PPMV,
    ecl_ppbv: float = 0.55,
    lnox_cell_kgn_m2s: torch.Tensor | None = None,
    fire_c_cell_kgc_m2s: torch.Tensor | None = None,
    uv_j_o2_scale: float = 1.0,
    volc_sad_um2_cm3: torch.Tensor | None = None,
) -> None:
    """Advance 3D prognostic ozone chemistry & dry deposition by one host timestep `dt` [s].

    Uses exact Riccati-exponential integration (unconditionally stable, positivity-preserving,
    zero Forward Euler, zero GPU-starving Python loops).
    """
    if can_use_crm_metal(pres_c) and pres_c.numel() > 0:
        o3_new, tco_du, o3_sfc, drydep, chem_net = dispatch_ozone_chem_metal(
            o3_vmr=oz.o3_vmr,
            pres_c=pres_c,
            temp_c=temp_c,
            rho_d=rho_d,
            dz_c=dz_c,
            q_v=q_v,
            mu0=mu0,
            surf_type=surf_type,
            lai_cell=lai_cell,
            f_snow_cell=f_snow_cell,
            dt=dt,
            n2o_ppmv=n2o_ppmv,
            ch4_ppmv=ch4_ppmv,
            ecl_ppbv=ecl_ppbv,
            lnox_cell_kgn_m2s=lnox_cell_kgn_m2s,
            fire_c_cell_kgc_m2s=fire_c_cell_kgc_m2s,
            surf_land_id=int(SURF_LAND),
            surf_ocean_id=int(SURF_OCEAN),
        )
        oz.o3_vmr = o3_new
        if volc_sad_um2_cm3 is not None or abs(float(uv_j_o2_scale) - 1.0) > 1.0e-5:
            _apply_volcanic_solar_ozone_pert(oz, pres_c, temp_c, rho_d, dz_c, mu0, dt, ecl_ppbv, uv_j_o2_scale, volc_sad_um2_cm3)
        else:
            oz.tco_du = tco_du
            oz.o3_sfc_ppbv = o3_sfc
            oz.o3_drydep_kgm2s = drydep
            oz.o3_chem_net_kgm2s = chem_net
        return

    nz, nc = pres_c.shape
    dtype = pres_c.dtype

    chi0 = torch.clamp(oz.o3_vmr, min=1.0e-10, max=4.0e-5)
    p_pa = torch.clamp(pres_c, min=1.0)
    p_hpa = p_pa * 0.01
    t_k = torch.clamp(temp_c, min=140.0, max=350.0)
    dm = torch.clamp(rho_d * dz_c, min=1.0e-5)  # [Nz, N_cells] kg m^-2

    # Air molecular density [molec cm^-3] and layer column [molec cm^-2]
    m_cm3 = (p_pa / (K_BOLTZMANN * t_k)) * 1.0e-6
    dn_air = (dm / M_AIR_KG_MOLEC) * 1.0e-4

    # Overhead vertical columns of O2 and prognostic O3 [molec cm^-2] (self-shielding!)
    n_o2_above = O2_VMR * (p_pa / (M_AIR_KG_MOLEC * C.GRAVITY)) * 1.0e-4
    dn_o3 = chi0 * dn_air
    dn_o3_rev = torch.flip(dn_o3, dims=[0])
    n_o3_top = chi0[-1:] * (p_pa[-1:] / (M_AIR_KG_MOLEC * C.GRAVITY)) * 1.0e-4 * 0.5
    n_o3_above = torch.flip(n_o3_top + torch.cumsum(dn_o3_rev, dim=0) - 0.5 * dn_o3_rev, dims=[0])

    # Spherical-shell solar slant air-mass factor (identical to radiation.py)
    mu0_pos = torch.clamp(mu0, min=0.0, max=1.0).unsqueeze(0)  # [1, N_cells]
    is_day = (mu0_pos > 1.0e-4).to(dtype)
    amf = 35.0 / torch.sqrt(1224.0 * mu0_pos * mu0_pos + 1.0)
    s_o2 = n_o2_above * amf
    s_o3 = n_o3_above * amf

    # 1. Chapman O2 & O3 photolysis rates [s^-1]
    j_o2 = is_day * (
        J_O2_HZ_0 * torch.exp(-torch.clamp(SIGMA_O2_HZ * s_o2 + SIGMA_O3_HZ * s_o3, max=60.0))
        + J_O2_SR_0 * torch.exp(-torch.clamp(SIGMA_O2_SR * s_o2 + SIGMA_O3_SR * s_o3, max=60.0))
    )
    j_o3_ht = J_O3_HT_0 * torch.exp(-torch.clamp(SIGMA_O3_HT * s_o3, max=60.0))
    j_o3_ch = J_O3_CH_0 * torch.exp(-torch.clamp(SIGMA_O3_CH * s_o3, max=60.0))
    j_o3 = is_day * (j_o3_ht + j_o3_ch)
    j_o1d_ht = is_day * J_O1D_HT_0 * torch.exp(-torch.clamp(SIGMA_O1D_HT * s_o3, max=60.0))
    j_o1d_uvb = is_day * J_O1D_UVB_0 * torch.exp(-torch.clamp(SIGMA_O1D_UVB * s_o3, max=60.0)) * torch.sqrt(mu0_pos + 1.0e-6)
    j_o1d = j_o1d_ht + j_o1d_uvb

    # 2. JPL Arrhenius Chapman kinetics
    k2 = 6.0e-34 * torch.pow(300.0 / t_k, 2.4)
    k3 = 8.0e-12 * torch.exp(-2060.0 / t_k)
    o_over_o3 = j_o3 / torch.clamp(k2 * (O2_VMR * m_cm3) * m_cm3, min=1.0e-30)
    p_chap = 2.0 * j_o2 * O2_VMR
    l2_chap = 2.0 * k3 * o_over_o3 * m_cm3

    # 3. Stratospheric catalytic NOx (N2O) & HOx (prognostic H2O) + PSC heterogeneous loss
    f_strat = torch.sigmoid((130.0 - p_hpa) / 25.0)
    f_trop = 1.0 - f_strat
    n2o_fac = max(float(n2o_ppmv), 0.01) / N2O_PI_PPMV
    noy_vmr = 12.0e-9 * n2o_fac * torch.exp(-((torch.log(torch.clamp(p_hpa, min=0.3) / 6.0)) / 1.65) ** 2) * f_strat
    k_no2_o = 5.1e-12 * torch.exp(210.0 / t_k)
    l1_nox = 2.0 * k_no2_o * (0.45 * noy_vmr * m_cm3) * o_over_o3

    h2o_vmr = torch.clamp(q_v * (28.9647 / 18.015), min=1.0e-6)
    h2o_strat = torch.clamp(h2o_vmr, min=4.5e-6)
    k_ho2_o = 3.0e-11 * torch.exp(200.0 / t_k)
    ho2_vmr = 2.5e-10 * torch.sqrt(torch.clamp((h2o_strat / 5.0e-6) * (j_o1d_ht / 1.0e-4), min=0.0)) * f_strat
    l1_hox = 2.0 * k_ho2_o * (ho2_vmr * m_cm3) * o_over_o3

    # Polar Stratospheric Cloud (PSC) heterogeneous halocarbon loss (T < 195 K, sunlit, 10-200 hPa)
    ecl_ratio = max(float(ecl_ppbv), 0.0) / ECL_REF_PPBV
    psc_p_win = torch.sigmoid((220.0 - p_hpa) / 20.0) * torch.sigmoid((p_hpa - 8.0) / 3.0)
    psc_t_win = torch.sigmoid((PSC_TEMP_K - t_k) / 2.0)
    l1_psc = PSC_LOSS_RATE_S * (ecl_ratio * ecl_ratio) * is_day * psc_p_win * psc_t_win

    # 4. Tropospheric chemistry: CH4 background + CRM LNOx + wildfire precursors vs O(1D)+H2O & HO2 sink
    ch4_fac = math.sqrt(max(float(ch4_ppmv), 0.05) / CH4_PI_PPMV)
    p_trop = f_trop * TROP_BG_PROD_S * ch4_fac * mu0_pos

    # Couple CRM lightning LNOx into mid/upper-tropospheric O3 production (200-700 hPa anvil profile)
    if lnox_cell_kgn_m2s is not None:
        w_lnox = torch.exp(-((p_hpa - 350.0) / 180.0) ** 2) * f_trop
        w_lnox_norm = (w_lnox * dm) / torch.clamp((w_lnox * dm).sum(dim=0, keepdim=True), min=1.0)
        # Convert kg N m^-2 s^-1 -> mol O3 / (mol air s^-1) via M_air / M_N * LNOX_O3_PROD_EFF
        mol_n_rate = torch.clamp(lnox_cell_kgn_m2s, min=0.0).unsqueeze(0) / 0.014007
        mol_air_col = dm / 0.0289647
        p_lnox = LNOX_O3_PROD_EFF * (mol_n_rate * w_lnox_norm) / torch.clamp(mol_air_col, min=1.0e-3)
        p_trop = p_trop + p_lnox * (0.25 + 0.75 * is_day)

    # Couple wildfire CO/NOx/VOC smoke plumes into lower-tropospheric O3 production (p > 600 hPa)
    if fire_c_cell_kgc_m2s is not None:
        w_fire = torch.sigmoid((p_hpa - 650.0) / 80.0) * f_trop
        w_fire_norm = (w_fire * dm) / torch.clamp((w_fire * dm).sum(dim=0, keepdim=True), min=1.0)
        mol_c_rate = torch.clamp(fire_c_cell_kgc_m2s, min=0.0).unsqueeze(0) / 0.012011
        mol_air_col = dm / 0.0289647
        p_fire = FIRE_O3_PROD_EFF * (mol_c_rate * w_fire_norm) / torch.clamp(mol_air_col, min=1.0e-3)
        p_trop = p_trop + p_fire * is_day

    f_o1d_h2o = (7.0 * h2o_vmr) / (1.0 + 7.0 * h2o_vmr)
    l1_trop = f_trop * (j_o1d * f_o1d_h2o + TROP_HO2_SINK_S * (0.3 + 0.7 * mu0_pos))

    # 5. Surface dry deposition in k = 0 (Wesely, 1989) coupled to EEO LAI & snow/sea-ice
    is_land = (surf_type == SURF_LAND).to(dtype)
    is_ocn = (surf_type == SURF_OCEAN).to(dtype)
    f_veg = 1.0 - torch.exp(-0.5 * torch.clamp(lai_cell, min=0.0))
    f_snow = torch.clamp(f_snow_cell, 0.0, 1.0)
    vd_land = f_snow * VD_O3_BARE_SNOW + (1.0 - f_snow) * (
        (1.0 - f_veg) * VD_O3_BARE_SNOW
        + f_veg * (VD_O3_CANOPY_CUTIC + VD_O3_CANOPY_STOM * torch.clamp(mu0, min=0.0, max=1.0))
    )
    vd_sfc = is_land * vd_land + is_ocn * VD_O3_OCEAN + (1.0 - is_land - is_ocn) * VD_O3_BARE_SNOW
    k_dep_0 = vd_sfc / torch.clamp(dz_c[0], min=10.0)  # [N_cells] s^-1

    # 6. Exact unconditionally stable Riccati-exponential chemistry step
    p_tot = p_chap + p_trop
    l1_tot = l1_nox + l1_hox + l1_psc + l1_trop
    l1_with_dep = l1_tot.clone()
    l1_with_dep[0] = l1_with_dep[0] + k_dep_0
    l2_tot = l2_chap

    disc = torch.sqrt(torch.clamp(l1_with_dep * l1_with_dep + 4.0 * p_tot * l2_tot, min=0.0))
    denom = l1_with_dep + disc
    chi_eq = torch.where(denom > 1.0e-30, (2.0 * p_tot) / torch.clamp(denom, min=1.0e-30), chi0)
    l_eff = l1_with_dep + l2_tot * (chi0 + chi_eq)
    relax = -torch.expm1(-torch.clamp(l_eff * dt, min=0.0, max=50.0))
    chi_new = torch.clamp(chi0 + (chi_eq - chi0) * relax, min=1.0e-10, max=4.0e-5)

    # 7. Mass-conserving PBL turbulent vertical mixing across lowest ~1.5 km (replenishes k=0 dry dep!)
    n_mix = min(6, nz - 1)
    if n_mix > 0:
        z_rel = [0.5 * dz_c[0]]
        for k in range(1, n_mix + 1):
            z_rel.append(z_rel[-1] + 0.5 * (dz_c[k - 1] + dz_c[k]))
        h_pbl = 1500.0
        for k in range(n_mix):
            z_int = 0.5 * (z_rel[k] + z_rel[k + 1])
            k_z = torch.where(
                z_int < h_pbl,
                12.0 * (z_int / h_pbl) * ((1.0 - z_int / h_pbl) ** 2) + 2.0,
                torch.full_like(z_int, 0.5),
            )
            dz_int = torch.clamp(0.5 * (dz_c[k] + dz_c[k + 1]), min=10.0)
            rho_int = 0.5 * (rho_d[k] + rho_d[k + 1])
            exch = (dt * rho_int * k_z) / dz_int
            frac = exch / (1.0 + exch * (1.0 / dm[k] + 1.0 / dm[k + 1]))
            dflux = frac * (chi_new[k] - chi_new[k + 1])
            chi_new[k] = chi_new[k] - dflux / dm[k]
            chi_new[k + 1] = chi_new[k + 1] + dflux / dm[k + 1]
        chi_new = torch.clamp(chi_new, min=1.0e-10, max=4.0e-5)

    # Diagnostics: column O3 [DU], surface O3 [ppbv], dry deposition & net chem [kg O3 m^-2 s^-1]
    tco_du = (chi_new * dn_air).sum(dim=0) / DU_MOLEC_CM2
    chi_mid_0 = 0.5 * (chi0[0] + chi_new[0])
    drydep_kgm2s = (chi_mid_0 * M_O3_OVER_AIR * dm[0]) * k_dep_0
    chem_net_kgm2s = ((chi_new - chi0) * M_O3_OVER_AIR * dm).sum(dim=0) / max(dt, 1.0)

    oz.o3_vmr = chi_new
    if volc_sad_um2_cm3 is not None or abs(float(uv_j_o2_scale) - 1.0) > 1.0e-5:
        _apply_volcanic_solar_ozone_pert(oz, pres_c, temp_c, rho_d, dz_c, mu0, dt, ecl_ppbv, uv_j_o2_scale, volc_sad_um2_cm3)
    else:
        oz.tco_du = tco_du
        oz.o3_sfc_ppbv = chi_new[0] * 1.0e9
        oz.o3_drydep_kgm2s = drydep_kgm2s
        oz.o3_chem_net_kgm2s = chem_net_kgm2s


def _apply_volcanic_solar_ozone_pert(
    oz: PrognosticOzoneState,
    pres_c: torch.Tensor,
    temp_c: torch.Tensor,
    rho_d: torch.Tensor,
    dz_c: torch.Tensor,
    mu0: torch.Tensor,
    dt: float,
    ecl_ppbv: float,
    uv_j_o2_scale: float,
    volc_sad_um2_cm3: torch.Tensor | None,
) -> None:
    """Apply volcanic sulfate heterogeneous O3 depletion & 11-yr solar UV modulation (exact exponential)."""
    p_hpa = torch.clamp(pres_c * 0.01, min=0.1)
    is_day = (mu0 > 1.0e-4).to(pres_c.dtype).unsqueeze(0)
    chi = oz.o3_vmr
    if volc_sad_um2_cm3 is not None:
        # Volcanic sulfate heterogeneous halogen activation (Solomon et al., 1996, JGR):
        # Accelerates ClOx/BrOx destruction on sunlit cold stratospheric sulfate surfaces (10-150 hPa).
        ecl_fac = max(float(ecl_ppbv), 0.1) / ECL_REF_PPBV
        p_win = torch.sigmoid((180.0 - p_hpa) / 20.0) * torch.sigmoid((p_hpa - 8.0) / 3.0)
        t_cold = torch.sigmoid((215.0 - temp_c) / 6.0)
        k_volc = (1.0 / (60.0 * C.SECONDS_PER_DAY)) * (ecl_fac * ecl_fac) * torch.clamp(volc_sad_um2_cm3 / 20.0, 0.0, 6.0) * is_day * p_win * t_cold
        chi = chi * torch.exp(-torch.clamp(k_volc * float(dt), min=0.0, max=5.0))
    if abs(float(uv_j_o2_scale) - 1.0) > 1.0e-5:
        # 11-yr Schwabe UV modulation of upper-stratospheric Chapman O2 photolysis (1-30 hPa)
        p_uv = torch.sigmoid((40.0 - p_hpa) / 8.0) * torch.sigmoid((p_hpa - 0.5) / 0.5)
        k_uv = (float(uv_j_o2_scale) - 1.0) * (1.0 / (15.0 * C.SECONDS_PER_DAY)) * is_day * p_uv
        chi = chi * torch.exp(torch.clamp(k_uv * float(dt), min=-0.2, max=0.2))
    chi = torch.clamp(chi, min=1.0e-10, max=4.0e-5)
    oz.o3_vmr = chi
    dm = torch.clamp(rho_d * dz_c, min=1.0e-5)
    dn_air = (dm / M_AIR_KG_MOLEC) * 1.0e-4
    oz.tco_du = (chi * dn_air).sum(dim=0) / DU_MOLEC_CM2
    oz.o3_sfc_ppbv = chi[0] * 1.0e9


def diagnose_ozone_metrics(
    oz: PrognosticOzoneState,
    area_cell: torch.Tensor,
    lat_cell: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute global float64 ozone diagnostics (TCO [DU], tropical/polar DU, sfc [ppbv], drydep [Tg/yr])."""
    area = area_cell.detach().cpu().to(torch.float64)
    a_sum = float(area.sum().item())
    tco = oz.tco_du.detach().cpu().to(torch.float64)
    sfc = oz.o3_sfc_ppbv.detach().cpu().to(torch.float64)
    dep = oz.o3_drydep_kgm2s.detach().cpu().to(torch.float64)
    vmr = oz.o3_vmr.detach().cpu().to(torch.float64)

    tco_glob = float((tco * area).sum().item() / max(a_sum, 1.0))
    sfc_glob = float((sfc * area).sum().item() / max(a_sum, 1.0))
    dep_tg_yr = float((dep * area).sum().item() * SEC_PER_YEAR * 1.0e-9)
    peak_ppmv = float(vmr.max().item() * 1.0e6)

    tco_trop = tco_glob
    tco_high = tco_glob
    if lat_cell is not None:
        lat_deg = torch.rad2deg(lat_cell.detach().cpu().to(torch.float64)).abs()
        m_tr = lat_deg < 20.0
        m_hi = lat_deg > 45.0
        if bool(m_tr.any()):
            tco_trop = float((tco[m_tr] * area[m_tr]).sum().item() / max(float(area[m_tr].sum().item()), 1.0))
        if bool(m_hi.any()):
            tco_high = float((tco[m_hi] * area[m_hi]).sum().item() / max(float(area[m_hi].sum().item()), 1.0))

    return {
        "o3_tco_du": tco_glob,
        "o3_tco_trop_du": tco_trop,
        "o3_tco_high_du": tco_high,
        "o3_tco_min_du": float(tco.min().item()),
        "o3_peak_ppmv": peak_ppmv,
        "o3_sfc_ppbv": sfc_glob,
        "o3_drydep_tg_yr": dep_tg_yr,
    }
