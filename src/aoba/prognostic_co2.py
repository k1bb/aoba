"""Emissions-Driven Interactive Atmospheric CO2 (Global Closed Budget + Optional 3D SCVT Transport).

Enables CMIP7 `esm-piControl`, `esm-hist`, and Zero-Emission Commitment (`ZEC`) experiments
by closing the coupled land-ocean-geochemical-anthropogenic atmospheric carbon budget:

    d(M_CO2)/dt = E_anthro + E_volcanic - NBP_land - fgCO2_ocean - F_weathering

with `1 ppmv = 2.124 Pg C` (Ballantyne et al., 2012; Friedlingstein et al., 2023).

When `enable_co2_3d_transport = True`, additionally evolves a 3D prognostic mixing-ratio field
`co2_vmr [Nz, N_cells]` [mol mol^-1] advected by AOBA's fused SCVT Zalesak FCT solver, with
spatially explicit surface flux injection (`-NBP_land`, `-fgCO2_ocean`, volcanic/anthro sources)
and mass-conserving PBL vertical mixing so Northern Hemisphere seasonal forest breathing
(Keeling Curve) and interhemispheric gradients emerge autonomously without geographic maps.

All global reservoir updates use float64 exact mass conservation (zero float32 ULP drift).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from aoba import constants as C

SEC_PER_YEAR: float = C.DAYS_PER_YEAR * C.SECONDS_PER_DAY
# 1 ppmv dry-air CO2 = 2.124 Pg C (Ciais et al., 2013 IPCC AR5; Friedlingstein et al., 2023)
PGC_PER_PPMV: float = 2.124
KG_C_PER_MOL: float = 0.012011
M_AIR_KG_MOL: float = 0.0289647


@dataclass
class PrognosticCO2State:
    """Interactive emissions-driven atmospheric CO2 state (float64 global + optional 3D VMR)."""

    co2_ppmv_glob: float
    # Optional 3D atmospheric CO2 VMR [Nz, N_cells] [mol mol^-1]
    co2_vmr: torch.Tensor | None = None
    # Instantaneous global flux diagnostics [Pg C yr^-1]
    nbp_land_pgc_yr: float = 0.0
    fgco2_ocean_pgc_yr: float = 0.0
    anthro_pgc_yr: float = 0.0
    volcanic_pgc_yr: float = 0.0
    weathering_pgc_yr: float = 0.0
    # Surface interhemispheric gradient (NH - SH) [ppmv] when 3D transport is active
    ih_grad_ppmv: float = 0.0


def init_prognostic_co2_state(
    co2_ppmv_init: float = C.CO2_PREIND_PPMV,
    *,
    nz: int = 0,
    n_cells: int = 0,
    enable_3d: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> PrognosticCO2State:
    """Initialize interactive atmospheric CO2 state uniformly (100% map-free)."""
    c0 = max(float(co2_ppmv_init), 10.0)
    vmr = None
    if enable_3d and nz > 0 and n_cells > 0:
        vmr = torch.full((nz, n_cells), c0 * 1.0e-6, device=device, dtype=dtype)
    return PrognosticCO2State(co2_ppmv_glob=c0, co2_vmr=vmr)


def step_prognostic_co2(
    state: PrognosticCO2State,
    area_cell: torch.Tensor,
    dt: float,
    *,
    land_nbp_kgc_m2s: torch.Tensor | None = None,
    ocean_fgco2_molc_m2s: torch.Tensor | None = None,
    ocean_area_cell: torch.Tensor | None = None,
    ocean_fgco2_host_molc_m2s: torch.Tensor | None = None,
    anthro_pgc_yr: float = 0.0,
    volcanic_pgc_yr: float = 0.0,
    weathering_pgc_yr: float = 0.0,
    rho_d: torch.Tensor | None = None,
    dz_c: torch.Tensor | None = None,
    lat_cell: torch.Tensor | None = None,
) -> None:
    """Advance closed global (and optional 3D) atmospheric CO2 budget by `dt` [s].

    Sign conventions:
    - `land_nbp_kgc_m2s > 0` = net carbon uptake by land (removes CO2 from atmosphere).
    - `ocean_fgco2_molc_m2s > 0` = net air-sea carbon uptake by ocean (removes CO2 from atmosphere).
    - `anthro_pgc_yr > 0`, `volcanic_pgc_yr > 0` = carbon emissions into atmosphere.
    - `weathering_pgc_yr > 0` = geological CO2 drawdown.
    """
    a_host = area_cell.detach().cpu().to(torch.float64)
    a_tot = max(float(a_host.sum().item()), 1.0)
    nc = a_host.shape[0]

    # 1. Global land NBP [Pg C yr^-1] (supports both host [nc] and MAML subgrid [nc*n_sub])
    nbp_pgc_yr = 0.0
    nbp_host_kgc_m2s = None
    if land_nbp_kgc_m2s is not None:
        m = land_nbp_kgc_m2s.shape[0]
        if m != nc and (m % nc == 0):
            ns = m // nc
            nbp_host_kgc_m2s = land_nbp_kgc_m2s.reshape(nc, ns).mean(dim=1)
        else:
            nbp_host_kgc_m2s = land_nbp_kgc_m2s
        nbp64 = nbp_host_kgc_m2s.detach().cpu().to(torch.float64)
        # kg C s^-1 -> Pg C yr^-1 (1 Pg = 1e12 kg)
        nbp_pgc_yr = float((nbp64 * a_host).sum().item() * SEC_PER_YEAR * 1.0e-12)

    # 2. Global ocean air-sea CO2 uptake [Pg C yr^-1]
    fgco2_pgc_yr = 0.0
    if ocean_fgco2_molc_m2s is not None:
        a_ocn = (
            ocean_area_cell.detach().cpu().to(torch.float64)
            if ocean_area_cell is not None
            else a_host
        )
        fg64 = ocean_fgco2_molc_m2s.detach().cpu().to(torch.float64)
        fgco2_pgc_yr = float((fg64 * a_ocn).sum().item() * KG_C_PER_MOL * SEC_PER_YEAR * 1.0e-12)

    # 3. Exact float64 global atmospheric CO2 mass update [ppmv]
    dt_yr = float(dt) / SEC_PER_YEAR
    net_emit_pgc_yr = (
        float(anthro_pgc_yr)
        + float(volcanic_pgc_yr)
        - nbp_pgc_yr
        - fgco2_pgc_yr
        - float(weathering_pgc_yr)
    )
    d_ppmv = (net_emit_pgc_yr * dt_yr) / PGC_PER_PPMV
    state.co2_ppmv_glob = max(10.0, min(50000.0, state.co2_ppmv_glob + d_ppmv))
    state.nbp_land_pgc_yr = nbp_pgc_yr
    state.fgco2_ocean_pgc_yr = fgco2_pgc_yr
    state.anthro_pgc_yr = float(anthro_pgc_yr)
    state.volcanic_pgc_yr = float(volcanic_pgc_yr)
    state.weathering_pgc_yr = float(weathering_pgc_yr)

    # 4. Optional 3D prognostic CO2 VMR surface injection, PBL mixing, & exact global mass lock
    if state.co2_vmr is not None and rho_d is not None and dz_c is not None:
        vmr = state.co2_vmr
        nz = vmr.shape[0]
        dev = vmr.device
        dtype = vmr.dtype
        dm = torch.clamp(rho_d * dz_c, min=1.0e-5)  # [nz, nc] kg dry air m^-2

        # Net surface carbon flux into atmosphere [kg C m^-2 s^-1]
        f_sfc = torch.full(
            (nc,),
            (float(anthro_pgc_yr) + float(volcanic_pgc_yr) - float(weathering_pgc_yr)) * 1.0e12 / (SEC_PER_YEAR * a_tot),
            device=dev,
            dtype=dtype,
        )
        if nbp_host_kgc_m2s is not None:
            f_sfc = f_sfc - nbp_host_kgc_m2s.to(device=dev, dtype=dtype)
        fg_host = ocean_fgco2_host_molc_m2s if ocean_fgco2_host_molc_m2s is not None else (
            ocean_fgco2_molc_m2s if (ocean_fgco2_molc_m2s is not None and ocean_fgco2_molc_m2s.shape[0] == nc) else None
        )
        if fg_host is not None:
            f_sfc = f_sfc - fg_host.to(device=dev, dtype=dtype) * KG_C_PER_MOL

        # Convert kg C m^-2 s^-1 -> d(VMR) in lowest layer k=0: (f_sfc / KG_C_PER_MOL) / (dm[0] / M_AIR_KG_MOL) * dt
        d_vmr_0 = (f_sfc * (M_AIR_KG_MOL / KG_C_PER_MOL) * float(dt)) / dm[0]
        vmr = vmr.clone()
        vmr[0] = torch.clamp(vmr[0] + d_vmr_0, min=1.0e-5, max=0.05)

        # Mass-conserving PBL vertical mixing across lowest 6 layers (same scheme as ozone_chemistry.py)
        n_mix = min(6, nz - 1)
        if n_mix > 0:
            z_rel = [0.5 * dz_c[0]]
            for k in range(1, n_mix + 1):
                z_rel.append(z_rel[-1] + 0.5 * (dz_c[k - 1] + dz_c[k]))
            for k in range(n_mix):
                z_int = 0.5 * (z_rel[k] + z_rel[k + 1])
                k_z = torch.where(
                    z_int < 1500.0,
                    12.0 * (z_int / 1500.0) * ((1.0 - z_int / 1500.0) ** 2) + 2.0,
                    torch.full_like(z_int, 0.5),
                )
                dz_int = torch.clamp(0.5 * (dz_c[k] + dz_c[k + 1]), min=10.0)
                rho_int = 0.5 * (rho_d[k] + rho_d[k + 1])
                exch = (float(dt) * rho_int * k_z) / dz_int
                frac = exch / (1.0 + exch * (1.0 / dm[k] + 1.0 / dm[k + 1]))
                dflux = frac * (vmr[k] - vmr[k + 1])
                vmr[k] = vmr[k] - dflux / dm[k]
                vmr[k + 1] = vmr[k + 1] + dflux / dm[k + 1]

        # Reconcile mass-weighted global mean of 3D VMR exactly to float64 co2_ppmv_glob
        dm_a = dm * area_cell.to(device=dev, dtype=dtype).unsqueeze(0)
        mean_vmr = (vmr * dm_a).sum() / torch.clamp(dm_a.sum(), min=1.0)
        target_vmr = state.co2_ppmv_glob * 1.0e-6
        vmr = torch.clamp(vmr + (target_vmr - mean_vmr), min=1.0e-5, max=0.05)
        state.co2_vmr = vmr

        if lat_cell is not None:
            nh = (lat_cell >= 0.0).to(dtype)
            sh = 1.0 - nh
            a_dev = area_cell.to(device=dev, dtype=dtype)
            nh_mean = (vmr[0] * a_dev * nh).sum() / torch.clamp((a_dev * nh).sum(), min=1.0)
            sh_mean = (vmr[0] * a_dev * sh).sum() / torch.clamp((a_dev * sh).sum(), min=1.0)
            state.ih_grad_ppmv = float((nh_mean - sh_mean).item() * 1.0e6)
