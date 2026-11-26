"""CMIP7 Data Request (v1.2 ESM-BCV) Optional Output Diagnostics & Pressure-Level Interpolation.

Produces CF-1.11 / UGRID-1.0 / CMIP7-compliant monthly Zarr v3 datasets (`cmip7_monthly.zarr`
and fixed grid `cmip7_fx.zarr`) covering the Baseline Climate Variables for Earth System
Modelling (ESM-BCVs) across Atmosphere (2D surface/TOA + 3D `plev19`), Ocean (2D + 3D `olevel`),
Sea Ice, and Land/Biosphere realms.

Disabled by default (`cmip7_output_enabled = false`) during multi-decadal spin-up so zero
extra compute or storage overhead is incurred until production / intercomparison runs.

References
----------
- CMIP7 Data Request v1.2 (WCRP-CMIP, 2025): Baseline Climate Variables for Earth System
  Modelling (ESM-BCVs), standardized `plev19` pressure coordinates, and CF-1.11 sign conventions.

AOBA Formulation Reference: Sec. 12.2 (CMIP7 Data Request v1.2 ESM-BCV Zarr v3 Diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import numpy as np
import torch
import xarray as xr

from aoba import constants as C
from aoba.scvt_mesh import reconstruct_cell_winds
from aoba.surface import (
    RHO_SNOWPACK,
    SOIL_LAYER_DEPTHS,
    SURF_ICESHEET,
    SURF_LAND,
    SURF_OCEAN,
)
from aoba.thermodynamics import diagnose_pressure_temperature, qsat_mixed

if TYPE_CHECKING:
    from aoba.coupled_model import AOBAState

# Official CMIP7 standardized 19 pressure levels [Pa]
CMIP7_PLEV19_PA: tuple[float, ...] = (
    100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0, 40000.0,
    30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 7000.0, 5000.0,
    3000.0, 2000.0, 1000.0, 500.0, 100.0,
)

# Official CMIP7 variable metadata registry: short_name -> (standard_name, long_name, units, positive, realm)
CMIP7_META: dict[str, tuple[str, str, str, str, str]] = {
    # Atmosphere 2D / Surface & TOA
    "tas": ("air_temperature", "Near-Surface Air Temperature", "K", "", "atmos"),
    "tasmax": ("air_temperature", "Daily Maximum Near-Surface Air Temperature", "K", "", "atmos"),
    "tasmin": ("air_temperature", "Daily Minimum Near-Surface Air Temperature", "K", "", "atmos"),
    "ts": ("surface_temperature", "Surface Skin Temperature", "K", "", "atmos"),
    "ps": ("surface_air_pressure", "Surface Air Pressure", "Pa", "", "atmos"),
    "psl": ("air_pressure_at_mean_sea_level", "Sea Level Pressure", "Pa", "", "atmos"),
    "huss": ("specific_humidity", "Near-Surface Specific Humidity", "1", "", "atmos"),
    "hurs": ("relative_humidity", "Near-Surface Relative Humidity", "%", "", "atmos"),
    "uas": ("eastward_wind", "Eastward Near-Surface Wind", "m s-1", "", "atmos"),
    "vas": ("northward_wind", "Northward Near-Surface Wind", "m s-1", "", "atmos"),
    "sfcWind": ("wind_speed", "Near-Surface Wind Speed", "m s-1", "", "atmos"),
    "pr": ("precipitation_flux", "Precipitation", "kg m-2 s-1", "", "atmos"),
    "prc": ("convective_precipitation_flux", "Convective Precipitation", "kg m-2 s-1", "", "atmos"),
    "prsn": ("snowfall_flux", "Snowfall Flux", "kg m-2 s-1", "", "atmos"),
    "evspsbl": ("water_evapotranspiration_flux", "Evaporation Including Sublimation and Transpiration", "kg m-2 s-1", "", "atmos"),
    "prw": ("atmosphere_mass_content_of_water_vapor", "Water Vapor Path", "kg m-2", "", "atmos"),
    "clwvi": ("atmosphere_mass_content_of_cloud_condensed_water", "Condensed Water Path", "kg m-2", "", "atmos"),
    "clivi": ("atmosphere_mass_content_of_cloud_ice", "Ice Water Path", "kg m-2", "", "atmos"),
    "clt": ("cloud_area_fraction", "Total Cloud Cover Percentage", "%", "", "atmos"),
    "hfss": ("surface_upward_sensible_heat_flux", "Surface Upward Sensible Heat Flux", "W m-2", "up", "atmos"),
    "hfls": ("surface_upward_latent_heat_flux", "Surface Upward Latent Heat Flux", "W m-2", "up", "atmos"),
    "tauu": ("surface_downward_eastward_stress", "Surface Downward Eastward Wind Stress", "Pa", "down", "atmos"),
    "tauv": ("surface_downward_northward_stress", "Surface Downward Northward Wind Stress", "Pa", "down", "atmos"),
    "rsdt": ("toa_incoming_shortwave_flux", "TOA Incident Shortwave Radiation", "W m-2", "down", "atmos"),
    "rsut": ("toa_outgoing_shortwave_flux", "TOA Outgoing Shortwave Radiation", "W m-2", "up", "atmos"),
    "rlut": ("toa_outgoing_longwave_flux", "TOA Outgoing Longwave Radiation", "W m-2", "up", "atmos"),
    "rsds": ("surface_downwelling_shortwave_flux_in_air", "Surface Downwelling Shortwave Radiation", "W m-2", "down", "atmos"),
    "rsus": ("surface_upwelling_shortwave_flux_in_air", "Surface Upwelling Shortwave Radiation", "W m-2", "up", "atmos"),
    "rlds": ("surface_downwelling_longwave_flux_in_air", "Surface Downwelling Longwave Radiation", "W m-2", "down", "atmos"),
    "rlus": ("surface_upwelling_longwave_flux_in_air", "Surface Upwelling Longwave Radiation", "W m-2", "up", "atmos"),
    # Atmosphere 3D on plev19
    "ta": ("air_temperature", "Air Temperature", "K", "", "atmos"),
    "ua": ("eastward_wind", "Eastward Wind", "m s-1", "", "atmos"),
    "va": ("northward_wind", "Northward Wind", "m s-1", "", "atmos"),
    "wap": ("lagrangian_tendency_of_air_pressure", "Omega (=dp/dt)", "Pa s-1", "", "atmos"),
    "zg": ("geopotential_height", "Geopotential Height", "m", "", "atmos"),
    "hus": ("specific_humidity", "Specific Humidity", "1", "", "atmos"),
    "hur": ("relative_humidity", "Relative Humidity", "%", "", "atmos"),
    "cl": ("cloud_area_fraction_in_atmosphere_layer", "Percentage Cloud Cover", "%", "", "atmos"),
    # Ocean 2D & 3D (olevel)
    "tos": ("sea_surface_temperature", "Sea Surface Temperature", "degC", "", "ocean"),
    "sos": ("sea_surface_salinity", "Sea Surface Salinity", "0.001", "", "ocean"),
    "zos": ("sea_surface_height_above_geoid", "Sea Surface Height Above Geoid", "m", "", "ocean"),
    "mlotst": ("ocean_mixed_layer_thickness_defined_by_sigma_t", "Ocean Mixed Layer Thickness", "m", "", "ocean"),
    "hfds": ("surface_downward_heat_flux_in_sea_water", "Downward Heat Flux at Sea Water Surface", "W m-2", "down", "ocean"),
    "thetao": ("sea_water_potential_temperature", "Sea Water Potential Temperature", "degC", "", "ocean"),
    "so": ("sea_water_salinity", "Sea Water Salinity", "0.001", "", "ocean"),
    "uo": ("sea_water_x_velocity", "Sea Water Eastward Velocity", "m s-1", "", "ocean"),
    "vo": ("sea_water_y_velocity", "Sea Water Northward Velocity", "m s-1", "", "ocean"),
    "wo": ("upward_sea_water_velocity", "Sea Water Vertical Velocity", "m s-1", "", "ocean"),
    # Sea Ice
    "siconc": ("sea_ice_area_fraction", "Sea-Ice Area Percentage (Ocean Grid)", "%", "", "seaIce"),
    "sithick": ("sea_ice_thickness", "Sea Ice Thickness", "m", "", "seaIce"),
    "sivol": ("sea_ice_volume_per_area", "Sea-Ice Volume per Area", "m", "", "seaIce"),
    "sisnthick": ("surface_snow_thickness", "Snow Thickness on Sea Ice", "m", "", "seaIce"),
    # Land, Snow & Biosphere
    "snc": ("surface_snow_area_fraction", "Snow Area Percentage", "%", "", "landIce"),
    "snw": ("surface_snow_amount", "Surface Snow Amount", "kg m-2", "", "landIce"),
    "mrso": ("mass_content_of_water_in_soil", "Total Soil Moisture Content", "kg m-2", "", "land"),
    "mrsos": ("mass_content_of_water_in_soil_layer", "Moisture in Upper Portion of Soil Column (0-10 cm)", "kg m-2", "", "land"),
    "mrro": ("runoff_flux", "Total Runoff", "kg m-2 s-1", "", "land"),
    "mrros": ("surface_runoff_flux", "Surface and Glacier Runoff", "kg m-2 s-1", "", "land"),
    "tsl": ("soil_temperature", "Temperature of Soil", "K", "", "land"),
    "gpp": ("gross_primary_productivity_of_biomass_expressed_as_carbon", "Carbon Mass Flux out of Atmosphere Due to Gross Primary Production", "kg m-2 s-1", "", "land"),
    "lai": ("leaf_area_index", "Leaf Area Index", "1", "", "land"),
    "tran": ("transpiration_flux", "Transpiration Water Flux", "kg m-2 s-1", "up", "land"),
    # AOBA-Specific Autonomous EEO Biosphere Traits & Canopy CO2 Fluxes
    "rleaf": ("leaf_respiration_carbon_flux", "Canopy Leaf Dark Respiration Carbon Flux (MMRT)", "kg m-2 s-1", "up", "land"),
    "an_net": ("net_canopy_carbon_assimilation_flux", "Canopy Net CO2 Assimilation Flux (GPP - R_leaf)", "kg m-2 s-1", "down", "land"),
    "hcanopy": ("canopy_height", "Autonomous EEO Potential Canopy Height", "m", "", "land"),
    "sroot": ("root_zone_water_storage_capacity", "Optimal Drought-Bridging Root-Zone Storage Capacity S_R", "kg m-2", "", "land"),
    "fc4": ("c4_plant_area_fraction", "Autonomous C4 Photosynthetic Pathway Fraction", "1", "", "land"),
    "lai_c3": ("c3_leaf_area_index", "C3 Leaf Area Index ((1 - f_c4) * LAI)", "1", "", "land"),
    "lai_c4": ("c4_leaf_area_index", "C4 Leaf Area Index (f_c4 * LAI)", "1", "", "land"),
    "fevg": ("evergreen_plant_area_fraction", "Autonomous Evergreen Leaf-Habit Fraction", "1", "", "land"),
    "lai_evg": ("evergreen_leaf_area_index", "Evergreen Leaf Area Index", "1", "", "land"),
    "lai_dec": ("deciduous_leaf_area_index", "Deciduous Leaf Area Index", "1", "", "land"),
    "vcmax25": ("canopy_vcmax_at_25C", "Canopy-Effective Rubisco Vcmax at 25 degC", "umol m-2 s-1", "", "land"),
    "jmax25": ("canopy_jmax_at_25C", "Canopy-Effective Electron Transport Jmax at 25 degC", "umol m-2 s-1", "", "land"),
    "vcmax25_c3": ("c3_vcmax_at_25C", "C3 Acclimated Rubisco Vcmax at 25 degC", "umol m-2 s-1", "", "land"),
    "jmax25_c3": ("c3_jmax_at_25C", "C3 Acclimated Electron Transport Jmax at 25 degC", "umol m-2 s-1", "", "land"),
    "vcmax25_c4": ("c4_vcmax_at_25C", "C4 Acclimated Rubisco Vcmax at 25 degC", "umol m-2 s-1", "", "land"),
    "jmax25_c4": ("c4_jmax_at_25C", "C4 Acclimated Electron Transport Jmax at 25 degC", "umol m-2 s-1", "", "land"),
    "vpmax25_c4": ("c4_vpmax_at_25C", "C4 Acclimated PEPC Vpmax at 25 degC", "umol m-2 s-1", "", "land"),
    "r25_leaf": ("canopy_leaf_respiration_capacity_at_25C", "Canopy-Effective Leaf Dark Respiration Capacity Rd25 at 25 degC", "umol m-2 s-1", "", "land"),
    "fapar": ("fraction_of_absorbed_photosynthetically_active_radiation", "Canopy Fraction of Absorbed PAR (1 - exp(-k*LAI))", "1", "", "land"),
    "chi_leaf": ("leaf_orientation_index", "Optimal Ross-Goudriaan Leaf Orientation Index chi_L", "1", "", "land"),
    "n_avail_fac": ("nitrogen_availability_factor", "Nitrogen Availability Sufficiency Factor phi_N", "1", "", "land"),
    "t_home": ("home_thermal_acclimation_temperature", "Multi-Year Thermal-Home Adaptation Temperature T_home", "K", "", "land"),
    "wue": ("ecosystem_water_use_efficiency", "Ecosystem Water-Use Efficiency (GPP / Transpiration)", "gC kg-1", "", "land"),
    "alb_vis": ("surface_vis_albedo", "Canopy/Surface Visible Direct Albedo", "1", "", "land"),
    "alb_nir": ("surface_nir_albedo", "Canopy/Surface Near-Infrared Direct Albedo", "1", "", "land"),
    "koppen_code": ("koppen_geiger_climate_class", "Emergent 30-Class Koppen-Geiger Climate Index", "1", "", "land"),
    "koppen_major": ("koppen_geiger_major_group", "Emergent 5-Group Koppen-Geiger Climate Zone", "1", "", "land"),
    # Terrestrial Carbon Cycle (when enable_land_carbon=true)
    "cVeg": ("vegetation_carbon_content", "Carbon Mass in Vegetation", "kg m-2", "", "land"),
    "cSoil": ("soil_carbon_content", "Carbon Mass in Soil Pool", "kg m-2", "", "land"),
    "npp": ("net_primary_productivity_of_biomass_expressed_as_carbon", "Net Primary Productivity", "kg m-2 s-1", "", "land"),
    "ra": ("plant_respiration_carbon_flux", "Autotrophic Respiration Carbon Flux", "kg m-2 s-1", "up", "land"),
    "rh": ("heterotrophic_respiration_carbon_flux", "Heterotrophic Respiration Carbon Flux", "kg m-2 s-1", "up", "land"),
    "nbp": ("surface_net_downward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_all_land_processes", "Net Biome Productivity", "kg m-2 s-1", "down", "land"),
    "fFire": ("surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_emission_from_fires_excluding_anthropogenic_land_use_change", "Wildfire Carbon Emission Flux", "kg m-2 s-1", "up", "land"),
    # Ocean Overturning Streamfunction & Poleward Heat Transport
    "msftmz_atl": ("ocean_meridional_overturning_volume_streamfunction", "Atlantic Meridional Overturning Streamfunction (AMOC)", "Sv", "", "ocean"),
    "msftmz_glob": ("ocean_meridional_overturning_volume_streamfunction", "Global Meridional Overturning Streamfunction", "Sv", "", "ocean"),
    "oht_atl": ("northward_ocean_heat_transport", "Atlantic Northward Ocean Heat Transport", "PW", "", "ocean"),
    "oht_glob": ("northward_ocean_heat_transport", "Global Northward Ocean Heat Transport", "PW", "", "ocean"),
    # OSSP Spatial Teacher-Student Shadow Divergence Maps (Host Atmosphere & Land Surface)
    "shadow_atm_dthm_k": ("ossp_shadow_atm_theta_m_rmse", "Pre-Update OSSP CRM Shadow Column RMSE of theta_m", "K", "", "atmos"),
    "shadow_atm_dprec_mmd": ("ossp_shadow_atm_precip_mae", "Pre-Update OSSP CRM Shadow Cell-Mean Precipitation |dP|", "mm day-1", "", "atmos"),
    "shadow_atm_unc": ("ossp_shadow_atm_epistemic_uncertainty", "OSSP CRM Ensemble Epistemic Uncertainty", "1", "", "atmos"),
    "shadow_rad_dolr_wm2": ("ossp_shadow_rad_olr_mae", "Pre-Update OSSP Radiation Shadow |dOLR|", "W m-2", "", "atmos"),
    "shadow_rad_bias_olr_wm2": ("ossp_shadow_rad_olr_bias", "Pre-Update OSSP Radiation Shadow OLR Bias (Student - Physical)", "W m-2", "", "atmos"),
    "shadow_land_dthm0_k": ("ossp_shadow_land_theta_m0_rmse", "Pre-Update OSSP Land-Tile Shadow RMSE of Lowest-Layer theta_m0", "K", "", "land"),
    "shadow_land_dprec_mmd": ("ossp_shadow_land_precip_rmse", "Pre-Update OSSP Land-Tile Shadow RMSE of Tile Precipitation", "mm day-1", "", "land"),
    "shadow_land_dlwd_wm2": ("ossp_shadow_land_lwdn_mae", "Pre-Update OSSP Land-Tile Shadow |dLW_dn|", "W m-2", "", "land"),
    "shadow_land_dswd_wm2": ("ossp_shadow_land_swdn_mae", "Pre-Update OSSP Land-Tile Shadow |dSW_dn|", "W m-2", "", "land"),
    "shadow_teacher_visits": ("ossp_shadow_teacher_visit_count", "Monthly Teacher Evaluation Visit Count per Host Cell", "1", "", "atmos"),
}

CMIP7_PLEV19_VARS: tuple[str, ...] = ("ta", "ua", "va", "wap", "zg", "hus", "hur", "cl")
CMIP7_OCEAN_2D_VARS: tuple[str, ...] = ("tos", "sos", "zos", "mlotst", "siconc", "sithick", "sivol", "sisnthick")
CMIP7_OCEAN_3D_VARS: tuple[str, ...] = ("thetao", "so", "uo", "vo", "wo")
CMIP7_SOIL_3D_VARS: tuple[str, ...] = ("tsl",)


def _var_attrs(name: str, *, cell_methods: str = "area: mean time: mean") -> dict[str, str]:
    std_name, long_name, units, positive, realm = CMIP7_META[name]
    attrs = {
        "standard_name": std_name,
        "long_name": long_name,
        "units": units,
        "realm": realm,
        "cell_methods": cell_methods,
    }
    if positive:
        attrs["positive"] = positive
    return attrs


def interp_model_to_plev19(
    field_3d: torch.Tensor,
    pres_c: torch.Tensor,
    p_sfc: torch.Tensor,
    plev_pa: tuple[float, ...] = CMIP7_PLEV19_PA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized log-pressure interpolation `[nz, nc] -> ([n_plev, nc], valid_mask[n_plev, nc])`.

    Masks target pressure levels below the surface (`p_target > p_sfc`) or above model top
    (`p_target < pres_c[-1]`) as invalid (`valid_mask = False`, value = 0.0) so monthly means
    follow CMIP7 NaN-masking rules accurately.
    """
    dev = field_3d.device
    dtype = field_3d.dtype
    nz, nc = field_3d.shape
    p_tgt = torch.tensor(plev_pa, device=dev, dtype=dtype)  # [np]
    n_p = p_tgt.shape[0]

    # Sort ascending in -ln(p) (model k=0 is surface high p -> low -ln(p); k=nz-1 is top -> high -ln(p))
    x_src = -torch.log(torch.clamp(pres_c, min=1.0)).transpose(0, 1).contiguous()  # [nc, nz]
    f_src = field_3d.transpose(0, 1).contiguous()  # [nc, nz]
    x_tgt = -torch.log(torch.clamp(p_tgt, min=1.0)).unsqueeze(0).expand(nc, n_p).contiguous()  # [nc, np]

    idx_hi = torch.searchsorted(x_src, x_tgt).clamp(1, nz - 1)
    idx_lo = idx_hi - 1
    x0 = torch.gather(x_src, 1, idx_lo)
    x1 = torch.gather(x_src, 1, idx_hi)
    f0 = torch.gather(f_src, 1, idx_lo)
    f1 = torch.gather(f_src, 1, idx_hi)
    w = ((x_tgt - x0) / torch.clamp(x1 - x0, min=1.0e-6)).clamp(0.0, 1.0)
    f_interp = ((1.0 - w) * f0 + w * f1).transpose(0, 1).contiguous()  # [np, nc]

    p_col = p_tgt.unsqueeze(1)  # [np, 1]
    valid = (p_col <= p_sfc.unsqueeze(0)) & (p_col >= pres_c[-1:].expand(n_p, nc))
    return torch.where(valid, f_interp, torch.zeros_like(f_interp)), valid


def export_cmip7_fx_xarray(model: "AOBAState") -> xr.Dataset:
    """Export CMIP7 fixed grid variables (`areacella`, `sftlf`, `sftgif`, `orog`, `deptho`)."""
    mesh = model.mesh
    area = mesh.area_cell.detach().cpu().numpy().astype(np.float32)
    st = mesh.surf_type.detach().cpu().numpy()
    sftlf = np.where(st != SURF_OCEAN, 100.0, 0.0).astype(np.float32)
    sftgif = np.where(st == SURF_ICESHEET, 100.0, 0.0).astype(np.float32)
    orog = mesh.z_topo.detach().cpu().numpy().astype(np.float32)

    data_vars: dict[str, Any] = {
        "areacella": (("n_cell",), area, {"standard_name": "cell_area", "units": "m2", "realm": "atmos"}),
        "sftlf": (("n_cell",), sftlf, {"standard_name": "land_area_fraction", "units": "%", "realm": "atmos"}),
        "sftgif": (("n_cell",), sftgif, {"standard_name": "land_ice_area_fraction", "units": "%", "realm": "landIce"}),
        "orog": (("n_cell",), orog, {"standard_name": "surface_altitude", "units": "m", "realm": "atmos"}),
    }
    coords: dict[str, Any] = {
        "lat_cell": ("n_cell", np.rad2deg(mesh.lat_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_north"}),
        "lon_cell": ("n_cell", np.rad2deg(mesh.lon_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_east"}),
    }
    if model.ogcm is not None:
        ocn_mesh = model.ocn_mesh if model.ocn_mesh is not None else mesh
        ocn_dim = "n_cell_ocn" if model.ocn_mesh is not None else "n_cell"
        h_col = model.ogcm.h_col.detach().cpu().numpy().astype(np.float32)
        wet0 = model.ogcm.wet_c[0].detach().cpu().numpy()
        deptho = np.where(wet0, h_col, np.nan).astype(np.float32)
        data_vars["deptho"] = (
            (ocn_dim,), deptho,
            {"standard_name": "sea_floor_depth_below_geoid", "units": "m", "realm": "ocean"},
        )
        data_vars["areacello"] = (
            (ocn_dim,), ocn_mesh.area_cell.detach().cpu().numpy().astype(np.float32),
            {"standard_name": "cell_area", "units": "m2", "realm": "ocean"},
        )
        if model.ocn_mesh is not None:
            coords["lat_cell_ocn"] = ("n_cell_ocn", np.rad2deg(ocn_mesh.lat_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_north"})
            coords["lon_cell_ocn"] = ("n_cell_ocn", np.rad2deg(ocn_mesh.lon_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_east"})

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "Conventions": "CF-1.11 UGRID-1.0",
            "cmip7_dreq": "v1.2 ESM-BCV fx",
            "glevel": model.cfg.glevel,
            "ocean_glevel": model.cfg.ocean_glevel if model.cfg.ocean_glevel > 0 else model.cfg.glevel,
        },
    )


def _diagnose_tas_cell(model: "AOBAState") -> torch.Tensor:
    """Fast 2-m air temperature diagnostic `[nc]` [K] for daily `tasmax`/`tasmin` & monthly `tas`."""
    from aoba.coupled_model import resolve_crm_sfc_tiles

    _, _, temp0 = diagnose_pressure_temperature(
        model.atmos.rho_d[0:1], model.atmos.theta_m[0:1], model.atmos.q_v[0:1]
    )
    if model.sfc_sub is not None:
        ss = model.sfc_sub
        n_sub = resolve_crm_sfc_tiles(model.cfg)
        t_skin = torch.where(ss.surf_type == SURF_OCEAN, ss.t_ocean_ml, ss.t_canopy).reshape(model.mesh.n_cells, n_sub).mean(dim=-1)
    else:
        t_skin = torch.where(model.mesh.surf_type == SURF_OCEAN, model.sfc.t_ocean_ml, model.sfc.t_canopy)
    return 0.35 * t_skin + 0.65 * temp0[0]


def sample_cmip7_fields(
    model: "AOBAState",
    *,
    include_plev19: bool = True,
    include_ocean_3d: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Sample all instantaneous CMIP7 ESM-BCV variables on device.

    Returns
    -------
    fields : dict[str, Tensor]
        Instantaneous field values (masked entries set to 0.0 for masked sums).
    valid_masks : dict[str, Tensor]
        Optional boolean/float valid masks (present for `plev19` and ocean/land masked variables).
    """
    mesh = model.mesh
    atmos = model.atmos
    crm = model.crm
    sfc = model.sfc
    rc = model.rad_cache
    nc = mesh.n_cells
    nz, _, ny, nx = crm.rho_d.shape

    def _s2c(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(nc, -1).mean(dim=-1)

    pres_c, _, temp_c = diagnose_pressure_temperature(atmos.rho_d, atmos.theta_m, atmos.q_v)
    u_c, v_c = reconstruct_cell_winds(mesh, atmos.u_edge)

    dz0 = atmos.z_c[0] - atmos.z_w[0]
    p_sfc = pres_c[0] + C.GRAVITY * atmos.rho_d[0] * (1.0 + atmos.q_v[0]) * dz0
    tv0 = temp_c[0] * (1.0 + C.ZVIR * atmos.q_v[0])
    # WMO hypsometric sea-level pressure reduction
    psl = p_sfc * torch.exp((C.GRAVITY * atmos.z_w[0]) / (C.R_D * torch.clamp(tv0 + 0.5 * 0.0065 * atmos.z_w[0], min=180.0)))

    t_skin = torch.where(mesh.surf_type == SURF_OCEAN, sfc.t_ocean_ml, sfc.t_canopy)
    tas = 0.35 * t_skin + 0.65 * temp_c[0]
    qsat0 = torch.clamp(qsat_mixed(temp_c[0], pres_c[0])[0], min=1.0e-9)
    hurs = torch.clamp(100.0 * atmos.q_v[0] / qsat0, 0.0, 100.0)

    # Column water vapor & cloud paths + CRM subcolumn cloud fraction
    dz_4 = atmos.dz_c[:, :, None, None]
    prw = (atmos.rho_d * atmos.q_v * atmos.dz_c).sum(dim=0)
    lwp_sub = (crm.rho_d * crm.q_c * dz_4).sum(dim=0)
    iwp_sub = (crm.rho_d * crm.q_i * dz_4).sum(dim=0)
    clwvi = (lwp_sub + iwp_sub).reshape(nc, -1).mean(dim=-1)
    clivi = iwp_sub.reshape(nc, -1).mean(dim=-1)
    clt = (((lwp_sub + iwp_sub) > 1.0e-4).to(tas.dtype).reshape(nc, -1).mean(dim=-1)) * 100.0

    # Precipitation (total, convective from strong CRM updraft subcolumns, snowfall)
    rain = rc.get("rain", torch.zeros_like(tas))
    snow = rc.get("snow", torch.zeros_like(tas))
    pr = rain + snow
    if "rain_sub" in rc and "snow_sub" in rc:
        p_sub = (rc["rain_sub"] + rc["snow_sub"]).reshape(nc, -1)
        w_max_sub = crm.w.max(dim=0).values.reshape(nc, -1)
        if w_max_sub.shape[1] != p_sub.shape[1] and (w_max_sub.shape[1] % p_sub.shape[1] == 0):
            w_max_sub = w_max_sub.reshape(nc, p_sub.shape[1], -1).max(dim=-1).values
        elif w_max_sub.shape[1] != p_sub.shape[1]:
            w_max_sub = w_max_sub.max(dim=-1, keepdim=True).values.expand_as(p_sub)
        prc = (p_sub * (w_max_sub > 0.5).to(tas.dtype)).mean(dim=-1)
    else:
        prc = torch.zeros_like(pr)

    evap = rc.get("evap", torch.zeros_like(tas))
    hfss = rc.get("shf", torch.zeros_like(tas))
    olr = rc.get("olr", torch.zeros_like(tas))
    asr = rc.get("asr", torch.zeros_like(tas))
    rsdt = model.last_s_toa if model.last_s_toa is not None else torch.zeros_like(tas)
    rsut = torch.clamp(rsdt - asr, min=0.0)

    sw1_dir = rc.get("sw1_dir", torch.zeros_like(tas))
    sw1_dif = rc.get("sw1_dif", torch.zeros_like(tas))
    sw2_dir = rc.get("sw2_dir", torch.zeros_like(tas))
    sw2_dif = rc.get("sw2_dif", torch.zeros_like(tas))
    rsds = sw1_dir + sw1_dif + sw2_dir + sw2_dif
    rlds = rc.get("lw_dn", torch.zeros_like(tas))

    sd = model.last_sfc_diag_sub
    if sd is not None:
        hfls = _s2c(sd.lhf)
        # AOBA tau_x/tau_y is drag acting on atmosphere; CMIP7 tauu/tauv is downward stress onto surface
        tauu = -_s2c(sd.tau_x)
        tauv = -_s2c(sd.tau_y)
        sw1_d_s = rc.get("sw1_dir_sub", torch.zeros_like(sd.lhf))
        sw1_f_s = rc.get("sw1_dif_sub", torch.zeros_like(sd.lhf))
        sw2_d_s = rc.get("sw2_dir_sub", torch.zeros_like(sd.lhf))
        sw2_f_s = rc.get("sw2_dif_sub", torch.zeros_like(sd.lhf))
        rsus = _s2c(
            sw1_d_s * sd.alb_vis_dir + sw1_f_s * sd.alb_vis_dif
            + sw2_d_s * sd.alb_nir_dir + sw2_f_s * sd.alb_nir_dif
        )
        lw_d_s = rc.get("lw_dn_sub", torch.zeros_like(sd.lhf))
        rlus = _s2c(
            sd.emiss_sfc * C.STEFAN_BOLTZMANN * (torch.clamp(sd.t_rad_sfc, min=150.0) ** 4)
            + (1.0 - sd.emiss_sfc) * lw_d_s
        )
        f_ice = _s2c(sd.f_seaice)
        f_snow = _s2c(sd.f_snow_land)
        mrro = _s2c(sd.runoff_total)
        mrros = _s2c(sd.runoff_ice)
        gpp_c = _s2c(sd.gpp) * 12.011e-9  # umol CO2 m^-2 s^-1 -> kg C m^-2 s^-1
        rleaf_c = (_s2c(sd.resp_leaf) * 12.011e-9) if sd.resp_leaf is not None else torch.zeros_like(tas)
        an_net_c = (_s2c(sd.an_net) * 12.011e-9) if sd.an_net is not None else (gpp_c - rleaf_c)
        tran_c = _s2c(sd.transp) if sd.transp is not None else torch.zeros_like(tas)
    else:
        hfls = evap * C.L_V0
        tauu = torch.zeros_like(tas)
        tauv = torch.zeros_like(tas)
        rsus = torch.zeros_like(tas)
        rlus = C.STEFAN_BOLTZMANN * (torch.clamp(t_skin, min=150.0) ** 4)
        f_ice = torch.zeros_like(tas)
        f_snow = torch.zeros_like(tas)
        mrro = torch.zeros_like(tas)
        mrros = torch.zeros_like(tas)
        gpp_c = rc.get("gpp", torch.zeros_like(tas)) * 12.011e-9
        rleaf_c = torch.zeros_like(tas)
        an_net_c = gpp_c
        tran_c = torch.zeros_like(tas)

    fields: dict[str, torch.Tensor] = {
        "tas": tas,
        "ts": t_skin,
        "ps": p_sfc,
        "psl": psl,
        "huss": atmos.q_v[0],
        "hurs": hurs,
        "uas": u_c[0],
        "vas": v_c[0],
        "sfcWind": torch.hypot(u_c[0], v_c[0]),
        "pr": pr,
        "prc": prc,
        "prsn": snow,
        "evspsbl": evap,
        "prw": prw,
        "clwvi": clwvi,
        "clivi": clivi,
        "clt": clt,
        "hfss": hfss,
        "hfls": hfls,
        "tauu": tauu,
        "tauv": tauv,
        "rsdt": rsdt,
        "rsut": rsut,
        "rlut": olr,
        "rsds": rsds,
        "rsus": rsus,
        "rlds": rlds,
        "rlus": rlus,
    }
    valid_masks: dict[str, torch.Tensor] = {}

    # 3D Atmosphere on CMIP7 plev19
    if include_plev19:
        w_center = 0.5 * (atmos.w[:-1] + atmos.w[1:])
        wap_m = -atmos.rho_d * (1.0 + atmos.q_n + atmos.q_p) * C.GRAVITY * w_center
        qsat_3d = torch.clamp(qsat_mixed(temp_c, pres_c)[0], min=1.0e-9)
        hur_3d = torch.clamp(100.0 * atmos.q_v / qsat_3d, 0.0, 100.0)
        cl_3d = (((crm.q_c + crm.q_i) > 1.0e-5).to(tas.dtype).reshape(atmos.q_c.shape[0], nc, -1).mean(dim=-1)) * 100.0
        for name, src in (
            ("ta", temp_c),
            ("ua", u_c),
            ("va", v_c),
            ("wap", wap_m),
            ("zg", atmos.z_c),
            ("hus", atmos.q_v),
            ("hur", hur_3d),
            ("cl", cl_3d),
        ):
            vals, vmask = interp_model_to_plev19(src, pres_c, p_sfc)
            fields[name] = vals
            valid_masks[name] = vmask.to(tas.dtype)

    # Ocean 2D & 3D + Sea Ice (supports independent fine ocean mesh `ocn_mesh` cleanly!)
    is_ocn = (mesh.surf_type == SURF_OCEAN)
    is_ocn_f = is_ocn.to(tas.dtype)
    ocn_mesh = model.ocn_mesh if model.ocn_mesh is not None else mesh
    ocn_sfc = model.ocn_sfc if model.ocn_sfc is not None else sfc
    is_ocn_o = (ocn_mesh.surf_type == SURF_OCEAN)
    is_ocn_of = is_ocn_o.to(tas.dtype)
    if model.ogcm is not None:
        og = model.ogcm
        wet0_f = og.wet_c[0].to(tas.dtype)
        fields["tos"] = (og.temp[0] - C.T_FREEZE) * wet0_f
        fields["sos"] = og.sal[0] * wet0_f
        fields["zos"] = og.eta * wet0_f
        fields["mlotst"] = (og.h_bl if og.h_bl is not None else og.h_c[0]) * wet0_f
        hfds_raw = (rsds - rsus) + (rlds - rlus) - hfss - hfls
        fields["hfds"] = hfds_raw * is_ocn_f
        for k in ("tos", "sos", "zos", "mlotst"):
            valid_masks[k] = wet0_f
        valid_masks["hfds"] = is_ocn_f
        if include_ocean_3d:
            wet_cf = og.wet_c.to(tas.dtype)
            uo_c, vo_c = reconstruct_cell_winds(ocn_mesh, og.u_edge * og.wet_e.to(og.u_edge.dtype))
            wo_c = 0.5 * (og.w_vel[:-1] + og.w_vel[1:])
            fields["thetao"] = (og.temp - C.T_FREEZE) * wet_cf
            fields["so"] = og.sal * wet_cf
            fields["uo"] = uo_c * wet_cf
            fields["vo"] = vo_c * wet_cf
            fields["wo"] = wo_c * wet_cf
            for k in CMIP7_OCEAN_3D_VARS:
                valid_masks[k] = wet_cf
    else:
        fields["tos"] = (ocn_sfc.t_ocean_ml - C.T_FREEZE) * is_ocn_of
        valid_masks["tos"] = is_ocn_of

    # Sea Ice (evaluated on native ocean mesh `ocn_sfc` when fine ocean is active)
    from aoba.surface import MC_SEAICE_CRIT

    f_ice_o = torch.clamp(torch.sqrt(torch.clamp(ocn_sfc.m_seaice, min=0.0) / MC_SEAICE_CRIT), 0.0, 1.0)
    sivol = (torch.clamp(ocn_sfc.m_seaice, min=0.0) / C.RHO_SEAICE) * is_ocn_of
    siconc = f_ice_o * 100.0 * is_ocn_of
    has_ice = is_ocn_o & (f_ice_o > 1.0e-3)
    sithick = torch.where(has_ice, sivol / torch.clamp(f_ice_o, min=1.0e-3), torch.zeros_like(sivol))
    sisnthick = torch.where(has_ice, (torch.clamp(ocn_sfc.swe_seaice, min=0.0) / RHO_SNOWPACK) / torch.clamp(f_ice_o, min=1.0e-3), torch.zeros_like(sivol))
    fields["siconc"] = siconc
    fields["sivol"] = sivol
    fields["sithick"] = sithick
    fields["sisnthick"] = sisnthick
    valid_masks["siconc"] = is_ocn_of
    valid_masks["sivol"] = is_ocn_of
    valid_masks["sithick"] = has_ice.to(tas.dtype)
    valid_masks["sisnthick"] = has_ice.to(tas.dtype)

    # Land, Snow & Biosphere
    is_terr = (mesh.surf_type != SURF_OCEAN)
    is_land = (mesh.surf_type == SURF_LAND)
    is_terr_f = is_terr.to(tas.dtype)
    is_land_f = is_land.to(tas.dtype)
    dz_soil = torch.tensor(SOIL_LAYER_DEPTHS, device=tas.device, dtype=tas.dtype).unsqueeze(1)
    soil_mass = (sfc.w_liq_soil + sfc.w_ice_soil) * dz_soil * 1000.0  # [n_soil, nc] [kg m^-2]
    fields["snc"] = torch.clamp(f_snow, 0.0, 1.0) * 100.0 * is_terr_f
    fields["snw"] = torch.clamp(sfc.swe_land, min=0.0) * is_terr_f
    fields["mrso"] = soil_mass.sum(dim=0) * is_terr_f
    # Exact 0-10 cm upper soil moisture (Layer 0 = 0-5 cm + top 5 cm of Layer 1 = 5-20 cm)
    mrsos_10cm = soil_mass[0] + (0.05 / SOIL_LAYER_DEPTHS[1]) * soil_mass[1]
    fields["mrsos"] = mrsos_10cm * is_terr_f
    fields["mrro"] = mrro * is_terr_f
    fields["mrros"] = mrros * is_terr_f
    fields["tsl"] = sfc.t_soil * is_terr_f.unsqueeze(0)
    bio = sfc.bio
    fc4 = torch.clamp(bio.f_c4, 0.0, 1.0)
    fevg = torch.clamp(bio.f_evg, 0.0, 1.0) if bio.f_evg is not None else torch.zeros_like(fc4)
    lai_evg = bio.lai_evg if bio.lai_evg is not None else (fevg * bio.lai)
    lai_dec = bio.lai_dec if bio.lai_dec is not None else ((1.0 - fevg) * bio.lai)
    vcmax_eff = (1.0 - fc4) * bio.vcmax25_c3 + fc4 * bio.vcmax25_c4
    jmax_eff = (1.0 - fc4) * bio.jmax25_c3 + fc4 * bio.jmax25_c4
    r25_eff = (1.0 - fc4) * bio.r25_c3 + fc4 * bio.r25_c4

    fields["gpp"] = gpp_c * is_land_f
    fields["lai"] = bio.lai * is_land_f
    fields["tran"] = tran_c * is_land_f
    fields["rleaf"] = rleaf_c * is_land_f
    fields["an_net"] = an_net_c * is_land_f
    fields["hcanopy"] = bio.h_canopy * is_land_f
    fields["sroot"] = bio.s_root_cap * is_land_f
    fields["fc4"] = fc4 * is_land_f
    fields["lai_c3"] = ((1.0 - fc4) * bio.lai) * is_land_f
    fields["lai_c4"] = (fc4 * bio.lai) * is_land_f
    fields["fevg"] = fevg * is_land_f
    fields["lai_evg"] = lai_evg * is_land_f
    fields["lai_dec"] = lai_dec * is_land_f
    fields["vcmax25"] = vcmax_eff * is_land_f
    fields["jmax25"] = jmax_eff * is_land_f
    fields["vcmax25_c3"] = bio.vcmax25_c3 * is_land_f
    fields["jmax25_c3"] = bio.jmax25_c3 * is_land_f
    fields["vcmax25_c4"] = bio.vcmax25_c4 * is_land_f
    fields["jmax25_c4"] = bio.jmax25_c4 * is_land_f
    fields["vpmax25_c4"] = bio.vpmax25_c4 * is_land_f
    fields["r25_leaf"] = r25_eff * is_land_f
    fields["fapar"] = (1.0 - torch.exp(-0.5 * torch.clamp(bio.lai, min=0.0))) * is_land_f
    fields["chi_leaf"] = (bio.chi_leaf if bio.chi_leaf is not None else torch.zeros_like(bio.lai)) * is_land_f
    fields["n_avail_fac"] = (bio.n_avail_fac if bio.n_avail_fac is not None else torch.ones_like(bio.lai)) * is_land_f
    fields["t_home"] = (bio.ema_temp_home if bio.ema_temp_home is not None else bio.ema_temp_mid) * is_land_f
    has_tran = is_land & (tran_c > 1.0e-9)
    fields["wue"] = torch.where(has_tran, (gpp_c * 1.0e3) / torch.clamp(tran_c, min=1.0e-9), torch.zeros_like(gpp_c))
    if sd is not None:
        fields["alb_vis"] = _s2c(sd.alb_vis_dir) * is_land_f
        fields["alb_nir"] = _s2c(sd.alb_nir_dir) * is_land_f

    if model.land_carbon is not None:
        lc = model.land_carbon
        def _lc2c(t: torch.Tensor) -> torch.Tensor:
            if t.shape[0] == nc:
                return t
            if t.shape[0] % nc == 0:
                return t.reshape(nc, -1).mean(dim=-1)
            return t[:nc]
        c_veg = _lc2c(lc.c_leaf + lc.c_wood + lc.c_root)
        c_soil_tot = _lc2c(lc.c_soil.sum(dim=0) if lc.c_soil.ndim == 2 else lc.c_soil) + _lc2c(lc.c_lit_fast + lc.c_lit_slow)
        fields["cVeg"] = c_veg * is_land_f
        fields["cSoil"] = c_soil_tot * is_land_f
        fields["npp"] = _lc2c(lc.npp_kgc_m2s) * is_land_f
        fields["ra"] = _lc2c(lc.ra_kgc_m2s) * is_land_f
        fields["rh"] = _lc2c(lc.rh_kgc_m2s) * is_land_f
        fields["nbp"] = _lc2c(lc.nbp_kgc_m2s) * is_land_f
        fields["fFire"] = _lc2c(lc.fire_c_kgc_m2s) * is_land_f
        for k in ("cVeg", "cSoil", "npp", "ra", "rh", "nbp", "fFire"):
            valid_masks[k] = is_land_f

    for k in ("snc", "snw", "mrso", "mrsos", "mrro", "mrros"):
        valid_masks[k] = is_terr_f
    valid_masks["tsl"] = is_terr_f.unsqueeze(0).expand_as(sfc.t_soil)
    for k in (
        "gpp", "lai", "tran", "rleaf", "an_net", "hcanopy", "sroot",
        "fc4", "lai_c3", "lai_c4", "fevg", "lai_evg", "lai_dec",
        "vcmax25", "jmax25", "vcmax25_c3", "jmax25_c3",
        "vcmax25_c4", "jmax25_c4", "vpmax25_c4", "r25_leaf",
        "fapar", "chi_leaf", "n_avail_fac", "t_home",
    ):
        valid_masks[k] = is_land_f
    valid_masks["wue"] = has_tran.to(tas.dtype)
    if sd is not None:
        valid_masks["alb_vis"] = is_land_f
        valid_masks["alb_nir"] = is_land_f

    return fields, valid_masks


@dataclass
class CMIP7MonthlyAccumulator:
    """Accumulates CMIP7 monthly mean fields (with NaN-safe masking) and daily `tasmax`/`tasmin`."""

    include_plev19: bool = True
    include_ocean_3d: bool = True
    sums: dict[str, torch.Tensor] = field(default_factory=dict)
    counts: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_count: int = 0
    day_tas_max: torch.Tensor | None = None
    day_tas_min: torch.Tensor | None = None
    tasmax_sum: torch.Tensor | None = None
    tasmin_sum: torch.Tensor | None = None
    day_extrema_count: int = 0

    def update_step_extrema(self, model: "AOBAState", *, end_of_day: bool) -> None:
        """Track daily max/min of 2-m air temperature for CMIP7 `tasmax` and `tasmin`."""
        tas = _diagnose_tas_cell(model)
        if self.day_tas_max is None:
            self.day_tas_max = tas.clone()
            self.day_tas_min = tas.clone()
        else:
            self.day_tas_max = torch.maximum(self.day_tas_max, tas)
            self.day_tas_min = torch.minimum(self.day_tas_min, tas)
        if end_of_day:
            if self.tasmax_sum is None:
                self.tasmax_sum = self.day_tas_max.clone()
                self.tasmin_sum = self.day_tas_min.clone()
            else:
                self.tasmax_sum += self.day_tas_max
                self.tasmin_sum += self.day_tas_min
            self.day_extrema_count += 1
            self.day_tas_max = None
            self.day_tas_min = None

    def sample(self, model: "AOBAState") -> None:
        """Add one instantaneous sample of all CMIP7 fields to the active monthly window."""
        fields, masks = sample_cmip7_fields(
            model,
            include_plev19=self.include_plev19,
            include_ocean_3d=self.include_ocean_3d,
        )
        if self.sample_count == 0:
            self.sums = {k: v.clone() for k, v in fields.items()}
            self.counts = {k: m.clone() for k, m in masks.items()}
        else:
            for k, v in fields.items():
                if k in self.sums:
                    self.sums[k] += v
                else:
                    self.sums[k] = v.clone()
            for k, m in masks.items():
                if k in self.counts:
                    self.counts[k] += m
                else:
                    self.counts[k] = m.clone()
        self.sample_count += 1

    def to_xarray(self, model: "AOBAState") -> xr.Dataset:
        """Finalize the current monthly window into a CF-1.11 / UGRID-1.0 / CMIP7 `xarray.Dataset`."""
        mesh = model.mesh
        t_days = np.array([model.sim_time_sec / C.SECONDS_PER_DAY], dtype=np.float64)
        inv = 1.0 / float(max(self.sample_count, 1))

        ocn_cell_dim = "n_cell_ocn" if model.ocn_mesh is not None else "n_cell"
        data_vars: dict[str, Any] = {}
        for name, s_arr in self.sums.items():
            if name in self.counts:
                cnt = self.counts[name]
                mean_t = torch.where(cnt > 0.0, s_arr / torch.clamp(cnt, min=1.0), torch.full_like(s_arr, float("nan")))
            else:
                mean_t = s_arr * inv
            arr_np = mean_t.detach().cpu().numpy().astype(np.float32)

            if name in CMIP7_PLEV19_VARS:
                dims = ("time", "plev", "n_cell")
                arr_out = arr_np[None, :, :]
            elif name in CMIP7_OCEAN_3D_VARS:
                dims = ("time", "lev", ocn_cell_dim)
                arr_out = arr_np[None, :, :]
            elif name in CMIP7_OCEAN_2D_VARS:
                dims = ("time", ocn_cell_dim)
                arr_out = arr_np[None, :]
            elif name in CMIP7_SOIL_3D_VARS:
                dims = ("time", "depth_soil", "n_cell")
                arr_out = arr_np[None, :, :]
            else:
                dims = ("time", "n_cell")
                arr_out = arr_np[None, :]
            data_vars[name] = (dims, arr_out, _var_attrs(name))

        # Monthly mean of daily max/min near-surface temperature (`tasmax`, `tasmin`)
        if self.tasmax_sum is not None and self.day_extrema_count > 0:
            inv_e = 1.0 / float(self.day_extrema_count)
            data_vars["tasmax"] = (
                ("time", "n_cell"),
                (self.tasmax_sum * inv_e).detach().cpu().numpy().astype(np.float32)[None, :],
                _var_attrs("tasmax", cell_methods="area: mean time: maximum within days time: mean over days"),
            )
            data_vars["tasmin"] = (
                ("time", "n_cell"),
                (self.tasmin_sum * inv_e).detach().cpu().numpy().astype(np.float32)[None, :],
                _var_attrs("tasmin", cell_methods="area: mean time: minimum within days time: mean over days"),
            )
        elif "tas" in data_vars:
            data_vars["tasmax"] = (("time", "n_cell"), data_vars["tas"][1].copy(), _var_attrs("tasmax"))
            data_vars["tasmin"] = (("time", "n_cell"), data_vars["tas"][1].copy(), _var_attrs("tasmin"))

        # Emergent Koppen-Geiger climate classification
        from aoba.coupled_model import diagnose_model_koppen_geiger

        kg30, kg5 = diagnose_model_koppen_geiger(model)
        data_vars["koppen_code"] = (("time", "n_cell"), kg30.detach().cpu().numpy()[None, :], _var_attrs("koppen_code", cell_methods="area: mean time: point"))
        data_vars["koppen_major"] = (("time", "n_cell"), kg5.detach().cpu().numpy()[None, :], _var_attrs("koppen_major", cell_methods="area: mean time: point"))

        # Spatial Teacher-Student Shadow Divergence Maps (when OSSP shadow diagnostics active)
        if getattr(model, "crm_distill", None) is not None or getattr(model, "rad_distill", None) is not None:
            from aoba.crm_self_distill import consume_distill_shadow_maps

            s_maps = consume_distill_shadow_maps(
                getattr(model, "crm_distill", None),
                getattr(model, "rad_distill", None),
                reset=False,
            )
            for sk, sv in s_maps.items():
                if sk in CMIP7_META:
                    data_vars[sk] = (("time", "n_cell"), sv.detach().cpu().numpy().astype(np.float32)[None, :], _var_attrs(sk))

        coords: dict[str, Any] = {
            "time": ("time", t_days, {"units": "days since 2000-03-20 00:00:00"}),
            "lat_cell": ("n_cell", np.rad2deg(mesh.lat_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_north"}),
            "lon_cell": ("n_cell", np.rad2deg(mesh.lon_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_east"}),
            "depth_soil": ("depth_soil", np.cumsum(SOIL_LAYER_DEPTHS, dtype=np.float32) - 0.5 * np.array(SOIL_LAYER_DEPTHS, dtype=np.float32), {"units": "m", "positive": "down"}),
        }
        if model.ocn_mesh is not None:
            coords["lat_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lat_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_north"})
            coords["lon_cell_ocn"] = ("n_cell_ocn", np.rad2deg(model.ocn_mesh.lon_cell.detach().cpu().numpy()).astype(np.float32), {"units": "degrees_east"})
        if self.include_plev19:
            coords["plev"] = ("plev", np.array(CMIP7_PLEV19_PA, dtype=np.float32), {"units": "Pa", "positive": "down", "axis": "Z"})
        if model.ogcm is not None:
            coords["lev"] = ("lev", model.ogcm.z_c.detach().cpu().numpy().astype(np.float32), {"units": "m", "positive": "down", "axis": "Z"})
            # Atlantic & Global Meridional Overturning Streamfunction & Poleward Ocean Heat Transport
            from aoba.ocean_ogcm import compute_moc_and_oht_profiles

            ocn_mesh_m = model.ocn_mesh if model.ocn_mesh is not None else mesh
            moc_d = compute_moc_and_oht_profiles(model.ogcm, ocn_mesh_m)
            coords["lat_moc"] = ("lat_moc", moc_d["lat_moc"], {"units": "degrees_north"})
            data_vars["msftmz_atl"] = (("time", "lat_moc", "lev"), moc_d["msftmz_atl"][None, :, :], _var_attrs("msftmz_atl"))
            data_vars["msftmz_glob"] = (("time", "lat_moc", "lev"), moc_d["msftmz_glob"][None, :, :], _var_attrs("msftmz_glob"))
            data_vars["oht_atl"] = (("time", "lat_moc"), moc_d["oht_atl"][None, :], _var_attrs("oht_atl"))
            data_vars["oht_glob"] = (("time", "lat_moc"), moc_d["oht_glob"][None, :], _var_attrs("oht_glob"))

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "Conventions": "CF-1.11 UGRID-1.0",
                "cmip7_dreq": "v1.2 ESM-BCV",
                "model": "AOBA (Autonomous Optimality Biosphere-Atmosphere)",
                "glevel": model.cfg.glevel,
                "ocean_glevel": model.cfg.ocean_glevel if model.cfg.ocean_glevel > 0 else model.cfg.glevel,
                "co2_ppmv": float(model.current_co2_ppmv()),
            },
        )

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()
        self.sample_count = 0
        self.day_tas_max = None
        self.day_tas_min = None
        self.tasmax_sum = None
        self.tasmin_sum = None
        self.day_extrema_count = 0

    def state_dict_cpu(self) -> dict[str, Any]:
        return {
            "sums": {k: v.detach().cpu() for k, v in self.sums.items()},
            "counts": {k: v.detach().cpu() for k, v in self.counts.items()},
            "sample_count": self.sample_count,
            "day_tas_max": self.day_tas_max.detach().cpu() if self.day_tas_max is not None else None,
            "day_tas_min": self.day_tas_min.detach().cpu() if self.day_tas_min is not None else None,
            "tasmax_sum": self.tasmax_sum.detach().cpu() if self.tasmax_sum is not None else None,
            "tasmin_sum": self.tasmin_sum.detach().cpu() if self.tasmin_sum is not None else None,
            "day_extrema_count": self.day_extrema_count,
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        self.sums = {k: v.to(device=device) for k, v in (state.get("sums") or {}).items()}
        self.counts = {k: v.to(device=device) for k, v in (state.get("counts") or {}).items()}
        self.sample_count = int(state.get("sample_count", 0))
        self.day_tas_max = state["day_tas_max"].to(device=device) if state.get("day_tas_max") is not None else None
        self.day_tas_min = state["day_tas_min"].to(device=device) if state.get("day_tas_min") is not None else None
        self.tasmax_sum = state["tasmax_sum"].to(device=device) if state.get("tasmax_sum") is not None else None
        self.tasmin_sum = state["tasmin_sum"].to(device=device) if state.get("tasmin_sum") is not None else None
        self.day_extrema_count = int(state.get("day_extrema_count", 0))
