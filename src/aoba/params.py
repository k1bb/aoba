"""Unified TOML configuration, parameter registry, and runtime module binding for AOBA.

Centralizes every grid setting, external forcing, output stream schedule, physical constant,
and physics tuning parameter in a single human-readable TOML file (`config/aoba_default.toml`)
while keeping programmatic defaults and `AOBAConfig(...)` calls backward-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from aoba import biosphere as _bio
from aoba import constants as C
from aoba import gravity_wave as _gwd
from aoba import icesheet as _ice
from aoba import microphysics as _mp
from aoba import ocean_ekman as _ocn
from aoba import ocean_ogcm as _ogcm
from aoba import orography as _oro
from aoba import surface as _sfc
from aoba import turbulence as _turb
from aoba.coupled_model import AOBAConfig


def default_toml_path() -> Path:
    """Return path to the repository-level authoritative `config/aoba_default.toml`."""
    return Path(__file__).resolve().parents[2] / "config" / "aoba_default.toml"


@dataclass
class OutputConfig:
    """Output file types, temporal averaging/snapshot intervals, and partitioning settings."""

    out_dir: str = ""
    years: int = 30
    end_date: str = ""                          # Optional end date (YYYY-MM-DD); empty uses `years`
    days_per_year: float = C.DAYS_PER_YEAR
    zarr_dir_partition: str = "yearly"          # "yearly" (`year_XXXX/`) or "single"
    daily_csv_enabled: bool = True
    daily_csv_interval_days: float = 1.0
    monthly_mean_enabled: bool = True
    monthly_mean_periods_per_year: int = 12     # 12 -> monthly means; 24 -> semi-monthly, etc.
    monthly_mean_sample_hours: float = 6.0      # Sampling cadence within each averaging window
    monthly_mean_include_2d: bool = True
    monthly_mean_include_3d: bool = True
    monthly_mean_include_subgrid_2d: bool = True
    snapshots_enabled: bool = True
    snapshots_periods_per_year: int = 12        # Instantaneous snapshot cadence per year
    snapshots_include_2d: bool = True
    snapshots_include_3d: bool = True
    snapshots_include_subgrid_2d: bool = True
    restart_interval_years: int = 1
    restart_monthly: bool = True                # Save mid-year restart checkpoint at end of each month
    cmip7_output_enabled: bool = False          # Optional CMIP7 Data Request v1.2 ESM-BCV monthly Zarr stream (off during spin-up)
    cmip7_include_plev19: bool = True           # Include 3D atmospheric variables interpolated to CMIP7 plev19
    cmip7_include_ocean_3d: bool = True         # Include 3D ocean variables on model z-levels (olevel)


# Module mapping for `[section]` -> `(module_obj, module_attr_name)`
# Every constant listed here is live-synchronized into `src/aoba/*` at runtime.
_MODULE_PARAM_MAP: dict[str, dict[str, tuple[Any, str]]] = {
    "constants": {
        "earth_radius": (C, "EARTH_RADIUS"),
        "gravity": (C, "GRAVITY"),
        "omega_earth": (C, "OMEGA_EARTH"),
        "seconds_per_day": (C, "SECONDS_PER_DAY"),
        "days_per_year": (C, "DAYS_PER_YEAR"),
        "r_d": (C, "R_D"),
        "cp_d": (C, "CP_D"),
        "r_v": (C, "R_V"),
        "cp_v": (C, "CP_V"),
        "p_ref": (C, "P_REF"),
        "t_freeze": (C, "T_FREEZE"),
        "t_triple": (C, "T_TRIPLE"),
        "e_triple": (C, "E_TRIPLE"),
        "l_v0": (C, "L_V0"),
        "l_f0": (C, "L_F0"),
        "t_ice_all": (C, "T_ICE_ALL"),
        "t_liq_all": (C, "T_LIQ_ALL"),
        "stefan_boltzmann": (C, "STEFAN_BOLTZMANN"),
        "von_karman": (C, "VON_KARMAN"),
        "rho_seawater": (C, "RHO_SEAWATER"),
        "cp_seawater": (C, "CP_SEAWATER"),
        "t_freeze_ocean": (C, "T_FREEZE_OCEAN"),
        "rho_seaice": (C, "RHO_SEAICE"),
        "k_therm_seaice": (C, "K_THERM_SEAICE"),
        "k_therm_snow": (C, "K_THERM_SNOW"),
        "o2_mixing_ratio": (C, "O2_MIXING_RATIO"),
    },
    "radiation": {
        "solar_frac_sw1": (C, "SOLAR_FRAC_SW1"),
        "lw_window_frac": (C, "LW_WINDOW_FRAC"),
        "o3_g1": (C, "O3_G1"),
        "o3_g2": (C, "O3_G2"),
        "o3_g3": (C, "O3_G3"),
        "par_frac_of_sw1": (C, "PAR_FRAC_OF_SW1"),
        "umol_per_joule_par": (C, "UMOL_PER_JOULE_PAR"),
    },
    "microphysics": {
        "rho_rain": (_mp, "RHO_RAIN"),
        "rho_snow": (_mp, "RHO_SNOW"),
        "rho_graupel": (_mp, "RHO_GRAUPEL"),
        "n0_rain": (_mp, "N0_RAIN"),
        "n0_snow": (_mp, "N0_SNOW"),
        "n0_graupel": (_mp, "N0_GRAUPEL"),
        "a_fall_rain": (_mp, "A_FALL_RAIN"),
        "b_fall_rain": (_mp, "B_FALL_RAIN"),
        "a_fall_snow": (_mp, "A_FALL_SNOW"),
        "b_fall_snow": (_mp, "B_FALL_SNOW"),
        "a_fall_graupel": (_mp, "A_FALL_GRAUPEL"),
        "b_fall_graupel": (_mp, "B_FALL_GRAUPEL"),
        "e_rc": (_mp, "E_RC"),
        "e_sc": (_mp, "E_SC"),
        "e_gc": (_mp, "E_GC"),
        "e_ri": (_mp, "E_RI"),
        "e_si": (_mp, "E_SI"),
        "e_gi": (_mp, "E_GI"),
        "qc_autoconv_thresh": (_mp, "QC_AUTOCONV_THRESH"),
        "qi_aggreg_thresh": (_mp, "QI_AGGREG_THRESH"),
        "rate_autoconv": (_mp, "RATE_AUTOCONV"),
        "rate_ice_aggreg": (_mp, "RATE_ICE_AGGREG"),
        "vt_cloud_ice": (_mp, "VT_CLOUD_ICE"),
        "cloud_droplet_conc_cm3": (_mp, "CLOUD_DROPLET_CONC_CM3"),
        "ccn_ocean_base_cm3": (_mp, "CCN_OCEAN_BASE_CM3"),
        "ccn_land_base_cm3": (_mp, "CCN_LAND_BASE_CM3"),
        "ccn_dust_scale_cm3": (_mp, "CCN_DUST_SCALE_CM3"),
        "inp_dust_enhance": (_mp, "INP_DUST_ENHANCE"),
        "t0_precip": (_mp, "T0_PRECIP"),
        "t00_precip": (_mp, "T00_PRECIP"),
        "t0_graupel": (_mp, "T0_GRAUPEL"),
        "t00_graupel": (_mp, "T00_GRAUPEL"),
    },
    "turbulence": {
        "mynn_b1": (_turb, "MYNN_B1"),
        "mynn_b2": (_turb, "MYNN_B2"),
        "mynn_c2": (_turb, "MYNN_C2"),
        "mynn_c3": (_turb, "MYNN_C3"),
        "mynn_c5": (_turb, "MYNN_C5"),
        "mynn_gamma1": (_turb, "MYNN_GAMMA1"),
        "mynn_pr_n": (_turb, "MYNN_PR_N"),
        "tke_min": (_turb, "TKE_MIN"),
        "rsl_mu_m": (_turb, "RSL_MU_M"),
        "rsl_mu_h": (_turb, "RSL_MU_H"),
        "rsl_nu": (_turb, "RSL_NU"),
        "rsl_lambda": (_turb, "RSL_LAMBDA"),
        "kinematic_visc_air": (_turb, "KINEMATIC_VISC_AIR"),
    },
    "surface": {
        "cs_dry_soil": (_sfc, "CS_DRY_SOIL"),
        "kappa_dry_soil": (_sfc, "KAPPA_DRY_SOIL"),
        "soil_porosity": (_sfc, "SOIL_POROSITY"),
        "psi_sat_soil": (_sfc, "PSI_SAT_SOIL"),
        "b_clapp_soil": (_sfc, "B_CLAPP_SOIL"),
        "k_sat_soil": (_sfc, "K_SAT_SOIL"),
        "ice_impedance_omega": (_sfc, "ICE_IMPEDANCE_OMEGA"),
        "wc_max_per_lai": (_sfc, "WC_MAX_PER_LAI"),
        "alpha_intercept_liq": (_sfc, "ALPHA_INTERCEPT_LIQ"),
        "f_wet_max": (_sfc, "F_WET_MAX"),
        "dsl_dmax_mm": (_sfc, "DSL_DMAX_MM"),
        "land_eb_iterations": (_sfc, "LAND_EB_ITERATIONS"),
        "rho_snowpack": (_sfc, "RHO_SNOWPACK"),
        "swe_glacier_max": (_sfc, "SWE_GLACIER_MAX"),
        "tau_snow_age": (_sfc, "TAU_SNOW_AGE"),
        "alb_snow_vis_new": (_sfc, "ALB_SNOW_VIS_NEW"),
        "alb_snow_nir_new": (_sfc, "ALB_SNOW_NIR_NEW"),
        "alb_snow_vis_old": (_sfc, "ALB_SNOW_VIS_OLD"),
        "alb_snow_nir_old": (_sfc, "ALB_SNOW_NIR_OLD"),
        "emissivity_land": (_sfc, "EMISSIVITY_LAND"),
        "emissivity_ocean": (_sfc, "EMISSIVITY_OCEAN"),
        "mc_seaice_crit": (_sfc, "MC_SEAICE_CRIT"),
        "alb_seaice_vis": (_sfc, "ALB_SEAICE_VIS"),
        "alb_seaice_nir": (_sfc, "ALB_SEAICE_NIR"),
    },
    "biosphere": {
        "theta_nrh": (_bio, "THETA_NRH"),
        "cost_c_smith": (_bio, "COST_C_SMITH"),
        "phi_psii_ref_c3": (_bio, "PHI_PSII_REF_C3"),
        "beta_c3": (_bio, "BETA_C3"),
        "beta_c4": (_bio, "BETA_C4"),
        "gamma_star25_c3": (_bio, "GAMMA_STAR25_C3"),
        "kc25_c3": (_bio, "KC25_C3"),
        "ko25_c3": (_bio, "KO25_C3"),
        "dha_gamma": (_bio, "DHA_GAMMA"),
        "dha_kc_c3": (_bio, "DHA_KC_C3"),
        "dha_ko_c3": (_bio, "DHA_KO_C3"),
        "gamma_star25_c4": (_bio, "GAMMA_STAR25_C4"),
        "kp25_umol_c4": (_bio, "KP25_UMOL_C4"),
        "dha_kp_c4": (_bio, "DHA_KP_C4"),
        "kc25_c4": (_bio, "KC25_C4"),
        "dha_kc_c4": (_bio, "DHA_KC_C4"),
        "ko25_c4": (_bio, "KO25_C4"),
        "dha_ko_c4": (_bio, "DHA_KO_C4"),
        "phi_leak_c4": (_bio, "PHI_LEAK_C4"),
        "gbs_c4": (_bio, "GBS_C4"),
        "ha_vcmax": (_bio, "HA_VCMAX"),
        "ha_jmax": (_bio, "HA_JMAX"),
        "ha_vpmax": (_bio, "HA_VPMAX"),
        "hd_enzyme": (_bio, "HD_ENZYME"),
        "b_resp_ren": (_bio, "B_RESP_REN"),
        "leaf_scat_vis": (_bio, "LEAF_SCAT_VIS"),
        "leaf_scat_nir": (_bio, "LEAF_SCAT_NIR"),
        "kd_prime_par": (_bio, "KD_PRIME_PAR"),
        "rho_cd_par": (_bio, "RHO_CD_PAR"),
        "z_foliage_cost": (_bio, "Z_FOLIAGE_COST"),
        "k_beer_lai": (_bio, "K_BEER_LAI"),
        "sigma_zhou": (_bio, "SIGMA_ZHOU"),
        "alpha_ema_acclim": (_bio, "ALPHA_EMA_ACCLIM"),
        "evg_cost_ratio_base": (_bio, "EVG_COST_RATIO_BASE"),
        "evg_cost_ratio_cold": (_bio, "EVG_COST_RATIO_COLD"),
        "evg_cold_eff_penalty": (_bio, "EVG_COLD_EFF_PENALTY"),
        "leaf_aging_b_yr": (_bio, "LEAF_AGING_B_YR"),
        "decid_flush_days": (_bio, "DECID_FLUSH_DAYS"),
        "evg_offseason_resp25": (_bio, "EVG_OFFSEASON_RESP25"),
        "evg_frost_crit_k": (_bio, "EVG_FROST_CRIT_K"),
        "evg_frost_cost": (_bio, "EVG_FROST_COST"),
    },
    "ocean_ekman": {
        "h_thermocline": (_ocn, "H_THERMOCLINE"),
        "r_ekman_damp_days": (_ocn, "R_EKMAN_DAMP_DAYS"),
        "kappa_ocn_horiz": (_ocn, "KAPPA_OCN_HORIZ"),
        "kappa_ocn_vert": (_ocn, "KAPPA_OCN_VERT"),
        "g_prime_ocn": (_ocn, "G_PRIME_OCN"),
        "tau_sw_drag_days": (_ocn, "TAU_SW_DRAG_DAYS"),
        "enable_shallow_water_ocn": (_ocn, "ENABLE_SHALLOW_WATER_OCN"),
        "kappa_gm": (_ogcm, "KAPPA_GM"),
        "kappa_redi_horiz": (_ogcm, "KAPPA_REDI_HORIZ"),
        "enable_full_redi": (_ogcm, "ENABLE_FULL_REDI"),
        "kpp_ri_crit": (_ogcm, "KPP_RI_CRIT"),
        "kpp_cs_nonlocal": (_ogcm, "KPP_CS_NONLOCAL"),
        "shear_nu_0": (_ogcm, "SHEAR_NU_0"),
        "tidal_e0_wm2": (_ogcm, "TIDAL_E0_WM2"),
        "sw_pen_zeta2_m": (_ogcm, "SW_PEN_ZETA2_M"),
        "kappa_v_bg": (_ogcm, "KAPPA_V_BG"),
        "kappa_v_abyss": (_ogcm, "KAPPA_V_ABYSS"),
        "seaice_h_ridge_m": (_ogcm, "SEAICE_H_RIDGE_M"),
    },
    "orography": {
        "hn_crit": (_oro, "HN_CRIT"),
        "cd_block": (_oro, "CD_BLOCK"),
        "g_wave": (_oro, "G_WAVE"),
    },
    "gravity_wave": {
        "ngwd_front_thresh": (_gwd, "NGWD_FRONT_THRESH"),
        "ngwd_front_launch_hpa": (_gwd, "NGWD_FRONT_LAUNCH_HPA"),
        "ngwd_front_tau0": (_gwd, "NGWD_FRONT_TAU0"),
        "ngwd_front_cw": (_gwd, "NGWD_FRONT_CW"),
        "ngwd_front_eff": (_gwd, "NGWD_FRONT_EFF"),
        "ngwd_conv_tau0": (_gwd, "NGWD_CONV_TAU0"),
        "ngwd_conv_qref": (_gwd, "NGWD_CONV_QREF"),
        "ngwd_conv_eff": (_gwd, "NGWD_CONV_EFF"),
        "ngwd_wavelength_m": (_gwd, "NGWD_WAVELENGTH_M"),
        "ngwd_cmax": (_gwd, "NGWD_CMAX"),
        "ngwd_tend_max": (_gwd, "NGWD_TEND_MAX"),
    },
    "icesheet_physics": {
        "rho_ice": (_ice, "RHO_ICE"),
        "rho_mantle": (_ice, "RHO_MANTLE"),
        "a_glen_per_year": (_ice, "A_GLEN_PER_YEAR"),
        "h_ice_threshold_m": (_ice, "H_ICE_THRESHOLD_M"),
        "h_ice_max_m": (_ice, "H_ICE_MAX_M"),
    },
}

# Exact section keys mapping to `AOBAConfig` and `OutputConfig`
_GRID_KEYS: tuple[str, ...] = (
    "glevel", "nz", "z_top", "dz_min", "z_sponge_base", "tau_sponge_min", "terrain_scale_exp",
    "crm_nx", "crm_ny", "crm_dx", "dt_host", "n_host_dyn_substeps", "n_crm_dyn_substeps",
    "rad_stride", "crm_col_chunk", "rad_col_chunk",
)
_FORCING_KEYS: tuple[str, ...] = (
    "co2_ppmv", "co2_mode", "co2_scenario", "co2_csv_path",
    "ch4_ppmv", "n2o_ppmv", "rad_scheme",
    "o3_mode", "o3_file_path", "o3_lw_coupling", "enable_nonoro_gwd",
    "solar_constant", "omega", "obliquity_deg", "eccentricity", "perihelion_deg",
    "start_date", "calendar_type", "insolation_mode", "fixed_insolation_date", "diurnal_cycle",
    "use_earth2014", "earth2014_dir", "earth2014_stride", "enable_superparam",
    "enable_dust", "dust_emis_factor",
    "h_ml_ocean", "ocean_mode", "ocean_nz", "enable_seaice_drift", "enable_ocean_bgc",
    "re_liq_um", "re_ice_um", "cloud_sw_inhomog", "microphysics_scheme",
)
_ICESHEET_CFG_KEYS: tuple[str, ...] = (
    "enable_dynamic_icesheet", "icesheet_ema_years", "icesheet_accel_factor",
    "icesheet_accel_spinup_years", "icesheet_isostasy_tau_yr",
)
_OUTPUT_KEYS: tuple[str, ...] = tuple(OutputConfig.__dataclass_fields__.keys())


def extract_current_config_dict(
    aoba_cfg: AOBAConfig | None = None,
    out_cfg: OutputConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Snapshot all current grid, forcing, icesheet, output, and physics module constants into a dict."""
    ac = aoba_cfg or AOBAConfig()
    oc = out_cfg or OutputConfig()
    data: dict[str, dict[str, Any]] = {
        "grid": {k: getattr(ac, k) for k in _GRID_KEYS},
        "forcing": {k: getattr(ac, k) for k in _FORCING_KEYS},
        "icesheet": {k: getattr(ac, k) for k in _ICESHEET_CFG_KEYS},
        "output": {k: getattr(oc, k) for k in _OUTPUT_KEYS},
    }
    for sec, mapping in _MODULE_PARAM_MAP.items():
        data[sec] = {k: getattr(mod, attr) for k, (mod, attr) in mapping.items()}
    return data


_INITIAL_DEFAULTS = extract_current_config_dict()


def default_config_dict() -> dict[str, dict[str, Any]]:
    """Return a deep copy of the baseline default AOBA configuration dict."""
    return {sec: dict(vals) for sec, vals in _INITIAL_DEFAULTS.items()}


def merge_config_dicts(base: dict[str, dict[str, Any]], override: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Deep-merge `override` sections into a copy of `base`."""
    merged = {sec: dict(vals) for sec, vals in base.items()}
    for sec, sec_vals in override.items():
        if isinstance(sec_vals, dict):
            if sec not in merged:
                merged[sec] = {}
            for k, v in sec_vals.items():
                merged[sec][k] = v
    return merged


def load_aoba_toml(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load an AOBA TOML configuration file merged cleanly over baseline defaults."""
    raw = tomllib.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return merge_config_dicts(default_config_dict(), raw)


def _format_toml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == 0.0:
            return "0.0"
        s = repr(val)
        return s if ("." in s or "e" in s or "E" in s) else f"{s}.0"
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(x) for x in val) + "]"
    return str(val)


def dump_aoba_toml(cfg_dict: dict[str, dict[str, Any]]) -> str:
    """Serialize an AOBA configuration dictionary cleanly to standard TOML text."""
    section_titles = {
        "grid": "Horizontal, Vertical & Subgrid CRM Mesh Settings",
        "forcing": "Planetary Boundary, Orbital & Atmospheric Forcing",
        "icesheet": "Dynamic Equilibrium / Accelerated SIA Ice-Sheet Options",
        "output": "Output Files, Time Intervals (Mean vs Instantaneous) & Restart",
        "constants": "Fundamental Physical & Planetary Constants",
        "radiation": "4-Band Analytic Two-Stream Radiation Parameters",
        "microphysics": "KR03 1-Moment Mixed-Phase Cloud Microphysics Parameters",
        "turbulence": "MYNN Level 2.5 & De Ridder (2010) RSL Parameters",
        "surface": "5-Layer Soil, Snowpack, Canopy Interception & Sea-Ice Parameters",
        "biosphere": "Two-Leaf Non-Rectangular-Hyperbola EEO Biosphere Parameters",
        "ocean_ekman": "Codron (2012) 1.5-Layer Wind-Driven Voronoi Ekman Ocean Parameters",
        "orography": "Lott & Miller (1997) Subgrid Orographic Drag Parameters",
        "gravity_wave": "Map-Free Flow-Dependent Non-Orographic GWD (Beres 2004; Charron & Manzini 2002; Richter 2010)",
        "icesheet_physics": "Glen Flow Law & Lithospheric Isostasy Physics Constants",
    }
    lines = [
        "# =============================================================================",
        "# AOBA Coupled Earth System Model — Unified Configuration & Parameter Registry",
        "# Generated / edited via AOBA unified TOML & GUI configurator.",
        "# =============================================================================",
        "",
    ]
    for sec, vals in cfg_dict.items():
        title = section_titles.get(sec, sec)
        lines.append(f"# --- {title} ---")
        lines.append(f"[{sec}]")
        for k, v in vals.items():
            lines.append(f"{k} = {_format_toml_value(v)}")
        lines.append("")
    return "\n".join(lines)


def save_aoba_toml(cfg_dict: dict[str, dict[str, Any]], path: str | Path) -> Path:
    """Write `cfg_dict` to a TOML file on disk."""
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_aoba_toml(cfg_dict), encoding="utf-8")
    return out_path


def apply_module_parameters(cfg_dict: dict[str, dict[str, Any]]) -> None:
    """Apply all physics constant sections in `cfg_dict` onto `src/aoba/*` modules."""
    from aoba import coupled_model as _cm

    for sec, mapping in _MODULE_PARAM_MAP.items():
        sec_dict = cfg_dict.get(sec, {})
        for k, (mod, attr) in mapping.items():
            if k in sec_dict:
                val = sec_dict[k]
                cur = getattr(mod, attr)
                if isinstance(cur, float):
                    setattr(mod, attr, float(val))
                elif isinstance(cur, int):
                    setattr(mod, attr, int(val))
                else:
                    setattr(mod, attr, val)

    C.recompute_derived_constants()
    _mp.recompute_derived_constants()
    _turb.recompute_derived_constants()
    _ocn.recompute_derived_constants()
    _ice.recompute_derived_constants()

    # Keep direct imports inside coupled_model synchronized as well
    _cm.EMISSIVITY_LAND = _sfc.EMISSIVITY_LAND
    _cm.EMISSIVITY_OCEAN = _sfc.EMISSIVITY_OCEAN
    _cm.TKE_MIN = _turb.TKE_MIN
    _cm.SWE_GLACIER_MAX = _sfc.SWE_GLACIER_MAX
    _cm.H_ICE_THRESHOLD_M = _ice.H_ICE_THRESHOLD_M


def reset_default_parameters() -> None:
    """Restore all `src/aoba/*` physics constants back to factory defaults."""
    apply_module_parameters(default_config_dict())


def build_configs_from_dict(cfg_dict: dict[str, dict[str, Any]], apply_modules: bool = True) -> tuple[AOBAConfig, OutputConfig]:
    """Instantiate `(AOBAConfig, OutputConfig)` from a merged config dict and apply module parameters."""
    if apply_modules:
        apply_module_parameters(cfg_dict)
    g = cfg_dict.get("grid", {})
    f = cfg_dict.get("forcing", {})
    ic = cfg_dict.get("icesheet", {})
    o = cfg_dict.get("output", {})
    ac = AOBAConfig(**{**{k: g[k] for k in _GRID_KEYS if k in g}, **{k: f[k] for k in _FORCING_KEYS if k in f}, **{k: ic[k] for k in _ICESHEET_CFG_KEYS if k in ic}})
    oc = OutputConfig(**{k: o[k] for k in _OUTPUT_KEYS if k in o})
    return ac, oc


# Concise UI metadata (labels, units, categories) for interactive GUI configurator
SECTION_META: dict[str, dict[str, str]] = {
    "grid": {"ja": "格子・時間ステップ", "en": "Grid & Time Stepping"},
    "forcing": {"ja": "惑星・軌道・大気外力", "en": "Planetary & Orbital Forcing"},
    "icesheet": {"ja": "動的氷床・加速設定", "en": "Dynamic Ice Sheet"},
    "output": {"ja": "出力ファイル・時間間隔", "en": "Output Streams & Cadence"},
    "constants": {"ja": "基礎物理定数", "en": "Fundamental Constants"},
    "radiation": {"ja": "4バンド放射パラメータ", "en": "4-Band Radiation"},
    "microphysics": {"ja": "KR03 雲微物理", "en": "Cloud Microphysics"},
    "turbulence": {"ja": "MYNN 2.5 / RSL 乱流", "en": "Turbulence & RSL"},
    "surface": {"ja": "陸面土壌・積雪・海氷", "en": "Surface & Cryosphere"},
    "biosphere": {"ja": "Two-Leaf EEO 植生", "en": "EEO Biosphere"},
    "ocean_ekman": {"ja": "1.5層エクマン海洋", "en": "Ekman Ocean"},
    "orography": {"ja": "山岳重力波ドラッグ", "en": "Orographic Drag"},
    "gravity_wave": {"ja": "非地形性重力波 (対流・前線)", "en": "Non-Orographic GWD"},
    "icesheet_physics": {"ja": "SIA 氷床物理定数", "en": "SIA Ice-Sheet Physics"},
}
