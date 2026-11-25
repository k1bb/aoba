# AOBA Experiment Configuration Guide (Recommended Presets by Scientific Objective)

From a single codebase and centralized parameter registry (`config/aoba_default.toml`), AOBA scales seamlessly from exploratory laptop runs on Apple Silicon Macs to HighResMIP-class general-circulation emergence experiments and full Earth System Model (ESM) integrations on NVIDIA A100/H100 clusters.

This guide summarizes nine reference configurations, including their physical design rationale, concrete TOML settings, hardware requirements, and expected throughput. Every preset is shipped as a ready-to-run TOML file under `config/presets/` and can be launched directly via `python scripts/run_g5_30yr.py --config config/presets/<file>.toml` or loaded in AOBA Studio.

---

## 1. Quick Comparison Matrix

| Configuration | Preset TOML (`config/presets/`) | Atmosphere (`glevel` / `nz` / CRM) | Ocean (`ocean_glevel` / `ocean_nz`) | Radiation / Ozone | BGC & Additional Physics | Target Hardware (1 GPU) | Typical Speed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Mac Exploratory Preset** | `mac_g5_quickstart.toml` | `5` (110 km) / `32` (42 km) / `4×1` | 1.5-layer Ekman–SW (`ekman_sw`, or `ogcm_3d` 18-lvl) | `ckd` or `four_band` / `latitude_seasonal` | Biogeophysics ON, BGC OFF | Apple Silicon Mac (MPS, 16–32 GB) | **~3.8 h/yr** (MPS measured `37.7 s/day`, `4.5 s` init) |
| **2. Circulation Emergence (Aquaplanet)** | `aquaplanet_jet_emergence.toml` | `6` (55 km) / `56` (65 km) / `8×1` | 1.5-layer Ekman–SW (`h_ml=30 m`), no land | `ckd` or `rrtmgp` / `prognostic` or `rcemip` | All OFF (pure dynamics + convection + rad) | A100 / 4090 / M3–M4 Max | ~2–7 h/yr |
| **3. Circulation Emergence (High-Res Earth Physics Core)** | `earth_g6g7_rrtmgp_o3.toml` | `6` (55 km) / `56` (65 km) / `8×1` | `7` (28 km, eddy-permitting) / `36` (15 m top) | `rrtmgp` (240 g-pt) / `prognostic` | Biogeophysics ON, lightning & other BGC OFF | NVIDIA A100 (40/80 GB) / H100 | **~10.1 h/yr** (A100 measured `99.9 s/day`, `14.1 GB VRAM`; at `nz=44`: **`8.1 h/yr`**, `80.1 s/day`) |
| **4. Balanced Multi-Decadal Spinup (Phase-1 PI-Control)** | `earth_g5_phase1_picontrol.toml` | `5` (110 km) / `44` (48 km) / `8×1` | `0` (same 110 km) / `18` | `ckd` (50 g-pt) / `latitude_seasonal` | Biogeophysics + dust + nat. aerosols + EVP ice | NVIDIA A100 (40 GB) | **~2.2 h/yr** (A100 measured `21.8 s/day`) |
| **5. Full ESM** | `earth_full_esm.toml` | `5–6` / `56` (65 km) / `8×1` | `6–7` / `32–36` | `rrtmgp` or `ckd_hd` / `prognostic` | All BGC, dynamic lakes, MAML, waves, isotopes, ice sheet ON | NVIDIA A100/H100 (40–80 GB) | ~6–28 h/yr |
| **6. Paleoclimate (6 ka Green Sahara) & Exoplanet** | `paleo_green_sahara_6ka.toml` | `4–5` / `44` (48 km) / `8×1` | `0` (`ogcm_3d` 18-lvl or `ekman_sw`) | `ckd` or `rrtmgp` / `prognostic` | Orbital forcing + dynamic lakes (`init_fill=0`), or `planet_preset` | Mac MPS / A100 | ~1–4 h/yr |
| **7. Organized Convection (2D `RCE_large`) + Eddy-Permitting Ocean** | `earth_g5_rcelarge_g7ocn.toml` | `5` (110 km) / `44` (48 km) / `64×1` (256 km, `sfc_tiles=8`) | `7` (28 km, eddy-permitting) / `18` | `ckd` / `latitude_seasonal` (paired CRM+Rad distill `1/31`, `dwell=240`) | Biogeophysics + dust + nat. aerosols + EVP ice + 50% persistent RCEMIP anchor | Apple M4 Pro / NVIDIA A100 | **~15.8 h/yr** (M4 Pro MPS measured `1.30 s/step`, `1.28 s` init; ~3 h/yr est. on A100) |
| **8. High-Top 3D `8×8` Climate Baseline + Eddy-Permitting Ocean** | `earth_g5_3d_8x8_g7ocn.toml` | `5` (110 km) / `56` (65 km) / `8×8` 3D (32×32 km, `4×4` square tiles) | `7` (28 km, eddy-permitting) / `36` (`ocean_stride=3`) | `rrtmgp` + 3D TICA / `prognostic` (paired CRM+Rad distill `32` teachers, `SparseCRMPool`) | `six_class_2m` microphysics + non-oro GWD + current-rel wind + EVP ice + 50% persistent anchor | Apple M4 Pro (48 GB) / NVIDIA A100 | **~8.6 h/yr** (M4 Pro MPS measured `0.71 s/step`, `3.63 GB` active MPS; ~1.5 h/yr est. on A100) |
| **9. High-Top 3D `16×64` `RCE_large` Comparison + Eddy-Permitting Ocean** | `earth_g5_3d_rcelarge_g7ocn.toml` | `5` (110 km) / `56` (65 km) / `64×16` 3D (256×64 km, `2×8` square tiles) | `7` (28 km, eddy-permitting) / `36` (`ocean_stride=3`) | `rrtmgp` + 3D TICA / `prognostic` (paired CRM+Rad distill `32` teachers, `SparseCRMPool`) | `six_class_2m` microphysics + non-oro GWD + current-rel wind + EVP ice + 50% persistent anchor | Apple M4 Pro (48 GB) / NVIDIA A100 | **~17.1 h/yr** (M4 Pro MPS measured `1.41 s/step`, `4.19 GB` active MPS; ~3.0 h/yr est. on A100) |

> **Always-On Biogeophysics**: Two-leaf $\mathrm{C}_3/\mathrm{C}_4$ P-Model photosynthesis, thermal acclimation $T_{\mathrm{home}}$, Lambert-$W_0$ prognostic LAI, evergreen/deciduous habit $f_{\mathrm{evg}}$, canopy height $h_c$, root capacity $S_R$, discrete-crown clumping $\Omega(\mu_0, L)$, and non-iterative optimal leaf orientation $\chi_L$ reside directly in the land-surface physics core (`surface.py` + `biosphere.py`). Turning off terrestrial carbon/nitrogen, marine BGC, aerosols, dust, and lightning leaves all vegetation–albedo–transpiration–roughness feedbacks fully active.

---

## 2. Configuration 1: Mac / Laptop Exploratory Preset (Lightweight Glevel-5 Core)

Designed for interactive testing of global clouds, precipitation bands, seasonal cycles, autonomous vegetation (LAI, canopy height, leaf angle), and sea ice on Apple Silicon Macs (MPS, 16–32 GB unified memory) or consumer GPUs.

- **Design rationale**: Retains `glevel = 5` (~110 km) to resolve synoptic eddies while reducing vertical layers (`nz = 32`), CRM columns (`4×1`), radiation (`ckd` or `four_band`), and ocean (`ekman_sw`, or `ogcm_3d` with 18 levels when testing 3D ocean circulation).

```toml
[grid]
glevel = 5
nz = 32
z_top = 42000.0
crm_nx = 4
crm_ny = 1
crm_dx = 4000.0
dt_host = 720.0
n_host_dyn_substeps = 1
n_crm_dyn_substeps = 10
rad_stride = 5
crm_col_chunk = 12288
rad_col_chunk = 16384

[forcing]
rad_scheme = "ckd"                # Switch to "four_band" for fastest laptop runs
o3_mode = "latitude_seasonal"
enable_superparam = true
microphysics_scheme = "two_moment"
enable_nonoro_gwd = true
use_earth2014 = true
earth2014_stride = 2
ocean_mode = "ekman_sw"           # Or "ogcm_3d", ocean_nz = 18 for 3D ocean tests
h_ml_ocean = 50.0
enable_river_routing = true
# BGC off (P-model, LAI, habit, h_c, clumping, and optimal chi_L remain ON)
enable_dust = false
enable_natural_aerosols = false
enable_land_carbon = false
enable_land_nitrogen = false
enable_ocean_bgc = false
enable_lightning = false
enable_dynamic_lakes = false

[output]
years = 1
monthly_mean_include_subgrid_2d = false
snapshots_include_subgrid_2d = false
```

---

## 3. Configuration 2: Spontaneous General-Circulation & Jet-Latitude Emergence (Idealized Aquaplanet)

Isolates the spontaneous emergence of the Hadley cell edge, subtropical jet, eddy-driven polar-front jet, multiple jets under altered rotation rate $\Omega$, and poleward jet shifts under $\mathrm{CO}_2$ warming on a symmetric water-covered sphere free of land–sea contrast, topographic stationary waves, and prescribed latitude-fixed ozone maps.

- **Key physical requirements**:
  1. **Baroclinic eddy resolution**: Resolving the midlatitude Rossby deformation radius $L_d \approx 800\text{–}1000\ \mathrm{km}$ with effective resolution $\sim 6\Delta x$ requires `glevel = 5` (minimum) or `glevel = 6` (~55 km, ideal for clean double-jet separation).
  2. **Zero latitudinally prescribed forcing**: Use `o3_mode = "prognostic"` (3D photochemistry + Brewer–Dobson transport) or uniform `"rcemip"` rather than present-day climatological ozone tables. AOBA's non-orographic GWD (`enable_nonoro_gwd = true`) is strictly map-free (excited from fronts and CRM convection) and safe to keep enabled.
  3. **Rapid equilibration**: A thin mixed layer (`ocean_mode = "ekman_sw"`, `h_ml_ocean = 30.0`) equilibrates in 1–2 years, yielding converged zonal-mean jets and eddy momentum-flux convergence within a 5-year run.

```toml
[grid]
glevel = 6                        # Or glevel = 5 (dt_host = 720, n_host_dyn_substeps = 1)
nz = 44
z_top = 48000.0
crm_nx = 8
crm_ny = 1
dt_host = 600.0
n_host_dyn_substeps = 2
rad_stride = 6

[forcing]
use_earth2014 = false             # Flat symmetric aquaplanet
insolation_mode = "annual_mean"   # Or "seasonal" for seasonal jet migration
rad_scheme = "ckd"                # Or "rrtmgp" for reference spectroscopy
o3_mode = "prognostic"            # Or "rcemip" for idealized uniform ozone
o3_lw_coupling = true
enable_superparam = true
enable_nonoro_gwd = true
ocean_mode = "ekman_sw"
h_ml_ocean = 30.0
# Sweep rotation rate omega (e.g., 0.5x, 1x, 2x, 4x) or co2_ppmv (284 -> 1136)
omega = 7.292115e-05
co2_ppmv = 284.3
enable_dust = false
enable_natural_aerosols = false
enable_land_carbon = false
enable_land_nitrogen = false
enable_ocean_bgc = false
enable_lightning = false

[output]
years = 5
monthly_mean_include_subgrid_2d = false
snapshots_include_subgrid_2d = false
```

---

## 4. Configuration 3: General-Circulation & Ocean-Front Coupling (High-Res Earth Physics Core)

HighResMIP-class realistic-Earth physical configuration coupling a 55 km superparameterized atmosphere (`glevel = 6`), an eddy-permitting 28 km / 36-level 3D ocean (`ocean_glevel = 7`, sharp Kuroshio, Gulf Stream, and ACC SST fronts), official 240 $g$-point HITRAN / MT\_CKD 4.1 radiation (`rrtmgp`), 3D prognostic ozone, and map-free land biogeophysics.

- **Key features**:
  - Independent conservative Voronoi ocean coupling (`ocean_glevel = 7`, `ocean_nz = 36`, 15 m top layer stretched to 5500 m, `earth2014_stride = 1` for full 5-arc-min bathymetry and straits).
  - Lightning (high parametric uncertainty) and non-$\mathrm{O}_3$ BGC/aerosols/dust are disabled to focus compute on fluid dynamics, radiation, ozone–circulation coupling, and biogeophysics. Because `o3_mode = "prognostic"` self-consistently solves Chapman photochemistry, $\mathrm{NO}_x/\mathrm{HO}_x/\mathrm{PSC}$ loss, and Brewer–Dobson 3D advection from model meteorology and scalar $\mathrm{N}_2\mathrm{O}/\mathrm{CH}_4/\mathrm{ECl}$, realistic stratospheric and tropopause ozone emerges autonomously even with other BGC modules off.
  - **Iterative 1-year workflow**: Run in 1-year chunks (`years = 1`, `restart_monthly = true`) to inspect TOA radiation balance, jet latitudes, SST fronts, and sea ice before extending.
  - **Faster variant**: Stepping the ocean down to `ocean_glevel = 6` (~55 km), `ocean_nz = 32` cuts OGCM cost by ~6× and speeds up the full coupled model by ~1.5–1.7×.

```toml
[grid]
glevel = 6
nz = 56
z_top = 65000.0
z_sponge_base = 52000.0
crm_nx = 8
crm_ny = 1
crm_dx = 4000.0
dt_host = 600.0
n_host_dyn_substeps = 2
n_crm_dyn_substeps = 10
rad_stride = 6
crm_col_chunk = 49152
rad_col_chunk = 32768

[forcing]
rad_scheme = "rrtmgp"             # Official HITRAN / MT_CKD 4.1 240 g-pt (128 LW + 112 SW)
o3_mode = "prognostic"            # Autonomous 3D O3 photochemistry + Brewer-Dobson transport
o3_lw_coupling = true
enable_nonoro_gwd = true
use_earth2014 = true
earth2014_stride = 1              # Full 5-arc-min topography/bathymetry for 28 km G7 ocean
enable_superparam = true
microphysics_scheme = "two_moment"
re_liq_um = 10.0
re_ice_um = 48.0
cloud_sw_inhomog = 0.62

ocean_mode = "ogcm_3d"
ocean_glevel = 7                  # Or 6 (55 km), ocean_nz = 32 for faster runs
ocean_nz = 36                     # 15 m top layer stretched geometrically to 5500 m
h_ml_ocean = 15.0
enable_seaice_drift = true
enable_seaice_evp_itd = true
enable_ogcm_fct_zstar = true
enable_ogcm_scale_aware_gm = true
enable_ogcm_submeso = true
enable_ogcm_tidal_mixing = true
enable_current_rel_wind = true
enable_river_routing = true

# Lightning, aerosols, dust, and non-O3 BGC off (biogeophysics stays 100% ON)
enable_lightning = false
enable_dust = false
enable_natural_aerosols = false
enable_aerosol_rad_forcing = false
enable_snow_bc_darkening = false
enable_land_carbon = false
enable_land_nitrogen = false
enable_fire_veg_feedback = false
enable_ocean_bgc = false
enable_bgc_fe_lnox_coupling = false
enable_bvoc_soa = false
enable_prognostic_ch4_n2o = false
enable_silicate_weathering = false
enable_dynamic_lakes = false

[output]
years = 1
restart_interval_years = 1
restart_monthly = true
monthly_mean_include_subgrid_2d = false
snapshots_include_subgrid_2d = false
```

---

## 5. Configuration 4: Balanced Multi-Decadal Spinup (Phase-1 PI-Control)

Primary workhorse configuration for 30–100 year coupled spinup, drift verification, and abrupt $4\times\mathrm{CO}_2$ equilibrium climate sensitivity (ECS) experiments. Runs at `~21.8 s/model-day` (`~2.2 h/year`) on a single NVIDIA A100 (~66 GPU-h for 30 years).

- `glevel = 5`, `nz = 44`, `crm_nx = 8, crm_ny = 1`, `dt_host = 720.0`, `rad_stride = 5`
- `rad_scheme = "ckd"`, `o3_mode = "latitude_seasonal"` (or `"prognostic"`)
- `ocean_mode = "ogcm_3d"`, `ocean_glevel = 0` (110 km), `ocean_nz = 18`, `h_ml_ocean = 50.0`, with EVP-ITD sea ice, FCT-$z^*$, scale-aware GM90/submesoscale, tidal mixing, and Priority-Flood river routing enabled
- Mineral dust (`enable_dust = true`) and natural aerosols (`enable_natural_aerosols = true`) enabled; terrestrial C/N and marine BGC can be turned on in Phase-2 after physical spinup.

---

## 6. Configuration 5: Full Earth System Model (All Coupled Components)

Activates AOBA's full suite of universal Earth-system components: nonhydrostatic dynamics, 2D CRM, `rrtmgp` with 3D TICA, 6-class 2-moment microphysics, prognostic 3D ozone, fine 3D OGCM, EVP-ITD sea ice, surface waves, FlexPFT marine BGC/OMZ, coupled terrestrial C/N + 5-layer permafrost + SPITFIRE wildfire, MAML subgrid orography/wetlands, map-free dynamic lakes (`enable_dynamic_lakes`), dust & natural aerosols, SNICAR snow darkening, water isotopes, prognostic greenhouse-gas cycles, silicate weathering, and dynamic SIA ice sheets.

```toml
[grid]
glevel = 5                        # Or glevel = 6, dt_host = 600, n_host_dyn_substeps = 2
nz = 44
crm_nx = 8
crm_ny = 1
dt_host = 720.0
rad_stride = 5

[forcing]
rad_scheme = "rrtmgp"
enable_rad_3d_tica = true
microphysics_scheme = "six_class_2m"
o3_mode = "prognostic"
o3_lw_coupling = true
co2_mode = "prognostic"           # Or "constant" / "scenario"
enable_co2_3d_transport = true
enable_prognostic_ch4_n2o = true
enable_silicate_weathering = true

ocean_mode = "ogcm_3d"
ocean_glevel = 6                  # Or 7 for 28 km eddy-permitting ocean
ocean_nz = 32
h_ml_ocean = 20.0
enable_seaice_drift = true
enable_seaice_evp_itd = true
enable_ocean_bgc = true
enable_ocean_o2_omz = true
enable_wave_coupling = true
enable_current_rel_wind = true

enable_river_routing = true
enable_dynamic_lakes = true       # Map-free dynamic lakes (Earth2014/Priority-Flood + 2-layer FLake)
lake_init_fill_frac = 0.5
enable_maml_subgrid_topo = true
enable_maml_wetlands_lakes = true
enable_land_carbon = true
enable_land_nitrogen = true
enable_fire_veg_feedback = true
enable_lightning = true
enable_dust = true
enable_natural_aerosols = true
enable_aerosol_rad_forcing = true
enable_snow_bc_darkening = true
enable_bgc_fe_lnox_coupling = true
enable_bvoc_soa = true
enable_water_isotopes = true

[icesheet]
enable_dynamic_icesheet = true
icesheet_ema_years = 5.0
icesheet_accel_factor = 100.0
icesheet_accel_spinup_years = 20.0
```

---

## 7. Configuration 6: Paleoclimate, Glacial Cycles & Exoplanets

- **Mid-Holocene Green Sahara / Glacial–Interglacial**:
  - `enable_milankovitch_orbit = true`, `milankovitch_start_kyr = -6.0` (6 ka) or `-21.0` (LGM)
  - `enable_dynamic_lakes = true`, `lake_init_fill_frac = 0.0` (autonomous emergence of Mega-Lake Chad or proglacial Lake Agassiz purely from precipitation, runoff, and bedrock/isostatic depressions)
  - `enable_dynamic_icesheet = true`, `icesheet_accel_factor = 100.0` (automatic drainage and lake-basin reorganization as ice sheets advance/retreat)
- **Tidally Locked Exoplanets (e.g., Proxima Centauri b, TRAPPIST-1e)**:
  - `planet_preset = "proxima_b"` (or `"trappist_1e"`, `"mars_early"`), `enable_tidal_lock = true`, `use_earth2014 = false`, `rad_scheme = "ckd"` or `"rrtmgp"`, `o3_mode = "prognostic"` or `"rcemip"`.

---

## 8. GPU Resource & Memory Guidelines

1. **One Simulation = One GPU**: AOBA saturates single-GPU memory bandwidth via fused Metal/CUDA kernels without multi-GPU domain decomposition. Allocating 2+ GPUs to a single job yields no speedup; instead, run independent ensemble members or sensitivity cases (e.g., $1\times\mathrm{CO}_2$ vs. $4\times\mathrm{CO}_2$, or `G5` multi-decadal alongside `G6+G7` high-res) concurrently as separate 1-GPU jobs.
2. **VRAM Chunking**: On 40–80 GB GPUs (A100/H100), setting `crm_col_chunk = 49152` runs G6 unchunked for peak throughput. On memory-constrained GPUs, lowering `crm_col_chunk` to `8192` and `rad_col_chunk` to `16384` reduces peak VRAM with zero numerical change.
3. **Zarr v3 Mesh Cache & Universal Cross-Grid Restart Remapping**: `build_scvt_mesh` automatically caches `glevel >= 4` SCVT grids and Earth2014 boundary fields as CF/UGRID-compliant Zarr v3 stores under `~/aoba_outputs/_mesh_cache/` (`AOBA_OUTPUT_DIR/_mesh_cache/`), cutting repeat G6+G7 initialization from ~5 minutes to **~1 second**. Moreover, `scripts/run_g5_30yr.py` supports universal cross-grid restarts across any change in `(glevel, nz, z_top, crm_nx/ny, ocean_glevel, ocean_nz)` using surface-class-aware / wet-ocean spherical `cKDTree` remapping and hydrostatic stratospheric extension (e.g., extending a `48 km / 44-layer` Year-1 checkpoint to `65 km / 56 layers` or up-scaling a G5 spinup to G6/G7).
4. **Online Self-Distilling 2D/3D CRM, Teacher-Only `SparseCRMPool`, & Paired ICA Radiation (`enable_crm_self_distill`, `enable_rad_self_distill`)**: Setting `enable_crm_self_distill = true` under `[forcing]` executes AOBA's genuine CRM only on a rotating spatially stratified + high-epistemic-uncertainty subset (`1 / crm_distill_stride`, default `1/13 ≈ 7.7%` of columns), while a pure-GEMM residual column MLP (`CRMSelfDistillNet`) equipped with exact algebraic conservation layers (column water mass, moist enthalpy, and horizontal momentum), periodic 2D/3D surface-heterogeneity clustering descriptors (Moran's $I$, patch length scale $L_{\mathrm{patch}}$, and $\rho(\mathrm{SHF}, \mathrm{LHF})$), a FiLM-modulated periodic 2D spatial placement head (`SubgridSpatialHead`), and zero-mean near-surface microclimate feedback predicts subgrid tendencies and cloud/precipitation distributions on the remaining columns and updates online from a GPU-resident dual-zone replay buffer. Enabling `enable_rad_self_distill = true` pairs this with online per-subcolumn ICA radiation self-distillation (`RadSelfDistillNet`) synchronized with the teacher cohort, enforcing exact per-subcolumn TOA–atmosphere–surface radiative energy closure, zero night-side shortwave flux, and exact per-tile surface albedo/emissivity response while keeping land biogeophysics, host dynamics, and the 3D OGCM 100% first-principles. For arbitrary 3D CRMs with decoupled surface tiles (`crm_ny > 1`, `crm_nx > 1`, `0 < crm_sfc_tiles < crm_ny * crm_nx`), AOBA automatically factorizes the surface mesh into an isotropic 2D tile grid (`resolve_crm_sfc_tile_grid`, e.g. `16×64 → 2×8` square `32×32 km` tiles or `16×16 → 4×4`) and activates a teacher-only slot-preserving `SparseCRMPool` that allocates full-3D dynamics and 3D-TICA radiation state strictly on the `ceil(N_cells / stride)` active teachers (`~0.15 GB` instead of `24 GB` for G5 `16×64`, preserving uninterrupted multi-day 3D memory on the 50% persistent anchor cohort). On Apple M4 Pro MPS, paired CRM + radiation self-distillation accelerates G5 PI-Control (`CKD`, `rad_stride = 5`) by **2.31×** (`9.59 → 4.14 h/yr`). Combined with fused GPU 3D OGCM TRiSK/FCT+$z^*$+Redi kernels (3.4× faster G7 36-level ocean) and conservative asynchronous ocean coupling (`ocean_stride = 3`), the full high-top (`nz = 56, 65 km`) prognostic-O₃ + RRTMGP + 3D-TICA + 6-class 2M microphysics + G7 36-level OGCM configuration runs on M4 Pro MPS in `3.63 GB` at **`0.71 s/step` (`8.6 h/yr`, `2.79 sim-yr/day`)** for the 3D `8×8` baseline (`earth_g5_3d_8x8_g7ocn.toml`) and **`1.41 s/step` (`17.1 h/yr`, `1.40 sim-yr/day`)** for the 3D `16×64` `RCE_large` comparison (`earth_g5_3d_rcelarge_g7ocn.toml`).
5. **Output Volume Control**: For high-resolution experiments, set `monthly_mean_include_subgrid_2d = false` and `snapshots_include_subgrid_2d = false` while keeping `restart_monthly = true` to relay 1-year jobs cleanly.
