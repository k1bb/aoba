<img src="docs/assets/aoba_icon.png" width="96" align="right" alt="AOBA Icon" />

# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **Language**: English | [日本語 (README_ja.md)](README_ja.md)

**AOBA** (**A**utonomous **O**ptimality **B**iosphere–**A**tmosphere) is a lightweight, GPU-native superparameterized global nonhydrostatic Earth System Model (ESM) accelerated by **zero-pretraining online self-distilling machine learning (OSSP)**, coupled with an eco-evolutionary optimality (EEO) terrestrial biosphere and an independent-grid 3D SCVT Voronoi OGCM. Its acronym echoes the Japanese word *aoba* (**青葉**, "fresh green leaves"), symbolizing the autonomous emergence of living vegetation and climate zones on a coupled planet. Given only minimal external boundary conditions — **land–sea/ice-sheet geometry, surface topography and bathymetry (`Earth2014`), planetary rotation period, solar insolation, and atmospheric gas concentrations** — AOBA investigates how Earth's climate zones, ocean circulation, clouds, aerosols, and ecosystems emerge from universal physical laws without cumulus parameterizations, ocean $Q$-fluxes, soil/albedo lookup maps, or plant-functional-type (PFT) tables.

## Scientific Motivation & Design Philosophy

### Structural Limitations in Conventional Climate Models
- **Implicit Prescription of Present-Day Climate Geography**: Geographically mapped land-surface parameters (PFTs, soil-texture and albedo tables), prescribed ocean $Q$-fluxes, and regionally tuned gravity-wave or dust-erodibility maps embed present-day circulation and biogeographic patterns into external boundary conditions. Such constraints hinder ab initio tests of climate-zone emergence and can damp large-scale reorganizations (tropical widening, greening deserts, deep-time paleoclimate shifts) under altered radiative or orbital forcing.
- **Regime Dependence of Cumulus Parameterizations**: Simulated precipitation, cloud–radiative feedbacks, and teleconnections remain sensitive to cumulus closures designed primarily around oceanic deep convection, often degrading over land or in shallow-convective and stratocumulus regimes.
- **Computational Barrier of Explicit Superparameterization & Out-of-Distribution Fragility of Offline ML Emulators**: Running an explicit cloud-resolving model (CRM) and independent-column radiation in every global column is prohibitively expensive for multi-decadal coupled ESM runs, whereas conventional offline-trained neural emulators freeze weights to a single training climate and can drift or violate conservation laws under abrupt $4\times\mathrm{CO}_2$, paleoclimate, or exoplanet regimes.
- **Structural Discontinuity & Barriers to Access**: Legacy models often couple dissimilar host and subgrid dynamical cores, rely on institution-gated boundary archives, fixed-format Fortran, HPC-only architectures, dispersed namelists, and monolingual documentation.

### The AOBA Philosophy
- **Zero-Pretraining Online ML Acceleration with Exact Physical Conservation (OSSP)**: AOBA accelerates both 2D/3D superparameterization and subcolumn ICA/3D-TICA radiative transfer via **Online Surrogate Superparameterization (`OSSP`)**. A rotating, spatially stratified + epistemic-uncertainty-guided subset of columns runs the genuine nonhydrostatic CRM and radiation teacher while lightweight residual neural surrogates — equipped with exact algebraic water, moist-enthalpy, momentum, and radiative flux-divergence conservation layers — advance the remaining columns and train continuously online on GPU. Because no present-day climate weights or geographic coordinates are prescribed, the surrogates adapt automatically across abrupt $4\times\mathrm{CO}_2$, glacial, and exoplanet regimes, while continuous **prequential shadow diagnostics** quantify surrogate-versus-physics divergence in real time.
- **Emergence from Minimal Geographic Constraints**: Aside from surface topography/bathymetry (`Earth2014`) and optional climatological ozone, AOBA uses no spatially mapped boundary lookup tables; climate zones, terrestrial ecosystems, wildfires, mineral dust, marine ecosystems, cloud regimes, ocean circulation, and gravity-wave spectra emerge from universal equations and global parameters.
- **Unified Multi-Scale Framework & Identical Host–Child Dynamics**: Both superparameterized multi-decadal climate runs (`enable_superparam = true`) and direct cloud-resolving simulations (`enable_superparam = false`) execute within one codebase. The global host (parent) and embedded CRM (child) solve the exact same flux-form compressible nonhydrostatic Euler equations on orthogonal Arakawa C-grids with Charney–Phillips vertical staggering, RK3-HEVI integration, Zalesak FCT transport, and shared physics.
- **Automatic Resolution-Aware Scaling & Hardware Portability (Laptop to HPC)**: Resolution-dependent physical and numerical operators (subgrid LWC-variance microphysics, biharmonic hyperviscosity, divergence damping, Hallberg scale-aware GM90, Smagorinsky mixing, CFL substeps) scale automatically from active mesh geometry. Automatic accelerator dispatch (`cuda` → `mps` → `cpu`), **101 single-source fused Metal/CUDA kernels**, and automatic CUDA Graphs enable high throughput across Apple Silicon laptops, cloud GPUs, and supercomputing clusters (such as JAMSTEC ES4 A100s).
- **Open, Readable, Bilingual Scientific Software & Interactive Workbench**: Written in clean, type-annotated, literature-cited Python with zero gated datasets, a single centralized 322-parameter TOML registry, synchronized English/Japanese textbook documentation, and the interactive `AOBA Studio` Web GUI.

## Core Design Principles

1. **Online Self-Distilling ML Acceleration (`OSSP`: CRM + Paired ICA Radiation + `SparseCRMPool` + Shadow Diagnostics)**:
   - **Teacher–Student Online Distillation (`src/aoba/crm_self_distill.py`)**: Executes the genuine 2D/3D nonhydrostatic CRM and full ICA/3D-TICA radiative transfer on a rotating stratified + high-epistemic-uncertainty cohort (`1 / crm_distill_stride`, with continuous convective-memory dwell and a 50% persistent anchor cohort) while two-head epistemic residual MLPs (`CRMSelfDistillNet` and `RadSelfDistillNet`) predict tendencies on the remaining columns and train continuously online from GPU-resident dual-zone replay buffers (recency ring + long-horizon reservoir).
   - **Exact Conservation & Subgrid Organization Coupling**: Uses strictly local, map-free thermodynamic profiles, three prognostic convective-memory states $(M_{P,\mathrm{fast}}, M_{P,\mathrm{slow}}, M_{B,\mathrm{slow}})$ (Mapes & Neale, 2011; Colin et al., 2019), and 16 periodic 2D/3D surface-heterogeneity clustering descriptors (Moran's $I$, patch length scale $L_{\mathrm{patch}}$, $\rho(\mathrm{SHF}, \mathrm{LHF})$; Patton et al., 2005). Fused GPU closure kernels enforce exact algebraic column water-mass, sensible/latent enthalpy, horizontal momentum, and per-subcolumn TOA–atmosphere–surface radiative energy closure to machine precision (Beucler et al., 2021), paired with a periodic `SubgridSpatialHead` for zero-mean cold-pool microclimate and gust-outflow coupling to MAML surface tiles.
   - **Teacher-Only `SparseCRMPool` & Prequential Shadow Diagnostics**: For large 3D CRMs with decoupled surface tiles (e.g., `64 × 16` `RCE_large`), `SparseCRMPool` allocates full 3D CRM state strictly on active teacher slots (`~0.15 GB` instead of `~37 GB`). Enabling `enable_distill_shadow_diag = true` evaluates both surrogates on active teacher columns strictly prior to each online gradient update, producing unbiased out-of-sample host-atmosphere and land-tile divergence metrics across abrupt forcing transitions.
   - **Measured Single-GPU Throughput**: Delivers **`25.4 SYPD` (`0.94 h/yr`)** for a high-top 3D `8 × 8` CRM + eddy-permitting `OG7` ocean configuration (`earth_g5_3d_8x8_g7ocn.toml`) and **`10.0 SYPD` (`2.40 h/yr`, `23.7 s/day`)** for the full 17-component concentration-driven `ALL-ESM` configuration (`AG4z64 + OG7z64`, `32 × 1` `2 km` CRM, `RRTMGP + 3D-TICA`, `six_class_2m`, prognostic $\mathrm{O}_3$, FlexPFT BGC/$\mathrm{O}_2$, EVP-ITD sea ice, MAML topo/lakes, C/N/fire, aerosols, isotopes) on a single NVIDIA A100-40GB.

2. **Unified Compressible Nonhydrostatic Dynamics on Orthogonal C-Grids (SCVT Host + 2D/3D CRM)**:
   - **Global Host Core (`src/aoba/scvt_dynamics.py`)**: Solves 3D flux-form compressible nonhydrostatic Euler equations on a **Spherical Centroidal Voronoi Tessellation (SCVT)** orthogonal Arakawa C-grid (`Glevel-4` to `Glevel-7+`) with **TRiSK** operators, **Charney–Phillips** vertical staggering (`nz = 32` to `64`, up to `z_top = 65 km`), height-smoothed terrain-following coordinates (Klemp, 2011), RK3-HEVI implicit vertical acoustics, positive-definite Zalesak FCT advection, Lott & Miller (1997) subgrid orographic blocking/GWD, and map-free flow-dependent frontal/convective non-orographic GWD (`src/aoba/gravity_wave.py`; Ringler et al., 2008, 2010; Thuburn et al., 2009; Skamarock et al., 2012; Gassmann, 2013; Nishizawa et al., 2025).
   - **Embedded 2D/3D CRM & Turbulence Hierarchy (`src/aoba/crm.py`, `src/aoba/turbulence.py`)**: Embeds a periodic compressible C-grid/Charney–Phillips CRM in each host column (configurable 2D curtain such as `32 × 1` at `2 km` or 3D `8 × 8` / `64 × 16`), coupled with implicit **MYNN Level 2.5** PBL turbulence (Nakanishi & Niino, 2009) and anisotropic **Smagorinsky–Lilly** 3D SGS diffusion.

3. **Selectable Radiation (`four_band` / `ckd` / `ckd_hd` / `rrtmgp` + 3D-TICA) & Cloud Microphysics (`one_moment` / `two_moment` / `six_class_2m`)**:
   - **Radiative Transfer**: Four unified schemes under `rad_scheme`: analytical 4-band (`four_band`), 16-band 50 $g$-point Correlated-$k$ (`ckd`), 111 $g$-point high-detail CKD (`ckd_hd`), and official 240 $g$-point HITRAN / MT\_CKD 4.1 **RRTMGP** (`rrtmgp`, 128 LW + 112 SW; Pincus et al., 2019), with optional 3D Tilted Independent Column Approximation (`enable_rad_3d_tica`; Várnai & Davies, 1999) and interactive aerosol/dust/cloud/$\mathrm{O}_3$ radiative coupling.
   - **Cloud Microphysics & Lightning**: Three selectable schemes under `microphysics_scheme` — 1-moment mixed-phase (`one_moment`; Khairoutdinov & Randall, 2003), 2-moment cloud liquid/ice (`two_moment`), and full 6-class 2-moment (`six_class_2m`: vapor, cloud liquid/ice, rain, snow, graupel; Morrison et al., 2005; Seifert & Beheng, 2006) — with map-free natural sea-salt/sulfate/dust CCN & INP activation and non-inductive subcolumn lightning & $\mathrm{LNO}_x$ (`src/aoba/lightning.py`; McCaul et al., 2009).

4. **PFT-Free Two-Leaf $\mathrm{C}_3/\mathrm{C}_4$ EEO Biosphere, MAML Subgrid Land/Lakes, C/N Permafrost, Wildfire, & River Routing**:
   - **Autonomous EEO Vegetation & Thermal Acclimation (`src/aoba/biosphere.py`)**: Non-rectangular-hyperbola optimality photosynthesis ($V_{c\max 25}, J_{\max 25}, V_{p\max 25}, R_{25}$), three-pillar thermal acclimation ($T_{\mathrm{growth}}, T_{\mathrm{home}}$), closed-form sunlit/shaded two-leaf partitioning, discrete-crown clumping and optimal leaf orientation, climate-adapted soil-moisture optimality $\tau(D_{\max})$, drought-bridging root-zone capacity $S_R$, autonomous evergreen/deciduous habit $f_{\mathrm{evg}}$ with winter frost damage, Lambert-$W_0$ prognostic LAI, and vascular canopy height $h_c$ coupled to analytical Roughness Sublayer (RSL) similarity (de Pury & Farquhar, 1997; Prentice et al., 2014; Smith et al., 2019; Scott & Smith, 2022; Wang et al., 2023; Schneider et al., 2025; Zhou et al., 2025).
   - **MAML Subgrid Orography, Dynamic Lakes, C/N Cycle, Wildfire, & Rivers**: Subgrid topographic microclimate downscaling and TOPMODEL wetlands (`src/aoba/maml_land_subgrid.py`), map-free dynamic lakes using `Earth2014` depression volume + 2-layer FLake thermodynamics (`src/aoba/lakes.py`), 5-layer freeze–thaw soil heat/Richards hydrology and snow aging/BC darkening (`src/aoba/surface.py`), coupled terrestrial C–N pools with 5-layer permafrost C/N and discrete MAML SPITFIRE wildfire (`src/aoba/land_carbon.py`), and Priority-Flood Voronoi river & solute routing (`src/aoba/river_routing.py`).

5. **Independent-Grid 3D `z*` FCT Voronoi OGCM, EVP-ITD Sea Ice, FlexPFT Marine BGC + $\mathrm{O}_2$/OMZ, Waves, & Offline Deep-Ocean Acceleration**:
   - **Selectable Ocean Core (`ocean_mode = "ogcm_3d"` / `"ekman_sw"`)**: Supports independent fine Voronoi ocean grids (`ocean_glevel = 7` ~28–55 km, `ocean_nz = 64` stretched from 2 m top layer) with exact $z^*$ free surface, Zalesak FCT tracer advection, Roquet (2015) EOS, Hallberg (2013) scale-aware GM90 + rotated Redi isoneutral diffusion, Fox-Kemper submesoscale restratification, KPP + Ri shear + internal-tide deep mixing, current-relative wind stress, surface wave coupling (`src/aoba/ocean_waves.py`), and C-grid Hollingsworth–Bernoulli divergence damping (`src/aoba/ocean_ogcm.py`), alongside the fast 1.5-layer Ekman–shallow-water core (`src/aoba/ocean_ekman.py`).
   - **EVP-ITD Sea Ice, Enthalpy-Conserving Coupler, & FlexPFT BGC**: 5-category ice thickness distribution (ITD) with Voronoi elastic–viscous–plastic (EVP) rheology (`src/aoba/seaice_evp.py`), exact total water+ice enthalpy surface coupling (`src/aoba/ocean_coupler.py`), and FlexPFT marine biogeochemistry with OMIP/mocsy 2.0 carbonate chemistry, $\mathrm{O}_2$/OMZ denitrification, and atmospheric dust-Fe / $\mathrm{LNO}_x$ / riverine solute coupling (`src/aoba/ocean_bgc.py`).
   - **Self-Forced Offline OGCM Deep-Ocean Acceleration (`scripts/run_offline_ogcm_spinup.py`)**: Accelerates millennial deep-ocean thermohaline and BGC spin-up by cycling multi-year Phase-1 coupled atmospheric fluxes into the standalone 3D OGCM + FlexPFT BGC + EVP-ITD ice core (auto-stopping at `< 0.01 K/century` deep drift) before seamless Phase-3 coupled re-equilibration.

6. **Atmospheric Chemistry, Natural Aerosols, Water Isotopes, Dynamic Ice Sheets, & Planetary Extensions**:
   - **Prognostic 3D Ozone & Natural Aerosols**: Four ozone modes (`rcemip`, `latitude_seasonal`, `climatology`, `prognostic` 3D Chapman + $\mathrm{HO}_x/\mathrm{NO}_x/\mathrm{ClO}_x$/PSC photochemistry + Brewer–Dobson transport; `src/aoba/ozone_chemistry.py`), 4-species natural aerosols (sea salt, FlexPFT-coupled DMS/$\mathrm{SO}_4$, wildfire BC/OC, BVOC-SOA; `src/aoba/aerosols.py`), map-free mineral dust (`src/aoba/dust.py`), snow BC/dust darkening, and stable water isotopes ($\mathrm{H}_2^{18}\mathrm{O}$, $\mathrm{HDO}$; `src/aoba/water_isotopes.py`).
   - **Optional Emission-Driven, Paleoclimate, & Exoplanet Switches**: Selectable $\mathrm{CO}_2$ modes (`constant` / `scenario` / `csv` / `prognostic` 3D transport), prognostic $\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$ (`src/aoba/trace_gas_cycle.py`), volcanic/solar forcing (`src/aoba/volcanic_solar.py`), silicate weathering & Milankovitch orbital evolution (`src/aoba/planetary_geochem.py`), accelerated Glen-law SIA dynamic ice sheets (`src/aoba/icesheet.py`), and exoplanet presets (`earth`, `proxima_b`, `trappist_1e`, `mars_early`; `src/aoba/planetary_presets.py`).

7. **101 Single-Source Fused Metal / CUDA Kernels, Universal Grid Remapping, & Centralized 322-Parameter `AOBA Studio`**:
   - **Single-Source GPU Backend**: All **101 fused GPU kernels** across `kernels.py`, `kernels_extra.py`, `kernels_physics.py`, and `kernels_esm.py` execute natively on Apple Silicon (`mps` Metal shaders) and NVIDIA GPUs (`cuda` via deterministic Metal→CUDA C++17 transpilation in `src/aoba/kernels_cuda.py`), with automatic CUDA Graph capture/replay (`src/aoba/runtime.py`) and zero mid-step CPU–GPU synchronizations.
   - **Centralized Registry, Presets, & Diagnostics**: All **322 global parameters across 14 sections** live in [`config/aoba_default.toml`](config/aoba_default.toml) with 100% bilingual documentation (`src/aoba/param_docs.py`), **11 ready-to-run presets** in `config/presets/`, universal cross-grid restart remapping (`AG4 → AG6`, arbitrary `nz`/`ocean_glevel`), 30-class Köppen–Geiger diagnostics (`src/aoba/koppen.py`), CMIP7 Data Request v1.2 output (`src/aoba/cmip7_diag.py`), and the bilingual `AOBA Studio` Web workbench (`scripts/aoba_config_gui.py`).

8. **Single-Precision (`float32`) Design Rationale & Numerical Safeguards**:
   - **Why `float32`?**: Consumer/laptop GPUs (Apple Silicon `mps`, lacking native `float64`) and mainstream cloud GPUs severely restrict FP64 throughput; native `float32` doubles effective memory bandwidth across PDE solvers.
   - **Precision & Stability Safeguards**: (i) Increment/telescoping flux-difference implicit tridiagonal solvers for 3D ocean mixing and 5-layer soil heat/Richards hydrology; (ii) Neumaier–Kahan compensated summation (`_kahan_add`) for all large thermal, salinity, and terrestrial/marine carbon–nitrogen reservoirs; (iii) anomaly-form 3D ocean advection and `float64`-scaled carbonate/hydrostatic solves; (iv) exact total water+ice enthalpy surface–ocean coupling; (v) $C^\infty$-smooth Kalnay–Kanamitsu implicit surface–atmosphere conductance limiter (`src/aoba/surface.py`) preventing $2\Delta t$ tall-canopy sensible-heat oscillations; (vi) resolution-aware C-grid Hollingsworth–Bernoulli divergence damping (`src/aoba/ocean_ogcm.py`); and (vii) exact `expm1`/`log1p`, $\Delta\tau \to 0$ Taylor forms, and overflow-free Lambert $W_0$.

## Repository Structure

```text
aoba/
├── README.md                 # Project overview (English)
├── README_ja.md              # Project overview (Japanese)
├── config/
│   ├── aoba_default.toml     # Centralized 322-parameter TOML registry (14 sections)
│   └── presets/              # 11 ready-to-run experiment presets (Mac to ES4/A100 Flagship)
├── docs/
│   ├── architecture.md       # System architecture & literature traceability (EN/JA)
│   ├── configurations.md     # Experiment presets, OSSP & GPU guidelines (EN/JA)
│   ├── formulation.md        # Complete mathematical model description (EN/JA)
│   ├── roadmap.md            # Development roadmap (EN/JA)
│   └── assets/               # Icon & 7 schematic SVG diagrams
├── data/                     # Local Earth2014 topography/bathymetry & validation caches
├── src/
│   └── aoba/                 # PyTorch + 101 single-source Metal/CUDA modules (55 files)
├── tests/                    # 27 unit, conservation, kernel & OSSP test suites (CPU/MPS/CUDA)
├── scripts/                  # Coupled runners, offline OGCM accelerator, GUI & plotting tools
├── notebooks/                # Colab A100/H100 auto-resume notebook
└── pyproject.toml
```

## Quick Start

```bash
# 1. Automated environment bootstrap (verifies dependencies, downloads Earth2014 grids, smoke tests)
python setup_aoba.py

# Optional editable install (scripts/tests also auto-resolve src/ and auto-fetch Earth2014)
pip install --no-build-isolation --no-deps -e .

# Run core verification & OSSP / kernel / conservation test suites
python tests/test_phase1_foundation.py
python tests/test_phase1_physics_biosphere.py
python tests/test_phase2_phase3_global.py
python tests/test_crm_self_distill.py
python tests/test_esm_kernels.py
python tests/test_ogcm_fct_zstar.py
python tests/test_params_and_gui.py

# Launch interactive bilingual TOML/Grid GUI workbench (AOBA Studio)
python scripts/aoba_config_gui.py --toml config/aoba_default.toml

# Run presets (Mac exploratory, 3D 8x8 OSSP baseline, or full 17-component ALL-ESM spin-up)
python -u scripts/run_g5_30yr.py --config config/presets/mac_g5_quickstart.toml
python -u scripts/run_g5_30yr.py --config config/presets/earth_g5_3d_8x8_g7ocn.toml
python -u scripts/run_g5_30yr.py --config config/presets/earth_g4z64_g7z64_all_esm_spinup.toml

# Phase-2 self-forced offline 3D OGCM + FlexPFT BGC + EVP-ITD deep-ocean acceleration
python -u scripts/run_offline_ogcm_spinup.py \
  --coupled-dir ~/aoba_outputs/<phase1_run> \
  --out-dir ~/aoba_outputs/<phase2_offline> \
  --years 1000 --stop-deep-drift 0.01
```

## Simulation Output Format

- **Native `Zarr v3` I/O**: All model history streams, CMIP7 diagnostics, restart checkpoints, and preprocessed mesh caches use **CF-1.11 / UGRID-1.0 Zarr v3 (`.zarr`)** (defaulting to `~/aoba_outputs`, configurable via `AOBA_OUTPUT_DIR`). NetCDF4 (`.nc`) conversion for external post-processing tools (`cdo`, Panoply) is provided by `scripts/zarr_to_netcdf.py`.

## Validation Benchmarks and Model Hierarchy

- **Single-Column RCE Testbed (`scripts/run_rce.py`)**: Single host column coupling an embedded 3D C-grid/Charney–Phillips HEVI CRM, MYNN 2.5 / RSL turbulence, selectable radiation, and Two-Leaf $\mathrm{C}_3/\mathrm{C}_4$ EEO land / slab ocean (Wing et al., 2018).
- **Global Dynamical Core Benchmark (`scripts/run_held_suarez.py`)**: Global SCVT Voronoi C-grid (TRiSK) + Charney–Phillips HEVI nonhydrostatic host core with smoothed `Earth2014` topography and Held–Suarez verification (Held & Suarez, 1994).
- **OSSP Online Self-Distillation & Regime-Shift Shadow Benchmark**: Coupled `piControl` ($284.3\ \mathrm{ppmv}$) to `abrupt-4×CO2` ($1137.2\ \mathrm{ppmv}$) integrations tracking prequential host-atmosphere and land-tile shadow divergence (`enable_distill_shadow_diag = true`) alongside 30-class Köppen–Geiger climate-zone emergence (Beck et al., 2018, 2023).
- **Multi-Scale & Planetary Experiments**: Aquaplanet jet emergence, HighResMIP-class `AG6z64 + OG7z64` full-ESM coupled runs, Mid-Holocene Green Sahara / glacial cycles, and tidally locked exoplanet regimes.

## License

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
