<img src="docs/assets/aoba_icon.png" width="96" align="right" alt="AOBA Icon" />

# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **Language**: English | [日本語 (README_ja.md)](README_ja.md)

**AOBA** (**A**utonomous **O**ptimality **B**iosphere–**A**tmosphere) is a lightweight, GPU-native superparameterized global nonhydrostatic climate model coupled with an eco-evolutionary optimality (EEO) terrestrial biosphere and a selectable 3D SCVT Voronoi OGCM / Ekman–shallow-water ocean. Its acronym also echoes the Japanese word *aoba* (**青葉**, "fresh green leaves"), symbolizing the autonomous emergence of living vegetation and climate zones on a coupled atmosphere–ocean–biosphere planet. Given only minimal external boundary conditions—**land–sea/ice-sheet geometry, surface topography and bathymetry (`Earth2014`), planetary rotation period, solar insolation, and atmospheric gas concentrations**—AOBA investigates how Earth's climate and biome zones emerge from first principles and how a minimally constrained Earth system responds to radiative and orbital perturbations without cumulus parameterizations, ocean $Q$-fluxes, soil/albedo lookup maps, or plant-functional-type (PFT) tuning tables.

## Scientific Motivation & Design Philosophy

### Structural Limitations in Conventional Climate Models
- **Implicit Prescription of Present-Day Climate Geography**: Geographically mapped land-surface parameters (plant functional types, soil-texture and albedo tables), prescribed ocean $Q$-fluxes, and regionally calibrated non-orographic gravity-wave sources embed present-day circulation and biogeographic patterns directly into external boundary conditions. Such constraints limit studies of how Earth's climate zones and biomes emerge ab initio, and may artificially damp large-scale reorganizations (such as tropical widening or deep-time paleoclimate shifts) under altered radiative or orbital forcing.
- **Regime Dependence of Cumulus Parameterizations**: Simulated precipitation, cloud–radiative feedbacks, and tropical–extratropical teleconnections remain highly sensitive to cumulus closure assumptions, which are typically formulated around oceanic deep convection and frequently exhibit structural biases over land or in shallow-convective and stratocumulus regimes.
- **Structural Discontinuity Across Scales and Submodels**: Legacy climate models often separate coarse-resolution climate integrations from high-resolution cloud-resolving models—or, in hybrid superparameterized frameworks, couple dissimilar dynamical cores (e.g., a hydrostatic global host paired with a separate anelastic CRM)—introducing numerical and thermodynamic inconsistencies across scales.
- **Barriers to Reproducibility, Legibility, and Access**: Restricted source licenses, external boundary or initial datasets requiring institutional access, HPC-specific architectures with limited GPU portability, legacy fixed-format Fortran codebases, dispersed component-level configuration files, opaque job monitoring, and monolingual documentation collectively create high technical barriers for independent researchers and undergraduate students.

### The AOBA Philosophy
- **Emergence from Minimal Geographic Constraints**: Aside from surface topography/bathymetry (`Earth2014`) and optional ozone forcing, AOBA requires no geographically mapped boundary datasets; climate zones, terrestrial ecosystems, cloud regimes, ocean circulation, and gravity-wave spectra emerge autonomously from universal physical equations and global parameters.
- **Unified Multi-Scale Framework (SP Climate to Cloud-Resolving)**: Both computationally efficient superparameterized (SP) multi-decadal climate simulations and direct high-resolution cloud-resolving simulations operate seamlessly within the same software and physical infrastructure.
- **Identical Dynamical Foundation for Host and Subgrid Models**: The global host (parent) and embedded CRM (child) solve the exact same flux-form compressible nonhydrostatic Euler equations on orthogonal C-grids with Charney–Phillips vertical staggering, RK3-HEVI integration, Zalesak FCT transport, and shared physics—eliminating parent–child structural mismatches.
- **Hardware Portability from Laptops to Supercomputers**: Automatic backend detection (`cuda` → `mps` → `cpu`) and single-source fused GPU kernels enable high-throughput execution on consumer laptops, cloud GPUs, and HPC clusters alike.
- **Open, Readable, and Self-Contained Scientific Software**: Built in clean, type-annotated, literature-cited Python with zero gated boundary-condition archives (and planned public release of spun-up initial states).
- **Centralized Configuration, Bilingual Documentation, and Interactive Workbench**: All global parameters are consolidated into a single TOML registry controllable via both CUI scripts and the bilingual (English/Japanese) `AOBA Studio` Web GUI for configuration, execution, real-time monitoring, and lightweight diagnostics/visualization.

## Core Design Principles

1. **Unified Compressible Nonhydrostatic Dynamics on Orthogonal C-Grids (1-to-1 SCVT Host + 3D CRM)**:
   - **Global Host Core**: Solves the 3D flux-form compressible nonhydrostatic equations on a **Spherical Centroidal Voronoi Tessellation (SCVT) orthogonal Arakawa C-grid** with **TRiSK** geostrophic/energy-compatible operators, **Charney–Phillips** vertical staggering, Horizontally Explicit Vertically Implicit (HEVI) time integration, and subgrid low-level mountain blocking + orographic gravity-wave drag derived directly from 5-arc-min `Earth2014` subgrid relief (Ringler et al., 2008, 2010; Thuburn et al., 2009; Klemp, 2011; Skamarock et al., 2012; Gassmann, 2013; Lott & Miller, 1997).
   - **1-to-1 Embedded CRM & Turbulence Hierarchy**: Each host Voronoi cell embeds its own periodic nonhydrostatic C-grid/Charney–Phillips CRM (default `8 × 1` 2D curtain at $\Delta x = 4\ \mathrm{km}$, or `8 × 8` 3D at $\Delta x = 4\ \mathrm{km}$, $N_z = 44$ stratosphere-extended levels to $z_{\mathrm{top}} = 48\ \mathrm{km}$; Khairoutdinov & Randall, 2003; Khairoutdinov et al., 2005; Yamazaki & Miura, 2024; Nishizawa et al., 2025), with dimension-aware Coriolis handling (disabled in 1D/2D; perturbation-only in 3D), **MYNN Level 2.5** boundary-layer closure (Nakanishi & Niino, 2009), and anisotropic **Smagorinsky–Lilly** 3D SGS turbulence.
2. **PFT-Free Two-Leaf C₃/C₄ Optimality Biosphere, Multi-Layer Freeze–Thaw Soil, and Snow Physics**:
   - **Two-Leaf EEO Acclimation & Stomatal Closure**: Canopy biochemical capacities ($V_{c\max 25}, J_{\max 25}, V_{p\max 25}$) and leaf dark respiration ($R_{25}$) follow the non-rectangular-hyperbola optimality formulation of Smith et al. (2019) and Scott & Smith (2022), acclimate via exponential moving average (EMA) smoothing of local-solar-noon downward shortwave irradiance followed by two-stream canopy light absorption, and partition analytically into sunlit and shaded leaves in proportion to the acclimated absorbed PPFD profile (de Pury & Farquhar, 1997; Dai et al., 2004; Prentice et al., 2014; Heskel et al., 2016; Smith et al., 2019; Mengoli et al., 2022; Scott & Smith, 2022; Luo et al., 2024; Ren et al., 2024, 2025; Lavergne et al., 2026).
   - **Land Hydrology, Optimal Soil-Moisture Response, & Permafrost Thermodynamics**: Partitions evapotranspiration into sunlit/shaded transpiration, wet-canopy interception loss, and stability-corrected bare-soil evaporation (Dai et al., 2004; Sakaguchi & Zeng, 2009; Lee & Khairoutdinov, 2015; MATSIRO6, 2021), paired with time-integrated optimal soil-moisture stress using climate-adapted water-saving timescale $\tau(D_{\max})$ (Lu et al., 2016; Holtzman et al., 2024), drought-bridging root-zone storage $S_R$ (Gao et al., 2014; Davis et al., 2017), prognostic EEO $\mathrm{LAI}$ with autonomous evergreen/deciduous leaf-habit optimality and winter frost damage (Kikuzawa, 1991; Sakai & Larcher, 1987; Givnish, 2002; Wang et al., 2023; Cai et al., 2025; Zhou et al., 2025), moisture-dependent bare-soil VIS/NIR albedo (Oleson et al., 2013), multi-layer implicit soil heat conduction with freeze–thaw phase change (SCALE-RM, 2025; MATSIRO6, 2021), and fractional snow cover with snow-burial canopy masking, aging albedo, and ocean-conserving runoff (Yang et al., 1997; Swenson & Lawrence, 2012; Nitta et al., 2014; Abolafia-Rosenzweig et al., 2021).
3. **Displacement-Following Smoothed Terrain Datum & Modern Roughness Sublayer (RSL)**: Canopy height $h_c$ scales with local resource limits (Kempes et al., 2011), determining zero-plane displacement $d$, roughness length $z_{0m}$, and analytical RSL stability corrections (Raupach, 1994; Harman & Finnigan, 2007; De Ridder, 2010; Bonan et al., 2018; Lee et al., 2020; Meier et al., 2022) above the smoothed terrain datum $z_{\mathrm{datum}} = z_{\mathrm{topo}} + d$ (Klemp, 2011).
4. **16-Band Correlated-k Radiation, 2-Moment Cloud Microphysics, & Selectable 3D OGCM / Shallow-Water Ocean**: Defaults to a deterministic 16-band (50 g-point) Correlated-$k$ radiative transfer solver (`rad_scheme = "ckd"` or legacy `"four_band"`, resolving $\mathrm{H_2O}, \mathrm{CO_2}, \mathrm{O_3}, \mathrm{CH_4}, \mathrm{N_2O}$), 2-moment prognostic $(n_c, n_i)$ cloud microphysics with map-free natural sea-salt/dust CCN & INP activation (`microphysics_scheme = "two_moment"` or `"one_moment"`), and a selectable 18-layer 3D SCVT Voronoi OGCM (`ocean_mode = "ogcm_3d"`, `Earth2014 BED2014` bathymetry, salinity, Roquet 2015 EOS, GM90 + full rotated Redi tensor, KPP boundary layer + non-local convection, Ri shear mixing, topographic internal-tide deep mixing, Jerlov shortwave penetration, Hibler 1979 dynamic sea-ice drift) alongside the lightweight 1.5-layer Ekman + shallow-water ocean (`ocean_mode = "ekman_sw"`).
5. **Dynamic / Accelerated SIA Ice-Sheet & Single-File TOML + Interactive GUI**:
   - **Dynamic SIA + EMA Ice-Sheet Coupling**: Prognoses continental ice thickness on the SCVT grid using Glen's flow law ($n=3$), EMA-filtered annual SMB, relaxed lithospheric isostasy, spinup acceleration, dynamic `SURF_LAND <-> SURF_ICESHEET` masking, and conservative barometric vertical coordinate remapping (`src/aoba/icesheet.py`).
   - **Unified Configuration & Web Workbench (`AOBA Studio`)**: Centralizes all 256 parameters across 14 sections — CMIP6 1850 `piControl` defaults, stratosphere-extended grid (`nz = 44`, `z_top = 48 km`, low-top 24 km preset also available), calendar/perpetual-insolation, time-varying CO₂ forcing (`constant` / CMIP5 RCP / CMIP6 SSP / NOAA GML / 800 ka Antarctic ice-core / custom CSV, refreshable via `scripts/fetch_co2_data.py`), configurable ozone (`latitude_seasonal` / `rcemip` / external CMIP6/7 `climatology` via `scripts/fetch_ozone_data.py` + 9.6 μm LW), map-free flow-dependent non-orographic GWD (`src/aoba/gravity_wave.py`), dynamic ice-sheet, output/monthly-restart cadence, and physical constants in [`config/aoba_default.toml`](config/aoba_default.toml), editable via the zero-dependency Web UI ([`scripts/aoba_config_gui.py`](scripts/aoba_config_gui.py)).
6. **Automatic Hardware-Portable Single-Source GPU Acceleration (43 Fused Metal / CUDA Kernels)**: Automatically detects the host accelerator (`cuda` → `mps` → `cpu`) and dispatches all 43 fused GPU kernels natively on Apple Silicon (`mps` Metal shaders) and NVIDIA GPUs (`cuda` via single-source CUDA C++17 transpilation in `src/aoba/kernels_cuda.py`), with a ready-to-run Google Colab A100/H100 auto-resume notebook ([`notebooks/aoba_colab_spinup.ipynb`](notebooks/aoba_colab_spinup.ipynb)).

## Repository Structure

```text
aoba/
├── README.md              # Project overview (English)
├── README_ja.md           # Project overview (Japanese)
├── docs/
│   ├── architecture.md    # Governing architecture & literature traceability (English)
│   ├── architecture_ja.md # Governing architecture & literature traceability (Japanese)
│   ├── formulation.md     # Full mathematical model description (English)
│   └── formulation_ja.md  # Full mathematical model description (Japanese)
├── data/                  # Local boundary (Earth2014) & validation (Köppen-Geiger) data (git-ignored)
├── references/            # Local literature archive (74 PDFs; strictly git-ignored)
├── src/
│   └── aoba/              # PyTorch + Metal/CUDA model source packages (15 modules)
├── tests/                 # Unit, physical-forcing, and conservation tests (CPU + MPS)
├── scripts/               # RCE, Held-Suarez, Global AOBA runners & Zarr-to-NetCDF tool
└── pyproject.toml
```

## Quick Start

```bash
# 1. Automated environment bootstrap (verifies dependencies, downloads Earth2014 grids, runs tests & smoke run)
python setup_aoba.py

# Optional editable install (tests/scripts also auto-resolve `src/` & auto-download Earth2014 on first run)
pip install --no-build-isolation --no-deps -e .

# Run the full verification suite on CPU + Apple Silicon MPS GPU
python tests/test_phase1_foundation.py
python tests/test_phase1_physics_biosphere.py
python tests/test_phase1_crm_rce.py
python tests/test_phase2_phase3_global.py
python tests/test_params_and_gui.py

# Launch RCE, Held-Suarez, or full Global 1:1 Superparameterized AOBA (Zarr v3 output)
python scripts/run_rce.py --surface land --days 2.0
python scripts/run_held_suarez.py --glevel 3 --days 5.0
python scripts/run_global_aoba.py --glevel 3 --co2-ppmv 400 --days 5.0

# Launch interactive TOML/Grid GUI workbench (AOBA Studio) & run multi-year integration
python scripts/aoba_config_gui.py --toml config/aoba_default.toml
python -u scripts/run_g5_30yr.py --config config/aoba_default.toml
```

## Storage & Git Protection Policy

- **Simulation Outputs (`Zarr v3` Native)**: All model history streams, restart checkpoints, and mesh caches use **CF/UGRID-compliant Zarr v3 (`.zarr`)** exclusively (defaulting to `~/aoba_outputs`, configurable via `AOBA_OUTPUT_DIR`); conversion to NetCDF4 (`.nc`) for external tools (`cdo`, Panoply) is handled strictly by `scripts/zarr_to_netcdf.py`.
- **Protected Local Directories**: Both `references/` and `data/` are backed up locally via Google Drive and triple-locked (`.gitignore`, `.git/info/exclude`, and `.git/hooks/pre-commit`) so they are never pushed to GitHub.

## Validation Benchmarks and Model Hierarchy

- **Single-Column RCE Testbed**: Single host column (`B = 1`) coupling an embedded `8 × 8` 3D C-grid/Charney–Phillips HEVI CRM, MYNN 2.5 / RSL turbulence, 4-band radiation, and subgrid Two-Leaf C₃/C₄ EEO land / slab-ocean radiative–convective equilibrium (Wing et al., 2018).
- **Global Dynamical Core Benchmark**: Global SCVT Voronoi C-grid (TRiSK) + Charney–Phillips HEVI nonhydrostatic host core with smoothed `Earth2014` topography, Lott & Miller (1997) subgrid orographic drag, and Held–Suarez verification (Held & Suarez, 1994).
- **Global 1:1 Coupled Simulation**: Global `1:1` superparameterized coupling over continents, `Earth2014` topography, and 1.5-layer Ekman slab ocean + fractional sea ice, evaluated against observed Köppen–Geiger climate zones (Beck et al., 2018, 2023).
- **Climate Experiments & Factorial Attribution**: Rotational experiments, perpetual-date insolation runs, and elevated-$\mathrm{CO}_2$ factorial attribution (separating radiative forcing, physiological stomatal/$\mathrm{C}_3\text{--}\mathrm{C}_4$/$\mathrm{LAI}$ feedbacks, and wind-driven Ekman ocean response).

## License

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
