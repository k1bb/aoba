# AOBA Development Roadmap

> **Language**: English | [日本語 (roadmap_ja.md)](roadmap_ja.md)

This document summarizes planned verification milestones, resolution targets, parallelization architecture, and candidate physical extensions for AOBA beyond the baseline coupled Earth System Model configuration (`Glevel-5`, `8 × 1` CRM, `nz = 44`, `ocean_nz = 18`, 1850 `piControl`).

---

## 1. Equilibrium Spin-Up and Quantitative Benchmarking

The primary milestone for the baseline coupled configuration is multi-century equilibration and quantitative evaluation:
- **Three-Stage Equilibrium Spin-Up (`30 + 1,000 + 70 yr`)**: Execute a 30-year coupled integration, an accelerated offline 3D SCVT ocean spin-up (up to 1,000 years, terminating automatically once 50-year deep-ocean drift falls below `0.01 K/century`), and a 70-year coupled continuation.
- **Conservation and Physical Verification**: Audit multi-decadal closure of top-of-atmosphere (TOA) radiative balance, global freshwater and salinity budgets, ocean heat content drift, AMOC and ACC transports, sea-ice extent, terrestrial and marine carbon pools, and thermal acclimation responses.
- **Observational and Multi-Model Evaluation**: Benchmark equilibrated climatologies against WOA23 ocean hydrography, ERA5 atmospheric reanalysis, and CMIP6 `piControl` multi-model ensembles.

---

## 2. Unified-Grid Resolution and Ocean Vertical Refinement

AOBA shares a single spherical centroidal Voronoi tessellation (SCVT) horizontal mesh across atmosphere, ocean, and land. Relative to the first baroclinic Rossby deformation radius $L_d$ (approximately `200 km` near the equator, `20–40 km` in mid-latitudes, and `5–15 km` at high latitudes), each grid level corresponds to a distinct physical regime:

| Grid Level | Global Cells | Spacing $\Delta x$ | Ocean / Coupled Regime | Resolved Phenomena & Remaining Constraints |
|---|---:|---:|---|---|
| **Glevel-5** (baseline) | 10,242 | ~224 km (~2.0°) | Coarse OGCM | Large-scale wind-driven gyres, AMOC/ACC structure, and global heat/carbon budgets; narrow straits, equatorial undercurrent (EUC), and western boundary current separation remain coarse. |
| **Glevel-6** | 40,962 | ~112 km (~1.0°) | Standard ESM scale | Matches standard ~1° basin-scale circulation and biogeochemistry; uniform 1° spacing without equatorial refinement softens ENSO dynamics and western boundary separation. |
| **Glevel-7** | 163,842 | ~56 km (~0.5°) | Non-eddy-resolving ESM target | Resolves equatorial upwelling, EUC, ENSO variability, and major straits (e.g., Indonesian Throughflow) on a uniform grid alongside GM eddy parameterization. |
| **Glevel-8** | 655,362 | ~28 km (~0.25°) | Eddy-permitting | Spontaneous mesoscale eddy generation in low-to-mid latitudes and sharper SST-front air–sea coupling; high-latitude eddies remain partially unresolved. |
| **Glevel-9** | 2,621,442 | ~14 km (~0.125°) | Mid-latitude eddy-resolving | Realistic Kuroshio/Gulf Stream separation, Southern Ocean eddy heat transport, and seasonal-to-annual global storm-permitting (`1 × 1`) integrations. |
| **Glevel-10 / 11** | 10.5M / 41.9M | ~7.0 / 3.5 km | Global cloud-resolving (`1 × 1`) | Multi-week MJO / DYAMOND-class global cloud-resolving integrations and high-latitude shelf/eddy dynamics (requires multi-node GPU domain decomposition). |

### Ocean Vertical Refinement (`ocean_nz`)
- Increasing `ocean_nz` from `18` to `40–50` layers (with finer upper-500 m spacing) sharpens the thermocline, EUC, deep-water overflows, and euphotic-zone biogeochemistry.
- Because the coupled atmosphere and embedded CRM account for over 90% of wall-clock runtime, expanding `ocean_nz` adds only a few percent to total compute cost while substantially improving ocean vertical structure.

---

## 3. Multi-GPU Parallelization

The current implementation executes on a single GPU (`cuda` / `mps`), well suited for `Glevel-5` (`8 × 1` CRM) multi-decadal integrations and independent ensemble members across 1-GPU jobs. Scaling to `8 × 8` 3D CRM, `Glevel-6`/`Glevel-7` climate runs, and `1 × 1` global cloud-resolving (`Glevel-9`–`11`) integrations motivates a two-stage multi-GPU architecture:

1. **Stage 1 — Intra-Node Column Parallelism (`torch.distributed` over NVLink)**:
   - Distribute independent host-column physics workloads (embedded CRM, 16-band CKD radiation, 2-moment cloud microphysics, MYNN PBL, and surface/biosphere tiles—roughly 85–90% of runtime) across 4–8 GPUs within a single node while gathering tendencies on the host SCVT mesh.
   - Provides near-linear throughput and memory scaling for `Glevel-5` with `8 × 8` 3D CRM and `Glevel-6`–`7` superparameterized climate runs without horizontal halo communication.
2. **Stage 2 — SCVT Horizontal Domain Decomposition (MPI / NCCL Halo Exchange)**:
   - Partition the SCVT Voronoi mesh across nodes with 1-ring/2-ring halo exchanges for host dynamics, FCT transport, and 3D OGCM/EVP sea-ice operators.
   - Enables configurations exceeding single-GPU memory (`> 40–80 GB`), including eddy-permitting/eddy-resolving coupled climate (`Glevel-8`–`9`) and global `1 × 1` cloud-resolving experiments (`Glevel-10`–`11`).

---

## 4. Candidate Earth System Extensions

Potential physical and biogeochemical extensions beyond the baseline ESM configuration include:
- **Emission-Driven Carbon Cycle (`esm-piControl`)**: Closed 3D atmospheric $\mathrm{CO}_2$ transport coupled to prognostic terrestrial and marine carbon fluxes so atmospheric $\mathrm{CO}_2$ evolves freely.
- **Interactive Atmospheric Chemistry**: Prognostic stratospheric and tropospheric $\mathrm{O}_3$ photochemistry ($\mathrm{HO}_x/\mathrm{NO}_x/\mathrm{ClO}_x$, polar stratospheric clouds), wetland $\mathrm{CH}_4$ emissions, soil $\mathrm{N}_2\mathrm{O}$, and interactive OH lifetime feedbacks.
- **Terrestrial Nutrient Cycling, Deep Permafrost, and Vegetation Demography**: Explicit soil inorganic nitrogen and phosphorus cycles ($\mathrm{NH}_4^+, \mathrm{NO}_3^-$, biological nitrogen fixation, leaching) regulating $\mathrm{CO}_2$ fertilization, deeper anaerobic permafrost carbon and methane dynamics, and size/age-structured vegetation demography with hydraulic mortality.
- **Modal Aerosol Microphysics and Ice-Nucleating Particles**: Size-resolved aerosol modes (nucleation, coagulation, internal mixing), black-carbon deposition darkening on snow and ice, and prognostic ice-nucleating particle (INP) coupling in mixed-phase clouds.
- **High-Top Atmosphere and Ice-Shelf Coupling**: Extended stratosphere/mesosphere (`z_top ~ 80 km`, `70–90` layers) for QBO and sudden stratospheric warming representation, together with sub-ice-shelf ocean basal melting and iceberg calving feedbacks.
- **Optional External Historical and Scenario Forcing**: Modular plug-in support for anthropogenic aerosol emissions, land-use change, volcanic stratospheric aerosol optical depth, and solar spectral irradiance, kept separate from the map-free universal physics core.
