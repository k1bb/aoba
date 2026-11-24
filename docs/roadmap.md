# AOBA Future Development Roadmap

> **Language**: English | [日本語 (roadmap_ja.md)](roadmap_ja.md)

This document records planned architectural, resolution, parallelization, and physical extensions for AOBA beyond the baseline coupled Earth System Model configuration (`Glevel-5`, `8 × 1` CRM, `nz = 44`, `ocean_nz = 18`, 1850 `piControl`).

---

## 1. Immediate Priority: Baseline Spin-Up, Diagnostics, and Quantitative Validation

Before adding new physical complexity, development focuses on verifying and stabilizing the complete coupled baseline:
- **Multi-Stage Equilibrium Spin-Up (`30 + 1,000 + 70 yr`)**: Run a 30-year coupled integration, an accelerated offline 3D SCVT ocean spin-up (up to 1,000 years, auto-stopping below `0.01 K/century` deep-ocean drift), and a 70-year coupled continuation.
- **Physical Conservation & Bug Auditing**: Verify multi-decadal closure of top-of-atmosphere (TOA) radiative balance, global water/salinity conservation, ocean heat content drift, AMOC/ACC transport, sea-ice extent, terrestrial/marine carbon pools, and thermal-acclimation behavior.
- **Observational & Multi-Model Benchmarking**: Quantitatively evaluate equilibrated climatologies against WOA23 ocean hydrography, ERA5 atmospheric reanalysis, and CMIP6 `piControl` multi-model ensembles.

---

## 2. Unified-Grid Resolution & Ocean Vertical Refinement

AOBA maintains a single spherical centroidal Voronoi tessellation (SCVT) horizontal grid across atmosphere, ocean, and land. Matching the first baroclinic Rossby deformation radius $L_d$ (roughly `200 km` near the equator, `20–40 km` in mid-latitudes, and `5–15 km` at high latitudes) defines distinct scientific regimes:

| Grid Level | Global Cells | Spacing $\Delta x$ | Ocean / Coupled Regime | Key Capabilities & Remaining Limits |
|---|---:|---:|---|---|
| **Glevel-5** (baseline) | 10,242 | ~224 km (~2.0°) | Coarse OGCM | Large-scale gyres, AMOC/ACC structure, and global heat/carbon budgets; narrow straits, equatorial undercurrent (EUC), and western boundary separation remain coarse. |
| **Glevel-6** | 40,962 | ~112 km (~1.0°) | Standard ESM entry | Matches standard CMIP6 ~1° ocean horizontal scale for basin-scale circulation and BGC; uniform 1° spacing without equatorial refinement still softens ENSO and western boundary separation. |
| **Glevel-7** | 163,842 | ~56 km (~0.5°) | Non-eddy-resolving ESM sweet spot | Resolves equatorial upwelling, EUC, ENSO dynamics, and major straits (e.g., Indonesian Throughflow) on a uniform grid alongside GM eddy parameterization. |
| **Glevel-8** | 655,362 | ~28 km (~0.25°) | Eddy-permitting | Spontaneous mesoscale eddies in low-to-mid latitudes and sharper SST-front air–sea coupling; high-latitude eddies remain in the gray zone. |
| **Glevel-9** | 2,621,442 | ~14 km (~0.125°) | Mid-latitude eddy-resolving | Realistic Kuroshio/Gulf Stream separation, Southern Ocean eddy heat transport, and seasonal-to-annual global storm-permitting (`1 × 1`) simulations. |
| **Glevel-10 / 11** | 10.5M / 41.9M | ~7.0 / 3.5 km | Global cloud-resolving (`1 × 1`) | Multi-week MJO / DYAMOND-class global cloud-resolving experiments and high-latitude shelf/eddy resolution (requires multi-node GPU domain decomposition). |

### High-Leverage Ocean Upgrade: Vertical Layers (`ocean_nz`)
- Increasing `ocean_nz` from `18` to `40–50` layers (with refined upper-500 m spacing) substantially sharpens the thermocline, EUC, deep-water overflows, and euphotic-zone biogeochemistry.
- Because the coupled atmosphere and embedded CRM account for over 90% of wall-clock runtime, doubling or tripling `ocean_nz` adds only a few percent to total compute cost and is the highest-return ocean upgrade once baseline spin-up is verified.

---

## 3. Multi-GPU Parallelization

The current codebase executes on a single GPU (`cuda` / `mps`), which is well suited for `Glevel-5` (`8 × 1` CRM) multi-decadal runs and embarrassingly parallel ensembles (e.g., `piControl`, paleoclimate, and $\mathrm{CO}_2$ perturbations across independent 1-GPU jobs). Scaling to 3D `8 × 8` CRM, `Glevel-6/7` climate runs, or `1 × 1` global cloud-resolving (`Glevel-9`–`11`) simulations motivates a two-stage multi-GPU design:

1. **Stage 1 — Intra-Node Column Parallelism (`torch.distributed` over NVLink)**:
   - Partition the independent host-column workloads (embedded CRM, 16-band CKD radiation, 2-moment cloud microphysics, MYNN PBL, and biosphere/surface tiles — ~85–90% of total runtime) evenly across 4–8 GPUs within a single node while retaining host SCVT dynamics on a shared or replicated rank.
   - Enables near-linear speedup and VRAM scaling for `Glevel-5` with `8 × 8` 3D CRM and `Glevel-6`–`7` superparameterized climate runs without complex horizontal halo exchanges.
2. **Stage 2 — SCVT Horizontal Domain Decomposition (MPI / NCCL Halo Exchange)**:
   - Partition the SCVT Voronoi mesh across nodes with 1-ring/2-ring halo exchanges for host dynamics, FCT transport, and 3D OGCM/EVP sea-ice operators.
   - Unlocks VRAM beyond a single GPU (`> 40–80 GB`) for eddy-permitting/eddy-resolving coupled climate (`Glevel-8`–`9`) and global `1 × 1` cloud-resolving MJO/DYAMOND experiments (`Glevel-10`–`11`).

---

## 4. Deferred Earth System & Physical Extensions

The following processes are intentionally deferred during the current validation phase and cataloged for potential future exploration:
- **Emission-Driven Carbon Cycle (`esm-piControl`)**: Fully closed 3D atmospheric $\mathrm{CO}_2$ advection coupled to prognostic land and ocean carbon fluxes so atmospheric $\mathrm{CO}_2$ evolves freely rather than being prescribed.
- **Interactive Atmospheric Chemistry**: Prognostic stratospheric/tropospheric $\mathrm{O}_3$ photochemistry ($\mathrm{HO}_x/\mathrm{NO}_x/\mathrm{ClO}_x$, PSCs), wetland $\mathrm{CH}_4$ emissions, soil $\mathrm{N}_2\mathrm{O}$, and interactive OH lifetime feedbacks.
- **Terrestrial Nutrient Cycling, Deep Permafrost, & Demography**: Explicit soil inorganic nitrogen and phosphorus cycles ($\mathrm{NH}_4^+, \mathrm{NO}_3^-$, biological N fixation, leaching) to constrain $\mathrm{CO}_2$ fertilization, deeper anaerobic permafrost C/$\mathrm{CH}_4$ dynamics, and size/age-structured vegetation demography with hydraulic mortality.
- **Modal Aerosol Microphysics & Ice-Nucleating Particles**: Size-resolved aerosol modes (nucleation, coagulation, internal mixing), black-carbon deposition darkening on snow/ice, and prognostic INP coupling in mixed-phase clouds.
- **High-Top Atmosphere & Ice-Shelf Coupling**: Extended stratosphere/mesosphere (`z_top ~ 80 km`, `70–90` layers) for QBO and sudden stratospheric warming dynamics, plus sub-ice-shelf ocean basal melting and iceberg calving feedbacks.
- **Optional External Historical / Scenario Forcing Layer**: Plug-in support for anthropogenic aerosol emissions, land-use change, volcanic stratospheric aerosol optical depth, and solar spectral irradiance, kept strictly modular outside AOBA's map-free universal physics core.
