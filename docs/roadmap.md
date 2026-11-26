# AOBA Development Roadmap

> **Language**: English | [日本語 (roadmap_ja.md)](roadmap_ja.md)

This roadmap outlines upcoming scientific benchmarks, multi-GPU scaling stages, and candidate physical extensions built upon AOBA's current coupled Earth System Model platform (unified SCVT nonhydrostatic dynamics, Online Surrogate Superparameterization `OSSP`, 101 single-source Metal/CUDA kernels, `RRTMGP + 3D-TICA`, 6-class 2-moment microphysics, prognostic 3D $\mathrm{O}_3$, independent fine 64-layer $z^*$ FCT `OG7` ocean, EVP-ITD sea ice, FlexPFT marine BGC + $\mathrm{O}_2$/OMZ, autonomous EEO biosphere + C/N/fire + MAML topo/lakes, natural aerosols, dust, water isotopes, and CMIP7 Zarr v3 output).

---

## 1. Multi-Century Equilibrium Spin-Up and Flagship `piControl` / `abrupt-4×CO2` Evaluation

The immediate scientific milestone is completing and evaluating the full 17-component concentration-driven `ALL-ESM` campaign (`earth_g4z64_g7z64_all_esm_spinup.toml` → `earth_g6z64_g7z64_all_esm_flagship.toml`):
- **Three-Stage Accelerated Equilibrium Spin-Up**: Couple Phase-1 `AG4z64 + OG7z64 ALL-ESM` (~15 yr), Phase-2 self-forced offline `OG7z64 + FlexPFT BGC + O₂/OMZ + EVP-ITD` deep-ocean spin-up (`scripts/run_offline_ogcm_spinup.py`, auto-stopping at `< 0.01 K/century` deep drift), and Phase-3 coupled re-equilibration (~20 yr), followed by Universal Grid Remap (`AG4 → AG6`) verification and flagship **10-yr `piControl` + 10-yr `abrupt-4×CO2`** production at `AG6z64 + OG7z64`.
- **Prequential OSSP Regime-Shift Audit**: Quantify out-of-sample host-atmosphere and land-tile shadow divergence (`enable_distill_shadow_diag = true`) across the instantaneous $4\times\mathrm{CO}_2$ jump and subsequent online adaptation, verifying conservation closure and climate-sensitivity fidelity without pre-trained climate weights.
- **Observational & CMIP Benchmarking**: Evaluate equilibrated climatologies, seasonal precipitation/OLR/cloud patterns, Köppen–Geiger climate zones, autonomous $\mathrm{C}_3/\mathrm{C}_4$/LAI/canopy biomes, AMOC/ACC/OMZ hydrography, and ozone distributions against ERA5, CERES EBAF, GPCP, MODIS, WOA23, and CMIP6/7 ensembles.

---

## 2. Unified-Grid Resolution Hierarchy & Multi-GPU Parallelization

AOBA currently achieves single-GPU saturation across Apple Silicon (`mps`) and NVIDIA A100/H100 (`cuda`) via 101 fused kernels and CUDA Graphs (`~25.4 SYPD` for 3D `8×8` `OSSP + OG7` and `~10.0 SYPD` for full 17-component `AG4z64 + OG7z64 ALL-ESM` on 1 A100). Scaling further to eddy-resolving oceans (`OG8–OG9`) and direct global cloud-resolving (`1 × 1`, `G10–G11`) integrations motivates a two-stage multi-GPU architecture:

| Grid Level | Global Cells | Spacing $\Delta x$ | Coupled / Ocean Regime | Target Parallelization |
|---|---:|---:|---|---|
| **Glevel-4 / 5** | 2,562 / 10,242 | ~448 / 224 km | Rapid OSSP spin-up & exploratory coupled ESM | 1 GPU (Mac MPS / A100) |
| **Glevel-6** | 40,962 | ~112 km (~1.0°) | Flagship HighResMIP-class superparameterized ESM | 1 GPU (A100/H100) |
| **Glevel-7** | 163,842 | ~56 km (~0.5°) | Eddy-permitting ocean (`OG7z64`), EUC, ENSO & major straits | 1 GPU (independent ocean) / Stage 1 |
| **Glevel-8** | 655,362 | ~28 km (~0.25°) | Low/mid-latitude mesoscale eddy-permitting ocean & sharp SST fronts | Stage 1–2 (4–8 GPUs) |
| **Glevel-9** | 2,621,442 | ~14 km (~0.125°) | Mid-latitude eddy-resolving ocean & global storm-permitting (`1 × 1`) | Stage 2 (Multi-node NCCL/MPI) |
| **Glevel-10 / 11** | 10.5M / 41.9M | ~7.0 / 3.5 km | Global cloud-resolving (`1 × 1`) DYAMOND / MJO integrations | Stage 2 (Multi-node NCCL/MPI) |

1. **Stage 1 — Intra-Node Column Parallelism (`torch.distributed` over NVLink)**: Partition independent column physics (teacher CRMs, radiation, microphysics, MAML land tiles) across 4–8 GPUs on a single node without horizontal halo exchange, scaling large 3D `RCE_large` CRMs and `Glevel-7` atmospheres linearly.
2. **Stage 2 — SCVT Horizontal Domain Decomposition (MPI / NCCL Halo Exchange)**: Partition primal/dual SCVT cells across nodes with 1-ring/2-ring halo exchanges for host dynamics, FCT tracers, 3D OGCM, and EVP sea ice to exceed single-GPU VRAM (`> 80 GB`) at `Glevel-8`–`11`.

---

## 3. Candidate Next-Generation Physical & Biogeochemical Extensions

Extensions beyond the current 17-component ESM suite include:
- **Terrestrial Phosphorus Cycle & Size-Structured Vegetation Demography**: Coupling inorganic/organic soil phosphorus cycling alongside the existing C–N cycle to constrain tropical forest productivity, plus height/age-structured demographic competition and hydraulic drought mortality.
- **Modal Aerosol Microphysics**: Extending the current 4-species mass aerosols (`sea salt, sulfate, BC, OC/SOA`) and dust to internally mixed lognormal modes with explicit nucleation, coagulation, and sulfate-coated dust INP aging.
- **Sub-Ice-Shelf Ocean Cavities & Interactive Iceberg Drift**: Coupling floating ice-shelf basal melting and Lagrangian/Eulerian iceberg drift to the existing dynamic SIA ice-sheet and 3D `z*` OGCM for transient Heinrich-event and Antarctic retreat simulations.
- **Mesosphere-Extended High-Top (`z_top ~ 80–90 km`) & Coupled Stratospheric Chemistry–QBO**: Expanding vertical top height above `65 km` with interactive $\mathrm{H}_2\mathrm{O}$ methane oxidation and detailed halogen/tropospheric $\mathrm{VOC}-\mathrm{NO}_x-\mathrm{O}_3$ chemistry.
- **Emission-Driven Multi-Millennial Paleoclimate & Glacial Cycles**: Long transient integrations combining the existing Milankovitch orbital module, silicate weathering, prognostic 3D $\mathrm{CO}_2/\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$, dynamic SIA ice sheets, and dynamic lakes across full 100 kyr glacial–interglacial transitions.
