# AOBA Project Guidelines

## 0. Core Scientific & Architectural Invariants
Every model component, parameterization, and tooling change in AOBA MUST uphold the following invariants:
- **Minimal Geographic Boundary Prescription**: Except for surface topography / bathymetry / ice-sheet geometry (`Earth2014`) and optional ozone climatology, **NEVER** introduce spatially prescribed geographic maps (PFT distributions, soil/albedo lookup maps, ocean $Q$-fluxes, regional dust-erodibility or aerosol inventories, or latitude-tuned non-orographic GWD source tables) that implicitly constrain the simulation toward present-day Earth climate or biome patterns. Climate zones, ecosystems, circulation, clouds, and wave sources must emerge from universal physical principles and global parameters so planetary, orbital, and radiative climate reorganizations remain unconstrained.
- **Cumulus-Free Explicit Convection**: Never introduce conventional cumulus parameterizations; resolve deep, shallow, continental, and marine convection uniformly via explicit nonhydrostatic dynamics.
- **Seamless Multi-Scale & Unified Parent–Child Formulation**: Support both low-resolution superparameterized (SP) multi-decadal climate integrations (`enable_superparam = true`) and high-resolution cloud-resolving simulations (`enable_superparam = false` / standalone CRM) within a single codebase, using the exact same governing equation set and numerical discretization (flux-form compressible nonhydrostatic Euler equations, orthogonal Arakawa C-grid, Charney–Phillips vertical staggering, RK3-HEVI, Zalesak FCT, and shared physics) for both the global host (parent) and embedded CRM (child).
- **Legible, Modern Python Implementation**: Author all core modules in `src/aoba/` in clean, type-annotated, self-documenting Python with explicit literature citations, avoiding opaque legacy shorthand.
- **Single-Codebase Hardware Portability (Laptop to HPC)**: Maintain automatic hardware detection and fused GPU kernel dispatch (`cuda` → `mps` → `cpu`) so AOBA executes efficiently across personal laptops, cloud GPUs, and supercomputing clusters without HPC-only lock-in.
- **Fully Open & Self-Contained Reproducibility**: Avoid closed dependencies or institution-gated boundary/initial datasets; rely solely on minimal publicly retrievable grids (auto-downloaded) and openly distributable restart states.
- **Centralized Global Parameter Registry & Unified GUI/CUI**: Centralize all tunable global parameters in `config/aoba_default.toml` (`src/aoba/params.py`), operable identically via CUI (`scripts/run_g5_30yr.py`) and the interactive Web workbench (`scripts/aoba_config_gui.py` for configuration, execution, live monitoring, and lightweight analysis/visualization).
- **Bilingual Educational Accessibility (EN/JA)**: Maintain synchronized English and Japanese documentation and UI annotations so undergraduates and international researchers can readily study, configure, and extend the model.

## 1. Bilingual Documentation Rule (Mandatory)
- Every documentation file in this repository MUST always be created and kept synchronized in **both English and Japanese** (e.g., `README.md` + `README_ja.md`, `docs/architecture.md` + `docs/architecture_ja.md`, `docs/formulation.md` + `docs/formulation_ja.md`).
- Follow the clarity and mathematical completeness of `Description of MIROC6 AGCM`, `Description of MATSIRO6`, and `Detailed formulation of SCALE-RM` as documentation standards: write out continuous governing equations, coordinate transforms, staggered-grid spatial discretizations, time-integration steps, physical constants, and execution flowcharts explicitly.
- Every piece of code in `src/aoba/` must cite its supporting peer-reviewed paper or technical manual (author, year, section, equation number).

## 2. Strict Protection of `references/` and `data/` Directories
- `references/` (papers/PDFs) and `data/` (topography & validation datasets) are stored for local use and Google Drive backup ONLY.
- **NEVER** stage, commit, or push anything inside `references/` or `data/` to Git / GitHub (both directories are triple-locked via `.gitignore`, `.git/info/exclude`, and `.git/hooks/pre-commit`).

## 3. Runtime & I/O Conventions
- Always use `/Users/andrew/miniforge/envs/myenv/bin/python` when executing Python commands.
- **Single Native Model I/O Format (`Zarr v3`)**: All simulation history streams, restart checkpoints, and preprocessed mesh/boundary caches read or written by the model core MUST use **CF/UGRID-compliant Zarr v3 (`.zarr`)** exclusively (defaulting to `~/aoba_outputs` via `aoba.get_output_dir()`). NetCDF4 (`.nc`) conversion is strictly separated into standalone post-processing utilities.
