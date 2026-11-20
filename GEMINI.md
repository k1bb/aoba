# AOBA Project Guidelines

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
