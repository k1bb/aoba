# AOBA Project Guidelines

## 1. Bilingual Documentation Rule (Mandatory)
- Every documentation file in this repository MUST always be created and kept synchronized in **both English and Japanese** (e.g., `README.md` + `README_ja.md`, `docs/architecture.md` + `docs/architecture_ja.md`, and any future `docs/<topic>.md` + `docs/<topic>_ja.md`).
- Whenever a design specification, equation, parameter table, or usage guide is added or updated, update both the English and Japanese versions in the same commit.

## 2. Strict Protection of `references/` Directory
- The `references/` directory stores reference papers and literature for local study and Google Drive backup ONLY.
- **NEVER** stage, commit, or push anything inside `references/` to Git / GitHub (`references/` is protected via `.gitignore`, `.git/info/exclude`, and a local `pre-commit` hook).

## 3. Runtime & Output Conventions
- Always use `/Users/andrew/miniforge/envs/myenv/bin/python` when executing Python commands.
- Simulation outputs, NetCDF dumps, and checkpoints must be written outside the Google Drive repository (defaulting to `~/aoba_outputs` via `aoba.get_output_dir()`).
