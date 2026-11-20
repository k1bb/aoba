# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **Language**: English | [日本語 (README_ja.md)](README_ja.md)

**AOBA** is a lightweight, GPU-native superparameterized global climate model coupled with an eco-evolutionary optimality (EEO) terrestrial biosphere. It is designed to investigate coupled feedbacks between plant-regulated land surface boundary conditions and the atmospheric general circulation without relying on cumulus parameterizations, prescribed gravity-wave source distributions, or plant-functional-type (PFT) tuning.

## Core Design Principles

1. **Explicit Moist Convection and Gravity Waves (Superparameterization)**: Each column of a global T42 spectral hydrostatic dynamical core embeds a periodic 3D anelastic cloud-resolving model (`8 × 8` horizontal columns, $\Delta x = \Delta y = 4\ \mathrm{km}$, 50 vertical levels reaching $z_{\mathrm{top}} \approx 38\ \mathrm{km}$). Deep convection, cold pools, omnidirectional convective gravity-wave generation, stratospheric wave breaking, and 3D convective momentum transport emerge from resolved nonhydrostatic dynamics.
2. **Parameter-Sparse Autonomous Vegetation (P-Model + EEO LAI)**: Every embedded CRM column over land couples directly to an independent surface tile governed by the leaf-level P-model (Prentice et al., 2014; Stocker et al., 2020) and EEO leaf-area prediction (Cai et al., 2025; Zhou et al., 2025). Canopy fractional absorbed PAR ($\mathrm{fAPAR}$) and leaf area index ($\mathrm{LAI}$) adjust dynamically to the lesser of carbon-cost light limitation and root-zone water-supply limitation without empirical PFT tables.
3. **Displacement-Following Terrain Datum**: The atmospheric lower boundary is defined at the effective aerodynamic datum $z_{\mathrm{datum}} = z_{\mathrm{topo}} + d$ (surface elevation plus zero-plane displacement), preventing tall forest canopies from protruding above the lowest atmospheric model level across realistic global topography.
4. **Analytical Four-Band Two-Stream Radiation & Mixed-Layer Slab Ocean**: A lookup-table-free four-band radiative transfer solver (UV/PAR, near-infrared, $\mathrm{H}_2\mathrm{O}$/$\mathrm{CO}_2$ absorption band, and 8–12 $\mu\mathrm{m}$ atmospheric window) couples clouds, water vapor, stratospheric ozone heating, $\mathrm{CO}_2$, and surface PAR/albedo over a thermodynamic mixed-layer slab ocean and sea-ice surface.
5. **Single-Codebase GPU Portability (`Float32` PyTorch)**: Runs natively on Apple Silicon GPUs (`mps` Unified Memory) for local multi-year integrations and scales without code changes to NVIDIA GPUs (`cuda`).

## Repository Structure

```text
aoba/
├── README.md              # Project overview (English)
├── README_ja.md           # Project overview (Japanese)
├── docs/
│   ├── architecture.md    # Mathematical formulation & coupling specification (English)
│   └── architecture_ja.md # Mathematical formulation & coupling specification (Japanese)
├── references/            # Local literature archive (strictly git-ignored; never pushed)
├── src/
│   └── aoba/
│       ├── __init__.py
│       └── runtime.py     # Device selection (MPS / CUDA / CPU) & output path management
├── tests/                 # Unit and conservation tests
├── scripts/               # Benchmark & experiment runners
└── pyproject.toml
```

## Output Directory Policy

To prevent large NetCDF/checkpoint files from saturating cloud-synced storage, simulation outputs default to `~/aoba_outputs` (configurable via the `AOBA_OUTPUT_DIR` environment variable).

## Development Roadmap

- **Phase 1**: Single-column host (`B = 1`) embedded `8 × 8` 3D CRM + 4-band radiation + subgrid P-model/EEO-LAI radiative–convective equilibrium (RCE) testbed.
- **Phase 2**: Global T42 hydrostatic spectral primitive-equation host core with realistic smoothed topography and Held–Suarez verification.
- **Phase 3**: Global superparameterized coupling (`B = 8192`) over realistic continents, topography, and mixed-layer slab ocean.
- **Phase 4**: Factorial experiments isolating autonomous vegetation feedbacks and convective parameterization artifacts (including radiative vs. physiological $\mathrm{CO}_2$ forcing).

## License

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
