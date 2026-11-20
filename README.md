# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **Language**: English | [日本語 (README_ja.md)](README_ja.md)

**AOBA** is a lightweight, GPU-native superparameterized global nonhydrostatic climate model coupled with an eco-evolutionary optimality (EEO) terrestrial biosphere and a wind-driven Ekman slab ocean. Given only minimal external boundary conditions—**land–sea/ice-sheet geometry, surface topography (`Earth2014`), planetary rotation period, solar insolation, and atmospheric gas concentrations**—AOBA investigates whether the coupled atmosphere–surface ocean–ecosystem system can autonomously generate Earth's climate and biome zones and how such a minimally constrained system predicts future climate-zone shifts without cumulus parameterizations, ocean $Q$-fluxes, or plant-functional-type (PFT) tuning tables.

## Core Design Principles

1. **Unified Compressible Nonhydrostatic Dynamics on Orthogonal C-Grids (SCVT Host + 3D CRM)**:
   - **Global Host Core**: Solves the 3D flux-form compressible nonhydrostatic equations on a **Spherical Centroidal Voronoi Tessellation (SCVT) orthogonal Arakawa C-grid** with **TRiSK** geostrophic/energy-compatible operators and **Charney–Phillips** vertical staggering using Horizontally Explicit Vertically Implicit (HEVI) time integration (Ringler et al., 2008, 2010; Thuburn et al., 2009; Klemp, 2011; Skamarock et al., 2012; Gassmann, 2013). This eliminates polar singularities, spectral Gibbs ringing near steep mountain ranges, and A-grid checkerboard vertical-velocity ($w$) noise.
   - **Embedded 3D CRM (Superparameterization)**: Embeds periodic `8 × 8` 3D nonhydrostatic C-grid/Charney–Phillips CRMs ($\Delta x = \Delta y = 4\ \mathrm{km}$, $N_z = 32$ levels to $z_{\mathrm{top}} \approx 24\ \mathrm{km}$; Khairoutdinov & Randall, 2003; Khairoutdinov et al., 2005; Nishizawa et al., 2025), supporting both 1-to-1 coupling (`Glevel-4`: $2,562$ cells) and 4-to-1 icosahedral blockwise coupling (`Glevel-5`: $10,242$ cells at $\sim 120\ \mathrm{km}$ host resolution paired with $2,562$ CRMs; Yamazaki & Miura, 2024).
2. **PFT-Free Autonomous C₃/C₄ Vegetation, Prognostic LAI, and Root-Zone Sizing**: Land surface tiles couple directly to CRM/host columns (Lee & Khairoutdinov, 2015; Lin et al., 2023) via least-cost C₃ and C₄ optimality models (Prentice et al., 2014; Wang et al., 2017; Stocker et al., 2020; Mengoli et al., 2022; Scott & Smith, 2022; Luo et al., 2024; Lavergne et al., 2026), EEO leaf-area dynamics (Cai et al., 2025; Zhou et al., 2025), and drought-bridging root-zone storage capacity (Gao et al., 2014; Davis et al., 2017). Soil drying regulates transpiration and canopy $\mathrm{LAI}$ dynamically without empirical PFT stress curves.
3. **Displacement-Following Smoothed Terrain Datum & Autonomous Canopy Structure**: Canopy height $h_c$ scales with local resource limits (Kempes et al., 2011), determining zero-plane displacement $d$ and roughness length $z_{0m}$ continuously from $h_c$ and $\mathrm{LAI}$ (Raupach, 1994; Harman & Finnigan, 2007) above the smoothed terrain-following datum $z_{\mathrm{datum}} = z_{\mathrm{topo}} + d$ (Klemp, 2011).
4. **Analytical Four-Band Two-Stream Radiation & 1.5-Layer Voronoi Ekman Slab Ocean**: A lookup-table-free four-band two-stream radiative solver (Lacis & Hansen, 1974; Toon et al., 1989; Slingo, 1989; Ebert & Curry, 1992; Weaver & Ramanathan, 1995; Koll & Cronin, 2018) couples clouds, water vapor, $\mathrm{CO}_2$, and surface PAR/albedo over a zero-$Q$-flux 1.5-layer wind-driven Ekman slab ocean (Codron, 2012) and thermodynamic sea ice (Semtner, 1976) discretized on the exact same Voronoi C-grid.
5. **Single-Codebase GPU Portability (`Float32` PyTorch + Fused Vertical HEVI)**: Runs natively on Apple Silicon GPUs (`mps` Unified Memory, ~1–1.5 GB RAM) at ~7–11 simulated years per day on a Mac mini (`M4 Pro`), and scales without model changes to NVIDIA GPUs (`cuda`).

## Repository Structure

```text
aoba/
├── README.md              # Project overview (English)
├── README_ja.md           # Project overview (Japanese)
├── docs/
│   ├── architecture.md    # Governing equations & literature traceability (English)
│   └── architecture_ja.md # Governing equations & literature traceability (Japanese)
├── data/                  # Local boundary (Earth2014) & validation (Köppen-Geiger) data (git-ignored)
├── references/            # Local literature archive (strictly git-ignored; never pushed)
├── src/
│   └── aoba/
│       ├── __init__.py
│       └── runtime.py     # Device selection (MPS / CUDA / CPU) & path management
├── tests/                 # Unit and conservation tests
├── scripts/               # Benchmark & experiment runners
└── pyproject.toml
```

## Storage & Git Protection Policy

- **Simulation Outputs (`Zarr v3` Native)**: All model history streams, restart checkpoints, and mesh caches use **CF/UGRID-compliant Zarr v3 (`.zarr`)** exclusively (defaulting to `~/aoba_outputs`, configurable via `AOBA_OUTPUT_DIR`); conversion to NetCDF4 (`.nc`) for external tools (`cdo`, Panoply) is handled strictly by standalone post-processing scripts.
- **Protected Local Directories**: Both `references/` and `data/` are backed up locally via Google Drive and triple-locked (`.gitignore`, `.git/info/exclude`, and `.git/hooks/pre-commit`) so they are never pushed to GitHub.

## Development Roadmap

- **Phase 1**: Single-column host (`B = 1`) embedded `8 × 8` 3D C-grid/Charney–Phillips HEVI CRM + 4-band radiation + subgrid C₃/C₄ EEO land / slab-ocean radiative–convective equilibrium (RCE) testbed (Wing et al., 2018).
- **Phase 2**: Global SCVT Voronoi C-grid (TRiSK) + Charney–Phillips HEVI nonhydrostatic host core with smoothed `Earth2014` topography and Held–Suarez verification (Held & Suarez, 1994).
- **Phase 3**: Global superparameterized coupling (`Glevel-4` 1:1 SP and `Glevel-5` 4:1 blockwise SP) over continents, `Earth2014` topography, and 1.5-layer Ekman slab ocean + sea ice, benchmarked against observed Köppen–Geiger climate zones (Beck et al., 2018, 2023).
- **Phase 4**: Future climate-zone projection and factorial experiments under elevated $\mathrm{CO}_2$ (isolating radiative forcing, physiological stomatal/$\mathrm{C}_3\text{--}\mathrm{C}_4$/$\mathrm{LAI}$ feedbacks, and wind-driven Ekman ocean response).

## License

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
