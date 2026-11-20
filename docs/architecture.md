# AOBA Model Architecture and Literature-Grounded Formulation

> **Language**: English | [日本語 (architecture_ja.md)](architecture_ja.md)

## 1. Scientific Objective and Documentation Standard

AOBA investigates two fundamental questions:
1. Given **only** external physical boundary conditions—land–sea/ice-sheet geometry and surface elevation from `Earth2014` (Hirt & Rexer, 2015), planetary rotation rate $\Omega$, solar orbital insolation $S_0$, and well-mixed atmospheric gas concentrations ($\mathrm{CO}_2, \mathrm{O}_2$)—can a coupled atmosphere–surface ocean–terrestrial ecosystem autonomously reproduce Earth's climate and biome zones (benchmarked against Beck et al., 2018, 2023)?
2. How does such a minimally constrained model—free of cumulus parameterizations, prescribed ocean $Q$-fluxes, and plant-functional-type (PFT) parameter tables—predict future shifts in global climate zones under elevated $\mathrm{CO}_2$?

**Documentation & Traceability Rule**: Following the mathematical completeness of *Description of MIROC6 AGCM* (2021), *Description of MATSIRO6* (2021), and *Detailed formulation of SCALE-RM* (2025), every continuous equation, coordinate transformation, staggered stencil, and constant implemented in `src/aoba/` must be documented explicitly and linked to its supporting source publication (author, year, section, equation number).

---

## 2. Unified Nonhydrostatic Dynamical Framework and Grid Hierarchy

### 2.1 Global Host Core: Spherical Centroidal Voronoi C-Grid (TRiSK) + Charney–Phillips HEVI
- **Primary References**: Ringler, Ju, & Gunzburger (2008, *Ocean Dynamics*, 58, 475–498); Thuburn, Ringler, Skamarock, & Klemp (2009, *J. Comput. Phys.*, 228, 8321–8335); Ringler, Thuburn, Klemp, & Skamarock (2010, *J. Comput. Phys.*, 229, 3065–3090); Klemp (2011, *Mon. Wea. Rev.*, 139, 2163–2169); Staniforth & Thuburn (2012, *Q. J. R. Meteorol. Soc.*, 138, 1–26); Skamarock et al. (2012, *Mon. Wea. Rev.*, 140, 3090–3105); Gassmann (2013, *Q. J. R. Meteorol. Soc.*, 139, 152–175); Held & Suarez (1994, *Bull. Amer. Meteor. Soc.*, 75, 1825–1830).
- **Governing Equations**: Fully compressible nonhydrostatic flux-form Euler equations conserving dry-air density $\rho_d$, edge-normal horizontal momentum, vertical momentum $\rho_d w$, moist potential temperature $\rho_d \theta_m$ (including vapor-buoyancy virtual effects; Seidel & Yang, 2024), and moisture scalars $(\rho_d q_n, \rho_d q_p)$.
- **Horizontal SCVT Orthogonal C-Grid**: Generated from recursive icosahedral subdivision and spherical Lloyd iterations (Ringler et al., 2008):
  - `Glevel-4`: $N_{\mathrm{cells}} = 2,562$ cells ($\Delta x \approx 240\ \mathrm{km}$), $N_{\mathrm{edges}} = 7,680$.
  - `Glevel-5`: $N_{\mathrm{cells}} = 10,242$ cells ($\Delta x \approx 120\ \mathrm{km}$), $N_{\mathrm{edges}} = 30,720$.
  - Strictly orthogonal primal Voronoi edges and dual Delaunay segments avoid both A-grid $2\Delta x$ pressure/divergence decoupling in vertical velocity $w$ and Cubed-Sphere non-orthogonal corner imprinting. **TRiSK** tangential velocity reconstruction maintains exact stationary geostrophic modes, energy conservation, and potential-vorticity compatibility, supplemented by fourth-order hyperdiffusion/Smagorinsky dissipation of non-divergent hexagonal rotational modes (Skamarock et al., 2012; Gassmann, 2013).
- **Vertical Charney–Phillips Grid & Smoothed Terrain-Following Coordinate**: $N_z = 32$ levels up to $z_{\mathrm{top}} \approx 24\ \mathrm{km}$ ($p_{\mathrm{top}} \approx 25\ \mathrm{hPa}$): ~13 boundary-layer levels below $2\ \mathrm{km}$ ($\Delta z_1 = 60\ \mathrm{m}$), ~14 free-tropospheric levels ($2\text{--}16\ \mathrm{km}$), and 5 coarse lower-stratospheric Rayleigh sponge levels ($16\text{--}24\ \mathrm{km}$). Potential temperature $\theta_m$ and vertical velocity $w$ are co-located at layer interfaces (**Charney–Phillips** staggering; Gassmann, 2013), eliminating the vertical $2\Delta z$ computational buoyancy mode over progressive height-smoothed terrain surfaces (Klemp, 2011).
- **HEVI Time Integration**: Three-stage Runge–Kutta (RK3) with Horizontally Explicit Vertically Implicit (HEVI) acoustic/gravity treatment solved via fused batched Thomas tridiagonal kernels and positive-definite Flux-Corrected Transport (FCT) scalar advection (Skamarock et al., 2012; Nishizawa et al., 2025).

### 2.2 Displacement-Following Aerodynamic Datum and Autonomous Canopy Geometry
- **Primary References**: Hirt & Rexer (2015, *Int. J. Appl. Earth Obs. Geoinf.*, 39, 103–112); Kempes et al. (2011, *PLoS ONE*, 6, e20551); Raupach (1994, *Boundary-Layer Meteorol.*, 71, 211–216); Harman & Finnigan (2007, *Boundary-Layer Meteorol.*, 123, 339–363).
- **Boundary Relief & Exact Surface Mask**: `Earth2014` (`SUR2014`, `TBI2014`, `ICE2014`) defines cell-averaged ice-top/land elevation $z_{\mathrm{topo}}$ and exact algebraic masks distinguishing open water ($\mathrm{SUR} - \mathrm{TBI} > 0$), grounded ice sheets ($\mathrm{ICE} > 0,\ \mathrm{SUR} = \mathrm{TBI}$), and dry land including below-sea-level endorheic basins.
- **Effective Aerodynamic Datum**: Local potential canopy height $h_c$ is predicted from vascular allometry and local water–energy budgets (Kempes et al., 2011), yielding continuous zero-plane displacement $d(h_c, \mathrm{LAI})$ and roughness length $z_{0m}(h_c, \mathrm{LAI})$ (Raupach, 1994) with roughness-sublayer (RSL) stability functions (Harman & Finnigan, 2007). Placing the lower coordinate datum at:
  $$
  z_{\mathrm{datum}} = z_{\mathrm{topo}} + d
  $$
  with $\Delta z_1 = 60\ \mathrm{m}$ ($\tilde{z}_1 = 30\ \mathrm{m}$) guarantees $\tilde{z}_1 > (h_c - d) > z_{0m}$ everywhere.

### 2.3 Embedded 3D Nonhydrostatic Cloud-Resolving Model (CRM) and Blockwise Coupling
- **Primary References**: Khairoutdinov & Randall (2003, *J. Atmos. Sci.*, 60, 607–623); Khairoutdinov, Randall, & DeMott (2005, *J. Atmos. Sci.*, 62, 2136–2154); Yamazaki & Miura (2024, *JAMES*, 16, e2023MS003836); Nishizawa, Tomita, & Team SCALE (2025, *SCALE-RM v5.5.5*, Ch. 2–6 & App. A.2).
- **Unified Solver & Tensor Layout**: Doubly periodic $32\ \mathrm{km} \times 32\ \mathrm{km}$ square orthogonal Arakawa C-grid + vertical Charney–Phillips grid (`[B_crm, nz = 32, ny = 8, nx = 8]`, $\Delta x = \Delta y = 4\ \mathrm{km}$, $\Delta t_{\mathrm{CRM}} = 12\text{--}15\ \mathrm{s}$) sharing the exact same compressible nonhydrostatic HEVI vertical solver, thermodynamics, and FCT tracer advection as the global host core.
- **1-to-1 and 4-to-1 Blockwise Superparameterization**: Supports `1:1` coupling on `Glevel-4` ($B_{\mathrm{crm}} = 2,562$) and `4:1` parent-cell blockwise coupling on `Glevel-5` ($10,242$ host cells at $\sim 120\ \mathrm{km}$ resolution coupled to $B_{\mathrm{crm}} = 2,562$ CRMs; Yamazaki & Miura, 2024), returning 3D convective heating, moistening, and momentum tendencies to the host circulation.

---

## 3. Cloud Microphysics and Four-Band Analytical Two-Stream Radiation

### 3.1 Two-Variable Diagnostic-Phase Cloud Microphysics
- **Primary Reference**: Khairoutdinov & Randall (2003, *J. Atmos. Sci.*, 60, App. A–B).
- **Formulation**: Two moisture scalars—non-precipitating water $q_n = q_v + q_c + q_i$ and precipitating water $q_p = q_r + q_s$—partition condensate $\max(0, q_n - q^*(T, p))$ and precipitation diagnostically between liquid and ice via a linear temperature ramp $\omega(T) \in [0, 1]$ over $[253.15, 273.16]\ \mathrm{K}$, capturing autoconversion, accretion, sedimentation, evaporation/sublimation, fusion latent heat, and cirrus radiative effects.

### 3.2 Lookup-Table-Free Four-Band Two-Stream Radiation
- **Primary References**: Toon et al. (1989, *J. Geophys. Res.*, 94, 16,287–16,301); Lacis & Hansen (1974, *J. Atmos. Sci.*, 31, 118–133); Slingo (1989, *J. Atmos. Sci.*, 46, 1419–1427); Ebert & Curry (1992, *J. Geophys. Res.*, 97, 3831–3836); Weaver & Ramanathan (1995, *J. Geophys. Res.*, 100, 11,585–11,591); Koll & Cronin (2018, *PNAS*, 115, 10293–10298); Wing et al. (2018, *Geosci. Model Dev.*, 11, 793–813); Sherwood et al. (2020, *Rev. Geophys.*, 58, e2019RG000678).
- **Solver Architecture**: Unified tridiagonal two-stream solver (Toon et al., 1989) evaluated every $\Delta t_{\mathrm{rad}} = 900\ \mathrm{s}$ across columns with instantaneous solar-zenith scaling $\mu_0(t)$:
  - **SW1 (UV + Visible / PAR, ~50% of solar constant)**: Analytical ozone absorption (Lacis & Hansen, 1974) with horizontally uniform background $\mathrm{O}_3(p)$ (Wing et al., 2018, Eq. 4) forming the tropopause thermal cap; liquid/ice cloud $\delta$-Eddington scattering (Slingo, 1989; Ebert & Curry, 1992); surface visible albedo $\alpha_{\mathrm{vis}}$; transmitted surface flux directly supplies $\mathrm{PAR}$.
  - **SW2 (Near-Infrared, ~50% of solar constant)**: Analytical $\mathrm{H}_2\mathrm{O}$ absorption (Lacis & Hansen, 1974), cloud scattering/absorption, and vegetation/snow NIR reflectance $\alpha_{\mathrm{nir}}$.
  - **LW1 ($\mathrm{H}_2\mathrm{O}$ + $\mathrm{CO}_2$ Band, fraction $1 - \beta \approx 0.68$)**: Pressure-broadened $\mathrm{H}_2\mathrm{O}$ optical depth plus logarithmic $\mathrm{CO}_2$ optical depth (Weaver & Ramanathan, 1995; Koll & Cronin, 2018; Sherwood et al., 2020) and cloud emissivity, driving tropospheric radiative cooling ($\sim 1.5\text{--}2\ \mathrm{K\ d^{-1}}$).
  - **LW2 (8–12 $\mu\mathrm{m}$ Atmospheric Window, fraction $\beta \approx 0.32$)**: Water-vapor self-continuum absorption ($\propto q_v^2 p / p_0$; Koll & Cronin, 2018) and liquid/ice cloud absorption (Ebert & Curry, 1992), capturing clear-sky OLR linearity, nocturnal surface cooling, and anvil greenhouse trapping.

---

## 4. Subgrid Autonomous Biosphere and 1.5-Layer Voronoi Ekman Slab Ocean

Surface tiles couple directly to overlying atmospheric columns following the SLM/MAML paradigm (Lee & Khairoutdinov, 2015; Lin et al., 2023).

### 4.1 Optimality-Based C₃ and C₄ Photosynthesis and Autonomous C₃/C₄ Partitioning
- **Primary References**: Farquhar, von Caemmerer, & Berry (1980); Bernacchi et al. (2001, 2003); Prentice et al. (2014); Wang et al. (2017); Stocker et al. (2020); Mengoli et al. (2022); Scott & Smith (2022); Luo et al. (2024); Lavergne et al. (2026).
- **Acclimation & Partitioning**: Least-cost C₃ ($\chi_{\mathrm{C3}}$) and C₄ (Scott & Smith, 2022) biochemical capacities acclimate to midday conditions ($\tau_{\mathrm{acclim}} = 15\ \mathrm{d}$; Mengoli et al., 2022), while instantaneous stomatal conductance $g_s$ and surface energy balance ($R_n = H + \lambda E + G$; MATSIRO6 Document Writing Team, 2021) run every step. Natural C₄ fraction $F_4 \in [0, 1]$ is determined dynamically from the optimality-predicted competitive advantage of C₄ versus C₃ assimilation (Luo et al., 2024; Lavergne et al., 2026).

### 4.2 Root-Zone Storage Sizing, Water-Supply Closure, and Prognostic EEO LAI
- **Primary References**: Gao et al. (2014); Davis et al. (2017); Sellers et al. (1986); Cai et al. (2025); Zhou et al. (2025).
- **Drought-Bridging Root Zone & Prognostic LAI**: Ecosystem root-zone storage capacity $W_{\max} = S_R$ adapts to seasonal cumulative water deficits (Gao et al., 2014), bounding instantaneous transpiration via root-zone water supply $S_w(W) = c_w W / S_R$ (Davis et al., 2017). Annual peak greenness $\mathrm{fAPAR}_{\max} = \max\!\left(0,\, \min(\mathrm{fAPAR}_c,\, \mathrm{fAPAR}_w)\right)$ and daily Lambert-$W_0$ steady-state leaf area $L_s$ drive prognostic $\mathrm{LAI}$ ($\alpha = 0.067$; Cai et al., 2025; Zhou et al., 2025), modulating transpiration, roughness, and two-stream VIS/NIR albedo (Sellers et al., 1986; MATSIRO6 Document Writing Team, 2021) alongside a single-layer snow/glacier surface.

### 4.3 1.5-Layer Wind-Driven Voronoi Ekman Slab Ocean and Thermodynamic Sea Ice
- **Primary References**: Codron (2012, *Climate Dynamics*, 38, 379–389); Semtner (1976, *J. Phys. Oceanogr.*, 6, 379–389).
- **Zero-$Q$-Flux Voronoi C-Grid Ekman Ocean**: Discretized on the exact same SCVT Voronoi C-grid (`normal Ekman transport` $= 0$ across coastal edges), surface wind stress drives mixed-layer ($H_{\mathrm{ml}} = 50\ \mathrm{m}$) Ekman transport, coastal/equatorial upwelling from a prognostic return-flow layer, and eddy diffusion with **zero prescribed $Q$-flux** (Codron, 2012), coupled to thermodynamic sea ice below $T_f = 271.35\ \mathrm{K}$ (Semtner, 1976).
