# AOBA Earth System Model — Physical & Numerical Formulation

## Contents

1. [System Architecture, Physical Constants, Calendar & Orbital Solar Geometry](#1-system-architecture-physical-constants-calendar--orbital-solar-geometry)
2. [Spatial Discretization: SCVT Voronoi Host, Embedded CRM & Vertical STF Coordinate](#2-spatial-discretization-scvt-voronoi-host-embedded-crm--vertical-stf-coordinate)
3. [Moist Compressible Thermodynamics & Mixed-Phase Saturation Adjustment](#3-moist-compressible-thermodynamics--mixed-phase-saturation-adjustment)
4. [Atmospheric Radiative Transfer: Selectable Schemes (`rad_scheme`) & 3D TICA](#4-atmospheric-radiative-transfer-selectable-schemes-rad_scheme--3d-tica)
5. [Cloud Microphysics: Selectable Schemes (`microphysics_scheme`) & Non-Inductive Lightning / $\mathrm{LNO}_x$](#5-cloud-microphysics-selectable-schemes-microphysics_scheme--non-inductive-lightning--lno_x)
6. [Boundary-Layer Turbulence, Roughness Sublayer (RSL), MYNN 2.5 & 3D Smagorinsky–Lilly SGS](#6-boundary-layer-turbulence-roughness-sublayer-rsl-mynn-25--3d-smagorinskylilly-sgs)
7. [Land Surface, Autonomous Two-Leaf $\mathrm{C}_3/\mathrm{C}_4$ EEO Biosphere, Soil Hydrology, Terrestrial C/N, Wildfire & River Routing](#7-land-surface-autonomous-two-leaf-c_3c_4-eeo-biosphere-soil-hydrology-terrestrial-cn-wildfire--river-routing)
8. [Global Nonhydrostatic SCVT Dynamics, Embedded CRM, Gravity-Wave Drag & SP Coupling](#8-global-nonhydrostatic-scvt-dynamics-embedded-crm-gravity-wave-drag--sp-coupling)
9. [Ocean Circulation (`ocean_mode`), Independent Fine Ocean, Surface Waves, EVP–ITD Sea Ice & FlexPFT Marine BGC](#9-ocean-circulation-ocean_mode-independent-fine-ocean-surface-waves-evpitd-sea-ice--flexpft-marine-bgc)
10. [Atmospheric Chemistry (`o3_mode`), Natural/Volcanic Aerosols, Carbon & Trace-Gas Cycles (`co2_mode`) & Water Isotopes](#10-atmospheric-chemistry-o3_mode-naturalvolcanic-aerosols-carbon--trace-gas-cycles-co2_mode--water-isotopes)
11. [Dynamic Ice-Sheet Model, Milankovitch Orbital Evolution & General Planetary Presets](#11-dynamic-ice-sheet-model-milankovitch-orbital-evolution--general-planetary-presets)
12. [Centralized 324-Parameter TOML Registry, Diagnostics (Köppen–Geiger / CMIP7), Zarr v3 & AOBA Studio](#12-centralized-324-parameter-toml-registry-diagnostics-köppengeiger--cmip7-zarr-v3--aoba-studio)

---

## 1. System Architecture, Physical Constants, Calendar & Orbital Solar Geometry

### 1.1 System Architecture, Design Principles & Coupled Time-Stepping Sequence (`src/aoba/coupled_model.py`)

AOBA is a coupled Earth System Model (ESM) uniting compressible nonhydrostatic atmospheric dynamics, broadband radiative transfer, selectable cloud microphysics, boundary-layer turbulence and roughness-sublayer (RSL) similarity, multi-layer freeze–thaw Richards soil hydrology, an autonomous two-leaf $\mathrm{C}_3/\mathrm{C}_4$ Eco-Evolutionary Optimality (EEO) terrestrial biosphere, coupled terrestrial carbon–nitrogen–permafrost–SPITFIRE wildfire dynamics, Priority-Flood river routing, selectable ocean circulation (1.5-layer or 3D SCVT OGCM), prognostic surface waves, five-category EVP–ITD dynamic sea ice, FlexPFT marine biogeochemistry and dissolved $\mathrm{O}_2$/OMZ denitrification, 3D ozone photochemistry, natural and stratospheric volcanic aerosols, closed-loop $\mathrm{CO}_2/\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$ and silicate weathering, 3D stable water isotopologues ($\mathrm{H}_2^{18}\mathrm{O}$, $\mathrm{HDO}$), and dynamic SIA ice sheets within a single codebase.

Spatially fixed present-day Earth maps (PFT vegetation maps, regional dust-erodibility maps, ocean heat-flux corrections, latitude-tuned non-orographic gravity-wave source tables) and empirical cumulus parameterizations are eliminated. External geographic boundary fields are restricted to surface elevation, ocean bathymetry, initial ice-sheet geometry (`Earth2014`), and optional ozone climatology. Both the global host grid and the embedded Cloud-Resolving Model (CRM; configurable horizontal spacing `crm_dx`, default $4\ \mathrm{km}$) integrate the same flux-form compressible nonhydrostatic Euler equations, switching seamlessly between superparameterization (`enable_superparam = true`) and standalone global cloud-resolving integration (`enable_superparam = false`). Major physical scheme selectors (`rad_scheme`, `microphysics_scheme`, `ocean_mode`, `o3_mode`, `co2_mode`, `planet_preset`) are configured centrally in [`config/aoba_default.toml`](../config/aoba_default.toml).

![Figure 1.1: AOBA Unified Earth System Architecture and Multi-Scale Coupling](assets/fig_01_aoba_overview.svg)

Each host time step $\Delta t_{\mathrm{host}}$ (default $720\ \mathrm{s}$) in `step_aoba_model` executes the following sequence:
1. Orbital timing, insolation, and solar-noon check: `AobaCalendar` advances seasonal time $t_{\mathrm{seas}}$ and diurnal time $t_{\mathrm{diur}}$, applying optional Milankovitch orbital evolution, 11-year Schwabe solar-cycle modulation, and tidal-lock or Keplerian insolation.
2. Subcolumn meteorology extraction: lowest-level winds, temperature, humidity, and column profiles are assembled over $M_{\mathrm{sub}} = N_c N_y N_x$ subcolumns, diagnosing surface pressure $p_{\mathrm{sfc}}$ and reference height $z_{\mathrm{ref,rel}}$.
3. Surface albedos and strided independent-column radiation (every `rad_stride` steps, default $5 \times 720\ \mathrm{s} = 3600\ \mathrm{s}$): evaluates two-band direct/diffuse albedos, composite ocean/ice radiative temperature $T_{\mathrm{rad,ocn}}$, 2M/6C cloud effective radii, optional 3D TICA cloud shift, aerosol/volcanic optics, ozone VMR, and the active radiation scheme.
4. Subgrid MAML surface and ocean/sea-ice step: applies optional MAML terrain downscaling and current-relative air–sea winds, advances `step_surface`, applies wetland evaporation enhancement and wave/sea-spray enthalpy and drag feedbacks, integrates `step_ogcm_voronoi` (with optional independent fine ocean coupler and river routing) or `step_ekman_ocean_voronoi`, and transfers ocean increments $(\Delta T_{\mathrm{ml}}, \Delta m_{\mathrm{ice}}, \Delta\mathrm{SWE}_{\mathrm{ice}})$ back to subcolumns with exact freezing-enthalpy conservation.
5. Orographic and non-orographic gravity-wave drag: reconstructs host cell winds, evaluates Lott–Miller orographic drag and optional frontogenesis/convection non-orographic GWD (forced by previous-step $Q_{\mathrm{conv}}$), and projects tendencies onto SCVT edges.
6. Host SCVT RK3–HEVI dynamics and batched Zalesak FCT: integrates $n_{\mathrm{hdyn}}$ CFL-determined calls to `step_scvt_rk3_hevi`, averages mass fluxes, and advects all tracers ($q_n, q_p, q_{\mathrm{dust}}, q_{\mathrm{ss}}, q_{\mathrm{so4}}, q_{\mathrm{bc}}, q_{\mathrm{oc}}, \chi_{\mathrm{O_3}}, \chi_{\mathrm{CO_2}}, q_v^{18}, q_v^{\mathrm{D}}$) in a single `fct_advect_scvt_batch` call, yielding large-scale tendencies $(\partial\theta_m/\partial t)_{\mathrm{LS}}, (\partial q_n/\partial t)_{\mathrm{LS}}$.
7. 1:1 Superparameterization coupling (when `enable_superparam = true`): synchronizes CRM mean density with the host while preserving perturbations, runs `step_crm_physics_and_dynamics` (RK3–HEVI + 3D SGS + MYNN 2.5 + microphysics), synchronizes back to the host via exact mass-weighted averaging, projects mean momentum feedback onto SCVT edges, and diagnoses convective heating $Q_{\mathrm{conv}}$.
8. Chemistry, biogeochemistry, and phenology: evaluates CRM lightning/$\mathrm{LNO}_x$, mineral dust, Priestley–Taylor potential evapotranspiration $\mathrm{PET}$, solar-noon EEO enzyme acclimation, terrestrial C/N and SPITFIRE wildfire, natural aerosols and BVOC SOA, SNICAR snow darkening, silicate weathering, marine BGC atmospheric deposition, prognostic $\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$, volcanic aerosols, prognostic $\mathrm{CO}_2$, prognostic 3D $\mathrm{O}_3$, 3D water isotopologues, daily LAI/root/height update, and monthly Köppen–Geiger accumulators.
9. Annual dynamic ice sheet and vertical remapping: updates Glen SIA ice thickness and isostasy at annual boundaries and executes hydrostatic barometric STF vertical remapping whenever smoothed terrain changes by $\ge 0.05\ \mathrm{m}$.

The hydrostatically extrapolated surface pressure $p_{\mathrm{sfc}}$, canopy-displaced reference height $z_{\mathrm{ref,rel}}$, radiatively balanced composite ocean/sea-ice skin temperature $T_{\mathrm{rad,ocn}}$, and Priestley–Taylor (1972) potential evapotranspiration $\mathrm{PET}$ evaluated in Steps 2, 3, and 8 are defined as:

$$
p_{\mathrm{sfc}} = p_0 + g\,\rho_{d,0}(1 + q_{v,0})(z_{c,0} - z_{w,0}) \tag{1.1}
$$

$$
z_{\mathrm{ref,rel}} = \max\!\big(z_{\mathrm{rel},0} - 0.65\,h_c,\ 5.0\ \mathrm{m}\big) \tag{1.2}
$$

$$
T_{\mathrm{rad,ocn}} = \Big[(1 - f_{\mathrm{ice}})T_{\mathrm{ml}}^4 + f_{\mathrm{ice}}T_{\mathrm{ice}}^4\Big]^{1/4} \tag{1.3}
$$

$$
\mathrm{PET} = 1.26\,\frac{s}{s + \gamma_{\mathrm{psy}}}\,\frac{R_n^+}{L_{v0}} \tag{1.4}
$$

where $p_0, \rho_{d,0}, q_{v,0}$ are lowest-layer pressure, dry-air density, and water-vapor mixing ratio (centered at $z_{c,0}$ above surface $z_{w,0}$, height $z_{\mathrm{rel},0} = z_{c,0} - z_{w,0}$), $h_c$ is canopy height, $f_{\mathrm{ice}}, T_{\mathrm{ml}}, T_{\mathrm{ice}}$ are sea-ice fraction, mixed-layer temperature, and sea-ice surface temperature, $s = de_{sw}/dT$ is the saturation vapor-pressure slope, $\gamma_{\mathrm{psy}} = c_{pd} p_{\mathrm{sfc}}/(\varepsilon_w L_{v0})$ is the psychrometric constant, and $R_n^+ = \max(R_n, 0)$ is positive surface net radiation. Eq. (1.1) accounts for the lowest half-layer hydrostatic mass, and Eq. (1.3) preserves exact total upward longwave emission from mixed ocean/sea-ice cells under the Stefan–Boltzmann law.

### 1.2 Fundamental Physical Constants (`src/aoba/constants.py`)

All fundamental constants reside in `src/aoba/constants.py` and the `[constants]` TOML section; `recompute_derived_constants()` updates derived quantities ($c_{vd}, \kappa_d, \gamma_d, \varepsilon_w, R_v/R_d, \delta_v, L_{s0}$) automatically when planetary parameters are modified.

| Symbol | Default Value | Unit | Physical Definition |
| :--- | :--- | :--- | :--- |
| $R_E$ | $6.371 \times 10^6$ | $\mathrm{m}$ | Mean planetary radius (`earth_radius`) |
| $g$ | $9.79764$ | $\mathrm{m\ s^{-2}}$ | Surface gravitational acceleration (`gravity`) |
| $\Omega_E$ | $7.292115 \times 10^{-5}$ | $\mathrm{rad\ s^{-1}}$ | Reference sidereal rotation rate (`omega_earth`) |
| $T_{\mathrm{day}}, N_{\mathrm{yr}}$ | $86400.0,\ 365.2425$ | $\mathrm{s},\ \mathrm{d}$ | Solar day length and tropical year duration |
| $S_0, \varepsilon, e, \varpi$ | $1361.0,\ 23.44^\circ,\ 0.0167,\ 283.0^\circ$ | $\mathrm{W\ m^{-2}}, ^\circ, -, ^\circ$ | Solar constant, obliquity, eccentricity, longitude of perihelion |
| $R_d, c_{pd}, c_{vd}$ | $287.04,\ 1004.64,\ 717.60$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Dry-air gas constant and isobaric/isochoric specific heats |
| $R_v, c_{pv}$ | $461.50,\ 1846.0$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Water-vapor gas constant and isobaric specific heat |
| $\varepsilon_w, \delta_v$ | $0.62197,\ 0.60779$ | $-$ | $R_d/R_v$ and virtual temperature coefficient $R_v/R_d - 1$ |
| $p_0, T_{\mathrm{freeze}}, T_0, e_0$ | $10^5,\ 273.15,\ 273.16,\ 611.2$ | $\mathrm{Pa},\ \mathrm{K},\ \mathrm{K},\ \mathrm{Pa}$ | Reference pressure, freezing point, triple point temperature and vapor pressure |
| $L_{v0}, L_{f0}, L_{s0}$ | $2.501\times 10^6,\ 3.337\times 10^5,\ 2.8347\times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Latent heats of vaporization, fusion, and sublimation |
| $\sigma_{\mathrm{SB}}, R^*$ | $5.670374419\times 10^{-8},\ 8.314462618$ | $\mathrm{W\ m^{-2}\ K^{-4}},\ \mathrm{J\ mol^{-1}\ K^{-1}}$ | Stefan–Boltzmann constant and universal gas constant |
| $\kappa, X_{\mathrm{O}_2}$ | $0.40,\ 0.20946$ | $-,\ \mathrm{mol\ mol^{-1}}$ | von Kármán constant and atmospheric $\mathrm{O}_2$ volume mixing ratio |
| $\rho_{\mathrm{sw}}, c_{p,\mathrm{sw}}, T_{f,\mathrm{ocn}}$ | $1025.0,\ 3990.0,\ 271.35$ | $\mathrm{kg\ m^{-3}},\ \mathrm{J\ kg^{-1}\ K^{-1}},\ \mathrm{K}$ | Seawater reference density, specific heat, and freezing point |
| $\rho_{\mathrm{ice}}, k_{\mathrm{ice}}, k_{\mathrm{snow}}$ | $917.0,\ 2.034,\ 0.3096$ | $\mathrm{kg\ m^{-3}},\ \mathrm{W\ m^{-1}\ K^{-1}}$ | Ice density and sea-ice/snow thermal conductivities |

### 1.3 Calendar Systems and Orbital Time Mapping (`src/aoba/calendar.py`)

Three calendar systems are supported: `gregorian` (proleptic Gregorian including negative BCE years, with leap years divisible by 4 except centuries not divisible by 400), `noleap` (365 days/year), and `360_day` ($12 \times 30 = 360$ days/year). For elapsed time $t\ [\mathrm{s}]$, day fraction $f_{\mathrm{day}}$, decimal calendar year $Y_{\mathrm{dec}}$, and vernal-equinox-referenced orbital year fraction $f_{\mathrm{yr}}$ are:

$$
\begin{aligned}
f_{\mathrm{day}} &= \frac{t \bmod T_{\mathrm{day}}}{T_{\mathrm{day}}}, \\
Y_{\mathrm{dec}} &= Y + \frac{(\mathrm{doy} - 1) + f_{\mathrm{day}}}{N_{\mathrm{days}}(Y)}, \\
f_{\mathrm{yr}} &= \frac{(\mathrm{doy} - 1 + f_{\mathrm{day}} - d_{\mathrm{VE}}) \bmod N_{\mathrm{days}}(Y)}{N_{\mathrm{days}}(Y)}
\end{aligned} \tag{1.5}
$$

where $Y$ is calendar year, $\mathrm{doy}$ is 1-based day of year, $N_{\mathrm{days}}(Y)$ is year length, and $d_{\mathrm{VE}}$ is the vernal-equinox day offset (80 for `360_day`, 79 for leap Gregorian, 78 for non-leap). In `insolation_mode = "perpetual_date"`, seasonal orbital phase is frozen at noon ($f_{\mathrm{day}} = 0.5$) of the specified date.

### 1.4 Orbital Solar Geometry and Zenith Angle (`src/aoba/radiation.py`, `src/aoba/coupled_model.py`)

Following Berger (1978) and Davis et al. (2017), starting from the eccentric anomaly $E_0$ at vernal equinox ($\nu_0 = -\varpi$), the mean anomaly $M$, third-order equation-of-center true anomaly $\nu$, and true solar longitude $\lambda_s$ are evaluated as:

$$
\begin{aligned}
E_0 &= 2\arctan\!\left(\sqrt{\frac{1-e}{1+e}}\tan\frac{-\varpi}{2}\right), \\
M &= E_0 - e\sin E_0 + 2\pi f_{\mathrm{yr}}
\end{aligned} \tag{1.6}
$$

$$
\begin{aligned}
\nu &= M + \left(2e - \frac{1}{4}e^3\right)\sin M + \frac{5}{4}e^2\sin(2M) + \frac{13}{12}e^3\sin(3M), \\
\lambda_s &= \nu + \varpi
\end{aligned} \tag{1.7}
$$

The Keplerian inverse-square Earth–Sun distance factor $(a/r)^2$, top-of-atmosphere solar flux $S_{\mathrm{TOA}}$, and solar declination $\delta$ follow directly:

$$
\begin{aligned}
\left(\frac{a}{r}\right)^2 &= \left[\frac{1 + e\cos\nu}{1 - e^2}\right]^2, \\
S_{\mathrm{TOA}} &= S_0\left(\frac{a}{r}\right)^2, \\
\sin\delta &= \sin\varepsilon\sin\lambda_s
\end{aligned} \tag{1.8}
$$

At latitude $\phi$ and longitude $\lambda$, local solar hour angle $H \in [-\pi, \pi]$ (zero at local noon) and instantaneous solar zenith cosine $\mu_0$ (`diurnal_cycle = true`) are:

$$
\begin{aligned}
H &= \operatorname{atan2}\!\Big(\sin(2\pi f_{\mathrm{day}} + \lambda - \pi),\ \cos(2\pi f_{\mathrm{day}} + \lambda - \pi)\Big), \\
\mu_0(\phi, \lambda, t) &= \max\!\big(0,\ \sin\phi\sin\delta + \cos\phi\cos\delta\cos H\big)
\end{aligned} \tag{1.9}
$$

When diurnal cycling is disabled (`diurnal_cycle = false`), the exact 24-hour mean cosine $\bar{\mu}_0$ is computed from the half-daylight hour angle $h_0$ ($\pi$ in polar day, $0$ in polar night):

$$
\begin{aligned}
h_0 &= \arccos\!\big(\operatorname{clamp}(-\tan\phi\tan\delta, -1, 1)\big), \\
\bar{\mu}_0(\phi, t) &= \max\!\left(0,\ \frac{h_0\sin\phi\sin\delta + \cos\phi\cos\delta\sin h_0}{\pi}\right)
\end{aligned} \tag{1.10}
$$

---

## 2. Spatial Discretization: SCVT Voronoi Host, Embedded CRM & Vertical STF Coordinate

### 2.1 Icosahedral SCVT Voronoi Mesh and Arakawa C-Grid Geometry (`src/aoba/scvt_mesh.py`)

The horizontal host uses a quasi-uniform pole-singularity-free Spherical Centroidal Voronoi Tessellation (SCVT; Ringler et al., 2008, 2010). An icosahedron on the unit sphere is recursively subdivided `glevel` $= g$ times and relaxed via 10–12 Lloyd iterations so generator points coincide with spherical Voronoi centroids. Total cell count $N_c$ (12 pentagons and $N_c - 12$ hexagons), dual triangle count $N_v$, edge count $N_e$, and exact spherical triangle area $A_\triangle$ (Eriksson's formula) are:

$$
\begin{aligned}
N_c &= 10\cdot 4^g + 2, \qquad N_v = 2N_c - 4, \qquad N_e = 3N_c - 6, \\
A_\triangle(\mathbf{a},\mathbf{b},\mathbf{c}) &= 2\operatorname{atan2}\!\Big(|\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})|,\ \max\big(1 + \mathbf{a}\cdot\mathbf{b} + \mathbf{b}\cdot\mathbf{c} + \mathbf{c}\cdot\mathbf{a},\ 10^{-15}\big)\Big)
\end{aligned} \tag{2.1}
$$

Each edge $e$ connects adjacent cells $(c_0, c_1)$ and dual vertices $(v_0, v_1)$ with mutually orthogonal unit normal $\mathbf{n}_e$ ($c_0\to c_1$) and unit tangent $\mathbf{t}_e = \mathbf{r}_e\times\mathbf{n}_e$. Primal great-circle center distance $d_e$, dual Voronoi edge length $l_e$, and diamond area $A_e$ are:

$$
\begin{aligned}
d_e &= R_E\arccos(\mathbf{r}_{c_0}\cdot\mathbf{r}_{c_1}), \\
l_e &= R_E\arccos(\mathbf{v}_0\cdot\mathbf{v}_1), \\
A_e &= \frac{1}{2}\,d_e l_e
\end{aligned} \tag{2.2}
$$

Mass and thermodynamic scalars are stored at Voronoi cell centers and normal velocity $u_e$ at edge midpoints (orthogonal Arakawa C-grid). In superparameterized runs, each host cell embeds a periodic 2D/3D CRM with configurable horizontal spacing $\Delta x = \Delta y$ (`crm_dx`, default $4\ \mathrm{km}$) and column array (`crm_nx` $\times$ `crm_ny`), coupled 1:1 to independent subgrid land tiles (MAML).

The SCVT Voronoi Arakawa C-grid exhibits the following structural advantages and numerical limitations, together with AOBA's stabilization measures:
- Advantages:
  1. Absence of polar singularities and quasi-uniform resolution: unlike latitude–longitude grids, meridian convergence at the poles does not occur, eliminating polar filters and maintaining a uniform global advective/acoustic CFL time step.
  2. Exact primal–dual orthogonality and global conservation: Voronoi edges are orthogonal to primal segments joining cell centers, permitting concise second-order finite-volume divergence and horizontal pressure-gradient operators that satisfy the discrete Gauss divergence theorem ($\sum_i A_i (\nabla\cdot F)_i \equiv 0$) to machine precision.
  3. Isotropic wave dispersion and exact energy conservation: the Arakawa C-grid provides accurate inertia–gravity wave dispersion relative to the Rossby deformation radius, and skew-symmetric TRiSK Coriolis weights (Eq. 2.5) preserve steady geostrophic modes and kinetic energy (Thuburn et al., 2009; Ringler et al., 2010).
  4. Smooth variable-resolution extensibility: centroidal Voronoi tessellations generalize naturally to regionally refined meshes via a spatial density function without hanging nodes.
- Limitations and mitigations in AOBA:
  1. Computational modes from the 2:1 vorticity-to-mass DOF mismatch (Thuburn, 2008; Gassmann, 2011): hexagonal meshes possess $N_e \approx 3N_c$ edge velocities and $N_v \approx 2N_c$ dual-triangle vorticity DOFs per $N_c$ mass DOFs, admitting spurious grid-scale inertia–gravity and stationary Rossby modes (Hollingsworth-type instability). AOBA combines exact skew-symmetric TRiSK weights with scale-selective fourth-order hyperviscosity and horizontal divergence damping (Section 8.2) to suppress grid-scale noise without damping synoptic waves.
  2. Local first-order tangential reconstruction accuracy and pentagon imprinting (Peixoto, 2016; Gassmann, 2018): near the 12 pentagonal cells or where primal and dual edge midpoints are offset, TRiSK tangential velocity reconstruction degrades locally to first order. AOBA applies 10–12 Lloyd SCVT iterations so generators match Voronoi centroids, minimizing geometric distortion and pentagon imprinting.
  3. Indirect memory access overhead: unstructured cell–edge gather/scatter and 10-edge TRiSK stencils incur higher memory-indexing overhead than structured grids; AOBA implements all horizontal operators via batched tensor gathers and fused Metal/CUDA kernels.

![Figure 2.1: Horizontal SCVT-CRM Multi-Scale Mesh and Vertical Charney-Phillips Staggering](assets/fig_02_grid_staggering.svg)

### 2.2 Discrete SCVT Operators, Skew-Symmetric TRiSK Weights & Wind Reconstruction

Cell-centered divergence $(\nabla\cdot F)_i$, dual-vertex relative vorticity $\zeta_v$, edge vector Laplacian $(\nabla^2 u)_e$, and cell scalar Laplacian $(\nabla^2\phi)_i$ follow Ringler et al. (2010):

$$
\begin{aligned}
(\nabla\cdot F)_i &= \frac{1}{A_i}\sum_{e\in E(i)} s_{i,e}\,l_e F_e, \\
\zeta_v &= \frac{1}{A_v}\sum_{e\in E(v)} s_{v,e}\,d_e u_e
\end{aligned} \tag{2.3}
$$

$$
\begin{aligned}
(\nabla^2 u)_e &= \frac{(\nabla\cdot u)_{c_1} - (\nabla\cdot u)_{c_0}}{d_e} - \frac{\zeta_{v_1} - \zeta_{v_0}}{\max(l_e, 100)}, \\
(\nabla^2 \phi)_i &= \nabla\cdot\!\left(\frac{\phi_{c_1}-\phi_{c_0}}{d_e}\right)
\end{aligned} \tag{2.4}
$$

where $A_i, A_v$ are cell and dual-triangle areas, and $s_{i,e}, s_{v,e} \in \{+1, -1\}$ are outward normal and counterclockwise circulation signs. Tangential velocity reconstruction weights $w_{e,e'}$ (Thuburn et al., 2009; Ringler et al., 2010, Eq. 23) are built from cumulative kite-area fractions $t_{e,e'}$ around each cell $i \in \{c_0, c_1\}$ and explicitly skew-symmetrized so Coriolis work vanishes to floating-point precision:

$$
\begin{aligned}
w_{e,e'}^{\mathrm{raw}} &= s_{i,e}\,s_{i,e'}\,t_{e,e'}\,\frac{l_{e'}}{d_e}, \\
w_{e_1,e_2} &= \frac{g_{e_1} w_{e_1,e_2}^{\mathrm{raw}} - g_{e_2} w_{e_2,e_1}^{\mathrm{raw}}}{2 g_{e_1}}, \qquad g_e = l_e d_e \implies l_{e_1}d_{e_1}w_{e_1,e_2} + l_{e_2}d_{e_2}w_{e_2,e_1} = 0
\end{aligned} \tag{2.5}
$$

Cell-centered zonal/meridional wind reconstruction $(u_i, v_i)$ (Perot, 2000; Ringler et al., 2010, Eq. 29) and edge-normal tendency projection $v_e$ are:

$$
\begin{aligned}
(u_i, v_i) &= \frac{1}{A_i}\sum_{e\in E(i)} \frac{1}{2}\,d_e l_e u_e \big(\mathbf{n}_e\cdot\hat{\mathbf{e}}_\lambda,\ \mathbf{n}_e\cdot\hat{\mathbf{e}}_\phi\big), \\
v_e &= \frac{1}{2}\Big[(u_{c_0}+u_{c_1})(\mathbf{n}_e\cdot\hat{\mathbf{e}}_\lambda) + (v_{c_0}+v_{c_1})(\mathbf{n}_e\cdot\hat{\mathbf{e}}_\phi)\Big]
\end{aligned} \tag{2.6}
$$

### 2.3 `Earth2014` Surface Aggregation and Lott–Miller Subgrid Orographic Tensor

5-arc-minute ($2160\times 4320$) `Earth2014` datasets (`SUR2014`, `ICE2014`, `TBI2014`, `BED2014`; Hirt & Rexer, 2015) are aggregated with $\cos\phi$ area weights onto nearest Voronoi cells. Cells with ocean fraction $f_{\mathrm{ocn}} \ge 0.5$ are classified as ocean (`SURF_OCEAN=0`), remaining cells with ice-sheet fraction $f_{\mathrm{ice}} \ge f_{\mathrm{land}}$ as ice sheets (`SURF_ICESHEET=2`), and the remainder as land (`SURF_LAND=1`). From high-resolution topographic gradient second moments $(K, M, L)$, the Lott & Miller (1997, App. A) subgrid standard deviation $\mu$, principal ridge orientation $\psi$, anisotropy ratio $\gamma$ ($\gamma=1$ isotropic, $\gamma\to 0$ linear ridge), and RMS slope $\sigma$ are computed:

$$
\begin{aligned}
K &= \frac{1}{2}\big(\overline{h_x^2} + \overline{h_y^2}\big), \qquad M = \frac{1}{2}\big(\overline{h_x^2} - \overline{h_y^2}\big), \qquad L = \overline{h_x h_y}, \\
\mu &= \sqrt{\max(\overline{h^2} - \bar{h}^2, 0)}, \qquad \psi = \frac{1}{2}\operatorname{atan2}(L, M), \\
\gamma &= \sqrt{\frac{\max(K - \sqrt{M^2+L^2},\ 0)}{\max(K + \sqrt{M^2+L^2},\ 10^{-12})}}, \qquad \sigma = \sqrt{\overline{h_x^2} + \overline{h_y^2}}
\end{aligned} \tag{2.7}
$$

Resolved land topography $z_{\mathrm{topo}}$ undergoes two passes of weighted Laplacian smoothing to eliminate grid-scale pressure-gradient noise:

$$
z_i^{(m+1)} = 0.75\,z_i^{(m)} + \frac{0.25}{\deg(i)}\sum_{j\in N(i)} z_j^{(m)} \qquad (m = 0, 1) \tag{2.8}
$$

where $N(i)$ is the neighbor set of cell $i$ and $\deg(i) \in \{5, 6\}$ is its degree.

### 2.4 Charney–Phillips Vertical Grid, Hybrid Stretching, Klemp STF Coordinate & Sponge (`src/aoba/vertical_grid.py`)

Both host and CRM share a Charney–Phillips staggered vertical grid placing vertical velocity $w$ and interface heights $z_w$ on $N_z + 1$ half-levels, and density, pressure, horizontal winds, moist potential temperature, and tracers at $N_z$ full-level centers $z_{c,k} = \frac{1}{2}(z_{w,k}+z_{w,k+1})$. Reference computational layer thicknesses $\Delta\zeta_k$ (initialized in `float64`; default $N_z = 44$, $z_{\mathrm{top}} = 48\ \mathrm{km}$, $\Delta z_{\min} = 60\ \mathrm{m}$) follow a hybrid power-law stretching (gentle $s^{1.35}$ in the boundary layer and steep $s^{4.2}$ aloft):

$$
\begin{aligned}
s_k &= \frac{k}{\max(N_z - 1, 1)}, \qquad S_k = 0.38\,s_k^{1.35} + 0.62\,s_k^{4.2}, \\
\Delta\zeta_k &= \Delta z_{\min} + \frac{z_{\mathrm{top}} - N_z \Delta z_{\min}}{\sum_{j=0}^{N_z-1} S_j}\,S_k
\end{aligned} \tag{2.9}
$$

To minimize horizontal pressure-gradient errors over steep orography while keeping high-top stratospheric coordinate surfaces above $24\ \mathrm{km}$ flat, the Smoothed Terrain-Following (STF) coordinate applies the Klemp (2011, Eqs. 4 & 7) cosine-power decay $A(\zeta_w)$ over $z_{\mathrm{atten}} = \min(z_{\mathrm{top}}, 24\ \mathrm{km})$ (default $p =$ `terrain_scale_exp` $= 1.5$) with minimum-thickness compression protection ($\gamma_{\min} = 0.20$):

$$
\begin{aligned}
z_{\mathrm{atten}} &= \min(z_{\mathrm{top}},\ 24000\ \mathrm{m}), \qquad A(\zeta_w) = \left[\cos\!\left(\frac{\pi}{2}\,\min\!\left(\frac{\zeta_w}{z_{\mathrm{atten}}},\ 1\right)\right)\right]^{2p}, \\
z_{w,k} &= \max\!\Big(\zeta_{w,k} + A(\zeta_{w,k})h_s,\ z_{w,k-1} + \gamma_{\min}\Delta\zeta_{c,k-1}\Big)
\end{aligned} \tag{2.10}
$$

To damp upward-propagating internal gravity waves below the rigid lid, a $\sin^2$ Rayleigh sponge layer ($\tau_{\mathrm{sp,min}} = 300\ \mathrm{s}$) acts above $z_{\mathrm{sp}}$ (default `z_sponge_base = 36 km`):

$$
\gamma_r(\zeta) = \frac{1}{\tau_{\mathrm{sp,min}}}\sin^2\!\left(\frac{\pi}{2}\,\operatorname{clamp}\!\left(\frac{\zeta - z_{\mathrm{sp}}}{\max(z_{\mathrm{top}} - z_{\mathrm{sp}}, 1)},\ 0,\ 1\right)\right) \tag{2.11}
$$

---

## 3. Moist Compressible Thermodynamics & Mixed-Phase Saturation Adjustment

### 3.1 Compressible Equation of State and Condensate Loading (`src/aoba/thermodynamics.py`)

The compressible nonhydrostatic dynamical core prognoses dry-air density $\rho_d$ and moist potential temperature $\theta_m$ (Skamarock et al., 2012, Eqs. 5–6). Exner pressure $\Pi$, total pressure $p$, and absolute temperature $T$ are diagnosed as:

$$
\theta_m = \theta\left(1 + \frac{R_v}{R_d} q_v\right), \qquad \xi = \max\!\left(\frac{R_d\rho_d\theta_m}{p_0},\ 10^{-8}\right), \qquad \Pi = \xi^{R_d / c_{vd}} \tag{3.1}
$$

$$
p = p_0\,\xi^{\gamma_d} = p_0\,\xi\,\Pi, \qquad T = \frac{\Pi\,\theta_m}{1 + (R_v / R_d)\max(q_v, 0)}, \qquad \gamma_d = \frac{c_{pd}}{c_{vd}} \approx 1.400 \tag{3.2}
$$

Total moist air density $\rho_{\mathrm{tot}}$ entering buoyancy and pressure-gradient forces includes water vapor and all condensate loading (cloud liquid $q_c$, cloud ice $q_i$, and precipitation $q_p$):

$$
\rho_{\mathrm{tot}} = \rho_d\left(1 + q_v + q_c + q_i + q_p\right) \tag{3.3}
$$

### 3.2 Saturation Vapor Pressure and Mixed-Phase Partitioning

Saturation vapor pressures over liquid water $e_{sw}(T)$ (Bolton, 1980, Eq. 10) and over ice $e_{si}(T)$ (Murray, 1967) are evaluated as:

$$
e_{sw}(T) = 611.2\exp\!\left(\frac{17.67(T - 273.15)}{T - 29.65}\right), \qquad e_{si}(T) = 611.2\exp\!\left(\frac{21.8745584(T - 273.15)}{T - 7.66}\right) \tag{3.4}
$$

Across the mixed-phase transition zone ($T_{00n} = 253.15\ \mathrm{K}$ to $T_{0n} = 273.16\ \mathrm{K}$), a piecewise-linear liquid fraction $\omega_n(T)$ defines the mixed-phase saturation mixing ratio $q_{\mathrm{sat}}(T, p)$ and exact derivative $\partial q_{\mathrm{sat}}/\partial T$ (Khairoutdinov & Randall, 2003, Eqs. A8–A14):

$$
\omega_n(T) = \operatorname{clamp}\!\left(\frac{T - T_{00n}}{T_{0n} - T_{00n}},\ 0,\ 1\right), \qquad q_{sx}(T, p) = \frac{\varepsilon_w e_{sx}(T)}{\max(p - e_{sx}(T),\ 0.05 p)}, \qquad q_{\mathrm{sat}} = \omega_n q_{sw} + (1 - \omega_n) q_{si} \tag{3.5}
$$

### 3.3 Enthalpy-Conserving Newton–Raphson Saturation Adjustment and RCEMIP Sounding

Conserving liquid–ice temperature $T_{li}$ and total non-precipitating moisture $q_n = q_v + q_c + q_i$, four vectorized Newton–Raphson iterations solve for equilibrium saturation temperature $T$ in supersaturated cells ($q_n > q_{\mathrm{sat}}(T_{li}, p)$), explicitly including the mixed-phase latent-heat transition slope inside the Jacobian $f'(T)$:

$$
\begin{aligned}
T_{li} &= T - \frac{L_{v0} q_c + L_{s0} q_i}{c_{pd}}, \\
L_{\mathrm{eff}}(T) &= \omega_n(T) L_{v0} + \big(1 - \omega_n(T)\big) L_{s0}, \\
f(T) &= T - T_{li} - \frac{L_{\mathrm{eff}}(T)}{c_{pd}}\max\!\big(q_n - q_{\mathrm{sat}}(T, p),\ 0\big) = 0
\end{aligned} \tag{3.6}
$$

$$
\begin{aligned}
f'(T) &= 1 + \frac{L_{\mathrm{eff}}(T)}{c_{pd}}\frac{\partial q_{\mathrm{sat}}}{\partial T} - \frac{\mathbf{1}_{(T_{00n}, T_{0n})}(L_{v0} - L_{s0})}{c_{pd}(T_{0n} - T_{00n})}\,q_{\mathrm{cond}}, \\
T^{(m+1)} &= T^{(m)} - \frac{f(T^{(m)})}{\max(f'(T^{(m)}), 1)}
\end{aligned} \tag{3.7}
$$

Upon convergence, total condensate $q_{\mathrm{cond}}$ and phase mixing ratios $(q_v, q_c, q_i)$ are determined as:

$$
\begin{aligned}
q_{\mathrm{cond}} &= \max\!\big(q_n - q_{\mathrm{sat}}(T, p),\ 0\big), \\
q_v &= q_n - q_{\mathrm{cond}}, \qquad q_c = \omega_n(T)\,q_{\mathrm{cond}}, \qquad q_i = \big(1 - \omega_n(T)\big)\,q_{\mathrm{cond}}
\end{aligned} \tag{3.8}
$$

For idealized radiative–convective equilibrium benchmarks, the Wing et al. (2018, Sec. 3.2.3) RCEMIP analytical initial sounding ($z_t = 15\ \mathrm{km}$, $\Gamma = 6.7\ \mathrm{K\ km^{-1}}$, $z_{q1} = 4\ \mathrm{km}$, $z_{q2} = 7.5\ \mathrm{km}$, $p_{\mathrm{sfc}} = 1014.8\ \mathrm{hPa}$) is provided via `rcemip_analytic_sounding`.

---

## 4. Atmospheric Radiative Transfer: Selectable Schemes (`rad_scheme`) & 3D TICA

### 4.1 Selectable Radiation Scheme Hierarchy (`rad_scheme`)

`[forcing] rad_scheme` selects among four radiative transfer schemes sharing an identical interface for surface VIS/NIR direct and diffuse shortwave fluxes, PAR, downward longwave flux, TOA energy balance (ASR/OLR), and atmospheric heating rates.

| `rad_scheme` | Spectral Resolution | Gas Absorption & Optics | Advantages | Limitations & Considerations |
| :--- | :--- | :--- | :--- | :--- |
| `"four_band"` | 4 bands (2 SW + 2 LW) | Lacis–Hansen $\mathrm{O}_3/\mathrm{H}_2\mathrm{O}$, analytic $\mathrm{CO}_2/\mathrm{H}_2\mathrm{O}$ LW, clouds, dust | Lowest compute cost; zero external files; suited for idealized aquaplanets and rapid prototyping. | Omits $\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$ and 4 non-dust natural aerosols; band-overlap accuracy degrades under extreme $\mathrm{CO}_2$. |
| `"ckd"` *(default)* | 16 bands / 50 $g$-pts | 6 SW (18 $g$) + 10 LW (32 $g$); $\mathrm{H}_2\mathrm{O}, \mathrm{CO}_2, \mathrm{CH}_4, \mathrm{N}_2\mathrm{O}, \mathrm{O}_3, \mathrm{O}_2$, clouds, dust | Zero external tables; treats five major GHGs and water-vapor continua with balanced speed and accuracy for multi-decadal integrations. | Neglects LW cloud scattering and uses additive binary-gas overlap, leaving small forcing biases at extreme concentrations relative to RRTMGP. |
| `"ckd_hd"` | 16 base / 111 $g$-pts | 39 SW + 72 LW $g$-pts, Chou LW cloud scattering, 5 natural + volcanic aerosols | Self-contained (no external files) while integrating LW cloud scattering and band-resolved natural/volcanic aerosol optics in fused GPU kernels. | Roughly $2\times$ the radiative cost of `ckd`; absorption coefficients rely on $g$-point splitting rather than line-by-line HITRAN lookup tables. |
| `"rrtmgp"` | 30 bands / 240 $g$-pts | Official HITRAN + MT\_CKD 4.1 (128 LW + 112 SW), 9 gases $(T, \ln p, \eta)$ | Highest CMIP-grade accuracy with official HITRAN spectroscopy and exact binary-species $\eta$ overlap interpolation (GPU-accelerated). | Requires initial auto-download of BSD-3 coefficient files; highest compute cost among the four schemes; clamps outside table $T\in[160, 355]\ \mathrm{K}$. |

![Figure 4.1: Atmospheric Radiative Transfer Physics and Selectable 4-Scheme Hierarchy](assets/fig_03_radiation_schemes.svg)

### 4.2 Shared Two-Stream Solvers (Toon et al., 1989) (`src/aoba/radiation.py`)

All four schemes solve shortwave transfer with the Toon et al. (1989) $\delta$-Quadrature $2N\times 2N$ tridiagonal two-stream solver (`toon_two_stream_sw`) and longwave transfer with the uncoupled pure-absorption ($\omega_0 = 0$) linear-in-$\tau$ two-stream sweep (`toon_two_stream_lw`), using top-down indexing ($n=0$ at TOA to $n=N-1$ at the surface).

For shortwave transfer, Joseph, Wiscombe & Weinman (1976) $\delta$-scaling ($f = g_0^2$) and Quadrature coefficients ($\mu_1 = 1/\sqrt{3}$; Toon et al., 1989, Table 1) are applied to layer optical depth $\tau$, single-scattering albedo $\omega_0$, and asymmetry factor $g_0$:

$$
\begin{aligned}
\tau' &= \max\!\big(\tau(1 - \omega_0 g_0^2),\ 10^{-7}\big), \\
\omega' &= \operatorname{clamp}\!\left(\frac{\omega_0(1 - g_0^2)}{1 - \omega_0 g_0^2},\ 0,\ 0.999999\right), \\
g' &= \frac{g_0 - g_0^2}{1 - g_0^2}
\end{aligned} \tag{4.1}
$$

$$
\begin{aligned}
\gamma_1 &= \frac{\sqrt{3}}{2}\big[2 - \omega'(1 + g')\big], \qquad \gamma_2 = \frac{\sqrt{3}}{2}\omega'(1 - g'), \\
\gamma_3 &= \frac{1}{2}\big(1 - \sqrt{3}\,g'\mu_0\big), \qquad \gamma_4 = 1 - \gamma_3
\end{aligned} \tag{4.2}
$$

The two-stream eigenvalues $\lambda_n$, semi-infinite reflectance $\Gamma_n$, cumulative optical depth $\tau_{c,n}$, homogeneous basis factors $(e_{1,n}, e_{2,n}, e_{3,n}, e_{4,n})$, and direct-beam particular solutions $C_n^\pm(\tau)$ (with resonance denominator $\lambda_n^2 - \mu_0^{-2}$ regularized by $\pm 10^{-5}$) are formed as:

$$
\begin{aligned}
\lambda_n &= \sqrt{\max(\gamma_{1,n}^2 - \gamma_{2,n}^2,\ 10^{-12})}, \qquad \Gamma_n = \frac{\gamma_{2,n}}{\gamma_{1,n} + \lambda_n}, \qquad \tau_{c,n} = \sum_{j=0}^{n-1}\tau'_j, \\
e_{1,n} &= 1 + \Gamma_n e^{-\lambda_n\tau'_n}, \qquad e_{2,n} = 1 - \Gamma_n e^{-\lambda_n\tau'_n}, \\
e_{3,n} &= \Gamma_n + e^{-\lambda_n\tau'_n}, \qquad e_{4,n} = \Gamma_n - e^{-\lambda_n\tau'_n}
\end{aligned} \tag{4.3}
$$

$$
C_n^\pm(\tau) = \frac{\omega'_n F_s\big[(\gamma_{1,n} \mp \mu_0^{-1})\gamma_{3/4,n} + \gamma_{4/3,n}\gamma_{2,n}\big]}{\lambda_n^2 - \mu_0^{-2}}\,e^{-(\tau_{c,n}+\tau)/\mu_0} \tag{4.4}
$$

Combining zero downward diffuse flux at TOA ($F_0^-(0) = 0$), interlayer continuity (Toon et al., 1989, Eqs. 39–43), and surface direct/diffuse reflection $(r_{\mathrm{dir}}, r_{\mathrm{dif}})$ yields a $2N\times 2N$ tridiagonal system solved by `solve_tridiagonal_batched`:

$$
F^\uparrow_N = r_{\mathrm{dir}}\,\mu_0 F_s\,e^{-\tau_{c,N}/\mu_0} + r_{\mathrm{dif}}\,F^\downarrow_{\mathrm{dif},N} \tag{4.5}
$$

For longwave transfer with diffusivity factor $D$ ($D=2$ in `four_band`, $D=1.66$ in `ckd`/`ckd_hd`, band-optimal secant angle in `rrtmgp`), the uncoupled linear-in-$\tau$ Planck sweep uses cancellation-free Taylor switching for small $\Delta\tau_{\mathrm{eff}}$:

$$
\begin{aligned}
\Delta\tau_{\mathrm{eff}} &= \operatorname{clamp}(D\tau,\ 10^{-6},\ 50), \qquad E = e^{-\Delta\tau_{\mathrm{eff}}}, \\
\alpha &= \begin{cases} -\operatorname{expm1}(-\Delta\tau_{\mathrm{eff}})/\Delta\tau_{\mathrm{eff}}, & \Delta\tau_{\mathrm{eff}} > 10^{-3} \\ 1 - \frac{1}{2}\Delta\tau_{\mathrm{eff}} + \frac{1}{6}\Delta\tau_{\mathrm{eff}}^2, & \Delta\tau_{\mathrm{eff}} \le 10^{-3} \end{cases}, \\
w_1 &= 1 - \alpha, \qquad w_2 = \max(\alpha - E,\ 0)
\end{aligned} \tag{4.6}
$$

$$
\begin{aligned}
F^\downarrow_{n+1} &= F^\downarrow_n E_n + w_{2,n} B_{\mathrm{top},n} + w_{1,n} B_{\mathrm{bot},n}, \\
F^\uparrow_N &= \epsilon_s B_s + (1 - \epsilon_s) F^\downarrow_N, \\
F^\uparrow_n &= F^\uparrow_{n+1} E_n + w_{1,n} B_{\mathrm{top},n} + w_{2,n} B_{\mathrm{bot},n}
\end{aligned} \tag{4.7}
$$

`ckd_hd` and `rrtmgp` employ the algebraically equivalent Fu & Liou (1992) linear-in-$\tau$ weight $w_{\mathrm{lin}}$:

$$
w_{\mathrm{lin}} = \begin{cases} \Delta\tau_{\mathrm{eff}}^{-1} - \dfrac{E}{1 - E}, & \Delta\tau_{\mathrm{eff}} > 10^{-3} \\ \dfrac{1}{2} - \dfrac{\Delta\tau_{\mathrm{eff}}}{12}, & \Delta\tau_{\mathrm{eff}} \le 10^{-3} \end{cases} \tag{4.8}
$$

Layer radiative heating rates $Q_{\mathrm{rad},k}$ follow from vertical convergence of net upward flux $F^{\mathrm{net}} = F^\uparrow - F^\downarrow$:

$$
\begin{aligned}
Q_{\mathrm{rad},k} &= -\frac{F^{\mathrm{net}}_{k+1} - F^{\mathrm{net}}_k}{C_{\mathrm{layer},k}}, \\
C_{\mathrm{layer},k} &= \begin{cases}(c_{pd} + c_{pv} q_{v,k})\rho_{d,k}\Delta z_k & (\text{four\_band, ckd}) \\ c_{pd}\rho_{d,k}\Delta z_k & (\text{ckd\_hd, rrtmgp})\end{cases}
\end{aligned} \tag{4.9}
$$

### 4.3 Option 1: `rad_scheme = "four_band"` (4-Band Analytical Radiation, `src/aoba/radiation.py`)

Partitions shortwave radiation into SW1 ($0.25\text{–}0.69\ \mu\mathrm{m}$, solar weight $f_{\mathrm{SW1}} = 0.459760$) and SW2 ($0.69\text{–}4.00\ \mu\mathrm{m}$, $f_{\mathrm{SW2}} = 0.540240$), and longwave radiation into non-window LW1 ($1-\beta = 0.680$) and the $8\text{–}12.5\ \mu\mathrm{m}$ atmospheric window LW2 ($\beta = 0.320$).

SW1 evaluates the spherical-shell air-mass factor $M(\mu_0)$ (Rodgers, 1967), Lacis & Hansen (1974, Eqs. 8–10) ozone UV+VIS absorptance $A_{\mathrm{oz}}(x)$, and Rayleigh scattering $\Delta\tau_{\mathrm{ray}}$; SW2 converts Lacis & Hansen (1974, Eq. 21) water-vapor absorptance $A_{\mathrm{wv}}(y)$ over scaled vapor path $y$ into layer optical depths:

$$
\begin{aligned}
M(\mu_0) &= \frac{35}{\sqrt{1224\mu_0^2 + 1}}, \qquad \Delta\tau_{\mathrm{ray}} = 0.095\,\frac{\Delta p}{p_0}, \\
A_{\mathrm{wv}}(y) &= \frac{2.9\,y}{(1 + 141.5\,y)^{0.635} + 5.925\,y}
\end{aligned} \tag{4.10}
$$

Cloud shortwave optics combine Slingo (1989) liquid and Ebert & Curry (1992) ice parameterizations with the Cahalan et al. (1994) subgrid cloud inhomogeneity factor $\chi_{\mathrm{SW}}$ (default `cloud_sw_inhomog = 0.62`; $W$ is $\mathrm{LWP}$ or $\mathrm{IWP}\ [\mathrm{g\ m^{-2}}]$ and $r_e$ is effective radius $[\mu\mathrm{m}]$):

$$
\begin{aligned}
\tau_{\mathrm{cld}} &= \chi_{\mathrm{SW}}\,W\left(a + \frac{b}{r_e}\right), \\
\omega_0 &= \operatorname{clamp}\big(1 - (c + d\,r_e),\ 0,\ 0.999999\big), \\
g &= e + f\,r_e
\end{aligned} \tag{4.11}
$$

Cloud LW absorption uses Stephens (1978) ($\kappa_{\mathrm{liq}} = 0.090\ \mathrm{m^2\ g^{-1}}$) and Ebert & Curry (1992) scaled by the diffusivity ratio $1.66/2.0 = 0.83$, and mineral dust uses Balkanski et al. (2007) ($k_{\mathrm{ext}} = 600/400$, $k_{\mathrm{abs}} = 100/150\ \mathrm{m^2\ kg^{-1}}$).

LW1 and LW2 gas optical depths include logarithmic $\mathrm{CO}_2$ ratio $r_{\mathrm{CO_2}}$, stratospheric $15\ \mu\mathrm{m}$ cooling $c_{\mathrm{strat}}$, Roberts et al. (1976) water-vapor self-continuum absorption, and optional $9.6\ \mu\mathrm{m}$ ozone ($4.5\,m_{\mathrm{O_3}}$):

$$
\begin{aligned}
r_{\mathrm{CO_2}} &= \frac{\max(C_{\mathrm{CO_2}}, 1)}{284.3}, \\
\Delta\tau_{\mathrm{CO_2,LW1}} &= \max(0.02,\ 0.85 + 0.22\ln r_{\mathrm{CO_2}})\,\frac{\Delta(p^2)}{p_0^2} + (0.020 + c_{\mathrm{strat}})\,\frac{\Delta p}{p_0}, \\
\Delta\tau_{\mathrm{wv,LW1}} &= 0.30\left(\frac{p}{p_0}\right)^{0.55} q_v\,\Delta m_d
\end{aligned} \tag{4.12}
$$

$$
\Delta\tau_{\mathrm{wv,LW2}} = \left(0.45\,\frac{e_v}{p_0}\exp\!\left[1800\left(\frac{1}{T} - \frac{1}{296}\right)\right] + 0.0025\,\frac{p}{p_0}\right)q_v\,\Delta m_d \tag{4.13}
$$

Surface photosynthetically active radiation $\mathrm{PAR}$ and photon flux density $\mathrm{PPFD}$ passed to terrestrial photosynthesis are:

$$
\mathrm{PAR} = 0.88\,F^\downarrow_{\mathrm{SW1,sfc}}, \qquad \mathrm{PPFD} = 4.57\,\mathrm{PAR} \quad [\mu\mathrm{mol\ m^{-2}\ s^{-1}}] \tag{4.14}
$$

### 4.4 Option 2: `rad_scheme = "ckd"` (16-Band / 50 $g$-Point Correlated-$k$, `src/aoba/radiation_ckd.py`)

Resolves 6 SW bands ($0.20\text{–}4.00\ \mu\mathrm{m}$, 18 $g$-points; $\mathrm{PAR} = 0.94\,F^\downarrow_{\mathrm{SW1,sfc}}$) and 10 LW bands (wavenumber edges $0, 350, 500, 630, 700, 820, 980, 1080, 1180, 1400, \infty\ \mathrm{cm^{-1}}$, 32 $g$-points) for $\mathrm{H}_2\mathrm{O}, \mathrm{CO}_2, \mathrm{CH}_4, \mathrm{N}_2\mathrm{O}, \mathrm{O}_3, \mathrm{O}_2$. Band Planck integrals use the 6-term Widger & Woodall (1976) series $F_{\mathrm{tail}}(\nu, T)$ so band emissions sum identically to $\sigma_{\mathrm{SB}} T^4$:

$$
\begin{aligned}
F_{\mathrm{tail}}(\nu, T) &= \frac{15}{\pi^4}\sum_{n=1}^{6}\frac{e^{-nx}}{n^4}\Big[(nx)^3 + 3(nx)^2 + 6(nx) + 6\Big], \\
x &= \frac{c_2 \nu}{\max(T, 100)}, \qquad c_2 = 1.438776877\ \mathrm{cm\ K}, \\
B_b(T) &= \sigma_{\mathrm{SB}} T^4\big[F_{\mathrm{tail}}(\nu_{b,\mathrm{low}}, T) - F_{\mathrm{tail}}(\nu_{b,\mathrm{high}}, T)\big]
\end{aligned} \tag{4.15}
$$

Stratospheric photolytic roll-off $s_{\mathrm{strat}}(p)$ scales $\mathrm{CH}_4$ and $\mathrm{N}_2\mathrm{O}$. Each LW $g$-point gas optical depth includes Voigt pressure floors, $\mathrm{CO}_2$ hot-band Boltzmann scaling, band-specific water-vapor self and foreign continua $\tau_{b,\mathrm{cont}}$, clouds (Hu & Stamnes, 1993; Ebert & Curry, 1992), and dust, summed over quadrature weights $w_{b,g}$:

$$
\begin{aligned}
s_{\mathrm{strat}}(p) &= \min\!\left(\frac{p}{8000},\ 1\right)^{0.38}, \\
\tau_{g,\mathrm{line}} &= k_{\mathrm{H_2O}}\!\left(\frac{p+80}{p_0}\right)^{\gamma_h}\!u_{\mathrm{H_2O}} + k_{\mathrm{CO_2}}\!\left(\frac{p+p_{\mathrm{vc}}}{p_0}\right)^{\gamma_c}\!e^{b_{T,c}(260^{-1}-T^{-1})} u_{\mathrm{CO_2}} \\
&\quad + k_{\mathrm{O_3}}\!\left(\frac{p+120}{p_0}\right)^{\gamma_o}\!u_{\mathrm{O_3}} + k_{\mathrm{CH_4}}\!\left(\frac{p+100}{p_0}\right)^{\gamma_m}\!s_{\mathrm{strat}} u_{\mathrm{CH_4}} + k_{\mathrm{N_2O}}\!\left(\frac{p+100}{p_0}\right)^{\gamma_n}\!s_{\mathrm{strat}} u_{\mathrm{N_2O}}
\end{aligned} \tag{4.16}
$$

$$
\begin{aligned}
\tau_{b,\mathrm{cont}} &= \Big[C_{\mathrm{self},b}\,e_{\mathrm{atm}}\,e^{1800(T^{-1}-296^{-1})} + C_{\mathrm{for},b}\,p_{\mathrm{atm}}\Big]\,u_{\mathrm{H_2O}}, \\
F^\pm(z) &= \sum_{b}\sum_{g=1}^{N_{g,b}} w_{b,g}\,F_{b,g}^\pm(z)
\end{aligned} \tag{4.17}
$$

### 4.5 Option 3: `rad_scheme = "ckd_hd"` (111 $g$-Point CKD-HD & Aerosol Direct Delta, `src/aoba/radiation_hd.py`)

Subdivides each base 16-band $g$-point via mean-preserving splits $(0.52w, 0.72k)$ and $(0.48w, 1.30k)$ with additional refinement in strongly absorbing LW $\mathrm{CO}_2$/$\mathrm{O}_3$ and SW UV bands to yield 111 $g$-points (39 SW + 72 LW; MSTRN11-class, Sekiguchi & Nakajima, 2008). Incorporates Chou et al. (1999) LW cloud backscattering adjustments ($0.91\text{–}0.97$) and band-scaled optical properties (Ångström $\alpha \approx 1.1$) for all five natural aerosols plus volcanic $\mathrm{H}_2\mathrm{SO}_4$. Also provides `apply_aerosol_rad_delta` for fast broadband SW/LW aerosol direct radiative perturbation using two SW two-stream sweeps and LW window emissivity $\varepsilon_a$:

$$
\varepsilon_a = 1 - \exp(-1.66\,\tau_{a,\mathrm{LW}}) \tag{4.18}
$$

### 4.6 Option 4: `rad_scheme = "rrtmgp"` (Official HITRAN / MT\_CKD 4.1 240 $g$-Point, `src/aoba/radiation_rrtmgp.py`)

Auto-downloads official BSD-3 RTE+RRTMGP gas-optics coefficient files (`rrtmgp-gas-lw-g128.nc`: 16 bands / 128 LW $g$-points; `rrtmgp-gas-sw-g112.nc`: 14 bands / 112 SW $g$-points; Pincus et al., 2019) covering 9 gases ($\mathrm{H}_2\mathrm{O}, \mathrm{CO}_2, \mathrm{O}_3, \mathrm{N}_2\mathrm{O}, \mathrm{CO}, \mathrm{CH}_4, \mathrm{O}_2, \mathrm{N}_2$ plus continua). Dry-air molecular column density $N_{\mathrm{dry}}$ is scaled by $10^{-20}$ to prevent `float32` underflow, with trilinear $(T, \ln p, \eta)$ interpolation across the binary major-species mixing ratio $\eta$ above and below tropopause pressure $p_{\mathrm{ref,trop}}$:

$$
\begin{aligned}
f_{\mathrm{ref}}(j_T) &= \frac{\mathrm{VMR}_{\mathrm{ref}}(j_T, s_1)}{\mathrm{VMR}_{\mathrm{ref}}(j_T, s_2)}, \\
\eta &= \frac{v_{s_1}}{v_{s_1} + f_{\mathrm{ref}} v_{s_2}}, \\
\tau_{\mathrm{maj}} &= N_{\mathrm{dry}}\big(v_{s_1} + f_{\mathrm{ref}} v_{s_2}\big)\,k_{\mathrm{maj}}(T, \ln p, \eta)
\end{aligned} \tag{4.19}
$$

Minor gases and MT\_CKD 4.1 continua apply bilinear $(T, \eta)$ interpolation with density/dilution scaling; LW Planck emission combines `totplnk` with trilinear `plank_fraction` and band-optimal secant diffusivity angles $\operatorname{clamp}(\sec\theta_{\mathrm{opt}}, 1.45, 2.05)$. Cloud, dust, and aerosol optics are evaluated as smooth analytic functions of band center wavenumber $\tilde{\nu}_b$.

### 4.7 3D Tilted Independent Column Approximation (`enable_rad_3d_tica`, `src/aoba/radiation_hd.py`)

When `enable_rad_3d_tica = true`, cloud liquid and ice arrays $(q_c, q_i)$ are shifted horizontally per level along the periodic CRM array according to solar zenith angle $\theta_z$ ($\mu_0 > 0.08$) and hour-angle sign $s = \operatorname{sgn}(H)$ (Várnai & Davies, 1999), approximating 3D cloud-side illumination and oblique shading while strictly conserving domain cloud mass:

$$
\begin{aligned}
k_{\mathrm{shift}}(z) &= \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{z_{\mathrm{rel}}\min(\tan\theta_z, 2.5)}{\max(\Delta x_{\mathrm{CRM}}, 100)}\right),\ 0,\ 3\right), \\
i_x^{\mathrm{src}} &= (i_x - s\,k_{\mathrm{shift}}) \bmod N_x
\end{aligned} \tag{4.20}
$$

---

## 5. Cloud Microphysics: Selectable Schemes (`microphysics_scheme`) & Non-Inductive Lightning / $\mathrm{LNO}_x$

### 5.1 Selectable Microphysics Scheme Hierarchy (`microphysics_scheme`)

`[forcing] microphysics_scheme` selects among three bulk cloud microphysics schemes.

| `microphysics_scheme` | Prognostic Hydrometeors | Advantages | Limitations & Considerations |
| :--- | :--- | :--- | :--- |
| `"one_moment"` | $q_n, q_p$ (fixed effective radii) | Minimum prognostic variables, memory footprint, and compute cost; ideal for RCE and dynamical-core benchmarks. | Fixed droplet/crystal sizes preclude aerosol indirect effects (Twomey/lifetime); rain, snow, and graupel rely on temperature partitioning. |
| `"two_moment"` *(default)* | $q_n, q_p, n_c, n_i$ | Couples with sea-salt/dust CCN and Cooper–DeMott INP for aerosol indirect effects, prognostic $r_e$, and size-dependent ice fall speeds at low cost. | Precipitating species (rain, snow, graupel) share a single bulk $q_p$ partitioned by temperature rather than tracking independent riming histories. |
| `"six_class_2m"` | $(q_c,n_c),(q_i,n_i),(q_r,n_r),(q_s,n_s),(q_g,n_g)$ | Explicitly prognoses rain, snow, and graupel mass/number, riming, supercooled freezing, melting, and species-specific sedimentation for highest deep-convection and lightning fidelity. | Expands prognostic moisture to 11 variables, increasing CRM memory usage and advection/microphysics cost. |

![Figure 5.1: Selectable Cloud Microphysics Schemes, Hydrometeor Pathways & Non-Inductive Charging](assets/fig_04_microphysics_convection.svg)

### 5.2 Shared Hydrometeor Constants and Temperature Partitioning (`src/aoba/microphysics.py`)

Following Khairoutdinov & Randall (2003, App. A–B), rain ($r$), snow ($s$), and graupel ($g$) assume exponential Marshall–Palmer size distributions $n_m(D)$ and density-scaled power-law terminal fall speeds $v_m(D)$ (reference density $\rho_0 = 1.29\ \mathrm{kg\ m^{-3}}$):

$$
\begin{aligned}
n_m(D) &= N_{0m}\,\exp(-\lambda_m D), \\
v_m(D) &= a_m D^{b_m}\left(\frac{\rho_0}{\rho_d}\right)^{1/2}
\end{aligned} \tag{5.1}
$$

where $D$ is particle diameter $[\mathrm{m}]$, $N_{0m}$ is the intercept parameter $[\mathrm{m^{-4}}]$, $\lambda_m = (\pi\rho_m N_{0m}/(\rho_d q_m))^{1/4}$ is the slope parameter, and $(a_m, b_m)$ are fall-speed coefficients.

| Species $m$ | $\rho_m\ [\mathrm{kg\ m^{-3}}]$ | $N_{0m}\ [\mathrm{m^{-4}}]$ | $a_m$ | $b_m$ | $a_{fm}$ | $b_{fm}$ | $C_m$ | $E_{mc}$ | $E_{mi}$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rain ($r$) | $1000$ | $8.0\times 10^6$ | $842.0$ | $0.80$ | $0.78$ | $0.31$ | $1.0$ | $1.0$ | $1.0$ |
| Snow ($s$) | $100$ | $3.0\times 10^6$ | $4.84$ | $0.25$ | $0.65$ | $0.44$ | $2/\pi$ | $1.0$ | $0.1$ |
| Graupel ($g$) | $400$ | $4.0\times 10^6$ | $94.5$ | $0.50$ | $0.78$ | $0.31$ | $1.0$ | $1.0$ | $0.1$ |

In `one_moment` and `two_moment`, total precipitation $q_p$ is partitioned diagnostically via temperature ramps $\omega_p(T)$ ($T_{00p}=268.16\ \mathrm{K}$ to $T_{0p}=283.16\ \mathrm{K}$) and $\omega_g(T)$ ($T_{00g}=223.16\ \mathrm{K}$ to $T_{0g}=283.16\ \mathrm{K}$):

$$
\begin{aligned}
\omega_p(T) &= \operatorname{clamp}\!\left(\frac{T - T_{00p}}{T_{0p} - T_{00p}},\ 0,\ 1\right), \\
\omega_g(T) &= \operatorname{clamp}\!\left(\frac{T - T_{00g}}{T_{0g} - T_{00g}},\ 0,\ 1\right)
\end{aligned} \tag{5.2}
$$

$$
q_r = \omega_p q_p, \qquad q_s = (1 - \omega_p)(1 - \omega_g)q_p, \qquad q_g = (1 - \omega_p)\omega_g q_p \tag{5.3}
$$

### 5.3 Option 1: `microphysics_scheme = "one_moment"` (SAM 1-Moment)

1. Performs an initial $T_{li}$-conserving saturation adjustment for $(T, q_v, q_c, q_i)$.
2. Warm-rain autoconversion uses Berry (1968) / Tomita (2008; SCALE-RM Eqs. 8.221–8.222) when $N_d > 0$ (default $60\ \mathrm{cm^{-3}}$; falling back to the KR03 threshold $\alpha\max(q_c - q_{co}, 0)$ if $N_d = 0$), and ice aggregation follows KR03 Eq. A31 ($\beta = 10^{-3}\ \mathrm{s^{-1}}$, $q_{io} = 10^{-4}\ \mathrm{kg\ kg^{-1}}$):

$$
\begin{aligned}
D_d &= \max\!\left(0.146 - 5.964\times 10^{-2}\ln\frac{\max(N_d, 1)}{2000},\ 0.05\right), \\
\left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{auto}} &= \frac{16.7\,D_d(\rho_d q_c)^2 q_c}{5.0\,D_d\rho_d q_c + 3.6\times 10^{-5} N_d}
\end{aligned} \tag{5.4}
$$

$$
\left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{aggr}} = \beta\,e^{0.025(T - T_0)}\max(q_i - q_{io},\ 0) \tag{5.5}
$$

3. Continuous Marshall–Palmer accretion (KR03 Eqs. A27–A29; $l\in\{c, i\}$) and ventilated evaporation/sublimation in subsaturated air $S_{\mathrm{sub}}$ (KR03 Eqs. A21–A26) are computed as:

$$
\begin{aligned}
A_{\mathrm{pref},ml} &= \frac{\pi}{4}\,a_m N_{0m} E_{ml}\,\Gamma(3 + b_m)\,\rho_0^{1/2}\big(\pi\rho_m N_{0m}\big)^{-(3+b_m)/4}, \\
\left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{accr}} &= \rho_d^{-1/2}\sum_{m\in\{r,s,g\}}\Big[A_{\mathrm{pref},mc}\,q_c + e^{0.025(T-T_0)} A_{\mathrm{pref},mi}\,q_i\Big](\rho_d q_m)^{(3+b_m)/4}
\end{aligned} \tag{5.6}
$$

$$
\begin{aligned}
S_{\mathrm{sub}} &= \max\!\left(1 - \frac{q_v}{q_{\mathrm{sat}}},\ 0\right), \\
E_m &= \frac{2\pi C_m N_{0m}}{\rho_d (A+B)_{l/i}}\Big[a_{em}(\rho_d q_m)^{1/2} + b_{em}\rho_d^{-1/4}(\rho_d q_m)^{(5+b_m)/8}\Big]S_{\mathrm{sub}}
\end{aligned} \tag{5.7}
$$

4. After updating $T_{li}$ conservatively for phase-change latent heat, precipitation $q_p$ (fall flux $P_0$, effective fall rate $r_{p,k}$) and cloud ice $q_i$ ($V_{TI} = 0.75\ \mathrm{m\ s^{-1}}$) sediment via an unconditionally stable implicit donor-cell upper-bidiagonal Thomas sweep:

$$
\begin{aligned}
P_0 &= \rho_d^{-1/2}\sum_{m\in\{r,s,g\}} F_{\mathrm{pref},m}(\rho_d q_m)^{1 + b_m/4}, \\
r_{p,k} &= \min\!\left(\frac{P_{0,k}}{q_{p,k}},\ 15\,\rho_{d,k}\right), \\
\left(1 + \frac{\Delta t\,r_k}{m_k}\right) q_k^{n+1} - \frac{\Delta t\,r_{k+1}}{m_k} q_{k+1}^{n+1} &= q_k^n, \qquad m_k = \rho_{d,k}\Delta z_k
\end{aligned} \tag{5.8}
$$

A final saturation adjustment updates $(T, q_v, q_c, q_i)$ and $\theta_m$.

### 5.4 Option 2: `microphysics_scheme = "two_moment"` (Aerosol-Aware 2-Moment KK00)

Adds prognostic cloud-droplet and cloud-ice number mixing ratios $(n_c, n_i)\ [\mathrm{kg^{-1}}]$. Natural CCN activation $N_{\mathrm{act}}$ combines wind-driven marine sea salt $N_{\mathrm{CCN,ocn}}$ (continental background $95\ \mathrm{cm^{-3}}$), mineral dust mass concentration $D_{\mu\mathrm{g}} = 10^9\rho_d q_{\mathrm{dust}}\ [\mu\mathrm{g\ m^{-3}}]$, and updraft activation factor $f_{\mathrm{act}}$:

$$
\begin{aligned}
N_{\mathrm{CCN,ocn}} &= \operatorname{clamp}\!\big(30\,e^{0.055 U_{10}},\ 30,\ 140\big)\ [\mathrm{cm^{-3}}], \\
f_{\mathrm{act}} &= \left(\frac{w_+}{w_+ + 0.35}\right)^{0.45}, \\
N_{\mathrm{act}} &= 10^6\,\operatorname{clamp}\!\left(\left[N_{\mathrm{base}} + 120\left(\frac{D_{\mu\mathrm{g}}}{10}\right)^{0.5}\right]f_{\mathrm{act}},\ 15,\ 450\right)\ [\mathrm{m^{-3}}]
\end{aligned} \tag{5.9}
$$

Ice-nucleating particle concentration $N_{\mathrm{INP}}$ blends Cooper (1986) supercooling and DeMott et al. (2010) dust-enhanced heterogeneous nucleation $N_{\mathrm{het}}$ with homogeneous freezing $f_{\mathrm{hom}}$ below $238\ \mathrm{K}$:

$$
\begin{aligned}
f_{\mathrm{hom}} &= \operatorname{clamp}\!\left(\frac{238 - T}{6},\ 0,\ 1\right), \\
N_{\mathrm{het}} &= \operatorname{clamp}\!\left(5.0\,e^{0.304\,\mathrm{clamp}(273.15 - T,\,0,\,40)}\left[1 + 4.0\left(\frac{D_{\mu\mathrm{g}}}{10}\right)^{0.4}\right],\ 3\times 10^4,\ 5\times 10^5\right), \\
N_{\mathrm{INP}} &= (1 - f_{\mathrm{hom}})N_{\mathrm{het}} + 2\times 10^5\,f_{\mathrm{hom}}
\end{aligned} \tag{5.10}
$$

Warm-rain autoconversion adopts scale-aware Khairoutdinov & Kogan (2000) with resolution-dependent subgrid LWC variance enhancement $C_{\mathrm{KK00}}(\Delta x)$ bridging 50 m LES ($1350$) to cloud-resolving grids (Morrison & Gettelman, 2008; Boutle et al., 2014; $42000$ at $\Delta x = 4\ \mathrm{km}$), while prognostic effective radii $(r_{e,l}, r_{e,i})$ ($\kappa_{\mathrm{disp}} = 0.72$, $\rho_{i,\mathrm{cld}} = 500\ \mathrm{kg\ m^{-3}}$) couple into radiation (Eq. 4.11) and size-aware cloud-ice sedimentation velocity $V_{TI,\mathrm{eff}}$:

$$
\begin{aligned}
C_{\mathrm{KK00}}(\Delta x) &= \operatorname{clamp}\!\left(1350\left(\frac{\max(\Delta x,\,50)}{50}\right)^{0.7845},\ 1350,\ 1.2\times 10^5\right), \\
\left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{auto,KK00}} &= C_{\mathrm{KK00}}(\Delta x)\,q_c^{2.47}\,N_{c,\mathrm{cm^{-3}}}^{-1.79}
\end{aligned} \tag{5.11}
$$

$$
\begin{aligned}
r_{e,l} &= \operatorname{clamp}\!\left(10^6\left(\frac{3 q_c}{4\pi\rho_r\kappa_{\mathrm{disp}} n_c}\right)^{1/3},\ 4,\ 25\right)\ [\mu\mathrm{m}], \\
r_{e,i} &= \operatorname{clamp}\!\left(0.85\times 10^6\left(\frac{3 q_i}{4\pi\rho_{i,\mathrm{cld}} n_i}\right)^{1/3},\ 15,\ 120\right)\ [\mu\mathrm{m}], \\
V_{TI,\mathrm{eff}} &= \operatorname{clamp}\!\left(0.75\,\frac{r_{e,i}}{48},\ 0.10,\ 1.20\right)\ [\mathrm{m\ s^{-1}}]
\end{aligned} \tag{5.12}
$$

### 5.5 Option 3: `microphysics_scheme = "six_class_2m"` (6-Class 2-Moment, `src/aoba/microphysics_6class.py`)

Independently prognoses all 6 classes (11 variables) including rain $(q_r, n_r)$, snow $(q_s, n_s)$, and graupel $(q_g, n_g)$ inside the fused GPU kernel `microphysics_6class_col`, integrating all conversions via bounded exponential factors $1 - e^{-k\Delta t} = -\operatorname{expm1}(-k\Delta t)$ (Morrison et al., 2005; Seifert & Beheng, 2006). Warm autoconversion $k_{\mathrm{auto}}$, accretion $k_{\mathrm{accr}}$, snow/graupel riming below $273.15\ \mathrm{K}$ ($k_{\mathrm{rime}}$), heavy-riming snow-to-graupel conversion $k_{s\to g}$, and supercooled rain freezing below $268.15\ \mathrm{K}$ ($k_{\mathrm{frz}}$) are:

$$
\begin{aligned}
C_{\mathrm{KK00,6c}}(\Delta x) &= \operatorname{clamp}\!\left(1350\left(\frac{\max(\Delta x,\,50)}{50}\right)^{0.5254},\ 1350,\ 6\times 10^4\right), \\
k_{\mathrm{auto}} &= C_{\mathrm{KK00,6c}}(\Delta x)\,q_c^{1.47}\,N_c^{-1.79}, \qquad k_{\mathrm{accr}} = 67\,q_r^{1.15}\,q_c^{0.15}, \\
k_{\mathrm{rime}} &= 0.6\,q_s^{0.80} + 1.2\,q_g^{0.80}, \qquad k_{s\to g} = 250\,q_c, \\
k_{\mathrm{frz}} &= 0.015\,\exp\!\big(0.25(268.15 - T)\big)
\end{aligned} \tag{5.13}
$$

Following ice aggregation ($q_i\to q_s$ above threshold $q_{i,\mathrm{th}} = 6\times 10^{-5}\ \mathrm{kg\ kg^{-1}}$), snow/graupel melting above $273.15\ \mathrm{K}$, sub-cloud rain evaporation (relative to liquid saturation $q_{\mathrm{sat,w}}$), and snow/graupel sublimation (relative to ice saturation $q_{\mathrm{sat,i}}$), latent-heating updates reconstruct exact moist potential temperature $\theta_m^{n+1} = (T^{n+1}/\Pi)(1 + (R_v/R_d) q_v^{n+1})$. Species-specific terminal fall speeds $(v_i, v_r, v_s, v_g)$ with air-density scaling $f_\rho$ drive top-down implicit upwind sedimentation:

$$
\begin{aligned}
f_\rho &= \sqrt{\operatorname{clamp}\!\left(\frac{1.2}{\rho_d},\ 0.5,\ 6.0\right)}, \qquad v_i = \operatorname{clamp}\!\left(0.22 + 0.45\left(\frac{q_i}{10^{-4}}\right)^{0.25},\ 0.18,\ 0.80\right), \\
v_r &= 6.5\,(10^3\rho_d q_r)^{0.14} f_\rho, \qquad v_s = 1.1\,(10^3\rho_d q_s)^{0.10} f_\rho, \qquad v_g = 4.2\,(10^3\rho_d q_g)^{0.18} f_\rho
\end{aligned} \tag{5.14}
$$

$$
q_{x,k}^{n+1} = \frac{m_k q_{x,k}^n + \Delta t\,F_{x,k+1}}{m_k + \Delta t\,\rho_{d,k} v_{x,k}}, \qquad F_{x,k} = \rho_{d,k} v_{x,k} q_{x,k}^{n+1}, \qquad x\in\{r, s, g\} \tag{5.15}
$$

### 5.6 Selectable CRM Non-Inductive Lightning and $\mathrm{LNO}_x$ Schemes (`lightning_scheme`)

Both lightning schemes operate on the embedded CRM hydrometeor and vertical-velocity fields without geographic lookup tables and supply identical `LightningDiagnostics` fields (subcolumn total/CG flash rates and mass-conserving $\mathrm{LNO}_x$ deposition) to SPITFIRE wildfire, prognostic ozone chemistry, and terrestrial/marine nitrogen budgets.

| `lightning_scheme` | Prognostic Charge Variables | Physics Scope & Discharge Treatment | Relative Cost & Recommended Use |
| :--- | :--- | :--- | :--- |
| `"diagnostic"` *(default)* | None (diagnostic) | McCaul et al. (2009) mixed-phase graupel-updraft flux & IWP proxy with Price & Rind (1993) cold-cloud-depth IC/CG ratio. | Negligible overhead; recommended for multi-decadal 2D superparameterized climate runs. |
| `"explicit_bulk"` | 5-species charge densities $\rho_{q,s}\ (s\in\{c,r,i,s,g\})$ | SCALE-LT-class explicit Takahashi (1978) / Sato et al. (2019) $T$–LWC polarity-reversal riming electrification, conservative 3D transport & gravitational sedimentation, exact FFT + Thomas Poisson electric-field solve, and Dwyer (2003) / Fierro et al. (2013) dielectric-breakdown IC/CG discharges. | Moderate CRM overhead (5 extra charge arrays + 1 non-iterative FFT-Thomas solve per host step); suited for 3D CRM deep-convection and thunderstorm-electrification studies. |

#### 5.6.1 Diagnostic Non-Inductive Charging Scheme (`lightning_scheme = "diagnostic"`, `src/aoba/lightning.py`)

From CRM-resolved updrafts $w_+ = \max(\bar{w}_k, 0)$ and graupel mixing ratio $q_g$, a mixed-phase Gaussian weight $W_T(T)$ centered on the $-15^\circ\mathrm{C}$ ice–graupel charge-separation zone determines the McCaul et al. (2009) / Saunders et al. (1991) non-inductive charging metrics $(G_{\max}, G_{\mathrm{chg}})$ and total flash rate $\mathcal{F}_{\mathrm{tot}}$ (global calibration $c_{\mathrm{cal}} = 3.6$):

$$
\begin{aligned}
W_T(T) &= \exp\!\left(-\left(\frac{T - 258.15}{7.0}\right)^2\right), \\
G_{\max} &= \max_k\!\big[W_T(T_k)\,\rho_{d,k}\,w_{+,k}\,(10^3 q_{g,k})\big], \\
G_{\mathrm{chg}} &= \max_k\!\big[W_T(T_k)\,\rho_{d,k}\,w_{+,k}^{1.25}\,(10^3 q_{g,k})\big]
\end{aligned} \tag{5.16}
$$

$$
\begin{aligned}
F_1 &= 0.042\,G_{\mathrm{chg}}, \qquad F_2 = 0.20\,\mathrm{IWP}_{\mathrm{tot}}\left(\frac{G_{\max}}{G_{\max} + 1.0}\right)^2, \\
\mathcal{F}_{\mathrm{tot}} &= \frac{c_{\mathrm{cal}}(0.95 F_1 + 0.05 F_2)}{300\times 10^6}\ [\mathrm{flashes\ m^{-2}\ s^{-1}}]
\end{aligned} \tag{5.17}
$$

The intracloud-to-cloud-to-ground ratio $Z$ and CG fraction $f_{\mathrm{CG}}$ follow the Price & Rind (1993, Eq. 1) quartic polynomial in cold-cloud thickness $H\in[5.5, 14.0]\ \mathrm{km}$ (between $0^\circ\mathrm{C}$ and cloud top where $q_i + q_{p,\mathrm{ice}} \ge 10^{-5}$). Using Schumann & Huntrieser (2007) yield $Y_{\mathrm{N}} = 3.5\ \mathrm{kg\ N\ flash^{-1}}$ ($S_{\mathrm{N,loc}} = Y_{\mathrm{N}}\mathcal{F}_{\mathrm{tot}}$), conservative $\mathrm{LNO}_x$ deposition blends 50% local and 50% global-mean production:

$$
\begin{aligned}
Z &= \operatorname{clamp}\!\big(0.021 H^4 - 0.648 H^3 + 7.493 H^2 - 36.54 H + 63.09,\ 1,\ 19\big), \\
f_{\mathrm{CG}} &= \frac{1}{1 + Z}, \qquad D_{\mathrm{N,LNOx}} = 0.50\,S_{\mathrm{N,loc}} + 0.50\,\overline{S}_{\mathrm{N,glob}}
\end{aligned} \tag{5.18}
$$

#### 5.6.2 Explicit Bulk Cloud Electrification & Discharge Scheme (`lightning_scheme = "explicit_bulk"`, `src/aoba/lightning_explicit.py`)

In `explicit_bulk` mode, volumetric electric charge densities $\rho_{q,s}\ [\mathrm{C\ m^{-3}}]$ are prognosed across five condensed hydrometeor species $s \in \{c, r, i, s_{\mathrm{nw}}, g\}$ on the 2D/3D CRM mesh. Rebounding graupel–ice and graupel–snow collisions transfer charge $\delta q_{\mathrm{coll}}(T, \mathrm{LWC})$ to graupel following a smooth Takahashi (1978) / Sato et al. (2019) polarity-reversal law active within $-40^\circ\mathrm{C} < T_C < 0^\circ\mathrm{C}$ ($\delta q_{\max} = 2.5\times 10^{-14}\ \mathrm{C}$, liquid water content $\mathrm{LWC} = 10^3\rho_d q_c\ [\mathrm{g\ m^{-3}}]$):

$$
\begin{aligned}
T_{\mathrm{rev}}(\mathrm{LWC}) &= -10.0 - 10.0\,\tanh\!\big(0.65(\mathrm{LWC} - 0.55)\big), \\
\delta q_{\mathrm{coll}} &= \delta q_{\max}\,\tanh\!\left(\frac{T_C - T_{\mathrm{rev}}}{4.0}\right)\frac{\mathrm{LWC}}{\mathrm{LWC} + 0.15}\,\sigma\!\left(\frac{T_C + 40}{3.0}\right)\sigma\!\left(-\frac{T_C}{1.5}\right)\,\mathbf{1}_{T_C < 0}
\end{aligned} \tag{5.19}
$$

where $\sigma(x) = (1 + e^{-x})^{-1}$ is the logistic gate. Gravitational sweep-out collision rates $R_{gi}, R_{gs}$ (rebound fraction $\alpha_{\mathrm{reb}} = 0.90$) generate strict zero-sum charge separation ($\Delta\rho_{q,g} = -(\Delta\rho_{q,i} + \Delta\rho_{q,s})$), after which all five charge species undergo conservative periodic horizontal donor-cell advection, unconditionally stable implicit vertical advection–sedimentation with terminal fall speeds $v_{t,s}$, melting transfer from graupel/snow to rain above $0^\circ\mathrm{C}$, and exponential fair-weather ionic relaxation ($\tau_{\mathrm{ion}} = 600\ \mathrm{s}$; Chiu, 1978):

$$
\begin{aligned}
R_{gx} &= \alpha_{\mathrm{reb}}\,\frac{\pi}{4}(D_g + D_x)^2\,|v_{t,g} - v_{t,x}|\,N_g N_x, \qquad x\in\{i, s\}, \\
\frac{\partial \rho_{q,s}}{\partial t} &= -\nabla_h\cdot(\mathbf{u}_h \rho_{q,s}) - \frac{\partial}{\partial z}\big[(w - v_{t,s})\rho_{q,s}\big] + S_{\mathrm{ni},s} + S_{\mathrm{melt},s} - \frac{\rho_{q,s}}{\tau_{\mathrm{rel},s}}
\end{aligned} \tag{5.20}
$$

From total space charge $\rho_q = \sum_s \rho_{q,s}$, the 2D/3D electrostatic potential $\Phi_e$ and electric field $\mathbf{E} = -\nabla \Phi_e$ satisfy Poisson's equation with periodic horizontal boundaries and grounded Dirichlet boundaries $\Phi_e(0) = \Phi_e(z_{\mathrm{top}}) = 0$, solved non-iteratively in a single step by batched 2D real FFT in $(x, y)$ coupled to the stretched-grid tridiagonal Thomas solver in $z$ for each discrete horizontal eigenvalue $\lambda_{m,n}$:

$$
\begin{aligned}
\nabla^2 \Phi_e &= -\frac{\rho_q}{\varepsilon_0}, \qquad \mathbf{E} = -\nabla \Phi_e, \\
\lambda_{m,n} &= \frac{4}{\Delta x^2}\sin^2\!\left(\frac{\pi m}{N_x}\right) + \frac{4}{\Delta y^2}\sin^2\!\left(\frac{\pi n}{N_y}\right)
\end{aligned} \tag{5.21}
$$

Whenever peak $|\mathbf{E}|$ in a CRM column exceeds the density-scaled runaway-breakdown threshold $E_{\mathrm{crit}} = E_{\mathrm{crit},0}(\rho_d/\rho_0)$ ($E_{\mathrm{crit},0} = 120\ \mathrm{kV\ m^{-1}}$; Dwyer, 2003; Sato et al., 2019), a discharge channel spans all connected grid volumes with $|\mathbf{E}| \ge 0.30\,E_{\mathrm{crit}}$ (Fierro et al., 2013), neutralizing fraction $f_{\mathrm{neut}} = \min(0.30\min(\max(|\mathbf{E}|/E_{\mathrm{crit}}), 3), 0.80)$ of local charge. Pure intracloud (IC) discharges remove the channel-mean charge imbalance proportionally to $|\rho_{q,s}|$ so net space charge is conserved identically:

$$
\begin{aligned}
E_{\mathrm{crit}} &= E_{\mathrm{crit},0}\,\max\!\left(\frac{\rho_d}{\rho_0},\ 0.05\right), \\
\Delta\rho_{q,s}^{\mathrm{neut}} &= -f_{\mathrm{neut}}\,\rho_{q,s} - \mathbf{1}_{\mathrm{IC}}\,\frac{|\rho_{q,s}|}{\sum_{s'}|\rho_{q,s'}|}\,\overline{\Delta\rho_q^{\mathrm{raw}}}_{\mathrm{chan}}
\end{aligned} \tag{5.22}
$$

Discharges whose breakdown channel extends below $3.5\ \mathrm{km}$ AGL or induces surface vertical field $|E_z(z_0)| \ge 0.25\,E_{\mathrm{crit}}(z_0)$ are classified as cloud-to-ground (CG) flashes. Subcolumn total flash-rate density $\mathcal{F}_{\mathrm{tot}}$ follows from the neutralized bipolar charge magnitude ($Q_{\mathrm{flash}} = 15\ \mathrm{C\ flash^{-1}}$; Rakov & Uman, 2003), feeding the same Schumann & Huntrieser (2007) mass-conserving $\mathrm{LNO}_x$ deposition flux:

$$
\begin{aligned}
\mathcal{F}_{\mathrm{tot}} &= \frac{1}{2\,Q_{\mathrm{flash}}\,\Delta t}\int_0^{z_{\mathrm{top}}} \left|\sum_s \Delta\rho_{q,s}^{\mathrm{neut}}\right|\,dz\ [\mathrm{flashes\ m^{-2}\ s^{-1}}], \\
D_{\mathrm{N,LNOx}} &= 0.50\,Y_{\mathrm{N}}\mathcal{F}_{\mathrm{tot}} + 0.50\,\overline{Y_{\mathrm{N}}\mathcal{F}_{\mathrm{tot}}}_{\mathrm{glob}}
\end{aligned} \tag{5.23}
$$

When Online Self-Distilling Superparameterization (`enable_crm_self_distill = true`) is active, full-resolution 2 km CRM lightning (`explicit_bulk` or `diagnostic`) executes strictly on the active teacher subset (`1 / stride` host cells, resetting newly rotated teacher slots to neutral charge while evolving dwelling slots continuously across `crm_distill_dwell_steps`), and a compact map-free two-part Hurdle MLP (`LightningSelfDistillNet`, taking 14 universal column thermodynamic, mixed-phase hydrometeor, CAPE-proxy, surface buoyancy, and precipitation features with a zero-cloud gate) distills strike probability $P(\mathcal{F}_{\mathrm{tot}} > 0 \mid \mathbf{x})$, conditional intensity $\mathbb{E}[\ln(1 + \mathcal{F}_{\mathrm{tot}}/\mathcal{F}_0) \mid \text{strike}, \mathbf{x}]$, and CG fraction $f_{\mathrm{CG}}$ online onto all student columns.

#### 5.6.3 Optional Mesospheric Red Sprites & Sprite Halos (`enable_red_sprites = true`, `src/aoba/red_sprites.py`)

When `enable_red_sprites = true` (default `false`), a map-free quasi-electrostatic (QE) mesospheric Transient Luminous Event (TLE) module evaluates sprite halos, prompt downward streamers, sub-breakdown delayed sprites, $\mathrm{N}_2$ optical emissions, and mesospheric $\mathrm{NO}_x$ production across a 26-level mesospheric grid ($z \in [38, 88]\ \mathrm{km}$, $\Delta z = 2\ \mathrm{km}$) anchored hydrostatically to AOBA's upper-atmosphere $(\rho, T)$. Solar zenith angle $\mu_0$ modulates the Wait & Spies (1964) / Pasko et al. (1997) D-region ionospheric electron density $N_e(z)$ ($h'_{\mathrm{night}} = 85\ \mathrm{km}, \beta_{\mathrm{night}} = 0.50\ \mathrm{km}^{-1}$; $h'_{\mathrm{day}} = 70\ \mathrm{km}, \beta_{\mathrm{day}} = 0.30\ \mathrm{km}^{-1}$) and total conductivity $\sigma(z)$, yielding dielectric relaxation time $\tau_\sigma(z) = \varepsilon_0/\sigma(z)$:

$$
\begin{aligned}
N_e(z) &= 1.43\times 10^{13}\exp\!\big(-0.15\,h' + (\beta - 0.15)(z - h')\big)\ [\mathrm{m^{-3}}], \\
\sigma(z) &= q_e N_e(z)\,\mu_{e0}\frac{N_0}{N(z)} + 5\times 10^{-14}\exp\!\left(\frac{z}{6\ \mathrm{km}}\right), \qquad \tau_\sigma(z) = \frac{\varepsilon_0}{\sigma(z)}
\end{aligned} \tag{5.24}
$$

Above a CG flash removing vertical charge moment change $\Delta M_Q = Q_{\mathrm{CG}} z_Q\ [\mathrm{C\ km}]$ over impulse timescale $\tau_f = 1\ \mathrm{ms}$, the conducting-ground image dipole QE field $E_{\mathrm{QE}}(z)$ is screened analytically by dielectric relaxation ($r = \tau_\sigma/\tau_f$, $u = \ln r$) and compared against the Townsend density-scaled conventional breakdown field $E_k(z) = E_{k0}(N/N_0)$ ($E_{k0} = 3.2\times 10^6\ \mathrm{V\ m^{-1}} \approx 120\ \mathrm{Td}$; Pasko et al., 1997):

$$
\begin{aligned}
E_{\mathrm{QE}}(z) &= \frac{\Delta M_Q}{\pi\varepsilon_0}\frac{z}{(z^2 - z_Q^2)^2}, \\
E_{\mathrm{peak}}(z) &= E_{\mathrm{QE}}(z)\,\exp\!\left(-\frac{u}{e^u - 1}\right), \qquad E_k(z) = E_{k0}\frac{N(z)}{N_0}
\end{aligned} \tag{5.25}
$$

Sprite halos occur above $E_{\mathrm{peak}} \ge 0.50\,E_k$ (Qin et al., 2014), prompt downward streamers above $E_{\mathrm{peak}} \ge E_k$ (Luque & Ebert, 2009), and delayed sprites via $\mathrm{O}^- + \mathrm{N}_2 \to \mathrm{N}_2\mathrm{O} + e^-$ associative electron detachment across $0.70\,E_k \le E_{\mathrm{peak}} < E_k$ whenever $\tau_\sigma \ge 1.5\ \mathrm{ms}$ (Luque & Gordillo-Vázquez, 2012). Integrating analytically over a stratiform-anvil-conditioned log-normal CG charge-moment distribution ($\sigma_{\ln} = 1.15$) yields exact exceedance probabilities via $\operatorname{erfc}$, while downward streamer propagation halts at stopping altitude $z_{\mathrm{stop}}$ where $E_{\mathrm{peak}}(z)$ falls below Ebert et al. (2010) positive/negative stability fields ($E_{\mathrm{st},0}^+ = 4.4\times 10^5$, $E_{\mathrm{st},0}^- = 1.25\times 10^6\ \mathrm{V\ m^{-1}}$):

$$
\begin{aligned}
P(\Delta M_Q \ge \Delta M_{Q,\mathrm{crit}}) &= \frac{1}{2}\operatorname{erfc}\!\left(\frac{\ln(\Delta M_{Q,\mathrm{crit}} / \Delta M_{Q,\mathrm{med}})}{\sqrt{2}\,\sigma_{\ln}}\right), \\
z_{\mathrm{stop}} &= \min\!\big\{z \le z_{\mathrm{init}} : E_{\mathrm{peak}}(z) \ge E_{\mathrm{st}}^\pm(z)\big\}
\end{aligned} \tag{5.26}
$$

Optical intensities of the crimson $\mathrm{N}_2$ 1st Positive band ($1\mathrm{PN}_2$, 650–750 nm; $A_{1P} = 1.7\times 10^5\ \mathrm{s^{-1}}$, $\mathrm{N}_2$ quenching below $\sim 53\ \mathrm{km}$) and violet/blue $\mathrm{N}_2$ 2nd Positive band ($2\mathrm{PN}_2$, 337 nm; $A_{2P} = 2.0\times 10^7\ \mathrm{s^{-1}}$, $\mathrm{O}_2$ quenching below $\sim 30\ \mathrm{km}$) reproduce the red-head to blue-tendril altitude transition (Pasko et al., 1997), and column mesospheric $\mathrm{NO}_x$ production follows Sentman et al. (2008; $Y_{\mathrm{NOx}} = 5\times 10^{22}\ \mathrm{molecules\ sprite^{-1}}$):

$$
\begin{aligned}
Q_{1P}(z) &= \frac{A_{1P}}{A_{1P} + k_{q,1P} f_{\mathrm{N}_2} N(z)}, \qquad Q_{2P}(z) = \frac{A_{2P}}{A_{2P} + k_{q,2P} f_{\mathrm{O}_2} N(z)}, \\
S_{\mathrm{NOx,sprite}} &= \mathcal{F}_{\mathrm{sprite}}\,Y_{\mathrm{NOx}}\,\operatorname{clamp}\!\left(\frac{z_{\mathrm{init}} - z_{\mathrm{stop}}}{25\ \mathrm{km}},\ 0.3,\ 2.0\right)
\end{aligned} \tag{5.27}
$$

---

## 6. Boundary-Layer Turbulence, Roughness Sublayer (RSL), MYNN 2.5 & 3D Smagorinsky–Lilly SGS

### 6.1 Canopy and Ocean Aerodynamic Roughness (`src/aoba/turbulence.py`)

Vegetated zero-plane displacement $d_0$ and roughness lengths $(z_{0m}, z_{0h})$ are derived analytically from prognostic canopy height $h_c$ and exposed $\mathrm{LAI}$ following Raupach (1994, Eqs. 8, 10–11; $c_{d1} = 7.5$, $\psi_{cw} = 0.193$, $z_{0,\mathrm{soil}} = 0.01\ \mathrm{m}$):

$$
x = \sqrt{7.5\,\max(\mathrm{LAI}, 10^{-4})}, \qquad 1 - \frac{d_0}{h_c} = \frac{1 - e^{-x}}{x}, \qquad \frac{u_*}{U_h} = \min\!\left(\sqrt{0.003 + 0.15\,\mathrm{LAI}},\ 0.30\right) \tag{6.1}
$$

$$
z_{0m,\mathrm{land}} = \max\!\left(h_c\left(1 - \frac{d_0}{h_c}\right)\exp\!\left(-\frac{\kappa}{u_*/U_h} + 0.193\right),\ z_{0,\mathrm{soil}}\right), \qquad z_{0h,\mathrm{land}} = e^{-2} z_{0m,\mathrm{land}} \tag{6.2}
$$

Ocean surface roughness combines smooth viscous sublayer and Charnock wave-drag regimes (Miller, Beljaars & Palmer, 1992, Eqs. 3–5; SCALE-RM Eqs. 8.360–8.362; $\nu_a = 1.5\times 10^{-5}\ \mathrm{m^2\ s^{-1}}$):

$$
z_{0m,\mathrm{ocn}} = \frac{0.11\,\nu_a}{u_*} + \frac{0.018\,u_*^2}{g}, \qquad z_{0h,\mathrm{ocn}} = \frac{0.40\,\nu_a}{u_*} + 1.4\times 10^{-5}, \qquad z_{0q,\mathrm{ocn}} = \frac{0.62\,\nu_a}{u_*} + 1.3\times 10^{-4} \tag{6.3}
$$

### 6.2 Monin–Obukhov Similarity and De Ridder (2010) Analytical RSL Correction

Enhanced turbulent mixing by coherent eddies within the canopy roughness sublayer ($z \le z_* = \max(2h_c - d_0, 0)$, dimensionless height $\chi = z/z_*$) is incorporated into Paulson–Dyer stability functions $(\Phi_i, \Psi_i)$ via the analytical RSL correction $\psi_i^*(\zeta, \chi)$ (De Ridder, 2010, Eq. 22; $\mu_M = 2.59$, $\mu_H = 0.95$, $\nu_{\mathrm{RSL}} = 0.50$, $\lambda_{\mathrm{RSL}} = 1.50$):

$$
\psi_i^*(\zeta, \chi) = \Phi_i\!\left(\left(1 + \frac{\nu_{\mathrm{RSL}}}{\mu_i\chi}\right)\zeta\right)\frac{1}{\lambda_{\mathrm{RSL}}}\ln\!\left(1 + \frac{\lambda_{\mathrm{RSL}}}{\mu_i\chi}\right)e^{-\mu_i\chi}, \qquad i\in\{M, H\} \tag{6.4}
$$

Given effective gust-floored wind $U_{\mathrm{eff}}$ and bulk Richardson number $\mathrm{Ri}_b$, four fixed-point iterations converge stability $\zeta$ and profile functions $f_i$, yielding bulk transfer coefficients $(C_M, C_H, C_Q)$ and friction velocity $u_*$:

$$
\begin{aligned}
U_{\mathrm{eff}} &= \sqrt{u_1^2 + v_1^2 + 0.25}, \\
f_i &= \max\!\left(\ln\frac{z}{z_{0i}} - \Psi_i(\zeta) + \Psi_i(\zeta_{0i}) - \big[\psi_i^*(\zeta_{0i}, \chi_{0i}) - \psi_i^*(\zeta, \chi)\big],\ 0.8\right), \\
\zeta &\leftarrow \operatorname{clamp}\!\left(\mathrm{Ri}_b\frac{f_M^2}{f_H},\ -5,\ 2\right)
\end{aligned} \tag{6.5}
$$

$$
C_M = \frac{\kappa^2}{f_M^2}, \qquad C_H = \frac{\kappa^2}{f_M f_H}, \qquad C_Q = \frac{\kappa^2}{f_M f_Q}, \qquad u_* = \sqrt{C_M}\,U_{\mathrm{eff}} \tag{6.6}
$$

### 6.3 Implicit MYNN Level 2.5 PBL Closure (`step_mynn_level25`)

Implements Nakanishi & Niino (2009) and SCALE-RM v5.5.5 (Sec. 8.2.1, Eqs. 8.96–8.176) with closure constants $B_1 = 24.0, B_2 = 15.0, C_2 = 0.75, C_3 = 0.352, C_5 = 0.20, \gamma_1 = 0.235, \mathrm{Pr}_N = 0.74$ (derived $A_1 = 1.180, A_2 = 0.6645, C_1 = 0.1371, \gamma_2 = 0.5525, R_{f1} = 0.3737, R_{f2} = 0.3131, R_{fc} = 0.2984$). Liquid–ice potential temperature $\theta_l$ serves as the conserved thermodynamic variable. The turbulent integral scale $L_T$ weights excess turbulent velocity $q_{\mathrm{exc}} = \max(q - \sqrt{2\,\mathrm{TKE}_{\min}}, 0)$ over the boundary-layer depth envelope ($z_{\mathrm{rel}} \le 4\ \mathrm{km}$, bounded to $[20, 500]\ \mathrm{m}$), and the master mixing length $L$ harmonically combines surface-layer ($L_S$), integral ($L_T$), and buoyancy ($L_B$) length scales subject to an altitude-tapered upper envelope $L_{\max}(z) = 30 + 270\exp(-\max(z - 3\ \mathrm{km}, 0)/1.5\ \mathrm{km})\ \mathrm{m}$:

$$
\begin{aligned}
\theta_l &= \theta - \frac{L_{v0} q_c + L_{s0} q_i}{c_{pd}\Pi}, \\
L_T &= 0.23\,\frac{\int_{0}^{4\ \mathrm{km}} q_{\mathrm{exc}}\,z_{\mathrm{rel}}\,dz}{\int_{0}^{4\ \mathrm{km}} q_{\mathrm{exc}}\,dz}, \\
L &= \big(L_S^{-1} + L_T^{-1} + L_B^{-1}\big)^{-1}
\end{aligned} \tag{6.7}
$$

A Gaussian subgrid partial-condensation closure ($\sigma_s$, normalized supersaturation $Q_1$, cloud fraction $R$, effective cloud fraction $\tilde{R}$) evaluates moist buoyancy frequency $N^2$, while Level-2 equilibrium $q_2^2$ and ratio $\alpha_c$ determine Level-2.5 stability functions $(S_M, S_H)$ and eddy diffusivities $(K_M, K_H)$ (realizability-bounded to $[0, 500]\ \mathrm{m^2\ s^{-1}}$):

$$
\begin{aligned}
\sigma_s &= \frac{1}{2}\,a L\sqrt{\frac{B_2}{2}}\left|\frac{\partial q_n}{\partial z} - b\frac{\partial \theta_l}{\partial z}\right|, \qquad Q_1 = \frac{a(q_n - q_{sl})}{2\sigma_s}, \\
N^2 &= \frac{g}{\theta}\left(\beta_\theta\frac{\partial \theta_l}{\partial z} + \beta_q\frac{\partial q_n}{\partial z}\right), \\
q_2^2 &= B_1 L^2 S_{M2}(1 - R_f) S^2, \qquad \alpha_c = \min\!\left(\frac{q}{q_2},\ 1\right), \\
K_M &= \min(L q S_M,\ 500), \qquad K_H = \min(L q S_H,\ 500)
\end{aligned} \tag{6.8}
$$

Twice turbulent kinetic energy ($q^2 = 2e$, bounded to $[2\,\mathrm{TKE}_{\min}, 100]\ \mathrm{m^2\ s^{-2}}$) advances with explicit positive production $P_+ = \max(P, 0)$ and implicit tridiagonal Thomas diffusion ($3 K_M$), negative buoyancy sink, and equilibrium-consistent viscous dissipation $2 q_{\mathrm{diss}}/(B_1 L)$ where $q_{\mathrm{diss}} = \max\!\big(q,\ (B_1 L P_+ / 2)^{1/3}\big)$ guarantees monotonic convergence to the local cubic balance at large host time steps:

$$
\frac{\partial q^2}{\partial t} = 2\left(K_M S^2 - K_H N^2\right) + \frac{1}{\rho_d}\frac{\partial}{\partial z}\left(3\rho_d K_M \frac{\partial q^2}{\partial z}\right) - \frac{2 q^3}{B_1 L} \tag{6.9}
$$

Implicit horizontal momentum diffusion folds opposing surface drag ($\tau_{\mathrm{sfc}}\cdot u_0 < 0$) into the lowest-layer diagonal coefficient $r_{\mathrm{drag}}$ to prevent large-step overshoot, after which $(\theta_l, q_n)$ diffuse implicitly via $K_H$ and undergo saturation adjustment to recover $\theta_m$:

$$
r_{\mathrm{drag}} = \frac{|\tau_{\mathrm{sfc}}|}{\max(|u_0|,\ 0.10)} \tag{6.10}
$$

### 6.4 Anisotropic 3D Smagorinsky–Lilly SGS Closure (`compute_smagorinsky_lilly_3d`)

Horizontal subgrid-scale mixing in the CRM applies Lilly (1962) with the Scotti, Meneveau & Lilly (1993) grid-aspect-ratio correction $f(a)$ (SCALE-RM Eqs. 8.25–8.40; $C_S = 0.18$, $\mathrm{Pr}_N = 0.70$, $\mathrm{Ri}_c = 0.25$, $g_c = (1 - \mathrm{Pr}_N)/\mathrm{Ri}_c = 1.2$):

$$
\begin{aligned}
a &= \max\!\left(\frac{\Delta_h}{\Delta z},\ \frac{\Delta z}{\Delta_h}\right), \qquad \lambda = f(a)(\Delta x\Delta y\Delta z)^{1/3}, \\
\nu_{\mathrm{SGS}} &= \begin{cases} (C_S\lambda)^2 |S|\sqrt{1 - 16\,\mathrm{Ri}}, & \mathrm{Ri} < 0 \\ (C_S\lambda)^2 |S|(1 - \mathrm{Ri}/\mathrm{Ri}_c)^4, & 0 \le \mathrm{Ri} < \mathrm{Ri}_c \\ 0, & \mathrm{Ri} \ge \mathrm{Ri}_c \end{cases}, \\
D_{\mathrm{SGS}} &= \begin{cases} \dfrac{(C_S\lambda)^2 |S|}{\mathrm{Pr}_N}\sqrt{1 - 40\,\mathrm{Ri}}, & \mathrm{Ri} < 0 \\ \dfrac{(C_S\lambda)^2 |S|}{\mathrm{Pr}_N}(1 - \mathrm{Ri}/\mathrm{Ri}_c)^4(1 - g_c\mathrm{Ri}), & 0 \le \mathrm{Ri} < \mathrm{Ri}_c \\ 0, & \mathrm{Ri} \ge \mathrm{Ri}_c \end{cases}
\end{aligned} \tag{6.11}
$$

---

## 7. Land Surface, Autonomous Two-Leaf $\mathrm{C}_3/\mathrm{C}_4$ EEO Biosphere, Soil Hydrology, Terrestrial C/N, Wildfire & River Routing

### 7.1 Land-Surface, Biosphere, and Biogeochemical Coupling Overview

AOBA's land system eliminates prescribed PFT and soil-texture maps, solving two-leaf canopy radiative transfer, autonomous $\mathrm{C}_3/\mathrm{C}_4$ Eco-Evolutionary Optimality (EEO) photosynthetic and respiratory acclimation, prognostic optimal LAI/leaf-habit/root-zone/canopy-height traits, three-source energy balance, hysteretic snow cover and SNICAR darkening, 5-layer freeze–thaw Richards hydrology and thermal conduction, MAML subgrid terrain microclimate and dynamic wetlands, Stocker–Prentice terrestrial C/N and 5-layer permafrost C/N, MAML discrete-patch SPITFIRE wildfire, and Priority-Flood Voronoi river routing over each independent CRM subgrid tile.

![Figure 7.1: MAML Subgrid Land Surface, Autonomous EEO Biosphere, Soil Hydrology & Terrestrial C/N](assets/fig_05_land_maml_eeo.svg)

### 7.2 Canopy Radiative Transfer, Discrete-Crown Clumping, Optimal Leaf Angle & Two-Leaf Allocation (`src/aoba/biosphere.py`)

Rather than imposing a fixed spherical leaf-angle distribution, leaf projection $G(\mu_0; \chi_L)$ and inverse diffuse optical depth $\bar{\mu}(\chi_L)$ follow Ross (1975), Goudriaan (1977), and Dai et al. (2004, Eq. A4c; smoothly evaluated via a 3-term Taylor expansion as $\phi_2 \to 0$ so $\bar{\mu} \to (2\phi_1)^{-1} = 1$) as functions of the leaf orientation index $\chi_L \in [-0.40, +0.50]$ ($0$: spherical, $<0$: erectophile, $>0$: planophile). A non-iterative closed-form optimality target $\chi_L^*$ balances high-LAI light penetration (Monsi & Saeki, 1953; Hikosaka & Hirose, 1997) against aridity, VPD, and high-insolation heat/transpiration avoidance (Ehleringer & Werk, 1986; Falster & Westoby, 2003; Pisek et al., 2013), acclimating daily on a 30-day timescale:

$$
\begin{aligned}
\phi_1 &= 0.5 - 0.633\,\chi_L - 0.330\,\chi_L^2, \qquad \phi_2 = 0.877(1 - 2\phi_1), \qquad G(\mu_0;\chi_L) = \phi_1 + \phi_2\,\mu_0, \\
\bar{\mu}(\chi_L) &= \frac{1}{\phi_2}\left[1 - \frac{\phi_1}{\phi_2}\ln\!\left(1 + \frac{\phi_2}{\phi_1}\right)\right], \\
\chi_{\mathrm{light}} &= 0.25\tanh\!\left(\frac{2.5 - L}{2.0}\right), \qquad S_{\mathrm{dry}} = \operatorname{clamp}\!\left(\frac{1.2 - \mathrm{MI}}{1.2},\ 0,\ 1\right), \\
S_{\mathrm{vpd}} &= \frac{D_{\mathrm{mid}}}{D_{\mathrm{mid}} + 1500\ \mathrm{Pa}}, \qquad S_{\mathrm{sun}} = \operatorname{clamp}\!\left(\frac{\bar{\mu}_{0,\mathrm{mid}} - 0.35}{0.55},\ 0,\ 1\right), \\
\chi_L^* &= \operatorname{clamp}\!\Big(\chi_{\mathrm{light}} - 0.55(0.60 S_{\mathrm{dry}} + 0.40 S_{\mathrm{vpd}})(0.40 + 0.60 S_{\mathrm{sun}}),\ -0.40,\ +0.50\Big)
\end{aligned} \tag{7.1}
$$

where $\mathrm{MI} = P_{\mathrm{ann}}/\mathrm{PET}_{\mathrm{ann}}$ is the annual moisture index and $D_{\mathrm{mid}}, \bar{\mu}_{0,\mathrm{mid}}$ are EMA midday VPD and solar zenith cosine. To eliminate the sparse-canopy over-absorption bias of horizontally uniform infinite turbid-medium canopies (de Pury & Farquhar, 1997) without PFT lookup tables, Poisson-distributed discrete spherical crowns of minimum internal foliage density $L_{c,0} = 1.50\ \mathrm{m^2\ m^{-2}}$ yield a zenith-angle-dependent analytical clumping index $\Omega(\mu, L_c) \in [0.35, 1.0]$ (Li & Strahler, 1992; Nilson, 1999; Ni-Meister et al., 2010; Braghiere et al., 2019):

$$
\begin{aligned}
f_c(\mu) &= 1 - \exp\!\left(-\frac{L_c}{L_{c,0}\max(\mu,\ 0.15)}\right), \qquad k_0(\mu) = \frac{G(\mu;\chi_L)}{\mu}, \\
P_{\mathrm{gap}}(\mu) &= \big(1 - f_c(\mu)\big) + f_c(\mu)\exp\!\left(-\frac{k_0(\mu)\,L_c}{f_c(\mu)}\right), \\
\Omega(\mu, L_c) &= \frac{-\ln P_{\mathrm{gap}}(\mu)}{k_0(\mu)\,L_c}, \qquad \lim_{L_c\to 0}\Omega(\mu, L_c) = \frac{1 - e^{-G(\mu;\chi_L) L_{c,0}}}{G(\mu;\chi_L) L_{c,0}}
\end{aligned} \tag{7.2}
$$

In sparse savannas and shrublands, inter-crown bare gaps and intra-crown self-shading reduce $\Omega < 1$, whereas oblique low-sun angles increase shadow overlap and closed canopies ($L_c \gg L_{c,0}$) converge smoothly to $\Omega \to 1$. Using McCree (1972) photon conversions ($4.57$ direct, $4.24\ \mu\mathrm{mol\ J^{-1}}$ diffuse) and leaf scattering coefficients ($\sigma_{\mathrm{vis}} = 0.15$, $\sigma_{\mathrm{nir}} = 0.80$), clumped direct extinction $k_{db}$, scattering-adjusted extinction $k'_b$, diffuse extinction $k'_d$ (using diffuse clumping $\Omega_d = \Omega(\mu_d, L_c)$ at $\mu_d = \operatorname{clamp}(0.55\bar{\mu}, 0.25, 0.90)$), and canopy top reflectances $(\rho_{cb}, \rho_{cd})$ are:

$$
\begin{aligned}
k_{db} &= \Omega(\mu_0, L_c)\,\frac{G(\mu_0;\chi_L)}{\max(\mu_0,\ 0.01)}, \qquad k'_b = k_{db}\sqrt{1 - \sigma_{\mathrm{vis}}}, \qquad k'_d = \frac{0.719\,\Omega_d}{\bar{\mu}(\chi_L)}, \\
\rho_h &= \frac{1 - \sqrt{1 - \sigma_{\mathrm{vis}}}}{1 + \sqrt{1 - \sigma_{\mathrm{vis}}}}, \qquad \rho_{cb} = \frac{2\rho_h k_{db}}{1 + k_{db}}, \qquad \rho_{cd} = 0.036
\end{aligned} \tag{7.3}
$$

Total, sunlit, and shaded absorbed PAR $(I_c, I_{\mathrm{sun}}, I_{\mathrm{sh}})$ and sunlit/shaded leaf area indices $(L_{\mathrm{sun}}, L_{\mathrm{sh}})$ are evaluated analytically:

$$
\begin{aligned}
I_c &= (1 - \rho_{cb}) I_{b0}\big(1 - e^{-k'_b L_c}\big) + (1 - \rho_{cd}) I_{d0}\big(1 - e^{-k'_d L_c}\big), \\
L_{\mathrm{sun}} &= \frac{1 - e^{-k_{db} L_c}}{k_{db}}, \qquad L_{\mathrm{sh}} = L_c - L_{\mathrm{sun}}
\end{aligned} \tag{7.4}
$$

$$
\begin{aligned}
I_{\mathrm{sun}} &= I_{b0}(1 - \sigma_{\mathrm{vis}})\big(1 - e^{-k_{db} L_c}\big) + I_{b0}\!\left[\frac{(1-\rho_{cb})k'_b\big(1 - e^{-(k'_b+k_{db})L_c}\big)}{k'_b+k_{db}} - \frac{1-\sigma_{\mathrm{vis}}}{2}\big(1 - e^{-2k_{db} L_c}\big)\right] \\
&\quad + I_{d0}\,\frac{(1-\rho_{cd})k'_d\big(1 - e^{-(k'_d+k_{db})L_c}\big)}{k'_d+k_{db}}, \\
I_{\mathrm{sh}} &= \max(I_c - I_{\mathrm{sun}},\ 0)
\end{aligned} \tag{7.5}
$$

Assuming optimal vertical enzyme allocation proportional to the EMA-smoothed midday absorbed irradiance profile $(\tilde{I}_{b0}, \tilde{I}_{d0}, \tilde{k}'_b, \tilde{\rho}_{cb}, \tilde{I}_c)$ (Field, 1983; Farquhar, 1989; de Pury & Farquhar, 1997), integrating against the instantaneous sunlit probability $f_{\mathrm{sun}}(L) = e^{-k_{db} L}$ partitions every whole-canopy capacity $X_{\mathrm{tot}}\in\{V_{c\max 25}, J_{\max 25}, V_{p\max 25}, R_{25}\}$ in closed form:

$$
\begin{aligned}
X_{\mathrm{sun}} &= \frac{X_{\mathrm{tot}}}{\tilde{I}_c}\left[\frac{(1-\tilde{\rho}_{cb})\tilde{k}'_b\tilde{I}_{b0}}{\tilde{k}'_b + k_{db}}\big(1 - e^{-(\tilde{k}'_b+k_{db})L_c}\big) + \frac{(1-\rho_{cd})k'_d\tilde{I}_{d0}}{k'_d + k_{db}}\big(1 - e^{-(k'_d+k_{db})L_c}\big)\right], \\
X_{\mathrm{sh}} &= \max(X_{\mathrm{tot}} - X_{\mathrm{sun}},\ 10^{-6})
\end{aligned} \tag{7.6}
$$

Direct and diffuse VIS/NIR canopy albedos and forest-floor transmittances use the general Ross–Goudriaan ($G(\mu_0;\chi_L), \bar{\mu}(\chi_L)$) Sellers (1985, Eq. 15) / Dai, Dickinson & Wang (2004, App. A) two-stream analytical solution evaluated at clumped effective optical depths ($L_{\mathrm{dir}} = \Omega(\mu_0, L_c)L_c$, $L_{\mathrm{dif}} = \Omega_d L_c$) with smooth regularization of the $\sigma_{\mathrm{sing}} = (\bar{\mu} k_{\mathrm{opt}})^2 - (b^2 - c^2) \to 0$ singularity, strictly conserving radiative energy.

### 7.3 Daily EEO Acclimation ($\mathrm{C}_3$ and $\mathrm{C}_4$ P-Model) and Leaf Respiration (`src/aoba/biosphere.py`)

Midday incident visible radiation, zenith cosine, leaf temperature $T_g$, VPD $D$, and nighttime temperature $T_{\mathrm{night}}$ are smoothed with a 15-day EMA ($\alpha_{\mathrm{EMA}} = 0.067$; Mengoli et al., 2022), alongside a 3-year growing-season ($T > 273.15\ \mathrm{K}$) thermal-home adaptation temperature $T_{\mathrm{home}}$. Relative water viscosity $\eta^*(T)$ (Vogel equation; Prentice et al., 2014), PSII quantum yield $\phi_{\mathrm{PSII}}(T)$ (Bernacchi et al., 2003; Smith et al., 2019), and MMRT dark-respiration scaling $f_r(T)$ (Heskel et al., 2016; Ren et al., 2024, 2025) are defined as ($T_C = T - 273.15$):

$$
\begin{aligned}
\eta^*(T) &= \frac{\exp\!\left(-3.719 + \dfrac{580}{T - 138}\right)}{\exp\!\left(-3.719 + \dfrac{580}{298.15 - 138}\right)}, \\
\phi_{\mathrm{PSII}}(T) &= 0.257\,\frac{\max(0.352 + 0.022 T_C - 3.4\times 10^{-4} T_C^2,\ 0.02)}{0.656}, \\
f_r(T) &= \exp\!\big(0.1012\,T_C - 0.0005\,T_C^2 - 2.2175\big)
\end{aligned} \tag{7.7}
$$

Peaked Arrhenius enzyme kinetics (Kumarathunge et al., 2019; Schneider, Gessler & Stocker, 2025; deactivation energy $H_d = 200\ \mathrm{kJ\ mol^{-1}}$, reference $T_0 = 298.15\ \mathrm{K}$) incorporate acclimated activation energy $H_a$ and entropy $\Delta S$:

$$
\begin{aligned}
f_{\mathrm{enz}}(T; T_g, T_{\mathrm{home}}) &= \exp\!\left[\frac{H_a(T - T_0)}{R^* T_0 T}\right]\frac{1 + \exp\!\left(\dfrac{T_0\Delta S - H_d}{R^* T_0}\right)}{1 + \exp\!\left(\dfrac{T\Delta S - H_d}{R^* T}\right)}, \\
H_{a,V}(T_g) &= 42600 + 1140\,T_{g,C}, \qquad \Delta S_V(T_g) = 645.13 - 0.38\,T_{g,C}, \\
H_{a,J} &= 40710, \qquad \Delta S_J(T_g, T_{\mathrm{home}}) = 658.77 - 0.84\,T_{\mathrm{home},C} - 0.52(T_{g,C} - T_{\mathrm{home},C}), \\
H_{a,P} &= 55000, \qquad \Delta S_P(T_g) = \Delta S_V(T_g)
\end{aligned} \tag{7.8}
$$

**$\mathrm{C}_3$ P-Model (Smith et al., 2019, Eqs. 1–20)**: Solving the non-rectangular hyperbola $\theta J^2 - (\phi_{\mathrm{PSII}} I_{\mathrm{abs}} + J_{\max})J + \phi_{\mathrm{PSII}} I_{\mathrm{abs}} J_{\max} = 0$ ($\theta = 0.85$, marginal electron-transport cost $c = 0.05336251$, water-to-carboxylation cost ratio $\beta_{\mathrm{C3}} = 146.0$) under the least-cost hypothesis yields stomatal sensitivity $\xi_{\mathrm{C3}}$, optimal $\chi_{\mathrm{C3}}$, electron-transport factor $m_{\mathrm{C3}}$, and closed-form optimality coefficients $(\omega, \omega^*)$:

$$
\begin{aligned}
\xi_{\mathrm{C3}} &= \sqrt{\frac{\beta_{\mathrm{C3}}(K_{m,\mathrm{C3}} + \Gamma^*_{\mathrm{C3}})}{1.6\,\eta^*}}, \\
\chi_{\mathrm{C3}} &= \operatorname{clamp}\!\left(\frac{\Gamma^*_{\mathrm{C3}}}{c_a} + \left(1 - \frac{\Gamma^*_{\mathrm{C3}}}{c_a}\right)\frac{\xi_{\mathrm{C3}}}{\xi_{\mathrm{C3}} + \sqrt{D}},\ 0.05,\ 0.95\right), \\
m_{\mathrm{C3}} &= \frac{c_{i,\mathrm{C3}} - \Gamma^*_{\mathrm{C3}}}{c_{i,\mathrm{C3}} + 2\Gamma^*_{\mathrm{C3}}}, \qquad c_{i,\mathrm{C3}} = \chi_{\mathrm{C3}} c_a
\end{aligned} \tag{7.9}
$$

$$
\begin{aligned}
\omega &= -(1 - 2\theta) + \operatorname{sgn}\!\left(1 - 2\theta\frac{4c}{m}\right)\sqrt{(1 - \theta)\left[\frac{1}{\dfrac{4c}{m}\left(1 - \theta\dfrac{4c}{m}\right)} - 4\theta\right]}, \\
\omega^* &= 1 + \omega - \sqrt{(1 + \omega)^2 - 4\theta\omega}
\end{aligned} \tag{7.10}
$$

Under the coordination hypothesis ($A_c = A_j$), optimal acclimated capacities $(J_{\max,\mathrm{opt}}^{\mathrm{C3}}, V_{c\max,\mathrm{opt}}^{\mathrm{C3}})$ and $25^\circ\mathrm{C}$ nighttime respiration capacity $R_{25}$ ($b_{\mathrm{resp}} = 0.018$; Ren et al., 2025, Eq. 13) are (zero when $m \le 4c$):

$$
\begin{aligned}
J_{\max,\mathrm{opt}}^{\mathrm{C3}} &= \phi_{\mathrm{PSII}} I_{\mathrm{abs}}\,\omega_{\mathrm{C3}}, \\
A_{j,\mathrm{opt}}^{\mathrm{C3}} &= \frac{\phi_{\mathrm{PSII}} I_{\mathrm{abs}}\,m_{\mathrm{C3}}\,\omega^*_{\mathrm{C3}}}{8\theta}, \\
V_{c\max,\mathrm{opt}}^{\mathrm{C3}} &= A_{j,\mathrm{opt}}^{\mathrm{C3}}\,\frac{c_{i,\mathrm{C3}} + K_{m,\mathrm{C3}}}{c_{i,\mathrm{C3}} - \Gamma^*_{\mathrm{C3}}}, \\
R_{25} &= \frac{b_{\mathrm{resp}}\,V_{c\max,\mathrm{opt}}}{f_r(T_{\mathrm{night}})}
\end{aligned} \tag{7.11}
$$

**$\mathrm{C}_4$ P-Model (Scott & Smith, 2022, JAMES, Eqs. 1–26; von Caemmerer, 2000)**: Represents mesophyll PEPC carboxylation ($K_{p25} = 60.5\ \mu\mathrm{mol\ mol^{-1}}$, $\Delta H_{a,K_p} = 27.2\ \mathrm{kJ\ mol^{-1}}$, $\beta_{\mathrm{C4}} = 166.0$) pumping $\mathrm{CO}_2$ into bundle-sheath cells (leakiness $\phi_L = 0.20$, bundle-sheath conductance $g_{bs} = 0.003\ \mathrm{mol\ m^{-2}\ s^{-1}}$, $\Gamma^*_{25,\mathrm{C4}} = 2.60\ \mathrm{Pa}$, $K_{c25,\mathrm{C4}} = 121.0\ \mathrm{Pa}$, $K_{o25,\mathrm{C4}} = 29200\ \mathrm{Pa}$). Optimal mesophyll ratio $\chi_{m,\mathrm{C4}}$ and $c_m = \chi_{m,\mathrm{C4}} c_a$ are:

$$
\begin{aligned}
\xi_{\mathrm{C4}} &= \sqrt{\frac{\beta_{\mathrm{C4}} K_p}{1.6\,\eta^*}}, \\
\chi_{m,\mathrm{C4}} &= \operatorname{clamp}\!\left(\frac{\xi_{\mathrm{C4}}}{\xi_{\mathrm{C4}} + \sqrt{D}},\ 0.05,\ 0.92\right)
\end{aligned} \tag{7.12}
$$

Incorporating the $5\ \mathrm{mol\ e^- / mol\ CO_2}$ RuBP/PEP stoichiometry ($4/5 = 0.80$), bundle-sheath leakage ATP overhead $(1 - \phi_L)$ (effective ratio $0.64$), and chilling sensitivity gate $g_{\mathrm{chill}}(T_g)$, two fixed-point iterations couple bundle-sheath $\mathrm{CO}_2$ partial pressure $c_{bs}$ and light-limited assimilation $A_{l,\mathrm{opt}}^{\mathrm{C4}}$:

$$
\begin{aligned}
g_{\mathrm{chill}}(T_g) &= \frac{1}{1 + \exp\!\big(-0.35(T_{g,C} - 12.0)\big)}, \\
m_{\mathrm{C4}} &= 0.80(1 - \phi_L)\,g_{\mathrm{chill}}(T_g)\,\frac{c_{bs} - \Gamma^*_{\mathrm{C4}}}{c_{bs} + 2\Gamma^*_{\mathrm{C4}}}, \\
A_{l,\mathrm{opt}}^{\mathrm{C4}} &= \frac{\phi_{\mathrm{PSII}} I_{\mathrm{abs}}\,m_{\mathrm{C4}}\,\omega^*_{\mathrm{C4}}}{8\theta}
\end{aligned} \tag{7.13}
$$

$$
\begin{aligned}
L_{\mathrm{leak}} &= \frac{\phi_L}{1 - \phi_L}\,A_{l,\mathrm{opt}}^{\mathrm{C4}}, \\
c_{bs} &= c_m + \frac{L_{\mathrm{leak}}\times 10^{-6}}{g_{bs}\,\max(L_{\mathrm{exp}}, 0.1)}\,p_{\mathrm{sfc}}
\end{aligned} \tag{7.14}
$$

The resulting three-enzyme coordinated capacities ($J_{\max}$, $V_{p\max}$, $V_{c\max}$) acclimate into $25^\circ\mathrm{C}$ pools on the same 15-day EMA timescale:

$$
\begin{aligned}
J_{\max,\mathrm{opt}}^{\mathrm{C4}} &= \phi_{\mathrm{PSII}} I_{\mathrm{abs}}\,\omega_{\mathrm{C4}}, \\
V_{p\max,\mathrm{opt}}^{\mathrm{C4}} &= \big(A_{l,\mathrm{opt}}^{\mathrm{C4}} + L_{\mathrm{leak}}\big)\frac{K_p + c_m}{c_m}, \\
V_{c\max,\mathrm{opt}}^{\mathrm{C4}} &= A_{l,\mathrm{opt}}^{\mathrm{C4}}\,\frac{c_{bs} + K_{m,\mathrm{C4}}}{c_{bs} - \Gamma^*_{\mathrm{C4}}}
\end{aligned} \tag{7.15}
$$

**Autonomous $\mathrm{C}_4$ dominance $F_4$ (Scott & Smith, 2022, Eq. 26; Luo et al., 2024; Lavergne et al., 2026)**: Relaxes on an ~60-day ecological turnover timescale ($0.25\alpha_{\mathrm{EMA}}$) toward the midday relative net-assimilation advantage $\Delta A_{\mathrm{rel}}$ gated by Beer–Lambert canopy-gap openness $e^{-k L}$ ($k = 0.50$) that excludes $\mathrm{C}_4$ grasses beneath closed forests:

$$
\begin{aligned}
\Delta A_{\mathrm{rel}} &= \frac{A_{n,\mathrm{C4}}^{\mathrm{mid}} - A_{n,\mathrm{C3}}^{\mathrm{mid}}}{|A_{n,\mathrm{C3}}^{\mathrm{mid}}| + 1.0}, \\
F_{4,\mathrm{tgt}} &= e^{-k L}\,\frac{1}{1 + \exp(-8\,\Delta A_{\mathrm{rel}})}
\end{aligned} \tag{7.16}
$$

### 7.4 Sub-Daily Two-Leaf Gas Exchange, Liebig $\mathrm{C}_3/\mathrm{C}_4$ Co-Limitation & Weighted Mixing, Leaf Boundary Layer & Soil Water Stress $\beta_{\mathrm{opt}}$

At sub-daily time steps, acclimated stomatal sensitivity $\xi(T_g)$ and quantum yield $\phi_{\mathrm{PSII}}(T_g)$ remain constant over the diurnal cycle (Schneider et al., 2025). On each leaf class $f\in\{\mathrm{sun}, \mathrm{sh}\}$, active capacities $(V_c, J_{\max}, V_p)$ scale with soil water stress $\beta_{\mathrm{opt}}$ and peaked Arrhenius temperature responses, and electron transport rates $(J_{\mathrm{C3}}, J_{\mathrm{C4}})$ solve the non-rectangular hyperbola ($\theta = 0.85$).

Because two-leaf sunlit/shaded partitioning and MAML subgrid heterogeneity naturally produce smooth canopy-scale transitions between light- and carboxylation-limited regimes, AOBA avoids the systematic coordination-point attenuation introduced by single-leaf empirical hyperbolic smoothing (roughly $-12\%$ for $\mathrm{C}_3$ and $-29\text{–}41\%$ for two-stage $\mathrm{C}_4$) and applies exact Liebig minimum co-limitation (Farquhar et al., 1980; von Caemmerer, 2000) to both $\mathrm{C}_3$ and $\mathrm{C}_4$ assimilation:

$$
\begin{aligned}
A_{c,\mathrm{C3}} &= V_{c,\mathrm{C3}}\,\max\!\left(\frac{c_{i,\mathrm{C3}} - \Gamma^*_{\mathrm{C3}}}{c_{i,\mathrm{C3}} + K_{m,\mathrm{C3}}},\ 0\right), \qquad A_{j,\mathrm{C3}} = \frac{J_{\mathrm{C3}}}{4}\,\max\!\left(\frac{c_{i,\mathrm{C3}} - \Gamma^*_{\mathrm{C3}}}{c_{i,\mathrm{C3}} + 2\Gamma^*_{\mathrm{C3}}},\ 0\right), \\
A_{\mathrm{C3}} &= \min\!\big(A_{c,\mathrm{C3}},\ A_{j,\mathrm{C3}}\big)
\end{aligned} \tag{7.17}
$$

$$
\begin{aligned}
A_{c,\mathrm{C4}} &= V_{c,\mathrm{C4}}, \qquad A_{l,\mathrm{C4}} = 0.20(1 - \phi_L)\,J_{\mathrm{C4}}\,g_{\mathrm{chill}}(T_l), \\
A_{p,\mathrm{C4}} &= (1 - \phi_L)\,V_{p,\mathrm{C4}}\,\frac{c_{m,\mathrm{C4}}}{\max(c_{m,\mathrm{C4}} + K_p,\ 0.5)}, \\
A_{\mathrm{C4}} &= \min\!\big(A_{c,\mathrm{C4}},\ A_{l,\mathrm{C4}},\ A_{p,\mathrm{C4}}\big)
\end{aligned} \tag{7.18}
$$

where $T_l$ is sunlit or shaded leaf temperature. Eqs. (7.17)–(7.18) recover an exact $1.000$ ratio between midday instantaneous assimilation and the acclimation coordination targets of Eqs. (7.11) and (7.13). Pathway-specific molar stomatal conductances and $F_4$-weighted mixing of gross assimilation $A_{\mathrm{mix}}$, leaf dark respiration $R_{d,\mathrm{mix}}$, and stomatal conductance $g_{sw,\mathrm{mol}}$ (with minimum cuticular floor $g_0 = 0.002\,\beta_{\mathrm{opt}}\ \mathrm{mol\ m^{-2}\ s^{-1}}$) are:

$$
\begin{aligned}
g_{sw,3} &= \frac{1.6\,\max(A_{\mathrm{C3}} - R_{d,\mathrm{C3}},\ 0)\times 10^{-6}}{\max\!\big((c_s/p_{\mathrm{sfc}})(1 - \chi_{\mathrm{C3}}),\ 10^{-6}\big)}, \\
g_{sw,4} &= \frac{1.6\,\max(A_{\mathrm{C4}} - R_{d,\mathrm{C4}},\ 0)\times 10^{-6}}{\max\!\big((c_s/p_{\mathrm{sfc}})(1 - \chi_{m,\mathrm{C4}}),\ 10^{-6}\big)}, \\
A_{\mathrm{mix}} &= (1 - F_4) A_{\mathrm{C3}} + F_4 A_{\mathrm{C4}}, \\
R_{d,\mathrm{mix}} &= (1 - F_4) R_{d,\mathrm{C3}} + F_4 R_{d,\mathrm{C4}}, \\
g_{sw,\mathrm{mol}} &= (1 - F_4) g_{sw,3} + F_4 g_{sw,4} + g_0
\end{aligned} \tag{7.19}
$$

Two fixed-point passes couple the leaf-surface microclimate $(c_s, D_s)$ to finite leaf boundary-layer conductance $g_{bw}$ ($g_{bc} = g_{bw}/1.37$; Dai et al., 2004, Eqs. 11, 45; Bonan et al., 2018) and convert molar conductance to kinematic units $g_{sw,\mathrm{ms}}$:

$$
\begin{aligned}
c_s &= \operatorname{clamp}\!\left(c_a - \frac{\max(A_{\mathrm{mix}} - R_{d,\mathrm{mix}},\ 0)\times 10^{-6}}{g_{bc}}\,p_{\mathrm{sfc}},\ 0.35\,c_a,\ c_a\right), \\
D_s &= \max\!\left(\frac{g_{bw}}{g_{bw} + g_{sw,\mathrm{mol}}}\,D_{\mathrm{af}},\ 20.0\right), \\
g_{sw,\mathrm{ms}} &= g_{sw,\mathrm{mol}}\,\frac{R^* T_l}{p_{\mathrm{sfc}}}
\end{aligned} \tag{7.20}
$$

Stomatal soil-water stress $\beta_{\mathrm{opt}}\in[0,1]$ follows time-integrated optimal stomatal theory (Holtzman et al., 2024, Eqs. 2, 7; Lu, Duursma & Medlyn, 2016) from climatological maximum growing-season dry-spell duration $D_{\max}$, 15-day mean potential ET $E_{\mathrm{pot}}$, root-zone capacity $S_R$, and available root-zone water $W_{\mathrm{avail}}$ ($w_c = 0.05 S_R$):

$$
\begin{aligned}
\tau &= 0.85\,D_{\max} + 6.59\ [\mathrm{days}], \\
\beta_{\mathrm{opt}} &= \operatorname{clamp}\!\left(\sqrt{\frac{\max(W_{\mathrm{avail}} - w_c,\ 0)}{\max(\tau E_{\mathrm{pot}},\ 0.10 S_R)}},\ 0,\ 1\right)
\end{aligned} \tag{7.21}
$$

### 7.5 Prognostic EEO LAI (Lambert-$W_0$ Derivation), Evergreen–Deciduous Habit, $S_R$ & Canopy Height $h_c$

Daily `update_daily_lai_root_height` evaluates full-canopy ($\mathrm{fAPAR} = 1$) potential daily GPP $A_{0,\mathrm{day}}$ (with Körner, 2012 $5^\circ\mathrm{C}$ cambial growth gate) and moisture-gated $A_{0,\mathrm{fav}}$, accumulating annual EMAs ($A_{0,\mathrm{yr}}, A_{0,\mathrm{fav,yr}}, P_{\mathrm{yr}}, \mathrm{PET}_{\mathrm{yr}}, f_{\mathrm{gsl}}$):

$$
\begin{aligned}
A_{0,\mathrm{day}} &= \mathrm{LUE}_{\mathrm{C3}}\,\mathrm{PPFD}_{\mathrm{day}}\,\frac{1}{1 + \exp\!\big(-0.5(T_C - 5.0)\big)}, \\
A_{0,\mathrm{fav}} &= A_{0,\mathrm{day}}\,\beta_{\mathrm{opt}}
\end{aligned} \tag{7.22}
$$

**1. Annual Ceiling $\mathrm{fAPAR}_{\max}$ and $L_{\max}$ (Cai et al., 2025, Eqs. 1–2, 12; Zhou et al., 2025, Eq. 6)**: From aridity index $\mathrm{AI} = \mathrm{PET}_{\mathrm{yr}}/P_{\mathrm{yr}}$ and usable precipitation fraction $f_0(\mathrm{AI})$, water-limited $\mathrm{fAPAR}_w$ and energy-limited $\mathrm{fAPAR}_c$ (solving marginal profit $\partial[A_{0,\mathrm{yr}}(1 - e^{-k L}) - z_{\mathrm{eff}} L]/\partial L = 0$) combine via smooth log-sum-exp minimum ($K_{\mathrm{lse}} = 10$):

$$
\begin{aligned}
\mathrm{AI} &= \frac{\mathrm{PET}_{\mathrm{yr}}}{\max(P_{\mathrm{yr}},\ 1.0)}, \\
f_0(\mathrm{AI}) &= 0.65\exp\!\left(-0.604169\left[\ln\frac{\max(\mathrm{AI},\ 1.9)}{1.9}\right]^2\right), \\
\mathrm{fAPAR}_w &= \frac{f_0(\mathrm{AI})\,P_{\mathrm{mol,yr}}\,c_a(1 - \chi_{\mathrm{C3}})}{1.6\,D\,A_{0,\mathrm{yr}}}, \qquad \mathrm{fAPAR}_c = 1 - \frac{z_{\mathrm{eff}}}{k A_{0,\mathrm{yr}}}
\end{aligned} \tag{7.23}
$$

$$
\begin{aligned}
\mathrm{fAPAR}_{\max} &= \operatorname{clamp}\!\Big(\operatorname{lsemin}_{K=10}(\mathrm{fAPAR}_w,\ \mathrm{fAPAR}_c),\ 0,\ 0.96\Big), \\
L_{\max} &= -\frac{1}{k}\ln(1 - \mathrm{fAPAR}_{\max})
\end{aligned} \tag{7.24}
$$

**2. Seasonal Steady-State Deciduous LAI $L_s$ & Lambert-$W_0$ Derivation (Zhou et al., 2025, Eqs. 15–20)**: Seasonal leaf-area carbon balance equates daily leaf production—daily assimilation $A_{0,\mathrm{fav}}(1 - e^{-k L_s})$ times allocation efficiency $m_{\mathrm{alloc}}$ ($\sigma_{\mathrm{Zhou}} = 0.771$)—to foliage turnover:

$$
\begin{aligned}
m_{\mathrm{alloc}} &= \frac{\sigma_{\mathrm{Zhou}}\,\mathrm{GSL}\,L_{\max}}{A_{0,\mathrm{fav,yr}}\,\mathrm{fAPAR}_{\max}}, \qquad \mu = m_{\mathrm{alloc}}\,A_{0,\mathrm{day}}\,\beta_{\mathrm{opt}}, \\
L_s &= \mu\left(1 - e^{-k L_s}\right)
\end{aligned} \tag{7.25}
$$

Rewriting Eq. (7.25) as $k(L_s - \mu) = -k\mu e^{-k L_s} = -k\mu e^{-k\mu} e^{-k(L_s - \mu)}$, i.e., $[-k(L_s - \mu)]e^{-k(L_s - \mu)} = -k\mu e^{-k\mu}$, yields the exact closed-form solution via the principal Lambert-$W$ branch $W_0$ ($W_0(x)e^{W_0(x)} = x$, evaluated by five cubic Halley iterations):

$$
L_s = \min\!\left(\mu + \frac{1}{k} W_0\!\left(-k\mu e^{-k\mu}\right),\ L_{\max}\right) \tag{7.26}
$$

For $u = k\mu \le 1$, $W_0(-u e^{-u}) = -u$ gives $L_s \equiv 0$ (dormancy/arid threshold), whereas for $u > 1$, $W_0(-u e^{-u}) \in (-1, 0) > -u$ produces a smooth positive leaf area $L_s > 0$.

**3. Autonomous Evergreen–Deciduous Habit $f_{\mathrm{evg}}$ & Nitrogen/Cold-Coupled Cost $z_{\mathrm{eff}}$ (Kikuzawa, 1991; Givnish, 2002; Sakai & Larcher, 1987; Wang et al., 2023)**: Relative to deciduous cost $c_{\mathrm{dec}} = 12.227\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$, climatological minimum annual temperature $T_{\min,\mathrm{yr}}$ sets cold-hardening factor $f_{\mathrm{cold}}$, evergreen leaf cost $c_{\mathrm{evg}}$, needle efficiency $\eta_{\mathrm{evg}}$, and severe-frost damage $\mathrm{frost}_{\mathrm{pen}}$ below $-40^\circ\mathrm{C}$:

$$
\begin{aligned}
f_{\mathrm{cold}} &= \sigma\!\left(\frac{273.15 - T_{\min,\mathrm{yr}}}{5.0}\right), \\
c_{\mathrm{dec,eff}} &= c_{\mathrm{dec}}\big[1 + 0.80(1 - \phi_N)\big], \qquad c_{\mathrm{evg}} = \big[1.35 + 0.55\,f_{\mathrm{cold}} + 0.80(1 - \phi_N)\big]\,c_{\mathrm{dec}}, \qquad \eta_{\mathrm{evg}} = 1 - 0.14\,f_{\mathrm{cold}}, \\
\mathrm{frost}_{\mathrm{pen}} &= 18.0\,\sigma\!\left(\frac{233.15 - T_{\min,\mathrm{yr}}}{4.0}\right)
\end{aligned} \tag{7.27}
$$

Kikuzawa (1991, Eq. 5) optimal evergreen leaf lifespan $\tau_{\mathrm{evg}}$ ($b = 15\ \mathrm{yr}$), off-season maintenance respiration $f_{\mathrm{resp,off}}$ ($Q_{10}\approx 2$), and 25-day deciduous leaf-out lag determine annual net carbon gains $(g_{\mathrm{evg}}, g_{\mathrm{dec}})$ and target evergreen fraction $f_{\mathrm{evg,tgt}}$:

$$
\begin{aligned}
\tau_{\mathrm{evg}} &= \operatorname{clamp}\!\left(\sqrt{\frac{2 b\, c_{\mathrm{evg}}}{\eta_{\mathrm{evg}} a_{\mathrm{leaf}}}},\ 1,\ 8\right), \\
g_{\mathrm{evg}} &= \eta_{\mathrm{evg}}\!\left(1 - \frac{\tau_{\mathrm{evg}}}{2b}\right)a_{\mathrm{leaf}} - \frac{c_{\mathrm{evg}}}{\tau_{\mathrm{evg}}} - 32\,f_{\mathrm{resp,off}}(1 - f_{\mathrm{gsl}}) - \mathrm{frost}_{\mathrm{pen}}, \\
g_{\mathrm{dec}} &= \left(1 - \frac{25}{365.2425 f_{\mathrm{gsl}}}\right)\!\left(1 - \frac{f_{\mathrm{gsl}}}{2 b_{\mathrm{dec}}}\right)a_{\mathrm{leaf}} - c_{\mathrm{dec,eff}}, \\
f_{\mathrm{evg,tgt}} &= \sigma\!\left(\frac{4(g_{\mathrm{evg}} - g_{\mathrm{dec}})}{c_{\mathrm{dec}}}\right)
\end{aligned} \tag{7.28}
$$

Amortized evergreen cost, short-season tundra penalty, and nitrogen sufficiency $\phi_N \in [0.25, 1.0]$ from the terrestrial CN model (Section 7.8; Stocker & Prentice, 2024) form the effective leaf cost $z_{\mathrm{eff}}$, prognosing total $L = L_{\mathrm{evg}} + L_{\mathrm{dec}}$ across dual timescales ($\tau_{\mathrm{evg}}$ with $-40^\circ\mathrm{C}$ acute frost-kill for $L_{\mathrm{evg}}$; 15-day EMA for $L_{\mathrm{dec}}$):

$$
z_{\mathrm{eff}} = \max\!\left((1 - f_{\mathrm{evg}})c_{\mathrm{dec}} + f_{\mathrm{evg}}\frac{c_{\mathrm{evg}}}{\tau_{\mathrm{evg}}},\ 0.55\,c_{\mathrm{dec}}\right)\Big[1 + 1.5\,\sigma\big(20(0.28 - f_{\mathrm{gsl}})\big)\Big]\Big[1 + 0.80(1 - \phi_N)\Big] \tag{7.29}
$$

**4. Optimal Drought-Bridging Root-Zone Storage $S_R$ (Gao et al., 2014; Davis et al., 2017)**: Accumulates seasonal cumulative water deficit $D_{\mathrm{cum}}$ whenever canopy evaporative demand exceeds precipitation and updates $S_R$ toward $S_{R,\mathrm{tgt}}$ with a $10\times$ faster asymmetric expansion ratchet under drying ($10\alpha_{\mathrm{yr}}$ vs. $\alpha_{\mathrm{yr}}$):

$$
\begin{aligned}
D_{\mathrm{cum}}^{n+1} &= \operatorname{clamp}\!\big(D_{\mathrm{cum}}^n + \mathrm{fAPAR}_{\max}\mathrm{PET}_{\mathrm{day}} - P_{\mathrm{day}},\ 0,\ 800\big)\ [\mathrm{mm}], \\
S_{R,\mathrm{tgt}} &= \operatorname{clamp}\!\big(50 + 1.35\,D_{\mathrm{cum}},\ 50,\ 650\big)\ [\mathrm{kg\ m^{-2}}]
\end{aligned} \tag{7.30}
$$

**5. Available Water-Balance Index, Productivity Index & Thermal Treeline Canopy Height $h_c$ (Kempes et al., 2011; Körner, 2012; Paulsen & Körner, 2014)**: Potential maximum tree height reflects the balance of hydraulic water supply (available water-balance index $\mathrm{MI}$, the nondimensional precipitation-to-PET ratio saturating at $1.2$), annual carbon available for stem wood construction (canopy productivity index $\mathrm{PI}$ normalized by $180\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$), and the thermal xylogenesis treeline threshold ($\ge 94$ days above $5^\circ\mathrm{C}$, $f_{\mathrm{gsl}} \approx 0.30$), updated via a 1-year EMA:

$$
\begin{aligned}
\mathrm{MI} &= \operatorname{clamp}\!\left(\frac{P_{\mathrm{yr}}}{\max(\mathrm{PET}_{\mathrm{yr}},\ 1.0)},\ 0,\ 1.2\right), \\
\mathrm{PI} &= \operatorname{clamp}\!\left(\frac{\mathrm{fAPAR}_{\max}\,A_{0,\mathrm{yr}}}{180.0},\ 0,\ 2.0\right), \\
h_{c,\mathrm{tgt}} &= \operatorname{clamp}\!\Big(34.0\,\mathrm{MI}^{0.85}\,\mathrm{PI}^{0.75}\,\sigma\big(30(f_{\mathrm{gsl}} - 0.30)\big),\ 0.05,\ 40.0\Big)\ [\mathrm{m}]
\end{aligned} \tag{7.31}
$$

### 7.6 Three-Source Energy Balance, Hysteretic Snow/SNICAR & 5-Layer Freeze–Thaw Richards Soil (`src/aoba/surface.py`)

**Snow Cover, Canopy Burial, Snow/Soil Albedos & SNICAR Darkening**: Following Swenson & Lawrence (2012, Eqs. 4–6), seasonal peak SWE $S_{n,\max}$ and subgrid topographic roughness $\sigma_{\mathrm{topo}}$ define melt-shape factor $N_{\mathrm{melt}}$ and hysteretic snow cover $A_{\mathrm{sn}}$, while MATSIRO6 (2021, Eqs. 2.1–2.3) determines snow-buried exposed canopy height $h_{\mathrm{exp}}$ and exposed LAI $L_{\mathrm{exp}}$ ($d_{\mathrm{sn}} = S_n/\rho_{\mathrm{sn}}$, $\rho_{\mathrm{sn}} = 250\ \mathrm{kg\ m^{-3}}$):

$$
\begin{aligned}
N_{\mathrm{melt}} &= \frac{200}{\operatorname{clamp}(\sigma_{\mathrm{topo}},\ 10,\ 1000)}, \\
A_{\mathrm{sn}} &= \min\!\left(\tanh\frac{S_n}{25},\ 1 - \left[\frac{\arccos(2 S_n/S_{n,\max} - 1)}{\pi}\right]^{N_{\mathrm{melt}}}\right), \\
h_{\mathrm{exp}} &= (1 - A_{\mathrm{sn}})h_c + A_{\mathrm{sn}}\max(h_c - d_{\mathrm{sn}},\ 0), \qquad L_{\mathrm{exp}} = L\frac{h_{\mathrm{exp}}}{\max(h_c,\ 0.05)}
\end{aligned} \tag{7.32}
$$

Snow albedo evolves with Yang et al. (1997) / MATSIRO6 snow age $A_g$ ($\tau_{\mathrm{age}} = 10^6\ \mathrm{s}$, zenith factor $0.4(1-\mu_0)^5$) across VIS ($0.90\to 0.65$) and NIR ($0.70\to 0.35$). When `enable_snow_bc_darkening = true`, top-2-cm BC and dust masses $(C_{\mathrm{BC}}, C_{\mathrm{dust}})$ (refreshed by $15\ \mathrm{kg\ m^{-2}}$ snowfall and 45-day washout) apply SNICAR visible darkening $d_{\mathrm{frac}}$ (Flanner et al., 2007), while bare-soil albedo darkens smoothly with surface saturation $s_0$ (CLM4.5 Sec. 3.2):

$$
\begin{aligned}
d_{\mathrm{frac}} &= \operatorname{clamp}\!\big(180\sqrt{C_{\mathrm{BC}}} + 3.5\sqrt{C_{\mathrm{dust}}},\ 0,\ 0.45\big), \\
\alpha_{\mathrm{soil,vis}} &= 0.14 + 0.14(1 - s_0)^{1.5}, \qquad \alpha_{\mathrm{soil,nir}} = 0.28 + 0.22(1 - s_0)^{1.5}
\end{aligned} \tag{7.33}
$$

**CLM5 Canopy Interception, Dry Surface Layer (DSL) Resistance & $3\times 3$ Newton Energy Balance**: Canopy interception $f_{\mathrm{int}}$, maximum canopy water storage $W_{c,\max}$, wetted leaf fraction $f_{\mathrm{wet}}$ (CLM5 Eqs. 7.4, 7.12, 7.23), Kelvin soil-pore humidity $h_{\mathrm{soil}}$, and Swenson & Lawrence (2014; CLM5 Eqs. 5.72–5.81) DSL soil resistance $r_{\mathrm{soil}}$ ($d_{\mathrm{sl}} \le 15\ \mathrm{mm}$, tortuosity $\tau_{\mathrm{soil}}$, vapor diffusivity $D_{\mathrm{vap}}(T)$) are:

$$
\begin{aligned}
f_{\mathrm{int}} &= \tanh(L_{\mathrm{exp}}), \qquad W_{c,\max} = 0.10\,L_{\mathrm{exp}}, \qquad f_{\mathrm{wet}} = \left(\frac{W_c}{W_{c,\max}}\right)^{2/3}, \\
h_{\mathrm{soil}} &= \exp\!\left(\frac{\psi_0 g}{R_v T_{s,0}}\right), \qquad r_{\mathrm{soil}} = \frac{d_{\mathrm{sl}}}{D_{\mathrm{vap}}(T_{s,0})\,\tau_{\mathrm{soil}}}
\end{aligned} \tag{7.34}
$$

Under-canopy aerodynamic resistance $r_{d,\mathrm{uc}}$ and leaf boundary-layer conductances $(g_{b,\mathrm{sun}}, g_{b,\mathrm{sh}})$ couple sunlit leaf ($T_{\mathrm{sun}}$), shaded leaf ($T_{\mathrm{sh}}$), and ground ($T_{\mathrm{gnd}}$) temperatures with canopy air $(T_{\mathrm{af}}, q_{\mathrm{af}})$, stomatal conductance, and emergent canopy biomass plus canopy-airspace heat capacity $C_{\mathrm{can}}(L_{\mathrm{exp}}, h_{\mathrm{exp}})$ (Swenson et al., 2019; Bonan, 2019) via a $3\times 3$ cofactor-Jacobian implicit Newton solver (`LAND_EB_ITERATIONS = 3`):

$$
\begin{aligned}
C_{\mathrm{can}} &= 4200\,L_{\mathrm{exp}} + 5500\,h_{\mathrm{exp}}, \qquad c_i = \frac{f_i\,C_{\mathrm{can}}}{\Delta t} \quad (i \in \{\mathrm{sun}, \mathrm{sh}\}), \quad c_{\mathrm{gnd}} = 0, \\
R_{n,i} &- H_i(T_i, T_{\mathrm{af}}) - \lambda_i E_i(T_i, q_{\mathrm{af}}) - c_i(T_i - T_i^n) - \delta_{i,2} G_0(T_{\mathrm{gnd}}, T_{s,0}) = 0, \qquad i \in \{\mathrm{sun}, \mathrm{sh}, \mathrm{gnd}\}
\end{aligned} \tag{7.35}
$$

To eliminate $2\Delta t$ explicit surface–atmosphere sensible-heat oscillations over tall forest canopies coupled to a thin lowest atmospheric layer ($\Delta z_1 = 60\ \mathrm{m}$), the aerodynamic bulk transfer coefficient $C_x$ ($x \in \{h, m\}$) passes through a $C^\infty$-smooth quadratic Kalnay–Kanamitsu (1988) implicit lowest-layer conductance limiter (`_implicit_coupled_transfer`, $\alpha_{\max} = 1.50$):

$$
\alpha = \frac{C_x\,\Delta t}{\Delta z_1}, \qquad C_{x,\mathrm{eff}} = \frac{C_x}{\sqrt{1 + (\alpha / \alpha_{\max})^2}} \tag{7.35b}
$$

The canopy heat storage $S_{\mathrm{can}} = c_{\mathrm{sun}}(T_{\mathrm{sun}}^{n+1} - T_{\mathrm{sun}}^n) + c_{\mathrm{sh}}(T_{\mathrm{sh}}^{n+1} - T_{\mathrm{sh}}^n)$ is subtracted from the net surface energy residual when forming the soil heat flux $G_{\mathrm{soil}} = \mathrm{SW}_{\mathrm{abs}} + \mathrm{LW}_\downarrow - \mathrm{LW}_\uparrow - \mathrm{SHF} - \mathrm{LHF} - Q_{\mathrm{melt}} - S_{\mathrm{can}}$, ensuring exact land–canopy–soil energy conservation. During snowmelt, $T_{\mathrm{gnd}}$ is clamped at $273.15\ \mathrm{K}$ with excess energy routed conservatively to melt; snowpack exceeding $S_{\max} = 1000\ \mathrm{kg\ m^{-2}}$ calves as glacial runoff while recording climatic surface mass balance $\mathrm{SMB}$.

**Dynamic Root Depth, 5-Layer Implicit Richards Hydrology & Enthalpy-Conserving Freeze–Thaw Conduction**: Across 5 universal-loam layers ($0.05, 0.15, 0.40, 0.80, 1.60\ \mathrm{m}$; $w_{\mathrm{sat}} = 0.43$, $\psi_s = -0.20\ \mathrm{m}$, $b = 5.39$, $K_{\mathrm{sat}} = 7.0\times 10^{-6}\ \mathrm{m\ s^{-1}}$), dynamic rooting depth $z_{\mathrm{root}}$, ice-impedance factor $f_{\mathrm{ice}}$ (Swenson et al., 2012; CLM5 Eq. 7.89), and saturated/frozen surface-runoff fraction $f_{\mathrm{sat}}$ are:

$$
\begin{aligned}
z_{\mathrm{root}} &= \operatorname{clamp}\!\left(\frac{S_R}{\rho_w(\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}})},\ 0.15,\ 3.0\right)\ [\mathrm{m}], \\
f_{\mathrm{ice}} &= 10^{-6\,w_i/w_{\mathrm{sat}}}, \qquad f_{\mathrm{sat}} = (1 - f_{\mathrm{ice},0}) + f_{\mathrm{ice},0}\,s_0^4
\end{aligned} \tag{7.36}
$$

Linearized implicit tridiagonal Richards transport with bottom gravitational drainage ($q_{\mathrm{drain}} = K_{\mathrm{bot}}$) advances liquid soil moisture:

$$
\begin{aligned}
\frac{\partial \theta_\ell}{\partial t} &= \frac{\partial}{\partial z}\left[f_{\mathrm{ice}} K_{\mathrm{sat}}\left(\frac{\theta_\ell}{w_{\mathrm{sat}}}\right)^{2b+3}\left(\frac{\partial \psi(\theta_\ell)}{\partial z} + 1\right)\right] - S_{\mathrm{root}}, \\
\psi(\theta_\ell) &= \psi_s\left(\frac{\theta_\ell}{w_{\mathrm{sat}}}\right)^{-b}
\end{aligned} \tag{7.37}
$$

Soil heat conduction conserves layer internal energy $U_k$ across the $0^\circ\mathrm{C}$ zero-curtain phase transition (SCALE-RM Sec. 8.7.1; MATSIRO6 Ch. 10) before integrating implicit 5-layer Fourier diffusion with moisture/ice-dependent thermal conductivity $\kappa_{s,k}$:

$$
\begin{aligned}
C_{\mathrm{vol},k} &= 1.26\times 10^6 + c_\ell\rho_w w_{\ell,k} + c_i\rho_i w_{i,k}, \\
U_k &= C_{\mathrm{vol},k}(T_{s,k} - T_0) - L_{f0}\rho_i w_{i,k}, \\
\kappa_{s,k} &= 0.30\left[1 + 5.5\tanh\!\left(\frac{w_{\ell,k} + w_{i,k}}{0.25}\right)\right] + 0.8\,w_{i,k}
\end{aligned} \tag{7.38}
$$

### 7.7 MAML Subgrid Orography & Dynamic Wetlands/Lakes (`src/aoba/maml_land_subgrid.py`)

When `enable_maml_subgrid_topo = true`, `Earth2014` standard deviation $\sigma_{\mathrm{topo}}$ assigns zero-mean elevation offsets $\Delta z_m \in [-1800, 1800]\ \mathrm{m}$ and alternating aspect solar multipliers $s_{\mathrm{sw},m}$ across the $N_{\mathrm{sub}} = N_y N_x$ MAML tiles, applying lapse-rate ($\Gamma_m = 6.5\ \mathrm{K\ km^{-1}}$) and barometric microclimate downscaling (Fiddes & Gruber, 2014):

$$
\begin{aligned}
T_{\mathrm{air},m} &= T_{\mathrm{air}} - \Gamma_m\Delta z_m, \qquad p_{\mathrm{sfc},m} = p_{\mathrm{sfc}}\exp\!\left(-\frac{g\Delta z_m}{R_d T_{\mathrm{air}}}\right), \\
s_{\mathrm{sw},m} &= 1 \pm 0.12\min\!\left(\frac{\sigma_{\mathrm{topo}}}{400},\ 1\right), \qquad \sum_{m=1}^{N_{\mathrm{sub}}}\Delta z_m = 0
\end{aligned} \tag{7.39}
$$

When `enable_maml_wetlands_lakes = true`, dynamic inundated fraction $f_{\mathrm{wet}} \in [0, 0.85]$ relaxes exponentially ($\tau_{\mathrm{wet}} = 10\ \mathrm{d}$) toward a TOPMODEL (Beven & Kirkby, 1979) target combining topographic flatness $f_{\mathrm{flat}}$, valley weight $w_{\mathrm{val}}$, thaw factor $f_{\mathrm{thaw}}$, soil saturation excess, and river overbank flooding $f_{\mathrm{flood}}$:

$$
\begin{aligned}
f_{\mathrm{flat}} &= \left(1 + \frac{\sigma_{\mathrm{topo}}}{80}\right)^{-1}, \qquad f_{\mathrm{flood}} = 0.35\,\frac{Q_{\mathrm{riv}}}{Q_{\mathrm{riv}} + 4000}, \\
f_{\mathrm{thaw}} &= \operatorname{clamp}\!\left(\frac{T_{s,0} - 270.15}{5.0},\ 0.15,\ 1.0\right), \\
f_{\mathrm{wet,tgt}} &= \operatorname{clamp}\!\left(f_{\mathrm{flat}}\,w_{\mathrm{val}}\,f_{\mathrm{thaw}}\left(\left[\frac{\max(s_0 - 0.58,\ 0)}{0.42}\right]^{1.6} + f_{\mathrm{flood}}\right),\ 0,\ 0.85\right)
\end{aligned} \tag{7.40}
$$

### 7.8 Coupled Terrestrial Carbon–Nitrogen Cycle, 5-Layer Permafrost C/N & MAML SPITFIRE Wildfire (`src/aoba/land_carbon.py`)

Six carbon pools (leaf $C_l$, wood $C_w$, fine root $C_r$, fast litter $C_{lf}$, slow litter $C_{ls}$, 5-layer soil $C_{s,0..4}$) and seven nitrogen pools (organic $N_l, N_w, N_r, N_{lf}, N_{ls}, N_{s,0..4}$ plus mineral $N_{\min}$) advance conservatively with Neumaier/Kahan summation:

$$
\begin{aligned}
\frac{dC_{\mathrm{tot}}}{dt} &= \mathrm{GPP} - R_{a,\mathrm{eff}} - R_h - F_{\mathrm{fire,C}} \equiv \mathrm{NBP}, \\
\frac{dN_{\mathrm{tot}}}{dt} &= F_{\mathrm{dep}} + F_{\mathrm{BNF}} - F_{\mathrm{leach}} - F_{\mathrm{denit}} - F_{\mathrm{fire,N}}
\end{aligned} \tag{7.41}
$$

1. **Autotrophic respiration & functional-balance dynamic allocation**: Combines leaf respiration, $Q_{10}=2.0$ wood/root maintenance respiration, and $25\%$ growth respiration (Clark et al., 2011). Height-dependent baseline wood allocation $f_{w0}$ and nitrogen-responsive fine-root functional balance (Stocker & Prentice, 2024) shift carbon into roots under N scarcity ($\phi_N < 1$):

$$
\begin{aligned}
f_{w0} &= 0.05 + 0.42\,\frac{h_c}{h_c + 4.0}, \\
f_{\mathrm{root}} &= \operatorname{clamp}\!\big(0.45(1 - f_{w0}) + 0.35(1 - \phi_N),\ 0.15,\ 0.72\big)
\end{aligned} \tag{7.42}
$$

2. **Stocker & Prentice (2024) CN-Model & FUN/Houlton BNF**: Optimal foliar N demand follows Dong et al. (2022) from $V_{c\max 25}$ and emergent leaf mass per area $\mathrm{LMA}$ ($\mathrm{CN}_{l,\mathrm{opt}} = \operatorname{clamp}(\mathrm{LMA}/N_{\mathrm{area,opt}}, 15, 65)$; fixed root $45$, wood $250$, soil $14$). After internal resorption ($50\%$ leaf, $25\%$ root), soil $N_{\min}$ uptake combines transpiration mass flow $u_{\mathrm{mass}}$ and fine-root diffusion $u_{\mathrm{diff}}$:

$$
\begin{aligned}
\mathrm{LMA} &= 0.045 + 0.055\,f_{\mathrm{evg}} + 0.025\,f_{\mathrm{cold}} + 0.035(1 - \phi_N)\ [\mathrm{kg\ C\ m^{-2}}], \qquad \tau_{\mathrm{leaf,eff}} = \tau_{\mathrm{leaf},0}\big[1 + 0.80(1 - \phi_N)\big], \\
N_{\mathrm{area,opt}} &= 1.76\times 10^{-5}\,V_{c\max 25} + 0.016\,\mathrm{LMA}, \\
u_{\mathrm{mass}} &= \Big[1 - \exp\!\left(-\frac{E_{\mathrm{tr}}\Delta t}{W_{\mathrm{root,col}}}\right)\Big]N_{\min}, \\
k_{\mathrm{diff}} &= 8.0\,f_{\mathrm{moist}}\,f_{\mathrm{freeze}}\,\frac{C_r}{C_r + 0.25}, \qquad u_{\mathrm{diff}} = \big(1 - e^{-k_{\mathrm{diff}}\Delta t_{\mathrm{yr}}}\big)(N_{\min} - u_{\mathrm{mass}})
\end{aligned} \tag{7.43}
$$

Remaining demand invests up to $2.2\%$ of positive NPP in symbiotic BNF at the Houlton et al. (2008) / Fisher et al. (2010, FUN Eq. 6) temperature-dependent carbon cost (minimum $\sim 6.3\ \mathrm{kg\ C / kg\ N}$ at $25.15^\circ\mathrm{C}$, rising steeply to $\sim 187$ at $0^\circ\mathrm{C}$, added to $R_{a,\mathrm{eff}}$), diagnosing instantaneous N sufficiency $\phi_N^{\mathrm{inst}}$:

$$
\begin{aligned}
\mathrm{cost}_{\mathrm{BNF}}(T_C) &= 5.0\,\exp\!\left(-\left[-3.62 + 0.27\,T_C\left(1 - \frac{T_C}{50.30}\right)\right]\right), \\
\phi_N^{\mathrm{inst}} &= \operatorname{clamp}\!\left(\frac{N_{\mathrm{retrans}} + u_{\mathrm{soil}} + \mathrm{BNF}_{\mathrm{sym}}}{D_{N,\mathrm{gross}}},\ 0.15,\ 1.0\right)
\end{aligned} \tag{7.44}
$$

The 15-day EMA nitrogen sufficiency $\phi_N$ feeds back into both Eq. (7.42) ($f_{\mathrm{root}}$) and Eq. (7.29) ($z_{\mathrm{eff}}$). Asymbiotic BNF, lightning $\mathrm{LNO}_x$ deposition, mineralization/immobilization, wet leaching ($0.60\ \mathrm{yr^{-1}}$), and anaerobic denitrification ($0.80\ \mathrm{yr^{-1}}$) close the $N_{\min}$ budget.

3. **Five-layer permafrost soil C/N (Koven et al., 2013)**: Layer decomposition includes depth attenuation $\exp(-z_{\mathrm{mid},k}/0.50\ \mathrm{m})$, $Q_{10}=2.0$, moisture scaling, frozen-soil permafrost suppression ($0.05$), and conservative vertical cryoturbation ($D_{\mathrm{cryo}} = 5.0\times 10^{-4}\ \mathrm{m^2\ yr^{-1}}$).

4. **Process-based MAML SPITFIRE wildfire (Thonicke et al., 2010; Rothermel, 1972; Byram, 1959; Van Wagner, 1973)**: Accumulates Nesterov dryness index $\mathrm{DI}$ (reset on rain $> 3\ \mathrm{mm\ d^{-1}}$ or $f_{\mathrm{snow}} > 0.20$) into fire danger index $\mathrm{FDI}$, partitions the surface fuel bed into 1-hour fine litter plus dry-cured $\mathrm{C}_4$ herbaceous fuel ($c_{1\mathrm{h}} = c_{\mathrm{lit,fast}} + f_{\mathrm{cure}} f_{\mathrm{C4}} c_{\mathrm{leaf}}$, $f_{\mathrm{cure}} = \operatorname{clamp}(0.25 + 0.75\,\mathrm{FDI},\ 0.25,\ 1.0)$) versus 100-hour coarse woody litter ($c_{100\mathrm{h}} = c_{\mathrm{lit,slow}}$) with differential Rothermel moisture damping $(\eta_{M}, \eta_{M,100\mathrm{h}})$, fuel-bed-dependent baseline spread rate $R_{\mathrm{base}} \in [3.2, 8.5]\ \mathrm{m\ min^{-1}}$, wind and topographic slope acceleration ($\phi_w + \phi_s$, $\phi_s = 3.5\tan^2\theta$ from subgrid `Earth2014` roughness), wind-elliptical burned area $A_{\mathrm{fire}}$, Byram fireline intensity ($I_B \ge 50\ \mathrm{kW\ m^{-1}}$ self-extinction), Van Wagner scorch height $\mathrm{SH}$, and allometric bark thickness $d_{\mathrm{bark}}$ for crown/cambial mortality $p_{\mathrm{mort}}$:

$$
\begin{aligned}
\mathrm{FDI} &= 1 - \exp(-3\times 10^{-4}\,\mathrm{DI}), \qquad R_f = \operatorname{clamp}\!\big(R_{\mathrm{base}}\,\eta_M\,(1 + \phi_w + \phi_s)\,f_{\mathrm{fuel}},\ 0,\ 60\big), \\
A_{\mathrm{fire}} &= \frac{\pi}{4 L_B}\big[(R_f + R_b)\,t_{\mathrm{fire}}\big]^2, \qquad F_c = 2.0\big(0.80\,c_{1\mathrm{h}} + 0.35\,\eta_{M,100\mathrm{h}}\,c_{100\mathrm{h}}\big), \\
I_B &= 18000\,F_c\,\frac{R_f}{60}\ [\mathrm{kW\ m^{-1}}], \qquad \mathrm{SH} = 0.11\,I_B^{2/3}\ [\mathrm{m}], \qquad d_{\mathrm{bark}} = 0.8 + 1.8\left(\frac{h_c}{10}\right)^{0.6}\ [\mathrm{cm}]
\end{aligned} \tag{7.45}
$$

Cloud-to-ground lightning (blended 50/50 with parent-cell mean for dry-lightning ignition) advances discrete MAML patch burn-hazard accumulators with 2D neighbor fire spread, emitting pyrogenic C, BC/OC smoke, and volatilized N ($85\%$ volatilized, $15\%$ returned to $N_{\min}$ as ash) while directly resetting burnt-patch $\mathrm{LAI}$ and $h_c$.

### 7.9 Priority-Flood SCVT Voronoi River Routing (`src/aoba/river_routing.py`)

Derives a globally acyclic Voronoi river network directly from `Earth2014` elevations via the Barnes, Lehman & Mulla (2014) Priority-Flood algorithm without prescribed river-direction maps. Slope-dependent kinematic velocity $v_{\mathrm{riv},i}$ (Miller et al., 1994; Mizukami et al., 2016) routes terrestrial runoff conservatively to coastal ocean cells via exact exponential linear-reservoir outflow and Kahan accumulation:

$$
\begin{aligned}
v_{\mathrm{riv},i} &= \operatorname{clamp}\!\left(0.50\sqrt{\frac{S_i}{10^{-3}}},\ 0.15,\ 2.00\right)\ [\mathrm{m\ s^{-1}}], \\
f_{\mathrm{out},i} &= -\operatorname{expm1}\!\left(-\frac{v_{\mathrm{riv},i}}{L_i}\Delta t\right), \qquad Q_{\mathrm{out},i} = f_{\mathrm{out},i}\,S_{\mathrm{riv},i}, \\
\Delta S_{\mathrm{riv},i} &= R_{\mathrm{liq},i} A_i \Delta t + \sum_{j\in\mathrm{up}(i)} Q_{\mathrm{out},j} - Q_{\mathrm{out},i}
\end{aligned} \tag{7.46}
$$

When terrestrial carbon-nitrogen cycling and marine biogeochemistry are active, the same Priority-Flood graph routes leached dissolved nitrogen $F_{\mathrm{leach},i}$, fluvial carbon export $F_{C,\mathrm{riv},i} = 0.02\,R_{h,i}$ (Regnier et al., 2013), and weathering alkalinity $F_{\mathrm{Alk},i}$ down to coastal ocean cells with Kahan-compensated channel storage $S_{X,i}$. Dissolved nitrogen undergoes temperature-dependent in-stream and riparian denitrification ($k_{\mathrm{denit},20} = 0.03\ \mathrm{d^{-1}}$, $Q_{10} = 2$; Seitzinger et al., 2002), while bioavailable estuarine iron surviving flocculation ($c_{\mathrm{Fe,est}} = 10^{-7}\ \mathrm{mol\ kg^{-1}}$; Raiswell & Canfield, 2012) scales with freshwater coastal inflow $F_{\mathrm{mouth},i}$:

$$
\begin{aligned}
f_{\mathrm{denit},i} &= -\operatorname{expm1}\!\left(-k_{\mathrm{denit},20}\,2^{(T_{\mathrm{sfc},i}-293.15)/10}\Delta t\right), \\
\Delta S_{X,i} &= F_{X,i} A_i \Delta t + \sum_{j\in\mathrm{up}(i)} f_{\mathrm{out},j}(1 - f_{\mathrm{loss},j})S_{X,j} - \big[f_{\mathrm{out},i} + (1 - f_{\mathrm{out},i})f_{\mathrm{loss},i}\big]S_{X,i}, \\
F_{\mathrm{Fe,mouth},i} &= c_{\mathrm{Fe,est}}\,F_{\mathrm{mouth},i}
\end{aligned} \tag{7.46b}
$$

where $f_{\mathrm{loss},i} = f_{\mathrm{denit},i}$ for nitrogen and $f_{\mathrm{loss},i} = 0$ for fluvial carbon and alkalinity. Coastal ocean deliveries $(F_{\mathrm{N,mouth}}, F_{\mathrm{C,mouth}}, F_{\mathrm{Alk,mouth}}, F_{\mathrm{Fe,mouth}})$ are conservatively remapped onto the ocean mesh and enter surface ocean $\mathrm{NO}_3$, $\mathrm{DIC}$, $\mathrm{Alk}$, and $\mathrm{Fe}_{\mathrm{sfc}}$ (Section 9.8).

### 7.10 Map-Free Autonomous Dynamic Lake Model (Topographic Depressions & 2-Layer FLake Thermodynamics, `src/aoba/lakes.py`, `enable_dynamic_lakes`)

Inland lakes expand, contract, freeze, overflow, or desiccate completely in response to climate and topographic evolution (e.g., Green Sahara Mega-Lake Chad, Aral Sea desiccation, proglacial Lake Agassiz) without prescribed lake-mask or depth lookup maps. Maximum geometric basin capacities $(f_{\mathrm{cap},i}, h_{\mathrm{cap},i}, V_{\mathrm{cap},i})$ emerge solely from solid-Earth `Earth2014` bedrock depressions ($z_{\mathrm{SUR}} - z_{\mathrm{BED}} > 0.5\ \mathrm{m}$ on non-ocean, non-ice pixels yielding $f_{\mathrm{bed},i}, V_{\mathrm{bed},i}$) combined with Barnes et al. (2014) Priority-Flood closed-depression sill depth $d_{\mathrm{pf},i} = \max(z_{\mathrm{filled},i} - z_{\mathrm{valley},i}, 0)$ evaluated on valley-floor elevation $z_{\mathrm{valley},i}$ (`z_min_land`, eliminating coarse-mesh river-canyon aliasing along the Amazon and Congo while updating dynamically under glacial isostasy):

$$
\begin{aligned}
f_{\mathrm{pf},i} &= f_{\max}\,\frac{d_{\mathrm{pf},i}}{d_{\mathrm{pf},i} + \max(\sigma_{\mathrm{topo},i},\ 50\ \mathrm{m})}, \qquad V_{\mathrm{pf},i} = \frac{1}{1.5}\,(f_{\mathrm{pf},i} A_i)\,\min(d_{\mathrm{pf},i},\ d_{\max}), \\
f_{\mathrm{cap},i} &= \min\!\big(f_{\max},\ \max(f_{\mathrm{bed},i},\ f_{\mathrm{pf},i})\big), \qquad h_{\mathrm{cap},i} = \operatorname{clamp}\!\left(\frac{\max(V_{\mathrm{bed},i}, V_{\mathrm{pf},i})}{f_{\mathrm{cap},i} A_i},\ 1.0,\ d_{\max}\right), \\
V_{\mathrm{cap},i} &= f_{\mathrm{cap},i} A_i h_{\mathrm{cap},i}
\end{aligned} \tag{7.47}
$$

with $f_{\max} = 0.85$ (`lake_max_frac`) and $d_{\max} = 400\ \mathrm{m}$ (`lake_max_depth_m`). Prognostic liquid-plus-ice water-equivalent basin volume $V_i \in [0, V_{\mathrm{cap},i}]$ (updated with Kahan compensation) maps via exact parabolic-bowl hypsometry to lake area fraction $f_{\mathrm{lake},i}$ and mean depth $h_{\mathrm{lake},i}$ ($f_{\mathrm{lake},i} = 0$ when $h_{\mathrm{lake},i} < 0.05\ \mathrm{m}$), coupled conservatively to Priority-Flood river routing (Eq. 7.46):

$$
\begin{aligned}
\phi_i &= \operatorname{clamp}\!\left(\frac{V_i}{V_{\mathrm{cap},i}},\ 0,\ 1\right), \qquad f_{\mathrm{lake},i} = f_{\mathrm{cap},i}\,\phi_i^{2/3}, \qquad h_{\mathrm{lake},i} = h_{\mathrm{cap},i}\,\phi_i^{1/3}, \\
M_{\mathrm{in},i} &= (1 - f_{\mathrm{lake},i}) R_{\mathrm{liq},i} A_i \Delta t + \sum_{j\in\mathrm{up}(i)} Q_{\mathrm{out},j}, \qquad M_{\mathrm{spill},i} = \rho_w\max\!\left(V_i + \frac{M_{\mathrm{in},i}}{\rho_w} - V_{\mathrm{cap},i},\ 0\right)
\end{aligned} \tag{7.48}
$$

satisfying $f_{\mathrm{lake},i} A_i h_{\mathrm{lake},i} \equiv V_i$ identically so both endorheic closed basins ($V_i < V_{\mathrm{cap},i} \Rightarrow M_{\mathrm{spill},i} = 0$; Caspian, Chad, Eyre) and exorheic through-flow lakes ($V_i = V_{\mathrm{cap},i} \Rightarrow M_{\mathrm{spill},i} > 0$; Great Lakes, Baikal) emerge from a single governing balance.

Lake thermodynamics follows Mironov (2008, FLake) and Hostetler & Bartlein (1990) with a two-layer liquid column (epilimnion $T_{\mathrm{ml}}, h_{\mathrm{ml}}$; hypolimnion $T_{\mathrm{bot}}, h_{\mathrm{bot}} = h_{\mathrm{liq}} - h_{\mathrm{ml}}$) and prognostic lake-ice/snow cover $(h_{\mathrm{ice}}, \mathrm{SWE}_{\mathrm{ice}}, T_{\mathrm{ice}})$. Over open water, Beer–Lambert solar penetration ($f_{\mathrm{sw,ml}} = 1 - e^{-k_{\mathrm{ext}} h_{\mathrm{ml}}}$, default $k_{\mathrm{ext}} = 0.5\ \mathrm{m^{-1}}$) and an unconditionally stable implicit backward-Euler surface heat balance (including snowfall melting $-L_f P_{\mathrm{snow}}$) advance $T_{\mathrm{ml}}$ and initiate surface ice whenever $T_{\mathrm{ml}}^* < T_f$:

$$
\begin{aligned}
F_{\mathrm{net},0} &= f_{\mathrm{sw,ml}} F_{\mathrm{sw}} + \epsilon_l F_{\mathrm{lw}\downarrow} - \epsilon_l \sigma T_{\mathrm{ml}}^4 - c_p g_a (T_{\mathrm{ml}} - T_a) - L_v g_a\big(q_{\mathrm{sat}}(T_{\mathrm{ml}}) - q_v\big) - L_f P_{\mathrm{snow}}, \\
\Delta T_{\mathrm{ml}} &= \frac{F_{\mathrm{net},0}\,\Delta t}{\rho_w c_{pw} h_{\mathrm{ml}} + \Delta t\left(4\epsilon_l\sigma T_{\mathrm{ml}}^3 + c_p g_a + L_v g_a \frac{\partial q_{\mathrm{sat}}}{\partial T}\right)}, \qquad \Delta h_{\mathrm{ice}} = \frac{\rho_w c_{pw} h_{\mathrm{ml}}\max(T_f - T_{\mathrm{ml}}^*,\ 0)}{\rho_{\mathrm{ice}} L_f}
\end{aligned} \tag{7.49}
$$

On ice-covered lakes ($h_{\mathrm{ice}} > 10^{-4}\ \mathrm{m}$), an implicit skin balance with series snow-plus-ice conductivity $k_{\mathrm{eff}}$ determines $T_{\mathrm{ice}} \le T_f$, surface snow/ice melt, basal conductive freezing, and exact exponential under-ice water melt, transitioning smoothly from Fresnel open-water albedo (Eq. 9.1) to bare ice (VIS $0.55$ / NIR $0.30$) and snow-covered ice (VIS $0.82$ / NIR $0.52$):

$$
\begin{aligned}
k_{\mathrm{eff}} &= \left(\frac{h_{\mathrm{ice}}}{k_{\mathrm{ice}}} + \frac{\mathrm{SWE}_{\mathrm{ice}}}{\rho_{\mathrm{snow}} k_{\mathrm{snow}}}\right)^{-1}, \qquad F_{\mathrm{cond}} = k_{\mathrm{eff}}(T_f - T_{\mathrm{ice}}), \\
\Delta h_{\mathrm{ice,base}} &= \frac{F_{\mathrm{cond}}\,\Delta t - \rho_w c_{pw} h_{\mathrm{ml}}(T_{\mathrm{ml}} - T_f)\big(1 - e^{-\gamma_w \Delta t / h_{\mathrm{ml}}}\big)}{\rho_{\mathrm{ice}} L_f}
\end{aligned} \tag{7.50}
$$

Using the quadratic freshwater equation of state $\rho_f(T)$ with maximum density at $T_{\mathrm{md}} = 277.13\ \mathrm{K}$ ($3.98^\circ\mathrm{C}$), instantaneous enthalpy-conserving convective overturning mixes the column whenever $\rho_f(T_{\mathrm{ml}}) \ge \rho_f(T_{\mathrm{bot}})$; under stable stratification, wind–buoyancy mixed-layer entrainment/detrainment toward $h_{\mathrm{eq}}$ and exact analytical two-layer exponential thermocline diffusion ($K_z = 2\times 10^{-5}\ \mathrm{m^2\ s^{-1}}$) conserve total column heat identically:

$$
\begin{aligned}
\rho_f(T) &= \rho_w\Big[1 - 6.8\times 10^{-6}\,(T - 277.13)^2\Big], \qquad T_{\mathrm{mix}} = \frac{h_{\mathrm{ml}} T_{\mathrm{ml}} + h_{\mathrm{bot}} T_{\mathrm{bot}}}{h_{\mathrm{ml}} + h_{\mathrm{bot}}}, \\
h_{\mathrm{eq}} &= \operatorname{clamp}\!\left(\frac{0.25\,u_{*w}}{|f| + 0.15\,N + 10^{-4}},\ \min(1.0, h_{\mathrm{liq}}),\ h_{\mathrm{liq}}\right), \qquad d_{\mathrm{fac}} = -\operatorname{expm1}\!\left(-\frac{K_z\,\Delta t}{0.5\,h_{\mathrm{ml}} h_{\mathrm{bot}}}\right)
\end{aligned} \tag{7.51}
$$

Finally, cell albedos, thermal emission, turbulent fluxes, and mineral dust emission are conservatively tiled between land $(1 - f_{\mathrm{lake},i})$ and lake $f_{\mathrm{lake},i}$ (so desiccated basins automatically expose dust sources), executed on GPUs in one fused launch via `dynamic_lake_step_col` (`src/aoba/kernels_esm.py`):

$$
\begin{aligned}
\alpha_{\mathrm{cell}} &= (1 - f_{\mathrm{lake}})\,\alpha_{\mathrm{land}} + f_{\mathrm{lake}}\,\alpha_{\mathrm{lake}}, \qquad T_{\mathrm{rad,cell}}^4 = (1 - f_{\mathrm{lake}})\,T_{\mathrm{rad,land}}^4 + f_{\mathrm{lake}}\,T_{\mathrm{rad,lake}}^4, \\
(H, \mathrm{LHF}, E, \boldsymbol{\tau})_{\mathrm{cell}} &= (1 - f_{\mathrm{lake}})\,(H, \mathrm{LHF}, E, \boldsymbol{\tau})_{\mathrm{land}} + f_{\mathrm{lake}}\,(H, \mathrm{LHF}, E, \boldsymbol{\tau})_{\mathrm{lake}}, \qquad F_{\mathrm{dust,eff}} = (1 - f_{\mathrm{lake}})\,F_{\mathrm{dust,land}}
\end{aligned} \tag{7.52}
$$

---

## 8. Global Nonhydrostatic SCVT Dynamics, Embedded CRM, Gravity-Wave Drag & SP Coupling

### 8.1 Unified Compressible Euler Equations, Hydrostatic Initialization & Held–Suarez Benchmark (`src/aoba/scvt_dynamics.py`)

Both the global SCVT host and the embedded CRM solve the identical flux-form compressible nonhydrostatic Euler equations for coupled dry mass $m = \rho_d\Delta z_c$, coupled moist potential temperature $\Theta_m = m\theta_m$, horizontal wind $\mathbf{u}_h$, and vertical velocity $w$:

$$
\begin{aligned}
\frac{\partial m}{\partial t} + \nabla_h\cdot(m\mathbf{u}_h) + \frac{\partial \Omega}{\partial \zeta} &= 0, \\
\frac{\partial \Theta_m}{\partial t} + \nabla_h\cdot(m\mathbf{u}_h\theta_m) + \frac{\partial (\Omega\theta_m)}{\partial \zeta} &= m S_{\theta_m}
\end{aligned} \tag{8.1}
$$

$$
\frac{\partial \mathbf{u}_h}{\partial t} + (\zeta + f)\hat{\mathbf{k}}\times\mathbf{u}_h + \nabla_h K + \tilde{w}\frac{\partial \mathbf{u}_h}{\partial z} = -\frac{1}{\rho_{\mathrm{tot}}}\nabla_h p + \mathbf{F}_{\mathrm{turb}} + \mathbf{F}_{\mathrm{GWD}} + \mathbf{F}_{\mathrm{sp}} + \mathcal{D}_u \tag{8.2}
$$

$$
\frac{\partial w}{\partial t} + (\mathbf{u}\cdot\nabla)w = -\frac{1}{\rho_{\mathrm{tot}}}\frac{\partial p'}{\partial z} - g\frac{\rho_{\mathrm{tot}}'}{\rho_{\mathrm{tot}}} + F_{w,\mathrm{turb}} + F_{w,\mathrm{sp}} - \gamma_r w \tag{8.3}
$$

where the terrain-following contravariant vertical mass flux $\Omega$ (with rigid boundary conditions $\Omega_0 = \Omega_{N_z} = 0$) and cell-centered kinetic energy $K_i$ are:

$$
\Omega = \rho_{dw}\big(w - \mathbf{u}_h\cdot\nabla z_w\big), \qquad K_i = \frac{1}{2 A_i}\sum_{e\in E(i)} A_e u_e^2 \tag{8.4}
$$

The initial atmosphere is constructed by discrete trapezoidal hydrostatic integration in `float64` (equatorial surface temperature $300\ \mathrm{K}$ with $35\ \mathrm{K}$ equator-to-pole contrast, $6.5\ \mathrm{K\ km^{-1}}$ tropospheric lapse rate, U.S. Standard Atmosphere 1976 stratospheric inversion, optional baroclinic wave perturbation, and exponentially decaying relative humidity). The Held & Suarez (1994, Eqs. 1–4) dry dynamical-core benchmark forcing is also built in.

### 8.2 SCVT Host RK3–HEVI Acoustic-Split Solver & Batched Zalesak FCT (`src/aoba/scvt_dynamics.py`)

At each dynamical step $\Delta t_{\mathrm{dyn}}$ (automatically sub-stepped to $\Delta t_{\mathrm{adv,host}} = \min(900, 0.40 d_{\min}/100)$), scale-selective horizontal divergence damping $\nu_{\mathrm{div}}$, gentle eigenvalue-matched fourth-order hyperdiffusion $(\nu_4, \nu_{4,\mathrm{div}}, \nu_{4,s})$ (Nyquist e-folding timescale $\sim 5\text{–}6\ \mathrm{h}$), and thresholded monotone 2D Smagorinsky eddy mixing $K_h$ (Skamarock et al., 2012, Sec. 3d; Zängl et al., 2015, Sec. 2.5) suppress grid-scale noise and runaway single-cell convective grid-point storms while preserving full effective horizontal resolution on synoptic fronts, jets, and cloud bands:

$$
\begin{aligned}
\nu_{\mathrm{div}} &= 0.02\,\frac{d_{\min}^2}{\Delta t_{\mathrm{dyn}}}, \qquad \nu_4 = 0.0002\,\frac{d_{\min}^4}{\Delta t_{\mathrm{dyn}}}, \\
\nu_{4,\mathrm{div}} &= 0.0018\,\frac{d_{\min}^4}{\Delta t_{\mathrm{dyn}}}, \qquad \nu_{4,s} = 0.0010\,\frac{d_{\min}^4}{\Delta t_{\mathrm{dyn}}}, \\
\mathcal{D}_u &= \nu_{\mathrm{div}}\nabla(\nabla\cdot\mathbf{u}) - \nu_4\nabla^4\mathbf{u} - \nu_{4,\mathrm{div}}\nabla\!\big(\nabla^2(\nabla\cdot\mathbf{u})\big), \\
\mathcal{D}_{\Theta_m} &= -\nabla\cdot\!\big(\nu_{4,s}\,\bar{m}_e\,\nabla(\nabla^2\theta_m)\big), \qquad \Delta w_{\mathrm{diff}} = -\nu_{4,s}\Delta t_{\mathrm{dyn}}\,\nabla^4 w, \\
K_{h,\mathrm{sc}} &= \min\!\left((0.90\,d_{c,e})^2\max\!\big(-(\nabla\cdot\mathbf{u})_{c_0},\ -(\nabla\cdot\mathbf{u})_{c_1},\ 0\big),\ 0.12\,\frac{d_{\min}^2}{\Delta t}\right)
\end{aligned} \tag{8.5}
$$

On horizontally converging storm edges ($\max(-(\nabla\cdot\mathbf{u})_{c_0}, -(\nabla\cdot\mathbf{u})_{c_1}) > 0$), the threshold-free, scale-invariant monotone flux-divergence operator $\nabla\cdot(\bar{m}_e K_{h,\mathrm{sc}}\nabla\phi)$ mixes $q_n$, $\theta_m$, and $w$ while $K_{h,\mathrm{div}}\nabla(\nabla\cdot\mathbf{u})$ ($C_{s,\mathrm{div}} = 1.25$) damps purely divergent storm inflow. Upper-tropospheric divergent anvils ($\nabla\cdot\mathbf{u} \ge 0$) and rotational jets experience zero Smagorinsky mixing ($K_h = 0$), preserving cloud-top sharpness, exact spherical conservation, and the discrete maximum principle without Gibbs ringing.

Within each of the three Wicker–Skamarock RK3 stages ($\Delta\tau \in \{\Delta t_{\mathrm{dyn}}/3, \Delta t_{\mathrm{dyn}}/2, \Delta t_{\mathrm{dyn}}\}$), slow tendencies comprising the nonlinear TRiSK Coriolis/vorticity term $\sum_{e'} w_{e,e'}\frac{\eta_e+\eta_{e'}}{2}u_{e'}$, kinetic-energy gradient $-\nabla K$, vertical momentum advection, and dissipation are frozen while fast acoustic sub-steps ($\Delta t_{\mathrm{ac,max}} = \min(\max(0.16 d_{\min}/350, 1), 300)\ \mathrm{s}$, vertical implicit weight $\beta_{\mathrm{HEVI}} = 0.60$) are integrated. Over steep terrain, horizontal pressure gradients apply the Klemp (2011) hydrostatic exponential factors $(f_{z0}, f_{z1})$, horizontal mass flux uses forward off-centering $U_h^{(k)}$, and explicit vertical advection of $\theta_m$ at interior interfaces $k + \tfrac{1}{2}$ uses the conservative MPAS-A third-order upwind-biased flux reconstruction ($\beta_v = 0.25$; Skamarock & Gassmann, 2011, Eq. 5; Skamarock et al., 2012, Eq. 20) to damp grid-scale $2\Delta z$ vertical computational modes without static vertical diffusion:

$$
\begin{aligned}
f_{z0} &= \exp\!\left(-\frac{\rho_{\mathrm{tot},c_0} g}{2 p_{c_0}}\Delta z_{10}\right), \qquad f_{z1} = \exp\!\left(+\frac{\rho_{\mathrm{tot},c_1} g}{2 p_{c_1}}\Delta z_{10}\right), \\
U_h^{(k)} &= \bar{m}_e\big(1.10\,u^{(k+1)} - 0.10\,u^{(k)}\big), \qquad p^{(k)} = p_s + \frac{\gamma_d p_s}{\Theta_{m,s}}\big(\Theta_m^{(k)} - \Theta_{m,s}\big), \\
\theta_{m,k+1/2}^{\mathrm{up}} &= \frac{7(\theta_{m,k} + \theta_{m,k+1}) - (\theta_{m,k-1} + \theta_{m,k+2})}{12} - \frac{\beta_v\,\operatorname{sgn}(\Omega_{k+1/2})}{12}\Big[3(\theta_{m,k+1} - \theta_{m,k}) - (\theta_{m,k+2} - \theta_{m,k-1})\Big]
\end{aligned} \tag{8.6}
$$

Vertical acoustic waves reduce via the linearized equation of state to an $(N_z - 1)$-layer tridiagonal Helmholtz system for $\delta\Omega_k$:

$$
\begin{aligned}
a_k\,\delta\Omega_{k-1} + b_k\,\delta\Omega_k + c_k\,\delta\Omega_{k+1} &= r_{\Omega,k}, \\
m^{(k+1)} &= m^\dagger - \beta_{\mathrm{HEVI}}\Delta t_{\mathrm{ac}}\Delta\delta\Omega
\end{aligned} \tag{8.7}
$$

Using the third-stage time-averaged mass fluxes $(\bar{U}_h, \bar{\Omega})$, all moisture, aerosol, chemistry, $\mathrm{CO}_2$, and water-isotope tracers are advected in a single batched Zalesak (1979) Flux-Corrected Transport pass (`fct_advect_scvt_batch`: first-order donor-cell low-order flux, second-order horizontal / fourth-order vertical high-order flux, and local-extremum limiters $C_h, C_z \in [0, 1]$) ensuring strict non-negativity and mass conservation.

### 8.3 Embedded CRM RK3–HEVI Core & Standalone RCE Driver (`src/aoba/crm.py`)

The embedded CRM (doubly periodic 2D/3D Arakawa C-grid / Charney–Phillips, configurable horizontal spacing $\Delta x = \Delta y$ via `[grid] crm_dx`, default $4\ \mathrm{km}$) integrates the same RK3–HEVI compressible Euler equations as the host. Horizontal momentum advection uses the third-order Wicker–Skamarock upwind-biased flux difference, $\theta_m$ uses fourth-order centered plus $0.25$ upwind dissipation, and all scalar tracers use 3D Zalesak FCT (fourth-order horizontal and vertical high-order fluxes). Horizontal subgrid turbulence uses the 3D Smagorinsky–Lilly SGS scheme of Section 6.4 (including background numerical viscosity $\nu_{\mathrm{bg}} = 0.004\,\Delta x\Delta y/10$), and vertical turbulence uses the MYNN 2.5 scheme of Section 6.3.

To relax the horizontal advective CFL constraint under strong mean winds, an exact Galilean shift $U_{\mathrm{shift}} = \frac{1}{2}(\max u + \min u)$ (plus $V_{\mathrm{shift}}$ in 3D) is subtracted per host column prior to sub-cycling and restored afterward. Dimensionally consistent Coriolis forcing (disabled in 1D/2D; acting on horizontal perturbation winds $(u', v')$ in 3D) and a perturbation-only Rayleigh sponge layer are included. In the convective gray zone ($N_x > 1$ or $N_y > 1$), whenever surface buoyancy flux $B_{\mathrm{sfc}} > 0$ produces unresolved subgrid boundary-layer thermals, self-extinguishing Deardorff (1970) thermal perturbations are seeded in the lowest three levels ($w_z = (1.0, 0.65, 0.35)$) with exact mass-weighted zero horizontal mean ($\overline{\rho_d\,\delta\theta'_m} = 0$):

$$
\begin{aligned}
w_* &= \big(\max(B_{\mathrm{sfc}}, 0)\,z_{\mathrm{ref}}\big)^{1/3}, \qquad \theta_* = \frac{\max(B_{\mathrm{sfc}}, 0)\,\theta_0 / g}{\max(w_*, 0.10)}, \\
f_{\mathrm{gz}} &= \operatorname{clamp}\!\left(\frac{\Delta x_{\mathrm{CRM}}}{4000\ \mathrm{m}},\ 0,\ 1\right), \qquad \sigma_{\theta} = \max\!\Big(\min(0.8\,f_{\mathrm{gz}}\,\theta_*,\ 0.25\ \mathrm{K}) - \operatorname{std}_{x,y}(\theta_{m,0}),\ 0\Big)
\end{aligned} \tag{8.7b}
$$

where $z_{\mathrm{ref}} = 1000\ \mathrm{m}$ and $\theta_0 = 300\ \mathrm{K}$. Seeding vanishes identically in stable nocturnal boundary layers ($B_{\mathrm{sfc}} \le 0$), at fine LES spacing ($\Delta x_{\mathrm{CRM}} \to 0$), and once resolved CRM thermal variance reaches the Deardorff target. A standalone periodic 3D CRM radiative–convective equilibrium driver (`step_coupled_crm_rce`; Wing et al., 2018; default $S_0 = 551.58\ \mathrm{W\ m^{-2}}$, $\mathrm{CO}_2 = 400\ \mathrm{ppmv}$) shares the same core.

### 8.4 Subgrid Orographic Blocking & Mountain Gravity-Wave Drag (`src/aoba/orography.py`)

Using the `Earth2014` subgrid orographic tensor $(\mu, \psi, \gamma, \sigma)$ from Eq. (2.7), Lott & Miller (1997, Eqs. 7–22) low-level flow blocking and internal mountain gravity-wave drag are computed ($H_{nc} = 0.50$, $C_d = 1.00$, $G = 0.80$). From the angle $\phi$ between the low-level mean wind $(u_{\mathrm{low}}, v_{\mathrm{low}})$ and the major ridge axis, the elliptical directional factor $D_{\mathrm{dir}}$ is formed; whenever nondimensional mountain height $H_n > H_{nc}$, implicit wake blocking acts below blocked depth $Z_b$:

$$
\begin{aligned}
D_{\mathrm{dir}} &= \sqrt{(B\cos^2\phi + C\sin^2\phi)^2 + (B - C)^2\sin^2\phi\cos^2\phi}, \\
H_n &= \frac{2\mu N_{\mathrm{low}}}{U_{\mathrm{low}}}, \qquad Z_b = 2\mu\,\max\!\left(1 - \frac{H_{nc}}{H_n},\ 0\right), \\
r_{\mathrm{blk}} &= C_d\frac{\sigma}{2\mu}D_{\mathrm{dir}}\sqrt{\frac{Z_b - z_{\mathrm{rel}}}{z_{\mathrm{rel}} + \mu}}\,|\mathbf{u}|, \qquad \left(\frac{\partial\mathbf{u}}{\partial t}\right)_{\mathrm{blk}} = -\frac{r_{\mathrm{blk}}}{1 + \Delta t\,r_{\mathrm{blk}}}\,\mathbf{u}
\end{aligned} \tag{8.8}
$$

The residual effective ridge height $h_{\mathrm{wave}} = \max(2\mu - Z_b, 0)$ launches mountain-wave momentum flux $\tau_{\mathrm{launch}}$ upward subject to Lindzen saturation $\tau_{\mathrm{sat},k}$ and zero top flux $\tau_{N_z} = 0$:

$$
\begin{aligned}
k_{\mathrm{wave}} &= G\,\frac{\sigma}{4\mu}\,D_{\mathrm{dir}}, \qquad \tau_{\mathrm{launch}} = \rho_{\mathrm{low}}\,k_{\mathrm{wave}}\,N_{\mathrm{low}}\,U_{\mathrm{low}}\,h_{\mathrm{wave}}^2, \\
\tau_{\mathrm{sat},k} &= \frac{\bar{\rho}_k\,k_{\mathrm{wave}}\,u_{\mathrm{proj}}^3}{N_k H_{nc}^2}, \qquad \left(\frac{\partial\mathbf{u}}{\partial t}\right)_{\mathrm{wave},k} = -\frac{\tau_k - \tau_{k+1}}{\rho_{d,k}\Delta z_k}\,\hat{\mathbf{u}}
\end{aligned} \tag{8.9}
$$

### 8.5 Map-Free Non-Orographic Gravity-Wave Drag (`src/aoba/gravity_wave.py`)

When `enable_nonoro_gwd = true`, a $N_{\mathrm{bins}} = 17$ phase-speed spectrum ($c_m \in [-80, +80]\ \mathrm{m\ s^{-1}}$, horizontal wavelength $\lambda_h = 200\ \mathrm{km}$) is launched from two purely physical sources without latitude-tuned source tables (Charron & Manzini, 2002; Beres et al., 2004, 2005; Richter et al., 2010):

1. **Frontal source (near $600\ \mathrm{hPa}$)**: Evaluates the 2D Hoskins (1982) frontogenesis function $F$ via Gauss–Green gradients, launching a Gaussian spectrum (width $c_w = 30\ \mathrm{m\ s^{-1}}$) with amplitude $A_{\mathrm{fr}}$ where $F > 0.5 F_{\mathrm{th}}$ ($F_{\mathrm{th}} = 1.25\times 10^{-15}\ \mathrm{K^2\ m^{-2}\ s^{-1}}$):

$$
\begin{aligned}
F &= -\theta_x^2 u_x - \theta_y^2 v_y - \theta_x\theta_y(v_x + u_y), \\
A_{\mathrm{fr}} &= \sigma\!\left(2.5\,\frac{F - F_{\mathrm{th}}}{F_{\mathrm{th}}}\right)
\end{aligned} \tag{8.10}
$$

2. **Convective source (CRM convective heating $Q_{\mathrm{conv}}$)**: From heating depth $h_d \in [1, 16]\ \mathrm{km}$ and layer-mean buoyancy frequency $N_h$, sets phase-speed width $\sigma_c$ and flux amplitude $\tau_{\mathrm{scale}}$, launching Rayleigh spectra $S_m$ along both zonal and meridional axes from the heating top $k_{\mathrm{top}}$:

$$
\begin{aligned}
\sigma_c &= \operatorname{clamp}\!\left(\frac{N_h h_d}{\pi},\ 10,\ 60\right)\ [\mathrm{m\ s^{-1}}], \qquad \tau_{\mathrm{scale}} = \tau_{0,\mathrm{conv}}\,\min\!\left(\frac{Q_{\max}}{Q_{\mathrm{ref}}},\ 4\right)^2, \\
x_m &= \frac{|c_m - u_{\mathrm{src}}|}{\sigma_c}, \qquad S_m = x_m\,\exp\!\left(-\frac{x_m^2}{2}\right)
\end{aligned} \tag{8.11}
$$

Each bin propagates upward with complete absorption at critical levels ($(c_m - u_{w,k}) s_m \le 2\ \mathrm{m\ s^{-1}}$), Lindzen (1981) wave breaking saturation $\tau_{\mathrm{sat},k}(c_m)$, and an open radiating upper boundary:

$$
\begin{aligned}
\tau_{\mathrm{sat},k}(c_m) &= \frac{1}{2}\,k_h\,\mathrm{eff}_{\mathrm{gw}}\,\frac{\rho_{w,k}}{N_{w,k}}|c_m - u_{w,k}|^3, \\
\left(\frac{\partial u}{\partial t}\right)_{\mathrm{GWD},k} &= \sum_{m=1}^{17}\frac{s_m\max(\tau_{\mathrm{in},m} - \tau_{\mathrm{out},m},\ 0)}{\rho_{d,k}\Delta z_{c,k}}
\end{aligned} \tag{8.12}
$$

### 8.6 Superparameterization Coupling & Standalone Global CRM Mode (`src/aoba/coupled_model.py`)

When `enable_superparam = true`, the embedded CRM integrates `n_crm_dyn_substeps` steps (default $12\times 60\ \mathrm{s}$) forced by host large-scale tendencies $(\partial\theta_m/\partial t)_{\mathrm{LS}}$ and $(\partial q_n/\partial t)_{\mathrm{LS}}$, then synchronizes back to the host via **exact mass-weighted averaging** that eliminates thermodynamic ratcheting drift, projects domain-mean CRM wind increments onto host edges via Eq. (2.6), and diagnoses convective heating $Q_{\mathrm{conv}}$ for the next Beres gravity-wave step:

$$
\begin{aligned}
\rho_{d,\mathrm{host}} &= \overline{\rho_{d,\mathrm{crm}}}, \qquad \phi_{\mathrm{host}} = \frac{\overline{\rho_{d,\mathrm{crm}}\phi_{\mathrm{crm}}}}{\max(\overline{\rho_{d,\mathrm{crm}}},\ 10^{-5})}, \qquad \phi \in \{\theta_m, q_n, q_p, q_v, q_c, q_i\}, \\
Q_{\mathrm{conv}} &= \max\!\left(\frac{\theta_m^{\mathrm{post}} - \theta_m^{\mathrm{pre}}}{\Delta t}\,\Pi - \overline{Q_{\mathrm{rad}}},\ 0\right)
\end{aligned} \tag{8.13}
$$

Setting `enable_superparam = false` bypasses the embedded CRM and executes a unified standalone global cloud-resolving model (GCRM) on the SCVT nonhydrostatic mesh.

| Coupling Mode (`enable_superparam`) | Architecture | Advantages | Limitations & Considerations |
| :--- | :--- | :--- | :--- |
| `true` *(default: Superparameterization SP)* | Coarse global SCVT host (G3–G5) + embedded CRM (configurable `crm_dx`, default $4\ \mathrm{km}$) + MAML per cell | Enables multi-decadal to multi-century climate integrations on a single GPU while explicitly representing deep convection, MJO, diurnal convection, and subgrid land heterogeneity without cumulus parameterization. | A 2D curtain (`crm_ny=1`) simplifies 3D rotational mesoscale dynamics inside the CRM, and direct subgrid propagation between neighboring host columns occurs only through host-mean fields. |
| `false` (Standalone Global CRM / GCRM) | No embedded CRM; single global SCVT nonhydrostatic mesh | Eliminates scale separation and periodic CRM boundary assumptions entirely; resolves continuous 3D global convection, fronts, and orographic flows at high grid levels (G8–G10). | Explicit convection requires horizontal grid spacing $\le 4\text{–}10\ \mathrm{km}$; running standalone on coarse meshes (G3–G5) yields excessive or delayed grid-scale convection. |

### 8.7 Online Surrogate Superparameterization (OSSP; `src/aoba/crm_self_distill.py`)

When `enable_crm_self_distill = true` and `enable_rad_self_distill = true`, AOBA activates Online Surrogate Superparameterization (OSSP), executing the genuine 2D/3D CRM and full ICA / 3D-TICA radiation solver on a rotating, spatially stratified subset of teacher host columns (`1 / crm_distill_stride`, augmented by high-epistemic-uncertainty out-of-distribution columns) while evaluating the remaining columns via two online-trained two-head epistemic residual MLPs (`CRMSelfDistillNet` and `RadSelfDistillNet`) backed by GPU ring-and-reservoir replay buffers. No pre-trained climate weights are prescribed, so planetary, paleoclimate, and radiative regime shifts are tracked on the fly.

Every surrogate column uses an operator-split baseline. Stiff 1D vertical boundary-layer mixing and surface-flux injection $(\mathrm{SHF}, E, \tau_x, \tau_y)$ are first advanced analytically across all host columns via the fused MYNN Level 2.5 tridiagonal solver (Section 6) in conservative pure-mixing mode (`do_sat_adj = false`, preserving subgrid cloud condensate $q_c, q_i$ without grid-mean phase conversion while enforcing $q_v + q_c + q_i = q_n$ after host FCT advection of $q_n, q_p, q_c, q_i$). To prevent autoregressive condensate self-feedback and train-test covariate shift, the surrogate network ingests conserved moist thermodynamic profiles $(T_{li}, \theta_l, q_n, q_n / q_{\mathrm{sat}}(T_{li}))$, large-scale forcing, and surface-tile heterogeneity invariants shared identically by teacher and student columns. Inside fused GPU kernels (`crm_distill_closure_col`, `rad_distill_closure_col`), a pressure-tapered convective envelope $s_{\mathrm{conv}}(\Pi)$ equals unity throughout the troposphere ($p \ge 210\ \mathrm{hPa}$) and vanishes smoothly at and above the $90\ \mathrm{hPa}$ tropopause cold point ($\Pi \le 0.50$), while the layer activity gate $g_{\mathrm{act},k} = \max(g_{\mathrm{col}}, g_{\mathrm{pbl}}(\Pi_k))$ combines a column moisture activity gate $g_{\mathrm{col}}$ ($\max_k \mathrm{RH}_k > 0.72$) with a smooth boundary-layer gate $g_{\mathrm{pbl}}(\Pi)$ ($z \lesssim 3.1\ \mathrm{km}$) so clear-sky boundary-layer turbulence residuals are represented while dry upper-tropospheric air remains invariant. Raw convective temperature and moisture logits pass through a two-pass vertical binomial filter $[0.25, 0.50, 0.25]$ (eliminating grid-scale $2\Delta z$ vertical modes) before forming bounded convective residuals ($|\Delta T_{\mathrm{raw}}| \le 0.15\ \mathrm{K}$, $|\Delta q_{n,\mathrm{raw}}| \le \Delta q_{\lim}$). Moisture redistribution applies exact zero-sum projection weighted by $w_{\mathrm{bal},k} = \Delta q_{\lim,k}$ (so any uniform logit offset lies strictly in the zero-tendency nullspace and never pumps lower-tropospheric residuals aloft), allows falling precipitating hydrometeors ($q_{p,0} > q_{p,\mathrm{conv}}$) to exit directly as surface precipitation without multiplying $q_n$ by a column scale factor, gates suspended precipitation prediction $q_{p,\mathrm{prov}} = s_{\mathrm{conv}} g_{\mathrm{col}} \min(q_{n,\mathrm{base}}, 2\times 10^{-3})\cdot 0.25\,\sigma(r_2 - 2)^2$ by column humidity $g_{\mathrm{col}}$ with growth capped at $0.15\,W_{q_n}$ (preventing dry/cloudy columns from draining $q_n$ into radiatively invisible suspended $q_p$), extracts only the net non-precipitating sink preferentially from warm liquid clouds and moist lower/mid-tropospheric layers ($\mathrm{RH} > 0.60$, weighted by $2.0\,q_c + 0.25\,q_i$ so heavy warm rain does not strip upper-tropospheric cirrus anvils dry), partitions condensate with a smooth $C^1$ quadratic humidity gate $[\operatorname{clamp}((\mathrm{RH}-0.55)/0.40, 0, 1)]^2$ and mixed-phase ice fraction $f_{\mathrm{ice}} = (1 - \omega_{\mathrm{liq}})\sigma(r_4)$ ($f_{\mathrm{ice}} = 1$ for $T \le 253.15\ \mathrm{K}$), and conservatively rains out excess liquid ($q_c > 5\times 10^{-4}\ \mathrm{kg\ kg^{-1}}$) and ice ($q_i > \min(q_{\mathrm{sat}}, 10^{-4}\ \mathrm{kg\ kg^{-1}})$):

$$
\begin{aligned}
s_{\mathrm{conv}}(\Pi) &= \left[\operatorname{clamp}\!\left(\frac{\Pi - 0.50}{0.14},\ 0,\ 1\right)\right]^2, \qquad g_{\mathrm{col}} = \operatorname{clamp}\!\left(\frac{\max_{k\in\mathrm{conv}}(q_{n,k}/q_{\mathrm{sat},k}) - 0.72}{0.25},\ 0,\ 1\right), \\
g_{\mathrm{act},k} &= \max\!\left(g_{\mathrm{col}},\ \operatorname{clamp}\!\left(\frac{\Pi_k - 0.90}{0.05},\ 0,\ 1\right)\right), \qquad \Delta q_{\lim,k} = g_{\mathrm{act},k}\min\!\big(0.015\,q_{n,\mathrm{base},k},\ 1.0\times 10^{-4}\big), \\
\Delta q_{n,\mathrm{bal},k} &= \Delta q_{n,\mathrm{raw},k} - \frac{\sum_j \Delta m_j \Delta q_{n,\mathrm{raw},j}}{\sum_j \Delta m_j w_{\mathrm{bal},j}}\,w_{\mathrm{bal},k}, \qquad \phi_{\mathrm{pbl}}(\Pi) = \operatorname{clamp}\!\left(1 - \frac{\Pi_0 - \Pi}{0.015\,\Pi_0},\ 0,\ 1\right) s_{\mathrm{conv}}, \\
\Delta T_{\mathrm{shf},k} &= \frac{\mathrm{SHF}\,\Delta t}{c_{pd}\sum_j \Delta m_j\,\phi_{\mathrm{pbl},j}}\,\phi_{\mathrm{pbl},k}
\end{aligned} \tag{8.14}
$$

Sensible enthalpy is closed in true temperature space $T = \Pi\theta_m / (1 + (R_v/R_d) q_v)$ by depositing cloud phase-change and precipitation-sink latent heat $\Delta T_{\mathrm{lat},k}$ directly in the layers where condensation occurs, closing only the residual precipitation and snow-melt budget via $h_{\mathrm{corr}}$, and applying a two-pass enthalpy-conserving red-black dry-convective adjustment to remove any super-adiabatic pairs ($\theta_k > \theta_{k+1}$) before reconstructing $\theta_m^{n+1}$:

$$
\begin{aligned}
\Delta T_{\mathrm{lat},k} &= \frac{L_{v0}\big[(q_{c,k}^{\mathrm{pf}} + q_{i,k}^{\mathrm{pf}}) - (q_{c,k}^0 + q_{i,k}^0) + \Delta q_{\mathrm{psink},k}\big] + L_{f0}\big(q_{i,k}^{n+1} - q_{i,k}^0\big)}{c_{pd}}, \\
\Delta H_{\mathrm{precip}} &= L_{v0}\big(P_{\mathrm{tot,orig}}\Delta t - M_{\mathrm{psink}} + \Delta M_p\big) + L_{f0} P_{\mathrm{snow}}\Delta t, \\
h_{\mathrm{corr}} &= \frac{\Delta H_{\mathrm{precip}} - c_{pd}\sum_k \Delta m_k \Delta T_{\mathrm{raw},k}}{c_{pd}\sum_k \Delta m_k s_{\mathrm{conv},k}}, \qquad \Delta T_{\mathrm{cons},k} = \Delta T_{\mathrm{shf},k} + \Delta T_{\mathrm{lat},k} + \Delta T_{\mathrm{raw},k} + h_{\mathrm{corr}}\,s_{\mathrm{conv},k}, \\
\theta_{m,k}^{n+1} &= \theta_{m,k}^{\mathrm{post,ls}} + \frac{\Delta T_{\mathrm{cons},k}\!\left(1 + \frac{R_v}{R_d} q_{v,k}^{n+1}\right) + T_k^{\mathrm{post,ls}}\frac{R_v}{R_d}\!\left(q_{v,k}^{n+1} - q_{v,k}^0\right)}{\max(\Pi_k,\ 0.05)}
\end{aligned} \tag{8.15}
$$

For each radiation surrogate subcolumn, shortwave fluxes vanish identically on the night side ($\mu_0 \le 0$), longwave fluxes scale analytically with surface Planck emission $\sigma T_{\mathrm{sfc}}^4$, and the radiative heating profile satisfies exact column flux-divergence closure. To represent multi-hour convective lifecycle hysteresis and diurnal buoyancy phase lag without teacher–student covariate shift (Mapes & Neale, 2011; Colin et al., 2019), three prognostic scalar memory states $(M_{P,\mathrm{fast}}, M_{P,\mathrm{slow}}, M_{B,\mathrm{slow}})$ advance identically across all teacher and student columns via exact exponential integration ($\alpha_\tau = -\operatorname{expm1}(-\Delta t/\tau)$ with $\tau_{P,\mathrm{fast}} = 1.5\ \mathrm{h}$, $\tau_{P,\mathrm{slow}} = 4.0\ \mathrm{h}$, $\tau_B = 3.0\ \mathrm{h}$) and supply four dimensionless memory inputs $\big(\ln(1 + M_{P,\mathrm{fast}}), \ln(1 + M_{P,\mathrm{slow}}), 200\,M_{B,\mathrm{slow}}, 200(B_{\mathrm{sfc}}^+ - M_{B,\mathrm{slow}})\big)$:

$$
\begin{aligned}
M_{P,\mathrm{fast}}^{n+1} &= M_{P,\mathrm{fast}}^n + \alpha_{1.5\mathrm{h}}\big(P_{\mathrm{mm/d}} - M_{P,\mathrm{fast}}^n\big), \qquad
M_{P,\mathrm{slow}}^{n+1} = M_{P,\mathrm{slow}}^n + \alpha_{4\mathrm{h}}\big(P_{\mathrm{mm/d}} - M_{P,\mathrm{slow}}^n\big), \\
M_{B,\mathrm{slow}}^{n+1} &= M_{B,\mathrm{slow}}^n + \alpha_{3\mathrm{h}}\big(\max(B_{\mathrm{sfc}}, 0) - M_{B,\mathrm{slow}}^n\big)
\end{aligned} \tag{8.15b}
$$

Lowest-layer student subgrid anomalies $(\theta_{m,0}', q_{v,0}')$ and gust outflows $(u_0', v_0')$ are reconstructed directly around the cell mean from the periodic `SubgridSpatialHead` state prediction (`crm_student_tile_update_col`) with exact zero-mean conservation rather than accumulating onto previous-step anomalies. Online optimization combines boundary-layer-emphasized true-temperature, vertical lapse-rate, per-step tendency ($\Delta q_n, \Delta T$), and saturation-normalized moisture ($\Delta q_n / q_{\mathrm{sat}}$) Smooth-$L_1$ losses with a nullspace gauge-fixing penalty on uniform-projection temperature and momentum logits to prevent saturation drift along projection nullspaces:

$$
\sum_{k=0}^{N_z-1} c_{p,\mathrm{moist},k}\,\Delta m_k\,Q_{\mathrm{rad},k} = (\mathrm{ASR} - \mathrm{OLR}) - (F_{\mathrm{SW,sfc}}^{\mathrm{net}} + F_{\mathrm{LW,sfc}}^{\mathrm{net}}) \tag{8.16}
$$

When `enable_distill_shadow_diag` is active, each surrogate is evaluated in shadow mode on the current active teacher columns strictly prior to the online gradient update (`evaluate_crm_shadow_on_teachers`, `evaluate_rad_shadow_on_teachers`), yielding an unbiased prequential out-of-sample measure of physical-versus-surrogate divergence across regime shifts such as abrupt $\mathrm{CO}_2$ quadrupling without perturbing model state or training trajectories. Host-atmosphere coupling fidelity weights vertical profiles by dry-air column mass $\Delta m_{k,i} = \rho_{d,k,i}\Delta z_{k,i}$ (for $\theta_m, q_n, q_c + q_i, u, v$) and by moist heat capacity $c_{p,\mathrm{moist},k,i}\Delta m_{k,i}$ (for radiative heating $Q_{\mathrm{rad}}$) alongside cell-mean precipitation $P$ and top-of-atmosphere fluxes $(\mathrm{OLR}, \mathrm{ASR})$:

$$
\begin{aligned}
\mathrm{RMSE}_{\mathrm{atm}}(\psi) &= \sqrt{\frac{\sum_{i\in\mathcal{T}}\sum_{k=0}^{N_z-1} \Delta m_{k,i}\big(\hat{\psi}_{k,i} - \psi_{k,i}^{\mathrm{phys}}\big)^2}{\sum_{i\in\mathcal{T}}\sum_{k=0}^{N_z-1} \Delta m_{k,i}}}, \\
\mathrm{RMSE}_{\mathrm{rad}}(Q_{\mathrm{rad}}) &= \sqrt{\frac{\sum_{m\in\mathcal{T}_{\mathrm{rad}}}\sum_{k=0}^{N_z-1} c_{p,\mathrm{moist},k,m}\Delta m_{k,m}\big(\hat{Q}_{\mathrm{rad},k,m} - Q_{\mathrm{rad},k,m}^{\mathrm{phys}}\big)^2}{\sum_{m\in\mathcal{T}_{\mathrm{rad}}}\sum_{k=0}^{N_z-1} c_{p,\mathrm{moist},k,m}\Delta m_{k,m}}}
\end{aligned} \tag{8.17}
$$

where $\mathcal{T}$ denotes the evaluated teacher host columns, $\mathcal{T}_{\mathrm{rad}}$ the evaluated teacher radiation subcolumns, $\hat{\psi}$ the pre-update surrogate prediction, and $\psi^{\mathrm{phys}}$ the physical CRM or radiative transfer output. Land-surface coupling fidelity restricts the comparison to land surface tiles ($\mathbb{I}_{\mathrm{land},i,j} = 1$) at MAML tile resolution $(N_{y,\mathrm{tile}}, N_{x,\mathrm{tile}})$, applying the identical rank-weighted tile precipitation, zero-mean cold-pool microclimate $(\theta_{m,0}, q_{v,0})$, and gust-augmented surface wind speed $|\mathbf{u}_0|$ reconstruction used by student cells together with tile-level downward shortwave ($\mathrm{SW}_\downarrow$), photosynthetically active radiation ($\mathrm{PAR}$), and downward longwave ($\mathrm{LW}_\downarrow$) fluxes:

$$
\begin{aligned}
\mathrm{RMSE}_{\mathrm{land}}(\phi) &= \sqrt{\frac{\sum_{i\in\mathcal{T}}\sum_{j=1}^{N_{\mathrm{tile}}} \mathbb{I}_{\mathrm{land},i,j}\big(\hat{\phi}_{i,j} - \phi_{i,j}^{\mathrm{phys}}\big)^2}{\sum_{i\in\mathcal{T}}\sum_{j=1}^{N_{\mathrm{tile}}} \mathbb{I}_{\mathrm{land},i,j}}}, \\
r_{\mathrm{land,anom}}(P) &= \operatorname{corr}_{\mathbb{I}_{\mathrm{land}}=1}\!\left(\hat{P}_{i,j} - \overline{\hat{P}}_i,\ P_{i,j}^{\mathrm{phys}} - \overline{P}_i^{\mathrm{phys}}\right)
\end{aligned} \tag{8.18}
$$

where $\phi \in \{\theta_{m,0}, q_{v,0}, |\mathbf{u}_0|, P_{\mathrm{tile}}, \mathrm{SW}_\downarrow, \mathrm{PAR}, \mathrm{LW}_\downarrow\}$ and overbars denote host-cell tile means.

---

## 9. Ocean Circulation (`ocean_mode`), Independent Fine Ocean, Surface Waves, EVP–ITD Sea Ice & FlexPFT Marine BGC

### 9.1 Selectable Ocean Dynamical Modes (`ocean_mode`)

`[forcing] ocean_mode` selects between two dynamical ocean cores; setting `ocean_glevel > glevel` additionally couples an independent higher-resolution Voronoi ocean mesh.

| `ocean_mode` | Vertical Structure | Advantages | Limitations & Considerations |
| :--- | :--- | :--- | :--- |
| `"ekman_sw"` | 1.5-layer (mixed layer + reduced-gravity thermocline) | Reproduces equatorial upwelling, Rossby/Kelvin waves, and subtropical gyres without prescribed $Q$-fluxes at minimal cost, equilibrating within a few years. | Cannot represent salinity, deep-water formation, thermohaline circulation (AMOC), deep-ocean heat uptake, or 3D marine biogeochemistry. |
| `"ogcm_3d"` *(default)* | 3D multi-layer (default 18 layers / $6000\ \mathrm{m}$) | Self-consistently simulates temperature, salinity, sea-surface height, AMOC/ACC, KPP/tidal mixing, GM90/submesoscale eddies, and 3D marine BGC over real bathymetry, with optional `ocean_glevel > glevel` refinement. | Deep-ocean thermohaline equilibrium requires centuries to millennia of spin-up, and barotropic sub-cycling plus 3D advection–diffusion increases computational cost. |

![Figure 9.1: Selectable Ocean Modes, Surface Waves, EVP-ITD Sea Ice & FlexPFT Marine Biogeochemistry](assets/fig_06_ocean_ice_bgc.svg)

### 9.2 Sea-Surface Albedo, Mixed-Layer Heat Budget & Thermodynamic Sea Ice with Leads (`src/aoba/surface.py`)

Open-ocean direct albedo $\alpha_{\mathrm{dir}}(\mu_0)$ (Briegleb et al., 1986, Eq. 11; diffuse albedo $0.06$, MIROC6 Sec. 3.8.1.1), sea-ice fractional cover $f_i$ (SCALE-RM Eq. 8.342), snow cover over sea ice $f_{\mathrm{sn,ice}}$ (modulating VIS $0.72\to 0.90$ and NIR $0.42\to 0.70$ albedos), and series ice–snow conductive heat flux $G_{oi}$ are:

$$
\begin{aligned}
\alpha_{\mathrm{dir}}(\mu_0) &= \frac{0.026}{\mu_0^{1.7} + 0.065} + 0.15(\mu_0 - 0.1)(\mu_0 - 0.5)(\mu_0 - 1.0), \\
f_i &= \operatorname{clamp}\!\left(\sqrt{\frac{m_{\mathrm{ice}}}{450}},\ 0,\ 1\right), \qquad f_{\mathrm{sn,ice}} = \tanh\!\left(\frac{\mathrm{SWE}_{\mathrm{ice}}}{15}\right), \\
G_{oi} &= \left[\frac{h_i}{2 k_{\mathrm{ice}}} + \frac{h_{\mathrm{sn}}}{k_{\mathrm{snow}}}\right]^{-1}\big(T_{f,\mathrm{ocn}} - T_i\big)
\end{aligned} \tag{9.1}
$$

Snowfall onto open water imparts latent melt cooling $-L_{f0} P_{\mathrm{snow}}$, whereas snowfall onto ice accumulates as snowpack. Sea-ice skin temperature $T_i \le 273.15\ \mathrm{K}$ solves an implicit surface energy balance. Over open water, upward turbulent and longwave fluxes are coupled semi-implicitly to the mixed-layer heat capacity $C_w = \rho_{\mathrm{sw}} c_{p,\mathrm{sw}} h_{\mathrm{ml}}$ via the surface flux sensitivity $\lambda_{\mathrm{ow}} = \partial(H_{\mathrm{ow}} + \mathrm{LHF}_{\mathrm{ow}} + \mathrm{LW}_{\uparrow,\mathrm{ow}})/\partial T_{\mathrm{ml}}$ (Polcher et al., 1998; Best et al., 2004), eliminating explicit $2\Delta t$ oscillations even for thin top ocean layers ($h_{\mathrm{ml}} \le 2\ \mathrm{m}$) while preserving atmosphere–ocean energy conservation identically:

$$
\begin{aligned}
\lambda_{\mathrm{ow}} &= \rho_s c_p g_{a,h} + L_{v0}\,\rho_s g_{a,q}\frac{\partial q_{\mathrm{sat}}}{\partial T} + 4\,\varepsilon_o \sigma T_{\mathrm{ml}}^3, \\
\Delta T_{\mathrm{ml}} &= \frac{\Delta t\,\big[(1 - f_i)\,G_{\mathrm{ow}}^0 - f_i\,G_{oi}\big]}{C_w + \Delta t\,(1 - f_i)\,\lambda_{\mathrm{ow}}}, \\
H_{\mathrm{ow}} &= H_{\mathrm{ow}}^0 + \rho_s c_p g_{a,h}\,\Delta T_{\mathrm{ml}}, \qquad E_{\mathrm{ow}} = E_{\mathrm{ow}}^0 + \rho_s g_{a,q}\frac{\partial q_{\mathrm{sat}}}{\partial T}\,\Delta T_{\mathrm{ml}}, \\
\mathrm{LW}_{\mathrm{emit,ow}} &= \varepsilon_o \sigma T_{\mathrm{ml}}^4 + 4\,\varepsilon_o \sigma T_{\mathrm{ml}}^3\,\Delta T_{\mathrm{ml}}
\end{aligned} \tag{9.1b}
$$

where $G_{\mathrm{ow}}^0 = \mathrm{SW}_{\mathrm{abs,ow}} + \mathrm{LW}_\downarrow - \mathrm{LW}_{\uparrow,\mathrm{ow}}^0 - H_{\mathrm{ow}}^0 - L_{v0} E_{\mathrm{ow}}^0 - L_{f0} P_{\mathrm{snow}}$ is the level-$n$ net downward open-water flux, $\partial q_{\mathrm{sat}}/\partial T = q_{\mathrm{sat}}(T_{\mathrm{ml}})\,4302.645 / (T_{\mathrm{ml}} - 29.65)^2$ follows Bolton (1980, Eq. 10), and $T_{\mathrm{ml}}$ updates via Neumaier/Kahan summation of $\Delta T_{\mathrm{ml}}$, converting any supercooling below $T_{f,\mathrm{ocn}}$ conservatively into sea-ice formation.

### 9.3 Option 1: `ocean_mode = "ekman_sw"` (1.5-Layer Ekman + Shallow-Water Ocean, `src/aoba/ocean_ekman.py`)

Following Codron (2012, Eqs. 1–10), edge-centered regularized Ekman mass transport ($\tau_{\mathrm{Ek}} = 2\ \mathrm{d}$, $r_E = \tau_{\mathrm{Ek}}^{-1}$, zero coastal flux), reduced-gravity ($g' = 0.03\ \mathrm{m\ s^{-2}}$) shallow-water thermocline thickness $H_{\mathrm{sw}} \in [h_1+20, 800]\ \mathrm{m}$ (WS-RK3 sub-cycling, scale-adaptive Munk viscosity $A_h$, damping $\tau_{\mathrm{sw}} = 180\ \mathrm{d}$), and Ekman-removed geostrophic velocity $u_{g,e}$ are solved as:

$$
\begin{aligned}
\mathbf{M}_{\mathrm{Ek},e} &= \frac{r_E\boldsymbol{\tau}_e - f_e\hat{\mathbf{k}}\times\boldsymbol{\tau}_e}{f_e^2 + r_E^2}, \qquad \frac{\partial H_{\mathrm{sw}}}{\partial t} = -\nabla_{\mathrm{SCVT}}\cdot(H_{\mathrm{sw},\mathrm{up}} u_{\mathrm{sw}}), \\
u_{g,e} &= u_{\mathrm{sw},e} - \frac{M_{n,\mathrm{Ek},e}}{\rho_w\bar{H}_{\mathrm{sw},e}}
\end{aligned} \tag{9.2}
$$

Two-layer enthalpy advection, Ekman pumping $W_E = \nabla\cdot\mathbf{M}_1$, diapycnal exchange ($\kappa_v = 2\times 10^{-5}\ \mathrm{m\ s^{-1}}$), coastal glacial-calving melt cooling $Q_{\mathrm{calve}}$, and conservative freezing are integrated with Kahan compensated summation.

### 9.4 Option 2: `ocean_mode = "ogcm_3d"` (3D Multi-Layer SCVT OGCM, `src/aoba/ocean_ogcm.py`)

Integrates the 3D hydrostatic Boussinesq primitive equations on 18 partial-step vertical layers ($50\ \mathrm{m}$ surface to $880\ \mathrm{m}$ abyssal; $6000\ \mathrm{m}$ total depth) over `Earth2014 BED2014` bathymetry.

1. **Nonlinear Equation of State (Roquet et al., 2015, Table 1)**: Relative to $T_{\mathrm{ref}} = 283.15\ \mathrm{K}$ and $S_{\mathrm{ref}} = 34.7\ \mathrm{psu}$, density anomaly $\rho'(T, S, z)$ includes thermal expansion, thermobaricity, cabbeling, and haline contraction:

$$
\rho'(T, S, z) = \rho_w\Big[-\big(\alpha_0 + \alpha_T\Delta T + \alpha_z z + \alpha_S\Delta S\big)\Delta T + \beta_0\Delta S\Big] \tag{9.3}
$$

2. **Surface Virtual Salinity Flux & Jerlov Type I Shortwave Penetration**: Surface virtual salinity flux $\Delta S_0$ from evaporation $E$, precipitation $P$, river runoff $R$, and brine rejection ($S_{\mathrm{ice}} = 4\ \mathrm{psu}$), together with the Paulson & Simpson (1977) two-exponential shortwave penetration profile $I(z)$, are:

$$
\begin{aligned}
\Delta S_0 &= \frac{S_{\mathrm{ref}}(E - P - R)\Delta t + (S_{\mathrm{ref}} - S_{\mathrm{ice}})\Delta m_{\mathrm{ice}}}{\rho_w h_0}, \\
I(z) &= 0.58\,e^{-z/0.35} + 0.42\,e^{-z/23.0}
\end{aligned} \tag{9.4}
$$

Hydrostatic pressure gradients integrate nominal layer thicknesses (Pacanowski & Gnanadesikan, 1998) so resting stratified oceans over partial bottom steps produce identically zero spurious pressure-gradient errors.

3. **KPP Boundary Layer, Interior Shear, Abyssal Tidal Mixing & Convective Adjustment**: Following Large, McWilliams & Doney (1994), bulk Richardson number $\mathrm{Ri}_{b,k} < 0.30$ determines boundary-layer depth $h_{\mathrm{BL}}$, cubic shape function $G(\sigma)$, KPP diffusivity $K_{v,\mathrm{KPP}}$, and unstable nonlocal flux $F_{\mathrm{NL}}$ ($C_s = 6.33$). Interior shear mixing $K_{v,\mathrm{shear}}$, Bryan–Lewis background diffusivity $K_{v,\mathrm{bg}}$, and bottom-slope ($|\nabla H|$) / roughness ($\sigma_{\mathrm{topo}}$) internal tidal mixing (St. Laurent et al., 2002; Simmons et al., 2004; $\zeta_{\mathrm{tide}} = 500\ \mathrm{m}$) sum into $K_{v,\mathrm{tot}}$ (elevated to $\kappa_{v,\mathrm{conv}} = 1.0\ \mathrm{m^2\ s^{-1}}$ where $N^2 < 0$, and multiplied by $E_L$ for $z < h_{\mathrm{BL}}$ under wave coupling):

$$
\begin{aligned}
G(\sigma) &= \sigma(1 - \sigma)^2, \qquad K_{v,\mathrm{KPP}} = h_{\mathrm{BL}} w_s G(\sigma), \qquad F_{\mathrm{NL}} = -C_s G(\sigma) w^{\mathrm{sfc}}, \\
K_{v,\mathrm{shear}}(\mathrm{Ri}_g) &= \nu_0\left[1 - \min\!\left(\frac{\mathrm{Ri}_g}{0.70},\ 1\right)^2\right]^3, \\
K_{v,\mathrm{tot}}(z) &= K_{v,\mathrm{bg}}(z) + \min\!\left(\frac{0.0667\,E_{\mathrm{tide}}\,F_{\mathrm{tide}}(z)}{\rho_w\max(N^2, 10^{-7})},\ 5\times 10^{-3}\right) + K_{v,\mathrm{KPP}}(z) + K_{v,\mathrm{shear}}(\mathrm{Ri}_g)
\end{aligned} \tag{9.5}
$$

4. **3D Baroclinic WS-RK3 Momentum + Split-Explicit Barotropic Free Surface (Ringler et al., 2010, 2013; Griffies & Hallberg, 2000; Shchepetkin & McWilliams, 2005)**: 3D baroclinic momentum advances in vector-invariant form via 3-stage WS-RK3 with energy-conserving TRiSK vorticity flux $-(\zeta + f)\hat{\mathbf{k}}\times\mathbf{u}_o$, Voronoi kinetic-energy Bernoulli gradient $-\nabla K$, wet-masked scale-selective 4th-order biharmonic hyperviscosity $-\nu_4\nabla^4\mathbf{u}_o$ paired with a weak residual Laplacian Munk floor $A_h\nabla^2\mathbf{u}_o$ ($\delta_{M4} = 0.65\,\Delta c_{\min}$, $\delta_{M2} = 0.25\,\Delta c_{\min}$, CFL-capped for stability across arbitrary mesh resolutions), semi-implicit Rayleigh and quadratic bottom drag ($C_{d,\mathrm{bot}} = 2.5\times 10^{-3}$), and implicit vertical viscosity (Thomas tridiagonal). The 2D barotropic free-surface mode $(u_{\mathrm{bt}}, \eta)$ forced by depth-averaged tendency $F_{\mathrm{baro}}$ is integrated over up to 192 sub-cycles (fused Metal/CUDA kernel) with zero-steady-drag fast-fluctuation damping acting exclusively on departures from the sub-cycle anchor velocity $(u_{\mathrm{bt}} - u_{\mathrm{bt}}^{\mathrm{bar}})$ to eliminate boxcar resonance without damping geostrophic currents, after which time-averaged $\bar{u}_{\mathrm{bt}}$ replaces the depth-mean component of the 3D velocity:

$$
\begin{aligned}
\frac{\partial \mathbf{u}_o}{\partial t} + (\zeta + f)\hat{\mathbf{k}}\times\mathbf{u}_o &= -g\nabla\eta - \nabla\!\left(\frac{p_{\mathrm{hyd}}}{\rho_0} + K - a_{\mathrm{div}}\nabla\cdot\mathbf{u}_o\right) + \frac{\partial}{\partial z}\left(A_v\frac{\partial \mathbf{u}_o}{\partial z}\right) + A_h\nabla^2\mathbf{u}_o - \nu_4\nabla^4\mathbf{u}_o - r_{\mathrm{drag}}\mathbf{u}_o, \\
K_{k,i} &= \frac{1}{4 A_i}\sum_{e\in\mathcal{E}(i)} d_{c,e} d_{v,e}\,u_{k,e}^2, \qquad a_{\mathrm{div}} = 0.030\,\frac{\Delta c_{\min}^2}{\Delta t}, \\
\nu_4 &= \min\!\left(\beta\,\delta_{M4}^5,\ 10^{-3}\frac{\Delta c_{\min}^4}{\Delta t}\right), \qquad A_h = \min\!\left(\beta\,\delta_{M2}^3,\ 0.04\frac{\Delta c_{\min}^2}{\Delta t}\right)
\end{aligned} \tag{9.6}
$$

Here the resolution-scaled divergence damping term $+a_{\mathrm{div}}\nabla(\nabla\cdot\mathbf{u}_o)$ is folded directly into the Bernoulli potential $p_{\mathrm{hyd}}/\rho_0 + K$ at zero extra kernel cost, suppressing the C-grid Hollingsworth–Bernoulli divergent monopole instability while leaving non-divergent geostrophic currents and eddies untouched.

5. **Resolution-Aware GM90, Fox-Kemper Submesoscale Restratification, FCT-$z^*$ Advection, Rotated Redi Diffusion & Increment Implicit Vertical Diffusion**: From neutral slopes $S_x$ ($|S_x| \le 10^{-2}$, tapered linearly across the boundary layer), Hallberg (2013) Rossby-radius-aware GM90 streamfunction $\Psi_{\mathrm{GM}}$ ($\kappa_{\mathrm{GM}} = 800\ \mathrm{m^2\ s^{-1}}$) and Fox-Kemper et al. (2008, Eq. 5; 2011) mixed-layer eddy streamfunction $\Psi_{\mathrm{MLE}}$ yield bolus velocity $u_{\mathrm{GM},k}$. Tracers advance in $z^*$ coordinates ($h^*_k, w^*_{k+1/2}$; Adcroft & Campin, 2004; Danilov et al., 2017) with anomaly-form Zalesak FCT advection, full rotated Redi diffusion ($\kappa_{\mathrm{Redi}} = 1000\ \mathrm{m^2\ s^{-1}}$, diagonal $K_{v,\mathrm{Redi}} \propto S_x^2$ folded into the implicit solve), and increment-form implicit vertical diffusion with telescoping flux reconstruction and Neumaier/Kahan summation:

$$
\begin{aligned}
\Psi_{\mathrm{GM}} &= \kappa_{\mathrm{GM}}\left[1 + \left(\frac{L_d}{\Delta c_e}\right)^2\right]^{-1} S_x, \\
\Psi_{\mathrm{MLE}} &= C_{\mathrm{FK}}\,h_{\mathrm{BL}}^2\,(\partial_x b_{\mathrm{ml}})\,(1 - \xi^2)\left(1 + \frac{5}{21}\xi^2\right), \\
u_{\mathrm{GM},k} &= -\frac{\Psi_{k+1/2} - \Psi_{k-1/2}}{h_{e,k}}, \qquad h^*_k = h_{c,k}\left(1 + \frac{\eta}{H}\right), \qquad (I + L)\Delta q = -L q
\end{aligned} \tag{9.7}
$$

When EVP is disabled, a fallback free-drift plus plastic-ridging sea-ice transport (`step_seaice_drift_voronoi`) is available.

### 9.5 Independent Fine Ocean Grid Coupling (`ocean_glevel > glevel`) & Current-Relative Wind (`src/aoba/ocean_coupler.py`)

For each fine ocean cell $j$ mapped to parent atmospheric cell $p(j) = \arg\max_i(\mathbf{x}_{\mathrm{ocn},j}\cdot\mathbf{x}_{\mathrm{atm},i})$, ocean-to-atmosphere fields use exact wet-area-weighted averaging, atmosphere-to-ocean intensive fluxes are directly gathered, and extensive runoff/river fluxes apply global-integral conservation scaling. Surface heating and ice-mass changes from the coarse atmospheric step are combined into an exact total water+ice enthalpy increment $\Delta E_{\mathrm{atm}} = \rho_{\mathrm{sw}} c_{p,\mathrm{sw}} h_{\mathrm{ml,atm}}\,\Delta T_{\mathrm{ml,atm}} - L_{f0}\,\Delta m_{\mathrm{ice,atm}}$ (`apply_atm_surface_increments_to_ocn_sfc`) and partitioned on each fine ocean cell $j$ at its local freezing point $T_f(S_j)$ (so fine warm open-ocean cells and fine ice-covered cells beneath one mixed coarse cell conserve exact enthalpy without artificial sea-ice ratcheting), accompanied by an exact exponential linearized bulk air–sea thermal feedback ($\lambda_{\mathrm{as}} = 40\ \mathrm{W\ m^{-2}\ K^{-1}}$; Haney, 1971; Barnier et al., 1995; Ma et al., 2016) relative to the inherited parent marine temperature $T_{\mathrm{atm},q(j)}$. When running 3D CRM or standalone GCRM with `enable_current_rel_wind = true`, air–sea fluxes use the current-relative wind $\mathbf{U}_{\mathrm{rel}}$ (Renault et al., 2016):

$$
\begin{aligned}
\bar{\psi}_{\mathrm{atm},i} &= \frac{\sum_{j:\,p(j)=i} A_{\mathrm{ocn},j}\,w_{\mathrm{wet},j}\,\psi_{\mathrm{ocn},j}}{\sum_{j:\,p(j)=i} A_{\mathrm{ocn},j}\,w_{\mathrm{wet},j}}, \\
\Delta E_{\mathrm{atm},q(j)} &= \rho_{\mathrm{sw}} c_{p,\mathrm{sw}} h_{\mathrm{ml,atm},q(j)}\,\Delta T_{\mathrm{ml},q(j)} - L_{f0}\,\Delta m_{\mathrm{ice},q(j)}, \\
T_{\mathrm{ocn},j}^{n+1} &= T_{\mathrm{ocn},j}^n + \frac{\Delta E_{\mathrm{rem},j}}{\rho_{\mathrm{sw}} c_{p,\mathrm{sw}} h_{\mathrm{ml},j}} - \left[1 - \exp\!\left(-\frac{\lambda_{\mathrm{as}}\,\Delta t}{\rho_{\mathrm{sw}} c_{p,\mathrm{sw}} h_{\mathrm{ml},j}}\right)\right]\big(T_{\mathrm{ocn},j}^n - T_{\mathrm{atm},q(j)}^n\big), \\
\mathbf{U}_{\mathrm{rel}} &= \mathbf{u}_a(z_1) - \mathbf{u}_o(z=0)
\end{aligned} \tag{9.8}
$$

### 9.6 Prognostic Surface Waves, Voronoi Swell Propagation, Stokes/Langmuir Mixing, Wave-Age Drag & Sea Spray (`src/aoba/ocean_waves.py`)

When `enable_wave_coupling = true`, significant wave height $H_s$ and peak period $T_p$ first undergo conservative Voronoi horizontal propagation of wave energy $E_w = H_s^2$ and energy-weighted period $P_w = E_w T_p$ across open-ocean edges ($w_{\mathrm{open},e} = f_{\mathrm{open},c_0} f_{\mathrm{open},c_1}$, vanishing at coastlines and sea-ice boundaries) at deep-water group velocity $c_g = g T_p / (4\pi)$, with resolution-aware diffusivity $K_{\mathrm{swell},e}$ ($L_{\mathrm{sw}} = 6\times 10^5\ \mathrm{m}$) and a cell-wise CFL-bounded effective conductance $G_e^{\mathrm{eff}}$ ensuring positivity and energy conservation across all ocean resolutions:

$$
\begin{aligned}
c_g &= \frac{g T_p}{4\pi}, \qquad K_{\mathrm{swell},e} = \frac{1}{2}\,c_{g,e}\,\max(d_{c,e},\ L_{\mathrm{sw}}), \\
G_e &= w_{\mathrm{open},e}\,K_{\mathrm{swell},e}\,\frac{d_{v,e}}{d_{c,e}}, \qquad G_e^{\mathrm{eff}} = \frac{G_e}{1 + 2.2\max(S_{c_0},\ S_{c_1})}, \qquad S_i = \frac{\Delta t}{A_i}\sum_{e\ni i} G_e
\end{aligned} \tag{9.9a}
$$

Following horizontal swell transport, $(H_s, T_p)$ relax via exact asymmetric exponential integration ($\tau_{\mathrm{grow}} = 6\ \mathrm{h}$, period-dependent swell decay $\tau_{\mathrm{decay}}(T_p) = 18\ \mathrm{h}\cdot\operatorname{clamp}((T_p/8)^{1.5}, 1, 4)$; Ardhuin et al., 2009) toward Pierson–Moskowitz equilibrium $(H_{s,\mathrm{eq}}, T_{p,\mathrm{eq}})$. Surface Stokes drift $u_s$ and turbulent Langmuir number $\mathrm{La}_t$ determine KPP enhancement $E_L$ (McWilliams & Sullivan, 2000; Takaya et al., 2010) and wave-age drag multiplier $\chi_{\mathrm{drag}}$ (Janssen, 1991):

$$
\begin{aligned}
H_{s,\mathrm{eq}} &= 0.0246\,U_{10}^2, \qquad T_{p,\mathrm{eq}} = 0.81\,U_{10}, \\
u_s &= \frac{\pi^3 H_s^2}{g T_p^3}, \qquad \mathrm{La}_t = \sqrt{\frac{u_{*w}}{u_s}}, \\
E_L &= \operatorname{clamp}\!\left(\sqrt{1 + \frac{0.08}{\mathrm{La}_t^4} + \frac{0.04}{\mathrm{La}_t^2}},\ 1.0,\ 2.2\right), \qquad \chi_{\mathrm{drag}} = \operatorname{clamp}\!\left(\left(\frac{25.0}{c_p/u_{*a}}\right)^{0.25},\ 0.85,\ 1.35\right)
\end{aligned} \tag{9.9}
$$

In high winds ($U_{10} > 12\ \mathrm{m\ s^{-1}}$), breaking-wave sea spray ($F_{\mathrm{spray}}$) contributes additional latent and sensible heat fluxes (Fairall et al., 2009; Andreas et al., 2015):

$$
\begin{aligned}
F_{\mathrm{spray}} &= \left(\frac{\max(U_{10} - 12,\ 0)}{20}\right)^{2.6} f_{\mathrm{open}}, \\
\Delta\mathrm{LH}_{\mathrm{spray}} &= 120\,F_{\mathrm{spray}}\,\frac{\Delta q}{0.006}, \\
\Delta\mathrm{SH}_{\mathrm{spray}} &= 25\,F_{\mathrm{spray}}\,\frac{\Delta T_{sa}}{2} - 0.25\,\Delta\mathrm{LH}_{\mathrm{spray}}
\end{aligned} \tag{9.10}
$$

### 9.7 Five-Category ITD & EVP Dynamical Sea Ice (`src/aoba/seaice_evp.py`)

When `enable_seaice_evp_itd = true`, sea-ice area and volume $(a_n, v_n)$ are prognosed across five thickness categories (boundaries $0, 0.6, 1.4, 2.4, 3.6, \infty\ \mathrm{m}$). After synchronizing thermodynamic mass changes, category rebinning and volume-conserving mechanical ridging ($A = \sum_n a_n > 1$, doubling thickness from category $n\to n+1$) are applied. Under Hibler (1979) compressive strength $P$ ($P^* = 2.75\times 10^4\ \mathrm{N\ m^{-2}}$, $C^* = 20$, yield-curve ratio $e = 2$), a fused GPU kernel runs $N_{\mathrm{EVP}} = 24$ elastic–viscous–plastic sub-cycles ($T_{\mathrm{damp}} = 0.36\Delta t$; Hunke & Dukowicz, 1997, Eqs. 16–17) coupled to a two-stage semi-implicit midpoint momentum solver:

$$
\begin{aligned}
P &= P^* V\,\exp\!\big(-C^*(1 - A)\big), \qquad \Delta = \sqrt{\dot{\varepsilon}_I^2 + e^{-2}\dot{\varepsilon}_{II}^2 + \Delta_{\min}^2}, \\
\sigma_1^{\mathrm{new}} &= \frac{\sigma_1 + \alpha_e\zeta(\dot{\varepsilon}_I - \Delta)}{1 + \alpha_e}, \qquad \sigma_s^{\mathrm{new}} = \frac{\sigma_s + \alpha_e\eta\dot{\varepsilon}_{II}}{1 + \alpha_e}
\end{aligned} \tag{9.11}
$$

$$
m_{\mathrm{ice}}\frac{\partial\mathbf{u}_{\mathrm{ice}}}{\partial t} = -m_{\mathrm{ice}} f\hat{\mathbf{k}}\times\mathbf{u}_{\mathrm{ice}} + A\boldsymbol{\tau}_a + A\rho_w C_w|\mathbf{u}_o - \mathbf{u}_{\mathrm{ice}}|(\mathbf{u}_o - \mathbf{u}_{\mathrm{ice}}) - m_{\mathrm{ice}} g\nabla\eta + \frac{1}{2}\nabla\sigma_1 \tag{9.12}
$$

All 11 sea-ice fields ($a_{0..4}, v_{0..4}, \mathrm{SWE}_{\mathrm{ice}}$) are advected conservatively by 2-stage SSP-RK2 (Heun, horizontal diffusivity $K_{\mathrm{ice}} = 250\ \mathrm{m^2\ s^{-1}}$), followed by post-advection ridging, rebinning, and $m_{\mathrm{ice}} = \rho_{\mathrm{ice}}\sum_n v_n$ synchronization.

### 9.8 FlexPFT Marine Biogeochemistry, OMIP/mocsy 2.0 Carbonate System & $\mathrm{O}_2$ / OMZ Denitrification (`src/aoba/ocean_bgc.py`)

When `enable_ocean_bgc = true`, seven (or eight with $\mathrm{O}_2$) 3D tracers ($\mathrm{NO}_3, P, Z, D_N, D_C, \mathrm{DIC}, \mathrm{Alk}, \mathrm{O}_2$) share the OGCM batched anomaly-form advection, rotated Redi diffusion, and increment implicit vertical mixing. Surface dissolved iron $\mathrm{Fe}_{\mathrm{sfc}}\ [\mathrm{nM}]$ is prognosed by relaxing mineral dust deposition ($3.5\%$ Fe content, $2\%$ solubility), pyrogenic wildfire black-carbon deposition ($4\%$ Fe/BC mass, $15\%$ solubility; Ito et al., 2019), and coastal riverine bioavailable iron delivery (Eq. 7.46b) against a 90-day scavenging timescale ($\tau_{\mathrm{Fe}} = 90\ \mathrm{d}$, background $\mathrm{Fe}_{\mathrm{bg}} = 0.25\ \mathrm{nM}$), coupling iron limitation $\mathrm{Fe}_{\mathrm{sfc}}/(\mathrm{Fe}_{\mathrm{sfc}}+0.20)$ into underwater PAR, while atmospheric $\mathrm{LNO}_x$, wildfire-volatilized nitrogen, and riverine nitrogen, carbon, and alkalinity enter surface $\mathrm{NO}_3$, $\mathrm{DIC}$, and $\mathrm{Alk}$ via Kahan summation:

$$
\begin{aligned}
S_{\mathrm{Fe}} &= \frac{1.253\times 10^4\,F_{\mathrm{dust,dep}} + 1.07\times 10^5\,F_{\mathrm{BC,dep}} + 10^6\,F_{\mathrm{Fe,riv}}}{h_0}, \\
\mathrm{Fe}_{\mathrm{sfc}}^{n+1} &= \mathrm{Fe}_{\mathrm{sfc}}^n + \big(\mathrm{Fe}_{\mathrm{bg}} + S_{\mathrm{Fe}}\tau_{\mathrm{Fe}} - \mathrm{Fe}_{\mathrm{sfc}}^n\big)\Big[-\operatorname{expm1}\!\left(-\frac{\Delta t}{\tau_{\mathrm{Fe}}}\right)\Big]
\end{aligned} \tag{9.12b}
$$

1. **FlexPFT Optimal Photoacclimation & Flexible N:C Quota (Pahlow et al., 2013; Smith et al., 2016; Sasai et al., 2022, Eqs. 2–7)**: Under chlorophyll self-shading ($k_d = 0.04 + 0.03\,\mathrm{Chl}$), layer-mean PAR $I_k$, and Arrhenius temperature scaling $F(T)$ ($E_a = 48\ \mathrm{kJ\ mol^{-1}}$), optimal chloroplast $\mathrm{Chl:C}$ ratio $\hat{\theta}$ (overflow-safe Lambert-$W_0(e^y)$), optimal cell N:C quota $Q$, specific growth rate $\mu_{\mathrm{fl}}$, and chlorophyll $\mathrm{Chl}$ are solved analytically:

$$
\begin{aligned}
\hat{\theta} &= \frac{1}{\zeta^{\mathrm{Chl}}} + \frac{V_0}{\alpha_I I}\Big[1 - W_0(e^y)\Big], \\
Q &= Q_s\left(1 + \sqrt{1 + \frac{1}{Q_s(V_L/\hat{V}^N + \zeta^N)}}\right), \\
\mu_{\mathrm{fl}} &= \frac{f_V\hat{V}^N}{Q}, \qquad \mathrm{Chl} = P\frac{\hat{\theta}}{Q}
\end{aligned} \tag{9.13}
$$

Holling Type III zooplankton grazing, linear and quadratic mortality, $\mathrm{CaCO}_3$ rain ratio $r_{\mathrm{CaCO_3}} = 0.07$ ($-2\ \mathrm{eq/mol\ CaCO_3}$ alkalinity), exponential detritus remineralization, and unconditionally stable implicit sinking ($w_s = 15\ \mathrm{m\ d^{-1}}$, instantaneous benthic remineralization for exact mass conservation) are integrated in a fused GPU kernel.

2. **OMIP / mocsy 2.0 Surface Carbonate Equilibrium & Wanninkhof (2014) Air–Sea $\mathrm{CO}_2$ Exchange (Orr & Epitalon, 2015; Orr et al., 2017)**: Using Lueker et al. (2000) $K_1, K_2$, Dickson (1990) $K_B$, Uppström (1974) total boron $B_T$, and Millero (1995) $K_W$, seven bounded Newton–Raphson iterations solve $A_T(x) = A_{\mathrm{obs}}$ for the scaled variable $x = [\mathrm{H}^+]\times 10^8$ to avoid `float32` underflow, combined with Weiss (1974) / Weiss & Price (1980) solubility and Wanninkhof (2014) gas-transfer velocity $k_w$:

$$
\begin{aligned}
A_T(x) &= C_T\frac{K_{1s} x + 2 K_{1s} K_{2s}}{x^2 + K_{1s} x + K_{1s} K_{2s}} + B_T\frac{K_{Bs}}{x + K_{Bs}} + \left(\frac{K_{Ws}}{x} - x\right)\times 10^{-8}, \qquad \mathrm{pH}_T = 8 - \log_{10} x, \\
k_w &= (1 - f_{\mathrm{ice}})\cdot 6.97\times 10^{-7}\,U_{10}^2\sqrt{\frac{660}{\mathrm{Sc}_{\mathrm{CO_2}}}}, \qquad f_{g\mathrm{CO_2}} = k_w\big([\mathrm{CO}_2]_{\mathrm{sat}} - [\mathrm{CO}_2^*]\big)
\end{aligned} \tag{9.14}
$$

3. **3D Dissolved Oxygen $\mathrm{O}_2$ & OMZ Anaerobic Denitrification**: When `enable_ocean_o2_omz = true`, $\mathrm{O}_2$ is prognosed using Garcia & Gordon (1992) saturation solubility and Anderson & Sarmiento (1994) stoichiometry $R_{\mathrm{O_2:C}} = 138/106$. Inside oxygen minimum zones ($\mathrm{O}_2 < 20\ \mathrm{mmol\ m^{-3}}$), remineralization transitions smoothly from aerobic respiration to nitrate-reducing anaerobic denitrification ($R_{\mathrm{denit}} = 0.8\ \mathrm{mol\ N / mol\ O_2}$, $K_{\mathrm{O_2}} = 5.0\ \mathrm{mmol\ m^{-3}}$), coupling to marine $\mathrm{N}_2\mathrm{O}$ emissions (Section 10.6):

$$
\begin{aligned}
\frac{\partial \mathrm{O}_2}{\partial t} &= \mathcal{T}(\mathrm{O}_2) + R_{\mathrm{O_2:C}}\big(\mathrm{NPP} - R_{\mathrm{remin,aer}}\big) + \frac{k_w^{\mathrm{O_2}}(1-f_{\mathrm{ice}})}{h_0}\big(\mathrm{O}_{2,\mathrm{sat}} - \mathrm{O}_2\big)\delta_{k,0}, \\
f_{\mathrm{denit}} &= \frac{1}{1 + (\mathrm{O}_2/5.0)^2}
\end{aligned} \tag{9.15}
$$

To close the centennial-to-millennial marine fixed-nitrogen budget against OMZ denitrification, warm ($T_c > 18\ ^\circ\mathrm{C}$), nitrate-depleted, iron-modulated surface marine diazotrophic $\mathrm{N}_2$ fixation (Monteiro et al., 2011; Landolfi et al., 2017) adds $\Delta\mathrm{NO}_{3,\mathrm{fix}}$ to Layer 0 with compensating nitrification alkalinity change $-\Delta\mathrm{NO}_{3,\mathrm{fix}}$:

$$
\Delta\mathrm{NO}_{3,\mathrm{fix}} = r_{\mathrm{fix},\max}\,\frac{\Delta t}{86400}\left[\operatorname{clamp}\!\left(\frac{T_{c,0} - 18}{10},\ 0,\ 1\right)\right]^2 \frac{K_{N,\mathrm{inh}}}{K_{N,\mathrm{inh}} + \mathrm{NO}_{3,0}}\,\frac{\mathrm{Fe}_{\mathrm{sfc}}}{\mathrm{Fe}_{\mathrm{sfc}} + 0.20}\,(1 - f_{\mathrm{ice}}) \tag{9.15b}
$$

with $r_{\mathrm{fix},\max} = 0.012\ \mathrm{mmol\ N\ m^{-3}\ d^{-1}}$ and $K_{N,\mathrm{inh}} = 1.5\ \mathrm{mmol\ N\ m^{-3}}$.

---

## 10. Atmospheric Chemistry (`o3_mode`), Natural/Volcanic Aerosols, Carbon & Trace-Gas Cycles (`co2_mode`) & Water Isotopes

### 10.1 Overview of Material Cycles, Trace Gases & Isotope Coupling

Atmospheric ozone, natural and volcanic aerosols, closed-loop carbon, methane, and nitrous-oxide cycles, and 3D stable water isotopes ($\mathrm{H}_2^{18}\mathrm{O}$, $\mathrm{HDO}$) are bidirectionally coupled with SCVT FCT transport, radiative transfer, cloud microphysics, and terrestrial/marine biogeochemistry.

![Figure 10.1: Closed-Loop Material Cycles, Water Isotopes, Dynamic Ice Sheet & Planetary Presets](assets/fig_07_biogeochem_planetary.svg)

### 10.2 Atmospheric Ozone: Selectable Modes (`o3_mode`) & 3D Photochemistry (`src/aoba/ozone_forcing.py`, `src/aoba/ozone_chemistry.py`)

`[forcing] o3_mode` selects among four ozone representations.

| `o3_mode` | Dimensionality | Physics / Data Source | Advantages | Limitations & Considerations |
| :--- | :--- | :--- | :--- | :--- |
| `"rcemip"` | 1D ($p$) | Wing et al. (2018) gamma analytic profile | Zero computational overhead; exact compliance with RCEMIP intercomparison specifications. | Omits latitudinal gradients, seasonality, and polar-vortex ozone variability. |
| `"latitude_seasonal"` *(default)* | 2D ($\phi, p, t$) | Map-free latitude–season Brewer–Dobson analytic profile | Requires no external files, incurs near-zero cost, reproduces mid-/high-latitude springtime ozone maxima, and generalizes to paleoclimate or idealized experiments. | Lacks interactive feedbacks with dynamical transport, volcanic aerosols, PSCs, and $\mathrm{CH}_4/\mathrm{LNO}_x$ precursors. |
| `"climatology"` | 2D zonal monthly mean | External CMIP6 / CMIP7 input4MIPs NetCDF | Matches standard historical and CMIP forcing protocols directly. | Requires external datasets and remains fixed under non-modern orbits or climate reorganizations. |
| `"prognostic"` | 3D ($\mathbf{x}, z, t$) | 3D FCT + Chapman UV + $\mathrm{NO}_x/\mathrm{HO}_x$ + PSC + tropospheric precursors + dry deposition + volcanic/solar response | Self-consistently predicts Brewer–Dobson transport, polar ozone depletion, volcanic/solar responses, and planetary atmospheres. | Adds 3D tracer transport and photochemical compute overhead; stratospheric temperature biases directly affect PSC activation. |

In `o3_mode = "latitude_seasonal"`, peak-pressure altitude $g_3(\phi)$ and seasonal phase $f_{\mathrm{seas}}(\phi, \mathrm{DOY})$ capture poleward Brewer–Dobson transport and springtime maxima:

$$
\begin{aligned}
g_3(\phi) &= 9.5 + 6.5\sin^2\phi, \\
f_{\mathrm{seas}}(\phi, \mathrm{DOY}) &= 1 + 0.16\sin\phi\cos\!\left(\frac{2\pi(\mathrm{DOY} - 85)}{365.2425}\right)
\end{aligned} \tag{10.1}
$$

In `o3_mode = "prognostic"` (fused GPU kernel `ozone_chem_col`), slant-column paths $(S_{\mathrm{O_2}}, S_{\mathrm{O_3}})$ along spherical air-mass factor $\mathrm{AMF} = 35/\sqrt{1224\mu_0^2 + 1}$ yield Herzberg, Schumann–Runge, Hartley, Chappuis, and UV-B photolysis rates $(J_{\mathrm{O_2}}, J_{\mathrm{O_3}}, J_{\mathrm{O(^1D)}})$. Combined with JPL Evaluation 19 (Burkholder et al., 2019) Chapman kinetics $(k_2, k_3)$, stratospheric $\mathrm{NO}_x$ (coupled to $\mathrm{N}_2\mathrm{O}$), $\mathrm{HO}_x$ (coupled to $q_v$), polar stratospheric cloud (PSC) loss ($T < 195\ \mathrm{K}$, scaling as $(\mathrm{eCl}/3.5)^2$), tropospheric $\mathrm{CH}_4/\mathrm{LNO}_x/\text{fire}/\text{BVOC}$ precursor production (coupling biogenic VOC emissions via $F_{\mathrm{prec}} = F_{\mathrm{fire,C}} + 0.5\,F_{\mathrm{SOA}}$) and $\mathrm{O}(^1D)+\mathrm{H}_2\mathrm{O}$ loss, and Wesely (1989) dry deposition ($0.0005$ ocean, $0.0008$ bare soil, $0.0018$ cuticle, $0.0042\,\mu_0\ \mathrm{m\ s^{-1}}$ stomatal), the odd-oxygen Riccati equation is integrated via its cancellation-free positive root $\chi_{\mathrm{eq}}$ and exact exponential relaxation:

$$
\begin{aligned}
\frac{d\chi_{\mathrm{O_3}}}{dt} &= P - L_1\chi_{\mathrm{O_3}} - L_2\chi_{\mathrm{O_3}}^2, \qquad \chi_{\mathrm{eq}} = \frac{2P}{L_1 + \sqrt{L_1^2 + 4 P L_2}}, \\
\chi_{\mathrm{O_3}}^{n+1} &= \chi_{\mathrm{O_3}}^n + (\chi_{\mathrm{eq}} - \chi_{\mathrm{O_3}}^n)\Big[-\operatorname{expm1}\!\Big(-\big(L_1 + L_2(\chi_{\mathrm{O_3}}^n + \chi_{\mathrm{eq}})\big)\Delta t\Big)\Big]
\end{aligned} \tag{10.2}
$$

Pair-conservative implicit vertical PBL mixing over the lowest six layers ($h_{\mathrm{PBL}} = 1500\ \mathrm{m}$) and exponential stratospheric perturbations from volcanic sulfate surface-area density $\mathrm{SAD}$ (Solomon et al., 1996) and the 11-year solar UV multiplier $s_{\mathrm{UV}}$ follow immediately.

### 10.3 Map-Free Mineral Dust (`src/aoba/dust.py`)

Without any prescribed dust-erodibility map ($S \equiv 1$), DEAD / CLM5 Ch. 31 physical emission (Zender et al., 2003; Lawrence et al., 2020) is evaluated on each MAML land tile. The Iversen & White (1982) dry threshold friction velocity $u_{*,t,\mathrm{dry}}$ is scaled by the Fécan et al. (1999) soil-moisture inhibition factor $f_w$ ($w_t = 0.198$). Using Owen (1964) saltation-enhanced friction velocity $u_{*,s}$, White (1979) horizontal saltation flux $Q_s$, Marticorena & Bergametti (1995) sandblasting efficiency $\alpha = 0.04786\ \mathrm{m^{-1}}$, three-lognormal transported mass fraction $\sum M_{i,j} = 0.8712$, bare snow-free unfrozen fraction $f_m$, and global tuning coefficient $T_{\mathrm{emis}}$ (default `dust_emis_factor = 2.5e-6`), vertical dust emission is:

$$
\begin{aligned}
u_{*,t,\mathrm{dry}} &= \frac{0.22655}{\sqrt{\rho}}, \qquad u_{*,t} = u_{*,t,\mathrm{dry}}\,f_w, \qquad u_{*,s} = u_* + 0.003\big(\max(u_{10} - u_{10,t},\ 0)\big)^2, \\
Q_s &= 2.61\frac{\rho u_{*,s}^3}{g}\left(1 - \frac{u_{*,t}}{u_{*,s}}\right)\left(1 + \frac{u_{*,t}}{u_{*,s}}\right)^2, \\
F_{\mathrm{dust}} &= T_{\mathrm{emis}}\,f_m\,\alpha\,Q_s\sum M_{i,j}
\end{aligned} \tag{10.3}
$$

In the atmospheric column, total mobilized dust $F_{\mathrm{dust}}$ is partitioned using the CLM5 three-lognormal source distribution (Table 31.1) into two independently advected SCVT FCT size modes: a long-range fine mode $q_{\mathrm{dust,fine}}$ ($0.1\text{–}2.5\ \mathrm{\mu m}$, emission share $f_{\mathrm{fine}} = 0.208$) and a rapidly settling coarse mode $q_{\mathrm{dust,coarse}}$ ($2.5\text{–}10\ \mathrm{\mu m}$, emission share $f_{\mathrm{coarse}} = 0.792$), with total dust mass $q_{\mathrm{dust}} = q_{\mathrm{dust,fine}} + q_{\mathrm{dust,coarse}}$. Each mode is integrated by the fused column GPU kernel for lowest-layer injection, pair-conservative implicit PBL mixing ($h_{\mathrm{PBL}} = 1200\ \mathrm{m}$), top-down implicit gravitational settling and dry deposition, and exact exponential wet scavenging:

$$
\begin{aligned}
F_{\mathrm{dust,fine}} &= f_{\mathrm{fine}}\,F_{\mathrm{dust}}, \qquad F_{\mathrm{dust,coarse}} = f_{\mathrm{coarse}}\,F_{\mathrm{dust}}, \\
q_{\mathrm{dust}} &= q_{\mathrm{dust,fine}} + q_{\mathrm{dust,coarse}}, \qquad \mathrm{AOD}_{\mathrm{dust}} = \sum_k \rho_{d,k}\Delta z_k \left(1100\,q_{\mathrm{dust,fine},k} + 450\,q_{\mathrm{dust,coarse},k}\right)
\end{aligned} \tag{10.3a}
$$

| Mode | Diameter Range | Emission Share | $v_{\mathrm{settle}}\ [\mathrm{m\ s^{-1}}]$ | $v_{\mathrm{dep}}\ [\mathrm{m\ s^{-1}}]$ | $\alpha_{\mathrm{wet}}\ [\mathrm{m^2\ kg^{-1}}]$ | $\mathrm{MEE}_{550}\ [\mathrm{m^2\ kg^{-1}}]$ |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| Fine ($q_{\mathrm{dust,fine}}$) | $0.1\text{–}2.5\ \mathrm{\mu m}$ | $0.208$ | $5.0\times 10^{-4}$ | $1.5\times 10^{-3}$ | $0.40$ | $1100$ |
| Coarse ($q_{\mathrm{dust,coarse}}$) | $2.5\text{–}10.0\ \mathrm{\mu m}$ | $0.792$ | $8.0\times 10^{-3}$ | $1.2\times 10^{-2}$ | $0.65$ | $450$ |

### 10.4 Four Natural Aerosols: Sea Salt, DMS Sulfate, Wildfire BC/OC & BVOC SOA (`src/aoba/aerosols.py`)

Four 3D aerosol tracers ($q_{\mathrm{ss}}, q_{\mathrm{so4}}, q_{\mathrm{bc}}, q_{\mathrm{oc}}$) advance via SCVT FCT transport and a fused 4-species column GPU kernel. Sea-salt emission $F_{\mathrm{ss}}$ (Jaeglé et al., 2011; Gong, 2003), marine DMS-derived non-sea-salt sulfate $F_{\mathrm{SO_4}}$ (Kloster et al., 2006; Wanninkhof, 2014; diagnosing $[\mathrm{DMS}]_w$ from FlexPFT chlorophyll with oxidation yield $Y_{\mathrm{SO_4}} = 0.75$), and forest BVOC-derived secondary organic aerosol $F_{\mathrm{SOA}}$ (Guenther et al., 2012 MEGAN; added to OC) are given below, while wildfire BC and OC are injected directly from SPITFIRE:

$$
\begin{aligned}
f_{\mathrm{SST}}(T_c) &= 0.3 + 0.1\,T_c - 0.0076\,T_c^2 + 0.00021\,T_c^3, \\
F_{\mathrm{ss}} &= f_{\mathrm{open}}\cdot 2.5\times 10^{-13}\,f_{\mathrm{SST}}(T_c)\,u_{10}^{3.41}, \\
F_{\mathrm{SOA}} &= \mathbb{I}_{\mathrm{land}}\cdot 3.2\times 10^{-11}\big(1 - e^{-0.5\mathrm{LAI}}\big)e^{0.09(T_c - 30)}\left(0.35 + 0.65\,\frac{\mathrm{PAR}}{\mathrm{PAR} + 120}\right)
\end{aligned} \tag{10.4}
$$

| Species | $v_{\mathrm{settle}}\ [\mathrm{m\ s^{-1}}]$ | $v_{\mathrm{dry}}\ [\mathrm{m\ s^{-1}}]$ | $\alpha_{\mathrm{wet}}\ [\mathrm{m^2\ kg^{-1}}]$ | $\mathrm{MEE}_{550,\mathrm{dry}}\ [\mathrm{m^2\ kg^{-1}}]$ | $\gamma_{\mathrm{RH}}$ | $f_{\max}$ | $\omega_{\mathrm{SW,dry}}$ | $g_{\mathrm{SW,dry}}$ | $k_{\mathrm{abs,LW}}\ [\mathrm{m^2\ kg^{-1}}]$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sea Salt (SS) | $4\times 10^{-3}$ | $5\times 10^{-3}$ | $0.80$ | $1500$ | $0.52$ | $4.5$ | $0.99$ | $0.76$ | $80$ |
| Sulfate ($\mathrm{SO}_4$) | $4\times 10^{-4}$ | $1.5\times 10^{-3}$ | $0.60$ | $3800$ | $0.40$ | $3.2$ | $0.99$ | $0.70$ | $120$ |
| Black Carbon (BC) | $2\times 10^{-4}$ | $1.0\times 10^{-3}$ | $0.30$ | $10000$ | $0.00$ | $1.0$ | $0.25$ | $0.35$ | $450$ |
| Organic Carbon (OC/SOA) | $2\times 10^{-4}$ | $1.0\times 10^{-3}$ | $0.50$ | $3200$ | $0.20$ | $1.9$ | $0.92$ | $0.70$ | $180$ |

Hygroscopic water uptake scales shortwave extinction for sea salt, sulfate, and organic carbon via $f_i(\mathrm{RH})$ (Chin et al., 2002; Tang & Munkelwitz, 1994), adding pure scattering ($g_w = 0.78$) while holding dry absorption constant:

$$
f_i(\mathrm{RH}) = \operatorname{clamp}\!\left(\big(1 - \operatorname{clamp}(\mathrm{RH},\ 0,\ 0.95)\big)^{-\gamma_i},\ 1.0,\ f_{\max,i}\right), \qquad \tau_{\mathrm{ext},i} = \rho_d\Delta z\,q_i\,\mathrm{MEE}_{550,\mathrm{dry},i}\,f_i(\mathrm{RH}) \tag{10.4a}
$$

Wildfire BC and OC emissions are lofted uniformly across $[0, z_{\mathrm{plume}}]$ according to Byram fireline intensity $I_B\ [\mathrm{kW\ m^{-1}}]$ (Freitas et al., 2007; Val Martin et al., 2012), while biogenic SOA enters the lowest layer:

$$
z_{\mathrm{plume}} = \begin{cases} 0, & I_B \le 10\ \mathrm{kW\ m^{-1}}, \\ \operatorname{clamp}\!\left(300 + 220\sqrt{I_B / 100},\ 300,\ 3500\right)\ \mathrm{m}, & I_B > 10\ \mathrm{kW\ m^{-1}} \end{cases} \tag{10.4b}
$$

Surface hygroscopic concentrations diagnose $N_{\mathrm{CCN}}$ for two-moment microphysics, and exact column mass-budget closure yields the BC surface deposition flux $D_{\mathrm{BC}}$ for SNICAR snow darkening.

### 10.5 Stratospheric Volcanic Aerosols, SAI & 11-Year Solar Cycle (`src/aoba/volcanic_solar.py`)

When `enable_volcanic_aerosols = true`, discrete eruption pulses (Gaussian latitude distribution) or continuous SAI injections of stratospheric $\mathrm{SO}_2$ ($\tau_{\mathrm{ox}} = k_1^{-1} = 35\ \mathrm{d}$) and sulfuric-acid aerosol $\mathrm{H}_2\mathrm{SO}_4$ ($\tau_{\mathrm{set}} = k_2^{-1} = 1.10\ \mathrm{yr}$, mass ratio $\alpha_m = 98.08/64.066$) are integrated by the exact two-stage Bateman analytic solution, followed by Brewer–Dobson horizontal diffusion ($K_h \le 6\times 10^5\ \mathrm{m^2\ s^{-1}}$) and log-pressure vertical distribution centered at $40\ \mathrm{hPa}$ to produce optical properties ($\mathrm{MEE}_{\mathrm{SW}} = 3400$, $\omega = 0.998$, $g = 0.72$, $k_{\mathrm{LW}} = 135$) and surface-area density $\mathrm{SAD}$. When `enable_solar_cycle = true`, the 11-year Schwabe cycle (Lean et al., 2005; default amplitude $A_{\mathrm{TSI}} = 1.0\ \mathrm{W\ m^{-2}}$) modulates irradiance $\Delta\mathrm{TSI}$ and stratospheric UV photolysis $s_{\mathrm{UV}}$:

$$
\begin{aligned}
S^{n+1} &= S^n e^{-k_1\Delta t}, \qquad A^{n+1} = A^n e^{-k_2\Delta t} + S^n\,\alpha_m\frac{k_1}{k_1 - k_2}\left(e^{-k_2\Delta t} - e^{-k_1\Delta t}\right), \\
\Delta\mathrm{TSI} &= 0.5\,A_{\mathrm{TSI}}\sin\varphi, \qquad s_{\mathrm{UV}} = 1 + 0.015\,A_{\mathrm{TSI}}\sin\varphi
\end{aligned} \tag{10.5}
$$

### 10.6 Atmospheric $\mathrm{CO}_2$ (`co2_mode`), Prognostic $\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$ & Silicate Weathering (`co2_forcing.py`, `prognostic_co2.py`, `trace_gas_cycle.py`, `planetary_geochem.py`)

`[forcing] co2_mode` selects among four atmospheric $\mathrm{CO}_2$ treatments.

| `co2_mode` | Behavior | Advantages | Limitations & Considerations |
| :--- | :--- | :--- | :--- |
| `"constant"` *(default)* | Fixed mixing ratio `co2_ppmv` | Holds radiative forcing strictly constant; suited for equilibrium climate sensitivity and pre-industrial spin-ups. | Carbon-cycle fluxes do not feed back onto atmospheric $\mathrm{CO}_2$. |
| `"scenario"` | `float64` linear interpolation of 25 built-in scenarios (NOAA, CMIP5 RCP, CMIP6 SSP, 800 kyr Antarctic ice-core records) | Enables immediate concentration-driven CMIP, historical, and glacial-cycle experiments. | Concentrations remain externally prescribed without interactive carbon-cycle feedback. |
| `"csv"` | Linear interpolation of a user-supplied `(year, co2_ppmv)` CSV table | Supports custom paleoclimate trajectories and step/pulse concentration protocols. | Requires the user to provide a valid CSV file. |
| `"prognostic"` | Emissions-driven closed-loop global carbon budget (optionally coupled to 3D FCT transport) | Captures full carbon–climate feedbacks (C4MIP) among terrestrial $\mathrm{NBP}$, ocean $f_{g\mathrm{CO_2}}$, silicate weathering, and volcanic/anthropogenic emissions. | Unspun-up land or ocean carbon pools induce initial $\mathrm{CO}_2$ drift, requiring prior equilibrium spin-up. |

In `co2_mode = "prognostic"`, the global carbon budget (conversion mass $M_{\mathrm{ppmv}} = 2.124\ \mathrm{Pg\ C\ ppmv^{-1}}$) is integrated conservatively in `float64`; the optional 3D $\mathrm{CO}_2$ tracer applies surface flux injection, implicit PBL mixing, and a global mass-locking correction:

$$
\frac{d[\mathrm{CO}_2]_{\mathrm{glob}}}{dt} = \frac{E_{\mathrm{anthro}} + E_{\mathrm{volc}} - \mathrm{NBP}_{\mathrm{land}} - F_{\mathrm{ocean}} - F_{\mathrm{weath}}}{2.124} \tag{10.6}
$$

When `enable_prognostic_ch4_n2o = true`, anaerobic wetland methanogenesis (coupled to $R_h$, soil saturation excess, and dynamic wetland fraction), wildfire, and geological emissions ($20\ \mathrm{Tg\ CH_4\ yr^{-1}}$) are balanced against temperature-, ozone-, and self-feedback-dependent OH oxidation plus soil uptake ($120\ \mathrm{yr}^{-1}$); terrestrial denitrification/$R_h$ and marine POC/OMZ denitrification $\mathrm{N}_2\mathrm{O}$ emissions are balanced against ozone-dependent stratospheric photolysis (base lifetime $116\ \mathrm{yr}$) via exact exponential updates ($M_{\mathrm{CH_4}} = 2844\ \mathrm{Tg\ CH_4\ ppmv^{-1}}$, $M_{\mathrm{N_2O}} = 4966\ \mathrm{Tg\ N\ ppmv^{-1}}$):

$$
\begin{aligned}
k_{\mathrm{OH}} &= \frac{1}{9.5}\,e^{0.032(\bar{T}_s - 287)}\left(\frac{300}{\overline{\mathrm{TCO}}}\right)^{0.35}\left(\frac{[\mathrm{CH}_4]}{0.80825}\right)^{-0.32}, \\
k_{\mathrm{N_2O}} &= \frac{1}{116.0}\left(\frac{300}{\overline{\mathrm{TCO}}}\right)^{0.25}
\end{aligned} \tag{10.7}
$$

When `enable_silicate_weathering = true`, Walker et al. (1981) snow-free land silicate weathering $F_{w,i}$ is integrated against volcanic outgassing $V$ via an implicit linearized exponential scheme with optional geological acceleration, supplying $2\ \mathrm{mol\ eq / mol\ C}$ of riverine alkalinity to marine BGC:

$$
F_{w,i} = \mathbb{I}_{\mathrm{land}}(1 - f_{\mathrm{snow}})\,K_{\mathrm{base}}\,\exp\!\left(\frac{T_s - 288.15}{13.7}\right)\left(\frac{q_{\mathrm{runoff}}}{10^{-5}}\right)^{0.65}\left(\frac{[\mathrm{CO}_2]}{284.3}\right)^{0.30} \tag{10.8}
$$

### 10.7 Prognostic 3D Stable Water Isotopes ($\mathrm{H}_2^{18}\mathrm{O}$, $\mathrm{HDO}$, `src/aoba/water_isotopes.py`)

When `enable_water_isotopes = true`, VSMOW-normalized 3D heavy vapor fields $(q_v^{18}, q_v^{\mathrm{D}})$ are advected by SCVT FCT, diagnosing per-mil delta notation $\delta_x$ [‰] and deuterium excess $d$. Liquid–vapor equilibrium fractionation follows Majoube (1971); ice–vapor equilibrium fractionation (Majoube, 1971; Merlivat & Nief, 1967) is combined with Jouzel & Merlivat (1984) supersaturation ($S_i$) kinetic fractionation (diffusivity ratios $(D/D')_{18}=1.0285$, $(D/D')_{\mathrm{D}}=1.0251$) and blended linearly over $253.15\text{–}273.15\ \mathrm{K}$:

$$
\begin{aligned}
\delta_x &= (R_x - 1)\times 1000, \qquad d = \delta\mathrm{D} - 8\,\delta^{18}\mathrm{O}, \\
\ln\alpha_{\ell,18} &= \frac{1137.0}{T^2} - \frac{0.4156}{T} - 0.0020667, \qquad \ln\alpha_{\ell,\mathrm{D}} = \frac{24844.0}{T^2} - \frac{76.248}{T} + 0.052612
\end{aligned} \tag{10.9}
$$

$$
\begin{aligned}
S_i &= 1 + 0.003\,\operatorname{clamp}(273.15 - T,\ 0,\ 60), \\
\alpha_{i,x} &= \alpha_{i,x}^{\mathrm{eq}}\,\frac{S_i}{1 + \alpha_{i,x}^{\mathrm{eq}}(S_i - 1)(D/D')_x}, \\
R_{\mathrm{evap,ocn},x} &= \frac{\alpha_{\ell,\mathrm{sfc},x}^{-1} - h R_{0,x}}{1 - h}(1 - k_x)
\end{aligned} \tag{10.10}
$$

Ocean evaporation applies Craig & Gordon (1965) and Merlivat & Jouzel (1979) wind-speed-dependent kinetic fractionation switching from the smooth regime ($k_{18}=0.0060$ for $U_{10} < 7\ \mathrm{m\ s^{-1}}$) to the rough regime ($k_{18}=\operatorname{clamp}(0.000285\,U_{10}+0.00082, 0.0028, 0.012)$ for $U_{10} \ge 7\ \mathrm{m\ s^{-1}}$) with $k_{\mathrm{D}} = 0.88\,k_{18}$. Terrestrial evapotranspiration partitions into non-fractionating transpiration ($1 - f_{\mathrm{bare}}$) and Craig–Gordon fractionating bare-soil evaporation ($f_{\mathrm{bare}} = \exp(-0.5\,\mathrm{LAI})$, $k_{18,\mathrm{soil}} = 0.0120$). Condensation depletion uses the exact analytic Rayleigh distillation update, and snowfall $\delta^{18}\mathrm{O}_{\mathrm{snow}}, \delta\mathrm{D}_{\mathrm{snow}}$ is accumulated mass-weighted for ice-core comparison:

$$
\begin{aligned}
f_{\mathrm{rem}} &= \exp\!\left(-\frac{(P_{\mathrm{rain}} + P_{\mathrm{snow}})\Delta t}{\mathrm{CWV}}\right), \\
R_{k,x}^{n+1} &= R_{k,x}^n\,f_{\mathrm{rem}}^{\alpha_{\mathrm{eff},k,x} - 1}
\end{aligned} \tag{10.11}
$$

Liquid rain falling through unsaturated sub-cloud air undergoes Stewart (1975) / Risi et al. (2008) partial re-evaporation ($f_{\mathrm{drop}} = 1 - f_{\mathrm{reevap}}$, $\kappa_x = (D/D')_x^{0.58}$, $P_{\mathrm{ref}} = 1.5\times 10^{-4}\ \mathrm{kg\ m^{-2}\ s^{-1}}$) and mass-conserving return of depleted vapor to the lowest layer, reproducing the tropical amount effect and sub-cloud deuterium-excess reduction:

$$
\begin{aligned}
f_{\mathrm{reevap}} &= \frac{P_{\mathrm{rain}}}{P_{\mathrm{rain}} + P_{\mathrm{snow}}}\,\operatorname{clamp}\!\left(\frac{0.35(1 - h_{\mathrm{sub}})}{1 + P_{\mathrm{rain}}/P_{\mathrm{ref}}},\ 0,\ 0.45\right), \\
\gamma_x &= \frac{\alpha_{\ell,0,x}\,h_{\mathrm{sub}}}{1 - \alpha_{\ell,0,x}\kappa_x(1 - h_{\mathrm{sub}})}, \qquad \beta_x = \frac{1 - \alpha_{\ell,0,x}\kappa_x(1 - h_{\mathrm{sub}})}{\alpha_{\ell,0,x}\kappa_x(1 - h_{\mathrm{sub}})}, \\
R_{\mathrm{drop},x} &= \gamma_x R_{0,x} + (R_{\mathrm{cond},x} - \gamma_x R_{0,x})\,f_{\mathrm{drop}}^{\beta_x}
\end{aligned} \tag{10.12}
$$

---

## 11. Dynamic Ice-Sheet Model, Milankovitch Orbital Evolution & General Planetary Presets

### 11.1 Glen ($n=3$) SIA Ice Sheet, Glacial Isostasy & Hydrostatic Vertical Remapping (`src/aoba/icesheet.py`)

When `enable_dynamic_icesheet = true`, annual surface mass balance $\mathrm{SMB}$ is exponentially smoothed (default $\tau_{\mathrm{EMA}} = 5\ \mathrm{yr}$) and integrated over accelerated ice-sheet time $\Delta t_{\mathrm{ice}} = \alpha_{\mathrm{acc}}\Delta t_{\mathrm{clim}}$ (default $\alpha_{\mathrm{acc}} = 100$) using an upwind finite-volume Shallow Ice Approximation (SIA) with Glen's flow law ($n=3$, $A_{\mathrm{Glen}} = 10^{-16}\ \mathrm{Pa^{-3}\ yr^{-1}}$; Mahaffy, 1976; Bueler et al., 2005) over up to 256 sub-cycles, with immediate marine calving over ocean cells:

$$
\begin{aligned}
\Gamma_{\mathrm{SIA}} &= \frac{2}{5}\,A_{\mathrm{Glen}}(\rho_i g)^3, \qquad D_e = \min\!\left(\Gamma_{\mathrm{SIA}} H_{\mathrm{up},e}^5 \left(\frac{s_{c_1}-s_{c_0}}{d_{c,e}}\right)^2,\ D_{\mathrm{cap}}\right), \\
H^{m+1} &= \operatorname{clamp}\!\Big(H^m + \Delta t_{\mathrm{sub}}\big(\nabla_{\mathrm{SCVT}}\cdot(D_e\nabla s) + b_{\mathrm{ice}}\big),\ 0,\ 6000\Big)
\end{aligned} \tag{11.1}
$$

Bedrock elevation $b$ relaxes exponentially ($\tau_{\mathrm{iso}} = 1000\ \mathrm{yr}$) toward local isostatic equilibrium $b_{\mathrm{eq}}$ with mantle density $\rho_m = 3300\ \mathrm{kg\ m^{-3}}$ (Pollard & DeConto, 2012):

$$
\begin{aligned}
b_{\mathrm{eq}} &= b_0 - \frac{\rho_i}{\rho_m}\,H, \\
b^{n+1} &= b^n + \left(1 - e^{-\Delta t_{\mathrm{ice,yr}}/\tau_{\mathrm{iso}}}\right)(b_{\mathrm{eq}} - b^n)
\end{aligned} \tag{11.2}
$$

Crossing the threshold $H_{\mathrm{thr}} = 5.0\ \mathrm{m}$ automatically transitions `SURF_LAND` $\leftrightarrow$ `SURF_ICESHEET`. Whenever two-pass smoothed surface elevation changes exceed $0.05\ \mathrm{m}$, `remap_atmosphere_and_crm_vertical` rebuilds the STF vertical coordinate, linearly interpolates $\ln p, \theta_m, q$, and winds, and applies hydrostatic isentropic downward extrapolation where deglaciation exposes lower altitudes:

$$
\begin{aligned}
p_{\mathrm{extrap}} &= p_0\left[1 - \frac{g(z_{\mathrm{new}} - z_{\mathrm{old},0})}{c_{pd}\theta_{m,0}\Pi_0}\right]^{1/\kappa_d}, \\
\rho_d &= \frac{p_{\mathrm{new}}}{R_d\theta_{m,\mathrm{new}}\Pi_{\mathrm{new}}}
\end{aligned} \tag{11.3}
$$

### 11.2 Analytic Milankovitch Orbital Evolution (`src/aoba/planetary_geochem.py`)

When `enable_milankovitch_orbit = true`, eccentricity $e(t)$ (periods $405.0, 94.9, 123.8, 98.9, 131.2\ \mathrm{kyr}$), obliquity $\varepsilon(t)$ (periods $41.00, 39.73, 53.62, 40.52, 28.91\ \mathrm{kyr}$), and longitude of perihelion $\varpi(t)$ (period $21.74\ \mathrm{kyr}$) evolve continuously from J2000 time $t\ [\mathrm{kyr}]$ (starting at `milankovitch_start_kyr`, e.g., `-21.0` for LGM) via Berger (1978) / Laskar et al. (2004) five-term trigonometric series:

$$
\begin{aligned}
e(t) &= 0.0278 + \sum_{j=1}^5 A_{e,j}\cos\!\left(\frac{2\pi t}{P_{e,j}} + \phi_{e,j}\right), \\
\varepsilon(t) &= 23.320^\circ + \sum_{j=1}^5 A_{\varepsilon,j}\cos\!\left(\frac{2\pi t}{P_{\varepsilon,j}} + \phi_{\varepsilon,j}\right), \\
\varpi(t) &= \left(283.0^\circ + 360^\circ\frac{t}{21.74}\right)\bmod 360^\circ
\end{aligned} \tag{11.4}
$$

### 11.3 General Planetary & Exoplanet Presets (`planet_preset`, `src/aoba/planetary_presets.py`)

`[forcing] planet_preset` selects among six planetary configurations.

| `planet_preset` | Key Parameter Overrides | Surface Initial Condition | Reference |
| :--- | :--- | :--- | :--- |
| `"earth"` *(default)* | None | `Earth2014` continents, bathymetry & ice sheets | — |
| `"aquaplanet"` | `use_earth2014=false`, $\varepsilon=0^\circ, e=0$, rivers/land C/dust disabled | 100% flat ocean, $T_{\mathrm{ml}}=288\ \mathrm{K}$, zero ice | Neale & Hoskins (2000) |
| `"snowball_earth"` | $S_0=1280$, $\mathrm{CO}_2=200\ \mathrm{ppmv}$, weathering & SNICAR enabled | Ocean $m_{\mathrm{ice}}=2750\ \mathrm{kg\ m^{-2}}$ ($\sim 3\ \mathrm{m}$), $T_{\mathrm{ice}}=245\ \mathrm{K}$; land $\mathrm{SWE}=120$ | Pierrehumbert et al. (2011) |
| `"tidally_locked"` | `enable_tidal_lock=true`, $P_{\mathrm{rot}}=10\ \mathrm{d}$, $\varepsilon=e=0$, diurnal off | 100% flat ocean | Yang et al. (2013) |
| `"trappist1e"` | `enable_tidal_lock=true`, $S_0=900$, $P_{\mathrm{rot}}=6.101\ \mathrm{d}$, $\mathrm{CO}_2=1000\ \mathrm{ppmv}$ | 100% flat ocean | Turbet et al. (2018); Fauchez et al. (2020) |
| `"mars_like"` | $S_0=589$, $\Omega=7.0882\times 10^{-5}$, $\varepsilon=25.19^\circ$, $e=0.0934$, $\mathrm{CO}_2=4\%$ | 100% dry bare land ($\sigma_{\mathrm{topo}}=150\ \mathrm{m}, T=218\ \mathrm{K}, w_\ell=0.005, \mathrm{LAI}=0$) | — |

When `enable_tidal_lock = true`, the solar zenith cosine takes a stationary distribution centered at substellar longitude $\lambda_{\mathrm{sub}}$ (`substellar_lon_deg`):

$$
\mu_0(\phi, \lambda) = \max\!\big(0,\ \cos\phi\cos(\lambda - \lambda_{\mathrm{sub}})\big), \qquad S_{\mathrm{TOA}} = S_0 \tag{11.5}
$$

---

## 12. Centralized 324-Parameter TOML Registry, Diagnostics (Köppen–Geiger / CMIP7), Zarr v3 & AOBA Studio

### 12.1 Single TOML Registry, Fused GPU Kernels & Native Zarr v3 I/O

All 14 sections and 324 tunable global parameters are centralized in [`config/aoba_default.toml`](../config/aoba_default.toml) (`src/aoba/params.py`), with 100% bilingual physical documentation across all 323 unique keys maintained in [`src/aoba/param_docs.py`](../src/aoba/param_docs.py) and 11 ready-to-run experiment presets in `config/presets/`. Hardware-portable `float32` execution (`cuda` $\to$ `mps` $\to$ `cpu` via `src/aoba/runtime.py`) compiles 104 single-source fused Metal kernels (`kernels.py`, `kernels_physics.py`, `kernels_extra.py`, `kernels_esm.py`) automatically to native CUDA C++17 via `kernels_cuda.py` alongside automatic CUDA Graph capture/replay, executing vertical tridiagonal sweeps, settling/sinking, radiation, microphysics, explicit bulk lightning, CRM sub-cycles, EVP sea ice, barotropic ocean sub-cycles, and OSSP ML feature/closure/spatial-head operations with zero mid-step CPU–GPU synchronizations. All native model outputs (`src/aoba/io_zarr.py`) use CF-1.11 / UGRID-1.0 Zarr v3 (`mesh.zarr`, `monthly.zarr`, `snapshots.zarr`, and optionally `cmip7_monthly.zarr` / `cmip7_fx.zarr`), alongside `daily_global.csv` and atomic checkpoints `restart.pt` / `restart_mid.pt` (supporting universal cross-grid spherical and hydrostatic vertical remapping).

| TOML Section | Keys | Scope |
| :--- | ---: | :--- |
| `[grid]` | 16 | SCVT `glevel`, vertical `nz`/`z_top`, CRM `(nx, ny, dx)`, time steps, chunking |
| `[forcing]` | 100 | All selectable schemes (`rad_scheme`, `microphysics_scheme`, `lightning_scheme`, `ocean_mode`, `o3_mode`, `co2_mode`, `planet_preset`), OSSP distillation & shadow diagnostics, orbit, calendar, ESM switches (including optional `enable_red_sprites`) |
| `[icesheet]` | 5 | Dynamic SIA ice-sheet switch, EMA timescale, acceleration factor, isostasy timescale |
| `[output]` | 23 | Zarr v3 cadence/splitting, CSV, mid-run checkpoints, CMIP7 output |
| `[constants]` | 26 | Planetary radius, gravity, gas constants, latent heats, ocean/ice constants |
| `[radiation]` | 7 | Band fractions, PAR conversion, analytic ozone coefficients |
| `[microphysics]` | 32 | Hydrometeor densities, intercepts, fall-speed relations, CCN/INP scales |
| `[turbulence]` | 13 | MYNN 2.5 closure constants, De Ridder RSL parameters, air viscosity |
| `[surface]` | 24 | Soil thermal/hydrological constants, canopy interception, snow/ice albedos, emissivities |
| `[biosphere]` | 41 | $\mathrm{C}_3/\mathrm{C}_4$ EEO photosynthesis, activation energies, LAI costs, phenology |
| `[ocean_ekman]` | 18 | 1.5-layer ocean constants and 3D OGCM GM90/Redi/KPP/tidal/shortwave constants |
| `[orography]` | 3 | Lott–Miller blocking height, wake drag, and mountain-wave amplitude coefficients |
| `[gravity_wave]` | 11 | Frontal and convective non-orographic gravity-wave spectrum constants |
| `[icesheet_physics]` | 5 | Ice/mantle densities, Glen $A$, mask threshold, maximum ice thickness |

### 12.2 30-Class Köppen–Geiger Climate Classification & CMIP7 ESM-BCV Diagnostics (`src/aoba/koppen.py`, `src/aoba/cmip7_diag.py`)

- **30-Class Köppen–Geiger Classification (`classify_koppen_geiger`; Beck et al., 2018, 2023, Table 1)**: From monthly mean surface air temperature $T_m\ [^\circ\mathrm{C}]$ and monthly precipitation totals $P_m\ [\mathrm{mm}]$ (hemisphere-aware April–September vs. October–March high-sun half-year), autonomously diagnoses the 30 climate codes (`1=Af` .. `30=EF`) and five major groups in strict precedence order `E` (Polar: $T_{\mathrm{hot}} < 10^\circ\mathrm{C}$) $\to$ `B` (Arid: $P_{\mathrm{ann}} < 10 P_{\mathrm{th}}$) $\to$ `A` (Tropical) $\to$ `C` (Temperate) $\to$ `D` (Cold).
- **CMIP7 Data Request v1.2 ESM-BCV Output (`cmip7_output_enabled = true`)**: Exports atmospheric 2D/TOA fields, standard 19-level `plev19` ($1000\text{–}1\ \mathrm{hPa}$) log-pressure interpolated 3D fields (`ta, ua, va, wap, zg, hus, hur, cl`), 2D/3D ocean (`tos, sos, zos, mlotst, hfds, thetao, so, uo, vo, wo`), sea ice, land/cryosphere, and AOBA autonomous EEO biosphere traits (`hcanopy, sroot, fc4, fevg, vcmax25, jmax25, vpmax25_c4`, etc.) in CF-1.11 sign-convention Zarr v3.

### 12.3 AOBA Studio Workbench (`scripts/aoba_config_gui.py`)

The local web application (`python scripts/aoba_config_gui.py --port 8765`) provides bilingual editing and dropdown selection for all 324 parameters, experiment presets, three synchronized rotating globes (`T₂`, `Precip`, and 10 ESM diagnostics) compared against observational reference ranges (ERA5, CERES EBAF, GPCP, MODIS, NSIDC), and an embedded documentation viewer with KaTeX mathematical typesetting (automatically falling back to the built-in recursive-descent renderer offline) and inline SVG schematic diagrams.
