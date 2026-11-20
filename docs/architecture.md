# AOBA Model Architecture and Governing Formulation

> **Language**: English | [日本語 (architecture_ja.md)](architecture_ja.md)

## 1. Scientific Objective

AOBA couples universal eco-evolutionary optimality (EEO) vegetation theory to the global atmospheric general circulation without cumulus parameterization, prescribed non-orographic gravity-wave drag sources, or plant-functional-type (PFT) parameter tables. External spatial boundary conditions are restricted to Earth's orbital insolation, land–sea geometry, and surface elevation $z_{\mathrm{topo}}(\lambda, \varphi)$.

---

## 2. Multi-Scale Grid Hierarchy and Coordinate System

### 2.1 Host Global Dynamical Core
- **Equations**: 3D hydrostatic primitive equations in vorticity–divergence $(\zeta, D)$, virtual temperature $T_v$, specific humidity $q_v$, and log surface pressure $\ln p_s$ form.
- **Horizontal Grid**: T42 triangular spectral truncation on a `128 × 64` Gaussian grid ($N_{\mathrm{host}} = 8,192$ columns, $\Delta \lambda \approx 2.8125^\circ$; optional T63 dynamical advection truncation).
- **Vertical Grid**: $N_z = 50$ terrain-following $\sigma = p / p_s$ (or hybrid $\sigma\text{--}p$) levels extending from the surface to $z_{\mathrm{top}} \approx 38\ \mathrm{km}$ ($p_{\mathrm{top}} \approx 3\ \mathrm{hPa}$), with ~15 levels below $2\ \mathrm{km}$ ($\Delta z_1 = 60\ \mathrm{m}$).
- **Time Integration**: Semi-implicit leapfrog (with Robert–Asselin–Williams filter) or RK3 at $\Delta t_{\mathrm{GCM}} = 900\ \mathrm{s}$ ($15\ \mathrm{min}$).

### 2.2 Displacement-Following Aerodynamic Datum (Canopy–Terrain Treatment)
To prevent tall canopies ($h_c \le 45\ \mathrm{m}$) from exceeding the lowest atmospheric reference height $z_1$, the coordinate origin $\tilde{z} = 0$ of the atmospheric boundary layer and embedded CRM is placed at the effective aerodynamic datum:
$$
z_{\mathrm{datum}} = z_{\mathrm{topo}} + d
$$
where $z_{\mathrm{topo}}$ is smoothed surface elevation and $d \approx \frac{2}{3} h_c$ is the zero-plane displacement height. With lowest CRM layer thickness $\Delta z_1 = 60\ \mathrm{m}$ (first scalar/momentum cell center at $\tilde{z}_1 = 30\ \mathrm{m}$ above $z_{\mathrm{datum}}$), the geometric ordering
$$
\tilde{z}_1 = 30\ \mathrm{m} > (h_c - d) > z_{0m}
$$
holds identically for all physically attainable canopy heights, ensuring nonsingular Monin–Obukhov similarity profiles supplemented by roughness-sublayer (RSL) mixing enhancement.

### 2.3 Embedded 3D Anelastic Cloud-Resolving Model (CRM)
- **Batch Tensor Layout**: `[B = 8192, nz = 50, ny = 8, nx = 8]` in `Float32` (~26.2M total cells; ~105 MB per 3D field, ~3 GB total working memory).
- **Domain & Staggering**: Doubly periodic $32\ \mathrm{km} \times 32\ \mathrm{km}$ horizontal domain ($\Delta x = \Delta y = 4\ \mathrm{km}$) on an Arakawa C-grid with Charney–Phillips vertical staggering ($\theta_l, w$ at half-levels; $u, v, p', q_n, q_p$ at full-levels) in height above local datum $\tilde{z} = z - z_{\mathrm{datum}}$, using the local host column reference state $\bar{\rho}(\tilde{z}; p_s), \bar{\theta}(\tilde{z}; p_s)$.
- **Elliptic Pressure Solver**: Enforces $\nabla_h \cdot (\bar{\rho}\mathbf{u}_h) + \partial_{\tilde{z}}(\bar{\rho}w) = 0$ via precomputed real orthogonal $8 \times 8$ discrete Fourier matrices (`matmul` along $x$ and $y$) coupled with a batched vertical Thomas tridiagonal solver, avoiding complex-FFT backend limitations on GPU hardware. Time step $\Delta t_{\mathrm{CRM}} = 15\text{--}20\ \mathrm{s}$.
- **Momentum & Gravity-Wave Closure**: Full 3D vector winds $(u, v, w)$ resolve omnidirectional convective gravity-wave excitation, vertical propagation into the stratosphere ($z \le 38\ \mathrm{km}$), critical-level absorption, nonlinear wave breaking, and unbiased 3D convective momentum tendencies $(\partial_t \bar{u}, \partial_t \bar{v})_{\mathrm{CRM}}$ returned directly to the host jet stream without prescribed gravity-wave sources.

---

## 3. Moist Physics and Four-Band Analytical Radiation

### 3.1 Two-Variable Diagnostic-Phase Cloud Microphysics
Following Khairoutdinov & Randall (2003), water species are represented by two prognostic CRM scalars:
1. Total non-precipitating water $q_n = q_v + q_c + q_i$
2. Total precipitating water $q_p = q_r + q_s$

Condensate $\max(0, q_n - q^*(T, p))$ and precipitation $q_p$ are partitioned diagnostically between liquid and ice phases via a linear temperature ramp $\omega(T) \in [0, 1]$ between $T_{00} = 253.15\ \mathrm{K}$ and $T_0 = 273.16\ \mathrm{K}$, accounting for both vaporization and fusion latent heats as well as upper-tropospheric ice-cloud radiative effects.

### 3.2 Lookup-Table-Free Four-Band Two-Stream Radiation
Evaluated every $\Delta t_{\mathrm{rad}} = 900\ \mathrm{s}$ across CRM columns (with instantaneous solar-zenith scaling $\mu_0(t)$ on shortwave fluxes at CRM steps):
- **SW1 (UV + Visible / PAR, ~50% of TOA solar flux)**: Upper-atmosphere $\mathrm{O}_3(p)$ absorption (generating stratospheric meridional temperature structure solely via solar geometry), cloud $\delta$-Eddington scattering, and surface visible albedo $\alpha_{\mathrm{vis}}$; transmitted surface flux provides $\mathrm{PAR}$ directly to the P-model.
- **SW2 (Near-Infrared, ~50% of TOA solar flux)**: Tropospheric $\mathrm{H}_2\mathrm{O}$ absorption, cloud scattering/absorption, and high vegetation NIR reflectance $\alpha_{\mathrm{nir}}$.
- **LW1 ($\mathrm{H}_2\mathrm{O}$ + $\mathrm{CO}_2$ Absorption Band, ~68% of thermal emission)**: Pressure-broadened water vapor optical depth plus logarithmic $\mathrm{CO}_2$ optical depth and cloud emissivity, driving tropospheric radiative cooling ($\sim 1.5\text{--}2\ \mathrm{K\ d^{-1}}$).
- **LW2 (8–12 $\mu\mathrm{m}$ Atmospheric Window, ~32% of thermal emission)**: Water-vapor continuum absorption ($\propto q_v^2 p / p_0$) and strong liquid/ice cloud absorption, capturing nocturnal clear-sky surface cooling and anvil greenhouse trapping.

---

## 4. Subgrid Autonomous Land Surface and Slab Ocean

Each host cell contains `8 × 8 = 64` surface tiles (`[B = 8192, ny = 8, nx = 8]`) coupled 1-to-1 with the bottom layer of the 3D CRM.

### 4.1 Leaf-Level Pure P-Model (Unstressed Biochemistry & Stomata)
- **Optimal $c_i / c_a$ Ratio**: Least-cost coordination determines $\chi = \frac{\Gamma^*}{c_a} + \left(1 - \frac{\Gamma^*}{c_a}\right)\frac{\xi}{\xi + \sqrt{D}}$, where $D$ is vapor pressure deficit, $\Gamma^*(T, p_s)$ is the photorespiratory compensation point, and $\xi(T, \eta^*)$ depends on temperature and water viscosity.
- **Acclimation**: Photosynthetic capacities $(V_{c\max 25}, J_{\max 25})$ acclimate toward midday environmental exponential moving averages ($\tau_{\mathrm{acclim}} = 15\ \mathrm{d}$), while instantaneous stomatal conductance $g_s$ and surface energy balance ($R_n = H + \lambda E + G$) operate at each surface step.

### 4.2 EEO Canopy Greenness and Prognostic LAI
Following Cai et al. (2025, *Commun. Earth Environ.*) and Zhou et al. (2025, *Glob. Change Biol.*), target equilibrium $\mathrm{fAPAR}^*$ is governed by the co-limitation of carbon-cost light optimality and water-supply optimality without PFT hydraulic traits:
1. **Energy/Carbon-Cost Limit**:
   $$
   \mathrm{fAPAR}_c = 1 - \frac{z_L}{k A_0}
   $$
   where $A_0$ is potential P-model assimilation at $\mathrm{fAPAR} = 1$, $k = 0.5$ is the canopy extinction coefficient, and $z_L$ is the unit foliage maintenance/construction cost rate.
2. **Water-Supply Limit**:
   $$
   \mathrm{fAPAR}_w = \frac{c_a(1 - \chi)}{1.6 D}\frac{S_w}{A_0}
   $$
   where $S_w(W)$ is sustainable root-zone water supply rate from a single-layer soil moisture bucket $\frac{dW}{dt} = P_{\mathrm{rain}} + M_{\mathrm{snow}} - E - R_{\mathrm{runoff}}$ ($0 \le W \le W_{\max}$).
3. **Realized Canopy State**:
   $$
   \mathrm{fAPAR}^* = \max\!\left(0,\, \min(\mathrm{fAPAR}_c,\, \mathrm{fAPAR}_w)\right), \qquad \mathrm{LAI}^* = -\frac{1}{k}\ln(1 - \mathrm{fAPAR}^*)
   $$
   Prognostic $\mathrm{LAI}$ relaxes toward $\mathrm{LAI}^*$ over leaf growth/senescence timescales, modulating canopy transpiration, roughness, and two-band surface albedo $(\alpha_{\mathrm{vis}}, \alpha_{\mathrm{nir}})$ alongside a single-layer snow bucket $S_{\mathrm{snow}}$.

### 4.3 Mixed-Layer Slab Ocean and Thermodynamic Sea Ice
Ocean tiles evolve a mixed-layer temperature $C_{\mathrm{ml}} \partial_t T_{\mathrm{ocn}} = F_{\mathrm{net,sfc}}$ ($H_{\mathrm{ml}} = 50\ \mathrm{m}$, zero prescribed Q-flux by default) with a freezing-point ($T_f = 271.35\ \mathrm{K}$) thermodynamic sea-ice thickness/albedo transition.
