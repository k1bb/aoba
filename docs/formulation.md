# Detailed Mathematical Formulation of AOBA

> **Language**: English | [日本語 (formulation_ja.md)](formulation_ja.md)

This document presents the complete continuous equations, coordinate transformations, staggered spatial discretizations, time-integration schemes, closed-form physical/ecological/cryospheric/oceanic/ice-sheet closures, and unified TOML parameter registry of **AOBA**, written to the mathematical completeness standard of *Description of MIROC6 AGCM* (2021), *Description of MATSIRO6* (2021), and *Detailed formulation of SCALE-RM* (Nishizawa et al., 2025). Every equation, symbol, and default constant maps 1-to-1 onto `src/aoba/` and `config/aoba_default.toml`.

---

## Contents

1. [Physical Constants and Orbital Solar Geometry (`constants.py`, `radiation.py`)](#1-physical-constants-and-orbital-solar-geometry)
2. [Vertical Charney–Phillips Grid and Smoothed Terrain-Following Coordinate (`vertical_grid.py`)](#2-vertical-charneyphillips-grid-and-smoothed-terrain-following-coordinate)
3. [Moist Compressible Thermodynamics and Mixed-Phase Saturation Adjustment (`thermodynamics.py`)](#3-moist-compressible-thermodynamics-and-mixed-phase-saturation-adjustment)
4. [Four-Band Lookup-Table-Free Two-Stream Radiative Transfer (`radiation.py`, `kernels.py`)](#4-four-band-lookup-table-free-two-stream-radiative-transfer)
5. [Mixed-Phase Single-Moment Cloud Microphysics and Sedimentation (`microphysics.py`)](#5-mixed-phase-single-moment-cloud-microphysics-and-sedimentation)
6. [Roughness Sublayer (RSL), Implicit MYNN Level 2.5 PBL, and 3D SGS Turbulence (`turbulence.py`, `crm.py`)](#6-roughness-sublayer-rsl-implicit-mynn-level-25-pbl-and-3d-sgs-turbulence)
7. [Two-Leaf Non-Rectangular-Hyperbola EEO Biosphere (`biosphere.py`)](#7-two-leaf-non-rectangular-hyperbola-eeo-biosphere)
8. [Land Hydrology, Five-Layer Freeze–Thaw Soil, Snow, and Ocean/Sea-Ice (`surface.py`)](#8-land-hydrology-five-layer-freezethaw-soil-snow-and-oceansea-ice)
9. [Embedded 3D Nonhydrostatic RK3–HEVI CRM and Zalesak FCT (`crm.py`)](#9-embedded-3d-nonhydrostatic-rk3hevi-crm-and-zalesak-fct)
10. [Global SCVT C-Grid Core, Orographic Drag, Ekman Ocean, and MAML+ICA Coupling (`scvt_mesh.py`, `scvt_dynamics.py`, `orography.py`, `ocean_ekman.py`, `coupled_model.py`, `koppen.py`)](#10-global-scvt-c-grid-core-orographic-drag-ekman-ocean-and-mamlica-coupling)
11. [Dynamic / Accelerated SIA Ice-Sheet Model and Conservative Barometric Remapping (`icesheet.py`)](#11-dynamic--accelerated-sia-ice-sheet-model-and-conservative-barometric-remapping)
12. [Unified Single-File TOML Configuration, Output Cadence, GUI, and 188-Parameter Registry (`params.py`, `config/aoba_default.toml`, `scripts/aoba_config_gui.py`)](#12-unified-single-file-toml-configuration-output-cadence-gui-and-188-parameter-registry)

---

## 1. Physical Constants and Orbital Solar Geometry

### 1.1 Fundamental Physical Constants (`src/aoba/constants.py`)

Table 1.1 lists the planetary, thermodynamic, radiative, and oceanic constants shared across AOBA (Wing et al., 2018, Table 1; Khairoutdinov & Randall, 2003, Appendix A; MIROC6 AGCM, 2021, Sec. 1.3.2; Semtner, 1976). Derived constants (`c_vd = c_pd - R_d`, `gamma_d = c_pd / c_vd`, `kappa_d = R_d / c_pd`, `eps_water = R_d / R_v`, `z_vir = R_v / R_d - 1`, `L_s0 = L_v0 + L_f0`) are automatically refreshed via `constants.recompute_derived_constants()` whenever TOML overrides are loaded.

| Symbol | Physical Quantity | Default Value | Units | Reference |
| :--- | :--- | :--- | :--- | :--- |
| $R_E$ | Mean Earth radius | $6.371 \times 10^6$ | $\mathrm{m}$ | Wing et al. (2018) |
| $g$ | Mean surface gravity | $9.79764$ | $\mathrm{m\ s^{-2}}$ | Wing et al. (2018) |
| $\Omega$ | Planetary rotation rate | $7.292115 \times 10^{-5}$ | $\mathrm{rad\ s^{-1}}$ | MIROC6 (2021) |
| $S_0$ | Total solar irradiance | $1361.0$ | $\mathrm{W\ m^{-2}}$ | Wing et al. (2018) |
| $R_d$ | Dry-air gas constant | $287.04$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pd}$ | Dry-air isobaric specific heat | $1004.64$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{vd}$ | Dry-air isochoric specific heat ($c_{pd} - R_d$) | $717.60$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Skamarock et al. (2012) |
| $R_v$ | Water-vapor gas constant | $461.50$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pv}$ | Water-vapor isobaric specific heat | $1846.0$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $\epsilon$ | Molecular weight ratio $R_d / R_v$ | $0.621972$ | $-$ | Skamarock et al. (2012) |
| $\delta_{\mathrm{vir}}$ | Virtual factor $R_v / R_d - 1$ | $0.607790$ | $-$ | Skamarock et al. (2012) |
| $p_0$ | Reference surface pressure | $1.0 \times 10^5$ | $\mathrm{Pa}$ | Skamarock et al. (2012) |
| $T_0$ | Freezing point temperature | $273.15$ | $\mathrm{K}$ | CODATA |
| $T_{\mathrm{tp}}, e_{\mathrm{tp}}$ | Water triple-point temperature & saturation vapor pressure | $273.16\ \mathrm{K},\ 611.2\ \mathrm{Pa}$ | $-$ | Bolton (1980) |
| $L_{v0}$ | Latent heat of vaporization at $0\ ^\circ\mathrm{C}$ | $2.501 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{f0}$ | Latent heat of fusion at $0\ ^\circ\mathrm{C}$ | $3.337 \times 10^5$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{s0}$ | Latent heat of sublimation ($L_{v0} + L_{f0}$) | $2.8347 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $\sigma$ | Stefan–Boltzmann constant | $5.670374419 \times 10^{-8}$ | $\mathrm{W\ m^{-2}\ K^{-4}}$ | CODATA |
| $\kappa$ | von Kármán constant | $0.40$ | $-$ | Nakanishi & Niino (2009) |
| $\rho_w, c_{pw}$ | Seawater density & specific heat | $1025.0\ \mathrm{kg\ m^{-3}},\ 3990.0\ \mathrm{J\ kg^{-1}\ K^{-1}}$ | $-$ | MIROC6 (2021) |
| $T_{f,\mathrm{ocn}}$ | Ocean freezing temperature | $271.35$ | $\mathrm{K}$ | Semtner (1976) |
| $\rho_i, k_i, k_s$ | Sea-ice density, ice conductivity, snow conductivity | $917.0\ \mathrm{kg\ m^{-3}},\ 2.034,\ 0.3096\ \mathrm{W\ m^{-1}\ K^{-1}}$ | $-$ | Semtner (1976) |

### 1.2 Orbital Insolation and Solar Zenith Angle (`src/aoba/radiation.py`)

Following Berger (1978) and Davis et al. (2017, Eqs. 2–8), from elapsed simulation time $t\ [\mathrm{s}]$, fractional day of year $N_d = (t / T_{\mathrm{day}}) \bmod D_{\mathrm{yr}}$ ($T_{\mathrm{day}} = 86400\ \mathrm{s},\ D_{\mathrm{yr}} = 365.2425$), orbital eccentricity $e$, obliquity $\varepsilon$, and longitude of perihelion $\tilde{\omega}$, the mean anomaly relative to vernal equinox (day 80) $\nu_m = \frac{2\pi}{D_{\mathrm{yr}}}(N_d - 80.0) - \tilde{\omega}$, true anomaly $\nu$, and true heliocentric longitude $\lambda_s$ are:
$$
\begin{aligned}
\nu &= \nu_m + \left(2e - \frac{1}{4}e^3\right)\sin\nu_m + \frac{5}{4}e^2 \sin(2\nu_m), \\
\lambda_s &= \nu + \tilde{\omega}.
\end{aligned}
$$
Solar declination $\delta_s$, instantaneous top-of-atmosphere (TOA) solar irradiance $S_{\mathrm{inst}}(t)$, local solar hour angle $H(t, \lambda)$, and cosine of solar zenith angle $\mu_0(\varphi, \lambda, t)$ are evaluated as:
$$
\begin{aligned}
\sin\delta_s &= \sin\varepsilon \sin\lambda_s, \\
S_{\mathrm{inst}}(t) &= S_0 \left(\frac{1 + e\cos\nu}{1 - e^2}\right)^2, \\
H(t, \lambda) &= 2\pi \left[\frac{t \bmod T_{\mathrm{day}}}{T_{\mathrm{day}}} - \frac{1}{2}\right] + \lambda, \\
\mu_0(\varphi, \lambda, t) &= \max\!\left[0,\ \sin\varphi \sin\delta_s + \cos\varphi \cos\delta_s \cos H(t, \lambda)\right].
\end{aligned}
$$
Whenever $H(t, \lambda) < 0 \le H(t + \Delta t, \lambda)$ with $\Delta H < \pi$, local solar noon crossing is flagged to sample downward visible fluxes for plant photosynthetic acclimation.

#### 1.2.1 Calendar Management, Perpetual Insolation, and Daily-Mean Zenith (`src/aoba/calendar.py`)
`AobaCalendar` maps elapsed simulation seconds $t$ from `start_date` (default `"2000-03-20"`, Vernal Equinox) under `calendar_type` (`gregorian` with 4/100/400 leap years, `noleap` fixed 365-day, or `360_day` idealized 12×30-day) to calendar dates and vernal-equinox-referenced orbital year fraction $f_{\mathrm{yr}} \in [0, 1)$. For `insolation_mode = "perpetual_date"`, seasonal orbital position is locked at `fixed_insolation_date`, and when `diurnal_cycle = false`, the diurnal cycle is replaced by the analytical daily-mean cosine zenith angle:
$$
H_0(\varphi, \delta_s) = \arccos\!\left[\mathrm{clip}\!\left(-\tan\varphi \tan\delta_s,\ -1,\ 1\right)\right], \qquad \overline{\mu}_0(\varphi, \delta_s) = \max\!\left[0,\ \frac{H_0 \sin\varphi \sin\delta_s + \cos\varphi \cos\delta_s \sin H_0}{\pi}\right].
$$

---

## 2. Vertical Charney–Phillips Grid and Smoothed Terrain-Following Coordinate

### 2.1 Stretched Vertical Levels, Rayleigh Sponge, and Charney–Phillips Staggering (`src/aoba/vertical_grid.py`)

AOBA discretizes the atmosphere from $\zeta_0 = 0\ \mathrm{m}$ to model top $z_{\mathrm{top}} = 24,000\ \mathrm{m}$ (default `[grid]` value) across $N_z = 32$ layers with Charney–Phillips vertical staggering (Gassmann, 2013, Sec. 3):
- **Half-level interfaces** ($k + \frac{1}{2}$ for $k = 0, \dots, N_z$): vertical velocity $w$, interface height $z_w$, and radiative fluxes $F^\uparrow, F^\downarrow$.
- **Full-level centers** ($k \in \{0, \dots, N_z - 1\}$): dry-air density $\rho_d$, moist potential temperature $\theta_m$, pressure $p$, temperature $T$, horizontal winds, TKE $q^2$, and moisture mixing ratios $(q_v, q_c, q_i, q_n, q_p)$.

Nominal thicknesses $\Delta \zeta_k = \Delta \zeta_0 r^k$ stretch geometrically from $\Delta \zeta_0 = 60.0\ \mathrm{m}$ such that $\sum_{k=0}^{N_z-1}\Delta\zeta_k = z_{\mathrm{top}}$ ($r \approx 1.132$ solved via Newton–Raphson), placing $\sim 13$ levels below $2\ \mathrm{km}$. Above $z_{\mathrm{sp}} = 17,500\ \mathrm{m}$, a $\sin^2$ Rayleigh sponge damps gravity waves (Klemp et al., 2008; Skamarock et al., 2012) with default $\tau_{\mathrm{sp,min}} = 300.0\ \mathrm{s}$:
$$
\gamma_R(\zeta) = \frac{1}{\tau_{\mathrm{sp,min}}} \sin^2\!\left[\frac{\pi}{2}\,\mathrm{clip}\!\left(\frac{\zeta - z_{\mathrm{sp}}}{z_{\mathrm{top}} - z_{\mathrm{sp}}},\, 0,\, 1\right)\right].
$$

### 2.2 Klemp (2011) Smoothed Terrain-Following Coordinate and Canopy Reference Height

To suppress terrain-induced pressure-gradient truncation errors, physical interface heights follow Klemp (2011, Eqs. 4–7) with attenuation exponent `terrain_scale_exp` $= p = 1.5$ and progressively smoothed terrain surfaces $h_{s,k}(\mathbf{x})$ ($h_{s,0} = z_{\mathrm{topo}}$):
$$
z_{w, k}(\mathbf{x}) = \zeta_{w, k} + A(\zeta_{w, k})\, h_{s, k}(\mathbf{x}), \qquad A(\zeta) = \cos^{2p}\!\left(\frac{\pi \zeta}{2 z_{\mathrm{top}}}\right), \qquad \Delta z_{c, k} = z_{w, k+1} - z_{w, k} \ge 0.25\,\Delta \zeta_k.
$$
The lower boundary of the atmospheric mesh remains anchored on the solid Earth / ice-sheet surface ($z_{w,0} = z_{\mathrm{topo}}$), while forest canopy displacement height enters surface-layer similarity directly through the aerodynamic reference height:
$$
\tilde{z}_{\mathrm{ref}} = \max\!\left(z_{c,0} - z_{w,0} - 0.65\, h_c,\ 5.0\ \mathrm{m}\right).
$$
This avoids daily coordinate remapping from growing vegetation while accurately triggering De Ridder (2010) RSL enhancement inside tall canopies ($h_c \le 40\ \mathrm{m}$). Conversely, whenever dynamic ice-sheet evolution changes solid surface elevation ($|z_{\mathrm{topo}}^{\mathrm{new}} - z_{w,0}| \ge 0.05\ \mathrm{m}$; Section 11), `compute_stf_heights` updates all layers and barometrically remaps host and CRM columns.

---

## 3. Moist Compressible Thermodynamics and Mixed-Phase Saturation Adjustment

### 3.1 Equation of State and Vapor-Buoyancy Density (`src/aoba/thermodynamics.py`)

With dry-air mixing ratios $(q_v, q_c, q_i)$, non-precipitating water $q_n = q_v + q_c + q_i$, and precipitating water $q_p = q_r + q_s + q_g$, modified moist potential temperature $\theta_m$, compressible pressure $p$, Exner function $\Pi$, temperature $T$, and total moist density $\rho$ obey (Skamarock et al., 2012, Eqs. 5–6; Seidel & Yang, 2024):
$$
\begin{aligned}
\theta_m &= \theta \left(1 + \frac{R_v}{R_d} q_v\right), \qquad p = p_0 \left(\frac{R_d \rho_d \theta_m}{p_0}\right)^{c_{pd} / c_{vd}}, \\
\Pi &= \left(\frac{p}{p_0}\right)^{R_d / c_{pd}}, \qquad T = \frac{\Pi \theta_m}{1 + (R_v / R_d) q_v}, \qquad \rho = \rho_d \left(1 + q_n + q_p\right).
\end{aligned}
$$

### 3.2 Diagnostic Mixed-Phase Partitioning and Newton–Raphson Saturation Adjustment

Condensate $q_{\mathrm{cond}} = q_c + q_i$ partitions linearly between $T_{00n} = 253.15\ \mathrm{K}$ and $T_{0n} = 273.16\ \mathrm{K}$ (Khairoutdinov & Randall, 2003, Eqs. A8–A14):
$$
\omega_n(T) = \mathrm{clip}\!\left(\frac{T - T_{00n}}{T_{0n} - T_{00n}},\ 0,\ 1\right), \qquad q_c = \omega_n(T)\, q_{\mathrm{cond}}, \qquad q_i = \left[1 - \omega_n(T)\right] q_{\mathrm{cond}}.
$$
Liquid and hexagonal-ice saturation pressures (Bolton, 1980, Eq. 10; Murray, 1967) define the blended saturation mixing ratio $q_{\mathrm{sat}}(T, p)$:
$$
\begin{aligned}
e_{sw}(T) &= e_{\mathrm{tp}} \exp\!\left[\frac{17.67\,(T - T_0)}{T - 29.65}\right], \qquad e_{si}(T) = e_{\mathrm{tp}} \exp\!\left[\frac{21.8745584\,(T - T_0)}{T - 7.66}\right], \\
q_{\mathrm{sat}}(T, p) &= \omega_n(T) \frac{\epsilon e_{sw}(T)}{p - e_{sw}(T)} + \left[1 - \omega_n(T)\right] \frac{\epsilon e_{si}(T)}{p - e_{si}(T)}.
\end{aligned}
$$
Conserving liquid–ice equivalent temperature $T_{li} = T - (L_{v0} q_c + L_{s0} q_i)/c_{pd}$, supersaturated air undergoes four damped Newton–Raphson iterations on the moist-enthalpy residual:
$$
f(T) = T - T_{li} - \frac{L_{\mathrm{eff}}(T)}{c_{pd}}\left[q_n - q_{\mathrm{sat}}(T, p)\right] = 0, \qquad L_{\mathrm{eff}}(T) = \omega_n(T) L_{v0} + \left[1 - \omega_n(T)\right] L_{s0}.
$$

---

## 4. Four-Band Lookup-Table-Free Two-Stream Radiative Transfer

### 4.1 Unified Multi-Layer Two-Stream Tridiagonal Solver (`src/aoba/radiation.py`, `src/aoba/kernels.py`)

For $N = N_z$ layers indexed top-down ($n = 0$ TOA to $n = N - 1$ surface), upward/downward diffuse fluxes $F_n^\pm(\tau)$ satisfy Toon et al. (1989, Eqs. 11–12):
$$
\frac{\partial F_n^+}{\partial \tau} = \gamma_{1n} F_n^+ - \gamma_{2n} F_n^- - S_n^+(\tau), \qquad \frac{\partial F_n^-}{\partial \tau} = \gamma_{2n} F_n^+ - \gamma_{1n} F_n^- + S_n^-(\tau).
$$
- **Shortwave (SW1, SW2)**: Applies Joseph, Wiscombe, & Weinman (1976) $\delta$-scaling ($f = g_0^2$, $\tau' = (1-\omega_0 f)\tau_0$, $\omega' = (1-f)\omega_0/(1-\omega_0 f)$, $g' = (g_0-f)/(1-f)$) with **Quadrature** ($\mu_1 = 1/\sqrt{3}$; Toon et al., 1989, Table 1):
  $$
  \gamma_1 = \frac{\sqrt{3}}{2}\left[2 - \omega'(1 + g')\right], \quad \gamma_2 = \frac{\sqrt{3}}{2}\omega'(1 - g'), \quad \gamma_3 = \frac{1}{2}\left(1 - \sqrt{3} g' \mu_0\right), \quad \gamma_4 = 1 - \gamma_3,
  $$
  and solar direct-beam particular sources $C_n^\pm(\tau)$ driven by $F_{\mathrm{TOA}} e^{-(\tau_{c,n}+\tau)/\mu_0}$.
- **Longwave (LW1, LW2)**: Applies **Hemispheric Mean** ($\mu_1 = 1/2, \omega_0 = 0 \implies \gamma_1 = 2, \gamma_2 = 0$) with linear-in-$\tau$ Planck emission $F_B(\tau) = F_{B,0n} + F_{B,1n}\tau$ yielding particular solutions $C_n^\pm(\tau) = F_{B,0n} + F_{B,1n}(\tau \pm (\gamma_{1n} + \gamma_{2n})^{-1})$.

Defining $\lambda_n = \sqrt{\gamma_{1n}^2 - \gamma_{2n}^2}$, $\Gamma_n = \gamma_{2n}/(\gamma_{1n} + \lambda_n)$, and $e_{1n}\dots e_{4n}$ (Toon et al., 1989, Eq. 44), TOA, interface continuity, and surface albedo/emission boundary conditions assemble a $2N \times 2N$ tridiagonal system $A_l Y_{l-1} + B_l Y_l + D_l Y_{l+1} = E_l$ solved simultaneously across all subcolumns via the fused Metal/CUDA Thomas kernel `solve_tridiagonal_batched`. Direct and diffuse downward shortwave fluxes are output separately at the surface.

### 4.2 Band Parameterizations and Cloud Single-Scattering Optics

With $\Delta m_{d,n} = \rho_{d,n}\Delta z_n$, $\mathrm{LWP} = 10^3 \rho_d q_c \Delta z$, $\mathrm{IWP} = 10^3 \rho_d q_i \Delta z$ ($\mathrm{g\ m^{-2}}$), effective radii $(r_{e,l}, r_{e,i}) = (10, 48)\ \mu\mathrm{m}$, and subgrid SW cloud inhomogeneity factor $\chi_{\mathrm{SW}} = \text{cloud\_sw\_inhomog} = 0.85$ scaling $\tau_{l}, \tau_{i}$ to mitigate plane-parallel albedo bias (Cahalan et al., 1994, Eqs. 5–7; Rossow & Cairns, 1995):

1. **SW1 ($0.25\text{--}0.69\ \mu\mathrm{m}$, solar weight $0.459760$)**: Background $\mathrm{O}_3(p) = 3.6478\, p_{\mathrm{hPa}}^{0.83209}\exp(-p_{\mathrm{hPa}}/11.3515)\times 10^{-6}$ (Wing et al., 2018), Rodgers slant airmass $M(\mu_0) = 35 / \sqrt{1224\mu_0^2 + 1}$, Lacis & Hansen (1974, Eqs. 8–10) UV+Chappuis absorption $A_{\mathrm{oz}}(x)$, Rayleigh optical depth $0.095\,\Delta p/p_0$, and Slingo (1989, Band 1) + Ebert & Curry (1992, Band 1) cloud optics:
   $$
   \begin{aligned}
   \tau_{l1} &= \chi_{\mathrm{SW}}\,\mathrm{LWP}\left(2.817\times 10^{-2} + \frac{1.305}{r_{e,l}}\right), & \omega_{l1} &= 1 - (-5.62\times 10^{-8} + 1.63\times 10^{-7} r_{e,l}), & g_{l1} &= 0.829 + 2.482\times 10^{-3} r_{e,l}, \\
   \tau_{i1} &= \chi_{\mathrm{SW}}\,\mathrm{IWP}\left(3.448\times 10^{-3} + \frac{2.431}{r_{e,i}}\right), & \omega_{i1} &= 1 - 10^{-5}, & g_{i1} &= 0.7661 + 5.851\times 10^{-4} r_{e,i}.
   \end{aligned}
   $$
   Surface $\mathrm{PPFD} = 4.57 \times 0.88\, F_{\mathrm{SW1}}^\downarrow(z_{\mathrm{sfc}})$ ($\mu\mathrm{mol\ m^{-2}\ s^{-1}}$).
2. **SW2 ($0.69\text{--}4.00\ \mu\mathrm{m}$, solar weight $0.540240$)**: Pressure–temperature scaled precipitable water $\Delta y = 0.1\, q_v \Delta m_d (p/p_0)\sqrt{T_0/T}$ drives Lacis & Hansen (1974, Eq. 21) $A_{\mathrm{wv}}(y/\mu_0)$, plus Slingo (1989, Bands 2–4) and Ebert & Curry (1992, Bands 2–5) cloud optics scaled by $\chi_{\mathrm{SW}}$ ($\chi_{\mathrm{SW}}\tau_{l2}, \omega_{l2}, g_{l2}, \chi_{\mathrm{SW}}\tau_{i2}, \omega_{i2}, g_{i2}$).
3. **LW1 ($\mathrm{H}_2\mathrm{O} + \mathrm{CO}_2$, Planck fraction $0.680$)** and **LW2 ($8.0\text{--}12.5\ \mu\mathrm{m}$ window, fraction $0.320$)**:
   $$
   \begin{aligned}
   \Delta \tau_{\mathrm{LW1}} &= 0.30 \left(\frac{p}{p_0}\right)^{0.55} q_v \Delta m_d + \max\!\left[0.02,\ 0.85 + 0.22 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} + 0.020 \frac{\Delta p}{p_0} + 0.83 \left[0.090\,\mathrm{LWP} + \left(0.00324 + \frac{1.033}{r_{e,i}}\right)\mathrm{IWP}\right], \\
   \Delta \tau_{\mathrm{LW2}} &= \left[0.45 \frac{e_v}{p_0} \exp\!\left(\frac{1800}{T} - \frac{1800}{296}\right) + 0.0025 \frac{p}{p_0}\right] q_v \Delta m_d + \max\!\left[0,\ 0.015 + 0.024 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} + 0.83 \left[0.090\,\mathrm{LWP} + \left(0.00160 + \frac{1.166}{r_{e,i}}\right)\mathrm{IWP}\right].
   \end{aligned}
   $$
Layer radiative heating rate $(\partial_t T)_{\mathrm{rad},k} = -(F_{\mathrm{net},k+1/2} - F_{\mathrm{net},k-1/2}) / [(c_{pd} + c_{pv} q_{v,k})\rho_{d,k}\Delta z_k]$ strictly conserves column radiative energy.

---

## 5. Mixed-Phase Single-Moment Cloud Microphysics and Sedimentation

### 5.1 Two Prognostic Moist Variables and Diagnostic Partitioning (`src/aoba/microphysics.py`)

Following SAM (Khairoutdinov & Randall, 2003, Appendices A–B), AOBA integrates $q_n = q_v + q_c + q_i$ and $q_p = q_r + q_s + q_g$. Rain, snow, and graupel partition via temperature ramps $\omega_p(T)$ over $[268.16, 283.16]\ \mathrm{K}$ and $\omega_g(T)$ over $[223.16, 283.16]\ \mathrm{K}$:
$$
\omega_p(T) = \mathrm{clip}\!\left(\frac{T - T_{00p}}{T_{0p} - T_{00p}}, 0, 1\right), \quad \omega_g(T) = \mathrm{clip}\!\left(\frac{T - T_{00g}}{T_{0g} - T_{00g}}, 0, 1\right), \quad q_r = \omega_p q_p,\quad q_s = (1 - \omega_p)(1 - \omega_g)q_p,\quad q_g = (1 - \omega_p)\omega_g q_p.
$$

### 5.2 Size Distributions, Microphysical Transfers, and Subcycled Sedimentation

Marshall–Palmer size distributions $N_x(D) = N_{0x}\exp(-\lambda_x D)$ with slope $\lambda_x = (\pi \rho_x N_{0x} / (\rho_d q_x))^{1/4}$ and power-law fall speeds $V_x(D) = a_x D^{b_x}(\rho_0/\rho_d)^{1/2}$ yield:
1. **Autoconversion**: Continuous Berry (1968) / Tomita (2008) warm-rain conversion (SCALE-RM Eqs. 8.221–8.222, default droplet number concentration $N_d = 60\ \mathrm{cm^{-3}}$, $D_d = \max[0.146 - 5.964\times 10^{-2}\ln(N_d / 2000),\, 0.05]$; setting $N_d \le 0$ reverts to the KR03 Kessler threshold $\alpha_c \max(q_c - q_{c0}, 0)$) plus KR03 ice aggregation ($V_{TI} = 0.40\ \mathrm{m\ s^{-1}}$):
   $$
   \left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{auto}} = \frac{16.7\, D_d\, \rho_d^2\, q_c^3}{5.0\, D_d\, \rho_d\, q_c + 3.6\times 10^{-5} N_d} + \beta_i \exp\!\left[0.025(T - T_0)\right]\max(q_i - q_{i0}, 0).
   $$
2. **Accretion**: $(\partial_t q_p)_{\mathrm{accr}} = \sum_{x \in \{r,s,g\}}\sum_{y \in \{c,i\}} \frac{\pi a_x N_{0x}\Gamma(3 + b_x)}{4} E_{xy}(\rho_0/\rho_d)^{1/2}(\rho_d q_x / (\pi \rho_x N_{0x}))^{(3+b_x)/4} q_y$.
3. **Ventilated evaporation & sublimation** ($S = q_v / q_{\mathrm{sat}} < 1$):
   $$
   \left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{evap}} = \sum_{x \in \{r,s,g\}} \frac{2\pi N_{0x}(S - 1)\left[a_{vx}\lambda_x^{-2} + b_{vx}\mathrm{Sc}^{1/3}\sqrt{\frac{a_x}{\nu_a}\left(\frac{\rho_0}{\rho_d}\right)^{1/2}}\Gamma\!\left(\frac{5+b_x}{2}\right)\lambda_x^{-(5+b_x)/2}\right]}{\rho_d\left(\frac{L_{\mathrm{eff}}^2}{K_a R_v T^2} + \frac{R_v T}{D_v e_{\mathrm{sat}}(T)}\right)} \le 0.
   $$
4. **CFL-subcycled upwind sedimentation**: Mass-weighted fall speed $V_p = \omega_p V_r + (1 - \omega_p)[(1 - \omega_g)V_s + \omega_g V_g]$ with $V_x = \frac{a_x \Gamma(4+b_x)}{6\lambda_x^{b_x}}(\rho_0/\rho_d)^{1/2}$ is subcycled at $\max(V_p \Delta t_{\mathrm{sed}}/\Delta z_k) \le 0.8$ and updates moist potential temperature conservatively via $\Delta \theta_m = \frac{1 + (R_v/R_d)q_v}{\Pi c_{pd}}[\Delta(L_{\mathrm{eff},n}q_{\mathrm{cond}}) + \Delta(L_{\mathrm{eff},p}q_p)]$ (SCALE-RM Eq. 2.64).

---

## 6. Roughness Sublayer (RSL), Implicit MYNN Level 2.5 PBL, and 3D SGS Turbulence

### 6.1 Canopy and Ocean Roughness (`src/aoba/turbulence.py`)

- **Terrestrial surfaces (Raupach, 1994, Eqs. 8, 10–11)**: Snow-exposed canopy height $h_{\mathrm{exp}}$ and LAI $L_{\mathrm{exp}}$ set $d/h_{\mathrm{exp}}$ and $z_{0m}/h_{\mathrm{exp}}$ continuously with bare soil $z_{0m,\mathrm{soil}} = 0.01\ \mathrm{m}$, snow $z_{0m,\mathrm{snow}} = 0.002\ \mathrm{m}$, and Stanton ratio $z_{0h} = z_{0q} = e^{-2} z_{0m}$.
- **Open ocean (Miller, Beljaars, & Palmer, 1992, Eqs. 3–5)**:
  $$
  z_{0m} = 0.11\frac{\nu_a}{u_*} + \frac{\alpha_{\mathrm{Ch}} u_*^2}{g}, \qquad z_{0h} = 0.40\frac{\nu_a}{u_*} + 1.4\times 10^{-5}, \qquad z_{0q} = 0.62\frac{\nu_a}{u_*} + 1.3\times 10^{-4}.
  $$

### 6.2 Analytical Exponential-Integral RSL Surface-Layer Similarity (De Ridder, 2010)

At reference height $z = \tilde{z}_{\mathrm{ref}}$ with gustiness-floored wind $U_{\mathrm{eff}} = \sqrt{u_1^2 + v_1^2 + 0.25}$, $\zeta = z / L_{\mathrm{MO}}$ is solved iteratively from $\mathrm{Ri}_B$ using Paulson ($\zeta < 0$) and Beljaars–Holtslag ($\zeta \ge 0$) functions supplemented inside canopy depth $z_* = \max(2h_{\mathrm{exp}} - d,\ 0)$ by De Ridder (2010, Eqs. 6–7, 22):
$$
\begin{aligned}
F_\alpha &= \ln\!\left(\frac{z}{z_{0\alpha}}\right) - \psi_\alpha(\zeta) + \psi_\alpha\!\left(\zeta \frac{z_{0\alpha}}{z}\right) - \left[\hat{\psi}_\alpha\!\left(\zeta \frac{z_{0\alpha}}{z}, \chi_{0\alpha}\right) - \hat{\psi}_\alpha(\zeta, \chi)\right], \\
\hat{\psi}_\alpha(\zeta, \eta) &= \phi_\alpha\!\left(\zeta \frac{1 + \nu_{\mathrm{RSL}}/\mu_\alpha}{\eta}\right)\left(1 - \frac{1}{c_{2\alpha}}\right) e^{-\mu_\alpha \eta}\left[\ln(\mu_\alpha \eta) + E_1(\mu_\alpha \eta)\right],
\end{aligned}
$$
yielding $C_M = \kappa^2 / F_M^2,\ C_H = \kappa^2 / (F_M F_H),\ C_Q = \kappa^2 / (F_M F_Q)$.

### 6.3 Implicit MYNN Level 2.5 PBL and 3D Smagorinsky–Lilly SGS

Doubled TKE $q^2 = 2e$ (Nakanishi & Niino, 2009; SCALE-RM Sec. 8.2.1 Eqs. 8.96–8.176) uses master length scale $L^{-1} = L_S^{-1} + L_T^{-1} + L_B^{-1}$ ($L_T = 0.23 \int q \tilde{z} dz / \int q dz$), Gaussian subgrid condensation variance $\sigma_s$ yielding partial cloudiness $R$ and moist buoyancy $\beta_\theta, \beta_q$, Level-2 equilibrium $q_{\mathrm{L2}}^2$ with disequilibrium factor $\alpha_c = \min(q/q_{\mathrm{L2}}, 1)$, and realizable stability functions $(S_M, S_H)$:
$$
S_M = \alpha_c A_1 \frac{\phi_3 - 3 C_1 \phi_4}{\phi_2 \phi_4 + \phi_5 \phi_3}, \qquad S_H = \alpha_c A_2 \frac{\phi_2 + 3 C_1 \phi_5}{\phi_2 \phi_4 + \phi_5 \phi_3}, \qquad \frac{\partial(\rho_d q^2)}{\partial t} = \frac{\partial}{\partial z}\!\left(\rho_d K_q \frac{\partial q^2}{\partial z}\right) + 2\rho_d\left(K_M S^2 - K_H N^2 - \frac{q^3}{B_1 L}\right),
$$
integrated implicitly with `solve_tridiagonal_batched`. Horizontal 3D CRM SGS mixing uses Richardson-stabilized Smagorinsky–Lilly with Scotti et al. (1993) grid aspect correction $f(a)$: $\nu_{\mathrm{SGS}} = (C_s f(a)\Delta)^2 |\mathbf{S}|\sqrt{\max(0, 1 - \mathrm{Ri}/\mathrm{Ri}_c)}$.

---

## 7. Two-Leaf Non-Rectangular-Hyperbola EEO Biosphere

### 7.1 Least-Cost C₃/C₄ Optimality and Universal Dark Respiration (`src/aoba/biosphere.py`)

Without PFT tables, C₃ optimality (Prentice et al., 2014; Smith et al., 2019, Eqs. 1–20) sets $\chi_{\mathrm{C3}} = c_i/c_a$ from $\beta_{\mathrm{C3}} = 146$ and solves exact non-rectangular-hyperbola coordination ($\theta = 0.85, c = 0.05336251$) for $(J_{\max,\mathrm{opt}}, V_{c\max,\mathrm{opt}})$ and peaked Arrhenius $25\ ^\circ\mathrm{C}$ capacities (Kattge & Knorr, 2007). Specifically, PSII quantum yield $\varphi_{\mathrm{PSII}}(T_C)$ (Bernacchi et al., 2003, normalized to $0.257$ at $20\ ^\circ\mathrm{C}$), entropy-acclimated peaked Arrhenius kinetics $f_P(T, T_g, H_a)$ ($H_d = 200{,}000\ \mathrm{J\ mol^{-1}}, T_0 = 298.15\ \mathrm{K}, T_{g,C} = \mathrm{clip}(T_g - 273.15, 0, 40)$), and relative water viscosity $\eta^*(T)$ are given by:
$$
\begin{aligned}
\varphi_{\mathrm{PSII}}(T_C) &= 0.257 \cdot \frac{\max\!\left(0.352 + 0.022 T_C - 3.4\times 10^{-4} T_C^2,\ 0.02\right)}{0.656}, \qquad T_C = \mathrm{clip}(T - 273.15,\ -5,\ 50), \\
f_P(T, T_g, H_a) &= \exp\!\left[\frac{H_a (T - T_0)}{R_{\mathrm{gas}} T_0 T}\right]\frac{1 + \exp\!\left(\dfrac{T_0 \Delta S - H_d}{R_{\mathrm{gas}} T_0}\right)}{1 + \exp\!\left(\dfrac{T \Delta S - H_d}{R_{\mathrm{gas}} T}\right)}, \qquad \Delta S = \begin{cases} 668.39 - 1.07\, T_{g,C} & (V_{c\max}, V_{p\max}), \\ 659.70 - 0.75\, T_{g,C} & (J_{\max}), \end{cases} \\
\eta^*(T) &= \exp\!\left[\frac{580}{T - 138} - \frac{580}{T_0 - 138}\right], \qquad V_{c\max 25} = \frac{V_{c\max,\mathrm{opt}}}{\max(f_P(T_g, T_g, H_{a,V}), 0.05)}, \qquad J_{\max 25} = \frac{J_{\max,\mathrm{opt}}}{\max(f_P(T_g, T_g, H_{a,J}), 0.05)}.
\end{aligned}
$$
C₄ optimality (Scott & Smith, 2022, Eqs. 1–25) determines $\chi_{m,\mathrm{C4}}$ ($\beta_{\mathrm{C4}} = 166$) and iterates bundle-sheath pressure $C_{bs}$ under leakiness $\phi_L = 0.20$, $g_{bs} = 0.003 L_{\mathrm{exp}}$, and 5-electron ATP stoichiometry to yield $(J_{\max,\mathrm{C4}}, V_{p\max,\mathrm{C4}}, V_{c\max,\mathrm{C4}})$. Nighttime respiration scales as $R_{\mathrm{leaf}}(T_{\mathrm{night}}) = 0.018\, V_{c\max,\mathrm{opt}}$ with MMRT temperature function $f_R(T) = \exp[0.1012(T_C - 25) - 0.0005(T_C^2 - 25^2)]$ (Heskel et al., 2016; Ren et al., 2024, 2025).

### 7.2 Midday Downward-SW EMA and Closed-Form Two-Leaf Allocation

Local-solar-noon $(\mathrm{VIS}_b, \mathrm{VIS}_d, \mu_0)$ are EMA-smoothed ($\alpha_{\mathrm{EMA}} = 0.067$), routed through spherical-leaf canopy radiative transfer (de Pury & Farquhar, 1997) at current $L_{\mathrm{exp}}$ to obtain acclimated $\tilde{I}_c$, and analytically integrated against instantaneous sunlit probability $\exp(-k_{db}L)$ ($k_{db} = 0.5/\mu_0(t)$) to partition every capacity $X \in \{V_{c\max 25}, J_{\max 25}, V_{p\max 25}, R_{25}\}$ into $X_{\mathrm{sun}}$ and $X_{\mathrm{sh}} = \max(10^{-6}, X - X_{\mathrm{sun}})$. Sub-daily sunlit and shaded leaves independently evaluate assimilation (scaled by non-empirical time-integrated soil-moisture factor $\beta_{\mathrm{opt}}$), optimal stomatal conductances $g_{sw,\mathrm{sun}}, g_{sw,\mathrm{sh}}$, two-stream canopy albedos (Sellers, 1985; Dai et al., 2004, App. A, with spherical limit $\bar{\mu}=1.0, K=0.5/\max(\mu_0,0.02), \beta=0.5$ and smooth $\sigma$-singularity regularization $K_{\mathrm{reg}} = K + 2\times 10^{-4}$ when $|\sigma_{\mathrm{raw}}| < 10^{-4}$), autonomous C₄ dominance $F_4 \in [0, 1]$ (Luo et al., 2024), autonomous evergreen vs. deciduous leaf-habit optimality $f_{\mathrm{evg}} \in [0, 1]$ (Kikuzawa, 1991; Sakai & Larcher, 1987; Givnish, 2002; Wang et al., 2023), and dual-timescale Lambert-$W_0$ prognostic LAI (Cai et al., 2025; Zhou et al., 2025; without double-counting $\beta_{\mathrm{opt}}$ inside daily potential GPP $A_{0,\mathrm{day}}$). Following Lu, Duursma, & Medlyn (2016) and Holtzman et al. (2024, Eqs. 2, 7 & Fig. 4), the ecosystem water-saving timescale $\tau$ is determined solely from the climatological growing-season maximum consecutive dry-day length $D_{\max}$ ($P < 0.5\ \mathrm{mm\ d^{-1}}, T > 0\ ^\circ\mathrm{C}$), yielding:
$$
\tau = 0.85\, D_{\max} + 6.59\ [\mathrm{days}], \qquad \beta_{\mathrm{opt}} = \mathrm{clip}\!\left(\sqrt{\frac{\max(W_{\mathrm{eff}} - 0.05\, S_R,\, 0)}{\max(\tau\, E_{\mathrm{pot,EMA}},\, 0.10\, S_R)}},\ 0,\ 1\right),
$$
where $W_{\mathrm{eff}} = f_{\mathrm{unfrozen}} W_{\mathrm{root}}$ is unfrozen root-zone water and $E_{\mathrm{pot,EMA}}$ is the 15-day EMA potential ET demand evaluated from Priestley & Taylor (1972, *Mon. Wea. Rev.*, 100, 81–92), $E_{\mathrm{pot}} = 1.26\,\frac{s}{s + \gamma}\,\frac{\max(R_n,\, 0)}{L_{v0}}$ with Clausius–Clapeyron slope $s = \mathrm{d}e_s / \mathrm{d}T$, psychrometric constant $\gamma = c_{pd} p_{\mathrm{sfc}} / (\varepsilon_w L_{v0})$, and surface net radiation $R_n = S_{\mathrm{net}} + \varepsilon_{\mathrm{sfc}}(L^\downarrow - \sigma T_{\mathrm{rad,sfc}}^4)$.

Using 8 global physiological parameters (no PFT maps), evergreen vs. deciduous leaf habit emerges from Kikuzawa (1991, Eq. 5) leaf-longevity carbon profit optimization combined with cold-hardening leaf construction cost ($f_{\mathrm{cold}} = \sigma[(T_0 - T_{\min,\mathrm{EMA}})/5]$; Wang et al., 2023), deciduous spring-flush lag ($\Delta t_{\mathrm{flush}} = 25\ \mathrm{d}$), temperature-dependent off-season respiration ($Q_{10} \approx 2$, $R_{\mathrm{off},25} = 32\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$), and extreme continental winter frost damage below $T_{\mathrm{frost}} = 233.15\ \mathrm{K}$ ($-40\ ^\circ\mathrm{C}$; Sakai & Larcher, 1987; Givnish, 2002):
$$
\begin{aligned}
C_{\mathrm{evg}} &= (1.35 + 0.55 f_{\mathrm{cold}})\, z, \qquad \eta_{\mathrm{evg}} = \max(1 - 0.14 f_{\mathrm{cold}},\ 0.40), \qquad \tau_{\mathrm{evg}} = \mathrm{clip}\!\left(\sqrt{\frac{2 b_{\mathrm{evg}} C_{\mathrm{evg}}}{\eta_{\mathrm{evg}} a_{\mathrm{leaf}}}},\ 1,\ 7\right)\ [\mathrm{yr}], \\
G_{\mathrm{dec}} &= \left(1 - \frac{\Delta t_{\mathrm{flush}}}{365.24 f_{\mathrm{GSL}}}\right)_+ \left(1 - \frac{f_{\mathrm{GSL}}}{2 b_{\mathrm{dec}}}\right) a_{\mathrm{leaf}} - z, \\
G_{\mathrm{evg}} &= \eta_{\mathrm{evg}} \left(1 - \frac{\tau_{\mathrm{evg}}}{2 b_{\mathrm{evg}}}\right) a_{\mathrm{leaf}} - \frac{C_{\mathrm{evg}}}{\tau_{\mathrm{evg}}} - R_{\mathrm{off},25} e^{0.0693(T_{\mathrm{off}} - 298.15)}(1 - f_{\mathrm{GSL}}) - C_{\mathrm{frost}}\,\sigma\!\left(\frac{T_{\mathrm{frost}} - T_{\min,\mathrm{EMA}}}{4}\right), \\
f_{\mathrm{evg}}^{\mathrm{target}} &= \sigma\!\left(\frac{4(G_{\mathrm{evg}} - G_{\mathrm{dec}})}{z}\right), \qquad z_{\mathrm{eff}} = (1 - f_{\mathrm{evg}})\, z + f_{\mathrm{evg}} \frac{C_{\mathrm{evg}}}{\tau_{\mathrm{evg}}}, \qquad \mathrm{LAI} = \mathrm{LAI}_{\mathrm{evg}} + \mathrm{LAI}_{\mathrm{dec}},
\end{aligned}
$$
where $b_{\mathrm{evg}} = 15\ \mathrm{yr}$, $b_{\mathrm{dec}} = b_{\mathrm{evg}}/2.5$, $C_{\mathrm{frost}} = 18\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$, and $a_{\mathrm{leaf}} = A_{0,\mathrm{fav,yr}} \mathrm{fAPAR}_{\max,0} / \max(\mathrm{LAI}_{\max,0}, 0.50)$ is the moisture-gated favorable-season canopy assimilation per unit LAI. The evergreen canopy $\mathrm{LAI}_{\mathrm{evg}}$ relaxes toward $f_{\mathrm{evg}} \mathrm{LAI}_{\max}(z_{\mathrm{eff}})$ at multi-year needle-turnover rate $(365.24 \tau_{\mathrm{evg}})^{-1}$ (augmented by acute winter frost-kill rate $\alpha_{\mathrm{frost}} = 0.025\,\sigma[(T_{\mathrm{frost}} - T_{\min,\mathrm{day}})/4]$ and drought dieback when $\mathrm{LAI}_{\mathrm{evg}} > \mathrm{LAI}_{\max}$), while the deciduous canopy $\mathrm{LAI}_{\mathrm{dec}}$ relaxes toward $(1 - f_{\mathrm{evg}}) L_s$ at the 15-day phenological rate $\alpha_{\mathrm{EMA}} = 0.067$. This autonomously produces year-round evergreen tropical rainforests, winter-retaining boreal taiga, seasonal temperate/monsoon deciduous forests, and winter-deciduous continental *Larix* taiga below $-40\ ^\circ\mathrm{C}$. Root-zone storage $S_R \in [50, 650]\ \mathrm{kg\ m^{-2}}$ (Gao et al., 2014) and vascular canopy height $h_c \in [0.05, 40]\ \mathrm{m}$ (Kempes et al., 2011) evolve dynamically via:
$$
\begin{aligned}
a_s &= \frac{\omega}{2}\frac{0.5}{0.5 + \mu_s}\left[1 - \frac{\mu_s}{0.5 + \mu_s}\ln\!\left(\frac{0.5 + \mu_s}{\mu_s}\right)\right], \qquad \beta_0 = \mathrm{clip}\!\left(\frac{(1 + \bar{\mu}K)a_s}{\omega \bar{\mu} K},\ 0.05,\ 0.95\right), \\
D_{\mathrm{cum}}^{(d)} &= \mathrm{clip}\!\left(D_{\mathrm{cum}}^{(d-1)} + (\mathrm{fAPAR}_{\max} E_{\mathrm{pot}} - P),\ 0,\ 800\right), \qquad S_{R,\mathrm{target}} = \mathrm{clip}\!\left(50 + 1.35\, D_{\mathrm{cum}}^{(d)},\ 50,\ 650\right), \\
h_{c,\mathrm{target}} &= \mathrm{clip}\!\left[32.0 \left(\mathrm{clip}\!\left(\frac{P_{\mathrm{yr}}}{\max(E_{\mathrm{pot,yr}}, 50)},\ 0,\ 2.5\right)\right)^{0.85} \left(\mathrm{clip}\!\left(\frac{\mathrm{fAPAR}_{\max} A_{0,\mathrm{yr}}}{180},\ 0,\ 2.0\right)\right)^{0.75},\ 0.05,\ 40.0\right].
\end{aligned}
$$

---

## 8. Land Hydrology, Five-Layer Freeze–Thaw Soil, Snow, and Ocean/Sea-Ice

### 8.1 Three-Source Canopy–Ground Energy Balance, RSL/P-Model Coupling, Snowpack, and 5-Layer Freeze–Thaw Richards Soil (`src/aoba/surface.py`)

Vegetated land ($L_{\mathrm{exp}} \ge 0.01$) resolves an explicit 3-source temperature network—sunlit leaves $T_{\ell,\mathrm{sun}}$, shaded leaves $T_{\ell,\mathrm{sh}}$, and ground/snow $T_g$—coupled through a diagnostic zero-storage canopy-air node $(T_{\mathrm{af}}, q_{\mathrm{af}})$ to atmospheric aerodynamic conductance $g_a$ (Dai et al., 2004, Sec. 2b–c; Bonan et al., 2018, Eqs. 8–13, 21–24; CLM5 Sec. 5.2–5.3). To prevent $k B^{-1}$ double-counting when leaf boundary-layer resistance $r_b$ is explicit, vegetated canopies smoothly blend scalar roughness to $z_{0h,\mathrm{can}} = z_{0m,\mathrm{can}}$ (Oleson et al., 2013, p. 95) while bare soil/snow retains $z_{0h} = 0.135335\, z_{0m}$. Within-canopy RSL exponential wind decay $u(L) = u(h)e^{-\alpha_u L}$ yields closed-form distinct boundary-layer conductances $(g_{b,\mathrm{sun}}, g_{b,\mathrm{sh}})$ (two-sided sensible conductance $2 g_{b,j}$), while bare-soil resistance uses the CLM5 dry-surface-layer scheme $r_{\mathrm{soil}} = D_{\mathrm{SL}}/(D_v \tau)$ (Swenson & Lawrence, 2014; CLM5 Eqs. 5.76–5.81) and ground pathways $g_{\mathrm{soil}} = (1 - A_{\mathrm{sn}})/(r_{d,\mathrm{uc}} + r_{\mathrm{soil}})$, $g_{\mathrm{snow}} = A_{\mathrm{sn}} / r_{d,\mathrm{uc}}$ (CLM5 Eq. 5.110; Sakaguchi & Zeng, 2009) with a single $g_a^{-1}$ in series. Wetted foliage $f_{\mathrm{wet}} = \min\{[W_c/(p_{\mathrm{liq}} L)]^{2/3}, f_{\mathrm{wet,max}}\}$ ($p_{\mathrm{liq}} = 0.1\ \mathrm{kg\ m^{-2}}$; CLM5 Eq. 7.23) drives water-limited interception loss $f'_{\mathrm{wet}} g_{b,j}$ (shortfall returned as sensible heat; CLM5 Eqs. 5.111, 5.139) fed by liquid-only interception $f_{\mathrm{pi}} = \alpha_{\mathrm{liq}} \tanh L$ (CLM5 Eq. 7.4). Inside each of the `land_eb_iterations = 3` coupled $3\times 3$ Newton iterations, the two-leaf P-model (`compute_two_leaf_gas_exchange`) is re-evaluated at the current $(T_{\ell,\mathrm{sun}}, T_{\ell,\mathrm{sh}})$ and coupled to leaf-surface microclimate $(c_s, D_s)$ via $c_s = c_{\mathrm{af}} - A_n P / g_{bc}$ ($g_{bc} = g_{bw}/1.37$; Dai et al., 2004, Eq. 11) and $D_s = [g_{bw}/(g_{bw} + g_{sw})] D_{\ell,\mathrm{af}}$ (Wang et al., 2017; Bonan et al., 2018), yielding pure stomatal conductances $(g_{sw,\mathrm{sun}}, g_{sw,\mathrm{sh}})$ in series with $(g_{b,\mathrm{sun}}, g_{b,\mathrm{sh}})$ and exact radiative/turbulent energy conservation ($S_{\mathrm{sun}} + S_{\mathrm{sh}} + S_g = S_{\mathrm{abs}}$; 1-bounce LW closure, CLM5 Sec. 4.2); snow-covered ground $T_g$ is clamped at $T_0$ to drive melt $M_{\mathrm{snow}}$ (MATSIRO6 Sec. 6.3.3). Subgrid topographic snow fraction $A_{\mathrm{sn}}$ (Swenson & Lawrence, 2012; Nitta et al., 2014), snow-burial masking $h_{\mathrm{exp}} = \max(h_c - d_{\mathrm{sn}}, 0)$ (MATSIRO6 Eqs. 2.1–2.3), two-band snow aging albedo (Yang et al., 1997), and glacier runoff calving ($S_{n,\max} = 1000\ \mathrm{kg\ m^{-2}}$) couple with 5-layer ($\Delta z_s = [0.05, 0.15, 0.40, 0.80, 1.60]\ \mathrm{m}$) mass-conserving implicit Richards soil hydrology ($K_{\mathrm{sat}} = 7\times 10^{-6}\ \mathrm{m\ s^{-1}}$, Clapp & Hornberger 1978 retention/conductivity, ice impedance $f_{\mathrm{ice},k} = 10^{-\Omega_{\mathrm{ice}}\theta_{\mathrm{ice},k}/\phi}$ with $\Omega_{\mathrm{ice}} = 6$, map-free saturation/infiltration-excess runoff, free bottom gravity drainage $q_{\mathrm{drain}} = K_4$, and Gao et al. 2014 $S_R$-derived dynamic rooting depth $z_{\mathrm{root}} = \mathrm{clip}(S_R / [\rho_w(\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}})], 0.15, 3.0\ \mathrm{m})$ whose vertically integrated liquid plant-available water $W_{\mathrm{avail}}$ drives Holtzman et al. 2024 $\beta_{\mathrm{opt}}$; CLM5 Sec. 7.3–7.5) and internal-energy $U_l = C_l(T_l - T_0) - L_{f0} M_{i,l}$ freeze–thaw heat conduction (`solve_tridiagonal_batched`). Open water uses Briegleb et al. (1986) direct albedo, and fractional sea ice $f_i = \min(\sqrt{M_i/300}, 1)$ solves area-weighted Semtner (1976) lead/ice thermodynamics.

### 8.2 Map-Free Mineral Dust Mobilization and Transport (`src/aoba/dust.py`)

Mineral dust emission follows the CLM5 DEAD scheme (Lawrence et al., 2020, Ch. 31, Eqs. 31.1–31.14; Oleson et al., 2013, Ch. 24; Zender et al., 2003; Mahowald et al., 2006) without regional erodibility maps ($S = 1$), governed by one global tuning factor $T = \text{dust\_emis\_factor} = 1.5\times 10^{-5}$ (calibrated for map-free $S = 1, M_{\mathrm{clay}} = 0.20$):
$$
F_{\mathrm{dust}} = T\, f_m\, \alpha\, Q_s \sum_{i=1}^{3} M_i, \qquad f_m = (1 - A_{\mathrm{sn}})(1 - f_v)\frac{w_{\mathrm{liq},1}}{w_{\mathrm{liq},1} + w_{\mathrm{ice},1}}, \qquad f_v = \mathrm{clip}\!\left(\frac{L_{\mathrm{exp}}}{0.3},\,0,\,1\right).
$$
With universal loam clay fraction $M_{\mathrm{clay}} = 0.20$ and sandblasting efficiency $\alpha = 100\exp[(13.4 M_{\mathrm{clay}} - 6.0)\ln 10]\ \mathrm{m^{-1}}$ (Marticorena & Bergametti, 1995), the dry Iversen & White (1982) threshold friction speed for optimal saltation diameter $D_{\mathrm{osp}} = 75\ \mu\mathrm{m}$ ($\rho_{\mathrm{osp}} = 2650\ \mathrm{kg\ m^{-3}}$) is scaled by the Fécan et al. (1999) moisture factor $f_w = \sqrt{1 + 1.21 [100(w - w_t)]^{0.68}}$ for gravimetric topsoil water $w = 1000(w_{\mathrm{liq},1}+w_{\mathrm{ice},1}) / [(1 - \phi_{\mathrm{soil}})\rho_{\mathrm{solid}}] > w_t = 0.17 + 0.14 M_{\mathrm{clay}}$ (using the CLM tuning multiplier $a = M_{\mathrm{clay}}^{-1}$). The saltation friction speed includes the Owen (1964) effect $u_{*s} = u_* + 0.003(U_{10} - U_{10,t})^2$ ($U_{10} \ge U_{10,t}$), driving the White (1979) streamwise flux $Q_s = c_s (\rho_{\mathrm{atm}} u_{*s}^3 / g)(1 - r)(1 + r)^2$ ($r = u_{*t}/u_{*s} < 1$, $c_s = 2.61$), integrated over the $0.1\text{–}10\ \mu\mathrm{m}$ transported dust fraction ($\sum M_i = 0.871$). Column dust mixing ratio $q_{\mathrm{dust}}$ undergoes PBL mixing, implicit gravitational settling ($v_{\mathrm{settle}} = 4\times 10^{-3}\ \mathrm{m\ s^{-1}}$) / surface dry deposition ($v_{\mathrm{dep}} = 6\times 10^{-3}\ \mathrm{m\ s^{-1}}$), precipitation washout ($\alpha_{\mathrm{scav}} = 0.50\ \mathrm{m^2\ kg^{-1}}$), and broadband SW/LW radiative coupling (Balkanski et al., 2007).

---

## 9. Embedded 3D Nonhydrostatic RK3–HEVI CRM and Zalesak FCT

### 9.1 Compressible RK3–HEVI Solver and 3D FCT (`src/aoba/crm.py`)

Each SCVT cell embeds a doubly periodic orthogonal C-grid / Charney–Phillips CRM (default `[Nz=32, Ny=1, Nx=8], dx=4000 m`, configurable via TOML `[grid]`) integrating flux-form $(\rho_d, u, v, w, \Theta_m = \rho_d \theta_m)$ via 3-stage Wicker & Skamarock (2002) RK3, 3rd-order upwind-biased horizontal momentum advection, and off-centered ($\beta_{\mathrm{HEVI}} = 0.60$) Horizontally Explicit / Vertically Implicit acoustic substeps solving a $(N_z - 1)\times (N_z - 1)$ tridiagonal Helmholtz equation for $\delta F_z$ via `solve_tridiagonal_batched`. To prevent double-counting background Coriolis acceleration against the host pressure-gradient balance (Khairoutdinov et al., 2005; Yamazaki & Miura, 2024), CRM Coriolis is switched automatically by dimensionality: disabled ($f = 0$) for 1D (`1×1`) and 2D (`nx==1` or `ny==1`) curtains, and applied strictly to perturbation winds $\mathbf{u}' = \mathbf{u} - \overline{\mathbf{u}}$ in 3D (`nx>1, ny>1`). Moist scalars $(q_n, q_p)$ are transported identically by stage-consistent mass fluxes $(F_x, F_y, F_z)$ using positive-definite Zalesak (1979) 3D FCT.

---

## 10. Global SCVT C-Grid Core, Orographic Drag, Ekman Ocean, and MAML+ICA Coupling

### 10.1 SCVT TRiSK Host Dynamics, Lott–Miller Drag, Ekman Ocean, and MAML+ICA Cycle

The global host solves compressible nonhydrostatic equations on a Sphere Centroidal Voronoi C-grid ($N_{\mathrm{cells}} = 10\cdot 4^g + 2$; Ringler et al., 2008, 2010; Thuburn et al., 2009) using skew-symmetric TRiSK Coriolis reconstruction ($A_e w_{e,e'} + A_{e'} w_{e',e} = 0$), Klemp STF metric terms, HEVI vertical coupling, and SCVT Zalesak FCT, coupled to `Earth2014` subgrid Lott & Miller (1997) blocked-flow + Lindzen gravity-wave drag (`src/aoba/orography.py`) and a zero-$Q$-flux 1.5-layer Voronoi Ekman + reduced-gravity shallow-water ocean ($H_2 = 200\ \mathrm{m}$, $g' = 0.03\ \mathrm{m\ s^{-2}}$, TRiSK Coriolis, scale-aware Munk eddy viscosity, sea-ice stress attenuation $1 - 0.5 f_{\mathrm{ice}}$, coastal zero normal velocity/transport, geostrophic + Ekman decomposition $\mathbf{M}_1 = \mathbf{M}_E + \rho_w H_1 \mathbf{u}_g$ with variable Layer-2 thickness $H_2 = \max(H_{\mathrm{sw}} - H_1, 20\ \mathrm{m})$ and exact enthalpy-conserving return flux $\mathbf{M}_2 = \rho_w \mathbf{V}_{\mathrm{sw}} - \mathbf{M}_1$, conservative glacial runoff latent-heat absorption; `src/aoba/ocean_ekman.py`, Codron, 2012). To prevent `float32` swamping loss on MPS GPUs when accumulating small heat fluxes onto large-heat-capacity ocean temperatures ($~280\ \mathrm{K}$), both `ocean_ekman.py` ($T_1, T_2$) and `surface.py` (atmosphere-ocean mixed-layer flux $T_1$) use **Kahan/Neumaier compensated summation** ($y = \Delta T + c,\ T' = T + y,\ c' = (T - T') + y$). Under the **MAML + ICA** coupling architecture (`src/aoba/coupled_model.py`), every `[Ny, Nx]` CRM subcolumn maintains its own full 1-to-1 land/biosphere/soil/ocean column state `sfc_sub`, receives independent 4-band ICA radiation fluxes (`rad_stride = 5` steps), advances `step_surface` locally, synchronizes ocean heat/ice increments with the SCVT Ekman/shallow-water ocean, injects large-scale host tendencies into `step_crm_physics_and_dynamics`, and feeds back domain-mean thermodynamic/momentum tendencies while diagnosing online 30-class Köppen–Geiger biomes (Beck et al., 2018, 2023).

---

## 11. Dynamic / Accelerated SIA Ice-Sheet Model and Conservative Barometric Remapping

Enabled via `enable_dynamic_icesheet = true` (`src/aoba/icesheet.py`), AOBA prognoses continental ice thickness $H(\mathbf{x})$ and bedrock elevation $b(\mathbf{x})$ initialized from `Earth2014.BED2014` and `ICE2014`.

### 11.1 Climatic Surface Mass Balance (SMB) and EMA Filtering

Over terrestrial/ice-sheet cells, subgrid net ice mass flux $\mathrm{SMB}\ [\mathrm{kg\ m^{-2}\ s^{-1}}]$ (snowfall minus sublimation, snowmelt, and bare-ice surface melt potential at $T_{\mathrm{skin}} > T_0$) is accumulated each climate year into $\overline{\mathrm{SMB}}_y$ and smoothed with exponential moving average memory $\tau_{\mathrm{EMA}}$ (default `5.0 yr`):
$$
\widetilde{\mathrm{SMB}}_y = (1 - \alpha_{\mathrm{EMA}})\,\widetilde{\mathrm{SMB}}_{y-1} + \alpha_{\mathrm{EMA}}\,\overline{\mathrm{SMB}}_y, \qquad \alpha_{\mathrm{EMA}} = 1 - \exp\!\left(-\frac{\Delta t_{\mathrm{clim,yr}}}{\tau_{\mathrm{EMA}}}\right), \qquad b_{\mathrm{ice}} = \frac{\widetilde{\mathrm{SMB}}_y}{\rho_{\mathrm{ice}}}.
$$

### 11.2 Glen Flow Law ($n=3$) Upstream Finite-Volume SIA and Time Acceleration

Using isothermal Glen's flow law ($n = 3, A_{\mathrm{Glen}} = 1.0\times 10^{-16}\ \mathrm{Pa^{-3}\ yr^{-1}}, \rho_{\mathrm{ice}} = 917\ \mathrm{kg\ m^{-3}}$) and ice surface $s = b + H$, 2D Shallow Ice Approximation flow obeys:
$$
\frac{\partial H}{\partial t_{\mathrm{ice}}} = \nabla_{\mathrm{SCVT}} \cdot \left(D_{\mathrm{SIA}}\nabla s\right) + b_{\mathrm{ice}}, \qquad D_{\mathrm{SIA}} = \Gamma_{\mathrm{SIA}}\, H^5 |\nabla s|^2, \qquad \Gamma_{\mathrm{SIA}} = 0.4 A_{\mathrm{Glen}} (\rho_{\mathrm{ice}} g)^3.
$$
On each SCVT edge $e = (c_0 \to c_1)$, upstream Type-I diffusivity evaluation (Mahaffy, 1976; Bueler et al., 2005) selects upwind ice thickness $H_{\mathrm{up}, e}$ along descending surface slope $(s_{c_1} - s_{c_0})/d_{c,e}$ and updates $H \in [0, 6000\ \mathrm{m}]$ via CFL-subcycled (`D * dt_sub / dx_min^2 <= 0.12`) finite-volume divergence `compute_scvt_divergence`, calving ice discharged onto ocean cells. Asynchronous ice integration proceeds for $\Delta t_{\mathrm{ice}} = \alpha_{\mathrm{eff}}(t)\Delta t_{\mathrm{clim}}$:
$$
\alpha_{\mathrm{eff}}(t) = \begin{cases} \alpha_0 & (\tau_{\mathrm{spin}} \le 0), \\ 1 + (\alpha_0 - 1)\exp\!\left(-t_{\mathrm{clim}} / \tau_{\mathrm{spin}}\right) & (\tau_{\mathrm{spin}} > 0). \end{cases}
$$

### 11.3 Relaxed Bedrock Isostasy, Mask Transitions, and Barometric Vertical Remapping

Subglacial bedrock relaxes toward mantle equilibrium ($\rho_m = 3300\ \mathrm{kg\ m^{-3}}$) over timescale $\tau_{\mathrm{iso}}$ (default `1000 yr`):
$$
b_{\mathrm{eq}} = b_0 - \frac{\rho_{\mathrm{ice}}}{\rho_m} H, \qquad b^{\mathrm{new}} = b + \left[1 - \exp\!\left(-\frac{\Delta t_{\mathrm{ice}}}{\tau_{\mathrm{iso}}}\right)\right](b_{\mathrm{eq}} - b), \qquad z_{\mathrm{topo}}^{\mathrm{new}} = b^{\mathrm{new}} + H,
$$
followed by 2-pass Klemp conservative neighbor Laplacian smoothing ($z \leftarrow 0.75 z + 0.25 \bar{z}_{\mathrm{nb}}$). Crossings of $H_{\mathrm{crit}} = 5.0\ \mathrm{m}$ trigger dynamic `SURF_LAND <-> SURF_ICESHEET` mask transitions and seed pioneer vegetation ($\mathrm{LAI} = 0.25, h_c = 0.3\ \mathrm{m}$) onto newly deglaciated land. Whenever $\max |z_{\mathrm{topo,smooth}}^{\mathrm{new}} - z_{w,0}^{\mathrm{old}}| \ge 0.05\ \mathrm{m}$, `remap_atmosphere_and_crm_vertical` rebuilds $z_w^{\mathrm{new}}, z_c^{\mathrm{new}}$ via `compute_stf_heights`, interpolates $(\ln p, \theta_m, q_n, q_p, q_v, q_c, q_i, u, v, w, q^2)$ onto $z_c^{\mathrm{new}}$ with hydrostatic barometric extrapolation below old surface elevation:
$$
p(z_c^{\mathrm{new}}) = p_0^{\mathrm{old}} \left[1 - \frac{g (z_c^{\mathrm{new}} - z_{c,0}^{\mathrm{old}})}{c_{pd}\theta_{m,0}^{\mathrm{old}}\Pi_0^{\mathrm{old}}}\right]^{c_{pd}/R_d},
$$
and consistently re-diagnoses compressible density $\rho_d^{\mathrm{new}} = p^{\mathrm{new}} / (R_d \theta_m^{\mathrm{new}} \Pi^{\mathrm{new}})$ across both host and all CRM columns.

---

## 12. Unified Single-File TOML Configuration, Output Cadence, GUI, and 216-Parameter Registry

### 12.1 TOML Architecture & Interactive Configurator (`src/aoba/params.py`, `src/aoba/runtime.py`, `scripts/aoba_config_gui.py`)

All grid parameters, orbital/planetary/calendar forcings, dynamic ice-sheet controls, output streams/cadences, and module physical constants (**13 TOML sections, 216 parameters**) are centralized in [`config/aoba_default.toml`](../config/aoba_default.toml). Loading a custom TOML via `scripts/run_g5_30yr.py --config custom.toml` binds `(AOBAConfig, OutputConfig)`, synchronizes all `src/aoba/*` module globals (`recompute_derived_constants()`), and applies hardware auto-tuning (`detect_hardware_profile()`, `compute_optimal_chunks()`, `apply_runtime_tuning()`) to size `crm_col_chunk` / `rad_col_chunk` within a 45% device-memory budget and pin CPU threads to performance cores without altering resolution.

`scripts/aoba_config_gui.py` provides a zero-dependency local Web workbench (**AOBA Studio**, via `http.server`) spanning configuration, run launching, live monitoring, and CPU real-time visualization: live 3D icosahedral SCVT + CRM + vertical-levels diagrams updating dynamically with `glevel`, planetary/paleoclimate presets (`0.5×Ω`, LGM, `4×CO2`), calendar/perpetual-insolation controls, hardware auto-tune button, output storage estimators, a 216-parameter searchable TOML editor, subprocess run launch/monitoring (`daily_global.csv`, calendar dates, and `run.log`), and an idle-CPU spherical regridder rendering **three synchronized, rotating Orthographic Earth globes for daily-mean `T2` ($0.35 T_{\mathrm{sfc}} + 0.65 T_{a,z_1}$), daily-mean `Precip [mm d⁻¹]`, and `LAI`** (with optional 1-model-day-per-revolution spin sync) from `live_preview.npz`, `snapshots.zarr`, or `monthly.zarr`.

### 12.2 Output Files & Temporal Averaging vs Instantaneous Specifications (`[output]`)

- `mesh.zarr`: Static SCVT coordinates, cell areas, topography (`z_topo`, `z_bed`, `sigma_topo`), surface masks.
- `daily_global.csv` (`daily_csv_enabled`, `daily_csv_interval_days`): Global-mean time-averaged daily time series.
- `monthly.zarr` (`monthly_mean_enabled`, `monthly_mean_periods_per_year = 12`, `monthly_mean_sample_hours = 6.0`, `monthly_mean_include_{2d,3d,subgrid_2d}`): Time-averaged 2D, 3D (`temp`, `pres`, `q_v`, `u_zonal`, `v_merid`), and CRM subgrid 2D (`lwp_sub`, `iwp_sub`, `w_max_sub`, `precip_mm_day_sub`, `sw_dn_sub`, `lw_dn_sub`, `olr_sub`, `shf_sub`, `evap_mm_day_sub`, `gpp_sub`, `t_sfc_sub`, `m_seaice_sub`, `swe_land_sub`, `w_root_sub`, `lai_sub`) climate fields.
- `snapshots.zarr` (`snapshots_enabled`, `snapshots_periods_per_year = 12`, `snapshots_include_{2d,3d,subgrid_2d}`): Instantaneous state snapshots (`export_aoba_snapshot_xarray`, including `h_canopy`, `vcmax25_c3`, `f_c4`, and `f_evg`) partitioned per year (`zarr_dir_partition = "yearly"`) or concatenated (`"single"`).
- `restart.pt` / `restart_mid.pt`: Atomic restart checkpoints written every `restart_interval_years` and at the end of every calendar month when `restart_monthly = true` (enabling lossless mid-year Zarr/CSV resume).

### 12.3 Complete 216-Parameter Registry (`config/aoba_default.toml`)

| Section (# keys) | Keys = Default Values |
| :--- | :--- |
| `[grid]` (16) | `glevel = 3`, `nz = 32`, `z_top = 24000.0`, `dz_min = 60.0`, `z_sponge_base = 17500.0`, `tau_sponge_min = 300.0`, `terrain_scale_exp = 1.5`, `crm_nx = 8`, `crm_ny = 1`, `crm_dx = 4000.0`, `dt_host = 720.0`, `n_host_dyn_substeps = 1`, `n_crm_dyn_substeps = 18`, `rad_stride = 5`, `crm_col_chunk = 16384`, `rad_col_chunk = 32768` |
| `[forcing]` (21) | `co2_ppmv = 400.0`, `solar_constant = 1361.0`, `omega = 7.292115e-5`, `obliquity_deg = 23.44`, `eccentricity = 0.0167`, `perihelion_deg = 283.0`, `start_date = "2000-03-20"`, `calendar_type = "gregorian"`, `insolation_mode = "seasonal"`, `fixed_insolation_date = "2000-03-20"`, `diurnal_cycle = true`, `use_earth2014 = true`, `earth2014_dir = "data/boundary/earth2014_5min"`, `earth2014_stride = 2`, `enable_superparam = true`, `enable_dust = true`, `dust_emis_factor = 1.5e-5`, `h_ml_ocean = 50.0`, `re_liq_um = 10.0`, `re_ice_um = 48.0`, `cloud_sw_inhomog = 0.85` |
| `[icesheet]` (5) | `enable_dynamic_icesheet = false`, `icesheet_ema_years = 5.0`, `icesheet_accel_factor = 100.0`, `icesheet_accel_spinup_years = 0.0`, `icesheet_isostasy_tau_yr = 1000.0` |
| `[output]` (20) | `out_dir = ""`, `years = 30`, `end_date = ""`, `days_per_year = 365.2425`, `zarr_dir_partition = "yearly"`, `daily_csv_enabled = true`, `daily_csv_interval_days = 1.0`, `monthly_mean_enabled = true`, `monthly_mean_periods_per_year = 12`, `monthly_mean_sample_hours = 6.0`, `monthly_mean_include_2d = true`, `monthly_mean_include_3d = true`, `monthly_mean_include_subgrid_2d = true`, `snapshots_enabled = true`, `snapshots_periods_per_year = 12`, `snapshots_include_2d = true`, `snapshots_include_3d = true`, `snapshots_include_subgrid_2d = true`, `restart_interval_years = 1`, `restart_monthly = true` |
| `[constants]` (26) | `earth_radius = 6.371e6`, `gravity = 9.79764`, `omega_earth = 7.292115e-5`, `seconds_per_day = 86400.0`, `days_per_year = 365.2425`, `r_d = 287.04`, `cp_d = 1004.64`, `r_v = 461.5`, `cp_v = 1846.0`, `p_ref = 1e5`, `t_freeze = 273.15`, `t_triple = 273.16`, `e_triple = 611.2`, `l_v0 = 2.501e6`, `l_f0 = 3.337e5`, `t_ice_all = 253.15`, `t_liq_all = 273.16`, `stefan_boltzmann = 5.670374419e-8`, `von_karman = 0.4`, `rho_seawater = 1025.0`, `cp_seawater = 3990.0`, `t_freeze_ocean = 271.35`, `rho_seaice = 917.0`, `k_therm_seaice = 2.034`, `k_therm_snow = 0.3096`, `o2_mixing_ratio = 0.20946` |
| `[radiation]` (7) | `solar_frac_sw1 = 0.45976`, `lw_window_frac = 0.32`, `o3_g1 = 3.6478`, `o3_g2 = 0.83209`, `o3_g3 = 11.3515`, `par_frac_of_sw1 = 0.88`, `umol_per_joule_par = 4.57` |
| `[microphysics]` (28) | `rho_rain = 1000.0`, `rho_snow = 100.0`, `rho_graupel = 400.0`, `n0_rain = 8e6`, `n0_snow = 3e6`, `n0_graupel = 4e6`, `a_fall_rain = 842.0`, `b_fall_rain = 0.8`, `a_fall_snow = 4.84`, `b_fall_snow = 0.25`, `a_fall_graupel = 94.5`, `b_fall_graupel = 0.5`, `e_rc = 1.0`, `e_sc = 1.0`, `e_gc = 1.0`, `e_ri = 1.0`, `e_si = 0.1`, `e_gi = 0.1`, `qc_autoconv_thresh = 0.001`, `qi_aggreg_thresh = 1e-4`, `rate_autoconv = 0.001`, `rate_ice_aggreg = 0.001`, `vt_cloud_ice = 0.4`, `cloud_droplet_conc_cm3 = 60.0`, `t0_precip = 283.16`, `t00_precip = 268.16`, `t0_graupel = 283.16`, `t00_graupel = 223.16` |
| `[turbulence]` (13) | `mynn_b1 = 24.0`, `mynn_b2 = 15.0`, `mynn_c2 = 0.75`, `mynn_c3 = 0.352`, `mynn_c5 = 0.2`, `mynn_gamma1 = 0.235`, `mynn_pr_n = 0.74`, `tke_min = 1e-4`, `rsl_mu_m = 2.59`, `rsl_mu_h = 0.95`, `rsl_nu = 0.5`, `rsl_lambda = 1.5`, `kinematic_visc_air = 1.5e-5` |
| `[surface]` (24) | `cs_dry_soil = 1.26e6`, `kappa_dry_soil = 0.3`, `soil_porosity = 0.43`, `psi_sat_soil = -0.2`, `b_clapp_soil = 5.39`, `k_sat_soil = 7e-6`, `ice_impedance_omega = 6.0`, `wc_max_per_lai = 0.1`, `alpha_intercept_liq = 1.0`, `f_wet_max = 1.0`, `dsl_dmax_mm = 15.0`, `land_eb_iterations = 3`, `rho_snowpack = 250.0`, `swe_glacier_max = 1000.0`, `tau_snow_age = 1e6`, `alb_snow_vis_new = 0.9`, `alb_snow_nir_new = 0.7`, `alb_snow_vis_old = 0.65`, `alb_snow_nir_old = 0.35`, `emissivity_land = 0.98`, `emissivity_ocean = 0.985`, `mc_seaice_crit = 300.0`, `alb_seaice_vis = 0.72`, `alb_seaice_nir = 0.42` |
| `[biosphere]` (41) | `theta_nrh = 0.85`, `cost_c_smith = 0.05336251`, `phi_psii_ref_c3 = 0.257`, `beta_c3 = 146.0`, `beta_c4 = 166.0`, `gamma_star25_c3 = 4.332`, `kc25_c3 = 41.03`, `ko25_c3 = 28210.0`, `dha_gamma = 37830.0`, `dha_kc_c3 = 79430.0`, `dha_ko_c3 = 36380.0`, `gamma_star25_c4 = 2.6`, `kp25_umol_c4 = 60.5`, `dha_kp_c4 = 27200.0`, `kc25_c4 = 121.0`, `dha_kc_c4 = 64200.0`, `ko25_c4 = 29200.0`, `dha_ko_c4 = 10500.0`, `phi_leak_c4 = 0.2`, `gbs_c4 = 0.003`, `ha_vcmax = 71513.0`, `ha_jmax = 49884.0`, `ha_vpmax = 55000.0`, `hd_enzyme = 2e5`, `b_resp_ren = 0.018`, `leaf_scat_vis = 0.15`, `leaf_scat_nir = 0.8`, `kd_prime_par = 0.719`, `rho_cd_par = 0.036`, `z_foliage_cost = 12.227`, `k_beer_lai = 0.5`, `sigma_zhou = 0.771`, `alpha_ema_acclim = 0.067`, `evg_cost_ratio_base = 1.35`, `evg_cost_ratio_cold = 0.55`, `evg_cold_eff_penalty = 0.14`, `leaf_aging_b_yr = 15.0`, `decid_flush_days = 25.0`, `evg_offseason_resp25 = 32.0`, `evg_frost_crit_k = 233.15`, `evg_frost_cost = 18.0` |
| `[ocean_ekman]` (7) | `h_thermocline = 200.0`, `r_ekman_damp_days = 2.0`, `kappa_ocn_horiz = 2000.0`, `kappa_ocn_vert = 2e-5`, `g_prime_ocn = 0.03`, `tau_sw_drag_days = 180.0`, `enable_shallow_water_ocn = true` |
| `[orography]` (3) | `hn_crit = 0.5`, `cd_block = 1.0`, `g_wave = 0.8` |
| `[icesheet_physics]` (5) | `rho_ice = 917.0`, `rho_mantle = 3300.0`, `a_glen_per_year = 1e-16`, `h_ice_threshold_m = 5.0`, `h_ice_max_m = 6000.0` |
