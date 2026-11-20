# Detailed Mathematical Formulation of AOBA

> **Language**: English | [日本語 (formulation_ja.md)](formulation_ja.md)

This document provides the complete mathematical formulation and numerical discretization of **AOBA**, written in the style of *Description of MIROC6 AGCM* (2021), *Description of MATSIRO6* (2021), and *Detailed formulation of SCALE-RM* (Nishizawa et al., 2025). Every equation corresponds to an implemented module in `src/aoba/` and cites its peer-reviewed source.

---

## 1. Physical Constants and Orbital Solar Geometry

### 1.1 Fundamental Physical Constants (`src/aoba/constants.py`)

Table 1.1 lists the fundamental planetary, thermodynamic, radiative, and oceanic constants used across AOBA, following Wing et al. (2018, Table 1), Khairoutdinov & Randall (2003, Appendix A), MIROC6 AGCM Document Writing Team (2021, Section 1.3.2), and Semtner (1976).

| Symbol | Description | Value | Units | Reference |
| :--- | :--- | :--- | :--- | :--- |
| $R_E$ | Mean Earth radius | $6.371 \times 10^6$ | $\mathrm{m}$ | Wing et al. (2018) |
| $g$ | Mean surface gravity | $9.79764$ | $\mathrm{m\ s^{-2}}$ | Wing et al. (2018) |
| $\Omega$ | Planetary rotation rate | $7.292115 \times 10^{-5}$ | $\mathrm{rad\ s^{-1}}$ | MIROC6 (2021) |
| $S_0$ | Present-day total solar irradiance | $1361.0$ | $\mathrm{W\ m^{-2}}$ | Wing et al. (2018) |
| $R_d$ | Dry-air gas constant | $287.04$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pd}$ | Dry-air isobaric specific heat | $1004.64$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{vd}$ | Dry-air isochoric specific heat ($c_{pd} - R_d$) | $717.60$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Skamarock et al. (2012) |
| $R_v$ | Water-vapor gas constant | $461.50$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pv}$ | Water-vapor isobaric specific heat | $1846.0$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $\epsilon$ | Molecular mass ratio $R_d / R_v$ | $0.62197$ | $-$ | Skamarock et al. (2012) |
| $p_0$ | Reference surface pressure | $1.0 \times 10^5$ | $\mathrm{Pa}$ | Skamarock et al. (2012) |
| $L_{v0}$ | Latent heat of vaporization at $0\ ^\circ\mathrm{C}$ | $2.501 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{f0}$ | Latent heat of fusion at $0\ ^\circ\mathrm{C}$ | $3.337 \times 10^5$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{s0}$ | Latent heat of sublimation ($L_{v0} + L_{f0}$) | $2.8347 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $\sigma$ | Stefan–Boltzmann constant | $5.670374 \times 10^{-8}$ | $\mathrm{W\ m^{-2}\ K^{-4}}$ | CODATA |

### 1.2 Orbital Insolation and Solar Zenith Angle (`src/aoba/radiation.py`)

Following Berger (1978) and Davis et al. (2017, Equations 2–8), let $t$ be elapsed simulation time in seconds, $N_d = (t / 86400) \bmod 365.2425$ the fractional day of year, $e$ the orbital eccentricity, $\varepsilon$ the obliquity, and $\tilde{\omega}$ the heliocentric longitude of perihelion. Defining mean anomaly $\nu_m = \frac{2\pi}{365.2425}(N_d - 80.0) - \tilde{\omega}$, the true anomaly $\nu$ and true heliocentric longitude $\lambda_s$ are evaluated as:
$$
\begin{aligned}
\nu &= \nu_m + \left(2e - \frac{1}{4}e^3\right)\sin\nu_m + \frac{5}{4}e^2 \sin(2\nu_m), \\
\lambda_s &= \nu + \tilde{\omega}.
\end{aligned}
$$
Solar declination $\delta_s$, instantaneous top-of-atmosphere (TOA) solar irradiance $S_{\mathrm{inst}}(t)$, and cosine of solar zenith angle $\mu_0(\varphi, \lambda, t)$ at latitude $\varphi$ and longitude $\lambda$ are given by:
$$
\begin{aligned}
\sin\delta_s &= \sin\varepsilon \sin\lambda_s, \\
S_{\mathrm{inst}}(t) &= S_0 \left(\frac{1 + e\cos\nu}{1 - e^2}\right)^2, \\
\mu_0(\varphi, \lambda, t) &= \max\!\left[0,\ \sin\varphi \sin\delta_s + \cos\varphi \cos\delta_s \cos H(t, \lambda)\right],
\end{aligned}
$$
where $H(t, \lambda) = 2\pi \left[\frac{t \bmod 86400}{86400} - \frac{1}{2}\right] + \lambda$ is the local hour angle.

---

## 2. Vertical Charney–Phillips Grid and Smoothed Terrain-Following Coordinate (`src/aoba/vertical_grid.py`)

### 2.1 Stretched Nominal Coordinate and Charney–Phillips Variable Placement

AOBA adopts $N_z = 32$ vertical layers extending from the lower aerodynamic datum ($\zeta = 0$) to $z_{\mathrm{top}} = 24,000\ \mathrm{m}$ following Charney–Phillips vertical staggering (Gassmann, 2013, Section 3):
- **Half-level interfaces** ($k + \frac{1}{2}$ for $k = 0, \dots, N_z$, stored as array indices `0 .. nz`): vertical velocity $w$, moist potential temperature $\theta_m$, physical interface height $z_w$, and radiative fluxes $F^\uparrow, F^\downarrow$.
- **Full-level layer centers** ($k \in \{0, \dots, N_z - 1\}$): dry-air density $\rho_d$, air pressure $p$, temperature $T$, horizontal velocity components, and moisture mixing ratios $(q_v, q_c, q_i, q_n, q_p)$.

Nominal layer thicknesses $\Delta \zeta_k = \zeta_{w, k+1} - \zeta_{w, k}$ stretch smoothly from $\Delta \zeta_0 = 60.0\ \mathrm{m}$ at the surface ($\sim 12$ levels below $2\ \mathrm{km}$ for boundary-layer resolution and $\sim 15$ levels across the free troposphere) to coarse buffer layers in the lower stratosphere ($17.5\text{--}24\ \mathrm{km}$), where a $\sin^2$ Rayleigh sponge profile damps upward-propagating gravity waves (Klemp et al., 2008; Skamarock et al., 2012):
$$
\gamma_R(\zeta) = \frac{1}{\tau_{\mathrm{sp,min}}} \sin^2\!\left[\frac{\pi}{2}\,\mathrm{clip}\!\left(\frac{\zeta - z_{\mathrm{sp}}}{z_{\mathrm{top}} - z_{\mathrm{sp}}},\, 0,\, 1\right)\right].
$$

### 2.2 Displacement-Following Smoothed Terrain-Following (STF) Transformation

To eliminate Monin–Obukhov surface-layer singularities over tall forest canopies ($h_c \le 45\ \mathrm{m}$) and reduce pressure-gradient truncation errors over steep mountains, the lower coordinate boundary is placed at the effective aerodynamic datum $z_{\mathrm{datum}}(\mathbf{x}) = z_{\mathrm{topo}}(\mathbf{x}) + d(\mathbf{x})$, where $z_{\mathrm{topo}}$ is surface elevation and $d$ is zero-plane displacement (Harman & Finnigan, 2007). Following Klemp (2011, Equations 4–7), physical interface heights $z_{w, k}(\mathbf{x})$ are defined as:
$$
z_{w, k}(\mathbf{x}) = \zeta_{w, k} + A(\zeta_{w, k})\, h_{s, k}(\mathbf{x}), \qquad A(\zeta) = \cos^{2p}\!\left(\frac{\pi \zeta}{2 z_{\mathrm{top}}}\right),
$$
where $h_{s, 0}(\mathbf{x}) = z_{\mathrm{datum}}(\mathbf{x})$, higher coordinate surfaces $h_{s, k}$ undergo progressive horizontal Laplacian smoothing aloft, and minimum fractional vertical spacing $\Delta z_k \ge \gamma_{\min} \Delta \zeta_k$ ($\gamma_{\min} = 0.20$) is strictly enforced. Because $\Delta z_0 = 60\ \mathrm{m}$ yields first-layer relative height $\tilde{z}_0 = z_{c,0} - z_{\mathrm{datum}} = 30\ \mathrm{m}$, the geometric condition $\tilde{z}_0 > (h_c - d) > z_{0m}$ holds identically across all terrestrial biomes.

---

## 3. Moist Compressible Thermodynamics and Mixed-Phase Saturation Adjustment (`src/aoba/thermodynamics.py`)

### 3.1 Equation of State and Vapor-Buoyancy Density

Let $q_v$, $q_c$, and $q_i$ denote mixing ratios (per unit mass of dry air $\rho_d$) of water vapor, cloud liquid water, and cloud ice, with total non-precipitating water $q_n = q_v + q_c + q_i$ and total precipitating water $q_p = q_r + q_s$. Following Skamarock et al. (2012, Equations 5–6) and Seidel & Yang (2024), modified moist potential temperature $\theta_m$, full compressible air pressure $p$, Exner function $\Pi$, temperature $T$, and total moist density $\rho$ are related by:
$$
\begin{aligned}
\theta_m &= \theta \left(1 + \frac{R_v}{R_d} q_v\right), \\
p &= p_0 \left(\frac{R_d \rho_d \theta_m}{p_0}\right)^{c_{pd} / c_{vd}}, \\
\Pi &= \left(\frac{p}{p_0}\right)^{R_d / c_{pd}}, \qquad T = \frac{\Pi \theta_m}{1 + (R_v / R_d) q_v}, \\
\rho &= \rho_d \left(1 + q_n + q_p\right).
\end{aligned}
$$

### 3.2 Diagnostic Mixed-Phase Partitioning and Newton–Raphson Saturation Adjustment

Following Khairoutdinov & Randall (2003, Equations A8–A14), cloud condensate $q_{\mathrm{cond}} = q_c + q_i$ is partitioned diagnostically between liquid and ice via a linear temperature ramp $\omega_n(T)$ between $T_{00n} = 253.15\ \mathrm{K}$ and $T_{0n} = 273.16\ \mathrm{K}$:
$$
\omega_n(T) = \max\!\left[0,\ \min\!\left(1,\ \frac{T - T_{00n}}{T_{0n} - T_{00n}}\right)\right], \qquad q_c = \omega_n(T) q_{\mathrm{cond}}, \qquad q_i = \left[1 - \omega_n(T)\right] q_{\mathrm{cond}}.
$$
Saturation vapor pressures over planar liquid water $e_{sw}(T)$ (Bolton, 1980, Equation 10) and hexagonal ice $e_{si}(T)$ (Murray, 1967) define the blended saturation mixing ratio:
$$
q_{\mathrm{sat}}(T, p) = \omega_n(T) \frac{\epsilon e_{sw}(T)}{p - e_{sw}(T)} + \left[1 - \omega_n(T)\right] \frac{\epsilon e_{si}(T)}{p - e_{si}(T)}.
$$
Given liquid–ice equivalent temperature $T_{li} = T - \left(L_{v0} q_c + L_{s0} q_i\right) / c_{pd}$ and total non-precipitating mixing ratio $q_n$ at pressure $p$, supersaturated parcels ($q_n > q_{\mathrm{sat}}(T_{li}, p)$) are adjusted to exact saturation via four damped Newton–Raphson iterations on the moist-enthalpy conservation residual:
$$
f(T) = T - T_{li} - \frac{L_{\mathrm{eff}}(T)}{c_{pd}} \left[q_n - q_{\mathrm{sat}}(T, p)\right] = 0, \qquad L_{\mathrm{eff}}(T) = \omega_n(T) L_{v0} + \left[1 - \omega_n(T)\right] L_{s0}.
$$

---

## 4. Four-Band Lookup-Table-Free Two-Stream Radiative Transfer (`src/aoba/radiation.py`, `src/aoba/kernels.py`)

### 4.1 Unified Multi-Layer Two-Stream Tridiagonal Solver

For each spectral band and atmospheric column divided into $N = N_z$ homogeneous layers indexed top-down ($n = 1$ at TOA, $n = N$ adjacent to the surface) with layer optical thickness $\tau_n$, single-scattering albedo $\omega_{0n}$, and asymmetry factor $g_n$, upward and downward diffuse fluxes $F_n^\pm(\tau)$ ($\tau \in [0, \tau_n]$) satisfy the general two-stream system (Toon et al., 1989, Equations 11–12):
$$
\begin{aligned}
\frac{\partial F_n^+}{\partial \tau} &= \gamma_{1n} F_n^+ - \gamma_{2n} F_n^- - S_n^+(\tau), \\
\frac{\partial F_n^-}{\partial \tau} &= \gamma_{2n} F_n^+ - \gamma_{1n} F_n^- + S_n^-(\tau).
\end{aligned}
$$
Following Table 1 and Table 2 of Toon et al. (1989):
- **Shortwave bands (SW1, SW2)** apply Joseph, Wiscombe, & Weinman (1976) $\delta$-scaling ($f = g_n^2$, $\tau_n' = (1 - \omega_{0n} f)\tau_n$, $\omega_{0n}' = \frac{(1 - f)\omega_{0n}}{1 - \omega_{0n} f}$, $g_n' = \frac{g_n}{1 + g_n}$) combined with the **Quadrature** closure ($\mu_1 = 1/\sqrt{3}$):
  $$
  \gamma_1 = \frac{\sqrt{3}}{2}\left[2 - \omega_0'(1 + g')\right], \qquad \gamma_2 = \frac{\sqrt{3}}{2}\omega_0'(1 - g'), \qquad \gamma_3 = \frac{1}{2}\left(1 - \sqrt{3} g' \mu_0\right), \qquad \gamma_4 = 1 - \gamma_3.
  $$
- **Longwave bands (LW1, LW2)** apply the **Hemispheric Mean** closure ($\mu_1 = 1/2$, $\omega_0 = 0 \implies \gamma_1 = 2,\ \gamma_2 = 0$), which preserves exact blackbody unit emissivity as $\tau \to \infty$, with linear-in-$\tau$ Planck band flux $F_{B,n}(\tau) = F_{B,0n} + F_{B,1n}\tau$ ($F_{B,1n} = [F_B(T_{\mathrm{bot}, n}) - F_B(T_{\mathrm{top}, n})]/\tau_n$) yielding particular solutions (Toon et al., 1989, Equations 26–27):
  $$
  C_n^\pm(\tau) = F_{B,0n} + F_{B,1n}\left(\tau \pm \frac{1}{\gamma_{1n} + \gamma_{2n}}\right).
  $$

Defining $\lambda_n = \sqrt{\gamma_{1n}^2 - \gamma_{2n}^2}$, $\Gamma_n = \gamma_{2n} / (\gamma_{1n} + \lambda_n)$, and $e_{1n}, \dots, e_{4n}$ via Equations (44) of Toon et al. (1989), enforcing TOA, internal continuity, and surface albedo/emission boundary conditions reduces the $N$-layer problem to a $2N \times 2N$ tridiagonal linear system:
$$
A_l Y_{l-1} + B_l Y_l + D_l Y_{l+1} = E_l \qquad (l = 1, \dots, 2N),
$$
which AOBA solves simultaneously across all atmospheric columns via the hardware-fused batched Thomas kernel (`solve_tridiagonal_batched` in `src/aoba/kernels.py`).

### 4.2 Band Partitioning and Gaseous / Cloud Optical Depths

1. **SW1 (UV + Visible / PAR, $0.25\text{--}0.69\ \mu\mathrm{m}$, solar fraction $w_{\mathrm{SW1}} = 0.459760$)**:
   - Background ozone volume mixing ratio follows Wing et al. (2018, Equation 1):
     $$
     \mathrm{O}_3(p) = g_1 \, p_{\mathrm{hPa}}^{g_2} \exp\!\left(-\frac{p_{\mathrm{hPa}}}{g_3}\right) \times 10^{-6},
     $$
     with $g_1 = 3.6478$, $g_2 = 0.83209$, and $g_3 = 11.3515\ \mathrm{hPa}$. Cumulative slant ozone path $x = u_{\mathrm{O3}} M(\mu_0)$ with Rodgers magnification $M(\mu_0) = 35 / \sqrt{1224\mu_0^2 + 1}$ drives analytical UV + Chappuis absorption $A_{\mathrm{oz}}(x)$ (Lacis & Hansen, 1974, Equations 8–12), supplemented by Rayleigh molecular scattering ($\Delta \tau_{\mathrm{ray}} = 0.095 \Delta p / p_0$).
   - Liquid and ice cloud single-scattering properties $(\tau, \omega_0, g)$ follow Slingo (1989, Table 1 Band 1) and Ebert & Curry (1992, Table 2 Band 1). Downward surface SW1 flux yields photosynthetic photon flux density $\mathrm{PPFD} = 4.57 \times 0.88 \, F_{\mathrm{SW1}}^\downarrow(z_{\mathrm{datum}})$ ($\mathrm{\mu mol\ m^{-2}\ s^{-1}}$).
2. **SW2 (Near-Infrared, $0.69\text{--}4.00\ \mu\mathrm{m}$, solar fraction $w_{\mathrm{SW2}} = 0.540240$)**:
   - Scaled precipitable water path $\Delta y = 0.1 \, q_v \Delta m_d (p/p_0)\sqrt{T_0/T}$ ($\mathrm{cm}$) drives Lacis & Hansen (1974, Equation 21) absorptance:
     $$
     A_{\mathrm{wv}}(y / \mu_0) = \frac{2.9 (y / \mu_0)}{\left[1 + 141.5 (y / \mu_0)\right]^{0.635} + 5.925 (y / \mu_0)},
     $$
     combined with solar-weighted NIR liquid/ice cloud scattering and absorption (Slingo, 1989, Equation 25 & Table 1 Bands 2–4; Ebert & Curry, 1992, Table 2 Bands 2–5).
3. **LW1 ($\mathrm{H}_2\mathrm{O} + \mathrm{CO}_2$ Rotational–Vibrational Band, Planck fraction $1 - \beta = 0.680$)**:
   - Combines pressure-scaled $\mathrm{H}_2\mathrm{O}$ line absorption and logarithmic $15\ \mu\mathrm{m}$ $\mathrm{CO}_2$ optical depth (Weaver & Ramanathan, 1995; Koll & Cronin, 2018; Sherwood et al., 2020) with liquid and ice cloud absorption (Stephens, 1978; Ebert & Curry, 1992, Table 3):
     $$
     \Delta \tau_{\mathrm{LW1}} = 0.30 \left(\frac{p}{p_0}\right)^{0.55} q_v \Delta m_d + \max\!\left[0.02,\ 0.85 + 0.22 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} + 0.020 \frac{\Delta p}{p_0} + \Delta \tau_{\mathrm{cld,LW1}}.
     $$
4. **LW2 ($8.0\text{--}12.5\ \mu\mathrm{m}$ Atmospheric Window, Planck fraction $\beta = 0.320$)**:
   - Combines temperature-dependent $\mathrm{H}_2\mathrm{O}$ self-continuum absorption (Roberts et al., 1976; Koll & Cronin, 2018), weak foreign continuum, $\mathrm{CO}_2$ window-edge overlap, and cloud window absorption (Ebert & Curry, 1992, Table 3 Band 2):
     $$
     \Delta \tau_{\mathrm{LW2}} = \left[0.45 \frac{e_v}{p_0} \exp\!\left(\frac{1800}{T} - \frac{1800}{296}\right) + 0.0025 \frac{p}{p_0}\right] q_v \Delta m_d + \max\!\left[0,\ 0.015 + 0.024 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} + \Delta \tau_{\mathrm{cld,LW2}}.
     $$

### 4.3 Energy-Conserving Radiative Heating Rate

Defining net upward flux at half-level interface $k + \frac{1}{2}$ as $F_{\mathrm{net}, k+1/2} = (F_{\mathrm{SW}, k+1/2}^\uparrow - F_{\mathrm{SW}, k+1/2}^\downarrow) + (F_{\mathrm{LW}, k+1/2}^\uparrow - F_{\mathrm{LW}, k+1/2}^\downarrow)$, the radiative temperature tendency of layer $k$ is:
$$
\left(\frac{\partial T}{\partial t}\right)_{\mathrm{rad}, k} = -\frac{F_{\mathrm{net}, k+1/2} - F_{\mathrm{net}, k-1/2}}{(c_{pd} + c_{pv} q_{v,k}) \rho_{d,k} \Delta z_k},
$$
which guarantees exact column-wise radiative energy conservation between TOA net irradiance and surface net radiative flux.


---

## 5. Mixed-Phase Single-Moment Cloud Microphysics and Sedimentation (`src/aoba/microphysics.py`)

### 5.1 Prognostic Partitioning of Precipitating Hydrometeors

Following the System for Atmospheric Modeling (SAM; Khairoutdinov & Randall, 2003, Appendices A–B), AOBA prognoses two moist tracer variables: total non-precipitating water $q_n = q_v + q_c + q_i$ and total precipitating water $q_p = q_r + q_s + q_g$ (rain, snow, and graupel). While $(q_c, q_i)$ are partitioned via $\omega_n(T)$ in Equation (3.2), precipitating condensate $q_p$ is partitioned diagnostically into rain $q_r$, snow $q_s$, and graupel $q_g$ using linear temperature ramps $\omega_p(T)$ over $[268.15, 273.16]\ \mathrm{K}$ and $\omega_g(T)$ over $[223.15, 273.16]\ \mathrm{K}$ (Khairoutdinov & Randall, 2003, Equations A15–A17):
$$
\begin{aligned}
\omega_p(T) &= \mathrm{clip}\!\left(\frac{T - 268.15}{273.16 - 268.15},\ 0,\ 1\right), \qquad \omega_g(T) = \mathrm{clip}\!\left(\frac{T - 223.15}{273.16 - 223.15},\ 0,\ 1\right), \\
q_r &= \omega_p q_p, \qquad q_s = (1 - \omega_p)(1 - \omega_g) q_p, \qquad q_g = (1 - \omega_p)\omega_g q_p.
\end{aligned}
$$

### 5.2 Conversion, Accretion, Evaporation/Sublimation, and Sub-Cycled Sedimentation

Marshall–Palmer exponential size distributions $N_x(D) = N_{0x} \exp(-\lambda_x D)$ with intercept parameters $(N_{0r}, N_{0s}, N_{0g}) = (8 \times 10^6,\ 3 \times 10^6,\ 4 \times 10^6)\ \mathrm{m^{-4}}$ and power-law fall speeds $V_x(D) = a_x D^{b_x} (\rho_0 / \rho_d)^{1/2}$ govern microphysical transfer rates (Kessler, 1969; Lin, Farley, & Orville, 1983; Khairoutdinov & Randall, 2003, Equations A18–A33):
1. **Autoconversion** of cloud liquid ($q_c > q_{c0} = 10^{-3}\ \mathrm{kg\ kg^{-1}}$) and temperature-activated ice aggregation ($q_i > q_{i0} = 10^{-4}\ \mathrm{kg\ kg^{-1}}$):
   $$
   \left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{auto}} = \alpha_{\mathrm{liq}} \max(q_c - q_{c0},\ 0) + \beta_{\mathrm{ice}} \exp\!\left[0.025(T - 273.16)\right] \max(q_i - q_{i0},\ 0),
   $$
   with $\alpha_{\mathrm{liq}} = 10^{-3}\ \mathrm{s^{-1}}$ and $\beta_{\mathrm{ice}} = 10^{-3}\ \mathrm{s^{-1}}$.
2. **Continuous accretion** of $(q_c, q_i)$ by falling rain, snow, and graupel via analytic Gamma-integral collection kernels $A_{xy} q_y \left(\rho_d q_x\right)^{(3 + b_x)/4}$ ($x \in \{r, s, g\},\ y \in \{c, i\}$; Khairoutdinov & Randall, 2003, Equations A21–A27).
3. **Sub-saturated evaporation and sublimation** ($q_v < q_{\mathrm{sat}}$ and $q_c + q_i = 0$) including ventilation factors $(a_{vx} \lambda_x^{-2} + b_{vx} \mathrm{Sc}^{1/3} \sqrt{a_x / \nu} \Gamma((5 + b_x)/2) \lambda_x^{-(5 + b_x)/2})$ and psychrometric thermal-diffusion impedance (Khairoutdinov & Randall, 2003, Equations A29–A32).
4. **Mass-weighted terminal velocity and monotone upwind sedimentation**: Bulk precipitation fall speed is evaluated as:
   $$
   V_p = \omega_p V_r + (1 - \omega_p)\left[(1 - \omega_g) V_s + \omega_g V_g\right], \qquad V_x = \frac{a_x \Gamma(4 + b_x)}{6 \lambda_x^{b_x}} \left(\frac{\rho_0}{\rho_d}\right)^{1/2}.
   $$
   Downward precipitation mass flux $P_z = \rho_d q_p V_p$ is integrated via CFL-subcycled ($\max(V_p \Delta t_{\mathrm{sed}} / \Delta z_k) \le 0.8$) 1st-order donor-cell upwind updates, conserving total water mass and moist static energy across phase changes:
   $$
   \Delta \theta_m = \frac{1 + (R_v / R_d) q_v}{\Pi c_{pd}} \left[\Delta(L_{\mathrm{eff},n} q_{\mathrm{cond}}) + \Delta(L_{\mathrm{eff},p} q_p)\right].
   $$

---

## 6. Surface Roughness Sublayer (RSL), MYNN Level 2.5 PBL, and 3D SGS Turbulence (`src/aoba/turbulence.py`)

### 6.1 Canopy and Ocean Aerodynamic Roughness

1. **Vegetated terrestrial surfaces (Raupach, 1994, Equations 8 & 10–11)**: Given prognostic canopy height $h_c$ and exposed leaf area index $L_c$ (frontal area index $\Lambda = L_c / 2$), zero-plane displacement $d$ and momentum roughness length $z_{0m}$ are determined without land-cover lookup tables:
   $$
   \begin{aligned}
   \frac{d}{h_c} &= 1 - \frac{1 - \exp\!\left(-\sqrt{c_{d1} L_c}\right)}{\sqrt{c_{d1} L_c}}, \\
   \frac{z_{0m}}{h_c} &= \left(1 - \frac{d}{h_c}\right) \exp\!\left(-\kappa \gamma_u + \psi_{h,\mathrm{cw}}\right), \qquad \gamma_u = \min\!\left[\left(C_S + C_R \frac{L_c}{2}\right)^{-1/2},\ \gamma_{\max}\right],
   \end{aligned}
   $$
   with $c_{d1} = 7.5$, $C_S = 0.003$, $C_R = 0.30$, $\psi_{h,\mathrm{cw}} = 0.193$, $\gamma_{\max} = 3.33$, and scalar Stanton relation $\ln(z_{0m}/z_{0h}) = k B^{-1} \approx 2.0$ (smoothly blending to bare-soil $z_{0m,\mathrm{soil}} = 0.01\ \mathrm{m}$ and snow $z_{0m,\mathrm{snow}} = 0.002\ \mathrm{m}$ as $L_c \to 0$).
2. **Open ocean (Miller, Beljaars, & Palmer, 1992, Equations 3–5; Nishizawa et al., 2025, Equations 8.360–8.362)**: Combining smooth viscous sublayer resistance with Charnock capillary–gravity wave roughness ($\alpha_{\mathrm{Ch}} = 0.018$, $\nu_a = 1.5 \times 10^{-5}\ \mathrm{m^2\ s^{-1}}$):
   $$
   z_{0m} = 0.11 \frac{\nu_a}{u_*} + \frac{\alpha_{\mathrm{Ch}} u_*^2}{g}, \qquad z_{0h} = 0.40 \frac{\nu_a}{u_*} + 1.4 \times 10^{-5}, \qquad z_{0q} = 0.62 \frac{\nu_a}{u_*} + 1.3 \times 10^{-4}.
   $$

### 6.2 Monin–Obukhov Similarity with Analytical Roughness Sublayer (RSL) Enhancement

Over tall forests where $\tilde{z}_1 = z_{c,0} - (z_{\mathrm{topo}} + d)$ lies inside the roughness sublayer ($z_* = \max(2 h_c - d,\ 0)$; Harman & Finnigan, 2007), standard Monin–Obukhov similarity underestimates turbulent exchange. AOBA incorporates the closed-form exponential-integral RSL correction $\hat{\psi}_{M,H}(\zeta, \tilde{z}_1 / z_*)$ of De Ridder (2010, Equations 6–7 & 22; Bonan et al., 2018; Lee et al., 2020):
$$
\begin{aligned}
F_M &= \ln\!\left(\frac{\tilde{z}_1}{z_{0m}}\right) - \psi_M(\zeta) + \psi_M\!\left(\zeta \frac{z_{0m}}{\tilde{z}_1}\right) + \hat{\psi}_M\!\left(\zeta, \frac{\tilde{z}_1}{z_*}\right), \\
\hat{\psi}_\alpha(\zeta, \eta) &= \phi_\alpha\!\left(\zeta \eta^{-1}(1 + \nu_{\mathrm{RSL}}/\mu_\alpha)\right) \left(1 - \frac{1}{c_{2\alpha}}\right) \exp(-\mu_\alpha \eta) \left[\ln(\mu_\alpha \eta) + E_1(\mu_\alpha \eta)\right],
\end{aligned}
$$
solved iteratively for stability parameter $\zeta = \tilde{z}_1 / L_{\mathrm{MO}}$ from bulk Richardson number $\mathrm{Ri}_B$ using Businger–Dyer–Paulson ($\zeta < 0$) and Beljaars & Holtslag (1991; $\zeta \ge 0$) functions, yielding drag and scalar transfer coefficients $C_M = \kappa^2 / F_M^2$ and $C_{H,Q} = \kappa^2 / (F_M F_{H,Q})$.

### 6.3 Implicit MYNN Level 2.5 Boundary-Layer Closure

Vertical turbulent fluxes across all atmospheric columns follow the Mellor–Yamada–Nakanishi–Niino Level 2.5 closure (Nakanishi & Niino, 2009; Nishizawa et al., 2025, Section 8.2.1, Equations 8.96–8.176) prognostic in doubled turbulent kinetic energy $q^2 = 2e$:
$$
\frac{\partial (\rho_d q^2)}{\partial t} = \frac{\partial}{\partial z}\!\left(\rho_d K_q \frac{\partial q^2}{\partial z}\right) + 2 \rho_d \left(P_s + P_b - \frac{q^3}{B_1 L}\right),
$$
where $K_M = L q S_M$, $K_H = L q S_H$, $K_q = 3 L q S_M$, and the master length scale $L$ combines surface, turbulent-depth, and buoyancy length scales:
$$
\frac{1}{L} = \frac{1}{L_S} + \frac{1}{L_T} + \frac{1}{L_B}, \qquad L_T = 0.23 \frac{\int_0^\infty q \tilde{z} \, dz}{\int_0^\infty q \, dz}.
$$
Gaussian subgrid liquid-ice saturation fluctuations ($\sigma_s$, partial cloudiness $R$, subgrid condensate $Q_l$) determine moist buoyancy coefficients $(\beta_\theta, \beta_q)$ in $P_b$ and $N^2$, while Level 2.5 stability functions $(S_M, S_H)$ include the disequilibrium factor $\alpha_c = \min(q / q_{\mathrm{L2}},\ 1)$. Both the TKE equation and vertical scalar/momentum diffusion are integrated unconditionally stably via `solve_tridiagonal_batched`.

### 6.4 Anisotropic 3D Smagorinsky–Lilly SGS Closure for the CRM

Within the 3D CRM, horizontal subgrid-scale mixing follows the Richardson-dependent Smagorinsky–Lilly model (Lilly, 1962; Nishizawa et al., 2025, Section 8.1.2, Equations 8.25–8.40) corrected for pancake grid aspect ratio $a = \max(\Delta x, \Delta y) / \Delta z_k$ via Scotti, Meneveau, & Lilly (1993):
$$
\nu_{\mathrm{SGS}} = (C_s f(a) \Delta)^2 |\mathbf{S}| \sqrt{\max(0,\ 1 - \mathrm{Ri}/\mathrm{Ri}_c)}, \qquad \Delta = (\Delta x \Delta y \Delta z_k)^{1/3},
$$
with $C_s = 0.18$, $\mathrm{Ri}_c = 0.25$, and turbulent Prandtl number $\mathrm{Pr}_{\mathrm{SGS}}^{-1} = 3$ under unstable stratification.

---

## 7. Two-Leaf Non-Rectangular-Hyperbola EEO Biosphere (`src/aoba/biosphere.py`)

### 7.1 Least-Cost Optimal Stomatal Ratio and Non-Rectangular-Hyperbola Coordination

AOBA predicts global photosynthetic biochemistry, stomatal conductance, C₃/C₄ dominance, leaf area index, root-zone storage capacity, and canopy height **entirely without prescribed PFT maps**.
1. **C₃ Optimal $c_i / c_a$ and Non-Rectangular-Hyperbola Capacities (Prentice et al., 2014; Smith et al., 2019, Equations 1–20)**:
   Minimizing combined maintenance costs of carboxylation and water transport ($\beta_{\mathrm{C3}} = 146$) yields optimal intercellular-to-ambient $\mathrm{CO}_2$ ratio $\chi_{\mathrm{C3}}$:
   $$
   \chi_{\mathrm{C3}} = \frac{\Gamma^*}{c_a} + \left(1 - \frac{\Gamma^*}{c_a}\right) \frac{\xi_{\mathrm{C3}}}{\xi_{\mathrm{C3}} + \sqrt{D}}, \qquad \xi_{\mathrm{C3}} = \sqrt{\frac{\beta_{\mathrm{C3}}(K_M + \Gamma^*)}{1.6 \eta^*(T_g)}}.
   $$
   Electron transport $J$ follows the Farquhar & Wong (1984) non-rectangular hyperbola with curvature $\theta = 0.85$:
   $$
   \theta J^2 - (\varphi_{\mathrm{PSII}} I_c + J_{\max}) J + \varphi_{\mathrm{PSII}} I_c J_{\max} = 0.
   $$
   Enforcing marginal electron-transport optimality $\partial A_j / \partial J_{\max} = c$ ($c = 0.05336251$) and biochemical coordination $A_c = A_j$ yields the exact signed continuous branch for $\omega = J_{\max,\mathrm{opt}} / (\varphi_{\mathrm{PSII}} I_c)$ across all $m > 4c$ ($m = (c_i - \Gamma^*)/(c_i + 2\Gamma^*)$):
   $$
   \begin{aligned}
   \omega &= -(1 - 2\theta) + \mathrm{sgn}\!\left(1 - \frac{8\theta c}{m}\right) \sqrt{(1 - \theta)\left[\frac{1}{(4c/m)(1 - \theta \cdot 4c/m)} - 4\theta\right]}, \\
   \omega^* &= 1 + \omega - \sqrt{(1 + \omega)^2 - 4\theta \omega}, \\
   J_{\max,\mathrm{opt}} &= \varphi_{\mathrm{PSII}}(T_g)\, \tilde{I}_c \, \omega, \qquad V_{c\max,\mathrm{opt}} = \varphi_{\mathrm{PSII}}(T_g)\, \tilde{I}_c \, \frac{m}{m_c} \frac{\omega^*}{8\theta},
   \end{aligned}
   $$
   where $m_c = (c_i - \Gamma^*)/(c_i + K_M)$, vanishing smoothly for $m \le 4c$. Reference $25\ ^\circ\mathrm{C}$ capacities $(V_{c\max 25}, J_{\max 25})$ are recovered via entropy-acclimated peaked Arrhenius functions (Kattge & Knorr, 2007; Smith et al., 2019, Equation 20).
2. **C₄ Three-Enzyme Coordination (Scott & Smith, 2022, Equations 1–25; von Caemmerer & Furbank, 1999)**:
   Mesophyll drawdown $\chi_{m,\mathrm{C4}} = \xi_{\mathrm{C4}} / (\xi_{\mathrm{C4}} + \sqrt{D})$ with $\xi_{\mathrm{C4}} = \sqrt{\beta_{\mathrm{C4}} K_p / (1.6 \eta^*)}$ ($\beta_{\mathrm{C4}} = 166$), bundle-sheath leakiness $\phi_L = 0.20$, bundle-sheath conductance $g_{bs} = 0.003 L_c\ \mathrm{mol\ m^{-2}\ s^{-1}}$, and 5-electron ATP stoichiometry ($k_e = 5\ \mathrm{mol(e^-)\ mol(CO_2)^{-1}}$) determine $(J_{\max,\mathrm{C4}}, V_{p\max,\mathrm{C4}}, V_{c\max,\mathrm{C4}})$ via two fixed-point iterations on bundle-sheath partial pressure $C_{bs} = C_m + (L / g_{bs}) p_{\mathrm{sfc}}$.
3. **Universal EEO Leaf Dark Respiration (Heskel et al., 2016; Ren et al., 2024, 2025, Equation 13)**:
   Nocturnal leaf respiration scales directly with optimal Rubisco capacity as $R_{\mathrm{leaf}}(T_{\mathrm{night}}) = b_{\mathrm{resp}} V_{c\max,\mathrm{opt}}$ ($b_{\mathrm{resp}} = 0.018$), mapped across sub-daily temperatures via macromolecular rate theory:
   $$
   f_R(T) = \exp\!\left[0.1012(T_C - 25) - 0.0005(T_C^2 - 25^2)\right].
   $$

### 7.2 Midday Incident Shortwave EMA and Closed-Form Two-Leaf Enzyme Allocation

To capture nonlinear canopy light saturation without prescribing an empirical nitrogen extinction parameter $k_n$, AOBA executes a strict two-stage acclimation and sub-daily allocation scheme (de Pury & Farquhar, 1997; Hashimoto, 2026):
1. **Local solar noon downward-shortwave EMA**: At each terrestrial column's local solar noon crossing ($H = 0$), incident downward direct-beam and diffuse visible shortwave fluxes $(\mathrm{VIS}_b, \mathrm{VIS}_d)$ and solar zenith cosine $\mu_0$ are smoothed via an exponential moving average ($\alpha_{\mathrm{EMA}} = 0.067$, $\sim 15$-day memory) into $(\widetilde{\mathrm{VIS}}_b, \widetilde{\mathrm{VIS}}_d, \tilde{\mu}_0)$.
2. **Spherical-canopy two-stream light absorption**: Evaluating the de Pury & Farquhar (1997) spherical-leaf radiative transfer on $(\widetilde{\mathrm{VIS}}_b, \widetilde{\mathrm{VIS}}_d, \tilde{\mu}_0)$ at the **current** snow-exposed leaf area index $L_c$ yields whole-canopy acclimation irradiance $\tilde{I}_c = \tilde{I}_{cb} + \tilde{I}_{cd}$ and vertical profile $\tilde{I}_l(L) = \tilde{I}_{lb}(L) + \tilde{I}_{ld}(L)$ with attenuation rates $\tilde{k}_b' = 0.5\sqrt{1 - \sigma_P}/\tilde{\mu}_0$ and $k_d' = 0.719$.
3. **Exact closed-form sunlit/shaded partitioning**: Assuming within-canopy photosynthetic protein distribution $X_l(L) = X (\tilde{I}_l(L) / \tilde{I}_c)$ tracks the acclimated absorbed light profile, integrating analytically against the **instantaneous** sunlit probability $\exp(-k_{db}(t) L)$ ($k_{db}(t) = 0.5 / \mu_0(t)$) allocates every canopy capacity pool $X \in \{V_{c\max 25}, J_{\max 25}, V_{p\max 25}, R_{25}\}$ conservatively into sunlit ($X_{\mathrm{sun}}$) and shaded ($X_{\mathrm{sh}}$) fractions:
   $$
   \begin{aligned}
   X_{\mathrm{sun}} &= \frac{X}{\tilde{I}_c} \left[\tilde{I}_{b0}(1 - \tilde{\rho}_{cb})\tilde{k}_b' \frac{1 - \exp\!\left[-(\tilde{k}_b' + k_{db})L_c\right]}{\tilde{k}_b' + k_{db}} + \tilde{I}_{d0}(1 - \rho_{cd})k_d' \frac{1 - \exp\!\left[-(k_d' + k_{db})L_c\right]}{k_d' + k_{db}}\right], \\
   X_{\mathrm{sh}} &= \max(10^{-6},\ X - X_{\mathrm{sun}}).
   \end{aligned}
   $$
   Sub-daily gross assimilation, hyperbolic co-limitation, and optimal stomatal conductances $g_{sw,\mathrm{sun}}, g_{sw,\mathrm{sh}} = 1.6 A_n (R_{\mathrm{gas}} T_l / p_{\mathrm{sfc}}) / [c_a(1 - \chi)]$ ($\mathrm{m\ s^{-1}}$) are then evaluated separately on $(I_{\mathrm{sun}}, X_{\mathrm{sun}})$ and $(I_{\mathrm{sh}}, X_{\mathrm{sh}})$.

### 7.3 Two-Stream Canopy Albedo, Autonomous $F_4$, and Lambert-$W_0$ Prognostic Structure

1. **Map-free canopy–ground two-stream albedo**: Visible and near-infrared direct/diffuse surface albedos $(\alpha_{\mathrm{VIS,dir}}, \alpha_{\mathrm{VIS,dif}}, \alpha_{\mathrm{NIR,dir}}, \alpha_{\mathrm{NIR,dif}})$ are solved analytically from $(L_c, \mu_0)$ and underlying soil/snow reflectance via Sellers (1985) and Dai et al. (2004, Appendix A, Equations A1–A19).
2. **Autonomous C₄ fraction $F_4$** (Scott & Smith, 2022, Equation 26; Luo et al., 2024; Lavergne et al., 2026): Driven by midday net assimilation advantage $\Delta A = (A_{n,\mathrm{C4}} - A_{n,\mathrm{C3}})/(|A_{n,\mathrm{C3}}| + 1)$ and overstory light openness $f_{\mathrm{open}}(L_c) = \mathrm{clip}(1.15 - 0.18 L_c,\ 0.02,\ 1)$.
3. **Prognostic EEO LAI, root-zone storage $S_R$, and canopy height $h_c$**:
   - Annual peak greenness $\mathrm{fAPAR}_{\max}$ combines energy limitation $1 - z/(k A_{0,\mathrm{yr}})$ ($z = 12.227\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$, $k = 0.5$) and water limitation $f_0(\mathrm{AI}) P_{\mathrm{yr}} c_a (1 - \chi)/(1.6 D A_{0,\mathrm{yr}})$ via smooth log-sum-exp minimization ($K = 10$; Cai et al., 2025, Equation 12; Zhou et al., 2025, Equations 6 & 11).
   - Daily steady-state leaf area index $L_s$ is obtained analytically via the principal branch of the Lambert $W$ function (Zhou et al., 2025, Equations 15–20):
     $$
     L_s = \min\!\left\{\mu + \frac{1}{k} W_0\!\left[-k\mu \exp(-k\mu)\right],\ \mathrm{LAI}_{\max}\right\}, \qquad \mu = \frac{\sigma_{\mathrm{Zhou}} \, \mathrm{GSL} \, \mathrm{LAI}_{\max}}{A_{0,\mathrm{yr}} \, \mathrm{fAPAR}_{\max}} A_{0,\mathrm{day}},
     $$
     updated with $\alpha_{\mathrm{EMA}} = 0.067$, while drought-bridging root capacity $S_R$ scales with cumulative seasonal water deficit (Gao et al., 2014) and potential canopy height $h_c$ scales with vascular moisture–productivity surplus (Kempes et al., 2011).

---

## 8. Land Hydrology, Five-Layer Freeze–Thaw Permafrost Soil, Snow, and Sea Ice (`src/aoba/surface.py`)

### 8.1 Three-Component Evapotranspiration and Linearized Skin Energy Balance

Total terrestrial upward water-vapor flux combines wet-canopy interception evaporation ($f_{\mathrm{wet}} = (W_c / W_{c,\max})^{2/3}$; MATSIRO6 Chapter 7), dry-canopy sunlit + shaded stomatal transpiration ($g_{sw,\mathrm{sun}} + g_{sw,\mathrm{sh}}$), snow sublimation over fractional snow cover $A_{\mathrm{sn}}$, and bare-soil evaporation regulated by a dry-surface vapor-diffusion resistance $r_{\mathrm{soil}} = (L_{\mathrm{dry}} / D_v)(\tau_{\mathrm{tort}} / \theta_{\mathrm{air}})$ (Sakaguchi & Zeng, 2009, Equations 7–10). Surface skin temperature $\Delta T_{\mathrm{skin}}$ is solved implicitly at every step from the linearized radiative–turbulent–conductive energy balance, clamping $T_{\mathrm{skin}} \le T_0 = 273.15\ \mathrm{K}$ when snow is present so excess energy drives snowmelt directly.

### 8.2 Fractional Snow Cover, Snow-Burial Masking, Aging Albedo, and Glacier Calving

1. **Subgrid snow fraction** $A_{\mathrm{sn}}$ accounts for Earth2014 subgrid topographic standard deviation $\sigma_{\mathrm{topo}}$ and hysteretic melt depletion curves (Swenson & Lawrence, 2012, Equations 3–6; Nitta et al., 2014).
2. **Vegetation snow-burial masking** reduces exposed canopy height and LAI inside the snow-covered fraction via $h_{\mathrm{buried}} = \max(h_c - d_{\mathrm{snow}},\ 0)$ and $L_{\mathrm{buried}} = L_c (h_{\mathrm{buried}} / h_c)$ (MATSIRO6 Equations 2.1–2.3), reproducing bright snow-covered tundra alongside dark snow-shedding boreal taiga without PFT tables.
3. **Prognostic snow aging** evolves nondimensional grain age $\tau_{\mathrm{sn}}$ thermally and resets upon fresh snowfall (Yang et al., 1997; MATSIRO6 Section 8.8.1), while permanent snow accumulation exceeding $W_{\mathrm{sn},\max} = 1000\ \mathrm{kg\ m^{-2}}$ calves into solid glacial runoff $R_{\mathrm{ice}}$ routed to coastal ocean cells (MATSIRO6 Equation 8.44).

### 8.3 Five-Layer Implicit Soil Thermodynamics with Zero-Curtain Phase Change

Across five soil layers ($\Delta z_s = [0.05, 0.15, 0.40, 0.80, 1.60]\ \mathrm{m}$; $3.0\ \mathrm{m}$ total depth), internal energy relative to $T_0 = 273.15\ \mathrm{K}$ is defined as $U_l = C_l (T_l - T_0) - L_{f0} M_{i,l}$, where $C_l = C_{\mathrm{dry}} + c_w M_{w,l} + c_i M_{i,l}$ (Nishizawa et al., 2025, Section 8.7.1, Equations 8.390–8.418; MATSIRO6 Chapter 10). Exact $0\ ^\circ\mathrm{C}$ liquid–ice phase partitioning $M_{i,l}^{\mathrm{new}} = \mathrm{clip}(-U_l / L_{f0},\ 0,\ M_{w,l} + M_{i,l})$ is followed by implicit Fourier heat conduction solved via `solve_tridiagonal_batched` in Celsius anomalies $(T_l - T_0)$ to preserve full single-precision accuracy on GPU.

### 8.4 Solar-Zenith Ocean Albedo and Fractional-Lead Thermodynamic Sea Ice

Open-ocean direct-beam albedo increases strongly at low sun angles via Briegleb et al. (1986):
$$
\alpha_{\mathrm{ocn,dir}}(\mu_0) = \frac{0.026}{\mu_0^{1.7} + 0.065} + 0.15(\mu_0 - 0.1)(\mu_0 - 0.5)(\mu_0 - 1.0),
$$
with diffuse albedo $\alpha_{\mathrm{ocn,dif}} = 0.06$. Sea-ice fractional area follows $f_i = \min(\sqrt{M_i / M_{i,\mathrm{crit}}},\ 1)$ ($M_{i,\mathrm{crit}} = 300\ \mathrm{kg\ m^{-2}}$; Nishizawa et al., 2025, Equation 8.342), solving separate turbulent and radiative budgets over open-water leads $(1 - f_i)$ and snow-covered ice $f_i$ (Semtner, 1976).

---

## 9. Embedded 3D Nonhydrostatic RK3–HEVI CRM and Zalesak 3D FCT (`src/aoba/crm.py`)

### 9.1 Flux-Form Compressible Nonhydrostatic Equations and Implicit HEVI Solver

Each host cell embeds a doubly periodic $N_x \times N_y = 8 \times 8$ ($\Delta x = \Delta y = 4\ \mathrm{km}$) Arakawa C-grid / Charney–Phillips 3D CRM solving the flux-form compressible moist Euler equations in $(\rho_d, u, v, w, \Theta_m = \rho_d \theta_m)$ (Skamarock et al., 2012, Equations 7–11; Nishizawa et al., 2025, Chapter 3):
$$
\begin{aligned}
\frac{\partial \rho_d}{\partial t} + \nabla_h \cdot (\rho_d \mathbf{u}_h) + \frac{\partial F_z}{\partial z} &= 0, \qquad F_z = \rho_{d,w} w, \\
\frac{\partial \Theta_m}{\partial t} + \nabla_h \cdot (\Theta_m \mathbf{u}_h) + \frac{\partial (\theta_{m,w} F_z)}{\partial z} &= S_{\Theta_m}, \\
\rho_{\mathrm{tot},w} \frac{\partial w}{\partial t} + \rho_{\mathrm{tot},w} (\mathbf{u} \cdot \nabla w) &= -\frac{\partial p}{\partial z} - \rho_{\mathrm{tot},w} g - \rho_{\mathrm{tot},w} \gamma_R(z_w) w + D_w.
\end{aligned}
$$
Time integration employs the 3-stage Wicker & Skamarock (2002) Runge–Kutta scheme ($\Delta \tau \in \{\Delta t/3,\ \Delta t/2,\ \Delta t\}$) with off-centered ($\beta = 0.60$) Horizontally Explicit / Vertically Implicit (HEVI) coupling (Klemp, Skamarock, & Dudhia, 2007). Eliminating $(\delta \rho_d, \delta \Theta_m)$ in favor of $\delta F_z = F_z^{\mathrm{new}} - F_z^{\mathrm{old}}$ via $\partial p / \partial \Theta_m = \gamma p / \Theta_m$ yields a strictly diagonally dominant $(N_z - 1) \times (N_z - 1)$ tridiagonal Helmholtz system solved across all $N_{\mathrm{cells}} \times N_y \times N_x$ sub-columns via `solve_tridiagonal_batched`, conserving 3D dry mass and $\Theta_m$ to machine precision.

### 9.2 Multidimensional Positive-Definite Zalesak FCT Tracer Transport

Moist tracers $(q_n, q_p)$ are advected using the stage-consistent mass fluxes $(F_x, F_y, F_z)$ via 3D Flux-Corrected Transport (Zalesak, 1979, Equations 1–17; Nishizawa et al., 2025, Appendix A.2, Equations A.20–A.35), blending 1st-order monotone donor-cell upwind fluxes $\mathbf{F}^L$ with 4th-order centered fluxes $\mathbf{F}^H$ via neighborhood anti-diffusive limiters $C \in [0, 1]$ so $q_n, q_p \ge 0$ and $\sum \rho_d q \Delta V$ is conserved identically.

---

## 10. Global SCVT Voronoi C-Grid Core, Orographic Drag, Voronoi Ekman Ocean, and 1:1 Coupling

### 10.1 SCVT Voronoi Mesh, TRiSK Operators, and Earth2014 Aggregation (`src/aoba/scvt_mesh.py`, `scvt_dynamics.py`)

Recursive icosahedral subdivision ($N_{\mathrm{cells}} = 10 \cdot 4^g + 2$) and spherical Lloyd relaxation (Ringler, Ju, & Gunzburger, 2008) define spherical Voronoi cells of exact spherical excess area $A_i$ ($\sum_i A_i = 4\pi R_E^2$), geodesic cell distances $d_{c,e}$, Voronoi edge lengths $l_e$, and dual Delaunay triangle areas $A_v$. Horizontal dynamics follow the vector-invariant C-grid formulation of Thuburn et al. (2009) and Ringler et al. (2010, Equations 17–26):
$$
\frac{\partial u_e}{\partial t} = \sum_{e' \in \mathrm{EC}(e)} w_{e,e'} \frac{\eta_e + \eta_{e'}}{2} u_{e'} - \frac{K_{c_1} - K_{c_0}}{d_{c,e}} - \frac{1}{\rho_{\mathrm{tot},e}} \left[\left(\frac{\partial p}{\partial n}\right)_\zeta - \frac{\partial p}{\partial z}\left(\frac{\partial z}{\partial n}\right)_\zeta\right] + D_{u,e},
$$
where $\eta_v = \zeta_v + f_v$ is absolute vorticity and TRiSK weights $w_{e,e'}$ satisfy the exact skew-symmetry identity $A_e w_{e,e'} + A_{e'} w_{e',e} = 0$ ($A_e = \frac{1}{2} d_{c,e} l_e$), ensuring zero spurious Coriolis/rotational kinetic-energy generation, stabilized by divergence damping and $\nabla^4$ hyperdiffusion (Skamarock et al., 2012, Section 3d). Five-arc-minute `Earth2014` grids (`SUR2014`, `ICE2014`, `TBI2014`; Hirt & Rexer, 2015) are aggregated onto each Voronoi cell to yield cell-mean $z_{\mathrm{topo}}$, surface masks (`Ocean`, `Land`, `IceSheet`), and subgrid gradient covariance invariants $(\mu, \sigma, \gamma, \psi)$.

### 10.2 Lott & Miller (1997) Subgrid Orographic Drag (`src/aoba/orography.py`)

When inverse Froude number $H_n = \bar{N}_L (2\mu) / |\mathbf{U}_L|$ exceeds $H_{nc} = 0.50$, flow below $Z_b = 2\mu(1 - H_{nc}/H_n)$ experiences directional elliptic bluff-body wake drag, while residual mountain height $h_{\mathrm{eff}} = 2\mu - Z_b$ launches upward-propagating internal gravity waves attenuated via the Lindzen saturation stress $\tau_{\mathrm{sat}}(z) \propto \rho U_{\mathrm{proj}}^3 / (N H_{nc}^2)$ (Lott & Miller, 1997, Equations 7–18).

### 10.3 Zero-$Q$-Flux 1.5-Layer Voronoi Ekman Ocean (`src/aoba/ocean_ekman.py`)

Following Codron (2012, Equations 1–10), wind stress $\boldsymbol{\tau}_{\mathrm{ocn}}$ drives mixed-layer ($H_1 = 50\ \mathrm{m}$, temperature $T_1$) Ekman mass transport across interior ocean–ocean Voronoi edges ($r_E = (2\ \mathrm{days})^{-1}$, zero normal flux at all coastal edges):
$$
M_{E,x} = \frac{r_E \tau_{x,\mathrm{ocn}} + f \tau_{y,\mathrm{ocn}}}{f^2 + r_E^2}, \qquad M_{E,y} = \frac{r_E \tau_{y,\mathrm{ocn}} - f \tau_{x,\mathrm{ocn}}}{f^2 + r_E^2},
$$
balanced by an equal and opposite return flow $-\mathbf{M}_E$ in a thermocline layer ($H_2 = 200\ \mathrm{m}$, temperature $T_2$) and vertical Ekman upwelling $W_E = \nabla_{\mathrm{SCVT}} \cdot \mathbf{M}_E$, conserving global ocean enthalpy to machine precision without prescribed $Q$-flux maps while absorbing latent heat of fusion from routed coastal glacial runoff (MIROC6 Section 4.1.2).

### 10.4 1:1 Superparameterization Coupling and Emergent Köppen–Geiger Diagnostics (`src/aoba/coupled_model.py`, `src/aoba/koppen.py`)

Every SCVT Voronoi cell couples 1:1 to its embedded $8 \times 8 \times 32$ 3D CRM at interval $\Delta t_{\mathrm{host}}$ (Khairoutdinov et al., 2005; Yamazaki & Miura, 2024): host dynamical tendencies $(\partial_t \theta_m, \partial_t q_n)_{\mathrm{LS}}$ and cell-center winds force the CRM sub-columns alongside radiative heating and surface fluxes, and horizontal-mean CRM states and momentum tendencies feed back directly onto the host SCVT state. Accumulated monthly surface air temperature $T_m$ and precipitation $P_m$ classify all terrestrial cells into the exact 30-class Köppen–Geiger system (`Af`–`EF`; Beck et al., 2018, 2023, Table 1), enabling direct quantitative evaluation of emergent present-day biomes and elevated-$\mathrm{CO}_2$ climate-zone shifts.
