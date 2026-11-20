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
