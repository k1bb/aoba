# AOBA 詳細数式および離散化記述書

> **言語 (Language)**: [English (formulation.md)](formulation.md) | 日本語

本文書は、*Description of MIROC6 AGCM* (2021)、*Description of MATSIRO6* (2021)、および *Detailed formulation of SCALE-RM* (Nishizawa et al., 2025) の記述様式に準拠し、**AOBA** の連続系支配方程式・座標変換・離散化アルゴリズム・物理定数を網羅的に記述するモデル記述書である。すべての式は `src/aoba/` 内の実装コードと1対1に対応し、根拠となる原著文献を明示する。

---

## 1. 基礎物理定数および太陽軌道幾何

### 1.1 基礎物理定数（`src/aoba/constants.py`）

表 1.1 に AOBA 全体で共通使用する惑星・熱力学・放射・海洋定数を示す（Wing et al., 2018, Table 1; Khairoutdinov & Randall, 2003, Appendix A; MIROC6 AGCM Document Writing Team, 2021, Section 1.3.2; Semtner, 1976）。

| 記号 | 物理量 | 値 | 単位 | 根拠文献 |
| :--- | :--- | :--- | :--- | :--- |
| $R_E$ | 地球平均半径 | $6.371 \times 10^6$ | $\mathrm{m}$ | Wing et al. (2018) |
| $g$ | 標準重力加速度 | $9.79764$ | $\mathrm{m\ s^{-2}}$ | Wing et al. (2018) |
| $\Omega$ | 地球自転角速度 | $7.292115 \times 10^{-5}$ | $\mathrm{rad\ s^{-1}}$ | MIROC6 (2021) |
| $S_0$ | 現在気候の太陽定数 | $1361.0$ | $\mathrm{W\ m^{-2}}$ | Wing et al. (2018) |
| $R_d$ | 乾燥空気の気体定数 | $287.04$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pd}$ | 乾燥空気の定圧比熱 | $1004.64$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{vd}$ | 乾燥空気の定積比熱（$c_{pd} - R_d$） | $717.60$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Skamarock et al. (2012) |
| $R_v$ | 水蒸気の気体定数 | $461.50$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pv}$ | 水蒸気の定圧比熱 | $1846.0$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $\epsilon$ | 水蒸気・乾燥空気分子量比 $R_d / R_v$ | $0.62197$ | $-$ | Skamarock et al. (2012) |
| $p_0$ | 基準気圧 | $1.0 \times 10^5$ | $\mathrm{Pa}$ | Skamarock et al. (2012) |
| $L_{v0}$ | $0\ ^\circ\mathrm{C}$ における蒸発潜熱 | $2.501 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{f0}$ | $0\ ^\circ\mathrm{C}$ における融解潜熱 | $3.337 \times 10^5$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{s0}$ | $0\ ^\circ\mathrm{C}$ における昇華潜熱（$L_{v0} + L_{f0}$） | $2.8347 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $\sigma$ | Stefan–Boltzmann 定数 | $5.670374 \times 10^{-8}$ | $\mathrm{W\ m^{-2}\ K^{-4}}$ | CODATA |

### 1.2 軌道要素に基づく太陽入射と天頂角余弦（`src/aoba/radiation.py`）

Berger (1978) および Davis et al. (2017, Equations 2–8) に従い、経過秒数 $t$、通日 $N_d = (t / 86400) \bmod 365.2425$、軌道離心率 $e$、地軸傾斜角 $\varepsilon$、近日点黄経 $\tilde{\omega}$ から平均近点角 $\nu_m = \frac{2\pi}{365.2425}(N_d - 80.0) - \tilde{\omega}$ を定義すると、真近点角 $\nu$ と太陽黄経 $\lambda_s$ は次式で与えられる：
$$
\begin{aligned}
\nu &= \nu_m + \left(2e - \frac{1}{4}e^3\right)\sin\nu_m + \frac{5}{4}e^2 \sin(2\nu_m), \\
\lambda_s &= \nu + \tilde{\omega}.
\end{aligned}
$$
太陽赤緯 $\delta_s$、日地距離補正を含む瞬時大気上端（TOA）太陽放射照度 $S_{\mathrm{inst}}(t)$、および緯度 $\varphi$・経度 $\lambda$ における太陽天頂角余弦 $\mu_0(\varphi, \lambda, t)$ は次式となる：
$$
\begin{aligned}
\sin\delta_s &= \sin\varepsilon \sin\lambda_s, \\
S_{\mathrm{inst}}(t) &= S_0 \left(\frac{1 + e\cos\nu}{1 - e^2}\right)^2, \\
\mu_0(\varphi, \lambda, t) &= \max\!\left[0,\ \sin\varphi \sin\delta_s + \cos\varphi \cos\delta_s \cos H(t, \lambda)\right].
\end{aligned}
$$
ここで $H(t, \lambda) = 2\pi \left[\frac{t \bmod 86400}{86400} - \frac{1}{2}\right] + \lambda$ は地方時角である。

---

## 2. 鉛直 Charney–Phillips 格子と平滑化地形追従座標（`src/aoba/vertical_grid.py`）

### 2.1 鉛直ストレッチ格子と Charney–Phillips 変数配置

鉛直領域は実効空力基準面（$\zeta = 0$）から $z_{\mathrm{top}} = 24,000\ \mathrm{m}$ までの $N_z = 32$ 層とし、Charney–Phillips 鉛直スタガード配置（Gassmann, 2013, Section 3）を採用する：
- **ハーフレベル（層境界）**（$k + \frac{1}{2},\ k = 0, \dots, N_z$；配列添字 `0 .. nz`）：鉛直流 $w$、湿潤温位 $\theta_m$、物理境界高度 $z_w$、上下向き放射フラックス $F^\uparrow, F^\downarrow$。
- **フルレベル（層中心）**（$k \in \{0, \dots, N_z - 1\}$）：乾燥空気密度 $\rho_d$、気圧 $p$、気温 $T$、水平風速成分、および水物質混合比 $(q_v, q_c, q_i, q_n, q_p)$。

基準層厚 $\Delta \zeta_k = \zeta_{w, k+1} - \zeta_{w, k}$ は最下層 $\Delta \zeta_0 = 60.0\ \mathrm{m}$（高度 $2\ \mathrm{km}$ 以下に12層を集中させて大気境界層を解像し、自由対流圏に約15層を配置）から滑らかに伸長させ、上部 $17.5\text{--}24\ \mathrm{km}$ の下部成層圏緩衝層には上方伝播波の反射を防ぐ $\sin^2$ 型レイリー減衰スポンジ層を課す（Klemp et al., 2008; Skamarock et al., 2012）：
$$
\gamma_R(\zeta) = \frac{1}{\tau_{\mathrm{sp,min}}} \sin^2\!\left[\frac{\pi}{2}\,\mathrm{clip}\!\left(\frac{\zeta - z_{\mathrm{sp}}}{z_{\mathrm{top}} - z_{\mathrm{sp}}},\, 0,\, 1\right)\right].
$$

### 2.2 変位追従型の平滑化地形追従（STF）座標変換

高木林（$h_c \le 45\ \mathrm{m}$）における接地層相似則の特異点と急峻山岳上の気圧傾度力誤差を同時に防ぐため、鉛直座標下端を地表面標高 $z_{\mathrm{topo}}(\mathbf{x})$ にゼロ面変位 $d(\mathbf{x})$ を加えた実効空力基準面 $z_{\mathrm{datum}}(\mathbf{x}) = z_{\mathrm{topo}}(\mathbf{x}) + d(\mathbf{x})$ に定義する（Harman & Finnigan, 2007）。Klemp (2011, Equations 4–7) に従い、物理層境界高度 $z_{w, k}(\mathbf{x})$ を次式で定める：
$$
z_{w, k}(\mathbf{x}) = \zeta_{w, k} + A(\zeta_{w, k})\, h_{s, k}(\mathbf{x}), \qquad A(\zeta) = \cos^{2p}\!\left(\frac{\pi \zeta}{2 z_{\mathrm{top}}}\right).
$$
ここで $h_{s, 0}(\mathbf{x}) = z_{\mathrm{datum}}(\mathbf{x})$ であり、上空の座標面 $h_{s, k}$ には段階的な水平ラプラシアン平滑化を適用するとともに、最小許容層厚比 $\Delta z_k \ge \gamma_{\min} \Delta \zeta_k$（$\gamma_{\min} = 0.20$）を厳密に保証する。最下層厚 $\Delta z_0 = 60\ \mathrm{m}$ により第1層中心の相対高度は $\tilde{z}_0 = z_{c,0} - z_{\mathrm{datum}} = 30\ \mathrm{m}$ となり、すべてのバイオームで $\tilde{z}_0 > (h_c - d) > z_{0m}$ が恒等的に成立する。

---

## 3. 圧縮性湿潤熱力学と混合相飽和調節（`src/aoba/thermodynamics.py`）

### 3.1 状態方程式と水蒸気浮力・凝結水荷重密度

乾燥空気単位質量あたりの水蒸気・雲水・雲氷混合比を $q_v, q_c, q_i$ とし、非降水総水分 $q_n = q_v + q_c + q_i$ および降水総水分 $q_p = q_r + q_s$ を定義する。Skamarock et al. (2012, Equations 5–6) および Seidel & Yang (2024) に従い、湿潤修正温位 $\theta_m$、完全圧縮性気圧 $p$、Exner 関数 $\Pi$、気温 $T$、および全湿潤密度 $\rho$ の関係を次式で与える：
$$
\begin{aligned}
\theta_m &= \theta \left(1 + \frac{R_v}{R_d} q_v\right), \\
p &= p_0 \left(\frac{R_d \rho_d \theta_m}{p_0}\right)^{c_{pd} / c_{vd}}, \\
\Pi &= \left(\frac{p}{p_0}\right)^{R_d / c_{pd}}, \qquad T = \frac{\Pi \theta_m}{1 + (R_v / R_d) q_v}, \\
\rho &= \rho_d \left(1 + q_n + q_p\right).
\end{aligned}
$$

### 3.2 診断氷相分配と Newton–Raphson 飽和調節法

Khairoutdinov & Randall (2003, Equations A8–A14) に準拠し、雲凝結水 $q_{\mathrm{cond}} = q_c + q_i$ の液相・氷相比率は $T_{00n} = 253.15\ \mathrm{K}$ から $T_{0n} = 273.16\ \mathrm{K}$ の線形ランプ関数 $\omega_n(T)$ で診断配分する：
$$
\omega_n(T) = \max\!\left[0,\ \min\!\left(1,\ \frac{T - T_{00n}}{T_{0n} - T_{00n}}\right)\right], \qquad q_c = \omega_n(T) q_{\mathrm{cond}}, \qquad q_i = \left[1 - \omega_n(T)\right] q_{\mathrm{cond}}.
$$
液水面飽和蒸気圧 $e_{sw}(T)$（Bolton, 1980, Equation 10）と氷面飽和蒸気圧 $e_{si}(T)$（Murray, 1967）を $\omega_n(T)$ で内挿して混合相飽和混合比を定義する：
$$
q_{\mathrm{sat}}(T, p) = \omega_n(T) \frac{\epsilon e_{sw}(T)}{p - e_{sw}(T)} + \left[1 - \omega_n(T)\right] \frac{\epsilon e_{si}(T)}{p - e_{si}(T)}.
$$
気圧 $p$・非降水総水分 $q_n$・液氷等価温度 $T_{li} = T - \left(L_{v0} q_c + L_{s0} q_i\right) / c_{pd}$ が与えられたとき、過飽和気塊（$q_n > q_{\mathrm{sat}}(T_{li}, p)$）は以下の湿潤エンタルピー保存残差方程式に対する4回の Newton–Raphson 反復により機械精度で飽和調節される：
$$
f(T) = T - T_{li} - \frac{L_{\mathrm{eff}}(T)}{c_{pd}} \left[q_n - q_{\mathrm{sat}}(T, p)\right] = 0, \qquad L_{\mathrm{eff}}(T) = \omega_n(T) L_{v0} + \left[1 - \omega_n(T)\right] L_{s0}.
$$

---

## 4. ルックアップテーブル不要の4バンド2ストリーム放射伝達（`src/aoba/radiation.py`, `src/aoba/kernels.py`）

### 4.1 多層統一2ストリーム三重対角ソルバ

大気コラムを上端から下端へ向かう $N = N_z$ 個の均質層（$n = 1$ が TOA、$n = N$ が最下層；層光学的厚さ $\tau_n$、単一散乱アルベド $\omega_{0n}$、非対称因子 $g_n$）に分割すると、各層内の上下向き拡散フラックス $F_n^\pm(\tau)$（$\tau \in [0, \tau_n]$）は以下の一般化2ストリーム方程式に従う（Toon et al., 1989, Equations 11–12）：
$$
\begin{aligned}
\frac{\partial F_n^+}{\partial \tau} &= \gamma_{1n} F_n^+ - \gamma_{2n} F_n^- - S_n^+(\tau), \\
\frac{\partial F_n^-}{\partial \tau} &= \gamma_{2n} F_n^+ - \gamma_{1n} F_n^- + S_n^-(\tau).
\end{aligned}
$$
Toon et al. (1989) の Table 1・Table 2 に示される極限精度条件に従い、短波と長波で以下のクロージャを使い分ける：
- **短波バンド（SW1, SW2）**：Joseph, Wiscombe, & Weinman (1976) の前方散乱 $\delta$-スケーリング（$f = g_n^2$, $\tau_n' = (1 - \omega_{0n} f)\tau_n$, $\omega_{0n}' = \frac{(1 - f)\omega_{0n}}{1 - \omega_{0n} f}$, $g_n' = \frac{g_n}{1 + g_n}$）を施した **Quadrature 法**（$\mu_1 = 1/\sqrt{3}$）：
  $$
  \gamma_1 = \frac{\sqrt{3}}{2}\left[2 - \omega_0'(1 + g')\right], \qquad \gamma_2 = \frac{\sqrt{3}}{2}\omega_0'(1 - g'), \qquad \gamma_3 = \frac{1}{2}\left(1 - \sqrt{3} g' \mu_0\right), \qquad \gamma_4 = 1 - \gamma_3.
  $$
- **長波バンド（LW1, LW2）**：$\tau \to \infty$ で厳密に黒体射出率 $1$ へ収束する **Hemispheric Mean 法**（$\mu_1 = 1/2$, $\omega_0 = 0 \implies \gamma_1 = 2,\ \gamma_2 = 0$）を用い、層内の光学的厚さに対する Planck バンド放射発散度の1次展開 $F_{B,n}(\tau) = F_{B,0n} + F_{B,1n}\tau$（$F_{B,1n} = [F_B(T_{\mathrm{bot}, n}) - F_B(T_{\mathrm{top}, n})]/\tau_n$）から特解を次式で与える（Toon et al., 1989, Equations 26–27）：
  $$
  C_n^\pm(\tau) = F_{B,0n} + F_{B,1n}\left(\tau \pm \frac{1}{\gamma_{1n} + \gamma_{2n}}\right).
  $$

$\lambda_n = \sqrt{\gamma_{1n}^2 - \gamma_{2n}^2}$、$\Gamma_n = \gamma_{2n} / (\gamma_{1n} + \lambda_n)$、および Toon et al. (1989, Equations 44) の係数 $e_{1n}, \dots, e_{4n}$ を用いて TOA・層間連続・地表面反射／射出境界条件を課すことで、全層の放射伝達は以下の $2N$ 元三重対角連立一次方程式に帰着する：
$$
A_l Y_{l-1} + B_l Y_l + D_l Y_{l+1} = E_l \qquad (l = 1, \dots, 2N).
$$
AOBA はこれを融合バッチ Thomas カーネル（`src/aoba/kernels.py` の `solve_tridiagonal_batched`）によって全大気コラム並列に一括求解する。

### 4.2 4バンド分割と気体・雲光学特性

1. **SW1（紫外・可視光・PAR バンド：$0.25\text{--}0.69\ \mu\mathrm{m}$、太陽エネルギー配分比 $w_{\mathrm{SW1}} = 0.459760$）**：
   - 背景オゾン体積混合比は Wing et al. (2018, Equation 1) の水平一様ガンマ分布で与える：
     $$
     \mathrm{O}_3(p) = g_1 \, p_{\mathrm{hPa}}^{g_2} \exp\!\left(-\frac{p_{\mathrm{hPa}}}{g_3}\right) \times 10^{-6}
     $$
     （$g_1 = 3.6478,\ g_2 = 0.83209,\ g_3 = 11.3515\ \mathrm{hPa}$）。Rodgers 拡大因子 $M(\mu_0) = 35 / \sqrt{1224\mu_0^2 + 1}$ を掛けた傾斜オゾン量 $x = u_{\mathrm{O3}} M(\mu_0)$ に対して Lacis & Hansen (1974, Equations 8–12) の紫外＋Chappuis 帯吸収関数 $A_{\mathrm{oz}}(x)$ と分子 Rayleigh 散乱（$\Delta \tau_{\mathrm{ray}} = 0.095 \Delta p / p_0$）を適用する。
   - 液相雲・氷相雲の光学特性 $(\tau, \omega_0, g)$ は Slingo (1989, Table 1 Band 1) および Ebert & Curry (1992, Table 2 Band 1) に従い、地表到達下向き SW1 フラックスから光合成有効光量子束密度 $\mathrm{PPFD} = 4.57 \times 0.88 \, F_{\mathrm{SW1}}^\downarrow(z_{\mathrm{datum}})$（$\mathrm{\mu mol\ m^{-2}\ s^{-1}}$）を直接算出する。
2. **SW2（近赤外バンド：$0.69\text{--}4.00\ \mu\mathrm{m}$、太陽エネルギー配分比 $w_{\mathrm{SW2}} = 0.540240$）**：
   - 圧力・温度スケーリング可降水量 $\Delta y = 0.1 \, q_v \Delta m_d (p/p_0)\sqrt{T_0/T}$（$\mathrm{cm}$）に対する Lacis & Hansen (1974, Equation 21) の水蒸気吸収関数：
     $$
     A_{\mathrm{wv}}(y / \mu_0) = \frac{2.9 (y / \mu_0)}{\left[1 + 141.5 (y / \mu_0)\right]^{0.635} + 5.925 (y / \mu_0)}
     $$
     と、近赤外域の太陽重み平均雲光学特性（Slingo, 1989, Equation 25 & Table 1 Bands 2–4; Ebert & Curry, 1992, Table 2 Bands 2–5）を結合する。
3. **LW1（水蒸気・$\mathrm{CO}_2$ 回転振動吸収帯：Planck 配分比 $1 - \beta = 0.680$）**：
   - 圧力スケーリング水蒸気線吸収と対数型 $15\ \mu\mathrm{m}$ $\mathrm{CO}_2$ 吸収（Weaver & Ramanathan, 1995; Koll & Cronin, 2018; Sherwood et al., 2020）および水雲・氷雲吸収（Stephens, 1978; Ebert & Curry, 1992, Table 3）を次式で与える：
     $$
     \Delta \tau_{\mathrm{LW1}} = 0.30 \left(\frac{p}{p_0}\right)^{0.55} q_v \Delta m_d + \max\!\left[0.02,\ 0.85 + 0.22 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} + 0.020 \frac{\Delta p}{p_0} + \Delta \tau_{\mathrm{cld,LW1}}.
     $$
4. **LW2（波長 $8.0\text{--}12.5\ \mu\mathrm{m}$ 大気の窓バンド：Planck 配分比 $\beta = 0.320$）**：
   - 温度依存性を持つ水蒸気 e-type 自己連続吸収（Roberts et al., 1976; Koll & Cronin, 2018）、外来連続吸収、$\mathrm{CO}_2$ 窓端重なり、および雲の窓領域吸収（Ebert & Curry, 1992, Table 3 Band 2）を次式で与える：
     $$
     \Delta \tau_{\mathrm{LW2}} = \left[0.45 \frac{e_v}{p_0} \exp\!\left(\frac{1800}{T} - \frac{1800}{296}\right) + 0.0025 \frac{p}{p_0}\right] q_v \Delta m_d + \max\!\left[0,\ 0.015 + 0.024 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} + \Delta \tau_{\mathrm{cld,LW2}}.
     $$

### 4.3 エネルギー保存型の放射加熱率

層境界 $k + \frac{1}{2}$ における正味上向き全放射フラックスを $F_{\mathrm{net}, k+1/2} = (F_{\mathrm{SW}, k+1/2}^\uparrow - F_{\mathrm{SW}, k+1/2}^\downarrow) + (F_{\mathrm{LW}, k+1/2}^\uparrow - F_{\mathrm{LW}, k+1/2}^\downarrow)$ とすると、第 $k$ 層の放射温度変化率は次式で求まり、TOA 放射収支と地表面正味放射の差に対して鉛直積分で厳密にエネルギーを保存する：
$$
\left(\frac{\partial T}{\partial t}\right)_{\mathrm{rad}, k} = -\frac{F_{\mathrm{net}, k+1/2} - F_{\mathrm{net}, k-1/2}}{(c_{pd} + c_{pv} q_{v,k}) \rho_{d,k} \Delta z_k}.
$$
