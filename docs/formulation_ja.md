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


---

## 5. 混合相1モーメント雲微物理と降水沈降過程（`src/aoba/microphysics.py`）

### 5.1 降水粒子の温度ランプ診断分配

System for Atmospheric Modeling（SAM; Khairoutdinov & Randall, 2003, Appendices A–B）に準拠し、AOBA は非降水総水分 $q_n = q_v + q_c + q_i$ と降水総水分 $q_p = q_r + q_s + q_g$（雨・雪・あられ）の2変数を予報する。雲凝結水 $(q_c, q_i)$ が第 3.2 節の $\omega_n(T)$ で診断分配されるのと同様に、降水総水分 $q_p$ は $[268.15, 273.16]\ \mathrm{K}$ の液相ランプ $\omega_p(T)$ と $[223.15, 273.16]\ \mathrm{K}$ のあられランプ $\omega_g(T)$ によって雨 $q_r$・雪 $q_s$・あられ $q_g$ へ診断配分される（Khairoutdinov & Randall, 2003, Equations A15–A17）：
$$
\begin{aligned}
\omega_p(T) &= \mathrm{clip}\!\left(\frac{T - 268.15}{273.16 - 268.15},\ 0,\ 1\right), \qquad \omega_g(T) = \mathrm{clip}\!\left(\frac{T - 223.15}{273.16 - 223.15},\ 0,\ 1\right), \\
q_r &= \omega_p q_p, \qquad q_s = (1 - \omega_p)(1 - \omega_g) q_p, \qquad q_g = (1 - \omega_p)\omega_g q_p.
\end{aligned}
$$

### 5.2 自動変換・捕捉成長・蒸発／昇華・サブサイクル沈降

Marshall–Palmer 型指数粒径分布 $N_x(D) = N_{0x} \exp(-\lambda_x D)$（切片定数 $(N_{0r}, N_{0s}, N_{0g}) = (8 \times 10^6,\ 3 \times 10^6,\ 4 \times 10^6)\ \mathrm{m^{-4}}$）およびべき乗落下速度 $V_x(D) = a_x D^{b_x} (\rho_0 / \rho_d)^{1/2}$ に基づき、以下の微物理変換率を評価する（Kessler, 1969; Lin, Farley, & Orville, 1983; Khairoutdinov & Randall, 2003, Equations A18–A33）：
1. **雲水自動変換および雲氷凝集変換**（閾値 $q_{c0} = 10^{-3}\ \mathrm{kg\ kg^{-1}},\ q_{i0} = 10^{-4}\ \mathrm{kg\ kg^{-1}}$）：
   $$
   \left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{auto}} = \alpha_{\mathrm{liq}} \max(q_c - q_{c0},\ 0) + \beta_{\mathrm{ice}} \exp\!\left[0.025(T - 273.16)\right] \max(q_i - q_{i0},\ 0)
   $$
   （$\alpha_{\mathrm{liq}} = 10^{-3}\ \mathrm{s^{-1}},\ \beta_{\mathrm{ice}} = 10^{-3}\ \mathrm{s^{-1}}$）。
2. **連続捕捉成長（Accretion）**：落下する雨・雪・あられによる雲水・雲氷の捕捉をガンマ関数積分の解析解 $A_{xy} q_y (\rho_d q_x)^{(3 + b_x)/4}$（$x \in \{r, s, g\},\ y \in \{c, i\}$）から算出する（Khairoutdinov & Randall, 2003, Equations A21–A27）。
3. **未飽和気層での降水蒸発・昇華**：無雲かつ未飽和（$q_v < q_{\mathrm{sat}}$）の気層において、通風因子および熱伝導・水蒸気拡散の乾湿計抵抗を含む蒸発・昇華率を評価する（Khairoutdinov & Randall, 2003, Equations A29–A32）。
4. **質量重み終端落下速度と単調風上沈降スキーム**：バルク降水落下速度は次式で与えられる：
   $$
   V_p = \omega_p V_r + (1 - \omega_p)\left[(1 - \omega_g) V_s + \omega_g V_g\right], \qquad V_x = \frac{a_x \Gamma(4 + b_x)}{6 \lambda_x^{b_x}} \left(\frac{\rho_0}{\rho_d}\right)^{1/2}.
   $$
   下向き降水質量フラックス $P_z = \rho_d q_p V_p$ は CFL 条件（$\max(V_p \Delta t_{\mathrm{sed}} / \Delta z_k) \le 0.8$）を満たすサブサイクル1次精度単調風上法で時間積分し、相変化に伴う潜熱放出・吸収を湿潤修正温位へ厳密に反映させる：
   $$
   \Delta \theta_m = \frac{1 + (R_v / R_d) q_v}{\Pi c_{pd}} \left[\Delta(L_{\mathrm{eff},n} q_{\mathrm{cond}}) + \Delta(L_{\mathrm{eff},p} q_p)\right].
   $$

---

## 6. 粗度サブレイヤ（RSL）接地層・MYNN Level 2.5 境界層・3次元 SGS 乱流（`src/aoba/turbulence.py`）

### 6.1 植生キャノピーおよび海面の空気力学的粗度

1. **植生陸面（Raupach, 1994, Equations 8 & 10–11）**：外部土地被覆マップを用いず、予報樹高 $h_c$ と積雪露出葉面積指数 $L_c$（前面面積指数 $\Lambda = L_c / 2$）からゼロ面変位 $d$ と運動量粗度長 $z_{0m}$ を直接算出する：
   $$
   \begin{aligned}
   \frac{d}{h_c} &= 1 - \frac{1 - \exp\!\left(-\sqrt{c_{d1} L_c}\right)}{\sqrt{c_{d1} L_c}}, \\
   \frac{z_{0m}}{h_c} &= \left(1 - \frac{d}{h_c}\right) \exp\!\left(-\kappa \gamma_u + \psi_{h,\mathrm{cw}}\right), \qquad \gamma_u = \min\!\left[\left(C_S + C_R \frac{L_c}{2}\right)^{-1/2},\ \gamma_{\max}\right]
   \end{aligned}
   $$
   （$c_{d1} = 7.5,\ C_S = 0.003,\ C_R = 0.30,\ \psi_{h,\mathrm{cw}} = 0.193,\ \gamma_{\max} = 3.33$；スカラー粗度は $\ln(z_{0m}/z_{0h}) = k B^{-1} \approx 2.0$ とし、$L_c \to 0$ では裸地粗度 $0.01\ \mathrm{m}$・雪面粗度 $0.002\ \mathrm{m}$ へ滑らかに接続する）。
2. **開水面（Miller, Beljaars, & Palmer, 1992, Equations 3–5; Nishizawa et al., 2025, Equations 8.360–8.362）**：弱風時の分子粘性境界層と強風時の Charnock 波浪粗度（$\alpha_{\mathrm{Ch}} = 0.018,\ \nu_a = 1.5 \times 10^{-5}\ \mathrm{m^2\ s^{-1}}$）を結合する：
   $$
   z_{0m} = 0.11 \frac{\nu_a}{u_*} + \frac{\alpha_{\mathrm{Ch}} u_*^2}{g}, \qquad z_{0h} = 0.40 \frac{\nu_a}{u_*} + 1.4 \times 10^{-5}, \qquad z_{0q} = 0.62 \frac{\nu_a}{u_*} + 1.3 \times 10^{-4}.
   $$

### 6.2 解析的粗度サブレイヤ（RSL）補正付き Monin–Obukhov 相似則

高木林上で最下層高度 $\tilde{z}_1 = z_{c,0} - (z_{\mathrm{topo}} + d)$ が粗度サブレイヤ厚 $z_* = \max(2 h_c - d,\ 0)$ の内部に入る場合（Harman & Finnigan, 2007）、De Ridder (2010, Equations 6–7 & 22; Bonan et al., 2018; Lee et al., 2020) による指数積分 $E_1$ を用いた閉形式 RSL 補正関数 $\hat{\psi}_{M,H}(\zeta, \tilde{z}_1 / z_*)$ を導入する：
$$
\begin{aligned}
F_M &= \ln\!\left(\frac{\tilde{z}_1}{z_{0m}}\right) - \psi_M(\zeta) + \psi_M\!\left(\zeta \frac{z_{0m}}{\tilde{z}_1}\right) + \hat{\psi}_M\!\left(\zeta, \frac{\tilde{z}_1}{z_*}\right), \\
\hat{\psi}_\alpha(\zeta, \eta) &= \phi_\alpha\!\left(\zeta \eta^{-1}(1 + \nu_{\mathrm{RSL}}/\mu_\alpha)\right) \left(1 - \frac{1}{c_{2\alpha}}\right) \exp(-\mu_\alpha \eta) \left[\ln(\mu_\alpha \eta) + E_1(\mu_\alpha \eta)\right].
\end{aligned}
$$
バルク Richardson 数 $\mathrm{Ri}_B$ から Businger–Dyer–Paulson（不安定側）および Beljaars & Holtslag (1991)（安定側）の普遍関数を用いて無次元安定度 $\zeta = \tilde{z}_1 / L_{\mathrm{MO}}$ を反復求解し、バルク輸送係数 $C_M = \kappa^2 / F_M^2,\ C_{H,Q} = \kappa^2 / (F_M F_{H,Q})$ を得る。

### 6.3 陰解法 MYNN Level 2.5 境界層乱流クロージャ

鉛直大気境界層混合および層積雲頂乱流には、2倍乱流運動エネルギー $q^2 = 2e$ を予報する Mellor–Yamada–Nakanishi–Niino Level 2.5 スキーム（Nakanishi & Niino, 2009; Nishizawa et al., 2025, Section 8.2.1, Equations 8.96–8.176）を採用する：
$$
\frac{\partial (\rho_d q^2)}{\partial t} = \frac{\partial}{\partial z}\!\left(\rho_d K_q \frac{\partial q^2}{\partial z}\right) + 2 \rho_d \left(P_s + P_b - \frac{q^3}{B_1 L}\right),
$$
ここで乱流拡散係数は $K_M = L q S_M,\ K_H = L q S_H,\ K_q = 3 L q S_M$ であり、マスター混合距離 $L$ は接地スケール $L_S$・境界層積分スケール $L_T$・浮力スケール $L_B$ の調和平均で定義される：
$$
\frac{1}{L} = \frac{1}{L_S} + \frac{1}{L_T} + \frac{1}{L_B}, \qquad L_T = 0.23 \frac{\int_0^\infty q \tilde{z} \, dz}{\int_0^\infty q \, dz}.
$$
サブグリッド正規分布凝結診断（飽和分散 $\sigma_s$、部分雲量 $R$、サブグリッド凝結水量 $Q_l$）から湿潤浮力係数 $(\beta_\theta, \beta_q)$ を評価し、非平衡補正因子 $\alpha_c = \min(q / q_{\mathrm{L2}},\ 1)$ を含む安定度関数 $(S_M, S_H)$ とともに `solve_tridiagonal_batched` による完全陰解法で時間積分する。

### 6.4 3次元非等方 Smagorinsky–Lilly サブグリッド乱流モデル

3次元 CRM 内の水平サブグリッド乱流混合には、扁平格子アスペクト比 $a = \max(\Delta x, \Delta y) / \Delta z_k$ に対する Scotti, Meneveau, & Lilly (1993) 補正関数 $f(a)$ を組み込んだ Richardson 数依存型 Smagorinsky–Lilly モデル（Nishizawa et al., 2025, Section 8.1.2, Equations 8.25–8.40）を適用する：
$$
\nu_{\mathrm{SGS}} = (C_s f(a) \Delta)^2 |\mathbf{S}| \sqrt{\max(0,\ 1 - \mathrm{Ri}/\mathrm{Ri}_c)}, \qquad \Delta = (\Delta x \Delta y \Delta z_k)^{1/3}
$$
（$C_s = 0.18,\ \mathrm{Ri}_c = 0.25$）。

---

## 7. 非直角双曲線型 Two-Leaf 生態進化最適性（EEO）陸上生態系（`src/aoba/biosphere.py`）

### 7.1 最小コスト最適気孔比および非直角双曲線光合成協調理論

AOBA は外部の植生機能型（PFT）地図を一切用いず、全球の光合成生化学容量・気孔コンダクタンス・C₃/C₄ 優占度・葉面積指数・根圏貯水容量・樹高を生態進化最適性（EEO）第一原理から自律予測する。
1. **C₃ 最小コスト $c_i / c_a$ と非直角双曲線型 P モデル（Prentice et al., 2014; Smith et al., 2019, Equations 1–20）**：
   カルボキシル化維持コストと蒸散水輸送維持コストの和（$\beta_{\mathrm{C3}} = 146$）の最小化から、最適葉内／大気 $\mathrm{CO}_2$ 分圧比 $\chi_{\mathrm{C3}}$ が導かれる：
   $$
   \chi_{\mathrm{C3}} = \frac{\Gamma^*}{c_a} + \left(1 - \frac{\Gamma^*}{c_a}\right) \frac{\xi_{\mathrm{C3}}}{\xi_{\mathrm{C3}} + \sqrt{D}}, \qquad \xi_{\mathrm{C3}} = \sqrt{\frac{\beta_{\mathrm{C3}}(K_M + \Gamma^*)}{1.6 \eta^*(T_g)}}.
   $$
   電子伝達速度 $J$ は曲率因子 $\theta = 0.85$ を持つ Farquhar & Wong (1984) の非直角双曲線に従う：
   $$
   \theta J^2 - (\varphi_{\mathrm{PSII}} I_c + J_{\max}) J + \varphi_{\mathrm{PSII}} I_c J_{\max} = 0.
   $$
   限界投資条件 $\partial A_j / \partial J_{\max} = c$（$c = 0.05336251$）および $A_c = A_j$ の生化学的協調条件を連立すると、$m = (c_i - \Gamma^*)/(c_i + 2\Gamma^*) > 4c$ の全域で連続微分可能な符号付き解析解 $(\omega, \omega^*)$ が得られる：
   $$
   \begin{aligned}
   \omega &= -(1 - 2\theta) + \mathrm{sgn}\!\left(1 - \frac{8\theta c}{m}\right) \sqrt{(1 - \theta)\left[\frac{1}{(4c/m)(1 - \theta \cdot 4c/m)} - 4\theta\right]}, \\
   \omega^* &= 1 + \omega - \sqrt{(1 + \omega)^2 - 4\theta \omega}, \\
   J_{\max,\mathrm{opt}} &= \varphi_{\mathrm{PSII}}(T_g)\, \tilde{I}_c \, \omega, \qquad V_{c\max,\mathrm{opt}} = \varphi_{\mathrm{PSII}}(T_g)\, \tilde{I}_c \, \frac{m}{m_c} \frac{\omega^*}{8\theta}
   \end{aligned}
   $$
   （$m_c = (c_i - \Gamma^*)/(c_i + K_M)$；$m \le 4c$ では酵素投資は滑らかにゼロとなる）。基準温度 $25\ ^\circ\mathrm{C}$ 換算容量 $(V_{c\max 25}, J_{\max 25})$ はエントロピー順化型ピーク付き Arrhenius 関数（Kattge & Knorr, 2007; Smith et al., 2019, Equation 20）により算出する。
2. **C₄ 三酵素協調モデル（Scott & Smith, 2022, Equations 1–25; von Caemmerer & Furbank, 1999）**：
   葉肉細胞 $\mathrm{CO}_2$ 比 $\chi_{m,\mathrm{C4}} = \xi_{\mathrm{C4}} / (\xi_{\mathrm{C4}} + \sqrt{D})$（$\xi_{\mathrm{C4}} = \sqrt{\beta_{\mathrm{C4}} K_p / (1.6 \eta^*)},\ \beta_{\mathrm{C4}} = 166$）、維管束鞘細胞からの漏出率 $\phi_L = 0.20$、維管束鞘コンダクタンス $g_{bs} = 0.003 L_c\ \mathrm{mol\ m^{-2}\ s^{-1}}$、および PEP 再生を含む5電子化学量論（$k_e = 5\ \mathrm{mol(e^-)\ mol(CO_2)^{-1}}$）を用い、維管束鞘 $\mathrm{CO}_2$ 分圧 $C_{bs} = C_m + (L / g_{bs}) p_{\mathrm{sfc}}$ に関する2回の不動点反復から最適 $(J_{\max,\mathrm{C4}}, V_{p\max,\mathrm{C4}}, V_{c\max,\mathrm{C4}})$ を決定する。
3. **普遍 EEO 葉暗呼吸（Heskel et al., 2016; Ren et al., 2024, 2025, Equation 13）**：
   夜間葉呼吸は最適 Rubisco 容量に比例して $R_{\mathrm{leaf}}(T_{\mathrm{night}}) = b_{\mathrm{resp}} V_{c\max,\mathrm{opt}}$（$b_{\mathrm{resp}} = 0.018$）と定まり、日内の瞬時温度応答は高分子速度理論（MMRT）に従う：
   $$
   f_R(T) = \exp\!\left[0.1012(T_C - 25) - 0.0005(T_C^2 - 25^2)\right].
   $$

### 7.2 正午下向き短波放射の EMA 平滑化と閉形式 Two-Leaf 酵素配分

経験的な群落内窒素減衰係数 $k_n$ を与えることなく群落内の非線形光飽和を表現するため、以下の厳密な2段階順化・配分アルゴリズムを実装する（de Pury & Farquhar, 1997; Hashimoto, 2026）：
1. **地方太陽正午の下向き短波放射 EMA**：各陸面コラムの地方太陽正午通過時（$H = 0$）における下向き可視直達・散乱短波放射フラックス $(\mathrm{VIS}_b, \mathrm{VIS}_d)$ および太陽天頂角余弦 $\mu_0$ を指数移動平均（$\alpha_{\mathrm{EMA}} = 0.067$、時定数約15日）により平滑化し、$(\widetilde{\mathrm{VIS}}_b, \widetilde{\mathrm{VIS}}_d, \tilde{\mu}_0)$ を保持する。
2. **球面葉群2ストリーム光吸収計算**：平滑化された正午放射環境 $(\widetilde{\mathrm{VIS}}_b, \widetilde{\mathrm{VIS}}_d, \tilde{\mu}_0)$ を**現在時刻の積雪露出葉面積指数 $L_c$** における de Pury & Farquhar (1997) 球面葉分布モデルへ入力し、順化条件の群落全吸収光量子束 $\tilde{I}_c = \tilde{I}_{cb} + \tilde{I}_{cd}$ と群落内光減衰プロファイル $\tilde{I}_l(L) = \tilde{I}_{lb}(L) + \tilde{I}_{ld}(L)$（減衰係数 $\tilde{k}_b' = 0.5\sqrt{1 - \sigma_P}/\tilde{\mu}_0,\ k_d' = 0.719$）を求める。
3. **陽葉・陰葉への閉形式解析的配分**：群落内の光合成酵素分布が順化吸収光プロファイルに比例（$X_l(L) = X (\tilde{I}_l(L) / \tilde{I}_c)$）すると仮定し、各タイムステップの**瞬時直達光減衰確率** $\exp(-k_{db}(t) L)$（$k_{db}(t) = 0.5 / \mu_0(t)$）を掛けて群落積分することで、任意の全群落容量 $X \in \{V_{c\max 25}, J_{\max 25}, V_{p\max 25}, R_{25}\}$ を陽葉プール $X_{\mathrm{sun}}$ と陰葉プール $X_{\mathrm{sh}}$ へ厳密保存で解析配分する：
   $$
   \begin{aligned}
   X_{\mathrm{sun}} &= \frac{X}{\tilde{I}_c} \left[\tilde{I}_{b0}(1 - \tilde{\rho}_{cb})\tilde{k}_b' \frac{1 - \exp\!\left[-(\tilde{k}_b' + k_{db})L_c\right]}{\tilde{k}_b' + k_{db}} + \tilde{I}_{d0}(1 - \rho_{cd})k_d' \frac{1 - \exp\!\left[-(k_d' + k_{db})L_c\right]}{k_d' + k_{db}}\right], \\
   X_{\mathrm{sh}} &= \max(10^{-6},\ X - X_{\mathrm{sun}}).
   \end{aligned}
   $$
   配分された $(I_{\mathrm{sun}}, X_{\mathrm{sun}})$ と $(I_{\mathrm{sh}}, X_{\mathrm{sh}})$ のそれぞれに対して非直角双曲線型光合成速度と最適気孔コンダクタンス $g_{sw,\mathrm{sun}}, g_{sw,\mathrm{sh}} = 1.6 A_n (R_{\mathrm{gas}} T_l / p_{\mathrm{sfc}}) / [c_a(1 - \chi)]$（$\mathrm{m\ s^{-1}}$）を独立評価する。

### 7.3 2ストリーム群落アルベド・自律 C₄ 割合 $F_4$・Lambert-$W_0$ 予報 LAI

1. **地図不要の2ストリーム群落アルベド**：可視・近赤外の直達／散乱アルベド $(\alpha_{\mathrm{VIS,dir}}, \alpha_{\mathrm{VIS,dif}}, \alpha_{\mathrm{NIR,dir}}, \alpha_{\mathrm{NIR,dif}})$ を、土壌水分依存の裸地反射率・積雪反射率と $(L_c, \mu_0)$ から Sellers (1985) および Dai et al. (2004, Appendix A, Equations A1–A19) の解析解により算出する。
2. **自律 C₄ 優占率 $F_4$**（Scott & Smith, 2022, Equation 26; Luo et al., 2024; Lavergne et al., 2026）：正午の純同化速度差 $\Delta A = (A_{n,\mathrm{C4}} - A_{n,\mathrm{C3}})/(|A_{n,\mathrm{C3}}| + 1)$ と樹冠開放度 $f_{\mathrm{open}}(L_c) = \mathrm{clip}(1.15 - 0.18 L_c,\ 0.02,\ 1)$ から動的に予報する。
3. **予報 EEO 葉面積指数・根圏貯水容量 $S_R$・樹高 $h_c$**：
   - 年最大緑被率 $\mathrm{fAPAR}_{\max}$ はエネルギー制限項 $1 - z/(k A_{0,\mathrm{yr}})$（$z = 12.227\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}},\ k = 0.5$）と水供給制限項 $f_0(\mathrm{AI}) P_{\mathrm{yr}} c_a (1 - \chi)/(1.6 D A_{0,\mathrm{yr}})$ の滑らかな log-sum-exp 最小化（$K = 10$；Cai et al., 2025, Equation 12; Zhou et al., 2025, Equations 6 & 11）で決定する。
   - 日定常葉面積指数 $L_s$ は Lambert $W$ 関数の主枝 $W_0$ を用いた閉形式解で与えられる（Zhou et al., 2025, Equations 15–20）：
     $$
     L_s = \min\!\left\{\mu + \frac{1}{k} W_0\!\left[-k\mu \exp(-k\mu)\right],\ \mathrm{LAI}_{\max}\right\}, \qquad \mu = \frac{\sigma_{\mathrm{Zhou}} \, \mathrm{GSL} \, \mathrm{LAI}_{\max}}{A_{0,\mathrm{yr}} \, \mathrm{fAPAR}_{\max}} A_{0,\mathrm{day}}.
     $$
   - 乾季乗り切りに必要な最適根圏貯水容量 $S_R$ は累積水不足量から（Gao et al., 2014）、潜在樹高 $h_c$ は維管束水・生産力収支（Kempes et al., 2011）から自律予報される。

---

## 8. 陸面水文・5層凍結融解土壌・積雪氷床・海氷過程（`src/aoba/surface.py`）

### 8.1 3成分蒸発散と陰的線形化地表面熱収支

陸面からの総水蒸気フラックスは、濡れキャノピー遮断蒸発（$f_{\mathrm{wet}} = (W_c / W_{c,\max})^{2/3}$；MATSIRO6 第7章）、陽葉＋陰葉の気孔蒸散（$g_{sw,\mathrm{sun}} + g_{sw,\mathrm{sh}}$）、積雪被覆率 $A_{\mathrm{sn}}$ 上の昇華、および乾燥表層の水蒸気分子拡散抵抗 $r_{\mathrm{soil}} = (L_{\mathrm{dry}} / D_v)(\tau_{\mathrm{tort}} / \theta_{\mathrm{air}})$ を組み込んだ裸地蒸発（Sakaguchi & Zeng, 2009, Equations 7–10）の和として計算される。地表面スキン温度変化 $\Delta T_{\mathrm{skin}}$ は放射・顕熱・潜熱・地中伝導熱の1次テイラー展開から毎ステップ陰的に解かれ、積雪存在時に $T_{\mathrm{skin}} > 273.15\ \mathrm{K}$ となった余剰エネルギーはすべて融雪熱へ振り向けられる。

### 8.2 サブグリッド積雪被覆率・植生埋没効果・積雪老化・氷河流出

1. **サブグリッド積雪被覆率 $A_{\mathrm{sn}}$**：Earth2014 のサブグリッド標高標準偏差 $\sigma_{\mathrm{topo}}$ と融雪期の履歴曲線を考慮した Swenson & Lawrence (2012, Equations 3–6) および Nitta et al. (2014) に従う。
2. **植生の積雪埋没効果**：積雪域内の露出樹高・LAI を $h_{\mathrm{buried}} = \max(h_c - d_{\mathrm{snow}},\ 0)$、$L_{\mathrm{buried}} = L_c (h_{\mathrm{buried}} / h_c)$ により減少させることで（MATSIRO6 Equations 2.1–2.3）、ツンドラの雪面高アルベドと北方タイガ林の低アルベド（樹冠露出）の対比を PFT 表なしで再現する。
3. **積雪粒径老化と氷河流出**：無次元積雪年齢 $\tau_{\mathrm{sn}}$ の熱成長と新雪リセットにより可視・近赤外雪面アルベドを更新し（Yang et al., 1997; MATSIRO6 Section 8.8.1）、積雪水当量が上限 $W_{\mathrm{sn},\max} = 1000\ \mathrm{kg\ m^{-2}}$ を超えた超過分は氷河分離流出 $R_{\mathrm{ice}}$ として沿岸海洋セルへ還元する（MATSIRO6 Equation 8.44）。

### 8.3 ゼロカーテン相変化を伴う5層陰解法土壌熱力学

5層の土壌層（層厚 $\Delta z_s = [0.05, 0.15, 0.40, 0.80, 1.60]\ \mathrm{m}$、全深 $3.0\ \mathrm{m}$）に対し、基準温度 $T_0 = 273.15\ \mathrm{K}$ からの相対内部エネルギー $U_l = C_l (T_l - T_0) - L_{f0} M_{i,l}$（$C_l = C_{\mathrm{dry}} + c_w M_{w,l} + c_i M_{i,l}$）を定義する（Nishizawa et al., 2025, Section 8.7.1, Equations 8.390–8.418; MATSIRO6 第10章）。$0\ ^\circ\mathrm{C}$ における液水・土壌氷の厳密エンタルピー保存相分配 $M_{i,l}^{\mathrm{new}} = \mathrm{clip}(-U_l / L_{f0},\ 0,\ M_{w,l} + M_{i,l})$ に続き、摂氏偏差 $(T_l - T_0)$ に関する陰解法 Fourier 熱伝導方程式を `solve_tridiagonal_batched` で解くことで、GPU 単精度演算でも丸め誤差のない永久凍土・活動層サイクルを表現する。

### 8.4 太陽天頂角依存の海面アルベドとリード面積率付き熱力学海氷

開水面の直達短波アルベドは低太陽高度での Fresnel 反射増大を表す Briegleb et al. (1986) 式に従う：
$$
\alpha_{\mathrm{ocn,dir}}(\mu_0) = \frac{0.026}{\mu_0^{1.7} + 0.065} + 0.15(\mu_0 - 0.1)(\mu_0 - 0.5)(\mu_0 - 1.0)
$$
（散乱アルベド $\alpha_{\mathrm{ocn,dif}} = 0.06$）。海氷密接度は $f_i = \min(\sqrt{M_i / M_{i,\mathrm{crit}}},\ 1)$（$M_{i,\mathrm{crit}} = 300\ \mathrm{kg\ m^{-2}}$；Nishizawa et al., 2025, Equation 8.342）で与え、開水リード面 $(1 - f_i)$ と海氷・氷上積雪面 $f_i$ の熱収支を分離評価する（Semtner, 1976）。

---

## 9. 埋め込み3次元非静力学 RK3–HEVI 雲解像モデルと Zalesak FCT（`src/aoba/crm.py`）

### 9.1 フラックス形式圧縮性非静力学方程式系と陰的鉛直 HEVI ソルバ

各ホストセル内部には二重周期境界を持つ $N_x \times N_y = 8 \times 8$（$\Delta x = \Delta y = 4\ \mathrm{km}$）の Arakawa C 格子／Charney–Phillips 3次元 CRM が 1:1 で埋め込まれ、$(\rho_d, u, v, w, \Theta_m = \rho_d \theta_m)$ に関するフラックス形式完全圧縮性湿潤 Euler 方程式を解く（Skamarock et al., 2012, Equations 7–11; Nishizawa et al., 2025, Chapter 3）：
$$
\begin{aligned}
\frac{\partial \rho_d}{\partial t} + \nabla_h \cdot (\rho_d \mathbf{u}_h) + \frac{\partial F_z}{\partial z} &= 0, \qquad F_z = \rho_{d,w} w, \\
\frac{\partial \Theta_m}{\partial t} + \nabla_h \cdot (\Theta_m \mathbf{u}_h) + \frac{\partial (\theta_{m,w} F_z)}{\partial z} &= S_{\Theta_m}, \\
\rho_{\mathrm{tot},w} \frac{\partial w}{\partial t} + \rho_{\mathrm{tot},w} (\mathbf{u} \cdot \nabla w) &= -\frac{\partial p}{\partial z} - \rho_{\mathrm{tot},w} g - \rho_{\mathrm{tot},w} \gamma_R(z_w) w + D_w.
\end{aligned}
$$
時間積分には Wicker & Skamarock (2002) の3段 Runge–Kutta 法（$\Delta \tau \in \{\Delta t/3,\ \Delta t/2,\ \Delta t\}$）と偏心重み $\beta = 0.60$ の水平陽解法・鉛直陰解法（HEVI; Klemp, Skamarock, & Dudhia, 2007）を用いる。状態方程式の微分関係 $\partial p / \partial \Theta_m = \gamma p / \Theta_m$ を用いて $(\delta \rho_d, \delta \Theta_m)$ を消去することで、鉛直質量フラックス増分 $\delta F_z = F_z^{\mathrm{new}} - F_z^{\mathrm{old}}$ に関する狭義対角優位な $(N_z - 1)$ 元三重対角 Helmholtz 方程式が導かれ、`solve_tridiagonal_batched` による全サブコラム一括求解により乾燥空気質量と $\Theta_m$ を機械精度で保存する。

### 9.2 多次元正定値 Zalesak FCT トレーサー移流

水物質混合比 $(q_n, q_p)$ は連続の式と整合する質量フラックス $(F_x, F_y, F_z)$ を用い、1次精度単調風上フラックス $\mathbf{F}^L$ と4次精度中心差分フラックス $\mathbf{F}^H$ を Zalesak (1979, Equations 1–17; Nishizawa et al., 2025, Appendix A.2, Equations A.20–A.35) の3次元反拡散リミッタ $C \in [0, 1]$ で結合して移流させる。これにより $q_n, q_p \ge 0$ の厳密な非負性と3次元水物質総量の完全保存が同時に保証される。

---

## 10. 全球 SCVT ボロノイ C 格子力学コア・山岳抵抗・ボロノイ Ekman 海洋・1:1 結合

### 10.1 SCVT ボロノイ格子・TRiSK 演算子・Earth2014 集約（`src/aoba/scvt_mesh.py`, `scvt_dynamics.py`）

正二十面体の再帰的細分割（セル数 $N_{\mathrm{cells}} = 10 \cdot 4^g + 2$）と球面 Lloyd 緩和法（Ringler, Ju, & Gunzburger, 2008）により、厳密な球面過剰面積 $A_i$（$\sum_i A_i = 4\pi R_E^2$）、セル中心間測地線距離 $d_{c,e}$、ボロノイ辺長 $l_e$、および双対 Delaunay 三角形面積 $A_v$ を構築する。水平運動方程式には Thuburn et al. (2009) および Ringler et al. (2010, Equations 17–26) のベクトル不変形式を採用する：
$$
\frac{\partial u_e}{\partial t} = \sum_{e' \in \mathrm{EC}(e)} w_{e,e'} \frac{\eta_e + \eta_{e'}}{2} u_{e'} - \frac{K_{c_1} - K_{c_0}}{d_{c,e}} - \frac{1}{\rho_{\mathrm{tot},e}} \left[\left(\frac{\partial p}{\partial n}\right)_\zeta - \frac{\partial p}{\partial z}\left(\frac{\partial z}{\partial n}\right)_\zeta\right] + D_{u,e}.
$$
ここで絶対渦度 $\eta_v = \zeta_v + f_v$ に掛かる接線風速再構成重み $w_{e,e'}$ は反対称恒等式 $A_e w_{e,e'} + A_{e'} w_{e',e} = 0$（$A_e = \frac{1}{2} d_{c,e} l_e$）を厳密に満たし、コリオリ項および非線形渦度フラックス項による偽の運動エネルギー生成を恒等的にゼロに抑えるとともに、発散減衰および水平4階超粘性 $\nabla^4$ により格子スケール雑音を制御する（Skamarock et al., 2012, Section 3d）。さらに5分角 `Earth2014` データ（`SUR2014`, `ICE2014`, `TBI2014`; Hirt & Rexer, 2015）を各ボロノイセルへ球面面積重み集約し、平均標高 $z_{\mathrm{topo}}$、海陸氷床マスク、およびサブグリッド勾配共分散テンソル不変量 $(\mu, \sigma, \gamma, \psi)$ を算出する。

### 10.2 Lott & Miller (1997) サブグリッド山岳ブロック流・重力波抵抗（`src/aoba/orography.py`）

逆 Froude 数 $H_n = \bar{N}_L (2\mu) / |\mathbf{U}_L|$ が臨界値 $H_{nc} = 0.50$ を超える安定成層下では、ブロック高度 $Z_b = 2\mu(1 - H_{nc}/H_n)$ 以下の気層に楕円山岳の方向依存性を持つ形状抵抗を陰的に作用させ、上部の有効山岳振幅 $h_{\mathrm{eff}} = 2\mu - Z_b$ から励起される内部重力波運動量フラックスを Lindzen 飽和応力条件 $\tau_{\mathrm{sat}}(z) \propto \rho U_{\mathrm{proj}}^3 / (N H_{nc}^2)$ に基づき鉛直上方へ伝播・砕波減速させる（Lott & Miller, 1997, Equations 7–18）。

### 10.3 $Q$-flux 不要の 1.5 層ボロノイ風成 Ekman 海洋（`src/aoba/ocean_ekman.py`）

Codron (2012, Equations 1–10) に従い、外洋ボロノイ辺上で海面風応力 $\boldsymbol{\tau}_{\mathrm{ocn}}$ から表層混合層（層厚 $H_1 = 50\ \mathrm{m}$、水温 $T_1$）の風成 Ekman 質量輸送を診断する（赤道正則化摩擦 $r_E = (2\ \mathrm{days})^{-1}$；すべての海陸境界辺では法線輸送を厳密にゼロとする）：
$$
M_{E,x} = \frac{r_E \tau_{x,\mathrm{ocn}} + f \tau_{y,\mathrm{ocn}}}{f^2 + r_E^2}, \qquad M_{E,y} = \frac{r_E \tau_{y,\mathrm{ocn}} - f \tau_{x,\mathrm{ocn}}}{f^2 + r_E^2}.
$$
第2層躍層（層厚 $H_2 = 200\ \mathrm{m}$、水温 $T_2$）における逆向き補償流 $-\mathbf{M}_E$ と鉛直 Ekman 湧昇・沈降 $W_E = \nabla_{\mathrm{SCVT}} \cdot \mathbf{M}_E$、および水平中規模渦拡散を組み合わせることで、外部 $Q$-flux 気候値マップを一切用いずに全球海洋熱量を機械精度で保存しつつ赤道湧昇冷水舌と亜熱帯極向き熱輸送を表現する。また陸域・氷床からの氷河分離流出に伴う融解潜熱冷却を最寄りの沿岸海洋セルへ保存的に還元する（MIROC6 Section 4.1.2）。

### 10.4 1:1 超パラメタリゼーション結合と創発 Köppen–Geiger 気候区分診断（`src/aoba/coupled_model.py`, `src/aoba/koppen.py`）

すべての全球 SCVT ボロノイセルは結合時間間隔 $\Delta t_{\mathrm{host}}$ ごとに内部の $8 \times 8 \times 32$ 3次元 CRM と 1:1 で双方向結合する（Khairoutdinov et al., 2005; Yamazaki & Miura, 2024）。ホスト格子の大規模移流傾向 $(\partial_t \theta_m, \partial_t q_n)_{\mathrm{LS}}$ とセル中心風速が放射加熱率・地表面フラックスとともに CRM サブコラムを駆動し、CRM の水平平均状態と対流運動量輸送傾向がホスト SCVT 状態へ直接フィードバックされる。さらにシミュレーションから蓄積された月平均地上気温 $T_m$ と月降水量 $P_m$ に対し Beck et al. (2018, 2023, Table 1) の厳密な30区分 Köppen–Geiger 分類（`Af`〜`EF`；極気候 `E` を乾燥気候 `B` に優先判定、南北半球の夏冬反転対応）を適用することで、最小境界条件のみから創発する現在気候帯の再現性と高 $\mathrm{CO}_2$ 下での気候帯シフトを直接定量評価する。
