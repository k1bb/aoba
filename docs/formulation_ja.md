# AOBA 詳細数理定式化モデル記述書 (Model Formulation Document)

> **言語 (Language)**: [English (formulation.md)](formulation.md) | 日本語

本書は、結合気候・自律生態系モデル **AOBA** の連続系支配方程式、座標変換、スタガード空間離散化、時間積分法、物理・生態系・雪氷・海洋・動的氷床サブモデルの閉形式定式化、および統合設定・パラメータレジストリの全仕様を記述したモデル記述書である。各式の定数・記号は `src/aoba/` の実装コードおよび `config/aoba_default.toml` と 1対1 で対応する。

---

## 目次

1. [基礎物理定数と天文学的軌道日射幾何 (`constants.py`, `radiation.py`)](#1-基礎物理定数と天文学的軌道日射幾何)
2. [鉛直 Charney–Phillips 格子と平滑化地形追従座標 (`vertical_grid.py`)](#2-鉛直-charneyphillips-格子と平滑化地形追従座標)
3. [圧縮性湿潤熱力学と混合相ニュートン・ラフソン飽和調整 (`thermodynamics.py`)](#3-圧縮性湿潤熱力学と混合相ニュートンラフソン飽和調整)
4. [4バンド・ルックアップテーブル不要2ストリーム放射伝達 (`radiation.py`, `kernels.py`)](#4-4バンドルックアップテーブル不要2ストリーム放射伝達)
5. [混合相1モーメント雲微物理・降水落下 (`microphysics.py`)](#5-混合相1モーメント雲微物理降水落下)
6. [粗度サブレイヤ (RSL)・陰的 MYNN Level 2.5 PBL・3D SGS 乱流 (`turbulence.py`, `crm.py`)](#6-粗度サブレイヤ-rsl陰的-mynn-level-25-pbl3d-sgs-乱流)
7. [Two-Leaf 非直角双曲線 EEO 自律生態系 (`biosphere.py`)](#7-two-leaf-非直角双曲線-eeo-自律生態系)
8. [陸面水文・5層凍土熱伝導・積雪・海氷・開水面収支 (`surface.py`)](#8-陸面水文5層凍土熱伝導積雪海氷開水面収支)
9. [埋め込み 3D 非静力学 RK3–HEVI CRM と Zalesak FCT (`crm.py`)](#9-埋め込み-3d-非静力学-rk3hevi-crm-と-zalesak-fct)
10. [全球 SCVT C-grid 力学・山岳ドラッグ・エクマン海洋・MAML+ICA 結合 (`scvt_mesh.py`, `scvt_dynamics.py`, `orography.py`, `ocean_ekman.py`, `coupled_model.py`, `koppen.py`)](#10-全球-scvt-c-grid-力学山岳ドラッグエクマン海洋mamlica-結合)
11. [動的／平衡加速 SIA 氷床モデルと保存的気圧鉛直再補間 (`icesheet.py`)](#11-動的平衡加速-sia-氷床モデルと保存的気圧鉛直再補間)
12. [単一 TOML 設定管理・出力仕様・GUI コンフィギュレータ・全パラメータ一覧 (`params.py`, `config/aoba_default.toml`, `scripts/aoba_config_gui.py`)](#12-単一-toml-設定管理出力仕様gui-コンフィギュレータ全パラメータ一覧)

---

## 1. 基礎物理定数と天文学的軌道日射幾何

### 1.1 基礎物理定数 (`src/aoba/constants.py`)

AOBA で共有される基礎物理定数（Wing et al., 2018, Table 1; Khairoutdinov & Randall, 2003, Appendix A; MIROC6 AGCM, 2021, Sec. 1.3.2; Semtner, 1976）を表 1.1 に示す。乾燥気体・水蒸気の定積比熱および関連導出定数は、`c_vd = c_pd - R_d`、`gamma_d = c_pd / c_vd`、`kappa_d = R_d / c_pd`、`eps_water = R_d / R_v`、`z_vir = R_v / R_d - 1`、`L_s0 = L_v0 + L_f0` として `constants.recompute_derived_constants()` により TOML 設定変更時にも常に整合更新される。

| 記号 | 物理量 | 標準値 | 単位 | 根拠文献 |
| :--- | :--- | :--- | :--- | :--- |
| $R_E$ | 地球平均半径 | $6.371 \times 10^6$ | $\mathrm{m}$ | Wing et al. (2018) |
| $g$ | 地表面標準重力加速度 | $9.79764$ | $\mathrm{m\ s^{-2}}$ | Wing et al. (2018) |
| $\Omega$ | 地球自転角速度 | $7.292115 \times 10^{-5}$ | $\mathrm{rad\ s^{-1}}$ | MIROC6 (2021) |
| $S_0$ | 太陽定数 | $1361.0$ | $\mathrm{W\ m^{-2}}$ | Wing et al. (2018) |
| $R_d$ | 乾燥空気気体定数 | $287.04$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pd}$ | 乾燥空気定圧比熱 | $1004.64$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{vd}$ | 乾燥空気定積比熱 ($c_{pd} - R_d$) | $717.60$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Skamarock et al. (2012) |
| $R_v$ | 水蒸気気体定数 | $461.50$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $c_{pv}$ | 水蒸気定圧比熱 | $1846.0$ | $\mathrm{J\ kg^{-1}\ K^{-1}}$ | Wing et al. (2018) |
| $\epsilon$ | 分子量比 $R_d / R_v$ | $0.621972$ | $-$ | Skamarock et al. (2012) |
| $\delta_{\mathrm{vir}}$ | 仮温度係数 $R_v / R_d - 1$ | $0.607790$ | $-$ | Skamarock et al. (2012) |
| $p_0$ | 基準表面気圧 | $1.0 \times 10^5$ | $\mathrm{Pa}$ | Skamarock et al. (2012) |
| $T_0$ | 氷点温度 | $273.15$ | $\mathrm{K}$ | CODATA |
| $T_{\mathrm{tp}}, e_{\mathrm{tp}}$ | 水の三重点温度・飽和蒸気圧 | $273.16\ \mathrm{K},\ 611.2\ \mathrm{Pa}$ | $-$ | Bolton (1980) |
| $L_{v0}$ | $0\ ^\circ\mathrm{C}$ における蒸発潜熱 | $2.501 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{f0}$ | $0\ ^\circ\mathrm{C}$ における融解潜熱 | $3.337 \times 10^5$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $L_{s0}$ | $0\ ^\circ\mathrm{C}$ における昇華潜熱 ($L_{v0} + L_{f0}$) | $2.8347 \times 10^6$ | $\mathrm{J\ kg^{-1}}$ | Wing et al. (2018) |
| $\sigma$ | Stefan–Boltzmann 定数 | $5.670374419 \times 10^{-8}$ | $\mathrm{W\ m^{-2}\ K^{-4}}$ | CODATA |
| $\kappa$ | von Kármán 定数 | $0.40$ | $-$ | Nakanishi & Niino (2009) |
| $\rho_w, c_{pw}$ | 海水密度・定圧比熱 | $1025.0\ \mathrm{kg\ m^{-3}},\ 3990.0\ \mathrm{J\ kg^{-1}\ K^{-1}}$ | $-$ | MIROC6 (2021) |
| $T_{f,\mathrm{ocn}}$ | 海氷凍結基準温度 | $271.35$ | $\mathrm{K}$ | Semtner (1976) |
| $\rho_i, k_i, k_s$ | 海氷密度・海氷熱伝導率・積雪熱伝導率 | $917.0\ \mathrm{kg\ m^{-3}},\ 2.034,\ 0.3096\ \mathrm{W\ m^{-1}\ K^{-1}}$ | $-$ | Semtner (1976) |

### 1.2 天文学的軌道要素と太陽天頂角 (`src/aoba/radiation.py`)

Berger (1978) および Davis et al. (2017, Eqs. 2–8) に従い、経過計算時間 $t\ [\mathrm{s}]$ から通日 $N_d = (t / T_{\mathrm{day}}) \bmod D_{\mathrm{yr}}$（標準 $T_{\mathrm{day}} = 86400\ \mathrm{s},\ D_{\mathrm{yr}} = 365.2425\ \mathrm{days}$）、離心率 $e$、地軸傾斜角 $\varepsilon$、春分点基準の近日点黄経 $\tilde{\omega}$ を用いて、春分点（第80日）からの平均太陽黄経 $\lambda_m = \frac{2\pi}{D_{\mathrm{yr}}}(N_d - 80.0)$、平均近点角 $\nu_m = \lambda_m - \tilde{\omega}$、真近点角 $\nu$、および太陽の真黄経 $\lambda_s$ を計算する：
$$
\begin{aligned}
\nu &= \nu_m + \left(2e - \frac{1}{4}e^3\right)\sin\nu_m + \frac{5}{4}e^2 \sin(2\nu_m), \\
\lambda_s &= \nu + \tilde{\omega}.
\end{aligned}
$$
これより太陽赤緯 $\delta_s$、瞬時大気上端（TOA）法線入射太陽放射照度 $S_{\mathrm{inst}}(t)$、地方太陽時角 $H(t, \lambda)$、および緯度 $\varphi$・経度 $\lambda$ における太陽天頂角余弦 $\mu_0$ は次式で与えられる：
$$
\begin{aligned}
\sin\delta_s &= \sin\varepsilon \sin\lambda_s, \\
S_{\mathrm{inst}}(t) &= S_0 \left(\frac{1 + e\cos\nu}{1 - e^2}\right)^2, \\
H(t, \lambda) &= 2\pi \left[\frac{t \bmod T_{\mathrm{day}}}{T_{\mathrm{day}}} - \frac{1}{2}\right] + \lambda, \\
\mu_0(\varphi, \lambda, t) &= \max\!\left[0,\ \sin\varphi \sin\delta_s + \cos\varphi \cos\delta_s \cos H(t, \lambda)\right].
\end{aligned}
$$
時刻 $t \to t + \Delta t$ のステップ間で $H(t, \lambda) < 0 \le H(t + \Delta t, \lambda)$ かつ $\Delta H < \pi$ が成立するとき、そのコラムは地方太陽正午を通過したと判定され、植物順化用の正午下向き短波放射が抽出される。

#### 1.2.1 カレンダー管理・固定日射実験・日平均日射 (`src/aoba/calendar.py`)
`AobaCalendar` は開始年月日 `start_date`（既定 `"2000-03-20"`、春分）および暦種別 `calendar_type`（`gregorian`: 4/100/400年規則の閏年あり 365/366 日、`noleap`: 365日固定、`360_day`: 毎月30日・年360日）を管理し、経過秒数 $t$ を暦年月日と春分点基準の年割合 $f_{\mathrm{yr}} \in [0, 1)$ へ写像する（`gregorian`/`noleap` は3月20日、`360_day` は3月21日を春分基準 $f_{\mathrm{yr}} = 0$ とする）。`insolation_mode = "perpetual_date"` の場合は季節 $f_{\mathrm{yr}}$ を `fixed_insolation_date` の正午に固定し、さらに `diurnal_cycle = false` の場合は日変化を止めた解析的日平均太陽天頂角余弦を用いる：
$$
H_0(\varphi, \delta_s) = \arccos\!\left[\mathrm{clip}\!\left(-\tan\varphi \tan\delta_s,\ -1,\ 1\right)\right], \qquad \overline{\mu}_0(\varphi, \delta_s) = \max\!\left[0,\ \frac{H_0 \sin\varphi \sin\delta_s + \cos\varphi \cos\delta_s \sin H_0}{\pi}\right].
$$

---

## 2. 鉛直 Charney–Phillips 格子と平滑化地形追従座標

### 2.1 ストレッチ鉛直格子・レイリースポンジ層と Charney–Phillips 配置 (`src/aoba/vertical_grid.py`)

AOBA は地表面（$\zeta_0 = 0\ \mathrm{m}$）からモデル上端 $z_{\mathrm{top}} = 24,000\ \mathrm{m}$（既定設定、TOML `[grid]` で可変）までの気柱を $N_z = 32$ 層に分割し、鉛直 **Charney–Phillips** スタガード配置（Gassmann, 2013, Sec. 3）を採用する：
- **半整数面（層境界）** $k + \frac{1}{2}$（配列添字 $k = 0, \dots, N_z$）: 鉛直流速 $w$、物理高度 $z_w$、放射フラックス $F^\uparrow, F^\downarrow$、および湿潤熱力学の層境界鉛直フラックスに用いる境界変数。
- **整数面（層中心）** $k \in \{0, \dots, N_z - 1\}$: 乾燥空気密度 $\rho_d$、湿潤温位 $\theta_m$、気圧 $p$、気温 $T$、水平風速、乱流運動エネルギー（TKE） $q^2$、および水物質比湿 $(q_v, q_c, q_i, q_n, q_p)$。

公称層厚 $\Delta \zeta_k = \zeta_{w, k+1} - \zeta_{w, k}$ は最下層 $\Delta \zeta_0 = 60.0\ \mathrm{m}$ から幾何級数 $\Delta \zeta_k = \Delta \zeta_0 r^k$ として上端 $\sum_{k=0}^{N_z-1} \Delta \zeta_k = z_{\mathrm{top}}$ を満たすよう公比 $r$ を Newton–Raphson 法で厳密解する（既定設定 $N_z = 32, z_{\mathrm{top}} = 24\ \mathrm{km}, \Delta \zeta_0 = 60\ \mathrm{m}$ では $r \approx 1.132$ となり、高度 $2\ \mathrm{km}$ 以下に約13層を配して境界層と層積雲頂を解像する）。スポンジ基準高度 $z_{\mathrm{sp}} = 17,500\ \mathrm{m}$ 以高の上部対流圏・下部成層圏には $\sin^2$ 型のレイリー減衰プロファイル（Klemp et al., 2008; Skamarock et al., 2012）を課す：
$$
\gamma_R(\zeta) = \frac{1}{\tau_{\mathrm{sp,min}}} \sin^2\!\left[\frac{\pi}{2}\,\mathrm{clip}\!\left(\frac{\zeta - z_{\mathrm{sp}}}{z_{\mathrm{top}} - z_{\mathrm{sp}}},\, 0,\, 1\right)\right],
$$
ここで既定最小減衰時定数は $\tau_{\mathrm{sp,min}} = 300.0\ \mathrm{s}$ である。

### 2.2 Klemp (2011) 平滑化高度ベース地形追従座標 (STF) と植生基準高度

急峻な山岳上での水平気圧傾度力誤差と上空への格子波の浸透を抑えるため、Klemp (2011, Eqs. 4–7) の平滑化高度座標（STF）を用いる。下端地形高度 $z_{\mathrm{topo}}(\mathbf{x})$ から `terrain_scale_exp` $= p = 1.5$ による減衰重み：
$$
A(\zeta) = \cos^{2p}\!\left(\frac{\pi \zeta}{2 z_{\mathrm{top}}}\right)
$$
を定義し、上空層ほど大きくなる地形平滑化係数を用いて各層境界に漸進平滑地形 $h_{s, k}(\mathbf{x})$（$h_{s,0} = z_{\mathrm{topo}}$）を適用することで、物理層境界高度 $z_{w, k}(\mathbf{x})$ と層中心高度 $z_{c, k}(\mathbf{x})$ を定める：
$$
z_{w, k}(\mathbf{x}) = \zeta_{w, k} + A(\zeta_{w, k})\, h_{s, k}(\mathbf{x}), \qquad z_{c, k}(\mathbf{x}) = \frac{1}{2}\left(z_{w, k}(\mathbf{x}) + z_{w, k+1}(\mathbf{x})\right), \qquad \Delta z_{c, k} = z_{w, k+1} - z_{w, k}.
$$
なお隣接層間で $\Delta z_{c, k} \ge 0.25\,\Delta \zeta_k$ の正値単調条件を保証している。森林の樹冠高 $h_c$ 成長による大気座標の頻繁な変形を回避しつつ物理的対数風速分布の整合性を保つため、大気メッシュ下端は固体地球／氷床上面 $z_{w,0} = z_{\mathrm{topo}}$ に置き、地表面交換・粗度サブレイヤの参照高度 $\tilde{z}_{\mathrm{ref}}$ のみに有効空力高さ補正 $d_{\mathrm{eff}} \approx 0.65 h_c$ を反映する：
$$
\tilde{z}_{\mathrm{ref}} = \max\!\left(z_{c,0} - z_{w,0} - 0.65\, h_c,\ 5.0\ \mathrm{m}\right).
$$
第1層中心高度が $z_{c,0} - z_{w,0} \approx 30\ \mathrm{m}$ 以上であるため、樹高 $h_c \le 40\ \mathrm{m}$ の熱帯雨林上でも $\tilde{z}_{\mathrm{ref}}$ は正値を維持し、De Ridder (2010) RSL 補正により粗度サブレイヤ内部の交換を厳密表現できる。一方、第11章の動的氷床結合を有効化した際に固体氷床厚変化による $|z_{\mathrm{topo}}^{\mathrm{new}} - z_{w,0}| \ge 0.05\ \mathrm{m}$ が生じた場合は、`compute_stf_heights` により全層の $z_w, z_c, \Delta z_c$ を再構築して大気・CRM 鉛直列を再補間する。

---

## 3. 圧縮性湿潤熱力学と混合相ニュートン・ラフソン飽和調整

### 3.1 完全圧縮性状態方程式と水蒸気浮力密度 (`src/aoba/thermodynamics.py`)

乾燥空気密度 $\rho_d$ に対する比湿（混合比）として、水蒸気 $q_v$、雲水 $q_c$、雲氷 $q_i$、総非降水水分 $q_n = q_v + q_c + q_i$、総降水水分 $q_p = q_r + q_s + q_g$ を定義する。Skamarock et al. (2012, Eqs. 5–6) および Seidel & Yang (2024) に従い、乾燥空気圧力を水蒸気効果で拡張した修正湿潤温位 $\theta_m$、全気圧 $p$、Exner 関数 $\Pi$、気温 $T$、および総湿潤密度 $\rho$ は以下で結ばれる：
$$
\begin{aligned}
\theta_m &= \theta \left(1 + \frac{R_v}{R_d} q_v\right), \\
p &= p_0 \left(\frac{R_d \rho_d \theta_m}{p_0}\right)^{c_{pd} / c_{vd}}, \\
\Pi &= \left(\frac{p}{p_0}\right)^{R_d / c_{pd}}, \qquad T = \frac{\Pi \theta_m}{1 + (R_v / R_d) q_v}, \\
\rho &= \rho_d \left(1 + q_n + q_p\right).
\end{aligned}
$$

### 3.2 診断混合相分配と4回減衰ニュートン・ラフソン飽和調整

雲凝結物 $q_{\mathrm{cond}} = q_c + q_i$ の液相・氷相分配は、Khairoutdinov & Randall (2003, Eqs. A8–A14) に従い $T_{00n} = 253.15\ \mathrm{K}$ から $T_{0n} = 273.16\ \mathrm{K}$ の線形温度ランプ $\omega_n(T)$ で診断する：
$$
\omega_n(T) = \mathrm{clip}\!\left(\frac{T - T_{00n}}{T_{0n} - T_{00n}},\ 0,\ 1\right), \qquad q_c = \omega_n(T)\, q_{\mathrm{cond}}, \qquad q_i = \left[1 - \omega_n(T)\right] q_{\mathrm{cond}}.
$$
平面過冷却水に対する飽和水蒸気圧 $e_{sw}(T)$（Bolton, 1980, Eq. 10）と六方晶氷に対する飽和水蒸気圧 $e_{si}(T)$（Murray, 1967）を $\omega_n(T)$ で加重混合した混合飽和比湿 $q_{\mathrm{sat}}(T, p)$ を定義する：
$$
\begin{aligned}
e_{sw}(T) &= e_{\mathrm{tp}} \exp\!\left[\frac{17.67\,(T - T_0)}{T - 29.65}\right], \qquad e_{si}(T) = e_{\mathrm{tp}} \exp\!\left[\frac{21.8745584\,(T - T_0)}{T - 7.66}\right], \\
q_{\mathrm{sat}}(T, p) &= \omega_n(T) \frac{\epsilon e_{sw}(T)}{p - e_{sw}(T)} + \left[1 - \omega_n(T)\right] \frac{\epsilon e_{si}(T)}{p - e_{si}(T)}.
\end{aligned}
$$
液水・氷水等価温度 $T_{li} = T - \left(L_{v0} q_c + L_{s0} q_i\right) / c_{pd}$ を保存不変量とみなし、過飽和気塊（$q_n > q_{\mathrm{sat}}(T_{li}, p)$）に対して湿潤エンタルピー保存残差方程式を最大4回の Newton–Raphson 反復により解き、$T, q_v, q_c, q_i$ を同時確定する：
$$
f(T) = T - T_{li} - \frac{L_{\mathrm{eff}}(T)}{c_{pd}} \left[q_n - q_{\mathrm{sat}}(T, p)\right] = 0, \qquad L_{\mathrm{eff}}(T) = \omega_n(T) L_{v0} + \left[1 - \omega_n(T)\right] L_{s0}.
$$

---

## 4. 4バンド・ルックアップテーブル不要2ストリーム放射伝達

### 4.1 統一多層2ストリーム三重対角ソルバ (`src/aoba/radiation.py`, `src/aoba/kernels.py`)

放射計算は高度最上層 $n = 0$（TOA）から最下層 $n = N - 1$（地表直上）への下向き添字系で行う。各層 $n$ の光学的厚さ $\tau_n$、単一散乱アルベド $\omega_{0n}$、非対称因子 $g_n$ に対し、上向き・下向き拡散放射フラックス $F_n^\pm(\tau)$ は Toon et al. (1989, Eqs. 11–12) の一般2ストリーム方程式を満たす：
$$
\begin{aligned}
\frac{\partial F_n^+}{\partial \tau} &= \gamma_{1n} F_n^+ - \gamma_{2n} F_n^- - S_n^+(\tau), \\
\frac{\partial F_n^-}{\partial \tau} &= \gamma_{2n} F_n^+ - \gamma_{1n} F_n^- + S_n^-(\tau).
\end{aligned}
$$
- **短波2バンド (SW1, SW2)**: Joseph, Wiscombe, & Weinman (1976) の $\delta$-エディントン換算（前方散乱尖頭成分 $f = g_0^2$）：
  $$
  \tau' = \tau_0 (1 - \omega_0 f), \qquad \omega' = \frac{(1 - f)\omega_0}{1 - \omega_0 f}, \qquad g' = \frac{g_0 - f}{1 - f},
  $$
  に対して **Quadrature 法**（$\mu_1 = 1/\sqrt{3}$; Toon et al., 1989, Table 1）を適用する：
  $$
  \gamma_1 = \frac{\sqrt{3}}{2}\left[2 - \omega'(1 + g')\right], \quad \gamma_2 = \frac{\sqrt{3}}{2}\omega'(1 - g'), \quad \gamma_3 = \frac{1}{2}\left(1 - \sqrt{3} g' \mu_0\right), \quad \gamma_4 = 1 - \gamma_3,
  $$
  直達光減衰源項 $C_n^+(\tau), C_n^-(\tau)$ は層上面累積光学的厚さ $\tau_{c, n} = \sum_{m < n}\tau'_m$ を用いて与えられる：
  $$
  C_n^+(\tau) = \frac{\omega' F_{\mathrm{TOA}} \left[(\gamma_1 - \mu_0^{-1})\gamma_3 + \gamma_4 \gamma_2\right]}{\lambda_n^2 - \mu_0^{-2}} e^{-(\tau_{c,n} + \tau)/\mu_0}, \quad C_n^-(\tau) = \frac{\omega' F_{\mathrm{TOA}} \left[(\gamma_1 + \mu_0^{-1})\gamma_4 + \gamma_2 \gamma_3\right]}{\lambda_n^2 - \mu_0^{-2}} e^{-(\tau_{c,n} + \tau)/\mu_0}.
  $$
- **長波2バンド (LW1, LW2)**: 散乱無視の **Hemispheric Mean** 閉合（$\mu_1 = 1/2, \omega_0 = 0 \implies \gamma_1 = 2, \gamma_2 = 0$）を採用し、各層内の射出プランク関数 $F_B(\tau) = F_{B,0n} + F_{B,1n}\tau$（層内線形勾配 $F_{B,1n} = [F_B(T_{\mathrm{bot}, n}) - F_B(T_{\mathrm{top}, n})]/\tau_n$）に対し、Toon et al. (1989, Eqs. 26–27) の特解：
  $$
  C_n^\pm(\tau) = F_{B,0n} + F_{B,1n}\left(\tau \pm \frac{1}{\gamma_{1n} + \gamma_{2n}}\right)
  $$
  を得る。全領域で $\lambda_n = \sqrt{\gamma_{1n}^2 - \gamma_{2n}^2}$、$\Gamma_n = \gamma_{2n}/(\gamma_{1n} + \lambda_n)$、および層伝播係数 $e_{1n} = 1 + \Gamma_n e^{-\lambda_n \tau_n}, \dots, e_{4n} = \Gamma_n - e^{-\lambda_n \tau_n}$（Toon et al., 1989, Eq. 44）を定義し、層境界のフラックス連続条件・TOA 条件（$F_{\mathrm{dif}}^\downarrow(0) = 0$）および地表面境界条件（直達アルベド $\alpha_{\mathrm{dir}}$・拡散アルベド $\alpha_{\mathrm{dif}}$、射出率 $\varepsilon_{\mathrm{sfc}}$）を連立すると $2N \times 2N$ の厳密三重対角線形系：
  $$
  A_l Y_{l-1} + B_l Y_l + D_l Y_{l+1} = E_l \qquad (l = 1, \dots, 2N)
  $$
  に帰着する。これを全サブコラム一括で Metal/CUDA 融合カーネル `solve_tridiagonal_batched` により直接解く。短波ではさらに地表到達拡散フラックス $F_{\mathrm{SW}, \mathrm{dif}}^\downarrow$ と直達フラックス $F_{\mathrm{SW}, \mathrm{dir}}^\downarrow = \mu_0 F_{\mathrm{TOA}} e^{-\tau_{\mathrm{tot}}/\mu_0}$ を分離出力する。

### 4.2 4バンドの気体・雲光学的厚さと有効粒子半径パラメータ化

全気柱の層乾燥質量を $\Delta m_{d, n} = \rho_{d, n} \Delta z_n\ [\mathrm{kg\ m^{-2}}]$、雲水・雲氷パスを $\mathrm{LWP}_n = 10^3 \rho_{d, n} q_{c, n} \Delta z_n$、$\mathrm{IWP}_n = 10^3 \rho_{d, n} q_{i, n} \Delta z_n\ [\mathrm{g\ m^{-2}}]$（規定有効半径 $r_{e,\mathrm{liq}} = 10\ \mu\mathrm{m},\ r_{e,\mathrm{ice}} = 50\ \mu\mathrm{m}$）とおく。

1. **SW1（UV・可視光・PAR 帯, $0.25\text{--}0.69\ \mu\mathrm{m}$, 分割比 $w_{\mathrm{SW1}} = 0.459760$）**:
   - オゾン体積混合比 $\mathrm{O}_3$ は `o3_mode` で選択可能：`"rcemip"` は Wing et al. (2018, Eq. 1) の水平一様プロファイル：
     $$
     \mathrm{O}_3(p) = g_1\, p_{\mathrm{hPa}}^{g_2} \exp\!\left(-\frac{p_{\mathrm{hPa}}}{g_3}\right) \times 10^{-6}, \qquad (g_1, g_2, g_3) = (3.6478,\ 0.83209,\ 11.3515\ \mathrm{hPa}),
     $$
     `"latitude_seasonal"` は Brewer–Dobson 循環による高緯度増大・対流圏界面高度の緯度勾配・各半球春季極大を再現する地図不要の解析的2D緯度×季節パラメータ化（Fortuin & Kelder, 1998）、`"climatology"` は外部の月別帯状平均 CMIP6/CMIP7 input4MIPs 気候値を `(month, lat, ln p)` で補間する（モデル上端 `z_top` を成層圏・中間圏まで拡張してもテーブル上端より上で $p^{g_2}$ 減衰外挿を行い範囲外にならない）。また `o3_lw_coupling = true` 時は `LW2` 窓領域に 9.6 μm 帯吸収 $+4.5\, m_{\mathrm{O3}}$ を加算する（Ramanathan & Dickinson, 1979）。
     これに Rodgers 球面大気路程補正係数 $M(\mu_0) = 35 / \sqrt{1224\mu_0^2 + 1}$ による積算斜めオゾン路程 $x = u_{\mathrm{O3}} M(\mu_0)$ に対し、Lacis & Hansen (1974, Eqs. 8–10) のUV・Chappuis 吸収：
     $$
     A_{\mathrm{oz}}(x) = \frac{1.082\, x}{(1 + 138.6 x)^{0.805}} + \frac{0.0658\, x}{1 + (103.6 x)^3} + \frac{0.02118\, x}{1 + 0.042 x + 0.000323 x^2},
     $$
     およびレイリー散乱光学的厚さ $\Delta \tau_{\mathrm{ray}} = 0.095\, \Delta p / p_0$ を与える。
   - **雲光学特性**（Slingo, 1989, Band 1; Ebert & Curry, 1992, Band 1; Cahalan et al., 1994 平面平行アルベド補正係数 $\chi_{\mathrm{SW}} = \text{cloud\_sw\_inhomog} = 0.85$、既定有効半径 $r_{e,l} = 10.0\ \mu\mathrm{m},\ r_{e,i} = 48.0\ \mu\mathrm{m}$）：
     $$
     \begin{aligned}
     \tau_{l1} &= \chi_{\mathrm{SW}}\,\mathrm{LWP}\left(2.817\times 10^{-2} + \frac{1.305}{r_{e,l}}\right), & \omega_{l1} &= 1 - (-5.62\times 10^{-8} + 1.63\times 10^{-7} r_{e,l}), & g_{l1} &= 0.829 + 2.482\times 10^{-3} r_{e,l}, \\
     \tau_{i1} &= \chi_{\mathrm{SW}}\,\mathrm{IWP}\left(3.448\times 10^{-3} + \frac{2.431}{r_{e,i}}\right), & \omega_{i1} &= 1 - 10^{-5}, & g_{i1} &= 0.7661 + 5.851\times 10^{-4} r_{e,i}.
     \end{aligned}
     $$
   - 地表下向き SW1 フラックスから生態系へ渡す光合成有効光量子束密度 $\mathrm{PPFD}$ は $\mathrm{PPFD} = 4.57 \times 0.88\, F_{\mathrm{SW1}}^\downarrow(z_{\mathrm{sfc}})\ [\mu\mathrm{mol\ m^{-2}\ s^{-1}}]$ で算出する。

2. **SW2（近赤外線帯, $0.69\text{--}4.00\ \mu\mathrm{m}$, 分割比 $w_{\mathrm{SW2}} = 0.540240$）**:
   - 気圧温度換算可降水量増分 $\Delta y = 0.1\, q_v \Delta m_d (p / p_0)\sqrt{T_0 / T}\ [\mathrm{cm}]$ の積算から、Lacis & Hansen (1974, Eq. 21) の太陽放射吸収率：
     $$
     A_{\mathrm{wv}}(y / \mu_0) = \frac{2.9\, (y / \mu_0)}{\left[1 + 141.5\, (y / \mu_0)\right]^{0.635} + 5.925\, (y / \mu_0)}
     $$
     により各層の吸収光学的厚さを抽出し、Slingo (1989, Bands 2–4 平均)・Ebert & Curry (1992, Bands 2–5 平均) に Cahalan et al. (1994) 非一様係数 $\chi_{\mathrm{SW}}$ を乗じて合成する：
     $$
     \begin{aligned}
     \tau_{l2} &= \chi_{\mathrm{SW}}\,\mathrm{LWP}\left(2.456\times 10^{-2} + \frac{1.400}{r_{e,l}}\right), & \omega_{l2} &= 1 - (0.01260 + 9.01\times 10^{-4} r_{e,l}), & g_{l2} &= 0.783 + 5.01\times 10^{-3} r_{e,l}, \\
     \tau_{i2} &= \chi_{\mathrm{SW}}\,\mathrm{IWP}\left(3.448\times 10^{-3} + \frac{2.431}{r_{e,i}}\right), & \omega_{i2} &= 1 - (0.02437 + 2.52\times 10^{-4} r_{e,i}), & g_{i2} &= 0.7867 + 5.90\times 10^{-4} r_{e,i}.
     \end{aligned}
     $$

3. **LW1（$\mathrm{H}_2\mathrm{O} + \mathrm{CO}_2$ 回転振動吸収帯, Planck 関数重み $1 - \beta_{\mathrm{win}} = 0.680$）**:
   - 気圧スケーリング済み水蒸気線吸収、対数 $15\ \mu\mathrm{m}\ \mathrm{CO}_2$ 吸収（Weaver & Ramanathan, 1995; Koll & Cronin, 2018; Sherwood et al., 2020）、背景混合気体吸収、および Stephens (1978)・Ebert & Curry (1992, Eq. 12 & Table 3) の雲吸収（半球平均拡散因子 $2.0$ と Ebert–Curry $\beta_{\mathrm{diff}}=1.66$ の整合係数 $0.83$ を乗じる）の和をとる：
     $$
     \begin{aligned}
     \Delta \tau_{\mathrm{LW1}} &= 0.30 \left(\frac{p}{p_0}\right)^{0.55} q_v \Delta m_d + \max\!\left[0.02,\ 0.85 + 0.22 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} + 0.020 \frac{\Delta p}{p_0} \\
     &\quad + 0.83 \left[0.090\, \mathrm{LWP} + \left(0.00324 + \frac{1.033}{r_{e,i}}\right)\mathrm{IWP}\right].
     \end{aligned}
     $$

4. **LW2（$8.0\text{--}12.5\ \mu\mathrm{m}$ 大気の窓領域, Planck 関数重み $\beta_{\mathrm{win}} = 0.320$）**:
   - Roberts et al. (1976) の水蒸気自己連続吸収、衝突連続吸収、微弱 $\mathrm{CO}_2$ 窓重なり（$0.024\ln(c_a/c_{a,0})$）、および窓域雲吸収の和で与える：
     $$
     \begin{aligned}
     \Delta \tau_{\mathrm{LW2}} &= \left[0.45 \frac{e_v}{p_0} \exp\!\left(\frac{1800}{T} - \frac{1800}{296}\right) + 0.0025 \frac{p}{p_0}\right] q_v \Delta m_d + \max\!\left[0,\ 0.015 + 0.024 \ln\!\left(\frac{c_a}{c_{a,0}}\right)\right] \frac{\Delta(p^2)}{p_0^2} \\
     &\quad + 0.83 \left[0.090\, \mathrm{LWP} + \left(0.00160 + \frac{1.166}{r_{e,i}}\right)\mathrm{IWP}\right].
     \end{aligned}
     $$

### 4.3 エネルギー保存型の放射加熱率

層境界 $k + \frac{1}{2}$ における正味上向きフラックスを $F_{\mathrm{net}, k+1/2} = (F_{\mathrm{SW}, k+1/2}^\uparrow - F_{\mathrm{SW}, k+1/2}^\downarrow) + (F_{\mathrm{LW}, k+1/2}^\uparrow - F_{\mathrm{LW}, k+1/2}^\downarrow)$ とおくと、各層 $k$ の放射加熱率は：
$$
\left(\frac{\partial T}{\partial t}\right)_{\mathrm{rad}, k} = -\frac{F_{\mathrm{net}, k+1/2} - F_{\mathrm{net}, k-1/2}}{(c_{pd} + c_{pv} q_{v,k}) \rho_{d,k} \Delta z_k}
$$
となり、大気上端放射収支（$\mathrm{ASR} - \mathrm{OLR}$）と地表面正味放射・気柱内部エネルギー変化の間の放射エネルギー保存が数値的に厳密に成立する。

### 4.4 16バンド（50 g点）決定論的 Correlated-k 放射伝達 (`rad_scheme = "ckd"`, `src/aoba/radiation_ckd.py`)

AOBA は既定で完全自己完結の **16バンド（50 g点）決定論的 Correlated-k（$k$-分布）放射伝達ソルバ**（Goody et al., 1989; Lacis & Oinas, 1991; Fu & Liou, 1992; Clough & Iacono, 1995; Mlawer et al., 1997, 2012, 2023; Etminan et al., 2016; Pincus et al., 2020）を採用する（旧4バンド式も `rad_scheme = "four_band"` で選択可能）：
- **長波 10 バンド（32 g点、伸縮境界 $0 \to 350 \to 500 \to 630 \to 700 \to 820 \to 980 \to 1080 \to 1180 \to 1400 \to \infty\ \mathrm{cm^{-1}}$）**: 各バンドのプランク関数重み $B_b(T)$ を Widger & Woodall (1976) の6項級数で解析評価（$\sum_{b=0}^{9} B_b(T) \equiv \sigma T^4$ を厳密保存）。各 g 点で MT\_CKD 水蒸気自己／衝突連続吸収、圧力・Voigt 広がり付き correlated-$k$ 吸収係数 $k_{b,g}(p,T) = k_{0,b,g}[(p+p_{\mathrm{voigt}})/p_0]^{\gamma}\exp[\beta(T_{\mathrm{ref}}^{-1}-T^{-1})]$（$\mathrm{H_2O}$・$\mathrm{CO_2}$ 15 μm 中心／翼部・10 μm ホットバンド・4.3 μm・$\mathrm{O_3}$ 9.6 / 14.3 μm・$\mathrm{CH_4}$ 7.66 μm・$\mathrm{N_2O}$ 7.78 / 8.6 / 17 μm、高空 $p < 80\ \mathrm{hPa}$ の光化学減衰 $(p/80\ \mathrm{hPa})^{0.38}$ を含む）、バンド別液水雲（Hu & Stamnes, 1993）・氷雲（Ebert & Curry, 1992, Table 3）・鉱物ダスト 9.5 μm 珪酸塩共鳴吸収（Balkanski et al., 2007）を拡散因子 $D = 1.66$ の線形 $\tau$ 法で積分する。
- **短波 6 バンド（18 g点、$0.20\text{--}4.00\ \mu\mathrm{m}$）**: $\mathrm{O_3}$ Hartley / Huggins / Chappuis 吸収（モデル上端より上のオーバーヘッドオゾン減衰を含む）、$\lambda^{-4}$ レイリー散乱、$\mathrm{O_2}$ A/B バンド、$\mathrm{H_2O}$ 近赤外帯、さらに $\mathrm{CO_2}+\mathrm{CH_4}$ 近赤外太陽吸収（Etminan et al., 2016 の短波メタン強制を再現）を、3D 診断有効半径 $(r_{e,\mathrm{liq}}, r_{e,\mathrm{ice}})$ を直接受け取る Slingo (1989) 液水雲・Ebert & Curry (1992) 氷雲光学と結合して Toon $\delta$-Quadrature 法で解く。AFGL 4気団プロファイルにおける晴天 $2\times\mathrm{CO_2}$ FDH 成層圏調整強制力（SARF）は $\approx 4.7\ \mathrm{W\ m^{-2}}$（全天換算 $\approx 3.9\ \mathrm{W\ m^{-2}}$）、$4\times\mathrm{CO_2}/2\times\mathrm{CO_2} \approx 2.09$、10 hPa 成層圏寒冷化は $\approx -7\ \mathrm{K}$ を再現する。

---

## 5. 混合相1モーメント雲微物理・降水落下

### 5.1 予報2変数と降水粒子種の診断分配 (`src/aoba/microphysics.py`)

SAM (Khairoutdinov & Randall, 2003, Appendices A–B) に基づき、予報湿潤変数を非降水総水分 $q_n = q_v + q_c + q_i$ と降水総水分 $q_p = q_r + q_s + q_g$ の2本に集約する。$q_p$ の雨・雪・あられへの診断分配は温度ランプ $\omega_p(T)$（範囲 $[T_{00p}, T_{0p}] = [268.16, 283.16]\ \mathrm{K}$）および $\omega_g(T)$（範囲 $[T_{00g}, T_{0g}] = [223.16, 283.16]\ \mathrm{K}$）を用いて行う：
$$
\begin{aligned}
\omega_p(T) &= \mathrm{clip}\!\left(\frac{T - T_{00p}}{T_{0p} - T_{00p}},\ 0,\ 1\right), \qquad \omega_g(T) = \mathrm{clip}\!\left(\frac{T - T_{00g}}{T_{0g} - T_{00g}},\ 0,\ 1\right), \\
q_r &= \omega_p q_p, \qquad q_s = (1 - \omega_p)(1 - \omega_g) q_p, \qquad q_g = (1 - \omega_p)\omega_g q_p.
\end{aligned}
$$

### 5.2 粒子粒径分布・自動変換・捕捉成長・昇華蒸発・落下沈降

降水粒子種 $x \in \{r, s, g\}$ は指数型 Marshall–Palmer 粒径分布 $N_x(D) = N_{0x} \exp(-\lambda_x D)$（切片 $(N_{0r}, N_{0s}, N_{0g}) = (8\times 10^6, 3\times 10^6, 4\times 10^6)\ \mathrm{m^{-4}}$、密度 $(\rho_r, \rho_s, \rho_g) = (1000, 100, 400)\ \mathrm{kg\ m^{-3}}$）とべき則終端速度 $V_x(D) = a_x D^{b_x} (\rho_0 / \rho_d)^{1/2}$（$(a_r, b_r)=(842, 0.8),\ (a_s, b_s)=(4.84, 0.25),\ (a_g, b_g)=(94.5, 0.5)$）に従う。傾きパラメータ $\lambda_x$ は質量から：
$$
\lambda_x = \left(\frac{\pi \rho_x N_{0x}}{\rho_d q_x}\right)^{1/4}
$$
と与えられる。Khairoutdinov & Randall (2003, Eqs. A18–A33) の各素過程は以下の閉形式で定式化される：
1. **自動変換 (Autoconversion)**（暖雨過程は Berry, 1968 / Tomita, 2008; SCALE-RM Eqs. 8.221–8.222, 雲粒数濃度 $N_d = 60\ \mathrm{cm^{-3}}$, $D_d = \max[0.146 - 5.964\times 10^{-2}\ln(N_d / 2000),\, 0.05]$、$N_d \le 0$ 時は KR03 閾値式 $\alpha_c \max(q_c - q_{c0}, 0)$ へフォールバック；氷晶凝集は $q_{i0} = 10^{-4}\ \mathrm{kg\ kg^{-1}}, \beta_i = 10^{-3}\ \mathrm{s^{-1}}, V_{TI} = 0.40\ \mathrm{m\ s^{-1}}$）：
   $$
   \left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{auto}} = \frac{16.7\, D_d\, \rho_d^2\, q_c^3}{5.0\, D_d\, \rho_d\, q_c + 3.6\times 10^{-5} N_d} + \beta_i \exp\!\left[0.025(T - T_0)\right] \max(q_i - q_{i0},\ 0).
   $$
2. **捕捉成長 (Accretion)**（捕捉効率 $E_{rc}=E_{sc}=E_{gc}=E_{ri}=1.0,\ E_{si}=E_{gi}=0.1$）：
   $$
   \left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{accr}} = \sum_{x \in \{r, s, g\}} \sum_{y \in \{c, i\}} \frac{\pi a_x N_{0x} \Gamma(3 + b_x)}{4} E_{xy} \left(\frac{\rho_0}{\rho_d}\right)^{1/2} \left(\frac{\rho_d q_x}{\pi \rho_x N_{0x}}\right)^{(3 + b_x)/4} q_y.
   $$
3. **未飽和域における降水の蒸発・昇華**：雲非存在時または蒸発可能域で $S = q_v / q_{\mathrm{sat}} < 1$ のとき、ベンチレーション係数 $(a_{vx}, b_{vx})$ と水蒸気拡散係数 $D_v$・空気熱伝導係数 $K_a$ による熱・物質輸送インピーダンスを組み合わせて評価する：
   $$
   \left(\frac{\partial q_p}{\partial t}\right)_{\mathrm{evap}} = \sum_{x \in \{r, s, g\}} \frac{2\pi N_{0x} (S - 1) \left[a_{vx}\lambda_x^{-2} + b_{vx} \mathrm{Sc}^{1/3}\sqrt{\frac{a_x}{\nu_a}\left(\frac{\rho_0}{\rho_d}\right)^{1/2}}\Gamma\!\left(\frac{5 + b_x}{2}\right)\lambda_x^{-(5 + b_x)/2}\right]}{\rho_d \left(\frac{L_{\mathrm{eff}}^2}{K_a R_v T^2} + \frac{R_v T}{D_v e_{\mathrm{sat}}(T)}\right)} \le 0.
   $$
4. **質量重み終端速度と CFL サブサイクリング風上落下**：
   $$
   V_x = \frac{a_x \Gamma(4 + b_x)}{6 \lambda_x^{b_x}} \left(\frac{\rho_0}{\rho_d}\right)^{1/2}, \qquad V_p = \omega_p V_r + (1 - \omega_p)\left[(1 - \omega_g) V_s + \omega_g V_g\right].
   $$
   （加えて微物理中の雲氷自重落下を $V_{t,i} = 0.4\ \mathrm{m\ s^{-1}}$ として考慮可）。鉛直クーラン数 $\max(V_p \Delta t_{\mathrm{sed}} / \Delta z_k) \le 0.8$ を満たすようステップ内サブサイクル化し、1次精度風上法で地上降水フラックス（降雨・降雪）を算出する。相変化による湿潤温位変化は Nishizawa et al. (2025, Eq. 2.64) に従い保存的に更新する：
   $$
   \Delta \theta_m = \frac{1 + (R_v / R_d) q_v}{\Pi c_{pd}} \left[\Delta(L_{\mathrm{eff}, n} q_{\mathrm{cond}}) + \Delta(L_{\mathrm{eff}, p} q_p)\right].
   $$

### 5.3 2モーメント雲微物理と予報有効半径 (`microphysics_scheme = "two_moment"`, `src/aoba/microphysics.py`)

`microphysics_scheme = "two_moment"`（既定値。`"one_moment"` で従来の KR03 1モーメントを保持）のとき、AOBA は雲粒・雲氷の比数濃度 $(n_c, n_i)\ [\mathrm{kg^{-1}}]$ を $(q_n, q_p)$ と共に融合 SIMD FCT ペアカーネルで移流しながら予報し、Metal カーネル `crm_microphysics_2m_col` で微物理過程を解く：
- **マップフリー自然 CCN・INP 活性化**（Twomey, 1959; O'Dowd et al., 1997; Andreae, 2009; Cooper, 1986; DeMott et al., 2010; Field et al., 2017）：産業革命前の清浄な海洋 CCN は海上風速依存の海塩エアロゾル（$N_{\mathrm{ccn,ocn}} = 45\exp(0.055 u_{10})\ \mathrm{cm^{-3}}$）、陸域 CCN は自然生物起源基準（$95\ \mathrm{cm^{-3}}$）＋予報鉱物ダストから評価し、解像上昇流とサブグリッド TKE を合わせた実効上昇流 $w_{\mathrm{eff}} = \max(w, 0) + 0.7\sqrt{q^2/2}$ で雲粒を活性化する。氷晶核生成は Cooper (1986) 温度則・DeMott et al. (2010) ダスト浸漬氷晶核増強・混相雲二次氷晶下限（$3\times 10^4\ \mathrm{m^{-3}}$）・$235\ \mathrm{K}$ 未満の均質凍結を組み合わせる。
- **2モーメント自動変換・捕捉消滅と放射光学結合**（Khairoutdinov & Kogan, 2000, Eq. 29; Morrison et al., 2005; Martin et al., 1994; Wyser, 1998）：暖雨自動変換は $(\partial_t q_r)_{\mathrm{auto}} = C_{\mathrm{KK}}\, q_c^{2.47} N_{c,\mathrm{cm}^{-3}}^{-1.79}$（4 km サブグリッド LWC 分散補正 $C_{\mathrm{KK}} = 42000$）に従い、変換・捕捉に比例して $(n_c, n_i)$ を減少させ、粒径依存落下速度 $V_{t,i}(r_{e,i})$ で雲氷数濃度を沈降させる。予報 3D 有効半径 $r_{e,\mathrm{liq}} = [3 q_c / (4\pi \rho_w \kappa_{\mathrm{disp}} n_c)]^{1/3} \in [4, 25]\ \mu\mathrm{m}$（$\kappa_{\mathrm{disp}} = 0.72$）および $r_{e,\mathrm{ice}} = 0.85[3 q_i / (4\pi \rho_{ci} n_i)]^{1/3} \in [15, 120]\ \mu\mathrm{m}$ を Metal `rad_ckd_col` 放射ソルバへ直接結合する。

---

## 6. 粗度サブレイヤ (RSL)・陰的 MYNN Level 2.5 PBL・3D SGS 乱流

### 6.1 植生・裸地・海洋の空力粗度パラメータ化 (`src/aoba/turbulence.py`)

1. **陸面・植生（Raupach, 1994, Eqs. 8, 10–11）**: PFT 表を用いず、雪埋没補正後の露出樹冠高 $h_{\mathrm{exp}}$ および露出葉面積指数 $L_{\mathrm{exp}}$ からゼロ面変位 $d$ と運動量粗度長 $z_{0m}$ を直接計算する：
   $$
   \frac{d}{h_{\mathrm{exp}}} = 1 - \frac{1 - \exp\!\left(-\sqrt{c_{d1} L_{\mathrm{exp}}}\right)}{\sqrt{c_{d1} L_{\mathrm{exp}}}}, \qquad \frac{z_{0m}}{h_{\mathrm{exp}}} = \left(1 - \frac{d}{h_{\mathrm{exp}}}\right)\exp\!\left(-\kappa \gamma_u + \psi_{h,\mathrm{cw}}\right),
   $$
   ここで $\gamma_u = \min\!\left[(C_S + C_R L_{\mathrm{exp}}/2)^{-1/2},\ \gamma_{\max}\right]$、$c_{d1} = 7.5,\ C_S = 0.003,\ C_R = 0.30,\ \psi_{h,\mathrm{cw}} = 0.193,\ \gamma_{\max} = 3.33$ である。裸地 ($L_{\mathrm{exp}} \to 0$) では $z_{0m,\mathrm{soil}} = 0.01\ \mathrm{m}$、積雪面では $z_{0m,\mathrm{snow}} = 0.002\ \mathrm{m}$ へ連続移行し、熱・水蒸気粗度長はスタントン数関係式 $z_{0h} = z_{0q} = e^{-2} z_{0m} \approx 0.135335\, z_{0m}$ とおく。
2. **開水面（Miller, Beljaars, & Palmer, 1992, Eqs. 3–5; SCALE-RM Eqs. 8.360–8.362）**: 弱風時の滑面粘性項（$\nu_a = 1.5\times 10^{-5}\ \mathrm{m^2\ s^{-1}}$）と強風時の Charnock 波浪項（$\alpha_{\mathrm{Ch}} = 0.018$）を結合する：
   $$
   z_{0m} = 0.11\frac{\nu_a}{u_*} + \frac{\alpha_{\mathrm{Ch}} u_*^2}{g}, \qquad z_{0h} = 0.40\frac{\nu_a}{u_*} + 1.4\times 10^{-5}, \qquad z_{0q} = 0.62\frac{\nu_a}{u_*} + 1.3\times 10^{-4}.
   $$

### 6.2 De Ridder (2010) 閉形式粗度サブレイヤ (RSL) 付き Monin–Obukhov 相似則

有効参照高度 $z = \tilde{z}_{\mathrm{ref}}$（実効風速 $U_{\mathrm{eff}} = \sqrt{u_1^2 + v_1^2 + 0.25}$）におけるバルク・リチャードソン数 $\mathrm{Ri}_B = \frac{g z (\theta_{v1} - \theta_{v,\mathrm{sfc}})}{\bar{\theta}_v U_{\mathrm{eff}}^2}$ から、安定度パラメータ $\zeta = z / L_{\mathrm{MO}}$ を反復求解する。不安定域 ($\zeta < 0$) では Paulson–Businger–Dyer 関数、安定域 ($\zeta \ge 0$) では Beljaars & Holtslag (1991) の積分形 $\psi_{M,H}(\zeta)$ を用いる。樹冠上の粗度サブレイヤ厚 $z_* = \max(2h_{\mathrm{exp}} - d,\ 0)$ が存在する陸面タイルでは、無次元比 $\chi = z / z_*,\ \chi_{0\alpha} = z_{0\alpha}/z_*$ に対して De Ridder (2010, Eqs. 6–7, 22; Bonan et al., 2018; Lee et al., 2020) の指数積分修正項 $\hat{\psi}_\alpha(\zeta, \chi)$ を差し引く（$\mu_M = 2.59,\ \mu_H = 0.95,\ \nu_{\mathrm{RSL}} = 0.50,\ \lambda_{\mathrm{RSL}} = 1.50$）：
$$
\begin{aligned}
F_\alpha &= \ln\!\left(\frac{z}{z_{0\alpha}}\right) - \psi_\alpha(\zeta) + \psi_\alpha\!\left(\zeta \frac{z_{0\alpha}}{z}\right) - \left[\hat{\psi}_\alpha\!\left(\zeta \frac{z_{0\alpha}}{z}, \chi_{0\alpha}\right) - \hat{\psi}_\alpha(\zeta, \chi)\right], \qquad \alpha \in \{M, H, Q\}, \\
\hat{\psi}_\alpha(\zeta, \eta) &= \phi_\alpha\!\left(\zeta \frac{1 + \nu_{\mathrm{RSL}}/\mu_\alpha}{\eta}\right) \left(1 - \frac{1}{c_{2\alpha}}\right) \exp(-\mu_\alpha \eta)\left[\ln(\mu_\alpha \eta) + E_1(\mu_\alpha \eta)\right],
\end{aligned}
$$
ここで $E_1(x) = \int_x^\infty t^{-1} e^{-t} dt$ は指数積分である。収束した $\zeta$ から抵抗・交換係数 $C_M = \kappa^2 / F_M^2,\ C_H = \kappa^2 / (F_M F_H),\ C_Q = \kappa^2 / (F_M F_Q)$ および摩擦速度 $u_* = \sqrt{C_M}\, U_{\mathrm{eff}}$ を得る。

### 6.3 陰的 MYNN Level 2.5 境界層乱流クロージャ

倍増乱流運動エネルギー $q^2 = 2e$（Nakanishi & Niino, 2009; Nishizawa et al., 2025, Sec. 8.2.1 Eqs. 8.96–8.176）を予報する。液水・氷水保存温位 $\theta_l = \theta - (L_{v0} q_c + L_{s0} q_i)/(c_{pd}\Pi)$ と総非降水水分 $q_n$ に対し、境界層高度積分で定める乱流長 $L_T$・接地長 $L_S$・浮力長 $L_B$ からマスター混合長 $L$ を構成する：
$$
\frac{1}{L} = \frac{1}{L_S} + \frac{1}{L_T} + \frac{1}{L_B}, \qquad L_T = 0.23 \frac{\int_0^\infty q \tilde{z}\, dz}{\int_0^\infty q\, dz}, \qquad L_B = \begin{cases} \frac{q}{N} \left(1 + 5\sqrt{\frac{q_{\mathrm{conv}}}{L_T N}}\right) & (\zeta < 0,\ N^2 > 0), \\ q / N & (\zeta \ge 0,\ N^2 > 0). \end{cases}
$$
サブグリッドガウス凝結分散 $\sigma_s = \frac{1}{2} a L \sqrt{B_2 / 2} \left|\frac{\partial q_n}{\partial z} - b \frac{\partial \theta_l}{\partial z}\right|$ と規格化過剰水蒸気 $Q_1 = a(q_n - q_{\mathrm{sat}})/(2\sigma_s)$ から部分雲量 $R = \frac{1}{2}[1 + \mathrm{erf}(Q_1/\sqrt{2})]$・部分凝結水 $Q_l$・実効雲量 $\tilde{R}$ を求め、湿潤浮力係数 $\beta_\theta, \beta_q$ を介して湿潤ブラント・バイサラ振動数 $N^2 = \frac{g}{\theta}\left(\beta_\theta \frac{\partial \theta_l}{\partial z} + \beta_q \frac{\partial q_n}{\partial z}\right)$ と勾配リチャードソン数 $\mathrm{Ri} = N^2 / S^2$ を評価する。次いで Level 2 平衡定数（$B_1 = 24.0,\ B_2 = 15.0,\ \gamma_1 = 0.235,\ \mathrm{Pr}_N = 0.74,\ C_2 = 0.75,\ C_3 = 0.352,\ C_5 = 0.2$）から平衡フラックスリチャードソン数 $R_f$ の2次方程式・平衡値 $q_{\mathrm{L2}}^2 = B_1 L^2 S_{M2}(1 - R_f) S^2$・非平衡補正係数 $\alpha_c = \min(q / q_{\mathrm{L2}},\ 1)$ を求め、無次元次元数 $G_M = (L^2 / q^2) S^2,\ G_H = -(L^2 / q^2) N^2$ から Level 2.5 安定度関数 $(S_M, S_H)$ と渦拡散係数 $(K_M, K_H, K_q) = (L q S_M,\ L q S_H,\ 3 L q S_M)$ を確定する：
$$
\begin{aligned}
S_M &= \alpha_c A_1 \frac{\phi_3 - 3 C_1 \phi_4}{\phi_2 \phi_4 + \phi_5 \phi_3}, \qquad S_H = \alpha_c A_2 \frac{\phi_2 + 3 C_1 \phi_5}{\phi_2 \phi_4 + \phi_5 \phi_3}, \\
\frac{\partial (\rho_d q^2)}{\partial t} &= \frac{\partial}{\partial z}\!\left(\rho_d K_q \frac{\partial q^2}{\partial z}\right) + 2 \rho_d \left(K_M S^2 - K_H N^2 - \frac{q^3}{B_1 L}\right).
\end{aligned}
$$
$q^2$ 方程式および $(u, v, \theta_m, q_n)$ の鉛直乱流拡散方程式は、最下層地表フラックスを下部境界源項として、陰的バッチ Thomas ソルバ `solve_tridiagonal_batched` により無条件安定に同時積分する。

### 6.4 3D 非等方 Smagorinsky–Lilly SGS クロージャ (CRM 用)

3D CRM の水平・非等方 SGS 乱流混合は、変形速度テンソル大きさ $|\mathbf{S}| = \sqrt{2 S_{ij} S_{ij}}$ とアスペクト比 $a = \max(\Delta x, \Delta y) / \Delta z_k$ に対する Scotti, Meneveau, & Lilly (1993) 補正 $f(a)$ を含む Richardson 数依存 Smagorinsky–Lilly 式（Lilly, 1962; Nishizawa et al., 2025, Sec. 8.1.2）で表す（$C_s = 0.18,\ \mathrm{Ri}_c = 0.25,\ \mathrm{Pr}_{\mathrm{SGS}}^{-1} = 3$）：
$$
\nu_{\mathrm{SGS}} = (C_s f(a) \Delta)^2 |\mathbf{S}| \sqrt{\max\!\left(0,\ 1 - \frac{\mathrm{Ri}}{\mathrm{Ri}_c}\right)}, \qquad \Delta = (\Delta x \Delta y \Delta z_k)^{1/3}, \qquad D_{\mathrm{SGS}} = \mathrm{Pr}_{\mathrm{SGS}}^{-1}\, \nu_{\mathrm{SGS}}.
$$

---

## 7. Two-Leaf 非直角双曲線 EEO 自律生態系

### 7.1 最適気孔比・非直角双曲線光合成配位・普遍的暗呼吸 (`src/aoba/biosphere.py`)

AOBA は PFT 分類・経験的パラメータ表を一切使わず、植物の資源獲得コスト最小化（EEO）のみから C₃・C₄ 双方の光合成能力・葉暗呼吸・気孔開度を自律決定する。

1. **C₃ 光合成最適性 (Prentice et al., 2014; Smith et al., 2019, Eqs. 1–20)**:
   カルボキシル化酵素維持コストと通水コスト比 $\beta_{\mathrm{C3}} = 146$、気温依存の相対水粘性率 $\eta^*(T_g)$、CO₂補償点 $\Gamma^*$、見かけのミカエリス定数 $K_M = K_c(1 + O_i / K_o)$ から、細胞間隙対大気 CO₂ 比 $\chi_{\mathrm{C3}} = c_i / c_a$ とその限界値 $m, m_c$ を決める：
   $$
   \chi_{\mathrm{C3}} = \frac{\Gamma^*}{c_a} + \left(1 - \frac{\Gamma^*}{c_a}\right)\frac{\xi_{\mathrm{C3}}}{\xi_{\mathrm{C3}} + \sqrt{D}}, \qquad \xi_{\mathrm{C3}} = \sqrt{\frac{\beta_{\mathrm{C3}}(K_M + \Gamma^*)}{1.6 \eta^*(T_g)}}, \qquad m = \frac{c_i - \Gamma^*}{c_i + 2\Gamma^*}, \qquad m_c = \frac{c_i - \Gamma^*}{c_i + K_M}.
   $$
   Farquhar & Wong (1984) の非直角双曲線電子伝達速度（曲率 $\theta = 0.85$）の下で、限界電子伝達コスト最適条件 $\partial A_j / \partial J_{\max} = c$（$c = 0.05336251$）と生化学的協調条件 $A_c = A_j$ を連立解くと、$m > 4c$ に対して厳密な解析分岐 $\omega, \omega^*$ が得られる：
   $$
   \begin{aligned}
   \omega &= -(1 - 2\theta) + \mathrm{sgn}\!\left(1 - \frac{8\theta c}{m}\right)\sqrt{(1 - \theta)\left[\frac{1}{(4c/m)(1 - \theta\cdot 4c/m)} - 4\theta\right]}, \\
   \omega^* &= 1 + \omega - \sqrt{(1 + \omega)^2 - 4\theta\omega}, \\
   J_{\max,\mathrm{opt}} &= \varphi_{\mathrm{PSII}}(T_g)\, \tilde{I}_c\, \omega, \qquad V_{c\max,\mathrm{opt}} = \varphi_{\mathrm{PSII}}(T_g)\, \tilde{I}_c\, \frac{m}{m_c}\frac{\omega^*}{8\theta}.
   \end{aligned}
   $$
   ここで $\varphi_{\mathrm{PSII}}(T_g)$ は Bernacchi et al. (2003) / Schneider et al. (2025, Eq. 12) に基づく生育温度順化量子収率、$f_P(T, T_g, T_{\mathrm{home}})$ は Kumarathunge et al. (2019) / Schneider, Gessler, & Stocker (2025, Table 1 & Eq. 7) に基づく15日生育温度 $T_{g,C} = \mathrm{clip}(T_g - 273.15,\ 0,\ 40)\ [^\circ\mathrm{C}]$ とマップ不要の多年（約3年）気候適応温度 $T_{\mathrm{home},C}$ を分離したピーク型アレニウス関数（$H_d = 200{,}000\ \mathrm{J\ mol^{-1}},\ T_0 = 298.15\ \mathrm{K}$）、$\eta^*(T)$ は水の相対粘性率である。サブデイリー計算では気孔感度 $\xi'_{\mathrm{C3}}(T_g)$ および量子収率 $\varphi_{\mathrm{PSII}}(T_g)$ を15日順化値で日内一定に保ち、瞬時の葉温・VPD 応答と分離する（Schneider et al., 2025, Eq. 11; Gan et al., 2026, Eq. 18）：
   $$
   \begin{aligned}
   \varphi_{\mathrm{PSII}}(T_{g,C}) &= 0.257 \cdot \frac{\max\!\left(0.352 + 0.022 T_{g,C} - 3.4\times 10^{-4} T_{g,C}^2,\ 0.02\right)}{0.656}, \\
   f_P(T, T_g, T_{\mathrm{home}}) &= \exp\!\left[\frac{H_a (T - T_0)}{R_{\mathrm{gas}} T_0 T}\right]\frac{1 + \exp\!\left(\dfrac{T_0 \Delta S - H_d}{R_{\mathrm{gas}} T_0}\right)}{1 + \exp\!\left(\dfrac{T \Delta S - H_d}{R_{\mathrm{gas}} T}\right)}, \quad \begin{cases} H_{a,V} = 42{,}600 + 1{,}140\,T_{g,C},\ \Delta S_V = 645.13 - 0.38\, T_{g,C}, \\ H_{a,J} = 40{,}710,\ \Delta S_J = 658.77 - 0.84\, T_{\mathrm{home},C} - 0.52\,(T_{g,C} - T_{\mathrm{home},C}), \end{cases} \\
   \eta^*(T) &= \exp\!\left[\frac{580}{T - 138} - \frac{580}{T_0 - 138}\right], \qquad \xi'_{\mathrm{C3}} = \sqrt{\frac{\beta_{\mathrm{C3}}(K_m(T_g) + \Gamma^*(T_g))}{1.6\,\eta^*(T_g)}}, \qquad c_i = \frac{\xi'_{\mathrm{C3}}\, c_s + \Gamma^*(T_{\mathrm{leaf}})\sqrt{D_s}}{\xi'_{\mathrm{C3}} + \sqrt{D_s}}.
   \end{aligned}
   $$
   また $25\ ^\circ\mathrm{C}$ 基準容量は $V_{c\max 25} = V_{c\max,\mathrm{opt}} / \max(f_P(T_g, T_g, H_{a,V}),\, 0.05)$ および $J_{\max 25} = J_{\max,\mathrm{opt}} / \max(f_P(T_g, T_g, H_{a,J}),\, 0.05)$ で換算される。

2. **C₄ 三酵素協調最適性 (Scott & Smith, 2022, Eqs. 1–25; von Caemmerer, 2000)**:
   葉肉 CO₂ ドローダウン比 $\chi_{m,\mathrm{C4}} = \xi_{\mathrm{C4}} / (\xi_{\mathrm{C4}} + \sqrt{D})$（$\xi_{\mathrm{C4}} = \sqrt{\beta_{\mathrm{C4}} K_p / (1.6 \eta^*)},\ \beta_{\mathrm{C4}} = 166$）、維管束鞘漏出率 $\phi_L = 0.20$、維管束鞘コンダクタンス $g_{bs} = 0.003 L_{\mathrm{exp}}\ \mathrm{mol\ m^{-2}\ s^{-1}}$、および5電子化学量論と維管束鞘漏出に伴う ATP 追加消費を合わせた実効量子収率比 $0.80(1 - \phi_L) = 0.64$（日内ガス交換の電子伝達律速項 $A_{l,\mathrm{C4}} = 0.20(1 - \phi_L) J_4$ と整合）のもとで、維管束鞘分圧 $C_{bs} = C_m + (L / g_{bs}) p_{\mathrm{sfc}}$ を2回固定点反復し、C₄ の三酵素容量 $(J_{\max,\mathrm{C4}}, V_{p\max,\mathrm{C4}}, V_{c\max,\mathrm{C4}})$ を同時に導く。

3. **普遍的葉暗呼吸スケーリング (Heskel et al., 2016; Ren et al., 2024, 2025, Eq. 13)**:
   夜間葉呼吸量は最適 Rubisco 容量と比例関係 $R_{\mathrm{leaf}}(T_{\mathrm{night}}) = b_{\mathrm{resp}} V_{c\max,\mathrm{opt}}$（$b_{\mathrm{resp}} = 0.018$）にあり、高分子反応速度論（MMRT）温度関数：
   $$
   f_R(T) = \exp\!\left[0.1012(T_C - 25) - 0.0005(T_C^2 - 25^2)\right]
   $$
   により $25\ ^\circ\mathrm{C}$ 基準呼吸容量 $R_{25}$ へ変換され、各時刻の陽葉温・陰葉温で評価される。

### 7.2 正午下向き短波 EMA と Two-Leaf 光合成・気孔コンダクタンスの閉形式配分

経験的な窒素減衰パラメータ $k_n$ に頼らず樹冠内の光飽和・自己遮蔽を正確に扱うため、AOBA は2段階の順化・配分アルゴリズムを実行する（de Pury & Farquhar, 1997）：
1. **正午下向き短波の EMA 平滑化**: 地方太陽正午において地表到達の可視直達・散乱光 $(\mathrm{VIS}_b, \mathrm{VIS}_d)$ と天頂角余弦 $\mu_0$、日最高域気温・VPD、および夜温を $\alpha_{\mathrm{EMA}} = 0.067$（約15日記憶時定数）で平滑化し $(\widetilde{\mathrm{VIS}}_b, \widetilde{\mathrm{VIS}}_d, \tilde{\mu}_0)$ を得る。
2. **球面葉樹冠2ストリーム光吸収**: 現在の露出葉面積指数 $L_{\mathrm{exp}}$ の下で球面葉傾斜角・散乱パラメータ（$\sigma_P = 0.15,\ k_d' = 0.719,\ \rho_{cd} = 0.036,\ \tilde{k}_b' = 0.5\sqrt{1-\sigma_P}/\tilde{\mu}_0$）により順化吸収 PPFD $\tilde{I}_c$ を計算し、上記 EEO 最適容量 $X \in \{V_{c\max 25}, J_{\max 25}, V_{p\max 25}, R_{25}\}$ を更新する。
3. **陽葉・陰葉への閉形式積分配分**: 順化光環境プロファイル $\tilde{I}_l(L)$ と瞬時陽葉比率 $f_{\mathrm{sun}}(L) = \exp(-k_{db} L)$（$k_{db} = 0.5 / \mu_0(t)$）との積を層積算 $0 \to L_{\mathrm{exp}}$ で解析積分する：
   $$
   X_{\mathrm{sun}} = \frac{X}{\tilde{I}_c}\left[\tilde{I}_{b0}(1 - \tilde{\rho}_{cb})\tilde{k}_b'\frac{1 - e^{-(\tilde{k}_b' + k_{db})L_{\mathrm{exp}}}}{\tilde{k}_b' + k_{db}} + \tilde{I}_{d0}(1 - \rho_{cd})k_d'\frac{1 - e^{-(k_d' + k_{db})L_{\mathrm{exp}}}}{k_d' + k_{db}}\right], \qquad X_{\mathrm{sh}} = \max(10^{-6},\ X - X_{\mathrm{sun}}).
   $$
   各サブステップで陽葉 $(I_{\mathrm{sun}}, X_{\mathrm{sun}}, T_{\mathrm{sun}}, D_{\mathrm{sun}})$ と陰葉 $(I_{\mathrm{sh}}, X_{\mathrm{sh}}, T_{\mathrm{sh}}, D_{\mathrm{sh}})$ ごとに光合成速度（C₃/C₄ の双曲線最小協調 $A_n = \min_{\theta_{\mathrm{col}}}(A_c, A_j) - R_d$ に土壌水分ストレス係数 $\beta_{\mathrm{opt}}$ を乗じる）と気孔コンダクタンス $g_{sw,\mathrm{sun}}, g_{sw,\mathrm{sh}} = 1.6 A_n (R_{\mathrm{gas}} T_l / p_{\mathrm{sfc}}) / [c_a (1 - \chi)]\ [\mathrm{m\ s^{-1}}]$ を計算する。ここで $\beta_{\mathrm{opt}}$ は経験的な線形関数や PFT テーブルを用いず、Lu, Duursma, & Medlyn (2016) および Holtzman et al. (2024, Eqs. 2, 7 & Fig. 4) の時間積分最適気孔理論に基づき、生育期（$T > 0\ ^\circ\mathrm{C}$）の年最大連続無降水日数（$P < 0.5\ \mathrm{mm\ d^{-1}}$）の気候値 $D_{\max}$ から決まる節水タイムスケール $\tau$ を介して次式で算出する：
   $$
   \tau = 0.85\, D_{\max} + 6.59\ [\mathrm{days}], \qquad \beta_{\mathrm{opt}} = \mathrm{clip}\!\left(\sqrt{\frac{\max(W_{\mathrm{eff}} - 0.05\, S_R,\, 0)}{\max(\tau\, E_{\mathrm{pot,EMA}},\, 0.10\, S_R)}},\ 0,\ 1\right),
   $$
   ここで $W_{\mathrm{eff}} = f_{\mathrm{unfrozen}} W_{\mathrm{root}}$ は未凍結有効根圏水分、$E_{\mathrm{pot,EMA}}$ は Priestley & Taylor (1972, *Mon. Wea. Rev.*, 100, 81–92) のポテンシャル蒸発散 $E_{\mathrm{pot}} = 1.26\,\frac{s}{s + \gamma}\,\frac{\max(R_n,\, 0)}{L_{v0}}$（$s = \mathrm{d}e_s / \mathrm{d}T$, $\gamma = c_{pd} p_{\mathrm{sfc}} / (\varepsilon_w L_{v0})$, 地表面正味放射 $R_n = S_{\mathrm{net}} + \varepsilon_{\mathrm{sfc}}(L^\downarrow - \sigma T_{\mathrm{rad,sfc}}^4)$）の 15日移動平均である。なお、日次予報 LAI の潜在光合成 $A_{0,\mathrm{day}}$ にはこの $\beta_{\mathrm{opt}}$ を重複乗算せず、長期水制限は $\mathrm{fAPAR}_w$ と $f_0(\mathrm{AI})$ のみで評価する。

### 7.3 二ストリーム樹冠アルベド・自律的 C₄ 比率 $F_4$・Lambert-$W_0$ 予報植生構造

1. **地図不要の2ストリーム樹冠アルベド (Sellers, 1985; Dai et al., 2004, App. A)**: 球面葉分布（$G = 0.5,\ \bar{\mu} = 1.0,\ K = 0.5/\max(\mu_0, 0.02),\ \beta = 0.5$）と葉散乱係数 $\omega \in \{\sigma_{\mathrm{VIS}} = 0.15,\ \sigma_{\mathrm{NIR}} = 0.80\}$ に対し、単一散乱アルベド・後方散乱比を次式で求め、$\sigma_{\mathrm{raw}} = (\bar{\mu} K)^2 - (b^2 - c^2) \to 0$ 近傍では $|\sigma_{\mathrm{raw}}| < 10^{-4}$ 時に $K_{\mathrm{reg}} = K + 2\times 10^{-4}$ として特異点を滑らかに回避した閉形式アルベド $(\alpha_{\mathrm{dir}}, \alpha_{\mathrm{dif}})$ を算出する：
   $$
   \begin{aligned}
   a_s &= \frac{\omega}{2}\frac{0.5}{0.5 + \mu_s}\left[1 - \frac{\mu_s}{0.5 + \mu_s}\ln\!\left(\frac{0.5 + \mu_s}{\mu_s}\right)\right], \qquad \beta_0 = \mathrm{clip}\!\left(\frac{(1 + \bar{\mu}K)a_s}{\omega \bar{\mu} K},\ 0.05,\ 0.95\right), \\
   b &= 1 - (1 - \beta)\omega, \qquad c = \omega\beta, \qquad h = \frac{\sqrt{b^2 - c^2}}{\bar{\mu}}, \qquad p_{1,2} = b \pm \bar{\mu} h.
   \end{aligned}
   $$
2. **自律的 C₄ 優占率 $F_4 \in [0, 1]$ (Scott & Smith, 2022; Luo et al., 2024; Still et al., 2003; Lavergne et al., 2026)**: 正午純光合成優位性 $\Delta A_{\mathrm{adv}} = (A_{n,\mathrm{C4}} - A_{n,\mathrm{C3}}) / (|A_{n,\mathrm{C3}}| + 1)$ と森林閉鎖による Beer–Lambert 樹冠透過率 $f_{\mathrm{open}} = \exp(-k\,\mathrm{LAI})$ を結合した目標値 $f_{\mathrm{open}} \sigma(8\,\Delta A_{\mathrm{adv}})$ へ約60日時定数（$0.25\,\alpha_{\mathrm{EMA}}$）で緩和する。
3. **自律的 常緑／落葉生活型最適化 $f_{\mathrm{evg}} \in [0, 1]$・二重時定数予報 EEO $\mathrm{LAI}$・根圏貯水容量 $S_R$・樹冠高 $h_c$**: 年次指数平滑水文統計（$\alpha_{\mathrm{yr}} = 1/365.2425$）から乾燥度指数 $\mathrm{AI} = \mathrm{PET}_{\mathrm{yr}} / \max(P_{\mathrm{yr}}, 10)$ を求め、Zhou et al. (2025, Eq. 6) の $f_0(\mathrm{AI}) = 0.65 \exp\!\left[-0.604169 \left(\ln \max(\mathrm{AI}, 1.9)/1.9\right)^2\right]$ を介して水制限上限 $\mathrm{fAPAR}_w = f_0 P_{\mathrm{mol,yr}} c_a (1 - \chi_{\mathrm{C3}})/(1.6 D_s A_{0,\mathrm{yr}})$ と基準エネルギー制限上限 $\mathrm{fAPAR}_{c,0} = 1 - z/(k A_{0,\mathrm{yr}})$ の対数和指数（$K = 10$; Cai et al., 2025, Eq. 12）平滑最小値から基準 $\mathrm{fAPAR}_{\max,0}, \mathrm{LAI}_{\max,0}$（$k=0.5,\ z=12.227\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$）と好適期単位葉面積同化量 $a_{\mathrm{leaf}} = A_{0,\mathrm{fav,yr}} \mathrm{fAPAR}_{\max,0} / \max(\mathrm{LAI}_{\max,0}, 0.50)$ を評価する。
   さらに PFT 地図を一切用いず **8 個の全球生理パラメータのみ**で、Kikuzawa (1991, Eq. 5) の最適葉寿命・Wang et al. (2023) の耐寒構造投資（$f_{\mathrm{cold}} = \sigma[(T_0 - T_{\min,\mathrm{EMA}})/5]$）・落葉展葉ラグ（$\Delta t_{\mathrm{flush}} = 25\ \mathrm{d}$）・非生育期維持呼吸（$Q_{10} \approx 2,\ R_{\mathrm{off},25} = 32\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$）・および $-40\ ^\circ\mathrm{C}$ 以下の極寒冬季凍害コスト（$T_{\mathrm{frost}} = 233.15\ \mathrm{K},\ C_{\mathrm{frost}} = 18\ \mathrm{mol\ C\ m^{-2}\ yr^{-1}}$; Sakai & Larcher, 1987; Givnish, 2002）から常緑優占率 $f_{\mathrm{evg}}$ と実効償却葉コスト $z_{\mathrm{eff}}$ を自律決定する：
   $$
   \begin{aligned}
   C_{\mathrm{evg}} &= (1.35 + 0.55 f_{\mathrm{cold}})\, z, \qquad \eta_{\mathrm{evg}} = \max(1 - 0.14 f_{\mathrm{cold}},\ 0.40), \qquad \tau_{\mathrm{evg}} = \mathrm{clip}\!\left(\sqrt{\frac{2 b_{\mathrm{evg}} C_{\mathrm{evg}}}{\eta_{\mathrm{evg}} a_{\mathrm{leaf}}}},\ 1,\ 7\right)\ [\mathrm{yr}], \\
   G_{\mathrm{dec}} &= \left(1 - \frac{\Delta t_{\mathrm{flush}}}{365.24 f_{\mathrm{GSL}}}\right)_+ \left(1 - \frac{f_{\mathrm{GSL}}}{2 b_{\mathrm{dec}}}\right) a_{\mathrm{leaf}} - z, \\
   G_{\mathrm{evg}} &= \eta_{\mathrm{evg}} \left(1 - \frac{\tau_{\mathrm{evg}}}{2 b_{\mathrm{evg}}}\right) a_{\mathrm{leaf}} - \frac{C_{\mathrm{evg}}}{\tau_{\mathrm{evg}}} - R_{\mathrm{off},25} e^{0.0693(T_{\mathrm{off}} - 298.15)}(1 - f_{\mathrm{GSL}}) - C_{\mathrm{frost}}\,\sigma\!\left(\frac{T_{\mathrm{frost}} - T_{\min,\mathrm{EMA}}}{4}\right), \\
   f_{\mathrm{evg}}^{\mathrm{target}} &= \sigma\!\left(\frac{4(G_{\mathrm{evg}} - G_{\mathrm{dec}})}{z}\right), \qquad z_{\mathrm{eff}} = \max\!\left[(1 - f_{\mathrm{evg}})\, z + f_{\mathrm{evg}} \frac{C_{\mathrm{evg}}}{\tau_{\mathrm{evg}}},\ 0.55\, z\right]\left[1 + 1.5\,\sigma\!\left(20(0.28 - f_{\mathrm{GSL}})\right)\right] \quad (b_{\mathrm{evg}} = 15\ \mathrm{yr},\ b_{\mathrm{dec}} = b_{\mathrm{evg}}/2.5).
   \end{aligned}
   $$
   ここで $f_{\mathrm{GSL}}$ は好適生育日（$T > 5\ ^\circ\mathrm{C}$ かつ $\beta_{\mathrm{opt}} > 0.20$; Körner, 2012）の年間割合であり、$z_{\mathrm{eff}}$ を用いて $\mathrm{fAPAR}_c = 1 - z_{\mathrm{eff}}/(k A_{0,\mathrm{yr}})$・$\mathrm{fAPAR}_{\max}$・$\mathrm{LAI}_{\max} = -\ln(1 - \mathrm{fAPAR}_{\max})/k$ を更新する。落葉側平衡葉面積指数 $L_s$ は Lambert $W$ 関数の主枝 $W_0$（Halley 3次反復5回；Zhou et al., 2025, Eqs. 15–20）により：
   $$
   L_s = \min\!\left[\mu + \frac{1}{k} W_0\!\left(-k\mu e^{-k\mu}\right),\ \mathrm{LAI}_{\max}\right], \qquad \mu = \frac{\sigma_{\mathrm{Zhou}}\, \mathrm{GSL}\, \mathrm{LAI}_{\max}}{A_{0,\mathrm{fav,yr}}\, \mathrm{fAPAR}_{\max}} A_{0,\mathrm{day,fav}} \qquad (\sigma_{\mathrm{Zhou}} = 0.771)
   $$
   と求める。常緑成分 $\mathrm{LAI}_{\mathrm{evg}}$ は多年葉寿命の緩慢な回転率 $(365.24 \tau_{\mathrm{evg}})^{-1}$（日最低気温 $T_{\min,\mathrm{day}} < T_{\mathrm{frost}}$ 時の急性凍害枯死率 $\alpha_{\mathrm{frost}} = 0.025\,\sigma[(T_{\mathrm{frost}} - T_{\min,\mathrm{day}})/4]$ および乾燥超過枯死を含む）で $f_{\mathrm{evg}} \mathrm{LAI}_{\max}$ へ緩和し、落葉成分 $\mathrm{LAI}_{\mathrm{dec}}$ は15日時定数（$\alpha_{\mathrm{EMA}} = 0.067$）で $(1 - f_{\mathrm{evg}}) L_s$ へ緩和して総葉面積指数 $\mathrm{LAI} = \mathrm{LAI}_{\mathrm{evg}} + \mathrm{LAI}_{\mathrm{dec}}$ を予報する。これにより、湿潤熱帯常緑林・冬季も針葉を保持する亜寒帯タイガ・季節落葉性の温帯／モンスーン林・$-40\ ^\circ\mathrm{C}$ 以下の極寒シベリア落葉カラマツ（*Larix*）林・および Körner (2012) の生育期熱的森林限界（$f_{\mathrm{GSL}} \approx 0.30$）以高の高山／北極ツンドラが第一原理から自律創発する。さらに根圏最大貯水容量 $S_R \in [50, 650]\ \mathrm{kg\ m^{-2}}$（Gao et al., 2014）と最大樹高 $h_c \in [0.05, 40]\ \mathrm{m}$（Kempes et al., 2011; Körner, 2012）は日積算水欠損 $D_{\mathrm{cum}}$ と水・光合成余剰スケーリングおよび森林限界ゲートにより次式で自律進化する（乾燥側の $S_R$ 増加は $10\,\alpha_{\mathrm{yr}}$ で速く追従し、湿潤側の縮小は $\alpha_{\mathrm{yr}}$ で緩やかに戻る）：
   $$
   \begin{aligned}
   D_{\mathrm{cum}}^{(d)} &= \mathrm{clip}\!\left(D_{\mathrm{cum}}^{(d-1)} + (\mathrm{fAPAR}_{\max} E_{\mathrm{pot}} - P),\ 0,\ 800\right), \qquad S_{R,\mathrm{target}} = \mathrm{clip}\!\left(50 + 1.35\, D_{\mathrm{cum}}^{(d)},\ 50,\ 650\right), \\
   h_{c,\mathrm{target}} &= \mathrm{clip}\!\left[34.0 \left(\mathrm{clip}\!\left(\frac{P_{\mathrm{yr}}}{\max(E_{\mathrm{pot,yr}}, 50)},\ 0,\ 1.2\right)\right)^{0.85} \left(\mathrm{clip}\!\left(\frac{\mathrm{fAPAR}_{\max} A_{0,\mathrm{yr}}}{180},\ 0,\ 2.0\right)\right)^{0.75} \sigma\!\left(30(f_{\mathrm{GSL}} - 0.30)\right),\ 0.05,\ 40.0\right].
   \end{aligned}
   $$

---

## 8. 陸面水文・5層凍土熱伝導・積雪・海氷・開水面収支

### 8.1 3源（陽葉・陰葉・地表）エネルギー収支・RSL 整合葉面境界層・葉面 P モデル反復連成 (`src/aoba/surface.py`)

植生陸面（$L_{\mathrm{exp}} \ge 0.01$）は、**陽葉温度 $T_{\ell,\mathrm{sun}}$・陰葉温度 $T_{\ell,\mathrm{sh}}$・地表面／積雪面温度 $T_g$ の独立した3温度ノード**と、熱・水蒸気容量ゼロの診断的**樹冠内空気ノード $(T_{\mathrm{af}}, q_{\mathrm{af}})$** を介して大気空気力学コンダクタンス $g_a = C_H U_{\mathrm{eff}}$ に接続する（Dai et al., 2004, Sec. 2b–c; Bonan et al., 2018, Eqs. 8–13, 21–24; CLM5 Sec. 5.2–5.3）。無植生面（$L_{\mathrm{exp}} < 0.01$）では単一地表収支へ厳密帰着する。

1. **RSL 整合の粗度長と陽葉・陰葉別境界層コンダクタンス**: 明示的な葉面層流境界層抵抗 $r_b$ を持つ植生面では $k B^{-1}$ 二重計上を防ぐためスカラー粗度長を $z_{0h,\mathrm{can}} = z_{0m,\mathrm{can}}$ へ滑らかに遷移させ（Oleson et al., 2013, p. 95）、裸地・裸雪面のみ $z_{0h} = 0.135335\, z_{0m}$ とする。樹冠内の RSL 指数風速減衰 $u(L) = u(h)e^{-\alpha_u L}$ を陽葉率 $e^{-k_b L}$・陰葉率 $1 - e^{-k_b L}$ について解析積分し、閉形式で独立の葉面境界層コンダクタンス $(g_{b,\mathrm{sun}}, g_{b,\mathrm{sh}})$（両面顕熱 $2 g_{b,j}$）を与える。
2. **気孔蒸散・遮断蒸発・葉面 $(c_s, D_s)$ P モデル連成**: 濡れ樹冠率 $f_{\mathrm{wet}} = \min\{[W_c / (p_{\mathrm{liq}} L_{\mathrm{exp}})]^{2/3},\ f_{\mathrm{wet,max}}\}$（$p_{\mathrm{liq}}=0.1\ \mathrm{kg\ m^{-2}}$; CLM5 Eq. 7.23）に対し、乾いた陽葉・陰葉それぞれで直列合成 $g_{\mathrm{trans},j} = (1 - f_{\mathrm{wet}})\frac{g_{b,j} g_{sw,j}}{g_{b,j} + g_{sw,j}}$、濡れ葉では利用可能水量制限付き $g_{\mathrm{inter},j} = f'_{\mathrm{wet}} g_{b,j}$（不足潜熱は顕熱へ戻す；CLM5 Eqs. 5.111, 5.139；降雨遮断率 $f_{\mathrm{pi}} = \alpha_{\mathrm{liq}} \tanh L_{\mathrm{exp}}$、CLM5 Eq. 7.4）を与える。P モデル（`compute_two_leaf_gas_exchange`）は反復ごとに現在の $(T_{\ell,\mathrm{sun}}, T_{\ell,\mathrm{sh}})$ で再評価され、葉面 $\mathrm{CO_2}$ 分圧 $c_s = c_{\mathrm{af}} - A_n P / g_{bc}$（$g_{bc} = g_{bw}/1.37$; Dai et al., 2004, Eq. 11）と葉面 VPD $D_s = \frac{g_{bw}}{g_{bw} + g_{sw}} D_{\ell,\mathrm{af}}$（Wang et al., 2017; Bonan et al., 2018）を 2 パス内部反復で整合させることで、$g_{sw,j}$ を純粋気孔コンダクタンスとして二重計上なく $g_{b,j}$ と直列結合する。
3. **裸地蒸発・積雪昇華**: 表層飽和度 $s_0 = (w_{\mathrm{liq},0} + w_{\mathrm{ice},0}) / \phi_{\mathrm{soil}}$（$\phi_{\mathrm{soil}} = 0.43$）に対し、CLM5 の乾燥表層（DSL）土壌抵抗 $r_{\mathrm{soil}} = D_{\mathrm{SL}} / (D_v \tau)$（Swenson & Lawrence, 2014; CLM5 Eqs. 5.76–5.81）、ケルビン則相対湿度 $h_{\mathrm{soil}} = \exp(\psi_0 g / (R_v T_g))$、および Sakaguchi & Zeng (2009) の樹冠下安定度補正抵抗 $r_{d,\mathrm{uc}}$ から $g_{\mathrm{soil}} = (1 - A_{\mathrm{sn}})/(r_{d,\mathrm{uc}} + r_{\mathrm{soil}})$、積雪面は $g_{\mathrm{snow}} = A_{\mathrm{sn}} / r_{d,\mathrm{uc}}$ を与え（CLM5 Eq. 5.110）、大気側抵抗 $g_a^{-1}$ は樹冠空気節点で 1 回のみ作用させる。未飽和空気による極乾燥土壌への偽凝結は $q_{\mathrm{soil}} = q_{\mathrm{af}}$・$\partial q_{\mathrm{soil}}/\partial T_g = 0$ として厳密に禁止する（CLM5 Sec. 5.2）。

短波吸収（$S_{\mathrm{sun}} + S_{\mathrm{sh}} + S_g = S_{\mathrm{abs}}$）と1回反射を含む長波収支（CLM5 Sec. 4.2、$R_{\mathrm{lw,sun}} + R_{\mathrm{lw,sh}} + R_{\mathrm{lw},g} = L_\downarrow - L_\uparrow$）を厳密保存させた上で、$(T_{\ell,\mathrm{sun}}, T_{\ell,\mathrm{sh}}, T_g)$ の 3×3 結合ニュートン方程式を毎ステップ `land_eb_iterations = 3` 回反復（振動反転時は緩和係数 0.5）して葉温・気孔コンダクタンス・蒸発散の平衡解を求める。積雪存在時（$\mathrm{SWE} > 10^{-3}\ \mathrm{kg\ m^{-2}}$）は地表／積雪ノード $T_g \le T_0$ にクランプして余剰エネルギーを融雪熱量 $M_{\mathrm{snow}}$ へ振り分ける（MATSIRO6, Sec. 6.3.3）。

### 8.2 部分積雪被覆・雪埋没遮蔽・エージングアルベド・氷河氷流出

1. **部分積雪面積率 $A_{\mathrm{sn}}$**: Swenson & Lawrence (2012) と Nitta et al. (2014) に基づき、`Earth2014` サブグリッド標高標準偏差 $\sigma_{\mathrm{topo}}$ と融雪ヒステリシス曲線を表現する。
2. **雪埋没による植生遮蔽**: 積雪深 $d_{\mathrm{sn}} = \mathrm{SWE} / \rho_{\mathrm{sn}}$（$\rho_{\mathrm{sn}} = 250\ \mathrm{kg\ m^{-3}}$）により、$h_{\mathrm{exp}} = \max(h_c - d_{\mathrm{sn}},\ 0),\ L_{\mathrm{exp}} = \mathrm{LAI} \cdot (h_{\mathrm{exp}} / \max(h_c, 10^{-3}))$ と減衰させる（MATSIRO6, Eqs. 2.1–2.3）。
3. **2バンド積雪エージングアルベド**: 新雪（$\alpha_{\mathrm{VIS,new}}=0.90, \alpha_{\mathrm{NIR,new}}=0.70$）から旧雪（$0.65, 0.35$）への無次元積雪齢 $\tau_{\mathrm{sn}}$ の熱的進行と降雪リセット・太陽天頂角補正を計算する（Yang et al., 1997; MATSIRO6, Sec. 8.8）。
4. **氷河・積雪上限流出と氷床への SMB 引き渡し**: 非動的氷床モードの陸域・氷床域では、上限 $\mathrm{SWE}_{\max} = 1000\ \mathrm{kg\ m^{-2}}$ を超える余剰積雪は氷河氷流出（calving）として海洋へ輸送され融解潜熱を吸収させる（MIROC6, Sec. 4.1.2）。動的氷床モード有効時（第11章）の接地氷床および積雪域では、降雪・昇華・融雪・氷面融解ポテンシャルの正味表面質量収支 $\mathrm{SMB}\ [\mathrm{kg\ m^{-2}\ s^{-1}}]$ を `SurfaceFluxDiagnostics.smb_ice_flux` として積算・供給する。

### 8.3 5層陰的リチャーズ土壌水文・動的根圏深さ・5層凍土熱伝導（ゼロカーテン効果）

層厚 $\Delta z_s = [0.05, 0.15, 0.40, 0.80, 1.60]\ \mathrm{m}$（総厚 $3.0\ \mathrm{m}$）の5層土壌各層 $k$ に対し、単一ローム土壌の保水・透水曲線（Clapp & Hornberger, 1978; $\phi = 0.43, \psi_s = -0.20\ \mathrm{m}, b = 5.39, K_{\mathrm{sat}} = 7\times 10^{-6}\ \mathrm{m\ s^{-1}}$）と凍土透水阻害 $f_{\mathrm{ice},k} = 10^{-\Omega_{\mathrm{ice}} (\theta_{\mathrm{ice},k}/\phi)}$（$\Omega_{\mathrm{ice}} = 6$; Swenson et al., 2012; CLM5 Eq. 7.89）を用い、線形化陰的リチャーズ方程式を三重対角ソルバ `solve_tridiagonal_batched` で厳密質量保存に解く（CLM5 Sec. 7.3–7.5, Eqs. 7.89–7.129）。地表面は地図不要の浸透能超過・表層飽和／凍結超過流出を分離し、裸地蒸発 $E_{\mathrm{soil}}$ は最上層（$0\text{–}5\ \mathrm{cm}$）からのみ抽出し、底面は自由重力排水 $q_{\mathrm{drain}} = K_4$ を基底流出とする。Gao et al. (2014) の最適根圏容量 $S_R$ から動的根圏深さ $z_{\mathrm{root}} = \mathrm{clip}(S_R / [\rho_w(\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}})],\ 0.15,\ 3.0\ \mathrm{m})$ を定め、根圏層にわたる鉛直積分液相有効水量 $W_{\mathrm{avail}} = \sum_k \rho_w \Delta z_k \alpha_{\mathrm{root},k} \max(\theta_{\mathrm{liq},k} - \theta_{\mathrm{wp}}, 0)$ を Holtzman et al. (2024, Eq. 1) の気孔水分ストレス $\beta_{\mathrm{opt}}$ および各層の蒸散吸水重みへ直接接続する。

熱力学は SCALE-RM (Sec. 8.7.1) と MATSIRO6 (Ch. 10) に従い氷点基準内部エネルギー $U_l = C_l (T_l - T_0) - L_{f0} M_{i,l}$（$C_l = C_{\mathrm{dry}} + c_{w,\mathrm{liq}} M_{w,l} + c_{i,\mathrm{sol}} M_{i,l}$）を定義し、最上層へ地中熱流 $G$ を加えたのち、$0\ ^\circ\mathrm{C}$ での厳密相分配 $M_{i,l}^{\mathrm{new}} = \mathrm{clip}(-U_l / L_{f0},\ 0,\ M_{w,l}+M_{i,l})$ を実行し、含水・含氷依存熱伝導率 $\kappa_l = \kappa_{\mathrm{dry}}(1 + 5.5\tanh((w_l+i_l)/0.25)) + 0.8\, i_l$ を用いたフーリエ熱伝導を温度偏差 $(T_l - T_0)$ について三重対角陰解法で解く。

### 8.4 太陽天頂角依存海洋アルベドと部分海氷（リード）熱力学

開水面の直達アルベドは Briegleb et al. (1986)、拡散アルベドは $0.06$ とする：
$$
\alpha_{\mathrm{ocn,dir}}(\mu_0) = \frac{0.026}{\mu_0^{1.7} + 0.065} + 0.15(\mu_0 - 0.1)(\mu_0 - 0.5)(\mu_0 - 1.0).
$$
海氷被覆率（リード補数）は $f_i = \mathrm{clip}(\sqrt{M_i / M_{i,\mathrm{crit}}},\ 0,\ 1)$（$M_{i,\mathrm{crit}} = 450\ \mathrm{kg\ m^{-2}}$; Semtner, 1976; SCALE-RM Eq. 8.342）とし、開水面 $(1 - f_i)$ と積雪載荷海氷面 $f_i$（積雪載荷海氷固有の可視・近赤外アルベドによるタイル別吸収短波 $S_{\mathrm{abs,ice}}$ を用いた Semtner, 1976 型熱伝導と凍結・融解収支）を個別に解いて面積加重平均フラックスを与える。

### 8.5 地図不要の鉱物ダスト放出・鉛直輸送・放射結合 (`src/aoba/dust.py`)

鉱物ダスト放出は CLM5 DEAD スキーム（Lawrence et al., 2020, Ch. 31, Eqs. 31.1–31.14; Oleson et al., 2013, Ch. 24; Zender et al., 2003; Mahowald et al., 2006）に準拠し、地域放出源マップを用いず（$S = 1$）、地図不要設定（$S = 1, M_{\mathrm{clay}} = 0.20$）向けに校正した単一の全球調整係数 $T = \text{dust\_emis\_factor} = 1.5\times 10^{-5}$ で鉛直放出フラックスを与える：
$$
F_{\mathrm{dust}} = T\, f_m\, \alpha\, Q_s \sum_{i=1}^{3} M_i, \qquad f_m = (1 - A_{\mathrm{sn}})(1 - f_v)\frac{w_{\mathrm{liq},1}}{w_{\mathrm{liq},1} + w_{\mathrm{ice},1}}, \qquad f_v = \mathrm{clip}\!\left(\frac{L_{\mathrm{exp}}}{0.3},\,0,\,1\right).
$$
単一ローム土壌の粘土質量分率 $M_{\mathrm{clay}} = 0.20$ に対するサンドブラスティング効率 $\alpha = 100\exp[(13.4 M_{\mathrm{clay}} - 6.0)\ln 10]\ \mathrm{m^{-1}}$（Marticorena & Bergametti, 1995）、最適跳躍粒径 $D_{\mathrm{osp}} = 75\ \mu\mathrm{m}$（$\rho_{\mathrm{osp}} = 2650\ \mathrm{kg\ m^{-3}}$）の Iversen & White (1982) 乾燥しきい摩擦速度に、Fécan et al. (1999) の土壌水分係数 $f_w = \sqrt{1 + 1.21 [100(w - w_t)]^{0.68}}$（重量含水率 $w = 1000(w_{\mathrm{liq},1}+w_{\mathrm{ice},1}) / [(1 - \phi_{\mathrm{soil}})\rho_{\mathrm{solid}}] > w_t = 0.17 + 0.14 M_{\mathrm{clay}}$、CLM 調整係数 $a = M_{\mathrm{clay}}^{-1}$ を適用）を乗じて $u_{*t}$ を求める。さらに Owen (1964) 効果 $u_{*s} = u_* + 0.003(U_{10} - U_{10,t})^2$（$U_{10} \ge U_{10,t}$）を反映した White (1979) 水平跳躍フラックス $Q_s = c_s (\rho_{\mathrm{atm}} u_{*s}^3 / g)(1 - r)(1 + r)^2$（$r = u_{*t}/u_{*s} < 1$, $c_s = 2.61$）に輸送粒径帯（$0.1\text{–}10\ \mu\mathrm{m}$）の質量分率和 $\sum M_i = 0.871$ を乗じる。大気柱ダスト混合比 $q_{\mathrm{dust}}$ は PBL 鉛直混合・陰的重力沈降（$v_{\mathrm{settle}} = 4\times 10^{-3}\ \mathrm{m\ s^{-1}}$）・乾性沈着（$v_{\mathrm{dep}} = 6\times 10^{-3}\ \mathrm{m\ s^{-1}}$）・降水洗浄（$\alpha_{\mathrm{scav}} = 0.50\ \mathrm{m^2\ kg^{-1}}$）を受け、Balkanski et al. (2007) の広帯域短波・長波光学特性で放射結合する。

### 8.6 地図不要 EEO 陸域 CN モデル・CRM 落雷／$\mathrm{LNO}_x$・MAML パッチ SPITFIRE 山火事 (`src/aoba/lightning.py`, `src/aoba/land_carbon.py`)

`enable_lightning = true` および `enable_land_carbon = true` のもとで、地域マップ・PFT マップを一切使わずに陸域炭素・窒素・落雷・山火事の結合循環を全球パラメータのみで閉じる：
1. **埋め込み CRM 非誘導電荷分離「落雷」＆ 自然 $\mathrm{LNO}_x$ 沈着 (`src/aoba/lightning.py`)**: 各 CRM サブコラム `[N_cells, Ny, Nx]` で $-15\ ^\circ\mathrm{C}$ 混合相帯の上昇霰フラックス $F_1 = 0.042\,\max_k[w_{-15}\rho_d w_+ (1000 q_g)]$ と鉛直積算総氷質量 $F_2 = 0.20 \int \rho_d(q_i + q_{p,\mathrm{ice}})\,\mathrm{d}z$ を合成し、単一の全球校正係数 $c_{\mathrm{cal}}$ で全発雷頻度 $F_{\mathrm{tot}} = c_{\mathrm{cal}}(0.95 F_1 + 0.05 F_2)/(300\times 10^6)\ [\mathrm{flashes\ m^{-2}\ s^{-1}}]$ を診断する（McCaul et al., 2009, Eqs. 1–3）。対地落雷比 $f_{\mathrm{CG}} = [1 + Z(H_{\mathrm{cold}})]^{-1}$ は寒冷雲厚 $H \in [5.5, 14]\ \mathrm{km}$ の 4 次多項式 $Z(H) = 0.021 H^4 - 0.648 H^3 + 7.493 H^2 - 36.54 H + 63.09$（Price & Rind, 1993, Eq. 1）で与える。1 閃光あたり $3.5\ \mathrm{kg\ N}$（$250\ \mathrm{mol\ NO}_x$; Schumann & Huntrieser, 2007）を生成し、局所対流洗浄 50% ＋ 全球一様背景沈着 50% で厳密質量保存の自然大気窒素沈着を陸海へ供給する。
2. **Stocker & Prentice (2024) CN モデル ＆ FUN/Houlton 最適生物的窒素固定 (`enable_land_nitrogen = true`)**: 葉・細根・木・速/遅リター・5 層凍土有機物・土壌無機 N（`n_min`）の C/N 予報プール（Neumaier/Kahan 補償和）に対し、P モデル最適 $V_{c\max 25}$ と LMA から葉 N 需要 $N_{\mathrm{area}} = n_{\mathrm{rub}} V_{c\max 25} + n_{\mathrm{struct}}\mathrm{LMA}$（Dong et al., 2022）を定め、落葉時再転流（葉 50%・根 25%）・第一原理溶質輸送（蒸散移流＋細根拡散）N 吸収・および土壌温度依存の共生 BNF 炭素コスト $\mathrm{Cost}_{\mathrm{fix}}(T_s) = 5.0 / \exp[-3.62 + 0.27 T_s(1 - 0.5 T_s/25.15)]\ [\mathrm{kg\ C\ (kg\ N)^{-1}}]$（Houlton et al., 2008; Fisher et al., 2010, FUN Eq. 6）で N 供給を解き、排水溶脱・嫌気脱窒・山火事 N 揮散（85%、残り 15% は灰として `n_min` へ還元）で外部収支を閉じる。窒素充足度 $\phi_N \in [0.15, 1.0]$ の低下に応じて機能的バランス則により細根配分 $f_{\mathrm{root}} = \mathrm{clip}(f_{\mathrm{root},0} + 0.35(1 - \phi_N), 0.15, 0.72)$ を増加させ、さらに EEO 実効葉コスト $z_{\mathrm{eff}}$ を $1 + 0.80(1 - \phi_N)$ 倍して LAI・樹高を自律調節する（Stocker & Prentice, 2024; Stocker et al., 2025）。
3. **MAML サブグリッド離散パッチ SPITFIRE 山火事 (`enable_fire_veg_feedback = true`)**: サブコラム対地落雷着火（$n_{\mathrm{ig}} = 0.04 F_{\mathrm{CG}}\mathrm{FDI}\,\Delta t$）・Rothermel (1972) 含水率減衰延焼（$\mathrm{ROS}_f, \mathrm{ROS}_b$）・風速楕円焼失面積（$L_B, t_{\mathrm{fire}}$）・Byram (1959) 火線強度（$I_B \ge 50\ \mathrm{kW\ m^{-1}}$ で自己持続）・Van Wagner (1973) 焦げ高さ・樹高依存樹皮厚による耐火性から、`[N_cells, Ny, Nx]` MAML 格子上で離散パッチ燃焼と 2D 隣接パッチ延焼を実行し、焼失パッチのみの `lai` と `h_canopy` を直接縮減して同一セル内の森林–サバンナ混在モザイクを創発させる（Thonicke et al., 2010）。

---

## 9. 埋め込み 3D 非静力学 RK3–HEVI CRM と Zalesak FCT

### 9.1 3段 Runge–Kutta + 水平陽的・鉛直陰的 (HEVI) 圧縮性流体ソルバ (`src/aoba/crm.py`)

各 SCVT ホストセルには水平二重周期の直交 Arakawa C-grid / Charney–Phillips 3D CRM（既定 `[Nz=32, Ny=1, Nx=8], dx=4000 m` または 3D `[8, 8], dx=4000 m`）が埋め込まれ、乾燥空気密度 $\rho_d$、風速 $(u, v, w)$、湿潤温位フラックス変数 $\Theta_m = \rho_d \theta_m$ に対する完全圧縮性フラックス形式方程式を解く（Skamarock et al., 2012, Eqs. 7–11; SCALE-RM Ch. 3）：
$$
\begin{aligned}
\frac{\partial \rho_d}{\partial t} + \nabla_h \cdot (\rho_d \mathbf{u}_h) + \frac{\partial F_z}{\partial z} &= 0, \qquad F_z = \rho_{d,w} w, \\
\frac{\partial \Theta_m}{\partial t} + \nabla_h \cdot (\Theta_m \mathbf{u}_h) + \frac{\partial (\theta_{m,w} F_z)}{\partial z} &= S_{\Theta_m}, \\
\rho_{\mathrm{tot}, w}\frac{\partial w}{\partial t} &= -\frac{\partial p}{\partial z} - \rho_{\mathrm{tot}, w} g - \rho_{\mathrm{tot}, w} \gamma_R(z_w) w + \text{Adv}_w + D_w.
\end{aligned}
$$
時間積分は Wicker & Skamarock (2002) の3段 RK3 法（部分時間刻み $\Delta\tau \in \{\Delta t/3, \Delta t/2, \Delta t\}$）で行い、遅い移流（水平運動量は3次風上 Wicker–Skamarock 差分化、鉛直は1次風上）・ホスト大規模強制・SGS 粘性を外力傾向として保持しつつ、速い音波・重力波を複数音響サブステップで積分する。埋め込み周期境界 CRM 内のコリオリ力については、ホスト大規模気圧傾度力との二重評価を防ぐため次元に応じて自動切り替えを行い（Khairoutdinov et al., 2005; Yamazaki & Miura, 2024）、1D (`1×1`) および 2D (`nx==1` または `ny==1`) ではコリオリ項を無効（$f=0$）、3D (`nx>1` かつ `ny>1`) では背景平均風を差し引いた擾乱風 $\mathbf{u}' = \mathbf{u} - \overline{\mathbf{u}}$ にのみコリオリ力（$+f v',\, -f u'$）を作用させる。また、周期境界 2D カーテン（および無回転 3D 領域）の力学サブステップは各列の中点風速 $U_{\mathrm{shift}} = \tfrac{1}{2}(u_{\max} + u_{\min})$ を差し引いた厳密ガリレイ共動座標系で積分し（Grabowski, 2001; Khairoutdinov et al., 2005）、鉛直シアや物理過程を一切変えずに成層圏極夜ジェット等の強風時の水平 CFL を半減させる。各音響サブステップでは Klemp, Skamarock, & Dudhia (2007) のオフセンタ HEVI 陰的結合パラメータ $\beta_{\mathrm{HEVI}} = 0.60$ を導入し、線型化状態方程式 $\delta p = (\gamma p / \Theta_m) \delta \Theta_m$ を鉛直方向へ連立消去して鉛直質量フラックス増分 $\delta F_z$ に関する狭義対角優位な $(N_z - 1) \times (N_z - 1)$ 三重対角 Helmholtz 系を全 CRM コラム一括で `solve_tridiagonal_batched` により求解し、3次元乾燥空気質量と $\Theta_m$ を計算機イプシロン精度で厳密保存する。

### 9.2 多次元正値保存 Zalesak 3D FCT スカラー移流

予報水分トレーサ $(q_n, q_p)$ の移流には、RK3 第3段で蓄積した段階整合質量フラックス $(F_x, F_y, F_z)$ を用いて Zalesak (1979, Eqs. 1–17; SCALE-RM App. A.2 Eqs. A.20–A.35) の3D Flux-Corrected Transport (FCT) を実行する。1次風上単調流束 $\mathbf{F}^L$ による予更新値 $q^{\mathrm{td}}$ と、水平4次中心＋鉛直2/4次高次流束 $\mathbf{F}^H$ の差分である反拡散流束 $\mathbf{A} = \mathbf{F}^H - \mathbf{F}^L$ に対し、局所近傍極値 $(q^{\min}, q^{\max})$ からはみ出さない正・負許容制限比 $(R^+, R^-) \in [0, 1]$ を求めて各面の制限係数 $C_f$ を掛けることで、$q_n, q_p \ge 0$ の正値性と総質量の厳密保存を両立する。

---

## 10. 全球 SCVT C-grid 力学・山岳ドラッグ・エクマン海洋・MAML+ICA 結合

### 10.1 球面重心ボロノイ格子 (SCVT) と TRiSK 非静力学ホストコア (`src/aoba/scvt_mesh.py`, `scvt_dynamics.py`)

正二十面体の再帰 $g$ 回分割（セル総数 $N_{\mathrm{cells}} = 10 \cdot 4^g + 2$、辺数 $N_{\mathrm{edges}} = 30 \cdot 4^g$、頂点数 $N_{\mathrm{verts}} = 20 \cdot 4^g$）と球面上 Lloyd 緩和（Ringler et al., 2008）により、厳密球面余剰面積 $\sum_i A_i = 4\pi R_E^2$ を持つ準一様 SCVT 直交 C-grid を構築する。また 5分角 `Earth2014` (`SUR2014`, `TBI2014`, `ICE2014`, `BED2014`) を各ボロノイセル内へ面積集計し、地形標高 $z_{\mathrm{topo}}$、岩盤標高 $z_{\mathrm{bed}}$、初期氷床厚 $h_{\mathrm{ice},0}$、表面種別マスク（`Ocean`, `Land`, `IceSheet`）、およびサブグリッド起伏統計量 $(\mu, \sigma, \gamma, \psi)$ を算出する。

辺法線風速 $u_e$ の発展方程式は Thuburn et al. (2009) および Ringler et al. (2010, Eqs. 17–26) のベクトル不変 TRiSK 定式化に klemp STF 気圧傾度計量項を組み合わせた形をとる：
$$
\frac{\partial u_e}{\partial t} = \sum_{e' \in \mathrm{EC}(e)} w_{e, e'} \frac{\eta_e + \eta_{e'}}{2} u_{e'} - \frac{K_{c_1} - K_{c_0}}{d_{c, e}} - \frac{1}{\rho_{\mathrm{tot}, e}} \left[\left(\frac{\partial p}{\partial n}\right)_\zeta - \frac{\partial p}{\partial z}\left(\frac{\partial z}{\partial n}\right)_\zeta\right] + D_{u, e},
$$
ここで絶対渦度 $\eta_v = \zeta_v + f_v$、歪対称重み $w_{e,e'}$ は恒等式 $A_e w_{e,e'} + A_{e'} w_{e',e} = 0$ を満たしてコリオリ力・回転項のエネルギー生成をゼロに保つ。水平混合項 $D_{u,e}$ は3D発散ダンピング・Smagorinsky 粘性および4階超粘性 $\nabla^4$ で構成され、鉛直方向には第9章と共通構造の圧縮性 HEVI ソルバ（地形勾配に伴う反変鉛直速度 $u \cdot \nabla z$ を含む）および SCVT 上の Zalesak FCT 移流を統合する。

### 10.2 Lott & Miller (1997) サブグリッド山岳ドラッグ ＆ マップ不要・流れ依存 非地形性重力波 (`src/aoba/orography.py`, `src/aoba/gravity_wave.py`)

各セルの `Earth2014` 起伏共分散統計 $(\mu = \sigma_{\mathrm{topo}},\ \text{勾配 } \sigma,\ \text{異方性 } \gamma,\ \text{主軸角 } \psi)$ と下層入射流 $(\mathbf{U}_L, \bar{N}_L)$ から逆フルード数 $H_n = \bar{N}_L (2\mu) / |\mathbf{U}_L|$ を評価する。風向と主軸のなす角 $\phi = \theta_{\mathrm{wind}} - \psi$ から楕円指向性因子 $D_{\mathrm{dir}}(\phi, \gamma)$ を求め、$H_n > H_{nc} = 0.50$ のときブロッキング高さ $Z_b = 2\mu(1 - H_{nc}/H_n)$ 以下には陰的減衰率付きの鈍頭物体ウェイクドラッグ（$C_d = 1.00$）を課す。残余山高 $h_{\mathrm{eff}} = 2\mu - Z_b$ からは重力波起動応力 $\tau_{\mathrm{launch}} = \rho_L k_w \bar{N}_L |\mathbf{U}_L| h_{\mathrm{eff}}^2$（$G = 0.80$）を発射し、上層各境界で Lindzen 飽和応力 $\tau_{\mathrm{sat}}(z) \propto \rho U_{\mathrm{proj}}^3 / (N H_{nc}^2)$ を超過する運動量フラックスを砕波減速として大気へ還元する（MIROC6, Sec. 3.5）。

さらに `enable_nonoro_gwd = true` 時には、処方波源マップを一切用いずに成層圏循環・極渦・QBO を自律駆動する**流れ依存（flow-dependent）非地形性重力波ドラッグ**（`src/aoba/gravity_wave.py`）を結合する：
1. **前線・ジェット由来波源**（Hoskins, 1982; Charron & Manzini, 2002, Eqs. 1–7; Richter et al., 2010, Sec. 2a, Eqs. 1–2）：$600\ \mathrm{hPa}$ 付近で SCVT ガウス・グリーン水平勾配から2次元前線形成関数
   $$
   \mathcal{F} = -\left(\frac{\partial \theta}{\partial x}\right)^2\frac{\partial u}{\partial x} - \left(\frac{\partial \theta}{\partial y}\right)^2\frac{\partial v}{\partial y} - \frac{\partial \theta}{\partial x}\frac{\partial \theta}{\partial y}\left(\frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}\right)
   $$
   を算出し、$\mathcal{F} > \mathcal{F}_{\mathrm{crit}} = 1.25\times 10^{-15}\ \mathrm{K^2\ m^{-2}\ s^{-1}}$ を超える前線域から風向に沿ったガウス型位相速度スペクトル $\tau_{\mathrm{fr}}(c_m) \propto \tau_{\mathrm{bgnd}}\exp[-(c_m - u_{\mathrm{src}})^2/c_w^2]$（$c_w = 30\ \mathrm{m\ s^{-1}}$）を発射する。
2. **1:1 CRM 対流加熱由来波源**（Beres, Alexander, & Holton, 2004, Eqs. 6–12; Beres et al., 2005; Richter et al., 2010, Sec. 2b）：対流圏内の CRM 潜熱加熱プロファイル $Q_{\mathrm{conv}}(z) > 0$ から最大加熱率 $Q_0$・加熱層厚 $h_d$（$Q > 0.1 Q_0$）・加熱層平均風 $\mathbf{u}_h$・安定度 $N_h$ を直接診断し、固有位相速度ピーク $\sigma_c = \mathrm{clip}(N_h h_d / \pi, 10, 60)\ \mathrm{m\ s^{-1}}$ の2峰スペクトル $\tau_{\mathrm{conv}}(c_m) \propto \tau_{c,0}(Q_0/Q_{\mathrm{ref}})^2 (|c_m - u_h|/\sigma_c)\exp[-\frac{1}{2}((c_m - u_h)/\sigma_c)^2]$ を東西・南北2軸へ発射する。
3. **上端透過スペクトル伝播と Lindzen 飽和**（Lindzen, 1981; Alexander & Dunkerton, 1999; Richter et al., 2010, Eqs. 4–10）：離散位相速度ビン $c_m \in [-80, +80]\ \mathrm{m\ s^{-1}}$ を上向き伝播し、臨界層吸収（$\hat{c}_m = c_m - u_{\mathrm{proj}}$ の符号反転または $|\hat{c}_m| \le 2\ \mathrm{m\ s^{-1}}$）および Lindzen 飽和 $\tau_{\mathrm{sat}, m} = \frac{1}{2}\varepsilon_{\mathrm{eff}}\rho k_h |\hat{c}_m|^3 / N$（$\lambda_h = 200\ \mathrm{km}$）で砕波減速を与える。最上端は**開放透過境界（open-top）**とし、未砕波フラックスがモデル上端（24–80 km）で偽の暴走加速を起こさない設計としている。

### 10.3 処方 $Q$ フラックス不要 1.5層ボロノイエクマン＋浅水波海洋 (`src/aoba/ocean_ekman.py`)

Codron (2012, Eqs. 1–10) の風成エクマン輸送に、同一の SCVT ボロノイ C-grid 上の **1.5層低減重力浅水波力学（Reduced-Gravity Shallow-Water Dynamics）** を統合し、表層混合層（厚 $H_1$・水温 $T_1$）と内部補償温度躍層（厚 $H_2 = 200\ \mathrm{m}$・水温 $T_2$）、および活動上層厚 $H_{\mathrm{sw}} = H_1 + H_2 + \eta$（換算重力 $g' = 0.03\ \mathrm{m\ s^{-2}}$）と海洋エッジ法線流速 $u_{\mathrm{sw},e}$ を解く（全海陸境界辺では法線流速・フラックスとも厳密にゼロ）。海氷被覆率 $f_{\mathrm{ice}}$ による応力減衰 $\boldsymbol{\tau}_{\mathrm{ocn}} = -(1 - 0.5 f_{\mathrm{ice}})\boldsymbol{\tau}_{\mathrm{sfc}}$ とダンピング率 $r_E = (2\ \mathrm{days})^{-1}$ からエクマン質量輸送：
$$
M_{E,x} = \frac{r_E \tau_{x,\mathrm{ocn}} + f \tau_{y,\mathrm{ocn}}}{f^2 + r_E^2}, \qquad M_{E,y} = \frac{r_E \tau_{y,\mathrm{ocn}} - f \tau_{x,\mathrm{ocn}}}{f^2 + r_E^2},
$$
を求めると同時に、浅水運動量方程式（TRiSK コリオリ回転項・圧力傾度 $-g'\nabla H_{\mathrm{sw}}$・海面風応力 $\tau_{\mathrm{ocn}}/(\rho_w H_{\mathrm{sw}})$・スケール依存 Munk 水平渦粘性 $A_H \nabla^2 u_{\mathrm{sw}}$・線形層間抵抗 $-r_{\mathrm{sw}} u_{\mathrm{sw}}$）および風上層厚移流 $\partial_t H_{\mathrm{sw}} = -\nabla_{\mathrm{SCVT}}\cdot(H_{\mathrm{sw}} \mathbf{u}_{\mathrm{sw}})$ を CFL サブステップ付きで時間積分する。これにより赤道ケルビン波・ロスビー波および風成亜熱帯循環（Sverdrup バランスと西岸強化流）が創発する。

混合層輸送 $\mathbf{M}_1$ には、二重計上を避けるため全浅水流速からエクマン成分を差し引いた地衡流・循環成分 $\mathbf{u}_g = \mathbf{u}_{\mathrm{sw}} - \mathbf{M}_E / (\rho_w H_{\mathrm{sw}})$ を用いて $\mathbf{M}_1 = \mathbf{M}_E + \rho_w H_1 \mathbf{u}_g$、可変層厚（$H_2 = \max(H_{\mathrm{sw}} - H_1,\ 20\ \mathrm{m})$）の第2層質量フラックスを $\mathbf{M}_2 = \rho_w \mathbf{V}_{\mathrm{sw}} - \mathbf{M}_1$、湧昇フラックスを $W_{12} = \nabla_{\mathrm{SCVT}} \cdot \mathbf{M}_1$ として、可変層厚エンタルピー $\rho_w c_p (H_1 T_1 + H_2 T_2)$ が厳密保存されるよう混合層・躍層間の風上鉛直熱交換、水平渦拡散（$\kappa_h = 2000\ \mathrm{m^2\ s^{-1}}$）、躍層鉛直拡散（$\kappa_v = 2\times 10^{-5}\ \mathrm{m\ s^{-1}}$）を積分する。さらに全陸域からの河川液体流出・氷山流出の総和を海洋セルへ還元し、氷融解に伴う潜熱冷却を海洋エンタルピーから厳密差し引きして全球水・エネルギー収支を完結させる（MIROC6, Sec. 4.1.2）。なお Apple Silicon MPS GPU 等の単精度（`float32`）環境において熱容量の大きい海洋温度（$~280\ \mathrm{K}$）へ微小フラックス $\Delta T$ を連続加算する際の丸め消滅（swamping）を防ぐため、`ocean_ekman.py` の $T_1, T_2$ 更新および `surface.py` の大気・海氷熱収支による $T_1$ 更新には **Kahan/Neumaier 補償加算**（$y = \Delta T + c,\ T' = T + y,\ c' = (T - T') + y$）を組み込み、実質倍精度相当の長期エネルギー保存性を維持している。

### 10.3b 選択可能な 3次元多層 SCVT OGCM および海氷力学移流 (`ocean_mode = "ogcm_3d"` vs `"ekman_sw"`, `src/aoba/ocean_ogcm.py`)

AOBA では 1.5層浅水・エクマン海洋（`ocean_mode = "ekman_sw"`, 10.3節）に加え、`Earth2014 BED2014` 実海底地形（`mesh.z_bed`）上に構築された **3次元多層静水圧ブシネスク SCVT ボロノイ z-レベル OGCM**（`ocean_mode = "ogcm_3d"`, 既定 `ocean_nz = 18` 層・水深 $0\text{--}6000\ \mathrm{m}$; Ringler et al., 2013）を TOML / GUI から切り替え可能である：
1. **非線形状態方程式**（Roquet et al., 2015）: $\rho'(T, S, z) = \rho_0[-\alpha_0 \Delta T - \tfrac{1}{2}\alpha_T \Delta T^2 - \alpha_z z \Delta T + \beta_0 \Delta S - \alpha_S \Delta T \Delta S]$ により熱膨張・キャベリング・熱圧効果・塩分収縮を再現する。
2. **TRiSK スプリット陽解法自由表面と厳密保存型トレーサ移流**（Ringler et al., 2010, 2013; Campin et al., 2004）: 3D 傾圧運動量（TRiSK コリオリ・静水圧傾度・Munk 水平渦粘性・2次海底摩擦・陰的鉛直粘性）を高速 RK3 2D 順圧自由表面モード $(\bar{u}_e, \eta)$ と分離結合し、連続の式 $w_{k-1/2} = w_{k+1/2} - \nabla\cdot(h_k \mathbf{u}_{\mathrm{adv},k})$ と Campin et al. (2004) の表層体積更新 $h_{0,\mathrm{new}} = h_0 + \eta_{\mathrm{new}}$ により定数場保存と全球熱・塩分保存を処方 $Q$ フラックスなしで両立する。
3. **Gent–McWilliams (1990) 渦ボーラス輸送 ＆ フル回転 Redi (1982) 等密度面拡散テンソル**（Danabasoglu & McWilliams, 1995; Griffies et al., 1998, *JPO*, 28, 805–820; Ferrari et al., 2008）: KPP 境界層内で線形テーパーした等密度面傾き $\mathbf{S} = \mathrm{clip}(-\nabla_h\rho'/\partial_z\rho', \pm 0.01)$ から厳密無発散ボーラス流速 $\mathbf{u}_{\mathrm{GM}} = -\partial_z(\kappa_{\mathrm{GM}}\mathbf{S})$（$\kappa_{\mathrm{GM}} = 800\ \mathrm{m^2\ s^{-1}}$）と、ボロノイ双対面積射影を用いた小勾配フル Redi 拡散テンソル $\mathbf{F}_{\mathrm{Redi}} = -\kappa_{\mathrm{Redi}} [[1, \mathbf{S}], [\mathbf{S}^T, |\mathbf{S}|^2]] \nabla q$（$\kappa_{\mathrm{Redi}} = 1000\ \mathrm{m^2\ s^{-1}}$）を構成し、z 面水平拡散の偽鉛直混合（Veronis 効果）を抑制しつつ鉛直対角項 $\kappa_{\mathrm{Redi}}|\mathbf{S}|^2$ を陰的三重対角ソルバへ無条件安定に組み込む。
4. **KPP 表層境界層・シア不安定混合・海底地形性内部潮汐混合・短波透過吸収**（Large, McWilliams, & Doney, 1994, *Rev. Geophys.*, 32, Eqs. 19–28; Pacanowski & Philander, 1981; Paulson & Simpson, 1977; St. Laurent et al., 2002; Simmons et al., 2004, *Ocean Modelling*, 6, Eqs. 3–6; Huang, 1993）:
   - バルク・リチャードソン数 $\mathrm{Ri}_b(z) = \mathrm{Ri}_c = 0.30$ から KPP 境界層深さ $h_{\mathrm{BL}}$（安定浮力 $B_0 > 0$ 時はエクマン・Monin–Obukhov 制限付き）を診断し、3次形状関数 $K_{v,\mathrm{KPP}} = h_{\mathrm{BL}} w_s(\sigma)\sigma(1-\sigma)^2$ および海面冷却・ブライン排出時（$B_0 < 0$）の厳密保存型非局所対流輸送 $F_{\mathrm{nl},q} = -C_s \sigma(1-\sigma)^2 \overline{w'q'}_0$（$C_s = 6.33$）を解く。
   - 内部域では勾配リチャードソン数依存シア混合 $K_{v,\mathrm{shear}} = \nu_0[1 - (\min(\mathrm{Ri}_g, \mathrm{Ri}_0)/\mathrm{Ri}_0)^2]^3$（$\nu_0 = 5\times 10^{-3}\ \mathrm{m^2\ s^{-1}}$、$\mathrm{Ri}_0 = 0.70$）および KPP＋シア依存の陰的鉛直渦粘性 $A_v$ を適用する。
   - `Earth2014 BED2014` 海底地形勾配 $|\nabla H|$ に応じた海底強化型の内部潮汐混合 $K_{v,\mathrm{tide}} = 0.0667\, E_{\mathrm{tide}}(|\nabla H|)\exp[-(H - z)/\zeta_{\mathrm{tide}}]/[\rho_0 N^2 \zeta_{\mathrm{tide}}(1 - e^{-H/\zeta_{\mathrm{tide}}})]$（$\zeta_{\mathrm{tide}} = 500\ \mathrm{m}$、$E_0 = 8\times 10^{-3}\ \mathrm{W\ m^{-2}}$）を Bryan–Lewis 背景床（$1.5\times 10^{-5}\text{–}3\times 10^{-5}\ \mathrm{m^2\ s^{-1}}$）へ加算し、不安定成層（$\partial_z\rho' < 0$）では瞬間対流調節（$K_v \to 1.0\ \mathrm{m^2\ s^{-1}}$）を行う。
   - Jerlov Type I の2指数短波透過則 $I(z) = 0.58 e^{-z/0.35} + 0.42 e^{-z/23.0}$ により開水面吸収日射の一部を第0層から亜表層へ熱量厳密保存のまま再配分し、$(E - P - R_{\mathrm{river}} - R_{\mathrm{glacier}})$＋海氷凍結ブライン排出／融解淡水の保存的仮想塩分フラックスと結合する。
5. **海氷の力学的移流（ドリフト）**（`enable_seaice_drift = true`; Hibler, 1979; Leppäranta, 2011）: 風応力＋海洋表層流・コリオリ偏向角 $20^\circ$・厚氷リジング抵抗 $[1 + (h_i / h_{\mathrm{ridge}})^2]^{-1}$ により、海陸境界フラックスゼロを保ちつつ `m_seaice` と `swe_seaice` を保存移流する。

### 10.4 MAML + ICA サブグリッド 1:1 結合とケッペン・ガイガー診断 (`src/aoba/coupled_model.py`, `src/aoba/koppen.py`)

ホスト各セル $c$ の直下に $N_y \times N_x$ 個の CRM サブコラム $(j, i)$ が存在し、さらにその直下に各サブコラム 1対1 で独立した地表面タイル状態 `sfc_sub`（土壌5層・積雪・樹冠・Two-Leaf EEO 植生履歴・海洋混合層状態）を保持する **MAML (Multiple Atmosphere Multiple Land) + ICA (Independent Column Approximation) 結合**（Lee & Khairoutdinov, 2015; Lin et al., 2023）を構成する。結合ステップ $\Delta t_{\mathrm{host}}$ ごとに以下の順で進行する：
1. **ICA 4バンド放射**: 全サブコラム $[N_z, N_{\mathrm{cells}} \cdot N_y N_x]$ ごとの雲量・水分・地表面温度・アルベドを用いてサブグリッド単位で放射加熱率・地表下向き放射を算出（チャンク分割対応）。
2. **MAML 地表面結合＋ホストエクマン海洋同期**: 各サブコラム固有の最下層風・気温・比湿・降水から独立に地表面状態・Two-Leaf 光合成・蒸発散を更新し、海洋タイルについては細胞平均混合層状態 $(T_1, M_{\mathrm{ice}})$ へ集約した上で SCVT エクマン海洋ステップを実行し、温度変化量 $\Delta T_1, \Delta M_{\mathrm{ice}}$ をサブグリッド海洋タイルへフィードバックする。
3. **ホスト SCVT 大気積分と 1:1 CRM 超パラメータ化**: ホスト SCVT 力学コアを HEVI＋FCT＋山岳ドラッグで進め、得られた大規模乾燥力学・移流強制 $\left(\partial_t \theta_m, \partial_t q_n\right)_{\mathrm{LS}}$ を全 CRM サブコラムへ注入して `step_crm_physics_and_dynamics`（MYNN 2.5 + 3D RK3-HEVI + KR03 雲微物理 + 3D FCT + サブサイクリング）を実行し、水平平均値でホスト大気場を同期させる。
4. **ケッペン・ガイガー気候帯自律診断**: サブグリッド日値（日積算降水・気温・日最低気温）から年次 Two-Leaf 順化・LAI・$S_R$・$h_c$ と月別気候値 $(T_m, P_m)$ を集計し、Beck et al. (2018, 2023, Table 1) の厳密30クラス分類（`Af`–`EF`）をオンライン診断する。

---

## 11. 動的／平衡加速 SIA 氷床モデルと保存的気圧鉛直再補間

自転周期改変実験や古気候・温暖化平衡実験など、千年〜万年規模の氷床応答を現実的な計算時間で模擬し、かつ氷床厚変化に伴う地形高度変化を大気格子へ無矛盾に反映するためのオプトイン型結合モジュール（`src/aoba/icesheet.py`）である。`Earth2014.BED2014`（岩盤標高 $b_{\mathrm{init}}$）および `ICE2014`（初期氷厚 $H_0$）を SCVT 格子へ集計して初期化される。

### 11.1 気候表面質量収支 (SMB) と EMA 平衡平滑フィルタ

各タイムステップにおいて、陸域・接地氷床セルのサブグリッド平均正味氷質量変化率 $\mathrm{SMB}(t)\ [\mathrm{kg\ m^{-2}\ s^{-1}}]$ を `step_surface` より受け取る。積雪域・氷床域での降雪流入、積雪・氷面昇華、積雪融解および氷面融解ポテンシャル（$T_{\mathrm{skin}} > T_0$ 時の過剰エネルギーによる氷融解 $\dot{m}_{\mathrm{ice}} = E_{\mathrm{excess}} / L_{f0}$、ただし現在厚が $H_{\mathrm{ice}} = 0$ で積雪も無い領域における負の融解は正味消耗 0 として扱う）を1気候年分積算し、年平均気候表面質量収支 $\overline{\mathrm{SMB}}_y$ を得る。大気変動の年々ノイズを抑えて平衡氷床形態を滑らかに抽出するため、時定数 $\tau_{\mathrm{EMA}}$（既定 `icesheet_ema_years = 5.0 yr`）の指数移動平均（EMA）を適用する：
$$
\widetilde{\mathrm{SMB}}_y = (1 - \alpha_{\mathrm{EMA}})\,\widetilde{\mathrm{SMB}}_{y-1} + \alpha_{\mathrm{EMA}}\, \overline{\mathrm{SMB}}_y, \qquad \alpha_{\mathrm{EMA}} = 1 - \exp\!\left(-\frac{\Delta t_{\mathrm{clim,yr}}}{\tau_{\mathrm{EMA}}}\right),
$$
これにより氷厚換算質量収支源項 $b_{\mathrm{ice}} = \widetilde{\mathrm{SMB}}_y / \rho_{\mathrm{ice}}\ [\mathrm{m\ s^{-1}}]$ が確定する。

### 11.2 Glen 流動則 ($n = 3$) に基づく SCVT 上流有限体積 SIA 方程式と加速制御

浅氷近似（Shallow Ice Approximation: SIA; Vialov, 1958; Cuffey & Paterson, 2010; Bueler et al., 2005）により、氷表面高度 $s = b + H$ の勾配 $|\nabla s|$ によって駆動される2次元等温 Glen 流動（べき指数 $n = 3$、流動軟度係数 $A_{\mathrm{Glen}} = 1.0\times 10^{-16}\ \mathrm{Pa^{-3}\ yr^{-1}}$、氷密度 $\rho_{\mathrm{ice}} = 917\ \mathrm{kg\ m^{-3}}$）は、非線形拡散係数 $D_{\mathrm{SIA}}$ を持つ質量保存則として表される：
$$
\frac{\partial H}{\partial t_{\mathrm{ice}}} = \nabla_{\mathrm{SCVT}} \cdot \left(D_{\mathrm{SIA}} \nabla s\right) + b_{\mathrm{ice}}, \qquad D_{\mathrm{SIA}} = \Gamma_{\mathrm{SIA}}\, H^{5}\, |\nabla s|^{2}, \qquad \Gamma_{\mathrm{SIA}} = \frac{2 A_{\mathrm{Glen}}}{n + 2} (\rho_{\mathrm{ice}} g)^3 = 0.4 A_{\mathrm{Glen}} (\rho_{\mathrm{ice}} g)^3.
$$
SCVT ボロノイ辺 $e = (c_0 \to c_1)$ 上の離散化では、正値保存と数値安定性のため Mahaffy (1976) の **Type-I 上流評価**を用い、表面勾配 $\frac{\partial s}{\partial n}_e = (s_{c_1} - s_{c_0})/d_{c,e}$ の向きに応じて上流側セルの氷厚 $H_{\mathrm{up}, e}$ で拡散係数 $D_e = \min(\Gamma_{\mathrm{SIA}} H_{\mathrm{up}, e}^5 (\partial_n s_e)^2,\ D_{\mathrm{cap}})$ を評価し、`compute_scvt_divergence` で面積加重保存フラックス発散を閉じる。陽的時間刻み $\Delta t_{\mathrm{sub}}$ は最大拡散係数から CFL 制限（$D_{\mathrm{sub}} \Delta t_{\mathrm{sub}} / \Delta x_{\min}^2 \le 0.12$）で自動サブサイクリングする。海セル ($f_{\mathrm{ocn}} \ge 0.5$) へ流れ込んだ氷厚は氷山流出（calving）として取り除き、陸氷は $H \in [0, 6000\ \mathrm{m}]$ に保たれる。

スピンアップ時の高速平衡到達と本計算時の物理時間同期を両立するため、実効氷床時間加速率 $\alpha_{\mathrm{eff}}(t)$ を導入し $\Delta t_{\mathrm{ice}} = \alpha_{\mathrm{eff}}(t)\, \Delta t_{\mathrm{clim}}$ で積分する：
$$
\alpha_{\mathrm{eff}}(t) = \begin{cases} \alpha_0 & (\tau_{\mathrm{spin}} \le 0), \\ 1 + (\alpha_0 - 1)\exp\!\left(-\dfrac{t_{\mathrm{clim}}}{\tau_{\mathrm{spin}}}\right) & (\tau_{\mathrm{spin}} > 0), \end{cases}
$$
既定設定は $\alpha_0 = 100.0$、減衰時定数 $\tau_{\mathrm{spin}} = 0.0$（設定によりスピンアップ初期のみ100倍速とし指数的に実時間 $\alpha=1$ へ収束可能）。

### 11.3 緩和型リソスタシー・アイソスタシーと地形平滑化・土地被覆遷移

氷床荷重に伴う上部マントル（密度 $\rho_m = 3300\ \mathrm{kg\ m^{-3}}$、均衡比 $\rho_{\mathrm{ice}} / \rho_m \approx 0.2779$）の盤面沈降・隆起応答を、氷河時間時定数 $\tau_{\mathrm{iso}}$（既定 $1000\ \mathrm{yr}$、$\tau_{\mathrm{iso}} \le 0$ で瞬間均衡）のもとで更新する（Le Meur & Huybrechts, 1996; Pollard & DeConto, 2012）：
$$
b_{\mathrm{eq}} = b_0 - \frac{\rho_{\mathrm{ice}}}{\rho_m} H, \qquad b^{\mathrm{new}} = b + \left[1 - \exp\!\left(-\frac{\Delta t_{\mathrm{ice}}}{\tau_{\mathrm{iso}}}\right)\right](b_{\mathrm{eq}} - b).
$$
新たな地表面標高は陸上で $z_{\mathrm{topo}}^{\mathrm{new}} = b^{\mathrm{new}} + H$（海上で 0）となり、さらに Klemp (2011) 整合の 2パス近傍保存ラプラシアン平滑（$z \leftarrow 0.75 z + 0.25 \bar{z}_{\mathrm{nb}}$）を施した平滑標高 $z_{\mathrm{topo,smooth}}^{\mathrm{new}}$ を生成する。接地氷厚が閾値 $H_{\mathrm{crit}} = 5.0\ \mathrm{m}$ を横切ると、セル・サブグリッドの地表面種別が `SURF_LAND <-> SURF_ICESHEET` 間で動的に切り替わり、氷床が後退して新露出した陸域には開拓草本（pioneer vegetation: $\mathrm{LAI}=0.25, h_c=0.3\ \mathrm{m}$）を播種して EEO 生態系の自律侵入を開始させる。

### 11.4 保存的気圧・対数線形補間による大気および埋め込み CRM の鉛直再構築

表面地形変化 $\max |z_{\mathrm{topo,smooth}}^{\mathrm{new}} - z_{w,0}^{\mathrm{old}}| \ge 0.05\ \mathrm{m}$ が生じた際、`compute_stf_heights` によりホストと全サブグリッド CRM の新鉛直座標 $z_w^{\mathrm{new}}, z_c^{\mathrm{new}}$ を構築したのち、以下の順序でホスト (`GlobalAtmosState`) および全 CRM 列 (`CRMState`) を再配置する：
1. **静水圧整合の外挿と対数気圧補間**: 旧座標領域内部（$z_c^{\mathrm{new}} \ge z_{c,0}^{\mathrm{old}}$）では $\ln p$、$\theta_m$、$(q_n, q_p, q_v, q_c, q_i)$ および速度場 $(u, v, w, q^2)$ を区分線形補間する。氷床融解に伴い新たな表面高度が旧地表面より低くなった領域（$z_c^{\mathrm{new}} < z_{c,0}^{\mathrm{old}}$）については、非物理的な定数外挿を避け、静水圧バランスに基づく測高公式で気圧を下向き外挿する：
   $$
   p(z_c^{\mathrm{new}}) = p_0^{\mathrm{old}} \left[1 - \frac{g (z_c^{\mathrm{new}} - z_{c,0}^{\mathrm{old}})}{c_{pd} \theta_{m,0}^{\mathrm{old}} \Pi_0^{\mathrm{old}}}\right]^{c_{pd}/R_d}.
   $$
2. **圧縮性状態方程式による密度再診断**: 補間・外挿された新気圧 $p^{\mathrm{new}}$ と $\theta_m^{\mathrm{new}}$ から乾燥空気密度を $\rho_d^{\mathrm{new}} = p^{\mathrm{new}} / (R_d \theta_m^{\mathrm{new}} \Pi^{\mathrm{new}})$ と厳密に再算出し、圧力・浮力バランスの擾乱を最小限に抑えたまま統合を継続する。またホスト大気辺法線風速 $u_e$ は左右セルの新層高度平均を用いて再補間する。

### 11.5 マップフリー 3D 予報大気オゾン光化学・乾性沈着 (`src/aoba/ozone_chemistry.py`)

`o3_mode = "prognostic"` を指定すると、処方オゾン気候値マップを完全に排し、3次元オゾン体積混合比 $\chi_{\mathrm{O}_3}(z, c)$（`[Nz, N_cells]`）をホスト SCVT Zalesak FCT 移流 (`fct_advect_scvt_batch`) により保存移流しつつ短波 UV/Chappuis・9.6 μm 長波放射 (`radiation.py`) と双方向結合する。局所化学収支は奇酸素 Riccati 方程式で記述される：
$$
\frac{d\chi_{\mathrm{O}_3}}{dt} = P - L_1 \chi_{\mathrm{O}_3} - L_2 \chi_{\mathrm{O}_3}^2.
$$
生成項 $P = 2 J_{\mathrm{O}_2}(s_{\mathrm{O}_2}, s_{\mathrm{O}_3}) f_{\mathrm{O}_2} + P_{\mathrm{trop}}(\mathrm{CH}_4, \mathrm{LNO}_x, F_{\mathrm{fire}})$ は、球殻大気路程 $m(\mu_0) = 35 / \sqrt{1224\mu_0^2 + 1}$ に沿う上空予報 $\mathrm{O}_3$ の UV 自己遮蔽を含む 2 バンド $\mathrm{O}_2$ 光解離（Herzberg 連続帯＋Schumann–Runge 帯；Brasseur & Solomon, 2005）と、背景 $\mathrm{CH}_4$・埋め込み CRM 落雷 `LNOx`・森林火災前駆体による対流圏生成からなる。二次消失 $L_2 = 2 k_3(T)([\mathrm{O}]/[\mathrm{O}_3]) M$ は JPL Arrhenius 速度定数に基づく Chapman 反応（Burkholder et al., 2019）であり、一次消失 $L_1$ は $\mathrm{N}_2\mathrm{O}$ 由来 $\mathrm{NO}_x$・予報水蒸気 $q_v$ 由来 $\mathrm{HO}_x$ の成層圏触媒破壊、$T < 195\ \mathrm{K}$ かつ日照下で `(o3_ecl_ppbv / 3.5)^2` に比例する極成層圏雲 (PSC) 不均一反応（Cariolle & Teyssèdre, 2007）、対流圏 $\mathrm{O}(^1\mathrm{D})+\mathrm{H}_2\mathrm{O}$・$\mathrm{HO}_2$ 消失、および創発 EEO 植生 `LAI`・積雪海氷に連動した地表面乾性沈着（Wesely, 1989）の和である。時間積分には桁落ちのない正平衡根 $\chi_+ = 2P / (L_1 + \sqrt{L_1^2 + 4 P L_2})$ と厳密指数緩和 $\chi^{n+1} = \chi^n + (\chi_+ - \chi^n)[-\mathrm{expm1}(-(L_1 + L_2(\chi^n + \chi_+))\Delta t)]$ を用い、無条件安定・非負保存を保証する。

### 11.6 独立高解像度海洋グリッド結合 (`src/aoba/ocean_coupler.py`)

`ocean_glevel > 0`（大気 `glevel` と異なる値）を指定すると、3D OGCM・FlexPFT 海洋生物地球化学・EVP–ITD 海氷力学を独立の高解像度正二十面体 SCVT メッシュ（`ocn_mesh`；例：渦許容 `Glevel-7` ~60 km や渦解像 `Glevel-8` ~30 km）上で実行できる。球面 Voronoi 包含関係 $\mathrm{parent}[j] = \arg\max_i (\mathbf{x}_{\mathrm{ocn},j} \cdot \mathbf{x}_{\mathrm{atm},i})$ に基づく `OceanAtmosCoupler` は、示強外力（風応力・10 m 風速・PAR）を細格子へ写像し、流出・淡水フラックスは海岸線マスク差を跨いで全球積分を厳密保存する。大気地表面ステップの熱・海氷増分 $(\Delta T_{\mathrm{ml}}, \Delta m_{\mathrm{ice}}, \Delta \mathrm{SWE}_{\mathrm{ice}})$ を潜熱凍結保存つきで細格子 `ocn_sfc` へ直接適用することで SST 前線や中規模渦の粗格子拡散を防ぎ、更新後の SST・海氷・表層クロロフィルを厳密な海洋面積重み平均で大気格子へ返す。

### 11.7 スイッチ選択式「万能マップフリー ESM」拡張群

高度な ESM 物理・生物地球化学結合はすべて `config/aoba_default.toml` の `[forcing]` スイッチ／セレクタにより自在に切替可能である：
- **6 種 2 モーメント雲微物理** (`microphysics_scheme = "six_class_2m"`, `src/aoba/microphysics_6class.py`; Morrison et al., 2005; Seifert & Beheng, 2006)：埋め込み CRM 上で雨 (`q_r, n_r`)・雪 (`q_s, n_s`)・霰 (`q_g, n_g`) を独立予報し、真の予報霰 `q_g` を CRM 非誘導帯電落雷へ直接供給する。
- **29 バンド／111 g-pt MSTRN11 級放射 ＆ 公式 HITRAN/RRTMGP（240 g-pt）＋ 3D TICA** (`rad_scheme = "ckd_hd" | "rrtmgp"`, `enable_rad_3d_tica`, `src/aoba/radiation_hd.py`, `src/aoba/radiation_rrtmgp.py`; Pincus et al., 2019; Sekiguchi & Nakajima, 2008; Chou et al., 1999; Várnai & Davies, 1999)：`"ckd_hd"` は自己完結 111 g-pt＋Chou 長波雲散乱＋5 種エアロゾル直接放射効果を提供し、`"rrtmgp"` は公式 BSD-3 HITRAN＋MT_CKD 4.1 気体光学テーブル（LW 128 g-pt＋SW 112 g-pt＝計 240 g-pt）を自動取得して GPU バッチ 3 重線形 `(T, ln p, eta)` 2 気体結合補間で高精度に解く。
- **SNICAR 雪氷 BC/ダスト暗化・陸海 Fe/`LNOx` 結合・森林 BVOC→SOA** (`enable_snow_bc_darkening`, `enable_bgc_fe_lnox_coupling`, `enable_bvoc_soa`; Flanner et al., 2007; Jickells et al., 2005; Guenther et al., 2012)：BC・ダスト沈着による雪氷可視アルベド低下、ダスト可溶 Fe と落雷 `LNOx` の FlexPFT 海洋施肥、および EEO 植生由来 BVOC→SOA→CCN 循環を閉じる。
- **予報自然 $\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$・珪酸塩風化サーモスタット・ミランコビッチ軌道** (`enable_prognostic_ch4_n2o`, `enable_silicate_weathering`, `enable_milankovitch_orbit`; `src/aoba/trace_gas_cycle.py`, `src/aoba/planetary_geochem.py`; Prather et al., 2012; Walker et al., 1981; Berger, 1978)：湿地・火災 $\mathrm{CH}_4$–$\mathrm{OH}$、土壌・海洋 $\mathrm{N}_2\mathrm{O}$–UV、岩石風化 $\mathrm{CO}_2$–アルカリ度収支、および解析軌道進化を統合する。
- **海流相対風速フラックス・渦許容 OGCM 物理** (`enable_current_rel_wind`, `enable_ogcm_scale_aware_gm`, `enable_ogcm_submeso`, `enable_ogcm_tidal_mixing`; Renault et al., 2016; Hallberg, 2013; Fox-Kemper et al., 2008; St. Laurent et al., 2002)：3D CRM（`crm_ny > 1, crm_nx > 1`）または `enable_superparam = false` の場合に限り海面相対風速 $\mathbf{u}_a - \mathbf{u}_o$ で海面フラックスを計算し、解像度適応 GM・サブメソスケール再成層・海底粗さ潮汐混合と連携させる。

---

## 12. 単一 TOML 設定管理・出力仕様・GUI コンフィギュレータ・全パラメータ一覧

### 12.1 閉ループ万能 ESM 拡張（`prognostic_co2.py`, `ocean_bgc.py`, `volcanic_solar.py`）
- **排出駆動・完全双方向の大気 $\mathrm{CO}_2$ 予報（`co2_mode = "prognostic"`, `enable_co2_3d_transport`）**: 陸域 NBP・海洋 $\mathrm{fgCO}_2$・ケイ酸塩風化・火山・人為排出の収支 $\mathrm{d}C_{\mathrm{CO}_2}/\mathrm{d}t = (E_{\mathrm{anthro}} + E_{\mathrm{volc}} - \mathrm{NBP}_{\mathrm{land}} - F_{\mathrm{ocn}} - F_{\mathrm{weath}}) / 2.124\ \mathrm{PgC\ ppmv}^{-1}$ を倍精度で厳密保存し、オプションで 3 次元 SCVT FCT 移流・PBL 混合によりキーリング曲線と南北勾配を再現する。
- **海洋溶存酸素 ($\mathrm{O}_2$)・貧酸素水塊 (OMZ)・海洋脱窒（`enable_ocean_o2_omz`）**: Garcia & Gordon (1992) の酸素溶解度、Wanninkhof (2014) の大気海洋交換、$R_{\mathrm{O}_2:\mathrm{C}} = 138/106$ の光合成・再無機化、および貧酸素域（$\mathrm{O}_2 < 20\ \mathrm{mmol\ m}^{-3}$）での嫌気的脱窒を解き、海洋 $\mathrm{N}_2\mathrm{O}$ 放出へ結合する。
- **成層圏火山性エアロゾル・太陽活動 11 年周期（`enable_volcanic_aerosols`, `enable_solar_cycle`）**: 厳密な 2 段 Bateman 解析解による $\mathrm{SO}_2 \to \mathrm{H}_2\mathrm{SO}_4$ 酸化（$\tau_{\mathrm{ox}} = 35\ \mathrm{日}$）および沈降（$\tau_{\mathrm{set}} = 1.1\ \mathrm{年}$）と SCVT ラプラシアン拡散を放射（SW 散乱・LW 吸収）および成層圏不均一オゾン破壊へ結合し、11 年 Schwabe 周期の TSI・UV 光解離変調を加える。
- **MAML サブグリッド山岳微気候・動的湿地湖沼・波浪/ラングミュア/海塩飛沫結合（`maml_land_subgrid.py`, `ocean_waves.py`）**: `Earth2014` の `sigma_topo` から親格子平均ゼロのサブグリッド標高減率・斜面日射を与える微気候ダウンスケーリング（`enable_maml_subgrid_topo`）、地形平坦度と河川流量から冠水率を動的診断して蒸発・湿地 $\mathrm{CH}_4$ へ繋ぐ湿地・湖沼モデル（`enable_maml_wetlands_lakes`）、および有義波高 $H_s$・周期 $T_p$ 予報から Stokes ドリフトのラングミュア KPP 混合強化・波齢依存粗度・強風時海塩飛沫エンタルピーフラックスを結合する波浪モジュール（`enable_wave_coupling`）。

### 12.1 一元テキスト管理アーキテクチャと実行・GUI ツール (`src/aoba/params.py`, `src/aoba/co2_forcing.py`, `src/aoba/ozone_forcing.py`, `src/aoba/gravity_wave.py`, `scripts/aoba_config_gui.py`)

モデルの再現性と透明性を保証するため、すべての格子設定（成層圏拡張標準 `nz = 44, z_top = 48 km`、低トップ 24 km 切替可）・強制外力・非地形性重力波・動的氷床設定・出力ストリーム・物理定数（**14セクション・全256項目**）を単一の人間可読な TOML ファイル [`config/aoba_default.toml`](../config/aoba_default.toml) に集約管理する。大気 $\mathrm{CO_2}$ 濃度は `co2_mode` により **固定値 (`constant`)・シナリオ補間 (`scenario`)・独自 CSV (`csv`)** の3通りを選択でき、`scenario` 時はモデルの連続暦年（負の BCE 年を含む）に応じて同梱テーブル `src/aoba/assets/co2_scenarios.csv`（CMIP5 RCP・CMIP6 historical/SSP・南極氷床コア80万年復元・NOAA GML 接続）または `scripts/fetch_co2_data.py`／AOBA Studio から随時更新した最新キャッシュ `data/forcing/co2/co2_latest.csv` を線形補間する。大気オゾン $\mathrm{O_3}$ も同様に `o3_mode` により **解析的2D緯度×季節 (`latitude_seasonal`、既定)・1D RCEMIP (`rcemip`; Wing et al., 2018)・外部帯状平均気候値 (`climatology`; `scripts/fetch_ozone_data.py` で CMIP6/CMIP7 input4MIPs から取得し、高空モデルトップ拡張時はテーブル上端より上で $p^{g_2}$ 減衰外挿)** を選択でき、`o3_lw_coupling = true` で 9.6 μm 窓領域長波吸収およびオーバーヘッド $\mathrm{O_3}$・成層圏 15 μm $\mathrm{CO_2}$ 冷却を有効化する。
- **ランタイム同期・ハードウェア自動チューニング (`src/aoba/params.py`, `src/aoba/runtime.py`)**: `load_aoba_toml()` で読み込んだ設定辞書に対し `build_configs_from_dict(cfg_dict, apply_modules=True)` を呼ぶと、`(AOBAConfig, OutputConfig)` インスタンスを生成すると同時に、`_MODULE_PARAM_MAP` に登録された `src/aoba/*` モジュールの全定数をライブ更新し、各モジュールの `recompute_derived_constants()`（`microphysics`, `turbulence`, `ocean_ekman`, `icesheet` 内の Metal 定数配列や導出係数を含む）を再計算する。さらに `detect_hardware_profile()` / `compute_optimal_chunks()` / `apply_runtime_tuning()` により実行マシンの GPU コア数・利用可能メモリ・P-core 数を自動検出し、メモリ予算 $B = 0.45\, M_{\mathrm{dev}}$ の範囲内で解像度を変えずに最速 1 パス／安全分割となる `crm_col_chunk`・`rad_col_chunk` と CPU スレッド数を自動適用する。
- **Web GUI 統合ワークベンチ (`scripts/aoba_config_gui.py` / AOBA Studio)**: 標準ライブラリ（`http.server`）のみで動作するゼロ依存ローカル Web アプリ（`python scripts/aoba_config_gui.py --port 8765`）を備え、モデル設定から実行起動、プロセス監視、CPU リアルタイム可視化まで一気通貫で提供する：
  1. `glevel`（0〜6）の変更に連動してリアルタイム描画される**正二十面体 SCVT Voronoi 球面メッシュ可視化図 ＋ 1ホストセル対埋め込み CRM 領域スケール比較図 ＋ ストレッチ鉛直層・スポンジ層図**。
  2. 自転半減実験 (`0.5×Ω`)・LGM・`4×CO₂` プリセット、**CO₂/O₃ シナリオ選択・推移グラフ・NOAA/NCEI/RCMIP/ESGF 最新データ取得ボタン**、**カレンダー／固定日射設定**、および**氷床加速率 $\alpha_{\mathrm{eff}}(t)$／EMA 重み減衰曲線**プレビュー。
  3. 出力ファイル種別・間隔設定、予想年間 Zarr/リスタート容量試算、および256パラメータ横断検索・TOML 保存。
  4. **実行コントロール・ライブ監視・CPU リアルタイム日平均 3地球プレビュー**: GUI 上からの計算実行・停止、稼働中外部ジョブ（`run.log` / `daily_global.csv` / カレンダー日付 `YYYY-MM-DD`）の監視に加え、GPU 計算を一切阻害しない待機 CPU 上の `cKDTree` 球面再格子化エンジンにより、**日平均 `T2`（地表付近気温 $0.35 T_{\mathrm{sfc}} + 0.65 T_{a,z_1}$）・日平均降水量 `Precip [mm/d]`・`LAI`** の3つの正射図法地球儀を「1モデル日＝1周（$360^\circ / T_{\mathrm{wall,day}}$）」の実時間同期自転または任意速度でリアルタイム可視化する（`live_preview.npz`, `snapshots.zarr`, `monthly.zarr` に対応）。

### 12.2 出力ファイル種別・平均値 vs 瞬時値仕様 (`OutputConfig`, `src/aoba/io_zarr.py`)

`[output]` セクションにより、CF-1.11 / UGRID-1.0 準拠の Zarr v3 ストアおよび日次 CSV・リスタートの粒度と変数群（**2D表面**・**3D大気**・**CRM サブグリッド 2D**）を柔軟に選択できる。

| 出力ファイル | 内容種別 | TOML 制御キー・時間間隔 | 含まれる主な物理変数 |
| :--- | :--- | :--- | :--- |
| `mesh.zarr` | 静的格子・地形 | 初期化時に1回出力 | セル緯度経度・面積・`z_topo`・`z_bed`・`sigma_topo`・`surf_type`・`koppen` 等 |
| `daily_global.csv` | **時間平均** (全球日別) | `daily_csv_enabled`, `daily_csv_interval_days` (既定 `1.0` 日) | $T_s$, $\mathrm{OLR}$, $\mathrm{ASR}$, $\mathrm{NetTOA}$, 降水, $\mathrm{GPP}$, $\mathrm{LAI}$, 海氷面積, 計算速度 |
| `monthly.zarr` | **時間平均** (月/季節/年) | `monthly_mean_enabled`, `monthly_mean_periods_per_year` (既定 `12`), `monthly_mean_sample_hours` (既定 `6.0` h), `monthly_mean_include_{2d,3d,subgrid_2d}` | **2D**: `t_ocean_ml`, `t_canopy`, `m_seaice`, `swe_land`, `w_root`, `lai`, `precip_mm_day`, `olr`, `asr`, `evap_mm_day`, `gpp`, (`h_ice`, `z_topo_dyn`), `koppen_code/major` / **3D**: `temp`, `pres`, `q_v`, `u_zonal`, `v_merid` / **SG2D**: `lwp_sub`, `iwp_sub`, `w_max_sub`, `precip_mm_day_sub`, `sw_dn_sub`, `lw_dn_sub`, `olr_sub`, `shf_sub`, `evap_mm_day_sub`, `gpp_sub`, `t_sfc_sub`, `m_seaice_sub`, `swe_land_sub`, `w_root_sub`, `lai_sub` |
| `snapshots.zarr` | **瞬時値** (スナップショット) | `snapshots_enabled`, `snapshots_periods_per_year` (既定 `12`), `snapshots_include_{2d,3d,subgrid_2d}` | 上記2D (`h_canopy`, `vcmax25_c3`, `f_c4`, `f_evg` を含む)・3D変数 (`temp`, `pres`, `q_v`, `u_zonal`, `v_merid`) および上記サブグリッド2D瞬時場 (`*_sub`) |
| `restart.pt` / `restart_mid.pt` | チェックポイント | `restart_interval_years` (既定 `1` 年), `restart_monthly` (既定 `true`), `zarr_dir_partition` | 完全な `GlobalAtmosState`, `CRMState`, `SurfaceState` (`sfc` & `sfc_sub`), `EkmanOceanState`, `IceSheetState`, `ClimateAccumulator`。`restart_monthly=true` 時は毎月末に `restart_mid.pt` を自動保存し、年内途中から Zarr/CSV 欠損なく決定論的再開可能 |

### 12.3 全モデル設定・物理パラメータレジストリ（全14セクション・256項目）

表 12.1 に [`config/aoba_default.toml`](../config/aoba_default.toml) に定義された全256パラメータ（既定は CMIP6 1850 `piControl` ＋ G5 8×1 44層 ＋ Correlated-k ＋ 3D OGCM ＋ 2モーメント微物理）のセクション別完全リストを示す。

#### `[grid]` — 水平・鉛直・サブグリッド CRM 格子設定（16項目）
| TOML キー | 標準値 | 説明・単位 |
| :--- | :--- | :--- |
| `glevel` | `5` | SCVT 再帰分割レベル（$N_{\mathrm{cells}} = 10\cdot 4^g + 2$: G3=642, G4=2562, G5=10242） |
| `nz` | `44` | ホスト・CRM 共通鉛直層数（対流圏〜成層圏界面；低トップ時は `32`） |
| `z_top` | `48000.0` | モデル上端高度 [m]（成層圏界面 ~0.85 hPa；低トップ時は `24000.0`） |
| `dz_min` | `60.0` | 地表最下層厚 [m] |
| `z_sponge_base` | `36000.0` | レイリー減衰スポンジ層底面高度 [m]（低トップ時は `17500.0`） |
| `tau_sponge_min` | `300.0` | モデル上端最小スポンジ減衰時定数 [s] |
| `terrain_scale_exp` | `1.5` | Klemp STF 減衰指数 $p$ [-] |
| `crm_nx`, `crm_ny`, `crm_dx` | `8`, `1`, `4000.0` | 埋め込み CRM の $x, y$ 格子数および水平格子幅 [m] |
| `dt_host` | `720.0` | ホスト大気・物理結合時間刻み [s] |
| `n_host_dyn_substeps`, `n_crm_dyn_substeps` | `1`, `12` | ホストおよび CRM の1結合ステップあたり音響・力学サブステップ数 |
| `rad_stride` | `5` | 放射計算の呼出し間隔（ステップ数） |
| `crm_col_chunk`, `rad_col_chunk` | `16384`, `32768` | GPU/CPU メモリ制御用バッチチャンクサイズ [columns] |

#### `[forcing]`・`[icesheet]`・`[output]` — 外力・氷床制御・出力管理（60項目）
| セクション | キー ＝ 標準値 |
| :--- | :--- |
| `[forcing]` (35) | `co2_ppmv = 284.3`, `co2_mode = "constant"`, `co2_scenario = "obs_historical_noaa"`, `co2_csv_path = ""`, `ch4_ppmv = 0.80825`, `n2o_ppmv = 0.27302`, `rad_scheme = "ckd"`, `o3_mode = "latitude_seasonal"`, `o3_file_path = ""`, `o3_lw_coupling = true`, `enable_nonoro_gwd = true`, `solar_constant = 1361.0`, `omega = 7.292115e-5`, `obliquity_deg = 23.44`, `eccentricity = 0.0167`, `perihelion_deg = 283.0`, `start_date = "2000-03-20"`, `calendar_type = "gregorian"`, `insolation_mode = "seasonal"`, `fixed_insolation_date = "2000-03-20"`, `diurnal_cycle = true`, `use_earth2014 = true`, `earth2014_dir = "data/boundary/earth2014_5min"`, `earth2014_stride = 2`, `enable_superparam = true`, `enable_dust = true`, `dust_emis_factor = 1.5e-5`, `h_ml_ocean = 50.0`, `ocean_mode = "ogcm_3d"`, `ocean_nz = 18`, `enable_seaice_drift = true`, `re_liq_um = 10.0`, `re_ice_um = 48.0`, `cloud_sw_inhomog = 0.65`, `microphysics_scheme = "two_moment"` |
| `[icesheet]` (5) | `enable_dynamic_icesheet = false`, `icesheet_ema_years = 5.0`, `icesheet_accel_factor = 100.0`, `icesheet_accel_spinup_years = 0.0`, `icesheet_isostasy_tau_yr = 1000.0` |
| `[output]` (20) | `out_dir = ""`, `years = 30`, `end_date = ""`, `days_per_year = 365.2425`, `zarr_dir_partition = "yearly"`, `daily_csv_enabled = true`, `daily_csv_interval_days = 1.0`, `monthly_mean_enabled = true`, `monthly_mean_periods_per_year = 12`, `monthly_mean_sample_hours = 6.0`, `monthly_mean_include_2d = true`, `monthly_mean_include_3d = true`, `monthly_mean_include_subgrid_2d = true`, `snapshots_enabled = true`, `snapshots_periods_per_year = 12`, `snapshots_include_2d = true`, `snapshots_include_3d = true`, `snapshots_include_subgrid_2d = true`, `restart_interval_years = 1`, `restart_monthly = true` |

#### `[constants]` 〜 `[icesheet_physics]` — 全物理・生態系・雪氷・海洋定数レジストリ（180項目）
| セクション | 項目数 | 登録キー ＝ 標準値（`src/aoba/` 対応モジュール名） |
| :--- | :--- | :--- |
| `[constants]` | 26 | `earth_radius = 6.371e6`, `gravity = 9.79764`, `omega_earth = 7.292115e-5`, `seconds_per_day = 86400.0`, `days_per_year = 365.2425`, `r_d = 287.04`, `cp_d = 1004.64`, `r_v = 461.5`, `cp_v = 1846.0`, `p_ref = 1e5`, `t_freeze = 273.15`, `t_triple = 273.16`, `e_triple = 611.2`, `l_v0 = 2.501e6`, `l_f0 = 3.337e5`, `t_ice_all = 253.15`, `t_liq_all = 273.16`, `stefan_boltzmann = 5.670374419e-8`, `von_karman = 0.4`, `rho_seawater = 1025.0`, `cp_seawater = 3990.0`, `t_freeze_ocean = 271.35`, `rho_seaice = 917.0`, `k_therm_seaice = 2.034`, `k_therm_snow = 0.3096`, `o2_mixing_ratio = 0.20946` (`constants.py`) |
| `[radiation]` | 7 | `solar_frac_sw1 = 0.45976`, `lw_window_frac = 0.32`, `o3_g1 = 3.6478`, `o3_g2 = 0.83209`, `o3_g3 = 11.3515`, `par_frac_of_sw1 = 0.88`, `umol_per_joule_par = 4.57` (`constants.py` / `radiation.py` / `radiation_ckd.py`) |
| `[microphysics]` | 32 | `rho_rain = 1000.0`, `rho_snow = 100.0`, `rho_graupel = 400.0`, `n0_rain = 8e6`, `n0_snow = 3e6`, `n0_graupel = 4e6`, `a_fall_rain = 842.0`, `b_fall_rain = 0.8`, `a_fall_snow = 4.84`, `b_fall_snow = 0.25`, `a_fall_graupel = 94.5`, `b_fall_graupel = 0.5`, `e_rc = 1.0`, `e_sc = 1.0`, `e_gc = 1.0`, `e_ri = 1.0`, `e_si = 0.1`, `e_gi = 0.1`, `qc_autoconv_thresh = 0.001`, `qi_aggreg_thresh = 1e-4`, `rate_autoconv = 0.001`, `rate_ice_aggreg = 0.001`, `vt_cloud_ice = 0.4`, `cloud_droplet_conc_cm3 = 60.0`, `ccn_ocean_base_cm3 = 45.0`, `ccn_land_base_cm3 = 95.0`, `ccn_dust_scale_cm3 = 120.0`, `inp_dust_enhance = 4.0`, `t0_precip = 283.16`, `t00_precip = 268.16`, `t0_graupel = 283.16`, `t00_graupel = 223.16` (`microphysics.py`) |
| `[turbulence]` | 13 | `mynn_b1 = 24.0`, `mynn_b2 = 15.0`, `mynn_c2 = 0.75`, `mynn_c3 = 0.352`, `mynn_c5 = 0.2`, `mynn_gamma1 = 0.235`, `mynn_pr_n = 0.74`, `tke_min = 1e-4`, `rsl_mu_m = 2.59`, `rsl_mu_h = 0.95`, `rsl_nu = 0.5`, `rsl_lambda = 1.5`, `kinematic_visc_air = 1.5e-5` (`turbulence.py`) |
| `[surface]` | 24 | `cs_dry_soil = 1.26e6`, `kappa_dry_soil = 0.3`, `soil_porosity = 0.43`, `psi_sat_soil = -0.2`, `b_clapp_soil = 5.39`, `k_sat_soil = 7e-6`, `ice_impedance_omega = 6.0`, `wc_max_per_lai = 0.1`, `alpha_intercept_liq = 1.0`, `f_wet_max = 1.0`, `dsl_dmax_mm = 15.0`, `land_eb_iterations = 3`, `rho_snowpack = 250.0`, `swe_glacier_max = 1000.0`, `tau_snow_age = 1e6`, `alb_snow_vis_new = 0.9`, `alb_snow_nir_new = 0.7`, `alb_snow_vis_old = 0.65`, `alb_snow_nir_old = 0.35`, `emissivity_land = 0.98`, `emissivity_ocean = 0.985`, `mc_seaice_crit = 450.0`, `alb_seaice_vis = 0.72`, `alb_seaice_nir = 0.42` (`surface.py`) |
| `[biosphere]` | 41 | `theta_nrh = 0.85`, `cost_c_smith = 0.05336251`, `phi_psii_ref_c3 = 0.257`, `beta_c3 = 146.0`, `beta_c4 = 166.0`, `gamma_star25_c3 = 4.332`, `kc25_c3 = 41.03`, `ko25_c3 = 28210.0`, `dha_gamma = 37830.0`, `dha_kc_c3 = 79430.0`, `dha_ko_c3 = 36380.0`, `gamma_star25_c4 = 2.6`, `kp25_umol_c4 = 60.5`, `dha_kp_c4 = 27200.0`, `kc25_c4 = 121.0`, `dha_kc_c4 = 64200.0`, `ko25_c4 = 29200.0`, `dha_ko_c4 = 10500.0`, `phi_leak_c4 = 0.2`, `gbs_c4 = 0.003`, `ha_vcmax = 71513.0`, `ha_jmax = 49884.0`, `ha_vpmax = 55000.0`, `hd_enzyme = 2e5`, `b_resp_ren = 0.018`, `leaf_scat_vis = 0.15`, `leaf_scat_nir = 0.8`, `kd_prime_par = 0.719`, `rho_cd_par = 0.036`, `z_foliage_cost = 12.227`, `k_beer_lai = 0.5`, `sigma_zhou = 0.771`, `alpha_ema_acclim = 0.067`, `evg_cost_ratio_base = 1.35`, `evg_cost_ratio_cold = 0.55`, `evg_cold_eff_penalty = 0.14`, `leaf_aging_b_yr = 15.0`, `decid_flush_days = 25.0`, `evg_offseason_resp25 = 32.0`, `evg_frost_crit_k = 233.15`, `evg_frost_cost = 18.0` (`biosphere.py`) |
| `[ocean_ekman]` | 18 | `h_thermocline = 200.0`, `r_ekman_damp_days = 2.0`, `kappa_ocn_horiz = 2000.0`, `kappa_ocn_vert = 2e-5`, `g_prime_ocn = 0.03`, `tau_sw_drag_days = 180.0`, `enable_shallow_water_ocn = true`, `kappa_gm = 800.0`, `kappa_redi_horiz = 1000.0`, `enable_full_redi = true`, `kpp_ri_crit = 0.3`, `kpp_cs_nonlocal = 6.33`, `shear_nu_0 = 5e-3`, `tidal_e0_wm2 = 8e-3`, `sw_pen_zeta2_m = 23.0`, `kappa_v_bg = 1.5e-5`, `kappa_v_abyss = 3e-5`, `seaice_h_ridge_m = 3.0` (`ocean_ekman.py` / `ocean_ogcm.py`) |
| `[orography]` | 3 | `hn_crit = 0.5`, `cd_block = 1.0`, `g_wave = 0.8` (`orography.py`) |
| `[gravity_wave]` | 11 | `ngwd_front_thresh = 1.25e-15`, `ngwd_front_launch_hpa = 600.0`, `ngwd_front_tau0 = 1.5e-3`, `ngwd_front_cw = 30.0`, `ngwd_front_eff = 0.1`, `ngwd_conv_tau0 = 2e-3`, `ngwd_conv_qref = 2e-4`, `ngwd_conv_eff = 0.25`, `ngwd_wavelength_m = 2e5`, `ngwd_cmax = 80.0`, `ngwd_tend_max = 1.157e-3` (`gravity_wave.py`) |
| `[icesheet_physics]` | 5 | `rho_ice = 917.0`, `rho_mantle = 3300.0`, `a_glen_per_year = 1e-16`, `h_ice_threshold_m = 5.0`, `h_ice_max_m = 6000.0` (`icesheet.py`) |
