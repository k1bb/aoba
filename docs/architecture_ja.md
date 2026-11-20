# AOBA モデルアーキテクチャおよび根拠文献トレーサビリティ仕様書

> **言語 (Language)**: [English (architecture.md)](architecture.md) | 日本語

## 1. 科学的目的とモデル記述書基準

AOBA は以下の2つの根源的な問いを解くために設計された結合気候・生態系モデルである：
1. 外部境界条件として**「`Earth2014`（Hirt & Rexer, 2015）による海陸・氷床分布と地表面標高 $z_{\mathrm{topo}}$、自転角速度 $\Omega$、太陽軌道入射 $S_0$、大気ガス濃度（$\mathrm{CO}_2, \mathrm{O}_2$）」のみ**を与えたとき、大気大循環・海洋表層・陸面生態系の結合系は地球の気候帯・バイオーム帯を自律的に再現できるか（Beck et al., 2018, 2023 のケッペン・ガイガー気候区分で検証）。
2. 積雲パラメタリゼーション・処方海洋 $Q$ フラックス・植物機能型（PFT）パラメータ表といった経験的境界制約を排した最小制約モデルは、高 $\mathrm{CO}_2$ 下における将来の気候帯再編をどのように予測するか。

**モデル記述書・文献トレーサビリティ原則**: *Description of MIROC6 AGCM* (2021)、*Description of MATSIRO6* (2021)、および *Detailed formulation of SCALE-RM* (2025) の明快さと数学的網羅性を基準とし、`src/aoba/` 内のすべての連続系方程式・座標変換・スタガード空間離散化・時間積分・物理定数を明示的に文書化するとともに、根拠となる査読論文・技術報告書の著者・出版年・節番号・式番号をコード内に明記する。

---

## 2. 統一非静力学力学フレームワークとマルチスケール格子階層

### 2.1 全球ホスト力学コア：球面重心ボロノイ直交 C-grid (TRiSK) ＋ 鉛直 Charney–Phillips HEVI
- **根拠文献**: Ringler, Ju, & Gunzburger (2008, *Ocean Dynamics*, 58, 475–498); Thuburn, Ringler, Skamarock, & Klemp (2009, *J. Comput. Phys.*, 228, 8321–8335); Ringler, Thuburn, Klemp, & Skamarock (2010, *J. Comput. Phys.*, 229, 3065–3090); Klemp (2011, *Mon. Wea. Rev.*, 139, 2163–2169); Staniforth & Thuburn (2012, *Q. J. R. Meteorol. Soc.*, 138, 1–26); Skamarock et al. (2012, *Mon. Wea. Rev.*, 140, 3090–3105); Gassmann (2013, *Q. J. R. Meteorol. Soc.*, 139, 152–175); Held & Suarez (1994, *Bull. Amer. Meteor. Soc.*, 75, 1825–1830).
- **支配方程式**: 乾燥空気密度 $\rho_d$、セル辺法線運動量、鉛直運動量 $\rho_d w$、湿潤温位フラックス変数 $\rho_d \theta_m$（水蒸気浮力効果を含む；Seidel & Yang, 2024）、および水物質 $(\rho_d q_n, \rho_d q_p)$ を予報する3次元フラックス保存型完全圧縮性非静力学オイラー方程式系。
- **水平 SCVT 直交 Arakawa C-grid**: 正二十面体再帰分割と球面上の Lloyd 反復（Ringler et al., 2008）により生成される準一様ボロノイ格子を用いる：
  - `Glevel-4`: $N_{\mathrm{cells}} = 2,562$ セル（$\Delta x \approx 240\ \mathrm{km}$）、$N_{\mathrm{edges}} = 7,680$。
  - `Glevel-5`: $N_{\mathrm{cells}} = 10,242$ セル（$\Delta x \approx 120\ \mathrm{km}$）、$N_{\mathrm{edges}} = 30,720$。
  - ボロノイ辺と双対ドロネー線分が厳密に直交するため、A-grid 特有の $2\Delta x$ 圧力・発散デカップリングによる鉛直流 $w$ の数値ノイズも、Cubed-Sphere 特有の非直交頂点・パネル境界の刻印も生じない。**TRiSK** 接線風速再構成により定常地衡流モードの厳密維持・エネルギー保存・渦位（PV）整合性を満たし、六角形 C-grid 固有の非発散回転モードは4階超粘性／Smagorinsky 散逸で制御する（Skamarock et al., 2012; Gassmann, 2013）。
- **鉛直 Charney–Phillips 格子 ＆ 平滑化高度ベース地形追従座標**: 上端 $z_{\mathrm{top}} \approx 24\ \mathrm{km}$（$p_{\mathrm{top}} \approx 25\ \mathrm{hPa}$）までの $N_z = 32$ 層（$2\ \mathrm{km}$ 以下の境界層に約13層・最下層厚 $\Delta z_1 = 60\ \mathrm{m}$、自由対流圏 $2\text{--}16\ \mathrm{km}$ に約14層、上部 $16\text{--}24\ \mathrm{km}$ にレイリー減衰スポンジ緩衝層5層）。湿潤温位 $\theta_m$ と鉛直流 $w$ をともに層境界に置く **Charney–Phillips 配置**（Gassmann, 2013）と高度方向に段階平滑化した地形追従座標（Klemp, 2011）により、鉛直 $2\Delta z$ 浮力計算モードと急峻山岳上の偽上昇流を防ぐ。
- **HEVI 時間積分**: 3段 Runge–Kutta（RK3）＋ 鉛直陰解法（HEVI：融合バッチ Thomas 三重対角カーネル）および正値単調 Flux-Corrected Transport（FCT）スカラー移流を用いる（Skamarock et al., 2012; Nishizawa et al., 2025）。

### 2.2 変位追従型の実効空力基準面と自律樹冠構造
- **根拠文献**: Hirt & Rexer (2015, *Int. J. Appl. Earth Obs. Geoinf.*, 39, 103–112); Kempes et al. (2011, *PLoS ONE*, 6, e20551); Raupach (1994, *Boundary-Layer Meteorol.*, 71, 211–216); Harman & Finnigan (2007, *Boundary-Layer Meteorol.*, 123, 339–363).
- **`Earth2014` 境界標高と厳密な海陸・氷床マスク**: 5分格子 `Earth2014`（`SUR2014`, `TBI2014`, `ICE2014`）からセル面積平均標高 $z_{\mathrm{topo}}$ とともに、開水面（$\mathrm{SUR} - \mathrm{TBI} > 0$）、接地氷床（$\mathrm{ICE} > 0,\ \mathrm{SUR} = \mathrm{TBI}$）、および海面下内陸盆地を含む非氷河陸地を代数的に厳密分離する。
- **実効空力基準面**: 維管束アロメトリと局所資源収支から最大樹冠高 $h_c$ を決定し（Kempes et al., 2011）、ゼロ面変位 $d(h_c, \mathrm{LAI})$ と運動量粗度長 $z_{0m}(h_c, \mathrm{LAI})$ を連続評価して粗度サブレイヤ（RSL）補正と接続する（Raupach, 1994; Harman & Finnigan, 2007）。鉛直座標の下端原点を
  $$
  z_{\mathrm{datum}} = z_{\mathrm{topo}} + d
  $$
  に定義することで、最下層中心（$\tilde{z}_1 = 30\ \mathrm{m}$）において常に $\tilde{z}_1 > (h_c - d) > z_{0m}$ が成立する。

### 2.3 埋め込み3次元非静力学雲解像モデル（3D CRM）とブロック結合
- **根拠文献**: Khairoutdinov & Randall (2003, *J. Atmos. Sci.*, 60, 607–623); Khairoutdinov, Randall, & DeMott (2005, *J. Atmos. Sci.*, 62, 2136–2154); Yamazaki & Miura (2024, *JAMES*, 16, e2023MS003836); Nishizawa, Tomita, & Team SCALE (2025, *SCALE-RM v5.5.5*, Ch. 2–6 & App. A.2).
- **統一ソルバとテンソル構造**: 水平二重周期 $32\ \mathrm{km} \times 32\ \mathrm{km}$ の直交正方形 Arakawa C-grid × 鉛直 Charney–Phillips 格子（`[B_crm, nz = 32, ny = 8, nx = 8]`、$\Delta x = \Delta y = 4\ \mathrm{km}$、$\Delta t_{\mathrm{CRM}} = 12\text{--}15\ \mathrm{s}$）を用い、ホスト全球コアと完全に同一の圧縮性非静力学 HEVI 鉛直ソルバ・熱力学・FCT 移流を共有する。
- **1:1 フル SP および 4:1 ブロック SP**: `Glevel-4` での 1:1 結合（$B_{\mathrm{crm}} = 2,562$）に加え、`Glevel-5`（$10,242$ セル、約 $120\ \mathrm{km}$ 解像度）の親ボロノイ4セル単位で CRM（$B_{\mathrm{crm}} = 2,562$）を共有する 4:1 ブロック結合（Yamazaki & Miura, 2024）に対応し、積乱雲による加熱・加湿および3次元対流運動量輸送傾向項をホスト循環へ返す。

---

## 3. 雲微物理および4バンド解析的2ストリーム放射

### 3.1 2変数・診断氷相分配型 1モーメント雲微物理
- **根拠文献**: Khairoutdinov & Randall (2003, *J. Atmos. Sci.*, 60, App. A–B).
- **定式化**: 非降水総水分比湿 $q_n = q_v + q_c + q_i$ と降水総水分比湿 $q_p = q_r + q_s$ の2変数を予報し、凝結水 $\max(0, q_n - q^*(T, p))$ と降水 $q_p$ の液相・氷相比率を $253.15\text{--}273.16\ \mathrm{K}$ の線形ランプ関数 $\omega(T) \in [0, 1]$ で診断配分することで、自動変換・捕捉成長・落下・蒸発／昇華・凍結潜熱・氷雲放射効果を表現する。

### 3.2 ルックアップテーブル不要の4バンド解析的2ストリーム放射
- **根拠文献**: Toon et al. (1989, *J. Geophys. Res.*, 94, 16,287–16,301); Lacis & Hansen (1974, *J. Atmos. Sci.*, 31, 118–133); Slingo (1989, *J. Atmos. Sci.*, 46, 1419–1427); Ebert & Curry (1992, *J. Geophys. Res.*, 97, 3831–3836); Weaver & Ramanathan (1995, *J. Geophys. Res.*, 100, 11,585–11,591); Koll & Cronin (2018, *PNAS*, 115, 10293–10298); Wing et al. (2018, *Geosci. Model Dev.*, 11, 793–813); Sherwood et al. (2020, *Rev. Geophys.*, 58, e2019RG000678).
- **ソルバ構成**: Toon et al. (1989) の統一三重対角2ストリームソルバを $\Delta t_{\mathrm{rad}} = 900\ \mathrm{s}$ ごとに評価する（サブステップでは瞬時太陽天頂角余弦 $\mu_0(t)$ でスケーリング）：
  - **SW1（紫外・可視光・PAR：太陽定数の約 50%）**: Wing et al. (2018, Eq. 4) の一様背景オゾン $\mathrm{O}_3(p)$ と Lacis & Hansen (1974) の吸収式で対流圏界面の安定キャップを形成し、水雲・氷雲の $\delta$-Eddington 散乱（Slingo, 1989; Ebert & Curry, 1992）と地表可視アルベド $\alpha_{\mathrm{vis}}$ を解く。地表透過光がそのまま生態系の $\mathrm{PAR}$ となる。
  - **SW2（近赤外：太陽定数の約 50%）**: Lacis & Hansen (1974) の水蒸気吸収、雲の散乱・吸収、植生・雪氷の近赤外反射率 $\alpha_{\mathrm{nir}}$ を解く。
  - **LW1（水蒸気・CO₂ 吸収帯：比率 $1 - \beta \approx 0.68$）**: 圧力ブロードニング水蒸気光学的厚さと対数型 $\mathrm{CO}_2$ 光学的厚さ（Weaver & Ramanathan, 1995; Koll & Cronin, 2018; Sherwood et al., 2020）および雲射出率から対流圏放射冷却（約 $1.5\text{--}2\ \mathrm{K\ d^{-1}}$）を生成する。
  - **LW2（8–12 $\mu\mathrm{m}$ 大気の窓バンド：比率 $\beta \approx 0.32$）**: 水蒸気自己連続吸収（$\propto q_v^2 p / p_0$；Koll & Cronin, 2018）と水雲・氷雲の質量吸収（Ebert & Curry, 1992）を解き、晴天 OLR 線形性・夜間地表冷却・かなとこ雲温室効果を再現する。

---

## 4. サブグリッド自律生態系と 1.5層ボロノイ・エクマン海洋

地表面タイルは SLM／MAML 方式（Lee & Khairoutdinov, 2015; Lin et al., 2023）に従って上空の大気コラムと直接フラックスを交換する。

### 4.1 最適性理論に基づく C₃・C₄ 光合成順化と C₃/C₄ 割合の自律決定
- **根拠文献**: Farquhar, von Caemmerer, & Berry (1980); Bernacchi et al. (2001, 2003); Prentice et al. (2014); Wang et al. (2017); Stocker et al. (2020); Mengoli et al. (2022); Scott & Smith (2022); Luo et al. (2024); Lavergne et al. (2026).
- **順化と優占率決定**: 最小コスト C₃（$\chi_{\mathrm{C3}}$）および C₄（Scott & Smith, 2022）の光合成能力は昼間環境に対して時定数 $\tau_{\mathrm{acclim}} = 15\ \mathrm{d}$ で順化し（Mengoli et al., 2022）、瞬時の気孔コンダクタンス $g_s$ と地表熱収支（$R_n = H + \lambda E + G$；MATSIRO6 Document Writing Team, 2021）を毎ステップ解く。自然植生の C₄ 割合 $F_4 \in [0, 1]$ は C₄・C₃ 最適光合成の相対優位性から自律決定される（Luo et al., 2024; Lavergne et al., 2026）。

### 4.2 最適根圏貯水容量・根圏水供給閉合と予報 EEO LAI
- **根拠文献**: Gao et al. (2014); Davis et al. (2017); Sellers et al. (1986); Cai et al. (2025); Zhou et al. (2025).
- **根圏容量設計と予報 LAI**: 根圏有効貯水容量 $W_{\max} = S_R$ を乾季累積水不足に対する質量収支最適性原理（Gao et al., 2014）から自律決定し、根圏からの最大水供給速度 $S_w(W) = c_w W / S_R$（Davis et al., 2017）で実蒸散を閉じる。年最大緑度 $\mathrm{fAPAR}_{\max} = \max\!\left(0,\, \min(\mathrm{fAPAR}_c,\, \mathrm{fAPAR}_w)\right)$ と日次の Lambert $W_0$ 定常葉面積 $L_s$ に対して予報 $\mathrm{LAI}$ が追従し（$\alpha = 0.067$；Cai et al., 2025; Zhou et al., 2025）、乾季の土壌乾燥を落葉・LAI 低下として表現しつつ、2ストリーム樹冠反射（Sellers et al., 1986; MATSIRO6 Document Writing Team, 2021）および積雪・氷床面から2バンドアルベド $(\alpha_{\mathrm{vis}}, \alpha_{\mathrm{nir}})$ を決定する。

### 4.3 1.5層ボロノイ C-grid 風成エクマン海洋および熱力学海氷
- **根拠文献**: Codron (2012, *Climate Dynamics*, 38, 379–389); Semtner (1976, *J. Phys. Oceanogr.*, 6, 379–389).
- **処方 $Q$ フラックス不要のエクマン・スラブ海洋**: ホストと同一の SCVT ボロノイ C-grid 上（海岸線エッジで法線流量 $= 0$）で、海面風応力が駆動する表層混合層（$H_{\mathrm{ml}} = 50\ \mathrm{m}$）のエクマン輸送・沿岸／赤道湧昇・下層補償流との熱交換・水平渦拡散を**処方 $Q$ フラックス完全ゼロ**で解き（Codron, 2012）、結氷点（$T_f = 271.35\ \mathrm{K}$）以下では Semtner (1976) の熱力学海氷モデルと結合する。
