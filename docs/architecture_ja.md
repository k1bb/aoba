# AOBA モデルアーキテクチャおよび根拠文献トレーサビリティ仕様書

> **言語 (Language)**: [English (architecture.md)](architecture.md) | 日本語

## 1. 科学的目的

AOBA は以下の2つの根源的な問いを解くために設計された結合気候・生態系モデルである：
1. 外部境界条件として**「`Earth2014`（Hirt & Rexer, 2015）による海陸・氷床分布と地表面標高 $z_{\mathrm{topo}}$、自転角速度 $\Omega$、太陽軌道入射 $S_0$、大気ガス濃度（$\mathrm{CO}_2, \mathrm{O}_2$）」のみ**を与えたとき、大気大循環・海洋・陸面生態系の結合系は地球の気候帯・バイオーム帯を自律的に再現できるか（Beck et al., 2018, 2023 のケッペン・ガイガー気候区分で検証）。
2. 積雲パラメタリゼーション・処方海洋 $Q$ フラックス・外部土壌／アルベド地図・植物機能型（PFT）パラメータ表といった経験的境界制約を排した最小制約モデルは、高 $\mathrm{CO}_2$ 下における将来の気候帯再編をどのように予測するか。

---

## 2. 統一非静力学力学フレームワーク・乱流階層・サブグリッド山岳ドラッグ

### 2.1 全球ホスト力学コア：球面重心ボロノイ直交 C-grid (TRiSK) ＋ 鉛直 Charney–Phillips HEVI
- **根拠文献**: Ringler, Ju, & Gunzburger (2008, *Ocean Dynamics*, 58, 475–498); Thuburn, Ringler, Skamarock, & Klemp (2009, *J. Comput. Phys.*, 228, 8321–8335); Ringler, Thuburn, Klemp, & Skamarock (2010, *J. Comput. Phys.*, 229, 3065–3090); Klemp (2011, *Mon. Wea. Rev.*, 139, 2163–2169); Staniforth & Thuburn (2012, *Q. J. R. Meteorol. Soc.*, 138, 1–26); Skamarock et al. (2012, *Mon. Wea. Rev.*, 140, 3090–3105); Gassmann (2013, *Q. J. R. Meteorol. Soc.*, 139, 152–175); Held & Suarez (1994, *Bull. Amer. Meteor. Soc.*, 75, 1825–1830); Nishizawa, Tomita, & Team SCALE (2025, *SCALE-RM v5.5.5*, Ch. 2).
- **支配方程式**: 乾燥空気密度 $\rho_d$、セル辺法線運動量、鉛直運動量 $\rho_d w$、湿潤温位フラックス変数 $\rho_d \theta_m$（水蒸気浮力効果および圧縮性系で整合した非断熱加熱・降水エンタルピー流束を含む；Seidel & Yang, 2024; Nishizawa et al., 2025, Sec. 2.5–2.7）、および水物質 $(\rho_d q_n, \rho_d q_p)$ を予報する3次元フラックス保存型完全圧縮性非静力学オイラー方程式系。
- **水平 SCVT 直交 Arakawa C-grid**: 正二十面体再帰分割と球面上の Lloyd 反復（Ringler et al., 2008）により生成される準一様ボロノイ格子を用いる：
  - `Glevel-4`（標準構成）: $N_{\mathrm{cells}} = 2,562$ セル（$\Delta x \approx 240\ \mathrm{km}$）、$N_{\mathrm{edges}} = 7,680$。
  - `Glevel-5`（高解像度構成）: $N_{\mathrm{cells}} = 10,242$ セル（$\Delta x \approx 120\ \mathrm{km}$）、$N_{\mathrm{edges}} = 30,720$。
  - ボロノイ辺と双対ドロネー線分が厳密に直交するため、A-grid 特有の $2\Delta x$ 圧力・発散デカップリングによる鉛直流 $w$ の数値ノイズも、Cubed-Sphere 特有の非直交頂点・パネル境界の刻印も生じない。**TRiSK** 接線風速再構成により定常地衡流モードの厳密維持・エネルギー保存・渦位（PV）整合性を満たし、水平 Smagorinsky 粘性と4階超粘性（$\nabla^4$）で六角形 C-grid の非発散回転モードを制御する（Skamarock et al., 2012; Gassmann, 2013）。
- **鉛直 Charney–Phillips 格子 ＆ 平滑化高度ベース地形追従座標**: 上端 $z_{\mathrm{top}} \approx 24\ \mathrm{km}$（$p_{\mathrm{top}} \approx 25\ \mathrm{hPa}$）までの $N_z = 32$ 層（$2\ \mathrm{km}$ 以下の境界層に13層・最下層厚 $\Delta z_1 = 60\ \mathrm{m}$、自由対流圏 $2\text{--}17.5\ \mathrm{km}$ に14層、下部成層圏 $17.5\text{--}24\ \mathrm{km}$ にレイリー減衰スポンジ緩衝層5層）。$\theta_m$ と $w$ をともに層境界に置く **Charney–Phillips 配置**（Gassmann, 2013）と高度平滑化地形追従座標（Klemp, 2011）により、鉛直 $2\Delta z$ 浮力計算モードと急峻山岳上の偽上昇流を防ぐ。
- **サブグリッド山岳ブロッキング ＆ 地形性重力波ドラッグ**: 5分格子 `Earth2014` から各ボロノイセル内で厳密集計するサブグリッド標高標準偏差 $\sigma_{\mathrm{topo}}$・異方性・勾配を用い、低層山岳ブロッキング抵抗と対流圏重力波砕波ドラッグを評価する（Lott & Miller, 1997, *Q. J. R. Meteorol. Soc.*, 123, 101–127; MIROC6 AGCM Document Writing Team, 2021, Sec. 3.9）。
- **HEVI 時間積分**: 3段 Runge–Kutta（RK3）＋ 鉛直陰解法（HEVI：Metal/CUDA 融合バッチ Thomas 三重対角カーネル）および Zalesak 正値単調 Flux-Corrected Transport（FCT）スカラー移流を用いる（Skamarock et al., 2012; Nishizawa et al., 2025, App. A.2）。

### 2.2 変位追従型の実効空力基準面と最新粗度サブレイヤ（RSL）
- **根拠文献**: Hirt & Rexer (2015); Kempes et al. (2011); Raupach (1994); Harman & Finnigan (2007); De Ridder (2010, *Boundary-Layer Meteorol.*, 134, 257–267); Bonan et al. (2018, *GMD*, 11, 1467–1496); Lee et al. (2020, *GMD*, 13, 521–536); Abolafia-Rosenzweig et al. (2021, *JAMES*, 13, e2021MS002665); Meier et al. (2022, *GMD*, 15, 2365–2393); Miller, Beljaars, & Palmer (1992, *J. Climate*, 5, 418–434).
- **`Earth2014` 境界標高と厳密な海陸・氷床マスク**: 5分格子 `Earth2014`（`SUR2014`, `TBI2014`, `ICE2014`）からセル平均標高 $z_{\mathrm{topo}}$・サブグリッド標準偏差 $\sigma_{\mathrm{topo}}$ とともに、開水面（$\mathrm{SUR} - \mathrm{TBI} > 0$）、接地氷床（$\mathrm{ICE} > 0,\ \mathrm{SUR} = \mathrm{TBI}$）、および海面下内陸盆地を含む非氷河陸地を代数的に厳密分離する。
- **実効空力基準面と解析的 RSL 相似則**: 維管束アロメトリと局所資源収支から最大樹冠高 $h_c$ を決定し（Kempes et al., 2011）、積雪深による雪埋没補正（MATSIRO6, 2021, Eqs. 2.1–2.3; Abolafia-Rosenzweig et al., 2021）を施したうえで、ゼロ面変位 $d(h_c, \mathrm{LAI})$・運動量粗度長 $z_{0m}(h_c, \mathrm{LAI})$（Raupach, 1994）および指数関数積分による閉形式 RSL 安定度補正 $\hat{\Psi}_{M,H}$（Harman & Finnigan, 2007; De Ridder, 2010; Bonan et al., 2018; Lee et al., 2020）を接続する。大気格子の下端は固体地球・氷床表面 $z_{w,0} = z_{\mathrm{topo}}$ に固定しつつ、地表面交換参照高度を $\tilde{z}_{\mathrm{ref}} = \max(z_{c,0} - z_{w,0} - 0.65\, h_c,\ 5.0\ \mathrm{m})$ として樹高変位を整合反映する。開水面上の運動量・熱・水蒸気粗度長には弱風時の滑面粘性項＋Charnock 波浪項を用いる（Miller et al., 1992）。

### 2.3 純粋 1:1 MAML+ICA 埋め込み CRM・MYNN Level 2.5 境界層・3D SGS 乱流
- **根拠文献**: Khairoutdinov & Randall (2003); Khairoutdinov, Randall, & DeMott (2005); Lee & Khairoutdinov (2015); Lin et al. (2023); Nakanishi & Niino (2009, *JMSJ*, 87, 895–912); Yamazaki & Miura (2024); Nishizawa, Tomita, & Team SCALE (2025, Sec. 8.1.2 & Sec. 8.2.1).
- **純粋 1:1 MAML + ICA スーパーパラメタリゼーション結合**: すべてのホストボロノイセルに1対1対応で水平二重周期の直交 Arakawa C-grid × 鉛直 Charney–Phillips CRM（既定 `[N_cells, nz = 32, ny = 1, nx = 8]`, $\Delta x = 8\ \mathrm{km}$、TOML で `8 × 8`, $\Delta x = 4\ \mathrm{km}$ 等へ任意設定可）を埋め込み、サブグリッド各列独立に4バンド放射（ICA）と地表面・植生状態（MAML）を保持・交換して3次元対流加熱・加湿・運動量輸送傾向項を返す。
- **乱流クロージャ**: 各 CRM コラムの鉛直大気境界層・層積雲頂混合には浮力長と凝結水分散を含む陰的 **MYNN Level 2.5** TKE クロージャ（Nakanishi & Niino, 2009; Nishizawa et al., 2025, Sec. 8.2.1）を用い、CRM 内の3次元サブグリッド乱流には非等方 **Smagorinsky–Lilly** 渦拡散（Nishizawa et al., 2025, Sec. 8.1.2）を併用する。

---

## 3. 雲微物理および4バンド解析的2ストリーム放射

### 3.1 2変数・診断氷相分配型 1モーメント雲微物理
- **根拠文献**: Khairoutdinov & Randall (2003, *J. Atmos. Sci.*, 60, App. A–B); Berry (1968); Tomita (2008); Nishizawa et al. (2025, Sec. 8.4.2).
- **定式化**: 非降水総水分比湿 $q_n = q_v + q_c + q_i$ と降水総水分比湿 $q_p = q_r + q_s$ を予報し、$253.15\text{--}273.16\ \mathrm{K}$ の線形ランプ関数 $\omega(T) \in [0, 1]$ で液相・氷相を診断配分して連続的暖雨自動変換（Berry, 1968 / Tomita, 2008）・捕捉成長・落下・蒸発／昇華・凍結潜熱・氷雲放射効果を表現する。

### 3.2 ルックアップテーブル不要の4バンド解析的2ストリーム放射
- **根拠文献**: Toon et al. (1989); Lacis & Hansen (1974); Slingo (1989); Ebert & Curry (1992); Cahalan et al. (1994); Weaver & Ramanathan (1995); Koll & Cronin (2018); Wing et al. (2018); Sherwood et al. (2020).
- **ソルバ構成**: Toon et al. (1989) の統一三重対角2ストリームソルバを $\Delta t_{\mathrm{rad}} = 900\ \mathrm{s}$ ごとに全コラム一括評価し（サブステップでは瞬時太陽天頂角余弦 $\mu_0(t)$ でスケーリング）、SW1（UV・可視光／PAR）と SW2（近赤外）のそれぞれで**地表到達の直達光・散乱光を分離出力**するとともに、LW1（水蒸気＋$\mathrm{CO}_2$ 吸収帯）と LW2（$8\text{--}12\ \mu\mathrm{m}$ 大気の窓領域）の長波放射伝達を解く。

---

## 4. サブグリッド Two-Leaf 自律生態系・多層凍土・雪氷・1.5層エクマン海洋

ホスト各セル内の 64 個の CRM コラムは、SLM／MAML 方式（Lee & Khairoutdinov, 2015; Lin et al., 2023）により直下の地表面タイルと 1対1 で直接フラックスを交換する。

### 4.1 Two-Leaf C₃/C₄ 最適性光合成・暗呼吸順化と気孔閉合
- **根拠文献**: Farquhar, von Caemmerer, & Berry (1980); de Pury & Farquhar (1997); Bernacchi et al. (2001, 2003); Dai, Dickinson, & Wang (2004); Medlyn et al. (2011); Prentice et al. (2014); Heskel et al. (2016); Wang et al. (2017); Kumarathunge et al. (2019); Stocker et al. (2020); Mengoli et al. (2022); Scott & Smith (2022); Luo et al. (2024); Ren et al. (2024, 2025); Schneider, Gessler, & Stocker (2025); Gan et al. (2026); Lavergne et al. (2026).
- **正午下向き短波 EMA・3本柱の最適温度順化・Two-Leaf 光合成容量の閉形式解析配分**:
  1. 正午（地方太陽時 $11\text{:}30\text{--}12\text{:}30$）の下向き可視直達・散乱短波放射 $(\mathrm{VIS}_b, \mathrm{VIS}_d)$、正午太陽天頂角余弦 $\cos\theta_z$、昼間生育気温 $T_{\mathrm{growth}}$・VPD、および夜間気温を 15日指数移動平均（EMA）で平滑化するとともに、外部地図不要の多年（約3年）生育期気候適応温度 $T_{\mathrm{home}}$ を自律追跡する。
  2. EMA 平滑化された正午下向き放射 $(\widetilde{\mathrm{VIS}}_b, \widetilde{\mathrm{VIS}}_d, \widetilde{\mu}_0)$ とその時点の積雪露出葉面積指数 $L_c$ に対して球面葉傾斜角2ストリーム樹冠光吸収計算（de Pury & Farquhar, 1997）を行い、順化時の吸収 PPFD 鉛直分布 $\tilde{I}_l(L) = \tilde{I}_{lb}(L) + \tilde{I}_{ld}(L)$ および全樹冠吸収 $\tilde{I}_c$ を求めて、PFT フリーの C₃（$V_{c\max 25}, J_{\max 25}$）・C₄（$V_{c\max 25}, J_{\max 25}, V_{p\max 25}$）最適容量および共順化葉暗呼吸 $R_{25}$（Heskel et al., 2016; Ren et al., 2024, 2025）を更新する。
  3. 各サブデイリータイムステップでは陽葉・陰葉容量を閉形式で厳密配分し（de Pury & Farquhar, 1997）、瞬時の葉温・VPD 応答 $(T_{\mathrm{leaf}}, D_s, c_s)$ から 3 つの順化過程――(i) 光合成容量 $(V_{c\max 25}, J_{\max 25}, R_{25})$、(ii) Kumarathunge et al. (2019) / Schneider et al. (2025) の温度順化酵素カイネティクス（$H_{a,V}(T_{\mathrm{growth}}), \Delta S_V(T_{\mathrm{growth}}), \Delta S_J(T_{\mathrm{home}}, T_{\mathrm{growth}})$）と量子収率 $\varphi(T_{\mathrm{growth}})$、(iii) 日内一定に保つ順化気孔感度 $\xi'(T_{\mathrm{growth}})$（Schneider et al., 2025; Gan et al., 2026）――を明確に分離することで、PFT 処方なしに観測整合的な $T_{\mathrm{opt}}$–$T_{\mathrm{growth}}$ 線形スケーリングと $A_{\mathrm{opt}}$ 緩衝作用を自律創発する。

### 4.2 蒸発散3成分分離・外部地図不要アルベド・最適根圏水文・多層凍土熱伝導・積雪過程
- **根拠文献**: Clapp & Hornberger (1978); Sellers et al. (1986); Yang et al. (1997); Takata et al. (2003); Dai et al. (2004, App. A); Sakaguchi & Zeng (2009); Swenson & Lawrence (2012); Oleson et al. (2013); Gao et al. (2014); Nitta et al. (2014); Lee & Khairoutdinov (2015); Davis et al. (2017); MATSIRO6 Document Writing Team (2021); Cai et al. (2025); Nishizawa et al. (2025, Sec. 8.7.1); Zhou et al. (2025).
- **蒸発散3成分と予報 EEO LAI**: 陸面潜熱フラックスを「陽葉・陰葉からの気孔蒸散（乾季水不足から自律設計される根圏貯水容量 $S_R$ と最大水供給速度 $S_w$ で閉合；Gao et al., 2014; Davis et al., 2017）」「濡れ樹冠からの利用水量制限付き遮断蒸発（$f_{\mathrm{wet}} = [W_c/(p_{\mathrm{liq}} L)]^{2/3}$、$p_{\mathrm{liq}} = 0.1\ \mathrm{kg\ m^{-2}}$；CLM5 技術文書 Ch. 5, 7）」「CLM5 の乾燥表層（DSL）土壌抵抗および樹冠下大気安定度補正を伴う裸地・林床蒸発（Swenson & Lawrence, 2014; Sakaguchi & Zeng, 2009）」の3成分で構成し、予報 $\mathrm{LAI} = \mathrm{LAI}_{\mathrm{evg}} + \mathrm{LAI}_{\mathrm{dec}}$ は冬季凍害を含む自律的常緑・落葉葉寿命最適性（$f_{\mathrm{evg}} \in [0, 1]$; Kikuzawa, 1991; Sakai & Larcher, 1987; Givnish, 2002; Wang et al., 2023）と Lambert-$W_0$ 落葉定常解（Cai et al., 2025; Zhou et al., 2025）を統合する。
- **外部地図不要の2バンドアルベドと積雪・氷床水収支**: 裸地土壌の VIS/NIR アルベドを表層土壌飽和度の連続関数として与え（Oleson et al., 2013; Lee & Khairoutdinov, 2015）、特異点除去済み2ストリーム樹冠放射（Dai et al., 2004, App. A）を重ねる。積雪過程にはサブグリッド標高標準偏差 $\sigma_{\mathrm{topo}}$ と融雪ヒステリシスを考慮した部分積雪被覆率 $A_{\mathrm{sn}}$（Swenson & Lawrence, 2012; Nitta et al., 2014）、積雪深による樹冠・$\mathrm{LAI}$ の雪埋没マスキング（MATSIRO6, 2021, Eqs. 2.1–2.3）、温度エージングと太陽天頂角依存性を含む2バンド積雪アルベド（Yang et al., 1997; MATSIRO6, 2021, Sec. 8.8）、および氷河流出上限（$S_{n,\max} = 1000\ \mathrm{kg\ m^{-2}}$；MATSIRO6, 2021, Eq. 8.44）を導入し、陸面からの総流出（液水・氷）は全球水・融解潜熱収支が閉じるよう海洋開水面へ保存還元する（MIROC6, 2021, Sec. 4.1.2）。
- **多層土壌熱伝導と凍結・融解（永久凍土熱力学）**: 深さ約 $3\ \mathrm{m}$ までの5層土壌格子上で、含水率依存の熱伝導率と $0\ ^\circ\mathrm{C}$ での液水・氷瞬間相変化（ゼロカーテン効果）を伴う陰的 Thomas 熱伝導方程式を解き、大陸の日変化・季節変化の熱慣性と凍土・永久凍土を表現する（MATSIRO6, 2021, Ch. 10; Nishizawa et al., 2025, Sec. 8.7.1）。

### 4.3 1.5層ボロノイ C-grid 風成エクマン海洋および部分海氷（リード）モデル
- **根拠文献**: Semtner (1976); Briegleb et al. (1986); Miller, Beljaars, & Palmer (1992); Codron (2012); MIROC6 AGCM Document Writing Team (2021, Sec. 3.8); Nishizawa et al. (2025, Sec. 8.6).
- **処方 $Q$ フラックス不要のエクマン＋浅水波海洋・部分海氷**: ホストと同一の SCVT ボロノイ C-grid 上（海岸線エッジで法線流速・流量 $= 0$）で、海面風応力が駆動する表層混合層（$H_{\mathrm{ml}} = 50\ \mathrm{m}$）のエクマン熱輸送および 1.5層低減重力浅水波力学（$g' = 0.03\ \mathrm{m\ s^{-2}}$、TRiSK コリオリ、Munk 水平渦粘性、風成亜熱帯循環・西岸強化流・赤道ケルビン／ロスビー波）を**処方 $Q$ フラックス完全ゼロ**で解き（Codron, 2012）、部分海氷被覆率 $f_i = \min(\sqrt{M_i/M_c}, 1)$・開水面／海氷別フラックス面積加重・降雪融解潜熱冷却・太陽天頂角依存海面アルベドを備えた熱力学海氷モデルと結合する（Semtner, 1976; Briegleb et al., 1986; MIROC6, 2021, Sec. 3.8; Nishizawa et al., 2025, Sec. 8.6）。

---

## 5. 動的／平衡加速 SIA 氷床モデルおよび単一 TOML 設定管理基盤

### 5.1 動的・平衡加速 SIA 氷床モデルと大気鉛直座標の再配置 (`src/aoba/icesheet.py`)
- **根拠文献**: Vialov (1958); Mahaffy (1976); Le Meur & Huybrechts (1996); Bueler et al. (2005); Cuffey & Paterson (2010); Pollard & DeConto (2012).
- **結合アーキテクチャ**: `Earth2014.BED2014` および `ICE2014` から岩盤標高と初期氷厚を構築し、年次 EMA 平滑表面質量収支（SMB）と Glen 流動則 ($n=3$) による SCVT 上流有限体積 SIA 拡散、緩和型リソスタシー・アイソスタシー、およびスピンアップ加速率 $\alpha_{\mathrm{eff}}(t)$ を統合する。氷床厚変化による $|z_{\mathrm{topo}}^{\mathrm{new}} - z_{w,0}| \ge 0.05\ \mathrm{m}$ 発生時には Klemp STF 座標を再構築し、測高公式外挿＋対数気圧線形補間でホスト大気および全 CRM コラムを再補間しつつ `SURF_LAND <-> SURF_ICESHEET` 被覆遷移と開拓植生播種を自律実行する。

### 5.2 単一 TOML パラメータ一元管理と対話型 GUI (`src/aoba/params.py`, `scripts/aoba_config_gui.py`)
- **透明性・再現性基盤**: 全格子設定 (`glevel`, 成層圏拡張 `nz = 44, z_top = 48 km`, CRM 格子数・間隔)・軌道／惑星／時間変動 CO₂・オゾン強制（`src/aoba/co2_forcing.py`, `src/aoba/ozone_forcing.py`）・16バンド CKD 放射 (`rad_scheme`)・FESOM2 型 Zalesak FCT 移流＋厳密 $z^*$ 座標 3D SCVT ボロノイ OGCM（`src/aoba/ocean_ogcm.py`）・FlexPFT 海洋生物地球化学＋OMIP/mocsy 2.0 炭酸塩化学（`src/aoba/ocean_bgc.py`）・MPAS-Seaice 型ボロノイ EVP 海氷力学＋5区分 ITD（`src/aoba/seaice_evp.py`）・マップ不要 Priority-Flood ボロノイ河川流路網（`src/aoba/river_routing.py`）・マップ不要陸域炭素循環＋5層永久凍土土壌炭素＋自律 SPITFIRE 森林火災（`src/aoba/land_carbon.py`）・マップ不要自然エアロゾル（海塩・FlexPFT 連動海洋 DMS 硫酸塩・火災 BC/OC; `src/aoba/aerosols.py`）・2モーメント雲微物理 (`microphysics_scheme`)・マップ不要の流れ依存非地形性重力波（`src/aoba/gravity_wave.py`）・動的氷床オプション・出力ファイル種別／時間間隔（異鉛直格子リスタート自動リマップ対応）・全物理定数（14セクション・265項目、全6つの ESM 拡張をデフォルト有効化）を単一の [`config/aoba_default.toml`](../config/aoba_default.toml) で一元管理し、`scripts/aoba_config_gui.py` から対話型可視化図付きで編集・保存して `scripts/run_g5_30yr.py --config ...` へ直接投入できる。

### 5.3 単一ソース自動切替 Metal / CUDA GPU バックエンド (`src/aoba/kernels.py`, `src/aoba/kernels_cuda.py`, `src/aoba/runtime.py`)
- **ハードウェア自動検知・ディスパッチ**: `get_device()`・`detect_hardware_profile()`・`get_active_gpu_backend()` により実行環境の計算機資源を自動判別し、最適デバイス（`cuda` → `mps` → `cpu`）およびメモリ安全チャンクサイズ（`compute_optimal_chunks()`）へ自動的に切り替える。
- **全43カーネルの単一ソース CUDA 自動トランスパイル**: 32レーン SIMD CRM RK3-HEVI・FCT ペア移流・異方性 SGS・1M/2M 雲微物理・16バンド CKD／4バンド放射・MYNN PBL・3ソース陸面／植生・ダスト・地形性／非地形性重力波・SCVT Voronoi 力学の全43個の融合 GPU カーネルを `kernels.py` / `kernels_extra.py` / `kernels_physics.py` に一元記述している。Apple Silicon (`mps`) では `torch.mps.compile_shader` で直接コンパイルし、NVIDIA GPU (`cuda`) では `src/aoba/kernels_cuda.py` が同一ソースをネイティブ CUDA C++17（32レーンワープシャッフル `__shfl_sync`/`__shfl_xor_sync`・共有メモリ同期 `__syncthreads()`）へ決定論的に変換・ビルドしてキャッシュ（`AOBA_CUDA_CACHE_DIR`）する。
- **Google Colab / クラウド自動レジューム運用**: [`notebooks/aoba_colab_spinup.ipynb`](../notebooks/aoba_colab_spinup.ipynb) により Google Drive 上へ毎月末チェックポイント（`restart_mid.pt`）・Zarr v3 出力・Earth2014 地形データ・コンパイル済み CUDA 拡張を永続化し、Colab A100/H100 の 24時間セッション上限や GCP Spot VM の中断を跨いでシームレスに長期スピンアップを継続できる。
