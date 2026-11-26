<img src="docs/assets/aoba_icon.png" width="96" align="right" alt="AOBA Icon" />

# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **言語 (Language)**: [English (README.md)](README.md) | 日本語

**AOBA**（**A**utonomous **O**ptimality **B**iosphere–**A**tmosphere）は、**事前学習不要のオンライン自己蒸留型機械学習加速（OSSP）**、生態進化的最適性（EEO: Eco-Evolutionary Optimality）に基づく自律陸面生態系、球面重心ボロノイ直交 C-grid 上の超パラメータ化全球非静力学大気、および独立高解像度 3D SCVT ボロノイ OGCM を統合した軽量・GPU ネイティブな地球システムモデル（ESM）です。モデル名は英語の頭字語であると同時に日本語の**「青葉（あおば）」**にかけており、大気・海洋・陸域生態系の結合系から生命ある緑の植生と気候帯が自律的に芽吹く様子を象徴しています。外部境界条件として**「海陸・氷床分布、地表面標高・海底地形（`Earth2014`）、自転周期、太陽放射、大気ガス濃度」のみ**を与え、積雲パラメタリゼーション、処方海洋 $Q$ フラックス、土壌・アルベド地図、植物機能型（PFT）パラメータ表を排した最小制約系において、地球の気候帯・海洋循環・雲・エアロゾル・生態系が普遍的な物理法則からいかに創発するかを解明することを目指しています。

## 開発の動機とモデル設計思想

### 既存の気候モデルが抱える構造的課題
- **地理的境界条件による現在気候の暗黙的制約**: PFT・土壌・アルベドの空間マップ、処方海洋 $Q$ フラックス、地域調整された重力波源やダスト発生源地図は、現在気候の循環・生物地理を外部入力として暗黙に固定している。そのため気候帯や生態系の自律創発の検証や、放射強制・軌道要素が大きく変化する状況（熱帯拡大、緑のサハラ、古気候レジーム遷移など）の表現を制約し得る。
- **積雲パラメタリゼーションのレジーム依存性**: 降水・雲放射フィードバック・テレコネクションは積雲スキームの閉合仮定に強く依存し、熱帯海洋上の深い対流を前提としたスキームでは陸上対流や浅い積雲・層積雲レジームで構造的バイアスが生じやすい。
- **陽的スーパーパラメタリゼーションの計算コストとオフライン ML エミュレータの分布外脆弱性**: 全全球コラムで雲解像モデル（CRM）と独立コラム放射を直接実行すると長期結合 ESM 計算のコストが膨大になる一方、現在気候データで事前学習して重みを固定するオフライン機械学習エミュレータは、急激な $4\times\mathrm{CO}_2$・古気候・系外惑星レジームで保存則破綻や分布外ドリフトを起こしやすい。
- **構造的不連続と参入障壁**: ホストとサブグリッドで異なる力学コアを結合する不整合、機関照会を要する境界データ、レガシー Fortran、特定スパコン専用実装、散在する設定ファイル、英語限定のドキュメントが研究・教育利用の障壁となっている。

### AOBA の設計思想
- **厳密保存則付き・事前学習不要のオンライン機械学習加速（OSSP）**: AOBA は 2D/3D スーパーパラメタリゼーションおよびサブコラム ICA/3D-TICA 放射計算を、**オンライン代理スーパーパラメタリゼーション（`OSSP`: Online Surrogate Superparameterization）** で加速する。空間層化＋認識的不確実性誘導で選んだ教師コラムのみ本物の非静力学 CRM・放射ソルバを実行し、残りのコラムは水質量・湿潤エンタルピー・水平運動量・放射フラックス収束を機械精度で厳密保存する軽量残差ニューラル代理モデルで前進させつつ GPU 上でリアルタイムにオンライン学習する。事前学習重みも緯度経度入力も持たないため、急激な $4\times\mathrm{CO}_2$・氷期・系外惑星へ自動適応し、さらに **逐次シャドウ乖離診断（Shadow Divergence Diagnostics）** により代理モデルと実物理の誤差を常時定量監視できる。
- **最小限の地理的境界条件からの自律創発**: 地表面標高・海底地形（`Earth2014`）および任意のオゾン気候値を除き、空間マップ型境界データを一切使用せず、気候帯・陸域生態系・森林火災・ダスト・海洋生態系・雲・海洋循環・重力波スペクトルを普遍方程式とグローバルパラメータのみから創発させる。
- **統一マルチスケール基盤と親・子モデルの完全同一力学**: 低解像度 SP 長期気候積分（`enable_superparam = true`）と高解像度直接雲解像計算（`enable_superparam = false`）を単一コードで実行でき、全球ホスト（親）と埋め込み CRM（子）の双方に同一のフラックス保存型完全圧縮性非静力学オイラー方程式・直交 Arakawa C-grid・Charney–Phillips 鉛直配置・RK3-HEVI・Zalesak FCT を採用する。
- **自動解像度スケーリングとラップトップ〜スパコン貫通 GPU ポータビリティ**: 格子間隔・時間刻み依存の物理・数値パラメータ（サブグリッド LWC 分散微物理、重調和粘性、発散減衰、Hallberg スケール適応 GM90、Smagorinsky 混合、CFL サブステップ）を稼働メッシュから自動算出する。アクセラレータ自動検知（`cuda` → `mps` → `cpu`）、**全 101 個の単一ソース融合 Metal/CUDA カーネル**、および自動 CUDA Graphs により、Apple Silicon ノート PC からクラウド GPU・地球シミュレータ（ES4 A100）まで同一コードで高速実行する。
- **オープン・可読・日英バイリンガル研究基盤**: 現代的で型注釈・文献引用付きの Python 実装、単一の 322 パラメータ TOML レジストリ、日英同期の教科書型ドキュメント、および対話型 Web ワークベンチ `AOBA Studio` を提供する。

## 主要アーキテクチャと機能

1. **オンライン自己蒸留型機械学習加速（`OSSP`：CRM ＋ 対称 ICA 放射 ＋ `SparseCRMPool` ＋ シャドウ診断）**:
   - **教師・生徒オンライン蒸留（`src/aoba/crm_self_distill.py`）**: 空間層化＋認識的不確実性（2-head アンサンブル）により選定された教師コラム群（`1 / crm_distill_stride`、対流記憶を保持する連続滞在ウィンドウ＋50% 永続アンカー群）でのみ本物の 2D/3D 非静力学 CRM および ICA/3D-TICA 放射を実行し、残余コラムを軽量残差 MLP（`CRMSelfDistillNet` および `RadSelfDistillNet`）で推論しつつ、GPU 常駐の二層リプレイバッファ（短期リング＋長期リザーバ）から常時オンライン更新します。
   - **厳密保存クロージャとサブグリッド組織化結合**: 緯度経度に依存しない局所熱力学プロファイル、3 変数の予報対流記憶状態 $(M_{P,\mathrm{fast}}, M_{P,\mathrm{slow}}, M_{B,\mathrm{slow}})$（Mapes & Neale, 2011; Colin et al., 2019）、および 16 種の周期 2D/3D 地表面不均一性・クラスタリング不変量（Moran の $I$、パッチ長スケール $L_{\mathrm{patch}}$、$\rho(\mathrm{SHF}, \mathrm{LHF})$；Patton et al., 2005）を入力します。融合 GPU クロージャカーネルがコラム水質量・顕熱/潜熱エンタルピー・水平運動量・サブコラムごとの TOA–大気–地表面放射収支を機械精度で厳密保存し（Beucler et al., 2021）、周期 `SubgridSpatialHead` を通じてゼロ平均コールドプール微気候およびガスト流出を MAML 陸面タイルへ結合します。
   - **教師専用 `SparseCRMPool` と逐次シャドウ乖離診断**: 表面タイルを独立解像する大規模 3D CRM（例：`64 × 16` `RCE_large`）では、`SparseCRMPool` が活性教師スロットのみに 3D CRM 状態メモリを確保します（`~37 GB` → `~0.15 GB`）。`enable_distill_shadow_diag = true` を有効化すると、毎回のオンライン勾配更新直前に教師コラム上で代理モデルをシャドウ評価し、急激な $\mathrm{CO}_2$ 4倍増などのレジーム遷移におけるホスト大気・陸面タイル双方の out-of-sample 乖離を無バイアスに記録します。
   - **1 GPU 実測スループット**: 単一 NVIDIA A100-40GB 上で、高トップ 3D `8 × 8` CRM ＋ 渦許容 `OG7` 海洋構成（`earth_g5_3d_8x8_g7ocn.toml`）が **`25.4 SYPD`（`0.94 時間/年`）**、17 の濃度駆動 ESM 要素を全投入した完全結合構成（`AG4z64 + OG7z64`, `32 × 1` `2 km` CRM, `RRTMGP + 3D-TICA`, `six_class_2m`, 予報 $\mathrm{O}_3$, FlexPFT 海洋 BGC/$\mathrm{O}_2$, EVP-ITD 海氷, MAML 地形/湖沼, 陸域 C/N/火災, エアロゾル, 水同位体）が **`10.0 SYPD`（`2.40 時間/年`, `23.7 秒/日`）** を達成しています。

2. **直交 C-grid × Charney–Phillips 統一圧縮性非静力学力学（SCVT ホスト ＋ 2D/3D CRM）**:
   - **全球ホスト力学コア（`src/aoba/scvt_dynamics.py`）**: 球面重心ボロノイ格子（SCVT、`Glevel-4`〜`Glevel-7+`）の直交 Arakawa C-grid 上に TRiSK 演算子、Charney–Phillips 鉛直配置（`nz = 32`〜`64`、上端 `z_top = 65 km` まで対応）、平滑化地形追従座標（Klemp, 2011）、RK3-HEVI 鉛直陰解法、正値保存 Zalesak FCT 移流、Lott & Miller (1997) サブグリッド地形性ブロッキング/重力波ドラッグ、およびマップ不要の流れ依存非地形性重力波ドラッグ（`src/aoba/gravity_wave.py`）を統合しています（Ringler et al., 2008, 2010; Thuburn et al., 2009; Skamarock et al., 2012; Gassmann, 2013; Nishizawa et al., 2025）。
   - **埋め込み 2D/3D CRM と乱流階層（`src/aoba/crm.py`, `src/aoba/turbulence.py`）**: 各ホストセルに周期圧縮性非静力学 CRM（`32 × 1` `2 km` 2D カーテンから `8 × 8`・`64 × 16` 3D まで可変）を埋め込み、陰解法 **MYNN Level 2.5** 境界層乱流（Nakanishi & Niino, 2009）と非等方 **Smagorinsky–Lilly** 3D SGS 拡散を結合します。

3. **選択式放射伝達（`four_band` / `ckd` / `ckd_hd` / `rrtmgp` ＋ 3D-TICA）と雲微物理（`one_moment` / `two_moment` / `six_class_2m`）**:
   - **放射伝達**: `rad_scheme` により解析 4 バンド（`four_band`）、16 バンド 50 g点 Correlated-$k$（`ckd`）、111 g点高精細 CKD（`ckd_hd`）、および公式 240 g点 HITRAN / MT\_CKD 4.1 **RRTMGP**（`rrtmgp`, LW 128 + SW 112；Pincus et al., 2019）を切替でき、3D 傾斜独立コラム近似（`enable_rad_3d_tica`；Várnai & Davies, 1999）およびエアロゾル・ダスト・雲・オゾン放射結合に対応します。
   - **雲微物理と発雷**: `microphysics_scheme` により 1 モーメント混相（`one_moment`；Khairoutdinov & Randall, 2003）、2 モーメント雲水/雲氷（`two_moment`）、および 6 クラス 2 モーメント（`six_class_2m`：水蒸気・雲水・雲氷・雨・雪・霰；Morrison et al., 2005; Seifert & Beheng, 2006）を選択でき、自然海塩・硫酸塩・ダスト由来の CCN/INP 活性化およびサブコラム非誘導帯電発雷・$\mathrm{LNO}_x$ 生成（`src/aoba/lightning.py`；McCaul et al., 2009）を備えます。

4. **PFT フリー Two-Leaf $\mathrm{C}_3/\mathrm{C}_4$ EEO 生態系、MAML サブグリッド地形・湖沼、陸域 C/N 凍土、火災、河川ルーティング**:
   - **自律 EEO 植生と熱順化（`src/aoba/biosphere.py`）**: 非直角双曲線最適性光合成（$V_{c\max 25}, J_{\max 25}, V_{p\max 25}, R_{25}$）、3 本柱熱順化（$T_{\mathrm{growth}}, T_{\mathrm{home}}$）、陽葉・陰葉閉形式配分、離散樹冠クランピングと最適葉傾斜角、気候適応土壌水分応答 $\tau(D_{\max})$、最適根圏貯水容量 $S_R$、冬季凍害を含む自律常緑/落葉比率 $f_{\mathrm{evg}}$、Lambert-$W_0$ 予報 LAI、および管束水力樹冠高 $h_c$ と解析的粗度サブレイヤ（RSL）を統合します（de Pury & Farquhar, 1997; Prentice et al., 2014; Smith et al., 2019; Scott & Smith, 2022; Wang et al., 2023; Schneider et al., 2025; Zhou et al., 2025）。
   - **MAML 地形・動的湖沼・C/N 循環・火災・河川**: MAML サブグリッド標高微気候ダウンスケーリングおよび TOPMODEL 湿地（`src/aoba/maml_land_subgrid.py`）、`Earth2014` 窪地体積と 2 層 FLake 熱力学によるマップ不要の動的湖沼（`src/aoba/lakes.py`）、5 層凍土熱伝導/Richards 浸透・積雪エージング/BC 暗色化（`src/aoba/surface.py`）、5 層永久凍土 C/N を含む陸域炭素–窒素循環と離散パッチ SPITFIRE 森林火災（`src/aoba/land_carbon.py`）、および Priority-Flood ボロノイ河川・溶存物質ルーティング（`src/aoba/river_routing.py`）を結合します。

5. **独立高解像度 3D $z^*$ FCT ボロノイ OGCM、EVP-ITD 海氷、FlexPFT 海洋 BGC ＋ $\mathrm{O}_2$/OMZ、波浪、オフライン深海加速**:
   - **選択式海洋コア（`ocean_mode = "ogcm_3d"` / `"ekman_sw"`）**: 独立細格子ボロノイ海洋（`ocean_glevel = 7` 約 28–55 km、`ocean_nz = 64` 表層 2 m 伸縮座標）に対応し、厳密 $z^*$ 自由表面、Zalesak FCT トレーサ移流、Roquet (2015) 状態方程式、Hallberg (2013) スケール適応 GM90 ＋ 回転 Redi 拡散、Fox-Kemper サブメソスケール再成層、KPP・Ri シア・内部潮汐深海混合、海流相対風応力、表面波浪結合（`src/aoba/ocean_waves.py`）、および C-grid Hollingsworth–Bernoulli 発散減衰（`src/aoba/ocean_ogcm.py`）を実装しています（軽量な 1.5 層エクマン浅水コア `src/aoba/ocean_ekman.py` も選択可）。
   - **EVP-ITD 海氷・厳密エンタルピー結合器・FlexPFT 海洋 BGC**: 5 カテゴリー氷厚分布（ITD）＋ボロノイ弾粘塑性（EVP）力学（`src/aoba/seaice_evp.py`）、水＋海氷の総エンタルピーを厳密保存する海洋–海氷結合器（`src/aoba/ocean_coupler.py`）、OMIP/mocsy 2.0 炭酸化学・$\mathrm{O}_2$/OMZ 脱窒・大気ダスト鉄/$\mathrm{LNO}_x$/河川溶存物質と結合した FlexPFT 海洋生物地球化学（`src/aoba/ocean_bgc.py`）を備えます。
   - **自己強制オフライン OGCM 深海スピンアップ加速（`scripts/run_offline_ogcm_spinup.py`）**: Phase-1 結合計算の複数年大気フラックスを単独 3D OGCM ＋ FlexPFT BGC ＋ EVP-ITD 海氷へ循環投入し（深海ドリフト `< 0.01 K/century` で自動停止）、数百年〜千年規模の深海・海洋炭素平衡化を大幅に省資源・高速化します。

6. **大気化学・自然エアロゾル・水同位体・動的氷床・古気候/系外惑星拡張**:
   - **予報 3D オゾン・自然エアロゾル・水同位体**: 4 種のオゾンモード（`rcemip`, `latitude_seasonal`, `climatology`, `prognostic` 3D Chapman ＋ $\mathrm{HO}_x/\mathrm{NO}_x/\mathrm{ClO}_x$/PSC 光化学＋Brewer–Dobson 輸送；`src/aoba/ozone_chemistry.py`）、4 種の自然エアロゾル（海塩・FlexPFT 結合 DMS/$\mathrm{SO}_4$・火災 BC/OC・BVOC-SOA；`src/aoba/aerosols.py`）、マップ不要の鉱物ダスト（`src/aoba/dust.py`）、積雪 BC/ダスト暗色化、および安定水同位体（$\mathrm{H}_2^{18}\mathrm{O}$, $\mathrm{HDO}$；`src/aoba/water_isotopes.py`）を搭載します。
   - **任意の排出量駆動・古気候・系外惑星オプション**: 選択式 $\mathrm{CO}_2$ モード（`constant` / `scenario` / `csv` / `prognostic` 3D 輸送）、予報 $\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$（`src/aoba/trace_gas_cycle.py`）、火山・太陽強制（`src/aoba/volcanic_solar.py`）、珪酸塩風化とミランコビッチ軌道変動（`src/aoba/planetary_geochem.py`）、加速 Glen 流動則 SIA 動的氷床（`src/aoba/icesheet.py`）、および惑星プリセット（`earth`, `proxima_b`, `trappist_1e`, `mars_early`；`src/aoba/planetary_presets.py`）を備えます。

7. **101 個の単一ソース融合 Metal/CUDA カーネル、汎用異格子リスタート再補間、322 パラメータ一元管理 `AOBA Studio`**:
   - **単一ソース GPU バックエンド**: `kernels.py`・`kernels_extra.py`・`kernels_physics.py`・`kernels_esm.py` に記述された **全 101 個の融合 GPU カーネル**が、Apple Silicon（`mps` Metal シェーダー）および NVIDIA GPU（`src/aoba/kernels_cuda.py` による決定論的 Metal→CUDA C++17 自動トランスパイル）の双方でネイティブ動作し、自動 CUDA Graphs キャプチャ/リプレイ（`src/aoba/runtime.py`）によりステップ途中の CPU–GPU 同期ゼロを実現しています。
   - **設定管理・プリセット・標準診断**: 全 14 セクション・**322 パラメータ**を [`config/aoba_default.toml`](config/aoba_default.toml) に一元化し（日英 100% 物理解説 `src/aoba/param_docs.py`）、**11 本の即実行プリセット**（`config/presets/`）、任意解像度・層数間（例：`AG4 → AG6`）の汎用リスタート再補間、30 クラス Köppen–Geiger 気候帯診断（`src/aoba/koppen.py`）、CMIP7 Data Request v1.2 出力（`src/aoba/cmip7_diag.py`）、および対話型 Web ワークベンチ `AOBA Studio`（`scripts/aoba_config_gui.py`）を提供します。

8. **単精度（`float32`）ネイティブ設計と長期保存性・数値安定性対策**:
   - **なぜ単精度ベースか**: Apple Silicon（`mps` はネイティブ `float64` 非対応）や汎用 NVIDIA GPU では倍精度演算が強く制限されます。メモリ帯域律速の流体・物理ソルバを `float32` ネイティブ化することで実効メモリ帯域を 2 倍に高め、ラップトップからクラウド GPU まで高速に実行できます。
   - **単精度誤差・ドリフト・振動対策**: ① 3D 海洋鉛直混合・5 層土壌熱伝導/Richards 浸透における増分・テレスコーピングフラックス差分型三重対角陰解法、② 全大容量リザーバ（海洋水温・塩分・陸域/海洋炭素窒素プール）に対する Neumaier–Kahan 補償加算（`_kahan_add`）、③ 基準値偏差移流および `float64` スケール炭酸平衡・初期静水圧積分、④ 水＋海氷総エンタルピー保存型海面結合、⑤ 高樹冠上の $2\Delta t$ 顕熱振動を抑止する $C^\infty$ 滑らかな Kalnay–Kanamitsu 陰的バルク輸送係数リミッタ（`src/aoba/surface.py`）、⑥ C-grid Hollingsworth–Bernoulli 発散減衰（`src/aoba/ocean_ogcm.py`）、⑦ `expm1`/`log1p`・$\Delta\tau \to 0$ Taylor 展開・オーバーフロー回避型 Lambert $W_0$。

## ディレクトリ構成

```text
aoba/
├── README.md                 # プロジェクト概要（英語）
├── README_ja.md              # プロジェクト概要（日本語）
├── config/
│   ├── aoba_default.toml     # 322 パラメータ一元管理 TOML（14 セクション）
│   └── presets/              # 11 本の用途別プリセット（Mac 探索〜ES4/A100 本番）
├── docs/
│   ├── architecture.md       # 支配構成・根拠文献トレーサビリティ仕様書（EN/JA）
│   ├── configurations.md     # 実験構成ガイド・OSSP ＆ GPU 運用指針（EN/JA）
│   ├── formulation.md        # 数理定式化モデル記述書（EN/JA）
│   ├── roadmap.md            # 開発ロードマップ（EN/JA）
│   └── assets/               # アイコン ＆ 概念図 SVG 7 点
├── data/                     # Earth2014 地形・検証データ（ローカル専用）
├── src/
│   └── aoba/                 # PyTorch + 101 融合 Metal/CUDA パッケージ（全 55 モジュール）
├── tests/                    # 単体・保存則・カーネル・OSSP 検証スイート（全 27 本）
├── scripts/                  # 結合実行・オフライン OGCM 加速・GUI・可視化ツール
├── notebooks/                # Colab A100/H100 自動再開ノートブック
└── pyproject.toml
```

## クイックスタート

```bash
# 1. 一括セットアップ（依存確認・Earth2014 地形データ取得・基本テスト）
python setup_aoba.py

# （任意）editable インストール（不要でも scripts/tests は自動パス解決＆自動DL）
pip install --no-build-isolation --no-deps -e .

# 主要検証・OSSP・GPU カーネル・保存則テストの実行
python tests/test_phase1_foundation.py
python tests/test_phase1_physics_biosphere.py
python tests/test_phase2_phase3_global.py
python tests/test_crm_self_distill.py
python tests/test_esm_kernels.py
python tests/test_ogcm_fct_zstar.py
python tests/test_params_and_gui.py

# 日英対話型 Web ワークベンチ（AOBA Studio）の起動
python scripts/aoba_config_gui.py --toml config/aoba_default.toml

# プリセット実行（Mac 探索 / 3D 8x8 OSSP 基準 / 全 17 ESM 要素結合スピンアップ）
python -u scripts/run_g5_30yr.py --config config/presets/mac_g5_quickstart.toml
python -u scripts/run_g5_30yr.py --config config/presets/earth_g5_3d_8x8_g7ocn.toml
python -u scripts/run_g5_30yr.py --config config/presets/earth_g4z64_g7z64_all_esm_spinup.toml

# Phase-2 自己強制オフライン 3D OGCM + FlexPFT BGC + EVP-ITD 深海スピンアップ加速
python -u scripts/run_offline_ogcm_spinup.py \
  --coupled-dir ~/aoba_outputs/<phase1_run> \
  --out-dir ~/aoba_outputs/<phase2_offline> \
  --years 1000 --stop-deep-drift 0.01
```

## 計算出力フォーマット

- **ネイティブ `Zarr v3` 入出力**: モデル本体のヒストリ出力・CMIP7 診断・リスタートチェックポイント・事前計算メッシュはすべて **CF-1.11 / UGRID-1.0 準拠の `Zarr v3 (.zarr)`** に統一し、デフォルトで `~/aoba_outputs`（環境変数 `AOBA_OUTPUT_DIR` で変更可）に保存します。外部解析ツール（`cdo`, Panoply 等）向けの `NetCDF4 (.nc)` 変換は `scripts/zarr_to_netcdf.py` で行います。

## 検証ベンチマークとモデル構成階層

- **単一カラム RCE テストベッド（`scripts/run_rce.py`）**: 1 ホストカラムに 3D C-grid / Charney–Phillips HEVI CRM、MYNN 2.5 / RSL 乱流、選択式放射、Two-Leaf $\mathrm{C}_3/\mathrm{C}_4$ EEO 陸面・スラブ海洋を結合した放射対流平衡（Wing et al., 2018）。
- **全球力学コアベンチマーク（`scripts/run_held_suarez.py`）**: `Earth2014` 平滑化実地形を備えた全球 SCVT ボロノイ C-grid（TRiSK）＋ Charney–Phillips HEVI ホストコアによる Held–Suarez 検証（Held & Suarez, 1994）。
- **OSSP オンライン自己蒸留 ＆ レジーム遷移シャドウ検証**: `piControl`（$284.3\ \mathrm{ppmv}$）から `abrupt-4×CO2`（$1137.2\ \mathrm{ppmv}$）への遷移におけるホスト大気・陸面タイル双方の逐次シャドウ乖離（`enable_distill_shadow_diag = true`）および 30 クラス Köppen–Geiger 気候帯創発（Beck et al., 2018, 2023）の定量評価。
- **マルチスケール・惑星応用実験**: 水惑星ジェット創発、HighResMIP 級 `AG6z64 + OG7z64` 全 ESM 結合計算、中期完新世「緑のサハラ」／氷期サイクル、潮汐固定系外惑星シミュレーション。

## ライセンス

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
