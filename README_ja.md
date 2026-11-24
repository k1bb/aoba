<img src="docs/assets/aoba_icon.png" width="96" align="right" alt="AOBA Icon" />

# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **言語 (Language)**: [English (README.md)](README.md) | 日本語

**AOBA**（**A**utonomous **O**ptimality **B**iosphere–**A**tmosphere）は、生態進化的最適性（EEO: Eco-Evolutionary Optimality）に基づく自律陸面植生モデル、球面重心ボロノイ直交 C-grid 上の 1対1 超パラメータ化（Superparameterization）全球非静力学大気、および選択可能な 3D SCVT ボロノイ OGCM／1.5層エクマン浅水海洋を結合した軽量かつ GPU ネイティブな全球気候モデルです。モデル名は、英語の頭字語であると同時に日本語の**「青葉（あおば）」**にかけており、大気・海洋・陸域生態系の結合系から生命ある緑の植生と気候帯が自律的に芽吹く様子を象徴しています。外部境界条件として**「海陸・氷床分布、地表面標高・海底地形（`Earth2014`）、自転周期、太陽放射、大気ガス濃度」のみ**を与え、積雲パラメタリゼーション、処方海洋 $Q$ フラックス、外部土壌・アルベド地図、および植物機能型（PFT）パラメータ表を排した最小制約系において、地球の気候帯・バイオームがどのように創発し、放射強制や軌道要素の変化に対してどう応答するかを解明することを目指しています。

## 開発の動機とモデル設計思想

### 既存の気候モデルが抱える構造的課題
- **地理的境界条件による現在気候の暗黙的制約**: 植生機能型（PFT）・土壌・アルベドの空間分布データ、処方海洋 $Q$ フラックス、あるいは緯度・地域ごとに経験調整された非地形性重力波源などの地理的境界条件は、現在気候における大気循環・気候帯・バイオームの空間構造を外部入力として暗黙に組み込んでいる。その結果、地球の気候帯や生態系分布がいかにして自律的に創発するかを検証する研究や、放射強制・軌道要素が大きく変化した際の気候帯の再編（熱帯域の拡大や古気候レジーム遷移など）を先入観なく表現するうえで本質的な制約となり得る。
- **積雲パラメタリゼーションのレジーム依存性**: 気候感度・降水分布・雲放射フィードバックは積雲対流スキームの選択に強く依存するが、多くのスキームは熱帯海洋上の深い積雲を念頭に構築されており、陸上対流や浅い積雲・下層層積雲などの多様なレジームにおいて必ずしも整合的な物理描像を与えない。
- **スケール間・親子モデル間の構造的不連続**: 従来の枠組みでは、低解像度の長期気候モデルと高解像度の雲解像モデルが別個のコード基盤に分断されており、超パラメータ化（SP）モデルにおいても静力学ホスト大気と非弾性 CRM のように異なる方程式系・離散化手法を事後的に結合する構成が多く、スケール間の力学的・熱力学的整合性に課題が残る。
- **再現性・可読性・計算環境および教育面での参入障壁**: 非公開のソースコード、公開モデルであっても境界条件や初期値データの入手にモデル開発機関への個別照会を要する運用、特定スパコン環境に依存し GPU 活用が進みにくい設計、歴史的な固定書式・変数名制約を引きずる難読なレガシーコード、各コンポーネントへ散在した設定体系と不透明なジョブ監視、さらに英語のみに限定された技術文書は、学部生やスパコン資源を持たない研究者にとって極めて高い参入障壁となっている。

### AOBA の設計思想
- **最小限の地理的境界条件からの自律創発**: 地表面標高・海底地形（`Earth2014`）および任意のオゾン分布を除き、空間マップ型の地理的境界条件を一切必要としない。気候帯・陸域バイオーム・雲構造・海洋循環・重力波スペクトルを、普遍的な物理法則とグローバルパラメータのみから自律的に創発させる。
- **低解像度 SP 気候計算から高解像度雲解像計算までの単一基盤**: 超パラメータ化（SP）を用いた低解像度・長期気候積分と、SP を用いず直接対流を解像する高解像度雲解像計算（GCRM / CRM）を、同一の物理・数値基盤上でシームレスに実行できる。
- **親モデルと子モデルにおける方程式系・離散化基盤の完全統一**: 全球ホスト大気（親）と埋め込み CRM（子）の双方に同一のフラックス保存型完全圧縮性非静力学オイラー方程式系・直交 Arakawa C-grid・Charney–Phillips 鉛直配置・RK3-HEVI 時間積分・Zalesak FCT 移流・共通物理過程を採用し、異種モデル結合に伴う構造的不整合を排除する。
- **ラップトップからスパコンまで一貫動作する GPU ポータビリティ**: ハードウェア自動検知（`cuda` → `mps` → `cpu`）と単一ソース融合 `float32` GPU カーネルにより、手元のノート PC（Apple Silicon）からクラウド GPU・大型計算機まで同一コードで高速に実行でき、一般流通 GPU の倍精度性能制約を回避しつつ明示的な単精度誤差対策で長期保存性を担保する。
- **完全オープンかつ自己完結した可読性の高い Python 実装**: 詳細な外部地理境界データに依存せず（スピンアップ済み初期値も今後公開予定）、現代的で可読性の高い Python コードと厳密な文献トレーサビリティにより、学生・研究者が物理過程を直接読み解き改変できる研究基盤を提供する。
- **グローバルパラメータの一元管理・日英バイリンガル・統合 GUI/CUI**: 全グローバルパラメータを単一の TOML ファイルに集約し、CUI によるバッチ実行と日英両対応の Web ワークベンチ（`AOBA Studio`：設定編集・実行制御・リアルタイム監視、および今後の簡易解析・可視化機能）の両面から直感的に運用できる。

## 基本設計思想

1. **直交 C-grid × 鉛直 Charney–Phillips 圧縮性非静力学方程式系への統一（純粋 1:1 SCVT ホスト ＋ 3D CRM）**:
   - **全球ホスト力学コア**: **球面重心ボロノイ格子（SCVT）上の直交 Arakawa C-grid** に **TRiSK** 演算子・**鉛直 Charney–Phillips 配置**・**鉛直陰解法（HEVI）** を組み合わせた3次元フラックス保存型完全圧縮性非静力学方程式系を解き、5分格子 `Earth2014` のサブグリッド起伏から直接評価する**低層山岳ブロッキング＋対流圏地形性重力波ドラッグ**（Lott & Miller, 1997）を備えます（Ringler et al., 2008, 2010; Thuburn et al., 2009; Klemp, 2011; Skamarock et al., 2012; Gassmann, 2013）。
   - **1:1 埋め込み CRM ＆ 乱流階層**: 各ホストボロノイセルに周期 CRM（既定 `8 × 1` 2D カーテン $\Delta x = 4\ \mathrm{km}$、または `8 × 8` 3D $\Delta x = 4\ \mathrm{km}$、鉛直44層成層圏拡張・上端 $z_{\mathrm{top}} = 48\ \mathrm{km}$；Khairoutdinov & Randall, 2003; Khairoutdinov et al., 2005; Yamazaki & Miura, 2024; Nishizawa et al., 2025）を埋め込み、次元に応じたコリオリ処理（1D/2D は無効、3D は擾乱風のみ）、鉛直境界層・層積雲乱流に **MYNN Level 2.5**（Nakanishi & Niino, 2009）、3D CRM 内に非等方 **Smagorinsky–Lilly** SGS クロージャを用います。
2. **PFT フリーの Two-Leaf C₃/C₄ 最適性生態系・多層凍土熱伝導・積雪物理**:
   - **Two-Leaf EEO 順化と気孔閉合**: Smith et al. (2019) および Scott & Smith (2022) の非直角双曲線型最適性 P モデルに準拠し、各コラムの地方太陽正午における下向き短波放射（直達・散乱）を指数移動平均（EMA）した後に樹冠2ストリーム光吸収計算を行って順化光環境を求め、順化時の吸収 PPFD 鉛直分布に比例させて $V_{c\max 25}, J_{\max 25}, V_{p\max 25}$ および葉暗呼吸 $R_{25}$ を陽葉・陰葉（Two-Leaf）へ閉形式解析配分します（de Pury & Farquhar, 1997; Dai et al., 2004; Prentice et al., 2014; Heskel et al., 2016; Smith et al., 2019; Mengoli et al., 2022; Scott & Smith, 2022; Luo et al., 2024; Ren et al., 2024, 2025; Lavergne et al., 2026）。
   - **陸面水文・土壌水分最適応答・永久凍土熱力学**: 蒸発散を陽葉／陰葉蒸散・濡れ樹冠遮断蒸発・安定度補正付き裸地土壌蒸発の3成分に分離し（Dai et al., 2004; Sakaguchi & Zeng, 2009; Lee & Khairoutdinov, 2015; MATSIRO6, 2021）、気候適応型節水タイムスケール $\tau(D_{\max})$ に基づく時間積分最適土壌水分応答（Lu et al., 2016; Holtzman et al., 2024）、最適根圏貯水容量 $S_R$（Gao et al., 2014; Davis et al., 2017）、冬季凍害を含む自律的常緑／落葉最適性を備えた予報 EEO $\mathrm{LAI}$（Kikuzawa, 1991; Sakai & Larcher, 1987; Givnish, 2002; Wang et al., 2023; Cai et al., 2025; Zhou et al., 2025）、土壌水分依存の2バンド裸地アルベド（Oleson et al., 2013）、凍結・融解相変化を伴う多層土壌熱伝導（SCALE-RM, 2025; MATSIRO6, 2021）、および部分積雪被覆率・雪埋没樹冠マスキング・2バンド積雪エージング・海洋還元水収支（Yang et al., 1997; Swenson & Lawrence, 2012; Nitta et al., 2014; Abolafia-Rosenzweig et al., 2021）を統合します。
3. **変位追従型の平滑化地形基準面と最新粗度サブレイヤ（RSL）**: 局所の水・エネルギー制限から最大樹冠高 $h_c$ を決定し（Kempes et al., 2011）、ゼロ面変位 $d$・粗度長 $z_{0m}$・解析的 RSL 安定度補正（Raupach, 1994; Harman & Finnigan, 2007; De Ridder, 2010; Bonan et al., 2018; Lee et al., 2020; Meier et al., 2022）を実効基準面 $z_{\mathrm{datum}} = z_{\mathrm{topo}} + d$（Klemp, 2011）上で接続します。
4. **16バンド Correlated-k 放射・2モーメント雲微物理・3D OGCM／浅水選択式海洋**: 既定で決定論的 16バンド（50 g点）Correlated-$k$ 放射ソルバ（`rad_scheme = "ckd"` / `"four_band"`、$\mathrm{H_2O}, \mathrm{CO_2}, \mathrm{O_3}, \mathrm{CH_4}, \mathrm{N_2O}$ 対応）、マップ不要の自然海塩・ダスト CCN/INP 活性化を備える 2モーメント雲微物理（`microphysics_scheme = "two_moment"` / `"one_moment"`）、および `Earth2014 BED2014` 海底地形・塩分・Roquet 2015 非線形 EOS・GM90＋フル回転 Redi テンソル・KPP 境界層＆非局所対流・Ri シア混合・地形性内部潮汐深海混合・Jerlov 短波透過・Hibler 1979 動的海氷漂流を備えた 18層 3D SCVT OGCM（`ocean_mode = "ogcm_3d"`）と軽量な 1.5層エクマン浅水海洋（`ocean_mode = "ekman_sw"`）を自由に切り替えて実行できます。
5. **動的・平衡加速 SIA 氷床結合 ＆ 単一 TOML パラメータ一元管理＋対話型 GUI**:
   - **動的氷床モデル**: `Earth2014` 岩盤標高上に Glen 流動則 ($n=3$) 浅氷近似（SIA）＋EMA 気候 SMB 平滑＋アイソスタシー＋スピンアップ加速率を備え、氷床消長に伴う地形変化を測高外挿・対数気圧鉛直再補間により大気・CRM 座標へ無矛盾に反映します（`src/aoba/icesheet.py`）。
   - **モデル設定と Web Studio (`AOBA Studio`)**: 全14セクション・256項目の CMIP6 1850 `piControl` 既定値・成層圏拡張格子設定（標準 `nz = 44`, `z_top = 48 km`、低トップ 24 km 切替可）・カレンダー／固定日射設定・時間変動 CO₂・オゾン強制・マップ不要の流れ依存非地形性重力波・出力／月末リスタート設定・物理定数を単一ファイル [`config/aoba_default.toml`](config/aoba_default.toml) に集約し、Web GUI ([`scripts/aoba_config_gui.py`](scripts/aoba_config_gui.py)) で設定・監視・**CPU 処理による T₂ / 日平均降水量 / LAI の回転3地球正射図法リアルタイムプレビュー**を提供します。
6. **ハードウェア自動検知・単一ソース GPU 高速化（全43融合 Metal / CUDA カーネル）**: 実行環境のアクセラレータ（`cuda` → `mps` → `cpu`）を自動検知し、Apple Silicon GPU（`mps` Metal シェーダー）および NVIDIA GPU（`src/aoba/kernels_cuda.py` による単一ソース自動 CUDA C++17 トランスパイル）の両方で全43カーネルをネイティブ実行します。Google Colab（A100/H100）で Google Drive へ自動チェックポイント保存・レジュームできるノートブック（[`notebooks/aoba_colab_spinup.ipynb`](notebooks/aoba_colab_spinup.ipynb)）も同梱しています。
7. **単精度（`float32`）ベース実装の理由と数値精度・長期保存性対策**:
   - **なぜ単精度ベースか**: 一般に流通する GPU（ネイティブ `float64` を持たない Apple Silicon `mps` や、FP64 性能が FP32 の 1/32〜1/64 に制限されるコンシューマ・汎用クラウド NVIDIA GPU など）では倍精度演算のパフォーマンスが大きく制約されます。また、大気・海洋の偏微分方程式ソルバはメモリ帯域律速であるため、`float32` を基本とすることでメモリ転送量を半減し、手元のノート PC からクラウド GPU まで実用的な高速計算を実現しています。
   - **単精度に起因する誤差・ドリフト防止策**: 桁落ち・丸め停滞（ULP 未満の微小傾向の消失）・数十年〜百年積分での保存則ドリフトを防ぐため、AOBA では以下を標準実装しています：① 3D 海洋鉛直混合（`src/aoba/ocean_ogcm.py`）および多層土壌熱伝導・Richards 浸透（`src/aoba/surface.py`）における**増分・厳密テレスコーピングフラックス差分型三重対角陰解法**（深海の微小拡散係数 $c_{\mathrm{dif}}/h < \mathrm{ulp}(1.0\text{f})/2$ による対角丸め落ち由来の偽の熱・塩分生成を完全に排除）、② 海洋 3D 水温・塩分、混合層水温、Jerlov 短波透過、氷河カービング、海面塩分フラックスに対する **Neumaier–Kahan 補償加算**（`_kahan_add`）、③ 3D 海洋移流の**基準値偏差形式**（$T - T_{\mathrm{ref}},\ S - S_{\mathrm{ref}}$）および偏差状態方程式、④ SCVT ホスト・CRM 初期静水圧場の **`float64` 鉛直積分**、⑤ 光学的薄層（`src/aoba/radiation_ckd.py` の $\Delta\tau \to 0$）における **Taylor 展開**および雲微物理の非整数冪下限保護（`src/aoba/microphysics.py`）。

## ディレクトリ構成

```text
aoba/
├── README.md              # プロジェクト概要（英語）
├── README_ja.md           # プロジェクト概要（日本語）
├── docs/
│   ├── architecture.md    # 支配構成・根拠文献トレーサビリティ仕様書（英語）
│   ├── architecture_ja.md # 支配構成・根拠文献トレーサビリティ仕様書（日本語）
│   ├── formulation.md     # 数理定式化モデル記述書（英語）
│   ├── formulation_ja.md  # 数理定式化モデル記述書（日本語）
│   ├── roadmap.md         # 将来の開発課題・ロードマップ（英語）
│   └── roadmap_ja.md      # 将来の開発課題・ロードマップ（日本語）
├── data/                  # 境界地形データ（Earth2014）・検証データセット
├── src/
│   └── aoba/              # PyTorch + Metal/CUDA モデル本体パッケージ（全37モジュール）
├── tests/                 # 単体・物理強制力・保存則テスト（CPU + MPS 両対応）
├── scripts/               # RCE・Held-Suarez・全球AOBA実行スクリプト ＆ NetCDF変換ツール
└── pyproject.toml
```

## クイックスタート

```bash
# 1. 一括セットアップ（依存ライブラリ確認・Earth2014 地形データ取得・テスト・スモークラン）
python setup_aoba.py

# （任意）editable インストール（不要でも全テスト・スクリプトは自動パス解決＆初回地形自動DLされます）
pip install --no-build-isolation --no-deps -e .

# 全検証テストスイートの実行（CPU および Apple Silicon MPS GPU 両対応）
python tests/test_phase1_foundation.py
python tests/test_phase1_physics_biosphere.py
python tests/test_phase1_crm_rce.py
python tests/test_phase2_phase3_global.py
python tests/test_params_and_gui.py

# 3D RCE・Held-Suarez・全球 1:1 超パラメータ化 AOBA の実行（CF/UGRID Zarr v3 出力）
python scripts/run_rce.py --surface land --days 2.0
python scripts/run_held_suarez.py --glevel 3 --days 5.0
python scripts/run_global_aoba.py --glevel 3 --co2-ppmv 400 --days 5.0

# 統合 TOML 設定・GUI コンフィギュレータ (AOBA Studio) の起動とマルチ年実行
python scripts/aoba_config_gui.py --toml config/aoba_default.toml
python -u scripts/run_g5_30yr.py --config config/aoba_default.toml
```

## 計算出力フォーマット

- **ネイティブ `Zarr v3` 入出力**: モデル本体のヒストリ出力・リスタートファイル・事前計算メッシュはすべて **CF/UGRID 準拠の `Zarr v3 (.zarr)`** に統一し、デフォルトで `~/aoba_outputs`（環境変数 `AOBA_OUTPUT_DIR` で変更可）に出力します。外部ツール（`cdo` や Panoply 等）向けの `NetCDF4 (.nc)` 変換は後処理スクリプト `scripts/zarr_to_netcdf.py` で行います。

## 検証ベンチマークとモデル構成階層

- **単一カラム RCE テストベッド**: 1 ホストカラム（`B = 1`）に `8 × 8` 3D C-grid / Charney–Phillips HEVI CRM、MYNN 2.5 / RSL 乱流、4バンド放射、サブグリッド Two-Leaf C₃/C₄ EEO 陸面・スラブ海洋を結合した放射対流平衡（RCE）構成（Wing et al., 2018）。
- **全球力学ベンチマーク**: `Earth2014` 平滑化実地形・Lott & Miller (1997) サブグリッド山岳ドラッグを備えた全球 SCVT ボロノイ C-grid（TRiSK）＋ Charney–Phillips HEVI 非静力学ホストコアによる Held–Suarez 検証（Held & Suarez, 1994）。
- **全球 1:1 結合気候シミュレーション**: 大陸・実地形・1.5層エクマン海洋＋部分海氷上での `1:1` 全球スーパーパラメタリゼーション結合と、観測ケッペン・ガイガー気候帯（Beck et al., 2018, 2023）に対する比較検証。
- **気候実験・要因分離**: 自転速度変更、固定日射実験、高 CO₂ 下における気候帯シフトの要因分離（CO₂ 放射強制力 vs 気孔・C₃/C₄・LAI 生理学的応答、風成海洋熱輸送応答）。

## ライセンス

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
