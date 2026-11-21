<img src="docs/assets/aoba_icon.png" width="96" align="right" alt="AOBA Icon" />

# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **言語 (Language)**: [English (README.md)](README.md) | 日本語

**AOBA** は、生態進化的最適性（EEO: Eco-Evolutionary Optimality）に基づく自律陸面植生モデル、球面重心ボロノイ直交 C-grid 上の 1対1 超パラメータ化（Superparameterization）全球非静力学大気、および風成エクマン熱輸送スラブ海洋を結合した軽量かつ GPU ネイティブな全球気候モデルです。外部境界条件として**「海陸・氷床分布、地表面標高（`Earth2014`）、自転周期、太陽放射、大気ガス濃度」のみ**を与えたとき、大気循環・海洋表層・陸面生態系の結合系が地球の気候帯を自律的に再現できるか、そして経験的制約を最小限に抑えたモデルが将来の気候帯シフトをどう予測するかを解明するために設計されています（積雲パラメタリゼーション、処方海洋 $Q$ フラックス、外部土壌・アルベド地図、および植物機能型［PFT］パラメータ表を一切用いません）。

## 基本設計思想

1. **直交 C-grid × 鉛直 Charney–Phillips 圧縮性非静力学方程式系への統一（純粋 1:1 SCVT ホスト ＋ 3D CRM）**:
   - **全球ホスト力学コア**: **球面重心ボロノイ格子（SCVT）上の直交 Arakawa C-grid** に **TRiSK** 演算子・**鉛直 Charney–Phillips 配置**・**鉛直陰解法（HEVI）** を組み合わせた3次元フラックス保存型完全圧縮性非静力学方程式系を解き、5分格子 `Earth2014` のサブグリッド起伏から直接評価する**低層山岳ブロッキング＋対流圏地形性重力波ドラッグ**（Lott & Miller, 1997）を備えます（Ringler et al., 2008, 2010; Thuburn et al., 2009; Klemp, 2011; Skamarock et al., 2012; Gassmann, 2013）。
   - **1:1 埋め込み CRM ＆ 乱流階層**: 各ホストボロノイセルに周期 CRM（既定 `8 × 1` 2D カーテン $\Delta x = 8\ \mathrm{km}$、または `8 × 8` 3D $\Delta x = 4\ \mathrm{km}$、鉛直32層・上端 $z_{\mathrm{top}} \approx 24\ \mathrm{km}$；Khairoutdinov & Randall, 2003; Khairoutdinov et al., 2005; Yamazaki & Miura, 2024; Nishizawa et al., 2025）を埋め込み、次元に応じたコリオリ処理（1D/2D は無効、3D は擾乱風のみ）、鉛直境界層・層積雲乱流に **MYNN Level 2.5**（Nakanishi & Niino, 2009）、3D CRM 内に非等方 **Smagorinsky–Lilly** SGS クロージャを用います。
2. **PFT フリーの Two-Leaf C₃/C₄ 最適性生態系・多層凍土熱伝導・積雪物理**:
   - **Two-Leaf EEO 順化と気孔閉合**: Smith et al. (2019) および Scott & Smith (2022) の非直角双曲線型最適性 P モデルに準拠し、各コラムの地方太陽正午における下向き短波放射（直達・散乱）を指数移動平均（EMA）した後に樹冠2ストリーム光吸収計算を行って順化光環境を求め、順化時の吸収 PPFD 鉛直分布に比例させて $V_{c\max 25}, J_{\max 25}, V_{p\max 25}$ および葉暗呼吸 $R_{25}$ を陽葉・陰葉（Two-Leaf）へ閉形式解析配分します（de Pury & Farquhar, 1997; Dai et al., 2004; Prentice et al., 2014; Heskel et al., 2016; Smith et al., 2019; Mengoli et al., 2022; Scott & Smith, 2022; Luo et al., 2024; Ren et al., 2024, 2025; Lavergne et al., 2026）。
   - **陸面水文・土壌水分最適応答・永久凍土熱力学**: 蒸発散を陽葉／陰葉蒸散・濡れ樹冠遮断蒸発・安定度補正付き裸地土壌蒸発の3成分に分離し（Dai et al., 2004; Sakaguchi & Zeng, 2009; Lee & Khairoutdinov, 2015; MATSIRO6, 2021）、気候適応型節水タイムスケール $\tau(D_{\max})$ に基づく時間積分最適土壌水分応答（Lu et al., 2016; Holtzman et al., 2024）、最適根圏貯水容量 $S_R$（Gao et al., 2014; Davis et al., 2017）、予報 EEO $\mathrm{LAI}$（Cai et al., 2025; Zhou et al., 2025）、土壌水分依存の2バンド裸地アルベド（Oleson et al., 2013）、凍結・融解相変化を伴う多層土壌熱伝導（SCALE-RM, 2025; MATSIRO6, 2021）、および部分積雪被覆率・雪埋没樹冠マスキング・2バンド積雪エージング・海洋還元水収支（Yang et al., 1997; Swenson & Lawrence, 2012; Nitta et al., 2014; Abolafia-Rosenzweig et al., 2021）を統合します。
3. **変位追従型の平滑化地形基準面と最新粗度サブレイヤ（RSL）**: 局所の水・エネルギー制限から最大樹冠高 $h_c$ を決定し（Kempes et al., 2011）、ゼロ面変位 $d$・粗度長 $z_{0m}$・解析的 RSL 安定度補正（Raupach, 1994; Harman & Finnigan, 2007; De Ridder, 2010; Bonan et al., 2018; Lee et al., 2020; Meier et al., 2022）を実効基準面 $z_{\mathrm{datum}} = z_{\mathrm{topo}} + d$（Klemp, 2011）上で接続します。
4. **4バンド解析的2ストリーム放射 ＆ ボロノイ C-grid 1.5層エクマン海洋・部分海氷**: ルックアップテーブル不要の4バンド2ストリーム放射ソルバ（Lacis & Hansen, 1974; Toon et al., 1989; Slingo, 1989; Ebert & Curry, 1992; Weaver & Ramanathan, 1995; Koll & Cronin, 2018）と、処方 $Q$ フラックス不要の 1.5層風成エクマン輸送＋渦拡散スラブ海洋（Codron, 2012）、部分海氷面積率（リード）を伴う熱力学海氷（Semtner, 1976; SCALE-RM, 2025）、太陽天頂角依存海面アルベド（Briegleb et al., 1986）、および滑面粘性＋Charnock 海面粗度（Miller et al., 1992）を結合します。
5. **動的・平衡加速 SIA 氷床結合 ＆ 単一 TOML パラメータ一元管理＋対話型 GUI**:
   - **動的氷床モデル**: `Earth2014` 岩盤標高上に Glen 流動則 ($n=3$) 浅氷近似（SIA）＋EMA 気候 SMB 平滑＋アイソスタシー＋スピンアップ加速率を備え、氷床消長に伴う地形変化を測高外挿・対数気圧鉛直再補間により大気・CRM 座標へ無矛盾に反映します（`src/aoba/icesheet.py`）。
   - **モデル設定と Web Studio (`AOBA Studio`)**: 全195項目の格子設定（`glevel`, `nz`, CRM 格子数・間隔）・カレンダー／固定日射設定・惑星外力・出力／月末リスタート設定・物理定数を単一ファイル [`config/aoba_default.toml`](config/aoba_default.toml) に集約し、Web GUI ([`scripts/aoba_config_gui.py`](scripts/aoba_config_gui.py)) で設定の視覚化・ハードウェア自動チャンク最適化・実行監視・**CPU 処理による T₂ / 日平均降水量 / LAI の回転3地球正射図法リアルタイムプレビュー**を提供します。
6. **単一コードでの GPU ポータビリティ（`Float32` PyTorch ＋ Metal/CUDA 融合鉛直 HEVI ソルバ）**: Apple Silicon GPU（`mps`、Kahan 補償加算による海洋長期保存対応）および NVIDIA GPU（`cuda`）で単一コードのまま計算を実行できます。

## ディレクトリ構成

```text
aoba/
├── README.md              # プロジェクト概要（英語）
├── README_ja.md           # プロジェクト概要（日本語）
├── docs/
│   ├── architecture.md    # 支配構成・根拠文献トレーサビリティ仕様書（英語）
│   ├── architecture_ja.md # 支配構成・根拠文献トレーサビリティ仕様書（日本語）
│   ├── formulation.md     # 数理定式化モデル記述書（英語）
│   └── formulation_ja.md  # 数理定式化モデル記述書（日本語）
├── data/                  # 境界条件（Earth2014）・検証データ（Köppen-Geiger）（Git完全除外）
├── references/            # 参考文献・論文PDF保管用（計74本；Git完全除外）
├── src/
│   └── aoba/              # PyTorch + Metal/CUDA モデル本体パッケージ（全15モジュール）
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

## 出力ファイルとローカルディレクトリの保護方針

- **計算出力の一本化（`Zarr v3` ネイティブ）**: モデル本体のヒストリ出力・リスタートファイル・事前計算メッシュはすべて **CF/UGRID 準拠の `Zarr v3 (.zarr)` 一本**に統一し、デフォルトで `~/aoba_outputs`（環境変数 `AOBA_OUTPUT_DIR` で変更可）に出力します。外部ツール（`cdo` や Panoply 等）向けの `NetCDF4 (.nc)` 変換は独立した後処理スクリプト `scripts/zarr_to_netcdf.py` で行います。
- **参考文献・データディレクトリの三重ロック保護**: `references/` と `data/` は Google Drive 上で自動バックアップされつつ、`.gitignore`、`.git/info/exclude`、および Git `pre-commit` フックにより GitHub へは一切アップロードされません。

## 検証ベンチマークとモデル構成階層

- **単一カラム RCE テストベッド**: 1 ホストカラム（`B = 1`）に `8 × 8` 3D C-grid / Charney–Phillips HEVI CRM、MYNN 2.5 / RSL 乱流、4バンド放射、サブグリッド Two-Leaf C₃/C₄ EEO 陸面・スラブ海洋を結合した放射対流平衡（RCE）構成（Wing et al., 2018）。
- **全球力学ベンチマーク**: `Earth2014` 平滑化実地形・Lott & Miller (1997) サブグリッド山岳ドラッグを備えた全球 SCVT ボロノイ C-grid（TRiSK）＋ Charney–Phillips HEVI 非静力学ホストコアによる Held–Suarez 検証（Held & Suarez, 1994）。
- **全球 1:1 結合気候シミュレーション**: 大陸・実地形・1.5層エクマン海洋＋部分海氷上での `1:1` 全球スーパーパラメタリゼーション結合と、観測ケッペン・ガイガー気候帯（Beck et al., 2018, 2023）に対する比較検証。
- **気候実験・要因分離**: 自転速度変更、固定日射実験、高 CO₂ 下における気候帯シフトの要因分離（CO₂ 放射強制力 vs 気孔・C₃/C₄・LAI 生理学的応答、風成海洋熱輸送応答）。

## ライセンス

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
