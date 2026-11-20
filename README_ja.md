# AOBA (Autonomous Optimality Biosphere–Atmosphere)

> **言語 (Language)**: [English (README.md)](README.md) | 日本語

**AOBA** は、生態進化的最適性（EEO: Eco-Evolutionary Optimality）に基づく自律陸面植生モデル、球面重心ボロノイ直交 C-grid 上の超パラメータ化（Superparameterization）全球非静力学大気、および風成エクマン熱輸送スラブ海洋を結合した軽量かつ GPU ネイティブな全球気候モデルです。外部境界条件として**「海陸・氷床分布、地表面標高（`Earth2014`）、自転周期、太陽放射、大気ガス濃度」のみ**を与えたとき、大気循環・海洋表層・陸面生態系の結合系が地球の気候帯を自律的に再現できるか、そして経験的制約を最小限に抑えたモデルが将来の気候帯シフトをどう予測するかを解明するために設計されています（積雲パラメタリゼーション、処方海洋 $Q$ フラックス、および植物機能型［PFT］パラメータ表を一切用いません）。

## 基本設計思想

1. **直交 C-grid × 鉛直 Charney–Phillips 圧縮性非静力学方程式系への統一（SCVT ホスト ＋ 3D CRM）**:
   - **全球ホスト力学コア**: **球面重心ボロノイ格子（SCVT）上の直交 Arakawa C-grid** に **TRiSK**（地衡流モード厳密維持・エネルギー／渦位整合演算子）と **鉛直 Charney–Phillips 配置**、および **鉛直陰解法（HEVI）** を組み合わせた3次元フラックス保存型完全圧縮性非静力学方程式系を解きます（Ringler et al., 2008, 2010; Thuburn et al., 2009; Klemp, 2011; Skamarock et al., 2012; Gassmann, 2013）。極特異点、山岳脇のスペクトル・ギブス振動（偽降水）、および A-grid 特有の鉛直流 $w$ の格子ノイズを根本から排除します。
   - **埋め込み 3D CRM（スーパーパラメタリゼーション）**: ホストと同じ圧縮性非静力学 C-grid／Charney–Phillips HEVI 方程式による周期 `8 × 8` 3D CRM（$\Delta x = \Delta y = 4\ \mathrm{km}$、鉛直32層・上端 $z_{\mathrm{top}} \approx 24\ \mathrm{km}$；Khairoutdinov & Randall, 2003; Khairoutdinov et al., 2005; Nishizawa et al., 2025）を埋め込みます。`Glevel-4`（2,562セル・約 240 km）での 1:1 結合に加え、`Glevel-5`（10,242セル・**約 120 km 解像度**）の親セル4個ごとに1個の CRM（計 2,562個）を共有する 4:1 ブロック結合（Yamazaki & Miura, 2024）にも対応します。
2. **PFT フリーの C₃/C₄ 自律植生・予報 LAI・最適根圏貯水容量**: 陸面タイルを大気コラムと直結し（Lee & Khairoutdinov, 2015; Lin et al., 2023）、最小コスト仮説に基づく C₃・C₄ 光合成順化および C₄ 優占率推定（Prentice et al., 2014; Wang et al., 2017; Stocker et al., 2020; Mengoli et al., 2022; Scott & Smith, 2022; Luo et al., 2024; Lavergne et al., 2026）、EEO 予報葉面積モデル（Cai et al., 2025; Zhou et al., 2025）、および乾季水収支から自律決定される根圏貯水容量（Gao et al., 2014; Davis et al., 2017）により、土壌乾燥応答を経験的ストレス関数ではなく根圏水供給制限と LAI 変動によって表現します。
3. **変位追従型の平滑化地形基準面と自律樹冠構造**: 局所の水・エネルギー制限から最大樹冠高 $h_c$ を決定し（Kempes et al., 2011）、$h_c$ と $\mathrm{LAI}$ の連続関数としてゼロ面変位 $d$ と粗度長 $z_{0m}$ を算出（Raupach, 1994; Harman & Finnigan, 2007）、平滑化高度座標（Klemp, 2011）の下端を実効基準面 $z_{\mathrm{datum}} = z_{\mathrm{topo}} + d$ に置きます。
4. **4バンド解析的2ストリーム放射 ＆ ボロノイ C-grid 1.5層エクマン海洋**: ルックアップテーブル不要の4バンド2ストリーム放射ソルバ（Lacis & Hansen, 1974; Toon et al., 1989; Slingo, 1989; Ebert & Curry, 1992; Weaver & Ramanathan, 1995; Koll & Cronin, 2018）と、ホストと同一のボロノイ C-grid 上で解く処方 $Q$ フラックス不要の 1.5層風成エクマン輸送＋渦拡散スラブ海洋（Codron, 2012）および熱力学海氷（Semtner, 1976）を結合します。
5. **単一コードでの GPU ポータビリティ（`Float32` PyTorch ＋ 融合鉛直 HEVI ソルバ）**: Apple Silicon GPU（Mac mini `M4 Pro`、メモリ使用量 約 1〜1.5 GB）で1日あたり約 7〜11 年分の全球気候積分が動作し、NVIDIA GPU（`cuda`）へそのままスケールします。

## ディレクトリ構成

```text
aoba/
├── README.md              # プロジェクト概要（英語）
├── README_ja.md           # プロジェクト概要（日本語）
├── docs/
│   ├── architecture.md    # 支配方程式・根拠文献トレーサビリティ仕様書（英語）
│   └── architecture_ja.md # 支配方程式・根拠文献トレーサビリティ仕様書（日本語）
├── data/                  # 境界条件（Earth2014）・検証データ（Köppen-Geiger）（Git完全除外）
├── references/            # 参考文献・論文PDF保管用（Git完全除外）
├── src/
│   └── aoba/
│       ├── __init__.py
│       └── runtime.py     # GPUデバイス自動選択（MPS / CUDA / CPU）とパス管理
├── tests/                 # 単体テスト・保存則テスト
├── scripts/               # ベンチマーク・実験実行スクリプト
└── pyproject.toml
```

## 出力ファイルとローカルディレクトリの保護方針

- **計算出力の一本化（`Zarr v3` ネイティブ）**: モデル本体のヒストリ出力・リスタートファイル・事前計算メッシュはすべて **CF/UGRID 準拠の `Zarr v3 (.zarr)` 一本**に統一し、デフォルトで `~/aoba_outputs`（環境変数 `AOBA_OUTPUT_DIR` で変更可）に出力します。外部ツール（`cdo` や Panoply 等）向けの `NetCDF4 (.nc)` 変換はモデル本体から完全に切り離し、独立した後処理スクリプトで行います。
- **参考文献・データディレクトリの三重ロック保護**: `references/` と `data/` は Google Drive 上で自動バックアップされつつ、`.gitignore`、`.git/info/exclude`、および Git `pre-commit` フックにより GitHub へは一切アップロードされません。

## 開発ロードマップ

- **Phase 1**: 単一ホストカラム（`B = 1`）での `8 × 8` 3D C-grid／Charney–Phillips HEVI CRM ＋ 4バンド放射 ＋ サブグリッド C₃/C₄ EEO 陸面／スラブ海洋による放射対流平衡（RCE）テストベッドの構築と検証（Wing et al., 2018）。
- **Phase 2**: `Earth2014` 平滑化実地形を備えた全球 SCVT ボロノイ C-grid（TRiSK）＋ Charney–Phillips HEVI 非静力学ホストコアの構築と Held–Suarez ベンチマーク検証（Held & Suarez, 1994）。
- **Phase 3**: 実大陸・実地形・1.5層エクマン海洋＋海氷上での全球スーパーパラメタリゼーション結合（`Glevel-4` 1:1 SP および `Glevel-5` 4:1 ブロック SP）と、観測ケッペン・ガイガー気候帯（Beck et al., 2018, 2023）に対する自律再現検証。
- **Phase 4**: 高 CO₂ 下における将来気候帯シフト予測と要因分離実験（CO₂ 放射強制力 vs 気孔・C₃/C₄・LAI 生理学的応答、および風成海洋熱輸送応答の切り分け）。

## ライセンス

MIT License. Copyright (c) 2026 Keiichi Hashimoto.
