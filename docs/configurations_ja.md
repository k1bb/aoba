# AOBA 実験構成ガイド（用途別推奨コンフィギュレーション集）

AOBA は単一のコードベースと統一パラメータレジストリ（`config/aoba_default.toml`）から、個人ラップトップ（Apple Silicon Mac）での手軽な物理確認から、スーパーコンピュータ（NVIDIA A100/H100）を用いた HighResMIP 級の高解像度大循環創発実験や完全結合地球システムモデル（Full ESM）実験までシームレスに構成できる。

本章では代表的な 9 つの実験構成について、物理設計の意図・具体的な TOML パラメータ設定・推奨計算資源と実行速度の目安をまとめる。すべての構成は `config/presets/` に即実行可能な TOML ファイルとして同梱されており、`python scripts/run_g5_30yr.py --config config/presets/<file>.toml` または AOBA Studio の設定読込からそのまま起動できる。

---

## 1. 用途別構成の比較早見表

| 構成名 | プリセット TOML (`config/presets/`) | 大気格子 (`glevel` / `nz` / CRM) | 海洋格子 (`ocean_glevel` / `ocean_nz`) | 放射 / オゾン | 物質循環・追加物理 | 推奨環境（1 GPU） | 実行速度の目安 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Mac お試し構成** | `mac_g5_quickstart.toml` | `5` (110 km) / `32` (42 km) / `4×1` | 1.5 層 Ekman–浅水 (`ekman_sw`、または `ogcm_3d` 18層) | `ckd` or `four_band` / `latitude_seasonal` | 生物物理オン、BGC 類オフ | Apple Silicon Mac (MPS, 16–32 GB) | **~3.8 時間/年**（MPS 実測 `37.7 秒/日`, 初期化 `4.5 秒`） |
| **2. 大循環創発（アクアプラネット）** | `aquaplanet_jet_emergence.toml` | `6` (55 km) / `56` (65 km) / `8×1` | 1.5 層 Ekman–浅水 (`h_ml=30 m`)、陸地なし | `ckd` or `rrtmgp` / `prognostic` or `rcemip` | すべてオフ（純粋力学＋対流＋放射） | A100 / 4090 / M3–M4 Max | ~2–7 時間/年 |
| **3. 大循環創発（現実地球・高解像度物理コア）** | `earth_g6g7_rrtmgp_o3.toml` | `6` (55 km) / `56` (65 km) / `8×1` | `7` (28 km, 渦許容) / `36` (表層 15 m) | `rrtmgp` (240 g-pt) / `prognostic` | 生物物理フルオン、雷・他 BGC オフ | NVIDIA A100 (40/80 GB) / H100 | **~10.1 時間/年**（A100 実測 `99.9 秒/日`, `14.1 GB VRAM`; `nz=44` 時は **`8.1 時間/年`**, `80.1 秒/日`） |
| **4. 長期結合バランス構成（Phase-1 PI-Control）** | `earth_g5_phase1_picontrol.toml` | `5` (110 km) / `44` (48 km) / `8×1` | `0` (同 110 km) / `18` | `ckd` (50 g-pt) / `latitude_seasonal` | 生物物理＋ダスト＋自然エアロゾル＋EVP 海氷 | NVIDIA A100 (40 GB) | **~2.2 時間/年**（A100 実測 `21.8 秒/日`） |
| **5. フル ESM 構成** | `earth_full_esm.toml` | `5–6` / `56` (65 km) / `8×1` | `6–7` / `32–36` | `rrtmgp` or `ckd_hd` / `prognostic` | 全 BGC・動的湖・MAML・波浪・同位体・氷床オン | NVIDIA A100/H100 (40–80 GB) | ~6–28 時間/年 |
| **6. 古気候（6 ka 緑のサハラ）・系外惑星** | `paleo_green_sahara_6ka.toml` | `4–5` / `44` (48 km) / `8×1` | `0` (`ogcm_3d` 18層 or `ekman_sw`) | `ckd` or `rrtmgp` / `prognostic` | 軌道進化＋動的湖沼（`init_fill=0`）、または `planet_preset` | Mac MPS / A100 | ~1–4 時間/年 |
| **7. 対流自己組織化（2D `RCE_large`）＋渦許容海洋** | `earth_g5_rcelarge_g7ocn.toml` | `5` (110 km) / `44` (48 km) / `64×1` (256 km, `sfc_tiles=8`) | `7` (28 km, 渦許容) / `18` | `ckd` / `latitude_seasonal`（CRM＋放射ペア蒸留 `1/31`, `dwell=240`） | 生物物理＋ダスト＋自然エアロゾル＋EVP 海氷＋50% 永続 RCEMIP アンカー | Apple M4 Pro / NVIDIA A100 | **~15.8 時間/年**（M4 Pro MPS 実測 `1.30 秒/step`, 初期化 `1.28 秒`; A100 推定 ~3 時間/年） |
| **8. ハイトップ 3D `8×8` 気候ラン基本構成＋渦許容海洋** | `earth_g5_3d_8x8_g7ocn.toml` | `5` (110 km) / `56` (65 km) / `8×8` 3D (32×32 km, 1:1 `8×8` MAML+ICA) | `7` (28 km, 渦許容) / `36` (`ocean_stride=3`) | `rrtmgp`＋3D TICA / `prognostic`（CRM＋放射ペア蒸留 `32` 教師） | `six_class_2m` 6 クラス 2M 雲微物理＋非地形 GWD＋海流相対風＋EVP 海氷＋50% 永続アンカー | Apple M4 Pro (48 GB) / NVIDIA A100 | **~5.3 時間/年**（M4 Pro MPS 実測 `0.43 秒/step`, `4.55 模擬年/日`, 実効 MPS `6.25 GB`; A100 推定 ~0.9 時間/年） |
| **9. ハイトップ 3D `16×64` `RCE_large` 比較構成＋渦許容海洋** | `earth_g5_3d_rcelarge_g7ocn.toml` | `5` (110 km) / `56` (65 km) / `64×16` 3D (256×64 km, `4×16` 正方タイル) | `7` (28 km, 渦許容) / `36` (`ocean_stride=3`) | `rrtmgp`＋3D TICA / `prognostic`（CRM＋放射ペア蒸留 `32` 教師, `SparseCRMPool`） | `six_class_2m` 6 クラス 2M 雲微物理＋非地形 GWD＋海流相対風＋EVP 海氷＋50% 永続アンカー | Apple M4 Pro (48 GB) / NVIDIA A100 | **~16.1 時間/年**（M4 Pro MPS 実測 `1.33 秒/step`, `1.49 模擬年/日`, 実効 MPS `6.79 GB`; A100 推定 ~2.5 時間/年） |

> **生物物理（Biogeophysics）の常時稼働について**：2 葉 $\mathrm{C}_3/\mathrm{C}_4$ P-Model 光合成・熱順化 $T_{\mathrm{home}}$・Lambert-$W_0$ 予報 LAI・常緑/落葉葉習性 $f_{\mathrm{evg}}$・樹高 $h_c$・根圏容量 $S_R$・離散樹冠クランピング $\Omega(\mu_0, L)$・非反復最適葉傾角 $\chi_L$ は陸面コア（`surface.py` + `biosphere.py`）に統合されているため、陸域炭素・窒素・海洋 BGC・エアロゾル・ダスト・雷をすべてオフにしても陸面の植生・アルベド・蒸散・粗度フィードバックは完全に動作する。

---

## 2. 構成 1：Mac / ノートPC お試し構成（Glevel-5 軽量物理コア）

Apple Silicon Mac（MPS, ユニファイドメモリ 16〜32 GB）や手元のデスクトップ GPU で、全球の雲・降水帯・季節変化・自律植生（LAI・樹高・葉角）・海氷を手軽に試すための構成。

- **設計の狙い**：`glevel = 5`（約 110 km）で中緯度傾圧渦を解像しつつ、鉛直層を `32`、埋め込み CRM を `4×1`、放射を `four_band`（または `ckd`）、海洋を軽量な `ekman_sw`（3D 海洋を見たい場合は `ogcm_3d`・`ocean_nz = 18`）にすることで、ノート PC でも 1 か月〜1 年の積分を現実的な時間で確認できる。

```toml
[grid]
glevel = 5
nz = 32
z_top = 42000.0
crm_nx = 4
crm_ny = 1
crm_dx = 4000.0
dt_host = 720.0
n_host_dyn_substeps = 1
n_crm_dyn_substeps = 10
rad_stride = 5
crm_col_chunk = 12288
rad_col_chunk = 16384

[forcing]
rad_scheme = "ckd"                # さらに軽量にする場合は "four_band"
o3_mode = "latitude_seasonal"
enable_superparam = true
microphysics_scheme = "two_moment"
enable_nonoro_gwd = true
use_earth2014 = true
earth2014_stride = 2
ocean_mode = "ekman_sw"           # 3D海洋も試す場合は "ogcm_3d", ocean_nz = 18
h_ml_ocean = 50.0
enable_river_routing = true
# 物質循環オフ（P-model・LAI・常緑/落葉・樹高・クランピング・葉角は自動でオン）
enable_dust = false
enable_natural_aerosols = false
enable_land_carbon = false
enable_land_nitrogen = false
enable_ocean_bgc = false
enable_lightning = false
enable_dynamic_lakes = false

[output]
years = 1
monthly_mean_include_subgrid_2d = false
snapshots_include_subgrid_2d = false
```

---

## 3. 構成 2：大循環・ジェット緯度の自発的創発（理想化アクアプラネット）

陸海分布・山岳定常波・緯度固定オゾン分布をすべて排除した完全対称な球面海洋上で、**ハドレー循環幅・亜熱帯ジェット・渦駆動の寒帯前線ジェット（および自転速度 $\Omega$ 変化に伴う多重ジェットや $\mathrm{CO}_2$ 増加に伴う極側シフト）が第一原理のみから自発的に創発すること**を検証する構成。

- **重要な物理条件**：
  1. **傾圧渦の解像**：中緯度ロスビー変形半径 $L_d \approx 800\text{–}1000\ \mathrm{km}$ を実効解像度（$\sim 6\Delta x$）で解像するため `glevel = 5`（最低ライン）〜 `6`（理想、約 55 km）を用いる。
  2. **緯度依存の外力を完全除去**：`o3_mode = "prognostic"`（3次元光化学＋Brewer–Dobson 輸送）または一様基準プロファイル `"rcemip"` を選び、現代地球の緯度に固定されたオゾン気候値を避ける。AOBA の非地形 GWD（`enable_nonoro_gwd = true`）は前線と対流から自律励起するマップフリー実装のためオンにして問題ない。
  3. **高速な熱平衡**：`ocean_mode = "ekman_sw"`, `h_ml_ocean = 30.0` と薄めの混合層にすることで、1〜2 年で熱的準平衡に達し、計 5 年積分で明瞭な東西平均ジェット構造と渦運動量フラックス収束が得られる。

```toml
[grid]
glevel = 6                        # 手軽に試す場合は 5 (dt_host = 720, n_host_dyn_substeps = 1)
nz = 44
z_top = 48000.0
crm_nx = 8
crm_ny = 1
dt_host = 600.0
n_host_dyn_substeps = 2
rad_stride = 6

[forcing]
use_earth2014 = false             # 全面平坦海洋（アクアプラネット）
insolation_mode = "annual_mean"   # 季節移動を見る場合は "seasonal"
rad_scheme = "ckd"                # 厳密比較時は "rrtmgp"
o3_mode = "prognostic"            # 純粋理論比較時は "rcemip" も可
o3_lw_coupling = true
enable_superparam = true
enable_nonoro_gwd = true
ocean_mode = "ekman_sw"
h_ml_ocean = 30.0
# 感度実験用：自転角速度 omega (例: 0.5x, 1x, 2x, 4x) や co2_ppmv (284 -> 1136) を変更
omega = 7.292115e-05
co2_ppmv = 284.3
enable_dust = false
enable_natural_aerosols = false
enable_land_carbon = false
enable_land_nitrogen = false
enable_ocean_bgc = false
enable_lightning = false

[output]
years = 5
monthly_mean_include_subgrid_2d = false
snapshots_include_subgrid_2d = false
```

---

## 4. 構成 3：大循環・ジェット緯度＆海流前線結合（現実地球・高解像度物理コア）

現実地球の地形・海陸分布のもとで、**渦許容（eddy-permitting）海洋の黒潮・メキシコ湾流・南極周極流 (ACC) のシャープな SST 前線**、**最高精度の公式 HITRAN / MT\_CKD 4.1 放射 (`rrtmgp`)**、**3 次元予報オゾン**、および**マップフリー陸面生物物理**を結合し、ジェット緯度・ストームトラック・モンスーンの自発的創発を最高精度で調べる構成（HighResMIP 級）。

- **構成のポイント**：
  - 大気 `glevel = 6`（55 km, 44 層, `8×1` CRM）に対し、海洋を独立細格子 `ocean_glevel = 7`（約 28 km, `ocean_nz = 36` 層, 表層 15 m〜5500 m）で保存結合する（`earth2014_stride = 1` でフル 5 分解像度の海底地形・海峡を反映）。
  - 雷（不確実性が大きいためオフ）や炭素・窒素・海洋 BGC・エアロゾル・ダストはオフにし、物理力学・放射・予報オゾン・陸面生物物理に計算資源を集中する。なお `o3_mode = "prognostic"` は Chapman 光化学・$\mathrm{NO}_x/\mathrm{HO}_x/\mathrm{PSC}$ 破壊・Brewer–Dobson 3D 移流輸送を単独で閉じるため、他物質循環オフでも現実的な成層圏・対流圏界面オゾン分布が創発する。
  - **段階的運用**：1 年積分（`years = 1`, `restart_monthly = true`）ずつ投入し、毎年の放射収支・ジェット緯度・SST 前線・海氷を確認しながら調整・継続する運用に適する。
  - **軽量バリアント**：海洋解像度だけ `ocean_glevel = 6`（約 55 km）, `ocean_nz = 32` に 1 段落とすと、海洋計算コストが約 $1/6$ に下がり全体が約 1.5〜1.7 倍高速化する。

```toml
[grid]
glevel = 6
nz = 56
z_top = 65000.0
z_sponge_base = 52000.0
crm_nx = 8
crm_ny = 1
crm_dx = 4000.0
dt_host = 600.0
n_host_dyn_substeps = 2
n_crm_dyn_substeps = 10
rad_stride = 6
crm_col_chunk = 49152
rad_col_chunk = 32768

[forcing]
rad_scheme = "rrtmgp"             # 公式 HITRAN / MT_CKD 4.1 240 g-pt (128 LW + 112 SW)
o3_mode = "prognostic"            # 自発的 3D オゾン光化学＋Brewer-Dobson 輸送
o3_lw_coupling = true
enable_nonoro_gwd = true
use_earth2014 = true
earth2014_stride = 1              # Glevel-7 (28 km) 海岸線・海峡用のフル 5 分解像度
enable_superparam = true
microphysics_scheme = "two_moment"
re_liq_um = 10.0
re_ice_um = 48.0
cloud_sw_inhomog = 0.62

ocean_mode = "ogcm_3d"
ocean_glevel = 7                  # 軽量化する場合は 6 (55 km), ocean_nz = 32
ocean_nz = 36                     # 表層 15 m から 5500 m へ幾何ストレッチ
h_ml_ocean = 15.0
enable_seaice_drift = true
enable_seaice_evp_itd = true
enable_ogcm_fct_zstar = true
enable_ogcm_scale_aware_gm = true
enable_ogcm_submeso = true
enable_ogcm_tidal_mixing = true
enable_current_rel_wind = true
enable_river_routing = true

# 雷・BGC・エアロゾル・ダストはオフ（P-model・LAI・常緑/落葉・樹高・クランピング・葉角はオン）
enable_lightning = false
enable_dust = false
enable_natural_aerosols = false
enable_aerosol_rad_forcing = false
enable_snow_bc_darkening = false
enable_land_carbon = false
enable_land_nitrogen = false
enable_fire_veg_feedback = false
enable_ocean_bgc = false
enable_bgc_fe_lnox_coupling = false
enable_bvoc_soa = false
enable_prognostic_ch4_n2o = false
enable_silicate_weathering = false
enable_dynamic_lakes = false

[output]
years = 1
restart_interval_years = 1
restart_monthly = true
monthly_mean_include_subgrid_2d = false
snapshots_include_subgrid_2d = false
```

---

## 5. 構成 4：長期結合スピンアップ標準構成（Phase-1 PI-Control）

30〜100 年規模の結合気候スピンアップ、気候ドリフト検証、$4\times\mathrm{CO}_2$ 平衡気候感度 (ECS) 実験の主力構成。A100 1 GPU で約 `21.8 秒/モデル日`（約 `2.2 時間/年`）で進行し、30 年積分を 48 時間ジョブ 1.5 回分（~66 GPU-h）で完走できる。

- `glevel = 5`, `nz = 44`, `crm_nx = 8, crm_ny = 1`, `dt_host = 720.0`, `rad_stride = 5`
- `rad_scheme = "ckd"`, `o3_mode = "latitude_seasonal"`（または `"prognostic"`）
- `ocean_mode = "ogcm_3d"`, `ocean_glevel = 0`（大気と同 110 km）, `ocean_nz = 18`, `h_ml_ocean = 50.0`, EVP-ITD 海氷・FCT-$z^*$・GM90/サブメソ・潮汐混合・河川ルーティングすべてオン
- ダスト（`enable_dust = true`）＋ 4 種自然エアロゾル（`enable_natural_aerosols = true`）オン、炭素・窒素・海洋 BGC はスピンアップ速度優先で Phase-1 ではオフ（Phase-2 でオン化）。

---

## 6. 構成 5：フル ESM（全地球システム完全結合）構成

大気非静力学・埋め込み 2D CRM・最高精度放射・6 種 2 モーメント雲微物理・3D オゾン・渦許容海洋・EVP-ITD 海氷・波浪・海洋 BGC/OMZ・陸域 C/N/永久凍土/山火事・MAML サブグリッド地形/湿地・マップフリー動的湖沼・全エアロゾル・水同位体・温室効果ガス循環・ケイ酸塩風化・動的氷床まで、AOBA の全コンポーネントを結合する構成。

```toml
[grid]
glevel = 5                        # 高解像度版は glevel = 6, dt_host = 600, n_host_dyn_substeps = 2
nz = 44
crm_nx = 8
crm_ny = 1
dt_host = 720.0
rad_stride = 5

[forcing]
rad_scheme = "rrtmgp"             # 3D 雲側方散乱も有効にする場合は enable_rad_3d_tica = true
enable_rad_3d_tica = true
microphysics_scheme = "six_class_2m"
o3_mode = "prognostic"
o3_lw_coupling = true
co2_mode = "prognostic"           # 濃度規定なら "constant" または "scenario"
enable_co2_3d_transport = true
enable_prognostic_ch4_n2o = true
enable_silicate_weathering = true

ocean_mode = "ogcm_3d"
ocean_glevel = 6                  # 28 km 海洋にする場合は 7
ocean_nz = 32
h_ml_ocean = 20.0
enable_seaice_drift = true
enable_seaice_evp_itd = true
enable_ocean_bgc = true
enable_ocean_o2_omz = true
enable_wave_coupling = true
enable_current_rel_wind = true

enable_river_routing = true
enable_dynamic_lakes = true       # マップフリー動的湖沼（岩盤＋Priority-Flood＋2層FLake）
lake_init_fill_frac = 0.5
enable_maml_subgrid_topo = true
enable_maml_wetlands_lakes = true
enable_land_carbon = true
enable_land_nitrogen = true
enable_fire_veg_feedback = true
enable_lightning = true
enable_dust = true
enable_natural_aerosols = true
enable_aerosol_rad_forcing = true
enable_snow_bc_darkening = true
enable_bgc_fe_lnox_coupling = true
enable_bvoc_soa = true
enable_water_isotopes = true

[icesheet]
enable_dynamic_icesheet = true
icesheet_ema_years = 5.0
icesheet_accel_factor = 100.0
icesheet_accel_spinup_years = 20.0
```

---

## 7. 構成 6：古気候・氷期・系外惑星構成（マップフリー設計の応用）

- **完新世中期「緑のサハラ」／氷期–間氷期実験**：
  - `enable_milankovitch_orbit = true`, `milankovitch_start_kyr = -6.0`（6 ka）または `-21.0`（LGM）
  - `enable_dynamic_lakes = true`（`lake_init_fill_frac = 0.0` でゼロからメガチャド湖や氷河湖アガシを自律形成）
  - `enable_dynamic_icesheet = true`, `icesheet_accel_factor = 100.0`（氷床の成長・後退と地殻アイソスタシーによる盆地・河川網の再編を追跡）
- **潮汐固定系外惑星（例：Proxima Centauri b / TRAPPIST-1e）**：
  - `planet_preset = "proxima_b"`（または `"trappist_1e"`, `"mars_early"`）, `enable_tidal_lock = true`, `use_earth2014 = false`, `rad_scheme = "ckd"` or `"rrtmgp"`, `o3_mode = "prognostic"` or `"rcemip"`。

---

## 8. GPU 資源運用とメモリ調整の指針

1. **1 実行 ＝ 1 GPU 設計**：AOBA は単一 GPU 上でメモリ帯域を飽和させる融合カーネル設計であるため、1 本のジョブに 2 GPU 以上を割り当てても計算は高速化しない。複数 GPU を活用する場合は、異なる設定・アンサンブル・感度実験（例：$1\times\mathrm{CO}_2$ と $4\times\mathrm{CO}_2$、あるいは `G5` 長期ランと `G6+G7` 高解像度ラン）を**独立した 1 GPU ジョブとして同時並列投入**するのが最も高効率である。
2. **VRAM 容量に応じたチャンク調整**：高解像度（`glevel = 6` 以上）で VRAM に余裕がある GPU（A100 40/80 GB）では `crm_col_chunk = 49152` として非分割一括処理すると最速になる。メモリ容量が小さい GPU では `crm_col_chunk = 8192` / `rad_col_chunk = 16384` に下げると、精度を一切変えずにピーク VRAM を大幅に削減できる。
3. **Zarr v3 メッシュキャッシュとユニバーサル異格子リスタート**：`build_scvt_mesh` は `glevel >= 4` の SCVT 格子と Earth2014 境界データを `~/aoba_outputs/_mesh_cache/`（`AOBA_OUTPUT_DIR/_mesh_cache/`）に CF/UGRID 準拠の Zarr v3 ストアとして自動キャッシュするため、2 回目以降の G6+G7 起動は **1 秒台** で完了する。また `scripts/run_g5_30yr.py` のリスタート読込は `(glevel, nz, z_top, crm_nx/ny, ocean_glevel, ocean_nz)` の任意変更に対応しており、海陸・湿潤セル対応の球面 `cKDTree` 補間と静水圧成層圏延長（例：`z_top = 48 km / 44 層` から `65 km / 56 層` への継続や、G5 スピンアップから G6/G7 への引き継ぎ）を自動で実行する。
4. **オンライン自己蒸留 2D/3D CRM・教師専用 `SparseCRMPool` ＆ ペア ICA 放射加速（`enable_crm_self_distill`, `enable_rad_self_distill`）**：`[forcing]` で `enable_crm_self_distill = true` を有効化すると、空間層化ローテーション班と 2 ヘッドアンサンブル不確実性上位列の計 `1 / crm_distill_stride`（既定 `1/13 ≒ 7.7%`）のみ本物 CRM を連続実行し、残りの列を厳密保存則レイヤー（水質量・湿潤エンタルピー・水平運動量を代数的に厳密保存）・2D/3D 地表面不均一性クラスタ記述子（Moran's $I$、パッチ代表長 $L_{\mathrm{patch}}$、$\rho(\mathrm{SHF}, \mathrm{LHF})$）・FiLM 変調付き周期 2D 空間配置ヘッド（`SubgridSpatialHead`）・平均ゼロ接地層微気象フィードバック・および教師あり対流コールドプール／下降気流–地表面結合（平均ゼロの接地層浮力偏差 $\Delta\theta_{m,\mathrm{cp}}$・水蒸気偏差 $\Delta q_{v,\mathrm{cp}}$・水平運動量と渦運動エネルギーを保存する放射状密度流ガスト $u_{\mathrm{gust}}$）を備えた純粋密行列積学生ネットワーク（`CRMSelfDistillNet`）で推論しつつ、GPU 常駐デュアルゾーンリプレイバッファから毎ステップ自己蒸留する。さらに `enable_rad_self_distill = true` を併用すると、教師班と同期したサブカラム単位の ICA 放射自己蒸留（`RadSelfDistillNet`）が有効化され、TOA–大気–地表面の厳密放射エネルギー保存・夜間短波厳密ゼロ・各陸面タイルの真のアルベド／射出率応答を保持しながら、陸面・大気力学・3D 海洋を 100% 第一原理のまま高速化する。任意の 3D CRM 格子で地表タイルを疎結合した場合（`crm_ny > 1`, `crm_nx > 1`, `0 < crm_sfc_tiles < crm_ny * crm_nx`）、AOBA は地表格子を等方的な 2D タイル格子へ自動因数分解（`resolve_crm_sfc_tile_grid`、例：`16×64 → 4×16` の `16×16 km` 正方タイルや `2×8` の `32×32 km` 正方タイル）したうえで、スロット保持型の教師専用 `SparseCRMPool` を自動起動し、フル 3D 力学および 3D-TICA 放射配列を稼働教師 `ceil(N_cells / stride)` 列のみに割り当てる（G5 `16×64` で約 `37 GB → 0.15 GB` へ削減しつつ、教師–学生交代時も陸面タイル数を完全に固定維持し、50% 永続アンカー班の複数日 3D 組織化記憶を途切れなく保持）。さらに AOBA の 82 本の単一ソース融合 Metal/CUDA カーネル（大気 SCVT RK3-HEVI・3D CRM 音波ループ・3D OGCM TRiSK/FCT+$z^*$+Redi/KPP/GM・圧縮陸面/海面 Newton–Raphson ソルバー・ML 特徴抽出/厳密保存則クロージャ/タイル更新カーネル）および厳密フラックス保存の非同期海洋結合（`ocean_stride = 3`）を組み合わせることで、ハイトップ（`nz = 56, 65 km`）＋予報オゾン＋RRTMGP＋3D-TICA＋6 クラス 2M 雲微物理＋G7 36 層渦許容海洋構成を、Apple M4 Pro MPS 上において完全 1:1 結合の 3D `8×8` 基本構成（`earth_g5_3d_8x8_g7ocn.toml`、`crm_sfc_tiles = 0`、655,488 枚の 4 km MAML+ICA タイル）で実効 MPS `6.25 GB`・**`0.43 秒/step`（`5.3 時間/年`、`4.55 模擬年/日`）**、比較用の 3D `16×64` `RCE_large` 構成（`earth_g5_3d_rcelarge_g7ocn.toml`、`crm_sfc_tiles = 64`、基本構成と同数の 655,488 枚 `4×16` タイル）で実効 MPS `6.79 GB`・**`1.33 秒/step`（`16.1 時間/年`、`1.49 模擬年/日`）** で実行できる。
5. **出力容量の最適化**：高解像度実験で Zarr 出力を抑えたい場合は `monthly_mean_include_subgrid_2d = false`, `snapshots_include_subgrid_2d = false` とし、日次診断・月次 2D/3D 平均・月次リスタート（`restart_monthly = true`）を有効にして 1 年単位で安全にリレー実行する。
