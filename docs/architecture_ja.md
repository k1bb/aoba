# AOBA モデルアーキテクチャおよび根拠文献仕様書

> **言語 (Language)**: [English (architecture.md)](architecture.md) | 日本語

## 1. 科学的目標

AOBA は以下の 2 つの地球システム科学上の基本問題を検証します。
1. 外部境界条件として**最小限の物理情報のみ** — `Earth2014` による海陸・氷床分布および標高・海底地形（Hirt & Rexer, 2015）、自転角速度 $\Omega$、軌道太陽放射 $S_0$、大気温室効果ガス濃度 — を与えたとき、大気・海洋・雪氷圏・生態系・大気化学を結合した地球システムモデルが地球の気候帯・海洋循環・雲・エアロゾル・バイオームを自律的に再現できるか（Beck et al., 2018, 2023 との比較検証）。
2. 積雲パラメタリゼーション、処方海洋 $Q$ フラックス、土壌・アルベド地図、植物機能型（PFT）表、および現在気候で事前学習した固定 ML 重みを排した最小制約 ESM が、急激な $4\times\mathrm{CO}_2$ 強制・古気候軌道変動・異なる惑星パラメータに対してどのように応答するか。

---

## 2. 統一圧縮性非静力学力学・乱流・重力波ドラッグ・空力基準面

### 2.1 全球ホスト力学コア：球面重心ボロノイ C-grid（TRiSK）＋ Charney–Phillips HEVI（`src/aoba/scvt_dynamics.py`）
- **主要文献**: Ringler, Ju, & Gunzburger (2008); Thuburn, Ringler, Skamarock, & Klemp (2009); Ringler, Thuburn, Klemp, & Skamarock (2010); Klemp (2011); Staniforth & Thuburn (2012); Skamarock et al. (2012); Gassmann (2013); Zalesak (1979); Held & Suarez (1994); Nishizawa, Tomita, & Team SCALE (2025, *SCALE-RM v5.5.5*, Ch. 2).
- **支配方程式**: 乾燥空気密度 $\rho_d$、エッジ法線水平運動量、鉛直運動量 $\rho_d w$、湿潤温位 $\rho_d \theta_m$（水蒸気浮力と圧縮性非断熱エンタルピー項を整合的に含む；Seidel & Yang, 2024; Nishizawa et al., 2025）、および正値保存 Zalesak FCT 移流による水分・トレーサ類を保存する 3 次元フラックス形式完全圧縮性非静力学オイラー方程式系。
- **SCVT 直交 C-grid と鉛直座標**: 正二十面体再帰分割と球面 Lloyd 最適化により `Glevel-4`（2,562 セル、約 448 km）、`Glevel-5`（10,242 セル、約 224 km）、`Glevel-6`（40,962 セル、約 112 km）、`Glevel-7`（163,842 セル、約 56 km）に対応し、解像度自動スケーリング付き Smagorinsky 粘性および重調和 $\nabla^4$ 超粘性を備えます。Klemp (2011) 平滑化地形追従座標上の Charney–Phillips 鉛直配置は `nz = 32`（$z_{\mathrm{top}} = 42\ \mathrm{km}$）から高トップ成層圏解像 `nz = 64`（$z_{\mathrm{top}} = 65\ \mathrm{km}$, $\Delta z_1 = 60\ \mathrm{m}$）まで柔軟に拡張可能です。
- **地形性およびマップ不要の非地形性重力波ドラッグ（`src/aoba/orography.py`, `src/aoba/gravity_wave.py`）**: 5 分格子 `Earth2014` 起伏から直接算出する低層山岳ブロッキング・地形性 GWD（Lott & Miller, 1997）と、前線および対流加熱から励起する流れ依存の非地形性 GWD（Charron & Manzini, 2002; Beres, 2004; Richter et al., 2010）を統合します。

### 2.2 埋め込み 2D/3D CRM・MYNN Level 2.5 PBL・3D SGS 乱流（`src/aoba/crm.py`, `src/aoba/turbulence.py`）
- **主要文献**: Khairoutdinov & Randall (2003); Khairoutdinov, Randall, & DeMott (2005); Lee & Khairoutdinov (2015); Lin et al. (2023); Nakanishi & Niino (2009); Yamazaki & Miura (2024); Nishizawa et al. (2025, Sec. 8.1.2 & 8.2.1).
- **親子完全統一非静力学コア**: 各ホストボロノイセルに周期圧縮性 C-grid / Charney–Phillips RK3-HEVI CRM（`32 × 1` $\Delta x = 2\ \mathrm{km}$ の 2D カーテンから 3D `8 × 8`・`64 × 16` `RCE_large` まで可変）を埋め込み、全球ホストと同一の熱力学・放射・微物理・乱流ソルバを共有し、陰解法 **MYNN Level 2.5** 鉛直境界層混合および非等方 **Smagorinsky–Lilly** 3D SGS 拡散を適用します。

### 2.3 変位追従空力基準面・RSL 相似則・陰的地表面結合（`src/aoba/surface.py`）
- **主要文献**: Hirt & Rexer (2015); Kempes et al. (2011); Raupach (1994); Harman & Finnigan (2007); De Ridder (2010); Bonan et al. (2018); Lee et al. (2020); Abolafia-Rosenzweig et al. (2021); Meier et al. (2022); Miller, Beljaars, & Palmer (1992); Kalnay & Kanamitsu (1988).
- **樹冠基準面と輸送係数安定化**: 積雪埋没補正後の予報樹冠高 $h_c$ からゼロ面変位 $d$、粗度長 $z_{0m}$、および解析的粗度サブレイヤ（RSL）安定度補正 $\hat{\Psi}_{M,H}$ を $\tilde{z}_{\mathrm{ref}} = \max(z_{c,0} - z_{w,0} - 0.65 h_c,\ 5.0\ \mathrm{m})$ 上で連続算定します。さらに $C^\infty$ 滑らかな 2 次 Kalnay–Kanamitsu 陰的地表面–大気バルク輸送係数リミッタ（$\alpha_{\max} = 1.50$）により、薄い最下層（$\Delta z_1 = 60\ \mathrm{m}$）と高樹冠の組み合わせで生じる $2\Delta t$ 顕熱振動を抑止します。

---

## 3. オンライン代理スーパーパラメタリゼーション（`OSSP`：事前学習不要の ML 加速；`src/aoba/crm_self_distill.py`）

- **主要文献**: Beucler et al. (2021, *Phys. Rev. Lett.*, 126, 098302); Patton, Sullivan, & Moeng (2005); Mapes & Neale (2011); Colin et al. (2019); Yu et al. (2023/2025, ClimSim); Kochkov et al. (2024, *Nature*, 632, 1060–1066).
- **オンライン教師–生徒蒸留**: `enable_crm_self_distill = true` および `enable_rad_self_distill = true` により、空間層化＋認識的不確実性誘導で選んだ教師コラム群（`1 / crm_distill_stride`、連続滞在ウィンドウと 50% 永続アンカー群により対流記憶を保持）でのみ本物の非静力学 CRM および ICA/3D-TICA 放射を実行し、残りの生徒コラムは 2-head 認識的残差 MLP（`CRMSelfDistillNet`, `RadSelfDistillNet`）で高速推論しつつ GPU 常駐の二層リプレイバッファ（短期リング＋長期リザーバ）から連続的にオンライン学習します。
- **マップ不要特徴量・組織化不変量・厳密保存クロージャ**: 入力は局所保存熱力学プロファイル $(T_{li}, \theta_l, q_n, q_n/q_{\mathrm{sat}})$、3 変数の予報対流記憶状態 $(M_{P,\mathrm{fast}}, M_{P,\mathrm{slow}}, M_{B,\mathrm{slow}})$、および 16 種の周期 2D/3D 地表面不均一性不変量（Moran の $I$、パッチ長スケール $L_{\mathrm{patch}}$、$\rho(\mathrm{SHF}, \mathrm{LHF})$）のみで構成されます。融合 GPU カーネル（`crm_distill_closure_col`, `rad_distill_closure_col`）がコラム水質量・顕熱/潜熱エンタルピー・水平運動量・夜側短波ゼロ・サブコラム TOA–大気–地表放射収支を機械精度で厳密保存し、周期 `SubgridSpatialHead` がゼロ平均コールドプール微気候とガスト流出を MAML 地表面タイルへ再構成します。
- **教師専用 `SparseCRMPool` と逐次シャドウ乖離診断**: 表面タイルを独立解像する大規模 3D CRM（`crm_sfc_tiles > 0`）では、`SparseCRMPool` が活性教師スロットにのみ 3D CRM 状態メモリを確保します（G5 `64 × 16` で `~37 GB` → `~0.15 GB`）。`enable_distill_shadow_diag = true` により毎回の勾配更新直前に教師コラム上で両代理モデルをシャドウ評価し、気候レジーム遷移におけるホスト大気・陸面タイル双方の無バイアス out-of-sample 誤差と降水空間相関を常時出力します。

---

## 4. 選択式放射伝達・雲微物理・非誘導帯電発雷

### 4.1 放射伝達（`rad_scheme` ＋ 3D-TICA；`src/aoba/radiation*.py`）
- **主要文献**: Toon et al. (1989); Lacis & Hansen (1974); Slingo (1989); Ebert & Curry (1992); Hu & Stamnes (1993); Cahalan et al. (1994); Chou et al. (1999); Várnai & Davies (1999); Sekiguchi & Nakajima (2008); Pincus et al. (2019); Mlawer et al. (2023).
- **統一選択スキーム**: 単一の 2 ストリーム三重対角インタフェース上で 4 スキームを切替可能です：① `"four_band"`（高速解析 4 バンド）、② `"ckd"`（16 バンド 50 g点 Correlated-$k$）、③ `"ckd_hd"`（111 g点高精細 CKD ＋ Chou 長波雲散乱 ＋ 5 種エアロゾル）、④ `"rrtmgp"`（公式 240 g点 HITRAN / MT\_CKD 4.1：LW 128 + SW 112）。`enable_rad_3d_tica = true` により雲側面入射・遮蔽を表現する 3D 傾斜独立コラム近似（3D-TICA）を有効化できます。

### 4.2 雲微物理と選択制発雷（`microphysics_scheme`, `lightning_scheme`, `src/aoba/microphysics*.py`, `src/aoba/lightning*.py`）
- **主要文献**: Khairoutdinov & Randall (2003); Khairoutdinov & Kogan (2000); Morrison et al. (2005); Seifert & Beheng (2006); Morrison & Gettelman (2008); DeMott et al. (2010); Boutle et al. (2014); McCaul et al. (2009); Price & Rind (1993); Schumann & Huntrieser (2007); Takahashi (1978); Chiu (1978); Dwyer (2003); Fierro et al. (2013); Sato et al. (2019).
- **選択スキーム**: `"one_moment"`（2 変数診断混相）、`"two_moment"`（雲水・雲氷数濃度 $n_c, n_i$ 予報、海塩/硫酸塩/ダスト CCN・INP 活性化、解像度依存 KK00 LWC 分散スケーリング）、および `"six_class_2m"`（水蒸気・雲水・雲氷・雨・雪・霰の完全 6 クラス 2 モーメント）。発雷（`lightning_scheme`）は高速な `"diagnostic"` 非誘導帯電診断（`src/aoba/lightning.py`）と、任意の SCALE-LT 相当 `"explicit_bulk"` 5 種水物質電荷予報＋FFT・Thomas 厳密ポアソン電場＋絶縁破壊 IC/CG 放電スキーム（`src/aoba/lightning_explicit.py`）を選択でき、質量保存型 $\mathrm{LNO}_x$ 生成をオゾン化学および陸域・海洋窒素循環へ結合します。

---

## 5. 自律 EEO 生態系・MAML 陸面・動的湖沼・陸域 C/N 凍土・火災・河川

### 5.1 Two-Leaf $\mathrm{C}_3/\mathrm{C}_4$ 最適性生態系（`src/aoba/biosphere.py`）
- **主要文献**: Farquhar, von Caemmerer, & Berry (1980); de Pury & Farquhar (1997); Bernacchi et al. (2001, 2003); Dai, Dickinson, & Wang (2004); Medlyn et al. (2011); Prentice et al. (2014); Heskel et al. (2016); Smith et al. (2019); Kumarathunge et al. (2019); Stocker et al. (2020); Mengoli et al. (2022); Scott & Smith (2022); Kikuzawa (1991); Givnish (2002); Sakai & Larcher (1987); Wang et al. (2023); Luo et al. (2024); Ren et al. (2024, 2025); Schneider, Gessler, & Stocker (2025); Cai et al. (2025); Zhou et al. (2025); Gan et al. (2026); Lavergne et al. (2026).
- **定式化**: 地方正午短波放射 EMA による樹冠 2 ストリーム光順化、3 本柱熱順化（$T_{\mathrm{growth}}, T_{\mathrm{home}}$）、陽葉・陰葉閉形式配分、離散樹冠クランピングおよび最適葉傾斜角 $\chi_L$、時間積分最適土壌水分応答 $\tau(D_{\max})$（Lu et al., 2016; Holtzman et al., 2024）、最適根圏貯水容量 $S_R$（Gao et al., 2014; Davis et al., 2017）、冬季凍害を含む自律常緑/落葉最適性 $f_{\mathrm{evg}}$、Lambert-$W_0$ 予報 LAI、および管束水力樹冠高 $h_c$ を統合します。

### 5.2 MAML サブグリッド地形・湿地・動的湖沼・土壌積雪・C/N 凍土・火災・河川
- **主要文献**: Clapp & Hornberger (1978); Beven & Kirkby (1979); Yang et al. (1997); Sakaguchi & Zeng (2009); Flanner et al. (2007); Mironov (2008); Thonicke et al. (2010); Swenson & Lawrence (2012, 2014); Koven et al. (2013); Oleson et al. (2013); Barnes et al. (2014); Fiddes & Gruber (2014); Nitta et al. (2014); Mizukami et al. (2016); MATSIRO6 (2021); Dong et al. (2022); Stocker & Prentice (2024); Regnier et al. (2013).
- **統合陸面システム**:
  - **MAML サブグリッド地形・湿地（`src/aoba/maml_land_subgrid.py`）**: 標高帯微気候ダウンスケーリング（地表気圧と最下層大気気圧の整合的測高スケーリング）および TOPMODEL 湿地被覆率。
  - **マップ不要の動的湖沼（`src/aoba/lakes.py`）**: `Earth2014` Priority-Flood 窪地体積に基づく湖沼水収支と 2 層 FLake 熱力学・結氷/融氷。
  - **土壌・積雪（`src/aoba/surface.py`）**: 5 層凍結・融解陰的土壌熱伝導および Richards 浸透、土壌水分依存裸地アルベド、BC/ダスト SNICAR 暗色化付き積雪エージング、氷河上限流出。
  - **陸域 C/N・永久凍土・MAML SPITFIRE 火災（`src/aoba/land_carbon.py`）**: Neumaier–Kahan 補償加算付きの葉・木部・根・リター＋ 5 層永久凍土 C/N プール、生物窒素固定・溶脱・$\phi_N$ 制限フィードバック、および 2D 隣接延焼と燃焼 BC/OC/C 排出を含む離散パッチ SPITFIRE 森林火災。
  - **Priority-Flood 河川・溶存物質ルーティング（`src/aoba/river_routing.py`）**: マップ不要のボロノイ流路網により淡水・アルカリ度・栄養塩・溶存鉄を沿岸海洋セルへ保存輸送します。

---

## 6. 選択式海洋（`ocean_mode`）・EVP-ITD 海氷・波浪・FlexPFT 海洋 BGC・オフライン深海加速

- **主要文献**: Semtner (1976); Briegleb et al. (1986); Redi (1982); Gent & McWilliams (1990); Large et al. (1994); Hunke & Dukowicz (1997); Griffies & Hallberg (2000); Adcroft & Campin (2004); Shchepetkin & McWilliams (2005); Fox-Kemper et al. (2008); Codron (2012); Hallberg (2013); Roquet et al. (2015); Orr & Epitalon (2015); Danilov et al. (2017); Orr et al. (2017); Turner et al. (2022); Sasai et al. (2022); Takaya et al. (2010); Andreas et al. (2015).
- **3D $z^*$ FCT SCVT ボロノイ OGCM（`src/aoba/ocean_ogcm.py`, `src/aoba/ocean_coupler.py`）**: 独立細格子海洋（`ocean_glevel = 6–7`, `ocean_nz = 18–64` 表層 2 m〜6,000 m）、厳密 $z^*$ 鉛直座標、Zalesak FCT トレーサ移流、Roquet (2015) EOS、Hallberg スケール適応 GM90 ＋ 回転 Redi テンソル、Fox–Kemper サブメソスケール再成層、KPP・Ri シア・内部潮汐深海混合、海流相対風応力、解像度依存 C-grid Hollingsworth–Bernoulli 発散減衰（$a_{\mathrm{div}} = 0.030\,\Delta c_{\min}^2/\Delta t$）、および水＋海氷総エンタルピー厳密保存結合器を備えます。理想実験向けの高速 1.5 層エクマン浅水モード（`ocean_mode = "ekman_sw"`, `src/aoba/ocean_ekman.py`）も選択可能です。
- **EVP-ITD 海氷・表面波浪・FlexPFT 海洋 BGC（`src/aoba/seaice_evp.py`, `src/aoba/ocean_waves.py`, `src/aoba/ocean_bgc.py`）**: 5 カテゴリーボロノイ EVP-ITD 海氷、波浪粗度・Stokes ドリフト・Langmuir 乱流結合、および OMIP/mocsy 2.0 炭酸化学・$\mathrm{O}_2$/OMZ 脱窒・大気ダスト鉄/$\mathrm{LNO}_x$/河川溶存物質と結合した FlexPFT 海洋 BGC を統合します。
- **Phase-2 オフライン深海スピンアップ加速（`scripts/run_offline_ogcm_spinup.py`）**: Phase-1 結合計算の複数年大気フラックスで単独 3D OGCM ＋ FlexPFT BGC ＋ EVP-ITD 海氷を高速循環積分し（深海ドリフト `< 0.01 K/century` で自動停止）、千年規模の深海平衡化を大幅省資源化します。

---

## 7. 大気化学・自然エアロゾル・水同位体・動的氷床・惑星拡張

- **予報 3D オゾン（`src/aoba/ozone_chemistry.py`, `src/aoba/ozone_forcing.py`）**: `o3_mode`（`rcemip`, `latitude_seasonal`, `climatology`, `prognostic` 3D Chapman ＋ $\mathrm{HO}_x/\mathrm{NO}_x/\mathrm{ClO}_x$/PSC 光化学＋Wesely 乾性沈着＋Brewer–Dobson FCT 輸送；Solomon et al., 1996; Brasseur & Solomon, 2005; Burkholder et al., 2019）。
- **自然エアロゾル・ダスト・水同位体（`src/aoba/aerosols.py`, `src/aoba/dust.py`, `src/aoba/water_isotopes.py`）**: マップ不要の 4 種自然エアロゾル（海塩、FlexPFT 結合 DMS/$\mathrm{SO}_4$、火災 BC/OC、BVOC-SOA；Gong, 2003; Jaeglé et al., 2011; Guenther et al., 2012）、塩跳動鉱物ダスト（Marticorena & Bergametti, 1995; Zender et al., 2003）、放射・積雪暗色化結合、および 3D 安定水同位体（$\mathrm{H}_2^{18}\mathrm{O}$, $\mathrm{HDO}$；Craig & Gordon, 1965; Majoube, 1971; Merlivat & Jouzel, 1979）。
- **任意の炭素・微量気体循環、火山/太陽、動的 SIA 氷床、系外惑星（`src/aoba/prognostic_co2.py`, `src/aoba/trace_gas_cycle.py`, `src/aoba/volcanic_solar.py`, `src/aoba/planetary_geochem.py`, `src/aoba/icesheet.py`, `src/aoba/planetary_presets.py`）**: `co2_mode = constant / scenario / csv / prognostic`、予報 $\mathrm{CH}_4/\mathrm{N}_2\mathrm{O}$、火山成層圏 $\mathrm{SO}_2$・太陽周期強制、珪酸塩風化と Laskar (2004) ミランコビッチ軌道進化、Glen 流動則（$n=3$）加速 SIA 氷床＋アイソスタシー＋測高鉛直再補間（Bueler et al., 2005; Pollard & DeConto, 2012）、および岩石系外惑星プリセット（`proxima_b`, `trappist_1e`, `mars_early`）を備えます。

---

## 8. 324 パラメータ一元管理・104 カーネル Metal/CUDA バックエンド・標準診断

- **統合 TOML と `AOBA Studio`（`src/aoba/params.py`, `src/aoba/param_docs.py`, `scripts/aoba_config_gui.py`）**: 全 14 セクション・**324 パラメータ**を [`config/aoba_default.toml`](../config/aoba_default.toml) に一元化し（日英 100% 物理解説）、**11 本の実験プリセット**（`config/presets/`）、任意解像度・層数間リスタート再補間（`AG4 → AG6` 等）、任意の中間圏レッドスプライト・ハロー TLE モジュール（`src/aoba/red_sprites.py`）、30 クラス Köppen–Geiger 分類（`src/aoba/koppen.py`）、および CMIP7 Data Request v1.2 準拠 Zarr v3 出力（`src/aoba/cmip7_diag.py`, `src/aoba/io_zarr.py`）を提供します。
- **104 カーネル単一ソース Metal/CUDA ＋ CUDA Graphs バックエンド（`src/aoba/kernels*.py`, `src/aoba/kernels_cuda.py`, `src/aoba/runtime.py`）**: SCVT RK3-HEVI 力学、2D/3D CRM、FCT 移流、1M/2M/6 クラス微物理、RRTMGP/CKD-HD/CKD/4 バンド放射、3D 予報オゾン、MYNN PBL、圧縮陸面/生態系 Newton–Raphson、MAML/湖沼/C-N/火災/診断型・陽的バルク発雷、ダスト/エアロゾル/水同位体、5 カテゴリー EVP-ITD 海氷、3D $z^*$ OGCM/KPP/GM/Redi/FlexPFT BGC、OGW/NGWD、および OSSP 特徴抽出・厳密保存クロージャ・空間ヘッド・アンサンブル計算を網羅する **全 104 個の融合 GPU カーネル**を Metal Shading Language で単一記述し、NVIDIA GPU 上では決定論的に CUDA C++17 へ自動トランスパイル・キャッシュするとともに、CUDA Graphs 自動キャプチャ/リプレイによりホスト CPU オーバーヘッドを排除します。
