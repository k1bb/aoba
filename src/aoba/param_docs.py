"""Bilingual (JA/EN) physical descriptions for all AOBA TOML configuration parameters."""

from __future__ import annotations

PARAM_DOCS: dict[str, dict[str, str]] = {
    # =========================================================================
    # [grid] (16)
    # =========================================================================
    "glevel": {
        "ja": "正二十面体 SCVT 球面格子の再帰分割レベル (0〜6)。N_cells = 10×4^g + 2（G3:642, G4:2562, G5:10242セル）。",
        "en": "Icosahedral SCVT horizontal recursion level (0–6). N_cells = 10×4^g + 2 (G3=642, G4=2562, G5=10242 cells).",
    },
    "nz": {
        "ja": "ホスト大気と埋め込み CRM に共通する鉛直層数 [-]。成層圏標準は 44 層、高速対流圏実験は 32 層。",
        "en": "Number of shared vertical levels [-] for host atmosphere and embedded CRM (44 for stratosphere, 32 for troposphere).",
    },
    "z_top": {
        "ja": "大気モデル上端高度 [m]。48000 m (~0.85 hPa) で成層圏界面をカバー、24000 m で低トップ対流圏構成。",
        "en": "Atmospheric model top altitude [m] (48000 m ≈ 0.85 hPa stratopause; 24000 m low-top troposphere).",
    },
    "dz_min": {
        "ja": "地表最下層の格子厚 [m]。小さくすると境界層解像度が上がるが CFL 条件が厳しくなる。",
        "en": "Lowest atmospheric layer thickness [m]. Finer values resolve the PBL better but tighten vertical CFL.",
    },
    "z_sponge_base": {
        "ja": "モデル上部のレイリー減衰スポンジ層底面高度 [m]。上方伝播する重力波の反射を吸収する。",
        "en": "Base altitude [m] of the upper-level Rayleigh damping sponge absorbing upward-propagating gravity waves.",
    },
    "tau_sponge_min": {
        "ja": "モデル上端での最小スポンジ減衰時定数 [s]。小さいほど上端波減衰が強い。",
        "en": "Minimum Rayleigh sponge damping timescale [s] at the model top (smaller = stronger wave absorption).",
    },
    "terrain_scale_exp": {
        "ja": "Klemp (2011) 平滑化地形追従座標 (STF) の鉛直減衰指数 p [-]。大きいほど山岳の歪みが早く消える。",
        "en": "Vertical decay exponent p [-] in Klemp (2011) smoothed terrain-following (STF) coordinates.",
    },
    "crm_nx": {
        "ja": "各ホストセル内に埋め込む CRM の x 方向（東西）カラム数 [-]。8 で 32 km 幅（dx=4 km 時）。",
        "en": "Embedded CRM column count along x [-] per host cell (8 cols × 4 km = 32 km curtain).",
    },
    "crm_ny": {
        "ja": "埋め込み CRM の y 方向（南北）カラム数 [-]。1 で 2.5D カーテン（SP-CAM 標準）、>1 で 3D CRM。",
        "en": "Embedded CRM column count along y [-] (1 = 2D curtain for fast climate runs; >1 = full 3D CRM).",
    },
    "crm_dx": {
        "ja": "埋め込み CRM の水平格子幅 [m]。4000 m で積雲対流コアを直接解像（積雲パラメタリゼーション不要）。",
        "en": "Embedded CRM horizontal grid spacing [m] (4000 m explicitly resolves convective updrafts without cumulus schemes).",
    },
    "dt_host": {
        "ja": "ホスト大気・物理過程・SP 結合の時間刻み [s]。標準 720 s（12 分）。",
        "en": "Coupled host atmosphere, physics, and superparameterization time step [s] (default 720 s = 12 min).",
    },
    "n_host_dyn_substeps": {
        "ja": "1 結合ステップあたりのホスト力学最小サブステップ数 [-]。CFL 条件で自動増加される。",
        "en": "Minimum host RK3-HEVI dynamical substeps per dt_host [-] (auto-increased when required by CFL).",
    },
    "n_crm_dyn_substeps": {
        "ja": "1 結合ステップあたりの埋め込み CRM 力学サブステップ数 [-]。12 で dt_crm = 60 s（dt_host=720 s 時）。",
        "en": "Embedded CRM RK3-HEVI substeps per dt_host [-] (12 gives dt_crm = 60 s at dt_host = 720 s).",
    },
    "rad_stride": {
        "ja": "放射計算を呼び出すステップ間隔 [-]。5 で 1 時間ごと（dt_host=720 s 時）。",
        "en": "Radiation call interval in host steps [-] (5 = hourly radiative updates at dt_host = 720 s).",
    },
    "crm_col_chunk": {
        "ja": "CRM 計算時の GPU/CPU メモリ制御チャンク幅 [カラム数]。解像度には影響しない。",
        "en": "Column batch chunk size [columns] for CRM memory control; does not alter numerical results.",
    },
    "rad_col_chunk": {
        "ja": "放射計算時の GPU/CPU メモリ制御チャンク幅 [カラム数]。解像度には影響しない。",
        "en": "Column batch chunk size [columns] for radiation memory control; does not alter numerical results.",
    },

    # =========================================================================
    # [forcing] (83)
    # =========================================================================
    "co2_ppmv": {
        "ja": "大気 CO₂ 濃度 [ppmv]（co2_mode='constant' 時の固定値、'prognostic' 時の初期値。産業革命前 284.3）。",
        "en": "Atmospheric CO₂ mole fraction [ppmv] (fixed value in 'constant' mode or initial value in 'prognostic' mode).",
    },
    "co2_mode": {
        "ja": "大気 CO₂ モード選択: constant(固定) / scenario(CMIP・古気候補間) / csv(独自表) / prognostic(排出駆動双方向予報)。",
        "en": "CO₂ mode selector: 'constant', 'scenario' (CMIP/paleo/NOAA), 'csv' (custom), or 'prognostic' (emissions-driven).",
    },
    "co2_scenario": {
        "ja": "co2_mode='scenario' 時に参照する CO₂ 推移シナリオ名（NOAA 観測・SSP・RCP・氷床コア 80 万年等）。",
        "en": "Scenario identifier used when co2_mode='scenario' (e.g., obs_historical_noaa, cmip6_ssp245, paleo_800kyr).",
    },
    "co2_csv_path": {
        "ja": "co2_mode='csv' 時に読み込む独自 (year, co2_ppmv) CSV ファイルパス。",
        "en": "Path to a custom (year, co2_ppmv) CSV table when co2_mode='csv'.",
    },
    "ch4_ppmv": {
        "ja": "大気メタン CH₄ 濃度 [ppmv]（既定 0.80825 = 1850 年 piControl）。予報有効時は初期値となる。",
        "en": "Atmospheric CH₄ concentration [ppmv] (default 0.80825 = 1850 piControl; initial value if prognostic).",
    },
    "n2o_ppmv": {
        "ja": "大気一酸化二窒素 N₂O 濃度 [ppmv]（既定 0.27302 = 1850 年 piControl）。予報有効時は初期値となる。",
        "en": "Atmospheric N₂O concentration [ppmv] (default 0.27302 = 1850 piControl; initial value if prognostic).",
    },
    "rad_scheme": {
        "ja": "放射伝達スキーム選択: four_band(高速4帯) / ckd(標準16帯50g点) / ckd_hd(高精度29帯111g点) / rrtmgp(公式HITRAN 240g点)。",
        "en": "Radiation scheme selector: 'four_band' (fast), 'ckd' (default 16-band), 'ckd_hd' (29-band), or 'rrtmgp' (HITRAN 240 g-pt).",
    },
    "enable_rad_3d_tica": {
        "ja": "3D 傾斜独立カラム近似 (TICA) の有効化。太陽天頂角に応じ雲を水平シフトし、雲側面散乱・影を近似する。",
        "en": "Enable 3D Tilted Independent Column Approximation (TICA) shifting cloud columns along slant solar beams.",
    },
    "o3_mode": {
        "ja": "オゾン O₃ モード選択: rcemip(1D) / latitude_seasonal(2D解析) / climatology(外部CMIP) / prognostic(3D光化学予報)。",
        "en": "Ozone mode: 'rcemip' (1D), 'latitude_seasonal' (2D analytic), 'climatology' (CMIP), or 'prognostic' (3D photochemistry).",
    },
    "o3_file_path": {
        "ja": "o3_mode='climatology' 時に読み込む外部 CMIP6/CMIP7 帯状平均オゾンファイルパス（空欄で自動探索）。",
        "en": "Path to external CMIP6/CMIP7 zonal-mean ozone file when o3_mode='climatology' (empty = auto-discover).",
    },
    "o3_lw_coupling": {
        "ja": "オゾン 9.6 μm 窓領域長波吸収および成層圏放射冷却結合の有効化。",
        "en": "Enable 9.6 μm ozone longwave window absorption and stratospheric radiative coupling.",
    },
    "o3_ecl_ppbv": {
        "ja": "成層圏等価有効塩素量 EESC [ppbv]。極域成層圏雲 (PSC) によるオゾンホール破壊率を制御（現代~3.5、1850年~0.55）。",
        "en": "Stratospheric equivalent effective chlorine EESC [ppbv] controlling polar PSC ozone loss (~0.55 PI, ~3.5 modern).",
    },
    "enable_nonoro_gwd": {
        "ja": "地図不要・流れ依存の非地形性重力波抵抗（前線・対流起源）の有効化。成層圏ジェットや QBO の駆動に重要。",
        "en": "Enable map-free flow-dependent non-orographic gravity wave drag (frontal & convective sources).",
    },
    "solar_constant": {
        "ja": "太陽定数 S₀ [W m⁻²]（現代地球 1361.0、微弱な若い太陽 1280、TRAPPIST-1e 900）。",
        "en": "Total solar irradiance S₀ [W m⁻²] at 1 AU (modern Earth 1361.0, faint young Sun ~1280, TRAPPIST-1e 900).",
    },
    "omega": {
        "ja": "惑星自転角速度 Ω [rad s⁻¹]（地球 7.292115×10⁻⁵）。コリオリ力とハドレー循環幅を直接支配する。",
        "en": "Planetary angular rotation rate Ω [rad s⁻¹] (Earth = 7.292115e-5); governs Coriolis force and Hadley width.",
    },
    "obliquity_deg": {
        "ja": "自転軸傾斜角（黄道傾斜角）[deg]（現代地球 23.44°）。季節変化の強さを決定する。",
        "en": "Planetary obliquity (axial tilt) [deg] (modern Earth 23.44°); controls seasonal amplitude.",
    },
    "eccentricity": {
        "ja": "公転軌道離心率 e [-]（現代地球 0.0167、円軌道 0.0）。近日点と遠日点の日射差を生む。",
        "en": "Orbital eccentricity e [-] (modern Earth 0.0167; 0.0 = circular orbit).",
    },
    "perihelion_deg": {
        "ja": "春分点から測った近日点黄経 [deg]（現代地球 283°＝北半球冬に近日点）。",
        "en": "Longitude of perihelion from vernal equinox [deg] (modern Earth 283°, perihelion in NH winter).",
    },
    "start_date": {
        "ja": "シミュレーション開始暦日（YYYY-MM-DD 形式、例: 2000-03-20）。",
        "en": "Simulation calendar start date in YYYY-MM-DD format (e.g., 2000-03-20).",
    },
    "calendar_type": {
        "ja": "暦法選択: gregorian(閏年あり) / noleap(365日固定) / 360_day(理想360日暦)。",
        "en": "Calendar convention: 'gregorian' (leap years), 'noleap' (365 days/yr), or '360_day' (idealized 12×30 days).",
    },
    "insolation_mode": {
        "ja": "日射季節モード: seasonal(通常の年周季節変化) / perpetual_date(指定日固定) / annual_mean(年平均固定)。",
        "en": "Insolation mode: 'seasonal' (full annual cycle), 'perpetual_date' (fixed day), or 'annual_mean'.",
    },
    "fixed_insolation_date": {
        "ja": "insolation_mode='perpetual_date' 時に固定する暦日（YYYY-MM-DD、例: 春分固定なら 2000-03-20）。",
        "en": "Calendar date (YYYY-MM-DD) held fixed when insolation_mode='perpetual_date'.",
    },
    "diurnal_cycle": {
        "ja": "日周日射サイクルの有効化。false にすると各緯度の日平均太陽天頂角で連続照射する。",
        "en": "Enable diurnal solar cycle. If false, applies smooth daily-mean insolation at each latitude.",
    },
    "use_earth2014": {
        "ja": "Earth2014 実地球地形・海底地形・氷床幾何データを使用する（false で理想化大陸／水惑星）。",
        "en": "Load real Earth2014 topography, bathymetry, and ice-sheet geometry (false = idealized geometry).",
    },
    "earth2014_dir": {
        "ja": "Earth2014 境界データ（5分格子）を格納・自動ダウンロードするディレクトリパス。",
        "en": "Local directory caching/auto-downloading Earth2014 5-arcmin surface elevation/bathymetry files.",
    },
    "earth2014_stride": {
        "ja": "Earth2014 読み込み時の間引きストライド [-]。2 で高速・省メモリに地形と粗さ σ_topo を抽出。",
        "en": "Subsampling stride [-] when reading Earth2014 grids to accelerate mesh initialization.",
    },
    "enable_superparam": {
        "ja": "埋め込み CRM による超パラメタリゼーション (SP) の有効化。false にすると単独の全球雲解像モデルモードになる。",
        "en": "Enable embedded-CRM superparameterization (SP). False switches AOBA into standalone global CRM mode.",
    },
    "enable_dust": {
        "ja": "地図不要の鉱物ダスト飛散・3D 輸送・放射・氷晶核 (INP) 結合の有効化。",
        "en": "Enable map-free wind-driven mineral dust emission, 3D FCT transport, radiative forcing, and INP coupling.",
    },
    "dust_emis_factor": {
        "ja": "鉱物ダスト放出スケーリング係数 [kg s² m⁻⁵]。裸地摩擦速度 u* に比例するダスト光学的厚さを調整する。",
        "en": "Global mineral dust emission scaling factor [kg s² m⁻⁵] controlling bare-soil dust aerosol optical depth.",
    },
    "h_ml_ocean": {
        "ja": "1.5層エクマン海洋 (`ekman_sw`) 使用時の海洋混合層厚 [m]（既定 50 m）。",
        "en": "Ocean mixed-layer depth [m] used in the 1.5-layer Ekman ocean mode ('ekman_sw'; default 50 m).",
    },
    "ocean_mode": {
        "ja": "海洋モデル選択: ogcm_3d(3次元多層OGCM・海底地形・塩分・GM90) / ekman_sw(高速1.5層浅水エクマン海洋)。",
        "en": "Ocean model selector: 'ogcm_3d' (3D multi-layer SCVT OGCM) or 'ekman_sw' (fast 1.5-layer shallow-water Ekman).",
    },
    "ocean_glevel": {
        "ja": "独立高解像度海洋グリッドの分割レベル [-]。0 で大気と同一格子、>0（例: 6〜7）で細密海洋格子を結合。",
        "en": "Independent fine ocean SCVT glevel [-] (0 = match atmosphere grid; >0 enables fine-ocean coupling).",
    },
    "ocean_nz": {
        "ja": "3D OGCM の鉛直層数 [-]（既定 18 層、表層〜深海 5000 m）。",
        "en": "Number of vertical depth layers [-] in the 3D OGCM (default 18 layers spanning surface to abyss).",
    },
    "enable_seaice_drift": {
        "ja": "風応力・海流による海氷の水平力学移流（漂流）を有効化する。",
        "en": "Enable wind- and current-driven horizontal sea-ice dynamical drift.",
    },
    "enable_ocean_bgc": {
        "ja": "FlexPFT 海洋生物地球化学（NO₃・PO₄・鉄 Fe・DIC・アルカリ度・クロロフィル・DMS）の有効化。",
        "en": "Enable 3D FlexPFT marine biogeochemistry (NO₃, PO₄, Fe, DIC, Alk, emergent chlorophyll, DMS).",
    },
    "enable_river_routing": {
        "ja": "地形勾配に基づく D8 全球河川流路・Manning 流速ルーティング（陸域流出 $\to$ 沿岸海洋淡水・栄養塩供給）。",
        "en": "Enable topographic D8 global river routing with Manning flow delivering runoff and nutrients to coastal ocean.",
    },
    "enable_land_carbon": {
        "ja": "陸域 6 プール炭素循環（葉・木部・根・リター・土壌炭素）および純生態系生産量 NBP の予報を有効化。",
        "en": "Enable prognostic 6-pool terrestrial carbon cycle (leaf, wood, root, litter, soil C) and land NBP.",
    },
    "enable_land_nitrogen": {
        "ja": "陸域窒素循環（無機 N プール・生物的窒素固定 BNF・窒素制限による光合成 Vcmax 調整）を有効化。",
        "en": "Enable terrestrial nitrogen cycle (mineral N pool, BNF, and N-limitation feedback on Vcmax and LAI).",
    },
    "enable_lightning": {
        "ja": "埋め込み CRM 上昇流・氷／霰粒子に基づく発雷頻度および雷起源 NOₓ (LNOx) 放出を有効化。",
        "en": "Enable CRM convective updraft/ice-graupel lightning flash rate and lightning NOₓ (LNOx) production.",
    },
    "enable_fire_veg_feedback": {
        "ja": "SPITFIRE 機構に基づく MAML パッチ山火事（発雷着火・乾燥度・植生燃焼・煙エアロゾル・CO₂ 放出）を有効化。",
        "en": "Enable SPITFIRE wildfire ignition/spread on MAML land patches with vegetation mortality and smoke/CO₂ emissions.",
    },
    "enable_seaice_evp_itd": {
        "ja": "5 カテゴリ氷厚分布 (ITD) と弾粘塑性 (EVP) レオロジーによる高精度海氷力学（融合 GPU カーネル）を有効化。",
        "en": "Enable 5-category Ice Thickness Distribution (ITD) and Elastic-Viscous-Plastic (EVP) sea-ice rheology.",
    },
    "enable_ogcm_fct_zstar": {
        "ja": "OGCM における Zalesak FCT トレーサ移流および z* 自由表面座標（淡水収支の厳密保存）を有効化。",
        "en": "Enable Zalesak FCT tracer advection and z* free-surface vertical coordinates in the 3D OGCM.",
    },
    "enable_ogcm_scale_aware_gm": {
        "ja": "第一傾圧ロスビー変形半径に応じ中規模渦拡散 κ_GM を自動減衰する解像度適応 GM90 (Hallberg, 2013) を有効化。",
        "en": "Enable Hallberg (2013) scale-aware tapering of Gent–McWilliams eddy diffusivity for eddy-permitting ocean grids.",
    },
    "enable_ogcm_submeso": {
        "ja": "海洋混合層内の前線サブメソスケール再成層パラメタリゼーション (Fox-Kemper et al., 2008) を有効化。",
        "en": "Enable Fox-Kemper et al. (2008) mixed-layer submesoscale eddy restratification in the OGCM.",
    },
    "enable_ogcm_tidal_mixing": {
        "ja": "Earth2014 海底粗さに基づく深海内部潮汐混合 (St. Laurent et al., 2002) を有効化。深層循環 (AMOC) 維持に寄与。",
        "en": "Enable St. Laurent et al. (2002) rough-bathymetry abyssal internal tidal mixing sustaining overturning circulation.",
    },
    "enable_current_rel_wind": {
        "ja": "海面フラックス・風応力を海流相対風速 (u_a - u_o) で計算（3D CRM または standalone CRM 時に自動適用）。",
        "en": "Compute air-sea fluxes using current-relative wind (u_a - u_o) when 3D CRM or standalone CRM is active.",
    },
    "enable_natural_aerosols": {
        "ja": "自然起源エアロゾル（海塩・海洋 DMS 硫酸塩・山火事 BC/OC）の 3D 放出・輸送・乾湿沈着を有効化。",
        "en": "Enable 3D prognostic natural aerosols (sea salt, marine DMS sulfate, wildfire BC/OC) with dry/wet deposition.",
    },
    "enable_aerosol_rad_forcing": {
        "ja": "予報エアロゾル（ダスト・海塩・硫酸塩・BC・OC/SOA）の直接放射強制（短波散乱・吸収）を放射へ結合。",
        "en": "Couple prognostic 5-species aerosols into shortwave/longwave direct radiative transfer.",
    },
    "enable_snow_bc_darkening": {
        "ja": "SNICAR 方式による黒色炭素 (BC)・ダスト沈着の積雪・海氷アルベド暗化効果を有効化。",
        "en": "Enable SNICAR-style black carbon and dust darkening of snow and sea-ice visible albedo.",
    },
    "enable_bgc_fe_lnox_coupling": {
        "ja": "鉱物ダスト可溶性鉄 (Fe) および雷 LNOx の海洋沈着を FlexPFT 海洋基礎生産へ結合する。",
        "en": "Couple atmospheric dust soluble iron (Fe) and lightning LNOx deposition into FlexPFT marine productivity.",
    },
    "enable_bvoc_soa": {
        "ja": "EEO 森林からの生物起源揮発性有機化合物 (BVOC) 放出と二次有機エアロゾル (SOA)/CCN 生成を有効化。",
        "en": "Enable terrestrial EEO forest BVOC emissions forming hygroscopic secondary organic aerosols (SOA) and CCN.",
    },
    "enable_prognostic_ch4_n2o": {
        "ja": "湿地・山火事・土壌・海洋 OMZ からの CH₄ / N₂O 放出と OH・成層圏 UV 消滅を解く温室効果ガス予報を有効化。",
        "en": "Enable prognostic global CH₄ and N₂O budgets balancing natural sources against OH and stratospheric UV sinks.",
    },
    "enable_silicate_weathering": {
        "ja": "数万年〜数億年スケールのケイ酸塩風化–火山 CO₂ サーモスタット（Walker et al., 1981）を有効化。",
        "en": "Enable Walker et al. (1981) carbonate–silicate weathering thermostat regulating long-term atmospheric CO₂.",
    },
    "enable_co2_3d_transport": {
        "ja": "co2_mode='prognostic' 時に 3 次元 CO₂ 移流・PBL 混合を有効化し、地域的な CO₂ 季節変動と南北勾配を再現。",
        "en": "Enable 3D SCVT FCT transport and PBL mixing of prognostic CO₂ to resolve seasonal and meridional gradients.",
    },
    "co2_anthro_pgc_yr": {
        "ja": "人為的 CO₂ 排出速度 [Pg C yr⁻¹]（co2_mode='prognostic' 時に大気炭素収支へ直接加算。現代 ~10 PgC/yr）。",
        "en": "Prescribed anthropogenic CO₂ emission rate [Pg C yr⁻¹] for prognostic CO₂ experiments (~10 PgC/yr modern).",
    },
    "enable_ocean_o2_omz": {
        "ja": "海洋 3D 溶存酸素 (O₂)・貧酸素水塊 (OMZ)・嫌気的脱窒および海洋 N₂O 放出の予報を有効化。",
        "en": "Enable 3D marine dissolved O₂, oxygen minimum zones (OMZ), anaerobic denitrification, and marine N₂O fluxes.",
    },
    "enable_volcanic_aerosols": {
        "ja": "成層圏火山性エアロゾル（SO₂ → H₂SO₄ 酸化・拡散・沈降）の放射・オゾン不均一破壊結合を有効化。",
        "en": "Enable stratospheric volcanic SO₂ → H₂SO₄ aerosol life cycle coupled to SW/LW radiation and heterogeneous O₃ loss.",
    },
    "volcanic_eruption_year": {
        "ja": "離散的大規模火山噴火パルスを発火するシミュレーション年 [yr]（負値で無効）。",
        "en": "Simulation year [yr] at which to inject a discrete explosive volcanic SO₂ pulse (< 0 disables).",
    },
    "volcanic_so2_tg": {
        "ja": "離散的火山噴火パルスの成層圏 SO₂ 注入量 [Tg SO₂]（例: 1991年ピナツボ噴火 ≈ 17 Tg）。",
        "en": "Stratospheric SO₂ mass [Tg SO₂] injected by a discrete eruption pulse (e.g., 17 Tg for 1991 Mt. Pinatubo).",
    },
    "volcanic_eruption_lat_deg": {
        "ja": "離散的火山噴火の緯度 [deg N]（例: ピナツボ 15°N、熱帯噴火は両半球へ拡散）。",
        "en": "Latitude [deg N] of the discrete volcanic eruption plume (tropical eruptions spread into both hemispheres).",
    },
    "volcanic_eruption_width_deg": {
        "ja": "噴火初期プルームの南北ガウス半値幅 [deg]。",
        "en": "Initial meridional Gaussian half-width [deg] of the injected volcanic SO₂ plume.",
    },
    "sai_so2_tg_yr": {
        "ja": "成層圏エアロゾル注入 (SAI) ジオエンジニアリング実験用の連続 SO₂ 注入率 [Tg SO₂ yr⁻¹]（0 で無効）。",
        "en": "Continuous stratospheric aerosol injection (SAI) SO₂ rate [Tg SO₂ yr⁻¹] for geoengineering experiments (0 = off).",
    },
    "sai_lat_band_deg": {
        "ja": "連続 SAI 注入を適用する低緯度帯幅 [±deg]。",
        "en": "Meridional half-band [±deg] over which continuous SAI SO₂ is injected.",
    },
    "enable_solar_cycle": {
        "ja": "11 年シュワーベ太陽活動周期による全太陽放射照度 (TSI) および成層圏 UV 光解離の変調を有効化。",
        "en": "Enable 11-year Schwabe solar cycle modulation of total solar irradiance (TSI) and stratospheric UV photolysis.",
    },
    "solar_cycle_amp_wm2": {
        "ja": "太陽活動周期の極大〜極小 TSI 振幅 [W m⁻²]（観測値 ≈ 1.0 W m⁻²）。",
        "en": "Peak-to-trough TSI amplitude [W m⁻²] of the solar cycle (observed ≈ 1.0 W m⁻²).",
    },
    "solar_cycle_period_yr": {
        "ja": "太陽活動周期の長さ [yr]（標準 11.0 年）。",
        "en": "Solar activity cycle period [yr] (default 11.0 yr).",
    },
    "solar_cycle_phase_yr": {
        "ja": "太陽活動周期の基準位相年 [yr]。",
        "en": "Reference phase year [yr] for the sinusoidal solar activity cycle.",
    },
    "volcanic_co2_pgc_yr": {
        "ja": "地質学的火山脱ガス CO₂ 供給率 [Pg C yr⁻¹]（ケイ酸塩風化サーモスタットとの長期平衡用、既定 0.03）。",
        "en": "Background volcanic CO₂ outgassing rate [Pg C yr⁻¹] balancing silicate weathering (default 0.03).",
    },
    "enable_maml_subgrid_topo": {
        "ja": "MAML サブグリッド山岳微気候ダウンスケーリング（Earth2014 σ_topo からの標高気温減率・斜面日射差）を有効化。",
        "en": "Enable MAML subgrid orographic lapse-rate and slope-aspect microclimate downscaling from Earth2014 σ_topo.",
    },
    "enable_maml_wetlands_lakes": {
        "ja": "MAML 動的湿地・氾濫原・湖沼の冠水率診断（水面蒸発・湿地 CH₄ 放出への結合）を有効化。",
        "en": "Enable dynamic MAML inundated wetlands, floodplains, and lakes coupled to open-water evaporation and CH₄.",
    },
    "enable_wave_coupling": {
        "ja": "予報海洋波浪 (H_s, T_p)・Stokes ドリフト Langmuir 混合・波齢粗度・強風時海塩飛沫フラックスを有効化。",
        "en": "Enable prognostic ocean waves (H_s, T_p), Stokes-drift Langmuir KPP mixing, wave drag, and sea-spray fluxes.",
    },
    "enable_water_isotopes": {
        "ja": "3 次元水安定同位体 (H₂¹⁸O, HDO) の輸送・蒸発/凝結分別および積雪・氷床コア δ¹⁸O アーカイブを有効化。",
        "en": "Enable 3D prognostic stable water isotopes (H₂¹⁸O, HDO), Rayleigh/Craig–Gordon fractionation, and snow δ¹⁸O archive.",
    },
    "planet_preset": {
        "ja": "汎用惑星プリセット: earth / aquaplanet(全球海) / snowball_earth(全球凍結) / tidally_locked / trappist1e / mars_like。",
        "en": "Universal planetary preset: 'earth', 'aquaplanet', 'snowball_earth', 'tidally_locked', 'trappist1e', or 'mars_like'.",
    },
    "enable_tidal_lock": {
        "ja": "同期潮汐固定日射（太陽直下点を経度固定、常昼半球と常夜半球を形成する系外惑星モード）を有効化。",
        "en": "Enable synchronous tidally locked insolation with a fixed substellar point (eyeball exoplanet climate).",
    },
    "substellar_lon_deg": {
        "ja": "潮汐固定モード (`enable_tidal_lock = true`) 時の太陽直下点経度 [deg E]。",
        "en": "Fixed substellar longitude [deg E] when enable_tidal_lock is true.",
    },
    "weathering_accel_factor": {
        "ja": "ケイ酸塩風化・火山脱ガスの地質時間加速倍率 [-]。古気候の長期 CO₂ 平衡を短時間で到達させる。",
        "en": "Geological acceleration multiplier [-] for silicate weathering and volcanic CO₂ equilibration.",
    },
    "enable_milankovitch_orbit": {
        "ja": "Berger (1978) / Laskar (2004) に基づくミランコビッチ軌道要素（離心率・傾斜角・歳差）の連続進化を有効化。",
        "en": "Enable continuous analytical Milankovitch orbital evolution (eccentricity, obliquity, precession).",
    },
    "milankovitch_start_kyr": {
        "ja": "ミランコビッチ軌道計算の開始エポック（J2000 基準 [kyr]、例: 最終氷期極大期 LGM なら -21.0）。",
        "en": "Starting orbital epoch relative to J2000 [kyr] (e.g., -21.0 for Last Glacial Maximum).",
    },
    "milankovitch_accel_factor": {
        "ja": "ミランコビッチ軌道進化の時間加速倍率 [-]（例: 100 で 100 モデル年に 1 万年分の軌道変化）。",
        "en": "Orbital acceleration factor [-] for paleoclimate transient runs (e.g., 100 compresses 10 kyr into 100 model yr).",
    },
    "re_liq_um": {
        "ja": "1 モーメント微物理時（またはフォールバック時）の液雲有効半径 r_e [μm]（既定 10 μm）。",
        "en": "Fallback / 1-moment liquid cloud droplet effective radius r_e [μm] (default 10 μm).",
    },
    "re_ice_um": {
        "ja": "1 モーメント微物理時（またはフォールバック時）の氷雲有効半径 r_e [μm]（既定 48 μm）。",
        "en": "Fallback / 1-moment ice cloud crystal effective radius r_e [μm] (default 48 μm).",
    },
    "cloud_sw_inhomog": {
        "ja": "雲内部のサブグリッド短波光学的不均一性係数 χ_SW [-]（Cahalan et al., 1994; 既定 0.62）。小さいほど雲アルベド低下。",
        "en": "Subgrid shortwave cloud optical inhomogeneity factor χ_SW [-] (Cahalan et al., 1994; default 0.62).",
    },
    "microphysics_scheme": {
        "ja": "雲微物理スキーム選択: one_moment(1M SAM) / two_moment(標準2M KK00) / six_class_2m(6種2M Morrison雨雪霰)。",
        "en": "Cloud microphysics selector: 'one_moment' (SAM 1M), 'two_moment' (default KK00), or 'six_class_2m' (Morrison 6-class).",
    },

    # =========================================================================
    # [icesheet] (5)
    # =========================================================================
    "enable_dynamic_icesheet": {
        "ja": "Glen 流動則 (n=3) SIA 動的氷床モデルと基盤アイソスタシー・鉛直格子再構築を有効化。",
        "en": "Enable Glen (n=3) SIA dynamic ice-sheet flow, bedrock isostasy, and conservative vertical grid remapping.",
    },
    "icesheet_ema_years": {
        "ja": "氷床表面質量収支 (SMB) の年々変動を平滑化する指数移動平均 (EMA) 時定数 [yr]。",
        "en": "Exponential moving-average (EMA) smoothing timescale [yr] for climatic ice-sheet surface mass balance.",
    },
    "icesheet_accel_factor": {
        "ja": "氷床流動・質量収支の時間加速倍率 [-]（1.0 で実時間、100.0 で 1 モデル年あたり 100 年分の氷床進化）。",
        "en": "Ice-sheet asynchronous time acceleration factor [-] (1.0 = real time; 100.0 = 100 ice-years per climate year).",
    },
    "icesheet_accel_spinup_years": {
        "ja": "氷床加速倍率を 1.0 へ指数減衰させるスピンアップ年数 [yr]（0.0 で一定倍率維持）。",
        "en": "Spin-up e-folding timescale [yr] relaxing acceleration toward 1.0 (0.0 keeps constant acceleration).",
    },
    "icesheet_isostasy_tau_yr": {
        "ja": "氷床荷重に対するマントル・リソスフェアのアイソスタシー緩和時定数 [yr]（既定 1000 年）。",
        "en": "Glacial isostatic adjustment (bedrock–mantle relaxation) timescale [yr] (default 1000 yr).",
    },

    # =========================================================================
    # [output] (23)
    # =========================================================================
    "out_dir": {
        "ja": "計算結果（Zarr / CSV / リスタート）の保存先ディレクトリ（空欄で ~/aoba_outputs を自動使用）。",
        "en": "Output directory for Zarr stores, CSV diagnostics, and restarts (empty = default ~/aoba_outputs).",
    },
    "years": {
        "ja": "積分年数 [yr]（end_date が空欄のときに有効）。",
        "en": "Total integration length [yr] (used when end_date is empty).",
    },
    "end_date": {
        "ja": "任意の終了暦日（YYYY-MM-DD 形式）。指定すると years より優先される。",
        "en": "Optional explicit calendar end date (YYYY-MM-DD); overrides 'years' when non-empty.",
    },
    "days_per_year": {
        "ja": "1 年の公転日数 [days yr⁻¹]（現代地球グレゴリオ平均 365.2425 日）。",
        "en": "Orbital year length [days yr⁻¹] (default 365.2425 days for Gregorian Earth).",
    },
    "zarr_dir_partition": {
        "ja": "Zarr 出力分割方式: yearly(年別 year_XXXX/ フォルダへ分割・推奨) / single(単一ストアへ追記)。",
        "en": "Zarr partitioning strategy: 'yearly' (per-year year_XXXX/ folders, recommended) or 'single'.",
    },
    "daily_csv_enabled": {
        "ja": "全球日平均時系列ファイル `daily_global.csv` の書き出しを有効化。",
        "en": "Enable writing daily global-mean diagnostic time series (`daily_global.csv`).",
    },
    "daily_csv_interval_days": {
        "ja": "`daily_global.csv` の出力時間間隔 [日]（既定 1.0 日）。",
        "en": "Output cadence [days] for `daily_global.csv` (default 1.0 day).",
    },
    "monthly_mean_enabled": {
        "ja": "時間平均 2D/3D/サブグリッド気候場 `monthly.zarr` の出力を有効化。",
        "en": "Enable writing time-averaged climate fields to `monthly.zarr`.",
    },
    "monthly_mean_periods_per_year": {
        "ja": "`monthly.zarr` の年間平均区間数 [-]（12 で月平均、24 で半月平均）。",
        "en": "Number of averaging periods per year [-] in `monthly.zarr` (12 = monthly, 24 = semi-monthly).",
    },
    "monthly_mean_sample_hours": {
        "ja": "月平均積算時のサンプリング間隔 [時間]（既定 6.0 時間＝1日4回積算）。",
        "en": "Sampling cadence [hours] accumulated into each `monthly.zarr` average (default 6.0 h).",
    },
    "monthly_mean_include_2d": {
        "ja": "`monthly.zarr` に 2D 地表・TOA・海洋・ESM 診断変数を含める。",
        "en": "Include 2D surface, TOA, ocean, and ESM diagnostic variables in `monthly.zarr`.",
    },
    "monthly_mean_include_3d": {
        "ja": "`monthly.zarr` に 3D 大気変数（気温・気圧・水蒸気・風速等）を含める。",
        "en": "Include 3D atmospheric variables (T, p, q_v, u, v) in `monthly.zarr`.",
    },
    "monthly_mean_include_subgrid_2d": {
        "ja": "`monthly.zarr` に埋め込み CRM / MAML サブグリッド 2D 統計変数を含める。",
        "en": "Include embedded CRM / MAML subgrid 2D statistics in `monthly.zarr`.",
    },
    "snapshots_enabled": {
        "ja": "瞬時スナップショット `snapshots.zarr`（全 ESM 2D/3D 瞬時場）の出力を有効化。",
        "en": "Enable writing instantaneous state snapshots to `snapshots.zarr`.",
    },
    "snapshots_periods_per_year": {
        "ja": "`snapshots.zarr` の年間出力回数 [-]（既定 12 回＝毎月1回）。",
        "en": "Number of instantaneous snapshots written per year [-] (default 12).",
    },
    "snapshots_include_2d": {
        "ja": "`snapshots.zarr` に 2D 表面・ESM 変数（オゾン・同位体・波浪・湿地・CO₂ 等）を含める。",
        "en": "Include 2D surface and Full-ESM variables (O₃, isotopes, waves, wetlands, CO₂) in `snapshots.zarr`.",
    },
    "snapshots_include_3d": {
        "ja": "`snapshots.zarr` に 3D 大気瞬時場を含める。",
        "en": "Include 3D instantaneous atmospheric fields in `snapshots.zarr`.",
    },
    "snapshots_include_subgrid_2d": {
        "ja": "`snapshots.zarr` に CRM / MAML サブグリッド 2D 瞬時場を含める。",
        "en": "Include CRM / MAML subgrid 2D instantaneous fields in `snapshots.zarr`.",
    },
    "restart_interval_years": {
        "ja": "年次チェックポイント `restart.pt` の保存間隔 [年]（既定 1 年）。",
        "en": "Annual checkpoint (`restart.pt`) save interval [years] (default 1).",
    },
    "restart_monthly": {
        "ja": "毎月末に中間リスタート `restart_mid.pt` を自動保存（途中中断からの完全決定論的再開を可能にする）。",
        "en": "Save mid-year checkpoint `restart_mid.pt` at every month end for lossless interruption recovery.",
    },
    "cmip7_output_enabled": {
        "ja": "CMIP7 Data Request v1.2 準拠の標準化月平均 Zarr ストリーム出力を有効化（スピンアップ中は false 推奨）。",
        "en": "Enable CMIP7 Data Request v1.2 standardized monthly Zarr stream (keep false during spin-up).",
    },
    "cmip7_include_plev19": {
        "ja": "CMIP7 出力時に 3D 大気変数を標準 19 気圧面 (plev19) へ鉛直補間して出力する。",
        "en": "Interpolate 3D atmospheric variables onto CMIP7 standard 19 pressure levels (plev19).",
    },
    "cmip7_include_ocean_3d": {
        "ja": "CMIP7 出力時に 3D 海洋変数（水温・塩分・流速・BGC）を海洋鉛直層 (olevel) で出力する。",
        "en": "Include 3D ocean variables (thetao, so, uo, vo, BGC) on model ocean levels (olevel) in CMIP7 output.",
    },

    # =========================================================================
    # [constants] (26)
    # =========================================================================
    "earth_radius": {
        "ja": "惑星平均半径 R_E [m]（地球 6.371×10⁶ m）。SCVT セル面積・距離・球面微分演算子に反映。",
        "en": "Mean planetary radius R_E [m] (Earth 6.371e6 m); scales SCVT areas, distances, and spherical operators.",
    },
    "gravity": {
        "ja": "表面重力加速度 g [m s⁻²]（地球 9.79764 m s⁻²）。静力学平衡と浮力を支配する。",
        "en": "Surface gravitational acceleration g [m s⁻²] (Earth 9.79764); governs hydrostatics and buoyancy.",
    },
    "omega_earth": {
        "ja": "地球基準自転角速度 Ω_E [rad s⁻¹]（7.292115×10⁻⁵）。既定の参照定数。",
        "en": "Reference Earth angular rotation rate Ω_E [rad s⁻¹] (7.292115e-5).",
    },
    "seconds_per_day": {
        "ja": "1 恒星/太陽日の秒数 [s day⁻¹]（地球 86400 s）。日周サイクル・日次積算の基準。",
        "en": "Length of one planetary solar day [s day⁻¹] (Earth 86400 s).",
    },
    "r_d": {
        "ja": "乾燥空気の気体定数 R_d [J kg⁻¹ K⁻¹]（287.04）。状態方程式 p = ρ R T に使用。",
        "en": "Specific gas constant for dry air R_d [J kg⁻¹ K⁻¹] (287.04).",
    },
    "cp_d": {
        "ja": "乾燥空気の定圧比熱 c_pd [J kg⁻¹ K⁻¹]（1004.64）。断熱減率と温位変換に使用。",
        "en": "Specific heat capacity of dry air at constant pressure c_pd [J kg⁻¹ K⁻¹] (1004.64).",
    },
    "r_v": {
        "ja": "水蒸気の気体定数 R_v [J kg⁻¹ K⁻¹]（461.5）。仮温位浮力と Clausius–Clapeyron 式に使用。",
        "en": "Specific gas constant for water vapor R_v [J kg⁻¹ K⁻¹] (461.5).",
    },
    "cp_v": {
        "ja": "水蒸気の定圧比熱 c_pv [J kg⁻¹ K⁻¹]（1846.0）。",
        "en": "Specific heat capacity of water vapor at constant pressure c_pv [J kg⁻¹ K⁻¹] (1846.0).",
    },
    "p_ref": {
        "ja": "温位・エクスナー関数の基準気圧 p₀ [Pa]（100000 Pa = 1000 hPa）。",
        "en": "Reference pressure p₀ [Pa] for potential temperature and Exner function (100000 Pa).",
    },
    "t_freeze": {
        "ja": "純水の凝固点温度 [K]（273.15 K = 0 °C）。積雪・凍土・相変化の基準。",
        "en": "Freezing point of pure liquid water [K] (273.15 K = 0 °C).",
    },
    "t_triple": {
        "ja": "水の三重点温度 [K]（273.16 K）。飽和水蒸気圧式の基準点。",
        "en": "Triple-point temperature of water [K] (273.16 K) used in saturation vapor pressure.",
    },
    "e_triple": {
        "ja": "水の三重点飽和水蒸気圧 [Pa]（611.2 Pa）。",
        "en": "Saturation vapor pressure at the triple point of water [Pa] (611.2 Pa).",
    },
    "l_v0": {
        "ja": "0 °C における水の蒸発潜熱 L_v0 [J kg⁻¹]（2.501×10⁶）。",
        "en": "Latent heat of vaporization of water at 0 °C [J kg⁻¹] (2.501e6).",
    },
    "l_f0": {
        "ja": "0 °C における水の融解潜熱 L_f0 [J kg⁻¹]（3.337×10⁵）。",
        "en": "Latent heat of fusion of ice at 0 °C [J kg⁻¹] (3.337e5).",
    },
    "t_ice_all": {
        "ja": "混合相雲が 100% 氷晶となる下限温度 [K]（既定 253.15 K = -20 °C）。",
        "en": "Temperature [K] below which mixed-phase cloud condensate is 100% ice (default 253.15 K).",
    },
    "t_liq_all": {
        "ja": "混合相雲が 100% 過冷却水滴となる上限温度 [K]（既定 273.16 K）。",
        "en": "Temperature [K] above which cloud condensate is 100% liquid (default 273.16 K).",
    },
    "stefan_boltzmann": {
        "ja": "シュテファン・ボルツマン定数 σ [W m⁻² K⁻⁴]（5.670374419×10⁻⁸）。黒体長波放射 σT⁴ に使用。",
        "en": "Stefan–Boltzmann constant σ [W m⁻² K⁻⁴] (5.670374419e-8) for thermal blackbody emission.",
    },
    "von_karman": {
        "ja": "カルマン定数 κ [-]（0.40）。接地境界層および粗度サブレイヤ相似則に使用。",
        "en": "von Kármán constant κ [-] (0.40) in surface-layer and RSL similarity theory.",
    },
    "rho_seawater": {
        "ja": "海水の基準密度 ρ_sw [kg m⁻³]（1025.0）。海洋熱容量・応力変換に使用。",
        "en": "Reference seawater density ρ_sw [kg m⁻³] (1025.0).",
    },
    "cp_seawater": {
        "ja": "海水の定圧比熱 c_p,sw [J kg⁻¹ K⁻¹]（3990.0）。海洋貯熱量計算に使用。",
        "en": "Specific heat capacity of seawater c_p,sw [J kg⁻¹ K⁻¹] (3990.0).",
    },
    "t_freeze_ocean": {
        "ja": "塩分 ~35 psu 海水の凝固点温度 [K]（271.35 K = -1.8 °C）。海氷結氷・融解基準。",
        "en": "Freezing point of saline ocean water [K] (271.35 K = -1.8 °C) governing sea-ice formation.",
    },
    "rho_seaice": {
        "ja": "海氷密度 ρ_ice [kg m⁻³]（917.0）。海氷質量から氷厚への換算に使用。",
        "en": "Sea-ice bulk density ρ_ice [kg m⁻³] (917.0).",
    },
    "k_therm_seaice": {
        "ja": "海氷の熱伝導率 k_ice [W m⁻¹ K⁻¹]（2.034）。海氷を通じた海洋から大気への伝導熱流を支配。",
        "en": "Thermal conductivity of sea ice [W m⁻¹ K⁻¹] (2.034) governing conductive heat flux through ice.",
    },
    "k_therm_snow": {
        "ja": "積雪の熱伝導率 k_snow [W m⁻¹ K⁻¹]（0.3096）。海氷・陸地上の断熱効果を決定。",
        "en": "Thermal conductivity of snowpack [W m⁻¹ K⁻¹] (0.3096) controlling surface insulation.",
    },
    "o2_mixing_ratio": {
        "ja": "大気中の酸素モル分率 [mol mol⁻¹]（現代地球 0.20946）。光呼吸およびオゾン光化学に使用。",
        "en": "Atmospheric O₂ mole fraction [mol mol⁻¹] (Earth 0.20946) used in photorespiration and ozone chemistry.",
    },

    # =========================================================================
    # [radiation] (7)
    # =========================================================================
    "solar_frac_sw1": {
        "ja": "全太陽放射のうち可視・紫外域 (SW1, <0.7 μm) が占める割合 [-]（既定 0.45976）。",
        "en": "Fraction [-] of total solar irradiance in the UV+VIS band (SW1, <0.7 μm; default 0.45976).",
    },
    "lw_window_frac": {
        "ja": "黒体長波放射のうち大気窓領域 (8–12 μm) に配分する割合 [-]（4バンド放射用）。",
        "en": "Fraction [-] of blackbody LW flux assigned to the 8–12 μm atmospheric window (4-band scheme).",
    },
    "o3_g1": {
        "ja": "RCEMIP 解析的オゾン鉛直プロファイル係数 g₁ [ppmv]（ピーク濃度スケール）。",
        "en": "RCEMIP analytic ozone profile peak scale parameter g₁ [ppmv].",
    },
    "o3_g2": {
        "ja": "解析的オゾン鉛直プロファイルの上空減衰指数 g₂ [-]（高空外挿のべき乗則にも使用）。",
        "en": "Analytic ozone profile upper-stratospheric pressure exponent g₂ [-].",
    },
    "o3_g3": {
        "ja": "解析的オゾン鉛直プロファイルのピーク気圧係数 g₃ [hPa]（成層圏ピーク高度を制御）。",
        "en": "Analytic ozone profile peak-pressure coefficient g₃ [hPa].",
    },
    "par_frac_of_sw1": {
        "ja": "SW1（可視＋紫外）のうち光合成有効放射 PAR (400–700 nm) が占める割合 [-]（既定 0.88）。",
        "en": "Fraction [-] of SW1 irradiance falling in photosynthetically active radiation (PAR, 400–700 nm).",
    },
    "umol_per_joule_par": {
        "ja": "PAR エネルギーから光合成光量子束密度 PPFD への換算係数 [μmol photons J⁻¹]（既定 4.57）。",
        "en": "Conversion factor from PAR energy to photon flux density PPFD [μmol photons J⁻¹] (default 4.57).",
    },

    # =========================================================================
    # [microphysics] (32)
    # =========================================================================
    "rho_rain": {"ja": "雨滴の粒子密度 ρ_r [kg m⁻³]（1000.0）。", "en": "Bulk density of rain drops ρ_r [kg m⁻³] (1000.0)."},
    "rho_snow": {"ja": "雪片のかさ密度 ρ_s [kg m⁻³]（100.0）。", "en": "Bulk density of snow aggregates ρ_s [kg m⁻³] (100.0)."},
    "rho_graupel": {"ja": "霰（あられ）の粒子密度 ρ_g [kg m⁻³]（400.0）。", "en": "Bulk density of rimed graupel ρ_g [kg m⁻³] (400.0)."},
    "n0_rain": {"ja": "Marshall–Palmer 雨滴粒径分布の切片パラメータ N₀,r [m⁻⁴]（8×10⁶）。", "en": "Rain exponential size-distribution intercept N₀,r [m⁻⁴] (8e6)."},
    "n0_snow": {"ja": "雪粒径分布の切片パラメータ N₀,s [m⁻⁴]（3×10⁶）。", "en": "Snow exponential size-distribution intercept N₀,s [m⁻⁴] (3e6)."},
    "n0_graupel": {"ja": "霰粒径分布の切片パラメータ N₀,g [m⁻⁴]（4×10⁶）。", "en": "Graupel exponential size-distribution intercept N₀,g [m⁻⁴] (4e6)."},
    "a_fall_rain": {"ja": "雨滴終端落下速度則 v_r = a_r D^{b_r} の係数 a_r [m^{1-b} s⁻¹]（842.0）。", "en": "Rain terminal fall-speed prefactor a_r in v = a D^b (842.0)."},
    "b_fall_rain": {"ja": "雨滴終端落下速度則のべき指数 b_r [-]（0.80）。", "en": "Rain terminal fall-speed exponent b_r [-] (0.80)."},
    "a_fall_snow": {"ja": "雪片終端落下速度則の係数 a_s [m^{1-b} s⁻¹]（4.84）。", "en": "Snow terminal fall-speed prefactor a_s (4.84)."},
    "b_fall_snow": {"ja": "雪片終端落下速度則のべき指数 b_s [-]（0.25）。", "en": "Snow terminal fall-speed exponent b_s [-] (0.25)."},
    "a_fall_graupel": {"ja": "霰終端落下速度則の係数 a_g [m^{1-b} s⁻¹]（94.5）。", "en": "Graupel terminal fall-speed prefactor a_g (94.5)."},
    "b_fall_graupel": {"ja": "霰終端落下速度則のべき指数 b_g [-]（0.50）。", "en": "Graupel terminal fall-speed exponent b_g [-] (0.50)."},
    "e_rc": {"ja": "雨滴による雲水捕捉効率 E_rc [-]（1.0）。", "en": "Collection efficiency E_rc [-] of cloud liquid by rain (1.0)."},
    "e_sc": {"ja": "雪片による雲水捕捉（ライミング）効率 E_sc [-]（1.0）。", "en": "Riming collection efficiency E_sc [-] of cloud liquid by snow (1.0)."},
    "e_gc": {"ja": "霰による雲水捕捉効率 E_gc [-]（1.0）。", "en": "Collection efficiency E_gc [-] of cloud liquid by graupel (1.0)."},
    "e_ri": {"ja": "雨滴による雲氷捕捉効率 E_ri [-]（1.0）。", "en": "Collection efficiency E_ri [-] of cloud ice by rain (1.0)."},
    "e_si": {"ja": "雪片による雲氷凝集捕捉効率 E_si [-]（0.1）。", "en": "Aggregation collection efficiency E_si [-] of cloud ice by snow (0.1)."},
    "e_gi": {"ja": "霰による雲氷捕捉効率 E_gi [-]（0.1）。", "en": "Collection efficiency E_gi [-] of cloud ice by graupel (0.1)."},
    "qc_autoconv_thresh": {
        "ja": "1M スキームで雲水が雨へ自動変換を開始する閾値混合比 q_c0 [kg kg⁻¹]（1×10⁻³）。",
        "en": "Cloud liquid mixing-ratio threshold q_c0 [kg kg⁻¹] for 1M autoconversion (1e-3).",
    },
    "qi_aggreg_thresh": {
        "ja": "雲氷が雪へ凝集変換を開始する閾値混合比 q_i0 [kg kg⁻¹]（1×10⁻⁴）。",
        "en": "Cloud ice mixing-ratio threshold q_i0 [kg kg⁻¹] for ice-to-snow aggregation (1e-4).",
    },
    "rate_autoconv": {
        "ja": "1M スキームの雲水自動変換時定数逆数 α_auto [s⁻¹]（1×10⁻³）。",
        "en": "1M cloud-to-rain autoconversion rate constant [s⁻¹] (1e-3).",
    },
    "rate_ice_aggreg": {
        "ja": "雲氷から雪への凝集変換速度定数 [s⁻¹]（1×10⁻³）。",
        "en": "Cloud-ice to snow aggregation rate constant [s⁻¹] (1e-3).",
    },
    "vt_cloud_ice": {
        "ja": "雲氷（巻雲）の代表沈降速度 v_ti [m s⁻¹]（既定 0.75; Heymsfield & Iaquinta, 2000）。上層雲寿命と OLR を大きく左右する。",
        "en": "Characteristic cloud-ice sedimentation speed [m s⁻¹] (0.75; Heymsfield & Iaquinta, 2000); strongly affects cirrus lifetime and OLR.",
    },
    "cloud_droplet_conc_cm3": {
        "ja": "1M/参照雲粒数濃度 N_c [cm⁻³]（既定 60）。2M ではエアロゾルから動的予報される。",
        "en": "Reference cloud droplet number concentration N_c [cm⁻³] (60; prognostic in 2M schemes).",
    },
    "ccn_ocean_base_cm3": {
        "ja": "清浄な海洋大気の背景雲凝結核 (CCN) 数濃度 [cm⁻³]（既定 30; Andreae, 2009）。",
        "en": "Background maritime cloud condensation nuclei (CCN) concentration [cm⁻³] (default 30; Andreae, 2009).",
    },
    "ccn_land_base_cm3": {
        "ja": "清浄な陸域大気の背景 CCN 数濃度 [cm⁻³]（既定 95）。",
        "en": "Background continental CCN concentration [cm⁻³] (default 95).",
    },
    "ccn_dust_scale_cm3": {
        "ja": "鉱物ダスト質量から追加活性化される CCN 数濃度スケール [cm⁻³]（既定 120）。",
        "en": "Additional CCN activation scale [cm⁻³] from prognostic mineral dust loading (default 120).",
    },
    "inp_dust_enhance": {
        "ja": "鉱物ダスト存在下での氷晶核 (INP) 増強倍率 [-]（既定 4.0）。混合相雲の凍結率を高める。",
        "en": "Ice-nucleating particle (INP) enhancement multiplier [-] due to mineral dust (default 4.0).",
    },
    "t0_precip": {"ja": "1M/2M 降水種診断で雨 100% となる上限温度 [K]（283.16 K）。", "en": "Upper temperature [K] for 100% rain partitioning in diagnostic precipitation (283.16 K)."},
    "t00_precip": {"ja": "1M/2M 降水種診断で固体降水（雪・霰）100% となる下限温度 [K]（268.16 K）。", "en": "Lower temperature [K] for 100% frozen precipitation partitioning (268.16 K)."},
    "t0_graupel": {"ja": "固体降水中の霰比率が最大となる上限温度 [K]（283.16 K）。", "en": "Upper temperature [K] controlling graupel fraction in diagnostic frozen precipitation (283.16 K)."},
    "t00_graupel": {"ja": "固体降水がすべて雪となる下限温度 [K]（223.16 K）。", "en": "Lower temperature [K] below which diagnostic frozen precipitation is pure snow (223.16 K)."},

    # =========================================================================
    # [turbulence] (13)
    # =========================================================================
    "mynn_b1": {"ja": "MYNN Level 2.5 乱流運動エネルギー (TKE) 散逸定数 B₁ [-]（24.0）。", "en": "MYNN Level 2.5 TKE dissipation closure constant B₁ [-] (24.0)."},
    "mynn_b2": {"ja": "MYNN Level 2.5 温位分散散逸定数 B₂ [-]（15.0）。", "en": "MYNN Level 2.5 temperature-variance dissipation constant B₂ [-] (15.0)."},
    "mynn_c2": {"ja": "MYNN 圧力歪み相関定数 C₂ [-]（0.75）。", "en": "MYNN pressure-strain correlation constant C₂ [-] (0.75)."},
    "mynn_c3": {"ja": "MYNN 浮力圧力相関定数 C₃ [-]（0.352）。", "en": "MYNN buoyancy-pressure correlation constant C₃ [-] (0.352)."},
    "mynn_c5": {"ja": "MYNN 速度剪断圧力相関定数 C₅ [-]（0.20）。", "en": "MYNN shear-pressure correlation constant C₅ [-] (0.20)."},
    "mynn_gamma1": {"ja": "MYNN 等方化定数 γ₁ [-]（0.235）。", "en": "MYNN return-to-isotropy constant γ₁ [-] (0.235)."},
    "mynn_pr_n": {"ja": "中立成層時の乱流プラントル数 Pr_n = K_m/K_h [-]（0.74）。", "en": "Neutral turbulent Prandtl number Pr_n = K_m/K_h [-] (0.74)."},
    "tke_min": {"ja": "数値安定性のための TKE 下限値 [m² s⁻²]（1×10⁻⁴）。", "en": "Minimum floor value for turbulent kinetic energy TKE [m² s⁻²] (1e-4)."},
    "rsl_mu_m": {"ja": "De Ridder (2010) 粗度サブレイヤ (RSL) 運動量補正パラメータ μ_m [-]（2.59）。", "en": "De Ridder (2010) Roughness Sublayer (RSL) momentum parameter μ_m [-] (2.59)."},
    "rsl_mu_h": {"ja": "De Ridder (2010) RSL 熱・水蒸気補正パラメータ μ_h [-]（0.95）。", "en": "De Ridder (2010) RSL scalar (heat/moisture) parameter μ_h [-] (0.95)."},
    "rsl_nu": {"ja": "De Ridder (2010) RSL 減衰スケール定数 ν [-]（0.50）。", "en": "De Ridder (2010) RSL decay-scale constant ν [-] (0.50)."},
    "rsl_lambda": {"ja": "De Ridder (2010) RSL 長さ比定数 λ [-]（1.50）。", "en": "De Ridder (2010) RSL length-scale ratio λ [-] (1.50)."},
    "kinematic_visc_air": {"ja": "空気の分子動粘性係数 ν_air [m² s⁻¹]（1.5×10⁻⁵）。滑面海面粗度計算に使用。", "en": "Molecular kinematic viscosity of air [m² s⁻¹] (1.5e-5) for aerodynamically smooth ocean drag."},

    # =========================================================================
    # [surface] (24)
    # =========================================================================
    "cs_dry_soil": {"ja": "乾燥土壌の体積熱容量 [J m⁻³ K⁻¹]（1.26×10⁶）。", "en": "Volumetric heat capacity of dry mineral soil [J m⁻³ K⁻¹] (1.26e6)."},
    "kappa_dry_soil": {"ja": "乾燥土壌の熱伝導率 [W m⁻¹ K⁻¹]（0.30）。", "en": "Thermal conductivity of dry soil [W m⁻¹ K⁻¹] (0.30)."},
    "soil_porosity": {"ja": "土壌空隙率（飽和体積含水率）θ_sat [m³ m⁻³]（0.43）。", "en": "Soil porosity (saturated volumetric water content) θ_sat [m³ m⁻³] (0.43)."},
    "psi_sat_soil": {"ja": "飽和土壌マトリックポテンシャル ψ_sat [m]（-0.20 m）。", "en": "Saturated soil matric potential ψ_sat [m] (-0.20 m)."},
    "b_clapp_soil": {"ja": "Clapp–Hornberger 土壌水分特性曲線のべき指数 b [-]（5.39）。", "en": "Clapp–Hornberger soil water retention exponent b [-] (5.39)."},
    "k_sat_soil": {"ja": "飽和透水係数 K_sat [m s⁻¹]（7×10⁻⁶）。土壌浸透と地下流出速度を支配。", "en": "Saturated hydraulic conductivity K_sat [m s⁻¹] (7e-6) controlling infiltration and drainage."},
    "ice_impedance_omega": {"ja": "凍土氷含有率による透水阻害指数 Ω [-]（6.0）。永久凍土の浸透抑制を表現。", "en": "Frozen-soil hydraulic impedance exponent Ω [-] (6.0) reducing drainage in permafrost."},
    "wc_max_per_lai": {"ja": "単位 LAI あたりの樹冠遮断貯留容量 [kg m⁻² LAI⁻¹]（0.10 mm/LAI）。", "en": "Canopy rainfall interception storage capacity per unit LAI [kg m⁻² LAI⁻¹] (0.10)."},
    "alpha_intercept_liq": {"ja": "降水が樹冠に遮断される効率係数 [-]（1.0）。", "en": "Canopy rainfall interception efficiency factor [-] (1.0)."},
    "f_wet_max": {"ja": "樹冠濡れ面積率の上限 [-]（1.0）。", "en": "Maximum wetted canopy fraction [-] (1.0)."},
    "dsl_dmax_mm": {"ja": "乾燥表土層 (DSL) の最大厚 [mm]（15.0 mm）。裸地蒸発の乾燥抵抗を制御。", "en": "Maximum dry surface layer (DSL) thickness [mm] (15.0 mm) controlling bare-soil evaporation resistance."},
    "land_eb_iterations": {"ja": "3 源（陽葉・陰葉・地表）連成エネルギー収支の陰的反復回数 [-]（既定 3）。", "en": "Number of coupled implicit iterations [-] for the 3-source sunlit/shaded/ground energy balance (3)."},
    "rho_snowpack": {"ja": "積雪の代表かさ密度 [kg m⁻³]（250.0）。積雪深と遮蔽率の換算に使用。", "en": "Bulk density of terrestrial snowpack [kg m⁻³] (250.0)."},
    "swe_glacier_max": {"ja": "陸面積雪水当量 (SWE) の上限キャップ [kg m⁻²]（1000.0）。超過分は氷河流出または氷床 SMB へ送る。", "en": "Maximum land snow water equivalent (SWE) cap [kg m⁻²] (1000.0); excess feeds glacier runoff or ice-sheet SMB."},
    "tau_snow_age": {"ja": "積雪アルベド経年低下（エージング）の時定数 [s]（1×10⁶ s ≈ 11.6 日）。", "en": "Snow albedo aging relaxation timescale [s] (1e6 s ≈ 11.6 days)."},
    "alb_snow_vis_new": {"ja": "新雪の可視域アルベド [-]（0.90）。", "en": "Fresh snow visible-band albedo [-] (0.90)."},
    "alb_snow_nir_new": {"ja": "新雪の近赤外域アルベド [-]（0.70）。", "en": "Fresh snow near-infrared albedo [-] (0.70)."},
    "alb_snow_vis_old": {"ja": "経年雪・融雪期雪の可視域アルベド下限 [-]（0.65）。", "en": "Aged/melting snow visible-band albedo floor [-] (0.65)."},
    "alb_snow_nir_old": {"ja": "経年雪の近赤外域アルベド下限 [-]（0.35）。", "en": "Aged snow near-infrared albedo floor [-] (0.35)."},
    "emissivity_land": {"ja": "陸面・植生・積雪の熱赤外射出率 ε_land [-]（0.98）。", "en": "Longwave thermal emissivity of land surfaces [-] (0.98)."},
    "emissivity_ocean": {"ja": "開水面の熱赤外射出率 ε_ocn [-]（0.985）。", "en": "Longwave thermal emissivity of open ocean [-] (0.985)."},
    "mc_seaice_crit": {"ja": "海氷被覆率が飽和へ近づく臨界海氷質量 [kg m⁻²]（450.0 ≈ 氷厚 0.5 m）。", "en": "Characteristic sea-ice mass [kg m⁻²] (450.0 ≈ 0.5 m) for subgrid sea-ice fraction/lead closure."},
    "alb_seaice_vis": {"ja": "裸海氷の可視域アルベド [-]（0.72）。", "en": "Bare sea-ice visible-band albedo [-] (0.72)."},
    "alb_seaice_nir": {"ja": "裸海氷の近赤外域アルベド [-]（0.42）。", "en": "Bare sea-ice near-infrared albedo [-] (0.42)."},

    # =========================================================================
    # [biosphere] (41)
    # =========================================================================
    "theta_nrh": {"ja": "非直角双曲線光合成曲線の曲率パラメータ θ [-]（0.85）。Rubisco 律速と電子伝達律速の滑らかな遷移を表す。", "en": "Non-rectangular hyperbola curvature θ [-] (0.85) blending Rubisco- and electron-transport-limited photosynthesis."},
    "cost_c_smith": {"ja": "Smith et al. (2019) 電子伝達維持コスト係数 c* [-]（0.05336251）。最適 Jmax:Vcmax 比を決定。", "en": "Smith et al. (2019) unit electron-transport maintenance cost c* [-] (0.05336251)."},
    "phi_psii_ref_c3": {"ja": "C₃ 光合成の基準光化学系 II 量子収率 φ₀ [-]（0.257）。", "en": "Reference C₃ photosystem-II quantum yield φ₀ [-] (0.257)."},
    "beta_c3": {"ja": "C₃ 植物の蒸散コスト対カルボキシル化コスト比 β_C3 [-]（146.0）。最適気孔比 χ を決定。", "en": "C₃ ratio of transpiration-to-carboxylation unit costs β_C3 [-] (146.0) setting optimal χ = ci/ca."},
    "beta_c4": {"ja": "C₄ 植物のコスト比 β_C4 [-]（166.0）。C₄ の高い水利用効率を支配。", "en": "C₄ cost ratio β_C4 [-] (166.0) governing higher C₄ water-use efficiency."},
    "gamma_star25_c3": {"ja": "25 °C における C₃ 光呼吸 CO₂ 補償点 Γ*₂₅ [Pa]（4.332 Pa）。", "en": "C₃ photorespiratory CO₂ compensation point at 25 °C Γ*₂₅ [Pa] (4.332 Pa)."},
    "kc25_c3": {"ja": "25 °C における C₃ Rubisco の CO₂ ミカエリス定数 K_c25 [Pa]（41.03 Pa）。", "en": "C₃ Rubisco Michaelis–Menten constant for CO₂ at 25 °C K_c25 [Pa] (41.03 Pa)."},
    "ko25_c3": {"ja": "25 °C における C₃ Rubisco の O₂ 競合阻害定数 K_o25 [Pa]（28210 Pa）。", "en": "C₃ Rubisco Michaelis–Menten constant for O₂ at 25 °C K_o25 [Pa] (28210 Pa)."},
    "dha_gamma": {"ja": "Γ* のアレニウス活性化エネルギー ΔH_a [J mol⁻¹]（37830）。", "en": "Arrhenius activation energy ΔH_a [J mol⁻¹] for Γ* (37830)."},
    "dha_kc_c3": {"ja": "C₃ K_c の活性化エネルギー [J mol⁻¹]（79430）。", "en": "Arrhenius activation energy [J mol⁻¹] for C₃ K_c (79430)."},
    "dha_ko_c3": {"ja": "C₃ K_o の活性化エネルギー [J mol⁻¹]（36380）。", "en": "Arrhenius activation energy [J mol⁻¹] for C₃ K_o (36380)."},
    "gamma_star25_c4": {"ja": "25 °C における C₄ 維管束鞘内の CO₂ 補償点 [Pa]（2.6 Pa）。", "en": "C₄ bundle-sheath CO₂ compensation point at 25 °C [Pa] (2.6 Pa)."},
    "kp25_umol_c4": {"ja": "25 °C における C₄ PEP カルボキシラーゼのミカエリス定数 K_p25 [Pa]（60.5）。", "en": "C₄ PEP carboxylase Michaelis constant at 25 °C K_p25 (60.5)."},
    "dha_kp_c4": {"ja": "C₄ K_p の活性化エネルギー [J mol⁻¹]（27200）。", "en": "Arrhenius activation energy [J mol⁻¹] for C₄ K_p (27200)."},
    "kc25_c4": {"ja": "25 °C における C₄ Rubisco K_c25 [Pa]（121.0）。", "en": "C₄ Rubisco K_c25 at 25 °C [Pa] (121.0)."},
    "dha_kc_c4": {"ja": "C₄ K_c の活性化エネルギー [J mol⁻¹]（64200）。", "en": "Arrhenius activation energy [J mol⁻¹] for C₄ K_c (64200)."},
    "ko25_c4": {"ja": "25 °C における C₄ Rubisco K_o25 [Pa]（29200）。", "en": "C₄ Rubisco K_o25 at 25 °C [Pa] (29200)."},
    "dha_ko_c4": {"ja": "C₄ K_o の活性化エネルギー [J mol⁻¹]（10500）。", "en": "Arrhenius activation energy [J mol⁻¹] for C₄ K_o (10500)."},
    "phi_leak_c4": {"ja": "C₄ 維管束鞘からの CO₂ 漏出率 φ_leak [-]（0.20）。", "en": "Fraction [-] of CO₂ leaking back from C₄ bundle-sheath cells (0.20)."},
    "gbs_c4": {"ja": "C₄ 維管束鞘の CO₂ コンダクタンス g_bs [mol m⁻² s⁻¹]（0.003）。", "en": "C₄ bundle-sheath CO₂ conductance g_bs [mol m⁻² s⁻¹] (0.003)."},
    "ha_vcmax": {"ja": "最大カルボキシル化速度 V_cmax の活性化エネルギー [J mol⁻¹]（71513）。", "en": "Activation energy [J mol⁻¹] for Rubisco capacity V_cmax (71513)."},
    "ha_jmax": {"ja": "最大電子伝達速度 J_max の活性化エネルギー [J mol⁻¹]（40710）。", "en": "Activation energy [J mol⁻¹] for electron transport capacity J_max (40710)."},
    "ha_vpmax": {"ja": "C₄ PEP カルボキシラーゼ最大速度 V_pmax の活性化エネルギー [J mol⁻¹]（55000）。", "en": "Activation energy [J mol⁻¹] for C₄ PEP carboxylase capacity V_pmax (55000)."},
    "hd_enzyme": {"ja": "高温での光合成酵素失活エネルギー H_d [J mol⁻¹]（200000）。熱ストレス上限を規定。", "en": "High-temperature enzyme deactivation energy H_d [J mol⁻¹] (200000)."},
    "b_resp_ren": {"ja": "Ren et al. 普遍的葉暗呼吸比 R_d25 / V_cmax25 [-]（0.018）。", "en": "Universal leaf dark respiration ratio R_d25 / V_cmax25 [-] (0.018)."},
    "leaf_scat_vis": {"ja": "個葉の可視域 (PAR) 散乱係数 ω_vis [-]（0.15）。樹冠 2 ストリームアルベドに使用。", "en": "Single-leaf visible (PAR) scattering coefficient ω_vis [-] (0.15)."},
    "leaf_scat_nir": {"ja": "個葉の近赤外域 (NIR) 散乱係数 ω_nir [-]（0.80）。植生の高い近赤外反射を生む。", "en": "Single-leaf near-infrared scattering coefficient ω_nir [-] (0.80)."},
    "kd_prime_par": {"ja": "散乱を考慮した散乱光 PAR の樹冠消散係数 k'_d [-]（0.719）。", "en": "Diffuse PAR canopy extinction coefficient k'_d [-] including leaf scattering (0.719)."},
    "rho_cd_par": {"ja": "厚い樹冠での散乱光 PAR 反射率 ρ_cd [-]（0.036）。", "en": "Deep-canopy diffuse PAR reflectance ρ_cd [-] (0.036)."},
    "z_foliage_cost": {
        "ja": "単位 LAI 建設・維持コスト z [mol C m⁻² yr⁻¹]（12.227）。Lambert-W₀ 解による最適 LAI の上限を決定。",
        "en": "Unit foliage carbon construction/maintenance cost z [mol C m⁻² yr⁻¹] (12.227) setting optimal LAI.",
    },
    "k_beer_lai": {"ja": "Beer–Lambert 法則の標準樹冠光消散係数 k [-]（0.50、球面葉群分布）。", "en": "Beer–Lambert canopy light extinction coefficient k [-] (0.50, spherical leaf angle)."},
    "sigma_zhou": {"ja": "Zhou et al. 最適 LAI における水制限指数 σ [-]（0.771）。乾燥域の植生限界を決定。", "en": "Zhou et al. moisture-limitation scaling exponent σ [-] (0.771) governing arid-zone LAI."},
    "alpha_ema_acclim": {
        "ja": "光合成順化（Vcmax25/Jmax25 最適再調整）の日次 EMA 係数 [-]（0.067 ≈ 15 日時定数）。",
        "en": "Daily EMA acclimation rate [-] for photosynthetic enzyme adjustment (0.067 ≈ 15-day memory).",
    },
    "evg_cost_ratio_base": {"ja": "温暖気候での常緑葉対落葉葉の相対建設コスト比 [-]（1.35）。", "en": "Warm-climate evergreen-to-deciduous leaf construction cost ratio [-] (1.35)."},
    "evg_cost_ratio_cold": {"ja": "寒冷気候での針葉樹（常緑）コスト優位係数 [-]（0.55）。北方針葉樹林の自律出現に寄与。", "en": "Cold-climate evergreen needleleaf cost advantage factor [-] (0.55) enabling emergent boreal forests."},
    "evg_cold_eff_penalty": {"ja": "低温下での常緑葉光合成効率ペナルティ係数 [-]（0.14）。", "en": "Cold-season photosynthetic efficiency penalty [-] for evergreen foliage (0.14)."},
    "leaf_aging_b_yr": {"ja": "葉齢進行に伴う光合成能力低下スケール [yr]（15.0）。", "en": "Leaf-aging photosynthetic decline timescale [yr] (15.0)."},
    "decid_flush_days": {"ja": "落葉樹が春に展葉・展開完了するまでの日数 [日]（25.0 日）。", "en": "Spring leaf-flush duration [days] for deciduous phenology (25.0 days)."},
    "evg_offseason_resp25": {"ja": "非成長期（冬・乾季）における常緑葉の維持呼吸コスト基準値 [g C m⁻² yr⁻¹]（32.0）。", "en": "Off-season baseline maintenance respiration cost [g C m⁻² yr⁻¹] for evergreen foliage (32.0)."},
    "evg_frost_crit_k": {"ja": "常緑樹が深刻な凍害コストを受ける臨界最低気温 [K]（233.15 K = -40 °C）。シベリア落葉針葉樹帯の境界。", "en": "Critical extreme-frost threshold [K] (233.15 K = -40 °C) favoring cold-deciduous taiga over evergreens."},
    "evg_frost_cost": {"ja": "極寒凍害ペナルティの強度係数 [-]（18.0）。", "en": "Strength coefficient [-] of extreme-frost damage cost on evergreen foliage (18.0)."},

    # =========================================================================
    # [ocean_ekman] (18)
    # =========================================================================
    "h_thermocline": {"ja": "1.5層浅水海洋の基準躍層厚 H_th [m]（200.0 m）。傾圧ロスビー波速 c = √(g'H) を決定。", "en": "Reference thermocline thickness H_th [m] (200.0 m) for 1.5-layer shallow-water ocean mode."},
    "r_ekman_damp_days": {"ja": "エクマン流速のレイリー摩擦減衰時定数 [日]（2.0 日）。赤道発散防止に必須。", "en": "Rayleigh friction damping timescale [days] (2.0 days) regulating equatorial Ekman velocity."},
    "kappa_ocn_horiz": {"ja": "1.5層海洋の水平渦拡散係数 κ_h [m² s⁻¹]（2000.0）。", "en": "Horizontal eddy diffusivity κ_h [m² s⁻¹] (2000.0) in the 1.5-layer ocean."},
    "kappa_ocn_vert": {"ja": "1.5層海洋の鉛直湧昇・混合拡散係数 κ_v [m² s⁻¹]（2×10⁻⁵）。", "en": "Vertical diffusivity κ_v [m² s⁻¹] (2e-5) in the 1.5-layer ocean."},
    "g_prime_ocn": {"ja": "1.5層浅水海洋の還元重力 g' [m s⁻²]（0.03）。エルニーニョ型赤道波の位相速度を決定。", "en": "Reduced gravity g' [m s⁻²] (0.03) governing baroclinic Kelvin/Rossby wave speeds."},
    "tau_sw_drag_days": {"ja": "1.5層浅水海洋の地衡流運動量減衰時定数 [日]（180.0 日）。", "en": "Linear drag timescale [days] (180.0 days) damping geostrophic currents in 1.5-layer ocean."},
    "enable_shallow_water_ocn": {"ja": "1.5層エクマン海洋で浅水波動力学（傾圧ロスビー・ケルビン波）を有効化する。", "en": "Enable 1.5-layer shallow-water baroclinic wave dynamics in 'ekman_sw' ocean mode."},
    "kappa_gm": {"ja": "3D OGCM の Gent–McWilliams (1990) 等密度面平坦化渦輸送係数 κ_GM [m² s⁻¹]（既定 800）。", "en": "Gent–McWilliams (1990) isopycnal thickness diffusivity κ_GM [m² s⁻¹] (default 800) in 3D OGCM."},
    "kappa_redi_horiz": {"ja": "3D OGCM の Redi (1982) 等密度面トレーサ拡散係数 κ_Redi [m² s⁻¹]（既定 1000）。", "en": "Redi (1982) along-isopycnal tracer diffusivity κ_Redi [m² s⁻¹] (default 1000) in 3D OGCM."},
    "enable_full_redi": {"ja": "完全な等密度面回転拡散テンソル (Redi) を有効化する（false で水平拡散近似）。", "en": "Enable full rotated isopycnal Redi diffusion tensor in 3D OGCM (false = horizontal fallback)."},
    "kpp_ri_crit": {"ja": "KPP 海洋境界層深さ (MLD) を決める臨界バルクリチャードソン数 Ri_c [-]（既定 0.30）。", "en": "Critical bulk Richardson number Ri_c [-] (0.30) determining KPP ocean mixed-layer depth."},
    "kpp_cs_nonlocal": {"ja": "KPP 不安定成層時の非局所対流輸送係数 c_s [-]（6.33; Large et al., 1994）。", "en": "KPP convective non-local transport coefficient c_s [-] (6.33; Large et al., 1994)."},
    "shear_nu_0": {"ja": "海洋内部シアー不安定混合の最大拡散係数 ν₀ [m² s⁻¹]（5×10⁻³）。赤道潜流下の混合を表現。", "en": "Maximum interior shear-instability diffusivity ν₀ [m² s⁻¹] (5e-3) beneath equatorial currents."},
    "tidal_e0_wm2": {"ja": "深海内部潮汐エネルギー散逸基準フラックス E₀ [W m⁻²]（8×10⁻³）。", "en": "Abyssal internal tidal energy dissipation flux scale E₀ [W m⁻²] (8e-3)."},
    "sw_pen_zeta2_m": {"ja": "海洋表層への短波透過 e-folding 深さ ζ₂ [m]（既定 23.0 m、Jerlov Type I 清浄海水）。", "en": "Shortwave solar penetration e-folding depth ζ₂ [m] (23.0 m, Jerlov Type I water)."},
    "kappa_v_bg": {"ja": "海洋上部〜中層の背景鉛直拡散係数 κ_v,bg [m² s⁻¹]（1.5×10⁻⁵）。", "en": "Background interior diapycnal diffusivity κ_v,bg [m² s⁻¹] (1.5e-5)."},
    "kappa_v_abyss": {"ja": "深海（>2000 m）の背景鉛直拡散係数 κ_v,abyss [m² s⁻¹]（3×10⁻⁵）。", "en": "Abyssal background diapycnal diffusivity κ_v,abyss [m² s⁻¹] (3e-5)."},
    "seaice_h_ridge_m": {"ja": "海氷圧縮時の氷丘（リジング）代表厚 [m]（3.0 m）。", "en": "Characteristic ridged sea-ice thickness [m] (3.0 m) limiting compressive ice piling."},

    # =========================================================================
    # [orography] (3)
    # =========================================================================
    "hn_crit": {"ja": "Lott & Miller (1997) 山岳ブロッキング（迂回流）が開始する臨界無次元山岳高度 H_nc [-]（0.50）。", "en": "Critical non-dimensional mountain height H_nc [-] (0.50) triggering Lott–Miller low-level blocking."},
    "cd_block": {"ja": "サブグリッド山岳低層ブロッキングの抗力係数 C_d [-]（1.0）。", "en": "Low-level subgrid orographic blocking drag coefficient C_d [-] (1.0)."},
    "g_wave": {"ja": "山岳重力波（山岳波）発生振幅係数 G [-]（0.80）。ジェット気流と極渦強度を調整。", "en": "Orographic gravity-wave launch amplitude factor G [-] (0.80)."},

    # =========================================================================
    # [gravity_wave] (11)
    # =========================================================================
    "ngwd_front_thresh": {"ja": "前線起源非地形性重力波を発射する前線生成関数閾値 [K² m⁻² s⁻¹]（1.25×10⁻¹⁵）。", "en": "Frontogenesis threshold [K² m⁻² s⁻¹] (1.25e-15) triggering frontal non-orographic GWD."},
    "ngwd_front_launch_hpa": {"ja": "前線重力波の発射高度気圧面 [hPa]（600 hPa）。", "en": "Launch pressure level [hPa] (600 hPa) for frontal non-orographic gravity waves."},
    "ngwd_front_tau0": {"ja": "前線重力波の基準運動量フラックス τ₀ [Pa]（1.5×10⁻³）。", "en": "Reference momentum flux τ₀ [Pa] (1.5e-3) for frontal gravity waves."},
    "ngwd_front_cw": {"ja": "前線重力波スペクトルの代表位相速度幅 c_w [m s⁻¹]（30.0）。", "en": "Characteristic phase-speed spectral width c_w [m s⁻¹] (30.0) for frontal GWD."},
    "ngwd_front_eff": {"ja": "前線重力波の砕波運動量付与効率 [-]（0.10）。", "en": "Efficiency factor [-] (0.10) scaling frontal GWD momentum deposition."},
    "ngwd_conv_tau0": {"ja": "対流起源重力波の基準運動量フラックス τ₀ [Pa]（2.0×10⁻³）。赤道 QBO 駆動に重要。", "en": "Reference momentum flux τ₀ [Pa] (2e-3) for convective gravity waves driving tropical QBO."},
    "ngwd_conv_qref": {"ja": "対流重力波強度を規格化する基準非断熱加熱率 Q_ref [K s⁻¹]（2×10⁻⁴）。", "en": "Reference convective heating scale Q_ref [K s⁻¹] (2e-4) normalizing convective GWD."},
    "ngwd_conv_eff": {"ja": "対流重力波の砕波運動量付与効率 [-]（0.25）。", "en": "Efficiency factor [-] (0.25) scaling convective GWD momentum deposition."},
    "ngwd_wavelength_m": {"ja": "非地形性重力波の代表水平波長 λ_h [m]（200 km）。", "en": "Representative horizontal wavelength λ_h [m] (200 km) of non-orographic gravity waves."},
    "ngwd_cmax": {"ja": "非地形性重力波スペクトルの最大位相速度 [m s⁻¹]（80.0）。", "en": "Maximum phase speed [m s⁻¹] (80.0) in the non-orographic GWD spectrum."},
    "ngwd_tend_max": {"ja": "成層圏上部での数値安定性用 GWD 風速変化率上限 [m s⁻²]（1.157×10⁻³ ≈ 100 m/s/day）。", "en": "Upper bound on GWD wind tendency [m s⁻²] (1.157e-3 ≈ 100 m s⁻¹ day⁻¹) for numerical stability."},

    # =========================================================================
    # [icesheet_physics] (5)
    # =========================================================================
    "rho_ice": {"ja": "大陸氷床氷の密度 ρ_ice [kg m⁻³]（917.0）。駆動応力とアイソスタシー比に使用。", "en": "Continental ice-sheet density ρ_ice [kg m⁻³] (917.0)."},
    "rho_mantle": {"ja": "上部マントル密度 ρ_m [kg m⁻³]（3300.0）。平衡沈降比 ρ_ice/ρ_m ≈ 0.278 を決定。", "en": "Upper mantle density ρ_m [kg m⁻³] (3300.0) setting equilibrium isostatic ratio ρ_ice/ρ_m."},
    "a_glen_per_year": {"ja": "Glen 流動則 (n=3) の氷軟化係数 A [Pa⁻³ yr⁻¹]（1×10⁻¹⁶）。大きいほど氷床が速く薄く広がる。", "en": "Glen flow-law (n=3) rate factor A [Pa⁻³ yr⁻¹] (1e-16) controlling ice-sheet fluidity."},
    "h_ice_threshold_m": {"ja": "陸地と氷床セルのマスク切替を行う臨界氷厚 [m]（5.0 m）。", "en": "Ice thickness threshold [m] (5.0 m) triggering dynamic Land <-> IceSheet mask transitions."},
    "h_ice_max_m": {"ja": "氷床厚の数値的上限キャップ [m]（6000.0 m）。", "en": "Numerical upper cap on continental ice-sheet thickness [m] (6000.0 m)."},
}
