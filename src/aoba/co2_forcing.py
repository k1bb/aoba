"""Time-varying atmospheric CO2 forcing tables, external fetchers, and interpolators for AOBA.

Supports three configuration modes in ``[forcing]``:
1. ``co2_mode = "constant"``: Fixed ``co2_ppmv`` value (default, backward-compatible).
2. ``co2_mode = "scenario"``: Named scenario from either the locally updated cache
   (``data/forcing/co2/co2_latest.csv``) or the repository-bundled baseline table
   (``src/aoba/data/co2_scenarios.csv``), linearly interpolated at the model's decimal
   calendar year.
3. ``co2_mode = "csv"``: User-supplied ``(year, co2_ppmv)`` CSV file.

External Sources Supported by ``fetch_latest_co2_sources()``:
- NOAA GML Global Marine Surface Annual Mean (``co2_annmean_gl.csv``, 1979-present)
- NOAA GML Mauna Loa Annual Mean (``co2_annmean_mlo.csv``, 1959-present)
- NCEI / Bereiter et al. (2015, Geophys. Res. Lett., 42, 546-553) Antarctic Ice Core
  800 kyr Composite (``antarctica2015co2composite.txt``, ~803,718 BCE - 1950 CE)
- RCMIP v5.1.0 (Nicholls et al., 2020, Geosci. Model Dev., 13, 5175-5190;
  Meinshausen et al., 2011, 2017, 2020) for CMIP5 RCPs and CMIP6 Historical/SSPs
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np


VALID_CO2_MODES: tuple[str, ...] = ("constant", "scenario", "csv")

NOAA_GML_GLOBAL_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.csv"
NOAA_GML_MLO_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"
NCEI_PALEO_800KA_URL = (
    "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/antarctica2015co2composite.txt"
)
RCMIP_CONC_V510_URL = (
    "https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/"
    "rcmip-concentrations-annual-means-v5-1-0.csv"
)


@dataclass(frozen=True)
class CO2ScenarioMeta:
    """Metadata describing a bundled or fetched CO2 concentration scenario."""

    key: str
    group: str
    label_en: str
    label_ja: str
    source: str


CO2_SCENARIO_CATALOG: tuple[CO2ScenarioMeta, ...] = (
    CO2ScenarioMeta(
        "obs_historical_noaa",
        "Recent Obs",
        "Historical + NOAA Global Marine Obs (1750-latest)",
        "歴史的強制 + NOAA 全球観測 (1750〜最新年)",
        "CMIP6 historical + NOAA GML co2_annmean_gl",
    ),
    CO2ScenarioMeta(
        "obs_mauna_loa",
        "Recent Obs",
        "Historical + NOAA Mauna Loa Obs (1750-latest)",
        "歴史的強制 + NOAA マウナロア観測 (1750〜最新年)",
        "CMIP6 historical + NOAA GML co2_annmean_mlo",
    ),
    CO2ScenarioMeta(
        "obs_to_ssp126",
        "Recent Obs + SSP",
        "NOAA Global Obs -> CMIP6 SSP1-2.6 (1750-2500)",
        "NOAA 最新観測 -> CMIP6 SSP1-2.6 (1750〜2500)",
        "NOAA GML + Meinshausen et al. (2020)",
    ),
    CO2ScenarioMeta(
        "obs_to_ssp245",
        "Recent Obs + SSP",
        "NOAA Global Obs -> CMIP6 SSP2-4.5 (1750-2500)",
        "NOAA 最新観測 -> CMIP6 SSP2-4.5 (1750〜2500)",
        "NOAA GML + Meinshausen et al. (2020)",
    ),
    CO2ScenarioMeta(
        "obs_to_ssp370",
        "Recent Obs + SSP",
        "NOAA Global Obs -> CMIP6 SSP3-7.0 (1750-2500)",
        "NOAA 最新観測 -> CMIP6 SSP3-7.0 (1750〜2500)",
        "NOAA GML + Meinshausen et al. (2020)",
    ),
    CO2ScenarioMeta(
        "obs_to_ssp585",
        "Recent Obs + SSP",
        "NOAA Global Obs -> CMIP6 SSP5-8.5 (1750-2500)",
        "NOAA 最新観測 -> CMIP6 SSP5-8.5 (1750〜2500)",
        "NOAA GML + Meinshausen et al. (2020)",
    ),
    CO2ScenarioMeta(
        "cmip6_historical",
        "CMIP6",
        "CMIP6 Historical (1750-2014)",
        "CMIP6 歴史実験 (1750〜2014)",
        "Meinshausen et al. (2017) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp119",
        "CMIP6",
        "CMIP6 SSP1-1.9 (1750-2500)",
        "CMIP6 SSP1-1.9 (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp126",
        "CMIP6",
        "CMIP6 SSP1-2.6 (1750-2500)",
        "CMIP6 SSP1-2.6 (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp245",
        "CMIP6",
        "CMIP6 SSP2-4.5 (1750-2500)",
        "CMIP6 SSP2-4.5 (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp370",
        "CMIP6",
        "CMIP6 SSP3-7.0 (1750-2500)",
        "CMIP6 SSP3-7.0 (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp434",
        "CMIP6",
        "CMIP6 SSP4-3.4 (1750-2500)",
        "CMIP6 SSP4-3.4 (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp460",
        "CMIP6",
        "CMIP6 SSP4-6.0 (1750-2500)",
        "CMIP6 SSP4-6.0 (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp534_over",
        "CMIP6",
        "CMIP6 SSP5-3.4-OS Overshoot (1750-2500)",
        "CMIP6 SSP5-3.4-OS オーバーシュート (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_ssp585",
        "CMIP6",
        "CMIP6 SSP5-8.5 (1750-2500)",
        "CMIP6 SSP5-8.5 (1750〜2500)",
        "Meinshausen et al. (2020) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip5_historical",
        "CMIP5",
        "CMIP5 Historical (1765-2005)",
        "CMIP5 歴史実験 (1765〜2005)",
        "Meinshausen et al. (2011) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip5_rcp26",
        "CMIP5",
        "CMIP5 RCP2.6 (1765-2500)",
        "CMIP5 RCP2.6 (1765〜2500)",
        "Meinshausen et al. (2011) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip5_rcp45",
        "CMIP5",
        "CMIP5 RCP4.5 (1765-2500)",
        "CMIP5 RCP4.5 (1765〜2500)",
        "Meinshausen et al. (2011) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip5_rcp60",
        "CMIP5",
        "CMIP5 RCP6.0 (1765-2500)",
        "CMIP5 RCP6.0 (1765〜2500)",
        "Meinshausen et al. (2011) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip5_rcp85",
        "CMIP5",
        "CMIP5 RCP8.5 (1765-2500)",
        "CMIP5 RCP8.5 (1765〜2500)",
        "Meinshausen et al. (2011) via RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "paleo_antarctica_800ka",
        "Paleoclimate",
        "Antarctic Ice Core 800 kyr Composite (803,718 BCE-1950 CE)",
        "南極氷床コア 80万年復元記録 (803,718 BCE〜1950 CE)",
        "Bereiter et al. (2015) / NCEI Paleoclimatology",
    ),
    CO2ScenarioMeta(
        "cmip6_picontrol",
        "Idealized",
        "CMIP6 piControl (284.32 ppmv)",
        "CMIP6 産業革命前定常 piControl (284.32 ppmv)",
        "Meinshausen et al. (2017)",
    ),
    CO2ScenarioMeta(
        "cmip6_1pctco2",
        "Idealized",
        "CMIP6 1pctCO2 (+1%/yr from 1850)",
        "CMIP6 1%/年漸増 1pctCO2 (1850〜)",
        "RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_abrupt_2xco2",
        "Idealized",
        "CMIP6 abrupt-2xCO2 (568.63 ppmv)",
        "CMIP6 倍増 abrupt-2xCO2 (568.63 ppmv)",
        "RCMIP v5.1.0",
    ),
    CO2ScenarioMeta(
        "cmip6_abrupt_4xco2",
        "Idealized",
        "CMIP6 abrupt-4xCO2 (1137.27 ppmv)",
        "CMIP6 4倍増 abrupt-4xCO2 (1137.27 ppmv)",
        "RCMIP v5.1.0",
    ),
)

_RCMIP_SCENARIO_MAP: dict[str, str] = {
    "historical": "cmip6_historical",
    "ssp119": "cmip6_ssp119",
    "ssp126": "cmip6_ssp126",
    "ssp245": "cmip6_ssp245",
    "ssp370": "cmip6_ssp370",
    "ssp434": "cmip6_ssp434",
    "ssp460": "cmip6_ssp460",
    "ssp534-over": "cmip6_ssp534_over",
    "ssp585": "cmip6_ssp585",
    "historical-cmip5": "cmip5_historical",
    "rcp26": "cmip5_rcp26",
    "rcp45": "cmip5_rcp45",
    "rcp60": "cmip5_rcp60",
    "rcp85": "cmip5_rcp85",
    "piControl": "cmip6_picontrol",
    "1pctCO2": "cmip6_1pctco2",
    "abrupt-2xCO2": "cmip6_abrupt_2xco2",
    "abrupt-4xCO2": "cmip6_abrupt_4xco2",
}


def bundled_co2_table_path() -> Path:
    """Return path to the repository-bundled CO2 scenario CSV table."""
    return Path(__file__).resolve().parent / "assets" / "co2_scenarios.csv"


def local_co2_cache_dir() -> Path:
    """Return path to the local (gitignored) updated CO2 forcing directory ``data/forcing/co2``."""
    return Path(__file__).resolve().parents[2] / "data" / "forcing" / "co2"


def local_co2_cache_path() -> Path:
    """Return path to the locally updated CO2 scenario CSV cache."""
    return local_co2_cache_dir() / "co2_latest.csv"


def local_co2_status_path() -> Path:
    """Return path to the JSON status metadata for locally fetched CO2 sources."""
    return local_co2_cache_dir() / "co2_sources_status.json"


@dataclass(frozen=True)
class CO2ForcingSeries:
    """Time-varying (or constant) atmospheric CO2 forcing evaluator."""

    mode: str
    name: str
    years: np.ndarray | None
    values: np.ndarray | None
    constant_ppmv: float

    @property
    def is_time_varying(self) -> bool:
        return self.mode != "constant" and self.years is not None and self.values is not None and len(self.years) > 0

    def evaluate(self, decimal_year: float | None = None) -> float:
        """Return atmospheric CO2 concentration [ppmv] at ``decimal_year`` (clamped at endpoints)."""
        if not self.is_time_varying or decimal_year is None:
            return float(self.constant_ppmv)
        assert self.years is not None and self.values is not None
        return float(np.interp(float(decimal_year), self.years, self.values))


def _load_long_table(csv_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a ``scenario,year,co2_ppmv`` CSV into sorted ``(years, values)`` arrays per scenario."""
    buckets: dict[str, list[tuple[float, float]]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scn = (row.get("scenario") or "").strip()
            y_str = (row.get("year") or "").strip()
            v_str = (row.get("co2_ppmv") or "").strip()
            if not scn or not y_str or not v_str:
                continue
            buckets.setdefault(scn, []).append((float(y_str), float(v_str)))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scn, pts in buckets.items():
        pts.sort(key=lambda t: t[0])
        ys = np.array([t[0] for t in pts], dtype=np.float64)
        vs = np.array([t[1] for t in pts], dtype=np.float64)
        out[scn] = (ys, vs)
    return out


def load_all_available_scenarios(prefer_cache: bool = True) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    """Load merged scenario dict, preferring ``data/forcing/co2/co2_latest.csv`` over bundled CSV."""
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_label = "bundled"
    bundled = bundled_co2_table_path()
    if bundled.exists():
        scenarios.update(_load_long_table(bundled))
    cache = local_co2_cache_path()
    if prefer_cache and cache.exists():
        scenarios.update(_load_long_table(cache))
        source_label = "local_cache"
    return scenarios, source_label


def _load_custom_two_col_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a user CSV containing ``year`` and ``co2_ppmv`` (or 2 numeric columns)."""
    pts: list[tuple[float, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"Empty CO2 CSV file: {csv_path}")
    reader = csv.reader(lines)
    header = next(reader)
    y_idx, v_idx = 0, 1
    lowered = [h.strip().lower() for h in header]
    is_header = False
    for i, col in enumerate(lowered):
        if col in ("year", "yr", "year_ce", "time"):
            y_idx = i
            is_header = True
        elif col in ("co2_ppmv", "co2", "co2_ppm", "mean", "value"):
            v_idx = i
            is_header = True
    if not is_header:
        pts.append((float(header[0]), float(header[1])))
    for row in reader:
        if len(row) <= max(y_idx, v_idx):
            continue
        pts.append((float(row[y_idx]), float(row[v_idx])))
    if not pts:
        raise ValueError(f"No valid (year, co2_ppmv) rows in {csv_path}")
    pts.sort(key=lambda t: t[0])
    return np.array([p[0] for p in pts], dtype=np.float64), np.array([p[1] for p in pts], dtype=np.float64)


def resolve_co2_series(
    co2_mode: str = "constant",
    co2_ppmv: float = 400.0,
    co2_scenario: str = "obs_historical_noaa",
    co2_csv_path: str = "",
) -> CO2ForcingSeries:
    """Build a ``CO2ForcingSeries`` from AOBA configuration fields."""
    mode = str(co2_mode or "constant").strip().lower()
    if mode not in VALID_CO2_MODES:
        raise ValueError(f"Invalid co2_mode '{co2_mode}'; expected one of {VALID_CO2_MODES}")

    if mode == "constant":
        return CO2ForcingSeries(
            mode="constant",
            name=f"constant:{float(co2_ppmv):.2f}",
            years=None,
            values=None,
            constant_ppmv=float(co2_ppmv),
        )

    if mode == "csv":
        p = Path(str(co2_csv_path).strip()).expanduser()
        if not str(co2_csv_path).strip() or not p.exists():
            raise FileNotFoundError(f"co2_mode='csv' requires an existing file, got '{co2_csv_path}'")
        ys, vs = _load_custom_two_col_csv(p)
        return CO2ForcingSeries(
            mode="csv",
            name=p.name,
            years=ys,
            values=vs,
            constant_ppmv=float(co2_ppmv),
        )

    scenarios, _ = load_all_available_scenarios(prefer_cache=True)
    key = str(co2_scenario or "").strip()
    if key not in scenarios:
        avail = ", ".join(sorted(scenarios.keys()))
        raise KeyError(f"Unknown co2_scenario '{key}'. Available: {avail}")
    ys, vs = scenarios[key]
    return CO2ForcingSeries(
        mode="scenario",
        name=key,
        years=ys,
        values=vs,
        constant_ppmv=float(co2_ppmv),
    )


def _fetch_text(url: str, timeout: float = 25.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "AOBA-EarthSystemModel/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_noaa_annmean_csv(text: str) -> list[tuple[float, float]]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    pts: list[tuple[float, float]] = []
    for row in reader:
        y = row.get("year")
        m = row.get("mean")
        if y is not None and m is not None:
            pts.append((float(y), float(m)))
    pts.sort(key=lambda t: t[0])
    return pts


def _parse_ncei_paleo_txt(text: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("\ufeff#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            age_bp = float(parts[0])
            co2 = float(parts[1])
        except ValueError:
            continue
        year_ce = round(1950.0 - age_bp, 2)
        pts.append((year_ce, co2))
    pts.sort(key=lambda t: t[0])
    return pts


def _parse_rcmip_co2_csv(text: str) -> dict[str, list[tuple[float, float]]]:
    reader = csv.DictReader(io.StringIO(text))
    year_cols = [c for c in (reader.fieldnames or []) if c.isdigit()]
    out: dict[str, list[tuple[float, float]]] = {}
    for row in reader:
        if row.get("Variable") != "Atmospheric Concentrations|CO2" or row.get("Region") != "World":
            continue
        scn_raw = (row.get("Scenario") or "").strip()
        if scn_raw not in _RCMIP_SCENARIO_MAP:
            continue
        target = _RCMIP_SCENARIO_MAP[scn_raw]
        pts: list[tuple[float, float]] = []
        for yc in year_cols:
            val = (row.get(yc) or "").strip()
            if val:
                pts.append((float(yc), float(val)))
        if pts:
            out[target] = pts
    return out


def _write_long_table(csv_path: Path, scenarios: dict[str, list[tuple[float, float]]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".tmp.csv")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "year", "co2_ppmv"])
        for scn in sorted(scenarios.keys()):
            for y, v in scenarios[scn]:
                y_fmt = f"{int(y)}" if float(y).is_integer() else f"{y:.2f}"
                writer.writerow([scn, y_fmt, f"{v:.4f}"])
    tmp_path.replace(csv_path)


def _merge_obs_with_historical(
    hist_pts: list[tuple[float, float]],
    obs_pts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not obs_pts:
        return list(hist_pts)
    first_obs = obs_pts[0][0]
    merged = [(y, v) for y, v in hist_pts if y < first_obs] + list(obs_pts)
    merged.sort(key=lambda t: t[0])
    return merged


def _splice_obs_to_ssp(
    obs_merged: list[tuple[float, float]],
    ssp_pts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not obs_merged:
        return list(ssp_pts)
    last_obs_yr = obs_merged[-1][0]
    spliced = list(obs_merged) + [(y, v) for y, v in ssp_pts if y > last_obs_yr]
    spliced.sort(key=lambda t: t[0])
    return spliced


def fetch_latest_co2_sources(
    include_rcmip: bool = True,
    include_paleo: bool = True,
    write_bundled: bool = False,
) -> dict[str, Any]:
    """Download latest CO2 records from NOAA GML, NCEI Paleo, and optionally RCMIP v5.1.0.

    Saves merged annual/paleo tables to ``data/forcing/co2/co2_latest.csv`` (gitignored)
    and writes status metadata to ``data/forcing/co2/co2_sources_status.json``.
    If ``write_bundled=True``, also updates ``src/aoba/data/co2_scenarios.csv``.
    """
    existing, _ = load_all_available_scenarios(prefer_cache=True)
    scenarios: dict[str, list[tuple[float, float]]] = {
        k: list(zip(ys.tolist(), vs.tolist())) for k, (ys, vs) in existing.items()
    }
    status: dict[str, Any] = {"sources": {}, "scenarios_count": 0}

    if include_rcmip:
        rcmip_txt = _fetch_text(RCMIP_CONC_V510_URL, timeout=60.0)
        rcmip_parsed = _parse_rcmip_co2_csv(rcmip_txt)
        scenarios.update(rcmip_parsed)
        status["sources"]["rcmip_v510"] = {
            "url": RCMIP_CONC_V510_URL,
            "scenarios": sorted(rcmip_parsed.keys()),
            "ok": True,
        }

    if include_paleo:
        paleo_txt = _fetch_text(NCEI_PALEO_800KA_URL, timeout=30.0)
        paleo_pts = _parse_ncei_paleo_txt(paleo_txt)
        if paleo_pts:
            scenarios["paleo_antarctica_800ka"] = paleo_pts
            status["sources"]["ncei_paleo_800ka"] = {
                "url": NCEI_PALEO_800KA_URL,
                "points": len(paleo_pts),
                "min_year": paleo_pts[0][0],
                "max_year": paleo_pts[-1][0],
                "ok": True,
            }

    noaa_gl_pts = _parse_noaa_annmean_csv(_fetch_text(NOAA_GML_GLOBAL_URL, timeout=20.0))
    noaa_mlo_pts = _parse_noaa_annmean_csv(_fetch_text(NOAA_GML_MLO_URL, timeout=20.0))

    hist_base = scenarios.get("cmip6_historical", [])
    obs_gl = _merge_obs_with_historical(hist_base, noaa_gl_pts)
    obs_mlo = _merge_obs_with_historical(hist_base, noaa_mlo_pts)
    scenarios["obs_historical_noaa"] = obs_gl
    scenarios["obs_mauna_loa"] = obs_mlo

    for ssp_suffix in ("ssp126", "ssp245", "ssp370", "ssp585"):
        ssp_pts = scenarios.get(f"cmip6_{ssp_suffix}")
        if ssp_pts:
            scenarios[f"obs_to_{ssp_suffix}"] = _splice_obs_to_ssp(obs_gl, ssp_pts)

    status["sources"]["noaa_gml_global"] = {
        "url": NOAA_GML_GLOBAL_URL,
        "latest_year": int(noaa_gl_pts[-1][0]) if noaa_gl_pts else None,
        "latest_ppmv": noaa_gl_pts[-1][1] if noaa_gl_pts else None,
        "ok": bool(noaa_gl_pts),
    }
    status["sources"]["noaa_gml_mlo"] = {
        "url": NOAA_GML_MLO_URL,
        "latest_year": int(noaa_mlo_pts[-1][0]) if noaa_mlo_pts else None,
        "latest_ppmv": noaa_mlo_pts[-1][1] if noaa_mlo_pts else None,
        "ok": bool(noaa_mlo_pts),
    }
    status["scenarios_count"] = len(scenarios)

    cache_csv = local_co2_cache_path()
    _write_long_table(cache_csv, scenarios)
    status_path = local_co2_status_path()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_bundled:
        _write_long_table(bundled_co2_table_path(), scenarios)

    return status


def get_co2_catalog_summary(prefer_cache: bool = True) -> dict[str, Any]:
    """Return catalog metadata, year ranges, and sample curves for CLI/GUI display."""
    scenarios, active_source = load_all_available_scenarios(prefer_cache=prefer_cache)
    status_meta: dict[str, Any] = {}
    sp = local_co2_status_path()
    if sp.exists():
        try:
            status_meta = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            status_meta = {}

    items: list[dict[str, Any]] = []
    for meta in CO2_SCENARIO_CATALOG:
        arr = scenarios.get(meta.key)
        if arr is None:
            continue
        ys, vs = arr
        # Downsample preview curve to <= 120 points for lightweight GUI plotting
        if len(ys) > 120:
            idx = np.linspace(0, len(ys) - 1, 120, dtype=int)
            p_ys = [round(float(ys[i]), 2) for i in idx]
            p_vs = [round(float(vs[i]), 2) for i in idx]
        else:
            p_ys = [round(float(y), 2) for y in ys]
            p_vs = [round(float(v), 2) for v in vs]
        items.append({
            "key": meta.key,
            "group": meta.group,
            "label_en": meta.label_en,
            "label_ja": meta.label_ja,
            "source": meta.source,
            "min_year": float(ys[0]),
            "max_year": float(ys[-1]),
            "first_ppmv": round(float(vs[0]), 2),
            "last_ppmv": round(float(vs[-1]), 2),
            "points": int(len(ys)),
            "preview_years": p_ys,
            "preview_ppmv": p_vs,
        })
    return {
        "active_source": active_source,
        "cache_exists": local_co2_cache_path().exists(),
        "status": status_meta,
        "scenarios": items,
    }
