"""Prescribed atmospheric ozone profiles, external input4MIPs fetchers, and interpolators for AOBA.

Supports three modes via ``[forcing] o3_mode``:
1. ``"rcemip"`` (default): Horizontally uniform Wing et al. (2018, Geosci. Model Dev., 11, Eq. 1)
   pressure-dependent gamma distribution ``O3_ppmv(p) = g1 * p_hPa^g2 * exp(-p_hPa / g3)``.
2. ``"latitude_seasonal"``: Map-free analytical 2D latitude-season-pressure parameterization
   reproducing the Brewer-Dobson poleward increase, lower tropical tropopause / higher polar
   tropopause pressure shift, and spring-maximum seasonal cycle (Fortuin & Kelder, 1998).
3. ``"climatology"``: External monthly zonal-mean ``(month, p_hPa, lat)`` climatology loaded from
   ``data/boundary/ozone/o3_zonal_climatology.npz`` (fetched from ESGF input4MIPs CMIP6/CMIP7) or
   a user-specified ``.npz`` / ``.nc`` file via ``o3_file_path``, bilinearly interpolated in
   ``(month, lat, ln p)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np
import torch

from aoba import constants as C


VALID_O3_MODES: tuple[str, ...] = ("rcemip", "latitude_seasonal", "climatology")

ESGF_LLNL_SEARCH_URL = "https://esgf-node.llnl.gov/esg-search/search"


def default_ozone_dir() -> Path:
    """Return local ozone boundary directory ``data/boundary/ozone``."""
    return Path(__file__).resolve().parents[2] / "data" / "boundary" / "ozone"


def default_ozone_climatology_path() -> Path:
    """Return default path to the cached monthly zonal-mean ozone climatology NPZ."""
    return default_ozone_dir() / "o3_zonal_climatology.npz"


def default_ozone_status_path() -> Path:
    """Return path to the JSON metadata describing the downloaded ozone climatology."""
    return default_ozone_dir() / "o3_status.json"


def analytical_latitude_seasonal_ozone_vmr(
    pres_pa: torch.Tensor,
    lat_rad: torch.Tensor | None,
    doy: float = 80.0,
) -> torch.Tensor:
    """Map-free analytical 2D latitude-season-pressure ozone VMR [mol/mol].

    Captures three primary features of observed zonal-mean ozone (Fortuin & Kelder, 1998):
    1. Tropopause/peak-pressure descent from tropics (~10 hPa) to high latitudes (~16 hPa).
    2. Brewer-Dobson poleward column increase (~260 DU tropics -> ~360-400 DU mid/high lat).
    3. Springtime maximum (~DOY 85 NH / ~DOY 268 SH) and autumn minimum.
    """
    p_hpa = torch.clamp(pres_pa * 0.01, min=1.0e-3)
    if lat_rad is None:
        o3_ppmv = C.O3_G1 * torch.pow(p_hpa, C.O3_G2) * torch.exp(-p_hpa / C.O3_G3)
        return o3_ppmv * 1.0e-6

    lat = lat_rad
    while lat.dim() < p_hpa.dim():
        lat = lat.unsqueeze(0)

    sin_lat = torch.sin(lat)
    sin2 = sin_lat * sin_lat

    # Peak pressure scale g3 increases poleward (tropopause lowers from ~100 to ~300 hPa)
    g3_lat = 9.5 + 6.5 * sin2
    # Seasonal cycle peaks in late winter / early spring in each hemisphere
    theta_s = 2.0 * math.pi * (float(doy) - 85.0) / 365.2425
    seas_fac = 1.0 + 0.16 * sin_lat * math.cos(theta_s)
    # Scale amplitude g1 so integrated column grows poleward while preserving g2
    g1_lat = C.O3_G1 * (0.86 + 0.34 * sin2) * seas_fac * torch.pow(C.O3_G3 / g3_lat, C.O3_G2 + 1.0)
    o3_ppmv = torch.clamp(g1_lat * torch.pow(p_hpa, C.O3_G2) * torch.exp(-p_hpa / g3_lat), min=0.0)
    return o3_ppmv * 1.0e-6


@dataclass
class ZonalOzoneClimatology:
    """Monthly zonal-mean ozone climatology ``vmr[12, n_plev, n_lat]`` [mol/mol]."""

    lat_deg: np.ndarray       # [n_lat] ascending (-90..90)
    p_hpa: np.ndarray         # [n_plev] ascending (0.1..1000 hPa)
    ln_p: np.ndarray          # [n_plev] log(p_hpa)
    vmr: np.ndarray           # [12, n_plev, n_lat] mol/mol
    source_name: str = "external"

    @classmethod
    def from_file(cls, path: str | Path) -> "ZonalOzoneClimatology":
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Ozone climatology file not found: {p}")

        if p.suffix.lower() == ".npz":
            data = np.load(p)
            lat_deg = np.asarray(data["lat_deg"], dtype=np.float64)
            p_hpa = np.asarray(data["p_hpa"], dtype=np.float64)
            vmr = np.asarray(data["vmr"], dtype=np.float64)
            src = str(data["source_name"]) if "source_name" in data else p.name
            return cls._normalize_axes(lat_deg, p_hpa, vmr, src)

        if p.suffix.lower() in (".nc", ".nc4"):
            import xarray as xr

            ds = xr.open_dataset(p)
            try:
                var_name = "vmro3" if "vmro3" in ds else next(iter(ds.data_vars))
                da = ds[var_name]
                if "lon" in da.dims:
                    da = da.mean(dim="lon")
                lat_dim = next(d for d in da.dims if "lat" in d.lower())
                p_dim = next(d for d in da.dims if d.lower() in ("plev", "lev", "level", "p"))
                t_dim = next((d for d in da.dims if "time" in d.lower() or "month" in d.lower()), None)

                lat_deg = np.asarray(ds[lat_dim].values, dtype=np.float64)
                p_raw = np.asarray(ds[p_dim].values, dtype=np.float64)
                p_units = str(ds[p_dim].attrs.get("units", "")).lower()
                p_hpa = p_raw * 0.01 if ("pa" in p_units and "hpa" not in p_units) or np.nanmax(p_raw) > 2000.0 else p_raw

                if t_dim is not None:
                    da = da.transpose(t_dim, p_dim, lat_dim)
                    arr = np.asarray(da.values, dtype=np.float64)
                    if arr.shape[0] > 12 and arr.shape[0] % 12 == 0:
                        arr = arr.reshape(-1, 12, arr.shape[1], arr.shape[2]).mean(axis=0)
                    elif arr.shape[0] < 12:
                        arr = np.broadcast_to(arr[:1], (12, arr.shape[1], arr.shape[2])).copy()
                else:
                    da = da.transpose(p_dim, lat_dim)
                    arr = np.broadcast_to(np.asarray(da.values, dtype=np.float64)[None, ...], (12, len(p_hpa), len(lat_deg))).copy()
                return cls._normalize_axes(lat_deg, p_hpa, arr, p.name)
            finally:
                ds.close()

        raise ValueError(f"Unsupported ozone climatology format '{p.suffix}'; expected .npz or .nc")

    @classmethod
    def _normalize_axes(
        cls,
        lat_deg: np.ndarray,
        p_hpa: np.ndarray,
        vmr: np.ndarray,
        source_name: str,
    ) -> "ZonalOzoneClimatology":
        if lat_deg[0] > lat_deg[-1]:
            lat_deg = lat_deg[::-1].copy()
            vmr = vmr[:, :, ::-1].copy()
        if p_hpa[0] > p_hpa[-1]:
            p_hpa = p_hpa[::-1].copy()
            vmr = vmr[:, ::-1, :].copy()
        p_hpa = np.clip(p_hpa, 1.0e-3, 1200.0)
        vmr = np.nan_to_num(np.clip(vmr, 0.0, 1.0e-3), nan=0.0)
        return cls(
            lat_deg=lat_deg,
            p_hpa=p_hpa,
            ln_p=np.log(p_hpa),
            vmr=vmr,
            source_name=source_name,
        )

    def evaluate_torch(
        self,
        pres_pa: torch.Tensor,
        lat_rad: torch.Tensor,
        doy: float = 80.0,
    ) -> torch.Tensor:
        """Interpolate ``vmr[12, p, lat]`` onto ``pres_pa [Nz, N_col]`` and ``lat_rad [N_col]``."""
        dtype = pres_pa.dtype
        device = pres_pa.device
        cache = getattr(self, "_dev_cache", None)
        if cache is None:
            cache = {}
            self._dev_cache = cache
        ckey = (device.type, device.index or 0, str(dtype))
        entry = cache.get(ckey)
        if entry is None:
            entry = (
                torch.as_tensor(self.lat_deg, device=device, dtype=dtype).contiguous(),
                torch.as_tensor(self.ln_p, device=device, dtype=dtype).contiguous(),
                torch.as_tensor(self.vmr, device=device, dtype=dtype).contiguous(),
            )
            cache[ckey] = entry
        lat_tbl, ln_p_tbl, vmr_tbl = entry

        # Cyclic monthly interpolation (month centers at DOY ~ 15.2, 45.6, ...)
        m_float = ((float(doy) - 15.2) / 365.2425) * 12.0
        m0 = int(math.floor(m_float)) % 12
        m1 = (m0 + 1) % 12
        w_m = m_float - math.floor(m_float)
        vmr_m = (1.0 - w_m) * vmr_tbl[m0] + w_m * vmr_tbl[m1]  # [n_plev, n_lat]

        # Vectorized linear interpolation in latitude on native device
        lat_d = torch.rad2deg(lat_rad).to(dtype).contiguous()
        n_lat = lat_tbl.numel()
        j1 = torch.searchsorted(lat_tbl, lat_d).clamp(0, n_lat - 1)
        j0 = (j1 - 1).clamp(0, n_lat - 1)
        w_lat = torch.clamp((lat_d - lat_tbl[j0]) / torch.clamp(lat_tbl[j1] - lat_tbl[j0], min=1.0e-6), 0.0, 1.0)
        vmr_lat = (1.0 - w_lat).unsqueeze(0) * vmr_m[:, j0] + w_lat.unsqueeze(0) * vmr_m[:, j1]  # [n_plev, N_col]

        # Vectorized linear interpolation in ln(p) on native device across all columns
        p_hpa = torch.clamp(pres_pa * 0.01, min=1.0e-4, max=1200.0)
        ln_p_col = torch.log(p_hpa).contiguous()
        n_p = ln_p_tbl.numel()
        k1 = torch.searchsorted(ln_p_tbl, ln_p_col).clamp(0, n_p - 1)
        k0 = (k1 - 1).clamp(0, n_p - 1)
        w_p = torch.clamp((ln_p_col - ln_p_tbl[k0]) / torch.clamp(ln_p_tbl[k1] - ln_p_tbl[k0], min=1.0e-6), 0.0, 1.0)
        v0 = torch.gather(vmr_lat, 0, k0)
        v1 = torch.gather(vmr_lat, 0, k1)
        prof = (1.0 - w_p) * v0 + w_p * v1

        p_top_tbl = float(self.p_hpa[0])
        above = p_hpa < p_top_tbl
        return torch.where(above, prof * torch.pow(torch.clamp(p_hpa / p_top_tbl, max=1.0), C.O3_G2), prof)


@dataclass
class OzoneForcingProvider:
    """Resolved ozone forcing evaluator bound to an ``AOBAState``."""

    mode: str
    climatology: ZonalOzoneClimatology | None = None
    fallback_note: str = ""

    def compute_vmr(
        self,
        pres_pa: torch.Tensor,
        lat_rad: torch.Tensor | None = None,
        doy: float = 80.0,
    ) -> torch.Tensor:
        if self.mode == "climatology" and self.climatology is not None and lat_rad is not None:
            return self.climatology.evaluate_torch(pres_pa, lat_rad, doy=doy)
        if self.mode in ("latitude_seasonal", "climatology") and lat_rad is not None:
            return analytical_latitude_seasonal_ozone_vmr(pres_pa, lat_rad, doy=doy)
        p_hpa = torch.clamp(pres_pa * 0.01, min=1.0e-3)
        o3_ppmv = C.O3_G1 * torch.pow(p_hpa, C.O3_G2) * torch.exp(-p_hpa / C.O3_G3)
        return o3_ppmv * 1.0e-6


def resolve_ozone_provider(
    o3_mode: str = "rcemip",
    o3_file_path: str = "",
) -> OzoneForcingProvider:
    """Resolve ``OzoneForcingProvider`` from ``[forcing]`` configuration."""
    mode = str(o3_mode or "rcemip").strip().lower()
    if mode not in VALID_O3_MODES:
        raise ValueError(f"Invalid o3_mode '{o3_mode}'; expected one of {VALID_O3_MODES}")

    if mode in ("rcemip", "latitude_seasonal"):
        return OzoneForcingProvider(mode=mode)

    custom = str(o3_file_path or "").strip()
    if custom:
        clim = ZonalOzoneClimatology.from_file(custom)
        return OzoneForcingProvider(mode="climatology", climatology=clim)

    def_path = default_ozone_climatology_path()
    if def_path.exists():
        clim = ZonalOzoneClimatology.from_file(def_path)
        return OzoneForcingProvider(mode="climatology", climatology=clim)

    return OzoneForcingProvider(
        mode="climatology",
        climatology=None,
        fallback_note="External ozone file not found; falling back to latitude_seasonal analytical profile.",
    )


def fetch_input4mips_ozone_climatology(
    mip_era: str = "CMIP6",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Download an input4MIPs monthly ``vmro3`` file from ESGF and save zonal-mean climatology NPZ.

    Supported ``mip_era``:
    - ``"CMIP6"``: ``UReading-CCMI-1-0`` (2000-2014 climatological average)
    - ``"CMIP7"``: ``FZJ-CMIP-ozone-1-1`` (recent historical slice)
    """
    import xarray as xr

    era = str(mip_era or "CMIP6").strip().upper()
    query = (
        f"{ESGF_LLNL_SEARCH_URL}?project=input4MIPs&variable_id=vmro3"
        f"&mip_era={era}&type=File&limit=10&format=application%2Fsolr%2Bjson"
    )
    req = urllib.request.Request(query, headers={"User-Agent": "AOBA-EarthSystemModel/1.0"})
    with urllib.request.urlopen(req, timeout=25.0) as r:
        docs = json.loads(r.read().decode("utf-8")).get("response", {}).get("docs", [])

    # Pick a compact recent file (prefer 2000-2014 or 1950-1999)
    chosen_url = ""
    chosen_title = ""
    for doc in docs:
        title = str(doc.get("title", ""))
        urls = [str(u).split("|")[0] for u in doc.get("url", []) if "HTTPServer" in str(u)]
        if not urls:
            continue
        if "200001-201412" in title or "195001-199912" in title or not chosen_url:
            chosen_url = urls[0]
            chosen_title = title
            if "200001-201412" in title or "195001-199912" in title:
                break

    if not chosen_url:
        raise RuntimeError(f"No HTTPServer vmro3 file found on ESGF for mip_era={era}")

    target = Path(out_path).expanduser() if out_path else default_ozone_climatology_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_nc = target.with_suffix(".download.nc")

    dl_req = urllib.request.Request(chosen_url, headers={"User-Agent": "AOBA-EarthSystemModel/1.0"})
    with urllib.request.urlopen(dl_req, timeout=120.0) as r, open(tmp_nc, "wb") as f_out:
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f_out.write(chunk)

    try:
        clim = ZonalOzoneClimatology.from_file(tmp_nc)
    finally:
        if tmp_nc.exists():
            tmp_nc.unlink()

    np.savez_compressed(
        target,
        lat_deg=clim.lat_deg.astype(np.float32),
        p_hpa=clim.p_hpa.astype(np.float32),
        vmr=clim.vmr.astype(np.float32),
        source_name=np.array(f"{era}:{chosen_title}"),
    )
    status = {
        "ok": True,
        "mip_era": era,
        "title": chosen_title,
        "url": chosen_url,
        "path": str(target),
        "shape": list(clim.vmr.shape),
        "p_hpa_range": [float(clim.p_hpa[0]), float(clim.p_hpa[-1])],
    }
    default_ozone_status_path().write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def get_ozone_status_summary() -> dict[str, Any]:
    """Return status of cached external ozone climatology for CLI/GUI display."""
    p = default_ozone_climatology_path()
    sp = default_ozone_status_path()
    meta: dict[str, Any] = {}
    if sp.exists():
        try:
            meta = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {
        "modes": list(VALID_O3_MODES),
        "cache_exists": p.exists(),
        "cache_path": str(p),
        "meta": meta,
    }
