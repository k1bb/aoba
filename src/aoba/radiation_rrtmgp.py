"""Official Open BSD-3 RTE+RRTMGP (HITRAN + MT_CKD 4.1) GPU-Batched Two-Stream Radiative Transfer.

Selectable via `rad_scheme = "rrtmgp"` (alongside `"ckd"`, `"ckd_hd"`, and `"four_band"`).
Auto-downloads the official open-source BSD-3 correlated-k gas-optics NetCDF tables from
`earth-system-radiation/rrtmgp-data` into `data/forcing/rrtmgp/` on first use:
- `rrtmgp-gas-lw-g128.nc`: **16 LW bands / 128 g-points** (10-3250 cm^-1, HITRAN + MT_CKD 4.1,
  2-gas eta overlap tables, exact `totplnk` & `plank_fraction`, band-optimal diffusivity angles).
- `rrtmgp-gas-sw-g112.nc`: **14 SW bands / 112 g-points** (820-50000 cm^-1, HITRAN + MT_CKD 4.1,
  exact g-point Rayleigh scattering `rayl_lower`/`rayl_upper`, and spectral solar irradiance).

Implements the exact trilinear `(T, ln p, eta)` major-gas + `(T, eta)` minor-gas/MT_CKD-continuum
interpolation algorithm of Pincus, Mlawer, & Delamere (2019, *J. Adv. Model. Earth Syst.*, 11,
3074-3089) fully vectorized in PyTorch (batched across all 128 LW / 112 SW g-points simultaneously
on MPS/CUDA/CPU), coupled with Slingo (1989) liquid, Ebert & Curry (1992) ice, Chou et al. (1999)
LW cloud scattering, Balkanski et al. (2007) dust, and AOBA's 5-species natural aerosols.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import urllib.request
import numpy as np
import torch

from aoba import constants as C
from aoba.kernels import can_use_crm_metal, dispatch_rad_rrtmgp_metal
from aoba.radiation import RadiationOutput, background_ozone_vmr, toon_two_stream_sw
from aoba.radiation_ckd import CH4_PREIND_PPMV, N2O_PREIND_PPMV

RRTMGP_BASE_URL: str = "https://raw.githubusercontent.com/earth-system-radiation/rrtmgp-data/main"
RRTMGP_LW_FILE: str = "rrtmgp-gas-lw-g128.nc"
RRTMGP_SW_FILE: str = "rrtmgp-gas-sw-g112.nc"

AVOGADRO: float = 6.02214076e23
M_AIR_KG_MOLEC: float = 28.9647e-3 / AVOGADRO
N2_VMR: float = 0.78084
CO_VMR_BG: float = 1.0e-7  # Clean pre-industrial background CO (~100 ppbv)

_TABLE_CACHE: dict[tuple[str, str], "_RRTMGPTables"] = {}


def default_rrtmgp_dir() -> Path:
    """Return local cache directory `data/forcing/rrtmgp`."""
    return Path(__file__).resolve().parents[2] / "data" / "forcing" / "rrtmgp"


def ensure_rrtmgp_tables(cache_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Ensure official BSD-3 RRTMGP LW (128 g-pt) & SW (112 g-pt) NetCDF files exist locally."""
    d = Path(cache_dir) if cache_dir is not None else default_rrtmgp_dir()
    d.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fn in (RRTMGP_LW_FILE, RRTMGP_SW_FILE):
        p = d / fn
        if not p.exists() or p.stat().st_size < 1_000_000:
            tmp = p.with_suffix(".download.nc")
            req = urllib.request.Request(f"{RRTMGP_BASE_URL}/{fn}", headers={"User-Agent": "AOBA-ESM/1.0"})
            with urllib.request.urlopen(req, timeout=120.0) as r, open(tmp, "wb") as f_out:
                while True:
                    chunk = r.read(131072)
                    if not chunk:
                        break
                    f_out.write(chunk)
            tmp.replace(p)
        paths.append(p)
    return paths[0], paths[1]


@dataclass
class _MinorSpec:
    """Compact preprocessed metadata for one RRTMGP minor-absorber interval."""

    gas_idx: int             # Index into VMR table (1..19; -1 if unused trace halocarbon)
    scaling_gas_idx: int     # Scaling gas index (0 if none)
    scale_by_comp: bool      # True -> (1 - vmr_s)/(1 + vmr_h2o), False -> vmr_s/(1 + vmr_h2o)
    scale_by_dens: bool      # True -> multiply by (p/T)*(T_ref/P_ref)
    gpt_start: int           # 0-based inclusive g-point start
    gpt_end: int             # 0-based exclusive g-point end
    k_start: int             # 0-based start offset in kminor contributor axis
    band_idx: int            # 0-based band index for eta lookup


@dataclass
class _GasOpticsTable:
    """GPU-resident float32 RRTMGP gas optics lookup tables (LW or SW)."""

    n_bnd: int
    n_gpt: int
    press_ref_ln0: float
    press_ref_dln: float
    press_ref_trop: float
    temp_ref_0: float
    temp_ref_d: float
    ref_p: float
    ref_t: float
    n_press: int
    n_temp: int
    n_eta: int
    gpt_to_bnd: torch.Tensor       # [n_gpt] int64
    bnd_wavenum_mid: torch.Tensor  # [n_bnd] float32 [cm^-1]
    key_species: torch.Tensor      # [n_bnd, 2, 2] int64 (0..19)
    vmr_ref: torch.Tensor          # [n_temp, 20, 2] float32
    kmajor: torch.Tensor           # [n_temp, n_press+1, n_eta, n_gpt] float32 [cm^2 / molec]
    kminor_lower: torch.Tensor     # [n_temp, n_eta, n_contrib_l] float32
    kminor_upper: torch.Tensor     # [n_temp, n_eta, n_contrib_u] float32
    minor_lower: list[_MinorSpec]
    minor_upper: list[_MinorSpec]
    # LW-specific
    plank_fraction: torch.Tensor | None = None   # [n_temp, n_press+1, n_eta, n_gpt] float32
    totplnk: torch.Tensor | None = None          # [n_bnd, 196] float32 [W m^-2 sr^-1]
    diff_angle_bnd: torch.Tensor | None = None   # [n_bnd] float32 secant diffusivity factor
    # SW-specific
    rayl_lower: torch.Tensor | None = None       # [n_temp, n_eta, n_gpt] float32
    rayl_upper: torch.Tensor | None = None       # [n_temp, n_eta, n_gpt] float32
    solar_frac_gpt: torch.Tensor | None = None   # [n_gpt] float32 (sums to 1.0)
    is_vis_gpt: torch.Tensor | None = None       # [n_gpt] bool (nu >= 14285 cm^-1 <-> lambda <= 0.7 um)
    is_par_gpt: torch.Tensor | None = None       # [n_gpt] bool (14285 <= nu <= 25000 cm^-1 <-> 0.4-0.7 um)


@dataclass
class _RRTMGPTables:
    lw: _GasOpticsTable
    sw: _GasOpticsTable


def _parse_minor_specs(
    ds,
    suffix: str,
    name_to_idx: dict[str, int],
    gpt_to_bnd_np: np.ndarray,
) -> list[_MinorSpec]:
    gases = [x.decode().strip().lower() for x in ds[f"minor_gases_{suffix}"].values]
    sgases = [x.decode().strip().lower() for x in ds[f"scaling_gas_{suffix}"].values]
    scomp = ds[f"scale_by_complement_{suffix}"].values
    sdens = ds[f"minor_scales_with_density_{suffix}"].values
    lims = ds[f"minor_limits_gpt_{suffix}"].values
    kstarts = ds[f"kminor_start_{suffix}"].values
    # Map minor alias identifiers (h2o_self, h2o_frgn -> h2o)
    minor_alias: dict[str, str] = {}
    gm_arr = [x.decode().strip().lower() for x in ds["gas_minor"].values]
    id_arr = [x.decode().strip().lower() for x in ds["identifier_minor"].values]
    for g_m, id_m in zip(gm_arr, id_arr):
        minor_alias[id_m] = g_m

    specs: list[_MinorSpec] = []
    for i, g_id in enumerate(gases):
        base_name = minor_alias.get(g_id, g_id)
        g_idx = name_to_idx.get(base_name, -1)
        # Keep only gases AOBA represents (h2o, co2, o3, n2o, co, ch4, o2, n2 -> 1..8)
        if g_idx < 1 or g_idx > 8:
            continue
        s_name = sgases[i]
        s_idx = name_to_idx.get(s_name, 0) if s_name else 0
        g0 = int(lims[i, 0]) - 1
        g1 = int(lims[i, 1])  # exclusive
        specs.append(
            _MinorSpec(
                gas_idx=g_idx,
                scaling_gas_idx=s_idx,
                scale_by_comp=bool(int(scomp[i]) != 0),
                scale_by_dens=bool(int(sdens[i]) != 0),
                gpt_start=g0,
                gpt_end=g1,
                k_start=int(kstarts[i]) - 1,
                band_idx=int(gpt_to_bnd_np[g0]),
            )
        )
    return specs


def _load_one_table(path: Path, is_lw: bool, device: torch.device) -> _GasOpticsTable:
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        gas_names = [x.decode().strip().lower() for x in ds["gas_names"].values]
        name_to_idx = {name: i + 1 for i, name in enumerate(gas_names)}
        pr = np.asarray(ds["press_ref"].values, dtype=np.float64)
        tr = np.asarray(ds["temp_ref"].values, dtype=np.float64)
        lims = np.asarray(ds["bnd_limits_gpt"].values, dtype=np.int64)
        wn = np.asarray(ds["bnd_limits_wavenumber"].values, dtype=np.float64)
        n_bnd = lims.shape[0]
        n_gpt = int(ds.sizes["gpt"])
        gpt_to_bnd_np = np.zeros(n_gpt, dtype=np.int64)
        for b in range(n_bnd):
            gpt_to_bnd_np[lims[b, 0] - 1 : lims[b, 1]] = b

        minor_l = _parse_minor_specs(ds, "lower", name_to_idx, gpt_to_bnd_np)
        minor_u = _parse_minor_specs(ds, "upper", name_to_idx, gpt_to_bnd_np)

        # Pre-multiply kmajor/kminor/rayl by 1e20 to keep float32 well inside [1e-10, 1e10] (ZERO underflow!)
        # and divide col_dry by 1e20 during evaluation -> exact float32 dynamic range!
        k_scale = 1.0e20
        kmajor = torch.tensor(np.asarray(ds["kmajor"].values, dtype=np.float64) * k_scale, dtype=torch.float32, device=device)
        kminor_l = torch.tensor(np.asarray(ds["kminor_lower"].values, dtype=np.float64) * k_scale, dtype=torch.float32, device=device)
        kminor_u = torch.tensor(np.asarray(ds["kminor_upper"].values, dtype=np.float64) * k_scale, dtype=torch.float32, device=device)

        wn_mid = 0.5 * (wn[:, 0] + wn[:, 1])

        tab = _GasOpticsTable(
            n_bnd=n_bnd,
            n_gpt=n_gpt,
            press_ref_ln0=float(math.log(pr[0])),
            press_ref_dln=float(math.log(pr[1]) - math.log(pr[0])),  # -0.20
            press_ref_trop=float(ds["press_ref_trop"].values),
            temp_ref_0=float(tr[0]),
            temp_ref_d=float(tr[1] - tr[0]),
            ref_p=float(ds["absorption_coefficient_ref_P"].values),
            ref_t=float(ds["absorption_coefficient_ref_T"].values),
            n_press=len(pr),
            n_temp=len(tr),
            n_eta=int(ds.sizes["mixing_fraction"]),
            gpt_to_bnd=torch.tensor(gpt_to_bnd_np, dtype=torch.long, device=device),
            bnd_wavenum_mid=torch.tensor(wn_mid, dtype=torch.float32, device=device),
            key_species=torch.tensor(np.asarray(ds["key_species"].values, dtype=np.int64), dtype=torch.long, device=device),
            vmr_ref=torch.tensor(np.asarray(ds["vmr_ref"].values, dtype=np.float32), dtype=torch.float32, device=device),
            kmajor=kmajor,
            kminor_lower=kminor_l,
            kminor_upper=kminor_u,
            minor_lower=minor_l,
            minor_upper=minor_u,
        )
        if is_lw:
            tab.plank_fraction = torch.tensor(np.asarray(ds["plank_fraction"].values, dtype=np.float32), dtype=torch.float32, device=device)
            tab.totplnk = torch.tensor(np.asarray(ds["totplnk"].values, dtype=np.float32), dtype=torch.float32, device=device)
            opt_ang = np.asarray(ds["optimal_angle_fit"].values, dtype=np.float32)
            # Band-mean secant diffusivity factor (clamped safely around 1.66)
            tab.diff_angle_bnd = torch.tensor(np.clip(opt_ang[:, 1], 1.45, 2.05), dtype=torch.float32, device=device)
        else:
            tab.rayl_lower = torch.tensor(np.asarray(ds["rayl_lower"].values, dtype=np.float64) * k_scale, dtype=torch.float32, device=device)
            tab.rayl_upper = torch.tensor(np.asarray(ds["rayl_upper"].values, dtype=np.float64) * k_scale, dtype=torch.float32, device=device)
            s_q = np.asarray(ds["solar_source_quiet"].values, dtype=np.float64)
            s_f = np.asarray(ds["solar_source_facular"].values, dtype=np.float64)
            s_s = np.asarray(ds["solar_source_sunspot"].values, dtype=np.float64)
            mg = float(ds["mg_default"].values)
            sb = float(ds["sb_default"].values)
            s_raw = np.clip(s_q + mg * s_f + sb * s_s, 0.0, None)
            s_frac = s_raw / max(float(s_raw.sum()), 1.0)
            tab.solar_frac_gpt = torch.tensor(s_frac, dtype=torch.float32, device=device)
            wn_gpt = wn_mid[gpt_to_bnd_np]
            tab.is_vis_gpt = torch.tensor(wn_gpt >= 14285.0, dtype=torch.bool, device=device)
            tab.is_par_gpt = torch.tensor((wn_gpt >= 14285.0) & (wn_gpt <= 25000.0), dtype=torch.bool, device=device)
        return tab
    finally:
        ds.close()


def get_rrtmgp_tables(device: torch.device, cache_dir: str | Path | None = None) -> _RRTMGPTables:
    """Return GPU-cached official RRTMGP LW & SW gas-optics tables."""
    key = (str(device), str(cache_dir or ""))
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    lw_p, sw_p = ensure_rrtmgp_tables(cache_dir)
    tabs = _RRTMGPTables(
        lw=_load_one_table(lw_p, is_lw=True, device=device),
        sw=_load_one_table(sw_p, is_lw=False, device=device),
    )
    _TABLE_CACHE[key] = tabs
    return tabs


def _eval_gas_optics(
    tab: _GasOpticsTable,
    p_pa: torch.Tensor,         # [N, M] top-down
    t_k: torch.Tensor,          # [N, M] top-down
    col_dry_e20: torch.Tensor,  # [N, M] col_dry [molec cm^-2] * 1e-20
    vmrs: torch.Tensor,         # [9, N, M] index 0=1.0 (dry), 1=h2o, 2=co2, 3=o3, 4=n2o, 5=co, 6=ch4, 7=o2, 8=n2
    *,
    want_planck: bool = False,
    want_rayl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Vectorized RRTMGP trilinear (T, ln p, eta) major + minor gas optics `[N, M, n_gpt]`."""
    N, M = p_pa.shape
    dev = p_pa.device
    dtype = p_pa.dtype
    ng = tab.n_gpt
    nb = tab.n_bnd

    # 1. Exact temperature & log-pressure interpolation indices and fractional weights
    t_pos = (t_k - tab.temp_ref_0) / tab.temp_ref_d
    jt0 = torch.clamp(torch.floor(t_pos).to(torch.long), 0, tab.n_temp - 2)
    ft = torch.clamp(t_pos - jt0.to(dtype), 0.0, 1.0)
    jt1 = jt0 + 1

    ln_p = torch.log(torch.clamp(p_pa, min=0.5))
    p_pos = (ln_p - tab.press_ref_ln0) / tab.press_ref_dln  # ascending 0..58 from 1097 hPa -> 0.01 hPa
    is_upper = p_pa <= tab.press_ref_trop  # [N, M]
    tropo = is_upper.to(torch.long)        # 0 = lower, 1 = upper

    # Lower atm: p_pos in [0, 12] -> jp0 in [0..11], jp_km = jp0 (levels 0..12 in kmajor)
    # Upper atm: p_pos in [12, 58] -> jp0 in [12..57], jp_km = jp0 + 1 (levels 13..59 in kmajor)
    jp_low = torch.clamp(torch.floor(p_pos).to(torch.long), 0, 11)
    fp_low = torch.clamp(p_pos - jp_low.to(dtype), 0.0, 1.0)
    jp_up = torch.clamp(torch.floor(p_pos).to(torch.long), 12, tab.n_press - 2)
    fp_up = torch.clamp(p_pos - jp_up.to(dtype), 0.0, 1.0)

    jp0_km = torch.where(is_upper, jp_up + 1, jp_low)
    jp1_km = jp0_km + 1
    fp = torch.where(is_upper, fp_up, fp_low)

    # 2. Binary-species mixing fraction eta & combined column col_mix per band at jt0 and jt1
    # key_species: [nb, 2, 2] -> s1, s2 for every (N, M, nb)
    s1 = tab.key_species[:, :, 0][:, tropo].permute(1, 2, 0)  # [N, M, nb]
    s2 = tab.key_species[:, :, 1][:, tropo].permute(1, 2, 0)  # [N, M, nb]
    has_key = (s1 > 0) | (s2 > 0)

    # Gather actual VMRs [N, M, nb] from vmrs [9, N, M]
    vmrs_t = vmrs.permute(1, 2, 0)  # [N, M, 9]
    v1 = torch.gather(vmrs_t, 2, s1)
    v2 = torch.gather(vmrs_t, 2, s2)

    # Reference VMR ratio f_ref = vmr_ref[jt, s1, tropo] / vmr_ref[jt, s2, tropo] at jt0 and jt1
    def _eta_and_colmix(jt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        jt_b = jt.unsqueeze(-1).expand(N, M, nb)
        tr_b = tropo.unsqueeze(-1).expand(N, M, nb)
        vr1 = tab.vmr_ref[jt_b, s1, tr_b]
        vr2 = tab.vmr_ref[jt_b, s2, tr_b]
        f_ref = vr1 / torch.clamp(vr2, min=1.0e-30)
        v_mix = v1 + f_ref * v2
        eta = torch.where(has_key & (v_mix > 1.0e-30), v1 / torch.clamp(v_mix, min=1.0e-30), torch.zeros_like(v1))
        eta_scaled = torch.clamp(eta * float(tab.n_eta - 1), 0.0, float(tab.n_eta - 1) - 1.0e-5)
        je0 = torch.clamp(torch.floor(eta_scaled).to(torch.long), 0, tab.n_eta - 2)
        fe = torch.clamp(eta_scaled - je0.to(dtype), 0.0, 1.0)
        cmix = torch.where(has_key, col_dry_e20.unsqueeze(-1) * v_mix, torch.zeros_like(v_mix))
        return je0, fe, cmix

    je0_t0, fe_t0, cmix_t0 = _eta_and_colmix(jt0)
    je0_t1, fe_t1, cmix_t1 = _eta_and_colmix(jt1)

    # Expand band-level indices to g-point axis [N, M, ng]
    g2b = tab.gpt_to_bnd
    g_idx = torch.arange(ng, device=dev, dtype=torch.long).view(1, 1, ng).expand(N, M, ng)
    jt0_g = jt0.unsqueeze(-1).expand(N, M, ng)
    jt1_g = jt1.unsqueeze(-1).expand(N, M, ng)
    jp0_g = jp0_km.unsqueeze(-1).expand(N, M, ng)
    jp1_g = jp1_km.unsqueeze(-1).expand(N, M, ng)

    je0_t0_g = je0_t0[:, :, g2b]
    fe_t0_g = fe_t0[:, :, g2b]
    cmix_t0_g = cmix_t0[:, :, g2b]
    je0_t1_g = je0_t1[:, :, g2b]
    fe_t1_g = fe_t1[:, :, g2b]
    cmix_t1_g = cmix_t1[:, :, g2b]

    ft_g = ft.unsqueeze(-1)
    fp_g = fp.unsqueeze(-1)

    # 3. Trilinear (T, P, eta) interpolation of kmajor [N, M, ng]
    def _interp_3d(table4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # At jt0
        v000 = table4d[jt0_g, jp0_g, je0_t0_g, g_idx]
        v001 = table4d[jt0_g, jp0_g, je0_t0_g + 1, g_idx]
        v010 = table4d[jt0_g, jp1_g, je0_t0_g, g_idx]
        v011 = table4d[jt0_g, jp1_g, je0_t0_g + 1, g_idx]
        k_t0 = (1.0 - fp_g) * ((1.0 - fe_t0_g) * v000 + fe_t0_g * v001) + fp_g * ((1.0 - fe_t0_g) * v010 + fe_t0_g * v011)
        # At jt1
        v100 = table4d[jt1_g, jp0_g, je0_t1_g, g_idx]
        v101 = table4d[jt1_g, jp0_g, je0_t1_g + 1, g_idx]
        v110 = table4d[jt1_g, jp1_g, je0_t1_g, g_idx]
        v111 = table4d[jt1_g, jp1_g, je0_t1_g + 1, g_idx]
        k_t1 = (1.0 - fp_g) * ((1.0 - fe_t1_g) * v100 + fe_t1_g * v101) + fp_g * ((1.0 - fe_t1_g) * v110 + fe_t1_g * v111)
        return k_t0, k_t1

    k_t0, k_t1 = _interp_3d(tab.kmajor)
    tau = torch.clamp((1.0 - ft_g) * (k_t0 * cmix_t0_g) + ft_g * (k_t1 * cmix_t1_g), min=0.0)

    # 4. Minor gases & exact MT_CKD 4.1 H2O self/foreign continuum (lower & upper atm)
    dens_fac = (p_pa / torch.clamp(t_k, min=100.0)) * (tab.ref_t / tab.ref_p)
    inv_1ph2o = 1.0 / (1.0 + vmrs[1])
    low_mask = (~is_upper).to(dtype)
    up_mask = is_upper.to(dtype)

    def _add_minors(specs: list[_MinorSpec], kminor: torch.Tensor, mask_half: torch.Tensor) -> None:
        if not bool(mask_half.any()):
            return
        for sp in specs:
            g_col = col_dry_e20 * vmrs[sp.gas_idx] * mask_half
            if sp.scale_by_dens:
                g_col = g_col * dens_fac
            if sp.scaling_gas_idx > 0:
                vs = vmrs[sp.scaling_gas_idx]
                g_col = g_col * ((1.0 - vs) * inv_1ph2o if sp.scale_by_comp else vs * inv_1ph2o)
            span = sp.gpt_end - sp.gpt_start
            c_idx = torch.arange(sp.k_start, sp.k_start + span, device=dev, dtype=torch.long).view(1, 1, span).expand(N, M, span)
            jt0_s = jt0.unsqueeze(-1).expand(N, M, span)
            jt1_s = jt1.unsqueeze(-1).expand(N, M, span)
            je0_0 = je0_t0[:, :, sp.band_idx].unsqueeze(-1).expand(N, M, span)
            fe_0 = fe_t0[:, :, sp.band_idx].unsqueeze(-1)
            je0_1 = je0_t1[:, :, sp.band_idx].unsqueeze(-1).expand(N, M, span)
            fe_1 = fe_t1[:, :, sp.band_idx].unsqueeze(-1)
            km0 = (1.0 - fe_0) * kminor[jt0_s, je0_0, c_idx] + fe_0 * kminor[jt0_s, je0_0 + 1, c_idx]
            km1 = (1.0 - fe_1) * kminor[jt1_s, je0_1, c_idx] + fe_1 * kminor[jt1_s, je0_1 + 1, c_idx]
            tau[:, :, sp.gpt_start : sp.gpt_end] = tau[:, :, sp.gpt_start : sp.gpt_end] + torch.clamp(
                ((1.0 - ft_g) * km0 + ft_g * km1) * g_col.unsqueeze(-1), min=0.0
            )

    _add_minors(tab.minor_lower, tab.kminor_lower, low_mask)
    _add_minors(tab.minor_upper, tab.kminor_upper, up_mask)

    pfrac = None
    if want_planck and tab.plank_fraction is not None:
        pf0, pf1 = _interp_3d(tab.plank_fraction)
        pfrac = torch.clamp((1.0 - ft_g) * pf0 + ft_g * pf1, min=0.0)

    tau_rayl = None
    if want_rayl and tab.rayl_lower is not None and tab.rayl_upper is not None:
        rl0 = (1.0 - fe_t0_g) * tab.rayl_lower[jt0_g, je0_t0_g, g_idx] + fe_t0_g * tab.rayl_lower[jt0_g, je0_t0_g + 1, g_idx]
        rl1 = (1.0 - fe_t1_g) * tab.rayl_lower[jt1_g, je0_t1_g, g_idx] + fe_t1_g * tab.rayl_lower[jt1_g, je0_t1_g + 1, g_idx]
        ru0 = (1.0 - fe_t0_g) * tab.rayl_upper[jt0_g, je0_t0_g, g_idx] + fe_t0_g * tab.rayl_upper[jt0_g, je0_t0_g + 1, g_idx]
        ru1 = (1.0 - fe_t1_g) * tab.rayl_upper[jt1_g, je0_t1_g, g_idx] + fe_t1_g * tab.rayl_upper[jt1_g, je0_t1_g + 1, g_idx]
        k_rayl = torch.where(is_upper.unsqueeze(-1), (1.0 - ft_g) * ru0 + ft_g * ru1, (1.0 - ft_g) * rl0 + ft_g * rl1)
        tau_rayl = torch.clamp(k_rayl * col_dry_e20.unsqueeze(-1), min=0.0)

    return tau, pfrac, tau_rayl


def _interp_totplnk(totplnk: torch.Tensor, temp_k: torch.Tensor) -> torch.Tensor:
    """Interpolate official RRTMGP Planck radiance table `totplnk [16, 196]` -> flux `[..., 16]` [W m^-2]."""
    # Grid is 160..355 K at 1 K resolution; returns pi * totplnk [W m^-2]
    t_pos = torch.clamp(temp_k - 160.0, 0.0, 194.999)
    i0 = torch.floor(t_pos).to(torch.long)
    w = (t_pos - i0.to(temp_k.dtype)).unsqueeze(-1)
    # totplnk.T: [196, 16]
    tp_t = totplnk.transpose(0, 1)
    rad = (1.0 - w) * tp_t[i0] + w * tp_t[i0 + 1]
    return math.pi * rad


def compute_rrtmgp_radiation(
    pres_c: torch.Tensor,
    pres_w: torch.Tensor,
    temp_c: torch.Tensor,
    temp_w: torch.Tensor,
    dz_c: torch.Tensor,
    rho_d_c: torch.Tensor,
    qv_c: torch.Tensor,
    qc_c: torch.Tensor,
    qi_c: torch.Tensor,
    t_sfc: torch.Tensor,
    albedo_vis: torch.Tensor,
    albedo_nir: torch.Tensor,
    mu0: torch.Tensor,
    s_toa: torch.Tensor,
    co2_ppmv: float = C.CO2_PREIND_PPMV,
    ch4_ppmv: float = CH4_PREIND_PPMV,
    n2o_ppmv: float = N2O_PREIND_PPMV,
    emiss_sfc: float | torch.Tensor = 0.98,
    re_liq_um: float | torch.Tensor = 10.0,
    re_ice_um: float | torch.Tensor = 48.0,
    cloud_sw_inhomog: float = 0.85,
    albedo_vis_dif: torch.Tensor | None = None,
    albedo_nir_dif: torch.Tensor | None = None,
    q_dust_c: torch.Tensor | None = None,
    o3_vmr_c: torch.Tensor | None = None,
    o3_lw_coupling: bool = True,
    aero_opt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    col_batch: int = 512,
) -> RadiationOutput:
    """Evaluate official HITRAN/MT_CKD-4.1 RRTMGP (128 LW + 112 SW g-points) two-stream radiation.

    Chunks columns internally (`col_batch = 512`) so `[N, col_batch, 128]` fits comfortably in
    GPU cache/VRAM on both Apple Silicon MPS and NVIDIA A100/H100.
    """
    orig_shape = t_sfc.shape
    N = pres_c.shape[0]
    M_tot = t_sfc.numel()
    dev = pres_c.device
    dtype = pres_c.dtype
    tabs = get_rrtmgp_tables(dev)

    if can_use_crm_metal(pres_c) and M_tot > 0:
        sw_up, sw_dn, lw_up, lw_dn, q_sw, q_lw, sfc_out = dispatch_rad_rrtmgp_metal(
            tabs=tabs,
            pres_c=pres_c,
            temp_c=temp_c,
            temp_w=temp_w,
            dz_c=dz_c,
            rho_d_c=rho_d_c,
            qv_c=qv_c,
            qc_c=qc_c,
            qi_c=qi_c,
            t_sfc=t_sfc,
            albedo_vis=albedo_vis,
            albedo_nir=albedo_nir,
            mu0=mu0,
            s_toa=s_toa,
            co2_ppmv=co2_ppmv,
            ch4_ppmv=ch4_ppmv,
            n2o_ppmv=n2o_ppmv,
            emiss_sfc=emiss_sfc,
            re_liq_um=re_liq_um,
            re_ice_um=re_ice_um,
            cloud_sw_inhomog=cloud_sw_inhomog,
            albedo_vis_dif=albedo_vis_dif,
            albedo_nir_dif=albedo_nir_dif,
            q_dust_c=q_dust_c,
            o3_vmr_c=o3_vmr_c,
            o3_lw_coupling=o3_lw_coupling,
            aero_opt=aero_opt,
        )
        sfc_par = sfc_out[4]
        return RadiationOutput(
            sw_flux_up=sw_up,
            sw_flux_dn=sw_dn,
            lw_flux_up=lw_up,
            lw_flux_dn=lw_dn,
            q_rad_sw=q_sw,
            q_rad_lw=q_lw,
            q_rad_net=q_sw + q_lw,
            toa_insolation=sw_dn[-1],
            toa_asr=sw_dn[-1] - sw_up[-1],
            toa_olr=lw_up[-1],
            sfc_sw_net=sw_dn[0] - sw_up[0],
            sfc_lw_net=lw_dn[0] - lw_up[0],
            sfc_par_wm2=sfc_par,
            sfc_ppfd=C.UMOL_PER_JOULE_PAR * sfc_par,
            sfc_sw1_dir_dn=sfc_out[0],
            sfc_sw1_dif_dn=sfc_out[1],
            sfc_sw2_dir_dn=sfc_out[2],
            sfc_sw2_dif_dn=sfc_out[3],
        )

    # Flatten column dimensions to [N, M_tot] and flip to top-down [0 = TOA, N-1 = surface]
    p_c_all = torch.flip(pres_c.reshape(N, M_tot), dims=[0])
    t_c_all = torch.flip(temp_c.reshape(N, M_tot), dims=[0])
    t_w_all = torch.flip(temp_w.reshape(N + 1, M_tot), dims=[0])
    dz_all = torch.flip(dz_c.reshape(N, M_tot), dims=[0])
    rho_all = torch.flip(rho_d_c.reshape(N, M_tot), dims=[0])
    qv_all = torch.clamp(torch.flip(qv_c.reshape(N, M_tot), dims=[0]), min=0.0)
    qc_all = torch.clamp(torch.flip(qc_c.reshape(N, M_tot), dims=[0]), min=0.0)
    qi_all = torch.clamp(torch.flip(qi_c.reshape(N, M_tot), dims=[0]), min=0.0)
    o3_all = torch.clamp(torch.flip(o3_vmr_c.reshape(N, M_tot), dims=[0]), min=0.0) if o3_vmr_c is not None else background_ozone_vmr(p_c_all)
    qd_all = torch.clamp(torch.flip(q_dust_c.reshape(N, M_tot), dims=[0]), min=0.0) if q_dust_c is not None else None
    rel_all = torch.clamp(torch.flip(re_liq_um.reshape(N, M_tot), dims=[0]), 3.0, 40.0) if isinstance(re_liq_um, torch.Tensor) else torch.full((N, M_tot), max(min(float(re_liq_um), 40.0), 3.0), device=dev, dtype=dtype)
    rei_all = torch.clamp(torch.flip(re_ice_um.reshape(N, M_tot), dims=[0]), 10.0, 150.0) if isinstance(re_ice_um, torch.Tensor) else torch.full((N, M_tot), max(min(float(re_ice_um), 150.0), 10.0), device=dev, dtype=dtype)

    ts_all = t_sfc.reshape(M_tot)
    av_all = albedo_vis.reshape(M_tot)
    an_all = albedo_nir.reshape(M_tot)
    avf_all = av_all if albedo_vis_dif is None else albedo_vis_dif.reshape(M_tot)
    anf_all = an_all if albedo_nir_dif is None else albedo_nir_dif.reshape(M_tot)
    mu0_all = mu0.reshape(M_tot)
    stoa_all = s_toa.reshape(M_tot)
    em_all = emiss_sfc.reshape(M_tot) if isinstance(emiss_sfc, torch.Tensor) else torch.full((M_tot,), float(emiss_sfc), device=dev, dtype=dtype)

    ao_td = None
    if aero_opt is not None:
        ao_td = tuple(torch.flip(x.reshape(N, M_tot), dims=[0]) for x in aero_opt)

    sw_up_td = torch.empty((N + 1, M_tot), device=dev, dtype=dtype)
    sw_dn_td = torch.empty((N + 1, M_tot), device=dev, dtype=dtype)
    lw_up_td = torch.empty((N + 1, M_tot), device=dev, dtype=dtype)
    lw_dn_td = torch.empty((N + 1, M_tot), device=dev, dtype=dtype)
    s1_dir_all = torch.empty(M_tot, device=dev, dtype=dtype)
    s1_dif_all = torch.empty(M_tot, device=dev, dtype=dtype)
    s2_dir_all = torch.empty(M_tot, device=dev, dtype=dtype)
    s2_dif_all = torch.empty(M_tot, device=dev, dtype=dtype)
    par_all = torch.empty(M_tot, device=dev, dtype=dtype)

    chi = float(cloud_sw_inhomog)
    # Band-level cloud LW absorption coefficients for 16 RRTMGP bands (Hu & Stamnes 1993; Ebert & Curry 1992; Chou 1999)
    lw_wn = tabs.lw.bnd_wavenum_mid  # [16]
    # Smooth wavenumber-dependent liquid/ice LW absorption [m^2 g^-1] with Chou (1999) scattering factor
    k_liq_lw_b = (0.065 + 0.020 * torch.exp(-((lw_wn - 650.0) / 350.0) ** 2)).view(1, 1, 16)
    k_liq_lw_r = (0.28 + 0.10 * torch.exp(-((lw_wn - 950.0) / 250.0) ** 2)).view(1, 1, 16)
    k_ice_lw_b = (0.0025 + 0.0025 * torch.exp(-lw_wn / 600.0)).view(1, 1, 16)
    k_ice_lw_r = (1.05 + 0.12 * torch.exp(-((lw_wn - 900.0) / 300.0) ** 2)).view(1, 1, 16)
    k_dust_lw_b = (95.0 + 115.0 * torch.exp(-((lw_wn - 1030.0) / 160.0) ** 2)).view(1, 1, 16)
    # Band-level SW wavelength [um] for 14 RRTMGP SW bands -> Slingo liquid & Ebert-Curry ice + Angstrom aerosol
    sw_lam_um = 1.0e4 / torch.clamp(tabs.sw.bnd_wavenum_mid, min=500.0)  # [14]
    sw_aero_sc = torch.clamp((0.55 / sw_lam_um) ** 1.05, 0.15, 2.0).view(1, 1, 14)
    sw_dust_kext_b = (1200.0 * torch.clamp((0.55 / sw_lam_um) ** 0.25, 0.5, 1.3)).view(1, 1, 14)
    sw_dust_ssa_b = torch.clamp(0.92 - 0.12 * torch.exp(-((sw_lam_um - 0.40) / 0.18) ** 2), 0.78, 0.96).view(1, 1, 14)
    sw_cld_om_l = torch.clamp(1.0 - 0.0015 * (sw_lam_um ** 2.2), 0.88, 0.9999).view(1, 1, 14)
    sw_cld_om_i = torch.clamp(1.0 - 0.0025 * (sw_lam_um ** 2.1), 0.85, 0.9999).view(1, 1, 14)

    for i0 in range(0, M_tot, max(col_batch, 1)):
        i1 = min(i0 + max(col_batch, 1), M_tot)
        M = i1 - i0
        sl = slice(i0, i1)
        p_c = p_c_all[:, sl]
        t_c = t_c_all[:, sl]
        t_w = t_w_all[:, sl]
        dz = dz_all[:, sl]
        rho_d = rho_all[:, sl]
        qv = qv_all[:, sl]
        qc = qc_all[:, sl]
        qi = qi_all[:, sl]
        o3 = o3_all[:, sl]
        rel = rel_all[:, sl]
        rei = rei_all[:, sl]

        dm_dry = torch.clamp(rho_d * dz, min=1.0e-5)
        col_dry_e20 = (dm_dry / M_AIR_KG_MOLEC) * 1.0e-24  # [molec cm^-2] * 1e-20
        lwp_gm2 = 1000.0 * qc * dm_dry
        iwp_gm2 = 1000.0 * qi * dm_dry
        dm_dust = (qd_all[:, sl] * dm_dry) if qd_all is not None else None

        # Assemble 9-species VMR tensor [9, N, M]: 0=1.0(dry), 1=h2o, 2=co2, 3=o3, 4=n2o, 5=co, 6=ch4, 7=o2, 8=n2
        h2o_vmr = torch.clamp(qv * (28.9647 / 18.01528), min=1.0e-7)
        strat_fall = torch.pow(torch.clamp(p_c / 8000.0, max=1.0), 0.38)
        vmrs = torch.stack([
            torch.ones_like(p_c),
            h2o_vmr,
            torch.full_like(p_c, max(float(co2_ppmv), 0.0) * 1.0e-6),
            o3 if o3_lw_coupling else torch.zeros_like(o3),
            torch.full_like(p_c, max(float(n2o_ppmv), 0.0) * 1.0e-6) * strat_fall,
            torch.full_like(p_c, CO_VMR_BG),
            torch.full_like(p_c, max(float(ch4_ppmv), 0.0) * 1.0e-6) * strat_fall,
            torch.full_like(p_c, C.O2_MIXING_RATIO),
            torch.full_like(p_c, N2_VMR),
        ], dim=0)

        # ---- A. Longwave 128-g-point Official RRTMGP ----
        tau_lw_gas, pfrac, _ = _eval_gas_optics(tabs.lw, p_c, t_c, col_dry_e20, vmrs, want_planck=True)
        assert pfrac is not None and tabs.lw.totplnk is not None and tabs.lw.diff_angle_bnd is not None
        # Add cloud + dust + aerosol LW optical depth mapped to 128 g-points
        tau_lw_cld_b = (k_liq_lw_b + k_liq_lw_r / rel.unsqueeze(-1)) * lwp_gm2.unsqueeze(-1) + (k_ice_lw_b + k_ice_lw_r / rei.unsqueeze(-1)) * iwp_gm2.unsqueeze(-1)
        if dm_dust is not None:
            tau_lw_cld_b = tau_lw_cld_b + k_dust_lw_b * dm_dust.unsqueeze(-1)
        if ao_td is not None:
            tau_lw_cld_b = tau_lw_cld_b + torch.clamp(ao_td[3][:, sl], min=0.0).unsqueeze(-1)
        g2b_lw = tabs.lw.gpt_to_bnd
        tau_lw_tot = tau_lw_gas + tau_lw_cld_b[:, :, g2b_lw]
        diff_g = tabs.lw.diff_angle_bnd[g2b_lw].view(1, 1, 128)
        dtau_lw = torch.clamp(diff_g * tau_lw_tot, min=0.0, max=50.0)
        trans_lw = torch.exp(-dtau_lw)
        em_lw = -torch.expm1(-dtau_lw)

        # Exact Planck source per g-point from official totplnk [16, 196] * plank_fraction [N, M, 128]
        b_lev_bnd = _interp_totplnk(tabs.lw.totplnk, t_w)  # [N+1, M, 16]
        b_sfc_bnd = _interp_totplnk(tabs.lw.totplnk, ts_all[sl])  # [M, 16]
        b_top_g = b_lev_bnd[:-1][:, :, g2b_lw] * pfrac
        b_bot_g = b_lev_bnd[1:][:, :, g2b_lw] * pfrac
        b_sfc_g = b_sfc_bnd[:, g2b_lw] * pfrac[-1]

        w_lin = torch.where(dtau_lw > 1.0e-3, (1.0 / torch.clamp(dtau_lw, min=1.0e-3)) - trans_lw / torch.clamp(em_lw, min=1.0e-6), 0.5 - dtau_lw / 12.0)
        s_dn = em_lw * ((1.0 - w_lin) * b_bot_g + w_lin * b_top_g)
        s_up = em_lw * ((1.0 - w_lin) * b_top_g + w_lin * b_bot_g)

        f_dn_list = [torch.zeros((M, 128), device=dev, dtype=dtype)]
        for k in range(N):
            f_dn_list.append(f_dn_list[-1] * trans_lw[k] + s_dn[k])
        em_s = em_all[sl].unsqueeze(-1)
        f_up_rev = [em_s * b_sfc_g + (1.0 - em_s) * f_dn_list[-1]]
        for k in range(N - 1, -1, -1):
            f_up_rev.append(f_up_rev[-1] * trans_lw[k] + s_up[k])
        f_up_rev.reverse()
        lw_dn_td[:, sl] = torch.stack(f_dn_list, dim=0).sum(dim=-1)
        lw_up_td[:, sl] = torch.stack(f_up_rev, dim=0).sum(dim=-1)

        # ---- B. Shortwave 112-g-point Official RRTMGP + Batched Toon 2-Stream ----
        # Always include true O3 in SW even if o3_lw_coupling is False
        vmrs[3] = o3
        tau_sw_gas, _, tau_rayl = _eval_gas_optics(tabs.sw, p_c, t_c, col_dry_e20, vmrs, want_rayl=True)
        assert tau_rayl is not None and tabs.sw.solar_frac_gpt is not None
        assert tabs.sw.is_vis_gpt is not None and tabs.sw.is_par_gpt is not None
        g2b_sw = tabs.sw.gpt_to_bnd

        # Band-level cloud + dust + 5-species aerosol SW optics [N, M, 14]
        tau_l_b = (chi * lwp_gm2 * (0.0028 + 1.50 / rel)).unsqueeze(-1).expand(N, M, 14)
        tau_i_b = (chi * iwp_gm2 * (0.0034 + 2.40 / rei)).unsqueeze(-1).expand(N, M, 14)
        tau_ext_b = tau_l_b + tau_i_b
        scat_b = tau_l_b * sw_cld_om_l + tau_i_b * sw_cld_om_i
        asym_b = tau_l_b * sw_cld_om_l * 0.85 + tau_i_b * sw_cld_om_i * 0.80
        if dm_dust is not None:
            td = sw_dust_kext_b * dm_dust.unsqueeze(-1)
            tau_ext_b = tau_ext_b + td
            scat_b = scat_b + td * sw_dust_ssa_b
            asym_b = asym_b + td * sw_dust_ssa_b * 0.72
        if ao_td is not None:
            tau_ext_b = tau_ext_b + sw_aero_sc * torch.clamp(ao_td[0][:, sl], min=0.0).unsqueeze(-1)
            scat_b = scat_b + sw_aero_sc * torch.clamp(ao_td[1][:, sl], min=0.0).unsqueeze(-1)
            asym_b = asym_b + sw_aero_sc * ao_td[2][:, sl].unsqueeze(-1)

        tau_sw_tot = tau_sw_gas + tau_rayl + tau_ext_b[:, :, g2b_sw]
        scat_sw_tot = tau_rayl + scat_b[:, :, g2b_sw]
        om_sw = torch.clamp(scat_sw_tot / torch.clamp(tau_sw_tot, min=1.0e-8), 0.0, 0.999999)
        g_sw = torch.clamp(asym_b[:, :, g2b_sw] / torch.clamp(scat_sw_tot, min=1.0e-8), -0.95, 0.95)

        # Batched Toon 2-stream across all M * 112 (col, g-point) columns simultaneously!
        M112 = M * 112
        mu0_b = mu0_all[sl].unsqueeze(-1).expand(M, 112).reshape(M112)
        f_toa_b = (stoa_all[sl].unsqueeze(-1) * tabs.sw.solar_frac_gpt.unsqueeze(0)).reshape(M112)
        vis_m = tabs.sw.is_vis_gpt.unsqueeze(0)
        alb_dir_b = torch.where(vis_m, av_all[sl].unsqueeze(-1), an_all[sl].unsqueeze(-1)).reshape(M112)
        alb_dif_b = torch.where(vis_m, avf_all[sl].unsqueeze(-1), anf_all[sl].unsqueeze(-1)).reshape(M112)

        u_sw, d_sw, s_dir, s_dif = toon_two_stream_sw(
            tau_sw_tot.reshape(N, M112),
            om_sw.reshape(N, M112),
            g_sw.reshape(N, M112),
            mu0_b,
            f_toa_b,
            alb_dir_b,
            albedo_sfc_dif=alb_dif_b,
            return_sfc_split=True,
        )
        sw_up_td[:, sl] = u_sw.reshape(N + 1, M, 112).sum(dim=-1)
        sw_dn_td[:, sl] = d_sw.reshape(N + 1, M, 112).sum(dim=-1)
        s_dir_g = s_dir.reshape(M, 112)
        s_dif_g = s_dif.reshape(M, 112)
        vis_f = tabs.sw.is_vis_gpt.to(dtype).unsqueeze(0)
        par_f = tabs.sw.is_par_gpt.to(dtype).unsqueeze(0)
        s1_dir_all[sl] = (s_dir_g * vis_f).sum(dim=-1)
        s1_dif_all[sl] = (s_dif_g * vis_f).sum(dim=-1)
        s2_dir_all[sl] = (s_dir_g * (1.0 - vis_f)).sum(dim=-1)
        s2_dif_all[sl] = (s_dif_g * (1.0 - vis_f)).sum(dim=-1)
        par_all[sl] = ((s_dir_g + s_dif_g) * par_f).sum(dim=-1)

    # Convert top-down -> AOBA bottom-up and restore original column shape
    sh_n1 = (N + 1,) + orig_shape
    sh_n = (N,) + orig_shape
    sw_up = torch.flip(sw_up_td, dims=[0]).reshape(sh_n1)
    sw_dn = torch.flip(sw_dn_td, dims=[0]).reshape(sh_n1)
    lw_up = torch.flip(lw_up_td, dims=[0]).reshape(sh_n1)
    lw_dn = torch.flip(lw_dn_td, dims=[0]).reshape(sh_n1)
    dm_bu = torch.clamp(rho_d_c * dz_c, min=1.0e-5)
    q_sw = ((sw_dn - sw_up)[1:] - (sw_dn - sw_up)[:-1]) / (C.CP_D * dm_bu)
    q_lw = ((lw_dn - lw_up)[1:] - (lw_dn - lw_up)[:-1]) / (C.CP_D * dm_bu)
    sfc_par = par_all.reshape(orig_shape)

    return RadiationOutput(
        sw_flux_up=sw_up,
        sw_flux_dn=sw_dn,
        lw_flux_up=lw_up,
        lw_flux_dn=lw_dn,
        q_rad_sw=q_sw,
        q_rad_lw=q_lw,
        q_rad_net=q_sw + q_lw,
        toa_insolation=sw_dn[-1],
        toa_asr=sw_dn[-1] - sw_up[-1],
        toa_olr=lw_up[-1],
        sfc_sw_net=sw_dn[0] - sw_up[0],
        sfc_lw_net=lw_dn[0] - lw_up[0],
        sfc_par_wm2=sfc_par,
        sfc_ppfd=C.UMOL_PER_JOULE_PAR * sfc_par,
        sfc_sw1_dir_dn=s1_dir_all.reshape(orig_shape),
        sfc_sw1_dif_dn=s1_dif_all.reshape(orig_shape),
        sfc_sw2_dir_dn=s2_dir_all.reshape(orig_shape),
        sfc_sw2_dif_dn=s2_dif_all.reshape(orig_shape),
    )
