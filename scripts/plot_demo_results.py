#!/usr/bin/env python3
"""Generate publication-grade diagnostic figures & quantitative validation for AOBA.

Reads:
- ~/aoba_outputs/aoba_g3_demo_diagnostics.pt (from scripts/run_demo_simulation.py)
- data/validation/koppen_geiger/1991_2020/koppen_geiger_0p5.tif (Beck et al., 2023)
- data/validation/koppen_geiger/legend.txt

Writes 6 high-resolution PNG figures into the specified output directory (e.g. Antigravity
artifact directory) and prints a quantitative climate/circulation/biome summary.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
import torch

from aoba import constants as C
from aoba.koppen import KOPPEN_NAMES
from aoba.runtime import get_output_dir
from aoba.surface import SURF_ICESHEET, SURF_LAND, SURF_OCEAN


# Official Beck et al. (2023) 30-class RGB palette (parsed from legend.txt / Table 1)
BECK_RGB_255: dict[int, tuple[int, int, int]] = {
    1: (0, 0, 255),       2: (0, 120, 255),     3: (70, 170, 250),
    4: (255, 0, 0),       5: (255, 150, 150),   6: (245, 165, 0),     7: (255, 220, 100),
    8: (255, 255, 0),     9: (200, 200, 0),     10: (150, 150, 0),
    11: (150, 255, 150),  12: (100, 200, 100),  13: (50, 150, 50),
    14: (200, 255, 80),   15: (100, 255, 80),   16: (50, 200, 0),
    17: (255, 0, 255),    18: (200, 0, 200),    19: (150, 50, 150),   20: (150, 100, 150),
    21: (170, 175, 255),  22: (90, 120, 220),   23: (75, 80, 180),    24: (50, 0, 135),
    25: (0, 255, 255),    26: (55, 200, 255),   27: (0, 125, 125),    28: (0, 70, 95),
    29: (178, 178, 178),  30: (102, 102, 102),
}

MAJOR_NAMES = {
    1: "A (Tropical)",
    2: "B (Arid)",
    3: "C (Temperate)",
    4: "D (Cold)",
    5: "E (Polar)",
}
MAJOR_COLORS = {
    1: "#1a75ff",
    2: "#f57c00",
    3: "#66bb6a",
    4: "#26c6da",
    5: "#8d8d8d",
}


def code30_to_major5(code: np.ndarray) -> np.ndarray:
    """Map Beck et al. (2018, 2023) integer class 1..30 to major group 1..5 (0 stays 0)."""
    maj = np.zeros_like(code, dtype=np.int64)
    maj[(code >= 1) & (code <= 3)] = 1
    maj[(code >= 4) & (code <= 7)] = 2
    maj[(code >= 8) & (code <= 16)] = 3
    maj[(code >= 17) & (code <= 28)] = 4
    maj[(code >= 29) & (code <= 30)] = 5
    return maj


def build_voronoi_polygons_platecarree(
    lat_cell_deg: np.ndarray,
    lon_cell_deg: np.ndarray,
    verts_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    lat_vert_deg: np.ndarray,
    lon_vert_deg: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build antimeridian-safe PlateCarree polygons `(polys, cell_indices)` for all SCVT cells."""
    polys: list[np.ndarray] = []
    owners: list[int] = []

    for i in range(len(lat_cell_deg)):
        nv = int(n_edges_on_cell[i])
        v_ids = verts_on_cell[i, :nv]
        v_lat = lat_vert_deg[v_ids].astype(np.float64)
        v_lon = lon_vert_deg[v_ids].astype(np.float64)
        c_lon = float(lon_cell_deg[i])
        c_lat = float(lat_cell_deg[i])

        # Polar cap cell (contains North or South Pole): build clean lat-lon cap rectangle
        if abs(c_lat) > 84.5:
            lat_edge = float(np.mean(v_lat))
            if c_lat > 0:
                cap = np.array([[-179.9, lat_edge], [179.9, lat_edge], [179.9, 89.95], [-179.9, 89.95]])
            else:
                cap = np.array([[-179.9, -89.95], [179.9, -89.95], [179.9, lat_edge], [-179.9, lat_edge]])
            polys.append(cap)
            owners.append(i)
            continue

        # Unwrap vertex longitudes around cell center longitude c_lon so dlon in [-180, 180]
        dlon = ((v_lon - c_lon + 180.0) % 360.0) - 180.0
        unwrapped_lon = c_lon + dlon

        # Order vertices counter-clockwise in local tangent plane
        ang = np.arctan2(v_lat - c_lat, dlon * max(math.cos(math.radians(c_lat)), 0.1))
        order = np.argsort(ang)
        ulon = unwrapped_lon[order]
        ulat = v_lat[order]

        if ulon.min() >= -180.0 and ulon.max() <= 180.0:
            polys.append(np.column_stack([ulon, ulat]))
            owners.append(i)
        elif ulon.max() > 180.0:
            # Split across +180 antimeridian by emitting clamped East copy and wrapped West copy
            p_east = np.column_stack([np.clip(ulon, -180.0, 179.99), ulat])
            p_west = np.column_stack([np.clip(ulon - 360.0, -179.99, 180.0), ulat])
            polys.append(p_east)
            owners.append(i)
            polys.append(p_west)
            owners.append(i)
        else:
            # ulon.min() < -180.0
            p_west = np.column_stack([np.clip(ulon, -179.99, 180.0), ulat])
            p_east = np.column_stack([np.clip(ulon + 360.0, -180.0, 179.99), ulat])
            polys.append(p_west)
            owners.append(i)
            polys.append(p_east)
            owners.append(i)

    return polys, np.array(owners, dtype=np.int64)


def plot_voronoi_robinson(
    ax,
    polys: list[np.ndarray],
    owners: np.ndarray,
    values: np.ndarray,
    cmap,
    vmin: float,
    vmax: float,
    mask: np.ndarray | None = None,
    bg_color: str = "#e6eef5",
    edge_lw: float = 0.15,
):
    """Render SCVT Voronoi cells onto a Cartopy Robinson axis."""
    ax.set_global()
    ax.set_facecolor(bg_color)
    if mask is not None:
        keep = mask[owners]
        sub_polys = [polys[k] for k in range(len(polys)) if keep[k]]
        sub_vals = values[owners[keep]]
    else:
        sub_polys = polys
        sub_vals = values[owners]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    coll = PolyCollection(
        sub_polys,
        array=sub_vals,
        cmap=cmap,
        norm=norm,
        edgecolors=(0, 0, 0, 0.12) if edge_lw > 0 else "none",
        linewidths=edge_lw,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.add_collection(coll)
    ax.coastlines(linewidth=0.55, color="#222222", zorder=4)
    gl = ax.gridlines(linewidth=0.3, color="#888888", alpha=0.45, linestyle=":")
    return coll


def zonal_bin_2d(
    field_nz_nc: np.ndarray,
    lat_deg: np.ndarray,
    area_cell: np.ndarray,
    n_bins: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted zonal average of `[Nz, N_cells]` field into `n_bins` latitude bands."""
    edges = np.linspace(-90.0, 90.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nz = field_nz_nc.shape[0]
    out = np.zeros((nz, n_bins), dtype=np.float64)
    for b in range(n_bins):
        if b < n_bins - 1:
            sel = (lat_deg >= edges[b]) & (lat_deg < edges[b + 1])
        else:
            sel = (lat_deg >= edges[b]) & (lat_deg <= edges[b + 1])
        if np.any(sel):
            w = area_cell[sel]
            out[:, b] = (field_nz_nc[:, sel] * w[None, :]).sum(axis=1) / w.sum()
    return centers, out


def zonal_bin_1d(
    field_nc: np.ndarray,
    lat_deg: np.ndarray,
    area_cell: np.ndarray,
    n_bins: int = 18,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted zonal average of `[N_cells]` field into `n_bins` latitude bands."""
    edges = np.linspace(-90.0, 90.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        if b < n_bins - 1:
            sel = (lat_deg >= edges[b]) & (lat_deg < edges[b + 1])
        else:
            sel = (lat_deg >= edges[b]) & (lat_deg <= edges[b + 1])
        if mask is not None:
            sel = sel & mask
        if np.any(sel):
            w = area_cell[sel]
            out[b] = (field_nc[sel] * w).sum() / w.sum()
    return centers, out


def compute_meridional_streamfunction(
    v_nz_nc: np.ndarray,
    pres_nz_nc: np.ndarray,
    temp_nz_nc: np.ndarray,
    dz_nz_nc: np.ndarray,
    lat_deg: np.ndarray,
    area_cell: np.ndarray,
    n_bins: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mass-conserving Eulerian meridional streamfunction Psi_M [10^10 kg s^-1]."""
    rho = pres_nz_nc / (C.R_D * np.maximum(temp_nz_nc, 150.0))
    rv_dz = rho * v_nz_nc * dz_nz_nc  # [Nz, N_cells] [kg m^-1 s^-1]
    lat_c, rv_zon = zonal_bin_2d(rv_dz, lat_deg, area_cell, n_bins=n_bins)
    # Remove tiny barotropic column-mean net mass drift so Psi_M(top) == Psi_M(bottom) == 0
    col_sum = rv_zon.sum(axis=0, keepdims=True)
    dz_zon = zonal_bin_2d(dz_nz_nc, lat_deg, area_cell, n_bins=n_bins)[1]
    rv_corr = rv_zon - col_sum * (dz_zon / np.maximum(dz_zon.sum(axis=0, keepdims=True), 1.0))
    cos_lat = np.cos(np.radians(lat_c))[None, :]
    psi = 2.0 * math.pi * C.EARTH_RADIUS * cos_lat * np.cumsum(rv_corr, axis=0) * 1.0e-10
    return lat_c, psi


def aggregate_beck_to_scvt(
    tif_path: Path,
    lat_cell_deg: np.ndarray,
    lon_cell_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Modal-aggregate Beck et al. (2023) 0.5-deg GeoTIFF onto SCVT Voronoi cells."""
    beck_0p5 = np.array(Image.open(tif_path))  # [360, 720], uint8, 0=ocean, 1..30=classes
    lats = 89.75 - 0.5 * np.arange(360)
    lons = -179.75 + 0.5 * np.arange(720)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Unit-sphere nearest Voronoi generator assignment for all land pixels (beck_0p5 > 0)
    valid = beck_0p5 > 0
    v_lat_r = np.radians(lat2d[valid])
    v_lon_r = np.radians(lon2d[valid])
    v_codes = beck_0p5[valid].astype(np.int64)
    v_weights = np.cos(v_lat_r)

    px_xyz = np.stack(
        [np.cos(v_lat_r) * np.cos(v_lon_r), np.cos(v_lat_r) * np.sin(v_lon_r), np.sin(v_lat_r)],
        axis=1,
    )  # [N_px, 3]

    c_lat_r = np.radians(lat_cell_deg)
    c_lon_r = np.radians(lon_cell_deg)
    cell_xyz = np.stack(
        [np.cos(c_lat_r) * np.cos(c_lon_r), np.cos(c_lat_r) * np.sin(c_lon_r), np.sin(c_lat_r)],
        axis=1,
    )  # [N_cells, 3]

    # Chunked dot-product nearest-cell assignment
    nc = len(lat_cell_deg)
    counts_30 = np.zeros((nc, 31), dtype=np.float64)
    chunk = 20000
    for i0 in range(0, len(v_codes), chunk):
        i1 = min(i0 + chunk, len(v_codes))
        dots = px_xyz[i0:i1] @ cell_xyz.T
        nearest = np.argmax(dots, axis=1)
        np.add.at(counts_30, (nearest, v_codes[i0:i1]), v_weights[i0:i1])

    modal_code30 = np.zeros(nc, dtype=np.int64)
    has_obs = counts_30[:, 1:].sum(axis=1) > 0
    modal_code30[has_obs] = 1 + np.argmax(counts_30[has_obs, 1:], axis=1)
    return modal_code30, beck_0p5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diag",
        type=str,
        default=str(get_output_dir() / "aoba_g3_demo_diagnostics.pt"),
    )
    parser.add_argument(
        "--beck-tif",
        type=str,
        default="data/validation/koppen_geiger/1991_2020/koppen_geiger_0p5.tif",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "figure.dpi": 220,
    })

    d = torch.load(args.diag, map_location="cpu", weights_only=False)
    lat_deg = d["lat_deg"].numpy()
    lon_deg = d["lon_deg"].numpy()
    area = d["area_cell"].numpy()
    surf_type = d["surf_type"].numpy()
    z_topo = d["z_topo"].numpy()
    zeta_c_km = d["zeta_c"].numpy() * 1.0e-3
    dz_c = np.diff(d["zeta_w"].numpy())[:, None] * np.ones((1, len(lat_deg)))

    is_ocean = surf_type == SURF_OCEAN
    is_land_or_ice = surf_type != SURF_OCEAN
    is_veg_land = surf_type == SURF_LAND

    polys, owners = build_voronoi_polygons_platecarree(
        lat_deg,
        lon_deg,
        d["verts_on_cell"].numpy(),
        d["n_edges_on_cell"].numpy(),
        d["lat_vert_deg"].numpy(),
        d["lon_vert_deg"].numpy(),
    )

    # =========================================================================
    # FIGURE 1: Global Spinup, Seasonal Cycle & Radiative-Hydrological Balance
    # =========================================================================
    ts = {k: v.numpy() for k, v in d["ts"].items()}
    days = ts["day"]
    ann_mask_ts = (days >= 1.0) & (days <= 365.0)

    fig = plt.figure(figsize=(13.2, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)

    # (a) Surface temperature
    ax = fig.add_subplot(gs[0, 0])
    ax.axvspan(days[0], 0, color="#eeeeee", alpha=0.8, label="Spinup (60 d)")
    ax.plot(days, ts["t_sfc"] - 273.15, color="#c62828", lw=1.7, label="Global mean $T_{\\mathrm{sfc}}$")
    ax.axhline(float(ts["t_sfc"][ann_mask_ts].mean() - 273.15), color="#c62828", ls="--", lw=1.0, alpha=0.7,
               label=f"Annual mean ({ts['t_sfc'][ann_mask_ts].mean()-273.15:.1f} °C)")
    ax.axvline(0, color="#555555", ls=":", lw=1.0)
    ax.axvline(365, color="#555555", ls=":", lw=1.0)
    ax.set_xlabel("Simulation Day Relative to Jan 1 [days]")
    ax.set_ylabel("Surface Temperature [°C]")
    ax.set_title("(a) Global Mean Surface Temperature & Seasonal Cycle")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92)

    # (b) TOA ASR vs OLR
    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(days[0], 0, color="#eeeeee", alpha=0.8)
    ax.plot(days, ts["asr"], color="#ef6c00", lw=1.5, label="TOA Absorbed Solar (ASR)")
    ax.plot(days, ts["olr"], color="#1565c0", lw=1.5, label="TOA Outgoing LW (OLR)")
    ax.axvline(0, color="#555555", ls=":", lw=1.0)
    ax.axvline(365, color="#555555", ls=":", lw=1.0)
    asr_yr = float(ts["asr"][ann_mask_ts].mean())
    olr_yr = float(ts["olr"][ann_mask_ts].mean())
    ax.set_xlabel("Simulation Day Relative to Jan 1 [days]")
    ax.set_ylabel("Radiative Flux [$\\mathrm{W\\ m^{-2}}$]")
    ax.set_title(f"(b) TOA Radiative Balance (Annual ASR={asr_yr:.1f}, OLR={olr_yr:.1f} $\\mathrm{{W\\ m^{{-2}}}}$)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92)

    # (c) Zonal-mean TOA ASR vs OLR & Meridional Heat Transport implied
    ax = fig.add_subplot(gs[1, 0])
    lat_b, asr_z = zonal_bin_1d(d["ann"]["asr"].numpy(), lat_deg, area, n_bins=18)
    _, olr_z = zonal_bin_1d(d["ann"]["olr"].numpy(), lat_deg, area, n_bins=18)
    ax.plot(lat_b, asr_z, color="#ef6c00", lw=2.0, marker="o", ms=3.5, label="Annual ASR($\\phi$)")
    ax.plot(lat_b, olr_z, color="#1565c0", lw=2.0, marker="s", ms=3.5, label="Annual OLR($\\phi$)")
    ax.fill_between(lat_b, olr_z, asr_z, where=(asr_z >= olr_z), color="#ffcc80", alpha=0.45, label="Tropical Net Heating")
    ax.fill_between(lat_b, olr_z, asr_z, where=(asr_z < olr_z), color="#90caf9", alpha=0.45, label="Polar Net Cooling")
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("Latitude [°N]")
    ax.set_ylabel("TOA Flux [$\\mathrm{W\\ m^{-2}}$]")
    ax.set_title("(c) Zonal-Mean Annual TOA Energy Budget")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    # (d) Global Precipitation & Polar Sea-Ice Extent
    ax = fig.add_subplot(gs[1, 1])
    ax.axvspan(days[0], 0, color="#eeeeee", alpha=0.8)
    l1 = ax.plot(days, ts["precip"], color="#00897b", lw=1.3, alpha=0.88,
                 label=f"Global Precip (mean {ts['precip'][ann_mask_ts].mean():.2f} mm/d)")
    ax.set_xlabel("Simulation Day Relative to Jan 1 [days]")
    ax.set_ylabel("Precipitation [$\\mathrm{mm\\ d^{-1}}$]", color="#00695c")
    ax.tick_params(axis="y", labelcolor="#00695c")
    ax.axvline(0, color="#555555", ls=":", lw=1.0)
    ax.axvline(365, color="#555555", ls=":", lw=1.0)
    ax.grid(True, ls=":", alpha=0.5)

    ax2 = ax.twinx()
    l2 = ax2.plot(days, ts["seaice"], color="#3949ab", lw=1.7, ls="-.",
                  label=f"Global Sea-Ice Area (mean {ts['seaice'][ann_mask_ts].mean():.1f} $\\mathrm{{Mkm^2}}$)")
    ax2.set_ylabel("Sea-Ice Area [$10^6\\ \\mathrm{km^2}$]", color="#3949ab")
    ax2.tick_params(axis="y", labelcolor="#3949ab")
    lines = l1 + l2
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right", fontsize=8.0, framealpha=0.92)
    ax.set_title("(d) Global Hydrological Cycle & Polar Sea-Ice Area")

    p1 = out_dir / "fig1_spinup_and_energy_budget.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: Atmospheric General Circulation (Zonal Wind, Hadley/Ferrel, P & E)
    # =========================================================================
    fig = plt.figure(figsize=(14.0, 11.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 0.85], hspace=0.38, wspace=0.28)

    z_mask = zeta_c_km <= 28.0
    z_plot = zeta_c_km[z_mask]

    for col_i, (seas_key, seas_label) in enumerate([("djf", "DJF (Boreal Winter)"), ("jja", "JJA (Boreal Summer)")]):
        sd = d[seas_key]
        lat_b, u_z = zonal_bin_2d(sd["u"].numpy(), lat_deg, area, n_bins=18)
        _, t_z = zonal_bin_2d(sd["temp"].numpy(), lat_deg, area, n_bins=18)
        _, p_z = zonal_bin_2d(sd["pres"].numpy(), lat_deg, area, n_bins=18)
        theta_z = t_z * np.power(100000.0 / np.maximum(p_z, 10.0), C.R_D / C.CP_D)

        ax = fig.add_subplot(gs[0, col_i])
        u_levels = np.linspace(-25.0, 50.0, 16)
        cf = ax.contourf(lat_b, z_plot, u_z[z_mask], levels=u_levels, cmap="RdBu_r",
                         norm=mcolors.TwoSlopeNorm(vmin=-25.0, vcenter=0.0, vmax=50.0), extend="both")
        cs0 = ax.contour(lat_b, z_plot, u_z[z_mask], levels=[0.0], colors="k", linewidths=1.3)
        cs_th = ax.contour(lat_b, z_plot, theta_z[z_mask], levels=np.arange(260, 420, 20),
                           colors="#333333", linewidths=0.65, linestyles="--", alpha=0.75)
        ax.clabel(cs_th, fmt="%d K", fontsize=7.0)
        cb = fig.colorbar(cf, ax=ax, pad=0.02)
        cb.set_label("$\\bar{u}$ [$\\mathrm{m\\ s^{-1}}$]")
        ax.set_xlim(-85, 85)
        ax.set_ylim(0, 28)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_ylabel("Altitude [km]")
        panel_ltr = "a" if col_i == 0 else "b"
        ax.set_title(f"({panel_ltr}) {seas_label}: Zonal Wind $\\bar{{u}}$ & $\\bar{{\\theta}}$")
        ax.grid(True, ls=":", alpha=0.35)

        # Row 2: Meridional Streamfunction Psi_M & Cloud Condensate
        z_m2 = zeta_c_km <= 20.0
        lat_psi, psi_m = compute_meridional_streamfunction(
            sd["v"].numpy(), sd["pres"].numpy(), sd["temp"].numpy(), dz_c, lat_deg, area, n_bins=18
        )
        _, qc_z = zonal_bin_2d(sd["qcond"].numpy() * 1000.0, lat_deg, area, n_bins=18)

        axm = fig.add_subplot(gs[1, col_i])
        p_max = max(float(np.percentile(np.abs(psi_m[z_m2]), 98)) * 1.15, 8.0)
        p_lev = np.linspace(-p_max, p_max, 17)
        cfm = axm.contourf(lat_psi, zeta_c_km[z_m2], psi_m[z_m2], levels=p_lev, cmap="PuOr_r", extend="both")
        axm.contour(lat_psi, zeta_c_km[z_m2], psi_m[z_m2], levels=[0.0], colors="k", linewidths=1.1)
        axm.contour(lat_psi, zeta_c_km[z_m2], qc_z[z_m2], levels=[0.015, 0.04, 0.08],
                    colors="#00695c", linewidths=0.95, linestyles="-")
        cbm = fig.colorbar(cfm, ax=axm, pad=0.02)
        cbm.set_label("$\\Psi_M$ [$10^{10}\\ \\mathrm{kg\\ s^{-1}}$]")
        axm.set_xlim(-85, 85)
        axm.set_ylim(0, 20)
        axm.set_xticks([-60, -30, 0, 30, 60])
        axm.set_xlabel("Latitude [°N]")
        axm.set_ylabel("Altitude [km]")
        panel_ltr2 = "c" if col_i == 0 else "d"
        short_s = "DJF" if col_i == 0 else "JJA"
        axm.set_title(f"({panel_ltr2}) {short_s}: Hadley/Ferrel $\\Psi_M$ & Clouds (green)")
        axm.grid(True, ls=":", alpha=0.35)

    # Row 3 (spanning both columns): Zonal-Mean Precipitation (Annual, DJF, JJA) vs Evaporation
    axp = fig.add_subplot(gs[2, :])
    lat_b, p_ann_z = zonal_bin_1d(d["ann"]["precip"].numpy(), lat_deg, area, n_bins=24)
    _, p_djf_z = zonal_bin_1d(d["djf"]["precip"].numpy(), lat_deg, area, n_bins=24)
    _, p_jja_z = zonal_bin_1d(d["jja"]["precip"].numpy(), lat_deg, area, n_bins=24)
    _, e_ann_z = zonal_bin_1d(d["ann"]["evap"].numpy(), lat_deg, area, n_bins=24)

    axp.plot(lat_b, p_ann_z, color="#0d47a1", lw=2.3, label="Annual Mean Precipitation $P$")
    axp.plot(lat_b, p_djf_z, color="#1e88e5", lw=1.5, ls="--", label="DJF Precipitation (ITCZ south of Eq.)")
    axp.plot(lat_b, p_jja_z, color="#e53935", lw=1.5, ls="-.", label="JJA Precipitation (ITCZ north of Eq.)")
    axp.plot(lat_b, e_ann_z, color="#ef6c00", lw=2.0, ls=":", label="Annual Mean Evaporation $E$")
    axp.fill_between(lat_b, e_ann_z, p_ann_z, where=(p_ann_z >= e_ann_z), color="#90caf9", alpha=0.30, label="$P > E$ (ITCZ & Storm Tracks)")
    axp.fill_between(lat_b, e_ann_z, p_ann_z, where=(p_ann_z < e_ann_z), color="#ffe0b2", alpha=0.38, label="$P < E$ (Subtropical Dry Zones)")
    axp.set_xlim(-90, 90)
    axp.set_ylim(bottom=0.0)
    axp.set_xticks([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])
    axp.set_xlabel("Latitude [°N]")
    axp.set_ylabel("Water Flux [$\\mathrm{mm\\ d^{-1}}$]")
    axp.set_title("(e) Zonal-Mean Hydrological Balance: ITCZ Migration, Subtropical Sub-Humid Belts & Storm Tracks")
    axp.grid(True, ls=":", alpha=0.5)
    axp.legend(ncol=3, loc="upper right", fontsize=8.3, framealpha=0.92)

    p2 = out_dir / "fig2_atmospheric_circulation.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: Global Surface Climate, Precipitation, Cryosphere & Ekman Ocean
    # =========================================================================
    fig = plt.figure(figsize=(14.2, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.08)

    # (a) DJF Surface Air Temperature [°C]
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    c1 = plot_voronoi_robinson(ax, polys, owners, d["djf"]["t_air"].numpy() - 273.15, "RdYlBu_r", -35.0, 35.0)
    cb = fig.colorbar(c1, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="both")
    cb.set_label("DJF Near-Surface Air Temperature [°C]")
    ax.set_title("(a) Boreal Winter (DJF) Air Temperature")

    # (b) JJA Surface Air Temperature [°C]
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    c2 = plot_voronoi_robinson(ax, polys, owners, d["jja"]["t_air"].numpy() - 273.15, "RdYlBu_r", -35.0, 35.0)
    cb = fig.colorbar(c2, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="both")
    cb.set_label("JJA Near-Surface Air Temperature [°C]")
    ax.set_title("(b) Boreal Summer (JJA) Air Temperature")

    # (c) Annual Total Precipitation [mm yr^-1]
    ax = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
    p_ann_mmyr = d["ann"]["precip"].numpy() * 365.2425
    c3 = plot_voronoi_robinson(ax, polys, owners, p_ann_mmyr, "YlGnBu", 0.0, 2600.0)
    cb = fig.colorbar(c3, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="max")
    cb.set_label("Annual Total Precipitation [$\\mathrm{mm\\ yr^{-1}}$]")
    ax.set_title("(c) Annual Precipitation (ITCZ, Monsoons & Storm Tracks)")

    # (d) Zero-Q-Flux Ekman Ocean Upwelling [m yr^-1] over Ocean + Snow/Ice Surface Albedo on Land/Ice
    ax = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
    # Ekman vertical velocity w_E [m day^-1] = W_E [kg m^-2 s^-1] / 1025 * 86400 -> m yr^-1
    w_ek_myr = (d["ann"]["w_ekman"].numpy() / C.RHO_SEAWATER) * C.SECONDS_PER_DAY * 365.2425
    c4 = plot_voronoi_robinson(
        ax, polys, owners, w_ek_myr, "BrBG_r", -60.0, 60.0, mask=is_ocean, bg_color="#dcd7cd"
    )
    # Mark sea-ice edge (f_seaice >= 0.15) with stipple circles
    ice_cells = is_ocean & (d["ann"]["f_seaice"].numpy() >= 0.15)
    if np.any(ice_cells):
        ax.scatter(
            lon_deg[ice_cells], lat_deg[ice_cells], s=14, facecolors="none", edgecolors="#0d47a1",
            linewidths=0.9, transform=ccrs.PlateCarree(), zorder=5, label="Sea Ice ($f_{\\mathrm{ice}}\\geq 0.15$)"
        )
        ax.legend(loc="lower left", fontsize=7.8, framealpha=0.9)
    cb = fig.colorbar(c4, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="both")
    cb.set_label("Wind-Driven Ekman Upwelling $w_E$ [$\\mathrm{m\\ yr^{-1}}$] (>0 Upwelling)")
    ax.set_title("(d) Codron Zero-$Q$-Flux Ocean Upwelling & Polar Sea Ice")

    p3 = out_dir / "fig3_global_surface_and_ocean.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 4: Emergent Autonomous Two-Leaf EEO Terrestrial Biosphere
    # =========================================================================
    bio = {k: v.numpy() for k, v in d["biosphere"].items()}
    fig = plt.figure(figsize=(14.6, 13.4))
    gs = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.08)

    # (a) Peak / Growing-Season LAI [m^2 m^-2]
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ca = plot_voronoi_robinson(
        ax, polys, owners, bio["lai_peak"], "YlGn", 0.0, 6.0, mask=is_land_or_ice, bg_color="#dceaf7"
    )
    cb = fig.colorbar(ca, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="max")
    cb.set_label("Seasonal Peak Leaf Area Index [$\\mathrm{m^2\\ m^{-2}}$]")
    ax.set_title("(a) Emergent Prognostic LAI (Cai 2025 + Zhou 2025 Lambert-$W_0$)")

    # (b) Seasonal Phenology Amplitude: JJA minus DJF LAI [m^2 m^-2]
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    dlai_seas = bio["lai_jja"] - bio["lai_djf"]
    cb_col = plot_voronoi_robinson(
        ax, polys, owners, dlai_seas, "PiYG", -3.0, 3.0, mask=is_land_or_ice, bg_color="#dceaf7"
    )
    cb = fig.colorbar(cb_col, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="both")
    cb.set_label("$\\Delta\\mathrm{LAI}_{\\mathrm{JJA - DJF}}$ [$\\mathrm{m^2\\ m^{-2}}$]")
    ax.set_title("(b) Autonomous Seasonal Leaf Phenology (JJA $-$ DJF)")

    # (c) Annual Gross Primary Production [g C m^-2 yr^-1]
    # 1 umol CO2 m^-2 s^-1 = 12.011e-6 * 86400 * 365.2425 = 379.03 g C m^-2 yr^-1
    gpp_gcyr = bio["gpp_ann_mean"] * (12.011e-6 * C.SECONDS_PER_DAY * C.DAYS_PER_YEAR)
    ax = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
    cc = plot_voronoi_robinson(
        ax, polys, owners, gpp_gcyr, "viridis", 0.0, 2400.0, mask=is_land_or_ice, bg_color="#dceaf7"
    )
    cb = fig.colorbar(cc, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="max")
    cb.set_label("Annual GPP [$\\mathrm{g\\ C\\ m^{-2}\\ yr^{-1}}$]")
    ax.set_title("(c) Two-Leaf NRH P-Model Gross Primary Production")

    # (d) Potential Vascular Canopy Height h_c [m] (Kempes et al., 2011)
    ax = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
    cd = plot_voronoi_robinson(
        ax, polys, owners, bio["h_canopy"], "YlGnBu", 0.0, 32.0, mask=is_land_or_ice, bg_color="#dceaf7"
    )
    cb = fig.colorbar(cd, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="max")
    cb.set_label("Potential Canopy Height $h_c$ [m]")
    ax.set_title("(d) Vascular Allometric Canopy Height (Kempes et al., 2011)")

    # (e) Acclimated Canopy Rubisco Capacity V_cmax25 (C3, Peak) [umol m^-2 s^-1]
    ax = fig.add_subplot(gs[2, 0], projection=ccrs.Robinson())
    ce = plot_voronoi_robinson(
        ax, polys, owners, bio["vcmax25_c3_peak"], "plasma", 0.0, 140.0, mask=is_land_or_ice, bg_color="#dceaf7"
    )
    cb = fig.colorbar(ce, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="max")
    cb.set_label("Canopy $V_{c\\max,25}$ [$\\mathrm{\\mu mol\\ m^{-2}\\ s^{-1}}$]")
    ax.set_title("(e) Smith et al. (2019) Acclimated Canopy $V_{c\\max,25}$")

    # (f) Autonomous C4 Photosynthetic Fraction F_4 [-] (Scott & Smith, 2022)
    ax = fig.add_subplot(gs[2, 1], projection=ccrs.Robinson())
    cf4 = plot_voronoi_robinson(
        ax, polys, owners, bio["f_c4_ann"], "OrRd", 0.0, 0.85, mask=is_land_or_ice, bg_color="#dceaf7"
    )
    cb = fig.colorbar(cf4, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, extend="max")
    cb.set_label("Autonomous $\\mathrm{C_4}$ Advantage Fraction $F_4$ [-]")
    ax.set_title("(f) Emergent $\\mathrm{C_4}$ Grassland/Savanna Belt (Zero PFT Input)")

    p4 = out_dir / "fig4_emergent_biosphere.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 5: Köppen-Geiger Comparison vs. Observed Beck et al. (2023, 1991-2020)
    # =========================================================================
    obs_code30_scvt, beck_0p5 = aggregate_beck_to_scvt(Path(args.beck_tif), lat_deg, lon_deg)
    sim_code30 = d["koppen"]["code_30"].numpy()
    sim_major5 = d["koppen"]["major_5"].numpy()
    obs_major5 = code30_to_major5(obs_code30_scvt)

    # Discrete 1..30 Beck RGB colormap
    rgb_list = [(0.88, 0.93, 0.97)] + [
        (BECK_RGB_255[k][0] / 255.0, BECK_RGB_255[k][1] / 255.0, BECK_RGB_255[k][2] / 255.0)
        for k in range(1, 31)
    ]
    beck_cmap = mcolors.ListedColormap(rgb_list)
    beck_norm = mcolors.BoundaryNorm(np.arange(-0.5, 31.5, 1.0), beck_cmap.N)

    fig = plt.figure(figsize=(14.6, 10.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.85], hspace=0.26, wspace=0.14)

    # (a) AOBA Emergent 30-Class Köppen-Geiger on SCVT Land/Ice Mesh
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("#dceaf7")
    keep_l = is_land_or_ice[owners]
    coll_sim = PolyCollection(
        [polys[k] for k in range(len(polys)) if keep_l[k]],
        array=sim_code30[owners[keep_l]],
        cmap=beck_cmap,
        norm=beck_norm,
        edgecolors=(0, 0, 0, 0.14),
        linewidths=0.18,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.add_collection(coll_sim)
    ax.coastlines(linewidth=0.55, color="#222222", zorder=4)
    ax.gridlines(linewidth=0.3, color="#888888", alpha=0.45, linestyle=":")
    ax.set_title("(a) AOBA Emergent Köppen–Geiger (Minimal Boundaries)")

    # (b) Observed Beck et al. (2023, 1991-2020) Modal-Aggregated onto Same SCVT Land/Ice Mesh
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("#dceaf7")
    valid_obs_mask = (obs_code30_scvt > 0) & is_land_or_ice
    keep_o = valid_obs_mask[owners]
    coll_obs = PolyCollection(
        [polys[k] for k in range(len(polys)) if keep_o[k]],
        array=obs_code30_scvt[owners[keep_o]],
        cmap=beck_cmap,
        norm=beck_norm,
        edgecolors=(0, 0, 0, 0.14),
        linewidths=0.18,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.add_collection(coll_obs)
    ax.coastlines(linewidth=0.55, color="#222222", zorder=4)
    ax.gridlines(linewidth=0.3, color="#888888", alpha=0.45, linestyle=":")
    ax.set_title("(b) Observed Beck et al. (2023, 1991–2020) on SCVT Mesh")

    # Compact 30-class Beck legend banner between top and bottom rows
    present_codes = sorted(
        set(np.unique(sim_code30[is_land_or_ice]).tolist())
        | set(np.unique(obs_code30_scvt[valid_obs_mask]).tolist())
    )
    present_codes = [c for c in present_codes if 1 <= c <= 30]
    patches_30 = [
        mpatches.Patch(
            facecolor=(BECK_RGB_255[c][0] / 255.0, BECK_RGB_255[c][1] / 255.0, BECK_RGB_255[c][2] / 255.0),
            edgecolor="#444444",
            linewidth=0.4,
            label=KOPPEN_NAMES[c],
        )
        for c in present_codes
    ]
    fig.legend(
        handles=patches_30,
        loc="center",
        bbox_to_anchor=(0.5, 0.455),
        ncol=min(len(patches_30), 15),
        fontsize=7.6,
        framealpha=0.95,
        handlelength=1.1,
        columnspacing=0.8,
        title="Beck et al. (2023) 30-Class Köppen–Geiger Legend",
        title_fontsize=8.2,
    )

    # (c) Global Land Area Shares (%) Across the 5 Major Köppen Groups (AOBA vs Beck 2023)
    eval_cells = valid_obs_mask
    w_eval = area[eval_cells]
    w_tot_eval = w_eval.sum()

    sim_shares = [100.0 * w_eval[sim_major5[eval_cells] == m].sum() / w_tot_eval for m in range(1, 6)]
    obs_shares = [100.0 * w_eval[obs_major5[eval_cells] == m].sum() / w_tot_eval for m in range(1, 6)]

    exact_major_acc = 100.0 * w_eval[sim_major5[eval_cells] == obs_major5[eval_cells]].sum() / w_tot_eval
    exact_30_acc = 100.0 * w_eval[sim_code30[eval_cells] == obs_code30_scvt[eval_cells]].sum() / w_tot_eval

    axc = fig.add_subplot(gs[1, 0])
    x = np.arange(5)
    bw = 0.36
    b1 = axc.bar(x - bw / 2, obs_shares, width=bw, color="#37474f", alpha=0.88, label="Beck et al. (2023) Observed")
    b2 = axc.bar(x + bw / 2, sim_shares, width=bw, color=[MAJOR_COLORS[m] for m in range(1, 6)],
                 edgecolor="#222222", linewidth=0.7, label="AOBA Emergent")
    axc.set_xticks(x)
    axc.set_xticklabels([MAJOR_NAMES[m] for m in range(1, 6)], fontsize=8.5)
    axc.set_ylabel("Fraction of Global Land Area [%]")
    axc.set_ylim(0, max(max(sim_shares), max(obs_shares)) * 1.24)
    axc.set_title(f"(c) Global Land Area by Major Biome Group (Major Match = {exact_major_acc:.1f}%)")
    axc.grid(True, axis="y", ls=":", alpha=0.5)
    axc.legend(loc="upper right", fontsize=8.3, framealpha=0.92)
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        axc.text(rect.get_x() + rect.get_width() / 2.0, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2)

    # (d) Latitudinal Dominant Major-Group Agreement / Fractional Composition
    axd = fig.add_subplot(gs[1, 1])
    lat_edges = np.linspace(-60.0, 80.0, 15)
    lat_mid = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    sim_lat_frac = np.zeros((5, len(lat_mid)))
    obs_lat_frac = np.zeros((5, len(lat_mid)))
    for bi in range(len(lat_mid)):
        sel = eval_cells & (lat_deg >= lat_edges[bi]) & (lat_deg < lat_edges[bi + 1])
        if np.any(sel):
            wb = area[sel]
            for m in range(1, 6):
                sim_lat_frac[m - 1, bi] = 100.0 * wb[sim_major5[sel] == m].sum() / wb.sum()
                obs_lat_frac[m - 1, bi] = 100.0 * wb[obs_major5[sel] == m].sum() / wb.sum()

    bottom_s = np.zeros(len(lat_mid))
    for m in range(1, 6):
        axd.bar(lat_mid - 1.8, obs_lat_frac[m - 1], width=3.2, bottom=bottom_s,
                color=MAJOR_COLORS[m], alpha=0.45, edgecolor="none")
        bottom_s += obs_lat_frac[m - 1]
    bottom_a = np.zeros(len(lat_mid))
    for m in range(1, 6):
        axd.bar(lat_mid + 1.8, sim_lat_frac[m - 1], width=3.2, bottom=bottom_a,
                color=MAJOR_COLORS[m], alpha=0.95, edgecolor="#222222", linewidth=0.35,
                label=MAJOR_NAMES[m] if bi == len(lat_mid) - 1 else None)
        bottom_a += sim_lat_frac[m - 1]

    axd.set_xlim(-62, 82)
    axd.set_ylim(0, 100)
    axd.set_xlabel("Latitude Band [°N] (Left faded = Observed, Right solid = AOBA)")
    axd.set_ylabel("Land Band Composition [%]")
    axd.set_title("(d) Zonal Biome Transitions (60°S–80°N: Obs vs AOBA)")
    axd.grid(True, axis="y", ls=":", alpha=0.4)
    axd.legend(loc="upper left", ncol=2, fontsize=7.4, framealpha=0.92)

    p5 = out_dir / "fig5_koppen_geiger_comparison.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 6: Active 1:1 Embedded 3D SP-CRM Subgrid Convection Showcase (MPS GPU)
    # =========================================================================
    sp = d["sp_crm"]
    w_crm = 0.5 * (sp["w"][:-1].numpy() + sp["w"][1:].numpy())  # [Nz, Nc, 8, 8]
    qc_crm = sp["q_cond"].numpy() * 1000.0                      # [g/kg]
    qp_crm = sp["q_p"].numpy() * 1000.0                         # [g/kg]
    th_crm = sp["theta_m_anom"].numpy()                         # [K]

    # Pick the most vigorously convecting tropical ocean cell, tropical land cell, & storm-track cell
    col_wmax = np.abs(w_crm).max(axis=(0, 2, 3))
    trop_ocn_idx = int(np.argmax(np.where(is_ocean & (np.abs(lat_deg) < 20.0), col_wmax, -1.0)))
    trop_lnd_idx = int(np.argmax(np.where(is_veg_land & (np.abs(lat_deg) < 20.0), col_wmax, -1.0)))

    fig = plt.figure(figsize=(15.2, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.44)
    x_km = (np.arange(8) + 0.5) * 4.0
    z_crm_m = zeta_c_km <= 16.0
    z_c16 = zeta_c_km[z_crm_m]

    for row_i, (c_idx, tag) in enumerate([
        (trop_ocn_idx, f"Tropical Ocean Cell ({lat_deg[trop_ocn_idx]:+.1f}°, {lon_deg[trop_ocn_idx]:+.1f}°)"),
        (trop_lnd_idx, f"Tropical Land Cell ({lat_deg[trop_lnd_idx]:+.1f}°, {lon_deg[trop_lnd_idx]:+.1f}°)"),
    ]):
        # Slice through peak updraft y-index
        w_col = w_crm[:, c_idx]
        iy_peak = int(np.unravel_index(np.argmax(np.abs(w_col)), w_col.shape)[1])

        # Col 0: x-z vertical velocity w & cloud condensate contours
        ax = fig.add_subplot(gs[row_i, 0])
        wm = max(float(np.abs(w_col[z_crm_m, iy_peak]).max()) * 1.05, 0.3)
        cfw = ax.contourf(x_km, z_c16, w_col[z_crm_m, iy_peak], levels=np.linspace(-wm, wm, 17),
                          cmap="RdBu_r", extend="both")
        q_slice = qc_crm[z_crm_m, c_idx, iy_peak]
        if q_slice.max() > 0.01:
            ax.contour(x_km, z_c16, q_slice, levels=np.linspace(0.02, max(float(q_slice.max()), 0.1), 4),
                       colors="#1b5e20", linewidths=1.1)
        cb = fig.colorbar(cfw, ax=ax, pad=0.02)
        cb.set_label("$w$ [$\\mathrm{m\\ s^{-1}}$]")
        ax.set_xlabel("CRM $x$ [km]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title(f"({'a' if row_i==0 else 'd'}) {tag}\n$x$–$z$ Updraft $w$ & Cloud $q_c+q_i$")
        ax.grid(True, ls=":", alpha=0.4)

        # Col 1: x-z buoyant theta_m' + precip shaft q_p
        ax = fig.add_subplot(gs[row_i, 1])
        th_sl = th_crm[z_crm_m, c_idx, iy_peak]
        tm = max(float(np.abs(th_sl).max()) * 1.05, 0.08)
        cft = ax.contourf(x_km, z_c16, th_sl, levels=np.linspace(-tm, tm, 17), cmap="Spectral_r", extend="both")
        qp_sl = qp_crm[z_crm_m, c_idx, iy_peak]
        if qp_sl.max() > 0.005:
            ax.contour(x_km, z_c16, qp_sl, levels=3, colors="#0d47a1", linewidths=1.0, linestyles="--")
        cb = fig.colorbar(cft, ax=ax, pad=0.02)
        cb.set_label("$\\theta_m'$ [K]")
        ax.set_xlabel("CRM $x$ [km]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title(f"({'b' if row_i==0 else 'e'}) Buoyancy $\\theta_m'$ & Precip Shaft (blue)")
        ax.grid(True, ls=":", alpha=0.4)

        # Col 2: Horizontal plan view x-y of mid-tropospheric w (z ~ 3-5 km)
        iz_mid = int(np.argmin(np.abs(zeta_c_km - 3.5)))
        ax = fig.add_subplot(gs[row_i, 2])
        w_xy = w_col[iz_mid]
        wxy_m = max(float(np.abs(w_xy).max()) * 1.05, 0.2)
        im = ax.imshow(w_xy, origin="lower", extent=[0, 32, 0, 32], cmap="RdBu_r", vmin=-wxy_m, vmax=wxy_m)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("$w$ at 3.5 km [$\\mathrm{m\\ s^{-1}}$]")
        ax.set_xlabel("CRM $x$ [km]")
        ax.set_ylabel("CRM $y$ [km]")
        ax.set_title(f"({'c' if row_i==0 else 'f'}) 32×32 km Plan View ($z={zeta_c_km[iz_mid]:.1f}$ km)")

    p6 = out_dir / "fig6_sp_crm_subgrid_convection.png"
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)

    # Print Quantitative Summary JSON for the walkthrough report
    w_glob = area / area.sum()
    w_land = area[is_veg_land] / area[is_veg_land].sum()
    gpp_pg_yr = float((gpp_gcyr[is_veg_land] * area[is_veg_land]).sum() * 1.0e-15)

    summary = {
        "global_mean_t_sfc_c": round(float(ts["t_sfc"][ann_mask_ts].mean() - 273.15), 2),
        "global_mean_t_sfc_djf_c": round(float((d["djf"]["t_sfc"].numpy() * w_glob).sum() - 273.15), 2),
        "global_mean_t_sfc_jja_c": round(float((d["jja"]["t_sfc"].numpy() * w_glob).sum() - 273.15), 2),
        "global_mean_asr_wm2": round(asr_yr, 2),
        "global_mean_olr_wm2": round(olr_yr, 2),
        "global_mean_precip_mm_day": round(float(ts["precip"][ann_mask_ts].mean()), 2),
        "global_mean_precip_mm_yr": round(float(ts["precip"][ann_mask_ts].mean() * 365.2425), 1),
        "seaice_mean_mkm2": round(float(ts["seaice"][ann_mask_ts].mean()), 2),
        "djf_jet_max_ms": round(float(zonal_bin_2d(d["djf"]["u"].numpy(), lat_deg, area, 18)[1].max()), 1),
        "jja_jet_max_ms": round(float(zonal_bin_2d(d["jja"]["u"].numpy(), lat_deg, area, 18)[1].max()), 1),
        "land_mean_peak_lai": round(float((bio["lai_peak"][is_veg_land] * w_land).sum()), 2),
        "land_mean_ann_lai": round(float((bio["lai_ann_mean"][is_veg_land] * w_land).sum()), 2),
        "global_total_gpp_pgC_yr": round(gpp_pg_yr, 1),
        "land_mean_h_canopy_m": round(float((bio["h_canopy"][is_veg_land] * w_land).sum()), 2),
        "koppen_major5_exact_match_pct": round(float(exact_major_acc), 1),
        "koppen_30class_exact_match_pct": round(float(exact_30_acc), 1),
        "koppen_major_shares_obs_pct": {MAJOR_NAMES[m]: round(float(obs_shares[m - 1]), 1) for m in range(1, 6)},
        "koppen_major_shares_aoba_pct": {MAJOR_NAMES[m]: round(float(sim_shares[m - 1]), 1) for m in range(1, 6)},
        "sp_crm_max_updraft_ms": round(float(np.abs(w_crm).max()), 2),
    }
    print("=== AOBA QUANTITATIVE EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
