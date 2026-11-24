#!/usr/bin/env python3
"""Plot 6-panel AOBA Ocean & Coupled Spin-Up Monitoring Dashboard from `spinup_monitor.csv`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols: dict[str, np.ndarray] = {}
    for k in rows[0].keys():
        if k == "date":
            cols[k] = np.array([r[k] for r in rows])
        else:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[k]))
                except Exception:
                    vals.append(np.nan)
            cols[k] = np.array(vals, dtype=np.float64)
    return cols


def plot_spinup_dashboard(csv_path: Path, out_png: Path) -> None:
    d = _load_csv(csv_path)
    if not d or len(d["sim_days"]) < 2:
        return

    yr = d["sim_days"] / 365.2425
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), dpi=160, sharex=True)

    # 1. Surface & Sea-Surface Temperature
    ax = axes[0, 0]
    ax.plot(yr, d["ts_k"] - 273.15, lw=1.4, color="#1f77b4", label="Global $T_s$")
    ax.plot(yr, d["sst_global_k"] - 273.15, lw=1.4, color="#d62728", label="Global SST")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Global Surface & Sea-Surface Temperature")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=9)

    # 2. Planetary Energy Budget: Net TOA vs Ocean Heat Uptake
    ax = axes[0, 1]
    ax.axhline(0.0, color="#444444", lw=0.9, ls="--")
    ax.plot(yr, d["net_toa_wm2"], lw=1.3, color="#2ca02c", label="Net TOA (ASR - OLR)")
    ax.plot(yr, d["ohc_flux_wm2"], lw=1.3, color="#ff7f0e", alpha=0.85, label="Ocean dOHC/dt")
    ax.set_ylabel("Heat Flux [W m$^{-2}$]")
    ax.set_title("Planetary Energy Balance (TOA vs Ocean Uptake)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=9)

    # 3. Depth-Stratified Ocean Potential Temperature Anomalies (relative to Year 0)
    ax = axes[1, 0]
    ax.axhline(0.0, color="#444444", lw=0.8, ls="--")
    for key, lbl, col in [
        ("t_ocn_global_k", "Global Ocean", "#111111"),
        ("t_ocn_0_300m_k", "0-300 m (Upper)", "#d62728"),
        ("t_ocn_300_1500m_k", "300-1500 m (Mid)", "#ff7f0e"),
        ("t_ocn_deep_k", ">1500 m (Deep)", "#1f77b4"),
    ]:
        arr = d[key]
        if np.isfinite(arr[0]):
            ax.plot(yr, arr - arr[0], lw=1.4, color=col, label=f"{lbl} (t0={arr[0]-273.15:.2f}°C)")
    ax.set_ylabel("$\\Delta T_{\\mathrm{ocn}}$ [K]")
    ax.set_title("3D Ocean Temperature Drift by Depth Band")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=8.5, ncol=2)

    # 4. Global Ocean Salinity & Sea-Surface Salinity
    ax = axes[1, 1]
    ax.plot(yr, d["s_ocn_global_psu"], lw=1.5, color="#1f77b4", label="Global Vol-Mean $S$")
    ax2 = ax.twinx()
    ax2.plot(yr, d["sss_global_psu"], lw=1.2, ls="--", color="#9467bd", label="Global SSS")
    ax.set_ylabel("Global Mean $S$ [psu]", color="#1f77b4")
    ax2.set_ylabel("Global SSS [psu]", color="#9467bd")
    ax.set_title("Ocean Salinity Conservation & Surface Salinity")
    ax.grid(True, ls=":", alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="best", frameon=True, framealpha=0.92, fontsize=9)

    # 5. AMOC & ACC Transports
    ax = axes[2, 0]
    ax.plot(yr, d["amoc_26n_sv"], lw=1.4, color="#d62728", label="AMOC 26.5°N")
    ax.plot(yr, d["amoc_max_sv"], lw=1.2, ls="--", color="#ff7f0e", label="AMOC Max (20-60°N)")
    ax_acc = ax.twinx()
    ax_acc.plot(yr, d["acc_drake_sv"], lw=1.3, color="#1f77b4", label="ACC Drake Passage")
    ax.set_ylabel("AMOC [Sv]", color="#d62728")
    ax_acc.set_ylabel("ACC [Sv]", color="#1f77b4")
    ax.set_xlabel("Simulation Time [years]")
    ax.set_title("Overturning Circulation (AMOC) & ACC Transport")
    ax.grid(True, ls=":", alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_acc.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="best", frameon=True, framealpha=0.92, fontsize=8.5, ncol=2)

    # 6. Ocean KE, Sea-Ice Area (NH/SH), & Biosphere LAI
    ax = axes[2, 1]
    ax.plot(yr, d["seaice_nh_mkm2"], lw=1.4, color="#1f77b4", label="NH Sea Ice")
    ax.plot(yr, d["seaice_sh_mkm2"], lw=1.4, color="#17becf", label="SH Sea Ice")
    ax_ke = ax.twinx()
    ax_ke.plot(yr, d["ocn_ke_jm2"], lw=1.2, ls=":", color="#8c564b", label="Ocean KE")
    ax.set_ylabel("Sea-Ice Area [10$^6$ km$^2$]")
    ax_ke.set_ylabel("Ocean KE [J m$^{-2}$]", color="#8c564b")
    ax.set_xlabel("Simulation Time [years]")
    ax.set_title("Hemispheric Sea-Ice Area & Ocean Kinetic Energy")
    ax.grid(True, ls=":", alpha=0.5)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_ke.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="best", frameon=True, framealpha=0.92, fontsize=8.5, ncol=2)

    last_date = d["date"][-1]
    fig.suptitle(
        f"AOBA Glevel-5 3D OGCM piControl Spin-Up Monitor — {yr[-1]:.2f} yr ({last_date})",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = out_png.with_suffix(".tmp.png")
    fig.savefig(tmp_png, bbox_inches="tight")
    plt.close(fig)
    tmp_png.replace(out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AOBA ocean spin-up monitor dashboard")
    parser.add_argument("--csv", type=str, required=True, help="Path to spinup_monitor.csv")
    parser.add_argument("--out", type=str, default="", help="Output PNG path")
    args = parser.parse_args()
    csv_p = Path(args.csv).expanduser()
    out_p = Path(args.out).expanduser() if args.out else (csv_p.parent / "ocean_spinup_dashboard.png")
    plot_spinup_dashboard(csv_p, out_p)
    print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
