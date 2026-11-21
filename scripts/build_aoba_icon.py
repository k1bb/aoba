#!/usr/bin/env python3
"""Generate pure flat-vector SCVT Voronoi AOBA icon (SVG + 512px PNG) with 'A' tile selection.

Constructs a level-3 Spherical Centroidal Voronoi Tessellation (SCVT) oriented so one
equilateral icosahedral triangular face faces directly forward with its vertex up.
Voronoi cells are partitioned into deep ocean blue vs. AOBA leaf green to spell a crisp
capital 'A' solely from the geodesic grid cells (no external leaf or swoosh overlays).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in (_SRC_DIR, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_scvt_preview_cache import _extract_voronoi_preview_level  # noqa: E402


def _upright_icosahedral_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal basis `(ex, ey, ez)` centering upright face `[v2, v3, v4]`."""
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    v2 = np.array([-1.0, -phi, 0.0], dtype=np.float64)
    v3 = np.array([1.0, -phi, 0.0], dtype=np.float64)
    v4 = np.array([0.0, -1.0, phi], dtype=np.float64)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)
    v4 /= np.linalg.norm(v4)

    ez = v2 + v3 + v4
    ez /= np.linalg.norm(ez)
    ey = v4 - np.dot(v4, ez) * ez
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    ex /= np.linalg.norm(ex)
    return ex, ey, ez


def is_a_letter_cell(x: float, y: float, z: float) -> bool:
    """Return True iff cell center `(x, y, z)` in upright-face frame belongs to letter 'A'."""
    if z <= 0.30 or y > 0.64 or y < -0.45:
        return False
    gx = abs(x) / z
    gy = y / z
    t = (0.78 - gy) / (0.78 - (-0.68))
    outer_hw = 0.13 + 0.76 * t
    inner_hw = max(0.0, outer_hw - 0.34)
    if gy >= 0.44:
        return gx <= outer_hw
    if -0.27 <= gy <= -0.04 and gx <= outer_hw:
        return True
    return inner_hw <= gx <= outer_hw


def build_voronoi_a_geometry(glevel: int = 3) -> list[dict]:
    """Return sorted visible Voronoi cell descriptors (`pts`, `cx`, `cy`, `is_a`, `shade`, `wave`)."""
    d = _extract_voronoi_preview_level(glevel)
    xyz_c = d[f"g{glevel}_xyz_cell"].astype(np.float64) / 32767.0
    xyz_v = d[f"g{glevel}_xyz_vert"].astype(np.float64) / 32767.0
    voc = d[f"g{glevel}_verts_on_cell"]
    nvoc = d[f"g{glevel}_n_verts_on_cell"]

    ex, ey, ez = _upright_icosahedral_basis()
    cx = xyz_c @ ex
    cy = xyz_c @ ey
    cz = xyz_c @ ez
    vx = xyz_v @ ex
    vy = xyz_v @ ey

    gcx, gcy, gr = 256.0, 256.0, 200.0
    cells: list[dict] = []
    for i in np.argsort(cz):
        if cz[i] < -0.10:
            continue
        nv = int(nvoc[i])
        vids = voc[i, :nv]
        px = gcx + gr * vx[vids]
        py = gcy - gr * vy[vids]  # SVG / Canvas top-down Y
        cpx = gcx + gr * float(cx[i])
        cpy = gcy - gr * float(cy[i])
        a_flag = is_a_letter_cell(float(cx[i]), float(cy[i]), float(cz[i]))
        shade_idx = 0 if cz[i] > 0.72 else (1 if cz[i] > 0.42 else 2)
        # Normalized diagonal Othello wave coordinate in [0, 1]
        wave = max(0.0, min(1.0, 0.5 + 0.36 * float(cx[i]) - 0.34 * float(cy[i])))
        cells.append({
            "pts": [(round(float(px[k]), 2), round(float(py[k]), 2)) for k in range(nv)],
            "cx": round(cpx, 2),
            "cy": round(cpy, 2),
            "is_a": bool(a_flag),
            "shade": int(shade_idx),
            "wave": round(wave, 4),
            "math_pts": [(float(gcx + gr * vx[vids[k]]), float(gcy + gr * vy[vids[k]])) for k in range(nv)],
        })
    return cells


def write_svg(cells: list[dict], out_svg: Path) -> None:
    """Write pure-vector SVG with CSS-animated Othello flip wave (`0%` = green 'A' on blue)."""
    blue_pal = ("#2563eb", "#1d4ed8", "#1e3a8a")
    green_pal = ("#22c55e", "#16a34a", "#15803d")

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">',
        '  <defs>',
        '    <clipPath id="globeClip"><circle cx="256" cy="256" r="200"/></clipPath>',
        '    <style>',
        '      .othello-tile {',
        '        transform-box: fill-box;',
        '        transform-origin: center;',
        '      }',
    ]
    for s in range(3):
        b, g = blue_pal[s], green_pal[s]
        lines.extend([
            f'      .ot-b2g-{s} {{ animation: flip_b2g_{s} 9.8s ease-in-out infinite; }}',
            f'      .ot-g2b-{s} {{ animation: flip_g2b_{s} 9.8s ease-in-out infinite; }}',
            f'      @keyframes flip_b2g_{s} {{',
            f'        0%, 34%   {{ transform: rotate(-32deg) scaleX(1) rotate(32deg); fill: {b}; }}',
            f'        38%       {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {b}; }}',
            f'        38.1%     {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {g}; }}',
            f'        43%, 72%  {{ transform: rotate(-32deg) scaleX(1) rotate(32deg); fill: {g}; }}',
            f'        76%       {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {g}; }}',
            f'        76.1%     {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {b}; }}',
            f'        81%, 100% {{ transform: rotate(-32deg) scaleX(1) rotate(32deg); fill: {b}; }}',
            '      }',
            f'      @keyframes flip_g2b_{s} {{',
            f'        0%, 34%   {{ transform: rotate(-32deg) scaleX(1) rotate(32deg); fill: {g}; }}',
            f'        38%       {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {g}; }}',
            f'        38.1%     {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {b}; }}',
            f'        43%, 72%  {{ transform: rotate(-32deg) scaleX(1) rotate(32deg); fill: {b}; }}',
            f'        76%       {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {b}; }}',
            f'        76.1%     {{ transform: rotate(-32deg) scaleX(0.06) rotate(32deg); fill: {g}; }}',
            f'        81%, 100% {{ transform: rotate(-32deg) scaleX(1) rotate(32deg); fill: {g}; }}',
            '      }',
        ])
    lines.extend([
        '    </style>',
        '  </defs>',
        '  <rect x="14" y="14" width="484" height="484" rx="110" ry="110" fill="#0b1322" stroke="#1e293b" stroke-width="2.5"/>',
        '  <g clip-path="url(#globeClip)">',
        '    <circle cx="256" cy="256" r="200" fill="#0b1322"/>',
    ])

    for c in cells:
        s = c["shade"]
        init_fill = green_pal[s] if c["is_a"] else blue_pal[s]
        cls = f'othello-tile ot-g2b-{s}' if c["is_a"] else f'othello-tile ot-b2g-{s}'
        delay = round(c["wave"] * 1.4, 3)
        pts_str = " ".join(f"{x},{y}" for x, y in c["pts"])
        lines.append(
            f'    <polygon class="{cls}" points="{pts_str}" fill="{init_fill}" '
            f'stroke="#0b1322" stroke-width="2.0" stroke-linejoin="round" '
            f'style="animation-delay:{delay}s;"/>'
        )

    lines.extend([
        '  </g>',
        '  <circle cx="256" cy="256" r="200" fill="none" stroke="#38bdf8" stroke-width="2.2"/>',
        '</svg>',
    ])
    out_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(cells: list[dict], out_png: Path, size_px: int = 512) -> None:
    """Render high-precision anti-aliased 512x512 flat-vector PNG from the exact same Voronoi cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon as MplPolygon
    from PIL import Image

    blue_pal = ("#2563eb", "#1d4ed8", "#1e3a8a")
    green_pal = ("#22c55e", "#16a34a", "#15803d")

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (14, 14), 484, 484,
            boxstyle="round,pad=0,rounding_size=110",
            facecolor="#0b1322", edgecolor="#1e293b", linewidth=2.5,
        )
    )

    gcx, gcy, gr = 256.0, 256.0, 200.0
    clip = Circle((gcx, gcy), gr, transform=ax.transData)
    ax.add_patch(Circle((gcx, gcy), gr, facecolor="#0b1322", edgecolor="none"))

    for c in cells:
        col = green_pal[c["shade"]] if c["is_a"] else blue_pal[c["shade"]]
        poly = MplPolygon(
            c["math_pts"],
            closed=True,
            facecolor=col,
            edgecolor="#0b1322",
            linewidth=1.7,
            joinstyle="round",
        )
        poly.set_clip_path(clip)
        ax.add_patch(poly)

    ax.add_patch(Circle((gcx, gcy), gr, facecolor="none", edgecolor="#38bdf8", linewidth=2.2))
    fig.savefig(out_png, transparent=True, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)
    Image.open(out_png).resize((size_px, size_px), Image.LANCZOS).save(out_png)


def export_gui_icon_json(cells: list[dict]) -> str:
    """Return compact JSON payload for real-time Othello-flipping Canvas 2D icon in GUI."""
    compact = [
        [c["cx"], c["cy"], 1 if c["is_a"] else 0, c["shade"], c["wave"],
         [coord for pt in c["pts"] for coord in pt]]
        for c in cells
    ]
    return json.dumps(compact, separators=(",", ":"))


def main() -> None:
    cells = build_voronoi_a_geometry(glevel=3)
    out_dir = _REPO_ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_svg = out_dir / "aoba_icon.svg"
    out_png = out_dir / "aoba_icon.png"
    write_svg(cells, out_svg)
    write_png(cells, out_png, size_px=512)
    n_a = sum(1 for c in cells if c["is_a"])
    print(f"Generated {out_svg} and {out_png} ({len(cells)} front cells, {n_a} green 'A' cells)")


if __name__ == "__main__":
    main()
