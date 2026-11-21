#!/usr/bin/env python3
"""Precompute real SCVT Voronoi polygon meshes (`G0..G6`) with Earth2014 masks for GUI display."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aoba.runtime import ensure_earth2014_data  # noqa: E402
from aoba.scvt_mesh import (  # noqa: E402
    _circumcenters_on_sphere,
    _icosahedron_generators,
    _lloyd_relax_scvt,
)


def _extract_voronoi_preview_level(
    glevel: int,
    *,
    earth_tree: cKDTree | None = None,
    earth_is_ocn: np.ndarray | None = None,
    earth_is_ice: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build exact SCVT cell centers, Voronoi vertex coordinates, CCW cell rings, and edges."""
    n_iter = 12 if glevel <= 5 else 6
    xyz_cell = _lloyd_relax_scvt(_icosahedron_generators(glevel), n_iter=n_iter)
    hull = ConvexHull(xyz_cell)
    tris = hull.simplices.copy()
    xyz_vert = _circumcenters_on_sphere(xyz_cell, tris)

    # Orient Delaunay triangles CCW
    p0, p1, p2 = xyz_cell[tris[:, 0]], xyz_cell[tris[:, 1]], xyz_cell[tris[:, 2]]
    neg = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), xyz_vert) < 0.0
    tris[neg, 1], tris[neg, 2] = tris[neg, 2].copy(), tris[neg, 1].copy()

    n_cells = xyz_cell.shape[0]
    n_verts = xyz_vert.shape[0]

    # Collect incident Voronoi vertices for each cell and unique Voronoi edges
    cell_verts: list[list[int]] = [[] for _ in range(n_cells)]
    edge_dict: dict[tuple[int, int], list[int]] = {}
    for v_idx in range(n_verts):
        c0, c1, c2 = int(tris[v_idx, 0]), int(tris[v_idx, 1]), int(tris[v_idx, 2])
        cell_verts[c0].append(v_idx)
        cell_verts[c1].append(v_idx)
        cell_verts[c2].append(v_idx)
        for a, b in ((c0, c1), (c1, c2), (c2, c0)):
            key = (a, b) if a < b else (b, a)
            edge_dict.setdefault(key, []).append(v_idx)

    verts_on_cell = np.full((n_cells, 6), -1, dtype=np.int32)
    n_verts_on_cell = np.zeros(n_cells, dtype=np.uint8)
    for i in range(n_cells):
        v_ids = cell_verts[i]
        rc = xyz_cell[i]
        lon_c = math.atan2(rc[1], rc[0])
        lat_c = math.asin(max(-1.0, min(1.0, float(rc[2]))))
        ex = np.array([-math.sin(lon_c), math.cos(lon_c), 0.0])
        ey = np.array([-math.sin(lat_c) * math.cos(lon_c), -math.sin(lat_c) * math.sin(lon_c), math.cos(lat_c)])
        ang_pairs = []
        for vid in v_ids:
            d = xyz_vert[vid] - rc
            ang_pairs.append((math.atan2(float(np.dot(d, ey)), float(np.dot(d, ex))), vid))
        ang_pairs.sort()
        deg = min(len(ang_pairs), 6)
        n_verts_on_cell[i] = deg
        for k in range(deg):
            verts_on_cell[i, k] = ang_pairs[k][1]

    verts_on_edge = np.array([pair[:2] for pair in edge_dict.values()], dtype=np.int32)

    # Assign exact Earth2014 cell surface mask (0=Ocean, 1=Land, 2=IceSheet)
    if earth_tree is not None and earth_is_ocn is not None and earth_is_ice is not None:
        # Query nearest SCVT cell for 15-arc-min Earth2014 sample grid (fast exact cell majority)
        _, cell_id = cKDTree(xyz_cell).query(earth_tree.data, workers=2)
        w_tot = np.bincount(cell_id, minlength=n_cells).astype(np.float64)
        w_ocn = np.bincount(cell_id, weights=earth_is_ocn, minlength=n_cells)
        w_ice = np.bincount(cell_id, weights=earth_is_ice, minlength=n_cells)
        safe = np.maximum(w_tot, 1.0)
        f_ocn = w_ocn / safe
        f_ice = w_ice / safe
        surf_type = np.where(f_ocn >= 0.5, 0, np.where(f_ice >= 0.5, 2, 1)).astype(np.uint8)
    else:
        surf_type = np.zeros(n_cells, dtype=np.uint8)

    return {
        f"g{glevel}_xyz_cell": np.rint(xyz_cell * 32767.0).astype(np.int16),
        f"g{glevel}_xyz_vert": np.rint(xyz_vert * 32767.0).astype(np.int16),
        f"g{glevel}_verts_on_cell": verts_on_cell,
        f"g{glevel}_n_verts_on_cell": n_verts_on_cell,
        f"g{glevel}_verts_on_edge": verts_on_edge,
        f"g{glevel}_surf_type": surf_type,
    }


def build_preview_archive(out_path: Path | None = None, max_glevel: int = 6) -> Path:
    """Build and save compressed `config/scvt_preview_meshes.npz` covering `G0..G{max_glevel}`."""
    target = out_path or (_REPO_ROOT / "config" / "scvt_preview_meshes.npz")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Load Earth2014 at stride=3 (15-arc-min = 720x1440 = 1.03M pts, ultra-fast <0.1s per level)
    earth_tree = None
    earth_ocn = None
    earth_ice = None
    try:
        e_dir = ensure_earth2014_data(None, auto_download=False)
        sur_p = e_dir / "Earth2014.SUR2014.5min.geod.bin"
        ice_p = e_dir / "Earth2014.ICE2014.5min.geod.bin"
        tbi_p = e_dir / "Earth2014.TBI2014.5min.geod.bin"
        if sur_p.exists() and ice_p.exists() and tbi_p.exists():
            stride = 3
            sur = np.fromfile(sur_p, dtype=">i2").reshape(2160, 4320)[::stride, ::stride].astype(np.float32)
            ice = np.fromfile(ice_p, dtype=">i2").reshape(2160, 4320)[::stride, ::stride].astype(np.float32)
            tbi = np.fromfile(tbi_p, dtype=">i2").reshape(2160, 4320)[::stride, ::stride].astype(np.float32)
            ny, nx = sur.shape
            lats = -0.5 * math.pi + (np.arange(ny) + 0.5) * (math.pi / ny)
            lons = -math.pi + (np.arange(nx) + 0.5) * (2.0 * math.pi / nx)
            lon2, lat2 = np.meshgrid(lons, lats)
            xyz_e = np.stack(
                [(np.cos(lat2) * np.cos(lon2)).ravel(), (np.cos(lat2) * np.sin(lon2)).ravel(), np.sin(lat2).ravel()],
                axis=-1,
            )
            earth_tree = cKDTree(xyz_e)
            earth_ocn = ((sur == 0.0) & (tbi < 0.0) & (ice == 0.0)).ravel().astype(np.float64)
            earth_ice = ((ice > 0.0) & (sur > 0.0)).ravel().astype(np.float64)
    except Exception:
        pass

    arrays: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for g in range(max_glevel + 1):
        tg0 = time.perf_counter()
        lvl = _extract_voronoi_preview_level(
            g, earth_tree=earth_tree, earth_is_ocn=earth_ocn, earth_is_ice=earth_ice
        )
        arrays.update(lvl)
        print(
            f"Built Glevel-{g}: {lvl[f'g{g}_xyz_cell'].shape[0]:,} Voronoi cells, "
            f"{lvl[f'g{g}_verts_on_edge'].shape[0]:,} edges ({time.perf_counter() - tg0:.2f}s)",
            flush=True,
        )

    np.savez_compressed(target, **arrays)
    print(f"Saved {target} ({target.stat().st_size / 1024.0:.1f} KB) in {time.perf_counter() - t0:.2f}s", flush=True)
    return target


if __name__ == "__main__":
    build_preview_archive()
