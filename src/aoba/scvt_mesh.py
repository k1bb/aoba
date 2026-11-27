"""Spherical Centroidal Voronoi Tessellation (SCVT) C-Grid Mesh, TRiSK Weights, & Earth2014 Builder.

Constructs the global quasi-uniform icosahedral hexagonal/pentagonal SCVT Voronoi mesh
(`N_cells = 10 * 4^glevel + 2`), computes exact spherical geometric quantities and skew-symmetric
TRiSK tangential reconstruction weights (`Thuburn et al., 2009`; `Ringler et al., 2010`), and
aggregates 5-arc-min `Earth2014` (`Hirt & Rexer, 2015`) into cell-mean elevation, subgrid
orographic tensor statistics for `Lott & Miller (1997)`, land/ocean/ice-sheet masks, and
coastal runoff routing tables.

References
----------
- Ringler, Ju, & Gunzburger (2008, Ocean Modelling, 22, 104-115):
  Spherical Centroidal Voronoi Tessellations via Lloyd relaxation.
- Thuburn, Ringler, Skamarock, & Klemp (2009, J. Comput. Phys., 228, 8321-8335, Eqs. 33-38) &
  Ringler, Thuburn, Klemp, & Skamarock (2010, J. Comput. Phys., 229, 3065-3090, Eqs. 21-26):
  Orthogonal C-grid operators and energy-conserving skew-symmetric TRiSK Coriolis weights
  `w_{e,e'} = R_{e,e'} * (A_{v1} / A_e +/- ...)`.
- Klemp (2011, Mon. Wea. Rev., 139, 2163-2177):
  Smoothed terrain-following (STF) coordinate surface smoothing.
- Hirt & Rexer (2015, Int. J. Appl. Earth Obs. Geoinf., 39, 103-112):
  5-arc-min `Earth2014` global topography, bathymetry, and ice-sheet thickness (`SUR2014`,
  `ICE2014`, `TBI2014`).
- Lott & Miller (1997, Q. J. R. Meteorol. Soc., 123, 101-127, Appendix A):
  Subgrid orographic standard deviation `mu`, RMS slope `sigma`, anisotropy `gamma`, and
  principal ridge angle `psi`.

AOBA Formulation Reference: Sec. 2.1-2.3, Eqs. (2.1)-(2.8) (SCVT Voronoi C-Grid, TRiSK Weights & Earth2014 Tensor).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import torch

from aoba import constants as C
from aoba.kernels import (
    can_use_elem_metal,
    dispatch_scvt_project_winds_metal,
    dispatch_scvt_reconstruct_winds_metal,
)
from aoba.runtime import ensure_earth2014_data


@dataclass
class SCVTMesh:
    """Global SCVT orthogonal Voronoi C-grid mesh and Earth2014 boundary tensors."""

    glevel: int
    n_cells: int
    n_edges: int
    n_verts: int

    # Cell centers [n_cells]
    lat_cell: torch.Tensor          # Latitude [rad] in [-pi/2, pi/2]
    lon_cell: torch.Tensor          # Longitude [rad] in [-pi, pi]
    xyz_cell: torch.Tensor          # Unit-sphere Cartesian coordinates [n_cells, 3]
    area_cell: torch.Tensor         # Exact spherical Voronoi area A_i [m^2]
    f_cell: torch.Tensor            # Cell Coriolis parameter 2 * Omega * sin(lat) [s^-1]

    # Edges [n_edges]: normal vector points from cell_e[0] -> cell_e[1]
    lat_edge: torch.Tensor          # Edge midpoint latitude [rad]
    lon_edge: torch.Tensor          # Edge midpoint longitude [rad]
    f_edge: torch.Tensor            # Edge Coriolis parameter [s^-1]
    dc_edge: torch.Tensor           # Geodesic distance between cell_e[0] and cell_e[1] [m]
    dv_edge: torch.Tensor           # Voronoi primal edge length between vert_e[0] and vert_e[1] [m]
    area_edge: torch.Tensor         # Edge diamond area 0.5 * dc_edge * dv_edge [m^2]
    cells_on_edge: torch.Tensor     # [n_edges, 2] int64: (cell_0, cell_1)
    verts_on_edge: torch.Tensor     # [n_edges, 2] int64: (vert_0, vert_1) oriented so t_e = n_e x r_e -> v0->v1
    east_normal_edge: torch.Tensor  # Zonal component of unit normal n_e . e_east [n_edges]
    north_normal_edge: torch.Tensor # Meridional component of unit normal n_e . e_north [n_edges]

    # Cell-to-edge stencil [n_cells, max_deg=6] (padded with -1 / 0.0 for the 12 pentagons)
    edges_on_cell: torch.Tensor     # [n_cells, 6] int64 (valid edge index or 0 when sign == 0)
    edge_sign_on_cell: torch.Tensor # [n_cells, 6] float: +1 if outward from cell, -1 if inward, 0 if pad
    n_edges_on_cell: torch.Tensor   # [n_cells] int64 (5 or 6)

    # Vertex (dual triangle) quantities [n_verts, 3]
    area_vert: torch.Tensor         # Dual spherical triangle area A_v [n_verts] [m^2]
    f_vert: torch.Tensor            # Vertex Coriolis parameter [n_verts] [s^-1]
    edges_on_vert: torch.Tensor     # [n_verts, 3] int64
    edge_sign_on_vert: torch.Tensor # [n_verts, 3] float (+1/-1 for Stokes circulation curl)

    # Exact skew-symmetric TRiSK tangential velocity weights [n_edges, 10]
    trisk_edges: torch.Tensor       # [n_edges, 10] int64 neighbor edge indices e'
    trisk_weights: torch.Tensor     # [n_edges, 10] float weights w_{e,e'} (Ringler et al., 2010, Eq. 23)

    # Earth2014 surface & subgrid orographic boundary fields [n_cells]
    surf_type: torch.Tensor         # [n_cells] int64: 0=Ocean, 1=Land, 2=IceSheet
    ocean_frac: torch.Tensor        # [n_cells] float in [0, 1]
    icesheet_frac: torch.Tensor     # [n_cells] float in [0, 1]
    z_topo: torch.Tensor            # Cell-mean surface elevation [m]
    z_topo_smooth: torch.Tensor     # Laplacian-smoothed elevation for Klemp STF coordinate [m]
    sigma_topo: torch.Tensor        # Subgrid elevation std-dev mu [m] (Lott & Miller, 1997)
    slope_topo: torch.Tensor        # Subgrid RMS terrain slope sigma [-] (Lott & Miller, 1997)
    aniso_topo: torch.Tensor        # Subgrid anisotropy gamma in [0, 1] [-] (Lott & Miller, 1997)
    angle_topo: torch.Tensor        # Subgrid ridge orientation psi [rad] (Lott & Miller, 1997)
    coastal_ocean_idx: torch.Tensor # [n_cells] int64 nearest ocean cell index for river/glacier runoff
    z_bed: torch.Tensor | None = None       # Cell-mean subglacial bedrock/bathymetry elevation [m] (Earth2014 BED2014)
    h_ice_init: torch.Tensor | None = None  # Cell-mean initial ice-sheet thickness [m] (Earth2014 ICE2014)
    z_min_land: torch.Tensor | None = None  # Minimum non-ocean surface elevation in cell [m] (Earth2014 SUR2014 valley floor)
    lake_basin_frac_bed: torch.Tensor | None = None  # Solid-Earth bedrock depression area fraction (SUR > BED, non-ocean/ice) [-]
    lake_basin_vol_bed: torch.Tensor | None = None   # Cell-mean bedrock depression depth (volume / area_cell) [m]

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "SCVTMesh":
        """Move all mesh tensors to target compute device (`cpu` or `mps`) and float `dtype`."""
        dev = torch.device(device)
        fields = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                if val.is_floating_point():
                    fields[name] = val.to(device=dev, dtype=dtype)
                else:
                    fields[name] = val.to(device=dev)
            else:
                fields[name] = val
        return SCVTMesh(**fields)


# ---------------------------------------------------------------------------
# 1. Icosahedral Subdivision & Spherical Lloyd SCVT Relaxation
# ---------------------------------------------------------------------------
def _icosahedron_generators(glevel: int) -> np.ndarray:
    """Generate `N = 10 * 4^glevel + 2` points on unit sphere by midpoint subdivision."""
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    v_list = verts.tolist()
    for _ in range(glevel):
        edge_mid: dict[tuple[int, int], int] = {}
        new_faces = []
        for i0, i1, i2 in faces:
            def _mid(a: int, b: int) -> int:
                key = (min(a, b), max(a, b))
                if key not in edge_mid:
                    m = 0.5 * (np.array(v_list[a]) + np.array(v_list[b]))
                    m /= np.linalg.norm(m)
                    edge_mid[key] = len(v_list)
                    v_list.append(m.tolist())
                return edge_mid[key]

            m01 = _mid(i0, i1)
            m12 = _mid(i1, i2)
            m20 = _mid(i2, i0)
            new_faces.extend(
                [[i0, m01, m20], [i1, m12, m01], [i2, m20, m12], [m01, m12, m20]]
            )
        faces = np.array(new_faces, dtype=np.int64)

    pts = np.array(v_list, dtype=np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _spherical_triangle_area_unit(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact spherical excess area of triangles `(a, b, c)` on unit sphere via Eriksson's formula."""
    det = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    denom = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2.0 * np.arctan2(det, np.maximum(denom, 1.0e-15))


def _circumcenters_on_sphere(pts: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Compute outward unit-sphere circumcenters of spherical Delaunay triangles `simplices`."""
    p0 = pts[simplices[:, 0]]
    p1 = pts[simplices[:, 1]]
    p2 = pts[simplices[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    # Ensure outward orientation (dot with triangle centroid > 0)
    cent = (p0 + p1 + p2) / 3.0
    flip = np.einsum("ij,ij->i", n, cent) < 0.0
    n[flip] *= -1.0
    norm = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-15)
    return n / norm


def _lloyd_relax_scvt(pts: np.ndarray, n_iter: int = 12) -> np.ndarray:
    """Relax points toward a Spherical Centroidal Voronoi Tessellation (Ringler et al., 2008)."""
    cur = pts.copy()
    n_cells = cur.shape[0]
    for _ in range(n_iter):
        hull = ConvexHull(cur)
        tris = hull.simplices.copy()
        v_xyz = _circumcenters_on_sphere(cur, tris)
        # Orient each Delaunay triangle CCW around outward normal
        p0, p1, p2 = cur[tris[:, 0]], cur[tris[:, 1]], cur[tris[:, 2]]
        neg = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), v_xyz) < 0.0
        tris[neg, 1], tris[neg, 2] = tris[neg, 2].copy(), tris[neg, 1].copy()

        # Accumulate area-weighted centroids around each generator using edge midpoints
        new_cent = np.zeros_like(cur)
        for loc in range(3):
            ia = tris[:, loc]
            ib = tris[:, (loc + 1) % 3]
            mid = cur[ia] + cur[ib]
            mid /= np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1.0e-15)
            # Sub-triangle (cur[ia], mid, v_xyz) belongs to cell ia; (mid, cur[ib], v_xyz) to ib
            w_a = _spherical_triangle_area_unit(cur[ia], mid, v_xyz)
            c_a = cur[ia] + mid + v_xyz
            c_a /= np.maximum(np.linalg.norm(c_a, axis=1, keepdims=True), 1.0e-15)
            np.add.at(new_cent, ia, w_a[:, None] * c_a)

            w_b = _spherical_triangle_area_unit(mid, cur[ib], v_xyz)
            c_b = mid + cur[ib] + v_xyz
            c_b /= np.maximum(np.linalg.norm(c_b, axis=1, keepdims=True), 1.0e-15)
            np.add.at(new_cent, ib, w_b[:, None] * c_b)

        cur = new_cent / np.maximum(np.linalg.norm(new_cent, axis=1, keepdims=True), 1.0e-15)
    return cur


# ---------------------------------------------------------------------------
# 2. Complete Topology, Spherical Areas, & Skew-Symmetric TRiSK Weights
# ---------------------------------------------------------------------------
def _build_voronoi_cgrid_topology(
    xyz_cell: np.ndarray,
    radius: float,
) -> dict[str, np.ndarray]:
    """Build exact Voronoi C-grid topology and Ringler et al. (2010, Eq. 23) TRiSK weights."""
    n_cells = xyz_cell.shape[0]
    hull = ConvexHull(xyz_cell)
    tris = hull.simplices.copy()
    n_verts = tris.shape[0]

    xyz_vert = _circumcenters_on_sphere(xyz_cell, tris)
    # Orient every Delaunay triangle CCW around outward `xyz_vert`
    p0, p1, p2 = xyz_cell[tris[:, 0]], xyz_cell[tris[:, 1]], xyz_cell[tris[:, 2]]
    neg = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), xyz_vert) < 0.0
    tris[neg, 1], tris[neg, 2] = tris[neg, 2].copy(), tris[neg, 1].copy()

    # Dual spherical triangle areas A_v [m^2]
    area_vert_unit = _spherical_triangle_area_unit(
        xyz_cell[tris[:, 0]], xyz_cell[tris[:, 1]], xyz_cell[tris[:, 2]]
    )
    area_vert = area_vert_unit * (radius * radius)

    # Map each Delaunay edge (min(c0, c1), max(c0, c1)) -> two incident vertices (v_left, v_right)
    edge_to_verts: dict[tuple[int, int], list[int]] = {}
    for v_idx, (c0, c1, c2) in enumerate(tris):
        for a, b in ((c0, c1), (c1, c2), (c2, c0)):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edge_to_verts.setdefault(key, []).append(int(v_idx))

    edge_keys = sorted(edge_to_verts.keys())
    n_edges = len(edge_keys)
    cells_on_edge = np.zeros((n_edges, 2), dtype=np.int64)
    verts_on_edge = np.zeros((n_edges, 2), dtype=np.int64)
    xyz_edge = np.zeros((n_edges, 3), dtype=np.float64)
    dc_edge = np.zeros(n_edges, dtype=np.float64)
    dv_edge = np.zeros(n_edges, dtype=np.float64)
    east_normal = np.zeros(n_edges, dtype=np.float64)
    north_normal = np.zeros(n_edges, dtype=np.float64)

    cell_edges_list: list[list[tuple[float, int, float]]] = [[] for _ in range(n_cells)]
    vert_edges_list: list[list[tuple[int, float]]] = [[] for _ in range(n_verts)]

    for e_idx, (c0, c1) in enumerate(edge_keys):
        v_pair = edge_to_verts[(c0, c1)]
        v0, v1 = v_pair[0], v_pair[1]

        r0 = xyz_cell[c0]
        r1 = xyz_cell[c1]
        # Edge midpoint on Voronoi segment
        re = xyz_vert[v0] + xyz_vert[v1]
        re /= max(np.linalg.norm(re), 1.0e-15)
        xyz_edge[e_idx] = re

        # Unit normal n_e tangent to sphere at re pointing from c0 -> c1
        n_raw = r1 - r0
        n_raw = n_raw - np.dot(n_raw, re) * re
        n_e = n_raw / max(np.linalg.norm(n_raw), 1.0e-15)

        # Right-handed tangential unit vector t_e = r_e x n_e (so n_e x t_e = +r_e outward!)
        t_e = np.cross(re, n_e)
        if np.dot(xyz_vert[v1] - xyz_vert[v0], t_e) < 0.0:
            v0, v1 = v1, v0

        cells_on_edge[e_idx] = [c0, c1]
        verts_on_edge[e_idx] = [v0, v1]

        # Great-circle lengths dc_edge and dv_edge [m]
        dc_edge[e_idx] = radius * math.acos(max(-1.0, min(1.0, float(np.dot(r0, r1)))))
        dv_edge[e_idx] = radius * math.acos(max(-1.0, min(1.0, float(np.dot(xyz_vert[v0], xyz_vert[v1])))))

        # Local eastward & northward unit vectors at edge midpoint
        lon_e = math.atan2(re[1], re[0])
        lat_e = math.asin(max(-1.0, min(1.0, re[2])))
        e_east = np.array([-math.sin(lon_e), math.cos(lon_e), 0.0])
        e_north = np.array([-math.sin(lat_e) * math.cos(lon_e), -math.sin(lat_e) * math.sin(lon_e), math.cos(lat_e)])
        east_normal[e_idx] = float(np.dot(n_e, e_east))
        north_normal[e_idx] = float(np.dot(n_e, e_north))

        # Record CCW-sorted angle around each incident cell c0 (+1 outward) and c1 (-1 inward)
        for c_id, sgn in ((c0, 1.0), (c1, -1.0)):
            rc = xyz_cell[c_id]
            # Local tangent basis at cell center for CCW ordering around +rc
            lon_c = math.atan2(rc[1], rc[0])
            lat_c = math.asin(max(-1.0, min(1.0, rc[2])))
            ex_c = np.array([-math.sin(lon_c), math.cos(lon_c), 0.0])
            ey_c = np.array([-math.sin(lat_c) * math.cos(lon_c), -math.sin(lat_c) * math.sin(lon_c), math.cos(lat_c)])
            diff = re - rc
            ang = math.atan2(float(np.dot(diff, ey_c)), float(np.dot(diff, ex_c)))
            cell_edges_list[c_id].append((ang, e_idx, sgn))

        # Stokes curl around v1 (+n_e circulates CCW around v0 or v1?):
        # Since t_e points v0 -> v1, along boundary of dual triangle v0, n_e points CCW/CW:
        # Specifically, for dual triangle centered at v, edge e connects two cell centers (c0 -> c1 along n_e).
        # Around v, circulation along Delaunay boundary in CCW direction around +r_v has tangent tau_v = r_v x (r_mid - r_v).
        for v_id in (v0, v1):
            rv = xyz_vert[v_id]
            tau_ccw = np.cross(rv, re - rv)
            sgn_v = 1.0 if np.dot(n_e, tau_ccw) > 0.0 else -1.0
            vert_edges_list[v_id].append((e_idx, sgn_v))

    # Build padded cell-edge arrays [n_cells, 6] and exact spherical Voronoi areas A_i [m^2]
    edges_on_cell = np.zeros((n_cells, 6), dtype=np.int64)
    edge_sign_on_cell = np.zeros((n_cells, 6), dtype=np.float64)
    n_edges_on_cell = np.zeros(n_cells, dtype=np.int64)
    area_cell = np.zeros(n_cells, dtype=np.float64)
    # Also store partial Kite areas R_{i, v} (area of sub-sector inside cell i around vertex v) for TRiSK!
    kite_area: dict[tuple[int, int], float] = {}

    for i in range(n_cells):
        sorted_ce = sorted(cell_edges_list[i], key=lambda x: x[0])
        deg = len(sorted_ce)
        n_edges_on_cell[i] = deg
        rc = xyz_cell[i]
        # Ordered Voronoi vertices around cell i in CCW order:
        # For outward edge (sgn=+1), t_e = rc x n_e goes v0 -> v1 CCW around rc!
        v_ring = []
        for k, (_, e_id, sgn) in enumerate(sorted_ce):
            edges_on_cell[i, k] = e_id
            edge_sign_on_cell[i, k] = sgn
            va, vb = int(verts_on_edge[e_id, 0]), int(verts_on_edge[e_id, 1])
            # Start vertex in CCW order along edge k
            v_start = va if sgn > 0.0 else vb
            v_end = vb if sgn > 0.0 else va
            v_ring.append((e_id, v_start, v_end))

        # Exact spherical Voronoi area = sum of spherical triangles (rc, v_start, v_end)
        for e_id, vs, ve in v_ring:
            re = xyz_edge[e_id]
            a1 = float(_spherical_triangle_area_unit(rc[None, :], xyz_vert[vs][None, :], re[None, :])[0]) * (radius ** 2)
            a2 = float(_spherical_triangle_area_unit(rc[None, :], re[None, :], xyz_vert[ve][None, :])[0]) * (radius ** 2)
            area_cell[i] += a1 + a2
            kite_area[(i, vs)] = kite_area.get((i, vs), 0.0) + a1
            kite_area[(i, ve)] = kite_area.get((i, ve), 0.0) + a2

    edges_on_vert = np.zeros((n_verts, 3), dtype=np.int64)
    edge_sign_on_vert = np.zeros((n_verts, 3), dtype=np.float64)
    for v_id in range(n_verts):
        for k, (e_id, sgn_v) in enumerate(vert_edges_list[v_id][:3]):
            edges_on_vert[v_id, k] = e_id
            edge_sign_on_vert[v_id, k] = sgn_v

    # --- Exact TRiSK Tangential Reconstruction Weights w_{e, e'} (Thuburn et al., 2009, Eqs. 33-38;
    #     Ringler et al., 2010, Eq. 23) ---
    # On each cell i sharing edge e, walk CCW from e around cell i's boundary edges e_1, ..., e_{deg-1}:
    # Accumulate R_{i, v_k} / A_i from 0.5 down by kite_area[(i, v_k)] / area_cell[i] so that:
    #   d_e * u_perp(e) = sum_{e' in EC(e)} w_{e,e'} * l_{e'} * u_normal(e')
    # with exact skew-symmetry: d_e * l_e * w_{e,e'} + d_{e'} * l_{e'} * w_{e',e} == 0!
    trisk_edges = np.zeros((n_edges, 10), dtype=np.int64)
    trisk_weights = np.zeros((n_edges, 10), dtype=np.float64)

    for e_idx in range(n_edges):
        c0, c1 = int(cells_on_edge[e_idx, 0]), int(cells_on_edge[e_idx, 1])
        de = max(dc_edge[e_idx], 1.0)
        slot = 0
        # For cell c0 (where n_e is outward, sgn_e = +1) and cell c1 (where n_e is inward, sgn_e = -1):
        for c_id, sgn_e in ((c0, 1.0), (c1, -1.0)):
            deg = int(n_edges_on_cell[c_id])
            ce_ids = [int(edges_on_cell[c_id, k]) for k in range(deg)]
            ce_sgns = [float(edge_sign_on_cell[c_id, k]) for k in range(deg)]
            pos_e = ce_ids.index(e_idx)

            # Walk CCW starting from the vertex at the CCW-head of edge e around cell c_id
            # Let alpha_0 = 0.5, and at each step m = 1..deg-1 subtract kite_area[(c_id, v_between)] / area_cell[c_id]
            running_t = 0.5
            for step in range(1, deg):
                prev_k = (pos_e + step - 1) % deg
                curr_k = (pos_e + step) % deg
                e_prev = ce_ids[prev_k]
                sgn_prev = ce_sgns[prev_k]
                # CCW end vertex of e_prev around c_id is verts_on_edge[e_prev, 1] if sgn_prev > 0 else [0]
                v_corner = int(verts_on_edge[e_prev, 1] if sgn_prev > 0.0 else verts_on_edge[e_prev, 0])
                running_t -= kite_area.get((c_id, v_corner), area_cell[c_id] / deg) / area_cell[c_id]

                ep = ce_ids[curr_k]
                sgn_ep = ce_sgns[curr_k]
                # Thuburn et al. (2009, Eq. 35-38): weight factor guaranteeing exact skew-symmetry!
                # Check orientation: for uniform plane hexagon with n_e east, t_e north:
                # walking CCW around left cell c0 (center west of e): e is at east face; CCW goes NE -> NW -> W -> SW -> SE
                # running_t takes values (+1/3, +1/6, 0, -1/6, -1/3). At NE face, outward normal is (+1/2, +sqrt(3)/2);
                # to reconstruct +v_north at e from outward flux (-sgn_ep * u_ep), sign is -sgn_e * sgn_ep * running_t!
                le_p = max(dv_edge[ep], 1.0)
                w_raw = sgn_e * sgn_ep * running_t * (le_p / de)
                if slot < 10:
                    trisk_edges[e_idx, slot] = ep
                    trisk_weights[e_idx, slot] = w_raw
                    slot += 1

    # Symmetrize pair weights (w_{e,e'}*de*le - w_{e',e}*de'*le') / 2 to eliminate 1e-15 FP asymmetry
    pair_map: dict[tuple[int, int], tuple[int, float]] = {}
    for e_idx in range(n_edges):
        for s in range(10):
            ep = int(trisk_edges[e_idx, s])
            w = float(trisk_weights[e_idx, s])
            if abs(w) > 0.0 or ep != 0:
                pair_map[(e_idx, ep)] = (s, w)

    for (e1, e2), (s1, w1) in list(pair_map.items()):
        if e1 < e2 and (e2, e1) in pair_map:
            s2, w2 = pair_map[(e2, e1)]
            # Note: our w_raw already included (le_p / de) so u_tan(e) = sum w_{e,ep} * u_norm(ep);
            # exact energy conservation requires dv[e1]*dc[e1]*w1 + dv[e2]*dc[e2]*w2 == 0!
            g1 = dv_edge[e1] * dc_edge[e1]
            g2 = dv_edge[e2] * dc_edge[e2]
            anti = 0.5 * (g1 * w1 - g2 * w2)
            trisk_weights[e1, s1] = anti / max(g1, 1.0)
            trisk_weights[e2, s2] = -anti / max(g2, 1.0)

    area_edge = 0.5 * dc_edge * dv_edge
    lat_cell = np.arcsin(np.clip(xyz_cell[:, 2], -1.0, 1.0))
    lon_cell = np.arctan2(xyz_cell[:, 1], xyz_cell[:, 0])
    lat_edge = np.arcsin(np.clip(xyz_edge[:, 2], -1.0, 1.0))
    lon_edge = np.arctan2(xyz_edge[:, 1], xyz_edge[:, 0])
    lat_vert = np.arcsin(np.clip(xyz_vert[:, 2], -1.0, 1.0))

    return {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "xyz_cell": xyz_cell,
        "area_cell": area_cell,
        "lat_edge": lat_edge,
        "lon_edge": lon_edge,
        "dc_edge": dc_edge,
        "dv_edge": dv_edge,
        "area_edge": area_edge,
        "cells_on_edge": cells_on_edge,
        "verts_on_edge": verts_on_edge,
        "east_normal_edge": east_normal,
        "north_normal_edge": north_normal,
        "edges_on_cell": edges_on_cell,
        "edge_sign_on_cell": edge_sign_on_cell,
        "n_edges_on_cell": n_edges_on_cell,
        "area_vert": area_vert,
        "lat_vert": lat_vert,
        "edges_on_vert": edges_on_vert,
        "edge_sign_on_vert": edge_sign_on_vert,
        "trisk_edges": trisk_edges,
        "trisk_weights": trisk_weights,
    }


# ---------------------------------------------------------------------------
# 3. 5-arc-min Earth2014 Aggregation & Subgrid Orographic Tensor (Lott & Miller, 1997)
# ---------------------------------------------------------------------------
def _aggregate_earth2014_to_scvt(
    xyz_cell: np.ndarray,
    edges_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    cells_on_edge: np.ndarray,
    earth2014_dir: Path,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Aggregate 5-arc-min `Earth2014` (`SUR2014`, `ICE2014`, `TBI2014`) onto SCVT Voronoi cells.

    Computes exact cosine-latitude area-weighted cell means, surface types, coastal runoff targets,
    and Lott & Miller (1997, Appendix A) subgrid elevation standard deviation `mu`, RMS slope
    `sigma`, anisotropy `gamma`, and ridge orientation `psi`.
    """
    n_cells = xyz_cell.shape[0]
    sur_path = earth2014_dir / "Earth2014.SUR2014.5min.geod.bin"
    ice_path = earth2014_dir / "Earth2014.ICE2014.5min.geod.bin"
    tbi_path = earth2014_dir / "Earth2014.TBI2014.5min.geod.bin"
    bed_path = earth2014_dir / "Earth2014.BED2014.5min.geod.bin"

    sur_full = np.fromfile(sur_path, dtype=">i2").reshape(2160, 4320)[::stride, ::stride].astype(np.float32)
    ice_full = np.fromfile(ice_path, dtype=">i2").reshape(2160, 4320)[::stride, ::stride].astype(np.float32)
    tbi_full = np.fromfile(tbi_path, dtype=">i2").reshape(2160, 4320)[::stride, ::stride].astype(np.float32)
    if bed_path.exists():
        bed_full = np.fromfile(bed_path, dtype=">i2").reshape(2160, 4320)[::stride, ::stride].astype(np.float32)
    else:
        bed_full = sur_full - np.maximum(ice_full, 0.0)

    ny, nx = sur_full.shape
    dlat_rad = math.pi / ny
    dlon_rad = 2.0 * math.pi / nx
    lats = -0.5 * math.pi + (np.arange(ny, dtype=np.float32) + 0.5) * dlat_rad
    lons = -math.pi + (np.arange(nx, dtype=np.float32) + 0.5) * dlon_rad

    cos_lat = np.maximum(np.cos(lats), 1.0e-5)
    dy_m = C.EARTH_RADIUS * dlat_rad
    dx_m = C.EARTH_RADIUS * dlon_rad * cos_lat[:, None]

    # Central-difference subgrid terrain gradients dh/dx, dh/dy [m m^-1] for Lott & Miller (1997, App. A)
    dh_dx = (np.roll(sur_full, -1, axis=1) - np.roll(sur_full, 1, axis=1)) / (2.0 * dx_m)
    dh_dy = np.zeros_like(sur_full)
    dh_dy[1:-1] = (sur_full[2:] - sur_full[:-2]) / (2.0 * dy_m)

    is_ocean_px = ((sur_full == 0.0) & ((tbi_full < 0.0) | (bed_full < 0.0)) & (ice_full == 0.0)).astype(np.float32)
    is_ice_px = ((ice_full > 0.0) & (sur_full > 0.0)).astype(np.float32)
    # Pure solid-Earth bedrock depression depth (SUR > BED on non-ocean, non-ice pixels; zero prescribed lake masks)
    d_bath_px = np.where((is_ocean_px == 0.0) & (is_ice_px == 0.0) & (sur_full != 0.0), np.maximum(sur_full - bed_full, 0.0), 0.0).astype(np.float32)

    # Assign every 5-arc-min pixel to its nearest Voronoi generator on the unit sphere (exact Voronoi definition!)
    tree = cKDTree(xyz_cell)
    w_tot = np.zeros(n_cells, dtype=np.float64)
    sum_ocn = np.zeros(n_cells, dtype=np.float64)
    sum_ice = np.zeros(n_cells, dtype=np.float64)
    sum_ice_h = np.zeros(n_cells, dtype=np.float64)
    sum_bed = np.zeros(n_cells, dtype=np.float64)
    sum_h = np.zeros(n_cells, dtype=np.float64)
    sum_h2 = np.zeros(n_cells, dtype=np.float64)
    sum_hx2 = np.zeros(n_cells, dtype=np.float64)
    sum_hy2 = np.zeros(n_cells, dtype=np.float64)
    sum_hxy = np.zeros(n_cells, dtype=np.float64)
    sum_bath_frac = np.zeros(n_cells, dtype=np.float64)
    sum_bath_vol = np.zeros(n_cells, dtype=np.float64)
    z_min_land = np.full(n_cells, 1.0e9, dtype=np.float64)

    cos_lons = np.cos(lons)
    sin_lons = np.sin(lons)
    # Process in latitude chunks of 180 rows to keep peak RAM tiny (< 40 MB) and fast (~0.4 s)
    chunk = 180
    for j0 in range(0, ny, chunk):
        j1 = min(j0 + chunk, ny)
        clat = np.cos(lats[j0:j1])[:, None]
        slat = np.sin(lats[j0:j1])[:, None]
        px_x = (clat * cos_lons[None, :]).ravel()
        px_y = (clat * sin_lons[None, :]).ravel()
        px_z = np.broadcast_to(slat, (j1 - j0, nx)).ravel()
        px_xyz = np.column_stack([px_x, px_y, px_z])

        _, cid = tree.query(px_xyz, workers=-1)
        w_px = np.broadcast_to(cos_lat[j0:j1, None], (j1 - j0, nx)).ravel().astype(np.float64)

        h_c = sur_full[j0:j1].ravel().astype(np.float64)
        hx_c = dh_dx[j0:j1].ravel().astype(np.float64)
        hy_c = dh_dy[j0:j1].ravel().astype(np.float64)
        ocn_c = is_ocean_px[j0:j1].ravel()
        db_c = d_bath_px[j0:j1].ravel().astype(np.float64)

        np.add.at(w_tot, cid, w_px)
        np.add.at(sum_ocn, cid, w_px * ocn_c)
        np.add.at(sum_ice, cid, w_px * is_ice_px[j0:j1].ravel())
        np.add.at(sum_ice_h, cid, w_px * (is_ice_px[j0:j1] * np.maximum(ice_full[j0:j1], 0.0)).ravel())
        np.add.at(sum_bed, cid, w_px * bed_full[j0:j1].ravel())
        np.add.at(sum_h, cid, w_px * h_c)
        np.add.at(sum_h2, cid, w_px * h_c * h_c)
        np.add.at(sum_hx2, cid, w_px * hx_c * hx_c)
        np.add.at(sum_hy2, cid, w_px * hy_c * hy_c)
        np.add.at(sum_hxy, cid, w_px * hx_c * hy_c)
        np.add.at(sum_bath_frac, cid, w_px * (db_c > 0.5))
        np.add.at(sum_bath_vol, cid, w_px * db_c)
        non_ocn = ocn_c == 0.0
        if np.any(non_ocn):
            np.minimum.at(z_min_land, cid[non_ocn], h_c[non_ocn])

    w_safe = np.maximum(w_tot, 1.0e-12)
    ocean_frac = np.clip(sum_ocn / w_safe, 0.0, 1.0)
    icesheet_frac = np.clip(sum_ice / w_safe, 0.0, 1.0)
    land_frac = np.clip(1.0 - ocean_frac - icesheet_frac, 0.0, 1.0)

    # Dominant surface type: 0 = Ocean (ocean_frac >= 0.5), else 2 = IceSheet if ice >= land, else 1 = Land
    surf_type = np.where(ocean_frac >= 0.5, 0, np.where(icesheet_frac >= land_frac, 2, 1)).astype(np.int64)

    z_topo = sum_h / w_safe
    # Zero mean surface elevation over open-ocean cells
    z_topo = np.where(surf_type == 0, 0.0, z_topo)
    h_ice_init = np.where(surf_type == 0, 0.0, np.maximum(sum_ice_h / w_safe, 0.0))
    z_bed = sum_bed / w_safe
    z_min_land = np.where((surf_type == 0) | (z_min_land > 1.0e8), z_topo, z_min_land)
    lake_basin_frac_bed = np.where(surf_type == 1, np.clip(sum_bath_frac / w_safe, 0.0, 1.0), 0.0)
    lake_basin_vol_bed = np.where(surf_type == 1, np.maximum(sum_bath_vol / w_safe, 0.0), 0.0)

    var_h = np.maximum(sum_h2 / w_safe - (sum_h / w_safe) ** 2, 0.0)
    sigma_topo = np.where(surf_type == 0, 0.0, np.sqrt(var_h))

    # Lott & Miller (1997, Appendix A, Eqs. A1-A5): gradient tensor K, M, L -> psi, gamma, sigma
    hx2 = sum_hx2 / w_safe
    hy2 = sum_hy2 / w_safe
    hxy = sum_hxy / w_safe

    k_lm = 0.5 * (hx2 + hy2)
    m_lm = 0.5 * (hx2 - hy2)
    l_lm = hxy
    amp_lm = np.sqrt(m_lm * m_lm + l_lm * l_lm)

    angle_topo = 0.5 * np.arctan2(l_lm, m_lm)
    # Principal variances along & across ridge: lambda_1 = K + amp, lambda_2 = K - amp
    lam1 = np.maximum(k_lm + amp_lm, 1.0e-12)
    lam2 = np.maximum(k_lm - amp_lm, 0.0)
    aniso_topo = np.where(surf_type == 0, 1.0, np.sqrt(lam2 / lam1))
    slope_topo = np.where(surf_type == 0, 0.0, np.sqrt(hx2 + hy2))

    # Klemp (2011) 2-pass conservative Laplacian terrain smoothing for numerical metric smoothness
    z_smooth = z_topo.copy()
    for _ in range(2):
        z_next = z_smooth.copy()
        for i in range(n_cells):
            deg = int(n_edges_on_cell[i])
            nb_sum = 0.0
            for k in range(deg):
                e_id = int(edges_on_cell[i, k])
                c0, c1 = int(cells_on_edge[e_id, 0]), int(cells_on_edge[e_id, 1])
                nb = c1 if c0 == i else c0
                nb_sum += z_smooth[nb]
            z_next[i] = 0.75 * z_smooth[i] + 0.25 * (nb_sum / max(deg, 1))
        z_smooth = np.where(surf_type == 0, 0.0, z_next)

    # Nearest ocean cell index on unit sphere for conservative river & glacial runoff routing
    ocn_indices = np.where(surf_type == 0)[0]
    if len(ocn_indices) > 0:
        ocn_tree = cKDTree(xyz_cell[ocn_indices])
        _, nearest_loc = ocn_tree.query(xyz_cell)
        coastal_ocean_idx = ocn_indices[nearest_loc].astype(np.int64)
    else:
        coastal_ocean_idx = np.arange(n_cells, dtype=np.int64)

    return {
        "surf_type": surf_type,
        "ocean_frac": ocean_frac,
        "icesheet_frac": icesheet_frac,
        "z_topo": z_topo,
        "z_topo_smooth": z_smooth,
        "sigma_topo": sigma_topo,
        "slope_topo": slope_topo,
        "aniso_topo": aniso_topo,
        "angle_topo": angle_topo,
        "coastal_ocean_idx": coastal_ocean_idx,
        "z_bed": z_bed,
        "h_ice_init": h_ice_init,
        "z_min_land": z_min_land,
        "lake_basin_frac_bed": lake_basin_frac_bed,
        "lake_basin_vol_bed": lake_basin_vol_bed,
    }


_SCVT_CACHE_SCHEMA_VERSION = 2

_SCVT_TOPO_KEYS: tuple[str, ...] = (
    "lat_cell", "lon_cell", "xyz_cell", "area_cell",
    "lat_edge", "lon_edge", "dc_edge", "dv_edge", "area_edge",
    "cells_on_edge", "verts_on_edge", "east_normal_edge", "north_normal_edge",
    "edges_on_cell", "edge_sign_on_cell", "n_edges_on_cell",
    "lat_vert", "area_vert", "edges_on_vert", "edge_sign_on_vert",
    "trisk_edges", "trisk_weights",
)

_SCVT_BNDRY_KEYS: tuple[str, ...] = (
    "surf_type", "ocean_frac", "icesheet_frac",
    "z_topo", "z_topo_smooth", "sigma_topo", "slope_topo", "aniso_topo", "angle_topo",
    "coastal_ocean_idx", "z_bed", "h_ice_init", "z_min_land",
    "lake_basin_frac_bed", "lake_basin_vol_bed",
)


def _scvt_mesh_cache_path(
    glevel: int,
    n_lloyd_iter: int,
    radius: float,
    has_earth2014: bool,
    stride: int,
    planet_preset: str = "earth",
) -> Path:
    """Return deterministic CF/UGRID Zarr v3 cache path under `aoba.get_output_dir()/_mesh_cache`."""
    import aoba

    cache_dir = aoba.get_output_dir() / "_mesh_cache"
    p_key = str(planet_preset or "earth").strip().lower()
    if p_key == "pangaea":
        e14_tag = "pangaea250ma"
    else:
        e14_tag = f"e14s{max(int(stride), 1)}" if has_earth2014 else "aqua"
    name = f"scvt_g{int(glevel)}_l{int(n_lloyd_iter)}_r{int(round(radius))}_{e14_tag}_v{_SCVT_CACHE_SCHEMA_VERSION}.zarr"
    return cache_dir / name


def _load_scvt_mesh_cache_zarr(cache_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Load cached SCVT topology & boundary arrays from a CF/UGRID Zarr v3 store, or return None."""
    if not cache_path.is_dir():
        return None
    try:
        import zarr

        grp = zarr.open_group(str(cache_path), mode="r")
        if int(grp.attrs.get("schema_version", -1)) != _SCVT_CACHE_SCHEMA_VERSION:
            return None
        topo = {k: np.asarray(grp[k][:]) for k in _SCVT_TOPO_KEYS}
        bndry = {k: np.asarray(grp[k][:]) for k in _SCVT_BNDRY_KEYS}
        return topo, bndry
    except Exception:
        return None


def _save_scvt_mesh_cache_zarr(
    cache_path: Path,
    glevel: int,
    radius: float,
    topo: dict[str, np.ndarray],
    bndry: dict[str, np.ndarray],
) -> None:
    """Atomically write SCVT topology & boundary arrays to a CF/UGRID Zarr v3 store."""
    import os
    import shutil
    import zarr

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp_{os.getpid()}"
    try:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        grp = zarr.open_group(str(tmp_path), mode="w", zarr_format=3)
        grp.attrs.update({
            "cf_role": "mesh_topology",
            "topology_dimension": 2,
            "glevel": int(glevel),
            "n_cells": int(topo["lat_cell"].shape[0]),
            "n_edges": int(topo["dc_edge"].shape[0]),
            "n_verts": int(topo["area_vert"].shape[0]),
            "radius_m": float(radius),
            "schema_version": _SCVT_CACHE_SCHEMA_VERSION,
        })
        for k in _SCVT_TOPO_KEYS:
            arr = topo[k]
            arr_store = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.int64)
            grp.create_array(k, data=arr_store)
        for k in _SCVT_BNDRY_KEYS:
            arr = bndry[k]
            arr_store = arr.astype(np.float32) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.int64)
            grp.create_array(k, data=arr_store)
        if not cache_path.exists():
            try:
                os.replace(tmp_path, cache_path)
            except OSError:
                shutil.rmtree(tmp_path, ignore_errors=True)
        else:
            shutil.rmtree(tmp_path, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)


def build_scvt_mesh(
    glevel: int = 3,
    n_lloyd_iter: int = 12,
    use_earth2014: bool = True,
    earth2014_dir: str | Path = "data/boundary/earth2014_5min",
    earth2014_stride: int = 2,
    omega: float = C.OMEGA_EARTH,
    radius: float = C.EARTH_RADIUS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    use_cache: bool | None = None,
    planet_preset: str = "earth",
) -> SCVTMesh:
    """Build complete SCVT Voronoi C-grid mesh (`glevel = 0..8`) on `device` with Zarr v3 disk cache."""
    import os

    p_key = str(planet_preset or "earth").strip().lower()
    use_e14_eff = bool(use_earth2014 and p_key != "pangaea")
    if use_e14_eff:
        try:
            e_dir = ensure_earth2014_data(earth2014_dir, auto_download=True)
        except Exception as exc:
            print(f"[AOBA Warning] Failed to auto-download Earth2014 ({exc}); falling back if cached.", flush=True)
            e_dir = Path(earth2014_dir)
    else:
        e_dir = Path(earth2014_dir)
    has_e14 = bool(use_e14_eff and (e_dir / "Earth2014.SUR2014.5min.geod.bin").exists())

    enable_cache = (glevel >= 4) if use_cache is None else bool(use_cache)
    if os.environ.get("AOBA_DISABLE_MESH_CACHE", "0") == "1":
        enable_cache = False

    cache_path = (
        _scvt_mesh_cache_path(glevel, n_lloyd_iter, radius, has_e14, earth2014_stride, planet_preset=p_key)
        if enable_cache
        else None
    )
    cached = _load_scvt_mesh_cache_zarr(cache_path) if cache_path is not None else None
    if cached is not None:
        topo, bndry = cached
        n_cells = int(topo["lat_cell"].shape[0])
        n_edges = int(topo["dc_edge"].shape[0])
        n_verts = int(topo["area_vert"].shape[0])
    else:
        pts0 = _icosahedron_generators(glevel)
        xyz_cell = _lloyd_relax_scvt(pts0, n_iter=n_lloyd_iter) if n_lloyd_iter > 0 else pts0
        topo = _build_voronoi_cgrid_topology(xyz_cell, radius=radius)

        n_cells = xyz_cell.shape[0]
        n_edges = topo["dc_edge"].shape[0]
        n_verts = topo["area_vert"].shape[0]

        if p_key == "pangaea":
            from aoba.planetary_presets import build_pangaea_boundary_fields

            bndry = build_pangaea_boundary_fields(
                xyz_cell=xyz_cell,
                lat_cell=topo["lat_cell"],
                lon_cell=topo["lon_cell"],
                edges_on_cell=topo["edges_on_cell"],
                n_edges_on_cell=topo["n_edges_on_cell"],
                cells_on_edge=topo["cells_on_edge"],
            )
        elif has_e14:
            bndry = _aggregate_earth2014_to_scvt(
                xyz_cell,
                topo["edges_on_cell"],
                topo["n_edges_on_cell"],
                topo["cells_on_edge"],
                e_dir,
                stride=max(earth2014_stride, 1),
            )
        else:
            # Idealized aquaplanet / Held-Suarez flat boundary fallback
            bndry = {
                "surf_type": np.zeros(n_cells, dtype=np.int64),
                "ocean_frac": np.ones(n_cells, dtype=np.float64),
                "icesheet_frac": np.zeros(n_cells, dtype=np.float64),
                "z_topo": np.zeros(n_cells, dtype=np.float64),
                "z_topo_smooth": np.zeros(n_cells, dtype=np.float64),
                "sigma_topo": np.zeros(n_cells, dtype=np.float64),
                "slope_topo": np.zeros(n_cells, dtype=np.float64),
                "aniso_topo": np.ones(n_cells, dtype=np.float64),
                "angle_topo": np.zeros(n_cells, dtype=np.float64),
                "coastal_ocean_idx": np.arange(n_cells, dtype=np.int64),
                "z_bed": np.full(n_cells, -4000.0, dtype=np.float64),
                "h_ice_init": np.zeros(n_cells, dtype=np.float64),
                "z_min_land": np.zeros(n_cells, dtype=np.float64),
                "lake_basin_frac_bed": np.zeros(n_cells, dtype=np.float64),
                "lake_basin_vol_bed": np.zeros(n_cells, dtype=np.float64),
            }
        if cache_path is not None:
            _save_scvt_mesh_cache_zarr(cache_path, glevel, radius, topo, bndry)

    dev = torch.device(device)
    def _tf(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).to(device=dev, dtype=dtype)

    def _ti(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.int64)).to(device=dev)

    lat_cell_t = _tf(topo["lat_cell"])
    lat_edge_t = _tf(topo["lat_edge"])
    lat_vert_t = _tf(topo["lat_vert"])

    return SCVTMesh(
        glevel=glevel,
        n_cells=n_cells,
        n_edges=n_edges,
        n_verts=n_verts,
        lat_cell=lat_cell_t,
        lon_cell=_tf(topo["lon_cell"]),
        xyz_cell=_tf(topo["xyz_cell"]),
        area_cell=_tf(topo["area_cell"]),
        f_cell=2.0 * omega * torch.sin(lat_cell_t),
        lat_edge=lat_edge_t,
        lon_edge=_tf(topo["lon_edge"]),
        f_edge=2.0 * omega * torch.sin(lat_edge_t),
        dc_edge=_tf(topo["dc_edge"]),
        dv_edge=_tf(topo["dv_edge"]),
        area_edge=_tf(topo["area_edge"]),
        cells_on_edge=_ti(topo["cells_on_edge"]),
        verts_on_edge=_ti(topo["verts_on_edge"]),
        east_normal_edge=_tf(topo["east_normal_edge"]),
        north_normal_edge=_tf(topo["north_normal_edge"]),
        edges_on_cell=_ti(topo["edges_on_cell"]),
        edge_sign_on_cell=_tf(topo["edge_sign_on_cell"]),
        n_edges_on_cell=_ti(topo["n_edges_on_cell"]),
        area_vert=_tf(topo["area_vert"]),
        f_vert=2.0 * omega * torch.sin(lat_vert_t),
        edges_on_vert=_ti(topo["edges_on_vert"]),
        edge_sign_on_vert=_tf(topo["edge_sign_on_vert"]),
        trisk_edges=_ti(topo["trisk_edges"]),
        trisk_weights=_tf(topo["trisk_weights"]),
        surf_type=_ti(bndry["surf_type"]),
        ocean_frac=_tf(bndry["ocean_frac"]),
        icesheet_frac=_tf(bndry["icesheet_frac"]),
        z_topo=_tf(bndry["z_topo"]),
        z_topo_smooth=_tf(bndry["z_topo_smooth"]),
        sigma_topo=_tf(bndry["sigma_topo"]),
        slope_topo=_tf(bndry["slope_topo"]),
        aniso_topo=_tf(bndry["aniso_topo"]),
        angle_topo=_tf(bndry["angle_topo"]),
        coastal_ocean_idx=_ti(bndry["coastal_ocean_idx"]),
        z_bed=_tf(bndry["z_bed"]),
        h_ice_init=_tf(bndry["h_ice_init"]),
        z_min_land=_tf(bndry["z_min_land"]),
        lake_basin_frac_bed=_tf(bndry["lake_basin_frac_bed"]),
        lake_basin_vol_bed=_tf(bndry["lake_basin_vol_bed"]),
    )


# ---------------------------------------------------------------------------
# 4. Vector Calculus Operators on the SCVT Voronoi C-Grid
# ---------------------------------------------------------------------------
def reconstruct_cell_winds(
    mesh: SCVTMesh,
    u_edge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct cell-center zonal & meridional winds `(u_zonal, v_merid)` `[..., n_cells]`
    from edge normal velocities `u_edge` `[..., n_edges]` via Perot (2000) / Ringler et al. (2010, Eq. 29)."""
    if u_edge.ndim in (1, 2) and u_edge.shape[-1] == mesh.n_edges and can_use_elem_metal(u_edge):
        return dispatch_scvt_reconstruct_winds_metal(mesh, u_edge)

    # Weighted least-squares / Perot vector reconstruction:
    # u_vec_i = (1 / A_i) * sum_{e in cell i} (0.5 * dc_e * dv_e) * u_e * n_e
    w_e = 0.5 * mesh.dc_edge * mesh.dv_edge  # [n_edges] = area_edge
    ue_east = u_edge * w_e * mesh.east_normal_edge
    ue_north = u_edge * w_e * mesh.north_normal_edge

    # Gather [..., n_cells, 6]
    idx = mesh.edges_on_cell  # [n_cells, 6]
    active = (mesh.edge_sign_on_cell != 0.0).to(u_edge.dtype)  # [n_cells, 6]

    u_zonal = (ue_east[..., idx] * active).sum(dim=-1) / mesh.area_cell
    v_merid = (ue_north[..., idx] * active).sum(dim=-1) / mesh.area_cell
    return u_zonal, v_merid


def project_cell_vectors_to_edges(
    mesh: SCVTMesh,
    vec_east_cell: torch.Tensor,
    vec_north_cell: torch.Tensor,
) -> torch.Tensor:
    """Project cell-centered `(east, north)` vector tendencies `[..., n_cells]` onto C-grid edge normals."""
    if (
        vec_east_cell.ndim in (1, 2)
        and vec_east_cell.shape == vec_north_cell.shape
        and vec_east_cell.shape[-1] == mesh.n_cells
        and can_use_elem_metal(vec_east_cell)
        and not vec_north_cell.requires_grad
    ):
        return dispatch_scvt_project_winds_metal(mesh, vec_east_cell, vec_north_cell)

    c0 = mesh.cells_on_edge[:, 0]
    c1 = mesh.cells_on_edge[:, 1]
    ve = 0.5 * (vec_east_cell[..., c0] + vec_east_cell[..., c1])
    vn = 0.5 * (vec_north_cell[..., c0] + vec_north_cell[..., c1])
    return ve * mesh.east_normal_edge + vn * mesh.north_normal_edge
