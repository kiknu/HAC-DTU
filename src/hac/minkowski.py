"""
Minkowski minimisation: reconstruct a convex polyhedron from facet areas and
outward unit normals.

Python port of `workspaces/fortran/minkowski.f` (M. Kaasalainen, 2005; see
Kaasalainen, Torppa & Muinonen 2001 for the algorithm). Given a set of facet
areas and outward unit normals (e.g. recovered from lightcurve inversion),
this iteratively finds the support-function values (facet distances from the
origin) whose dual convex hull has the prescribed facet areas/normals, then
converts the dual hull back into the primal vertex set.

The Fortran `convhull` subroutine (a hand-rolled 3D gift-wrapping algorithm)
is replaced by `scipy.spatial.ConvexHull`: both compute the same convex hull,
but scipy's qhull binding is far faster than a literal O(n^2) translation. Everything else follows the reference
algorithm's control flow (accept/reject steps, adaptive step size, etc.)
term for term.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

_TINY = 1e-8


def _convex_hull(points: np.ndarray) -> ConvexHull:
    try:
        return ConvexHull(points)
    except QhullError:
        # Joggle the input to break exact degeneracies (e.g. the very first
        # iteration, where all dual points share the same scale factor).
        return ConvexHull(points, qhull_options="QJ")


def _facet_polygons(hull: ConvexHull, n_points: int):
    """
    For every point that is a vertex of `hull`, return the ordered list of
    3D coordinates of the "wedge" points obtained by walking the vertex link
    (the cyclic sequence of triangles incident to that vertex).

    This is the primal-space polygon dual to that hull vertex: consecutive
    wedges share an edge of the vertex link, and each wedge's plane equation
    gives one vertex of the primal polytope (see module docstring / notebook
    section 6 for the derivation: w = -normal/offset solves w.y = 1 for the
    three dual points spanning that wedge's triangle).
    """
    simplices = hull.simplices
    equations = hull.equations

    incident: dict[int, list[tuple[int, int, int]]] = {}
    for row in range(simplices.shape[0]):
        a, b, c = simplices[row]
        incident.setdefault(int(a), []).append((int(b), int(c), row))
        incident.setdefault(int(b), []).append((int(a), int(c), row))
        incident.setdefault(int(c), []).append((int(a), int(b), row))

    polygons: dict[int, np.ndarray] = {}
    for vertex, wedges in incident.items():
        link_adj: dict[int, list[tuple[int, int]]] = {}
        for p, q, row in wedges:
            link_adj.setdefault(p, []).append((q, row))
            link_adj.setdefault(q, []).append((p, row))

        start = wedges[0][0]
        rows_in_order = []
        prev = None
        cur = start
        for _ in range(len(wedges)):
            options = link_adj[cur]
            nxt, row = options[0] if options[0][0] != prev else options[1]
            rows_in_order.append(row)
            prev, cur = cur, nxt

        eqs = equations[rows_in_order]
        polygons[vertex] = -eqs[:, :3] / eqs[:, 3:4]

    return polygons


def _polygon_perimeter(poly: np.ndarray) -> float:
    if len(poly) < 2:
        return 0.0
    nxt = np.roll(poly, -1, axis=0)
    return float(np.linalg.norm(nxt - poly, axis=1).sum())


def _polygon_area_vector(poly: np.ndarray) -> np.ndarray:
    nxt = np.roll(poly, -1, axis=0)
    return np.cross(poly, nxt).sum(axis=0)


def _polygon_cmass_contribution(poly: np.ndarray, d_i: float) -> np.ndarray:
    """Fan-triangulation contribution to (unnormalised) centre of mass * 24V."""
    k = len(poly)
    if k < 3:
        return np.zeros(3)
    apex = poly[0]
    a = poly[1:-1] - apex
    b = poly[2:] - apex
    cross = np.cross(a, b)
    lengths = np.linalg.norm(cross, axis=1)
    triple_sum = apex[None, :] + poly[1:-1] + poly[2:]
    return float(d_i) * (lengths[:, None] * triple_sum).sum(axis=0)


def minkowski_reconstruct(
    areas: np.ndarray,
    normals: np.ndarray,
    epsilon: float = 0.005,
    max_iter: int = 3000,
    verbose: bool = False,
) -> np.ndarray:
    """
    Reconstruct a convex polyhedron from facet areas + outward unit normals.

    Parameters
    ----------
    areas: (n,) facet areas.
    normals: (n, 3) outward unit normals (need not be pre-normalised).
    epsilon: convergence threshold on the angle alpha_k between the current
        and target area vectors (in gradient/area space); the reference
        implementation defaults to 0.005.
    max_iter: safety cap on iterations (the reference algorithm is provably
        convergent but has no built-in cap; we add one so a numerical issue
        can't hang the notebook).
    verbose: print alpha_k / volume every 25 iterations.

    Returns
    -------
    vertices: (m, 3) array of the reconstructed polytope's vertices (a point
        cloud; take its convex hull to obtain faces, e.g. via
        `build_convex_stl`).
    """
    normals = np.asarray(normals, dtype=np.float64)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    initarea = np.asarray(areas, dtype=np.float64).copy()
    n = len(initarea)

    dots = normals @ normals.T
    dhelp = 1.0 - dots**2
    mask = dhelp > 1e-4
    initm = float(np.max(1.0 / np.sqrt(dhelp[mask]))) if np.any(mask) else 1.0

    scale = 1000.0 / initarea.sum()
    initarea *= scale
    initareasum = 100.0  # fixed reference value, as in the Fortran source
    initarealen = np.sqrt(np.dot(initarea, initarea))

    d = np.full(n, initarealen**2 / initareasum, dtype=np.float64)

    ncoef = 0
    posroute = 1
    prevolume = 0.0
    premaxpxk = 0.0
    prearea = np.zeros(n)
    eta = 1.3

    xk = np.zeros(n)
    cmass = np.zeros(3)
    scoef = 1.0
    arealen = 0.0

    for iteration in range(1, max_iter + 1):
        y = normals / d[:, None]
        hull = _convex_hull(y)
        polygons = _facet_polygons(hull, n)
        hull_vertex_set = set(int(v) for v in hull.vertices)

        all_points = np.concatenate(
            [polygons[i] for i in hull_vertex_set], axis=0
        )

        h = np.zeros(n)
        pxk = np.zeros(n)
        for i in range(n):
            if i not in hull_vertex_set:
                dots_i = all_points @ normals[i]
                max_val = float(dots_i.max())
                maxind = int(np.argmax(dots_i))
                poly = [all_points[maxind]]
                ties = np.nonzero(dots_i == max_val)[0]
                for t in ties:
                    if t != maxind:
                        poly.append(all_points[t])
                        break
                polygons[i] = np.asarray(poly)
                h[i] = d[i] - max_val
            pxk[i] = _polygon_perimeter(polygons[i])
        maxpxk = float(pxk.max())

        area = np.zeros(n)
        volume = 0.0
        for i in range(n):
            addvect = _polygon_area_vector(polygons[i])
            area[i] = float(np.linalg.norm(addvect)) / 2.0
            volume += d[i] * area[i] / 3.0

        if volume >= prevolume:
            cmass = np.zeros(3)
            for i in range(n):
                cmass += _polygon_cmass_contribution(polygons[i], d[i])
            cmass /= 24 * volume

            xk = d - normals @ cmass - h
            scoef = initarealen**2 / np.dot(xk, initarea)
            xk = xk * scoef
            area = area * scoef**2
            maxpxk *= scoef
            volume *= scoef**3

            if prevolume != 0.0:
                retemp = (
                    eta**ncoef
                    * (arealen * np.sin(alphak)) ** 2
                    / (6 * initm * premaxpxk)
                )
            else:
                retemp = 0.0

            if (volume - prevolume) >= retemp and posroute == 1:
                ncoef += 1
            else:
                ncoef -= 1
        else:
            ncoef -= 1
            volume = prevolume
            maxpxk = premaxpxk
            area = prearea.copy()

        prevolume = volume
        premaxpxk = maxpxk
        prearea = area.copy()

        arealen = np.sqrt(np.dot(area, area))
        alphak = np.arccos(np.dot(area, initarea) / (arealen * initarealen))

        if verbose and iteration % 50 == 0:
            print(f"  iter {iteration:5d}: volume={volume:.6f} alphak={alphak:.6f}")

        if alphak < epsilon:
            break

        tk = eta**ncoef * arealen * np.sin(alphak) / (3 * initm * maxpxk)
        coeff = np.dot(area, initarea) / initarealen**2
        f = area - coeff * initarea
        ek = f / (arealen * np.sin(alphak))

        newx = xk + tk * ek
        if np.all(newx > 0):
            d = newx
            posroute = 1
        else:
            posroute = -1
            denom = coeff * initarea - area
            w = xk / denom
            positive_w = w[w > 0]
            if positive_w.size == 0:
                raise RuntimeError(
                    "Minkowski iteration failed to find a valid backtracking "
                    "step (no positive w); check input areas/normals for a "
                    "closed, non-degenerate convex configuration."
                )
            wmin = float(positive_w.min())
            tk = 0.9 * arealen * wmin * np.sin(alphak)
            d = xk + tk * ek
    else:
        raise RuntimeError(
            f"Minkowski iteration did not converge to alphak < {epsilon} "
            f"within {max_iter} iterations (last alphak={alphak:.6f})."
        )

    if verbose:
        print(f"Converged after {iteration} iterations: alphak={alphak:.6f}")

    scoef = scoef * np.sqrt(initarealen / arealen)
    scoef = scoef / np.sqrt(scale)

    final_points = np.concatenate(
        [polygons[i] for i in hull_vertex_set], axis=0
    )
    vertices = scoef * (final_points - cmass[None, :])
    return vertices


def _merge_close_vertices(vertices: np.ndarray, tol: float | None) -> np.ndarray:
    """
    Collapse near-coincident points to their cluster centroid.

    `minkowski_reconstruct` emits one coordinate copy of each primal vertex
    per incident facet (see `_facet_polygons`), computed from that facet's
    own wedge equations. At the algorithm's convergence tolerance (`epsilon`,
    not machine precision), these independent estimates of the same physical
    vertex agree only to ~epsilon, not exactly -- so without merging, a
    single true vertex becomes a small cluster of near-duplicates and the
    final hull is fragmented into many spurious micro-facets.
    """
    if tol is None:
        bbox_diag = float(
            np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        )
        tol = 1e-3 * bbox_diag

    from scipy.spatial import cKDTree

    tree = cKDTree(vertices)
    pairs = tree.query_pairs(r=tol)

    parent = list(range(len(vertices)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    roots = np.array([find(i) for i in range(len(vertices))])
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    merged = np.zeros((len(unique_roots), 3))
    counts = np.zeros(len(unique_roots))
    np.add.at(merged, inverse, vertices)
    np.add.at(counts, inverse, 1)
    return merged / counts[:, None]


def build_convex_mesh(vertices: np.ndarray, merge_tol: float | None = None):
    """
    Take the convex hull of a (possibly redundant) vertex point cloud and
    return (vertices, triangles) with outward-oriented, consistently wound
    triangles suitable for STL export.

    `merge_tol` controls collapsing of near-duplicate vertices before
    hulling (see `_merge_close_vertices`); pass 0 to disable.
    """
    if merge_tol != 0:
        vertices = _merge_close_vertices(vertices, merge_tol)

    hull = ConvexHull(vertices)
    verts = hull.points
    tris = hull.simplices.copy()

    tri_pts = verts[tris]
    normals_geom = np.cross(
        tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]
    )
    outward = hull.equations[:, :3]
    flip = np.sum(normals_geom * outward, axis=1) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]

    return verts, tris


def save_stl(vertices: np.ndarray, triangles: np.ndarray, out_path):
    """Write a triangulated mesh to an STL file using numpy-stl."""
    from stl import mesh as stl_mesh

    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    m.vectors[:] = vertices[triangles]
    m.save(str(out_path))
    return m
