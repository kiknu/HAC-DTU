"""Evaluation-only solid overlap on a shared world-coordinate voxel lattice.

The world-center mapping follows origin/inr's lc2mesh/metrics.py (faac378).
Unlike mesh-local sparse indices, these coordinates retain translations. No
ghost boxes, alignment, normalization, or reference-driven fitting is applied
by the metric itself.
"""

from __future__ import annotations

import numpy as np
import trimesh


# Model cylinder radii from origin/inr:src/hac/constants.py.
CHALLENGE_RADII = {1: 1.12, 2: 1.42, 3: 0.88, 4: 1.475, 5: 1.22,
                  6: 0.925, 7: 1.205, 8: 1.24, 9: 0.67, 10: 3.95}


def normalize_mesh(mesh: trimesh.Trimesh, *, target_radius: float | None = None) -> tuple[trimesh.Trimesh, dict]:
    """Center and scale to z=[-1,1], optionally setting the xy cylinder radius.

    With target_radius, match origin/inr's normalize_mesh_to_challenge_cylinder
    using bounds centering: xy and z receive separate positive scale factors.
    Without it, preserve the original uniform-scale evaluation convention.
    """
    if not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh vertices must be nonempty and finite")
    bounds = np.asarray(mesh.bounds)
    center = bounds.mean(axis=0)
    height = float(bounds[1, 2] - bounds[0, 2])
    if height <= 0:
        raise ValueError("mesh must have positive z extent")
    scale = 2.0 / height
    result = mesh.copy()
    centered = np.asarray(mesh.vertices) - center
    if target_radius is not None:
        if not np.isfinite(target_radius) or target_radius <= 0:
            raise ValueError("target cylinder radius must be finite and positive")
        radius = float(np.linalg.norm(centered[:, :2], axis=1).max())
        if radius <= 0:
            raise ValueError("mesh must have positive radial extent")
        scale_xy = float(target_radius) / radius
        result.vertices = centered * np.array([scale_xy, scale_xy, scale])
        return result, dict(center=center.tolist(), scale_xy=scale_xy, scale_z=scale,
                            target_radius=float(target_radius), original_xy_radius=radius,
                            convention="bounding-box center; xy radius=R and z=[-1,1]; no rotation")
    result.vertices = centered * scale
    return result, dict(center=center.tolist(), scale=scale,
                        convention="bounding-box center; uniform scale to z=[-1,1]; no rotation")


def mesh_validity(mesh: trimesh.Trimesh, *, check_intersections: bool = False) -> dict:
    """Audit topology and optionally transverse crossings of a small mesh.

The crossing checker ignores coplanar contacts and is quadratic. Never apply
it to the large public references; their audit explicitly records this limit.
"""
    reasons = []
    if not len(mesh.faces) or not len(mesh.vertices):
        reasons.append("empty mesh")
    if not np.isfinite(mesh.vertices).all():
        reasons.append("non-finite vertices")
    if reasons:
        return dict(valid=False, reasons=reasons, intersection_check="not run")
    if not mesh.is_watertight:
        reasons.append("mesh is not watertight")
    if not mesh.is_winding_consistent:
        reasons.append("mesh winding is inconsistent")
    if np.any(mesh.area_faces <= 1e-12):
        reasons.append("mesh contains degenerate faces")
    if mesh.volume <= 0:
        reasons.append("mesh has nonpositive signed volume")
    crossing_check = "not checked"
    if check_intersections and not reasons:
        if len(mesh.faces) > 4096:
            raise ValueError("quadratic intersection check limited to 4096 faces")
        import torch
        from shape_model import _has_transverse_self_intersections

        crossing_check = "transverse nonadjacent faces; coplanar contacts excluded"
        has_crossings = _has_transverse_self_intersections(
            torch.tensor(np.asarray(mesh.vertices), dtype=torch.float64),
            torch.tensor(np.asarray(mesh.faces), dtype=torch.int64),
        )
        if has_crossings:
            reasons.append("mesh contains transverse self-intersections")
    return dict(valid=not reasons, reasons=reasons, watertight=bool(mesh.is_watertight),
                winding_consistent=bool(mesh.is_winding_consistent),
                intersection_check=crossing_check, faces=len(mesh.faces))


def occupied_voxels(mesh: trimesh.Trimesh, pitch: float, *, max_grid_cells: int = 32_000_000) -> np.ndarray:
    """Return sorted unique integer world coordinates of filled voxels.

Both meshes use the lattice with world origin zero. Dense interior filling
is bounded before allocation; integer rows avoid Python tuple/set overhead.
Callers must validate the mesh before treating its filling as solid volume.
"""
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("voxel pitch must be finite and positive")
    dimensions = np.ceil(np.asarray(mesh.extents) / pitch) + 3
    if np.prod(dimensions) > max_grid_cells:
        raise ValueError("voxel grid exceeds the cell budget; increase pitch")
    grid = mesh.voxelized(pitch, method="subdivide").fill()
    coordinates = np.rint(grid.points / pitch).astype(np.int64)
    return np.unique(coordinates, axis=0)


def overlap_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    """Dice and IoU for two unique sets of integer voxel coordinates."""
    def rows(values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != 3 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("voxel coordinates must be an integer (N, 3) array")
        values = np.ascontiguousarray(np.unique(values, axis=0), dtype=np.int64)
        return values.view(np.dtype((np.void, 3 * values.dtype.itemsize))).ravel()

    a, b = rows(predicted), rows(target)
    if not len(a) or not len(b):
        raise ValueError("solid voxel sets must be nonempty")
    intersection = int(len(np.intersect1d(a, b, assume_unique=True)))
    union = len(a) + len(b) - intersection
    return dict(solid_dice=2.0 * intersection / (len(a) + len(b)),
                solid_iou=intersection / union, predicted_voxels=len(a),
                reference_voxels=len(b), intersection_voxels=intersection,
                union_voxels=union)


def solid_voxel_metrics(predicted: trimesh.Trimesh, target: trimesh.Trimesh,
                        *, pitch: float = 0.02, check_intersections: bool = True) -> dict:
    """Score meshes as supplied; invalid geometry returns null scores and audits.

Target topology is checked, but target self-intersections are not exhaustively
checked. The optional transverse crossing check applies only to prediction.
"""
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("voxel pitch must be finite and positive")
    audits = dict(predicted=mesh_validity(predicted, check_intersections=check_intersections),
                  reference=mesh_validity(target))
    result = dict(pitch=float(pitch), validity=audits,
                  valid=all(audit["valid"] for audit in audits.values()))
    if not result["valid"]:
        return dict(result, solid_dice=None, solid_iou=None)
    return dict(result, **overlap_metrics(occupied_voxels(predicted, pitch),
                                         occupied_voxels(target, pitch)))
