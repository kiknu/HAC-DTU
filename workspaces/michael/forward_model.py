#!/usr/bin/env python3
"""
Integrated soft-rasterizer for non-convex asteroid lightcurves.

This module implements a differentiable, disk-integrated brightness model for
asteroid lightcurves using a soft visibility rasterizer. It supports both
per-frame and batched multi-camera evaluations, with optional torch.compile
acceleration.

Key features:
- Orthographic projection with soft Z-buffer (larger z = closer).
- Lambertian, Lommel-Seeliger, and combined BRDFs with foreshortening.
- Backface culling and soft barycentric coverage for stability.
- Batched visibility tables for efficient multi-camera computation.
- Rotation conventions matching the original PyTorch3D pipeline.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.checkpoint import checkpoint

from shape_model import TriangleMesh

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")

# --------------------------------------------------------------------------- #
# Data I/O
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    model_num: int
    origin: Literal["real", "simulated"]
    curve_type: Literal["binary", "intensity"]
    data: pd.DataFrame


PROJECT_DIR = Path(__file__).resolve().parent
CHALLENGE_DATA_DIR = PROJECT_DIR / "challenge_data"


def get_model_folder(model_num: int) -> Path:
    suffix = "public" if model_num <= 3 else "secret"
    model_dir = CHALLENGE_DATA_DIR / f"AsteroidModel{model_num:02d}_shape_{suffix}"
    assert model_dir.is_dir(), f"Model directory not found at {model_dir}."
    return model_dir


def get_model_stl_path(model_num: int) -> Path:
    if model_num > 3:
        raise ValueError(f"Model {model_num} is secret.")
    path = get_model_folder(model_num) / f"asteroid{model_num:d}.stl"
    assert path.is_file(), f"STL not found at {path}."
    return path


def load_asteroid_mesh(model_num: int, device: torch.device):
    mesh = trimesh.load(get_model_stl_path(model_num), force="mesh")
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return verts, faces

def load_data(
    model_num: int,
    origin: Literal["real", "simulated"],
    curve_type: Literal["binary", "intensity"],
) -> Dataset:
    if origin not in ("real", "simulated"):
        raise ValueError("origin must be 'real' or 'simulated'")
    if curve_type not in ("binary", "intensity"):
        raise ValueError("curve_type must be 'binary' or 'intensity'")
    model_dir = get_model_folder(model_num)
    lc_dir = model_dir / f"Asteroid{model_num:d}_lightcurve_data"
    assert lc_dir.is_dir()

    fname = f"Asteroid{model_num:02d}_lightcurve_{curve_type}"
    if origin == "simulated":
        fname += "_blender"
    fname += ".txt"

    data = pd.read_csv(lc_dir / fname, header=None, index_col=0)
    data.index.name = "frame"
    data = data.rename(
        columns={
            1: "0_hor1",
            2: "0_hor2",
            3: "0_top",
            4: "0_bottom",
            5: "45_hor1",
            6: "45_hor2",
            7: "45_top",
            8: "45_bottom",
            9: "90_hor1",
            10: "90_hor2",
            11: "90_top",
            12: "90_bottom",
            13: "135_hor1",
            14: "135_hor2",
            15: "135_top",
            16: "135_bottom",
            17: "225_hor1",
            18: "225_hor2",
            19: "225_top",
            20: "225_bottom",
            21: "270_hor1",
            22: "270_hor2",
            23: "270_top",
            24: "270_bottom",
            25: "315_hor1",
            26: "315_hor2",
            27: "315_top",
            28: "315_bottom",
        }
    )
    return Dataset(
        model_num=model_num,
        origin=origin,
        curve_type=curve_type,
        data=data,
    )


# --------------------------------------------------------------------------- #
# Core geometry and visibility helpers
# --------------------------------------------------------------------------- #

def _ortho_basis(direction: torch.Tensor):
    """Build right-handed orthonormal basis [u, w, d] for projection."""
    d = F.normalize(direction, dim=-1)
    up_z = d.new_tensor([0.0, 0.0, 1.0])
    up_y = d.new_tensor([0.0, 1.0, 0.0])
    up = torch.where(d[2].abs() < 0.9, up_z, up_y)
    u = F.normalize(torch.linalg.cross(d, up, dim=-1), dim=-1)
    w = torch.linalg.cross(d, u, dim=-1)
    return torch.stack([u, w, d], dim=-1)  # (3, 3)


def _ortho_bases(dirs: torch.Tensor) -> torch.Tensor:
    """Batched version of _ortho_basis: (D, 3) -> (D, 3, 3)."""
    d = F.normalize(dirs, dim=-1)
    up_z = d.new_tensor([0.0, 0.0, 1.0]).expand_as(d)
    up_y = d.new_tensor([0.0, 1.0, 0.0]).expand_as(d)
    up = torch.where((d[:, 2].abs() < 0.9).unsqueeze(-1), up_z, up_y)
    u = F.normalize(torch.linalg.cross(d, up, dim=-1), dim=-1)
    w = torch.linalg.cross(d, u, dim=-1)
    return torch.stack([u, w, d], dim=-1)


def _soft_visibility(
    centroids: torch.Tensor,
    face_xy: torch.Tensor,
    face_z: torch.Tensor,
    gamma: float = 0.02,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Differentiable orthographic visibility via Z-softmax.

    Larger z = closer. For each face i, visibility is
        V_i = exp(z_i/g) / [ exp(z_i/g) + Σ_{j≠i} c_ij exp(z_ij/g) ]
    where c_ij is barycentric coverage of query point i on face j.
    """
    n_faces = centroids.shape[0]

    x0, x1, x2 = face_xy[:, 0, 0], face_xy[:, 1, 0], face_xy[:, 2, 0]
    y0, y1, y2 = face_xy[:, 0, 1], face_xy[:, 1, 1], face_xy[:, 2, 1]
    z0, z1, z2 = face_z[:, 0], face_z[:, 1], face_z[:, 2]

    e1x, e1y = x1 - x0, y1 - y0
    e2x, e2y = x2 - x0, y2 - y0
    det = e1x * e2y - e1y * e2x  # (F,)

    degenerate = det.abs() < 1e-9
    safe_det = torch.where(degenerate, torch.ones_like(det), det)
    inv_det = torch.where(degenerate, torch.zeros_like(det), 1.0 / safe_det)

    qx = centroids[:, 0:1]
    qy = centroids[:, 1:2]

    dpx = qx - x0.unsqueeze(0)
    dpy = qy - y0.unsqueeze(0)

    lam1 = (dpx * e2y.unsqueeze(0) - dpy * e2x.unsqueeze(0)) * inv_det.unsqueeze(0)
    lam2 = (dpy * e1x.unsqueeze(0) - dpx * e1y.unsqueeze(0)) * inv_det.unsqueeze(0)
    lam0 = 1.0 - lam1 - lam2

    inside = (
        torch.sigmoid(lam0 / eps)
        * torch.sigmoid(lam1 / eps)
        * torch.sigmoid(lam2 / eps)
    )
    inside = inside * (~degenerate).to(inside.dtype).unsqueeze(0)

    z_interp = (
        lam0 * z0.unsqueeze(0)
        + lam1 * z1.unsqueeze(0)
        + lam2 * z2.unsqueeze(0)
    )
    z_interp = torch.clamp(
        z_interp,
        min=face_z.min(dim=1).values.unsqueeze(0),
        max=face_z.max(dim=1).values.unsqueeze(0),
    )

    eye = torch.eye(n_faces, dtype=torch.bool, device=centroids.device)
    inside = inside.masked_fill(eye, 0.0)

    z_query = (z0 + z1 + z2) / 3.0

    return _visibility_from_depths(
        inside=inside,
        z_interp=z_interp,
        z_query=z_query,
        gamma=gamma,
    )


def _direction_visibility(
    verts: torch.Tensor,
    faces: torch.Tensor,
    direction: torch.Tensor,
    gamma: float,
    eps: float,
    keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft visibility for one direction, optionally restricted to faces."""
    basis = _ortho_basis(direction)
    proj = verts @ basis
    face_p = proj[faces]  # (F, 3, 3)

    if keep is None:
        centroids = face_p.mean(dim=1)[:, :2]
        return _soft_visibility(centroids, face_p[:, :, :2], face_p[:, :, 2], gamma, eps)

    idx = keep.nonzero(as_tuple=True)[0]
    face_sub = face_p[idx]
    centroids = face_sub.mean(dim=1)[:, :2]
    vis_sub = _soft_visibility(centroids, face_sub[:, :, :2], face_sub[:, :, 2], gamma, eps)
    vis = torch.zeros(faces.shape[0], dtype=verts.dtype, device=verts.device)
    vis.index_copy_(0, idx, vis_sub)
    return vis


def _surface_geometry(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return face areas, geometric normals, and smooth centroid normals."""
    triangles = verts[faces]
    cross = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    )
    areas = 0.5 * cross.norm(dim=-1)
    geometric = F.normalize(cross, dim=-1)

    # This matches PyTorch3D's area-weighted vertex-normal construction.  A
    # centroid quadrature cannot reproduce per-pixel interpolation exactly,
    # but averaging the three vertex normals removes the largest flat-shading
    # discrepancy while keeping this path inexpensive and differentiable.
    vertex_sums = torch.zeros_like(verts)
    vertex_sums = vertex_sums.index_add(
        0,
        faces.reshape(-1),
        cross[:, None, :].expand(-1, 3, -1).reshape(-1, 3),
    )
    vertex_normals = F.normalize(vertex_sums, dim=-1)
    smooth = F.normalize(vertex_normals[faces].mean(dim=1), dim=-1)
    return areas, geometric, smooth


def _threshold_gate(
    radiance: torch.Tensor,
    threshold: float | None,
    beta: float,
) -> torch.Tensor:
    """Differentiable approximation to the challenge's pixel cutoff."""
    if threshold is None or threshold <= 0.0:
        return torch.ones_like(radiance)
    tau = torch.as_tensor(threshold, dtype=radiance.dtype, device=radiance.device)
    slope = torch.as_tensor(beta, dtype=radiance.dtype, device=radiance.device)
    soft = torch.sigmoid(slope * (radiance - tau))
    background = torch.sigmoid(-slope * tau)
    return ((soft - background) / (1.0 - background).clamp_min(1e-8)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Single-frame brightness
# --------------------------------------------------------------------------- #

def compute_brightness(
    verts: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    view_dir: torch.Tensor,
    gamma: float = 0.02,
    eps: float = 1e-3,
    brdf_type: Literal["lambertian", "lommel_seeliger", "combined"] = "combined",
    lambertian_fraction: float = 0.5,
    smooth_shading: bool = True,
    intensity_threshold: float | None = None,
    intensity_beta: float = 64.0,
    cull_backfaces: bool = True,
) -> torch.Tensor:
    """Disk-integrated brightness for a single (sun, view) pair."""
    areas, geometric_normals, smooth_normals = _surface_geometry(verts, faces)
    nhat = smooth_normals if smooth_shading else geometric_normals

    mu0 = torch.clamp((nhat * sun_dir).sum(dim=-1), min=0.0)
    mu = torch.clamp((nhat * view_dir).sum(dim=-1), min=0.0)
    geometric_mu0 = (geometric_normals * sun_dir).sum(dim=-1)
    geometric_mu = (geometric_normals * view_dir).sum(dim=-1)

    if brdf_type == "lambertian":
        radiance = mu0
    elif brdf_type == "lommel_seeliger":
        radiance = mu0 / (mu0 + mu + 1e-8)
    else:  # combined
        weight = torch.as_tensor(
            lambertian_fraction, dtype=mu.dtype, device=mu.device
        ).clamp(0.0, 1.0)
        lambert = mu0
        lommel_seeliger = mu0 / (mu0 + mu + 1e-8)
        radiance = weight * lambert + (1.0 - weight) * lommel_seeliger

    keep_v = (geometric_mu > 0) if cull_backfaces else None
    vis = _direction_visibility(verts, faces, view_dir, gamma, eps, keep_v)

    keep_s = (geometric_mu0 > 0) if cull_backfaces else None
    ill = _direction_visibility(verts, faces, sun_dir, gamma, eps, keep_s)

    radiance = radiance * ill
    radiance = radiance * _threshold_gate(
        radiance, intensity_threshold, intensity_beta
    )
    projected_mu = geometric_mu.clamp_min(0.0)
    return (areas * projected_mu * radiance * vis).sum()


_COMPILED_BRIGHTNESS = None


def _get_compiled_brightness():
    global _COMPILED_BRIGHTNESS
    if _COMPILED_BRIGHTNESS is None:
        _COMPILED_BRIGHTNESS = torch.compile(compute_brightness, dynamic=False)
    return _COMPILED_BRIGHTNESS


def compute_lightcurve(
    verts: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    view_dir: torch.Tensor,
    angles: torch.Tensor,
    gamma: float = 0.02,
    eps: float = 1e-3,
    brdf_type: Literal["lambertian", "lommel_seeliger", "combined"] = "combined",
    lambertian_fraction: float = 0.5,
    smooth_shading: bool = True,
    intensity_threshold: float | None = None,
    intensity_beta: float = 64.0,
    cull_backfaces: bool = True,
    rotation_sign: float = -1.0,
    phase_offset: float = 0.0,
    use_compile: bool = False,
) -> torch.Tensor:
    """
    Lightcurve for N rotation angles.

    Rotates the mesh by psi_k = rotation_sign * angles[k] + phase_offset
    about +z (implemented by rotating sun/view directions by R_z(-psi_k)).
    """
    N = angles.shape[0]
    psi = rotation_sign * angles + phase_offset
    c, s = torch.cos(psi), torch.sin(psi)

    sun_dirs = torch.stack(
        [
            c * sun_dir[0] + s * sun_dir[1],
            -s * sun_dir[0] + c * sun_dir[1],
            torch.full_like(psi, sun_dir[2]),
        ],
        dim=1,
    )
    view_dirs = torch.stack(
        [
            c * view_dir[0] + s * view_dir[1],
            -s * view_dir[0] + c * view_dir[1],
            torch.full_like(psi, view_dir[2]),
        ],
        dim=1,
    )

    fn = _get_compiled_brightness() if use_compile else compute_brightness
    cull = False if use_compile else cull_backfaces

    values = []
    for k in range(N):
        values.append(
            fn(
                verts,
                faces,
                sun_dirs[k],
                view_dirs[k],
                gamma=gamma,
                eps=eps,
                brdf_type=brdf_type,
                lambertian_fraction=lambertian_fraction,
                smooth_shading=smooth_shading,
                intensity_threshold=intensity_threshold,
                intensity_beta=intensity_beta,
                cull_backfaces=cull,
            )
        )
    return torch.stack(values)


# --------------------------------------------------------------------------- #
# Batched visibility table and multi-camera lightcurves
# --------------------------------------------------------------------------- #

def _rotate_dirs_z(d0: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Apply R_z(-psi_k) to fixed world direction(s)."""
    shape = (psi.shape[0],) + (1,) * (d0.dim() - 1)
    c, s = torch.cos(psi).view(shape), torch.sin(psi).view(shape)
    x, y, z = d0[..., 0], d0[..., 1], d0[..., 2]
    xr = c * x + s * y
    yr = -s * x + c * y
    return torch.stack([xr, yr, z.expand_as(xr)], dim=-1)

def _visibility_from_depths(
    inside: torch.Tensor,
    z_interp: torch.Tensor,
    z_query: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Compute

        V_i = 1 / (1 + sum_j c_ij exp((z_ij - z_i) / gamma))

    using a softmax with an explicit self logit of zero.

    `inside` and `z_interp` have shape (..., Q, O), while `z_query`
    has shape (..., Q). Larger z is closer.
    """
    tiny = torch.finfo(inside.dtype).tiny

    log_coverage = torch.where(
        inside > 0.0,
        torch.log(inside.clamp_min(tiny)),
        torch.full_like(inside, -torch.inf),
    )

    other_logits = (
        log_coverage
        + (z_interp - z_query.unsqueeze(-1)) / gamma
    )

    # The self-contribution is exactly exp(0) = 1.
    self_logits = torch.zeros_like(z_query).unsqueeze(-1)
    logits = torch.cat([self_logits, other_logits], dim=-1)

    return torch.softmax(logits, dim=-1)[..., 0]

def _soft_visibility_batch(
    face_xy,
    face_z,
    gamma,
    eps,
    valid=None,
    query_xy=None,
    query_z=None,
    query_face_indices=None,
):
    """Batched soft visibility: (B, K, 3, 2) and (B, K, 3) -> (B, K)."""
    B, K = face_z.shape[:2]
    x0, x1, x2 = face_xy[..., 0, 0], face_xy[..., 1, 0], face_xy[..., 2, 0]
    y0, y1, y2 = face_xy[..., 0, 1], face_xy[..., 1, 1], face_xy[..., 2, 1]
    z0, z1, z2 = face_z[..., 0], face_z[..., 1], face_z[..., 2]

    e1x, e1y = x1 - x0, y1 - y0
    e2x, e2y = x2 - x0, y2 - y0
    det = e1x * e2y - e1y * e2x  # (B, K)

    degenerate = det.abs() < 1e-9
    safe_det = torch.where(degenerate, torch.ones_like(det), det)
    inv_det = torch.where(degenerate, torch.zeros_like(det), 1.0 / safe_det)

    if query_xy is None:
        cx, cy = (x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0
    else:
        cx, cy = query_xy[..., 0], query_xy[..., 1]

    dpx = cx.unsqueeze(2) - x0.unsqueeze(1)
    dpy = cy.unsqueeze(2) - y0.unsqueeze(1)
    lam1 = (dpx * e2y.unsqueeze(1) - dpy * e2x.unsqueeze(1)) * inv_det.unsqueeze(1)
    lam2 = (dpy * e1x.unsqueeze(1) - dpx * e1y.unsqueeze(1)) * inv_det.unsqueeze(1)
    lam0 = 1.0 - lam1 - lam2

    inside = torch.sigmoid(lam0 / eps) * torch.sigmoid(lam1 / eps) * torch.sigmoid(lam2 / eps)
    occ_ok = ~degenerate
    if valid is not None:
        occ_ok = occ_ok & valid
    inside = inside * occ_ok.to(inside.dtype).unsqueeze(1)

    z_interp = lam0 * z0.unsqueeze(1) + lam1 * z1.unsqueeze(1) + lam2 * z2.unsqueeze(1)
    z_interp = torch.clamp(
        z_interp,
        min=face_z.min(dim=-1).values.unsqueeze(1),
        max=face_z.max(dim=-1).values.unsqueeze(1),
    )

    if query_face_indices is None:
        self_mask = torch.eye(K, dtype=torch.bool, device=face_z.device)
    else:
        self_mask = F.one_hot(
            query_face_indices, num_classes=K
        ).to(torch.bool)
    inside = inside.masked_fill(self_mask.unsqueeze(0), 0.0)

    if query_z is None:
        z_query = (z0 + z1 + z2) / 3.0
    else:
        z_query = query_z

    return _visibility_from_depths(
        inside=inside,
        z_interp=z_interp,
        z_query=z_query,
        gamma=gamma,
    )


def _visibility_chunk(
    verts,
    faces,
    nhat,
    dirs,
    gamma,
    eps,
    k_keep,
    mask_backface_occluders=True,
    vis_dtype=None,
    projection="orthographic",
    camera_distance_radii=5.0,
    visibility_samples=1,
):
    """Soft visibility for a chunk of directions -> (B, F)."""
    B, n_faces = dirs.shape[0], faces.shape[0]
    bases = _ortho_bases(dirs)
    proj = torch.einsum("vi,bij->bvj", verts, bases)  # (B, V, 3)
    face_p = proj[:, faces]  # (B, F, 3, 3)
    query_xy = None
    query_z = None
    if projection == "orthographic":
        mu = torch.einsum("fi,bi->bf", nhat, dirs)  # (B, F)
    elif projection == "perspective":
        radius = torch.linalg.vector_norm(verts, dim=-1).max()
        distance = radius * camera_distance_radii
        tiny = torch.finfo(verts.dtype).eps * distance.clamp_min(1.0)
        vertex_depth = (distance - proj[..., 2]).clamp_min(tiny)

        # Coordinates live on the plane through the mesh origin.  The depth
        # transform is monotone and tends to orthographic z as D -> infinity,
        # which keeps the existing visibility-softness scale meaningful.
        screen_xy = distance * proj[..., :2] / vertex_depth[..., None]
        screen_z = distance * (distance / vertex_depth - 1.0)
        face_p = torch.cat([screen_xy, screen_z[..., None]], dim=-1)[:, faces]

        centroids = verts[faces].mean(dim=1)
        centroid_proj = torch.einsum("fi,bij->bfj", centroids, bases)
        centroid_depth = (distance - centroid_proj[..., 2]).clamp_min(tiny)
        query_xy = distance * centroid_proj[..., :2] / centroid_depth[..., None]
        query_z = distance * (distance / centroid_depth - 1.0)
        camera_positions = dirs * distance
        face_view_dirs = F.normalize(
            camera_positions[:, None, :] - centroids[None, :, :], dim=-1
        )
        mu = torch.einsum("fi,bfi->bf", nhat, face_view_dirs)
    else:
        raise ValueError("projection must be 'orthographic' or 'perspective'")

    query_face_indices = None
    if visibility_samples == 3:
        barycentric = verts.new_tensor(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ]
        )
        sample_points = torch.einsum(
            "sk,fki->fsi", barycentric, verts[faces]
        )
        sample_proj = torch.einsum(
            "fsi,bij->bfsj", sample_points, bases
        )
        if projection == "orthographic":
            sample_xy = sample_proj[..., :2]
            sample_z = sample_proj[..., 2]
        else:
            sample_depth = (distance - sample_proj[..., 2]).clamp_min(tiny)
            sample_xy = (
                distance * sample_proj[..., :2] / sample_depth[..., None]
            )
            sample_z = distance * (distance / sample_depth - 1.0)
        query_xy = sample_xy.reshape(B, n_faces * visibility_samples, 2)
        query_z = sample_z.reshape(B, n_faces * visibility_samples)
        query_face_indices = torch.arange(
            n_faces, device=faces.device
        ).repeat_interleave(visibility_samples)
    elif visibility_samples != 1:
        raise ValueError("visibility_samples must be 1 or 3")

    if k_keep is not None and k_keep < n_faces:
        if visibility_samples != 1:
            raise ValueError("sub-face visibility sampling does not support culling")
        idx = mu.topk(k_keep, dim=1, sorted=False).indices  # (B, K)
        face_sub = torch.gather(face_p, 1, idx[:, :, None, None].expand(B, k_keep, 3, 3))
        mu_sub = torch.gather(mu, 1, idx)
        if vis_dtype is not None:
            face_sub = face_sub.to(vis_dtype)
        valid = (mu_sub > 0.0) if mask_backface_occluders else None
        query_xy_sub = (
            None
            if query_xy is None
            else torch.gather(
                query_xy, 1, idx[:, :, None].expand(B, k_keep, 2)
            )
        )
        query_z_sub = (
            None if query_z is None else torch.gather(query_z, 1, idx)
        )
        vis_sub = _soft_visibility_batch(
            face_sub[..., :2],
            face_sub[..., 2],
            gamma,
            eps,
            valid,
            query_xy_sub,
            query_z_sub,
        )
        vis_sub = vis_sub.to(verts.dtype) * (mu_sub > 0.0)
        return proj.new_zeros(B, n_faces).scatter(1, idx, vis_sub)

    fp = face_p.to(vis_dtype) if vis_dtype is not None else face_p
    valid = (mu > 0.0) if mask_backface_occluders else None
    vis = _soft_visibility_batch(
        fp[..., :2],
        fp[..., 2],
        gamma,
        eps,
        valid,
        query_xy,
        query_z,
        query_face_indices,
    )
    vis = vis.to(verts.dtype)
    if mask_backface_occluders:
        if visibility_samples == 1:
            vis = vis * (mu > 0.0)
        else:
            vis = vis.reshape(B, n_faces, visibility_samples)
            vis = vis * (mu > 0.0)[..., None]
    elif visibility_samples != 1:
        vis = vis.reshape(B, n_faces, visibility_samples)
    return vis


_COMPILED_VIS_CHUNK = None


def _get_compiled_vis_chunk():
    global _COMPILED_VIS_CHUNK
    if _COMPILED_VIS_CHUNK is None:
        _COMPILED_VIS_CHUNK = torch.compile(_visibility_chunk, dynamic=False)
    return _COMPILED_VIS_CHUNK


def visibility_table(
    verts,
    faces,
    nhat,
    dirs,
    *,
    gamma,
    eps,
    cull_frac=None,
    dir_chunk=4,
    use_compile=True,
    use_checkpoint=False,
    mask_backface_occluders=True,
    vis_dtype=None,
    projection="orthographic",
    camera_distance_radii=5.0,
    visibility_samples=1,
):
    """(D, 3) directions -> (D, F) soft visibility, chunked over D."""
    D, n_faces = dirs.shape[0], faces.shape[0]
    if visibility_samples not in (1, 3):
        raise ValueError("visibility_samples must be 1 or 3")
    if visibility_samples != 1 and cull_frac is not None:
        raise ValueError("sub-face visibility sampling does not support culling")
    k_keep = None if cull_frac is None else min(n_faces, int(cull_frac * n_faces))
    fn = _get_compiled_vis_chunk() if use_compile else _visibility_chunk
    outs = []
    for i in range(0, D, dir_chunk):
        d = dirs[i : i + dir_chunk]
        n_real = d.shape[0]
        if n_real < dir_chunk:
            d = torch.cat([d, d[-1:].expand(dir_chunk - n_real, -1)], dim=0)
        args = (
            verts,
            faces,
            nhat,
            d,
            gamma,
            eps,
            k_keep,
            mask_backface_occluders,
            vis_dtype,
            projection,
            camera_distance_radii,
            visibility_samples,
        )
        v = (
            checkpoint(fn, *args, use_reentrant=False)
            if use_checkpoint
            else fn(*args)
        )
        outs.append(v[:n_real])
    return torch.cat(outs, dim=0)


def _integrate_chunk(
    areas,
    geometric_nhat,
    shading_nhat,
    sun_dirs,
    view_dirs,
    solar_vis_table,
    observer_vis_table,
    inv_sun,
    inv_view,
    observer_area_scale,
    brdf_type,
    lambertian_fraction=0.5,
    intensity_threshold=None,
    intensity_beta=64.0,
):
    """Photometric contraction for a chunk of frames -> (n, C)."""
    ill = solar_vis_table[inv_sun]  # (n, F)
    vis = observer_vis_table[inv_view]  # (n, C, F)
    mu0 = torch.einsum("fi,ni->nf", shading_nhat, sun_dirs).clamp_min(0.0)
    if view_dirs.ndim == 3:
        mu = torch.einsum(
            "fi,nci->ncf", shading_nhat, view_dirs
        ).clamp_min(0.0)
        projected_mu = torch.einsum(
            "fi,nci->ncf", geometric_nhat, view_dirs
        ).clamp_min(0.0)
    elif view_dirs.ndim == 4:
        mu = torch.einsum(
            "fi,ncfi->ncf", shading_nhat, view_dirs
        ).clamp_min(0.0)
        projected_mu = torch.einsum(
            "fi,ncfi->ncf", geometric_nhat, view_dirs
        ).clamp_min(0.0)
    else:
        raise ValueError("view_dirs must have shape (N,C,3) or (N,C,F,3)")
    if observer_area_scale is not None:
        projected_mu = projected_mu * observer_area_scale
    m0 = mu0[:, None, :]
    if brdf_type == "lambertian":
        radiance = m0
    elif brdf_type == "lommel_seeliger":
        radiance = m0 / (m0 + mu + 1e-8)
    else:  # combined
        weight = torch.as_tensor(
            lambertian_fraction, dtype=m0.dtype, device=m0.device
        ).clamp(0.0, 1.0)
        lambert = m0
        lommel_seeliger = m0 / (m0 + mu + 1e-8)
        radiance = weight * lambert + (1.0 - weight) * lommel_seeliger

    # Shadowing changes the actual pixel radiance, so the camera response is
    # applied after solar visibility rather than only to the unshadowed BRDF.
    if ill.ndim == 2:
        radiance = radiance * ill[:, None, :]
        radiance = radiance * _threshold_gate(
            radiance, intensity_threshold, intensity_beta
        )
        return torch.einsum(
            "ncf,f->nc", radiance * projected_mu * vis, areas
        )

    sample_radiance = radiance[..., None] * ill[:, None, :, :]
    sample_radiance = sample_radiance * _threshold_gate(
        sample_radiance, intensity_threshold, intensity_beta
    )
    return torch.einsum(
        "ncfs,f->nc",
        sample_radiance * projected_mu[..., None] * vis,
        areas / ill.shape[-1],
    )

def _integrate_chunk_binary(
    areas,
    geometric_nhat,
    shading_nhat,
    sun_dirs,
    view_dirs,
    solar_vis_table,
    observer_vis_table,
    inv_sun,
    inv_view,
    observer_area_scale,
    tau,
    beta,
    brdf_type="combined",
    lambertian_fraction=0.5,
):
    ill = solar_vis_table[inv_sun]
    vis = observer_vis_table[inv_view]

    mu0 = torch.einsum("fi,ni->nf", shading_nhat, sun_dirs).clamp_min(0.0)
    if view_dirs.ndim == 3:
        mu = torch.einsum(
            "fi,nci->ncf", shading_nhat, view_dirs
        ).clamp_min(0.0)
        projected_mu = torch.einsum(
            "fi,nci->ncf", geometric_nhat, view_dirs
        ).clamp_min(0.0)
    elif view_dirs.ndim == 4:
        mu = torch.einsum(
            "fi,ncfi->ncf", shading_nhat, view_dirs
        ).clamp_min(0.0)
        projected_mu = torch.einsum(
            "fi,ncfi->ncf", geometric_nhat, view_dirs
        ).clamp_min(0.0)
    else:
        raise ValueError("view_dirs must have shape (N,C,3) or (N,C,F,3)")
    if observer_area_scale is not None:
        projected_mu = projected_mu * observer_area_scale
    m0 = mu0[:, None, :]

    if brdf_type == "lambertian":
        radiance = m0
    elif brdf_type == "lommel_seeliger":
        radiance = m0 / (m0 + mu + 1e-8)
    else:
        weight = torch.as_tensor(
            lambertian_fraction, dtype=m0.dtype, device=m0.device
        ).clamp(0.0, 1.0)
        lambert = m0
        lommel_seeliger = m0 / (m0 + mu + 1e-8)
        radiance = weight * lambert + (1.0 - weight) * lommel_seeliger

    tau_t = torch.as_tensor(
        tau, dtype=radiance.dtype, device=radiance.device
    )
    beta_t = torch.as_tensor(
        beta, dtype=radiance.dtype, device=radiance.device
    )

    s0 = torch.sigmoid(-beta_t * tau_t)
    if ill.ndim == 2:
        radiance = radiance * ill[:, None, :]
        s = torch.sigmoid(beta_t * (radiance - tau_t))
        lit = ((s - s0) / (1.0 - s0).clamp_min(1e-8)).clamp_min(0.0)
        return torch.einsum(
            "ncf,f->nc", projected_mu * vis * lit, areas
        )

    sample_radiance = radiance[..., None] * ill[:, None, :, :]
    s = torch.sigmoid(beta_t * (sample_radiance - tau_t))
    lit = ((s - s0) / (1.0 - s0).clamp_min(1e-8)).clamp_min(0.0)
    return torch.einsum(
        "ncfs,f->nc",
        projected_mu[..., None] * vis * lit,
        areas / ill.shape[-1],
    )


@dataclass(frozen=True)
class ObservationGeometry:
    """Shape-independent direction bookkeeping for one lightcurve dataset."""

    angles: torch.Tensor
    sun_dir0: torch.Tensor
    view_dirs0: torch.Tensor
    sun_dirs: torch.Tensor
    view_dirs: torch.Tensor
    unique_dirs: torch.Tensor
    inv_sun: torch.Tensor
    inv_view: torch.Tensor
    unique_sun_dirs: torch.Tensor
    unique_view_dirs: torch.Tensor
    inv_sun_separate: torch.Tensor
    inv_view_separate: torch.Tensor

    @classmethod
    def build(
        cls,
        angles: torch.Tensor,
        sun_dir0: torch.Tensor,
        view_dirs0: torch.Tensor,
        *,
        rotation_sign: float = -1.0,
        phase_offset: float = 0.0,
        dedupe_tol: float = 1e-4,
    ) -> "ObservationGeometry":
        if angles.ndim != 1:
            raise ValueError("angles must have shape (N,)")
        if view_dirs0.ndim != 2 or view_dirs0.shape[1] != 3:
            raise ValueError("view_dirs0 must have shape (C, 3)")

        psi = rotation_sign * angles + phase_offset
        sun_dirs = _rotate_dirs_z(sun_dir0, psi)
        view_dirs = _rotate_dirs_z(view_dirs0, psi)
        all_dirs = torch.cat(
            [sun_dirs, view_dirs.reshape(-1, 3)], dim=0
        )
        key = torch.round(all_dirs / dedupe_tol)
        unique_key, inverse = torch.unique(
            key, dim=0, return_inverse=True
        )
        unique_dirs = F.normalize(unique_key * dedupe_tol, dim=-1)
        n_frames = angles.shape[0]
        n_cameras = view_dirs0.shape[0]
        sun_key = torch.round(sun_dirs / dedupe_tol)
        unique_sun_key, inv_sun_separate = torch.unique(
            sun_key, dim=0, return_inverse=True
        )
        view_key = torch.round(view_dirs.reshape(-1, 3) / dedupe_tol)
        unique_view_key, inv_view_separate = torch.unique(
            view_key, dim=0, return_inverse=True
        )
        return cls(
            angles=angles,
            sun_dir0=sun_dir0,
            view_dirs0=view_dirs0,
            sun_dirs=sun_dirs,
            view_dirs=view_dirs,
            unique_dirs=unique_dirs,
            inv_sun=inverse[:n_frames],
            inv_view=inverse[n_frames:].reshape(n_frames, n_cameras),
            unique_sun_dirs=F.normalize(
                unique_sun_key * dedupe_tol, dim=-1
            ),
            unique_view_dirs=F.normalize(
                unique_view_key * dedupe_tol, dim=-1
            ),
            inv_sun_separate=inv_sun_separate,
            inv_view_separate=inv_view_separate.reshape(
                n_frames, n_cameras
            ),
        )


@dataclass(frozen=True)
class PreparedVisibility:
    """Visibility frozen at a solver linearization point."""

    table: torch.Tensor
    gamma: float
    observer_table: torch.Tensor | None = None
    projection: str = "orthographic"
    camera_distance_radii: float | None = None
    visibility_samples: int = 1


@dataclass(frozen=True)
class PhotometricPrediction:
    intensity: torch.Tensor
    binary: torch.Tensor | None = None


class PhotometricForwardModel:
    """Non-convex mesh renderer independent of shape parameterization.

    Every face remains a potential occluder.  Soft depth tests are evaluated
    independently in observer and illumination directions, capturing both
    self-occlusion and cast shadows for closed, consistently oriented meshes.
    """

    def __init__(
        self,
        geometry: ObservationGeometry,
        *,
        eps: float = 1e-3,
        brdf_type: Literal[
            "lambertian", "lommel_seeliger", "combined"
        ] = "combined",
        lambertian_fraction: float = 0.5,
        shading: Literal["flat", "smooth"] = "smooth",
        projection: Literal["orthographic", "perspective"] = "orthographic",
        camera_distance_radii: float = 5.0,
        visibility_samples: Literal[1, 3] = 1,
        tau_intensity: float | None = None,
        beta_intensity: float = 64.0,
        tau_binary: float = 0.10,
        beta_binary: float = 8.0,
        frame_chunk: int = 120,
        direction_chunk: int = 8,
        mask_backface_occluders: bool = True,
    ):
        self.observations = geometry
        self.eps = float(eps)
        self.brdf_type = brdf_type
        if not 0.0 <= lambertian_fraction <= 1.0:
            raise ValueError("lambertian_fraction must lie in [0, 1]")
        self.lambertian_fraction = float(lambertian_fraction)
        if shading not in ("flat", "smooth"):
            raise ValueError("shading must be 'flat' or 'smooth'")
        self.shading = shading
        if projection not in ("orthographic", "perspective"):
            raise ValueError("projection must be 'orthographic' or 'perspective'")
        if projection == "perspective" and camera_distance_radii <= 1.0:
            raise ValueError("perspective camera distance must exceed one radius")
        self.projection = projection
        self.camera_distance_radii = float(camera_distance_radii)
        if visibility_samples not in (1, 3):
            raise ValueError("visibility_samples must be 1 or 3")
        self.visibility_samples = int(visibility_samples)
        self.tau_intensity = (
            None if tau_intensity is None else float(tau_intensity)
        )
        self.beta_intensity = float(beta_intensity)
        self.tau_binary = float(tau_binary)
        self.beta_binary = float(beta_binary)
        self.frame_chunk = int(frame_chunk)
        self.direction_chunk = int(direction_chunk)
        self.mask_backface_occluders = bool(mask_backface_occluders)

    @staticmethod
    def face_geometry(
        mesh: TriangleMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        areas, geometric, _ = _surface_geometry(mesh.vertices, mesh.faces)
        return areas, geometric

    def shading_normals(self, mesh: TriangleMesh) -> torch.Tensor:
        _, geometric, smooth = _surface_geometry(mesh.vertices, mesh.faces)
        return smooth if self.shading == "smooth" else geometric

    def prepare_visibility(
        self,
        mesh: TriangleMesh,
        gamma: float,
        *,
        checkpoint_visibility: bool = False,
        detach: bool = False,
    ) -> PreparedVisibility:
        _, normals = self.face_geometry(mesh)
        common = {
            "gamma": gamma,
            "eps": self.eps,
            "cull_frac": None,
            "dir_chunk": self.direction_chunk,
            "use_compile": False,
            "use_checkpoint": checkpoint_visibility,
            "mask_backface_occluders": self.mask_backface_occluders,
            "visibility_samples": self.visibility_samples,
        }
        if self.projection == "orthographic":
            table = visibility_table(
                mesh.vertices,
                mesh.faces,
                normals,
                self.observations.unique_dirs,
                projection="orthographic",
                **common,
            )
            observer_table = None
        else:
            table = visibility_table(
                mesh.vertices,
                mesh.faces,
                normals,
                self.observations.unique_sun_dirs,
                projection="orthographic",
                **common,
            )
            observer_table = visibility_table(
                mesh.vertices,
                mesh.faces,
                normals,
                self.observations.unique_view_dirs,
                projection="perspective",
                camera_distance_radii=self.camera_distance_radii,
                **common,
            )
        if detach:
            table = table.detach()
            if observer_table is not None:
                observer_table = observer_table.detach()
        return PreparedVisibility(
            table=table,
            gamma=float(gamma),
            observer_table=observer_table,
            projection=self.projection,
            camera_distance_radii=(
                self.camera_distance_radii
                if self.projection == "perspective"
                else None
            ),
            visibility_samples=self.visibility_samples,
        )

    def predict(
        self,
        mesh: TriangleMesh,
        gamma: float,
        *,
        prepared: PreparedVisibility | None = None,
        checkpoint_visibility: bool = False,
        include_binary: bool = False,
    ) -> PhotometricPrediction:
        if prepared is not None and not math.isclose(
            prepared.gamma, float(gamma), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("prepared visibility was built for another gamma")
        if prepared is not None and prepared.projection != self.projection:
            raise ValueError("prepared visibility was built for another projection")
        if (
            prepared is not None
            and prepared.visibility_samples != self.visibility_samples
        ):
            raise ValueError("prepared visibility used another sample count")
        if (
            prepared is not None
            and self.projection == "perspective"
            and not math.isclose(
                float(prepared.camera_distance_radii),
                self.camera_distance_radii,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("prepared visibility used another camera distance")

        areas, geometric_normals = self.face_geometry(mesh)
        shading_normals = self.shading_normals(mesh)
        if prepared is None:
            prepared = self.prepare_visibility(
                mesh,
                gamma,
                checkpoint_visibility=checkpoint_visibility,
            )

        obs = self.observations
        if prepared.observer_table is None:
            observer_table = prepared.table
            inv_sun = obs.inv_sun
            inv_view = obs.inv_view
        else:
            observer_table = prepared.observer_table
            inv_sun = obs.inv_sun_separate
            inv_view = obs.inv_view_separate
        face_centroids = mesh.vertices[mesh.faces].mean(dim=1)
        radius = torch.linalg.vector_norm(mesh.vertices, dim=-1).max()
        camera_distance = radius * self.camera_distance_radii
        intensity_chunks: list[torch.Tensor] = []
        binary_chunks: list[torch.Tensor] = []
        n_frames = obs.angles.shape[0]
        for start in range(0, n_frames, self.frame_chunk):
            stop = start + self.frame_chunk
            central_view_dirs = obs.view_dirs[start:stop]
            view_dirs = central_view_dirs
            observer_area_scale = None
            if self.projection == "perspective":
                camera_positions = central_view_dirs * camera_distance
                camera_delta = (
                    camera_positions[:, :, None, :]
                    - face_centroids[None, None, :, :]
                )
                view_dirs = F.normalize(
                    camera_delta, dim=-1
                )
                ray_distance = torch.linalg.vector_norm(
                    camera_delta, dim=-1
                )
                axial_depth = (
                    camera_distance
                    - torch.einsum(
                        "fi,nci->ncf",
                        face_centroids,
                        central_view_dirs,
                    )
                ).clamp_min(torch.finfo(mesh.vertices.dtype).eps)
                observer_area_scale = (
                    camera_distance.square()
                    * ray_distance
                    / axial_depth.pow(3)
                )
            args = (
                areas,
                geometric_normals,
                shading_normals,
                obs.sun_dirs[start:stop],
                view_dirs,
                prepared.table,
                observer_table,
                inv_sun[start:stop],
                inv_view[start:stop],
                observer_area_scale,
            )
            intensity_chunks.append(
                _integrate_chunk(
                    *args,
                    self.brdf_type,
                    self.lambertian_fraction,
                    self.tau_intensity,
                    self.beta_intensity,
                )
            )
            if include_binary:
                binary_chunks.append(
                    _integrate_chunk_binary(
                        *args,
                        self.tau_binary,
                        self.beta_binary,
                        self.brdf_type,
                        self.lambertian_fraction,
                    )
                )

        return PhotometricPrediction(
            intensity=torch.cat(intensity_chunks, dim=0),
            binary=(
                torch.cat(binary_chunks, dim=0)
                if include_binary
                else None
            ),
        )

def compute_lightcurves_multicam(
    verts,
    faces,
    sun_dir0,
    view_dirs0,
    angles,
    *,
    gamma=0.02,
    eps=1e-3,
    brdf_type="combined",
    lambertian_fraction=0.5,
    smooth_shading=True,
    projection="orthographic",
    camera_distance_radii=5.0,
    visibility_samples=1,
    intensity_threshold=None,
    intensity_beta=64.0,
    rotation_sign=-1.0,
    phase_offset=0.0,
    cull_frac=None,
    dir_chunk=4,
    frame_chunk=60,
    use_compile=True,
    use_checkpoint=False,
    dedupe_tol=1e-4,
    vis_dtype=None,
    verbose=False,
):
    """Lightcurves for all cameras at once. Returns ``(N, C)``.

    All faces are retained by default because normal-ranked face culling can
    discard valid occluders on a non-convex body.  ``cull_frac`` remains an
    explicit performance approximation for callers that accept that tradeoff.
    """
    if projection == "perspective":
        if cull_frac is not None:
            raise ValueError("perspective mode does not support face culling")
        geometry = ObservationGeometry.build(
            angles,
            sun_dir0,
            view_dirs0,
            rotation_sign=rotation_sign,
            phase_offset=phase_offset,
            dedupe_tol=dedupe_tol,
        )
        renderer = PhotometricForwardModel(
            geometry,
            eps=eps,
            brdf_type=brdf_type,
            lambertian_fraction=lambertian_fraction,
            shading="smooth" if smooth_shading else "flat",
            projection="perspective",
            camera_distance_radii=camera_distance_radii,
            visibility_samples=visibility_samples,
            tau_intensity=intensity_threshold,
            beta_intensity=intensity_beta,
            frame_chunk=frame_chunk,
            direction_chunk=dir_chunk,
        )
        return renderer.predict(
            TriangleMesh(verts, faces),
            gamma,
            checkpoint_visibility=use_checkpoint,
        ).intensity
    if projection != "orthographic":
        raise ValueError("projection must be 'orthographic' or 'perspective'")
    if visibility_samples != 1:
        raise ValueError(
            "sub-face visibility sampling requires the mesh-only renderer"
        )

    N, C = angles.shape[0], view_dirs0.shape[0]

    areas, geometric_normals, smooth_normals = _surface_geometry(verts, faces)
    shading_normals = smooth_normals if smooth_shading else geometric_normals

    psi = rotation_sign * angles + phase_offset
    sun_dirs = _rotate_dirs_z(sun_dir0, psi)  # (N, 3)
    view_dirs = _rotate_dirs_z(view_dirs0, psi)  # (N, C, 3)

    # Dedupe directions
    all_dirs = torch.cat([sun_dirs, view_dirs.reshape(-1, 3)], dim=0)
    key = torch.round(all_dirs / dedupe_tol)
    uniq_key, inv = torch.unique(key, dim=0, return_inverse=True)
    uniq_dirs = F.normalize(uniq_key * dedupe_tol, dim=-1)
    if verbose:
        print(
            f"visibility passes: {uniq_key.shape[0]} unique "
            f"of {all_dirs.shape[0]} (frame,dir) pairs"
        )

    table = visibility_table(
        verts,
        faces,
        geometric_normals,
        uniq_dirs,
        gamma=gamma,
        eps=eps,
        cull_frac=cull_frac,
        dir_chunk=dir_chunk,
        use_compile=use_compile,
        use_checkpoint=use_checkpoint,
        vis_dtype=vis_dtype,
    )  # (U, F)

    inv_sun, inv_view = inv[:N], inv[N:].reshape(N, C)
    out = []
    for i in range(0, N, frame_chunk):
        args = (
            areas,
            geometric_normals,
            shading_normals,
            sun_dirs[i : i + frame_chunk],
            view_dirs[i : i + frame_chunk],
            table,
            table,
            inv_sun[i : i + frame_chunk],
            inv_view[i : i + frame_chunk],
            None,
            brdf_type,
            lambertian_fraction,
            intensity_threshold,
            intensity_beta,
        )
        out.append(
            checkpoint(_integrate_chunk, *args, use_reentrant=False)
            if use_checkpoint
            else _integrate_chunk(*args)
        )
    return torch.cat(out, dim=0)  # (N, C)


CAMERA_ALPHA_DEG = {
    "0_hor1": 0.0,
    "0_hor2": 0.0,
    "0_top": 21.0,
    "0_bottom": -21.0,
    "45_hor1": 0.0,
    "45_hor2": 0.0,
    "45_top": 26.0,
    "45_bottom": -24.0,
    "90_hor1": 0.0,
    "90_hor2": 0.0,
    "90_top": 26.0,
    "90_bottom": -24.0,
    "135_hor1": 0.0,
    "135_hor2": 0.0,
    "135_top": 26.0,
    "135_bottom": -24.0,
    "225_hor1": 0.0,
    "225_hor2": 0.0,
    "225_top": 24.0,
    "225_bottom": -26.0,
    "270_hor1": 0.0,
    "270_hor2": 0.0,
    "270_top": 24.0,
    "270_bottom": -26.0,
    "315_hor1": 0.0,
    "315_hor2": 0.0,
    "315_top": 24.0,
    "315_bottom": -26.0,
}


def build_view_dirs(columns, device):
    """Column names ('45_top', ...) -> (C, 3) world-frame view directions."""
    dirs = []
    for col in columns:
        az, kind = col.split("_")
        if col not in CAMERA_ALPHA_DEG:
            raise ValueError(f"unknown challenge camera column: {col}")
        phi = np.radians(float(az) - 180.0)
        alpha = np.radians(CAMERA_ALPHA_DEG[col])
        dirs.append(
            [
                np.cos(phi) * np.cos(alpha),
                np.sin(phi) * np.cos(alpha),
                np.sin(alpha),
            ]
        )
    return F.normalize(
        torch.tensor(np.array(dirs), dtype=torch.float32, device=device), dim=-1
    )


@torch.no_grad()
def max_frontface_frac(nhat, dirs):
    """Run once per mesh to choose cull_frac (add ~0.05 margin)."""
    mu = torch.einsum("fi,di->df", nhat, dirs)
    return (mu > 0).float().mean(dim=1).max().item()


# --------------------------------------------------------------------------- #
# Main driver (demo)
# --------------------------------------------------------------------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = 3
    N = 842
    origin = "simulated"
    curve_type = "intensity"
    target_column = "45_hor1"

    GAMMA = 0.02
    EPS = 1e-3

    verts, faces = load_asteroid_mesh(target_model, device)
    print(f"Loaded mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

    dataset = load_data(target_model, origin, curve_type)
    real_curve = dataset.data[target_column].values[:N]

    vmin, vmax = verts.min(dim=0).values, verts.max(dim=0).values
    centroid = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()
    verts = (verts - centroid) / extent

    sun_dir_0 = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    phi_deg = float(target_column.split("_")[0])
    alpha_map = {"hor1": 0, "hor2": 0, "top": 26, "bottom": -26}
    alpha_deg = alpha_map.get(target_column.split("_")[1], 0)
    phi = np.radians(phi_deg - 180)
    alpha = np.radians(alpha_deg)
    camera_vector = np.array(
        [
            np.cos(phi) * np.cos(alpha),
            np.sin(phi) * np.cos(alpha),
            np.sin(alpha),
        ]
    )
    view_dir_0 = torch.tensor(
        camera_vector / np.linalg.norm(camera_vector), device=device, dtype=torch.float32
    )

    angles = torch.arange(N, device=device) * (2 * np.pi / N)

    print("Computing integrated lightcurve …")
    t0 = time.time()
    with torch.no_grad():
        simulated = compute_lightcurve(
            verts,
            faces,
            sun_dir_0,
            view_dir_0,
            angles,
            gamma=GAMMA,
            eps=EPS,
            brdf_type="combined",
        )
    simulated = simulated.cpu().numpy()
    print(f"Forward pass ({N} pts): {time.time()-t0:.2f}s")

    sim_norm = simulated / simulated.mean()
    real_norm = real_curve / real_curve.mean()

    plt.figure(figsize=(10, 5))
    x_axis = np.linspace(0, 360, N)

    plt.plot(
        x_axis,
        sim_norm,
        color="blue",
        linewidth=2.0,
        label="Integrated Soft-Rasterizer (fixed)",
    )
    plt.plot(
        x_axis,
        real_norm,
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label=f"{origin} Data ({target_column})",
    )

    plt.title(f"Lightcurve Comparison: Model {target_model}")
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Normalized Intensity (Mean = 1.0)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("lightcurve_comparison_integrated.png", dpi=150)
    print("Saved plot to 'lightcurve_comparison_integrated.png'")

    print("\nGradient sanity check (4 angles) …")
    verts_g = verts.clone().requires_grad_(True)
    sim_g = compute_lightcurve(
        verts_g,
        faces,
        sun_dir_0,
        view_dir_0,
        angles[:4],
        gamma=GAMMA,
        eps=EPS,
        brdf_type="combined",
    )
    loss = sim_g.sum()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  Backward pass: {time.time()-t0:.3f}s")
    print(f"  ||∇_verts||  : {verts_g.grad.norm().item():.4f}")


def main_multicam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts, faces = load_asteroid_mesh(3, device)
    vmin, vmax = verts.min(0).values, verts.max(0).values
    verts = (verts - (vmin + vmax) / 2) / (vmax - vmin).max()

    dataset = load_data(3, "simulated", "intensity")
    cols = list(dataset.data.columns)
    view_dirs0 = build_view_dirs(cols, device)
    sun_dir0 = torch.tensor([-1.0, 0.0, 0.0], device=device)
    N = 360
    angles = torch.arange(N, device=device) * (2 * np.pi / N)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # warm-up = compile
        _ = compute_lightcurves_multicam(
            verts, faces, sun_dir0, view_dirs0, angles, verbose=True
        )
        torch.cuda.synchronize()
        t0 = time.time()
        lcs = compute_lightcurves_multicam(
            verts, faces, sun_dir0, view_dirs0, angles
        )
        torch.cuda.synchronize()
    print(
        f"forward: {time.time()-t0:.2f}s, "
        f"peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    )
    lcs = lcs / lcs.mean(dim=0, keepdim=True)


if __name__ == "__main__":
    main_multicam()
