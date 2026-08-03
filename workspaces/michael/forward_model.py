#!/usr/bin/env python3
"""
Integrated soft-rasterizer for non-convex asteroid lightcurves — FIXED.

Drop-in replacement for forward_model.py. Four changes relative to the
original, marked with FIX 1..4 in the code:

FIX 1 (depth sign, root cause)
    The projection z = v · d is the coordinate ALONG the direction toward
    the observer / sun, so LARGER z means CLOSER. The original softmax
    weighted exp(-z/gamma), i.e. it selected the *farthest* surface: the
    model was rendering the hidden side of the asteroid, and the signal
    that remained was soft-leakage noise from thin regions near the limb.

FIX 2 (self-term must never leave the denominator)
    In the original, `det_mask` could remove a face's own depth term from
    its softmax denominator (whenever the face projects edge-on). The
    visibility then became exp_q / (~0 + 1e-10) ~ 1e10 instead of <= 1,
    producing isolated giant spikes. The query face's exact depth is now
    always added to the denominator explicitly, so vis is guaranteed to
    lie in (0, 1].

FIX 3 (robust handling of (near-)edge-on occluders)
    `det_safe = det + 1e-10` flips the sign of the barycentric coordinates
    whenever det is small and negative (about half of all faces have
    negative signed det). Near-degenerate faces are now excluded as
    occluders outright, and interpolated depths are clamped to the
    occluding face's own vertex depth range as a safety net.

FIX 4 (foreshortening factor mu = cos_v in the scattering law)
    The challenge intensity curve is a *pixel sum*. In an orthographic
    image a face covers ~ A * mu pixels, so the old PyTorch3D pipeline
    implicitly weighted every face by its projected area. The integrated
    model must therefore carry the factor mu explicitly:
        Lambert          : A * mu0            ->  A * mu * mu0
        Lommel-Seeliger  : A * mu0/(mu+mu0)   ->  A * mu * mu0/(mu+mu0)
    (mu0 = cos incidence, mu = cos emission; matches the old
    LambertianOrthographicShader / LommelSeeligerOrthographicShader whose
    pixel values were mu0 and mu0/(mu+mu0) respectively.)

FIX 5 (rotation direction / phase)
    The old pipeline passed column-vector matrices from
    `axis_angle_to_matrix` into `Transform3d.rotate`, which applies
    matrices to row vectors — so it actually rotated the mesh by -theta.
    `compute_lightcurve` now exposes `rotation_sign` (default -1,
    matching the old pipeline's real behaviour) and `phase_offset`
    (the arbitrary common start point of the aligned challenge curves).
    Fit both with evaluate_forward_model.py.

Also: optional backface culling (`cull_backfaces=True`) cuts both passes
to roughly the front half of the faces — ~4x less memory/compute for the
O(F^2) visibility matrix. Exact for closed (watertight) meshes, which the
challenge STLs are: along any ray the nearest surface is always
front-facing, so back faces can never be the winning occluder.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trimesh

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

# --------------------------------------------------------------------------- #
# 1.  Data I/O  (unchanged)
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    model_num: int
    origin: Literal["real", "simulated"]
    curve_type: Literal["binary", "intensity"]
    data: pd.DataFrame


PROJECT_DIR = Path(__file__).resolve().parents[2]
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
    """Load vertices & faces directly from STL via trimesh."""
    mesh = trimesh.load(get_model_stl_path(model_num), force="mesh")
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return verts, faces


def load_data(
    model_num: int,
    origin: Literal["real", "simulated"],
    curve_type: Literal["binary", "intensity"],
) -> Dataset:
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
            1: "0_hor1", 2: "0_hor2", 3: "0_top", 4: "0_bottom",
            5: "45_hor1", 6: "45_hor2", 7: "45_top", 8: "45_bottom",
            9: "90_hor1", 10: "90_hor2", 11: "90_top", 12: "90_bottom",
            13: "135_hor1", 14: "135_hor2", 15: "135_top", 16: "135_bottom",
            17: "225_hor1", 18: "225_hor2", 19: "225_top", 20: "225_bottom",
            21: "270_hor1", 22: "270_hor2", 23: "270_top", 24: "270_bottom",
            25: "315_hor1", 26: "315_hor2", 27: "315_top", 28: "315_bottom",
        }
    )
    return Dataset(model_num=model_num, origin=origin,
                   curve_type=curve_type, data=data)


# --------------------------------------------------------------------------- #
# 2.  Core rasterizer  (no pixels, only integrated brightness)
# --------------------------------------------------------------------------- #

def _ortho_basis(direction: torch.Tensor):
    """Build right-handed orthonormal basis [u, w, d] for projection.

    Written with pure tensor ops (no .item() / Python branching) so that
    torch.compile can capture it without graph breaks.
    """
    d = F.normalize(direction, dim=-1)
    up_z = d.new_tensor([0.0, 0.0, 1.0])
    up_y = d.new_tensor([0.0, 1.0, 0.0])
    up = torch.where(d[2].abs() < 0.9, up_z, up_y)
    u = F.normalize(torch.linalg.cross(d, up, dim=-1), dim=-1)
    w = torch.linalg.cross(d, u, dim=-1)
    return torch.stack([u, w, d], dim=-1)          # (3, 3)


def _soft_visibility(
    centroids: torch.Tensor,
    face_xy: torch.Tensor,
    face_z: torch.Tensor,
    gamma: float = 0.02,
    eps: float = 1e-3,
    z_far: float = 1e6,
) -> torch.Tensor:
    """
    Differentiable orthographic visibility via Z-softmax.

    Depth convention: z is the coordinate along the direction toward the
    observer (or sun), so LARGER z = CLOSER. For each face i we query its
    centroid against all faces j; the soft weight is

        V_i = exp(z_i/g) / [ exp(z_i/g) + sum_{j != i} c_ij * exp(z_ij/g) ]

    where z_ij is the interpolated depth of face j at query point i and
    c_ij in [0, 1] is the soft coverage ("query inside triangle j"). For
    a point fully covered by exactly one other surface this reduces to
    V_i = sigmoid((z_i - z_cover)/gamma). V is guaranteed to be in (0, 1].
    (z_far is kept in the signature for API compatibility; it is no
    longer needed because coverage weights the exponential directly.)

    Parameters
    ----------
    centroids : (F, 2)  query points (centroids of each projected face).
    face_xy   : (F, 3, 2) projected triangle vertices.
    face_z    : (F, 3)  depths of the three vertices (toward observer).

    Returns
    -------
    vis : (F,) soft visibility weight in (0, 1] for every face.
    """
    n_faces = centroids.shape[0]

    # Unpack triangle vertices
    x0, x1, x2 = face_xy[:, 0, 0], face_xy[:, 1, 0], face_xy[:, 2, 0]
    y0, y1, y2 = face_xy[:, 0, 1], face_xy[:, 1, 1], face_xy[:, 2, 1]
    z0, z1, z2 = face_z[:, 0], face_z[:, 1], face_z[:, 2]

    # Edge vectors & 2-D signed determinant (= 2 * projected area)
    e1x, e1y = x1 - x0, y1 - y0
    e2x, e2y = x2 - x0, y2 - y0
    det = e1x * e2y - e1y * e2x                     # (F,)

    # FIX 3: exclude (near-)edge-on faces as occluders instead of shifting
    # the *signed* determinant by +1e-10 (which flips the barycentrics'
    # sign whenever det is small and negative).
    degenerate = det.abs() < 1e-9
    safe_det = torch.where(degenerate, torch.ones_like(det), det)
    inv_det = torch.where(degenerate, torch.zeros_like(det), 1.0 / safe_det)

    # Query points: (F, 1)
    qx = centroids[:, 0:1]
    qy = centroids[:, 1:2]

    # All-pairs differences: (F, F)
    dpx = qx - x0.unsqueeze(0)
    dpy = qy - y0.unsqueeze(0)

    # Barycentric coordinates of query i w.r.t. face j
    lam1 = (dpx * e2y.unsqueeze(0) - dpy * e2x.unsqueeze(0)) * inv_det.unsqueeze(0)
    lam2 = (dpy * e1x.unsqueeze(0) - dpx * e1y.unsqueeze(0)) * inv_det.unsqueeze(0)
    lam0 = 1.0 - lam1 - lam2

    # Soft "inside triangle" indicator
    inside = (
        torch.sigmoid(lam0 / eps)
        * torch.sigmoid(lam1 / eps)
        * torch.sigmoid(lam2 / eps)
    )
    inside = inside * (~degenerate).to(inside.dtype).unsqueeze(0)

    # Interpolated depth at query point i on face j
    z_interp = (
        lam0 * z0.unsqueeze(0)
        + lam1 * z1.unsqueeze(0)
        + lam2 * z2.unsqueeze(0)
    )
    # FIX 3 (safety net): a genuine covering point can never lie outside
    # face j's own vertex depth range; clamping removes any residual
    # barycentric garbage from nearly-degenerate projections.
    z_interp = torch.clamp(
        z_interp,
        min=face_z.min(dim=1).values.unsqueeze(0),
        max=face_z.max(dim=1).values.unsqueeze(0),
    )

    # FIX 2: exclude the query face from its own occluder set; its exact
    # depth term is added back to the denominator explicitly below, so it
    # can never vanish and vis <= 1 always holds.
    eye = torch.eye(n_faces, dtype=torch.bool, device=centroids.device)
    inside = inside.masked_fill(eye, 0.0)

    # Depth of each face's own centroid
    z_query = (z0 + z1 + z2) / 3.0

    # FIX 1: larger z = closer to the observer, so the softmax favours
    # LARGE z (note the + sign in the logits — near wins).
    #
    # The coverage enters as a multiplicative WEIGHT on the exponential,
    #     w_j = inside_j * exp(z_j / gamma),
    # not as a blend of the depth toward a far sentinel. Blending the
    # depth (z*inside - z_far*(1-inside)) would punch soft "cracks" along
    # every edge of the occluding surface: a query point near a shared
    # mesh edge sees inside ~ 0.5 from each neighbour and the occluder
    # would drop out of the softmax entirely.
    logits = z_interp / gamma
    logits_query = z_query / gamma
    max_logit = torch.maximum(logits.max(dim=1).values, logits_query)
    exp_others = (inside * torch.exp(logits - max_logit.unsqueeze(1))).sum(dim=1)
    exp_query = torch.exp(logits_query - max_logit)
    vis = exp_query / (exp_query + exp_others)

    return vis


def _direction_visibility(
    verts: torch.Tensor,
    faces: torch.Tensor,
    direction: torch.Tensor,
    gamma: float,
    eps: float,
    keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Project the mesh along `direction` and evaluate soft visibility.

    If `keep` (bool mask over faces) is given, both the query set and the
    occluder set are restricted to those faces (backface culling). Exact
    for closed meshes; culled faces get vis = 0, which is consistent
    because their scattering weight is zero anyway.
    """
    basis = _ortho_basis(direction)
    proj = verts @ basis
    face_p = proj[faces]                            # (F, 3, 3)

    if keep is None:
        centroids = face_p.mean(dim=1)[:, :2]
        return _soft_visibility(centroids, face_p[:, :, :2],
                                face_p[:, :, 2], gamma, eps)

    idx = keep.nonzero(as_tuple=True)[0]
    face_sub = face_p[idx]
    centroids = face_sub.mean(dim=1)[:, :2]
    vis_sub = _soft_visibility(centroids, face_sub[:, :, :2],
                               face_sub[:, :, 2], gamma, eps)
    vis = torch.zeros(faces.shape[0], dtype=verts.dtype, device=verts.device)
    return vis.index_copy(0, idx, vis_sub)


def compute_brightness(
    verts: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    view_dir: torch.Tensor,
    gamma: float = 0.02,
    eps: float = 1e-3,
    brdf_type: Literal["lambertian", "lommel_seeliger", "combined"] = "combined",
    cull_backfaces: bool = True,
) -> torch.Tensor:
    """
    Disk-integrated brightness for a single (sun, view) pair.

    Matches the pixel-sum of the old PyTorch3D pipeline in the hard limit:
    the pixel footprint of a face is its projected area A*mu, and the old
    shaders wrote mu0 (Lambert) resp. mu0/(mu+mu0) (Lommel-Seeliger) into
    each pixel, so the integrated contribution per face is
    A * mu * shader_value  (FIX 4).
    """
    # ---- per-face geometry -------------------------------------------------
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = torch.cross(e1, e2, dim=-1)
    areas = 0.5 * normals.norm(dim=-1)
    nhat = F.normalize(normals, dim=-1)

    # ---- scattering law ------------------------------------------------------
    mu0 = torch.clamp((nhat * sun_dir).sum(dim=-1), min=0.0)    # cos incidence
    mu = torch.clamp((nhat * view_dir).sum(dim=-1), min=0.0)    # cos emission

    # FIX 4: every branch carries the foreshortening factor mu.
    if brdf_type == "lambertian":
        scatter = mu0 * mu
    elif brdf_type == "lommel_seeliger":
        scatter = mu0 * mu / (mu0 + mu + 1e-8)
    else:  # combined — mirrors old CombinedShader = 0.5 * (Lambert + LS)
        scatter = 0.5 * (mu0 * mu + mu0 * mu / (mu0 + mu + 1e-8))

    # ---- view visibility (self-occlusion) ----------------------------------
    keep_v = (mu > 0) if cull_backfaces else None
    vis = _direction_visibility(verts, faces, view_dir, gamma, eps, keep_v)

    # ---- solar illumination (self-shadowing) -------------------------------
    keep_s = (mu0 > 0) if cull_backfaces else None
    ill = _direction_visibility(verts, faces, sun_dir, gamma, eps, keep_s)

    # ---- integrated brightness ---------------------------------------------
    return (areas * scatter * vis * ill).sum()


# Cache for the torch.compile'd brightness function (built lazily).
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
    cull_backfaces: bool = True,
    rotation_sign: float = -1.0,
    phase_offset: float = 0.0,
    use_compile: bool = False,
) -> torch.Tensor:
    """
    Compute lightcurve for N rotation angles.

    Frame k corresponds to the MESH rotated by

        psi_k = rotation_sign * angles[k] + phase_offset      (about +z),

    implemented by rotating the sun/view directions by R_z(-psi_k).

    FIX 5 (rotation_sign default = -1): the old PyTorch3D pipeline fed
    column-vector matrices from `axis_angle_to_matrix` into
    `Transform3d.rotate`, which applies matrices to ROW vectors
    (points @ R = R^T points). It therefore actually rotated the mesh by
    -theta, not +theta. rotation_sign=-1 reproduces that behaviour;
    rotation_sign=+1 reproduces this file's previous (opposite) spin.
    `phase_offset` absorbs the arbitrary common start point the challenge
    organisers chose when aligning the measured curves. Fit both with
    evaluate_forward_model.py, then hard-code them here.

    use_compile=True routes each brightness evaluation through
    torch.compile (forces cull_backfaces=False internally: culling
    produces data-dependent shapes, which break compilation). First call
    pays a one-off compilation cost; subsequent calls are fused kernels.
    """
    N = angles.shape[0]
    psi = rotation_sign * angles + phase_offset
    c, s = torch.cos(psi), torch.sin(psi)

    # R_z(-psi) applied to the direction vectors
    sun_dirs = torch.stack([
        c * sun_dir[0] + s * sun_dir[1],
        -s * sun_dir[0] + c * sun_dir[1],
        torch.full_like(psi, sun_dir[2]),
    ], dim=1)

    view_dirs = torch.stack([
        c * view_dir[0] + s * view_dir[1],
        -s * view_dir[0] + c * view_dir[1],
        torch.full_like(psi, view_dir[2]),
    ], dim=1)

    if use_compile:
        fn = _get_compiled_brightness()
        cull = False
    else:
        fn = compute_brightness
        cull = cull_backfaces

    brightness = torch.zeros(N, device=verts.device, dtype=verts.dtype)
    for k in range(N):
        brightness[k] = fn(
            verts, faces,
            sun_dirs[k], view_dirs[k],
            gamma=gamma, eps=eps, brdf_type=brdf_type,
            cull_backfaces=cull,
        )
    return brightness

from torch.utils.checkpoint import checkpoint


def _ortho_bases(dirs: torch.Tensor) -> torch.Tensor:
    """Batched version of _ortho_basis: (D, 3) unit dirs -> (D, 3, 3)."""
    d = F.normalize(dirs, dim=-1)
    up_z = d.new_tensor([0.0, 0.0, 1.0]).expand_as(d)
    up_y = d.new_tensor([0.0, 1.0, 0.0]).expand_as(d)
    up = torch.where((d[:, 2].abs() < 0.9).unsqueeze(-1), up_z, up_y)
    u = F.normalize(torch.linalg.cross(d, up, dim=-1), dim=-1)
    w = torch.linalg.cross(d, u, dim=-1)
    return torch.stack([u, w, d], dim=-1)


def _rotate_dirs_z(d0: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Apply the frame rotation R_z(-psi_k) to fixed world dirs.

    d0: (..., 3), psi: (N,)  ->  (N, ..., 3). Identical formula to the
    per-frame stack in compute_lightcurve (FIX 5 conventions preserved).
    """
    shape = (psi.shape[0],) + (1,) * (d0.dim() - 1)
    c, s = torch.cos(psi).view(shape), torch.sin(psi).view(shape)
    x, y, z = d0[..., 0], d0[..., 1], d0[..., 2]
    xr = c * x + s * y
    yr = -s * x + c * y
    return torch.stack([xr, yr, z.expand_as(xr)], dim=-1)


def _soft_visibility_batch(face_xy, face_z, gamma, eps, valid=None):
    """Batched soft visibility. Dims: b = direction, i = query, j = occluder.

    face_xy: (B, K, 3, 2), face_z: (B, K, 3), valid: (B, K) bool or None.
    Returns (B, K) in (0, 1]. All FIXes 1-3 preserved.

    Note: the softmax is rewritten as V_i = 1 / (1 + sum_j c_ij e^{(z_ij-z_i)/g}),
    algebraically identical to exp_q/(exp_q+exp_others) but it drops one
    (B,K,K) reduction (the max trick) and stays finite for tiny gamma via
    the clamp(max=60) on the exponent (exp(60) ~ 1e26 << fp32 max).
    """
    B, K = face_z.shape[:2]
    x0, x1, x2 = face_xy[..., 0, 0], face_xy[..., 1, 0], face_xy[..., 2, 0]
    y0, y1, y2 = face_xy[..., 0, 1], face_xy[..., 1, 1], face_xy[..., 2, 1]
    z0, z1, z2 = face_z[..., 0], face_z[..., 1], face_z[..., 2]

    e1x, e1y = x1 - x0, y1 - y0
    e2x, e2y = x2 - x0, y2 - y0
    det = e1x * e2y - e1y * e2x                                    # (B, K)

    degenerate = det.abs() < 1e-9                                  # FIX 3
    safe_det = torch.where(degenerate, torch.ones_like(det), det)
    inv_det = torch.where(degenerate, torch.zeros_like(det), 1.0 / safe_det)

    cx, cy = (x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0            # queries

    dpx = cx.unsqueeze(2) - x0.unsqueeze(1)                        # (B, Kq, Ko)
    dpy = cy.unsqueeze(2) - y0.unsqueeze(1)
    lam1 = (dpx * e2y.unsqueeze(1) - dpy * e2x.unsqueeze(1)) * inv_det.unsqueeze(1)
    lam2 = (dpy * e1x.unsqueeze(1) - dpx * e1y.unsqueeze(1)) * inv_det.unsqueeze(1)
    lam0 = 1.0 - lam1 - lam2

    inside = (torch.sigmoid(lam0 / eps) * torch.sigmoid(lam1 / eps)
              * torch.sigmoid(lam2 / eps))
    occ_ok = ~degenerate
    if valid is not None:                     # exclude back-facing occluders
        occ_ok = occ_ok & valid               # (exact for watertight meshes)
    inside = inside * occ_ok.to(inside.dtype).unsqueeze(1)

    z_interp = (lam0 * z0.unsqueeze(1) + lam1 * z1.unsqueeze(1)
                + lam2 * z2.unsqueeze(1))
    z_interp = torch.clamp(z_interp,                               # FIX 3 net
                           min=face_z.min(dim=-1).values.unsqueeze(1),
                           max=face_z.max(dim=-1).values.unsqueeze(1))

    eye = torch.eye(K, dtype=torch.bool, device=face_z.device)     # FIX 2
    inside = inside.masked_fill(eye, 0.0)

    z_query = (z0 + z1 + z2) / 3.0
    delta = (z_interp - z_query.unsqueeze(2)) / gamma              # FIX 1 sign
    w = (inside * torch.exp(delta.clamp(max=60.0))).sum(dim=2)
    return 1.0 / (1.0 + w)


def _visibility_chunk(verts, faces, nhat, dirs, gamma, eps, k_keep,
                      mask_backface_occluders=True, vis_dtype=None):
    """Soft visibility for a fixed-size chunk of directions -> (B, F).

    Compile-safe replacement for the nonzero()-based culling: keep the
    k_keep faces with LARGEST mu per direction (topk = static shape).
    Dropped faces get vis=0; they have mu <= tail-threshold, so their
    scattering weight is (near-)zero anyway — exact in the same sense as
    cull_backfaces for closed meshes when k_keep >= #front-facing.
    """
    B, n_faces = dirs.shape[0], faces.shape[0]
    bases = _ortho_bases(dirs)                                     # (B,3,3)
    proj = torch.einsum('vi,bij->bvj', verts, bases)               # (B,V,3)
    face_p = proj[:, faces]                                        # (B,F,3,3)
    mu = torch.einsum('fi,bi->bf', nhat, dirs)                     # (B,F)

    if k_keep is not None and k_keep < n_faces:
        idx = mu.topk(k_keep, dim=1, sorted=False).indices         # (B,K)
        face_sub = torch.gather(
            face_p, 1, idx[:, :, None, None].expand(B, k_keep, 3, 3))
        mu_sub = torch.gather(mu, 1, idx)
        if vis_dtype is not None:                                  # optional bf16
            face_sub = face_sub.to(vis_dtype)
        valid = (mu_sub > 0.0) if mask_backface_occluders else None
        vis_sub = _soft_visibility_batch(face_sub[..., :2], face_sub[..., 2],
                                         gamma, eps, valid)
        vis_sub = vis_sub.to(verts.dtype) * (mu_sub > 0.0)
        return proj.new_zeros(B, n_faces).scatter(1, idx, vis_sub)

    fp = face_p.to(vis_dtype) if vis_dtype is not None else face_p
    valid = (mu > 0.0) if mask_backface_occluders else None
    vis = _soft_visibility_batch(fp[..., :2], fp[..., 2], gamma, eps, valid)
    vis = vis.to(verts.dtype)
    return vis * (mu > 0.0) if mask_backface_occluders else vis


_COMPILED_VIS_CHUNK = None
def _get_compiled_vis_chunk():
    global _COMPILED_VIS_CHUNK
    if _COMPILED_VIS_CHUNK is None:
        _COMPILED_VIS_CHUNK = torch.compile(_visibility_chunk, dynamic=False)
    return _COMPILED_VIS_CHUNK


def visibility_table(verts, faces, nhat, dirs, *, gamma, eps,
                     cull_frac=0.65, dir_chunk=4,
                     use_compile=True, use_checkpoint=False,
                     mask_backface_occluders=True, vis_dtype=None):
    """(D, 3) directions -> (D, F) soft visibility, chunked over D.

    Chunks are padded to `dir_chunk` so torch.compile sees ONE static shape
    (single compilation, no recompiles). use_checkpoint=True recomputes each
    chunk in backward: autograd then never holds more than one chunk's
    (B,K,K) intermediates — essential for gradient-based reconstruction.
    """
    D, n_faces = dirs.shape[0], faces.shape[0]
    k_keep = None if cull_frac is None else min(n_faces, int(cull_frac * n_faces))
    fn = _get_compiled_vis_chunk() if use_compile else _visibility_chunk
    outs = []
    for i in range(0, D, dir_chunk):
        d = dirs[i:i + dir_chunk]
        n_real = d.shape[0]
        if n_real < dir_chunk:                                     # pad last
            d = torch.cat([d, d[-1:].expand(dir_chunk - n_real, -1)], dim=0)
        args = (verts, faces, nhat, d, gamma, eps, k_keep,
                mask_backface_occluders, vis_dtype)
        v = (checkpoint(fn, *args, use_reentrant=False)
             if use_checkpoint else fn(*args))
        outs.append(v[:n_real])
    return torch.cat(outs, dim=0)


def _integrate_chunk(areas, nhat, sun_dirs, view_dirs, vis_table,
                     inv_sun, inv_view, brdf_type):
    """Photometric contraction for a chunk of frames -> (n, C)."""
    ill = vis_table[inv_sun]                                        # (n, F)
    vis = vis_table[inv_view]                                       # (n, C, F)
    mu0 = torch.einsum('fi,ni->nf', nhat, sun_dirs).clamp_min(0.0)
    mu = torch.einsum('fi,nci->ncf', nhat, view_dirs).clamp_min(0.0)
    l = mu0[:, None, :] * mu                                        # FIX 4
    if brdf_type == "lambertian":
        scatter = l
    elif brdf_type == "lommel_seeliger":
        scatter = l / (mu0[:, None, :] + mu + 1e-8)
    else:                                                           # combined
        scatter = 0.5 * l * (1.0 + 1.0 / (mu0[:, None, :] + mu + 1e-8))
    return torch.einsum('ncf,nf->nc', scatter * vis, areas[None, :] * ill)


def compute_lightcurves_multicam(
    verts, faces, sun_dir0, view_dirs0, angles, *,
    gamma=0.02, eps=1e-3, brdf_type="combined",
    rotation_sign=-1.0, phase_offset=0.0,
    cull_frac=0.65, dir_chunk=4, frame_chunk=60,
    use_compile=True, use_checkpoint=False,
    dedupe_tol=1e-4, vis_dtype=None, verbose=False,
):
    """Lightcurves for ALL cameras at once. Returns (N, C).

    sun_dir0: (3,), view_dirs0: (C, 3) world-frame unit dirs, angles: (N,).
    Directions are deduped after quantisation to ~0.006 deg: on the 360-frame
    grid all curves collapse onto 3 elevation rings (~1080 unique O(F^2)
    passes for 28 cameras instead of 10440).
    """
    N, C = angles.shape[0], view_dirs0.shape[0]

    # Per-face geometry: shared by every frame and camera.
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * normals.norm(dim=-1)
    nhat = F.normalize(normals, dim=-1)

    # Body-frame directions for every (frame, camera).  (FIX 5 conventions.)
    psi = rotation_sign * angles + phase_offset
    sun_dirs = _rotate_dirs_z(sun_dir0, psi)                        # (N, 3)
    view_dirs = _rotate_dirs_z(view_dirs0, psi)                     # (N, C, 3)

    # Dedupe: sun pass shared by all cameras + ring coincidences.
    all_dirs = torch.cat([sun_dirs, view_dirs.reshape(-1, 3)], dim=0)
    key = torch.round(all_dirs / dedupe_tol)
    uniq_key, inv = torch.unique(key, dim=0, return_inverse=True)
    uniq_dirs = F.normalize(uniq_key * dedupe_tol, dim=-1)
    if verbose:
        print(f"visibility passes: {uniq_key.shape[0]} unique "
              f"of {all_dirs.shape[0]} (frame,dir) pairs")

    table = visibility_table(
        verts, faces, nhat, uniq_dirs, gamma=gamma, eps=eps,
        cull_frac=cull_frac, dir_chunk=dir_chunk, use_compile=use_compile,
        use_checkpoint=use_checkpoint, vis_dtype=vis_dtype)         # (U, F)

    inv_sun, inv_view = inv[:N], inv[N:].reshape(N, C)
    out = []
    for i in range(0, N, frame_chunk):                              # (N,C,F) never
        args = (areas, nhat, sun_dirs[i:i + frame_chunk],           #  materialised
                view_dirs[i:i + frame_chunk], table,
                inv_sun[i:i + frame_chunk], inv_view[i:i + frame_chunk],
                brdf_type)
        out.append(checkpoint(_integrate_chunk, *args, use_reentrant=False)
                   if use_checkpoint else _integrate_chunk(*args))
    return torch.cat(out, dim=0)                                    # (N, C)


CAMERA_ALPHA_DEG = {"hor1": 0.0, "hor2": 0.0, "top": 26.0, "bottom": -26.0}

def build_view_dirs(columns, device):
    """Column names ('45_top', ...) -> (C, 3) world-frame view directions."""
    dirs = []
    for col in columns:
        az, kind = col.split("_")
        phi, alpha = np.radians(float(az) - 180.0), np.radians(CAMERA_ALPHA_DEG[kind])
        dirs.append([np.cos(phi) * np.cos(alpha),
                     np.sin(phi) * np.cos(alpha), np.sin(alpha)])
    return F.normalize(torch.tensor(np.array(dirs), dtype=torch.float32,
                                    device=device), dim=-1)


@torch.no_grad()
def max_frontface_frac(nhat, dirs):
    """Run once per mesh to choose cull_frac (add ~0.05 margin)."""
    mu = torch.einsum('fi,di->df', nhat, dirs)
    return (mu > 0).float().mean(dim=1).max().item()

# --------------------------------------------------------------------------- #
# 3.  Main driver
# --------------------------------------------------------------------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = 3
    N = 842
    origin = "real"
    curve_type = "intensity"
    target_column = "45_hor1"

    # Soft-rasterizer hyper-parameters
    GAMMA = 0.02          # shadow softness (fraction of unit body size)
    EPS   = 1e-3          # barycentric edge softness

    # ------------------------------------------------------------------
    # Load geometry & data
    # ------------------------------------------------------------------
    verts, faces = load_asteroid_mesh(target_model, device)
    print(f"Loaded mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

    dataset = load_data(target_model, origin, curve_type)
    real_curve = dataset.data[target_column].values[:N]

    # Centre and scale to unit extent (numerical stability)
    vmin, vmax = verts.min(dim=0).values, verts.max(dim=0).values
    centroid = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()
    verts = (verts - centroid) / extent

    # ------------------------------------------------------------------
    # Build fixed world-frame directions (same conventions as original)
    # ------------------------------------------------------------------
    # Sun fixed at -x  -> direction from asteroid toward sun is [-1,0,0]
    sun_dir_0 = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # Camera direction derived from target_column
    phi_deg = float(target_column.split("_")[0])
    alpha_map = {"hor1": 0, "hor2": 0, "top": 26, "bottom": -26}
    if phi_deg == 0 and "top" in target_column:
        alpha_deg = 26
    elif phi_deg == 0 and "bottom" in target_column:
        alpha_deg = -26
    else:
        alpha_deg = alpha_map.get(target_column.split("_")[1], 0)

    phi   = np.radians(phi_deg - 180)
    alpha = np.radians(alpha_deg)
    camera_vector = np.array([
        np.cos(phi) * np.cos(alpha),
        np.sin(phi) * np.cos(alpha),
        np.sin(alpha),
    ])
    view_dir_0 = torch.tensor(camera_vector / np.linalg.norm(camera_vector),
                              device=device, dtype=torch.float32)

    # Rotation angles: one full rotation, endpoint EXCLUDED (linspace with
    # both endpoints would make frame N-1 duplicate frame 0).
    angles = torch.arange(N, device=device) * (2 * np.pi / N)

    # ------------------------------------------------------------------
    # Forward evaluation
    # ------------------------------------------------------------------
    print("Computing integrated lightcurve …")
    t0 = time.time()
    with torch.no_grad():
        simulated = compute_lightcurve(
            verts, faces,
            sun_dir_0, view_dir_0, angles,
            gamma=GAMMA, eps=EPS, brdf_type="combined",
        )
    simulated = simulated.cpu().numpy()
    print(f"Forward pass ({N} pts): {time.time()-t0:.2f}s")

    # Normalise (same as original)
    sim_norm = simulated / simulated.mean()
    real_norm = real_curve / real_curve.mean()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    x_axis = np.linspace(0, 360, N)

    plt.plot(x_axis, sim_norm, color="blue", linewidth=2.0,
             label="Integrated Soft-Rasterizer (fixed)")
    plt.plot(x_axis, real_norm, color="red", linewidth=1.5,
             linestyle="--", alpha=0.8,
             label=f"{origin} Data ({target_column})")

    plt.title(f"Lightcurve Comparison: Model {target_model}")
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Normalized Intensity (Mean = 1.0)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("lightcurve_comparison_integrated.png", dpi=150)
    print("Saved plot to 'lightcurve_comparison_integrated.png'")

    # ------------------------------------------------------------------
    # Gradient sanity-check  (differentiability w.r.t. vertices)
    # ------------------------------------------------------------------
    print("\nGradient sanity check (4 angles) …")
    verts_g = verts.clone().requires_grad_(True)
    sim_g = compute_lightcurve(
        verts_g, faces,
        sun_dir_0, view_dir_0, angles[:4],
        gamma=GAMMA, eps=EPS, brdf_type="combined",
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

from torch.func import jvp, vmap

def jacobian_fwd(fn, p, chunk=8):
    """J[m, i] = d r_m / d p_i via chunked forward-mode. Returns (M, n)."""
    n = p.numel()
    I = torch.eye(n, device=p.device, dtype=p.dtype)
    blocks = []
    for i in range(0, n, chunk):
        tangents = I[i:i + chunk]
        # vmap over tangents; fall back to a Python loop if vmap chokes
        Jc = vmap(lambda v: jvp(fn, (p,), (v,))[1])(tangents)   # (c, M)
        blocks.append(Jc)
    return torch.cat(blocks).T.contiguous()

def main_multicam():
    device = torch.device("cuda")
    verts, faces = load_asteroid_mesh(3, device)
    vmin, vmax = verts.min(0).values, verts.max(0).values
    verts = (verts - (vmin + vmax) / 2) / (vmax - vmin).max()

    dataset = load_data(3, "simulated", "intensity")
    cols = list(dataset.data.columns)                      # all 28
    view_dirs0 = build_view_dirs(cols, device)
    sun_dir0 = torch.tensor([-1.0, 0.0, 0.0], device=device)
    N = 360                                                # Blender grid!
    angles = torch.arange(N, device=device) * (2 * np.pi / N)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        lcs = compute_lightcurves_multicam(verts, faces, sun_dir0, view_dirs0,
                                           angles, verbose=True)   # warm-up = compile
        torch.cuda.synchronize(); t0 = time.time()
        lcs = compute_lightcurves_multicam(verts, faces, sun_dir0, view_dirs0, angles)
        torch.cuda.synchronize()
    print(f"forward: {time.time()-t0:.2f}s, "
          f"peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    lcs = lcs / lcs.mean(dim=0, keepdim=True)              # per-curve normalisation

if __name__ == "__main__":
    main_multicam()
    
    
    # ---- template & parameterization -----------------------------------------
    sphere = trimesh.creation.icosphere(subdivisions=3)          # 642 v, 1280 f
    dirs_t  = F.normalize(torch.tensor(sphere.vertices, dtype=torch.float32,
                                    device=dev), dim=-1)      # (n, 3)
    faces_t = torch.tensor(sphere.faces, device=dev)
    n = dirs_t.shape[0]

    E = torch.tensor(sphere.edges_unique, device=dev)            # graph Laplacian
    Lap = torch.zeros(n, n, device=dev)
    Lap[E[:, 0], E[:, 1]] = -1.0; Lap[E[:, 1], E[:, 0]] = -1.0
    Lap += torch.diag(-Lap.sum(dim=1))

    def verts_from(p):                       # p = log-radii, r > 0 guaranteed
        return 0.5 * torch.exp(p).unsqueeze(1) * dirs_t

    def residuals(p, gamma):
        B = compute_lightcurves_multicam(
            verts_from(p), faces_t, sun_dir0, view_dirs0, angles,
            gamma=gamma, eps=1e-3, use_compile=False, use_checkpoint=False,
            dir_chunk=4)
        model = B / B.mean(dim=0, keepdim=True)
        r_photo = ((model - data_360) * w_curve).reshape(-1)     # w: 1/sigma per curve
        r_lap   = lam_lap * (Lap @ p)                            # smooth log-radius
        r_scale = lam_g * (p.mean().unsqueeze(0))                # pin mean size
        return torch.cat([r_photo, r_lap, r_scale])

    # ---- Levenberg–Marquardt ---------------------------------------------------
    def lm_fit(p, gamma, iters=25, lam=1e-2):
        fn = lambda q: residuals(q, gamma)
        with torch.no_grad():
            r = fn(p); cost = 0.5 * (r @ r).item()
        for it in range(iters):
            J = jacobian_fwd(fn, p, chunk=8)                     # (M, n)
            J64, r64 = J.double(), r.double()
            JtJ, Jtr = J64.T @ J64, J64.T @ r64                  # fp64: GN squares cond(J)
            dscale = JtJ.diagonal().clamp_min(1e-8 * JtJ.diagonal().max())
            while True:                                          # damping retries reuse J
                delta = torch.linalg.solve(JtJ + lam * torch.diag(dscale), -Jtr)
                with torch.no_grad():
                    r_new = fn(p + delta.float())
                    cost_new = 0.5 * (r_new @ r_new).item()
                pred = 0.5 * (delta @ (lam * dscale * delta - Jtr)).item()
                rho = (cost - cost_new) / max(pred, 1e-30)
                if rho > 1e-3:
                    p = p + delta.float(); r, cost = r_new, cost_new
                    lam = max(lam * max(1/3, 1 - (2*rho - 1)**3), 1e-10); break
                lam *= 3.0
                if lam > 1e8: return p                           # stalled
            print(f"[{it}] chi2={2*cost:.5g} lam={lam:.2e} |step|={delta.norm():.2e}")
            if delta.norm() < 1e-6 * p.norm(): break
        return p

    p = torch.zeros(n, device=dev)                               # unit sphere init
    for g in (0.05, 0.02, 0.01):                                 # gamma annealing
        p = lm_fit(p, gamma=g)
