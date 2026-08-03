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


if __name__ == "__main__":
    main()