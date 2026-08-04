#!/usr/bin/env python3
"""
common.py — Base classes, data I/O, and model-agnostic soft-rasterizer
for differentiable asteroid lightcurve inversion.

This module contains:
  • Abstract base classes for shape models and photometric engines.
  • A generic LightcurvePhotometricEngine that pre-computes direction
    bookkeeping and integrates soft-rasterized brightness for any mesh.
  • Data loaders, view-direction builders, and geometry helpers.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.checkpoint import checkpoint

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


PROJECT_DIR = Path(__file__).resolve().parents[0]
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
    return Dataset(
        model_num=model_num,
        origin=origin,
        curve_type=curve_type,
        data=data,
    )


def estimate_curve_weights(data: torch.Tensor) -> torch.Tensor:
    """Per-curve 1/sigma from robust MAD of the first difference."""
    d = data[1:] - data[:-1]
    mad = (d - d.median(dim=0).values).abs().median(dim=0).values
    sigma = (1.4826 / math.sqrt(2.0)) * mad
    return 1.0 / sigma.clamp_min(2e-3)


# --------------------------------------------------------------------------- #
# Base classes
# --------------------------------------------------------------------------- #

class ShapeModel(ABC):
    """Parameterized shape representation."""

    @abstractmethod
    def get_params(self) -> torch.Tensor:
        """Current parameter vector (1D)."""
        pass

    @abstractmethod
    def set_params(self, params: torch.Tensor) -> None:
        pass

    @abstractmethod
    def num_params(self) -> int:
        pass

    @abstractmethod
    def compute_geometry(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure functional: return (verts, faces) for the given params."""
        pass

    def get_vertices(self) -> torch.Tensor:
        return self.compute_geometry(self.get_params())[0]

    def get_faces(self) -> torch.Tensor:
        return self.compute_geometry(self.get_params())[1]

    @abstractmethod
    def regularization_residuals_at(self, params: torch.Tensor) -> torch.Tensor:
        """Prior residual vector for given params (e.g. Laplacian*logr, scale gauge)."""
        pass

    def regularization_residuals(self) -> torch.Tensor:
        return self.regularization_residuals_at(self.get_params())

    def residual_norm(self) -> float:
        r = self.regularization_residuals()
        return 0.5 * (r @ r).item()


class PhotometricEngine(ABC):
    """Computes brightness/residuals from geometry (verts, faces) and data."""

    @abstractmethod
    def brightness_from_geometry(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        gamma: float,
        context: Optional[Any] = None,
    ) -> torch.Tensor:
        """(N, C) raw brightness for all frames/cameras."""
        pass

    def brightness(
        self,
        shape_model: ShapeModel,
        gamma: float,
        frozen_vis: bool = False,
    ) -> torch.Tensor:
        verts, faces = shape_model.get_vertices(), shape_model.get_faces()
        ctx = self.linearization_context(verts, faces, gamma) if frozen_vis else None
        return self.brightness_from_geometry(verts, faces, gamma, ctx)

    @abstractmethod
    def photometric_residuals_from_geometry(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        gamma: float,
        weights: Optional[torch.Tensor] = None,
        context: Optional[Any] = None,
    ) -> torch.Tensor:
        """(N*C,) data residuals: (brightness/mean - data) * weight."""
        pass

    def photometric_residuals(
        self,
        shape_model: ShapeModel,
        gamma: float,
        weights: Optional[torch.Tensor] = None,
        frozen_vis: bool = False,
    ) -> torch.Tensor:
        verts, faces = shape_model.get_vertices(), shape_model.get_faces()
        ctx = self.linearization_context(verts, faces, gamma) if frozen_vis else None
        return self.photometric_residuals_from_geometry(verts, faces, gamma, weights, ctx)

    def linearization_context(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        gamma: float,
    ) -> Any:
        """Opaque context for frozen linearization (e.g. visibility table)."""
        return None

    @property
    @abstractmethod
    def data(self) -> torch.Tensor:
        """(N, C) observed mean‑normalized lightcurves."""
        pass


class PhotometricModel:
    """Combines a ShapeModel and PhotometricEngine; provides full residual vector."""

    def __init__(self, shape_model: ShapeModel, engine: PhotometricEngine):
        self.shape = shape_model
        self.engine = engine

    def residuals_at(
        self,
        params: torch.Tensor,
        gamma: float,
        weights: Optional[torch.Tensor] = None,
        context: Optional[Any] = None,
    ) -> torch.Tensor:
        """Pure functional residual evaluation for AD."""
        verts, faces = self.shape.compute_geometry(params)
        photo = self.engine.photometric_residuals_from_geometry(
            verts, faces, gamma, weights, context
        )
        reg = self.shape.regularization_residuals_at(params)
        return torch.cat([photo, reg])

    def photo_residuals_at(
        self,
        params: torch.Tensor,
        gamma: float,
        context: Optional[Any] = None,
    ) -> torch.Tensor:
        """Photometric part only (for initialization scoring)."""
        verts, faces = self.shape.compute_geometry(params)
        return self.engine.photometric_residuals_from_geometry(
            verts, faces, gamma, context=context
        )

    def residuals(
        self,
        gamma: float,
        weights: Optional[torch.Tensor] = None,
        frozen_vis: bool = False,
    ) -> torch.Tensor:
        """Stateful residual evaluation using current shape parameters."""
        ctx = None
        if frozen_vis:
            with torch.no_grad():
                verts, faces = self.shape.get_vertices(), self.shape.get_faces()
                ctx = self.engine.linearization_context(verts, faces, gamma)
        return self.residuals_at(self.shape.get_params(), gamma, weights, ctx)

    def get_params(self):
        return self.shape.get_params()

    def set_params(self, params):
        self.shape.set_params(params)

    def num_params(self):
        return self.shape.num_params()


# --------------------------------------------------------------------------- #
# Lightcurve photometric engine (model-agnostic)
# --------------------------------------------------------------------------- #

class LightcurvePhotometricEngine(PhotometricEngine):
    """
    Pre-computes direction bookkeeping (frame rotation, deduped visibility
    passes) and integrates soft-rasterized brightness for any mesh.
    """

    def __init__(
        self,
        data: torch.Tensor,
        angles: torch.Tensor,
        sun_dir0: torch.Tensor,
        view_dirs0: torch.Tensor,
        *,
        eps: float = 1e-3,
        brdf_type: Literal["lambertian", "lommel_seeliger", "combined"] = "combined",
        rotation_sign: float = -1.0,
        phase_offset: float = 0.0,
        frame_chunk: int = 120,
        dir_chunk: int = 8,
        dedupe_tol: float = 1e-4,
        cull_frac: Optional[float] = 0.65,
        w_curve: Optional[torch.Tensor] = None,
        use_compile: bool = True,
        use_checkpoint: bool = False,
        mask_backface_occluders: bool = True,
        vis_dtype: Optional[torch.dtype] = None,
    ):
        self.device = data.device
        self._data = data
        self.angles = angles
        self.sun_dir0 = sun_dir0
        self.view_dirs0 = view_dirs0
        self.N, self.C = data.shape
        self.eps = eps
        self.brdf_type = brdf_type
        self.frame_chunk = frame_chunk
        self.dir_chunk = dir_chunk
        self.cull_frac = cull_frac
        self.use_compile = use_compile
        self.use_checkpoint = use_checkpoint
        self.mask_backface_occluders = mask_backface_occluders
        self.vis_dtype = vis_dtype

        w = estimate_curve_weights(data) if w_curve is None else w_curve
        self.w = w.reshape(1, self.C)

        # Body-frame directions for every (frame, camera); dedupe ONCE here.
        psi = rotation_sign * angles + phase_offset
        self.sun_dirs = _rotate_dirs_z(sun_dir0, psi)  # (N, 3)
        self.view_dirs = _rotate_dirs_z(view_dirs0, psi)  # (N, C, 3)
        alld = torch.cat([self.sun_dirs, self.view_dirs.reshape(-1, 3)], 0)
        key = torch.round(alld / dedupe_tol)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        self.uniq_dirs = F.normalize(uniq * dedupe_tol, dim=-1)  # (U, 3)
        self.inv_sun = inv[: self.N]
        self.inv_view = inv[self.N :].reshape(self.N, self.C)

        # Store config for cheap subsampled copies
        self._cfg = dict(
            eps=eps,
            brdf_type=brdf_type,
            rotation_sign=rotation_sign,
            phase_offset=phase_offset,
            frame_chunk=frame_chunk,
            dir_chunk=dir_chunk,
            dedupe_tol=dedupe_tol,
            cull_frac=cull_frac,
            use_compile=use_compile,
            use_checkpoint=use_checkpoint,
            mask_backface_occluders=mask_backface_occluders,
            vis_dtype=vis_dtype,
        )

    @property
    def data(self) -> torch.Tensor:
        return self._data

    def subsampled(self, stride: int) -> "LightcurvePhotometricEngine":
        """Return a new engine on every `stride`-th frame (cheap init search)."""
        return LightcurvePhotometricEngine(
            self.data[::stride],
            self.angles[::stride],
            self.sun_dir0,
            self.view_dirs0,
            w_curve=self.w.flatten(),
            **self._cfg,
        )

    # ---- internal visibility table -----------------------------------------

    def _build_table(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        gamma: float,
        ckpt: bool = False,
    ) -> torch.Tensor:
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        areas = 0.5 * normals.norm(dim=-1)
        nhat = F.normalize(normals, dim=-1)
        return visibility_table(
            verts,
            faces,
            nhat,
            self.uniq_dirs,
            gamma=gamma,
            eps=self.eps,
            cull_frac=self.cull_frac,
            dir_chunk=self.dir_chunk,
            use_compile=self.use_compile,
            use_checkpoint=ckpt,
            mask_backface_occluders=self.mask_backface_occluders,
            vis_dtype=self.vis_dtype,
        )

    def linearization_context(
        self, verts: torch.Tensor, faces: torch.Tensor, gamma: float
    ) -> Any:
        with torch.no_grad():
            return self._build_table(verts, faces, gamma, ckpt=False)

    # ---- brightness / residuals --------------------------------------------

    def brightness_from_geometry(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        gamma: float,
        context: Optional[Any] = None,
    ) -> torch.Tensor:
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        areas = 0.5 * normals.norm(dim=-1)
        nhat = F.normalize(normals, dim=-1)

        table = context if context is not None else self._build_table(verts, faces, gamma)
        out = []
        for i in range(0, self.N, self.frame_chunk):
            j = i + self.frame_chunk
            out.append(
                _integrate_chunk(
                    areas,
                    nhat,
                    self.sun_dirs[i:j],
                    self.view_dirs[i:j],
                    table,
                    self.inv_sun[i:j],
                    self.inv_view[i:j],
                    self.brdf_type,
                )
            )
        return torch.cat(out, 0)  # (N, C)

    def photometric_residuals_from_geometry(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        gamma: float,
        weights: Optional[torch.Tensor] = None,
        context: Optional[Any] = None,
    ) -> torch.Tensor:
        B = self.brightness_from_geometry(verts, faces, gamma, context)
        model = B / B.mean(dim=0, keepdim=True)
        w = weights if weights is not None else self.w
        return ((model - self._data) * w).reshape(-1)


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


def _rotate_dirs_z(d0: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Apply R_z(-psi_k) to fixed world direction(s)."""
    shape = (psi.shape[0],) + (1,) * (d0.dim() - 1)
    c, s = torch.cos(psi).view(shape), torch.sin(psi).view(shape)
    x, y, z = d0[..., 0], d0[..., 1], d0[..., 2]
    xr = c * x + s * y
    yr = -s * x + c * y
    return torch.stack([xr, yr, z.expand_as(xr)], dim=-1)


# --------------------------------------------------------------------------- #
# Soft visibility
# --------------------------------------------------------------------------- #

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
    qy = centroids[:, 1:1]

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
    """Soft visibility for one direction, optionally restricted to faces."""
    basis = _ortho_basis(direction)
    proj = verts @ basis
    face_p = proj[faces]  # (F, 3, 3)

    if keep is None:
        centroids = face_p.mean(dim=1)[:, :2]
        return _soft_visibility(centroids, face_p[:, :, :2], face_p[:, :, 2], gamma, eps)

    idx = keep.nonzero(as_tuple=True)[0]
    vis = torch.zeros(faces.shape[0], dtype=verts.dtype, device=verts.device)
    if idx.numel() == 0:
        return vis
    face_sub = face_p[idx]
    centroids = face_sub.mean(dim=1)[:, :2]
    vis_sub = _soft_visibility(centroids, face_sub[:, :, :2], face_sub[:, :, 2], gamma, eps)
    vis.index_copy_(0, idx, vis_sub)
    return vis


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
    cull_backfaces: bool = True,
) -> torch.Tensor:
    """Disk-integrated brightness for a single (sun, view) pair."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = torch.cross(e1, e2, dim=-1)
    areas = 0.5 * normals.norm(dim=-1)
    nhat = F.normalize(normals, dim=-1)

    mu0 = torch.clamp((nhat * sun_dir).sum(dim=-1), min=0.0)
    mu = torch.clamp((nhat * view_dir).sum(dim=-1), min=0.0)

    if brdf_type == "lambertian":
        scatter = mu0 * mu
    elif brdf_type == "lommel_seeliger":
        scatter = mu0 * mu / (mu0 + mu + 1e-8)
    else:  # combined
        scatter = 0.5 * (mu0 * mu + mu0 * mu / (mu0 + mu + 1e-8))

    keep_v = (mu > 0) if cull_backfaces else None
    vis = _direction_visibility(verts, faces, view_dir, gamma, eps, keep_v)

    keep_s = (mu0 > 0) if cull_backfaces else None
    ill = _direction_visibility(verts, faces, sun_dir, gamma, eps, keep_s)

    return (areas * scatter * vis * ill).sum()


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

    brightness = torch.zeros(N, device=verts.device, dtype=verts.dtype)
    for k in range(N):
        brightness[k] = fn(
            verts,
            faces,
            sun_dirs[k],
            view_dirs[k],
            gamma=gamma,
            eps=eps,
            brdf_type=brdf_type,
            cull_backfaces=cull,
        )
    return brightness


# --------------------------------------------------------------------------- #
# Batched visibility table and multi-camera lightcurves
# --------------------------------------------------------------------------- #

def _soft_visibility_batch(face_xy, face_z, gamma, eps, valid=None):
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

    cx, cy = (x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0

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

    eye = torch.eye(K, dtype=torch.bool, device=face_z.device)
    inside = inside.masked_fill(eye, 0.0)

    z_query = (z0 + z1 + z2) / 3.0
    delta = (z_interp - z_query.unsqueeze(2)) / gamma
    w = (inside * torch.exp(delta.clamp(max=60.0))).sum(dim=2)
    return 1.0 / (1.0 + w)


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
):
    """Soft visibility for a chunk of directions -> (B, F)."""
    B, n_faces = dirs.shape[0], faces.shape[0]
    bases = _ortho_bases(dirs)
    proj = torch.einsum("vi,bij->bvj", verts, bases)  # (B, V, 3)
    face_p = proj[:, faces]  # (B, F, 3, 3)
    mu = torch.einsum("fi,bi->bf", nhat, dirs)  # (B, F)

    if k_keep is not None and k_keep < n_faces:
        idx = mu.topk(k_keep, dim=1, sorted=False).indices  # (B, K)
        face_sub = torch.gather(face_p, 1, idx[:, :, None, None].expand(B, k_keep, 3, 3))
        mu_sub = torch.gather(mu, 1, idx)
        if vis_dtype is not None:
            face_sub = face_sub.to(vis_dtype)
        valid = (mu_sub > 0.0) if mask_backface_occluders else None
        vis_sub = _soft_visibility_batch(face_sub[..., :2], face_sub[..., 2], gamma, eps, valid)
        vis_sub = vis_sub.to(verts.dtype) * (mu_sub > 0.0)
        return proj.new_zeros(B, n_faces).scatter(1, idx, vis_sub)

    fp = face_p.to(vis_dtype) if vis_dtype is not None else face_p
    valid = (mu > 0.0) if mask_backface_occluders else None
    vis = _soft_visibility_batch(fp[..., :2], fp[..., 2], gamma, eps, valid)
    vis = vis.to(verts.dtype)
    if mask_backface_occluders:
        vis = vis * (mu > 0.0)
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
    cull_frac=0.65,
    dir_chunk=4,
    use_compile=True,
    use_checkpoint=False,
    mask_backface_occluders=True,
    vis_dtype=None,
):
    """(D, 3) directions -> (D, F) soft visibility, chunked over D."""
    D, n_faces = dirs.shape[0], faces.shape[0]
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
        )
        v = (
            checkpoint(fn, *args, use_reentrant=False)
            if use_checkpoint
            else fn(*args)
        )
        outs.append(v[:n_real])
    return torch.cat(outs, dim=0)


def _integrate_chunk(areas, nhat, sun_dirs, view_dirs, vis_table, inv_sun, inv_view, brdf_type):
    """Photometric contraction for a chunk of frames -> (n, C)."""
    ill = vis_table[inv_sun]  # (n, F)
    vis = vis_table[inv_view]  # (n, C, F)
    mu0 = torch.einsum("fi,ni->nf", nhat, sun_dirs).clamp_min(0.0)
    mu = torch.einsum("fi,nci->ncf", nhat, view_dirs).clamp_min(0.0)
    l = mu0[:, None, :] * mu  # foreshortening
    if brdf_type == "lambertian":
        scatter = l
    elif brdf_type == "lommel_seeliger":
        scatter = l / (mu0[:, None, :] + mu + 1e-8)
    else:  # combined
        scatter = 0.5 * l * (1.0 + 1.0 / (mu0[:, None, :] + mu + 1e-8))
    return torch.einsum("ncf,nf->nc", scatter * vis, areas[None, :] * ill)


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
    rotation_sign=-1.0,
    phase_offset=0.0,
    cull_frac=0.65,
    dir_chunk=4,
    frame_chunk=60,
    use_compile=True,
    use_checkpoint=False,
    dedupe_tol=1e-4,
    vis_dtype=None,
    verbose=False,
):
    """Lightcurves for all cameras at once. Returns (N, C)."""
    N, C = angles.shape[0], view_dirs0.shape[0]

    # Per‑face geometry
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * normals.norm(dim=-1)
    nhat = F.normalize(normals, dim=-1)

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
        nhat,
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
            nhat,
            sun_dirs[i : i + frame_chunk],
            view_dirs[i : i + frame_chunk],
            table,
            inv_sun[i : i + frame_chunk],
            inv_view[i : i + frame_chunk],
            brdf_type,
        )
        out.append(
            checkpoint(_integrate_chunk, *args, use_reentrant=False)
            if use_checkpoint
            else _integrate_chunk(*args)
        )
    return torch.cat(out, dim=0)  # (N, C)


CAMERA_ALPHA_DEG = {"hor1": 0.0, "hor2": 0.0, "top": 26.0, "bottom": -26.0}


def build_view_dirs(columns, device):
    """Column names ('45_top', ...) -> (C, 3) world-frame view directions."""
    dirs = []
    for col in columns:
        az, kind = col.split("_")
        phi, alpha = np.radians(float(az) - 180.0), np.radians(CAMERA_ALPHA_DEG[kind])
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
