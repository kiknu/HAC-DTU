#!/usr/bin/env python3
"""
star_forward.py — Star-shaped (radial offset) shape model and helpers.

Implements ShapeModel for an icosphere template with per-vertex log-radius
parameters.  Includes data-driven ellipsoid initialization and synthetic
shape generation.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from common import ShapeModel, LightcurvePhotometricEngine


def make_template(subdivisions=3, device="cpu"):
    """Icosphere template: unit directions, faces, random-walk Laplacian.

    subdiv 2 -> 162 verts, subdiv 3 -> 642 verts / 1280 faces.
    """
    sph = trimesh.creation.icosphere(subdivisions=subdivisions)
    dirs = F.normalize(
        torch.tensor(sph.vertices, dtype=torch.float32, device=device), dim=-1
    )
    faces = torch.tensor(sph.faces, dtype=torch.long, device=device)
    e = torch.tensor(sph.edges_unique, dtype=torch.long, device=device)
    n = dirs.shape[0]
    A = torch.zeros(n, n, device=device)
    A[e[:, 0], e[:, 1]] = 1.0
    A[e[:, 1], e[:, 0]] = 1.0
    L = torch.eye(n, device=device) - A / A.sum(dim=1, keepdim=True)
    return dirs, faces, L


def ellipsoid_log_radius(dirs, a, b, c, theta):
    """log r(u) of a triaxial ellipsoid (axes a,b,c; a-axis rotated by theta
    about +z), zero-mean (scale gauge)."""
    ct, st = math.cos(theta), math.sin(theta)
    ux = ct * dirs[:, 0] + st * dirs[:, 1]
    uy = -st * dirs[:, 0] + ct * dirs[:, 1]
    uz = dirs[:, 2]
    r = (ux**2 / a**2 + uy**2 / b**2 + uz**2 / c**2).rsqrt()
    p = torch.log(r)
    return p - p.mean()


class StarShapeModel(ShapeModel):
    """Star-shaped mesh: verts(p) = r0 * exp(p_i) * dir_i."""

    def __init__(
        self,
        dirs: torch.Tensor,
        faces: torch.Tensor,
        L: torch.Tensor,
        r0: float = 0.5,
        p: Optional[torch.Tensor] = None,
        lam_lap: float = 30.0,
        lam_scale: float = 10.0,
    ):
        self.dirs = dirs
        self.faces = faces
        self.L = L
        self.r0 = r0
        self.lam_lap = lam_lap
        self.lam_scale = lam_scale
        self.n = dirs.shape[0]
        if p is None:
            p = torch.zeros(self.n, device=dirs.device, dtype=dirs.dtype)
        self.p = p.clone().requires_grad_(True)

    def get_params(self) -> torch.Tensor:
        return self.p

    def set_params(self, params: torch.Tensor) -> None:
        with torch.no_grad():
            self.p.copy_(params.detach().to(self.p.device))

    def num_params(self) -> int:
        return self.n

    def compute_geometry(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        verts = self.r0 * torch.exp(params)[:, None] * self.dirs
        return verts, self.faces

    def regularization_residuals_at(self, params: torch.Tensor) -> torch.Tensor:
        r_lap = self.lam_lap * (self.L @ params)
        r_scale = self.lam_scale * params.mean().reshape(1)
        return torch.cat([r_lap, r_scale])


def generate_initial_guess(
    engine: LightcurvePhotometricEngine,
    shape: StarShapeModel,
    *,
    ab: Optional[float] = None,
    cb_grid: Tuple[float, ...] = (0.7, 0.85, 1.0),
    n_theta: int = 12,
    search_stride: int = 4,
    gamma: float = 0.05,
    verbose: bool = True,
):
    """
    Data-driven triaxial-ellipsoid initialization for a star-shaped model.

    (1) Elongation a/b from equatorial curves.
    (2) Spin phase theta0 and flattening c/b by brute force on a subsampled
        engine, scored with the actual forward model.
    """
    data = engine.data
    eq = engine.view_dirs0[:, 2].abs() < 0.05
    curves = data[:, eq] if bool(eq.any()) else data
    if ab is None:
        hi = torch.quantile(curves, 0.95, dim=0)
        lo = torch.quantile(curves, 0.05, dim=0).clamp_min(1e-3)
        ab = float((hi / lo).median().clamp(1.0, 2.5))

    coarse = engine.subsampled(search_stride)
    best = (float("inf"), None, None)
    for cb in cb_grid:
        for k in range(n_theta):
            theta = math.pi * k / n_theta
            p_try = ellipsoid_log_radius(shape.dirs, ab, 1.0, cb, theta)
            verts, faces = shape.compute_geometry(p_try)
            r_ph = coarse.photometric_residuals_from_geometry(verts, faces, gamma)
            cost = float((r_ph ** 2).sum())
            if cost < best[0]:
                best = (cost, p_try, (cb, theta))
    if verbose:
        print(
            f"[init] a/b={ab:.3f}  c/b={best[2][0]:.2f}  "
            f"theta0={math.degrees(best[2][1]):.0f} deg  "
            f"coarse photo chi2={best[0]:.3e}"
        )
    return best[1]


@torch.no_grad()
def radial_field(verts, faces, dirs, chunk=256, tol=1e-6):
    """Outermost hit distance from the ORIGIN along unit directions `dirs`.

    Möller-Trumbore specialized to a shared origin at 0.
    """
    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0
    n = torch.cross(e1, e2, dim=-1)
    w_u = torch.cross(v0, e2, dim=-1)
    w_v = torch.cross(e1, v0, dim=-1)
    a = (v0 * n).sum(-1)

    r = dirs.new_zeros(dirs.shape[0])
    for i in range(0, dirs.shape[0], chunk):
        d = dirs[i : i + chunk]
        S = d @ n.T
        ok = S.abs() > 1e-12
        inv = torch.where(ok, 1.0 / S, torch.zeros_like(S))
        t = a.unsqueeze(0) * inv
        u = -(d @ w_u.T) * inv
        v = -(d @ w_v.T) * inv
        hit = ok & (t > tol) & (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)
        r[i : i + chunk] = torch.where(hit, t, torch.zeros_like(t)).max(dim=1).values

    miss = r <= 0
    if bool(miss.any()):
        print(
            f"[radial_field] {int(miss.sum())} rays missed; "
            f"filling with median radius"
        )
        r[miss] = r[~miss].median()
    return r


def star_shaped_proxy(verts, faces, subdivisions=4):
    """Star-shaped remesh onto an icosphere template. Torch in, torch out."""
    sph = trimesh.creation.icosphere(subdivisions=subdivisions)
    dirs = F.normalize(
        torch.tensor(sph.vertices, dtype=verts.dtype, device=verts.device), dim=-1
    )
    r = radial_field(verts, faces, dirs)
    f = torch.tensor(sph.faces, dtype=torch.long, device=verts.device)
    return dirs * r[:, None], f


def synthetic_potato(
    device,
    subdivisions=4,
    seed=11,
    ab=1.5,
    cb=0.8,
    bump_rms=0.10,
    smooth_steps=60,
    theta=0.5,
):
    """Ellipsoid + smooth random log-radius bumps; runs without challenge data."""
    dirs, faces, L = make_template(subdivisions, device)
    q = torch.randn(
        dirs.shape[0], generator=torch.Generator().manual_seed(seed)
    )
    q = q.to(device)
    for _ in range(smooth_steps):
        q = q - 0.5 * (L @ q)
    q = q / q.std() * bump_rms
    p = ellipsoid_log_radius(dirs, ab, 1.0, cb, theta) + q
    return 0.5 * torch.exp(p - p.mean())[:, None] * dirs, faces
