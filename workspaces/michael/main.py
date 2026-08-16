#!/usr/bin/env python3
"""
gn_shape_fit.py — Gauss-Newton / Levenberg-Marquardt shape inversion from
multi-camera lightcurves, on top of forward_model.py.

Shape parameterizations implement one mesh-producing protocol.  The built-in
models are a per-vertex star shape and a spherical-harmonic octantoid.

Residual stack (single damped GN solve over all rows):
    r_photo : (model / mean(model) - data) * w_c      w_c = 1/sigma_c per curve
    r_prior : representation-specific shape regularization and scale gauge

Update backends (compute_update):
    "frozen": explicit J by chunked forward-mode AD with the O(F^2) soft
              visibility/shadow tables DETACHED at the linearization point.
              Fast; the LM rho-test uses exact residuals so bad steps are
              rejected. Recommended workhorse.
    "exact" : explicit J through everything (small templates / debugging).
    "cg"    : matrix-free exact GN: PCG on (JtJ + lam*D) d = -Jt r, one jvp +
              one VJP per iteration; reverse graph cached per linearization,
              visibility checkpointed so backward memory ~ one dir-chunk.

Direction bookkeeping and non-convex visibility live in the mesh-only
PhotometricForwardModel.  ShapeFitProblem composes that renderer with any
ShapeModel and exposes pure residual closures safe for
torch.func.{jvp,vjp,vmap}.
"""

import math
import time
from typing import Protocol
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.func import jvp, vmap
from shape_model import (
    EllipsoidInitialization,
    OctantoidShapeModel,
    ShapeModel,
    StarShapeModel,
    TriangleMesh,
)
from forward_model import (
    ObservationGeometry,
    PhotometricForwardModel,
    PreparedVisibility,
    build_view_dirs,
    compute_lightcurves_multicam,
    load_asteroid_mesh,
)

# --------------------------------------------------------------------------- #
# Template, priors, weights
# --------------------------------------------------------------------------- #

def make_template(subdivisions=3, device="cpu"):
    """Icosphere template: unit directions, faces, random-walk Laplacian.

    subdiv 2 -> 162 verts, subdiv 3 -> 642 verts / 1280 faces (closest
    icosphere to the ~512 target; regular valence => well-behaved Laplacian).
    """
    sph = trimesh.creation.icosphere(subdivisions=subdivisions)
    dirs = F.normalize(torch.tensor(sph.vertices, dtype=torch.float32,
                                    device=device), dim=-1)
    faces = torch.tensor(sph.faces, dtype=torch.long, device=device)
    e = torch.tensor(sph.edges_unique, dtype=torch.long, device=device)
    n = dirs.shape[0]
    A = torch.zeros(n, n, device=device)
    A[e[:, 0], e[:, 1]] = 1.0
    A[e[:, 1], e[:, 0]] = 1.0
    L = torch.eye(n, device=device) - A / A.sum(dim=1, keepdim=True)
    return dirs, faces, L


def estimate_curve_weights(data):
    """Per-curve 1/sigma from the high-frequency component (robust MAD of the
    first difference). Makes photometric rows ~ N(0,1) at the solution, so
    lam_lap has a consistent meaning across datasets."""
    d = data[1:] - data[:-1]
    mad = (d - d.median(dim=0).values).abs().median(dim=0).values
    sigma = (1.4826 / math.sqrt(2.0)) * mad
    return 1.0 / sigma.clamp_min(2e-3)


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

# --------------------------------------------------------------------------- #
# Fitter: forward model + residuals with all bookkeeping precomputed
# --------------------------------------------------------------------------- #

class LeastSquaresProblem(Protocol):
    """Representation-independent contract consumed by the solver."""

    n_params: int
    n_parameters: int
    n_photo: int

    def residuals(self, p, gamma, table=None, ckpt=False): ...
    def residual_function(self, p, gamma, mode): ...
    def project_parameters(self, p): ...
    def validate_trial(self, p) -> bool: ...


class ShapeFitProblem:
    """Lightcurve least-squares problem composed from a shape and renderer."""

    def __init__(
        self,
        data,
        angles,
        sun_dir0,
        view_dirs0,
        *,
        data_bin=None,
        tau_bin=0.10,
        beta_bin=8.0,
        bin_sigma_floor=2e-3,
        lam_bin=1.0,
        w_curve=None,
        w_bin=None,
        shape_model: ShapeModel | None = None,
        representation="star",
        subdiv=3,
        lmax=10,
        lmax_a=None,
        lmax_b=None,
        lmax_c=None,
        detail_lmax=None,
        detail_scale=0.20,
        max_log_scale=3.0,
        eps=1e-3,
        brdf_type="combined",
        projection="orthographic",
        camera_distance_radii=5.0,
        visibility_samples=1,
        rotation_sign=-1.0,
        phase_offset=0.0,
        lam_lap=30.0,
        lam_scale=10.0,
        lam_shape=0.3,
        lam_detail=3.0,
        sh_degree_power=2.0,
        r0=0.5,
        frame_chunk=120,
        dir_chunk=8,
        dedupe_tol=1e-4,
    ):
        if detail_lmax is not None:
            raise ValueError(
                "Cartesian detail is not part of the pure octantoid model"
            )
        del detail_scale, lam_detail

        self.device = data.device
        self.data = data
        self.data_bin = data_bin
        self.angles = angles
        self.sun_dir0 = sun_dir0
        self.view_dirs0 = view_dirs0
        self.N, self.C = data.shape
        if angles.shape != (self.N,):
            raise ValueError("angles must have shape (N,)")
        if view_dirs0.shape != (self.C, 3):
            raise ValueError("view_dirs0 must have shape (C, 3)")
        if data_bin is not None and data_bin.shape != data.shape:
            raise ValueError("data_bin must have the same shape as data")

        self.n_int = self.N * self.C
        self.n_bin = self.N * self.C if data_bin is not None else 0
        self.n_photo = self.n_int + self.n_bin
        self.tau_bin = float(tau_bin)
        self.bin_sigma_floor = float(bin_sigma_floor)
        self.lam_bin = float(lam_bin)
        self.eps = float(eps)
        self.brdf_type = brdf_type
        self.projection = projection
        self.camera_distance_radii = float(camera_distance_radii)
        self.visibility_samples = int(visibility_samples)
        self.frame_chunk = int(frame_chunk)
        self.dir_chunk = int(dir_chunk)

        if shape_model is None:
            dirs, faces, laplacian = make_template(subdiv, self.device)
            if representation == "star":
                shape_model = StarShapeModel(
                    dirs,
                    faces,
                    laplacian,
                    r0=r0,
                    lam_lap=lam_lap,
                    lam_scale=lam_scale,
                )
            elif representation == "octantoid":
                shape_model = OctantoidShapeModel(
                    dirs,
                    faces,
                    lmax=lmax,
                    lmax_a=lmax_a,
                    lmax_b=lmax_b,
                    lmax_c=lmax_c,
                    r0=r0,
                    lam_shape=lam_shape,
                    degree_power=sh_degree_power,
                    max_log_scale=max_log_scale,
                )
            else:
                raise ValueError(
                    "representation must be 'star' or 'octantoid'"
                )

        self.shape_model = shape_model
        self.n_params = shape_model.n_parameters
        self.n_parameters = self.n_params
        self.faces_t = shape_model.faces
        self.dirs_t = getattr(shape_model, "directions", None)
        self.L = getattr(shape_model, "laplacian", None)
        self.r0 = float(shape_model.r0)
        if isinstance(shape_model, StarShapeModel):
            self.representation = "star"
            self.octantoid = None
        elif isinstance(shape_model, OctantoidShapeModel):
            self.representation = "octantoid"
            self.octantoid = shape_model
        else:
            self.representation = type(shape_model).__name__
            self.octantoid = None

        self.w = (
            estimate_curve_weights(data) if w_curve is None else w_curve
        ).reshape(1, self.C)
        if data_bin is None:
            self.w_bin = None
        elif w_bin is not None:
            self.w_bin = w_bin.reshape(1, self.C)
        else:
            differences = data_bin[1:] - data_bin[:-1]
            mad = (
                differences - differences.median(dim=0).values
            ).abs().median(dim=0).values
            sigma = (1.4826 / math.sqrt(2.0)) * mad
            self.w_bin = (
                lam_bin / sigma.clamp_min(bin_sigma_floor)
            ).reshape(1, self.C)

        self.observation_geometry = ObservationGeometry.build(
            angles,
            sun_dir0,
            view_dirs0,
            rotation_sign=rotation_sign,
            phase_offset=phase_offset,
            dedupe_tol=dedupe_tol,
        )
        self.forward_model = PhotometricForwardModel(
            self.observation_geometry,
            eps=eps,
            brdf_type=brdf_type,
            projection=projection,
            camera_distance_radii=camera_distance_radii,
            visibility_samples=visibility_samples,
            tau_binary=tau_bin,
            beta_binary=beta_bin,
            frame_chunk=frame_chunk,
            direction_chunk=dir_chunk,
        )
        self.sun_dirs = self.observation_geometry.sun_dirs
        self.view_dirs = self.observation_geometry.view_dirs
        self.uniq_dirs = self.observation_geometry.unique_dirs
        self.inv_sun = self.observation_geometry.inv_sun
        self.inv_view = self.observation_geometry.inv_view

        # Only observation settings are needed when creating a subsampled
        # problem; the immutable shape model is reused.
        self._cfg = dict(
            tau_bin=tau_bin,
            beta_bin=beta_bin,
            bin_sigma_floor=bin_sigma_floor,
            lam_bin=lam_bin,
            eps=eps,
            brdf_type=brdf_type,
            projection=projection,
            camera_distance_radii=camera_distance_radii,
            visibility_samples=visibility_samples,
            rotation_sign=rotation_sign,
            phase_offset=phase_offset,
            frame_chunk=frame_chunk,
            dir_chunk=dir_chunk,
            dedupe_tol=dedupe_tol,
        )

    @property
    def beta_bin(self):
        return self.forward_model.beta_binary

    @beta_bin.setter
    def beta_bin(self, value):
        self.forward_model.beta_binary = float(value)

    def configure_stage(self, stage):
        if "beta_bin" in stage:
            self.beta_bin = float(stage["beta_bin"])

    def verts(self, p):
        return self.shape_model.mesh(p).vertices

    def shape_prior(self, p):
        return self.shape_model.prior_residuals(p)

    def project_parameters(self, p):
        return self.shape_model.project_parameters(p)

    def validate_trial(self, p) -> bool:
        return self.shape_model.validate_mesh(
            self.shape_model.mesh(p)
        ).valid

    def _geom(self, p):
        mesh = self.shape_model.mesh(p)
        areas, normals = self.forward_model.face_geometry(mesh)
        return mesh.vertices, areas, normals

    def _table(self, verts, nhat, gamma, ckpt=False):
        del nhat
        return self.forward_model.prepare_visibility(
            TriangleMesh(verts, self.faces_t),
            gamma,
            checkpoint_visibility=ckpt,
        )

    def curves_all(self, p, gamma, table=None, ckpt=False, want_bin=True):
        want_bin = want_bin and self.data_bin is not None
        prepared = (
            None
            if table is None
            else table
            if isinstance(table, PreparedVisibility)
            else PreparedVisibility(table, float(gamma))
        )
        prediction = self.forward_model.predict(
            self.shape_model.mesh(p),
            gamma,
            prepared=prepared,
            checkpoint_visibility=ckpt,
            include_binary=want_bin,
        )
        return prediction.intensity, prediction.binary

    def curves(self, p, gamma, table=None, ckpt=False):
        return self.curves_all(
            p, gamma, table=table, ckpt=ckpt, want_bin=False
        )[0]

    def residuals(self, p, gamma, table=None, ckpt=False):
        intensity, binary = self.curves_all(
            p, gamma, table=table, ckpt=ckpt
        )
        model = intensity / intensity.mean(
            dim=0, keepdim=True
        ).clamp_min(1e-9)
        parts = [((model - self.data) * self.w).reshape(-1)]
        if binary is not None:
            model_binary = binary / binary.mean(
                dim=0, keepdim=True
            ).clamp_min(1e-9)
            parts.append(
                ((model_binary - self.data_bin) * self.w_bin).reshape(-1)
            )
        parts.append(self.shape_model.prior_residuals(p))
        return torch.cat(parts)

    def residual_function(self, p, gamma, mode):
        if mode == "exact":
            return lambda q: self.residuals(q, gamma)
        if mode != "frozen":
            raise ValueError(f"unknown linearization mode {mode!r}")
        with torch.no_grad():
            prepared = self.forward_model.prepare_visibility(
                self.shape_model.mesh(p), gamma, detach=True
            )
        return lambda q: self.residuals(q, gamma, table=prepared)

    @torch.no_grad()
    def photo_cost(self, p, gamma):
        residual = self.residuals(p, gamma)
        return float((residual[: self.n_photo] ** 2).sum())

    def subsampled(self, stride):
        return ShapeFitProblem(
            self.data[::stride],
            self.angles[::stride],
            self.sun_dir0,
            self.view_dirs0,
            data_bin=(
                None if self.data_bin is None else self.data_bin[::stride]
            ),
            w_curve=self.w.flatten(),
            w_bin=(None if self.w_bin is None else self.w_bin.flatten()),
            shape_model=self.shape_model,
            **self._cfg,
        )


# Backward-compatible import name; implementation is now model-agnostic.
ShapeFitter = ShapeFitProblem


# --------------------------------------------------------------------------- #
# 1) Initial guess
# --------------------------------------------------------------------------- #

def generate_initial_guess(fitter, *, ab=None, cb_grid=(0.7, 0.85, 1.0),
                           n_theta=12, search_stride=4, gamma=0.05,
                           verbose=True):
    """
    Data-driven triaxial-ellipsoid initialization.

    (1) Elongation a/b from the equatorial curves: for a geometrically
        scattering ellipsoid spinning about c, the equatorial brightness is
        ~ projected area  pi*c*sqrt(a^2 sin^2 + b^2 cos^2), so max/min = a/b.
        Robust quantiles stand in for max/min. (Absolute levels are lost to
        the per-curve mean normalization, so c is NOT observable this way.)
    (2) Spin phase theta0 and flattening c/b by brute force, scored with the
        actual forward model on a frame-subsampled fitter (cheap; ellipsoids
        are 180-deg symmetric so theta in [0, pi)).

    Returns a parameter vector in the fitter's shape representation.
    """
    eq = fitter.view_dirs0[:, 2].abs() < 0.05
    curves = fitter.data[:, eq] if bool(eq.any()) else fitter.data
    if ab is None:
        hi = torch.quantile(curves, 0.95, dim=0)
        lo = torch.quantile(curves, 0.05, dim=0).clamp_min(1e-3)
        ab = float((hi / lo).median().clamp(1.0, 2.5))

    coarse = fitter.subsampled(search_stride)
    best = (float("inf"), None, None)
    for cb in cb_grid:
        for k in range(n_theta):
            theta = math.pi * k / n_theta

            p_try = fitter.shape_model.initial_parameters(
                EllipsoidInitialization(ab, 1.0, cb, theta)
            )
            cost = coarse.photo_cost(p_try, gamma)
            if cost < best[0]:
                best = (cost, p_try, (cb, theta))
    if verbose:
        print(f"[init] a/b={ab:.3f}  c/b={best[2][0]:.2f}  "
              f"theta0={math.degrees(best[2][1]):.0f} deg  "
              f"coarse photo chi2={best[0]:.3e}")
    return best[1]

# --------------------------------------------------------------------------- #
# 2) Update step (three backends)
# --------------------------------------------------------------------------- #

def build_jacobian(fn, p, chunk=16):
    """J[m, i] = d r_m / d p_i via chunked forward-mode AD. Returns (M, n).
    Forward mode is the right direction here: n ~ 642 << M ~ 6000+."""
    n = p.numel()
    I = torch.eye(n, device=p.device, dtype=p.dtype)
    blocks = []
    for i in range(0, n, chunk):
        Jc = vmap(lambda v: jvp(fn, (p,), (v,))[1])(I[i:i + chunk])  # (c, M)
        blocks.append(Jc)
    return torch.cat(blocks, 0).T.contiguous()


def _pcg(A, b, Minv, iters=50, tol=1e-3):
    """Jacobi-preconditioned CG for the SPD system A x = b."""
    x = torch.zeros_like(b)
    bn = float(b.norm())
    if bn < 1e-30:
        return x
    r = b.clone()
    z = Minv * r
    d = z.clone()
    rz = float(r @ z)
    for _ in range(iters):
        Ad = A(d)
        alpha = rz / max(float(d @ Ad), 1e-30)
        x = x + alpha * d
        r = r - alpha * Ad
        if float(r.norm()) <= tol * bn:
            break
        z = Minv * r
        rz_new = float(r @ z)
        d = z + (rz_new / rz) * d
        rz = rz_new
    return x


def compute_update(
    fitter: LeastSquaresProblem,
    p,
    gamma,
    lam,
    method="frozen",
    lin=None,
    jac_chunk=16,
    cg_iters=50,
    cg_tol=1e-3,
    hutch_probes=8,
):
    """
    One damped Gauss-Newton (LM) step around p:  (JtJ + lam*D) delta = -Jt r.

    Returns (delta, pred_red, lin). Pass `lin` back in when retrying with a
    larger lam: the expensive linearization (J / VJP graph / diag estimate)
    is reused, only the cheap solve is repeated.

    pred_red is the reduction of the GN quadratic model — compare against the
    actual reduction of the EXACT cost (rho-test) in the outer loop.
    """
    # ---------- explicit-J backends ----------------------------------------
    if method in ("frozen", "exact"):
        if lin is None:
            fn = fitter.residual_function(p, gamma, method)
            with torch.no_grad():
                r = fn(p)          # frozen table built at p => exact residual
            J = build_jacobian(fn, p, chunk=jac_chunk).double()
            JtJ = J.T @ J                       # fp64: GN squares cond(J)
            Jtr = J.T @ r.double()
            D = JtJ.diagonal().clone()
            D = D.clamp_min(1e-10 * float(D.max()) + 1e-300)
            lin = dict(JtJ=JtJ, Jtr=Jtr, D=D)
        JtJ, Jtr, D = lin["JtJ"], lin["Jtr"], lin["D"]
        delta = torch.linalg.solve(JtJ + lam * torch.diag(D), -Jtr)
        pred = 0.5 * float(delta @ (lam * D * delta - Jtr))
        return delta.to(p.dtype), pred, lin

    # ---------- matrix-free exact GN with PCG -------------------------------
    if method != "cg":
        raise ValueError(f"unknown method {method!r}")
    if lin is None:
        p_leaf = p.detach().clone().requires_grad_(True)
        r_graph = fitter.residuals(p_leaf, gamma, ckpt=True)  # cached VJP graph

        def JT(u):
            (g,) = torch.autograd.grad(r_graph, p_leaf, grad_outputs=u,
                                       retain_graph=True)
            return g

        g0 = JT(r_graph.detach())                             # Jt r
        # Hutchinson estimate of diag(JtJ):  E[(Jt z)_i^2] = sum_m J_mi^2
        Dd = torch.zeros_like(p)
        M = r_graph.numel()
        for _ in range(hutch_probes):
            z = torch.randint(0, 2, (M,), device=p.device,
                              dtype=p.dtype) * 2.0 - 1.0
            u = JT(z)
            Dd += u * u
        Dd = Dd / hutch_probes
        Dd = Dd.clamp_min(1e-8 * float(Dd.max()) + 1e-30)
        lin = dict(JT=JT, g=g0, D=Dd)
    JT, g, D = lin["JT"], lin["g"], lin["D"]

    fn_fwd = lambda q: fitter.residuals(q, gamma)             # jvp: no ckpt

    def Av(v):
        Jv = jvp(fn_fwd, (p,), (v,))[1]
        return JT(Jv) + lam * D * v

    delta = _pcg(Av, -g, 1.0 / ((1.0 + lam) * D), iters=cg_iters, tol=cg_tol)
    JtJd = Av(delta) - lam * D * delta                        # exact pred for
    pred = float(-(delta @ g) - 0.5 * (delta @ JtJd))         # the inexact step
    return delta, pred, lin

# --------------------------------------------------------------------------- #
# 3) Solver (LM outer loop with gamma annealing)
# --------------------------------------------------------------------------- #

def solve(
    fitter: LeastSquaresProblem,
    p0,
    stages=None,
    lam0=1e-2,
    max_retries=10,
    step_tol=1e-4,
    verbose=True,
):
    """
    Levenberg-Marquardt driver. Each stage fixes (gamma, method); lam is
    shared across stages and adapted by the Nielsen rho-rule. Damping retries
    reuse the linearization via `lin`. Trial steps are always scored with the
    EXACT residual (fresh visibility tables at the trial point), so the
    frozen-J approximation can only slow convergence, never corrupt it.
    """
    if stages is None:
        stages = [
            dict(gamma=0.06,  iters=6,  method="frozen", beta_bin=6.0),   # proportions
            dict(gamma=0.03,  iters=8,  method="frozen", beta_bin=12.0),
            dict(gamma=0.015, iters=8,  method="cg",     beta_bin=25.0,   # carve
                 update_kwargs=dict(cg_iters=40, cg_tol=1e-3)),           # concavities
        ]

    p, lam, history = p0.clone(), lam0, []
    for si, st in enumerate(stages):
        gamma = st["gamma"]
        method = st.get("method", "frozen")
        kw = st.get("update_kwargs", {})

        configure_stage = getattr(fitter, "configure_stage", None)
        if configure_stage is not None:
            configure_stage(st)
        for it in range(st["iters"]):
            with torch.no_grad():
                r = fitter.residuals(p, gamma)
                cost = 0.5 * float(r @ r)
            lin, accepted, accepted_delta = None, False, None
            rho, cost_new = float("nan"), cost
            for _ in range(max_retries):
                delta, pred, lin = compute_update(
                    fitter,
                    p,
                    gamma,
                    lam,
                    method=method,
                    lin=lin,
                    **kw,
                )

                p_trial = fitter.project_parameters(p + delta)

                if not fitter.validate_trial(p_trial):
                    lam = min(lam * 4.0, 1e9)
                    continue

                with torch.no_grad():
                    r_new = fitter.residuals(p_trial, gamma)
                    cost_new = 0.5 * float(r_new @ r_new)

                rho = (cost - cost_new) / max(pred, 1e-30)

                if math.isfinite(cost_new) and rho > 1e-3:
                    accepted_delta = p_trial - p
                    p = p_trial
                    lam = max(
                        lam * max(
                            1.0 / 3.0,
                            1.0 - (2.0 * rho - 1.0) ** 3,
                        ),
                        1e-12,
                    )
                    accepted = True
                    break
                lam = min(lam * 4.0, 1e9)
            step = float(accepted_delta.norm()) if accepted else 0.0
            photo = float((r_new[:fitter.n_photo] ** 2).sum()) if accepted \
                else float((r[:fitter.n_photo] ** 2).sum())
            history.append(dict(stage=si, iter=it, gamma=gamma, cost=cost,
                                cost_new=cost_new if accepted else cost,
                                rho=rho, lam=lam, step=step, photo=photo,
                                p=p.detach().clone()))
            if verbose:
                rms = math.sqrt(photo / fitter.n_photo)
                print(f"[s{si} i{it:02d}] g={gamma:.3f} cost {cost:.4e} -> "
                      f"{history[-1]['cost_new']:.4e}  rho={rho:5.2f} "
                      f"lam={lam:.1e} |dp|={step:.2e}  photo-rms={rms:.3f}")
            if not accepted:
                if verbose:
                    print("   no acceptable step — moving to next stage")
                break
            if step < step_tol * max(float(p.norm()), 1.0):
                if verbose:
                    print("   converged (step tolerance)")
                break
    return p, history

# --------------------------------------------------------------------------- #
# 4) Test fit (synthetic self-consistency, mild inverse-crime mitigation)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def radial_field(verts, faces, dirs, chunk=256, tol=1e-6):
    """Outermost hit distance from the ORIGIN along unit directions `dirs`.

    Möller-Trumbore specialized to a shared origin at 0: reduces to three
    (B,3)@(3,F) matmuls per chunk. No trimesh.ray / rtree / embree needed.

    verts (V,3) float, faces (F,3) long, dirs (D,3) float — same device.
    Returns r (D,): the star-shaped radial support of the mesh.
    """
    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0
    n   = torch.cross(e1, e2, dim=-1)          # det = -d·n
    w_u = torch.cross(v0, e2, dim=-1)          # u·det =  d·w_u
    w_v = torch.cross(e1, v0, dim=-1)          # v·det =  d·w_v
    a   = (v0 * n).sum(-1)                     # t = a / (d·n)

    r = dirs.new_zeros(dirs.shape[0])
    for i in range(0, dirs.shape[0], chunk):
        d = dirs[i:i + chunk]                                  # (B, 3)
        S = d @ n.T                                            # (B, F) = d·n
        ok = S.abs() > 1e-12                                   # skip edge-on/degenerate
        inv = torch.where(ok, 1.0 / S, torch.zeros_like(S))
        t = a.unsqueeze(0) * inv
        u = -(d @ w_u.T) * inv                                 # /det = /(-S)
        v = -(d @ w_v.T) * inv
        hit = ok & (t > tol) & (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)
        r[i:i + chunk] = torch.where(hit, t, torch.zeros_like(t)).max(dim=1).values

    miss = r <= 0
    if bool(miss.any()):                       # can't happen for a watertight
        print(f"[radial_field] {int(miss.sum())} rays missed; "  # mesh enclosing 0
              f"filling with median radius")
        r[miss] = r[~miss].median()
    return r


def star_shaped_proxy(verts, faces, subdivisions=4):
    """Star-shaped remesh onto an icosphere template. Torch in, torch out."""
    sph = trimesh.creation.icosphere(subdivisions=subdivisions)
    dirs = F.normalize(torch.tensor(sph.vertices, dtype=verts.dtype,
                                    device=verts.device), dim=-1)
    r = radial_field(verts, faces, dirs)
    f = torch.tensor(sph.faces, dtype=torch.long, device=verts.device)
    return dirs * r[:, None], f


def synthetic_potato(device, subdivisions=4, seed=11, ab=1.5, cb=0.8,
                     bump_rms=0.10, smooth_steps=60, theta=0.5):
    """Ellipsoid + smooth random log-radius bumps; runs without challenge data."""
    dirs, faces, L = make_template(subdivisions, device)
    q = torch.randn(dirs.shape[0], generator=torch.Generator().manual_seed(seed))
    q = q.to(device)
    for _ in range(smooth_steps):
        q = q - 0.5 * (L @ q)
    q = q / q.std() * bump_rms
    p = ellipsoid_log_radius(dirs, ab, 1.0, cb, theta) + q
    return 0.5 * torch.exp(p - p.mean())[:, None] * dirs, faces


def fd_check(fitter, p, gamma, h=1e-3):
    """Directional finite-difference check of the EXACT Jacobian (jvp path)."""
    v = torch.randn_like(p)
    v = v / v.norm()
    fn = lambda q: fitter.residuals(q, gamma)
    with torch.no_grad():
        fd = (fn(p + h * v) - fn(p - h * v)) / (2 * h)
    an = jvp(fn, (p,), (v,))[1]
    return float((fd - an).norm() / (an.norm() + 1e-30))


def test_fit(model_num=1, n_frames=840, fit_stride=3, noise=0.01,
             subdiv=3, lam_lap=30.0, seed=0, device=None, make_plots=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cpu":
        print("[test] CPU detected — shrinking problem for a smoke test")
        n_frames, proxy_subdiv = 240, 3
    else:
        proxy_subdiv = 4

    # ---- ground truth (challenge STL if available, else synthetic) ---------
    try:
        verts_gt, faces_gt = load_asteroid_mesh(model_num, device)
        vmin, vmax = verts_gt.min(0).values, verts_gt.max(0).values
        verts_gt = (verts_gt - (vmin + vmax) / 2) / (vmax - vmin).max()
    except Exception as e:
        print(f"[test] falling back to synthetic shape ({e})")
        verts_gt, faces_gt = synthetic_potato(device)
    gt_mesh = trimesh.Trimesh(verts_gt.cpu().numpy(),
                              faces_gt.cpu().numpy(), process=False)
    v_dat_t, f_dat_t = star_shaped_proxy(verts_gt, faces_gt,
                                         subdivisions=proxy_subdiv)
    print(f"[test] generating data: {f_dat_t.shape[0]} faces, ...")
    v_dat, f_dat = v_dat_t.cpu().numpy(), f_dat_t.cpu().numpy()
    proxy_mesh = trimesh.Trimesh(v_dat, f_dat, process=False)
    v_dat_t = torch.tensor(v_dat, dtype=torch.float32, device=device)
    f_dat_t = torch.tensor(f_dat, dtype=torch.long, device=device)

    # ---- synthetic multi-camera data (21 cameras, your staggered layout) ---
    cols = [f"{az}_{k}" for az in (0, 45, 90, 135, 225, 270, 315)
            for k in ("hor1", "top", "bottom")]
    view_dirs0 = build_view_dirs(cols, device)
    sun_dir0 = torch.tensor([-1.0, 0.0, 0.0], device=device)
    angles = torch.arange(n_frames, device=device, dtype=torch.float32) \
        * (2 * math.pi / n_frames)

    print(f"[test] generating data: {proxy_mesh.faces.shape[0]} faces, "
          f"{n_frames} frames x {len(cols)} cameras ...")
    t0 = time.time()
    with torch.no_grad():
        B = compute_lightcurves_multicam(
            v_dat_t, f_dat_t, sun_dir0, view_dirs0, angles,
            gamma=0.01, eps=1e-3,                # != fit gammas on purpose
            use_compile=False, dir_chunk=2, frame_chunk=60, verbose=True)
    B = B * (1.0 + noise * torch.randn_like(B))
    data = B / B.mean(dim=0, keepdim=True)
    print(f"[test] data generated in {time.time()-t0:.1f}s")

    # ---- fit ----------------------------------------------------------------
    fitter = ShapeFitter(data[::fit_stride], angles[::fit_stride],
                         sun_dir0, view_dirs0, subdiv=subdiv, lam_lap=lam_lap)
    print(f"[test] fitter: {fitter.n_params} params, "
          f"{fitter.n_photo} photometric residuals, "
          f"{fitter.uniq_dirs.shape[0]} unique visibility directions")

    p0 = generate_initial_guess(fitter)
    print(f"[test] jacobian FD check (exact path): "
          f"rel err {fd_check(fitter, p0, 0.03):.2e}")

    t0 = time.time()
    p, hist = solve(fitter, p0)
    print(f"[test] solve time {time.time()-t0:.1f}s")

    from visualize_fit import visualize_fit
    viz = visualize_fit(fitter, p0, p, reference=(v_dat_t, f_dat_t),
                        ref_label=f"star proxy (model {model_num})",
                        out_prefix="gn_fit")
    # optionally: reference=(verts_gt, faces_gt) to include star-approx error
    # return dict(p=p, p0=p0, history=hist, fitter=fitter, report=report,
    #             reference=(v_dat_t, f_dat_t), viz=viz)

    from visualize_fit import evolution_gif
    evolution_gif(fitter, [h["p"] for h in hist if "p" in h],
                  reference=(v_dat_t, f_dat_t), out_gif="gn_fit_evolution.gif")

    # ---- metrics vs. the star-shaped target ---------------------------------
    dirs_np = fitter.dirs_t.cpu().numpy().astype(np.float64)
    r_gt = radial_field(v_dat_t, f_dat_t, fitter.dirs_t).cpu().numpy()
    report = {}
    for name, pp in (("init", p0), ("fit ", p)):
        mesh_m = fitter.shape_model.mesh(pp)
        r_m = radial_field(
            mesh_m.vertices, mesh_m.faces, fitter.dirs_t
        ).cpu().numpy()
        e = np.log(r_m) - np.log(r_gt)
        e -= e.mean()                                  # scale gauge
        report[name.strip()] = (float(np.sqrt((e**2).mean())),
                                float(np.abs(e).max()))
        print(f"[test] {name}: radius RMS err {100*report[name.strip()][0]:.2f}%"
              f"   max {100*report[name.strip()][1]:.2f}%")

    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with torch.no_grad():
            Bf = fitter.curves(p, gamma=0.015)
            model = (Bf / Bf.mean(0, keepdim=True)).cpu().numpy()
        dat = fitter.data.cpu().numpy()
        x = np.degrees(fitter.angles.cpu().numpy())
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, ci in zip(axes, (0, len(cols)//2, len(cols)-1)):
            ax.plot(x, dat[:, ci], "k.", ms=2, label=f"data {cols[ci]}")
            ax.plot(x, model[:, ci], "r-", lw=1.2, label="fit")
            ax.legend(); ax.grid(alpha=0.3)
        axes[-1].set_xlabel("rotation angle [deg]")
        fig.tight_layout(); fig.savefig("gn_fit_curves.png", dpi=140)

        plt.figure(figsize=(6, 4))
        plt.semilogy([h["cost_new"] for h in hist], ".-")
        plt.xlabel("LM iteration"); plt.ylabel("0.5 ||r||^2"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("gn_fit_convergence.png", dpi=140)

        plt.figure(figsize=(5, 5))
        mesh_m = fitter.shape_model.mesh(p)
        r_m = radial_field(
            mesh_m.vertices, mesh_m.faces, fitter.dirs_t
        ).cpu().numpy()
        s = np.exp(np.mean(np.log(r_gt) - np.log(r_m)))
        plt.plot(r_gt, s * r_m, ".", ms=3)
        lim = [r_gt.min()*0.9, r_gt.max()*1.1]
        plt.plot(lim, lim, "k--", lw=1)
        plt.xlabel("ground-truth radius"); plt.ylabel("fitted radius (scale-aligned)")
        plt.tight_layout(); plt.savefig("gn_fit_radii.png", dpi=140)
        print("[test] wrote gn_fit_curves.png / gn_fit_convergence.png / gn_fit_radii.png")

    return dict(p=p, p0=p0, history=hist, fitter=fitter, report=report)


if __name__ == "__main__":
    test_fit(model_num=3)   # falls back to a synthetic potato if data missing
