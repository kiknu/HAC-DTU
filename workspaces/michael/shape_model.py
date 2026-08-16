#!/usr/bin/env python3
"""Parameterized asteroid shapes with a common mesh-producing interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TriangleMesh:
    """A differentiable vertex array with fixed triangular connectivity."""

    vertices: torch.Tensor
    faces: torch.Tensor


@dataclass(frozen=True)
class EllipsoidInitialization:
    """Scale-free triaxial ellipsoid used to initialize a shape model."""

    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    theta: float = 0.0


@dataclass(frozen=True)
class MeshValidation:
    valid: bool
    reason: str = ""


@runtime_checkable
class ShapeModel(Protocol):
    """Contract required by the photometric inverse problem."""

    n_parameters: int
    faces: torch.Tensor

    def initial_parameters(
        self, spec: EllipsoidInitialization | None = None
    ) -> torch.Tensor: ...

    def mesh(self, parameters: torch.Tensor) -> TriangleMesh: ...

    def prior_residuals(self, parameters: torch.Tensor) -> torch.Tensor: ...

    def project_parameters(self, parameters: torch.Tensor) -> torch.Tensor: ...

    def validate_mesh(self, mesh: TriangleMesh) -> MeshValidation: ...


def ellipsoid_log_radius(
    directions: torch.Tensor, spec: EllipsoidInitialization
) -> torch.Tensor:
    """Zero-mean log radius of a z-aligned, in-plane-rotated ellipsoid."""

    ct, st = math.cos(spec.theta), math.sin(spec.theta)
    ux = ct * directions[:, 0] + st * directions[:, 1]
    uy = -st * directions[:, 0] + ct * directions[:, 1]
    uz = directions[:, 2]
    radius = (
        ux.square() / spec.a**2
        + uy.square() / spec.b**2
        + uz.square() / spec.c**2
    ).rsqrt()
    log_radius = torch.log(radius)
    return log_radius - log_radius.mean()


@torch.no_grad()
def _has_transverse_self_intersections(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> bool:
    """Detect non-adjacent transverse triangle intersections.

    Octantoid trial validation only needs a rejection test.  Candidate pairs
    are first reduced with AABBs, then tested by intersecting the three edges
    of each triangle with the other triangle.  Coplanar contacts are ignored;
    they are caught in practice by the area/orientation checks or by a later
    transverse crossing.
    """

    n_faces = faces.shape[0]
    if n_faces < 2:
        return False

    tri = vertices[faces]
    pairs = torch.triu_indices(
        n_faces, n_faces, offset=1, device=vertices.device
    )
    fi, fj = pairs[0], pairs[1]

    # Faces meeting at an edge or vertex are valid neighbours, not crossings.
    left = faces[fi]
    right = faces[fj]
    adjacent = (left[:, :, None] == right[:, None, :]).any(dim=(1, 2))
    fi, fj = fi[~adjacent], fj[~adjacent]
    if fi.numel() == 0:
        return False

    ti, tj = tri[fi], tri[fj]
    overlap = (
        (ti.amax(dim=1) + eps >= tj.amin(dim=1))
        & (tj.amax(dim=1) + eps >= ti.amin(dim=1))
    ).all(dim=1)
    ti, tj = ti[overlap], tj[overlap]
    if ti.shape[0] == 0:
        return False

    def edges_hit_triangle(edges: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # edges: (P, 3, 2, 3), target: (P, 3, 3)
        origin = edges[:, :, 0]
        direction = edges[:, :, 1] - origin
        v0 = target[:, None, 0]
        e1 = (target[:, 1] - target[:, 0])[:, None]
        e2 = (target[:, 2] - target[:, 0])[:, None]
        pvec = torch.linalg.cross(direction, e2, dim=-1)
        det = (e1 * pvec).sum(dim=-1)
        nonparallel = det.abs() > eps
        inv_det = torch.where(nonparallel, det.reciprocal(), torch.zeros_like(det))
        tvec = origin - v0
        u = (tvec * pvec).sum(dim=-1) * inv_det
        qvec = torch.linalg.cross(tvec, e1, dim=-1)
        v = (direction * qvec).sum(dim=-1) * inv_det
        t = (e2 * qvec).sum(dim=-1) * inv_det
        return (
            nonparallel
            & (u > eps)
            & (v > eps)
            & (u + v < 1.0 - eps)
            & (t > eps)
            & (t < 1.0 - eps)
        ).any(dim=1)

    edge_index = vertices.new_tensor(
        [[0, 1], [1, 2], [2, 0]], dtype=torch.long
    )
    edges_i = ti[:, edge_index]
    edges_j = tj[:, edge_index]
    return bool(
        (edges_hit_triangle(edges_i, tj) | edges_hit_triangle(edges_j, ti))
        .any()
        .item()
    )


def validate_triangle_mesh(
    mesh: TriangleMesh,
    *,
    check_self_intersections: bool = False,
    area_epsilon: float = 1e-10,
) -> MeshValidation:
    """Validate numerical and topological invariants of a trial mesh."""

    vertices, faces = mesh.vertices, mesh.faces
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return MeshValidation(False, "vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        return MeshValidation(False, "faces must have shape (F, 3)")
    if not bool(torch.isfinite(vertices).all()):
        return MeshValidation(False, "mesh contains non-finite vertices")

    triangles = vertices[faces]
    normal = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    )
    if bool((normal.norm(dim=-1) <= area_epsilon).any()):
        return MeshValidation(False, "mesh contains degenerate faces")

    if check_self_intersections and _has_transverse_self_intersections(
        vertices, faces
    ):
        return MeshValidation(False, "mesh contains self-intersections")

    return MeshValidation(True)


def real_spherical_harmonics(
    dirs: torch.Tensor,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate real orthonormal spherical harmonics on unit directions.

    Columns are ordered by ``l = 0..lmax`` and ``m = -l..l``.  Negative
    orders are sine modes and positive orders are cosine modes.
    """

    if lmax < 0:
        raise ValueError("lmax must be non-negative")

    u = F.normalize(dirs, dim=-1)
    cos_theta = u[:, 2].clamp(-1.0, 1.0)
    sin_theta = torch.sqrt((1.0 - cos_theta.square()).clamp_min(0.0))
    phi = torch.atan2(u[:, 1], u[:, 0])

    legendre: list[list[torch.Tensor | None]] = [
        [None for _ in range(lmax + 1)] for _ in range(lmax + 1)
    ]
    legendre[0][0] = torch.ones_like(cos_theta)
    for m in range(1, lmax + 1):
        legendre[m][m] = (
            -(2 * m - 1) * sin_theta * legendre[m - 1][m - 1]
        )
    for m in range(lmax):
        legendre[m + 1][m] = (
            (2 * m + 1) * cos_theta * legendre[m][m]
        )
    for m in range(lmax + 1):
        for degree in range(m + 2, lmax + 1):
            legendre[degree][m] = (
                (2 * degree - 1)
                * cos_theta
                * legendre[degree - 1][m]
                - (degree + m - 1) * legendre[degree - 2][m]
            ) / (degree - m)

    columns, degrees, orders = [], [], []
    for degree in range(lmax + 1):
        for order in range(-degree, degree + 1):
            abs_order = abs(order)
            log_norm = 0.5 * (
                math.log(2 * degree + 1)
                - math.log(4.0 * math.pi)
                + math.lgamma(degree - abs_order + 1)
                - math.lgamma(degree + abs_order + 1)
            )
            norm = math.exp(log_norm)
            if order < 0:
                value = (
                    math.sqrt(2.0)
                    * norm
                    * legendre[degree][abs_order]
                    * torch.sin(abs_order * phi)
                )
            elif order == 0:
                value = norm * legendre[degree][0]
            else:
                value = (
                    math.sqrt(2.0)
                    * norm
                    * legendre[degree][order]
                    * torch.cos(order * phi)
                )
            columns.append(value)
            degrees.append(degree)
            orders.append(order)

    return (
        torch.stack(columns, dim=1),
        torch.tensor(degrees, dtype=torch.long, device=dirs.device),
        torch.tensor(orders, dtype=torch.long, device=dirs.device),
    )


class StarShapeModel:
    """Per-vertex log-radius model on a fixed spherical triangulation."""

    def __init__(
        self,
        directions: torch.Tensor,
        faces: torch.Tensor,
        laplacian: torch.Tensor,
        *,
        r0: float = 0.5,
        lam_lap: float = 30.0,
        lam_scale: float = 10.0,
        parameter_limit: float = 2.0,
    ):
        self.directions = F.normalize(directions, dim=-1)
        self.dirs = self.directions  # compatibility with older diagnostics
        self.faces = faces
        self.laplacian = laplacian
        self.r0 = float(r0)
        self.lam_lap = float(lam_lap)
        self.lam_scale = float(lam_scale)
        self.parameter_limit = float(parameter_limit)
        self.n_parameters = self.directions.shape[0]
        self.n_params = self.n_parameters

    def initial_parameters(
        self, spec: EllipsoidInitialization | None = None
    ) -> torch.Tensor:
        if spec is None:
            return self.directions.new_zeros(self.n_parameters)
        return ellipsoid_log_radius(self.directions, spec)

    def mesh(self, parameters: torch.Tensor) -> TriangleMesh:
        if parameters.numel() != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, "
                f"received {parameters.numel()}."
            )
        vertices = (
            self.r0 * torch.exp(parameters)[:, None] * self.directions
        )
        return TriangleMesh(vertices, self.faces)

    def vertices(self, parameters: torch.Tensor) -> torch.Tensor:
        return self.mesh(parameters).vertices

    def prior_residuals(self, parameters: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.lam_lap * (self.laplacian @ parameters),
                self.lam_scale * parameters.mean().reshape(1),
            ]
        )

    def project_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return parameters.clamp(-self.parameter_limit, self.parameter_limit)

    def validate_mesh(self, mesh: TriangleMesh) -> MeshValidation:
        # Radial positive scaling cannot change the topology of the template.
        return validate_triangle_mesh(mesh)


class OctantoidShapeModel:
    """Pure Kaasalainen--Viikinkoski octantoid shape support.

    ``a``, ``b`` and ``c`` are independent real spherical-harmonic fields:

        x = r0 ux exp(a), y = r0 uy exp(a+b), z = r0 uz exp(a+c).

    The constant coefficient of ``a`` is omitted as a scale gauge.  Constant
    ``b`` and ``c`` coefficients are retained and unregularized, so a triaxial
    ellipsoid has zero spectral complexity.
    """

    def __init__(
        self,
        directions: torch.Tensor,
        faces: torch.Tensor,
        *,
        lmax: int | None = 10,
        lmax_a: int | None = None,
        lmax_b: int | None = None,
        lmax_c: int | None = None,
        r0: float = 0.5,
        lam_shape: float = 0.3,
        degree_power: float = 2.0,
        max_log_scale: float | None = 3.0,
        check_self_intersections: bool = True,
    ):
        common = 10 if lmax is None else int(lmax)
        self.lmax_a = common if lmax_a is None else int(lmax_a)
        self.lmax_b = common if lmax_b is None else int(lmax_b)
        self.lmax_c = common if lmax_c is None else int(lmax_c)
        self.lmax = self.lmax_a  # compatibility name

        self.directions = F.normalize(directions, dim=-1)
        self.dirs = self.directions
        self.faces = faces
        self.r0 = float(r0)
        self.lam_shape = float(lam_shape)
        self.degree_power = float(degree_power)
        self.max_log_scale = max_log_scale
        self.check_self_intersections = bool(check_self_intersections)

        self.Y_a, self.degree_a, self.order_a = real_spherical_harmonics(
            self.directions, self.lmax_a
        )
        self.Y_b, self.degree_b, self.order_b = real_spherical_harmonics(
            self.directions, self.lmax_b
        )
        self.Y_c, self.degree_c, self.order_c = real_spherical_harmonics(
            self.directions, self.lmax_c
        )
        self.K_a = self.Y_a.shape[1]
        self.K_b = self.Y_b.shape[1]
        self.K_c = self.Y_c.shape[1]
        self.K = self.K_a
        self.n_parameters = self.K_a - 1 + self.K_b + self.K_c
        self.n_params = self.n_parameters

    def zero_parameters(self) -> torch.Tensor:
        return self.directions.new_zeros(self.n_parameters)

    def unpack(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if parameters.numel() != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, "
                f"received {parameters.numel()}."
            )
        q = 0
        ca = torch.cat(
            [
                parameters.new_zeros(1),
                parameters[q : q + self.K_a - 1],
            ]
        )
        q += self.K_a - 1
        cb = parameters[q : q + self.K_b]
        q += self.K_b
        cc = parameters[q : q + self.K_c]
        return ca, cb, cc

    def fields(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ca, cb, cc = self.unpack(parameters)
        return self.Y_a @ ca, self.Y_b @ cb, self.Y_c @ cc

    def mesh(self, parameters: torch.Tensor) -> TriangleMesh:
        a, b, c = self.fields(parameters)
        log_scales = torch.stack([a, a + b, a + c], dim=-1)
        # Smooth saturation protects rejected LM trials from exp overflow.
        if self.max_log_scale is not None:
            limit = float(self.max_log_scale)
            log_scales = limit * torch.tanh(log_scales / limit)
        vertices = self.r0 * self.directions * torch.exp(log_scales)
        return TriangleMesh(vertices, self.faces)

    def vertices(self, parameters: torch.Tensor) -> torch.Tensor:
        return self.mesh(parameters).vertices

    def _spectral_residuals(
        self,
        parameters: torch.Tensor,
        *,
        lam_shape: float,
        degree_power: float,
    ) -> torch.Tensor:
        ca, cb, cc = self.unpack(parameters)

        def weighted(coeff: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
            # Squaring these residuals yields sum l^degree_power c_lm^2.
            weight = degree.to(dtype=parameters.dtype).pow(
                0.5 * degree_power
            )
            return lam_shape * weight * coeff

        return torch.cat(
            [
                weighted(ca, self.degree_a)[1:],
                weighted(cb, self.degree_b),
                weighted(cc, self.degree_c),
            ]
        )

    def prior_residuals(self, parameters: torch.Tensor) -> torch.Tensor:
        return self._spectral_residuals(
            parameters,
            lam_shape=self.lam_shape,
            degree_power=self.degree_power,
        )

    def regularization(
        self,
        parameters: torch.Tensor,
        *,
        lam_shape: float | None = None,
        lam_detail: float | None = None,
        degree_power: float | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper for the previous prototype API."""

        del lam_detail
        return self._spectral_residuals(
            parameters,
            lam_shape=(
                self.lam_shape if lam_shape is None else float(lam_shape)
            ),
            degree_power=(
                self.degree_power
                if degree_power is None
                else float(degree_power)
            ),
        )

    def project_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return parameters

    def validate_mesh(self, mesh: TriangleMesh) -> MeshValidation:
        return validate_triangle_mesh(
            mesh,
            check_self_intersections=self.check_self_intersections,
        )

    @torch.no_grad()
    def from_log_radius(self, log_radius: torch.Tensor) -> torch.Tensor:
        if log_radius.shape != (self.directions.shape[0],):
            raise ValueError("log_radius must have shape (V,)")
        target = log_radius - log_radius.mean()
        coeff = torch.linalg.lstsq(
            self.Y_a[:, 1:].double(), target.double().unsqueeze(1)
        ).solution[:, 0]
        parameters = self.zero_parameters()
        parameters[: self.K_a - 1] = coeff.to(
            dtype=self.directions.dtype, device=self.directions.device
        )
        return parameters

    def initial_parameters(
        self, spec: EllipsoidInitialization | None = None
    ) -> torch.Tensor:
        if spec is None:
            return self.zero_parameters()
        return self.from_log_radius(ellipsoid_log_radius(self.directions, spec))


# The old public name remains importable while callers migrate to the explicit
# ShapeModel suffix.
OctantoidShape = OctantoidShapeModel
