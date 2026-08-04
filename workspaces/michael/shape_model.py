#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def real_spherical_harmonics(
    dirs: torch.Tensor,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Real orthonormal spherical harmonics evaluated on unit directions.

    Column ordering:
        l = 0, ..., lmax
        m = -l, ..., +l

    Convention:
        m < 0: sine mode
        m = 0: zonal mode
        m > 0: cosine mode

    Returns
    -------
    Y:
        (V, (lmax + 1)^2) basis matrix.
    degree:
        (K,) spherical-harmonic degree l for each column.
    order:
        (K,) order m for each column.

    The basis is fixed, so gradients are only needed with respect to
    coefficients, not directions.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")

    u = F.normalize(dirs, dim=-1)
    cos_theta = u[:, 2].clamp(-1.0, 1.0)
    sin_theta = torch.sqrt(
        (1.0 - cos_theta.square()).clamp_min(0.0)
    )
    phi = torch.atan2(u[:, 1], u[:, 0])

    # Associated Legendre polynomials including the
    # Condon-Shortley phase.
    P = [
        [None for _ in range(lmax + 1)]
        for _ in range(lmax + 1)
    ]
    P[0][0] = torch.ones_like(cos_theta)

    # P_m^m
    for m in range(1, lmax + 1):
        P[m][m] = (
            -(2 * m - 1)
            * sin_theta
            * P[m - 1][m - 1]
        )

    # P_{m+1}^m
    for m in range(lmax):
        P[m + 1][m] = (
            (2 * m + 1)
            * cos_theta
            * P[m][m]
        )

    # Upward recurrence in l.
    for m in range(lmax + 1):
        for l in range(m + 2, lmax + 1):
            P[l][m] = (
                (2 * l - 1) * cos_theta * P[l - 1][m]
                - (l + m - 1) * P[l - 2][m]
            ) / (l - m)

    columns = []
    degrees = []
    orders = []

    sqrt2 = math.sqrt(2.0)

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            mm = abs(m)

            log_norm = 0.5 * (
                math.log(2 * l + 1)
                - math.log(4.0 * math.pi)
                + math.lgamma(l - mm + 1)
                - math.lgamma(l + mm + 1)
            )
            norm = math.exp(log_norm)

            if m < 0:
                ylm = (
                    sqrt2
                    * norm
                    * P[l][mm]
                    * torch.sin(mm * phi)
                )
            elif m == 0:
                ylm = norm * P[l][0]
            else:
                ylm = (
                    sqrt2
                    * norm
                    * P[l][m]
                    * torch.cos(m * phi)
                )

            columns.append(ylm)
            degrees.append(l)
            orders.append(m)

    Y = torch.stack(columns, dim=1)

    degree = torch.tensor(
        degrees, dtype=torch.long, device=dirs.device
    )
    order = torch.tensor(
        orders, dtype=torch.long, device=dirs.device
    )

    return Y, degree, order


class OctantoidShape:
    """
    Differentiable octantoid on a fixed spherical triangulation.

    The canonical representation is

        x = r0 * ux * exp(a)
        y = r0 * uy * exp(a + b)
        z = r0 * uz * exp(a + c),

    with a, b and c represented using real spherical harmonics.

    The constant coefficient of `a` is omitted because it is an
    unobservable global-scale mode for mean-normalized lightcurves.

    An optional Cartesian SH offset field can be enabled for strongly
    irregular bodies. Its constant terms are omitted to avoid adding
    an unobservable global translation.
    """

    def __init__(
        self,
        dirs: torch.Tensor,
        faces: torch.Tensor,
        *,
        lmax: int = 10,
        r0: float = 0.5,
        detail_lmax: Optional[int] = None,
        detail_scale: float = 0.20,
        max_log_scale: Optional[float] = 3.0,
    ):
        self.dirs = F.normalize(dirs, dim=-1)
        self.faces = faces
        self.r0 = float(r0)

        self.lmax = int(lmax)
        self.Y, self.degree, self.order = (
            real_spherical_harmonics(self.dirs, self.lmax)
        )
        self.K = self.Y.shape[1]

        # a has no l=0 coefficient; b and c have all coefficients.
        self.n_base_params = (self.K - 1) + self.K + self.K

        self.detail_lmax = detail_lmax
        self.detail_scale = float(detail_scale)
        self.max_log_scale = max_log_scale

        if detail_lmax is None:
            self.Y_detail = None
            self.degree_detail = None
            self.K_detail = 0
            self.n_detail_params = 0
        else:
            (
                self.Y_detail,
                self.degree_detail,
                _,
            ) = real_spherical_harmonics(
                self.dirs, int(detail_lmax)
            )
            self.K_detail = self.Y_detail.shape[1]

            # Three Cartesian fields, without constant translation.
            self.n_detail_params = 3 * (self.K_detail - 1)

        self.n_params = (
            self.n_base_params + self.n_detail_params
        )

    def zero_parameters(self) -> torch.Tensor:
        return self.dirs.new_zeros(self.n_params)

    def unpack(
        self,
        p: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        if p.numel() != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, "
                f"received {p.numel()}."
            )

        q = 0

        a_free = p[q : q + self.K - 1]
        q += self.K - 1

        # Omit global scale a_00.
        ca = torch.cat([p.new_zeros(1), a_free])

        cb = p[q : q + self.K]
        q += self.K

        cc = p[q : q + self.K]
        q += self.K

        if self.n_detail_params:
            # Shape: (xyz, nonconstant SH coefficients).
            detail = p[q:].reshape(3, self.K_detail - 1)
        else:
            detail = None

        return ca, cb, cc, detail

    def fields(
        self,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ca, cb, cc, _ = self.unpack(p)
        return self.Y @ ca, self.Y @ cb, self.Y @ cc

    def vertices(self, p: torch.Tensor) -> torch.Tensor:
        ca, cb, cc, detail = self.unpack(p)

        a = self.Y @ ca
        b = self.Y @ cb
        c = self.Y @ cc

        log_coordinate_scales = torch.stack(
            [a, a + b, a + c], dim=-1
        )

        # Smoothly prevent exponent overflow during rejected LM steps.
        if self.max_log_scale is not None:
            limit = float(self.max_log_scale)
            log_coordinate_scales = (
                limit
                * torch.tanh(log_coordinate_scales / limit)
            )

        verts = (
            self.r0
            * self.dirs
            * torch.exp(log_coordinate_scales)
        )

        if detail is not None:
            # Constant Y00 has been removed, so there is no explicit
            # translation parameter.
            raw_offset = self.Y_detail[:, 1:] @ detail.T

            # Bound the optional deformation. Subtracting the discrete
            # mean eliminates any small residual translation.
            offset = torch.tanh(raw_offset)
            offset = offset - offset.mean(dim=0, keepdim=True)

            verts = (
                verts
                + self.r0 * self.detail_scale * offset
            )

        return verts

    def regularization(
        self,
        p: torch.Tensor,
        *,
        lam_shape: float,
        lam_detail: float,
        degree_power: float = 2.0,
    ) -> torch.Tensor:
        """
        Degree-weighted spectral regularization.

        For degree_power=2, residual weights are approximately
        1 + l(l+1), strongly suppressing high-frequency oscillations.
        """
        ca, cb, cc, detail = self.unpack(p)

        degree = self.degree.to(dtype=p.dtype)
        weight = (
            1.0 + degree * (degree + 1.0)
        ).pow(0.5 * degree_power)

        residuals = [
            lam_shape * weight[1:] * ca[1:],
            lam_shape * weight * cb,
            lam_shape * weight * cc,
        ]

        if detail is not None:
            degree_d = self.degree_detail.to(dtype=p.dtype)
            weight_d = (
                1.0 + degree_d * (degree_d + 1.0)
            ).pow(0.5 * degree_power)

            residuals.append(
                (
                    lam_detail
                    * detail
                    * weight_d[None, 1:]
                ).reshape(-1)
            )

        return torch.cat(residuals)

    @torch.no_grad()
    def from_log_radius(
        self,
        log_radius: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize the octantoid from a star-shaped log-radius field.

        Only the `a` field is initialized; b, c and detail start at zero.
        Global scale is removed.
        """
        if log_radius.shape != (self.dirs.shape[0],):
            raise ValueError("log_radius must have shape (V,)")

        target = log_radius - log_radius.mean()

        A = self.Y[:, 1:].double()
        rhs = target.double().unsqueeze(1)

        coeff = torch.linalg.lstsq(A, rhs).solution[:, 0]
        coeff = coeff.to(
            dtype=self.dirs.dtype,
            device=self.dirs.device,
        )

        p = self.zero_parameters()
        p[: self.K - 1] = coeff
        return p
