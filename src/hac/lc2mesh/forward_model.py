"""Differentiable asteroid forward photometric model utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from stl import mesh


def normalize_vectors(
    vectors: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize vectors to unit length along ``dim``."""
    return vectors / vectors.norm(dim=dim, keepdim=True).clamp_min(float(eps))


def observer_geometry_to_unit_vector(
    theta_deg: float,
    alpha_deg: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert HAC observer angles to a unit direction vector."""
    theta = torch.deg2rad(
        torch.tensor(float(theta_deg) - 180.0, dtype=dtype, device=device)
    )
    alpha = torch.deg2rad(torch.tensor(float(alpha_deg), dtype=dtype, device=device))

    vec = torch.stack(
        [
            torch.cos(theta) * torch.cos(alpha),
            torch.sin(theta) * torch.cos(alpha),
            torch.sin(alpha),
        ],
        dim=0,
    )
    return normalize_vectors(vec, dim=0)


def observers_to_unit_tensor(
    observer_map: Mapping[str, object],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[list[str], torch.Tensor]:
    """Build ordered observer names and stacked observer vectors."""
    names = list(observer_map.keys())
    e_obs = [
        observer_geometry_to_unit_vector(
            float(observer_map[name].theta),
            float(observer_map[name].alpha),
            device=device,
            dtype=dtype,
        )
        for name in names
    ]
    return names, torch.stack(e_obs, dim=0)


def load_stl_geometry_torch(
    stl_path: str | Path,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Load STL triangles and derived geometric tensors as torch tensors."""
    stl_mesh = mesh.Mesh.from_file(str(stl_path))
    triangles_np = stl_mesh.vectors.astype(np.float64)

    edge1 = triangles_np[:, 1, :] - triangles_np[:, 0, :]
    edge2 = triangles_np[:, 2, :] - triangles_np[:, 0, :]
    cross = np.cross(edge1, edge2)

    twice_area = np.linalg.norm(cross, axis=1)
    normals_np = cross / np.maximum(twice_area[:, None], 1e-12)
    areas_np = 0.5 * twice_area
    centroids_np = triangles_np.mean(axis=1)

    return {
        "triangles": torch.as_tensor(triangles_np, dtype=dtype, device=device),
        "normals": torch.as_tensor(normals_np, dtype=dtype, device=device),
        "areas": torch.as_tensor(areas_np, dtype=dtype, device=device),
        "centroids": torch.as_tensor(centroids_np, dtype=dtype, device=device),
    }


def fit_scale_offset_vectorized(
    model_normalized: torch.Tensor,
    observed: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form per-camera affine calibration coefficients."""
    x = model_normalized
    y = observed

    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    cov = ((x - x_mean) * (y - y_mean)).mean(dim=1)
    var = ((x - x_mean) ** 2).mean(dim=1)

    k = cov / var.clamp_min(float(eps))
    b = y_mean[:, 0] - k * x_mean[:, 0]
    return k, b


def fourier_lowpass_periodic(
    curves: torch.Tensor,
    n_harmonics: int,
) -> torch.Tensor:
    """Periodic Fourier low-pass smoothing of curves ``[..., P]``.

    Keeps the DC component plus the first ``n_harmonics`` harmonics and zeroes
    the rest. Fully differentiable with respect to ``curves``, so it can be used
    inside optimization loops as built-in forward-output smoothing.
    """
    n_phases = int(curves.shape[-1])
    if n_phases < 8:
        return curves

    h = int(max(1, min(int(n_harmonics), n_phases // 2 - 1)))
    coeff = torch.fft.rfft(curves, dim=-1)
    keep = torch.zeros(coeff.shape[-1], dtype=torch.bool, device=curves.device)
    keep[: h + 1] = True
    coeff = torch.where(keep, coeff, torch.zeros_like(coeff))
    return torch.fft.irfft(coeff, n=n_phases, dim=-1)


class AsteroidForwardModel(torch.nn.Module):
    """Differentiable LS+Lambert asteroid forward operator implemented in PyTorch.

    Supports:
    - ``visibility_mode='convex'``: classical ``mu > 0`` and ``mu0 > 0`` visibility.
    - ``visibility_mode='nonconvex'``: approximate self-occlusion via centroid z-buffer
      masks for both observer and sun directions.
    """

    def __init__(
        self,
        normals: torch.Tensor,
        *,
        facet_centroids: torch.Tensor | None = None,
        visibility_mode: str = "convex",
        occlusion_method: str = "zbuffer",
        zbuffer_grid: int = 64,
        zbuffer_depth_tol: float = 1e-4,
        occlusion_stride: int = 8,
        lambert_weight: float = 0.1,
        alpha0: float = float(np.deg2rad(30.0)),
        k_phase: float = 0.02,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        normals_t = normalize_vectors(normals.to(dtype=torch.float64), dim=1, eps=eps)
        self.register_buffer("normals", normals_t)

        self.lambert_weight = float(lambert_weight)
        self.alpha0 = float(alpha0)
        self.k_phase = float(k_phase)
        self.eps = float(eps)

        self.visibility_mode = str(visibility_mode).lower()
        if self.visibility_mode not in {"convex", "nonconvex"}:
            raise ValueError("visibility_mode must be one of {'convex', 'nonconvex'}")

        self.occlusion_method = str(occlusion_method).lower()
        self.zbuffer_grid = int(max(8, zbuffer_grid))
        self.zbuffer_depth_tol = float(max(0.0, zbuffer_depth_tol))
        self.occlusion_stride = int(max(1, occlusion_stride))

        if facet_centroids is not None:
            centroids_t = facet_centroids.to(dtype=torch.float64)
            if centroids_t.shape != (self.n_facets, 3):
                raise ValueError(
                    f"facet_centroids must have shape ({self.n_facets}, 3), got {tuple(centroids_t.shape)}"
                )
            self.register_buffer("centroids", centroids_t)
        else:
            self.centroids = None

        if self.visibility_mode == "nonconvex" and self.centroids is None:
            raise ValueError("facet_centroids must be provided when visibility_mode='nonconvex'")

    @property
    def n_facets(self) -> int:
        return int(self.normals.shape[0])

    @staticmethod
    def rotate_vectors_z(vectors: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """Rotate vectors around +z for all phase angles."""
        c = torch.cos(phases)
        s = torch.sin(phases)

        if vectors.ndim == 1:
            x = c * vectors[0] - s * vectors[1]
            y = s * vectors[0] + c * vectors[1]
            z = torch.ones_like(x) * vectors[2]
            return torch.stack([x, y, z], dim=-1)

        if vectors.ndim == 2:
            x = c[:, None] * vectors[None, :, 0] - s[:, None] * vectors[None, :, 1]
            y = s[:, None] * vectors[None, :, 0] + c[:, None] * vectors[None, :, 1]
            z = vectors[None, :, 2].expand_as(x)
            return torch.stack([x, y, z], dim=-1)

        raise ValueError("vectors must have shape [3] or [N, 3]")

    def _zbuffer_visibility_mask(
        self,
        direction: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate non-convex visibility using projected centroid depth competition."""
        cand = candidate_mask.bool().reshape(-1)
        idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
        if idx.numel() <= 1:
            return cand

        if self.centroids is None:
            return cand

        d = normalize_vectors(direction.reshape(1, 3), dim=1, eps=self.eps)[0]
        if float(torch.abs(d[0]).item()) < 0.9:
            ref = torch.tensor([1.0, 0.0, 0.0], dtype=d.dtype, device=d.device)
        else:
            ref = torch.tensor([0.0, 1.0, 0.0], dtype=d.dtype, device=d.device)

        u = torch.cross(d, ref, dim=0)
        un = torch.linalg.norm(u)
        if float(un.item()) < self.eps:
            ref = torch.tensor([0.0, 0.0, 1.0], dtype=d.dtype, device=d.device)
            u = torch.cross(d, ref, dim=0)
            un = torch.linalg.norm(u)
            if float(un.item()) < self.eps:
                return cand

        u = u / un
        v = torch.cross(d, u, dim=0)

        pts = self.centroids.index_select(0, idx)
        x = pts @ u
        y = pts @ v
        depth = pts @ d

        x_min = torch.min(x)
        x_max = torch.max(x)
        y_min = torch.min(y)
        y_max = torch.max(y)
        x_span = x_max - x_min
        y_span = y_max - y_min

        if float(x_span.item()) < self.eps or float(y_span.item()) < self.eps:
            return cand

        g = self.zbuffer_grid
        ix = torch.clamp(
            ((x - x_min) / x_span.clamp_min(self.eps) * (g - 1)).long(),
            0,
            g - 1,
        )
        iy = torch.clamp(
            ((y - y_min) / y_span.clamp_min(self.eps) * (g - 1)).long(),
            0,
            g - 1,
        )
        cell = iy * g + ix

        max_depth = torch.full(
            (g * g,),
            -torch.inf,
            dtype=depth.dtype,
            device=depth.device,
        )

        if hasattr(max_depth, "scatter_reduce_"):
            max_depth.scatter_reduce_(0, cell, depth, reduce="amax", include_self=True)
        else:
            unique_cell = torch.unique(cell)
            for c in unique_cell:
                mask = cell == c
                max_depth[c] = torch.max(depth[mask])

        depth_span = torch.max(depth) - torch.min(depth)
        tol = self.zbuffer_depth_tol * torch.clamp(depth_span, min=1.0)
        visible_local = depth >= (max_depth[cell] - tol)

        out = torch.zeros_like(cand)
        out[idx] = visible_local
        return out

    def _visibility_mask(
        self,
        direction: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch non-convex visibility approximation."""
        if self.occlusion_method == "zbuffer":
            return self._zbuffer_visibility_mask(direction=direction, candidate_mask=candidate_mask)
        return candidate_mask.bool()

    def _interpolate_visibility_periodic(
        self,
        sample_idx: torch.Tensor,
        sampled_mask: torch.Tensor,
        n_phase: int,
    ) -> torch.Tensor:
        """Periodic interpolation of sampled visibility masks to full phase resolution."""
        if int(sample_idx.numel()) == int(n_phase):
            return sampled_mask.bool()

        x = torch.cat(
            [
                sample_idx.to(dtype=torch.float64),
                sample_idx[:1].to(dtype=torch.float64) + float(n_phase),
            ],
            dim=0,
        )

        vals = torch.cat(
            [sampled_mask.to(dtype=torch.float64), sampled_mask[:, :1, :].to(dtype=torch.float64)],
            dim=1,
        )

        target = torch.arange(n_phase, dtype=torch.float64, device=sample_idx.device)
        right = torch.searchsorted(x, target, right=True)
        right = torch.clamp(right, min=1, max=x.numel() - 1)
        left = right - 1

        x_left = x[left]
        x_right = x[right]
        t = ((target - x_left) / (x_right - x_left).clamp_min(self.eps)).view(1, n_phase, 1)

        vals_left = vals[:, left, :]
        vals_right = vals[:, right, :]
        interp = (1.0 - t) * vals_left + t * vals_right
        return interp >= 0.5

    def build_forward_matrix_components(
        self,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build LS/Lambert component matrices ``(A_ls, A_prod)`` of shape [C, P, F].

        The full matrix for any Lambert weight ``w`` is ``A(w) = A_ls + w * A_prod``;
        visibility masks and phase factors are computed only once, which makes
        Lambert-weight sweeps cheap.
        """
        phases = phases.to(dtype=self.normals.dtype, device=self.normals.device)
        e_sun = normalize_vectors(
            e_sun.to(dtype=self.normals.dtype, device=self.normals.device),
            dim=-1,
            eps=self.eps,
        )
        e_obs = normalize_vectors(
            e_obs.to(dtype=self.normals.dtype, device=self.normals.device),
            dim=-1,
            eps=self.eps,
        )

        e_sun_rot = self.rotate_vectors_z(e_sun, phases)  # [P, 3]
        e_obs_rot = self.rotate_vectors_z(e_obs, phases)  # [P, C, 3]

        mu0 = torch.einsum("pd,fd->pf", e_sun_rot, self.normals)[:, None, :]  # [P,1,F]
        mu = torch.einsum("pcd,fd->pcf", e_obs_rot, self.normals)  # [P,C,F]

        mu0_pos = torch.clamp(mu0, min=0.0)
        mu_pos = torch.clamp(mu, min=0.0)
        prod = mu_pos * mu0_pos

        ls_term = prod / (mu_pos + mu0_pos + self.eps)

        cos_alpha = torch.einsum("pd,pcd->pc", e_sun_rot, e_obs_rot)
        cos_alpha = torch.clamp(cos_alpha, min=-1.0, max=1.0)
        alpha = torch.acos(cos_alpha)
        phase_factor = torch.exp(-alpha / self.alpha0) + self.k_phase * alpha

        if self.visibility_mode == "convex":
            visible = (mu0 > 0.0) & (mu > 0.0)
            a_ls = torch.where(
                visible,
                phase_factor[:, :, None] * ls_term,
                torch.zeros_like(ls_term),
            )
            a_prod = torch.where(
                visible,
                phase_factor[:, :, None] * prod,
                torch.zeros_like(prod),
            )
            return (
                a_ls.permute(1, 0, 2).contiguous(),
                a_prod.permute(1, 0, 2).contiguous(),
            )  # [C,P,F] each

        n_phase = int(phases.numel())
        n_cam = int(e_obs.shape[0])

        visible_obs_convex = mu > 0.0  # [P,C,F]
        visible_sun_convex = mu0 > 0.0  # [P,1,F]

        sample_idx = torch.arange(
            0,
            n_phase,
            self.occlusion_stride,
            dtype=torch.long,
            device=phases.device,
        )
        if int(sample_idx[-1].item()) != (n_phase - 1):
            sample_idx = torch.cat(
                [sample_idx, torch.tensor([n_phase - 1], device=phases.device, dtype=torch.long)]
            )
            sample_idx = torch.unique(sample_idx, sorted=True)

        sampled_obs = torch.zeros(
            (n_cam, int(sample_idx.numel()), self.n_facets),
            dtype=torch.bool,
            device=phases.device,
        )
        sampled_sun = torch.zeros(
            (int(sample_idx.numel()), self.n_facets),
            dtype=torch.bool,
            device=phases.device,
        )

        for s_i, p_i in enumerate(sample_idx.tolist()):
            sampled_sun[s_i] = self._visibility_mask(
                direction=e_sun_rot[p_i],
                candidate_mask=visible_sun_convex[p_i, 0],
            )
            for cam_i in range(n_cam):
                sampled_obs[cam_i, s_i] = self._visibility_mask(
                    direction=e_obs_rot[p_i, cam_i],
                    candidate_mask=visible_obs_convex[p_i, cam_i],
                )

        sampled = sampled_obs & sampled_sun[None, :, :]
        visibility_cam_phase = self._interpolate_visibility_periodic(
            sample_idx=sample_idx,
            sampled_mask=sampled,
            n_phase=n_phase,
        )

        visible = visibility_cam_phase.permute(1, 0, 2)  # [P,C,F]
        a_ls = torch.where(
            visible,
            phase_factor[:, :, None] * ls_term,
            torch.zeros_like(ls_term),
        )
        a_prod = torch.where(
            visible,
            phase_factor[:, :, None] * prod,
            torch.zeros_like(prod),
        )
        return (
            a_ls.permute(1, 0, 2).contiguous(),
            a_prod.permute(1, 0, 2).contiguous(),
        )  # [C,P,F] each

    def build_forward_matrix(
        self,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Build LS+Lambert matrix ``A`` with shape [n_cams, n_phases, n_facets]."""
        a_ls, a_prod = self.build_forward_matrix_components(
            phases=phases, e_sun=e_sun, e_obs=e_obs
        )
        if self.lambert_weight == 0.0:
            return a_ls
        return a_ls + self.lambert_weight * a_prod

    def predict_raw_from_matrix(self, g: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Compute raw predictions ``A g`` from a precomputed matrix ``A``.

        Parameters
        ----------
        g:
            Facet weights with shape ``[n_facets]``.
        A:
            Forward matrix with shape ``[n_cameras, n_phases, n_facets]``.
        """
        g_t = g.to(dtype=self.normals.dtype, device=self.normals.device)
        if g_t.ndim != 1 or g_t.shape[0] != self.n_facets:
            raise ValueError(f"g must have shape [{self.n_facets}], got {tuple(g_t.shape)}")

        a_t = A.to(dtype=self.normals.dtype, device=self.normals.device)
        if a_t.ndim != 3:
            raise ValueError(f"A must have 3 dims [C, P, F], got {tuple(a_t.shape)}")
        if a_t.shape[-1] != self.n_facets:
            raise ValueError(
                f"A last dimension must equal n_facets={self.n_facets}, got {a_t.shape[-1]}"
            )

        return torch.einsum("cpf,f->cp", a_t, g_t)

    def predict_normalized_from_matrix(self, g: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Compute per-camera normalized predictions from precomputed matrix ``A``."""
        raw = self.predict_raw_from_matrix(g=g, A=A)
        return raw / raw.mean(dim=1, keepdim=True).clamp_min(self.eps)

    def evaluate_raw(
        self,
        g: torch.Tensor,
        *,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(A, A g)`` where ``A g`` has shape [n_cams, n_phases]."""
        A = self.build_forward_matrix(phases=phases, e_sun=e_sun, e_obs=e_obs)
        raw = self.predict_raw_from_matrix(g=g, A=A)
        return A, raw

    def forward(
        self,
        g: torch.Tensor,
        *,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-camera normalized predictions from facet weights ``g``."""
        A = self.build_forward_matrix(phases=phases, e_sun=e_sun, e_obs=e_obs)
        return self.predict_normalized_from_matrix(g=g, A=A)
