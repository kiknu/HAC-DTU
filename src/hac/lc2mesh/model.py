"""INR model definitions for lightcurve-driven mesh reconstruction."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w0: float = 30.0,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.w0 = float(w0)
        self.is_first = is_first
        self.is_last = is_last
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = (
            1.0 / self.in_dim
            if self.is_first
            else math.sqrt(6.0 / self.in_dim) / self.w0
        )
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y if self.is_last else torch.sin(self.w0 * y)


class NeRPRadialNetwork(nn.Module):
    """Pure SIREN deformation field driven by raw 3D spatial coordinates."""

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 8,
        first_w0: float = 15.0,
        w0: float = 30.0,
        out_scale: float = 0.35,
        out_dim: int = 1,
        output_activation: str = "tanh",
    ):
        super().__init__()
        self.out_scale = float(out_scale)
        self.out_dim = int(out_dim)
        if output_activation not in {"tanh", "identity"}:
            raise ValueError("output_activation must be 'tanh' or 'identity'")
        self.output_activation = output_activation

        depth = max(3, int(depth))
        layers: list[SirenLayer] = [
            # GaussianPositionalEncoder removed. First layer takes raw coords.
            SirenLayer(in_dim, hidden_dim, w0=first_w0, is_first=True),
        ]
        for _ in range(depth - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, w0=w0))
        layers.append(SirenLayer(hidden_dim, self.out_dim, w0=w0, is_last=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        displacement = self.model(x)
        if self.output_activation == "tanh":
            displacement = torch.tanh(displacement)
        displacement = self.out_scale * displacement
        return displacement.squeeze(-1) if self.out_dim == 1 else displacement


class LightcurveToMesh(nn.Module):
    """
    Coarse+detail SIREN deformation field over a star-shaped prior mesh.
    Relies on explicit loss functions (Laplacian, Edge-length) for regularization
    rather than internal forward-pass smoothing.

    Parameterization notes (defaults chosen to avoid the boxy / range-capped
    artifacts of the legacy configuration):

    - ``input_encoding='direction'``: the SIRENs see unit directions
      ``v/|v|`` instead of raw ellipsoid coordinates. Raw coordinates span
      different ranges per axis (R, R, 1), giving the field an axis-aligned
      anisotropic frequency response — a structural bias toward boxy,
      axis-aligned geometry. Unit directions make the input domain isotropic.
    - ``output_activation='identity'``: no hard tanh cap. A saturating tanh
      produces plateaus of constant maximal displacement with sharp
      transitions (flat, boxy patches) whenever the target needs more than
      ``max_radius_perturb``; with 'identity', ``max_radius_perturb`` is a
      soft initial gain instead of a hard bound.
    - ``deformation_mode='radial'``: multiplicative log-radial map
      ``v -> v * exp(field)``. Unbounded outward growth, radius always
      positive, mesh stays star-shaped (no fold-overs / tangential drift).
      'normal' (displace along supplied vertex normals) and 'vector'
      (free 3D displacement) reproduce the legacy behaviours.
    - Learnable anisotropic global scale ``exp(log_global_scale)`` (3,): lets
      the mesh stretch beyond the prior's cylinder bounds per axis without
      saturating the INR. Axis *ratios* are observable from mean-normalized
      lightcurves; a uniform scale is not (it cancels in the normalization),
      so absolute size must come from the evaluation frame, not the loss.
    - Lower default frequencies (coarse_w0=5, detail_w0=15): biases the field
      toward smooth low-order shapes. Predicted curves are low-pass filtered
      before the loss, so w0=30 mostly added unconstrained high-frequency
      creases.
    """

    def __init__(
        self,
        inr_hidden_dim: int = 256,
        inr_num_layers: int = 8,
        coarse_w0: float = 5.0,  # Low frequency for smooth base structure
        detail_w0: float = 15.0,  # Moderate frequency for surface detail
        max_radius_perturb: float = 0.35,
        use_learnable_temperature: bool = True,
        initial_temperature: float = 1.0,
        deformation_mode: str = "radial",
        output_activation: str = "identity",
        input_encoding: str = "direction",
        use_global_scale: bool = True,
    ):
        super().__init__()
        if deformation_mode not in {"radial", "normal", "vector"}:
            raise ValueError("deformation_mode must be 'radial', 'normal' or 'vector'")
        if input_encoding not in {"direction", "raw"}:
            raise ValueError("input_encoding must be 'direction' or 'raw'")
        self.deformation_mode = deformation_mode
        self.input_encoding = input_encoding
        field_dim = 3 if deformation_mode == "vector" else 1

        coarse_depth = max(4, inr_num_layers - 2)
        self.coarse_deformer = NeRPRadialNetwork(
            in_dim=3,
            hidden_dim=inr_hidden_dim,
            depth=coarse_depth,
            first_w0=coarse_w0,
            w0=coarse_w0,
            out_scale=0.75 * max_radius_perturb,
            out_dim=field_dim,
            output_activation=output_activation,
        )
        self.detail_deformer = NeRPRadialNetwork(
            in_dim=3,
            hidden_dim=inr_hidden_dim,
            depth=inr_num_layers,
            first_w0=detail_w0,
            w0=detail_w0,
            out_scale=0.25 * max_radius_perturb,
            out_dim=field_dim,
            output_activation=output_activation,
        )

        if use_learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(float(initial_temperature)).log()
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(1.0).log()
            )

        # Anisotropic global scale (init 1.0 per axis). Lets the deformed mesh
        # leave the prior's cylinder bounds without saturating the INR field.
        if use_global_scale:
            self.log_global_scale = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("log_global_scale", torch.zeros(3))

    def _apply_input_encoding(self, base_vertices: torch.Tensor) -> torch.Tensor:
        if self.input_encoding == "direction":
            return base_vertices / base_vertices.norm(dim=-1, keepdim=True)
        else:  # raw
            return base_vertices

    def _apply_deformation(self, base_vertices: torch.Tensor, field: torch.Tensor, vertex_normals: torch.Tensor | None) -> torch.Tensor:
        if self.deformation_mode == "radial":
            return base_vertices * torch.exp(field).unsqueeze(-1)
        elif self.deformation_mode == "normal":
            if vertex_normals is None:
                raise ValueError("deformation_mode='normal' requires vertex_normals")
            return base_vertices + vertex_normals * field.unsqueeze(-1)
        else:  # vector
            return base_vertices + field

    def forward(
        self,
        base_vertices: torch.Tensor,
        vertex_normals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base_vertices: (..., N, 3) tensor of prior mesh vertices.
            vertex_normals: (..., N, 3) true vertex normals of the prior mesh.
                Required for ``deformation_mode='normal'``; ignored otherwise.
        Returns:
            vertices: Deformed vertices (..., N, 3).
            residual: The total vector displacement applied (vertices - base).
        """
        x = self._apply_input_encoding(base_vertices)
        field = self.coarse_deformer(x)
        field = torch.exp(self.log_temperature) * field
        vertices = self._apply_deformation(base_vertices, field, vertex_normals)

        field = self.detail_deformer(x)
        field = torch.exp(self.log_temperature) * field
        vertices = self._apply_deformation(vertices, field, vertex_normals)

        vertices = vertices * torch.exp(self.log_global_scale)
        residual = vertices - base_vertices

        return vertices, residual