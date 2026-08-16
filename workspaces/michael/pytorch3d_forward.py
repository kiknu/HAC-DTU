"""Clean PyTorch3D lightcurve model for octantoid inversion.

The renderer deliberately has one image formation path.  Intensity and binary
lightcurves are two reductions of the same grayscale image, matching the
challenge definitions.  A soft reduction is used while fitting; ``exact``
mode quantizes to an eight-bit image and applies the published hard rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_model import OctantoidShapeModel, TriangleMesh
from render_safety import DEFAULT_FACE_LIMIT, enforce_face_limit, estimate_raster_memory

_PYTORCH3D_IMPORT_ERROR: ImportError | None = None
try:
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.renderer import (
        BlendParams,
        FoVOrthographicCameras,
        FoVPerspectiveCameras,
        MeshRasterizer,
        RasterizationSettings,
        look_at_view_transform,
    )
    from pytorch3d.renderer.blending import softmax_rgb_blend
    from pytorch3d.structures import Meshes
except ImportError as exc:  # Keep CPU-only extraction helpers importable.
    _PYTORCH3D_IMPORT_ERROR = exc


CAMERA_ALPHA_DEG = {
    "0_hor1": 0.0, "0_hor2": 0.0, "0_top": 21.0, "0_bottom": -21.0,
    "45_hor1": 0.0, "45_hor2": 0.0, "45_top": 26.0, "45_bottom": -24.0,
    "90_hor1": 0.0, "90_hor2": 0.0, "90_top": 26.0, "90_bottom": -24.0,
    "135_hor1": 0.0, "135_hor2": 0.0, "135_top": 26.0, "135_bottom": -24.0,
    "225_hor1": 0.0, "225_hor2": 0.0, "225_top": 24.0, "225_bottom": -26.0,
    "270_hor1": 0.0, "270_hor2": 0.0, "270_top": 24.0, "270_bottom": -26.0,
    "315_hor1": 0.0, "315_hor2": 0.0, "315_top": 24.0, "315_bottom": -26.0,
}
CHALLENGE_COLUMNS = tuple(CAMERA_ALPHA_DEG)
UNIQUE_COLUMNS = tuple(name for name in CHALLENGE_COLUMNS if not name.endswith("hor2"))


def challenge_view_directions(
    columns: Sequence[str], *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Return object-to-observer unit directions for challenge column names."""
    values = []
    for name in columns:
        if name not in CAMERA_ALPHA_DEG:
            raise ValueError(f"unknown challenge camera column {name!r}")
        azimuth, _ = name.split("_", 1)
        phi = math.radians(float(azimuth) - 180.0)
        alpha = math.radians(CAMERA_ALPHA_DEG[name])
        values.append(
            [math.cos(phi) * math.cos(alpha),
             math.sin(phi) * math.cos(alpha), math.sin(alpha)]
        )
    return F.normalize(torch.tensor(values, device=device, dtype=dtype), dim=-1)


@dataclass(frozen=True)
class ObservationBatch:
    """Rotation grid and fixed world-frame illumination/camera geometry."""

    angles: torch.Tensor
    view_directions: torch.Tensor
    sun_direction: torch.Tensor
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.angles.ndim != 1:
            raise ValueError("angles must have shape (N,)")
        if self.view_directions.ndim != 2 or self.view_directions.shape[1] != 3:
            raise ValueError("view_directions must have shape (C, 3)")
        if self.sun_direction.shape != (3,):
            raise ValueError("sun_direction must have shape (3,)")
        if self.columns and len(self.columns) != self.view_directions.shape[0]:
            raise ValueError("columns must match the camera count")

    @classmethod
    def challenge(
        cls,
        angles: torch.Tensor,
        columns: Sequence[str] = CHALLENGE_COLUMNS,
    ) -> "ObservationBatch":
        directions = challenge_view_directions(
            columns, device=angles.device, dtype=angles.dtype
        )
        return cls(
            angles=angles,
            view_directions=directions,
            sun_direction=angles.new_tensor([-1.0, 0.0, 0.0]),
            columns=tuple(columns),
        )


@dataclass(frozen=True)
class RendererConfig:
    image_size: int = 96
    shadow_size: int | None = None
    faces_per_pixel: int = 4
    # A non-zero blur makes the rasterizer include near-miss faces whose
    # barycentric backward divides by a near-zero projected area (edge-on
    # faces), producing unbounded gradients at aligned geometries.  The
    # softmax blend already smooths edges, so exact rasterization is used.
    blur_radius: float = 0.0
    framing_scale: float = 1.20
    camera_distance_radii: float = 5.0
    light_distance_radii: float = 4.0
    lambertian_fraction: float = 0.5
    shadow_bias_radii: float = 2e-3
    shadow_softness_radii: float = 2e-3
    intensity_slope: float = 96.0
    binary_slope: float = 96.0
    batch_size: int = 2
    face_limit: int = DEFAULT_FACE_LIMIT
    allow_large_mesh: bool = False
    max_raster_memory_mib: float = 512.0
    require_no_grad: bool = False
    max_frames: int | None = None
    max_cameras: int | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.faces_per_pixel <= 0 or self.batch_size <= 0:
            raise ValueError("image and batch sizes must be positive")
        if self.face_limit <= 0 or self.max_raster_memory_mib <= 0:
            raise ValueError("face and raster-memory limits must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when set")
        if self.max_cameras is not None and self.max_cameras <= 0:
            raise ValueError("max_cameras must be positive when set")
        if self.camera_distance_radii <= 1.0:
            raise ValueError("camera_distance_radii must exceed one")
        if not 0.0 <= self.lambertian_fraction <= 1.0:
            raise ValueError("lambertian_fraction must lie in [0, 1]")

    @classmethod
    def smoke(cls) -> "RendererConfig":
        """32x32, one image per chunk, and an 80-face no-grad ceiling."""
        return cls(
            image_size=32, shadow_size=32, faces_per_pixel=1, batch_size=1,
            face_limit=80, require_no_grad=True, max_frames=1, max_cameras=1,
        )

    @classmethod
    def gradient_test(cls) -> "RendererConfig":
        """64x64, batch one, four layers, and at most 320 faces."""
        return cls(
            image_size=64, shadow_size=64, faces_per_pixel=4, batch_size=1,
            face_limit=320, max_frames=2, max_cameras=1,
        )

    @classmethod
    def optimization(cls) -> "RendererConfig":
        """96x96, batch two, four layers, and a 1,280-face proxy."""
        return cls(
            image_size=96, shadow_size=96, faces_per_pixel=4, batch_size=2,
            face_limit=1_280,
        )

    @classmethod
    def evaluation(cls) -> "RendererConfig":
        """256x256 no-grad evaluation in single-image chunks."""
        return cls(
            image_size=256, shadow_size=256, faces_per_pixel=4, batch_size=1,
            face_limit=1_280, require_no_grad=True,
        )


@dataclass(frozen=True)
class LightcurvePrediction:
    intensity: torch.Tensor
    binary: torch.Tensor
    raw_intensity: torch.Tensor
    raw_binary: torch.Tensor
    images: torch.Tensor | None = None


def _normalize_curves(values: torch.Tensor) -> torch.Tensor:
    return values / values.mean(dim=0, keepdim=True).clamp_min(1e-9)


def _soft_gate(values: torch.Tensor, threshold: torch.Tensor, slope: float) -> torch.Tensor:
    """A sigmoid gate which is exactly zero for a black background."""
    soft = torch.sigmoid(slope * (values - threshold))
    background = torch.sigmoid(-slope * threshold)
    return ((soft - background) / (1.0 - background).clamp_min(1e-8)).clamp(0.0, 1.0)


@torch.no_grad()
def otsu_threshold_uint8(images: torch.Tensor) -> torch.Tensor:
    """Per-image Otsu threshold for grayscale images in [0, 1]."""
    if images.ndim != 3:
        raise ValueError("images must have shape (B, H, W)")
    quantized = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.long)
    thresholds = []
    bins = torch.arange(256, device=images.device, dtype=torch.float64)
    for image in quantized:
        hist = torch.bincount(image.reshape(-1), minlength=256).to(torch.float64)
        weight0 = torch.cumsum(hist, dim=0)
        weight1 = hist.sum() - weight0
        sum0 = torch.cumsum(hist * bins, dim=0)
        total = sum0[-1]
        mean0 = sum0 / weight0.clamp_min(1.0)
        mean1 = (total - sum0) / weight1.clamp_min(1.0)
        variance = weight0 * weight1 * (mean0 - mean1).square()
        variance[(weight0 == 0) | (weight1 == 0)] = -1.0
        thresholds.append(torch.argmax(variance).to(images.dtype) / 255.0)
    return torch.stack(thresholds)


def extract_lightcurves(
    images: torch.Tensor,
    n_frames: int,
    n_cameras: int,
    *,
    mode: Literal["soft", "exact"],
    intensity_slope: float = 96.0,
    binary_slope: float = 96.0,
) -> LightcurvePrediction:
    """Reduce grayscale images to both challenge observables."""
    if images.shape[0] != n_frames * n_cameras:
        raise ValueError("image batch does not match frame/camera dimensions")
    gray = images.reshape(n_frames * n_cameras, *images.shape[-2:]).clamp(0.0, 1.0)
    otsu = otsu_threshold_uint8(gray).to(gray).view(-1, 1, 1)
    if mode == "exact":
        quantized = (gray * 255.0).round() / 255.0
        raw_intensity = torch.where(quantized > (25.5 / 255.0), quantized, 0.0).sum((1, 2))
        raw_binary = (quantized > otsu).to(gray.dtype).sum((1, 2))
    elif mode == "soft":
        intensity_gate = _soft_gate(
            gray, gray.new_tensor(25.5 / 255.0), intensity_slope
        )
        binary_gate = _soft_gate(gray, otsu, binary_slope)
        raw_intensity = (gray * intensity_gate).sum((1, 2))
        raw_binary = binary_gate.sum((1, 2))
    else:
        raise ValueError("mode must be 'soft' or 'exact'")
    raw_intensity = raw_intensity.reshape(n_frames, n_cameras)
    raw_binary = raw_binary.reshape(n_frames, n_cameras)
    return LightcurvePrediction(
        intensity=_normalize_curves(raw_intensity),
        binary=_normalize_curves(raw_binary),
        raw_intensity=raw_intensity,
        raw_binary=raw_binary,
    )


def _rotation_matrices_z(angles: torch.Tensor) -> torch.Tensor:
    c, s = angles.cos(), angles.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [c, -s, zero, s, c, zero, zero, zero, one], dim=-1
    ).reshape(-1, 3, 3)


class PyTorch3DLightcurveModel(nn.Module):
    """Render one mesh into joint intensity and binary lightcurves."""

    def __init__(self, config: RendererConfig | None = None):
        super().__init__()
        if _PYTORCH3D_IMPORT_ERROR is not None:
            raise ImportError(
                "rendering requires PyTorch3D; install it against the pinned "
                "Torch 2.6 cu124 runtime"
            ) from _PYTORCH3D_IMPORT_ERROR
        self.config = config or RendererConfig()

    def _meshes_for_frames(
        self, mesh: TriangleMesh, angles: torch.Tensor, repeats: int
    ) -> tuple[Meshes, torch.Tensor]:
        # Rotate the mesh by -angle to match the CPU forward model, which
        # rotates the sun and observer directions by +angle (rotation_sign=-1
        # in ObservationGeometry).  Rotating observers by +theta is
        # equivalent to rotating the body by -theta.
        rotations = _rotation_matrices_z(-angles)
        rotated = torch.einsum("nij,vj->nvi", rotations, mesh.vertices)
        expanded = rotated[:, None].expand(-1, repeats, -1, -1).reshape(
            -1, rotated.shape[1], 3
        )
        faces = mesh.faces.unsqueeze(0).expand(expanded.shape[0], -1, -1)
        return Meshes(verts=expanded, faces=faces), rotated

    def _observer_cameras(
        self, directions: torch.Tensor, radius: torch.Tensor, distance: torch.Tensor
    ) -> FoVPerspectiveCameras:
        eye = directions * (radius * distance)
        at = torch.zeros_like(eye)
        up = eye.new_tensor([0.0, 0.0, 1.0]).expand_as(eye)
        R, T = look_at_view_transform(eye=eye, at=at, up=up)
        fov = 2.0 * torch.atan(
            eye.new_tensor(self.config.framing_scale) / distance
        ) * (180.0 / math.pi)
        return FoVPerspectiveCameras(
            device=eye.device, R=R, T=T, fov=fov.expand(eye.shape[0]),
            znear=max(1e-4, float(radius.detach()) * 0.01),
            zfar=float(radius.detach()) * (float(distance.detach()) + 2.0),
        )

    def _light_cameras(
        self, sun: torch.Tensor, n_frames: int, radius: torch.Tensor
    ) -> FoVOrthographicCameras:
        directions = F.normalize(sun, dim=-1).expand(n_frames, -1)
        eye = directions * radius * self.config.light_distance_radii
        at = torch.zeros_like(eye)
        up = eye.new_tensor([0.0, 0.0, 1.0]).expand_as(eye)
        R, T = look_at_view_transform(eye=eye, at=at, up=up)
        extent = float(radius.detach()) * self.config.framing_scale
        return FoVOrthographicCameras(
            device=eye.device, R=R, T=T,
            min_x=-extent, max_x=extent, min_y=-extent, max_y=extent,
            znear=1e-4,
            zfar=float(radius.detach()) * (self.config.light_distance_radii + 2.0),
        )

    def _shadow_maps(
        self, frame_meshes: Meshes, cameras: FoVOrthographicCameras
    ) -> torch.Tensor:
        size = self.config.shadow_size or self.config.image_size
        settings = RasterizationSettings(
            image_size=size, blur_radius=0.0, faces_per_pixel=1,
        )
        fragments = MeshRasterizer(cameras=cameras, raster_settings=settings)(frame_meshes)
        return fragments.zbuf[..., 0]

    def _sample_shadows(
        self,
        positions: torch.Tensor,
        light_cameras: FoVOrthographicCameras,
        shadow_depth: torch.Tensor,
        frame_indices: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        batch, height, width, layers, _ = positions.shape
        cameras = light_cameras[frame_indices]
        points = positions.reshape(batch, -1, 3)
        ndc = cameras.transform_points_ndc(points)
        view = cameras.get_world_to_view_transform().transform_points(points)
        # PyTorch3D NDC has +x left and +y up; grid_sample has +x right,
        # +y down, hence both signs are reversed.
        grid = -ndc[..., :2].reshape(batch, height, width * layers, 2)
        depths = shadow_depth[frame_indices].unsqueeze(1)
        sampled = F.grid_sample(
            depths, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[:, 0].reshape(batch, height, width, layers)
        valid = F.grid_sample(
            (depths > 0).to(depths), grid, mode="nearest",
            padding_mode="zeros", align_corners=False,
        )[:, 0].reshape(batch, height, width, layers)
        point_depth = view[..., 2].reshape(batch, height, width, layers)
        bias = radius * self.config.shadow_bias_radii
        softness = (radius * self.config.shadow_softness_radii).clamp_min(1e-8)
        lit = torch.sigmoid((sampled + bias - point_depth) / softness)
        return torch.where(valid > 0.5, lit, torch.ones_like(lit))

    def _render_chunk(
        self,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        camera_eyes: torch.Tensor,
        sun: torch.Tensor,
        light_cameras: FoVOrthographicCameras,
        shadow_depth: torch.Tensor,
        frame_indices: torch.Tensor,
        radius: torch.Tensor,
        lambertian_fraction: torch.Tensor,
    ) -> torch.Tensor:
        settings = RasterizationSettings(
            image_size=self.config.image_size,
            blur_radius=self.config.blur_radius,
            faces_per_pixel=self.config.faces_per_pixel,
        )
        fragments = MeshRasterizer(cameras=cameras, raster_settings=settings)(meshes)
        faces = meshes.faces_packed()
        position_attributes = meshes.verts_packed()[faces]
        normal_attributes = meshes.verts_normals_packed()[faces]
        positions = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, position_attributes
        )
        normals = F.normalize(
            interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, normal_attributes
            ), dim=-1,
        )
        view = F.normalize(camera_eyes[:, None, None, None, :] - positions, dim=-1)
        sun_dir = F.normalize(sun, dim=-1).view(1, 1, 1, 1, 3)
        mu = (normals * view).sum(dim=-1).clamp_min(0.0)
        mu0 = (normals * sun_dir).sum(dim=-1).clamp_min(0.0)
        lambert = mu0
        # Clamp the L-S denominator: at silhouette pixels mu0 and mu both
        # approach zero, and the 1e-8 divisor makes autograd derivatives
        # unbounded.  Pixels with mu0 + mu < 1e-3 have radiance below the
        # 0.1 intensity gate, so the lightcurves are unchanged.
        lommel = mu0 / (mu0 + mu).clamp_min(1e-3)
        radiance = lambertian_fraction * lambert + (1.0 - lambertian_fraction) * lommel
        radiance = radiance * self._sample_shadows(
            positions, light_cameras, shadow_depth, frame_indices, radius
        )
        colors = radiance[..., None].expand(-1, -1, -1, -1, 3)
        rgba = softmax_rgb_blend(
            colors, fragments,
            BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0)),
            znear=cameras.get_znear(), zfar=getattr(cameras, "zfar", None),
        )
        return rgba[..., :3].mean(dim=-1)

    def forward(
        self,
        mesh: TriangleMesh,
        observations: ObservationBatch,
        *,
        extraction: Literal["soft", "exact"] = "soft",
        camera_distance_radii: torch.Tensor | float | None = None,
        phase_offset: torch.Tensor | float = 0.0,
        lambertian_fraction: torch.Tensor | float | None = None,
        return_images: bool = False,
    ) -> LightcurvePrediction:
        vertices = mesh.vertices
        if self.config.require_no_grad and torch.is_grad_enabled():
            raise RuntimeError("this renderer stage must run inside torch.no_grad()")
        enforce_face_limit(
            int(mesh.faces.shape[0]),
            limit=self.config.face_limit,
            allow_large_mesh=self.config.allow_large_mesh,
            context="differentiable input mesh",
        )
        estimate = estimate_raster_memory(
            self.config.batch_size,
            self.config.image_size,
            self.config.faces_per_pixel,
            requires_grad=torch.is_grad_enabled() and vertices.requires_grad,
        )
        if estimate.mebibytes > self.config.max_raster_memory_mib:
            raise RuntimeError(
                f"estimated raster memory is {estimate.mebibytes:.1f} MiB per chunk, "
                f"above the {self.config.max_raster_memory_mib:.1f} MiB limit"
            )
        dtype, device = vertices.dtype, vertices.device
        distance = torch.as_tensor(
            self.config.camera_distance_radii if camera_distance_radii is None
            else camera_distance_radii, dtype=dtype, device=device,
        )
        mixture = torch.as_tensor(
            self.config.lambertian_fraction if lambertian_fraction is None
            else lambertian_fraction, dtype=dtype, device=device,
        ).clamp(0.0, 1.0)
        phase = torch.as_tensor(phase_offset, dtype=dtype, device=device)
        angles = observations.angles.to(device=device, dtype=dtype) + phase
        views = F.normalize(observations.view_directions.to(device=device, dtype=dtype), dim=-1)
        sun = F.normalize(observations.sun_direction.to(device=device, dtype=dtype), dim=-1)
        n_frames, n_cameras = angles.numel(), views.shape[0]
        if self.config.max_frames is not None and n_frames > self.config.max_frames:
            raise ValueError(
                f"renderer stage permits at most {self.config.max_frames} frame(s)"
            )
        if self.config.max_cameras is not None and n_cameras > self.config.max_cameras:
            raise ValueError(
                f"renderer stage permits at most {self.config.max_cameras} camera(s)"
            )
        radius = torch.linalg.vector_norm(vertices, dim=-1).max().clamp_min(1e-6)

        _, rotated = self._meshes_for_frames(mesh, angles, 1)
        frame_meshes = Meshes(
            verts=rotated, faces=mesh.faces.unsqueeze(0).expand(n_frames, -1, -1)
        )
        light_cameras = self._light_cameras(sun, n_frames, radius)
        shadow_depth = self._shadow_maps(frame_meshes, light_cameras)

        all_meshes, _ = self._meshes_for_frames(mesh, angles, n_cameras)
        all_views = views.unsqueeze(0).expand(n_frames, -1, -1).reshape(-1, 3)
        camera_eyes = all_views * radius * distance
        all_cameras = self._observer_cameras(all_views, radius, distance)
        frame_indices = torch.arange(n_frames, device=device).repeat_interleave(n_cameras)

        rendered = []
        for start in range(0, n_frames * n_cameras, self.config.batch_size):
            stop = min(start + self.config.batch_size, n_frames * n_cameras)
            rendered.append(
                self._render_chunk(
                    all_meshes[start:stop],
                    all_cameras[list(range(start, stop))],
                    camera_eyes[start:stop], sun, light_cameras, shadow_depth,
                    frame_indices[start:stop], radius, mixture,
                )
            )
        images = torch.cat(rendered, dim=0)
        prediction = extract_lightcurves(
            images, n_frames, n_cameras, mode=extraction,
            intensity_slope=self.config.intensity_slope,
            binary_slope=self.config.binary_slope,
        )
        return LightcurvePrediction(
            intensity=prediction.intensity,
            binary=prediction.binary,
            raw_intensity=prediction.raw_intensity,
            raw_binary=prediction.raw_binary,
            images=images.reshape(n_frames, n_cameras, *images.shape[-2:])
            if return_images else None,
        )


class OctantoidLightcurveModel(nn.Module):
    """Single public model mapping octantoid coefficients to both curves."""

    def __init__(
        self,
        shape: OctantoidShapeModel,
        renderer: PyTorch3DLightcurveModel | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.renderer = renderer or PyTorch3DLightcurveModel()

    def forward(
        self,
        parameters: torch.Tensor,
        observations: ObservationBatch,
        **kwargs,
    ) -> LightcurvePrediction:
        return self.renderer(self.shape.mesh(parameters), observations, **kwargs)


def make_octantoid_model(
    directions: torch.Tensor,
    faces: torch.Tensor,
    *,
    lmax: int = 10,
    r0: float = 0.5,
    config: RendererConfig | None = None,
) -> OctantoidLightcurveModel:
    shape = OctantoidShapeModel(
        directions, faces, lmax=lmax, r0=r0,
        lam_shape=0.3, degree_power=2.0,
    )
    return OctantoidLightcurveModel(shape, PyTorch3DLightcurveModel(config))
