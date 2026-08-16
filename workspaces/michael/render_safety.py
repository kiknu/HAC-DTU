"""Shared limits for memory-safe differentiable rasterization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_FACE_LIMIT = 5_000
MEBIBYTE = 1024 * 1024


@dataclass(frozen=True)
class RasterMemoryEstimate:
    batch_size: int
    resolution: int
    faces_per_pixel: int
    requires_grad: bool
    bytes: int

    @property
    def mebibytes(self) -> float:
        return self.bytes / MEBIBYTE


def estimate_raster_memory(
    batch_size: int,
    resolution: int,
    faces_per_pixel: int,
    *,
    requires_grad: bool,
) -> RasterMemoryEstimate:
    """Conservatively estimate peak per-chunk raster/interpolation storage.

    The estimate covers PyTorch3D fragment buffers plus this renderer's world
    positions, normals, shading values, and autograd intermediates. It is a
    guardrail rather than a CUDA allocator prediction.
    """
    if min(batch_size, resolution, faces_per_pixel) <= 0:
        raise ValueError("batch size, resolution, and faces per pixel must be positive")
    bytes_per_pixel_layer = 192 if requires_grad else 64
    fixed_bytes_per_pixel = 32
    pixels = batch_size * resolution * resolution
    total = pixels * (
        fixed_bytes_per_pixel + faces_per_pixel * bytes_per_pixel_layer
    )
    return RasterMemoryEstimate(
        batch_size, resolution, faces_per_pixel, requires_grad, total
    )


def enforce_face_limit(
    face_count: int,
    *,
    limit: int = DEFAULT_FACE_LIMIT,
    allow_large_mesh: bool = False,
    context: str = "mesh",
) -> None:
    if face_count < 0 or limit <= 0:
        raise ValueError("face counts and limits must be valid")
    if face_count > limit and not allow_large_mesh:
        raise ValueError(
            f"{context} has {face_count:,} faces; the safety limit is {limit:,}. "
            "Set allow_large_mesh=True only after reviewing raster memory."
        )


def refuse_public_raw_stl(path: str | Path) -> None:
    """Reject public asteroid STL files on differentiable-renderer paths."""
    candidate = Path(path)
    if candidate.suffix.lower() == ".stl" and candidate.stem.lower().startswith("asteroid"):
        raise ValueError(
            "raw public STL meshes are reference data only; build an 80-face "
            "smoke mesh or a 1,280-face optimizer proxy before rendering"
        )
