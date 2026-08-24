"""Optional Mitsuba scalar-CPU reference for forward-model parity checks.

This module deliberately does not participate in optimization.  It fixes the
coordinate and image-response ambiguities in the WIP Jeppe implementation and
provides a small independent direct-light renderer for validation scenes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import trimesh


@dataclass(frozen=True)
class MitsubaReferenceConfig:
    width: int = 128
    height: int = 128
    samples_per_pixel: int = 16
    projection: Literal["orthographic", "perspective"] = "orthographic"
    camera_distance_radii: float = 5.0
    framing_scale: float = 1.2
    reflectance: float = 1.0

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.samples_per_pixel) <= 0:
            raise ValueError("image dimensions and sample count must be positive")
        if self.projection not in ("orthographic", "perspective"):
            raise ValueError("projection must be orthographic or perspective")
        if self.camera_distance_radii <= 1.0 or self.framing_scale <= 1.0:
            raise ValueError("camera distance and framing scale are invalid")
        if not 0.0 <= self.reflectance <= 1.0:
            raise ValueError("reflectance must be in [0, 1]")


def _mitsuba():
    try:
        import mitsuba as mi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Mitsuba reference support is not installed; run "
            "`uv sync --extra reference`"
        ) from exc
    # The scalar variant is portable to Apple Silicon and intentionally avoids
    # making LLVM, Metal, or differentiability part of the reference contract.
    # Set directly instead of querying ``mi.variant()`` first: querying an
    # unset variant can probe LLVM/Metal and emit irrelevant warnings on a
    # scalar-only Apple setup.
    mi.set_variant("scalar_rgb")
    return mi


def prepare_reference_obj(source: Path, cache_dir: Path) -> Path:
    """Normalize a source mesh and export a cached OBJ for Mitsuba.

    Output is written only below the caller's artifact/cache directory, never
    beside challenge input data.
    """

    source = Path(source)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{source.stem}_normalized.obj"
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return target
    mesh = trimesh.load(source, force="mesh", process=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = vertices - center
    radius = np.linalg.norm(vertices, axis=1).max()
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("reference mesh has an invalid radius")
    mesh.vertices = vertices / radius
    target.write_text(trimesh.exchange.obj.export_obj(mesh), encoding="utf-8")
    return target


def make_reference_scene(
    obj_path: Path,
    *,
    rotation_radians: float,
    view_direction: Sequence[float],
    object_to_sun: Sequence[float] = (-1.0, 0.0, 0.0),
    config: MitsubaReferenceConfig = MitsubaReferenceConfig(),
):
    """Construct a direct-light scene matching PyTorch3D conventions.

    ``view_direction`` and ``object_to_sun`` point outwards from the object.
    Mitsuba's directional-emitter direction points along light travel, hence
    the explicit negation below.  The asteroid rotates by ``-phase`` about z,
    matching :class:`PyTorch3DLightcurveModel`.
    """

    config.__post_init__()
    mi = _mitsuba()
    view = np.asarray(view_direction, dtype=np.float64)
    view /= np.linalg.norm(view)
    sun = np.asarray(object_to_sun, dtype=np.float64)
    sun /= np.linalg.norm(sun)
    eye = view * config.camera_distance_radii
    sensor_to_world = mi.ScalarTransform4f.look_at(
        origin=eye.tolist(), target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0]
    )
    sensor: dict = {
        "type": config.projection,
        "to_world": sensor_to_world,
        "sampler": {
            "type": "independent",
            "sample_count": config.samples_per_pixel,
        },
        "film": {
            "type": "hdrfilm",
            "pixel_format": "luminance",
            "component_format": "float32",
            "width": config.width,
            "height": config.height,
            "rfilter": {"type": "box"},
        },
    }
    if config.projection == "orthographic":
        # An orthographic sensor covers [-1, 1]^2 before this scale. The mesh
        # has unit maximum radius, so framing_scale prevents clipping.
        sensor["to_world"] = sensor_to_world @ mi.ScalarTransform4f.scale(
            [config.framing_scale, config.framing_scale, 1.0]
        )
    else:
        sensor["fov"] = math.degrees(
            2.0
            * math.atan(config.framing_scale / config.camera_distance_radii)
        )
    object_transform = mi.ScalarTransform4f.rotate(
        axis=[0.0, 0.0, 1.0], angle=-math.degrees(rotation_radians)
    )
    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "direct"},
            "sensor": sensor,
            "sun": {
                "type": "directional",
                "direction": (-sun).tolist(),
                "irradiance": {"type": "uniform", "value": 1.0},
            },
            "asteroid": {
                "type": "obj",
                "filename": str(obj_path),
                "face_normals": False,
                "to_world": object_transform,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "uniform",
                        "value": config.reflectance,
                    },
                },
            },
        }
    )


def integrate_reference_image(
    image: np.ndarray,
    *,
    response: Literal["linear", "challenge_8bit"] = "linear",
    display_scale: float | None = None,
) -> float:
    """Reduce a Mitsuba HDR image without silently mixing response spaces."""

    values = np.asarray(image, dtype=np.float64).squeeze()
    if values.ndim not in (2, 3) or np.any(~np.isfinite(values)):
        raise ValueError("reference image must be a finite grayscale/RGB array")
    if values.ndim == 3:
        values = values.mean(axis=-1)
    values = np.clip(values, 0.0, None)
    if response == "linear":
        return float(values.sum())
    if response != "challenge_8bit":
        raise ValueError("response must be linear or challenge_8bit")
    if display_scale is None or display_scale <= 0.0:
        raise ValueError(
            "challenge_8bit integration requires an explicit positive "
            "display_scale for HDR-to-display conversion"
        )
    quantized = np.rint(np.clip(values * display_scale, 0.0, 1.0) * 255.0)
    return float(np.where(quantized > 25.5, quantized / 255.0, 0.0).sum())


def render_reference_image(scene) -> np.ndarray:
    mi = _mitsuba()
    return np.asarray(mi.render(scene), dtype=np.float32)


def render_reference_lightcurves(
    obj_path: Path,
    rotation_radians: Sequence[float],
    view_directions: np.ndarray,
    *,
    config: MitsubaReferenceConfig = MitsubaReferenceConfig(),
    response: Literal["linear", "challenge_8bit"] = "linear",
    display_scale: float | None = None,
) -> np.ndarray:
    """Render normalized curves for a mesh, phase grid, and camera matrix."""

    angles = np.asarray(rotation_radians, dtype=np.float64)
    views = np.asarray(view_directions, dtype=np.float64)
    if angles.ndim != 1 or views.ndim != 2 or views.shape[1] != 3:
        raise ValueError("angles must be (N,) and view_directions must be (C, 3)")
    brightness = np.empty((len(angles), len(views)), dtype=np.float64)
    for frame, angle in enumerate(angles):
        for camera, view in enumerate(views):
            scene = make_reference_scene(
                obj_path,
                rotation_radians=float(angle),
                view_direction=view,
                config=config,
            )
            brightness[frame, camera] = integrate_reference_image(
                render_reference_image(scene),
                response=response,
                display_scale=display_scale,
            )
    means = brightness.mean(axis=0, keepdims=True)
    if np.any(~np.isfinite(means)) or np.any(means <= 0.0):
        raise RuntimeError("reference render produced an invalid lightcurve")
    return brightness / means


def mitsuba_smoke_test(cache_dir: Path) -> dict[str, object]:
    """Render a tiny sphere through the scalar direct-light reference path."""

    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sphere_path = cache_dir / "mitsuba_smoke.obj"
        if not sphere_path.is_file():
            sphere = trimesh.creation.icosphere(subdivisions=1)
            sphere_path.write_text(
                trimesh.exchange.obj.export_obj(sphere), encoding="utf-8"
            )
        scene = make_reference_scene(
            sphere_path,
            rotation_radians=0.0,
            view_direction=(-1.0, 0.0, 0.2),
            config=MitsubaReferenceConfig(width=16, height=16, samples_per_pixel=1),
        )
        image = render_reference_image(scene)
        brightness = integrate_reference_image(image)
        if not math.isfinite(brightness) or brightness <= 0.0:
            raise RuntimeError("reference render returned no finite brightness")
        return {"ok": True, "brightness": brightness, "shape": list(image.shape)}
    except Exception as exc:  # pragma: no cover - backend/environment specific
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
