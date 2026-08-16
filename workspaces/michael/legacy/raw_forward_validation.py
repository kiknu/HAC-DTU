#!/usr/bin/env python3
"""Accuracy-first validation on untouched Gmsh meshes using Embree rays.

This is deliberately separate from the differentiable optimizer renderer.  It
does not normalize, weld, simplify, or otherwise rewrite the supplied mesh.
Hard primary visibility and hard solar shadows scale to asteroid 1's 800k
triangles, unlike the quadratic face-centroid visibility used for fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow running this legacy script from the legacy/ directory while importing
# the current modules in the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from forward_model import CAMERA_ALPHA_DEG, get_model_folder, load_data


GMSH_NODES_PER_ELEMENT = {
    1: 2,
    2: 3,
    3: 4,
    4: 4,
    5: 8,
    6: 6,
    7: 5,
    8: 3,
    9: 6,
    10: 9,
    11: 10,
    12: 27,
    13: 18,
    14: 14,
    15: 1,
    16: 8,
    17: 20,
    18: 15,
    19: 13,
    20: 9,
    21: 10,
    22: 12,
    23: 15,
    24: 15,
    25: 21,
    26: 4,
    27: 5,
    28: 6,
    29: 20,
    30: 35,
    31: 56,
}


@dataclass(frozen=True)
class RawSurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    node_tags: np.ndarray
    source: Path


def _expect_line(stream, expected: bytes) -> None:
    actual = stream.readline().strip()
    if actual != expected:
        raise ValueError(f"expected {expected!r}, found {actual!r}")


def load_gmsh22_binary_surface(path: Path) -> RawSurfaceMesh:
    """Read Gmsh 2.2 binary nodes and first-order surface triangles verbatim."""
    path = Path(path)
    with path.open("rb") as stream:
        _expect_line(stream, b"$MeshFormat")
        version, binary, data_size = stream.readline().decode().split()
        if version != "2.2" or binary != "1" or data_size != "8":
            raise ValueError("only little-endian Gmsh 2.2 binary files are supported")
        if struct.unpack("<i", stream.read(4))[0] != 1:
            raise ValueError("unsupported Gmsh byte order")
        stream.readline()
        _expect_line(stream, b"$EndMeshFormat")
        _expect_line(stream, b"$Nodes")
        n_nodes = int(stream.readline())
        node_dtype = np.dtype([("tag", "<i4"), ("xyz", "<f8", (3,))])
        nodes = np.fromfile(stream, dtype=node_dtype, count=n_nodes)
        stream.readline()
        _expect_line(stream, b"$EndNodes")
        _expect_line(stream, b"$Elements")
        n_elements = int(stream.readline())

        triangle_tags: list[np.ndarray] = []
        remaining = n_elements
        while remaining:
            element_type, block_size, n_tags = struct.unpack("<iii", stream.read(12))
            if element_type not in GMSH_NODES_PER_ELEMENT:
                raise ValueError(f"unsupported Gmsh element type {element_type}")
            record_width = 1 + n_tags + GMSH_NODES_PER_ELEMENT[element_type]
            block = np.fromfile(
                stream, dtype="<i4", count=block_size * record_width
            ).reshape(block_size, record_width)
            if element_type == 2:
                triangle_tags.append(block[:, -3:].copy())
            remaining -= block_size
        stream.readline()
        _expect_line(stream, b"$EndElements")

    if not triangle_tags:
        raise ValueError(f"no first-order surface triangles in {path}")
    tags = nodes["tag"].astype(np.int64, copy=False)
    face_tags = np.concatenate(triangle_tags).astype(np.int64, copy=False)
    order = np.argsort(tags)
    positions = np.searchsorted(tags[order], face_tags)
    if np.any(positions >= len(tags)):
        raise ValueError("element block refers to an unknown node tag")
    if not np.array_equal(tags[order][positions], face_tags):
        raise ValueError("element block refers to an unknown node tag")
    faces = order[positions]
    return RawSurfaceMesh(
        vertices=np.asarray(nodes["xyz"], dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        node_tags=tags,
        source=path,
    )


def get_raw_mesh_path(model_num: int) -> Path:
    matches = sorted(get_model_folder(model_num).glob("*.msh"))
    if len(matches) != 1:
        raise ValueError(f"expected one raw .msh for asteroid {model_num}, got {matches}")
    return matches[0]


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, np.finfo(values.dtype).tiny)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    sums = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(sums, faces[:, corner], cross)
    return _normalize(sums)


def _interpolated_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    face_indices: np.ndarray,
    locations: np.ndarray,
) -> np.ndarray:
    triangles = vertices[faces[face_indices]]
    a = triangles[:, 0]
    e0 = triangles[:, 1] - a
    e1 = triangles[:, 2] - a
    point = locations - a
    d00 = np.einsum("ij,ij->i", e0, e0)
    d01 = np.einsum("ij,ij->i", e0, e1)
    d11 = np.einsum("ij,ij->i", e1, e1)
    d20 = np.einsum("ij,ij->i", point, e0)
    d21 = np.einsum("ij,ij->i", point, e1)
    denominator = d00 * d11 - d01 * d01
    b1 = (d11 * d20 - d01 * d21) / denominator
    b2 = (d00 * d21 - d01 * d20) / denominator
    barycentric = np.column_stack([1.0 - b1 - b2, b1, b2])
    normals = np.einsum(
        "ij,ijk->ik", barycentric, vertex_normals[faces[face_indices]]
    )
    return _normalize(normals)


def _ortho_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = _normalize(np.asarray(direction, dtype=np.float64)[None])[0]
    up = np.array([0.0, 0.0, 1.0])
    if abs(direction[2]) >= 0.9:
        up = np.array([0.0, 1.0, 0.0])
    horizontal = _normalize(np.cross(direction, up)[None])[0]
    vertical = np.cross(direction, horizontal)
    return horizontal, vertical


def _rotate_z(direction: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = direction
    return np.array([cosine * x - sine * y, sine * x + cosine * y, z])


def camera_direction(column: str) -> np.ndarray:
    if column not in CAMERA_ALPHA_DEG:
        raise ValueError(f"unknown camera {column}")
    azimuth = math.radians(float(column.split("_", 1)[0]))
    elevation = math.radians(CAMERA_ALPHA_DEG[column])
    return np.array(
        [
            -math.cos(azimuth) * math.cos(elevation),
            -math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


class EmbreeDirectRenderer:
    """Orthographic direct-light reference with exact first-hit visibility."""

    def __init__(self, mesh: RawSurfaceMesh, resolution: int = 128):
        self.raw = mesh
        self.mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,
            validate=False,
        )
        self.intersector = RayMeshIntersector(self.mesh)
        self.resolution = int(resolution)
        if self.resolution < 16:
            raise ValueError("resolution must be at least 16")
        self.vertex_normals = _vertex_normals(mesh.vertices, mesh.faces)
        self.face_normals = np.asarray(self.mesh.face_normals)
        self.radius = float(np.linalg.norm(mesh.vertices, axis=1).max())
        self.half_extent = 1.25 * self.radius
        self.ray_epsilon = max(1e-8, 2e-5 * self.radius)

    def visible_samples(
        self,
        sun_direction: np.ndarray,
        view_direction: np.ndarray,
        *,
        shading: str = "smooth",
        projection: str = "orthographic",
        camera_distance_radii: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mu0, mu, illuminated)`` for visible camera pixels."""
        sun = _normalize(np.asarray(sun_direction)[None])[0]
        view = _normalize(np.asarray(view_direction)[None])[0]
        horizontal, vertical = _ortho_basis(view)
        step = 2.0 * self.half_extent / self.resolution
        coordinates = (
            np.arange(self.resolution, dtype=np.float64) + 0.5
        ) * step - self.half_extent
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
        plane = xx.reshape(-1, 1) * horizontal + yy.reshape(-1, 1) * vertical
        if projection == "orthographic":
            origins = plane + view * (3.0 * self.radius)
            directions = np.broadcast_to(-view, origins.shape)
        elif projection == "perspective":
            camera_position = view * (camera_distance_radii * self.radius)
            origins = np.broadcast_to(camera_position, plane.shape)
            directions = _normalize(plane - origins)
        else:
            raise ValueError("projection must be 'orthographic' or 'perspective'")
        all_face_indices = self.intersector.intersects_first(origins, directions)
        hit_rays = np.flatnonzero(all_face_indices >= 0)
        face_indices = all_face_indices[hit_rays]
        if not len(face_indices):
            return np.empty(0), np.empty(0), np.empty(0, dtype=bool)
        hit_origins = origins[hit_rays]
        hit_directions = directions[hit_rays]
        plane_origins = self.raw.vertices[self.raw.faces[face_indices, 0]]
        plane_normals = self.face_normals[face_indices]
        distances = np.einsum(
            "ij,ij->i", plane_origins - hit_origins, plane_normals
        ) / np.einsum("ij,ij->i", hit_directions, plane_normals)
        locations = hit_origins + distances[:, None] * hit_directions

        if shading == "smooth":
            normals = _interpolated_normals(
                self.raw.vertices,
                self.raw.faces,
                self.vertex_normals,
                face_indices,
                locations,
            )
        elif shading == "flat":
            normals = self.face_normals[face_indices]
        else:
            raise ValueError("shading must be 'flat' or 'smooth'")

        view_samples = -hit_directions
        mu = np.clip(np.einsum("ij,ij->i", normals, view_samples), 0.0, None)
        mu0 = np.clip(normals @ sun, 0.0, None)
        candidates = (mu > 0.0) & (mu0 > 0.0)
        illuminated = np.zeros(len(face_indices), dtype=bool)
        if candidates.any():
            shadow_origins = locations[candidates] + sun * self.ray_epsilon
            shadow_dirs = np.broadcast_to(sun, shadow_origins.shape)
            illuminated[candidates] = ~self.intersector.intersects_any(
                shadow_origins, shadow_dirs
            )

        return mu0, mu, illuminated

    def render(
        self,
        sun_direction: np.ndarray,
        view_direction: np.ndarray,
        *,
        brdf: str = "lambertian",
        lambertian_fraction: float = 0.5,
        shading: str = "smooth",
        intensity_threshold: float | None = None,
        projection: str = "orthographic",
        camera_distance_radii: float = 5.0,
    ) -> float:
        mu0, mu, illuminated = self.visible_samples(
            sun_direction,
            view_direction,
            shading=shading,
            projection=projection,
            camera_distance_radii=camera_distance_radii,
        )
        if not illuminated.any():
            return 0.0
        mu0 = mu0[illuminated]
        mu = mu[illuminated]

        if brdf == "lambertian":
            radiance = mu0
        elif brdf == "lommel_seeliger":
            radiance = mu0 / np.maximum(mu0 + mu, 1e-12)
        elif brdf == "combined":
            lommel = mu0 / np.maximum(mu0 + mu, 1e-12)
            radiance = lambertian_fraction * mu0 + (
                1.0 - lambertian_fraction
            ) * lommel
        else:
            raise ValueError(f"unknown BRDF {brdf}")
        if intensity_threshold is not None:
            radiance = np.where(radiance > intensity_threshold, radiance, 0.0)
        return float(radiance.sum())

    def render_grid(
        self,
        sun_direction: np.ndarray,
        view_direction: np.ndarray,
        *,
        lambertian_fractions: np.ndarray,
        intensity_thresholds: np.ndarray,
        shading: str = "smooth",
        projection: str = "orthographic",
        camera_distance_radii: float = 5.0,
    ) -> np.ndarray:
        """Evaluate a Lambert/LS and response grid after one ray trace."""
        mu0, mu, illuminated = self.visible_samples(
            sun_direction,
            view_direction,
            shading=shading,
            projection=projection,
            camera_distance_radii=camera_distance_radii,
        )
        fractions = np.asarray(lambertian_fractions, dtype=np.float64)
        thresholds = np.asarray(intensity_thresholds, dtype=np.float64)
        if not illuminated.any():
            return np.zeros((len(fractions), len(thresholds)))
        mu0 = mu0[illuminated]
        mu = mu[illuminated]
        lommel = mu0 / np.maximum(mu0 + mu, 1e-12)
        radiance = fractions[:, None] * mu0 + (1.0 - fractions[:, None]) * lommel
        return np.stack(
            [np.where(radiance > threshold, radiance, 0.0).sum(axis=1)
             for threshold in thresholds],
            axis=1,
        )

    def render_hapke_grid(
        self,
        sun_direction: np.ndarray,
        view_direction: np.ndarray,
        *,
        single_scattering_albedos: np.ndarray,
        phase_terms: np.ndarray,
        shading: str = "smooth",
        projection: str = "orthographic",
        camera_distance_radii: float = 5.0,
    ) -> np.ndarray:
        """Evaluate an isotropic Hapke multiple-scattering approximation."""
        mu0, mu, illuminated = self.visible_samples(
            sun_direction,
            view_direction,
            shading=shading,
            projection=projection,
            camera_distance_radii=camera_distance_radii,
        )
        albedos = np.asarray(single_scattering_albedos, dtype=np.float64)
        phase = np.asarray(phase_terms, dtype=np.float64)
        if not illuminated.any():
            return np.zeros((len(albedos), len(phase)))
        mu0 = mu0[illuminated][None, :]
        mu = mu[illuminated][None, :]
        root = np.sqrt(np.maximum(1.0 - albedos[:, None], 0.0))
        h0 = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * root)
        h = (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * root)
        multiple = h0 * h - 1.0
        lommel = mu0 / np.maximum(mu0 + mu, 1e-12)
        return np.stack(
            [(lommel * (value + multiple)).sum(axis=1) for value in phase],
            axis=1,
        )


def curve_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    predicted = predicted / predicted.mean()
    observed = observed / observed.mean()
    residual = predicted - observed
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "correlation": float(np.corrcoef(predicted, observed)[0, 1]),
        "amplitude_ratio": float(predicted.std() / observed.std()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asteroids", default="1,2,3")
    parser.add_argument("--columns", default="0_hor1,225_bottom")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--brdf", choices=("lambertian", "lommel_seeliger", "combined"), default="lambertian"
    )
    parser.add_argument("--lambertian-fraction", type=float, default=0.5)
    parser.add_argument("--shading", choices=("flat", "smooth"), default="smooth")
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional cutoff in linear model-radiance units. The default is "
            "zero because the challenge cutoff is defined after video tone mapping."
        ),
    )
    parser.add_argument(
        "--projection", choices=("orthographic", "perspective"), default="orthographic"
    )
    parser.add_argument("--camera-distance-radii", type=float, default=5.0)
    parser.add_argument(
        "--phase-degrees",
        type=float,
        default=0.0,
        help="Physical -z mesh rotation at frame zero.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw_forward_validation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asteroid_numbers = [int(value) for value in args.asteroids.split(",")]
    columns = [value.strip() for value in args.columns.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    for model_num in asteroid_numbers:
        started = time.perf_counter()
        raw = load_gmsh22_binary_surface(get_raw_mesh_path(model_num))
        renderer = EmbreeDirectRenderer(raw, resolution=args.resolution)
        dataset = load_data(model_num, "simulated", "intensity").data
        frame_indices = np.arange(0, len(dataset), args.frame_stride)
        angles = np.radians(frame_indices.astype(np.float64) + args.phase_degrees)
        print(
            f"asteroid {model_num}: {len(raw.vertices):,} vertices, "
            f"{len(raw.faces):,} untouched faces"
        )
        for column in columns:
            values = []
            view0 = camera_direction(column)
            sun0 = np.array([-1.0, 0.0, 0.0])
            for angle in angles:
                values.append(
                    renderer.render(
                        _rotate_z(sun0, angle),
                        _rotate_z(view0, angle),
                        brdf=args.brdf,
                        lambertian_fraction=args.lambertian_fraction,
                        shading=args.shading,
                        intensity_threshold=args.intensity_threshold,
                        projection=args.projection,
                        camera_distance_radii=args.camera_distance_radii,
                    )
                )
            predicted = np.asarray(values)
            observed = dataset[column].to_numpy()[frame_indices]
            metrics = curve_metrics(predicted, observed)
            key = f"asteroid{model_num}_{column}"
            predictions[key] = predicted / predicted.mean()
            rows.append(
                {
                    "asteroid": model_num,
                    "column": column,
                    **metrics,
                    "seconds_since_mesh_load": time.perf_counter() - started,
                }
            )
            print(f"  {column}: " + ", ".join(f"{k}={v:.5f}" for k, v in metrics.items()))

    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    metadata = {
        "asteroids": asteroid_numbers,
        "columns": columns,
        "frame_stride": args.frame_stride,
        "resolution": args.resolution,
        "brdf": args.brdf,
        "lambertian_fraction": args.lambertian_fraction,
        "shading": args.shading,
        "intensity_threshold": args.intensity_threshold,
        "projection": args.projection,
        "camera_distance_radii": args.camera_distance_radii,
        "phase_degrees": args.phase_degrees,
        "mesh_policy": "Gmsh nodes/faces passed through without normalization or simplification",
        "visibility": "Embree hard primary rays and direct solar shadow rays",
        "indirect_bounces": 0,
    }
    (args.output_dir / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
