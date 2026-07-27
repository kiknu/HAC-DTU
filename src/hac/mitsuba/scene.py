import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.patch import mi
from hac.types import ObserverGeometry

SUN_DIRECTION = np.asarray([1.0, 0.0, 0.0])
UP_DIRECTION = np.asarray([0.0, 0.0, 1.0])
ROTATION_DIRECTION = (
    -UP_DIRECTION
)  # Can be changed later in case we find misalignment in their experimental setup(s).

# Default floor for projected separation between phase centers.
MIN_PROJECTED_PHASE_CENTER_SEPARATION = 5.0
ORTHOGRAPHIC_FRAMING_MARGIN = 1.25


@dataclass(frozen=True, slots=True)
class RenderingSettings:
    width: int
    height: int
    spp: int


@dataclass(frozen=True, slots=True)
class BatchSceneLayout:
    """World-space geometry used to construct a batch scene."""

    phase_offsets: np.ndarray
    sensor_positions: np.ndarray
    observer_names: tuple[str, ...]
    asteroid_radius: float
    orthographic_view_scale: float
    minimum_projected_separation: float


@dataclass(frozen=True, slots=True)
class BatchSceneResult:
    """A loaded Mitsuba scene and the exact layout used to build it."""

    scene: Any
    layout: BatchSceneLayout


def _observer_position(observer: ObserverGeometry) -> np.ndarray:
    theta = np.radians(observer.theta)
    alpha = np.radians(observer.alpha)
    return (
        np.asarray(
            [
                -np.cos(theta) * np.cos(alpha),
                -np.sin(theta) * np.cos(alpha),
                np.sin(alpha),
            ]
        )
        * observer.r_obs
    )


def _phase_stack_offsets(
    n_phases: int,
    observers: tuple[ObserverGeometry, ...],
    minimum_projected_separation: float,
) -> np.ndarray:
    """Place phase centers on z while preserving their image-plane separation."""
    max_abs_alpha = max(abs(observer.alpha) for observer in observers)
    if max_abs_alpha >= 90.0:
        raise ValueError(
            "A z-axis phase stack cannot separate observations with "
            f"|alpha| >= 90 degrees; got {max_abs_alpha}."
        )

    # A z displacement L projects to L*cos(alpha) in an observer's image
    # plane. Azimuth theta is irrelevant, which is why one simple stack works
    # for every xy viewing direction in the challenge. The largest |alpha|
    # gives the smallest projected separation and therefore sets the spacing.
    minimum_projection_factor = np.cos(np.radians(max_abs_alpha))
    phase_spacing = minimum_projected_separation / minimum_projection_factor

    # Centering the stack around the origin keeps the largest world-space
    # coordinates half as large as a stack starting at z=0.
    phase_numbers = np.arange(n_phases, dtype=float)
    phase_numbers -= 0.5 * (n_phases - 1)
    offsets = np.zeros((n_phases, 3), dtype=float)
    offsets[:, 2] = phase_numbers * phase_spacing
    return offsets


def _make_batch_scene_layout(
    mesh_vertices: np.ndarray,
    n_phases: int,
    observers: dict[str, ObserverGeometry],
    minimum_projected_phase_center_separation: float,
) -> BatchSceneLayout:
    observer_items = tuple(observers.items())
    observer_geometries = tuple(observer for _, observer in observer_items)

    # A sphere centered on the local origin bounds every rotated copy of the
    # mesh. Besides setting the camera scale, this lets unusually large input
    # meshes increase the separation instead of silently violating isolation.
    asteroid_radius = float(np.max(np.linalg.norm(mesh_vertices, axis=1)))
    orthographic_view_scale = ORTHOGRAPHIC_FRAMING_MARGIN * asteroid_radius

    # Keep phase silhouettes outside every camera crop and prevent different
    # phase instances from shadowing one another along the directional light.
    camera_exclusion_distance = asteroid_radius + np.sqrt(2.0) * orthographic_view_scale
    shadow_exclusion_distance = 2.0 * asteroid_radius
    minimum_projected_separation = max(
        minimum_projected_phase_center_separation,
        camera_exclusion_distance,
        shadow_exclusion_distance,
    )
    phase_offsets = _phase_stack_offsets(
        n_phases,
        observer_geometries,
        minimum_projected_separation,
    )
    relative_observer_positions = np.stack(
        [_observer_position(observer) for observer in observer_geometries]
    )
    sensor_positions = (
        relative_observer_positions[:, np.newaxis, :] + phase_offsets[np.newaxis, :, :]
    )
    return BatchSceneLayout(
        phase_offsets=phase_offsets,
        sensor_positions=sensor_positions,
        observer_names=tuple(name for name, _ in observer_items),
        asteroid_radius=asteroid_radius,
        orthographic_view_scale=orthographic_view_scale,
        minimum_projected_separation=minimum_projected_separation,
    )


def _load_batch_scene(
    mesh_obj_path: Path,
    mesh_vertices: np.ndarray,
    n_phases: int,
    observers: dict[str, ObserverGeometry],
    rendering_settings: RenderingSettings,
    minimum_projected_phase_center_separation: float,
):
    observer_items = tuple(observers.items())
    observer_count = len(observer_items)
    sample_count = observer_count * n_phases
    layout = _make_batch_scene_layout(
        mesh_vertices,
        n_phases,
        observers,
        minimum_projected_phase_center_separation,
    )
    offsets = layout.phase_offsets

    def sample_indices(sample_index: int) -> tuple[int, int]:
        # Samples are observer-major: all phases for observer 0, followed by
        # all phases for observer 1, and so on.
        return divmod(sample_index, n_phases)

    def instance_transform(phase_index: int):
        phase_degrees = 360.0 * phase_index / n_phases
        # Matrix composition applies the local rotation first and then places
        # that rotated copy at its z-stack position.
        return mi.ScalarTransform4f.translate(
            offsets[phase_index].tolist()
        ) @ mi.ScalarTransform4f.rotate(
            axis=ROTATION_DIRECTION.tolist(),
            angle=phase_degrees,
        )

    def sensor_transform(sample_index: int):
        observer_index, phase_index = sample_indices(sample_index)
        instance_position = offsets[phase_index]
        sensor_position = layout.sensor_positions[observer_index, phase_index]

        # Translate the observer together with its intended phase instance.
        # This preserves the requested viewing geometry while centering the
        # orthographic crop on exactly one member of the z stack.
        return mi.ScalarTransform4f.look_at(
            origin=sensor_position.tolist(),
            target=instance_position.tolist(),
            up=UP_DIRECTION.tolist(),
        ) @ mi.ScalarTransform4f.scale(
            [layout.orthographic_view_scale, layout.orthographic_view_scale, 1.0]
        )

    batch_sensor = {
        "type": "batch",
        "sampler": {
            "type": "independent",
            "sample_count": rendering_settings.spp,
        },
        "film": {
            "type": "hdrfilm",
            "pixel_format": "luminance",
            "width": rendering_settings.width * sample_count,
            "height": rendering_settings.height,
            "sample_border": True,
            "rfilter": {"type": "gaussian"},
        },
        **{
            f"sensor_{sample_index:04d}": {
                "type": "orthographic",
                "to_world": sensor_transform(sample_index),
                "film": {
                    "type": "hdrfilm",
                    "pixel_format": "luminance",
                    "width": rendering_settings.width,
                    "height": rendering_settings.height,
                    "sample_border": True,
                },
            }
            for sample_index in range(sample_count)
        },
    }

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {
                "type": "direct_projective",
                # TODO: Maybe prb_projective and add max_depth?
            },
            "sensor": batch_sensor,
            "sun": {
                "type": "directional",
                "direction": SUN_DIRECTION.tolist(),
                "irradiance": {
                    "type": "rawconstant",
                    "value": 1.0,
                },
            },
            # All phases reference this one shape group. Mitsuba consequently
            # stores one vertex array and accumulates every phase's gradients
            # into the same underlying mesh parameters.
            "asteroid_group": {
                "type": "shapegroup",
                "asteroid": {
                    "type": "obj",
                    "filename": str(mesh_obj_path),
                    "face_normals": True,
                    "bsdf": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "rawconstant",
                            "value": 1.0,
                        },
                    },
                },
            },
            **{
                f"asteroid_{phase_index:04d}": {
                    "type": "instance",
                    "to_world": instance_transform(phase_index),
                    "shapegroup": {
                        "type": "ref",
                        "id": "asteroid_group",
                    },
                }
                for phase_index in range(n_phases)
            },
        }
    )

    return BatchSceneResult(scene=scene, layout=layout)


def make_batch_scene(
    mesh: MitsubaMesh,
    n_phases: int,
    observers: dict[str, ObserverGeometry],
    rendering_settings: RenderingSettings,
    bounding_cylinder_radius: float | None = None,
    normalize_mesh: bool = True,
    minimum_projected_phase_center_separation: float = MIN_PROJECTED_PHASE_CENTER_SEPARATION,
):
    """
    Build one scene containing every requested rotation and observer.

    The scene contains one shared-mesh instance per phase and one batch-sensor
    entry per ``(observer, phase)`` pair. Phases are evenly spaced over
    ``[0, 360)`` degrees and stacked along the z-axis so they cannot contribute
    to one another's images.

    Args:
        mesh: MitsubaMesh object representing the asteroid.
        n_phases: Number of equally spaced phases to include.
        observers: Mapping from observer names to their geometries.
        rendering_settings: Mitsuba film and sampler settings.
        bounding_cylinder_radius:
            The radius of the challenge cylinder in which the mesh should fit.
            This is given as prior information to prevent non-uniqueness
            and can be found in ``constants.py``.
        normalize_mesh: Whether to normalize the mesh into that cylinder before
            loading it into Mitsuba.
        minimum_projected_phase_center_separation: Lower bound on the projected
            separation between phase centers. Larger mesh-based exclusion
            distances still take precedence.

    Returns:
        The loaded Mitsuba scene and the exact world-space layout used for it.
    """

    if n_phases <= 0:
        raise ValueError(f"n_phases must be positive; got {n_phases}.")
    if not observers:
        raise ValueError("At least one observer is required.")
    if (
        rendering_settings.width <= 0
        or rendering_settings.height <= 0
        or rendering_settings.spp <= 0
    ):
        raise ValueError("Rendering width, height, and spp must all be positive.")
    if minimum_projected_phase_center_separation <= 0:
        raise ValueError("minimum_projected_phase_center_separation must be positive.")

    mesh_trimesh = mesh.to_trimesh()
    if len(mesh_trimesh.vertices) == 0:
        raise ValueError("Cannot build a scene from a mesh without vertices.")

    if normalize_mesh:
        if bounding_cylinder_radius is None:
            raise ValueError(
                "bounding_cylinder_radius must be provided if normalize_mesh is True"
            )

        normalized_mesh = normalize_mesh_to_challenge_cylinder(
            mesh=mesh_trimesh,
            target_R=bounding_cylinder_radius,
            center_xy="bounds",
            copy=True,
        )

        # Mitsuba loads the OBJ synchronously in mi.load_dict(), so the
        # temporary normalized file may be removed as soon as that returns.
        with tempfile.NamedTemporaryFile(
            prefix="hac-normalized-mesh", suffix=".obj"
        ) as tmp_file:
            mesh_obj_path = Path(tmp_file.name)
            normalized_mesh.export(mesh_obj_path)
            return _load_batch_scene(
                mesh_obj_path=mesh_obj_path,
                mesh_vertices=np.asarray(normalized_mesh.vertices, dtype=float),
                n_phases=n_phases,
                observers=observers,
                rendering_settings=rendering_settings,
                minimum_projected_phase_center_separation=minimum_projected_phase_center_separation,
            )

    return _load_batch_scene(
        mesh_obj_path=mesh.obj_path,
        mesh_vertices=np.asarray(mesh_trimesh.vertices, dtype=float),
        n_phases=n_phases,
        observers=observers,
        rendering_settings=rendering_settings,
        minimum_projected_phase_center_separation=minimum_projected_phase_center_separation,
    )
