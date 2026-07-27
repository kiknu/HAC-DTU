"""
Largely vibe-coded to inspect the Mitsuba Z-stack scene layout and sensor renders.
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure

from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.patch import mi
from hac.mitsuba.scene import (
    SUN_DIRECTION,
    RenderingSettings,
    make_batch_scene,
)
from hac.types import ObserverGeometry


def _mesh_trace(
    mesh: MitsubaMesh,
    offsets: np.ndarray,
    bounding_cylinder_radius: float | None,
    normalize_mesh: bool,
    mesh_face_target: int,
) -> go.Mesh3d:
    """Create lightweight Plotly copies of the mesh at every phase."""
    display_mesh = mesh.to_trimesh()
    if normalize_mesh:
        if bounding_cylinder_radius is None:
            raise ValueError(
                "bounding_cylinder_radius must be provided if normalize_mesh is True"
            )
        display_mesh = normalize_mesh_to_challenge_cylinder(
            mesh=display_mesh,
            target_R=bounding_cylinder_radius,
            center_xy="bounds",
            copy=True,
        )

    if len(display_mesh.faces) > mesh_face_target:
        display_mesh = display_mesh.simplify_quadric_decimation(
            face_count=mesh_face_target
        )

    local_vertices = np.asarray(display_mesh.vertices, dtype=float)
    local_faces = np.asarray(display_mesh.faces, dtype=int)
    phase_count = len(offsets)
    vertex_count = len(local_vertices)

    vertices = np.empty((phase_count * vertex_count, 3), dtype=float)
    faces = np.empty((phase_count * len(local_faces), 3), dtype=int)
    for phase_index, offset in enumerate(offsets):
        phase_radians = 2.0 * np.pi * phase_index / phase_count
        cosine = np.cos(phase_radians)
        sine = np.sin(phase_radians)
        # This is the same positive-angle rotation around -z used by Mitsuba.
        rotation = np.asarray(
            [
                [cosine, sine, 0.0],
                [-sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        vertex_start = phase_index * vertex_count
        face_start = phase_index * len(local_faces)
        vertices[vertex_start : vertex_start + vertex_count] = (
            local_vertices @ rotation.T + offset
        )
        faces[face_start : face_start + len(local_faces)] = local_faces + vertex_start

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name="Asteroid meshes",
        color="gray",
        opacity=1.0,
        flatshading=False,
        lighting={
            "ambient": 0.55,
            "diffuse": 0.8,
            "specular": 0.1,
            "roughness": 0.2,
        },
        hoverinfo="skip",
    )


def make_interactive_layout(
    mesh: MitsubaMesh,
    n_phases: int,
    observers: dict[str, ObserverGeometry],
    rendering_settings: RenderingSettings,
    bounding_cylinder_radius: float | None,
    normalize_mesh: bool,
    mesh_face_target: int = 10_000,
    minimum_projected_phase_center_separation: float = 5.0,
) -> go.Figure:
    """Build a batch scene and visualize its exact instance and sensor layout."""
    if mesh_face_target <= 0:
        raise ValueError(f"mesh_face_target must be positive; got {mesh_face_target}.")
    batch_scene = make_batch_scene(
        mesh=mesh,
        n_phases=n_phases,
        observers=observers,
        rendering_settings=rendering_settings,
        minimum_projected_phase_center_separation=minimum_projected_phase_center_separation,
        bounding_cylinder_radius=bounding_cylinder_radius,
        normalize_mesh=normalize_mesh,
    )
    layout = batch_scene.layout
    offsets = layout.phase_offsets
    sensor_positions = layout.sensor_positions
    phase_degrees = 360.0 * np.arange(n_phases) / n_phases

    figure = go.Figure()
    figure.add_trace(
        _mesh_trace(
            mesh=mesh,
            offsets=offsets,
            bounding_cylinder_radius=bounding_cylinder_radius,
            normalize_mesh=normalize_mesh,
            mesh_face_target=mesh_face_target,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=offsets[:, 0],
            y=offsets[:, 1],
            z=offsets[:, 2],
            mode="lines+markers",
            name="Phase centers",
            line={"color": "dimgray", "width": 6},
            marker={
                "size": 8,
                "color": phase_degrees,
                "colorscale": "Turbo",
                "line": {"color": "black", "width": 2},
                "colorbar": {"title": "Phase (degrees)"},
            },
            text=[
                f"phase {phase_index}<br>{degrees:.2f}°"
                for phase_index, degrees in enumerate(phase_degrees)
            ],
            hovertemplate=(
                "%{text}<br>center=(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
            ),
        )
    )

    sensor_hover_text = []
    sensor_elevations = []
    for observer_name in layout.observer_names:
        observer = observers[observer_name]
        for phase_index, degrees in enumerate(phase_degrees):
            sensor_hover_text.append(
                f"{observer_name}<br>phase {phase_index} ({degrees:.2f}°)"
                f"<br>theta={observer.theta:.1f}°, alpha={observer.alpha:.1f}°"
            )
            sensor_elevations.append(observer.alpha)

    flat_sensor_positions = sensor_positions.reshape(-1, 3)
    figure.add_trace(
        go.Scatter3d(
            x=flat_sensor_positions[:, 0],
            y=flat_sensor_positions[:, 1],
            z=flat_sensor_positions[:, 2],
            mode="markers",
            name="Sensors",
            marker={
                "size": 7,
                "color": sensor_elevations,
                "colorscale": "RdBu",
                "cmin": -90,
                "cmax": 90,
                "line": {"color": "black", "width": 1.5},
                "colorbar": {"title": "Observer α"},
            },
            text=sensor_hover_text,
            hovertemplate=(
                "%{text}<br>position=(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
            ),
        )
    )

    ray_coordinates: tuple[list[float | None], ...] = ([], [], [])
    ray_arrow_tips = []
    ray_arrow_directions = []
    ray_lengths = []
    for observer_index in range(len(layout.observer_names)):
        for phase_index, phase_center in enumerate(offsets):
            sensor_position = sensor_positions[observer_index, phase_index]
            sensor_to_target = phase_center - sensor_position
            ray_length = np.linalg.norm(sensor_to_target)
            unit_direction = sensor_to_target / ray_length
            ray_arrow_tips.append(sensor_position + 0.8 * sensor_to_target)
            ray_arrow_directions.append(unit_direction)
            ray_lengths.append(ray_length)
            for coordinates, coordinate_index in zip(
                ray_coordinates, range(3), strict=True
            ):
                coordinates.extend(
                    (
                        float(sensor_position[coordinate_index]),
                        float(phase_center[coordinate_index]),
                        None,
                    )
                )

    figure.add_trace(
        go.Scatter3d(
            x=ray_coordinates[0],
            y=ray_coordinates[1],
            z=ray_coordinates[2],
            mode="lines",
            name="Sensor rays",
            line={"color": "royalblue", "width": 5},
            hoverinfo="skip",
        )
    )
    ray_arrow_tips = np.asarray(ray_arrow_tips)
    ray_arrow_directions = np.asarray(ray_arrow_directions)
    ray_arrow_size = 0.15 * layout.asteroid_radius
    figure.add_trace(
        go.Cone(
            x=ray_arrow_tips[:, 0],
            y=ray_arrow_tips[:, 1],
            z=ray_arrow_tips[:, 2],
            u=ray_arrow_directions[:, 0],
            v=ray_arrow_directions[:, 1],
            w=ray_arrow_directions[:, 2],
            name="Sensor ray arrows",
            anchor="tip",
            sizemode="absolute",
            sizeref=ray_arrow_size,
            colorscale=[[0.0, "royalblue"], [1.0, "royalblue"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    sun_ray_length = max(5.0, max(observer.r_obs for observer in observers.values()))
    sun_coordinates: tuple[list[float | None], ...] = ([], [], [])
    sun_arrow_tips = []
    for phase_center in offsets:
        start = phase_center - sun_ray_length * SUN_DIRECTION
        end = phase_center + sun_ray_length * SUN_DIRECTION
        sun_arrow_tips.append(phase_center + 0.75 * sun_ray_length * SUN_DIRECTION)
        for coordinates, coordinate_index in zip(
            sun_coordinates, range(3), strict=True
        ):
            coordinates.extend(
                (float(start[coordinate_index]), float(end[coordinate_index]), None)
            )

    figure.add_trace(
        go.Scatter3d(
            x=sun_coordinates[0],
            y=sun_coordinates[1],
            z=sun_coordinates[2],
            mode="lines",
            name="Sun rays (+x)",
            line={"color": "darkorange", "width": 5},
            hoverinfo="skip",
        )
    )
    sun_arrow_tips = np.asarray(sun_arrow_tips)
    sun_arrow_directions = np.repeat(SUN_DIRECTION[np.newaxis, :], n_phases, axis=0)
    figure.add_trace(
        go.Cone(
            x=sun_arrow_tips[:, 0],
            y=sun_arrow_tips[:, 1],
            z=sun_arrow_tips[:, 2],
            u=sun_arrow_directions[:, 0],
            v=sun_arrow_directions[:, 1],
            w=sun_arrow_directions[:, 2],
            name="Sun ray arrows",
            anchor="tip",
            sizemode="absolute",
            sizeref=min(
                0.4 * layout.asteroid_radius,
                0.06 * sun_ray_length,
            ),
            colorscale=[[0.0, "darkorange"], [1.0, "darkorange"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    figure.update_layout(
        title=(
            "Mitsuba batch scene — drag to orbit, scroll to zoom, "
            "click legend entries to hide"
        ),
        template="plotly_white",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z (phase stack)",
            "camera": {"projection": {"type": "orthographic"}},
            "aspectmode": "data",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 0, "r": 0, "b": 0, "t": 90},
    )
    return figure


def make_mitsuba_render_grid(
    mesh: MitsubaMesh,
    n_phases: int,
    observers: dict[str, ObserverGeometry],
    rendering_settings: RenderingSettings,
    bounding_cylinder_radius: float | None,
    normalize_mesh: bool,
    seed: int = 0,
    minimum_projected_phase_center_separation: float = 5.0,
) -> Figure:
    """Render the batch-scene sensor images in an observer-by-phase grid."""
    batch_scene = make_batch_scene(
        mesh=mesh,
        n_phases=n_phases,
        observers=observers,
        rendering_settings=rendering_settings,
        bounding_cylinder_radius=bounding_cylinder_radius,
        normalize_mesh=normalize_mesh,
        minimum_projected_phase_center_separation=minimum_projected_phase_center_separation,
    )
    layout = batch_scene.layout
    observer_count = len(layout.observer_names)
    rendered_strip = np.asarray(mi.render(batch_scene.scene, seed=seed))
    if rendered_strip.ndim != 3:
        raise ValueError(
            "Expected a height-by-width-by-channel Mitsuba render; "
            f"got shape {rendered_strip.shape}."
        )

    height, strip_width, channel_count = rendered_strip.shape
    sample_count = observer_count * n_phases
    expected_strip_width = rendering_settings.width * sample_count
    if height != rendering_settings.height or strip_width != expected_strip_width:
        raise ValueError(
            "The batch render does not match the configured film dimensions: "
            f"got {rendered_strip.shape}, expected "
            f"({rendering_settings.height}, {expected_strip_width}, channels)."
        )

    rendered_images = rendered_strip.reshape(
        rendering_settings.height,
        observer_count,
        n_phases,
        rendering_settings.width,
        channel_count,
    ).transpose(1, 2, 0, 3, 4)
    figure, axes = plt.subplots(
        observer_count,
        n_phases,
        figsize=(3.0 * n_phases, 3.0 * observer_count),
        squeeze=False,
        constrained_layout=True,
    )
    phase_degrees = 360.0 * np.arange(n_phases) / n_phases
    for observer_index, observer_name in enumerate(layout.observer_names):
        for phase_index, degrees in enumerate(phase_degrees):
            axis = axes[observer_index, phase_index]
            image = rendered_images[observer_index, phase_index]
            if channel_count == 1:
                axis.imshow(image[..., 0], cmap="gray", vmin=0.0)
            else:
                axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if observer_index == 0:
                axis.set_title(f"Phase {phase_index} ({degrees:.0f}°)")
        axes[observer_index, 0].set_ylabel(
            observer_name,
            rotation=0,
            ha="right",
            va="center",
        )

    figure.suptitle("Mitsuba sensor renders (rows: observers, columns: phases)")
    return figure
