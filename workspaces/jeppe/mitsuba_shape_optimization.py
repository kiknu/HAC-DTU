"""Optimize an asteroid mesh and visualize the resulting fit."""

# Runtime setup must precede imports from hac.mitsuba.
# ruff: noqa: E402

# %%
import datetime
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DSDIR", "/work3/projects/HAC-DTU/challenge_data")

import json
import math
import time
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import trimesh

# Select the variant before importing hac.mitsuba, which imports Mitsuba.
mi.set_variant("cuda_ad_mono")

from hac.constants import MODEL_BASE_RADII, OBSERVERS
from hac.data import get_model_stl_path, load_data
from hac.mesh import (
    construct_initial_ellipsoid_mesh,
    normalize_mesh_to_challenge_cylinder,
)
from hac.mitsuba.forward import ForwardModel
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.scene import RenderingSettings
from hac.mitsuba.visualize import bitmap_frames, write_movie
from hac.visualize import (
    plot_lightcurve_comparison,
    plot_optimization_loss,
)

# %%
MODEL_NUM = 2
EXPERIMENTS = tuple(OBSERVERS)
MOVIE_EXPERIMENT = "0_hor1"

OPTIMIZATION_PHASES = 180
OPTIMIZATION_WIDTH = 64
OPTIMIZATION_HEIGHT = 64
OPTIMIZATION_SPP = 8
OPTIMIZATION_ITERATIONS = 300
LEARNING_RATE = 2e-2
LARGE_STEPS_LAMBDA = 25.0
INITIAL_GUESS_SUBDIVISIONS = 5
INITIAL_GUESS_TARGET_FACE_COUNT = 2_500

MOVIE_PHASES = 360
MOVIE_WIDTH = 512
MOVIE_HEIGHT = 512
MOVIE_SPP = 16
MOVIE_FPS = 30

# Passed to hac.mitsuba.scene. Its mesh-dependent separation constraints take
# precedence, ensuring that no phase copy can enter another camera crop.
MIN_PROJECTED_PHASE_CENTER_SEPARATION = 5.0

OUTPUT_DIR = (
    Path().resolve().parent
    / f"shape_optimization_with_viz_model{MODEL_NUM}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)
TRUE_MOVIE_PATH = OUTPUT_DIR / "true_object.mp4"
INITIAL_GUESS_MOVIE_PATH = OUTPUT_DIR / "initial_guess.mp4"
OPTIMIZED_MOVIE_PATH = OUTPUT_DIR / "optimized_object.mp4"
COMPARISON_MOVIE_PATH = OUTPUT_DIR / "true_vs_optimized.mp4"
OPTIMIZED_MESH_PATH = OUTPUT_DIR / "optimized_asteroid.obj"
INITIAL_GUESS_MESH_PATH = OUTPUT_DIR / "initial_guess.obj"
NORMALIZED_REFERENCE_MESH_PATH = OUTPUT_DIR / "normalized_reference.obj"
LIGHTCURVE_PLOT_PATH = OUTPUT_DIR / "lightcurve_fit.png"
LOSS_PLOT_PATH = OUTPUT_DIR / "optimization_loss.png"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def normalize_positions_to_challenge_cylinder(flat_positions, target_radius):
    """Differentiably normalize positions into the known challenge cylinder."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )

    x_center = 0.5 * (dr.min(vertices.x) + dr.max(vertices.x))
    y_center = 0.5 * (dr.min(vertices.y) + dr.max(vertices.y))
    centered_x = vertices.x - x_center
    centered_y = vertices.y - y_center
    radial_max = dr.max(dr.sqrt(dr.square(centered_x) + dr.square(centered_y)))

    z_min = dr.min(vertices.z)
    z_max = dr.max(vertices.z)
    z_center = 0.5 * (z_min + z_max)

    normalized_vertices = mi.Point3f(
        centered_x * (target_radius / radial_max),
        centered_y * (target_radius / radial_max),
        (vertices.z - z_center) * (2.0 / (z_max - z_min)),
    )
    return dr.ravel(normalized_vertices)


def movie_frames(mesh_path: Path):
    forward = ForwardModel(
        MitsubaMesh(obj_path=mesh_path),
        n_phases=MOVIE_PHASES,
        observers={MOVIE_EXPERIMENT: OBSERVERS[MOVIE_EXPERIMENT]},
        rendering_settings=RenderingSettings(
            width=MOVIE_WIDTH,
            height=MOVIE_HEIGHT,
            spp=MOVIE_SPP,
        ),
        normalize_mesh=False,
        minimum_projected_phase_center_separation=(
            MIN_PROJECTED_PHASE_CENTER_SEPARATION
        ),
    )
    return bitmap_frames(forward.render(seed=1)[0])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    # Retain a high-resolution reference visualized in the same canonical
    # coordinates as the initial guess.
    model_stl_path = get_model_stl_path(model_num=MODEL_NUM)
    reference_mesh = normalize_mesh_to_challenge_cylinder(
        trimesh.load_mesh(model_stl_path),
        target_R=MODEL_BASE_RADII[MODEL_NUM],
    )
    reference_mesh.export(NORMALIZED_REFERENCE_MESH_PATH)

    cylinder_radius = MODEL_BASE_RADII[MODEL_NUM]
    initial_guess = normalize_mesh_to_challenge_cylinder(
        construct_initial_ellipsoid_mesh(
            radius=cylinder_radius,
            subdivisions=INITIAL_GUESS_SUBDIVISIONS,
        ).simplify_quadric_decimation(face_count=INITIAL_GUESS_TARGET_FACE_COUNT),
        target_R=cylinder_radius,
    )
    initial_guess.export(INITIAL_GUESS_MESH_PATH)
    optimization_observers = {
        experiment: OBSERVERS[experiment] for experiment in EXPERIMENTS
    }
    optimization_rendering_settings = RenderingSettings(
        width=OPTIMIZATION_WIDTH,
        height=OPTIMIZATION_HEIGHT,
        spp=OPTIMIZATION_SPP,
    )
    forward = ForwardModel(
        MitsubaMesh(obj_path=INITIAL_GUESS_MESH_PATH),
        n_phases=OPTIMIZATION_PHASES,
        observers=optimization_observers,
        rendering_settings=optimization_rendering_settings,
        normalize_mesh=False,
        minimum_projected_phase_center_separation=(
            MIN_PROJECTED_PHASE_CENTER_SEPARATION
        ),
    )
    faces_np = (
        forward.params[forward.face_key].numpy().reshape(-1, 3).astype(np.int64).copy()
    )
    target_data = load_data(
        model_num=MODEL_NUM,
        origin="simulated",
        curve_type="intensity",
    ).data
    target_phase_indices = (
        np.arange(OPTIMIZATION_PHASES) * len(target_data) // OPTIMIZATION_PHASES
    )
    target = mi.TensorXf(
        np.stack(
            [
                np.asarray(target_data[experiment], dtype=np.float32)[
                    target_phase_indices
                ]
                for experiment in EXPERIMENTS
            ]
        )
    )
    target = target / dr.reshape(dr.mean(target, axis=1), (forward.observer_count, 1))

    reference_forward = ForwardModel(
        MitsubaMesh(obj_path=NORMALIZED_REFERENCE_MESH_PATH),
        n_phases=OPTIMIZATION_PHASES,
        observers=optimization_observers,
        rendering_settings=optimization_rendering_settings,
        normalize_mesh=False,
        minimum_projected_phase_center_separation=(
            MIN_PROJECTED_PHASE_CENTER_SEPARATION
        ),
    )
    reference_curve = np.asarray(reference_forward.render_lightcurves(seed=0)).copy()
    true_shape_forward_curve_mse = float(
        np.mean(np.square(reference_curve - np.asarray(target)))
    )
    del reference_forward
    print(f"True-shape forward curve MSE: {true_shape_forward_curve_mse:.8f}")

    large_steps = mi.ad.LargeSteps(
        forward.params[forward.position_key],
        forward.params[forward.face_key],
        LARGE_STEPS_LAMBDA,
    )
    optimizer = mi.ad.Adam(lr=LEARNING_RATE, uniform=True)
    optimizer["u"] = large_steps.to_differential(forward.params[forward.position_key])

    loss_history = []
    initial_guess_curve = None
    best_loss = math.inf
    best_iteration = None
    best_positions_np = None

    for iteration in range(OPTIMIZATION_ITERATIONS):
        forward.params[forward.position_key] = (
            normalize_positions_to_challenge_cylinder(
                large_steps.from_differential(optimizer["u"]),
                cylinder_radius,
            )
        )
        forward.params.update()

        curve = forward.render_lightcurves(seed=0)
        if initial_guess_curve is None:
            initial_guess_curve = np.asarray(curve).copy()

        loss = dr.mean(dr.square(curve - target))
        loss_value = float(loss.array[0])
        if loss_value < best_loss:
            best_loss = loss_value
            best_iteration = iteration + 1
            best_positions_np = (
                forward.params[forward.position_key]
                .numpy()
                .reshape(-1, 3)
                .astype(np.float32)
                .copy()
            )

        dr.backward(loss)
        optimizer.step()

        loss_history.append(loss_value)
        print(
            f"Iteration {iteration + 1:03d}/{OPTIMIZATION_ITERATIONS}: "
            f"loss={loss_value:.8f}"
        )

    if best_positions_np is None:
        raise RuntimeError("Shape optimization did not produce a checkpoint")
    if initial_guess_curve is None:
        raise RuntimeError("Shape optimization did not render an initial curve")

    forward.params[forward.position_key] = mi.Float(best_positions_np.ravel())
    forward.params.update()
    final_curve = np.asarray(forward.render_lightcurves(seed=0)).copy()
    final_loss = float(np.mean(np.square(final_curve - np.asarray(target))))

    optimized_mesh = trimesh.Trimesh(
        vertices=best_positions_np,
        faces=faces_np,
        process=False,
    )
    optimized_mesh.export(OPTIMIZED_MESH_PATH)

    plot_lightcurve_comparison(
        np.asarray(target),
        initial_guess_curve,
        final_curve,
        EXPERIMENTS,
        ("HAC target", "Initial guess", "Optimized shape"),
        LIGHTCURVE_PLOT_PATH,
        title=(
            f"Model {MODEL_NUM}: all {forward.observer_count} observers, "
            f"final MSE={final_loss:.6g}"
        ),
    )
    plot_optimization_loss(
        loss_history,
        LOSS_PLOT_PATH,
        reference_loss=true_shape_forward_curve_mse,
        reference_label="True-shape forward MSE",
    )

    true_movie_frames = movie_frames(NORMALIZED_REFERENCE_MESH_PATH)
    initial_guess_movie_frames = movie_frames(INITIAL_GUESS_MESH_PATH)
    optimized_movie_frames = movie_frames(OPTIMIZED_MESH_PATH)
    comparison_movie_frames = np.concatenate(
        [true_movie_frames, initial_guess_movie_frames, optimized_movie_frames],
        axis=2,
    )
    write_movie(TRUE_MOVIE_PATH, true_movie_frames, MOVIE_FPS)
    write_movie(INITIAL_GUESS_MOVIE_PATH, initial_guess_movie_frames, MOVIE_FPS)
    write_movie(OPTIMIZED_MOVIE_PATH, optimized_movie_frames, MOVIE_FPS)
    write_movie(COMPARISON_MOVIE_PATH, comparison_movie_frames, MOVIE_FPS)

    elapsed_time = time.perf_counter() - start_time
    summary = {
        "model_num": MODEL_NUM,
        "experiments": list(EXPERIMENTS),
        "observer_count": forward.observer_count,
        "optimization_phases_per_observer": OPTIMIZATION_PHASES,
        "optimization_target_count": forward.observer_count * OPTIMIZATION_PHASES,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "optimization_phase_count": OPTIMIZATION_PHASES,
        "learning_rate": LEARNING_RATE,
        "cylinder_radius": cylinder_radius,
        "shape_scale_constraint": "differentiable_challenge_cylinder_normalization",
        "final_z_min": float(np.min(best_positions_np[:, 2])),
        "final_z_max": float(np.max(best_positions_np[:, 2])),
        "final_radial_max": float(
            np.max(np.linalg.norm(best_positions_np[:, :2], axis=1))
        ),
        "initial_guess_subdivisions": INITIAL_GUESS_SUBDIVISIONS,
        "initial_guess_target_face_count": INITIAL_GUESS_TARGET_FACE_COUNT,
        "initial_guess_vertex_count": len(initial_guess.vertices),
        "initial_guess_face_count": len(initial_guess.faces),
        "initial_guess_mesh": str(INITIAL_GUESS_MESH_PATH),
        "normalized_reference_mesh": str(NORMALIZED_REFERENCE_MESH_PATH),
        "movie_experiment": MOVIE_EXPERIMENT,
        "true_shape_forward_curve_mse": true_shape_forward_curve_mse,
        "initial_guess_total_loss": loss_history[0],
        "best_iteration": best_iteration,
        "final_loss": final_loss,
        "true_movie": str(TRUE_MOVIE_PATH),
        "initial_guess_movie": str(INITIAL_GUESS_MOVIE_PATH),
        "optimized_movie": str(OPTIMIZED_MOVIE_PATH),
        "comparison_movie": str(COMPARISON_MOVIE_PATH),
        "optimized_mesh": str(OPTIMIZED_MESH_PATH),
        "minimum_projected_phase_center_separation": (
            forward.layout.minimum_projected_separation
        ),
        "orthographic_view_scale": forward.layout.orthographic_view_scale,
        "elapsed_time_seconds": elapsed_time,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
