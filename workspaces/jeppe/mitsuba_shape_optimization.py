# %%
import json
import math
from pathlib import Path

import drjit as dr
import imageio_ffmpeg
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import trimesh

from hac.constants import MODEL_BASE_RADII, OBSERVERS
from hac.data import get_model_stl_path, load_data
from hac.mesh import construct_initial_ellipsoid_mesh
from hac.mitsuba import MitsubaMesh

mi.set_variant("llvm_ad_mono")


# %%
MODEL_NUM = 2
EXPERIMENTS = tuple(OBSERVERS)
OBSERVER_COUNT = len(EXPERIMENTS)
MOVIE_EXPERIMENT = "0_hor1"

OPTIMIZATION_FRAMES = 24
OPTIMIZATION_WIDTH = 40
OPTIMIZATION_HEIGHT = 40
OPTIMIZATION_SPP = 2
OPTIMIZATION_ITERATIONS = 300
LEARNING_RATE = 6e-2
LARGE_STEPS_LAMBDA = 25.0
PROXY_SUBDIVISIONS = 5
PROXY_FACE_COUNT = 1_000

MOVIE_FRAMES = 180
MOVIE_WIDTH = 160
MOVIE_HEIGHT = 160
MOVIE_SPP = 8
MOVIE_FPS = 30

ASTEROID_SCALE = 0.2
MIN_INSTANCE_SPACING = 50.0
MIN_PROJECTED_INSTANCE_SEPARATION = 5.0

OUTPUT_DIR = Path(f"workspaces/jeppe/shape_optimization_model{MODEL_NUM}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRUE_MOVIE_PATH = OUTPUT_DIR / "true_object.mp4"
OPTIMIZED_MOVIE_PATH = OUTPUT_DIR / "optimized_object.mp4"
COMPARISON_MOVIE_PATH = OUTPUT_DIR / "true_vs_optimized.mp4"
OPTIMIZED_MESH_PATH = OUTPUT_DIR / "optimized_asteroid.obj"
PROXY_MESH_PATH = OUTPUT_DIR / "optimization_proxy.obj"
LIGHTCURVE_PLOT_PATH = OUTPUT_DIR / "lightcurve_fit.png"
LOSS_PLOT_PATH = OUTPUT_DIR / "optimization_loss.png"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


# %%
def observer_position(experiment):
    observer = OBSERVERS[experiment]
    theta = math.radians(observer.theta)
    alpha = math.radians(observer.alpha)
    return np.array(
        [
            -math.cos(theta) * math.cos(alpha),
            -math.sin(theta) * math.cos(alpha),
            math.sin(alpha),
        ],
        dtype=float,
    ) * observer.r_obs


def build_batch_scene(
    obj_path,
    *,
    frame_count,
    width,
    height,
    spp,
    integrator_type="direct",
    experiments=EXPERIMENTS,
):
    experiments = tuple(experiments)
    observer_count = len(experiments)
    sample_count = observer_count * frame_count
    sun_direction = np.array([1.0, 0.0, 0.0])

    # Each phase needs one rotated asteroid instance. Every observer assigned
    # to that phase looks at the same instance.
    grid_side = math.ceil(frame_count ** (1.0 / 3.0))
    grid_indices = np.arange(frame_count)
    grid_coordinates = np.column_stack(
        np.unravel_index(grid_indices, (grid_side, grid_side, grid_side))
    ).astype(float)
    grid_coordinates -= 0.5 * (grid_side - 1)
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, grid_coordinates.shape)
    grid_coordinates += jitter

    # A 3D lattice can contain pairs that nearly overlap when projected along
    # a camera or illumination ray. Scale the complete layout until every pair
    # is safely separated in all projections used by this scene.
    ray_directions = [sun_direction]
    ray_directions.extend(
        observer_position(experiment) for experiment in experiments
    )
    ray_directions = np.asarray(ray_directions)
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)

    minimum_projected_distance = math.inf
    if frame_count > 1:
        pair_deltas = (
            grid_coordinates[:, None, :] - grid_coordinates[None, :, :]
        )
        upper_triangle = np.triu_indices(frame_count, k=1)
        for ray_direction in ray_directions:
            projected_deltas = pair_deltas - (
                np.sum(pair_deltas * ray_direction, axis=-1)[..., None]
                * ray_direction
            )
            projected_distances = np.linalg.norm(projected_deltas, axis=-1)
            minimum_projected_distance = min(
                minimum_projected_distance,
                float(np.min(projected_distances[upper_triangle])),
            )

    layout_scale = max(
        MIN_INSTANCE_SPACING,
        MIN_PROJECTED_INSTANCE_SEPARATION / minimum_projected_distance,
    )
    offsets = grid_coordinates * layout_scale

    def sample_indices(sample_index):
        return divmod(sample_index, frame_count)

    def instance_transform(phase_index):
        phi = 360.0 * phase_index / frame_count
        return (
            mi.ScalarTransform4f.translate(offsets[phase_index].tolist())
            @ mi.ScalarTransform4f.rotate(
                axis=[0.0, 0.0, -1.0],
                angle=phi,
            )
            @ mi.ScalarTransform4f.scale(ASTEROID_SCALE)
        )

    def sensor_transform(sample_index):
        observer_index, phase_index = sample_indices(sample_index)
        camera_position = observer_position(experiments[observer_index])
        instance_position = offsets[phase_index]
        return mi.ScalarTransform4f.look_at(
            origin=(instance_position + camera_position).tolist(),
            target=instance_position.tolist(),
            up=[0.0, 0.0, 1.0],
        )

    integrator = {"type": integrator_type}
    if integrator_type == "direct_projective":
        integrator["sppi"] = 0

    batch_sensor = {
        "type": "batch",
        "sampler": {"type": "independent", "sample_count": spp},
        "film": {
            "type": "hdrfilm",
            "pixel_format": "luminance",
            "width": width * sample_count,
            "height": height,
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
                    "width": width,
                    "height": height,
                    "sample_border": True,
                },
            }
            for sample_index in range(sample_count)
        },
    }

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": integrator,
            "sensor": batch_sensor,
            "sun": {
                "type": "directional",
                "direction": sun_direction.tolist(),
                "irradiance": {"type": "rawconstant", "value": 1.0},
            },
            "asteroid_group": {
                "type": "shapegroup",
                "asteroid": {
                    "type": "obj",
                    "filename": str(obj_path),
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
                for phase_index in range(frame_count)
            },
        }
    )
    params = mi.traverse(scene)
    position_key = next(
        key for key in params.keys() if key.endswith("vertex_positions")
    )
    face_key = f"{position_key.removesuffix('vertex_positions')}faces"
    return scene, params, position_key, face_key


def render_batch(
    scene,
    params,
    *,
    observer_count,
    frame_count,
    width,
    height,
    spp,
    seed,
):
    sample_count = observer_count * frame_count
    strip = mi.render(scene, params=params, spp=spp, seed=seed)
    return dr.transpose(
        dr.reshape(strip, (height, sample_count, width, 1)),
        axes=(1, 0, 2, 3),
    )


def normalized_lightcurves(rendered_frames, observer_count, frame_count):
    image_axes = tuple(range(1, rendered_frames.ndim))
    lightcurves = dr.reshape(
        dr.sum(rendered_frames, axis=image_axes),
        (observer_count, frame_count),
    )
    means = dr.reshape(dr.mean(lightcurves, axis=1), (observer_count, 1))
    return lightcurves / means


def bitmap_frames(rendered_frames):
    frame_count, height, width, _ = rendered_frames.shape
    strip = dr.transpose(rendered_frames, axes=(1, 0, 2, 3))
    strip = dr.reshape(strip, (height, frame_count * width, 1))
    bitmap = np.asarray(mi.util.convert_to_bitmap(strip))
    return np.transpose(
        bitmap.reshape(height, frame_count, width, 3),
        axes=(1, 0, 2, 3),
    )


def write_movie(path, frames, fps):
    height, width = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def create_optimization_proxy(radius, proxy_obj_path):
    proxy_mesh = construct_initial_ellipsoid_mesh(
        radius=radius,
        subdivisions=PROXY_SUBDIVISIONS,
    ).simplify_quadric_decimation(
        face_count=PROXY_FACE_COUNT,
    )
    proxy_mesh.export(proxy_obj_path)
    return proxy_mesh


# %%
# Keep the high-resolution model only for reference visualization
model_stl_path = get_model_stl_path(model_num=MODEL_NUM)
true_obj_path = model_stl_path.with_suffix(".obj")
if not true_obj_path.exists():
    true_obj_path = Path(MitsubaMesh.from_stl_path(model_stl_path).to_filename())

proxy_mesh = create_optimization_proxy(
    MODEL_BASE_RADII[MODEL_NUM],
    PROXY_MESH_PATH,
)
scene, params, position_key, face_key = build_batch_scene(
    PROXY_MESH_PATH,
    frame_count=OPTIMIZATION_FRAMES,
    width=OPTIMIZATION_WIDTH,
    height=OPTIMIZATION_HEIGHT,
    spp=OPTIMIZATION_SPP,
    integrator_type="direct_projective",
)

faces_np = params[face_key].numpy().reshape(-1, 3).astype(np.int64).copy()

target_data = load_data(
    model_num=MODEL_NUM,
    origin="simulated",
    curve_type="intensity",
).data
target_indices = (
    np.arange(OPTIMIZATION_FRAMES) * len(target_data) // OPTIMIZATION_FRAMES
)
target = mi.TensorXf(
    np.stack(
        [
            np.asarray(target_data[experiment], dtype=np.float32)[target_indices]
            for experiment in EXPERIMENTS
        ]
    )
)
target_means = dr.reshape(dr.mean(target, axis=1), (OBSERVER_COUNT, 1))
target = target / target_means


# %%
large_steps = mi.ad.LargeSteps(
    params[position_key],
    params[face_key],
    LARGE_STEPS_LAMBDA,
)
optimizer = mi.ad.Adam(lr=LEARNING_RATE, uniform=True)
optimizer["u"] = large_steps.to_differential(params[position_key])

loss_history = []
initial_curve = None
best_loss = math.inf
best_iteration = None
best_positions_np = None

for iteration in range(OPTIMIZATION_ITERATIONS):
    params[position_key] = large_steps.from_differential(optimizer["u"])
    params.update()

    rendered_frames = render_batch(
        scene,
        params,
        observer_count=OBSERVER_COUNT,
        frame_count=OPTIMIZATION_FRAMES,
        width=OPTIMIZATION_WIDTH,
        height=OPTIMIZATION_HEIGHT,
        spp=OPTIMIZATION_SPP,
        seed=0,
    )
    curve = normalized_lightcurves(
        rendered_frames,
        OBSERVER_COUNT,
        OPTIMIZATION_FRAMES,
    )
    if initial_curve is None:
        initial_curve = np.asarray(curve).copy()

    loss = dr.mean(dr.square(curve - target))
    loss_value = float(loss.array[0])
    if loss_value < best_loss:
        best_loss = loss_value
        best_iteration = iteration + 1
        best_positions_np = (
            params[position_key].numpy().reshape(-1, 3).astype(np.float32).copy()
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

params[position_key] = mi.Float(best_positions_np.ravel())
params.update()
optimized_positions_np = best_positions_np

final_frames = render_batch(
    scene,
    params,
    observer_count=OBSERVER_COUNT,
    frame_count=OPTIMIZATION_FRAMES,
    width=OPTIMIZATION_WIDTH,
    height=OPTIMIZATION_HEIGHT,
    spp=OPTIMIZATION_SPP,
    seed=0,
)
final_curve = np.asarray(
    normalized_lightcurves(
        final_frames,
        OBSERVER_COUNT,
        OPTIMIZATION_FRAMES,
    )
).copy()
final_loss = float(np.mean(np.square(final_curve - np.asarray(target))))


# %%
optimized_mesh = trimesh.Trimesh(
    vertices=optimized_positions_np,
    faces=faces_np,
    process=False,
)
optimized_mesh.export(OPTIMIZED_MESH_PATH)

target_np = np.asarray(target)
column_count = 4
row_count = math.ceil(OBSERVER_COUNT / column_count)
fig, axes = plt.subplots(
    row_count,
    column_count,
    figsize=(14, 2.6 * row_count),
    sharex=True,
)
for observer_index, experiment in enumerate(EXPERIMENTS):
    ax = axes.flat[observer_index]
    ax.plot(target_np[observer_index], label="HAC target", linewidth=2)
    ax.plot(initial_curve[observer_index], label="Initial shape")
    ax.plot(final_curve[observer_index], label="Optimized shape")
    ax.set_title(experiment)
    ax.grid(alpha=0.2)

for ax in axes.flat[OBSERVER_COUNT:]:
    ax.set_visible(False)

fig.supxlabel("Optimization phase sample")
fig.supylabel("Normalized intensity")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)
fig.suptitle(
    f"Model {MODEL_NUM}: all {OBSERVER_COUNT} observers, "
    f"final MSE={final_loss:.6g}",
    y=0.995,
)
fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.965))
fig.savefig(LIGHTCURVE_PLOT_PATH, dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(1, len(loss_history) + 1), loss_history, marker="o")
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean squared error")
ax.set_yscale("log")
ax.set_title("Large Steps optimization loss")
fig.tight_layout()
fig.savefig(LOSS_PLOT_PATH, dpi=160)
plt.close(fig)


# %%
true_movie_scene, true_movie_params, _, _ = build_batch_scene(
    true_obj_path,
    frame_count=MOVIE_FRAMES,
    width=MOVIE_WIDTH,
    height=MOVIE_HEIGHT,
    spp=MOVIE_SPP,
    experiments=(MOVIE_EXPERIMENT,),
)

true_rendered = render_batch(
    true_movie_scene,
    true_movie_params,
    observer_count=1,
    frame_count=MOVIE_FRAMES,
    width=MOVIE_WIDTH,
    height=MOVIE_HEIGHT,
    spp=MOVIE_SPP,
    seed=1,
)
true_movie_frames = bitmap_frames(true_rendered)

optimized_movie_scene, optimized_movie_params, _, _ = build_batch_scene(
    OPTIMIZED_MESH_PATH,
    frame_count=MOVIE_FRAMES,
    width=MOVIE_WIDTH,
    height=MOVIE_HEIGHT,
    spp=MOVIE_SPP,
    experiments=(MOVIE_EXPERIMENT,),
)
optimized_rendered = render_batch(
    optimized_movie_scene,
    optimized_movie_params,
    observer_count=1,
    frame_count=MOVIE_FRAMES,
    width=MOVIE_WIDTH,
    height=MOVIE_HEIGHT,
    spp=MOVIE_SPP,
    seed=1,
)
optimized_movie_frames = bitmap_frames(optimized_rendered)

comparison_movie_frames = np.concatenate(
    [true_movie_frames, optimized_movie_frames],
    axis=2,
)

write_movie(TRUE_MOVIE_PATH, true_movie_frames, MOVIE_FPS)
write_movie(OPTIMIZED_MOVIE_PATH, optimized_movie_frames, MOVIE_FPS)
write_movie(COMPARISON_MOVIE_PATH, comparison_movie_frames, MOVIE_FPS)

summary = {
    "model_num": MODEL_NUM,
    "experiments": list(EXPERIMENTS),
    "observer_count": OBSERVER_COUNT,
    "optimization_frames_per_observer": OPTIMIZATION_FRAMES,
    "optimization_target_count": OBSERVER_COUNT * OPTIMIZATION_FRAMES,
    "optimization_iterations": OPTIMIZATION_ITERATIONS,
    "optimization_instance_count": OPTIMIZATION_FRAMES,
    "proxy_subdivisions": PROXY_SUBDIVISIONS,
    "proxy_target_face_count": PROXY_FACE_COUNT,
    "proxy_vertex_count": len(proxy_mesh.vertices),
    "proxy_face_count": len(proxy_mesh.faces),
    "proxy_mesh": str(PROXY_MESH_PATH),
    "movie_experiment": MOVIE_EXPERIMENT,
    "initial_loss": loss_history[0],
    "best_iteration": best_iteration,
    "final_loss": final_loss,
    "true_movie": str(TRUE_MOVIE_PATH),
    "optimized_movie": str(OPTIMIZED_MOVIE_PATH),
    "comparison_movie": str(COMPARISON_MOVIE_PATH),
    "optimized_mesh": str(OPTIMIZED_MESH_PATH),
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
