# %%
import math

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

from hac.constants import OBSERVERS
from hac.data import get_model_stl_path, load_data
from hac.mitsuba import MitsubaMesh, RenderingSettings

# The AD variant is essential: mi.render() now returns an AD-enabled TensorXf.
mi.set_variant("llvm_ad_mono")


# %%
MODEL_NUM = 3
EXPERIMENT = "0_hor1"
OBSERVER = OBSERVERS[EXPERIMENT]

WIDTH = 128
HEIGHT = 128
SPP = 32
N_PHI = 360
ASTEROID_SCALE = 0.2
INSTANCE_SPACING = 10.0


# %%
mesh = MitsubaMesh.from_stl_path(get_model_stl_path(model_num=MODEL_NUM))
rendering_settings = RenderingSettings(width=WIDTH, height=HEIGHT, spp=SPP)

theta_rad = math.radians(OBSERVER.theta)
alpha_rad = math.radians(OBSERVER.alpha)
observer_position = [
    -math.cos(theta_rad) * math.cos(alpha_rad) * OBSERVER.r_obs,
    -math.sin(theta_rad) * math.cos(alpha_rad) * OBSERVER.r_obs,
    math.sin(alpha_rad) * OBSERVER.r_obs,
]

# Place instances along an axis perpendicular to both the view and the Sun.
# This prevents cross-instance intersections in primary and shadow rays.
view_direction = [-value / OBSERVER.r_obs for value in observer_position]
sun_direction = [1.0, 0.0, 0.0]
separation_axis = [
    view_direction[1] * sun_direction[2]
    - view_direction[2] * sun_direction[1],
    view_direction[2] * sun_direction[0]
    - view_direction[0] * sun_direction[2],
    view_direction[0] * sun_direction[1]
    - view_direction[1] * sun_direction[0],
]
separation_norm = math.sqrt(sum(value * value for value in separation_axis))
if separation_norm < 1e-8:
    separation_axis = [0.0, 1.0, 0.0]
else:
    separation_axis = [value / separation_norm for value in separation_axis]


def instance_offset(index):
    centered_index = index - 0.5 * (N_PHI - 1)
    return [
        centered_index * INSTANCE_SPACING * component
        for component in separation_axis
    ]


def instance_transform(index):
    phi = 360.0 * index / N_PHI
    return (
        mi.ScalarTransform4f.translate(instance_offset(index))
        @ mi.ScalarTransform4f.rotate(axis=[0.0, 0.0, -1.0], angle=phi)
        @ mi.ScalarTransform4f.scale(ASTEROID_SCALE)
    )


def sensor_transform(index):
    offset = instance_offset(index)
    return mi.ScalarTransform4f.look_at(
        origin=[offset[i] + observer_position[i] for i in range(3)],
        target=offset,
        up=[0.0, 0.0, 1.0],
    )


# The comprehensions instantiate scene nodes only. Rendering itself is a single
# vectorized call through Mitsuba's batch sensor, with no list of image tensors.
batch_sensor = {
    "type": "batch",
    "sampler": {
        "type": "independent",
        "sample_count": rendering_settings.spp,
    },
    "film": {
        "type": "hdrfilm",
        "pixel_format": "luminance",
        "width": rendering_settings.width * N_PHI,
        "height": rendering_settings.height,
        "rfilter": {"type": "gaussian"},
    },
    **{
        f"sensor_{index:03d}": {
            "type": "orthographic",
            "to_world": sensor_transform(index),
            "film": {
                "type": "hdrfilm",
                "pixel_format": "luminance",
                "width": WIDTH,
                "height": HEIGHT,
            },
        }
        for index in range(N_PHI)
    },
}

scene_dict = {
    "type": "scene",
    "integrator": {"type": "direct"},
    "sensor": batch_sensor,
    "sun": {
        "type": "directional",
        "direction": sun_direction,
        "irradiance": {"type": "rawconstant", "value": 1.0},
    },
    "asteroid_group": {
        "type": "shapegroup",
        "asteroid": {
            "type": "obj",
            "filename": mesh.to_filename(),
            "face_normals": False,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rawconstant", "value": 1.0},
            },
        },
    },
    **{
        f"asteroid_{index:03d}": {
            "type": "instance",
            "to_world": instance_transform(index),
            "shapegroup": {"type": "ref", "id": "asteroid_group"},
        }
        for index in range(N_PHI)
    },
}

scene = mi.load_dict(scene_dict)
params = mi.traverse(scene)
params

POSITION_KEY = next(
    key for key in params.keys() if key.endswith("vertex_positions")
)
base_vertex_positions = params[POSITION_KEY]

# Enable this before rendering when vertex gradients are needed.
# dr.enable_grad(base_vertex_positions)


# %%
# One renderer invocation produces a horizontal strip of all phase views.
batch_render: mi.TensorXf = mi.render(scene, params=params, seed=0, spp=SPP)

# Split the strip and move its phase axis to the front. Both operations remain
# in the TensorXf AD graph and perform no host/NumPy conversion.
renders_dr: mi.TensorXf = dr.transpose(
    dr.reshape(batch_render, (HEIGHT, N_PHI, WIDTH, 1)),
    axes=(1, 0, 2, 3),
)

assert renders_dr.shape == (N_PHI, HEIGHT, WIDTH, 1)


# %%
# Integrate over height, width, and channel while preserving the frame axis.
image_axes = tuple(range(1, renders_dr.ndim))
lightcurve_dr: mi.TensorXf = dr.sum(renders_dr, axis=image_axes)

threshold_mask = lightcurve_dr > 25.5
lightcurve_thresholded_dr = dr.select(threshold_mask, lightcurve_dr, 0.0)
lightcurve_normalized_dr = lightcurve_thresholded_dr / dr.mean(
    lightcurve_thresholded_dr
)


# %%
# These conversions reproduce the bitmap-based curve from mitsuba_test.py for
# comparison only. The differentiable tensors above remain attached to the AD
# graph, and the full strip is converted at once without a per-frame loop.
bitmap_strip = np.asarray(mi.util.convert_to_bitmap(batch_render))
bitmap_renders = np.transpose(
    bitmap_strip.reshape(HEIGHT, N_PHI, WIDTH, 3),
    axes=(1, 0, 2, 3),
)
lightcurve_bitmap = np.sum(
    np.where(bitmap_renders > 25.5, bitmap_renders, 0),
    axis=(1, 2, 3),
)
lightcurve_bitmap_normalized = lightcurve_bitmap / np.mean(lightcurve_bitmap)

hac_lightcurves = load_data(
    model_num=MODEL_NUM,
    origin="simulated",
    curve_type="intensity",
)
hac_target_values = np.asarray(hac_lightcurves.data[EXPERIMENT], dtype=np.float32)
if hac_target_values.shape != (N_PHI,):
    raise ValueError(
        f"Expected {N_PHI} HAC target samples, got {hac_target_values.shape[0]}"
    )

target = mi.TensorXf(hac_target_values)
target = target / dr.mean(target)
loss = dr.mean(dr.square(lightcurve_normalized_dr - target))
loss_value = loss.array[0]

lightcurve_for_plot = np.asarray(lightcurve_normalized_dr)
target_for_plot = np.asarray(target)

comparison_figure, comparison_ax = plt.subplots()
comparison_ax.plot(lightcurve_bitmap_normalized, label="Mitsuba simulation")
comparison_ax.plot(target_for_plot, label="HAC simulation target")
comparison_ax.plot(lightcurve_for_plot, label="Mitsuba Dr.Jit simulation")
comparison_ax.set_xlabel("Rotation sample")
comparison_ax.set_ylabel("Normalized intensity")
comparison_ax.set_title(f"Mean squared error: {loss_value:.6g}")
comparison_ax.legend()
comparison_figure
loss_value


# %%
# Example reverse-mode use:
# dr.backward(loss)
# vertex_gradient = dr.grad(base_vertex_positions)
