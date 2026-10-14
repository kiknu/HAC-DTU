"""Starts from a
SMALL, uniformly-scaled sphere (radius = SPHERE_RADIUS_FRACTION *
cylinder_radius, well inside the challenge cylinder) instead of a
cylinder-fitted ellipsoid -- tests whether growing outward (adding mass)
from a small seed finds a different local minimum than carving inward
(subtracting mass) from an already-cylinder-sized ellipsoid. Same Chamfer
prior / no-hard-hull-clamp setup as that script otherwise.
"""

# Runtime setup must precede imports from hac.mitsuba.
# ruff: noqa: E402
# %%
import os

# Updated challenge lightcurve data (see module docstring). Must be set
# before any hac import -- hac.constants resolves CHALLENGE_DATA_DIR from
# DSDIR at import time.
os.environ.setdefault("DSDIR", "/work3/projects/HAC-DTU")

import datetime
import json
import math
import time
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

mi.set_variant("cuda_ad_mono")

from hac.constants import CHALLENGE_DATA_DIR, MODEL_BASE_RADII, OBSERVERS
from hac.data import get_model_stl_path, load_data
from hac.eval import calculate_voxel_measure, calculate_voxel_measure_solid
from hac.lc2mesh.data import create_icosphere
from hac.lc2mesh.utils import sample_convex_hull_surface
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.mitsuba.forward import ForwardModel
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.scene import ORTHOGRAPHIC_FRAMING_MARGIN, RenderingSettings
from hac.mitsuba.visualize import bitmap_frames, write_movie
from hac.visualize import plot_lightcurve_comparison, plot_optimization_loss

print(f"CHALLENGE_DATA_DIR resolved to: {CHALLENGE_DATA_DIR}")
_EXPECTED_CHALLENGE_DATA_DIR = "/work3/projects/HAC-DTU/challenge_data"
assert str(CHALLENGE_DATA_DIR) == _EXPECTED_CHALLENGE_DATA_DIR, (
    f"DSDIR did not take effect -- reading from {CHALLENGE_DATA_DIR} instead "
    f"of the expected shared {_EXPECTED_CHALLENGE_DATA_DIR}."
)

# %%
IS_NOTEBOOK = True
try:
    get_ipython()
except NameError:
    IS_NOTEBOOK = False

MODEL_NUM = int(os.environ.get("MODEL_NUM", "6"))
TARGET_ORIGIN = "simulated"
TARGET_CURVE_TYPE = os.environ.get("TARGET_CURVE_TYPE", "intensity")  # "intensity", "binary", or "joint" (both, weighted sum)
# Weight of the binary term relative to the intensity term when
# TARGET_CURVE_TYPE == "joint" -- the current best single-curve-type result
# is binary alone (with annealing); this tests whether combining both curve
# types beats it.
JOINT_BINARY_WEIGHT = 1.0

# Which convex-inversion reconstruction to use as the Chamfer PRIOR
# (soft pull, see module docstring) -- NOT the initial guess in this script
# (that's the ellipsoid below).
CHAMFER_PRIOR_ORIGIN = "simulated"
CHAMFER_PRIOR_CURVE_TYPE = "intensity"
CHAMFER_PRIOR_STL_PATH = Path(f"asteroid{MODEL_NUM}_{CHAMFER_PRIOR_ORIGIN}_{CHAMFER_PRIOR_CURVE_TYPE}.stl")
# Oracle experiment: swap the convex-inversion reconstruction above for the
# TRUE shape's own convex hull (requires MODEL_NUM <= 3, a known reference
# shape) -- isolates whether the convex-inversion algorithm's own
# reconstruction error is limiting the Chamfer prior's usefulness, versus
# the prior just being "some reasonably-sized convex envelope" regardless of
# how accurately it was recovered. "convex_inversion" (default, matches
# every prior run) or "true_hull" (cheating upper bound -- never a valid
# comparison point for a real reconstruction pipeline, only a diagnostic).
CHAMFER_PRIOR_SOURCE = os.environ.get("CHAMFER_PRIOR_SOURCE", "convex_inversion")
if CHAMFER_PRIOR_SOURCE not in ("convex_inversion", "true_hull"):
    raise ValueError(f"Unknown CHAMFER_PRIOR_SOURCE: {CHAMFER_PRIOR_SOURCE!r}")
# Weight of the Chamfer-prior term (mean squared distance from each vertex
# to its nearest point on the convex-inversion surface) relative to the
# curve-fit loss. 0 disables it entirely (pure ellipsoid-vs-lightcurve
# optimization, no prior pull at all -- an even more aggressive test of
# whether the hard hull clamp was the only thing keeping things stable).
LAMBDA_CHAMFER = float(os.environ.get("LAMBDA_CHAMFER", "10"))
# Anneal the Chamfer-prior weight linearly to 0 over the first
# CHAMFER_ANNEAL_END_FRACTION of the run -- lets it act as a stabilizing
# seed early on (when the mesh is still far from any reasonable shape, e.g.
# growing from the small sphere) and then get out of the way later, so the
# optimizer can move further from the convex-inversion shape if the
# lightcurve data actually supports it. OFF by default (matches the
# already-tested fixed-weight behavior); the nearest-neighbor match is
# still computed every iteration even after the weight decays to 0 (cheap
# relative to the Mitsuba render, and its value stays useful to watch).
CHAMFER_ANNEAL_ENABLED = os.environ.get("CHAMFER_ANNEAL_ENABLED", "1") == "1"
CHAMFER_ANNEAL_END_FRACTION = float(os.environ.get("CHAMFER_ANNEAL_END_FRACTION", "1"))
# Number of area-uniform surface samples on the convex-inversion mesh used
# as the (fixed) Chamfer target point cloud -- same default as the INR
# pipeline's own Chamfer prior (inr_forward_train_asteroid3.py).
CHAMFER_N_SAMPLES = 2000
# Small starting sphere -- a plain icosphere (uniform radius, NOT fitted
# anisotropically to the cylinder like the ellipsoid variant), scaled to
# SPHERE_RADIUS_FRACTION * cylinder_radius and centered at the origin, so it
# starts well inside the true shape with room to grow via gradient descent
# (see module docstring for why the per-iteration cylinder normalization
# had to be replaced with a clamp for this to actually happen). Already
# isotropic by construction, so no Botsch-Kobbelt remesh-at-start is needed.
SPHERE_SUBDIVISIONS = 2  # 4 -> 2562 vertices / 5120 faces
SPHERE_RADIUS_FRACTION = float(os.environ.get("SPHERE_RADIUS_FRACTION", "0.8"))

ENABLE_MID_RUN_REMESH = False

# Remove duplicates
EXPERIMENTS = tuple(
    experiment
    for experiment in OBSERVERS
    if TARGET_ORIGIN != "simulated" or not experiment.endswith("_hor2")
)
MOVIE_EXPERIMENT = "0_hor1"

OPTIMIZATION_PHASES = int(os.environ.get("OPTIMIZATION_PHASES", "90"))
OPTIMIZATION_WIDTH = 32 # 64
OPTIMIZATION_HEIGHT = 32 # 64
OPTIMIZATION_SPP = 8 # 8
OPTIMIZATION_ITERATIONS = 1000
LEARNING_RATE = 1e-2
LARGE_STEPS_LAMBDA = float(os.environ.get("LARGE_STEPS_LAMBDA", "12.0")) 
LOSS_TYPE = "mse"  # "l1" or "mse"

# Used when TARGET_CURVE_TYPE == "binary".
BINARY_SOFT_THRESHOLD_TEMPERATURE = 0.02
BINARY_TEMPERATURE_ANNEAL_END = 0.0075
BINARY_TEMPERATURE_ANNEAL_FRACTION = 0.7
BINARY_TEMPERATURE_GRAD_COLLAPSE_EPS = 1e-12

# Botsch-Kobbelt isotropic remeshing (gpytoolbox.remesh_botsch), applied
# mid-optimization to add resolution so the optimizer has local vertices to
# carve out concavities with -- mirrors Mitsuba's own shape-optimization
# tutorial (mitsuba.readthedocs.io/en/stable/src/inverse_rendering/
# shape_optimization.html)
REMESH_AT_ITERATIONS = (
    (OPTIMIZATION_ITERATIONS // 2,)
    if ENABLE_MID_RUN_REMESH and OPTIMIZATION_ITERATIONS >= 20
    else ()
)
REMESH_BOTSCH_KOBBELT_ITERATIONS = 5
REMESH_TARGET_EDGE_LENGTH_FACTOR = 0.5

# Discourage the optimized shape from bulging outward past the ellipsoid
# initial guess's OWN convex hull -- OFF by default here (unlike the base
# script), since the whole point of this variant is to let the optimizer
# explore outside the convex-inversion mesh's hull; the Chamfer prior above
# is meant to be this script's stabilizer instead.
# - "hard_clamp": positions are clamped to max_radii every iteration before
#   rendering.
# - "soft_penalty": a smooth quadratic-overshoot term (see
#   convex_hull_penalty) is added to the loss instead of clamping, keeping
#   the whole pipeline differentiable end-to-end. CONVEX_HULL_PENALTY_WEIGHT
#   trades off strictness vs. smoothness, the same way LARGE_STEPS_LAMBDA
#   trades off detail vs. smoothness.
ENFORCE_CONVEX_HULL_CONSTRAINT = os.environ.get("ENFORCE_CONVEX_HULL_CONSTRAINT", "0") == "1"
CONVEX_HULL_CONSTRAINT_MODE = "hard_clamp"  # "soft_penalty" or "hard_clamp"
CONVEX_HULL_PENALTY_WEIGHT = 10.0

# Whether to apply clamp_positions_to_cylinder every iteration at all.
ENFORCE_CYLINDER_CONSTRAINT = os.environ.get("ENFORCE_CYLINDER_CONSTRAINT", "1") == "1"

################################################################
# NOT FOR REALISTIC RECONSTRUCTION
# Only let vertices move where we KNOW (from the true
# shape, ray-cast against it) there's a genuine concavity -- everything
# else stays pinned exactly at its initial-guess position. This is a
# cheating/upper-bound experiment
CONCAVITY_MASK_EXPERIMENT = False
# Minimum relative radial deficit (true-shape radius vs. hull radius, same
# direction from the hull centroid) to classify a vertex as "inside a real
# concavity" (free to move) rather than "just the usual gap between a flat
# hull facet and the mildly curved true surface"
CONCAVITY_RELATIVE_THRESHOLD = 0.05
################################################################

MOVIE_PHASES = 360
MOVIE_WIDTH = 512
MOVIE_HEIGHT = 512
MOVIE_SPP = 16
MOVIE_FPS = 30

# Mesh-evolution movie: captures the mesh every EVOLUTION_MOVIE_INTERVAL
EVOLUTION_MOVIE_ENABLED = True
EVOLUTION_MOVIE_INTERVAL = 100
EVOLUTION_MOVIE_PHASES_PER_CHECKPOINT = 60

MIN_PROJECTED_PHASE_CENTER_SEPARATION = 5.0

# Mesh-space ground-truth comparison
VOXEL_SIMILARITY_PITCH = 0.02

_hull_tag = "_hull" if ENFORCE_CONVEX_HULL_CONSTRAINT else ""
_cylinder_tag = "" if ENFORCE_CYLINDER_CONSTRAINT else "_nocylinder"
_lambda_tag = f"_lslambda{LARGE_STEPS_LAMBDA:g}" if LARGE_STEPS_LAMBDA != 25.0 else ""
_chamfer_tag = f"_chamfer{LAMBDA_CHAMFER:g}" if LAMBDA_CHAMFER != 0.3 else ""
_anneal_tag = "_anneal" if CHAMFER_ANNEAL_ENABLED else ""
_sphere_tag = f"_r{SPHERE_RADIUS_FRACTION:g}" if SPHERE_RADIUS_FRACTION != 0.4 else ""
_prior_source_tag = "_truehullprior" if CHAMFER_PRIOR_SOURCE == "true_hull" else ""
OUTPUT_DIR = Path(
    f"mitsuba_opt_small_sphere_chamfer_prior_model{MODEL_NUM}_{TARGET_ORIGIN}_{TARGET_CURVE_TYPE}{_hull_tag}{_cylinder_tag}{_lambda_tag}{_chamfer_tag}{_anneal_tag}{_sphere_tag}{_prior_source_tag}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)
TRUE_MOVIE_PATH = OUTPUT_DIR / "true_object.mp4"
INITIAL_GUESS_MOVIE_PATH = OUTPUT_DIR / "initial_guess.mp4"
OPTIMIZED_MOVIE_PATH = OUTPUT_DIR / "optimized_object.mp4"
COMPARISON_MOVIE_PATH = OUTPUT_DIR / "true_vs_optimized.mp4"
OPTIMIZED_MESH_PATH = OUTPUT_DIR / "optimized_asteroid.obj"
INITIAL_GUESS_MESH_PATH = OUTPUT_DIR / "initial_guess.obj"
NORMALIZED_REFERENCE_MESH_PATH = OUTPUT_DIR / "normalized_reference.obj"
CHAMFER_PRIOR_MESH_PATH = OUTPUT_DIR / "chamfer_prior_convex_inversion.obj"
LIGHTCURVE_PLOT_PATH = OUTPUT_DIR / "lightcurve_fit.png"
INTENSITY_LIGHTCURVE_PLOT_PATH = OUTPUT_DIR / "lightcurve_fit_intensity.png"
BINARY_LIGHTCURVE_PLOT_PATH = OUTPUT_DIR / "lightcurve_fit_binary.png"
LOSS_PLOT_PATH = OUTPUT_DIR / "optimization_loss.png"
VOXEL_SIMILARITY_PLOT_PATH = OUTPUT_DIR / "voxel_similarity.png"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
EVOLUTION_MOVIE_PATH = OUTPUT_DIR / "mesh_evolution.mp4"
EVOLUTION_CHECKPOINT_DIR = OUTPUT_DIR / "evolution_checkpoints"


def dr_loss(curve, target):
    diff = curve - target
    return dr.mean(dr.abs(diff)) if LOSS_TYPE == "l1" else dr.mean(dr.square(diff))


def np_loss(curve, target):
    diff = curve - target
    return float(np.mean(np.abs(diff)) if LOSS_TYPE == "l1" else np.mean(np.square(diff)))


def compute_otsu_threshold_per_observer(frames_tensor, observer_count):
    """Per-observer Otsu threshold from each observer's own phase-0 frame --
    mirrors hac.simulate.generate_binary_curve's threshold_otsu(frames[0]),
    computed once from a reference render and reused as a fixed constant.
    Returns an (observer_count, 1, 1, 1, 1) mi.TensorXf, ready to broadcast
    against ForwardModel.render()'s (observer, phase, y, x, channel)
    output. See BINARY_SOFT_THRESHOLD_TEMPERATURE."""
    frames_np = np.asarray(frames_tensor)
    thresholds = np.empty(observer_count, dtype=np.float32)
    for i in range(observer_count):
        frame0 = frames_np[i, 0, :, :, 0]
        if frame0.max() <= frame0.min():
            # Degenerate (e.g. fully unlit) frame -- Otsu is undefined on a
            # constant image. Fall back to that constant value so nothing
            # ever counts as "lit" for this observer.
            thresholds[i] = frame0.max()
        else:
            thresholds[i] = threshold_otsu(frame0)
    return mi.TensorXf(thresholds.reshape(observer_count, 1, 1, 1, 1))


def soft_binary_curve_from_frames(frames_tensor, threshold, temperature, observer_count):
    """Mean-normalized soft-binary curve from an ALREADY-rendered frames
    tensor -- same math as ForwardModel.render_binary_lightcurves, but
    reused here to avoid a redundant re-render when the frames were already
    produced (e.g. to compute the Otsu threshold itself)."""
    soft_lit = 1.0 / (1.0 + dr.exp(-(frames_tensor - threshold) / temperature))
    lightcurves = dr.sum(soft_lit, axis=(2, 3, 4))
    observer_means = dr.reshape(dr.mean(lightcurves, axis=1), (observer_count, 1))
    return lightcurves / observer_means


def clamp_positions_to_cylinder(flat_positions, cylinder_radius):
    """Per-vertex CLAMP to the known challenge cylinder bound (radial
    distance from the z-axis <= cylinder_radius, z in [-1, 1]). Unlike the
    force-touch normalize_positions_to_challenge_cylinder used in the base
    script and the ellipsoid+Chamfer variant, this does NOT rescale the
    mesh to exactly touch the boundary every iteration, only prevents
    exceeding it. See module docstring for why this script needs a clamp
    instead: growth from a small starting sphere has to happen via actual
    gradient descent, not an automatic rescale to full cylinder size."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )
    radial = dr.sqrt(dr.square(vertices.x) + dr.square(vertices.y))
    clamped_radial = dr.minimum(radial, cylinder_radius)
    scale = clamped_radial / dr.maximum(radial, 1e-8)
    clamped_z = dr.maximum(dr.minimum(vertices.z, 1.0), -1.0)
    clamped_vertices = mi.Point3f(vertices.x * scale, vertices.y * scale, clamped_z)
    return dr.ravel(clamped_vertices)


def compute_convex_hull_face_data(mesh: trimesh.Trimesh):
    """Outward face normals and plane offsets (dot(v, n_i) = offset_i for
    any vertex v on face i) for a convex mesh -- fixed for the whole
    optimization run, computed once from the pristine initial guess (before
    any subdivision/remeshing changes its vertex/face resolution; those
    don't change the underlying convex shape)."""
    face_normals = mesh.face_normals
    face_offsets = np.einsum("ij,ij->i", mesh.vertices[mesh.faces[:, 0]], face_normals)
    return face_normals, face_offsets


def compute_radial_bounds(vertices_np, centroid_np, hull_face_normals, hull_face_offsets):
    """For each vertex, the distance from centroid_np to the convex hull's
    boundary along that vertex's OWN current direction -- the minimum
    positive distance to any hull face plane along the ray, which is exact
    for a convex polytope containing the centroid. Recomputed whenever the
    vertex set changes (initially, and after each remesh)."""
    directions = vertices_np - centroid_np
    radii = np.linalg.norm(directions, axis=1)
    unit_directions = directions / np.maximum(radii, 1e-8)[:, None]

    denom = unit_directions @ hull_face_normals.T  # (V, F)
    numer = hull_face_offsets[None, :] - (centroid_np @ hull_face_normals.T)  # (F,) -> (1, F)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate = np.where(denom > 1e-9, numer / denom, np.inf)
    return candidate.min(axis=1)  # (V,)


def chamfer_prior_penalty(flat_positions, matched_targets_flat):
    """Mean squared distance from each vertex to its nearest point on the
    (fixed) convex-inversion prior surface -- a soft, ONE-directional
    Chamfer-style pull toward the prior shape. `matched_targets_flat` is a
    frozen per-vertex nearest-neighbor snapshot recomputed every iteration
    (via cKDTree, outside the autodiff graph -- only the resulting distance
    is differentiated, not the match itself, standard practice for this
    kind of nearest-neighbor loss). See LAMBDA_CHAMFER."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )
    targets = mi.Point3f(
        dr.gather(mi.Float, matched_targets_flat, 3 * vertex_indices),
        dr.gather(mi.Float, matched_targets_flat, 3 * vertex_indices + 1),
        dr.gather(mi.Float, matched_targets_flat, 3 * vertex_indices + 2),
    )
    return dr.mean(dr.squared_norm(vertices - targets))


def convex_hull_penalty(flat_positions, centroid, max_radii):
    """Mean squared radial overshoot past max_radii, along each vertex's own
    current direction from centroid -- a smooth, fully differentiable
    stand-in for a hard clamp. Zero for any vertex still inside the hull,
    so it exerts no gradient pressure until a vertex actually violates the
    constraint. See ENFORCE_CONVEX_HULL_CONSTRAINT."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )

    radius = dr.norm(vertices - centroid)
    overshoot = dr.maximum(radius - max_radii, 0.0)
    return dr.mean(dr.square(overshoot))


def clamp_positions_to_convex_hull(flat_positions, centroid, max_radii):
    """Pull any vertex back along its own current direction from centroid
    if it has moved past max_radii -- see ENFORCE_CONVEX_HULL_CONSTRAINT."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )

    centered = vertices - centroid
    radius = dr.norm(centered)
    unit_direction = centered / dr.maximum(radius, 1e-8)
    clamped_radius = dr.minimum(radius, max_radii)
    clamped_vertices = centroid + unit_direction * clamped_radius
    return dr.ravel(clamped_vertices)


def compute_true_shape_radial_distances(unit_directions_np, centroid_np, true_mesh):
    """For each unit direction from centroid_np, the distance to the TRUE
    reference mesh's surface along that ray -- inf for any ray that misses
    (shouldn't happen for a watertight, roughly star-shaped-from-centroid
    mesh, but handled defensively). See CONCAVITY_MASK_EXPERIMENT."""
    ray_origins = np.tile(centroid_np, (len(unit_directions_np), 1))
    locations, index_ray, _ = true_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=unit_directions_np, multiple_hits=False,
    )
    true_radius = np.full(len(unit_directions_np), np.inf)
    if len(index_ray) > 0:
        dists = np.linalg.norm(locations - centroid_np, axis=1)
        for ray_i, dist in zip(index_ray, dists):
            true_radius[ray_i] = min(true_radius[ray_i], dist)
    return true_radius


def compute_concavity_mask(vertices_np, centroid_np, max_radii_np, true_mesh):
    """Classify each vertex as "free to move" (True, inside a genuine
    concavity of the true shape) or "frozen" (False, the hull is already
    correct there) -- see CONCAVITY_MASK_EXPERIMENT and
    CONCAVITY_RELATIVE_THRESHOLD."""
    directions = vertices_np - centroid_np
    radii = np.linalg.norm(directions, axis=1)
    unit_directions = directions / np.maximum(radii, 1e-8)[:, None]
    true_radius = compute_true_shape_radial_distances(unit_directions, centroid_np, true_mesh)
    relative_deficit = (max_radii_np - true_radius) / np.maximum(max_radii_np, 1e-8)
    concave = relative_deficit > CONCAVITY_RELATIVE_THRESHOLD
    return np.where(np.isfinite(true_radius), concave, False)


def apply_concavity_freeze(flat_positions, frozen_flat_positions, concave_mask):
    """Overwrite any vertex NOT classified as concave back to its frozen
    (true-convex-hull) position -- see CONCAVITY_MASK_EXPERIMENT."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    positions = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )
    frozen = mi.Point3f(
        dr.gather(mi.Float, frozen_flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, frozen_flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, frozen_flat_positions, 3 * vertex_indices + 2),
    )
    blended = mi.Point3f(
        dr.select(concave_mask, positions.x, frozen.x),
        dr.select(concave_mask, positions.y, frozen.y),
        dr.select(concave_mask, positions.z, frozen.z),
    )
    return dr.ravel(blended)


def movie_frames(mesh_path: Path, n_phases: int = MOVIE_PHASES, orthographic_view_scale_override: float | None = None):
    """`orthographic_view_scale_override`: by default each call auto-fits the
    camera to THIS mesh's own bounding radius (so different-sized meshes
    each fill their own frame, hiding relative size). Pass a fixed value
    (shared across all meshes being compared) to make relative size visible
    instead -- essential for this script, whose whole premise is a
    deliberately tiny starting mesh."""
    forward = ForwardModel(
        MitsubaMesh(obj_path=mesh_path),
        n_phases=n_phases,
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
        orthographic_view_scale_override=orthographic_view_scale_override,
    )
    return bitmap_frames(forward.render(seed=1)[0])


def add_text_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn a few lines of white (black-outlined) text into the top-left
    corner of a uint8 RGB frame -- used for the mesh-evolution movie's
    per-checkpoint iteration/voxel-similarity caption."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font_size = max(14, frame.shape[0] // 24)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()
    x, y = 10, 10
    for line in lines:
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += font_size + 6
    return np.asarray(image)


def combine_movie_frames(known_model_frames, initial_guess_frames, optimized_frames):
    """Return the known-model panel (uncaptioned -- reused as-is for the
    standalone true_object.mp4) and the three-panel comparison frames, each
    panel individually captioned ("Ground Truth" / "Initial Guess" / "Final
    Reconstruction") for the true_vs_optimized.mp4 comparison movie only."""
    has_known_model = known_model_frames is not None
    if known_model_frames is None:
        known_model_frames = np.zeros_like(initial_guess_frames)

    known_model_caption = "Ground Truth" if has_known_model else "Ground Truth (not available)"
    captioned_known_model = np.stack(
        [add_text_overlay(frame, [known_model_caption]) for frame in known_model_frames], axis=0
    )
    captioned_initial_guess = np.stack(
        [add_text_overlay(frame, ["Initial Guess"]) for frame in initial_guess_frames], axis=0
    )
    captioned_optimized = np.stack(
        [add_text_overlay(frame, ["Final Reconstruction"]) for frame in optimized_frames], axis=0
    )

    comparison_frames = np.concatenate(
        [captioned_known_model, captioned_initial_guess, captioned_optimized],
        axis=2,
    )
    return known_model_frames, comparison_frames


def remesh_botsch_kobbelt(vertices_np, faces_np):
    """Isotropic Botsch-Kobbelt remeshing via gpytoolbox, same call as
    Mitsuba's shape-optimization tutorial: target edge length is half the
    mesh's current mean edge length."""
    from gpytoolbox import remesh_botsch

    v0, v1, v2 = (vertices_np[faces_np[:, i]] for i in range(3))
    mean_edge_length = np.mean(
        np.concatenate(
            [
                np.linalg.norm(v1 - v0, axis=1),
                np.linalg.norm(v2 - v1, axis=1),
                np.linalg.norm(v0 - v2, axis=1),
            ]
        )
    )
    target_edge_length = float(mean_edge_length) * REMESH_TARGET_EDGE_LENGTH_FACTOR

    v_new, f_new = remesh_botsch(
        vertices_np.astype(np.float64),
        faces_np.astype(np.int32),
        i=REMESH_BOTSCH_KOBBELT_ITERATIONS,
        h=target_edge_length,
        project=True,
    )
    return v_new.astype(np.float32), f_new.astype(np.int64)


def load_target(curve_type, observer_count):
    """Mean-normalized target lightcurve tensor for `curve_type`, resampled
    onto the OPTIMIZATION_PHASES grid -- used for whichever curve type(s)
    TARGET_CURVE_TYPE selects (one, or both when 'joint')."""
    target_data = load_data(
        model_num=MODEL_NUM, origin=TARGET_ORIGIN, curve_type=curve_type,
    ).data
    target_phase_indices = (
        np.arange(OPTIMIZATION_PHASES) * len(target_data) // OPTIMIZATION_PHASES
    )
    target = mi.TensorXf(
        np.stack(
            [
                np.asarray(target_data[experiment], dtype=np.float32)[target_phase_indices]
                for experiment in EXPERIMENTS
            ]
        )
    )
    return target / dr.reshape(dr.mean(target, axis=1), (observer_count, 1))


def build_forward(mesh_path, optimization_observers, optimization_rendering_settings):
    forward = ForwardModel(
        MitsubaMesh(obj_path=mesh_path),
        n_phases=OPTIMIZATION_PHASES,
        observers=optimization_observers,
        rendering_settings=optimization_rendering_settings,
        normalize_mesh=False,
        minimum_projected_phase_center_separation=MIN_PROJECTED_PHASE_CENTER_SEPARATION,
    )
    faces_np = (
        forward.params[forward.face_key].numpy().reshape(-1, 3).astype(np.int64).copy()
    )
    return forward, faces_np


def build_optimization_state(mesh_path, optimization_observers, optimization_rendering_settings):
    """(Re)build the forward model, LargeSteps parametrization, and Adam
    optimizer from a mesh file. Needed both for the initial mesh and after
    each remesh: LargeSteps/Adam are tied to a fixed vertex/face count, so a
    topology change requires rebuilding both from scratch."""
    forward, faces_np = build_forward(
        mesh_path, optimization_observers, optimization_rendering_settings
    )
    large_steps = mi.ad.LargeSteps(
        forward.params[forward.position_key],
        forward.params[forward.face_key],
        LARGE_STEPS_LAMBDA,
    )
    optimizer = mi.ad.Adam(lr=LEARNING_RATE, uniform=True)
    optimizer["u"] = large_steps.to_differential(forward.params[forward.position_key])
    return forward, faces_np, large_steps, optimizer


def main():
    if CONCAVITY_MASK_EXPERIMENT and not ENFORCE_CONVEX_HULL_CONSTRAINT:
        raise ValueError(
            "CONCAVITY_MASK_EXPERIMENT reuses the hull constraint's max_radii -- "
            "requires ENFORCE_CONVEX_HULL_CONSTRAINT=True."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if EVOLUTION_MOVIE_ENABLED:
        EVOLUTION_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    evolution_checkpoints = []
    start_time = time.perf_counter()

    cylinder_radius = MODEL_BASE_RADII[MODEL_NUM]
    model_stl_path = get_model_stl_path(model_num=MODEL_NUM) if MODEL_NUM <= 3 else None
    if model_stl_path is not None:
        reference_mesh = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(model_stl_path),
            target_R=cylinder_radius,
        )
        reference_mesh.export(NORMALIZED_REFERENCE_MESH_PATH)
    else:
        print(
            f"Model {MODEL_NUM} has no known reference shape; "
            "skipping reference mesh and reference loss."
        )

    sphere_radius = SPHERE_RADIUS_FRACTION * cylinder_radius
    sphere_verts_unit, sphere_faces = create_icosphere(subdivisions=SPHERE_SUBDIVISIONS)
    initial_guess = trimesh.Trimesh(
        vertices=(sphere_verts_unit * sphere_radius).astype(np.float32),
        faces=sphere_faces,
        process=False,
    )
    print(
        f"Initial guess (small sphere, radius={sphere_radius:.4f} = "
        f"{SPHERE_RADIUS_FRACTION:g} * cylinder_radius {cylinder_radius:g}): "
        f"{len(initial_guess.vertices)} vertices, {len(initial_guess.faces)} faces"
    )
    initial_guess.export(INITIAL_GUESS_MESH_PATH)

    if CHAMFER_PRIOR_SOURCE == "true_hull":
        if model_stl_path is None:
            raise ValueError(
                "CHAMFER_PRIOR_SOURCE='true_hull' requires a known reference shape "
                f"(MODEL_NUM <= 3); got MODEL_NUM={MODEL_NUM}."
            )
        # reference_mesh is already normalized to the challenge cylinder above
        # -- its convex hull inherits that normalization, so no separate
        # normalize_mesh_to_challenge_cylinder call is needed here (unlike
        # the convex-inversion STL, which is loaded fresh and unnormalized).
        chamfer_prior_mesh = reference_mesh.convex_hull
        print(
            "CHAMFER_PRIOR_SOURCE='true_hull': using the TRUE shape's own convex hull "
            "as the Chamfer prior (oracle/diagnostic experiment, not a valid "
            "reconstruction-pipeline result)."
        )
    else:
        if not CHAMFER_PRIOR_STL_PATH.exists():
            raise FileNotFoundError(
                f"Convex-inversion Chamfer-prior mesh not found at {CHAMFER_PRIOR_STL_PATH} -- "
                "run convex_inversion_multi_asteroid3.py (or the matching asteroid/data-kind "
                "script) first to produce it."
            )
        chamfer_prior_mesh = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(CHAMFER_PRIOR_STL_PATH, force="mesh"),
            target_R=cylinder_radius,
        )
    chamfer_prior_mesh.export(CHAMFER_PRIOR_MESH_PATH)
    chamfer_reference_points_np = (
        sample_convex_hull_surface(
            np.asarray(chamfer_prior_mesh.vertices), n_points=CHAMFER_N_SAMPLES, seed=42
        )
        if LAMBDA_CHAMFER > 0.0
        else None
    )
    chamfer_reference_tree = (
        cKDTree(chamfer_reference_points_np) if chamfer_reference_points_np is not None else None
    )
    if chamfer_reference_points_np is not None:
        _prior_description = (
            "TRUE shape's convex hull (oracle)"
            if CHAMFER_PRIOR_SOURCE == "true_hull"
            else str(CHAMFER_PRIOR_STL_PATH)
        )
        print(
            f"Chamfer prior ({_prior_description}): {chamfer_reference_points_np.shape[0]} "
            f"surface samples, weight={LAMBDA_CHAMFER}"
        )
    else:
        print("LAMBDA_CHAMFER=0 -- Chamfer prior disabled, pure ellipsoid-vs-lightcurve optimization.")

    optimization_observers = {
        experiment: OBSERVERS[experiment] for experiment in EXPERIMENTS
    }
    optimization_rendering_settings = RenderingSettings(
        width=OPTIMIZATION_WIDTH,
        height=OPTIMIZATION_HEIGHT,
        spp=OPTIMIZATION_SPP,
    )
    forward, faces_np, large_steps, optimizer = build_optimization_state(
        INITIAL_GUESS_MESH_PATH, optimization_observers, optimization_rendering_settings
    )

    if EVOLUTION_MOVIE_ENABLED:
        # Iteration 0 -- the initial guess, before any optimization step, so
        # the evolution movie shows the starting shape too. Captured via
        # Mitsuba's OWN loaded vertex order (forward.params), same convention
        # as every later checkpoint below.
        initial_positions_np = (
            forward.params[forward.position_key].numpy().reshape(-1, 3).astype(np.float32).copy()
        )
        initial_checkpoint_mesh = trimesh.Trimesh(
            vertices=initial_positions_np, faces=faces_np, process=False
        )
        initial_checkpoint_mesh_path = EVOLUTION_CHECKPOINT_DIR / "iter000.obj"
        initial_checkpoint_mesh.export(initial_checkpoint_mesh_path)
        initial_checkpoint_voxel_solid = (
            calculate_voxel_measure_solid(reference_mesh, initial_checkpoint_mesh, pitch=VOXEL_SIMILARITY_PITCH)
            if model_stl_path is not None
            else None
        )
        evolution_checkpoints.append(
            {
                "iteration": 0,
                "mesh_path": initial_checkpoint_mesh_path,
                "voxel_similarity_solid": initial_checkpoint_voxel_solid,
            }
        )
        voxel_str = (
            f", voxel_similarity_solid={initial_checkpoint_voxel_solid:.4f}"
            if initial_checkpoint_voxel_solid is not None
            else ""
        )
        print(f"  [evolution movie] captured checkpoint at iteration 0 (initial guess){voxel_str}")

    if ENFORCE_CONVEX_HULL_CONSTRAINT:
        # Fixed for the whole run: subdivision/remeshing change vertex/face
        # resolution but not the underlying convex shape, so the hull face
        # planes computed here stay valid throughout. Computed from
        # Mitsuba's OWN loaded vertex/face order (forward.params), NOT the
        # separately trimesh-loaded `initial_guess` object -- Mitsuba's OBJ
        # loader does not preserve trimesh's OBJ-export vertex order (same
        # point set, different permutation -- confirmed directly via
        # workspaces/christian/test_vertex_order.py), so per-vertex data
        # (max_radii) computed from one ordering and applied against the
        # other silently clamps/penalizes each vertex using a DIFFERENT
        # vertex's bound. That mismatch was the actual root cause of the
        # "irregular shape" symptom and the inflated/climbing loss seen in
        # both the hard-clamp and soft-penalty variants -- not a genuine
        # flaw in either constraint approach.
        # Unlike the base script, the initial guess here is a fine
        # icosphere-topology ellipsoid (~5000 small triangular faces), NOT
        # an already-near-convex mesh -- using its OWN raw faces as "hull
        # facets" (the base script's approach) would give each vertex ~zero
        # slack. Use its TRUE convex_hull (a small number of large flat
        # facets) instead -- see module docstring.
        mitsuba_vertices_np = (
            forward.params[forward.position_key].numpy().reshape(-1, 3).astype(np.float64)
        )
        hull_mesh = trimesh.Trimesh(
            vertices=mitsuba_vertices_np, faces=faces_np, process=False
        ).convex_hull
        hull_face_normals, hull_face_offsets = compute_convex_hull_face_data(hull_mesh)
        hull_centroid_np = np.asarray(hull_mesh.centroid, dtype=np.float64)
        hull_centroid = mi.Point3f(*hull_centroid_np.tolist())
        max_radii = mi.Float(
            compute_radial_bounds(
                mitsuba_vertices_np, hull_centroid_np,
                hull_face_normals, hull_face_offsets,
            ).astype(np.float32)
        )
        print(
            f"Convex hull constraint enabled: {len(hull_face_offsets)} hull faces, "
            f"centroid={hull_centroid_np.round(4).tolist()}"
        )

        if CONCAVITY_MASK_EXPERIMENT:
            concave_mask_np = compute_concavity_mask(
                mitsuba_vertices_np, hull_centroid_np, np.array(max_radii), reference_mesh
            )
            concave_mask = mi.Bool(concave_mask_np.tolist())
            frozen_positions = mi.Float(
                mitsuba_vertices_np.astype(np.float32).flatten().tolist()
            )
            print(
                f"Concavity mask: {int(concave_mask_np.sum())}/{len(concave_mask_np)} "
                f"vertices free to move ({100 * concave_mask_np.mean():.1f}%), "
                f"threshold={CONCAVITY_RELATIVE_THRESHOLD}"
            )

    target_intensity = None
    target_binary = None
    if TARGET_CURVE_TYPE in ("intensity", "joint"):
        target_intensity = load_target("intensity", forward.observer_count)
    if TARGET_CURVE_TYPE in ("binary", "joint"):
        target_binary = load_target("binary", forward.observer_count)

    if TARGET_CURVE_TYPE in ("binary", "joint") and model_stl_path is None:
        raise ValueError(
            f"TARGET_CURVE_TYPE={TARGET_CURVE_TYPE!r} needs a known reference shape "
            "(MODEL_NUM <= 3) to calibrate the per-observer Otsu threshold against."
        )

    binary_threshold = None
    true_shape_forward_intensity_mse = None
    true_shape_forward_binary_mse = None
    true_shape_forward_curve_mse = None
    if model_stl_path is not None:
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
        if TARGET_CURVE_TYPE in ("binary", "joint"):
            reference_frames = reference_forward.render(seed=0)
            binary_threshold = compute_otsu_threshold_per_observer(
                reference_frames, reference_forward.observer_count
            )
            reference_curve_binary = np.asarray(
                soft_binary_curve_from_frames(
                    reference_frames, binary_threshold,
                    BINARY_SOFT_THRESHOLD_TEMPERATURE, reference_forward.observer_count,
                )
            ).copy()
            true_shape_forward_binary_mse = np_loss(reference_curve_binary, np.asarray(target_binary))
            print(
                "Binary soft-threshold (per observer, from true-shape render): "
                f"{np.asarray(binary_threshold).flatten().round(4).tolist()}"
            )
            print(f"True-shape forward binary curve {LOSS_TYPE.upper()}: {true_shape_forward_binary_mse:.8f}")
        if TARGET_CURVE_TYPE in ("intensity", "joint"):
            reference_curve_intensity = np.asarray(reference_forward.render_lightcurves(seed=0)).copy()
            true_shape_forward_intensity_mse = np_loss(reference_curve_intensity, np.asarray(target_intensity))
            print(f"True-shape forward intensity curve {LOSS_TYPE.upper()}: {true_shape_forward_intensity_mse:.8f}")
        del reference_forward
        if TARGET_CURVE_TYPE == "intensity":
            true_shape_forward_curve_mse = true_shape_forward_intensity_mse
        elif TARGET_CURVE_TYPE == "binary":
            true_shape_forward_curve_mse = true_shape_forward_binary_mse
        else:
            true_shape_forward_curve_mse = (
                true_shape_forward_intensity_mse + JOINT_BINARY_WEIGHT * true_shape_forward_binary_mse
            )
        print(f"True-shape forward combined curve {LOSS_TYPE.upper()}: {true_shape_forward_curve_mse:.8f}")

    # Convex-inversion mesh's OWN forward curve -- used only for the
    # lightcurve comparison plot below
    convex_inversion_curve_intensity = None
    convex_inversion_curve_binary = None
    convex_inversion_forward = ForwardModel(
        MitsubaMesh(obj_path=CHAMFER_PRIOR_MESH_PATH),
        n_phases=OPTIMIZATION_PHASES,
        observers=optimization_observers,
        rendering_settings=optimization_rendering_settings,
        normalize_mesh=False,
        minimum_projected_phase_center_separation=MIN_PROJECTED_PHASE_CENTER_SEPARATION,
    )
    if TARGET_CURVE_TYPE in ("binary", "joint"):
        convex_inversion_curve_binary = np.asarray(
            soft_binary_curve_from_frames(
                convex_inversion_forward.render(seed=0), binary_threshold,
                BINARY_SOFT_THRESHOLD_TEMPERATURE, convex_inversion_forward.observer_count,
            )
        ).copy()
    if TARGET_CURVE_TYPE in ("intensity", "joint"):
        convex_inversion_curve_intensity = np.asarray(
            convex_inversion_forward.render_lightcurves(seed=0)
        ).copy()
    del convex_inversion_forward

    loss_history = []
    remesh_events = []
    initial_guess_curve_intensity = None
    initial_guess_curve_binary = None
    best_loss = math.inf
    best_iteration = None
    best_positions_np = None
    best_faces_np = None

    # Binary-curve temperature annealing state -- see
    # BINARY_TEMPERATURE_ANNEAL_END's comment above.
    current_binary_temperature = BINARY_SOFT_THRESHOLD_TEMPERATURE
    last_safe_binary_temperature = BINARY_SOFT_THRESHOLD_TEMPERATURE
    binary_temperature_frozen = False
    binary_temperature_history = []
    binary_anneal_iterations = max(1, int(OPTIMIZATION_ITERATIONS * BINARY_TEMPERATURE_ANNEAL_FRACTION))

    for iteration in range(OPTIMIZATION_ITERATIONS):
        positions = large_steps.from_differential(optimizer["u"])
        if ENFORCE_CYLINDER_CONSTRAINT:
            positions = clamp_positions_to_cylinder(positions, cylinder_radius)
        if ENFORCE_CONVEX_HULL_CONSTRAINT and CONVEX_HULL_CONSTRAINT_MODE == "hard_clamp":
            positions = clamp_positions_to_convex_hull(positions, hull_centroid, max_radii)
        if CONCAVITY_MASK_EXPERIMENT:
            positions = apply_concavity_freeze(positions, frozen_positions, concave_mask)
        forward.params[forward.position_key] = positions
        forward.params.update()

        chamfer_term = None
        if chamfer_reference_tree is not None:
            # Nearest-neighbor match recomputed every iteration since the
            # vertices move -- done in numpy/cKDTree (outside the autodiff
            # graph, matched target treated as frozen); only the resulting
            # per-vertex distance is differentiated. See chamfer_prior_penalty.
            current_positions_np_for_chamfer = positions.numpy().reshape(-1, 3).astype(np.float64)
            _, chamfer_nn_indices = chamfer_reference_tree.query(current_positions_np_for_chamfer, k=1)
            chamfer_matched_targets_flat = mi.Float(
                chamfer_reference_points_np[chamfer_nn_indices].astype(np.float32).flatten()
            )
            chamfer_term = chamfer_prior_penalty(positions, chamfer_matched_targets_flat)

        curve_intensity = None
        curve_binary = None
        if TARGET_CURVE_TYPE in ("intensity", "joint"):
            curve_intensity = forward.render_lightcurves(seed=0)
        if TARGET_CURVE_TYPE in ("binary", "joint"):
            if not binary_temperature_frozen:
                progress = min(1.0, iteration / binary_anneal_iterations)
                current_binary_temperature = BINARY_SOFT_THRESHOLD_TEMPERATURE * (
                    (BINARY_TEMPERATURE_ANNEAL_END / BINARY_SOFT_THRESHOLD_TEMPERATURE) ** progress
                )
            binary_temperature_history.append(current_binary_temperature)
            curve_binary = forward.render_binary_lightcurves(
                threshold=binary_threshold,
                temperature=current_binary_temperature,
                seed=0,
            )
        if initial_guess_curve_intensity is None and curve_intensity is not None:
            initial_guess_curve_intensity = np.asarray(curve_intensity).copy()
        if initial_guess_curve_binary is None and curve_binary is not None:
            initial_guess_curve_binary = np.asarray(curve_binary).copy()

        if TARGET_CURVE_TYPE == "intensity":
            curve_loss = dr_loss(curve_intensity, target_intensity)
        elif TARGET_CURVE_TYPE == "binary":
            curve_loss = dr_loss(curve_binary, target_binary)
        else:
            curve_loss = (
                dr_loss(curve_intensity, target_intensity)
                + JOINT_BINARY_WEIGHT * dr_loss(curve_binary, target_binary)
            )
        loss = curve_loss
        chamfer_value = 0.0
        current_chamfer_weight = LAMBDA_CHAMFER
        if CHAMFER_ANNEAL_ENABLED:
            anneal_end_iteration = max(1, int(OPTIMIZATION_ITERATIONS * CHAMFER_ANNEAL_END_FRACTION))
            anneal_progress = min(1.0, (iteration + 1) / anneal_end_iteration)
            current_chamfer_weight = LAMBDA_CHAMFER * (1.0 - anneal_progress)
        if chamfer_term is not None:
            chamfer_value = float(chamfer_term.array[0])
            loss = loss + current_chamfer_weight * chamfer_term
        hull_penalty_value = 0.0
        if ENFORCE_CONVEX_HULL_CONSTRAINT and CONVEX_HULL_CONSTRAINT_MODE == "soft_penalty":
            hull_penalty = convex_hull_penalty(positions, hull_centroid, max_radii)
            hull_penalty_value = float(hull_penalty.array[0])
            loss = loss + CONVEX_HULL_PENALTY_WEIGHT * hull_penalty
        loss_value = float(curve_loss.array[0])
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
            best_faces_np = faces_np.copy()

        dr.backward(loss)

        binary_temperature_str = ""
        if TARGET_CURVE_TYPE in ("binary", "joint"):
            binary_temperature_str = f" T={current_binary_temperature:.5f}"
            # NOTE: in "joint" mode this watches the COMBINED gradient
            # (intensity + binary terms), so it will rarely fire even once
            # binary's own contribution has collapsed, since intensity keeps
            # contributing signal -- in practice this just means annealing
            # proceeds all the way to BINARY_TEMPERATURE_ANNEAL_END, which
            # matches what actually happened in "binary"-only runs anyway
            # (see BINARY_TEMPERATURE_ANNEAL_END's comment).
            if not binary_temperature_frozen:
                grad_norm = float(np.linalg.norm(np.asarray(dr.grad(optimizer["u"]))))
                if grad_norm < BINARY_TEMPERATURE_GRAD_COLLAPSE_EPS:
                    binary_temperature_frozen = True
                    current_binary_temperature = last_safe_binary_temperature
                    print(
                        f"  [binary temperature annealing] gradient collapsed at "
                        f"T={binary_temperature_history[-1]:.5f} (grad_norm={grad_norm:.3e}); "
                        f"freezing temperature at last-safe value "
                        f"{last_safe_binary_temperature:.5f} for remaining iterations."
                    )
                else:
                    last_safe_binary_temperature = current_binary_temperature

        optimizer.step()

        loss_history.append(loss_value)
        hull_penalty_str = (
            f" hull_penalty={hull_penalty_value:.8f}"
            if ENFORCE_CONVEX_HULL_CONSTRAINT and CONVEX_HULL_CONSTRAINT_MODE == "soft_penalty"
            else ""
        )
        chamfer_str = (
            f" chamfer={chamfer_value:.8f} (weight={current_chamfer_weight:.4f})"
            if chamfer_term is not None
            else ""
        )
        print(
            f"Iteration {iteration + 1:03d}/{OPTIMIZATION_ITERATIONS}: "
            f"loss={loss_value:.8f}{chamfer_str}{hull_penalty_str}{binary_temperature_str}"
        )

        if EVOLUTION_MOVIE_ENABLED and (iteration + 1) % EVOLUTION_MOVIE_INTERVAL == 0:
            checkpoint_positions_np = (
                forward.params[forward.position_key]
                .numpy()
                .reshape(-1, 3)
                .astype(np.float32)
                .copy()
            )
            checkpoint_mesh = trimesh.Trimesh(
                vertices=checkpoint_positions_np, faces=faces_np, process=False
            )
            checkpoint_mesh_path = EVOLUTION_CHECKPOINT_DIR / f"iter{iteration + 1:03d}.obj"
            checkpoint_mesh.export(checkpoint_mesh_path)
            checkpoint_voxel_solid = (
                calculate_voxel_measure_solid(reference_mesh, checkpoint_mesh, pitch=VOXEL_SIMILARITY_PITCH)
                if model_stl_path is not None
                else None
            )
            evolution_checkpoints.append(
                {
                    "iteration": iteration + 1,
                    "mesh_path": checkpoint_mesh_path,
                    "voxel_similarity_solid": checkpoint_voxel_solid,
                }
            )
            voxel_str = (
                f", voxel_similarity_solid={checkpoint_voxel_solid:.4f}"
                if checkpoint_voxel_solid is not None
                else ""
            )
            print(f"  [evolution movie] captured checkpoint at iteration {iteration + 1}{voxel_str}")

        if (iteration + 1) in REMESH_AT_ITERATIONS:
            current_positions_np = (
                forward.params[forward.position_key]
                .numpy()
                .reshape(-1, 3)
                .astype(np.float32)
                .copy()
            )
            remeshed_vertices, remeshed_faces = remesh_botsch_kobbelt(
                current_positions_np, faces_np
            )
            remeshed_mesh = trimesh.Trimesh(
                vertices=remeshed_vertices, faces=remeshed_faces, process=False
            )
            remeshed_mesh_path = OUTPUT_DIR / f"remeshed_after_iter{iteration + 1:03d}.obj"
            remeshed_mesh.export(remeshed_mesh_path)
            print(
                f"Remeshed after iteration {iteration + 1}: "
                f"{current_positions_np.shape[0]} -> {remeshed_vertices.shape[0]} vertices, "
                f"{faces_np.shape[0]} -> {remeshed_faces.shape[0]} faces"
            )
            forward, faces_np, large_steps, optimizer = build_optimization_state(
                remeshed_mesh_path, optimization_observers, optimization_rendering_settings
            )
            if ENFORCE_CONVEX_HULL_CONSTRAINT:
                # Topology changed -- max_radii is per-vertex, must be
                # rebuilt for the new vertex set. The hull itself (face
                # normals/offsets/centroid) is unchanged. Uses Mitsuba's OWN
                # post-reload vertex order (forward.params), NOT
                # remeshed_vertices (gpytoolbox/trimesh-export order) --
                # same vertex-order mismatch as the initial setup, see the
                # comment there.
                mitsuba_vertices_np = (
                    forward.params[forward.position_key]
                    .numpy()
                    .reshape(-1, 3)
                    .astype(np.float64)
                )
                max_radii = mi.Float(
                    compute_radial_bounds(
                        mitsuba_vertices_np, hull_centroid_np,
                        hull_face_normals, hull_face_offsets,
                    ).astype(np.float32)
                )
                if CONCAVITY_MASK_EXPERIMENT:
                    concave_mask_np = compute_concavity_mask(
                        mitsuba_vertices_np, hull_centroid_np,
                        np.array(max_radii), reference_mesh,
                    )
                    concave_mask = mi.Bool(concave_mask_np.tolist())
                    frozen_positions = mi.Float(
                        mitsuba_vertices_np.astype(np.float32).flatten().tolist()
                    )
                    print(
                        f"Concavity mask rebuilt after remesh: "
                        f"{int(concave_mask_np.sum())}/{len(concave_mask_np)} "
                        f"vertices free to move ({100 * concave_mask_np.mean():.1f}%)"
                    )
            remesh_events.append(
                {
                    "after_iteration": iteration + 1,
                    "vertex_count": int(remeshed_vertices.shape[0]),
                    "face_count": int(remeshed_faces.shape[0]),
                    "mesh_path": str(remeshed_mesh_path),
                }
            )

    if best_positions_np is None or best_faces_np is None:
        raise RuntimeError("Shape optimization did not produce a checkpoint")
    if initial_guess_curve_intensity is None and initial_guess_curve_binary is None:
        raise RuntimeError("Shape optimization did not render an initial curve")

    # best_positions_np/best_faces_np may predate a later remesh (different
    # topology than the live `forward`), so rebuild a forward from the saved
    # checkpoint mesh rather than assuming it still matches `forward`.
    optimized_mesh = trimesh.Trimesh(
        vertices=best_positions_np, faces=best_faces_np, process=False
    )
    if not ENFORCE_CYLINDER_CONSTRAINT:
        # Nothing pinned the mesh to the known cylinder scale/position
        # during training -- rescale once here, post-hoc, purely so the
        # reported voxel-similarity score is comparable with every other
        # run this session. See ENFORCE_CYLINDER_CONSTRAINT's comment.
        optimized_mesh = normalize_mesh_to_challenge_cylinder(optimized_mesh, target_R=cylinder_radius)
    optimized_mesh.export(OPTIMIZED_MESH_PATH)

    initial_guess_voxel_similarity = None
    initial_guess_voxel_similarity_solid = None
    final_voxel_similarity = None
    final_voxel_similarity_solid = None
    if model_stl_path is not None:
        # `initial_guess`/`optimized_mesh` are already on the challenge-
        # cylinder scale (normalize_mesh_to_challenge_cylinder /
        # clamp_positions_to_cylinder respectively), matching
        # `reference_mesh` -- same convention convex_inversion_multi.py's
        # evaluate_against_ground_truth relies on before voxelizing.
        initial_guess_voxel_similarity = calculate_voxel_measure(
            reference_mesh, initial_guess, pitch=VOXEL_SIMILARITY_PITCH
        )
        initial_guess_voxel_similarity_solid = calculate_voxel_measure_solid(
            reference_mesh, initial_guess, pitch=VOXEL_SIMILARITY_PITCH
        )
        final_voxel_similarity = calculate_voxel_measure(
            reference_mesh, optimized_mesh, pitch=VOXEL_SIMILARITY_PITCH
        )
        final_voxel_similarity_solid = calculate_voxel_measure_solid(
            reference_mesh, optimized_mesh, pitch=VOXEL_SIMILARITY_PITCH
        )
        print(
            f"Voxel similarity vs. ground truth -- initial guess: "
            f"surface-shell={initial_guess_voxel_similarity:.4f} "
            f"solid-fill={initial_guess_voxel_similarity_solid:.4f}"
        )
        print(
            f"Voxel similarity vs. ground truth -- optimized:     "
            f"surface-shell={final_voxel_similarity:.4f} "
            f"solid-fill={final_voxel_similarity_solid:.4f}"
        )

    final_forward, _ = build_forward(
        OPTIMIZED_MESH_PATH, optimization_observers, optimization_rendering_settings
    )
    final_hard_threshold_binary_mse = None
    final_curve_intensity = None
    final_curve_binary = None
    final_loss_intensity = None
    final_loss_binary = None
    if TARGET_CURVE_TYPE in ("binary", "joint"):
        final_curve_binary = np.asarray(
            final_forward.render_binary_lightcurves(
                threshold=binary_threshold,
                temperature=current_binary_temperature,
                seed=0,
            )
        ).copy()
        # Secondary, non-differentiable sanity metric -- the true
        # above-threshold pixel count, which the T=0.0075-ish annealed soft
        # curve should now closely track (see BINARY_TEMPERATURE_ANNEAL_END).
        final_frames_np = np.asarray(final_forward.render(seed=0))
        threshold_np = np.asarray(binary_threshold).reshape(forward.observer_count, 1, 1, 1, 1)
        hard_lit = (final_frames_np > threshold_np).astype(np.float64)
        hard_curve = hard_lit.sum(axis=(2, 3, 4))
        hard_curve = hard_curve / hard_curve.mean(axis=1, keepdims=True)
        final_hard_threshold_binary_mse = float(np.mean((hard_curve - np.asarray(target_binary)) ** 2))
        final_loss_binary = np_loss(final_curve_binary, np.asarray(target_binary))
        print(
            f"Final binary temperature: {current_binary_temperature:.5f} "
            f"(frozen={binary_temperature_frozen}) -- "
            f"hard-threshold MSE={final_hard_threshold_binary_mse:.8f}, "
            f"final binary curve {LOSS_TYPE.upper()}={final_loss_binary:.8f}"
        )
    if TARGET_CURVE_TYPE in ("intensity", "joint"):
        final_curve_intensity = np.asarray(final_forward.render_lightcurves(seed=0)).copy()
        final_loss_intensity = np_loss(final_curve_intensity, np.asarray(target_intensity))
        print(f"Final intensity curve {LOSS_TYPE.upper()}={final_loss_intensity:.8f}")

    if TARGET_CURVE_TYPE == "intensity":
        final_loss = final_loss_intensity
    elif TARGET_CURVE_TYPE == "binary":
        final_loss = final_loss_binary
    else:
        final_loss = final_loss_intensity + JOINT_BINARY_WEIGHT * final_loss_binary
    print(f"Final combined curve {LOSS_TYPE.upper()}={final_loss:.8f}")

    chamfer_prior_curve_label = (
        "True hull (oracle)" if CHAMFER_PRIOR_SOURCE == "true_hull" else "Convex inversion"
    )
    if TARGET_CURVE_TYPE in ("intensity", "joint"):
        plot_lightcurve_comparison(
            np.asarray(target_intensity),
            convex_inversion_curve_intensity,
            final_curve_intensity,
            EXPERIMENTS,
            ("HAC target", chamfer_prior_curve_label, "Optimized shape"),
            LIGHTCURVE_PLOT_PATH if TARGET_CURVE_TYPE == "intensity" else INTENSITY_LIGHTCURVE_PLOT_PATH,
            title=(
                f"Model {MODEL_NUM} ({TARGET_ORIGIN}/intensity): all "
                f"{forward.observer_count} observers, final {LOSS_TYPE.upper()}={final_loss_intensity:.6g}"
            ),
        )
    if TARGET_CURVE_TYPE in ("binary", "joint"):
        plot_lightcurve_comparison(
            np.asarray(target_binary),
            convex_inversion_curve_binary,
            final_curve_binary,
            EXPERIMENTS,
            ("HAC target", chamfer_prior_curve_label, "Optimized shape"),
            LIGHTCURVE_PLOT_PATH if TARGET_CURVE_TYPE == "binary" else BINARY_LIGHTCURVE_PLOT_PATH,
            title=(
                f"Model {MODEL_NUM} ({TARGET_ORIGIN}/binary): all "
                f"{forward.observer_count} observers, final {LOSS_TYPE.upper()}={final_loss_binary:.6g}"
            ),
        )
    plot_optimization_loss(
        loss_history,
        LOSS_PLOT_PATH,
        reference_loss=true_shape_forward_curve_mse,
        reference_label=(
            f"True-shape forward {LOSS_TYPE.upper()}" if true_shape_forward_curve_mse is not None else None
        ),
    )

    # Plot the voxel similarity values already computed for the mesh-evolution
    # checkpoints.  These checkpoints are captured every EVOLUTION_MOVIE_INTERVAL
    # iterations (25 by default), so no additional voxelization is needed here.
    # The metric is calculate_voxel_measure_solid(reference_mesh, checkpoint_mesh),
    # i.e. similarity against the TRUE/reference shape, not its convex hull.
    if model_stl_path is not None and evolution_checkpoints:
        voxel_history = [
            checkpoint
            for checkpoint in evolution_checkpoints
            if checkpoint["voxel_similarity_solid"] is not None
        ]
        if voxel_history:
            voxel_iterations = [checkpoint["iteration"] for checkpoint in voxel_history]
            voxel_similarities = [
                checkpoint["voxel_similarity_solid"] for checkpoint in voxel_history
            ]

            plt.figure(figsize=(8, 5))
            plt.plot(voxel_iterations, voxel_similarities, marker="o")
            plt.xlabel("Optimization iteration")
            plt.ylabel("Voxel similarity (solid)")
            plt.title(f"Model {MODEL_NUM}: voxel similarity vs. iteration")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(VOXEL_SIMILARITY_PLOT_PATH, dpi=200)
            plt.close()
            print(f"Saved voxel-similarity plot: {VOXEL_SIMILARITY_PLOT_PATH}")

    # One SHARED camera scale (tied to the known cylinder_radius, not each
    # mesh's own extent) across every movie below -- otherwise each mesh's
    # camera auto-fits to its own bounding radius and a deliberately tiny
    # mesh (the small-sphere initial guess, or an early evolution
    # checkpoint) would appear to fill the frame just like the full-size
    # ground truth, hiding their true relative size.
    shared_orthographic_view_scale = ORTHOGRAPHIC_FRAMING_MARGIN * cylinder_radius
    true_movie_frames = (
        movie_frames(
            NORMALIZED_REFERENCE_MESH_PATH,
            orthographic_view_scale_override=shared_orthographic_view_scale,
        )
        if model_stl_path is not None
        else None
    )
    initial_guess_movie_frames = movie_frames(
        INITIAL_GUESS_MESH_PATH, orthographic_view_scale_override=shared_orthographic_view_scale
    )
    optimized_movie_frames = movie_frames(
        OPTIMIZED_MESH_PATH, orthographic_view_scale_override=shared_orthographic_view_scale
    )
    true_movie_frames, comparison_movie_frames = combine_movie_frames(
        true_movie_frames, initial_guess_movie_frames, optimized_movie_frames
    )
    write_movie(TRUE_MOVIE_PATH, true_movie_frames, MOVIE_FPS)
    write_movie(INITIAL_GUESS_MOVIE_PATH, initial_guess_movie_frames, MOVIE_FPS)
    write_movie(OPTIMIZED_MOVIE_PATH, optimized_movie_frames, MOVIE_FPS)
    write_movie(COMPARISON_MOVIE_PATH, comparison_movie_frames, MOVIE_FPS)

    if EVOLUTION_MOVIE_ENABLED and evolution_checkpoints:
        print(
            f"Rendering mesh-evolution movie ({len(evolution_checkpoints)} checkpoints, "
            f"{EVOLUTION_MOVIE_PHASES_PER_CHECKPOINT} phases each)..."
        )
        evolution_frame_chunks = []
        for checkpoint in evolution_checkpoints:
            checkpoint_frames = movie_frames(
                checkpoint["mesh_path"],
                n_phases=EVOLUTION_MOVIE_PHASES_PER_CHECKPOINT,
                orthographic_view_scale_override=shared_orthographic_view_scale,
            )
            iteration_label = (
                "Iteration 0 (initial guess)"
                if checkpoint["iteration"] == 0
                else f"Iteration {checkpoint['iteration']}/{OPTIMIZATION_ITERATIONS}"
            )
            caption_lines = [iteration_label]
            if checkpoint["voxel_similarity_solid"] is not None:
                caption_lines.append(
                    f"Voxel similarity (solid): {checkpoint['voxel_similarity_solid']:.4f}"
                )
            evolution_frame_chunks.append(
                np.stack(
                    [add_text_overlay(frame, caption_lines) for frame in checkpoint_frames],
                    axis=0,
                )
            )
        evolution_movie_frames = np.concatenate(evolution_frame_chunks, axis=0)
        write_movie(EVOLUTION_MOVIE_PATH, evolution_movie_frames, MOVIE_FPS)
        print(
            f"Saved mesh-evolution movie: {EVOLUTION_MOVIE_PATH} "
            f"({evolution_movie_frames.shape[0]} frames)"
        )

    elapsed_time = time.perf_counter() - start_time
    summary = {
        "model_num": MODEL_NUM,
        "target_origin": TARGET_ORIGIN,
        "target_curve_type": TARGET_CURVE_TYPE,
        "initial_guess_source": "small_sphere",
        "sphere_subdivisions": SPHERE_SUBDIVISIONS,
        "sphere_radius_fraction": SPHERE_RADIUS_FRACTION,
        "sphere_radius": sphere_radius,
        "chamfer_prior_source": CHAMFER_PRIOR_SOURCE if LAMBDA_CHAMFER > 0.0 else None,
        "chamfer_prior_stl": (
            str(CHAMFER_PRIOR_STL_PATH)
            if LAMBDA_CHAMFER > 0.0 and CHAMFER_PRIOR_SOURCE == "convex_inversion"
            else None
        ),
        "lambda_chamfer": LAMBDA_CHAMFER,
        "chamfer_anneal_enabled": CHAMFER_ANNEAL_ENABLED,
        "chamfer_anneal_end_fraction": CHAMFER_ANNEAL_END_FRACTION if CHAMFER_ANNEAL_ENABLED else None,
        "final_chamfer_weight": current_chamfer_weight,
        "chamfer_n_samples": CHAMFER_N_SAMPLES if LAMBDA_CHAMFER > 0.0 else None,
        "experiments": list(EXPERIMENTS),
        "observer_count": forward.observer_count,
        "optimization_phases_per_observer": OPTIMIZATION_PHASES,
        "optimization_target_count": forward.observer_count * OPTIMIZATION_PHASES,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "large_steps_lambda": LARGE_STEPS_LAMBDA,
        "loss_type": LOSS_TYPE,
        "joint_binary_weight": JOINT_BINARY_WEIGHT if TARGET_CURVE_TYPE == "joint" else None,
        "binary_soft_threshold_temperature_start": (
            BINARY_SOFT_THRESHOLD_TEMPERATURE if TARGET_CURVE_TYPE in ("binary", "joint") else None
        ),
        "binary_soft_threshold_temperature_anneal_end": (
            BINARY_TEMPERATURE_ANNEAL_END if TARGET_CURVE_TYPE in ("binary", "joint") else None
        ),
        "binary_soft_threshold_temperature_final": (
            current_binary_temperature if TARGET_CURVE_TYPE in ("binary", "joint") else None
        ),
        "binary_temperature_anneal_frozen_early": (
            binary_temperature_frozen if TARGET_CURVE_TYPE in ("binary", "joint") else None
        ),
        "final_hard_threshold_binary_mse": final_hard_threshold_binary_mse,
        "true_shape_forward_intensity_mse": true_shape_forward_intensity_mse,
        "true_shape_forward_binary_mse": true_shape_forward_binary_mse,
        "final_loss_intensity": final_loss_intensity,
        "final_loss_binary": final_loss_binary,
        "voxel_similarity_pitch": VOXEL_SIMILARITY_PITCH if model_stl_path is not None else None,
        "initial_guess_voxel_similarity": initial_guess_voxel_similarity,
        "initial_guess_voxel_similarity_solid": initial_guess_voxel_similarity_solid,
        "final_voxel_similarity": final_voxel_similarity,
        "final_voxel_similarity_solid": final_voxel_similarity_solid,
        "binary_threshold_per_observer": (
            np.asarray(binary_threshold).flatten().tolist()
            if binary_threshold is not None else None
        ),
        "cylinder_radius": cylinder_radius,
        "shape_scale_constraint": "differentiable_challenge_cylinder_normalization",
        "convex_hull_constraint_enabled": ENFORCE_CONVEX_HULL_CONSTRAINT,
        "cylinder_constraint_enabled": ENFORCE_CYLINDER_CONSTRAINT,
        "convex_hull_constraint_type": CONVEX_HULL_CONSTRAINT_MODE,
        "convex_hull_penalty_weight": CONVEX_HULL_PENALTY_WEIGHT,
        "concavity_mask_experiment_enabled": CONCAVITY_MASK_EXPERIMENT,
        "concavity_relative_threshold": (
            CONCAVITY_RELATIVE_THRESHOLD if CONCAVITY_MASK_EXPERIMENT else None
        ),
        "concavity_free_vertex_fraction": (
            float(concave_mask_np.mean()) if CONCAVITY_MASK_EXPERIMENT else None
        ),
        "final_z_min": float(np.min(best_positions_np[:, 2])),
        "final_z_max": float(np.max(best_positions_np[:, 2])),
        "final_radial_max": float(np.max(np.linalg.norm(best_positions_np[:, :2], axis=1))),
        "initial_guess_vertex_count": len(initial_guess.vertices),
        "initial_guess_face_count": len(initial_guess.faces),
        "initial_guess_mesh": str(INITIAL_GUESS_MESH_PATH),
        "remesh_at_iterations": list(REMESH_AT_ITERATIONS),
        "remesh_botsch_kobbelt_iterations": REMESH_BOTSCH_KOBBELT_ITERATIONS,
        "remesh_target_edge_length_factor": REMESH_TARGET_EDGE_LENGTH_FACTOR,
        "remesh_events": remesh_events,
        "best_checkpoint_vertex_count": len(best_positions_np),
        "best_checkpoint_face_count": len(best_faces_np),
        "reference_shape_available": model_stl_path is not None,
        "normalized_reference_mesh": (
            str(NORMALIZED_REFERENCE_MESH_PATH) if model_stl_path is not None else None
        ),
        "movie_experiment": MOVIE_EXPERIMENT,
        "true_shape_forward_curve_mse": true_shape_forward_curve_mse,
        "initial_guess_total_loss": loss_history[0],
        "best_iteration": best_iteration,
        "final_loss": final_loss,
        "true_movie": str(TRUE_MOVIE_PATH),
        "true_movie_is_placeholder": model_stl_path is None,
        "initial_guess_movie": str(INITIAL_GUESS_MOVIE_PATH),
        "optimized_movie": str(OPTIMIZED_MOVIE_PATH),
        "comparison_movie": str(COMPARISON_MOVIE_PATH),
        "evolution_movie": (
            str(EVOLUTION_MOVIE_PATH)
            if EVOLUTION_MOVIE_ENABLED and evolution_checkpoints
            else None
        ),
        "evolution_movie_interval": EVOLUTION_MOVIE_INTERVAL if EVOLUTION_MOVIE_ENABLED else None,
        "voxel_similarity_plot": (
            str(VOXEL_SIMILARITY_PLOT_PATH)
            if model_stl_path is not None and evolution_checkpoints
            else None
        ),
        "evolution_movie_checkpoint_count": len(evolution_checkpoints),
        "evolution_checkpoints": [
            {
                "iteration": checkpoint["iteration"],
                "voxel_similarity_solid": checkpoint["voxel_similarity_solid"],
            }
            for checkpoint in evolution_checkpoints
        ],
        "optimized_mesh": str(OPTIMIZED_MESH_PATH),
        "minimum_projected_phase_center_separation": (
            forward.layout.minimum_projected_separation
        ),
        "orthographic_view_scale": forward.layout.orthographic_view_scale,
        "elapsed_time_seconds": elapsed_time,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


# %%
if __name__ == "__main__":
    print(f"Running mitsuba shape optimization seeded from convex inversion on model {MODEL_NUM}")
    main()
