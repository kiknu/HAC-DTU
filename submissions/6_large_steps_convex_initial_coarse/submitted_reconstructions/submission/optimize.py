"""
Optimize a HAC asteroid shape with Mitsuba Large Steps.

Only supports the `direct` integrator (orthographic assumption).
"""

import argparse
import datetime
import json
import math
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

DataSource = Literal["real", "simulated"]
CurveType = Literal["intensity", "binary"]
LearningRateSchedule = Literal["constant", "exponential", "linear"]

INTEGRATOR: dict[str, Any] = {"type": "direct"}
PROJECTION: Literal["orthographic"] = "orthographic"

# Remeshing is controlled only by --remesh-level or --remesh-interval; the rest
# of the Botsch-Kobbelt / adaptive-trigger tuning is fixed here.
_REMESH_BOTSCH_ITERATIONS = 5  # gpytoolbox remesh_botsch smoothing iterations
_REMESH_EDGE_FACTOR = 0.5  # target edge = this x current mean edge (~4x faces)
_REMESH_MAX_EVENTS = 8  # safety cap for --remesh-interval; also caps --remesh-level
_REMESH_ADAPTIVE_WINDOW = 20  # iterations of loss history judged for stagnation
_REMESH_ADAPTIVE_MIN_IMPROVEMENT = 0.02  # relative loss gain below which we remesh
_REMESH_ADAPTIVE_COOLDOWN = 20  # minimum iterations between adaptive remeshes

# The ground-truth shape ships only for these models, so the voxel measures
# against it can only be evaluated there.
_KNOWN_SHAPE_MAX_MODEL = 3


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    model_num: int
    data_source: DataSource
    curve_type: CurveType
    initial_mesh: Path | None
    output_dir: Path | None
    variant: str
    cuda_device: str
    phases: int | None
    width: int
    height: int
    spp: int
    spp_grad: int
    iterations: int
    learning_rate: float
    lr_schedule: LearningRateSchedule
    lr_final_factor: float
    large_steps_lambda: float
    seed: int
    validation_interval: int
    validation_spp: int
    early_stop_patience: int
    initial_subdivisions: int
    initial_face_count: int
    checkpoint_interval: int
    reduction_temperature: float
    evaluate_reference: bool
    voxel_metrics: bool
    voxel_pitch_divisor: float
    render_movies: bool
    movie_observer: str
    movie_phases: int
    movie_width: int
    movie_height: int
    movie_spp: int
    movie_fps: int
    # Exactly one of these enables remeshing; both None means fixed topology.
    remesh_level: int | None  # up to N adaptive (loss-triggered) remeshes
    remesh_interval: int | None  # remesh every N iterations (fixed schedule)

    @property
    def data_sources(self) -> tuple[DataSource, ...]:
        return (self.data_source,)

    @property
    def voxel_metrics_enabled(self) -> bool:
        """Voxel measures need the reference shape, which is public for models 1-3."""
        return self.voxel_metrics and self.model_num <= _KNOWN_SHAPE_MAX_MODEL

    def learning_rate_at(self, iteration: int) -> float:
        """Scheduled learning rate for 1-based ``iteration``.

        Decays from ``learning_rate`` at the first iteration to
        ``learning_rate * lr_final_factor`` at the last one. The schedule runs
        over the whole run and is not reset by remeshing.
        """
        if self.lr_schedule == "constant":
            return self.learning_rate
        progress = (iteration - 1) / max(self.iterations - 1, 1)
        if self.lr_schedule == "exponential":
            return self.learning_rate * self.lr_final_factor**progress
        return self.learning_rate * (1.0 - (1.0 - self.lr_final_factor) * progress)

    @property
    def remesh_enabled(self) -> bool:
        return self.remesh_level is not None or self.remesh_interval is not None

    @property
    def remesh_max_events(self) -> int:
        if self.remesh_level is not None:
            return min(self.remesh_level, _REMESH_MAX_EVENTS)
        return _REMESH_MAX_EVENTS


@dataclass(frozen=True, slots=True)
class RunPaths:
    output_dir: Path
    resolved_config: Path
    initial_mesh: Path
    best_mesh: Path
    last_mesh: Path
    normalized_reference_mesh: Path
    checkpoints: Path
    history: Path
    curves: Path
    lightcurve_plot: Path
    loss_plot: Path
    voxel_plot: Path
    initial_views: Path
    best_views: Path
    reference_views: Path
    summary: Path
    initial_movie: Path
    best_movie: Path
    reference_movie: Path
    comparison_movie: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "RunPaths":
        return cls(
            output_dir=output_dir,
            resolved_config=output_dir / "resolved_config.json",
            initial_mesh=output_dir / "initial_mesh.obj",
            best_mesh=output_dir / "best_mesh.obj",
            last_mesh=output_dir / "last_mesh.obj",
            normalized_reference_mesh=output_dir / "normalized_reference_mesh.obj",
            checkpoints=output_dir / "checkpoints",
            history=output_dir / "history.jsonl",
            curves=output_dir / "curves.npz",
            lightcurve_plot=output_dir / "lightcurve_fit.png",
            loss_plot=output_dir / "optimization_loss.png",
            voxel_plot=output_dir / "voxel_measures.png",
            initial_views=output_dir / "initial_views.png",
            best_views=output_dir / "best_views.png",
            reference_views=output_dir / "reference_views.png",
            summary=output_dir / "summary.json",
            initial_movie=output_dir / "initial_shape.mp4",
            best_movie=output_dir / "best_shape.mp4",
            reference_movie=output_dir / "reference_shape.mp4",
            comparison_movie=output_dir / "comparison.mp4",
        )


def _positive(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0:
        parser.error(f"{name} must be positive")


def parse_args(argv: Sequence[str] | None = None) -> RunConfiguration:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_num", type=int, choices=range(1, 11))
    parser.add_argument(
        "--data-source",
        choices=("real", "simulated"),
        default="simulated",
    )
    parser.add_argument(
        "--curve-type",
        choices=("intensity", "binary"),
        default="intensity",
    )
    parser.add_argument(
        "--initial-mesh",
        type=Path,
        help=(
            "optional warm-start mesh; otherwise construct an ellipsoid. Either "
            "way the result is resampled to --initial-face-count and normalized "
            "to the challenge cylinder"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--variant",
        default=os.environ.get("MITSUBA_VARIANT", "cuda_ad_mono"),
    )
    parser.add_argument(
        "--cuda-device",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )

    parser.add_argument(
        "--phases",
        type=int,
        default=None,
        help=(
            "subsample the light curves to this many frames by index (uneven, "
            "no averaging); omit to use every native frame (~841 real, 360 blender)"
        ),
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--spp-grad", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "exponential", "linear"),
        default="constant",
        help=(
            "decay the learning rate from --learning-rate to --learning-rate x "
            "--lr-final-factor over --iterations; exponential is recommended "
            "(constant relative decay, so late fine-detail stages still move)"
        ),
    )
    parser.add_argument(
        "--lr-final-factor",
        type=float,
        default=0.1,
        help="final / initial learning rate; ignored for --lr-schedule constant",
    )
    parser.add_argument("--large-steps-lambda", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-spp", type=int, default=16)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="validation checks without improvement; 0 disables early stopping",
    )
    parser.add_argument(
        "--initial-subdivisions",
        type=int,
        default=5,
        help="icosphere detail for the generated ellipsoid; unused with --initial-mesh",
    )
    parser.add_argument(
        "--initial-face-count",
        type=int,
        default=2_500,
        help="face budget the initial mesh is resampled to, warm starts included",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="iterations between mesh checkpoints; 0 disables them",
    )
    parser.add_argument("--reduction-temperature", type=float, default=0.01)
    parser.add_argument("--evaluate-reference", action="store_true")
    parser.add_argument(
        "--no-voxel-metrics",
        dest="voxel_metrics",
        action="store_false",
        help=(
            "skip the FIPS voxel volume difference against the reference shape; "
            f"it is tracked at every validation check for models 1-"
            f"{_KNOWN_SHAPE_MAX_MODEL} and ignored for the secret models"
        ),
    )
    parser.add_argument(
        "--voxel-pitch-divisor",
        type=float,
        default=40.0,
        help=(
            "voxel pitch for the voxel measures, as the model's challenge cylinder "
            "radius divided by this value (higher is finer and slower)"
        ),
    )
    parser.add_argument("--render-movies", action="store_true")
    parser.add_argument("--movie-observer", default="0_hor1")
    parser.add_argument("--movie-phases", type=int, default=360)
    parser.add_argument("--movie-width", type=int, default=512)
    parser.add_argument("--movie-height", type=int, default=512)
    parser.add_argument("--movie-spp", type=int, default=16)
    parser.add_argument("--movie-fps", type=int, default=30)

    # Botsch-Kobbelt remeshing (needs gpytoolbox). Pick one style, or neither
    # for fixed topology; all other tuning is fixed in the _REMESH_* constants.
    remesh = parser.add_mutually_exclusive_group()
    remesh.add_argument(
        "--remesh-level",
        type=int,
        default=None,
        metavar="N",
        help=(
            "progressive refinement: up to N remeshes, each triggered when the "
            f"training loss stagnates (N capped at {_REMESH_MAX_EVENTS})"
        ),
    )
    remesh.add_argument(
        "--remesh-interval",
        type=int,
        default=None,
        metavar="ITERS",
        help="fixed-schedule remeshing every ITERS iterations",
    )
    args = parser.parse_args(argv)

    for name in (
        "width",
        "height",
        "spp",
        "iterations",
        "validation_interval",
        "validation_spp",
        "initial_face_count",
        "movie_phases",
        "movie_width",
        "movie_height",
        "movie_spp",
        "movie_fps",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    for name in (
        "learning_rate",
        "lr_final_factor",
        "large_steps_lambda",
        "reduction_temperature",
        "voxel_pitch_divisor",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.spp_grad < 0:
        parser.error("--spp-grad must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if args.initial_subdivisions < 0:
        parser.error("--initial-subdivisions must be non-negative")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be non-negative")
    if args.phases is not None and args.phases <= 0:
        parser.error("--phases must be positive")
    if args.remesh_level is not None and args.remesh_level <= 0:
        parser.error("--remesh-level must be positive")
    if args.remesh_interval is not None and args.remesh_interval <= 0:
        parser.error("--remesh-interval must be positive")
    if args.evaluate_reference and args.model_num > 3:
        parser.error("--evaluate-reference is only available for models 1-3")

    return RunConfiguration(
        model_num=args.model_num,
        data_source=args.data_source,
        curve_type=args.curve_type,
        initial_mesh=args.initial_mesh,
        output_dir=args.output_dir,
        variant=args.variant,
        cuda_device=args.cuda_device,
        phases=args.phases,
        width=args.width,
        height=args.height,
        spp=args.spp,
        spp_grad=args.spp_grad,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_final_factor=args.lr_final_factor,
        large_steps_lambda=args.large_steps_lambda,
        seed=args.seed,
        validation_interval=args.validation_interval,
        validation_spp=args.validation_spp,
        early_stop_patience=args.early_stop_patience,
        initial_subdivisions=args.initial_subdivisions,
        initial_face_count=args.initial_face_count,
        checkpoint_interval=args.checkpoint_interval,
        reduction_temperature=args.reduction_temperature,
        evaluate_reference=args.evaluate_reference,
        voxel_metrics=args.voxel_metrics,
        voxel_pitch_divisor=args.voxel_pitch_divisor,
        render_movies=args.render_movies,
        movie_observer=args.movie_observer,
        movie_phases=args.movie_phases,
        movie_width=args.movie_width,
        movie_height=args.movie_height,
        movie_spp=args.movie_spp,
        movie_fps=args.movie_fps,
        remesh_level=args.remesh_level,
        remesh_interval=args.remesh_interval,
    )


def initialize_mitsuba(configuration: RunConfiguration):
    """Select the runtime before importing anything from ``hac.mitsuba``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = configuration.cuda_device
    import mitsuba as mi

    mi.set_variant(configuration.variant)
    return mi


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2) + "\n")


def _resolve_output_dir(configuration: RunConfiguration) -> Path:
    if configuration.output_dir is not None:
        return configuration.output_dir.resolve()
    from hac.constants import WORKSPACE_DIR

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    return (
        WORKSPACE_DIR
        / "optimize_shape_v5"
        / (
            f"model{configuration.model_num}_{configuration.data_source}_"
            f"{configuration.curve_type}_{timestamp}"
        )
    )


def _remesh_botsch(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Botsch-Kobbelt isotropic remesh (gpytoolbox), as in the Large Steps tutorial.

    Target edge length is ``_REMESH_EDGE_FACTOR`` times the mesh's current mean
    edge length (0.5 roughly quadruples the face count).
    """
    try:
        from gpytoolbox import remesh_botsch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "remeshing requires the 'gpytoolbox' package (pip install gpytoolbox)."
        ) from exc

    edge_lengths = np.concatenate(
        [
            np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1),
            np.linalg.norm(vertices[faces[:, 2]] - vertices[faces[:, 1]], axis=1),
            np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=1),
        ]
    )
    target_edge_length = float(np.mean(edge_lengths)) * _REMESH_EDGE_FACTOR
    new_vertices, new_faces = remesh_botsch(
        vertices.astype(np.float64),
        faces.astype(np.int32),
        i=_REMESH_BOTSCH_ITERATIONS,
        h=target_edge_length,
        project=True,
    )
    return new_vertices.astype(np.float32), new_faces.astype(np.int64)


def _should_remesh(
    configuration: RunConfiguration,
    iteration: int,
    loss_history: list[float],
    last_remesh_iteration: int,
    event_count: int,
) -> bool:
    """True if a remesh should be triggered after ``iteration`` (1-based)."""
    if not configuration.remesh_enabled:
        return False
    if event_count >= configuration.remesh_max_events:
        return False
    if configuration.remesh_interval is not None:  # fixed schedule
        return iteration - last_remesh_iteration >= configuration.remesh_interval
    # --remesh-level: remesh when the training loss stagnates (or rises)
    if iteration - last_remesh_iteration < _REMESH_ADAPTIVE_COOLDOWN:
        return False
    if iteration < _REMESH_ADAPTIVE_WINDOW:
        return False
    window_start_loss = loss_history[-_REMESH_ADAPTIVE_WINDOW]
    relative_improvement = (window_start_loss - loss_history[-1]) / max(
        window_start_loss, 1e-12
    )
    return relative_improvement < _REMESH_ADAPTIVE_MIN_IMPROVEMENT


def _resample_to_face_count(
    mesh: trimesh.Trimesh,
    face_count: int,
) -> trimesh.Trimesh:
    """Bring ``mesh`` to approximately ``face_count`` triangles.

    Loop-subdivide (~4x faces per pass) while the mesh is coarser than the
    target, then quadric-decimate down onto it. Both halves are no-ops for a
    mesh that already sits at ``face_count``, so this is safe to apply
    unconditionally.

    The optimizer's face budget is a deliberate choice - coarse starts let
    --remesh-level add detail only where the loss asks for it - so warm-start
    meshes are held to it just like the generated ellipsoid.

    Decimating onto a count that is not a clean subdivision of the source can
    leave holes: trimesh's quadric decimation is not manifold-preserving, and the
    subdivide-then-decimate path (face_count above the source density) is the
    usual way to hit that. A non-watertight start is warned about below rather
    than rejected - the Large Steps parameterization still runs on it, but the
    solve is better behaved on a closed surface.
    """
    was_watertight = mesh.is_watertight
    source_face_count = len(mesh.faces)

    while len(mesh.faces) < face_count:
        mesh = mesh.subdivide()
    subdivided_face_count = len(mesh.faces)
    if len(mesh.faces) > face_count:
        mesh = mesh.simplify_quadric_decimation(face_count=face_count)

    if was_watertight and not mesh.is_watertight:
        upsampled = subdivided_face_count > source_face_count
        warnings.warn(
            f"Resampling {source_face_count} faces to {face_count} produced a "
            "non-watertight mesh (quadric decimation is not manifold-preserving)"
            + (
                f"; --initial-face-count is above the source density, so the mesh "
                f"was subdivided to {subdivided_face_count} faces first and then "
                "decimated back down, which is the step that tends to open holes. "
                "Lowering --initial-face-count to at or below the source density "
                "avoids it, as does letting --remesh-level add the detail instead."
                if upsampled
                else ". Try a face count closer to the source density."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return mesh


def _prepare_initial_mesh(
    configuration: RunConfiguration,
    cylinder_radius: float,
) -> trimesh.Trimesh:
    from hac.mesh import (
        construct_initial_ellipsoid_mesh,
        normalize_mesh_to_challenge_cylinder,
    )

    if configuration.initial_mesh is not None:
        if not configuration.initial_mesh.is_file():
            raise FileNotFoundError(
                f"Initial mesh not found: {configuration.initial_mesh}"
            )
        mesh = trimesh.load_mesh(configuration.initial_mesh, force="mesh")
    else:
        # --initial-subdivisions only shapes the generated icosphere; a
        # warm-start mesh arrives with whatever tessellation it was built with.
        mesh = construct_initial_ellipsoid_mesh(
            radius=cylinder_radius,
            subdivisions=configuration.initial_subdivisions,
        )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Initial mesh must be a non-empty triangle mesh.")
    mesh = _resample_to_face_count(mesh, configuration.initial_face_count)
    return normalize_mesh_to_challenge_cylinder(mesh, target_R=cylinder_radius)


def _build_forward(
    mesh_path: Path,
    configuration: RunConfiguration,
    *,
    n_phases: int,
    width: int | None = None,
    height: int | None = None,
    spp: int | None = None,
):
    from hac.constants import OBSERVERS
    from hac.mitsuba.forward import ForwardModel
    from hac.mitsuba.mesh import MitsubaMesh
    from hac.mitsuba.scene import RenderingSettings

    return ForwardModel(
        MitsubaMesh(mesh_path),
        n_phases=n_phases,
        observers=dict(OBSERVERS),
        rendering_settings=RenderingSettings(
            width=configuration.width if width is None else width,
            height=configuration.height if height is None else height,
            spp=configuration.spp if spp is None else spp,
        ),
        normalize_mesh=False,
        integrator=INTEGRATOR,
        projection=PROJECTION,
    )


def _build_optimization_state(
    mesh_path: Path,
    configuration: RunConfiguration,
    *,
    n_phases: int,
):
    """(Re)build the forward model, Large Steps parametrization and Adam optimizer.

    Called for the initial mesh and again after every remesh: Large Steps and
    Adam are tied to a fixed vertex/face count, so a topology change rebuilds
    both from scratch. Returns ``(forward, faces, large_steps, optimizer)``.
    """
    import mitsuba as mi

    forward = _build_forward(mesh_path, configuration, n_phases=n_phases)
    # Read faces before keep() drops the face parameter from params.
    faces = _faces_numpy(forward)
    large_steps = mi.ad.LargeSteps(
        forward.params[forward.position_key],
        forward.params[forward.face_key],
        configuration.large_steps_lambda,
    )
    # Faces and the batch scene's thousands of static sensor/instance parameters
    # do not change between remeshes; excluding them keeps mi.render() from
    # traversing them every iteration. Flush access-tracking candidates first so
    # keep() cannot leave a removed parameter pending for the next update().
    forward.params.update()
    forward.params.keep([forward.position_key])

    optimizer = mi.ad.Adam(lr=configuration.learning_rate, uniform=True)
    optimizer["u"] = large_steps.to_differential(forward.params[forward.position_key])
    return forward, faces, large_steps, optimizer


def _configure_reduction(
    forward,
    configuration: RunConfiguration,
    *,
    binary_thresholds: np.ndarray | None = None,
) -> np.ndarray | None:
    import drjit as dr
    import mitsuba as mi

    from hac.mitsuba.forward import (
        calibrate_binary_thresholds,
        reduction_binary_smooth,
        reduction_intensity_smooth,
    )

    if configuration.curve_type == "intensity":
        forward.reduction_function = lambda frames: reduction_intensity_smooth(
            frames,
            temperature=configuration.reduction_temperature,
        )
        return None

    if binary_thresholds is None:
        with dr.suspend_grad():
            initial_frames = forward.render(
                seed=configuration.seed,
                spp=configuration.validation_spp,
            )
        binary_thresholds = calibrate_binary_thresholds(initial_frames)
    # One calibrated threshold per observer, broadcast over (phase, y, x, channel).
    # Built once here so it stays a constant inside the frozen training step.
    threshold_tensor = mi.TensorXf(binary_thresholds.reshape(-1, 1, 1, 1, 1))
    forward.reduction_function = lambda frames: reduction_binary_smooth(
        frames,
        threshold=threshold_tensor,
        temperature=configuration.reduction_temperature,
    )
    return binary_thresholds


def _positions_numpy(forward) -> np.ndarray:
    positions = np.asarray(forward.params[forward.position_key], dtype=np.float32)
    return positions.reshape(-1, 3).copy()


def _faces_numpy(forward) -> np.ndarray:
    faces = np.asarray(forward.params[forward.face_key], dtype=np.int64)
    return faces.reshape(-1, 3).copy()


def _export_mesh(path: Path, positions: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=positions, faces=faces, process=False).export(path)


def _voxel_measures(
    reference_mesh: trimesh.Trimesh,
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    pitch: float,
) -> tuple[float, float]:
    """FIPS voxel volume difference between the current shape and the reference.

    Both measures are zero for identical shapes and grow towards 1 as the shapes
    diverge. The reference is passed in already loaded: re-reading a dense STL
    dominates the runtime of the metric otherwise.
    """
    from hac.fips.metrics import relative_volume_difference_voxelized

    measure1, measure2 = relative_volume_difference_voxelized(
        reference_mesh,
        trimesh.Trimesh(vertices=positions, faces=faces, process=False),
        pitch=pitch,
    )
    return float(measure1), float(measure2)


def _best_voxel_measure(
    history: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    """Lowest (best) entry of ``history`` for one measure, with its iteration."""
    if not history:
        return None
    best = min(history, key=lambda entry: entry[key])
    return {"iteration": best["iteration"], "value": best[key]}


def _apply_optimizer_positions(
    forward,
    optimizer,
    large_steps,
    cylinder_radius: float,
):
    from hac.mitsuba.mesh import normalize_positions_to_challenge_cylinder

    positions = normalize_positions_to_challenge_cylinder(
        large_steps.from_differential(optimizer["u"]),
        cylinder_radius,
    )
    forward.params[forward.position_key] = positions
    forward.params.update()
    return positions


def _reduce_lightcurves_and_backward(
    frames,
    reduction_function,
    target,
):
    """Reduce rendered frames and backpropagate within a frozen AD scope.

    ``frames`` retains the external render graph, so backward propagation from
    the frozen reduction still reaches the current Large Steps variable.
    """
    import drjit as dr

    curves = reduction_function(frames)
    loss = dr.mean(dr.square(curves - target))

    # Frozen functions have an isolated gradient scope, so backward must be
    # part of the recording rather than applied to the returned loss later.
    dr.backward(loss)
    return loss


def _evaluate(
    forward,
    target: np.ndarray,
    *,
    seed: int,
    spp: int,
) -> tuple[float, np.ndarray]:
    import drjit as dr

    with dr.suspend_grad():
        curves = np.asarray(forward.render_lightcurves(seed=seed, spp=spp)).copy()
    return float(np.mean(np.square(curves - target))), curves


def _check_finite(name: str, values: Any) -> None:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains non-finite values.")


def _compare_reloaded_checkpoint_loss(
    reloaded_loss: float,
    in_memory_loss: float,
    *,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> dict[str, float | bool]:
    """Compare checkpoint renders without invalidating a completed run."""
    absolute_difference = abs(reloaded_loss - in_memory_loss)
    relative_difference = absolute_difference / max(abs(in_memory_loss), 1e-15)
    matches = math.isclose(
        reloaded_loss,
        in_memory_loss,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )
    if not matches:
        warnings.warn(
            "Fresh ForwardModel loss differs from the in-memory best checkpoint: "
            f"{reloaded_loss} versus {in_memory_loss} "
            f"(absolute difference {absolute_difference:.6g}, "
            f"relative difference {relative_difference:.3%}). Continuing with "
            "the reloaded checkpoint; mesh serialization, visibility changes, and "
            "Monte Carlo rendering can cause small differences.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "matches_tolerance": matches,
        "absolute_loss_difference": absolute_difference,
        "relative_loss_difference": relative_difference,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


def _append_history(file, entry: Mapping[str, Any]) -> None:
    file.write(json.dumps(_json_value(entry)) + "\n")
    file.flush()


def _save_diagnostic_views(
    mesh_path: Path,
    output_path: Path,
    configuration: RunConfiguration,
    *,
    title: str,
) -> None:
    from hac.constants import OBSERVERS
    from hac.mitsuba.forward import ForwardModel
    from hac.mitsuba.mesh import MitsubaMesh
    from hac.mitsuba.scene import RenderingSettings
    from hac.mitsuba.visualize import render_forward_views

    names = ("0_top", "0_bottom", "0_hor1")
    observers = {name: OBSERVERS[name] for name in names}
    diagnostic = ForwardModel(
        MitsubaMesh(mesh_path),
        n_phases=1,
        observers=observers,
        rendering_settings=RenderingSettings(
            width=max(128, configuration.width),
            height=max(128, configuration.height),
            spp=configuration.validation_spp,
        ),
        normalize_mesh=False,
        projection=PROJECTION,
        integrator=INTEGRATOR,
    )
    figure, _ = render_forward_views(
        diagnostic,
        title=title,
        output_path=output_path,
        seed=configuration.seed,
        spp=configuration.validation_spp,
    )
    import matplotlib.pyplot as plt

    plt.close(figure)


def _movie_frames(
    mesh_path: Path,
    configuration: RunConfiguration,
):
    from hac.constants import OBSERVERS
    from hac.mitsuba.forward import ForwardModel
    from hac.mitsuba.mesh import MitsubaMesh
    from hac.mitsuba.scene import RenderingSettings
    from hac.mitsuba.visualize import bitmap_frames

    if "/" in configuration.movie_observer:
        source, observer_name = configuration.movie_observer.split("/", 1)
        if source not in configuration.data_sources:
            raise ValueError(f"Movie source {source!r} is not active in this run.")
    else:
        observer_name = configuration.movie_observer
    if observer_name not in OBSERVERS:
        raise ValueError(f"Unknown movie observer {observer_name!r}.")
    forward = ForwardModel(
        MitsubaMesh(mesh_path),
        n_phases=configuration.movie_phases,
        observers={observer_name: OBSERVERS[observer_name]},
        rendering_settings=RenderingSettings(
            width=configuration.movie_width,
            height=configuration.movie_height,
            spp=configuration.movie_spp,
        ),
        normalize_mesh=False,
        projection=PROJECTION,
        integrator=INTEGRATOR,
    )
    return bitmap_frames(
        forward.render(seed=configuration.seed, spp=configuration.movie_spp)[0]
    )


def _render_movies(
    paths: RunPaths,
    configuration: RunConfiguration,
    reference_available: bool,
) -> dict[str, str]:
    from hac.mitsuba.visualize import write_movie

    initial = _movie_frames(paths.initial_mesh, configuration)
    best = _movie_frames(paths.best_mesh, configuration)
    panels = [initial, best]
    write_movie(paths.initial_movie, initial, configuration.movie_fps)
    write_movie(paths.best_movie, best, configuration.movie_fps)
    artifacts = {
        "initial_movie": str(paths.initial_movie),
        "best_movie": str(paths.best_movie),
    }
    if reference_available:
        reference = _movie_frames(
            paths.normalized_reference_mesh,
            configuration,
        )
        panels.insert(0, reference)
        write_movie(paths.reference_movie, reference, configuration.movie_fps)
        artifacts["reference_movie"] = str(paths.reference_movie)
    comparison = np.concatenate(panels, axis=2)
    write_movie(paths.comparison_movie, comparison, configuration.movie_fps)
    artifacts["comparison_movie"] = str(paths.comparison_movie)
    return artifacts


def run(configuration: RunConfiguration) -> dict[str, Any]:
    mi = initialize_mitsuba(configuration)
    import drjit as dr

    from hac.constants import MODEL_BASE_RADII, OBSERVERS
    from hac.data import get_model_stl_path, prepare_lightcurve_targets
    from hac.mesh import normalize_mesh_to_challenge_cylinder
    from hac.mitsuba.scene import DEFAULT_BSDF
    from hac.visualize import (
        plot_lightcurve_comparison,
        plot_optimization_loss,
        plot_voxel_measures,
    )

    output_dir = _resolve_output_dir(configuration)
    paths = RunPaths.from_output_dir(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    cylinder_radius = MODEL_BASE_RADII[configuration.model_num]
    targets = prepare_lightcurve_targets(
        model_num=configuration.model_num,
        data_source=configuration.data_source,
        curve_type=configuration.curve_type,
        n_phases=configuration.phases,
    )
    # One render phase per kept light curve frame (index-subsampled, not resampled).
    n_phases = targets.shape[1]
    # dr.freeze is used only for the intensity reduction (the binary reduction
    # closes over a threshold tensor freeze cannot traverse) and only without
    # remeshing (a topology change would invalidate the frozen recording).
    freeze_training = (
        configuration.curve_type == "intensity" and not configuration.remesh_enabled
    )
    _write_json(
        paths.resolved_config,
        {
            **asdict(configuration),
            "resolved_output_dir": output_dir,
            "data_sources": configuration.data_sources,
            "n_phases": n_phases,
            "projection": PROJECTION,
            "integrator": INTEGRATOR,
            "frozen_training": freeze_training,
            "voxel_metrics_enabled": configuration.voxel_metrics_enabled,
        },
    )
    initial_mesh = _prepare_initial_mesh(configuration, cylinder_radius)
    initial_mesh.export(paths.initial_mesh)

    # The reference shape is needed up front for the voxel measures, and reused
    # by --evaluate-reference below rather than being loaded a second time. It is
    # only written out for --evaluate-reference, which renders it from a file; the
    # voxel measures work off the in-memory mesh and the STL is tens of megabytes.
    reference_mesh: trimesh.Trimesh | None = None
    reference_stl_path: Path | None = None
    if configuration.voxel_metrics_enabled or configuration.evaluate_reference:
        reference_stl_path = get_model_stl_path(configuration.model_num)
        reference_mesh = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(reference_stl_path, force="mesh"),
            target_R=cylinder_radius,
        )
        if configuration.evaluate_reference:
            reference_mesh.export(paths.normalized_reference_mesh)
    voxel_pitch = (
        cylinder_radius / configuration.voxel_pitch_divisor
        if configuration.voxel_metrics_enabled
        else None
    )

    forward, faces, large_steps, optimizer = _build_optimization_state(
        paths.initial_mesh, configuration, n_phases=n_phases
    )
    if tuple(forward.observer_names) != tuple(OBSERVERS):
        raise RuntimeError(
            "Target/render observer order differs: "
            f"{tuple(OBSERVERS)} versus {forward.observer_names}."
        )
    position_keys = [
        key for key in forward.params.keys() if key.endswith("vertex_positions")
    ]
    if position_keys != [forward.position_key]:
        raise RuntimeError(
            f"Expected one shared optimizable mesh; found {position_keys}."
        )
    binary_thresholds = _configure_reduction(forward, configuration)
    target_tensor = mi.TensorXf(targets.copy())

    initial_loss, initial_curves = _evaluate(
        forward,
        targets,
        seed=configuration.seed,
        spp=configuration.validation_spp,
    )
    _save_diagnostic_views(
        paths.initial_mesh,
        paths.initial_views,
        configuration,
        title=f"Model {configuration.model_num}: initial shape",
    )

    training_step = (
        dr.freeze(_reduce_lightcurves_and_backward)
        if freeze_training
        else _reduce_lightcurves_and_backward
    )
    remesh_events: list[dict[str, Any]] = []
    last_remesh_iteration = 0

    best_loss = initial_loss
    best_iteration = 0
    best_positions = _positions_numpy(forward)
    best_faces = faces.copy()
    _export_mesh(paths.best_mesh, best_positions, best_faces)
    training_losses: list[float] = []
    validation_history: list[dict[str, Any]] = [
        {"iteration": 0, "loss": initial_loss}
    ]
    voxel_history: list[dict[str, Any]] = []
    validation_checks_without_improvement = 0
    iterations_completed = 0

    initial_entry: dict[str, Any] = {
        "kind": "validation",
        "iteration": 0,
        "loss": initial_loss,
        "best": True,
    }
    if voxel_pitch is not None:
        assert reference_mesh is not None
        measure1, measure2 = _voxel_measures(
            reference_mesh,
            best_positions,
            best_faces,
            pitch=voxel_pitch,
        )
        voxel_history.append(
            {"iteration": 0, "measure1": measure1, "measure2": measure2}
        )
        initial_entry["voxel_measure1"] = measure1
        initial_entry["voxel_measure2"] = measure2
        print(
            f"Initial shape: voxel_measure1={measure1:.6f}, "
            f"voxel_measure2={measure2:.6f} (pitch={voxel_pitch:.4g})"
        )

    with paths.history.open("w") as history_file:
        _append_history(history_file, initial_entry)
        for iteration in range(1, configuration.iterations + 1):
            positions = _apply_optimizer_positions(
                forward,
                optimizer,
                large_steps,
                cylinder_radius,
            )
            _check_finite("vertex positions", positions)
            frames = forward.render(
                seed=configuration.seed + iteration,
                spp=configuration.spp,
                spp_grad=configuration.spp_grad,
            )
            loss = training_step(
                frames,
                forward.reduction_function,
                target_tensor,
            )
            dr.eval(loss)
            loss_value = float(np.asarray(loss).reshape(-1)[0])
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite training loss at iteration {iteration}."
                )
            gradient = dr.grad(optimizer["u"])
            _check_finite("Large Steps gradient", gradient)
            # Set every step: remeshing rebuilds Adam at the base learning rate.
            learning_rate = configuration.learning_rate_at(iteration)
            optimizer.set_learning_rate(learning_rate)
            optimizer.step()

            training_losses.append(loss_value)
            iterations_completed = iteration
            entry: dict[str, Any] = {
                "kind": "training",
                "iteration": iteration,
                "loss": loss_value,
                "learning_rate": learning_rate,
            }
            _append_history(history_file, entry)
            print(
                f"Iteration {iteration:04d}/{configuration.iterations}: "
                f"training_loss={loss_value:.8f}, lr={learning_rate:.3g}"
            )

            should_validate = (
                iteration % configuration.validation_interval == 0
                or iteration == configuration.iterations
            )
            should_checkpoint = (
                configuration.checkpoint_interval > 0
                and iteration % configuration.checkpoint_interval == 0
            )
            if should_validate or should_checkpoint:
                _apply_optimizer_positions(
                    forward,
                    optimizer,
                    large_steps,
                    cylinder_radius,
                )
            if should_checkpoint:
                checkpoint_path = paths.checkpoints / f"iteration_{iteration:04d}.obj"
                _export_mesh(checkpoint_path, _positions_numpy(forward), faces)

            if should_validate:
                validation_loss, _ = _evaluate(
                    forward,
                    targets,
                    seed=configuration.seed,
                    spp=configuration.validation_spp,
                )
                improved = validation_loss < best_loss
                if improved:
                    best_loss = validation_loss
                    best_iteration = iteration
                    best_positions = _positions_numpy(forward)
                    best_faces = faces.copy()
                    _export_mesh(paths.best_mesh, best_positions, best_faces)
                    validation_checks_without_improvement = 0
                else:
                    validation_checks_without_improvement += 1
                validation_entry: dict[str, Any] = {
                    "kind": "validation",
                    "iteration": iteration,
                    "loss": validation_loss,
                    "best": improved,
                }
                validation_history.append(
                    {"iteration": iteration, "loss": validation_loss}
                )
                print(
                    f"  validation_loss={validation_loss:.8f}, best={improved}"
                )
                if voxel_pitch is not None:
                    assert reference_mesh is not None
                    measure1, measure2 = _voxel_measures(
                        reference_mesh,
                        _positions_numpy(forward),
                        faces,
                        pitch=voxel_pitch,
                    )
                    voxel_history.append(
                        {
                            "iteration": iteration,
                            "measure1": measure1,
                            "measure2": measure2,
                        }
                    )
                    validation_entry["voxel_measure1"] = measure1
                    validation_entry["voxel_measure2"] = measure2
                    print(
                        f"  voxel_measure1={measure1:.6f}, "
                        f"voxel_measure2={measure2:.6f}"
                    )
                _append_history(history_file, validation_entry)
                if (
                    configuration.early_stop_patience > 0
                    and validation_checks_without_improvement
                    >= configuration.early_stop_patience
                ):
                    print(
                        "Stopping early after "
                        f"{validation_checks_without_improvement} validation checks "
                        "without improvement."
                    )
                    break

            if _should_remesh(
                configuration,
                iteration,
                training_losses,
                last_remesh_iteration,
                len(remesh_events),
            ):
                current_positions = _positions_numpy(forward)
                old_vertex_count = len(current_positions)
                old_face_count = len(faces)
                new_vertices, new_faces = _remesh_botsch(current_positions, faces)
                remesh_path = paths.checkpoints / f"remesh_iter_{iteration:04d}.obj"
                _export_mesh(remesh_path, new_vertices, new_faces)
                del forward, large_steps, optimizer
                dr.flush_malloc_cache()
                forward, faces, large_steps, optimizer = _build_optimization_state(
                    remesh_path, configuration, n_phases=n_phases
                )
                _configure_reduction(
                    forward, configuration, binary_thresholds=binary_thresholds
                )
                last_remesh_iteration = iteration
                remesh_events.append(
                    {
                        "after_iteration": iteration,
                        "vertex_count": int(len(new_vertices)),
                        "face_count": int(len(new_faces)),
                        "mesh_path": str(remesh_path),
                    }
                )
                _append_history(
                    history_file,
                    {
                        "kind": "remesh",
                        "iteration": iteration,
                        "vertex_count": int(len(new_vertices)),
                        "face_count": int(len(new_faces)),
                    },
                )
                print(
                    f"  remesh #{len(remesh_events)} after iteration {iteration}: "
                    f"{old_vertex_count} -> {len(new_vertices)} vertices, "
                    f"{old_face_count} -> {len(new_faces)} faces"
                )

    _apply_optimizer_positions(forward, optimizer, large_steps, cylinder_radius)
    last_positions = _positions_numpy(forward)
    _export_mesh(paths.last_mesh, last_positions, faces)
    _export_mesh(paths.best_mesh, best_positions, best_faces)

    frozen_training_recordings = (
        training_step.n_recordings if freeze_training else 0
    )
    del training_step
    del forward
    dr.flush_malloc_cache()
    final_forward = _build_forward(paths.best_mesh, configuration, n_phases=n_phases)
    _configure_reduction(
        final_forward,
        configuration,
        binary_thresholds=binary_thresholds,
    )
    final_loss, final_curves = _evaluate(
        final_forward,
        targets,
        seed=configuration.seed,
        spp=configuration.validation_spp,
    )
    checkpoint_reconstruction = _compare_reloaded_checkpoint_loss(
        final_loss,
        best_loss,
    )
    _save_diagnostic_views(
        paths.best_mesh,
        paths.best_views,
        configuration,
        title=f"Model {configuration.model_num}: optimized shape",
    )

    reference_loss = None
    reference_available = False
    if configuration.evaluate_reference:
        assert reference_mesh is not None  # prepared before the optimization loop
        reference_forward = _build_forward(
            paths.normalized_reference_mesh,
            configuration,
            n_phases=n_phases,
        )
        _configure_reduction(
            reference_forward,
            configuration,
            binary_thresholds=binary_thresholds,
        )
        reference_loss, _ = _evaluate(
            reference_forward,
            targets,
            seed=configuration.seed,
            spp=configuration.validation_spp,
        )
        reference_available = True
        _save_diagnostic_views(
            paths.normalized_reference_mesh,
            paths.reference_views,
            configuration,
            title=f"Model {configuration.model_num}: reference shape",
        )
        del reference_forward
        dr.flush_malloc_cache()

    plot_lightcurve_comparison(
        targets,
        initial_curves,
        final_curves,
        tuple(OBSERVERS),
        ("HAC target", "Initial shape", "Optimized shape"),
        paths.lightcurve_plot,
        title=(
            f"Model {configuration.model_num}: {configuration.data_source} "
            f"{configuration.curve_type}, validation MSE={final_loss:.6g}"
        ),
    )
    plot_optimization_loss(
        training_losses,
        paths.loss_plot,
        reference_loss=reference_loss,
        reference_label=(
            "Reference-shape forward MSE" if reference_loss is not None else None
        ),
    )
    best_voxel_measure1 = _best_voxel_measure(voxel_history, "measure1")
    best_voxel_measure2 = _best_voxel_measure(voxel_history, "measure2")
    if voxel_history:
        plot_voxel_measures(
            [entry["iteration"] for entry in voxel_history],
            [entry["measure1"] for entry in voxel_history],
            [entry["measure2"] for entry in voxel_history],
            paths.voxel_plot,
            title=(
                f"Model {configuration.model_num}: voxel volume difference "
                f"(pitch={voxel_pitch:.4g})"
            ),
        )
        for label, best in (
            ("measure1", best_voxel_measure1),
            ("measure2", best_voxel_measure2),
        ):
            assert best is not None
            print(
                f"Best voxel {label}={best['value']:.6f} "
                f"at iteration {best['iteration']}"
            )
    np.savez_compressed(
        paths.curves,
        target=targets,
        initial=initial_curves,
        best=final_curves,
        observer_names=np.asarray(tuple(OBSERVERS)),
    )

    movie_artifacts: dict[str, str] = {}
    if configuration.render_movies:
        movie_artifacts = _render_movies(
            paths,
            configuration,
            reference_available,
        )

    radial = np.linalg.norm(best_positions[:, :2], axis=1)
    summary = {
        "configuration": asdict(configuration),
        "model_num": configuration.model_num,
        "observer_names": tuple(OBSERVERS),
        "n_phases": n_phases,
        "projection": PROJECTION,
        "integrator": INTEGRATOR,
        "frozen_training": freeze_training,
        "frozen_training_recordings": frozen_training_recordings,
        "remesh": {
            "enabled": configuration.remesh_enabled,
            "mode": (
                None
                if not configuration.remesh_enabled
                else "interval"
                if configuration.remesh_interval is not None
                else "adaptive"
            ),
            "level": configuration.remesh_level,
            "interval": configuration.remesh_interval,
            "max_events": (
                configuration.remesh_max_events
                if configuration.remesh_enabled
                else 0
            ),
            "edge_factor": _REMESH_EDGE_FACTOR,
            "botsch_iterations": _REMESH_BOTSCH_ITERATIONS,
            "events": remesh_events,
        },
        "shared_mesh_parameter": final_forward.position_key,
        "shared_bsdf": DEFAULT_BSDF,
        "vertex_count": len(best_positions),
        "face_count": len(best_faces),
        "iterations_completed": iterations_completed,
        "initial_validation_loss": initial_loss,
        "best_iteration": best_iteration,
        "best_validation_loss": best_loss,
        "final_validation_loss": final_loss,
        "checkpoint_reconstruction": checkpoint_reconstruction,
        "reference_loss": reference_loss,
        "voxel_metrics": {
            "enabled": configuration.voxel_metrics_enabled,
            "pitch": voxel_pitch,
            "pitch_divisor": configuration.voxel_pitch_divisor,
            # Normalized to the challenge cylinder in memory, exactly as the
            # optimized shape is, so the two are compared in the same frame.
            "reference_stl": (
                str(reference_stl_path) if voxel_pitch is not None else None
            ),
            "best_measure1": best_voxel_measure1,
            "best_measure2": best_voxel_measure2,
            # The lowest-loss shape is not necessarily the closest one by volume.
            "at_best_loss_iteration": next(
                (
                    entry
                    for entry in voxel_history
                    if entry["iteration"] == best_iteration
                ),
                None,
            ),
            "history": voxel_history,
        },
        "training_loss_history": training_losses,
        "validation_history": validation_history,
        "binary_thresholds": (
            None
            if binary_thresholds is None
            else {
                name: float(binary_thresholds[index])
                for index, name in enumerate(OBSERVERS)
            }
        ),
        "constraint": {
            "target_radius": cylinder_radius,
            "radial_max": float(np.max(radial)),
            "z_min": float(np.min(best_positions[:, 2])),
            "z_max": float(np.max(best_positions[:, 2])),
        },
        "artifacts": {
            "resolved_config": str(paths.resolved_config),
            "initial_mesh": str(paths.initial_mesh),
            "best_mesh": str(paths.best_mesh),
            "last_mesh": str(paths.last_mesh),
            "history": str(paths.history),
            "curves": str(paths.curves),
            "lightcurve_plot": str(paths.lightcurve_plot),
            "loss_plot": str(paths.loss_plot),
            "voxel_plot": str(paths.voxel_plot) if voxel_history else None,
            "initial_views": str(paths.initial_views),
            "best_views": str(paths.best_views),
            "reference_mesh": (
                str(paths.normalized_reference_mesh) if reference_available else None
            ),
            "reference_views": (
                str(paths.reference_views) if reference_available else None
            ),
            **movie_artifacts,
        },
        "elapsed_time_seconds": time.perf_counter() - start_time,
    }
    _write_json(paths.summary, summary)
    print(json.dumps(_json_value(summary), indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    configuration = parse_args(argv)
    run(configuration)


if __name__ == "__main__":
    main()
