"""End-to-end GPU inversion from challenge lightcurves."""

from __future__ import annotations

import json
import hashlib
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import trimesh

from forward_validation import load_benchmark_mesh
from pytorch3d_forward import (
    ObservationBatch,
    OctantoidLightcurveModel,
    PyTorch3DLightcurveModel,
    RendererConfig,
)
from shape_model import (EllipsoidInitialization, OctantoidShapeModel,
                         RadialHarmonicShapeModel, sphere_template)

from .calibration import Calibration, phase_degrees, curve_metrics
from .provenance import code_identity, dataset_identity, fingerprint
from .datasets import AsteroidDataset, load_lightcurves
from .reporting import (
    align_and_score_public_meshes,
    curve_report,
    export_mesh,
    plot_convergence,
    shape_report,
)
from .trajectory import FitTrajectory


@dataclass(frozen=True)
class FitConfig:
    asteroid: int
    origin: str = "simulated"
    curve_type: str = "intensity"
    steps: int = 300
    learning_rate: float = 2e-3
    initial_learning_rate: float = 8e-3
    min_learning_rate: float = 1e-5
    escape_steps: int = 30
    stabilization_steps: int = 50
    frame_count: int = 12
    validation_frame_count: int = 24
    validation_interval: int = 10
    evaluation_stride: int = 1
    subdivisions: int = 3
    faces: int | None = None
    representation: str = "octantoid"
    initialization: str = "ellipsoid"
    lmax: int = 6
    image_size: int = 96
    regularization_weight: float = 5e-3
    regularization_denominator: float | None = None
    degree_power: float = 2.0
    gradient_clip: float = 10.0
    seed: int = 0
    device: str = "cuda"
    checkpoint_interval: int = 25
    schema_version: int = 3
    training_block_frames: int = 1
    initialization_noise: float = 0.0
    calibration_transfer: bool = False
    train_indices: tuple[int, ...] = ()
    validation_indices: tuple[int, ...] = ()
    early_stopping: bool = False
    plateau_min_steps: int = 200
    plateau_patience: int = 100
    plateau_confirmation_steps: int = 40
    plateau_relative_delta: float = 0.005
    plateau_absolute_delta: float = 0.0001
    plateau_lr_factor: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_indices", tuple(self.train_indices))
        object.__setattr__(self, "validation_indices", tuple(self.validation_indices))
        if self.schema_version != 3:
            raise ValueError("unsupported fit schema")
        if self.representation not in ("radial_sh", "octantoid"):
            raise ValueError("representation must be radial_sh or octantoid")
        if self.initialization not in ("sphere", "ellipsoid"):
            raise ValueError("initialization must be sphere or ellipsoid")
        if self.initialization == "sphere" and self.initialization_noise:
            raise ValueError("sphere initialization requires zero initialization noise")
        if self.faces is not None and (isinstance(self.faces, bool) or not isinstance(self.faces, int)
                                       or self.faces < 8 or self.faces % 2):
            raise ValueError("faces must be an even integer of at least 8")
        if self.lmax < 0 or (self.representation == "radial_sh" and self.lmax == 0):
            raise ValueError("invalid harmonic degree")
        if self.regularization_denominator is not None and (
                not math.isfinite(self.regularization_denominator) or self.regularization_denominator <= 0):
            raise ValueError("regularization denominator must be positive and finite")
        if self.training_block_frames <= 0 or self.initialization_noise < 0:
            raise ValueError("invalid chunk size or initialization noise")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not 0 <= self.escape_steps < self.steps:
            raise ValueError("escape_steps must be in 0..steps-1")
        if not 0 <= self.stabilization_steps <= self.steps - self.escape_steps:
            raise ValueError("stabilization_steps leaves no refinement phase")
        if min(self.learning_rate, self.initial_learning_rate, self.min_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.validation_interval <= 0 or self.validation_frame_count <= 0:
            raise ValueError("validation settings must be positive")
        if self.regularization_weight < 0 or self.gradient_clip <= 0:
            raise ValueError("regularization and clipping settings are invalid")
        if (min(self.plateau_min_steps, self.plateau_patience, self.plateau_confirmation_steps) <= 0
                or not 0 < self.plateau_lr_factor < 1
                or not math.isfinite(self.plateau_relative_delta)
                or not math.isfinite(self.plateau_absolute_delta)
                or min(self.plateau_relative_delta, self.plateau_absolute_delta) < 0):
            raise ValueError("invalid plateau settings")

    @classmethod
    def smoke(cls, asteroid: int, device: str = "cpu") -> "FitConfig":
        return cls(
            asteroid=asteroid,
            steps=2,
            frame_count=2,
            validation_frame_count=2,
            validation_interval=1,
            escape_steps=1,
            stabilization_steps=0,
            evaluation_stride=60,
            subdivisions=1,
            lmax=2,
            image_size=32,
            device=device,
            checkpoint_interval=1,
        )


def _sample_indices(length: int, count: int) -> np.ndarray:
    if count <= 0 or count > length:
        raise ValueError(f"frame count must be in 1..{length}")
    return np.linspace(0, length, count, endpoint=False, dtype=np.int64)


def rotation_split(
    length: int, train_count: int, validation_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint phase-interleaved train/validation frames."""

    train = _sample_indices(length, train_count)
    if validation_count <= 0 or train_count + validation_count > length:
        raise ValueError("train and validation frames must fit in the dataset")
    if length == 360 and train_count in (12, 72) and validation_count == 72:
        return train, np.arange(2, 360, 5, dtype=np.int64)
    candidates = np.setdiff1d(np.arange(length, dtype=np.int64), train)
    positions = (
        (np.arange(validation_count, dtype=np.float64) + 0.5)
        * len(candidates)
        / validation_count
    ).astype(np.int64)
    validation = candidates[np.clip(positions, 0, len(candidates) - 1)]
    if len(np.unique(validation)) != validation_count:
        raise RuntimeError("could not construct distinct validation frames")
    return train, validation


@dataclass(frozen=True)
class OptimizationState:
    learning_rate: float
    maximum_degree: int | None
    regularization_multiplier: float
    stage: str


@dataclass
class PlateauState:
    """Validation-only stopping state, independent of selected-best tracking."""

    reference_rmse: float | None = None
    reference_step: int = 0
    confirmation_start: int | None = None
    learning_rate_ceiling: float | None = None
    stop_reason: str | None = None

    def learning_rate(self, scheduled: float) -> float:
        return min(scheduled, self.learning_rate_ceiling) if self.learning_rate_ceiling is not None else scheduled

    def observe(self, config: FitConfig, step: int, rmse: float, learning_rate: float) -> str | None:
        if not math.isfinite(rmse):
            raise ValueError("nonfinite plateau validation score")
        if not config.early_stopping or self.stop_reason is not None:
            return None
        # The iteration budget wins even if confirmation finishes on this update.
        if step >= config.steps:
            self.stop_reason = "iteration_cap"
            return self.stop_reason
        if step < config.escape_steps:
            return None
        if self.reference_rmse is None:
            self.reference_rmse, self.reference_step = rmse, step
            return "reference_initialized"
        threshold = max(config.plateau_absolute_delta,
                        config.plateau_relative_delta * self.reference_rmse)
        if self.reference_rmse - rmse >= threshold:
            recovered = self.confirmation_start is not None
            self.reference_rmse, self.reference_step = rmse, step
            self.confirmation_start = None
            return "confirmation_recovered" if recovered else "meaningful_improvement"
        if self.confirmation_start is not None:
            if step - self.confirmation_start >= config.plateau_confirmation_steps:
                self.stop_reason = "validation_plateau"
                return self.stop_reason
        elif step >= config.plateau_min_steps and step - self.reference_step >= config.plateau_patience:
            self.confirmation_start = step
            self.learning_rate_ceiling = max(config.min_learning_rate,
                                            learning_rate * config.plateau_lr_factor)
            return "confirmation_started"
        return None


def optimization_state(config: FitConfig, step: int) -> OptimizationState:
    """Return the staged optimizer settings for a zero-based update index."""

    if not 0 <= step < config.steps:
        raise ValueError("step is outside the configured optimization range")
    if step < config.escape_steps:
        fraction = step / max(config.escape_steps - 1, 1)
        blend = 0.5 * (1.0 + math.cos(math.pi * fraction))
        learning_rate = config.learning_rate + (
            config.initial_learning_rate - config.learning_rate
        ) * blend
        return OptimizationState(learning_rate, 2, 0.25, "escape")

    progress = (step - config.escape_steps) / max(
        config.steps - config.escape_steps - 1, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    learning_rate = config.min_learning_rate + (
        config.learning_rate - config.min_learning_rate
    ) * cosine
    stabilization_start = config.steps - config.stabilization_steps
    if config.stabilization_steps and step >= stabilization_start:
        stabilization_progress = (step - stabilization_start + 1) / config.stabilization_steps
        return OptimizationState(
            learning_rate, None, 1.0 + stabilization_progress, "stabilization"
        )
    return OptimizationState(learning_rate, None, 1.0, "refinement")


def _normalized(values: np.ndarray) -> np.ndarray:
    return values / np.clip(values.mean(axis=0, keepdims=True), 1e-9, None)


def _rotation_degrees(dataset, indices, convention="frame_label"):
    return phase_degrees(dataset, indices, convention)


def _initial_shape(
    shape: OctantoidShapeModel,
    dataset: AsteroidDataset,
    indices: np.ndarray | None = None,
) -> torch.Tensor:
    equatorial = [i for i, name in enumerate(dataset.columns) if "hor" in name]
    curves = dataset.curves[:, equatorial]
    if indices is not None:
        curves = curves[np.asarray(indices, dtype=np.int64)]
    hi = np.quantile(curves, 0.95, axis=0)
    lo = np.clip(np.quantile(curves, 0.05, axis=0), 1e-3, None)
    elongation = float(np.clip(np.median(hi / lo), 1.0, 2.5))
    return shape.initial_parameters(
        EllipsoidInitialization(a=elongation, b=1.0, c=0.85, theta=0.0)
    )


def _renderer_config(config: FitConfig) -> RendererConfig:
    face_count = config.faces if config.faces is not None else 20 * (4 ** config.subdivisions)
    return RendererConfig(
        image_size=config.image_size,
        shadow_size=config.image_size,
        faces_per_pixel=1 if config.image_size <= 32 else 4,
        batch_size=1 if config.image_size <= 32 else 16,
        face_limit=face_count,
        intensity_slope=8.0,
        binary_slope=8.0,
    )


def _build_model(config: FitConfig, device: torch.device, calibration: Calibration | None = None):
    sphere = sphere_template(subdivisions=config.subdivisions, faces=config.faces)
    directions = torch.as_tensor(
        np.asarray(sphere.vertices), dtype=torch.float32, device=device
    )
    faces = torch.as_tensor(sphere.faces, dtype=torch.long, device=device)
    shape_type = RadialHarmonicShapeModel if config.representation == "radial_sh" else OctantoidShapeModel
    shape = shape_type(
        directions,
        faces,
        lmax=config.lmax,
        r0=0.5,
        lam_shape=1.0,
        degree_power=config.degree_power,
        check_self_intersections=False,
    )
    renderer_config = replace(_renderer_config(config), face_limit=len(faces))
    if calibration is not None:
        renderer_config = calibration.renderer_config(renderer_config)
    renderer = PyTorch3DLightcurveModel(renderer_config).to(device)
    return shape, OctantoidLightcurveModel(shape, renderer).to(device)


def spectral_penalty(prior: torch.Tensor, config: FitConfig) -> torch.Tensor:
    """Keep the coefficient penalty independent of dimension when requested."""
    denominator = config.regularization_denominator
    return prior.square().mean() if denominator is None else prior.square().sum() / denominator


@torch.no_grad()
def _evaluate(
    model: OctantoidLightcurveModel,
    parameters: torch.Tensor,
    dataset: AsteroidDataset,
    calibration: Calibration,
    stride: int = 1,
    indices: np.ndarray | None = None,
    block_frames: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    selected = (np.arange(0, len(dataset.frames), stride, dtype=np.int64)
                if indices is None else np.asarray(indices, dtype=np.int64))
    # Always normalize over the full rotation before selecting scored samples.
    indices = np.arange(len(dataset.frames), dtype=np.int64)
    raw_blocks = []
    for start in range(0, len(indices), block_frames):
        block = indices[start : start + block_frames]
        angles = torch.as_tensor(
            np.radians(_rotation_degrees(dataset, block, calibration.phase_convention)), dtype=torch.float32,
            device=parameters.device,
        )
        observations = ObservationBatch.challenge(angles, dataset.columns)
        prediction = model(
            parameters,
            observations,
            extraction="exact",
            camera_distance_radii=calibration.camera_distance_radii,
            lambertian_fraction=calibration.lambertian_fraction,
            compute_binary=dataset.curve_type == "binary",
        )
        raw = prediction.raw_binary if dataset.curve_type == "binary" else prediction.raw_intensity
        raw_blocks.append(raw.cpu().numpy())
    return selected, _normalized(np.concatenate(raw_blocks, axis=0))[selected]


def _validation_rmse(
    model: OctantoidLightcurveModel,
    parameters: torch.Tensor,
    dataset: AsteroidDataset,
    calibration: Calibration,
    indices: np.ndarray,
) -> float:
    _, prediction = _evaluate(
        model, parameters, dataset, calibration, indices=indices
    )
    observed = dataset.curves[indices]
    score = float(np.sqrt(np.mean((prediction - observed) ** 2)))
    if not math.isfinite(score):
        raise RuntimeError("non-finite validation score")
    return score



def two_pass_backward(render_raw, blocks, target):
    """Exact chain rule through complete-set normalization, bounded renderer graphs.

    render_raw(block) must be deterministic. The small raw array is the only
    first-pass graph; the renderer is recomputed one block at a time.
    """
    with torch.no_grad():
        raw = torch.cat([render_raw(block) for block in blocks], dim=0)
    raw.requires_grad_(True)
    normalized = raw / raw.mean(dim=0, keepdim=True).clamp_min(1e-9)
    loss = (normalized - target).square().mean()
    derivative, = torch.autograd.grad(loss, raw)
    if not torch.isfinite(loss) or not torch.isfinite(derivative).all():
        raise RuntimeError("non-finite brightness objective/derivative")
    offset = 0
    for block in blocks:
        rendered = render_raw(block)
        count = rendered.shape[0]
        rendered.backward(derivative[offset:offset+count])
        offset += count
    return loss.detach()


def objective_identity(config, calibration, dataset, train, validation, initial, shape=None):
    if shape is None:
        template = sphere_template(subdivisions=config.subdivisions, faces=config.faces)
        vertices, faces = np.asarray(template.vertices), np.asarray(template.faces)
    else:
        vertices = shape.directions.detach().cpu().numpy()
        faces = shape.faces.detach().cpu().numpy()
    template_hash = hashlib.sha256(np.asarray(vertices, dtype="<f4").tobytes()
                                   + np.asarray(faces, dtype="<i8").tobytes()).hexdigest()
    return fingerprint(dict(config=asdict(config), calibration=asdict(calibration),
                            template_sha256=template_hash,
                            dataset=dataset_identity(dataset), code=code_identity(),
                            renderer=asdict(calibration.renderer_config(_renderer_config(config))),
                            train=train.tolist(), validation=validation.tolist(),
                            initial=initial.detach().cpu().tolist(),
                            normalization="train-set; validation full-turn; v3"))

def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _save_checkpoint(
    path: Path,
    step: int,
    parameters: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameter_history: list[np.ndarray],
    vertex_history: list[np.ndarray],
    losses: list[float],
    best: dict,
    validation_history: list[dict[str, float | int]],
    config: FitConfig,
    objective_fingerprint: str,
    plateau_state: PlateauState | None = None,
    optimization_history: list[dict] | None = None,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "parameters": parameters.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "parameter_history": parameter_history,
            "vertex_history": vertex_history,
            "losses": losses,
            "best": best,
            "validation_history": validation_history,
            "checkpoint_version": 3,
            "objective_fingerprint": objective_fingerprint,
            "config": asdict(config),
            "plateau_state": asdict(plateau_state) if plateau_state is not None else None,
            "optimization_history": optimization_history or [],
        },
        temporary,
    )
    temporary.replace(path)


def run_inversion(
    config: FitConfig,
    calibration: Calibration,
    output_dir: Path,
    *,
    resume: Path | None = None,
    initial_parameters: torch.Tensor | np.ndarray | None = None,
    stop_after_steps: int | None = None,
    deadline: float | None = None,
) -> dict:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Under WSL, verify nvidia-smi "
            "and /dev/dxg; do not silently fall back to CPU."
        )
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    dataset = load_lightcurves(
        config.asteroid, origin=config.origin, curve_type=config.curve_type
    )
    expected_cameras = 21 if config.origin == "simulated" else 28
    if len(dataset.columns) != expected_cameras:
        raise RuntimeError(
            f"expected all {expected_cameras} camera channels, got {len(dataset.columns)}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration.validate_scope(config.asteroid, config.calibration_transfer)
    shape, model = _build_model(config, device, calibration)
    train_indices, validation_indices = rotation_split(
        len(dataset.frames), config.frame_count, config.validation_frame_count
    )
    if config.train_indices:
        train_indices = np.asarray(config.train_indices, dtype=np.int64)
    if config.validation_indices:
        validation_indices = np.asarray(config.validation_indices, dtype=np.int64)
    if (len(train_indices) != config.frame_count or len(validation_indices) != config.validation_frame_count
            or len(np.unique(train_indices)) != len(train_indices)
            or len(np.unique(validation_indices)) != len(validation_indices)
            or np.intersect1d(train_indices, validation_indices).size
            or min(train_indices.min(), validation_indices.min()) < 0
            or max(train_indices.max(), validation_indices.max()) >= len(dataset.frames)):
        raise ValueError("invalid explicit phase partition")
    train_target = _normalized(dataset.curves[train_indices])
    target = torch.as_tensor(train_target, dtype=torch.float32, device=device)
    angles = torch.as_tensor(
        np.radians(_rotation_degrees(dataset, train_indices, calibration.phase_convention)),
        dtype=torch.float32, device=device,
    )
    observations = ObservationBatch.challenge(angles, dataset.columns)

    p0 = ((shape.initial_parameters() if config.initialization == "sphere"
           else _initial_shape(shape, dataset, train_indices)).to(device) if initial_parameters is None
          else torch.as_tensor(initial_parameters, dtype=torch.float32, device=device).clone())
    if p0.shape != (shape.n_parameters,) or not torch.isfinite(p0).all():
        raise ValueError("invalid initial shape coefficients")
    if config.initialization == "sphere" and torch.count_nonzero(p0):
        raise ValueError("sphere initialization requires zero coefficients")
    if config.initialization_noise:
        generator = torch.Generator(device=device).manual_seed(config.seed)
        degrees = shape.parameter_degrees().to(device)
        noise = torch.randn(p0.shape, generator=generator, device=device) * config.initialization_noise
        p0 = shape.project_parameters(p0 + noise * ((degrees >= 1) & (degrees <= 2)))
    identity = objective_identity(config, calibration, dataset, train_indices, validation_indices, p0, shape)
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("checkpoint_version") != 3:
            raise ValueError("legacy checkpoints cannot resume the corrected optimizer")
        if saved.get("objective_fingerprint") != identity:
            raise ValueError("checkpoint objective fingerprint does not match calibration, data, code or splits")
        del saved
    (output_dir / "manifest.json").write_text(json.dumps(dict(
        objective_fingerprint=identity, config=asdict(config), calibration=asdict(calibration),
        dataset=dataset_identity(dataset), code=code_identity(),
        train_indices=train_indices.tolist(), validation_indices=validation_indices.tolist(),
        remaining_indices=np.setdiff1d(np.arange(len(dataset.frames)),
                                       np.union1d(train_indices, validation_indices)).tolist(),
        initial_parameters=p0.detach().cpu().tolist(),
        normalization="train set; validation and reporting use full turn",
        validation_scope="conditional on calibration using public full-turn data"), indent=2))
    parameters = p0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameters], lr=config.initial_learning_rate)
    start_step = 0
    plateau = PlateauState()
    optimization_history: list[dict] = []
    parameter_history = [parameters.detach().cpu().numpy().copy()]
    vertex_history = [shape.mesh(parameters).vertices.detach().cpu().numpy().copy()]
    losses: list[float] = []
    initial_validation = _validation_rmse(
        model, parameters, dataset, calibration, validation_indices
    )
    validation_history: list[dict[str, float | int]] = [
        {"step": 0, "rmse": initial_validation}
    ]
    best = {
        "validation_rmse": initial_validation, "step": 0,
        "parameters": parameters.detach().cpu().clone(), "history_index": 0,
    }
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state.get("checkpoint_version") != 3:
            raise ValueError("legacy checkpoints cannot resume the staged optimizer")
        if state.get("objective_fingerprint") != identity:
            raise ValueError("checkpoint objective fingerprint does not match calibration, data, code or splits")
        if state.get("config") != asdict(config):
            raise ValueError("checkpoint configuration does not match this fit")
        parameters.data.copy_(state["parameters"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        parameter_history = list(state["parameter_history"])
        vertex_history = list(state["vertex_history"])
        losses = list(state["losses"])
        validation_history = list(state["validation_history"])
        best = state["best"]
        if config.early_stopping and state.get("plateau_state") is None:
            raise ValueError("early-stopping checkpoint is missing plateau state")
        plateau = PlateauState(**state["plateau_state"]) if state.get("plateau_state") else PlateauState()
        optimization_history = list(state.get("optimization_history", []))

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    blocks = [slice(i, min(i+config.training_block_frames, len(train_indices)))
              for i in range(0, len(train_indices), config.training_block_frames)]
    def render_raw(block):
        obs = ObservationBatch.challenge(observations.angles[block], dataset.columns)
        result = model(parameters, obs, extraction="soft",
                       compute_binary=config.curve_type == "binary")
        return result.raw_binary if config.curve_type == "binary" else result.raw_intensity
    parameter_degrees = shape.parameter_degrees().to(device)
    for step in range(start_step, config.steps if plateau.stop_reason is None else start_step):
        stage = optimization_state(config, step)
        effective_lr = plateau.learning_rate(stage.learning_rate)
        optimizer.param_groups[0]["lr"] = effective_lr
        optimizer.zero_grad(set_to_none=True)
        data_loss = two_pass_backward(render_raw, blocks, target)
        prior = shape.prior_residuals(parameters)
        regularization = (
            config.regularization_weight
            * stage.regularization_multiplier
            * spectral_penalty(prior, config)
        )
        loss = data_loss + regularization
        loss_value = float(loss.detach())
        regularization.backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise RuntimeError(f"non-finite parameter gradient at step {step+1}")
        if stage.maximum_degree is not None and parameters.grad is not None:
            parameters.grad[parameter_degrees > stage.maximum_degree] = 0
        torch.nn.utils.clip_grad_norm_([parameters], max_norm=config.gradient_clip)
        optimizer.step()
        with torch.no_grad():
            parameters.copy_(shape.project_parameters(parameters))
            mesh = shape.mesh(parameters)
            if not torch.isfinite(mesh.vertices).all():
                raise RuntimeError(f"non-finite mesh at optimization step {step + 1}")
        losses.append(loss_value)
        parameter_history.append(parameters.detach().cpu().numpy().copy())
        vertex_history.append(mesh.vertices.detach().cpu().numpy().copy())
        optimization_history.append(dict(step=step+1, learning_rate=effective_lr,
                                         stage=stage.stage, data_loss=float(data_loss),
                                         regularization=float(regularization.detach())))
        validation_rmse = None
        if ((step + 1) % config.validation_interval == 0 or step + 1 == config.steps
                or (config.early_stopping and step+1 == config.escape_steps)):
            validation_rmse = _validation_rmse(
                model, parameters, dataset, calibration, validation_indices
            )
            event = plateau.observe(config, step+1, validation_rmse, effective_lr)
            validation_history.append({"step": step + 1, "rmse": validation_rmse,
                                       "plateau_event": event,
                                       "learning_rate_ceiling": plateau.learning_rate_ceiling})
            if validation_rmse < float(best["validation_rmse"]):
                best = {
                    "validation_rmse": validation_rmse,
                    "step": step + 1,
                    "parameters": parameters.detach().cpu().clone(),
                    "history_index": len(parameter_history) - 1,
                }
        if ((step + 1) % config.checkpoint_interval == 0 or step + 1 == config.steps
                or plateau.stop_reason is not None):
            _save_checkpoint(
                output_dir / "checkpoint.pt", step + 1, parameters, optimizer,
                parameter_history, vertex_history, losses, best,
                validation_history, config, identity, plateau, optimization_history,
            )
        validation_text = (
            "" if validation_rmse is None else f" val={validation_rmse:.6e}"
        )
        if plateau.stop_reason is not None:
            print(f"[asteroid {config.asteroid:02d}] stopped at {step+1}: {plateau.stop_reason}{validation_text}", flush=True)
            break
        if ((stop_after_steps is not None and step+1 >= stop_after_steps)
                or (deadline is not None and time.time() >= deadline)) and step+1 < config.steps:
            _save_checkpoint(output_dir / "checkpoint.pt", step+1, parameters, optimizer,
                             parameter_history, vertex_history, losses, best,
                             validation_history, config, identity, plateau, optimization_history)
            return dict(status="paused", iterations_run=step+1,
                        runtime_seconds=time.perf_counter()-started,
                        peak_memory_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None)
        print(
            f"[asteroid {config.asteroid:02d}] step {step + 1:04d}/{config.steps} "
            f"loss={loss_value:.6e} data={float(data_loss):.6e} "
            f"reg={float(regularization):.6e} lr={effective_lr:.2e} "
            f"stage={stage.stage}{validation_text}"
        )

    with torch.no_grad():
        raw = torch.cat([render_raw(block) for block in blocks], dim=0)
        final_prediction = raw / raw.mean(dim=0, keepdim=True).clamp_min(1e-9)
        final_prior = shape.prior_residuals(parameters)
        final_state = optimization_state(config, len(losses) - 1)
        final_objective = (
            (final_prediction - target).square().mean()
            + config.regularization_weight
            * final_state.regularization_multiplier
            * spectral_penalty(final_prior, config)
        )
    trajectory_losses = np.asarray([*losses, float(final_objective)], dtype=np.float64)
    trajectory = FitTrajectory(
        np.stack(parameter_history), np.stack(vertex_history),
        shape.faces.detach().cpu().numpy(), trajectory_losses,
        best_index=int(best["history_index"]),
    )
    trajectory.save(output_dir / "trajectory.npz")
    _save_checkpoint(
        output_dir / "checkpoint.pt", len(losses), parameters, optimizer,
        parameter_history, vertex_history, losses, best, validation_history, config, identity,
        plateau, optimization_history,
    )
    faces = trajectory.faces
    parameters = best["parameters"].to(device)
    best_vertices = shape.mesh(parameters).vertices.detach().cpu().numpy()
    export_mesh(trajectory.vertices[0], faces, output_dir / "mesh_initial.stl")
    export_mesh(best_vertices, faces, output_dir / "mesh_final.stl")
    plot_convergence(trajectory_losses, output_dir / "convergence.png")

    eval_indices, initial_prediction = _evaluate(
        model, p0, dataset, calibration, config.evaluation_stride
    )
    _, fitted_prediction = _evaluate(
        model, parameters, dataset, calibration, config.evaluation_stride
    )
    observed = dataset.curves[eval_indices]
    metrics = curve_report(
        _rotation_degrees(dataset, eval_indices), observed, initial_prediction,
        fitted_prediction, dataset.columns, output_dir,
    )
    reference = None
    shape_metrics = None
    report_initial, report_fitted = trajectory.vertices[0], best_vertices
    if config.asteroid <= 3:
        proxy = load_benchmark_mesh(
            config.asteroid, faces.shape[0], device,
            output_dir / "mesh_cache", "star_proxy",
        )
        reference = (
            proxy.vertices.detach().cpu().numpy(), proxy.faces.detach().cpu().numpy()
        )
        report_initial, report_fitted, shape_metrics = align_and_score_public_meshes(
            report_initial, report_fitted, faces, reference
        )
    shape_report(
        (report_initial, faces), (report_fitted, faces),
        output_dir, reference,
    )
    full_metrics = curve_metrics(fitted_prediction, observed, dataset.columns)
    np.savez_compressed(output_dir / "curves.npz", indices=eval_indices, observed=observed,
                        initial=initial_prediction, fitted=fitted_prediction, columns=dataset.columns)
    (output_dir / "full_metrics.json").write_text(json.dumps(full_metrics, indent=2))
    remaining = np.setdiff1d(np.arange(len(dataset.frames)), np.union1d(train_indices, validation_indices))
    remaining_mask = np.isin(eval_indices, remaining)
    summary = {
        "status": "complete",
        "pooled_rmse": full_metrics["pooled_rmse"],
        "remaining_rotation_rmse": float(np.sqrt(np.mean((fitted_prediction[remaining_mask]-observed[remaining_mask])**2))) if remaining_mask.any() else None,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
        "objective_fingerprint": identity,
        "validation_scope": "conditional on calibration using public full-turn data",
        "asteroid": config.asteroid,
        "representation": config.representation,
        "parameter_count": shape.n_parameters,
        "face_count": len(faces),
        "initial_loss": float(trajectory_losses[0]),
        "final_loss": float(final_objective),
        "best_validation_rmse": float(best["validation_rmse"]),
        "best_iteration": int(best["step"]),
        "iterations_run": len(losses),
        "iteration_cap": config.steps,
        "stop_reason": plateau.stop_reason or "iteration_cap",
        "mean_initial_rmse": float(np.mean([row["initial_rmse"] for row in metrics])),
        "mean_fitted_rmse": float(np.mean([row["fitted_rmse"] for row in metrics])),
        "runtime_seconds": time.perf_counter() - started,
        "camera_count": len(dataset.columns),
        "training_frames": config.frame_count,
        "validation_frames": config.validation_frame_count,
        "evaluation_frames": len(eval_indices),
        "public_shape_metrics": shape_metrics,
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "calibration": asdict(calibration),
        "dataset": str(dataset.source),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "summary": summary,
        "validation_history": validation_history,
        "plateau_state": asdict(plateau),
        "optimization_history": optimization_history,
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def default_run_directory(root: Path, asteroid: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root) / f"asteroid{asteroid:02d}_{stamp}"
