"""End-to-end GPU inversion from challenge lightcurves."""

from __future__ import annotations

import json
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
from shape_model import EllipsoidInitialization, OctantoidShapeModel

from .calibration import Calibration
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
    steps: int = 400
    learning_rate: float = 2e-3
    frame_count: int = 12
    evaluation_stride: int = 1
    subdivisions: int = 3
    lmax: int = 6
    image_size: int = 96
    prior_weight: float = 5e-2
    seed: int = 0
    device: str = "cuda"
    checkpoint_interval: int = 25
    early_stopping_patience: int = 50

    @classmethod
    def smoke(cls, asteroid: int, device: str = "cpu") -> "FitConfig":
        return cls(
            asteroid=asteroid,
            steps=2,
            frame_count=2,
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


def _normalized(values: np.ndarray) -> np.ndarray:
    return values / np.clip(values.mean(axis=0, keepdims=True), 1e-9, None)


def _rotation_degrees(dataset: AsteroidDataset, indices: np.ndarray) -> np.ndarray:
    if dataset.origin == "simulated":
        return np.asarray(dataset.frames[indices], dtype=np.float64)
    return 360.0 * np.asarray(indices, dtype=np.float64) / len(dataset.frames)


def _initial_shape(shape: OctantoidShapeModel, dataset: AsteroidDataset) -> torch.Tensor:
    equatorial = [i for i, name in enumerate(dataset.columns) if "hor" in name]
    curves = dataset.curves[:, equatorial]
    hi = np.quantile(curves, 0.95, axis=0)
    lo = np.clip(np.quantile(curves, 0.05, axis=0), 1e-3, None)
    elongation = float(np.clip(np.median(hi / lo), 1.0, 2.5))
    return shape.initial_parameters(
        EllipsoidInitialization(a=elongation, b=1.0, c=0.85, theta=0.0)
    )


def _renderer_config(config: FitConfig) -> RendererConfig:
    face_count = 20 * (4 ** config.subdivisions)
    return RendererConfig(
        image_size=config.image_size,
        shadow_size=config.image_size,
        faces_per_pixel=1 if config.image_size <= 32 else 4,
        batch_size=1 if config.image_size <= 32 else 16,
        face_limit=face_count,
        intensity_slope=8.0,
        binary_slope=8.0,
    )


def _build_model(config: FitConfig, device: torch.device):
    sphere = trimesh.creation.icosphere(subdivisions=config.subdivisions)
    directions = torch.as_tensor(
        np.asarray(sphere.vertices), dtype=torch.float32, device=device
    )
    faces = torch.as_tensor(sphere.faces, dtype=torch.long, device=device)
    shape = OctantoidShapeModel(
        directions,
        faces,
        lmax=config.lmax,
        r0=0.5,
        lam_shape=0.3,
        degree_power=2.0,
        check_self_intersections=False,
    )
    renderer = PyTorch3DLightcurveModel(_renderer_config(config)).to(device)
    return shape, OctantoidLightcurveModel(shape, renderer).to(device)


@torch.no_grad()
def _evaluate(
    model: OctantoidLightcurveModel,
    parameters: torch.Tensor,
    dataset: AsteroidDataset,
    calibration: Calibration,
    stride: int,
    block_frames: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(0, len(dataset.frames), stride, dtype=np.int64)
    raw_blocks = []
    for start in range(0, len(indices), block_frames):
        block = indices[start : start + block_frames]
        angles = torch.as_tensor(
            np.radians(_rotation_degrees(dataset, block)), dtype=torch.float32,
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
    return indices, _normalized(np.concatenate(raw_blocks, axis=0))


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
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    best: dict,
) -> None:
    torch.save(
        {
            "step": step,
            "parameters": parameters.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "parameter_history": parameter_history,
            "vertex_history": vertex_history,
            "losses": losses,
            "scheduler": scheduler.state_dict(),
            "best": best,
        },
        path,
    )


def run_inversion(
    config: FitConfig,
    calibration: Calibration,
    output_dir: Path,
    *,
    resume: Path | None = None,
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
    shape, model = _build_model(config, device)
    train_indices = _sample_indices(len(dataset.frames), config.frame_count)
    train_target = _normalized(dataset.curves[train_indices])
    target = torch.as_tensor(train_target, dtype=torch.float32, device=device)
    angles = torch.as_tensor(
        np.radians(_rotation_degrees(dataset, train_indices)),
        dtype=torch.float32, device=device,
    )
    observations = ObservationBatch.challenge(angles, dataset.columns)

    p0 = _initial_shape(shape, dataset).to(device)
    parameters = p0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameters], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, threshold=1e-4,
        min_lr=1e-5,
    )
    start_step = 0
    parameter_history = [parameters.detach().cpu().numpy().copy()]
    vertex_history = [shape.mesh(parameters).vertices.detach().cpu().numpy().copy()]
    losses: list[float] = []
    best = {
        "loss": float("inf"), "step": 0,
        "parameters": parameters.detach().cpu().clone(), "history_index": 0,
    }
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        parameters.data.copy_(state["parameters"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        parameter_history = list(state["parameter_history"])
        vertex_history = list(state["vertex_history"])
        losses = list(state["losses"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        best = state.get("best", best)

    started = time.perf_counter()
    for step in range(start_step, config.steps):
        optimizer.zero_grad(set_to_none=True)
        rendered = model(
            parameters,
            observations,
            extraction="soft",
            camera_distance_radii=calibration.camera_distance_radii,
            lambertian_fraction=calibration.lambertian_fraction,
            compute_binary=config.curve_type == "binary",
        )
        prediction = rendered.binary if config.curve_type == "binary" else rendered.intensity
        data_loss = (prediction - target).square().mean()
        prior = shape.prior_residuals(parameters)
        loss = data_loss + config.prior_weight * prior.square().mean()
        loss_value = float(loss.detach())
        if loss_value < float(best["loss"]):
            best = {
                "loss": loss_value,
                "step": step,
                "parameters": parameters.detach().cpu().clone(),
                "history_index": len(parameter_history) - 1,
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameters], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            parameters.copy_(shape.project_parameters(parameters))
            mesh = shape.mesh(parameters)
            if not torch.isfinite(mesh.vertices).all():
                raise RuntimeError(f"non-finite mesh at optimization step {step + 1}")
        losses.append(loss_value)
        parameter_history.append(parameters.detach().cpu().numpy().copy())
        vertex_history.append(mesh.vertices.detach().cpu().numpy().copy())
        if (step + 1) % config.checkpoint_interval == 0 or step + 1 == config.steps:
            _save_checkpoint(
                output_dir / "checkpoint.pt", step + 1, parameters, optimizer,
                parameter_history, vertex_history, losses, scheduler, best,
            )
        scheduler.step(loss_value)
        print(
            f"[asteroid {config.asteroid:02d}] step {step + 1:04d}/{config.steps} "
            f"loss={loss_value:.6e} data={float(data_loss):.6e} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if step - int(best["step"]) >= config.early_stopping_patience:
            print(
                f"[asteroid {config.asteroid:02d}] early stop: no improvement "
                f"for {config.early_stopping_patience} steps"
            )
            break

    with torch.no_grad():
        rendered = model(
            parameters,
            observations,
            extraction="soft",
            camera_distance_radii=calibration.camera_distance_radii,
            lambertian_fraction=calibration.lambertian_fraction,
            compute_binary=config.curve_type == "binary",
        )
        final_prediction = (
            rendered.binary if config.curve_type == "binary" else rendered.intensity
        )
        final_prior = shape.prior_residuals(parameters)
        final_objective = (
            (final_prediction - target).square().mean()
            + config.prior_weight * final_prior.square().mean()
        )
    if float(final_objective) < float(best["loss"]):
        best = {
            "loss": float(final_objective),
            "step": len(losses),
            "parameters": parameters.detach().cpu().clone(),
            "history_index": len(parameter_history) - 1,
        }
    trajectory_losses = np.asarray([*losses, float(final_objective)], dtype=np.float64)
    trajectory = FitTrajectory(
        np.stack(parameter_history), np.stack(vertex_history),
        shape.faces.detach().cpu().numpy(), trajectory_losses,
        best_index=int(best["history_index"]),
    )
    trajectory.save(output_dir / "trajectory.npz")
    _save_checkpoint(
        output_dir / "checkpoint.pt", len(losses), parameters, optimizer,
        parameter_history, vertex_history, losses, scheduler, best,
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
    observed = _normalized(dataset.curves[eval_indices])
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
    summary = {
        "asteroid": config.asteroid,
        "initial_loss": float(trajectory_losses[0]),
        "final_loss": float(best["loss"]),
        "best_iteration": int(best["step"]),
        "iterations_run": len(losses),
        "mean_initial_rmse": float(np.mean([row["initial_rmse"] for row in metrics])),
        "mean_fitted_rmse": float(np.mean([row["fitted_rmse"] for row in metrics])),
        "runtime_seconds": time.perf_counter() - started,
        "camera_count": len(dataset.columns),
        "training_frames": config.frame_count,
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
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def default_run_directory(root: Path, asteroid: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root) / f"asteroid{asteroid:02d}_{stamp}"
