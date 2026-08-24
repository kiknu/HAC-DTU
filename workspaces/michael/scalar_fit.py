"""Portable scalar-loss octantoid inversion and experiment reporting.

This module owns the representation-independent parts of the Adam workflow:
device selection, versioned states, loss construction, optimization, metrics,
and artifacts.  PyTorch3D remains the production renderer, but importing this
module does not require its native extension to be installed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Protocol, Sequence

# Set this before importing the legacy data module, which imports matplotlib.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-asteroid-invert")

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from challenge_geometry import UNIQUE_COLUMNS
from forward_model import load_data
from pytorch3d_forward import (
    LightcurvePrediction,
    ObservationBatch,
    OctantoidLightcurveModel,
    PyTorch3DLightcurveModel,
    RendererConfig,
)
from shape_model import (
    EllipsoidInitialization,
    OctantoidShapeModel,
    TriangleMesh,
    validate_triangle_mesh,
)


STATE_VERSION = 1
CALIBRATION_VERSION = 1
DEFAULT_COLUMNS = tuple(UNIQUE_COLUMNS)


class LightcurveModel(Protocol):
    """Small protocol used by the optimizer and its backend-free tests."""

    shape: OctantoidShapeModel

    def __call__(
        self,
        parameters: torch.Tensor,
        observations: ObservationBatch,
        **kwargs: Any,
    ) -> LightcurvePrediction: ...


@dataclass(frozen=True)
class DeviceCapabilities:
    platform: str
    machine: str
    torch_version: str
    cuda_available: bool
    cuda_runtime: str | None
    mps_built: bool
    mps_available: bool
    pytorch3d_available: bool
    pytorch3d_error: str | None
    mitsuba_available: bool
    rtree_available: bool


def device_capabilities() -> DeviceCapabilities:
    """Return runtime facts without attempting a render."""

    from pytorch3d_forward import _PYTORCH3D_IMPORT_ERROR

    mps_backend = getattr(torch.backends, "mps", None)
    return DeviceCapabilities(
        platform=platform.system(),
        machine=platform.machine(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_runtime=torch.version.cuda,
        mps_built=bool(mps_backend and mps_backend.is_built()),
        mps_available=bool(mps_backend and mps_backend.is_available()),
        pytorch3d_available=_PYTORCH3D_IMPORT_ERROR is None,
        pytorch3d_error=(
            None
            if _PYTORCH3D_IMPORT_ERROR is None
            else f"{type(_PYTORCH3D_IMPORT_ERROR).__name__}: "
            f"{_PYTORCH3D_IMPORT_ERROR}"
        ),
        mitsuba_available=importlib.util.find_spec("mitsuba") is not None,
        rtree_available=importlib.util.find_spec("rtree") is not None,
    )


def resolve_device(
    requested: str = "auto", *, allow_experimental_mps: bool = False
) -> torch.device:
    """Resolve a device conservatively.

    Apple Silicon defaults to CPU because PyTorch3D's native rasterizer and
    ``grid_sample`` backward are not generally available on MPS.  MPS can be
    requested only with an explicit experimental opt-in.
    """

    capabilities = device_capabilities()
    requested = requested.lower()
    if requested == "auto":
        if capabilities.cuda_available:
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda":
        if not capabilities.cuda_available:
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not allow_experimental_mps:
            raise RuntimeError(
                "MPS is experimental for this renderer; pass "
                "--experimental-mps after running `asteroid-invert doctor`"
            )
        if not capabilities.mps_available:
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested != "cpu":
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device("cpu")


def renderer_smoke_test(
    device: torch.device,
    *,
    backward: bool = True,
    projection: Literal["orthographic", "perspective"] = "perspective",
) -> dict[str, Any]:
    """Exercise the exact native operations required by inversion."""

    capabilities = device_capabilities()
    if not capabilities.pytorch3d_available:
        return {
            "ok": False,
            "device": str(device),
            "error": capabilities.pytorch3d_error,
        }
    try:
        sphere = trimesh.creation.icosphere(subdivisions=1)
        directions = torch.as_tensor(
            np.asarray(sphere.vertices), dtype=torch.float32, device=device
        )
        faces = torch.as_tensor(
            np.asarray(sphere.faces), dtype=torch.long, device=device
        )
        shape = OctantoidShapeModel(
            directions, faces, lmax=2, check_self_intersections=False
        )
        config = replace(
            RendererConfig.gradient_test(),
            max_frames=1,
            max_cameras=1,
            projection=projection,
        )
        model = OctantoidLightcurveModel(
            shape, PyTorch3DLightcurveModel(config)
        ).to(device)
        parameters = shape.zero_parameters().requires_grad_(backward)
        observations = ObservationBatch.challenge(
            torch.zeros(1, device=device), ("0_top",)
        )
        prediction = model(parameters, observations)
        if backward:
            prediction.raw_intensity.sum().backward()
            if parameters.grad is None or not bool(torch.isfinite(parameters.grad).all()):
                raise RuntimeError("renderer backward returned an invalid gradient")
        return {
            "ok": True,
            "device": str(device),
            "raw_intensity": float(prediction.raw_intensity.detach().cpu().item()),
            "backward": backward,
            "projection": projection,
        }
    except Exception as exc:  # Diagnostic command must return the real cause.
        return {
            "ok": False,
            "device": str(device),
            "error": f"{type(exc).__name__}: {exc}",
        }


@dataclass(frozen=True)
class RendererHyperparameters:
    projection: Literal["orthographic", "perspective"] = "perspective"
    camera_distance_radii: float = 5.0
    lambertian_fraction: float = 0.5
    shadow_softness_radii: float = 2e-3
    calibration_version: int = CALIBRATION_VERSION

    def validate(self) -> None:
        if self.projection not in ("orthographic", "perspective"):
            raise ValueError("invalid projection")
        if self.camera_distance_radii <= 1.0:
            raise ValueError("camera distance must exceed one radius")
        if not 0.0 <= self.lambertian_fraction <= 1.0:
            raise ValueError("lambertian_fraction must be in [0, 1]")
        if self.shadow_softness_radii <= 0.0:
            raise ValueError("shadow softness must be positive")

    @classmethod
    def load(cls, path: Path | None) -> "RendererHyperparameters":
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        # Calibration files contain both the selected parameters and audit
        # metadata; plain parameter JSON is accepted as well.
        values = payload.get("selected", payload)
        result = cls(
            projection=str(values["projection"]),
            camera_distance_radii=float(values["camera_distance_radii"]),
            lambertian_fraction=float(values["lambertian_fraction"]),
            shadow_softness_radii=float(
                values.get("shadow_softness_radii", 2e-3)
            ),
            calibration_version=int(
                payload.get("calibration_version", CALIBRATION_VERSION)
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class FitConfig:
    iterations: int = 200
    learning_rate: float = 2e-3
    phase_learning_rate: float = 1e-2
    loss: Literal["huber", "mse"] = "huber"
    huber_delta: float = 0.05
    prior_weight: float = 5e-2
    gradient_clip: float = 5.0
    camera_group_size: int = 3
    evaluation_interval: int = 10
    self_intersection_interval: int = 25
    initialize_phase: bool = True
    phase_grid_step_degrees: float = 2.0
    seed: int = 0

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.learning_rate <= 0.0 or self.phase_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.loss not in ("huber", "mse"):
            raise ValueError("loss must be huber or mse")
        if self.huber_delta <= 0.0 or self.prior_weight < 0.0:
            raise ValueError("loss and prior settings are invalid")
        if self.camera_group_size <= 0 or self.evaluation_interval <= 0:
            raise ValueError("group size and evaluation interval must be positive")
        if self.phase_grid_step_degrees <= 0.0:
            raise ValueError("phase grid step must be positive")


@dataclass
class OctantoidState:
    parameters: torch.Tensor
    phase_radians: float
    subdivisions: int
    lmax_a: int
    lmax_b: int
    lmax_c: int
    r0: float = 0.5
    template_hash: str = ""
    renderer_hyperparameters: dict[str, Any] | None = None
    state_version: int = STATE_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "state_version": int(self.state_version),
            "parameters": self.parameters.detach().to("cpu", torch.float32),
            "phase_radians": float(self.phase_radians),
            "subdivisions": int(self.subdivisions),
            "lmax_a": int(self.lmax_a),
            "lmax_b": int(self.lmax_b),
            "lmax_c": int(self.lmax_c),
            "r0": float(self.r0),
            "template_hash": str(self.template_hash),
            "renderer_hyperparameters": self.renderer_hyperparameters,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.payload(), path)
        return path

    @classmethod
    def load(cls, path: Path) -> "OctantoidState":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        version = int(payload.get("state_version", 0))
        if version != STATE_VERSION:
            raise ValueError(
                f"unsupported octantoid state version {version}; expected "
                f"{STATE_VERSION}"
            )
        return cls(
            parameters=torch.as_tensor(payload["parameters"]).detach().clone(),
            phase_radians=float(payload["phase_radians"]),
            subdivisions=int(payload["subdivisions"]),
            lmax_a=int(payload["lmax_a"]),
            lmax_b=int(payload["lmax_b"]),
            lmax_c=int(payload["lmax_c"]),
            r0=float(payload["r0"]),
            template_hash=str(payload.get("template_hash", "")),
            renderer_hyperparameters=payload.get("renderer_hyperparameters"),
            state_version=version,
        )


def _template_hash(faces: torch.Tensor) -> str:
    values = faces.detach().to("cpu", torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def create_octantoid_shape(
    subdivisions: int,
    device: torch.device,
    *,
    lmax_a: int = 6,
    lmax_b: int | None = None,
    lmax_c: int | None = None,
    r0: float = 0.5,
    check_self_intersections: bool = True,
) -> OctantoidShapeModel:
    if subdivisions < 1:
        raise ValueError("subdivisions must be at least one")
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    directions = torch.as_tensor(
        np.asarray(sphere.vertices), dtype=torch.float32, device=device
    )
    faces = torch.as_tensor(
        np.asarray(sphere.faces), dtype=torch.long, device=device
    )
    return OctantoidShapeModel(
        directions,
        faces,
        lmax=None,
        lmax_a=lmax_a,
        lmax_b=lmax_a if lmax_b is None else lmax_b,
        lmax_c=lmax_a if lmax_c is None else lmax_c,
        r0=r0,
        lam_shape=0.3,
        degree_power=2.0,
        check_self_intersections=check_self_intersections,
    )


def new_octantoid_state(
    shape: OctantoidShapeModel,
    subdivisions: int,
    initialization: EllipsoidInitialization | None = None,
) -> OctantoidState:
    return OctantoidState(
        parameters=shape.initial_parameters(initialization).detach().cpu(),
        phase_radians=0.0,
        subdivisions=subdivisions,
        lmax_a=shape.lmax_a,
        lmax_b=shape.lmax_b,
        lmax_c=shape.lmax_c,
        r0=shape.r0,
        template_hash=_template_hash(shape.faces),
    )


def shape_from_state(
    state: OctantoidState,
    device: torch.device,
    *,
    subdivisions: int | None = None,
    check_self_intersections: bool = True,
) -> OctantoidShapeModel:
    chosen_subdivisions = state.subdivisions if subdivisions is None else subdivisions
    shape = create_octantoid_shape(
        chosen_subdivisions,
        device,
        lmax_a=state.lmax_a,
        lmax_b=state.lmax_b,
        lmax_c=state.lmax_c,
        r0=state.r0,
        check_self_intersections=check_self_intersections,
    )
    if state.parameters.numel() != shape.n_parameters:
        raise ValueError(
            "state coefficient count does not match its harmonic metadata"
        )
    if subdivisions is None and state.template_hash:
        actual = _template_hash(shape.faces)
        if actual != state.template_hash:
            raise ValueError("state template does not match this implementation")
    return shape


@dataclass(frozen=True)
class LightcurveData:
    asteroid: int
    origin: Literal["real", "simulated"]
    columns: tuple[str, ...]
    degrees: np.ndarray
    target: torch.Tensor
    observations: ObservationBatch
    frame_stride: int


def load_lightcurve_data(
    asteroid: int,
    origin: Literal["real", "simulated"],
    device: torch.device,
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    frame_stride: int = 1,
) -> LightcurveData:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    dataset = load_data(asteroid, origin, "intensity")
    missing = set(columns) - set(dataset.data.columns)
    if missing:
        raise ValueError(f"unknown lightcurve columns: {sorted(missing)}")
    full = np.asarray(dataset.data.loc[:, list(columns)], dtype=np.float32)
    means = full.mean(axis=0, keepdims=True)
    if np.any(~np.isfinite(means)) or np.any(np.abs(means) < 1e-12):
        raise ValueError("lightcurve contains an invalid mean")
    full = full / means
    indices = np.arange(0, len(full), frame_stride)
    degrees = 360.0 * indices.astype(np.float64) / len(full)
    angles = torch.as_tensor(
        np.radians(degrees), dtype=torch.float32, device=device
    )
    target = torch.as_tensor(full[indices], dtype=torch.float32, device=device)
    return LightcurveData(
        asteroid=asteroid,
        origin=origin,
        columns=tuple(columns),
        degrees=degrees,
        target=target,
        observations=ObservationBatch.challenge(angles, columns),
        frame_stride=frame_stride,
    )


def photometric_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: Literal["huber", "mse"],
    huber_delta: float,
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target curves must have the same shape")
    if kind == "mse":
        return F.mse_loss(predicted, target)
    if kind == "huber":
        return F.huber_loss(predicted, target, delta=huber_delta)
    raise ValueError("loss must be huber or mse")


def _camera_groups(count: int, group_size: int) -> list[list[int]]:
    return [list(range(i, min(i + group_size, count))) for i in range(0, count, group_size)]


def _subset_observations(
    observations: ObservationBatch, indices: Sequence[int]
) -> ObservationBatch:
    tensor_index = torch.as_tensor(
        indices, dtype=torch.long, device=observations.view_directions.device
    )
    columns = (
        tuple(observations.columns[i] for i in indices)
        if observations.columns
        else ()
    )
    return ObservationBatch(
        angles=observations.angles,
        view_directions=observations.view_directions[tensor_index],
        sun_direction=observations.sun_direction,
        columns=columns,
    )


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Map radians to [-pi, pi] without changing the local derivative."""

    return torch.atan2(torch.sin(phase), torch.cos(phase))


@dataclass
class FitResult:
    state: OctantoidState
    history: list[dict[str, float | int | str]]
    initial_prediction: np.ndarray
    fitted_prediction: np.ndarray
    observed: np.ndarray
    degrees: np.ndarray
    columns: tuple[str, ...]
    elapsed_seconds: float
    invalid_steps: int


def fit_octantoid(
    model: LightcurveModel,
    data: LightcurveData,
    initial_state: OctantoidState,
    config: FitConfig,
    hyperparameters: RendererHyperparameters,
) -> FitResult:
    """Fit an octantoid state with Adam and a scalar photometric objective."""

    config.validate()
    hyperparameters.validate()
    torch.manual_seed(config.seed)
    device = data.target.device
    parameters = torch.nn.Parameter(
        initial_state.parameters.to(device=device, dtype=torch.float32).clone()
    )
    phase = torch.nn.Parameter(
        torch.tensor(initial_state.phase_radians, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [parameters], "lr": config.learning_rate},
            {"params": [phase], "lr": config.phase_learning_rate},
        ]
    )
    render_kwargs = {
        "camera_distance_radii": hyperparameters.camera_distance_radii,
        "lambertian_fraction": hyperparameters.lambertian_fraction,
    }

    def predict_all(extraction: str = "soft") -> torch.Tensor:
        return model(
            parameters,
            data.observations,
            phase_offset=phase,
            extraction=extraction,
            **render_kwargs,
        ).intensity

    with torch.no_grad():
        initial_prediction = predict_all("exact").detach().cpu().numpy()
        if config.initialize_phase and abs(initial_state.phase_radians) < 1e-12:
            from forward_validation import estimate_phase

            phase_degrees, _ = estimate_phase(
                initial_prediction,
                data.target.detach().cpu().numpy(),
                data.degrees,
                step_degrees=config.phase_grid_step_degrees,
            )
            phase.copy_(
                torch.tensor(
                    math.radians(phase_degrees),
                    device=device,
                    dtype=torch.float32,
                )
            )
        initial_soft = predict_all("soft")
        initial_photo = float(
            photometric_loss(
                initial_soft,
                data.target,
                kind=config.loss,
                huber_delta=config.huber_delta,
            )
        )
        initial_prior = float(model.shape.prior_residuals(parameters).square().mean())
        best_objective = initial_photo + config.prior_weight * initial_prior
    best_parameters = parameters.detach().clone()
    best_phase = phase.detach().clone()
    groups = _camera_groups(data.target.shape[1], config.camera_group_size)
    history: list[dict[str, float | int | str]] = []
    invalid_steps = 0
    start_time = time.perf_counter()

    for step in range(config.iterations):
        group = groups[step % len(groups)]
        subset = _subset_observations(data.observations, group)
        previous_parameters = parameters.detach().clone()
        previous_phase = phase.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            parameters,
            subset,
            phase_offset=phase,
            extraction="soft",
            **render_kwargs,
        ).intensity
        photo = photometric_loss(
            prediction,
            data.target[:, group],
            kind=config.loss,
            huber_delta=config.huber_delta,
        )
        prior_residuals = model.shape.prior_residuals(parameters)
        prior = prior_residuals.square().mean()
        total = photo + config.prior_weight * prior
        total.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [parameters, phase], config.gradient_clip
            )
        )
        optimizer.step()
        with torch.no_grad():
            phase.copy_(wrap_phase(phase))
            fast_validation = validate_triangle_mesh(model.shape.mesh(parameters))
            full_validation = fast_validation
            if (
                fast_validation.valid
                and config.self_intersection_interval > 0
                and (step + 1) % config.self_intersection_interval == 0
            ):
                full_validation = model.shape.validate_mesh(
                    model.shape.mesh(parameters)
                )
            if not full_validation.valid:
                parameters.copy_(previous_parameters)
                phase.copy_(previous_phase)
                optimizer.state.clear()
                for group_config in optimizer.param_groups:
                    group_config["lr"] *= 0.5
                invalid_steps += 1

        full_score = math.nan
        full_objective = math.nan
        if (step + 1) % config.evaluation_interval == 0 or step + 1 == config.iterations:
            with torch.no_grad():
                full_prediction = predict_all("soft")
                full_score = float(
                    photometric_loss(
                        full_prediction,
                        data.target,
                        kind=config.loss,
                        huber_delta=config.huber_delta,
                    )
                )
                full_prior = float(
                    model.shape.prior_residuals(parameters).square().mean()
                )
                full_objective = full_score + config.prior_weight * full_prior
                if (
                    math.isfinite(full_objective)
                    and full_objective < best_objective
                ):
                    best_objective = full_objective
                    best_parameters = parameters.detach().clone()
                    best_phase = phase.detach().clone()

        history.append(
            {
                "iteration": step + 1,
                "camera_group": ";".join(data.columns[i] for i in group),
                "total_loss": float(total.detach()),
                "photometric_loss": float(photo.detach()),
                "prior_loss": float(prior.detach()),
                "full_photometric_loss": full_score,
                "full_objective": full_objective,
                "gradient_norm": gradient_norm,
                "phase_degrees": math.degrees(float(phase.detach().cpu())),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "valid": int(full_validation.valid),
            }
        )

    with torch.no_grad():
        parameters.copy_(best_parameters)
        phase.copy_(best_phase)
        fitted_prediction = predict_all("exact").detach().cpu().numpy()
    elapsed = time.perf_counter() - start_time
    final_state = OctantoidState(
        parameters=parameters.detach().cpu(),
        phase_radians=float(phase.detach().cpu()),
        subdivisions=initial_state.subdivisions,
        lmax_a=initial_state.lmax_a,
        lmax_b=initial_state.lmax_b,
        lmax_c=initial_state.lmax_c,
        r0=initial_state.r0,
        template_hash=initial_state.template_hash,
        renderer_hyperparameters=asdict(hyperparameters),
    )
    return FitResult(
        state=final_state,
        history=history,
        initial_prediction=initial_prediction,
        fitted_prediction=fitted_prediction,
        observed=data.target.detach().cpu().numpy(),
        degrees=data.degrees,
        columns=data.columns,
        elapsed_seconds=elapsed,
        invalid_steps=invalid_steps,
    )


def curve_metrics(
    predicted: np.ndarray, observed: np.ndarray, columns: Sequence[str]
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, column in enumerate(columns):
        pred = np.asarray(predicted[:, index], dtype=np.float64)
        obs = np.asarray(observed[:, index], dtype=np.float64)
        residual = pred - obs
        pred_std, obs_std = pred.std(), obs.std()
        correlation = (
            float(np.corrcoef(pred, obs)[0, 1])
            if pred_std > 1e-12 and obs_std > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "camera": column,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "correlation": correlation,
                "amplitude_ratio": float(pred_std / max(obs_std, 1e-12)),
            }
        )
    return rows


def _mean_metric(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def _write_plots(
    output_dir: Path,
    result: FitResult,
    *,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-asteroid-invert")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_columns = 3
    n_rows = math.ceil(len(result.columns) / n_columns)
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(15, 2.8 * n_rows), sharex=True
    )
    for index, axis in enumerate(np.asarray(axes).reshape(-1)):
        if index >= len(result.columns):
            axis.axis("off")
            continue
        axis.plot(result.degrees, result.observed[:, index], "k-", lw=1.2, label="observed")
        axis.plot(
            result.degrees,
            result.initial_prediction[:, index],
            color="tab:blue",
            ls="--",
            lw=1.0,
            label="initial",
        )
        axis.plot(
            result.degrees,
            result.fitted_prediction[:, index],
            color="tab:red",
            lw=1.1,
            label="fitted",
        )
        axis.set_title(result.columns[index])
        axis.grid(alpha=0.25)
    axes.reshape(-1)[0].legend(ncol=3, fontsize=8)
    figure.supxlabel("rotation [degrees]")
    figure.supylabel("mean-normalized intensity")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_dir / "lightcurves_before_after.png", dpi=160)
    plt.close(figure)

    residuals = np.stack(
        [
            result.initial_prediction - result.observed,
            result.fitted_prediction - result.observed,
        ]
    )
    limit = max(float(np.quantile(np.abs(residuals), 0.99)), 1e-4)
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for axis, values, label in zip(axes, residuals, ("initial", "fitted")):
        image = axis.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            extent=(result.degrees[0], result.degrees[-1], -0.5, len(result.columns) - 0.5),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_yticks(range(len(result.columns)), result.columns, fontsize=7)
        axis.set_title(f"{label} residual")
        figure.colorbar(image, ax=axis, label="prediction - observation")
    axes[-1].set_xlabel("rotation [degrees]")
    figure.tight_layout()
    figure.savefig(output_dir / "residual_heatmaps.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    iterations = [int(row["iteration"]) for row in result.history]
    axis.plot(
        iterations,
        [float(row["photometric_loss"]) for row in result.history],
        alpha=0.5,
        label="camera-group loss",
    )
    full_x = [
        int(row["iteration"])
        for row in result.history
        if math.isfinite(float(row["full_photometric_loss"]))
    ]
    full_y = [
        float(row["full_photometric_loss"])
        for row in result.history
        if math.isfinite(float(row["full_photometric_loss"]))
    ]
    axis.plot(full_x, full_y, "o-", label="all-camera loss")
    axis.set_yscale("log")
    axis.set_xlabel("Adam iteration")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "optimization_history.png", dpi=160)
    plt.close(figure)


def write_fit_artifacts(
    output_dir: Path,
    result: FitResult,
    shape: OctantoidShapeModel,
    fit_config: FitConfig,
    hyperparameters: RendererHyperparameters,
    *,
    asteroid: int,
    origin: str,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.state.save(output_dir / "octantoid_state.pt")
    with (output_dir / "optimization_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)
    np.savez_compressed(
        output_dir / "lightcurves.npz",
        degrees=result.degrees,
        columns=np.asarray(result.columns),
        observed=result.observed,
        initial=result.initial_prediction,
        fitted=result.fitted_prediction,
    )

    parameters = result.state.parameters.to(shape.directions.device)
    mesh = shape.mesh(parameters)
    exported = trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )
    exported.export(output_dir / "fitted_mesh.obj")
    exported.export(output_dir / "fitted_mesh.glb")

    initial_metrics = curve_metrics(
        result.initial_prediction, result.observed, result.columns
    )
    fitted_metrics = curve_metrics(
        result.fitted_prediction, result.observed, result.columns
    )
    with (output_dir / "per_camera_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "camera",
            "initial_rmse",
            "fitted_rmse",
            "initial_correlation",
            "fitted_correlation",
            "fitted_mae",
            "fitted_amplitude_ratio",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for before, after in zip(initial_metrics, fitted_metrics):
            writer.writerow(
                {
                    "camera": after["camera"],
                    "initial_rmse": before["rmse"],
                    "fitted_rmse": after["rmse"],
                    "initial_correlation": before["correlation"],
                    "fitted_correlation": after["correlation"],
                    "fitted_mae": after["mae"],
                    "fitted_amplitude_ratio": after["amplitude_ratio"],
                }
            )
    summary = {
        "asteroid": asteroid,
        "example_kind": (
            "public_full_mesh" if asteroid in (1, 2, 3) else "hidden_shape"
        ),
        "ground_truth_shape_available": asteroid in (1, 2, 3),
        "origin": origin,
        "sampled_frames": int(len(result.degrees)),
        "camera_curves": int(len(result.columns)),
        "device": str(device),
        "elapsed_seconds": result.elapsed_seconds,
        "invalid_steps": result.invalid_steps,
        "phase_degrees": math.degrees(result.state.phase_radians),
        "initial_mean_rmse": _mean_metric(initial_metrics, "rmse"),
        "fitted_mean_rmse": _mean_metric(fitted_metrics, "rmse"),
        "initial_mean_correlation": _mean_metric(initial_metrics, "correlation"),
        "fitted_mean_correlation": _mean_metric(fitted_metrics, "correlation"),
        "fit_config": asdict(fit_config),
        "renderer_hyperparameters": asdict(hyperparameters),
        "state": {
            key: value
            for key, value in result.state.payload().items()
            if key != "parameters"
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _write_plots(
        output_dir,
        result,
        title=(
            f"Asteroid {asteroid}: observed, prior, and Adam-corrected curves"
        ),
    )
    return summary


def write_resolution_evaluation(
    output_dir: Path,
    state: OctantoidState,
    data: LightcurveData,
    hyperparameters: RendererHyperparameters,
    device: torch.device,
    *,
    subdivisions: int = 4,
    image_size: int = 128,
) -> dict[str, Any]:
    """Evaluate fitted coefficients on a denser octantoid sampling grid."""

    shape = shape_from_state(
        state,
        device,
        subdivisions=subdivisions,
        check_self_intersections=False,
    )
    face_count = int(shape.faces.shape[0])
    config = replace(
        renderer_config_for_device(
            device, hyperparameters, image_size=image_size
        ),
        batch_size=1,
        face_limit=max(5_000, face_count),
        allow_large_mesh=face_count > 5_000,
        require_no_grad=True,
    )
    model = OctantoidLightcurveModel(
        shape, PyTorch3DLightcurveModel(config)
    ).to(device)
    parameters = state.parameters.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(
            parameters,
            data.observations,
            phase_offset=state.phase_radians,
            extraction="exact",
            camera_distance_radii=hyperparameters.camera_distance_radii,
            lambertian_fraction=hyperparameters.lambertian_fraction,
        ).intensity.detach().cpu().numpy()
    observed = data.target.detach().cpu().numpy()
    rows = curve_metrics(prediction, observed, data.columns)
    prefix = f"evaluation_subdiv{subdivisions}"
    output_dir = Path(output_dir)
    np.savez_compressed(
        output_dir / f"{prefix}_lightcurves.npz",
        degrees=data.degrees,
        columns=np.asarray(data.columns),
        observed=observed,
        fitted=prediction,
    )
    mesh = shape.mesh(parameters)
    exported = trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )
    exported.export(output_dir / f"{prefix}_mesh.obj")
    exported.export(output_dir / f"{prefix}_mesh.glb")
    summary = {
        "subdivisions": subdivisions,
        "vertices": int(mesh.vertices.shape[0]),
        "faces": face_count,
        "image_size": image_size,
        "mean_rmse": _mean_metric(rows, "rmse"),
        "mean_correlation": _mean_metric(rows, "correlation"),
    }
    with (output_dir / f"{prefix}_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)
    return summary


def write_public_shape_report(
    output_dir: Path,
    asteroid: int,
    shape: OctantoidShapeModel,
    initial_state: OctantoidState,
    fitted_state: OctantoidState,
    device: torch.device,
    *,
    reference_faces: int = 1_280,
) -> dict[str, Any]:
    """Compare a public fit with a radial proxy of its known full mesh."""

    if asteroid not in (1, 2, 3):
        raise ValueError("shape-reference reports exist only for asteroids 1-3")
    from forward_validation import load_benchmark_mesh
    from visualize_fit import visualize_fit

    output_dir = Path(output_dir)
    reference = load_benchmark_mesh(
        asteroid,
        reference_faces,
        device,
        output_dir.parent / "public_proxy_cache",
        "star_proxy",
    )
    fitter = SimpleNamespace(device=device, shape_model=shape)
    metrics = visualize_fit(
        fitter,
        initial_state.parameters.to(device),
        fitted_state.parameters.to(device),
        reference=(reference.vertices, reference.faces),
        ref_label=f"asteroid {asteroid} radial reference",
        out_prefix=str(output_dir / "shape"),
        interactive=False,
        turntable=False,
    )
    with (output_dir / "shape_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metrics, stream, indent=2)
    return metrics


def regenerate_report(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    saved = np.load(output_dir / "lightcurves.npz")
    columns = tuple(str(value) for value in saved["columns"])
    result = FitResult(
        state=OctantoidState.load(output_dir / "octantoid_state.pt"),
        history=[],
        initial_prediction=saved["initial"],
        fitted_prediction=saved["fitted"],
        observed=saved["observed"],
        degrees=saved["degrees"],
        columns=columns,
        elapsed_seconds=0.0,
        invalid_steps=0,
    )
    history_path = output_dir / "optimization_history.csv"
    with history_path.open("r", encoding="utf-8") as stream:
        result.history = list(csv.DictReader(stream))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    _write_plots(
        output_dir,
        result,
        title=(
            f"Asteroid {summary['asteroid']}: observed, prior, and "
            "Adam-corrected curves"
        ),
    )


def renderer_config_for_device(
    device: torch.device,
    hyperparameters: RendererHyperparameters,
    *,
    image_size: int | None = None,
) -> RendererConfig:
    config = (
        RendererConfig.cuda_optimization()
        if device.type == "cuda"
        else RendererConfig.cpu_optimization()
    )
    return replace(
        config,
        image_size=config.image_size if image_size is None else image_size,
        shadow_size=config.shadow_size if image_size is None else image_size,
        projection=hyperparameters.projection,
        camera_distance_radii=hyperparameters.camera_distance_radii,
        lambertian_fraction=hyperparameters.lambertian_fraction,
        shadow_softness_radii=hyperparameters.shadow_softness_radii,
    )


def calibrate_public_bodies(
    output_dir: Path,
    device: torch.device,
    *,
    asteroids: Sequence[int] = (1, 2, 3),
    frame_stride: int = 12,
    target_faces: int = 1_280,
    image_size: int = 64,
    lambertian_values: Sequence[float] = (0.25, 0.5, 0.75),
    projections: Sequence[str] = ("orthographic", "perspective"),
    distance_values: Sequence[float] = (3.5, 5.0, 8.0, 16.0),
    softness_values: Sequence[float] = (1e-3, 2e-3, 5e-3),
) -> dict[str, Any]:
    """Calibrate shared renderer settings with leave-one-body-out auditing."""

    if any(body not in (1, 2, 3) for body in asteroids):
        raise ValueError("calibration is restricted to public asteroids 1-3")
    if len(set(asteroids)) < 2:
        raise ValueError("leave-one-body-out calibration requires at least two bodies")
    asteroids = tuple(dict.fromkeys(asteroids))
    from forward_validation import estimate_phase, load_benchmark_mesh, periodic_shift

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "proxy_cache"
    datasets = {
        body: load_lightcurve_data(
            body, "simulated", device, frame_stride=frame_stride
        )
        for body in asteroids
    }
    meshes = {
        body: load_benchmark_mesh(
            body, target_faces, device, cache_dir, "star_proxy"
        )
        for body in asteroids
    }
    candidates: list[RendererHyperparameters] = []
    for projection, mixture, softness in itertools.product(
        projections, lambertian_values, softness_values
    ):
        distances = (5.0,) if projection == "orthographic" else distance_values
        for distance in distances:
            candidates.append(
                RendererHyperparameters(
                    projection=projection,
                    camera_distance_radii=float(distance),
                    lambertian_fraction=float(mixture),
                    shadow_softness_radii=float(softness),
                )
            )

    records: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        config = replace(
            renderer_config_for_device(
                device, candidate, image_size=image_size
            ),
            require_no_grad=True,
        )
        renderer = PyTorch3DLightcurveModel(config).to(device)
        for body in asteroids:
            data = datasets[body]
            with torch.no_grad():
                prediction = renderer(
                    meshes[body], data.observations, extraction="exact"
                ).intensity.detach().cpu().numpy()
            observed = data.target.detach().cpu().numpy()
            phase_degrees, _ = estimate_phase(
                prediction, observed, data.degrees, step_degrees=1.0
            )
            aligned = periodic_shift(
                prediction, data.degrees, phase_degrees
            )
            per_curve = np.sqrt(np.mean((aligned - observed) ** 2, axis=0))
            records.append(
                {
                    "candidate": candidate_index,
                    "asteroid": body,
                    "phase_degrees": phase_degrees,
                    "mean_rmse": float(per_curve.mean()),
                    **asdict(candidate),
                }
            )

    def candidate_score(index: int, bodies: Sequence[int]) -> float:
        values = [
            float(row["mean_rmse"])
            for row in records
            if row["candidate"] == index and row["asteroid"] in bodies
        ]
        return float(np.mean(values))

    folds = []
    for held_out in asteroids:
        training = [body for body in asteroids if body != held_out]
        best_index = min(
            range(len(candidates)), key=lambda index: candidate_score(index, training)
        )
        folds.append(
            {
                "held_out_asteroid": held_out,
                "selected_candidate": best_index,
                "training_mean_rmse": candidate_score(best_index, training),
                "held_out_mean_rmse": candidate_score(best_index, [held_out]),
            }
        )
    selected_index = min(
        range(len(candidates)), key=lambda index: candidate_score(index, asteroids)
    )
    selected = candidates[selected_index]
    selected_body_results = [
        {
            "asteroid": int(row["asteroid"]),
            "phase_degrees": float(row["phase_degrees"]),
            "mean_rmse": float(row["mean_rmse"]),
        }
        for row in records
        if row["candidate"] == selected_index
    ]
    payload = {
        "calibration_version": CALIBRATION_VERSION,
        "selected": asdict(selected),
        "selected_candidate": selected_index,
        "all_body_mean_rmse": candidate_score(selected_index, asteroids),
        "selected_body_results": selected_body_results,
        "leave_one_body_out": folds,
        "asteroids": list(asteroids),
        "frame_stride": frame_stride,
        "target_faces": target_faces,
        "image_size": image_size,
    }
    with (output_dir / "calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    with (output_dir / "calibration_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return payload
