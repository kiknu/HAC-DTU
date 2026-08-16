#!/usr/bin/env python3
"""Benchmark the mesh-only forward model against public Blender lightcurves."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-forward-validation")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from forward_model import (
    ObservationGeometry,
    PhotometricForwardModel,
    PreparedVisibility,
    build_view_dirs,
    get_model_stl_path,
    load_data,
)
from shape_model import TriangleMesh, validate_triangle_mesh
from render_safety import DEFAULT_FACE_LIMIT, enforce_face_limit


UNIQUE_COLUMNS = [
    f"{az}_{kind}"
    for az in (0, 45, 90, 135, 225, 270, 315)
    for kind in ("hor1", "top", "bottom")
]
CALIBRATION_COLUMNS = [
    f"{az}_{kind}"
    for az in (0, 90, 225, 315)
    for kind in ("hor1", "top")
]
ALIGNMENT_COLUMN = "0_hor1"
MESH_CACHE_VERSION = 3


@dataclass(frozen=True)
class BenchmarkConfig:
    asteroids: tuple[int, ...] = (1,)
    target_faces: int = 1280
    face_count_overrides: tuple[tuple[int, int], ...] = ()
    mesh_preprocessing: str = "star_proxy"
    frame_stride: int = 6
    gamma_grid: tuple[float, ...] = (0.01, 0.02, 0.04)
    lambertian_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    tau_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    beta_grid: tuple[float, ...] = (8.0, 16.0, 32.0)
    phase_step_degrees: float = 1.0
    calibration_columns: tuple[str, ...] = tuple(CALIBRATION_COLUMNS)
    convergence_faces: tuple[int, ...] = (80, 320, 1280)
    convergence: bool = True
    projection: str = "orthographic"
    camera_distance_radii: float = 5.0
    visibility_samples: int = 1
    device: str = "cpu"
    allow_large_mesh: bool = False


@dataclass
class AsteroidRuntime:
    model_num: int
    mesh: TriangleMesh
    columns: list[str]
    degrees: np.ndarray
    intensity: np.ndarray
    binary: np.ndarray
    renderer: PhotometricForwardModel
    prepared: dict[float, PreparedVisibility]
    endpoints: dict[float, tuple[np.ndarray, np.ndarray]]


def _parse_csv(values: str, cast=float) -> tuple:
    return tuple(cast(item.strip()) for item in values.split(",") if item.strip())


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def normalize_challenge_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center a mesh and scale it so its z bounds are exactly [-1, 1]."""
    result = mesh.copy()
    bounds = np.asarray(result.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    z_half_extent = 0.5 * (bounds[1, 2] - bounds[0, 2])
    if not np.isfinite(z_half_extent) or z_half_extent <= 0.0:
        raise ValueError("mesh has invalid z extent")
    result.vertices = (np.asarray(result.vertices) - center) / z_half_extent
    return result


def simplify_closed_mesh(
    source_mesh: trimesh.Trimesh, target_faces: int
) -> trimesh.Trimesh:
    """Repeatedly simplify a closed mesh while preserving its orientation."""
    result = source_mesh.copy()
    if len(result.faces) > target_faces:
        # Quality-first QEM is retained only as an explicit alternative to the
        # optimizer-topology star proxy.  Never force the sawed cube through
        # aggression=10: that reaches the count by destroying its geometry.
        for _ in range(12):
            simplified = result.simplify_quadric_decimation(
                face_count=target_faces, aggression=4
            )
            if len(simplified.faces) >= len(result.faces):
                break
            result = simplified
            if len(result.faces) <= target_faces:
                break
    if len(result.faces) > target_faces:
        raise ValueError(
            f"could not simplify mesh below {len(result.faces)} faces"
        )
    result.remove_unreferenced_vertices()
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError("simplified mesh is not closed and consistently oriented")
    if result.volume < 0.0:
        result.invert()
    return result


def star_proxy_mesh(
    source_mesh: trimesh.Trimesh, target_faces: int
) -> trimesh.Trimesh:
    """Sample the outer radial surface on the optimizer's icosphere topology."""
    candidates = [(20 * 4**level, level) for level in range(7)]
    _, subdivisions = min(
        candidates, key=lambda item: (abs(item[0] - target_faces), item[0])
    )
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    directions = np.asarray(sphere.vertices, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    intersector = RayMeshIntersector(source_mesh)
    locations, ray_indices, _ = intersector.intersects_location(
        np.zeros_like(directions), directions, multiple_hits=True
    )
    radii = np.zeros(len(directions), dtype=np.float64)
    distances = np.linalg.norm(locations, axis=1)
    np.maximum.at(radii, ray_indices, distances)
    if np.any(radii <= 0.0):
        raise ValueError(
            f"star proxy has {int(np.count_nonzero(radii <= 0.0))} missed rays"
        )
    result = trimesh.Trimesh(
        vertices=directions * radii[:, None],
        faces=np.asarray(sphere.faces),
        process=False,
        validate=False,
    )
    if result.volume < 0.0:
        result.invert()
    return result


def _cache_path(
    cache_dir: Path, model_num: int, target_faces: int, preprocessing: str
) -> Path:
    return cache_dir / (
        f"asteroid{model_num:02d}_{preprocessing}_{target_faces:d}_faces.npz"
    )


def load_benchmark_mesh(
    model_num: int,
    target_faces: int,
    device: torch.device,
    cache_dir: Path,
    preprocessing: str = "star_proxy",
    allow_large_mesh: bool = False,
) -> TriangleMesh:
    """Load, normalize, and cache a simplified public mesh."""
    if target_faces < 4:
        raise ValueError("target_faces must be at least 4")
    source = get_model_stl_path(model_num)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if preprocessing not in ("star_proxy", "qem"):
        raise ValueError("preprocessing must be 'star_proxy' or 'qem'")
    cache = _cache_path(cache_dir, model_num, target_faces, preprocessing)
    source_size = source.stat().st_size

    vertices: np.ndarray
    faces: np.ndarray
    if cache.is_file():
        saved = np.load(cache)
        cache_valid = (
            "cache_version" in saved
            and int(saved["cache_version"]) == MESH_CACHE_VERSION
            and int(saved["source_size"]) == source_size
            and str(saved["preprocessing"]) == preprocessing
        )
        if cache_valid:
            vertices = saved["vertices"]
            faces = saved["faces"]
        else:
            cache.unlink()
            return load_benchmark_mesh(
                model_num,
                target_faces,
                device,
                cache_dir,
                preprocessing,
                allow_large_mesh,
            )
    else:
        source_mesh = normalize_challenge_mesh(
            trimesh.load(source, force="mesh", process=True)
        )
        if preprocessing == "star_proxy":
            source_mesh = star_proxy_mesh(source_mesh, target_faces)
        else:
            source_mesh = simplify_closed_mesh(source_mesh, target_faces)
        vertices = np.asarray(source_mesh.vertices, dtype=np.float32)
        faces = np.asarray(source_mesh.faces, dtype=np.int64)
        np.savez_compressed(
            cache,
            vertices=vertices,
            faces=faces,
            source_size=np.asarray(source_size, dtype=np.int64),
            cache_version=np.asarray(MESH_CACHE_VERSION, dtype=np.int64),
            preprocessing=np.asarray(preprocessing),
        )

    mesh = TriangleMesh(
        torch.as_tensor(vertices, dtype=torch.float32, device=device),
        torch.as_tensor(faces, dtype=torch.long, device=device),
    )
    enforce_face_limit(
        int(mesh.faces.shape[0]),
        limit=DEFAULT_FACE_LIMIT,
        allow_large_mesh=allow_large_mesh,
        context=f"asteroid {model_num} proxy",
    )
    validation = validate_triangle_mesh(mesh)
    if not validation.valid:
        raise ValueError(f"invalid benchmark mesh: {validation.reason}")
    return mesh


def normalized_curves(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    means = values.mean(axis=0, keepdims=True)
    if np.any(~np.isfinite(means)) or np.any(np.abs(means) < 1e-12):
        raise ValueError("cannot normalize a curve with a zero or invalid mean")
    return values / means


def rotate_mesh_about_z(mesh: TriangleMesh, degrees: float) -> TriangleMesh:
    """Apply the fitted preprocessing rotation to a mesh about the spin axis."""
    theta = math.radians(degrees)
    rotation = mesh.vertices.new_tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return TriangleMesh(mesh.vertices @ rotation.T, mesh.faces)


def periodic_shift(
    values: np.ndarray, degrees: np.ndarray, shift_degrees: float
) -> np.ndarray:
    """Evaluate periodic samples at theta + shift using linear interpolation."""
    values = np.asarray(values)
    query = np.mod(np.asarray(degrees) + shift_degrees, 360.0)
    if values.ndim == 1:
        return np.interp(query, degrees, values, period=360.0)
    return np.column_stack(
        [np.interp(query, degrees, values[:, i], period=360.0)
         for i in range(values.shape[1])]
    )


def estimate_phase(
    predicted: np.ndarray,
    observed: np.ndarray,
    degrees: np.ndarray,
    step_degrees: float = 1.0,
) -> tuple[float, float]:
    candidates = np.arange(-180.0, 180.0, step_degrees)
    scores = np.asarray(
        [
            np.mean(
                (periodic_shift(predicted, degrees, shift) - observed) ** 2
            )
            for shift in candidates
        ]
    )
    best = int(np.argmin(scores))
    return float(candidates[best]), float(math.sqrt(scores[best]))


def phase_objective(
    predicted: np.ndarray,
    observed: np.ndarray,
    degrees: np.ndarray,
    shift_degrees: float,
    column_indices: Iterable[int] | None = None,
) -> float:
    """Mean per-curve RMSE, so high-amplitude curves do not dominate phase."""
    if column_indices is None:
        column_indices = range(predicted.shape[1])
    index = list(column_indices)
    shifted = periodic_shift(predicted[:, index], degrees, shift_degrees)
    per_curve_rmse = np.sqrt(
        np.mean((shifted - observed[:, index]) ** 2, axis=0)
    )
    return float(per_curve_rmse.mean())


def curve_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    residual = predicted - observed
    pred_std = float(predicted.std())
    obs_std = float(observed.std())
    if pred_std < 1e-12 or obs_std < 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(predicted, observed)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "correlation": correlation,
        "amplitude_ratio": float(pred_std / obs_std) if obs_std > 1e-12 else float("nan"),
    }


def _sample_data(model_num: int, frame_stride: int) -> tuple[np.ndarray, np.ndarray]:
    intensity = load_data(model_num, "simulated", "intensity").data
    binary = load_data(model_num, "simulated", "binary").data
    if list(intensity.columns) != list(binary.columns):
        raise ValueError("intensity and binary columns do not match")
    missing = set(UNIQUE_COLUMNS) - set(intensity.columns)
    if missing:
        raise ValueError(f"missing expected camera columns: {sorted(missing)}")
    return (
        intensity.loc[:, UNIQUE_COLUMNS].iloc[::frame_stride].to_numpy(),
        binary.loc[:, UNIQUE_COLUMNS].iloc[::frame_stride].to_numpy(),
    )


def build_runtime(
    model_num: int,
    config: BenchmarkConfig,
    device: torch.device,
    cache_dir: Path,
) -> AsteroidRuntime:
    intensity, binary = _sample_data(model_num, config.frame_stride)
    n_frames = intensity.shape[0]
    degrees = np.arange(0, 360, config.frame_stride, dtype=np.float64)[:n_frames]
    angles = torch.as_tensor(np.radians(degrees), dtype=torch.float32, device=device)
    view_dirs = build_view_dirs(UNIQUE_COLUMNS, device)
    sun_dir = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    geometry = ObservationGeometry.build(angles, sun_dir, view_dirs)
    renderer = PhotometricForwardModel(
        geometry,
        brdf_type="combined",
        projection=config.projection,
        camera_distance_radii=config.camera_distance_radii,
        visibility_samples=config.visibility_samples,
        frame_chunk=max(1, min(60, n_frames)),
        direction_chunk=1 if device.type == "cpu" else 8,
    )
    return AsteroidRuntime(
        model_num=model_num,
        mesh=load_benchmark_mesh(
            model_num,
            dict(config.face_count_overrides).get(model_num, config.target_faces),
            device,
            cache_dir,
            config.mesh_preprocessing,
            config.allow_large_mesh,
        ),
        columns=list(UNIQUE_COLUMNS),
        degrees=degrees,
        intensity=intensity,
        binary=binary,
        renderer=renderer,
        prepared={},
        endpoints={},
    )


@torch.no_grad()
def render_intensity_endpoints(
    runtime: AsteroidRuntime, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    if gamma in runtime.endpoints:
        return runtime.endpoints[gamma]
    prepared = runtime.renderer.prepare_visibility(runtime.mesh, gamma, detach=True)
    runtime.prepared[gamma] = prepared
    runtime.renderer.lambertian_fraction = 1.0
    lambert = runtime.renderer.predict(
        runtime.mesh, gamma, prepared=prepared
    ).intensity.cpu().numpy()
    runtime.renderer.lambertian_fraction = 0.0
    lommel = runtime.renderer.predict(
        runtime.mesh, gamma, prepared=prepared
    ).intensity.cpu().numpy()
    runtime.endpoints[gamma] = (lambert, lommel)
    return lambert, lommel


def mixed_intensity(
    runtime: AsteroidRuntime, gamma: float, lambertian_fraction: float
) -> np.ndarray:
    lambert, lommel = render_intensity_endpoints(runtime, gamma)
    return normalized_curves(
        lambertian_fraction * lambert
        + (1.0 - lambertian_fraction) * lommel
    )


def calibration_indices(columns: Iterable[str]) -> list[int]:
    return [i for i, column in enumerate(columns) if column in CALIBRATION_COLUMNS]


def fit_intensity_parameters(
    runtimes: dict[int, AsteroidRuntime], config: BenchmarkConfig
) -> tuple[dict[str, float], dict[int, float], list[dict[str, float]]]:
    calibration_bodies = [m for m in (1, 2) if m in runtimes]
    if not calibration_bodies:
        calibration_bodies = list(runtimes)
    rows: list[dict[str, float]] = []
    alignment_index = UNIQUE_COLUMNS.index(ALIGNMENT_COLUMN)
    train_index = calibration_indices(UNIQUE_COLUMNS)

    for gamma in config.gamma_grid:
        for weight in config.lambertian_grid:
            squared_errors = []
            phases = {}
            for model_num in calibration_bodies:
                runtime = runtimes[model_num]
                predicted = mixed_intensity(runtime, gamma, weight)
                phase, _ = estimate_phase(
                    predicted[:, alignment_index],
                    runtime.intensity[:, alignment_index],
                    runtime.degrees,
                    config.phase_step_degrees,
                )
                phases[model_num] = phase
                shifted = periodic_shift(predicted, runtime.degrees, phase)
                squared_errors.append(
                    (shifted[:, train_index] - runtime.intensity[:, train_index]) ** 2
                )
            rmse = float(np.sqrt(np.mean(np.concatenate(
                [item.reshape(-1) for item in squared_errors]
            ))))
            rows.append(
                {"gamma": float(gamma), "lambertian_fraction": float(weight), "rmse": rmse}
            )

    best = min(rows, key=lambda row: row["rmse"])
    phases: dict[int, float] = {}
    for model_num, runtime in runtimes.items():
        predicted = mixed_intensity(
            runtime, best["gamma"], best["lambertian_fraction"]
        )
        phase, _ = estimate_phase(
            predicted[:, alignment_index],
            runtime.intensity[:, alignment_index],
            runtime.degrees,
            config.phase_step_degrees,
        )
        phases[model_num] = phase
    return best, phases, rows


@torch.no_grad()
def render_binary(
    runtime: AsteroidRuntime,
    gamma: float,
    lambertian_fraction: float,
    tau: float,
    beta: float,
) -> np.ndarray:
    prepared = runtime.prepared.get(gamma)
    if prepared is None:
        render_intensity_endpoints(runtime, gamma)
        prepared = runtime.prepared[gamma]
    runtime.renderer.lambertian_fraction = lambertian_fraction
    runtime.renderer.tau_binary = tau
    runtime.renderer.beta_binary = beta
    prediction = runtime.renderer.predict(
        runtime.mesh, gamma, prepared=prepared, include_binary=True
    ).binary
    assert prediction is not None
    return normalized_curves(prediction.cpu().numpy())


def fit_binary_parameters(
    runtimes: dict[int, AsteroidRuntime],
    config: BenchmarkConfig,
    intensity_parameters: dict[str, float],
    phases: dict[int, float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    calibration_bodies = [m for m in (1, 2) if m in runtimes]
    if not calibration_bodies:
        calibration_bodies = list(runtimes)
    train_index = calibration_indices(UNIQUE_COLUMNS)
    rows: list[dict[str, float]] = []
    for tau in config.tau_grid:
        for beta in config.beta_grid:
            squared_errors = []
            for model_num in calibration_bodies:
                runtime = runtimes[model_num]
                predicted = render_binary(
                    runtime,
                    intensity_parameters["gamma"],
                    intensity_parameters["lambertian_fraction"],
                    tau,
                    beta,
                )
                shifted = periodic_shift(
                    predicted, runtime.degrees, phases[model_num]
                )
                squared_errors.append(
                    (shifted[:, train_index] - runtime.binary[:, train_index]) ** 2
                )
            rmse = float(np.sqrt(np.mean(np.concatenate(
                [item.reshape(-1) for item in squared_errors]
            ))))
            rows.append({"tau_binary": float(tau), "beta_binary": float(beta), "rmse": rmse})
    return min(rows, key=lambda row: row["rmse"]), rows


def fit_asteroid_calibration(
    runtime: AsteroidRuntime,
    config: BenchmarkConfig,
) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    """Jointly fit z rotation, scattering mixture, and visibility softness."""
    missing = set(config.calibration_columns) - set(runtime.columns)
    if missing:
        raise ValueError(f"unknown calibration columns: {sorted(missing)}")
    indices = [runtime.columns.index(column) for column in config.calibration_columns]
    shifts = np.arange(-180.0, 180.0, config.phase_step_degrees)
    search_rows: list[dict[str, float]] = []

    for gamma in config.gamma_grid:
        for weight in config.lambertian_grid:
            predicted = mixed_intensity(runtime, gamma, weight)
            objectives = np.asarray(
                [
                    phase_objective(
                        predicted,
                        runtime.intensity,
                        runtime.degrees,
                        shift,
                        indices,
                    )
                    for shift in shifts
                ]
            )
            best_index = int(np.argmin(objectives))
            search_rows.append(
                {
                    "asteroid": runtime.model_num,
                    "search_kind": "intensity",
                    "gamma": float(gamma),
                    "lambertian_fraction": float(weight),
                    "curve_shift_degrees": float(shifts[best_index]),
                    "mesh_rotation_degrees": float(-shifts[best_index]),
                    "mean_curve_rmse": float(objectives[best_index]),
                }
            )

    best = min(search_rows, key=lambda row: row["mean_curve_rmse"])
    predicted = mixed_intensity(
        runtime, best["gamma"], best["lambertian_fraction"]
    )
    phase_rows = [
        {
            "asteroid": runtime.model_num,
            "curve_shift_degrees": float(shift),
            "mesh_rotation_degrees": float(-shift),
            "mean_curve_rmse": phase_objective(
                predicted,
                runtime.intensity,
                runtime.degrees,
                shift,
                indices,
            ),
        }
        for shift in shifts
    ]

    binary_rows: list[dict[str, float]] = []
    for tau in config.tau_grid:
        for beta in config.beta_grid:
            binary = render_binary(
                runtime,
                best["gamma"],
                best["lambertian_fraction"],
                tau,
                beta,
            )
            objective = phase_objective(
                binary,
                runtime.binary,
                runtime.degrees,
                best["curve_shift_degrees"],
                indices,
            )
            binary_rows.append(
                {
                    "tau_binary": float(tau),
                    "beta_binary": float(beta),
                    "mean_curve_rmse": objective,
                }
            )
    binary_best = min(binary_rows, key=lambda row: row["mean_curve_rmse"])
    parameters = {
        "mesh_faces": int(runtime.mesh.faces.shape[0]),
        "gamma": best["gamma"],
        "lambertian_fraction": best["lambertian_fraction"],
        "curve_shift_degrees": best["curve_shift_degrees"],
        "mesh_rotation_degrees": best["mesh_rotation_degrees"],
        "tau_binary": binary_best["tau_binary"],
        "beta_binary": binary_best["beta_binary"],
        "intensity_mean_curve_rmse": best["mean_curve_rmse"],
        "binary_mean_curve_rmse": binary_best["mean_curve_rmse"],
    }
    for row in binary_rows:
        row.update(
            asteroid=runtime.model_num,
            search_kind="binary",
            gamma=best["gamma"],
            lambertian_fraction=best["lambertian_fraction"],
            curve_shift_degrees=best["curve_shift_degrees"],
        )
    return parameters, search_rows + binary_rows, phase_rows


def baseline_predictions(runtime: AsteroidRuntime) -> dict[str, np.ndarray]:
    intensity = mixed_intensity(runtime, 0.02, 0.5)
    binary = render_binary(runtime, 0.02, 0.5, 0.10, 8.0)
    return {"intensity": intensity, "binary": binary}


def calibrated_predictions(
    runtime: AsteroidRuntime, parameters: dict[str, float]
) -> dict[str, np.ndarray]:
    intensity = mixed_intensity(
        runtime, parameters["gamma"], parameters["lambertian_fraction"]
    )
    binary = render_binary(
        runtime,
        parameters["gamma"],
        parameters["lambertian_fraction"],
        parameters["tau_binary"],
        parameters["beta_binary"],
    )
    shift = parameters["curve_shift_degrees"]
    return {
        "intensity": periodic_shift(intensity, runtime.degrees, shift),
        "binary": periodic_shift(binary, runtime.degrees, shift),
    }


def evaluate_before_after(
    runtimes: dict[int, AsteroidRuntime],
    parameters: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, dict[int, dict[str, dict[str, np.ndarray]]]]:
    rows: list[dict] = []
    predictions: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for model_num, runtime in runtimes.items():
        states = {
            "uncalibrated": baseline_predictions(runtime),
            "calibrated": calibrated_predictions(runtime, parameters[model_num]),
        }
        predictions[model_num] = states
        for state, state_predictions in states.items():
            for observable in ("intensity", "binary"):
                observed = getattr(runtime, observable)
                predicted = state_predictions[observable]
                for index, column in enumerate(runtime.columns):
                    azimuth, camera = column.split("_")
                    rows.append(
                        {
                            "asteroid": model_num,
                            "state": state,
                            "observable": observable,
                            "column": column,
                            "azimuth_degrees": int(azimuth),
                            "camera": camera,
                            **curve_metrics(predicted[:, index], observed[:, index]),
                        }
                    )
    return pd.DataFrame(rows), predictions


def curve_split(model_num: int, column: str) -> str:
    if column == ALIGNMENT_COLUMN:
        return "alignment"
    if model_num in (1, 2) and column in CALIBRATION_COLUMNS:
        return "calibration"
    return "validation"


def evaluate_predictions(
    runtimes: dict[int, AsteroidRuntime],
    intensity_parameters: dict[str, float],
    binary_parameters: dict[str, float],
    phases: dict[int, float],
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    rows = []
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for model_num, runtime in runtimes.items():
        intensity = mixed_intensity(
            runtime,
            intensity_parameters["gamma"],
            intensity_parameters["lambertian_fraction"],
        )
        binary = render_binary(
            runtime,
            intensity_parameters["gamma"],
            intensity_parameters["lambertian_fraction"],
            binary_parameters["tau_binary"],
            binary_parameters["beta_binary"],
        )
        intensity = periodic_shift(intensity, runtime.degrees, phases[model_num])
        binary = periodic_shift(binary, runtime.degrees, phases[model_num])
        predictions[model_num] = {"intensity": intensity, "binary": binary}
        for observable, predicted, observed in (
            ("intensity", intensity, runtime.intensity),
            ("binary", binary, runtime.binary),
        ):
            for index, column in enumerate(runtime.columns):
                azimuth, camera = column.split("_")
                rows.append(
                    {
                        "asteroid": model_num,
                        "observable": observable,
                        "column": column,
                        "azimuth_degrees": int(azimuth),
                        "camera": camera,
                        "split": curve_split(model_num, column),
                        "phase_degrees": phases[model_num],
                        **curve_metrics(predicted[:, index], observed[:, index]),
                    }
                )
    return pd.DataFrame(rows), predictions


def aggregate_metrics(metrics: pd.DataFrame) -> list[dict]:
    result = []
    grouping = ["asteroid", "observable"]
    if "state" in metrics:
        grouping.insert(1, "state")
    elif "split" in metrics:
        grouping.append("split")
    for keys, group in metrics.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        row["asteroid"] = int(row["asteroid"])
        row.update(
            n_curves=int(len(group)),
            mean_rmse=float(group.rmse.mean()),
            mean_correlation=float(group.correlation.mean()),
            mean_amplitude_ratio=float(group.amplitude_ratio.mean()),
        )
        result.append(row)
    return result


def plot_overlays(
    runtimes: dict[int, AsteroidRuntime],
    predictions: dict[int, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    shown = ["0_hor1", "90_top", "225_bottom"]
    for model_num, runtime in runtimes.items():
        for observable in ("intensity", "binary"):
            observed = getattr(runtime, observable)
            figure, axes = plt.subplots(len(shown), 1, figsize=(10, 8), sharex=True)
            for axis, column in zip(axes, shown):
                index = runtime.columns.index(column)
                axis.plot(runtime.degrees, observed[:, index], label="Blender", linewidth=1.8)
                axis.plot(
                    runtime.degrees,
                    predictions[model_num][observable][:, index],
                    "o-",
                    markersize=2.5,
                    label="forward model",
                )
                axis.set_ylabel(column)
                axis.grid(alpha=0.25)
            axes[0].legend()
            axes[-1].set_xlabel("rotation angle [degrees]")
            figure.suptitle(f"Asteroid {model_num}: normalized {observable}")
            figure.tight_layout()
            figure.savefig(output_dir / f"asteroid{model_num}_{observable}_overlays.png", dpi=150)
            plt.close(figure)


def plot_residual_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    validation = metrics[metrics.split == "validation"]
    if validation.empty:
        return
    summary = validation.groupby(
        ["observable", "azimuth_degrees", "camera"], as_index=False
    ).rmse.mean()
    observables = list(summary.observable.unique())
    figure, axes = plt.subplots(1, len(observables), figsize=(6 * len(observables), 4), squeeze=False)
    for axis, observable in zip(axes[0], observables):
        subset = summary[summary.observable == observable]
        for camera, group in subset.groupby("camera"):
            axis.plot(group.azimuth_degrees, group.rmse, "o-", label=camera)
        axis.set_title(observable)
        axis.set_xlabel("camera azimuth / phase angle [degrees]")
        axis.set_ylabel("mean normalized RMSE")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "residuals_by_geometry.png", dpi=150)
    plt.close(figure)


def plot_before_after_overlays(
    runtimes: dict[int, AsteroidRuntime],
    predictions: dict[int, dict[str, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    shown = ["0_hor1", "90_top", "225_bottom"]
    for model_num, runtime in runtimes.items():
        for observable in ("intensity", "binary"):
            observed = getattr(runtime, observable)
            figure, axes = plt.subplots(len(shown), 1, figsize=(11, 8), sharex=True)
            for axis, column in zip(axes, shown):
                index = runtime.columns.index(column)
                axis.plot(
                    runtime.degrees, observed[:, index], color="black",
                    linewidth=2.0, label="Blender reference"
                )
                axis.plot(
                    runtime.degrees,
                    predictions[model_num]["uncalibrated"][observable][:, index],
                    color="tab:red", linestyle="--", alpha=0.75,
                    label="uncalibrated",
                )
                axis.plot(
                    runtime.degrees,
                    predictions[model_num]["calibrated"][observable][:, index],
                    color="tab:blue", marker="o", markersize=2.5,
                    linewidth=1.4, label="calibrated",
                )
                axis.set_ylabel(column)
                axis.grid(alpha=0.25)
            axes[0].legend(ncol=3)
            axes[-1].set_xlabel("rotation angle [degrees]")
            figure.suptitle(
                f"Asteroid {model_num}: {observable} before/after calibration"
            )
            figure.tight_layout()
            figure.savefig(
                output_dir / f"asteroid{model_num}_{observable}_before_after.png",
                dpi=160,
            )
            plt.close(figure)


def plot_phase_objectives(phase_search: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(phase_search.asteroid.unique()), 1, figsize=(10, 3.3 * len(phase_search.asteroid.unique())),
        sharex=True, squeeze=False,
    )
    for axis, (model_num, group) in zip(axes[:, 0], phase_search.groupby("asteroid")):
        best = group.loc[group.mean_curve_rmse.idxmin()]
        axis.plot(group.mesh_rotation_degrees, group.mean_curve_rmse)
        axis.axvline(
            best.mesh_rotation_degrees, color="tab:red", linestyle="--",
            label=f"best mesh rotation = {best.mesh_rotation_degrees:.1f}°",
        )
        axis.set_ylabel("mean curve RMSE")
        axis.set_title(f"Asteroid {model_num}")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1, 0].set_xlabel("mesh z-rotation [degrees]")
    figure.suptitle("Joint multi-camera phase objective")
    figure.tight_layout()
    figure.savefig(output_dir / "phase_alignment_objectives.png", dpi=160)
    plt.close(figure)


def plot_calibration_surfaces(search: pd.DataFrame, output_dir: Path) -> None:
    intensity = search[search.search_kind == "intensity"]
    asteroids = sorted(intensity.asteroid.unique())
    figure, axes = plt.subplots(1, len(asteroids), figsize=(5 * len(asteroids), 4), squeeze=False)
    for axis, model_num in zip(axes[0], asteroids):
        subset = intensity[intensity.asteroid == model_num]
        table = subset.pivot(
            index="gamma", columns="lambertian_fraction", values="mean_curve_rmse"
        ).sort_index(ascending=False)
        image_handle = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(table.columns)), [f"{v:g}" for v in table.columns])
        axis.set_yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
        axis.set_xlabel("Lambertian fraction")
        axis.set_ylabel("visibility softness γ")
        axis.set_title(f"Asteroid {model_num}")
        figure.colorbar(image_handle, ax=axis, label="best phase RMSE")
    figure.suptitle("Reflectivity/visibility calibration surface")
    figure.tight_layout()
    figure.savefig(output_dir / "reflectivity_calibration_surface.png", dpi=160)
    plt.close(figure)


def plot_error_heatmaps(metrics: pd.DataFrame, output_dir: Path) -> None:
    cameras = ["top", "hor1", "bottom"]
    azimuths = [0, 45, 90, 135, 225, 270, 315]
    for model_num in sorted(metrics.asteroid.unique()):
        figure, axes = plt.subplots(2, 2, figsize=(13, 7), squeeze=False)
        for row, observable in enumerate(("intensity", "binary")):
            observable_values = metrics[
                (metrics.asteroid == model_num)
                & (metrics.observable == observable)
            ]
            vmax = float(observable_values.rmse.max())
            for column, state in enumerate(("uncalibrated", "calibrated")):
                subset = observable_values[observable_values.state == state]
                matrix = np.full((len(cameras), len(azimuths)), np.nan)
                for i, camera in enumerate(cameras):
                    for j, azimuth in enumerate(azimuths):
                        match = subset[
                            (subset.camera == camera)
                            & (subset.azimuth_degrees == azimuth)
                        ]
                        if not match.empty:
                            matrix[i, j] = float(match.rmse.iloc[0])
                handle = axes[row, column].imshow(
                    matrix, aspect="auto", vmin=0.0, vmax=vmax, cmap="magma"
                )
                axes[row, column].set_xticks(range(len(azimuths)), azimuths)
                axes[row, column].set_yticks(range(len(cameras)), cameras)
                axes[row, column].set_title(f"{observable} — {state}")
                axes[row, column].set_xlabel("camera azimuth [degrees]")
                figure.colorbar(handle, ax=axes[row, column], label="normalized RMSE")
        figure.suptitle(f"Asteroid {model_num}: per-camera residual map")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"asteroid{model_num}_residual_heatmaps.png", dpi=160
        )
        plt.close(figure)


def plot_mesh_alignment(
    runtimes: dict[int, AsteroidRuntime],
    parameters: dict[int, dict[str, float]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, len(runtimes), figsize=(5 * len(runtimes), 5), squeeze=False)
    for axis, (model_num, runtime) in zip(axes[0], runtimes.items()):
        vertices = runtime.mesh.vertices.detach().cpu().numpy()
        faces = runtime.mesh.faces.detach().cpu().numpy()
        edges = np.unique(
            np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
            axis=0,
        )
        theta = math.radians(parameters[model_num]["mesh_rotation_degrees"])
        rotation = np.asarray(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        rotated = vertices[:, :2] @ rotation.T
        for xy, color, label, alpha in (
            (vertices[:, :2], "0.65", "published pose", 0.35),
            (rotated, "tab:blue", "calibrated pose", 0.7),
        ):
            segments = xy[edges]
            axis.add_collection(
                LineCollection(
                    segments, colors=color, linewidths=0.35, alpha=alpha, label=label
                )
            )
        extent = 1.1 * max(np.abs(vertices[:, :2]).max(), np.abs(rotated).max())
        axis.annotate(
            "sun", xy=(-0.55 * extent, 0), xytext=(-extent, 0),
            arrowprops=dict(arrowstyle="->", color="goldenrod", lw=2),
            color="goldenrod", va="center",
        )
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_aspect("equal")
        axis.set_title(
            f"Asteroid {model_num}\nrotation {parameters[model_num]['mesh_rotation_degrees']:.1f}°"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    handles = [
        plt.Line2D([0], [0], color="0.65", label="published pose"),
        plt.Line2D([0], [0], color="tab:blue", label="calibrated pose"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.suptitle("Top-view mesh preprocessing (light from −x)")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output_dir / "calibrated_mesh_orientations.png", dpi=160)
    plt.close(figure)


def plot_calibration_summary(
    metrics: pd.DataFrame,
    parameters: dict[int, dict[str, float]],
    output_dir: Path,
) -> None:
    summary = metrics.groupby(["asteroid", "state", "observable"], as_index=False).rmse.mean()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.18
    x = np.arange(len(parameters))
    for offset, (state, observable, color) in enumerate(
        [
            ("uncalibrated", "intensity", "#d95f02"),
            ("calibrated", "intensity", "#1b9e77"),
            ("uncalibrated", "binary", "#e6ab02"),
            ("calibrated", "binary", "#7570b3"),
        ]
    ):
        values = []
        for model_num in parameters:
            match = summary[
                (summary.asteroid == model_num)
                & (summary.state == state)
                & (summary.observable == observable)
            ]
            values.append(float(match.rmse.iloc[0]))
        axes[0].bar(
            x + (offset - 1.5) * width, values, width,
            label=f"{observable} {state}", color=color,
        )
    axes[0].set_xticks(x, [f"Asteroid {m}" for m in parameters])
    axes[0].set_ylabel("mean per-curve RMSE")
    axes[0].set_title("Fit improvement")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    table_rows = []
    for model_num, values in parameters.items():
        table_rows.append(
            [
                model_num,
                f"{int(values['mesh_faces'])}",
                f"{values['mesh_rotation_degrees']:.1f}°",
                f"{values['gamma']:.3f}",
                f"{values['lambertian_fraction']:.2f}",
                f"{values['tau_binary']:.2f}",
                f"{values['beta_binary']:.0f}",
            ]
        )
    axes[1].axis("off")
    axes[1].table(
        cellText=table_rows,
        colLabels=["body", "faces", "z rotation", "γ", "Lambert frac.", "binary τ", "binary β"],
        loc="center",
        cellLoc="center",
    )
    axes[1].set_title("Fitted preprocessing parameters")
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_summary.png", dpi=160)
    plt.close(figure)


@torch.no_grad()
def run_convergence(
    config: BenchmarkConfig,
    device: torch.device,
    output_dir: Path,
    calibration_parameters: dict[int, dict[str, float]],
) -> pd.DataFrame:
    columns = ["0_hor1", "90_top", "225_bottom"]
    degrees = np.arange(0.0, 360.0, 30.0)
    angles = torch.as_tensor(np.radians(degrees), dtype=torch.float32, device=device)
    views = build_view_dirs(columns, device)
    sun = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    geometry = ObservationGeometry.build(angles, sun, views)
    outputs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    face_counts = sorted(set(config.convergence_faces))
    for model_num in config.asteroids:
        parameters = calibration_parameters[model_num]
        for face_count in face_counts:
            mesh = load_benchmark_mesh(
                model_num,
                face_count,
                device,
                output_dir / "mesh_cache",
                config.mesh_preprocessing,
                config.allow_large_mesh,
            )
            renderer = PhotometricForwardModel(
                geometry,
                brdf_type="combined",
                lambertian_fraction=parameters["lambertian_fraction"],
                projection=config.projection,
                camera_distance_radii=config.camera_distance_radii,
                visibility_samples=config.visibility_samples,
                tau_binary=parameters["tau_binary"],
                beta_binary=parameters["beta_binary"],
                direction_chunk=1 if device.type == "cpu" else 8,
            )
            prediction = renderer.predict(
                mesh, parameters["gamma"], include_binary=True
            )
            assert prediction.binary is not None
            outputs[(model_num, face_count)] = (
                normalized_curves(prediction.intensity.cpu().numpy()),
                normalized_curves(prediction.binary.cpu().numpy()),
            )

    rows = []
    reference_faces = face_counts[-1]
    for model_num in config.asteroids:
        reference = outputs[(model_num, reference_faces)]
        for face_count in face_counts:
            candidate = outputs[(model_num, face_count)]
            for observable, predicted, observed in (
                ("intensity", candidate[0], reference[0]),
                ("binary", candidate[1], reference[1]),
            ):
                rows.append(
                    {
                        "asteroid": model_num,
                        "observable": observable,
                        "face_count": face_count,
                        "reference_face_count": reference_faces,
                        **curve_metrics(predicted.reshape(-1), observed.reshape(-1)),
                    }
                )
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for axis, observable in zip(axes[0], ("intensity", "binary")):
        subset = frame[frame.observable == observable]
        for model_num, group in subset.groupby("asteroid"):
            axis.plot(group.face_count, group.rmse, "o-", label=f"asteroid {model_num}")
        axis.set_title(observable)
        axis.set_xlabel("mesh faces")
        axis.set_ylabel(f"RMSE vs {reference_faces}-face result")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "mesh_convergence.png", dpi=150)
    plt.close(figure)
    return frame


def run_benchmark(config: BenchmarkConfig, output_dir: Path) -> dict:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    runtimes = {
        model_num: build_runtime(
            model_num, config, device, output_dir / "mesh_cache"
        )
        for model_num in config.asteroids
    }
    calibration_parameters: dict[int, dict[str, float]] = {}
    search_rows: list[dict] = []
    phase_rows: list[dict] = []
    for model_num, runtime in runtimes.items():
        parameters, body_search, body_phase = fit_asteroid_calibration(
            runtime, config
        )
        calibration_parameters[model_num] = parameters
        search_rows.extend(body_search)
        phase_rows.extend(body_phase)

        calibrated_mesh = rotate_mesh_about_z(
            runtime.mesh, parameters["mesh_rotation_degrees"]
        )
        trimesh.Trimesh(
            calibrated_mesh.vertices.detach().cpu().numpy(),
            calibrated_mesh.faces.detach().cpu().numpy(),
            process=False,
        ).export(output_dir / f"asteroid{model_num}_calibrated_mesh.stl")

    metrics, predictions = evaluate_before_after(
        runtimes, calibration_parameters
    )
    search_frame = pd.DataFrame(search_rows)
    phase_frame = pd.DataFrame(phase_rows)
    metrics.to_csv(output_dir / "curve_metrics_before_after.csv", index=False)
    search_frame.to_csv(output_dir / "calibration_parameter_search.csv", index=False)
    phase_frame.to_csv(output_dir / "phase_objective.csv", index=False)
    with (output_dir / "calibration_parameters.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {str(key): value for key, value in calibration_parameters.items()},
            stream,
            indent=2,
        )

    plot_before_after_overlays(runtimes, predictions, output_dir)
    plot_phase_objectives(phase_frame, output_dir)
    plot_calibration_surfaces(search_frame, output_dir)
    plot_error_heatmaps(metrics, output_dir)
    plot_mesh_alignment(runtimes, calibration_parameters, output_dir)
    plot_calibration_summary(metrics, calibration_parameters, output_dir)

    convergence_rows = []
    if config.convergence:
        convergence = run_convergence(
            config, device, output_dir, calibration_parameters
        )
        convergence.to_csv(output_dir / "mesh_convergence.csv", index=False)
        convergence_rows = convergence.to_dict(orient="records")

    result = {
        "config": asdict(config),
        "device": str(device),
        "calibration_parameters": {
            str(key): value for key, value in calibration_parameters.items()
        },
        "aggregate_metrics": aggregate_metrics(metrics),
        "convergence": convergence_rows,
        "runtime_seconds": time.perf_counter() - start,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asteroids", default="1")
    parser.add_argument("--target-faces", type=int, default=1280)
    parser.add_argument(
        "--mesh-preprocessing",
        choices=("star_proxy", "qem"),
        default="star_proxy",
        help=(
            "Use the optimizer's radial icosphere topology (default) or "
            "quality-first quadric decimation."
        ),
    )
    parser.add_argument(
        "--face-count-overrides",
        default="",
        help="Per-body overrides such as '2:1024'; use an empty string for none.",
    )
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--gamma-grid", default="0.01,0.02,0.04")
    parser.add_argument("--lambertian-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--tau-grid", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--beta-grid", default="8,16,32")
    parser.add_argument("--phase-step-degrees", type=float, default=1.0)
    parser.add_argument(
        "--calibration-columns",
        default=",".join(CALIBRATION_COLUMNS),
        help="Comma-separated calibration camera subset, or 'all'.",
    )
    parser.add_argument("--convergence-faces", default="80,320,1280")
    parser.add_argument(
        "--convergence", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--projection",
        choices=("orthographic", "perspective"),
        default="orthographic",
    )
    parser.add_argument("--camera-distance-radii", type=float, default=5.0)
    parser.add_argument(
        "--visibility-samples", type=int, choices=(1, 3), default=1
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/forward_validation")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny CPU-friendly configuration for a smoke test.",
    )
    parser.add_argument(
        "--allow-large-mesh",
        action="store_true",
        help=f"Explicitly override the default {DEFAULT_FACE_LIMIT:,}-face safety limit.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    calibration_columns = (
        tuple(UNIQUE_COLUMNS)
        if args.calibration_columns == "all"
        else _parse_csv(args.calibration_columns, str)
    )
    face_count_overrides = tuple(
        (int(item.split(":", 1)[0]), int(item.split(":", 1)[1]))
        for item in args.face_count_overrides.split(",")
        if item.strip()
    )
    config = BenchmarkConfig(
        asteroids=_parse_csv(args.asteroids, int),
        target_faces=args.target_faces,
        face_count_overrides=face_count_overrides,
        mesh_preprocessing=args.mesh_preprocessing,
        frame_stride=args.frame_stride,
        gamma_grid=_parse_csv(args.gamma_grid),
        lambertian_grid=_parse_csv(args.lambertian_grid),
        tau_grid=_parse_csv(args.tau_grid),
        beta_grid=_parse_csv(args.beta_grid),
        phase_step_degrees=args.phase_step_degrees,
        calibration_columns=calibration_columns,
        convergence_faces=_parse_csv(args.convergence_faces, int),
        convergence=args.convergence,
        projection=args.projection,
        camera_distance_radii=args.camera_distance_radii,
        visibility_samples=args.visibility_samples,
        device=args.device,
        allow_large_mesh=args.allow_large_mesh,
    )
    if args.quick:
        config = BenchmarkConfig(
            asteroids=config.asteroids,
            target_faces=min(config.target_faces, 64),
            face_count_overrides=(),
            mesh_preprocessing=config.mesh_preprocessing,
            frame_stride=max(config.frame_stride, 60),
            gamma_grid=(0.02,),
            lambertian_grid=(0.5,),
            tau_grid=(0.10,),
            beta_grid=(8.0,),
            phase_step_degrees=6.0,
            calibration_columns=config.calibration_columns,
            convergence_faces=config.convergence_faces,
            convergence=False,
            projection=config.projection,
            camera_distance_radii=config.camera_distance_radii,
            visibility_samples=config.visibility_samples,
            device=config.device,
            allow_large_mesh=config.allow_large_mesh,
        )
    if not config.asteroids or any(item not in (1, 2, 3) for item in config.asteroids):
        raise ValueError("asteroids must be a non-empty subset of 1,2,3")
    if config.frame_stride <= 0 or 360 % config.frame_stride != 0:
        raise ValueError("frame_stride must be a positive divisor of 360")
    if config.projection == "perspective" and config.camera_distance_radii <= 1.0:
        raise ValueError("perspective camera distance must exceed one radius")
    if config.target_faces < 4 or any(
        model_num not in (1, 2, 3) or face_count < 4
        for model_num, face_count in config.face_count_overrides
    ):
        raise ValueError("face counts must be >= 4 and overrides must target asteroids 1-3")
    if config.mesh_preprocessing not in ("star_proxy", "qem"):
        raise ValueError("unknown mesh preprocessing mode")
    if not all((config.gamma_grid, config.lambertian_grid, config.tau_grid, config.beta_grid)):
        raise ValueError("parameter grids must not be empty")
    if any(not 0.0 <= item <= 1.0 for item in config.lambertian_grid):
        raise ValueError("lambertian grid values must lie in [0, 1]")
    unknown_columns = set(config.calibration_columns) - set(UNIQUE_COLUMNS)
    if not config.calibration_columns or unknown_columns:
        raise ValueError(
            f"calibration columns must be non-empty and unique; unknown: {sorted(unknown_columns)}"
        )
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_benchmark(config, args.output_dir)
    print(json.dumps(
        {
            "output_dir": str(args.output_dir),
            "calibration_parameters": result["calibration_parameters"],
            "runtime_seconds": result["runtime_seconds"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
