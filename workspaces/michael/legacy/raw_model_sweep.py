#!/usr/bin/env python3
"""Joint raw-mesh calibration of scattering, response, phase, and projection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running this legacy script from the legacy/ directory while importing
# the current modules in the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from forward_model import load_data
from raw_forward_validation import (
    EmbreeDirectRenderer,
    _rotate_z,
    camera_direction,
    curve_metrics,
    get_raw_mesh_path,
    load_gmsh22_binary_surface,
)


def _csv(values: str, cast=float) -> tuple:
    return tuple(cast(value.strip()) for value in values.split(",") if value.strip())


def _camera_phase_angle(column: str) -> float:
    sun = np.array([-1.0, 0.0, 0.0])
    cosine = np.clip(np.dot(sun, camera_direction(column)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def render_grid(
    model_num: int,
    *,
    resolution: int,
    parameter1: np.ndarray,
    parameter2: np.ndarray,
    scattering: str,
    shading: str,
    projection: str,
    camera_distance_radii: float,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int, float]:
    started = time.perf_counter()
    raw = load_gmsh22_binary_surface(get_raw_mesh_path(model_num))
    renderer = EmbreeDirectRenderer(raw, resolution=resolution)
    dataset = load_data(model_num, "simulated", "intensity").data
    columns = [column for column in dataset.columns if not column.endswith("hor2")]
    observed = dataset.loc[:, columns].to_numpy(dtype=np.float64)
    n_frames, n_cameras = observed.shape
    predictions = np.zeros(
        (n_frames, n_cameras, len(parameter1), len(parameter2)),
        dtype=np.float64,
    )
    sun0 = np.array([-1.0, 0.0, 0.0])
    views0 = [camera_direction(column) for column in columns]
    for frame in range(n_frames):
        angle = np.radians(float(frame))
        sun = _rotate_z(sun0, angle)
        for camera, view0 in enumerate(views0):
            render_args = {
                "shading": shading,
                "projection": projection,
                "camera_distance_radii": camera_distance_radii,
            }
            if scattering == "mixture":
                values = renderer.render_grid(
                    sun,
                    _rotate_z(view0, angle),
                    lambertian_fractions=parameter1,
                    intensity_thresholds=parameter2,
                    **render_args,
                )
            elif scattering == "hapke":
                values = renderer.render_hapke_grid(
                    sun,
                    _rotate_z(view0, angle),
                    single_scattering_albedos=parameter1,
                    phase_terms=parameter2,
                    **render_args,
                )
            else:
                raise ValueError(f"unknown scattering grid {scattering}")
            predictions[frame, camera] = values
    return (
        predictions,
        observed,
        columns,
        len(raw.vertices),
        len(raw.faces),
        time.perf_counter() - started,
    )


def calibrate(
    predictions: np.ndarray,
    observed: np.ndarray,
    columns: list[str],
    parameter1: np.ndarray,
    parameter2: np.ndarray,
    parameter1_name: str,
    parameter2_name: str,
    phases: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    means = predictions.mean(axis=0, keepdims=True)
    if np.any(means <= 0.0):
        raise ValueError("a rendered calibration curve has zero mean")
    normalized = predictions / means
    observed = observed / observed.mean(axis=0, keepdims=True)

    search_rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    best_prediction: np.ndarray | None = None
    for phase in phases:
        shifted = np.roll(normalized, -int(phase), axis=0)
        per_curve = np.sqrt(
            np.mean((shifted - observed[:, :, None, None]) ** 2, axis=0)
        )
        objective = per_curve.mean(axis=0)
        for wi, value1 in enumerate(parameter1):
            for ti, value2 in enumerate(parameter2):
                row = {
                    "phase_degrees": float(phase),
                    parameter1_name: float(value1),
                    parameter2_name: float(value2),
                    "mean_curve_rmse": float(objective[wi, ti]),
                    "max_curve_rmse": float(per_curve[:, wi, ti].max()),
                }
                search_rows.append(row)
                if best is None or row["mean_curve_rmse"] < best["mean_curve_rmse"]:
                    best = row.copy()
                    best_prediction = shifted[:, :, wi, ti].copy()

    assert best is not None and best_prediction is not None
    metric_rows = []
    independent_rows = []
    for camera, column in enumerate(columns):
        metrics = curve_metrics(best_prediction[:, camera], observed[:, camera])
        metric_rows.append(
            {
                "column": column,
                "phase_angle_degrees": _camera_phase_angle(column),
                **metrics,
            }
        )

        curve_best: dict[str, float] | None = None
        for phase in phases:
            shifted = np.roll(normalized[:, camera], -int(phase), axis=0)
            rmse = np.sqrt(
                np.mean((shifted - observed[:, camera, None, None]) ** 2, axis=0)
            )
            index = np.unravel_index(int(np.argmin(rmse)), rmse.shape)
            candidate = {
                "column": column,
                "phase_angle_degrees": _camera_phase_angle(column),
                "phase_degrees": float(phase),
                parameter1_name: float(parameter1[index[0]]),
                parameter2_name: float(parameter2[index[1]]),
                "rmse": float(rmse[index]),
            }
            if curve_best is None or candidate["rmse"] < curve_best["rmse"]:
                curve_best = candidate
        assert curve_best is not None
        independent_rows.append(curve_best)

    return (
        best,
        pd.DataFrame(search_rows),
        pd.DataFrame(metric_rows).merge(
            pd.DataFrame(independent_rows),
            on=["column", "phase_angle_degrees"],
            suffixes=("_global", "_independent"),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asteroids", default="1,2,3")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--scattering", choices=("mixture", "hapke"), default="mixture")
    parser.add_argument("--lambertian-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--threshold-grid", default="0,0.02,0.05,0.1,0.15,0.2")
    parser.add_argument("--albedo-grid", default="0,0.2,0.4,0.6,0.8,0.9,0.95,0.99")
    parser.add_argument("--phase-term-grid", default="0.25,0.5,1,2,4")
    parser.add_argument("--phase-grid", default="-15:15")
    parser.add_argument("--shading", choices=("flat", "smooth"), default="smooth")
    parser.add_argument(
        "--projection", choices=("orthographic", "perspective"), default="orthographic"
    )
    parser.add_argument("--camera-distance-radii", type=float, default=5.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/raw_model_sweep")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scattering == "mixture":
        parameter1_name, parameter2_name = (
            "lambertian_fraction", "intensity_threshold"
        )
        parameter1 = np.asarray(_csv(args.lambertian_grid), dtype=np.float64)
        parameter2 = np.asarray(_csv(args.threshold_grid), dtype=np.float64)
    else:
        parameter1_name, parameter2_name = (
            "single_scattering_albedo", "phase_term"
        )
        parameter1 = np.asarray(_csv(args.albedo_grid), dtype=np.float64)
        parameter2 = np.asarray(_csv(args.phase_term_grid), dtype=np.float64)
    phase_start, phase_stop = (int(value) for value in args.phase_grid.split(":"))
    phases = np.arange(phase_start, phase_stop + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_num in _csv(args.asteroids, int):
        prediction, observed, columns, n_vertices, n_faces, runtime = render_grid(
            model_num,
            resolution=args.resolution,
            parameter1=parameter1,
            parameter2=parameter2,
            scattering=args.scattering,
            shading=args.shading,
            projection=args.projection,
            camera_distance_radii=args.camera_distance_radii,
        )
        best, search, metrics = calibrate(
            prediction,
            observed,
            columns,
            parameter1,
            parameter2,
            parameter1_name,
            parameter2_name,
            phases,
        )
        search.insert(0, "asteroid", model_num)
        metrics.insert(0, "asteroid", model_num)
        search.to_csv(args.output_dir / f"asteroid{model_num}_search.csv", index=False)
        metrics.to_csv(args.output_dir / f"asteroid{model_num}_metrics.csv", index=False)
        summary = {
            "asteroid": model_num,
            "vertices": n_vertices,
            "faces": n_faces,
            "runtime_seconds": runtime,
            **best,
        }
        summaries.append(summary)
        print(json.dumps(summary))

    settings = {
        "resolution": args.resolution,
        "scattering": args.scattering,
        parameter1_name + "_grid": parameter1.tolist(),
        parameter2_name + "_grid": parameter2.tolist(),
        "phase_grid": phases.tolist(),
        "shading": args.shading,
        "projection": args.projection,
        "camera_distance_radii": args.camera_distance_radii,
        "mesh_policy": "untouched Gmsh 2.2 vertices and triangles",
        "summaries": summaries,
    }
    (args.output_dir / "results.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
