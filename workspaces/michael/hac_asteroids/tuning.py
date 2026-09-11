"""Deterministic, resumable hyperparameter tuning for staged inversion."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import Calibration
from .inversion import FitConfig, run_inversion


SEARCH_SPACE = {
    "initial_learning_rate": [4e-3, 1.2e-2],
    "escape_steps": [20, 40],
    "regularization_weight": [1e-3, 2e-2],
    "degree_power": [1.5, 2.0, 3.0],
    "gradient_clip": [5.0, 10.0, 20.0],
    "frame_count": [12, 24],
}

REUSABLE_FIELDS = (
    "steps",
    "learning_rate",
    "initial_learning_rate",
    "min_learning_rate",
    "escape_steps",
    "stabilization_steps",
    "frame_count",
    "validation_frame_count",
    "validation_interval",
    "evaluation_stride",
    "subdivisions",
    "lmax",
    "image_size",
    "regularization_weight",
    "degree_power",
    "gradient_clip",
    "checkpoint_interval",
)


def _log_scale(value: float, low: float, high: float) -> float:
    return math.exp(math.log(low) + value * (math.log(high) - math.log(low)))


def _choice(value: float, choices: list[Any]) -> Any:
    return choices[min(int(value * len(choices)), len(choices) - 1)]


def generate_candidates(count: int = 12, seed: int = 0) -> list[dict[str, Any]]:
    """Generate a reproducible, space-filling set of tuning configurations."""

    if count <= 0:
        raise ValueError("candidate count must be positive")
    points = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed).draw(count)
    candidates = []
    for index, point in enumerate(points.tolist()):
        candidates.append(
            {
                "candidate_id": f"candidate_{index:02d}",
                "initial_learning_rate": _log_scale(point[0], 4e-3, 1.2e-2),
                "escape_steps": _choice(point[1], [20, 40]),
                "regularization_weight": _log_scale(point[2], 1e-3, 2e-2),
                "degree_power": _choice(point[3], [1.5, 2.0, 3.0]),
                "gradient_clip": _choice(point[4], [5.0, 10.0, 20.0]),
                "frame_count": _choice(point[5], [12, 24]),
            }
        )
    return candidates


def _trial_directory(
    root: Path, stage: str, candidate_id: str, asteroid: int, seed: int
) -> Path:
    return root / stage / candidate_id / f"asteroid{asteroid:02d}_seed{seed}"


def _read_completed(path: Path, config: FitConfig) -> dict | None:
    manifest = path / "run.json"
    if not manifest.exists():
        return None
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("config") != json.loads(json.dumps(asdict(config))):
        raise ValueError(f"completed trial has incompatible configuration: {path}")
    return document["summary"]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "stage", "candidate_id", "asteroid", "seed", "steps",
        "best_validation_rmse", "mean_fitted_rmse", "runtime_seconds",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["candidate_id"], []).append(
            float(row["best_validation_rmse"])
        )
    ranking = [
        {
            "candidate_id": candidate_id,
            "mean_validation_rmse": float(np.mean(values)),
            "worst_validation_rmse": float(np.max(values)),
            "run_count": len(values),
        }
        for candidate_id, values in grouped.items()
    ]
    return sorted(
        ranking,
        key=lambda row: (row["mean_validation_rmse"], row["worst_validation_rmse"]),
    )


def _fit_config(
    candidate: dict[str, Any], asteroid: int, seed: int, steps: int,
    device: str, *, final: bool,
) -> FitConfig:
    values = {key: value for key, value in candidate.items() if key != "candidate_id"}
    values["escape_steps"] = min(int(values["escape_steps"]), max(steps - 1, 0))
    stabilization_steps = min(
        50 if final else 20, steps - int(values["escape_steps"])
    )
    return FitConfig(
        asteroid=asteroid,
        seed=seed,
        steps=steps,
        device=device,
        stabilization_steps=stabilization_steps,
        **values,
    )


def _run_trial(
    root: Path,
    stage: str,
    candidate: dict[str, Any],
    asteroid: int,
    seed: int,
    steps: int,
    calibration: Calibration,
    device: str,
    *,
    final: bool,
) -> dict[str, Any]:
    output = _trial_directory(root, stage, candidate["candidate_id"], asteroid, seed)
    config = _fit_config(candidate, asteroid, seed, steps, device, final=final)
    summary = _read_completed(output, config)
    if summary is None:
        checkpoint = output / "checkpoint.pt"
        summary = run_inversion(
            config,
            calibration,
            output,
            resume=checkpoint if checkpoint.exists() else None,
        )
    return {
        "stage": stage,
        "candidate_id": candidate["candidate_id"],
        "asteroid": asteroid,
        "seed": seed,
        "steps": steps,
        "best_validation_rmse": summary["best_validation_rmse"],
        "mean_fitted_rmse": summary["mean_fitted_rmse"],
        "runtime_seconds": summary["runtime_seconds"],
    }


def run_tuning(
    calibration: Calibration,
    output_dir: Path,
    *,
    asteroids: tuple[int, ...] = (1, 2, 3),
    candidate_count: int = 12,
    finalist_count: int = 4,
    seeds: tuple[int, ...] = (0, 1),
    coarse_steps: int = 100,
    final_steps: int = 300,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the default 36-run coarse and 24-run finalist study."""

    if finalist_count <= 0 or finalist_count > candidate_count:
        raise ValueError("finalist count must be in 1..candidate_count")
    if not asteroids or not seeds:
        raise ValueError("at least one asteroid and seed are required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidates(candidate_count)
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        for asteroid in asteroids:
            rows.append(
                _run_trial(
                    root, "coarse", candidate, asteroid, 0, coarse_steps,
                    calibration, device, final=False,
                )
            )
            _write_rows(root / "leaderboard.csv", rows)

    coarse_ranking = _aggregate(rows)
    promoted_ids = [row["candidate_id"] for row in coarse_ranking[:finalist_count]]
    final_rows: list[dict[str, Any]] = []
    for candidate_id in promoted_ids:
        candidate = by_id[candidate_id]
        for asteroid in asteroids:
            for seed in seeds:
                row = _run_trial(
                    root, "final", candidate, asteroid, seed, final_steps,
                    calibration, device, final=True,
                )
                rows.append(row)
                final_rows.append(row)
                _write_rows(root / "leaderboard.csv", rows)

    final_ranking = _aggregate(final_rows)
    winner_id = final_ranking[0]["candidate_id"]
    winner = by_id[winner_id]
    winner_config = _fit_config(
        winner, asteroids[0], seeds[0], final_steps, device, final=True
    )
    reusable = {name: asdict(winner_config)[name] for name in REUSABLE_FIELDS}
    best_config = {
        "schema_version": 1,
        "selection": final_ranking[0],
        "fit_config": reusable,
    }
    (root / "best_config.json").write_text(
        json.dumps(best_config, indent=2) + "\n", encoding="utf-8"
    )
    study = {
        "schema_version": 1,
        "search_space": SEARCH_SPACE,
        "candidates": candidates,
        "coarse_ranking": coarse_ranking,
        "promoted_candidates": promoted_ids,
        "final_ranking": final_ranking,
        "winner": winner_id,
        "optimization_run_count": len(rows),
    }
    (root / "study.json").write_text(
        json.dumps(study, indent=2) + "\n", encoding="utf-8"
    )
    return study
