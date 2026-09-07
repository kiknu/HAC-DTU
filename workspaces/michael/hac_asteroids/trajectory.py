"""Portable, pickle-free optimization trajectory storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FitTrajectory:
    parameters: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    losses: np.ndarray
    best_index: int | None = None

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            parameters=np.asarray(self.parameters, dtype=np.float32),
            vertices=np.asarray(self.vertices, dtype=np.float32),
            faces=np.asarray(self.faces, dtype=np.int64),
            losses=np.asarray(self.losses, dtype=np.float64),
            best_index=np.asarray(
                -1 if self.best_index is None else self.best_index, dtype=np.int64
            ),
        )
        return path


def load_trajectory(path: Path) -> FitTrajectory:
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"parameters", "vertices", "faces", "losses"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"trajectory is missing arrays: {sorted(missing)}")
        values = [data[name].copy() for name in (
            "parameters", "vertices", "faces", "losses"
        )]
        best_index = int(data["best_index"]) if "best_index" in data.files else -1
        trajectory = FitTrajectory(*values, best_index=None if best_index < 0 else best_index)
    if trajectory.parameters.shape[0] != trajectory.vertices.shape[0]:
        raise ValueError("parameter and vertex history lengths differ")
    if trajectory.losses.shape != (trajectory.parameters.shape[0],):
        raise ValueError("loss history length differs from trajectory length")
    return trajectory
