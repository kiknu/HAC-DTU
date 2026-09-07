"""Strict discovery and loading for the ten challenge asteroid datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from pytorch3d_forward import CHALLENGE_COLUMNS, UNIQUE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "challenge_data"


@dataclass(frozen=True)
class AsteroidDataset:
    asteroid: int
    origin: Literal["real", "simulated"]
    curve_type: Literal["intensity", "binary"]
    frames: np.ndarray
    curves: np.ndarray
    columns: tuple[str, ...]
    source: Path

    def select(self, frame_indices: Sequence[int]) -> "AsteroidDataset":
        indices = np.asarray(frame_indices, dtype=np.int64)
        return AsteroidDataset(
            self.asteroid,
            self.origin,
            self.curve_type,
            self.frames[indices],
            self.curves[indices],
            self.columns,
            self.source,
        )


def model_directory(asteroid: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    if asteroid not in range(1, 11):
        raise ValueError("asteroid must be in the range 1..10")
    visibility = "public" if asteroid <= 3 else "secret"
    path = Path(data_root) / f"AsteroidModel{asteroid:02d}_shape_{visibility}"
    if not path.is_dir():
        raise FileNotFoundError(f"asteroid dataset directory not found: {path}")
    return path


def _lightcurve_path(
    asteroid: int,
    origin: str,
    curve_type: str,
    data_root: Path,
) -> Path:
    directory = model_directory(asteroid, data_root) / f"Asteroid{asteroid}_lightcurve_data"
    suffix = "_blender" if origin == "simulated" else ""
    pattern = f"Asteroid*_lightcurve_{curve_type}{suffix}.txt"
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def load_lightcurves(
    asteroid: int,
    *,
    origin: Literal["real", "simulated"] = "simulated",
    curve_type: Literal["intensity", "binary"] = "intensity",
    unique_cameras: bool | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> AsteroidDataset:
    if origin not in ("real", "simulated"):
        raise ValueError("origin must be 'real' or 'simulated'")
    if curve_type not in ("intensity", "binary"):
        raise ValueError("curve_type must be 'intensity' or 'binary'")
    path = _lightcurve_path(asteroid, origin, curve_type, Path(data_root))
    raw = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 29:
        raise ValueError(f"{path} must contain one frame column and 28 camera columns")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path} contains non-finite values")
    frames, curves = raw[:, 0], raw[:, 1:]
    columns = tuple(CHALLENGE_COLUMNS)
    pairs_equal = all(
        np.array_equal(
            curves[:, columns.index(f"{azimuth}_hor1")],
            curves[:, columns.index(f"{azimuth}_hor2")],
        )
        for azimuth in (0, 45, 90, 135, 225, 270, 315)
    )
    if unique_cameras is None:
        unique_cameras = pairs_equal
    if unique_cameras:
        for azimuth in (0, 45, 90, 135, 225, 270, 315):
            left = columns.index(f"{azimuth}_hor1")
            right = columns.index(f"{azimuth}_hor2")
            if not pairs_equal:
                raise ValueError(
                    f"cannot deduplicate {azimuth}_hor2: it differs from hor1 in {path}"
                )
        keep = [columns.index(name) for name in UNIQUE_COLUMNS]
        curves = curves[:, keep]
        columns = tuple(UNIQUE_COLUMNS)
    return AsteroidDataset(
        asteroid, origin, curve_type, frames, curves, columns, path
    )


def public_mesh_path(asteroid: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    if asteroid not in (1, 2, 3):
        raise ValueError(f"asteroid {asteroid} has no public mesh")
    path = model_directory(asteroid, data_root) / f"asteroid{asteroid}.stl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
