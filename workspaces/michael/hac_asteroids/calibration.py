"""Public-mesh calibration of fixed photometric nuisance parameters."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from forward_validation import load_benchmark_mesh
from pytorch3d_forward import ObservationBatch, PyTorch3DLightcurveModel, RendererConfig

from .datasets import load_lightcurves


@dataclass(frozen=True)
class Calibration:
    camera_distance_radii: float = 5.0
    lambertian_fraction: float = 0.5
    asteroids: tuple[int, ...] = ()
    frame_count: int = 0
    mean_rmse: float | None = None

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["asteroids"] = tuple(values.get("asteroids", ()))
        return cls(**values)


def _sample_indices(length: int, count: int) -> np.ndarray:
    if count <= 0 or count > length:
        raise ValueError(f"frame count must be in 1..{length}")
    return np.linspace(0, length, count, endpoint=False, dtype=np.int64)


@torch.no_grad()
def calibrate_public(
    asteroids: Iterable[int],
    output_dir: Path,
    *,
    frame_count: int = 12,
    distances: Iterable[float] = (3.5, 4.0, 5.0, 8.0, 12.0, 16.0),
    mixtures: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    device: str | torch.device = "cuda",
    image_size: int = 96,
) -> Calibration:
    asteroid_ids = tuple(int(value) for value in asteroids)
    if not asteroid_ids or any(value not in (1, 2, 3) for value in asteroid_ids):
        raise ValueError("calibration asteroids must be a non-empty subset of 1,2,3")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for public calibration but is unavailable")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "mesh_cache"
    config = replace(
        RendererConfig.optimization(), image_size=image_size, shadow_size=image_size,
        batch_size=32,
    )
    renderer = PyTorch3DLightcurveModel(config).to(device)
    prepared = []
    for asteroid in asteroid_ids:
        dataset = load_lightcurves(asteroid)
        indices = _sample_indices(len(dataset.frames), frame_count)
        degrees = dataset.frames[indices]
        angles = torch.as_tensor(np.radians(degrees), dtype=torch.float32, device=device)
        observations = ObservationBatch.challenge(angles, dataset.columns)
        target = dataset.curves[indices]
        target = target / np.clip(target.mean(axis=0, keepdims=True), 1e-9, None)
        mesh = load_benchmark_mesh(asteroid, 1_280, device, cache_dir, "star_proxy")
        prepared.append((asteroid, mesh, observations, target))

    rows: list[dict[str, float]] = []
    grid_path = output_dir / "calibration_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("camera_distance_radii", "lambertian_fraction", "asteroid", "rmse"),
        )
        writer.writeheader()
    for distance in distances:
        if distance <= 1.0:
            raise ValueError("camera distances must exceed one radius")
        for mixture in mixtures:
            if not 0.0 <= mixture <= 1.0:
                raise ValueError("Lambertian fractions must lie in [0, 1]")
            per_body = []
            for asteroid, mesh, observations, target in prepared:
                prediction = renderer(
                    mesh,
                    observations,
                    extraction="exact",
                    camera_distance_radii=float(distance),
                    lambertian_fraction=float(mixture),
                    compute_binary=False,
                ).intensity.cpu().numpy()
                rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
                per_body.append(rmse)
                rows.append({
                    "camera_distance_radii": float(distance),
                    "lambertian_fraction": float(mixture),
                    "asteroid": float(asteroid),
                    "rmse": rmse,
                })
            rows.append({
                "camera_distance_radii": float(distance),
                "lambertian_fraction": float(mixture),
                "asteroid": 0.0,
                "rmse": float(np.mean(per_body)),
            })
            with grid_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
                writer.writerows(rows[-(len(prepared) + 1):])
            print(
                f"[calibration] distance={float(distance):g} "
                f"mixture={float(mixture):g} mean_rmse={float(np.mean(per_body)):.6f}",
                flush=True,
            )
    aggregate = [row for row in rows if row["asteroid"] == 0.0]
    best = min(aggregate, key=lambda row: row["rmse"])
    calibration = Calibration(
        camera_distance_radii=best["camera_distance_radii"],
        lambertian_fraction=best["lambertian_fraction"],
        asteroids=asteroid_ids,
        frame_count=frame_count,
        mean_rmse=best["rmse"],
    )
    calibration.save(output_dir / "calibration.json")
    return calibration
