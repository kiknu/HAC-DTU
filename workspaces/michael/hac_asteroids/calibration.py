"""Per-body public-mesh calibration with full-turn finalist verification."""
from __future__ import annotations

import csv
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from forward_validation import load_benchmark_mesh
from pytorch3d_forward import (ObservationBatch, PyTorch3DLightcurveModel,
                             RendererConfig, extract_lightcurves, image_response)
from .datasets import load_lightcurves, public_mesh_path
from .provenance import code_identity, dataset_identity, file_sha256, fingerprint


@dataclass(frozen=True)
class Calibration:
    camera_distance_radii: float = 5.0
    lambertian_fraction: float = 0.5
    asteroids: tuple[int, ...] = ()
    frame_count: int = 0
    mean_rmse: float | None = None
    schema_version: int = 2
    response: str = "linear"
    radiance_gain: float = 1.0
    projection: str = "perspective"
    phase_convention: str = "frame_label"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version not in (1, 2):
            raise ValueError("unsupported calibration schema")
        if self.phase_convention not in ("frame_label", "zero_based"):
            raise ValueError("invalid phase convention")
        self.renderer_config()

    def renderer_config(self, base: RendererConfig | None = None):
        return replace(base or RendererConfig.optimization(),
                       camera_distance_radii=self.camera_distance_radii,
                       lambertian_fraction=self.lambertian_fraction,
                       response=self.response, radiance_gain=self.radiance_gain,
                       projection=self.projection)

    def validate_scope(self, asteroid: int, transfer: bool = False):
        if asteroid not in self.asteroids and not transfer:
            raise ValueError("calibration does not cover this body; explicitly enable calibration transfer")

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if values.get("schema_version", 1) == 1:
            warnings.warn("Legacy calibration: linear/gain-1 perspective, frame-label phases", UserWarning)
            values["schema_version"] = 1
        values["asteroids"] = tuple(values.get("asteroids", ()))
        return cls(**values)


def phase_degrees(dataset, indices, convention="frame_label"):
    indices = np.asarray(indices, dtype=np.int64)
    if convention not in ("frame_label", "zero_based"):
        raise ValueError("invalid phase convention")
    if dataset.origin == "simulated" and convention == "frame_label":
        return dataset.frames[indices].astype(np.float64)
    return 360.0 * indices / len(dataset.frames)


def normalize(values):
    return values / np.maximum(values.mean(axis=0, keepdims=True), 1e-9)


def curve_metrics(prediction, target, columns):
    """Inputs already share complete-set normalization; never rescale subsets."""
    residual = prediction - target
    errors = np.sqrt(np.mean(residual ** 2, axis=0))
    rows = []
    for j, name in enumerate(columns):
        p, y = prediction[:, j], target[:, j]
        dark = y < .1
        peak_delta = (int(p.argmax()) - int(y.argmax()) + len(y) // 2) % len(y) - len(y) // 2
        rows.append(dict(camera=name, rmse=float(errors[j]),
                         correlation=float(np.corrcoef(p, y)[0, 1]) if min(p.std(), y.std()) > 1e-12 else None,
                         peak_index=int(p.argmax()), observed_peak_index=int(y.argmax()),
                         peak_delta_samples=peak_delta, peak_amplitude=float(p.max()),
                         observed_peak_amplitude=float(y.max()),
                         dark_count=int(dark.sum()),
                         dark_mean=float(p[dark].mean()) if dark.any() else None,
                         dark_rmse=float(np.sqrt(np.mean((p[dark]-y[dark])**2))) if dark.any() else None))
    return dict(pooled_rmse=float(np.sqrt(np.mean(residual**2))),
                mean_camera_rmse=float(errors.mean()), cameras=rows)


def _candidate_key(candidate):
    return fingerprint(candidate)


@torch.no_grad()
def score_candidates(asteroid, candidates, indices, output_dir, *, device="cuda", image_size=96):
    """Stream linear endpoint images once per geometry/phase; cache only small raw curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_lightcurves(asteroid)
    base = replace(RendererConfig.optimization(), image_size=image_size,
                   shadow_size=image_size, batch_size=16)
    mesh = load_benchmark_mesh(asteroid, 1280, torch.device(device), output_dir / "mesh_cache")
    identity = dict(dataset=dataset_identity(dataset), mesh_sha256=file_sha256(public_mesh_path(asteroid)),
                    code=code_identity(), renderer=asdict(base), indices=list(map(int, indices)))
    groups = {}
    results = []
    for candidate in candidates:
        key = (candidate["projection"], candidate["camera_distance_radii"], candidate["phase_convention"])
        groups.setdefault(key, []).append(candidate)
    for (projection, distance, convention), group in groups.items():
        pending = []
        for candidate in group:
            path = output_dir / (fingerprint(dict(identity=identity, candidate=candidate)) + ".npz")
            if path.exists():
                with np.load(path) as saved:
                    results.append((candidate, saved["raw"]))
            else:
                pending.append((candidate, path, []))
        if not pending:
            continue
        renderer = PyTorch3DLightcurveModel(replace(base, projection=projection)).to(device)
        for start in range(0, len(indices), 6):
            block = indices[start:start+6]
            angles = torch.as_tensor(np.radians(phase_degrees(dataset, block, convention)),
                                     dtype=torch.float32, device=device)
            observations = ObservationBatch.challenge(angles, dataset.columns)
            endpoints = [renderer(mesh, observations, lambertian_fraction=m,
                                  camera_distance_radii=distance, compute_binary=False,
                                  return_linear_images=True).images for m in (0., 1.)]
            if start == 0:
                direct = renderer(mesh, observations, lambertian_fraction=.25,
                                  camera_distance_radii=distance, compute_binary=False,
                                  return_linear_images=True).images
                mixed = .75*endpoints[0]+.25*endpoints[1]
                # Repeated CUDA normal/scatter reductions have small rounding
                # differences. The absolute tolerance is < 1/78 of an 8-bit level.
                torch.testing.assert_close(mixed, direct, atol=5e-5, rtol=5e-5)
                with (output_dir / "mixing_checks.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(projection=projection, distance=distance,
                        convention=convention, frame_count=len(indices),
                        max_absolute_error=float((mixed-direct).abs().max()),
                        rms_error=float((mixed-direct).square().mean().sqrt()))) + "\n")
            for candidate, _, blocks in pending:
                mixture = candidate["lambertian_fraction"]
                pixels = image_response((1-mixture)*endpoints[0]+mixture*endpoints[1],
                                        candidate["response"], candidate["radiance_gain"])
                raw = extract_lightcurves(pixels.flatten(0, 1), len(block), len(dataset.columns),
                                          mode="exact", compute_binary=False).raw_intensity
                blocks.append(raw.cpu().numpy())
        for candidate, path, blocks in pending:
            raw = np.concatenate(blocks)
            if not np.isfinite(raw).all() or np.any(raw.mean(axis=0) <= 0):
                raise RuntimeError("nonfinite or dark calibration camera")
            np.savez_compressed(path, raw=raw, identity=json.dumps(identity), candidate=json.dumps(candidate))
            results.append((candidate, raw))
        print(f"[calibration A{asteroid}] {projection} d={distance} {convention}, {len(indices)} frames", flush=True)
    return [dict(candidate=candidate, metrics=curve_metrics(normalize(raw),
                 normalize(dataset.curves[indices]) if len(indices) < len(dataset.frames) else dataset.curves,
                 dataset.columns), raw=raw) for candidate, raw in results]


def _save_scores(path, rows):
    serial = [{k: v for k, v in row.items() if k != "raw"} for row in rows]
    path.write_text(json.dumps(serial, indent=2) + "\n")


@torch.no_grad()
def calibrate_public(asteroids: Iterable[int], output_dir: Path, *, frame_count=72,
                     distances=(2.5, 3.5, 5., 8., 16., 32.), mixtures=(0., .25, .5, .75, 1.),
                     responses=("linear", "srgb"), gains=(.5, 1., 2.),
                     phase_conventions=("frame_label", "zero_based"),
                     projections=("perspective", "orthographic"), device="cuda", image_size=96):
    asteroid_ids = tuple(map(int, asteroids))
    if not asteroid_ids or any(a not in (1, 2, 3) for a in asteroid_ids):
        raise ValueError("public calibration requires asteroids 1..3")
    if not 1 <= frame_count <= 360:
        raise ValueError("screening frame count must be in 1..360")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [dict(projection=p, camera_distance_radii=float(d), lambertian_fraction=float(m),
                       response=r, radiance_gain=float(g), phase_convention=c)
                  for p in projections
                  for d, m, r, g, c in product(distances if p == "perspective" else (16.,),
                                               mixtures, responses, gains, phase_conventions)]
    for candidate in candidates:
        Calibration(**candidate)
    legacy = dict(projection="perspective", camera_distance_radii=16., lambertian_fraction=.5,
                  response="linear", radiance_gain=1., phase_convention="frame_label")
    indices = np.linspace(0, 360, frame_count, endpoint=False, dtype=np.int64)
    screened, selected, full_rows = {}, {}, {}
    started = time.monotonic()
    for asteroid in asteroid_ids:
        body_dir = output_dir / f"asteroid{asteroid:02d}"
        rows = score_candidates(asteroid, candidates, indices, body_dir, device=device, image_size=image_size)
        rows.sort(key=lambda r: r["metrics"]["pooled_rmse"])
        screened[asteroid] = rows
        _save_scores(body_dir / "screen.json", rows)
    # Shared candidate is selected on the same grid, then checked on full turns for both bodies.
    totals = { _candidate_key(c): 0. for c in candidates }
    for rows in screened.values():
        for row in rows:
            totals[_candidate_key(row["candidate"])] += row["metrics"]["pooled_rmse"]
    shared = min(candidates, key=lambda c: totals[_candidate_key(c)])
    for asteroid in asteroid_ids:
        body_dir = output_dir / f"asteroid{asteroid:02d}"
        finalists = [r["candidate"] for r in screened[asteroid][:3]] + [legacy, shared]
        finalists = list({_candidate_key(c): c for c in finalists}.values())
        rows = score_candidates(asteroid, finalists, np.arange(360), body_dir, device=device, image_size=image_size)
        best = min(rows, key=lambda r: r["metrics"]["pooled_rmse"])
        c = best["candidate"]
        if c["projection"] == "perspective" and c["camera_distance_radii"] in (2.5, 32.):
            extension = dict(c, camera_distance_radii=2. if c["camera_distance_radii"] == 2.5 else 64.)
            rows += score_candidates(asteroid, [extension], np.arange(360), body_dir, device=device, image_size=image_size)
            best = min(rows, key=lambda r: r["metrics"]["pooled_rmse"])
        _save_scores(body_dir / "finalists.json", rows)
        full_rows[asteroid] = rows
        provenance = dict(dataset=dataset_identity(load_lightcurves(asteroid)),
                          mesh_sha256=file_sha256(public_mesh_path(asteroid)), code=code_identity(),
                          screening_indices=indices.tolist(), verification_indices=list(range(360)),
                          mesh_preprocessing="star_proxy, centered bounds, z half-extent 1, 1280 faces",
                          renderer=asdict(replace(RendererConfig.optimization(), image_size=image_size,
                                                 shadow_size=image_size, batch_size=16)),
                          scope="conditional on public full-turn intensity calibration",
                          boundary_optimum=best["candidate"]["camera_distance_radii"] in (2., 64.),
                          metrics=best["metrics"], runtime_seconds=time.monotonic()-started)
        calibration = Calibration(**best["candidate"], asteroids=(asteroid,), frame_count=360,
                                  mean_rmse=best["metrics"]["pooled_rmse"], provenance=provenance)
        calibration.save(body_dir / "calibration.json")
        selected[asteroid] = calibration
    sharing = {str(a): next(r["metrics"]["pooled_rmse"] for r in full_rows[a]
                            if r["candidate"] == shared) / selected[a].mean_rmse - 1 for a in asteroid_ids}
    (output_dir / "sharing.json").write_text(json.dumps(dict(candidate=shared, relative_degradation=sharing,
                        eligible=all(v <= .05 for v in sharing.values()), adopted=False,
                        reason="Retain explicit per-body configurations for the controlled campaign"), indent=2))
    if len(selected) == 1:
        calibration = next(iter(selected.values()))
        calibration.save(output_dir / "calibration.json")
        return calibration
    (output_dir / "calibrations.json").write_text(json.dumps({str(a): asdict(c) for a, c in selected.items()}, indent=2))
    return selected
