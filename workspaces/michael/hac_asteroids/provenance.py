"""Content identities for datasets, caches and resumable objectives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_identity() -> dict:
    names = ("pytorch3d_forward.py", "shape_model.py", "forward_validation.py", "forward_model.py",
             "hac_asteroids/inversion.py", "hac_asteroids/calibration.py",
             "hac_asteroids/datasets.py", "hac_asteroids/provenance.py")
    return {name: file_sha256(ROOT / name) for name in names}


def dataset_identity(dataset) -> dict:
    return dict(source=str(dataset.source), sha256=file_sha256(dataset.source),
                asteroid=dataset.asteroid, origin=dataset.origin,
                observable=dataset.curve_type, columns=list(dataset.columns),
                frame_labels=dataset.frames.tolist())
