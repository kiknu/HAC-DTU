"""Local, resumable 200-update octantoid L4 survey of all ten asteroids.

python -m hac_asteroids.l4_survey prepare --output-dir artifacts/<survey>
python -m hac_asteroids.campaign_runner --output-dir artifacts/<survey>
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import signal
import sys
import tarfile
import time
import traceback

import numpy as np
import torch

from .calibration import Calibration
from .datasets import load_lightcurves
from .inversion import _build_model
from .provenance import code_identity, dataset_identity, file_sha256, fingerprint
from .representation_reporting import report_asteroid, report_campaign, verify_campaign_artifacts, write_json
from .representation_study import (A1_CALIBRATION, CALIBRATIONS, HISTORICAL_RECIPE,
                                   ROOT, config_for as study_config, fits, status)

ASTEROIDS = tuple(range(1, 11))
PROFILE = "l4_survey"
VARIANT = "octantoid_l4"


def config_for(asteroid):
    if asteroid not in ASTEROIDS:
        raise ValueError("survey asteroid must be in 1..10")
    return replace(study_config(3, "octantoid", 4, 6, profile="octantoid"),
                   asteroid=asteroid, steps=200, escape_steps=40, stabilization_steps=40,
                   early_stopping=False, checkpoint_interval=25, calibration_transfer=asteroid > 3)


def _freeze_json(path, value):
    if path.exists():
        if fingerprint(json.loads(path.read_text())) != fingerprint(value):
            raise ValueError(f"frozen survey input changed: {path.name}; use a new output directory")
    else:
        write_json(path, value)


def prepare(root):
    """Freeze the scientific inputs on CPU; do not benchmark or change the cap."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _freeze_json(root / "study.json", dict(
        profile=PROFILE, asteroids=ASTEROIDS, expected_runs=10, steps=200,
        representation="octantoid", lmax=4, faces=1024, escape_steps=40, stabilization_steps=40,
        calibration_note="Asteroids 4–10 provisionally reuse asteroid 3 calibration; reference shapes are hidden.",
        estimated_runtime_hours=[4.6, 5.1]))
    calibrations, datasets, configs = {}, {}, {}
    for asteroid in ASTEROIDS:
        data = load_lightcurves(asteroid)
        if data.curves.shape != (360, 21) or not np.isfinite(data.curves).all():
            raise ValueError(f"invalid full-turn lightcurves for asteroid {asteroid}")
        datasets[str(asteroid)] = dataset_identity(data)
        source_asteroid = asteroid if asteroid <= 3 else 3
        source = A1_CALIBRATION if source_asteroid == 1 else (
            CALIBRATIONS / f"asteroid{source_asteroid:02d}/calibration.json")
        calibration = Calibration.load(source)
        calibration.validate_scope(asteroid, transfer=asteroid > 3)
        if calibration.provenance["dataset"]["sha256"] != dataset_identity(load_lightcurves(source_asteroid))["sha256"]:
            raise ValueError("source calibration dataset changed")
        destination = root / f"calibration/asteroid{asteroid:02d}/calibration.json"
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError("frozen calibration changed")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        calibrations[asteroid] = calibration
        config = config_for(asteroid)
        configs[f"{asteroid}/{VARIANT}"] = config

    shape, _ = _build_model(config_for(1), torch.device("cpu"))
    template = dict(directions=shape.directions.numpy(), faces=shape.faces.numpy(),
                    vertices=shape.mesh(shape.initial_parameters()).vertices.numpy())
    template_path = root / "sphere_1024.npz"
    if template_path.exists():
        with np.load(template_path) as saved:
            if not all(np.array_equal(saved[name], value) for name, value in template.items()):
                raise ValueError("frozen sphere changed")
    else:
        np.savez_compressed(template_path, **template)
    frozen = dict(configs={key: asdict(value) for key, value in configs.items()},
                  calibrations={str(a): asdict(c) for a, c in calibrations.items()},
                  datasets=datasets, code=code_identity(),
                  survey_source_sha256=file_sha256(Path(__file__)),
                  template_sha256=file_sha256(template_path), recipe_sha256=file_sha256(HISTORICAL_RECIPE))
    _freeze_json(root / "frozen_configuration.json", frozen)
    for key, config in configs.items():
        asteroid, variant = key.split("/")
        _freeze_json(root / f"asteroid{int(asteroid):02d}" / variant / "config.json", asdict(config))

    # Archive an explicit source allowlist; never include credentials or workspace contents.
    snapshot = root / "source_snapshot.tar.gz"
    if not snapshot.exists():
        sources = sorted(set(code_identity()) | {
            "hac_asteroids/l4_survey.py", "hac_asteroids/l4_survey_reporting.py",
            "hac_asteroids/representation_study.py", "hac_asteroids/representation_reporting.py",
            "hac_asteroids/campaign_runner.py", "hac_asteroids/reporting.py",
            "hac_asteroids/animation.py", "hac_asteroids/trajectory.py", "hac_asteroids/reference.py",
            "visualize_fit.py", "render_safety.py", "main.py", "pyproject.toml",
        })
        with tarfile.open(snapshot, "w:gz") as bundle:
            for name in sources:
                bundle.add(ROOT / name, arcname=name)
        write_json(root / "environment.json", dict(python=sys.version, executable=sys.executable,
                   torch=torch.__version__, cuda=torch.version.cuda,
                   source_sha256={name: file_sha256(ROOT / name) for name in sources}))
    return calibrations, configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with (root / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        try:
            calibrations, configs = prepare(root)
            if args.action == "prepare":
                # Re-preparation must not turn a completed campaign back into a queued one.
                if not (root / "status.json").exists():
                    status(root, "prepared")
                report_campaign(root)
                return 0
            if not torch.cuda.is_available():
                raise RuntimeError("This survey requires CUDA; CPU fallback is disabled")
            write_json(root / "gpu.json", dict(name=torch.cuda.get_device_name(0),
                       total_memory_mib=torch.cuda.get_device_properties(0).total_memory / 2**20))
            report_campaign(root)
            failures = []
            if args.action == "run":
                failures = fits(root, calibrations, configs, PROFILE)
            else:
                for asteroid in ASTEROIDS:
                    report_asteroid(root, asteroid, [configs[f"{asteroid}/{VARIANT}"]], calibrations)
            rows = report_campaign(root)
            complete = len(rows) == len(ASTEROIDS) and not failures
            if complete:
                status(root, "verifying_artifacts")
                verify_campaign_artifacts(root)
            status(root, "complete" if complete else "incomplete", measured_runs=len(rows),
                   failures=failures, invocation_seconds=time.time() - started)
            return 0 if complete else 1
        except KeyboardInterrupt:
            status(root, "interrupted", message="Resume the same survey; saved checkpoints are retained.")
            return 130
        except Exception:
            status(root, "failed", error=traceback.format_exc())
            raise


if __name__ == "__main__":
    raise SystemExit(main())
