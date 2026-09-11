"""Resumable radial harmonic versus octantoid experiments.

python -m hac_asteroids.representation_study run --output-dir artifacts/representations_1024_500
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import tarfile
import traceback

import numpy as np
import torch

from pytorch3d_forward import ObservationBatch
from .calibration import Calibration, calibrate_public, phase_degrees
from .datasets import load_lightcurves
from .inversion import (FitConfig, _build_model, _evaluate, _normalized, optimization_state,
                        run_inversion, spectral_penalty, two_pass_backward)
from .provenance import code_identity, dataset_identity, file_sha256, fingerprint
from .representation_reporting import (variants_for, audit_run, report_asteroid, report_campaign,
                                       verify_campaign_artifacts, write_json)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RECIPE = ROOT / "artifacts/tuning_a2_a3_20260907/best_config.json"
CALIBRATIONS = ROOT / "artifacts/corrections_225_bottom_20260908/calibration"
A1_CALIBRATION = ROOT / "artifacts/representations_1024_500_20260909/calibration/asteroid01/calibration.json"


@dataclass(frozen=True)
class StudyProfile:
    asteroids: tuple[int, ...]
    steps: int
    validation_interval: int
    escape_steps: int
    stabilization_steps: int
    benchmark_steps: tuple[int, ...]


PROFILES = {
    "full": StudyProfile((1, 2, 3), 500, 10, 40, 50, (0, 1, 2, 3, 4, 40, 41, 42, 450, 499)),
    "quick": StudyProfile((3,), 120, 20, 12, 12, (0, 12, 108)),
    "octantoid": StudyProfile((1, 2, 3), 500, 20, 40, 40, (0, 40, 460)),
}


def register_profile(root, profile):
    spec = PROFILES[profile]
    variants = variants_for(profile)
    description = dict(profile=profile, **asdict(spec), expected_runs=len(spec.asteroids) * len(variants))
    if profile == "octantoid":
        description.update(variants=variants, budget_seconds=8*3600, report_reserve_seconds=1800,
                           fit_contingency=1.1, candidate_caps=list(range(300,501,20)))
    path = root / "study.json"
    selected_steps = spec.steps
    if path.exists():
        saved_description = json.loads(path.read_text())
        selected_steps = saved_description.pop("selected_steps", spec.steps)
        if fingerprint(saved_description) != fingerprint(description):
            raise ValueError("study profile changed; use a new output directory")
        if profile == "octantoid" and selected_steps not in range(300,501,20):
            raise ValueError("invalid frozen octantoid iteration cap")
    frozen_path = root / "frozen_configuration.json"
    if frozen_path.exists():
        saved = json.loads(frozen_path.read_text())["configs"]
        expected = {f"{a}/{key}" for a in spec.asteroids for key, *_ in variants}
        if set(saved) != expected or any(c["steps"] != selected_steps for c in saved.values()):
            raise ValueError("study profile does not match frozen configurations")
    if not path.exists():
        write_json(path, description)
    return spec


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def config_for(asteroid, representation, degree, block_frames=6, *, profile="full", steps=None):
    spec = PROFILES[profile]
    if asteroid not in spec.asteroids:
        raise ValueError(f"asteroid {asteroid} is outside the {profile} study profile")
    if profile == "octantoid" and (representation != "octantoid" or degree not in (2,4,6)):
        raise ValueError("octantoid profile requires octantoid L2, L4 or L6")
    if steps is not None and (profile != "octantoid" or steps not in range(300,501,20)):
        raise ValueError("only octantoid supports a benchmark-selected cap of 300..500 by 20")
    values = json.loads(HISTORICAL_RECIPE.read_text())["fit_config"]
    values.update(asteroid=asteroid, steps=steps or spec.steps, faces=1024, representation=representation,
                  validation_interval=spec.validation_interval, escape_steps=spec.escape_steps,
                  stabilization_steps=spec.stabilization_steps,
                  lmax=degree, initialization="sphere", initialization_noise=0., seed=0,
                  regularization_denominator=146., degree_power=3., frame_count=72,
                  validation_frame_count=72, train_indices=tuple(range(0,360,5)),
                  validation_indices=tuple(range(2,360,5)), training_block_frames=block_frames,
                  evaluation_stride=1, device="cuda", early_stopping=profile == "octantoid")
    return FitConfig(**values)


def status(root, state, **details):
    value = dict(state=state, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 pid=os.getpid(), **details)
    write_json(root / "status.json", value)
    print(json.dumps(value), flush=True)


def prepare(root, profile="full"):
    asteroids = PROFILES[profile].asteroids
    root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This campaign requires CUDA; CPU fallback is disabled")
    for asteroid in asteroids:
        data = load_lightcurves(asteroid)
        if data.curves.shape != (360,21):
            raise ValueError("the study requires all 360 phases and 21 cameras")
    for asteroid in (a for a in asteroids if a != 1 or profile == "octantoid"):
        source = A1_CALIBRATION if asteroid == 1 else CALIBRATIONS / f"asteroid{asteroid:02d}/calibration.json"
        calibration = Calibration.load(source)
        calibration.validate_scope(asteroid)
        actual = dataset_identity(load_lightcurves(asteroid))
        if calibration.provenance["dataset"]["sha256"] != actual["sha256"]:
            raise ValueError("historical calibration target data changed")
        destination = root / f"calibration/asteroid{asteroid:02d}/calibration.json"
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("copied calibration changed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    first = root / "calibration/asteroid01/calibration.json"
    if 1 in asteroids and not first.exists():
        status(root, "calibrating", asteroid=1)
        calibrate_public((1,), root / "calibration", device="cuda", image_size=96)
    calibrations = {a: Calibration.load(root / f"calibration/asteroid{a:02d}/calibration.json") for a in asteroids}
    for asteroid, calibration in calibrations.items():
        calibration.validate_scope(asteroid)
        if calibration.provenance["dataset"]["sha256"] != dataset_identity(load_lightcurves(asteroid))["sha256"]:
            raise ValueError("calibration target data changed")
    return calibrations


def benchmark_variant(config, calibration, *, schedule_steps=None):
    shape, model = _build_model(config, torch.device("cuda"), calibration)
    parameters = shape.initial_parameters().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([parameters])
    dataset = load_lightcurves(config.asteroid)
    indices = np.array(config.train_indices)
    target = torch.tensor(_normalized(dataset.curves[indices]), dtype=torch.float32, device="cuda")
    angles = torch.tensor(np.radians(phase_degrees(dataset, indices, calibration.phase_convention)),
                          dtype=torch.float32, device="cuda")
    blocks = [slice(i, min(i+config.training_block_frames, len(indices)))
              for i in range(0, len(indices), config.training_block_frames)]

    def render(block):
        return model(parameters, ObservationBatch.challenge(angles[block], dataset.columns),
                     extraction="soft", compute_binary=False).raw_intensity

    times = []
    torch.cuda.reset_peak_memory_stats()
    if schedule_steps is None:
        schedule_steps = (0, config.escape_steps, config.steps-config.stabilization_steps, config.steps-1)
    for step in schedule_steps:
        stage = optimization_state(config, step)
        optimizer.param_groups[0]["lr"] = stage.learning_rate
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = two_pass_backward(render, blocks, target)
        prior = config.regularization_weight * stage.regularization_multiplier * spectral_penalty(shape.prior_residuals(parameters), config)
        prior.backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise RuntimeError("nonfinite benchmark gradient")
        if stage.maximum_degree is not None:
            parameters.grad[shape.parameter_degrees() > stage.maximum_degree] = 0
        torch.nn.utils.clip_grad_norm_([parameters], config.gradient_clip)
        optimizer.step()
        with torch.no_grad():
            parameters.copy_(shape.project_parameters(parameters))
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        print(f"benchmark {config.representation} L{config.lmax} schedule-step={step} "
              f"seconds={times[-1]:.3f} loss={float(loss):.6f}", flush=True)
    started = time.perf_counter()
    _evaluate(model, parameters, dataset, calibration)
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter()-started
    memory = torch.cuda.max_memory_allocated()/2**20
    row = dict(representation=config.representation, degree=config.lmax, parameters=shape.n_parameters,
               block_frames=config.training_block_frames, step_seconds=times,
               mean_step_seconds=float(np.mean(times[1:])), full_turn_seconds=evaluation_seconds,
               peak_memory_mib=memory,
               schedule_steps=list(schedule_steps),
               projected_fit_seconds=config.steps*float(np.mean(times[1:]))
               + ((config.steps + config.validation_interval - 1)//config.validation_interval + 5)*evaluation_seconds)
    del optimizer, parameters, model, shape
    torch.cuda.empty_cache()
    return row


def choose_iteration_cap(rows, elapsed_seconds=0.):
    """Budget all nine fits at the cap; never rely on plateau savings."""
    projections = []
    for steps in range(300,501,20):
        fitting = sum(steps*r["mean_step_seconds"] + ((steps+19)//20+5)*r["full_turn_seconds"]
                      for r in rows)
        total = elapsed_seconds + 1.1*fitting + 1800
        projections.append(dict(steps=steps, fit_seconds=fitting, total_seconds=total))
    feasible = [row for row in projections if row["total_seconds"] <= 8*3600]
    return (feasible[-1] if feasible else None), projections


def benchmark_octantoids(root, calibrations, *, started):
    spec = PROFILES["octantoid"]
    path = root / "benchmark.json"
    identity = dict(code=code_identity(), calibrations={str(a):asdict(c) for a,c in calibrations.items()},
                    recipe=file_sha256(HISTORICAL_RECIPE), gpu=torch.cuda.get_device_name(0),
                    profile=asdict(spec), campaign_source=file_sha256(Path(__file__)))
    if path.exists():
        saved = json.loads(path.read_text())
        if fingerprint(saved["identity"]) != fingerprint(identity):
            raise ValueError("benchmark identity changed; use a new campaign directory")
    else:
        for block_frames in (6,3,1):
            rows = []
            try:
                for asteroid in spec.asteroids:
                    for key, _, representation, degree in variants_for("octantoid"):
                        status(root, "benchmarking", asteroid=asteroid, variant=key, block_frames=block_frames)
                        row = benchmark_variant(config_for(asteroid, representation, degree, block_frames, profile="octantoid"),
                                                calibrations[asteroid], schedule_steps=spec.benchmark_steps)
                        rows.append(dict(asteroid=asteroid, **row))
                        write_json(root / "benchmark_progress.json", rows)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if block_frames == 1:
                    raise
                continue
            selected, projections = choose_iteration_cap(rows, time.time()-started)
            saved = dict(identity=identity, block_frames=block_frames, variants=rows, expected_runs=9,
                         selected_steps=selected["steps"] if selected else None, projections=projections,
                         projected_campaign_seconds=selected["total_seconds"] if selected else None,
                         projection_note="All nine combinations measured; includes preparation, 10% fitting contingency and 30 minutes reporting. No plateau savings assumed.")
            write_json(path, saved)
            break
    if saved["selected_steps"] is None:
        raise RuntimeError("Even 300 iterations exceeds the measured eight-hour budget; production fits were not launched. See benchmark.json.")
    description = json.loads((root / "study.json").read_text())
    if description.get("selected_steps", saved["selected_steps"]) != saved["selected_steps"]:
        raise ValueError("frozen iteration cap differs from benchmark")
    description["selected_steps"] = saved["selected_steps"]
    write_json(root / "study.json", description)
    return saved["block_frames"]


def benchmark(root, calibrations, profile="full", *, started=None):
    if profile == "octantoid":
        return benchmark_octantoids(root, calibrations, started=started if started is not None else time.time())
    spec = PROFILES[profile]
    benchmark_asteroid = 2 if 2 in spec.asteroids else spec.asteroids[0]
    path = root / "benchmark.json"
    identity = dict(code=code_identity(), calibration=asdict(calibrations[benchmark_asteroid]),
                    recipe=file_sha256(HISTORICAL_RECIPE), gpu=torch.cuda.get_device_name(0))
    if profile != "full":
        identity["profile"] = asdict(spec)
    if path.exists():
        saved = json.loads(path.read_text())
        if fingerprint(saved["identity"]) != fingerprint(identity):
            raise ValueError("benchmark identity changed; use a new campaign directory")
        return saved["block_frames"]
    for block_frames in (6,3,1):
        rows = []
        try:
            for key, _, representation, degree in variants_for(profile):
                status(root, "benchmarking", variant=key, block_frames=block_frames)
                row = benchmark_variant(config_for(benchmark_asteroid, representation, degree, block_frames, profile=profile),
                                        calibrations[benchmark_asteroid], schedule_steps=spec.benchmark_steps)
                rows.append(row)
                write_json(root / "benchmark_progress.json", rows)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if block_frames == 1:
                raise
            continue
        projected = len(spec.asteroids)*sum(row["projected_fit_seconds"] for row in rows)
        write_json(path, dict(identity=identity, block_frames=block_frames, variants=rows,
                             projected_campaign_seconds=projected, expected_runs=len(spec.asteroids)*len(variants_for(profile)),
                             **({"projected_nine_fit_seconds":projected} if profile == "full" else {}),
                             projection_note=f"Measured on asteroid {benchmark_asteroid}; excludes calibration, figures, geometry and retries."))
        return block_frames
    raise RuntimeError("no feasible memory configuration")


def freeze(root, calibrations, block_frames, profile="full"):
    asteroids = PROFILES[profile].asteroids
    steps = json.loads((root / "study.json").read_text())["selected_steps"] if profile == "octantoid" else None
    configs = {f"{a}/{key}": config_for(a, rep, degree, block_frames, profile=profile, steps=steps)
               for a in asteroids for key, _, rep, degree in variants_for(profile)}
    shape, _ = _build_model(next(iter(configs.values())), torch.device("cpu"))
    template_path = root / "sphere_1024.npz"
    directions = shape.directions.numpy()
    faces = shape.faces.numpy()
    initial = shape.mesh(shape.initial_parameters()).vertices.numpy()
    if template_path.exists():
        with np.load(template_path) as saved:
            if not all(np.array_equal(saved[key], value) for key, value in
                       (("directions",directions), ("faces",faces), ("vertices",initial))):
                raise ValueError("frozen sphere template changed")
    else:
        np.savez_compressed(template_path, directions=directions, faces=faces, vertices=initial)
    frozen = dict(configs={key:asdict(config) for key,config in configs.items()},
                  calibrations={str(a):asdict(c) for a,c in calibrations.items()},
                  datasets={str(a):dataset_identity(load_lightcurves(a)) for a in asteroids},
                  code=code_identity(), template_sha256=file_sha256(template_path),
                  recipe_sha256=file_sha256(HISTORICAL_RECIPE))
    path = root / "frozen_configuration.json"
    if path.exists() and fingerprint(json.loads(path.read_text())) != fingerprint(frozen):
        raise ValueError("frozen campaign configuration changed")
    if not path.exists():
        write_json(path, frozen)
    archive = root / "source_snapshot.tar.gz"
    if not archive.exists():
        sources = set(code_identity()) | {
            "hac_asteroids/representation_study.py", "hac_asteroids/representation_reporting.py",
            "hac_asteroids/reporting.py", "hac_asteroids/animation.py", "hac_asteroids/trajectory.py",
            "hac_asteroids/cli.py", "visualize_fit.py", "render_safety.py", "main.py",
            "README.md", "pyproject.toml", "tests/test_representations.py",
            "tests/test_quick_study.py", "tests/test_octantoid_study.py", "docs/representation-study.md",
        }
        with tarfile.open(archive, "w:gz") as bundle:
            for name in sorted(sources):
                bundle.add(ROOT / name, arcname=name)
        write_json(root / "environment.json", dict(python=sys.version, executable=sys.executable,
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    source_sha256={name:file_sha256(ROOT/name) for name in sorted(sources)}))
    for key, config in configs.items():
        asteroid, variant = key.split("/")
        write_json(root / f"asteroid{int(asteroid):02d}" / variant / "config.json", asdict(config))
    return configs


def fits(root, calibrations, configs, profile="full"):
    failures = []
    variants = variants_for(profile)
    for asteroid in sorted({c.asteroid for c in configs.values()}):
        for key, label, _, _ in variants:
            config = configs[f"{asteroid}/{key}"]
            run_dir = root / f"asteroid{asteroid:02d}" / key
            if (run_dir / "run.json").exists():
                audit_run(run_dir, config)
                continue
            checkpoint = run_dir / "checkpoint.pt"
            status(root, "fitting", asteroid=asteroid, variant=key,
                   checkpoint=str(checkpoint), resuming=checkpoint.exists())
            attempts_path = run_dir / "attempts.json"
            attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
            attempt = dict(started_unix=time.time(), status="running",
                           resume_step=int(torch.load(checkpoint, map_location="cpu", weights_only=False)["step"])
                           if checkpoint.exists() else 0)
            attempts.append(attempt)
            write_json(attempts_path, attempts)
            try:
                with (run_dir / "fit.log").open("a", buffering=1) as log:
                    with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                        run_inversion(config, calibrations[asteroid], run_dir,
                                      resume=checkpoint if checkpoint.exists() else None)
                audit_run(run_dir, config)
                attempt["status"] = "complete"
            except Exception:
                attempt["status"] = "failed"
                message = traceback.format_exc()
                (run_dir / "failure.txt").write_text(message)
                print(message, file=sys.stderr, flush=True)
                failures.append(f"asteroid{asteroid:02d}/{key}")
            finally:
                attempt["ended_unix"] = time.time()
                attempt["elapsed_seconds"] = attempt["ended_unix"] - attempt["started_unix"]
                attempt["peak_memory_mib"] = torch.cuda.max_memory_allocated()/2**20
                if attempt["status"] == "running":
                    attempt["status"] = "interrupted"
                write_json(attempts_path, attempts)
                torch.cuda.empty_cache()
        status(root, "reporting", asteroid=asteroid)
        try:
            report_asteroid(root, asteroid, [configs[f"{asteroid}/{key}"] for key,*_ in variants], calibrations)
        except Exception:
            message = traceback.format_exc()
            (root / f"asteroid{asteroid:02d}" / "report_failure.txt").write_text(message)
            print(message, file=sys.stderr, flush=True)
            failures.append(f"asteroid{asteroid:02d}/report")
        report_campaign(root)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "benchmark", "run", "report"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full",
                        help="full: nine 500-step fits; quick: three 120-step fits; octantoid: nine L2/L4/L6 fits with benchmark budget and plateau stopping")
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Prevent two accidental launches from writing to the same checkpoint.
    lock = (root / ".campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    started = time.time()
    try:
        spec = register_profile(root, args.profile)
        variants = variants_for(args.profile)
        status(root, "preparing", action=args.action, profile=args.profile)
        report_campaign(root)
        calibrations = prepare(root, args.profile)
        if args.action == "prepare":
            status(root, "prepared")
            return 0
        block_frames = benchmark(root, calibrations, args.profile, started=started)
        if args.action == "benchmark":
            status(root, "benchmarked", block_frames=block_frames)
            return 0
        configs = freeze(root, calibrations, block_frames, args.profile)
        report_campaign(root)
        failures = []
        if args.action == "run":
            failures = fits(root, calibrations, configs, args.profile)
        else:
            for asteroid in spec.asteroids:
                report_asteroid(root, asteroid, [configs[f"{asteroid}/{key}"] for key,*_ in variants], calibrations)
        rows = report_campaign(root)
        complete = len(rows) == len(spec.asteroids)*len(variants) and all(
            (root/f"asteroid{a:02d}/report_complete.json").exists() for a in spec.asteroids)
        if complete and not failures:
            status(root, "verifying_artifacts")
            verify_campaign_artifacts(root)
        status(root, "complete" if complete and not failures else "incomplete", measured_runs=len(rows),
               failures=failures, invocation_seconds=time.time()-started)
        return 0 if complete and not failures else 1
    except KeyboardInterrupt:
        status(root, "interrupted", message="Resume with the same command; saved checkpoints are retained.")
        return 130
    except Exception:
        status(root, "failed", error=traceback.format_exc())
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
