"""Command-line interface for calibration, inversion, and animation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch

def _ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("asteroids must be comma-separated integers") from error
    if not result or any(item not in range(1, 11) for item in result):
        raise argparse.ArgumentTypeError("asteroids must be a non-empty subset of 1..10")
    return result


def _floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one number is required")
    return result


def _ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("integers must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hac-asteroids",
        description="GPU asteroid lightcurve inversion and validation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="report CUDA readiness")
    doctor.set_defaults(handler=_doctor)

    validate = commands.add_parser("validate", help="calibrate on public asteroids")
    validate.add_argument("--asteroids", type=_ids, default=(3,))
    validate.add_argument("--frames", type=int, default=72, help="screening phases; finalists always use 360")
    validate.add_argument("--distances", type=_floats, default=(2.5, 3.5, 5, 8, 16, 32))
    validate.add_argument("--mixtures", type=_floats, default=(0, 0.25, 0.5, 0.75, 1))
    validate.add_argument("--image-size", type=int, default=96)
    validate.add_argument("--responses", nargs="+", choices=("linear", "srgb"), default=("linear", "srgb"))
    validate.add_argument("--gains", type=_floats, default=(.5, 1., 2.))
    validate.add_argument("--projections", nargs="+", choices=("perspective", "orthographic"), default=("perspective", "orthographic"))
    validate.add_argument("--phase-conventions", nargs="+", choices=("frame_label", "zero_based"), default=("frame_label", "zero_based"))
    validate.add_argument("--device", default="cuda")
    validate.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration"))
    validate.set_defaults(handler=_validate)

    fit = commands.add_parser("fit", help="fit octantoid meshes to lightcurves")
    fit.add_argument("--asteroids", type=_ids, default=(3,))
    fit.add_argument("--calibration", type=Path, required=True)
    fit.add_argument("--origin", choices=("simulated", "real"), default="simulated")
    fit.add_argument("--curve-type", choices=("intensity", "binary"), default="intensity")
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    fit.add_argument("--config", type=Path, help="JSON configuration from tune")
    fit.add_argument("--steps", type=int)
    fit.add_argument("--frames", type=int)
    fit.add_argument("--validation-frames", type=int)
    fit.add_argument("--validation-interval", type=int)
    fit.add_argument("--evaluation-stride", type=int)
    fit.add_argument("--learning-rate", type=float, help="refinement learning rate")
    fit.add_argument("--initial-learning-rate", type=float)
    fit.add_argument("--min-learning-rate", type=float)
    fit.add_argument("--escape-steps", type=int)
    fit.add_argument("--stabilization-steps", type=int)
    fit.add_argument("--regularization-weight", type=float)
    fit.add_argument("--degree-power", type=float)
    fit.add_argument("--gradient-clip", type=float)
    fit.add_argument("--lmax", type=int)
    fit.add_argument("--faces", type=int, help="exact template face count; overrides subdivisions")
    fit.add_argument("--representation", choices=("radial_sh", "octantoid"))
    fit.add_argument("--initialization", choices=("sphere", "ellipsoid"))
    fit.add_argument("--regularization-denominator", type=float)
    fit.add_argument("--training-block-frames", type=int)
    fit.add_argument("--initialization-noise", type=float)
    fit.add_argument("--calibration-transfer", action="store_true", default=None)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--smoke", action="store_true")
    fit.add_argument("--resume", type=Path)
    fit.set_defaults(handler=_fit)

    tune = commands.add_parser("tune", help="run resumable staged hyperparameter tuning")
    tune.add_argument("--asteroids", type=_ids, default=(1, 2, 3))
    tune.add_argument("--calibration", type=Path, required=True)
    tune.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning"))
    tune.add_argument("--candidates", type=int, default=12)
    tune.add_argument("--finalists", type=int, default=4)
    tune.add_argument("--seeds", type=_ints, default=(0, 1))
    tune.add_argument("--coarse-steps", type=int, default=100)
    tune.add_argument("--final-steps", type=int, default=300)
    tune.add_argument("--device", default="cuda")
    tune.set_defaults(handler=_tune)

    animate = commands.add_parser("animate", help="render a saved trajectory")
    animate.add_argument("trajectory", type=Path)
    animate.add_argument("--output", type=Path, required=True)
    animate.add_argument("--fps", type=int, default=12)
    animate.add_argument(
        "--all-iterations", action="store_true",
        help="include post-best iterations instead of ending on the selected mesh",
    )
    animate.set_defaults(handler=_animate)
    return parser


def _doctor(_args: argparse.Namespace) -> int:
    result = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["cuda_available"] else 1


def _validate(args: argparse.Namespace) -> int:
    from .calibration import calibrate_public

    if any(item > 3 for item in args.asteroids):
        raise ValueError("validation requires public asteroids 1, 2, or 3")
    result = calibrate_public(
        args.asteroids,
        args.output_dir,
        frame_count=args.frames,
        distances=args.distances,
        mixtures=args.mixtures,
        device=args.device,
        image_size=args.image_size,
        responses=args.responses, gains=args.gains, projections=args.projections,
        phase_conventions=args.phase_conventions,
    )
    print(json.dumps({str(a): asdict(c) for a, c in result.items()}
                     if isinstance(result, dict) else asdict(result), indent=2))
    return 0


def _fit(args: argparse.Namespace) -> int:
    from .calibration import Calibration
    from .inversion import FitConfig, default_run_directory, run_inversion

    calibration = Calibration.load(args.calibration)
    summaries = []
    if args.resume is not None and len(args.asteroids) != 1:
        raise ValueError("--resume can only be used with one asteroid")
    for asteroid in args.asteroids:
        if args.smoke:
            if args.config is not None:
                raise ValueError("--smoke and --config cannot be combined")
            config = FitConfig.smoke(asteroid, device=args.device)
            config = FitConfig(
                **{
                    **config.__dict__,
                    "origin": args.origin,
                    "curve_type": args.curve_type,
                    "calibration_transfer": bool(args.calibration_transfer),
                    "steps": args.steps if args.steps is not None else config.steps,
                    **{name: getattr(args, name) for name in (
                        "faces", "representation", "initialization", "regularization_denominator", "lmax"
                    ) if getattr(args, name) is not None},
                }
            )
        else:
            values = asdict(FitConfig(asteroid=asteroid))
            if args.config is not None:
                document = json.loads(args.config.read_text(encoding="utf-8"))
                loaded = document.get("fit_config", document)
                allowed = {field.name for field in fields(FitConfig)} - {"asteroid"}
                unknown = set(loaded) - allowed
                if unknown:
                    raise ValueError(f"unknown fit configuration fields: {sorted(unknown)}")
                values.update(loaded)
            overrides = {
                "steps": args.steps,
                "frame_count": args.frames,
                "validation_frame_count": args.validation_frames,
                "validation_interval": args.validation_interval,
                "evaluation_stride": args.evaluation_stride,
                "learning_rate": args.learning_rate,
                "initial_learning_rate": args.initial_learning_rate,
                "min_learning_rate": args.min_learning_rate,
                "escape_steps": args.escape_steps,
                "stabilization_steps": args.stabilization_steps,
                "regularization_weight": args.regularization_weight,
                "degree_power": args.degree_power,
                "gradient_clip": args.gradient_clip,
                "lmax": args.lmax,
                "faces": args.faces,
                "representation": args.representation,
                "initialization": args.initialization,
                "regularization_denominator": args.regularization_denominator,
                "training_block_frames": args.training_block_frames,
                "initialization_noise": args.initialization_noise,
                "calibration_transfer": args.calibration_transfer,
            }
            values.update({key: value for key, value in overrides.items() if value is not None})
            values.update(
                asteroid=asteroid,
                origin=args.origin,
                curve_type=args.curve_type,
                seed=args.seed,
                device=args.device,
            )
            config = FitConfig(**values)
        run_dir = default_run_directory(args.output_dir, asteroid)
        summaries.append(run_inversion(config, calibration, run_dir, resume=args.resume))
    print(json.dumps(summaries, indent=2))
    return 0


def _tune(args: argparse.Namespace) -> int:
    from .calibration import Calibration
    from .tuning import run_tuning

    study = run_tuning(
        Calibration.load(args.calibration),
        args.output_dir,
        asteroids=args.asteroids,
        candidate_count=args.candidates,
        finalist_count=args.finalists,
        seeds=args.seeds,
        coarse_steps=args.coarse_steps,
        final_steps=args.final_steps,
        device=args.device,
    )
    print(json.dumps(study, indent=2))
    return 0


def _animate(args: argparse.Namespace) -> int:
    from .animation import animate_trajectory

    output = animate_trajectory(
        args.trajectory, args.output, fps=args.fps,
        all_iterations=args.all_iterations,
    )
    print(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
