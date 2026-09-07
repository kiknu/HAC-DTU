"""Command-line interface for calibration, inversion, and animation."""

from __future__ import annotations

import argparse
import json
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
    validate.add_argument("--frames", type=int, default=12)
    validate.add_argument("--distances", type=_floats, default=(3.5, 4, 5, 8, 12, 16))
    validate.add_argument("--mixtures", type=_floats, default=(0, 0.25, 0.5, 0.75, 1))
    validate.add_argument("--image-size", type=int, default=96)
    validate.add_argument("--device", default="cuda")
    validate.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration"))
    validate.set_defaults(handler=_validate)

    fit = commands.add_parser("fit", help="fit octantoid meshes to lightcurves")
    fit.add_argument("--asteroids", type=_ids, default=(3,))
    fit.add_argument("--calibration", type=Path, required=True)
    fit.add_argument("--origin", choices=("simulated", "real"), default="simulated")
    fit.add_argument("--curve-type", choices=("intensity", "binary"), default="intensity")
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    fit.add_argument("--steps", type=int, default=400)
    fit.add_argument("--frames", type=int, default=12)
    fit.add_argument("--evaluation-stride", type=int, default=1)
    fit.add_argument("--learning-rate", type=float, default=2e-3)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--smoke", action="store_true")
    fit.add_argument("--resume", type=Path)
    fit.set_defaults(handler=_fit)

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
    )
    print(json.dumps(result.__dict__, indent=2))
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
            config = FitConfig.smoke(asteroid, device=args.device)
            config = FitConfig(
                **{
                    **config.__dict__,
                    "origin": args.origin,
                    "curve_type": args.curve_type,
                    "steps": args.steps if args.steps != 400 else config.steps,
                }
            )
        else:
            config = FitConfig(
                asteroid=asteroid,
                origin=args.origin,
                curve_type=args.curve_type,
                steps=args.steps,
                learning_rate=args.learning_rate,
                frame_count=args.frames,
                evaluation_stride=args.evaluation_stride,
                seed=args.seed,
                device=args.device,
            )
        run_dir = default_run_directory(args.output_dir, asteroid)
        summaries.append(run_inversion(config, calibration, run_dir, resume=args.resume))
    print(json.dumps(summaries, indent=2))
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
