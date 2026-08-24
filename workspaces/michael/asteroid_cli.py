"""Command-line interface for portable octantoid lightcurve inversion."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from pytorch3d_forward import OctantoidLightcurveModel, PyTorch3DLightcurveModel
from scalar_fit import (
    FitConfig,
    RendererHyperparameters,
    calibrate_public_bodies,
    create_octantoid_shape,
    device_capabilities,
    fit_octantoid,
    load_lightcurve_data,
    new_octantoid_state,
    OctantoidState,
    regenerate_report,
    renderer_config_for_device,
    renderer_smoke_test,
    resolve_device,
    shape_from_state,
    write_fit_artifacts,
    write_public_shape_report,
    write_resolution_evaluation,
)
from shape_model import EllipsoidInitialization


DEFAULT_ARTIFACTS = Path("artifacts")


def _csv(values: str, cast=float) -> tuple:
    return tuple(cast(value.strip()) for value in values.split(",") if value.strip())


def _require_renderer() -> None:
    capabilities = device_capabilities()
    if not capabilities.pytorch3d_available:
        raise RuntimeError(
            "PyTorch3D is not installed for this interpreter. Run "
            "`asteroid-invert doctor`, then install the platform-specific "
            f"renderer environment. Import failure: {capabilities.pytorch3d_error}"
        )


def command_doctor(args: argparse.Namespace) -> int:
    capabilities = device_capabilities()
    report = {"capabilities": asdict(capabilities), "device_policy": {}}
    try:
        selected = resolve_device(
            args.device, allow_experimental_mps=args.experimental_mps
        )
        report["device_policy"] = {
            "requested": args.device,
            "selected": str(selected),
        }
        if capabilities.pytorch3d_available:
            report["renderer_smoke"] = renderer_smoke_test(
                selected, backward=not args.forward_only
            )
        else:
            report["renderer_smoke"] = {
                "ok": False,
                "skipped": True,
                "reason": capabilities.pytorch3d_error,
            }
        if args.reference:
            if capabilities.mitsuba_available:
                from mitsuba_reference import mitsuba_smoke_test

                report["mitsuba_reference_smoke"] = mitsuba_smoke_test(
                    DEFAULT_ARTIFACTS / "doctor"
                )
            else:
                report["mitsuba_reference_smoke"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "Mitsuba is not installed",
                }
    except Exception as exc:
        report["device_policy"] = {
            "requested": args.device,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, indent=2))
    smoke = report.get("renderer_smoke", {})
    reference = report.get("mitsuba_reference_smoke")
    ok = bool(smoke.get("ok", False)) and (
        reference is None or bool(reference.get("ok", False))
    )
    return 0 if ok else 2


def _fit_output_dir(args: argparse.Namespace, loss: str) -> Path:
    if args.output_dir is not None:
        base = args.output_dir
    else:
        base = DEFAULT_ARTIFACTS / "fits" / f"asteroid{args.asteroid:02d}"
    return base / loss if getattr(args, "compare_losses", False) else base


def execute_fit(
    args: argparse.Namespace,
    *,
    loss_override: str | None = None,
) -> dict:
    _require_renderer()
    device = resolve_device(
        args.device, allow_experimental_mps=args.experimental_mps
    )
    smoke = renderer_smoke_test(device, backward=True)
    if not smoke["ok"]:
        raise RuntimeError(
            "the selected renderer failed its gradient smoke test: "
            f"{smoke.get('error', 'unknown failure')}"
        )
    calibration_path = args.calibration
    default_calibration = DEFAULT_ARTIFACTS / "calibration" / "calibration.json"
    if calibration_path is None and default_calibration.is_file():
        calibration_path = default_calibration
    hyperparameters = RendererHyperparameters.load(calibration_path)
    origin = args.origin or ("simulated" if args.asteroid <= 3 else "real")
    frame_stride = args.frame_stride
    if frame_stride is None:
        frame_stride = 6 if device.type == "cuda" else 12

    if args.initial_state is None:
        shape = create_octantoid_shape(
            args.subdivisions,
            device,
            lmax_a=args.lmax,
            check_self_intersections=True,
        )
        state = new_octantoid_state(
            shape,
            args.subdivisions,
            EllipsoidInitialization(
                a=args.axis_a,
                b=args.axis_b,
                c=args.axis_c,
                theta=math.radians(args.initial_rotation_degrees),
            ),
        )
    else:
        state = OctantoidState.load(args.initial_state)
        shape = shape_from_state(state, device, check_self_intersections=True)
        if calibration_path is None and state.renderer_hyperparameters:
            hyperparameters = RendererHyperparameters(
                **state.renderer_hyperparameters
            )
            hyperparameters.validate()

    renderer_config = renderer_config_for_device(
        device, hyperparameters, image_size=args.image_size
    )
    renderer = PyTorch3DLightcurveModel(renderer_config).to(device)
    model = OctantoidLightcurveModel(shape, renderer).to(device)
    data = load_lightcurve_data(
        args.asteroid,
        origin,
        device,
        frame_stride=frame_stride,
    )
    loss = loss_override or args.loss
    fit_config = FitConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        phase_learning_rate=args.phase_learning_rate,
        loss=loss,
        huber_delta=args.huber_delta,
        prior_weight=args.prior_weight,
        gradient_clip=args.gradient_clip,
        camera_group_size=(
            args.camera_group_size
            if args.camera_group_size is not None
            else (7 if device.type == "cuda" else 3)
        ),
        evaluation_interval=args.evaluation_interval,
        self_intersection_interval=args.self_intersection_interval,
        initialize_phase=not args.no_phase_search,
        phase_grid_step_degrees=args.phase_grid_step_degrees,
        seed=args.seed,
    )
    result = fit_octantoid(
        model, data, state, fit_config, hyperparameters
    )
    output_dir = _fit_output_dir(args, loss)
    summary = write_fit_artifacts(
        output_dir,
        result,
        shape,
        fit_config,
        hyperparameters,
        asteroid=args.asteroid,
        origin=origin,
        device=device,
    )
    if args.public_shape_report:
        summary["shape_metrics"] = write_public_shape_report(
            output_dir,
            args.asteroid,
            shape,
            state,
            result.state,
            device,
            reference_faces=args.reference_faces,
        )
    if args.evaluation_subdivisions is not None:
        summary["resolution_evaluation"] = write_resolution_evaluation(
            output_dir,
            result.state,
            data,
            hyperparameters,
            device,
            subdivisions=args.evaluation_subdivisions,
            image_size=(
                args.evaluation_image_size
                if args.evaluation_image_size is not None
                else (256 if device.type == "cuda" else 128)
            ),
        )
    summary["output_dir"] = str(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def command_fit(args: argparse.Namespace) -> int:
    summary = execute_fit(args)
    print(json.dumps(summary, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    summaries = []
    losses = ("huber", "mse") if args.compare_losses else (args.loss,)
    benchmark_root = args.output_dir or (DEFAULT_ARTIFACTS / "benchmark")
    for asteroid in _csv(args.asteroids, int):
        if asteroid not in (1, 2, 3):
            raise ValueError("benchmark accepts only public asteroids 1, 2, and 3")
        per_body = argparse.Namespace(**vars(args))
        per_body.asteroid = asteroid
        per_body.origin = "simulated"
        per_body.output_dir = benchmark_root / f"asteroid{asteroid:02d}"
        for loss in losses:
            summary = execute_fit(per_body, loss_override=loss)
            guardrail = {1: 0.03, 2: 0.20, 3: 0.06}[asteroid]
            summary["guardrail_rmse"] = guardrail
            summary["guardrail_passed"] = (
                summary["fitted_mean_rmse"] <= guardrail
            )
            summary["improved"] = (
                summary["fitted_mean_rmse"]
                < summary["initial_mean_rmse"]
            )
            summaries.append(summary)
    output = benchmark_root
    output.mkdir(parents=True, exist_ok=True)
    payload = {"runs": summaries}
    (output / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if all(item["improved"] for item in summaries) else 1


def command_calibrate(args: argparse.Namespace) -> int:
    _require_renderer()
    device = resolve_device(
        args.device, allow_experimental_mps=args.experimental_mps
    )
    result = calibrate_public_bodies(
        args.output_dir,
        device,
        asteroids=_csv(args.asteroids, int),
        frame_stride=args.frame_stride,
        target_faces=args.target_faces,
        image_size=args.image_size,
        lambertian_values=_csv(args.lambertian_values),
        projections=_csv(args.projections, str),
        distance_values=_csv(args.distance_values),
        softness_values=_csv(args.softness_values),
    )
    print(json.dumps(result, indent=2))
    return 0


def command_report(args: argparse.Namespace) -> int:
    regenerate_report(args.run_dir)
    print(f"Regenerated plots in {args.run_dir}")
    return 0


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--experimental-mps",
        action="store_true",
        help="Allow MPS only after the doctor command passes",
    )


def add_fit_arguments(
    parser: argparse.ArgumentParser, *, include_asteroid: bool = True
) -> None:
    if include_asteroid:
        parser.add_argument("--asteroid", type=int, required=True)
    parser.add_argument("--origin", choices=("real", "simulated"))
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--lmax", type=int, default=6)
    parser.add_argument("--axis-a", type=float, default=1.05)
    parser.add_argument("--axis-b", type=float, default=1.0)
    parser.add_argument("--axis-c", type=float, default=0.95)
    parser.add_argument("--initial-rotation-degrees", type=float, default=0.0)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--phase-learning-rate", type=float, default=1e-2)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--prior-weight", type=float, default=5e-2)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--camera-group-size", type=int)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument(
        "--evaluation-subdivisions",
        type=int,
        help="Optionally evaluate the fitted coefficients on a denser grid (4 gives 2,562 vertices)",
    )
    parser.add_argument("--evaluation-image-size", type=int)
    parser.add_argument(
        "--public-shape-report",
        action="store_true",
        help="Compare a public-body fit with its known radial reference mesh",
    )
    parser.add_argument("--reference-faces", type=int, default=1_280)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--self-intersection-interval", type=int, default=25)
    parser.add_argument(
        "--no-phase-search",
        action="store_true",
        help="Skip the coarse periodic phase initialization",
    )
    parser.add_argument("--phase-grid-step-degrees", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect renderer capabilities")
    add_device_arguments(doctor)
    doctor.add_argument("--forward-only", action="store_true")
    doctor.add_argument(
        "--reference", action="store_true", help="Also render a Mitsuba scene"
    )
    doctor.set_defaults(handler=command_doctor)

    fit = subparsers.add_parser("fit", help="Fit one asteroid with Adam")
    add_fit_arguments(fit)
    fit.set_defaults(handler=command_fit, compare_losses=False)

    benchmark = subparsers.add_parser(
        "benchmark", help="Fit and score public asteroids 1-3"
    )
    add_fit_arguments(benchmark, include_asteroid=False)
    benchmark.add_argument("--asteroids", default="1,2,3")
    benchmark.add_argument("--compare-losses", action="store_true")
    benchmark.set_defaults(
        handler=command_benchmark,
        asteroid=1,
        origin="simulated",
        public_shape_report=True,
    )

    calibrate = subparsers.add_parser(
        "calibrate", help="Tune shared renderer settings on public bodies"
    )
    calibrate.add_argument("--asteroids", default="1,2,3")
    calibrate.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ARTIFACTS / "calibration"
    )
    calibrate.add_argument("--frame-stride", type=int, default=12)
    calibrate.add_argument("--target-faces", type=int, default=1_280)
    calibrate.add_argument("--image-size", type=int, default=64)
    calibrate.add_argument("--lambertian-values", default="0.25,0.5,0.75")
    calibrate.add_argument("--projections", default="orthographic,perspective")
    calibrate.add_argument("--distance-values", default="3.5,5,8,16")
    calibrate.add_argument("--softness-values", default="0.001,0.002,0.005")
    add_device_arguments(calibrate)
    calibrate.set_defaults(handler=command_calibrate)

    report = subparsers.add_parser("report", help="Regenerate plots for a fit")
    report.add_argument("--run-dir", type=Path, required=True)
    report.set_defaults(handler=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
