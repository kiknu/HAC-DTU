"""Visualize how well a forward model matches the challenge lightcurves.

For each public asteroid, loads the observed challenge lightcurves, renders
the forward-model prediction on the 1,280-face star proxy, phase-aligns the
prediction to the data, and produces:

- a per-camera observed-vs-predicted panel figure (``*_curves.png``),
- a frame x camera residual heatmap (``*_residuals.png``),
- per-camera metrics as CSV (RMSE, MAE, Pearson correlation, amplitude ratio),
- a JSON summary with the estimated phase and aggregate metrics.

The CPU backend uses ``PhotometricForwardModel`` (the established forward
model); the GPU backend uses the differentiable PyTorch3D renderer.  For the
simulated (Blender) lightcurves the frames correspond exactly to rotation
degrees; for real-video lightcurves the frames are mapped linearly to one
rotation, which is an assumption documented here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from forward_model import (
    ObservationGeometry,
    PhotometricForwardModel,
    build_view_dirs,
    load_data,
)
from forward_validation import (
    UNIQUE_COLUMNS,
    curve_metrics,
    estimate_phase,
    load_benchmark_mesh,
    normalized_curves,
    periodic_shift,
)

DEFAULT_TAU_INTENSITY = 25.5 / 255.0
DEFAULT_OUTPUT_DIR = Path("artifacts/forward_match")


# --------------------------------------------------------------------------- #
# Data + prediction
# --------------------------------------------------------------------------- #

def load_observed(
    model_num: int,
    origin: str,
    curve_type: str,
    columns: list[str],
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed curves (F, C) and their rotation angles in degrees (F,)."""
    data = load_data(model_num, origin, curve_type).data
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"missing expected camera columns: {sorted(missing)}")
    observed = data.loc[:, columns].iloc[::frame_stride].to_numpy()
    n_frames = observed.shape[0]
    if origin == "simulated":
        # Blender lightcurves have one row per rotation degree.
        if n_frames > 360:
            raise ValueError(
                "simulated lightcurves are expected to span one rotation; "
                f"got {n_frames} rows after striding"
            )
        degrees = np.arange(0, 360, frame_stride, dtype=np.float64)[:n_frames]
    else:
        degrees = np.linspace(0.0, 360.0, n_frames, endpoint=False)
    return observed, degrees


def predict_cpu(
    mesh,
    columns: list[str],
    degrees: np.ndarray,
    device: torch.device,
    *,
    gamma: float,
    lambertian_fraction: float,
    tau_intensity: float,
    beta_intensity: float,
    tau_binary: float,
    beta_binary: float,
    curve_type: str,
) -> np.ndarray:
    angles = torch.as_tensor(
        np.radians(degrees), dtype=torch.float32, device=device
    )
    view_dirs = build_view_dirs(columns, device)
    sun_dir = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    geometry = ObservationGeometry.build(angles, sun_dir, view_dirs)
    renderer = PhotometricForwardModel(
        geometry,
        brdf_type="combined",
        lambertian_fraction=lambertian_fraction,
        shading="smooth",
        projection="orthographic",
        camera_distance_radii=5.0,
        tau_intensity=tau_intensity,
        beta_intensity=beta_intensity,
        tau_binary=tau_binary,
        beta_binary=beta_binary,
        frame_chunk=max(1, min(60, len(degrees))),
        direction_chunk=8,
    )
    prediction = renderer.predict(mesh, gamma=gamma, include_binary=True)
    values = prediction.binary if curve_type == "binary" else prediction.intensity
    return normalized_curves(values.detach().cpu().numpy())


def predict_gpu(
    mesh,
    columns: list[str],
    degrees: np.ndarray,
    device: torch.device,
    *,
    curve_type: str,
) -> np.ndarray:
    from pytorch3d_forward import (
        ObservationBatch,
        PyTorch3DLightcurveModel,
        RendererConfig,
    )

    model = PyTorch3DLightcurveModel(RendererConfig.evaluation()).to(device)
    obs = ObservationBatch.challenge(
        torch.as_tensor(np.radians(degrees), dtype=torch.float32, device=device),
        columns,
    )
    with torch.no_grad():
        prediction = model(mesh, obs)
    values = prediction.binary if curve_type == "binary" else prediction.intensity
    return values.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Metrics + figures
# --------------------------------------------------------------------------- #

def metrics_table(
    predicted: np.ndarray, observed: np.ndarray, columns: list[str]
) -> pd.DataFrame:
    rows = []
    for i, column in enumerate(columns):
        metrics = curve_metrics(predicted[:, i], observed[:, i])
        rows.append({"camera": column, **metrics})
    return pd.DataFrame(rows, columns=["camera", "rmse", "mae", "correlation",
                                       "amplitude_ratio"])


def plot_curve_match(
    predicted: np.ndarray,
    observed: np.ndarray,
    columns: list[str],
    degrees: np.ndarray,
    out_png: Path,
    *,
    title: str,
    phase_deg: float = 0.0,
    obs_label: str = "observed",
    pred_label: str = "model",
) -> None:
    """Per-camera observed-vs-predicted panels with per-camera metrics."""
    n_cameras = len(columns)
    n_cols = 3
    n_rows = int(np.ceil(n_cameras / n_cols))
    shifted = periodic_shift(predicted, degrees, phase_deg)
    metrics = metrics_table(shifted, observed, columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.3 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for i, column in enumerate(columns):
        ax = axes[i]
        ax.axhline(1.0, color="gray", lw=0.5, alpha=0.6)
        ax.plot(degrees, observed[:, i], "k.", ms=2.5, alpha=0.75, label=obs_label)
        ax.plot(degrees, shifted[:, i], "r-", lw=1.1, label=pred_label)
        if abs(phase_deg) > 1e-9:
            ax.plot(degrees, predicted[:, i], "b--", lw=0.7, alpha=0.6,
                    label="unshifted")
        row = metrics.iloc[i]
        ax.set_title(
            f"{column}  RMSE {row['rmse']:.3f}  r={row['correlation']:.3f}",
            fontsize=8,
        )
        ax.grid(alpha=0.25)
        ax.set_xlim(degrees.min(), degrees.max())
        ax.tick_params(labelsize=7)
        if i % n_cols == 0:
            ax.set_ylabel("normalized flux", fontsize=8)
    for j in range(n_cameras, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.supxlabel("rotation angle [deg]", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_residual_heatmap(
    predicted: np.ndarray,
    observed: np.ndarray,
    columns: list[str],
    degrees: np.ndarray,
    out_png: Path,
    *,
    title: str,
    phase_deg: float = 0.0,
) -> None:
    """Frame x camera residual matrix; rows are rotation angle, cols cameras."""
    shifted = periodic_shift(predicted, degrees, phase_deg)
    residual = shifted - observed
    fig, ax = plt.subplots(figsize=(max(5.0, 0.35 * len(columns)), 4.6))
    vmax = float(np.percentile(np.abs(residual), 98)) or 1.0
    im = ax.imshow(
        residual.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=[degrees.min(), degrees.max(), len(columns) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=6)
    ax.set_xlabel("rotation angle [deg]", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="model - observed", shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# End-to-end driver
# --------------------------------------------------------------------------- #

def visualize_forward_match(
    model_num: int,
    *,
    origin: str = "simulated",
    curve_type: str = "intensity",
    frame_stride: int = 6,
    phase_step_degrees: float = 1.0,
    device: torch.device | str = "auto",
    backend: str = "cpu",
    target_faces: int = 1_280,
    gamma: float = 0.02,
    lambertian_fraction: float = 0.5,
    tau_intensity: float = DEFAULT_TAU_INTENSITY,
    beta_intensity: float = 64.0,
    tau_binary: float = 0.10,
    beta_binary: float = 8.0,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'")
    if origin not in ("real", "simulated"):
        raise ValueError("origin must be 'real' or 'simulated'")
    if curve_type not in ("intensity", "binary"):
        raise ValueError("curve_type must be 'intensity' or 'binary'")
    if frame_stride <= 0 or 360 % frame_stride != 0:
        raise ValueError("frame_stride must be a positive divisor of 360")
    if model_num not in (1, 2, 3):
        raise ValueError("public challenge data exists only for asteroids 1-3")

    if isinstance(device, str):
        device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
    columns = list(UNIQUE_COLUMNS)
    observed, degrees = load_observed(
        model_num, origin, curve_type, columns, frame_stride
    )

    cache_dir = Path("artifacts/forward_validation")
    mesh = load_benchmark_mesh(
        model_num, target_faces, device, cache_dir, "star_proxy"
    )
    if backend == "cpu":
        predicted = predict_cpu(
            mesh, columns, degrees, device,
            gamma=gamma, lambertian_fraction=lambertian_fraction,
            tau_intensity=tau_intensity, beta_intensity=beta_intensity,
            tau_binary=tau_binary, beta_binary=beta_binary,
            curve_type=curve_type,
        )
    else:
        predicted = predict_gpu(
            mesh, columns, degrees, device, curve_type=curve_type
        )

    # Phase-align on the camera with the strongest observed signal.
    alignment_index = int(np.argmax(observed.std(axis=0)))
    phase, _ = estimate_phase(
        predicted[:, alignment_index],
        observed[:, alignment_index],
        degrees,
        phase_step_degrees,
    )
    shifted = periodic_shift(predicted, degrees, phase)
    metrics = metrics_table(shifted, observed, columns)
    aligned_metrics = curve_metrics(
        shifted.reshape(-1), observed.reshape(-1)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    label = (
        f"Asteroid {model_num} ({origin}, {curve_type}, {backend} backend, "
        f"{target_faces}-face proxy)"
    )
    title = (
        f"{label}\nphase {phase:+.1f} deg, mean RMSE "
        f"{aligned_metrics['rmse']:.3f}, mean r "
        f"{aligned_metrics['correlation']:.3f}"
    )
    curves_png = output_dir / f"asteroid{model_num}_{curve_type}_curves.png"
    residual_png = output_dir / f"asteroid{model_num}_{curve_type}_residuals.png"
    plot_curve_match(
        predicted, observed, columns, degrees, curves_png,
        title=title, phase_deg=phase,
    )
    plot_residual_heatmap(
        predicted, observed, columns, degrees, residual_png,
        title=f"Asteroid {model_num} residual matrix ({curve_type})",
        phase_deg=phase,
    )
    metrics_csv = output_dir / f"asteroid{model_num}_{curve_type}_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    summary = {
        "asteroid": model_num,
        "origin": origin,
        "curve_type": curve_type,
        "backend": backend,
        "phase_degrees": phase,
        "alignment_camera": columns[alignment_index],
        "mean_rmse": float(aligned_metrics["rmse"]),
        "mean_correlation": float(aligned_metrics["correlation"]),
        "per_camera_rmse": {
            row["camera"]: float(row["rmse"]) for _, row in metrics.iterrows()
        },
        "per_camera_correlation": {
            row["camera"]: float(row["correlation"])
            for _, row in metrics.iterrows()
        },
        "outputs": {
            "curves": str(curves_png),
            "residuals": str(residual_png),
            "metrics_csv": str(metrics_csv),
        },
    }
    with (output_dir / f"asteroid{model_num}_{curve_type}_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asteroids", default="1")
    parser.add_argument("--origin", choices=("real", "simulated"),
                        default="simulated")
    parser.add_argument("--curve-type", choices=("intensity", "binary"),
                        default="intensity")
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--phase-step-degrees", type=float, default=1.0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-faces", type=int, default=1_280)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--lambertian-fraction", type=float, default=0.5)
    parser.add_argument("--tau-intensity", type=float,
                        default=DEFAULT_TAU_INTENSITY)
    parser.add_argument("--beta-intensity", type=float, default=64.0)
    parser.add_argument("--tau-binary", type=float, default=0.10)
    parser.add_argument("--beta-binary", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asteroids = [int(item) for item in args.asteroids.split(",") if item.strip()]
    for model_num in asteroids:
        summary = visualize_forward_match(
            model_num,
            origin=args.origin,
            curve_type=args.curve_type,
            frame_stride=args.frame_stride,
            phase_step_degrees=args.phase_step_degrees,
            device=args.device,
            backend=args.backend,
            target_faces=args.target_faces,
            gamma=args.gamma,
            lambertian_fraction=args.lambertian_fraction,
            tau_intensity=args.tau_intensity,
            beta_intensity=args.beta_intensity,
            tau_binary=args.tau_binary,
            beta_binary=args.beta_binary,
            output_dir=args.output_dir,
        )
        print(
            f"asteroid {model_num}: phase {summary['phase_degrees']:+.1f} deg, "
            f"mean RMSE {summary['mean_rmse']:.4f}, "
            f"mean r {summary['mean_correlation']:.4f} "
            f"-> {summary['outputs']['curves']}"
        )


if __name__ == "__main__":
    main()
