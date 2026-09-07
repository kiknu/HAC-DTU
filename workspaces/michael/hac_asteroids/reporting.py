"""Plots and mesh exports for inversion runs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from visualize_fit import plot_shape_grid, radial_support


REPRESENTATIVE_CAMERAS = ("0_hor1", "45_top", "225_bottom", "315_top")


def export_mesh(vertices: np.ndarray, faces: np.ndarray, path: Path) -> Path:
    path = Path(path)
    trimesh.Trimesh(vertices, faces, process=False).export(path)
    return path


def plot_convergence(losses: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(np.arange(len(losses)), losses, "k-")
    axis.set(xlabel="iteration", ylabel="objective", title="Inversion convergence")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def curve_report(
    degrees: np.ndarray,
    observed: np.ndarray,
    initial: np.ndarray,
    fitted: np.ndarray,
    columns: tuple[str, ...],
    output_dir: Path,
) -> list[dict[str, float | str]]:
    output_dir = Path(output_dir)
    rows = []
    for index, name in enumerate(columns):
        residual = fitted[:, index] - observed[:, index]
        rows.append({
            "camera": name,
            "initial_rmse": float(np.sqrt(np.mean((initial[:, index] - observed[:, index]) ** 2))),
            "fitted_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "correlation": float(np.corrcoef(fitted[:, index], observed[:, index])[0, 1]),
        })
    with (output_dir / "camera_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = [columns.index(name) for name in REPRESENTATIVE_CAMERAS]
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 9), sharex=True)
    for axis, index in zip(axes, selected):
        axis.plot(degrees, observed[:, index], "k.", ms=3, label="observed")
        axis.plot(degrees, initial[:, index], color="steelblue", lw=1, label="initial")
        axis.plot(degrees, fitted[:, index], color="crimson", lw=1.4, label="fitted")
        axis.set_ylabel(columns[index])
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("rotation angle [deg]")
    fig.tight_layout()
    fig.savefig(output_dir / "forward_curves.png", dpi=160)
    plt.close(fig)

    residuals = fitted - observed
    fig, axis = plt.subplots(figsize=(11, 6))
    image = axis.imshow(
        residuals.T, aspect="auto", origin="lower",
        extent=(float(degrees[0]), float(degrees[-1]), -0.5, len(columns) - 0.5),
        cmap="coolwarm", vmin=-np.max(np.abs(residuals)), vmax=np.max(np.abs(residuals)),
    )
    axis.set_yticks(np.arange(len(columns)), columns)
    axis.set(xlabel="rotation angle [deg]", title="Fitted minus observed intensity")
    fig.colorbar(image, ax=axis, label="normalized residual")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_heatmap.png", dpi=160)
    plt.close(fig)
    return rows


def shape_report(
    initial: tuple[np.ndarray, np.ndarray],
    fitted: tuple[np.ndarray, np.ndarray],
    output_dir: Path,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    meshes = [initial, fitted]
    labels = ["initial", "fitted"]
    colors = [(0.55, 0.65, 0.85), (0.9, 0.55, 0.35)]
    if reference is not None:
        meshes.append(reference)
        labels.append("public reference")
        colors.append((0.68, 0.75, 0.68))
    plot_shape_grid(meshes, labels, colors, Path(output_dir) / "mesh_initial_final.png")


@torch.no_grad()
def align_and_score_public_meshes(
    initial: np.ndarray,
    fitted: np.ndarray,
    faces: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_vertices = torch.as_tensor(reference[0], dtype=torch.float32, device=device)
    reference_faces = torch.as_tensor(reference[1], dtype=torch.long, device=device)
    results, aligned = {}, []
    for label, values in (("initial", initial), ("fitted", fitted)):
        vertices = torch.as_tensor(values, dtype=torch.float32, device=device)
        directions = torch.nn.functional.normalize(vertices, dim=-1)
        radii = vertices.norm(dim=-1)
        target_radii = radial_support(reference_vertices, reference_faces, directions)
        scale = torch.exp((torch.log(target_radii) - torch.log(radii)).mean())
        error = torch.log(scale * radii) - torch.log(target_radii)
        aligned.append((scale * vertices).cpu().numpy())
        results[label] = {
            "radial_log_rms": float(error.square().mean().sqrt()),
            "radial_log_max": float(error.abs().max()),
            "scale": float(scale),
        }
    return aligned[0], aligned[1], results
