"""Render a saved mesh trajectory without rerunning optimization."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from visualize_fit import _add_mesh, _set_axes

from .trajectory import load_trajectory


def animate_trajectory(
    source: Path, output: Path, *, fps: int = 12, all_iterations: bool = False,
    stride: int = 1, axis_limit: float | None = None, label: str = "",
) -> Path:
    trajectory = load_trajectory(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(trajectory.vertices)
    if not all_iterations and trajectory.best_index is not None:
        frame_count = trajectory.best_index + 1
    indices = animation_indices(frame_count, stride, trajectory.best_index)
    vertices = trajectory.vertices
    limit = axis_limit or 1.08 * float(np.abs(vertices[:frame_count]).max())
    fig = plt.figure(figsize=(5, 5))
    axis = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        index = int(indices[frame])
        axis.cla()
        _add_mesh(axis, vertices[index], trajectory.faces)
        selected = " (selected best)" if index == trajectory.best_index else ""
        _set_axes(axis, limit, 22, 135, f"{label}\niteration {index}{selected}")

    animation = FuncAnimation(fig, update, frames=len(indices))
    writer = FFMpegWriter(fps=fps) if output.suffix.lower() == ".mp4" else PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=110)
    plt.close(fig)
    return output


def animation_indices(frame_count: int, stride: int, best_index: int | None = None) -> np.ndarray:
    """Sample history while retaining the initial, selected and terminal states."""
    if frame_count < 1 or stride < 1:
        raise ValueError("frame_count and stride must be positive")
    indices = set(range(0, frame_count, stride)) | {frame_count - 1}
    if best_index is not None and 0 <= best_index < frame_count:
        indices.add(best_index)
    return np.asarray(sorted(indices), dtype=np.int64)


def animate_comparison(sources: list[Path], labels: list[str], output: Path, *,
                       stride: int = 5, fps: int = 12, stop_reasons: list[str] | None = None) -> Path:
    """Compare complete histories at the same actual iteration and spatial scale."""
    histories = [load_trajectory(source) for source in sources]
    if len(labels) != len(histories) or not histories or (stop_reasons is not None and len(stop_reasons) != len(histories)):
        raise ValueError("comparison needs matching trajectories, labels and stopping reasons")
    indices = sorted(set().union(*(animation_indices(len(t.vertices), stride, t.best_index) for t in histories)))
    limit = 1.08 * max(float(np.abs(t.vertices).max()) for t in histories)
    fig = plt.figure(figsize=(5 * len(histories), 5))
    axes = [fig.add_subplot(1, len(histories), i + 1, projection="3d") for i in range(len(histories))]

    def update(frame):
        index = indices[frame]
        for i, (axis, trajectory, label) in enumerate(zip(axes, histories, labels)):
            actual = min(index, len(trajectory.vertices)-1)
            axis.cla()
            _add_mesh(axis, trajectory.vertices[actual], trajectory.faces)
            selected = " (selected best)" if actual == trajectory.best_index else ""
            stopped = (f"\nStopped: {stop_reasons[i].replace('_', ' ')}"
                       if stop_reasons and actual == len(trajectory.vertices)-1 else "")
            _set_axes(axis, limit, 22, 135, f"{label}\niteration {actual}{selected}{stopped}")

    animation = FuncAnimation(fig, update, frames=len(indices))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    return output
