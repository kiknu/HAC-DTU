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
    source: Path, output: Path, *, fps: int = 12, all_iterations: bool = False
) -> Path:
    trajectory = load_trajectory(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(trajectory.vertices)
    if not all_iterations and trajectory.best_index is not None:
        frame_count = trajectory.best_index + 1
    vertices = trajectory.vertices[:frame_count]
    limit = 1.08 * float(np.abs(vertices).max())
    fig = plt.figure(figsize=(5, 5))
    axis = fig.add_subplot(111, projection="3d")

    def update(index: int):
        axis.cla()
        _add_mesh(axis, vertices[index], trajectory.faces)
        _set_axes(axis, limit, 22, 135, f"iteration {index}")

    animation = FuncAnimation(fig, update, frames=frame_count)
    writer = FFMpegWriter(fps=fps) if output.suffix.lower() == ".mp4" else PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=110)
    plt.close(fig)
    return output
