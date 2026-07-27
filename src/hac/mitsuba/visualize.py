"""Visualization helpers for Mitsuba renders."""

from os import PathLike
from pathlib import Path
from typing import Any

import drjit as dr
import imageio_ffmpeg
import numpy as np
from numpy.typing import ArrayLike, NDArray

from hac.mitsuba.patch import mi


def bitmap_frames(rendered_frames: Any) -> NDArray[np.uint8]:
    """Convert ``(phase, height, width, 1)`` renders to uint8 RGB frames."""
    shape = tuple(rendered_frames.shape)
    if len(shape) != 4 or shape[-1] != 1:
        raise ValueError(
            f"rendered_frames must have shape (phase, height, width, 1); got {shape}."
        )

    frame_count, height, width, _ = shape
    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError(f"rendered_frames dimensions must be positive; got {shape}.")

    strip = dr.transpose(rendered_frames, axes=(1, 0, 2, 3))
    strip = dr.reshape(strip, (height, frame_count * width, 1))
    bitmap = np.asarray(mi.util.convert_to_bitmap(strip))
    return np.transpose(
        bitmap.reshape(height, frame_count, width, 3),
        axes=(1, 0, 2, 3),
    )


def write_movie(
    path: str | PathLike[str],
    frames: ArrayLike,
    fps: int,
) -> None:
    """Write uint8 RGB ``(frame, height, width, 3)`` data as an MP4 movie."""
    output_path = Path(path)
    frame_array = np.asarray(frames)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(
            "frames must have shape (frame, height, width, 3); "
            f"got {frame_array.shape}."
        )
    if 0 in frame_array.shape[:3]:
        raise ValueError(
            f"frames dimensions must be positive; got {frame_array.shape}."
        )
    if frame_array.dtype != np.uint8:
        raise ValueError(f"frames must have dtype uint8; got {frame_array.dtype}.")
    if fps <= 0:
        raise ValueError(f"fps must be positive; got {fps}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_array.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18"],
    )
    writer.send(None)
    try:
        for frame in frame_array:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()
