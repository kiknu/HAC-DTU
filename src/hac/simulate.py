import time
from pathlib import Path

import numpy as np
import pyvista as pv
from tqdm.auto import tqdm

from hac.data import get_model_stl_path
from hac.geometry import *
from hac.constants import *


def simulate_rotation_frames(
    mesh,
    cam_idx,
    out_path=None,
    rot_angles=np.arange(360),
    show_edges=False,
    cam_dist=30,
    focal_point=(0, 0, 0),
    image_size=(512, 512),
    light_dist=1000,
    verbose=True,
):
    """
    Simulate grayscale rendered frames of a rotating object.

    If out_path is None, frames are stored in memory and returned as a NumPy array.
    If out_path is provided, frames are written to a .npy memmap file and also
    returned as a NumPy array.

    Returns
    -------
    np.ndarray
        Array with shape (n_frames, height, width) and dtype uint8.
    """
    rot_angles = np.asarray(rot_angles, dtype=float)
    focal_point = np.asarray(focal_point, dtype=float)

    width, height = image_size
    frames_shape = (len(rot_angles), height, width)

    old_multi_samples = pv.global_theme.multi_samples
    pv.global_theme.multi_samples = 0

    if out_path is None:
        frames = np.empty(frames_shape, dtype=np.uint8)
    else:
        out_path = Path(out_path)
        frames = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.uint8,
            shape=frames_shape,
        )

    plotter = pv.Plotter(
        lighting="none",
        off_screen=True,
        window_size=image_size,
    )
    plotter.set_background("black")

    plotter.add_mesh(
        mesh,
        color="white",
        show_edges=show_edges,
        smooth_shading=False,
        ambient=0.0,
        diffuse=1.0,
        specular=0.0,
        pbr=True,
    )

    light = pv.Light(
        light_type="scene light",
        positional=False,
        intensity=1.0,
    )
    plotter.add_light(light)

    start = time.perf_counter()

    try:
        for i, rot_angle in enumerate(tqdm(rot_angles, desc=f"Rendering {cam_idx}")):
            direction = direction_from_angles(
                XY_ANGLES[cam_idx] + rot_angle,
                ELEV_ANGLES[cam_idx],
            )
            light_direction = direction_from_angles(rot_angle, 0)

            camera_position = focal_point + direction * cam_dist
            light_position = focal_point + light_direction * light_dist

            plotter.camera_position = [
                camera_position,
                focal_point,
                (0, 0, 1),
            ]

            # Orthographic projection (commented out as it gave worse results):
            # plotter.camera.parallel_projection = True
            # plotter.camera.parallel_scale = 6

            light.position = light_position
            light.focal_point = focal_point

            plotter.render()

            rgb = plotter.screenshot(
                return_img=True,
                transparent_background=False,
            )

            if rgb.shape[-1] != 3:
                raise ValueError(
                    f"Expected RGB image with 3 channels, got shape {rgb.shape}"
                )

            frames[i] = rgb[..., :3].mean(axis=-1).astype(np.uint8)

    finally:
        plotter.close()
        pv.global_theme.multi_samples = old_multi_samples

    if out_path is not None:
        frames.flush()

    elapsed = time.perf_counter() - start
    fps = len(rot_angles) / elapsed

    if verbose:
        if out_path is None:
            print("Generated frames in memory")
        else:
            print(f"Saved frames: {out_path}")

        print(f"Shape: {frames.shape}, dtype: {frames.dtype}")
        print(f"Elapsed: {elapsed:.2f} s")
        print(f"Speed: {fps:.2f} frames/s")

    return np.asarray(frames)


def generate_challenge_lightcurve(
    model_num: int,
    cam_idx: str,
    sim_root: str | Path | None = None,
) -> np.ndarray:
    """
    Generates a lightcurve for a challenge model with known shape.
    To do this, frames are generated to match one-to-one the blender frame sequence.
    The lightcurve is normalized like described on the challenge website.

    If sim_root is provided, saves the lightcurve as:
        sim_root / model_<model_num> / cam_<cam_idx>.npy
    """
    stl_path = get_model_stl_path(model_num)
    mesh = pv.read(stl_path)

    frames = simulate_rotation_frames(
        mesh=mesh,
        cam_idx=cam_idx,
        rot_angles=np.arange(360),
        show_edges=False,
        cam_dist=30,
        focal_point=(0, 0, 0),
        image_size=(1024, 1024),
        light_dist=1000,
        verbose=True,
    )

    thresholded_frames = np.where(
        frames > INTENSITY_CURVE_THRESHOLD,
        frames,
        0,
    )

    lightcurve = np.sum(thresholded_frames, axis=(1, 2)).astype(float)

    mean_intensity = lightcurve.mean()

    lightcurve = lightcurve / mean_intensity

    if sim_root is not None:
        model_dir = Path(sim_root) / f"model_{model_num:02d}"
        model_dir.mkdir(parents=True, exist_ok=True)

        save_path = model_dir / f"cam_{cam_idx}.npy"
        np.save(save_path, lightcurve)

        print(f"Saved lightcurve: {save_path}")

    return lightcurve

if __name__ == "__main__":
    generate_challenge_lightcurve(
        model_num=2,
        cam_idx="45_top",
        sim_root="workspaces/david/out/sim_lc"
    )