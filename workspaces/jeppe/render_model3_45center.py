"""Render asteroid model 3 as seen by observer 45center at phase 0.

Writes a high-resolution RGBA PNG with a transparent background, intended for
the report front page. The shaded image comes from the same Mitsuba scene used
by the shape optimization (sun fixed along +x, ``45_hor1`` observer). A second
render lit head-on from the camera provides the silhouette used as alpha, so the
unlit side of the asteroid stays opaque instead of vanishing with the background.

Usage: uv run python workspaces/jeppe/render_model3_45center.py [--size 3000] [--spp 512]
"""

# Runtime setup must precede imports from hac.mitsuba.
# ruff: noqa: E402

import argparse
import os

os.environ.setdefault("DSDIR", "/work3/projects/HAC-DTU/challenge_data")

from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from PIL import Image

try:
    mi.set_variant("cuda_ad_mono")
    dr.eval(dr.zeros(mi.Float, 1))  # fails without a usable GPU
except Exception:
    mi.set_variant("llvm_ad_mono")

import hac.mitsuba.scene as scene_module
from hac.constants import OBSERVERS
from hac.data import get_model_stl_path
from hac.mitsuba.forward import ForwardModel
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.scene import RenderingSettings, _observer_position

MODEL = 3
OBSERVER = "45_hor1"  # "45center" in the challenge video naming
PHASE_INDEX = 0
DEFAULT_OUTPUT = Path(__file__).parent / "figures" / "model3_45center_phase0.png"

EXPOSURE = 1.6  # Scales the physical intensity before the sRGB curve.
SHADOW_FLOOR = 0.10  # Lifts the unlit side so it reads against white paper.
CROP_MARGIN = 0.04  # Fraction of the object size kept around it.


def render_phase0(mesh: MitsubaMesh, size: int, spp: int) -> np.ndarray:
    observers = {OBSERVER: OBSERVERS[OBSERVER]}
    forward = ForwardModel(
        mesh,
        n_phases=1,
        observers=observers,
        rendering_settings=RenderingSettings(width=size, height=size, spp=spp),
        normalize_mesh=False,
    )
    frames = forward.render(seed=0)
    return np.asarray(frames[0, PHASE_INDEX, :, :, 0], dtype=np.float64)


def render_silhouette(mesh: MitsubaMesh, size: int, spp: int) -> np.ndarray:
    """Render with the sun placed at the camera so every visible pixel is lit."""
    original_sun = scene_module.SUN_DIRECTION
    camera = _observer_position(OBSERVERS[OBSERVER])
    scene_module.SUN_DIRECTION = -camera / np.linalg.norm(camera)
    try:
        return render_phase0(mesh, size, spp)
    finally:
        scene_module.SUN_DIRECTION = original_sun


def compose_rgba(shaded: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    alpha = np.clip(silhouette / 0.05, 0.0, 1.0)

    linear = np.clip(shaded * EXPOSURE, 0.0, 1.0)
    linear = SHADOW_FLOOR + (1.0 - SHADOW_FLOOR) * linear
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1 / 2.4) - 0.055,
    )
    grey = (np.clip(srgb, 0.0, 1.0) * 255).astype(np.uint8)
    rgba = np.dstack([grey, grey, grey, (alpha * 255).astype(np.uint8)])

    rows, cols = np.nonzero(alpha > 0.0)
    margin = int(CROP_MARGIN * max(np.ptp(rows), np.ptp(cols)))
    top, bottom = max(rows.min() - margin, 0), rows.max() + margin + 1
    left, right = max(cols.min() - margin, 0), cols.max() + margin + 1
    return rgba[top:bottom, left:right]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=3000, help="Render size in px.")
    parser.add_argument("--spp", type=int, default=512, help="Samples per pixel.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mesh = MitsubaMesh.from_stl_path(get_model_stl_path(MODEL))
    shaded = render_phase0(mesh, args.size, args.spp)
    silhouette = render_silhouette(mesh, args.size, args.spp)
    rgba = compose_rgba(shaded, silhouette)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(args.output, dpi=(300, 300))
    print(f"Wrote {args.output} ({rgba.shape[1]}x{rgba.shape[0]} px)")


if __name__ == "__main__":
    main()
