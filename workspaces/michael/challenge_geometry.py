"""Canonical camera names and elevations for the asteroid challenge."""

from __future__ import annotations


CAMERA_ALPHA_DEG: dict[str, float] = {
    "0_hor1": 0.0,
    "0_hor2": 0.0,
    "0_top": 21.0,
    "0_bottom": -21.0,
    "45_hor1": 0.0,
    "45_hor2": 0.0,
    "45_top": 26.0,
    "45_bottom": -24.0,
    "90_hor1": 0.0,
    "90_hor2": 0.0,
    "90_top": 26.0,
    "90_bottom": -24.0,
    "135_hor1": 0.0,
    "135_hor2": 0.0,
    "135_top": 26.0,
    "135_bottom": -24.0,
    "225_hor1": 0.0,
    "225_hor2": 0.0,
    "225_top": 24.0,
    "225_bottom": -26.0,
    "270_hor1": 0.0,
    "270_hor2": 0.0,
    "270_top": 24.0,
    "270_bottom": -26.0,
    "315_hor1": 0.0,
    "315_hor2": 0.0,
    "315_top": 24.0,
    "315_bottom": -26.0,
}

CHALLENGE_COLUMNS = tuple(CAMERA_ALPHA_DEG)
UNIQUE_COLUMNS = tuple(
    name for name in CHALLENGE_COLUMNS if not name.endswith("hor2")
)
