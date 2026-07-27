import os
from collections.abc import Sequence
from pathlib import Path

from hac.types import ObserverGeometry

PROJECT_DIR = Path(__file__).resolve().parents[2]
DSDIR_ENV_VAR = "DSDIR"


def _resolve_challenge_data_dir() -> Path:
    dsdir = os.environ.get(DSDIR_ENV_VAR)
    if not dsdir:
        return PROJECT_DIR / "challenge_data"

    challenge_data_dir = Path(dsdir).expanduser()
    if challenge_data_dir.name != "challenge_data":
        challenge_data_dir = challenge_data_dir / "challenge_data"

    return challenge_data_dir.resolve()


CHALLENGE_DATA_DIR = _resolve_challenge_data_dir()

# Maps position id to camera angle alpha in degrees
# See: https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
# This is left for backwards-compatibility, but may be removed in the future
# Consider using OBSERVERS instead
CAMERA_ANGLE_ALPHA: dict[str, float] = {
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

# All observer positions according to https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
_DEFAULT_OBSERVER_DISTANCE = 5.0  # Must be greater than largest asteroid radius (3.95)
OBSERVERS: dict[str, ObserverGeometry] = {
    "0_hor1": ObserverGeometry(theta=0.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE),
    "0_hor2": ObserverGeometry(theta=0.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE),
    "0_top": ObserverGeometry(theta=0.0, alpha=21.0, r_obs=_DEFAULT_OBSERVER_DISTANCE),
    "0_bottom": ObserverGeometry(
        theta=0.0, alpha=-21.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "45_hor1": ObserverGeometry(
        theta=45.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "45_hor2": ObserverGeometry(
        theta=45.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "45_top": ObserverGeometry(
        theta=45.0, alpha=26.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "45_bottom": ObserverGeometry(
        theta=45.0, alpha=-24.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "90_hor1": ObserverGeometry(
        theta=90.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "90_hor2": ObserverGeometry(
        theta=90.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "90_top": ObserverGeometry(
        theta=90.0, alpha=26.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "90_bottom": ObserverGeometry(
        theta=90.0, alpha=-24.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "135_hor1": ObserverGeometry(
        theta=135.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "135_hor2": ObserverGeometry(
        theta=135.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "135_top": ObserverGeometry(
        theta=135.0, alpha=26.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "135_bottom": ObserverGeometry(
        theta=135.0, alpha=-24.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "225_hor1": ObserverGeometry(
        theta=225.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "225_hor2": ObserverGeometry(
        theta=225.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "225_top": ObserverGeometry(
        theta=225.0, alpha=24.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "225_bottom": ObserverGeometry(
        theta=225.0, alpha=-26.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "270_hor1": ObserverGeometry(
        theta=270.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "270_hor2": ObserverGeometry(
        theta=270.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "270_top": ObserverGeometry(
        theta=270.0, alpha=24.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "270_bottom": ObserverGeometry(
        theta=270.0, alpha=-26.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "315_hor1": ObserverGeometry(
        theta=315.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "315_hor2": ObserverGeometry(
        theta=315.0, alpha=0.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "315_top": ObserverGeometry(
        theta=315.0, alpha=24.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
    "315_bottom": ObserverGeometry(
        theta=315.0, alpha=-26.0, r_obs=_DEFAULT_OBSERVER_DISTANCE
    ),
}

# The bounding cylinder radius for each asteroid model
MODEL_BASE_RADII: dict[int, float] = {
    1: 1.12,
    2: 1.42,
    3: 0.88,
    4: 1.475,
    5: 1.22,
    6: 0.925,
    7: 1.205,
    8: 1.24,
    9: 0.67,
    10: 3.95,
}
# Backward compatibility
BASE_RADII = list(MODEL_BASE_RADII.values())


class NamedAngleSequence(Sequence):
    def __init__(self, names: list[str], values: list[float | int]):
        self.names = names
        self.values = values
        self.by_name = dict(zip(names, values))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.by_name[key]
        return self.values[key]

    def __len__(self):
        return len(self.values)


POSITION_IDS = list(CAMERA_ANGLE_ALPHA.keys())

XY_ANGLES = NamedAngleSequence(
    POSITION_IDS,
    [int(position_id.split("_", 1)[0]) for position_id in POSITION_IDS],
)

ELEV_ANGLES = NamedAngleSequence(
    POSITION_IDS,
    [CAMERA_ANGLE_ALPHA[position_id] for position_id in POSITION_IDS],
)

INTENSITY_CURVE_THRESHOLD = 25.5
