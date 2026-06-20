import os
from pathlib import Path


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
CAMERA_ANGLE_ALPHA: dict[str, float] = {
    "0_hor1": 0.0,
    "0_hor2": 0.0,
    "0_top": 21.0,
    "0_bottom": -21.0,
    "45_hor1": 0.0,
    "45_hor2": 0.0,
    "45_top": 26.0,
    "45_bottom": -26.0,
    "90_hor1": 0.0,
    "90_hor2": 0.0,
    "90_top": 26.0,
    "90_bottom": -26.0,
    "135_hor1": 0.0,
    "135_hor2": 0.0,
    "135_top": 26.0,
    "135_bottom": -26.0,
    "225_hor1": 0.0,
    "225_hor2": 0.0,
    "225_top": 24.0,
    "225_bottom": -24.0,
    "270_hor1": 0.0,
    "270_hor2": 0.0,
    "270_top": 24.0,
    "270_bottom": -24.0,
    "315_hor1": 0.0,
    "315_hor2": 0.0,
    "315_top": 24.0,
    "315_bottom": -24.0,
}
