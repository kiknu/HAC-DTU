from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from hac.constants import CHALLENGE_DATA_DIR, BASE_RADII


def get_model_folder(model_num: int) -> Path:
    """
    Returns a Path object to the folder containing the data for the specified model number.
    """
    _model_folder_suffix = "public" if model_num <= 3 else "secret"
    model_folder_str = f"AsteroidModel{model_num:02d}_shape_{_model_folder_suffix}"

    model_dir = CHALLENGE_DATA_DIR / model_folder_str
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model data directory not found at {model_dir}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    return model_dir


def get_model_stl_path(model_num: int) -> Path:
    """
    Returns a Path object to the STL file containing the shape model for the specified model number.
    """
    if model_num > 3:
        raise ValueError(
            f"Model number {model_num} is a secret model and cannot be accessed."
        )

    model_dir = get_model_folder(model_num)

    stl_file_str = f"asteroid{model_num:d}.stl"
    stl_file_path = model_dir / stl_file_str
    if not stl_file_path.is_file():
        raise FileNotFoundError(
            f"STL file not found at {stl_file_path}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    return stl_file_path


@dataclass(frozen=True, slots=True)
class Dataset:
    """
    A class to represent a dataset of light curves for a specific model.

    See: https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
    """

    model_num: int  # The model number (1-10)
    origin: Literal[
        "real", "simulated"
    ]  # Whether the data is from camera setup (real) or Blender simulation (simulated)
    curve_type: Literal[
        "binary", "intensity"
    ]  # Whether the data is binary (0/1) or intensity (0-255)

    data: pd.DataFrame  # The light curve data, with standardized index and columns
    base_radius: float

    def melt_data_to_long(self) -> pd.DataFrame:
        """
        Melt the data from wide format to long format for use with seaborn.
        """
        long = self.data.reset_index().melt(
            id_vars="frame", var_name="camera", value_name="value"
        )

        long[["cam_angle", "cam_pos"]] = long["camera"].str.split("_", n=1, expand=True)
        long["cam_angle"] = long["cam_angle"].astype(int)

        long = long[["frame", "cam_angle", "cam_pos", "value"]]
        return long


def load_data(
    model_num: int,
    origin: Literal["real", "simulated"],
    curve_type: Literal["binary", "intensity"],
) -> Dataset:
    model_dir = get_model_folder(model_num)

    lightcurve_data_folder_str = f"Asteroid{model_num:d}_lightcurve_data"
    lightcurve_data_dir = model_dir / lightcurve_data_folder_str
    if not lightcurve_data_dir.is_dir():
        raise FileNotFoundError(
            f"Light curve data directory not found at {lightcurve_data_dir}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    data_file_str = f"Asteroid0{model_num}_lightcurve_{curve_type}"
    if origin == "simulated":
        data_file_str += "_blender"

    data_file_str += ".txt"

    data_file_path = lightcurve_data_dir / data_file_str
    if not data_file_path.is_file():
        raise FileNotFoundError(
            f"Data file not found at {data_file_path}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    # Load the data into a Dataframe
    data = pd.read_csv(data_file_path, header=None, index_col=0)

    # Standardize index and columns according to https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
    data.index.name = "frame"
    data = data.rename(
        columns={
            1: "0_hor1",
            2: "0_hor2",
            3: "0_top",
            4: "0_bottom",
            5: "45_hor1",
            6: "45_hor2",
            7: "45_top",
            8: "45_bottom",
            9: "90_hor1",
            10: "90_hor2",
            11: "90_top",
            12: "90_bottom",
            13: "135_hor1",
            14: "135_hor2",
            15: "135_top",
            16: "135_bottom",
            17: "225_hor1",
            18: "225_hor2",
            19: "225_top",
            20: "225_bottom",
            21: "270_hor1",
            22: "270_hor2",
            23: "270_top",
            24: "270_bottom",
            25: "315_hor1",
            26: "315_hor2",
            27: "315_top",
            28: "315_bottom",
        }
    )

    return Dataset(
        model_num=model_num,
        origin=origin,
        curve_type=curve_type,
        data=data,
        base_radius=BASE_RADII[model_num - 1],
    )
