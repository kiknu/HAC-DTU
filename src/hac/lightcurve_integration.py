"""
Implements the lightcurve integration methods outlined in the challenge description.
"""

import numpy as np
from skimage.filters import threshold_otsu


def binary(image) -> float:
    """
    Performs the binary method outlined in the challenge to produce a lightcurve
    from a 2D grayscale image.

    See: https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
    """

    threshold = threshold_otsu(image)
    binary_image = image > threshold
    return np.sum(binary_image)


def intensity(image) -> float:
    """
    Performs the intensity method outlined in the challenge to produce a lightcurve
    from a 2D grayscale image.

    See: https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
    """
    threshold = 25.5
    intensity_image = np.where(image > threshold, image, 0)
    return np.sum(intensity_image)
