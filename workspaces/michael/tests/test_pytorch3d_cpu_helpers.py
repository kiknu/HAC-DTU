import unittest

import numpy as np
import torch
import trimesh

from forward_validation import normalize_challenge_mesh
from pytorch3d_forward import (
    ObservationBatch,
    challenge_view_directions,
    extract_lightcurves,
    otsu_threshold_uint8,
)


class PyTorch3DIndependentHelperTests(unittest.TestCase):
    def test_otsu_and_exact_extraction(self):
        image = torch.zeros((1, 4, 4), dtype=torch.float32)
        image[:, :2] = 1.0
        threshold = otsu_threshold_uint8(image)
        self.assertEqual(threshold.shape, (1,))
        prediction = extract_lightcurves(
            image, 1, 1, mode="exact"
        )
        self.assertEqual(float(prediction.raw_intensity[0, 0]), 8.0)
        self.assertEqual(float(prediction.raw_binary[0, 0]), 8.0)

    def test_camera_geometry_without_renderer(self):
        directions = challenge_view_directions(
            ("0_top", "225_bottom"), device="cpu", dtype=torch.float32
        )
        elevations = torch.rad2deg(torch.asin(directions[:, 2]))
        torch.testing.assert_close(
            elevations, torch.tensor([21.0, -26.0]), atol=1e-5, rtol=0.0
        )
        observations = ObservationBatch.challenge(
            torch.tensor([0.0]), ("0_top", "225_bottom")
        )
        self.assertEqual(observations.view_directions.shape, (2, 3))

    def test_normalization_without_renderer(self):
        source = trimesh.creation.box(extents=(4.0, 6.0, 8.0))
        source.apply_translation((2.0, -3.0, 7.0))
        normalized = normalize_challenge_mesh(source)
        np.testing.assert_allclose(normalized.bounds[:, 2], (-1.0, 1.0))
        np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
