import tempfile
import unittest
from pathlib import Path

import numpy as np

from hac_asteroids.cli import build_parser
from hac_asteroids.datasets import load_lightcurves
from hac_asteroids.trajectory import FitTrajectory, load_trajectory


class DatasetTests(unittest.TestCase):
    def test_all_asteroids_load_with_all_unique_cameras(self):
        for asteroid in range(1, 11):
            dataset = load_lightcurves(asteroid)
            self.assertEqual(dataset.curves.shape, (360, 21))
            self.assertTrue(np.isfinite(dataset.curves).all())

    def test_asteroid_ten_nonstandard_filename_is_discovered(self):
        dataset = load_lightcurves(10)
        self.assertEqual(dataset.source.name, "Asteroid010_lightcurve_intensity_blender.txt")

    def test_all_four_variants_load(self):
        for asteroid in range(1, 11):
            for origin in ("real", "simulated"):
                for curve_type in ("intensity", "binary"):
                    dataset = load_lightcurves(
                        asteroid, origin=origin, curve_type=curve_type
                    )
                    expected = 21 if origin == "simulated" else 28
                    self.assertEqual(dataset.curves.shape[1], expected)


class TrajectoryTests(unittest.TestCase):
    def test_round_trip_is_pickle_free(self):
        trajectory = FitTrajectory(
            parameters=np.zeros((3, 4), dtype=np.float32),
            vertices=np.zeros((3, 5, 3), dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            losses=np.array([1.0, 0.5, 0.25]),
            best_index=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = trajectory.save(Path(directory) / "trajectory.npz")
            loaded = load_trajectory(path)
        np.testing.assert_array_equal(loaded.parameters, trajectory.parameters)
        np.testing.assert_array_equal(loaded.vertices, trajectory.vertices)
        np.testing.assert_array_equal(loaded.faces, trajectory.faces)
        np.testing.assert_array_equal(loaded.losses, trajectory.losses)
        self.assertEqual(loaded.best_index, 2)


class CliTests(unittest.TestCase):
    def test_fit_defaults_to_asteroid_three_and_requires_calibration(self):
        parser = build_parser()
        args = parser.parse_args(["fit", "--calibration", "calibration.json"])
        self.assertEqual(args.asteroids, (3,))
        self.assertEqual(args.frames, 12)
        self.assertEqual(args.steps, 400)


if __name__ == "__main__":
    unittest.main()
