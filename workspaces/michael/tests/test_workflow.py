import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hac_asteroids.cli import _fit, build_parser
from hac_asteroids.datasets import load_lightcurves
from hac_asteroids.inversion import FitConfig, optimization_state, rotation_split
from hac_asteroids.trajectory import FitTrajectory, load_trajectory
from hac_asteroids.tuning import _aggregate, generate_candidates, run_tuning


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
    def test_transfer_choice_reaches_smoke_and_normal_fits(self):
        from hac_asteroids.calibration import Calibration

        def check_scope(config, calibration, *_args, **_kwargs):
            calibration.validate_scope(config.asteroid, config.calibration_transfer)
            return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = Calibration(asteroids=(2,)).save(root / "calibration.json")
            for smoke in (False, True):
                for transfer in (False, True):
                    with self.subTest(smoke=smoke, transfer=transfer):
                        arguments = ["fit", "--asteroids", "8", "--calibration", str(calibration),
                                     "--device", "cpu", "--output-dir", str(root / "runs")]
                        if smoke:
                            arguments.append("--smoke")
                        if transfer:
                            arguments.append("--calibration-transfer")
                        args = build_parser().parse_args(arguments)
                        with patch("hac_asteroids.inversion.run_inversion", side_effect=check_scope) as run:
                            if transfer:
                                self.assertEqual(_fit(args), 0)
                            else:
                                with self.assertRaisesRegex(ValueError, "calibration transfer"):
                                    _fit(args)
                        self.assertEqual(run.call_args.args[0].calibration_transfer, transfer)

    def test_fit_preserves_explicit_transfer_in_saved_config(self):
        from hac_asteroids.calibration import Calibration

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = Calibration(asteroids=(2,)).save(root / "calibration.json")
            config_path = root / "fit.json"
            config_path.write_text(json.dumps({"calibration_transfer": True}))
            args = build_parser().parse_args([
                "fit", "--asteroids", "8", "--calibration", str(calibration),
                "--config", str(config_path), "--output-dir", str(root / "runs"),
            ])
            with patch("hac_asteroids.inversion.run_inversion", return_value={}) as run:
                self.assertEqual(_fit(args), 0)
            self.assertTrue(run.call_args.args[0].calibration_transfer)

    def test_fit_defaults_to_asteroid_three_and_requires_calibration(self):
        parser = build_parser()
        args = parser.parse_args(["fit", "--calibration", "calibration.json"])
        self.assertEqual(args.asteroids, (3,))
        self.assertIsNone(args.frames)
        self.assertIsNone(args.steps)
        self.assertEqual(FitConfig(asteroid=3).steps, 300)

    def test_tune_defaults_describe_sixty_optimization_runs(self):
        args = build_parser().parse_args(["tune", "--calibration", "calibration.json"])
        coarse = args.candidates * len(args.asteroids)
        final = args.finalists * len(args.asteroids) * len(args.seeds)
        self.assertEqual((coarse, final, coarse + final), (36, 24, 60))

    def test_fit_loads_tuned_config_and_applies_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps({
                    "camera_distance_radii": 16.0,
                    "lambertian_fraction": 0.25,
                }),
                encoding="utf-8",
            )
            tuned = root / "best_config.json"
            tuned.write_text(
                json.dumps({
                    "schema_version": 1,
                    "fit_config": {
                        "steps": 300,
                        "initial_learning_rate": 0.011,
                        "regularization_weight": 0.012,
                    },
                }),
                encoding="utf-8",
            )
            args = build_parser().parse_args([
                "fit", "--calibration", str(calibration), "--config", str(tuned),
                "--learning-rate", "0.0015", "--output-dir", str(root / "runs"),
            ])
            with patch("hac_asteroids.inversion.run_inversion", return_value={}) as run:
                self.assertEqual(_fit(args), 0)
            config = run.call_args.args[0]
            self.assertEqual(config.steps, 300)
            self.assertEqual(config.initial_learning_rate, 0.011)
            self.assertEqual(config.learning_rate, 0.0015)
            self.assertEqual(config.regularization_weight, 0.012)


class OptimizationPolicyTests(unittest.TestCase):
    def test_rotation_split_is_deterministic_and_disjoint(self):
        first = rotation_split(360, 24, 24)
        second = rotation_split(360, 24, 24)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(np.intersect1d(*first)), 0)

    def test_schedule_has_escape_refinement_and_stabilization(self):
        config = FitConfig(asteroid=3)
        escape = optimization_state(config, 0)
        refinement = optimization_state(config, config.escape_steps)
        stabilization = optimization_state(config, config.steps - 1)
        self.assertEqual(escape.stage, "escape")
        self.assertEqual(escape.maximum_degree, 2)
        self.assertAlmostEqual(escape.learning_rate, config.initial_learning_rate)
        self.assertEqual(refinement.stage, "refinement")
        self.assertIsNone(refinement.maximum_degree)
        self.assertEqual(stabilization.stage, "stabilization")
        self.assertAlmostEqual(stabilization.learning_rate, config.min_learning_rate)
        self.assertAlmostEqual(stabilization.regularization_multiplier, 2.0)

    def test_candidates_and_ranking_are_deterministic(self):
        self.assertEqual(generate_candidates(), generate_candidates())
        rows = [
            {"candidate_id": "a", "best_validation_rmse": 0.2},
            {"candidate_id": "a", "best_validation_rmse": 0.4},
            {"candidate_id": "b", "best_validation_rmse": 0.25},
            {"candidate_id": "b", "best_validation_rmse": 0.25},
        ]
        ranking = _aggregate(rows)
        self.assertEqual(ranking[0]["candidate_id"], "b")

    def test_small_tuning_study_is_resumable_and_writes_reusable_config(self):
        calls = []

        def fake_run(config, _calibration, output, *, resume=None):
            self.assertIsNone(resume)
            calls.append(config)
            summary = {
                "best_validation_rmse": 0.1 + config.initial_learning_rate,
                "mean_fitted_rmse": 0.2,
                "runtime_seconds": 1.0,
            }
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(
                json.dumps({"config": config.__dict__, "summary": summary}),
                encoding="utf-8",
            )
            return summary

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("hac_asteroids.tuning.run_inversion", side_effect=fake_run):
                study = run_tuning(
                    object(), output, asteroids=(1,), candidate_count=2,
                    finalist_count=1, seeds=(0, 1), coarse_steps=2,
                    final_steps=3, device="cpu",
                )
                self.assertEqual(len(calls), 4)
                run_tuning(
                    object(), output, asteroids=(1,), candidate_count=2,
                    finalist_count=1, seeds=(0, 1), coarse_steps=2,
                    final_steps=3, device="cpu",
                )
            self.assertEqual(len(calls), 4)
            self.assertEqual(study["optimization_run_count"], 4)
            best = json.loads((output / "best_config.json").read_text())
            self.assertEqual(best["fit_config"]["steps"], 3)
            self.assertNotIn("asteroid", best["fit_config"])


if __name__ == "__main__":
    unittest.main()
