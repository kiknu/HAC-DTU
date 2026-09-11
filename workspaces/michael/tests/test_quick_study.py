"""Short-study scope, scheduling, compatibility, and report regression checks."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hac_asteroids.calibration import Calibration
from hac_asteroids.inversion import optimization_state
from hac_asteroids.representation_reporting import VARIANTS, report_campaign
from hac_asteroids.representation_study import (
    PROFILES, benchmark, config_for, prepare, register_profile,
)


class QuickStudyTests(unittest.TestCase):
    def test_quick_schedule_preserves_the_scientific_controls(self):
        for _, _, representation, degree in VARIANTS:
            full = config_for(3, representation, degree)
            quick = config_for(3, representation, degree, profile="quick")
            changed = {k for k, value in asdict(full).items() if value != asdict(quick)[k]}
            self.assertEqual(changed, {"steps", "escape_steps", "stabilization_steps", "validation_interval"})
            self.assertEqual((quick.steps, quick.validation_interval), (120, 20))
            stages = [optimization_state(quick, step).stage for step in range(quick.steps)]
            self.assertEqual([stages.count(name) for name in ("escape", "refinement", "stabilization")], [12, 96, 12])
            self.assertEqual((quick.faces, quick.initialization, quick.initialization_noise), (1024, "sphere", 0.))
            self.assertEqual((quick.frame_count, quick.validation_frame_count, quick.image_size), (72, 72, 96))
        self.assertEqual(PROFILES["full"].asteroids, (1,2,3))
        self.assertEqual(PROFILES["quick"].asteroids, (3,))
        with self.assertRaisesRegex(ValueError, "outside"):
            config_for(1, "radial_sh", 6, profile="quick")

    def test_profiles_cannot_relabel_existing_studies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            register_profile(root, "quick")
            original = (root / "study.json").read_bytes()
            register_profile(root, "quick")
            with self.assertRaisesRegex(ValueError, "profile changed"):
                register_profile(root, "full")
            self.assertEqual((root / "study.json").read_bytes(), original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = {f"{a}/{key}": asdict(config_for(a, rep, degree))
                      for a in (1,2,3) for key,_,rep,degree in VARIANTS}
            (root / "frozen_configuration.json").write_text(json.dumps(dict(configs=frozen)))
            with self.assertRaisesRegex(ValueError, "does not match"):
                register_profile(root, "quick")
            self.assertFalse((root / "study.json").exists())
            register_profile(root, "full")

    def test_quick_preparation_only_reuses_asteroid_three_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("hac_asteroids.representation_study.calibrate_public") as calibrate:
                calibrations = prepare(root, "quick")
            self.assertEqual(set(calibrations), {3})
            calibrate.assert_not_called()
            self.assertTrue((root / "calibration/asteroid03/calibration.json").exists())
            self.assertFalse((root / "calibration/asteroid01").exists())
            self.assertFalse((root / "calibration/asteroid02").exists())

    def test_quick_benchmark_uses_its_own_schedule_and_run_count(self):
        calibration = Calibration(asteroids=(3,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("torch.cuda.get_device_name", return_value="test GPU"), \
                 patch("hac_asteroids.representation_study.benchmark_variant", return_value={"projected_fit_seconds":900.}) as run:
                self.assertEqual(benchmark(root, {3:calibration}, "quick"), 6)
                self.assertEqual(run.call_count, 3)
                for call in run.call_args_list:
                    self.assertEqual((call.args[0].asteroid, call.args[0].steps), (3,120))
                    self.assertEqual(call.kwargs["schedule_steps"], (0,12,108))
                # A JSON round trip must still accept the cached quick benchmark.
                benchmark(root, {3:calibration}, "quick")
                self.assertEqual(run.call_count, 3)
            result = json.loads((root / "benchmark.json").read_text())
            self.assertEqual(result["projected_campaign_seconds"], 2700.)
            self.assertEqual(result["expected_runs"], 3)
            self.assertNotIn("projected_nine_fit_seconds", result)

    def test_quick_report_uses_three_runs_and_step_120_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            register_profile(root, "quick")
            body = root / "asteroid03"
            body.mkdir()
            (body / "report_complete.json").write_text("{}")
            report_campaign(root)
            text = (root / "report.md").read_text()
            self.assertIn("**0/3**", text)
            self.assertIn("120 updates", text)
            self.assertIn("Step-120 RMSE", text)
            self.assertIn("asteroid03/lightcurves_step120.png", text)
            self.assertNotIn("500", text)
            self.assertNotIn("Asteroid 1", text)


if __name__ == "__main__":
    unittest.main()
