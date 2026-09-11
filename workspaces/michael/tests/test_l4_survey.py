"""Frozen survey inputs, hidden-reference reporting, and resumable CUDA smoke."""
from dataclasses import asdict, replace
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hac_asteroids.calibration import Calibration
from hac_asteroids.inversion import _build_model, optimization_state, run_inversion
from hac_asteroids.l4_survey import ASTEROIDS, config_for, prepare
from hac_asteroids.l4_survey_reporting import verify_body
from hac_asteroids.representation_reporting import (report_asteroid, report_campaign,
                                                   verify_campaign_artifacts, write_json)
from hac_asteroids.representation_study import CALIBRATIONS, fits


class L4SurveyTests(unittest.TestCase):
    def test_identical_spheres_and_exact_budget_for_all_ten(self):
        templates = []
        for a in ASTEROIDS:
            config = config_for(a)
            self.assertEqual((config.steps, config.faces, config.lmax, config.representation), (200, 1024, 4, "octantoid"))
            self.assertFalse(config.early_stopping)
            self.assertEqual(config.calibration_transfer, a > 3)
            self.assertEqual((config.frame_count, config.validation_frame_count, config.image_size), (72, 72, 96))
            self.assertFalse(set(config.train_indices) & set(config.validation_indices))
            stages = [optimization_state(config, i).stage for i in range(200)]
            self.assertEqual([stages.count(s) for s in ("escape", "refinement", "stabilization")], [40, 120, 40])
            shape, _ = _build_model(config, torch.device("cpu"))
            self.assertEqual(shape.n_parameters, 74)
            self.assertEqual(torch.count_nonzero(shape.initial_parameters()), 0)
            templates.append((shape.faces.numpy(), shape.mesh(shape.initial_parameters()).vertices.numpy()))
        for faces, vertices in templates:
            np.testing.assert_array_equal(faces, templates[0][0])
            np.testing.assert_array_equal(vertices, templates[0][1])

    def test_preparation_preserves_transfer_scope_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibrations, configs = prepare(root)
            self.assertEqual(len(configs), 10)
            for a in range(4, 11):
                self.assertEqual(calibrations[a], calibrations[3])
                with self.assertRaises(ValueError):
                    calibrations[a].validate_scope(a)
                calibrations[a].validate_scope(a, transfer=True)
                self.assertEqual((root / f"calibration/asteroid{a:02d}/calibration.json").read_bytes(),
                                 (CALIBRATIONS / "asteroid03/calibration.json").read_bytes())
            original = (root / "frozen_configuration.json").read_bytes()
            prepare(root)
            self.assertEqual((root / "frozen_configuration.json").read_bytes(), original)
            with tarfile.open(root / "source_snapshot.tar.gz") as bundle:
                self.assertFalse(any(Path(name).name == ".env" for name in bundle.getnames()))
            with patch("hac_asteroids.l4_survey.config_for", side_effect=lambda a: replace(config_for(a), steps=201)):
                with self.assertRaisesRegex(ValueError, "frozen survey input changed"):
                    prepare(root)
            self.assertEqual((root / "frozen_configuration.json").read_bytes(), original)

    def test_empty_report_labels_hidden_calibration_without_broken_artifact_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "study.json", dict(profile="l4_survey", asteroids=list(ASTEROIDS), steps=200))
            self.assertEqual(report_campaign(root), [])
            text = (root / "report.md").read_text()
            self.assertIn("**0/10**", text)
            self.assertIn("provisionally reuse asteroid 3", text)
            self.assertNotIn("shapes_overview.png", text)
            self.assertNotIn("status.json", text)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class L4SurveyCudaTests(unittest.TestCase):
    def test_hidden_body_resume_report_and_corrupt_gif_recovery(self):
        torch.set_num_threads(4)
        config = replace(config_for(4), steps=3, escape_steps=1, stabilization_steps=1,
                         validation_interval=1, checkpoint_interval=1)
        calibration = Calibration.load(CALIBRATIONS / "asteroid03/calibration.json")
        retained = os.environ.get("HAC_L4_SMOKE_DIR")
        with (nullcontext(retained) if retained else tempfile.TemporaryDirectory()) as directory:
            root = Path(directory)
            run = root / "asteroid04/octantoid_l4"
            write_json(root / "study.json", dict(profile="l4_survey", asteroids=[4], steps=3,
                       escape_steps=1, stabilization_steps=1))
            write_json(root / "frozen_configuration.json", {"configs": {"4/octantoid_l4": asdict(config)}})
            shape, _ = _build_model(config, torch.device("cpu"))
            np.savez(root / "sphere_1024.npz", faces=shape.faces.numpy(),
                     vertices=shape.mesh(shape.initial_parameters()).vertices.numpy())
            with patch("hac_asteroids.representation_reporting.public_mesh_path", side_effect=AssertionError("hidden reference")), \
                 patch("hac_asteroids.l4_survey_reporting.load_benchmark_mesh", side_effect=AssertionError("hidden proxy")):
                paused = run_inversion(config, calibration, run, stop_after_steps=2)
                self.assertEqual(paused["status"], "paused")
                final = run_inversion(config, calibration, run, resume=run / "checkpoint.pt")
                self.assertEqual(final["iterations_run"], 3)
                report_asteroid(root, 4, [config], {4: calibration})
                metrics = json.loads((run / "comparison_metrics.json").read_text())
                self.assertEqual(metrics["geometry"], {"initial": None, "selected": None, "last": None})
                self.assertTrue(metrics["audit"]["passed"])
                self.assertEqual(len(report_campaign(root)), 1)
                result = verify_campaign_artifacts(root)
                self.assertEqual((result["audited_runs"], len(result["decoded_gifs"])), (1, 1))
                gif = run / "full_history.gif"
                modified = gif.stat().st_mtime_ns
                with patch("hac_asteroids.representation_study.run_inversion", side_effect=AssertionError("completed fit rerun")), \
                     patch("hac_asteroids.l4_survey_reporting.measure_run", side_effect=AssertionError("completed report rerun")):
                    self.assertEqual(fits(root, {4: calibration}, {"4/octantoid_l4": config}, "l4_survey"), [])
                self.assertEqual(gif.stat().st_mtime_ns, modified)
                gif.write_bytes(b"interrupted GIF")
                with self.assertRaisesRegex(RuntimeError, "changed report artifact"):
                    verify_body(run.parent, config)
                report_asteroid(root, 4, [config], {4: calibration})
                self.assertTrue(verify_body(run.parent, config))


if __name__ == "__main__":
    unittest.main()
