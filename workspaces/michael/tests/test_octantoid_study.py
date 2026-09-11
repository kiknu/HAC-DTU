"""Budgeted octantoid campaigns and resumable validation-plateau stopping."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hac_asteroids.animation import animate_comparison
from hac_asteroids.calibration import Calibration
from hac_asteroids.datasets import load_lightcurves
from hac_asteroids.inversion import FitConfig, PlateauState, _build_model, run_inversion
from hac_asteroids.representation_reporting import audit_run, report_campaign, variants_for
from hac_asteroids.representation_study import (
    benchmark, choose_iteration_cap, config_for, prepare, register_profile,
)
from hac_asteroids.trajectory import FitTrajectory


class PlateauTests(unittest.TestCase):
    def setUp(self):
        self.config = config_for(3, "octantoid", 6, profile="octantoid", steps=400)

    def test_minimum_patience_and_confirmation(self):
        state = PlateauState()
        for step in range(20, 201, 20):
            event = state.observe(self.config, step, .1, .002)
            self.assertIsNone(state.stop_reason)
        self.assertEqual(event, "confirmation_started")
        self.assertEqual(state.confirmation_start, 200)
        self.assertAlmostEqual(state.learning_rate(.002), .0004)
        self.assertAlmostEqual(state.learning_rate(.00001), .00001)
        self.assertIsNone(state.observe(self.config, 220, .1, .0004))
        self.assertEqual(state.observe(self.config, 240, .1, .0004), "validation_plateau")

    def test_cumulative_small_improvements_and_absolute_floor(self):
        state = PlateauState()
        state.observe(self.config, 40, .1, .002)
        state.observe(self.config, 60, .0998, .002)
        self.assertEqual(state.reference_step, 40)
        self.assertEqual(state.observe(self.config, 80, .0994, .002), "meaningful_improvement")
        self.assertEqual(state.reference_step, 80)
        low = PlateauState(reference_rmse=.001, reference_step=40)
        low.observe(self.config, 60, .00095, .002)
        self.assertEqual(low.reference_step, 40)
        low.observe(self.config, 80, .00089, .002)
        self.assertEqual(low.reference_step, 80)

    def test_recovery_retains_reduced_rate_and_resets_patience(self):
        state = PlateauState(reference_rmse=.1, reference_step=40)
        state.observe(self.config, 200, .1, .002)
        self.assertEqual(state.observe(self.config, 220, .09, .0004), "confirmation_recovered")
        self.assertIsNone(state.confirmation_start)
        self.assertEqual(state.reference_step, 220)
        self.assertAlmostEqual(state.learning_rate(.002), .0004)
        self.assertIsNone(state.observe(self.config, 300, .09, .0003))
        self.assertEqual(state.observe(self.config, 320, .09, .0003), "confirmation_started")

    def test_cap_precedence_and_disabled_behavior(self):
        config = replace(self.config, steps=240)
        state = PlateauState(reference_rmse=.1, reference_step=40, confirmation_start=200)
        self.assertEqual(state.observe(config, 240, .1, .0004), "iteration_cap")
        disabled = PlateauState()
        disabled.observe(replace(config, early_stopping=False), 240, .1, .002)
        self.assertIsNone(disabled.reference_rmse)
        self.assertIsNone(disabled.stop_reason)

    def test_state_roundtrip_inside_confirmation(self):
        whole = PlateauState(reference_rmse=.1, reference_step=40)
        whole.observe(self.config, 200, .1, .002)
        resumed = PlateauState(**json.loads(json.dumps(asdict(whole))))
        for step, score in [(220,.09),(240,.09),(320,.09),(340,.09),(360,.09)]:
            self.assertEqual(whole.observe(self.config, step, score, whole.learning_rate(.002)),
                             resumed.observe(self.config, step, score, resumed.learning_rate(.002)))
        self.assertEqual(asdict(whole), asdict(resumed))


class OctantoidStudyTests(unittest.TestCase):
    def test_models_have_shared_initial_sphere_and_controls(self):
        meshes = []
        for degree, count in ((2,26),(4,74),(6,146)):
            config = config_for(1,"octantoid",degree,profile="octantoid",steps=380)
            shape,_ = _build_model(config,torch.device("cpu"))
            meshes.append(shape.mesh(shape.initial_parameters()))
            self.assertEqual(shape.n_parameters,count)
            self.assertEqual((config.faces,config.steps,config.regularization_denominator), (1024,380,146))
            self.assertEqual((config.escape_steps,config.stabilization_steps,config.validation_interval),(40,40,20))
            self.assertTrue(config.early_stopping)
            self.assertEqual((config.frame_count,config.validation_frame_count,config.image_size),(72,72,96))
        for mesh in meshes[1:]:
            torch.testing.assert_close(mesh.vertices,meshes[0].vertices,atol=0,rtol=0)
            torch.testing.assert_close(mesh.faces,meshes[0].faces)
        with self.assertRaises(ValueError):
            config_for(3,"radial_sh",6,profile="octantoid")

    def test_budget_selection_accounts_for_all_costs(self):
        rows = [dict(mean_step_seconds=6.5,full_turn_seconds=11.)]*9
        chosen,projections = choose_iteration_cap(rows,300.)
        self.assertEqual(chosen["steps"],360)
        self.assertLessEqual(chosen["total_seconds"],8*3600)
        self.assertGreater(next(r for r in projections if r["steps"] == 380)["total_seconds"],8*3600)
        self.assertIsNone(choose_iteration_cap([dict(mean_step_seconds=20.,full_turn_seconds=20.)]*9)[0])

    def test_frozen_profile_cap_and_legacy_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            register_profile(root,"octantoid")
            description = json.loads((root/"study.json").read_text())
            description["selected_steps"] = 380
            (root/"study.json").write_text(json.dumps(description))
            configs = {f"{a}/{key}":asdict(config_for(a,rep,degree,profile="octantoid",steps=380))
                       for a in (1,2,3) for key,_,rep,degree in variants_for("octantoid")}
            (root/"frozen_configuration.json").write_text(json.dumps(dict(configs=configs)))
            register_profile(root,"octantoid")
            with self.assertRaisesRegex(ValueError,"profile changed"):
                register_profile(root,"quick")
            description["selected_steps"] = 400
            (root/"study.json").write_text(json.dumps(description))
            with self.assertRaisesRegex(ValueError,"frozen configurations"):
                register_profile(root,"octantoid")

    def test_calibrations_are_reused_without_screening(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.cuda.is_available",return_value=True), \
                 patch("hac_asteroids.representation_study.calibrate_public") as calibrate:
                calibrations = prepare(Path(directory),"octantoid")
            self.assertEqual(set(calibrations),{1,2,3})
            calibrate.assert_not_called()

    def test_benchmark_all_nine_and_cached_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            register_profile(root,"octantoid")
            calibrations={a:Calibration(asteroids=(a,)) for a in (1,2,3)}
            with patch("torch.cuda.get_device_name",return_value="test GPU"), \
                 patch("hac_asteroids.representation_study.benchmark_variant",
                       return_value=dict(mean_step_seconds=6.,full_turn_seconds=10.)) as run:
                self.assertEqual(benchmark(root,calibrations,"octantoid"),6)
                self.assertEqual(run.call_count,9)
                self.assertEqual({(c.args[0].asteroid,c.args[0].lmax) for c in run.call_args_list},
                                 {(a,l) for a in (1,2,3) for l in (2,4,6)})
                self.assertEqual(benchmark(root,calibrations,"octantoid"),6)
                self.assertEqual(run.call_count,9)
            saved=json.loads((root/"benchmark.json").read_text())
            self.assertEqual(saved["selected_steps"],400)
            report_campaign(root)
            report=(root/"report.md").read_text()
            self.assertIn("**0/9**",report)
            self.assertIn("up to 400",report)
            self.assertNotIn("Radial L",report)

    def test_unequal_histories_hold_their_terminal_frame(self):
        from PIL import Image
        shape,_ = _build_model(replace(FitConfig.smoke(3),initialization="sphere"),torch.device("cpu"))
        mesh=shape.mesh(shape.initial_parameters())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sources=[]
            for i,length in enumerate((2,4,6)):
                vertices=np.stack([mesh.vertices.numpy()*(1+.1*j) for j in range(length)])
                sources.append(FitTrajectory(np.zeros((length,1)),vertices,mesh.faces.numpy(),
                                              np.ones(length),best_index=1).save(root/f"{i}.npz"))
            animate_comparison(sources,["L2","L4","L6"],root/"unequal.gif",stride=2,
                               stop_reasons=["validation_plateau","validation_plateau","iteration_cap"])
            with Image.open(root/"unequal.gif") as gif:
                self.assertEqual(gif.n_frames,6)
                for frame in range(gif.n_frames):
                    gif.seek(frame)
                    gif.load()


@unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
class OctantoidCudaTests(unittest.TestCase):
    def test_degrees_and_plateau_resume(self):
        data=load_lightcurves(3).select(np.arange(0,360,30))
        calibration=Calibration(asteroids=(3,),response="srgb")
        base=replace(FitConfig.smoke(3,"cuda"),steps=3,faces=1024,initialization="sphere",
                     evaluation_stride=1,regularization_denominator=146)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch("hac_asteroids.inversion.load_lightcurves",return_value=data), \
                 patch("hac_asteroids.inversion.shape_report"), \
                 patch("hac_asteroids.inversion.align_and_score_public_meshes",side_effect=lambda a,b,f,r:(a,b,{})):
                for degree in (2,4,6):
                    result=run_inversion(replace(base,lmax=degree),calibration,root/f"l{degree}")
                    self.assertEqual(result["iterations_run"],3)
                    self.assertTrue(np.isfinite(result["pooled_rmse"]))
                config=replace(base,steps=5,early_stopping=True,plateau_min_steps=2,
                               plateau_patience=1,plateau_confirmation_steps=1)
                with patch("hac_asteroids.inversion._validation_rmse",return_value=.1):
                    result=run_inversion(config,calibration,root/"whole")
                    self.assertEqual((result["iterations_run"],result["stop_reason"]),(3,"validation_plateau"))
                    run_inversion(config,calibration,root/"split",stop_after_steps=2)
                    checkpoint=root/"split/checkpoint.pt"
                    with self.assertRaisesRegex(ValueError,"fingerprint"):
                        run_inversion(replace(config,plateau_patience=2),calibration,root/"reject",resume=checkpoint)
                    resumed=run_inversion(config,calibration,root/"split",resume=checkpoint)
                    self.assertEqual(resumed["iterations_run"],3)
                    # A terminal checkpoint must finalize without further optimization.
                    again=run_inversion(config,calibration,root/"split",resume=checkpoint)
                    self.assertEqual(again["iterations_run"],3)
                self.assertTrue(audit_run(root/"split",config)["passed"])
                whole=torch.load(root/"whole/checkpoint.pt",weights_only=False)
                split=torch.load(checkpoint,weights_only=False)
                self.assertEqual(whole["plateau_state"],split["plateau_state"])
                torch.testing.assert_close(whole["parameters"],split["parameters"],atol=2e-5,rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
