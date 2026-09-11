"""Controlled representation comparisons and their integration with the fitter."""
from dataclasses import replace
from pathlib import Path
import json
from dataclasses import asdict
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hac_asteroids.animation import animation_indices
from hac_asteroids.calibration import Calibration
from hac_asteroids.cli import _fit, build_parser
from hac_asteroids.datasets import load_lightcurves
from hac_asteroids.inversion import FitConfig, _build_model, run_inversion, spectral_penalty
from hac_asteroids.representation_reporting import geometry_metrics, audit_run
from hac_asteroids.trajectory import FitTrajectory
from shape_model import OctantoidShapeModel, RadialHarmonicShapeModel, sphere_template


class RepresentationTests(unittest.TestCase):
    def test_exact_spherical_templates(self):
        for count in (512, 1024, 2048, 4096):
            mesh = sphere_template(faces=count)
            self.assertEqual(len(mesh.faces), count)
            self.assertTrue(mesh.is_watertight and mesh.is_winding_consistent)
            self.assertEqual(mesh.euler_number, 2)
            np.testing.assert_allclose(np.linalg.norm(mesh.vertices, axis=1), 1, atol=1e-14)
        first, second = sphere_template(faces=1024), sphere_template(faces=1024)
        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.faces, second.faces)

    def test_radial_is_exact_octantoid_subfamily_with_matching_gradients_and_prior(self):
        template = sphere_template(faces=1024)
        directions = torch.tensor(template.vertices, dtype=torch.float64)
        faces = torch.tensor(template.faces)
        radial = RadialHarmonicShapeModel(directions, faces, lmax=6, degree_power=3)
        octantoid = OctantoidShapeModel(directions, faces, lmax=6, lam_shape=1, degree_power=3)
        torch.manual_seed(18)
        p = (torch.randn(48, dtype=torch.float64) * .03).requires_grad_()
        q = torch.cat((p.detach(), torch.zeros(98, dtype=torch.float64))).requires_grad_()
        a, b = radial.mesh(p).vertices, octantoid.mesh(q).vertices
        torch.testing.assert_close(a, b)
        weight = torch.randn_like(a)
        ga = torch.autograd.grad((a * weight).sum(), p)[0]
        gb = torch.autograd.grad((b * weight).sum(), q)[0]
        torch.testing.assert_close(ga, gb[:48])
        config = FitConfig(asteroid=1, regularization_denominator=146)
        ra = spectral_penalty(radial.prior_residuals(p), config)
        rb = spectral_penalty(octantoid.prior_residuals(q), config)
        torch.testing.assert_close(ra, rb)
        torch.testing.assert_close(torch.autograd.grad(ra, p)[0], torch.autograd.grad(rb, q)[0][:48])
        displacement = a - radial.mesh(radial.zero_parameters()).vertices
        tangential = displacement - (displacement * radial.directions).sum(1)[:, None] * radial.directions
        torch.testing.assert_close(tangential, torch.zeros_like(tangential), atol=1e-14, rtol=0)

    def test_config_factory_and_cli_preserve_comparison(self):
        for rep, degree, count in (("radial_sh", 6, 48), ("radial_sh", 11, 143), ("octantoid", 6, 146)):
            config = FitConfig(asteroid=1, faces=1024, representation=rep, lmax=degree, initialization="sphere")
            shape, model = _build_model(config, torch.device("cpu"))
            self.assertEqual(shape.n_parameters, count)
            self.assertEqual(model.renderer.config.face_limit, 1024)
            mesh = shape.mesh(shape.initial_parameters())
            torch.testing.assert_close(mesh.vertices.norm(dim=1), torch.full((514,), .5))
        for kwargs in ({"faces":1023}, {"regularization_denominator":0},
                       {"initialization":"sphere", "initialization_noise":.01}):
            with self.assertRaises(ValueError):
                FitConfig(asteroid=1, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            cal = Calibration(asteroids=(1,)).save(Path(directory)/"calibration.json")
            for smoke in (False, True):
                argv = ["fit", "--asteroids", "1", "--calibration", str(cal), "--faces", "1024",
                        "--representation", "radial_sh", "--lmax", "11", "--initialization", "sphere",
                        "--regularization-denominator", "146", "--output-dir", directory]
                if smoke:
                    argv.append("--smoke")
                with patch("hac_asteroids.inversion.run_inversion", return_value={}) as run:
                    _fit(build_parser().parse_args(argv))
                actual = run.call_args.args[0]
                self.assertEqual((actual.faces, actual.representation, actual.lmax, actual.initialization,
                                  actual.regularization_denominator), (1024, "radial_sh", 11, "sphere", 146))

    def test_animation_retains_actual_selected_and_last_iterations(self):
        np.testing.assert_array_equal(animation_indices(12, 5, 7), [0, 5, 7, 10, 11])
        np.testing.assert_array_equal(animation_indices(8, 5, 7), [0, 5, 7])
        np.testing.assert_array_equal(animation_indices(1, 5, 0), [0])

    def test_geometry_scale_and_missing_rays_are_explicit(self):
        mesh = sphere_template(subdivisions=1)
        result = geometry_metrics(mesh.vertices * .5, mesh.faces, mesh, count=64)
        self.assertAlmostEqual(result["scale"], 2.)
        self.assertLess(result["radial_log_rms"], 1e-6)
        with patch("hac_asteroids.representation_reporting.outer_radii", return_value=np.zeros(64)):
            result = geometry_metrics(mesh.vertices, mesh.faces, mesh, count=64)
        self.assertEqual(result["missed_rays"], 64)
        self.assertIsNone(result["radial_log_rms"])

    def test_comparison_gifs_decode_and_include_history_endpoints(self):
        from PIL import Image
        from hac_asteroids.animation import animate_trajectory, animate_comparison
        mesh = sphere_template(subdivisions=1)
        initial = mesh.vertices.astype(np.float32) * .5
        vertices = np.stack([initial, initial * [1,1.3,1], initial * [1,1.3,.8]])
        trajectory = FitTrajectory(np.zeros((3,1)), vertices, mesh.faces, np.array([1.,.8,.9]), best_index=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = trajectory.save(root / "trajectory.npz")
            animate_trajectory(source, root / "selected.gif", stride=5)
            animate_trajectory(source, root / "full.gif", stride=5, all_iterations=True)
            animate_comparison([source]*3, ["Radial L6", "Radial L11", "Octantoid L6"], root / "comparison.gif")
            for name, count in (("selected.gif",2), ("full.gif",3), ("comparison.gif",3)):
                with Image.open(root / name) as gif:
                    self.assertEqual(gif.n_frames, count)
                    gif.seek(count-1)
                    gif.load()

    def test_audit_checks_exported_selected_state(self):
        import trimesh
        config = replace(FitConfig.smoke(2), faces=1024, representation="radial_sh", lmax=6,
                         initialization="sphere", steps=3)
        shape, _ = _build_model(config, torch.device("cpu"))
        parameters = np.zeros((4,48), dtype=np.float32)
        parameters[1:, 1] = [.01,.02,.03]
        vertices = np.stack([shape.mesh(torch.tensor(p)).vertices.numpy() for p in parameters])
        trajectory = FitTrajectory(parameters, vertices, shape.faces.numpy(), np.ones(4), best_index=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory.save(root/"trajectory.npz")
            checkpoint = dict(step=3, parameters=torch.tensor(parameters[-1]),
                              best=dict(history_index=2,parameters=torch.tensor(parameters[2])),
                              optimizer=dict(state={0:dict(step=torch.tensor(3.),exp_avg=torch.zeros(48))}))
            torch.save(checkpoint, root/"checkpoint.pt")
            (root/"run.json").write_text(json.dumps(dict(config=asdict(config),
                                           summary=dict(iterations_run=3,best_iteration=2))))
            trimesh.Trimesh(vertices[2], trajectory.faces, process=False).export(root/"mesh_final.stl")
            self.assertTrue(audit_run(root,config)["passed"])
            trimesh.Trimesh(vertices[-1], trajectory.faces, process=False).export(root/"mesh_final.stl")
            with self.assertRaisesRegex(RuntimeError,"audit failed"):
                audit_run(root,config)

    def test_benchmark_identity_survives_json_tuple_conversion(self):
        from hac_asteroids.provenance import code_identity, file_sha256
        from hac_asteroids.representation_study import benchmark, HISTORICAL_RECIPE
        calibration = Calibration(asteroids=(2,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = dict(code=code_identity(), calibration=asdict(calibration),
                            recipe=file_sha256(HISTORICAL_RECIPE), gpu="test GPU")
            (root/"benchmark.json").write_text(json.dumps(dict(identity=identity, block_frames=6)))
            with patch("torch.cuda.get_device_name", return_value="test GPU"), \
                 patch("hac_asteroids.representation_study.benchmark_variant") as run:
                self.assertEqual(benchmark(root,{2:calibration}),6)
                run.assert_not_called()


class ReplayTests(unittest.TestCase):
    def test_curve_replay_allows_float32_scatter_but_rejects_changed_predictions(self):
        from hac_asteroids.representation_reporting import verify_curve_replay
        saved = np.ones((360,21), dtype=np.float32)
        replay = saved.copy()
        replay[1,1] += 6e-5
        result = verify_curve_replay(saved,replay)
        self.assertLess(result["rms_difference"], 1e-6)
        replay[1,1] += .01
        with self.assertRaisesRegex(RuntimeError,"curves disagree"):
            verify_curve_replay(saved,replay)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RepresentationCudaTests(unittest.TestCase):
    def test_all_representations_fit_and_radial_resume_rejects_changes(self):
        data = load_lightcurves(2).select(np.arange(0, 360, 30))
        calibration = Calibration(asteroids=(2,), response="srgb")
        common = replace(FitConfig.smoke(2, "cuda"), steps=3, faces=1024,
                         initialization="sphere", evaluation_stride=1,
                         regularization_denominator=146)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("hac_asteroids.inversion.load_lightcurves", return_value=data), \
                 patch("hac_asteroids.inversion.shape_report"), \
                 patch("hac_asteroids.inversion.align_and_score_public_meshes", side_effect=lambda a,b,f,r:(a,b,{})):
                for rep, degree in (("radial_sh", 6), ("radial_sh", 11), ("octantoid", 6)):
                    config = replace(common, representation=rep, lmax=degree)
                    result = run_inversion(config, calibration, root/f"{rep}{degree}")
                    self.assertEqual(result["iterations_run"], 3)
                    self.assertTrue(np.isfinite(result["pooled_rmse"]))
                config = replace(common, representation="radial_sh", lmax=11)
                run_inversion(config, calibration, root/"split", stop_after_steps=1)
                checkpoint = root/"split/checkpoint.pt"
                for change in ({"lmax":6}, {"representation":"octantoid"},
                               {"regularization_denominator":143}, {"faces":512}):
                    with self.assertRaisesRegex(ValueError, "fingerprint"):
                        run_inversion(replace(config, **change), calibration, root/"reject", resume=checkpoint)
                run_inversion(config, calibration, root/"split", resume=checkpoint)
            whole = torch.load(root/"radial_sh11/checkpoint.pt", weights_only=False)
            split = torch.load(checkpoint, weights_only=False)
            torch.testing.assert_close(whole["parameters"], split["parameters"], atol=2e-5, rtol=1e-4)
            self.assertEqual(whole["best"]["step"], split["best"]["step"])


if __name__ == "__main__":
    unittest.main()
