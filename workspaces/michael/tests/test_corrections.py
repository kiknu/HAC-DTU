"""Regression tests for the intensity correction and normalized gradient contract."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
torch.set_num_threads(4)

from hac_asteroids.calibration import Calibration, phase_degrees, curve_metrics
from hac_asteroids.datasets import load_lightcurves
from hac_asteroids.inversion import (FitConfig, _build_model, _evaluate, _initial_shape,
                                    rotation_split, run_inversion, two_pass_backward)
from hac_asteroids.provenance import file_sha256, fingerprint
from pytorch3d_forward import image_response, extract_lightcurves, ObservationBatch


class CorrectionTests(unittest.TestCase):
    def test_known_orthographic_curve_fixture_and_conditional_subset_score(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/orthographic_response.json').read_text())
        body=fixture['bodies']['2']
        observed=np.array(body['observed'])[:,None]
        scores={name:curve_metrics(np.array(body[name])[:,None],observed,('225_bottom',))['pooled_rmse']
                for name in ('linear','srgb')}
        self.assertAlmostEqual(scores['linear'],.4351181,places=6)
        self.assertAlmostEqual(scores['srgb'],.1366035,places=6)
        indices=np.arange(2,360,5)
        pred=np.array(body['srgb'])[indices,None]
        score=curve_metrics(pred,observed[indices],('225_bottom',))['pooled_rmse']
        self.assertAlmostEqual(score,float(np.sqrt(np.mean((pred-observed[indices])**2))))

    def test_response_breakpoint_clipping_and_finite_gradients(self):
        x = torch.tensor([-.01, 0., .0031308, .01, 1., 2.], dtype=torch.float64, requires_grad=True)
        encoded = image_response(x, "srgb")
        expected = [0., 0., .040449936, .0998528227, 1., 1.]
        torch.testing.assert_close(encoded, torch.tensor(expected, dtype=torch.float64), atol=1e-8, rtol=1e-7)
        encoded.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(float(x.grad[1]), 12.92)
        for gain in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                image_response(x, gain=gain)

    def test_post_response_threshold_and_soft_limit(self):
        linear = torch.tensor([.02, .04, .2, .8]).reshape(2, 1, 2)
        srgb = image_response(linear, 'srgb')
        exact = extract_lightcurves(srgb, 2, 1, mode='exact', compute_binary=False)
        self.assertGreater(float(exact.raw_intensity[0]), 0.)
        self.assertEqual(float(extract_lightcurves(linear, 2, 1, mode='exact', compute_binary=False).raw_intensity[0]), 0.)
        soft = extract_lightcurves(srgb, 2, 1, mode='soft', intensity_slope=10000, compute_binary=False)
        torch.testing.assert_close(soft.raw_intensity, exact.raw_intensity, atol=2/255, rtol=0)

    def test_phase_and_explicit_split(self):
        data = load_lightcurves(2)
        np.testing.assert_array_equal(phase_degrees(data, [0, 359]), [1, 360])
        np.testing.assert_array_equal(phase_degrees(data, [0, 359], 'zero_based'), [0, 359])
        for count in (12, 72):
            train, val = rotation_split(360, count, 72)
            np.testing.assert_array_equal(val, np.arange(2, 360, 5))
            self.assertEqual(len(np.intersect1d(train, val)), 0)
        np.testing.assert_array_equal(rotation_split(360, 72, 72)[0], np.arange(0, 360, 5))

    def test_legacy_serialization_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration.json'
            path.write_text(json.dumps(dict(camera_distance_radii=16., lambertian_fraction=.5)))
            with self.assertWarns(UserWarning):
                legacy = Calibration.load(path)
            self.assertEqual((legacy.response, legacy.radiance_gain, legacy.phase_convention), ('linear', 1., 'frame_label'))
            with self.assertRaisesRegex(ValueError, 'calibration transfer'):
                legacy.validate_scope(8)
            legacy.validate_scope(8, transfer=True)
            replace(legacy, asteroids=(2, 3)).validate_scope(2)
            legacy.save(path)
            with self.assertWarns(UserWarning):
                self.assertEqual(Calibration.load(path), legacy)
            new = Calibration(asteroids=(2,), response='srgb', projection='orthographic', phase_convention='zero_based')
            new.save(path)
            self.assertEqual(Calibration.load(path), new)
            with self.assertRaises(ValueError):
                new.validate_scope(3)
            new.validate_scope(3, transfer=True)

    def test_cache_identity_uses_contents_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data'
            path.write_bytes(b'abcd')
            first = file_sha256(path)
            path.write_bytes(b'abce')
            self.assertNotEqual(first, file_sha256(path))
            self.assertNotEqual(fingerprint(dict(response='linear')), fingerprint(dict(response='srgb')))

    def test_same_size_mesh_edit_invalidates_cached_proxy(self):
        import trimesh
        from forward_validation import load_benchmark_mesh
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.stl'
            trimesh.creation.box(extents=(1.,1.,1.)).export(source)
            size=source.stat().st_size
            with patch('forward_validation.get_model_stl_path',return_value=source):
                first=load_benchmark_mesh(2,80,torch.device('cpu'),root/'cache')
                trimesh.creation.box(extents=(2.,1.,1.)).export(source)
                self.assertEqual(source.stat().st_size,size)
                second=load_benchmark_mesh(2,80,torch.device('cpu'),root/'cache')
            self.assertGreater(float(second.vertices[:,0].max()),1.9*float(first.vertices[:,0].max()))

    def test_two_pass_exact_mean_derivative_and_chunk_invariance(self):
        torch.manual_seed(5)
        features = torch.randn(11, 3, 4, dtype=torch.float64)
        target = torch.rand(11, 3, dtype=torch.float64)
        initial = torch.randn(4, dtype=torch.float64)
        p = initial.clone().requires_grad_(True)
        raw = (features @ p).exp() + .1
        loss = ((raw/raw.mean(0)-target)**2).mean()
        loss.backward()
        expected = p.grad.clone()
        for size in (1, 3, 11):
            q = initial.clone().requires_grad_(True)
            blocks = [slice(i, min(i+size, 11)) for i in range(0, 11, size)]
            observed = two_pass_backward(lambda b: (features[b] @ q).exp()+.1, blocks, target)
            torch.testing.assert_close(observed, loss)
            torch.testing.assert_close(q.grad, expected, atol=1e-12, rtol=1e-12)

    def test_evaluation_normalizes_full_turn_before_selecting(self):
        data = load_lightcurves(2)
        class Model:
            def __call__(self, p, observations, **kwargs):
                from types import SimpleNamespace
                raw = 2 + torch.sin(observations.angles)[:, None].expand(-1, 21)
                return SimpleNamespace(raw_intensity=raw)
        _, full = _evaluate(Model(), torch.zeros(1), data, Calibration(), block_frames=17)
        selected, values = _evaluate(Model(), torch.zeros(1), data, Calibration(), indices=np.array([0, 90]), block_frames=11)
        np.testing.assert_allclose(values, full[selected], atol=1e-6)
        self.assertGreater(abs(values.mean()-1), .1)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class CudaCorrectionTests(unittest.TestCase):
    def test_orthographic_response_has_finite_nonzero_shape_gradient(self):
        config=FitConfig.smoke(3,'cuda')
        shape,model=_build_model(config,torch.device('cuda'),Calibration(asteroids=(3,),projection='orthographic',response='srgb'))
        p=_initial_shape(shape,load_lightcurves(3),np.arange(0,360,30)).clone().requires_grad_(True)
        observations=ObservationBatch.challenge(torch.tensor([.123,.678,1.234],device='cuda'),('0_top','225_bottom'))
        prediction=model(p,observations,compute_binary=False).intensity
        (prediction-torch.tensor([[.8],[1.1],[1.1]],device='cuda')).square().mean().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.norm()),1e-5)

    def test_shadow_depth_occludes_points_behind_a_plane(self):
        from pytorch3d_forward import PyTorch3DLightcurveModel, RendererConfig
        renderer=PyTorch3DLightcurveModel(RendererConfig.optimization()).cuda()
        radius=torch.tensor(1.,device='cuda')
        camera=renderer._light_cameras(radius.new_tensor([-1.,0.,0.]),1,radius)
        depth=torch.full((1,32,32),3.,device='cuda')
        points=radius.new_tensor([[-1.,0.,0.],[0.,0.,0.]]).reshape(1,1,2,1,3).requires_grad_(True)
        lit=renderer._sample_shadows(points,camera,depth,torch.tensor([0],device='cuda'),radius)
        self.assertAlmostEqual(float(lit.flatten()[0]),float(torch.sigmoid(torch.tensor(1.))),places=4)
        self.assertLess(float(lit.flatten()[1]),1e-6)
        lit.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_resolved_sphere_rotation_is_nearly_constant(self):
        import trimesh
        from pytorch3d_forward import PyTorch3DLightcurveModel, RendererConfig
        from shape_model import TriangleMesh
        sphere=trimesh.creation.icosphere(subdivisions=3)
        mesh=TriangleMesh(torch.tensor(sphere.vertices,device='cuda',dtype=torch.float32),
                          torch.tensor(sphere.faces,device='cuda',dtype=torch.long))
        renderer=PyTorch3DLightcurveModel(replace(RendererConfig.optimization(),batch_size=4,response='srgb')).cuda()
        obs=ObservationBatch.challenge(torch.tensor([.1,.6,1.2,1.7,2.3],device='cuda'),('0_top','225_bottom'))
        with torch.no_grad():
            pred=renderer(mesh,obs,extraction='exact',compute_binary=False).intensity
        self.assertLess(float((pred-1).abs().max()),.03)

    def test_fixed_camera_shadow_depth_is_batch_invariant(self):
        from pytorch3d_forward import PyTorch3DLightcurveModel
        torch.set_float32_matmul_precision('high')
        config = FitConfig.smoke(2, 'cuda')
        shape, model = _build_model(config, torch.device('cuda'), Calibration(asteroids=(2,)))
        self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
        p = _initial_shape(shape, load_lightcurves(2), np.arange(0,360,30))
        mesh = shape.mesh(p)
        radius = mesh.vertices.norm(dim=-1).max()
        angles = torch.tensor([.123,.789,1.234], device='cuda')
        depths = []
        with torch.no_grad():
            for count in (1,3):
                meshes, _ = model.renderer._meshes_for_frames(mesh, angles[:count], 1)
                cameras = model.renderer._light_cameras(angles.new_tensor([-1.,0.,0.]), count, radius)
                depths.append(model.renderer._shadow_maps(meshes, cameras)[0])
        torch.testing.assert_close(depths[0], depths[1], atol=2e-6, rtol=2e-6)

    def test_renderer_chunk_gradient_and_projection_limit(self):
        config = replace(FitConfig.smoke(2, 'cuda'), lmax=2)
        cal = Calibration(asteroids=(2,), response='srgb')
        shape, model = _build_model(config, torch.device('cuda'), cal)
        p0 = _initial_shape(shape, load_lightcurves(2), np.arange(0, 360, 30))
        angles = torch.tensor([.123, .789, 1.234], device='cuda')
        target = torch.tensor([[.7, .9], [1., 1.2], [1.3, .9]], device='cuda')
        def raw(p, block):
            return model(p, ObservationBatch.challenge(angles[block], ('0_top', '225_bottom')), compute_binary=False).raw_intensity
        p = p0.clone().requires_grad_(True)
        brightness = raw(p, slice(None))
        loss = ((brightness/brightness.mean(0)-target)**2).mean()
        loss.backward()
        self.assertGreater(float(p.grad.norm()), 0)
        expected = p.grad.clone()
        for size in (1, 2):
            q = p0.clone().requires_grad_(True)
            two_pass_backward(lambda b: raw(q, b), [slice(i, i+size) for i in range(0, 3, size)], target)
            torch.testing.assert_close(q.grad, expected, atol=5e-5, rtol=2e-3)
        with torch.no_grad():
            # A 32-pixel crescent has only a few lit pixels; use a resolved
            # image for the projection limit, independently of the tiny gradient test.
            model.renderer.config = replace(model.renderer.config, image_size=96, shadow_size=96, batch_size=16)
            obs = ObservationBatch.challenge(angles, ('0_top', '225_bottom'))
            perspective = model(p0, obs, camera_distance_radii=1000., compute_binary=False).intensity
            model.renderer.config = replace(model.renderer.config, projection='orthographic')
            # Hold depth blending's near/far interval fixed as well as framing.
            ortho = model(p0, obs, camera_distance_radii=1000., compute_binary=False).intensity
            torch.testing.assert_close(perspective, ortho, atol=.015, rtol=.015)

    def test_checkpoint_resume_and_changed_objective_rejection(self):
        config = replace(FitConfig.smoke(2, 'cuda'), steps=3, evaluation_stride=1,
                         initialization_noise=.01, seed=17)
        calibration = Calibration(asteroids=(2,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Reporting/public shape alignment is unrelated to optimizer equivalence.
            small_data = load_lightcurves(2).select(np.arange(0, 360, 30))
            with patch('hac_asteroids.inversion.load_lightcurves', return_value=small_data), patch('hac_asteroids.inversion.shape_report'), patch('hac_asteroids.inversion.align_and_score_public_meshes', side_effect=lambda a,b,f,r:(a,b,{})):
                run_inversion(config, calibration, root/'whole')
                run_inversion(config, calibration, root/'split', stop_after_steps=1)
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    run_inversion(config, replace(calibration, response='srgb'), root/'reject', resume=root/'split/checkpoint.pt')
                run_inversion(config, calibration, root/'split', resume=root/'split/checkpoint.pt')
            whole = torch.load(root/'whole/checkpoint.pt', weights_only=False)
            split = torch.load(root/'split/checkpoint.pt', weights_only=False)
            # CUDA scatter reductions are not bitwise deterministic. This is
            # below 0.3% of one initial update, with identical Adam step counts.
            torch.testing.assert_close(whole['parameters'], split['parameters'], atol=2e-5, rtol=1e-4)
            self.assertEqual(whole['optimizer']['state'][0]['step'], split['optimizer']['state'][0]['step'])
            self.assertEqual(whole['best']['step'], split['best']['step'])
