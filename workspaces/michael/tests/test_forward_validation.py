"""Regression tests for public forward-model validation."""

import unittest

import numpy as np
import torch
import trimesh

from forward_model import (
    ObservationGeometry,
    PhotometricForwardModel,
    build_view_dirs,
    load_data,
)
from forward_validation import (
    UNIQUE_COLUMNS,
    curve_metrics,
    estimate_phase,
    normalize_challenge_mesh,
    normalized_curves,
    periodic_shift,
    phase_objective,
    rotate_mesh_about_z,
    simplify_closed_mesh,
    star_proxy_mesh,
)
from shape_model import TriangleMesh


class ForwardValidationTests(unittest.TestCase):
    def test_dataset_origin_is_strict_and_simulated_shape_is_known(self):
        with self.assertRaisesRegex(ValueError, "origin"):
            load_data(1, "blender", "intensity")
        dataset = load_data(1, "simulated", "intensity")
        self.assertEqual(dataset.data.shape, (360, 28))
        self.assertEqual(len(UNIQUE_COLUMNS), 21)
        self.assertFalse(any(column.endswith("hor2") for column in UNIQUE_COLUMNS))

    def test_mesh_normalization_uses_challenge_z_convention(self):
        source = trimesh.creation.box(extents=[4.0, 6.0, 8.0])
        source.apply_translation([2.0, -3.0, 7.0])
        normalized = normalize_challenge_mesh(source)
        np.testing.assert_allclose(normalized.bounds[:, 2], [-1.0, 1.0])
        np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0.0, atol=1e-12)
        self.assertTrue(normalized.is_watertight)
        self.assertTrue(normalized.is_winding_consistent)

    def test_mesh_decimation_preserves_closed_oriented_topology(self):
        source = trimesh.creation.icosphere(subdivisions=3)
        simplified = simplify_closed_mesh(source, 128)
        self.assertLessEqual(len(simplified.faces), 128)
        self.assertTrue(simplified.is_watertight)
        self.assertTrue(simplified.is_winding_consistent)
        self.assertGreater(simplified.volume, 0.0)

    def test_star_proxy_uses_optimizer_topology(self):
        source = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        proxy = star_proxy_mesh(source, target_faces=1024)
        self.assertEqual(len(proxy.faces), 1280)
        self.assertEqual(len(proxy.vertices), 642)
        self.assertTrue(proxy.is_watertight)
        self.assertTrue(proxy.is_winding_consistent)
        self.assertGreater(proxy.volume, 0.0)

    def test_periodic_phase_recovery_and_metrics(self):
        degrees = np.arange(0.0, 360.0, 6.0)
        base = 1.0 + 0.2 * np.cos(np.radians(degrees))
        observed = periodic_shift(base, degrees, 24.0)
        phase, rmse = estimate_phase(base, observed, degrees, step_degrees=1.0)
        self.assertAlmostEqual(phase, 24.0)
        self.assertLess(rmse, 1e-12)
        metrics = curve_metrics(periodic_shift(base, degrees, phase), observed)
        self.assertLess(metrics["rmse"], 1e-12)
        self.assertAlmostEqual(metrics["correlation"], 1.0)
        self.assertAlmostEqual(metrics["amplitude_ratio"], 1.0)

    def test_multicurve_phase_objective_uses_one_shared_shift(self):
        degrees = np.arange(0.0, 360.0, 6.0)
        predicted = np.column_stack(
            [
                1.0 + 0.2 * np.cos(np.radians(degrees)),
                1.0 + 0.1 * np.sin(2.0 * np.radians(degrees)),
            ]
        )
        observed = periodic_shift(predicted, degrees, -18.0)
        scores = {
            shift: phase_objective(predicted, observed, degrees, shift)
            for shift in range(-30, 31)
        }
        self.assertEqual(min(scores, key=scores.get), -18)

    def test_mesh_rotation_is_about_z(self):
        mesh = TriangleMesh(
            torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float32),
            torch.empty((0, 3), dtype=torch.long),
        )
        rotated = rotate_mesh_about_z(mesh, 90.0)
        torch.testing.assert_close(
            rotated.vertices,
            torch.tensor([[0.0, 1.0, 2.0]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_normalization_is_per_curve(self):
        values = np.asarray([[1.0, 10.0], [3.0, 30.0]])
        actual = normalized_curves(values)
        np.testing.assert_allclose(actual.mean(axis=0), [1.0, 1.0])

    def test_camera_elevations_match_asymmetric_challenge_geometry(self):
        columns = ["0_top", "45_bottom", "225_top", "225_bottom"]
        directions = build_view_dirs(columns, torch.device("cpu")).numpy()
        elevations = np.degrees(np.arcsin(directions[:, 2]))
        np.testing.assert_allclose(elevations, [21.0, -24.0, 24.0, -26.0], atol=1e-5)

    def test_intensity_threshold_is_applied_to_radiance(self):
        mesh = TriangleMesh(
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            torch.tensor([[0, 1, 2]]),
        )
        sun = torch.tensor([1.0, 0.0, 0.05])
        sun = sun / sun.norm()
        geometry = ObservationGeometry.build(
            torch.tensor([0.0]), sun, torch.tensor([[0.0, 0.0, 1.0]])
        )
        unthresholded = PhotometricForwardModel(
            geometry,
            brdf_type="lambertian",
            shading="flat",
            tau_intensity=None,
            direction_chunk=1,
            frame_chunk=1,
        ).predict(mesh, gamma=0.01).intensity
        thresholded = PhotometricForwardModel(
            geometry,
            brdf_type="lambertian",
            shading="flat",
            tau_intensity=0.10,
            beta_intensity=128.0,
            direction_chunk=1,
            frame_chunk=1,
        ).predict(mesh, gamma=0.01).intensity
        self.assertGreater(float(unthresholded), 0.0)
        self.assertLess(float(thresholded), 0.01 * float(unthresholded))

    def test_linear_intensity_threshold_is_opt_in(self):
        geometry = ObservationGeometry.build(
            torch.tensor([0.0]),
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([[-1.0, 0.0, 0.0]]),
        )
        self.assertIsNone(PhotometricForwardModel(geometry).tau_intensity)

    def test_combined_brdf_endpoints_match_pure_models(self):
        sphere = trimesh.creation.icosphere(subdivisions=1)
        mesh = TriangleMesh(
            torch.tensor(sphere.vertices, dtype=torch.float32),
            torch.tensor(sphere.faces, dtype=torch.long),
        )
        geometry = ObservationGeometry.build(
            torch.tensor([0.0, 0.3]),
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([[-1.0, 0.0, 0.0]]),
        )
        prepared = PhotometricForwardModel(geometry).prepare_visibility(mesh, 0.04)
        lambert = PhotometricForwardModel(
            geometry, brdf_type="lambertian"
        ).predict(mesh, 0.04, prepared=prepared).intensity
        lommel = PhotometricForwardModel(
            geometry, brdf_type="lommel_seeliger"
        ).predict(mesh, 0.04, prepared=prepared).intensity
        combined_lambert = PhotometricForwardModel(
            geometry, brdf_type="combined", lambertian_fraction=1.0
        ).predict(mesh, 0.04, prepared=prepared).intensity
        combined_lommel = PhotometricForwardModel(
            geometry, brdf_type="combined", lambertian_fraction=0.0
        ).predict(mesh, 0.04, prepared=prepared).intensity
        torch.testing.assert_close(combined_lambert, lambert)
        torch.testing.assert_close(combined_lommel, lommel)


if __name__ == "__main__":
    unittest.main()
