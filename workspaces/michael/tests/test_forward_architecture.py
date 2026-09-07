"""Regression tests for the analytic forward and shape architecture."""

import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.func import jvp

from forward_model import (
    ObservationGeometry,
    PhotometricForwardModel,
    compute_lightcurves_multicam,
)
from main import ShapeFitProblem, compute_update, make_template
from shape_model import (
    EllipsoidInitialization,
    OctantoidShapeModel,
    StarShapeModel,
    TriangleMesh,
    validate_triangle_mesh,
)


class ShapeModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.directions, self.faces, self.laplacian = make_template(1)

    def test_models_implement_common_contract(self):
        models = (
            StarShapeModel(self.directions, self.faces, self.laplacian),
            OctantoidShapeModel(
                self.directions,
                self.faces,
                lmax=2,
                check_self_intersections=False,
            ),
        )
        spec = EllipsoidInitialization(1.4, 1.0, 0.8, 0.2)
        for model in models:
            parameters = model.initial_parameters(spec)
            mesh = model.mesh(parameters)
            self.assertEqual(parameters.numel(), model.n_parameters)
            self.assertEqual(mesh.vertices.shape, self.directions.shape)
            self.assertTrue(model.validate_mesh(mesh).valid)
            self.assertEqual(
                model.prior_residuals(parameters).ndim, 1
            )

    def test_octantoid_degree_zero_is_unpenalized_ellipsoid(self):
        model = OctantoidShapeModel(
            self.directions,
            self.faces,
            lmax=2,
            max_log_scale=None,
            check_self_intersections=False,
        )
        parameters = model.zero_parameters()
        y00_b = float(model.Y_b[0, 0])
        y00_c = float(model.Y_c[0, 0])
        b_scale, c_scale = 0.7, 1.3
        parameters[model.K_a - 1] = math.log(b_scale) / y00_b
        parameters[model.K_a - 1 + model.K_b] = (
            math.log(c_scale) / y00_c
        )

        expected = model.r0 * self.directions * self.directions.new_tensor(
            [1.0, b_scale, c_scale]
        )
        torch.testing.assert_close(model.mesh(parameters).vertices, expected)
        torch.testing.assert_close(
            model.prior_residuals(parameters),
            torch.zeros_like(parameters),
        )

    def test_transverse_self_intersection_is_rejected(self):
        vertices = torch.tensor(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -0.4, -1.0],
                [0.0, -0.4, 1.0],
                [0.0, 0.8, 0.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        result = validate_triangle_mesh(
            TriangleMesh(vertices, faces), check_self_intersections=True
        )
        self.assertFalse(result.valid)
        self.assertIn("self-intersections", result.reason)


class ForwardModelTests(unittest.TestCase):
    def setUp(self):
        self.directions, self.faces, self.laplacian = make_template(1)
        self.angles = torch.arange(4, dtype=torch.float32) * (
            2.0 * math.pi / 4
        )
        self.sun = torch.tensor([-1.0, 0.0, 0.0])
        self.views = F.normalize(
            torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.3]]),
            dim=-1,
        )

    def test_mesh_only_renderer_matches_functional_forward(self):
        shape = StarShapeModel(
            self.directions, self.faces, self.laplacian
        )
        parameters = 0.08 * torch.randn(shape.n_parameters)
        mesh = shape.mesh(parameters)
        expected = compute_lightcurves_multicam(
            mesh.vertices,
            mesh.faces,
            self.sun,
            self.views,
            self.angles,
            gamma=0.04,
            eps=1e-3,
            cull_frac=None,
            dir_chunk=2,
            frame_chunk=4,
            use_compile=False,
        )
        renderer = PhotometricForwardModel(
            ObservationGeometry.build(
                self.angles, self.sun, self.views
            ),
            direction_chunk=2,
            frame_chunk=4,
        )
        actual = renderer.predict(mesh, gamma=0.04).intensity
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_perspective_renderer_matches_functional_forward(self):
        shape = StarShapeModel(
            self.directions, self.faces, self.laplacian
        )
        mesh = shape.mesh(0.05 * torch.randn(shape.n_parameters))
        expected = compute_lightcurves_multicam(
            mesh.vertices,
            mesh.faces,
            self.sun,
            self.views,
            self.angles,
            gamma=0.04,
            projection="perspective",
            camera_distance_radii=4.0,
            dir_chunk=2,
            frame_chunk=4,
            use_compile=False,
        )
        renderer = PhotometricForwardModel(
            ObservationGeometry.build(
                self.angles, self.sun, self.views
            ),
            projection="perspective",
            camera_distance_radii=4.0,
            direction_chunk=2,
            frame_chunk=4,
        )
        actual = renderer.predict(mesh, gamma=0.04).intensity
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_perspective_converges_to_orthographic_at_large_distance(self):
        sphere = trimesh.creation.icosphere(subdivisions=1)
        mesh = TriangleMesh(
            torch.tensor(sphere.vertices, dtype=torch.float32),
            torch.tensor(sphere.faces, dtype=torch.long),
        )
        geometry = ObservationGeometry.build(
            self.angles[:2], self.sun, self.views
        )
        common = dict(
            brdf_type="lambertian",
            tau_intensity=None,
            direction_chunk=2,
            frame_chunk=2,
        )
        orthographic = PhotometricForwardModel(
            geometry, projection="orthographic", **common
        ).predict(mesh, gamma=0.02).intensity
        perspective = PhotometricForwardModel(
            geometry,
            projection="perspective",
            camera_distance_radii=100.0,
            **common,
        ).predict(mesh, gamma=0.02).intensity
        torch.testing.assert_close(
            perspective, orthographic, rtol=0.02, atol=2e-3
        )

    def test_subface_samples_capture_partial_visibility(self):
        mesh = TriangleMesh(
            torch.tensor(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-0.2, -0.5, 0.5],
                    [0.2, -0.5, 0.5],
                    [0.0, -0.1, 0.5],
                ]
            ),
            torch.tensor([[0, 1, 2], [3, 4, 5]]),
        )
        direction = torch.tensor([0.0, 0.0, 1.0])
        geometry = ObservationGeometry.build(
            torch.tensor([0.0]), direction, direction[None]
        )
        values = []
        for samples in (1, 3):
            values.append(
                PhotometricForwardModel(
                    geometry,
                    brdf_type="lambertian",
                    shading="flat",
                    visibility_samples=samples,
                    tau_intensity=None,
                    direction_chunk=1,
                    frame_chunk=1,
                ).predict(mesh, gamma=0.005).intensity
            )
        self.assertGreater(float(values[1]), 1.5 * float(values[0]))

    def test_perspective_has_consistent_shape_derivative(self):
        shape = StarShapeModel(
            self.directions, self.faces, self.laplacian
        )
        mesh = shape.mesh(0.06 * torch.randn(shape.n_parameters))
        renderer = PhotometricForwardModel(
            ObservationGeometry.build(
                self.angles[:2], self.sun, self.views[:1]
            ),
            projection="perspective",
            camera_distance_radii=4.0,
            tau_intensity=None,
            direction_chunk=1,
            frame_chunk=2,
        )

        def prediction(vertices):
            return renderer.predict(
                TriangleMesh(vertices, mesh.faces), gamma=0.04
            ).intensity

        direction = 0.1 * torch.randn_like(mesh.vertices)
        _, analytic = jvp(
            prediction, (mesh.vertices,), (direction,)
        )
        step = 1e-3
        finite_difference = (
            prediction(mesh.vertices + step * direction)
            - prediction(mesh.vertices - step * direction)
        ) / (2.0 * step)
        torch.testing.assert_close(
            analytic, finite_difference, rtol=0.03, atol=2e-4
        )

    def test_problem_is_shape_model_agnostic(self):
        for shape in (
            StarShapeModel(
                self.directions, self.faces, self.laplacian
            ),
            OctantoidShapeModel(
                self.directions,
                self.faces,
                lmax=2,
                check_self_intersections=False,
            ),
        ):
            parameters = shape.initial_parameters()
            renderer = PhotometricForwardModel(
                ObservationGeometry.build(
                    self.angles, self.sun, self.views
                ),
                direction_chunk=2,
                frame_chunk=4,
            )
            raw = renderer.predict(
                shape.mesh(parameters), gamma=0.04
            ).intensity
            data = raw / raw.mean(dim=0, keepdim=True)
            problem = ShapeFitProblem(
                data,
                self.angles,
                self.sun,
                self.views,
                shape_model=shape,
                w_curve=torch.ones(2),
                dir_chunk=2,
                frame_chunk=4,
            )
            residual = problem.residuals(parameters, gamma=0.04)
            torch.testing.assert_close(
                residual[: problem.n_photo],
                torch.zeros(problem.n_photo),
                atol=1e-6,
                rtol=0.0,
            )
            frozen = problem.residual_function(
                parameters, 0.04, "frozen"
            )
            self.assertEqual(frozen(parameters).shape, residual.shape)

    def test_every_face_can_occlude_in_nonconvex_scene(self):
        # Two closed components share the same projected footprint.  This is
        # the limiting case of one lobe fully occluding another.
        front = trimesh.creation.box(extents=[1.0, 1.0, 0.4])
        rear = trimesh.creation.box(extents=[1.0, 1.0, 0.4])
        front.apply_translation([0.0, 0.0, 0.5])
        rear.apply_translation([0.0, 0.0, -0.5])
        vertices = torch.tensor(
            np.vstack([front.vertices, rear.vertices]),
            dtype=torch.float32,
        )
        faces = torch.tensor(
            np.vstack(
                [front.faces, rear.faces + len(front.vertices)]
            ),
            dtype=torch.long,
        )
        direction = torch.tensor([0.0, 0.0, 1.0])
        geometry = ObservationGeometry.build(
            torch.tensor([0.0]), direction, direction[None]
        )
        renderer = PhotometricForwardModel(
            geometry, direction_chunk=1, frame_chunk=1
        )
        mesh = TriangleMesh(vertices, faces)
        _, normals = renderer.face_geometry(mesh)
        visibility = renderer.prepare_visibility(mesh, gamma=0.01).table[0]
        front_facing = normals[:, 2] > 0.9
        front_component = front_facing.clone()
        front_component[len(front.faces) :] = False
        rear_component = front_facing.clone()
        rear_component[: len(front.faces)] = False
        self.assertGreater(float(visibility[front_component].min()), 0.99)
        self.assertLess(float(visibility[rear_component].max()), 1e-6)

    def test_occluded_scene_converges_to_analytic_reference(self):
        direction = torch.tensor([0.0, 0.0, 1.0])
        renderer = PhotometricForwardModel(
            ObservationGeometry.build(
                torch.tensor([0.0]), direction, direction[None]
            ),
            brdf_type="lambertian",
            tau_intensity=None,
            direction_chunk=1,
            frame_chunk=1,
        )
        exact = 2.0 * math.pi * 0.5**2 / 3.0
        errors = []
        for subdivisions in (1, 2):
            front = trimesh.creation.icosphere(
                subdivisions=subdivisions, radius=0.5
            )
            rear = front.copy()
            front.apply_translation([0.0, 0.0, 0.75])
            rear.apply_translation([0.0, 0.0, -0.75])
            vertices = torch.tensor(
                np.vstack([front.vertices, rear.vertices]),
                dtype=torch.float32,
            )
            faces = torch.tensor(
                np.vstack(
                    [front.faces, rear.faces + len(front.vertices)]
                ),
                dtype=torch.long,
            )
            brightness = renderer.predict(
                TriangleMesh(vertices, faces), gamma=0.002
            ).intensity.item()
            errors.append(abs(brightness / exact - 1.0))

        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], 0.02)

    def test_octantoid_exact_residual_has_correct_directional_derivative(self):
        torch.manual_seed(8)
        shape = OctantoidShapeModel(
            self.directions,
            self.faces,
            lmax=2,
            check_self_intersections=False,
        )
        parameters = 0.02 * torch.randn(shape.n_parameters)
        renderer = PhotometricForwardModel(
            ObservationGeometry.build(
                self.angles, self.sun, self.views
            ),
            direction_chunk=2,
            frame_chunk=4,
        )
        raw = renderer.predict(
            shape.mesh(parameters), gamma=0.04
        ).intensity
        problem = ShapeFitProblem(
            raw / raw.mean(dim=0, keepdim=True),
            self.angles,
            self.sun,
            self.views,
            shape_model=shape,
            w_curve=torch.ones(2),
            dir_chunk=2,
            frame_chunk=4,
        )
        direction = torch.randn_like(parameters)
        direction /= direction.norm()
        fn = lambda q: problem.residuals(q, gamma=0.04)
        analytic = jvp(fn, (parameters,), (direction,))[1]
        step = 3e-4
        finite_difference = (
            fn(parameters + step * direction)
            - fn(parameters - step * direction)
        ) / (2.0 * step)
        relative_error = (finite_difference - analytic).norm() / analytic.norm()
        self.assertLess(float(relative_error), 1e-2)

    def test_octantoid_gauss_newton_step_reduces_exact_cost(self):
        shape = OctantoidShapeModel(
            self.directions,
            self.faces,
            lmax=2,
            lam_shape=0.02,
            check_self_intersections=False,
        )
        target = shape.zero_parameters()
        target[shape.K_a] = 0.12
        renderer = PhotometricForwardModel(
            ObservationGeometry.build(
                self.angles, self.sun, self.views
            ),
            direction_chunk=2,
            frame_chunk=4,
        )
        raw = renderer.predict(shape.mesh(target), gamma=0.04).intensity
        problem = ShapeFitProblem(
            raw / raw.mean(dim=0, keepdim=True),
            self.angles,
            self.sun,
            self.views,
            shape_model=shape,
            w_curve=torch.ones(2),
            dir_chunk=2,
            frame_chunk=4,
        )
        initial = shape.zero_parameters()
        cost_before = 0.5 * problem.residuals(initial, 0.04).square().sum()
        delta, _, _ = compute_update(
            problem,
            initial,
            0.04,
            1e-2,
            method="frozen",
            jac_chunk=16,
        )
        cost_after = 0.5 * problem.residuals(
            initial + delta, 0.04
        ).square().sum()
        self.assertLess(float(cost_after), float(cost_before))


if __name__ == "__main__":
    unittest.main()
