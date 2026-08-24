import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asteroid_cli import build_parser
from mitsuba_reference import integrate_reference_image
from pytorch3d_forward import LightcurvePrediction, ObservationBatch
from scalar_fit import (
    FitConfig,
    LightcurveData,
    RendererHyperparameters,
    create_octantoid_shape,
    device_capabilities,
    fit_octantoid,
    new_octantoid_state,
    OctantoidState,
    photometric_loss,
    renderer_smoke_test,
    resolve_device,
    shape_from_state,
    wrap_phase,
)


class _ToyLightcurveModel:
    """Analytic renderer used to test optimizer orchestration without P3D."""

    def __init__(self, shape):
        self.shape = shape

    def __call__(
        self,
        parameters,
        observations,
        *,
        phase_offset=0.0,
        extraction="soft",
        **_,
    ):
        del extraction
        phase = torch.as_tensor(phase_offset).to(parameters)
        amplitude = 0.12 + 0.16 * torch.tanh(parameters[0])
        camera_scale = 1.0 + 0.15 * observations.view_directions[:, 2]
        values = 1.0 + amplitude * torch.cos(
            observations.angles[:, None] + phase
        ) * camera_scale[None]
        values = values / values.mean(dim=0, keepdim=True)
        return LightcurvePrediction(
            intensity=values,
            binary=values,
            raw_intensity=values,
            raw_binary=values,
        )


class ScalarFitTests(unittest.TestCase):
    def test_losses_and_phase_wrapping(self):
        predicted = torch.tensor([0.0, 2.0])
        target = torch.tensor([0.0, 0.0])
        mse = photometric_loss(predicted, target, kind="mse", huber_delta=0.5)
        huber = photometric_loss(predicted, target, kind="huber", huber_delta=0.5)
        self.assertAlmostEqual(float(mse), 2.0)
        self.assertLess(float(huber), float(mse))
        wrapped = wrap_phase(torch.tensor(3.0 * math.pi))
        self.assertAlmostEqual(abs(float(wrapped)), math.pi, places=5)

    def test_state_round_trip_and_resolution_transfer(self):
        device = torch.device("cpu")
        shape = create_octantoid_shape(
            1, device, lmax_a=2, check_self_intersections=False
        )
        state = new_octantoid_state(shape, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            state.save(path)
            loaded = OctantoidState.load(path)
        torch.testing.assert_close(loaded.parameters, state.parameters)
        transferred = shape_from_state(
            loaded, device, subdivisions=2, check_self_intersections=False
        )
        self.assertEqual(transferred.n_parameters, shape.n_parameters)
        self.assertGreater(len(transferred.directions), len(shape.directions))

    def test_adam_recovery_reduces_scalar_loss(self):
        device = torch.device("cpu")
        shape = create_octantoid_shape(
            1, device, lmax_a=2, check_self_intersections=False
        )
        model = _ToyLightcurveModel(shape)
        angles = torch.arange(24, dtype=torch.float32) * (2.0 * math.pi / 24)
        observations = ObservationBatch.challenge(
            angles, ("0_top", "225_bottom")
        )
        true_parameters = shape.zero_parameters()
        true_parameters[0] = 0.8
        with torch.no_grad():
            target = model(
                true_parameters, observations, phase_offset=0.4
            ).intensity
        data = LightcurveData(
            asteroid=0,
            origin="simulated",
            columns=observations.columns,
            degrees=np.arange(24) * 15.0,
            target=target,
            observations=observations,
            frame_stride=1,
        )
        state = new_octantoid_state(shape, 1)
        result = fit_octantoid(
            model,
            data,
            state,
            FitConfig(
                iterations=80,
                learning_rate=5e-2,
                phase_learning_rate=5e-2,
                loss="mse",
                prior_weight=0.0,
                camera_group_size=1,
                evaluation_interval=4,
                self_intersection_interval=0,
            ),
            RendererHyperparameters(),
        )
        initial = np.mean((result.initial_prediction - result.observed) ** 2)
        fitted = np.mean((result.fitted_prediction - result.observed) ** 2)
        self.assertLess(fitted, 0.25 * initial)
        self.assertAlmostEqual(
            result.state.phase_radians, 0.4, delta=0.12
        )

    def test_device_policy_and_cli_contract(self):
        capabilities = device_capabilities()
        selected = resolve_device("auto")
        if capabilities.cuda_available:
            self.assertEqual(selected.type, "cuda")
        else:
            self.assertEqual(selected.type, "cpu")
        parser = build_parser()
        benchmark = parser.parse_args(["benchmark", "--compare-losses"])
        self.assertEqual(benchmark.asteroids, "1,2,3")
        fit = parser.parse_args(["fit", "--asteroid", "4"])
        self.assertEqual(fit.loss, "huber")

    def test_real_cpu_renderer_forward_and_backward_when_installed(self):
        if not device_capabilities().pytorch3d_available:
            self.skipTest("PyTorch3D native extension is not installed")
        result = renderer_smoke_test(torch.device("cpu"), backward=True)
        self.assertTrue(result["ok"], result.get("error"))
        orthographic = renderer_smoke_test(
            torch.device("cpu"), backward=True, projection="orthographic"
        )
        self.assertTrue(orthographic["ok"], orthographic.get("error"))

    def test_mitsuba_hdr_integration_requires_explicit_display_scale(self):
        image = np.array([[0.0, 0.05], [0.2, 1.0]], dtype=np.float32)
        self.assertAlmostEqual(integrate_reference_image(image), 1.25)
        with self.assertRaisesRegex(ValueError, "display_scale"):
            integrate_reference_image(image, response="challenge_8bit")
        quantized = integrate_reference_image(
            image, response="challenge_8bit", display_scale=1.0
        )
        self.assertAlmostEqual(quantized, 1.2, places=2)


if __name__ == "__main__":
    unittest.main()
