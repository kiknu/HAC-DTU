import unittest

import numpy as np
import torch
import trimesh

from pytorch3d_forward import ObservationBatch, PyTorch3DLightcurveModel, RendererConfig
from shape_model import TriangleMesh


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device is not available")
class GpuPipelineTests(unittest.TestCase):
    def test_small_render_and_gradient(self):
        sphere = trimesh.creation.icosphere(subdivisions=1)
        vertices = torch.tensor(np.asarray(sphere.vertices), dtype=torch.float32, device="cuda")
        vertices.requires_grad_(True)
        faces = torch.tensor(sphere.faces, dtype=torch.long, device="cuda")
        renderer = PyTorch3DLightcurveModel(
            RendererConfig(
                image_size=32, shadow_size=32, faces_per_pixel=1,
                batch_size=1, face_limit=80, intensity_slope=8,
            )
        ).cuda()
        observations = ObservationBatch.challenge(
            torch.tensor([0.0], device="cuda"), ("0_top",)
        )
        result = renderer(
            TriangleMesh(vertices, faces), observations, compute_binary=False
        )
        result.intensity.sum().backward()
        self.assertTrue(torch.isfinite(result.intensity).all())
        self.assertTrue(torch.isfinite(vertices.grad).all())


if __name__ == "__main__":
    unittest.main()
