"""Transient GPU check: 256x256 no-grad evaluation of a 1,280-face proxy."""

import dataclasses

import numpy as np
import torch
import trimesh

from pytorch3d_forward import (
    ObservationBatch,
    OctantoidLightcurveModel,
    PyTorch3DLightcurveModel,
    RendererConfig,
)
from shape_model import OctantoidShapeModel

torch.manual_seed(0)

sphere = trimesh.creation.icosphere(subdivisions=3)
directions = torch.tensor(
    np.asarray(sphere.vertices), dtype=torch.float32, device="cuda"
)
faces = torch.tensor(sphere.faces, dtype=torch.long, device="cuda")

shape = OctantoidShapeModel(
    directions, faces, lmax=6, r0=0.5, lam_shape=0.3, degree_power=2.0
)
theta = 0.1 * torch.randn(shape.n_parameters, device="cuda")

# Evaluation profile: 256x256, batch one, no gradients.
config = dataclasses.replace(RendererConfig.evaluation(), intensity_slope=96.0)
model = OctantoidLightcurveModel(
    shape, PyTorch3DLightcurveModel(config)
).to("cuda")

angles = torch.tensor([0.0, 90.0], device="cuda")
columns = ("0_top", "225_bottom")
obs = ObservationBatch.challenge(angles, columns)

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    pred = model(theta, obs, return_images=True)

images = pred.images
print("images shape:", tuple(images.shape))
print("intensity:", pred.intensity.detach().cpu().numpy().tolist())
print("binary:   ", pred.binary.detach().cpu().numpy().tolist())
print("peak mem (MiB):", round(torch.cuda.max_memory_allocated() / 1048576, 2))

assert images.shape == (2, 2, 256, 256)
assert torch.isfinite(pred.intensity).all()
assert torch.isfinite(pred.binary).all()
assert float(pred.intensity.mean()) > 0.0
assert float(pred.binary.mean()) > 0.0
print("EVALUATION OK")
