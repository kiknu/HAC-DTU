"""Transient GPU gradient check: backprop through an 80-face mesh at 64x64."""

import torch
import trimesh

from pytorch3d_forward import (
    ObservationBatch,
    PyTorch3DLightcurveModel,
    RendererConfig,
)
from shape_model import TriangleMesh

torch.manual_seed(0)

sphere = trimesh.creation.icosphere(subdivisions=1)
vertices = torch.tensor(
    sphere.vertices, dtype=torch.float32, device="cuda", requires_grad=True
)
mesh = TriangleMesh(
    vertices, torch.tensor(sphere.faces, dtype=torch.long, device="cuda")
)
obs = ObservationBatch.challenge(
    torch.tensor([0.0, 90.0], device="cuda"), ("0_top",)
)
model = PyTorch3DLightcurveModel(RendererConfig.gradient_test()).to("cuda")

torch.cuda.reset_peak_memory_stats()
pred = model(mesh, obs, return_images=True)
pred.intensity.sum().backward()

grad = mesh.vertices.grad
print("images shape:", tuple(pred.images.shape))
print("intensity:", pred.intensity.detach().cpu().numpy().tolist())
print("grad norm:", float(grad.norm()))
print("grad finite:", bool(torch.isfinite(grad).all()))
print("grad nonzero:", bool(grad.abs().sum() > 0))
print("peak mem (MiB):", round(torch.cuda.max_memory_allocated() / 1048576, 2))

assert pred.images.shape == (2, 1, 64, 64)
assert torch.isfinite(grad).all()
assert bool(grad.abs().sum() > 0)
print("GRADIENT OK")
