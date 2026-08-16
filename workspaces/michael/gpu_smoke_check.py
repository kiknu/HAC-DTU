"""Transient GPU smoke check: one 80-face sphere image at 32x32, no-grad."""

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
assert sphere.faces.shape[0] == 80, sphere.faces.shape

mesh = TriangleMesh(
    torch.tensor(sphere.vertices, dtype=torch.float32, device="cuda"),
    torch.tensor(sphere.faces, dtype=torch.long, device="cuda"),
)
obs = ObservationBatch.challenge(
    torch.tensor([0.0], device="cuda"), ("0_top",)
)
model = PyTorch3DLightcurveModel(RendererConfig.smoke()).to("cuda")
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    pred = model(mesh, obs, return_images=True)

images = pred.images
print("images shape:", tuple(images.shape))
print("intensity:", pred.intensity.detach().cpu().numpy().tolist())
print("binary:   ", pred.binary.detach().cpu().numpy().tolist())
print("peak mem (MiB):", round(torch.cuda.max_memory_allocated() / 1048576, 2))

assert images.shape == (1, 1, 32, 32)
assert torch.isfinite(pred.intensity).all()
assert torch.isfinite(pred.binary).all()
assert float(pred.intensity.mean()) > 0.0
assert float(pred.binary.mean()) > 0.0
print("SMOKE OK")
