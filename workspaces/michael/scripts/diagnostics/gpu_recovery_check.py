"""Transient GPU check: differentiable octantoid recovery at 96x96.

Renders a synthetic ground-truth lightcurve with a random octantoid shape,
then recovers the shape coefficients with Adam through the differentiable
rasterizer.  Uses the 96x96 optimization profile, batch two, four faces per
pixel, and the 1,280-face icosphere topology.
"""

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
assert sphere.faces.shape[0] == 1_280
directions = torch.tensor(
    np.asarray(sphere.vertices), dtype=torch.float32, device="cuda"
)
faces = torch.tensor(sphere.faces, dtype=torch.long, device="cuda")

lmax = 6
shape = OctantoidShapeModel(
    directions, faces, lmax=lmax, r0=0.5,
    lam_shape=0.3, degree_power=2.0,
)
# Steep slope-96 gates make autograd-based fitting ill-conditioned (Adam
# amplifies the tiny gradients of the near-step gates into divergence).  Use
# gentle gates for the fitting stage; slope 96 stays for evaluation.
import dataclasses
config = dataclasses.replace(
    RendererConfig.optimization(),
    intensity_slope=8.0,
    binary_slope=8.0,
)
model = OctantoidLightcurveModel(
    shape, PyTorch3DLightcurveModel(config)
).to("cuda")

angles = torch.linspace(0.0, 315.0, 8, device="cuda")
columns = ("0_top", "45_top", "90_top", "225_bottom")
obs = ObservationBatch.challenge(angles, columns)

theta_true = 0.15 * torch.randn(shape.n_parameters, device="cuda")
with torch.no_grad():
    target = model(theta_true, obs).intensity

theta = torch.zeros(shape.n_parameters, device="cuda", requires_grad=True)
optimizer = torch.optim.Adam([theta], lr=2e-3)

losses = []
for step in range(400):
    optimizer.zero_grad()
    prediction = model(theta, obs).intensity
    loss = ((prediction - target) ** 2).mean()
    prior = shape.prior_residuals(theta)
    loss = loss + 5e-2 * (prior ** 2).mean()
    loss.backward()
    optimizer.step()
    losses.append(float(loss))

with torch.no_grad():
    final_rel = (
        float(((model(theta, obs).intensity - target) ** 2).mean())
        / max(losses[0], 1e-12)
    )
theta_error = float((theta - theta_true).norm() / theta_true.norm())

print(f"initial loss: {losses[0]:.6e}")
print(f"final loss:   {losses[-1]:.6e}  (rel {final_rel:.4f})")
print(f"theta rel error: {theta_error:.4f}")
print(
    "peak mem (MiB):",
    round(torch.cuda.max_memory_allocated() / 1048576, 2),
)
assert losses[-1] < 0.5 * losses[0], "recovery did not reduce the loss"
print("RECOVERY OK")