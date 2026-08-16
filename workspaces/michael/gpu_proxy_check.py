"""Transient GPU check: public asteroid proxies through the renderer.

Asteroid 1 (nearly spherical, flat rotation curves): compare per-camera
brightness structure and per-frame curve error against the CPU model.
Asteroid 3 (strongly triaxial, varying curves): per-frame Pearson correlation.
Uses 1,280-face star proxies only - never the raw STLs.
"""

from pathlib import Path

import numpy as np
import torch

from forward_model import (
    ObservationGeometry,
    PhotometricForwardModel,
    build_view_dirs,
)
from forward_validation import load_benchmark_mesh, normalized_curves
from pytorch3d_forward import (
    ObservationBatch,
    PyTorch3DLightcurveModel,
    RendererConfig,
)

device = torch.device("cuda")
cache_dir = Path("artifacts/forward_validation")
angles = torch.linspace(0.0, 330.0, 12, device=device)
columns = ("0_top", "45_top", "90_top", "225_bottom")


def curves(model_num: int):
    mesh = load_benchmark_mesh(
        model_num, 1_280, device, cache_dir, "star_proxy"
    )
    gpu = PyTorch3DLightcurveModel(RendererConfig.evaluation()).to(device)
    obs = ObservationBatch.challenge(angles, columns)
    with torch.no_grad():
        pred = gpu(mesh, obs)
    # Compare raw (pre-normalization) brightness; the challenge's 25.5/255
    # pixel cutoff is applied by both models (GPU soft gate / CPU tau).
    gpu_raw = pred.raw_intensity.detach().cpu().numpy()
    view_dirs = build_view_dirs(columns, device)
    geometry = ObservationGeometry.build(
        angles, angles.new_tensor([-1.0, 0.0, 0.0]), view_dirs
    )
    cpu_model = PhotometricForwardModel(
        geometry,
        brdf_type="combined",
        lambertian_fraction=0.5,
        shading="smooth",
        projection="perspective",
        camera_distance_radii=5.0,
        tau_intensity=25.5 / 255.0,
        beta_intensity=64.0,
    )
    cpu_raw = cpu_model.predict(mesh, gamma=0.02).intensity.detach().cpu().numpy()
    return gpu_raw, cpu_raw


def structure(raw: np.ndarray) -> np.ndarray:
    means = np.asarray(raw, dtype=np.float64).mean(axis=0)
    return means / means.sum()


gpu1, cpu1 = curves(1)
# Asteroid 1: per-camera brightness structure (flat rotation curves).
gpu_s, cpu_s = structure(gpu1), structure(cpu1)
ratio_errors = np.abs(gpu_s - cpu_s) / cpu_s
frame_rmse = np.sqrt(((normalized_curves(gpu1) - normalized_curves(cpu1)) ** 2).mean(axis=0))
print("asteroid 1 gpu structure:", np.round(gpu_s, 4))
print("asteroid 1 cpu structure:", np.round(cpu_s, 4))
print("asteroid 1 per-camera structure rel err:", np.round(ratio_errors, 4))
print("asteroid 1 per-frame curve RMSE      :", np.round(frame_rmse, 4))
assert ratio_errors.max() < 0.05, "asteroid 1 camera structure must agree"
# Camera 4 (225_bottom) is sub-threshold (structure 2.5%): its near-zero
# normalized curve is noise-dominated, so only the lit cameras are checked.
assert frame_rmse[:3].max() < 0.03, "asteroid 1 lit-camera curves must agree"

gpu3, cpu3 = curves(3)
gpu3n, cpu3n = normalized_curves(gpu3), normalized_curves(cpu3)
correlations = [
    float(np.corrcoef(gpu3n[:, i], cpu3n[:, i])[0, 1]) for i in range(len(columns))
]
rmse3 = np.sqrt(((gpu3n - cpu3n) ** 2).mean())
print("asteroid 3 per-camera pearson r:", np.round(correlations, 4))
print(f"asteroid 3 mean pearson r = {np.mean(correlations):.4f}, curve RMSE = {rmse3:.4f}")
assert np.mean(correlations) > 0.9, "asteroid 3 curves must correlate strongly"
print("PROXY VALIDATION OK")
