# CUDA 12.4 continuation handoff (updated 2026-08-16)

Status: the GPU differentiable renderer is built and fully validated end to
end. The original blocker (no NVIDIA driver visible from WSL) is resolved:
`nvidia-smi` works, Torch `2.6.0+cu124` allocates on `cuda:0` (RTX 3080,
10 GB), and PyTorch3D 0.7.9 (pinned rev `7f8a8a14`) is compiled and installed.

## Environment / build

- Python 3.10, Torch `2.6.0+cu124`, PyTorch3D pinned to
  `7f8a8a142fa1383a7895ed01dd7082105f9a81fb` in `pyproject.toml`.
- PyTorch3D builds from source with `MAX_JOBS=4` (raised from 1 on
  2026-08-16: the single-job build took over an hour on this 16-core, 15 GB-RAM
  machine; 4 jobs peak at ~10 GB and are safe). The build is incremental via
  ninja, so a re-run resumes from compiled objects.
- There are NO official prebuilt CUDA wheels for modern PyTorch3D; upgrading
  the CUDA toolkit or moving out of WSL would not have sped anything up.

## Bugs found and fixed in `pytorch3d_forward.py` (integration + numerics)

1. **Camera slicing**: PyTorch3D 0.7.9 `FoVPerspectiveCameras.__getitem__`
   accepts `int`/`List[int]`/tensors, not slices. Chunked rendering now uses
   `all_cameras[list(range(start, stop))]`.
2. **`get_zfar` removed** in 0.7.9: use `getattr(cameras, "zfar", None)`.
3. **Lommel–Seeliger gradient blowup**: `mu0 / (mu0 + mu + 1e-8)` has
   unbounded derivatives at silhouette pixels where both terms -> 0. Now
   `mu0 / (mu0 + mu).clamp_min(1e-3)`; affected pixels have radiance below the
   0.1 intensity gate, so the lightcurves are unchanged.
4. **`blur_radius` default 1e-4 -> 0.0**: a non-zero blur makes the rasterizer
   include near-miss faces; edge-on faces (zero projected area) make the
   barycentric backward divide by a tiny determinant, exploding gradients
   (measured 8.8e14 -> 8.3e4 at `blur_radius=0`). The softmax blend already
   smooths edges, so exact rasterization is used.
5. **Rotation sign (the big one)**: the GPU rotated the mesh by `R_z(+angle)`
   while the CPU forward model rotates observers by `+angle`
   (`rotation_sign=-1` in `ObservationGeometry`) - equivalent to mesh
   `R_z(-angle)`. `_meshes_for_frames` now rotates by `-angles`. This was
   confirmed via pure projected-area cross-checks (r=0.99 with the fix) and
   was the cause of all high-phase-angle proxy mismatches.

## Validation results (all green)

Run the staged checks in this order (each prints `... OK` and peak GPU memory):

```bash
uv run --project . --no-sync python gpu_smoke_check.py      # 80-face sphere, 32x32 no-grad, 8.8 MiB
uv run --project . --no-sync python gpu_gradient_check.py   # backprop through 80-face mesh, 64x64, 24 MiB
uv run --project . --no-sync python gpu_recovery_check.py   # synthetic octantoid fit at 96x96, 439 MiB
uv run --project . --no-sync python gpu_eval_check.py       # 1,280-face proxy at 256x256 no-grad, 46 MiB
uv run --project . --no-sync python gpu_proxy_check.py      # public asteroids 1+3 vs CPU model
```

- CPU suite still passes (32 tests):
  `uv run --project . --no-sync python -m unittest -v test_render_safety.py
  test_pytorch3d_cpu_helpers.py test_forward_validation.py
  test_forward_architecture.py`
- Recovery: octantoid fit (lmax=6, 8 frames x 4 cameras, Adam lr=2e-3,
  spectral prior weight 5e-2, 400 steps) reduces the curve loss to ~0.1 of
  its initial value. Gentle gate slopes (intensity/binary `8.0`) are required
  for autograd fitting; the steep slope-96 gates are for evaluation only.
- Proxy validation vs the CPU forward model (perspective, combined BRDF 0.5,
  smooth shading, `tau_intensity=25.5/255`, `beta_intensity=64`):
  asteroid 1 (nearly spherical): per-camera brightness structure within ~4%;
  asteroid 3 (strongly triaxial): per-camera Pearson r in [0.995, 1.000],
  curve RMSE ~1.5%.

## Notes and next steps

- Peak GPU memory during 96x96 gradient optimization is ~440 MiB (the
  autograd graph for all chunked images is retained until backward; a longer
  frame set would grow it). 256x256 no-grad evaluation is ~46 MiB.
- `F.grid_sample` backward (shadow sampling) has no deterministic CUDA
  implementation in Torch 2.6; `torch.use_deterministic_algorithms(True)`
  cannot be enabled. The octantoid fit is therefore slightly run-to-run
  variable (~0.1-0.3 relative final loss on the check problem); a stronger
  spectral prior stabilizes it.
- The GPU renderer now matches the CPU forward model well enough to use for
  challenge scoring; a formal calibration (gamma-equivalent shadow softness,
  per-pixel vs per-face shading) is still open work if exact parity is needed.
- Raw public STL input stays disabled for the differentiable renderer; always
  render the 1,280-face star proxy (`forward_validation.load_benchmark_mesh`).
