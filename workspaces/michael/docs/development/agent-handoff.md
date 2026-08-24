# Scalar-loss inversion continuation handoff (updated 2026-08-24)

## Current production path

- `scalar_fit.py` implements versioned octantoid states, Huber/MSE objectives,
  Adam fitting, phase co-estimation, leave-one-body-out renderer calibration,
  metrics, mesh export, and plots.
- `asteroid_cli.py` exposes `doctor`, `calibrate`, `benchmark`, `fit`, and
  `report` through the `asteroid-invert` entry point.
- `pytorch3d_forward.py` remains the production differentiable renderer. It
  now supports perspective and orthographic observer cameras and separate CPU
  and CUDA optimization profiles.
- `experiments/gauss_newton.py` (Gauss--Newton/LM) and `forward_model.py`
  (custom CPU renderer) are reference/experimental paths, not the default
  workflow.

## Platforms

- Python 3.10 and Torch 2.5.1 are locked for both platforms.
- Linux x86-64 uses the Torch CUDA 12.1 index.
- Apple Silicon uses the PyPI CPU Torch build. `auto` selects CPU on macOS.
- PyTorch3D 0.7.9 is pinned to revision `7f8a8a14`. On modern Apple Clang it
  requires `-Wno-invalid-specialization`; the isolated-build variables are in
  `pyproject.toml`.
- MPS is deliberately experimental. The native PyTorch3D extension is CPU/CUDA
  oriented, and MPS must pass `asteroid-invert doctor --device mps
  --experimental-mps` before use.

The M4 CPU build and real rasterizer backward pass were verified locally with:

```bash
uv sync --extra renderer
uv run --extra renderer asteroid-invert doctor --device cpu
```

## Validation status

- The doctor forward/backward smoke test passes on the M4 CPU.
- Mitsuba 3.9.1's scalar CPU variant and Rtree reference acceleration are
  installed; `asteroid-invert doctor --reference` renders successfully.
- A real two-step PyTorch3D/Adam smoke fit on asteroid 1 completed and reduced
  mean curve RMSE from 0.0463 to 0.0438. Its generated report is under ignored
  `artifacts/cpu_end_to_end_smoke/`.
- The intended 642-vertex/1,280-face topology also completed a real CPU
  forward/backward/report smoke run (two sampled phases, 32 px) in about eight
  seconds.
- The scalar orchestration suite tests checkpoint transfer, loss selection,
  phase wrapping, CLI defaults, and backend-free Adam recovery.
- Embree is optional and imported lazily. Small proxy tests use a dependency-
  free radial-intersection fallback; million-face reference proxy generation
  still requires an accelerated reference backend or an existing cache.

## Scientific conventions

- Fit the 21 unique curves; `hor2` duplicates `hor1`.
- Angles passed to PyTorch3D are radians and body rotation is `-angle`.
- Asteroids 1--3 use simulated intensity curves for public validation;
  asteroids 4--9 use real intensity curves for hidden-shape fitting.
- Full public meshes are reference inputs only. Use 642 vertices/1,280 faces
  for optimization and transfer coefficients to 2,562 vertices/5,120 faces
  for optional higher-resolution evaluation.
- Shared projection, camera-distance, scattering-mixture, and shadow-softness
  values are calibrated on public bodies. Only shape and a per-body phase are
  optimized for hidden bodies.
