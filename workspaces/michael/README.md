# Asteroid lightcurve inversion

This project reconstructs asteroid meshes from the Helsinki Asteroid
Challenge lightcurves. The maintained workflow uses a differentiable
PyTorch3D renderer, an octantoid shape model, and all 21 independent challenge
camera geometries.

## Environment

The environment is pinned to Python 3.10, Torch 2.6.0+cu124, and PyTorch3D
revision `7f8a8a14`. From the repository root:

```bash
uv sync
uv run hac-asteroids doctor
```

Full calibration and fitting require CUDA. Under WSL, `nvidia-smi` and
`/dev/dxg` must be visible to the process. A container or tool sandbox may hide
the device even when CUDA works in the normal WSL shell.

## End-to-end workflow

Calibrate the fixed camera/material settings on public asteroid 3:

```bash
uv run hac-asteroids validate \
  --asteroids 3 \
  --output-dir artifacts/calibration_asteroid3
```

Fit asteroid 3 from its simulated intensity lightcurves:

```bash
uv run hac-asteroids fit \
  --asteroids 3 \
  --calibration artifacts/calibration_asteroid3/calibration.json
```

The standard fit uses 12 evenly spaced rotation samples, all 21 independent
cameras, a 1,280-face degree-6 octantoid, and 400 Adam steps. It evaluates the
final model against every supplied rotation sample. Use `--smoke` for a
two-step integration check. Full runs fail explicitly when CUDA is absent;
smoke runs may use `--device cpu` where PyTorch3D supports it.

Every run writes a manifest, checkpoint, portable `trajectory.npz`, initial
and final STL meshes, convergence and forward-curve plots, an all-camera
residual heatmap, and per-camera metrics. Public asteroids additionally show a
reference mesh after fitting; that mesh is never part of the shape loss.

Render the stored trajectory without rerunning the fit:

```bash
uv run hac-asteroids animate \
  artifacts/runs/<run>/trajectory.npz \
  --output artifacts/runs/<run>/evolution.mp4
```

Multiple bodies can be fitted sequentially with `--asteroids 3,4,5`. The
loader supports asteroids 1–10, including asteroid 10's unusual `Asteroid010`
filenames. Simulated intensity is the default input. The duplicate `hor2`
curves are removed only after exact equality with `hor1` is verified.

## Development

```bash
uv run python -m unittest discover -v
```

See `docs/architecture.md` for the maintained/legacy boundary and
`docs/forward-model-investigation.md` for the detailed scientific investigation
behind the renderer settings.
