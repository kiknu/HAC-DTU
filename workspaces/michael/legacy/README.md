# Archived research scripts

These scripts are preserved for reproducing earlier raw-mesh and renderer
experiments. They are not called by `asteroid-invert`, are not part of the
supported fitting API, and may require optional reference dependencies.

The Gauss--Newton/Levenberg--Marquardt driver now lives in
`experiments/gauss_newton.py` as an experimental comparison. New inversion
work should use the scalar Adam path in `scalar_fit.py` through the
consolidated CLI. `outputs/` preserves results from historical runs; new runs
write to ignored `artifacts/` instead.

- `older_forward.py` is the superseded PyTorch3D renderer prototype.
- `raw_forward_validation.py` is the exact raw-mesh/Embree comparison.
- `raw_model_sweep.py` sweeps scattering and threshold settings against the
  raw-mesh reference.
