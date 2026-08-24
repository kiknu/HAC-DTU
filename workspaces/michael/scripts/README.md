# Repository scripts

- `validation/visualize_match.py` creates standalone observed-versus-predicted
  lightcurve reports.
- `diagnostics/gpu_*_check.py` preserves targeted CUDA experiments from the
  earlier desktop environment. These scripts require CUDA and are not the
  Apple Silicon test path.

Routine fitting, calibration, reporting, and platform checks should use the
installed `asteroid-invert` command.
