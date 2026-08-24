# Experiments

This directory contains unsupported optimizer comparisons. The retained
Gauss--Newton/Levenberg--Marquardt program is useful for reproducing earlier
memory and runtime findings, but new inversions should use `asteroid-invert
fit` and scalar Adam optimization.

Run it from the repository root with:

```bash
uv run --extra renderer python -m experiments.gauss_newton
```

Generated results are written below `artifacts/experiments/gauss_newton/`.
