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

Calibrate the fixed image-formation settings on public asteroid 3. The grid
screens 72 phases and verifies finalists over all 360 phases and 21 cameras:

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

The standard fit uses 12 evenly spaced training samples, 24 disjoint validation
samples, all 21 independent cameras, a 1,280-face degree-6 octantoid, and 300
Adam steps. It starts with an aggressive, low-degree escape phase, enables all
coefficients for cosine-decayed refinement, and finishes with stronger spectral
regularization. The best validation checkpoint is evaluated against every
supplied rotation sample. Use `--smoke` for a
two-step integration check. Full runs fail explicitly when CUDA is absent;
smoke runs may use `--device cpu` where PyTorch3D supports it.

Tune a reusable optimizer recipe on public asteroid 3:

```bash
uv run hac-asteroids tune \
  --asteroids 3 \
  --calibration artifacts/calibration_asteroid3/calibration.json \
  --output-dir artifacts/tuning
```

The default resumable study performs 36 coarse 100-step runs and 24 finalist
300-step runs when all three public bodies are explicitly included with a
compatible calibration. It ranks configurations by exact-renderer validation RMSE and
writes `leaderboard.csv`, `study.json`, and `best_config.json`. Apply the winner
without retuning each body:

```bash
uv run hac-asteroids fit \
  --asteroids 3 \
  --calibration artifacts/calibration_asteroid3/calibration.json \
  --config artifacts/tuning/best_config.json
```

Individual schedule and regularization values can also be overridden with
`--initial-learning-rate`, `--learning-rate`, `--escape-steps`,
`--regularization-weight`, `--degree-power`, and `--gradient-clip`.
Calibration files of every version require `--calibration-transfer` (or
`calibration_transfer: true` in the fit configuration) when fitting a body
outside their stated scope. Such transfer is an explicit modeling assumption.

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

## Radial harmonic versus octantoid comparison

For an octantoid-only L2/L4/L6 comparison on asteroids 1–3, use
`python -m hac_asteroids.representation_study run --profile octantoid --output-dir artifacts/octantoids_l2_l4_l6_20260910`.
This runs nine fresh 1,024-face sphere starts, selecting a shared 300–500 update
cap from a benchmark targeting eight hours including reporting. Fits can stop
after a sustained validation plateau and a lower-learning-rate confirmation.
Outputs include selected/terminal lightcurves and meshes, deformation GIFs,
geometry metrics and resumable checkpoints. See the
[study specification](docs/representation-study.md) for exact stopping rules.

The controlled representation study runs asteroids 1–3 from the same exact
1,024-face sphere for 500 updates each. It compares radial spherical harmonics
at degrees 6 and 11 (48 and 143 coefficients) with degree-6 octantoids (146).
Both the same-degree and similar-parameter-count comparisons use the same
optimizer, phase partition, calibration and spectral coefficient penalty.

```bash
python -m hac_asteroids.representation_study run \
  --output-dir artifacts/representations_1024_500_20260909
```

The command calibrates asteroid 1, reuses the corrected per-body calibrations
for 2–3, benchmarks memory/runtime, and freezes the experiment before fitting.
Repeating it resumes compatible checkpoints and skips audited completed fits.
`status.json`, per-fit `fit.log` files and `report.md` track progress. The report
includes lightcurves, selected/last geometry scores, deformation measurements,
STL meshes and individual/synchronized GIFs. These experimental comparisons are
conditional on public full-turn calibration and a shared optimizer recipe.

Individual fits also accept `--representation radial_sh|octantoid`, `--lmax`,
`--faces`, `--initialization sphere|ellipsoid`, and
`--regularization-denominator`. Exact face counts override `subdivisions`.
Omitting these options preserves the historical octantoid/ellipsoid defaults.
See [the experiment specification](docs/representation-study.md) for controls
and interpretation.

For a roughly one-hour screening comparison on asteroid 3, use the quick
profile. It keeps the same mesh, renderer, cameras and phase coverage, and
shortens each fit to 120 updates with validation every 20 updates:

```bash
python -m hac_asteroids.representation_study run --profile quick \
  --output-dir artifacts/representations_quick_a3_120_20260909
```

The quick profile reuses asteroid 3's calibration and benchmarks three updates
spanning its 12-step escape, 96-step refinement and 12-step stabilization
schedule. It writes three fit results, 121-state trajectories, and reports
labelled with step 120. Use a separate output directory from the full study;
changing profile within a saved study is rejected.

For report examples across all ten asteroids, run the local L4 survey:

```bash
python -m hac_asteroids.l4_survey prepare \
  --output-dir artifacts/octantoid_l4_all10_200_20260910
python -m hac_asteroids.campaign_runner \
  --output-dir artifacts/octantoid_l4_all10_200_20260910
```

This runs one fit at a time from identical 1,024-face spheres for exactly 200
updates (40 escape, 120 refinement, 40 stabilization), with no plateau stopping.
The measured RTX 3080 estimate is about five hours including reporting.
Asteroids 4–10 provisionally reuse asteroid 3's rendering calibration; their
reference shapes are hidden, so geometry error scores are unavailable.
Each fit immediately produces a report, selected and terminal meshes,
all-camera lightcurves, and a GIF covering all 200 updates at five-step
intervals. All 201 states remain in the trajectory. `report.md` and
`shapes_overview.png` collect the completed examples; `runner_status.json`
records the heartbeat. Repeat the supervisor command to resume after shutdown.
Use `python -m hac_asteroids.l4_survey report --output-dir <directory>` to
repair missing or corrupted reports without rerunning completed fits.

## Development

```bash
uv run python -m unittest discover -v
```

## Mathematical specification and known limitations

See [mathspec.md](mathspec.md) for the implemented forward model, octantoid
representation, and staged Adam solver, and [architecture](docs/architecture.md)
for the maintained/legacy boundary.

The legacy 12-phase training setup can miss narrow lightcurve peaks.
The experimental correction uses `--frames 72 --validation-frames 72
--training-block-frames 6`, with disjoint grids at indices 0 and 2 modulo 5.
Training accumulates exact complete-set normalization derivatives in two
renderer passes. Validation always normalizes over the full turn before scoring
selected rows. Scores are conditional on public full-turn calibration.

The [asteroid-2 diagnosis](docs/asteroid2-residual-diagnosis.md) documents
the verified data provenance, sampling bias, image-response mismatch, and shape
limitations. The [correction plan](docs/forward-model-fix-plan.md) defines the
acceptance gate. The maintained campaign and read-only controls are available as
`python -m hac_asteroids.correction_campaign --help` and
`python -m hac_asteroids.correction_controls --help`.
See the [correction results](docs/forward-model-correction-results.md) for the
implemented checks, calibration evidence and paired experiment status.

Linear/sRGB response, radiance gain, perspective/orthographic projection and a
body-wide phase convention are explicit calibration settings. Legacy files load
with their original defaults and a warning. Version-3 checkpoints reject changed
data, code, calibration or phase splits. Old trajectories remain readable.

The [August investigation](docs/forward-model-investigation.md) is historical
context, not a statement that the current renderer's discrepancies are resolved.
