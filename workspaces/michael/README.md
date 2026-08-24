# Differentiable asteroid lightcurve inversion

The recommended workflow fits octantoid shape coefficients and one rotational
phase with Adam through the PyTorch3D renderer. Huber loss is the default;
MSE is retained as a controlled comparison. Asteroids 1--3 are full public
examples with reference meshes, while asteroids 4--9 are hidden-shape cases.
The public meshes can contain millions of STL vertices and are reference data
only: optimization uses a 642-vertex icosphere and can evaluate the same
spherical-harmonic state on a 2,562-vertex icosphere.

Install the core environment and native renderer, then verify the actual
forward and backward operations:

```bash
uv sync --extra renderer
uv run --extra renderer asteroid-invert doctor --device auto
```

On Apple Silicon, `auto` deliberately selects CPU. PyTorch3D 0.7.9 is built
locally with the modern-Clang compatibility flag recorded in `pyproject.toml`.
MPS is not selected automatically because PyTorch3D's native rasterizer is not
an MPS backend; `--device mps --experimental-mps` is accepted only for explicit
capability testing. On Linux x86-64 the lock selects Torch's CUDA 12.1 build
and `auto` selects CUDA when it is usable.

The main commands are:

```bash
# Leave-one-body-out tuning on the known objects
uv run --extra renderer --extra reference asteroid-invert calibrate --asteroids 1,2,3 --device auto

# Public recovery and Huber-versus-MSE comparison
uv run --extra renderer --extra reference asteroid-invert benchmark --asteroids 1,2,3 --compare-losses

# Hidden shape: real intensity data and any compatible octantoid prior
uv run --extra renderer asteroid-invert fit --asteroid 4 \
  --initial-state prior.pt \
  --calibration artifacts/calibration/calibration.json \
  --evaluation-subdivisions 4

# Recreate all figures from a completed run without refitting
uv run asteroid-invert report --run-dir artifacts/fits/asteroid04
```

Each fit writes a versioned octantoid state, OBJ/GLB mesh, optimization
history, per-camera metrics, residual heatmaps, and before/after plots for the
21 unique camera curves. The duplicated `hor2` measurements are intentionally
excluded. On CPU, complete sampled rotation curves are retained while camera
groups are cycled through Adam updates, preserving per-curve normalization.
Set `DSDIR` to either the challenge-data directory or its parent when the
dataset is stored outside this workspace.
If `artifacts/calibration/calibration.json` exists, fitting loads it
automatically; `--calibration` overrides that location.

## Architecture

- `OctantoidShapeModel` maps spherical-harmonic coefficients to a
  fixed-connectivity `TriangleMesh` and supplies the spectral prior.
- `PyTorch3DLightcurveModel` is the production differentiable renderer and
  implements direct illumination, cast shadows, perspective/orthographic
  cameras, and challenge lightcurve extraction.
- `scalar_fit.py` owns device selection, checkpoints, Adam fitting,
  calibration, metrics, and artifacts.
- `challenge_geometry.py` is the single source of camera names and asymmetric
  elevation angles used by both renderers and data loading.
- `asteroid-invert` exposes the workflow without duplicating fitting logic.

The older `PhotometricForwardModel` remains an independent CPU parity
reference. The Gauss--Newton driver in `experiments/gauss_newton.py` is
retained as an experimental comparison, not the default optimizer.

The optional `mitsuba_reference.py` module is a non-differentiable oracle. It
uses Mitsuba's scalar CPU variant, direct illumination, smooth normals, the
same outward view/sun convention as PyTorch3D, and a `-phase` body rotation.
Raw HDR integration is the default; challenge-style thresholding requires an
explicit HDR-to-display scale. Verify both optional reference dependencies and
the production renderer with:

```bash
uv sync --extra renderer --extra reference
uv run --extra renderer --extra reference asteroid-invert doctor --device cpu --reference
```

## Repository layout

- Supported fitting and rendering modules remain at the repository root.
- `tests/` contains the automated regression suite.
- `scripts/diagnostics/` contains the older CUDA-specific checks, while
  `scripts/validation/` contains standalone scientific inspection tools.
- `experiments/` contains the retained Gauss--Newton/LM comparison.
- `legacy/` contains unsupported raw-model experiments and archived outputs.
- `docs/` contains validation notes, development handoffs, and research
  references. Generated meshes and figures belong under ignored `artifacts/`.

## Legacy and compatibility paths

- `experiments/gauss_newton.py` is the retained Gauss--Newton/LM experiment.
- `forward_model.py` is the custom CPU parity renderer; new fits use
  `pytorch3d_forward.py`.
- `scripts/validation/visualize_match.py` is a standalone forward-model
  comparison; `visualize_fit.py` remains a supported report helper.
- The programs under `scripts/diagnostics/` reproduce older CUDA checks.
  Automated device-independent coverage lives in `tests/` and the doctor
  command.

Available shape models are `StarShapeModel` (per-vertex log radius) and
`OctantoidShapeModel` (coupled real spherical-harmonic `a`, `b`, and `c`
fields). The octantoid follows Kaasalainen & Viikinkoski (2012): its `a_00`
global-scale mode is omitted, while constant `b_00` and `c_00` terms are kept
and unregularized so triaxial ellipsoids have zero complexity cost.

`lmax=<n>` remains a shorthand for assigning the same maximum degree to all
three octantoid fields. Cartesian offset/detail fields are intentionally not
part of the pure octantoid; a future local-detail model can implement the same
`ShapeModel` protocol.

## Non-convex assumptions

Meshes must be closed, consistently oriented, and sphere-topology for the
single-body models. Soft visibility is evaluated at face centroids, so results
must be checked for convergence when the template is refined. Octantoid trial
steps are rejected for non-finite vertices, degenerate faces, and transverse
self-intersections. Normal-ranked face culling is still
available as an explicit approximation in the functional renderer but is off
by default.

Run the complete regression suite with:

```bash
uv run --extra renderer python -m unittest discover -s tests -v
```

## Renderer safety profiles

Differentiable rendering rejects meshes above 5,000 faces unless
`allow_large_mesh=True` is set deliberately. `RendererConfig.smoke()`,
`gradient_test()`, `cpu_optimization()`, `cuda_optimization()`, and
`evaluation()` encode staged profiles. Public STL files are never passed to
the differentiable renderer. Optional raw-mesh validation dependencies are
installed with `uv sync --extra renderer --extra reference`; they are imported
lazily so ordinary fitting and tests do not depend on Embree or Mitsuba.

## Optimization-resolution validation

`forward_validation.py` compares the differentiable renderer with the Blender
lightcurves for public asteroids 1--3 at optimization-like resolution. It
normalizes each STL to the challenge convention (`z=-1` to `z=1`) and samples
its outer radial surface on the same icosphere topology used by the star-shape
optimizer. This avoids conflating renderer error with destructive mesh
decimation. It then
uses 21 unique camera curves: Blender's `hor1` and `hor2` curves are identical,
so `hor2` is not counted twice.

The requested face count selects the nearest icosphere level (for example,
`1024` selects 1,280 faces). The default benchmark uses every sixth Blender
frame, with a higher-resolution override for asteroid 2 because its planar
sawed-cube geometry is particularly sensitive to visibility. For each public body,
preprocessing jointly fits one physical z-axis mesh rotation,
visibility softness, and Lambert/Lommel--Seeliger mixture against all 21
unique intensity curves. It then fits the binary threshold and slope while
holding that same rotation fixed. A mean of per-curve RMSEs is minimized so
the high-amplitude phase-angle curves do not overwhelm low-amplitude views.
No per-camera shifts are permitted.
The intensity renderer is unthresholded by default. The published 10% cutoff
is applied to tone-mapped video pixels, and cannot be transferred to linear
BRDF radiance without Blender's exposure and color-transform settings. An
explicit linear-radiance threshold remains available for controlled ablations.

```bash
uv run python forward_validation.py
```

Results are written to `artifacts/forward_validation/`. The main inspection
outputs are:

- `calibration_summary.png`: before/after RMSE and fitted coefficients.
- `phase_alignment_objectives.png`: the full shared-phase objective per body.
- `reflectivity_calibration_surface.png`: scattering/softness grid search.
- `asteroid*_intensity_before_after.png` and corresponding binary plots.
- `asteroid*_residual_heatmaps.png`: errors for all 21 camera geometries.
- `calibrated_mesh_orientations.png` and exported calibrated STL meshes.
- `calibration_parameters.json`, search CSVs, per-curve metrics, and the
  80/320/1280 requested-face convergence comparison.

`mesh_rotation_degrees` is the rotation applied directly to the supplied mesh;
`curve_shift_degrees` is its equal-and-opposite periodic lightcurve shift.
The former can be applied programmatically with `rotate_mesh_about_z`.

Useful variants are:

```bash
# Small end-to-end CPU smoke run
uv run python forward_validation.py --quick

# All 360 frames (consider using CUDA)
uv run python forward_validation.py --frame-stride 1 --device auto

# Select bodies/resolution and skip convergence
uv run python forward_validation.py \
  --asteroids 1,3 --target-faces 768 --no-convergence

# Override individual preprocessing mesh resolutions
uv run python forward_validation.py \
  --target-faces 512 --face-count-overrides 2:1536

# Calibrate on a deliberately restricted camera subset
uv run python forward_validation.py \
  --calibration-columns 0_hor1,0_top,90_hor1,90_top
```

The public STL files contain roughly 800k, 538k, and 6k faces, while the soft
visibility calculation is quadratic in face count. Proxy meshes are
therefore cached below the artifact directory. Use the convergence table to
check that scientific conclusions are stable before increasing resolution.
`--mesh-preprocessing qem` retains quality-first quadric decimation as an
explicit diagnostic, but it will refuse an unsafe target rather than forcing
asteroid 2 through a geometry-destroying aggressive decimation.
Large fitted rotations without a substantial RMSE reduction should be treated
as model inadequacy rather than genuine phase corrections. In the current
data this diagnostic flags asteroid 2, whose planar/non-convex geometry is
strongly resolution-sensitive under centroid soft visibility; asteroids 1 and
3 require only small rotations and align closely.

## Raw-mesh accuracy reference (legacy)

The scripts below are superseded by `forward_validation.py` and are preserved
unchanged in `legacy/`. They still run from that directory (they insert the
repository root on `sys.path`).

`legacy/raw_forward_validation.py` is the accuracy benchmark. It reads the
supplied binary Gmsh 2.2 nodes and triangle indices directly and never
centers, scales, welds, simplifies, or resamples them. Embree hard primary
rays and independent solar shadow rays make the 800,000-face asteroid 1 and
537,950-face asteroid 2 tractable without the differentiable renderer's
quadratic face-pair table. This path is a direct-light reference, not part of
optimization.

```bash
uv run python legacy/raw_forward_validation.py \
  --asteroids 1,2,3 \
  --columns 0_hor1,225_bottom \
  --resolution 256 --frame-stride 1
```

The challenge camera elevations are asymmetric: 0-degree top/bottom are
`+21/-21`; 45/90/135 are `+26/-24`; and 225/270/315 are `+24/-26` degrees.
Both renderers use this full per-camera table. Smooth shading is the default
because the older PyTorch3D implementation interpolated vertex normals; pass
`--shading flat` to make the reference use geometric triangle normals.

The reference supports Lambert, Lommel--Seeliger, and their mixture, and
records its exact settings beside each result. It intentionally has zero
indirect bounces. Jeppe's Mitsuba implementation also uses a diffuse material
with a `direct`/`direct_projective` integrator, so it contains cast shadows but
no surface-to-surface self-reflections. Test isolated path-traced scenes before
adding an indirect term to the inversion model; otherwise camera response,
projection, and shadow-boundary errors can be misidentified as interreflection.

Orthographic projection is the default. A finite-distance diagnostic is
available with `--projection perspective --camera-distance-radii <value>`.
Treat its distance as an ablation parameter: the updated Blender files do not
include authoritative camera intrinsics, so fitting it is not a physical camera
calibration.

The same projection options are supported by the differentiable renderer and
`forward_validation.py`. Perspective mode uses a pinhole image plane through
the mesh origin, per-face view directions, perspective-correct reciprocal
depth, and the image-plane projected-area Jacobian; illumination remains a
parallel directional source. For example:

```bash
uv run python forward_validation.py \
  --asteroids 3 --projection perspective \
  --camera-distance-radii 16 --target-faces 1024 --no-convergence
```

`--visibility-samples 3` enables three-point triangle quadrature for observer
and shadow visibility. It costs roughly two to three times more than centroid
sampling and is therefore opt-in; on the legacy asteroid-3 1,024-face benchmark it
did not materially improve mean RMSE.

`legacy/raw_model_sweep.py` jointly calibrates one global phase, scattering
mixture, and optional linear-response threshold against all 21 unique cameras.
Its default threshold grid includes zero; use the Hapke grid only as a
diagnostic for local multiple scattering, not as a substitute for geometric
path tracing.

## Forward-model vs data match visuals

`scripts/validation/visualize_match.py` shows how well a forward model
reproduces the supplied challenge lightcurves. For each public asteroid it
loads the observed data, renders the prediction on the 1,280-face star proxy,
phase-aligns the prediction to the data, and writes per-camera curve panels,
a residual heatmap, a per-camera metrics CSV, and a JSON summary (phase, mean
RMSE, mean correlation). Both the CPU `PhotometricForwardModel` and the
differentiable PyTorch3D renderer are supported via `--backend`.

```bash
# CPU forward model vs the simulated (Blender) intensity lightcurves
uv run python scripts/validation/visualize_match.py --asteroids 1,2,3 --backend cpu

# GPU differentiable renderer vs asteroid 3's simulated intensity curves
uv run python scripts/validation/visualize_match.py --asteroids 3 --backend gpu

# Real-video binary curves, coarser sampling
uv run python scripts/validation/visualize_match.py --asteroids 1 --origin real \
  --curve-type binary --frame-stride 12
```

Outputs land in `artifacts/forward_match/` (override with `--output-dir`).
The Blender lightcurves have one row per rotation degree, so the phase is
expected to be small; for real-video lightcurves the frames are mapped
linearly to one rotation and the phase search absorbs the unknown starting
position. Fit hyperparameters such as `--gamma` and `--tau-intensity` default
to the calibrated values; pass the `calibration_parameters.json` produced by
`forward_validation.py` for the exact settings used there.
