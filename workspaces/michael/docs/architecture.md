# Maintained architecture

The supported user interface is `hac-asteroids`:

- `doctor` checks whether the current process can allocate on CUDA.
- `validate` calibrates fixed camera distance and scattering mixture against
  public proxy meshes.
- `fit` reconstructs one or more asteroids from lightcurves.
- `tune` selects one staged optimizer configuration on held-out rotations from
  public asteroids 1--3.
- `animate` turns a portable trajectory into an MP4 or GIF.

`hac_asteroids/` owns dataset discovery, orchestration, run manifests,
trajectory persistence, plots, and command-line behavior. The numerical core
remains in `shape_model.py`, `pytorch3d_forward.py`, `forward_model.py`, and
`forward_validation.py`; these modules have focused regression coverage.

The old raw-ray benchmarks, hard-coded demonstrations, and one-off GPU scripts
were development scaffolding and are not supported entry points. Their useful
invariants are represented by the unit and CUDA integration tests.

Generated runs belong under `artifacts/`, which is intentionally ignored.
Challenge input files are read-only and remain under `challenge_data/`.

The inversion objective combines normalized photometric error with a single
degree-weighted spherical-harmonic penalty. The constant scale gauge and
triaxial-ellipsoid coefficients remain unregularized. Checkpoint selection uses
disjoint validation rotations; public reference meshes are reporting-only and
never influence tuning.
