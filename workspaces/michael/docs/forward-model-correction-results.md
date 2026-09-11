# `225_bottom` correction campaign

Completed 2026-09-08. **Promotion verdict: FAIL.** All ten 300-step fits
completed; asteroid-2 channel RMSE improves 17.18%, below the required 20%.
Its pooled error improves, asteroid 3 passes the regression check, and the
saved-state audit finds no numerical failures. Corrected settings remain experimental.
The current machine-readable verdict and fit table are in the
[campaign report](../artifacts/corrections_225_bottom_20260908/report.md).
Historical tuning and diagnosis artifacts are preserved.

## Implemented behavior and verification

- Complete-set training normalization with two-pass, memory-bounded gradient
  accumulation. Validation renders all 360 phases, normalizes once, then scores
  the saved validation indices. Training chunks never define normalization sets.
- Experimental 72/72 phase partition: training `0,5,...,355`, validation
  `2,7,...,357`. Reports use the same remaining 216 indices for all variants,
  including the 12-phase legacy control.
- Explicit linear/sRGB response, positive radiance gain, perspective/orthographic
  projection, and body-wide frame-label/zero-based phase convention. Both exact
  and soft extraction receive the encoded image. Linear/gain-one legacy defaults
  remain available; old calibration files load with a warning. Every calibration
  version requires an explicit transfer choice for an uncovered body, including
  smoke runs and saved fit configurations.
- Versioned calibration and checkpoint files, dataset SHA-256 identities,
  configuration/code fingerprints, saved splits and initial coefficients, and
  content-based mesh caches. Incompatible optimization resumes are rejected.
- Full float32 camera matrix multiplication. A previous import enabled TF32
  globally; the resulting single-frame camera transforms differed from batched
  transforms, changing shadow depths by up to 0.00170 in the fixed-camera test.
  Full precision fixes the batch dependence without changing shadow bias,
  softness, or visibility physics.

The GPU-enabled suite passed 58 tests. Three additional targeted GPU checks
passed for a planar occluder, sphere rotational invariance, and nonzero finite
orthographic/sRGB shape gradients. The suite includes response breakpoints,
threshold ordering, phase indexing, normalization, serialization, cache identity,
chunked-gradient equivalence, and resumed versus uninterrupted optimization.
CUDA scatter reductions are compared with floating-point tolerances rather than
requiring bitwise trajectories.
An additional CPU test replaces a source STL with different geometry of the
same byte size and verifies that the public-proxy cache is rebuilt. These 62
checks all passed across the full suite and targeted follow-up runs. Two added
CLI regressions verify explicit calibration transfer in smoke mode and in a
saved fit configuration; the full test inventory contains 64 cases, all passed
across the full suite and follow-up runs. After the final scope-guard change,
seven focused checks passed, including the legacy serialization guard, all CLI
checks, and resumed-versus-uninterrupted CUDA optimization
([log](../artifacts/corrections_225_bottom_20260908/final_checks.log)).
Compilation and `git diff --check` also pass.

The [saved-state audit](../artifacts/corrections_225_bottom_20260908/audit_as_run.json)
passes all ten runs: 300 updates each, finite parameters/optimizer states/
trajectories, exact paired initializations, unchanged target hashes and frozen
core code, identical validation partitions, and selected-checkpoint consistency.
Both second starts contain 24 nonzero degree-1/2 perturbations only, with
specified Gaussian standard deviation 0.01 and realized sample deviation 0.00819.

## Data and response control

The unchanged simulated intensity targets have SHA-256 identities:

```text
A2 7599774eb0b4adf14ccb977962452d2fa674b455872ba34829de13df178cb041
A3 3d69c46fcfde2f17c04f10a136821a3ea2bad7c7e0dc1af8ee3ed6c96a7ab856
```

Applying the production response helper to the independent raw-ray orthographic
reference reproduces the known video improvement. Six freshly rendered phases
per body match the archived linear endpoint images. The decoded movie is a
separate forward-model control and never replaces the text target.

| Orthographic `225_bottom` control | Linear RMSE | sRGB RMSE |
| --- | ---: | ---: |
| Asteroid 2 | 0.435118 | 0.136604 |
| Asteroid 3 | 0.067675 | 0.027522 |

These scores pair the first video frame with zero rotation. No channel-specific
phase alignment is fitted.

The production soft threshold remains an approximation to exact extraction.
On the selected calibrated public proxies, the full-turn soft/exact pooled RMSE
is 0.08831 for A2 and 0.00742 for A3. Soft predictions score 0.17266 and 0.03931
against the text, compared with exact scores of 0.16192 and 0.03721. The response
transform is shared; the threshold surrogate accounts for this remaining gap.
Exact extraction is used for checkpoint selection and the improvement gate.

![Orthographic response control](../artifacts/corrections_225_bottom_20260908/figures/video_a2.png)

## Calibration and phase coverage

Each body screens the full documented 420-candidate grid separately at 72 phases.
The top three, legacy candidate, and shared-grid candidate are checked at all
360 phases and 21 cameras. Endpoint mixing is checked against direct renders;
the absolute and relative tolerances are both `5e-5`, well below an 8-bit level.
Per-group measured errors are retained in `mixing_checks.jsonl`.

| Body | Selected settings | Legacy pooled | Selected pooled | Legacy `225_bottom` | Selected `225_bottom` |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | Perspective 5 radii; mixture 0.5; sRGB; gain 0.5; zero-based phases | 0.367456 | 0.161920 | 0.832744 | 0.235200 |
| 3 | Orthographic; mixture 0.75; sRGB; gain 1; zero-based phases | 0.059155 | 0.037210 | 0.099400 | 0.062745 |

Neither finite-distance boundary wins. The shared-grid candidate worsens pooled
RMSE by 8.7% for asteroid 2 and 47.6% for asteroid 3 relative to the separate
calibrations, so sharing is rejected. These are empirical acquisition models;
no physical camera distance or Blender color-management setting is recovered.

The original sampling reversal is reproduced:

| Asteroid-2 sampling / subset normalization | 3.5-radii pooled RMSE | 16-radii pooled RMSE |
| --- | ---: | ---: |
| 12 phases | 0.385796 | 0.259812 |
| 72 phases, offset 0 | 0.274432 | 0.384799 |
| 360 phases | 0.261032 | 0.367456 |

All five offsets of the 72-phase grid prefer 3.5 over 16 for asteroid 2 and
16 over 3.5 for asteroid 3. Asteroid-2 `225_bottom` target means range from
0.9690 to 1.0151 across these offsets; the 12-phase mean is 0.5089. The broader
training set reduces the sampling bias, but does not make its mean identical to
the full-turn mean. Complete-turn validation removes subset renormalization
from checkpoint scoring.

![Five sampling offsets](../artifacts/corrections_225_bottom_20260908/figures/sampling_offsets.png)

## Geometry and visibility controls

These controls use common z-normalized coordinates and a fixed camera/framing
radius for every shape of each body: the raw reference radius (1.73205 for A2,
1.09948 for A3). This differs from production's per-shape radius gauge. A2's
public proxy radius is only 1.54660; consequently these fixed-camera scores
must not be compared directly with the radius-relative calibration table.

All three octantoid fields are fitted to public proxy vertex positions for
1,200 Adam steps with a global reporting scale. These are correspondence-based
geometry diagnostics, not proofs of the globally best representable surface.
They never initialize photometric inversion. Degree changes precede the mesh
resolution change; a memory preflight bounds the 5,120-face comparison.

| A2 fixed-camera control | Pooled RMSE | `225_bottom` RMSE | Mean predicted brightness when target < 0.1 |
| --- | ---: | ---: | ---: |
| Raw reference, hard rays | 0.14238 | 0.07281 | 0.03007 |
| Public proxy, hard rays | 0.26417 | 0.60179 | 0.43590 |
| Public proxy, production shadow maps | 0.27189 | 0.62253 | 0.46105 |
| Degree 6, 1,280 faces | 0.30542 | 0.69743 | 0.46603 |
| Degree 10, 1,280 faces | 0.31024 | 0.72069 | 0.50630 |
| Degree 10, 5,120 faces | 0.28003 | 0.63964 | 0.41993 |

A2 has a substantial raw-to-proxy gap. Increasing harmonic degree improves
geometry distance but not this photometric residual. Increasing resolution at
fixed degree helps this control without closing the gap. These results do not
justify promoting added capacity or reducing regularization in production.
A3's pooled errors are 0.01572 for the raw ray reference, 0.03465 for the ray
proxy, 0.03830 for the production proxy, 0.07288 for degree 6, 0.05599 for degree
10, and 0.05867 for degree 10 at 5,120 faces.

Fixed-camera sphere, box and sawed-cube tests compare hard solar rays against
shadow maps and smooth against flat reference normals. Image resolution,
shadow resolution, bias and softness vary separately. The results show
approximation sensitivity, not support for a general shadow-setting change.
Production visibility settings are retained. No additional scattering physics
is introduced.

Full per-camera metrics, dense radial geometry errors, meshes, runtime and
memory measurements are in `controls/geometry.json` and `controls/parity.json`.

## Inverse experiment and promotion gate

Ten 300-step fits use the historical winning staged-Adam schedule: three
variants on each body from the first start, then legacy and combined variants
from a second shared start. The second start adds seeded Gaussian noise with
standard deviation 0.01 only to degree-1/2 coefficients. Base coefficients are
saved once per body. Calibrations and configurations are frozen before fitting.

The promotion gate uses paired-start means: at least 20% lower A2 `225_bottom`
RMSE, lower A2 pooled RMSE, no more than 5% worse A3 pooled RMSE, and no numerical
failures. Validation and remaining rotations are **conditional on calibration
using public full-turn data**; this is not an end-to-end generalization test.

The asteroid-2 paired results miss the required channel improvement:

| Asteroid-2 metric, mean of two starts | Legacy | Combined |
| --- | ---: | ---: |
| Full-turn `225_bottom` RMSE | 1.317035 | 1.090786 |
| Full-turn pooled RMSE | 0.593259 | 0.483882 |
| Validation RMSE, 72 fixed rows | 0.591713 | 0.482114 |
| Remaining-rotation RMSE, 216 common rows | 0.593903 | 0.484380 |

The channel improves **17.18%**, below the agreed **20%** gate; pooled RMSE
improves **18.44%**. Channel improvement is 8.2% for start 0 and 25.9% for
start 1, so reporting only the better initialization would give a misleading
promotion decision. The corrected configuration remains experimental.
Asteroid 3 passes the regression gate: its paired pooled RMSE improves
**10.87%**, from 0.114147 to 0.101736 (the limit allowed at most 5% worsening).

| Asteroid-3 metric, mean of two starts | Legacy | Combined |
| --- | ---: | ---: |
| Full-turn `225_bottom` RMSE | 0.213455 | 0.157197 |
| Full-turn pooled RMSE | 0.114147 | 0.101736 |
| Validation RMSE, 72 fixed rows | 0.113794 | 0.101706 |
| Remaining-rotation RMSE, 216 common rows | 0.114188 | 0.101762 |

The four gate outcomes are: A2 channel **fail**, A2 pooled **pass**, A3
regression **pass**, numerical finiteness **pass**. The gate uses means of
per-run RMSE, not the RMSE of an ensemble-averaged curve. No end-to-end
generalization or statistical population claim follows from these two starts.

Coverage alone worsens first-start pooled RMSE on both bodies: 0.597734 to
0.690101 for A2, and 0.112970 to 0.120712 for A3. Removing sampling bias does
not by itself make this finite-step optimization find a better shape.

Asteroid-2 channel correlation, averaged over starts, increases from 0.2045
to 0.6246. Mean predicted brightness over the 198 observed dark samples
(target below 0.1) falls from 0.8899 to 0.7070. The substantial false
brightness floor remains. All four main observed peaks are still too small:

| Observed index / amplitude | Corrected start 0: offset / amplitude | Corrected start 1: offset / amplitude |
| --- | ---: | ---: |
| 79 / 3.871 | −9° / 1.955 | −6° / 2.401 |
| 169 / 3.871 | −9° / 1.653 | −9° / 1.707 |
| 259 / 3.508 | −5° / 1.750 | −5° / 1.814 |
| 349 / 3.076 | −10° / 1.087, window boundary | −6° / 1.565 |

These are maxima in a common circular ±10° window around each observed peak;
a boundary maximum does not establish a matched local peak. Global peak
indices and amplitudes, corresponding legacy values, and every camera are
also exported. The nearly equal observed peaks at indices 79 and 169 make a
single global-argmax timing difference less informative than the local table.

![A2 before/after, start 0](../artifacts/corrections_225_bottom_20260908/curves_a2_start0.png)
![A2 before/after, start 1](../artifacts/corrections_225_bottom_20260908/curves_a2_start1.png)

A2's mean radial-log geometry RMS against the public proxy changes from
0.26874 to 0.27560, a small worsening despite the photometric improvement.
It worsens for start 0 and improves for start 1. These geometry errors are
post-fit diagnostics and never contribute to checkpoint selection.

Optimization drifts after its best checkpoint in both A3 combined fits.
Start 0 selects step 50 (validation 0.09966), and start 1 selects step 60
(validation 0.10375). Their step-300 validation errors are 0.57709 and 0.54138.
All values remain finite. The saved convergence histories preserve this
behavior, and final scores use the selected checkpoint, as specified.

All ten fits have all-camera metrics, 360-sample curves, a portable
trajectory, meshes, plots and a checkpoint. Consolidated
[per-camera metrics](../artifacts/corrections_225_bottom_20260908/fit_camera_metrics.csv),
[local peak metrics](../artifacts/corrections_225_bottom_20260908/peak_metrics.csv)
and [per-run scores](../artifacts/corrections_225_bottom_20260908/results.csv)
accompany the report.

## Common-model and historical comparisons

All twelve saved meshes (ten campaign fits and two historical winners) are
rescored at all 360 phases and 21 cameras. Each body uses its selected image
model, one fixed raw-reference framing radius, and one global shape scale
aligned to the public proxy. These scores isolate saved-shape changes from
differences in the forward settings; they do not replace the specified gate.

| Body / saved shape | Common pooled RMSE | Common `225_bottom` RMSE | Radial-log geometry RMS |
| --- | ---: | ---: | ---: |
| A2 historical winner | 0.48482 | 1.14864 | 0.26906 |
| A2 legacy paired mean | 0.54672 | 1.17154 | 0.26874 |
| A2 combined paired mean | 0.48007 | 1.07641 | 0.27560 |
| A3 historical winner | 0.11516 | 0.20414 | 0.55982 |
| A3 legacy paired mean | 0.11025 | 0.20609 | 0.55463 |
| A3 combined paired mean | 0.10313 | 0.15907 | 0.57524 |

A2's corrected common-model pooled score is only about 1% below the historical
winner. Geometry error worsens slightly in the paired means for both bodies.
The full per-camera comparisons, mesh hashes and common framing radii are in
[common_evaluation.json](../artifacts/corrections_225_bottom_20260908/controls/common_evaluation.json).

Historical meshes also receive a full-turn rescore using their legacy image
settings: A2 pooled/channel RMSE is 0.54057/1.19800, and A3 is
0.11423/0.21732. These are freshly computed scores, not the historical study's
subset-normalized validation numbers. In particular, the new A2 legacy controls
do not recover that historical winner's score. The planned paired gate remains
the controlled comparison of matched starts.

The fixed-model degree-6 geometry diagnostics score 0.30542 pooled for A2 and
0.07288 for A3, below the combined inverse means of 0.48007 and 0.10313.
Together with the raw-to-proxy gap, this leaves limitations in both geometry
and the current inverse optimization. It does not establish a need for new
scattering physics. Higher-resolution inversion, broad retuning, and reduced
regularization are not promoted.

## Runtime and reproduction

Ten representative steps, including escape and refinement updates, were timed
before fitting. Six-frame accumulation reduces corrected step time from about
13.1 seconds to 6.3 seconds, with about 1.10 GiB peak allocated GPU memory.
The ten-fit projection is 4.53 hours; actual fit time totals **4.570 hours**
(16,453 seconds), with maximum allocated GPU memory **1,127.2 MiB**. The campaign targets eight hours of compute
and preserves a checkpointed stop before the 9.5-hour ceiling; fits have an
additional 5.5-hour wall-time limit. The conservative measured envelope from
the first benchmark through final GPU verification is **5.76 hours**, including
intervening engineering time, below both limits. High-resolution fitting and broad retuning
are not part of this campaign.

Run from the workspace with its installed Python environment:

```bash
python -m hac_asteroids.correction_campaign benchmark --output-dir <new-run>
python -m hac_asteroids.correction_campaign calibrate --output-dir <new-run>
python -m hac_asteroids.correction_controls video --output-dir <new-run>/controls
python -m hac_asteroids.correction_controls sampling --output-dir <new-run>/controls
python -m hac_asteroids.correction_campaign fits --output-dir <new-run>
python -m hac_asteroids.correction_controls geometry --output-dir <new-run>/controls
python -m hac_asteroids.correction_controls parity --output-dir <new-run>/controls
python -m hac_asteroids.correction_controls common --output-dir <new-run>/controls
python -m hac_asteroids.correction_figures <new-run>
```

The video control explicitly verifies archived diagnostic endpoints and records
their hashes; it requires those external artifacts. Other large reference and
render data stay outside source control. A small numeric video-curve fixture is
retained in `tests/fixtures/orthographic_response.json`. The as-run source archive,
software versions, configurations, fingerprints and dataset identities accompany
[the experiment](../artifacts/corrections_225_bottom_20260908/).
`source_snapshot.tar.gz` preserves the fitted code. The separate
`source_snapshot_final.tar.gz` includes the final scope guard, CLI fixes,
reporting, tests and documentation. The guard changes no numerical setting for
bodies 2/3, but the conservative code fingerprint rejects old-code resumes;
use the as-run source to replay those checkpoints. Portable trajectories
remain readable with the final code.
