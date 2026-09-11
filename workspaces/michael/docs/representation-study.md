# Controlled representation study

## Octantoid degree campaign

`--profile octantoid` runs nine independent sphere starts: L2, L4 and L6 on
each of asteroids 1–3 (26, 74 and 146 parameters). All starts use the same
1,024-face sphere, seed 0, zero coefficients, data split, 96-pixel renderer
and common spectral penalty described below. All three existing calibrations
are reused after dataset/provenance checks; no calibration screening is run.

The preflight measures all nine combinations using disposable parameters,
three updates spanning optimization stages, and a full-turn evaluation.
It chooses the largest common cap in 300,320,…,500 satisfying an eight-hour
projection: elapsed preparation + 1.1 × projected fitting/evaluation time +
30 minutes reporting. The estimate assumes every fit reaches its cap. If
300 iterations is infeasible, no production fits launch. The selected cap is
frozen in `study.json` and `benchmark.json`; runtime is an estimate, not a
hard deadline. The GPU executes fits sequentially.

There are 40 degree-limited escape updates, refinement, and 40 final
stabilization updates. Validation occurs every 20 updates. The selected
checkpoint always minimizes validation RMSE, independently of early stopping:

- Initialize a plateau reference at update 40. Meaningful improvement is a
  reduction of at least max(0.0001, 0.005 × reference RMSE); smaller reductions
  accumulate against that reference. Meaningful improvement resets patience.
- After at least 200 updates, 100 updates without meaningful improvement
  initiate 40 confirmation updates. Cap the learning rate at one-fifth of
  the entry rate, bounded below by the configured minimum. Keep scheduled
  regularization. The cap remains in effect after confirmation recovery.
- Meaningful improvement during confirmation resumes monitoring; otherwise
  finish with `validation_plateau`. The iteration cap takes precedence and
  is reported as `iteration_cap` even if reached during confirmation.

Checkpoints preserve plateau state and optimization diagnostics; resume must
match the objective and stopping configuration. Early stopping is disabled
by default for other profiles. Audits use actual completed updates, including
terminal objective evaluation, rather than assuming every fit reaches its cap.

Each asteroid has selected and terminal lightcurves (`lightcurves_last.png`),
selected 3D shapes, convergence/deformation plots, individual selected/full
GIFs and a synchronized GIF. A stopped model holds its last geometry, labelled
with its actual iteration and reason. Complete trajectories retain every
state; sampled GIFs retain initial, selected and terminal states. CSV/report
rows include actual updates, cap, reason, photometric and geometry metrics.
Plateau stopping does not demonstrate global optimality or faithful shape
recovery. The existing full/quick profiles remain available as described below.

```bash
python -m hac_asteroids.representation_study run --profile octantoid \
  --output-dir artifacts/octantoids_l2_l4_l6_20260910
```

## Original full and quick comparisons

Each of asteroids 1, 2 and 3 receives three independent 500-update inversions:
radial harmonics at L=6 (48 parameters), radial harmonics at L=11 (143), and
octantoids at L=6 (146). One saved 1,024-face sphere supplies all nine starts.
Its 514 vertices have radius 0.5; all initial coefficients are zero.

The `--profile quick` alternative runs only asteroid 3, with the same three
representations and 1,024-face sphere. Each fit receives 120 updates: 12 escape,
96 refinement and 12 stabilization updates. Validation occurs every 20 updates.
All data coverage, rendering settings, learning-rate endpoints, spectral
regularization and initialization remain as specified below. The quick study
targets roughly one hour including figures; it is a screening budget, not a
hard wall-time cutoff or a claim of convergence. It stores all 121 states.

## Mathematical controls

Both models use the existing real orthonormal spherical harmonics. For a unit
template direction u and h(q)=3 tanh(q/3), the radial model is
v=0.5 u exp(h(a(u))). Its constant coefficient is omitted. The octantoid is
v=0.5 u ⊙ exp(h([a,a+b,a+c])); constants of b and c remain free. Thus radial L6
is exactly the octantoid subfamily with b=c=0. Radial L11 provides a similar
coefficient budget without the octantoid's independent coordinate fields.

The spectral term is λ s(k) Σ l³ c² / 146 for **all three models**. Shared
coefficients therefore receive identical penalties. The octantoid's constant
b/c coefficients remain unpenalized, an intrinsic difference of this prior.
No claim of equal functional priors or independently optimal hyperparameters
is made. The shared spectral weight is 0.0019962845027891255.

The historical winning Adam recipe uses initial learning rate
0.004467358793366061, refinement rate 0.002 and minimum rate 0.00001, with
40 low-degree escape updates, refinement through update 450, and 50 final
stabilization updates. Degree clipping and the regularization stage multiplier
follow the maintained fitter. Seed is 0; no initialization noise is added.

## Data and selection

Use all 21 simulated intensity camera channels and the existing deduplication
checks. Train at indices 0,5,…,355; validate at 2,7,…,357. Training uses
complete-training-set normalization with two-pass chunked derivatives.
Validation renders and normalizes the full turn before selecting its rows.
All reports evaluate 360 phases; the remaining-rotation subset has 216 rows.

The corrected historical calibrations for asteroids 2/3 are copied unchanged.
Asteroid 1 uses the existing 420-candidate per-body screening and full-turn
finalist verification, including its existing boundary extension behavior.
Calibrations are frozen across representations. These are empirical,
experimental models; validation is conditional on public full-turn calibration.

The selected output minimizes exact-renderer validation RMSE among the sphere
and checkpoints every ten updates. All fits continue to update 500. Raw final
updates and selected-best states are reported separately. Geometry never
enters the inverse loss or checkpoint selection.

Post-fit replay audits record the maximum/RMS curve difference and allow
absolute/relative tolerances of 1e-4 for float32 GPU scatter variation. Original
saved curves remain the canonical selected scores. A short end-to-end check
observed a maximum discrepancy of 6.04e-5 at one sample and a pooled discrepancy
of 7.3e-7; the target RMSE changed by about 1e-8.

## Geometry and visualization

Measure outer radial support against the normalized raw public mesh along
4,096 common Fibonacci directions. Align one global scale by minimizing
squared log-radius differences; perform no rotation or translation fitting.
Report RMS/max log-radius error and any missed rays without imputation. This
metric describes the outer radial surface, not hidden cavities or every
possible non-star-shaped feature.

Measure radial and tangential displacement relative to the initial template
directions, in units of its radius. Inspect selected and last meshes for
degenerate triangles and transverse self-intersections. Invalid geometry is
flagged without hiding its photometric score.

The three-model figures compare all cameras, convergence, deformation and
selected shapes. The display reference is a fixed, labelled 1,280-face proxy;
geometry scoring uses the raw reference. GIFs sample every five updates and
always retain selected-best and terminal states. Individual selected-best GIFs
end at the selected state; full and synchronized histories reach update 500.
Views and limits are fixed, and labels give actual iteration numbers. Portable
trajectories retain all 501 states.

## Execution and recovery

Run `python -m hac_asteroids.representation_study run --output-dir <directory>`.
Add `--profile quick` for the asteroid-3 screening comparison. The default
`full` profile preserves the nine 500-step fits. Profiles are recorded in
`study.json` and cannot be switched inside an existing campaign directory.
The `prepare`, `benchmark`, and `report` actions expose individual stages.
CUDA is required. The benchmark tests all three representations using asteroid
2 and measures ten updates spanning all stages plus a full-turn evaluation.
The quick profile instead reuses only asteroid 3's calibration and times three
updates at schedule indices 0, 12 and 108, plus a full-turn evaluation for each
model. Both profiles derive runtime projections and report counts/labels from
their actual configurations.
Six-frame accumulation is preferred; an out-of-memory result retries every
variant at a common three-frame, then one-frame block size. This changes memory
partitioning, not training samples or normalization. Runtime projections exclude
calibration, reports and retries and are labelled as asteroid-2 measurements.

The frozen configuration records the template, data, calibration, recipe and
numerical-source identities. A source archive preserves the implementation.
Run sequentially by asteroid and representation; produce its comparison report
after each group of three. Checkpoints are atomic and written every 25 updates.
Interruptions retain the latest checkpoint; rerunning the same command resumes
it. At most 24 unsaved updates can be repeated. A file lock prevents concurrent
campaign writers. Failures are recorded, other fits continue, and the final
status stays incomplete until all nine measurements and three reports exist.
Attempt records preserve total fitting wall time across orderly interruption
and resume, including repeated updates. Reported runtimes sum those attempts.

For a frozen campaign that must continue independently of its launching
terminal, use `python -m hac_asteroids.campaign_runner --output-dir <directory>`.
The detached supervisor appends to the campaign log and writes
`runner_status.json` every 30 seconds, including a timestamp, process memory,
completed-fit count and eventual exit code. Locks prevent duplicate fits.
The same command resumes saved checkpoints after a machine/WSL restart;
detachment itself cannot keep computation running while the machine is down.
The supervisor does not change the frozen numerical configuration or source.

## Local L4 survey for the report

The separate `hac_asteroids.l4_survey` driver supports `prepare`, `run`, and
`report`, each requiring `--output-dir`. It leaves the earlier representation
study profiles and their benchmark identities unchanged. Prepare the survey,
then launch `hac_asteroids.campaign_runner` with the same output directory for
detached execution. Run only one campaign on the local GPU at a time.

The survey fits asteroids 1–10, in order, with octantoid L4 (74 coefficients),
1,024 faces, zero coefficients, and seed 0. Every fit performs exactly 200
updates: 40 escape, 120 refinement, and 40 stabilization, with early stopping
disabled. Historical learning rates, regularization, 96-pixel rendering,
six-frame training blocks, all 21 cameras, and the existing disjoint 72/72
training/validation phases are retained. Validation runs every 20 updates;
reported predictions cover all 360 phases. The estimated total is 4.6–5.1
hours on the RTX 3080, excluding implementation and setup.

Asteroids 1–3 reuse their individual calibrations. Asteroids 4–10 explicitly
transfer asteroid 3's calibration; its original scope and provenance are
preserved. Captions, GIFs, reports, and CSV rows mark this assumption as
provisional. Hidden-body geometry scores are null, and their inferred shapes
are examples conditional on these rendering settings. No hidden reference is
loaded or invented. Public geometry scoring remains on the CPU.

Each completed fit produces initial/selected/terminal shape views, selected
and terminal STL meshes and all-camera lightcurves, convergence and deformation
plots, and a full-history GIF with fixed axes and iteration labels. GIFs sample
every five updates while retaining selected and terminal states; trajectories
store all 201 states. An artifact manifest hashes the delivered files, verifies
links, and decodes images/GIFs before marking the report complete. Valid reports
are skipped on resume; damaged reports are regenerated. The aggregate report,
CSV, and PNG/PDF shape overview update after each asteroid. The campaign is
complete only after every fit and report passes its audit.

Use a fresh output directory: changing the iteration cap changes the learning
rate schedule, so earlier 360-update results cannot be relabelled as 200-update
fits. Checkpoints are written every 25 updates. An orderly interruption keeps
the latest checkpoint; up to 24 unsaved updates may repeat on resume. Detached
execution survives terminal exit, but the machine must remain awake.
