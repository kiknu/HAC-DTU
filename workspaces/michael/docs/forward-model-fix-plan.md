# Forward-model correction plan

Proposed 2026-09-08. **Implementation and verification are recorded in the
[correction results](forward-model-correction-results.md).** The original plan
below is retained as the experiment protocol; rollout depends on the paired gate.
This plan follows the [asteroid-2 residual diagnosis](asteroid2-residual-diagnosis.md).
It targets the maintained PyTorch3D intensity workflow, starting with public
asteroids 2 and 3. Preserve the completed tuning study as the historical baseline.

## Findings that determine the plan

| Finding | Consequence |
| --- | --- |
| Fresh meshes and all eight A2/A3 lightcurve files match local SHA-256 hashes. | Do not replace or “repair” the current inputs. |
| A2's text does not closely match either published `225_bottom` movie; A3's orthographic movie does. | Record video/text provenance separately; do not silently substitute video-derived curves. |
| Twelve calibration phases have A2 `225_bottom` mean 0.509 rather than 1.000 and reverse the camera-distance ranking. | Remove the 12-phase calibration default and broaden training coverage. |
| On the public proxy, changing distance 16 to 3.5 lowers full-turn channel RMSE from 0.833 to 0.488. | Recalibrate projection, but do not hard-code 3.5 or assume the same camera distance for both bodies. |
| Applying sRGB before thresholding lowers the raw-mesh reference error from 0.621 to 0.166. | Separate linear radiance from video-pixel formation and test response models explicitly. |
| Coarse geometry creates a false brightness floor; pure Lambert shading alone does not fix it. | Test representation and visibility before introducing multiple scattering. |

These effects are not additive. The raw-mesh score is a known-shape diagnostic,
not an achieved or guaranteed inverse-reconstruction score. sRGB is a tested
candidate, not a recovered Blender setting. A one-degree frame/pose ambiguity
also remains; the documented azimuths and rotation sign have no evidence of a
gross error.

## 1. Establish a reproducible baseline and correct phase coverage

- Promote the useful diagnostic controls from ignored run artifacts into a
  maintained, read-only validation harness. Keep large videos, meshes, and
  rendered caches outside source control; retain small numeric fixtures for CI.
- Record dataset SHA-256, real/simulated origin, observable, projection family,
  frame labels, phase convention, mesh preprocessing, renderer settings, and
  code/configuration fingerprints in calibration and fit manifests. Cache keys
  must include these fingerprints, not just source file size.
- Keep the original text data as the inversion target. Treat each video family
  as a separate validation dataset. Draft questions for the organizers about
  the movie corresponding to each July text column, initial pose, camera
  intrinsics, Blender material, exposure, and color management. Sending those
  questions is a separate user-authorized action; it does not block local tests.
- Use all 360 phases and all 21 cameras for final public-shape calibration.
  A 72-phase sweep may screen candidates, but cannot select the final winner
  without a full-turn check. Include five offsets of the 5-degree grid in the
  sampling stability diagnostic.
- Initially use 72 training phases and 72 disjoint validation phases for new
  360-frame fits: training indices `0,5,...,355`, validation indices
  `2,7,...,357`. Keep the remaining 216 samples out of checkpoint selection.
  Preserve explicit saved indices for replay and retain the small smoke mode.
- Normalize the training loss over its complete 72-frame set, never over
  individual render chunks. For validation and final reporting, render raw
  brightness over the full turn, normalize once, then select the scored indices;
  compare against the supplied full-turn-normalized observations. This removes
  subset-dependent score rescaling without using held-out target values in the
  training loss. Measure the remaining training-mean approximation explicitly.
- Keep memory bounded by recomputing render chunks for gradient accumulation.
  First collect raw training brightness without a renderer graph; differentiate
  the normalized loss with respect to that small brightness array; then rerender
  chunks and backpropagate those brightness derivatives. Add the prior once,
  mask degrees and clip gradients once, and perform one Adam update. Test this
  two-pass gradient against an unchunked reference, including mean derivatives.

**Gate:** reproduce the documented 12/360 ranking reversal; prove chunk-size
invariance; report sample-mean error and ranking stability on all five offsets.
Do not call the sparse approximation fixed merely because the frame count grew.

## 2. Make image formation and calibration explicit

The production image path should become:

```text
linear shaded radiance → gain → response transform → clamp to [0,1]
                      → exact or soft pixel extraction → curve normalization
```

- Add explicit `linear` and standard piecewise `srgb` response choices and a
  positive radiance gain. Apply the same response before both soft training
  extraction and exact 8-bit evaluation. Preserve the existing linear/gain-1
  behavior as an explicit legacy configuration; do not enable sRGB globally
  before the calibration gates pass. Keep threshold 25.5/255 after encoding.
- Make the simulated frame convention explicit: legacy frame-label degrees or
  zero-based rotation phase. Test both conventions against the public meshes
  and decoded movies, using one offset per body across all cameras. Do not
  shift individual columns to improve a fit or silently reinterpret old runs.
- Support an explicit orthographic camera for known orthographic movies.
  Keep the current radius-relative perspective convention in the first release,
  label it as such, and log effective camera distance and framing. Raw/proxy
  parity controls must additionally hold camera position and framing fixed in
  common z-normalized coordinates; otherwise geometry and camera changes mix.
  Replacing the inverse solver's scale gauge is a separate change, not a hidden
  part of this correction.
- Calibrate the two bodies separately before assessing whether parameters can
  be shared. Screen perspective distances `{2.5,3.5,5,8,16,32}` plus orthographic,
  Lambert fractions `{0,.25,.5,.75,1}`, responses `{linear,srgb}`, gains
  `{.5,1,2}`, and the two frame conventions. Reuse linear Lambert/LS endpoint
  images for material/response sweeps on a fixed geometry. Verify cached mixing
  against direct renders before relying on it.
- Select each body's best screened candidate **and the legacy candidate** by
  pooled full-turn RMSE over all cameras, reporting mean camera RMSE,
  `225_bottom`, correlation, and dark-interval error alongside it. Recheck the
  top three at all 360 phases. If a finite distance boundary wins, evaluate
  one outward extension (2 or 64 radii, as appropriate) and flag a remaining
  boundary optimum; do not label it a recovered physical distance.
- Compare a shared calibration with the separate calibrations. Adopt sharing
  only if neither body's pooled RMSE worsens by more than 5%; otherwise retain
  explicit per-body calibration files. Do not add per-camera gain/phase/distance
  freedom without new evidence about the acquisition setup.

**Interface changes:** extend renderer configuration and the calibration JSON
with response, gain, projection and phase convention; store source identity,
sampling, and calibration scope. Expose these controls through the existing
validation workflow rather than creating a second fitter. Legacy calibration
files load with legacy defaults and an explicit warning. Unknown-body reuse
requires an explicit transfer choice. Version checkpoints and fingerprint the
complete objective so changed calibration, response, or splits cannot silently
resume an old optimization. Keep old trajectories readable.

**Gate:** reproduce the response improvement on the known orthographic video,
not only on the ambiguous text target; verify finite gradients and soft/exact
consistency; improve full-turn all-camera calibration without hiding a worsening
`225_bottom` result. If no response candidate passes, keep legacy behavior and
report the unresolved model discrepancy rather than broadening the model silently.

## 3. Isolate shape capacity and visibility before changing the optimizer

- With calibration fixed, compare the raw reference, 1,280-face public proxy,
  direct geometry fits using all three degree-6 octantoid fields, and the
  photometrically fitted shape. Unlike the earlier radial-only control, fit
  `a`, `b`, and `c` to test the actual representation's capacity. Geometry fits
  are diagnostic controls only and never initialize production inversion.
- Change harmonic degree and mesh resolution separately: degree 6 versus 10
  at 1,280 faces, then 1,280 versus 5,120 faces at fixed degree. Measure geometry
  error, dark-interval brightness, full-curve error, runtime, and peak memory.
  Raise the face limit only for the bounded comparison after a memory preflight;
  do not remove the existing safety checks or rasterize the raw STL on GPU.
- At fixed geometry/camera/response, compare hard solar rays with shadow maps,
  and flat with smooth normals. Include simple analytic surfaces, grazing views,
  and the sawed cube. Vary image resolution, shadow resolution, bias, and softness
  one at a time. The previous combined sharp-shadow stress test is not evidence
  for blindly reducing bias. Change production visibility only after a parity
  test localizes an implementation error or a measured approximation tradeoff.
- Retain staged Adam and the existing winning schedule for the first corrected
  fits. Use the current regularization initially; only after the geometry
  controls show adequate capacity compare its current value with half that
  value. Leave broader learning-rate searches and escape-policy changes until
  these forward-model controls pass.

**Gate:** decide whether the remaining gap is raw-to-proxy, proxy-to-octantoid,
or octantoid-to-photometric-fit. Promote added capacity only when its validation
benefit justifies runtime and memory; do not spend solver time chasing a shape
the representation cannot express.

## 4. Validate the correction and roll it out conservatively

- Run CPU unit tests for response breakpoints and clipping, post-response
  thresholding, finite derivatives at zero, frame indexing, splits, normalization,
  provenance/cache invalidation, and legacy serialization. Add CUDA tests for
  two-pass gradient equivalence, sphere rotational invariance, known shadowing,
  camera/projection limits, and resumed versus uninterrupted runs.
- Compare legacy, coverage-only, and coverage-plus-recalibrated-response fits
  on both asteroids using the same 72/72 score partition and full-turn
  normalization. Rescore old saved meshes under that reporting convention;
  do not compare incompatible old and new validation numbers. Report both
  stage-local fit residuals and a common fixed-model evaluation of saved shapes.
- Use two seeds for the final legacy/corrected comparison, retaining 300 Adam
  steps initially. Report parameter/checkpoint finiteness, memory, runtime,
  validation error, untouched-rotation error, all-camera metrics, and public-shape
  error. Public meshes are for calibration/diagnostics/reporting, not shape loss.
- Label these held-out rotations as **conditional on public-mesh calibration**:
  calibration used full-turn observations, so they are not untouched end-to-end
  test data. Do not claim transfer/generalization from this two-body experiment.
- Proposed promotion criteria: at least 20% lower A2 `225_bottom` full-turn RMSE,
  lower A2 pooled error, and no more than 5% worse A3 pooled error on the paired
  two-seed means, with no numerical failures. Publish every camera's residuals
  and geometry metrics even if these aggregate gates pass. Failure leaves the
  corrected configuration experimental; it does not justify selecting a nicer
  channel or changing the target data.
- Update the mathematical specification and default workflow only after the
  implemented algorithm and these checks agree. Keep original artifacts and
  legacy configuration replay available; save corrected runs in a new directory.

### Experiment budget and deferred work

Budget the first validation campaign at **8 hours total compute**, with a hard
stop at **9.5 hours**, on the existing RTX 3080. This is not an estimate of
engineering time. Preflight 10 representative optimization steps and project
remaining cost before launching fits. Allocate approximately 2 hours to
calibration/reference checks, 4 to bounded paired fits, and 2 to capacity checks
and reporting. If the estimate exceeds budget, defer the degree/mesh expansion
and broad retuning first; complete the A2 paired comparison and an A3 regression
control before adding seeds. Mark any one-seed result provisional and do not
claim that the two-seed promotion gate passed. Checkpoint cleanly at the deadline.

Defer Hapke/multiple-scattering models, interreflection, per-camera nuisance
parameters, and a new optimizer until a residual survives the simpler controls.
Treat the binary first-frame Otsu discrepancy as a separate, tested correction;
it does not explain the intensity residual investigated here.
