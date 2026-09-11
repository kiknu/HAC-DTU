# Asteroid 2 `225_bottom`: residual diagnosis

Follow-up: [prioritized correction plan](forward-model-fix-plan.md), proposed
2026-09-08, with implementation stages, compatibility rules, validation gates,
and a bounded compute budget. The plan is not yet implemented.

Date: 2026-09-07. This is a new investigation of the maintained PyTorch3D
renderer and the completed September tuning study, not a rerun of the
[historical August investigation](forward-model-investigation.md).
No production renderer, optimizer, challenge input, or tuning result was
changed. The diagnostic scripts and numerical outputs are in
[`artifacts/diagnosis_a2_20260907/`](../artifacts/diagnosis_a2_20260907/).

## Conclusion and order of investigation

The large residual is not evidence, by itself, that the low-albedo scattering
approximation fails on white paint. There are several simpler, measured
contributors. The following order reflects simplicity of correction and
strength of evidence, not an additive allocation of the total error:

1. **Stale local inputs are ruled out; video/text pairing is not resolved.**
   Fresh public downloads match all eight local asteroid-2/3 lightcurve files
   and both meshes byte-for-byte. However, neither published asteroid-2
   `225_bottom` video reproduces the text curve closely. The corresponding
   asteroid-3 orthographic video does. This is evidence of differing release
   or generation settings, not proof that the text is corrupt.
2. **Sparse phase sampling biases calibration.** Twelve equally spaced
   phases miss asteroid 2's narrow peaks. Normalizing on those samples changes
   the objective enough to reverse the preferred camera distance.
3. **Projection and image-response assumptions are significant forward-model
   errors.** Under the current linear-pixel pipeline, a nearer camera improves
   the public-shape prediction. Encoding radiance before video-pixel
   thresholding gives another large improvement, without indirect light.
4. **The fitted shape and low-resolution representation miss important
   geometry.** Rounded/smoothed approximations produce light in intervals
   where the sawed cube is almost dark. This survives independent hard-ray
   rendering and is not just an Adam learning-rate problem.
5. **Rasterization, shadow-map bias, and normal interpolation are secondary
   approximation issues.** Their effects are measurable, but simply increasing
   resolution or changing smooth to flat normals does not explain the whole
   residual. Very small shadow bias can introduce self-shadowing artifacts.
6. **High-albedo BRDF physics and interreflection remain possible residual
   effects, not established root causes.** Pure Lambertian shading is already
   available and does not fix this channel. A dedicated bounce experiment has
   not been performed for asteroid 2 in this investigation.

The recommended first implementation work is better phase coverage and an
explicit, calibrated image-formation model, followed by shape/mesh capacity
and optimizer checks. Do not launch another large learning-rate search or add
multiple scattering before resolving those simpler issues.

## Measurements on matched full rotations

All numbers in this table use the same 360 supplied asteroid-2 samples and
`225_bottom`, separately mean-normalized over the full rotation. Mixture
fraction is 0.5, phase offset is zero, and brightness RMSE is dimensionless.
Distances are in each mesh's maximum-origin-radius units, as in production;
they are effective model parameters, not recovered Blender intrinsics.

| Shape and forward setting | RMSE | Correlation |
| --- | ---: | ---: |
| Saved tuned STL, current 96 px renderer, distance 16 | 1.1980 | 0.4894 |
| Public 1,280-face proxy, same renderer, distance 16 | 0.8327 | 0.7871 |
| Same proxy and renderer, distance 3.5 | 0.4882 | 0.9322 |
| Same proxy/distance, unthresholded linear intensity | 0.3806 | 0.9636 |
| Same proxy/distance, sRGB then 8-bit threshold | 0.2404 | 0.9856 |
| Raw public mesh, hard rays, 256 px, distance 3.5, current pixel rule | 0.6210 | 0.9072 |
| Same raw rendering, unthresholded linear intensity | 0.4620 | 0.9448 |
| Same raw rendering, sRGB then 8-bit threshold | 0.1661 | 0.9944 |

The raw-mesh rows change backend and resolution and are **not** a one-factor
continuation of the proxy rows. Within each block, only the stated image
reduction changes. These are diagnostic controls on a public shape, not new
inverse reconstructions or held-out challenge scores. The original winning
trajectory reported RMSE 1.19742; reloading its exported STL gives 1.19800.
The small replay difference is immaterial to the conclusions.

![Matched full-rotation shape, response, and sampling controls](../artifacts/diagnosis_a2_20260907/diagnosis_controls.png)

### 1. Calibration sampling is demonstrably misleading

The supplied `225_bottom` curve has full-turn mean 1.000. Its 12 calibration
samples, spaced every 30 degrees, have mean **0.508887**. The 72-frame grid has
mean 0.969000. The short grid consistently catches the rising flanks and
misses the peaks; subset mean normalization amplifies this distortion.

Re-evaluating the public proxy with the current exact extraction, mixture
0.5, and all 21 cameras gives:

| Phase samples | Distance 3.5: pooled RMSE | Distance 16: pooled RMSE | Preferred of these two |
| --- | ---: | ---: | --- |
| 12 | 0.38580 | 0.25981 | 16 |
| 72 | 0.27443 | 0.38480 | 3.5 |
| 360 | 0.26103 | 0.36746 | 3.5 |

The full-turn mean of per-camera RMSEs also improves, from 0.27443 at distance
16 to 0.20771 at 3.5. Batch sizes 16 and 32 give identical 12-frame results
in this replay, so this reversal is not a chunking artifact.

The archived shared calibration selected distance 16 and mixture 0.5 using
12 frames of each asteroid. Its exact archived scores differ slightly from
this current-source replay, but both favor the far camera on 12 frames;
the full-turn reversal is independently reproduced here. The original tuner
ranked optimizer recipes conditional on that fixed calibration. It cannot
correct a biased, frozen forward model by tuning Adam hyperparameters.

The same 12 phases are used for training. Disjoint validation phases help
checkpoint selection, but cannot supply gradients at peaks absent from the
training grid. This is a concrete sampling defect, not proof that Adam itself
is unsuitable. See [sampling.json](../artifacts/diagnosis_a2_20260907/sampling.json).

### 2. Linear radiance is not an 8-bit video pixel

The implemented direct radiance is

\[
L = S\left[\xi\mu_0+(1-\xi)\frac{\mu_0}{\max(\mu_0+\mu,10^{-3})}\right],
\qquad
\mu_0=\max(n\cdot s,0),\quad \mu=\max(n\cdot v,0).
\]

Observer projection supplies projected area; no additional viewing cosine
should be multiplied into pixel radiance. Solar visibility `S` is a soft
shadow-map comparison in production and an independent hard ray in the
reference. Occlusion is evaluated geometrically, including non-convex faces.

The [official threshold diagram](../artifacts/diagnosis_a2_20260907/official_thresholding_white.png)
says to convert a video frame to 8-bit grayscale, set values below 25.5 to
zero, and sum the remainder. Production implements that final pixel rule,
but feeds **linear model radiance directly into it**: it has no exposure,
display transfer function, tone mapping, material gain, or specular response.

A physical comparison needs the distinction

\[
L\ \longrightarrow\ T(gL)\ \longrightarrow\ Q_{8\text{-bit}}
\ \longrightarrow\ \text{threshold and sum}\ \longrightarrow\ \text{normalize}.
\]

The sRGB control uses the standard piecewise transfer function, with no
fitted exponent and no per-camera phase or gain. It is a plausible diagnostic,
**not a claim that the unpublished Blender scene uses exactly this transform**.
Blender exposure, AgX/Filmic/Standard choice, lighting strength, and material
settings have not been recovered.

A uniform albedo/exposure multiplier cancels only for **linear, unclipped,
unthresholded** integrated light. It need not cancel when applied before
thresholding, clipping, or tone mapping. Thus mean normalization does not
make image formation irrelevant. The broad cancellation statement in the
historical August document needs this qualification for the current renderer.

The new orthographic videos provide a control with **known projection**:
against asteroid 2's orthographic `225_bottom` video, the raw orthographic ray
reference improves from RMSE 0.4351 to 0.1366 with sRGB (same mixture 0.5,
first video frame treated as the initial pose). Allowing a one-frame phase
adjustment gives 0.0763 for sRGB. Thus the response effect persists when
camera distance is no longer a free explanation. This still does not identify
Blender's exact tone-mapping or exposure settings.

Projection and response are confounded: on the 72-frame *raw-mesh* all-camera
control, sRGB gives mean-camera RMSE 0.16278 at distance 3.5 and 0.15651 at
distance 16, even though `225_bottom` strongly favors the nearer camera.
Consequently, **do not hard-code 3.5 as the physical camera distance** based
on this channel alone. Joint, full-phase, all-camera calibration is needed.

### 3. Shape resolution and inversion are separate limitations

The raw asteroid-2 surface has 268,977 welded vertices and 537,950 triangles.
Its coordinate bounds are exactly `[-3,3]` on each axis, so bounding-box
recentering introduces no rotation-axis translation for this asteroid.
The production public proxy has 642 vertices and 1,280 triangles, sampled
radially from the source; the fitted model further restricts these vertices
to degree-6 octantoid fields.

On the 72-frame grid, during intervals where the target brightness is below
0.1, the hard-ray proxy has mean predicted brightness 0.471 under the current
pixel rule; the hard-ray raw surface gives 0.040. The production proxy gives
0.230. These are separately normalized curves. The absolute numbers depend
on pixel/shadow treatment, but the geometry-dependent false floor is robust.

An unregularized degree-6 harmonic least-squares projection of the public
log-radius field, with octantoid `b=c=0`, gives full-turn `225_bottom` RMSE
1.082 at distance 16 and 0.848 at 3.5. This is only a simple radial subfamily
control, **not** the best attainable octantoid fit. It shows why substituting
the unconstrained public proxy is not a pure test of optimizer convergence:
representation, regularization, and photometric local minima remain entangled.
See [representation.json](../artifacts/diagnosis_a2_20260907/representation.json).

The camera distance and field of view also follow the current mesh's maximum
radius. Changing the shape therefore changes the effective physical camera
position. That scale convention must be accounted for in future raw/proxy
image comparisons and physical calibration.

### 4. Geometry and renderer checks

The maintained camera convention agrees with the published arrangement:
sun `(-1,0,0)`, azimuth `beta-180°`, and the asymmetric elevations. For
`225_bottom`, elevation is -26° and the phase angle is approximately 129.46°.
It is a crescent view, especially sensitive to corners, grazing illumination,
visibility, and pixel thresholds. The body rotates by minus the supplied
angle. The raw/sRGB control reaches correlation 0.9944 at zero phase shift;
a reversal does not improve its best phase-aligned score. There is no evidence
here for a gross azimuth, elevation, rotation-sign, or column-label swap.

There is a **one-degree indexing ambiguity**: text frame labels are 1..360,
but the renderer uses them directly as degrees, while a video's first frame
normally represents the initial pose. Evaluating at `frame-1` instead reduces
the public-proxy distance-16 residual from 0.8327 to 0.7509, and distance-3.5
from 0.4882 to 0.3985. The exact shape-to-video initial pose needs confirmation;
this small phase issue is a contributor, not a complete explanation. Video/text
comparisons below pair the first video frame with the first text row before
searching any phase shift. See
[one_degree_phase.json](../artifacts/diagnosis_a2_20260907/one_degree_phase.json).

At distance 3.5, increasing both production image/shadow resolution from 96
to 256 changes proxy RMSE from 0.4882 to 0.5147 under the current pixel rule
(sRGB: 0.2404 to 0.2204). This is not a resolution-only cure. Smooth-to-flat
normals on the raw 256 px reference change sRGB RMSE from 0.1661 to 0.1905,
far smaller than the image-response effect.

A deliberately sharp 512 px shadow map with bias/softness reduced from
0.002 radii to 0.00001 makes the result worse. This combined stress test is
not a calibrated replacement: depth interpolation and grazing surfaces need
adequate bias to avoid self-shadowing artifacts. The hard-ray and rasterized
proxy predictions are not numerically interchangeable. A renderer parity
test should precede any production shadow change.

One unrelated specification issue surfaced: official binary extraction fixes
Otsu's threshold from the **first frame**, whereas the maintained helper
computes it per image. This cannot explain the present **intensity** residual.

### 5. What the white-paint hypothesis does and does not explain

The official site says all ten physical models are coated with matte white
paint. However, the completed study explicitly fits `origin="simulated"`;
the real coating description does not establish the simulated Blender BRDF
or color-management settings.

Lommel–Seeliger's single-scattering interpretation is most natural for dark
particulate surfaces, so it is reasonable to question it for white paint.
But the implemented mixture already includes the simpler diffuse endpoint:
on the raw 256 px, distance-3.5 control, changing mixture 0.5 to pure Lambert
**worsens** current-rule RMSE from 0.6210 to 0.7618. With sRGB, the same change
is only 0.1661 to 0.1708. Response/geometry dominate this particular comparison.

As a cross-check, asteroid 3's raw-mesh `225_bottom` control at distance 16
also improves from 0.07365 to 0.03530 when sRGB is applied before extraction.
This does not establish a universal calibration, but the effect is not unique
to asteroid 2's sawed-cube shape.

High-albedo BRDF multiple scattering and geometric surface-to-surface
interreflection are distinct effects. Neither has been conclusively excluded.
Adding indirect illumination tends to fill dark intervals, whereas excess
predicted light in those intervals is already one of the main failures.
That makes it a less economical first explanation, not an impossibility.

## Data provenance and reproducibility

The official [challenge page](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/)
was captured in [official-page.html](../artifacts/diagnosis_a2_20260907/official-page.html).
It announces July 31 corrected Blender curves for asteroids 2/3 and August 17
orthographic simulation videos. The local July archive already dates the
asteroid-2/3 Blender text files July 31, while its simulation movies are dated
May 7. Dates alone cannot establish whether the local files match today's
release; use file hashes and decoded video comparisons.

The full-folder transfer timed out after 5,546,563,685 of 6,403,635,457 bytes.
It was **not** treated as a complete archive. A subsequent browser-style
listing exposed the actual per-file public links, allowing selective
downloads of 18 relevant files (166,434,936 bytes) without another full ZIP.
The original input tree and July archive were preserved. The incomplete
transfer is retained as `current_challenge_data.zip.partial`, clearly separate
from the verified individual files under `current_release/`.

[provenance.json](../artifacts/diagnosis_a2_20260907/provenance.json) records
fresh/local/July-archive SHA-256 comparisons. All eight intensity/binary,
real/simulated text files, both public STL meshes, and the four selected older
simulation MP4s match exactly. Four new orthographic videos were additionally
downloaded. In particular, the simulated intensity hashes are:

```text
A2 7599774eb0b4adf14ccb977962452d2fa674b455872ba34829de13df178cb041
A3 3d69c46fcfde2f17c04f10a136821a3ea2bad7c7e0dc1af8ee3ed6c96a7ab856
```

All eight selected MP4s were decoded at their native 1024 × 1024 resolution
over all 360 frames. RGB was converted to 8-bit grayscale; intensities at
least 26 were summed, then each curve was mean-normalized. The table permits
a circular phase shift, so its nonzero discrepancies cannot be dismissed as
a first-frame indexing issue. These per-video alignments are diagnostics,
not independent phase parameters proposed for the inverse solver.

| Asteroid / camera | Older simulation video vs text: best RMSE | New orthographic video vs text: best RMSE |
| --- | ---: | ---: |
| 2 / `0_hor1` | 0.05828 | 0.00287 |
| 2 / `225_bottom` | 0.14822 | 0.39044 |
| 3 / `0_hor1` | 0.00974 | 0.000148 |
| 3 / `225_bottom` | 0.08546 | 0.00516 |

Asteroid 2's opposition view is much closer to the new orthographic video,
whereas its crescent text curve is closer to the older perspective video.
That makes camera-specific generation/projection settings, incomplete update
coverage, or other release-pairing differences plausible candidates. It is
**not enough to prove a mixed/corrupt text file**: videos and curves may
intentionally come from different scenes or extraction settings. The precise
Blender scene and the video corresponding to each July text column are the
most useful questions for the organizers. No external message was sent.

The near-exact asteroid-3 agreement is also a positive control on the video
decoder/extraction procedure. Full metrics and selected frame samples are in
[video_metrics.json](../artifacts/diagnosis_a2_20260907/video_metrics.json).

![Current official video curves compared with unchanged asteroid-2 text](../artifacts/diagnosis_a2_20260907/release_curves.png)

Reproduction scripts, run from the workspace using the installed project Python:

- `forward_probe.py`: matched public/fitted controls; GPU PyTorch3D proxies
  and CPU Embree reference rays, with saved endpoint images and curve arrays.
- `sampling_probe.py`: 12/72/360-frame calibration and batch-invariance audit.
- `representation_probe.py`: degree-6 known-shape projection control.
- `provenance_probe.py`: fresh/old/local SHA-256 comparisons and selected MP4
  extraction using a decoder installed under `/tmp`, outside project dependencies.
- `fetch_current.py`: bounded selective downloads through the actual public
  links in Dropbox's browser listing; no authentication or input overwrites.
- `video_reference_probe.py`: ordinal video/text alignment, known-orthographic
  renderer control, and one-degree phase-sensitivity checks.
- `summary_plot.py`: plots from saved arrays; no rerendering required.

The independent ray implementation rotates the observer/light into body
coordinates, traces first observer hits and solar shadow rays, and computes
the same Lambert/LS radiance endpoints. It does not decimate or radially
resample the raw surface. STL vertices are welded on load, but surface
triangles, coordinates, and rotation origin are retained. Numerical controls
are diagnostic and do not alter the production solver.
