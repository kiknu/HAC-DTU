# Forward-model mismatch investigation

Date: 2026-08-06

## Executive conclusion

The poor `225_bottom` fit is not evidence that the target is simply “brighter
than Lommel--Seeliger can represent.” Every comparison here uses a separate
mean normalization for each lightcurve, so a constant albedo or exposure
factor cancels. The opposition camera `0_hor1` is an unusually weak validation
case: it largely hides differences among scattering laws, shadow boundaries,
camera projection, and indirect illumination. `225_bottom` observes a 129.46
degree crescent and is sensitive to all of them.

The severe mismatch was traced to geometric preprocessing and camera-model
errors, not to a failure of Lommel--Seeliger brightness:

1. Exact first-hit visibility and solar shadows on the untouched raw mesh
   reduce asteroid 3's difficult high-phase error substantially.
2. The published camera elevations are asymmetric, rather than a common
   `+/-26` degrees.
3. A finite-distance perspective camera cuts asteroid 2's all-camera raw-mesh
   RMSE from about 0.302 to 0.184.
4. The previous `aggression=10` quadric decimation destroys asteroid 2's
   sawed-cube geometry. Embree on that damaged 1,024-face mesh scores 0.858;
   a 1,280-face proxy on the optimizer's radial topology scores 0.160.
5. One fixed Lambert/Lommel--Seeliger mixture is not sufficient across phase
   angles, but replacing it with the tested isotropic Hapke approximation does
   not improve asteroid 2.
6. Applying the challenge's nominal 10% video-pixel cutoff directly to linear
   BRDF radiance is incorrect without Blender's exposure and color transform.
   All three global raw-mesh fits select a zero linear cutoff.

Indirect self-reflection should therefore not be the first correction made.
Jeppe's current Mitsuba scripts use a `direct` or `direct_projective`
integrator and a diffuse material: they include hard cast shadows but no
surface-to-surface bounce. Any path-traced test must render each asteroid phase
in an isolated scene, because putting multiple phase copies in one path-traced
scene permits nonphysical energy exchange between copies.

With body-specific projection calibration and faithful optimizer-topology
proxies, the differentiable 1,280-face mean RMSEs are 0.0120, 0.1644, and
0.0398 for asteroids 1--3. Their untouched-raw reference values are 0.0104,
0.1835, and 0.0309. The former order-of-magnitude asteroid-2 failure is gone;
the bounded remaining asteroid-3 crescent error is a mesh-resolution effect.

## Validation policy

The accuracy reference reads binary Gmsh 2.2 nodes and first-order triangles
directly. It does not center, scale, weld, simplify, repair, resample, or
downsample the inputs. Only the sun/view geometry and physical z rotation vary.

| Asteroid | Raw vertices | Raw faces |
|---:|---:|---:|
| 1 | 400,002 | 800,000 |
| 2 | 268,977 | 537,950 |
| 3 | 3,000 | 5,996 |

An Embree image-space reference traces hard primary rays and independent solar
shadow rays. Smooth vertex normals are used for BRDF evaluation by default,
matching the old PyTorch3D shader, while geometric triangles determine ray
visibility. The optimizer remains free to use a downsampled mesh; raw meshes
are the validation oracle, not the optimization representation.

All reported objectives are the mean of 21 per-camera normalized-curve RMSEs.
`hor2` is excluded because it duplicates `hor1`. One global physical phase
offset is shared by all cameras. Per-camera fits are diagnostics only and are
never presented as a valid physical calibration.

## Correct forward math

Let `mu0 = max(n dot s, 0)` and `mu = max(n dot v, 0)`. For a visible surface
element with observer visibility `V`, solar visibility `S`, and geometric
projected area `mu_g dA`, the direct disk integral is

```text
Lambert:  dI proportional to V S (mu0) mu_g dA
LS:       dI proportional to V S [mu0 / (mu0 + mu)] mu_g dA
mixture:  f * Lambert + (1 - f) * LS
```

The previous centroid implementation effectively mixed radiance and projected
area terms. The corrected implementation keeps smooth shading normals in the
BRDF, geometric normals in projected area/backface decisions, and observer
visibility separate from solar shadowing.

## Global raw-mesh results

Orthographic, smooth normals, 128 by 128 pixels, full 360-frame sweep:

| Asteroid | Lambert fraction | Linear cutoff | Phase | Mean RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.8 | 0 | -1 deg | 0.01998 |
| 2 | 0.3 | 0 | 0 deg | 0.30156 |
| 3 | 0.4 | 0 | 0 deg | 0.03446 |

Asteroid 3 is the clearest diagnosis of the originally reported symptom. The
old downsampled orthographic model gave `0_hor1` RMSE 0.00496/correlation
0.9968 but `225_bottom` RMSE 0.1106/correlation 0.825. With the corrected
perspective model and a 1,280-face optimizer-topology proxy, those become
0.00588/0.9974 and 0.07395/0.9187. At 2,048 faces the latter remains similar at
0.07546/0.9196 under a slightly different global mixture, while aggregate RMSE
falls to 0.03571. The untouched raw reference at its calibrated distance
gives `225_bottom` RMSE 0.05349/correlation 0.9551. Most of the residual gap is
therefore caused by retained shape resolution and visibility morphology, not
by an absolute BRDF brightness ceiling.

The camera-independent mixture is still imperfect. For asteroids 1 and 3,
independent diagnostic fits prefer progressively more Lambertian scattering
at large phase. This is consistent with a fixed empirical LS/Lambert mixture
being too rigid, but it does not justify fitting a separate material to each
camera in the inversion.

Flat versus smooth shading changes asteroid 2's global RMSE by only about
0.001. Shading-normal convention is not the source of its large residual.

## Projection diagnosis for asteroid 2

The raw finite-distance ablation uses a pinhole camera aimed at the mesh origin;
distance is expressed in maximum-origin-radius units. Resolution is 96 by 96
except for the orthographic baseline (128 by 128).

| Projection / distance | Mean RMSE |
|---|---:|
| Orthographic | 0.302 |
| Perspective, 2.0 radii | 0.311 |
| Perspective, 2.5 radii | 0.222 |
| Perspective, 3.0 radii | 0.191 |
| Perspective, 3.5 radii | **0.183** |
| Perspective, 4.0 radii | 0.188 |
| Perspective, 5.0 radii | 0.201 |
| Perspective, 6.0 radii | 0.215 |

At 3.5 radii, the `225_bottom` curve correlation rises from about 0.796 to
0.972. Its global-fit amplitude is still wrong, so projection is a major term
but not a complete explanation.

One distance does not apply to all updated Blender curves. Full matched 128 by
128 raw sweeps give:

| Body | Orthographic RMSE | Best distance | Perspective RMSE | Maximum RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.01998 | 4 radii | **0.01041** | 0.02922 |
| 2 | 0.30156 | 3.5 radii | **0.18351** | 0.47481 |
| 3 | 0.03446 | 16 radii | **0.03094** | 0.05826 |

Asteroid 1 has a broad 4--5-radii optimum. Asteroid 3 is harmed at 3.5 radii
but improves over a broad 12--16-radii basin. The fitted Lambert fraction also
changes systematically with distance, demonstrating that projection and
inferred scattering are confounded. These distances are empirical image-model
parameters, not recovered Blender intrinsics.

At asteroid 2's 3.5-radii optimum, `225_bottom` improves from RMSE 0.8170 to
0.3494 and correlation 0.7958 to 0.9721. High-phase horizontal views remain
the largest residuals (about 0.475), so projection is a major term but not the
entire signal model.

The local file provenance is decisive: all public simulation MP4s are dated
May 7, asteroid 1's Blender lightcurves are dated May 8, and asteroid 2/3's
Blender lightcurves are dated July 31. The exact Blender lens, sensor fit,
distance, exposure, color transform, material, and bounce settings are not
included in `challenge_data`. The May simulation videos can help identify the
older render setup, but cannot be used as pixel-level ground truth for the
current July asteroid 2/3 curves.

This was verified by decoding all local simulation frames and recreating the
published video-pixel extraction:

| Local video versus text curve | RMSE | Correlation | Phase |
|---|---:|---:|---:|
| A1 `0center` | 0.000080 | 0.999951 | 0 deg |
| A1 `225bottom` | 0.000713 | 0.999925 | 0 deg |
| A2 `0center` | 0.058284 | 0.454034 | 0 deg |
| A2 `225bottom` | 0.148223 | 0.995936 | +1 deg |
| A3 `0center` | 0.009736 | 0.987752 | -1 deg |
| A3 `225bottom` | 0.085458 | 0.955866 | -2 deg |

Asteroid 1's binary extraction also matches to numerical precision. The local
MP4s are therefore authoritative for asteroid 1 only; current asteroid 2/3
validation must use the July text curves.

## Optimization-mesh isolation

The previous benchmark repeatedly ran quadric decimation at aggression 10.
The simplifier documents this setting as fast and low quality. On asteroid 2,
quality-first decimation stops near 8,900 faces; forcing 1,024 faces changes
the geometry enough that even exact Embree rendering fails:

| Asteroid 2 mesh/renderer | Orthographic RMSE | Perspective RMSE |
|---|---:|---:|
| Untouched raw, Embree | 0.3016 | 0.1835 |
| Damaged QEM 1,024, Embree | 0.7648 | 0.8576 |
| Radial optimizer proxy 1,280, Embree | 0.2858 | **0.1595** |
| Radial optimizer proxy 1,280, differentiable | - | **0.1644** |

This isolates the former asteroid-2 discrepancy to preprocessing rather than
the differentiable perspective equations. Validation now samples the raw
surface radially on the same icosphere topology used by the star-shape
optimizer. Quality-first QEM remains optional and refuses unsafe targets.

Final optimization-resolution intensity results, using one global phase and
one global scattering mixture per body:

| Body | Proxy faces | Distance | Mean RMSE | Max RMSE | `0_hor1` | `225_bottom` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,280 | 4 radii | 0.01205 | 0.03511 | 0.00248 | 0.02060 |
| 2 | 1,280 | 3.5 radii | 0.16442 | 0.45002 | 0.02088 | 0.31919 |
| 3 | 1,280 | 16 radii | 0.03978 | 0.08207 | 0.00588 | 0.07395 |

For asteroid 3, a generic 2,048-face closed proxy reduces aggregate RMSE to
0.03571 at a 143-second CPU validation cost. Three-point sub-face visibility
sampling instead changes 0.046382 to only 0.046361 at 1,024 faces while nearly
doubling runtime. Retained mesh detail is therefore more valuable than extra
quadrature at the same topology.

## Scattering and self-reflection tests

The tested isotropic Hapke-style local multiple-scattering term was

```text
mu0 / (mu0 + mu) * [P + H(mu0, w) H(mu, w) - 1].
```

| Asteroid 2 model | Projection | Best mean RMSE |
|---|---|---:|
| Lambert/LS mixture | Orthographic | 0.3016 |
| Hapke grid | Orthographic | 0.3029 |
| Lambert/LS mixture | Perspective, 3.5 radii | 0.1829 |
| Hapke grid | Perspective, 3.5 radii | 0.1829 |

The perspective Hapke optimum collapses to zero multiple-scattering albedo and
is identical to the LS solution. This rejects that particular local
multiple-scattering correction as a useful explanation.

A separate raw-Gmsh geometric-interreflection ablation was run on asteroid 3.
It used flat Lambert scattering, cosine-weighted hemisphere sampling, 256 rays
per face, one indirect bounce, frames every 3 degrees, and a common 3-degree
phase alignment. Only 4.59% of outward cosine rays reached another face.

| Camera/model | RMSE | Correlation | Indirect fraction |
|---|---:|---:|---:|
| `0_hor1`, direct | 0.02064 | 0.9843 | 0% |
| `0_hor1`, one bounce, albedo 0.5 | 0.01914 | 0.9879 | 1.31% |
| `0_hor1`, one bounce, albedo 1.0 | 0.01836 | 0.9893 | 2.59% |
| `225_bottom`, direct | **0.03336** | **0.9801** | 0% |
| `225_bottom`, one bounce, albedo 0.5 | 0.03644 | 0.9759 | 2.89% |
| `225_bottom`, one bounce, albedo 1.0 | 0.04247 | 0.9671 | 5.62% |

Even the unit-albedo upper bound improves opposition only slightly and clearly
worsens `225_bottom`. A zero-shift comparison reaches the same conclusion.
Thus one diffuse geometric self-reflection cannot explain the reported
mismatch. Higher bounces would be smaller still unless an unrealistically high
reflectance is assumed. This is an ablation rather than a reconstruction of
unknown Blender material settings, but it is strong evidence against prioritizing
self-reflection over visibility, projection, and response calibration.

## Implemented corrections

- Full asymmetric challenge elevation table:
  `0: +21/-21`, `45/90/135: +26/-24`, and
  `225/270/315: +24/-26` degrees.
- Correct Lambert and Lommel--Seeliger disk integration.
- Smooth BRDF normals with geometric projected-area/visibility normals.
- Independent observer visibility and solar self-shadowing.
- All faces retained as possible non-convex occluders by default.
- Differentiable response threshold applied after shadowing and now disabled by
  default; it remains an explicit opt-in parameter.
- Exact raw Gmsh loader and Embree validation renderer.
- Orthographic/perspective, flat/smooth, Lambert/LS/mixture/Hapke ablations.
- Differentiable pinhole projection with per-face view vectors,
  perspective-correct depth, and image-plane area Jacobian.
- Optimizer-topology radial proxies in place of destructive aggressive QEM.
- Optional three-point sub-face visibility quadrature (not selected by the
  accuracy/runtime ablation).
- Joint global calibration over all cameras with one shared phase offset.

## Recommended optimization configuration

Use the raw renderer to calibrate/check physical assumptions, then optimize a
downsampled closed, consistently oriented mesh. For the current optimizer:

- keep smooth shading and separate geometric normals;
- leave the linear intensity threshold disabled unless exposure/tone mapping
  is explicitly modeled;
- retain every face as a possible occluder when memory permits;
- use the radial optimizer topology for preprocessing/validation; do not force
  asteroid 2 through a 1,024-face QEM target;
- use perspective distances of roughly 4, 3.5, and 12--16 radii for the three
  public Blender datasets, while treating distance as a calibratable nuisance
  parameter for other datasets;
- calibrate one shared phase and one global scattering model, not per-camera
  nuisance parameters;
- run resolution convergence checks, especially on crescent cameras;
- do not interpret the fitted distance or scattering mixture physically until
  the current Blender scene settings are available.

## Reproduction

```bash
# Exact raw direct-light reference (legacy, moved to legacy/)
uv run python legacy/raw_forward_validation.py \
  --asteroids 1,2,3 --columns 0_hor1,225_bottom \
  --resolution 256 --frame-stride 1 --intensity-threshold 0

# Joint orthographic material/phase/response calibration (legacy)
uv run python legacy/raw_model_sweep.py \
  --asteroids 1,2,3 --resolution 128 \
  --projection orthographic --shading smooth

# Finite-distance projection diagnostic (legacy)
uv run python legacy/raw_model_sweep.py \
  --asteroids 2 --resolution 96 \
  --projection perspective --camera-distance-radii 3.5

# Local Hapke diagnostic (legacy)
uv run python legacy/raw_model_sweep.py \
  --asteroids 2 --resolution 96 --scattering hapke \
  --projection perspective --camera-distance-radii 3.5

# Optimization-topology differentiable validation (asteroid 2)
uv run python forward_validation.py \
  --asteroids 2 --target-faces 1024 --face-count-overrides '' \
  --mesh-preprocessing star_proxy --projection perspective \
  --camera-distance-radii 3.5 --no-convergence
```

The generated CSV/JSON evidence is under `artifacts/raw_model_sweep/`,
`artifacts/raw_forward_validation/`, and the
`artifacts/forward_validation_perspective_*` directories.

## Verification status

- 25 focused forward-model, shape-architecture, projection, derivative, mesh,
  and response tests pass.
- Perspective output converges to orthographic output at large camera distance.
- The perspective shape derivative agrees with a centered finite difference.
- Checkpointed perspective visibility backpropagates finite gradients.
- The three-body default quick benchmark completes end to end.
- Python compilation and `git diff --check` pass.
