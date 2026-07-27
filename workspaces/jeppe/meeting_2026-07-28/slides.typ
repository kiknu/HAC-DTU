#import "@preview/touying:0.7.4": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

#set text(size: 20pt)
#show figure.caption: set text(size: 16pt)
#show raw: it => box(it, outset: 3pt, fill: gray.transparentize(70%), radius: 5pt)
#show link: set text(font: "DejaVu Sans Mono", size: 0.8em, fill: navy.lighten(20%))

= Jeppe Update — 2026-07-28

== Mitsuba Forward
- Code from last time cleaned up a bit and available in `hac.mitsuba`
  - Still a little messy
  - I may break quite a few things in the future, but usable by others for experiments for now.
- Overview of new modules:
  - `scene`: module for constructing batch Mitsuba scenes and rendering them
    - `debug_scene`: to make the nice plots shown later
  - `visualize`: module for visualizing meshes and light curves
  - `forward`: module for forward rendering of meshes and light curves
    - Differentiable, alledgedly also _PyTorch_ compatible

#cols(columns: (1fr, 1fr))[
== Batch Scenes — $z$-stack
- To (efficiently) render + store gradients of mesh at multiple angles and _phases_ (of the rotation), we set up a single scene
  - Each _phase_ is an affine transformation on reference to the mesh
  - i.e. stacked and rotated along $z$
  - Each of these has sensors for enabled observation angles
- Live Demo?

][
  #figure(
    image("batch_scene.png"),
    caption: [Batch scene setup for asteroid 3, see `mitsuba_z_stack_viewer.ipynb` for details.]
  )
]

#cols(columns: (1fr, 1fr))[
== Batch Scenes — Forward
- Forward rendering is then just rendering each sensor individually
- Then applying appropriate reduction to get light curve
- Note: Currently this is just sum without thresholding
  - Does not match _Otsu/binary_ or _intensity_ methods
  - Given no 'stray light', so maybe thresholding not needed to match _intensity_ method well?
  - Tricky to get thresholding to be differentiable, but possible with _sigmoid_ or other
][
#figure(
  image("batch_scene_projections.png", height: 85%),
  caption: [
    Subset of sensor renderings in a batch scene for asteroid 3 using Mitsuba forward model.\
    Phases are #box[$ϕ ∈ {0°, 90°, 180°}$] across, respectively.
  ]
)
]

== Optimization
- Currently _relatively_ fast when run on A100 GPU
- Follows #link("https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html")
- Still lots to do, very basic at the moment
- Scale penalty is kind of wrong
  - Maybe only penalize (relative) point mass outside bounding cylinder?
- Try on all models
- Parameters currently:
  - Phases: $N_ϕ = 180$
  - Faces: $N_f = 2500$
  - Samples per pixel: $N_"spp" = 8$
  - Resolution: $(H, W) = (64, 64)$
  - Iterations: $N_"iter" = 300$
  - Adam with LR $α = 2×10^(-2)$
  - "Large Steps" parameter, $λ = 25.0$


== Showing reconstructions so far

== Coming up
- Try to improve forward model parameters by comparing known model meshes to their forward projections
- Make `ForwardModel` take a reduction method instead of hardcoding sum for lightcurve
- Maybe see if we can get speed-up with `u16` or `u8` instead of `u32`/`f32`?
- Making an `optimize` module under `hac.mitsuba` for common optimization routines

== Final remarks
- This method probable needs a good starting guess to work well
- Can we use Mitsuba forward with a convex inversion method?
- Would be keen to meet up with people in DTU Library / 307 for a half-day?
