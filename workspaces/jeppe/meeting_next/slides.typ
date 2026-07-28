#import "@preview/touying:0.7.4": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

#set text(size: 20pt)

= Jeppe Update — 2026-06-23

== Mitsuba
- Inverse rendering framework by team at EPFL
- Uses their own auto-diff framework (DrJit) as opposed to Jax or PyTorch
- Has built-in support for meshes (`obj`, `ply`, not `stl`)
- Has built in ray tracers
  - These are for more conventional rendering, so very much over-kill for our use case, but are differentiable and nice
- Can target NVIDIA GPUs using _CUDA_ and _OptiX_
  - _OptiX_ can leverage specialized hardware for ray tracing on NVIDIA GPUs

== Mitsuba - Light Curve (Asteroid 3)
Seems to produce sensible light curves:
#figure(image("asteroid3_0_hor1.png"))

== Mitsuba - Forward Rendering
#figure(image("forward_rendering_ast3.png", height: 80%))

== Alternatives and advantages
- Mitsuba
  - Uses somewhat esoteric _DrJit_ framework
  - Seems well-optimised
  - Scenery and some neat things are ready out-of-the-box
  - Alledgedly plays reasonably well with other frameworks

- Alternative could be PyTorch3D which uses more familiar PyTorch framework
  - Michael has had a go at this