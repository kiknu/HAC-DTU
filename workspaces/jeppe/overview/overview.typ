#set heading(numbering: "1.a")
#let vv = x => math.bold(math.upright(x))
#let vvh = x => math.hat(math.bold(math.upright(x)))
#let wider = h(3em)

= Light Curve Contribution

Under the condition an asteroid surface patch $dif s$ is _visible_ (not occulted by other parts of the asteroid) and _illuminated_ by the Sun, it contributes to the observed light curve as:
$
  dif L = S(μ, μ_0) ϖ dif s
$
where:
- $L$ is the observed brightness (light curve).
- $S(μ, μ_0)$ and $ϖ$ arise from the scattering law, see @sec:scattering_laws.
- $μ = vv(E) ⋅ vv(n)$ is the projection between the surface unit normal, $vv(n)$ and the unit vector toward the observer (Earth), $vv(E)$.
- $μ_0 = vv(E_0) ⋅ vv(n)$ is the projection between the surface unit normal, $vv(n)$ and the unit vector toward the Sun, $vv(E_0)$.

#figure(
  include "diagram_light_curve_geometry.typ",
  caption: [Geometry of light curve contribution.],
)

= Scattering Laws <sec:scattering_laws>

Common scattering laws include:
$
   S_"L" & = μ μ_0             && wider "Lambert" \
  S_"LS" & = S_"L" / (μ + μ_0) && wider "Lommel-Seeliger" \
$

= Geometry <sec:geometry>
#figure(
  grid(
    columns: 2,
    image("geometry_top_view.png"), image("geometry_side_view.png"),
  ),
  caption: [Geometry setup of the asteroid, Sun, and observer (Earth).],
)

So the geometry is set up as:
- Cartesian coordinates $(x, y, z)$
- Measurement angle $θ$ is the direction of view in the $x y$-plane, as labelled in the top-view figure. Thus $θ = 0°$ means the camera looks along $+x$ and is placed on the $-x$ side of the asteroid.
- Elevation angle $α$ is the angle above the $x y$-plane. Positive $α$ places the camera at positive $z$.

Thus we have objects:
- Sun at $vv(r)_"sun" = (-∞, 0, 0)$ with direction toward the Sun $vvh(E)_"sun" = (-1, 0, 0)$. The light travels in the opposite direction, along $+x$.
- Observer at $vv(r)_"obs" = r_"obs" (-cos(θ) cos(α), -sin(θ) cos(α), sin(α))$ with direction toward the observer $vvh(E)_"obs" = (-cos(θ) cos(α), -sin(θ) cos(α), sin(α))$
  - The optical/view direction from the observer to the asteroid is the opposite vector, $(cos(θ) cos(α), sin(θ) cos(α), -sin(α))$.
  - Where $r_"obs"$ is assumed to be sufficiently large that the asteroid observed approximately as its orthographic projection.

= Ray-Tracing <sec:ray_tracing>

Several algorithms exist, where it should be noted that we only require a relatively simple ray-tracing algorithm here due to the many simplifications that are possible here as opposed to the general case. For instance, we may assume monochromaticity and ignore atmospheric refraction if we assume the observer is unobstructed by an atmosphere.

The following algorithm has complexity $cal(O)(N^2)$ and can be used to determine _convex hull_.

== Conclusions 2026-06-19
- Standard fully-featured ray tracing algorithms are a poor fit, as we do not need things like:
  - Refraction
  - (Multiple scattering)
  - Polychromaticity

We can get away with simpler algorithms, in particular many papers seem to prefer doing the initial convex checks (visibility + illumination).