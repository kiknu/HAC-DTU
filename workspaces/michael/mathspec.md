# Mathematical specification: non-convex lightcurve inversion

This document describes the implementation used by `hac-asteroids fit`:
an octantoid triangle mesh, a PyTorch3D image renderer with self-occlusion
and cast shadows, and a regularized inverse problem solved with staged Adam.
The equations below specify the current defaults; run configurations can
override the mesh resolution, harmonic degree, and optimization settings.

| Component | Implementation |
| --- | --- |
| Octantoid fields, mesh, and spectral prior | [shape_model.py](shape_model.py), `OctantoidShapeModel` |
| Cameras, shadows, shading, and lightcurve extraction | [pytorch3d_forward.py](pytorch3d_forward.py), `PyTorch3DLightcurveModel` |
| Initialization, objective, schedule, and checkpoint selection | [hac_asteroids/inversion.py](hac_asteroids/inversion.py) |
| Fixed camera/material calibration | [hac_asteroids/calibration.py](hac_asteroids/calibration.py) |
| Observation loading and duplicate-camera removal | [hac_asteroids/datasets.py](hac_asteroids/datasets.py) |

The analytic soft-visibility renderer in `forward_model.py` and the
Gauss–Newton/Levenberg–Marquardt solver in `main.py` are separate implementations.
They are not the renderer and optimizer invoked by the maintained fit command.

## 1. Unknowns and observations

Let $\mathbf p\in\mathbb R^P$ contain the shape coefficients, $t$ index
rotation samples, and $c$ index cameras. The forward map is

$$
\mathbf p
\longmapsto \{\mathbf v_i(\mathbf p),F\}
\longmapsto I_{tc}(\mathbf p)
\longmapsto B_{tc}(\mathbf p)
\longmapsto \widehat y_{tc}(\mathbf p),
$$

where $F$ is fixed triangle connectivity, $I_{tc}$ is a grayscale image,
$B_{tc}$ is an unnormalized image statistic, and $\widehat y_{tc}$ is a
lightcurve normalized separately for each camera.

The default data are simulated intensity lightcurves. The seven azimuths
each supply horizontal, top, and bottom views, giving $C=21$ channels after
verifying and removing exact `hor2` duplicates of `hor1`. Real observations
retain 28 channels when the horizontal measurements differ. A fit selects
either intensity or binary lightcurves; it does not combine both losses.

Only $\mathbf p$ is optimized. The spin axis is the world $z$ axis, the
rotation samples are supplied, and camera distance and scattering mixture
are fixed by calibration. Although the renderer accepts a global phase
offset, the maintained inverse solver leaves it at zero.

## 2. Octantoid shape representation

### 2.1 Spherical template and harmonic basis

Start with unit directions $\mathbf u_i$ and faces $F$ from an icosphere.
With subdivision level $s$, the template has

$$
N_F=20\,4^s,\qquad N_V=10\,4^s+2.
$$

The default $s=3$ gives 642 vertices and 1,280 triangles. Connectivity is
unchanged during inversion; only vertex positions move.

Three independent scalar fields on the sphere are expanded in real,
orthonormal spherical harmonics:

$$
a(\mathbf u)=\sum_{\ell=0}^{L_a}\sum_{m=-\ell}^{\ell}
 a_{\ell m}Y_{\ell m}(\mathbf u),
\qquad
b(\mathbf u)=\sum_{\ell=0}^{L_b}\sum_{m=-\ell}^{\ell}
 b_{\ell m}Y_{\ell m}(\mathbf u),
\qquad
c(\mathbf u)=\sum_{\ell=0}^{L_c}\sum_{m=-\ell}^{\ell}
 c_{\ell m}Y_{\ell m}(\mathbf u).
$$

For colatitude $\vartheta$ and longitude $\varphi$, the convention is

$$
Y_{\ell m}=N_{\ell |m|}P_\ell^{|m|}(\cos\vartheta)
\begin{cases}
\sqrt{2}\sin(|m|\varphi),&m<0,\\
1,&m=0,\\
\sqrt{2}\cos(m\varphi),&m>0,
\end{cases}
\qquad
N_{\ell k}=\sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-k)!}{(\ell+k)!}}.
$$

The associated Legendre functions include the Condon–Shortley sign.
Basis columns are ordered by increasing $\ell$, then $m=-\ell,\ldots,\ell$.
Basis matrices $Y_a,Y_b,Y_c$ are evaluated once at the template directions;
field evaluation is then a matrix-vector product.

### 2.2 Vertex map and scale convention

The unsaturated octantoid map is

$$
\mathbf v_i=r_0
\begin{bmatrix}
u_{ix}\exp(a_i)\\
u_{iy}\exp(a_i+b_i)\\
u_{iz}\exp(a_i+c_i)
\end{bmatrix},
\qquad a_i=a(\mathbf u_i),\ b_i=b(\mathbf u_i),\ c_i=c(\mathbf u_i).
$$

Here $a$ controls a common radial deformation, while $b$ and $c$ independently
change the $y$ and $z$ coordinates. Setting $b=c=0$ gives a radial shape.
Allowing all three fields gives a broader family of non-convex surfaces.
Positive exponential factors preserve each coordinate's sign, which explains
the name *octantoid*; they do not impose convexity or guarantee an embedding
without self-intersections.

The constant $a_{00}$ is fixed to zero and omitted from $\mathbf p$. This
removes the explicit uniform log-scale coefficient. Constants $b_{00}$ and
$c_{00}$ remain free. Since $Y_{00}=1/\sqrt{4\pi}$, constant fields $b=B$,
$c=C$ with $a=0$ give an axis-aligned ellipsoid with semiaxes
$r_0(1,e^B,e^C)$ in the unsaturated map. The fit fixes $r_0=0.5$ as a size
convention; normalized photometry does not determine a physical length scale.

Parameters are packed as all nonconstant $a$ coefficients, all $b$
coefficients, then all $c$ coefficients. Thus

$$
P=(L_a+1)^2-1+(L_b+1)^2+(L_c+1)^2.
$$

The fit uses $L_a=L_b=L_c=6$, giving $P=146$.

The actual mesh function smoothly limits the three log-scales before
exponentiating. With the default limit $M=3$,

$$
h_M(q)=M\tanh(q/M),\qquad
\mathbf v_i=r_0\,\mathbf u_i\odot
\exp\!\left(h_M\!\left([a_i,a_i+b_i,a_i+c_i]\right)\right),
$$

with componentwise operations. Each coordinate multiplier lies between
$e^{-M}$ and $e^M$. Constant fields still produce an ellipsoid, now with
semiaxes $r_0(1,e^{h_M(B)},e^{h_M(C)})$. Setting `max_log_scale=None`
recovers the unsaturated formula.

### 2.3 Spectral regularization

For degree exponent $q$ and residual multiplier $\lambda_s$, the shape model
returns one prior residual per free coefficient:

$$
r^{\mathrm{prior}}_j=\lambda_s\ell_j^{q/2}p_j.
$$

The fit sets $\lambda_s=1$ and $q=2$. Squaring and averaging gives

$$
\mathcal R(\mathbf p)=\frac{1}{P}\sum_{j=1}^P
\ell_j^q p_j^2.
$$

High degrees are penalized more strongly. For $q>0$, the retained degree-zero
$b$ and $c$ coefficients contribute zero, so the constant-field ellipsoid
family has zero spectral penalty. The denominator still includes these
zero residual entries. This is the fit's only shape regularizer.

## 3. Observation geometry

For a camera named by azimuth $\beta_c$ and elevation $\alpha_c$, the
object-to-observer direction is

$$
\mathbf d_c=
\begin{bmatrix}
\cos(\beta_c-\pi)\cos\alpha_c\\
\sin(\beta_c-\pi)\cos\alpha_c\\
\sin\alpha_c
\end{bmatrix},
\qquad \mathbf s=(-1,0,0),
$$

where $\mathbf s$ points toward the directional light. Angles in equations
are radians; the challenge camera angles below are degrees.

| Azimuth | Horizontal elevation | Top elevation | Bottom elevation |
| --- | ---: | ---: | ---: |
| 0 | 0 | 21 | -21 |
| 45, 90, 135 | 0 | 26 | -24 |
| 225, 270, 315 | 0 | 24 | -26 |

At rotation sample $\theta_t$, the body vertices become

$$
\mathbf x_{ti}=R_z(-\theta_t-\phi_0)\mathbf v_i,
$$

with $\phi_0=0$ in the fit. Equivalently, one can keep the body fixed and
rotate the sun and viewing directions by $+\theta_t$. Simulated data use
either the supplied frame labels as degrees (`frame_label`, the legacy
convention) or $2\pi k/N_{\mathrm{data}}$ (`zero_based`). Calibration selects
one convention per body across every camera. Real data use the latter rule.

Define the current bounding radius
$R=\max(10^{-6},\max_i\|\mathbf v_i\|)$. Perspective cameras look at the
origin from

$$
\mathbf e_c=dR\mathbf d_c,\qquad
\mathrm{FoV}=2\arctan(f/d),\qquad f=1.20,
$$

where $d$ is the calibrated distance in bounding-radius units (default 5).
The camera up direction is $+z$. In camera coordinates, projection is
proportional to $(X/Z,Y/Z)/\tan(\mathrm{FoV}/2)$. The radius is recomputed
for every shape, so camera placement scales with the mesh. Near/far clipping
planes are $\max(10^{-4},0.01R)$ and $(d+2)R$.

An explicit orthographic observer uses image-plane bounds $[-fR,fR]$ and
constant viewing direction $\mathbf d_c$. The finite observer position then
sets view depth only. Read-only parity controls can specify a common reference
radius, fixing camera position and framing across meshes in common coordinates.
Production inversion retains the radius-relative scale convention.
Camera matrix multiplication uses full float32 precision: TF32 is disabled
because it makes shadow depths depend on matrix batch shape.

## 4. Non-convex image formation

### 4.1 Triangle rasterization and smooth normals

Each rotated mesh is rasterized from each observer camera. At every pixel,
PyTorch3D retains up to $K=4$ depth-ordered triangle fragments. The default fit
uses $96\times96$ pixels and zero rasterization blur. All faces participate
in visibility; faces are not removed merely because they are unilluminated.
Foreground triangles can therefore hide other parts of a non-convex body.

For an oriented face $f=(i,j,k)$, let

$$
\mathbf g_f=(\mathbf x_j-\mathbf x_i)\times(\mathbf x_k-\mathbf x_i),
\qquad A_f=\tfrac12\|\mathbf g_f\|.
$$

Vertex normals are area-weighted averages,
$\mathbf n_i=\operatorname{normalize}(\sum_{f\ni i}\mathbf g_f)$.
At a fragment with perspective-correct barycentric coordinates $w_i$, the
shader interpolates position and renormalizes the interpolated normal:

$$
\mathbf x=\sum_{i\in f}w_i\mathbf x_i,\qquad
\mathbf n=\operatorname{normalize}\left(\sum_{i\in f}w_i\mathbf n_i\right).
$$

Geometric triangles determine coverage and depth, while these smooth normals
determine radiance.

### 4.2 Directional illumination and cast shadows

For each rotation, a second rasterization creates a depth map viewed from
the light. This camera is orthographic, centered at $4R\mathbf s$, looking
at the origin with horizontal and vertical bounds $[-1.20R,1.20R]$.
It records the nearest face at each shadow-map pixel. Its finite position is
for depth mapping; illumination itself remains directional, with no
inverse-square light attenuation.

Project an observer fragment $\mathbf x$ into this light camera. Let
$z_L(\mathbf x)$ be its light-view depth and $D_L(\mathbf x)$ the bilinearly
sampled shadow-map depth. Its soft illumination visibility is

$$
S(\mathbf x)=\sigma\!\left(
\frac{D_L(\mathbf x)+b-z_L(\mathbf x)}{\eta}
\right),\qquad
b=0.002R,\quad\eta=\max(0.002R,10^{-8}),
$$

where $\sigma(z)=(1+e^{-z})^{-1}$. A surface behind the nearest surface
along the light direction receives a small $S$. The bias $b$ reduces
self-shadowing artifacts; $\eta$ softens the depth comparison. A separate
nearest-sampled validity mask identifies missing depth samples, which are
treated as lit ($S=1$). The default shadow map is also $96\times96$.

Observer depth ordering and this independent light depth test account for
self-occlusion and cast shadows, respectively. They are the essential
visibility operations for a non-convex shape.

### 4.3 Lambert/Lommel–Seeliger radiance

The viewing direction varies across the surface because the camera is
perspective:

$$
\mathbf v=\frac{\mathbf e_c-\mathbf x}{\|\mathbf e_c-\mathbf x\|},
\qquad \mu=\max(\mathbf n\cdot\mathbf v,0),\qquad
\mu_0=\max(\mathbf n\cdot\mathbf s,0).
$$

For calibrated Lambertian fraction $\xi\in[0,1]$, the fragment radiance is

$$
L(\mathbf x)=S(\mathbf x)
\left[\xi\mu_0+(1-\xi)\frac{\mu_0}{\max(\mu_0+\mu,10^{-3})}\right].
$$

The denominator floor limits derivatives at grazing angles. The material is
uniform: there is no albedo map, indirect illumination, or thermal emission.
Gain and response are applied after image blending.

There is no additional factor of $\mu$ in the shader. The pixel footprint
already accounts for geometric projected area and perspective. For
comparison, an unthresholded orthographic surface integral would contain
$V S L_0\max(\mathbf n_g\cdot\mathbf v,0)\,dA$, where $V$ is observer
visibility, $L_0$ is the bracketed radiance law, and $\mathbf n_g$ is the
geometric normal. Image-space integration accounts for that area factor
through projection.

### 4.4 Depth blending

PyTorch3D's `softmax_rgb_blend` combines fragment radiances over a black
background. For pixel $u$ and retained fragment $k$, let $m_{uk}$ mark a
valid fragment, $\delta_{uk}$ be the signed squared screen-space distance
returned by the rasterizer (negative inside), and $z_{uk}$ its view depth.
Define

$$
\pi_{uk}=m_{uk}\sigma(-\delta_{uk}/\sigma_r),\qquad
q_{uk}=m_{uk}\frac{z_{\mathrm{far}}-z_{uk}}
 {z_{\mathrm{far}}-z_{\mathrm{near}}},
\qquad \sigma_r=10^{-4},\quad\gamma_r=10^{-4}.
$$

The numerically stabilized weights, including the library's small background
weight, are

$$
q_u^*=\max(\epsilon,\max_k q_{uk}),\qquad
w_{uk}=\pi_{uk}\exp((q_{uk}-q_u^*)/\gamma_r),
\qquad
w_{u,\mathrm{bg}}=\max\!\left(\epsilon,
\exp((\epsilon-q_u^*)/\gamma_r)\right),\quad\epsilon=10^{-10}.
$$

All three color channels receive the same radiance, so their mean is

$$
I_{tc,u}=\frac{\sum_k w_{uk}L_{tc,uk}}
 {\sum_k w_{uk}+w_{u,\mathrm{bg}}}.
$$

Large inverse depth favors the nearest fragment. The returned alpha channel
is not used in lightcurve extraction. With zero rasterization blur, the
soft blend operates on the selected fragments; face selection and visibility
changes remain discrete. The renderer supplies useful local derivatives,
but is not globally smooth.

## 5. From images to normalized lightcurves

The blended linear image $I$ is transformed before either extraction rule:

$$
E=\operatorname{clip}_{[0,1]}(T(gI)),\qquad g>0.
$$

The explicit response choices are $T(x)=x$ (`linear`) and standard sRGB:

$$
T(x)=\begin{cases}12.92x,&x\le0.0031308,\\
1.055x^{1/2.4}-0.055,&x>0.0031308.\end{cases}
$$

The inactive power branch is clamped at the breakpoint before evaluation,
so its derivative is finite at zero. Legacy calibration files select linear
response and gain one, with a warning. The following extraction equations
use the encoded image $E$ in place of $I$. The same transform precedes soft
training and exact scoring. sRGB is an empirical candidate, not an identified
Blender color-management setting.

### 5.1 Hard extraction for validation and reporting

Quantize each image to eight bits:

$$
Q_u=\frac{\operatorname{round}(255I_u)}{255}.
$$

The intensity statistic uses the challenge's 10% cutoff:

$$
B^{\mathrm{int}}_{tc}=\sum_u Q_{tc,u}\,
\mathbf 1[Q_{tc,u}>25.5/255].
$$

Thus retained integer pixel values are 26 through 255. The binary statistic
counts pixels above a per-image Otsu threshold $\tau_{tc}$:

$$
B^{\mathrm{bin}}_{tc}=\sum_u\mathbf 1[Q_{tc,u}>\tau_{tc}].
$$

Otsu's threshold is computed from the 256-bin quantized histogram: choose
the split maximizing $n_0n_1(\bar q_0-\bar q_1)^2$, excluding empty classes,
and divide the selected bin index by 255. Ties use the first maximum.
`extraction="exact"` means these hard image reductions; shadow comparisons
and fragment blending still use the renderer above.

The binary path still uses per-image Otsu thresholds. The official first-frame
threshold convention is a separate known limitation; the intensity correction
campaign does not validate or change binary acquisition behavior.

### 5.2 Soft extraction for optimization

The training surrogate replaces threshold indicators with a sigmoid gate
that is exactly zero on a black background:

$$
G_\kappa(I;\tau)=\operatorname{clip}_{[0,1]}
\left(\frac{\sigma(\kappa(I-\tau))-\sigma(-\kappa\tau)}
 {\max(1-\sigma(-\kappa\tau),10^{-8})}\right).
$$

Training uses unquantized images and

$$
B^{\mathrm{int,soft}}_{tc}=\sum_u I_{tc,u}G_\kappa(I_{tc,u};0.1),
\qquad
B^{\mathrm{bin,soft}}_{tc}=\sum_u G_\kappa(I_{tc,u};\tau_{tc}).
$$

The maintained fit sets $\kappa=8$ for both observables, overriding the
standalone renderer's default 96. For binary fitting, Otsu's threshold is
recomputed from each image under `no_grad`; its dependence on the shape is
not differentiated.

### 5.3 Per-camera normalization

For the set $\mathcal T$ of frames being scored,

$$
\overline B_c=\frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}B_{tc},\qquad
\widehat y_{tc}=\frac{B_{tc}}{\max(\overline B_c,10^{-9})}.
$$

Training normalizes both prediction and target over its complete training
set. It never normalizes render chunks independently. Validation and reporting
render raw brightness over the entire rotation, normalize once, and only then
select scored rows; supplied full-turn-normalized observations are not
renormalized on subsets. Thus a held-out subset need not have mean one.
Changes to gain before encoding, clipping, and thresholding can affect curve
shape even after normalization.

## 6. Inverse problem and Adam algorithm

### 6.1 Data split and initialization

The legacy default remains 12 training and 24 validation rows until the
experimental correction passes its promotion gate. The controlled correction
uses training indices $0,5,\ldots,355$ and validation indices
$2,7,\ldots,357$ (72 each). The remaining 216 rows do not select checkpoints.
The 12-frame control uses the same 72 validation indices. Explicit indices
are saved for replay. Every row contains all camera channels.

The initial ellipsoid is estimated from training horizontal lightcurves:

$$
e=\operatorname{clip}_{[1,2.5]}
\left(\operatorname{median}_{c\in\mathrm{horizontal}}
\frac{Q_{0.95}(y_{\cdot c})}{\max(Q_{0.05}(y_{\cdot c}),10^{-3})}\right),
\qquad (A,B,C)=(e,1,0.85).
$$

Here the quantiles are taken over training rows of the loaded curves.
For each template direction, form the ellipsoid's radial distance

$$
\rho_i=\left(u_{ix}^2/A^2+u_{iy}^2/B^2+u_{iz}^2/C^2\right)^{-1/2},
\qquad h_i=\log\rho_i-\frac{1}{N_V}\sum_j\log\rho_j.
$$

Initialization solves a linear least-squares problem for the nonconstant
$a$ coefficients,

$$
\mathbf a_{\mathrm{free}}^{(0)}=
\arg\min_{\mathbf a}\|Y_a[:,1:]\mathbf a-\mathbf h\|_2^2,
\qquad \mathbf b^{(0)}=\mathbf c^{(0)}=0.
$$

This gives a truncated harmonic approximation to the ellipsoid's log radius,
followed by the usual saturation in the mesh map. It does not initialize
the ellipsoid through the constant $b,c$ coefficients. Paired experiments
save and reuse the same base coefficients for every variant. The second start
adds seeded independent Gaussian perturbations with standard deviation 0.01
only to degree-1/2 coefficients; changing a seed alone does not alter the
deterministic ellipsoid initialization.

### 6.2 Objective

With normalized training targets $y_{tc}$ and soft predictions
$\widehat y^{\mathrm{soft}}_{tc}(\mathbf p)$, update $k$ minimizes

$$
\mathcal J_k(\mathbf p)=
\underbrace{\frac{1}{N_{\mathrm{train}}C}\sum_{t,c}
\left(\widehat y^{\mathrm{soft}}_{tc}(\mathbf p)-y_{tc}\right)^2}
 _{\mathcal L_{\mathrm{data}}}
+\lambda\,s_k\,\mathcal R(\mathbf p),\qquad\lambda=5\times10^{-3}.
$$

All samples and camera channels receive equal weight. The stage multiplier
$s_k$ changes during the run. Gradients propagate through harmonic fields,
mesh geometry, camera placement, interpolated normals, shadow sampling,
image blending, soft extraction, and mean normalization. Rasterization's
discrete choices, detached camera clipping/shadow-map extent values, and
the binary Otsu threshold limit this differentiation to the implemented
local surrogate.

### 6.2.1 Memory-bounded exact normalization gradient

For each update, first render raw training brightness $B$ in chunks under
`no_grad`. Differentiate the normalized loss with respect to this small
$N_{\mathrm{train}}\times C$ array. Away from the mean floor, with
$D_{tc}=\partial\mathcal L/\partial\widehat y_{tc}$,

$$
\frac{\partial\mathcal L}{\partial B_{kc}}
=\frac{D_{kc}}{\overline B_c}
-\frac{\sum_t D_{tc}B_{tc}}{N_{\mathrm{train}}\overline B_c^2}.
$$

Autograd also handles the implemented clamped-mean branch. Recompute each
renderer chunk with gradients and backpropagate these brightness derivatives.
Add the prior once, mask degrees once, clip once, and take one Adam update.
No renderer graph from a prior chunk is retained. Algebraic tests and CUDA
renderer tests compare this chain rule against an unchunked loss.

### 6.3 Staged updates

The default run has $T=300$ Adam updates indexed $k=0,\ldots,T-1$, an
escape stage of $E=30$ updates, and a stabilization stage of $S=50$ updates.
Set $\eta_{\mathrm{initial}}=8\times10^{-3}$,
$\eta_{\mathrm{base}}=2\times10^{-3}$, and
$\eta_{\min}=10^{-5}$.

During escape, $0\le k<E$,

$$
\eta_k=\eta_{\mathrm{base}}+
\frac{\eta_{\mathrm{initial}}-\eta_{\mathrm{base}}}{2}
\left[1+\cos\left(\pi\frac{k}{\max(E-1,1)}\right)\right],
\qquad s_k=0.25.
$$

Only gradients of coefficients with $\ell\le2$ are enabled. Higher-degree
coefficients keep their initialized values; they are still present in the
rendered shape and objective.

After escape, $E\le k<T$,

$$
\eta_k=\eta_{\min}+
\frac{\eta_{\mathrm{base}}-\eta_{\min}}{2}
\left[1+\cos\left(\pi\frac{k-E}{\max(T-E-1,1)}\right)\right].
$$

All coefficient gradients are enabled. The regularization multiplier is
$s_k=1$ during refinement; for the final $S$ updates it ramps as

$$
s_k=1+\frac{k-(T-S)+1}{S},\qquad T-S\le k<T.
$$

Thus stabilization ends with twice the base regularization. Mesh resolution
and maximum harmonic degree remain fixed throughout the run.

After backpropagation and any degree mask, the full gradient vector is
clipped to Euclidean norm at most 10. `torch.optim.Adam` then updates
$\mathbf p$, using its default moment decays $(0.9,0.999)$, epsilon
$10^{-8}$, and zero weight decay. Adam's moment state is preserved across
stages. No line search or explicit Jacobian is constructed.

In pseudocode:

```text
load observations and fixed calibration
build harmonic bases and fixed icosphere faces
split rotation rows into training and validation sets
p = harmonic ellipsoid initialization
initialize Adam; score p on validation; best = p

for k = 0, ..., T - 1:
    choose learning rate, degree mask, and regularization multiplier
    collect raw training brightness in chunks without renderer graphs
    differentiate complete-training-set normalized loss with respect to brightness
    rerender chunks and backpropagate the corresponding brightness derivatives
    backpropagate the weighted spectral prior once
    apply degree mask; clip gradient norm
    take one Adam step
    check that mesh vertices are finite
    periodically render the full turn, normalize, then score validation rows
    save p as best if validation RMSE improves

export best and evaluate it on the full supplied rotation sequence
```

`project_parameters` is the identity for this shape model. Although a
separate mesh validator can test transverse self-intersections, the
maintained fit disables that test and only checks finite vertices after an
update. Smooth saturation and the spectral prior control deformation; the
solver does not enforce a non-intersecting surface constraint.

### 6.4 Checkpoint selection and evaluation

The initial shape, every tenth update, and the final update are scored on
validation rows using hard image extraction. Selection minimizes

$$
\mathrm{RMSE}_{\mathrm{val}}(\mathbf p)=
\sqrt{\frac{1}{N_{\mathrm{val}}C}\sum_{t,c}
\left(\widehat y^{\mathrm{exact}}_{tc}(\mathbf p)
-y^{\mathrm{val}}_{tc}\right)^2}.
$$

All configured updates run; the exported mesh is the best validation
checkpoint, which can be earlier than the final Adam iterate. By default,
final evaluation uses every supplied rotation row. It reuses the fit's
renderer resolution (96 pixels), rather than automatically switching to
the standalone 256-pixel evaluation preset. Public reference meshes are
used during inversion only for post-fit reporting and do not enter this
objective or checkpoint selection. Checkpoint version 3 fingerprints calibration, response,
projection, phase convention, dataset hashes, code, initial coefficients, and
explicit splits. Incompatible resumes are rejected; old trajectories remain
readable. Checkpoints are atomically replaced. CUDA scatter reductions permit
small floating-point differences across repeated runs, not bitwise replay.

## 7. Fixed calibration and scope

Calibration screens each body separately using 72 phases, 21 cameras,
distances $\{2.5,3.5,5,8,16,32\}$ plus orthographic projection, material
fractions $\{0,.25,.5,.75,1\}$, responses `{linear,srgb}`, gains
$\{.5,1,2\}$, and the two phase conventions. Linear Lambert/LS endpoint
images are reused only after a direct-render mixing check. The top three
screened candidates, the legacy candidate and a shared-grid candidate are
verified over all 360 phases. Selection minimizes pooled full-turn RMSE.
A winner at distance 2.5 or 32 triggers one outward extension to 2 or 64.
Distances remain empirical image-model parameters, not recovered intrinsics.

Each body receives a versioned calibration file. A shared-grid candidate is
compared against the separate calibrations; sharing is eligible only if neither
body's full-turn error worsens by more than 5%. The controlled campaign keeps
explicit per-body files. Transfer to an uncovered asteroid requires an explicit
transfer choice for every calibration file version. File hashes and configuration/code identities
invalidate stale caches.

Validation and remaining-rotation scores are **conditional on calibration using
public full-turn observations**. They are not end-to-end held-out test scores.
Orthographic movies are independent forward-model controls and never replace
the inversion text targets.

The [completed correction campaign](docs/forward-model-correction-results.md)
misses its 20% asteroid-2 channel improvement gate. Its 72/72 per-body calibrated
profiles remain experimental; legacy fit defaults are retained.

This is a local, regularized non-convex inverse problem. Its recovered shape
depends on the sampled viewing geometry, finite harmonic and mesh
resolution, direct-light scattering approximation, and optimizer
initialization. The soft training objective and hard validation metric
serve different roles: the former supplies gradients, while the latter
selects the shape under the requested image measurement rules.
