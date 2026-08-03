#!/usr/bin/env python3
"""
visualize_fit.py — visual diagnostics for lightcurve shape inversion.

Outputs (all headless-safe):
  <prefix>_shapes.png       initial / fitted / reference, shaded, matched views
  <prefix>_error.png        init & fit colored by signed radial log-error
  <prefix>_sections.png     equatorial + meridional radius profiles + error hist
  <prefix>_compare.glb      one scene (init | reference | fit) for any glTF viewer
  <prefix>_interactive.html optional plotly viewer (if plotly is installed)

`fitter` only needs .dirs_t, .faces_t, .r0, .device  (duck-typed; ShapeFitter works).
`reference` is (verts, faces) — pass the star-shaped proxy used to generate the
data to isolate inversion error from the star-shape approximation error, or the
raw GT mesh to see both combined. reference=None -> init/fit comparison only
(secret models).
"""

import math

import numpy as np
import torch
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Physically meaningful viewpoints: sun is along -x, cameras equatorial ±26 deg.
VIEWS = [
    (25, 135, "oblique"),
    (0, 180, "from sun (-x)"),
    (0, 90, "equator (+y)"),
    (90, -90, "north pole"),
]


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _as_torch_mesh(mesh, device):
    if isinstance(mesh, trimesh.Trimesh):
        v = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        f = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        return v, f
    v, f = mesh
    return (torch.as_tensor(v, dtype=torch.float32, device=device),
            torch.as_tensor(f, dtype=torch.long, device=device))


@torch.no_grad()
def radial_support(verts, faces, dirs, chunk=4096, tol=1e-6):
    """Outermost hit distance from the origin along unit dirs (Möller–Trumbore
    specialized to a shared origin; same math as main.radial_field)."""
    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0
    n = torch.cross(e1, e2, dim=-1)
    w_u = torch.cross(v0, e2, dim=-1)
    w_v = torch.cross(e1, v0, dim=-1)
    a = (v0 * n).sum(-1)
    r = dirs.new_zeros(dirs.shape[0])
    for i in range(0, dirs.shape[0], chunk):
        d = dirs[i:i + chunk]
        S = d @ n.T
        ok = S.abs() > 1e-12
        inv = torch.where(ok, 1.0 / S, torch.zeros_like(S))
        t = a.unsqueeze(0) * inv
        u = -(d @ w_u.T) * inv
        v = -(d @ w_v.T) * inv
        hit = ok & (t > tol) & (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol)
        r[i:i + chunk] = torch.where(hit, t, torch.zeros_like(t)).max(1).values
    miss = r <= 0
    if bool(miss.any()):
        r[miss] = r[~miss].median()
    return r


def fibonacci_dirs(n, device):
    i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    z = 1.0 - 2.0 * i / n
    rho = torch.sqrt((1 - z * z).clamp_min(0))
    return torch.stack([rho * torch.cos(phi), rho * torch.sin(phi), z], 1)


# ---------------------------------------------------------------------------
# matplotlib 3D rendering
# ---------------------------------------------------------------------------

def _face_shade(verts, faces, light=(0.4, 0.3, 0.85), ambient=0.35):
    v = verts[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    l = np.asarray(light, float)
    l /= np.linalg.norm(l)
    return ambient + (1.0 - ambient) * np.clip(n @ l, 0.0, 1.0)


def _add_mesh(ax, verts, faces, base_rgb=(0.75, 0.75, 0.78),
              face_values=None, vmax=None, cmap="coolwarm"):
    shade = _face_shade(verts, faces)
    if face_values is None:
        rgb = np.clip(np.asarray(base_rgb)[None, :] * shade[:, None], 0, 1)
        rgba = np.concatenate([rgb, np.ones((faces.shape[0], 1))], 1)
    else:
        norm = colors.Normalize(-vmax, vmax)
        rgba = plt.get_cmap(cmap)(norm(face_values))
        rgba[:, :3] *= (0.55 + 0.45 * shade)[:, None]   # hue-preserving light
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=rgba,
                                         edgecolor="none"))


def _set_axes(ax, lim, elev, azim, title=None):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")                 # matches the forward model
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def plot_shape_grid(meshes, labels, base_cols, out_png, views=VIEWS,
                    suptitle=None):
    lim = 1.08 * max(np.abs(v).max() for v, _ in meshes)
    nr, nc = len(views), len(meshes)
    fig = plt.figure(figsize=(3.1 * nc, 2.9 * nr))
    for i, (elev, azim, vname) in enumerate(views):
        for j, ((v, f), lab, col) in enumerate(zip(meshes, labels, base_cols)):
            ax = fig.add_subplot(nr, nc, i * nc + j + 1, projection="3d")
            _add_mesh(ax, v, f, base_rgb=col)
            _set_axes(ax, lim, elev, azim, lab if i == 0 else None)
            if j == 0:
                ax.text2D(-0.08, 0.5, vname, transform=ax.transAxes,
                          rotation=90, va="center", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_error_maps(meshes, face_vals, labels, out_png, vmax,
                    views=VIEWS[:3], cmap="coolwarm"):
    lim = 1.08 * max(np.abs(v).max() for v, _ in meshes)
    nr, nc = len(views), len(meshes)
    fig = plt.figure(figsize=(3.4 * nc, 3.0 * nr))
    for i, (elev, azim, vname) in enumerate(views):
        for j, ((v, f), e, lab) in enumerate(zip(meshes, face_vals, labels)):
            ax = fig.add_subplot(nr, nc, i * nc + j + 1, projection="3d")
            _add_mesh(ax, v, f, face_values=e, vmax=vmax, cmap=cmap)
            _set_axes(ax, lim, elev, azim, lab if i == 0 else None)
            if j == 0:
                ax.text2D(-0.08, 0.5, vname, transform=ax.transAxes,
                          rotation=90, va="center", fontsize=9)
    sm = cm.ScalarMappable(norm=colors.Normalize(-vmax, vmax),
                           cmap=plt.get_cmap(cmap))
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.65, pad=0.02,
                 label=r"$\Delta \log r$  (mesh $-$ reference)")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_sections(meshes_t, labels, errs_pct, out_png, n=721):
    """Polar slices: init/fit/reference overlaid, plus log-radius error hist."""
    device = meshes_t[0][0].device
    t = torch.linspace(0, 2 * math.pi, n, device=device)
    planes = [
        ("equatorial (z=0)",
         torch.stack([torch.cos(t), torch.sin(t), torch.zeros_like(t)], 1)),
        ("meridian (x-z)",
         torch.stack([torch.cos(t), torch.zeros_like(t), torch.sin(t)], 1)),
        ("meridian (y-z)",
         torch.stack([torch.zeros_like(t), torch.cos(t), torch.sin(t)], 1)),
    ]
    styles = [dict(color="steelblue", ls="--", lw=1.4),   # init
              dict(color="crimson", ls="-", lw=1.6),      # fit
              dict(color="k", ls="-", lw=2.2, alpha=.7)]  # reference
    order = [2, 0, 1]                                     # ref underneath
    tn = _to_np(t)
    fig = plt.figure(figsize=(11, 9))
    for k, (name, dirs) in enumerate(planes):
        ax = fig.add_subplot(2, 2, k + 1, projection="polar")
        for idx in order:
            v, f = meshes_t[idx]
            r = _to_np(radial_support(v, f, dirs))
            ax.plot(tn, r, label=labels[idx], **styles[idx])
        ax.set_title(name, fontsize=10)
        ax.set_rlabel_position(45)
        if k == 0:
            ax.legend(fontsize=8, loc="lower left",
                      bbox_to_anchor=(1.02, 0.85))
    ax = fig.add_subplot(2, 2, 4)
    ax.hist(errs_pct[0], bins=40, alpha=0.6, color="steelblue",
            label=f"init  (rms {np.sqrt((errs_pct[0]**2).mean()):.1f}%)")
    ax.hist(errs_pct[1], bins=40, alpha=0.6, color="crimson",
            label=f"fit   (rms {np.sqrt((errs_pct[1]**2).mean()):.1f}%)")
    ax.set_xlabel(r"radial error \$100\,\Delta\log r$  [%]")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# interactive exports
# ---------------------------------------------------------------------------

def export_glb(meshes, labels, out_path, vertex_colors=None):
    ext = max(float(v[:, 0].max() - v[:, 0].min()) for v, _ in meshes)
    scene = trimesh.Scene()
    nmesh = len(meshes)
    for i, ((v, f), lab) in enumerate(zip(meshes, labels)):
        off = np.array([(i - (nmesh - 1) / 2) * 1.35 * ext, 0, 0])
        m = trimesh.Trimesh(v + off, f, process=False)
        if vertex_colors is not None and vertex_colors[i] is not None:
            m.visual.vertex_colors = vertex_colors[i]
        scene.add_geometry(m, geom_name=lab)
    try:
        scene.export(out_path)
        return out_path
    except Exception as e:                     # pragma: no cover
        print(f"[viz] GLB export failed ({e})")
        return None


def export_plotly_html(meshes, labels, err_vertex, vmax, out_html):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    ext = max(float(v[:, 0].max() - v[:, 0].min()) for v, _ in meshes)
    fig = go.Figure()
    flat = dict(flatshading=True,
                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.05))
    base = ["lightsteelblue", None, "darkseagreen"]     # fit gets intensity
    nmesh = len(meshes)
    for i, ((v, f), lab) in enumerate(zip(meshes, labels)):
        off = (i - (nmesh - 1) / 2) * 1.35 * ext
        kw = dict(x=v[:, 0] + off, y=v[:, 1], z=v[:, 2],
                  i=f[:, 0], j=f[:, 1], k=f[:, 2], name=lab, **flat)
        if err_vertex is not None and err_vertex[i] is not None:
            kw.update(intensity=err_vertex[i], colorscale="RdBu",
                      reversescale=True, cmin=-vmax, cmax=vmax,
                      colorbar=dict(title="Δlog r"))
        else:
            kw.update(color=base[i % len(base)] or "lightsteelblue")
        fig.add_trace(go.Mesh3d(**kw))
    fig.update_layout(scene=dict(aspectmode="data"),
                      title=" | ".join(labels), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(out_html)
    return out_html


# ---------------------------------------------------------------------------
# animations (optional)
# ---------------------------------------------------------------------------

def turntable_gif(verts, faces, out_gif, face_vals=None, vmax=None,
                  frames=36, fps=12, elev=22):
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh(ax, verts, faces, face_values=face_vals, vmax=vmax)
    lim = 1.08 * np.abs(verts).max()
    _set_axes(ax, lim, elev, 0)
    anim = FuncAnimation(fig, lambda k: ax.view_init(elev, k * 360 / frames),
                         frames=frames)
    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


@torch.no_grad()
def evolution_gif(fitter, p_seq, reference, out_gif="fit_evolution.gif",
                  fps=5, elev=22, azim=135):
    """Animate the solver trajectory (record p per LM iteration in `solve`)."""
    ref_v, ref_f = _as_torch_mesh(reference, fitter.device)
    r_ref = radial_support(ref_v, ref_f, fitter.dirs_t)
    faces = _to_np(fitter.faces_t)
    frames = []
    for pp in p_seq:
        r = fitter.r0 * torch.exp(pp.detach())
        s = torch.exp(torch.mean(torch.log(r_ref) - torch.log(r)))
        v = _to_np((s * r)[:, None] * fitter.dirs_t)
        e = _to_np(torch.log(s * r) - torch.log(r_ref))
        frames.append((v, e[faces].mean(1)))
    vmax = max(1e-3, np.percentile(np.abs(np.stack([e for _, e in frames])), 99))
    lim = 1.08 * max(np.abs(v).max() for v, _ in frames)
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")

    def update(k):
        ax.cla()
        _add_mesh(ax, frames[k][0], faces, face_values=frames[k][1], vmax=vmax)
        _set_axes(ax, lim, elev, azim, f"iteration {k}")
    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_fit(fitter, p0, p, reference=None, *, ref_label="reference",
                  out_prefix="fit_viz", align_scale=True, interactive=True,
                  turntable=False):
    """Full report. Returns metrics dict; writes PNG/GLB/HTML files."""
    device, faces_np = fitter.device, _to_np(fitter.faces_t)

    # -- no reference (secret models): init/fit comparison only --------------
    if reference is None:
        ms, labs = [], ["initial", "fitted"]
        for pp in (p0, p):
            r = fitter.r0 * torch.exp(pp.detach())
            ms.append((_to_np(r[:, None] * fitter.dirs_t), faces_np))
        plot_shape_grid(ms, labs, [(0.55, 0.65, 0.85), (0.9, 0.55, 0.35)],
                        f"{out_prefix}_shapes.png")
        export_glb(ms, labs, f"{out_prefix}_compare.glb")
        return {}

    ref_v, ref_f = _as_torch_mesh(reference, device)
    r_ref_t = radial_support(ref_v, ref_f, fitter.dirs_t)

    # -- gauge-aligned meshes + per-vertex errors ----------------------------
    meshes_np, meshes_t, errs, metrics = [], [], [], {}
    for tag, pp in (("initial", p0), ("fitted", p)):
        r = fitter.r0 * torch.exp(pp.detach())
        s = (torch.exp(torch.mean(torch.log(r_ref_t) - torch.log(r)))
             if align_scale else torch.ones((), device=device))
        v_t = (s * r)[:, None] * fitter.dirs_t
        e = torch.log(s * r) - torch.log(r_ref_t)      # zero-mean if aligned
        meshes_t.append((v_t, fitter.faces_t))
        meshes_np.append((_to_np(v_t), faces_np))
        errs.append(_to_np(e))
        metrics[tag] = dict(rms=float(e.pow(2).mean().sqrt()),
                            max=float(e.abs().max()), scale=float(s))
    meshes_t.append((ref_v, ref_f))
    meshes_np.append((_to_np(ref_v), _to_np(ref_f)))
    labels = ["initial", "fitted", ref_label]

    # -- extra metrics: volume ratio + radial Hausdorff ----------------------
    dense = fibonacci_dirs(4096, device)
    r_ref_d = radial_support(ref_v, ref_f, dense)
    for tag, (v_t, f_t) in zip(("initial", "fitted"), meshes_t[:2]):
        r_d = radial_support(v_t, f_t, dense)
        metrics[tag]["hausdorff_rel"] = float(
            (r_d - r_ref_d).abs().max() / r_ref_d.mean())
        try:
            vol = abs(trimesh.Trimesh(*meshes_np[labels.index(tag)],
                                      process=False).volume)
            vol_ref = abs(trimesh.Trimesh(*meshes_np[2], process=False).volume)
            metrics[tag]["vol_ratio"] = vol / vol_ref
        except Exception:
            metrics[tag]["vol_ratio"] = float("nan")

    print(f"[viz] gauge scales: init x{metrics['initial']['scale']:.3f}, "
          f"fit x{metrics['fitted']['scale']:.3f}")
    print("[viz]           rms(dlogr)  max|dlogr|  vol/vol_ref  Hausdorff/<r>")
    for tag in ("initial", "fitted"):
        m = metrics[tag]
        print(f"[viz] {tag:8s}  {100*m['rms']:8.2f}%  {100*m['max']:8.2f}%  "
              f"{m['vol_ratio']:10.3f}  {m['hausdorff_rel']:10.3f}")

    # -- figures --------------------------------------------------------------
    plot_shape_grid(meshes_np, labels,
                    [(0.55, 0.65, 0.85), (0.9, 0.55, 0.35), (0.68, 0.75, 0.68)],
                    f"{out_prefix}_shapes.png",
                    suptitle="shape recovery (gauge-aligned)")
    vmax = max(1e-3, float(np.percentile(np.abs(np.concatenate(errs)), 99)))
    face_vals = [e[faces_np].mean(1) for e in errs]
    plot_error_maps(meshes_np[:2], face_vals, ["initial", "fitted"],
                    f"{out_prefix}_error.png", vmax)
    plot_sections(meshes_t, labels, [100 * e for e in errs],
                  f"{out_prefix}_sections.png")

    # -- interactive ----------------------------------------------------------
    cmap = plt.get_cmap("coolwarm")
    fit_vc = (cmap(colors.Normalize(-vmax, vmax)(errs[1])) * 255).astype(np.uint8)
    export_glb(meshes_np, labels, f"{out_prefix}_compare.glb",
               vertex_colors=[None, fit_vc, None])
    if interactive:
        export_plotly_html(meshes_np, labels, [None, errs[1], None], vmax,
                           f"{out_prefix}_interactive.html")
    if turntable:
        turntable_gif(*meshes_np[1], f"{out_prefix}_turntable.gif",
                      face_vals=face_vals[1], vmax=vmax)
    print(f"[viz] wrote {out_prefix}_shapes/error/sections.png, "
          f"{out_prefix}_compare.glb")
    return metrics


# ---------------------------------------------------------------------------
# standalone smoke test (no challenge data needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    class _FitterLike:
        def __init__(self, dirs, faces, r0=0.5):
            self.dirs_t, self.faces_t, self.r0 = dirs, faces, r0
            self.device = dirs.device

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sph = trimesh.creation.icosphere(3)
    dirs = F.normalize(torch.tensor(sph.vertices, dtype=torch.float32,
                                    device=dev), dim=-1)
    faces = torch.tensor(sph.faces, dtype=torch.long, device=dev)
    fl = _FitterLike(dirs, faces)
    # reference = 1.4:1:0.8 ellipsoid, "fit" = noisy version, init = sphere
    p_ref = -0.5 * torch.log(dirs[:, 0]**2 / 1.4**2 + dirs[:, 1]**2
                             + dirs[:, 2]**2 / 0.8**2)
    p_ref = p_ref - p_ref.mean()
    ref = (fl.r0 * torch.exp(p_ref)[:, None] * dirs, faces)
    visualize_fit(fl, torch.zeros_like(p_ref),
                  p_ref + 0.05 * torch.randn_like(p_ref), reference=ref,
                  out_prefix="viz_smoke")