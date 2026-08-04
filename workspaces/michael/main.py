#!/usr/bin/env python3
"""
main.py — Gauss-Newton / Levenberg-Marquardt solver and test harness.

Model-agnostic: operates purely through PhotometricModel (ShapeModel +
PhotometricEngine).  The LM step supports three backends:

  frozen : explicit J with visibility table DETACHED at linearization.
  exact  : explicit J through everything (small problems / debugging).
  cg     : matrix-free exact GN with Jacobi-PCG.

The outer loop implements Nielsen damping, stage scheduling, and convergence
checks.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp, vmap

from common import (
    PhotometricModel,
    LightcurvePhotometricEngine,
    build_view_dirs,
    load_asteroid_mesh,
    load_data,
)

# --------------------------------------------------------------------------- #
# Jacobian and linear algebra helpers
# --------------------------------------------------------------------------- #

def build_jacobian(fn, p, chunk=16):
    """J[m, i] = d r_m / d p_i via chunked forward-mode AD. Returns (M, n)."""
    n = p.numel()
    I = torch.eye(n, device=p.device, dtype=p.dtype)
    blocks = []
    for i in range(0, n, chunk):
        Jc = vmap(lambda v: jvp(fn, (p,), (v,))[1])(I[i : i + chunk])
        blocks.append(Jc)
    return torch.cat(blocks, 0).T.contiguous()


def _pcg(A, b, Minv, iters=50, tol=1e-3):
    """Jacobi-preconditioned CG for the SPD system A x = b."""
    x = torch.zeros_like(b)
    bn = float(b.norm())
    if bn < 1e-30:
        return x
    r = b.clone()
    z = Minv * r
    d = z.clone()
    rz = float(r @ z)
    for _ in range(iters):
        Ad = A(d)
        alpha = rz / max(float(d @ Ad), 1e-30)
        x = x + alpha * d
        r = r - alpha * Ad
        if float(r.norm()) <= tol * bn:
            break
        z = Minv * r
        rz_new = float(r @ z)
        d = z + (rz_new / rz) * d
        rz = rz_new
    return x


# --------------------------------------------------------------------------- #
# LM update (three backends)
# --------------------------------------------------------------------------- #

def compute_update(
    model: PhotometricModel,
    p,
    gamma,
    lam,
    method="frozen",
    lin=None,
    jac_chunk=16,
    cg_iters=50,
    cg_tol=1e-3,
    hutch_probes=8,
):
    """
    One damped Gauss-Newton (LM) step around p:  (JtJ + lam*D) delta = -Jt r.

    Returns (delta, pred_red, lin). Pass `lin` back in when retrying with a
    larger lam: the expensive linearization (J / VJP graph / diag estimate)
    is reused, only the cheap solve is repeated.
    """
    # ---------- explicit-J backends ----------------------------------------
    if method in ("frozen", "exact"):
        if lin is None:
            if method == "frozen":
                with torch.no_grad():
                    verts, faces = model.shape.compute_geometry(p)
                    ctx = model.engine.linearization_context(verts, faces, gamma)
                fn = lambda q: model.residuals_at(q, gamma, context=ctx)
            else:
                fn = lambda q: model.residuals_at(q, gamma)
            with torch.no_grad():
                r = fn(p)
            J = build_jacobian(fn, p, chunk=jac_chunk).double()
            JtJ = J.T @ J
            Jtr = J.T @ r.double()
            D = JtJ.diagonal().clone()
            D = D.clamp_min(1e-10 * float(D.max()) + 1e-300)
            lin = dict(JtJ=JtJ, Jtr=Jtr, D=D)
        JtJ, Jtr, D = lin["JtJ"], lin["Jtr"], lin["D"]
        delta = torch.linalg.solve(JtJ + lam * torch.diag(D), -Jtr)
        pred = 0.5 * float(delta @ (lam * D * delta - Jtr))
        return delta.to(p.dtype), pred, lin

    # ---------- matrix-free exact GN with PCG -------------------------------
    if method != "cg":
        raise ValueError(f"unknown method {method!r}")
    if lin is None:
        p_leaf = p.detach().clone().requires_grad_(True)
        r_graph = model.residuals_at(p_leaf, gamma)
        def JT(u):
            (g,) = torch.autograd.grad(
                r_graph, p_leaf, grad_outputs=u, retain_graph=True
            )
            return g

        g0 = JT(r_graph.detach())
        Dd = torch.zeros_like(p)
        M = r_graph.numel()
        for _ in range(hutch_probes):
            z = (
                torch.randint(0, 2, (M,), device=p.device, dtype=p.dtype) * 2.0
                - 1.0
            )
            u = JT(z)
            Dd += u * u
        Dd = Dd / hutch_probes
        Dd = Dd.clamp_min(1e-8 * float(Dd.max()) + 1e-30)
        lin = dict(JT=JT, g=g0, D=Dd)
    JT, g, D = lin["JT"], lin["g"], lin["D"]

    fn_fwd = lambda q: model.residuals_at(q, gamma)

    def Av(v):
        Jv = jvp(fn_fwd, (p,), (v,))[1]
        return JT(Jv) + lam * D * v

    delta = _pcg(
        Av, -g, 1.0 / ((1.0 + lam) * D), iters=cg_iters, tol=cg_tol
    )
    JtJd = Av(delta) - lam * D * delta
    pred = float(-(delta @ g) - 0.5 * (delta @ JtJd))
    return delta, pred, lin


# --------------------------------------------------------------------------- #
# LM outer loop
# --------------------------------------------------------------------------- #

def solve(
    model: PhotometricModel,
    p0,
    stages=None,
    lam0=1e-2,
    max_retries=10,
    step_tol=1e-4,
    verbose=True,
    record_trajectory=False,
):
    """
    Levenberg-Marquardt driver. Each stage fixes (gamma, method); lam is
    shared across stages and adapted by the Nielsen rho-rule. Damping retries
    reuse the linearization via `lin`. Trial steps are always scored with the
    EXACT residual (fresh visibility tables at the trial point), so the
    frozen-J approximation can only slow convergence, never corrupt it.

    If `record_trajectory` is True, a third element containing the parameter
    trajectory (initial + one clone per accepted step) is returned for GIFs.
    """
    if stages is None:
        stages = [
            dict(gamma=0.06, iters=6, method="frozen"),
            dict(gamma=0.03, iters=10, method="frozen"),
            dict(gamma=0.015, iters=8, method="frozen"),
        ]
    p, lam, history = p0.clone(), lam0, []
    trajectory = [p0.detach().clone()] if record_trajectory else None
    model.set_params(p0)
    for si, st in enumerate(stages):
        gamma, method = st["gamma"], st.get("method", "frozen")
        kw = st.get("update_kwargs", {})
        for it in range(st["iters"]):
            with torch.no_grad():
                r = model.residuals(gamma=gamma, frozen_vis=False)
                cost = 0.5 * float(r @ r)
            lin, accepted, rho, cost_new = None, False, float("nan"), cost
            for _ in range(max_retries):
                delta, pred, lin = compute_update(
                    model, p, gamma, lam, method=method, lin=lin, **kw
                )
                p_trial = (p + delta).clamp(-2.0, 2.0).detach()
                model.set_params(p_trial)
                with torch.no_grad():
                    r_new = model.residuals(gamma=gamma, frozen_vis=False)
                    cost_new = 0.5 * float(r_new @ r_new)
                rho = (cost - cost_new) / max(pred, 1e-30)
                if math.isfinite(cost_new) and rho > 1e-3:
                    p = p_trial
                    lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-12)
                    accepted = True
                    if record_trajectory:
                        trajectory.append(p.detach().clone())
                    break
                lam = min(lam * 4.0, 1e9)
                model.set_params(p)
            step = float(delta.norm()) if accepted else 0.0
            n_photo = model.engine.data.numel()
            photo = (
                float((r_new[:n_photo] ** 2).sum())
                if accepted
                else float((r[:n_photo] ** 2).sum())
            )
            history.append(
                dict(
                    stage=si,
                    iter=it,
                    gamma=gamma,
                    cost=cost,
                    cost_new=cost_new if accepted else cost,
                    rho=rho,
                    lam=lam,
                    step=step,
                    photo=photo,
                )
            )
            if verbose:
                rms = math.sqrt(photo / n_photo)
                print(
                    f"[s{si} i{it:02d}] g={gamma:.3f} cost {cost:.4e} -> "
                    f"{history[-1]['cost_new']:.4e}  rho={rho:5.2f} "
                    f"lam={lam:.1e} |dp|={step:.2e}  photo-rms={rms:.3f}"
                )
            if not accepted:
                if verbose:
                    print("   no acceptable step — moving to next stage")
                break
            if step < step_tol * max(float(p.norm()), 1.0):
                if verbose:
                    print("   converged (step tolerance)")
                break
    if record_trajectory:
        return p, history, trajectory
    return p, history


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def fd_check(model, p, gamma, h=1e-3):
    """Directional finite-difference check of the EXACT Jacobian (jvp path)."""
    v = torch.randn_like(p)
    v = v / v.norm()
    fn = lambda q: model.residuals_at(q, gamma)
    with torch.no_grad():
        fd = (fn(p + h * v) - fn(p - h * v)) / (2 * h)
    an = jvp(fn, (p,), (v,))[1]
    return float((fd - an).norm() / (an.norm() + 1e-30))


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #

def test_fit(
    model_num=1,
    fit_stride=1,
    subdiv=3,
    lam_lap=30.0,
    lam_scale=10.0,
    seed=0,
    device=None,
    make_plots=True,
):
    """
    Fit a star-shaped model to the real asteroid lightcurve data.

    The reference mesh is the challenge STL, centered and scaled to fit in a
    unit cube.  No synthetic fallback is used.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cpu":
        print("[test] CPU detected — using a smaller template for a smoke test")
        subdiv = 2

    # ---- reference mesh from challenge data --------------------------------
    verts_gt, faces_gt = load_asteroid_mesh(model_num, device)
    vmin, vmax = verts_gt.min(0).values, verts_gt.max(0).values
    verts_gt = (verts_gt - (vmin + vmax) / 2) / (vmax - vmin).max()
    print(
        f"[test] reference mesh: {verts_gt.shape[0]} verts, "
        f"{faces_gt.shape[0]} faces"
    )

    # ---- real intensity lightcurves ----------------------------------------
    dataset = load_data(model_num, origin="real", curve_type="intensity")
    cols = list(dataset.data.columns)
    data_df = dataset.data.dropna()
    if len(data_df) < len(dataset.data):
        print(f"[test] dropped {len(dataset.data) - len(data_df)} frames with NaNs")
    data = torch.tensor(data_df.to_numpy(dtype=np.float32), device=device)
    data = data / data.mean(dim=0, keepdim=True).clamp_min(1e-6)

    frame_idx = torch.tensor(data_df.index.to_numpy(dtype=np.float32), device=device)
    n_frames = int(frame_idx.max() - frame_idx.min() + 1)
    angles = 2 * math.pi * (frame_idx - frame_idx.min()) / n_frames

    view_dirs0 = build_view_dirs(cols, device)
    sun_dir0 = torch.tensor([-1.0, 0.0, 0.0], device=device)
    print(f"[test] real data: {data.shape[0]} frames x {len(cols)} cameras")

    # ---- fit ---------------------------------------------------------------
    from star_forward import StarShapeModel, make_template, generate_initial_guess
    from visualize_fit import (
        plot_intensity_and_binary_grids,
        plot_mesh_triptych,
        evolution_gif_from_model,
        visualize_fit,
    )

    dirs_t, faces_t, L = make_template(subdiv, device)
    shape = StarShapeModel(
        dirs_t, faces_t, L, r0=0.5, lam_lap=lam_lap, lam_scale=lam_scale
    )
    engine = LightcurvePhotometricEngine(
        data[::fit_stride],
        angles[::fit_stride],
        sun_dir0,
        view_dirs0,
        use_compile=False,
        dir_chunk=2,
        frame_chunk=60,
    )
    model = PhotometricModel(shape, engine)

    print(
        f"[test] model: {model.num_params()} params, "
        f"{engine.data.numel()} photometric residuals, "
        f"{engine.uniq_dirs.shape[0]} unique visibility directions"
    )

    p0 = generate_initial_guess(engine, shape, ab=None, gamma=0.05)
    print(
        f"[test] jacobian FD check (exact path): "
        f"rel err {fd_check(model, p0, 0.03):.2e}"
    )

    t0 = time.time()
    p, hist, trajectory = solve(model, p0, record_trajectory=True)
    print(f"[test] solve time {time.time()-t0:.1f}s")

    # ---- metrics vs. the reference mesh ------------------------------------
    from star_forward import radial_field

    r_gt = radial_field(verts_gt, faces_gt, shape.dirs).cpu().numpy()
    report = {}
    for name, pp in (("init", p0), ("fit ", p)):
        r_m = (shape.r0 * torch.exp(pp)).cpu().numpy()
        e = np.log(r_m) - np.log(r_gt)
        e -= e.mean()
        report[name.strip()] = (
            float(np.sqrt((e**2).mean())),
            float(np.abs(e).max()),
        )
        print(
            f"[test] {name}: radius RMS err {100*report[name.strip()][0]:.2f}%"
            f"   max {100*report[name.strip()][1]:.2f}%"
        )

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with torch.no_grad():
            Bf = model.engine.brightness(model.shape, gamma=0.015)
            model_curve = (Bf / Bf.mean(0, keepdim=True)).cpu().numpy()
        dat = engine.data.cpu().numpy()
        x = np.degrees(engine.angles.cpu().numpy())

        # 3x7 grids: intensity and Otsu-binary
        plot_intensity_and_binary_grids(
            dat, model_curve, x, cols, prefix="gn_fit"
        )

        # classic per-camera curve comparison
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, ci in zip(axes, (0, len(cols) // 2, len(cols) - 1)):
            ax.plot(x, dat[:, ci], "k.", ms=2, label=f"data {cols[ci]}")
            ax.plot(x, model_curve[:, ci], "r-", lw=1.2, label="fit")
            ax.legend()
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("rotation angle [deg]")
        fig.tight_layout()
        fig.savefig("gn_fit_curves.png", dpi=140)
        plt.close(fig)

        plt.figure(figsize=(6, 4))
        plt.semilogy([h["cost_new"] for h in hist], ".-")
        plt.xlabel("LM iteration")
        plt.ylabel("0.5 ||r||^2")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("gn_fit_convergence.png", dpi=140)
        plt.close()

        plt.figure(figsize=(5, 5))
        r_m = (shape.r0 * torch.exp(p)).cpu().numpy()
        s = np.exp(np.mean(np.log(r_gt) - np.log(r_m)))
        plt.plot(r_gt, s * r_m, ".", ms=3)
        lim = [r_gt.min() * 0.9, r_gt.max() * 1.1]
        plt.plot(lim, lim, "k--", lw=1)
        plt.xlabel("ground-truth radius")
        plt.ylabel("fitted radius (scale-aligned)")
        plt.tight_layout()
        plt.savefig("gn_fit_radii.png", dpi=140)
        plt.close()

        # 1x3 mesh triptych
        init_mesh = (shape.r0 * torch.exp(p0))[:, None] * shape.dirs, shape.faces
        fit_mesh = (shape.r0 * torch.exp(p))[:, None] * shape.dirs, shape.faces
        plot_mesh_triptych(
            init_mesh,
            fit_mesh,
            (verts_gt, faces_gt),
            "gn_fit_mesh_triptych.png",
            title="Initial / fitted / reference mesh",
        )

        # full diagnostic report (shaded grids, error maps, sections, GLB)
        class _FitterLike:
            def __init__(self, shape):
                self.dirs_t = shape.dirs
                self.faces_t = shape.faces
                self.r0 = shape.r0
                self.device = shape.dirs.device

        visualize_fit(
            _FitterLike(shape), p0, p, reference=(verts_gt, faces_gt),
            out_prefix="gn_fit", interactive=True, turntable=False,
        )

        # evolution GIF
        evolution_gif_from_model(
            shape, trajectory, (verts_gt, faces_gt),
            out_gif="gn_fit_evolution.gif", fps=4
        )

        print(
            "[test] wrote gn_fit_intensity_grid.png / gn_fit_binary_grid.png / "
            "gn_fit_mesh_triptych.png / gn_fit_evolution.gif"
        )

    return dict(p=p, p0=p0, history=hist, model=model, report=report)


if __name__ == "__main__":
    test_fit(model_num=1)
