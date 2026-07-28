"""
Forward-simulate lightcurves from a given STL shape and compare against the
challenge_data lightcurves (real + blender) for the corresponding asteroid
model.

Two kinds of curve are simulated:

- "intensity": the Lommel-Seeliger + Lambert (LSL) scattering forward
  operator (same physics as `ConvexInversion.build_forward` in
  workspaces/magnus/convexinversion.py and
  `hac.convex_forward.convex_forward_model_gpu`).

- "binary": a linear forward model for the challenge's "number of bright
  pixels" curve (see below).

This is a CPU-only re-implementation: the shape (stls/asteroid2_simplified.stl,
16 facets) is far too small to need the CuPy/GPU path.

## Binary curve model

Per https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/, the
binary curve is the count of pixels in the rendered frame whose grayscale
value clears an Otsu threshold computed once from the first frame -- i.e.
it approximates the projected area of the "clearly sunlit" part of the
asteroid, not the total reflected light.

`hac/simulate.py::simulate_rotation_frames` renders with a single
directional light and `ambient=0, diffuse=1, specular=0` (pure Lambertian,
PBR), so -- if that is indeed the renderer that produced the
challenge_data ground truth, which we haven't independently confirmed --
a facet's rendered pixel value only depends on its sun angle mu0, not the
camera angle mu. Under that assumption, thresholding the rendered image is
approximately thresholding mu0 directly, which motivates
`simulate_lightcurves_binary_illumination` as the more physically grounded
of the two binary models below. `simulate_lightcurves_binary` (mu-only,
silhouette/grazing-angle cutoff) is kept in the module for reference but is
no longer part of the default comparison pipeline in `main()` -- across
every fit tried, mu0-thresholding consistently outperformed it.

Both are linear in the facet areas for a fixed threshold, mirroring the
structure of the intensity forward operator / ConvexInversion's `A` matrix.

Usage:
    python forward_sim_stl.py
    python forward_sim_stl.py --stl stls/asteroid2_simplified.stl --asteroid 2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh

from hac.constants import CAMERA_ANGLE_ALPHA, MODEL_BASE_RADII
from hac.convex_forward import get_unit_vectors
from hac.data import load_data
from hac.mesh import normalize_mesh_to_challenge_cylinder

WORKSPACE_DIR = Path(__file__).resolve().parent


def facet_normals_and_areas(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-facet outward unit normals and areas, as in ConvexInversion.__init__."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    cross = np.cross(e1, e2)
    mag = np.linalg.norm(cross, axis=1)
    normals = cross / mag[:, None]
    areas = 0.5 * mag
    return normals, areas


def facet_geometry(
    normals: np.ndarray,
    cam_keys: list[str],
    e_sun: np.ndarray,
    n_phases: int = 360,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Precompute mu (facet-vs-observer) and mu0 (facet-vs-sun) cosines for
    every camera and rotational phase, shared by all curve models below.

    Returns (phases, {cam: (mu, mu0)}) where mu, mu0 have shape
    (n_phases, n_facets).
    """
    phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)
    cos_p, sin_p = np.cos(phases), np.sin(phases)

    geometry = {}
    for cam in cam_keys:
        phi_deg = float(cam.split("_")[0])
        alpha_deg = CAMERA_ANGLE_ALPHA[cam]
        _, e_obs = get_unit_vectors(phi_deg, alpha_deg)

        # Rotate sun & observer through rotational phase (z-axis rotation)
        e_sun_rot = np.column_stack([
            cos_p * e_sun[0] - sin_p * e_sun[1],
            sin_p * e_sun[0] + cos_p * e_sun[1],
            np.full(n_phases, e_sun[2]),
        ])
        e_obs_rot = np.column_stack([
            cos_p * e_obs[0] - sin_p * e_obs[1],
            sin_p * e_obs[0] + cos_p * e_obs[1],
            np.full(n_phases, e_obs[2]),
        ])

        mu0 = e_sun_rot @ normals.T
        mu = e_obs_rot @ normals.T
        geometry[cam] = (mu, mu0)

    return phases, geometry


def _normalize_curves(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for cam, b in raw.items():
        m = b.mean()
        out[cam] = b / m if m > 0 else np.full_like(b, np.nan)
    return out


def simulate_lightcurves(
    geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    areas: np.ndarray,
    lambert_weight: float = 0.1,
) -> dict[str, np.ndarray]:
    """LSL "intensity" forward model: total reflected light per phase."""
    raw = {}
    for cam, (mu, mu0) in geometry.items():
        mask = (mu0 > 0) & (mu > 0)
        # eq. 2 of doi:10.48550/arXiv.2502.16455
        S = np.where(
            mask, mu * mu0 / np.maximum(mu + mu0, 1e-30) + lambert_weight * mu * mu0, 0.0
        )
        raw[cam] = (S * areas).sum(axis=1)
    return _normalize_curves(raw)


def simulate_lightcurves_lambertian(
    geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    areas: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Pure Lambertian law (no Lommel-Seeliger term at all): S = mu * mu0.
    Has no free parameter -- included as a baseline since
    `hac/simulate.py::simulate_rotation_frames` renders the challenge
    Blender frames with `ambient=0, diffuse=1, specular=0`, i.e. exactly
    this law, so it's the "correct" model if that renderer produced the
    ground truth (see module docstring).
    """
    raw = {}
    for cam, (mu, mu0) in geometry.items():
        mask = (mu0 > 0) & (mu > 0)
        S = np.where(mask, mu * mu0, 0.0)
        raw[cam] = (S * areas).sum(axis=1)
    return _normalize_curves(raw)


def simulate_lightcurves_binary(
    geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    areas: np.ndarray,
    mu_threshold: float = 0.3,
) -> dict[str, np.ndarray]:
    """
    Silhouette / grazing-angle cutoff: sums the projected area (area * mu)
    of sunlit facets that face the camera more than `mu_threshold`.

    Ignores the sun angle mu0 (beyond requiring mu0 > 0), so it can't
    capture terminator/self-shadow darkening -- only "am I facing the
    camera edge-on or not".
    """
    raw = {}
    for cam, (mu, mu0) in geometry.items():
        mask = (mu0 > 0) & (mu > mu_threshold)
        raw[cam] = (np.where(mask, mu, 0.0) * areas).sum(axis=1)
    return _normalize_curves(raw)


def simulate_lightcurves_binary_illumination(
    geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    areas: np.ndarray,
    mu0_threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Lambertian-brightness cutoff: a facet's rendered pixel value is
    hypothesised to be proportional to mu0 alone (ambient=0, diffuse=1,
    specular=0 directional-light rendering -- see module docstring), so
    thresholding the image is approximately thresholding mu0. A facet's
    *projected* area (area * mu, for pixel-count purposes) is counted iff
    mu0 clears `mu0_threshold` and the facet is visible (mu > 0).
    """
    raw = {}
    for cam, (mu, mu0) in geometry.items():
        mask = (mu > 0) & (mu0 > mu0_threshold)
        raw[cam] = (np.where(mask, mu, 0.0) * areas).sum(axis=1)
    return _normalize_curves(raw)


def cam_group(cam: str) -> str:
    """
    Groups cameras by azimuthal angle phi (0, 45, 90, 135, 225, 270, 315)
    for independent per-angle parameter fitting. Each angle uses a
    physically distinct camera rig (e.g. the challenge Readme notes a beam
    splitter is used specifically for the 0-degree measurement, since the
    horizontal camera would otherwise block the light source), so there's
    no reason to expect one scattering weight / brightness threshold fits
    all of them equally well.

    0_hor1 and 0_hor2 are further split into their own individual groups:
    our forward model treats them as geometrically identical (phi=0,
    alpha=0 for both), but their real/blender lightcurves visibly differ
    from each other, so a shared 0-degree fit doesn't serve them equally.
    """
    if cam in ("0_hor1", "0_hor2"):
        return cam
    return cam.split("_")[0]


def _group_sort_key(group_name: str) -> tuple[float, str]:
    """Sorts camera groups by azimuthal angle first, then name (for 0_hor1/0_hor2 vs plain "0")."""
    return (float(group_name.split("_")[0]), group_name)


def fit_and_simulate_by_group(
    curve_fn,
    fit_geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_areas: np.ndarray,
    param_name: str,
    grid: np.ndarray,
    target_df: pd.DataFrame,
    cam_keys: list[str],
    sim_phase_axis: np.ndarray,
    sim_geometry: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    sim_areas: np.ndarray | None = None,
    group_fn=cam_group,
    align: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Fits `param_name` separately per camera group (see `cam_group`) instead
    of one global value shared by every camera, then simulates each
    group's cameras with its own fitted parameter and merges the results
    into a single {cam: curve} dict.

    `fit_geometry`/`fit_areas` are the shape used for the grid search
    (i.e. whose predicted curves are scored against `target_df`).
    `sim_geometry`/`sim_areas` are the shape the fitted parameters are then
    applied to for the returned curves; they default to `fit_geometry`/
    `fit_areas` (the normal self-fit case), but can be a *different* shape
    -- e.g. applying parameters fit on asteroid 2 to asteroid 3's mesh, to
    test how well they transfer.
    """
    if sim_geometry is None:
        sim_geometry = fit_geometry
    if sim_areas is None:
        sim_areas = fit_areas

    groups: dict[str, list[str]] = {}
    for cam in cam_keys:
        groups.setdefault(group_fn(cam), []).append(cam)

    curves: dict[str, np.ndarray] = {}
    params: dict[str, float] = {}
    for group_name, group_cams in sorted(groups.items(), key=lambda kv: _group_sort_key(kv[0])):
        best_param, best_mse = fit_param(
            curve_fn, fit_geometry, fit_areas, param_name, grid, target_df, group_cams, sim_phase_axis, align=align
        )
        params[group_name] = best_param
        sim_group_geometry = {cam: sim_geometry[cam] for cam in group_cams}
        curves.update(curve_fn(sim_group_geometry, sim_areas, **{param_name: best_param}))
        label = f"{group_name} deg" if group_name.isdigit() else group_name
        print(
            f"  {label:>9s} ({len(group_cams):2d} cams): best {param_name} = {best_param:.3f}  (mse={best_mse:.4f})"
        )
    return curves, params


def fit_param(
    curve_fn,
    geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    areas: np.ndarray,
    param_name: str,
    grid: np.ndarray,
    target_df: pd.DataFrame,
    cam_keys: list[str],
    sim_phase_axis: np.ndarray,
    align: bool = True,
) -> tuple[float, float]:
    """
    Grid-search `param_name` (e.g. lambert_weight, mu_threshold,
    mu0_threshold) to minimise mean MSE against target_df, which is
    resampled onto `sim_phase_axis` so this works equally for real data
    (842 frames for asteroid 2) or blender data (already n_phases long).

    If align=True (default), each candidate curve is circularly shifted to
    its best cross-correlation lag against the target before scoring --
    see `align_to_target` -- so the fit isn't thrown off by an unknown
    rotational phase offset. That matters most for real data (challenge
    Readme: "the real-world videos do not necessarily start from the same
    position"), but even blender curves can show an apparent offset for
    near-symmetric shapes, so it's applied uniformly.
    """
    targets = {cam: _interp_to_phase(target_df[cam].values.astype(float), sim_phase_axis) for cam in cam_keys}

    best_param, best_mse = float(grid[0]), np.inf
    for p in grid:
        curves = curve_fn(geometry, areas, **{param_name: p})
        mses = []
        for cam in cam_keys:
            sim = curves[cam]
            if not np.all(np.isfinite(sim)):
                mses = None
                break
            target = targets[cam]
            if align:
                sim, _ = align_to_target(sim, target)
            mses.append(np.mean((sim - target) ** 2))
        if mses is None:
            continue
        mse = float(np.mean(mses))
        if mse < best_mse:
            best_mse, best_param = mse, float(p)
    return best_param, best_mse


def _interp_to_phase(values: np.ndarray, sim_phase_axis: np.ndarray) -> np.ndarray:
    """Resample a periodic data column of any length onto the simulation's phase grid."""
    src_phase = np.arange(len(values)) / len(values)
    return np.interp(sim_phase_axis, src_phase, values, period=1.0)


def align_to_target(sim: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Circularly shift the periodic `sim` curve to the integer lag that
    maximises its Pearson correlation with `target`.

    A rotating asteroid's phase-zero convention need not line up between
    our forward model and a given lightcurve: per the challenge Readme,
    "the real-world videos do not necessarily start from the same
    position" as the blender-simulated ones. Cross-correlating removes
    that unknown rotational offset before scoring/plotting so MSE and
    correlation reflect shape agreement, not an arbitrary phase shift.

    Brute-force over all integer shifts -- n_phases is only a few hundred
    samples, so this is cheap (and avoids FFT circular-correlation sign
    conventions being easy to get backwards).
    """
    n = len(sim)
    sim0 = sim - sim.mean()
    target0 = target - target.mean()
    if np.allclose(sim0, 0) or np.allclose(target0, 0):
        return sim, 0
    scores = [np.dot(np.roll(sim0, k), target0) for k in range(n)]
    best_k = int(np.argmax(scores))
    return np.roll(sim, best_k), best_k


def compare_and_plot(
    sim_curves: dict[str, dict[str, np.ndarray]],
    ds_real,
    ds_blender,
    sim_phase_axis: np.ndarray,
    cam_keys: list[str],
    n_phases: int,
    title: str,
    out_path: Path,
    align_to: str | None = "real",
) -> tuple[pd.DataFrame, str]:
    """
    sim_curves: {label: {cam: curve}} -- one or more simulated curve sets
    to plot/score against real + blender data for the given curve_type.

    align_to selects a single circular shift per curve (see
    `align_to_target`), applied uniformly to both the plot and the
    mse/corr-vs-real and mse/corr-vs-blender metrics:
      - "real": shift to best match real (use when the curve's parameters
        were themselves fit against real).
      - "blender": shift to best match blender (use when fit against
        blender).
      - None: no shift -- raw curves at their natural phase-0 alignment,
        so the inherent rotational-phase offset is visible.

    The shift against real and the shift against blender can differ
    noticeably for a curve whose shape doesn't transfer well between the
    two (e.g. a near-binary curve fit to blender's odd 0_hor1/0_hor2
    shape), which is why this picks one rather than showing both.
    """
    suffix = f"aligned to {align_to}" if align_to else "raw"
    full_title = f"{title} [{suffix}]"

    rows = []
    plot_curves: dict[str, dict[str, np.ndarray]] = {cam: {} for cam in cam_keys}
    for cam in cam_keys:
        real_interp = _interp_to_phase(ds_real.data[cam].values.astype(float), sim_phase_axis)
        blender_interp = _interp_to_phase(ds_blender.data[cam].values.astype(float), sim_phase_axis)
        shift_target = {"real": real_interp, "blender": blender_interp}.get(align_to)

        row = {"camera": cam}
        for label, curves in sim_curves.items():
            sim = curves[cam]
            if shift_target is not None:
                sim_shifted, shift = align_to_target(sim, shift_target)
            else:
                sim_shifted, shift = sim, 0

            row[f"shift__{label}"] = shift
            row[f"mse_vs_real__{label}"] = np.mean((sim_shifted - real_interp) ** 2)
            row[f"corr_vs_real__{label}"] = np.corrcoef(sim_shifted, real_interp)[0, 1]
            row[f"mse_vs_blender__{label}"] = np.mean((sim_shifted - blender_interp) ** 2)
            row[f"corr_vs_blender__{label}"] = np.corrcoef(sim_shifted, blender_interp)[0, 1]

            plot_curves[cam][label] = sim_shifted
        rows.append(row)

    df = pd.DataFrame(rows).set_index("camera")
    report_lines = [
        f"=== {full_title}: per-camera comparison ===",
        df.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        "Overall:",
    ]
    report_lines += [f"  {col:35s} {df[col].mean():.4f}" for col in df.columns]
    report = "\n".join(report_lines)
    print("\n" + report)

    colors = ["tomato", "seagreen", "darkorange", "purple"]
    fig, axes = plt.subplots(7, 4, figsize=(16, 22), sharex=True)
    for ax, cam in zip(axes.ravel(), cam_keys):
        real_vals = ds_real.data[cam].values.astype(float)
        real_phase = np.arange(len(real_vals)) / len(real_vals)
        ax.plot(real_phase, real_vals, lw=1.0, color="black", alpha=0.6, label="real")
        ax.plot(
            np.arange(n_phases) / n_phases,
            ds_blender.data[cam].values,
            lw=1.0,
            color="steelblue",
            label="blender",
        )
        for color, label in zip(colors, sim_curves.keys()):
            ax.plot(
                sim_phase_axis, plot_curves[cam][label], lw=1.2, color=color, ls="--", label=f"{label} ({suffix})"
            )
        ax.set_title(cam, fontsize=8)
        ax.set_yticks([])

    axes[0, 0].legend(fontsize=6)
    fig.suptitle(full_title, y=1.001)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison figure to {out_path}")

    return df, report


def main():
    parser = argparse.ArgumentParser(
        description="Forward-simulate lightcurves from an STL shape and compare with challenge_data"
    )
    parser.add_argument(
        "--stl",
        type=str,
        default=str(WORKSPACE_DIR / "stls" / "asteroid2_simplified.stl"),
        help="Path to the STL shape to simulate",
    )
    parser.add_argument(
        "--asteroid",
        type=int,
        default=2,
        help="Asteroid model number, used to look up challenge_data lightcurves and the bounding-cylinder radius",
    )
    parser.add_argument("--n-phases", type=int, default=360)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(WORKSPACE_DIR / "sim_data"),
    )
    parser.add_argument(
        "--fit-stl",
        type=str,
        default=None,
        help="If given (with --fit-asteroid), fit lambert_weight/mu0_threshold against THIS shape/asteroid's "
        "data instead of --stl/--asteroid's own, then apply the fitted parameters to --stl. Use this to test "
        "how well parameters transfer to a shape they weren't optimized for.",
    )
    parser.add_argument(
        "--fit-asteroid",
        type=int,
        default=None,
        help="Asteroid number whose challenge_data to fit parameters against; see --fit-stl.",
    )
    args = parser.parse_args()

    fit_stl = args.fit_stl or args.stl
    fit_asteroid = args.fit_asteroid if args.fit_asteroid is not None else args.asteroid
    using_foreign_params = (fit_stl != args.stl) or (fit_asteroid != args.asteroid)

    if using_foreign_params:
        out_dir = Path(args.out_dir) / f"asteroid{args.asteroid}_params_from_asteroid{fit_asteroid}"
    else:
        out_dir = Path(args.out_dir) / f"asteroid{args.asteroid}"
    shifted_dir = out_dir / "shifted_data"
    non_shifted_dir = out_dir / "non_shifted_data"
    metrics_reports: list[str] = []

    if using_foreign_params:
        disclaimer = (
            f"NOTE: lambert_weight and mu0_threshold below were NOT fit against asteroid {args.asteroid}. "
            f"They were optimized against asteroid {fit_asteroid}'s shape ({fit_stl}) and data, then applied "
            f"as-is to asteroid {args.asteroid}'s shape ({args.stl}) to test how well they transfer. Treat "
            f"this as an out-of-sample evaluation, not a fresh fit for asteroid {args.asteroid}."
        )
        print(f"\n{disclaimer}")
        metrics_reports.append(disclaimer)

    e_sun = np.array([-1.0, 0.0, 0.0])
    cam_keys = list(CAMERA_ANGLE_ALPHA.keys())  # 28 camera angles, txt-file column order

    # --- shape: normalise the STL into the same D(0,R) x [-1,1] challenge
    # cylinder convention that the real challenge shape models use, so that
    # the fixed e_sun / rotation-about-z forward-model assumptions hold. ---
    target_R = MODEL_BASE_RADII[args.asteroid]
    raw_mesh = trimesh.load(args.stl, force="mesh")
    mesh = normalize_mesh_to_challenge_cylinder(raw_mesh, target_R=target_R, verbose=True)
    normals, areas = facet_normals_and_areas(mesh)
    print(f"\nLoaded {args.stl}: {len(areas)} facets")

    phases, geometry = facet_geometry(normals, cam_keys, e_sun, args.n_phases)
    sim_phase_axis = phases / (2 * np.pi)

    # --- challenge_data for comparison (always --asteroid's own data --
    # the point is checking predictions against asteroid {args.asteroid},
    # regardless of which asteroid the parameters were fit on) ---
    ds_real_int = load_data(args.asteroid, "real", "intensity")
    ds_blender_int = load_data(args.asteroid, "simulated", "intensity")
    ds_real_bin = load_data(args.asteroid, "real", "binary")
    ds_blender_bin = load_data(args.asteroid, "simulated", "binary")

    # --- fitting shape/data: same as above unless --fit-stl/--fit-asteroid
    # point somewhere else, in which case parameters are optimized on the
    # *fit* shape/data and only applied to (not re-fit on) --asteroid. ---
    if using_foreign_params:
        fit_target_R = MODEL_BASE_RADII[fit_asteroid]
        fit_raw_mesh = trimesh.load(fit_stl, force="mesh")
        fit_mesh = normalize_mesh_to_challenge_cylinder(fit_raw_mesh, target_R=fit_target_R, verbose=True)
        fit_normals, fit_areas = facet_normals_and_areas(fit_mesh)
        print(f"\nLoaded {fit_stl} (for parameter fitting only): {len(fit_areas)} facets")
        _, fit_geometry = facet_geometry(fit_normals, cam_keys, e_sun, args.n_phases)

        fit_ds_real_int = load_data(fit_asteroid, "real", "intensity")
        fit_ds_blender_int = load_data(fit_asteroid, "simulated", "intensity")
        fit_ds_real_bin = load_data(fit_asteroid, "real", "binary")
        fit_ds_blender_bin = load_data(fit_asteroid, "simulated", "binary")
    else:
        fit_geometry, fit_areas = geometry, areas
        fit_ds_real_int, fit_ds_blender_int = ds_real_int, ds_blender_int
        fit_ds_real_bin, fit_ds_blender_bin = ds_real_bin, ds_blender_bin

    # ================= intensity =================
    # Fit the LSL scattering law's Lambert weight separately per camera
    # angle group (see `cam_group`), once against real data and once
    # against blender data, so the two fits can be compared directly.
    # Also include the parameter-free pure-Lambertian law as a baseline
    # (see `simulate_lightcurves_lambertian`) in both comparisons.
    lambert_grid = np.linspace(0.0, 30.0, 150)
    lambertian_curves = simulate_lightcurves_lambertian(geometry, areas)

    for fit_name, fit_target in [("real", fit_ds_real_int.data), ("blender", fit_ds_blender_int.data)]:
        print(f"\nFitting lambert_weight per camera group (fit to asteroid {fit_asteroid} {fit_name}):")
        lsl_curves, _ = fit_and_simulate_by_group(
            simulate_lightcurves, fit_geometry, fit_areas, "lambert_weight", lambert_grid, fit_target, cam_keys,
            sim_phase_axis, sim_geometry=geometry, sim_areas=areas,
        )
        curves_to_plot = {f"LSL forward sim (fit to {fit_name})": lsl_curves, "Lambertian only": lambertian_curves}
        title = f"Asteroid {args.asteroid} — {Path(args.stl).name} intensity forward sim vs challenge_data (fit to {fit_name}"
        title += f", params from asteroid {fit_asteroid})" if using_foreign_params else ")"
        # Shift to match whichever dataset the parameters were fit against.
        for align_to, plot_dir in [(fit_name, shifted_dir), (None, non_shifted_dir)]:
            _, report = compare_and_plot(
                curves_to_plot,
                ds_real_int,
                ds_blender_int,
                sim_phase_axis,
                cam_keys,
                args.n_phases,
                title,
                plot_dir / f"asteroid{args.asteroid}_intensity_fit_{fit_name}.png",
                align_to=align_to,
            )
            metrics_reports.append(report)

    # ================= binary =================
    # Fit the illumination-threshold model's single free parameter
    # separately per camera angle group, once against real data and once
    # against blender data. (The mu-only silhouette-threshold model was
    # dropped here -- mu0-thresholding consistently outperformed it.)
    mu0_grid = np.linspace(0.02, 0.98, 49)

    for fit_name, fit_target in [("real", fit_ds_real_bin.data), ("blender", fit_ds_blender_bin.data)]:
        print(f"\nFitting mu0_threshold (illumination model) per camera group (fit to asteroid {fit_asteroid} {fit_name}):")
        mu0_curves, _ = fit_and_simulate_by_group(
            simulate_lightcurves_binary_illumination,
            fit_geometry,
            fit_areas,
            "mu0_threshold",
            mu0_grid,
            fit_target,
            cam_keys,
            sim_phase_axis,
            sim_geometry=geometry,
            sim_areas=areas,
        )

        curves_to_plot = {f"mu0-threshold illum. (fit to {fit_name})": mu0_curves}
        title = f"Asteroid {args.asteroid} — {Path(args.stl).name} binary forward sim vs challenge_data (fit to {fit_name}"
        title += f", params from asteroid {fit_asteroid})" if using_foreign_params else ")"
        # Shift to match whichever dataset the parameters were fit against.
        for align_to, plot_dir in [(fit_name, shifted_dir), (None, non_shifted_dir)]:
            _, report = compare_and_plot(
                curves_to_plot,
                ds_real_bin,
                ds_blender_bin,
                sim_phase_axis,
                cam_keys,
                args.n_phases,
                title,
                plot_dir / f"asteroid{args.asteroid}_binary_fit_{fit_name}.png",
                align_to=align_to,
            )
            metrics_reports.append(report)

    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text("\n\n".join(metrics_reports) + "\n")
    print(f"\nSaved metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()
