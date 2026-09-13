"""Score saved reconstruction histories without refitting or rendering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from .trajectory import load_trajectory
from .voxel_metrics import CHALLENGE_RADII, mesh_validity, normalize_mesh, occupied_voxels, overlap_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR = ROOT / "artifacts/prior_l4_all10_200_20260912"
DEFAULT_BASELINE = ROOT / "artifacts/octantoid_l4_all10_200_20260910"
LABELS = {"prior_l4": "Neural prior + L4", "sphere_l4": "Sphere + L4"}
COLORS = {"prior_l4": "tab:blue", "sphere_l4": "tab:orange"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_indices(length, stride, best):
    if length < 1 or stride < 1 or best is None or not 0 <= best < length:
        raise ValueError("invalid trajectory length, stride, or selected index")
    return sorted(set(range(0, length, stride)) | {0, best, length - 1})


def verify_stl(path, vertices, faces):
    mesh = trimesh.load_mesh(path, process=False)
    expected = vertices[faces]
    if mesh.triangles.shape != expected.shape or not np.allclose(
            mesh.triangles, expected, atol=1e-6, rtol=1e-6):
        raise ValueError(f"STL disagrees with saved trajectory: {path}")


def cached_json(path, identity):
    try:
        value = json.loads(Path(path).read_text())
        return value if value.get("identity") == identity else None
    except (OSError, ValueError):
        return None


def reference_voxels(mesh, pitch, folder, identity):
    path = folder / f"reference_{pitch:g}.npz"
    try:
        with np.load(path, allow_pickle=False) as saved:
            coordinates = saved["coordinates"].copy()
            if str(saved["identity"]) == identity:
                digest = hashlib.sha256(coordinates.tobytes()).hexdigest()
                if digest == str(saved["coordinates_sha256"]):
                    return coordinates
    except (OSError, ValueError, KeyError):
        pass
    coordinates = occupied_voxels(mesh, pitch)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, coordinates=coordinates, identity=np.asarray(identity),
                        coordinates_sha256=np.asarray(hashlib.sha256(coordinates.tobytes()).hexdigest()))
    temporary.replace(path)
    return coordinates


def score_histories(args):
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = {name: sha256(Path(__file__).with_name(name))
                   for name in ("voxel_metrics.py", "voxel_history.py", "trajectory.py")}
    rows, runs, references = [], [], []
    pitches = sorted(set(args.pitches))
    normalization = getattr(args, "normalization", "uniform")
    if normalization not in ("uniform", "cylinder"):
        raise ValueError("normalization must be uniform or cylinder")
    for asteroid in args.asteroids:
        target_radius = CHALLENGE_RADII[asteroid] if normalization == "cylinder" else None
        protocol = dict(mode=normalization, target_radius=target_radius)
        folder = output / f"asteroid{asteroid:02d}"
        folder.mkdir(exist_ok=True)
        reference_path = args.data_root / f"AsteroidModel{asteroid:02d}_shape_public/asteroid{asteroid}.stl"
        reference_hash = sha256(reference_path)
        print(f"Asteroid {asteroid}: preparing original reference", flush=True)
        raw = trimesh.load_mesh(reference_path, process=True)
        reference, transform = normalize_mesh(raw, target_radius=target_radius)
        del raw
        audit = mesh_validity(reference)
        reference_info = dict(asteroid=asteroid, path=str(reference_path.resolve()),
                              sha256=reference_hash, normalization=transform, validity=audit)
        references.append(reference_info)
        write_json(folder / "reference.json", reference_info)
        if not audit["valid"]:
            raise ValueError(f"invalid reference for asteroid {asteroid}: {audit['reasons']}")
        grids = {}
        for pitch in pitches:
            identity = fingerprint(dict(reference=reference_hash, pitch=pitch, code=code_hashes, normalization=protocol))
            grids[pitch] = reference_voxels(reference, pitch, folder, identity)
            print(f"  reference pitch={pitch:g}: {len(grids[pitch]):,} occupied voxels", flush=True)
        del reference
        for label, campaign, variant in (
                ("prior_l4", args.prior_campaign, "prior_l4"),
                ("sphere_l4", args.baseline_campaign, "octantoid_l4")):
            run_path = campaign / f"asteroid{asteroid:02d}" / variant
            trajectory = load_trajectory(run_path / "trajectory.npz")
            metadata = json.loads((run_path / "run.json").read_text())
            best = trajectory.best_index
            if metadata["summary"]["best_iteration"] != best:
                raise ValueError("selected index differs between trajectory and run metadata")
            if metadata["summary"]["iterations_run"] != len(trajectory.vertices) - 1:
                raise ValueError("trajectory does not cover recorded iterations")
            inputs = {name: sha256(run_path / name) for name in
                      ("trajectory.npz", "run.json", "mesh_initial.stl", "mesh_final.stl")}
            for filename, index in (("mesh_initial.stl", 0), ("mesh_final.stl", best)):
                verify_stl(run_path / filename, trajectory.vertices[index], trajectory.faces)
            indices = checkpoint_indices(len(trajectory.vertices), args.stride, best)
            run_info = dict(asteroid=asteroid, method=label, path=str(run_path.resolve()),
                            inputs=inputs, selected_iteration=best, terminal_iteration=len(trajectory.vertices)-1,
                            iterations=indices, fit_status=metadata["summary"]["status"])
            runs.append(run_info)
            body = folder / label
            (body / "checkpoints").mkdir(parents=True, exist_ok=True)
            for step in indices:
                identity = fingerprint(dict(inputs=inputs, reference=reference_hash, code=code_hashes,
                                            step=step, pitches=pitches, normalization=protocol))
                record_path = body / f"iteration_{step:04d}.json"
                record = cached_json(record_path, identity)
                stl_path = body / "checkpoints" / f"iteration_{step:04d}.stl"
                if record and stl_path.exists() and sha256(stl_path) == record["checkpoint_sha256"]:
                    rows.extend(record["measurements"])
                    continue
                mesh = trimesh.Trimesh(vertices=trajectory.vertices[step], faces=trajectory.faces, process=False)
                mesh.export(stl_path)
                verify_stl(stl_path, trajectory.vertices[step], trajectory.faces)
                normalized, transform = normalize_mesh(mesh, target_radius=target_radius)
                audit = mesh_validity(normalized, check_intersections=True)
                measurements = []
                for pitch in pitches:
                    scores = (overlap_metrics(occupied_voxels(normalized, pitch), grids[pitch])
                              if audit["valid"] else dict(solid_dice=None, solid_iou=None))
                    row = dict(asteroid=asteroid, method=label, iteration=step, pitch=pitch,
                               normalization_mode=normalization, target_radius=target_radius,
                               selected=step == best, terminal=step == len(trajectory.vertices)-1,
                               valid=audit["valid"], invalid_reason="; ".join(audit["reasons"]),
                               checkpoint=str(stl_path.relative_to(output)), normalization=transform, **scores)
                    measurements.append(row)
                write_json(record_path, dict(identity=identity, validity=audit, measurements=measurements,
                                             checkpoint_sha256=sha256(stl_path)))
                rows.extend(measurements)
                write_json(output / "status.json", dict(state="running", asteroid=asteroid, method=label,
                           iteration=step, measurements=len(rows), elapsed_seconds=time.monotonic()-started))
                print(f"  {label} step {step:3d}: Dice={measurements[0]['solid_dice']}", flush=True)
        del grids
    manifest = dict(code=code_hashes, references=references, runs=runs, pitches=pitches, stride=args.stride,
                    normalization_mode=normalization,
                    cylinder_radii=CHALLENGE_RADII if normalization == "cylinder" else None,
                    normalization_source="origin/inr src/hac/mesh.py::normalize_mesh_to_challenge_cylinder (center_xy=bounds)",
                    metric_source="origin/inr faac378 src/hac/lc2mesh/metrics.py; world-coordinate filled voxels",
                    campaign_reporting_note="Prior fits 1–3 completed, but campaign reporting failed lightcurve tolerance checks. "
                    "This evaluation independently checks trajectory/STL correspondence and mesh geometry; it does not repair those reports.")
    write_json(output / "manifest.json", manifest)
    write_json(output / "measurements.json", rows)
    make_report(output, rows, manifest)
    if getattr(args, "compare_with", None) is not None:
        compare_normalizations(output, rows, manifest, args.compare_with)
    invalid = sum(not row["valid"] for row in rows)
    write_json(output / "status.json", dict(state="complete" if not invalid else "complete_with_invalid_meshes",
               measurements=len(rows), invalid_measurements=invalid, elapsed_seconds=time.monotonic()-started))
    print(f"Finished: {len(rows)} measurements, {invalid} invalid; {output / 'report.md'}", flush=True)
    return rows


def draw_axes(axes, rows, asteroid, pitches):
    for ax, key, title in zip(axes, ("solid_dice", "solid_iou"), ("Solid Dice", "Solid IoU")):
        for method in LABELS:
            for index, pitch in enumerate(pitches):
                group = sorted((r for r in rows if r["asteroid"] == asteroid and r["method"] == method
                                and r["pitch"] == pitch), key=lambda r: r["iteration"])
                x = [r["iteration"] for r in group]
                y = [r[key] if r[key] is not None else np.nan for r in group]
                ax.plot(x, y, color=COLORS[method], linestyle="-" if index == 0 else "--",
                        alpha=1 if index == 0 else .65, label=f"{LABELS[method]}, pitch {pitch:g}")
                for row in group:
                    if row["selected"] and row[key] is not None and index == 0:
                        ax.scatter([row["iteration"]], [row[key]], marker="*", s=130,
                                   color=COLORS[method], edgecolors="black", zorder=5, clip_on=False)
        ax.set(title=f"Asteroid {asteroid}: {title}", xlabel="Optimization iteration", ylabel="Overlap score",
               ylim=(0, 1.02))
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc="best")


def make_report(output, rows, manifest):
    fields = [k for k in rows[0] if k != "normalization"]
    # Include count columns even if an early checkpoint is invalid.
    fields = list(dict.fromkeys(fields + [k for r in rows for k in r if k != "normalization"]))
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    pitches = manifest["pitches"]
    asteroids = sorted(set(r["asteroid"] for r in rows))
    figure, axes = plt.subplots(len(asteroids), 2, figsize=(13, 4 * len(asteroids)), squeeze=False)
    for index, asteroid in enumerate(asteroids):
        draw_axes(axes[index], rows, asteroid, pitches)
        body_figure, body_axes = plt.subplots(1, 2, figsize=(13, 4))
        draw_axes(body_axes, rows, asteroid, pitches)
        body_figure.suptitle("Stars mark lightcurve-selected checkpoints")
        body_figure.tight_layout()
        for extension in ("png", "pdf"):
            body_figure.savefig(output / f"asteroid{asteroid:02d}/accuracy.{extension}", dpi=160)
        plt.close(body_figure)
    mode = manifest.get("normalization_mode", "uniform")
    figure.suptitle(f"Voxel accuracy histories · {mode} normalization · stars mark lightcurve-selected checkpoints")
    figure.tight_layout(rect=(0, 0, 1, .975))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"accuracy_comparison.{extension}", dpi=160)
    plt.close(figure)
    summary, sensitivity = [], []
    for run in manifest["runs"]:
        group = [r for r in rows if r["asteroid"] == run["asteroid"] and r["method"] == run["method"]]
        for state, step in (("initial", 0), ("selected", run["selected_iteration"]), ("terminal", run["terminal_iteration"])):
            for pitch in pitches:
                row = next(r for r in group if r["iteration"] == step and r["pitch"] == pitch)
                summary.append(dict(asteroid=run["asteroid"], method=run["method"], state=state, iteration=step,
                                    pitch=pitch, solid_dice=row["solid_dice"], solid_iou=row["solid_iou"]))
        for pitch in pitches[1:]:
            for key in ("solid_dice", "solid_iou"):
                differences = []
                for step in run["iterations"]:
                    primary = next(r for r in group if r["iteration"] == step and r["pitch"] == pitches[0])
                    other = next(r for r in group if r["iteration"] == step and r["pitch"] == pitch)
                    if primary[key] is not None and other[key] is not None:
                        differences.append(abs(primary[key] - other[key]))
                sensitivity.append(dict(asteroid=run["asteroid"], method=run["method"], metric=key,
                                        comparison_pitch=pitch, max_absolute_change=max(differences) if differences else None))
    write_json(output / "summary.json", dict(states=summary, resolution_sensitivity=sensitivity))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    normalization_note = (
        "Each checkpoint and original reference is independently centered by its bounding-box midpoint. "
        "Horizontal coordinates are scaled to the asteroid's bounding-cylinder radius "
        "(asteroid 1: R=1.12; asteroid 2: R=1.42; asteroid 3: R=0.88), and z is independently scaled to [-1,1]. "
        "This follows origin/inr's challenge-cylinder normalization. It removes width-to-height mismatch as well as "
        "absolute size and position differences. It does not fit a rotation. Reference geometry never selects a checkpoint."
        if mode == "cylinder" else
        "Each checkpoint and original reference is independently centered by its bounding-box midpoint and uniformly "
        "scaled to z bounds [-1,1]. No rotation fitting or anisotropic scaling is performed. Scores therefore measure "
        "normalized shape overlap, excluding absolute size and position accuracy. Reference geometry never selects a checkpoint."
    )
    lines = ["# Voxel reconstruction accuracy: asteroids 1–3", "",
             "Solid Dice = 2|A∩B|/(|A|+|B|); solid IoU = |A∩B|/|A∪B|. Higher is better. "
             "A and B contain occupied surface and interior voxels on a shared world-coordinate lattice.", "",
             normalization_note, "",
             f"The primary voxel pitch is {pitches[0]:g}. Dashed curves check coarser resolutions. "
             "The horizontal axis is optimization iteration; per-step elapsed times were not recorded. "
             "Stars identify the lightcurve-selected state, which may differ from the terminal state or best geometric overlap.", "",
             "Original reference meshes are used without decimation or star-shaped proxies. Their watertightness, winding, "
             "face areas and signed volumes are checked; their self-intersections are not exhaustively checked. "
             "Each scored reconstruction also undergoes a transverse intersection check, which excludes coplanar contacts. "
             "Invalid meshes have null scores. Exported checkpoint STLs retain original solver coordinates; exact normalization "
             "transforms are saved in the measurement JSON.", "",
             manifest["campaign_reporting_note"], "",
             "Neural-network training/input provenance is unknown. Differences between methods are descriptive; "
             "these historical runs use different initialization and optimizer settings.", "",
             "![Accuracy histories](accuracy_comparison.png)", "",
             "| Asteroid | Method | State | Iteration | Dice | IoU |",
             "|---|---|---|---:|---:|---:|"]
    for row in summary:
        if row["pitch"] == pitches[0]:
            dice = "invalid" if row["solid_dice"] is None else f"{row['solid_dice']:.5f}"
            iou = "invalid" if row["solid_iou"] is None else f"{row['solid_iou']:.5f}"
            lines.append(f"| {row['asteroid']} | {LABELS[row['method']]} | {row['state']} | {row['iteration']} | {dice} | {iou} |")
    lines.extend(["", "## Resolution sensitivity", "", "Maximum absolute score changes from the primary pitch over sampled checkpoints:", "",
                  "| Asteroid | Method | Metric | Comparison pitch | Maximum change |", "|---|---|---|---:|---:|"])
    for row in sensitivity:
        difference = "n/a" if row["max_absolute_change"] is None else f"{row['max_absolute_change']:.5f}"
        lines.append(f"| {row['asteroid']} | {LABELS[row['method']]} | {row['metric']} | {row['comparison_pitch']:g} | {difference} |")
    lines.extend(["", "[Measurements CSV](measurements.csv) · [Measurements and transforms](measurements.json) · "
                  "[Summary CSV](summary.csv) · [Provenance](manifest.json) · [PDF figure](accuracy_comparison.pdf)", ""])
    (output / "report.md").write_text("\n".join(lines))


def compare_normalizations(output, rows, manifest, previous):
    """Compare matched saved states; never modify the earlier evaluation."""
    previous = Path(previous).resolve()
    if previous == output.resolve():
        raise ValueError("comparison directory must differ from output directory")
    old = json.loads((previous / "measurements.json").read_text())
    old_manifest = json.loads((previous / "manifest.json").read_text())
    def run_identity(value):
        return {(r["asteroid"], r["method"]): r["inputs"] for r in value["runs"]}
    if run_identity(old_manifest) != run_identity(manifest):
        raise ValueError("normalization comparison requires identical source fits")
    if {r["asteroid"]: r["sha256"] for r in old_manifest["references"]} != {
            r["asteroid"]: r["sha256"] for r in manifest["references"]}:
        raise ValueError("normalization comparison requires identical reference meshes")
    def key(row):
        return row["asteroid"], row["method"], row["iteration"], row["pitch"]
    old_by_key = {key(row): row for row in old}
    if set(old_by_key) != {key(row) for row in rows}:
        raise ValueError("normalization comparison requires matching checkpoints and pitches")
    comparison = []
    for row in rows:
        before = old_by_key[key(row)]
        values = dict(asteroid=row["asteroid"], method=row["method"], iteration=row["iteration"], pitch=row["pitch"],
                      selected=row["selected"], previous_mode=old_manifest.get("normalization_mode", "uniform"),
                      current_mode=manifest["normalization_mode"])
        for metric in ("solid_dice", "solid_iou"):
            values[f"previous_{metric}"] = before[metric]
            values[f"current_{metric}"] = row[metric]
            values[f"delta_{metric}"] = (row[metric] - before[metric]
                                          if row[metric] is not None and before[metric] is not None else None)
        comparison.append(values)
    write_json(output / "normalization_comparison.json", dict(previous_directory=str(previous),
               previous_measurements_sha256=sha256(previous / "measurements.json"), measurements=comparison))
    with (output / "normalization_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    asteroids = sorted({r["asteroid"] for r in rows})
    pitch = min(manifest["pitches"])
    figure, axes = plt.subplots(len(asteroids), 2, figsize=(13, 4*len(asteroids)), squeeze=False)
    for index, asteroid in enumerate(asteroids):
        for axis, metric in zip(axes[index], ("solid_dice", "solid_iou")):
            for method in LABELS:
                group = sorted((r for r in comparison if r["asteroid"] == asteroid and r["method"] == method
                                and r["pitch"] == pitch), key=lambda r: r["iteration"])
                for prefix, style in (("previous", "--"), ("current", "-")):
                    axis.plot([r["iteration"] for r in group],
                              [r[f"{prefix}_{metric}"] for r in group], color=COLORS[method], linestyle=style,
                              label=f"{LABELS[method]}: {group[0][prefix+'_mode']}")
            axis.set(title=f"Asteroid {asteroid}: {metric.replace('_', ' ')}", xlabel="Optimization iteration",
                     ylabel="Overlap score", ylim=(0, 1.02))
            axis.grid(alpha=.25)
            axis.legend(fontsize=7)
    figure.suptitle(f"Effect of normalization · voxel pitch {pitch:g} · same reconstructed meshes")
    figure.tight_layout(rect=(0, 0, 1, .975))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"normalization_comparison.{extension}", dpi=160)
    plt.close(figure)
    with (output / "report.md").open("a") as stream:
        stream.write("\n## Comparison with the previous normalization\n\n"
                     "The reconstructed meshes and voxel pitches are identical in both evaluations. Only the normalization changes. "
                     "Cylinder normalization removes differences in the maximum horizontal radius relative to height, so these "
                     "scores answer a different geometric question from the earlier uniform-scale scores.\n\n"
                     "![Normalization comparison](normalization_comparison.png)\n\n"
                     "[Comparison CSV](normalization_comparison.csv) · [Comparison provenance](normalization_comparison.json) · "
                     "[Comparison PDF](normalization_comparison.pdf)\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-campaign", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--baseline-campaign", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--data-root", type=Path, default=ROOT / "challenge_data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asteroids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--pitches", type=float, nargs="+", default=[0.02, 0.04])
    parser.add_argument("--normalization", choices=("uniform", "cylinder"), default="uniform")
    parser.add_argument("--compare-with", type=Path, help="existing evaluation directory to compare, without modifying it")
    args = parser.parse_args()
    if args.stride < 1 or not set(args.asteroids) <= {1, 2, 3} or any(not np.isfinite(p) or p <= 0 for p in args.pitches):
        parser.error("use asteroids 1–3, positive stride, and finite positive pitches")
    args.asteroids = sorted(set(args.asteroids))
    try:
        score_histories(args)
    except Exception as error:
        write_json(args.output_dir / "status.json", dict(state="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
