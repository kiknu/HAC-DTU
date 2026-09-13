"""Export completed fit states into the per-asteroid challenge cylinders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from .trajectory import load_trajectory
from .voxel_history import DEFAULT_BASELINE, DEFAULT_PRIOR, sha256, write_json
from .voxel_metrics import CHALLENGE_RADII, mesh_validity, normalize_mesh


def export_mesh(mesh, output, asteroid):
    """Write a normalized copy and audit the actual float32 STL on disk."""
    normalized, transform = normalize_mesh(mesh, target_radius=CHALLENGE_RADII[asteroid])
    audit = mesh_validity(normalized, check_intersections=len(mesh.faces) <= 4096)
    if not audit["valid"]:
        raise ValueError(f"invalid source mesh: {audit['reasons']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.export(output)
    loaded = trimesh.load_mesh(output, process=True)
    radius = float(np.linalg.norm(loaded.vertices[:, :2], axis=1).max())
    np.testing.assert_allclose(loaded.bounds[:, 2], [-1, 1], atol=1e-6, rtol=0)
    np.testing.assert_allclose(radius, CHALLENGE_RADII[asteroid], atol=1e-6, rtol=0)
    assert len(loaded.faces) == len(mesh.faces)
    written_audit = mesh_validity(loaded, check_intersections=len(mesh.faces) <= 4096)
    if not written_audit["valid"]:
        raise ValueError(f"invalid STL after float32 export: {written_audit['reasons']}")
    return dict(transform=transform, faces=len(loaded.faces), actual_xy_radius=radius,
                z_bounds=loaded.bounds[:, 2].tolist(), validity=written_audit, sha256=sha256(output))


def export_campaigns(output, campaigns, asteroids):
    output = Path(output)
    exports, skipped = [], []
    for label, campaign, variant in campaigns:
        campaign = Path(campaign)
        for asteroid in asteroids:
            source = campaign / f"asteroid{asteroid:02d}" / variant
            run_path, trajectory_path = source / "run.json", source / "trajectory.npz"
            if not run_path.exists() or not trajectory_path.exists():
                skipped.append(dict(asteroid=asteroid, method=label, reason="No completed run/trajectory available"))
                continue
            summary = json.loads(run_path.read_text())["summary"]
            if summary["status"] != "complete":
                skipped.append(dict(asteroid=asteroid, method=label, reason="Fit incomplete"))
                continue
            trajectory = load_trajectory(trajectory_path)
            if len(trajectory.vertices) != summary["iterations_run"] + 1:
                raise ValueError(f"trajectory length differs from completed run: {source}")
            if trajectory.best_index != summary["best_iteration"]:
                raise ValueError(f"selected iteration differs from completed run: {source}")
            inputs = dict(trajectory_sha256=sha256(trajectory_path), run_sha256=sha256(run_path))
            for state, step in (("initial", 0), ("selected", trajectory.best_index),
                                ("terminal", len(trajectory.vertices)-1)):
                relative = Path(f"asteroid{asteroid:02d}") / label / f"{state}.stl"
                mesh = trimesh.Trimesh(vertices=trajectory.vertices[step], faces=trajectory.faces, process=False)
                information = export_mesh(mesh, output / relative, asteroid)
                exports.append(dict(asteroid=asteroid, method=label, state=state, iteration=step,
                                    file=relative.as_posix(), radius=CHALLENGE_RADII[asteroid],
                                    source=f"{campaign.name}/asteroid{asteroid:02d}/{variant}",
                                    **inputs, **information))
                print(f"Exported {relative}: iteration {step}, R={CHALLENGE_RADII[asteroid]}", flush=True)
    write_json(output / "manifest.json", dict(normalization="bounds-centered; xy radius=R; z=[-1,1]; no rotation",
               radius_source="origin/inr:src/hac/constants.py MODEL_BASE_RADII",
               radii=CHALLENGE_RADII, exports=exports, skipped=skipped))
    lines = ["# Cylinder-normalized asteroid reconstructions", "",
             "These are resized copies of completed 200-iteration L4 fits. Each mesh is centered by its bounding box, "
             "scaled horizontally to the asteroid's specified cylinder radius, and independently scaled vertically "
             "to z=[-1,1]. No rotation is fitted. This removes width-to-height mismatch as well as absolute scale. "
             "Original optimization artifacts are preserved.", "",
             "`initial.stl` is the prepared starting mesh; `selected.stl` minimizes the saved lightcurve validation error; "
             "`terminal.stl` is the last iteration. Selection can return iteration zero. These files are for display/evaluation; "
             "they are not replacements for the solver's saved coordinate system or restart checkpoints.", "",
             "| Asteroid | Radius | Method | Initial | Selected (iteration) | Terminal |",
             "|---|---:|---|---|---|---|"]
    for asteroid in asteroids:
        for label, _, _ in campaigns:
            group = {r["state"]: r for r in exports if r["asteroid"] == asteroid and r["method"] == label}
            if group:
                links = [f"[{s}{' ('+str(group[s]['iteration'])+')' if s == 'selected' else ''}]({group[s]['file']})"
                         for s in ("initial", "selected", "terminal")]
                lines.append(f"| {asteroid} | {CHALLENGE_RADII[asteroid]} | {label} | {' | '.join(links)} |")
    lines.extend(["", "## Availability and checks", "",
                  f"Exported {len(exports)} STLs. Each serialized STL was reloaded and checked for radius, height, face count, "
                  "watertightness, winding, degenerate faces, and positive volume. Meshes with at most 4,096 faces also "
                  "underwent the transverse intersection check (coplanar contacts excluded).", "",
                  "Completed prior fits for asteroids 1–3 and 8 had separate lightcurve-report tolerance failures. "
                  "Their complete saved trajectories supply these exports; this bundle does not repair the photometric reports.", "",
                  "[Manifest, transforms and SHA-256 hashes](manifest.json)", ""])
    for row in skipped:
        lines.append(f"- Asteroid {row['asteroid']}, {row['method']}: {row['reason']}.")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return exports, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-campaign", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--baseline-campaign", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--asteroids", type=int, nargs="+", default=list(CHALLENGE_RADII))
    args = parser.parse_args()
    if not set(args.asteroids) <= set(CHALLENGE_RADII):
        parser.error("asteroids must be 1–10")
    export_campaigns(args.output_dir, [("sphere_l4", args.baseline_campaign, "octantoid_l4"),
                                     ("prior_l4", args.prior_campaign, "prior_l4")], sorted(set(args.asteroids)))


if __name__ == "__main__":
    main()
