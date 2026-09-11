"""Report examples for one L4 fit per asteroid, including hidden-shape bodies."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from forward_validation import load_benchmark_mesh
from visualize_fit import _add_mesh, _set_axes, plot_shape_grid
from .animation import animate_trajectory, animation_indices
from .inversion import FitConfig
from .provenance import file_sha256, fingerprint
from .representation_reporting import audit_run, measure_run, plot_curves, write_json
from .trajectory import load_trajectory

VARIANT = "octantoid_l4"


def calibration_label(asteroid):
    return "Provisional: asteroid 3 calibration" if asteroid > 3 else "Per-body public calibration"


def body_artifacts(config):
    return ["lightcurves_selected.png", f"lightcurves_step{config.steps}.png", "shapes.png",
            "convergence_deformation.png", "metrics.json", "report.md",
            *[f"{VARIANT}/{name}" for name in ("full_history.gif", "mesh_initial.stl", "mesh_final.stl",
              "mesh_last.stl", "trajectory.npz", "curves.npz", "curves_last.npz",
              "comparison_metrics.json", "audit.json", "run.json")]]


def verify_body(body, config):
    marker = json.loads((body / "report_complete.json").read_text())
    if marker["config_fingerprint"] != fingerprint(asdict(config)):
        raise RuntimeError("report configuration changed")
    if set(marker["artifacts"]) != set(body_artifacts(config)):
        raise RuntimeError("incomplete report manifest")
    decoded = {}
    for name, digest in marker["artifacts"].items():
        path = body / name
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"missing or changed report artifact: {path}")
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as picture:
                for i in range(getattr(picture, "n_frames", 1)):
                    picture.seek(i)
                    picture.load()
                if path.suffix == ".gif":
                    decoded[name] = picture.n_frames
    trajectory = load_trajectory(body / VARIANT / "trajectory.npz")
    expected_frames = len(animation_indices(config.steps + 1, 5, trajectory.best_index))
    if decoded[f"{VARIANT}/full_history.gif"] != expected_frames:
        raise RuntimeError("GIF does not cover the expected deformation history")
    for link in re.findall(r"\]\(([^)]+)\)", (body / "report.md").read_text()):
        if not (body / link).is_file():
            raise RuntimeError(f"missing per-asteroid report link: {link}")
    return decoded


def report_asteroid(root, config, calibration):
    root = Path(root)
    body = root / f"asteroid{config.asteroid:02d}"
    run = body / VARIANT
    if not (run / "run.json").exists():
        return
    marker_path = body / "report_complete.json"
    if marker_path.exists():
        try:
            verify_body(body, config)
            audit_run(run, config)
            return
        except (RuntimeError, ValueError, KeyError, OSError):
            marker_path.unlink()

    metrics = measure_run(run, config, calibration)
    trajectory = load_trajectory(run / "trajectory.npz")
    best = trajectory.best_index
    note = calibration_label(config.asteroid)
    selected_label = f"L4 selected, iteration {best} · {note}"
    plot_curves([run], [selected_label], body / "lightcurves_selected.png")
    plot_curves([run], [f"L4 iteration {config.steps} · {note}"],
                body / f"lightcurves_step{config.steps}.png", last=True, last_step=config.steps)
    meshes = [(trajectory.vertices[i], trajectory.faces) for i in (0, best, -1)]
    labels = ["Initial sphere", f"Selected: iteration {best}", f"Terminal: iteration {config.steps}"]
    colors = [(0.6, 0.6, 0.6), (0.25, 0.5, 0.8), (0.9, 0.55, 0.2)]
    if config.asteroid <= 3:
        # High-resolution references are used by CPU geometry scoring only.
        proxy = load_benchmark_mesh(config.asteroid, 1280, torch.device("cpu"), body / "mesh_cache")
        meshes = [(vertices * (metrics["geometry"][state]["scale"] or 1.), faces)
                  for (vertices, faces), state in zip(meshes, ("initial", "selected", "last"))]
        meshes.append((proxy.vertices.numpy(), proxy.faces.numpy()))
        labels.append("Public display proxy (1280 faces)")
        colors.append((0.6, 0.65, 0.6))
    plot_shape_grid(meshes, labels, colors, body / "shapes.png",
                    suptitle=f"Asteroid {config.asteroid} · Octantoid L4 · {note}")

    metadata = json.loads((run / "run.json").read_text())
    history = metadata["validation_history"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot([r["step"] for r in history], [r["rmse"] for r in history])
    axes[0].scatter([best], [metadata["summary"]["best_validation_rmse"]], marker="*", s=90, label="Selected")
    axes[0].legend()
    with np.load(run / "deformation.npz") as values:
        for axis, name in zip(axes[1:], ("radial_rms", "tangential_rms")):
            axis.plot(values["iteration"], values[name])
    for axis, title, ylabel in zip(axes, ("Validation error", "Radial deformation", "Tangential deformation"),
                                   ("RMSE", "RMS / initial radius", "RMS / initial radius")):
        axis.set(xlabel="iteration", ylabel=ylabel, title=title)
        axis.grid(alpha=.2)
    fig.suptitle(f"Asteroid {config.asteroid} · {note}")
    fig.tight_layout()
    fig.savefig(body / "convergence_deformation.png", dpi=180)
    plt.close(fig)
    temporary = run / "full_history.tmp.gif"
    animate_trajectory(run / "trajectory.npz", temporary, all_iterations=True, stride=5,
                       label=f"Asteroid {config.asteroid} · L4\n{note}")
    temporary.replace(run / "full_history.gif")

    summary = dict(asteroid=config.asteroid, calibration_status=note,
                   calibration_source_asteroid=min(config.asteroid, 3),
                   calibration_transfer=config.calibration_transfer, reference_available=config.asteroid <= 3,
                   iterations=metadata["summary"]["iterations_run"], selected_iteration=best,
                   initial_rmse=metrics["photometry"]["initial"]["full_turn"],
                   selected_rmse=metrics["photometry"]["selected"]["full_turn"],
                   terminal_rmse=metrics["photometry"]["last"]["full_turn"],
                   selected_validation_rmse=metrics["photometry"]["selected"]["validation"],
                   terminal_validation_rmse=metrics["photometry"]["last"]["validation"],
                   selected_remaining_rmse=metrics["photometry"]["selected"]["remaining"],
                   selected_radial_log_rms=(metrics["geometry"]["selected"] or {}).get("radial_log_rms"),
                   selected_mesh_valid=metrics["audit"]["mesh_validity"]["selected"]["valid"],
                   terminal_mesh_valid=metrics["audit"]["mesh_validity"]["last"]["valid"],
                   runtime_seconds=metrics["total_fit_wall_seconds"],
                   peak_memory_mib=metadata["summary"]["peak_memory_mib"])
    write_json(body / "metrics.json", summary)
    lines = [f"# Asteroid {config.asteroid}: octantoid L4", "", note + ".", "",
             f"Identical 1,024-face sphere start; {config.steps} updates. Best validation state: iteration {best}.", "",
             "![Lightcurves at selected state](lightcurves_selected.png)", "",
             f"[Lightcurves at iteration {config.steps}](lightcurves_step{config.steps}.png)", "",
             "![Initial, selected and terminal shapes](shapes.png)", "",
             "![Convergence and deformation](convergence_deformation.png)", "",
             f"![Deformation through iteration {config.steps}]({VARIANT}/full_history.gif)", "",
             f"[Selected mesh]({VARIANT}/mesh_final.stl) · [Terminal mesh]({VARIANT}/mesh_last.stl) · "
             f"[All saved states]({VARIANT}/trajectory.npz) · [Metrics](metrics.json)", "",
             "GIFs show five-update intervals, preserving the initial, selected and terminal states. "
             "The trajectory retains every update. Selected outputs minimize validation RMSE.", "",
             ("The true shape is hidden; these are inferred examples under transferred rendering settings. "
              "Reference geometry scores are unavailable." if config.asteroid > 3 else
              "Reference geometry is used only for evaluation and scale-aligned display, not inversion or state selection. "
              "Validation is conditional on calibration using public full-turn data. Display proxies are labelled; "
              "geometry scores use the raw reference surface."), ""]
    (body / "report.md").write_text("\n".join(lines))
    write_json(marker_path, dict(asteroid=config.asteroid, steps=config.steps,
               config_fingerprint=fingerprint(asdict(config)),
               artifacts={name: file_sha256(body / name) for name in body_artifacts(config)}))
    try:
        verify_body(body, config)
    except Exception:
        marker_path.unlink()
        raise


def report_campaign(root):
    root = Path(root)
    description = json.loads((root / "study.json").read_text())
    asteroids = description["asteroids"]
    rows, trajectories = [], {}
    for asteroid in asteroids:
        body = root / f"asteroid{asteroid:02d}"
        if (body / "report_complete.json").exists():
            rows.append(json.loads((body / "metrics.json").read_text()))
            trajectories[asteroid] = load_trajectory(body / VARIANT / "trajectory.npz")
    if rows:
        with (root / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        fig = plt.figure(figsize=(15, 6 * ((len(asteroids) + 4) // 5)))
        for index, asteroid in enumerate(asteroids):
            axis = fig.add_subplot((len(asteroids) + 4) // 5, 5, index + 1, projection="3d")
            title = f"Asteroid {asteroid}"
            if asteroid in trajectories:
                trajectory = trajectories[asteroid]
                _add_mesh(axis, trajectory.vertices[trajectory.best_index], trajectory.faces,
                          base_rgb=(.25, .5, .8))
                limit = 1.08 * float(np.abs(trajectory.vertices[trajectory.best_index]).max())
                title += f" · step {trajectory.best_index}"
            else:
                limit = .54
                title += " · pending"
            if asteroid > 3:
                title += "\nProvisional calibration"
            _set_axes(axis, limit, 22, 135, title)
        fig.suptitle("Octantoid L4 · best validation states · independently scaled views")
        fig.tight_layout()
        fig.savefig(root / "shapes_overview.png", dpi=200)
        fig.savefig(root / "shapes_overview.pdf")
        plt.close(fig)
    steps = description["steps"]
    escape, stabilization = description.get("escape_steps", 40), description.get("stabilization_steps", 40)
    lines = ["# Local octantoid L4 survey", "",
             f"Verified reports: **{len(rows)}/{len(asteroids)}**. Each fit starts from the same 1,024-face sphere and runs {steps} updates.", "",
             "Asteroids 1–3 use their individual public calibrations. Asteroids 4–10 provisionally reuse asteroid 3's "
             "calibration; their true shapes are hidden. These are fixed-recipe examples, not established unique shape reconstructions.", "",
             f"The 74 coefficients use the existing spectral penalty λ Σ l³c² / 146. The schedule is {escape} escape, "
             f"{steps - escape - stabilization} refinement and {stabilization} stabilization updates, with no plateau stopping. All 21 cameras are used; "
             "72 training phases (0,5,…355) and 72 validation phases (2,7,…357) are disjoint. "
             "Validation is evaluated every 20 updates. Reported curves span all 360 phases.", "",
             "| Asteroid | Calibration | Selected step | Initial RMSE | Selected RMSE | Terminal RMSE |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        a = row["asteroid"]
        lines.append(f"| [{a}](asteroid{a:02d}/report.md) | {row['calibration_status']} | {row['selected_iteration']} | "
                     f"{row['initial_rmse']:.5f} | {row['selected_rmse']:.5f} | {row['terminal_rmse']:.5f} |")
    if rows:
        lines += ["", "![Selected shapes](shapes_overview.png)", "",
                  "[Figure PDF](shapes_overview.pdf) · [Measurements CSV](results.csv)", ""]
    lines += ["", "Each linked asteroid report contains all-camera lightcurves, multiple shape views, "
              "selected and terminal STL meshes, convergence plots, and a deformation GIF through the final update.", "",
              "Geometry scores are null for hidden shapes. Public scores use the raw reference and global scale alignment; "
              "they measure the outer radial surface. Mesh validity is reported separately. Selected states minimize "
              "validation error and can precede the last update. Runtime estimates assume an available GPU and an awake machine.", ""]
    if (root / "status.json").exists():
        lines += ["[Execution status](status.json)", ""]
    (root / "report.md").write_text("\n".join(lines))
    return rows


def verify_campaign_artifacts(root):
    root = Path(root)
    frozen = json.loads((root / "frozen_configuration.json").read_text())
    decoded = {}
    for key, values in frozen["configs"].items():
        config = FitConfig(**values)
        body = root / f"asteroid{config.asteroid:02d}"
        audit_run(body / VARIANT, config)
        with np.load(root / "sphere_1024.npz") as sphere:
            trajectory = load_trajectory(body / VARIANT / "trajectory.npz")
            if not np.array_equal(sphere["faces"], trajectory.faces) or not np.allclose(
                    sphere["vertices"], trajectory.vertices[0], atol=1e-7, rtol=1e-7):
                raise RuntimeError(f"run {key} does not use the shared sphere")
        decoded.update({f"{body.name}/{name}": count for name, count in verify_body(body, config).items()})
    links = re.findall(r"\]\(([^)]+)\)", (root / "report.md").read_text())
    for link in links:
        if not (root / link).is_file():
            raise RuntimeError(f"missing campaign report link: {link}")
    result = dict(status="passed", audited_runs=len(frozen["configs"]),
                  resolved_report_links=len(links), decoded_gifs=decoded)
    write_json(root / "artifact_verification.json", result)
    return result
