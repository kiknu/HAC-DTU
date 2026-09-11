"""Post-fit measurements and figures for the controlled representation study."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from forward_validation import normalize_challenge_mesh, load_benchmark_mesh
from shape_model import TriangleMesh, validate_triangle_mesh
from visualize_fit import plot_shape_grid
from .animation import animate_trajectory, animate_comparison
from .calibration import curve_metrics
from .datasets import load_lightcurves, public_mesh_path
from .inversion import FitConfig, PlateauState, _build_model, _evaluate
from .reporting import export_mesh
from .trajectory import load_trajectory

VARIANTS = (("radial_sh_l6", "Radial SH L6", "radial_sh", 6),
            ("radial_sh_l11", "Radial SH L11", "radial_sh", 11),
            ("octantoid_l6", "Octantoid L6", "octantoid", 6))
OCTANTOID_VARIANTS = tuple((f"octantoid_l{degree}", f"Octantoid L{degree}", "octantoid", degree)
                         for degree in (2, 4, 6))
COLORS = ("tab:blue", "tab:orange", "tab:green")


def variants_for(profile):
    if profile == "l4_survey":
        return (("octantoid_l4", "Octantoid L4", "octantoid", 4),)
    return OCTANTOID_VARIANTS if profile == "octantoid" else VARIANTS


def study_description(root):
    path = Path(root) / "study.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def outer_radii(mesh, directions):
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    hits, indices, _ = RayMeshIntersector(mesh).intersects_location(
        np.zeros_like(directions), directions, multiple_hits=True)
    radii = np.zeros(len(directions))
    np.maximum.at(radii, indices, np.linalg.norm(hits, axis=1))
    return radii


def geometry_metrics(vertices, faces, reference, count=4096):
    i = np.arange(count, dtype=float) + .5
    z = 1 - 2 * i / count
    phi = i * np.pi * (3 - np.sqrt(5))
    directions = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    fitted = trimesh.Trimesh(vertices, faces, process=False)
    actual, target = outer_radii(fitted, directions), outer_radii(reference, directions)
    missed = (actual <= 0) | (target <= 0)
    if missed.any():
        return dict(sample_directions=count, missed_rays=int(missed.sum()),
                    scale=None, radial_log_rms=None, radial_log_max=None)
    scale = float(np.exp(np.mean(np.log(target) - np.log(actual))))
    error = np.log(scale * actual / target)
    return dict(sample_directions=count, missed_rays=0, scale=scale,
                radial_log_rms=float(np.sqrt(np.mean(error**2))),
                radial_log_max=float(np.max(np.abs(error))))


def verify_curve_replay(saved, replay):
    """Allow float32 GPU scatter variation, retaining the original scored curves."""
    if (saved.shape != replay.shape or not np.isfinite(replay).all()
            or not np.allclose(saved, replay, atol=1e-4, rtol=1e-4)):
        raise RuntimeError("saved fitted curves disagree with selected checkpoint")
    delta = replay - saved
    return dict(max_absolute_difference=float(np.max(np.abs(delta))),
                rms_difference=float(np.sqrt(np.mean(delta**2))), atol=1e-4, rtol=1e-4)


def audit_run(run_dir, config):
    trajectory = load_trajectory(run_dir / "trajectory.npz")
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((run_dir / "run.json").read_text())
    shape, _ = _build_model(config, torch.device("cpu"))
    best = int(checkpoint["best"]["history_index"])
    actual_steps = metadata["summary"]["iterations_run"]
    reason = metadata["summary"].get("stop_reason", "iteration_cap")
    valid_termination = ((actual_steps == config.steps and reason == "iteration_cap")
                         or (config.early_stopping and 0 < actual_steps < config.steps
                             and reason == "validation_plateau"
                             and checkpoint.get("plateau_state", {}).get("stop_reason") == reason))
    arrays = (trajectory.parameters, trajectory.vertices, trajectory.losses)
    checks = dict(
        iterations=valid_termination,
        states=len(trajectory.vertices) == actual_steps + 1,
        faces=len(trajectory.faces) == config.faces,
        finite=all(np.isfinite(a).all() for a in arrays),
        parameter_count=trajectory.parameters.shape[1] == shape.n_parameters,
        sphere=bool(np.all(trajectory.parameters[0] == 0)
                    and np.allclose(np.linalg.norm(trajectory.vertices[0], axis=1), .5, atol=1e-6)),
        template=bool(np.array_equal(trajectory.faces, shape.faces.numpy())
                      and np.allclose(trajectory.vertices[0], shape.mesh(shape.initial_parameters()).vertices.numpy(), atol=1e-7, rtol=1e-7)),
        selected_index=trajectory.best_index == best == metadata["summary"]["best_iteration"],
        selected_parameters=bool(np.array_equal(trajectory.parameters[best], checkpoint["best"]["parameters"].numpy())),
        last_parameters=bool(np.array_equal(trajectory.parameters[-1], checkpoint["parameters"].numpy())),
        config=FitConfig(**metadata["config"]) == config,
        checkpoint_step=checkpoint["step"] == actual_steps,
        optimizer_finite=all(bool(torch.isfinite(value).all())
                             for state in checkpoint["optimizer"]["state"].values()
                             for value in state.values() if isinstance(value, torch.Tensor)),
        optimizer_step=all(int(state["step"]) == actual_steps for state in checkpoint["optimizer"]["state"].values()),
    )
    if config.early_stopping:
        replay = PlateauState()
        history = checkpoint.get("optimization_history", [])
        checks["optimization_history"] = len(history) == actual_steps
        if checks["optimization_history"]:
            for validation in metadata["validation_history"][1:]:
                iteration = validation["step"]
                replay.observe(config, iteration, validation["rmse"], history[iteration-1]["learning_rate"])
            checks["plateau_replay"] = asdict(replay) == checkpoint.get("plateau_state")
            checks["stop_reason"] = replay.stop_reason == reason
    expected = shape.mesh(checkpoint["best"]["parameters"]).vertices.numpy()
    checks["selected_mesh"] = bool(np.allclose(expected, trajectory.vertices[best], atol=2e-6, rtol=2e-6))
    final_stl = trimesh.load(run_dir / "mesh_final.stl", process=False, force="mesh")
    checks["exported_mesh"] = bool(np.allclose(final_stl.triangles, trajectory.vertices[best][trajectory.faces], atol=1e-6))
    validity = {}
    for name, index in (("selected", best), ("last", actual_steps)):
        result = validate_triangle_mesh(TriangleMesh(torch.tensor(trajectory.vertices[index]),
                                                    torch.tensor(trajectory.faces)), check_self_intersections=True)
        validity[name] = asdict(result)
    result = dict(passed=all(checks.values()), checks=checks, mesh_validity=validity)
    write_json(run_dir / "audit.json", result)
    if not result["passed"]:
        raise RuntimeError(f"run audit failed: {run_dir}: {checks}")
    return result


def measure_run(run_dir, config, calibration):
    audit = audit_run(run_dir, config)
    trajectory = load_trajectory(run_dir / "trajectory.npz")
    metadata = json.loads((run_dir / "run.json").read_text())
    dataset = load_lightcurves(config.asteroid)
    attempts_path = run_dir / "attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    total_wall = sum(attempt.get("elapsed_seconds", 0.) for attempt in attempts) if attempts else metadata["summary"]["runtime_seconds"]
    with np.load(run_dir / "curves.npz") as archive:
        curves = {key: archive[key].copy() for key in archive.files}
    if curves["fitted"].shape != (360, 21) or not np.isfinite(curves["fitted"]).all():
        raise RuntimeError("invalid final full-turn curves")
    if not np.array_equal(curves["observed"], dataset.curves):
        raise RuntimeError("reported observations differ from the source dataset")
    shape, model = _build_model(config, torch.device(config.device), calibration)
    best_p = torch.tensor(trajectory.parameters[trajectory.best_index], device=config.device)
    _, checked_best = _evaluate(model, best_p, dataset, calibration)
    replay_check = verify_curve_replay(curves["fitted"], checked_best)
    last_p = torch.tensor(trajectory.parameters[-1], device=config.device)
    if trajectory.best_index == len(trajectory.vertices)-1:
        last_curves = curves["fitted"]
    else:
        _, last_curves = _evaluate(model, last_p, dataset, calibration)
    del model, shape
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    np.savez_compressed(run_dir / "curves_last.npz", indices=curves["indices"],
                        observed=curves["observed"], fitted=last_curves, columns=curves["columns"])
    export_mesh(trajectory.vertices[-1], trajectory.faces, run_dir / "mesh_last.stl")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    subsets = {"full_turn": np.arange(360), "validation": np.array(manifest["validation_indices"]),
               "remaining": np.array(manifest["remaining_indices"])}
    photometry = {}
    for name, values in (("initial", curves["initial"]), ("selected", curves["fitted"]), ("last", last_curves)):
        photometry[name] = {subset: float(np.sqrt(np.mean((values[indices]-dataset.curves[indices])**2)))
                            for subset, indices in subsets.items()}
    write_json(run_dir / "last_camera_metrics.json", curve_metrics(last_curves, dataset.curves, dataset.columns))
    geometry = {name: None for name in ("initial", "selected", "last")}
    if config.asteroid <= 3:
        reference = normalize_challenge_mesh(trimesh.load(public_mesh_path(config.asteroid), force="mesh", process=True))
        geometry = {name: geometry_metrics(trajectory.vertices[index], trajectory.faces, reference)
                    for name, index in (("initial", 0), ("selected", trajectory.best_index), ("last", -1))}
    initial = trajectory.vertices[0]
    directions = initial / np.linalg.norm(initial, axis=1, keepdims=True)
    displacement = (trajectory.vertices - initial) / .5
    radial = np.sum(displacement * directions[None], axis=2)
    tangential = displacement - radial[:, :, None] * directions[None]
    deformation = dict(iteration=np.arange(len(trajectory.vertices)),
                       radial_rms=np.sqrt(np.mean(radial**2, axis=1)),
                       tangential_rms=np.sqrt(np.mean(np.sum(tangential**2, axis=2), axis=1)))
    np.savez_compressed(run_dir / "deformation.npz", **deformation)
    measured = dict(photometry=photometry, geometry=geometry, audit=audit, curve_replay=replay_check,
                    total_fit_wall_seconds=total_wall, attempts=attempts,
                    summary=metadata["summary"], best_iteration=trajectory.best_index,
                    selected_improvement_fraction=1-photometry["selected"]["full_turn"]/photometry["initial"]["full_turn"])
    write_json(run_dir / "comparison_metrics.json", measured)
    return measured


def plot_curves(paths, labels, output, *, last=False, last_step=None):
    collections = []
    for path in paths:
        with np.load(path / ("curves_last.npz" if last else "curves.npz")) as archive:
            collections.append({key: archive[key].copy() for key in archive.files})
    fig, axes = plt.subplots(7, 3, figsize=(16, 22), sharex=True)
    for j, axis in enumerate(axes.flat):
        axis.plot(np.arange(1,361), collections[0]["observed"][:, j], "k-", lw=1, label="Observed")
        if not last:
            axis.plot(np.arange(1,361), collections[0]["initial"][:, j], color="gray", lw=.8, ls="--", label="Initial sphere")
        for curves, label, color in zip(collections, labels, COLORS):
            axis.plot(np.arange(1,361), curves["fitted"][:, j], color=color, lw=1, label=label)
        axis.set_title(str(collections[0]["columns"][j]))
        axis.set(xlabel="nominal data phase [deg]", ylabel="normalized intensity")
        axis.grid(alpha=.2)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(names))
    fig.suptitle((f"Step {last_step} curves" if last_step is not None else "Terminal curves")
                 if last else "Selected-best curves", y=.985)
    fig.tight_layout(rect=(0,0,1,.965))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def report_asteroid(root, asteroid, configs, calibrations):
    if study_description(root).get("profile") == "l4_survey":
        from .l4_survey_reporting import report_asteroid as report_survey_asteroid
        if len(configs) != 1:
            raise ValueError("L4 survey requires one configuration per asteroid")
        return report_survey_asteroid(root, configs[0], calibrations[asteroid])
    variants = variants_for(study_description(root).get("profile", "full"))
    step_counts = {c.steps for c in configs}
    if len(step_counts) != 1 or len(configs) != len(variants):
        raise ValueError("comparison reports require three equally budgeted configurations")
    steps = step_counts.pop()
    body = Path(root) / f"asteroid{asteroid:02d}"
    paths = [body / key for key, *_ in variants]
    if not all((path / "run.json").exists() for path in paths):
        return
    labels = [label for _, label, *_ in variants]
    for path, config in zip(paths, configs):
        if not (path / "comparison_metrics.json").exists():
            measure_run(path, config, calibrations[asteroid])
        else:
            audit_run(path, config)
    histories = [load_trajectory(path / "trajectory.npz") for path in paths]
    if not all(np.array_equal(histories[0].vertices[0], t.vertices[0])
               and np.array_equal(histories[0].faces, t.faces) for t in histories):
        raise RuntimeError("paired models do not share their initial sphere")
    plot_curves(paths, labels, body / "lightcurves_selected.png")
    adaptive = any(c.early_stopping for c in configs)
    terminal_labels = [f"{label} (iteration {len(t.vertices)-1})" for label,t in zip(labels,histories)]
    plot_curves(paths, terminal_labels if adaptive else labels,
                body / ("lightcurves_last.png" if adaptive else f"lightcurves_step{steps}.png"),
                last=True, last_step=None if adaptive else steps)
    for path, label in zip(paths, labels):
        plot_curves([path], [label], path / "lightcurves_all_cameras.png")
    proxy = load_benchmark_mesh(asteroid, 1280, torch.device("cpu"), body / "mesh_cache")
    meshes = [(histories[0].vertices[0], histories[0].faces)]
    for path, t in zip(paths, histories):
        measurement = json.loads((path / "comparison_metrics.json").read_text())
        scale = measurement["geometry"]["selected"]["scale"] or 1.
        meshes.append((t.vertices[t.best_index] * scale, t.faces))
    measurement = json.loads((paths[0] / "comparison_metrics.json").read_text())
    meshes[0] = (meshes[0][0] * (measurement["geometry"]["initial"]["scale"] or 1.), meshes[0][1])
    meshes.append((proxy.vertices.numpy(), proxy.faces.numpy()))
    plot_shape_grid(meshes, ["Initial sphere", *labels, "Public display proxy (1280 faces)"],
                    [(.6,.6,.6), (.25,.5,.8), (.9,.55,.2), (.35,.7,.45), (.6,.65,.6)],
                    body / "shapes_selected.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for path, label, color in zip(paths, labels, COLORS):
        run = json.loads((path / "run.json").read_text())
        history = run["validation_history"]
        axes[0].plot([row["step"] for row in history], [row["rmse"] for row in history], label=label, color=color)
        best = run["summary"]["best_iteration"]
        axes[0].scatter([best], [run["summary"]["best_validation_rmse"]], color=color, marker="*")
        if adaptive:
            axes[0].scatter([history[-1]["step"]], [history[-1]["rmse"]], color=color, marker="x")
        with np.load(path / "deformation.npz") as values:
            for axis, name in zip(axes[1:], ("radial_rms", "tangential_rms")):
                axis.plot(values["iteration"], values[name], label=label, color=color)
    for axis, title, ylabel in zip(axes, ("Validation convergence", "Radial displacement", "Tangential displacement"),
                                   ("RMSE", "RMS / initial radius", "RMS / initial radius")):
        axis.set(xlabel="iteration", ylabel=ylabel, title=title)
        axis.grid(alpha=.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(body / "convergence_deformation.png", dpi=160)
    plt.close(fig)
    limit = 1.08 * max(float(np.abs(t.vertices).max()) for t in histories)
    for path, label in zip(paths, labels):
        for name, all_iterations in (("evolution.gif", False), ("full_history.gif", True)):
            if not (path / name).exists():
                animate_trajectory(path / "trajectory.npz", path / name, stride=5,
                                   all_iterations=all_iterations, axis_limit=limit, label=label)
    if not (body / "deformation_comparison.gif").exists():
        reasons = [json.loads((path / "run.json").read_text())["summary"].get("stop_reason", "iteration_cap") for path in paths]
        animate_comparison([path / "trajectory.npz" for path in paths], labels,
                           body / "deformation_comparison.gif", stop_reasons=reasons if adaptive else None)
    write_json(body / "report_complete.json", dict(asteroid=asteroid, steps=steps,
               actual_steps=[len(t.vertices)-1 for t in histories], variants=[v[0] for v in variants]))


def report_campaign(root):
    root = Path(root)
    if study_description(root).get("profile") == "l4_survey":
        from .l4_survey_reporting import report_campaign as report_survey_campaign
        return report_survey_campaign(root)
    description_path = root / "study.json"
    description = json.loads(description_path.read_text()) if description_path.exists() else {}
    asteroids = tuple(description.get("asteroids", (1, 2, 3)))
    steps = description.get("selected_steps", description.get("steps", 500))
    adaptive = description.get("profile") == "octantoid"
    variants = variants_for(description.get("profile", "full"))
    expected_runs = len(asteroids) * len(variants)
    rows = []
    camera_rows = []
    for asteroid in asteroids:
        for key, label, representation, degree in variants:
            path = root / f"asteroid{asteroid:02d}" / key
            if not (path / "comparison_metrics.json").exists():
                continue
            metrics = json.loads((path / "comparison_metrics.json").read_text())
            summary = metrics["summary"]
            for state, filename in (("selected", "full_metrics.json"), ("last", "last_camera_metrics.json")):
                for camera in json.loads((path / filename).read_text())["cameras"]:
                    camera_rows.append(dict(asteroid=asteroid, model=label, state=state, **camera))
            rows.append(dict(asteroid=asteroid, model=label, representation=representation, degree=degree,
                             parameters=summary["parameter_count"], faces=summary["face_count"],
                             iterations=summary["iterations_run"], best_iteration=metrics["best_iteration"],
                             stop_reason=summary.get("stop_reason", "iteration_cap"), iteration_cap=steps,
                             initial_rmse=metrics["photometry"]["initial"]["full_turn"],
                             selected_rmse=metrics["photometry"]["selected"]["full_turn"],
                             last_rmse=metrics["photometry"]["last"]["full_turn"],
                             validation_rmse=metrics["photometry"]["selected"]["validation"],
                             remaining_rmse=metrics["photometry"]["selected"]["remaining"],
                             improvement_fraction=metrics["selected_improvement_fraction"],
                             radial_log_rms=metrics["geometry"]["selected"]["radial_log_rms"],
                             last_radial_log_rms=metrics["geometry"]["last"]["radial_log_rms"],
                             selected_mesh_valid=metrics["audit"]["mesh_validity"]["selected"]["valid"],
                             last_mesh_valid=metrics["audit"]["mesh_validity"]["last"]["valid"],
                             runtime_seconds=metrics.get("total_fit_wall_seconds", summary["runtime_seconds"]),
                             peak_memory_mib=max([summary["peak_memory_mib"]] +
                                                [a["peak_memory_mib"] for a in metrics.get("attempts", [])
                                                 if a.get("peak_memory_mib") is not None])))
    if rows:
        with (root / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if camera_rows:
        with (root / "camera_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(camera_rows[0]))
            writer.writeheader()
            writer.writerows(camera_rows)
    lines = ["# Octantoid degree comparison" if adaptive else "# Radial spherical harmonics versus octantoids", "",
             f"Measured runs: **{len(rows)}/{expected_runs}**. Each run uses one 1024-face sphere and {'up to ' if adaptive else ''}{steps} updates.", "",
             ("Octantoid L2/L4/L6 have 26/74/146 coefficients. " if adaptive else
              "Radial L6 has 48 coefficients; radial L11 has 143; octantoid L6 has 146. ") +
             "The common spectral prior is λ Σ l³c² / 146. All fits share the optimizer schedule, sphere and phase split.", "",
             f"| Asteroid | Model | Best step | Initial RMSE | Selected RMSE | {'Terminal' if adaptive else f'Step-{steps}'} RMSE | Radial-log RMS | Valid selected mesh | Stop step / reason |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        geometry = "unavailable" if row["radial_log_rms"] is None else f"{row['radial_log_rms']:.5f}"
        lines.append(f"| {row['asteroid']} | {row['model']} | {row['best_iteration']} | {row['initial_rmse']:.5f} | "
                     f"{row['selected_rmse']:.5f} | {row['last_rmse']:.5f} | {geometry} | {row['selected_mesh_valid']} | "
                     f"{row['iterations']} / {row['stop_reason']} |")
    lines += ["", "[Full measurements](results.csv) · [Per-camera measurements](camera_metrics.csv) · [Execution status](status.json)", ""]
    for asteroid in asteroids:
        body = root / f"asteroid{asteroid:02d}"
        if not (body / "report_complete.json").exists():
            continue
        rel = body.name
        lines += [f"## Asteroid {asteroid}", "",
                  f"![Selected lightcurves]({rel}/lightcurves_selected.png)", "",
                  f"![Selected shapes]({rel}/shapes_selected.png)", "",
                  f"![Convergence and deformation]({rel}/convergence_deformation.png)", "",
                  f"![Complete deformation history]({rel}/deformation_comparison.gif)", "",
                  (f"[Terminal lightcurves]({rel}/lightcurves_last.png)" if adaptive else
                   f"[Step-{steps} lightcurves]({rel}/lightcurves_step{steps}.png)"), ""]
        for key, label, *_ in variants:
            lines.append(f"- {label}: [selected evolution]({rel}/{key}/evolution.gif), "
                         f"[full history]({rel}/{key}/full_history.gif), [mesh]({rel}/{key}/mesh_final.stl), "
                         f"[metrics and audit]({rel}/{key}/comparison_metrics.json)")
        lines.append("")
    lines += ["## Interpretation", "",
              f"Selected outputs use minimum exact-renderer validation RMSE, which can occur before step {steps}. "
              "Full-history animations show all optimization stages at five-step intervals, retaining selected iterations.", "",
              "These are experimental, fixed-recipe inverse fits. The per-body image model is calibrated on public "
              "full-turn data, so validation is conditional on that calibration. Reference geometry never enters "
              "the inverse loss or checkpoint selection. Comparisons do not establish the best attainable result after "
              "separate tuning. Geometry scores measure outer radial surfaces against the raw public mesh after "
              "global scale alignment only; they do not characterize hidden cavities. Display proxies are labelled "
              "and are not the raw geometry used for scoring. Invalid meshes and missed rays are recorded explicitly.", ""]
    if adaptive:
        lines += ["Plateau stopping requires at least 200 updates, 100 updates without meaningful validation improvement "
                  "(max of 0.0001 and 0.5% relative), then 40 confirmation updates at a reduced learning rate. "
                  "It is an operational stopping rule, not proof of optimality. Stopped models hold their terminal "
                  "states in synchronized GIFs. The benchmark budget includes 10% fitting contingency and 30 minutes "
                  "for reporting; runtime is an estimate.", ""]
    (root / "report.md").write_text("\n".join(lines))
    return rows


def verify_campaign_artifacts(root):
    """Decode delivered animations and check report links before completion."""
    from PIL import Image
    root = Path(root)
    description = study_description(root)
    if description.get("profile") == "l4_survey":
        from .l4_survey_reporting import verify_campaign_artifacts as verify_survey
        return verify_survey(root)
    runs = [root / f"asteroid{a:02d}" / key for a in description.get("asteroids", (1,2,3))
            for key, *_ in variants_for(description.get("profile", "full"))]
    for run in runs:
        if not json.loads((run / "audit.json").read_text())["passed"]:
            raise RuntimeError(f"failed run audit: {run}")
    links = re.findall(r"\]\(([^)]+)\)", (root / "report.md").read_text())
    for link in links:
        if not (root / link).is_file():
            raise RuntimeError(f"missing report artifact: {link}")
    decoded = {}
    for path in sorted(root.glob("asteroid*/*.gif")) + sorted(root.glob("asteroid*/*/*.gif")):
        with Image.open(path) as gif:
            for frame in range(gif.n_frames):
                gif.seek(frame)
                gif.load()
            decoded[str(path.relative_to(root))] = gif.n_frames
    expected_gifs = len(description.get("asteroids", (1,2,3))) * 7
    if len(decoded) != expected_gifs:
        raise RuntimeError(f"expected {expected_gifs} GIFs, found {len(decoded)}")
    result = dict(status="passed", audited_runs=len(runs), resolved_report_links=len(links),
                  decoded_gifs=decoded)
    write_json(root / "artifact_verification.json", result)
    return result
