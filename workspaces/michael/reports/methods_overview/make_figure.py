"""Rebuild static report figures/tables from verified saved fits, without fitting.

Run with the project Python environment. All rendering is on CPU. Default
replays the frozen input list; --refresh-inputs discovers additional reported
200-update L4 fits. Inputs and outputs are hashed in evidence.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from forward_validation import normalize_challenge_mesh, star_proxy_mesh
from hac_asteroids.datasets import load_lightcurves, public_mesh_path
from hac_asteroids.trajectory import load_trajectory
from pytorch3d_forward import (ObservationBatch, PyTorch3DLightcurveModel,
                              RendererConfig, image_response)
from shape_model import OctantoidShapeModel, RadialHarmonicShapeModel, TriangleMesh

SURVEY = ROOT / "artifacts/octantoid_l4_all10_200_20260910"
DEGREE = ROOT / "artifacts/octantoids_l2_l4_l6_20260910"
QUICK = ROOT / "artifacts/representations_quick_a3_120_20260909"
FIG = HERE / "figures"
SOURCES = {}
CHECKS = {}
FITS = {}
BLUE = (.29, .55, .78)
GREY = (.69, .71, .73)
GREEN = (.58, .69, .56)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    digest = sha(path)
    if key in SOURCES and SOURCES[key] != digest:
        raise RuntimeError(f"Input changed while building report: {key}")
    SOURCES[key] = digest
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_fit(path):
    """Read-only checkpoint/mesh/curve audit; never calls report writers."""
    if path in FITS:
        return FITS[path]
    run = read_json(path / "run.json")
    cfg, summary = run["config"], run["summary"]
    cmp = read_json(path / "comparison_metrics.json")
    trajectory = load_trajectory(source(path / "trajectory.npz"))
    checkpoint = torch.load(source(path / "checkpoint.pt"), map_location="cpu", weights_only=False)
    with np.load(source(path / "curves.npz"), allow_pickle=False) as archive:
        curves = {key: archive[key].copy() for key in archive.files}
    best = trajectory.best_index
    assert best == summary["best_iteration"] == checkpoint["best"]["history_index"]
    assert len(trajectory.vertices) == checkpoint["step"] + 1 == summary["iterations_run"] + 1
    assert np.array_equal(trajectory.parameters[best], checkpoint["best"]["parameters"].numpy())
    assert np.array_equal(trajectory.parameters[-1], checkpoint["parameters"].numpy())
    assert np.all(trajectory.parameters[0] == 0)
    assert np.allclose(np.linalg.norm(trajectory.vertices[0], axis=1), .5, atol=1e-6)
    assert len(trajectory.faces) == 1024
    shape_class = OctantoidShapeModel if cfg["representation"] == "octantoid" else RadialHarmonicShapeModel
    shape = shape_class(torch.tensor(trajectory.vertices[0] / .5), torch.tensor(trajectory.faces),
                        lmax=cfg["lmax"], r0=.5, check_self_intersections=False)
    with torch.no_grad():
        rebuilt = shape.mesh(torch.tensor(trajectory.parameters[best])).vertices.numpy()
    assert np.allclose(rebuilt, trajectory.vertices[best], atol=2e-6, rtol=2e-6)
    mesh = trimesh.load(source(path / "mesh_final.stl"), force="mesh", process=False)
    assert np.allclose(mesh.triangles, trajectory.vertices[best][trajectory.faces], atol=1e-6)
    dataset = load_lightcurves(cfg["asteroid"])
    source(dataset.source)
    assert np.array_equal(curves["observed"], dataset.curves)
    assert np.array_equal(curves["columns"], dataset.columns)
    assert np.array_equal(curves["indices"], np.arange(360))
    assert curves["fitted"].shape == (360, 21)
    scores = {}
    for state, key in (("initial", "initial"), ("selected", "fitted")):
        residual = curves[key].astype(np.float64) - curves["observed"]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        assert np.isclose(rmse, cmp["photometry"][state]["full_turn"], rtol=0, atol=1e-12)
        scores[state] = rmse
    val = float(np.sqrt(np.mean((curves["fitted"][cfg["validation_indices"]]
                                 - curves["observed"][cfg["validation_indices"]]) ** 2)))
    assert np.isclose(val, summary["best_validation_rmse"], atol=1e-7, rtol=0)
    assert best == min(run["validation_history"], key=lambda h: h["rmse"])["step"]
    assert cmp["audit"]["passed"] and all(cmp["audit"]["checks"].values())
    CHECKS[str(path.relative_to(ROOT))] = dict(selected_iteration=best,
        update_budget=cfg["steps"], iterations=summary["iterations_run"],
        face_count=len(trajectory.faces), full_turn_rmse=scores,
        checkpoint_parameters_match=True, selected_mesh_rebuilt=True,
        exported_mesh_matches=True, observations_match_source=True,
        saved_audit_passed=True, saved_curve_replay=cmp["curve_replay"])
    value = dict(run=run, cmp=cmp, trajectory=trajectory, curves=curves, scores=scores)
    FITS[path] = value
    return value


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def mesh_image(vertices, faces, view, extent, color, size=380):
    """CPU orthographic mesh illustration, flat normals and fixed studio light.

    This is a geometry display, not the photometric forward model. No surface
    simplification, warping, alignment rotation, or shape smoothing is applied.
    """
    elev, azim = np.radians(view)
    eye = np.array([np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0.])
    up = np.cross(eye, right)
    coord = ((np.arange(size)+.5)/size*2-1)*extent
    x, y = np.meshgrid(coord, -coord)
    origins = x.ravel()[:, None]*right + y.ravel()[:, None]*up + 4*extent*eye
    rays = np.broadcast_to(-eye, origins.shape).copy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    ids, rayids, _ = RayMeshIntersector(mesh).intersects_id(origins, rays,
                                                         multiple_hits=False, return_locations=True)
    # Light fixed in world coordinates so matched geometry views share shading.
    light = np.array([.4, .3, .85]); light /= np.linalg.norm(light)
    normals = mesh.face_normals[ids]
    shade = .34 + .50*np.maximum(normals @ light, 0) + .16*np.maximum(normals @ eye, 0)
    image = np.ones((size*size, 3), dtype=np.float32)
    image[rayids] = np.clip(np.asarray(color)[None, :]*shade[:, None], 0, 1)
    return image.reshape(size, size, 3)


def mesh_grid(meshes, labels, colors, views, name, size=(7.2, 3.8)):
    extent = 1.10 * max(np.linalg.norm(vertices, axis=1).max() for vertices, _ in meshes)
    fig, axes = plt.subplots(len(views), len(meshes), figsize=size, squeeze=False)
    fig.subplots_adjust(left=.075, right=.995, bottom=.01, top=.90, wspace=.025, hspace=.07)
    for i, (elev, azim, label) in enumerate(views):
        for j, ((vertices, faces), color) in enumerate(zip(meshes, colors)):
            ax = axes[i,j]
            ax.imshow(mesh_image(vertices, faces, (elev, azim), extent, color))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(labels[j], fontsize=10)
            if j == 0:
                ax.text(-.08, .5, label, rotation=90, va="center", ha="center",
                        transform=ax.transAxes, fontsize=9)
    save(fig, name)
    return dict(views=[dict(elevation=e, azimuth=a, label=s) for e,a,s in views],
                common_half_extent=float(extent), display="CPU first-hit orthographic, flat normals, studio light")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-inputs", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    FIG.mkdir(exist_ok=True)
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "data").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    manifest = HERE / "inputs.json"
    if args.refresh_inputs or not manifest.exists():
        asteroids = [a for a in range(1,11) if (SURVEY/f"asteroid{a:02d}/report_complete.json").is_file()]
        write_json(manifest, dict(l4_asteroids=asteroids,
                   selection="Saved 200-update fits with report manifests; no result inferred from execution status."))
    asteroids = read_json(manifest)["l4_asteroids"]
    rows = []
    for a in range(1,11):
        label = str(a) + (r"$^\dagger$" if a > 3 else "")
        if a not in asteroids:
            rows.append(f"{label} & \\textemdash & \\textemdash & \\textemdash & \\textemdash \\\\")
            continue
        body = SURVEY/f"asteroid{a:02d}"
        marker = read_json(body/"report_complete.json")
        metrics = read_json(body/"metrics.json")
        # Hash only the scientific inputs consumed, not unrelated animations.
        for name in ("metrics.json", "octantoid_l4/run.json", "octantoid_l4/trajectory.npz",
                     "octantoid_l4/curves.npz", "octantoid_l4/comparison_metrics.json", "octantoid_l4/mesh_final.stl"):
            assert sha(source(body/name)) == marker["artifacts"][name]
        fit = verify_fit(body/"octantoid_l4")
        assert fit["run"]["config"]["steps"] == metrics["iterations"] == 200
        expected = dict(lmax=4, seed=0, image_size=96, frame_count=72,
                        validation_frame_count=72, escape_steps=40, stabilization_steps=40)
        assert all(fit["run"]["config"][k] == value for k, value in expected.items())
        assert metrics["calibration_transfer"] == (a > 3)
        assert metrics["selected_iteration"] == fit["trajectory"].best_index
        for state in ("initial", "selected"):
            assert metrics[f"{state}_rmse"] == fit["scores"][state]
        initial, selected = fit["scores"]["initial"], fit["scores"]["selected"]
        improvement = 100 * (initial-selected)/initial
        rows.append(f"{label} & {fit['trajectory'].best_index} & {initial:.4f} & {selected:.4f} & {improvement:.1f}\\% \\\\")
    (HERE/"tables/l4_rows.tex").write_text("\n".join(rows)+"\n")
    print("Verified L4 rows:", asteroids, flush=True)

    # Preserve the degree comparison, including its selected and terminal marks.
    fig, ax = plt.subplots(figsize=(7.2, 2.65), layout="constrained")
    degree_rows = []
    for degree, color in zip((2,4,6), ("#377eb8", "#21885a", "#c64c45")):
        fit = verify_fit(DEGREE/f"asteroid01/octantoid_l{degree}")
        history, summary = fit["run"]["validation_history"], fit["run"]["summary"]
        steps, errors = [h["step"] for h in history], [h["rmse"] for h in history]
        best = summary["best_iteration"]
        ax.plot(steps, errors, label=f"Octantoid L{degree}", color=color, linewidth=1.7)
        ax.scatter([best], [next(h["rmse"] for h in history if h["step"] == best)], marker="*", s=105, color=color, zorder=5)
        ax.scatter([steps[-1]], [errors[-1]], marker="x", s=40, color=color, zorder=5)
        degree_rows.append(f"L{degree} & {best} & {steps[-1]} & {fit['scores']['selected']:.6f} & {fit['cmp']['photometry']['last']['full_turn']:.6f} \\\\")
    ax.set(xlabel="Optimizer update", ylabel="Validation RMSE", xlim=(-5,330))
    ax.grid(alpha=.2); ax.legend(loc="upper left", frameon=False)
    save(fig, "degree_comparison")
    (HERE/"tables/degree_rows.tex").write_text("\n".join(degree_rows)+"\n")
    quick_rows=[]
    for variant, label in (("radial_sh_l6", "Radial harmonics L6"), ("radial_sh_l11", "Radial harmonics L11"), ("octantoid_l6", "Octantoid L6")):
        fit = verify_fit(QUICK/f"asteroid03/{variant}")
        quick_rows.append(f"{label} & {fit['run']['summary']['parameter_count']} & {fit['trajectory'].best_index} & {fit['scores']['selected']:.6f} & {fit['cmp']['photometry']['last']['full_turn']:.6f} \\\\")
    (HERE/"tables/quick_rows.tex").write_text("\n".join(quick_rows)+"\n")

    fit2 = verify_fit(SURVEY/"asteroid02/octantoid_l4")
    traj2 = fit2["trajectory"]
    # Actual 96x96 image formation on the selected fit, using its calibration.
    cal = fit2["run"]["calibration"]
    cfg = RendererConfig(image_size=96, shadow_size=96, face_limit=1024,
        camera_distance_radii=cal["camera_distance_radii"], lambertian_fraction=cal["lambertian_fraction"],
        response=cal["response"], radiance_gain=cal["radiance_gain"], projection=cal["projection"],
        intensity_slope=8., batch_size=2)
    obs = ObservationBatch.challenge(torch.tensor([np.pi/4], dtype=torch.float32), ("0_hor1", "225_bottom"))
    with torch.no_grad():
        prediction = PyTorch3DLightcurveModel(cfg)(TriangleMesh(torch.tensor(traj2.vertices[traj2.best_index]), torch.tensor(traj2.faces)),
            obs, extraction="exact", return_linear_images=True, compute_binary=False)
        linear = prediction.images[0]
        encoded = image_response(linear, cfg.response, cfg.radiance_gain)
        quantized = torch.round(255*encoded)/255
        thresholded = quantized*(quantized > .1)
    np.testing.assert_allclose(thresholded.sum((-2,-1)).numpy(), prediction.raw_intensity[0].numpy(), rtol=1e-6)
    panels = [x.numpy() for x in (linear, quantized, thresholded)]
    assert all(p.shape == (2,96,96) for p in panels)
    np.savez_compressed(HERE/"data/image_formation_native.npz", linear=panels[0], encoded=encoded.numpy(),
                        quantized=panels[1], thresholded=panels[2], columns=np.array(obs.columns), angle_degrees=45.)
    fig, axes = plt.subplots(2,3,figsize=(7.2,4.3))
    fig.subplots_adjust(left=.13,right=.89,bottom=.07,top=.89,wspace=.10,hspace=.14)
    for i, column in enumerate(obs.columns):
        for j, data in enumerate(panels):
            axes[i,j].imshow(data[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[i,j].set(xticks=[],yticks=[])
            native = FIG/f"image_formation_{column}_{('linear','quantized','thresholded')[j]}_96.png"
            Image.fromarray(np.round(np.clip(data[i],0,1)*255).astype(np.uint8)).save(native)
            assert Image.open(native).size == (96,96)
            if i==0:
                axes[i,j].set_title(("Linear radiance $I$", "sRGB + 8-bit $Q$", r"Retained $Q\,1[Q>0.1]$")[j],fontsize=10)
            if j==0:
                axes[i,j].set_ylabel(column,fontsize=10)
    fig.colorbar(axes[0,0].images[0], cax=fig.add_axes([.91,.15,.018,.65]),ticks=[0,.5,1])
    fig.text(.52,.01,"Asteroid 2 · L4 · selected 200/200 · rotation 45° · distance = 5R · ξ = 0.5 · g = 0.5",ha="center",fontsize=8)
    save(fig,"image_formation")
    CHECKS["image_formation"] = dict(device="cpu", native_shape=[2,96,96], rotation_degrees=45,
        selected_iteration=traj2.best_index, columns=list(obs.columns), renderer=asdict(cfg),
        thresholded_sums=thresholded.sum((-2,-1)).tolist(), raw_intensity_matches=True,
        display_interpolation="nearest", shared_gray_scale=[0,1])
    print("Rendered native image demonstration",flush=True)

    raw = {}
    for a, expected in ((2,537950),(3,5996)):
        raw[a] = normalize_challenge_mesh(trimesh.load(source(public_mesh_path(a)),force="mesh",process=True))
        assert len(raw[a].faces) == expected
    proxy_path = SURVEY/"asteroid03/mesh_cache/asteroid03_star_proxy_1280_faces.npz"
    with np.load(source(proxy_path), allow_pickle=False) as archive:
        proxy = (archive["vertices"].copy(), archive["faces"].copy())
    assert len(proxy[1]) == 1280
    rebuilt = star_proxy_mesh(raw[3],1280)
    assert np.array_equal(rebuilt.faces,proxy[1])
    assert np.allclose(rebuilt.vertices,proxy[0],atol=2e-6,rtol=2e-6)
    views3 = [(25,135,"Oblique: 25°, 135°"),(0,90,"Equator: 0°, 90°")]
    CHECKS["proxy"] = mesh_grid([(raw[3].vertices,raw[3].faces),proxy],
        ["Raw reference · 5,996 faces","Radial proxy · 1,280 faces"],[GREEN,GREEN],views3,"star_proxy",size=(6.4,3.5))
    CHECKS["proxy"].update(asteroid=3, reference_faces=5996, proxy_faces=1280, outermost_ray_rebuilt=True)
    for a, fit, views in ((2,fit2,[(25,135,"Oblique: 25°, 135°"),(90,-90,"North: 90°, −90°")]),
                          (3,verify_fit(QUICK/"asteroid03/octantoid_l6"),views3)):
        traj = fit["trajectory"]
        scales = [fit["cmp"]["geometry"][state]["scale"] for state in ("initial","selected")]
        meshes = [(traj.vertices[index]*scale,traj.faces) for index,scale in zip((0,traj.best_index),scales)]
        meshes.append((raw[a].vertices,raw[a].faces))
        degree = fit["run"]["config"]["lmax"]
        budget = fit["run"]["config"]["steps"]
        CHECKS[f"reconstruction_{a}"] = mesh_grid(meshes,
            ["Initial sphere · 1,024 faces",f"L{degree} · selected {traj.best_index}/{budget}",f"Raw reference · {len(raw[a].faces):,} faces"],
            [GREY,BLUE,GREEN],views,f"reconstruction_asteroid{a}")
        CHECKS[f"reconstruction_{a}"].update(selected_iteration=traj.best_index, update_budget=budget,
            scale_alignment=scales, rotation_alignment="none", reference="raw public mesh")
        print(f"Rendered asteroid {a} reconstruction comparison",flush=True)

    fit6 = verify_fit(SURVEY/"asteroid06/octantoid_l4")
    traj = fit6["trajectory"]
    vertices, faces = traj.vertices[traj.best_index],traj.faces
    extent=1.1*np.linalg.norm(vertices,axis=1).max()
    views6 = [(25,135,"Oblique: 25°, 135°"),(90,-90,"North: 90°, −90°"),(25,-45,"Opposite: 25°, −45°")]
    fig = plt.figure(figsize=(7.2,4.2))
    grid = fig.add_gridspec(2,3,height_ratios=[1.1,1],left=.105,right=.97,bottom=.17,top=.87,hspace=.32,wspace=.30)
    for i,(e,a,label) in enumerate(views6):
        ax=fig.add_subplot(grid[0,i]); ax.imshow(mesh_image(vertices,faces,(e,a),extent,BLUE)); ax.set_axis_off(); ax.set_title(label,fontsize=9)
    curves=fit6["curves"]
    curve_checks={}
    for i,column in enumerate(("0_hor1","90_top","225_bottom")):
        j=list(curves["columns"]).index(column)
        ax=fig.add_subplot(grid[1,i])
        ax.plot(curves["indices"],curves["observed"][:,j],color="#222222",lw=1.1,label="Observed")
        ax.plot(curves["indices"],curves["fitted"][:,j],color="#297fb4",lw=1.1,ls="--",label="Selected fit")
        rmse=float(np.sqrt(np.mean((curves["observed"][:,j]-curves["fitted"][:,j])**2)))
        ax.set(xlim=(0,360),xticks=[0,180,360],xlabel="Rotation (°)",title=f"{column} · RMSE {rmse:.3f}")
        ax.title.set_fontsize(9); ax.grid(alpha=.2)
        ax.tick_params(labelsize=8)
        if i==0: ax.set_ylabel("Normalized intensity",fontsize=9)
        curve_checks[column]=rmse
    fig.legend(*fig.axes[3].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Asteroid 6 · L4 · selected 200/200 · calibration transferred from asteroid 3",fontsize=10,y=.995)
    save(fig,"asteroid6")
    CHECKS["asteroid6"]=dict(selected_iteration=traj.best_index, update_budget=200,
        calibration_transfer=True, reference_available=False, views=views6, representative_channel_rmse=curve_checks)

    # Preserve the exact mathematical and calibration sources underlying the draft.
    for name in ("shape_model.py","pytorch3d_forward.py","forward_validation.py","hac_asteroids/calibration.py",
                 "hac_asteroids/inversion.py","hac_asteroids/l4_survey.py","hac_asteroids/datasets.py",
                 "hac_asteroids/representation_reporting.py", "hac_asteroids/trajectory.py",
                 "reports/methods_overview/make_figure.py"):
        source(ROOT/name)
    calibration_rows = []
    for a in (1,2,3):
        cal = read_json(SURVEY/f"calibration/asteroid{a:02d}/calibration.json")
        distance = f"{cal['camera_distance_radii']:g}" if cal["projection"] == "perspective" else r"\textemdash"
        assert cal["response"] == "srgb" and cal["phase_convention"] == "zero_based"
        calibration_rows.append(f"{a} & {cal['projection'].title()} & {distance} & {cal['lambertian_fraction']:.2f} & {cal['radiance_gain']:.1f} & {cal['mean_rmse']:.6f} " + r"\\")
    (HERE/"tables/calibration_rows.tex").write_text("\n".join(calibration_rows)+"\n")
    source(ROOT/"docs/forward-model-correction-results.md")
    source(SURVEY/"frozen_configuration.json")
    for name in ("aa19267-12.pdf","viikinkoski_1358.pdf","aa26456-15.pdf"):
        source(ROOT/"docs/references"/name)
    # Catch concurrent changes to consumed inputs without touching experiment files.
    assert all(sha(ROOT/path)==digest for path,digest in SOURCES.items())
    outputs={str(p.relative_to(HERE)):sha(p) for folder in (FIG,HERE/"tables",HERE/"data")
             for p in sorted(folder.iterdir()) if p.is_file()}
    write_json(HERE/"evidence.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        note="Saved fit measurements only. Figure generation does not fit, resume, or modify experiments.",
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                         matplotlib=matplotlib.__version__,trimesh=trimesh.__version__),
        sha256=SOURCES,outputs_sha256=outputs,checks=CHECKS))
    print(f"Built figures and tables; audited {len(FITS)} saved fits.",flush=True)


if __name__ == "__main__":
    main()
