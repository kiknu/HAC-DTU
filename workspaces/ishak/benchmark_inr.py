"""Controlled neural reconstruction ablations; the convex forward code is fixed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import trimesh

from hac.data import get_model_stl_path
from hac.lc2mesh.data import create_ellipsoidal_prior_mesh
from hac.lc2mesh.dataset import load_asteroid_data
from hac.lc2mesh.forward_model import AsteroidForwardModel
from hac.lc2mesh.metrics import compute_pearson_r, mesh_reconstruction_metrics
from hac.lc2mesh.model import LightcurveToMesh
from hac.lc2mesh.training import TrainingConfig, split_camera_indices, train_stage
from hac.lc2mesh.utils import (
    sample_convex_hull_surface,
    set_seed,
    vertex_normals_from_faces,
)
from hac.mesh import normalize_mesh_to_challenge_cylinder


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "src/hac/lc2mesh/model.py",
    "src/hac/lc2mesh/training.py",
    "src/hac/lc2mesh/forward_model.py",
    "src/hac/convex_forward.py",
    "src/hac/lc2mesh/dataset.py",
    "src/hac/lc2mesh/data.py",
    "src/hac/lc2mesh/metrics.py",
    "src/hac/lc2mesh/utils.py",
    "src/hac/data.py",
    "src/hac/mesh.py",
    "workspaces/ishak/benchmark_inr.py",
)
SOURCE_SHA256 = {
    path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for path in SOURCE_PATHS
}


def summarize_experiments(directory: Path) -> list[dict]:
    """Aggregate seed experiments without silently mixing protocols."""
    paths = sorted(directory.glob("ast*_seed*_*.json"))
    if not paths:
        raise ValueError(f"No benchmark results in {directory}")
    groups = {}
    protocols = set()
    for path in paths:
        result = json.loads(path.read_text())
        metadata = result["metadata"]
        protocols.add(
            (
                metadata["subdivisions"],
                metadata["hull_samples"],
                metadata["training_config_stage1"]["steps"],
                metadata["training_config_stage2"]["steps"],
                metadata["split_seed"],
                metadata["eval_samples"],
                metadata["eval_seed"],
                metadata["voxel_pitch"],
            )
        )
        key = (metadata["asteroid_id"], metadata["recipe"])
        groups.setdefault(key, []).append(result)
    if len(protocols) != 1:
        raise ValueError(
            "Summarize one resolution, training budget, split, and evaluation protocol at a time"
        )

    summaries = []
    for (asteroid_id, recipe), results in sorted(groups.items()):
        seeds = [result["metadata"]["seed"] for result in results]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"Duplicate seeds for asteroid {asteroid_id}, recipe {recipe}"
            )
        summary = {
            "asteroid_id": asteroid_id,
            "recipe": recipe,
            "seeds": sorted(seeds),
            "runs": len(results),
        }
        values = {
            "surface_chamfer": [
                result["metrics"]["refined"]["surface_chamfer"] for result in results
            ],
            "solid_dice": [
                result["metrics"]["refined"]["solid_dice"] for result in results
            ],
            "validation_mse": [result["validation_mse"] for result in results],
            "training_seconds": [result["training_seconds"] for result in results],
        }
        for metric, samples in values.items():
            summary[metric] = {
                "mean": statistics.mean(samples),
                "std": statistics.stdev(samples) if len(samples) > 1 else None,
            }
        summary["convex_prior"] = results[0]["metrics"]["convex_prior"]
        summary["parameters"] = results[0]["parameters"]
        summaries.append(summary)
    return summaries


def model_config(recipe: str, radius: float) -> dict:
    return {
        "inr_hidden_dim": 128,
        "inr_num_layers": 4,
        "coarse_w0": 30.0,
        "gcn_hidden_dim": 64 if recipe == "residual" else 128,
        "gcn_num_layers": 4 if recipe == "residual" else 6,
        "gcn_architecture": "residual" if recipe == "residual" else "plain",
        "inr_final_init_scale": 1.0 if recipe == "baseline" else 1e-3,
        "canonicalize_coarse": recipe != "baseline",
        "R": radius,
    }


def training_configs(
    recipe: str, steps: int, eval_every: int
) -> tuple[TrainingConfig, TrainingConfig]:
    if recipe == "baseline":
        common = dict(
            steps=steps,
            eval_every=eval_every,
            scheduler_patience=0,
            canonicalize=False,
            laplacian_weight=0,
            selection="training",
        )
        return (
            TrainingConfig(**common, restore_best=False),
            TrainingConfig(**common, lr_inr=1e-6, hull_weight=0),
        )
    common = dict(steps=steps, eval_every=eval_every)
    return (
        TrainingConfig(**common),
        TrainingConfig(**common, lr_inr=1e-6, hull_weight=0.1, displacement_weight=0.1),
    )


def run_experiment(args, asteroid_id: int, seed: int, recipe: str) -> dict:
    stem = f"ast{asteroid_id}_seed{seed}_{recipe}"
    result_path = args.output / f"{stem}.json"
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {result_path}; choose another output directory"
        )
    set_seed(seed)
    device = torch.device(args.device)
    data = load_asteroid_data(asteroid_id, device=device)
    prior = create_ellipsoidal_prior_mesh(
        asteroid_id=asteroid_id, subdivisions=args.subdivisions, device=device
    )
    base, faces = prior["prior_vertices_batch"], prior["prior_faces_t"]
    normals = vertex_normals_from_faces(base, faces)
    config = model_config(recipe, prior["R_CYLINDER"])
    net = LightcurveToMesh(faces, **config).to(device)
    reference_path = (
        ROOT
        / "workspaces/ishak/convex_inversions"
        / f"asteroid{asteroid_id}_simulated_intensity.stl"
    )
    reference = normalize_mesh_to_challenge_cylinder(
        trimesh.load_mesh(reference_path), prior["R_CYLINDER"]
    )
    hull = torch.as_tensor(
        sample_convex_hull_surface(
            reference.vertices, n_points=args.hull_samples, seed=seed
        ),
        device=device,
        dtype=base.dtype,
    ).unsqueeze(0)
    train_indices, validation_indices = split_camera_indices(
        data.n_cameras, seed=args.split_seed, device=device
    )

    def predict(vertices):
        triangles = vertices[faces]
        cross = torch.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1
        )
        twice_area = cross.norm(dim=1)
        facet_normals = cross / twice_area.clamp_min(1e-12).unsqueeze(1)
        forward = AsteroidForwardModel(normals=facet_normals, lambert_weight=0.5)
        matrix = forward.build_forward_matrix(
            phases=data.phases, e_sun=data.e_sun, e_obs=data.e_obs
        )
        raw = torch.einsum("cpf,f->cp", matrix, 0.5 * twice_area)
        return raw / raw.mean(dim=1, keepdim=True).clamp_min(1e-12)

    def progress(row):
        if (
            row["step"] == 0
            or row["step"] % args.print_every == 0
            or row["step"] == args.steps
        ):
            print(
                f"{stem}: step={row['step']} train={row['train_mse']:.6f} validation={row['validation_mse']:.6f}",
                flush=True,
            )

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_config, second_config = training_configs(recipe, args.steps, args.eval_every)
    first = train_stage(
        net,
        base,
        normals,
        predict,
        data.observed_norm,
        hull,
        train_indices,
        validation_indices,
        first_config,
        stage=1,
        progress=progress,
    )
    with torch.no_grad():
        coarse, _ = net.inr(base, normals)
        coarse = net.canonicalize_to_cylinder(coarse)
    second = train_stage(
        net,
        base,
        normals,
        predict,
        data.observed_norm,
        hull,
        train_indices,
        validation_indices,
        second_config,
        stage=2,
        progress=progress,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    with torch.no_grad():
        _, refined = net(base, normals)
        prediction = predict(refined[0])
        camera_mse = (prediction - data.observed_norm).square().mean(dim=1)
        camera_r = compute_pearson_r(data.observed_norm, prediction)

    true_path = get_model_stl_path(asteroid_id)
    true_mesh = normalize_mesh_to_challenge_cylinder(
        trimesh.load_mesh(true_path), prior["R_CYLINDER"]
    )
    faces_np = faces.cpu().numpy()
    metrics = {}
    for label, vertices in (
        ("initial", base),
        ("stage1", coarse),
        ("refined", refined),
    ):
        mesh = trimesh.Trimesh(
            vertices=vertices[0].detach().cpu().numpy(), faces=faces_np, process=False
        )
        metrics[label] = mesh_reconstruction_metrics(
            mesh, true_mesh, n_surface_samples=args.eval_samples, pitch=args.pitch
        )
        if label == "refined":
            mesh.export(args.output / f"{stem}.stl")
    metrics["convex_prior"] = mesh_reconstruction_metrics(
        reference, true_mesh, n_surface_samples=args.eval_samples, pitch=args.pitch
    )
    metadata = {
        "asteroid_id": asteroid_id,
        "seed": seed,
        "recipe": recipe,
        "subdivisions": args.subdivisions,
        "hull_samples": args.hull_samples,
        "lambert_weight": 0.5,
        "phase_alignment": False,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "split_seed": args.split_seed,
        "train_cameras": [data.camera_names[index] for index in train_indices.tolist()],
        "validation_cameras": [
            data.camera_names[index] for index in validation_indices.tolist()
        ],
        "validation_scope": "conditional on external simulated-lightcurve inversion prior; not a clean end-to-end holdout",
        "reference_path": str(reference_path.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "observation_sha256": hashlib.sha256(
            data.observed.cpu().numpy().tobytes()
        ).hexdigest(),
        "source_sha256": SOURCE_SHA256,
        "eval_samples": args.eval_samples,
        "eval_seed": 1729,
        "voxel_pitch": args.pitch,
        "model_config": config,
        "training_config_stage1": asdict(first_config),
        "training_config_stage2": asdict(second_config),
    }
    result = {
        "metadata": metadata,
        "metrics": metrics,
        "train_mse": float(camera_mse[train_indices].mean()),
        "validation_mse": float(camera_mse[validation_indices].mean()),
        "all_camera_mse": float(camera_mse.mean()),
        "mean_r": float(camera_r.mean()),
        "min_r": float(camera_r.min()),
        "per_camera_mse": camera_mse.cpu().tolist(),
        "training_seconds": training_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None,
        "parameters": sum(parameter.numel() for parameter in net.parameters()),
        "stage1": asdict(first),
        "stage2": asdict(second),
    }
    state = {name: value.detach().cpu() for name, value in net.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "model_state": state,
            "config": metadata,
            "faces": faces.cpu(),
            "base_vertices": base.cpu(),
        },
        args.output / f"{stem}.pth",
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run": stem,
                "validation_mse": result["validation_mse"],
                "shape": metrics["refined"],
                "seconds": training_seconds,
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asteroids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument(
        "--recipes",
        nargs="+",
        choices=["baseline", "training", "residual"],
        default=["baseline", "training", "residual"],
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--hull-samples", type=int, default=4096)
    parser.add_argument("--eval-samples", type=int, default=8192)
    parser.add_argument("--pitch", type=float, default=0.04)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summarize", type=Path, help="Summarize saved runs instead of training"
    )
    args = parser.parse_args()
    if args.summarize is not None:
        print(json.dumps(summarize_experiments(args.summarize), indent=2))
        return
    if args.output is None:
        parser.error("--output is required when running experiments")
    if args.steps < 0 or args.subdivisions < 1 or args.print_every < 1:
        parser.error(
            "steps must be nonnegative; subdivisions and print-every must be positive"
        )
    if not set(args.asteroids) <= {1, 2, 3}:
        parser.error("Shape benchmarks require public targets 1, 2, or 3")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    for asteroid_id in args.asteroids:
        for seed in args.seeds:
            for recipe in args.recipes:
                run_experiment(args, asteroid_id, seed, recipe)


if __name__ == "__main__":
    main()
