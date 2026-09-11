import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import trimesh

from hac.lc2mesh.metrics import mesh_reconstruction_metrics
from hac.lc2mesh.model import INR, LightcurveToMesh, MeshGCN, ResidualMeshGCN
from hac.lc2mesh.training import (
    TrainingConfig,
    displacement_laplacian,
    split_camera_indices,
    train_stage,
)
from workspaces.ishak.benchmark_inr import summarize_experiments


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.faces = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
        self.vertices = torch.tensor(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        )
        self.model = LightcurveToMesh(
            self.faces,
            inr_hidden_dim=16,
            inr_num_layers=3,
            gcn_hidden_dim=16,
            gcn_num_layers=3,
            R=1.42,
        )

    def test_cylinder_batch_independence_and_gradients(self):
        batch = torch.stack([self.vertices, self.vertices * 2 + 3]).requires_grad_()
        normalized = self.model.canonicalize_to_cylinder(batch)
        expected = torch.stack(
            [self.model.canonicalize_to_cylinder(mesh) for mesh in batch]
        )
        torch.testing.assert_close(normalized, expected)
        torch.testing.assert_close(normalized[..., 2].amax(-1), torch.ones(2))
        torch.testing.assert_close(normalized[..., 2].amin(-1), -torch.ones(2))
        torch.testing.assert_close(
            normalized[..., :2].norm(dim=-1).amax(-1), torch.full((2,), 1.42)
        )
        (normalized * torch.randn_like(normalized)).sum().backward()
        self.assertTrue(torch.isfinite(batch.grad).all())
        self.assertGreater(batch.grad.norm().item(), 0)

    def test_forward_preserves_batch_and_single_mesh_shapes(self):
        for vertices in (self.vertices, self.vertices.unsqueeze(0).expand(2, -1, -1)):
            coarse, refined = self.model(vertices)
            self.assertEqual(coarse.shape, vertices.shape)
            self.assertEqual(refined.shape, vertices.shape)

    def test_inr_identity_initialization(self):
        model = INR(inr_hidden_dim=16, inr_num_layers=3, final_init_scale=0)
        vertices, _ = model(self.vertices)
        torch.testing.assert_close(vertices, self.vertices)

    def test_residual_refiner_permutation_and_batch_equivariance(self):
        model = ResidualMeshGCN(self.faces, 4, hidden_dim=16, num_layers=2)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        permuted = ResidualMeshGCN(inverse[self.faces], 4, hidden_dim=16, num_layers=2)
        permuted.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            permuted(self.vertices[permutation]), model(self.vertices)[permutation]
        )
        batch = torch.stack([self.vertices, self.vertices * 2])
        torch.testing.assert_close(
            model(batch), torch.stack([model(mesh) for mesh in batch])
        )
        model(batch).square().sum().backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_residual_refiner_is_smaller(self):
        plain = MeshGCN(self.faces, 4)
        residual = ResidualMeshGCN(self.faces, 4)
        self.assertLess(
            sum(parameter.numel() for parameter in residual.parameters()),
            sum(parameter.numel() for parameter in plain.parameters()),
        )

    def test_shared_voxel_grid_detects_translation(self):
        target = trimesh.creation.box()
        shifted = target.copy().apply_translation([3, 0, 0])
        identical = mesh_reconstruction_metrics(
            target, target, n_surface_samples=128, pitch=0.2
        )
        translated = mesh_reconstruction_metrics(
            shifted, target, n_surface_samples=128, pitch=0.2
        )
        self.assertEqual(identical["solid_dice"], 1)
        self.assertEqual(translated["solid_dice"], 0)
        self.assertGreater(translated["surface_chamfer"], 1)

    def test_split_is_reproducible_disjoint_and_complete(self):
        train, validation = split_camera_indices(28)
        other_train, other_validation = split_camera_indices(28)
        torch.testing.assert_close(train, other_train)
        torch.testing.assert_close(validation, other_validation)
        self.assertFalse(torch.isin(train, validation).any())
        self.assertEqual(len(train) + len(validation), 28)

    def test_displacement_laplacian_ignores_constant_translation(self):
        constant = torch.ones(1, 4, 3)
        self.assertLess(
            displacement_laplacian(constant, self.model.gcn.adjacency).item(), 1e-12
        )

    def test_restored_checkpoint_matches_recorded_validation(self):
        base = self.vertices.unsqueeze(0)
        observed = self.vertices.T * 1.1
        result = train_stage(
            self.model,
            base,
            None,
            lambda vertices: vertices.T,
            observed,
            None,
            torch.tensor([0, 1]),
            torch.tensor([2]),
            TrainingConfig(steps=4, eval_every=1, hull_weight=0, canonicalize=False),
            stage=1,
        )
        with torch.no_grad():
            vertices, _ = self.model.inr(base)
            restored_score = (vertices[0].T[2] - observed[2]).square().mean().item()
        self.assertAlmostEqual(restored_score, result.best_score, places=6)

    def test_validation_cameras_do_not_change_optimization_gradients(self):
        other_model = LightcurveToMesh(
            self.faces,
            inr_hidden_dim=16,
            inr_num_layers=3,
            gcn_hidden_dim=16,
            gcn_num_layers=3,
            R=1.42,
        )
        other_model.load_state_dict(self.model.state_dict())
        observed = self.vertices.T.clone()
        modified = observed.clone()
        modified[2] += 100
        config = TrainingConfig(
            steps=2,
            eval_every=2,
            hull_weight=0,
            canonicalize=False,
            restore_best=False,
            scheduler_patience=0,
        )
        for model, target in ((self.model, observed), (other_model, modified)):
            train_stage(
                model,
                self.vertices.unsqueeze(0),
                None,
                lambda vertices: vertices.T,
                target,
                None,
                torch.tensor([0, 1]),
                torch.tensor([2]),
                config,
                stage=1,
            )
        for name, parameter in self.model.state_dict().items():
            torch.testing.assert_close(parameter, other_model.state_dict()[name])

    def test_stage2_tether_penalizes_inr_drift_without_graph_displacement(self):
        self.model.canonicalize_coarse = True
        with patch.object(self.model.gcn, "forward", side_effect=torch.zeros_like):
            result = train_stage(
                self.model,
                self.vertices.unsqueeze(0),
                None,
                lambda vertices: vertices.T,
                self.vertices.T * 1.1,
                None,
                torch.tensor([0, 1]),
                torch.tensor([2]),
                TrainingConfig(
                    steps=3, eval_every=1, hull_weight=0, displacement_weight=0.1
                ),
                stage=2,
            )
            self.assertLess(result.history["displacement"][0], 1e-12)
            self.assertGreater(result.history["displacement"][-1], 0)
            with torch.no_grad():
                _, vertices = self.model(self.vertices.unsqueeze(0))
                score = (vertices[0].T[2] - self.vertices.T[2] * 1.1).square().mean()
            self.assertAlmostEqual(float(score), result.best_score, places=6)

    def test_summary_statistics_and_protocol_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for seed, chamfer in ((41, 1.0), (42, 3.0)):
                result = {
                    "metadata": {
                        "asteroid_id": 1,
                        "recipe": "training",
                        "seed": seed,
                        "subdivisions": 3,
                        "hull_samples": 4096,
                        "training_config_stage1": {"steps": 600},
                        "training_config_stage2": {"steps": 600},
                        "split_seed": 1729,
                        "eval_samples": 8192,
                        "eval_seed": 1729,
                        "voxel_pitch": 0.04,
                    },
                    "metrics": {
                        "refined": {"surface_chamfer": chamfer, "solid_dice": 0.8},
                        "convex_prior": {},
                    },
                    "validation_mse": 0.01,
                    "training_seconds": 1,
                    "parameters": 100,
                }
                path = folder / f"ast1_seed{seed}_training.json"
                path.write_text(json.dumps(result))
            summary = summarize_experiments(folder)[0]
            self.assertEqual(summary["surface_chamfer"]["mean"], 2.0)
            self.assertAlmostEqual(summary["surface_chamfer"]["std"], 2**0.5)
            self.assertEqual(summary["seeds"], [41, 42])
            result["metadata"]["subdivisions"] = 4
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "one resolution"):
                summarize_experiments(folder)


if __name__ == "__main__":
    unittest.main()
