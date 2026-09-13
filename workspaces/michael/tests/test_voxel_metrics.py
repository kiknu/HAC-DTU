"""Geometric overlap and saved-history regression checks (CPU only)."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from hac_asteroids.trajectory import FitTrajectory
from hac_asteroids.voxel_metrics import CHALLENGE_RADII, normalize_mesh, occupied_voxels, overlap_metrics, solid_voxel_metrics
from hac_asteroids.voxel_history import checkpoint_indices, score_histories, verify_stl


class VoxelMetricTests(unittest.TestCase):
    def test_identity_disjoint_translation_and_symmetry(self):
        a = trimesh.creation.box()
        b = a.copy()
        b.apply_translation([2, 0, 0])
        for left, right, expected in ((a, a, 1.0), (a, b, 0.0), (b, a, 0.0)):
            result = solid_voxel_metrics(left, right, pitch=.1)
            self.assertTrue(result["valid"])
            self.assertEqual(result["solid_dice"], expected)
            self.assertEqual(result["solid_iou"], expected)

    def test_partial_overlap_on_different_local_origins(self):
        a = trimesh.creation.box()
        b = a.copy()
        b.apply_translation([.5, 0, 0])
        va, vb = occupied_voxels(a, .1), occupied_voxels(b, .1)
        ab, ba = overlap_metrics(va, vb), overlap_metrics(vb, va)
        # Inclusive surface voxelization gives 11 columns each, six shared.
        self.assertAlmostEqual(ab["solid_iou"], 6 / 16)
        self.assertAlmostEqual(ab["solid_dice"], 6 / 11)
        self.assertEqual(ab["solid_iou"], ba["solid_iou"])
        self.assertAlmostEqual(ab["solid_dice"], 2 * ab["solid_iou"] / (1 + ab["solid_iou"]))

    def test_normalization_preserves_aspect_and_orientation(self):
        mesh = trimesh.creation.box(extents=[2, 3, 4])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.3, [0, 0, 1]))
        mesh.apply_translation([4, 6, 8])
        normalized, transform = normalize_mesh(mesh)
        np.testing.assert_allclose(normalized.bounds[:, 2], [-1, 1])
        np.testing.assert_allclose(normalized.vertices,
                                   (mesh.vertices - transform["center"]) * transform["scale"])
        np.testing.assert_allclose(normalized.edges_unique_length / mesh.edges_unique_length, transform["scale"])
        np.testing.assert_allclose(normalized.face_normals, mesh.face_normals, atol=1e-12)

    def test_invalid_geometry_and_resource_limits(self):
        a = trimesh.creation.box()
        opened = trimesh.Trimesh(vertices=a.vertices, faces=a.faces[:-1], process=False)
        result = solid_voxel_metrics(opened, a, pitch=.1)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["solid_dice"])
        crossing = trimesh.util.concatenate([a, a.copy().apply_translation([.2, .2, .2])])
        result = solid_voxel_metrics(crossing, a, pitch=.1)
        self.assertFalse(result["valid"])
        for pitch in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                occupied_voxels(a, pitch)
        with self.assertRaisesRegex(ValueError, "cell budget"):
            occupied_voxels(a, .001, max_grid_cells=100)
        with self.assertRaises(ValueError):
            overlap_metrics(np.empty((0, 3), dtype=np.int64), np.zeros((1, 3), dtype=np.int64))

    def test_cylinder_normalization_removes_radial_axial_scale_only(self):
        original = trimesh.creation.box(extents=[2, 3, 4])
        original.apply_transform(trimesh.transformations.rotation_matrix(.3, [0, 0, 1]))
        altered = original.copy()
        altered.vertices *= [2, 2, .7]
        altered.apply_translation([4, -5, 6])
        saved = altered.vertices.copy()
        for radius in CHALLENGE_RADII.values():
            a, _ = normalize_mesh(original, target_radius=radius)
            b, transform = normalize_mesh(altered, target_radius=radius)
            np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-14)
            np.testing.assert_allclose(b.bounds[:, 2], [-1, 1])
            self.assertAlmostEqual(np.linalg.norm(b.vertices[:, :2], axis=1).max(), radius)
            self.assertNotEqual(transform["scale_xy"], transform["scale_z"])
            self.assertEqual(solid_voxel_metrics(a, b, pitch=.1)["solid_dice"], 1)
        np.testing.assert_array_equal(altered.vertices, saved)
        for radius in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                normalize_mesh(original, target_radius=radius)

    def test_checkpoint_indices(self):
        self.assertEqual(checkpoint_indices(12, 5, 7), [0, 5, 7, 10, 11])
        with self.assertRaises(ValueError):
            checkpoint_indices(12, 0, 7)

    def test_history_export_resume_and_input_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = trimesh.creation.box()
            reference = root / "data/AsteroidModel01_shape_public/asteroid1.stl"
            reference.parent.mkdir(parents=True)
            mesh.export(reference)
            vertices = np.stack([mesh.vertices, mesh.vertices * [1.1, 1, 1], mesh.vertices * [1.2, 1, 1]])
            for name, variant in (("prior", "prior_l4"), ("baseline", "octantoid_l4")):
                run = root / name / "asteroid01" / variant
                run.mkdir(parents=True)
                FitTrajectory(np.zeros((3, 1)), vertices, mesh.faces, np.zeros(3), 1).save(run / "trajectory.npz")
                (run / "run.json").write_text(json.dumps(dict(summary=dict(best_iteration=1, iterations_run=2, status="complete"))))
                mesh.export(run / "mesh_initial.stl")
                trimesh.Trimesh(vertices=vertices[1], faces=mesh.faces, process=False).export(run / "mesh_final.stl")
            args = argparse.Namespace(output_dir=root / "out", data_root=root / "data", prior_campaign=root / "prior",
                                      baseline_campaign=root / "baseline", asteroids=[1], pitches=[.1, .2], stride=2)
            rows = score_histories(args)
            self.assertEqual(len(rows), 12)
            self.assertEqual({r["iteration"] for r in rows}, {0, 1, 2})
            self.assertTrue(all(r["iteration"] == 1 for r in rows if r["selected"]))
            with patch("hac_asteroids.voxel_history.occupied_voxels", side_effect=AssertionError("cache missed")):
                score_histories(args)
            # Changing a source run invalidates its measurements, not the other run.
            changed = root / "prior/asteroid01/prior_l4/run.json"
            data = json.loads(changed.read_text())
            data["audit_note"] = "changed"
            changed.write_text(json.dumps(data))
            with patch("hac_asteroids.voxel_history.occupied_voxels", wraps=occupied_voxels) as voxelize:
                score_histories(args)
                self.assertEqual(voxelize.call_count, 6)
            stl = root / "out/asteroid01/prior_l4/checkpoints/iteration_0001.stl"
            with self.assertRaisesRegex(ValueError, "disagrees"):
                verify_stl(stl, vertices[0], mesh.faces)
            self.assertTrue((root / "out/accuracy_comparison.pdf").is_file())
            self.assertEqual(json.loads((root / "out/status.json").read_text())["state"], "complete")
            # A normalization switch must invalidate both reference and state caches.
            shutil.copytree(root / "out", root / "previous")
            args.normalization = "cylinder"
            args.compare_with = root / "previous"
            with patch("hac_asteroids.voxel_history.occupied_voxels", wraps=occupied_voxels) as voxelize:
                cylinder_rows = score_histories(args)
                self.assertEqual(voxelize.call_count, 14)
            self.assertTrue(all(r["target_radius"] == CHALLENGE_RADII[1] for r in cylinder_rows))
            self.assertTrue((root / "out/normalization_comparison.pdf").is_file())
            self.assertIn("width-to-height mismatch", (root / "out/report.md").read_text())
            with patch("hac_asteroids.voxel_history.occupied_voxels", side_effect=AssertionError("cache missed")):
                score_histories(args)


if __name__ == "__main__":
    unittest.main()
