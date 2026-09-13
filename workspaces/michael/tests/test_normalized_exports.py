"""Check serialized cylinder exports, provenance, and unavailable runs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from hac_asteroids.export_normalized import export_mesh, export_campaigns
from hac_asteroids.trajectory import FitTrajectory
from hac_asteroids.voxel_metrics import CHALLENGE_RADII


class ExportTests(unittest.TestCase):
    def test_all_radii_and_source_preservation(self):
        mesh = trimesh.creation.box(extents=[2, 3, 4])
        mesh.apply_translation([2, -3, 5])
        before = mesh.vertices.copy()
        with tempfile.TemporaryDirectory() as directory:
            for asteroid, radius in CHALLENGE_RADII.items():
                audit = export_mesh(mesh, Path(directory) / f"{asteroid}.stl", asteroid)
                self.assertAlmostEqual(audit["actual_xy_radius"], radius, places=6)
                self.assertEqual(audit["z_bounds"], [-1, 1])
                self.assertTrue(audit["validity"]["valid"])
        np.testing.assert_array_equal(mesh.vertices, before)

    def test_selected_and_terminal_are_distinct_and_missing_run_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "campaign/asteroid01/model"
            run.mkdir(parents=True)
            mesh = trimesh.creation.box()
            vertices = np.stack([mesh.vertices, mesh.vertices * [1.2, 1, 1]])
            FitTrajectory(np.zeros((2, 1)), vertices, mesh.faces, np.zeros(2), 0).save(run / "trajectory.npz")
            (run / "run.json").write_text(json.dumps(dict(summary=dict(status="complete", iterations_run=1, best_iteration=0))))
            rows, skipped = export_campaigns(root / "out", [("test", root / "campaign", "model")], [1, 2])
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["iteration"] for r in rows], [0, 0, 1])
            self.assertNotEqual(rows[1]["sha256"], rows[2]["sha256"])
            self.assertEqual(skipped[0]["asteroid"], 2)
            self.assertTrue((root / "out/manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
