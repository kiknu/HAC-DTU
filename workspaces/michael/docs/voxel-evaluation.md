# Voxel evaluation and normalized STL exports

The solid Dice and IoU implementation compares filled voxels on a shared
world-coordinate lattice. It avoids comparing indices from independent local
voxel origins. Metrics range from 0 (disjoint) to 1 (identical occupancy).

`hac_asteroids.voxel_metrics.normalize_mesh` supports two explicit conventions:

- Uniform: center the bounding box and apply one scale making z=[-1,1]. This
  preserves width-to-height mismatch in the score.
- Cylinder: center the bounding box, scale xy to the model's bounding radius,
  and independently scale z to [-1,1]. This removes width-to-height mismatch.
  No rotation is fitted. Radii follow `origin/inr`'s `MODEL_BASE_RADII`.

Run from `workspaces/michael` with the project's Python environment:

```bash
python -m hac_asteroids.voxel_history --normalization cylinder --output-dir artifacts/voxel_cylinder_a1_a3_20260913 --compare-with artifacts/voxel_history_a1_a3_20260913
python -m hac_asteroids.export_normalized --output-dir exports/cylinder_l4_20260913
python -m unittest tests.test_voxel_metrics tests.test_normalized_exports -q
```

Scoring uses the original public references for asteroids 1–3 and saved fit
trajectories at five-iteration intervals, with pitches 0.02 and 0.04. Reference
geometry never selects a fit checkpoint. Source datasets and optimizer
artifacts remain local and are required to recompute; the repository includes
the exported meshes and evaluation tables/plots.

The default exporter selects completed L4 sphere-start and neural-prior runs
for asteroids 1–10. It writes initial, lightcurve-selected, and terminal meshes
as normalized copies, preserving solver artifacts. Missing/incomplete runs
are listed in the manifest, not represented as completed results. Override
campaign directories or `--asteroids` to export another matching campaign.

Every exported STL is reloaded to check its radius, height, face count,
watertightness, winding, degeneracy, and positive volume. Transverse
self-intersections are checked for meshes with at most 4,096 faces; coplanar
contacts are excluded by that check. Evaluation of the large public reference
meshes checks topology but does not exhaustively check self-intersections.

[Normalized STL bundle](../exports/cylinder_l4_20260913/README.md) ·
[Voxel evaluation results](../exports/cylinder_l4_20260913/evaluation/README.md)
