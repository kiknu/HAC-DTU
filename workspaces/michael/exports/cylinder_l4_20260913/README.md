# Cylinder-normalized asteroid reconstructions

These are resized copies of completed 200-iteration L4 fits. Each mesh is centered by its bounding box, scaled horizontally to the asteroid's specified cylinder radius, and independently scaled vertically to z=[-1,1]. No rotation is fitted. This removes width-to-height mismatch as well as absolute scale. Original optimization artifacts are preserved.

`initial.stl` is the prepared starting mesh; `selected.stl` minimizes the saved lightcurve validation error; `terminal.stl` is the last iteration. Selection can return iteration zero. These files are for display/evaluation; they are not replacements for the solver's saved coordinate system or restart checkpoints.

| Asteroid | Radius | Method | Initial | Selected (iteration) | Terminal |
|---|---:|---|---|---|---|
| 1 | 1.12 | sphere_l4 | [initial](asteroid01/sphere_l4/initial.stl) | [selected (140)](asteroid01/sphere_l4/selected.stl) | [terminal](asteroid01/sphere_l4/terminal.stl) |
| 1 | 1.12 | prior_l4 | [initial](asteroid01/prior_l4/initial.stl) | [selected (0)](asteroid01/prior_l4/selected.stl) | [terminal](asteroid01/prior_l4/terminal.stl) |
| 2 | 1.42 | sphere_l4 | [initial](asteroid02/sphere_l4/initial.stl) | [selected (200)](asteroid02/sphere_l4/selected.stl) | [terminal](asteroid02/sphere_l4/terminal.stl) |
| 2 | 1.42 | prior_l4 | [initial](asteroid02/prior_l4/initial.stl) | [selected (200)](asteroid02/prior_l4/selected.stl) | [terminal](asteroid02/prior_l4/terminal.stl) |
| 3 | 0.88 | sphere_l4 | [initial](asteroid03/sphere_l4/initial.stl) | [selected (200)](asteroid03/sphere_l4/selected.stl) | [terminal](asteroid03/sphere_l4/terminal.stl) |
| 3 | 0.88 | prior_l4 | [initial](asteroid03/prior_l4/initial.stl) | [selected (0)](asteroid03/prior_l4/selected.stl) | [terminal](asteroid03/prior_l4/terminal.stl) |
| 4 | 1.475 | sphere_l4 | [initial](asteroid04/sphere_l4/initial.stl) | [selected (100)](asteroid04/sphere_l4/selected.stl) | [terminal](asteroid04/sphere_l4/terminal.stl) |
| 5 | 1.22 | sphere_l4 | [initial](asteroid05/sphere_l4/initial.stl) | [selected (200)](asteroid05/sphere_l4/selected.stl) | [terminal](asteroid05/sphere_l4/terminal.stl) |
| 5 | 1.22 | prior_l4 | [initial](asteroid05/prior_l4/initial.stl) | [selected (0)](asteroid05/prior_l4/selected.stl) | [terminal](asteroid05/prior_l4/terminal.stl) |
| 6 | 0.925 | sphere_l4 | [initial](asteroid06/sphere_l4/initial.stl) | [selected (200)](asteroid06/sphere_l4/selected.stl) | [terminal](asteroid06/sphere_l4/terminal.stl) |
| 7 | 1.205 | sphere_l4 | [initial](asteroid07/sphere_l4/initial.stl) | [selected (60)](asteroid07/sphere_l4/selected.stl) | [terminal](asteroid07/sphere_l4/terminal.stl) |
| 8 | 1.24 | sphere_l4 | [initial](asteroid08/sphere_l4/initial.stl) | [selected (0)](asteroid08/sphere_l4/selected.stl) | [terminal](asteroid08/sphere_l4/terminal.stl) |
| 8 | 1.24 | prior_l4 | [initial](asteroid08/prior_l4/initial.stl) | [selected (40)](asteroid08/prior_l4/selected.stl) | [terminal](asteroid08/prior_l4/terminal.stl) |
| 9 | 0.67 | sphere_l4 | [initial](asteroid09/sphere_l4/initial.stl) | [selected (60)](asteroid09/sphere_l4/selected.stl) | [terminal](asteroid09/sphere_l4/terminal.stl) |
| 9 | 0.67 | prior_l4 | [initial](asteroid09/prior_l4/initial.stl) | [selected (40)](asteroid09/prior_l4/selected.stl) | [terminal](asteroid09/prior_l4/terminal.stl) |
| 10 | 3.95 | sphere_l4 | [initial](asteroid10/sphere_l4/initial.stl) | [selected (60)](asteroid10/sphere_l4/selected.stl) | [terminal](asteroid10/sphere_l4/terminal.stl) |
| 10 | 3.95 | prior_l4 | [initial](asteroid10/prior_l4/initial.stl) | [selected (0)](asteroid10/prior_l4/selected.stl) | [terminal](asteroid10/prior_l4/terminal.stl) |

## Availability and checks

Exported 51 STLs. Each serialized STL was reloaded and checked for radius, height, face count, watertightness, winding, degenerate faces, and positive volume. Meshes with at most 4,096 faces also underwent the transverse intersection check (coplanar contacts excluded).

Completed prior fits for asteroids 1–3 and 8 had separate lightcurve-report tolerance failures. Their complete saved trajectories supply these exports; this bundle does not repair the photometric reports.

[Manifest, transforms and SHA-256 hashes](manifest.json)

- Asteroid 4, prior_l4: No completed run/trajectory available.
- Asteroid 6, prior_l4: No completed run/trajectory available.
- Asteroid 7, prior_l4: No completed run/trajectory available.

[Asteroid 1–3 voxel accuracy and normalization comparison](evaluation/README.md) · [Usage documentation](../../docs/voxel-evaluation.md)
