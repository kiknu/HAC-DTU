"""
Meshing utilities using `trimesh`.

We need conversion from STL to OBJ if we use Mitsuba.
"""

from pathlib import Path

import trimesh
import numpy as np


def convert_stl_to_obj(stl_path: Path, obj_path: Path | None = None) -> Path:
    """
    Converts an STL file to an OBJ file.

    If `obj_path` is not provided, the OBJ file will be saved in the same directory as the STL file with the same name but with an `.obj` extension.
    """
    if obj_path is None:
        obj_path = stl_path.with_suffix(".obj")

    mesh = trimesh.load(stl_path, force="mesh")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text(trimesh.exchange.obj.export_obj(mesh))

    return obj_path


def construct_initial_ellipsoid_mesh(
    radius: float,
    subdivisions: int = 5,
) -> trimesh.Trimesh:
    """
    Construct a mesh with an initial ellipsoid guess.

    Arguments:
    - radius: The radius of the ellipsoid.
    - subdivisions: The number of subdivisions for the icosphere.
        Higher values result in a more detailed mesh.
        The default level of 5 gives a very detailed mesh that should probably be simplified
        with `Trimesh.simplify_quadratic_decimation`
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    mesh = mesh.apply_scale([radius, radius, 1.0])

    return mesh


def normalize_mesh_to_challenge_cylinder(
    mesh,
    target_R,
    *,
    center_xy="bounds",
    copy=True,
    verbose=False,
):
    """
    Normalize a mesh into the challenge cylinder D(0,R) x [-1,1].

    After transformation, the output mesh approximately satisfies:
        min(z) = -1
        max(z) = +1
        max(sqrt(x^2 + y^2)) = target_R

    Parameters
    ----------
    mesh:
        trimesh.Trimesh object.

    target_R:
        Challenge cylinder base radius R for the selected model.

    center_xy:
        Controls how the x/y origin is chosen before radial scaling.

        "bounds":
            Center x/y using the midpoint of the x/y bounding box.

        "centroid":
            Center x/y using mesh.centroid[:2].

        None:
            Do not recenter x/y before scaling.

    copy:
        If True, transform and return a copy.
        If False, modify the input mesh in place.

    verbose:
        Print before/after normalization diagnostics.
    """

    if target_R <= 0:
        raise ValueError(f"target_R must be positive. Got {target_R}.")

    mesh_out = mesh.copy() if copy else mesh

    vertices = np.asarray(mesh_out.vertices, dtype=float)

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    z_span = z_max - z_min

    if z_span <= 0:
        raise ValueError("Cannot normalize z: mesh has zero z-extent.")

    if center_xy == "bounds":
        xy_center = np.array(
            [
                0.5 * (float(np.min(x)) + float(np.max(x))),
                0.5 * (float(np.min(y)) + float(np.max(y))),
            ],
            dtype=float,
        )

    elif center_xy == "centroid":
        xy_center = np.asarray(mesh_out.centroid[:2], dtype=float)

    elif center_xy is None:
        xy_center = np.array([0.0, 0.0], dtype=float)

    else:
        raise ValueError(
            "center_xy must be 'bounds', 'centroid', or None. "
            f"Got {center_xy!r}."
        )

    x_centered = x - xy_center[0]
    y_centered = y - xy_center[1]

    radial = np.sqrt(x_centered**2 + y_centered**2)
    current_R = float(np.max(radial))

    if current_R <= 0:
        raise ValueError("Cannot normalize x/y: mesh has zero radial extent.")

    z_mid = 0.5 * (z_min + z_max)

    scale_xy = float(target_R) / current_R
    scale_z = 2.0 / z_span

    transform = np.array(
        [
            [scale_xy, 0.0,      0.0,     -scale_xy * xy_center[0]],
            [0.0,      scale_xy, 0.0,     -scale_xy * xy_center[1]],
            [0.0,      0.0,      scale_z, -scale_z * z_mid],
            [0.0,      0.0,      0.0,      1.0],
        ],
        dtype=float,
    )

    mesh_out.apply_transform(transform)

    if verbose:
        v2 = np.asarray(mesh_out.vertices)
        r2 = np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)

        print("Challenge-cylinder normalization")
        print(f"  target R:             {target_R}")
        print(f"  center_xy mode:        {center_xy}")
        print(f"  original xy center:    {xy_center}")
        print(f"  original radial max:   {current_R}")
        print(f"  original z range:      [{z_min}, {z_max}]")
        print(f"  scale_xy:              {scale_xy}")
        print(f"  scale_z:               {scale_z}")
        print(f"  normalized radial max: {np.max(r2)}")
        print(f"  normalized z range:    [{np.min(v2[:, 2])}, {np.max(v2[:, 2])}]")

    return mesh_out
