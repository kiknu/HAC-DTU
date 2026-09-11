"""Read-only independent Embree first-hit and solar-ray reference.

Raw public surfaces stay on CPU. Geometry is never simplified by this module.
The optional reference radius holds camera position and framing fixed across meshes.
"""
import math
import numpy as np
import trimesh

def ray_images(mesh, angles, views, resolution, distance, smooth=True, reference_radius=None):
    """Independent pixel first hits + hard solar visibility; fixed mesh BVH.

    Pixel ray directions agree with the maintained renderer: square image,
    extent 1.2 R in the origin plane, observer eye distance d R, z-up.
    Return Lambert/LS endpoint radiances for cheap downstream response sweeps.
    No geometry repair/simplification or radial resampling is performed here.
    """
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    intersector = RayMeshIntersector(mesh)
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    triangles = verts[faces]
    area_normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    face_normals = area_normals / np.maximum(np.linalg.norm(area_normals, axis=1, keepdims=True), 1e-20)
    vertex_normals = np.zeros_like(verts)
    for k in range(3): np.add.at(vertex_normals, faces[:, k], area_normals)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-20)
    radius = np.linalg.norm(verts, axis=1).max() if reference_radius is None else reference_radius
    coords = (np.arange(resolution) + .5) / resolution * 2 - 1
    xx, yy = np.meshgrid(coords, -coords)
    xy = np.stack((xx.ravel(), yy.ravel()), axis=-1) * 1.2 * radius
    endpoint = np.zeros((2, len(angles), len(views), resolution, resolution), np.float32)
    for fi, angle in enumerate(np.radians(angles)):
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c,-s,0], [s,c,0], [0,0,1]])
        sun = np.array([-1.,0,0]) @ rot.T
        for ci, world_view in enumerate(views):
            view = world_view @ rot.T
            right = np.cross([0,0,1], view); right /= np.linalg.norm(right)
            up = np.cross(view, right)
            plane = xy[:, :1] * right + xy[:, 1:] * up
            if math.isinf(distance):
                origins = plane + 4*radius*view
                directions = np.broadcast_to(-view, origins.shape).copy()
            else:
                origins = np.broadcast_to(distance*radius*view, plane.shape).copy()
                directions = plane - origins
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            locations, ray_ids, face_ids = intersector.intersects_location(origins, directions, multiple_hits=False)
            n = face_normals[face_ids]
            if smooth:
                bary = trimesh.triangles.points_to_barycentric(triangles[face_ids], locations)
                n = (vertex_normals[faces[face_ids]] * bary[..., None]).sum(axis=1)
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
            mu0 = np.maximum(n @ sun, 0)
            mu = np.maximum(np.einsum("ij,ij->i", n, -directions[ray_ids]), 0)
            # Geometric offset avoids same-facet hits; a second tiny step along
            # the ray avoids re-hitting edges. Scale is relative to bounding R.
            solar_origins = locations + face_normals[face_ids]*radius*1e-5 + sun*radius*1e-5
            shadowed = intersector.intersects_any(solar_origins, np.broadcast_to(sun, solar_origins.shape).copy())
            lambert = mu0 * ~shadowed
            ls = mu0 / np.maximum(mu0+mu, 1e-3) * ~shadowed
            endpoint[0, fi, ci].ravel()[ray_ids] = lambert
            endpoint[1, fi, ci].ravel()[ray_ids] = ls
        if fi % 12 == 0: print(f"ray frame {fi}/{len(angles)}", flush=True)
    return endpoint
