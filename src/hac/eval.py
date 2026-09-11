import numpy as np
import trimesh


def calculate_voxel_measure(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Fast voxel-based similarity measure.

    Parameters
    ----------
    true_mesh : trimesh.Trimesh
        Ground truth mesh.

    pred_mesh : trimesh.Trimesh
        Predicted mesh.

    pitch : float
        Voxel size. Smaller values give higher accuracy but require
        more memory and computation.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    # Voxelize both meshes
    vox_true = true_mesh.voxelized(pitch)
    vox_pred = pred_mesh.voxelized(pitch)

    # Get occupied voxel indices
    true_voxels = set(map(tuple, vox_true.sparse_indices))
    pred_voxels = set(map(tuple, vox_pred.sparse_indices))

    if len(true_voxels) == 0 and len(pred_voxels) == 0:
        return 0.0

    # Symmetric difference
    mismatch_count = len(true_voxels.symmetric_difference(pred_voxels))

    # Denominator from your original formula
    total_volume_sum = len(true_voxels) + len(pred_voxels)

    score = 1.0 - mismatch_count / total_volume_sum

    return np.clip(score, 0.0, 1.0)


def load_as_trimesh(mshfile: str) -> trimesh.Trimesh:
    """Load a mesh file as a triangle `trimesh.Trimesh` (official grader loader).

    Uses ``meshio`` (matching the challenge implementation) to read the file and
    keep only triangle cells; falls back to ``trimesh.load_mesh`` when meshio is
    unavailable so common formats such as STL still load.
    """
    try:
        import meshio  # type: ignore[import-not-found]

        m = meshio.read(mshfile)
        return trimesh.Trimesh(vertices=m.points, faces=m.cells_dict["triangle"])
    except ImportError:
        return trimesh.load_mesh(mshfile)


def _as_trimesh(mesh: trimesh.Trimesh | str) -> trimesh.Trimesh:
    return mesh if isinstance(mesh, trimesh.Trimesh) else load_as_trimesh(mesh)


def relative_volume_difference_voxelized(
    mesh1: trimesh.Trimesh | str,
    mesh2: trimesh.Trimesh | str,
    pitch: float = 0.05,
) -> tuple[float, float]:
    """Official challenge voxel measures on a shared, aligned voxel grid.

    Both meshes are padded with the SAME two ghost boxes (at the shared
    bounding-box corners) so their filled voxel grids share an origin, pitch,
    and shape, making an element-wise comparison valid. This is what makes the
    measure translation-sensitive; comparing each mesh's own voxel indices would
    silently ignore a global offset.

    Returns ``(measure1, measure2)`` where both are DIFFERENCES (0 = identical,
    larger = more different), the opposite convention to
    :func:`calculate_voxel_measure`:

    - ``measure1 = 1 - |A ∩ B| / |A ∪ B|`` (one minus voxel IoU)
    - ``measure2 = (|A \\ B| + |B \\ A|) / (|A| + |B|)`` (symmetric difference)
    """
    A = _as_trimesh(mesh1)
    B = _as_trimesh(mesh2)

    min_bound = np.minimum(A.bounds[0], B.bounds[0])
    max_bound = np.maximum(A.bounds[1], B.bounds[1])

    ghost_min = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    ghost_min.apply_translation(min_bound)
    ghost_max = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    ghost_max.apply_translation(max_bound)

    A_padded = trimesh.util.concatenate([A, ghost_min, ghost_max])
    B_padded = trimesh.util.concatenate([B, ghost_min, ghost_max])

    filled_A = A_padded.voxelized(pitch, method="subdivide").fill().matrix.astype(bool)
    filled_B = B_padded.voxelized(pitch, method="subdivide").fill().matrix.astype(bool)

    # Shared origin and pitch, so any float-rounding shape mismatch is a trailing
    # off-by-one; embed both grids at the common origin to stay aligned.
    shape = np.maximum(filled_A.shape, filled_B.shape)

    def embed(matrix: np.ndarray) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        out[: matrix.shape[0], : matrix.shape[1], : matrix.shape[2]] = matrix
        return out

    filled_A = embed(filled_A)
    filled_B = embed(filled_B)

    intersection = np.logical_and(filled_A, filled_B).sum()
    union = np.logical_or(filled_A, filled_B).sum()
    diff_ab = np.logical_and(filled_A, np.logical_not(filled_B)).sum()
    diff_ba = np.logical_and(filled_B, np.logical_not(filled_A)).sum()
    vol_a = int(filled_A.sum())
    vol_b = int(filled_B.sum())

    measure1 = 1.0 - intersection / union if union else 0.0
    measure2 = (diff_ab + diff_ba) / (vol_a + vol_b) if (vol_a + vol_b) else 0.0
    return float(measure1), float(measure2)


def calculate_voxel_measure_solid(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """Solid-volume similarity in ``[0, 1]`` (1 = identical).

    Thin wrapper over :func:`relative_volume_difference_voxelized` returning
    ``1 - measure2`` on the aligned shared grid, so it matches the official
    filled-voxel computation instead of comparing each mesh's own voxel indices.
    """
    _, measure2 = relative_volume_difference_voxelized(
        true_mesh, pred_mesh, pitch=pitch
    )
    return float(np.clip(1.0 - measure2, 0.0, 1.0))
