import trimesh


def calculate_voxel_measure(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Voxel-based measure.

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

    # Compute the number of voxels in A\B and B\A
    a_over_b = len(pred_voxels - true_voxels)
    b_over_a = len(true_voxels - pred_voxels)
    sum_mismatch = a_over_b + b_over_a

    # Denominator from your original formula
    total_volume_sum = len(true_voxels) + len(pred_voxels)
    score = 1.0 - (sum_mismatch / total_volume_sum)

    return score
