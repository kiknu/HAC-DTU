import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from hac.constants import CAMERA_ANGLE_ALPHA
from hac.data import get_model_stl_path, load_data
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def get_unit_vectors(phi_deg, alpha_deg):
    """
    Compute the unit vectors for the light source and camera.

    Parameters:
    - phi_deg: Azimuthal angle (degrees) between light source and camera.
    - alpha_deg: Elevation angle (degrees) between xy-plane and camera.

    Returns:
    - light_vector: Unit vector for the light source (fixed at (-1, 0, 0)).
    - camera_vector: Unit vector for the camera.
    """
    # Convert angles to radians
    phi = np.radians(phi_deg - 180)  # Adjust phi to align with negative x-axis
    alpha = np.radians(alpha_deg)

    # Light source unit vector (fixed)
    light_vector = np.array([-1, 0, 0])

    # Camera unit vector
    camera_vector = np.array(
        [np.cos(phi) * np.cos(alpha), np.sin(phi) * np.cos(alpha), np.sin(alpha)]
    )

    return light_vector, camera_vector


def convex_forward_model_gpu(
    stl_path: str,
    scattering_law: str,
    e_sun: np.ndarray,
    e_obs: np.ndarray,
    Lambert_weight: float = 0,
    albedo: float = 1.0,
    n_phases: int = 360,
    phase_batch_size: int = 512,
    normalize=True,
    verbose=False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated version of convex_forward_model using CuPy.

    Vectorises over all facets and a batch of phases simultaneously.
    Memory scales as (phase_batch_size * n_facets * 4 bytes) per array;
    reduce phase_batch_size if you hit OOM on the GPU.

    Args:
        stl_path: Path to the .stl file.
        scattering_law: 'Lommel-Seeliger' or 'Lambert'.
        e_sun: Unit vector towards the Sun (shape: (3,)).
        e_obs: Unit vector towards the observer (shape: (3,)).
        albedo: Constant albedo (default: 1.0).
        n_phases: Number of rotational phases to sample (default: 100).
        phase_batch_size: Phases processed per GPU kernel launch (default: 512).

    Returns:
        phases: Array of rotational phases (0 to 2π).
        brightness: Normalised brightness array (same length as phases).
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not installed. Run: pip install cupy-cuda11x")

    if scattering_law not in ("Lommel-Seeliger", "Lambert", "LSL", "None"):
        raise ValueError(
            "scattering_law must be 'Lommel-Seeliger' or 'Lambert' or 'LSL'."
        )

    # ------------------------------------------------------------------ #
    # 1. Load mesh and precompute normals + areas on the CPU              #
    # ------------------------------------------------------------------ #
    shape = mesh.Mesh.from_file(stl_path)
    if verbose:
        print(f"Length of vectors in mesh: {len(shape.vectors)}")
        print(f"Using scattering law: {scattering_law}")
        print(f"With weight {Lambert_weight}") if scattering_law == "LSL" else print(
            " "
        )

    e_sun = np.asarray(e_sun, dtype=np.float32)
    e_obs = np.asarray(e_obs, dtype=np.float32)
    e_sun /= np.linalg.norm(e_sun)
    e_obs /= np.linalg.norm(e_obs)

    vectors = shape.vectors.astype(np.float32)  # (n_facets, 3, 3)
    edge1 = vectors[:, 1, :] - vectors[:, 0, :]  # (n_facets, 3)
    edge2 = vectors[:, 2, :] - vectors[:, 0, :]

    cross = np.cross(edge1, edge2)  # (n_facets, 3)
    norm_mag = np.linalg.norm(cross, axis=1, keepdims=True)
    normals_cpu = cross / norm_mag  # (n_facets, 3)  unit normals
    areas_cpu = 0.5 * norm_mag.squeeze(1)  # (n_facets,)

    # ------------------------------------------------------------------ #
    # 2. Move static data to GPU                                          #
    # ------------------------------------------------------------------ #
    normals_gpu = cp.asarray(normals_cpu)  # (n_facets, 3)
    areas_gpu = cp.asarray(areas_cpu)  # (n_facets,)

    # ------------------------------------------------------------------ #
    # 3. Generate phases and process in batches                           #
    # ------------------------------------------------------------------ #
    phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False).astype(np.float32)
    brightness_gpu = cp.zeros(n_phases, dtype=cp.float32)

    ex_sun, ey_sun, ez_sun = e_sun.tolist()
    ex_obs, ey_obs, ez_obs = e_obs.tolist()

    for start in range(0, n_phases, phase_batch_size):
        end = min(start + phase_batch_size, n_phases)
        batch_phases = cp.asarray(phases[start:end])  # (B,)

        cos_p = cp.cos(batch_phases)
        sin_p = cp.sin(batch_phases)

        # Rotate sun vector for each phase in batch: R_z(phase) @ e_sun
        # R_z = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        # Result: (B, 3)
        e_sun_rot = cp.stack(
            [
                cos_p * ex_sun - sin_p * ey_sun,
                sin_p * ex_sun + cos_p * ey_sun,
                cp.full(len(batch_phases), ez_sun, dtype=cp.float32),
            ],
            axis=1,
        )

        e_obs_rot = cp.stack(
            [
                cos_p * ex_obs - sin_p * ey_obs,
                sin_p * ex_obs + cos_p * ey_obs,
                cp.full(len(batch_phases), ez_obs, dtype=cp.float32),
            ],
            axis=1,
        )

        # Dot products with all normals: (B, n_facets)
        mu0 = e_sun_rot @ normals_gpu.T
        mu = e_obs_rot @ normals_gpu.T

        # Visibility mask
        mask = (mu0 > 0) & (mu > 0)

        # Scattering function
        if scattering_law == "Lommel-Seeliger":
            S = cp.where(mask, mu * mu0 / (mu + mu0), 0.0)
        elif scattering_law == "Lambert":  # Lambert
            S = cp.where(mask, mu * mu0, 0.0)
        elif scattering_law == "LSL":
            # S = cp.where(mask, (1-Lambert_weight)*mu * mu0 / (mu + mu0)+Lambert_weight*mu*mu0,0.0)
            S = cp.where(
                mask, mu * mu0 / (mu + mu0) + Lambert_weight * mu * mu0, 0.0
            )  # see e.g. eq (2) in https://arxiv.org/abs/2502.16455
        elif scattering_law == "None":
            S = cp.where(mask, mu, 0.0)

        # Brightness per phase: weighted sum over facets
        brightness_gpu[start:end] = (S * areas_gpu).sum(axis=1) * albedo

    brightness = cp.asnumpy(brightness_gpu)
    if normalize:
        return phases, brightness / brightness.mean()
    else:
        return phases, brightness


def _mse_for_weight(args):
    lw, stl_path, alphas_items, e_sun_list, blender_data = args
    e_sun = np.array(e_sun_list, dtype=np.float32)
    total_mse = 0.0
    for key, alpha_deg in alphas_items:
        phi_deg = float(key.split("_")[0])
        _, e_obs = get_unit_vectors(phi_deg, alpha_deg)
        _, lc_model = convex_forward_model_gpu(
            stl_path,
            scattering_law="LSL",
            Lambert_weight=lw,
            e_sun=e_sun,
            e_obs=e_obs,
            albedo=1.0,
            n_phases=360,
        )
        total_mse += np.mean((lc_model - blender_data[key]) ** 2)
    avg_mse = total_mse / len(alphas_items)
    print(f"Lambert_weight={lw:.4f}: avg MSE over all geometries: {avg_mse:.6f}")
    return avg_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convex forward model simulations")
    parser.add_argument("--asteroid", type=int, default=1)
    parser.add_argument("--opt_weight", action="store_true")
    parser.add_argument("--cam_angle", type=str, default="0_hor1")
    args = parser.parse_args()

    alphas_dict = CAMERA_ANGLE_ALPHA

    stl_path = str(get_model_stl_path(args.asteroid))
    ds_real = load_data(args.asteroid, "real", "intensity")
    ds_blender = load_data(args.asteroid, "simulated", "intensity")
    e_sun = np.array([-1.0, 0.0, 0.0])

    if args.opt_weight:
        lambert_weights = np.linspace(0, 20, 20)

        blender_data = {key: ds_blender.data[key].values for key in alphas_dict}
        alphas_items = list(alphas_dict.items())
        n_workers = min(len(lambert_weights), os.cpu_count())
        task_args = [
            (lw, stl_path, alphas_items, e_sun.tolist(), blender_data)
            for lw in lambert_weights
        ]

        print(
            f"Running {len(lambert_weights)} weights in parallel on {n_workers} CPUs..."
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            mse_per_weight = list(executor.map(_mse_for_weight, task_args))

        best_idx = int(np.argmin(mse_per_weight))
        print(
            f"\nBest Lambert_weight: {lambert_weights[best_idx]:.4f}  MSE: {mse_per_weight[best_idx]:.6f}"
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            lambert_weights,
            mse_per_weight,
            marker="o",
            color="steelblue",
            linewidth=1.5,
        )
        ax.axvline(
            lambert_weights[best_idx],
            color="tomato",
            linestyle="--",
            label=f"best w={lambert_weights[best_idx]:.2f}",
        )
        ax.set_xlabel("Lambert weight")
        ax.set_ylabel("Avg MSE (all geometries)")
        ax.set_title("LSL scattering: MSE vs Lambert weight")
        ax.legend()
        fig.tight_layout()
        out_path = f"plots/asteroid{args.asteroid}_mse_vs_lambert_weight.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure to {out_path}")

    else:
        camera = args.cam_angle  # e.g. "45_hor1"
        lw = 1
        scattering_law = "LSL"
        alpha_deg = alphas_dict[camera]
        phi_deg = int(camera.split("_")[0])
        print(f"alpha = {alpha_deg}")
        print(f"phi = {phi_deg}")
        _, e_obs = get_unit_vectors(phi_deg=phi_deg, alpha_deg=alpha_deg)
        print(f"e_obs = {e_obs}")
        phases, lc_model = convex_forward_model_gpu(
            stl_path,
            scattering_law=scattering_law,
            Lambert_weight=lw,
            e_sun=e_sun,
            e_obs=e_obs,
            albedo=1.0,
            n_phases=360,
            verbose=True,
        )
        real = ds_real.data[camera].values
        real_phases = np.linspace(0, 2 * np.pi, len(real), endpoint=False)
        blender = ds_blender.data[camera].values

        print(f"MSE: {np.mean((lc_model - blender) ** 2)}")
        plt.figure()
        plt.title(f"Asteroid: {args.asteroid} - Intensity_Camera_{camera}")
        if scattering_law == "LSL":
            plt.plot(phases, lc_model / blender, label=f"Convex forward (c={lw})")
        elif scattering_law == "Lambert":
            plt.plot(phases, lc_model / blender, label=f"Convex forward (Lambert)")
        plt.plot(phases, blender, label="Blender simulation")
        plt.plot(real_phases, real, label="Real")
        plt.legend()
        plt.show()
        if scattering_law == "LSL":
            plt.savefig(f"plots/ast_{args.asteroid}_cam_{camera}_lw_{lw}.png")
        elif scattering_law == "Lambert":
            plt.savefig(f"plots/ast_{args.asteroid}_cam_{camera}_Lambert.png")
