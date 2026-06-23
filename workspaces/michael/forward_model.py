import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    DirectionalLights,
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, axis_angle_to_matrix

torch.set_default_dtype(torch.float32)


# Define the project and challenge data directories
PROJECT_DIR = Path(os.environ.get("DSDIR", Path(__file__).resolve().parent.parent))
CHALLENGE_DATA_DIR = PROJECT_DIR / "challenge_data"

# Elevation angles from the challenge setup used in the convex baseline script.
CAMERA_ELEVATION_DEG = {
    "0_hor1": 0.0,
    "0_hor2": 0.0,
    "0_top": 21.0,
    "0_bottom": -21.0,
    "45_hor1": 0.0,
    "45_hor2": 0.0,
    "45_top": 26.0,
    "45_bottom": -26.0,
    "90_hor1": 0.0,
    "90_hor2": 0.0,
    "90_top": 26.0,
    "90_bottom": -26.0,
    "135_hor1": 0.0,
    "135_hor2": 0.0,
    "135_top": 26.0,
    "135_bottom": -26.0,
    "225_hor1": 0.0,
    "225_hor2": 0.0,
    "225_top": 24.0,
    "225_bottom": -24.0,
    "270_hor1": 0.0,
    "270_hor2": 0.0,
    "270_top": 24.0,
    "270_bottom": -24.0,
    "315_hor1": 0.0,
    "315_hor2": 0.0,
    "315_top": 24.0,
    "315_bottom": -24.0,
}


@dataclass
class Dataset:
    model_num: int
    origin: Literal["real", "simulated"]
    curve_type: Literal["binary", "intensity"]
    data: pd.DataFrame


class CombinedShader(nn.Module):
    def __init__(self, device, cameras, lights, albedo=1.0, blend_params=None):
        super().__init__()
        self.albedo = albedo
        self.Lambertian = LambertianOrthographicShader(
            device=device,
            cameras=cameras,
            lights=lights,
            albedo=albedo,
            blend_params=blend_params,
        )
        self.LS_shader = LommelSeeligerOrthographicShader(
            device=device,
            cameras=cameras,
            lights=lights,
            albedo=albedo,
            blend_params=blend_params,
        )

    def forward(self, fragments, meshes, **kwargs):
        lambertian = self.Lambertian(fragments, meshes, **kwargs)
        ls = self.LS_shader(fragments, meshes, **kwargs)
        return 0.5 * (lambertian + ls)


class LommelSeeligerOrthographicShader(nn.Module):
    """
    Calculates pixel intensities based on the Lommel-Seeliger reflection coefficient.
    Assumes an orthographic camera and directional lighting.
    """

    def __init__(self, device, cameras, lights, albedo=1.0, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.lights = lights
        self.albedo = albedo
        self.device = device

        self.blend_params = (
            blend_params
            if blend_params is not None
            else BlendParams(background_color=(0.0, 0.0, 0.0))
        )

    def forward(self, fragments, meshes, **kwargs):
        verts_normals = meshes.verts_normals_packed()
        faces = meshes.faces_packed()
        faces_verts_normals = verts_normals[faces]

        normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_normals
        )
        normals = F.normalize(normals, p=2, dim=-1)

        # Reversed to point FROM object TO camera/light
        view_dir = -self.cameras.R[:, :, 2].view(-1, 1, 1, 1, 3)
        view_dir = F.normalize(view_dir, p=2, dim=-1)

        light_dir = -self.lights.direction.view(-1, 1, 1, 1, 3)
        light_dir = F.normalize(light_dir, p=2, dim=-1)

        mu = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        mu_0 = torch.sum(normals * light_dir, dim=-1, keepdim=True)

        valid_mask = (mu > 0) & (mu_0 > 0)
        intensity = torch.zeros_like(mu)
        denominator = mu + mu_0

        intensity[valid_mask] = (
            mu_0[valid_mask] / denominator[valid_mask]
        ) * self.albedo

        pixel_colors = torch.ones_like(normals) * intensity

        images = softmax_rgb_blend(
            colors=pixel_colors, fragments=fragments, blend_params=self.blend_params
        )

        return images


class LambertianOrthographicShader(nn.Module):
    """
    Calculates pixel intensities based on the Lambertian reflection coefficient.
    Assumes an orthographic camera and directional lighting.
    """

    def __init__(self, device, cameras, lights, albedo=1.0, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.lights = lights
        self.albedo = albedo
        self.device = device

        self.blend_params = (
            blend_params
            if blend_params is not None
            else BlendParams(background_color=(0.0, 0.0, 0.0))
        )

    def forward(self, fragments, meshes, **kwargs):
        verts_normals = meshes.verts_normals_packed()
        faces = meshes.faces_packed()
        faces_verts_normals = verts_normals[faces]

        normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_normals
        )
        normals = F.normalize(normals, p=2, dim=-1)

        # Reversed to point FROM object TO camera/light
        view_dir = -self.cameras.R[:, :, 2].view(-1, 1, 1, 1, 3)
        view_dir = F.normalize(view_dir, p=2, dim=-1)

        light_dir = -self.lights.direction.view(-1, 1, 1, 1, 3)
        light_dir = F.normalize(light_dir, p=2, dim=-1)

        mu_0 = torch.sum(normals * light_dir, dim=-1, keepdim=True)
        valid_mask = mu_0 > 0

        intensity = torch.zeros_like(mu_0)
        intensity[valid_mask] = mu_0[valid_mask] * self.albedo

        pixel_colors = torch.ones_like(normals) * intensity

        images = softmax_rgb_blend(
            colors=pixel_colors, fragments=fragments, blend_params=self.blend_params
        )
        return images


def get_model_folder(model_num: int) -> Path:
    _model_folder_suffix = "public" if model_num <= 3 else "secret"
    model_folder_str = f"AsteroidModel{model_num:02d}_shape_{_model_folder_suffix}"
    model_dir = CHALLENGE_DATA_DIR / model_folder_str
    assert model_dir.is_dir(), f"Model directory not found at {model_dir}."
    return model_dir


def get_model_stl_path(model_num: int) -> Path:
    if model_num > 3:
        raise ValueError(f"Model number {model_num} is secret.")
    model_dir = get_model_folder(model_num)
    stl_file_path = model_dir / f"asteroid{model_num:d}.stl"
    assert stl_file_path.is_file(), f"STL file not found at {stl_file_path}."
    return stl_file_path


def load_asteroid_mesh(model_num: int, device: torch.device) -> Meshes:
    stl_path = get_model_stl_path(model_num)
    mesh_trimesh = trimesh.load(stl_path, force="mesh")

    with tempfile.NamedTemporaryFile(suffix=".obj", delete=True) as temp_obj_file:
        mesh_trimesh.export(temp_obj_file.name, file_type="obj")
        pytorch3d_mesh = load_objs_as_meshes([temp_obj_file.name], device=device)

    return pytorch3d_mesh


def load_data(
    model_num: int,
    origin: Literal["real", "simulated"],
    curve_type: Literal["binary", "intensity"],
) -> Dataset:
    model_dir = get_model_folder(model_num)

    # Changed :02d to :d to match the 'Asteroid3_lightcurve_data' format
    lightcurve_data_folder_str = f"Asteroid{model_num:d}_lightcurve_data"
    lightcurve_data_dir = model_dir / lightcurve_data_folder_str
    assert lightcurve_data_dir.is_dir(), (
        f"Light curve data directory not found at {lightcurve_data_dir}."
    )

    # The file names likely still use the zero-padded format, keep :02d here unless they also lack it
    data_file_str = f"Asteroid{model_num:02d}_lightcurve_{curve_type}"
    if origin == "simulated":
        data_file_str += "_blender"

    data_file_str += ".txt"

    data_file_path = lightcurve_data_dir / data_file_str
    assert data_file_path.is_file(), f"Data file not found at {data_file_path}."

    data = pd.read_csv(data_file_path, header=None, index_col=0)

    data.index.name = "frame"
    data = data.rename(
        columns={
            1: "0_hor1",
            2: "0_hor2",
            3: "0_top",
            4: "0_bottom",
            5: "45_hor1",
            6: "45_hor2",
            7: "45_top",
            8: "45_bottom",
            9: "90_hor1",
            10: "90_hor2",
            11: "90_top",
            12: "90_bottom",
            13: "135_hor1",
            14: "135_hor2",
            15: "135_top",
            16: "135_bottom",
            17: "225_hor1",
            18: "225_hor2",
            19: "225_top",
            20: "225_bottom",
            21: "270_hor1",
            22: "270_hor2",
            23: "270_top",
            24: "270_bottom",
            25: "315_hor1",
            26: "315_hor2",
            27: "315_top",
            28: "315_bottom",
        }
    )

    return Dataset(
        model_num=model_num,
        origin=origin,
        curve_type=curve_type,
        data=data,
    )


def get_camera_angles_deg(target_column: str) -> tuple[float, float]:
    try:
        phi_str, suffix = target_column.split("_", maxsplit=1)
        phi_deg = float(phi_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid target_column '{target_column}'. Expected format '<phi>_<suffix>'."
        ) from exc

    alpha_deg = CAMERA_ELEVATION_DEG.get(target_column)
    if alpha_deg is None:
        fallback_alpha = {"hor1": 0.0, "hor2": 0.0, "top": 26.0, "bottom": -26.0}
        if suffix not in fallback_alpha:
            raise ValueError(f"Unknown camera suffix '{suffix}' in '{target_column}'.")
        alpha_deg = fallback_alpha[suffix]

    return phi_deg, float(alpha_deg)


def normalize_curve(curve: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean_val = float(np.mean(curve))
    if abs(mean_val) < eps:
        raise ValueError("Cannot normalize curve with near-zero mean intensity.")
    return curve / mean_val


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def create_orthographic_camera(
    eye_pos: np.ndarray,
    target_pos: np.ndarray,
    device: torch.device,
    focal_length: float = 1.0,
) -> OrthographicCameras:
    eye_tensor = torch.from_numpy(eye_pos).float().view(1, 3).to(device)
    at_tensor = torch.from_numpy(target_pos).float().view(1, 3).to(device)
    R, T = look_at_view_transform(eye=eye_tensor, at=at_tensor, up=((0.0, 1.0, 0.0),))
    return OrthographicCameras(device=device, R=R, T=T, focal_length=focal_length)


def shift_align(candidate: np.ndarray, best_shift: int = 0) -> tuple[np.ndarray, int]:
    # Shift the candidate curve circularly by best_shift and return the aligned curve and the shift value.
    N = len(candidate)
    if N < 2:
        raise ValueError("Candidate curve must have at least 2 points for alignment.")
    aligned = np.roll(candidate, shift=best_shift)
    return aligned, best_shift


def render_mesh_batch(
    meshes: Meshes,
    camera: OrthographicCameras,
    device: torch.device,
    image_size: int = 32,
    light_direction: list = [[1.0, 0.0, 0.0]],
) -> torch.Tensor:
    if meshes.textures is None:
        verts_rgb = (
            torch.ones(
                [
                    meshes.verts_padded().shape[0],
                    meshes.verts_packed().shape[0] // meshes.verts_padded().shape[0],
                    3,
                ],
                device=device,
            )
            * 1.0
        )
        meshes.textures = TexturesVertex(verts_features=verts_rgb)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-3,
        faces_per_pixel=50,
    )

    lights = DirectionalLights(device=device, direction=light_direction)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=CombinedShader(
            device=device, cameras=camera, lights=lights, albedo=1.0
        ),
    )

    return renderer(meshes)


def save_first_frame(
    rendered_images: torch.Tensor, output_path: str = "first_frame.png"
):
    first_frame = rendered_images[0, ..., :3].detach().cpu().numpy()
    first_frame = np.clip(first_frame, 0.0, 1.0)
    plt.imsave(output_path, first_frame)
    print(f"Saved debug frame to '{output_path}'")


if __name__ == "__main__":
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = 3

    # 1. Load Data & Geometries
    asteroid_mesh = load_asteroid_mesh(model_num=target_model, device=compute_device)
    real_dataset = load_data(
        model_num=target_model, origin="real", curve_type="intensity"
    )

    # Extracting one geometry curve for fitting/comparison.
    target_column = "45_bottom"
    if target_column not in real_dataset.data.columns:
        raise ValueError(f"target_column '{target_column}' not found in loaded dataset.")

    real_intensity_curve = real_dataset.data[target_column].to_numpy(dtype=np.float64)
    real_frame_axis = real_dataset.data.index.to_numpy(dtype=np.float64)
    N = len(real_intensity_curve)
    if N < 2:
        raise ValueError("Need at least 2 points in real observations to compare curves.")

    print(f"Loaded {N} real samples for {target_column} (frames {real_frame_axis[0]}-{real_frame_axis[-1]}).")

    verts = asteroid_mesh.verts_packed()
    min_verts = verts.min(dim=0).values
    max_verts = verts.max(dim=0).values
    centroid = (min_verts + max_verts) / 2.0
    extent = (max_verts - min_verts).max()

    batched_meshes = asteroid_mesh.extend(N)

    # endpoint=False avoids duplicating the 0-degree sample at 360 degrees.
    angles = torch.arange(N, device=compute_device) * (2.0 * np.pi / N)
    axes = torch.zeros(N, 3, device=compute_device)
    axes[:, 2] = 1.0

    R_batched = axis_angle_to_matrix(axes * angles.unsqueeze(1))
    transforms = Transform3d(device=compute_device).rotate(R_batched)

    # Rotate around the asteroid centroid, not around the world origin.
    verts_padded = batched_meshes.verts_padded()
    centroid_batched = centroid.view(1, 1, 3)
    verts_centered = verts_padded - centroid_batched
    verts_rotated = transforms.transform_points(verts_centered) + centroid_batched
    rotated_batched_meshes = batched_meshes.update_padded(verts_rotated)

    target_position = centroid.cpu().numpy()
    distance = extent.item() * 2.0

    plane_scale = (2.0 / extent.item()) * 0.9

    # Extract camera geometry from the selected observational column.
    phi_deg, alpha_deg = get_camera_angles_deg(target_column)

    # Convert to radians
    phi = np.radians(phi_deg - 180)
    alpha = np.radians(alpha_deg)

    # Calculate the unit vector for the camera
    camera_vector = np.array(
        [np.cos(phi) * np.cos(alpha), np.sin(phi) * np.cos(alpha), np.sin(alpha)]
    )

    # Position the camera at the correct distance along this vector
    e_obs = target_position + (distance * camera_vector)

    sun_camera = create_orthographic_camera(
        eye_pos=e_obs,
        target_pos=target_position,
        device=compute_device,
        focal_length=plane_scale,
    )

    # Keep the Sun direction fixed; phase angle comes from camera placement.
    offset_light_direction = [[1.0, 0.0, 0.0]]
    resolution = 128

    rendered_images = render_mesh_batch(
        meshes=rotated_batched_meshes,
        camera=sun_camera,
        device=compute_device,
        image_size=resolution,
        light_direction=offset_light_direction,
    )

    save_first_frame(rendered_images, output_path="first_frame.png")

    # 2. Extract and Normalize Curves

    simulated_rgb = rendered_images[..., :3]
    simulated_alpha = rendered_images[..., 3:4]

    # Alpha-weighting suppresses soft background blending leakage.
    simulated_intensity_curve = (
        (simulated_rgb * simulated_alpha).sum(dim=(1, 2, 3)).detach().cpu().numpy()
    )

    real_normalized = normalize_curve(real_intensity_curve)
    simulated_normalized = normalize_curve(simulated_intensity_curve)
    simulated_aligned, best_shift = shift_align(
        candidate=simulated_normalized,
        best_shift=-10,
    )

    mse_before = float(np.mean((real_normalized - simulated_normalized) ** 2))
    mse_after = float(np.mean((real_normalized - simulated_aligned) ** 2))
    corr_before = pearson_corr(real_normalized, simulated_normalized)
    corr_after = pearson_corr(real_normalized, simulated_aligned)

    print(
        f"Alignment shift={best_shift} | MSE before={mse_before:.6f}, after={mse_after:.6f} | "
        f"Corr before={corr_before:.4f}, after={corr_after:.4f}"
    )

    # 3. Plot Comparison
    plt.figure(figsize=(10, 5))
    x_axis = real_frame_axis

    plt.plot(
        x_axis,
        simulated_normalized,
        color="royalblue",
        linewidth=1.2,
        linestyle=":",
        alpha=0.5,
        label="Simulated (raw phase)",
    )
    plt.plot(
        x_axis,
        simulated_aligned,
        color="blue",
        linewidth=2.0,
        label=f"Simulated (aligned, shift={best_shift})",
    )
    plt.plot(
        x_axis,
        real_normalized,
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label=f"Real Data ({target_column}, normalized)",
    )

    plt.title(
        f"Lightcurve Comparison: Model {target_model} | {target_column} | MSE={mse_after:.5f}"
    )
    plt.xlabel("Observation Frame")
    plt.ylabel("Normalized Intensity (Mean = 1.0)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("lightcurve_comparison.png", bbox_inches="tight")
    print("Saved comparative lightcurve plot to 'lightcurve_comparison.png'")
