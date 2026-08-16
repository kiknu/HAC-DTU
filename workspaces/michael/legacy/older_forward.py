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


@dataclass
class Dataset:
    model_num: int
    origin: Literal["real", "simulated"]
    curve_type: Literal["binary", "intensity"]
    data: pd.DataFrame


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


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHALLENGE_DATA_DIR = PROJECT_DIR / "challenge_data"


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
    raise RuntimeError(
        "raw public STL meshes must not enter the differentiable renderer; "
        "use an 80-face smoke mesh or a 1,280-face optimizer proxy"
    )
    # Kept below as historical reference; this path is intentionally disabled.
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
            * 1
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
        shader=CombinedShader(device=device, cameras=camera, lights=lights, albedo=1.0),
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

    # N=360 aligns with the challenge's specified interpolation grid for real/Blender data
    N = 842

    origin = "real"
    curve_type = "intensity"

    # 1. Load Data & Geometries
    asteroid_mesh = load_asteroid_mesh(model_num=target_model, device=compute_device)
    real_dataset = load_data(
        model_num=target_model, origin=origin, curve_type=curve_type
    )

    # Extracting the 0-degree phase angle, horizontal camera curve as an example
    target_column = "45_hor1"
    real_intensity_curve = real_dataset.data[target_column].values

    # Slice the real data to match the N=360 length if it contains extra frames
    real_intensity_curve = real_intensity_curve[:N]

    verts = asteroid_mesh.verts_packed()
    min_verts = verts.min(dim=0).values
    max_verts = verts.max(dim=0).values
    centroid = (min_verts + max_verts) / 2.0
    extent = (max_verts - min_verts).max()

    batched_meshes = asteroid_mesh.extend(N)

    angles = torch.linspace(0, 2 * np.pi, steps=N, device=compute_device)
    axes = torch.zeros(N, 3, device=compute_device)
    axes[:, 2] = 1.0

    R_batched = axis_angle_to_matrix(axes * angles.unsqueeze(1))
    transforms = Transform3d(device=compute_device).rotate(R_batched)

    verts_padded = batched_meshes.verts_padded()
    verts_rotated = transforms.transform_points(verts_padded)
    rotated_batched_meshes = batched_meshes.update_padded(verts_rotated)

    target_position = centroid.cpu().numpy()
    distance = extent.item() * 2.0

    # Camera on the X-axis looking towards the origin
    e_sun = target_position + np.array([-distance, 0.0, 0.0])
    plane_scale = (2.0 / extent.item()) * 0.9

    # Extract angles based on the target_column (e.g., "0_hor1", "45_top")
    phi_deg = float(target_column.split("_")[0])

    # Map the suffix to the correct elevation (alpha)
    alphas_dict = {"hor1": 0, "hor2": 0, "top": 26, "bottom": -26}
    # Note: 0-degree phase angles have slightly different elevations (21/-21) in the reference script.
    if phi_deg == 0 and "top" in target_column:
        alpha_deg = 26
    elif phi_deg == 0 and "bottom" in target_column:
        alpha_deg = -26
    else:
        suffix = target_column.split("_")[1]
        alpha_deg = alphas_dict.get(suffix, 0)

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

    # Co-aligned light direction to match the "0_hor1" zero-degree phase angle
    offset_light_direction = [[1.0, 0.0, 0.0]]
    resolution = 100

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

    # 25.5 on a 0-255 scale translates to 0.1 on a 0.0-1.0 scale
    threshold_val = 25.5 / 255.0

    # Zero out pixels below the threshold
    simulated_rgb[simulated_rgb < threshold_val] = 0.0

    simulated_intensity_curve = simulated_rgb.sum(dim=(1, 2, 3)).cpu().numpy()

    # simulated_intensity_curve = (
    #    rendered_images[..., :3].sum(dim=(1, 2, 3)).cpu().numpy()
    # )

    # Normalization prevents scaling mismatches between pixel-sums and physical sensor counts
    simulated_normalized = simulated_intensity_curve / simulated_intensity_curve.mean()
    real_normalized = real_intensity_curve / real_intensity_curve.mean()

    from scipy import signal

    # Calculate the optimal shift to align the phases
    correlation = signal.correlate(real_normalized, simulated_normalized, mode="full")
    shift = np.argmax(correlation) - (len(simulated_normalized) - 1)

    # Apply the shift
    real_aligned = real_normalized  # np.roll(real_normalized, -shift)

    # 3. Plot Comparison
    plt.figure(figsize=(10, 5))
    x_axis = np.linspace(0, 360, N)

    plt.plot(
        x_axis,
        simulated_normalized,
        color="blue",
        linewidth=2.0,
        label=f"Forward Model: Lommel-Seeliger",
    )

    plt.plot(
        x_axis,
        real_aligned,
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label=f"{origin} Data ({target_column})",
    )

    plt.title(f"Lightcurve Comparison: Model {target_model}")
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Normalized Intensity (Mean = 1.0)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("lightcurve_comparison.png", bbox_inches="tight")
    print("Saved comparative lightcurve plot to 'lightcurve_comparison.png'")
