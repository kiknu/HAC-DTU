import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from pytorch3d.renderer.blending import BlendParams, hard_rgb_blend, softmax_rgb_blend
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, axis_angle_to_matrix

torch.set_default_dtype(torch.float32)


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

        # Default to a pitch-black background.
        # Crucial for lightcurves: background pixels must sum to 0.
        self.blend_params = (
            blend_params
            if blend_params is not None
            else BlendParams(background_color=(0.0, 0.0, 0.0))
        )

    def forward(self, fragments, meshes, **kwargs):
        # 1. Extract and interpolate normals
        verts_normals = meshes.verts_normals_packed()
        faces = meshes.faces_packed()
        faces_verts_normals = verts_normals[faces]

        normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_normals
        )
        normals = F.normalize(normals, p=2, dim=-1)

        # 2. Compute vectors (Reversed to point FROM object TO camera/light)
        view_dir = -self.cameras.R[:, :, 2].view(-1, 1, 1, 1, 3)
        view_dir = F.normalize(view_dir, p=2, dim=-1)

        light_dir = -self.lights.direction.view(-1, 1, 1, 1, 3)
        light_dir = F.normalize(light_dir, p=2, dim=-1)

        # 3. Calculate cosines
        mu = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        mu_0 = torch.sum(normals * light_dir, dim=-1, keepdim=True)

        # 4. Apply Lommel-Seeliger
        valid_mask = (mu > 0) & (mu_0 > 0)
        intensity = torch.zeros_like(mu)
        denominator = mu + mu_0

        intensity[valid_mask] = (
            mu_0[valid_mask] / denominator[valid_mask]
        ) * self.albedo

        pixel_colors = torch.ones_like(normals) * intensity

        # Replaced hard_rgb_blend with softmax_rgb_blend for differentiability
        images = softmax_rgb_blend(
            colors=pixel_colors, fragments=fragments, blend_params=self.blend_params
        )

        return images


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
    stl_path = get_model_stl_path(model_num)
    mesh_trimesh = trimesh.load(stl_path, force="mesh")

    with tempfile.NamedTemporaryFile(suffix=".obj", delete=True) as temp_obj_file:
        mesh_trimesh.export(temp_obj_file.name, file_type="obj")
        pytorch3d_mesh = load_objs_as_meshes([temp_obj_file.name], device=device)

    return pytorch3d_mesh


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
            * 0.7
        )
        meshes.textures = TexturesVertex(verts_features=verts_rgb)

    # Adjusted for soft, differentiable rasterization
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-4,  # Allows continuous gradients across edges
        faces_per_pixel=50,  # Evaluates multiple faces for soft blending
    )

    lights = DirectionalLights(device=device, direction=light_direction)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=LommelSeeligerOrthographicShader(
            device=device, cameras=camera, lights=lights
        ),
    )

    return renderer(meshes)


def save_first_frame(
    rendered_images: torch.Tensor, output_path: str = "first_frame.png"
):
    """
    Extracts and saves the first frame from the rendered batch to verify
    camera framing, scale, and lighting.
    """
    # Extract the first image and limit to the 3 RGB channels (drop alpha)
    first_frame = rendered_images[0, ..., :3].detach().cpu().numpy()

    # Clip values to [0, 1] to prevent matplotlib warnings upon saving
    first_frame = np.clip(first_frame, 0.0, 1.0)

    plt.imsave(output_path, first_frame)
    print(f"Saved debug frame to '{output_path}'")


if __name__ == "__main__":
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define batch parameters
    N = 180  # Number of frames for one full rotation

    # Load base mesh
    asteroid_mesh = load_asteroid_mesh(model_num=3, device=compute_device)

    print(f"Number of vertices: {asteroid_mesh.verts_packed().shape[0]}")
    print(f"Number of faces: {asteroid_mesh.faces_packed().shape[0]}")

    # 1. Calculate bounding box and centroid
    verts = asteroid_mesh.verts_packed()
    min_verts = verts.min(dim=0).values
    max_verts = verts.max(dim=0).values
    centroid = (min_verts + max_verts) / 2.0
    extent = (max_verts - min_verts).max()

    # 2. Extend the base mesh to a batch of size N
    batched_meshes = asteroid_mesh.extend(N)

    # 3. Apply the batched transformations (rotation around Y-axis)
    from pytorch3d.transforms import Transform3d, axis_angle_to_matrix

    angles = torch.linspace(0, 4 * np.pi, steps=N, device=compute_device)
    axes = torch.zeros(N, 3, device=compute_device)
    axes[:, 1] = 1.0  # Set rotation around the Y-axis

    # Generate batched rotation matrices
    R_batched = axis_angle_to_matrix(axes * angles.unsqueeze(1))

    # Initialize the Transform3d object with the batched rotations
    transforms = Transform3d(device=compute_device).rotate(R_batched)

    # Extract padded vertices from the batched meshes (Shape: N, V, 3)
    verts_padded = batched_meshes.verts_padded()

    # Apply the rotation transform to the vertices
    verts_rotated = transforms.transform_points(verts_padded)

    # Rebuild the batched meshes with the correctly rotated vertices
    rotated_batched_meshes = batched_meshes.update_padded(verts_rotated)

    # 4. Position fixed camera relative to the centroid
    target_position = centroid.cpu().numpy()
    distance = extent.item() * 2.0
    e_sun = target_position + np.array([-distance, 0.0, 0.0])
    plane_scale = (2.0 / extent.item()) * 0.9

    sun_camera = create_orthographic_camera(
        eye_pos=e_sun,
        target_pos=target_position,
        device=compute_device,
        focal_length=plane_scale,
    )

    # 5. Render the batch configuration
    offset_light_direction = [[1.0, 1.0, 1.0]]
    resolution = 32

    rendered_images = render_mesh_batch(
        meshes=rotated_batched_meshes,
        camera=sun_camera,
        device=compute_device,
        image_size=resolution,
        light_direction=offset_light_direction,
    )

    save_first_frame(rendered_images, output_path="first_frame.png")

    # 6. Generate Lightcurve (Summing pixel intensities across the spatial dimensions)
    # rendered_images is of shape (N, H, W, 4). Slice the RGB channels, convert to grayscale/sum
    intensity_curve = rendered_images[..., :3].sum(dim=(1, 2, 3)).cpu().numpy()

    # Plot the resulting synthesized light curve
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, 360, N), intensity_curve, color="blue", linewidth=1.5)
    plt.title("Synthesized Intensity Curve (Lommel-Seeliger Model)")
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Total Reflected Intensity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("simulated_lightcurve.png", bbox_inches="tight")
    print("Saved lightcurve plot to 'simulated_lightcurve.png'")
