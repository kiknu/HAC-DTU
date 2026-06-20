import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    DirectionalLights,
    HardFlatShader,
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.structures import Meshes


class LommelSeeligerOrthographicShader(nn.Module):
    """
    Calculates pixel intensities based on the Lommel-Seeliger reflection coefficient.
    Assumes an orthographic camera and directional lighting.
    """

    def __init__(self, device, cameras, lights, albedo=1.0):
        super().__init__()
        self.cameras = cameras
        self.lights = lights
        self.albedo = albedo
        self.device = device

    def forward(self, fragments, meshes, **kwargs):
        # Extract interpolated surface normals (N, H, W, K, 3)
        normals = fragments.normals

        # 1. Compute view direction vector (V)
        # For PyTorch3D OrthographicCameras, rays are parallel.
        # The 3rd column of the rotation matrix R represents the +Z view direction in world space.
        view_dir = self.cameras.R[:, :, 2].view(-1, 1, 1, 1, 3)
        view_dir = F.normalize(view_dir, p=2, dim=-1)

        # 2. Compute light direction vector (L)
        light_dir = self.lights.direction.view(-1, 1, 1, 1, 3)
        light_dir = F.normalize(light_dir, p=2, dim=-1)

        # 3. Calculate cosines of emergence and incidence
        mu = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        mu_0 = torch.sum(normals * light_dir, dim=-1, keepdim=True)

        # 4. Lommel-Seeliger formula application
        # Valid surfaces must be facing both the camera and the light
        valid_mask = (mu > 0) & (mu_0 > 0)

        intensity = torch.zeros_like(mu)
        denominator = mu + mu_0

        # The reflection coefficient R_LS is proportional to 1 / (mu + mu_0).
        # The observed intensity scales by mu_0: Intensity = mu_0 * R_LS
        intensity[valid_mask] = (
            mu_0[valid_mask] / denominator[valid_mask]
        ) * self.albedo

        # Apply calculated intensity to fragments
        pixel_colors = torch.ones_like(normals) * intensity

        # Blend fragments to resolve depth and handle background
        images = hard_rgb_blend(
            pixel_colors, fragments, blend_colors=torch.zeros_like(pixel_colors)
        )

        return images


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHALLENGE_DATA_DIR = PROJECT_DIR / "challenge_data"


def get_model_folder(model_num: int) -> Path:
    """
    Returns a Path object to the folder containing the data for the specified model number.
    """
    _model_folder_suffix = "public" if model_num <= 3 else "secret"
    model_folder_str = f"AsteroidModel{model_num:02d}_shape_{_model_folder_suffix}"

    model_dir = CHALLENGE_DATA_DIR / model_folder_str
    assert model_dir.is_dir(), (
        f"Model data directory not found at {model_dir}. Run `.\\bin\\get_challenge_data.sh` to download it."
    )

    return model_dir


def get_model_stl_path(model_num: int) -> Path:
    """
    Returns a Path object to the STL file containing the shape model for the specified model number.
    """
    if model_num > 3:
        raise ValueError(
            f"Model number {model_num} is a secret model and cannot be accessed."
        )

    model_dir = get_model_folder(model_num)

    stl_file_str = f"asteroid{model_num:d}.stl"
    stl_file_path = model_dir / stl_file_str
    assert stl_file_path.is_file(), (
        f"STL file not found at {stl_file_path}. Run `.\\bin\\get_challenge_data.sh` to download it."
    )

    return stl_file_path


def load_asteroid_mesh(model_num: int, device: torch.device) -> Meshes:
    """
    Loads an asteroid STL file, converts it to OBJ format, and loads it into PyTorch3D.

    Args:
        model_num (int): The model number to load (1-3 for public models).
        device (torch.device, optional): The target device for the PyTorch3D mesh.
            Defaults to CUDA if available, otherwise CPU.

    Returns:
        Meshes: A PyTorch3D Meshes object containing the asteroid geometry.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieve the STL file path using the provided data script
    stl_path = get_model_stl_path(model_num)

    # Load the STL file using the trimesh library
    # force='mesh' ensures it loads as a Trimesh object rather than a Scene
    mesh_trimesh = trimesh.load(stl_path, force="mesh")

    # PyTorch3D's batch loading functions optimally read from OBJ files.
    # A temporary file is used to avoid polluting the dataset folder with duplicate meshes.
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=True) as temp_obj_file:
        # Export the loaded STL geometry into the temporary OBJ file
        mesh_trimesh.export(temp_obj_file.name, file_type="obj")

        # Load the newly created OBJ file directly into a PyTorch3D Meshes object
        # The function expects a list of file paths to construct a batched mesh
        pytorch3d_mesh = load_objs_as_meshes([temp_obj_file.name], device=device)

    return pytorch3d_mesh


def create_orthographic_camera(
    eye_pos: np.ndarray,
    target_pos: np.ndarray,
    device: torch.device,
    focal_length: float = 1.0,
) -> OrthographicCameras:
    """
    Creates a PyTorch3D orthographic camera looking at a specific target.
    """
    eye_tensor = torch.from_numpy(eye_pos).float().view(1, 3).to(device)
    at_tensor = torch.from_numpy(target_pos).float().view(1, 3).to(device)

    R, T = look_at_view_transform(eye=eye_tensor, at=at_tensor, up=((0.0, 1.0, 0.0),))

    camera = OrthographicCameras(device=device, R=R, T=T, focal_length=focal_length)
    return camera


def project_mesh_vertices(mesh, camera: OrthographicCameras) -> torch.Tensor:
    """
    Projects the 3D vertices of a PyTorch3D mesh into the camera's 2D NDC space.
    """
    # Extract the (N, 3) tensor of vertices
    verts = mesh.verts_packed()

    # Transform vertices to Normalized Device Coordinates
    # X and Y represent the 2D orthographic projection; Z represents depth.
    projected_verts = camera.transform_points_ndc(verts)

    return projected_verts


def render_mesh(
    mesh: Meshes,
    camera: OrthographicCameras,
    device: torch.device,
    image_size: int = 32,
    light_direction: list = [[1.0, 0.0, 0.0]],
) -> torch.Tensor:
    """
    Renders the mesh into an NxN image using PyTorch3D's MeshRenderer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add a basic grey texture if the mesh lacks one
    if mesh.textures is None:
        verts_rgb = (
            torch.ones([1, mesh.verts_packed().shape[0], 3], device=device) * 0.7
        )
        mesh.textures = TexturesVertex(verts_features=verts_rgb)

    # Define rasterization settings for an NxN image
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Setup a directional light source (parallel rays)
    # The direction defines the path the light travels.
    lights = DirectionalLights(device=device, direction=light_direction)

    # Create a renderer combining rasterizer and shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardFlatShader(device=device, cameras=camera, lights=lights),
    )

    # Render the image
    images = renderer(mesh)

    return images[0]


if __name__ == "__main__":
    angles = torch.linspace(0, 2 * np.pi, steps=N)

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the PyTorch3D mesh
    asteroid_mesh = load_asteroid_mesh(model_num=1, device=compute_device)

    # ---------------------------------------------------------
    # DYNAMIC FRAMING LOGIC
    # ---------------------------------------------------------
    # 1. Calculate bounding box and centroid
    verts = asteroid_mesh.verts_packed()
    min_verts = verts.min(dim=0).values
    max_verts = verts.max(dim=0).values

    centroid = (min_verts + max_verts) / 2.0
    extent = (max_verts - min_verts).max()

    # 2. Extend the base mesh to a batch of size N
    batched_meshes = asteroid_mesh.extend(N)

    # 3. Apply the batched transformations
    from pytorch3d.transforms import Transform3d

    transforms = Transform3d(device=device).rotate(R_batched)
    rotated_batched_meshes = batched_meshes.update_verts(
        transforms.transform_points(batched_meshes.verts_packed())
    )

    # 2. Position camera relative to the centroid
    target_position = centroid.cpu().numpy()

    # Place eye offset along the X-axis. (Distance strictly prevents Z-clipping)
    distance = extent.item() * 2.0
    e_sun = target_position + np.array([-distance, 0.0, 0.0])

    # 3. Scale orthographic plane to perfectly fit the max dimension
    # Multiplied by a small factor (e.g., 0.9) to add padding around the edges
    plane_scale = (2.0 / extent.item()) * 0.9

    # Initialize the camera setup
    sun_camera = create_orthographic_camera(
        eye_pos=e_sun,
        target_pos=target_position,
        device=compute_device,
        focal_length=plane_scale,
    )

    # ---------------------------------------------------------
    # RENDER CONFIGURATION
    # ---------------------------------------------------------
    # Offset the light direction so it strikes the object at an angle, revealing 3D contours.
    # [1.0, 1.0, 1.0] casts light diagonally across the geometry.
    offset_light_direction = [[1.0, 1.0, 1.0]]

    # Increase resolution to capture surface details
    resolution = 16
    rendered_image = render_mesh(
        mesh=asteroid_mesh,
        camera=sun_camera,
        device=compute_device,
        image_size=resolution,
        light_direction=offset_light_direction,
    )

    # Display and save the rendered image
    image_np = rendered_image.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np[..., :3])
    plt.title(f"Orthographic Projection ({resolution}x{resolution})")
    plt.axis("off")

    output_filename = "asteroid_projection.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Saved projection image to '{output_filename}'")
