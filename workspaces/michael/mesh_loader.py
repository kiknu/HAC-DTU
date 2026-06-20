import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    HardFlatShader,
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

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
    eye_pos: np.ndarray, device: torch.device
) -> OrthographicCameras:
    """
    Creates a PyTorch3D orthographic camera looking at the origin.

    Args:
        eye_pos (np.ndarray): The (X, Y, Z) position of the camera.
        device (torch.device, optional): The compute device.

    Returns:
        OrthographicCameras: The configured camera object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Format the coordinate vector for PyTorch3D
    eye_tensor = torch.from_numpy(eye_pos).float().view(1, 3).to(device)

    # Calculate the rotation (R) and translation (T) matrices.
    # The camera looks at the origin (0, 0, 0) with the Y-axis (0, 1, 0) pointing 'up'.
    R, T = look_at_view_transform(
        eye=eye_tensor, at=((-50.0, 0.0, 0.0),), up=((0.0, 10.0, 0.0),)
    )

    # Initialize the orthographic camera.
    # Note: If the mesh is large, adjusting the focal_length (e.g., focal_length=1.0/mesh_radius)
    # may be required to keep the projection within the [-1, 1] NDC bounds.
    camera = OrthographicCameras(device=device, R=R, T=T)

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

    # Setup a simple light source facing the object
    lights = PointLights(device=device, location=[[0.0, 0.0, -100.0]])

    # Create a renderer combining rasterizer and shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardFlatShader(device=device, cameras=camera, lights=lights),
    )

    # Render the image
    images = renderer(mesh)

    return images[0]


if __name__ == "__main__":
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the sun camera direction/position
    e_sun = np.array([-100.0, 0.0, 0.0])

    # Load the PyTorch3D mesh
    asteroid_mesh = load_asteroid_mesh(model_num=1, device=compute_device)

    # Initialize the camera setup
    sun_camera = create_orthographic_camera(e_sun, device=compute_device)

    # Project the vertices into screen space
    projected_vertices = project_mesh_vertices(asteroid_mesh, sun_camera)

    # Render the mesh to an NxN image
    resolution = 32
    rendered_image = render_mesh(
        asteroid_mesh, sun_camera, image_size=resolution, device=compute_device
    )

    print("Camera initialized with properties:")
    print(f"R matrix: {sun_camera.R.shape}")
    print(f"T vector: {sun_camera.T.shape}\n")
    print("Projection Results:")
    print(
        f"Original 3D Coordinates (first vertex): {asteroid_mesh.verts_packed()[0].cpu().numpy()}"
    )
    print(
        f"Projected NDC Coordinates (first vertex): {projected_vertices[0].cpu().numpy()}"
    )

    # Display and save the rendered image
    image_np = rendered_image.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np[..., :3])  # Display RGB channels, ignore alpha
    plt.title(f"Orthographic Projection ({resolution}x{resolution})")
    plt.axis("off")

    output_filename = "asteroid_projection.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Saved projection image to '{output_filename}'")
