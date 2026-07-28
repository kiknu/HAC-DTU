import pyvista as pv
import numpy as np

def read_shape(filename):
    with open(filename) as f:
        n_vertices, n_faces = map(int, f.readline().split())

        vertices = np.array([list(map(float, f.readline().split()))
                             for _ in range(n_vertices)])

        faces = []
        for _ in range(n_faces):
            k = int(f.readline())
            indices = list(map(int, f.readline().split()))
            faces.append([k] + indices)

    return vertices, faces

def plot_shape(file, image=False):
    vertices, faces = read_shape(file)

    # Full mesh
    faces_flat = [item for sublist in faces for item in sublist]
    mesh = pv.PolyData(vertices, faces_flat)

    # 
    if image:
        image_file = file + '_im.png'

    # Plot
    plotter = pv.Plotter(off_screen=image)
    plotter.add_mesh(
        mesh,
        show_edges=True
    )

    plotter.add_axes()
    if image:
        plotter.show(screenshot=image_file)
    else:
        plotter.show()
    return

asteroid_number = 2
filename = f'radii_shapes/A{asteroid_number}_b_int_shape_radii'

model = 'dodecahedron'  # folders in synthetic_models: cube, tile, etc.
filename = f'synthetic_models/{model}/shape_radii'
#filename = "radii_shapes/initial_cube_81"

plot_shape(filename, image=False)