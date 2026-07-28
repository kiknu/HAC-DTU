#%%
import numpy as np
from lightcurveinv import LightcurveInverter
import matplotlib.pyplot as plt
#%%
coords = np.loadtxt('asteroiddata/coords.txt', delimiter=' ')
obs_dir = coords[:, 3:]
sun_dir = np.array([1, 0, 0])

cylinder_R = np.array([1.12, 1.42, 0.88, 1.475, 1.22, 0.925, 1.205, 1.24, 0.67, 3.95])

cube_vertices = np.array([
    [-1, -1, -1],  # 0
    [ 1, -1, -1],  # 1
    [ 1,  1, -1],  # 2
    [-1,  1, -1],  # 3
    [-1, -1,  1],  # 4
    [ 1, -1,  1],  # 5
    [ 1,  1,  1],  # 6
    [-1,  1,  1],  # 7
], dtype=float)

cube_faces = [
    [0, 3, 2, 1],  # z = -1
    [4, 5, 6, 7],  # z = +1
    [0, 4, 7, 3],  # x = -1
    [1, 2, 6, 5],  # x = +1
    [0, 1, 5, 4],  # y = -1
    [3, 7, 6, 2],  # y = +1
]
cube_shape = (cube_vertices, cube_faces)

asteroid_numbers = [2]

for asteroid_number in asteroid_numbers:
    HAC_data = np.loadtxt(f'asteroiddata/Asteroid0{asteroid_number}_lightcurve_intensity_blender.txt', delimiter=',')

    LC = LightcurveInverter(HACdata=HAC_data, obs_dirs=obs_dir, sun_dir=sun_dir, subdivisions=3)

    LCresult = LC.fit_radii(sh_degree=10, sh_order=10, max_nfev=500, verbose=2, ftol=1e-4, sparsity_regularization=1, shape_penalty=None, cylinder_radius=cylinder_R[int(asteroid_number)-1],include_l00=False)
    LC.write_to_file(f'radii_shapes/A{asteroid_number}_b_int_shape_radii')
    print(LCresult.optimization_result.x)
