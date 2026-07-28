#%%
import numpy as np
from convexinversion import ConvexInversion

cylinder_R = np.array([1.12, 1.42, 0.88, 1.475, 1.22, 0.925, 1.205, 1.24, 0.67, 3.95])
asteroid = 4
rad = cylinder_R[asteroid-1]

CI = ConvexInversion(cylinder_radius=rad)

# %%
CI.load_HAC_data(model_num=asteroid,origin="simulated",curve_type="intensity")

# %%
CI.build_forward(lambert_weight=0.1)
# %%
CI.fit_areas(convexity_regularization=0.2,optimizer='cg',max_iter=20000,gtol=1e-8,avoid_gpu=False)

# %%
CI.lightcurve_fit()

# %%
CI.refine_areas()
# %%
vertices, faces = CI.minkowski_reconstruction()
np.savez(f'reconstructions/A{asteroid}.npz', vertices=vertices, faces=faces, dark_area_percentage=CI.dark_area_percentage)
# %%
#CI.plot_reconstruction()
# %%
