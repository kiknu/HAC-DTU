try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from stl import mesh as stl_mesh
import trimesh
import pyvista as pv
pv.set_jupyter_backend('static')
from scipy.optimize import minimize

from hac.data import get_model_stl_path, load_data
from hac.constants import CAMERA_ANGLE_ALPHA
from hac.convex_forward import get_unit_vectors#, convex_forward_model_gpu
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.eval import calculate_voxel_measure

from pathlib import Path
from hac.minkowski import minkowski_reconstruct, build_convex_mesh, save_stl

CAMERA_ANGLE_ALPHA_REM_DUPLICATES: dict[str, float] = {
    "0_hor1": 0.0, 
    "0_top": 21.0,
    "0_bottom": -21.0,
    "45_hor1": 0.0,
    "45_top": 26.0,
    "45_bottom": -24.0,
    "90_hor1": 0.0,
    "90_top": 26.0,
    "90_bottom": -24.0,
    "135_hor1": 0.0,
    "135_top": 26.0,
    "135_bottom": -24.0,
    "225_hor1": 0.0,
    "225_top": 24.0,
    "225_bottom": -26.0,
    "270_hor1": 0.0,
    "270_top": 24.0,
    "270_bottom": -26.0,
    "315_hor1": 0.0,
    "315_top": 24.0,
    "315_bottom": -26.0
}

class ConvexInversion:

    def __init__(self, cylinder_radius=1.05, max_faces = 1500):
        
        # initialising mesh with ellipsoid approximation of bounding cylinder
        self.cylinder_radius = cylinder_radius

        subdiv = np.floor(np.log(max_faces/20)/np.log(4)).astype(int)
        if subdiv < 0:
            subdiv = 0
        sphere_mesh = trimesh.creation.icosphere(subdivisions=subdiv)
        #sphere_mesh = trimesh.creation.box()
        print(f"Using {len(sphere_mesh.faces)} faces")
        sphere_bounded = normalize_mesh_to_challenge_cylinder(sphere_mesh,target_R=cylinder_radius)
        vertices = sphere_bounded.vertices
        faces = sphere_bounded.faces

        V = np.array(vertices)
        F = np.array(faces)

        # Face normals and areas
        e1 = V[F[:,1]] - V[F[:,0]]
        e2 = V[F[:,2]] - V[F[:,0]]
        cross = np.cross(e1, e2)
        mag   = np.linalg.norm(cross, axis=1)
        self.normals = cross / mag[:, None]
        self.areas_init = 0.5 * mag 
        self.n_facets = len(self.areas_init)

    def load_HAC_data(self, model_num, origin="simulated", curve_type="intensity"):
        """
        Load the lightcurve data for a specific asteroid model, origin, and curve type.
        Parameters:
        - model_num: The asteroid model number.
        - origin: The origin of the data, either "real" or "simulated".
        - curve_type: The type of lightcurve data, either "binary" or "intensity".

        Loads the normalised lightcurvedata along with observation geometry and times.
        In self:
        - self.obs_norm: Normalized lightcurve data (n_cams, n_phases)
        - self.e_sun: Sun direction vector
        - self.CAM_DICT: Dictionary mapping camera keys to their corresponding angles
        - self.cam_keys: List of camera keys corresponding to the lightcurve data
        - self.phases: Array of rotational phases corresponding to the lightcurve data
        """
        lightcurve_ds = load_data(model_num, origin, curve_type)
        if origin == "simulated":
            self.CAM_DICT = CAMERA_ANGLE_ALPHA_REM_DUPLICATES
            self.cam_keys   = list(self.CAM_DICT.keys())
            self.n_cams     = len(self.cam_keys)
            self.n_phases   = 360
            self.phases     = np.linspace(0, 2 * np.pi, self.n_phases, endpoint=False)
        elif origin == "real":
            #error: real data not yet supported
            raise NotImplementedError("Real data is not yet supported in this implementation.")

        self.obs_norm = np.empty((self.n_cams, self.n_phases))
        for i, cam in enumerate(self.cam_keys):
            lc = lightcurve_ds.data[cam].values.astype(np.float64)
            self.obs_norm[i] = lc# / lc.mean() # skip normalization since this is already done

        print(f"Observations: {self.n_cams} cameras × {self.n_phases} phases = {self.n_cams * self.n_phases} data points")
        self.e_sun = np.array([-1.0, 0.0, 0.0])
        self.asteroid = model_num
        return 
    
    def build_forward(self, lambert_weight=0.1):
        """
        Build the forward model for the convex inversion problem.
        Parameters:
        - lambert_weight: The weight for the Lambertian reflection component.
        """
        # check if data is loaded
        if not hasattr(self, 'obs_norm'):
            raise ValueError("Lightcurve data not loaded.")
        
        cos_p = np.cos(self.phases)  
        sin_p = np.sin(self.phases)            

        A_blocks = []
        for cam in self.cam_keys:
            phi_deg   = float(cam.split("_")[0])
            alpha_deg = self.CAM_DICT[cam]
            _, e_obs  = get_unit_vectors(phi_deg, alpha_deg)
            e_obs     = np.asarray(e_obs, dtype=np.float64)

            # Rotate sun & observer through rotational phase (z-axis rotation)
            e_sun_rot = np.column_stack([
                cos_p * self.e_sun[0] - sin_p * self.e_sun[1],
                sin_p * self.e_sun[0] + cos_p * self.e_sun[1],
                np.full(self.n_phases, self.e_sun[2])
            ])  # (n_phases, 3)
            e_obs_rot = np.column_stack([
                cos_p * e_obs[0] - sin_p * e_obs[1],
                sin_p * e_obs[0] + cos_p * e_obs[1],
                np.full(self.n_phases, e_obs[2])
            ])  # (n_phases, 3)

            mu0  = e_sun_rot @ self.normals.T   # (n_phases, n_facets)
            mu   = e_obs_rot @ self.normals.T   # (n_phases, n_facets)
            mask = (mu0 > 0) & (mu > 0)

            # LSL scattering law (eq. 2 of doi:10.48550/arXiv.2502.16455)
            # np.maximum avoids 0/0 when both mu and mu0 are zero (mask=False there anyway)
            S = np.where(mask, mu * mu0 / np.maximum(mu + mu0, 1e-30) + lambert_weight * mu * mu0, 0.0)
            A_blocks.append(S)

        self.A      = np.vstack(A_blocks)              # (n_cams * n_phases, n_facets)
        self.A_3d   = self.A.reshape(self.n_cams, self.n_phases, self.n_facets)
        self.A_mean = self.A_3d.mean(axis=1)               # (n_cams, n_facets)  — per-camera mean

        print(f"Forward matrix A: {self.A.shape}  ({self.A.nbytes / 1e6:.1f} MB)")
        return
    
    def cost_and_grad(self, a, lambda_s, _use_gpu):
        g    = self._xp.exp(self._xp.asarray(a, dtype=self._xp.float64))
        Ag   = self._A @ g
        Ag3d = Ag.reshape(self.n_cams, self.n_phases)
        m    = Ag3d.mean(axis=1)

        norm_model = Ag3d / m[:, None]
        res        = self._obs_norm - norm_model                  # (n_cams, n_phases)

        cost = float(0.5 * self._xp.dot(res.ravel(), res.ravel()))
        if _use_gpu and hasattr(cost, 'get'):
            cost = cost.get()

        term1 = self._A.T @ (res / m[:, None]).ravel()           # (n_facets,)
        term2 = self._A_mean.T @ (self._xp.sum(norm_model * res, axis=1) / m)  # (n_facets,)
        grad  = -g * (term1 - term2)

        conv_vec = self._normals.T @ g                            # (3,)
        cost    += float(lambda_s * self._xp.dot(conv_vec, conv_vec))
        grad    += 2.0 * lambda_s * g * (self._normals @ conv_vec)

        if _use_gpu:
            return float(cost), grad.get().astype(np.float64)
        return float(cost), np.asarray(grad, dtype=np.float64)
    
    def fit_areas(self, convexity_regularization=0.2, optimizer = 'cg', max_iter=20000, gtol = 1e-8, avoid_gpu=False):

        # Move matrices to GPU once; fall back to NumPy if CuPy is unavailable.
        if CUPY_AVAILABLE and not avoid_gpu:
            self._xp       = cp
            self._A        = cp.asarray(self.A)
            self._A_3d     = cp.asarray(self.A_3d)
            self._A_mean   = cp.asarray(self.A_mean)
            self._obs_norm = cp.asarray(self.obs_norm)
            self._normals  = cp.asarray(self.normals)
        else:
            self._xp       = np
            self._A        = self.A
            self._A_3d     = self.A_3d
            self._A_mean   = self.A_mean
            self._obs_norm = self.obs_norm
            self._normals  = self.normals
            print("CuPy unavailable — using NumPy CPU backend")

        _use_gpu = (self._xp is not np)

        if optimizer not in ["lbfgs", "cg"]:
            raise ValueError(f"Invalid optimizer '{optimizer}'. Choose 'lbfgs' or 'cg'.")
        
        sci_method = {"lbfgs": "L-BFGS-B", "cg": "CG"}[optimizer]

        print(f"Running {sci_method}  (n_params={self.n_facets})...")

        def func(a):
            return self.cost_and_grad(a=a, lambda_s=convexity_regularization, _use_gpu=_use_gpu)

        _iter = [0]
        def _cb(a):
            _iter[0] += 1
            if _iter[0] % 1000 == 0:
                c, _ = func(a)
                print(f"  iter {_iter[0]:4d}: cost = {c:.6f}")
        a_init = np.log(self.areas_init)
        result = minimize(
            func,
            a_init,
            jac=True,          # function returns (f, grad) tuple
            method=sci_method,
            callback=_cb,
            options={"maxiter": max_iter, "gtol": gtol},
        )
        self.a_opt = result.x
        self.areas_opt = np.exp(self.a_opt)
        print(f"\n{result.message}")
        print(f"Cost: {result.fun:.6f}")
        return
    
    def lightcurve_fit(self):
        """
        Show the lightcurve fit between the observed normalized lightcurve data and the predicted normalized lightcurve data from the convex inversion model.
        """
        Ag_opt     = (self.A @ self.areas_opt).reshape(self.n_cams, self.n_phases)
        pred_norm  = Ag_opt / Ag_opt.mean(axis=1, keepdims=True)
        phase_axis = np.linspace(0, 1, self.n_phases, endpoint=False)

        fig, axes = plt.subplots(5, 3, figsize=(16, 20), sharex=True)
        for ax, cam, i in zip(axes.ravel(), self.cam_keys, range(self.n_cams)):
            ax.plot(phase_axis, self.obs_norm[i],  lw=1.2, color="steelblue", label="blender")
            ax.plot(phase_axis, pred_norm[i], lw=1.2, color="tomato",    label="inversion", ls="--")
            ax.set_title(cam, fontsize=8)
            ax.set_yticks([])

        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Asteroid {self.asteroid} — lightcurve fit (inversion vs blender)", y=1.002)
        fig.tight_layout()
        plt.show()

        print(f"Overall MSE (normalised): {np.mean((self.obs_norm - pred_norm)**2):.6f}")
        return
    
    def refine_areas(self, small_tolerance=None):
        """
        Adds the dark area to satisfy convexity constraint. Optionally removes small areas.
        small_tolerance: If provided, areas smaller than this fraction of the mean area will be removed. Only to be used if Minkowski method is struggling.
        """
        areas_refined = self.areas_opt.copy()
        normals_refined = self.normals.copy()
        if small_tolerance is not None:
            mean_area = np.mean(areas_refined)
            keep_mask = areas_refined > (small_tolerance * mean_area)
            areas_refined = areas_refined[keep_mask]
            normals_refined = normals_refined[keep_mask]
            print(f"Removed {np.sum(~keep_mask)} small areas below {small_tolerance*100:.2f}% of mean area.")
        total_area = np.sum(areas_refined)
        
        closure = normals_refined.T @ areas_refined
        dark_area = np.linalg.norm(closure)
        if dark_area > total_area * 1e-10:  # Only add dark area if it's significant
            areas_refined = np.concatenate([areas_refined, np.array([dark_area])])
            normals_refined = np.vstack([normals_refined, -closure / dark_area])
            self.dark_area_percentage = 100 * dark_area / (total_area + dark_area)
            print(f"Added dark area. Dark area percentage: {self.dark_area_percentage:.6f}%.\nIf this is large (>1%), recompute the areas with larger convexity regularization.")
        else: 
            print("No significant dark area needed.")

        self.areas_refined = areas_refined
        self.normals_refined = normals_refined
        return
    
    def minkowski_reconstruction(self):
        """
        Reconstruct the convex shape using the Minkowski method based on the refined areas and normals.
        Returns the vertices and faces of the reconstructed convex mesh.
        """
        if not hasattr(self, 'areas_refined') or not hasattr(self, 'normals_refined'):
            raise ValueError("Refined areas and normals not computed. Call refine_areas() first.")

        vertices_cloud = minkowski_reconstruct(self.areas_refined, self.normals_refined, verbose=True, epsilon=0.005)
        mesh_verts, mesh_faces = build_convex_mesh(vertices_cloud)
        print(f"Reconstructed convex mesh: {len(mesh_verts)} vertices, {len(mesh_faces)} faces")

        # normalize to bounding cylinder
        vx = mesh_verts[:, 0]
        vy = mesh_verts[:, 1]
        max_radius = np.sqrt(vx**2 + vy**2).max()

        self.recon_vertices = mesh_verts*(self.cylinder_radius / max_radius)
        self.recon_faces = mesh_faces
        return self.recon_vertices, self.recon_faces

    def plot_reconstruction(self, show=True):
        """
        Plot the reconstructed convex mesh using PyVista.
        """
        if not hasattr(self, 'recon_vertices') or not hasattr(self, 'recon_faces'):
            raise ValueError("Reconstructed mesh not available. Call minkowski_reconstruction() first.")

        mesh = pv.PolyData(self.recon_vertices, np.hstack([np.full((len(self.recon_faces), 1), 3), self.recon_faces]))
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        plotter.add_axes()
        plotter.show_grid()
        if show:
            plotter.show()

            

        
                        

