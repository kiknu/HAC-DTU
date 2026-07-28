"""
lightcurve.py

A compact Python prototype for asteroid lightcurve inversion.

Dependencies:
    numpy
    scipy
    trimesh
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.special import lpmv
from trimesh.creation import icosphere
from numba import njit


def normalize(v, axis=-1, eps=1e-15):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)

def normals_to_spherical(normals):
    """Convert unit normals to spherical coordinates (theta, phi)."""
    x, y, z = normals.T
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi

def rotation_about_z(angle):
    """Rotation matrix about body-frame z axis."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

def lommel_seeliger(mu0, mu, eps=1e-12):
    """Simple Lommel-Seeliger-like scattering law."""
    return mu0 * mu / np.maximum(mu0 + mu, eps)

def mixed_lommel_lambert(mu0, mu, c=0.1, eps=1e-12):
    """
    Simple mixed scattering law:
        (1-c) * Lommel-Seeliger + c * Lambert
    """
    ls = lommel_seeliger(mu0, mu, eps=eps)
    lambert = mu0 * mu
    return ls + c * lambert

@njit(cache=True, fastmath=True)
def ray_triangle_intersection(origin, direction, v0, v1, v2, eps=1e-10):

    edge1 = v1 - v0
    edge2 = v2 - v0

    hx = direction[1]*edge2[2] - direction[2]*edge2[1]
    hy = direction[2]*edge2[0] - direction[0]*edge2[2]
    hz = direction[0]*edge2[1] - direction[1]*edge2[0]

    a = edge1[0]*hx + edge1[1]*hy + edge1[2]*hz

    if abs(a) < eps:
        return -1.0

    f = 1.0 / a

    s0 = origin[0] - v0[0]
    s1 = origin[1] - v0[1]
    s2 = origin[2] - v0[2]

    u = f*(s0*hx + s1*hy + s2*hz)

    if u < 0.0 or u > 1.0:
        return -1.0

    qx = s1*edge1[2] - s2*edge1[1]
    qy = s2*edge1[0] - s0*edge1[2]
    qz = s0*edge1[1] - s1*edge1[0]

    v = f*(direction[0]*qx + direction[1]*qy + direction[2]*qz)

    if v < 0.0 or u + v > 1.0:
        return -1.0

    t = f*(edge2[0]*qx + edge2[1]*qy + edge2[2]*qz)

    if t > eps:
        return t

    return -1.0


@njit
def any_blocked(
    origin,
    direction,
    blockers,
    faces,
    vertices
):
    for k in range(len(blockers)):
        b = blockers[k]

        f = faces[b]

        t_hit = ray_triangle_intersection(
            origin,
            direction,
            vertices[f[0]],
            vertices[f[1]],
            vertices[f[2]]
        )

        if t_hit > 0.0:
            return True

    return False



def build_sh_basis(theta, phi,
                   sh_degree,
                   sh_order,
                   include_l00=False):
    """
    Build real spherical-harmonic basis matrix.

    Returns
    -------
    basis : ndarray, shape (Nvertex, Ncoef)
    """

    basis = []

    costheta = np.cos(theta)

    for m in range(sh_order + 1):
        for l in range(m, sh_degree + 1):

            if (l == 0 and m == 0) and not include_l00:
                continue

            P = lpmv(m, l, costheta)

            if m == 0:

                basis.append(P)

            else:

                basis.append(
                    P * np.cos(m * phi)
                )

                basis.append(
                    P * np.sin(m * phi)
                )

    basis = np.column_stack(basis)

    return basis

def soft_max(x, alpha=50):
    x0 = np.max(x)
    return x0 + np.log(
        np.mean(
            np.exp(alpha * (x - x0))
        )
    ) / alpha

def soft_min(x, alpha=50):
    return -soft_max(-x, alpha)

@dataclass
class LCResult:
    vertices: np.ndarray
    faces: np.ndarray
    residuals: np.ndarray
    model_flux: np.ndarray
    parameters: np.ndarray
    success: bool
    message: str
    cost: float

class LightcurveInverter:
    """
    no comment
    """

    def __init__(
        self,
        HACdata = None,
        sun_dir=None,
        obs_dirs=None,
        subdivisions=3,
        init_vertices=None,
        init_faces=None,
        scattering="mixed",
        scattering_c=0.1
    ):
        self.times = HACdata[:, 0]
        self.period = self.times[-1]
        self.flux = HACdata[:, 1:]
        self.numLC = self.flux.shape[1]
        self.sun_dir = normalize(np.asarray(sun_dir, dtype=float))
        self.obs_dirs = normalize(np.asarray(obs_dirs, dtype=float))
        self.scattering = scattering
        self.scattering_c = scattering_c

        # only initialized if fit_radii is called
        self.sh_basis = None 
        self.rotations = None
        self.dvdc = None  # derivative of vertices w.r.t. spherical harmonic coefficients
        self.cache = {
        'c': None,
        'model_flux': None,
        'jac': None
        }

        # setting up vertex positions
        if init_vertices is not None and init_faces is not None:
            self.vertices = np.asarray(init_vertices, dtype=float)
            self.faces = np.asarray(init_faces, dtype=int)
        else:
            # initial icosphere
            mesh = icosphere(subdivisions=subdivisions, radius=1.0)
            self.vertices = np.asarray(mesh.vertices).copy()
            self.faces = np.asarray(mesh.faces).copy()
        self.vertex_normalized = normalize(self.vertices)
        self.theta, self.phi = normals_to_spherical(self.vertex_normalized)
        print(f"Initialized with {len(self.faces)} faces.")

        # precompute vertex to face look up
        self.vertex_to_face = []
        for i in range(len(self.vertices)):
            self.vertex_to_face.append(np.where(self.faces == i)[0])

        if self.sun_dir.shape != (3,):
            raise ValueError("sun_dir must have shape (3,)")
        if self.obs_dirs.shape != (self.numLC, 3):
            raise ValueError("obs_dirs must have shape (numLC, 3)")

    def _scatter(self, mu0, mu):
        if self.scattering == "lommel_seeliger":
            return lommel_seeliger(mu0, mu)
        if self.scattering == "mixed":
            return mixed_lommel_lambert(mu0, mu, c=self.scattering_c)
        raise ValueError(f"Unknown scattering law: {self.scattering}")

    def brightness_and_jacobian(self):
        """Compute model brightness for all epochs from vertices and faces."""
        centroids = np.mean(self.vertices[self.faces], axis=1)
        num_faces = len(centroids)
        e1 = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]]
        e2 = self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]]
        cross = np.cross(
            e1, e2
        )
        crossnorm = np.linalg.norm(cross, axis=1)
        areas = 0.5 * crossnorm
        normals = normalize(cross)

        de1dc = self.dvdc[self.faces[:, 1]] - self.dvdc[self.faces[:, 0]]
        de2dc = self.dvdc[self.faces[:, 2]] - self.dvdc[self.faces[:, 0]]
        de1dc = np.moveaxis(de1dc, 1, -1)
        de2dc = np.moveaxis(de2dc, 1, -1)

        dcrossdc = (
            np.cross(de1dc, e2[:, None, :])
            +
            np.cross(e1[:, None, :], de2dc)
        )

        dAdc = 0.5 * np.einsum(
            'fi,fki->fk',
            normals,
            dcrossdc
        )
                
        proj = (
            np.einsum(
                'fi,fki->fk',
                normals,
                dcrossdc
            )
        )
        dndc = (
            dcrossdc
            -
            normals[:, None, :] * proj[:, :, None]
        )

        dndc /= crossnorm[:, None, None]



        ncoef = self.dvdc.shape[-1]

        jac = np.zeros(
            (
                len(self.times),
                self.numLC,
                ncoef
            )
        )


        # precomputing potential blockers for each face
        blockers = []
        ishull = [] # list of faces that are on the convex hull (no blockers)
        for i in range(num_faces):

            ci = centroids[i]
            ni = normals[i]

            visible_vertices = np.where(
                (self.vertices - ci) @ ni > 0.0
            )[0]

            ishull.append(len(visible_vertices) == 0)

            local_blockers = set()

            for v in visible_vertices:
                local_blockers.update(self.vertex_to_face[v])

            local_blockers.discard(i)

            # filter blockers that are not facing the face
            filtered_blockers = []

            for b in local_blockers:

                cb = centroids[b]
                nb = normals[b]

                if np.dot(ci - cb, nb) > 0.0:
                    filtered_blockers.append(b)

            blockers.append(
                np.array(filtered_blockers, dtype=int)
            )

        sun_t = self.sun_dir
        obs_t = self.obs_dirs
        brightness = np.zeros_like(self.flux)
        for k,t in enumerate(self.times):

            mu0 = np.maximum(normals @ sun_t, 0.0)
            mu = np.maximum(normals @ obs_t.T, 0.0)

            # derivative of mu 
            mu0raw = normals @ sun_t            
            dmu0dc = np.einsum(
                'fki,i->fk',
                dndc,
                sun_t
            )
            dmu0dc[mu0raw <= 0] = 0.0
            muraw = normals @ obs_t.T            
            dmudc = np.einsum(
                'fki,ji->fjk',
                dndc,
                obs_t
            )
            dmudc[muraw <= 0,:] = 0.0

            
            illuminated = mu0 > 0.0
            visible = mu > 0.0

            flux_contribution = np.empty(self.numLC)
            for i in range(num_faces):

                flux_contribution.fill(0.0)

                # Lommel-Seeliger scattering law
                mu0i = mu0[i]
                mui = mu[i]                
                den = np.maximum(
                    mu0i + mui,
                    1e-12
                )
                ls = mu0i * mui / den
                lsl = ls + self.scattering_c * mu0i * mui

                # scattering derivatives
                dls_dmu0 = mui**2 / den**2
                dls_dmu = mu0i**2 / den**2
                dlsl_dmu0 = (
                    dls_dmu0
                    + self.scattering_c * mui
                )
                dlsl_dmu = (
                    dls_dmu
                    + self.scattering_c * mu0i
                )
                dlsldc = (
                    dlsl_dmu0[:, None]
                    * dmu0dc[i][None, :]
                ) + (
                    dlsl_dmu[:, None]
                    * dmudc[i]
                )


                # flux derivative w.r.t. spherical harmonic coefficients                
                dfluxdc = (
                    lsl[:, None]
                    * dAdc[i][None, :]
                )+(
                    areas[i]
                    * dlsldc
                )


                # precompute ray origin
                origin = centroids[i] + 1e-10 * normals[i]

                if illuminated[i]: # only compute flux contribution if face is illuminated
                    if ishull[i]:
                        flux_contribution[:] = areas[i] * lsl
                        jac[k] += dfluxdc
                    else:
                        # first check solar illumination
                        blocked = any_blocked(
                            origin,
                            sun_t,
                            blockers[i],
                            self.faces,
                            self.vertices
                        )
                        if not blocked:
                            for j in range(self.numLC):
                                if visible[i, j]:
                                    blocked = any_blocked(
                                        origin,
                                        obs_t[j],
                                        blockers[i],
                                        self.faces,
                                        self.vertices
                                    )
                                    if not blocked:
                                        flux_contribution[j] = areas[i] * lsl[j]
                                        jac[k,j,:] += dfluxdc[j]
                                    else:
                                        flux_contribution[j] = 0.0
                                else:
                                    flux_contribution[j] = 0.0
                brightness[k] += flux_contribution
            rot = self.rotations[k]
            sun_t = self.sun_dir @ rot.T
            obs_t = self.obs_dirs @ rot.T
        
        mean_brightness = np.mean(brightness, axis=0)
        brightness_norm = brightness / mean_brightness
        dmean_dc = np.mean(
            jac,
            axis=0
        )
        jac_norm = (
            jac / mean_brightness[None,:,None]
            -
            brightness[:,:,None]
            * dmean_dc[None,:,:]
            / mean_brightness[None,:,None]**2
        )
        return brightness_norm, jac_norm 
    
    def write_to_file(self, filename):
        """Write vertices and faces to a text file."""
        with open(filename, 'w') as f:
            f.write(f"{len(self.vertices)} {len(self.faces)}\n")
            for vertex in self.vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face in self.faces:
                f.write('3\n')
                f.write(f"{face[0]} {face[1]} {face[2]}\n")
    
    def fit_radii(
            self,
            sh_degree=6,
            sh_order=6,
            init_shape=None, # tuple of (vertices, faces) for initial convex shape containing origo
            phase0=0.0,
            max_nfev=500,
            verbose=0,
            shape_penalty=None,
            sparsity_regularization=None,
            cylinder_radius=None,
            include_l00=False, # if shape penalty is not none then must be True
            ftol=None
        ):
        """Fit spherical harmonic coefficients for radii."""
        self.sh_basis = build_sh_basis(
            self.theta, self.phi, 
            sh_degree, sh_order, include_l00
        )
        
        ncoef = self.sh_basis.shape[1]
        print(f"Fitting {ncoef} spherical harmonic coefficients")
        if init_shape is None:
            c_init = np.zeros(ncoef) # initial spherical harmonic coefficients
            if include_l00:
                c_init[0] = np.log(1.0)
        else:
            init_vertices, init_faces = init_shape
            normals = []
            offset = []
            for i in range(len(init_faces)):
                v0 = init_vertices[init_faces[i][0]]
                v1 = init_vertices[init_faces[i][1]]
                v2 = init_vertices[init_faces[i][2]]
                n = np.cross(v1 - v0, v2 - v0)
                n /= np.linalg.norm(n)
                h = np.dot(n, v0)
                normals.append(n)
                offset.append(h)
            normals = np.array(normals)
            offset = np.array(offset)
            denom = self.vertex_normalized @ normals.T 
            t = np.full_like(denom, np.inf)
            np.divide(
                offset[None, :],
                denom,
                out=t,
                where=denom > 1e-6
            )
            radii = np.min(t, axis=1)
            g_init = np.log(radii)
            c_init, _, _, _ = np.linalg.lstsq(self.sh_basis, g_init, rcond=None)

        # precompute rotation matrices for each epoch
        self.rotations = []
        for t in self.times:
            angle = 2.0 * np.pi * t / self.period + phase0
            rot = rotation_about_z(angle)
            self.rotations.append(rot)
        
        def spherical_harmonic_to_shape(c):

            g = self.sh_basis @ c

            radii = np.exp(g)

            self.vertices = self.vertex_normalized * radii[:, None]

            drdc = radii[:, None] * self.sh_basis
            self.dvdc = self.vertex_normalized[:, :, None] * drdc[:, None, :]

            return 
        
        if init_shape is not None:
            spherical_harmonic_to_shape(c_init)
            self.write_to_file("initial_shape")
        
        def residual_function(c):
            residuals = []

            # check cache
            if self.cache['c'] is not None and np.array_equal(c, self.cache['c']):
                model_flux = self.cache['model_flux']
                if shape_penalty is not None:
                    spherical_harmonic_to_shape(c) # for use in penalty residuals
            else:
                # update vertices from spherical harmonic coefficients
                spherical_harmonic_to_shape(c)
                model_flux, jac = self.brightness_and_jacobian()
                self.cache['c'] = c.copy()
                self.cache['model_flux'] = model_flux.copy()
                self.cache['jac'] = jac.copy()

            data_resid = (model_flux - self.flux).ravel()
            residuals.append(data_resid)

            if shape_penalty is not None:
                # add shape penalty to residuals
                if cylinder_radius is None:
                    raise ValueError("cylinder_radius must be provided for shape_penalty")
                # projections of vertices onto z-axis and radius in xy-plane
                vz = self.vertices[:, 2]
                vr = np.linalg.norm(self.vertices[:, :2], axis=1)
                height = soft_max(vz) - soft_min(vz)
                penalty = [
                    np.sqrt(shape_penalty) * (height - 2),
                    np.sqrt(shape_penalty) * (soft_max(vr) - cylinder_radius)
                ]
                residuals.append(penalty)

            if sparsity_regularization is not None:
                # add sparsity regularization to residuals
                sparsity_penalty = np.sqrt(sparsity_regularization) * np.abs(c)
                residuals.append(sparsity_penalty)

            return np.concatenate(residuals)
        
        def jacobian_function(c):
            Jacs = []
            # check cache
            if self.cache['c'] is not None and np.array_equal(c, self.cache['c']):
                jac = self.cache['jac']
                if shape_penalty is not None:
                    spherical_harmonic_to_shape(c) # for use in penalty jacobian
            else:
                # update vertices from spherical harmonic coefficients
                spherical_harmonic_to_shape(c)
                model_flux, jac = self.brightness_and_jacobian()
                self.cache['c'] = c.copy()
                self.cache['model_flux'] = model_flux.copy()
                self.cache['jac'] = jac.copy()

            data_jac = jac.reshape(-1, ncoef)
            Jacs.append(data_jac)

            if shape_penalty is not None:
                eps = 1e-6

                penalty_jac = np.zeros((2, ncoef))

                for k in range(ncoef):

                    cp = c.copy()
                    cm = c.copy()

                    cp[k] += eps
                    cm[k] -= eps

                    # ----- plus perturbation -----

                    spherical_harmonic_to_shape(cp)

                    vz = self.vertices[:, 2]
                    vr = np.linalg.norm(self.vertices[:, :2], axis=1)

                    hp = soft_max(vz) - soft_min(vz)
                    rp = soft_max(vr)

                    # ----- minus perturbation -----

                    spherical_harmonic_to_shape(cm)

                    vz = self.vertices[:, 2]
                    vr = np.linalg.norm(self.vertices[:, :2], axis=1)

                    hm = soft_max(vz) - soft_min(vz)
                    rm = soft_max(vr)

                    penalty_jac[0, k] = (
                        np.sqrt(shape_penalty)
                        * (hp - hm)
                        / (2 * eps)
                    )

                    penalty_jac[1, k] = (
                        np.sqrt(shape_penalty)
                        * (rp - rm)
                        / (2 * eps)
                    )

                # restore original shape
                spherical_harmonic_to_shape(c)
                Jacs.append(penalty_jac)

            if sparsity_regularization is not None:
                sparsity_jac = np.sqrt(sparsity_regularization) * np.diag(np.sign(c))
                Jacs.append(sparsity_jac)
            
            J = np.vstack(Jacs)
            return J

        if ftol is None: ftol = 1e-8
        sol = least_squares(fun=residual_function, x0=c_init, jac=jacobian_function, method="trf", max_nfev=max_nfev, ftol=ftol, verbose=verbose)
        
        # Find final shape
        g = self.sh_basis @ sol.x
        radii = np.exp(g)
        if not include_l00:
            radii /= np.mean(radii)
        self.vertices = self.vertex_normalized * radii[:, None]
        if shape_penalty is not None:
            # check shape constraints
            vz = self.vertices[:, 2]
            vr = np.linalg.norm(self.vertices[:, :2], axis=1)
            print(f"Bounding cylinder [{np.min(vz):.3f}, {np.max(vz):.3f}] x radius {np.max(vr):.3f}")
        
        model_flux, _ = self.brightness_and_jacobian()

        return LCResult(
            vertices=self.vertices,
            faces=self.faces,
            residuals=model_flux - self.flux,
            model_flux=model_flux,
            parameters=sol.x,
            success=sol.success,
            message=sol.message,
            cost=sol.cost,
        )
    
    '''def fit_radii_direct(
            self,
            phase0=0.0,
            max_nfev=500,
            verbose=0,
            shape_penalty=None,
            cylinder_radius=None,
            include_l00=False,
            ftol=None
        ):

        # precompute rotation matrices for each epoch
        self.rotations = []
        for t in self.times:
            angle = 2.0 * np.pi * t / self.period + phase0
            rot = rotation_about_z(angle)
            self.rotations.append(rot)
        
        def logradii_to_shape(g):

            radii = np.exp(g)

            self.vertices = self.vertex_normalized * radii[:, None]

            drdc = radii[:, None] * self.sh_basis
            self.dvdc = self.vertex_normalized[:, :, None] * drdc[:, None, :]

            return 
        
        def residual_function(c):
            residuals = []

            # check cache
            if self.cache['c'] is not None and np.array_equal(c, self.cache['c']):
                model_flux = self.cache['model_flux']
                if shape_penalty is not None:
                    spherical_harmonic_to_shape(c) # for use in penalty residuals
            else:
                # update vertices from spherical harmonic coefficients
                spherical_harmonic_to_shape(c)
                model_flux, jac = self.brightness_and_jacobian()
                self.cache['c'] = c.copy()
                self.cache['model_flux'] = model_flux.copy()
                self.cache['jac'] = jac.copy()

            data_resid = (model_flux - self.flux).ravel()
            residuals.append(data_resid)

            if shape_penalty is not None:
                # add shape penalty to residuals
                if cylinder_radius is None:
                    raise ValueError("cylinder_radius must be provided for shape_penalty")
                # projections of vertices onto z-axis and radius in xy-plane
                vz = self.vertices[:, 2]
                vr = np.linalg.norm(self.vertices[:, :2], axis=1)
                height = soft_max(vz) - soft_min(vz)
                penalty = [
                    np.sqrt(shape_penalty) * (height - 2),
                    np.sqrt(shape_penalty) * (soft_max(vr) - cylinder_radius)
                ]
                residuals.append(penalty)

            return np.concatenate(residuals)
        
        def jacobian_function(c):
            # check cache
            if self.cache['c'] is not None and np.array_equal(c, self.cache['c']):
                jac = self.cache['jac']
                if shape_penalty is not None:
                    spherical_harmonic_to_shape(c) # for use in penalty jacobian
            else:
                # update vertices from spherical harmonic coefficients
                spherical_harmonic_to_shape(c)
                model_flux, jac = self.brightness_and_jacobian()
                self.cache['c'] = c.copy()
                self.cache['model_flux'] = model_flux.copy()
                self.cache['jac'] = jac.copy()

            data_jac = jac.reshape(-1, ncoef)

            if shape_penalty is None:
                return data_jac

            eps = 1e-6

            penalty_jac = np.zeros((2, ncoef))

            for k in range(ncoef):

                cp = c.copy()
                cm = c.copy()

                cp[k] += eps
                cm[k] -= eps

                # ----- plus perturbation -----

                spherical_harmonic_to_shape(cp)

                vz = self.vertices[:, 2]
                vr = np.linalg.norm(self.vertices[:, :2], axis=1)

                hp = soft_max(vz) - soft_min(vz)
                rp = soft_max(vr)

                # ----- minus perturbation -----

                spherical_harmonic_to_shape(cm)

                vz = self.vertices[:, 2]
                vr = np.linalg.norm(self.vertices[:, :2], axis=1)

                hm = soft_max(vz) - soft_min(vz)
                rm = soft_max(vr)

                penalty_jac[0, k] = (
                    np.sqrt(shape_penalty)
                    * (hp - hm)
                    / (2 * eps)
                )

                penalty_jac[1, k] = (
                    np.sqrt(shape_penalty)
                    * (rp - rm)
                    / (2 * eps)
                )

            # restore original shape
            spherical_harmonic_to_shape(c)
            
            J = np.vstack([
                data_jac,
                penalty_jac
            ])
            return J

        if ftol is None: ftol = 1e-8
        sol = least_squares(fun=residual_function, x0=c_init, jac=jacobian_function, method="trf", max_nfev=max_nfev, ftol=ftol, verbose=verbose)
        
        # Find final shape
        g = self.sh_basis @ sol.x
        radii = np.exp(g)
        if not include_l00:
            radii /= np.mean(radii)
        self.vertices = self.vertex_normalized * radii[:, None]
        
        model_flux, _ = self.brightness_and_jacobian()

        return LCResult(
            vertices=self.vertices,
            faces=self.faces,
            residuals=model_flux - self.flux,
            model_flux=model_flux,
            success=sol.success,
            message=sol.message,
            cost=sol.cost,
        )'''