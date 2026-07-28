import numpy as np

def unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector.")

    return v / norm

def direction_from_angles(xy_deg, elev_deg):
    az = np.deg2rad(xy_deg) + np.pi
    el = np.deg2rad(elev_deg)

    return unit([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])