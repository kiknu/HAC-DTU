from dataclasses import dataclass

from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.patch import mi

import numpy as np

SUN_DIRECTION = np.asarray([1.0, 0.0, 0.0])
ASTEROID_POSITION = np.asarray([0.0, 0.0, 0.0])
UP_DIRECTION = np.asarray([0.0, 0.0, 1.0])
ROTATION_DIRECTION = (
    -UP_DIRECTION
)  # Can be changed later in case we find misalignment in their experimental setup(s).


@dataclass(frozen=True, slots=True)
class RenderingSettings:
    width: int
    height: int
    spp: int


def make_scene(
    mesh: MitsubaMesh,
    theta: float,
    alpha: float,
    phi: float,
    observer_distance: float,
    asteroid_scale: float,
    rendering_settings: RenderingSettings,
    albedo: float,
    sun_direction: np.ndarray = SUN_DIRECTION,
):
    """
    Create a Mitsuba Scene for the asteroid, Sun, and observer (Earth) setup.

    Arguments:
    - mesh: MitsubaMesh object representing the asteroid.
    - theta: Angle around -z axis where 0 degrees is aligned with -x axis.
    - alpha: Elevation angle from the xy-plane.
    - observer_distance: Distance from the asteroid to the observer.
    """
    theta_rad = np.radians(theta)
    alpha_rad = np.radians(alpha)

    observer_position = (
        np.asarray(
            [
                -np.cos(theta_rad) * np.cos(alpha_rad),
                -np.sin(theta_rad) * np.cos(alpha_rad),
                np.sin(alpha_rad),
            ]
        )
        * observer_distance
    )
    observer_transform = mi.Transform4f.look_at(
        origin=observer_position, target=ASTEROID_POSITION, up=UP_DIRECTION
    )

    asteroid_transform = mi.ScalarTransform4f.rotate(
        axis=ROTATION_DIRECTION, angle=phi
    ).scale(asteroid_scale)

    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "direct"},
            "sensor": {
                "type": "orthographic",  # NOTE: Could try `distant`, which is similar to the real physical setup, but does not allow us to do summation methods given in `lightcurve_integration.py`.
                "to_world": observer_transform,
                "sampler": {  # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#low-discrepancy-sampler-ldsampler
                    "type": "independent",
                    "sample_count": rendering_settings.spp,
                },
                "film": {
                    "type": "hdrfilm",
                    "pixel_format": "luminance",
                    "width": rendering_settings.width,
                    "height": rendering_settings.height,
                    "rfilter": {
                        "type": "gaussian",
                    },  # Should probably be changed, see: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html
                },
            },
            "sun": {
                "type": "directional",
                "direction": sun_direction,
                "irradiance": {
                    "type": "rawconstant",
                    "value": 1.0,
                },
            },
            "asteroid": {
                "type": "obj",
                "filename": mesh.to_filename(),
                "face_normals": False,
                "to_world": asteroid_transform,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rawconstant",
                        "value": albedo,  # This is called the 'reflectance' in the Mitsuba documentation, but I believe this is the same as the albedo
                    },
                },
            },
        }
    )
