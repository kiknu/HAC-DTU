from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObserverGeometry:
    """
    Represents the observer geometry.

    See: `workspaces/jeppe/overview.pdf` section "Geometry" for details.
    """
    # Angle around -z axis where 0 degrees is aligned with -x axis.
    # Referred to as 'measurement angle' in FIPS HAC documentation.
    # Measured in degrees.
    theta: float

    # Elevation angle from the xy-plane.
    # Referred to as the 'top camera angle' in FIPS HAC documentation.
    # Measured in degrees.
    alpha: float

    # Distance from the asteroid to the observer.
    # For orthographic projections this does not matter,
    # as long as it is outside the asteroid volume.
    r_obs: float
