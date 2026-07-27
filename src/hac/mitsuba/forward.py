"""Differentiable forward rendering of asteroid light curves."""

from collections.abc import Mapping
from typing import Any

import drjit as dr

from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.patch import mi
from hac.mitsuba.scene import (
    MIN_PROJECTED_PHASE_CENTER_SEPARATION,
    BatchSceneLayout,
    RenderingSettings,
    make_batch_scene,
)
from hac.types import ObserverGeometry


class ForwardModel:
    """A batch scene together with the mutable mesh parameters it renders."""

    def __init__(
        self,
        mesh: MitsubaMesh,
        *,
        n_phases: int,
        observers: Mapping[str, ObserverGeometry],
        rendering_settings: RenderingSettings,
        normalize_mesh: bool = False,
        bounding_cylinder_radius: float | None = None,
        minimum_projected_phase_center_separation: float = (
            MIN_PROJECTED_PHASE_CENTER_SEPARATION
        ),
    ) -> None:
        # Materialize the mapping so its insertion order remains the sample order
        # throughout scene construction, rendering, and light-curve comparison.
        ordered_observers = dict(observers)
        result = make_batch_scene(
            mesh=mesh,
            n_phases=n_phases,
            observers=ordered_observers,
            rendering_settings=rendering_settings,
            normalize_mesh=normalize_mesh,
            bounding_cylinder_radius=bounding_cylinder_radius,
            minimum_projected_phase_center_separation=(
                minimum_projected_phase_center_separation
            ),
        )

        self.scene = result.scene
        self.layout: BatchSceneLayout = result.layout
        self.params: Any = mi.traverse(self.scene)
        self.n_phases = n_phases
        self.observer_count = len(ordered_observers)
        self.rendering_settings = rendering_settings

        position_keys = [
            key for key in self.params.keys() if key.endswith("vertex_positions")
        ]
        if len(position_keys) != 1:
            raise RuntimeError(
                "Expected exactly one optimizable mesh in the batch scene, "
                f"found {len(position_keys)}."
            )

        self.position_key = position_keys[0]
        self.face_key = f"{self.position_key.removesuffix('vertex_positions')}faces"
        if self.face_key not in self.params:
            raise RuntimeError(
                f"Could not find faces corresponding to {self.position_key!r}."
            )

    def render(self, *, seed: int = 0):
        """Render frames as ``(observer, phase, y, x, channel)``."""
        if seed < 0:
            raise ValueError(f"seed must be non-negative; got {seed}.")

        settings = self.rendering_settings
        sample_count = self.observer_count * self.n_phases
        strip = mi.render(
            self.scene,
            params=self.params,
            spp=settings.spp,
            seed=seed,
        )

        # Mitsuba's batch sensor lays samples out horizontally in one long film.
        # Split that strip and restore the observer-major ordering used to build
        # the scene: all phases for observer 0, then all phases for observer 1.
        frames = dr.transpose(
            dr.reshape(
                strip,
                (settings.height, sample_count, settings.width, 1),
            ),
            axes=(1, 0, 2, 3),
        )
        return dr.reshape(
            frames,
            (
                self.observer_count,
                self.n_phases,
                settings.height,
                settings.width,
                1,
            ),
        )

    def render_lightcurves(self, *, seed: int = 0):
        """Render mean-normalized intensity curves for every observer."""
        lightcurves = dr.sum(self.render(seed=seed), axis=(2, 3, 4))
        observer_means = dr.reshape(
            dr.mean(lightcurves, axis=1),
            (self.observer_count, 1),
        )
        return lightcurves / observer_means
