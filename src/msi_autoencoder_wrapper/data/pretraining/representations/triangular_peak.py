"""Legacy triangular-peak renderer for synthetic spectra."""

from __future__ import annotations

import numpy as np

from .base import (
    SyntheticRepresentationContext,
    SyntheticRepresentationStrategy,
    register_representation_strategy,
)


@register_representation_strategy("triangular_peak")
class TriangularPeakRepresentation(SyntheticRepresentationStrategy):
    """Render each declared component as a triangular profile in binner space."""

    def __init__(self, peak_radius: int = 0, minimum_intensity: float = 0.1) -> None:
        """Configure the legacy profile and independent component amplitudes.

        :param peak_radius: Half-width of the triangular profile in bins.
        :type peak_radius: int
        :param minimum_intensity: Inclusive lower bound before the unit upper
            bound of the component-amplitude uniform distribution.
        :type minimum_intensity: float
        """
        if isinstance(peak_radius, bool) or not isinstance(peak_radius, int) or peak_radius < 0:
            raise ValueError("peak_radius must be a nonnegative integer.")
        if not 0.0 < minimum_intensity <= 1.0:
            raise ValueError("minimum_intensity must belong to (0, 1].")
        self.peak_radius = peak_radius
        self.minimum_intensity = float(minimum_intensity)

    def render(self, rng, context, definition):
        """Render a dense binner-space mixture, shape ``(M,)``."""
        spectrum = np.zeros(context.feature_count, dtype=np.float64)  # (M,)
        for component in definition.components:
            center = component.center
            left = max(0, center - self.peak_radius)
            right = min(context.feature_count, center + self.peak_radius + 1)
            positions = np.arange(left, right)  # (W,)
            profile = 1 - np.abs(positions - center) / (self.peak_radius + 1)  # (W,)
            amplitude = rng.uniform(self.minimum_intensity, 1.0)  # ()
            spectrum[left:right] += component.intensity_weight * amplitude * profile  # (W,)
        return spectrum
