"""Visualization views over prepared autoencoder analysis data."""

from .latent import plot_projection
from .reconstruction import (
    plot_metric_distribution,
    plot_spatial_image,
    plot_spectrum_comparison,
)

__all__ = [
    "plot_metric_distribution",
    "plot_projection",
    "plot_spatial_image",
    "plot_spectrum_comparison",
]
