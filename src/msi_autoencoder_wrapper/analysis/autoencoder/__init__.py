"""Single-image autoencoder analysis tools."""

from .analyzer import AutoencoderAnalysis
from .multi_analyzer import AutoencoderMultiAnalysis
from .results import MultiPreparedAnalysis, PreparationEstimate, PreparedAnalysis
from .visualizations import (
    plot_metric_distribution,
    plot_projection,
    plot_spatial_image,
    plot_spectrum_comparison,
)

__all__ = [
    "AutoencoderAnalysis",
    "AutoencoderMultiAnalysis",
    "MultiPreparedAnalysis",
    "PreparationEstimate",
    "PreparedAnalysis",
    "plot_metric_distribution",
    "plot_projection",
    "plot_spatial_image",
    "plot_spectrum_comparison",
]
