"""Inverse binner strategy registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

discover_modules(__name__, recursive=False)
from .threshold import QuantileInverseBinner
from .top_peaks import TopPeaksInverseBinner, TopPeaksNeighbourhoodInverseBinner
from .statistical import PassthroughInverseBinner, StatisticalInverseBinner

__all__ = [
    "PassthroughInverseBinner",
    "QuantileInverseBinner",
    "StatisticalInverseBinner",
    "TopPeaksInverseBinner",
    "TopPeaksNeighbourhoodInverseBinner",
]
