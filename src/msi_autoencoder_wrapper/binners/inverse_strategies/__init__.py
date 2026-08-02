"""Inverse binner strategy registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

discover_modules(__name__, recursive=False)
from .cumulative_mass import CumulativeMassInverseBinner
from .peak_region import PeakRegionInverseBinner
from .threshold import ThresholdInverseBinner
from .top_peaks import TopPeaksInverseBinner

__all__ = ["CumulativeMassInverseBinner", "PeakRegionInverseBinner", "ThresholdInverseBinner", "TopPeaksInverseBinner"]
