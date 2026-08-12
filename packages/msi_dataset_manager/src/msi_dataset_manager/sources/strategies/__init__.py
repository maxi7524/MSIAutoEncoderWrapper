"""Bundled external database strategies."""

from .metaspace import MetaspaceDatasetSource
from .pride import PrideDatasetSource

__all__ = ["MetaspaceDatasetSource", "PrideDatasetSource"]
