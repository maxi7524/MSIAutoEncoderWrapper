"""External database source contracts and strategy registry."""

from typing import Any

from .base import AnnotationDatasetSource, DatasetSource, SourceAnnotationExport
from .profiles import RotatingDatasetSource, read_source_profiles
from .source_manager import DatasetSourceManager

__all__ = [
    "DatasetSource",
    "AnnotationDatasetSource",
    "SourceAnnotationExport",
    "DatasetSourceManager",
    "MetaspaceDatasetSource",
    "PrideDatasetSource",
    "RotatingDatasetSource",
    "read_source_profiles",
]


def __getattr__(name: str) -> Any:
    """Load concrete provider strategies only when explicitly imported."""
    if name == "MetaspaceDatasetSource":
        from .strategies.metaspace import MetaspaceDatasetSource

        return MetaspaceDatasetSource
    if name == "PrideDatasetSource":
        from .strategies.pride import PrideDatasetSource

        return PrideDatasetSource
    raise AttributeError(name)
