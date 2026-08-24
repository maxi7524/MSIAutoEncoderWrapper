"""Reusable annotation mapping support for model datasets."""

from .config import AnnotationSettings, AnnotationTargetSettings
from .index import MappedSpectrumAnnotationIndex
from .mixin import AnnotationAwareDatasetMixin

__all__ = [
    "AnnotationAwareDatasetMixin",
    "AnnotationSettings",
    "AnnotationTargetSettings",
    "MappedSpectrumAnnotationIndex",
]
"""Dataset-level annotation mapping and selection contracts."""

from .config import AnnotationSettings, AnnotationTargetSettings
from .index import MappedSpectrumAnnotationIndex
from .mixin import AnnotationAwareDatasetMixin

__all__ = [
    "AnnotationAwareDatasetMixin",
    "AnnotationSettings",
    "AnnotationTargetSettings",
    "MappedSpectrumAnnotationIndex",
]
