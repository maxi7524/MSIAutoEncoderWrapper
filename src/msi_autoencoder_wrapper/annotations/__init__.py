"""Annotation readers independent of external provider formats."""

from .base_annotation_reader import MSIBaseAnnotationReader
from .annotations_manager import AnnotationReaderManager

__all__ = ["AnnotationReaderManager", "MSIBaseAnnotationReader"]
