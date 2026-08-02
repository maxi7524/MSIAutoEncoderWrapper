"""Annotation readers independent of external provider formats."""

from .base_annotation_reader import MSIBaseAnnotationReader
from .annotations_manager import AnnotationReaderManager
from .strategies.metaspace_csv_annotation_reader import MetaspaceCSVAnnotationReader
from .strategies.sqlite_annotation_reader import SQLiteAnnotationReader

__all__ = [
    "AnnotationReaderManager",
    "MetaspaceCSVAnnotationReader",
    "MSIBaseAnnotationReader",
    "SQLiteAnnotationReader",
]
