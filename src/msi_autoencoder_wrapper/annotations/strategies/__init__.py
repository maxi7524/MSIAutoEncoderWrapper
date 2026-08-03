"""Built-in annotation-reader strategies."""

from .metaspace_csv_annotation_reader import MetaspaceCSVAnnotationReader
from .sqlite_annotation_reader import SQLiteAnnotationReader

__all__ = ["MetaspaceCSVAnnotationReader", "SQLiteAnnotationReader"]
