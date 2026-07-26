"""Bundled normalized annotation readers."""

from .catalog_annotation_reader import CatalogAnnotationReader
from .merged_catalog_annotation_reader import MergedCatalogAnnotationReader

__all__ = ["CatalogAnnotationReader", "MergedCatalogAnnotationReader"]
