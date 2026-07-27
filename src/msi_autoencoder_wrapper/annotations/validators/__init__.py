"""Validators for the canonical annotation interface."""

from .annotation_store_validator import validate_annotation_store
from .annotation_validator import validate_annotation_record

__all__ = ["validate_annotation_record", "validate_annotation_store"]
