"""Validate canonical annotations exported by provider adapters."""

from __future__ import annotations

from typing import Any, Mapping

from ...utils.exceptions import raise_validation_error


def validate_annotation_record(record: Mapping[str, Any]) -> None:
    """Validate fields required by the provider-independent annotation schema."""
    formula = record.get("formula") or record.get("sumFormula")
    if not formula:
        raise_validation_error("CanonicalAnnotation", "formula is required")
    fdr = record.get("fdr")
    if fdr is not None and not 0 <= float(fdr) <= 1:
        raise_validation_error("CanonicalAnnotation", "fdr must be in [0, 1]")
    spectrum_ids = [int(value) for value in record.get("spectrum_ids") or ()]
    if any(value < 0 for value in spectrum_ids):
        raise_validation_error(
            "CanonicalAnnotation", "spectrum_ids cannot contain negative values"
        )
