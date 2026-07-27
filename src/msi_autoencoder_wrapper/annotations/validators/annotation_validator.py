"""Validation for canonical molecular annotation records."""

from __future__ import annotations

from typing import Any, Mapping

from ...utils.exceptions import raise_validation_error


def validate_annotation_record(record: Mapping[str, Any]) -> None:
    """Validate minimal molecule identity in a canonical annotation record.

    :param record: Canonical molecular annotation.
    :type record: Mapping[str, Any]
    :raises ValidationError: If neither a formula nor source annotation ID is
        present.
    """
    formula = record.get("formula", record.get("sumFormula"))
    annotation_id = record.get("annotation_id", record.get("id"))
    if formula is None and annotation_id is None:
        raise_validation_error(
            "AnnotationRecord", "An annotation requires a formula or source annotation ID."
        )
