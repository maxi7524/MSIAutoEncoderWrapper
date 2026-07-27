"""Validation of records returned by external database strategies."""

from __future__ import annotations

from typing import Any, Mapping

from ...utils.exceptions import raise_validation_error


def validate_source_record(record: Mapping[str, Any]) -> None:
    """Validate one source record before canonical normalization.

    :param record: Record returned by an external database adapter.
    :type record: Mapping[str, Any]
    :raises ValidationError: If stable identity or metadata is unavailable.
    """
    if not record.get("dataset_id"):
        raise_validation_error("DatasetSource", "A source record requires dataset_id.")
    if not isinstance(record.get("metadata", {}), Mapping):
        raise_validation_error("DatasetSource", "metadata must be a mapping.")
