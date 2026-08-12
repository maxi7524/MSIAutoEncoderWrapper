"""Configuration and query-selection validation."""

from __future__ import annotations

from typing import Any, Mapping

from ..utils.exceptions import raise_validation_error


def validate_selection(selection: Mapping[str, Any]) -> None:
    """Validate the provider-independent query selection structure.

    :param selection: Parsed query selection.
    :type selection: Mapping[str, Any]
    :raises ValidationError: If required fields are missing or malformed.
    """
    if not selection.get("source"):
        raise_validation_error("DatasetSelection", "source is required.")
    datasets = selection.get("datasets")
    if not isinstance(datasets, list):
        raise_validation_error("DatasetSelection", "datasets must be a list.")
    for record in datasets:
        if not isinstance(record, Mapping) or not record.get("dataset_id"):
            raise_validation_error(
                "DatasetSelection", "Every dataset record requires dataset_id."
            )
