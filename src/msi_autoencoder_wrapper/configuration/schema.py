"""Validation helpers for consolidated saved configurations."""

from __future__ import annotations

from typing import Any, Dict

from ..utils.exceptions import raise_validation_error


def validate_consolidated_configuration(config: Dict[str, Any]) -> None:
    """Validate sections required to restore a saved experiment.

    :param config: Consolidated configuration dictionary.
    :type config: Dict[str, Any]
    :raises ValidationError: If the root schema or a required section is invalid.
    """
    if not isinstance(config, dict):
        raise_validation_error(
            "Configuration", "The saved configuration must be a dictionary."
        )
    if config.get("schema_version") != 2:
        raise_validation_error(
            "Configuration",
            f"Unsupported root schema version '{config.get('schema_version')}'.",
        )
    experiment = config.get("experiment")
    data = config.get("data")
    if not isinstance(experiment, dict):
        raise_validation_error(
            "Configuration", "An experiment section is required."
        )
    if not isinstance(data, dict):
        raise_validation_error(
            "Configuration", "A data section is required."
        )
    if not isinstance(config.get("model"), dict):
        raise_validation_error(
            "Configuration", "A model section is required."
        )
