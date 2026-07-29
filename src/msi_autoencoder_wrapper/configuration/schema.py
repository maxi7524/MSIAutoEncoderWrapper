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
    if config.get("schema_version") != 1:
        raise_validation_error(
            "Configuration",
            f"Unsupported root schema version '{config.get('schema_version')}'.",
        )
    local_context = config.get("local_image_context")
    loaded_context = config.get("loaded_model_context")
    if not isinstance(local_context, dict):
        raise_validation_error(
            "Configuration", "A local_image_context section is required."
        )
    if not isinstance(loaded_context, dict):
        raise_validation_error(
            "Configuration", "A loaded_model_context section is required."
        )
    if not isinstance(local_context.get("components"), dict):
        raise_validation_error(
            "Configuration", "local_image_context.components must be a dictionary."
        )
    if not isinstance(loaded_context.get("model"), dict):
        raise_validation_error(
            "Configuration", "loaded_model_context.model must be a dictionary."
        )
