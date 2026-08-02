"""Validated output activation configuration for spectrum decoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import torch.nn as nn

from ......utils.exceptions import raise_validation_error


SUPPORTED_OUTPUT_ACTIVATIONS: Dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}
"""Output activations accepted in decoder configuration."""


def build_output_activation(config: Mapping[str, Any]) -> nn.Module:
    """Build a decoder output activation from a validated configuration.

    :param config: Mapping containing ``type`` and optional ``parameters``.
    :type config: Mapping[str, Any]
    :return: Configured PyTorch activation module.
    :rtype: torch.nn.Module
    :raises ValidationError: If the configuration or activation parameters are invalid.
    """
    if not isinstance(config, Mapping):
        raise_validation_error(
            "OutputActivation",
            "output_activation must be a mapping with 'type' and 'parameters'.",
        )

    activation_name = config.get("type")
    if not isinstance(activation_name, str) or activation_name not in SUPPORTED_OUTPUT_ACTIVATIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_ACTIVATIONS))
        raise_validation_error(
            "OutputActivation",
            f"Unsupported output activation '{activation_name}'. Supported values: {supported}.",
        )

    parameters = config.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise_validation_error(
            "OutputActivation",
            "output_activation.parameters must be a mapping.",
        )

    activation_type = SUPPORTED_OUTPUT_ACTIVATIONS[activation_name]
    try:
        return activation_type(**dict(parameters))
    except (TypeError, ValueError) as error:
        raise_validation_error(
            "OutputActivation",
            f"Invalid parameters for '{activation_name}': {error}",
        )
