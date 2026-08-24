"""Model-output normalization built from shared spectrum strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..utils.exceptions import raise_validation_error
from .strategies import ScalarNormalization


SUPPORTED_OUTPUT_NORMALIZATIONS = {"none", "tic", "max", "l2"}
"""Normalization strategies accepted at spectrum decoder outputs."""


class SpectrumOutputNormalization(nn.Module):
    """Apply a stateless samplewise normalization to decoder spectra.

    The scale produced for a decoder output is intentionally not retained.
    Decoder amplitude before normalization has no source-intensity meaning;
    source-space reconstruction uses the input normalization trace instead.

    :param kind: ``none``, ``tic``, ``max``, or ``l2``.
    :type kind: str
    :param epsilon: Positive denominator threshold.
    :type epsilon: float
    """

    def __init__(self, kind: str = "none", epsilon: float = 1e-12) -> None:
        super().__init__()
        if kind not in SUPPORTED_OUTPUT_NORMALIZATIONS:
            supported = ", ".join(sorted(SUPPORTED_OUTPUT_NORMALIZATIONS))
            raise_validation_error(
                "OutputNormalization",
                f"Unsupported output normalization '{kind}'. Supported values: {supported}.",
            )
        self.kind = kind
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise_validation_error(
                "OutputNormalization", "epsilon must be greater than zero."
            )
        self._strategy = (
            None
            if kind == "none"
            else ScalarNormalization(kind=kind, epsilon=self.epsilon)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return spectra in the configured output representation.

        :param values: Nonnegative decoder spectra with shape ``(B, M)``.
        :type values: torch.Tensor
        :return: Spectra normalized independently over ``M``.
        :rtype: torch.Tensor
        """
        if self._strategy is None:
            return values
        normalized, _ = self._strategy.transform(values)  # (B, M)
        return normalized


def build_output_normalization(
    config: Mapping[str, Any] | None,
) -> SpectrumOutputNormalization:
    """Build a spectrum output normalization from a portable mapping.

    :param config: Mapping containing ``type`` and optional ``parameters``.
    :type config: Mapping[str, Any] | None
    :return: Configured differentiable output normalization.
    :rtype: SpectrumOutputNormalization
    :raises ValidationError: If the configuration is malformed.
    """
    resolved = {"type": "none", "parameters": {}} if config is None else dict(config)
    kind = resolved.get("type")
    parameters = resolved.get("parameters", {})
    if not isinstance(kind, str):
        raise_validation_error(
            "OutputNormalization", "output_normalization.type must be a string."
        )
    if not isinstance(parameters, Mapping):
        raise_validation_error(
            "OutputNormalization",
            "output_normalization.parameters must be a mapping.",
        )
    try:
        return SpectrumOutputNormalization(kind=kind, **dict(parameters))
    except TypeError as error:
        raise_validation_error(
            "OutputNormalization",
            f"Invalid parameters for '{kind}': {error}",
        )
