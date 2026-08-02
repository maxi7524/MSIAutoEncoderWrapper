"""Compatibility checks between metric requirements and normalization traces."""

from __future__ import annotations

from .base import MetricRequirements
from ..normalization import NormalizationTrace
from ..utils.exceptions import raise_validation_error


def validate_metric_compatibility(
    requirements: MetricRequirements,
    trace: NormalizationTrace | None,
) -> None:
    """Validate that the current intensity representation is meaningful.

    :param requirements: Mathematical requirements declared by the metric.
    :type requirements: MetricRequirements
    :param trace: Applied normalization trace, or ``None`` for source values.
    :type trace: NormalizationTrace | None
    :raises ValidationError: If normalization invalidates metric semantics.
    """
    if trace is None:
        if requirements.required_space == "normalized":
            raise_validation_error("Metric", "The metric requires normalized inputs.")
        return
    capabilities = trace.capabilities
    if requirements.required_space == "source":
        raise_validation_error("Metric", "The metric requires source-space inputs.")
    if requirements.requires_nonnegative and not capabilities.preserves_nonnegativity:
        raise_validation_error("Metric", "Normalization does not preserve nonnegative intensities.")
    if requirements.requires_linear_intensity and not capabilities.preserves_linear_intensity:
        raise_validation_error("Metric", "Normalization does not preserve linear intensity semantics.")
    if not requirements.accepts_samplewise_scalar and capabilities.samplewise_scalar:
        raise_validation_error("Metric", "The metric does not accept samplewise scalar normalization.")
