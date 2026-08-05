"""Model-independent numerical metrics organized by object space."""

from .base import BaseMetric, ClassMetric, ClassificationMetric, MetricRequirements, SpectrumMetric
from .compatibility import validate_metric_compatibility, validate_metric_inputs
from .registry import MetricDefinition, MetricsRegistry, MetricsRunner
from .spectral_points import (
    SpectralPointMatch,
    match_spectral_points,
    peak_collision_rate,
    signal_retention_by_quantile,
    spectral_point_metrics,
)
from .strategies.classification import binary_cross_entropy, cross_entropy
from .strategies.classes import PerClassClassification
from .strategies.embedding import info_nce
from .strategies.masserstein import SpectrumMasserstein
from .strategies.spectrum import (
    cosine_similarity,
    feature_errors,
    mae,
    mse,
    sobolev,
    spectral_angle,
    tic_error,
)

__all__ = [
    "BaseMetric",
    "ClassMetric",
    "ClassificationMetric",
    "MetricDefinition",
    "MetricRequirements",
    "MetricsRegistry",
    "MetricsRunner",
    "PerClassClassification",
    "SpectrumMetric",
    "SpectralPointMatch",
    "SpectrumMasserstein",
    "binary_cross_entropy",
    "cross_entropy",
    "cosine_similarity",
    "feature_errors",
    "info_nce",
    "mae",
    "match_spectral_points",
    "mse",
    "peak_collision_rate",
    "signal_retention_by_quantile",
    "sobolev",
    "spectral_angle",
    "spectral_point_metrics",
    "tic_error",
    "validate_metric_compatibility",
    "validate_metric_inputs",
]

for _name, _space, _implementation in (
    ("mse", "spectrum", mse),
    ("mae", "spectrum", mae),
    ("sobolev", "spectrum", sobolev),
    ("cosine_similarity", "spectrum", cosine_similarity),
    ("spectral_angle", "spectrum", spectral_angle),
    ("tic_error", "spectrum", tic_error),
    ("masserstein", "spectrum", SpectrumMasserstein),
    ("cross_entropy", "classification", cross_entropy),
    ("binary_cross_entropy", "classification", binary_cross_entropy),
    ("f1", "class", PerClassClassification),
    ("info_nce", "embedding", info_nce),
):
    MetricsRegistry.register(_name, _space, _implementation)

del _implementation, _name, _space
