"""Synthetic spectral pretraining on the selected model axis."""

from .sampling import (
    SyntheticComponent,
    SyntheticSampleDefinition,
    SyntheticSamplingContext,
    SyntheticSamplingManager,
    SyntheticSamplingPlanEntry,
    SyntheticSamplingStrategy,
    get_sampling_strategy,
    register_sampling_strategy,
    sampling_strategy_names,
)
from .sources import (
    CandidateCatalogPeakSource,
    CandidateCatalogPeakSourceError,
    CataloguePeakSource,
    SyntheticPeakSource,
)
from .synthetic import (
    SyntheticSpectrumConfig,
    SyntheticSpectrumDataset,
    SyntheticSpectrumSample,
    build_synthetic_partitions,
)

__all__ = [
    "CataloguePeakSource",
    "CandidateCatalogPeakSource",
    "CandidateCatalogPeakSourceError",
    "SyntheticComponent",
    "SyntheticPeakSource",
    "SyntheticSampleDefinition",
    "SyntheticSamplingContext",
    "SyntheticSamplingManager",
    "SyntheticSamplingPlanEntry",
    "SyntheticSamplingStrategy",
    "SyntheticSpectrumConfig",
    "SyntheticSpectrumDataset",
    "SyntheticSpectrumSample",
    "build_synthetic_partitions",
    "get_sampling_strategy",
    "register_sampling_strategy",
    "sampling_strategy_names",
]
