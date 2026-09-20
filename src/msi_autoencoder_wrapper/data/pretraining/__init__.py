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
from .annotation_population import AnnotationPeakRecord, AnnotationPopulation
from .representations import (
    SyntheticRepresentationContext,
    SyntheticRepresentationManager,
    SyntheticRepresentationSpec,
    SyntheticRepresentationStrategy,
    get_representation_strategy,
    register_representation_strategy,
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
from .precomputed import (
    PrecomputedPopulationSpec,
    PrecomputedSyntheticConfig,
    PrecomputedSyntheticDataset,
    SyntheticArtifactStore,
    SyntheticManifest,
    SyntheticPrecomputeArtifact,
    SyntheticPrecomputeBuilder,
    build_precomputed_synthetic_partitions,
)

__all__ = [
    "AnnotationPeakRecord",
    "AnnotationPopulation",
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
    "SyntheticRepresentationContext",
    "SyntheticRepresentationManager",
    "SyntheticRepresentationSpec",
    "SyntheticRepresentationStrategy",
    "SyntheticSpectrumConfig",
    "SyntheticSpectrumDataset",
    "SyntheticSpectrumSample",
    "PrecomputedPopulationSpec",
    "PrecomputedSyntheticConfig",
    "PrecomputedSyntheticDataset",
    "SyntheticArtifactStore",
    "SyntheticManifest",
    "SyntheticPrecomputeArtifact",
    "SyntheticPrecomputeBuilder",
    "build_precomputed_synthetic_partitions",
    "build_synthetic_partitions",
    "get_sampling_strategy",
    "get_representation_strategy",
    "register_representation_strategy",
    "register_sampling_strategy",
    "sampling_strategy_names",
]
