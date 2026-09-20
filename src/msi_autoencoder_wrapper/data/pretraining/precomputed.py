"""Public API for persistent synthetic pretraining artifacts.

The implementation is divided by responsibility into configuration/builder,
artifact storage, and batch-time rendering modules.  Importing from this
module preserves the original public interface.
"""

from .precompute_artifact import (
    SyntheticArtifactStore,
    SyntheticManifest,
    SyntheticPrecomputeArtifact,
)
from .precompute_builder import (
    PrecomputedPopulationSpec,
    PrecomputedSyntheticConfig,
    SyntheticPrecomputeBuilder,
    build_precomputed_synthetic_partitions,
    precompute_request_fingerprint,
)
from .precompute_dataset import PrecomputedSyntheticDataset, _component_weights

__all__ = [
    "PrecomputedPopulationSpec",
    "PrecomputedSyntheticConfig",
    "PrecomputedSyntheticDataset",
    "SyntheticArtifactStore",
    "SyntheticManifest",
    "SyntheticPrecomputeArtifact",
    "SyntheticPrecomputeBuilder",
    "build_precomputed_synthetic_partitions",
    "precompute_request_fingerprint",
]
