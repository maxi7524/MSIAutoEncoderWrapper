"""Training-layer orchestration for persistent synthetic precompute artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..data.pretraining.precomputed import (
    PrecomputedSyntheticConfig,
    build_precomputed_synthetic_partitions,
)
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class SyntheticPrecomputePhase:
    """Resolve one explicit artifact-backed synthetic training population.

    This object is intentionally not an optimizer phase. It is the boundary
    between phase orchestration and model-independent synthetic data creation:
    it validates the request, loads or creates the persistent artifact, then
    returns ordinary train and validation datasets for the trainer.
    """

    parameters: Mapping[str, Any]

    def build_partitions(self, dataset: Any) -> dict[str, Any]:
        """Load or construct artifact-backed partitions for one training phase.

        :param dataset: Real dataset providing the axis, train split, schemas,
            and optional candidate catalogue.
        :type dataset: Any
        :return: Synthetic train/validation/test partition mapping.
        :rtype: dict[str, Any]
        """
        config = PrecomputedSyntheticConfig.from_mapping(self.parameters)
        logger.info(
            "Preparing synthetic precompute population: artifact=%s population=%s.",
            config.artifact_key,
            config.selected_population,
        )
        return build_precomputed_synthetic_partitions(dataset, self.parameters)
