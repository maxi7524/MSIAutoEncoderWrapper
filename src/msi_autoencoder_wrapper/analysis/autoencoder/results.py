"""In-memory result containers for autoencoder analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class PreparationEstimate:
    """Estimated memory required by one prepared analysis.

    :param sample_count: Number of spectra that will be processed.
    :type sample_count: int
    :param retained_bytes: Estimated bytes grouped by retained result name.
    :type retained_bytes: Dict[str, int]
    """

    sample_count: int
    retained_bytes: Dict[str, int]

    @property
    def total_bytes(self) -> int:
        """Return the estimated total retained size in bytes.

        :return: Sum of estimated result sizes.
        :rtype: int
        """
        return sum(self.retained_bytes.values())

    @property
    def total_mebibytes(self) -> float:
        """Return the estimated total retained size in MiB.

        :return: Estimated size in mebibytes.
        :rtype: float
        """
        return self.total_bytes / (1024**2)


@dataclass
class PreparedAnalysis:
    """Cache produced by a single explicit model traversal.

    Large matrices are retained only when requested. Lightweight per-spectrum
    and per-feature statistics are always available.

    :param spectrum_ids: Stable identifiers in result-row order.
    :type spectrum_ids: numpy.ndarray
    :param pixel_metrics: Scalar metrics indexed by stable spectrum identifier.
    :type pixel_metrics: Dict[int, Dict[str, float]]
    :param feature_metrics: Per-feature reconstruction statistics.
    :type feature_metrics: Dict[str, numpy.ndarray]
    :param arrays: Optional input, reconstruction, and latent matrices.
    :type arrays: Dict[str, numpy.ndarray]
    :param head_outputs: Optional logits grouped by head identifier.
    :type head_outputs: Dict[str, numpy.ndarray]
    :param targets: Optional target arrays grouped by target field.
    :type targets: Dict[str, numpy.ndarray]
    :param target_masks: Optional availability masks grouped by target field.
    :type target_masks: Dict[str, numpy.ndarray]
    """

    spectrum_ids: np.ndarray
    pixel_metrics: Dict[int, Dict[str, float]]
    feature_metrics: Dict[str, np.ndarray]
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    head_outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    targets: Dict[str, np.ndarray] = field(default_factory=dict)
    target_masks: Dict[str, np.ndarray] = field(default_factory=dict)

    def metric_array(self, metric: str) -> np.ndarray:
        """Return one pixel metric in row order.

        :param metric: Metric name stored in ``pixel_metrics``.
        :type metric: str
        :return: Metric values aligned with ``spectrum_ids``.
        :rtype: numpy.ndarray
        :raises KeyError: If the requested metric is unavailable.
        """
        return np.asarray(
            [self.pixel_metrics[int(index)][metric] for index in self.spectrum_ids],
            dtype=np.float32,
        )


@dataclass
class MultiPreparedAnalysis:
    """Prepared caches for models evaluated against one shared dataset.

    Input arrays, targets, and target masks are stored once and referenced by
    every per-model cache, preventing linear duplication of immutable data.

    :param models: Per-model prepared results.
    :type models: Mapping[str, PreparedAnalysis]
    :param shared_inputs: Optional shared binned input matrix.
    :type shared_inputs: numpy.ndarray | None
    :param shared_targets: Shared targets grouped by field.
    :type shared_targets: Dict[str, numpy.ndarray]
    :param shared_target_masks: Shared availability masks grouped by field.
    :type shared_target_masks: Dict[str, numpy.ndarray]
    """

    models: Mapping[str, PreparedAnalysis]
    shared_inputs: np.ndarray | None = None
    shared_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    shared_target_masks: Dict[str, np.ndarray] = field(default_factory=dict)

    def for_model(self, model_name: str) -> PreparedAnalysis:
        """Return one model's prepared cache.

        :param model_name: Analyzed model identifier.
        :type model_name: str
        :return: Prepared model result.
        :rtype: PreparedAnalysis
        :raises KeyError: If the model is not part of this analysis.
        """
        return self.models[model_name]
