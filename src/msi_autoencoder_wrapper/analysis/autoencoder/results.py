"""In-memory result containers for autoencoder analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

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
            dtype=np.float64,
        )
