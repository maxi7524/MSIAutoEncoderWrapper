"""Provider-independent spatial image values returned by MSI readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..utils.exceptions import raise_validation_error


Aggregation = str | Callable[[np.ndarray], float]


@dataclass(frozen=True)
class SpatialImage:
    """Store values mapped to the native MSI coordinate grid.

    :param values: Spatial values ordered as ``(z, y, x)``.
    :type values: numpy.ndarray
    :param valid_mask: Boolean grid identifying positions backed by spectra.
    :type valid_mask: numpy.ndarray
    :param extent: Inclusive native coordinate bounds ``(min_x, max_x, min_y,
        max_y, min_z, max_z)``.
    :type extent: tuple[int, int, int, int, int, int]
    """

    values: np.ndarray
    valid_mask: np.ndarray
    extent: tuple[int, int, int, int, int, int]


def aggregate_window(values: np.ndarray, aggregation: Aggregation) -> float:
    """Aggregate one non-empty intensity window.

    :param values: One-dimensional selected intensity values.
    :type values: numpy.ndarray
    :param aggregation: ``mean``, ``sum``, ``max``, ``median``, or a callable.
    :type aggregation: str | Callable[[numpy.ndarray], float]
    :return: Aggregated scalar intensity.
    :rtype: float
    :raises ValidationError: If the aggregation strategy is unsupported.
    """
    if callable(aggregation):
        return float(aggregation(values))
    functions = {
        "mean": np.mean,
        "sum": np.sum,
        "max": np.max,
        "median": np.median,
    }
    function = functions.get(aggregation)
    if function is None:
        raise_validation_error(
            "SpatialAggregation",
            "aggregation must be 'mean', 'sum', 'max', 'median', or a callable.",
        )
    return float(function(values))
