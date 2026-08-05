"""Shared validation, thresholding, ranking, and diagnostics for inverse binners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ...utils.exceptions import raise_validation_error


def validate_input(values: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return compatible one-dimensional floating-point intensity and axis arrays.

    :param values: Dense spectrum intensities.
    :type values: numpy.ndarray
    :param axis: Physical coordinates supplied by the forward binner.
    :type axis: numpy.ndarray
    :return: Float64 views of the validated intensities and coordinates.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValidationError: If either input is not one-dimensional or their lengths
        differ.
    """
    intensities = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(axis, dtype=np.float64)
    if intensities.ndim != 1 or coordinates.ndim != 1 or intensities.size != coordinates.size:
        raise_validation_error("InverseBinner", "Axis and intensities must be equal one-dimensional arrays.")
    return intensities, coordinates


def valid_candidate_mask(values: np.ndarray) -> np.ndarray:
    """Return the common finite, strictly positive inverse-selection mask.

    :param values: Dense spectrum intensities.
    :type values: numpy.ndarray
    :return: Boolean mask excluding NaN, infinity, zero, and negative values.
    :rtype: numpy.ndarray
    """
    return np.isfinite(values) & (values > 0.0)


def validate_budget(minimum: int = 0, maximum: int | None = None) -> tuple[int, int | None]:
    """Validate and normalize minimum and maximum selection counts.

    :param minimum: Minimum number of selected candidates.
    :type minimum: int
    :param maximum: Optional maximum number of selected candidates.
    :type maximum: int | None
    :return: Integer selection limits.
    :rtype: tuple[int, int | None]
    :raises ValidationError: If the limits are negative, empty, or reversed.
    """
    minimum = int(minimum)
    maximum = None if maximum is None else int(maximum)
    if minimum < 0 or (maximum is not None and maximum <= 0) or (maximum is not None and minimum > maximum):
        raise_validation_error("InverseBinner", "Selection limits require 0 <= minimum <= maximum and positive maximum.")
    return minimum, maximum


def rank_candidates(indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rank indices by descending weight and ascending index for deterministic ties.

    :param indices: Candidate positions in the dense spectrum.
    :type indices: numpy.ndarray
    :param weights: Ranking weight for every dense spectrum position.
    :type weights: numpy.ndarray
    :return: Candidate indices in selection priority order.
    :rtype: numpy.ndarray
    """
    indices = np.asarray(indices, dtype=np.int64)
    return indices[np.lexsort((indices, -np.asarray(weights)[indices]))]


def resolve_threshold(
    values: np.ndarray,
    strategy: str | Callable[..., Any],
    options: Mapping[str, Any] | None = None,
) -> float | np.ndarray:
    """Resolve a scalar or shape-compatible threshold for one spectrum.

    :param values: Dense spectrum used to calculate the threshold.
    :type values: numpy.ndarray
    :param strategy: Registered strategy name or custom callable.
    :type strategy: str | Callable[..., Any]
    :param options: Strategy-specific keyword arguments.
    :type options: Mapping[str, Any] | None
    :return: One finite scalar or one finite threshold per spectrum position.
    :rtype: float | numpy.ndarray
    :raises ValidationError: If the strategy is unknown or its result is invalid.
    """
    settings = dict(options or {})
    valid = values[np.isfinite(values)]
    if callable(strategy):
        threshold = strategy(values, **settings)
    elif strategy == "absolute":
        threshold = settings.get("value", 0.0)
    elif strategy == "relative_to_max":
        threshold = (float(np.max(valid)) if valid.size else 0.0) * float(settings.get("scale", 0.0))
    elif strategy == "quantile":
        threshold = float(np.quantile(valid, float(settings.get("quantile", 0.5)))) if valid.size else 0.0
    elif strategy == "mean_std":
        threshold = (float(np.mean(valid)) + float(settings.get("scale", 1.0)) * float(np.std(valid))) if valid.size else 0.0
    elif strategy == "median_mad":
        median = float(np.median(valid)) if valid.size else 0.0
        mad = float(np.median(np.abs(valid - median))) if valid.size else 0.0
        threshold = median + float(settings.get("scale", 1.0)) * mad
    else:
        raise_validation_error("InverseBinner", f"Unknown threshold strategy '{strategy}'.")
    resolved = np.asarray(threshold, dtype=np.float64)
    if resolved.ndim > 1 or (resolved.ndim == 1 and resolved.shape != values.shape) or not np.all(np.isfinite(resolved)):
        raise_validation_error("InverseBinner", "Threshold must be finite and scalar or match the spectrum shape.")
    return float(resolved) if resolved.ndim == 0 else resolved


def select_by_cumulative_mass(
    candidates: np.ndarray,
    weights: np.ndarray,
    retained_fraction: float,
    minimum: int = 0,
    maximum: int | None = None,
) -> tuple[np.ndarray, bool, float]:
    """Select the smallest ranked prefix satisfying a cumulative mass target.

    :param candidates: Eligible dense-spectrum indices.
    :type candidates: numpy.ndarray
    :param weights: Non-negative selection mass for every spectrum position.
    :type weights: numpy.ndarray
    :param retained_fraction: Requested fraction of total candidate mass.
    :type retained_fraction: float
    :param minimum: Minimum prefix length when candidates are available.
    :type minimum: int
    :param maximum: Optional maximum prefix length.
    :type maximum: int | None
    :return: Selected indices, whether they reached the target, and the retained
        fraction.
    :rtype: tuple[numpy.ndarray, bool, float]
    :raises ValidationError: If the retained fraction or selection budget is invalid.
    """
    minimum, maximum = validate_budget(minimum, maximum)
    if not 0.0 < retained_fraction <= 1.0:
        raise_validation_error("InverseBinner", "retained_fraction must belong to (0, 1].")
    ranked = rank_candidates(candidates, weights)
    if maximum is not None:
        ranked = ranked[:maximum]
    total = float(np.sum(weights[candidates]))
    if not ranked.size or total <= 0.0:
        return ranked[:0], False, 0.0
    required = int(np.searchsorted(np.cumsum(weights[ranked]), retained_fraction * total, side="left") + 1)
    count = max(minimum, required)
    selected = ranked[: min(count, ranked.size)]
    fraction = float(np.sum(weights[selected]) / total)
    return selected, fraction + 1e-12 >= retained_fraction, fraction


def base_diagnostics(values: np.ndarray, selected: np.ndarray, output_count: int, weights: np.ndarray | None = None) -> dict[str, Any]:
    """Build count, retention, and compression diagnostics for inverse selection.

    :param values: Original dense spectrum intensities.
    :type values: numpy.ndarray
    :param selected: Dense-spectrum indices contributing to the sparse output.
    :type selected: numpy.ndarray
    :param output_count: Number of emitted sparse output points.
    :type output_count: int
    :param weights: Optional alternative mass used for retention diagnostics.
    :type weights: numpy.ndarray | None
    :return: Shared diagnostic fields consumed by all inverse strategies.
    :rtype: dict[str, Any]
    """
    valid = valid_candidate_mask(values)
    raw_total = float(np.sum(values[valid]))
    selected_raw = float(np.sum(values[selected])) if selected.size else 0.0
    mass = values if weights is None else weights
    mass_total = float(np.sum(mass[valid]))
    selected_mass = float(np.sum(mass[selected])) if selected.size else 0.0
    return {
        "input_bin_count": int(values.size), "valid_bin_count": int(np.sum(valid)),
        "selected_bin_count": int(selected.size), "output_peak_count": int(output_count),
        "retained_raw_intensity_fraction": selected_raw / raw_total if raw_total else 0.0,
        "retained_mass_fraction": selected_mass / mass_total if mass_total else 0.0,
        "compression_ratio": output_count / values.size if values.size else 0.0,
    }
