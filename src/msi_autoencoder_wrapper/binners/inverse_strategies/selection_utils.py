"""Shared validation, thresholding, ranking, and diagnostics for inverse binners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ...utils.exceptions import raise_validation_error


def validate_input(values: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return compatible one-dimensional floating point input and axis arrays."""
    intensities = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(axis, dtype=np.float64)
    if intensities.ndim != 1 or coordinates.ndim != 1 or intensities.size != coordinates.size:
        raise_validation_error("InverseBinner", "Axis and intensities must be equal one-dimensional arrays.")
    return intensities, coordinates


def valid_candidate_mask(values: np.ndarray) -> np.ndarray:
    """Select finite, strictly positive candidates; inverse binners do not emit zeros."""
    return np.isfinite(values) & (values > 0.0)


def validate_budget(minimum: int = 0, maximum: int | None = None) -> tuple[int, int | None]:
    """Validate a selection budget."""
    minimum = int(minimum)
    maximum = None if maximum is None else int(maximum)
    if minimum < 0 or (maximum is not None and maximum <= 0) or (maximum is not None and minimum > maximum):
        raise_validation_error("InverseBinner", "Selection limits require 0 <= minimum <= maximum and positive maximum.")
    return minimum, maximum


def rank_candidates(indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rank indices by descending weight and ascending axis index for deterministic ties."""
    indices = np.asarray(indices, dtype=np.int64)
    return indices[np.lexsort((indices, -np.asarray(weights)[indices]))]


def resolve_threshold(
    values: np.ndarray,
    strategy: str | Callable[..., Any],
    options: Mapping[str, Any] | None = None,
) -> float | np.ndarray:
    """Resolve a scalar or shape-compatible threshold for one spectrum."""
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
    """Select the smallest deterministic prefix satisfying a cumulative mass target."""
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
    """Build diagnostics shared by every inverse strategy."""
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
