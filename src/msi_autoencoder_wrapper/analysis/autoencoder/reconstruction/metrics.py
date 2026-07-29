"""Pure reconstruction calculations for single- and multi-model analyses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

import numpy as np

from ....utils.exceptions import raise_validation_error

METRIC_DIRECTIONS: Mapping[str, str] = {
    "mse": "minimize",
    "mae": "minimize",
    "cosine_similarity": "maximize",
    "spectral_angle": "minimize",
    "tic_error": "absolute_minimize",
}


def reconstruction_metrics(
    inputs: np.ndarray,
    outputs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-spectrum and mean per-feature reconstruction metrics.

    :param inputs: Original binned spectra of shape ``(samples, features)``.
    :type inputs: numpy.ndarray
    :param outputs: Reconstructed spectra with the same shape.
    :type outputs: numpy.ndarray
    :return: Per-spectrum and per-feature metric arrays.
    :rtype: Dict[str, numpy.ndarray]
    :raises ValidationError: If input shapes differ or are not matrices.
    """
    original = np.asarray(inputs, dtype=np.float64)
    reconstructed = np.asarray(outputs, dtype=np.float64)
    if original.ndim != 2 or original.shape != reconstructed.shape:
        raise_validation_error(
            "AutoencoderAnalysis",
            "Reconstruction metrics require equal two-dimensional arrays.",
        )
    residual = original - reconstructed
    dot_product = np.sum(original * reconstructed, axis=1)
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(
        reconstructed, axis=1
    )
    cosine = np.divide(
        dot_product,
        denominator,
        out=np.zeros_like(dot_product),
        where=denominator > 0,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    return {
        "mse": np.mean(residual**2, axis=1),
        "mae": np.mean(np.abs(residual), axis=1),
        "cosine_similarity": cosine,
        "spectral_angle": np.arccos(cosine),
        "tic_input": np.sum(original, axis=1),
        "tic_reconstruction": np.sum(reconstructed, axis=1),
        "tic_error": np.sum(reconstructed, axis=1) - np.sum(original, axis=1),
        "feature_mse": np.mean(residual**2, axis=0),
        "feature_mae": np.mean(np.abs(residual), axis=0),
        "feature_bias": np.mean(reconstructed - original, axis=0),
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    """Return descriptive statistics for one metric distribution.

    :param values: Metric values.
    :type values: numpy.ndarray
    :return: Count, moments, extrema, and selected quantiles.
    :rtype: Dict[str, float]
    """
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def feature_error_distribution(
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    metric: str = "mse",
    quantiles: tuple[float, float] = (0.05, 0.95),
    chunk_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """Calculate per-feature distribution bands without a second full matrix.

    Features are processed in chunks so quantile calculation does not create a
    complete float64 contribution matrix in addition to retained inputs and
    reconstructions.

    :param inputs: Retained input spectra.
    :type inputs: numpy.ndarray
    :param reconstructions: Retained reconstructed spectra.
    :type reconstructions: numpy.ndarray
    :param metric: Additive feature metric, ``mse`` or ``mae``.
    :type metric: str
    :param quantiles: Lower and upper distribution quantiles.
    :type quantiles: tuple[float, float]
    :param chunk_size: Number of features processed together.
    :type chunk_size: int
    :return: Mean, median, lower, and upper feature profiles.
    :rtype: Dict[str, numpy.ndarray]
    :raises ValidationError: If settings or input arrays are invalid.
    """
    original = np.asarray(inputs)
    reconstructed = np.asarray(reconstructions)
    if original.shape != reconstructed.shape or original.ndim != 2:
        raise_validation_error(
            "ReconstructionAnalysis", "Feature profiles require equal matrices."
        )
    if metric not in {"mse", "mae"}:
        raise_validation_error(
            "ReconstructionAnalysis", "Feature metric must be 'mse' or 'mae'."
        )
    lower, upper = quantiles
    if not 0.0 <= lower <= upper <= 1.0 or chunk_size < 1:
        raise_validation_error(
            "ReconstructionAnalysis", "Invalid quantiles or feature chunk size."
        )
    profiles = {
        name: np.empty(original.shape[1], dtype=np.float64)
        for name in ("mean", "median", "lower", "upper")
    }
    for start in range(0, original.shape[1], chunk_size):
        stop = min(start + chunk_size, original.shape[1])
        residual = original[:, start:stop] - reconstructed[:, start:stop]
        contribution = residual**2 if metric == "mse" else np.abs(residual)
        profiles["mean"][start:stop] = np.mean(contribution, axis=0)
        profiles["median"][start:stop] = np.median(contribution, axis=0)
        profiles["lower"][start:stop] = np.quantile(contribution, lower, axis=0)
        profiles["upper"][start:stop] = np.quantile(contribution, upper, axis=0)
    return profiles


def rank_models(
    summaries: Mapping[str, Mapping[str, float]],
    metric: str,
) -> list[str]:
    """Rank model names using the semantic direction of one metric.

    :param summaries: Per-model distribution summaries.
    :type summaries: Mapping[str, Mapping[str, float]]
    :param metric: Metric name with a registered direction.
    :type metric: str
    :return: Best-to-worst model names.
    :rtype: list[str]
    """
    direction = METRIC_DIRECTIONS.get(metric, "minimize")
    key = lambda item: abs(item[1]["mean"]) if direction == "absolute_minimize" else item[1]["mean"]
    return [
        name
        for name, _ in sorted(
            summaries.items(),
            key=key,
            reverse=direction == "maximize",
        )
    ]
