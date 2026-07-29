"""Reconstruction metric calculations."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from ....utils.exceptions import raise_validation_error


def reconstruction_metrics(
    inputs: np.ndarray, outputs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute scalar reconstruction metrics for every spectrum.

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


def summarize(values: np.ndarray) -> Mapping[str, float]:
    """Summarize a one-dimensional metric distribution.

    :param values: Metric values.
    :type values: numpy.ndarray
    :return: Count, moments, extrema, and selected quantiles.
    :rtype: Mapping[str, float]
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
