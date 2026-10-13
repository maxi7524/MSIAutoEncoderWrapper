"""Reconstruction error tails and rare-feature diagnostics with frozen prevalence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import masserstein_distances, reconstruction_metrics
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def reconstruction_diagnostics(inputs: np.ndarray, outputs: np.ndarray, mass_axis: np.ndarray,
                               reference_prevalence: np.ndarray, *, rare_threshold: float,
                               device: str = "cpu", top_k: int = 20) -> dict[str, pd.DataFrame]:
    """Retain per-spectrum errors, feature rarity and largest individual failures.

    Prevalence is frozen from training/validation, not fitted on external test.
    Existing spectral and Masserstein functions are reused. Undefined relative
    errors at zero targets are NaN; absolute errors still include hallucinations.
    Complexity O(N*M + N*M log(top_k)), auxiliary storage O(N*M) for residuals.

    :param inputs: Nonnegative finite target spectra, shape ``(N, M)``.
    :param outputs: Nonnegative finite reconstructions, same shape.
    :param mass_axis: Strictly increasing feature axis, shape ``(M,)``.
    :param reference_prevalence: Frozen nonzero prevalence, shape ``(M,)``.
    :param rare_threshold: Maximum prevalence for a present rare feature.
    :param device: Device for existing Masserstein evaluation.
    :param top_k: Number of worst spectrum-feature errors retained.
    :return: Tables ``spectra``, ``features``, ``worst_errors`` and ``summary``.
    :raises ValueError: For invalid inputs or diagnostic settings.
    """
    x, y = np.asarray(inputs, dtype=np.float64), np.asarray(outputs, dtype=np.float64)  # (N, M), (N, M)
    axis, prevalence = np.asarray(mass_axis), np.asarray(reference_prevalence, dtype=float)  # (M,), (M,)
    if x.ndim != 2 or not x.size or y.shape != x.shape or axis.shape != (x.shape[1],) or prevalence.shape != axis.shape:
        raise ValueError("Reconstruction arrays and feature metadata are inconsistent.")
    if not all(np.isfinite(v).all() for v in [x, y, axis, prevalence]) or np.any(x < 0) or np.any(y < 0) or np.any(np.diff(axis) <= 0):
        raise ValueError("Spectra must be finite/nonnegative and the mass axis strictly increasing.")
    if np.any((prevalence < 0) | (prevalence > 1)) or not 0 <= rare_threshold <= 1 or top_k < 1:
        raise ValueError("Prevalence, rare threshold and top_k are invalid.")
    if max(float(x.max()), float(y.max())) > float(np.finfo(np.float32).max):
        raise ValueError("Source intensities cannot be represented by the maintained float32 metrics.")
    # Reuse the maintained computational definitions
    metrics = reconstruction_metrics(x, y)
    costs = masserstein_distances(x.astype(np.float32), y.astype(np.float32), axis, device=device)  # (N,)
    if not all(np.isfinite(v).all() for v in metrics.values()) or not np.isfinite(costs).all():
        raise ValueError("Nonfinite output from maintained float32 reconstruction metrics; inspect source scale.")
    error = np.abs(y - x)  # (N, M)
    relative = np.full(x.shape, np.nan)  # (N, M)
    np.divide(error, x, out=relative, where=x > 0)
    rare = (prevalence > 0) & (prevalence <= rare_threshold)  # (M,)
    spectra = pd.DataFrame({"row_position": np.arange(len(x)), "mse": metrics["mse"], "mae": metrics["mae"],
                            "spectral_angle": metrics["spectral_angle"], "masserstein": costs,
                            "max_absolute_error": error.max(axis=1),
                            "rare_mae": error[:, rare].mean(axis=1) if rare.any() else np.nan})
    records = []
    for j in range(x.shape[1]):
        positive = x[:, j] > 0  # (N,)
        records.append({"feature_index": j, "mz": float(axis[j]), "reference_prevalence": prevalence[j], "rare": bool(rare[j]),
                        "mae": float(error[:, j].mean()), "max_error": float(error[:, j].max()),
                        "positive_mae": float(error[positive, j].mean()) if positive.any() else np.nan,
                        "positive_relative_error": float(relative[positive, j].mean()) if positive.any() else np.nan,
                        "missed_positive_fraction": float(np.mean(y[positive, j] == 0)) if positive.any() else np.nan})
    # Preserve both ranking and exact spectral identities for inspection
    count = min(top_k, error.size)
    flat = error.ravel()  # (N*M,)
    candidates = np.argpartition(-flat, count - 1)[:count]  # (K,)
    selected = candidates[np.argsort(-flat[candidates], kind="stable")]  # (K,)
    rows, columns = np.unravel_index(selected, x.shape)  # (K,), (K,)
    worst = pd.DataFrame({"row_position": rows, "feature_index": columns, "mz": axis[columns],
                          "target": x[rows, columns], "prediction": y[rows, columns], "absolute_error": flat[selected],
                          "relative_error": relative[rows, columns], "rare": rare[columns]})
    summaries = []
    for name in ["mse", "mae", "spectral_angle", "masserstein", "max_absolute_error", "rare_mae"]:
        values = spectra[name].dropna().to_numpy()
        if len(values):
            summaries.append({"metric": name, "mean": float(values.mean()), "median": float(np.median(values)),
                              "q95": float(np.quantile(values, .95)), "q99": float(np.quantile(values, .99)), "max": float(values.max())})
    logger.info("Reconstruction diagnostics: %s spectra, %s features, %s rare features.", len(x), x.shape[1], int(rare.sum()))
    return {"spectra": spectra, "features": pd.DataFrame(records), "worst_errors": worst, "summary": pd.DataFrame(summaries)}
