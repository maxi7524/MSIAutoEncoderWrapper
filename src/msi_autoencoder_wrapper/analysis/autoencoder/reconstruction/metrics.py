"""Pure reconstruction calculations for single- and multi-model analyses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.signal import find_peaks

from ....metrics import (
    SpectrumMasserstein,
    cosine_similarity,
    feature_errors,
    mae,
    mse,
    spectral_angle,
    tic_error,
)
from ....utils.exceptions import raise_validation_error

METRIC_DIRECTIONS: Mapping[str, str] = {
    "mse": "minimize",
    "mae": "minimize",
    "cosine_similarity": "maximize",
    "spectral_angle": "minimize",
    "tic_error": "absolute_minimize",
    "masserstein": "minimize",
    "wasserstein": "minimize",
}


def masserstein_distances(
    inputs: np.ndarray,
    outputs: np.ndarray,
    mass_axis: np.ndarray,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    criterion_options: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Calculate the training Masserstein objective independently per spectrum.

    This function reuses :class:`SpectrumMasserstein`, which is also wrapped by
    the training criterion. ``reduction='none'`` is forced because ranking and
    spatial maps require one value for every ``spectrum_id``.

    :param inputs: Original binned spectra.
    :type inputs: numpy.ndarray
    :param outputs: Reconstructed spectra with the same shape.
    :type outputs: numpy.ndarray
    :param mass_axis: Physical m/z coordinate for every feature.
    :type mass_axis: numpy.ndarray
    :param batch_size: Number of spectra evaluated together.
    :type batch_size: int
    :param device: Torch execution device.
    :type device: str | torch.device
    :param criterion_options: Masserstein criterion parameters used in training.
    :type criterion_options: Mapping[str, Any] | None
    :return: Per-spectrum Masserstein costs.
    :rtype: numpy.ndarray
    :raises ValidationError: If arrays, axis, or batch size are incompatible.
    """
    original = np.asarray(inputs)
    reconstructed = np.asarray(outputs)
    axis = np.asarray(mass_axis, dtype=np.float32)
    if original.ndim != 2 or original.shape != reconstructed.shape:
        raise_validation_error(
            "ReconstructionAnalysis",
            "Masserstein requires equal two-dimensional spectrum arrays.",
        )
    if axis.ndim != 1 or axis.size != original.shape[1] or batch_size < 1:
        raise_validation_error(
            "ReconstructionAnalysis",
            "Masserstein axis size and batch_size must match the input matrix.",
        )

    # Criterion construction
    ## Keep every training option configurable while enforcing per-item output.
    options = dict(criterion_options or {})
    options["reduction"] = "none"
    metric = SpectrumMasserstein(**options).to(device)
    axis_tensor = torch.as_tensor(axis, dtype=torch.float32, device=device)

    # Bounded inference
    ## Chunking avoids transferring the complete retained image to the device.
    costs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(original), batch_size):
            stop = min(start + batch_size, len(original))
            target = torch.as_tensor(original[start:stop], device=device)
            prediction = torch.as_tensor(reconstructed[start:stop], device=device)
            values = metric(
                prediction,
                target,
                mass_axis=axis_tensor,
            )
            costs.append(values.detach().cpu().numpy())
    return np.concatenate(costs).astype(np.float32, copy=False)


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
    original = np.asarray(inputs)
    reconstructed = np.asarray(outputs)
    if original.ndim != 2 or original.shape != reconstructed.shape:
        raise_validation_error(
            "AutoencoderAnalysis",
            "Reconstruction metrics require equal two-dimensional arrays.",
        )
    target = torch.as_tensor(original, dtype=torch.float32)
    prediction = torch.as_tensor(reconstructed, dtype=torch.float32)
    feature_values = feature_errors(prediction, target)
    input_tic = target.sum(dim=1)
    reconstruction_tic = prediction.sum(dim=1)
    return {
        "mse": mse(prediction, target).numpy(),
        "mae": mae(prediction, target).numpy(),
        "cosine_similarity": cosine_similarity(prediction, target).numpy(),
        "spectral_angle": spectral_angle(prediction, target).numpy(),
        "tic_input": input_tic.numpy(),
        "tic_reconstruction": reconstruction_tic.numpy(),
        "tic_error": tic_error(prediction, target).numpy(),
        **{name: value.numpy() for name, value in feature_values.items()},
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    """Return descriptive statistics for one metric distribution.

    :param values: Metric values.
    :type values: numpy.ndarray
    :return: Count, moments, extrema, and selected quantiles.
    :rtype: Dict[str, float]
    """
    array = np.asarray(values, dtype=np.float32)
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
    complete float32 contribution matrix in addition to retained inputs and
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
        name: np.empty(original.shape[1], dtype=np.float32)
        for name in ("mean", "median", "lower", "upper")
    }
    for start in range(0, original.shape[1], chunk_size):
        stop = min(start + chunk_size, original.shape[1])
        # Match the core reconstruction metrics' float32 accumulation while
        # limiting conversion to one feature chunk instead of duplicating both
        # complete retained matrices.
        residual = original[:, start:stop].astype(
            np.float32, copy=False
        ) - reconstructed[:, start:stop].astype(np.float32, copy=False)
        contribution = residual**2 if metric == "mse" else np.abs(residual)
        profiles["mean"][start:stop] = np.mean(contribution, axis=0)
        profiles["median"][start:stop] = np.median(contribution, axis=0)
        profiles["lower"][start:stop] = np.quantile(contribution, lower, axis=0)
        profiles["upper"][start:stop] = np.quantile(contribution, upper, axis=0)
    return profiles


def peak_matching_errors(
    inputs: np.ndarray,
    outputs: np.ndarray,
    mass_axis: np.ndarray,
    window_bins: int = 5,
    prominence_fraction: float = 0.02,
    min_detection_fraction: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Match every detected peak in each input spectrum to its nearest local
    maximum in the corresponding reconstruction, and report the two errors
    separately: where the peak moved (m/z) and how its abundance changed
    (TIC-normalized intensity).

    A peak's mass axis position and its intensity are two independent
    reconstruction failure modes — a broadened peak envelope can sit at the
    correct m/z with a large intensity error, or vice versa — so they are
    never combined into one scalar here.

    :param inputs: Original binned spectra, shape ``(samples, features)``.
    :type inputs: numpy.ndarray
    :param outputs: Reconstructed spectra with the same shape.
    :type outputs: numpy.ndarray
    :param mass_axis: Physical m/z coordinate for every feature.
    :type mass_axis: numpy.ndarray
    :param window_bins: Half-width, in bins, of the reconstruction search
        window centered on each detected input peak.
    :type window_bins: int
    :param prominence_fraction: Minimum peak prominence in ``scipy.signal.find_peaks``,
        as a fraction of that spectrum's own maximum intensity (peak detection
        is run once per spectrum, so this adapts to each spectrum's own scale).
    :type prominence_fraction: float
    :param min_detection_fraction: Minimum reconstructed intensity at the matched
        bin, as a fraction of the original peak's own height, below which the
        peak is treated as undetected rather than mislocalized (see REMARK below).
    :type min_detection_fraction: float
    :return: Per-matched-peak arrays: ``spectrum_index``, ``peak_mz``,
        ``mz_error`` (absolute, physical units; ``NaN`` where ``detected`` is
        ``False``), ``relative_intensity_error`` (signed, TIC-normalized, always
        populated), ``original_intensity`` (raw, for filtering), ``detected``
        (bool).
    :rtype: Dict[str, numpy.ndarray]
    :raises ValidationError: If arrays, axis, or settings are incompatible.

    .. note::
        REMARK: Reconstruction decoders in this project end in a nonnegative
        (ReLU-family) output activation, so a search window with no real
        reconstructed signal is often *exactly* zero everywhere; ``np.argmax``
        then deterministically returns the window's first index regardless of
        model quality, producing an artificial constant "shift" that has
        nothing to do with localization. ``min_detection_fraction`` filters
        these ties out of ``mz_error`` (set to ``NaN``) while still reporting
        the (strongly negative) ``relative_intensity_error`` and the
        ``detected`` flag, so an undetected peak's own miss rate remains
        visible rather than silently inflating the localization-error stats.
        Peaks closer together than ``2 * window_bins`` bins can also match the
        same reconstructed local maximum (ambiguous assignment); acceptable for
        aggregate distributions, but individual rows are not a one-to-one
        peak correspondence in dense regions.
    """
    original = np.asarray(inputs, dtype=np.float64)
    reconstructed = np.asarray(outputs, dtype=np.float64)
    axis = np.asarray(mass_axis, dtype=np.float64)
    if original.ndim != 2 or original.shape != reconstructed.shape:
        raise_validation_error(
            "ReconstructionAnalysis",
            "Peak matching requires equal two-dimensional spectrum arrays.",
        )
    if axis.ndim != 1 or axis.size != original.shape[1]:
        raise_validation_error(
            "ReconstructionAnalysis",
            "Peak matching mass axis size must match the input matrix.",
        )
    if window_bins < 1 or not 0.0 < prominence_fraction < 1.0:
        raise_validation_error(
            "ReconstructionAnalysis",
            "window_bins must be >= 1 and prominence_fraction must be in (0, 1).",
        )
    if not 0.0 <= min_detection_fraction < 1.0:
        raise_validation_error(
            "ReconstructionAnalysis", "min_detection_fraction must be in [0, 1)."
        )

    spectrum_indices: list[int] = []
    peak_mz_values: list[float] = []
    mz_errors: list[float] = []
    relative_intensity_errors: list[float] = []
    original_intensities: list[float] = []
    detected_flags: list[bool] = []

    n_features = original.shape[1]
    for spectrum_index in range(original.shape[0]):
        original_spectrum = original[spectrum_index]
        reconstructed_spectrum = reconstructed[spectrum_index]
        original_tic = original_spectrum.sum()
        reconstructed_tic = reconstructed_spectrum.sum()
        if original_tic <= 0.0 or reconstructed_tic <= 0.0:
            continue

        peak_indices, _ = find_peaks(
            original_spectrum, prominence=prominence_fraction * original_spectrum.max()
        )
        for peak_index in peak_indices:
            window_start = max(0, peak_index - window_bins)
            window_stop = min(n_features, peak_index + window_bins + 1)
            matched_index = window_start + int(
                np.argmax(reconstructed_spectrum[window_start:window_stop])
            )

            original_peak_intensity = original_spectrum[peak_index]
            matched_intensity = reconstructed_spectrum[matched_index]
            detected = matched_intensity >= min_detection_fraction * original_peak_intensity

            original_relative_intensity = original_peak_intensity / original_tic
            reconstructed_relative_intensity = matched_intensity / reconstructed_tic

            spectrum_indices.append(spectrum_index)
            peak_mz_values.append(axis[peak_index])
            mz_errors.append(abs(axis[matched_index] - axis[peak_index]) if detected else np.nan)
            relative_intensity_errors.append(
                (reconstructed_relative_intensity - original_relative_intensity)
                / original_relative_intensity
            )
            original_intensities.append(original_peak_intensity)
            detected_flags.append(detected)

    return {
        "spectrum_index": np.asarray(spectrum_indices, dtype=np.int64),
        "peak_mz": np.asarray(peak_mz_values, dtype=np.float64),
        "mz_error": np.asarray(mz_errors, dtype=np.float64),
        "relative_intensity_error": np.asarray(relative_intensity_errors, dtype=np.float64),
        "original_intensity": np.asarray(original_intensities, dtype=np.float64),
        "detected": np.asarray(detected_flags, dtype=bool),
    }


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
    key = lambda item: (
        abs(item[1]["mean"]) if direction == "absolute_minimize" else item[1]["mean"]
    )
    return [
        name
        for name, _ in sorted(
            summaries.items(),
            key=key,
            reverse=direction == "maximize",
        )
    ]


def metric_order(values: np.ndarray, metric: str) -> np.ndarray:
    """Return row indices ordered from best to worst for one metric."""
    array = np.asarray(values)
    direction = METRIC_DIRECTIONS.get(metric, "minimize")
    sortable = np.abs(array) if direction == "absolute_minimize" else array
    order = np.argsort(sortable)
    return order[::-1] if direction == "maximize" else order
