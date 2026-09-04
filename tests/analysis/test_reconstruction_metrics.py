"""Tests for pure reconstruction-error calculations on synthetic spectra."""

from __future__ import annotations

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.metrics import peak_matching_errors
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def _gaussian_peak(axis: np.ndarray, center: float, height: float, width: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((axis - center) / width) ** 2)


@pytest.fixture
def mass_axis() -> np.ndarray:
    return np.linspace(100.0, 200.0, 1001)  # 0.1 m/z per bin


def test_perfect_reconstruction_has_zero_error_at_every_matched_peak(mass_axis: np.ndarray) -> None:
    spectrum = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3) + _gaussian_peak(mass_axis, 170.0, 4.0, 0.3)
    result = peak_matching_errors(spectrum[None, :], spectrum[None, :], mass_axis)

    assert result["spectrum_index"].size == 2
    assert np.all(result["detected"])
    np.testing.assert_allclose(result["mz_error"], 0.0, atol=1e-9)
    np.testing.assert_allclose(result["relative_intensity_error"], 0.0, atol=1e-6)


def test_shifted_peak_reports_the_exact_mz_shift(mass_axis: np.ndarray) -> None:
    bin_width = mass_axis[1] - mass_axis[0]
    shift_bins = 3
    original = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3)
    reconstructed = np.roll(original, shift_bins)

    result = peak_matching_errors(original[None, :], reconstructed[None, :], mass_axis, window_bins=5)

    assert result["spectrum_index"].size == 1
    np.testing.assert_allclose(result["mz_error"], shift_bins * bin_width, atol=1e-6)


def test_intensity_only_change_reports_zero_mz_error_and_the_exact_relative_change(mass_axis: np.ndarray) -> None:
    original = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3) + _gaussian_peak(mass_axis, 170.0, 10.0, 0.3)
    # Halve only the first peak; both peaks share equal height/width so TIC
    # normalization must be undone by hand to predict the expected ratio.
    reconstructed = _gaussian_peak(mass_axis, 150.0, 5.0, 0.3) + _gaussian_peak(mass_axis, 170.0, 10.0, 0.3)

    result = peak_matching_errors(original[None, :], reconstructed[None, :], mass_axis)

    order = np.argsort(result["peak_mz"])
    mz_errors = result["mz_error"][order]
    relative_errors = result["relative_intensity_error"][order]

    np.testing.assert_allclose(mz_errors, 0.0, atol=1e-9)
    original_tic, reconstructed_tic = original.sum(), reconstructed.sum()
    original_relative = np.array([10.0, 10.0]) / original_tic
    reconstructed_relative = np.array([5.0, 10.0]) / reconstructed_tic
    expected = (reconstructed_relative - original_relative) / original_relative
    np.testing.assert_allclose(relative_errors, expected, rtol=1e-5)
    assert relative_errors[0] < 0.0  # the halved peak lost relative abundance


def test_undetected_peak_reports_nan_mz_error_but_a_strongly_negative_intensity_error(mass_axis: np.ndarray) -> None:
    original = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3) + _gaussian_peak(mass_axis, 170.0, 10.0, 0.3)
    # The reconstruction completely misses the 150.0 peak (left at the ReLU floor of
    # a real decoder: exactly zero) but reproduces the 170.0 peak perfectly.
    reconstructed = np.zeros_like(mass_axis)
    reconstructed += _gaussian_peak(mass_axis, 170.0, 10.0, 0.3)

    result = peak_matching_errors(original[None, :], reconstructed[None, :], mass_axis)

    order = np.argsort(result["peak_mz"])
    detected = result["detected"][order]
    mz_errors = result["mz_error"][order]
    relative_errors = result["relative_intensity_error"][order]

    assert list(detected) == [False, True]
    assert np.isnan(mz_errors[0])
    assert mz_errors[1] == pytest.approx(0.0, abs=1e-9)
    assert relative_errors[0] == pytest.approx(-1.0, abs=1e-9)  # fully missing -> -100%


def test_detection_floor_is_configurable(mass_axis: np.ndarray) -> None:
    original = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3)
    # Reconstruction reaches 10% of the true peak height at the exact same bin.
    reconstructed = _gaussian_peak(mass_axis, 150.0, 1.0, 0.3)

    strict = peak_matching_errors(original[None, :], reconstructed[None, :], mass_axis, min_detection_fraction=0.20)
    lenient = peak_matching_errors(original[None, :], reconstructed[None, :], mass_axis, min_detection_fraction=0.05)

    assert not strict["detected"][0]
    assert np.isnan(strict["mz_error"][0])
    assert lenient["detected"][0]
    assert lenient["mz_error"][0] == pytest.approx(0.0, abs=1e-9)


def test_zero_signal_spectrum_is_skipped_without_dividing_by_zero(mass_axis: np.ndarray) -> None:
    silent = np.zeros_like(mass_axis)
    peaked = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3)
    inputs = np.stack([silent, peaked])
    outputs = np.stack([silent, peaked])

    result = peak_matching_errors(inputs, outputs, mass_axis)

    assert result["spectrum_index"].size == 1
    assert result["spectrum_index"][0] == 1


def test_multiple_spectra_are_indexed_independently(mass_axis: np.ndarray) -> None:
    first = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3)
    second = _gaussian_peak(mass_axis, 160.0, 8.0, 0.3)
    inputs = np.stack([first, second])
    outputs = np.stack([first, second])

    result = peak_matching_errors(inputs, outputs, mass_axis)

    assert sorted(result["spectrum_index"].tolist()) == [0, 1]


@pytest.mark.parametrize(
    ("window_bins", "prominence_fraction", "min_detection_fraction"),
    [(0, 0.02, 0.05), (-1, 0.02, 0.05), (5, 0.0, 0.05), (5, 1.0, 0.05), (5, -0.1, 0.05), (5, 0.02, 1.0), (5, 0.02, -0.1)],
)
def test_rejects_invalid_settings(
    mass_axis: np.ndarray, window_bins: int, prominence_fraction: float, min_detection_fraction: float
) -> None:
    spectrum = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3)
    with pytest.raises(ValidationError):
        peak_matching_errors(
            spectrum[None, :], spectrum[None, :], mass_axis,
            window_bins=window_bins, prominence_fraction=prominence_fraction,
            min_detection_fraction=min_detection_fraction,
        )


def test_rejects_mismatched_shapes(mass_axis: np.ndarray) -> None:
    spectrum = _gaussian_peak(mass_axis, 150.0, 10.0, 0.3)
    with pytest.raises(ValidationError):
        peak_matching_errors(spectrum[None, :], spectrum[None, :-1], mass_axis)
    with pytest.raises(ValidationError):
        peak_matching_errors(spectrum[None, :], spectrum[None, :], mass_axis[:-1])
