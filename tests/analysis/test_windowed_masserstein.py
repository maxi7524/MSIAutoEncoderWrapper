"""Analytical checks of the m/z-windowed Masserstein decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.metrics import masserstein_distances
from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.windowed import (
    window_assignment,
    window_edges,
    windowed_masserstein,
)


@pytest.fixture
def half_integer_axis() -> np.ndarray:
    """Unit-spaced centres ``100.5 ... 399.5``; window edges fall between centres."""
    return np.arange(100.5, 400.0, 1.0)


def _delta(axis: np.ndarray, position: float) -> np.ndarray:
    spectrum = np.zeros_like(axis, dtype=np.float32)
    spectrum[int(np.argmin(np.abs(axis - position)))] = 1.0
    return spectrum


def test_edges_are_aligned_to_multiples_of_the_width_on_both_production_axes() -> None:
    narrow = 200.0 + 0.55 * (np.arange(1273) + 0.5)
    wide = 100.0 + 0.55 * (np.arange(5273) + 0.5)

    np.testing.assert_allclose(window_edges(narrow, 100.0), np.arange(200.0, 901.0, 100.0))
    np.testing.assert_allclose(window_edges(wide, 100.0), np.arange(100.0, 3001.0, 100.0))


def test_assignment_uses_half_open_windows(half_integer_axis: np.ndarray) -> None:
    edges = window_edges(half_integer_axis, 100.0)
    assignment = window_assignment(half_integer_axis, edges)

    assert assignment[half_integer_axis == 199.5][0] == 0
    assert assignment[half_integer_axis == 200.5][0] == 1
    assert np.bincount(assignment).tolist() == [100, 100, 100]


def test_identical_spectra_have_zero_cost_in_every_window(half_integer_axis: np.ndarray) -> None:
    generator = np.random.default_rng(0)
    spectra = generator.random((4, half_integer_axis.size)).astype(np.float32)
    result = windowed_masserstein(spectra, spectra, half_integer_axis, window_edges(half_integer_axis, 100.0))

    np.testing.assert_allclose(result.total, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.contribution, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.within, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.mass_imbalance, 0.0, atol=1e-6)


def test_transport_across_windows_is_split_at_the_window_edge(half_integer_axis: np.ndarray) -> None:
    source = _delta(half_integer_axis, 150.5)[None, :]
    moved = _delta(half_integer_axis, 250.5)[None, :]
    result = windowed_masserstein(source, moved, half_integer_axis, window_edges(half_integer_axis, 100.0))

    ## All mass travels 100 m/z: 49.5 of the path lies in [100, 200), 50.5 in [200, 300)
    np.testing.assert_allclose(result.total, [100.0], rtol=1e-6)
    np.testing.assert_allclose(result.contribution[0], [49.5, 50.5, 0.0], atol=1e-4)
    np.testing.assert_allclose(result.input_mass[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.output_mass[0], [0.0, 1.0, 0.0])
    ## A window where one spectrum carries no mass has no defined local shape
    assert np.isnan(result.within[0, 0]) and np.isnan(result.within[0, 1])


def test_local_shape_error_is_renormalized_inside_the_window(half_integer_axis: np.ndarray) -> None:
    ## Both spectra split their mass evenly between two windows; only window 0 differs
    original = 0.5 * (_delta(half_integer_axis, 150.5) + _delta(half_integer_axis, 350.5))
    shifted = 0.5 * (_delta(half_integer_axis, 153.5) + _delta(half_integer_axis, 350.5))
    result = windowed_masserstein(original[None, :], shifted[None, :], half_integer_axis,
                                  window_edges(half_integer_axis, 100.0))

    np.testing.assert_allclose(result.total, [1.5], rtol=1e-6)
    np.testing.assert_allclose(result.contribution[0], [1.5, 0.0, 0.0], atol=1e-5)
    ## Renormalized to unit mass the local shift is the full 3 m/z
    np.testing.assert_allclose(result.within[0, 0], 3.0, rtol=1e-5)
    np.testing.assert_allclose(result.within[0, 2], 0.0, atol=1e-6)
    assert np.isnan(result.within[0, 1])


def test_contributions_reproduce_the_training_objective_on_random_spectra() -> None:
    axis = 200.0 + 0.55 * (np.arange(1273) + 0.5)
    generator = np.random.default_rng(3)
    inputs = generator.gamma(0.3, size=(16, axis.size)).astype(np.float32)
    outputs = generator.gamma(0.3, size=(16, axis.size)).astype(np.float32)
    result = windowed_masserstein(inputs, outputs, axis, window_edges(axis, 100.0), batch_size=5)
    reference = masserstein_distances(inputs, outputs, axis, batch_size=7)

    np.testing.assert_allclose(result.total, reference, rtol=1e-6)
    np.testing.assert_allclose(result.contribution.sum(axis=1), reference, rtol=1e-4)
    np.testing.assert_allclose(result.input_mass.sum(axis=1), 1.0, rtol=1e-5)
