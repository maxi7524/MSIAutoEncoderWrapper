"""Synthetic inverse-binner and coordinate-aware metric tests."""

import numpy as np
import pytest
from types import SimpleNamespace

from msi_autoencoder_wrapper.analysis.autoencoder.binning.analysis import BinningAnalysis
from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.metrics import (
    match_spectral_points,
    peak_collision_rate,
    signal_retention_by_quantile,
    spectral_point_metrics,
)


@pytest.fixture
def binner():
    BinnerManager.discover_strategies()
    return BinnerManager.get_binner("LinearBinning", bin_step=1.0, x_min=100.0, x_max=110.0)


def test_threshold_strategies_and_limit(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("ThresholdInverseBinner", binner=binner, threshold_strategy="relative_to_max", threshold_options={"scale": 0.25}, max_bins=2, track_diagnostics=True)
    x, y = inverse(np.asarray([np.nan, 1, 5, 2, 0, -1, 3, 0, 0, 0]))
    np.testing.assert_array_equal(y, [5, 3]); assert x.size == 2; assert inverse.last_diagnostics["valid_bin_count"] == 4


def test_threshold_diagnostics_are_opt_in(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("ThresholdInverseBinner", binner=binner, threshold_strategy="relative_to_max", threshold_options={"scale": 0.25}, max_bins=2)
    inverse(np.asarray([np.nan, 1, 5, 2, 0, -1, 3, 0, 0, 0]))
    assert inverse.last_diagnostics == {}


def test_cumulative_mass_limit_reports_unreached_target(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("CumulativeMassInverseBinner", binner=binner, retained_fraction=.95, max_bins=1, track_diagnostics=True)
    _, y = inverse(np.asarray([5, 4, 3, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(y, [5]); assert inverse.last_diagnostics["target_fraction_reached"] is False


def test_peak_region_centroid_preserves_region_mass(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("PeakRegionInverseBinner", binner=binner, region_strategy="fixed_window", region_options={"window_size": 1}, reduction_strategy="centroid", max_peaks=1, track_diagnostics=True)
    x, y = inverse(np.asarray([0, 1, 5, 2, 0, 0, 0, 0, 0, 0]))
    assert x == pytest.approx([102.625]); assert y == pytest.approx([8]); assert inverse.last_diagnostics["output_peak_count"] == 1


def test_top_peaks_diagnostics_are_opt_in(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("TopPeaksInverseBinner", binner=binner, max_bins=3, window_size=1)
    inverse(np.asarray([0, 1, 5, 2, 0, 0, 0, 0, 0, 0], dtype=float))
    assert inverse.last_diagnostics == {}
    inverse_tracked = BinnerManager.get_inverse_binner("TopPeaksInverseBinner", binner=binner, max_bins=3, window_size=1, track_diagnostics=True)
    inverse_tracked(np.asarray([0, 1, 5, 2, 0, 0, 0, 0, 0, 0], dtype=float))
    assert inverse_tracked.last_diagnostics["output_peak_count"] > 0


def test_one_to_one_and_local_mass_matching() -> None:
    reference_x = np.asarray([100.0, 101.0]); reference_y = np.asarray([3.0, 2.0]); candidate_x = np.asarray([99.995, 100.005, 101.005]); candidate_y = np.asarray([1.0, 2.0, 2.0])
    one = match_spectral_points(reference_x, reference_y, candidate_x, candidate_y, .01, "Da", "one_to_one")
    local = match_spectral_points(reference_x, reference_y, candidate_x, candidate_y, 100, "ppm", "local_mass")
    assert one.matched_reference_indices.size == 2; assert local.matched_candidate_intensity[0] == pytest.approx(3)
    metrics = spectral_point_metrics(reference_x, reference_y, candidate_x, candidate_y, local)
    assert metrics["peak_recall"] == 1; assert metrics["tic_relative_error"] == pytest.approx(0)


def test_empty_spectra_are_supported() -> None:
    match = match_spectral_points(np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), .01)
    metrics = spectral_point_metrics(np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), match)
    assert metrics["peak_recall"] == 1; assert metrics["peak_precision"] == 1


def test_coordinate_metrics_include_cosine_and_spectral_angle() -> None:
    reference_x = np.asarray([100.0, 101.0]); values = np.asarray([3.0, 4.0])
    match = match_spectral_points(reference_x, values, reference_x + 0.001, values, .01)
    metrics = spectral_point_metrics(reference_x, values, reference_x + 0.001, values, match)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["spectral_angle"] == pytest.approx(0.0)


def test_peak_collision_rate_flags_separately_resolvable_peaks() -> None:
    # Two source peaks 0.07 Da apart both land within 0.05 Da of one candidate bin:
    # they are close enough to be merged but far enough apart to be distinct peaks.
    reference_x = np.asarray([99.97, 100.04, 105.0]); reference_y = np.asarray([10.0, 5.0, 1.0])
    candidate_x = np.asarray([100.0, 105.0]); candidate_y = np.asarray([15.0, 1.0])
    rate = peak_collision_rate(reference_x, reference_y, candidate_x, candidate_y, tolerance=0.05, tolerance_unit="Da")
    assert rate == pytest.approx(0.5)


def test_peak_collision_rate_ignores_peaks_within_one_tolerance_of_each_other() -> None:
    # Two source peaks only 0.02 Da apart: closer together than the tolerance itself,
    # so they are not "separately resolvable" and must not count as a collision.
    reference_x = np.asarray([100.0, 100.02]); reference_y = np.asarray([10.0, 5.0])
    candidate_x = np.asarray([100.01]); candidate_y = np.asarray([15.0])
    rate = peak_collision_rate(reference_x, reference_y, candidate_x, candidate_y, tolerance=0.05, tolerance_unit="Da")
    assert rate == pytest.approx(0.0)


def test_peak_collision_rate_min_relative_height_drops_minor_contributors() -> None:
    reference_x = np.asarray([99.97, 100.04]); reference_y = np.asarray([10.0, 1.0])
    candidate_x = np.asarray([100.0]); candidate_y = np.asarray([11.0])
    rate = peak_collision_rate(reference_x, reference_y, candidate_x, candidate_y, tolerance=0.05, tolerance_unit="Da", min_relative_height=0.5)
    assert rate == pytest.approx(0.0)


def test_unmatched_candidate_intensity_fraction_reports_extra_candidate_mass() -> None:
    reference_x = np.asarray([100.0]); reference_y = np.asarray([5.0])
    candidate_x = np.asarray([100.0, 110.0]); candidate_y = np.asarray([5.0, 5.0])
    match = match_spectral_points(reference_x, reference_y, candidate_x, candidate_y, tolerance=0.01, tolerance_unit="Da", matching_strategy="one_to_one")
    metrics = spectral_point_metrics(reference_x, reference_y, candidate_x, candidate_y, match)
    assert metrics["unmatched_candidate_intensity_fraction"] == pytest.approx(0.5)


def test_signal_retention_by_quantile_separates_dominant_peak_loss_from_noise_loss() -> None:
    # One dominant peak (90) plus a long tail of small peaks (1 each); only the tail matches.
    reference_y = np.asarray([90.0] + [1.0] * 10)
    reference_x = np.arange(len(reference_y), dtype=float) * 10.0
    matched_indices = np.arange(1, len(reference_y))  # every point except the dominant one
    match = match_spectral_points(reference_x, reference_y, reference_x[matched_indices], reference_y[matched_indices], tolerance=0.01, tolerance_unit="Da", matching_strategy="one_to_one")
    retention = signal_retention_by_quantile(reference_y, match, quantiles=(0.5, 0.99))
    assert retention["q50"]["k_peaks"] == 1
    assert retention["q50"]["recall_at_quantile"] == pytest.approx(0.0)  # the dominant peak itself is lost
    assert retention["q99"]["recall_at_quantile"] >= 0.9  # almost everything else is retained


def test_signal_retention_by_quantile_handles_empty_reference() -> None:
    match = match_spectral_points(np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), 0.01)
    retention = signal_retention_by_quantile(np.asarray([]), match, quantiles=(0.5,))
    assert retention["q50"]["k_peaks"] == 0
    assert np.isnan(retention["q50"]["recall_at_quantile"])


def test_forward_sweep_never_invokes_context_inverse_binner(binner) -> None:
    class Reader:
        def GetSpectrum(self, _: int):
            return np.asarray([100.2, 101.2]), np.asarray([3.0, 2.0])

    class ForbiddenInverse:
        def __call__(self, _: np.ndarray):
            raise AssertionError("Forward sweep must not invoke an inverse binner")

    owner = SimpleNamespace(
        reader=Reader(), binner=binner, context=SimpleNamespace(inverse_binner=ForbiddenInverse()),
        default_model_name="model", models={"model": SimpleNamespace(dataset=range(1))},
        selected_ids=lambda ids, dataset: np.asarray([0]),
    )
    records = BinningAnalysis(owner).forward_sweep(
        [0.5, 1.0], tolerance=0.5, normalizations=("raw",)
    )
    assert {record["bin_step"] for record in records} == {0.5, 1.0}
    assert all(record["comparison"] == "binned_original" for record in records)
