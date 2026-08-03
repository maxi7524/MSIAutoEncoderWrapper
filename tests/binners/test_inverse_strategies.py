"""Synthetic inverse-binner and coordinate-aware metric tests."""

import numpy as np
import pytest
from types import SimpleNamespace

from msi_autoencoder_wrapper.analysis.autoencoder.binning.analysis import BinningAnalysis
from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.metrics import match_spectral_points, spectral_point_metrics


@pytest.fixture
def binner():
    BinnerManager.discover_strategies()
    return BinnerManager.get_binner("LinearBinning", bin_step=1.0, x_min=100.0, x_max=110.0)


def test_threshold_strategies_and_limit(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("ThresholdInverseBinner", binner=binner, threshold_strategy="relative_to_max", threshold_options={"scale": 0.25}, max_bins=2)
    x, y = inverse(np.asarray([np.nan, 1, 5, 2, 0, -1, 3, 0, 0, 0]))
    np.testing.assert_array_equal(y, [5, 3]); assert x.size == 2; assert inverse.last_diagnostics["valid_bin_count"] == 4


def test_cumulative_mass_limit_reports_unreached_target(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("CumulativeMassInverseBinner", binner=binner, retained_fraction=.95, max_bins=1)
    _, y = inverse(np.asarray([5, 4, 3, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(y, [5]); assert inverse.last_diagnostics["target_fraction_reached"] is False


def test_peak_region_centroid_preserves_region_mass(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("PeakRegionInverseBinner", binner=binner, region_strategy="fixed_window", region_options={"window_size": 1}, reduction_strategy="centroid", max_peaks=1)
    x, y = inverse(np.asarray([0, 1, 5, 2, 0, 0, 0, 0, 0, 0]))
    assert x == pytest.approx([102.625]); assert y == pytest.approx([8]); assert inverse.last_diagnostics["output_peak_count"] == 1


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
