"""Tests for registered forward and inverse binning implementations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.utils.exceptions import ValidationError
from msi_autoencoder_wrapper.data import RawSpectrumCollator, RawSpectrumSample, TargetSample, SpectrumBatch, SpectrumSpace


def test_binner_discovery_and_manager_factories() -> None:
    """Discovery and shared factory resolution create both binner directions."""
    BinnerManager.discover_strategies()
    binner = BinnerManager.get_binner(
        "LinearBinning", bin_step=1.0, x_min=100.0, x_max=110.0
    )
    inverse = BinnerManager.get_inverse_binner(
        "TopPeaksInverseBinner", binner=binner, max_peaks=3
    )

    assert binner.GetXAxisDepth() == 10
    assert inverse.get_config() == {"max_peaks": 3, "min_peak_distance": 1}


def test_binner_manager_accepts_classes_and_ready_instances() -> None:
    """Manual binner objects pass through the same manager without recreation."""
    binner_class = BinnerManager.BINNER_REGISTRY["LinearBinning"]
    from_class = BinnerManager.get_binner(
        binner_class, bin_step=1.0, x_min=100.0, x_max=110.0
    )
    from_instance = BinnerManager.get_binner(from_class)

    assert isinstance(from_class, binner_class)
    assert from_instance is from_class


def test_linear_binner_sums_values_on_a_regular_grid() -> None:
    """Forward binning aggregates in-range intensities and ignores out-of-range values."""
    binner = BinnerManager.get_binner(
        "LinearBinning", bin_step=1.0, x_min=100.0, x_max=110.0
    )
    raw = RawSpectrumCollator()([RawSpectrumSample(
        sample_id=0,
        mass_values=torch.tensor([100.2, 100.8, 105.2, 111.0]),
        intensities=torch.tensor([10.0, 20.0, 5.0, 100.0]),
        targets=TargetSample.empty(),
    )])
    result = binner(raw).spectra[0]

    assert result.shape == (10,)
    assert result[0] == pytest.approx(30.0)
    assert result[5] == pytest.approx(5.0)
    assert result.sum() == pytest.approx(35.0)


def test_top_peaks_inverse_binner_preserves_peak_neighborhood() -> None:
    """Inverse binning selects the configured number of points around the top peak."""
    binner = BinnerManager.get_binner(
        "LinearBinning", bin_step=1.0, x_min=100.0, x_max=110.0
    )
    inverse = BinnerManager.get_inverse_binner(
        "TopPeaksNeighbourhoodInverseBinner", binner=binner, max_peaks=1,
        region_options={"window_size": 1},
    )
    intensities = np.zeros(10)
    intensities[5] = 10.0

    batch = SpectrumBatch(
        sample_ids=torch.tensor([0]),
        spectra=torch.as_tensor(intensities).unsqueeze(0),
        space=SpectrumSpace(mass_axis=torch.as_tensor(binner.GetXAxis())),
    )
    result = inverse(batch)

    assert result.mass_values.numel() == 3
    assert result.intensities.max() == pytest.approx(10.0)


def test_binner_selects_inclusive_mass_ranges() -> None:
    """Mass-window lookup belongs to the configured binner grid."""
    binner = BinnerManager.get_binner(
        "LinearBinning", bin_step=1.0, x_min=100.0, x_max=110.0
    )

    np.testing.assert_array_equal(binner.GetBinIndices(102.5, 1.0), [1, 2, 3])
    np.testing.assert_array_equal(binner.GetMzRangeIndices(104.5, 106.5), [4, 5, 6])

    with pytest.raises(ValidationError, match="tolerance cannot be negative"):
        binner.GetBinIndices(102.5, -0.1)


def test_missing_binner_context_uses_global_validation_error() -> None:
    """Missing boundary and inverse dependencies use standardized validation errors."""
    with pytest.raises(ValidationError, match=r"\[LINEARBINNING ERROR\]"):
        BinnerManager.get_binner("LinearBinning", bin_step=1.0)

    with pytest.raises(ValidationError, match=r"\[INVERSEBINNER ERROR\]"):
        BinnerManager.get_inverse_binner("TopPeaksInverseBinner")
