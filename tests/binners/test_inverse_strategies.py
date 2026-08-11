"""Tests for the canonical vectorized inverse-binner API."""

from __future__ import annotations

import pytest
import torch

from msi_autoencoder_wrapper.binners import BinnerManager
from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace


@pytest.fixture
def binner():
    BinnerManager.discover_strategies()
    return BinnerManager.get_binner("LinearBinning", bin_step=1.0, x_min=100.0, x_max=110.0)


def _batch(binner, values: torch.Tensor) -> SpectrumBatch:
    return SpectrumBatch(
        sample_ids=torch.arange(values.shape[0]),
        spectra=values,
        space=SpectrumSpace(mass_axis=torch.as_tensor(binner.GetXAxis(), dtype=values.dtype)),
    )


def _row(result, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    start, end = int(result.offsets[index]), int(result.offsets[index + 1])
    return result.mass_values[start:end], result.intensities[start:end]


def test_quantile_projects_selected_points_to_irregular_axis(binner) -> None:
    axis = torch.tensor([100.1, 100.6, 101.7, 104.6, 109.7], dtype=torch.float64)
    inverse = BinnerManager.get_inverse_binner(
        "QuantileInverseBinner", binner=binner, quantile=0.75,
        reconstruction_mass_axis=axis,
    )
    batch = _batch(binner, torch.tensor([[0.0, 1.0, 2.0, 3.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64))

    result = inverse(batch)
    masses, intensities = _row(result, 0)

    assert torch.equal(masses, torch.tensor([101.7, 104.6], dtype=torch.float64))
    assert torch.equal(intensities, torch.tensor([2.0, 13.0], dtype=torch.float64))
    assert torch.equal(result.reconstruction_space.mass_axis, axis)


def test_top_peaks_none_and_oversized_limit_keep_all_peaks(binner) -> None:
    values = torch.tensor([[0.0, 5.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
    batch = _batch(binner, values)
    target_axis = torch.linspace(100.0, 110.0, 101)
    all_peaks = BinnerManager.get_inverse_binner(
        "TopPeaksInverseBinner", binner=binner, max_peaks=None,
        reconstruction_mass_axis=target_axis,
    )(batch)
    oversized = BinnerManager.get_inverse_binner(
        "TopPeaksInverseBinner", binner=binner, max_peaks=1000,
        reconstruction_mass_axis=target_axis,
    )(batch)

    assert torch.equal(all_peaks.mass_values, oversized.mass_values)
    assert torch.equal(all_peaks.intensities, torch.tensor([5.0, 4.0, 3.0]))


def test_top_peaks_limit_selects_strongest_maxima(binner) -> None:
    values = torch.tensor([[0.0, 5.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
    inverse = BinnerManager.get_inverse_binner("TopPeaksInverseBinner", binner=binner, max_peaks=2)

    result = inverse(_batch(binner, values))

    assert torch.equal(result.intensities, torch.tensor([5.0, 4.0]))


def test_top_peaks_collapses_plateau_to_lower_middle_index(binner) -> None:
    values = torch.tensor([[0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    inverse = BinnerManager.get_inverse_binner("TopPeaksInverseBinner", binner=binner)

    result = inverse(_batch(binner, values))

    assert torch.equal(result.mass_values, torch.tensor([102.5]))


def test_neighbourhood_expands_union_on_reconstruction_axis(binner) -> None:
    values = torch.tensor([[0.0, 5.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target_axis = torch.linspace(100.0, 110.0, 101)
    inverse = BinnerManager.get_inverse_binner(
        "TopPeaksNeighbourhoodInverseBinner", binner=binner, max_peaks=None,
        region_strategy="fixed_window", region_options={"window_size": 1},
        reconstruction_mass_axis=target_axis,
    )

    result = inverse(_batch(binner, values))

    assert result.mass_values.numel() > 3
    assert torch.unique(result.mass_values).numel() == result.mass_values.numel()


def test_batch_transform_equals_concatenated_single_row_transforms(binner) -> None:
    values = torch.tensor([
        [0.0, 5.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 8.0, 0.0, 2.0, 0.0, 7.0, 0.0, 0.0],
    ])
    inverse = BinnerManager.get_inverse_binner("TopPeaksInverseBinner", binner=binner)

    combined = inverse(_batch(binner, values))
    separate = [inverse(_batch(binner, row.unsqueeze(0))) for row in values]

    for index, expected in enumerate(separate):
        masses, intensities = _row(combined, index)
        assert torch.equal(masses, expected.mass_values)
        assert torch.equal(intensities, expected.intensities)


def test_empty_batch_is_supported(binner) -> None:
    inverse = BinnerManager.get_inverse_binner("TopPeaksInverseBinner", binner=binner)
    result = inverse(_batch(binner, torch.empty((0, binner.GetXAxisDepth()))))

    assert result.offsets.tolist() == [0]
    assert result.mass_values.numel() == 0


def test_shared_reader_axis_is_the_default_reconstruction_axis(binner) -> None:
    expected = torch.tensor([100.0, 100.4, 101.3, 109.9], dtype=torch.float64)

    class Reader:
        capabilities = type("Capabilities", (), {"shared_mass_axis": True})()

        @staticmethod
        def GetXAxis():
            return expected.numpy()

    context = type("Context", (), {"reader": Reader(), "binner": binner})()
    inverse = BinnerManager.get_inverse_binner(
        "TopPeaksInverseBinner", binner=binner, active_context=context
    )

    assert torch.equal(inverse.reconstruction_mass_axis, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_inverse_transform_matches_between_cpu_and_cuda(binner) -> None:
    values = torch.tensor([[0.0, 5.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
    inverse = BinnerManager.get_inverse_binner("TopPeaksInverseBinner", binner=binner)
    batch = _batch(binner, values)

    cpu = inverse(batch)
    cuda = inverse(batch.to("cuda"))

    assert torch.equal(cpu.offsets, cuda.offsets.cpu())
    assert torch.equal(cpu.mass_values, cuda.mass_values.cpu())
    assert torch.equal(cpu.intensities, cuda.intensities.cpu())
