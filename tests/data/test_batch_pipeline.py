"""Tests for typed packed and dense MSI batch contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.binners.binners_strategies.linear_binner import (
    LinearBinning,
)
from msi_autoencoder_wrapper.data import (
    BatchPreprocessor,
    LatentBatch,
    RawSpectrumCollator,
    RawSpectrumSample,
    SpectrumSpace,
    TargetSample,
)
from msi_autoencoder_wrapper.metrics import (
    MetricsRegistry,
    MetricsRunner,
    PerClassClassification,
    info_nce,
    mse,
    sobolev,
)


def _samples() -> list[RawSpectrumSample]:
    return [
        RawSpectrumSample(
            sample_id=4,
            mass_values=torch.tensor([0.1, 0.9, 1.2], dtype=torch.float64),
            intensities=torch.tensor([1.0, 2.0, 3.0]),
            targets=TargetSample(
                values={"condition": torch.tensor(0)},
                masks={"condition": torch.tensor(True)},
            ),
        ),
        RawSpectrumSample(
            sample_id=9,
            mass_values=torch.tensor([0.2, 1.8], dtype=torch.float64),
            intensities=torch.tensor([4.0, 5.0]),
            targets=TargetSample(
                values={"condition": torch.tensor(1)},
                masks={"condition": torch.tensor(False)},
            ),
        ),
    ]


def test_raw_collator_packs_variable_length_spectra_and_targets() -> None:
    """Raw points remain unpadded while sample alignment remains explicit."""
    batch = RawSpectrumCollator()(_samples())

    assert torch.equal(batch.sample_ids, torch.tensor([4, 9]))
    assert torch.equal(batch.offsets, torch.tensor([0, 3, 5]))
    assert torch.equal(batch.sample_indices, torch.tensor([0, 0, 0, 1, 1]))
    assert torch.equal(batch.targets.values["condition"], torch.tensor([0, 1]))
    assert torch.equal(batch.targets.masks["condition"], torch.tensor([True, False]))


def test_torch_linear_batch_binning_matches_single_spectrum_scipy() -> None:
    """The packed Torch backend preserves existing linear-bin semantics."""
    binner = LinearBinning(bin_step=1.0, x_min=0.0, x_max=2.0)
    raw = RawSpectrumCollator()(_samples())

    dense = binner.transform_batch(raw)
    expected = np.stack(
        [
            binner(
                sample.mass_values.numpy(),
                sample.intensities.numpy(),
            )
            for sample in _samples()
        ]
    )

    assert dense.spectra.shape == (2, 2)
    assert torch.allclose(dense.spectra, torch.from_numpy(expected).float())
    assert torch.allclose(dense.space.mass_axis, torch.tensor([0.5, 1.5], dtype=torch.float64))
    assert dense.targets is raw.targets


def test_batch_preprocessor_bins_and_normalizes_on_selected_device() -> None:
    """Preprocessing owns binning and normalization before model execution."""
    class Context:
        binner = LinearBinning(bin_step=1.0, x_min=0.0, x_max=2.0)

    class Dataset:
        active_context = Context()
        normalization = "tic"

        @staticmethod
        def normalize_batch(values: torch.Tensor) -> torch.Tensor:
            return values / values.sum(dim=1, keepdim=True)

    raw = RawSpectrumCollator()(_samples())

    dense = BatchPreprocessor(Dataset(), "cpu", "cpu")(raw)

    assert dense.device.type == "cpu"
    assert dense.space.device.type == "cpu"
    assert dense.space.normalization == "tic"
    assert torch.allclose(dense.spectra.sum(dim=1), torch.ones(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_linear_batch_binning_matches_between_cpu_and_cuda() -> None:
    """CPU and CUDA preprocessing use the same numerical implementation."""
    binner = LinearBinning(bin_step=1.0, x_min=0.0, x_max=2.0)
    raw = RawSpectrumCollator()(_samples())

    cpu = binner.transform_batch(raw)
    cuda = binner.transform_batch(raw.to("cuda"))

    assert torch.allclose(cpu.spectra, cuda.spectra.cpu())
    assert torch.allclose(cpu.space.mass_axis, cuda.space.mass_axis.cpu())


def test_spectrum_metrics_are_batch_invariant_and_axis_aware() -> None:
    """Metrics return one value per sample and Sobolev consumes physical spacing."""
    target = torch.tensor([[0.0, 1.0, 3.0], [1.0, 1.0, 1.0]])
    prediction = target + torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])
    axis = torch.tensor([100.0, 100.5, 102.0])

    batched_mse = mse(prediction, target)
    separate_mse = torch.stack(
        [mse(prediction[index], target[index])[0] for index in range(2)]
    )

    assert torch.allclose(batched_mse, separate_mse)
    assert sobolev(prediction, target, mass_axis=axis).shape == (2,)


def test_info_nce_returns_one_value_for_each_pair_direction() -> None:
    """Paired embedding metrics preserve the logical batch cardinality."""
    original = torch.eye(3)
    augmented = original.clone()

    values = info_nce(original, augmented, temperature=0.1)

    assert values.shape == (6,)
    assert bool(torch.isfinite(values).all())


def test_per_class_f1_accumulates_counts_across_batches() -> None:
    """Dataset-level class metrics do not average independently computed batches."""
    metric = PerClassClassification("f1", threshold=0.5)
    metric.update(
        torch.tensor([[10.0, -10.0], [-10.0, 10.0]]),
        torch.tensor([[1, 0], [0, 1]]),
    )
    metric.update(
        torch.tensor([[10.0, 10.0]]),
        torch.tensor([[0, 1]]),
    )

    assert torch.allclose(metric.compute(), torch.tensor([2.0 / 3.0, 1.0], dtype=torch.float64))


def test_spectrum_space_is_shared_without_axis_expansion() -> None:
    """A spectral axis remains one-dimensional instead of repeating per sample."""
    space = SpectrumSpace(torch.linspace(1000.0, 1600.0, 60001))

    assert space.mass_axis.shape == (60001,)
    assert space.feature_count == 60001

    latent = LatentBatch(
        sample_ids=torch.tensor([1, 2]),
        embeddings=torch.ones(2, 4),
        reconstruction_space=space,
    )
    assert latent.reconstruction_space is space


def test_metric_registry_resolves_names_within_object_spaces() -> None:
    """Analysis code can execute metrics by names without criterion imports."""
    target = torch.tensor([[0.0, 1.0]])
    prediction = torch.tensor([[1.0, 1.0]])

    result = MetricsRunner.compute("mse", "spectrum", prediction, target)

    assert result.item() == pytest.approx(0.5)
    assert "masserstein" in MetricsRegistry.available("spectrum")
