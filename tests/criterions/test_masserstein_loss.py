"""Tests for the differentiable Masserstein reconstruction objective."""

from __future__ import annotations

import pytest
import torch

from msi_autoencoder_wrapper.training.criterions.autoencoder.reconstruction.masserstein_loss import (
    MSIMassersteinLoss,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def _loss() -> MSIMassersteinLoss:
    """Return a small-axis Wasserstein loss used by analytical examples."""
    return MSIMassersteinLoss()


def test_masserstein_is_zero_for_identical_spectra_and_differentiable() -> None:
    """Debiased transport is zero at identity and retains an autograd graph."""
    original = torch.tensor([[0.1, 0.3, 0.6, 0.0]])
    reconstruction = original.clone().requires_grad_(True)

    value = _loss()(
        {"reconstruction": reconstruction},
        (torch.arange(1), original),
    )
    value.backward()

    assert value.item() == pytest.approx(0.0, abs=1e-5)
    assert reconstruction.grad is not None


def test_masserstein_uses_physical_transport_distance() -> None:
    """A one-bin shift costs one axis unit when direct transport is cheaper."""
    original = torch.tensor([[1.0, 0.0, 0.0]])
    reconstruction = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)

    value = _loss()(
        {"reconstruction": reconstruction},
        (torch.arange(1), original),
    )

    assert value.item() == pytest.approx(1.0, abs=1e-4)


def test_masserstein_uses_existing_tic_strategy_for_raw_inputs() -> None:
    """Temporary TIC normalization makes the raw-spectrum cost scale-invariant."""
    metric = _loss().metric
    target = torch.tensor([[2.0, 0.0, 0.0]])
    prediction = torch.tensor([[0.0, 0.0, 5.0]])

    value = metric(prediction, target)

    assert value.item() == pytest.approx(2.0, abs=1e-5)


def test_masserstein_uses_tic_representation_without_second_normalization() -> None:
    """TIC-normalized input space is passed directly to the transport formula."""
    metric = _loss().metric
    target = torch.tensor([[1.0, 0.0, 0.0]])
    prediction = torch.tensor([[0.0, 0.0, 1.0]])

    value = metric(
        prediction,
        target,
        inputs_tic_normalized=True,
    )

    assert value.item() == pytest.approx(2.0, abs=1e-5)


def test_masserstein_uses_cumulative_transport_for_distant_signal() -> None:
    """A unit signal moved across three physical units costs three units."""
    original = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reconstruction = torch.tensor([[0.0, 0.0, 0.0, 1.0]], requires_grad=True)

    value = _loss()(
        {"reconstruction": reconstruction},
        (torch.arange(1), original),
    )

    assert value.item() == pytest.approx(3.0, abs=1e-5)


def test_masserstein_parameters_use_global_validation_errors() -> None:
    """Invalid active settings use the shared exception format."""
    with pytest.raises(ValidationError, match="axis_step"):
        MSIMassersteinLoss(axis_step=0.0)


def test_masserstein_metric_rejects_negative_values_instead_of_clamping() -> None:
    """Invalid intensities are reported instead of silently changing the data."""
    metric = _loss().metric

    with pytest.raises(ValidationError, match="non-negative"):
        metric(
            torch.tensor([[0.5, -0.1]]),
            torch.tensor([[0.5, 0.1]]),
        )


def test_masserstein_batches_regular_high_resolution_axes(caplog) -> None:
    """A high-resolution grid is evaluated by the linear cumsum path."""
    original = torch.rand(4, 2048)
    reconstruction = torch.rand(4, 2048, requires_grad=True)
    loss = MSIMassersteinLoss(
        axis_step=0.1,
    )

    value = loss(
        {"reconstruction": reconstruction},
        (torch.arange(4), original),
    )
    value.backward()

    assert torch.isfinite(value)
    assert reconstruction.grad is not None
    assert "irregular axis" not in caplog.text


def test_masserstein_prepares_axis_widths_once() -> None:
    """A configured binner axis produces reusable physical bin widths."""
    metric = MSIMassersteinLoss(
        axis_step=0.1,
    ).metric
    axis = torch.arange(32, dtype=torch.float32) * 0.1
    metric.set_mass_axis(axis)

    width_pointer = metric._bin_widths.data_ptr()
    value = metric(torch.rand(2, 32), torch.rand(2, 32))

    assert metric.has_mass_axis
    assert metric._bin_widths.data_ptr() == width_pointer
    assert metric._bin_widths.shape == (31,)
    assert torch.isfinite(value)


def test_masserstein_rejects_axis_dimension_mismatch_after_configuration() -> None:
    """A prepared geometry cannot be applied to a different feature space."""
    metric = MSIMassersteinLoss().metric
    metric.set_mass_axis(torch.arange(4, dtype=torch.float32))

    with pytest.raises(ValidationError, match="configured mass axis"):
        metric(torch.rand(1, 3), torch.rand(1, 3))


def test_masserstein_rejects_changed_axis_after_configuration() -> None:
    """A changed same-size axis requires explicit geometry reconfiguration."""
    metric = MSIMassersteinLoss().metric
    metric.set_mass_axis(torch.arange(4, dtype=torch.float32))

    with pytest.raises(ValidationError, match="differs from the configured"):
        metric(
            torch.rand(1, 4),
            torch.rand(1, 4),
            mass_axis=torch.arange(4, dtype=torch.float32) * 0.5,
        )
