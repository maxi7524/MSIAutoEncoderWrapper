"""Tests for the differentiable Masserstein reconstruction objective."""

from __future__ import annotations

import pytest
import torch

from msi_autoencoder_wrapper.training.criterions.autoencoder.reconstruction.masserstein_loss import (
    MSIMassersteinLoss,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def _loss(denoising_penalty: float = 10.0) -> MSIMassersteinLoss:
    """Return a precise small-axis loss used by analytical examples."""
    return MSIMassersteinLoss(
        denoising_penalty=denoising_penalty,
        entropy_regularization=0.01,
        sinkhorn_iterations=80,
        debias=True,
    )


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

    value = _loss(denoising_penalty=10.0)(
        {"reconstruction": reconstruction},
        (torch.arange(1), original),
    )

    assert value.item() == pytest.approx(1.0, abs=1e-4)


def test_masserstein_prefers_auxiliary_point_for_distant_signal() -> None:
    """Distant signal is destroyed and recreated for a total cost of two kappa."""
    original = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reconstruction = torch.tensor([[0.0, 0.0, 0.0, 1.0]], requires_grad=True)

    value = _loss(denoising_penalty=1.0)(
        {"reconstruction": reconstruction},
        (torch.arange(1), original),
    )

    assert value.item() == pytest.approx(2.0, abs=1e-2)


def test_masserstein_parameters_use_global_validation_errors() -> None:
    """Invalid optimal-transport settings use the shared exception format."""
    with pytest.raises(ValidationError, match="denoising_penalty"):
        MSIMassersteinLoss(denoising_penalty=0.0)
