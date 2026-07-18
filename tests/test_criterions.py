"""Tests for criterion discovery, creation, and composition."""

from __future__ import annotations

import pytest
import torch

from msi_autoencoder_wrapper.training.criterions.criterions_manager import (
    CriterionsManager,
)
from msi_autoencoder_wrapper.utils.exceptions import IncompatibleInterfaceError


def test_criterion_discovery_returns_uniform_component_information() -> None:
    """Criterion discovery populates the model-scoped registry and metadata shape."""
    CriterionsManager.discover_criterions()

    available = CriterionsManager.get_available_criterions("autoencoder")

    assert "MSELoss" in available
    assert "SobolevLoss" in available
    assert set(available["MSELoss"]) == {"docstring", "parameters"}
    assert available["MSELoss"]["parameters"]["reduction"] == "mean"


@pytest.mark.parametrize("criterion_name", ["MSELoss", "SobolevLoss"])
def test_reconstruction_criterions_return_differentiable_scalars(criterion_name: str) -> None:
    """Reconstruction criteria accept the current standardized model output mapping."""
    criterion_class = CriterionsManager._REGISTRY["autoencoder"][criterion_name]
    criterion = criterion_class()
    original = torch.randn(4, 16)
    reconstruction = torch.randn(4, 16, requires_grad=True)

    loss = criterion(
        {"reconstruction": reconstruction},
        (torch.arange(4), original),
    )
    loss.backward()

    assert loss.ndim == 0
    assert reconstruction.grad is not None
    assert isinstance(criterion.GetConfig(), dict)


def test_composite_loss_combines_registered_criterions() -> None:
    """The manager builds a weighted loss and exposes component-level metrics."""
    composite = CriterionsManager.build_composite_loss(
        model_type="autoencoder",
        loss_setup={
            "MSELoss": {"weight": 1.0, "params": {}},
            "SobolevLoss": {"weight": 0.25, "params": {}},
        },
    )
    original = torch.randn(4, 16)
    reconstruction = torch.randn(4, 16, requires_grad=True)

    loss, metrics = composite(
        {"reconstruction": reconstruction},
        (torch.arange(4), original),
    )

    assert loss.ndim == 0
    assert set(metrics) == {"MSELoss", "SobolevLoss", "total_loss"}
    loss.backward()
    assert reconstruction.grad is not None


def test_criterion_output_contract_uses_global_interface_error() -> None:
    """Missing model output fields raise the standardized interface exception."""
    criterion_class = CriterionsManager._REGISTRY["autoencoder"]["MSELoss"]

    with pytest.raises(IncompatibleInterfaceError, match=r"\[MSELOSS INTERFACE ERROR\]"):
        criterion_class()({}, (torch.arange(1), torch.zeros(1, 4)))
