"""Tests for criterion discovery, creation, and composition."""

from __future__ import annotations

import pytest
import torch

from msi_autoencoder_wrapper.training.criterions.criterions_manager import (
    CriterionsManager,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.contrastive.infoNCE_loss import (
    MSIInfoNCELoss,
)
from msi_autoencoder_wrapper.utils.exceptions import IncompatibleInterfaceError


def test_criterion_discovery_returns_uniform_component_information() -> None:
    """Criterion discovery populates the model-scoped registry and metadata shape."""
    CriterionsManager.discover_criterions()
    available = CriterionsManager.get_available_criterions("autoencoder")

    assert set(available) == {"reconstruction", "contrastive", "head"}
    assert "MSELoss" in available["reconstruction"]
    assert "SobolevLoss" in available["reconstruction"]
    assert "InfoNCELoss" in available["contrastive"]
    assert "MultiLabelBCELoss" in available["head"]
    assert set(available["reconstruction"]["MSELoss"]) == {
        "docstring",
        "parameters",
    }
    assert available["reconstruction"]["MSELoss"]["parameters"]["reduction"] == "mean"


@pytest.mark.parametrize("criterion_name", ["MSELoss", "SobolevLoss"])
def test_reconstruction_criterions_return_differentiable_scalars(criterion_name: str) -> None:
    """Reconstruction criteria accept the standardized model output mapping."""
    criterion = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"][criterion_name]()
    original = torch.randn(4, 16)
    reconstruction = torch.randn(4, 16, requires_grad=True)

    loss = criterion({"reconstruction": reconstruction}, (torch.arange(4), original))
    loss.backward()

    assert loss.ndim == 0
    assert reconstruction.grad is not None
    assert isinstance(criterion.GetConfig(), dict)


def test_composite_loss_combines_registered_criterions() -> None:
    """The manager builds a weighted loss and exposes component metrics."""
    composite = CriterionsManager.build_composite_loss(
        "autoencoder",
        {
            "reconstruction": {
                "MSELoss": {"weight": 1.0, "params": {}},
                "SobolevLoss": {"weight": 0.25, "params": {}},
            },
        },
    )
    original = torch.randn(4, 16)
    reconstruction = torch.randn(4, 16, requires_grad=True)

    loss, metrics = composite(
        {"reconstruction": reconstruction}, (torch.arange(4), original)
    )

    assert loss.ndim == 0
    assert set(metrics) == {"MSELoss", "SobolevLoss", "total_loss"}
    loss.backward()
    assert reconstruction.grad is not None


def test_composite_loss_accepts_ready_criterion_instance() -> None:
    """A user-created criterion is retained in the composite loss."""
    criterion = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"]["MSELoss"]()

    composite = CriterionsManager.build_composite_loss(
        "autoencoder",
        {"reconstruction": {"custom_mse": criterion}},
    )

    assert composite.loss_functions["custom_mse"] is criterion


def test_criterion_output_contract_uses_global_interface_error() -> None:
    """Missing model output fields raise the standardized interface exception."""
    criterion_class = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"]["MSELoss"]
    with pytest.raises(
        IncompatibleInterfaceError,
        match=r"\[RECONSTRUCTIONCRITERION INTERFACE ERROR\]",
    ):
        criterion_class()({}, (torch.arange(1), torch.zeros(1, 4)))


def test_info_nce_builds_peak_bank_and_routes_augmented_projection() -> None:
    """Contrastive hooks sample envelopes, double the batch, and backpropagate."""

    class PeakDataset:
        def __len__(self) -> int:
            return 6

        def __getitem__(self, index: int):
            spectrum = torch.zeros(16)
            spectrum[3:6] = torch.tensor([0.5, 2.0 + index, 0.5])
            spectrum[10:13] = torch.tensor([0.25, 1.0, 0.25])
            return index, spectrum

    criterion = MSIInfoNCELoss(
        max_peaks_per_spectrum=1,
        peak_sample_size=4,
        peak_sample_seed=3,
    )
    cache = {}
    criterion.on_phase_start(torch.nn.Identity(), PeakDataset(), cache)
    batch = (
        torch.arange(4),
        torch.stack([PeakDataset()[index][1] for index in range(4)]),
    )
    augmented_batch = criterion.on_batch_start(batch, cache)
    projection = torch.randn(8, 3, requires_grad=True)

    value = criterion({"projection": projection}, augmented_batch)
    value.backward()

    assert cache["chemical_peak_bank"]
    assert augmented_batch[1].shape == (8, 16)
    assert torch.isfinite(value)
    assert projection.grad is not None
