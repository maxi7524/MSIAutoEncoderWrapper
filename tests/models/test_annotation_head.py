"""Tests for latent molecular classification components."""

import torch
import torch.nn as nn

from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.linear_classification_head import (
    LinearClassificationHead,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.multilabel_bce_loss import (
    MSIMultiLabelBCELoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.cross_entropy_loss import (
    MSIMaskedCrossEntropyLoss,
)
from msi_autoencoder_wrapper.training.engine.base_trainer import MSIPyTorchTrainer


def test_linear_classification_head_returns_class_logits() -> None:
    head = LinearClassificationHead(latent_dim=8, output_dim=3, hidden_dim=4)

    logits = head(torch.zeros(2, 8))

    assert logits.shape == (2, 3)


def test_multilabel_loss_reads_pixel_dataset_target_dictionary() -> None:
    criterion = MSIMultiLabelBCELoss(head_id="molecular", target_field="molecule")
    logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    batch = (
        torch.tensor([0]),
        torch.ones(1, 4),
        {"molecule": torch.tensor([[1.0, 0.0]])},
        {"molecule": torch.tensor([True])},
    )

    loss = criterion({"head_molecular": logits}, batch)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_cross_entropy_ignores_missing_head_targets() -> None:
    criterion = MSIMaskedCrossEntropyLoss(
        head_id="condition_a", target_field="condition"
    )
    logits = torch.randn(2, 3, requires_grad=True)
    batch = (
        torch.tensor([0, 1]),
        torch.ones(2, 4),
        {"condition": torch.tensor([2, 0])},
        {"condition": torch.tensor([True, False])},
    )

    loss = criterion({"head_condition_a": logits}, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.equal(logits.grad[1], torch.zeros(3))


def test_phase_freeze_can_select_one_named_head() -> None:
    model = nn.Module()
    model.encoder = nn.Linear(3, 3)
    model.heads = nn.ModuleDict(
        {"first": nn.Linear(3, 2), "second": nn.Linear(3, 2)}
    )

    MSIPyTorchTrainer._apply_freeze_configuration(
        model, ["encoder", "heads.first"]
    )

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.heads["first"].parameters())
    assert all(parameter.requires_grad for parameter in model.heads["second"].parameters())
