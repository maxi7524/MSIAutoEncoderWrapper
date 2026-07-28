"""Tests for latent molecular classification components."""

import torch

from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.linear_classification_head import (
    LinearClassificationHead,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.multilabel_bce_loss import (
    MSIMultiLabelBCELoss,
)


def test_linear_classification_head_returns_class_logits() -> None:
    head = LinearClassificationHead(latent_dim=8, output_dim=3, hidden_dim=4)

    logits = head(torch.zeros(2, 8))

    assert logits.shape == (2, 3)


def test_multilabel_loss_reads_pixel_dataset_target_dictionary() -> None:
    criterion = MSIMultiLabelBCELoss()
    logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    batch = (
        torch.tensor([0]),
        torch.ones(1, 4),
        {"molecule": torch.tensor([[1.0, 0.0]])},
    )

    loss = criterion({"head_molecule": logits}, batch)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
