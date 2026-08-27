"""Tests for latent molecular classification components."""

import pytest
import torch
import torch.nn as nn

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace, TargetBatch

from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.linear_classification_head import (
    LinearClassificationHead,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.multilabel_bce_loss import (
    MSIMultiLabelBCELoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.class_balanced_multilabel_bce_loss import (
    MSIClassBalancedMultiLabelBCELoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.nnpu_multilabel_loss import (
    MSINNPUMultiLabelLoss,
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


def test_class_balanced_bce_uses_train_frequency_and_per_class_masks() -> None:
    """Rare positives are reweighted before a normalized class mean."""
    criterion = MSIClassBalancedMultiLabelBCELoss(
        head_id="molecular",
        target_field="molecule",
        max_positive_weight=10.0,
    )
    training_targets = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )  # (N=4, C=2)
    training_mask = torch.ones_like(training_targets, dtype=torch.bool)  # (N, C)
    criterion._configure_training_statistics(training_targets, training_mask)
    logits = torch.zeros(4, 2, requires_grad=True)  # (B=4, C=2)
    batch = (
        torch.arange(4),
        torch.ones(4, 3),
        {"molecule": training_targets},
        {"molecule": training_mask},
    )

    loss = criterion({"head_molecular": logits}, batch)  # ()
    loss.backward()

    assert torch.allclose(criterion.positive_weights, torch.tensor([3.0, 1.0]))
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert torch.isfinite(logits.grad).all()


def test_nnpu_treats_zero_targets_as_unlabelled_without_explicit_sigmoid() -> None:
    """The nnPU objective applies stable logistic losses directly to logits."""
    criterion = MSINNPUMultiLabelLoss(
        head_id="molecular",
        target_field="molecule",
        prior_method="fixed",
        class_prior=[0.25, 0.5],
    )
    logits = torch.tensor(
        [[1.0, -0.5], [-1.0, 0.5], [0.25, -0.25], [-0.25, 0.25]],
        requires_grad=True,
    )  # (B=4, C=2)
    targets = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]
    )  # (B, C)
    mask = torch.ones_like(targets, dtype=torch.bool)  # (B, C)

    loss = criterion(
        {"head_molecular": logits},
        (torch.arange(4), torch.ones(4, 3), {"molecule": targets}, {"molecule": mask}),
    )  # ()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(criterion.class_priors, torch.tensor([0.25, 0.5]))


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


def test_epoch_average_precision_reports_masked_multilabel_macro_ap() -> None:
    """Epoch AP is computed from logits and respects per-class availability."""

    class FixedHead(nn.Module):
        def forward(self, spectra):
            return {"head_molecule_primary": spectra}

    batch = SpectrumBatch(
        sample_ids=torch.arange(4),
        spectra=torch.tensor([[4.0, -4.0], [-4.0, 4.0], [3.0, -3.0], [-3.0, 3.0]]),
        space=SpectrumSpace(torch.tensor([100.0, 101.0])),
        targets=TargetBatch(
            values={"molecule": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])},
            masks={"molecule": torch.ones(4, 2, dtype=torch.bool)},
            schemas={},
        ),
    )

    value = MSIPyTorchTrainer._evaluate_multilabel_average_precision(
        model=FixedHead(),
        dataloader=[batch],
        preprocessor=None,
        compute_device=torch.device("cpu"),
        head_id="molecule_primary",
        target_field="molecule",
    )

    assert value == pytest.approx(1.0)
