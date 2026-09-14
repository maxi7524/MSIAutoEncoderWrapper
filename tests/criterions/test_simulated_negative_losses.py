"""P/U/N_sim losses use only explicitly declared reliable negatives."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace, TargetBatch
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.nnpu_multilabel_loss import (
    MSINNPUMultiLabelLoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.simulated_negative_losses import (
    BCEPNLoss,
    ObservedPNUSCrossEntropyLoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.variational_pu_loss import (
    VariationalPULoss,
)


def _batch() -> SpectrumBatch:
    """Return two positives, two N_sim entries, and two unlabelled entries."""
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    simulated_negative = torch.tensor(
        [[False, False], [True, True], [False, False]]
    )
    return SpectrumBatch(
        sample_ids=torch.arange(3),
        spectra=torch.ones(3, 4),
        space=SpectrumSpace(torch.arange(4, dtype=torch.float32)),
        targets=TargetBatch(
            values={"molecule": targets},
            masks={
                "molecule": torch.ones_like(targets, dtype=torch.bool),
                simulated_negative_mask_key("molecule"): simulated_negative,
            },
            schemas={},
        ),
    )


def test_nnpu_excludes_nsim_from_pu_risk_and_adds_it_only_when_requested():
    """N_sim has no nnPU gradient unless the explicit BCE extension is enabled."""
    batch = _batch()
    logits = torch.zeros(3, 2, requires_grad=True)
    plain = MSINNPUMultiLabelLoss(
        "ion", "molecule", prior_method="fixed", class_prior=0.4
    )
    plain({"head_ion": logits}, batch).backward()
    torch.testing.assert_close(logits.grad[1], torch.zeros(2))

    extended_logits = torch.zeros(3, 2, requires_grad=True)
    extended = MSINNPUMultiLabelLoss(
        "ion",
        "molecule",
        prior_method="fixed",
        class_prior=0.4,
        simulated_negative_weight=1.0,
    )
    extended({"head_ion": extended_logits}, batch).backward()
    assert bool((extended_logits.grad[1] > 0).all())


def test_vpu_excludes_nsim_from_its_marginal_and_uses_separate_bce_term():
    """The VPU marginal is P/U, while the optional N_sim term is independent."""
    batch = _batch()
    logits = torch.zeros(3, 2, requires_grad=True)
    base = VariationalPULoss("ion", "molecule", consistency_weight=0)
    base({"head_ion": logits}, batch).backward()
    torch.testing.assert_close(logits.grad[1], torch.zeros(2))

    extended_logits = torch.zeros(3, 2, requires_grad=True)
    extended = VariationalPULoss(
        "ion",
        "molecule",
        consistency_weight=0,
        simulated_negative_weight=1.0,
    )
    extended({"head_ion": extended_logits}, batch).backward()
    assert bool((extended_logits.grad[1] > 0).all())


def test_pn_bce_ignores_u_and_observed_ce_preserves_three_way_contract():
    """Strict BCE excludes U, whereas observed CE labels N_sim, P, and U."""
    batch = _batch()
    binary_logits = torch.zeros(3, 2, requires_grad=True)
    BCEPNLoss("ion", "molecule")({"head_ion": binary_logits}, batch).backward()
    torch.testing.assert_close(binary_logits.grad[2], torch.zeros(2))

    three_way_logits = torch.zeros(3, 2, 3, requires_grad=True)
    loss = ObservedPNUSCrossEntropyLoss("ion", "molecule")(
        {"head_ion": three_way_logits}, batch
    )
    torch.testing.assert_close(loss, torch.tensor(3.0).log())
    loss.backward()
    assert torch.isfinite(three_way_logits.grad).all()
