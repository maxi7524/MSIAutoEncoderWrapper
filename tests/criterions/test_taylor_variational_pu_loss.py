"""Numerical regression tests for Taylor-VPU and its EMA teacher."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace, TargetBatch
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.taylor_variational_pu_loss import (
    TaylorVariationalPULoss,
)


def _batch() -> SpectrumBatch:
    """Return one positive, one unlabelled, and one explicit N_sim entry."""
    targets = torch.tensor([[1.0], [0.0], [0.0]])
    return SpectrumBatch(
        sample_ids=torch.arange(3),
        spectra=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        space=SpectrumSpace(torch.arange(2, dtype=torch.float32)),
        targets=TargetBatch(
            values={"molecule": targets},
            masks={
                "molecule": torch.ones_like(targets, dtype=torch.bool),
                simulated_negative_mask_key("molecule"): torch.tensor(
                    [[False], [False], [True]]
                ),
            },
            schemas={},
        ),
    )


def test_taylor_vpu_matches_equation_eight_and_excludes_nsim():
    """The core objective is the stated finite Taylor expansion on P/U."""
    batch = _batch()
    logits = torch.tensor([[0.4], [-0.8], [2.0]], requires_grad=True)
    criterion = TaylorVariationalPULoss(
        "ion", "molecule", order=3, consistency_weight=0
    )
    actual = criterion({"head_ion": logits}, batch)
    probability_u = logits[1, 0].sigmoid()
    expected = -F.logsigmoid(logits[0, 0]) - sum(
        (1.0 - probability_u).pow(index) / index for index in range(1, 4)
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad[2, 0] == 0


def test_taylor_vpu_adds_nsim_bce_without_changing_pu_term():
    """Declared reliable negatives contribute only through the configured BCE."""
    batch = _batch()
    logits = torch.zeros(3, 1, requires_grad=True)
    base = TaylorVariationalPULoss("ion", "molecule", consistency_weight=0)
    extended = TaylorVariationalPULoss(
        "ion", "molecule", consistency_weight=0, simulated_negative_weight=2.0
    )
    delta = extended({"head_ion": logits}, batch) - base({"head_ion": logits}, batch)
    torch.testing.assert_close(delta, 2.0 * F.softplus(logits[2, 0]))
    delta.backward()
    assert logits.grad[0, 0] == logits.grad[1, 0] == 0
    assert logits.grad[2, 0] > 0


class _HeadModel(nn.Module):
    """Minimal model exposing a Taylor-VPU-compatible head output."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 1, bias=False)

    def forward(self, spectra: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return one classification logit per input spectrum."""
        return {"head_ion": self.head(spectra)}


def test_taylor_vpu_ema_teacher_tracks_post_step_student_without_gradients():
    """EMA update follows the post-step student and keeps teacher detached."""
    batch = _batch()
    model = _HeadModel()
    with torch.no_grad():
        model.head.weight.fill_(2.0)
    criterion = TaylorVariationalPULoss(
        "ion", "molecule", consistency_weight=1.0, teacher_momentum=0.5
    )
    criterion.on_phase_start(model, dataset=None, transient_cache={})
    assert criterion._teacher is not None
    teacher_before = copy.deepcopy(criterion._teacher.state_dict())
    with torch.no_grad():
        model.head.weight.fill_(6.0)
    criterion.on_optimizer_step(model)
    teacher_weight = criterion._teacher.state_dict()["head.weight"]
    torch.testing.assert_close(teacher_weight, torch.full_like(teacher_weight, 4.0))
    assert all(parameter.grad is None for parameter in criterion._teacher.parameters())
    assert not torch.equal(teacher_before["head.weight"], teacher_weight)


def test_taylor_vpu_consistency_backpropagates_only_to_student():
    """Symmetric KL is finite and differentiates the student model only."""
    batch = _batch()
    model = _HeadModel()
    criterion = TaylorVariationalPULoss("ion", "molecule", consistency_weight=1.0)
    criterion.on_phase_start(model, dataset=None, transient_cache={})
    loss = criterion(model(batch.spectra), batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    assert criterion._teacher is not None
    assert all(parameter.grad is None for parameter in criterion._teacher.parameters())
