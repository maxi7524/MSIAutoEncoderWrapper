"""Regression tests for project-specific Taylor-VPU reject hybrids."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace, TargetBatch
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.selective_taylor_pu_loss import (
    DeepGamblerTaylorVariationalPULoss,
    SelectiveTaylorVariationalPULoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.taylor_pu_components import (
    taylor_variational_risk,
)


def _batch() -> SpectrumBatch:
    """Return one P, one U, and one explicit N_sim entry."""
    targets = torch.tensor([[1.0], [0.0], [0.0]])
    return SpectrumBatch(
        sample_ids=torch.arange(3),
        spectra=torch.ones(3, 2),
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


def test_selective_hybrid_keeps_taylor_core_on_p_and_u_only():
    """Without reject/N_sim terms, SelectiveNet channel zero is exact Taylor-VPU."""
    batch = _batch()
    output = torch.tensor(
        [[[0.3, 0.0, 0.0]], [[-0.5, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    loss = SelectiveTaylorVariationalPULoss(
        "ion", "molecule", order=2, reject_weight=0
    )({"head_ion": output}, batch)
    expected = taylor_variational_risk(
        output[..., 0],
        torch.tensor([[True], [False], [False]]),
        torch.tensor([[False], [True], [False]]),
        2,
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert output.grad[2, 0, 0] == 0


def test_deep_gambler_hybrid_uses_conditional_non_reject_probability():
    """Changing only reserve mass cannot alter the conditional Taylor-VPU score."""
    batch = _batch()
    first = torch.tensor(
        [[[0.2, 0.7, -3.0]], [[-0.3, -0.2, -2.0]], [[1.0, -1.0, 2.0]]]
    )
    second = first.clone()
    second[..., 2] += 20.0
    criterion = DeepGamblerTaylorVariationalPULoss(
        "ion", "molecule", gambling_weight=0
    )
    first_loss = criterion({"head_ion": first}, batch)
    second_loss = criterion({"head_ion": second}, batch)
    torch.testing.assert_close(first_loss, second_loss)


def test_reject_hybrids_backpropagate_through_all_active_channels():
    """Reliable P/N_sim terms train reject channels while P/U trains PU score."""
    batch = _batch()
    selective_output = torch.zeros(3, 1, 3, requires_grad=True)
    selective_loss = SelectiveTaylorVariationalPULoss("ion", "molecule")(
        {"head_ion": selective_output}, batch
    )
    selective_loss.backward()
    assert bool((selective_output.grad[0, 0].abs() > 0).all())

    gambler_output = torch.zeros(3, 1, 3, requires_grad=True)
    gambler_loss = DeepGamblerTaylorVariationalPULoss("ion", "molecule")(
        {"head_ion": gambler_output}, batch
    )
    gambler_loss.backward()
    assert bool((gambler_output.grad[0, 0].abs() > 0).all())
