"""Tests for evidential Taylor-PU supervision and uncertainty semantics."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace, TargetBatch
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.evidential_pu_loss import (
    EvidentialTaylorVariationalPULoss,
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


def test_evidential_head_uses_dirichlet_mean_for_taylor_pu_risk():
    """U affects the Taylor core but not direct supervised evidential labels."""
    batch = _batch()
    raw = torch.zeros(3, 1, 2, requires_grad=True)
    criterion = EvidentialTaylorVariationalPULoss(
        "ion", "molecule", evidential_weight=0
    )
    criterion({"head_ion": raw}, batch).backward()
    assert raw.grad[0].abs().sum() > 0
    assert raw.grad[1].abs().sum() > 0
    assert raw.grad[2].abs().sum() == 0


def test_evidential_term_trains_declared_simulated_negative_only():
    """N_sim receives direct evidential supervision when that extension is active."""
    batch = _batch()
    raw = torch.zeros(3, 1, 2, requires_grad=True)
    criterion = EvidentialTaylorVariationalPULoss("ion", "molecule")
    loss = criterion({"head_ion": raw}, batch)
    loss.backward()
    assert raw.grad[2].abs().sum() > 0
    alpha = criterion.concentrations(raw.detach())
    uncertainty = criterion.uncertainty(alpha)
    assert uncertainty.shape == (3, 1)
    assert bool(((uncertainty > 0) & (uncertainty <= 1)).all())
