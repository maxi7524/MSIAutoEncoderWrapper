"""Tests for global modified-spy JERM alternation."""

from __future__ import annotations

import torch
import torch.nn as nn

from msi_autoencoder_wrapper.data import SpectrumBatch, SpectrumSpace, TargetBatch
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
from msi_autoencoder_wrapper.data.jerm_spy import _nearest_spy_source_ids
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.jerm_loss import JERMLoss


class _JERMModel(nn.Module):
    """Minimal stateful model marker for the criterion lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))


def _batch() -> SpectrumBatch:
    """Return one labelled positive, two U entries, and one N_sim entry."""
    targets = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    return SpectrumBatch(
        sample_ids=torch.arange(4),
        spectra=torch.tensor([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [8.0, 8.0]]),
        space=SpectrumSpace(torch.arange(2, dtype=torch.float32)),
        targets=TargetBatch(
            values={"molecule": targets},
            masks={
                "molecule": torch.ones_like(targets, dtype=torch.bool),
                simulated_negative_mask_key("molecule"): torch.tensor(
                    [[False], [False], [False], [True]]
                ),
            },
            schemas={},
        ),
    )


def test_jerm_builds_global_modified_spy_set_then_trains_propensity():
    """The closest U point is a spy and the next epoch uses P_hat for H."""
    model = _JERMModel()
    criterion = JERMLoss("ion", "molecule", simulated_negative_weight=0.5)
    criterion.on_phase_start(model, dataset=None, transient_cache={})
    criterion._static_spy_source_ids = (torch.tensor([1]),)
    batch = _batch()
    output = torch.tensor(
        [[[2.0, 0.0]], [[1.0, 0.0]], [[-2.0, 0.0]], [[0.0, 0.0]]],
        requires_grad=True,
    )
    posterior_loss = criterion({"head_ion": output}, batch)
    posterior_loss.backward()
    assert output.grad[3, 0, 0] > 0
    criterion.on_epoch_end(model)
    assert criterion._alternating_stage == "propensity"
    assert criterion._pseudo_positive.shape == (4, 1)
    assert bool(criterion._pseudo_positive[1, 0])

    propensity_output = torch.zeros(4, 1, 2, requires_grad=True)
    propensity_loss = criterion({"head_ion": propensity_output}, batch)
    propensity_loss.backward()
    assert propensity_output.grad[..., 0].abs().sum() == 0
    assert propensity_output.grad[..., 1].abs().sum() > 0


def test_jerm_conditional_probability_matches_equation_sixteen():
    """Conditional positive probability is finite and bounded for extreme logits."""
    value = JERMLoss.latent_positive_probability(
        torch.tensor([[-100.0, 100.0]]), torch.tensor([[100.0, -100.0]])
    )
    assert bool(torch.isfinite(value).all())
    assert bool(((value >= 0) & (value <= 1)).all())


def test_jerm_static_precompute_matches_the_original_nearest_spy_relation():
    """The cache stores the same nearest-U spy IDs as the dense relation."""
    spy_ids = _nearest_spy_source_ids(
        torch.tensor([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [8.0, 8.0]]),
        torch.arange(4),
        torch.tensor([[True], [False], [False], [False]]),
        torch.tensor([[False], [True], [True], [False]]),
        positive_chunk_size=1,
        unlabelled_chunk_size=1,
    )

    assert len(spy_ids) == 1
    assert torch.equal(spy_ids[0], torch.tensor([1]))


def test_jerm_uses_compact_static_posterior_storage() -> None:
    """Production-style static storage avoids one Python record per source row."""
    criterion = JERMLoss("ion", "molecule")
    criterion._static_source_ids = torch.arange(4)
    criterion._static_labelled_positive = torch.tensor([[True], [False], [False], [False]])
    criterion._static_unlabelled = torch.tensor([[False], [True], [True], [False]])
    criterion._static_spy_source_ids = (torch.tensor([1]),)
    criterion._source_id_positions = {index: index for index in range(4)}
    criterion._posterior_conditional = torch.full((4, 1), torch.nan)
    criterion._posterior_seen = torch.zeros(4, dtype=torch.bool)

    batch = _batch()
    criterion._record_posterior_batch(
        batch,
        torch.tensor([[True], [False], [False], [False]]),
        torch.tensor([[False], [True], [True], [False]]),
        torch.tensor([[0.9], [0.8], [0.1], [0.0]]),
    )
    criterion._build_modified_spy_set()

    assert not criterion._epoch_records
    assert criterion._pseudo_ids.tolist() == [0, 1, 2, 3]
    assert criterion._pseudo_positive[:, 0].tolist() == [True, True, False, False]
    assert not bool(criterion._posterior_seen.any())
