"""P/U/N_sim mask sampling preserves declared supervision categories."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.data.supervision_sampling import SupervisionMaskBatchSampler


def test_supervision_mask_sampler_is_deterministic_and_respects_requested_pools():
    """Every emitted batch contains configured P/N_sim/U memberships."""
    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    available = torch.ones_like(targets, dtype=torch.bool)
    simulated_negative = torch.tensor(
        [[False], [False], [True], [True], [False], [False]]
    )
    first = SupervisionMaskBatchSampler(
        targets,
        available,
        simulated_negative,
        batch_size=6,
        proportions={"positive": 1, "simulated_negative": 1, "unlabelled": 1},
        steps_per_epoch=2,
        seed=9,
    )
    second = SupervisionMaskBatchSampler(
        targets,
        available,
        simulated_negative,
        batch_size=6,
        proportions={"positive": 1, "simulated_negative": 1, "unlabelled": 1},
        steps_per_epoch=2,
        seed=9,
    )
    batches = list(first)
    assert batches == list(second)
    for batch in batches:
        selected_targets = targets[batch, 0]
        selected_negative = simulated_negative[batch, 0]
        assert int((selected_targets > 0.5).sum()) == 2
        assert int(selected_negative.sum()) == 2
        assert int(((selected_targets == 0) & ~selected_negative).sum()) == 2


def test_supervision_mask_sampler_refreshes_generator_owned_nsim_masks():
    """Epoch changes refresh pools when a synthetic generator changes labels."""
    first = (
        torch.tensor([[1.0], [0.0]]),
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([[False], [True]]),
    )
    second = (
        torch.tensor([[0.0], [1.0]]),
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([[True], [False]]),
    )
    current = [first]
    sampler = SupervisionMaskBatchSampler(
        *first,
        batch_size=2,
        proportions={"positive": 1, "simulated_negative": 1},
        steps_per_epoch=1,
        supervision_provider=lambda: current[0],
    )
    current[0] = second
    sampler.set_epoch(1)
    assert sampler.pools["positive"] == [1]
    assert sampler.pools["simulated_negative"] == [0]
