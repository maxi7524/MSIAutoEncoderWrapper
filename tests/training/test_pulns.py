"""Numerical tests for PULNS selector episodes."""

from __future__ import annotations

import torch
import torch.nn as nn

from msi_autoencoder_wrapper.training.pulns import (
    PULNSSelectorController,
    pulns_intermediate_reward,
    pulns_reinforce_loss,
)


def test_pulns_intermediate_reward_agrees_with_selected_action_semantics():
    """Selecting a classifier-negative U sample receives positive reward."""
    probabilities = torch.tensor([0.1, 0.9])
    selected = torch.tensor([True, False])
    rewards = pulns_intermediate_reward(probabilities, selected)
    assert bool((rewards > 0).all())


def test_pulns_episode_builds_sequential_state_and_backpropagates_reinforce():
    """The selector has REINFORCE gradients while the classifier loss uses selected N."""
    torch.manual_seed(3)
    controller = PULNSSelectorController()
    representations = torch.tensor([[0.0], [1.0], [3.0]])
    logits = torch.tensor([[2.0], [-2.0], [1.0]], requires_grad=True)
    positives = torch.tensor([[True], [False], [False]])
    unlabelled = ~positives
    selector = nn.Linear(3, 1)
    with torch.no_grad():
        selector.weight.zero_()
        selector.bias.fill_(4.0)
    episode = controller.sample_episode(
        representations,
        logits,
        positives,
        unlabelled,
        lambda state: selector(state),
        deterministic=True,
    )
    assert episode.selected_negative[1:, 0].all()
    objective = pulns_reinforce_loss(
        episode, torch.tensor([0.2]), terminal_weight=1.0, discount=0.8
    )
    objective.backward()
    assert selector.weight.grad is not None
    classifier_loss = controller.classifier_loss(
        logits, positives, episode.selected_negative
    )
    classifier_loss.backward()
    assert logits.grad is not None
