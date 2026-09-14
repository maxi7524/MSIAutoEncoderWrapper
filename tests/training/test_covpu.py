"""Tests for CoVPU's exact flow, collaborator, and equation-(15) risk."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.training.covpu import (
    CoVPUPlanarFlow,
    covpu_consensus,
    covpu_rebalanced_variational_risk,
)


def test_planar_flow_identity_initialization_is_finite_and_shape_preserving():
    """CoVPU flow maps each two-dimensional variable with a log Jacobian."""
    flow = CoVPUPlanarFlow(3)
    value, determinant = flow(torch.randn(4, 2))
    assert value.shape == (4, 2)
    assert determinant.shape == (4,)
    assert bool(torch.isfinite(determinant).all())


def test_covpu_collaborator_selects_component_with_top_j_consensus():
    """The positive component is selected by high-confidence classifier votes."""
    probability = torch.tensor([[0.9], [0.8], [0.2], [0.1]])
    component = torch.tensor([[1], [1], [0], [0]])
    consensus = covpu_consensus(probability, component, torch.ones_like(component, dtype=torch.bool), 2)
    assert consensus.component_positive[:2, 0].all()
    assert consensus.pseudo_positive[:2, 0].all()
    assert not consensus.pseudo_positive[2:, 0].any()


def test_covpu_equation_fifteen_has_finite_gradients():
    """Rebalanced variational risk differentiates all PL, PU, and N groups."""
    logits = torch.tensor([[1.0], [0.5], [-1.0]], requires_grad=True)
    loss = covpu_rebalanced_variational_risk(
        logits,
        torch.tensor([[True], [False], [False]]),
        torch.tensor([[False], [True], [False]]),
        torch.tensor([[False], [False], [True]]),
        rebalanced_prior=0.5,
        labelled_positive_prior=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert bool(torch.isfinite(logits.grad).all())
