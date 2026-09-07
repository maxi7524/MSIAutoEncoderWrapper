"""Deterministic numerical tests for signal evidence and prior-free objectives."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from msi_autoencoder_wrapper.data.annotation_evidence import IonCatalogue, SignalEvidencePolicy
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.evidence_losses import (
    SignalMaskedBCELoss, ThreeStateCrossEntropyLoss,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.variational_pu_loss import VariationalPULoss, normalize_scores
from msi_autoencoder_wrapper.training.criterions.autoencoder.head.symmetric_pu_ranking_loss import SymmetricPURankingLoss


@pytest.fixture
def evidence_case():
    catalogue = IonCatalogue(("C2H4|+H", "C3H6|+H"), ((0,), (2,)), 4)
    spectra = torch.tensor([[0., 0., 2., 0.], [0., 0., 0., 1.], [3., 0., 0., 0.]])
    targets = torch.tensor([[1., 0.], [0., 0.], [0., 1.]])
    mask = torch.tensor([[True, True], [True, False], [True, True]])
    batch = (torch.arange(3), spectra, {"molecule": targets}, {"molecule": mask})
    return catalogue, batch


def test_evidence_preserves_positives_and_distinguishes_uncertain_unavailable(evidence_case):
    catalogue, batch = evidence_case
    states = SignalEvidencePolicy(bin_radius=0).classify(batch[1], batch[2]["molecule"], batch[3]["molecule"], catalogue)
    assert states.tolist() == [[1, 2], [0, -1], [2, 1]]


def test_masked_bce_excludes_uncertain_and_unavailable_gradients(evidence_case):
    catalogue, batch = evidence_case
    criterion = SignalMaskedBCELoss("ion", "molecule", evidence={"bin_radius": 0})
    criterion.catalogue = catalogue
    logits = torch.zeros(3, 2, requires_grad=True)
    loss = criterion({"head_ion": logits}, batch)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(2.).log(), rtol=1e-6, atol=1e-7)
    assert logits.grad[0, 0] < 0 and logits.grad[1, 0] > 0
    assert logits.grad[0, 1] == logits.grad[1, 1] == logits.grad[2, 0] == 0


def test_masked_bce_uses_conservative_global_sqrt_positive_weight(evidence_case):
    catalogue, batch = evidence_case
    criterion = SignalMaskedBCELoss(
        "ion", "molecule", evidence={"bin_radius": 0},
        positive_weight_mode="global_sqrt_negative_to_positive",
        max_positive_weight=10,
    )
    criterion.catalogue = catalogue
    criterion.positive_weights = criterion._positive_weights_from_counts(
        torch.tensor([1, 3]), torch.tensor([99, 12]),
    )
    torch.testing.assert_close(criterion.positive_weights, torch.full((2,), torch.sqrt(torch.tensor(111 / 4))))
    logits = torch.zeros(3, 2, requires_grad=True)
    loss = criterion({"head_ion": logits}, batch)
    loss.backward()
    assert logits.grad[0, 0].abs() > logits.grad[1, 0].abs()


def test_masked_bce_falls_back_to_global_weight_for_sparse_classes():
    criterion = SignalMaskedBCELoss(
        "ion", "molecule", positive_weight_mode="per_class_sqrt_negative_to_positive",
        max_positive_weight=10, minimum_positive_count=10,
    )
    weights = criterion._positive_weights_from_counts(
        torch.tensor([2, 20]), torch.tensor([98, 80]),
    )
    global_weight = torch.sqrt(torch.tensor(178 / 22))
    torch.testing.assert_close(weights[0], global_weight)
    torch.testing.assert_close(weights[1], torch.tensor(2.0))


def test_three_state_loss_supervises_uncertainty(evidence_case):
    catalogue, batch = evidence_case
    criterion = ThreeStateCrossEntropyLoss("ion", "molecule", evidence={"bin_radius": 0})
    criterion.catalogue = catalogue
    logits = torch.zeros(3, 2, 3, requires_grad=True)
    loss = criterion({"head_ion": logits}, batch)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(3.).log(), rtol=1e-6, atol=1e-7)
    assert logits.grad[0, 1, 2] < 0
    assert not bool(logits.grad[1, 1].any())


def test_vpu_matches_analytic_marginal_risk_and_has_finite_extreme_gradients(evidence_case):
    _, batch = evidence_case
    logits = torch.tensor([[-1000., 2.], [4., 3.], [0., -2.]], requires_grad=True)
    criterion = VariationalPULoss("ion", "molecule", consistency_weight=0)
    loss = criterion({"head_ion": logits}, batch)
    expected = torch.stack([
        torch.logsumexp(F.logsigmoid(logits[:, 0]), 0) - torch.tensor(3.).log() - F.logsigmoid(logits[0, 0]),
        torch.logsumexp(F.logsigmoid(logits[[0, 2], 1]), 0) - torch.tensor(2.).log() - F.logsigmoid(logits[2, 1]),
    ]).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
    loss.backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert logits.grad[1, 1] == 0


def test_vpu_negative_extension_preserves_core_risk(evidence_case):
    catalogue, batch = evidence_case
    logits = torch.zeros(3, 2, requires_grad=True)
    base = VariationalPULoss("ion", "molecule", consistency_weight=0)
    extended = VariationalPULoss("ion", "molecule", consistency_weight=0, negative_weight=2, evidence={"bin_radius": 0})
    extended.catalogue = catalogue
    delta = extended({"head_ion": logits}, batch) - base({"head_ion": logits}, batch)
    torch.testing.assert_close(delta, 2 * torch.tensor(2.).log(), rtol=1e-6, atol=1e-7)


def test_vpu_mixup_backpropagates_to_encoder(evidence_case):
    import weakref
    _, batch = evidence_case
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
        def forward(self, x):
            return {"head_ion": self.linear(x)}
    torch.manual_seed(17)
    model = Model()
    criterion = VariationalPULoss("ion", "molecule")
    criterion._model_ref = weakref.ref(model)
    loss = criterion(model(batch[1]), batch)
    loss.backward()
    assert bool(torch.isfinite(model.linear.weight.grad).all())
    assert model.linear.weight.grad.abs().sum() > 0
    assert not list(criterion.parameters())


def test_prior_free_ranking_prefers_correct_order():
    batch = (torch.tensor([0]), torch.ones(1, 2), {"molecule": torch.tensor([[1., 0.]])}, {"molecule": torch.ones(1, 2, dtype=torch.bool)})
    criterion = SymmetricPURankingLoss("ion", "molecule")
    good = criterion({"head_ion": torch.tensor([[4., -4.]])}, batch)
    bad = criterion({"head_ion": torch.tensor([[-4., 4.]])}, batch)
    assert good < bad
    torch.testing.assert_close(good + bad, torch.tensor(1.), rtol=1e-6, atol=1e-7)


def test_vpu_normalization_uses_supplied_training_maximum():
    scores = normalize_scores(torch.zeros(1, 2), torch.tensor([.5, 1.]))
    torch.testing.assert_close(scores, torch.tensor([[1., .5]]), rtol=1e-6, atol=1e-7)
