"""Analytical checks for simplex-to-sphere sensitivity diagnostics."""

import itertools

import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.latent.sensitivity import (
    SensitivityDiagnostic,
    canonical_direction,
    exact_output_jacobian,
    metric_operators,
    operator_statistics,
    paired_direction_angles,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.regularization.contractive_loss import MSIContractiveLoss
from msi_autoencoder_wrapper.utils.exceptions import IncompatibleInterfaceError


@pytest.fixture
def diagnostic_case():
    # Small nonuniform spectra include the fixed-support boundary case.
    spectra = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, 0.5, 0.25, 0.25]], dtype=torch.float64)
    jacobian = torch.tensor([[[1., 2., -1., 3.], [0., 1., 4., -2.]]], dtype=torch.float64).repeat(2, 1, 1)
    kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    return spectra, jacobian, kernel


def test_operators_match_dense_simplex_metrics(diagnostic_case):
    x, jacobian, kernel = diagnostic_case
    operators = metric_operators(jacobian, x, kernel, 0.5)
    projector = torch.eye(4, dtype=x.dtype) - torch.ones(4, 4, dtype=x.dtype) / 4
    convolution = torch.tensor([[.5, .25, 0, 0], [.25, .5, .25, 0], [0, .25, .5, .25], [0, 0, .25, .5]], dtype=x.dtype)
    incidence = torch.tensor([[-1., 0, 0], [1., -1, 0], [0, 1., -1], [0, 0, 1.]], dtype=x.dtype)
    for index in range(2):
        covariances = {
            "euclidean_ambient": torch.eye(4, dtype=x.dtype),
            "euclidean_simplex": projector,
            "fisher_diagonal": torch.diag(x[index]),
            "fisher_simplex": torch.diag(x[index]) - torch.outer(x[index], x[index]),
            "cramer_simplex": incidence @ incidence.T / .5,
            "gaussian_simplex": projector @ convolution @ projector @ convolution @ projector,
        }
        for name, covariance in covariances.items():
            gram, trace, spectral = operator_statistics(operators[name])
            expected = jacobian[index] @ covariance @ jacobian[index].T
            torch.testing.assert_close(gram[index], expected, rtol=1e-12, atol=1e-12)
            assert spectral[index] <= trace[index] + 1e-12
            assert spectral[index] >= trace[index] / 2 - 1e-12
    # A constant input covector vanishes on every mass-preserving tangent.
    constant = metric_operators(torch.ones_like(jacobian), x, kernel, 1.0)
    for name in ("euclidean_simplex", "fisher_simplex", "cramer_simplex", "gaussian_simplex"):
        torch.testing.assert_close(constant[name], torch.zeros_like(constant[name]), atol=1e-14, rtol=0)


def test_output_projection_and_local_angles():
    x = torch.tensor([[.2, .3, .5]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[2., 0, 1.], [-1., 3., 0], [0, 1., -2.], [1., -1., 2.]], dtype=x.dtype)
    gamma = torch.tensor([.5, 2., -1., 3.], dtype=x.dtype)
    beta = torch.arange(4, dtype=x.dtype)

    def mapping(value):
        raw = value @ weight.T
        z = torch.nn.functional.layer_norm(raw, (4,), gamma, beta, eps=.1)
        return canonical_direction(z, gamma, beta)

    u, q = mapping(x)
    ju = exact_output_jacobian(u, x)
    jq = exact_output_jacobian(q, x)
    projector = torch.eye(4, dtype=x.dtype) - torch.ones(4, 4, dtype=x.dtype) / 4 - q[0, :, None] * q[0, None, :]
    expected = (projector @ ju[0]) / torch.linalg.vector_norm(u[0])
    torch.testing.assert_close(jq[0], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(jq.sum(dim=1), torch.zeros_like(x), atol=1e-12, rtol=0)
    torch.testing.assert_close(q.unsqueeze(1) @ jq, torch.zeros(1, 1, 3, dtype=x.dtype), atol=1e-12, rtol=0)
    delta = torch.tensor([[.1, -.2, .1]], dtype=x.dtype)
    predicted = torch.linalg.vector_norm((jq @ delta.unsqueeze(-1)).squeeze(-1), dim=1)
    actual = paired_direction_angles(q, mapping(x + 1e-6 * delta)[1]) / 1e-6
    # First-order finite differences have O(step) truncation error.
    torch.testing.assert_close(actual, predicted, atol=1e-7, rtol=1e-5)
    assert paired_direction_angles(q, q).item() == 0
    torch.testing.assert_close(paired_direction_angles(q, -q), torch.tensor([torch.pi], dtype=x.dtype))


def test_hutchinson_all_rademacher_directions_recovers_trace(diagnostic_case):
    x, jacobian, kernel = diagnostic_case
    probes = torch.tensor(list(itertools.product([-1., 1.], repeat=2)), dtype=x.dtype)
    for operator in metric_operators(jacobian, x, kernel, 1.).values():
        gram, trace, _ = operator_statistics(operator)
        estimates = torch.einsum("pd,bdk,pk->bp", probes, gram, probes).mean(dim=1)
        torch.testing.assert_close(estimates, trace, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("input_geometry", ["euclidean", "fisher_rao"])
def test_adapter_matches_public_training_estimator_and_combined_penalty(diagnostic_case, input_geometry):
    x, jacobian, _ = diagnostic_case
    x = x[:1].clone().requires_grad_(True)
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(4, 2, bias=False, dtype=x.dtype)
    with torch.no_grad():
        model.encoder.weight.copy_(jacobian[0])
    adapter = SensitivityDiagnostic(model.encoder, input_geometry)
    criterion = MSIContractiveLoss(penalty_metric="spectral_plus_hinged", input_geometry=input_geometry, hinge_threshold=.3, hinge_alpha=2.)
    criterion.on_phase_start(model, None, {})
    torch.manual_seed(42)
    spectral = adapter.estimate_spectral_squared(model, x)
    torch.manual_seed(42)
    penalty = criterion({"latent_space": model.encoder(x)}, (None, x))
    expected = spectral + 2 * torch.relu(spectral.sqrt() - .3).square()
    torch.testing.assert_close(penalty, expected.mean(), atol=1e-12, rtol=1e-12)
    operator = adapter.transform_jacobian(jacobian[:1], x)
    exact = operator_statistics(operator)[2]
    assert 0 <= spectral.item() <= exact.item() * (1 + 1e-12)


def test_invalid_geometry_inputs_are_rejected(diagnostic_case):
    x, jacobian, kernel = diagnostic_case
    with pytest.raises(ValueError, match="spacing"):
        metric_operators(jacobian, x, kernel, 0)
    with pytest.raises(ValueError, match="kernel"):
        metric_operators(jacobian, x, kernel * 2, 1)
    with pytest.raises(ValueError, match="gamma"):
        canonical_direction(x, torch.zeros(4, dtype=x.dtype), torch.zeros(4, dtype=x.dtype))
    with pytest.raises(ValueError, match="zero radius"):
        canonical_direction(torch.ones_like(x), torch.ones(4, dtype=x.dtype), torch.zeros(4, dtype=x.dtype))
    with pytest.raises(ValueError, match="nonzero"):
        paired_direction_angles(torch.zeros_like(x), x)
    with pytest.raises(IncompatibleInterfaceError, match="TIC"):
        metric_operators(jacobian, x * 2, kernel, 1)
    with pytest.raises(IncompatibleInterfaceError, match="non-negative"):
        metric_operators(jacobian, x - .3, kernel, 1)


def test_fisher_vertex_has_no_fixed_support_tangent(diagnostic_case):
    _, jacobian, kernel = diagnostic_case
    vertex = torch.tensor([[1., 0, 0, 0]], dtype=jacobian.dtype).repeat(2, 1)
    operator = metric_operators(jacobian, vertex, kernel, 1)["fisher_simplex"]
    gram, trace, spectral = operator_statistics(operator)
    assert torch.count_nonzero(gram) == 0
    assert torch.count_nonzero(trace) == 0
    assert torch.count_nonzero(spectral) == 0


def test_input_fisher_forward_and_adjoint_are_consistent(diagnostic_case):
    x, jacobian, _ = diagnostic_case
    criterion = MSIContractiveLoss(penalty_metric="spectral", input_geometry="fisher_rao")
    direction, covector = jacobian[:, 0], jacobian[:, 1]
    forward = criterion._geometry_forward_direction(direction, x)
    adjoint = criterion._geometry_adjoint_direction(covector, x)
    torch.testing.assert_close((forward * covector).sum(dim=1), (direction * adjoint).sum(dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(forward.sum(dim=1), torch.zeros(2, dtype=x.dtype), rtol=0, atol=1e-12)
