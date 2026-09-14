"""Tests for the Torch projected-gradient baseline and its metrics."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.deconvolution import (
    DeconvolutionResult,
    NonnegativeProjectedGradientSolver,
    deconvolution_metrics,
)


def test_projected_gradient_recovers_an_orthogonal_sparse_mixture() -> None:
    """The baseline solves exact non-negative mixtures on an orthogonal dictionary."""
    matrix = torch.eye(4)  # (M=4, C=4)
    true_abundance = torch.tensor([[0.0, 1.5, 0.0, 0.25], [2.0, 0.0, 1.0, 0.0]])  # (B=2, C=4)
    spectra = true_abundance @ matrix.transpose(0, 1)  # (B=2, M=4)

    result = NonnegativeProjectedGradientSolver(iterations=2)(spectra, matrix)

    torch.testing.assert_close(result.abundances, true_abundance, rtol=0, atol=1e-6)
    torch.testing.assert_close(result.reconstruction, spectra, rtol=0, atol=1e-6)
    assert torch.all(result.objective < 1e-10)


def test_l1_penalty_removes_small_abundances() -> None:
    """The sparse proximal term has a measurable non-negative shrinkage effect."""
    matrix = torch.eye(2)  # (M=2, C=2)
    spectra = torch.tensor([[1.0, 0.05]])  # (B=1, M=2)

    result = NonnegativeProjectedGradientSolver(iterations=2, l1_weight=0.1)(spectra, matrix)

    assert result.abundances[0, 0] > 0
    assert result.abundances[0, 1] == 0


def test_metrics_report_exact_recovery_and_support() -> None:
    """Metric tensors quantify exact abundance, reconstruction, and support recovery."""
    abundances = torch.tensor([[1.0, 0.0], [0.0, 2.0]])  # (B=2, C=2)
    result = DeconvolutionResult(
        abundances=abundances,
        reconstruction=abundances,
        residual=torch.zeros_like(abundances),
        objective=torch.zeros(2),
    )

    metrics = deconvolution_metrics(result, true_abundances=abundances)

    torch.testing.assert_close(metrics["abundance_mae"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["abundance_rmse"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["reconstruction_mse"], torch.tensor(0.0))
    assert metrics["support_precision"] == 1
    assert metrics["support_recall"] == 1
    assert metrics["support_f1"] == 1
