"""Unit tests for campaign numerical benchmark reporting helpers."""

from __future__ import annotations

import pytest

from msi_autoencoder_wrapper.analysis.benchmarking.campaign_benchmark import (
    confidence_interval,
    contractive_complexity,
)


def test_confidence_interval_uses_the_single_observation_as_its_interval() -> None:
    """A single timing has no sampling uncertainty estimate."""
    mean, deviation, lower, upper = confidence_interval([2.5])
    assert (mean, deviation, lower, upper) == (2.5, 0.0, 2.5, 2.5)


def test_confidence_interval_rejects_invalid_samples() -> None:
    """Timing summaries require finite non-negative observations."""
    with pytest.raises(ValueError, match="finite"):
        confidence_interval([1.0, float("nan")])


def test_contractive_complexity_distinguishes_exact_and_spectral_work() -> None:
    """Exact Frobenius depends on latent dimension; spectral ignores probes."""
    exact = contractive_complexity(
        {"penalty_metric": "frobenius", "calculation_method": "exact_autograd_jacobian"},
        latent_dimension=10,
    )
    spectral = contractive_complexity(
        {"penalty_metric": "spectral", "num_probes": 32},
        latent_dimension=10,
    )
    assert "10 reverse-mode VJP" in exact
    assert "3·C_VJP" in spectral
    assert "4·C_JVP" in spectral


def test_contractive_complexity_supports_campaigns_without_contractive_loss() -> None:
    """Generic campaign profiling remains available without ContractiveLoss."""
    complexity = contractive_complexity(None, latent_dimension=1)
    assert complexity == "O(C_train_step(B, input_shape, model))"
