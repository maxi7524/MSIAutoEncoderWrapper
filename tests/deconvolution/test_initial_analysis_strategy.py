"""Tests for the configured initial deconvolution analysis workflow."""

from __future__ import annotations

from msi_autoencoder_wrapper.analysis.precompute.strategies.deconvolution.initial_analysis import (
    build_strategy,
)


def test_initial_strategy_declares_the_complete_feasibility_workflow() -> None:
    """Coverage, numerical baseline, and identifiability run in a fixed order."""
    strategy = build_strategy()

    assert strategy.name == "deconvolution.initial_analysis"
    assert strategy.model_type == "deconvolution"
    assert [stage.name for stage in strategy.stages] == [
        "deconvolution.candidate_coverage",
        "deconvolution.projected_gradient_baseline",
        "deconvolution.identifiability",
    ]
    assert strategy.stages[1].requires == ("model_catalog", "candidate_catalogue")
