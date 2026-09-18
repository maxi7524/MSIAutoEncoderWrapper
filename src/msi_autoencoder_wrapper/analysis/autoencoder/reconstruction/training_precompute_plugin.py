"""Precompute plugin exposing campaign-training diagnostics to strategies."""

from __future__ import annotations

from ...precompute.core.contracts import AnalysisPlugin, ArtifactSpec


def campaign_training_dynamics_plugin() -> AnalysisPlugin:
    """Return the per-source training-dynamics plugin."""
    from ..experiments import predictive_precompute

    def run(context) -> None:
        predictive_precompute.precompute_analysis(
            context.settings, "campaign_training_dynamics", allow_cpu=context.allow_cpu
        )

    return AnalysisPlugin(
        name="autoencoder.reconstruction.campaign_training_dynamics",
        requires=("model_catalog",),
        provides=(ArtifactSpec("campaign_training_dynamics", "campaign_training_dynamics", ("metadata.json",)),),
        run=run,
    )
