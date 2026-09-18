"""Precompute plugins that expose reconstruction analyses to workflow strategies."""

from __future__ import annotations

from ...precompute.core.contracts import AnalysisPlugin, ArtifactSpec


def reconstruction_local_plugin() -> AnalysisPlugin:
    """Return the detailed local reconstruction plugin."""
    from ..experiments import predictive_precompute

    def run(context) -> None:
        predictive_precompute.precompute_analysis(
            context.settings, "reconstruction_local", allow_cpu=context.allow_cpu
        )

    return AnalysisPlugin(
        name="autoencoder.reconstruction.local",
        requires=("model_catalog", "shared_inference"),
        provides=(ArtifactSpec("reconstruction_local", "reconstruction_local", ("metadata.json",)),),
        run=run,
    )


def reconstruction_global_plugin() -> AnalysisPlugin:
    """Return the campaign-wide reconstruction plugin."""
    from ..experiments import predictive_precompute

    def run(context) -> None:
        predictive_precompute.precompute_analysis(
            context.settings, "reconstruction_global", allow_cpu=context.allow_cpu
        )

    return AnalysisPlugin(
        name="autoencoder.reconstruction.global",
        requires=("model_catalog", "shared_inference"),
        provides=(ArtifactSpec("reconstruction_global", "reconstruction_global", ("metadata.json",)),),
        run=run,
    )
