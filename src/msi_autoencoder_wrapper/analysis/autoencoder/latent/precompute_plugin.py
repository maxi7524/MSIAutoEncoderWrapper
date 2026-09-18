"""Precompute plugins exposing latent-space analyses to workflow strategies."""

from __future__ import annotations

from ...precompute.core.contracts import AnalysisPlugin, ArtifactSpec


def latent_geometry_plugin() -> AnalysisPlugin:
    """Return the latent-geometry plugin for predictive autoencoder workflows."""
    from ..experiments import predictive_precompute

    def run(context) -> None:
        predictive_precompute.precompute_analysis(
            context.settings, "latent_geometry", allow_cpu=context.allow_cpu
        )

    return AnalysisPlugin(
        name="autoencoder.latent.geometry",
        requires=("model_catalog", "shared_inference"),
        provides=(ArtifactSpec("latent_geometry", "latent_geometry", ("metadata.json",)),),
        run=run,
    )
