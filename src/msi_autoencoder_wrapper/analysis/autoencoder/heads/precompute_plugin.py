"""Precompute plugin exposing shared predictive-head inference to strategies."""

from __future__ import annotations

from ...precompute.core.contracts import AnalysisPlugin, ArtifactSpec


def shared_inference_plugin() -> AnalysisPlugin:
    """Return the shared full-campaign inference stage for head analyses."""
    from ..experiments import predictive_precompute

    def run(context) -> None:
        predictive_precompute.precompute(context.catalog.records, context.settings)

    return AnalysisPlugin(
        name="autoencoder.heads.shared_inference",
        requires=("model_catalog",),
        provides=(ArtifactSpec("shared_inference", required_files=("latest.json",), root_setting="cache_directory"),),
        run=run,
        enabled_when_configured=False,
    )
