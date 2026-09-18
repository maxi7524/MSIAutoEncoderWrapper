"""Complete workflow adapter for the established contractive-regularization reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import contractive_precompute
from ...core.contracts import AnalysisPlugin, ArtifactSpec, PrecomputeStrategy


_ROUTINE_ORDER = (
    "campaign_training_dynamics",
    "prediction_sweep",
    "latent_geometry_sweep",
    "hinge_refinement",
    "perturbation_anatomy",
    "perturbation_response",
    "spectrum_reconstruction",
)


def _inventory(settings: dict[str, Any]) -> pd.DataFrame:
    """Adapt legacy contractive campaign records to the common catalog schema."""
    names = list(settings.get("campaigns", {}))
    frame = contractive_precompute.campaign_grid(settings, names).copy()
    if frame.empty:
        return pd.DataFrame(columns=["model_id", "source", "grid_id", "task_id", "repetition", "label", "ready"])
    frame["model_id"] = frame["model_name"]
    frame["source"] = frame["campaign"]
    frame["grid_id"] = frame["cell_label"]
    frame["task_id"] = frame["model_name"]
    frame["label"] = frame["cell_label"]
    frame["ready"] = True
    return frame


def _plugin(name: str) -> AnalysisPlugin:
    """Wrap one legacy contractive routine in the common strategy contract."""
    def run(context) -> None:
        contractive_precompute.precompute(context.settings, name, allow_cpu=context.allow_cpu)

    return AnalysisPlugin(
        name=f"autoencoder.contractive.{name}",
        requires=("model_catalog",),
        provides=(ArtifactSpec(name, name, ("metadata.json",)),),
        run=run,
    )


def build_strategy() -> PrecomputeStrategy:
    """Build the complete contractive-regularization notebook workflow."""
    return PrecomputeStrategy(
        name="autoencoder.contractive_regularization_analysis",
        model_type="autoencoder",
        stages=tuple(_plugin(name) for name in _ROUTINE_ORDER),
        load_settings=contractive_precompute.read_settings,
        inventory=_inventory,
    )
