"""Complete precompute workflow for general predictive-head analyses."""

from __future__ import annotations

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import predictive_campaign
from msi_autoencoder_wrapper.analysis.autoencoder.heads.precompute_plugin import shared_inference_plugin
from msi_autoencoder_wrapper.analysis.autoencoder.latent.precompute_plugin import latent_geometry_plugin
from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.precompute_plugin import (
    reconstruction_global_plugin,
    reconstruction_local_plugin,
)
from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.training_precompute_plugin import (
    campaign_training_dynamics_plugin,
)
from ...core.contracts import PrecomputeStrategy


def build_strategy() -> PrecomputeStrategy:
    """Build the stable workflow consumed by predictive-head notebook folders."""
    return PrecomputeStrategy(
        name="autoencoder.heads_general_analysis",
        model_type="autoencoder",
        stages=(
            campaign_training_dynamics_plugin(),
            shared_inference_plugin(),
            reconstruction_local_plugin(),
            reconstruction_global_plugin(),
            latent_geometry_plugin(),
        ),
        load_settings=predictive_campaign.load_settings,
        inventory=predictive_campaign.inventory,
    )
