"""Complete precompute workflow for the synthetic-pretraining campaign notebooks."""

from __future__ import annotations

from functools import partial

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign as campaign
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_inference as inference
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_populations as populations
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_precompute as family
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_reports  # noqa: F401
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_stage_reports  # noqa: F401
from msi_autoencoder_wrapper.analysis.autoencoder.experiments import pretraining_campaign_synthetic as synthetic
from ...core.contracts import AnalysisPlugin, ArtifactSpec, PrecomputeStrategy

#: Logical artifact provided by each shared cache level.
STAGE_ARTIFACTS = {"campaign_plan": "pretraining_plan", "populations": "pretraining_populations",
                   "synthetic_reference": "pretraining_synthetic_reference", "inference": "pretraining_inference"}


def _shared(name: str, requires: str, run) -> AnalysisPlugin:
    """One shared cache level; always enabled unless pruned by ``--analysis``."""
    return AnalysisPlugin(name=f"autoencoder.pretraining.{name}", requires=(requires,),
                          provides=(ArtifactSpec(STAGE_ARTIFACTS[name]),), run=run, enabled_when_configured=False)


def build_strategy() -> PrecomputeStrategy:
    """Build the ordered workflow: plan, populations, synthetic reference, inference, analyses."""
    stages = [
        _shared("campaign_plan", "model_catalog", family.run_campaign_plan),
        _shared("populations", STAGE_ARTIFACTS["campaign_plan"], populations.run_populations),
        _shared("synthetic_reference", STAGE_ARTIFACTS["populations"], synthetic.run_synthetic_reference),
        _shared("inference", STAGE_ARTIFACTS["synthetic_reference"], inference.run_inference),
    ]
    for name, analysis in family.ANALYSES.items():
        requires = ("model_catalog", *(STAGE_ARTIFACTS[stage] for stage in analysis.requires))
        stages.append(AnalysisPlugin(name=f"autoencoder.pretraining.{name}", requires=requires,
                                     provides=(ArtifactSpec(name, name),),
                                     run=partial(family.run_analysis, name=name)))
    return PrecomputeStrategy(name="autoencoder.pretraining_campaign_analysis", model_type="autoencoder",
                              stages=tuple(stages), load_settings=campaign.load_settings,
                              inventory=campaign.inventory)
