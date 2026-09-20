"""Configuration contract for the artifact-backed kidney pretraining schedule."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config


def test_kidney_pretraining_campaign_declares_reusable_axis_and_mixture_artifact():
    """One serial axis/permutation schedule is explicit and grid-expandable."""
    repository = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        repository
        / "assets/experiments/autoencoder_architecture/experiment_runs_configs"
        / "segmentation_model/20_09_26_metaspace_base_pretrain"
        / "pretraining_experiment.yaml"
    )
    plan = build_plan(config)
    artifact_tasks = [
        task
        for task in plan.tasks
        if any(
            phase.get("pretraining", {}).get("kind") == "precomputed_synthetic"
            for phase in task.parameters["training"]["phases"]
        )
    ]
    assert len(artifact_tasks) == 10
    for task in artifact_tasks:
        phases = task.parameters["training"]["phases"]
        axis, mixture, real = phases
        assert [phase["phase_name"] for phase in phases] == [
            "synthetic_axis_pretraining",
            "synthetic_permutation_pretraining",
            "real_adaptation",
        ]
        assert axis["epochs"] == 10
        assert mixture["epochs"] == 15
        axis_config = axis["pretraining"]
        mixture_config = mixture["pretraining"]
        assert axis_config["population"] == "axis"
        assert mixture_config["population"] == "permutation"
        assert axis_config["artifact"] == mixture_config["artifact"]
        populations = axis_config["artifact"]["populations"]
        assert populations["axis"] == {
            "strategy": "axis_coverage",
            "repetitions_per_bin": 5,
        }
        assert populations["permutation"] == {
            "strategy": "uniform_mixture",
            "repetitions_per_bin": 30,
            "min_fragments": 15,
            "max_fragments": 30,
            "detection_probability": 0.5,
        }
