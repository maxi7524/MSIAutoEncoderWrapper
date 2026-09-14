"""Regression coverage for paired expanded predictive-head campaigns."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.runtime.configuration.loading import load_experiment_config
from msi_autoencoder_wrapper.runtime.planning.plan import build_plan


ROOT = Path("assets/experiments/autoencoder_architecture/experiment_runs_configs/07_09_26_predictive_expanded")


def test_expanded_head_campaign_uses_one_paired_pnu_sampling_contract() -> None:
    """Every method has exactly its own head and a paired sampling contract."""
    config = load_experiment_config(ROOT / "predictive_heads_expanded.yaml")
    plan = build_plan(config)

    assert len(plan.tasks) == 50
    expected_sampling = {
        "BCEPNLoss": {"positive", "simulated_negative"},
        "ObservedPNUSCrossEntropyLoss": {"positive", "simulated_negative", "unlabelled"},
        "VariationalPULoss": {"positive", "simulated_negative", "unlabelled"},
        "TaylorVariationalPULoss": {"positive", "unlabelled"},
        "SelectivePNLoss": {"positive", "simulated_negative"},
        "DeepGamblerPNLoss": {"positive", "simulated_negative"},
        "SelectiveTaylorVariationalPULoss": {"positive", "simulated_negative", "unlabelled"},
        "DeepGamblerTaylorVariationalPULoss": {"positive", "simulated_negative", "unlabelled"},
        "EvidentialTaylorVariationalPULoss": {"positive", "simulated_negative", "unlabelled"},
    }
    for task in plan.tasks:
        dataset = task.parameters["factory_parameters"]["dataset"]["parameters"]
        policy = dataset["annotation_settings"]["targets"]["molecule"]["simulated_negative"]
        sampler = task.parameters["training"]["phases"][0]["supervision_sampling"]
        predictive_heads = task.parameters["factory_parameters"]["predictive"]["heads"]
        configured_head_losses = task.parameters["training"]["phases"][0]["criterions"]["heads"]
        assert policy["type"] == "SignalEvidenceSimulatedNegative"
        assert policy["parameters"]["relative_threshold"] == 0.0119
        assert len(predictive_heads) == 1
        assert set(configured_head_losses) == set(predictive_heads)
        assert sum(len(losses) for losses in configured_head_losses.values()) == 1
        criterion_definition = next(
            definition
            for losses in configured_head_losses.values()
            for definition in losses.values()
        )
        criterion = criterion_definition["target"]
        expected_pools = expected_sampling[criterion]
        if (
            criterion == "TaylorVariationalPULoss"
            and criterion_definition.get("params", {}).get("simulated_negative_weight", 0) > 0
        ):
            expected_pools = {"positive", "simulated_negative", "unlabelled"}
        assert set(sampler["proportions"]) == expected_pools
        assert all(value == 1 for value in sampler["proportions"].values())
        assert task.reproducibility["common_seeds"] == {"split": 42, "dataloader": 43}


def test_jerm_campaign_is_separate_and_does_not_use_replacement_sampling() -> None:
    """JERM retains full coverage required by its global modified-spy update."""
    config = load_experiment_config(ROOT / "predictive_heads_expanded_jerm.yaml")
    plan = build_plan(config)

    assert len(plan.tasks) == 5
    phase = plan.tasks[0].parameters["training"]["phases"][0]
    assert "supervision_sampling" not in phase
    assert phase["dataloader"]["shuffle"] is True
