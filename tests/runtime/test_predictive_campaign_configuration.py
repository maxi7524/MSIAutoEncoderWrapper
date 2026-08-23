"""Tests for the predictive autoencoder ablation campaign blueprint."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.data import TargetSchema
from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.workflows.configured import (
    _attach_predictive_components,
    _portable_component_descriptor,
)


def test_predictive_campaign_expands_paired_joint_objective_ablations() -> None:
    """Seven joint objectives run with paired splits and five independent seeds."""
    repository = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        repository
        / "assets"
        / "experiments"
        / "08_26"
        / "23_08_26_architecture_predictive"
        / "architecture_predictive_experiment.yaml"
    )

    plan = build_plan(config)

    assert len(plan.tasks) == 35
    assert {
        task.parameters["factory_parameters"]["variant"]["name"]
        for task in plan.tasks
    } == {"mlp-ae-512-256-latent-10-layernorm"}
    for task in plan.tasks:
        dataset = task.parameters["factory_parameters"]["dataset"]["parameters"]
        binning = task.parameters["factory_parameters"]["binning"]["parameters"]
        phase = task.parameters["training"]["phases"][0]
        assert dataset["split"]["strategy"] == "grouped"
        assert dataset["split"]["parameters"]["group_fields"] == "dataset_id"
        assert binning["bin_step"] == 0.55
        assert phase["phase_name"] == "joint_predictive"
        assert {
            objective["target"]
            for objective in phase["criterions"]["reconstruction"].values()
        } == {"MassersteinLoss"}

        contractive = phase["criterions"].get("regularization", {}).get(
            "contractive"
        )
        if contractive is not None:
            assert contractive["params"]["num_probes"] == 5

    objective_methods = set()
    for task in plan.tasks:
        criterions = task.parameters["training"]["phases"][0]["criterions"]
        contrastive = criterions.get("contrastive", {}).get("peak_permutation")
        if contrastive is not None:
            objective_methods.add(
                contrastive["params"]["peak_selection_method"]
            )
    assert objective_methods == {
        "permutation_random",
        "permutation_label_invariant",
    }


def test_predictive_components_resolve_target_width_and_nested_descriptors() -> None:
    """Planning resolves head dimensions and serializes named heads recursively."""

    class TargetDataset:
        @staticmethod
        def get_target_schemas():
            return {
                "molecule": TargetSchema(
                    name="molecule",
                    target_type="multi_label",
                    class_names=("a", "b", "c"),
                )
            }

    layout, model_parameters = _attach_predictive_components(
        {"encoder": {"strategy": "Encoder", "params": {"input_dim": 5}}},
        {
            "heads": {
                "molecule_primary": {
                    "target_field": "molecule",
                    "strategy": "LinearClassificationHead",
                    "parameters": {
                        "latent_dim": 2,
                        "output_dim": "auto_from_target",
                    },
                }
            }
        },
        TargetDataset(),
    )
    portable_heads = _portable_component_descriptor(layout["heads"])

    assert layout["heads"]["molecule_primary"]["params"]["output_dim"] == 3
    assert portable_heads["molecule_primary"]["type"] == "LinearClassificationHead"
    assert portable_heads["molecule_primary"]["parameters"]["output_dim"] == 3
    assert model_parameters == {
        "head_specs": {"molecule_primary": {"target_field": "molecule"}}
    }
