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
        / "autoencoder_architecture"
        / "experiment_runs_configs"
        / "23_08_26_architecture_predictive"
        / "architecture_predictive_experiment.yaml"
    )

    plan = build_plan(config)

    assert len(plan.tasks) == 35
    assert {
        task.parameters["factory_parameters"]["variant"]["name"]
        for task in plan.tasks
    } == {"conv1d-ae-32-16-8-latent-10"}
    for task in plan.tasks:
        dataset = task.parameters["factory_parameters"]["dataset"]["parameters"]
        binning = task.parameters["factory_parameters"]["binning"]["parameters"]
        architecture = task.parameters["factory_parameters"]["variant"]["parameters"]
        phase = task.parameters["training"]["phases"][0]
        assert dataset["split"]["strategy"] == "grouped"
        assert dataset["split"]["parameters"]["group_fields"] == "dataset_id"
        assert dataset["annotation_settings"]["mapping"]["x_mapping"] == "binner"
        assert (
            dataset["annotation_settings"]["targets"]["molecule"]
            ["empty_spectrum_policy"]
            == "exclude"
        )
        assert (
            dataset["annotation_settings"]["targets"]["molecule"]
            ["unobserved_label_policy"]
            == "unlabelled"
        )
        assert binning["bin_step"] == 0.55
        assert architecture["output_normalization"]["type"] == "tic"
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

        contrastive = phase["criterions"].get("contrastive", {}).get(
            "peak_permutation"
        )
        if contrastive is not None:
            assert contrastive["params"]["permutation_bank_size"] == 7000
            assert contrastive["params"]["permuted_peaks_per_view"] == 3
            assert contrastive["params"]["permutation_selection_attempts"] == 64

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


def test_contractive_metric_weight_campaign_expands_all_metric_weight_pairs() -> None:
    """The contractive campaign has twelve metric-weight cells and five repeats."""
    repository = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        repository
        / "assets"
        / "experiments"
        / "autoencoder_architecture"
        / "experiment_runs_configs"
        / "05_09_26_contractive_expaned"
        / "contractive_metric_weight_experiment.yaml"
    )

    plan = build_plan(config)

    assert len(plan.tasks) == 60
    metric_weights = set()
    for task in plan.tasks:
        contractive = task.parameters["training"]["phases"][0]["criterions"][
            "regularization"
        ]["contractive"]
        metric_weights.add(
            (contractive["params"]["penalty_metric"], contractive["weight"])
        )
        assert contractive["params"]["penalized_space"] == "u"

    expected_weights = {1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2}
    assert metric_weights == {
        (metric, weight)
        for metric in ("frobenius", "spectral", "hinged")
        for weight in expected_weights
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


def test_predictive_head_can_select_named_molecule_classes() -> None:
    """Named head classes select matching columns from dataset-derived targets."""

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
                "molecule_subset": {
                    "target_field": "molecule",
                    "class_names": ["c", "a"],
                    "strategy": "LinearClassificationHead",
                    "parameters": {"latent_dim": 2, "output_dim": "auto_from_target"},
                }
            }
        },
        TargetDataset(),
    )

    assert layout["heads"]["molecule_subset"]["params"]["output_dim"] == 2
    assert model_parameters == {
        "head_specs": {
            "molecule_subset": {
                "target_field": "molecule",
                "class_indices": (2, 0),
            }
        }
    }
