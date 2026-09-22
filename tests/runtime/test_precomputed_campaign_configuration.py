"""Configuration contract for the artifact-backed kidney pretraining campaign."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config


CONFIGURATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets/experiments/autoencoder_architecture/experiment_runs_configs"
    / "segmentation_model/20_09_26_metaspace_base_pretrain"
    / "pretraining_experiment.yaml"
)

EXPECTED_SYNTHETIC_SCHEDULES = {
    ("synthetic_single",),
    ("synthetic_permutation_base",),
    ("synthetic_joint_base",),
    ("synthetic_single_stage", "synthetic_permutation_base"),
    ("synthetic_joint_overlap",),
    ("synthetic_single_stage", "synthetic_permutation_overlap"),
    ("synthetic_joint_rare",),
    ("synthetic_single_stage", "synthetic_permutation_rare"),
    ("synthetic_joint_all",),
    ("synthetic_single_stage", "synthetic_permutation_all"),
}


def _campaign():
    """Load and materialize the canonical pretraining campaign."""
    config = load_experiment_config(CONFIGURATION_PATH)
    return config, build_plan(config)


def _synthetic_phases(task) -> list[dict[str, Any]]:
    """Return synthetic phases in execution order."""
    return [
        phase
        for phase in task.parameters["training"]["phases"]
        if phase.get("pretraining", {}).get("kind") == "precomputed_synthetic"
    ]


def _tasks_by_role(plan, group_id: str) -> dict[str, Any]:
    """Return one workflow group's tasks indexed by persisted artifact role."""
    return {
        task.workflow["role"]: task
        for task in plan.tasks
        if task.workflow["group_id"] == group_id
    }


def test_campaign_expands_the_complete_paired_ablation_matrix():
    """Every synthetic run expands to three independently persisted models."""
    _, plan = _campaign()
    assert len(plan.tasks) == 310

    grouped: dict[str, set[str]] = {}
    for task in plan.tasks:
        grouped.setdefault(task.workflow["group_id"], set()).add(
            task.workflow["role"]
        )
    assert len(grouped) == 110
    assert sum(roles == {"real_only"} for roles in grouped.values()) == 10
    assert sum(
        roles == {"pretrained", "frozen_head", "unfrozen_head"}
        for roles in grouped.values()
    ) == 100

    signatures = Counter(
        tuple(phase["phase_name"] for phase in _synthetic_phases(task))
        for task in plan.tasks
        if _synthetic_phases(task)
    )
    assert set(signatures) == EXPECTED_SYNTHETIC_SCHEDULES
    assert set(signatures.values()) == {10}
    assert sum(task.workflow["role"] == "real_only" for task in plan.tasks) == 10
    assert sum(task.workflow["role"] == "frozen_head" for task in plan.tasks) == 100
    assert sum(task.workflow["role"] == "unfrozen_head" for task in plan.tasks) == 100

    # Every axis/schedule pair reuses its split seed across five model seeds.
    for grid_id in {task.grid_id for task in plan.tasks}:
        tasks = [task for task in plan.tasks if task.grid_id == grid_id]
        roles = {task.workflow["role"] for task in tasks}
        assert len(tasks) == (5 if roles == {"real_only"} else 15)
        assert {task.reproducibility["common_seeds"]["split"] for task in tasks} == {
            42
        }
        assert len(
            {
                task.reproducibility["derived_run_seeds"]["training"]
                for task in tasks
            }
        ) == 5


def test_axes_keep_manual_fragment_bounds_and_data_contract():
    """Manual fragment bounds stay paired with their axis after nested expansion."""
    _, plan = _campaign()
    observed_axes = set()
    for task in plan.tasks:
        factory = task.parameters["factory_parameters"]
        binning = factory["binning"]["parameters"]
        phases = _synthetic_phases(task)
        if phases:
            artifact = phases[0]["pretraining"]["artifact"]
            base = artifact["populations"]["permutation_base"]
            observed_axes.add(
                (
                    binning["x_min"],
                    binning["x_max"],
                    base["min_fragments"],
                    base["max_fragments"],
                    artifact["key"],
                )
            )

        dataset = factory["dataset"]["parameters"]
        assert dataset["subset"]["fraction"] == 0.2
        assert dataset["split"]["fractions"] == {
            "train": 0.9,
            "validation": 0.05,
            "test": 0.05,
        }
        assert factory["cohort_selection"]["parameters"][
            "excluded_dataset_ids"
        ] == [
            "2024-02-20_01h45m20s",
            "2024-02-20_01h46m58s",
            "2024-02-20_01h54m41s",
        ]

    assert observed_axes == {
        (200.0, 900.0, 3, 61, "kidney-metaspace-200-900-v3"),
        (100.0, 3000.0, 4, 68, "kidney-metaspace-100-3000-v3"),
    }


def test_artifact_population_contract_has_exact_quotas_and_joint_bags():
    """Single, base and additive quota populations match the frozen contract."""
    _, plan = _campaign()
    phase = _synthetic_phases(
        next(task for task in plan.tasks if _synthetic_phases(task))
    )[0]
    artifact = phase["pretraining"]["artifact"]
    populations = artifact["populations"]

    assert populations["axis"] == {
        "strategy": "axis_coverage",
        "repetitions_per_bin": 10,
    }
    for name in (
        "permutation_base",
        "permutation_overlap",
        "permutation_rare",
        "permutation_all",
    ):
        assert populations[name]["strategy"] == "class_quota_mixture"
        assert populations[name]["class_quota"] == 30
        assert populations[name]["blank_bin_quota"] == 30
    assert "overlap_bonus_per_class" not in populations["permutation_base"]
    assert "rare_bonus_per_class" not in populations["permutation_base"]
    assert populations["permutation_overlap"]["overlap_bonus_per_class"] == 30
    assert populations["permutation_rare"]["rare_bonus_per_class"] == 30
    assert populations["permutation_rare"]["rare_fraction"] == 0.25
    assert populations["permutation_all"]["overlap_bonus_per_class"] == 30
    assert populations["permutation_all"]["rare_bonus_per_class"] == 30
    assert populations["permutation_all"]["rare_fraction"] == 0.25
    assert populations["joint_base"]["members"] == ["axis", "permutation_base"]
    assert populations["joint_overlap"]["members"] == [
        "axis",
        "permutation_overlap",
    ]
    assert populations["joint_rare"]["members"] == ["axis", "permutation_rare"]
    assert populations["joint_all"]["members"] == ["axis", "permutation_all"]
    assert artifact["representation"]["strategy"] == "isospec_envelope"
    assert artifact["component_mixing"] == {
        "distribution": "dirichlet",
        "minimum_annotated_concentration": 2.0,
        "blank_concentration": 1.0,
    }


def test_synthetic_and_real_phases_reuse_one_head_with_distinct_objectives():
    """The same logits use BCE during synthesis and the final VPU objective on real data."""
    _, plan = _campaign()
    pretrained = next(task for task in plan.tasks if _synthetic_phases(task))
    group = _tasks_by_role(plan, pretrained.workflow["group_id"])
    assert set(pretrained.parameters["factory_parameters"]["predictive"]["heads"]) == {
        "molecule_vpu"
    }
    phases = pretrained.parameters["training"]["phases"]
    synthetic = _synthetic_phases(pretrained)
    for phase in synthetic:
        assert phase["epochs"] == 10
        assert phase["criterions"] == {
            "reconstruction": {
                "masserstein": {"target": "MassersteinLoss", "weight": 1.0}
            },
            "heads": {
                "molecule_vpu": {
                    "bce": {"target": "MultiLabelBCELoss", "weight": 0.2}
                }
            },
        }

    pretrain_test = phases[-1]
    frozen = group["frozen_head"].parameters["training"]["phases"]
    unfrozen = group["unfrozen_head"].parameters["training"]["phases"]
    assert pretrain_test["phase_name"] == "pretrain_only_real_test"
    assert pretrain_test["epochs"] == 0
    assert pretrain_test["evaluation_data"] == "real"
    assert [phase["phase_name"] for phase in frozen] == [
        "real_adaptation_frozen_head"
    ]
    assert [phase["phase_name"] for phase in unfrozen] == [
        "real_adaptation_unfrozen_head"
    ]
    assert frozen[0]["freeze"] == ["heads.molecule_vpu"]
    assert unfrozen[0]["freeze"] == []
    assert all(
        "restore_model_state_from" not in phase
        for phase in (pretrain_test, frozen[0], unfrozen[0])
    )
    assert all(
        phase["evaluate_test"] is True
        for phase in (pretrain_test, frozen[0], unfrozen[0])
    )

    parent_id = pretrained.task_id
    assert pretrained.depends_on == ()
    assert group["frozen_head"].depends_on == (parent_id,)
    assert group["unfrozen_head"].depends_on == (parent_id,)
    assert group["frozen_head"].workflow["parent_task_id"] == parent_id
    assert group["unfrozen_head"].workflow["parent_task_id"] == parent_id

    real_objective = pretrain_test["criterions"]
    assert all(
        phase["criterions"] == real_objective
        for phase in (frozen[0], unfrozen[0])
    )
    assert real_objective["reconstruction"]["masserstein"] == {
        "target": "MassersteinLoss",
        "weight": 1.0,
    }
    assert real_objective["heads"]["molecule_vpu"]["vpu"]["weight"] == 0.2
    assert real_objective["regularization"]["contractive"]["weight"] == 2.0e-3
    contrastive = real_objective["contrastive"]["peak_permutation"]
    assert contrastive["weight"] == 0.001
    assert contrastive["params"]["temperature"] == 0.07
    assert contrastive["params"]["negative_weighting_method"] == (
        "multilabel_jaccard"
    )
    assert contrastive["params"]["overlapping_label_negative_weight"] == 0.1


def test_joint_and_staged_variants_reference_the_same_precompute_artifact():
    """Ordering ablations reuse one axis artifact and differ only in phase boundaries."""
    _, plan = _campaign()
    for task in plan.tasks:
        synthetic = _synthetic_phases(task)
        if not synthetic:
            continue
        artifacts = [phase["pretraining"]["artifact"] for phase in synthetic]
        assert all(artifact == artifacts[0] for artifact in artifacts)
        assert all("save_model_state_as" not in phase for phase in synthetic)
        if len(synthetic) == 2:
            assert synthetic[0]["pretraining"]["population"] == "axis"
            assert synthetic[1]["pretraining"]["population"].startswith(
                "permutation_"
            )
        else:
            population = synthetic[0]["pretraining"]["population"]
            assert population == "axis" or population == "permutation_base" or (
                population.startswith("joint_")
            )
