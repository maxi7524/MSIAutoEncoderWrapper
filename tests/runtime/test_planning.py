"""Tests for deterministic experiment planning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest

from msi_autoencoder_wrapper.runtime.planning import build_plan
from msi_autoencoder_wrapper.runtime.naming import run_identifier


def _config() -> dict:
    return {
        "_config_path": "/tmp/experiment.yaml",
        "experiment": {"name": "comparison"},
        "task": {
            "entrypoint": "tests.runtime.test_planning:task_entrypoint",
            "preflight_entrypoint": "tests.runtime.test_planning:task_entrypoint",
            "plan_entrypoint": "tests.runtime.test_planning:plan_entrypoint",
            "parameters": {"model": {"latent_dim": {"grid": "latent_dimensions"}}},
        },
        "runs": {"repetitions": 2},
        "seeds": {
            "common_seeds": {"split": 1, "dataloader": 2},
            "run_seeds": {"model_initialization": 123, "training": 456},
        },
        "grid": {"latent_dimensions": {"values": [8, 16]}},
        "execution": {"backend": "local"},
        "reports": [],
    }


def task_entrypoint(task: dict) -> dict:
    """Return task metadata for entrypoint resolution tests."""
    return {"task_id": task["task_id"]}


def plan_entrypoint(tasks: list[dict], _directory: object) -> list[dict]:
    """Return the task parameters unchanged for planner-only tests."""
    return [task["parameters"] for task in tasks]


def test_repetition_seed_is_shared_across_grid_variants() -> None:
    """Every model variant sees the same randomness for one repetition."""
    plan = build_plan(_config())

    assert len(plan.tasks) == 4
    first_seed = plan.tasks[0].reproducibility["derived_run_seeds"]["model_initialization"]
    assert first_seed == plan.tasks[2].reproducibility["derived_run_seeds"]["model_initialization"]
    assert first_seed != plan.tasks[1].reproducibility["derived_run_seeds"]["model_initialization"]
    assert plan.tasks[0].reproducibility["common_seeds"]["split"] == 1
    assert plan.tasks[0].parameters["model"]["latent_dim"] == 8
    assert plan.tasks[2].parameters["model"]["latent_dim"] == 16


def test_plan_is_stable_for_the_same_configuration() -> None:
    """Rerunning or resuming a campaign preserves resolved seeds."""
    first = build_plan(_config())
    second = build_plan(_config())

    assert first == second


def test_plan_fingerprint_changes_when_settings_change() -> None:
    """A materialized plan cannot be reused after changing its YAML settings."""
    first = build_plan(_config())
    changed = deepcopy(_config())
    changed["execution"]["max_parallel_runs"] = 2
    second = build_plan(changed)

    assert first.config_fingerprint != second.config_fingerprint


def test_plan_reuses_one_grid_cell_for_corresponding_head_and_criterion() -> None:
    """A selected grid field keeps one method's head and criterion coupled."""
    config = _config()
    config["task"]["parameters"] = {
        "predictive": {"grid": "methods", "select": "predictive"},
        "training": {
            "criterions": {"grid": "methods", "select": "criterions"},
        },
    }
    config["runs"] = {"repetitions": 1}
    config["grid"] = {
        "methods": {
            "values": [
                {
                    "predictive": {
                        "heads": {"binary": {"strategy": "LinearClassificationHead"}},
                    },
                    "criterions": {"heads": {"binary": {"loss": "BCEPNLoss"}}},
                },
                {
                    "predictive": {
                        "heads": {"selective": {"strategy": "SelectiveHead"}},
                    },
                    "criterions": {
                        "heads": {"selective": {"loss": "SelectivePNLoss"}},
                    },
                },
            ],
        },
    }

    plan = build_plan(config)

    assert len(plan.tasks) == 2
    assert plan.tasks[0].parameters["predictive"] == config["grid"]["methods"]["values"][0]["predictive"]
    assert plan.tasks[0].parameters["training"]["criterions"] == config["grid"]["methods"]["values"][0]["criterions"]


def test_selected_grid_value_resolves_references_to_another_grid() -> None:
    """A schedule may select axis-specific constants without a copied matrix."""
    config = _config()
    config["runs"] = {"repetitions": 1}
    config["task"]["parameters"] = {
        "axis": {"grid": "axes", "select": "binning"},
        "phases": {"grid": "schedules"},
    }
    config["grid"] = {
        "axes": {
            "values": [
                {"binning": "short", "minimum": 3},
                {"binning": "extended", "minimum": 4},
            ]
        },
        "schedules": {
            "values": [
                {
                    "min_fragments": {
                        "grid": "axes",
                        "select": "minimum",
                    }
                }
            ]
        },
    }

    plan = build_plan(config)

    assert [task.parameters for task in plan.tasks] == [
        {"axis": "short", "phases": {"min_fragments": 3}},
        {"axis": "extended", "phases": {"min_fragments": 4}},
    ]


def test_recursive_grid_value_is_rejected() -> None:
    """Nested resolution fails explicitly instead of recursing forever."""
    config = _config()
    config["runs"] = {"repetitions": 1}
    config["task"]["parameters"] = {"value": {"grid": "recursive"}}
    config["grid"] = {
        "recursive": {"values": [{"grid": "recursive"}]},
    }

    with pytest.raises(ValueError, match="recursive reference"):
        build_plan(config)


def test_pretraining_workflow_expands_parent_and_independent_children() -> None:
    """One phase schedule becomes three ordered model tasks with explicit lineage."""
    config = _config()
    config["runs"] = {"repetitions": 1}
    config["grid"] = {}
    config["task"]["workflow"] = {"strategy": "pretraining_branches"}
    config["task"]["parameters"] = {
        "training": {
            "phases": [
                {"phase_name": "synthetic", "workflow_role": "pretrained"},
                {"phase_name": "real_test", "workflow_role": "pretrained"},
                {"phase_name": "frozen", "workflow_role": "frozen_head"},
                {"phase_name": "unfrozen", "workflow_role": "unfrozen_head"},
            ]
        }
    }

    plan = build_plan(config)

    assert [task.workflow["role"] for task in plan.tasks] == [
        "pretrained",
        "frozen_head",
        "unfrozen_head",
    ]
    parent = plan.tasks[0]
    assert [
        phase["phase_name"]
        for phase in parent.parameters["training"]["phases"]
    ] == ["synthetic", "real_test"]
    assert plan.tasks[1].depends_on == (parent.task_id,)
    assert plan.tasks[2].depends_on == (parent.task_id,)
    assert plan.tasks[1].workflow["parent_task_id"] == parent.task_id
    assert plan.tasks[2].workflow["parent_task_id"] == parent.task_id
    assert plan.tasks[1].grid_id == parent.grid_id == plan.tasks[2].grid_id
    assert run_identifier("campaign", asdict(parent)).endswith("__pretrained")
    assert run_identifier("campaign", asdict(plan.tasks[1])).endswith("__frozen_head")
    assert run_identifier("campaign", asdict(plan.tasks[2])).endswith("__unfrozen_head")
