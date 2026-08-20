"""Tests for deterministic experiment planning."""

from __future__ import annotations

from copy import deepcopy

from msi_autoencoder_wrapper.runtime.planning import build_plan


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
