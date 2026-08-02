"""Tests for deterministic experiment planning."""

from __future__ import annotations

from msi_autoencoder_wrapper.execution.planning import build_plan


def _config() -> dict:
    return {
        "_config_path": "/tmp/experiment.yaml",
        "experiment": {"name": "comparison"},
        "task": {
            "entrypoint": "tests.execution.test_planning:task_entrypoint",
            "preflight_entrypoint": "tests.execution.test_planning:task_entrypoint",
            "parameters": {"model": {"latent_dim": 8}},
        },
        "runs": {"repetitions": 2, "base_seed": 123},
        "grid": {"model.latent_dim": [8, 16]},
        "execution": {"backend": "local"},
        "reports": [],
    }


def task_entrypoint(task: dict) -> dict:
    """Return task metadata for entrypoint resolution tests."""
    return {"task_id": task["task_id"]}


def test_repetition_seed_is_shared_across_grid_variants() -> None:
    """Every model variant sees the same randomness for one repetition."""
    plan = build_plan(_config())

    assert len(plan.tasks) == 4
    assert plan.tasks[0].seed == plan.tasks[2].seed
    assert plan.tasks[1].seed == plan.tasks[3].seed
    assert plan.tasks[0].seed != plan.tasks[1].seed
    assert plan.tasks[0].parameters["model"]["latent_dim"] == 8
    assert plan.tasks[2].parameters["model"]["latent_dim"] == 16


def test_plan_is_stable_for_the_same_configuration() -> None:
    """Rerunning or resuming a campaign preserves resolved seeds."""
    first = build_plan(_config())
    second = build_plan(_config())

    assert first == second
