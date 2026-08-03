"""Integration tests for the installed execution command."""

from __future__ import annotations

from pathlib import Path

import yaml

from msi_autoencoder_wrapper.execution.cli import main


def preflight_entrypoint(task: dict) -> dict:
    """Represent a successful pipeline probe without external resources."""
    return {"seed": task["seed"]}


def task_entrypoint(task: dict) -> dict:
    """Represent one completed lightweight experiment task."""
    return {"task_id": task["task_id"]}


def _write_config(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "experiment": {"name": "cli-smoke"},
        "task": {
            "entrypoint": "tests.execution.test_cli:task_entrypoint",
            "preflight_entrypoint": "tests.execution.test_cli:preflight_entrypoint",
            "parameters": {"model": {"latent_dim": 8}},
        },
        "runs": {"repetitions": 2, "base_seed": 11},
        "grid": {"model.latent_dim": [8, 16]},
        "execution": {"backend": "local", "max_parallel_runs": 1},
        "reports": [],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_plan_command_runs_preflight_and_materializes_all_tasks(tmp_path: Path) -> None:
    """Planning verifies the pipeline before writing deterministic task files."""
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "plan"
    _write_config(config)

    main(["plan", str(config), "--output", str(output)])

    assert len(list((output / "tasks").glob("task_*.yaml"))) == 4
    assert (output / "resolved-experiment.yaml").is_file()
