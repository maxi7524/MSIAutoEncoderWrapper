"""Integration tests for the installed execution command."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from msi_autoencoder_wrapper.runtime import build_plan, load_experiment_config
from msi_autoencoder_wrapper.runtime.cli import (
    _has_complete_plan,
    _set_experiment_directory,
    main,
)


def preflight_entrypoint(task: dict) -> dict:
    """Represent a successful pipeline probe without external resources."""
    return {"task_id": task["task_id"]}


def task_entrypoint(task: dict) -> dict:
    """Represent one completed lightweight experiment task."""
    return {
        "task_id": task["task_id"],
        "model_name": task["runtime"]["model_name"],
    }


def probe_entrypoint(task: dict) -> dict:
    """Represent one non-persistent pipeline probe."""
    return {"task_id": task["task_id"], "test_mode": True}


def plan_entrypoint(tasks: list[dict], directory: Path) -> list[dict]:
    """Mark parameters so the test can detect plan resolution."""
    return [{**task["parameters"], "planned": True} for task in tasks]


def _write_config(
    path: Path,
    *,
    with_plan_entrypoint: bool = False,
    with_test_entrypoint: bool = False,
) -> None:
    payload = {
        "schema_version": 1,
        "experiment": {"name": "cli-smoke"},
        "task": {
            "entrypoint": "tests.runtime.test_cli:task_entrypoint",
            "preflight_entrypoint": "tests.runtime.test_cli:preflight_entrypoint",
            "plan_entrypoint": "tests.runtime.test_cli:plan_entrypoint",
            "parameters": {
                "model": {"latent_dim": {"grid": "latent_dimensions"}},
                "training": {
                    "continuation": {
                        "enabled": True,
                        "resume": True,
                        "checkpoint_every_epochs": 1,
                    }
                },
            },
        },
        "runs": {"repetitions": 2},
        "seeds": {
            "common_seeds": {"split": 1, "dataloader": 2},
            "run_seeds": {"model_initialization": 11, "training": 12},
        },
        "grid": {"latent_dimensions": {"values": [8, 16]}},
        "execution": {"backend": "local", "max_parallel_runs": 1},
        "reports": [],
    }
    if with_test_entrypoint:
        payload["task"]["test_entrypoint"] = "tests.runtime.test_cli:probe_entrypoint"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_plan_command_runs_preflight_and_materializes_all_tasks(tmp_path: Path) -> None:
    """Planning verifies the pipeline before writing deterministic task files."""
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "plan"
    _write_config(config)

    main(["plan", str(config), "--output", str(output)])

    assert len(list((output / "tasks").glob("task_*.yaml"))) == 4
    assert (output / "resolved-experiment.yaml").is_file()


def test_run_reuses_complete_materialized_plan(tmp_path: Path, monkeypatch) -> None:
    """Running a planned campaign must not resolve its artifacts again."""
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "plan"
    _write_config(config, with_plan_entrypoint=True)
    main(["plan", str(config), "--output", str(output)])

    def fail_if_resolved_again(*args, **kwargs):
        raise AssertionError("The existing plan was resolved again")

    monkeypatch.setattr(
        "msi_autoencoder_wrapper.runtime.cli.resolve_plan", fail_if_resolved_again
    )
    main(["run", str(config), "--output", str(output)])

    statuses = list((output / "status").glob("task_*.yaml"))
    assert len(statuses) == 4

    monkeypatch.setattr(
        "msi_autoencoder_wrapper.runtime.cli.execute_task",
        lambda _task: (_ for _ in ()).throw(AssertionError("Completed task was rerun")),
    )
    main(["run", str(config), "--output", str(output)])

    first_status = yaml.safe_load(
        (output / "status" / "task_000000.yaml").read_text(encoding="utf-8")
    )
    model_name = first_status["records"]["task_000000"]["result"]["model_name"]
    manifest = yaml.safe_load(
        (output / "resolved-experiment.yaml").read_text(encoding="utf-8")
    )
    assert model_name == (
        "cli-smoke__cfg_"
        f"{manifest['config_fingerprint'][:12]}__grid_0000__rep_00"
    )


def test_changed_yaml_invalidates_a_materialized_plan(tmp_path: Path) -> None:
    """Plan reuse requires the exact experiment configuration fingerprint."""
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "plan"
    _write_config(config)
    main(["plan", str(config), "--output", str(output)])

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["execution"]["max_parallel_runs"] = 2
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    changed_plan = build_plan(load_experiment_config(config))
    assert not _has_complete_plan(output, changed_plan)

    with pytest.raises(FileExistsError, match="different configuration"):
        main(["run", str(config), "--output", str(output), "--dry-run"])


def test_default_campaign_directory_contains_configuration_fingerprint(
    tmp_path: Path,
) -> None:
    """Changed settings receive a sibling campaign instead of overwriting output."""
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path)
    config = load_experiment_config(config_path)

    first = _set_experiment_directory(config, None)
    config["runs"]["repetitions"] = 3
    second = _set_experiment_directory(config, None)

    assert first.parent == second.parent
    assert first.name.startswith("cli-smoke__cfg_")
    assert first != second


def test_run_id_creates_an_independent_instance_of_the_same_configuration(
    tmp_path: Path,
) -> None:
    """An explicit run identity isolates repeated execution without changing YAML."""
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path)
    config = load_experiment_config(config_path)

    default = _set_experiment_directory(config, None)
    first = _set_experiment_directory(config, None, "trial-a")
    second = _set_experiment_directory(config, None, "trial-b")

    assert first.name == f"{default.name}__run_trial-a"
    assert second.name == f"{default.name}__run_trial-b"
    with pytest.raises(ValueError, match="cannot be used together"):
        _set_experiment_directory(config, tmp_path / "output", "trial-a")


def test_run_test_run_probes_every_grid_cell_without_training_status(tmp_path: Path) -> None:
    """Test mode executes one task per grid cell in a separate status directory."""
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "plan"
    _write_config(config, with_test_entrypoint=True)

    main(["run", str(config), "--output", str(output), "--test-run"])

    assert len(list((output / "test-status").glob("task_*.yaml"))) == 2
    assert not (output / "status").exists()
