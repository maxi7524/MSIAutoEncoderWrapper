"""YAML loading and profile merging for experiment runtime campaigns."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def _repository_root(config_path: Path) -> Path:
    """Locate the repository containing an experiment configuration.

    :param config_path: Absolute path to the loaded experiment configuration.
    :type config_path: pathlib.Path
    :return: Repository root identified by its ``pyproject.toml`` file.
    :rtype: pathlib.Path
    :raises ValueError: If the configuration is not located below a repository root.
    """
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(
        "project_path_anchor: repository requires the experiment YAML to be located "
        "inside a repository containing pyproject.toml."
    )


def _resolve_project_path(factory_parameters: dict[str, Any], config_path: Path) -> None:
    """Resolve the workspace path using its explicit YAML anchor.

    :param factory_parameters: Factory configuration mutated with an absolute workspace path.
    :type factory_parameters: dict[str, typing.Any]
    :param config_path: Absolute path to the loaded experiment configuration.
    :type config_path: pathlib.Path
    :raises ValueError: If the path or its anchor is invalid.
    """
    project_path = factory_parameters.get("project_path")
    if project_path is None:
        return
    if not isinstance(project_path, str) or not project_path:
        raise ValueError("factory_parameters.project_path must be a non-empty string.")

    anchor = factory_parameters.get("project_path_anchor")
    if anchor not in {"repository", "working_directory"}:
        raise ValueError(
            "factory_parameters.project_path_anchor must be 'repository' or "
            "'working_directory'."
        )
    base_directory = (
        _repository_root(config_path) if anchor == "repository" else Path.cwd()
    )
    candidate = Path(project_path)
    factory_parameters["project_path"] = str(
        candidate.resolve() if candidate.is_absolute() else (base_directory / candidate).resolve()
    )


def load_experiment_config(path: Path | str) -> dict[str, Any]:
    """Load an experiment and merge its optional execution profile."""
    config_path = Path(path).resolve()
    config = _read_yaml(config_path)
    execution = config.get("execution", {})
    if isinstance(execution, dict) and execution.get("profile"):
        profile_path = (config_path.parent / execution["profile"]).resolve()
        profile = _read_yaml(profile_path)
        profile_execution = profile.get("execution", profile)
        execution = _merge(
            profile_execution,
            {key: value for key, value in execution.items() if key != "profile"},
        )
        execution["profile_path"] = str(profile_path)
        config["execution"] = execution
    config["_config_path"] = str(config_path)
    config["_config_directory"] = str(config_path.parent)
    factory_parameters = config.get("task", {}).get("parameters", {}).get(
        "factory_parameters", {}
    )
    _resolve_project_path(factory_parameters, config_path)
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate fields required before task materialization."""
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ValueError("experiment.name is required")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment["name"]) is None:
        raise ValueError(
            "experiment.name may contain only letters, numbers, '.', '_' and '-'"
        )
    task = config.get("task")
    if not isinstance(task, dict) or not task.get("entrypoint"):
        raise ValueError("task.entrypoint is required and must use 'module:function'")
    if ":" not in task["entrypoint"]:
        raise ValueError("task.entrypoint must use 'module:function'")
    preflight_entrypoint = task.get("preflight_entrypoint")
    if not isinstance(preflight_entrypoint, str) or ":" not in preflight_entrypoint:
        raise ValueError("task.preflight_entrypoint is required and must use 'module:function'")
    plan_entrypoint = task.get("plan_entrypoint")
    if not isinstance(plan_entrypoint, str) or ":" not in plan_entrypoint:
        raise ValueError("task.plan_entrypoint is required and must use 'module:function'")
    test_entrypoint = task.get("test_entrypoint")
    if test_entrypoint is not None and (
        not isinstance(test_entrypoint, str) or ":" not in test_entrypoint
    ):
        raise ValueError("task.test_entrypoint must use 'module:function'")
    runs = config.get("runs", {})
    if not isinstance(runs.get("repetitions", 1), int) or runs.get("repetitions", 1) < 1:
        raise ValueError("runs.repetitions must be a positive integer")
    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("seeds must be a mapping.")
    for section, required in {
        "common_seeds": {"split", "dataloader"},
        "run_seeds": {"model_initialization", "training"},
    }.items():
        values = seeds.get(section)
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError(f"seeds.{section} must contain exactly {sorted(required)}.")
        if any(not isinstance(value, int) for value in values.values()):
            raise ValueError(f"seeds.{section} values must be integers.")
    grid = config.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("grid must be a mapping of named grid definitions.")
    for name, definition in grid.items():
        if (
            not isinstance(name, str)
            or not isinstance(definition, dict)
            or not isinstance(definition.get("values"), list)
            or not definition["values"]
        ):
            raise ValueError("Every grid entry requires a non-empty values list.")
    execution = config.get("execution", {})
    if execution.get("backend", "local") not in {"local", "slurm"}:
        raise ValueError("execution.backend must be 'local' or 'slurm'")
    if execution.get("backend") == "slurm":
        staging = execution.get("staging", {})
        if not staging.get("enabled") or not staging.get("root"):
            raise ValueError("Slurm execution requires an enabled staging root")
        parallelism = execution.get("slurm", {}).get("array_parallelism", 1)
        if not isinstance(parallelism, int) or parallelism < 1:
            raise ValueError("execution.slurm.array_parallelism must be positive")
    max_parallel_runs = execution.get("max_parallel_runs", 1)
    if not isinstance(max_parallel_runs, int) or max_parallel_runs < 1:
        raise ValueError("execution.max_parallel_runs must be positive")
    reports = config.get("reports", [])
    if not isinstance(reports, list):
        raise ValueError("reports must be an ordered list")
    for report in reports:
        if not isinstance(report, (str, dict)):
            raise ValueError("Every report must be a path or mapping")
        if isinstance(report, dict) and not report.get("source"):
            raise ValueError("Every report mapping requires source")
    training = task.get("parameters", {}).get("training", {})
    continuation = training.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("task.parameters.training.continuation is required.")
    if not isinstance(continuation.get("enabled"), bool) or not isinstance(
        continuation.get("resume"), bool
    ):
        raise ValueError("continuation.enabled and continuation.resume must be booleans.")
    interval = continuation.get("checkpoint_every_epochs")
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("continuation.checkpoint_every_epochs must be positive.")
