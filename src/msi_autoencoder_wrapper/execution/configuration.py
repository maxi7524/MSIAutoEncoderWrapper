"""YAML loading and validation for experiment execution."""

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
    runs = config.get("runs", {})
    if not isinstance(runs.get("repetitions", 1), int) or runs.get("repetitions", 1) < 1:
        raise ValueError("runs.repetitions must be a positive integer")
    if not isinstance(runs.get("base_seed", 0), int):
        raise ValueError("runs.base_seed must be an integer")
    grid = config.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("grid must be a mapping of dotted paths to value lists")
    for path, values in grid.items():
        if not isinstance(path, str) or not isinstance(values, list) or not values:
            raise ValueError("Every grid entry must contain a non-empty value list")
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
