"""Expand experiment grids into deterministic, portable tasks."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PlannedTask:
    """One model run with resolved parameters and reproducibility metadata."""

    task_id: str
    grid_id: str
    repetition: int
    seed: int
    grid_parameters: dict[str, Any]
    entrypoint: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ExperimentPlan:
    """Materialized execution plan shared by all execution backends."""

    experiment_name: str
    config_path: str
    tasks: tuple[PlannedTask, ...]
    execution: dict[str, Any]
    reports: tuple[Any, ...]


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    current = target
    keys = path.split(".")
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Grid path does not reference a mapping: {path}")
        current = child
    if keys[-1] not in current:
        raise ValueError(f"Grid path does not exist: {path}")
    current[keys[-1]] = deepcopy(value)


def repetition_seed(base_seed: int, repetition: int) -> int:
    """Derive a stable seed shared by every grid variant in a repetition."""
    digest = hashlib.sha256(f"{base_seed}:{repetition}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def build_plan(config: dict[str, Any]) -> ExperimentPlan:
    """Create the Cartesian grid while preserving paired repetition seeds."""
    grid = config.get("grid", {})
    paths = list(grid)
    combinations = product(*(grid[path] for path in paths)) if paths else [()]
    repetitions = config.get("runs", {}).get("repetitions", 1)
    base_seed = config.get("runs", {}).get("base_seed", 0)
    controls = config.get("reproducibility", {}).get(
        "controls",
        [
            "dataset_split",
            "dataloader",
            "sampler",
            "augmentation",
            "model_initialization",
            "training",
        ],
    )
    tasks: list[PlannedTask] = []
    for grid_index, values in enumerate(combinations):
        assignments = dict(zip(paths, values))
        for repetition in range(repetitions):
            parameters = deepcopy(config["task"].get("parameters", {}))
            for path, value in assignments.items():
                _set_path(parameters, path, value)
            seed = repetition_seed(base_seed, repetition)
            parameters["reproducibility"] = {"seed": seed, "controls": list(controls)}
            task_index = len(tasks)
            tasks.append(
                PlannedTask(
                    task_id=f"task_{task_index:06d}",
                    grid_id=f"grid_{grid_index:04d}",
                    repetition=repetition,
                    seed=seed,
                    grid_parameters=assignments,
                    entrypoint=config["task"]["entrypoint"],
                    parameters=parameters,
                )
            )
    return ExperimentPlan(
        experiment_name=config["experiment"]["name"],
        config_path=config["_config_path"],
        tasks=tuple(tasks),
        execution=deepcopy(config.get("execution", {"backend": "local"})),
        reports=tuple(config.get("reports", [])),
    )


def materialize_plan(plan: ExperimentPlan, directory: Path) -> Path:
    """Write resolved task descriptors before any external work starts."""
    directory.mkdir(parents=True, exist_ok=True)
    tasks_directory = directory / "tasks"
    tasks_directory.mkdir(exist_ok=True)
    for task in plan.tasks:
        with (tasks_directory / f"{task.task_id}.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(asdict(task), stream, sort_keys=False)
    plan_path = directory / "resolved-experiment.yaml"
    payload = {
        "experiment_name": plan.experiment_name,
        "config_path": plan.config_path,
        "execution": plan.execution,
        "reports": list(plan.reports),
        "tasks": [asdict(task) for task in plan.tasks],
    }
    with plan_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    return plan_path
