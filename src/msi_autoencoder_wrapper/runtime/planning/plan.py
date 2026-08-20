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
    reproducibility: dict[str, dict[str, int]]
    grid_parameters: dict[str, Any]
    entrypoint: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ExperimentPlan:
    """Materialized execution plan shared by all execution backends."""

    experiment_name: str
    config_path: str
    config_fingerprint: str
    tasks: tuple[PlannedTask, ...]
    execution: dict[str, Any]
    reports: tuple[Any, ...]


def _grid_references(value: Any) -> set[str]:
    """Return all named grid references embedded in one task parameter tree."""
    if isinstance(value, dict):
        if set(value) == {"grid"}:
            reference = value["grid"]
            if not isinstance(reference, str) or not reference:
                raise ValueError("A grid reference must contain a non-empty grid name.")
            return {reference}
        return set().union(*(_grid_references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_grid_references(item) for item in value))
    return set()


def _replace_grid_references(value: Any, assignments: dict[str, Any]) -> Any:
    """Replace declarative ``{grid: name}`` nodes with selected values."""
    if isinstance(value, dict):
        if set(value) == {"grid"}:
            return deepcopy(assignments[value["grid"]])
        return {key: _replace_grid_references(item, assignments) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_grid_references(item, assignments) for item in value]
    return deepcopy(value)


def repetition_seed(base_seed: int, repetition: int, purpose: str) -> int:
    """Derive one stable but distinct seed for a repetition.

    The repetition number participates in the hash, so repetitions 0 through 4
    receive five different seeds. Grid variants within one repetition reuse that
    repetition's seed to make architecture and binning comparisons paired.
    """
    digest = hashlib.sha256(f"{base_seed}:{purpose}:{repetition}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def configuration_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable hash of user-controlled experiment settings.

    :param config: Loaded experiment configuration including runtime metadata.
    :type config: dict[str, typing.Any]
    :return: SHA-256 fingerprint excluding loader-injected private keys.
    :rtype: str
    """
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    serialized = yaml.safe_dump(public_config, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_plan(config: dict[str, Any]) -> ExperimentPlan:
    """Create the Cartesian grid while preserving paired repetition seeds."""
    # Cartesian grid construction
    ## Each tuple selects one value for every dotted grid path
    grid = config.get("grid", {})
    references = sorted(_grid_references(config["task"].get("parameters", {})))
    missing = set(references).difference(grid)
    if missing:
        raise ValueError(f"Grid references are undefined: {sorted(missing)}")
    values_by_name = {}
    for name in references:
        definition = grid[name]
        values = definition.get("values") if isinstance(definition, dict) else None
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid '{name}' requires a non-empty values list.")
        values_by_name[name] = values
    combinations = product(*(values_by_name[name] for name in references)) if references else [()]
    repetitions = config.get("runs", {}).get("repetitions", 1)
    seeds = config["seeds"]
    common_seeds = deepcopy(seeds["common_seeds"])
    run_seeds = seeds["run_seeds"]
    tasks: list[PlannedTask] = []

    # Task expansion
    ## Evaluate every grid cell with every independently seeded repetition
    for grid_index, values in enumerate(combinations):
        assignments = dict(zip(references, values))
        for repetition in range(repetitions):
            ### Resolve named grid values at their declaration sites
            parameters = _replace_grid_references(config["task"].get("parameters", {}), assignments)
            ### Keep data sampling fixed while varying model and training random streams
            derived_run_seeds = {
                name: repetition_seed(seed, repetition, name)
                for name, seed in run_seeds.items()
            }
            reproducibility = {
                "common_seeds": deepcopy(common_seeds),
                "run_seeds": deepcopy(run_seeds),
                "derived_run_seeds": derived_run_seeds,
            }
            task_index = len(tasks)
            tasks.append(
                PlannedTask(
                    task_id=f"task_{task_index:06d}",
                    grid_id=f"grid_{grid_index:04d}",
                    repetition=repetition,
                    reproducibility=reproducibility,
                    grid_parameters=assignments,
                    entrypoint=config["task"]["entrypoint"],
                    parameters=parameters,
                )
            )
    return ExperimentPlan(
        experiment_name=config["experiment"]["name"],
        config_path=config["_config_path"],
        config_fingerprint=configuration_fingerprint(config),
        tasks=tuple(tasks),
        execution=deepcopy(config.get("execution", {"backend": "local"})),
        reports=tuple(config.get("reports", [])),
    )


def materialize_plan(plan: ExperimentPlan, directory: Path) -> Path:
    """Write resolved task descriptors before any external work starts."""
    # Per-task descriptors
    ## Backends consume these files without expanding the original grid again
    directory.mkdir(parents=True, exist_ok=True)
    tasks_directory = directory / "tasks"
    tasks_directory.mkdir(exist_ok=True)
    for task in plan.tasks:
        with (tasks_directory / f"{task.task_id}.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(asdict(task), stream, sort_keys=False)
    # Campaign manifest
    ## Keep one aggregate record for inspection and result analysis
    plan_path = directory / "resolved-experiment.yaml"
    payload = {
        "runtime_schema_version": 1,
        "experiment_name": plan.experiment_name,
        "config_path": plan.config_path,
        "config_fingerprint": plan.config_fingerprint,
        "execution": plan.execution,
        "reports": list(plan.reports),
        "tasks": [asdict(task) for task in plan.tasks],
    }
    with plan_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    return plan_path
