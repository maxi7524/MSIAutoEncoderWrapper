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
    workflow: dict[str, Any] | None = None
    depends_on: tuple[str, ...] = ()


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
        if set(value) in ({"grid"}, {"grid", "select"}):
            reference = value["grid"]
            if not isinstance(reference, str) or not reference:
                raise ValueError("A grid reference must contain a non-empty grid name.")
            selection = value.get("select")
            if selection is not None and (
                not isinstance(selection, str) or not selection
            ):
                raise ValueError("A grid selection must contain a non-empty field name.")
            return {reference}
        return set().union(*(_grid_references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_grid_references(item) for item in value))
    return set()


def _replace_grid_references(
    value: Any,
    assignments: dict[str, Any],
    resolving: frozenset[str] = frozenset(),
) -> Any:
    """Replace grid nodes, including references nested in selected values."""
    if isinstance(value, dict):
        if set(value) in ({"grid"}, {"grid", "select"}):
            grid_name = value["grid"]
            if grid_name in resolving:
                raise ValueError(
                    f"Grid values contain a recursive reference to '{grid_name}'."
                )
            selected = _replace_grid_references(
                deepcopy(assignments[grid_name]),
                assignments,
                resolving | {grid_name},
            )
            field = value.get("select")
            if field is None:
                return selected
            if not isinstance(selected, dict) or field not in selected:
                raise ValueError(
                    f"Grid '{grid_name}' does not provide selected field '{field}'."
                )
            return deepcopy(selected[field])
        return {
            key: _replace_grid_references(item, assignments, resolving)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_grid_references(item, assignments, resolving) for item in value
        ]
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


def _workflow_roles(
    parameters: dict[str, Any],
    workflow: dict[str, Any] | None,
) -> tuple[tuple[str | None, dict[str, Any]], ...]:
    """Split one grid run into independently persisted workflow roles.

    :param parameters: Grid-resolved task parameters.
    :type parameters: dict[str, typing.Any]
    :param workflow: Optional workflow declaration from ``task.workflow``.
    :type workflow: dict[str, typing.Any] | None
    :return: Ordered ``(role, parameters)`` task variants.
    :rtype: tuple[tuple[str | None, dict[str, typing.Any]], ...]
    :raises ValueError: If a phase-role workflow is incomplete or malformed.
    """
    if workflow is None:
        return ((None, parameters),)
    if workflow.get("strategy") != "pretraining_branches":
        raise ValueError("task.workflow.strategy must be 'pretraining_branches'.")

    phases = parameters.get("training", {}).get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("A pretraining branch workflow requires training phases.")
    phases_by_role: dict[str, list[dict[str, Any]]] = {}
    role_order: list[str] = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("Every workflow training phase must be a mapping.")
        role = phase.get("workflow_role")
        if role not in {"real_only", "pretrained", "frozen_head", "unfrozen_head"}:
            raise ValueError(
                "Every workflow phase requires workflow_role equal to real_only, "
                "pretrained, frozen_head or unfrozen_head."
            )
        if role not in phases_by_role:
            phases_by_role[role] = []
            role_order.append(role)
        phases_by_role[role].append(deepcopy(phase))

    observed_roles = set(phases_by_role)
    if observed_roles == {"real_only"}:
        expected_order = ["real_only"]
    elif observed_roles == {"pretrained", "frozen_head", "unfrozen_head"}:
        expected_order = ["pretrained", "frozen_head", "unfrozen_head"]
    else:
        raise ValueError(
            "A workflow grid run must contain either real_only or exactly "
            "pretrained, frozen_head and unfrozen_head phases."
        )
    if role_order != expected_order:
        raise ValueError(
            f"Workflow roles must appear in order {expected_order}, got {role_order}."
        )

    variants = []
    for role in expected_order:
        role_parameters = deepcopy(parameters)
        role_parameters["training"]["phases"] = phases_by_role[role]
        variants.append((role, role_parameters))
    return tuple(variants)


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
    workflow = config["task"].get("workflow")

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
            grid_id = f"grid_{grid_index:04d}"
            group_id = f"{grid_id}__rep_{repetition:02d}"
            variants = _workflow_roles(parameters, workflow)
            role_task_ids = {
                role: f"task_{len(tasks) + offset:06d}"
                for offset, (role, _parameters) in enumerate(variants)
            }
            for role, role_parameters in variants:
                parent_role = (
                    "pretrained" if role in {"frozen_head", "unfrozen_head"} else None
                )
                parent_task_id = (
                    role_task_ids[parent_role] if parent_role is not None else None
                )
                tasks.append(
                    PlannedTask(
                        task_id=role_task_ids[role],
                        grid_id=grid_id,
                        repetition=repetition,
                        reproducibility=deepcopy(reproducibility),
                        grid_parameters=deepcopy(assignments),
                        entrypoint=config["task"]["entrypoint"],
                        parameters=role_parameters,
                        workflow=(
                            {
                                "group_id": group_id,
                                "role": role,
                                "parent_task_id": parent_task_id,
                            }
                            if role is not None
                            else None
                        ),
                        depends_on=(
                            (parent_task_id,) if parent_task_id is not None else ()
                        ),
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
    """Write a compact graph index with bounded additional serialization memory.

    :param plan: Resolved experiment plan.
    :type plan: ExperimentPlan
    :param directory: Output directory for descriptors and the graph index.
    :type directory: pathlib.Path
    :return: Path to the atomically published campaign index.
    :rtype: pathlib.Path
    """
    # Per-task descriptors
    ## Serialize one task at a time; shared source populations remain references.
    directory.mkdir(parents=True, exist_ok=True)
    tasks_directory = directory / "tasks"
    tasks_directory.mkdir(exist_ok=True)
    for task in plan.tasks:
        path = tasks_directory / f"{task.task_id}.yaml"
        temporary = path.with_suffix(".yaml.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(asdict(task), stream, sort_keys=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    # Campaign graph index
    ## List identity and lineage only; descriptors own task-specific configuration.
    plan_path = directory / "resolved-experiment.yaml"
    manifest = {
        "runtime_schema_version": 3,
        "experiment_name": plan.experiment_name,
        "config_path": plan.config_path,
        "config_fingerprint": plan.config_fingerprint,
        "execution": plan.execution,
        "reports": list(plan.reports),
    }
    temporary_path = plan_path.with_suffix(".yaml.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)
            stream.write("tasks:\n")
            for task in plan.tasks:
                yaml.safe_dump(
                    [{
                        "task_id": task.task_id,
                        "grid_id": task.grid_id,
                        "repetition": task.repetition,
                        "workflow": task.workflow,
                        "depends_on": list(task.depends_on),
                    }],
                    stream,
                    sort_keys=False,
                )
        temporary_path.replace(plan_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return plan_path
