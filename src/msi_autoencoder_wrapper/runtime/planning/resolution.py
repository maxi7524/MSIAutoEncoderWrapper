"""Materialize runtime-dependent experiment blueprints before training."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..workflows.entrypoints import resolve_entrypoint
from .plan import ExperimentPlan


def resolve_plan(plan: ExperimentPlan, directory: Path, resolver: str | None) -> ExperimentPlan:
    """Resolve shared artifacts and return tasks referencing those artifacts.

    :param plan: Grid-expanded experiment plan.
    :type plan: ExperimentPlan
    :param directory: Persistent plan output directory.
    :type directory: pathlib.Path
    :param resolver: Optional ``module:function`` plan resolver.
    :type resolver: str | None
    :return: Plan whose tasks contain resolved artifact references.
    :rtype: ExperimentPlan
    """
    if resolver is None:
        return plan
    resolved_tasks = resolve_entrypoint(resolver)(
        [asdict(task) for task in plan.tasks],
        directory.resolve(),
    )
    if not isinstance(resolved_tasks, list) or len(resolved_tasks) != len(plan.tasks):
        raise ValueError("Plan resolver must return one parameter mapping per task")
    tasks = tuple(
        replace(task, parameters=parameters)
        for task, parameters in zip(plan.tasks, resolved_tasks)
    )
    return replace(plan, tasks=tasks)
