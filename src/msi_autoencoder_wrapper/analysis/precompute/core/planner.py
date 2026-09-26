"""Validation of explicit precompute-stage order and artifact contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .context import AnalysisContext
from .contracts import AnalysisPlugin, PrecomputeStrategy


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated, ordered plugins that should run for one settings file."""

    stages: tuple[AnalysisPlugin, ...]


def build_plan(strategy: PrecomputeStrategy, context: AnalysisContext) -> ExecutionPlan:
    """Validate a strategy's declared order without reordering it.

    :param strategy: Complete workflow selected from the strategy registry.
    :type strategy: PrecomputeStrategy
    :param context: Resolved runtime context.
    :type context: AnalysisContext
    :return: Enabled stages in their declared order.
    :rtype: ExecutionPlan
    :raises ValueError: If a dependency is unavailable or two stages provide the same
        logical artifact.
    """
    available = {"model_catalog"}
    providers: set[str] = set()
    enabled: list[AnalysisPlugin] = []
    for plugin in strategy.stages:
        if not plugin.is_enabled(context):
            continue
        missing = sorted(set(plugin.requires) - available)
        if missing:
            raise ValueError(
                f"Strategy '{strategy.name}' runs '{plugin.name}' before its required "
                f"artifact(s): {missing}."
            )
        names = [spec.name for spec in plugin.provides]
        duplicate = sorted(set(names) & providers)
        if duplicate:
            raise ValueError(
                f"Strategy '{strategy.name}' has multiple providers for artifact(s): {duplicate}."
            )
        providers.update(names)
        available.update(names)
        enabled.append(plugin)
    return ExecutionPlan(tuple(enabled))


def restrict_plan(plan: ExecutionPlan, analyses: Sequence[str]) -> ExecutionPlan:
    """Keep only the stages producing the requested analyses and their dependencies.

    :param plan: Validated plan returned by :func:`build_plan`.
    :type plan: ExecutionPlan
    :param analyses: Analysis keys (``ArtifactSpec.analysis_name``) to produce.
    :type analyses: collections.abc.Sequence[str]
    :return: Plan with the original stage order, reduced to the required stages.
    :rtype: ExecutionPlan
    :raises ValueError: If a requested analysis is not produced by any enabled stage.
    """
    requested = set(analyses)
    produced = {spec.analysis_name for stage in plan.stages for spec in stage.provides}
    unknown = sorted(requested - produced)
    if unknown:
        raise ValueError(f"No enabled stage produces the requested analysis(es): {unknown}.")
    providers = {spec.name: stage for stage in plan.stages for spec in stage.provides}
    frontier = [stage for stage in plan.stages
                if any(spec.analysis_name in requested for spec in stage.provides)]
    required: set[str] = set()
    while frontier:
        stage = frontier.pop()
        if stage.name in required:
            continue
        required.add(stage.name)
        frontier.extend(providers[name] for name in stage.requires if name in providers)
    return ExecutionPlan(tuple(stage for stage in plan.stages if stage.name in required))
