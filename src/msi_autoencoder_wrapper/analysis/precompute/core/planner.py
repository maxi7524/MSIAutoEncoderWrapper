"""Validation of explicit precompute-stage order and artifact contracts."""

from __future__ import annotations

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
