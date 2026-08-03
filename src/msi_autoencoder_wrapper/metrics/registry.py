"""Registry and name-based execution for model-independent metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..utils.exceptions import raise_validation_error

MetricSpace = Literal["spectrum", "classification", "class", "embedding"]


@dataclass(frozen=True)
class MetricDefinition:
    """Describe one registered metric and the space containing its objects."""

    name: str
    space: MetricSpace
    implementation: Callable[..., Any]


class MetricsRegistry:
    """Resolve metrics by stable names without importing training components."""

    _registry: dict[MetricSpace, dict[str, MetricDefinition]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        space: MetricSpace,
        implementation: Callable[..., Any],
    ) -> None:
        """Register one implementation under an object-space namespace."""
        definitions = cls._registry.setdefault(space, {})
        if name in definitions and definitions[name].implementation is not implementation:
            raise_validation_error(
                "MetricsRegistry", f"Metric '{space}.{name}' is already registered."
            )
        definitions[name] = MetricDefinition(name, space, implementation)

    @classmethod
    def available(cls, space: MetricSpace | None = None) -> dict[str, Any]:
        """Return registered definitions for one space or every space."""
        if space is not None:
            return dict(cls._registry.get(space, {}))
        return {
            registered_space: dict(definitions)
            for registered_space, definitions in cls._registry.items()
        }

    @classmethod
    def resolve(cls, name: str, space: MetricSpace) -> MetricDefinition:
        """Return one definition or raise a standardized validation error."""
        definition = cls._registry.get(space, {}).get(name)
        if definition is None:
            raise_validation_error(
                "MetricsRegistry", f"Unknown metric '{space}.{name}'."
            )
        return definition


class MetricsRunner:
    """Execute registered stateless functions or configurable metric classes."""

    @staticmethod
    def compute(
        name: str,
        space: MetricSpace,
        *args: Any,
        metric_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Resolve and execute one registered metric."""
        implementation = MetricsRegistry.resolve(name, space).implementation
        metric = (
            implementation(**dict(metric_params or {}))
            if isinstance(implementation, type)
            else implementation
        )
        return metric(*args, **kwargs)
