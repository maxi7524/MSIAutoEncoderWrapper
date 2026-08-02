"""Registry for module-owned component configuration loaders."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from ..utils.exceptions import raise_validation_error


ComponentLoader = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]


class ConfigurationRegistry:
    """Dispatch component nodes without encoding their parameters centrally."""

    _loaders: Dict[str, ComponentLoader] = {}

    @classmethod
    def register(cls, component_type: str) -> Callable[[ComponentLoader], ComponentLoader]:
        """Register one module-owned loader by portable component type."""
        def decorator(loader: ComponentLoader) -> ComponentLoader:
            cls._loaders[component_type] = loader
            return loader
        return decorator

    @classmethod
    def load(
        cls,
        node: Mapping[str, Any],
        *,
        dependencies: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Load one component node through its owning module."""
        component_type = node.get("type")
        if not isinstance(component_type, str) or not component_type:
            raise_validation_error("Configuration", "A component type is required.")
        loader = cls._loaders.get(component_type)
        if loader is None:
            raise_validation_error(
                "Configuration", f"No loader is registered for '{component_type}'."
            )
        parameters = node.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise_validation_error(
                "Configuration", f"Parameters for '{component_type}' must be a mapping."
            )
        return loader(parameters, dict(dependencies or {}))

