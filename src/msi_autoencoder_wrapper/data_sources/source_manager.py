"""Registry and factory for external MSI data sources."""

from __future__ import annotations

from typing import Any, Dict, Type

from ..utils.module_search import discover_modules
from ..utils.validators import resolve_component, validate_subclass
from .base_source import DatasetSource


class DatasetSourceManager:
    """Register and instantiate external data-source adapters."""

    REGISTRY: Dict[str, Type[DatasetSource]] = {}

    @classmethod
    def register_source(cls, name: str) -> Any:
        """Return a decorator registering a data-source strategy."""
        def decorator(subclass: Type[DatasetSource]) -> Type[DatasetSource]:
            validate_subclass(subclass, DatasetSource, "DatasetSourceRegistry")
            cls.REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_source(cls, name: Any, **kwargs: Any) -> DatasetSource:
        """Resolve a registered key, implementation class, or ready instance."""
        return resolve_component(
            target=name,
            registry=cls.REGISTRY,
            component_type="DatasetSource",
            expected_type=DatasetSource,
            **kwargs,
        )

    @classmethod
    def discover_strategies(cls) -> None:
        """Import bundled source strategies."""
        discover_modules(f"{__package__}.strategies")
