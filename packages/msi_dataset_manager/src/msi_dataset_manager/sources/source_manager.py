"""Registry and factory for external MSI dataset sources."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Type
from .base import DatasetSource


class DatasetSourceManager:
    """Register and instantiate external data-source adapters."""

    REGISTRY: Dict[str, Type[DatasetSource]] = {}

    @classmethod
    def register_source(cls, name: str) -> Any:
        """Return a decorator registering a data-source strategy."""
        def decorator(subclass: Type[DatasetSource]) -> Type[DatasetSource]:
            if not issubclass(subclass, DatasetSource):
                raise TypeError(f"{subclass!r} must inherit DatasetSource")
            cls.REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_source(cls, name: Any, **kwargs: Any) -> DatasetSource:
        """Resolve a registered key, implementation class, or ready instance."""
        if isinstance(name, DatasetSource):
            return name
        if isinstance(name, type) and issubclass(name, DatasetSource):
            return name(**kwargs)
        try:
            implementation = cls.REGISTRY[str(name)]
        except KeyError as error:
            raise ValueError(f"Unknown DatasetSource: {name!r}") from error
        return implementation(**kwargs)

    @classmethod
    def discover_strategies(cls) -> None:
        """Import bundled source strategies."""
        package = importlib.import_module(f"{__package__}.strategies")
        for module in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
            importlib.import_module(module.name)
