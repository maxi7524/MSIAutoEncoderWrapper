"""Registry and factory for annotation-reader strategies."""

from __future__ import annotations

from typing import Any, Dict, Type

from ..utils.module_search import discover_modules
from ..utils.validators import resolve_component, validate_subclass
from .base_annotation_reader import MSIBaseAnnotationReader


class AnnotationReaderManager:
    """Register provider-independent annotation readers."""

    REGISTRY: Dict[str, Type[MSIBaseAnnotationReader]] = {}

    @classmethod
    def register_reader(cls, name: str) -> Any:
        """Return a decorator registering an annotation-reader strategy."""
        def decorator(
            subclass: Type[MSIBaseAnnotationReader],
        ) -> Type[MSIBaseAnnotationReader]:
            validate_subclass(subclass, MSIBaseAnnotationReader, "AnnotationReaderRegistry")
            cls.REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_reader(cls, name: Any, **kwargs: Any) -> MSIBaseAnnotationReader:
        """Resolve a registered key, reader class, or ready reader instance."""
        return resolve_component(
            target=name,
            registry=cls.REGISTRY,
            component_type="AnnotationReader",
            expected_type=MSIBaseAnnotationReader,
            **kwargs,
        )

    @classmethod
    def discover_strategies(cls) -> None:
        """Import bundled annotation-reader strategies."""
        discover_modules(f"{__package__}.strategies")
