"""Registry and factory for annotation-reader strategies."""

from __future__ import annotations

from typing import Any, Dict, Type

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
    def get_reader(cls, name: Any = None, **kwargs: Any) -> MSIBaseAnnotationReader:
        """Resolve a registered key, reader class, or ready reader instance."""
        cls.load_builtin_readers()
        target = name or "SQLiteAnnotationReader"
        return resolve_component(
            target=target,
            registry=cls.REGISTRY,
            component_type="AnnotationReader",
            expected_type=MSIBaseAnnotationReader,
            **kwargs,
        )

    @classmethod
    def load_builtin_readers(cls) -> None:
        """Import built-in SQLite and METASPACE CSV annotation readers."""
        from .strategies import (
            metaspace_csv_annotation_reader,
            sqlite_annotation_reader,
        )

        del metaspace_csv_annotation_reader
        del sqlite_annotation_reader
