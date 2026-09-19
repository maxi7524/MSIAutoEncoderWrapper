"""Contracts and registration for synthetic spectrum renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, TYPE_CHECKING, Type

import numpy as np

from ....utils.module_search import discover_modules
from ....utils.printing import extract_component_signatures
from ....utils.validators import resolve_component, validate_subclass

if TYPE_CHECKING:
    from ..sampling import SyntheticSampleDefinition
    from ..sources import SyntheticPeakSource


@dataclass(frozen=True)
class SyntheticRepresentationSpec:
    """Select one registered renderer and its immutable configuration."""

    strategy: str = "triangular_peak"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("representation strategy must be a nonempty string.")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("representation parameters must be a mapping.")
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_value(
        cls,
        value: "SyntheticRepresentationSpec | Mapping[str, Any] | None",
    ) -> "SyntheticRepresentationSpec":
        """Normalize a YAML representation declaration."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("representation must be a mapping.")
        raw = dict(value)
        strategy = raw.pop("strategy", None)
        parameters = raw.pop("parameters", {})
        if raw:
            raise ValueError(f"Unsupported representation keys: {sorted(raw)}.")
        return cls(strategy=strategy, parameters=parameters)


@dataclass(frozen=True)
class SyntheticRepresentationContext:
    """Static axis and source information available to every renderer."""

    source: "SyntheticPeakSource"
    feature_count: int
    mass_axis: np.ndarray | None = None
    mass_to_bin: Callable[[np.ndarray], np.ndarray] | None = None


class SyntheticRepresentationStrategy(ABC):
    """Convert a declared synthetic composition into a dense spectrum."""

    @abstractmethod
    def render(
        self,
        rng: np.random.Generator,
        context: SyntheticRepresentationContext,
        definition: "SyntheticSampleDefinition",
    ) -> np.ndarray:
        """Return a nonnegative dense spectrum of shape ``(M,)``."""


class SyntheticRepresentationManager:
    """Register, discover, document, and resolve spectrum renderers."""

    _REGISTRY: Dict[str, Type[SyntheticRepresentationStrategy]] = {}

    @classmethod
    def register_strategy(cls, name: str):
        """Register one renderer under its stable configuration name."""
        if not isinstance(name, str) or not name:
            raise ValueError("Synthetic representation strategy names must be nonempty strings.")

        def decorator(
            subclass: Type[SyntheticRepresentationStrategy],
        ) -> Type[SyntheticRepresentationStrategy]:
            validate_subclass(
                subclass,
                SyntheticRepresentationStrategy,
                "SyntheticRepresentationRegistry",
            )
            cls._REGISTRY[name] = subclass
            return subclass

        return decorator

    @classmethod
    def discover_strategies(cls) -> None:
        """Import renderer modules so registration decorators execute."""
        discover_modules(__package__)

    @classmethod
    def get_strategy(cls, name: Any, **kwargs: Any) -> SyntheticRepresentationStrategy:
        """Instantiate one registered renderer."""
        cls.discover_strategies()
        return resolve_component(
            target=name,
            registry=cls._REGISTRY,
            component_type="Synthetic representation strategy",
            expected_type=SyntheticRepresentationStrategy,
            **kwargs,
        )

    @classmethod
    def get_available_strategies(cls) -> Dict[str, Dict[str, Any]]:
        """Return renderer docstrings and constructor parameters."""
        cls.discover_strategies()
        return extract_component_signatures(cls._REGISTRY)


def register_representation_strategy(name: str):
    """Return the standard decorator for one spectrum renderer."""
    return SyntheticRepresentationManager.register_strategy(name)


def get_representation_strategy(name: Any, **kwargs: Any) -> SyntheticRepresentationStrategy:
    """Resolve one registered spectrum renderer."""
    return SyntheticRepresentationManager.get_strategy(name, **kwargs)
