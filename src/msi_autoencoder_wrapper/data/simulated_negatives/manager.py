"""Registry and declarative loader for simulated-negative strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Type

from ...utils.exceptions import raise_validation_error
from ...utils.module_search import discover_modules
from ...utils.validators import resolve_component, validate_subclass
from .base import SimulatedNegativeStrategy


class SimulatedNegativeManager:
    """Load registered reliable-negative sources from portable descriptors."""

    _REGISTRY: dict[str, Type[SimulatedNegativeStrategy]] = {}

    @classmethod
    def register_strategy(cls, name: str) -> Any:
        """Register one concrete reliable-negative strategy.

        :param name: Stable configuration token.
        :type name: str
        :return: Registration decorator.
        :rtype: Any
        """
        def decorator(
            subclass: Type[SimulatedNegativeStrategy],
        ) -> Type[SimulatedNegativeStrategy]:
            validate_subclass(
                subclass,
                SimulatedNegativeStrategy,
                "SimulatedNegativeRegistry",
            )
            cls._REGISTRY[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_strategy(cls, name: Any, **parameters: Any) -> SimulatedNegativeStrategy:
        """Instantiate one registered strategy.

        :param name: Registry token, strategy class, or existing strategy.
        :type name: Any
        :param parameters: Constructor parameters for a class-based strategy.
        :type parameters: Any
        :return: Resolved strategy instance.
        :rtype: SimulatedNegativeStrategy
        """
        cls.discover_strategies()
        return resolve_component(
            target=name,
            registry=cls._REGISTRY,
            component_type="SimulatedNegativeStrategy",
            expected_type=SimulatedNegativeStrategy,
            **parameters,
        )

    @classmethod
    def load_config(
        cls,
        config: Mapping[str, Any],
    ) -> SimulatedNegativeStrategy:
        """Load one strategy from a ``type``/``parameters`` descriptor.

        :param config: Portable strategy descriptor.
        :type config: Mapping[str, Any]
        :return: Resolved strategy instance.
        :rtype: SimulatedNegativeStrategy
        :raises ValidationError: If the descriptor is malformed.
        """
        if not isinstance(config, Mapping):
            raise_validation_error(
                "SimulatedNegativeConfiguration",
                "simulated_negative must be a mapping.",
            )
        strategy_type = config.get("type")
        if not isinstance(strategy_type, str) or not strategy_type:
            raise_validation_error(
                "SimulatedNegativeConfiguration",
                "simulated_negative.type must be a nonempty string.",
            )
        parameters = config.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise_validation_error(
                "SimulatedNegativeConfiguration",
                "simulated_negative.parameters must be a mapping.",
            )
        return cls.get_strategy(strategy_type, **dict(parameters))

    @classmethod
    def discover_strategies(cls) -> None:
        """Import bundled strategy modules and trigger registrations."""
        discover_modules(f"{__package__}.strategies")
