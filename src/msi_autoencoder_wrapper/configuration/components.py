"""Portable configuration contracts shared by library components."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ..utils.exceptions import raise_project_config_error


class ConfigurableComponent:
    """Let each component export the parameters required to reconstruct itself."""

    _config: Dict[str, Any]
    config_version: int = 1

    def get_config(self) -> Dict[str, Any]:
        """Return an isolated snapshot of constructor parameters.

        :return: Copy of the parameters required to reproduce the component.
        :rtype: Dict[str, Any]
        """
        return copy.deepcopy(self._config)

    def export_config(self) -> Dict[str, Any]:
        """Export this component's complete recursive configuration node."""
        return {
            "type": type(self).__name__,
            "module": type(self).__module__,
            "version": int(self.config_version),
            "parameters": make_json_compatible(self.get_config()),
        }

    @classmethod
    def from_config(cls, parameters: Mapping[str, Any], **dependencies: Any) -> Any:
        """Reconstruct this component using explicit runtime dependencies."""
        return cls(**dict(parameters), **dependencies)


def make_json_compatible(value: Any, path: str = "config") -> Any:
    """Convert supported configuration values into JSON-compatible values.

    :param value: Configuration value to normalize.
    :type value: Any
    :param path: Human-readable location used in validation errors.
    :type path: str
    :return: JSON-compatible representation of ``value``.
    :rtype: Any
    :raises ProjectConfigError: If a value cannot be represented safely in JSON.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): make_json_compatible(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            make_json_compatible(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if inspect.isclass(value):
        return f"{value.__module__}.{value.__qualname__}"

    config_getter = getattr(value, "get_config", None)
    if callable(config_getter):
        return get_component_config(value)

    scalar_converter = getattr(value, "item", None)
    if callable(scalar_converter):
        try:
            scalar_value = scalar_converter()
        except (RuntimeError, TypeError, ValueError):
            scalar_value = value
        if scalar_value is not value:
            return make_json_compatible(scalar_value, path)

    raise_project_config_error(
        context_name="Configuration",
        message=(
            f"Value at '{path}' with type '{type(value).__name__}' cannot be "
            "serialized to the portable JSON configuration."
        ),
    )


def get_component_config(component: Any) -> Dict[str, Any]:
    """Describe an initialized functional component and its parameters.

    :param component: Initialized reader, binner, dataset, model, or training component.
    :type component: Any
    :return: Component class identity and JSON-compatible parameters.
    :rtype: Dict[str, Any]
    :raises ProjectConfigError: If the component does not expose a configuration dictionary.
    """
    exporter = getattr(component, "export_config", None)
    if callable(exporter):
        descriptor = exporter()
        if not isinstance(descriptor, dict):
            raise_project_config_error(
                context_name="Configuration",
                message=f"Component '{type(component).__name__}' returned an invalid node.",
            )
        return make_json_compatible(descriptor)

    getter = getattr(component, "get_config", None)
    if not callable(getter):
        raise_project_config_error(
            context_name="Configuration",
            message=(
                f"Component '{type(component).__name__}' does not expose get_config()."
            ),
        )

    parameters = getter()
    if not isinstance(parameters, dict):
        raise_project_config_error(
            context_name="Configuration",
            message=(
                f"Component '{type(component).__name__}' returned a non-dictionary configuration."
            ),
        )

    component_class = type(component)
    return {
        "type": component_class.__name__,
        "module": component_class.__module__,
        "version": 1,
        "parameters": make_json_compatible(parameters),
    }


def describe_component_target(target: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Describe a configured registry key, class, or ready component instance.

    :param target: Registry key, component class, or initialized instance.
    :type target: Any
    :param parameters: Constructor parameters buffered for a key or class target.
    :type parameters: Dict[str, Any]
    :return: Portable component descriptor.
    :rtype: Dict[str, Any]
    """
    if isinstance(target, str):
        return {
            "type": target,
            "version": 1,
            "parameters": make_json_compatible(parameters),
        }
    if inspect.isclass(target):
        return {
            "type": target.__name__,
            "module": target.__module__,
            "version": 1,
            "parameters": make_json_compatible(parameters),
        }
    return get_component_config(target)
