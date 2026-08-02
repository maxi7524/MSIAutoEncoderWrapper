"""Portable configuration loading and orchestration."""

from typing import Any

from .components import (
    ConfigurableComponent,
    describe_component_target,
    get_component_config,
    make_json_compatible,
)

__all__ = [
    "ConfigurableComponent",
    "ConfigurationOrchestrator",
    "describe_component_target",
    "get_component_config",
    "make_json_compatible",
]


def __getattr__(name: str) -> Any:
    """Keep the component contract lightweight until orchestration is requested."""
    if name == "ConfigurationOrchestrator":
        from .orchestrator import ConfigurationOrchestrator

        return ConfigurationOrchestrator
    raise AttributeError(name)
