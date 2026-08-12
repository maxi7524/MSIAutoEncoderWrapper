"""Minimal serializable component contract for provider adapters."""

from typing import Any, Dict


class ConfigurableComponent:
    """Expose constructor configuration without model-system dependencies."""

    _config: Dict[str, Any]

    def get_config(self) -> Dict[str, Any]:
        """Return a shallow copy of serializable component configuration."""
        return dict(self._config)
