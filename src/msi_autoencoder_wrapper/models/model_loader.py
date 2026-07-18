"""Reconstruct configured model architectures from portable JSON documents."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .architectures.architectures_manager import ArchitecturesManager
from ..utils.exceptions import raise_model_initialization_error
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ModelLoader:
    """Build model instances from the loaded-model section of saved configuration."""

    @classmethod
    def build(cls, config: Dict[str, Any]) -> tuple[nn.Module, str, str | None]:
        """Build an uninitialized-weight model from a portable configuration.

        :param config: Complete saved configuration or loaded-model subsection.
        :type config: Dict[str, Any]
        :return: Model instance, model family, and optional model name.
        :rtype: tuple[torch.nn.Module, str, Optional[str]]
        :raises ModelNotInitializedError: If the architecture configuration is incomplete.
        """
        loaded_context = config.get("loaded_model_context", config)
        model_config = loaded_context.get("model")
        if not isinstance(model_config, dict):
            raise_model_initialization_error(
                model_name="LoadedModel",
                message="Saved configuration does not contain a loaded-model definition.",
            )

        model_type = model_config.get("type")
        if not isinstance(model_type, str) or not model_type:
            raise_model_initialization_error(
                model_name="LoadedModel",
                message="Saved model family is missing.",
            )

        component_config = model_config.get("components") or cls._runtime_components(
            model_config
        )
        if not isinstance(component_config, dict) or not component_config:
            raise_model_initialization_error(
                model_name=model_type,
                message="Saved model does not contain reconstructable components.",
            )

        ArchitecturesManager.discover_architectures()
        components_setup = {
            category: cls._component_setup(descriptor)
            for category, descriptor in component_config.items()
        }
        model_parameters = model_config.get("parameters", {})
        if not isinstance(model_parameters, dict):
            raise_model_initialization_error(
                model_name=model_type,
                message="Saved model parameters must be a dictionary.",
            )

        logger.info("Reconstructing loaded model family '%s'.", model_type)
        model = ArchitecturesManager.build_model(
            model_type=model_type,
            components_setup=components_setup,
            **model_parameters,
        )
        return model, model_type, model_config.get("name")

    @staticmethod
    def _runtime_components(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract component descriptors from the runtime fallback configuration."""
        runtime = model_config.get("runtime", {})
        runtime_parameters = runtime.get("parameters", {}) if isinstance(runtime, dict) else {}
        components = runtime_parameters.get("components", {})
        return components if isinstance(components, dict) else {}

    @classmethod
    def _component_setup(cls, descriptor: Any) -> Dict[str, Any]:
        """Convert a portable descriptor into an architecture-manager setup node."""
        if not isinstance(descriptor, dict):
            raise_model_initialization_error(
                model_name="LoadedModel",
                message="A saved component descriptor must be a dictionary.",
            )
        if "type" in descriptor:
            parameters = descriptor.get("parameters", {})
            if not isinstance(parameters, dict):
                raise_model_initialization_error(
                    model_name=str(descriptor.get("type")),
                    message="Saved component parameters must be a dictionary.",
                )
            return {"strategy": descriptor["type"], "params": parameters}
        return {
            child_name: cls._component_setup(child_descriptor)
            for child_name, child_descriptor in descriptor.items()
        }
