"""Coordinate independent configuration loaders without owning their details."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..utils.exceptions import raise_validation_error
from ..utils.logger import get_custom_logger
from .schema import validate_consolidated_configuration

logger = get_custom_logger(__name__)


class ConfigurationOrchestrator:
    """Restore a saved experiment through module-owned configuration loaders.

    :param wrapper: Wrapper receiving the restored runtime state.
    :type wrapper: Any
    """

    def __init__(self, wrapper: Any) -> None:
        self._wrapper = wrapper

    def load(
        self,
        model_name: str,
        image_name: Optional[str] = None,
        apply: bool = True,
        load_model: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Read and optionally restore a complete saved configuration.

        :param model_name: Saved model identifier.
        :type model_name: str
        :param image_name: Saved image context. Uses the workspace default when omitted.
        :type image_name: str | None
        :param apply: Apply configuration sections when true.
        :type apply: bool
        :param load_model: Restore model weights when applying the configuration.
        :type load_model: bool
        :param strict: Enforce exact model weight compatibility.
        :type strict: bool
        :return: The loaded consolidated configuration dictionary.
        :rtype: Dict[str, Any]
        :raises ValidationError: If no image context can be resolved.
        """
        workspace = self._wrapper.workspace
        resolved_image = image_name or workspace.default_img_name
        if not resolved_image:
            raise_validation_error(
                "Configuration",
                "Set a default image or pass image_name before loading configuration.",
            )
        config = workspace.load_config_json(resolved_image, model_name)
        validate_consolidated_configuration(config)
        if not apply:
            return config

        logger.info(
            "Restoring configuration for model '%s' and image '%s'.",
            model_name,
            resolved_image,
        )
        context_target = image_name
        self._wrapper.context_manager.load_context_config(
            config["local_image_context"],
            img_name_or_path=context_target,
        )
        loaded_context = config["loaded_model_context"]
        dataset_config = loaded_context.get("dataset")
        if dataset_config is not None:
            self._wrapper.active_dataset = (
                self._wrapper.models_manager.load_dataset_config(dataset_config)
            )
        if load_model:
            self._wrapper.models_manager.load_model(
                img_name=resolved_image,
                model_name=model_name,
                strict=strict,
            )
        logger.info("Saved configuration restoration completed.")
        return config
