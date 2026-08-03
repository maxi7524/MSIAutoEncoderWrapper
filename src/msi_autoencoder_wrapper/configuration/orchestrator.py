"""Coordinate independent configuration loaders without owning their details."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..utils.exceptions import raise_validation_error
from ..utils.logger import get_custom_logger
from .schema import validate_consolidated_configuration
from ..models.model_loader import ModelLoader

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
            raise_validation_error("Configuration", "Pass an experiment path instead.")
        config = workspace.load_config_json(resolved_image, model_name)
        validate_consolidated_configuration(config)
        if not apply:
            return config

        logger.info(
            "Restoring configuration for model '%s' and image '%s'.",
            model_name,
            resolved_image,
        )
        self._apply(config, load_model=load_model, strict=strict)
        if load_model:
            self._wrapper.models_manager.load_model(
                img_name=resolved_image,
                model_name=model_name,
                strict=strict,
            )
        logger.info("Saved configuration restoration completed.")
        return config

    def load_experiment(
        self,
        path: Path | str,
        *,
        load_model: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load one complete schema-v2 experiment from its model directory."""
        model_dir = Path(path)
        if not model_dir.is_absolute():
            workspace_candidate = self._wrapper.workspace.get_models_root() / model_dir
            model_dir = workspace_candidate if workspace_candidate.is_dir() else Path.cwd() / model_dir
        config_path = model_dir / "config" / "config.json"
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        validate_consolidated_configuration(config)
        self._apply(
            config,
            load_model=False,
            strict=strict,
            base_path=config_path.parent,
        )
        if load_model:
            model, _, _ = ModelLoader.load_artifact(model_dir, strict=strict)
            model_config = config["model"]
            self._wrapper.models_manager.attach_model(
                model,
                model_type=model_config["type"],
                model_name=model_config.get("name"),
                trained=True,
            )
        return config

    def _apply(
        self,
        config: Dict[str, Any],
        *,
        load_model: bool,
        strict: bool,
        base_path: Optional[Path] = None,
    ) -> None:
        data = config.get("data", {})
        context = data.get("context")
        context_type = config["experiment"]["context"].get("type")
        if context_type == "cohort" and isinstance(context, dict):
            self._wrapper.cohorts.load_config(
                context, activate=True, base_path=base_path
            )
        elif isinstance(context, dict):
            image_key = context.get("image_key")
            self._wrapper.context_manager.load_context_config(
                context, img_name_or_path=image_key, base_path=base_path
            )
        dataset_config = data.get("dataset")
        if isinstance(dataset_config, dict):
            parameters = dict(dataset_config.get("parameters", {}))
            parameters.pop("cohort", None)
            from ..models.datasets.dataset_manager import DatasetManager

            self._wrapper.active_dataset = DatasetManager.load_config(
                {**dataset_config, "parameters": parameters},
                active_context=self._wrapper.active_context,
                cohort_context=self._wrapper.cohorts.active_context,
            )
