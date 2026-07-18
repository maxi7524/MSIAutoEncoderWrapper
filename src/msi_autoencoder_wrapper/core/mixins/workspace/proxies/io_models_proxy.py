# Heading 1 (IO Models Proxy Implementation)
## Specialized component managing atomic serialization, config consolidation, and model exports

from __future__ import annotations
import json
import torch
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from .base_workspace_proxy import BaseWorkspaceProxy
from .getters_and_setters_proxy import GLOBAL_CONTEXT_KEY
from .....utils.exceptions import raise_workspace_error
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


class IoModelsProxy(BaseWorkspaceProxy):
    """
    Proxy component executing atomic read/write operations for model weights, 
    consolidated JSON configurations, training history, and standalone PyTorch models.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the IO models workspace proxy.
        """
        super().__init__(*args, **kwargs)

    # --------------------------------------------------
    # Section: Atomic Config Serialization (JSON)
    # --------------------------------------------------

    def save_config_json(self, img_name: str, model_name: str, config_dict: Dict[str, Any]) -> None:
        """
        Physically serializes and writes the consolidated configuration dictionary to config.json.

        :param img_name: Name of the active image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        :param config_dict: Consolidated configuration dictionary containing sub-module parameters.
        :type config_dict: Dict[str, Any]
        """
        # Directory initialization and resolution
        self.create_structure(img_name=img_name, model_name=model_name)
        config_dir = self.get_config_dir(img_name=img_name, model_name=model_name)
        config_path = config_dir / "config.json"

        try:
            # Serialise configuration dict to formatted JSON file
            with open(config_path, mode="w", encoding="utf-8") as json_file:
                json.dump(config_dict, json_file, indent=4, ensure_ascii=False)
            logger.info("Consolidated configuration JSON saved successfully at: %s", config_path)
        except Exception as err:
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Failed to write configuration JSON to path {config_path}: {err}"
            )

    def load_config_json(self, img_name: str, model_name: str) -> Dict[str, Any]:
        """
        Reads and returns the consolidated configuration dictionary from config.json.

        :param img_name: Name of the active image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        :return: Consolidated configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config_dir = self.get_config_dir(img_name=img_name, model_name=model_name)
        config_path = config_dir / "config.json"

        if not config_path.exists():
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Configuration file not found at path: {config_path}"
            )

        try:
            with open(config_path, mode="r", encoding="utf-8") as json_file:
                config_dict = json.load(json_file)
            logger.debug("Configuration JSON loaded cleanly from: %s", config_path)
            return config_dict
        except Exception as err:
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Failed to deserialize configuration JSON at path {config_path}: {err}"
            )

    # --------------------------------------------------
    # Section: Atomic Weights Serialization (state_dict)
    # --------------------------------------------------

    def save_model_weights(self, img_name: str, model_name: str, state_dict: Dict[str, Any]) -> None:
        """
        Serializes and writes the PyTorch model weights state dictionary to weights.pt.

        :param img_name: Name of the active image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        :param state_dict: PyTorch state dictionary containing model weights.
        :type state_dict: Dict[str, Any]
        """
        self.create_structure(img_name=img_name, model_name=model_name)
        config_dir = self.get_config_dir(img_name=img_name, model_name=model_name)
        weights_path = config_dir / "weights.pt"

        try:
            torch.save(state_dict, weights_path)
            logger.info("Model weights state dict saved successfully at: %s", weights_path)
        except Exception as err:
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Failed to write model weights to path {weights_path}: {err}"
            )

    def load_model_weights(self, img_name: str, model_name: str) -> Dict[str, Any]:
        """
        Reads and returns the PyTorch model weights state dictionary from weights.pt.

        :param img_name: Name of the active image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        :return: PyTorch state dictionary mapped to CPU.
        :rtype: Dict[str, Any]
        """
        config_dir = self.get_config_dir(img_name=img_name, model_name=model_name)
        weights_path = config_dir / "weights.pt"

        if not weights_path.exists():
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Model weights checkpoint file not found at path: {weights_path}"
            )

        try:
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            logger.debug("Model weights state dict loaded cleanly from: %s", weights_path)
            return state_dict
        except Exception as err:
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Failed to load model weights at path {weights_path}: {err}"
            )

    # --------------------------------------------------
    # Section: Standalone Deployment Serialization
    # --------------------------------------------------

    def save_torch_model(self, img_name: str, model_name: str, model_object: torch.nn.Module) -> None:
        """
        Serializes the entire PyTorch model object (architecture definition + weights)
        to allow independent, third-party loading without relying on this library's managers.

        :param img_name: Name of the active image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        :param model_object: Fully compiled and active PyTorch nn.Module object.
        :type model_object: torch.nn.Module
        """
        self.create_structure(img_name=img_name, model_name=model_name)
        config_dir = self.get_config_dir(img_name=img_name, model_name=model_name)
        standalone_path = config_dir / "model_deployment_full.pt"

        try:
            torch.save(model_object, standalone_path)
            logger.info("Standalone deployment model object saved successfully at: %s", standalone_path)
        except Exception as err:
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Failed to serialize standalone PyTorch model object to {standalone_path}: {err}"
            )

    # --------------------------------------------------
    # Section: History Logs Serialization
    # --------------------------------------------------

    def save_history(self, img_name: str, model_name: str, history_dict: Dict[str, Any]) -> None:
        """
        Writes the training epoch history logs to history.json.

        :param img_name: Name of the active image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        :param history_dict: Dictionary mapping metric names to their historical epoch values.
        :type history_dict: Dict[str, Any]
        """
        self.create_structure(img_name=img_name, model_name=model_name)
        config_dir = self.get_config_dir(img_name=img_name, model_name=model_name)
        history_path = config_dir / "history.json"

        try:
            with open(history_path, mode="w", encoding="utf-8") as json_file:
                json.dump(history_dict, json_file, indent=4, ensure_ascii=False)
            logger.info("Training history logs saved successfully at: %s", history_path)
        except Exception as err:
            raise_workspace_error(
                context_name="WorkspaceIO",
                message=f"Failed to write training history to path {history_path}: {err}"
            )

    # --------------------------------------------------
    # Section: High-Level Consolidated Save
    # --------------------------------------------------

    def save_all(
        self, 
        img_name: str, 
        model_name: str, 
        state_dict: Dict[str, Any], 
        config_dict: Dict[str, Any], 
        history_dict: Optional[Dict[str, Any]] = None,
        model_object: Optional[torch.nn.Module] = None
    ) -> None:
        """
        Consolidates and executes a complete save of all state representations associated with the active model.
        Guarantees that the model can be fully restored within the library or deployed externally.

        :param img_name: Name of the active image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        :param state_dict: PyTorch state dictionary containing model weights.
        :type state_dict: Dict[str, Any]
        :param config_dict: Consolidated configuration dictionary.
        :type config_dict: Dict[str, Any]
        :param history_dict: Optional dictionary containing historical epoch metrics.
        :type history_dict: Optional[Dict[str, Any]]
        :param model_object: Optional PyTorch model object for standalone deployment.
        :type model_object: Optional[torch.nn.Module]
        """
        logger.info("Executing comprehensive Workspace backup for model '%s' under context '%s'.", model_name, img_name)
        
        # Sequentially trigger atomic saving operations
        ## 1. Save unified JSON configuration
        self.save_config_json(img_name=img_name, model_name=model_name, config_dict=config_dict)
        
        ## 2. Save weights state dict
        self.save_model_weights(img_name=img_name, model_name=model_name, state_dict=state_dict)

        ## 3. Save training metrics history if provided
        if history_dict is not None:
            self.save_history(img_name=img_name, model_name=model_name, history_dict=history_dict)

        ## 4. Save entire standalone model graph if provided
        if model_object is not None:
            self.save_torch_model(img_name=img_name, model_name=model_name, model_object=model_object)

        logger.info("Comprehensive backup completed successfully.")