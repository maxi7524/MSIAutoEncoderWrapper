# Heading 1 (Models Manager Mixin and Proxy Interface)
## Unified interface integrating architectures, datasets, and training proxies through cooperative MRO

from __future__ import annotations
from typing import Any, Optional, Dict
import torch.nn as nn

# Proxy imports
from .proxies.architecture_proxy import ArchitectureProxy
from .proxies.dataset_proxy import DatasetProxy
from .proxies.training_proxy import TrainingProxy
from .proxies.model_runtime_proxy import ModelRuntimeProxy

# Centralized logging and configuration utilities
from ....utils.logger import get_custom_logger
from ....configuration import get_component_config, make_json_compatible

logger = get_custom_logger(__name__)


class ModelsManagerProxy(ArchitectureProxy, DatasetProxy, TrainingProxy, ModelRuntimeProxy):
    """
    Unified models manager controller. Aggregates building, dataset binding,
    and training operations through multiple inheritance, exposing a flat, zero-boilerplate API.
    """

    def __init__(self, wrapper_ref: Any, *args: Any, **kwargs: Any) -> None:
        """
        Initializes all inherited proxy domains using cooperative inheritance.

        :param wrapper_ref: Reference to the hosting master wrapper instance.
        :type wrapper_ref: Any
        """
        # Execute cooperative inheritance chain
        ## Unused parameters bubble up to BaseModelsManagerProxy and finally BaseWrapperProxy
        super().__init__(wrapper_ref=wrapper_ref, *args, **kwargs)

    # --------------------------------------------------
    # Section: Structural Visualization Utilities
    # --------------------------------------------------

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the currently loaded model configurations.

        :return: Structured summary of the target model type, active components, and applied presets.
        :rtype: str
        """
        # Heading 1 (String Formatting Orchestration Pass)
        ## Build structural baseline representation arrays
        preset_label = self._preset_name_used if getattr(self, "_preset_name_used", None) else "None"
        model_type_label = self.active_model_type if self.active_model_type else "None"
        
        summary_lines = [
            "ModelsManagerProxy Configuration Summary:",
            f"  - Active Model Type : {model_type_label}",
            f"  - Applied Preset    : {preset_label}",
            "  - Configured Buffer Components:"
        ]

        ## Iterate through cached structural parameters inside the active building buffer
        if self._building_buffer:
            for component_key, component_data in self._building_buffer.items():
                strategy_name = component_data.get("strategy", component_data.get("type", "Unknown"))
                summary_lines.append(f"    * {component_key} [{strategy_name}]")
        else:
            summary_lines.append("    * No components initialized in buffer.")

        return "\n".join(summary_lines)

    def print_model_config(self) -> None:
        """
        Explicitly prints the current structural layout ledger configuration onto stdout.
        """
        # Heading 1 (Stdout Stream Injection)
        ## Execute localized serialization string evaluation
        configuration_dump = self.__str__()
        print(configuration_dump)

    def get_model_config(self) -> Dict[str, Any]:
        """Return the current loaded-model context as a portable dictionary.

        The loaded-model context is independent from the active image context. Image
        pipeline configuration is collected separately by ``ContextManagerProxy``.

        :return: Consolidated architecture, dataset, and training configuration.
        :rtype: Dict[str, Any]
        """
        active_model = getattr(self._wrapper, "active_model", None)
        if active_model is None:
            raise RuntimeError("A compiled or loaded model is required for serialization.")
        runtime = get_component_config(active_model)
        runtime_parameters = runtime.get("parameters", {})
        model_parameters = runtime_parameters.get("parameters", {})
        component_configs = runtime_parameters.get("components", {})

        dataset_config = None
        active_dataset = getattr(self._wrapper, "active_dataset", None)
        if active_dataset is not None:
            dataset_config = get_component_config(active_dataset)

        split_config = None
        if active_dataset is not None:
            configured_split = getattr(active_dataset, "get_split_config", None)
            partitions = getattr(active_dataset, "_partitions", None)
            if callable(configured_split) and configured_split() is not None:
                split_config = configured_split()
            if partitions is not None:
                split_config = {
                    "config": split_config,
                    "manifest": partitions.manifest.get_config(),
                }

        return {
            "schema_version": 2,
            "model": {
                "name": self._active_model_name,
                "type": self.active_model_type,
                "preset": getattr(self, "_preset_name_used", None),
                "parameters": make_json_compatible(model_parameters),
                "components": component_configs,
            },
            "dataset": dataset_config,
            "split": split_config,
            "training": {
                "parameters": make_json_compatible(self._training_config),
            },
        }


class ModelsManagerMixin:
    """
    Mixin class designed to inject stateful network management proxy features into the master wrapper.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Instantiates the models manager command proxy and hooks it into the wrapper object context.
        """
        # Proxy assignment
        ## Instantiate the models manager proxy which inherits all interfaces dynamically
        self.models_manager = ModelsManagerProxy(wrapper_ref=self)
        
        # Active runtime states directly mounted on wrapper context
        self.active_model: Optional[nn.Module] = None
        self.active_dataset: Optional[Any] = None
        
        # Continue MRO chain initialization
        super().__init__(*args, **kwargs)
