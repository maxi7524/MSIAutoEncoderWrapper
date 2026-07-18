# Heading 1 (Models Manager Mixin and Proxy Interface)
## Unified interface integrating architectures, datasets, and training proxies through cooperative MRO

from __future__ import annotations
from typing import Any, Optional, Dict
import torch.nn as nn

# Proxy imports
from .proxies.architecture_proxy import ArchitectureProxy
from .proxies.dataset_proxy import DatasetProxy
from .proxies.training_proxy import TrainingProxy

# Scentralizowane logger i exceptions
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ModelsManagerProxy(ArchitectureProxy, DatasetProxy, TrainingProxy):
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