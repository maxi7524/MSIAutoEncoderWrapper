"""
Module providing stateful, unified neural network architecture and dataset binding configuration.
"""

import inspect
from typing import Any, Optional, Dict, List
import torch
import torch.nn as nn

from ....utils.logger import get_custom_logger
from ....utils.exceptions import ValidationError
from ....models.datasets.dataset_manager import DatasetManager
from ....models.architectures.architectures_manager import ArchitecturesManager
from ..io.active_context_mixin import ActiveContextProxy

# Logger initialization
logger = get_custom_logger(__name__)


class ModelsManagerProxy:
    """
    Unified proxy class executing reflection discovery and building operations for datasets and model layers.
    """

    def __init__(self, wrapper_ref: Any) -> None:
        """
        Initializes the unified models manager configuration bridge and runs packages self-discovery loops.

        :param wrapper_ref: Loose reference back to the coordinating facaded wrapper master object instance.
        :type wrapper_ref: Any
        """
        self._wrapper = wrapper_ref
        
        # Stateful configuration registers
        self.active_model_type: Optional[str] = None
        self._active_dataset_name: Optional[str] = None
        self._building_buffer: Dict[str, Any] = {}
        self._preset_name_used: Optional[str] = None

        # Execute unified self-discovery tracking loops during initialization
        # Heading 1 (Dynamic Registration Enforcement Loops)
        ## Trigger absolute modules reflection imports to populate static registration registries
        ArchitecturesManager.discover_architectures()
        DatasetManager.discover_strategies()

    # --------------------------------------------------
    # Section: Data Discovery & Inspection Methods (Reflect Engine)
    # --------------------------------------------------

    def get_available_datasets(self, print_return: bool = True, return_value: bool = False) -> Optional[Dict[str, Any]]:
        """
        Queries registered dataset components to extract baseline documentation metadata.

        :param print_return: Toggles markdown-style console printing, defaults to True.
        :type print_return: bool
        :param return_value: Returns the structured reflection dictionary if True, defaults to False.
        :type return_value: bool
        :return: Structured summary dictionary or None based on return_value flag.
        :rtype: Optional[Dict[str, Any]]
        """
        result = {}
        for name, cls_obj in DatasetManager._REGISTRY.items():
            doc = inspect.getdoc(cls_obj)
            result[name] = {
                "docstring": doc.split("\n")[0] if doc else "No description provided.",
                "parameters": {}
            }

        if print_return:
            print("\n" + "=" * 80 + "\n Available Dataset Strategies\n" + "=" * 80)
            for name, info in result.items():
                print(f"\n[Dataset]: '{name}'\n Description: {info['docstring']}")
            print("\n" + "=" * 80 + "\n")
            
        return result if return_value else None

    def get_available_model_types(self, print_return: bool = True, return_value: bool = False) -> Optional[Dict[str, str]]:
        """
        Lists all registered master model types along with their foundational class descriptions.

        :param print_return: Toggles formatted console printing, defaults to True.
        :type print_return: bool
        :param return_value: Returns the reflection mapping dictionary if True, defaults to False.
        :type return_value: bool
        :return: Map linking model type tokens to their class docstrings, or None.
        :rtype: Optional[Dict[str, str]]
        """
        result: Dict[str, str] = {}
        
        # Pull master model type definitions directly from registry mappings
        for m_type, cls_obj in ArchitecturesManager._MODEL_REGISTRY.items():
            doc = inspect.getdoc(cls_obj)
            result[m_type] = doc.split("\n")[0] if doc else "No description available."

        if print_return:
            print("\n" + "=" * 80)
            print(" Available Master Model Topologies")
            print("=" * 80)
            for m_type, description in result.items():
                print(f"\n - Model Type: '{m_type}'")
                print(f"   Core Role:  {description}")
            print("\n" + "=" * 80 + "\n")

        return result if return_value else None

    def get_available_component_categories(self, print_return: bool = True, return_value: bool = False) -> Optional[List[str]]:
        """
        Inspects the component registry to dynamically resolve all available slots for the active model type.

        :param print_return: Toggles formatted console output tracking layout slots, defaults to True.
        :type print_return: bool
        :param return_value: Returns the finalized categories list if True, defaults to False.
        :type return_value: bool
        :return: List of available component slots category names, or None.
        :rtype: Optional[List[str]]
        """
        if not self.active_model_type:
            raise ValueError("Discovery blocked: Establish active model type context via set_model_type() first.")

        # Extract the dynamic categories directly from your existing component registry dictionary
        # Heading 1 (Dynamic Schema Extraction Pass)
        ## Keys of this dictionary correspond exactly to populated categories ('encoder', 'decoder', etc.)
        categories = list(ArchitecturesManager._COMPONENT_REGISTRY.get(self.active_model_type, {}).keys())

        if print_return:
            print("\n" + "=" * 80)
            print(f" Dynamic Component Slots Available for Model Topology: '{self.active_model_type}'")
            print("=" * 80)
            if categories:
                for slot_name in categories:
                    print(f" - Component Slot: '{slot_name}'")
            else:
                print("   No component strategies have been discovered or registered for this model type yet.")
            print("\n" + "=" * 80 + "\n")

        return categories if return_value else None

    def get_available_strategies(self, category: str, print_return: bool = True, return_value: bool = False) -> Optional[Dict[str, Any]]:
        """
        Extracts concrete strategies, constructor parameters, and complete docstrings for a specific category.

        :param category: Target layer slot classification (e.g., 'encoder', 'decoder', 'projector').
        :type category: str
        :param print_return: Toggles parameterized console formatting matrix, defaults to True.
        :type print_return: bool
        :param return_value: Returns parsed metadata dictionaries if True, defaults to False.
        :type return_value: bool
        :return: Map linking strategy lookup tokens to structural parameters and descriptions or None.
        :rtype: Optional[Dict[str, Any]]
        """
        if not self.active_model_type:
            raise ValueError("Discovery blocked: Establish active model type context via set_model_type() first.")

        type_db = ArchitecturesManager._COMPONENT_REGISTRY.get(self.active_model_type, {})
        category_db = type_db.get(category, {})
        result = {}

        for name, cls_obj in category_db.items():
            doc = inspect.getdoc(cls_obj)
            init_method = getattr(cls_obj, "__init__", None)
            params = {}
            if init_method:
                sign = inspect.signature(init_method)
                for p_name, p_obj in sign.parameters.items():
                    if p_name != "self":
                        params[p_name] = str(p_obj.default) if p_obj.default != inspect.Parameter.empty else "REQUIRED"
            
            result[name] = {
                "docstring": doc if doc else "No documentation provided.",
                "parameters": params
            }

        if print_return:
            print("\n" + "=" * 80)
            print(f" Concrete Strategies Available for Slot: [{self.active_model_type}][{category}]")
            print("=" * 80)
            for name, info in result.items():
                print(f"\n - Strategy Identifier: '{name}'")
                print(f"   Description: {info['docstring'].splitlines()[0] if info['docstring'] else ''}")
                print("   Expected Parameters (kwargs):")
                if info["parameters"]:
                    for p_name, p_val in info["parameters"].items():
                        print(f"     * {p_name}: {p_val}")
                else:
                    print("     * None")
            print("=" * 80 + "\n")

        return result if return_value else None

    def get_available_presets(self, print_return: bool = True, return_value: bool = False) -> Optional[Dict[str, str]]:
        """
        Lists all registered configuration presets and extracts their analytical docstring descriptions.

        :param print_return: Toggles human-readable console summaries, defaults to True.
        :type print_return: bool
        :param return_value: Returns extracted configuration maps signatures if True, defaults to False.
        :type return_value: bool
        :return: Presets description dictionary index map or None.
        :rtype: Optional[Dict[str, str]]
        """
        if not self.active_model_type:
            raise ValueError("Discovery blocked: Establish active model type context via set_model_type() first.")
            
        preset_db = ArchitecturesManager._PRESET_REGISTRY.get(self.active_model_type, {})
        result: Dict[str, str] = {}
        
        for preset_name, preset_func in preset_db.items():
            doc = inspect.getdoc(preset_func)
            result[preset_name] = doc.split("\n")[0] if doc else "No description provided."

        if print_return:
            print("\n" + "=" * 80)
            print(f" Available Configuration Presets for Model Family: '{self.active_model_type}'")
            print("=" * 80)
            for name, summary in result.items():
                print(f"  - '{name}': {summary}")
            print("=" * 80 + "\n")
            
        return result if return_value else None
    
    # --------------------------------------------------
    # Section: Stateful Configuration Drivers
    # --------------------------------------------------

    def set_dataset(self, name: str) -> None:
        """
        Registers the target sampling strategy token to be constructed during final compilation.
        """
        if name not in DatasetManager._REGISTRY:
            raise KeyError(f"Dataset strategy '{name}' is unregistered within the DatasetManager registry.")
        self._active_dataset_name = name
        logger.debug("Buffered active target dataset strategy: %s", name)

    def set_model_type(self, model_type: str) -> None:
        """
        Locks the architectural pipeline context onto a specific model family type.
        """
        if model_type not in ArchitecturesManager._MODEL_REGISTRY:
            raise KeyError(f"Model family type '{model_type}' is unregistered within the system.")
        self.active_model_type = model_type
        self._building_buffer = {"heads": {}}
        self._preset_name_used = None
        logger.info("Operational network architecture context locked onto family: %s", model_type)

    def set_component(self, category: str, strategy: str, **kwargs: Any) -> None:
        """
        Appends an individual subcomponent layer setup block to the temporary building registry.
        """
        if not self.active_model_type:
            raise ValueError("Component setup rejected: Active model type is unassigned. Run set_model_type() first.")

        type_db = ArchitecturesManager._COMPONENT_REGISTRY.get(self.active_model_type, {})
        category_db = type_db.get(category, {})
        if strategy not in category_db:
            raise KeyError(f"Strategy '{strategy}' missing from registry branch: [{self.active_model_type}][{category}]")

        setup_block = {"type": strategy, "params": kwargs}

        if category in ("heads", "head"):
            head_task_name = kwargs.pop("task_name", f"task_{len(self._building_buffer['heads'])}")
            self._building_buffer["heads"][head_task_name] = setup_block
            logger.info("Appended multi-task head blueprint under key: %s", head_task_name)
        else:
            self._building_buffer[category] = setup_block
            logger.info("Registered configuration layout parameters for component: %s", category)

    # Heading 1 (Model Preset Configuration Bridge)
    def set_model_preset(self, name: str, **kwargs: Any) -> None:
        """
        Dynamically configures individual subcomponent parameters inside the model buffer using a preset.

        Validates the status of the active core context and its mounted readers, extracts the factory
        blueprint from the architecture registry, resolves parameters, and routes the generated layout
        definitions directly into the component buffer without triggering immediate compilation.

        :param name: Unique tracking token identifier for the registered architecture preset.
        :type name: str
        :param kwargs: Arbitrary dynamic parameter footprints used for overrides or model definitions.
        :raises ValueError: If active_model_type is unassigned or if active_context lacks an active reader.
        :raises KeyError: If the requested preset name is not registered for the current model family.
        """
        # Stateful environment validation
        ## 1. Ensure the master model category/type has been explicitly assigned
        if not self.active_model_type:
            logger.error("Preset injection rejected: active_model_type has not been initialized.")
            raise ValueError(
                "Cannot apply architecture preset: active_model_type is undefined. "
                "Set active_model_type first."
            )

        ## 2. Extract and validate the live session execution context from the master wrapper
        active_context = getattr(self._wrapper, "active_context", None)
        if not active_context or not getattr(active_context, "reader", None):
            logger.error("Preset injection rejected: Active execution context lacks an active reader session.")
            raise ValueError(
                "Cannot apply architecture preset: Active execution context does not contain "
                "an initialized data reader session. Mount an image context first."
            )

        logger.info(
            "Initiating model preset configuration layout lookup for family: %s, preset: %s",
            self.active_model_type,
            name
        )

        # Registry lookup sequence
        ## Verify existence of the requested preset blueprint within the global architecture registry
        preset_registry = ArchitecturesManager._PRESET_REGISTRY
        if self.active_model_type not in preset_registry or name not in preset_registry[self.active_model_type]:
            error_msg = f"Preset '{name}' not found for model type '{self.active_model_type}'."
            logger.error(error_msg)
            raise KeyError(error_msg)

        ## Retrieve the executable factory blueprint closure from the registry database
        preset_factory = preset_registry[self.active_model_type][name]

        # Preset layout execution pass
        ## Execute the factory method passing exclusively the active context and dynamic keyword overrides
        logger.debug("Executing preset factory blueprint with active context proxy reference.")
        preset_layout: Dict[str, Any] = preset_factory(active_context, **kwargs)

        # Component strategy extraction and buffer allocation loop
        ## Iteratively process each logical subcomponent block returned by the configuration blueprint
        for category, component_dict in preset_layout.items():
            ### Secure unified interface naming properties fallback parameters
            strategy = component_dict.get("strategy")
            params = component_dict.get("params", {})

            if not strategy:
                logger.error("Component extraction failed: Structural block for '%s' lacks a valid strategy key.", category)
                continue

            logger.info(
                "Preset parsing operational trace: Automatically routing component category '%s' using strategy '%s'.",
                category,
                strategy
            )

            ### Route definitions directly into the stateful component manager via proxy delegation
            self.set_component(category=category, strategy=strategy, **params)

        # Finalize state metadata allocation block
        ## Register the applied preset token within the buffer trace tracking variables without compiling the graph
        self._preset_name_used = name
        logger.info(
            "Model preset layout '%s' successfully staged into the configurations buffer. "
            "Graph compilation deferred for user modifications.",
            name
        )

    # --------------------------------------------------
    # Section: Compilation & Validation Gate
    # --------------------------------------------------

    def compile_model(self) -> nn.Module:
        """
        Executes structural checks, creates required directories, and instantiates the neural graph.
        """
        # Heading 1 (Structural Context Validation)
        missing_components: List[str] = []
        if not self._wrapper.active_context or not getattr(self._wrapper.active_context, "binner", None):
            missing_components.append("Active preprocessing pipeline context (Binner session unassigned)")
        if not self._active_dataset_name:
            missing_components.append("Target dataset strategy selection (Call set_dataset())")
        if not self.active_model_type:
            missing_components.append("Model family context environment allocation (Call set_model_type())")

        if missing_components:
            logger.error("Compilation aborted due to missing structural setup components.")
            raise ValidationError(missing_components)

        # Heading 1 (Experiment & Model Name Synchronization Pass)
        workspace = self._wrapper.workspace
        current_model_name = getattr(workspace, "active_model_name", None)

        if not current_model_name:
            fallback_name = self._preset_name_used if self._preset_name_used else f"custom_{self.active_model_type}"
            workspace.active_model_name = fallback_name
            logger.info("Workspace experiment name synchronized from modeling preset state: %s", fallback_name)

        # Heading 1 (Directory Tree Provisioning Block)
        if getattr(workspace, "create_required_directories", True):
            workspace.create_required_directories()

        # Heading 1 (Object Materialization and PyTorch Graph Assembly)
        ## 1. Materialize concrete PyTorch dataset sampler instance
        compiled_dataset = DatasetManager.get_dataset(
            name=self._active_dataset_name,
            active_context=self._wrapper.active_context
        )

        ## 2. Compile functional computational network master architecture graph
        compiled_network = ArchitecturesManager.build_model(
            model_type=self.active_model_type,
            components_setup=self._building_buffer
        )

        # Heading 1 (Mathematical Forward Graph Evaluation Trace)
        try:
            logger.debug("Executing functional forward simulation trace to verify mathematical consistency.")
            compiled_network.eval()
            with torch.no_grad():
                _, sample_tensor = compiled_dataset[0]
                mock_batch = sample_tensor.unsqueeze(0)
                _ = compiled_network(mock_batch)
        except Exception as error:
            logger.error("Forward execution simulation rejected: Underlying components dimensions mismatch.", exc_info=True)
            raise RuntimeError(f"Model graph compilation rejected. Forward pass validation failure: {error}") from error

        # Heading 1 (Hardware Allocation Pass)
        global_device = getattr(self._wrapper, "device", "cpu")
        compiled_network.to(global_device)
        logger.info("MSI Network master graph successfully compiled and migrated onto device: %s", global_device)

        # Assign variables back to active execution slots
        self._wrapper.active_model = compiled_network
        self._wrapper.active_dataset = compiled_dataset

        return compiled_network

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
        preset_label = self._preset_name_used if self._preset_name_used else "None"
        model_type_label = self.active_model_type if self.active_model_type else "None"
        
        summary_lines = [
            f"ModelsManagerProxy Configuration Summary:",
            f"  - Active Model Type : {model_type_label}",
            f"  - Applied Preset    : {preset_label}",
            f"  - Configured Buffer Components:"
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


# =====================================================================
# Section: Wrapper Collective Ingestion Mixin Block
# =====================================================================

class ModelsMixin:
    """
    Mixin class designed to inject stateful network management proxy features into the master wrapper.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Instantiates the models management command proxy bridge.
        """
        self.models = ModelsManagerProxy(wrapper_ref=self)
        self.active_model: Optional[nn.Module] = None
        self.active_dataset: Optional[Any] = None
        super().__init__(*args, **kwargs)