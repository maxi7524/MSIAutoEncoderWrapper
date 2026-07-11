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
        self._building_buffer: Dict[str, Any] = {"heads": {}}
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

    def set_component(self, category: str, target: str, **kwargs: Any) -> None:
        """
        Appends an individual subcomponent layer setup block to the temporary building registry.
        """
        if not self.active_model_type:
            raise ValueError("Component setup rejected: Active model type is unassigned. Run set_model_type() first.")

        type_db = ArchitecturesManager._COMPONENT_REGISTRY.get(self.active_model_type, {})
        category_db = type_db.get(category, {})
        if target not in category_db:
            raise KeyError(f"Strategy '{target}' missing from registry branch: [{self.active_model_type}][{category}]")

        setup_block = {"type": target, "params": kwargs}

        if category in ("heads", "head"):
            head_task_name = kwargs.pop("task_name", f"task_{len(self._building_buffer['heads'])}")
            self._building_buffer["heads"][head_task_name] = setup_block
            logger.info("Appended multi-task head blueprint under key: %s", head_task_name)
        else:
            self._building_buffer[category] = setup_block
            logger.info("Registered configuration layout parameters for component: %s", category)

    def set_model_preset(self, name: str, **kwargs: Any) -> None:
        """
        Loads an automated data-driven configuration preset layout mapping.
        """
        if not self.active_model_type:
            raise ValueError("Preset mounting rejected: Establish active model type contexts first.")
        if not self._active_dataset_name:
            raise ValueError("Preset mounting rejected: Target dataset must be set via set_dataset() first.")

        # Temporal sampling initialization
        logger.debug("Building short-lived dataset instance to process model preset heuristics.")
        tmp_dataset = DatasetManager.get_dataset(
            name=self._active_dataset_name,
            active_context=self._wrapper.active_context
        )

        preset_db = ArchitecturesManager._PRESET_REGISTRY.get(self.active_model_type, {})
        if name not in preset_db:
            raise KeyError(f"Model preset profile '{name}' not found under family '{self.active_model_type}'.")

        # Execute automatic hyperparameter calculation macro
        compiled_blueprints = preset_db[name](
            msi_dataset=tmp_dataset,
            **kwargs
        )

        # Unpack blueprints maps into the active compilation buffer
        for category, setup_map in compiled_blueprints.items():
            self._building_buffer[category] = setup_map

        self._preset_name_used = name
        logger.info("Successfully mounted model preset package layout into buffer: %s", name)

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
        if getattr(workspace, "auto_create_dirs", True):
            workspace.create_model_directories_layout()

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