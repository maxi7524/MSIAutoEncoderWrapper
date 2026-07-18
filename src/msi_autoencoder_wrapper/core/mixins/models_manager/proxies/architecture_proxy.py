# Heading 1 (Architecture Proxy Implementation)
## Specialized component managing model selection, components parameters reflection, preset configurations, and network graph compilation

from __future__ import annotations
import inspect
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import torch
import torch.nn as nn

# Base class and factory imports
from .base_models_manager_proxy import BaseModelsManagerProxy
from .....models.architectures.architectures_manager import ArchitecturesManager
from .....models.datasets.dataset_manager import DatasetManager

# Centralized utilities imports
from .....utils.logger import get_custom_logger
from .....utils.exceptions import raise_validation_error, raise_model_initialization_error
from ....utils.printing import extract_component_signatures, print_formatted_components

if TYPE_CHECKING:
    pass

# Logger initialization
logger = get_custom_logger(__name__)


class ArchitectureProxy(BaseModelsManagerProxy):
    """
    Proxy component executing reflection discovery and building operations for model layers.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the architecture manager proxy.
        """
        super().__init__(*args, **kwargs)

        ArchitecturesManager.discover_architectures()

    # --------------------------------------------------
    # Section: Strategy Discovery
    # --------------------------------------------------

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
        # Strategy lookup
        ## Extract classes from the main model registry
        result: Dict[str, str] = {}
        for m_type, cls_obj in ArchitecturesManager._MODEL_REGISTRY.items():
            doc = inspect.getdoc(cls_obj)
            result[m_type] = doc.split("\n")[0] if doc else "No description available."

        if print_return:
            ## Output formatting
            ### Format and delegate console rendering to the centralized system printer
            print_formatted_components(
                title="Available Master Model Topologies",
                key_label="Model Type",
                components_info={
                    name: {"docstring": desc, "parameters": {}} for name, desc in result.items()
                }
            )

        if return_value:
            return result
        return None

    def get_available_component_categories(self, print_return: bool = True, return_value: bool = False) -> Optional[List[str]]:
        """
        Lists all available structural building block sub-component types registered inside the factory.

        :param print_return: Toggles formatted console printing, defaults to True.
        :type print_return: bool
        :param return_value: Returns the list of category names if True, defaults to False.
        :type return_value: Optional[List[str]]
        """
        # Category discovery
        ## Fetch keys from the central component registry
        categories = list(ArchitecturesManager._COMPONENT_REGISTRY.keys())

        if print_return:
            ## Output rendering
            ### Print categorized list of modules using centralized system printer
            print_formatted_components(
                title="Registered Component Categories",
                key_label="Category",
                components_info={
                    cat: {"docstring": "Registered component package", "parameters": {}} for cat in categories
                }
            )

        if return_value:
            return categories
        return None

    def _get_available_components_info(
        self, 
        registry: Dict[str, Any], 
        title: str, 
        key_label: str, 
        print_return: bool,
        return_value: bool
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Internal helper utility to extract documentation and constructor signatures across registries.
        Delegated to unified printing module to avoid duplicate inspector code.

        :param registry: Target manager registry dictionary mapping keys to component classes.
        :type registry: Dict[str, Any]
        :param title: Header string used during beautiful print formatting sequences.
        :type title: str
        :param key_label: Contextual descriptor label pointing to the strategy type.
        :type key_label: str
        :param print_return: Flag determining whether data logs are pushed to stdout streams.
        :type print_return: bool
        :param return_value: Flag determining whether return dict with data logs.
        :type return_value: bool
        :return: Deeply nested mapping matching strategy aliases to structural property states, or None.
        :rtype: Optional[Dict[str, Dict[str, Any]]]
        """
        # Centralized Signature Extraction
        result = extract_component_signatures(registry=registry)

        if print_return:
            ## Print details using unified layout engine
            print_formatted_components(
                title=title,
                key_label=key_label,
                components_info=result
            )

        if return_value:
            return result
        return None

    def get_available_components(self, category: str, print_return: bool = True, return_value: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Queries registered architectural sub-components (encoders, decoders, heads) within a specified category.

        :param category: Target component classification category string key.
        :type category: str
        :param print_return: Toggles console formatting, defaults to True.
        :type print_return: bool
        :param return_value: Returns the deep reflection dictionary, defaults to False.
        :type return_value: bool
        :return: Nested dictionary of component classes and signatures, or None.
        :rtype: Optional[Dict[str, Dict[str, Any]]]
        """
        # Category validation
        ## Verify component registry availability
        if category not in ArchitecturesManager._COMPONENT_REGISTRY:
            raise_validation_error(
                context_name="ModelsManager",
                message=f"Category '{category}' is unregistered within the ArchitecturesManager components registry."
            )

        # Extraction delegation
        ## Route search parameters to the shared signature evaluator
        registry_target = ArchitecturesManager._COMPONENT_REGISTRY[category]
        return self._get_available_components_info(
            registry=registry_target,
            title=f"Available Components in Category: '{category}'",
            key_label=category.capitalize(),
            print_return=print_return,
            return_value=return_value
        )

    def get_available_model_presets(self, print_return: bool = True, return_value: bool = False) -> Optional[Dict[str, str]]:
        """
        Lists all registered configuration presets and extracts their analytical docstring descriptions.

        :param print_return: Toggles human-readable console summaries, defaults to True.
        :type print_return: bool
        :param return_value: Returns extracted configuration maps signatures if True, defaults to False.
        :type return_value: bool
        :return: Presets description dictionary index map or None.
        :rtype: Optional[Dict[str, str]]
        """
        # Validation checks
        ## 1. Ensure the active model type context has been set
        if not self.active_model_type:
            raise_validation_error(
                context_name="ModelsManager",
                message="Discovery blocked: Establish active model type context via set_model() first."
            )

        # Retrieve configurations
        ## Pull target closure blueprints registry
        preset_db = ArchitecturesManager._PRESET_REGISTRY.get(self.active_model_type, {})
        result: Dict[str, str] = {}

        for preset_name, preset_func in preset_db.items():
            doc = inspect.getdoc(preset_func)
            result[preset_name] = doc.split("\n")[0] if doc else "No description provided."

        if print_return:
            ## Delegate rendering task directly to unified printer helper
            print_formatted_components(
                title=f"Available Configuration Presets for Model Family: '{self.active_model_type}'",
                key_label="Preset",
                components_info={
                    name: {"docstring": desc, "parameters": {}} for name, desc in result.items()
                }
            )

        if return_value:
            return result
        return None

    # --------------------------------------------------
    # Section: Core Model Configuration and Assembly
    # --------------------------------------------------

    def set_model_type(self, model_type: str, model_name: str, **kwargs: Any) -> None:
        """
        Registers the master architecture topology name and saves structural initialization configurations.

        :param model_type: Master model family category name.
        :type model_type: str
        :param model_name: Identifier name key of the target class to build.
        :type model_name: str
        :param kwargs: Constructor parameter settings passed directly to class initialization.
        :type kwargs: Any
        """
        # Topology verification
        ## Verify topology exists in registry keys
        if model_type not in ArchitecturesManager._MODEL_REGISTRY:
            raise_model_initialization_error(
                model_name=model_name,
                message=f"Model type topology '{model_type}' is unregistered in ArchitecturesManager."
            )

        # State updates
        ## Cache configuration inside the stateful active building buffer dictionary
        self.active_model_type = model_type
        self._building_buffer["model"] = {
            "type": model_type,
            "strategy": model_name,
            "kwargs": kwargs
        }
        logger.debug("Buffered active model architecture type %s with implementation: %s", model_type, model_name)

    def set_component(self, category: str, name: str, **kwargs: Any) -> None:
        """
        Registers a specific structural subcomponent configuration within the building buffer.

        :param category: The architectural category designation (e.g., 'encoder', 'decoder').
        :type category: str
        :param name: The identifier token of the strategy to instantiate.
        :type name: str
        :param kwargs: Arbitrary initialization parameters passed to the component constructor.
        :type kwargs: Any
        :raises ValidationError: If the active model type is unset, or if category/strategy is unregistered.
        """
        # Active state validation
        ## Ensure the active model type is configured
        if not self.active_model_type:
            raise_validation_error(
                context_name="ModelsManager",
                message="Active model type is not set. Please select a model type or apply a preset first."
            )

        # Category validation
        ## Ensure the active model family is registered
        if self.active_model_type not in ArchitecturesManager._COMPONENT_REGISTRY:
            raise_validation_error(
                context_name="ModelsManager",
                message=f"Model type '{self.active_model_type}' is unregistered within ArchitecturesManager."
            )

        ## Ensure the requested category belongs to the active model type
        if category not in ArchitecturesManager._COMPONENT_REGISTRY[self.active_model_type]:
            raise_validation_error(
                context_name="ModelsManager",
                message=f"Category '{category}' is unregistered for model type '{self.active_model_type}'."
            )

        # Strategy verification
        ## Ensure the specific strategy is registered under the given category
        if name not in ArchitecturesManager._COMPONENT_REGISTRY[self.active_model_type][category]:
            raise_validation_error(
                context_name="ModelsManager",
                message=f"Strategy '{name}' is unregistered under category '{category}' for model type '{self.active_model_type}'."
            )

        # Stateful buffer updates
        ## Store the components configuration parameters inside the centralized shared buffer
        self._building_buffer[category] = {
            "strategy": name,
            "kwargs": kwargs
        }
        logger.debug("Buffered component strategy allocation: category='%s', strategy='%s'", category, name)

    def set_model_preset(self, name: str, **kwargs: Any) -> None:
        """
        Dynamically configures individual subcomponent parameters inside the model buffer using a preset.

        Validates the status of the active core context and its mounted readers, extracts the factory
        blueprint from the architecture registry, resolves parameters, and routes the generated layout
        definitions directly into the component buffer without triggering immediate compilation.

        :param name: Unique tracking token identifier for the registered architecture preset.
        :type name: str
        :param kwargs: Arbitrary dynamic parameter footprints used for overrides or model definitions.
        :type kwargs: Any
        """
        # Stateful environment validation
        ## 1. Ensure the master model category/type has been explicitly assigned
        if not self.active_model_type:
            logger.error("Preset injection rejected: active_model_type has not been initialized.")
            raise_validation_error(
                context_name="ModelsManager",
                message="Cannot apply architecture preset: active_model_type is undefined. Set active_model_type first."
            )

        ## 2. Extract and validate the live session execution context from the master wrapper
        active_context = getattr(self._wrapper, "active_context", None)
        if not active_context or not getattr(active_context, "reader", None):
            logger.error("Preset injection rejected: Active execution context lacks an active reader session.")
            raise_validation_error(
                context_name="ModelsManager",
                message="Cannot apply architecture preset: Active execution context does not contain an initialized data reader session. Mount an image context first."
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
            logger.error("Preset verification failed: %s", error_msg)
            raise_validation_error(
                context_name="ModelsManager",
                message=error_msg
            )

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
            self.set_component(category=category, name=strategy, **params)

        # Finalize state metadata allocation block
        ## Register the applied preset token within the buffer trace tracking variables without compiling the graph
        self._preset_name_used = name
        logger.info(
            "Model preset layout '%s' successfully staged into the configurations buffer. Graph compilation deferred for user modifications.",
            name
        )

    def compile_model(self, run_validation_pass: bool = True) -> nn.Module:
        """
        Synthesizes the buffered parameters, instantiates the master network graph layers,
        verifies tensor shapes compatibility, and migrates the model to the target hardware device.

        :param run_validation_pass: Enforces execution of forward pass check if True. Defaults to True.
        :type run_validation_pass: bool
        :return: Completed and compiled PyTorch nn.Module object.
        :rtype: nn.Module
        :raises ValidationError: If the active model family type is unassigned or compilation fails.
        """
        # Active state validation
        ## Ensure the active model family type is configured
        if not self.active_model_type:
            raise_model_initialization_error(
                model_name="Unassigned",
                message="Cannot compile model graph: No active model family has been set."
            )

        logger.info("Initializing PyTorch network compilation sequences from buffered blueprints.")

        # Architecture synthesis
        try:
            ## Consolidate active parameters and component dictionaries
            components_setup: Dict[str, Any] = {}
            model_kwargs: Dict[str, Any] = {}
            
            ### Extract general model hyperparameters if explicitly set in buffer
            if "model" in self._building_buffer:
                model_kwargs.update(self._building_buffer["model"].get("kwargs", {}))
            
            ### Extract structural components using correct dictionary interface mapping kwargs -> params
            for key in ["encoder", "decoder", "head", "projector"]:
                if key in self._building_buffer:
                    components_setup[key] = {
                        "strategy": self._building_buffer[key].get("strategy"),
                        "params": self._building_buffer[key].get("kwargs", {})
                    }

            ## Instantiate network through classmethod builder orchestration pass
            compiled_network = ArchitecturesManager.build_model(
                model_type=self.active_model_type,
                components_setup=components_setup,
                **model_kwargs
            )
            
        except Exception as error:
            logger.error("Failed to construct network layers graph from blueprints.", exc_info=True)
            raise_model_initialization_error(
                model_name="Assembly",
                message=f"Construct execution failure: {error}"
            )

        # Dataset initialization and layout mapping pass
        ## Extract active context reference from parent wrapper
        active_context = getattr(self._wrapper, "active_context", None)
        
        ## Instantiate the concrete dataset strategy tied explicitly to this context
        dataset_blueprint = self._building_buffer.get("dataset", {})
        dataset_name = dataset_blueprint.get("strategy") or self._active_dataset_name or "PixelDataset"
        dataset_kwargs = dataset_blueprint.get("kwargs", {}).copy()

        logger.info("Compiling and binding dataset strategy: %s to the active model graph.", dataset_name)
        
        try:
            ## Invoke DatasetManager factory, passing the vital active_context
            compiled_dataset = DatasetManager.get_dataset(
                name=dataset_name,
                active_context=active_context,
                **dataset_kwargs
            )
        except Exception as error:
            logger.error("Failed to instantiate dataset via DatasetManager.", exc_info=True)
            raise_model_initialization_error(
                model_name="Dataset",
                message=f"Dataset resolution failure: {error}"
            )

        # Dimensional checking validation
        if run_validation_pass:
            ## Execute validation pass checks using active dataset sample spectra
            if compiled_dataset is None:
                logger.warning("Forward pass verification skipped: active_dataset state is unassigned.")
            else:
                try:
                    compiled_network.eval()
                    with torch.no_grad():
                        ### Extract single tensor and dynamically correct dimensions
                        _, sample_tensor = compiled_dataset[0]
                        if sample_tensor.dim() == 1:
                            mock_batch = sample_tensor.unsqueeze(0)
                        else:
                            mock_batch = sample_tensor

                        ### Execute forward propagation execution pass
                        _ = compiled_network(mock_batch)
                except Exception as error:
                    logger.error("Forward execution simulation rejected: Underlying components dimensions mismatch.", exc_info=True)
                    raise_model_initialization_error(
                        model_name=self.active_model_type,
                        message=f"Model graph compilation rejected. Forward pass validation failure: {error}"
                    )

        # Hardware mapping migration
        global_device = getattr(self._wrapper, "device", "cpu")
        compiled_network.to(global_device)

        logger.info(
            "MSI Network master graph successfully compiled and migrated onto execution hardware: %s",
            global_device
        )

        # Update core state tracking references
        self._wrapper.active_model = compiled_network
        self._wrapper.active_dataset = compiled_dataset

        return compiled_network