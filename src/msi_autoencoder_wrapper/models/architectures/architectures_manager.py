"""
Central orchestration registry and compilation engine for flexible multi-task model architectures.
"""

import inspect
import pkgutil
import importlib
import sys
import os
from typing import Type, Dict, Any, Optional, Union, List
import torch.nn as nn

from ...utils.logger import get_custom_logger
from ...utils.validators import validate_constructor_kwargs

# Logger initialization
logger = get_custom_logger(__name__)


class ArchitecturesManager:
    """
    Automated factory and multidimensional lookup database tracking structural subcomponents across model types.
    """

    # Multi-level isolated component registry database
    ## Dictionary structure mapping [model_type][category][component_name] to concrete classes
    _COMPONENT_REGISTRY: Dict[str, Dict[str, Dict[str, Type[nn.Module]]]] = {}
    
    # Architecture model graphs blueprints registry
    _MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

    # Global presets configuration blueprint storage mapping [model_type][preset_name] to callable factory blueprints
    _PRESET_REGISTRY: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------
    # Section: Registration Decorator Factories
    # --------------------------------------------------

    @classmethod
    def register_model_type(cls, model_type: str) -> Any:
        """
        Decorator factory to register a specific master architecture family graph model.

        :param model_type: Unique lookup token identifier string for the model family type.
        :type model_type: str
        :return: Inner decorator function wrapping the targeted architecture network class.
        :rtype: Callable
        """
        def decorator(subclass: Type[nn.Module]) -> Type[nn.Module]:
            cls._MODEL_REGISTRY[model_type] = subclass
            logger.debug("Registered primary model type container blueprint for: %s", model_type)
            return subclass
        return decorator

    @classmethod
    def register_component(cls, model_type: str, category: str, name: str) -> Any:
        """
        Unified decorator factory to register any functional subnet layer block under a specific scope.

        :param model_type: Scope identifier representing the targeted model family (e.g., 'autoencoder').
        :type model_type: str
        :param category: Target subcomponent layer tracking slot (e.g., 'encoder', 'decoder', 'head').
        :type category: str
        :param name: Unique lookup strategy lookup string token.
        :type name: str
        :return: Inner modifier decorator wrapper closure.
        :rtype: Callable
        """
        def decorator(subclass: Type[nn.Module]) -> Type[nn.Module]:
            # Structural lookup provisioning loop
            if model_type not in cls._COMPONENT_REGISTRY:
                cls._COMPONENT_REGISTRY[model_type] = {}
            if category not in cls._COMPONENT_REGISTRY[model_type]:
                cls._COMPONENT_REGISTRY[model_type][category] = {}
                
            cls._COMPONENT_REGISTRY[model_type][category][name] = subclass
            logger.debug("Appended strategy block [%s][%s] token handle: %s", model_type, category, name)
            return subclass
        return decorator

    @classmethod
    def register_preset(cls, model_type: str, name: str) -> Any:
        """
        Decorator factory to register an automated hyperparameter configuration preset macro.

        :param model_type: Scope identifier representing the targeted model family (e.g., 'autoencoder').
        :type model_type: str
        :param name: Unique lookup token string representing the configuration profile name.
        :type name: str
        :return: Inner modifier decorator wrapper closure.
        :rtype: Callable
        """
        def decorator(func: Any) -> Any:
            if model_type not in cls._PRESET_REGISTRY:
                cls._PRESET_REGISTRY[model_type] = {}
            cls._PRESET_REGISTRY[model_type][name] = func
            logger.debug("Successfully mapped configuration profile preset blueprint [%s]['%s'].", model_type, name)
            return func
        return decorator

    # --------------------------------------------------
    # Section: Universal Model Compilation Pipeline
    # --------------------------------------------------

    # Heading 1 (Dynamic Master Model Assembly Pass)
    @classmethod
    def build_model(cls, model_type: str, components_setup: Dict[str, Dict[str, Any]], **kwargs: Any) -> nn.Module:
        """
        Dynamically instantiates registered network layers and wraps them inside the requested master architecture family.

        Iterates across the provided components configurations buffer layout, extracts the appropriate layer strategies from
        the component registries database, resolves parameters maps, and injects the assembled modules dictionary into the
        coordinating master graph model constructor.

        :param model_type: Unique lookup token identifier token defining the targeted master model family.
        :type model_type: str
        :param components_setup: Buffered layout parameters map dictionary defining sub-blocks strategies and kwargs.
        :type components_setup: Dict[str, Dict[str, Any]]
        :param kwargs: Arbitrary backend extension footprints preserved for master graph instantiation.
        :return: Completely assembled and initialized PyTorch network module master graph instance.
        :rtype: nn.Module
        :raises KeyError: If the target model family or any of its requested subcomponents strategies are unregistered.
        """
        # Master registry verification checkpoint
        if model_type not in cls._MODEL_REGISTRY:
            error_msg = f"Master architecture type '{model_type}' not found in model graph registry."
            logger.error(error_msg)
            raise KeyError(error_msg)

        master_model_class = cls._MODEL_REGISTRY[model_type]
        resolved_components: Dict[str, nn.Module] = {}

        logger.info("Initializing multi-component sub-graph resolution phase for model family: %s", model_type)

        # Component resolution loop
        ## Iteratively scan and instantiate every sub-block configuration defined within the building buffer setup
        for category, setup_ledger in components_setup.items():
            
            if not isinstance(setup_ledger, dict):
                logger.error("Invalid configuration footprint encountered for block: '%s'. Expected dictionary.", category)
                continue

            ## Heading 2 (Determine Configuration Node Level)
            ### Check if the current block represents a direct single component or a nested structural collection
            strategy = setup_ledger.get("strategy") or setup_ledger.get("type") or setup_ledger.get("target")

            if strategy:
                ### Case 1: Direct component configuration block detected
                if model_type not in cls._COMPONENT_REGISTRY or category not in cls._COMPONENT_REGISTRY[model_type] or strategy not in cls._COMPONENT_REGISTRY[model_type][category]:
                    error_msg = f"Component strategy '{strategy}' for category '{category}' is unregistered under family '{model_type}'."
                    logger.error(error_msg)
                    raise KeyError(error_msg)

                component_class = cls._COMPONENT_REGISTRY[model_type][category][strategy]
                params = setup_ledger.get("params", {})
                
                logger.info("Instantiating standard component sub-module: Category='%s' using Strategy='%s'.", category, strategy)
                resolved_components[category] = component_class(**params)

            else:
                ### Case 2: Nested sub-components collection dictionary detected (e.g., multi-task heads)
                logger.info("Nested layout schema detected for category: '%s'. Traversing sub-components branch...", category)
                resolved_sub_collection: Dict[str, nn.Module] = {}

                for sub_key, sub_setup in setup_ledger.items():
                    if not isinstance(sub_setup, dict):
                        continue

                    sub_strategy = sub_setup.get("strategy") or sub_setup.get("type") or sub_setup.get("target")
                    sub_params = sub_setup.get("params", {})

                    if not sub_strategy:
                        logger.error("Subcomponent resolution pass aborted: Missing strategy descriptor inside collection '%s' for key: '%s'.", category, sub_key)
                        continue

                    if model_type not in cls._COMPONENT_REGISTRY or category not in cls._COMPONENT_REGISTRY[model_type] or sub_strategy not in cls._COMPONENT_REGISTRY[model_type][category]:
                        error_msg = f"Nested strategy '{sub_strategy}' under collection '{category}' is unregistered for family '{model_type}'."
                        logger.error(error_msg)
                        raise KeyError(error_msg)

                    sub_component_class = cls._COMPONENT_REGISTRY[model_type][category][sub_strategy]
                    
                    logger.info("Instantiating nested sub-module: Collection='%s', Key='%s' using Strategy='%s'.", category, sub_key, sub_strategy)
                    resolved_sub_collection[sub_key] = sub_component_class(**sub_params)

                resolved_components[category] = resolved_sub_collection

        # Graph aggregation execution pass
        ## Instantiate the structural master graph wrapper passing the fully populated resolved components matrix
        logger.info("Injecting resolved computational components ledger dictionary into the master architecture graph.")
        compiled_model = master_model_class(
            resolved_components=resolved_components,
            **kwargs
        )

        return compiled_model

    # --------------------------------------------------
    # Section: Recursive Auto-Discovery Loop
    # --------------------------------------------------

    @classmethod
    def discover_architectures(cls, package_path: Optional[List[str]] = None, package_name: Optional[str] = None) -> None:
        """
        Recursively scans package layout structures to load concrete strategies, ignoring template blueprint schemas.

        If parameters are omitted, resolves paths automatically relative to the manager location module.

        :param package_path: Absolute filesystem boundary track corresponding to path properties, defaults to None.
        :type package_path: Optional[List[str]]
        :param package_name: Hierarchical module tracking string index root name, defaults to None.
        :type package_name: Optional[str]
        """
        # Dynamic self-resolution fallback loop
        if package_path is None or package_name is None:
            ## Extract module parameters corresponding to the package root container
            base_dir = os.path.dirname(os.path.abspath(__file__))
            package_path = [base_dir]
            
            ### Resolve parent package hierarchical dot token definition
            current_module = cls.__module__
            if "." in current_module:
                package_name = ".".join(current_module.split(".")[:-1])
            else:
                package_name = current_module
                
            logger.debug("Architectures discovery system resolved baseline roots automatically: %s", package_name)

        # Skanowanie pakietów składowych
        ## Dynamic extraction of submodules using standard pkgutil walkers
        for _, module_name, _ in pkgutil.walk_packages(package_path, package_name + "."):
            ### Explicit boundary enforcement to bypass development blueprint schemas template folders
            if "schema" in module_name or module_name.split(".")[-1].startswith("_"):
                continue
                
            if module_name not in sys.modules:
                logger.debug("Auto-discovery framework importing operational architecture module: %s", module_name)
                importlib.import_module(module_name)