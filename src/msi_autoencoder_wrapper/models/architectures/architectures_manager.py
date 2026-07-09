"""
Central orchestration registry and compilation engine for flexible multi-task model architectures.
"""

import inspect
import pkgutil
import importlib
import sys
from typing import Type, Dict, Any, Optional, Union
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

    @classmethod
    def build_model(cls, model_type: str, components_setup: Dict[str, Any], **kwargs: Any) -> nn.Module:
        """
        Resolves decoupled subnet strategies from registries and builds the complete structural model graph.

        :param model_type: Scope identifier representing the targeted model family.
        :type model_type: str
        :param components_setup: Structured configurations mapping categories to blueprints or strategies setup maps.
        :type components_setup: Dict[str, Any]
        :param kwargs: Fallback variables or proxy context handles passed down to constructors.
        :return: Fully compiled and functional structural PyTorch network module.
        :rtype: torch.nn.Module
        :raises KeyError: If the requested model type container or any strategy token is missing from registry caches.
        """
        # Model class validation lookup
        if model_type not in cls._MODEL_REGISTRY:
            logger.error("Compilation aborted: Model family type wrapper '%s' is unregistered.", model_type)
            raise KeyError(f"Model family type '{model_type}' not found within registries.")

        resolved_components: Dict[str, Any] = {}
        type_registry = cls._COMPONENT_REGISTRY.get(model_type, {})

        # Subcomponent extraction loop
        ## Iterate through configured setup fields to build separate networks layers
        for category, setup in components_setup.items():
            if setup is None:
                resolved_components[category] = None
                continue

            category_db = type_registry.get(category, {})

            ### Handle multi-head dictionaries separately to resolve plain python dict of instances
            if category == "heads" and isinstance(setup, dict) and "type" not in setup:
                head_modules = {}
                for head_key, head_val in setup.items():
                    comp_name = head_val["type"] if isinstance(head_val, dict) else head_val
                    comp_params = head_val.get("params", {}).copy() if isinstance(head_val, dict) else {}
                    comp_params.update(kwargs)
                    
                    if comp_name not in category_db:
                        raise KeyError(f"Auxiliary Head '{comp_name}' missing from registry under type '{model_type}'.")
                        
                    validate_constructor_kwargs(category_db[comp_name], comp_name, comp_params)
                    head_modules[head_key] = category_db[comp_name](**comp_params)
                resolved_components[category] = head_modules
                continue

            ### Resolve standard unified linear component modules (encoders, decoders, projectors)
            comp_name = setup["type"] if isinstance(setup, dict) else setup
            comp_params = setup.get("params", {}).copy() if isinstance(setup, dict) else {}
            comp_params.update(kwargs)

            if comp_name not in category_db:
                logger.error("Resolution blocked: Strategy block '%s' missing under namespace [%s][%s]", comp_name, model_type, category)
                raise KeyError(f"Component '{comp_name}' missing from category '{category}' under model type '{model_type}'.")

            validate_constructor_kwargs(category_db[comp_name], comp_name, comp_params)
            resolved_components[category] = category_db[comp_name](**comp_params)

        # Graph assembly execution block
        ## Inject resolved subcomponents directly as explicit keyword arguments to match original signature contracts
        logger.info("Compiling network subcomponents directly into master model container: %s", model_type)
        master_model_class = cls._MODEL_REGISTRY[model_type]
        
        compiled_model = master_model_class(
            encoder=resolved_components.get("encoder"),
            decoder=resolved_components.get("decoder"),
            projector=resolved_components.get("projector"),
            heads=resolved_components.get("heads"),
            **kwargs
        )
        
        return compiled_model

    @classmethod
    def get_preset_blueprint(cls, model_type: str, name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Resolves and executes a registered preset macro to generate a complete architecture blueprint.

        :param model_type: Scope identifier representing the targeted model family.
        :type model_type: str
        :param name: Unique configuration profile lookup token name.
        :type name: str
        :param kwargs: Arbitrary runtime parameters passed down to the macro preset factories.
        :return: Complete compiled components structural blueprint setup map.
        :rtype: Dict[str, Any]
        """
        if model_type not in cls._PRESET_REGISTRY or name not in cls._PRESET_REGISTRY[model_type]:
            error_msg = f"Preset profile '{name}' not found under model family '{model_type}'."
            logger.error(error_msg)
            raise KeyError(error_msg)
            
        return cls._PRESET_REGISTRY[model_type][name](**kwargs)

    # --------------------------------------------------
    # Section: Getting available components 
    # --------------------------------------------------

    @classmethod
    def get_available_components(cls, model_type: str, category: str) -> dict[str, str]:
        """
        Queries the internal multidimensional registry to extract registered strategies and signatures.

        Extracts clean structural summary docstrings for runtime API discovery in notebooks.

        :param model_type: Name of the targeted model family scope (e.g., 'autoencoder').
        :type model_type: str
        :param category: Structural layer classification slot (e.g., 'encoder', 'decoder').
        :type category: str
        :return: Map linking strategy lookup tokens to their respective high-level docstring summaries.
        :rtype: dict[str, str]
        """
        # Dictionary inspection layer
        type_db = cls._COMPONENT_REGISTRY.get(model_type, {})
        category_db = type_db.get(category, {})
        
        summary: dict[str, str] = {}
        
        # Inspection loop
        ## Scan found class types to extract documentation metadata parameters
        for comp_name, comp_class in category_db.items():
            doc = inspect.getdoc(comp_class)
            summary[comp_name] = doc.split("\n")[0] if doc else "No description available."
            
        return summary

    @classmethod
    def get_component_details(cls, model_type: str, category: str, name: str) -> dict[str, Any]:
        """
        Extracts detailed constructor specifications, signature parameter maps, and complete docstrings.

        :param model_type: Target model family scope token.
        :type model_type: str
        :param category: Structural layer category slot.
        :type category: str
        :param name: Strategy string lookup key identifier.
        :type name: str
        :return: Detailed dictionary containing constructor metadata attributes.
        :rtype: dict[str, Any]
        :raises KeyError: If the requested strategy configuration is missing from registries.
        """
        type_db = cls._COMPONENT_REGISTRY.get(model_type, {})
        category_db = type_db.get(category, {})
        
        if name not in category_db:
            raise KeyError(f"Strategy '{name}' is missing from database path: [{model_type}][{category}]")
            
        target_class = category_db[name]
        init_method = getattr(target_class, "__init__", None)
        
        # Extract operational signatures
        params = {}
        if init_method:
            sign = inspect.signature(init_method)
            for p_name, p_obj in sign.parameters.items():
                if p_name != "self":
                    params[p_name] = str(p_obj.default) if p_obj.default != inspect.Parameter.empty else "REQUIRED"

        return {
            "strategy_name": name,
            "class_type": target_class.__name__,
            "full_docstring": inspect.getdoc(target_class) or "No documentation provided.",
            "expected_constructor_kwargs": params
        }

    @classmethod
    def verify_model_type_exists(cls, model_type: str) -> bool:
        """
        Verifies if a specific model family type is securely bound inside the primary model registry database.

        :param model_type: The candidate family type token identifier string.
        :type model_type: str
        :return: True if registered, False otherwise.
        :rtype: bool
        """
        return model_type in cls._MODEL_REGISTRY

    @classmethod
    def list_available_presets(cls, model_type: str) -> list[str]:
        """
        Lists all registered configuration preset profile names for a given model family scope.
        """
        return list(cls._PRESET_REGISTRY.get(model_type, {}).keys())

    # --------------------------------------------------
    # Section: Recursive Auto-Discovery Loop
    # --------------------------------------------------

    @classmethod
    def discover_architectures(cls, package_path: list, package_name: str) -> None:
        """
        Recursively scans package layout structures to load concrete strategies, ignoring template blueprint schemas.

        :param package_path: Absolute filesystem boundary track corresponding to path properties.
        :type package_path: list
        :param package_name: Hierarchical module tracking string index root name.
        :type package_name: str
        """
        # Skanowanie pakietów składowych
        ## Dynamic extraction of submodules using standard pkgutil walkers
        for _, module_name, _ in pkgutil.walk_packages(package_path, package_name + "."):
            ### Explicit boundary enforcement to bypass development blueprint schemas template folders
            if "schema" in module_name or module_name.split(".")[-1].startswith("_"):
                continue
                
            if module_name not in sys.modules:
                logger.debug("Auto-discovery framework importing operational architecture module: %s", module_name)
                importlib.import_module(module_name)