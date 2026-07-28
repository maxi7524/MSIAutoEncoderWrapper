"""
Central orchestration registry and compilation engine for flexible multi-task model architectures.
"""

from typing import Type, Dict, Any, Optional, List
import torch.nn as nn

from ...utils.logger import get_custom_logger
from ...utils.module_search import discover_modules
from ...utils.validators import resolve_component, validate_subclass

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
            validate_subclass(subclass, nn.Module, "ArchitectureRegistry")
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
            validate_subclass(subclass, nn.Module, "ArchitectureComponentRegistry")
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
        :raises ProjectConfigError: If the model family or a requested component
            is unregistered, or required constructor parameters are missing.
        """
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
            strategy = cls._get_setup_target(setup_ledger)

            if strategy is not None:
                ### Case 1: Direct component configuration block detected
                component_registry = cls._COMPONENT_REGISTRY.get(model_type, {}).get(category, {})
                params = setup_ledger.get("params", {})
                
                logger.info("Instantiating standard component sub-module: Category='%s' using Strategy='%s'.", category, strategy)
                resolved_components[category] = resolve_component(
                    target=strategy,
                    registry=component_registry,
                    component_type=f"{model_type}.{category}",
                    expected_type=nn.Module,
                    **params,
                )

            else:
                ### Case 2: Nested sub-components collection dictionary detected (e.g., multi-task heads)
                logger.info("Nested layout schema detected for category: '%s'. Traversing sub-components branch...", category)
                resolved_sub_collection: Dict[str, nn.Module] = {}

                for sub_key, sub_setup in setup_ledger.items():
                    if not isinstance(sub_setup, dict):
                        continue

                    sub_strategy = cls._get_setup_target(sub_setup)
                    sub_params = sub_setup.get("params", {})

                    if sub_strategy is None:
                        logger.error("Subcomponent resolution pass aborted: Missing strategy descriptor inside collection '%s' for key: '%s'.", category, sub_key)
                        continue

                    logger.info("Instantiating nested sub-module: Collection='%s', Key='%s' using Strategy='%s'.", category, sub_key, sub_strategy)
                    registry_category = "head" if category == "heads" else category
                    component_registry = cls._COMPONENT_REGISTRY.get(model_type, {}).get(registry_category, {})
                    resolved_sub_collection[sub_key] = resolve_component(
                        target=sub_strategy,
                        registry=component_registry,
                        component_type=f"{model_type}.{category}",
                        expected_type=nn.Module,
                        **sub_params,
                    )

                resolved_components[category] = resolved_sub_collection

        # Graph aggregation execution pass
        ## Instantiate the structural master graph wrapper passing the fully populated resolved components matrix
        logger.info("Injecting resolved computational components ledger dictionary into the master architecture graph.")
        compiled_model = resolve_component(
            target=model_type,
            registry=cls._MODEL_REGISTRY,
            component_type="Architecture",
            expected_type=nn.Module,
            resolved_components=resolved_components,
            **kwargs,
        )

        return compiled_model

    @staticmethod
    def _get_setup_target(setup: Dict[str, Any]) -> Any:
        """Return the first explicitly configured component target without truth testing it."""
        for key in ("target", "strategy", "type"):
            target = setup.get(key)
            if target is not None:
                return target
        return None

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
        del package_path
        package_root = package_name or __package__
        discover_modules(package_root, excluded_parts={"schema"})
