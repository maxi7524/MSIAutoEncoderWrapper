from typing import Type, Dict, Any
from ..utils.logger import get_custom_logger
from .base_binner import MSIBaseBinner
from .base_inverse import MSIBaseInverseBinner
from ..utils.module_search import discover_modules
from ..utils.validators import resolve_component, validate_subclass

# Logger initialization
logger = get_custom_logger(__name__)


class BinnerManager:
    """
    Central manager and execution router for forward and inverse spectral binning strategies.
    
    This class maintains global isolated registries for mapping strategies, allowing
    for automatic resolution of preprocessing components from dynamic string configurations.
    """

    # Global component registries
    ## Dictionary storing mappings from unique identifiers to forward binner classes
    BINNER_REGISTRY: Dict[str, Type[MSIBaseBinner]] = {}
    ## Dictionary storing mappings from unique identifiers to inverse binner classes
    INVERSE_REGISTRY: Dict[str, Type[MSIBaseInverseBinner]] = {}

    @classmethod
    def register_binner(cls, name: str) -> Any:
        """
        Decorator factory to register a forward spectral binner strategy into the manager.

        :param name: Unique lookup string identifier for the binner strategy.
        :type name: str
        :return: Inner decorator function wrapping the targeted subclass.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseBinner]) -> Type[MSIBaseBinner]:
            validate_subclass(subclass, MSIBaseBinner, "BinnerRegistry")
            cls.BINNER_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_inverse_binner(cls, name: str) -> Any:
        """
        Decorator factory to register an inverse reconstruction strategy into the manager.

        :param name: Unique lookup string identifier for the inverse binner strategy.
        :type name: str
        :return: Inner decorator function wrapping the targeted subclass.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseInverseBinner]) -> Type[MSIBaseInverseBinner]:
            validate_subclass(subclass, MSIBaseInverseBinner, "InverseBinnerRegistry")
            # Registry updates
            ## Map the dynamic string token directly to the type constructor reference
            cls.INVERSE_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_binner(cls, name: Any, **kwargs: Any) -> MSIBaseBinner:
        """
        Resolves binner classes and executes setup configurations using dynamic property parameter maps.

        :param name: Registry key, binner class, or ready binner instance.
        :type name: Any
        :param kwargs: Structural operational properties delegated to constructors.
        :return: Concrete initialized instance implementing the MSIBaseBinner interface.
        :rtype: MSIBaseBinner
        :raises ProjectConfigError: If the requested binner is not registered or
            required constructor parameters are missing.
        """
        return resolve_component(
            target=name,
            registry=cls.BINNER_REGISTRY,
            component_type="Binner",
            expected_type=MSIBaseBinner,
            **kwargs,
        )

    @classmethod
    def get_inverse_binner(cls, name: Any, **kwargs: Any) -> MSIBaseInverseBinner:
        """
        Resolves inverse binner classes and executes configuration routines.

        :param name: Registry key, inverse binner class, or ready instance.
        :type name: Any
        :param kwargs: Parameters passed directly to target classes execution scopes.
        :return: Concrete initialized instance implementing the MSIBaseInverseBinner interface.
        :rtype: MSIBaseInverseBinner
        :raises ProjectConfigError: If the requested inverse binner is not
            registered or required constructor parameters are missing.
        """
        return resolve_component(
            target=name,
            registry=cls.INVERSE_REGISTRY,
            component_type="InverseBinner",
            expected_type=MSIBaseInverseBinner,
            **kwargs,
        )
    
    # --------------------------------------------------
    # Section: Automated Strategy Discovery
    # --------------------------------------------------

    @classmethod
    def discover_strategies(cls) -> None:
        """
        Explicitly imports local strategy packages.
        Dynamic scanning inside their __init__.py files triggers automatic decorator registrations.
        """
        discover_modules(f"{__package__}.binners_strategies")
        discover_modules(f"{__package__}.inverse_strategies")
