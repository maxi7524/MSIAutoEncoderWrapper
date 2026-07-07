from typing import Type, Dict, Any
from ..utils.logger import get_custom_logger
from .base_binner import MSIBaseBinner
from .base_inverse import MSIBaseInverseBinner

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
            # Registry updates
            ## Map the dynamic string token directly to the type constructor reference
            cls.INVERSE_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_binner(cls, name: str, **kwargs: Any) -> MSIBaseBinner:
        """
        Resolves binner classes and executes setup configurations using dynamic property parameter maps.

        :param name: Target component lookup verification key.
        :type name: str
        :param kwargs: Structural operational properties delegated to constructors.
        :return: Concrete initialized instance implementing the MSIBaseBinner interface.
        :rtype: MSIBaseBinner
        :raises KeyError: If the requested strategy name is not found within the registry.
        """
        # Strategy lookup block
        ## Validate existence of target component key in registration cache
        if name not in cls.BINNER_REGISTRY:
            error_msg = f"Binner '{name}' not found in registry. Available: {list(cls.BINNER_REGISTRY.keys())}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        # Instance generation pipeline
        ## Resolve the constructor class from mapping and initialize with provided parameters
        return cls.BINNER_REGISTRY[name](**kwargs)

    @classmethod
    def get_inverse_binner(cls, name: str, **kwargs: Any) -> MSIBaseInverseBinner:
        """
        Resolves inverse binner classes and executes configuration routines.

        :param name: Target structural lookup reference key.
        :type name: str
        :param kwargs: Parameters passed directly to target classes execution scopes.
        :return: Concrete initialized instance implementing the MSIBaseInverseBinner interface.
        :rtype: MSIBaseInverseBinner
        :raises KeyError: If the requested strategy name is not found within the registry.
        """
        # Strategy lookup block
        ## Validate existence of target component key in registration cache
        if name not in cls.INVERSE_REGISTRY:
            error_msg = f"Inverse Binner '{name}' not found in registry. Available: {list(cls.INVERSE_REGISTRY.keys())}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        # Instance generation pipeline
        ## Resolve the constructor class from mapping and initialize with provided parameters
        return cls.INVERSE_REGISTRY[name](**kwargs)
    
    # --------------------------------------------------
    # Section: Automated Strategy Discovery
    # --------------------------------------------------

    @classmethod
    def discover_strategies(cls) -> None:
        """
        Explicitly imports local strategy packages.
        Dynamic scanning inside their __init__.py files triggers automatic decorator registrations.
        """
        try:
            from . import binners_strategies
            from . import inverse_strategies
            logger.info("BinnerManager successfully auto-discovered and registered compression drivers.")
        except Exception as e:
            logger.exception("BinnerManager critical failure during automatic strategy discovery: %s", e)