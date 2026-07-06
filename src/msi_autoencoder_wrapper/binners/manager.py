from typing import Type, Dict, Any
from ..utils.logger import get_custom_logger
from .binners_strategies.base_binner import MSIBaseBinner
from .inverse_strategies.base_inverse import MSIBaseInverseBinner

# Logger initialization
## Retrieve configured synchronized logger instance for this module context
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
            # Registry updates
            ## Map the dynamic string token directly to the type constructor reference
            cls.BINNER_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_inverse_binner(cls, name: str) -> Any:
        """
        Decorator factory to register an inverse spectral binner strategy into the manager.

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
        Factory resolution method to instantiate a registered forward binner strategy.

        :param name: Unique registration identifier for the requested binner.
        :type name: str
        :param kwargs: Arbitrary keyword arguments passed to the strategy constructor.
        :return: Concrete initialized instance implementing the MSIBaseBinner interface.
        :rtype: msi_lib.binners.binners_strategies.base_binner.MSIBaseBinner
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
        Factory resolution method to instantiate a registered inverse binner strategy.

        :param name: Unique registration identifier for the requested inverse binner.
        :type name: str
        :param kwargs: Arbitrary keyword arguments passed to the strategy constructor.
        :return: Concrete initialized instance implementing the MSIBaseInverseBinner interface.
        :rtype: msi_lib.binners.inverse_strategies.base_inverse.MSIBaseInverseBinner
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