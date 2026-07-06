from typing import Type, Dict, Any
from .strategies.base_reader import MSIBaseReader


class LoaderManager:
    """
    Central registration gateway managing native file system data parsers and readers.
    
    This component coordinates mapping from specific platform tags into explicit I/O drivers,
    maintaining decoupling across hardware containers and model networks.
    """

    # Driver tracking registration mapping
    ## Internal data ledger tracking valid abstract file system interpreter blueprints
    _REGISTRY: Dict[str, Type[MSIBaseReader]] = {}

    @classmethod
    @property
    def REGISTRY(cls) -> Dict[str, Type[MSIBaseReader]]:
        """
        Exposes read-only class property referencing internally registered strategy maps hooks.
        
        :return: Mapping tracking registered structural string indicators against concrete object definitions.
        :rtype: Dict[str, Type[MSIBaseReader]]
        """
        return cls._REGISTRY

    @classmethod
    def register_loader(cls, name: str) -> Any:
        """
        Decorator function routing explicit strategy declarations into the active driver ledger.

        :param name: Unique lookup token string representing the data loader configuration.
        :type name: str
        :return: Standard structural inner modifier wrapper closure.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseReader]) -> Type[MSIBaseReader]:
            # Register structural mapping class handler
            cls._REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_loader(cls, name: str, **kwargs: Any) -> MSIBaseReader:
        """
        Resolves driver classes and executes safe instantiation setups using custom parameter maps.

        :param name: Target lookup key for the requested I/O driver strategy.
        :type name: str
        :param kwargs: Property keyword attributes delegated directly into class loaders.
        :return: Initialized concrete implementation sub-type inheriting from MSIBaseReader.
        :rtype: msi_lib.loader.strategies.base_loader.MSIBaseReader
        :raises KeyError: If no structural loader matches the requested string query name.
        """
        # Validate entry availability within driver cache
        if name not in cls._REGISTRY:
            raise KeyError(f"Loader '{name}' not found. Available: {list(cls._REGISTRY.keys())}")
        
        # Factory initialization sequence
        return cls._REGISTRY[name](**kwargs)