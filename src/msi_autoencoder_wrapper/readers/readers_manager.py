from typing import Type, Dict, Any
from .base_reader import MSIBaseReader
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

class ReaderManager:
    """
    Central registration gateway managing native file system data parsers and readers.
    
    This component coordinates mapping from specific platform tags into explicit I/O drivers,
    maintaining decoupling across hardware containers and model networks.
    """

    # Driver tracking registration mapping
    ## Internal data ledger tracking valid abstract file system interpreter blueprints
    REGISTRY: Dict[str, Type[MSIBaseReader]] = {}

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
            cls.REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_reader(cls, name: str, **kwargs: Any) -> MSIBaseReader:
        """
        Resolves driver classes and executes safe instantiation setups using custom parameter maps.

        :param name: Target lookup key for the requested I/O driver strategy.
        :type name: str
        :param kwargs: Property keyword attributes delegated directly into class loaders.
        :return: Initialized concrete implementation sub-type inheriting from MSIBaseReader.
        :rtype: MSIBaseReader
        :raises KeyError: If no structural loader matches the requested string query name.
        """
        # Validate entry availability within driver cache
        if name not in cls.REGISTRY:
            raise KeyError(f"Loader '{name}' not found. Available: {list(cls.REGISTRY.keys())}")
        
        # Factory initialization sequence
        return cls.REGISTRY[name](**kwargs)

    # --------------------------------------------------
    # Section: Automated Strategy Discovery
    # --------------------------------------------------

    @classmethod
    def discover_strategies(cls) -> None:
        """
        Explicitly imports local strategy packages. 
        Dynamic scanning inside strategies/__init__.py triggers automatic registration.
        """
        try:
            from . import strategies
            logger.info("ReaderManager successfully auto-discovered and registered strategy drivers.")
        except Exception as e:
            logger.exception("ReaderManager critical failure during automatic strategy discovery: %s", e)