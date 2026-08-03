from typing import Type, Dict, Any
from .base_reader import MSIBaseReader
from ..utils.logger import get_custom_logger
from ..utils.module_search import discover_modules
from ..utils.validators import resolve_component, validate_subclass

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
            validate_subclass(subclass, MSIBaseReader, "ReaderRegistry")
            # Register structural mapping class handler
            cls.REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_reader(cls, name: Any, **kwargs: Any) -> MSIBaseReader:
        """
        Resolves driver classes and executes safe instantiation setups using custom parameter maps.

        :param name: Registry key, reader class, or ready reader instance.
        :type name: Any
        :param kwargs: Property keyword attributes delegated directly into class loaders.
        :return: Initialized concrete implementation sub-type inheriting from MSIBaseReader.
        :rtype: MSIBaseReader
        :raises ProjectConfigError: If no reader matches the requested name or
            required constructor parameters are missing.
        """
        return resolve_component(
            target=name,
            registry=cls.REGISTRY,
            component_type="Reader",
            expected_type=MSIBaseReader,
            **kwargs,
        )

    # --------------------------------------------------
    # Section: Automated Strategy Discovery
    # --------------------------------------------------

    @classmethod
    def discover_strategies(cls) -> None:
        """
        Explicitly imports local strategy packages. 
        Dynamic scanning inside strategies/__init__.py triggers automatic registration.
        """
        discover_modules(f"{__package__}.strategies")
