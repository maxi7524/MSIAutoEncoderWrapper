"""
Central registration gateway and factory for managing Mass Spectrometry Imaging PyTorch dataset objects.
"""

from typing import Any, Dict, Type
from ...utils.logger import get_custom_logger
from .base_dataset import MSIBaseDataset
from ...utils.module_search import discover_modules
from ...utils.validators import resolve_component, validate_subclass

# Synchronized telemetry logger initialization
logger = get_custom_logger(__name__)


class DatasetManager:
    """
    Central manager coordinating alternative sampling strategies and dynamic instantiation parameters validation.
    
    Maintains global isolated registries for mapping dataset strategies, allowing
    for automatic resolution of sampling components from dynamic configurations.
    """

    # Global isolated dataset strategy registry mapping string keys to class definitions
    _REGISTRY: Dict[str, Type[MSIBaseDataset]] = {}

    @classmethod
    def register_dataset(cls, name: str) -> Any:
        """
        Decorator factory to register a concrete MSIBaseDataset implementation strategy.

        :param name: Unique lookup token identifier string for the dataset sampling variant.
        :type name: str
        :return: Inner decorator closure wrapping the targeted subclass.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseDataset]) -> Type[MSIBaseDataset]:
            validate_subclass(subclass, MSIBaseDataset, "DatasetRegistry")
            cls._REGISTRY[name] = subclass
            logger.debug("Successfully appended dataset blueprint token '%s' into manager registry.", name)
            return subclass
        return decorator

    @classmethod
    def get_dataset(cls, name: str, **kwargs: Any) -> MSIBaseDataset:
        """
        Resolves dataset strategy classes and executes initialization setups using custom parameters.

        :param name: Target lookup strategy string identifier key.
        :type name: str
        :param kwargs: Initialization parameters passed onto factory constructors.
        :return: Initialized concrete dataset instance complying with abstract contracts.
        :rtype: msi_autoencoder_wrapper.models.datasets.base_dataset.MSIBaseDataset
        :raises ProjectConfigError: If the dataset is unknown or required
            constructor parameters are missing.
        """
        logger.info("Resolving and instantiating dataset strategy '%s' from global registry.", name)
        return resolve_component(
            target=name,
            registry=cls._REGISTRY,
            component_type="Dataset",
            expected_type=MSIBaseDataset,
            **kwargs,
        )

    # --------------------------------------------------
    # Section: Automated Strategy Discovery
    # --------------------------------------------------

    @classmethod
    def discover_strategies(cls) -> None:
        """
        Explicitly imports local strategies packages.
        Dynamic scanning inside their __init__.py files triggers automatic decorator registrations.
        """
        discover_modules(f"{__package__}.strategies")
