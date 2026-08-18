"""
Central registration gateway and factory for managing Mass Spectrometry Imaging PyTorch dataset objects.
"""

from typing import Any, Dict, Type
from ...utils.logger import get_custom_logger
from .base_dataset import MSIBaseDataset
from ...utils.module_search import discover_modules
from ...utils.validators import resolve_component, validate_subclass
from ...utils.exceptions import raise_validation_error

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
            logger.debug(
                "Successfully appended dataset blueprint token '%s' into manager registry.",
                name,
            )
            return subclass

        return decorator

    @classmethod
    def get_dataset(cls, name: Any, **kwargs: Any) -> MSIBaseDataset:
        """
        Resolves dataset strategy classes and executes initialization setups using custom parameters.

        :param name: Registry key, dataset class, or ready dataset instance.
        :type name: Any
        :param kwargs: Initialization parameters passed onto factory constructors.
        :return: Initialized concrete dataset instance complying with abstract contracts.
        :rtype: msi_autoencoder_wrapper.models.datasets.base_dataset.MSIBaseDataset
        :raises ProjectConfigError: If the dataset is unknown or required
            constructor parameters are missing.
        """
        logger.info(
            "Resolving and instantiating dataset strategy '%s' from global registry.",
            name,
        )
        dataset = resolve_component(
            target=name,
            registry=cls._REGISTRY,
            component_type="Dataset",
            expected_type=MSIBaseDataset,
            **kwargs,
        )
        subset = kwargs.get("subset")
        if subset is not None:
            if not isinstance(subset, dict):
                raise_validation_error(
                    "DatasetConfiguration", "subset must be a dictionary or null."
                )
            dataset.subset(subset)
        return dataset

    @classmethod
    def load_config(
        cls,
        config: Dict[str, Any],
        active_context: Any = None,
        cohort_context: Any = None,
    ) -> MSIBaseDataset:
        """Restore a dataset from its module-owned portable configuration.

        :param config: Portable dataset component descriptor.
        :type config: Dict[str, Any]
        :param active_context: Active image context injected at runtime.
        :type active_context: Any
        :return: Restored dataset instance.
        :rtype: MSIBaseDataset
        :raises ValidationError: If the dataset descriptor is invalid.
        """
        if not isinstance(config, dict) or not config.get("type"):
            raise_validation_error(
                "DatasetConfiguration", "A dataset type is required."
            )
        parameters = config.get("parameters", {})
        if not isinstance(parameters, dict):
            raise_validation_error(
                "DatasetConfiguration", "Dataset parameters must be a dictionary."
            )
        cls.discover_strategies()
        dependencies = {}
        if str(config["type"]).startswith("Cohort"):
            dependencies["cohort_context"] = cohort_context
        else:
            dependencies["active_context"] = active_context
        return cls.get_dataset(name=config["type"], **dependencies, **parameters)

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
