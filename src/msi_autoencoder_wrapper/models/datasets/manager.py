from typing import Type, Dict, Any, Union

# Purely relative imports within the package hierarchy
from ...utils.logger import get_custom_logger
from ...readers.strategies.base_reader import MSIBaseReader
from ...binners.binners_strategies.base_binner import MSIBaseBinner
from .strategies.base_dataset import MSIBaseDataset

# Synchronized telemetry logger initialization
logger = get_custom_logger(__name__)


class DatasetManager:
    """
    Central registration gateway and factory for Mass Spectrometry Imaging (MSI) datasets.

    This manager orchestrates alternative sampling strategies, ensuring every instantiated
    component strictly enforces compliance with the baseline MSIBaseDataset abstract contract.
    """

    # Global isolated dataset strategy registry mapping tokens to class blueprints
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
            if not issubclass(subclass, MSIBaseDataset):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from MSIBaseDataset.")
            cls._REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def build_dataset(
        cls,
        setup: Union[dict[str, Any], MSIBaseDataset],
        loader: MSIBaseReader,
        binner: MSIBaseBinner
    ) -> MSIBaseDataset:
        """
        Resolves, type-verifies, and constructs the requested MSI dataset sampling graph node.

        This method supports text-based dictionary configuration parameters resolution
        or plain validation routing for pre-instantiated custom object representations.

        :param setup: Dataset configuration specifications map or an explicit active instance.
        :type setup: dict or MSIBaseDataset
        :param loader: Bound storage driver instance executing file I/O operations.
        :type loader: MSIBaseReader
        :param binner: Active forward transformation component managing grid-x-axis mappings.
        :type binner: MSIBaseBinner
        :return: Structural-type validated dataset engine instance integrated into PyTorch workflows.
        :rtype: MSIBaseDataset
        :raises TypeError: If the compiled object fails verification against MSIBaseDataset contracts.
        :raises KeyError: If the requested dictionary string configuration token is unknown.
        """
        # Resolve concrete target module from configuration blueprints or objects
        if isinstance(setup, dict):
            ds_type = setup["type"]
            if ds_type not in cls._REGISTRY:
                raise KeyError(f"Dataset token '{ds_type}' not found within registries.")
            
            # Extract initialization parameters and inject concrete dependent subcomponents
            dataset_instance = cls._REGISTRY[ds_type](
                loader=loader,
                binner=binner,
                **setup.get("params", {})
            )
        else:
            dataset_instance = setup

        # Strict contractual verification check execution step
        if not isinstance(dataset_instance, MSIBaseDataset):
            raise TypeError("Compiled Dataset component fails contract check against MSIBaseDataset.")

        # Capture complete baseline parameter configurations for serialization pipelines
        dataset_instance._config = {
            "dataset_setup": setup if isinstance(setup, dict) else {},
            "loader_setup": loader.GetConfig(),
            "binner_setup": binner.GetConfig()
        }

        logger.info("MSI PyTorch Dataset strategy compiled, type-verified, and initialized successfully.")
        return dataset_instance