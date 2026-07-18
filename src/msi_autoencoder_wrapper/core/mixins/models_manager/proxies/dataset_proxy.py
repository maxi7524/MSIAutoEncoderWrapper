# Heading 1 (Dataset Proxy Implementation)
## Specialized component managing datasets strategies reflection and state buffering configurations

from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING

# Base class and factory imports
from .base_models_manager_proxy import BaseModelsManagerProxy
from .....models.datasets.dataset_manager import DatasetManager
from .....models.datasets.base_dataset import MSIBaseDataset

# Centralized utilities imports
from .....utils.logger import get_custom_logger
from .....utils.validators import validate_component_target
from .....utils.printing import present_available_components

if TYPE_CHECKING:
    pass

# Logger initialization
logger = get_custom_logger(__name__)


class DatasetProxy(BaseModelsManagerProxy):
    """
    Proxy component executing reflection discovery and buffering operations for PyTorch datasets.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the dataset manager proxy.
        """
        super().__init__(*args, **kwargs)
        DatasetManager.discover_strategies()

    # --------------------------------------------------
    # Section: Strategy Discovery
    # --------------------------------------------------

    def get_available_datasets(self, print_return: bool = True, return_value: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Queries registered dataset components to extract baseline documentation metadata.

        :param print_return: Toggles formatted console printing, defaults to True.
        :type print_return: bool
        :param return_value: Returns the structured reflection dictionary if True, defaults to False.
        :type return_value: bool
        :return: Map linking dataset tokens to their constructor signatures and docstrings, or None.
        :rtype: Optional[Dict[str, Dict[str, Any]]]
        """
        return present_available_components(
            registry=DatasetManager._REGISTRY,
            title="Available Dataset Strategies",
            key_label="Dataset",
            print_return=print_return,
            return_value=return_value,
        )

    # --------------------------------------------------
    # Section: Target State Selection
    # --------------------------------------------------

    def set_dataset(self, name: Any, **kwargs: Any) -> None:
        """
        Registers the target sampling strategy token to be constructed during final compilation.

        :param name: Registry key, dataset class, or ready dataset instance.
        :type name: Any
        :param kwargs: Keyword arguments used to initialize the dataset constructor.
        :type kwargs: Any
        """
        validate_component_target(
            target=name,
            registry=DatasetManager._REGISTRY,
            component_type="Dataset",
            expected_type=MSIBaseDataset,
        )

        # State updates
        ## Cache configuration details inside active building buffer maps
        target_name = name if isinstance(name, str) else getattr(name, "__name__", type(name).__name__)
        self._active_dataset_name = target_name
        self._building_buffer["dataset"] = {
            "target": name,
            "strategy": target_name,
            "kwargs": kwargs,
        }
        logger.debug("Buffered active target dataset strategy: %s with parameters: %s", name, kwargs)
