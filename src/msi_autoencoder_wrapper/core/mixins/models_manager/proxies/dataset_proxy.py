# Heading 1 (Dataset Proxy Implementation)
## Specialized component managing datasets strategies reflection and state buffering configurations

from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING

# Base class and factory imports
from .base_models_manager_proxy import BaseModelsManagerProxy
from .....models.datasets.dataset_manager import DatasetManager

# Centralized utilities imports
from .....utils.logger import get_custom_logger
from .....utils.exceptions import raise_validation_error
from ....utils.printing import extract_component_signatures, print_formatted_components

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
        # Metadata extraction
        ## Fetch signatures and docstrings directly from the dataset registry
        registry_target = getattr(DatasetManager, "_REGISTRY", {})
        result = extract_component_signatures(registry=registry_target)

        if print_return:
            ## Output formatting
            ### Format and delegate console rendering to the centralized system printer
            print_formatted_components(
                title="Available Dataset Strategies",
                key_label="Dataset",
                components_info=result
            )

        if return_value:
            return result
        return None

    # --------------------------------------------------
    # Section: Target State Selection
    # --------------------------------------------------

    def set_dataset(self, name: str, **kwargs: Any) -> None:
        """
        Registers the target sampling strategy token to be constructed during final compilation.

        :param name: Unique tracking token identifier for the registered dataset strategy.
        :type name: str
        :param kwargs: Keyword arguments used to initialize the dataset constructor.
        :type kwargs: Any
        """
        # Strategy lookup verification
        ## Ensure dataset exists inside central registrations
        if name not in DatasetManager._REGISTRY:
            raise_validation_error(
                context_name="ModelsManager",
                message=f"Dataset strategy '{name}' is unregistered within the DatasetManager registry."
            )

        # State updates
        ## Cache configuration details inside active building buffer maps
        self._active_dataset_name = name
        self._building_buffer["dataset"] = {
            "strategy": name,
            "kwargs": kwargs
        }
        logger.debug("Buffered active target dataset strategy: %s with parameters: %s", name, kwargs)