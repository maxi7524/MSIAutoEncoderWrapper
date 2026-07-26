# Heading 1 (Base Models Proxy Implementation)
## Shared base class for all cooperative model manager sub-proxies

from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING
from ...base_wrapper_proxy import BaseWrapperProxy

if TYPE_CHECKING:
    from ....wrapper import MSIAutoEncoderWrapper


class BaseModelsManagerProxy(BaseWrapperProxy):
    """
    Common base class for Models sub-proxies, sharing state and building buffers.
    """

    def __init__(
        self, 
        wrapper_ref: Any,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initializes the shared models proxy parameters.

        :param wrapper_ref: Reference to the coordinating master wrapper instance.
        :type wrapper_ref: Any
        """
        # Cooperative inheritance execution
        super().__init__(wrapper_ref=wrapper_ref, *args, **kwargs)
        
        # Stateful configuration registers shared across sub-proxies
        self.active_model_type: Optional[str] = None
        self._active_model_name: Optional[str] = None
        self._active_dataset_name: Optional[str] = None
        self._building_buffer: Dict[str, Any] = {}
        self._training_config: Optional[Dict[str, Any]] = None
        self._training_history: Optional[Any] = None
        self._training_transient_cache: Dict[str, Any] = {}
