from abc import ABC, abstractmethod
import numpy as np
from .base_binner import MSIBaseBinner
from typing import Any, Optional

class MSIBaseInverseBinner(ABC):
    """
    Abstract Base Class establishing architectural interfaces for reverse signal reconstructors.
    """

    def __init__(self, binner: Optional[MSIBaseBinner] = None, active_context: Optional[Any] = None) -> None:
        """
        Binds the structural forward master grid tracking object to the inverse processing pipeline.

        :param binner: Active forward binner strategy. Falls back to active_context lookup if None.
        :type binner: Optional[MSIBaseBinner]
        :param active_context: Active execution session proxy tracking live datasets.
        :type active_context: Optional[Any]
        :raises ValueError: If no binner can be resolved directly or from the context.
        """
        self._config: dict[str, Any] = {}
        self.active_context = active_context
        
        # Resolve binner instance via direct injection or session context proxy
        if binner is not None:
            self._Binner = binner
        elif active_context and getattr(active_context, "binner", None) is not None:
            self._Binner = active_context.binner
        else:
            raise ValueError("Inverse Binner requires an explicit binner instance or a running active context with a registered binner.")

    def GetConfig(self) -> dict[str, Any]:
        """
        Retrieves serialized configuration dictionary specifications for pipeline preservation.

        :return: Map containing parameters required to instantiate identical inverse components.
        :rtype: dict
        """
        return self._config

    @abstractmethod
    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decodes model array projections back into discrete coordinate pairings.

        :param grid_ys: Dense array representing intensities projected directly on the regular master grid.
        :type grid_ys: np.ndarray
        :return: Matched tuple of arrays (resolved_xs, filtered_ys) containing clean localized predictions.
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        pass