from abc import ABC, abstractmethod
import numpy as np
from ..binners_strategies.base_binner import MSIBaseBinner
from typing import Any 

class MSIBaseInverseBinner(ABC):
    """
    Abstract Base Class establishing architectural interfaces for reverse signal reconstructors.
    
    .. note::
       Inverse binners operate on synthetic model predictions to filter or reconstruct empirical
       ion positions back into alternative coordinates or reduced sparse lists.
    """

    def __init__(self, binner: MSIBaseBinner) -> None:
        """
        Binds the structural forward master grid tracking object to the inverse processing pipeline.

        :param binner: Active forward binner strategy tracking structural grid geometry.
        :type binner: msi_lib.binners.binners_strategies.base_binner.MSIBaseBinner
        """
        # Structural binding sequence
        ## Retain reference to forward component to coordinate matching coordinate retrieval steps
        self._Binner = binner
        ## Isolated config mapping for parameter encapsulation
        self._config: dict[str, Any] = {}

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