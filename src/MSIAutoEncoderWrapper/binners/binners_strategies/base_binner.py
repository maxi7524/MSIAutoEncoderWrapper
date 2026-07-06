from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class MSIBaseBinner(ABC):
    """
    Abstract Base Class establishing structural interface benchmarks for forward spectral binning algorithms.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def GetConfig(self) -> dict[str, Any]:
        """Exposes configuration parameter properties required for serialization tasks."""
        return self._config

    @abstractmethod
    def __call__(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Projects irregular mass-to-charge spectrometry arrays onto the uniform grid coordinates.

        :param xs: One-dimensional array containing raw experimental mass-to-charge (m/z) positions.
        :type xs: np.ndarray
        :param ys: One-dimensional array containing corresponding raw empirical peak intensity metrics.
        :type ys: np.ndarray
        :return: Evenly mapped normalized intensity response vector.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def GetXMin(self) -> float:
        """Retrieves absolute starting floor mass boundary threshold configured across the shared grid."""
        pass

    @abstractmethod
    def GetXMax(self) -> float:
        """Retrieves absolute terminal ceiling mass boundary threshold configured across the shared grid."""
        pass

    @abstractmethod
    def GetXAxis(self) -> np.ndarray:
        """Retrieves the unified master grid alignment reference array matrix containing m/z points."""
        pass

    @abstractmethod
    def GetXAxisDepth(self) -> int:
        """Computes total absolute feature dimension channels length available within binned spaces."""
        pass