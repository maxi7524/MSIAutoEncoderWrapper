from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Optional

from ..utils.configuration import ConfigurableComponent
from ..utils.exceptions import raise_validation_error


class MSIBaseBinner(ConfigurableComponent, ABC):
    """
    Abstract Base Class establishing structural interface benchmarks for forward spectral binning algorithms.
    """

    def __init__(self, active_context: Optional[Any] = None) -> None:
        """
        Initializes the abstract binner base with an optional active context bridge.

        :param active_context: Active execution session proxy tracking live datasets.
        :type active_context: Optional[Any]
        """
        self._config: dict[str, Any] = {}
        self.active_context = active_context

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

    def transform_batch(self, batch: Any) -> Any:
        """Transform a packed raw batch on its current Torch device.

        :param batch: Packed raw spectrum batch.
        :type batch: RawSpectrumBatch
        :return: Dense spectrum batch on the same device.
        :rtype: SpectrumBatch
        :raises NotImplementedError: If the binner has no Torch batch backend.
        """
        del batch
        raise NotImplementedError(
            f"{type(self).__name__} does not implement batch Torch binning."
        )

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

    def GetMzRangeIndices(self, mz_min: float, mz_max: float) -> np.ndarray:
        """Return grid indices whose centers lie in an inclusive m/z range.

        :param mz_min: Inclusive lower mass bound.
        :type mz_min: float
        :param mz_max: Inclusive upper mass bound.
        :type mz_max: float
        :return: One-dimensional integer grid indices.
        :rtype: numpy.ndarray
        :raises ValidationError: If the range is reversed.
        """
        if mz_min > mz_max:
            raise_validation_error("Binner", "mz_min cannot be greater than mz_max.")
        axis = np.asarray(self.GetXAxis())
        return np.flatnonzero((axis >= float(mz_min)) & (axis <= float(mz_max)))

    def GetBinIndices(self, mz: float, tolerance: float = 0.0) -> np.ndarray:
        """Return grid indices in an absolute tolerance around one m/z value.

        :param mz: Center mass value.
        :type mz: float
        :param tolerance: Non-negative absolute tolerance.
        :type tolerance: float
        :return: One-dimensional integer grid indices.
        :rtype: numpy.ndarray
        :raises ValidationError: If tolerance is negative.
        """
        if tolerance < 0:
            raise_validation_error("Binner", "tolerance cannot be negative.")
        return self.GetMzRangeIndices(float(mz) - tolerance, float(mz) + tolerance)
