from abc import ABC, abstractmethod
import numpy as np
from typing import Any 

class MSIBaseBinner(ABC):
    """
    Abstract Base Class establishing structural interface for Mass Spectrometry Imaging binners.
    
    .. note::
       All subclasses must enforce a uniform m/z coordinate grid projection layout to guarantee
       input dimension compatibility across Convolutional Neural Network execution steps.
    """
    #TODO(dokumentacja): MSI Binner, przenieść dokumentacje z oryginalnego pliku 

    def __init__(self) -> None:
        """
        Initializes foundational state vectors and configuration mappings for serialization.
        """
        # Metadata storage initialization
        ## Dictionary blueprint intended to encapsulate parameter configurations for replicability
        self._config: dict[str, Any] = {}

    def GetConfig(self) -> dict[str, Any]:
        """
        Retrieves serialized parameter properties required to replicate the instance state.

        :return: Configuration parameters mapping containing structural parameters.
        :rtype: dict
        """
        return self._config

    @abstractmethod
    def __call__(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Transforms irregular spectrum coordinates into the master m/z index array grid.

        :param xs: Flat array containing the raw experimental mass-to-charge (m/z) positions.
        :type xs: np.ndarray
        :param ys: Flat array containing corresponding empirical peak intensity definitions.
        :type ys: np.ndarray
        :return: aligned intensity vector compliant with targeted deep learning input layers.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def GetXMin(self) -> float:
        """
        Retrieves the absolute lower operational boundary of the master m/z coordinate axis.

        :return: Lower bound mass spectrometry index value.
        :rtype: float
        """
        pass

    @abstractmethod
    def GetXMax(self) -> float:
        """
        Retrieves the absolute upper operational boundary of the master m/z coordinate axis.

        :return: Upper bound mass spectrometry index value.
        :rtype: float
        """
        pass

    @abstractmethod
    def GetXAxis(self) -> np.ndarray:
        """
        Retrieves the shared, master coordinate m/z axis alignment array vector.

        :return: One-dimensional vector defining regular grid sampling coordinates.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def GetXAxisDepth(self) -> int:
        """
        Retrieves total structural capacity of the regular grid representing input feature dimensions.

        :return: Absolute count of index positions available on the current operational master axis.
        :rtype: int
        """
        pass