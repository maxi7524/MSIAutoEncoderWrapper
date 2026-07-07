from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Any


class MSIBaseReader(ABC):
    """
    Abstract Base Class outlining operational requirements for parsing Mass Spectrometry Imaging containers.
    
    Provides foundational spectrum geometry properties and hands off grid axis calculations
    directly to assigned active binning engines to ensure strategic optimization.
    """

    def __init__(self, file_path: Path | str) -> None:
        """
        Initializes foundational file tracking reference handles.

        :param file_path: Target disk storage file path targeting mass spectrometry image data.
        :type file_path: pathlib.Path | str
        """
        # Explicit conversion routing
        ## Coerce the input path into a concrete Path object to standardize path format across OS platforms
        self.file_path = Path(file_path)
        ## Package foundational configuration metadata details for serialization
        self._config: dict[str, Any] = {"file_path": str(file_path)}

    def GetConfig(self) -> dict[str, Any]:
        """
        Exposes internal parameter dictionaries required for automated pipeline construction.

        :return: Storage configuration definition parameters metadata map.
        :rtype: dict
        """
        return self._config

    @abstractmethod
    def GetXMin(self) -> float:
        """
        Extracts boundary starting thresholds recorded across continuous mass spectrum profiling sequences.

        :return: First evaluation scalar index stored inside mass axis vectors.
        :rtype: float
        """
        pass

    @abstractmethod
    def GetXMax(self) -> float:
        """
        Extracts maximum ending parameters bound within continuous mass axis arrays.

        :return: Terminal physical registration value located inside spectrum mass vectors.
        :rtype: float
        """
        pass

    @abstractmethod
    def GetXAxis(self) -> np.ndarray:
        """
        Extracts complete explicit coordinate vectors outlining physical mass spectrometry measurements sequences.

        :return: High-performance numpy data matrix referencing full layout mass axis channels.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def GetXAxisDepth(self) -> int:
        """
        Queries cumulative absolute spectral dimension capacities configured across spatial profiles.

        :return: Total mass axis channel array depth capacity value.
        :rtype: int
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Computes total flat spatial spectrum units (pixels) contained inside the target file.

        :return: Absolute measurement tracking cumulative pixel density inside dataset boundaries.
        :rtype: int
        """
        pass

    @abstractmethod
    def get_raw_spectrum(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts original irregular mass spectrometry arrays from explicit storage locations.

        :param idx: Flat coordinate index mapping targeting a unique tissue image pixel position.
        :type idx: int
        :return: Aligned coordinate pairing containing raw empirical vectors (xs, ys).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        pass

    @abstractmethod
    def GetSpectrumPosition(self, idx: int) -> tuple[int, int, int]:
        """
        Decodes flat array tracking sequences back into authentic 3D spatial pixel coordinates.

        :param idx: Flat position tracking sequence integer index.
        :type idx: int
        :return: Aligned discrete position coordinates mapping array across spatial axes [X, Y, Z].
        :rtype: tuple[int, int, int]
        """
        pass

 
 