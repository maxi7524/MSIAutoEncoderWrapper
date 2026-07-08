from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Any, Optional, Dict, Tuple, Union


class MSIBaseReader(ABC):
    """
    Abstract Base Class outlining operational requirements for parsing Mass Spectrometry Imaging containers.
    """

    def __init__(self, file_path: Path | str, active_context: Optional[Any] = None) -> None:
        """
        Initializes foundational file tracking reference handles.

        :param file_path: Target disk storage file path targeting mass spectrometry image data.
        :type file_path: pathlib.Path | str
        :param active_context: Active execution session proxy tracking live datasets. Defaults to None.
        :type active_context: Optional[Any]
        """
        # Explicit conversion routing
        ## Coerce the input path into a concrete Path object to standardize path format across OS platforms
        self.file_path = Path(file_path)
        ## Package foundational configuration metadata details for serialization
        self._config: dict[str, Any] = {"file_path": str(file_path)}
        ## Store the structural active context reference hook to ensure design uniformity
        self.active_context = active_context


    def GetConfig(self) -> dict[str, Any]:
        """
        Exposes internal parameter dictionaries required for automated pipeline construction.

        :return: Storage configuration definition parameters metadata map.
        :rtype: dict
        """
        return self._config

# --------------------------------------------------
# Section: Abstract methods
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: X Axis global statistics 
    # --------------------------------------------------

    @abstractmethod
    def GetXMin(self) -> float:
        """
        Extracts boundary starting thresholds recorded across continuous mass spectrum profiles.
        """
        pass

    @abstractmethod
    def GetXMax(self) -> float:
        """
        Extracts boundary terminal thresholds recorded across continuous mass spectrum profiles.
        """
        pass

    @abstractmethod
    def GetXAxis(self) -> np.ndarray:
        """
        Retrieves the unified master calibration grid alignment vector tracking arrays.
        """
        pass

    @abstractmethod
    def GetXAxisDepth(self) -> int:
        """
        Queries cumulative absolute spectral dimension capacities configured across spatial profiles.
        """
        pass

    # --------------------------------------------------
    # Subsection: Location & spectra dimensions
    # --------------------------------------------------

    @abstractmethod
    def GetSpectrum(self, target: Union[int, Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the spectrum data (xs, ys) using either a flat integer index or spatial coordinates (x, y, z).

        :param target: Flat sequence integer index or a tuple containing discrete coordinates [X, Y, Z].
        :type target: Union[int, Tuple[int, int, int]]
        :return: Aligned mass spectrometry tracking tuple pairing (xs, ys).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        pass

    @abstractmethod
    def GetSpectrumPosition(self, idx: int) -> Tuple[int, int, int]:
        """
        Decodes flat array tracking sequences back into authentic 3D spatial pixel coordinates.

        :param idx: Flat position tracking sequence integer index.
        :type idx: int
        :return: Aligned discrete position coordinates mapping array across spatial axes [X, Y, Z].
        :rtype: Tuple[int, int, int]
        """
        pass

    @abstractmethod
    def GetNumberOfSpectra(self) -> int:
        """
        Returns the total number of spectra (pixels) available in the dataset.

        :return: Total spectrum density count.
        :rtype: int
        """
        pass

    # --------------------------------------------------
    # Subsection: Others
    # --------------------------------------------------

    @abstractmethod
    def GetMetaData(self) -> Dict[str, Any]:
        """
        Queries file headers to compile fundamental metadata descriptions.

        :return: Dictionary containing internal properties and spatial layout limits.
        :rtype: Dict[str, Any]
        """
        pass

    