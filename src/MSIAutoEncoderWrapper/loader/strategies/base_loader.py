from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Any


class MSIBaseLoader(ABC):
    """
    Abstract Base Class outlining operational requirements for parsing Mass Spectrometry Imaging storage containers.
    
    .. note::
       Loaders are strictly bounded to storage access mechanisms and spatial tracking logic,
       remaining completely separated from machine learning runtime types.
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
    def __len__(self) -> int:
        """
        Computes total flat spatial units contained inside the target file.

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
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        pass

    @abstractmethod
    def GetSpectrumPosition(self, idx: int) -> tuple[int, int, int]:
        """
        Decodes flat array tracking sequences back into authentic 3D spatial pixel coordinates.

        :param idx: Flat position tracking sequence integer index.
        :type idx: int
        :return: Aligned discrete position coordinates mapping array across spatial axes [X, Y, Z].
        :rtype: tuple(int, int, int)
        """
        pass

    @property
    @abstractmethod
    def total_pixels(self) -> int:
        """
        Property reporting total matrix footprint calculations.

        :return: Quantity tracking total spatial matrix records available for evaluation.
        :rtype: int
        """
        pass