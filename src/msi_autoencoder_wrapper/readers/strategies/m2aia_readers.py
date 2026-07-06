from pathlib import Path
import numpy as np
import m2aia as m2
from .base_reader import MSIBaseReader
from ..readers_manager import ReaderManager
from ...utils.logger import get_logger

logger = get_logger(__name__)


@ReaderManager.register_loader("M2aiaReader")
class M2aiaReader(MSIBaseReader):
    """
    Concrete data loader adapter linking native binary C++ M2aia bindings to the library ecosystem.
    
    This strategy handles file initialization, raw binary buffer parsing, and spatial tracking
    for standard imzML structures wrapped by the M2aia processing API.
    """

    def __init__(self, file_path: Path | str) -> None:
        """
        Executes binary interface connection routines targeting storage targets on disk.

        :param file_path: Exact file storage path pointing to targeted imzML files.
        :type file_path: pathlib.Path | str
        :raises RuntimeError: If native m2aia engine fails to execute load sequences.
        """
        # Anchor parent dependencies
        super().__init__(file_path)
        
        # Runtime signaling tracking telemetry
        logger.info(f"Initializing M2aia native image reader on file target: {file_path}")
        
        try:
            self._img = m2.ImzMLReader(str(self.file_path))
            self._img.LoadImage()
        except Exception as e:
            from ...core.utils.exceptions import ProjectConfigError
            raise ProjectConfigError(f"Critical M2aia engine failure opening target file {file_path}: {e}")

    def GetXMin(self) -> float:
        """
        Extracts boundary starting values from continuous mass spectrometry profiling axis.

        :return: First evaluation scalar mass value entry.
        :rtype: float
        """
        return float(self._img.GetXAxis()[0])
    
    def GetXMax(self) -> float:
        """
        Extracts maximum ending parameters bound within continuous mass spectrum axis.

        :return: Terminal validation mass boundary entry.
        :rtype: float
        """
        return float(self._img.GetXAxis()[-1])
    
    def GetXAxis(self) -> np.ndarray:
        """
        Retrieves continuous alignment array vectors containing precise mass-to-charge configurations.

        :return: Aligned physical mass-to-charge spectrometry channels coordinate system array.
        :rtype: np.ndarray
        """
        return np.array(self._img.GetXAxis(), dtype=np.float32)
    
    def GetXAxisDepth(self) -> int:
        """
        Queries native binary structures to extract absolute spectrum measurement channel depth metrics.

        :return: Cumulative capacity capability limit configuration tracking metric integer.
        :rtype: int
        """
        return int(self._img.GetXAxisDepth())

    def __len__(self) -> int:
        """
        Computes total absolute count of functional spectra (pixels) recorded within tissue borders.

        :return: Multi-dimensional space flat coordinate ceiling matrix size boundary tracking integer.
        :rtype: int
        """
        return int(self._img.GetSize()[0] * self._img.GetSize()[1] * self._img.GetSize()[2])

    def get_raw_spectrum(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts original irregular mass spectrometry data matrices directly from binary indices pointers.

        :param idx: Flat coordinate position sequential pointer targeting a single active tissue tracking node.
        :type idx: int
        :return: Aligned mass spectrometry response parameters arrays tracking tuple pairings (xs, ys).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        xs = self._img.GetXValues(idx)
        ys = self._img.GetYValues(idx)
        if xs is None or ys is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        return xs.astype(np.float32), ys.astype(np.float32)

    def GetSpectrumPosition(self, idx: int) -> tuple[int, int, int]:
        """
        Queries low-level C++ file headers to map flat pointer arrays context hooks back onto biological coordinates.

        :param idx: Flat vector alignment registration index tracker element.
        :type idx: int
        :return: Array sequence mapping values containing spatial discrete physical coordinates [X, Y, Z].
        :rtype: tuple[int, int, int]
        """
        pos = self._img.GetSpectrumPosition(idx)
        return int(pos[0]), int(pos[1]), int(pos[2])

    @property
    def native_reader(self) -> m2.ImzMLReader:
        """
        Exposes direct reference connections for executing low-level native commands.

        :return: Active low-level underlying C++ bridge instance handle reference.
        :rtype: m2.ImzMLReader
        """
        return self._img