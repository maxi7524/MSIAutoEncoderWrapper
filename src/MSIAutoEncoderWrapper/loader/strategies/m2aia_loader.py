from pathlib import Path
import numpy as np
import m2aia as m2
from msi_lib.loader.strategies.base_loader import MSIBaseLoader
from msi_lib.loader.manager import LoaderManager
from msi_lib.utils.logger import get_custom_logger

# Logger synchronization
## Ingest active telemetry logging component for tracking library operations
logger = get_custom_logger(__name__)


@LoaderManager.register_loader("M2aiaLoader")
class M2aiaLoader(MSIBaseLoader):
    """
    Concrete data loader adapter linking native binary C++ m2aia bindings to the library ecosystem.
    
    This strategy handles file initialization, raw binary buffer parsing, and spatial coordinate
    mapping for standard imzML structures wrapped by the m2aia processing API.
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
        
        # Binary interface setup sequence
        ## Construct native bridge object passing target disk localization path reference
        self._img = m2.ImzMLReader(str(file_path))
        ## Invoke blocking command binding raw files directly into active runtime environments
        self._img.Load()
        
        # Metrics caching step
        ## Extract flat length from indices tracking arrays to optimize lookup execution cycles
        self._total_pixels = len(self._img.GetSpectrumIndices())

    def __len__(self) -> int:
        """
        Exposes dataset length parameter limits derived during construction routines.

        :return: Quantitative size definition tracking total matrix coordinates elements.
        :rtype: int
        """
        return self._total_pixels

    def get_raw_spectrum(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Communicates across memory channels to capture original non-binned spectral definitions.

        :param idx: Unique flat location coordinates tracking pointer index.
        :type idx: int
        :return: Matrix combination containing actual independent mass indices and signal heights.
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        # Low-level memory query
        ## Capture intensity sequence buffer array directly from native C++ pipeline bindings
        ys = self._img.GetSpectrum(idx)
        ## Capture axis definition values describing constant spatial calibration indices
        xs = self._img.GetXAxis()
        
        # Null exception defense gate
        ## Evaluate matrix states to block memory leaks or array format execution corruption
        if ys is None or len(ys) == 0:
            ### Zero-length array allocation fallback
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            
        # Numerical format specification conversion
        ## Enforce 32-bit floating point structures to eliminate memory overhead and maintain PyTorch compatibility
        return xs.astype(np.float32), ys.astype(np.float32)

    def GetSpectrumPosition(self, idx: int) -> tuple[int, int, int]:
        """
        Queries native file headers to map flat pointers back onto biological space.

        :param idx: Flat vector alignment lookup key index.
        :type idx: int
        :return: Geometric array position coordinates values mapping along standard dimensions [X, Y, Z].
        :rtype: tuple(int, int, int)
        """
        # Spatial coordinate extraction execution step
        pos = self._img.GetSpectrumPosition(idx)
        return int(pos[0]), int(pos[1]), int(pos[2])

    @property
    def total_pixels(self) -> int:
        """
        Exposes read-only property tracking full pixel dataset array length metrics.

        :return: Capacity limit configuration tracking total elements density.
        :rtype: int
        """
        return self._total_pixels
    
    @property
    def native_reader(self) -> m2.ImzMLReader:
        """
        Exposes raw unmanaged handle connections for executing non-standard custom commands.

        :return: Active low-level underlying C++ bridge instance handle reference.
        :rtype: m2.ImzMLReader
        """
        return self._img