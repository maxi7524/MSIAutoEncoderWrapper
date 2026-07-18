from pathlib import Path
import numpy as np
import m2aia as m2
from typing import Optional, Any, Dict, Tuple, Union
from ..base_reader import MSIBaseReader
from ..readers_manager import ReaderManager
from ...utils.logger import get_custom_logger
from ...utils.exceptions import raise_incompatible_interface_error, raise_project_config_error

# Logger initialization
logger = get_custom_logger(__name__)


@ReaderManager.register_loader("M2aiaReader")
class M2aiaReader(MSIBaseReader):
    """
    Concrete data loader adapter linking native binary C++ M2aia bindings to the library ecosystem.
    """

    def __init__(self, file_path: Path | str, active_context: Optional[Any] = None) -> None:
        """
        Executes binary interface connection routines targeting storage targets on disk.

        :param file_path: Exact file storage path pointing to targeted imzML files.
        :type file_path: pathlib.Path | str
        :param active_context: Active execution session proxy tracking live datasets. Defaults to None.
        :type active_context: Optional[Any]
        """
        # Anchor parent dependencies using unified structural constructor signature
        super().__init__(file_path, active_context=active_context)
        
        # Runtime signaling tracking telemetry
        logger.info("Initializing M2aia native image reader on file target: %s", file_path)
        
        try:
            self._img = m2.ImzMLReader(str(self.file_path))
            self._img.Load()
        except Exception as error:
            logger.error("Critical native C++ library exception intercepted during loader initialization.", exc_info=True)
            raise_project_config_error(
                context_name="M2aiaReader",
                message=f"Failed to open target file '{file_path}': {error}",
            )

        # Lazy spatial lookup coordinates database cache
        self._spatial_lookup: Optional[Dict[Tuple[int, int, int], int]] = None

    def GetXMin(self) -> float:
        return float(self._img.GetXAxis()[0])

    def GetXMax(self) -> float:
        return float(self._img.GetXAxis()[-1])

    def GetXAxis(self) -> np.ndarray:
        return self._img.GetXAxis()

    def GetXAxisDepth(self) -> int:
        return int(self._img.GetXAxisDepth())

    def GetNumberOfSpectra(self) -> int:
        return int(self._img.GetNumberOfSpectra())

    def _ensure_spatial_lookup(self) -> None:
        """
        Internal helper compiling a fast O(1) hash coordinate lookup registry mapping 3D points back to flat indices.
        """
        if self._spatial_lookup is not None:
            return

        logger.debug("Building lazy structural spatial index lookup map for native image coordinate matching.")
        self._spatial_lookup = {}
        total_spectra = self.GetNumberOfSpectra()
        
        for idx in range(total_spectra):
            pos_key = self.GetSpectrumPosition(idx)
            self._spatial_lookup[pos_key] = idx

    def GetSpectrum(self, target: Union[int, Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        # Handle coordinate resolution based on input argument type
        if isinstance(target, (int, np.integer)):
            flat_idx = int(target)
        elif isinstance(target, tuple):
            self._ensure_spatial_lookup()
            flat_idx = self._spatial_lookup.get(target) # type: ignore
            if flat_idx is None:
                ### Return an empty spectrum sequence pairing if spatial bounds contain no measurements
                logger.debug("Requested coordinates %s map outside the empirical tissue boundary layout.", target)
                return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        else:
            raise_incompatible_interface_error(
                context_name="M2aiaReader",
                message=(
                    f"Unsupported spectrum target type '{type(target).__name__}'. "
                    "Use an integer index or a three-dimensional coordinate tuple."
                ),
            )

        # Execute direct native low-level C++ vector recovery commands
        xs, ys = self._img.GetSpectrum(flat_idx)
        
        if xs is None or ys is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        return xs.astype(np.float32), ys.astype(np.float32)

    def GetSpectrumPosition(self, idx: int) -> Tuple[int, int, int]:
        pos = self._img.GetSpectrumPosition(idx)
        return int(pos[0]), int(pos[1]), int(pos[2])

    def GetMetaData(self) -> Dict[str, Any]:
        # total_spectra = self.GetNumberOfSpectra()
        
        # # Coordinate matrix extraction block
        # ## Loop through coordinates to establish max bounding box geometry for the layout
        # max_x, max_y, max_z = 0, 0, 0
        # for idx in range(total_spectra):
        #     x, y, z = self.GetSpectrumPosition(idx)
        #     if x > max_x: max_x = x
        #     if y > max_y: max_y = y
        #     if z > max_z: max_z = z

        # return {
        #     "total_pixels": total_spectra,
        #     "x_axis_depth": self.GetXAxisDepth(),
        #     "x_range": (self.GetXMin(), self.GetXMax()),
        #     "spatial_boundaries": {
        #         "max_x": max_x + 1,
        #         "max_y": max_y + 1,
        #         "max_z": max_z + 1
        #     }
        # }
        return self._img.GetMetaData()

    @property
    def native_reader(self) -> m2.ImzMLReader:
        """
        Exposes direct reference connections for executing low-level native commands.

        :return: Active low-level underlying C++ bridge instance handle reference.
        :rtype: m2.ImzMLReader
        """
        return self._img
