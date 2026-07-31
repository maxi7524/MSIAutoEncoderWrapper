from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import Optional, Any, Dict, Tuple, Union
from collections.abc import Iterator
from ..base_reader import MSIBaseReader
from ..readers_manager import ReaderManager
from ...utils.logger import get_custom_logger
from ...utils.exceptions import (
    raise_incompatible_interface_error,
    raise_project_config_error,
    raise_validation_error,
)
from ..spatial import Aggregation, SpatialImage

# MPS Conflicts solution
try:
    import m2aia as m2
except ModuleNotFoundError as error:
    if error.name != "m2aia":
        raise
    m2: Any = None

try:
    import SimpleITK as sitk
except ModuleNotFoundError as error:
    if error.name != "SimpleITK":
        raise
    sitk: Any = None

# Logger initialization
logger = get_custom_logger(__name__)

# --------------------------------------------------
# Section: Helpers 
# --------------------------------------------------

# TODO - przenieśc to do rjestru 
def _require_m2aia() -> None:
    """Validate availability of the optional M2aia backend.

    :raises ImportError: If ``m2aia`` or ``SimpleITK`` is unavailable.
    """
    missing_dependencies = []

    if m2 is None:
        missing_dependencies.append("m2aia")

    if sitk is None:
        missing_dependencies.append("SimpleITK")

    if missing_dependencies:
        missing = ", ".join(missing_dependencies)
        raise ImportError(
            "The M2aia reader backend is unavailable because the following "
            f"optional dependencies are missing: {missing}. "
            "Select the PyImzML reader on macOS or install an environment "
            "extra containing the M2aia backend."
        )

# --------------------------------------------------
# Section: Main code 
# --------------------------------------------------

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
        #TODO - przenieść to do rejestru !! 
        _require_m2aia()

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

    def IterSpectra(
        self,
    ) -> Iterator[tuple[int, Tuple[int, int, int], np.ndarray, np.ndarray]]:
        """Yield spectra through the native M²aia iterator."""
        for spectrum_id, mz_axis, intensities in self._img.SpectrumIterator():
            spectrum_id = int(spectrum_id)
            yield (
                spectrum_id,
                self.GetSpectrumPosition(spectrum_id),
                np.asarray(mz_axis, dtype=np.float32),
                np.asarray(intensities, dtype=np.float32),
            )

    def GetIonImage(
        self,
        mz: float,
        tolerance: float,
        aggregation: Aggregation = "mean",
        fill_value: float = np.nan,
    ) -> SpatialImage:
        """Use native M²aia ion-image extraction for maximum aggregation."""
        if aggregation != "max":
            return super().GetIonImage(mz, tolerance, aggregation, fill_value)
        if tolerance < 0:
            raise_validation_error("IonImage", "tolerance cannot be negative.")
        values = sitk.GetArrayFromImage(
            self._img.GetImage(float(mz), float(tolerance))
        ).astype(np.float32, copy=False)
        mask = sitk.GetArrayFromImage(self._img.GetMaskImage()).astype(bool)
        result = np.where(mask, values, fill_value)
        coordinates = [
            self.GetSpectrumPosition(index)
            for index in range(self.GetNumberOfSpectra())
        ]
        xs, ys, zs = zip(*coordinates)
        return SpatialImage(
            values=result,
            valid_mask=mask,
            extent=(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)),
        )

    def MapSpectrumValuesToImage(
        self,
        values: np.ndarray,
        fill_value: float = np.nan,
    ) -> SpatialImage:
        """Map spectrum values with M²aia's native index and mask images."""
        array = np.asarray(values)
        count = self.GetNumberOfSpectra()
        if array.ndim != 1 or len(array) != count:
            return super().MapSpectrumValuesToImage(array, fill_value)
        indices = sitk.GetArrayFromImage(self._img.GetIndexImage()).astype(np.int64)
        mask = sitk.GetArrayFromImage(self._img.GetMaskImage()).astype(bool)
        result_dtype = (
            array.dtype
            if np.issubdtype(array.dtype, np.floating)
            else np.result_type(array.dtype, np.asarray(fill_value).dtype)
        )
        result = np.full(indices.shape, fill_value, dtype=result_dtype)
        result[mask] = array[indices[mask]]
        coordinates = [self.GetSpectrumPosition(index) for index in range(count)]
        xs, ys, zs = zip(*coordinates)
        return SpatialImage(
            values=result,
            valid_mask=mask,
            extent=(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)),
        )

    @property
    def native_reader(self) -> m2.ImzMLReader:
        """
        Exposes direct reference connections for executing low-level native commands.

        :return: Active low-level underlying C++ bridge instance handle reference.
        :rtype: m2.ImzMLReader
        """
        return self._img
