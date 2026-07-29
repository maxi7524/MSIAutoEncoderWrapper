from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from collections.abc import Iterator
from typing import Any, Optional, Dict, Tuple, Union

from ..utils.configuration import ConfigurableComponent
from ..utils.exceptions import raise_validation_error
from .spatial import Aggregation, SpatialImage, aggregate_window


class MSIBaseReader(ConfigurableComponent, ABC):
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

    def get_spectrum_at(
        self,
        coordinates: Tuple[int, int] | Tuple[int, int, int],
        coordinate_order: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a spectrum using the configured user coordinate convention.

        :param coordinates: Two- or three-dimensional user coordinates.
        :type coordinates: Tuple[int, int] | Tuple[int, int, int]
        :param coordinate_order: Optional ``xy`` or ``matrix`` override.
        :type coordinate_order: Optional[str]
        :return: Spectrum axis and intensity values.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        if len(coordinates) not in {2, 3}:
            raise_validation_error(
                context_name="SpatialReader",
                message="Spatial coordinates must contain two or three values.",
            )
        first, second, *remaining = coordinates
        z_position = remaining[0] if remaining else self.GetSpectrumPosition(0)[2]
        order = self._resolve_coordinate_order(coordinate_order)
        storage_coordinates = (
            (second, first, z_position)
            if order == "matrix"
            else (first, second, z_position)
        )
        return self.GetSpectrum(storage_coordinates)

    def get_region(
        self,
        first: slice | int = slice(None),
        second: slice | int = slice(None),
        z: slice | int = slice(None),
        coordinate_order: Optional[str] = None,
    ) -> Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]:
        """Return spectra whose coordinates match numeric spatial slices.

        Slice bounds refer to coordinate values, not array offsets. The returned
        keys use the selected user coordinate convention.

        :param first: X coordinate or matrix row selector.
        :type first: slice | int
        :param second: Y coordinate or matrix column selector.
        :type second: slice | int
        :param z: Z coordinate selector.
        :type z: slice | int
        :param coordinate_order: Optional ``xy`` or ``matrix`` override.
        :type coordinate_order: Optional[str]
        :return: Mapping from user coordinates to spectra.
        :rtype: Dict[Tuple[int, int, int], Tuple[numpy.ndarray, numpy.ndarray]]
        """
        order = self._resolve_coordinate_order(coordinate_order)
        region: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for spectrum_index in range(self.GetNumberOfSpectra()):
            x_position, y_position, z_position = self.GetSpectrumPosition(spectrum_index)
            user_coordinates = (
                (y_position, x_position, z_position)
                if order == "matrix"
                else (x_position, y_position, z_position)
            )
            if all(
                self._matches_coordinate(value, selector)
                for value, selector in zip(user_coordinates, (first, second, z))
            ):
                region[user_coordinates] = self.GetSpectrum(spectrum_index)
        return region

    def __getitem__(self, target: Any) -> Any:
        """Read a flat spectrum, coordinate, or sliced spatial region.

        :param target: Flat index, coordinate tuple, index slice, or spatial slices.
        :type target: Any
        :return: One spectrum, a spectrum list, or a coordinate-to-spectrum map.
        :rtype: Any
        """
        if isinstance(target, slice):
            indices = range(self.GetNumberOfSpectra())[target]
            return [self.GetSpectrum(index) for index in indices]
        if isinstance(target, tuple):
            if any(isinstance(item, slice) for item in target):
                selectors = (*target, slice(None), slice(None))[:3]
                return self.get_region(*selectors)
            return self.get_spectrum_at(target)
        return self.GetSpectrum(target)

    def IterSpectra(
        self,
    ) -> Iterator[tuple[int, Tuple[int, int, int], np.ndarray, np.ndarray]]:
        """Yield every spectrum with its stable index and native coordinates.

        :return: Iterator of ``(spectrum_id, coordinates, mz, intensities)``.
        :rtype: Iterator[tuple[int, tuple[int, int, int], numpy.ndarray, numpy.ndarray]]
        """
        for spectrum_id in range(self.GetNumberOfSpectra()):
            mz_axis, intensities = self.GetSpectrum(spectrum_id)
            yield (
                spectrum_id,
                self.GetSpectrumPosition(spectrum_id),
                mz_axis,
                intensities,
            )

    def GetIonImage(
        self,
        mz: float,
        tolerance: float,
        aggregation: Aggregation = "mean",
        fill_value: float = np.nan,
    ) -> SpatialImage:
        """Return a raw ion image on the reader's native spatial grid.

        :param mz: Center of the selected mass window.
        :type mz: float
        :param tolerance: Non-negative absolute window around ``mz``.
        :type tolerance: float
        :param aggregation: Window aggregation strategy.
        :type aggregation: str | Callable[[numpy.ndarray], float]
        :param fill_value: Value assigned to positions without spectra.
        :type fill_value: float
        :return: Ion intensities and the native spatial mask.
        :rtype: SpatialImage
        :raises ValidationError: If the mass window is invalid.
        """
        if tolerance < 0:
            raise_validation_error("IonImage", "tolerance cannot be negative.")
        values = np.zeros(self.GetNumberOfSpectra(), dtype=np.float32)
        for spectrum_id, _, mz_axis, intensities in self.IterSpectra():
            selected = np.abs(mz_axis - float(mz)) <= float(tolerance)
            if np.any(selected):
                values[spectrum_id] = aggregate_window(intensities[selected], aggregation)
        return self.MapSpectrumValuesToImage(values, fill_value=fill_value)

    def MapSpectrumValuesToImage(
        self,
        values: np.ndarray,
        fill_value: float = np.nan,
    ) -> SpatialImage:
        """Map one scalar per spectrum to the reader's native coordinate grid.

        :param values: One-dimensional values ordered by ``spectrum_id``.
        :type values: numpy.ndarray
        :param fill_value: Value assigned to positions without spectra.
        :type fill_value: float
        :return: Values arranged as ``(z, y, x)`` with a validity mask.
        :rtype: SpatialImage
        :raises ValidationError: If the value count differs from the spectrum count.
        """
        array = np.asarray(values)
        count = self.GetNumberOfSpectra()
        if array.ndim != 1 or len(array) != count:
            raise_validation_error(
                "SpatialReader",
                f"Expected one value for each of {count} spectra, got shape {array.shape}.",
            )
        coordinates = [self.GetSpectrumPosition(index) for index in range(count)]
        if not coordinates:
            raise_validation_error("SpatialReader", "Cannot map an empty MSI dataset.")
        xs, ys, zs = zip(*coordinates)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        shape = (max_z - min_z + 1, max_y - min_y + 1, max_x - min_x + 1)
        grid_dtype = (
            array.dtype
            if np.issubdtype(array.dtype, np.floating)
            else np.result_type(array.dtype, np.asarray(fill_value).dtype)
        )
        grid = np.full(shape, fill_value, dtype=grid_dtype)
        valid_mask = np.zeros(shape, dtype=bool)
        for spectrum_id, (x_position, y_position, z_position) in enumerate(coordinates):
            target = (z_position - min_z, y_position - min_y, x_position - min_x)
            grid[target] = array[spectrum_id]
            valid_mask[target] = True
        return SpatialImage(
            values=grid,
            valid_mask=valid_mask,
            extent=(min_x, max_x, min_y, max_y, min_z, max_z),
        )

    def _coordinate_order(self) -> str:
        """Resolve the wrapper-wide coordinate order with an XY fallback."""
        wrapper = getattr(self.active_context, "_wrapper", None)
        return getattr(wrapper, "coordinate_order", "xy")

    def _resolve_coordinate_order(self, override: Optional[str]) -> str:
        """Validate and resolve an optional coordinate-order override."""
        order = override if override is not None else self._coordinate_order()
        if order not in {"xy", "matrix"}:
            raise_validation_error(
                context_name="SpatialReader",
                message="coordinate_order must be either 'xy' or 'matrix'.",
            )
        return order

    @staticmethod
    def _matches_coordinate(value: int, selector: slice | int) -> bool:
        """Return whether one coordinate is selected by a numeric slice."""
        if isinstance(selector, int):
            return value == selector
        step = selector.step if selector.step is not None else 1
        if step == 0:
            raise_validation_error(
                context_name="SpatialReader",
                message="Spatial slice step cannot be zero.",
            )
        if step > 0:
            start = selector.start if selector.start is not None else 0
            stop = selector.stop if selector.stop is not None else value + 1
            return start <= value < stop and (value - start) % step == 0
        start = selector.start if selector.start is not None else value
        stop = selector.stop if selector.stop is not None else value - 1
        return stop < value <= start and (start - value) % abs(step) == 0


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
