"""Pure-Python imzML reader for original and latent-space images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from xml.etree import ElementTree

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser

from ..base_reader import MSIBaseReader
from ..readers_manager import ReaderManager
from ...utils.exceptions import raise_incompatible_interface_error, raise_project_config_error


@ReaderManager.register_loader("PyImzMLReader")
class PyImzMLReader(MSIBaseReader):
    """Read imzML/ibd pairs through pyimzML without requiring M²aia."""

    def __init__(self, file_path: Path | str, active_context: Optional[Any] = None) -> None:
        """Open an imzML file lazily through pyimzML.

        :param file_path: Path to the imzML document.
        :type file_path: pathlib.Path | str
        :param active_context: Optional active context for coordinate conventions.
        :type active_context: Optional[Any]
        """
        super().__init__(file_path=file_path, active_context=active_context)
        try:
            self._parser = ImzMLParser(str(self.file_path))
        except Exception as error:
            raise_project_config_error(
                context_name="PyImzMLReader",
                message=f"Failed to open '{self.file_path}': {error}",
            )
        self._coordinate_lookup = {
            tuple(coordinate): index
            for index, coordinate in enumerate(self._parser.coordinates)
        }

    def GetXMin(self) -> float:
        axis, _ = self.GetSpectrum(0)
        return float(np.min(axis))

    def GetXMax(self) -> float:
        axis, _ = self.GetSpectrum(0)
        return float(np.max(axis))

    def GetXAxis(self) -> np.ndarray:
        return self.GetSpectrum(0)[0]

    def GetXAxisDepth(self) -> int:
        return len(self.GetXAxis())

    def GetSpectrum(
        self,
        target: Union[int, Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(target, (int, np.integer)):
            spectrum_index = int(target)
        elif isinstance(target, tuple):
            spectrum_index = self._coordinate_lookup.get(tuple(target))
            if spectrum_index is None:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
        else:
            raise_incompatible_interface_error(
                context_name="PyImzMLReader",
                message="Spectrum target must be an index or three-dimensional coordinate.",
            )
        axis, intensities = self._parser.getspectrum(spectrum_index)
        return np.asarray(axis), np.asarray(intensities)

    def GetSpectrumPosition(self, idx: int) -> Tuple[int, int, int]:
        coordinate = self._parser.coordinates[idx]
        return int(coordinate[0]), int(coordinate[1]), int(coordinate[2])

    def GetNumberOfSpectra(self) -> int:
        return len(self._parser.coordinates)

    def GetMetaData(self) -> Dict[str, Any]:
        return {
            **dict(self._parser.imzmldict),
            "polarity": self._parser.polarity,
            "spectrum_mode": self._parser.spectrum_mode,
            "user_parameters": self._user_parameters(),
        }

    def _user_parameters(self) -> Dict[str, str]:
        """Extract user parameters embedded in the imzML XML document."""
        parameters: Dict[str, str] = {}
        for _, element in ElementTree.iterparse(self.file_path, events=("end",)):
            if element.tag.endswith("userParam"):
                parameters[element.attrib.get("name", "")] = element.attrib.get("value", "")
            element.clear()
        return parameters
