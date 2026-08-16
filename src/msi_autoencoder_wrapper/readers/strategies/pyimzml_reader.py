"""Pure-Python imzML reader for original and latent-space images."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from xml.etree import ElementTree

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser

from ..base_reader import MSIBaseReader, ReaderCapabilities, SpectrumReadBatch
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
        self._reader_pid = os.getpid()

    @property
    def capabilities(self) -> ReaderCapabilities:
        """Expose ordered fallback batching and worker-local parser handles."""
        return ReaderCapabilities(requires_worker_reopen=True)

    def _ensure_process_reader(self) -> None:
        """Reopen the parser when execution moves into a DataLoader worker."""
        current_pid = os.getpid()
        if self._reader_pid == current_pid:
            return
        self._parser = ImzMLParser(str(self.file_path))
        self._reader_pid = current_pid

    def GetSpectrumBatch(self, indices: Any) -> SpectrumReadBatch:
        """Read a batch in physical intensity-offset order and restore sampler order."""
        self._ensure_process_reader()
        sample_ids = np.asarray(indices, dtype=np.int64)
        offsets = np.asarray(self._parser.intensityOffsets, dtype=np.int64)[sample_ids]
        read_order = np.argsort(offsets, kind="stable")
        axes: list[np.ndarray | None] = [None] * len(sample_ids)
        values: list[np.ndarray | None] = [None] * len(sample_ids)
        for position in read_order:
            axis, intensities = self._parser.getspectrum(int(sample_ids[position]))
            axes[position] = np.asarray(axis, dtype=np.float32)
            values[position] = np.asarray(intensities, dtype=np.float32)
        return SpectrumReadBatch(
            sample_ids=sample_ids,
            mass_values=tuple(axis for axis in axes if axis is not None),
            intensities=tuple(value for value in values if value is not None),
            shared_mass_axis=False,
        )

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
        self._ensure_process_reader()
        if isinstance(target, (int, np.integer)):
            spectrum_index = int(target)
        elif isinstance(target, tuple):
            spectrum_index = self._coordinate_lookup.get(tuple(target))
            if spectrum_index is None:
                return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        else:
            raise_incompatible_interface_error(
                context_name="PyImzMLReader",
                message="Spectrum target must be an index or three-dimensional coordinate.",
            )
        axis, intensities = self._parser.getspectrum(spectrum_index)
        return np.asarray(axis, dtype=np.float32), np.asarray(intensities, dtype=np.float32)

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
