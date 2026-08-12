"""Small pyimzML adapter used by dataset operations."""

from pathlib import Path
from typing import Any

from pyimzml.ImzMLParser import ImzMLParser


class PyImzMLReader:
    """Expose the subset of the wrapper reader required by dataset management."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.parser = ImzMLParser(str(self.path))

    def GetNumberOfSpectra(self) -> int:
        return len(self.parser.coordinates)

    def GetSpectrum(self, index: int) -> tuple[Any, Any]:
        return self.parser.getspectrum(index)

    def GetSpectrumPosition(self, index: int) -> tuple[int, int, int]:
        coordinate = self.parser.coordinates[index]
        return tuple(coordinate) if len(coordinate) == 3 else (*coordinate, 1)
