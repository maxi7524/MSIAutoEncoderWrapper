"""
Small pyimzML adapter used by dataset operations.

It is separated as we do not need additional functionality from mian module 
"""

import warnings
from pathlib import Path
from typing import Any

from pyimzml.ImzMLParser import ImzMLParser


class PyImzMLReader:
    """Expose the subset of the wrapper reader required by dataset management."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        # pyimzML ontology compatibility
        ## Older imzML exports contain legacy CV names. pyimzML corrects them
        ## internally, so their absolute-path UserWarnings add terminal noise
        ## without changing parsing or the resulting spectra.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"pyimzml\.ontology\.ontology",
            )
            self.parser = ImzMLParser(str(self.path))

    def GetNumberOfSpectra(self) -> int:
        return len(self.parser.coordinates)

    def GetSpectrum(self, index: int) -> tuple[Any, Any]:
        return self.parser.getspectrum(index)

    def GetSpectrumPosition(self, index: int) -> tuple[int, int, int]:
        coordinate = self.parser.coordinates[index]
        return tuple(coordinate) if len(coordinate) == 3 else (*coordinate, 1)
