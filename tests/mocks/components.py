"""Reusable lightweight component implementations for unit and integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pyimzml.ImzMLParser import ImzMLParser

from msi_autoencoder_wrapper.binners.base_binner import MSIBaseBinner
from msi_autoencoder_wrapper.binners.base_inverse import MSIBaseInverseBinner
from msi_autoencoder_wrapper.models.datasets.base_dataset import MSIBaseDataset
from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)
from msi_autoencoder_wrapper.readers.base_reader import MSIBaseReader

MSI_FIXTURE_PATH = Path(__file__).parent / "msi" / "mouse_urinary_bladder_mock.imzML"


class MockMSIReader(MSIBaseReader):
    """Small reader implementation backed by the real bladder-image fixture."""

    def __init__(self, file_path: Path | str = MSI_FIXTURE_PATH, active_context: Any = None) -> None:
        """Initialize the parser for the compact test image.

        :param file_path: Compact imzML fixture path.
        :type file_path: pathlib.Path | str
        :param active_context: Optional image context reference.
        :type active_context: Any
        """
        super().__init__(file_path=file_path, active_context=active_context)
        self._parser = ImzMLParser(str(self.file_path))
        self._coordinate_lookup = {
            coordinate: index for index, coordinate in enumerate(self._parser.coordinates)
        }

    def GetXMin(self) -> float:
        mass_axis, _ = self.GetSpectrum(0)
        return float(mass_axis[0])

    def GetXMax(self) -> float:
        mass_axis, _ = self.GetSpectrum(0)
        return float(mass_axis[-1])

    def GetXAxis(self) -> np.ndarray:
        mass_axis, _ = self.GetSpectrum(0)
        return mass_axis

    def GetXAxisDepth(self) -> int:
        return len(self.GetXAxis())

    def GetSpectrum(
        self,
        target: Union[int, Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        spectrum_index = target if isinstance(target, int) else self._coordinate_lookup[target]
        mass_axis, intensities = self._parser.getspectrum(spectrum_index)
        return np.asarray(mass_axis), np.asarray(intensities)

    def GetSpectrumPosition(self, idx: int) -> Tuple[int, int, int]:
        return self._parser.coordinates[idx]

    def GetNumberOfSpectra(self) -> int:
        return len(self._parser.coordinates)

    def GetMetaData(self) -> Dict[str, Any]:
        return {
            "fixture": self.file_path.name,
            "source_image": "mouse_urinary_bladder.imzML",
            "source_spectrum_indices": [0, 1, 2, 260, 261, 262],
            "coordinates": list(self._parser.coordinates),
        }


@dataclass
class MockActiveContext:
    """Minimal active-image context shared by component tests."""

    reader: MockMSIReader
    binner: Optional[MSIBaseBinner] = None
    inverse_binner: Optional[MSIBaseInverseBinner] = None


class MockDataset(MSIBaseDataset):
    """Dataset using the compact MSI reader and an injected binner."""

    def _source_length(self) -> int:
        return self.active_context.reader.GetNumberOfSpectra()

    def _get_source_item(self, idx: int) -> Tuple[int, torch.Tensor]:
        mass_axis, intensities = self.active_context.reader.GetSpectrum(idx)
        binned = self.active_context.binner(mass_axis, intensities)
        return idx, torch.as_tensor(binned, dtype=torch.float32)


def build_small_autoencoder() -> torch.nn.Module:
    """Build a small configured autoencoder for runtime and persistence tests.

    :return: Autoencoder accepting 32-feature spectra and producing four latents.
    :rtype: torch.nn.Module
    """
    ArchitecturesManager.discover_architectures()
    return ArchitecturesManager.build_model(
        "autoencoder",
        {
            "encoder": {
                "strategy": "CNNEncoder",
                "params": {
                    "input_dim": 32,
                    "latent_dim": 4,
                    "channels": [1, 2],
                    "kernels": [3],
                    "strides": [2],
                    "spatial_dims": [32, 15],
                },
            },
            "decoder": {
                "strategy": "CNNDecoder",
                "params": {
                    "latent_dim": 4,
                    "channels": [1, 2],
                    "kernels": [3],
                    "strides": [2],
                    "spatial_dims": [32, 15],
                    "output_activation": {"type": "softplus", "parameters": {}},
                },
            },
        },
    )
