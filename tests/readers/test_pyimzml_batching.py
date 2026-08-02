"""Tests for locality-aware pyimzML fallback batching."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader


class _Parser:
    intensityOffsets = [300, 100, 200]

    def __init__(self) -> None:
        self.calls: list[int] = []

    def getspectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(index)
        return np.array([float(index)]), np.array([float(index + 10)])


def test_pyimzml_batch_reads_by_offset_and_restores_sampler_order() -> None:
    """Fallback I/O improves locality without changing logical sample alignment."""
    reader = object.__new__(PyImzMLReader)
    reader.file_path = Path("unused.imzML")
    reader._parser = _Parser()
    reader._reader_pid = os.getpid()

    batch = reader.GetSpectrumBatch([0, 1, 2])

    assert reader._parser.calls == [1, 2, 0]
    assert [axis.item() for axis in batch.mass_values] == [0.0, 1.0, 2.0]
    assert [values.item() for values in batch.intensities] == [10.0, 11.0, 12.0]
