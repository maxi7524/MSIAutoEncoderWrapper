"""Tests for reader and dataset bulk-loading contracts."""

from __future__ import annotations

import numpy as np

from msi_autoencoder_wrapper.data import RawDatasetView
from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import PixelDataset
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader


class _BulkReader:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def GetNumberOfSpectra(self) -> int:
        return 4

    def GetSpectra(self, indices: list[int]):
        self.calls.append(indices)
        return [
            (np.array([100.0, 101.0]), np.array([index, index + 1.0]))
            for index in indices
        ]


class _Context:
    def __init__(self, reader: _BulkReader) -> None:
        self.reader = reader

    def get_data_reader(self, source: str):
        assert source == "image"
        return self.reader


def test_raw_dataset_view_fetches_one_complete_batch_from_reader() -> None:
    """DataLoader batch fetching performs one dataset-level bulk request."""
    reader = _BulkReader()
    dataset = PixelDataset(active_context=_Context(reader), normalization="none")
    view = RawDatasetView(dataset)

    samples = view.__getitems__([2, 0, 3])

    assert reader.calls == [[2, 0, 3]]
    assert [sample.sample_id for sample in samples] == [2, 0, 3]
    np.testing.assert_array_equal(samples[0].mass_values.numpy(), [100.0, 101.0])


def test_pyimzml_bulk_reader_matches_scalar_reader(msi_fixture_path) -> None:
    """Bulk reads preserve spectrum ordering, axes and intensities exactly."""
    reader = PyImzMLReader(msi_fixture_path)
    indices = [2, 0, 5]

    bulk = reader.GetSpectra(indices)
    scalar = [reader.GetSpectrum(index) for index in indices]

    for bulk_spectrum, scalar_spectrum in zip(bulk, scalar):
        np.testing.assert_array_equal(bulk_spectrum[0], scalar_spectrum[0])
        np.testing.assert_array_equal(bulk_spectrum[1], scalar_spectrum[1])
