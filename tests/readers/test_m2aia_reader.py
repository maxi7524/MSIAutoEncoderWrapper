"""Integration tests for M2aiaReader using the compact real-data fixture."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pytest


class _ContinuousNativeReader:
    """Minimal pyM²aia-compatible reader recording native batch calls."""

    instances = 0

    def __init__(self, _: str) -> None:
        type(self).instances += 1
        self.batch_calls = 0

    def GetSpectrumType(self) -> str:
        return "ContinuousProfile"

    def GetSpectra(self, indices: np.ndarray) -> np.ndarray:
        self.batch_calls += 1
        return np.stack([indices, indices + 1], axis=1).astype(np.float32)

    def GetXAxis(self) -> np.ndarray:
        return np.array([100.0, 101.0], dtype=np.float64)


def test_m2aia_reader_loads_compact_bladder_fixture(
    msi_fixture_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The production reader loads spectra and coordinates from the small fixture."""
    monkeypatch.chdir(tmp_path)
    from msi_autoencoder_wrapper.readers.strategies.m2aia_readers import M2aiaReader

    reader = M2aiaReader(msi_fixture_path)

    mass_axis, intensities = reader.GetSpectrum((0, 0, 0))

    assert reader.GetNumberOfSpectra() == 6
    assert reader.GetSpectrumPosition(3) == (0, 1, 0)
    assert len(mass_axis) == len(intensities) == 1129
    assert np.isfinite(mass_axis).all()
    assert np.isfinite(intensities).all()


def test_m2aia_reader_exposes_native_spatial_operations(
    msi_fixture_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The adapter maps values and uses the native maximum ion-image operation."""
    monkeypatch.chdir(tmp_path)
    from msi_autoencoder_wrapper.readers.base_reader import MSIBaseReader
    from msi_autoencoder_wrapper.readers.strategies.m2aia_readers import M2aiaReader

    reader = M2aiaReader(msi_fixture_path)
    mapped = reader.MapSpectrumValuesToImage(
        np.arange(reader.GetNumberOfSpectra(), dtype=np.float32)
    )
    mass_axis, _ = reader.GetSpectrum(0)
    ion_image = reader.GetIonImage(float(mass_axis[0]), 0.0, aggregation="max")
    reference_image = MSIBaseReader.GetIonImage(
        reader,
        float(mass_axis[0]),
        0.0,
        aggregation="max",
    )
    iterated = list(reader.IterSpectra())

    assert mapped.values.shape == mapped.valid_mask.shape
    assert mapped.valid_mask.sum() == reader.GetNumberOfSpectra()
    assert ion_image.values.shape == mapped.values.shape
    np.testing.assert_allclose(
        ion_image.values[ion_image.valid_mask],
        reference_image.values[reference_image.valid_mask],
    )
    assert np.isfinite(ion_image.values[ion_image.valid_mask]).all()
    assert len(iterated) == reader.GetNumberOfSpectra()
    assert iterated[0][0] == 0
    assert iterated[0][1] == reader.GetSpectrumPosition(0)


def test_m2aia_reader_uses_one_native_call_for_continuous_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuous spectra bypass per-sample Python and preserve one shared axis."""
    from msi_autoencoder_wrapper.readers.strategies import m2aia_readers

    _ContinuousNativeReader.instances = 0
    monkeypatch.setattr(m2aia_readers.m2, "ImzMLReader", _ContinuousNativeReader)
    reader = m2aia_readers.M2aiaReader(tmp_path / "continuous.imzML")

    batch = reader.GetSpectrumBatch([3, 7])

    assert reader.capabilities.native_batch_read
    assert batch.shared_mass_axis
    assert reader.native_reader.batch_calls == 1
    np.testing.assert_array_equal(batch.sample_ids, np.array([3, 7]))
    np.testing.assert_allclose(batch.intensities, np.array([[3, 4], [7, 8]]))


def test_m2aia_reader_reopens_native_handle_after_process_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker process never reuses the parent C++ reader handle."""
    from msi_autoencoder_wrapper.readers.strategies import m2aia_readers

    _ContinuousNativeReader.instances = 0
    monkeypatch.setattr(m2aia_readers.m2, "ImzMLReader", _ContinuousNativeReader)
    reader = m2aia_readers.M2aiaReader(tmp_path / "continuous.imzML")
    worker_pid = os.getpid() + 1000
    monkeypatch.setattr(m2aia_readers.os, "getpid", lambda: worker_pid)

    reader.GetSpectrumBatch([0])

    assert _ContinuousNativeReader.instances == 2
