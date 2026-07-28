"""Integration tests for M2aiaReader using the compact real-data fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
