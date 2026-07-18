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
