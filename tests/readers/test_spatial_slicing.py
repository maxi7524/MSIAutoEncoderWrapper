"""Tests for spatial slices and wrapper-wide coordinate conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.utils.exceptions import ValidationError
from tests.mocks.components import MockMSIReader


def test_reader_slices_use_xy_coordinates_by_default(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Numeric slices select source coordinate values in XY order."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    reader = MockMSIReader(msi_fixture_path, active_context=wrapper.active_context)

    region = reader[2:4, 1:2]

    assert set(region) == {(2, 1, 1), (3, 1, 1)}
    np.testing.assert_array_equal(region[(2, 1, 1)][1], reader.GetSpectrum(1)[1])


def test_matrix_coordinate_order_reverses_user_axes_only(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Matrix row-column access maps to storage Y-X without changing raw positions."""
    wrapper = MSIAutoEncoderWrapper(
        project_path=str(tmp_path),
        coordinate_order="matrix",
    )
    reader = MockMSIReader(msi_fixture_path, active_context=wrapper.active_context)

    region = reader[1:2, 2:4]
    _, selected = reader[1, 2]

    assert set(region) == {(1, 2, 1), (1, 3, 1)}
    np.testing.assert_array_equal(selected, reader.GetSpectrum((2, 1, 1))[1])
    assert reader.GetSpectrumPosition(1) == (2, 1, 1)


def test_coordinate_order_can_be_changed_and_is_globally_validated(tmp_path: Path) -> None:
    """The wrapper convention is mutable and invalid values use global errors."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))

    wrapper.set_coordinate_order("matrix")
    assert wrapper.coordinate_order == "matrix"
    with pytest.raises(ValidationError, match="coordinate_order"):
        wrapper.set_coordinate_order("yx")


def test_reader_validates_coordinate_overrides_and_zero_slice_steps(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Reader-level spatial errors use the global validation format."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    reader = MockMSIReader(msi_fixture_path, active_context=wrapper.active_context)

    with pytest.raises(ValidationError, match="coordinate_order"):
        reader.get_region(coordinate_order="yx")
    with pytest.raises(ValidationError, match="step cannot be zero"):
        reader.get_region(slice(None, None, 0))
    with pytest.raises(ValidationError, match="two or three values"):
        reader.get_spectrum_at((1, 2, 3, 4))


def test_reader_maps_spectrum_values_to_native_grid(msi_fixture_path: Path) -> None:
    """One scalar per spectrum preserves coordinate positions and missing pixels."""
    reader = MockMSIReader(msi_fixture_path)
    values = np.arange(reader.GetNumberOfSpectra(), dtype=np.float32)

    image = reader.MapSpectrumValuesToImage(values)

    assert image.values.shape == (1, 2, 3)
    assert image.extent == (1, 3, 1, 2, 1, 1)
    assert image.valid_mask.all()
    assert image.values[0, 1, 2] == pytest.approx(values[5])


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("mean", 15.0), ("sum", 30.0), ("max", 20.0), ("median", 15.0)],
)
def test_reader_ion_image_supports_window_aggregations(
    aggregation: str,
    expected: float,
) -> None:
    """Raw ion images expose consistent provider-independent aggregation semantics."""

    class SmallReader(MockMSIReader):
        def GetNumberOfSpectra(self) -> int:
            return 1

        def GetSpectrum(self, target):
            del target
            return np.array([99.95, 100.0, 100.1]), np.array([10.0, 20.0, 30.0])

        def GetSpectrumPosition(self, idx: int):
            assert idx == 0
            return 1, 1, 1

    image = SmallReader().GetIonImage(100.0, 0.06, aggregation=aggregation)

    assert image.values[0, 0, 0] == pytest.approx(expected)
