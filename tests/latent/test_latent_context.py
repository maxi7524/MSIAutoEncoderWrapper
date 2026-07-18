"""Tests for imzML latent persistence and independent latent data access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.latent.imzml_store import LatentImzMLStore
from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import PixelDataset
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader
from msi_autoencoder_wrapper.utils.exceptions import ValidationError
from tests.mocks.components import MockMSIReader, build_small_autoencoder


def _latent_values() -> np.ndarray:
    """Return deterministic signed latent vectors for six source pixels."""
    return np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(6, 4)


def test_latent_store_writes_imzml_ibd_coordinates_and_metadata(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Latent output preserves source positions and embeds source metadata."""
    source_reader = MockMSIReader(msi_fixture_path)
    output_path = LatentImzMLStore.write(
        tmp_path / "latent.imzML",
        _latent_values(),
        source_reader,
    )
    latent_reader = PyImzMLReader(output_path)

    assert output_path.is_file()
    assert output_path.with_suffix(".ibd").is_file()
    assert latent_reader.GetNumberOfSpectra() == source_reader.GetNumberOfSpectra()
    assert [latent_reader.GetSpectrumPosition(i) for i in range(6)] == [
        source_reader.GetSpectrumPosition(i) for i in range(6)
    ]
    np.testing.assert_allclose(latent_reader.GetSpectrum(3)[1], _latent_values()[3])
    metadata = latent_reader.GetMetaData()["user_parameters"]
    assert metadata["msi_autoencoder_wrapper_space"] == "latent"
    assert metadata["msi_autoencoder_wrapper_latent_dimensions"] == "4"
    assert "source_image" in metadata["msi_autoencoder_wrapper_source_metadata"]


def test_latent_context_and_dataset_work_without_original_image(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A latent-only context supports slices and datasets without an image ledger."""
    latent_path = LatentImzMLStore.write(
        tmp_path / "latent_only.imzML",
        _latent_values(),
        MockMSIReader(msi_fixture_path),
    )
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    wrapper.active_context.load_latent(latent_path)
    region = wrapper.active_context.get_region(slice(1, 3), slice(1, 2))
    dataset = PixelDataset(active_context=wrapper.active_context, source="latent")
    _, first_latent = dataset[0]

    assert wrapper.active_context.data_source == "latent"
    assert set(region) == {(1, 1, 1), (2, 1, 1)}
    np.testing.assert_allclose(first_latent.numpy(), _latent_values()[0])
    with pytest.raises(ValidationError, match="No image context has been set"):
        wrapper.active_context.get_data_reader("image")


def test_autoencoder_transform_saves_and_activates_latent_imzml(
    tmp_path: Path,
    msi_fixture_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active autoencoder writes its transform and switches to latent access."""
    monkeypatch.chdir(tmp_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))
    reader = MockMSIReader(msi_fixture_path)
    binner = BinnerManager.get_binner(
        "LinearBinning",
        x_min=0.0,
        x_max=32.0,
        bin_step=1.0,
    )
    wrapper.context_manager.set_reader(reader, str(msi_fixture_path))
    wrapper.context_manager.set_binner(binner, str(msi_fixture_path))
    wrapper.workspace.set_active_image(str(msi_fixture_path))
    wrapper.active_dataset = PixelDataset(active_context=wrapper.active_context, source="image")
    wrapper.models_manager.attach_model(
        build_small_autoencoder(),
        model_name="latent-ae",
        trained=True,
    )

    output_path = wrapper.active_context.save_latent(
        tmp_path / "generated_latent.imzML",
        loader_config={"batch_size": 2},
    )

    assert output_path.with_suffix(".ibd").is_file()
    assert wrapper.active_context.data_source == "latent"
    assert wrapper.active_context.data_reader.GetXAxisDepth() == 4
    assert wrapper.active_context.get_region(slice(1, 4), slice(1, 3))
