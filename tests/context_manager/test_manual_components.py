"""Tests for ready component objects in the active image setup API."""

from __future__ import annotations

from pathlib import Path

import pytest

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.utils.exceptions import ValidationError
from tests.mocks.components import MockMSIReader


def test_context_setters_accept_ready_reader_and_binners(
    tmp_path: Path,
    msi_fixture_path: Path,
    monkeypatch,
) -> None:
    """User-created image components are stored without re-instantiation."""
    monkeypatch.chdir(tmp_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    reader = MockMSIReader(msi_fixture_path)
    binner = BinnerManager.get_binner(
        "LinearBinning",
        bin_step=1.0,
        x_min=reader.GetXMin(),
        x_max=reader.GetXMax(),
    )
    inverse_binner = BinnerManager.get_inverse_binner(
        "TopPeaksInverseBinner",
        binner=binner,
        max_bins=5,
    )

    wrapper.context_manager.set_reader(reader, str(msi_fixture_path))
    wrapper.context_manager.set_binner(binner, str(msi_fixture_path))
    wrapper.context_manager.set_inverse_binner(inverse_binner, str(msi_fixture_path))

    image_config = wrapper.context_manager.config_ledger[msi_fixture_path.stem]
    assert image_config["reader"] is reader
    assert image_config["binner"] is binner
    assert image_config["inverse_binner"] is inverse_binner


def test_reader_configuration_rejects_missing_workspace_image(tmp_path: Path) -> None:
    """An empty workspace remains valid until a reader needs a missing image."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))

    wrapper.workspace.set_active_image("missing-image")
    with pytest.raises(ValidationError, match="does not exist"):
        wrapper.context_manager.set_reader("PyImzMLReader", "missing-image")
