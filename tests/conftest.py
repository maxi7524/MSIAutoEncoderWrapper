"""Shared pytest fixtures backed by the compact bladder-image sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from tests.mocks.components import MSI_FIXTURE_PATH, MockActiveContext, MockMSIReader


@pytest.fixture
def msi_fixture_path() -> Path:
    """Return the compact imzML fixture path."""
    return MSI_FIXTURE_PATH


@pytest.fixture
def mock_reader(msi_fixture_path: Path) -> MockMSIReader:
    """Return a lightweight reader over the compact real-data sample."""
    return MockMSIReader(msi_fixture_path)


@pytest.fixture
def mock_active_context(mock_reader: MockMSIReader) -> MockActiveContext:
    """Return an active context with a reader and regular-grid binner."""
    BinnerManager.discover_strategies()
    context = MockActiveContext(reader=mock_reader)
    context.binner = BinnerManager.get_binner(
        "LinearBinning",
        bin_step=1.0,
        x_min=mock_reader.GetXMin(),
        x_max=mock_reader.GetXMax(),
        active_context=context,
    )
    return context
