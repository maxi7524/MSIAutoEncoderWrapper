"""Tests for stable and serializable pixel-spectrum normalization."""

from __future__ import annotations

import numpy as np
import pytest

from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import (
    PixelDataset,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def test_image_and_latent_sources_have_safe_normalization_defaults() -> None:
    """Images default to TIC scaling while latent components remain unchanged."""
    image_dataset = PixelDataset(source="image")
    latent_dataset = PixelDataset(source="latent")

    normalized = image_dataset._normalize(np.array([1.0, 2.0, 1.0], dtype=np.float32))

    assert normalized.sum() == pytest.approx(1.0)
    assert image_dataset.get_config()["normalization"] == "tic"
    assert latent_dataset.get_config()["normalization"] == "none"


def test_invalid_pixel_normalization_uses_global_validation_error() -> None:
    """Unsupported normalization names use the shared error format."""
    with pytest.raises(ValidationError, match="normalization"):
        PixelDataset(normalization="unsupported")
