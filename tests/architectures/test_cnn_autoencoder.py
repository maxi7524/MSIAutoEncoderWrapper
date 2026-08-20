"""Tests for the configurable one-dimensional convolutional autoencoder."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.presets.cnn_autoencoder_preset import (
    get_cnn_autoencoder_preset,
)


def test_cnn_preset_builds_requested_three_layer_autoencoder(mock_active_context) -> None:
    """The comparison preset preserves dimensions and reconstructs the input width."""
    setup = get_cnn_autoencoder_preset(
        mock_active_context,
        latent_dim=10,
        channels=[1, 32, 16, 8],
        kernels=[10, 7, 5],
        strides=[3, 3, 3],
    )
    model = ArchitecturesManager.build_model("autoencoder", setup)
    model.eval()
    input_dim = mock_active_context.binner.GetXAxisDepth()

    outputs = model(torch.rand(2, input_dim))

    assert outputs["latent_space"].shape == (2, 10)
    assert outputs["reconstruction"].shape == (2, input_dim)
    assert "projector" not in setup
    assert "heads" not in setup
