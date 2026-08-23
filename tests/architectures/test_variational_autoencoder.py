"""Tests for the AutoMSI-inspired native PyTorch VAE components."""

from __future__ import annotations

import torch
import torch.nn as nn

from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)
from msi_autoencoder_wrapper.models.model_loader import ModelLoader
from msi_autoencoder_wrapper.configuration import get_component_config


def test_variational_components_round_trip_through_recursive_model_config() -> None:
    ArchitecturesManager.discover_architectures()
    model = ArchitecturesManager.build_model(
        "autoencoder",
        {
            "encoder": {
                "type": "VariationalLinearEncoder",
                "params": {"input_dim": 16, "latent_dim": 3, "hidden_dims": [8]},
            },
            "decoder": {
                "type": "LinearSpectrumDecoder",
                "params": {
                    "latent_dim": 3,
                    "output_dim": 16,
                    "hidden_dims": [8],
                    "output_activation": {"type": "softplus", "parameters": {}},
                },
            },
        },
    )
    model.eval()
    outputs = model(torch.rand(4, 16))
    descriptor = get_component_config(model)["parameters"]
    config = {
        "model": {
            "name": "vae",
            "type": "autoencoder",
            "parameters": descriptor["parameters"],
            "components": descriptor["components"],
        }
    }
    restored, model_type, name = ModelLoader.build(config)

    assert outputs["latent_mean"].shape == (4, 3)
    assert outputs["latent_log_variance"].shape == (4, 3)
    assert outputs["reconstruction"].shape == (4, 16)
    assert any(isinstance(layer, nn.LayerNorm) for layer in model.encoder.modules())
    assert not any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder.modules())
    assert model_type == "autoencoder"
    assert name == "vae"
    assert set(restored(torch.rand(2, 16))) >= {
        "latent_space",
        "latent_mean",
        "latent_log_variance",
        "reconstruction",
    }
