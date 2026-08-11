"""Tests for the deterministic fully connected MSI autoencoder."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from msi_autoencoder_wrapper.configuration import get_component_config
from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.presets.mlp_autoencoder_preset import (
    get_mlp_autoencoder_preset,
)
from msi_autoencoder_wrapper.models.model_loader import ModelLoader
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def test_mlp_preset_defaults_decoder_depth_to_encoder_depth(mock_active_context) -> None:
    """Omitted decoder depth mirrors the explicitly requested encoder depth."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=5,
        encoder_layers=2,
    )

    assert setup["encoder"]["params"]["num_layers"] == 2
    assert setup["decoder"]["params"]["num_layers"] == 2
    assert setup["encoder"]["params"]["hidden_dim"] == 512
    assert setup["decoder"]["params"]["output_activation"]["type"] == "softplus"


def test_mlp_preset_supports_asymmetric_depths(mock_active_context) -> None:
    """Encoder and decoder hidden-layer counts can be varied independently."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=4,
        encoder_layers=2,
        decoder_layers=1,
        hidden_dim=16,
        output_activation={"type": "sigmoid", "parameters": {}},
    )
    model = ArchitecturesManager.build_model("autoencoder", setup)
    model.eval()

    outputs = model(torch.rand(1, mock_active_context.binner.GetXAxisDepth()))

    assert outputs["latent_space"].shape == (1, 4)
    assert outputs["reconstruction"].shape == (
        1,
        mock_active_context.binner.GetXAxisDepth(),
    )
    assert torch.all((outputs["reconstruction"] > 0) & (outputs["reconstruction"] < 1))
    assert sum(isinstance(layer, nn.Linear) for layer in model.encoder.modules()) == 3
    assert sum(isinstance(layer, nn.Linear) for layer in model.decoder.modules()) == 2


def test_mlp_autoencoder_has_finite_gradients_and_portable_config(
    mock_active_context,
) -> None:
    """The architecture trains and survives recursive configuration loading."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=3,
        encoder_layers=2,
        decoder_layers=2,
        hidden_dim=8,
    )
    model = ArchitecturesManager.build_model("autoencoder", setup)
    inputs = torch.rand(4, mock_active_context.binner.GetXAxisDepth())
    outputs = model(inputs)
    loss = nn.functional.mse_loss(outputs["reconstruction"], inputs)  # ()
    loss.backward()

    descriptor = get_component_config(model)["parameters"]
    restored, model_type, _ = ModelLoader.build(
        {
            "model": {
                "name": "mlp-ae",
                "type": "autoencoder",
                "parameters": descriptor["parameters"],
                "components": descriptor["components"],
            }
        }
    )

    assert torch.isfinite(outputs["reconstruction"]).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model_type == "autoencoder"
    assert restored.encoder.get_config()["num_layers"] == 2
    assert restored.decoder.get_config()["num_layers"] == 2


@pytest.mark.parametrize("invalid_depth", [0, -1, True, 1.5])
def test_mlp_preset_rejects_invalid_decoder_depth(
    mock_active_context,
    invalid_depth: object,
) -> None:
    """Reject non-positive and non-integral decoder depths."""
    with pytest.raises(ValidationError, match="decoder_layers"):
        get_mlp_autoencoder_preset(
            mock_active_context,
            latent_dim=3,
            encoder_layers=1,
            decoder_layers=invalid_depth,
        )


@pytest.mark.parametrize("invalid_depth", [0, -1, True, 1.5])
def test_mlp_preset_rejects_invalid_encoder_depth(
    mock_active_context,
    invalid_depth: object,
) -> None:
    """Reject non-positive and non-integral encoder depths."""
    with pytest.raises(ValidationError, match="encoder_layers"):
        get_mlp_autoencoder_preset(
            mock_active_context,
            latent_dim=3,
            encoder_layers=invalid_depth,
        )
