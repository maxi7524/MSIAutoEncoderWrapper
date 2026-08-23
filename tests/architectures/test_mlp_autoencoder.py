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


def test_mlp_preset_reverses_encoder_dimensions_for_default_decoder(
    mock_active_context,
) -> None:
    """Omitted decoder dimensions mirror the encoder in reverse order."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=5,
        encoder_hidden_dims=[512, 256],
    )

    assert setup["encoder"]["params"]["hidden_dims"] == [512, 256]
    assert setup["decoder"]["params"]["hidden_dims"] == [256, 512]
    assert setup["decoder"]["params"]["output_activation"]["type"] == "softplus"
    assert setup["encoder"]["params"]["normalization"] == "layer"
    assert setup["decoder"]["params"]["normalization"] == "layer"


def test_mlp_autoencoder_defaults_to_batch_independent_layer_normalization(
    mock_active_context,
) -> None:
    """The default dense architecture is independent of batch composition."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=4,
        encoder_hidden_dims=[16],
    )
    model = ArchitecturesManager.build_model("autoencoder", setup)

    assert any(isinstance(layer, nn.LayerNorm) for layer in model.encoder.modules())
    assert any(isinstance(layer, nn.LayerNorm) for layer in model.decoder.modules())
    assert not any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules())


def test_mlp_autoencoder_loads_legacy_batch_normalization_flag(
    mock_active_context,
) -> None:
    """Persisted configurations can still request their original BatchNorm graph."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=4,
        encoder_hidden_dims=[16],
        batch_normalization=True,
    )
    model = ArchitecturesManager.build_model("autoencoder", setup)

    assert any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder.modules())
    assert model.encoder.get_config()["normalization"] == "batch"


def test_mlp_preset_rejects_ambiguous_normalization_configuration(
    mock_active_context,
) -> None:
    """The canonical strategy and legacy alias cannot be combined."""
    with pytest.raises(ValidationError, match="not both"):
        get_mlp_autoencoder_preset(
            mock_active_context,
            latent_dim=4,
            encoder_hidden_dims=[16],
            normalization="layer",
            batch_normalization=False,
        )


def test_mlp_preset_supports_asymmetric_dimension_lists(mock_active_context) -> None:
    """Encoder and decoder layer dimensions can be varied independently."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=4,
        encoder_hidden_dims=[16, 8],
        decoder_hidden_dims=[12],
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
    encoder_linears = [
        layer for layer in model.encoder.modules() if isinstance(layer, nn.Linear)
    ]
    decoder_linears = [
        layer for layer in model.decoder.modules() if isinstance(layer, nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in encoder_linears] == [
        (mock_active_context.binner.GetXAxisDepth(), 16),
        (16, 8),
        (8, 4),
    ]
    assert [(layer.in_features, layer.out_features) for layer in decoder_linears] == [
        (4, 12),
        (12, mock_active_context.binner.GetXAxisDepth()),
    ]


def test_mlp_autoencoder_has_finite_gradients_and_portable_config(
    mock_active_context,
) -> None:
    """The architecture trains and survives recursive configuration loading."""
    setup = get_mlp_autoencoder_preset(
        mock_active_context,
        latent_dim=3,
        encoder_hidden_dims=[16, 8],
        decoder_hidden_dims=[7, 11],
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
    assert restored.encoder.get_config()["hidden_dims"] == [16, 8]
    assert restored.decoder.get_config()["hidden_dims"] == [7, 11]


@pytest.mark.parametrize("invalid_dims", [[], [0], [-1], [True], [1.5], "512"])
def test_mlp_preset_rejects_invalid_decoder_dimensions(
    mock_active_context,
    invalid_dims: object,
) -> None:
    """Reject empty, non-positive, and non-integral decoder dimensions."""
    with pytest.raises(ValidationError, match="decoder_hidden_dims"):
        get_mlp_autoencoder_preset(
            mock_active_context,
            latent_dim=3,
            encoder_hidden_dims=[8],
            decoder_hidden_dims=invalid_dims,
        )


@pytest.mark.parametrize("invalid_dims", [[], [0], [-1], [True], [1.5], "512"])
def test_mlp_preset_rejects_invalid_encoder_dimensions(
    mock_active_context,
    invalid_dims: object,
) -> None:
    """Reject empty, non-positive, and non-integral encoder dimensions."""
    with pytest.raises(ValidationError, match="encoder_hidden_dims"):
        get_mlp_autoencoder_preset(
            mock_active_context,
            latent_dim=3,
            encoder_hidden_dims=invalid_dims,
        )
