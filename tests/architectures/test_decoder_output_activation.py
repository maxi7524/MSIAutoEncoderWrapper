"""Tests for configurable spectrum decoder output activations."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from msi_autoencoder_wrapper.models.architectures.types.autoencoders.decoders.cnn_decoder import (
    CNNDecoder,
)
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.decoders.linear_spectrum_decoder import (
    LinearSpectrumDecoder,
)
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.decoders.output_activation import (
    SUPPORTED_OUTPUT_ACTIVATIONS,
    build_output_activation,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def test_supported_output_activations_are_declared_explicitly() -> None:
    """Expose the complete accepted configuration vocabulary in one registry."""
    assert set(SUPPORTED_OUTPUT_ACTIVATIONS) == {
        "relu",
        "sigmoid",
        "softplus",
    }


def test_output_activation_factory_passes_pytorch_parameters() -> None:
    """Forward activation-specific parameters from configuration to PyTorch."""
    activation = build_output_activation(
        {"type": "softplus", "parameters": {"beta": 2.0, "threshold": 10.0}}
    )

    assert isinstance(activation, nn.Softplus)
    assert activation.beta == 2.0
    assert activation.threshold == 10.0


def test_output_activation_factory_reports_supported_values() -> None:
    """Reject unknown configuration values with the accepted names in the error."""
    with pytest.raises(
        ValidationError,
        match="relu, sigmoid, softplus",
    ):
        build_output_activation({"type": "unknown", "parameters": {}})


@pytest.mark.parametrize("activation_name", sorted(SUPPORTED_OUTPUT_ACTIVATIONS))
def test_every_output_activation_is_nonnegative(activation_name: str) -> None:
    """Every activation admitted for an MSI spectrum decoder preserves its domain."""
    activation = build_output_activation(
        {"type": activation_name, "parameters": {}}
    )

    output = activation(torch.tensor([-10.0, 0.0, 10.0]))

    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)


def test_cnn_decoder_uses_configured_final_activation() -> None:
    """Apply the configured activation after the final transposed convolution."""
    decoder = CNNDecoder(
        latent_dim=2,
        spatial_dims=[8, 3],
        channels=[1, 2],
        kernels=[3],
        strides=[2],
        output_activation={"type": "relu", "parameters": {}},
    )
    final_convolution = decoder.deconv_blocks[-1][0]
    nn.init.zeros_(final_convolution.weight)
    nn.init.constant_(final_convolution.bias, -1.0)

    output = decoder(torch.zeros(2, 2))

    assert torch.equal(output, torch.zeros_like(output))
    assert decoder.get_config()["output_activation"]["type"] == "relu"


def test_linear_decoder_uses_configured_final_activation() -> None:
    """Use the same activation configuration interface in the linear decoder."""
    decoder = LinearSpectrumDecoder(
        latent_dim=2,
        output_dim=5,
        hidden_dims=[4],
        output_activation={"type": "sigmoid", "parameters": {}},
    )

    output = decoder(torch.randn(3, 2))

    assert torch.all(output > 0.0)
    assert torch.all(output < 1.0)
    assert decoder.get_config()["output_activation"]["type"] == "sigmoid"
