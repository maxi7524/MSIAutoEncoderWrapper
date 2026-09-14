"""Specialized PU and abstention heads retain their output contracts."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.models.architectures.types.autoencoders.base_architecture_autoencoder import (
    MSIBaseAutoencoderArchitecture,
)
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.deep_gambler_head import DeepGamblerHead
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.evidential_head import EvidentialHead
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.jerm_head import JERMHead
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.selective_head import SelectiveHead
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.encoders.mlp_encoder import MLPEncoder


def test_jerm_head_accepts_original_spectrum_for_propensity_prediction():
    """The posterior and propensity outputs remain class-aligned."""
    head = JERMHead(
        latent_dim=3,
        output_dim=2,
        propensity_feature_mode="latent_spectrum",
        input_dim=5,
    )
    output = head(torch.randn(4, 3), input_spectrum=torch.randn(4, 5))
    assert output.shape == (4, 2, 2)
    statistic_head = JERMHead(
        latent_dim=3,
        output_dim=2,
        propensity_feature_mode="latent_spectrum_statistics",
    )
    assert statistic_head(torch.randn(4, 3), input_spectrum=torch.rand(4, 5)).shape == (4, 2, 2)


def test_selective_and_gambler_heads_return_three_channels_per_class():
    """Both abstention heads preserve batch and class dimensions."""
    latent = torch.randn(4, 3)
    assert SelectiveHead(3, 2)(latent).shape == (4, 2, 3)
    assert DeepGamblerHead(3, 2)(latent).shape == (4, 2, 3)
    assert EvidentialHead(3, 2)(latent).shape == (4, 2, 2)


def test_autoencoder_passes_original_model_input_to_jerm_spectrum_head():
    """The model graph routes the same unmodified input to the propensity branch."""
    model = MSIBaseAutoencoderArchitecture(
        {
            "encoder": MLPEncoder(input_dim=5, latent_dim=3, hidden_dims=(4,)),
            "heads": {
                "jerm": JERMHead(
                    latent_dim=3,
                    output_dim=2,
                    propensity_feature_mode="spectrum_statistics",
                )
            },
        }
    )
    assert model(torch.rand(4, 5))["head_jerm"].shape == (4, 2, 2)
