"""Integration tests for the current architecture registry and builder contract."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)


def _autoencoder_setup() -> dict[str, dict[str, object]]:
    """Return a small, symmetric autoencoder component configuration."""
    return {
        "encoder": {
            "strategy": "CNNEncoder",
            "params": {
                "input_dim": 32,
                "latent_dim": 4,
                "channels": [1, 2],
                "kernels": [3],
                "strides": [2],
                "spatial_dims": [32, 15],
            },
        },
        "decoder": {
            "strategy": "CNNDecoder",
            "params": {
                "latent_dim": 4,
                "channels": [1, 2],
                "kernels": [3],
                "strides": [2],
                "spatial_dims": [32, 15],
            },
        },
        "projector": {
            "strategy": "LinearProjector",
            "params": {"latent_dim": 4, "projection_dim": 3},
        },
    }


def test_architecture_discovery_registers_autoencoder_components() -> None:
    """Recursive discovery populates model and scoped component registries."""
    ArchitecturesManager.discover_architectures()

    assert "autoencoder" in ArchitecturesManager._MODEL_REGISTRY
    autoencoder_components = ArchitecturesManager._COMPONENT_REGISTRY["autoencoder"]
    assert "CNNEncoder" in autoencoder_components["encoder"]
    assert "CNNDecoder" in autoencoder_components["decoder"]
    assert "LinearProjector" in autoencoder_components["projector"]


def test_build_model_assembles_registered_components() -> None:
    """The manager creates a model graph from registered component descriptors."""
    model = ArchitecturesManager.build_model(
        model_type="autoencoder",
        components_setup=_autoencoder_setup(),
    )
    inputs = torch.randn(4, 32)

    outputs = model(inputs)

    assert outputs["latent_space"].shape == (4, 4)
    assert outputs["reconstruction"].shape == inputs.shape
    assert outputs["projection"].shape == (4, 3)


def test_built_model_supports_gradient_flow_and_backbone_freezing() -> None:
    """Assembled graphs remain trainable and expose the architecture freeze contract."""
    model = ArchitecturesManager.build_model(
        model_type="autoencoder",
        components_setup=_autoencoder_setup(),
    )
    inputs = torch.randn(4, 32)

    loss = torch.nn.functional.mse_loss(model(inputs)["reconstruction"], inputs)
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    model.freeze_backbone()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
