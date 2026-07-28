"""Integration tests for the architecture registry and builder contract."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from msi_autoencoder_wrapper.core.mixins.models_manager.proxies.architecture_proxy import (
    ArchitectureProxy,
)
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
    components = ArchitecturesManager._COMPONENT_REGISTRY["autoencoder"]
    assert "CNNEncoder" in components["encoder"]
    assert "CNNDecoder" in components["decoder"]
    assert "LinearProjector" in components["projector"]


def test_build_model_assembles_registered_components() -> None:
    """The manager creates a model graph from registered component descriptors."""
    model = ArchitecturesManager.build_model("autoencoder", _autoencoder_setup())
    inputs = torch.randn(4, 32)

    outputs = model(inputs)

    assert outputs["latent_space"].shape == (4, 4)
    assert outputs["reconstruction"].shape == inputs.shape
    assert outputs["projection"].shape == (4, 3)


def test_build_model_assembles_multiple_named_heads_for_shared_target() -> None:
    setup = _autoencoder_setup()
    setup["heads"] = {
        "condition_linear": {
            "strategy": "LinearClassificationHead",
            "params": {"latent_dim": 4, "output_dim": 2},
        },
        "condition_deep": {
            "strategy": "LinearClassificationHead",
            "params": {"latent_dim": 4, "output_dim": 2, "hidden_dim": 3},
        },
    }

    model = ArchitecturesManager.build_model(
        "autoencoder",
        setup,
        head_specs={
            "condition_linear": {"target_field": "condition"},
            "condition_deep": {"target_field": "condition"},
        },
    )
    outputs = model(torch.randn(5, 32))

    assert outputs["head_condition_linear"].shape == (5, 2)
    assert outputs["head_condition_deep"].shape == (5, 2)
    assert model.head_specs["condition_linear"]["target_field"] == "condition"


def test_build_model_accepts_ready_component_instances() -> None:
    """Ready architecture components are attached without re-instantiation."""
    encoder_setup = _autoencoder_setup()["encoder"]
    encoder_class = ArchitecturesManager._COMPONENT_REGISTRY["autoencoder"]["encoder"]["CNNEncoder"]
    encoder = encoder_class(**encoder_setup["params"])

    model = ArchitecturesManager.build_model(
        "autoencoder",
        {"encoder": {"target": encoder, "params": {}}},
    )

    assert model.encoder is encoder
    assert model(torch.randn(2, 32))["latent_space"].shape == (2, 4)


def test_architecture_proxy_accepts_ready_component_instances() -> None:
    """The user-facing setup stores a ready component as the build target."""
    proxy = ArchitectureProxy(wrapper_ref=SimpleNamespace())
    proxy.active_model_type = "autoencoder"
    encoder_class = ArchitecturesManager._COMPONENT_REGISTRY["autoencoder"]["encoder"]["CNNEncoder"]
    encoder = encoder_class(**_autoencoder_setup()["encoder"]["params"])

    proxy.set_component("encoder", encoder)

    assert proxy._building_buffer["encoder"]["target"] is encoder


def test_built_model_supports_gradient_flow_and_backbone_freezing() -> None:
    """Assembled graphs remain trainable and expose the architecture freeze contract."""
    model = ArchitecturesManager.build_model("autoencoder", _autoencoder_setup())
    inputs = torch.randn(4, 32)

    loss = torch.nn.functional.mse_loss(model(inputs)["reconstruction"], inputs)
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    model.freeze_backbone()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
