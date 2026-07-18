"""Tests for automatic autoencoder runtime attachment and loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.utils.exceptions import ValidationError
from tests.mocks.components import MockDataset, build_small_autoencoder


def test_ready_autoencoder_is_detected_and_attached_without_an_image(
    tmp_path: Path,
) -> None:
    """A ready model receives its runtime interface without loading image data."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    model = build_small_autoencoder()

    attached = wrapper.models_manager.attach_model(
        torch_model=model,
        model_name="manual-ae",
        trained=True,
    )
    latent = wrapper.active_context.autoencoder.encode(np.ones(32, dtype=np.float32))
    reconstruction = wrapper.models_manager.autoencoder.decode(latent, grid_xs=True)

    assert attached is model
    assert wrapper.models_manager.active_model_type == "autoencoder"
    assert wrapper.active_context.autoencoder is wrapper.models_manager.autoencoder
    assert latent.shape == (1, 4)
    assert reconstruction.shape == (1, 32)


def test_compiled_autoencoder_is_attached_as_untrained(
    tmp_path: Path,
    mock_active_context,
) -> None:
    """The normal compilation path automatically creates the runtime interface."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    setup = build_small_autoencoder().get_config()["components"]
    wrapper.models_manager.set_model_type("autoencoder", "compiled-ae")
    for category, descriptor in setup.items():
        wrapper.models_manager.set_component(
            category,
            descriptor["type"],
            **descriptor["parameters"],
        )
    wrapper.models_manager.set_dataset(MockDataset(active_context=mock_active_context))

    compiled = wrapper.models_manager.compile_model(run_validation_pass=False)

    assert compiled is wrapper.active_model
    assert wrapper.models_manager.autoencoder is not None
    assert not wrapper.models_manager.autoencoder.is_trained
    with pytest.raises(ValidationError, match="has not been trained"):
        wrapper.models_manager.autoencoder.encode(torch.ones(32))


def test_saved_autoencoder_loads_and_attaches_without_original_image(
    tmp_path: Path,
) -> None:
    """JSON and weights reconstruct a ready autoencoder without an image context."""
    source_wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    source_model = build_small_autoencoder()
    source_wrapper.models_manager.attach_model(
        source_model,
        model_name="roundtrip-ae",
        trained=True,
    )
    source_wrapper.workspace.save_model(img_name="global")
    input_tensor = torch.randn(2, 32)
    expected = source_wrapper.models_manager.autoencoder.encode(input_tensor)

    loaded_wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    loaded = loaded_wrapper.models_manager.load_model("global", "roundtrip-ae")
    actual = loaded_wrapper.active_context.autoencoder.encode(input_tensor)

    assert loaded is loaded_wrapper.active_model
    assert loaded_wrapper.models_manager.autoencoder.is_trained
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
