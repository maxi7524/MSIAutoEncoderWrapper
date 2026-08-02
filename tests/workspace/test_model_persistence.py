"""Tests for portable model persistence and the thin workspace API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)
from msi_autoencoder_wrapper.models.model_loader import ModelLoader
from msi_autoencoder_wrapper.utils.exceptions import WorkspaceConfigError
from msi_autoencoder_wrapper.workspace.model_store import ModelStore
from tests.mocks.components import MockMSIReader, build_small_autoencoder


LAYOUT = {
    "imgs_dir": "imgs",
    "models_root": "models",
    "model_config_subdir": "config",
    "model_latent_subdir": "latent",
}


def test_model_store_defaults_to_json_and_weights(tmp_path: Path) -> None:
    """The default representation contains configuration and safe state weights."""
    store = ModelStore(tmp_path, LAYOUT)
    model = build_small_autoencoder()

    model_dir = store.save_model(
        context_name="image-a",
        model_name="autoencoder-a",
        config={"schema_version": 1, "parameter": Path("source.imzML")},
        state_dict=model.state_dict(),
    )
    loaded_config = store.load_config("image-a", "autoencoder-a")
    loaded_weights = store.load_weights("image-a", "autoencoder-a")

    assert loaded_config["parameter"] == "source.imzML"
    assert set(loaded_weights) == set(model.state_dict())
    assert (model_dir / "config" / "config.json").is_file()
    assert (model_dir / "config" / "weights.pt").is_file()
    assert not (model_dir / "config" / "model_deployment_full.pt").exists()


def test_complete_model_folder_export_is_explicit(tmp_path: Path) -> None:
    """Explicit export copies configuration, weights, history, and latent artifacts."""
    store = ModelStore(tmp_path / "workspace", LAYOUT)
    model = build_small_autoencoder()
    model_dir = store.save_model(
        context_name="image-a",
        model_name="autoencoder-a",
        config={"schema_version": 1},
        state_dict=model.state_dict(),
        history=[{"epoch": 1, "loss": 0.25}],
    )
    latent_dir = model_dir / "latent"
    latent_dir.mkdir(parents=True)
    (latent_dir / "latent.imzML").write_text("fixture", encoding="utf-8")
    destination = tmp_path / "portable" / "autoencoder-a"

    exported = store.export_model_folder(
        context_name="image-a",
        model_name="autoencoder-a",
        destination=destination,
    )

    assert (exported / "config" / "config.json").is_file()
    assert (exported / "config" / "weights.pt").is_file()
    assert (exported / "config" / "history.json").is_file()
    assert (exported / "latent" / "latent.imzML").is_file()
    with pytest.raises(WorkspaceConfigError):
        store.export_model_folder("image-a", "autoencoder-a", destination)


def test_workspace_save_model_keeps_image_and_loaded_model_contexts_separate(
    tmp_path: Path,
    msi_fixture_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The high-level save composes, but does not conflate, both active contexts."""
    monkeypatch.chdir(tmp_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    reader = MockMSIReader(msi_fixture_path)
    binner = BinnerManager.get_binner(
        "LinearBinning",
        x_min=reader.GetXMin(),
        x_max=reader.GetXMax(),
        bin_step=1.0,
    )
    wrapper.context_manager.set_reader(reader, str(msi_fixture_path))
    wrapper.context_manager.set_binner(binner, str(msi_fixture_path))
    wrapper.workspace.set_active_image(str(msi_fixture_path))

    wrapper.active_model = build_small_autoencoder()
    wrapper.models_manager.active_model_type = "autoencoder"
    wrapper.models_manager._active_model_name = "portable-ae"

    model_dir = wrapper.workspace.save_model()
    config_path = model_dir / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["experiment"]["context"] == {
        "type": "image",
        "key": msi_fixture_path.stem,
    }
    assert config["data"]["context"]["image_key"] == msi_fixture_path.stem
    assert config["model"]["name"] == "portable-ae"
    assert (model_dir / "config" / "weights.pt").is_file()


def test_named_heads_are_serialized_and_reconstructed_separately(
    tmp_path: Path,
) -> None:
    """Named head modules and target bindings survive configuration round trips."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    manager = wrapper.models_manager
    manager.active_model_type = "autoencoder"
    manager._active_model_name = "multi-head-ae"
    manager._building_buffer = {
        "encoder": {
            "target": "CNNEncoder",
            "kwargs": {
                "input_dim": 32,
                "latent_dim": 4,
                "channels": [1, 2],
                "kernels": [3],
                "strides": [2],
                "spatial_dims": [32, 15],
            },
        },
        "decoder": {
            "target": "CNNDecoder",
            "kwargs": {
                "latent_dim": 4,
                "channels": [1, 2],
                "kernels": [3],
                "strides": [2],
                "spatial_dims": [32, 15],
                "output_activation": {"type": "softplus", "parameters": {}},
            },
        },
        "heads": {
            "condition_primary": {
                "target": "LinearClassificationHead",
                "target_field": "condition",
                "kwargs": {"latent_dim": 4, "output_dim": 2},
            },
            "condition_secondary": {
                "target": "LinearClassificationHead",
                "target_field": "condition",
                "kwargs": {
                    "latent_dim": 4,
                    "output_dim": 2,
                    "hidden_dim": 3,
                },
            },
        },
    }
    ArchitecturesManager.discover_architectures()
    manager.compile_model(run_validation_pass=False)

    config = manager.get_model_config()
    reconstructed, model_type, model_name = ModelLoader.build(config)
    outputs = reconstructed(torch.randn(2, 32))

    assert set(config["model"]["components"]["heads"]) == {
        "condition_primary",
        "condition_secondary",
    }
    assert config["model"]["parameters"]["head_specs"] == {
        "condition_primary": {"target_field": "condition"},
        "condition_secondary": {"target_field": "condition"},
    }
    assert model_type == "autoencoder"
    assert model_name == "multi-head-ae"
    assert outputs["head_condition_primary"].shape == (2, 2)
    assert outputs["head_condition_secondary"].shape == (2, 2)


def test_partial_custom_layout_inherits_default_keys(tmp_path: Path) -> None:
    """A partial custom layout no longer causes missing-key failures."""
    wrapper = MSIAutoEncoderWrapper(
        project_path=str(tmp_path),
        layout={"models_root": "saved_models"},
    )

    assert wrapper.workspace.get_models_root() == tmp_path / "saved_models"
    assert "saved_models" in str(wrapper.workspace)
