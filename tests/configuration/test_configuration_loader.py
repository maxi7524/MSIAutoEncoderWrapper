"""Tests for modular saved-configuration restoration."""

from __future__ import annotations

import shutil
from pathlib import Path

from msi_autoencoder_wrapper import MSIAutoEncoderWrapper


def _configuration() -> dict:
    return {
        "schema_version": 2,
        "experiment": {
            "name": "model-a",
            "context": {"type": "image", "key": "image-a"},
        },
        "data": {
            "context": {
                "schema_version": 1,
                "scope": "local_image",
                "image_key": "image-a",
                "coordinate_order": "xy",
                "components": {
                    "reader": {
                        "type": "PyImzMLReader",
                        "parameters": {"file_path": "/non-portable/source.imzML"},
                    },
                    "binner": {
                        "type": "LinearBinning",
                        "parameters": {
                            "x_min": 400.0,
                            "x_max": 1000.0,
                            "bin_step": 1.0,
                        },
                    },
                    "inverse_binner": {
                        "type": "TopPeaksInverseBinner",
                        "parameters": {"max_bins": 20, "window_size": 1},
                    },
                },
            },
            "dataset": {
                "type": "PixelDataset",
                "parameters": {"normalization": "tic"},
            },
            "split": None,
        },
        "model": {"name": "model-a", "type": "autoencoder"},
        "training": {"parameters": None},
    }


def test_wrapper_orchestrates_module_owned_configuration_loaders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The facade coordinates sections and returns the original dictionary."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    wrapper.workspace.set_default_image_path("image-a")
    config = _configuration()
    calls: list[tuple] = []

    monkeypatch.setattr(
        wrapper.workspace,
        "load_config_json",
        lambda image_name, model_name: config,
    )
    monkeypatch.setattr(
        wrapper.context_manager,
        "load_context_config",
        lambda section, img_name_or_path=None, base_path=None: calls.append(
            ("context", section, img_name_or_path)
        ),
    )
    monkeypatch.setattr(
        wrapper.models_manager,
        "load_model",
        lambda **kwargs: calls.append(("model", kwargs)),
    )

    loaded = wrapper.load_configuration(model_name="model-a")

    assert loaded is config
    assert wrapper.active_dataset.normalization == "tic"
    assert [call[0] for call in calls] == ["context", "model"]
    assert calls[-1][1]["img_name"] == "image-a"


def test_configuration_can_be_read_without_applying_runtime_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """apply=False returns validated data without invoking section loaders."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    wrapper.workspace.set_default_image_path("image-a")
    config = _configuration()
    monkeypatch.setattr(
        wrapper.workspace,
        "load_config_json",
        lambda image_name, model_name: config,
    )
    monkeypatch.setattr(
        wrapper.context_manager,
        "load_context_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    assert wrapper.load_configuration("model-a", apply=False) is config


def test_image_context_owns_portable_component_restoration(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Saved absolute reader paths are replaced by workspace image resolution."""
    workspace = tmp_path / "workspace"
    image_dir = workspace / "datasets" / "image-a"
    image_dir.mkdir(parents=True)
    target_imzml = image_dir / "image-a.imzML"
    target_ibd = image_dir / "image-a.ibd"
    shutil.copy2(msi_fixture_path, target_imzml)
    shutil.copy2(msi_fixture_path.with_suffix(".ibd"), target_ibd)
    wrapper = MSIAutoEncoderWrapper(project_path=str(workspace))
    wrapper.workspace.set_default_image_path("image-a")

    restored = wrapper.context_manager.load_context_config(
        _configuration()["data"]["context"]
    )

    assert restored["reader"].file_path == target_imzml
    assert restored["binner"].GetXAxisDepth() == 600
    assert restored["inverse_binner"]._Binner is restored["binner"]


def test_dataset_manager_owns_dataset_configuration(tmp_path: Path) -> None:
    """The model proxy delegates dataset restoration to DatasetManager."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    dataset = wrapper.models_manager.load_dataset_config(
        {
            "type": "PixelDataset",
            "parameters": {"normalization": "max"},
        }
    )

    assert dataset is wrapper.active_dataset
    assert dataset.normalization == "max"
