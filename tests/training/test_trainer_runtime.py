"""Integration tests for DataLoader configuration and best checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)
from msi_autoencoder_wrapper.utils.configuration import ConfigurableComponent
from tests.mocks.components import MockMSIReader, build_small_autoencoder


class _TrainingDataset(ConfigurableComponent):
    """Small deterministic dataset compatible with the test autoencoder."""

    source = "image"

    def __init__(self) -> None:
        """Initialize a serializable synthetic dataset configuration."""
        self._config = {"source": self.source}

    def __len__(self) -> int:
        """Return the number of spectra."""
        return 8

    def __getitem__(self, index: int):
        """Return one deterministic 32-bin training spectrum."""
        return index, torch.linspace(0.0, 1.0, 32)


class _HeadTrainingDataset(ConfigurableComponent):
    """Small serializable dataset exposing one masked multi-label target."""

    source = "image"

    def __init__(self) -> None:
        """Initialize the dataset configuration used in saved checkpoints."""
        self._config = {
            "source": self.source,
            "target_specs": {
                "molecule": {
                    "type": "multi_label",
                    "class_mapping": {"C1H2|M+H": 0, "C2H4|M+H": 1},
                }
            },
        }

    def __len__(self) -> int:
        """Return the number of synthetic spectra."""
        return 8

    def __getitem__(self, index: int):
        """Return one spectrum, molecule target, and availability mask."""
        target = torch.tensor([float(index % 2 == 0), float(index % 2 == 1)])
        return (
            index,
            torch.linspace(0.0, 1.0, 32),
            {"molecule": target},
            {"molecule": torch.tensor(True)},
        )


def test_training_saves_and_restores_best_checkpoint(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Every improvement saves portable artifacts and marks history entries."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    reader = MockMSIReader(msi_fixture_path)
    wrapper.context_manager.set_reader(reader, str(msi_fixture_path))
    wrapper.workspace.set_active_image(str(msi_fixture_path))
    wrapper.active_dataset = _TrainingDataset()
    wrapper.models_manager.attach_model(
        build_small_autoencoder(),
        model_name="checkpoint-ae",
    )
    training_config = {
        "checkpoint": {"enabled": True, "restore_best": True},
        "phases": [
            {
                "phase_name": "reconstruction",
                "epochs": 2,
                "batch_size": 4,
                "dataloader": {"num_workers": 0, "shuffle": False},
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
                "criterions": {
                    "reconstruction": {
                        "mse": {"target": "MSELoss", "params": {}},
                    }
                },
            }
        ],
    }

    history = wrapper.models_manager.fit(training_config)
    config_dir = (
        tmp_path
        / "models"
        / msi_fixture_path.stem
        / "checkpoint-ae"
        / "config"
    )

    assert len(history) == 2
    assert any(entry["metrics"]["is_best"] for entry in history)
    assert (config_dir / "config.json").is_file()
    assert (config_dir / "weights.pt").is_file()
    assert (config_dir / "history.json").is_file()


def test_fit_checkpoints_and_reloads_named_heads(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """The real fit checkpoint path persists named heads and target bindings."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    reader = MockMSIReader(msi_fixture_path)
    wrapper.context_manager.set_reader(reader, str(msi_fixture_path))
    wrapper.workspace.set_active_image(str(msi_fixture_path))
    wrapper.active_dataset = _HeadTrainingDataset()

    component_setup = {
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
                "output_activation": {"type": "softplus", "parameters": {}},
            },
        },
        "heads": {
            "molecule_primary": {
                "strategy": "LinearClassificationHead",
                "params": {"latent_dim": 4, "output_dim": 2},
            }
        },
    }
    ArchitecturesManager.discover_architectures()
    model = ArchitecturesManager.build_model(
        "autoencoder",
        component_setup,
        head_specs={"molecule_primary": {"target_field": "molecule"}},
    )
    wrapper.models_manager.attach_model(model, model_name="checkpoint-head-ae")
    wrapper.models_manager._building_buffer = {
        category: {
            "target": setup["strategy"],
            "kwargs": setup["params"],
        }
        for category, setup in component_setup.items()
        if category != "heads"
    }
    wrapper.models_manager._building_buffer["heads"] = {
        "molecule_primary": {
            "target": "LinearClassificationHead",
            "target_field": "molecule",
            "kwargs": {"latent_dim": 4, "output_dim": 2},
        }
    }
    training_config = {
        "checkpoint": {"enabled": True, "restore_best": True},
        "phases": [
            {
                "phase_name": "reconstruction_and_molecule",
                "epochs": 1,
                "batch_size": 4,
                "dataloader": {"num_workers": 0, "shuffle": False},
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
                "criterions": {
                    "reconstruction": {
                        "mse": {"target": "MSELoss", "params": {}},
                    },
                    "heads": {
                        "molecule_primary": {
                            "bce": {
                                "target": "MultiLabelBCELoss",
                                "params": {},
                            }
                        }
                    },
                },
            }
        ],
    }

    history = wrapper.models_manager.fit(training_config)
    saved_config = wrapper.workspace.load_config_json(
        msi_fixture_path.stem,
        "checkpoint-head-ae",
    )
    reloaded = wrapper.models_manager.load_model(
        msi_fixture_path.stem,
        "checkpoint-head-ae",
    )
    outputs = reloaded(torch.ones(2, 32))

    loaded_model = saved_config["model"]
    assert len(history) == 1
    assert history[0]["metrics"]["is_best"]
    assert "molecule_primary" in loaded_model["components"]["heads"]
    assert loaded_model["parameters"]["head_specs"] == {
        "molecule_primary": {"target_field": "molecule"},
    }
    assert outputs["head_molecule_primary"].shape == (2, 2)
