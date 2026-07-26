"""Integration tests for DataLoader configuration and best checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
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
