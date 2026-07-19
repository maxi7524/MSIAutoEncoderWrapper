"""Tests for training resource estimation and batch-size recommendations."""

from __future__ import annotations

from pathlib import Path

import torch

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from tests.mocks.components import build_small_autoencoder


class _ResourceDataset:
    """Small dataset matching the test autoencoder input width."""

    def __len__(self) -> int:
        """Return the number of synthetic spectra."""
        return 16

    def __getitem__(self, index: int):
        """Return one index and a deterministic 32-bin spectrum."""
        return index, torch.linspace(0.0, 1.0, 32)


def _training_config(batch_size: int) -> dict:
    """Return a compact categorized training configuration."""
    return {
        "phases": [
            {
                "phase_name": "reconstruction",
                "epochs": 2,
                "batch_size": batch_size,
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
                "criterions": {
                    "reconstruction": {
                        "MSELoss": {"weight": 1.0, "params": {}},
                    }
                },
            }
        ]
    }


def _wrapper(tmp_path: Path) -> MSIAutoEncoderWrapper:
    """Return a wrapper with a model and lazy MSI dataset attached."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    wrapper.active_dataset = _ResourceDataset()
    wrapper.models_manager.attach_model(
        build_small_autoencoder(),
        model_name="resource-probe",
    )
    return wrapper


def test_resource_estimator_reports_ram_vram_disk_and_method(
    tmp_path: Path,
    capsys,
) -> None:
    """A probe report separates resources and records estimation limitations."""
    wrapper = _wrapper(tmp_path)

    report = wrapper.models_manager.estimate_training_resources(_training_config(8))

    assert report["method"] == "probe_forward_plus_static_training_state"
    assert report["estimated_disk_bytes"] > report["model_bytes"]
    assert report["phases"][0]["estimated_ram_bytes"] > 0
    assert report["phases"][0]["estimated_vram_bytes"] == 0
    assert report["phases"][0]["recommended_batch_size"] == 8
    assert report["limitations"]
    assert "Training resource estimate" in capsys.readouterr().out


def test_resource_estimator_can_reduce_batch_size_to_absolute_ram_limit(
    tmp_path: Path,
) -> None:
    """Automatic selection lowers a configured batch without mutating the input."""
    wrapper = _wrapper(tmp_path)
    batch_one_report = wrapper.models_manager.estimate_training_resources(
        _training_config(1)
    )
    batch_one_ram = batch_one_report["phases"][0]["estimated_ram_bytes"]
    config = _training_config(64)

    report = wrapper.models_manager.estimate_training_resources(
        config,
        resource_limits={"ram": batch_one_ram + 1, "vram": 1, "disk": 10**12},
        auto_adjust_batch_size=True,
    )

    assert report["phases"][0]["recommended_batch_size"] == 1
    assert report["phases"][0]["fits_limits"]
    assert report["recommended_training_config"]["phases"][0]["batch_size"] == 1
    assert config["phases"][0]["batch_size"] == 64
