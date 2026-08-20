"""Tests for atomic runtime continuation checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from msi_autoencoder_wrapper.runtime.checkpoints import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_training_checkpoint_round_trip_and_fingerprint_validation(tmp_path: Path) -> None:
    """A continuation checkpoint preserves optimizer and epoch state."""
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "task.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        phase_index=1,
        epoch=4,
        history=[{"epoch": 5}],
        best_loss=0.25,
        patience_counter=2,
        task_fingerprint="abc",
    )
    loaded = load_training_checkpoint(path, task_fingerprint="abc")

    assert loaded is not None
    assert loaded["phase_index"] == 1
    assert loaded["epoch"] == 4
    assert loaded["history"] == [{"epoch": 5}]
    assert "state" in loaded["optimizer_state"]
    with pytest.raises(ValueError, match="does not match"):
        load_training_checkpoint(path, task_fingerprint="different")
