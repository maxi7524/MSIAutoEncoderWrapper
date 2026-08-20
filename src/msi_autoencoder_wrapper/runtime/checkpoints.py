"""Atomic training checkpoints used to resume interrupted runtime tasks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_training_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    phase_index: int,
    epoch: int,
    history: list[dict[str, Any]],
    best_loss: float,
    patience_counter: int,
    task_fingerprint: str,
) -> None:
    """Atomically save all state required to continue after one epoch."""
    payload = {
        "version": 1,
        "task_fingerprint": task_fingerprint,
        "phase_index": phase_index,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(path: Path, *, task_fingerprint: str) -> dict[str, Any] | None:
    """Load and validate one runtime checkpoint when it matches the task."""
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Invalid runtime checkpoint: {path}")
    if payload.get("task_fingerprint") != task_fingerprint:
        raise ValueError(f"Runtime checkpoint does not match the current task: {path}")
    return payload


def restore_random_state(checkpoint: dict[str, Any]) -> None:
    """Restore Python, NumPy and Torch random streams from a checkpoint."""
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_state = checkpoint.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
