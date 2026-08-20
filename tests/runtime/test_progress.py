"""Tests for runtime progress display helpers."""

from __future__ import annotations

from msi_autoencoder_wrapper.runtime.cli import _task_label
from msi_autoencoder_wrapper.runtime.progress import format_duration


def test_format_duration_renders_known_and_missing_values() -> None:
    """Elapsed-time values use a stable human-readable representation."""
    assert format_duration(3723.9) == "01:02:03"
    assert format_duration(None) == "--:--:--"


def test_task_label_identifies_architecture_binning_and_repetition() -> None:
    """The terminal label distinguishes every planned model run."""
    task = {
        "task_id": "task_000025",
        "repetition": 0,
        "grid_parameters": {
            "architectures": {"name": "mlp-ae-512-256-latent-10"},
            "binning_steps": 0.5,
        },
    }

    assert _task_label(task) == (
        "task_000025 | mlp-ae-512-256-latent-10 | bin=0.5 | rep=0"
    )
