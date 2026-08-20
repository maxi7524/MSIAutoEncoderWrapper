"""Synthetic grid-experiment campaign builder shared by analysis tests.

Not itself a test module (no ``test_*`` functions) - a helper imported by the
``test_campaign_reader``/``test_architecture_overview_analysis``/
``test_training_dynamics_analysis`` modules to avoid recreating this fixture
filesystem layout three times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import yaml

from msi_autoencoder_wrapper.models.model_loader import ModelLoader

EXPERIMENT_NAME = "test-campaign"

# Tiny MLP encoder/decoder: batch_normalization disabled so parameter counts are a
# closed-form function of (input_dim, hidden_dims, latent_dim) that tests can assert
# on directly.
_ENCODER_PARAMETERS = {"latent_dim": 2, "hidden_dims": [4], "batch_normalization": False}
_DECODER_PARAMETERS = {
    "latent_dim": 2,
    "hidden_dims": [4],
    "batch_normalization": False,
    "output_activation": {"type": "softplus", "parameters": {}},
}


def _model_config(
    input_dim: int,
    architecture_name: str,
    binner_range: tuple[float, float, float],
) -> dict:
    binner_x_min, binner_x_max, binner_bin_step = binner_range
    return {
        "model": {
            "name": architecture_name,
            "type": "autoencoder",
            "parameters": {},
            "components": {
                "encoder": {
                    "type": "MLPEncoder",
                    "parameters": {**_ENCODER_PARAMETERS, "input_dim": input_dim},
                },
                "decoder": {
                    "type": "MLPDecoder",
                    "parameters": {**_DECODER_PARAMETERS, "output_dim": input_dim},
                },
            },
        },
        "data": {
            "context": {
                "components": {
                    "binner": {
                        "type": "LinearBinning",
                        "parameters": {
                            "x_min": binner_x_min,
                            "x_max": binner_x_max,
                            "bin_step": binner_bin_step,
                            "aggregation": "sum",
                        },
                    }
                }
            },
            "dataset": {
                "parameters": {
                    "normalization": "tic",
                    "split": {
                        "strategy": "random",
                        "seed": 42,
                        "assignments": {
                            "train": list(range(8)),
                            "validation": [8],
                            "test": [9],
                        },
                    },
                }
            },
        },
    }


def _history(durations: list[float], train_losses: list[float], val_losses: list[float]) -> list[dict]:
    entries = []
    for epoch, (duration, train_loss, val_loss) in enumerate(
        zip(durations, train_losses, val_losses), start=1
    ):
        entries.append(
            {
                "phase": "reconstruction",
                "metrics": {
                    "epoch": epoch,
                    "duration": duration,
                    "masserstein": train_loss,
                    "total_loss": train_loss,
                    "validation_masserstein": val_loss,
                    "validation_total_loss": val_loss,
                    "checkpoint_scope": "validation",
                    "is_best": epoch == len(durations),
                    "best_loss": min(val_losses[:epoch]),
                },
            }
        )
    # Trailing post-training test-split evaluation: no "epoch"/"duration", matching
    # the real runtime's history.json layout.
    entries.append(
        {
            "phase": "reconstruction",
            "split": "test",
            "metrics": {"masserstein": val_losses[-1], "total_loss": val_losses[-1]},
        }
    )
    return entries


def write_campaign_task(
    workspace: Path,
    *,
    task_id: str,
    architecture_name: str,
    preset: str,
    binning_step: float,
    repetition: int,
    input_dim: int,
    status: str = "completed",
    durations: Optional[list[float]] = None,
    train_losses: Optional[list[float]] = None,
    val_losses: Optional[list[float]] = None,
    binner_range: Optional[tuple[float, float, float]] = None,
    save_weights: bool = False,
) -> Path:
    """Write one synthetic task's status manifest and (if completed) model artifacts.

    :param binner_range: ``(x_min, x_max, bin_step)`` recorded under
        ``data.context.components.binner.parameters``. Defaults to an arbitrary
        ``(0.0, input_dim, 1.0)`` — only tests that actually rebuild a binner from
        this config (e.g. model-reconstruction tests) need to pass a real range.
    :type binner_range: tuple[float, float, float] | None
    :param save_weights: Also build a real (randomly initialized) model matching
        this config via :class:`ModelLoader` and save its ``state_dict`` as
        ``weights.pt``, so :meth:`ModelLoader.load_artifact` can load this task.
        Off by default — most tests only need ``config.json``/``history.json``.
    :type save_weights: bool
    """
    result = None
    if status == "completed":
        model_directory = workspace / "models" / "context" / task_id / "config"
        model_directory.mkdir(parents=True, exist_ok=True)
        resolved_binner_range = binner_range or (0.0, float(input_dim), 1.0)
        config = _model_config(input_dim, architecture_name, resolved_binner_range)
        (model_directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if save_weights:
            model, _, _ = ModelLoader.build(config)
            torch.save(model.state_dict(), model_directory / "weights.pt")
        history = _history(
            durations or [1.0, 1.0, 1.0],
            train_losses or [3.0, 2.0, 1.0],
            val_losses or [3.5, 2.5, 1.5],
        )
        (model_directory / "history.json").write_text(json.dumps(history), encoding="utf-8")
        result = {"model_path": str(model_directory.parent), "epochs": len(history)}

    status_directory = workspace / "configs" / "execution" / EXPERIMENT_NAME / "status"
    status_directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "records": {
            task_id: {
                "status": status,
                "task": {
                    "task_id": task_id,
                    "grid_parameters": {
                        "architectures": {
                            "name": architecture_name,
                            "preset": preset,
                            "parameters": {"latent_dim": 2},
                        },
                        "binning_steps": binning_step,
                    },
                    "repetition": repetition,
                },
                "result": result,
            }
        }
    }
    manifest_path = status_directory / f"{task_id}.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    # A progress snapshot must never be mistaken for a task manifest.
    (status_directory / f"{task_id}-progress.yaml").write_text("status: running\n", encoding="utf-8")
    return manifest_path
