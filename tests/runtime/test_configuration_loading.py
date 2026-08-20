"""Tests for explicit runtime workspace path anchors."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from msi_autoencoder_wrapper.runtime.configuration.loading import load_experiment_config


def _payload(project_path: str, anchor: str) -> dict:
    """Create the smallest valid runtime configuration for path-resolution tests."""
    return {
        "schema_version": 1,
        "experiment": {"name": "path-anchor"},
        "task": {
            "entrypoint": "tests.runtime.test_cli:task_entrypoint",
            "preflight_entrypoint": "tests.runtime.test_cli:preflight_entrypoint",
            "plan_entrypoint": "tests.runtime.test_cli:plan_entrypoint",
            "parameters": {
                "factory_parameters": {
                    "project_path": project_path,
                    "project_path_anchor": anchor,
                },
                "training": {
                    "continuation": {
                        "enabled": True,
                        "resume": True,
                        "checkpoint_every_epochs": 1,
                    }
                },
            },
        },
        "runs": {"repetitions": 1},
        "seeds": {
            "common_seeds": {"split": 1, "dataloader": 2},
            "run_seeds": {"model_initialization": 3, "training": 4},
        },
        "grid": {},
        "execution": {"backend": "local"},
        "reports": [],
    }


def test_project_path_requires_an_explicit_anchor(tmp_path: Path) -> None:
    """Relative workspace paths cannot silently depend on YAML nesting depth."""
    path = tmp_path / "experiment.yaml"
    payload = _payload("workspace", "yaml_directory")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="project_path_anchor"):
        load_experiment_config(path)
