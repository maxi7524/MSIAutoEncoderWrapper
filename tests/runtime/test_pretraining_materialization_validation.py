"""Tests for persisted pretraining campaign artifact validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
import yaml


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "assets/scripts/analysis/validate_model_materialization.py"
)
def _load_validator() -> ModuleType:
    """Load the campaign-local script as a testable module."""
    specification = importlib.util.spec_from_file_location(
        "pretraining_materialization_validator",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _fingerprint(task: dict[str, Any]) -> str:
    """Return the runtime-compatible task fingerprint."""
    serialized = yaml.safe_dump(task, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_campaign(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Write one complete synthetic branch group and its model bundles."""
    campaign = tmp_path / "campaign"
    (campaign / "tasks").mkdir(parents=True)
    (campaign / "status").mkdir()
    parent_id = "task_000000"
    roles = {
        parent_id: "pretrained",
        "task_000001": "frozen_head",
        "task_000002": "unfrozen_head",
    }
    tasks = []
    model_paths: dict[str, Path] = {}
    states = {
        "pretrained": {
            "encoder.weight": torch.tensor([[1.0, 2.0]]),
            "heads.molecule_vpu.weight": torch.tensor([[3.0, 4.0]]),
        },
        "frozen_head": {
            "encoder.weight": torch.tensor([[1.5, 2.0]]),
            "heads.molecule_vpu.weight": torch.tensor([[3.0, 4.0]]),
        },
        "unfrozen_head": {
            "encoder.weight": torch.tensor([[1.5, 2.0]]),
            "heads.molecule_vpu.weight": torch.tensor([[3.5, 4.0]]),
        },
    }

    for task_id, role in roles.items():
        task = {
            "task_id": task_id,
            "grid_id": "grid_0000",
            "repetition": 0,
            "workflow": {
                "group_id": "grid_0000__rep_00",
                "role": role,
                "parent_task_id": None if role == "pretrained" else parent_id,
            },
            "depends_on": [] if role == "pretrained" else [parent_id],
            "parameters": {"training": {"phases": []}},
        }
        tasks.append(task)
        descriptor = campaign / "tasks" / f"{task_id}.yaml"
        descriptor.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

        model_path = (
            tmp_path
            / "models"
            / "context"
            / f"campaign__grid_0000__rep_00__{role}"
        ).resolve()
        config_directory = model_path / "config"
        config_directory.mkdir(parents=True)
        (config_directory / "config.json").write_text("{}", encoding="utf-8")
        (config_directory / "history.json").write_text("[]", encoding="utf-8")
        torch.save(states[role], config_directory / "weights.pt")
        model_paths[role] = model_path

    parent_weights = model_paths["pretrained"] / "config" / "weights.pt"
    parent_hash = hashlib.sha256(parent_weights.read_bytes()).hexdigest()
    for task in tasks:
        task_id = task["task_id"]
        role = task["workflow"]["role"]
        result: dict[str, Any] = {"model_path": str(model_paths[role])}
        if role != "pretrained":
            result["initialization"] = {
                "parent_task_id": parent_id,
                "model_path": str(model_paths["pretrained"]),
                "weights_sha256": parent_hash,
            }
        status = {
            "records": {
                task_id: {
                    "status": "completed",
                    "task_fingerprint": _fingerprint(task),
                    "result": result,
                }
            }
        }
        (campaign / "status" / f"{task_id}.yaml").write_text(
            yaml.safe_dump(status, sort_keys=False),
            encoding="utf-8",
        )

    manifest = {"runtime_schema_version": 2, "tasks": tasks}
    (campaign / "resolved-experiment.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return campaign, model_paths


def test_complete_materialized_branch_group_passes(tmp_path: Path) -> None:
    """Three independent artifacts with verified lineage satisfy the contract."""
    validator = _load_validator()
    campaign, _ = _write_campaign(tmp_path)

    assert validator.validate_campaign(campaign) == []


def test_compact_schema_three_manifest_preserves_artifact_validation(tmp_path: Path) -> None:
    """The same model lineage is valid with compact task graph entries."""
    validator = _load_validator()
    campaign, _ = _write_campaign(tmp_path)
    manifest_path = campaign / "resolved-experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_schema_version"] = 3
    manifest["tasks"] = [
        {key: task[key] for key in (
            "task_id", "grid_id", "repetition", "workflow", "depends_on"
        )}
        for task in manifest["tasks"]
    ]
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    assert validator.validate_campaign(campaign) == []


def test_missing_workflow_parent_is_reported(tmp_path: Path) -> None:
    """The validator reports workflow links whose parent task is absent."""
    validator = _load_validator()
    campaign, _ = _write_campaign(tmp_path)
    manifest_path = campaign / "resolved-experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"] = [
        task for task in manifest["tasks"] if task["workflow"]["role"] != "pretrained"
    ]
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    issues = validator.validate_campaign(campaign)

    assert any("workflow parent task does not exist" in issue for issue in issues)


def test_child_provenance_must_match_parent_weights(tmp_path: Path) -> None:
    """A declared dependency is insufficient without the loaded-weight digest."""
    validator = _load_validator()
    campaign, _ = _write_campaign(tmp_path)
    status_path = campaign / "status" / "task_000002.yaml"
    status = yaml.safe_load(status_path.read_text(encoding="utf-8"))
    status["records"]["task_000002"]["result"]["initialization"][
        "weights_sha256"
    ] = "0" * 64
    status_path.write_text(yaml.safe_dump(status, sort_keys=False), encoding="utf-8")

    issues = validator.validate_campaign(campaign)

    assert any("weights_sha256 does not match parent weights" in issue for issue in issues)
