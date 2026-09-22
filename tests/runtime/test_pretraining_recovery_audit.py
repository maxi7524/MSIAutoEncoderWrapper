"""Tests for per-log and per-checkpoint recovered-campaign evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import torch


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "assets/experiments/autoencoder_architecture/experiment_runs_configs"
    / "segmentation_model/20_09_26_metaspace_base_pretrain/recovery_audit.py"
)


def _audit_module() -> ModuleType:
    """Load the campaign-local audit implementation."""
    specification = importlib.util.spec_from_file_location("pretraining_recovery_audit", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_log_parser_tracks_phase_and_snapshot_evidence(tmp_path: Path) -> None:
    """One scheduler log yields separate epoch, completion and save evidence."""
    module = _audit_module()
    log_path = tmp_path / "task_21534_100.log"
    log_path.write_text(
        "Initiating sequential training loop phase: synthetic_joint_all\n"
        "=== Epoch Summary [synthetic_joint_all] 1/2\n"
        "=== Epoch Summary [synthetic_joint_all] 2/2\n"
        "Restored the best checkpoint after phase 'synthetic_joint_all'.\n"
        "Stored completed phase 'synthetic_joint_all' as snapshot 'synthetic_pretrained'.\n"
        "Active model 'campaign__task_000100' saved under context\n",
        encoding="utf-8",
    )

    evidence = module._log(log_path)

    assert evidence["phases"] == ["synthetic_joint_all"]
    assert evidence["epochs"]["synthetic_joint_all"] == {1, 2}
    assert evidence["ended"] == ["synthetic_joint_all"]
    assert evidence["snapshots"] == [("synthetic_joint_all", "synthetic_pretrained")]
    assert evidence["saved"] == ["campaign__task_000100"]


def test_checkpoint_rejects_missing_or_mismatched_pretraining_state(tmp_path: Path) -> None:
    """A fingerprint match alone does not establish a usable pretrain snapshot."""
    module = _audit_module()
    path = tmp_path / "task_000100.pt"
    state = {"encoder.weight": torch.tensor([1.0, 2.0])}
    torch.save(
        {"task_fingerprint": "expected", "model_state": state,
         "phase_snapshots": {"synthetic_pretrained": {"encoder.weight": torch.tensor([3.0, 4.0])}}},
        path,
    )
    valid = module._checkpoint(path, "expected")
    assert valid["valid"] and valid["snapshot"]
    assert not module._checkpoint(path, "different")["valid"]

    torch.save(
        {"task_fingerprint": "expected", "model_state": state,
         "phase_snapshots": {"synthetic_pretrained": {"other.weight": torch.tensor([3.0, 4.0])}}},
        path,
    )
    malformed = module._checkpoint(path, "expected")
    assert malformed["valid"] and not malformed["snapshot"]


def test_report_keeps_individual_log_rows() -> None:
    """The final audit exposes every scheduler attempt, including failed ones."""
    module = _audit_module()
    logs = [
        {"task_id": "task_000001", "name": "task_10_1.log", "phases": [],
         "epochs": {}, "ended": [], "snapshots": [], "saved": [], "error": True},
        {"task_id": "task_000001", "name": "task_11_1.log", "phases": ["real_only"],
         "epochs": {"real_only": {1}}, "ended": ["real_only"], "snapshots": [],
         "saved": ["campaign__task_000001"], "error": False},
    ]

    report = module.render([], logs)

    assert "Inspected **0** task descriptors and **2** individual scheduler logs" in report
    assert "task_10_1.log" in report
    assert "task_11_1.log" in report
