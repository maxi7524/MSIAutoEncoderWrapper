"""Tests for the SLURM (entropy) campaign reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import (
    CampaignTask,
    read_entropy_campaign,
)


def _write_entropy_task(
    status_directory: Path,
    model_store_directory: Path,
    *,
    task_id: str,
    model_name: str,
    grid_id: str,
    repetition: int,
    status: str = "completed",
    scratch_model_path: str = "/tmp/someuser/msi-wrapper/run-01/kidney_workspace/models/kidney/unused",
) -> None:
    if status == "completed":
        model_directory = model_store_directory / model_name / "config"
        model_directory.mkdir(parents=True, exist_ok=True)
        (model_directory / "config.json").write_text(
            json.dumps({"model": {"name": "conv1d-ae"}}), encoding="utf-8"
        )
        (model_directory / "history.json").write_text(
            json.dumps([{"metrics": {"epoch": 1, "total_loss": 1.0}}]),
            encoding="utf-8",
        )
    manifest = {
        "records": {
            task_id: {
                "status": status,
                "result": (
                    {"model_path": scratch_model_path, "epochs": 1}
                    if status == "completed"
                    else None
                ),
                "task": {
                    "task_id": task_id,
                    "grid_id": grid_id,
                    "repetition": repetition,
                    "grid_parameters": {"objectives": {"heads": {}}},
                    "runtime": {"model_name": model_name},
                },
            }
        }
    }
    status_directory.mkdir(parents=True, exist_ok=True)
    (status_directory / f"{task_id}.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (status_directory / f"{task_id}-progress.yaml").write_text(
        "status: running\n", encoding="utf-8"
    )


def test_read_entropy_campaign_resolves_artifacts_via_model_name_not_scratch_path(
    tmp_path: Path,
) -> None:
    status_directory = tmp_path / "entropy_run" / "plan" / "status"
    model_store_directory = tmp_path / "workspace" / "models" / "kidney"
    _write_entropy_task(
        status_directory,
        model_store_directory,
        task_id="task_000000",
        model_name="campaign__cfg_abc123__grid_0000__rep_00",
        grid_id="grid_0000",
        repetition=0,
    )

    tasks = read_entropy_campaign(status_directory, model_store_directory)

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, CampaignTask)
    assert task.task_id == "task_000000"
    assert task.status == "completed"
    assert task.repetition == 0
    assert task.grid_parameters == {"objectives": {"heads": {}}}
    # The scratch result.model_path is never dereferenced; artifacts came from
    # model_store_directory/<runtime.model_name>/config/ instead.
    assert task.model_config == {"model": {"name": "conv1d-ae"}}
    assert task.history == [{"metrics": {"epoch": 1, "total_loss": 1.0}}]


def test_read_entropy_campaign_failed_task_has_no_artifacts(tmp_path: Path) -> None:
    status_directory = tmp_path / "entropy_run" / "plan" / "status"
    model_store_directory = tmp_path / "workspace" / "models" / "kidney"
    _write_entropy_task(
        status_directory,
        model_store_directory,
        task_id="task_000001",
        model_name="campaign__cfg_abc123__grid_0001__rep_00",
        grid_id="grid_0001",
        repetition=0,
        status="failed",
    )

    tasks = read_entropy_campaign(status_directory, model_store_directory)

    assert tasks[0].status == "failed"
    assert tasks[0].model_config is None
    assert tasks[0].history is None


def test_read_entropy_campaign_load_artifacts_false_skips_json(tmp_path: Path) -> None:
    status_directory = tmp_path / "entropy_run" / "plan" / "status"
    model_store_directory = tmp_path / "workspace" / "models" / "kidney"
    _write_entropy_task(
        status_directory,
        model_store_directory,
        task_id="task_000000",
        model_name="campaign__cfg_abc123__grid_0000__rep_00",
        grid_id="grid_0000",
        repetition=0,
    )

    tasks = read_entropy_campaign(
        status_directory, model_store_directory, load_artifacts=False
    )

    assert tasks[0].model_config is None
    assert tasks[0].history is None


def test_read_entropy_campaign_sorted_and_ignores_progress_files(tmp_path: Path) -> None:
    status_directory = tmp_path / "entropy_run" / "plan" / "status"
    model_store_directory = tmp_path / "workspace" / "models" / "kidney"
    for task_id, grid_id in (("task_000002", "grid_0002"), ("task_000000", "grid_0000")):
        _write_entropy_task(
            status_directory,
            model_store_directory,
            task_id=task_id,
            model_name=f"campaign__cfg_abc123__{grid_id}__rep_00",
            grid_id=grid_id,
            repetition=0,
        )

    tasks = read_entropy_campaign(status_directory, model_store_directory)

    assert [task.task_id for task in tasks] == ["task_000000", "task_000002"]


def test_read_entropy_campaign_missing_status_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_entropy_campaign(tmp_path / "missing", tmp_path / "models")
