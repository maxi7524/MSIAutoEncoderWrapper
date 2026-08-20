"""Tests for the generic grid-experiment campaign reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import (
    CampaignTask,
    read_campaign,
)
from msi_autoencoder_wrapper.utils.exceptions import WorkspaceConfigError

from tests.analysis._campaign_fixtures import EXPERIMENT_NAME, write_campaign_task


def test_read_campaign_parses_completed_task_and_ignores_progress_files(
    tmp_path: Path,
) -> None:
    write_campaign_task(
        tmp_path,
        task_id="task_000000",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
    )

    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, CampaignTask)
    assert task.task_id == "task_000000"
    assert task.status == "completed"
    assert task.repetition == 0
    assert task.grid_parameters["binning_steps"] == 0.5
    assert task.grid_parameters["architectures"]["name"] == "mlp-ae"
    assert task.model_config is not None
    assert task.model_config["model"]["name"] == "mlp-ae"
    assert task.history is not None
    # 3 default epochs + 1 trailing test-split evaluation entry.
    assert len(task.history) == 4


def test_read_campaign_load_artifacts_false_skips_json(tmp_path: Path) -> None:
    write_campaign_task(
        tmp_path,
        task_id="task_000000",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
    )

    tasks = read_campaign(tmp_path, EXPERIMENT_NAME, load_artifacts=False)

    assert tasks[0].model_config is None
    assert tasks[0].history is None


def test_read_campaign_incomplete_task_has_no_artifacts_even_when_requested(
    tmp_path: Path,
) -> None:
    write_campaign_task(
        tmp_path,
        task_id="task_000001",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=1,
        input_dim=6,
        status="failed",
    )

    tasks = read_campaign(tmp_path, EXPERIMENT_NAME, load_artifacts=True)

    assert tasks[0].status == "failed"
    assert tasks[0].result is None
    assert tasks[0].model_config is None
    assert tasks[0].history is None


def test_read_campaign_multiple_tasks_are_sorted_by_task_id(tmp_path: Path) -> None:
    write_campaign_task(
        tmp_path,
        task_id="task_000002",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
    )
    write_campaign_task(
        tmp_path,
        task_id="task_000000",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=1,
        input_dim=6,
    )
    write_campaign_task(
        tmp_path,
        task_id="task_000001",
        architecture_name="conv-ae",
        preset="CNNAutoencoder",
        binning_step=1.0,
        repetition=0,
        input_dim=3,
    )

    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    assert [task.task_id for task in tasks] == [
        "task_000000",
        "task_000001",
        "task_000002",
    ]


def test_read_campaign_missing_status_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceConfigError):
        read_campaign(tmp_path, "does-not-exist")
