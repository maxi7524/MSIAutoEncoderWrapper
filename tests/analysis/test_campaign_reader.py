"""Tests for the generic grid-experiment campaign reader."""

from __future__ import annotations

import shutil
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


def test_read_campaign_resolves_cfg_fingerprint_suffixed_directory(
    tmp_path: Path,
) -> None:
    """``runtime.naming.campaign_identifier`` namespaces campaign directories as
    ``<experiment_name>__cfg_<fingerprint>``; the reader must find that directory
    from the plain ``experiment_name`` alone when no exact-name directory exists."""
    write_campaign_task(
        tmp_path,
        task_id="task_000000",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
    )
    execution_root = tmp_path / "configs" / "execution"
    shutil.move(
        execution_root / EXPERIMENT_NAME,
        execution_root / f"{EXPERIMENT_NAME}__cfg_abc123def456",
    )

    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    assert len(tasks) == 1
    assert tasks[0].task_id == "task_000000"


def test_read_campaign_prefers_exact_directory_over_cfg_suffixed_one(
    tmp_path: Path,
) -> None:
    """A pre-existing exact-name directory (or compatibility symlink) always wins,
    even when a ``__cfg_*`` sibling is also present, so callers who already point at
    a specific materialized directory keep that exact behavior."""
    write_campaign_task(
        tmp_path,
        task_id="task_000000",
        architecture_name="exact-match",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
    )
    execution_root = tmp_path / "configs" / "execution"
    shutil.copytree(
        execution_root / EXPERIMENT_NAME,
        execution_root / f"{EXPERIMENT_NAME}__cfg_abc123def456",
    )

    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    assert tasks[0].grid_parameters["architectures"]["name"] == "exact-match"


def test_read_campaign_ambiguous_cfg_suffixed_directories_raises(
    tmp_path: Path,
) -> None:
    """Two distinct ``__cfg_*`` directories (e.g. the config changed and was
    re-planned) must never be silently merged or arbitrarily picked between."""
    write_campaign_task(
        tmp_path,
        task_id="task_000000",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
    )
    execution_root = tmp_path / "configs" / "execution"
    shutil.copytree(
        execution_root / EXPERIMENT_NAME,
        execution_root / f"{EXPERIMENT_NAME}__cfg_first000000",
    )
    shutil.move(
        execution_root / EXPERIMENT_NAME,
        execution_root / f"{EXPERIMENT_NAME}__cfg_second000000",
    )

    with pytest.raises(WorkspaceConfigError):
        read_campaign(tmp_path, EXPERIMENT_NAME)
