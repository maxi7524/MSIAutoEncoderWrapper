"""Tests for architecture capacity and dataset overview analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import read_campaign
from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.architecture_overview_analysis import (
    architecture_grid_parameters,
    dataset_summary,
    parameter_count_table,
    plot_parameter_counts,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError

from tests.analysis._campaign_fixtures import EXPERIMENT_NAME, write_campaign_task

# input_dim=6, hidden_dims=[4], latent_dim=2, batch_normalization=False:
# encoder (6*4+4) + (4*2+2) = 28 + 10 = 38; decoder (2*4+4) + (4*6+6) = 12 + 30 = 42.
_EXPECTED_TOTAL_PARAMETERS_INPUT_6 = 80


def _write_grid(tmp_path: Path) -> None:
    # 2 architectures x 2 binning steps x 2 repetitions; binning changes input_dim,
    # mirroring how a coarser bin step shrinks the m/z axis.
    for repetition in (0, 1):
        write_campaign_task(
            tmp_path,
            task_id=f"task_mlp_045_{repetition}",
            architecture_name="mlp-ae",
            preset="MLPAutoencoder",
            binning_step=0.45,
            repetition=repetition,
            input_dim=6,
        )
        write_campaign_task(
            tmp_path,
            task_id=f"task_mlp_090_{repetition}",
            architecture_name="mlp-ae",
            preset="MLPAutoencoder",
            binning_step=0.90,
            repetition=repetition,
            input_dim=3,
        )
        write_campaign_task(
            tmp_path,
            task_id=f"task_conv_045_{repetition}",
            architecture_name="conv-ae",
            preset="CNNAutoencoder",
            binning_step=0.45,
            repetition=repetition,
            input_dim=6,
        )


def test_parameter_count_table_deduplicates_repetitions_and_counts_real_parameters(
    tmp_path: Path,
) -> None:
    _write_grid(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    records = parameter_count_table(tasks)

    # (mlp-ae, 0.45), (mlp-ae, 0.90), (conv-ae, 0.45) = 3 rows, independent of the
    # 2 repetitions each (conv-ae is only present at binning_step=0.45).
    assert len(records) == 3
    row = next(
        row
        for row in records
        if row["architecture"] == "mlp-ae" and row["binning_step"] == 0.45
    )
    assert row["input_dim"] == 6
    assert row["latent_dim"] == 2
    assert row["total_parameters"] == _EXPECTED_TOTAL_PARAMETERS_INPUT_6


def test_parameter_count_table_reflects_input_dim_change_from_binning(
    tmp_path: Path,
) -> None:
    _write_grid(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    records = parameter_count_table(tasks)

    coarse = next(
        row
        for row in records
        if row["architecture"] == "mlp-ae" and row["binning_step"] == 0.90
    )
    fine = next(
        row
        for row in records
        if row["architecture"] == "mlp-ae" and row["binning_step"] == 0.45
    )
    assert coarse["input_dim"] == 3
    assert coarse["total_parameters"] < fine["total_parameters"]


def test_dataset_summary_reports_split_counts(tmp_path: Path) -> None:
    _write_grid(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    summary = dataset_summary(tasks)

    assert summary["train"] == 8
    assert summary["validation"] == 1
    assert summary["test"] == 1
    assert summary["total"] == 10
    assert summary["normalization"] == "tic"


def test_architecture_grid_parameters_rejects_missing_grid_axes() -> None:
    from msi_autoencoder_wrapper.analysis.autoencoder.experiments import CampaignTask

    task = CampaignTask(
        task_id="task_x",
        status="completed",
        grid_parameters={"unrelated_axis": 1},
        repetition=0,
        result=None,
        model_config=None,
        history=None,
    )

    with pytest.raises(ValidationError):
        architecture_grid_parameters(task)


def test_plot_parameter_counts_draws_one_line_per_architecture(tmp_path: Path) -> None:
    _write_grid(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    records = parameter_count_table(tasks)

    figure, ax = plot_parameter_counts(records)

    assert len(ax.lines) == 2  # one per architecture (mlp-ae, conv-ae)
    assert ax.get_yscale() == "log"  # log_scale defaults to True
