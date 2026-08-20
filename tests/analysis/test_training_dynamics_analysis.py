"""Tests for per-epoch training-dynamics analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import read_campaign
from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.architecture_overview_analysis import (
    parameter_count_table,
)
from msi_autoencoder_wrapper.analysis.autoencoder.reconstruction.training_dynamics_analysis import (
    _mean_curve,
    epoch_duration_summary,
    final_performance_summary,
    plot_epoch_duration,
    plot_training_curves_by_binning,
    training_curves_frame,
)

from tests.analysis._campaign_fixtures import EXPERIMENT_NAME, write_campaign_task


def _write_two_repetitions(tmp_path: Path) -> None:
    write_campaign_task(
        tmp_path,
        task_id="task_rep0",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
        durations=[10.0, 10.0, 10.0],
        train_losses=[3.0, 2.0, 1.0],
        val_losses=[3.5, 2.5, 1.5],
    )
    write_campaign_task(
        tmp_path,
        task_id="task_rep1",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=1,
        input_dim=6,
        durations=[20.0, 20.0, 20.0],
        train_losses=[4.0, 3.0, 2.0],
        val_losses=[4.5, 3.5, 2.5],
    )


def test_training_curves_frame_excludes_trailing_test_entry(tmp_path: Path) -> None:
    _write_two_repetitions(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)

    frame = training_curves_frame(tasks)

    # 2 repetitions x 3 epochs = 6 rows; the trailing test-split entry is excluded.
    assert len(frame) == 6
    assert all(row["epoch"] in (1, 2, 3) for row in frame)
    row = next(row for row in frame if row["task_id"] == "task_rep0" and row["epoch"] == 1)
    assert row["train_loss"] == 3.0
    assert row["validation_loss"] == 3.5
    assert row["best_loss"] == 3.5  # running-minimum validation loss as of epoch 1
    assert row["duration"] == 10.0
    assert row["architecture"] == "mlp-ae"
    assert row["binning_step"] == 0.5


def test_training_curves_frame_skips_tasks_without_history(tmp_path: Path) -> None:
    write_campaign_task(
        tmp_path,
        task_id="task_failed",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=0.5,
        repetition=0,
        input_dim=6,
        status="failed",
    )

    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    frame = training_curves_frame(tasks)

    assert frame == []


def test_mean_curve_averages_independently_per_epoch_with_ragged_repetitions() -> None:
    rows = [
        {"epoch": 1, "value": 10.0},
        {"epoch": 2, "value": 20.0},
        {"epoch": 1, "value": 30.0},
        {"epoch": 2, "value": 40.0},
        {"epoch": 3, "value": 50.0},  # only one repetition reaches epoch 3
    ]

    epochs, means = _mean_curve(rows, "value")

    np.testing.assert_array_equal(epochs, np.array([1, 2, 3]))
    np.testing.assert_allclose(means, np.array([20.0, 30.0, 50.0]))


def test_epoch_duration_summary_joins_parameter_counts(tmp_path: Path) -> None:
    _write_two_repetitions(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    frame = training_curves_frame(tasks)
    parameter_counts = parameter_count_table(tasks)

    summary = epoch_duration_summary(frame, parameter_counts)

    assert len(summary) == 1
    row = summary[0]
    assert row["architecture"] == "mlp-ae"
    assert row["binning_step"] == 0.5
    assert row["epoch_count"] == 6
    assert row["mean_epoch_duration"] == 15.0  # mean of (10,10,10,20,20,20)
    assert row["mean_total_duration"] == 45.0  # mean of per-run totals (30, 60)
    assert row["total_parameters"] == 80


def test_final_performance_summary_ranks_by_best_validation_loss(tmp_path: Path) -> None:
    _write_two_repetitions(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    frame = training_curves_frame(tasks)

    summary = final_performance_summary(frame)

    assert len(summary) == 1
    row = summary[0]
    assert row["architecture"] == "mlp-ae"
    assert row["binning_step"] == 0.5
    assert row["run_count"] == 2
    # rep0's running-minimum validation loss reaches 1.5 by epoch 3; rep1 reaches 2.5.
    assert row["best_task_id"] == "task_rep0"
    assert row["best_validation_loss"] == 1.5
    assert row["mean_best_validation_loss"] == 2.0


def test_plot_epoch_duration_draws_one_line_per_architecture(tmp_path: Path) -> None:
    _write_two_repetitions(tmp_path)
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    frame = training_curves_frame(tasks)
    parameter_counts = parameter_count_table(tasks)
    summary = epoch_duration_summary(frame, parameter_counts)

    figure, ax = plot_epoch_duration(summary)

    assert len(ax.lines) == 1  # one architecture (mlp-ae) in this fixture
    assert ax.get_yscale() == "linear"  # log_scale defaults to False here


def test_plot_training_curves_by_binning_returns_one_figure_per_binning_step(
    tmp_path: Path,
) -> None:
    _write_two_repetitions(tmp_path)
    write_campaign_task(
        tmp_path,
        task_id="task_other_binning",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=1.0,
        repetition=0,
        input_dim=3,
    )
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    frame = training_curves_frame(tasks)

    figures = plot_training_curves_by_binning(frame)

    assert set(figures.keys()) == {0.5, 1.0}
    figure, ax = figures[0.5]
    assert ax.get_yscale() == "log"  # log_scale defaults to True
    # 2 repetitions x 2 series (train, validation) unlabeled + 2 mean (labeled)
    # lines for the one architecture present at this binning step.
    assert len(ax.lines) == 6


def test_plot_training_curves_by_binning_shares_axis_scale_across_binning_steps(
    tmp_path: Path,
) -> None:
    _write_two_repetitions(tmp_path)
    write_campaign_task(
        tmp_path,
        task_id="task_other_binning",
        architecture_name="mlp-ae",
        preset="MLPAutoencoder",
        binning_step=1.0,
        repetition=0,
        input_dim=3,
        train_losses=[400.0, 300.0, 200.0],
        val_losses=[450.0, 350.0, 250.0],
    )
    tasks = read_campaign(tmp_path, EXPERIMENT_NAME)
    frame = training_curves_frame(tasks)

    figures = plot_training_curves_by_binning(frame)

    _, ax_low = figures[0.5]
    _, ax_high = figures[1.0]
    # Each binning step's own values differ by two orders of magnitude, so equal
    # limits are only possible if the range was computed once across the whole frame.
    assert ax_low.get_ylim() == ax_high.get_ylim()
    assert ax_low.get_xlim() == ax_high.get_xlim()
