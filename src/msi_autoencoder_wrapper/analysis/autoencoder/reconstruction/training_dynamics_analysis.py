"""Per-epoch training dynamics across one architecture x binning campaign.

Companion to :mod:`.architecture_overview_analysis`: shares its assumption about the
campaign's grid shape (``architectures`` x ``binning_steps``) via
:func:`~.architecture_overview_analysis.architecture_grid_parameters`, and reads
per-epoch loss/duration records out of each :class:`CampaignTask`'s loaded
``history.json``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from ..experiments.campaign_reader import CampaignTask
from .architecture_overview_analysis import architecture_grid_parameters

logger = get_custom_logger(__name__)

_LOSS_SERIES = (("train_loss", "solid", "train"), ("validation_loss", "dashed", "validation"))


def training_curves_frame(tasks: Sequence[CampaignTask]) -> list[Dict[str, Any]]:
    """Flatten every task's per-epoch history into one long-form record list.

    One row per ``(task, epoch)``. Only ``history`` entries carrying a
    ``metrics.epoch`` value are included — a trailing post-training evaluation entry
    (``split: test``, no ``epoch``/``duration``) is skipped here by design; it is not
    part of the training-time curve.

    :param tasks: Campaign tasks with loaded ``history`` (see
        :func:`~..experiments.campaign_reader.read_campaign`).
    :type tasks: Sequence[CampaignTask]
    :return: Records with ``task_id``, ``architecture``, ``binning_step``,
        ``repetition``, ``epoch``, ``train_loss`` (``total_loss``), ``validation_loss``
        (``validation_total_loss``), ``best_loss`` (the trainer's own running-minimum
        validation loss as of this epoch — monotonically non-increasing, so the last
        epoch's value is the best validation loss reached over the whole run; see
        :func:`final_performance_summary`), and ``duration`` (seconds).
    :rtype: list[Dict[str, Any]]
    """
    rows: list[Dict[str, Any]] = []
    tasks_without_history = 0
    for task in tasks:
        if task.history is None:
            tasks_without_history += 1
            continue
        grid = architecture_grid_parameters(task)
        for entry in task.history:
            metrics = entry.get("metrics", {})
            epoch = metrics.get("epoch")
            if epoch is None:
                continue
            rows.append(
                {
                    "task_id": task.task_id,
                    "architecture": grid["name"],
                    "binning_step": grid["binning_step"],
                    "repetition": task.repetition,
                    "epoch": int(epoch),
                    "train_loss": metrics.get("total_loss"),
                    "validation_loss": metrics.get("validation_total_loss"),
                    "best_loss": metrics.get("best_loss"),
                    "duration": metrics.get("duration"),
                }
            )
    if tasks_without_history:
        logger.warning(
            "%s task(s) had no loaded training history and were skipped.",
            tasks_without_history,
        )
    logger.info(
        "Built training-curve frame with %s epoch row(s) from %s task(s).",
        len(rows),
        len(tasks) - tasks_without_history,
    )
    return rows


def final_performance_summary(frame: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Rank (architecture, binning step) combinations by their best validation loss.

    Per run (task), the last epoch's ``best_loss`` is the trainer's own
    running-minimum validation loss — the checkpointed value restored at the end of
    training (``training.checkpoint.restore_best``) — so it is read directly rather
    than re-derived by taking ``min(validation_loss)`` independently here. Runs are
    then grouped by ``(architecture, binning_step)`` and summarized across
    repetitions.

    :param frame: Output of :func:`training_curves_frame`.
    :type frame: Sequence[Dict[str, Any]]
    :return: One record per (architecture, binning step), sorted ascending by
        ``mean_best_validation_loss`` (best combination first). Each record also
        carries ``std_best_validation_loss``, the single best-performing
        ``best_task_id``/``best_validation_loss`` among its repetitions, and
        ``run_count``.
    :rtype: list[Dict[str, Any]]
    """
    rows_by_task: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in frame:
        rows_by_task[row["task_id"]].append(row)

    runs: list[Dict[str, Any]] = []
    for task_id, rows in rows_by_task.items():
        last = max(rows, key=lambda row: row["epoch"])
        if last["best_loss"] is None:
            continue
        runs.append(
            {
                "task_id": task_id,
                "architecture": last["architecture"],
                "binning_step": last["binning_step"],
                "best_validation_loss": last["best_loss"],
            }
        )

    grouped: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["architecture"], run["binning_step"])].append(run)

    records = []
    for (architecture, binning_step), group in grouped.items():
        values = np.asarray([run["best_validation_loss"] for run in group], dtype=float)
        best_run = min(group, key=lambda run: run["best_validation_loss"])
        records.append(
            {
                "architecture": architecture,
                "binning_step": binning_step,
                "mean_best_validation_loss": float(np.mean(values)),
                "std_best_validation_loss": float(np.std(values)),
                "best_task_id": best_run["task_id"],
                "best_validation_loss": float(best_run["best_validation_loss"]),
                "run_count": len(group),
            }
        )
    return sorted(records, key=lambda row: row["mean_best_validation_loss"])


def epoch_duration_summary(
    frame: Sequence[Dict[str, Any]],
    parameter_counts: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Summarize training speed per (architecture, binning step), joined to capacity.

    Epoch duration depends on both the architecture and the input dimensionality (set
    by the binning step), so durations are grouped by the same
    ``(architecture, binning_step)`` key as :func:`architecture_overview_analysis.
    parameter_count_table`, not by architecture alone.

    :param frame: Output of :func:`training_curves_frame`.
    :type frame: Sequence[Dict[str, Any]]
    :param parameter_counts: Output of :func:`architecture_overview_analysis.
        parameter_count_table`, joined in for the ``total_parameters`` column.
    :type parameter_counts: Sequence[Dict[str, Any]]
    :return: Records with ``architecture``, ``binning_step``, ``total_parameters``,
        ``mean_epoch_duration``/``std_epoch_duration`` (seconds, pooled across every
        individual epoch measurement and repetition), ``mean_total_duration`` (seconds
        — one full run's summed epoch durations, averaged across repetitions; the
        practical "how long does training this model actually take" figure), and
        ``epoch_count`` (the number of individual epoch measurements pooled).
    :rtype: list[Dict[str, Any]]
    """
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    total_by_task: dict[str, float] = defaultdict(float)
    key_by_task: dict[str, tuple[str, float]] = {}
    for row in frame:
        if row["duration"] is None:
            continue
        key = (row["architecture"], row["binning_step"])
        grouped[key].append(row["duration"])
        total_by_task[row["task_id"]] += row["duration"]
        key_by_task[row["task_id"]] = key

    totals_grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for task_id, total in total_by_task.items():
        totals_grouped[key_by_task[task_id]].append(total)

    parameter_lookup = {
        (row["architecture"], row["binning_step"]): row["total_parameters"]
        for row in parameter_counts
    }
    records = []
    for key, durations in sorted(grouped.items()):
        values = np.asarray(durations, dtype=float)
        totals = np.asarray(totals_grouped[key], dtype=float)
        architecture, binning_step = key
        records.append(
            {
                "architecture": architecture,
                "binning_step": binning_step,
                "total_parameters": parameter_lookup.get(key),
                "mean_epoch_duration": float(np.mean(values)),
                "std_epoch_duration": float(np.std(values)),
                "mean_total_duration": float(np.mean(totals)),
                "epoch_count": int(values.size),
            }
        )
    return records


def plot_epoch_duration(
    records: Sequence[Dict[str, Any]],
    theme: VisualizationTheme | str | None = None,
    log_scale: bool = False,
) -> Any:
    """Plot mean epoch duration against binning step, one line per architecture.

    Mirrors :func:`architecture_overview_analysis.plot_parameter_counts`'s layout so
    the two are directly comparable side by side (same x-axis, same color-per-
    architecture convention).

    :param records: Output of :func:`epoch_duration_summary`.
    :type records: Sequence[Dict[str, Any]]
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :param log_scale: Use a logarithmic duration axis (durations in this campaign
        span under one order of magnitude, so this defaults to ``False`` unlike
        :func:`~.architecture_overview_analysis.plot_parameter_counts`).
    :type log_scale: bool
    :return: Figure and axes.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    resolved = resolve_theme(theme)
    figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    architectures = sorted({row["architecture"] for row in records})
    for index, architecture in enumerate(architectures):
        rows = sorted(
            (row for row in records if row["architecture"] == architecture),
            key=lambda row: row["binning_step"],
        )
        color = resolved.color_for_model(architecture, index)
        ax.plot(
            [row["binning_step"] for row in rows],
            [row["mean_epoch_duration"] for row in rows],
            marker="o",
            color=color,
            linewidth=resolved.line_width,
            markersize=resolved.marker_size,
            label=architecture,
        )
    if log_scale:
        ax.set_yscale("log")
    ax.set(
        xlabel="binning step (Δm/z)",
        ylabel="mean epoch duration (s)",
        title="Training speed vs. binning resolution",
    )
    ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    ax.legend(
        loc=resolved.legend_location,
        frameon=resolved.legend_frame,
        ncols=resolved.legend_columns,
    )
    return figure, ax


def plot_training_curves_by_binning(
    frame: Sequence[Dict[str, Any]],
    theme: VisualizationTheme | str | None = None,
    log_scale: bool = True,
) -> Dict[float, Any]:
    """Plot train and validation loss vs. epoch, one figure per binning step.

    Every binning-step figure shares one axes. Color encodes architecture
    (:meth:`VisualizationTheme.color_for_model`, consistent across every figure);
    linestyle encodes train (solid) vs. validation (dashed) — training and
    validation are overlaid on the same axes, not split into separate panels, so the
    generalization gap (the vertical distance between a solid and its matching dashed
    line) is directly visible rather than requiring the reader to compare two plots.
    For each architecture, every repetition is drawn as its own thin, low-alpha line
    (no legend entry — only the group matters, not any single run), overlaid with one
    full-opacity mean-across-repetitions line (the labeled legend entry, labeled
    ``"<architecture> (train)"``/``"<architecture> (validation)"``).

    Binning steps are never mixed on one figure: model saturation is read per
    binning, since input dimensionality (and therefore achievable reconstruction
    error) differs between them. Every returned axes shares an **identical** x/y
    range (computed once from the whole ``frame``, not per binning step) so the
    figures remain visually comparable to each other — without this, each binning
    step's independent autoscaling would make e.g. "does a coarser binning converge
    lower?" impossible to read off the plots directly.

    :param frame: Output of :func:`training_curves_frame`.
    :type frame: Sequence[Dict[str, Any]]
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :param log_scale: Use a logarithmic loss axis. Loss drops by roughly two orders
        of magnitude within the first few epochs in this campaign, which collapses
        every epoch past the first into a flat line near zero on a linear axis, so
        this defaults to ``True``.
    :type log_scale: bool
    :return: Figure and axes pair keyed by ``binning_step``.
    :rtype: Dict[float, Any]
    """
    resolved = resolve_theme(theme)
    binning_steps = sorted({row["binning_step"] for row in frame})
    architectures = sorted({row["architecture"] for row in frame})
    x_limits, y_limits = _shared_axis_limits(frame, log_scale=log_scale)

    figures: Dict[float, Any] = {}
    for binning_step in binning_steps:
        binning_rows = [row for row in frame if row["binning_step"] == binning_step]
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
        for index, architecture in enumerate(architectures):
            architecture_rows = [
                row for row in binning_rows if row["architecture"] == architecture
            ]
            if not architecture_rows:
                continue
            color = resolved.color_for_model(architecture, index)
            by_repetition: dict[Any, list[Dict[str, Any]]] = defaultdict(list)
            for row in architecture_rows:
                by_repetition[row["repetition"]].append(row)

            for value_key, linestyle, series_name in _LOSS_SERIES:
                # Individual repetitions
                ## Unlabeled ("_nolegend_") so the legend carries one entry per
                ## (architecture, series), not one per repetition.
                for repetition_rows in by_repetition.values():
                    ordered = sorted(repetition_rows, key=lambda row: row["epoch"])
                    ax.plot(
                        [row["epoch"] for row in ordered],
                        [row[value_key] for row in ordered],
                        color=color,
                        linestyle=linestyle,
                        linewidth=resolved.reference_line_width,
                        alpha=resolved.overlapping_signal_alpha,
                        label="_nolegend_",
                    )
                # Mean across repetitions
                mean_epochs, mean_values = _mean_curve(architecture_rows, value_key)
                ax.plot(
                    mean_epochs,
                    mean_values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=resolved.reconstruction_line_width,
                    label=f"{architecture} ({series_name})",
                )

        ax.set(
            xlabel="epoch",
            ylabel="loss",
            title=f"Training vs. validation loss (Δm/z={binning_step})",
        )
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
        ax.legend(
            loc=resolved.legend_location,
            frameon=resolved.legend_frame,
            ncols=resolved.legend_columns,
        )
        figure.tight_layout()
        figures[binning_step] = (figure, ax)
    return figures


def _shared_axis_limits(
    frame: Sequence[Dict[str, Any]],
    *,
    log_scale: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute one x/y range covering every binning step's train/validation values.

    :param frame: Output of :func:`training_curves_frame`.
    :type frame: Sequence[Dict[str, Any]]
    :param log_scale: Whether the caller will render the y-axis logarithmically
        (only strictly positive values are considered for the y-range in that case).
    :type log_scale: bool
    :return: ``(x_min, x_max)``, ``(y_min, y_max)`` with a small margin applied.
    :rtype: tuple[tuple[float, float], tuple[float, float]]
    """
    epochs = [row["epoch"] for row in frame]
    losses = [
        row[value_key]
        for row in frame
        for value_key, _, _ in _LOSS_SERIES
        if row[value_key] is not None and (not log_scale or row[value_key] > 0)
    ]
    x_min, x_max = (min(epochs), max(epochs)) if epochs else (0.0, 1.0)
    y_min, y_max = (min(losses), max(losses)) if losses else (1e-6, 1.0)
    if log_scale:
        y_range = (y_min * 0.9, y_max * 1.1)
    else:
        margin = 0.05 * (y_max - y_min or y_max or 1.0)
        y_range = (y_min - margin, y_max + margin)
    return (x_min, x_max), y_range


def _mean_curve(
    rows: Sequence[Dict[str, Any]],
    value_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Average one metric across repetitions independently at every epoch.

    Grouping by epoch (rather than assuming equal-length repetitions) tolerates
    repetitions that stopped early relative to others in the same group.
    """
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row[value_key]
        if value is not None:
            by_epoch[row["epoch"]].append(value)
    epochs = sorted(by_epoch)
    means = [float(np.mean(by_epoch[epoch])) for epoch in epochs]
    return np.asarray(epochs), np.asarray(means)
