"""Architecture capacity and dataset overview for one architecture x binning campaign.

Interprets :class:`~..experiments.campaign_reader.CampaignTask` records produced by a
specific grid shape: ``grid_parameters == {"architectures": {...}, "binning_steps":
<float>}`` (the shape written by ``architecture_binning_experiment.yaml``-style
configs). Unlike :mod:`..experiments.campaign_reader`, this module is intentionally
not generic — it is the one place that knows this campaign's grid axis names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt

from ....models.model_loader import ModelLoader
from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from ..experiments.campaign_reader import CampaignTask

logger = get_custom_logger(__name__)


def architecture_grid_parameters(task: CampaignTask) -> Dict[str, Any]:
    """Extract this campaign's architecture identity and binning step for one task.

    :param task: Campaign task with populated ``grid_parameters``.
    :type task: CampaignTask
    :return: ``name``, ``preset``, and ``binning_step`` (as ``float``).
    :rtype: Dict[str, Any]
    :raises ValidationError: If ``grid_parameters`` does not carry this campaign's
        expected ``architectures``/``binning_steps`` axes.
    """
    architecture = task.grid_parameters.get("architectures")
    binning_step = task.grid_parameters.get("binning_steps")
    if not isinstance(architecture, dict) or binning_step is None:
        raise_validation_error(
            "ArchitectureOverview",
            f"Task '{task.task_id}' grid_parameters must define 'architectures' "
            "(mapping) and 'binning_steps' (scalar).",
        )
    return {
        "name": architecture.get("name"),
        "preset": architecture.get("preset"),
        "binning_step": float(binning_step),
    }


def parameter_count_table(tasks: Sequence[CampaignTask]) -> list[Dict[str, Any]]:
    """Return one row per distinct (architecture, binning step) with its real parameter count.

    Parameter counts are computed by reconstructing each unique architecture with
    :class:`ModelLoader` directly from a completed task's saved ``config.json``
    (uninitialized weights — only the module graph is needed) and summing
    ``numel()`` over every parameter. Repetitions of the same grid cell share an
    identical architecture and input dimensionality, so only the first completed
    task seen for each (architecture, binning step) pair is built; the remaining
    repetitions are skipped for this table.

    :param tasks: Campaign tasks with loaded ``model_config`` (see
        :func:`~..experiments.campaign_reader.read_campaign`).
    :type tasks: Sequence[CampaignTask]
    :return: Records keyed by ``architecture``, ``preset``, ``binning_step``, sorted
        by ``(architecture, binning_step)``. Each also carries ``latent_dim``,
        ``input_dim`` (both read from the built encoder's own parameters, not
        re-derived), and ``total_parameters``.
    :rtype: list[Dict[str, Any]]
    """
    seen: Dict[tuple[str, float], Dict[str, Any]] = {}
    for task in tasks:
        if task.model_config is None:
            continue
        grid = architecture_grid_parameters(task)
        key = (grid["name"], grid["binning_step"])
        if key in seen:
            continue

        # Real parameter count
        ## Reconstruct the exact module graph saved for this task; weights are not
        ## needed since parameter *counts* do not depend on their values.
        model, _, _ = ModelLoader.build(task.model_config)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        encoder_parameters = task.model_config["model"]["components"]["encoder"]["parameters"]
        seen[key] = {
            "architecture": grid["name"],
            "preset": grid["preset"],
            "binning_step": grid["binning_step"],
            "latent_dim": encoder_parameters.get("latent_dim"),
            "input_dim": encoder_parameters.get("input_dim"),
            "total_parameters": int(total_parameters),
        }
    records = sorted(seen.values(), key=lambda row: (row["architecture"], row["binning_step"]))
    logger.info(
        "Computed parameter counts for %s architecture x binning combination(s).",
        len(records),
    )
    return records


def dataset_summary(tasks: Sequence[CampaignTask]) -> Dict[str, Any]:
    """Summarize the shared dataset split behind every task in this campaign.

    Reads the split assignment from the first task with a loaded ``model_config``.
    Every grid cell and repetition in this campaign shares the same subset fraction
    and split seed (``common_seeds.split``), so the split sample counts are identical
    across the whole grid; this function does not aggregate across tasks.

    :param tasks: Campaign tasks with loaded ``model_config``.
    :type tasks: Sequence[CampaignTask]
    :return: Normalization strategy, split strategy/seed, and one sample count per
        split partition plus their ``total``.
    :rtype: Dict[str, Any]
    :raises ValidationError: If no task carries a loaded ``model_config``.
    """
    task = next((candidate for candidate in tasks if candidate.model_config is not None), None)
    if task is None:
        raise_validation_error(
            "ArchitectureOverview",
            "No completed task with a loaded model configuration is available.",
        )
    dataset_parameters = task.model_config["data"]["dataset"]["parameters"]
    split = dataset_parameters.get("split", {})
    assignments = split.get("assignments", {})
    counts = {name: len(indices) for name, indices in assignments.items()}
    return {
        "normalization": dataset_parameters.get("normalization"),
        "split_strategy": split.get("strategy"),
        "split_seed": split.get("seed"),
        **counts,
        "total": sum(counts.values()),
    }


def plot_parameter_counts(
    records: Sequence[Dict[str, Any]],
    theme: VisualizationTheme | str | None = None,
    log_scale: bool = True,
) -> Any:
    """Plot total parameter count against binning step, one line per architecture.

    Architecture families in this campaign differ by close to two orders of
    magnitude in parameter count (e.g. a strided CNN vs. a dense MLP encoder/decoder
    of the same latent dimension) — on a linear axis the smaller architecture's line
    is visually indistinguishable from zero. ``log_scale`` defaults to ``True`` for
    this reason; set it to ``False`` only when comparing architectures already known
    to be within a similar order of magnitude.

    :param records: Output of :func:`parameter_count_table`.
    :type records: Sequence[Dict[str, Any]]
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :param log_scale: Use a logarithmic parameter-count axis.
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
            [row["total_parameters"] for row in rows],
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
        ylabel="total parameters",
        title="Model capacity vs. binning resolution",
    )
    ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    ax.legend(
        loc=resolved.legend_location,
        frameon=resolved.legend_frame,
        ncols=resolved.legend_columns,
    )
    return figure, ax
