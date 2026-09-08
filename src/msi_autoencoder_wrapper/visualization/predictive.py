"""Shared distribution-first figures for molecular head selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from .metrics import plot_violin_with_points
from .theme import resolve_theme


def _color(label, theme):
    """Assign stable related shades within a head-loss family."""
    families = ("ClassBalancedMultiLabelBCELoss", "PositiveWeightedMultiLabelBCELoss", "SignalMaskedBCELoss",
                "ThreeStateCrossEntropyLoss", "VariationalPULoss", "SymmetricPURankingLoss")
    if label in theme.model_overrides:
        return theme.model_overrides[label]
    family = next((index for index, name in enumerate(families) if name in label), 0)
    base = np.asarray(to_rgb(theme.model_palette[family % len(theme.model_palette)]))
    amount = .3 if "per_class" in label else .15 if "global" in label else 0
    return tuple(base + amount * (1 - base))


def distributions(frame: pd.DataFrame, *, value: str = "value", group: str = "label", title: str = "", theme=None):
    """Plot complete group distributions and a deterministic point overlay.

    :param frame: Observation-level records.
    :param value: Numeric column on the y-axis.
    :param group: Semantic grouping column on the x-axis.
    :param title: Figure title.
    :param theme: Existing visualization theme or preset.
    :return: Matplotlib figure. Overlay keeps <=200 evenly spaced ordered rows.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    figure, ax = plt.subplots(figsize=(max(10, frame[group].nunique() * 1.5), 6), dpi=resolved.figure_dpi)
    labels = []
    for position, (label, part) in enumerate(frame.groupby(group, sort=True)):
        values = part[value].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        labels.append(label)
        color = _color(label, resolved)
        if len(values) > 1 and np.ptp(values) > 0:
            shown = values[np.linspace(0, len(values) - 1, min(len(values), 200), dtype=int)]
            plot_violin_with_points(values, position, ax=ax, color=color, overlay_values=shown, theme=resolved)
        elif len(values):
            ax.scatter(np.full(len(values), position), values, color=color, alpha=resolved.primary_alpha)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    ax.set(ylabel=value, title=title)
    figure.tight_layout()
    return figure


def trajectories(frame: pd.DataFrame, *, x: str, y: str = "value", line: str = "model_id", title: str = "", theme=None):
    """Plot every individual trajectory with semantic model labels.

    :param frame: Long-form observations with label and line identifiers.
    :param x: Ordered horizontal coordinate.
    :param y: Vertical measurement column.
    :param line: Independent trajectory identifier.
    :param title: Figure title.
    :param theme: Existing theme.
    :return: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    for position, (label, part) in enumerate(frame.groupby("label", sort=True)):
        for number, (_, run) in enumerate(part.groupby(line)):
            run = run.sort_values(x)
            ax.plot(run[x], run[y], color=_color(label, resolved),
                    alpha=resolved.overlapping_signal_alpha, label=label if number == 0 else None)
    ax.set(xlabel=x, ylabel=y, title=title)
    if len(frame):
        ax.legend(fontsize=6)
    figure.tight_layout()
    return figure


def scatter(frame: pd.DataFrame, *, x: str, y: str, title: str = "", theme=None):
    """Plot paired measurements with one point per supplied observation.

    :param frame: Paired records, including label for grouping.
    :param x: Horizontal measurement.
    :param y: Vertical measurement.
    :param title: Figure title.
    :param theme: Existing theme.
    :return: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    for position, (label, part) in enumerate(frame.groupby("label", sort=True)):
        ax.scatter(part[x], part[y], label=label, alpha=resolved.marker_alpha,
                   color=_color(label, resolved), s=12)
    ax.set(xlabel=x, ylabel=y, title=title)
    if len(frame):
        ax.legend(fontsize=6)
    figure.tight_layout()
    return figure


def spectrum_cases(frame: pd.DataFrame, *, theme=None):
    """Show input and reconstruction for each explicitly retained example.

    :param frame: One model/split's long-form spectrum_cases table.
    :param theme: Existing theme.
    :return: Matplotlib figure with one panel per retained spectrum.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    groups = list(frame.groupby("row_position"))
    figure, axes = plt.subplots(max(1, len(groups)), 1, figsize=(12, max(3, 2.5 * len(groups))), squeeze=False)
    for ax, (row, part) in zip(axes[:, 0], groups):
        ax.plot(part.mz, part.input, label="Input", color=resolved.input_color)
        ax.plot(part.mz, part.reconstruction, label="Reconstruction", alpha=resolved.secondary_alpha)
        ax.set(xlabel="m/z", ylabel="TIC-normalized intensity", title=f"Row {row}; selected as worst: {bool(part.selected_as_worst.iloc[0])}")
        ax.legend()
    figure.tight_layout()
    return figure


def pair_histograms(frame: pd.DataFrame, *, title: str = "", theme=None):
    """Plot densities from complete pair counts without expanding the observations.

    :param frame: One split/space/metric of per-run histogram counts and bin edges.
    :param title: Figure title; must identify the measured distance or similarity.
    :param theme: Existing visualization theme.
    :return: Matplotlib figure; each line is one run's full pair distribution.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    for position, (label, group) in enumerate(frame.groupby("label", sort=True)):
        for number, (_, run) in enumerate(group.groupby("model_id")):
            run = run.sort_values("left_edge")
            widths = (run.right_edge - run.left_edge).to_numpy()  # (H,)
            total = run["count"].sum()
            density = run["count"].to_numpy() / (max(1, total) * widths)  # (H,)
            edges = np.r_[run.left_edge.to_numpy(), run.right_edge.iloc[-1]]  # (H+1,)
            ax.stairs(density, edges, color=_color(label, resolved),
                      alpha=resolved.overlapping_signal_alpha, label=label if number == 0 else None)
    ax.set(xlabel=frame.metric.iloc[0] if len(frame) else "distance", ylabel="Pair density", title=title)
    if len(frame):
        ax.legend(fontsize=6)
    figure.tight_layout()
    return figure
