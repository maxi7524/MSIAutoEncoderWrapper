"""Model-independent metric distribution and trade-off plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .theme import VisualizationTheme, resolve_theme


def plot_metric_distribution(values: np.ndarray, metric: str, ax: Axes | None = None, bins: int = 40, kind: str = "histogram", label: str | None = None, theme: VisualizationTheme | str | None = None, color: str | None = None, histtype: str = "bar"):
    """Plot one histogram or ECDF on an optional caller-owned axis.

    Repeated calls on the same ``ax`` overlay several groups. ``stepfilled`` draws a
    translucent area plus a stronger outline without duplicating its legend entry.
    """
    resolved = resolve_theme(theme)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    if kind == "histogram":
        density = histtype != "bar"
        if histtype == "stepfilled":
            ax.hist(values, bins=bins, density=density, histtype="stepfilled", color=color, alpha=resolved.distribution_fill_alpha, linewidth=0.0, label=label)
            ax.hist(values, bins=bins, density=density, histtype="step", color=color, alpha=resolved.distribution_edge_alpha, linewidth=resolved.distribution_line_width, label="_nolegend_")
        else:
            alpha = resolved.secondary_alpha if histtype == "bar" else resolved.distribution_edge_alpha
            ax.hist(values, bins=bins, density=density, histtype=histtype, linewidth=resolved.distribution_line_width, alpha=alpha, color=color, label=label)
    elif kind == "ecdf":
        ordered = np.sort(values)
        ax.step(ordered, np.arange(1, ordered.size + 1) / max(1, ordered.size), where="post", color=color, linewidth=resolved.line_width, alpha=resolved.primary_alpha, label=label)
    else:
        raise ValueError("kind must be histogram or ecdf")
    ax.set(xlabel=metric, ylabel="density" if (kind == "histogram" and histtype != "bar") else ("Spectrum count" if kind == "histogram" else "ECDF"))
    ax.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
    if label:
        ax.legend(fontsize=resolved.legend_font_size, loc=resolved.legend_location, ncols=resolved.legend_columns, frameon=resolved.legend_frame)
    return figure, ax


def plot_metric_tradeoff(x: Sequence[float], y: Sequence[float], x_label: str, metric: str, ax: Axes | None = None, label: str | None = None, theme: VisualizationTheme | str | None = None):
    """Plot one metric against a configurable complexity or compression axis."""
    resolved = resolve_theme(theme)
    if ax is None: figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else: figure = ax.figure
    color = resolved.color_for_model(label or metric)
    ax.plot(x, y, marker="o", color=color, linewidth=resolved.line_width, markersize=resolved.marker_size, markerfacecolor=color, markeredgecolor=resolved.panel_color, markeredgewidth=resolved.marker_edge_width, alpha=resolved.primary_alpha, label=label)
    ax.set(xlabel=x_label, ylabel=metric)
    ax.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
    if label:
        ax.legend(fontsize=resolved.legend_font_size, loc=resolved.legend_location, ncols=resolved.legend_columns, frameon=resolved.legend_frame)
    return figure, ax
