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


def plot_violin_with_points(values: np.ndarray, position: float, ax: Axes | None = None, width: float = 0.8, color: str | None = None, jitter: float = 0.05, point_size: float = 18.0, label: str | None = None, theme: VisualizationTheme | str | None = None):
    """Draw one violin body plus its raw values as jittered scatter, at one x position.

    A violin's estimated density shape can visually smooth over the fact that a
    small-repetition-count group (this project typically has 5) is really just a
    handful of points — overlaying the raw values keeps that honest instead of
    implying a smoother distribution than the data supports. Repeated calls on the
    same ``ax`` at different ``position``s build up a multi-group plot (same pattern
    as ``plot_metric_distribution``).
    """
    resolved = resolve_theme(theme)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    if values.size == 0:
        return figure, ax
    parts = ax.violinplot([values], positions=[position], widths=width, showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(resolved.distribution_fill_alpha)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(color)
    jittered_x = position + np.random.default_rng(0).uniform(-jitter, jitter, size=values.size)
    ax.scatter(jittered_x, values, color=color, s=point_size, alpha=resolved.primary_alpha, edgecolor=resolved.panel_color, linewidth=0.3, zorder=3, label=label)
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
