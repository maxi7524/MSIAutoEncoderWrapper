"""Model-independent metric distribution and trade-off plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .theme import VisualizationTheme, resolve_theme


def plot_metric_distribution(values: np.ndarray, metric: str, ax: Axes | None = None, bins: int = 40, kind: str = "histogram", label: str | None = None, theme: VisualizationTheme | str | None = None, color: str | None = None, histtype: str = "bar"):
    """Plot one histogram or ECDF on an optional caller-owned axis.

    Repeated calls on the same ``ax`` (different ``label``/``color``) overlay several
    groups on one plot — e.g. one series per swept parameter value. For more than ~2
    overlaid groups, pass ``histtype="step"`` (unfilled outlines) instead of the default
    filled bars — filled bars with more than a couple of overlapping series stack into
    one indistinguishable solid block regardless of alpha.
    """
    resolved = resolve_theme(theme); values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if ax is None: figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else: figure = ax.figure
    if kind == "histogram":
        fill_alpha = resolved.secondary_alpha if histtype == "bar" else resolved.primary_alpha
        ax.hist(values, bins=bins, density=(histtype != "bar"), histtype=histtype, linewidth=resolved.input_line_width, alpha=fill_alpha, color=color, label=label)
    elif kind == "ecdf":
        ordered = np.sort(values); ax.step(ordered, np.arange(1, ordered.size + 1) / max(1, ordered.size), where="post", color=color, label=label)
    else: raise ValueError("kind must be histogram or ecdf")
    ax.set(xlabel=metric, ylabel="density" if (kind == "histogram" and histtype != "bar") else ("Spectrum count" if kind == "histogram" else "ECDF")); ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    if label: ax.legend(fontsize=resolved.tick_font_size, loc=resolved.legend_location, frameon=resolved.legend_frame)
    return figure, ax


def plot_metric_tradeoff(x: Sequence[float], y: Sequence[float], x_label: str, metric: str, ax: Axes | None = None, label: str | None = None, theme: VisualizationTheme | str | None = None):
    """Plot one metric against a configurable complexity or compression axis."""
    resolved = resolve_theme(theme)
    if ax is None: figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else: figure = ax.figure
    ax.plot(x, y, marker="o", label=label); ax.set(xlabel=x_label, ylabel=metric); ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    if label: ax.legend()
    return figure, ax
