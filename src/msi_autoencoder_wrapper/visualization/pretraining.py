"""Figures of the pretraining-campaign analysis notebooks.

Every function takes canonical long-form tables written by
:mod:`msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_reports`
and returns a Matplotlib figure (or an interactive widget) styled only through
:mod:`msi_autoencoder_wrapper.visualization.theme`. Notebooks choose *what* is
compared; colours, sizes and transparencies are decided here.

Colour semantics are fixed across the notebooks:

* model groups (one spectral axis) use the catalog colour of their display label;
* pixel populations and class sets use fixed positions of the theme's class palette;
* images use the theme's image, error, residual and probability colour maps.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..analysis.autoencoder.spatial.images import assemble_image, image_extent
from .metrics import plot_violin_with_points
from .theme import VisualizationTheme, resolve_theme

#: Canonical order of pixel populations.
POPULATION_ORDER = ("train", "test", "test_extended", "test_combined", "heldout_image")
#: Canonical order of class sets.
CLASS_SET_ORDER = ("common", "boundary", "extension", "not_in_head")
#: Display names used on axes and legends.
DISPLAY_NAMES = {"train": "train", "test": "test (5 % withheld)", "test_extended": "test extension",
                 "test_combined": "test + extension", "heldout_image": "held-out images",
                 "common": "common range", "boundary": "boundary", "extension": "extension range",
                 "not_in_head": "not in head"}
#: Maximum number of individual pixels drawn on top of a violin.
OVERLAY_POINTS = 200


# --------------------------------------------------
# Section: colours and small helpers
# --------------------------------------------------

def population_color(population: str, theme: VisualizationTheme) -> str:
    """Fixed colour of one pixel population."""
    index = POPULATION_ORDER.index(population) if population in POPULATION_ORDER else len(POPULATION_ORDER)
    return theme.class_palette[index % len(theme.class_palette)]


def class_set_color(class_set: str, theme: VisualizationTheme) -> str:
    """Fixed colour of one class set."""
    index = CLASS_SET_ORDER.index(class_set) if class_set in CLASS_SET_ORDER else len(CLASS_SET_ORDER)
    return theme.class_palette[(index + 5) % len(theme.class_palette)]


def model_color(label: str, theme: VisualizationTheme, position: int = 0) -> str:
    """Catalog colour of one model group (display label)."""
    return theme.color_for_model(label, position)


def _grid(count: int, columns: int, theme: VisualizationTheme, width: float = 4.2, height: float = 3.4):
    """Figure with ``count`` panels in rows of ``columns``; unused panels are hidden."""
    columns = max(1, min(columns, count))
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), dpi=theme.figure_dpi,
                                squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return figure, axes.flat


def _title(ax, text: str, theme: VisualizationTheme) -> None:
    ax.set_title(text, loc=theme.title_location)


def _overlay(values: np.ndarray) -> np.ndarray:
    """Evenly spaced ordered subset drawn over a violin (the violin uses every value)."""
    values = np.sort(values[np.isfinite(values)])
    if values.size <= OVERLAY_POINTS:
        return values
    return values[np.linspace(0, values.size - 1, OVERLAY_POINTS, dtype=int)]


# --------------------------------------------------
# Section: distributions
# --------------------------------------------------

def population_distributions(pixels: pd.DataFrame, means: pd.DataFrame, metrics: Sequence[str],
                             populations: Sequence[str], *, titles: Optional[dict] = None,
                             columns: int = 3, theme: Any = None):
    """Pixel-level distributions per population with repetition means on top.

    :param pixels: One row per sampled pixel with ``population`` and metric columns.
    :param means: One row per model x population x metric with ``mean``.
    :param metrics: Metric columns, one panel each.
    :param populations: Populations in display order.
    :param titles: Optional panel title per metric.
    :param columns: Panels per row.
    :param theme: Visualization theme.
    :return: Figure. Violins: every sampled pixel pooled over repetitions; small points:
        at most 200 evenly spaced pixels; black diamonds: one mean per repetition.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(metrics), columns, resolved)
    for ax, metric in zip(axes, metrics):
        for position, population in enumerate(populations):
            values = pixels.loc[pixels.population == population, metric].to_numpy(float)
            if values.size == 0:
                continue
            plot_violin_with_points(values, position, ax=ax, color=population_color(population, resolved),
                                    overlay_values=_overlay(values), point_size=6, theme=resolved)
            repetition_means = means[(means.population == population) & (means.metric == metric)]["mean"]
            ax.scatter(np.full(len(repetition_means), position + 0.25), repetition_means, marker="D", s=22,
                       color=resolved.baseline_color, zorder=5)
        ax.set_xticks(range(len(populations)), [DISPLAY_NAMES.get(value, value) for value in populations],
                      rotation=20, ha="right")
        _title(ax, (titles or {}).get(metric, metric), resolved)
    figure.legend(handles=[Line2D([], [], marker="D", color=resolved.baseline_color, linestyle="",
                                  label="repetition mean")], loc="upper right", frameon=False)
    figure.tight_layout()
    return figure


def repetition_points(frame: pd.DataFrame, *, value: str, group: str, hue: str, order: Sequence[str],
                      hue_order: Sequence[str], colors: dict, ylabel: str, title: str = "",
                      reference: Optional[float] = None, ax=None, theme: Any = None):
    """One point per repetition, grouped on the x-axis and dodged by ``hue``.

    :return: Figure; horizontal bars mark the mean over repetitions of each group/hue.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(max(6, 1.3 * len(order) * max(1, len(hue_order))), 4.2),
                                  dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    width = 0.7 / max(1, len(hue_order))
    for position, name in enumerate(order):
        for offset, level in enumerate(hue_order):
            values = frame.loc[(frame[group] == name) & (frame[hue] == level), value].to_numpy(float)
            x = position - 0.35 + width * (offset + 0.5)
            jitter = np.random.default_rng(offset).uniform(-width / 5, width / 5, size=values.size)
            ax.scatter(x + jitter, values, color=colors[level], s=28, alpha=resolved.marker_alpha,
                       edgecolor=resolved.panel_color, linewidth=0.4, zorder=3,
                       label=DISPLAY_NAMES.get(level, level) if position == 0 else None)
            if values.size:
                ax.hlines(np.nanmean(values), x - width / 2.5, x + width / 2.5, color=colors[level], linewidth=2)
    if reference is not None:
        ax.axhline(reference, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                   linewidth=resolved.reference_line_width)
    ax.set_xticks(range(len(order)), [DISPLAY_NAMES.get(value, value) for value in order], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    _title(ax, title, resolved)
    ax.legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


def quantile_curves(quantiles: pd.DataFrame, *, metric: str, populations: Sequence[str], log: bool = False,
                    theme: Any = None):
    """Empirical quantile functions per population, one line per repetition.

    :return: Figure: x = quantile level, y = metric value.
    """
    resolved = resolve_theme(theme)
    figure, ax = plt.subplots(figsize=(7.5, 4.4), dpi=resolved.figure_dpi)
    frame = quantiles[quantiles.metric == metric]
    for population in populations:
        for _, lines in frame[frame.population == population].groupby("repetition"):
            ax.plot(lines["quantile"], lines["value"], color=population_color(population, resolved),
                    alpha=resolved.secondary_alpha, linewidth=resolved.line_width)
    ax.legend(handles=[Line2D([], [], color=population_color(value, resolved), label=DISPLAY_NAMES.get(value, value))
                       for value in populations], frameon=False)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("quantile level")
    ax.set_ylabel(metric)
    _title(ax, f"Quantile function of {metric}", resolved)
    figure.tight_layout()
    return figure


# --------------------------------------------------
# Section: training histories
# --------------------------------------------------

def loss_trajectories(history: pd.DataFrame, *, components: Sequence[str], labels: Sequence[str],
                      columns: int = 3, theme: Any = None):
    """Epoch trajectories per loss component: train solid, validation dashed, test marker.

    :param history: Long-form history with ``display_label``, ``repetition``, ``epoch``,
        ``split``, ``component`` and ``value``.
    :param components: Components, one panel each.
    :param labels: Model groups shown (display labels); one line per repetition.
    :return: Figure.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(components), columns, resolved)
    styles = {"train": "-", "validation": "--"}
    for ax, component in zip(axes, components):
        frame = history[(history.component == component) & history.display_label.isin(labels)]
        for position, label in enumerate(labels):
            color = model_color(label, resolved, position)
            for (split, _), lines in frame[frame.display_label == label].groupby(["split", "repetition"]):
                lines = lines.sort_values("epoch")
                if split == "test":
                    ax.scatter(lines.epoch, lines.value, marker="*", s=60, color=color, zorder=4)
                else:
                    ax.plot(lines.epoch, lines.value, linestyle=styles[split], color=color,
                            alpha=resolved.secondary_alpha, linewidth=resolved.line_width)
        ax.set_xlabel("epoch")
        _title(ax, component, resolved)
    handles = [Line2D([], [], color=model_color(label, resolved, position), label=label)
               for position, label in enumerate(labels)]
    handles += [Line2D([], [], color=resolved.baseline_color, linestyle="-", label="train"),
                Line2D([], [], color=resolved.baseline_color, linestyle="--", label="validation (sampled)"),
                Line2D([], [], color=resolved.baseline_color, marker="*", linestyle="", label="test (sampled)")]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                  bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def group_bands(trajectories: pd.DataFrame, *, components: Sequence[str], labels: Sequence[str], split: str,
                columns: int = 3, theme: Any = None):
    """Mean trajectory and min-max band across repetitions per model group.

    :return: Figure; the band spans the minimum and maximum repetition at each epoch.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(components), columns, resolved)
    for ax, component in zip(axes, components):
        frame = trajectories[(trajectories.component == component) & (trajectories.split == split)]
        for position, label in enumerate(labels):
            lines = frame[frame.display_label == label].sort_values("epoch")
            color = model_color(label, resolved, position)
            ax.plot(lines.epoch, lines["mean"], color=color, linewidth=resolved.line_width, label=label)
            ax.fill_between(lines.epoch, lines["min"], lines["max"], color=color, alpha=resolved.uncertainty_alpha)
        ax.set_xlabel("epoch")
        _title(ax, f"{component} ({split})", resolved)
    axes[0].legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


# --------------------------------------------------
# Section: m/z windows and spectra
# --------------------------------------------------

def window_profiles(summary: pd.DataFrame, *, quantity: str, populations: Sequence[str], statistic: str = "mean",
                    ylabel: Optional[str] = None, log: bool = False, ax=None, theme: Any = None):
    """Window statistic along the m/z axis per population, one line per repetition.

    :param summary: ``window_summary`` rows (model x population x window x quantity).
    :return: Figure: x = window centre (m/z), y = ``statistic`` of ``quantity``.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(10, 4.2), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    frame = summary[summary.quantity == quantity]
    for population in populations:
        color = population_color(population, resolved)
        for _, lines in frame[frame.population == population].groupby("repetition"):
            lines = lines.sort_values("window_lower")
            centres = (lines.window_lower + lines.window_upper) / 2
            ax.plot(centres, lines[statistic], marker="o", markersize=3, color=color,
                    alpha=resolved.secondary_alpha, linewidth=resolved.line_width)
    ax.legend(handles=[Line2D([], [], color=population_color(value, resolved), label=DISPLAY_NAMES.get(value, value))
                       for value in populations], frameon=False)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("m/z window centre")
    ax.set_ylabel(ylabel or f"{statistic} {quantity}")
    _title(ax, f"{quantity} per 100 Da window", resolved)
    figure.tight_layout()
    return figure


def matrix_heatmap(matrix: pd.DataFrame, *, title: str, colorbar_label: str, cmap: Optional[str] = None,
                   diverging: bool = False, annotate: bool = False, theme: Any = None):
    """Heat map of a pre-aggregated matrix (rows x columns).

    :return: Figure; ``diverging`` centres the colour scale at zero.
    """
    resolved = resolve_theme(theme)
    values = matrix.to_numpy(float)
    figure, ax = plt.subplots(figsize=(max(6, 0.55 * matrix.shape[1] + 3), max(2.5, 0.45 * matrix.shape[0] + 1.5)),
                              dpi=resolved.figure_dpi)
    if diverging:
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        image = ax.imshow(values, aspect="auto", cmap=cmap or resolved.residual_colormap, vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap or resolved.error_colormap)
    ax.set_xticks(range(matrix.shape[1]), [str(value) for value in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]), [str(value) for value in matrix.index])
    ax.grid(False)
    if annotate:
        for (row, column), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2g}", ha="center", va="center", fontsize=7,
                        color=resolved.panel_color if not diverging else resolved.text_color)
    figure.colorbar(image, ax=ax, label=colorbar_label)
    _title(ax, title, resolved)
    figure.tight_layout()
    return figure


def feature_profile(features: pd.DataFrame, *, groups: Sequence[str], title: str, theme: Any = None):
    """Mean input, mean reconstruction and mean absolute error along m/z.

    :param features: ``feature_errors`` rows of one population (repetition means are
        taken here) with ``group`` (``all`` or an image) and per-bin means.
    :return: Figure: top = mean input (grey) and output (colour), bottom = mean
        absolute error, one row of panels per group.
    """
    resolved = resolve_theme(theme)
    figure, axes = plt.subplots(2 * len(groups), 1, figsize=(11, 3.2 * len(groups)), dpi=resolved.figure_dpi,
                                sharex=True, squeeze=False)
    for position, group in enumerate(groups):
        frame = features[features.group == group].groupby(["bin", "mz"], as_index=False)[
            ["mean_input", "mean_output", "mean_abs_error", "mean_signed_error"]].mean()
        top, bottom = axes[2 * position, 0], axes[2 * position + 1, 0]
        top.plot(frame.mz, frame.mean_input, color=resolved.input_color, linewidth=resolved.input_line_width,
                 label="mean input")
        top.plot(frame.mz, frame.mean_output, color=resolved.model_palette[0], alpha=resolved.secondary_alpha,
                 linewidth=resolved.reconstruction_line_width, label="mean reconstruction")
        top.set_ylabel("TIC fraction")
        _title(top, f"{title}: {group}", resolved)
        top.legend(frameon=False, fontsize=resolved.legend_font_size)
        bottom.plot(frame.mz, frame.mean_signed_error, color=resolved.residual_color,
                    linewidth=resolved.residual_line_width, label="mean signed error (output - input)")
        bottom.axhline(0, color=resolved.baseline_color, linewidth=resolved.reference_line_width)
        bottom.set_ylabel("signed error")
        bottom.legend(frameon=False, fontsize=resolved.legend_font_size)
    axes[-1, 0].set_xlabel("m/z")
    figure.tight_layout()
    return figure


def spectrum_pairs(cases: pd.DataFrame, *, keys: Sequence[tuple], columns: int = 2, theme: Any = None,
                   window_edges: Optional[np.ndarray] = None):
    """Input versus reconstruction for selected spectra.

    :param cases: ``cases`` rows (one per bin) with ``population``, ``row``, ``input``, ``output``.
    :param keys: ``(population, row)`` pairs, one panel each.
    :param window_edges: Optional m/z window edges drawn as faint vertical lines.
    :return: Figure: input in grey, reconstruction in colour, residual below zero line.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(keys), columns, resolved, width=6.5, height=3.2)
    for ax, (population, row) in zip(axes, keys):
        frame = cases[(cases.population == population) & (cases.row == row)].sort_values("mz")
        ax.plot(frame.mz, frame.input, color=resolved.input_color, linewidth=resolved.input_line_width, label="input")
        ax.plot(frame.mz, frame.output, color=population_color(population, resolved), alpha=resolved.primary_alpha,
                linewidth=resolved.reconstruction_line_width, label="reconstruction")
        if window_edges is not None:
            for edge in window_edges:
                ax.axvline(edge, color=resolved.grid_color, linewidth=0.6, zorder=0)
        first = frame.iloc[0] if len(frame) else None
        detail = f"{DISPLAY_NAMES.get(population, population)}, row {row}"
        if first is not None:
            detail += f", {first.case_kind}, W1={first.masserstein:.2f}"
        _title(ax, detail, resolved)
        ax.set_xlabel("m/z")
    axes[0].legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


def spectrum_zoom(cases: pd.DataFrame, *, keys: Sequence[tuple], half_width: float,
                  worst_windows: Optional[dict] = None, theme: Any = None):
    """Input versus reconstruction of selected spectra, in full and in two zooms, with residuals.

    Columns: full m/z range; +/- ``half_width`` around the strongest input peak (peak
    shape and position); the window with the largest contribution to $W_1$ when given
    (where the cost is paid). Every column has the signal on top and the signed residual
    (reconstruction - input) below; the zoom limits are set from the values inside the
    zoom, so small peaks are not flattened by the base peak.

    :param cases: ``cases`` rows (one per bin) with ``population``, ``row``, ``mz``, ``input``,
        ``output``, ``case_kind`` and ``masserstein``.
    :param keys: ``(population, row)`` pairs, one pair of rows each.
    :param half_width: Half width (m/z) of the peak zoom.
    :param worst_windows: Optional ``{(population, row): (label, lower, upper)}`` of the
        largest-contribution window.
    :return: Figure.
    """
    resolved = resolve_theme(theme)
    panels = 3 if worst_windows else 2
    figure = plt.figure(figsize=(5.4 * panels, 3.9 * len(keys)), dpi=resolved.figure_dpi)
    ## One outer cell per case and range; signal and residual share the cell with a small gap
    outer = figure.add_gridspec(len(keys), panels, hspace=0.55, wspace=0.22)
    axes = np.empty((2 * len(keys), panels), dtype=object)
    for index in range(len(keys)):
        for column in range(panels):
            inner = outer[index, column].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.06)
            axes[2 * index, column] = figure.add_subplot(inner[0])
            axes[2 * index + 1, column] = figure.add_subplot(inner[1], sharex=axes[2 * index, column])
            axes[2 * index, column].tick_params(labelbottom=False)
    for index, (population, row) in enumerate(keys):
        frame = cases[(cases.population == population) & (cases.row == row)].sort_values("mz")
        mz, observed, produced = frame.mz.to_numpy(), frame.input.to_numpy(), frame.output.to_numpy()
        residual = produced - observed
        apex = float(mz[int(np.argmax(observed))]) if len(frame) else 0.0
        ranges = [(float(mz.min()), float(mz.max()), "full range"),
                  (apex - half_width, apex + half_width, f"strongest peak {apex:.1f} +/- {half_width:g}")]
        if worst_windows and (population, row) in worst_windows:
            label, lower, upper = worst_windows[(population, row)]
            ranges.append((float(lower), float(upper), f"largest-contribution window {label}"))
        for column, (lower, upper, caption) in enumerate(ranges):
            signal_ax, residual_ax = axes[2 * index, column], axes[2 * index + 1, column]
            inside = (mz >= lower) & (mz <= upper)
            signal_ax.plot(mz, observed, color=resolved.input_color, linewidth=resolved.input_line_width,
                           label="input")
            signal_ax.plot(mz, produced, color=population_color(population, resolved), alpha=resolved.primary_alpha,
                           linewidth=resolved.reconstruction_line_width, label="reconstruction")
            residual_ax.axhline(0.0, color=resolved.grid_color, linewidth=0.8)
            residual_ax.plot(mz, residual, color=population_color(population, resolved), linewidth=0.8)
            for ax in (signal_ax, residual_ax):
                ax.set_xlim(lower, upper)
            if inside.any():
                top = float(max(observed[inside].max(), produced[inside].max()))
                signal_ax.set_ylim(min(0.0, float(produced[inside].min())), 1.15 * top if top > 0 else 1.0)
                spread = float(np.abs(residual[inside]).max())
                residual_ax.set_ylim(-1.15 * spread if spread > 0 else -1.0, 1.15 * spread if spread > 0 else 1.0)
            first = frame.iloc[0] if len(frame) else None
            detail = f"{DISPLAY_NAMES.get(population, population)}, row {row}"
            if first is not None:
                detail += f", {first.case_kind}, W1={first.masserstein:.2f}"
            _title(signal_ax, f"{detail}\n{caption}", resolved)
            residual_ax.set_xlabel("m/z")
            if column == 0:
                signal_ax.set_ylabel("TIC fraction")
                residual_ax.set_ylabel("output - input")
    axes[0, 0].legend(frameon=False, fontsize=resolved.legend_font_size)
    return figure


# --------------------------------------------------
# Section: images
# --------------------------------------------------

def _color_range(values: np.ndarray, *, percentile: float, diverging: bool, positive: bool) -> tuple[float, float]:
    """Robust colour range of finite values.

    ``positive`` computes the upper percentile on the strictly positive values only, so
    an image that is zero almost everywhere (a sparse ion) keeps a usable range; a
    degenerate range is widened so every non-zero pixel stays visible.
    """
    values = values[np.isfinite(values)]
    if diverging:
        limit = float(np.percentile(np.abs(values), percentile)) if values.size else 1.0
        limit = limit if limit > 0 else float(np.abs(values).max(initial=0.0)) or 1.0
        return -limit, limit
    if positive:
        support = values[values > 0]
        low, high = 0.0, float(np.percentile(support, percentile)) if support.size else 1.0
    else:
        low, high = ((float(np.percentile(values, 100 - percentile)), float(np.percentile(values, percentile)))
                     if values.size else (0.0, 1.0))
    if high <= low:
        high = float(values.max(initial=low)) if values.size and values.max() > low else low + 1.0
    return low, high


def image_panels(frame: pd.DataFrame, *, value_columns: Sequence[str], column_titles: Sequence[str],
                 datasets: Sequence[str], cmap: Optional[str] = None, shared_scale: str = "column",
                 percentile: float = 99.0, diverging: bool = False, row_column: str = "dataset_id",
                 diverging_columns: Sequence[str] = (), positive_columns: Sequence[str] = (),
                 column_cmaps: Optional[dict] = None, theme: Any = None):
    """Grid of image maps: one row per ``row_column`` value, one column per value column.

    :param frame: Pixel rows with ``row_column``, ``x``, ``y`` and the value columns.
    :param datasets: Values of ``row_column`` shown as rows (e.g. images, or image-ion pairs).
    :param shared_scale: ``column`` shares the colour range of a column over all rows,
        ``panel`` scales every panel on its own, ``row`` shares one range over the
        non-diverging panels of a row (e.g. input and reconstruction of one channel).
    :param percentile: Upper percentile of the colour range (robust to single hot pixels).
    :param diverging: Symmetric range around zero with the residual colour map for every column.
    :param diverging_columns: Columns drawn diverging (e.g. an error image) while the others are not.
    :param positive_columns: Columns whose upper limit is taken over their positive pixels only.
    :param column_cmaps: Optional colour map per column; otherwise diverging columns use the
        residual colour map and the others ``cmap`` (default: the error colour map).
    :return: Figure; background (no pixel) is left blank.
    """
    resolved = resolve_theme(theme)
    signed = set(value_columns) if diverging else set(diverging_columns)
    positive = set(positive_columns)

    def colormap(column: str) -> str:
        if column_cmaps and column in column_cmaps:
            return column_cmaps[column]
        if column in signed:
            return resolved.residual_colormap
        return cmap or resolved.error_colormap

    def limits(values: np.ndarray, column: str) -> tuple[float, float]:
        return _color_range(values, percentile=percentile, diverging=column in signed, positive=column in positive)

    column_limits = {column: limits(frame[column].to_numpy(float), column) for column in value_columns}
    figure, axes = plt.subplots(len(datasets), len(value_columns), dpi=resolved.figure_dpi, squeeze=False,
                                figsize=(3.3 * len(value_columns), 4.2 * len(datasets)))
    for row, dataset in enumerate(datasets):
        subset = frame[frame[row_column] == dataset]
        extent = image_extent(subset.x.to_numpy(), subset.y.to_numpy())
        unsigned = [column for column in value_columns if column not in signed]
        row_limits = limits(subset[unsigned].to_numpy(float).ravel(), unsigned[0]) if unsigned else None
        for column, (name, title) in enumerate(zip(value_columns, column_titles)):
            ax = axes[row, column]
            image = assemble_image(subset.x.to_numpy(), subset.y.to_numpy(), subset[name].to_numpy(float), extent)
            if shared_scale == "column":
                low, high = column_limits[name]
            elif shared_scale == "row" and name not in signed:
                low, high = row_limits
            else:
                low, high = limits(image.ravel(), name)
            shown = ax.imshow(image, origin="upper", interpolation=resolved.image_interpolation, cmap=colormap(name),
                              vmin=low, vmax=high)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(title, fontsize=resolved.label_font_size)
            if column == 0:
                ax.set_ylabel(dataset, fontsize=resolved.tick_font_size)
            figure.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
    figure.tight_layout()
    return figure


def pixel_spectra(pixels: pd.DataFrame, *, datasets: Sequence[str], kinds: Sequence[str] = ("best", "median", "worst"),
                  theme: Any = None):
    """Input and reconstruction of deterministic held-out pixels (displayed bins only).

    :param pixels: ``browser_pixels`` rows (``dataset_id``, ``pixel_kind``, ``x``, ``y``,
        ``mz``, ``input``, ``output``).
    :return: Figure: one row per image, one column per pixel kind.
    """
    resolved = resolve_theme(theme)
    figure, axes = plt.subplots(len(datasets), len(kinds), figsize=(5.2 * len(kinds), 2.8 * len(datasets)),
                                dpi=resolved.figure_dpi, squeeze=False)
    for row, dataset in enumerate(datasets):
        for column, kind in enumerate(kinds):
            frame = pixels[(pixels.dataset_id == dataset) & (pixels.pixel_kind == kind)].sort_values("mz")
            ax = axes[row, column]
            ## Displayed bins are not contiguous; gaps are drawn as breaks, not interpolated
            gaps = np.r_[False, np.diff(frame.mz.to_numpy()) > 1.0]
            mz = np.where(gaps, np.nan, frame.mz.to_numpy())
            ax.plot(mz, frame.input, color=resolved.input_color, linewidth=resolved.input_line_width, label="input")
            ax.plot(mz, frame.output, color=resolved.model_palette[0], alpha=resolved.primary_alpha,
                    linewidth=resolved.reconstruction_line_width, label="reconstruction")
            if len(frame):
                ax.set_title(f"{dataset} {kind} ({int(frame.x.iloc[0])}, {int(frame.y.iloc[0])})",
                             fontsize=resolved.label_font_size, loc=resolved.title_location)
    axes[0, 0].legend(frameon=False, fontsize=resolved.legend_font_size)
    for ax in axes[-1]:
        ax.set_xlabel("m/z")
    figure.tight_layout()
    return figure


def label_panels(frame: pd.DataFrame, *, label_columns: Sequence[str], column_titles: Sequence[str],
                 datasets: Sequence[str], k: int, theme: Any = None):
    """Grid of categorical segment maps: one row per dataset, one column per labelling.

    :return: Figure with a shared qualitative colour per segment index.
    """
    resolved = resolve_theme(theme)
    palette = plt.get_cmap("tab20" if k > 10 else "tab10")
    colormap = ListedColormap([palette(index) for index in range(k)])
    figure, axes = plt.subplots(len(datasets), len(label_columns), dpi=resolved.figure_dpi, squeeze=False,
                                figsize=(3.3 * len(label_columns), 4.2 * len(datasets)))
    for row, dataset in enumerate(datasets):
        subset = frame[frame.dataset_id == dataset]
        extent = image_extent(subset.x.to_numpy(), subset.y.to_numpy())
        for column, (name, title) in enumerate(zip(label_columns, column_titles)):
            ax = axes[row, column]
            image = assemble_image(subset.x.to_numpy(), subset.y.to_numpy(), subset[name].to_numpy(float), extent)
            ax.imshow(image, origin="upper", interpolation="nearest", cmap=colormap, vmin=-0.5, vmax=k - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(title, fontsize=resolved.label_font_size)
            if column == 0:
                ax.set_ylabel(dataset, fontsize=resolved.tick_font_size)
    figure.legend(handles=[Patch(color=colormap(index), label=f"segment {index}") for index in range(k)],
                  loc="lower center", ncol=min(k, 8), frameon=False)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def rgb_panels(frame: pd.DataFrame, *, repetitions: Sequence, datasets: Sequence[str], theme: Any = None,
               column: str = "repetition", column_titles: Optional[Sequence[str]] = None):
    """RGB composites of the first three principal components of the latent codes.

    :param repetitions: Values of ``column`` shown as columns (repetitions by default).
    :param column: Column selecting the model of each panel column (e.g. ``shown_as``).
    :param column_titles: Optional titles of the panel columns.
    :return: Figure: one row per dataset, one column per value of ``column``.
    """
    resolved = resolve_theme(theme)
    figure, axes = plt.subplots(len(datasets), len(repetitions), dpi=resolved.figure_dpi, squeeze=False,
                                figsize=(3.3 * len(repetitions), 4.2 * len(datasets)))
    titles = list(column_titles) if column_titles is not None else [
        f"repetition {value}" if column == "repetition" else str(value) for value in repetitions]
    for row, dataset in enumerate(datasets):
        for position, repetition in enumerate(repetitions):
            subset = frame[(frame.dataset_id == dataset) & (frame[column] == repetition)]
            extent = image_extent(subset.x.to_numpy(), subset.y.to_numpy())
            channels = [assemble_image(subset.x.to_numpy(), subset.y.to_numpy(), subset[name].to_numpy(float), extent)
                        for name in ("r", "g", "b")]
            image = np.dstack(channels)
            image = np.where(np.isfinite(image), image, 1.0)
            ax = axes[row, position]
            ax.imshow(image, origin="upper", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(titles[position], fontsize=resolved.label_font_size)
            if position == 0:
                ax.set_ylabel(dataset, fontsize=resolved.tick_font_size)
    figure.tight_layout()
    return figure


def exploded_view(layers: pd.DataFrame, *, dataset: str, classes: Sequence[str], value: str, k: int,
                  theme: Any = None, title: str = ""):
    """Exploded image x m/z view: segmentation at the bottom, one ion layer per class above.

    Every layer is the image plane; layers are stacked at equal spacing in the order of
    their class m/z (the z tick labels give class and m/z), so classes with close m/z do
    not overlap. The bottom plane shows the latent segmentation, the upper planes show
    ``value`` (for example the METASPACE intensity or the head probability) normalized
    to [0, 1] per layer by its 99th percentile.

    :param layers: ``exploded_layers`` rows with ``class_name``, ``mz``, ``x``, ``y``,
        ``segment`` and ``value``.
    :return: Figure with a three-dimensional axis (x, y in pixels, z = layer ordered by m/z).
    """
    resolved = resolve_theme(theme)
    frame = layers[(layers.dataset_id == dataset) & layers.class_name.isin(classes)]
    base = frame.drop_duplicates("row")
    extent = image_extent(base.x.to_numpy(), base.y.to_numpy())
    grid_y, grid_x = np.mgrid[0:extent[3], 0:extent[2]]
    figure = plt.figure(figsize=(9, 8), dpi=resolved.figure_dpi)
    ax = figure.add_subplot(projection="3d")
    ordered = frame.drop_duplicates("class_name").sort_values("mz").reset_index(drop=True)
    ## Segmentation plane at height 0
    segments = assemble_image(base.x.to_numpy(), base.y.to_numpy(), base.segment.to_numpy(float), extent)
    palette = plt.get_cmap("tab20" if k > 10 else "tab10")
    colors = np.zeros(segments.shape + (4,))
    finite = np.isfinite(segments)
    colors[finite] = [palette(int(value)) for value in segments[finite]]
    ax.plot_surface(grid_x, grid_y, np.zeros(segments.shape), facecolors=colors, rstride=1, cstride=1,
                    shade=False, linewidth=0)
    ## Ion planes at heights 1..n in m/z order
    colormap = plt.get_cmap(resolved.probability_colormap)
    for height, record in enumerate(ordered.itertuples(), start=1):
        subset = frame[frame.class_name == record.class_name]
        image = assemble_image(subset.x.to_numpy(), subset.y.to_numpy(), subset[value].to_numpy(float), extent)
        finite = np.isfinite(image)
        high = np.percentile(image[finite], 99) if finite.any() else 1.0
        scaled = np.clip(np.where(finite, image, 0) / (high if high > 0 else 1.0), 0, 1)
        face = colormap(scaled)
        face[..., 3] = np.where(finite, 0.35 + 0.65 * scaled, 0.0)
        ax.plot_surface(grid_x, grid_y, np.full(image.shape, float(height)), facecolors=face, rstride=1, cstride=1,
                        shade=False, linewidth=0)
    ax.set_zticks(range(len(ordered) + 1),
                  ["segments", *[f"{name} ({mz:.1f})" for name, mz in zip(ordered.class_name, ordered.mz)]],
                  fontsize=7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.view_init(elev=18, azim=-60)
    ax.set_title(title or f"{dataset}: segments and {value}", fontsize=resolved.title_font_size)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colormap), ax=ax, shrink=0.5, pad=0.15,
                    label=f"{value} (per-layer 99th percentile = 1)")
    return figure


def pixel_browser(pixel_maps: pd.DataFrame, *, display_input: np.ndarray, display_output: np.ndarray,
                  bins: np.ndarray, mass_axis: np.ndarray, map_column: str = "masserstein_mean",
                  window_columns: Sequence[str] = (), theme: Any = None):
    """Interactive browser: pick a held-out image and a pixel, see its spectrum and window costs.

    :param pixel_maps: ``pixel_maps`` rows (one per held-out pixel, array-row order).
    :param display_input: Stored input spectra of the displayed bins, ``(N_h, K)``.
    :param display_output: Stored reconstructions of the displayed bins, ``(N_h, K)``.
    :param bins: Displayed bin indices, ``(K,)``.
    :param mass_axis: Bin centres of the axis, ``(M,)``.
    :param window_columns: ``contribution__<window>`` columns shown as a bar chart.
    :return: ``ipywidgets`` container. Interactive output is not stored in the saved notebook.
    """
    import ipywidgets as widgets

    resolved = resolve_theme(theme)
    datasets = sorted(pixel_maps.dataset_id.unique())
    dataset = widgets.Dropdown(options=datasets, description="image")
    x = widgets.IntSlider(description="x")
    y = widgets.IntSlider(description="y")
    output = widgets.Output()

    def update_ranges(*_):
        subset = pixel_maps[pixel_maps.dataset_id == dataset.value]
        x.min, x.max = int(subset.x.min()), int(subset.x.max())
        y.min, y.max = int(subset.y.min()), int(subset.y.max())
        x.value, y.value = int(subset.x.median()), int(subset.y.median())

    def draw(*_):
        subset = pixel_maps[pixel_maps.dataset_id == dataset.value]
        match = subset[(subset.x == x.value) & (subset.y == y.value)]
        output.clear_output(wait=True)
        with output:
            figure, axes = plt.subplots(1, 3, figsize=(15, 3.6), dpi=resolved.figure_dpi,
                                        gridspec_kw={"width_ratios": [1, 2.4, 1.4]})
            extent = image_extent(subset.x.to_numpy(), subset.y.to_numpy())
            image = assemble_image(subset.x.to_numpy(), subset.y.to_numpy(), subset[map_column].to_numpy(float), extent)
            axes[0].imshow(image, origin="upper", cmap=resolved.error_colormap, interpolation="nearest")
            axes[0].scatter([x.value - extent[0]], [y.value - extent[1]], marker="+", s=80, color="cyan")
            axes[0].set_title(map_column, fontsize=resolved.label_font_size)
            axes[0].grid(False)
            if match.empty:
                axes[1].text(0.5, 0.5, "no pixel at this position", ha="center")
            else:
                row = int(match.row.iloc[0])
                axes[1].plot(mass_axis[bins], np.asarray(display_input[row], float), color=resolved.input_color,
                             linewidth=resolved.input_line_width, label="input")
                axes[1].plot(mass_axis[bins], np.asarray(display_output[row], float), color=resolved.model_palette[0],
                             linewidth=resolved.reconstruction_line_width, alpha=resolved.primary_alpha,
                             label="reconstruction")
                axes[1].legend(frameon=False)
                axes[1].set_title(f"pixel ({x.value}, {y.value}), {map_column}={float(match[map_column].iloc[0]):.3f}",
                                  fontsize=resolved.label_font_size)
                if window_columns:
                    values = match[list(window_columns)].iloc[0].to_numpy(float)
                    axes[2].bar(range(len(values)), values, color=resolved.residual_color)
                    axes[2].set_xticks(range(len(values)), [name.split("__")[-1] for name in window_columns],
                                       rotation=60, fontsize=7)
                    axes[2].set_title("window contribution", fontsize=resolved.label_font_size)
            axes[1].set_xlabel("m/z")
            figure.tight_layout()
            plt.show()

    dataset.observe(lambda change: (update_ranges(), draw()), names="value")
    x.observe(draw, names="value")
    y.observe(draw, names="value")
    update_ranges()
    draw()
    return widgets.VBox([widgets.HBox([dataset, x, y]), output])


# --------------------------------------------------
# Section: relations and comparisons
# --------------------------------------------------

def paired_scatter(frame: pd.DataFrame, *, x: str, y: str, hue: str, hue_order: Sequence[str], xlabel: str,
                   ylabel: str, title: str, diagonal: bool = True, log: bool = False, ax=None, theme: Any = None,
                   colors: Optional[dict] = None):
    """Scatter of two paired quantities coloured by a categorical variable.

    :return: Figure; the diagonal ``y = x`` marks equality of the paired values.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.2, 5.6), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for position, level in enumerate(hue_order):
        subset = frame[frame[hue] == level]
        color = (colors or {}).get(level, class_set_color(level, resolved))
        ax.scatter(subset[x], subset[y], s=14, alpha=resolved.marker_alpha, color=color,
                   edgecolor=resolved.panel_color, linewidth=0.3, label=f"{DISPLAY_NAMES.get(level, level)} (n={len(subset)})")
    if diagonal:
        values = frame[[x, y]].to_numpy(float)
        finite = values[np.isfinite(values).all(axis=1)]
        if finite.size:
            low, high = finite.min(), finite.max()
            ax.plot([low, high], [low, high], color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                    linewidth=resolved.reference_line_width)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _title(ax, title, resolved)
    ax.legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


def line_by_repetition(frame: pd.DataFrame, *, x: str, y: str, color_by: str, levels: Sequence[str],
                       colors: dict, xlabel: str, ylabel: str, title: str, log_x: bool = False,
                       log_y: bool = False, reference: Optional[pd.DataFrame] = None, ax=None, theme: Any = None):
    """One line per repetition and level (e.g. sensitivity curves, eigenvalue spectra).

    :param reference: Optional ``x``/``y`` frame drawn as a dashed reference curve.
    :return: Figure.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.5, 4.4), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for level in levels:
        for _, lines in frame[frame[color_by] == level].groupby("repetition"):
            lines = lines.sort_values(x)
            ax.plot(lines[x], lines[y], marker="o", markersize=3, color=colors[level], alpha=resolved.secondary_alpha,
                    linewidth=resolved.line_width)
    if reference is not None:
        ax.plot(reference[x], reference[y], color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                linewidth=resolved.reference_line_width, label="reference")
    ax.legend(handles=[Line2D([], [], color=colors[level], label=DISPLAY_NAMES.get(level, level)) for level in levels],
              frameon=False)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _title(ax, title, resolved)
    figure.tight_layout()
    return figure


def cos_theta_histograms(samples: pd.DataFrame, *, populations: Sequence[str], uniform_sd: float,
                         theme: Any = None):
    """Distribution of pairwise latent cos(theta) against the uniform-sphere null.

    :param samples: ``angular_samples`` rows (``population``, ``repetition``, ``cos_theta``).
    :param uniform_sd: Standard deviation of cos(theta) for uniform points on the sphere.
    :return: Figure: one step histogram per repetition, dashed normal curve with the
        uniform null's mean 0 and standard deviation ``uniform_sd``.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(populations), len(populations), resolved, width=4.6)
    grid = np.linspace(-1, 1, 201)
    null = np.exp(-0.5 * (grid / uniform_sd) ** 2) / (uniform_sd * np.sqrt(2 * np.pi))
    for ax, population in zip(axes, populations):
        for _, frame in samples[samples.population == population].groupby("repetition"):
            ax.hist(frame.cos_theta, bins=60, range=(-1, 1), density=True, histtype="step",
                    color=population_color(population, resolved), alpha=resolved.secondary_alpha)
        ax.plot(grid, null, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                linewidth=resolved.reference_line_width, label="uniform-sphere null")
        ax.set_xlabel("cos(theta) between random pixel pairs")
        _title(ax, DISPLAY_NAMES.get(population, population), resolved)
    axes[0].legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


# --------------------------------------------------
# Section: pretraining variants against the baseline
# --------------------------------------------------

#: Label of the zero line of every improvement figure.
IMPROVEMENT_LABEL = "improvement over the baseline (positive = better)"


def _select(frame: pd.DataFrame, selector: dict) -> pd.DataFrame:
    """Rows whose columns equal the selector values (a list or tuple value selects any of them)."""
    mask = np.ones(len(frame), dtype=bool)
    for column, value in selector.items():
        values = frame[column].to_numpy()
        mask &= np.isin(values, list(value)) if isinstance(value, (list, tuple)) else values == value
    return frame[mask]


def improvement_forest(summary: pd.DataFrame, *, panels: Sequence[tuple], variants: Sequence[str], labels: dict,
                       colors: dict, paired: Optional[pd.DataFrame] = None, columns: int = 3,
                       xlabel: str = IMPROVEMENT_LABEL, row: str = "variant", theme: Any = None):
    """Mean paired improvement over the baseline per variant with its t interval.

    :param summary: Summary rows with ``variant``, ``mean``, ``ci_low``, ``ci_high`` and the
        columns used by the panel selectors (``summarize`` output).
    :param panels: ``(title, selector)`` pairs, one panel each; a selector maps columns to
        the value (or list of values) of the rows shown in the panel.
    :param variants: Variants in display order (top to bottom); missing ones stay empty.
    :param labels: Display label per variant.
    :param colors: Colour per variant.
    :param paired: Optional per-repetition rows (``variant``, ``improvement`` and the
        selector columns) drawn as small points around the mean.
    :param columns: Panels per row.
    :param xlabel: Label of the improvement axis.
    :param row: Column naming the rows (``variant`` by default; e.g. ``contrast`` or ``term``).
    :return: Figure: one row per variant, diamond = mean, bar = interval, dots = repetitions,
        vertical line = baseline (zero improvement).
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(panels), columns, resolved, width=4.8, height=0.34 * len(variants) + 1.5)
    positions = {variant: index for index, variant in enumerate(variants)}
    for panel, (ax, (title, selector)) in enumerate(zip(axes, panels)):
        rows = _select(summary, selector)
        for record in rows.itertuples():
            name = getattr(record, row)
            if name not in positions:
                continue
            y = positions[name]
            color = colors.get(name, resolved.baseline_color)
            ax.hlines(y, record.ci_low, record.ci_high, color=color, linewidth=2.2, alpha=resolved.primary_alpha)
            ax.scatter([record.mean], [y], marker="D", s=34, color=color, edgecolor=resolved.text_color,
                       linewidth=0.5, zorder=4)
        if paired is not None:
            points = _select(paired, selector)
            for variant, frame in points.groupby(row):
                if variant not in positions:
                    continue
                offsets = np.linspace(-0.18, 0.18, len(frame)) if len(frame) > 1 else np.zeros(1)
                ax.scatter(frame.improvement, positions[variant] + offsets, s=9, color=colors.get(variant),
                           alpha=resolved.secondary_alpha, zorder=3)
        ax.axvline(0.0, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                   linewidth=resolved.reference_line_width)
        ax.set_yticks(range(len(variants)), [labels.get(variant, variant) for variant in variants]
                      if panel % max(1, min(columns, len(panels))) == 0 else [""] * len(variants))
        ax.set_ylim(len(variants) - 0.5, -0.5)
        ax.set_xlabel(xlabel, fontsize=resolved.tick_font_size)
        _title(ax, title, resolved)
    figure.tight_layout()
    return figure


def epoch_curves(frame: pd.DataFrame, *, levels: Sequence[str], labels: dict, colors: dict, group: str = "variant",
                 x: str = "epoch", y: str = "mean", low: Optional[str] = None, high: Optional[str] = None,
                 line_styles: Optional[dict] = None, reference: Optional[float] = None, xlabel: str = "epoch",
                 ylabel: str = "", title: str = "", log_y: bool = False, marker: Optional[str] = "o", ax=None,
                 theme: Any = None):
    """One curve per level along epochs, with an optional band between two columns.

    :param frame: Rows with ``group``, ``x``, ``y`` and optionally the band columns.
    :param levels: Levels of ``group`` drawn, in legend order.
    :param low: Column of the lower band edge (e.g. ``ci_low`` or ``min``).
    :param high: Column of the upper band edge.
    :param line_styles: Optional line style per level.
    :param reference: Optional horizontal reference (e.g. 0 = baseline for improvements).
    :param marker: Point marker (``None`` for dense curves such as per-bin profiles).
    :return: Figure.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.5, 4.4), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for level in levels:
        lines = frame[frame[group] == level].sort_values(x)
        if lines.empty:
            continue
        color = colors.get(level, resolved.baseline_color)
        ax.plot(lines[x], lines[y], color=color, linewidth=resolved.line_width, marker=marker, markersize=3,
                linestyle=(line_styles or {}).get(level, "-"), label=labels.get(level, level))
        if low is not None and high is not None:
            ax.fill_between(lines[x].to_numpy(float), lines[low].to_numpy(float), lines[high].to_numpy(float),
                            color=color, alpha=resolved.uncertainty_alpha, linewidth=0)
    if reference is not None:
        ax.axhline(reference, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                   linewidth=resolved.reference_line_width)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _title(ax, title, resolved)
    ax.legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


def stage_paths(summary: pd.DataFrame, *, stages: Sequence[str], stage_labels: dict, variants: Sequence[str],
                labels: dict, colors: dict, ylabel: str = IMPROVEMENT_LABEL, title: str = "", ax=None,
                theme: Any = None):
    """Improvement over the baseline of every variant across the stages of its lineage.

    :param summary: Summary rows of one metric with ``variant``, ``stage``, ``mean``,
        ``ci_low`` and ``ci_high``.
    :param stages: Stages in lineage order (x positions).
    :return: Figure: one line per variant through the stage means, bars = intervals
        (dodged per variant), dashed zero line = baseline.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.5, 4.6), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    width = 0.5 / max(1, len(variants))
    for offset, variant in enumerate(variants):
        rows = summary[summary.variant == variant].set_index("stage").reindex(list(stages))
        x = np.arange(len(stages)) - 0.25 + width * (offset + 0.5)
        color = colors.get(variant, resolved.baseline_color)
        ax.plot(x, rows["mean"], color=color, marker="o", markersize=4, linewidth=resolved.line_width,
                label=labels.get(variant, variant))
        ax.vlines(x, rows.ci_low, rows.ci_high, color=color, linewidth=1.2, alpha=resolved.secondary_alpha)
    ax.axhline(0.0, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
               linewidth=resolved.reference_line_width)
    ax.set_xticks(range(len(stages)), [stage_labels.get(stage, stage) for stage in stages])
    ax.set_ylabel(ylabel, fontsize=resolved.tick_font_size)
    _title(ax, title, resolved)
    figure.tight_layout()
    return figure


def reliability_diagram(bins: pd.DataFrame, *, levels: Sequence[str], labels: dict, colors: dict,
                        group: str = "variant", title: str = "", ax=None, theme: Any = None):
    """Observed positive rate against the mean predicted probability per probability bin.

    Bins of every repetition of a level are pooled with their entry counts as weights.

    :param bins: ``calibration_bins`` rows (``bin``, ``entries``, ``mean_probability``,
        ``positive_rate`` and ``group``).
    :return: Figure: one line per level; marker area grows with the log number of entries;
        the diagonal is perfect calibration.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.0, 5.4), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for level in levels:
        frame = bins[(bins[group] == level) & (bins.entries > 0)]
        if frame.empty:
            continue
        weights = frame.entries.to_numpy(float)
        pooled = frame.assign(weighted_probability=frame.mean_probability * weights,
                              weighted_rate=frame.positive_rate * weights).groupby("bin")[
            ["weighted_probability", "weighted_rate", "entries"]].sum()
        probability = pooled.weighted_probability / pooled.entries
        rate = pooled.weighted_rate / pooled.entries
        color = colors.get(level, resolved.baseline_color)
        ax.plot(probability, rate, color=color, linewidth=resolved.line_width, label=labels.get(level, level))
        ax.scatter(probability, rate, s=4 + 3 * np.log10(pooled.entries.to_numpy(float) + 1) ** 2, color=color,
                   alpha=resolved.marker_alpha, zorder=3)
    ax.plot([0, 1], [0, 1], color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
            linewidth=resolved.reference_line_width)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("annotated fraction")
    _title(ax, title, resolved)
    ax.legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


def spectrum_overlays(cases: pd.DataFrame, *, keys: Sequence[tuple], levels: Sequence[str], labels: dict,
                      colors: dict, group: str = "variant", columns: int = 2, zoom: Optional[float] = None,
                      theme: Any = None):
    """Input spectrum with the reconstructions of several models on the same pixel.

    :param cases: ``cases`` rows (one per bin) with ``population``, ``row``, ``mz``, ``input``,
        ``output`` and ``group``; the input is identical for every level of one pixel.
    :param keys: ``(population, row)`` pairs, one panel each.
    :param levels: Models (levels of ``group``) drawn, in legend order.
    :param zoom: Optional half width (m/z) around the strongest input peak; the full range otherwise.
    :return: Figure: input in grey, one reconstruction per level in its colour.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(keys), columns, resolved, width=6.5, height=3.2)
    for ax, (population, row) in zip(axes, keys):
        frame = cases[(cases.population == population) & (cases.row == row)]
        reference = frame[frame[group] == levels[0]].sort_values("mz")
        ax.plot(reference.mz, reference.input, color=resolved.input_color, linewidth=resolved.input_line_width,
                label="input")
        for level in levels:
            lines = frame[frame[group] == level].sort_values("mz")
            ax.plot(lines.mz, lines.output, color=colors.get(level, resolved.baseline_color),
                    alpha=resolved.primary_alpha, linewidth=resolved.reconstruction_line_width,
                    label=labels.get(level, level))
        if zoom is not None and len(reference):
            apex = float(reference.mz.to_numpy()[int(np.argmax(reference.input.to_numpy()))])
            inside = (reference.mz >= apex - zoom) & (reference.mz <= apex + zoom)
            top = float(max(reference.input[inside].max(), frame[frame.mz.between(apex - zoom, apex + zoom)]
                            .output.max()))
            ax.set_xlim(apex - zoom, apex + zoom)
            ax.set_ylim(0.0, 1.15 * top if top > 0 else 1.0)
        _title(ax, f"{DISPLAY_NAMES.get(population, population)}, row {row}", resolved)
        ax.set_xlabel("m/z")
    axes[0].legend(frameon=False, fontsize=resolved.legend_font_size)
    figure.tight_layout()
    return figure


# --------------------------------------------------
# Section: absolute results against the real-only baseline
# --------------------------------------------------

#: Colours of a result that is better / worse than the real-only baseline.
BETTER_COLOR = "#1a9850"
WORSE_COLOR = "#d73027"


def _signed(difference: float, direction: str) -> float:
    """Difference to the baseline signed so that positive = better."""
    return difference if direction == "higher" else -difference


def value_violins(frame: pd.DataFrame, *, value: str, group: str, order: Sequence[str], labels: dict,
                  baseline: str, direction: str, ylabel: str, title: str = "", ax=None, theme: Any = None):
    """Absolute values per group with the real-only baseline as the reference group.

    Every group is a violin (when it has at least three values) with its points (one per
    repetition or per unit of the frame); the baseline is grey and its mean is drawn as a
    dashed line across the panel. Other groups are green when their mean is better than the
    baseline mean and red when it is worse, so the direction is readable at a glance.

    :param frame: Rows with ``group`` and ``value`` columns.
    :param order: Groups in display order; the baseline should be first.
    :param direction: ``higher`` or ``lower`` (which values are better).
    :return: Figure (a new one unless ``ax`` is given).
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=(max(5.0, 0.62 * len(order) + 1.5), 4.0), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    reference = frame.loc[frame[group] == baseline, value].astype(float)
    reference_mean = float(reference.mean()) if len(reference) else np.nan
    for position, name in enumerate(order):
        values = frame.loc[frame[group] == name, value].astype(float).dropna().to_numpy()
        if not values.size:
            continue
        if name == baseline or not np.isfinite(reference_mean):
            color = resolved.baseline_color
        else:
            color = BETTER_COLOR if _signed(values.mean() - reference_mean, direction) > 0 else WORSE_COLOR
        if values.size >= 3 and np.ptp(values) > 0:
            body = ax.violinplot([values], positions=[position], widths=0.7, showextrema=False)
            for patch in body["bodies"]:
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(0.25)
        offsets = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(1)
        ax.scatter(position + offsets, values, s=16, color=color, edgecolor=resolved.text_color, linewidth=0.3,
                   zorder=3)
        ax.hlines(values.mean(), position - 0.28, position + 0.28, color=color, linewidth=2.0, zorder=4)
    if np.isfinite(reference_mean):
        ax.axhline(reference_mean, color=resolved.baseline_color, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xticks(range(len(order)), [labels.get(name, name) for name in order], rotation=45, ha="right",
                  fontsize=resolved.tick_font_size)
    ax.set_ylabel(ylabel, fontsize=resolved.tick_font_size)
    heading = f"{title} ({direction} is better)" if title else f"{direction} is better"
    ## REMARK: long panel titles are wrapped so that neighbouring panels do not overlap.
    _title(ax, "\n".join(textwrap.wrap(heading, 48)), resolved)
    return figure


def metric_panels(frame: pd.DataFrame, *, panels: Sequence[tuple], group: str, order: Sequence[str], labels: dict,
                  baseline: str, value: str = "value", columns: int = 3, theme: Any = None):
    """Grid of :func:`value_violins`, one panel per ``(title, selector, direction)``.

    :param panels: ``(title, selector, direction)``; the selector maps columns to the value
        (or list of values) of the rows shown in the panel.
    :return: Figure.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(panels), columns, resolved, width=max(4.6, 0.5 * len(order) + 1.4), height=4.2)
    for ax, (title, selector, direction) in zip(axes, panels):
        value_violins(_select(frame, selector), value=value, group=group, order=order, labels=labels,
                      baseline=baseline, direction=direction, ylabel=value, title=title, ax=ax, theme=resolved)
    figure.tight_layout()
    return figure


def window_values(frame: pd.DataFrame, *, populations: Sequence[str], variants: Sequence[str], labels: dict,
                  colors: dict, baseline: str, ylabel: str, group: str = "variant", theme: Any = None):
    """Absolute window profile along m/z of the baseline and every variant.

    :param frame: One row per model and window (``population``, ``window_lower``,
        ``window_upper``, ``value`` and ``group``).
    :return: Figure: one panel per population; x = window centre (m/z); thick black line
        with a band = baseline mean and range over repetitions; thin lines = variant means.
    """
    resolved = resolve_theme(theme)
    figure, axes = _grid(len(populations), len(populations), resolved, width=7.2, height=4.4)
    for ax, population in zip(axes, populations):
        shown = frame[frame.population == population].assign(
            centre=lambda rows: (rows.window_lower + rows.window_upper) / 2.0)
        for name in [baseline, *variants]:
            rows = shown[shown[group] == name]
            if rows.empty:
                continue
            statistics = rows.groupby("centre").value.agg(["mean", "min", "max"]).sort_index()
            if name == baseline:
                ax.fill_between(statistics.index, statistics["min"], statistics["max"], color=resolved.text_color,
                                alpha=0.15, linewidth=0)
                ax.plot(statistics.index, statistics["mean"], color=resolved.text_color, linewidth=2.6,
                        marker="o", markersize=4, label=labels.get(name, name), zorder=5)
            else:
                ax.plot(statistics.index, statistics["mean"], color=colors.get(name, resolved.baseline_color),
                        linewidth=1.2, marker=".", alpha=resolved.primary_alpha, label=labels.get(name, name))
        ax.set_xlabel("window centre (m/z)")
        ax.set_ylabel(ylabel, fontsize=resolved.tick_font_size)
        _title(ax, DISPLAY_NAMES.get(population, population), resolved)
    axes[-1].legend(frameon=False, fontsize=resolved.legend_font_size, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    figure.tight_layout()
    return figure


def comparison_table(frame: pd.DataFrame, *, index: str, columns: str, baseline: str, direction: dict,
                     order: Optional[Sequence[str]] = None, column_order: Optional[Sequence[str]] = None,
                     value: str = "value", top: int = 3, digits: int = 3):
    """Mean +/- SD table coloured against the baseline row.

    Cells of every non-baseline row are green when the mean is better than the baseline mean
    of the same column and red when it is worse; the colour intensity grows with the size of
    the difference relative to the largest difference in the column. The ``top`` best rows of
    every column get a black frame.

    :param frame: Rows with ``index``, ``columns`` and ``value`` (one per repetition).
    :param direction: Column value mapped to ``higher`` or ``lower``; missing columns are not coloured.
    :param order: Row order (the baseline first); ``column_order``: column order.
    :return: ``pandas.io.formats.style.Styler``.
    """
    subset = frame[[index, columns, value]]
    grouped = subset.groupby([index, columns])[value]
    mean = grouped.mean().unstack(columns)
    sd = grouped.std().unstack(columns)
    if order is not None:
        mean, sd = mean.reindex([name for name in order if name in mean.index]), sd.reindex(
            [name for name in order if name in sd.index])
    if column_order is not None:
        kept = [name for name in column_order if name in mean.columns]
        mean, sd = mean[kept], sd[kept]
    text = mean.map(lambda number: f"{number:.{digits}f}") + " +/- " + sd.map(
        lambda number: f"{number:.{digits}f}" if np.isfinite(number) else "-")

    def styles(_: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame("", index=mean.index, columns=mean.columns)
        if baseline not in mean.index:
            return result
        for column in mean.columns:
            sense = direction.get(column)
            if sense is None:
                continue
            gains = (mean[column] - mean.loc[baseline, column]).map(lambda difference: _signed(difference, sense))
            gains = gains.drop(baseline)
            scale = float(np.nanmax(np.abs(gains.to_numpy()))) if gains.notna().any() else 0.0
            ranked = gains.dropna().sort_values(ascending=False).index[:top]
            for name, gain in gains.items():
                if not np.isfinite(gain) or scale == 0:
                    continue
                strength = 0.15 + 0.6 * min(abs(gain) / scale, 1.0)
                rgb = (26, 152, 80) if gain > 0 else (215, 48, 39)
                style = f"background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {strength:.2f})"
                if name in ranked:
                    style += "; border: 2px solid black"
                result.loc[name, column] = style
            result.loc[baseline, column] = "background-color: rgba(128, 128, 128, 0.25); font-weight: bold"
        return result

    return text.style.apply(styles, axis=None)


def _broken(mz: np.ndarray, *values: np.ndarray, gap: float = 1.0) -> tuple[np.ndarray, ...]:
    """Insert NaN where consecutive m/z values are further apart than ``gap`` (non-displayed windows)."""
    breaks = np.flatnonzero(np.diff(mz) > gap) + 1
    return tuple(np.insert(np.asarray(array, dtype=float), breaks, np.nan) for array in (mz, *values))


def _dense(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore bins dropped as all-zero: zero intensities, m/z interpolated over the bin index."""
    full = frame.set_index("bin").reindex(np.arange(int(frame["bin"].min()), int(frame["bin"].max()) + 1))
    full["mz"] = full["mz"].interpolate(limit_direction="both")
    return full.fillna({"input": 0.0, "baseline_output": 0.0, "output": 0.0}).reset_index()


def paired_spectra(spectra: pd.DataFrame, *, variant_label: str, variant_color: str, half_width: float = 15.0,
                   kinds: Sequence[str] = ("baseline_worst", "most_improved", "most_worsened", "variant_worst",
                                           "variant_best"), center: Optional[str] = None, dense: bool = False,
                   theme: Any = None):
    """Input, baseline reconstruction and variant reconstruction of the same pixels.

    :param spectra: Rows of one variant (``pixel_kind``, ``rank``, ``row``, ``mz``, ``input``,
        ``baseline_output``, ``output``, ``masserstein_baseline``, ``masserstein``; optional
        ``dataset_id``, ``x``, ``y``, ``population``, ``worst_window``).
    :param center: Optional column holding the zoom centre (m/z) per pixel, e.g. a rare peak;
        the strongest input peak otherwise. The centre is marked by a dotted line.
    :param dense: Restore all-zero bins that were dropped from the table (requires ``bin``).
    :return: Figure: one pair of rows per pixel (signal above, residual output - input
        below); columns = full range, +/- ``half_width`` around the centre, and the
        variant's largest-contribution window when ``worst_window`` is given (+/- 4
        ``half_width`` around the centre otherwise). Grey = input, black = baseline,
        colour = variant.
    """
    resolved = resolve_theme(theme)
    keys = [column for column in ("population", "pixel_kind", "rank", "row") if column in spectra.columns]
    pixels = (spectra.drop_duplicates(keys)
              .assign(order=lambda rows: rows.pixel_kind.map({kind: position for position, kind in enumerate(kinds)}))
              .dropna(subset=["order"]).sort_values([column for column in ("population", "order", "rank") if column in spectra.columns]))
    height = 3.9 * max(len(pixels), 1)
    figure = plt.figure(figsize=(16.5, height), dpi=resolved.figure_dpi)
    ## REMARK: a fixed top margin in inches keeps room for a suptitle without a blank band in tall figures.
    outer = figure.add_gridspec(max(len(pixels), 1), 3, hspace=0.6, wspace=0.22, top=1.0 - 0.75 / height,
                                bottom=0.45 / height)
    for index, pixel in enumerate(pixels.itertuples()):
        selector = np.ones(len(spectra), dtype=bool)
        for column in keys:
            if column != "rank":
                selector &= spectra[column].to_numpy() == getattr(pixel, column)
        frame = spectra[selector].sort_values("mz")
        frame = _dense(frame) if dense else frame
        mz, observed, base, produced = _broken(frame.mz.to_numpy(), frame.input.to_numpy(),
                                               frame.baseline_output.to_numpy(), frame.output.to_numpy())
        middle = (float(getattr(pixel, center)) if center else
                  float(frame.mz.to_numpy()[int(np.argmax(frame.input.to_numpy()))]))
        label = "rare peak" if center else "strongest peak"
        ranges = [(np.nanmin(mz), np.nanmax(mz), "full range"),
                  (middle - half_width, middle + half_width, f"{label} {middle:.1f} +/- {half_width:g}")]
        if "worst_window" in spectra.columns:
            lower, upper = (float(value) for value in str(pixel.worst_window).split("-"))
            ranges.append((lower, upper, f"variant's largest-contribution window {pixel.worst_window}"))
        else:
            ranges.append((middle - 4 * half_width, middle + 4 * half_width, f"{label} {middle:.1f} +/- {4 * half_width:g}"))
        where = " ".join(str(getattr(pixel, column)) for column in ("population", "dataset_id") if hasattr(pixel, column))
        if hasattr(pixel, "x"):
            where += f" ({pixel.x}, {pixel.y})"
        for column, (low, high, caption) in enumerate(ranges):
            inner = outer[index, column].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.06)
            signal = figure.add_subplot(inner[0])
            residual = figure.add_subplot(inner[1], sharex=signal)
            signal.tick_params(labelbottom=False)
            signal.plot(mz, observed, color=resolved.input_color, linewidth=resolved.input_line_width, label="input")
            signal.plot(mz, base, color=resolved.text_color, linewidth=resolved.reconstruction_line_width,
                        alpha=0.85, label="real-only baseline")
            signal.plot(mz, produced, color=variant_color, linewidth=resolved.reconstruction_line_width,
                        alpha=resolved.primary_alpha, label=variant_label)
            residual.axhline(0.0, color=resolved.grid_color, linewidth=0.8)
            residual.plot(mz, base - observed, color=resolved.text_color, linewidth=0.8)
            residual.plot(mz, produced - observed, color=variant_color, linewidth=0.8)
            if center and column > 0:
                signal.axvline(middle, color=resolved.baseline_color, linestyle=":", linewidth=1.0)
            inside = (mz >= low) & (mz <= high)
            if np.any(inside):
                top = float(np.nanmax(np.concatenate([observed[inside], base[inside], produced[inside]])))
                signal.set_ylim(0.0, 1.15 * top if top > 0 else 1.0)
                spread = float(np.nanmax(np.abs(np.concatenate([base[inside] - observed[inside],
                                                                produced[inside] - observed[inside]]))))
                residual.set_ylim(-1.15 * spread if spread > 0 else -1.0, 1.15 * spread if spread > 0 else 1.0)
            signal.set_xlim(low, high)
            residual.set_xlabel("m/z")
            ## The pixel is described once per row (first column); the other columns name their range only
            others = (f" (other baseline repetitions {pixel.masserstein_baseline_others:.2f})"
                      if hasattr(pixel, "masserstein_baseline_others") else "")
            heading = (f"{pixel.pixel_kind}: {where}\nW1 baseline {pixel.masserstein_baseline:.2f}{others} -> variant "
                       f"{pixel.masserstein:.2f}; {caption}")
            _title(signal, heading if column == 0 else caption, resolved)
            if index == 0 and column == 0:
                signal.legend(frameon=False, fontsize=resolved.legend_font_size)
    return figure
