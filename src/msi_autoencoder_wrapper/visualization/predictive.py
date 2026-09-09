"""Shared distribution-first figures for molecular head selection.

Every function here takes long-form analytical records and returns a Matplotlib
figure styled through :mod:`msi_autoencoder_wrapper.visualization.theme`. Notebooks
choose *which* comparison to draw; they never set colors, sizes or transparencies.

Colour carries one meaning throughout: the training objective's loss family. Runs
of the same family that differ only in their weighting scheme receive related
shades of one base colour, so a family reads as a block while its members stay
distinguishable.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from .metrics import plot_violin_with_points
from .theme import resolve_theme

#: Loss families in a fixed order, so a colour keeps its meaning across notebooks.
LOSS_FAMILIES = ("ClassBalancedMultiLabelBCELoss", "PositiveWeightedMultiLabelBCELoss",
                 "SignalMaskedBCELoss", "ThreeStateCrossEntropyLoss", "VariationalPULoss",
                 "SymmetricPURankingLoss")


def _color(label, theme):
    """Assign stable related shades within a head-loss family.

    :param label: Semantic condition label, normally ``name (LossClassName)``.
    :param theme: Resolved visualization theme.
    :return: RGB triple; lighter shades separate weighting variants of one family.
    :rtype: tuple[float, float, float]
    """
    if label in theme.model_overrides:
        return theme.model_overrides[label]
    text = str(label)
    family = next((index for index, name in enumerate(LOSS_FAMILIES) if name in text), None)
    if family is None:
        # REMARK: A label outside the known families (a paired contrast, a stratum)
        # still needs a stable colour. Hash it rather than collapsing every unknown
        # label onto the first palette entry, which previously made them identical.
        family = sum(text.encode()) % len(theme.model_palette)
    base = np.asarray(to_rgb(theme.model_palette[family % len(theme.model_palette)]))
    amount = .3 if "per_class" in text else .15 if "global" in text else 0
    return tuple(base + amount * (1 - base))


def short_label(label) -> str:
    """Return the display form of a condition label: its name without the loss class.

    The stored labels carry the loss class in parentheses so that tables stay
    unambiguous and so that :func:`_color` can group a family. On an axis that makes
    every tick several times longer than it needs to be and squeezes the data into a
    fraction of the figure. The class is already carried by colour, so displayed text
    shows the name alone. **Only ever apply this to displayed text**; colour lookup
    and grouping must keep using the full label.

    :param label: Full condition label, normally ``name (LossClassName)``.
    :return: The part before the parenthesis, unchanged if there is none.
    :rtype: str
    """
    return str(label).split(" (")[0]


def _group_order(frame: pd.DataFrame, group: str, order: Sequence[str] | None) -> list:
    """Return the requested group order restricted to groups actually present."""
    present = list(pd.unique(frame[group].dropna()))
    if order is None:
        return sorted(present, key=str)
    return [value for value in order if value in present] + sorted(
        (value for value in present if value not in set(order)), key=str)


def _finish(ax, *, xlabel=None, ylabel=None, title=None, legend=False, theme=None, rotate_ticks=False):
    """Apply shared axis decoration so notebooks never restyle an axis."""
    ax.set(**{key: value for key, value in
              (("xlabel", xlabel), ("ylabel", ylabel), ("title", title)) if value is not None})
    if rotate_ticks:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=theme.legend_font_size, loc=theme.legend_location,
                  ncols=theme.legend_columns, frameon=theme.legend_frame)
    ax.grid(theme.grid_visible, axis=theme.grid_axis, color=theme.grid_color,
            alpha=theme.grid_alpha, linewidth=theme.grid_line_width)


def distributions(frame: pd.DataFrame, *, value: str = "value", group: str = "label", title: str = "",
                  order: Sequence[str] | None = None, reference: float | None = None,
                  reference_label: str = "reference", ylabel: str | None = None, legend: bool = True,
                  ax=None, theme=None):
    """Plot complete group distributions and a deterministic point overlay.

    :param frame: Observation-level records.
    :param value: Numeric column on the y-axis.
    :param group: Semantic grouping column on the x-axis.
    :param title: Figure title.
    :param order: Explicit left-to-right group order; missing groups are dropped.
    :param reference: Optional horizontal reference level, e.g. a baseline mean.
    :param reference_label: Legend entry for ``reference``.
    :param ylabel: Axis label; defaults to ``value``.
    :param legend: Draw the reference-line legend; suppressed inside a facet grid.
    :param ax: Existing axis; a new figure is created when omitted.
    :param theme: Existing visualization theme or preset.
    :return: Matplotlib figure. Overlay keeps <=200 evenly spaced ordered rows.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    labels = _group_order(frame, group, order)
    if ax is None:
        figure, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6), dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for position, label in enumerate(labels):
        values = frame.loc[frame[group] == label, value].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        color = _color(label, resolved)
        if len(values) > 1 and np.ptp(values) > 0:
            shown = values[np.linspace(0, len(values) - 1, min(len(values), 200), dtype=int)]
            plot_violin_with_points(values, position, ax=ax, color=color, overlay_values=shown, theme=resolved)
        elif len(values):
            ax.scatter(np.full(len(values), position), values, color=color, alpha=resolved.primary_alpha)
    if reference is not None and np.isfinite(reference):
        ax.axhline(reference, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                   linewidth=resolved.reference_line_width, label=reference_label)
    ax.set_xticks(range(len(labels)), [short_label(label) for label in labels])
    _finish(ax, ylabel=ylabel or value, title=title, legend=legend and reference is not None,
            theme=resolved, rotate_ticks=True)
    figure.tight_layout()
    return figure


def metric_panels(frame: pd.DataFrame, *, metrics: Sequence[str], metric_column: str = "metric",
                  value: str = "value", group: str = "label", order: Sequence[str] | None = None,
                  references: dict | None = None, reference_label: str = "baseline mean",
                  columns: int = 2, title: str = "", theme=None):
    """Show several metrics side by side, each as a complete per-group distribution.

    A single metric answers one question; a decision needs the panel. Laying the
    metrics out on shared group positions makes it directly visible whether two
    metrics move together (a genuine quality change) or in opposite directions
    (an effect localized to one part of the class population).

    :param frame: Long-form records carrying a metric name column.
    :param metrics: Metric names, one panel each, in the given order.
    :param metric_column: Column holding the metric name.
    :param value: Numeric column plotted on every panel's y-axis.
    :param group: Semantic grouping column shared by all panels.
    :param order: Explicit left-to-right group order.
    :param references: Optional per-metric horizontal reference levels.
    :param reference_label: Legend entry naming what the reference level represents.
    :param columns: Panels per row.
    :param title: Figure title.
    :param theme: Existing visualization theme.
    :return: Matplotlib figure with one panel per requested metric.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    metrics = list(metrics)
    rows = int(np.ceil(len(metrics) / max(1, columns)))
    labels = _group_order(frame, group, order)
    width = max(8.0, len(labels) * 1.1) * min(columns, max(1, len(metrics)))
    figure, axes = plt.subplots(rows, columns, figsize=(width, 4.2 * rows),
                                dpi=resolved.figure_dpi, squeeze=False)
    for axis, metric in zip(axes.ravel(), metrics):
        distributions(frame[frame[metric_column] == metric], value=value, group=group,
                      title=metric, order=labels, ylabel=metric, ax=axis, theme=resolved,
                      reference=(references or {}).get(metric), reference_label=reference_label)
    for axis in axes.ravel()[len(metrics):]:
        axis.set_visible(False)
    if title:
        figure.suptitle(title, y=1.001, fontsize=resolved.title_font_size)
    figure.tight_layout()
    return figure


def trajectories(frame: pd.DataFrame, *, x: str, y: str = "value", line: str = "model_id",
                 group: str = "label", order: Sequence[str] | None = None, title: str = "",
                 xlabel: str | None = None, ylabel: str | None = None, log_y: bool = False,
                 legend: bool = True, ax=None, theme=None):
    """Plot every individual trajectory with semantic model labels.

    :param frame: Long-form observations with label and line identifiers.
    :param x: Ordered horizontal coordinate.
    :param y: Vertical measurement column.
    :param line: Independent trajectory identifier; one line is drawn per value.
    :param group: Semantic grouping column controlling colour and legend.
    :param order: Explicit legend order.
    :param title: Figure title.
    :param xlabel: Axis label; defaults to ``x``.
    :param ylabel: Axis label; defaults to ``y``.
    :param log_y: Use a logarithmic vertical axis.
    :param ax: Existing axis; a new figure is created when omitted.
    :param theme: Existing theme.
    :return: Matplotlib figure. Repetitions are never averaged away.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for label in _group_order(frame, group, order):
        part = frame[frame[group] == label]
        for number, (_, run) in enumerate(part.groupby(line, sort=True)):
            run = run.sort_values(x)
            ax.plot(run[x], run[y], color=_color(label, resolved),
                    alpha=resolved.overlapping_signal_alpha,
                    label=short_label(label) if number == 0 else None)
    if log_y:
        ax.set_yscale("log")
    _finish(ax, xlabel=xlabel or x, ylabel=ylabel or y, title=title, legend=legend, theme=resolved)
    figure.tight_layout()
    return figure


def curves(frame: pd.DataFrame, *, x: str = "x", y: str = "y", line: str = "model_id",
           group: str = "label", order: Sequence[str] | None = None, diagonal: bool = False,
           xlabel: str | None = None, ylabel: str | None = None, title: str = "", legend: bool = True,
           ax=None, theme=None):
    """Overlay one stored ranking curve per run on a shared coordinate grid.

    :param frame: Curve points interpolated onto the shared grid.
    :param x: Horizontal coordinate column, e.g. recall or false-positive rate.
    :param y: Vertical coordinate column, e.g. precision or true-positive rate.
    :param line: Identifier separating individual runs.
    :param group: Semantic grouping column controlling colour and legend.
    :param order: Explicit legend order.
    :param diagonal: Draw the chance diagonal, meaningful for an ROC curve only.
    :param xlabel: Axis label; defaults to ``x``.
    :param ylabel: Axis label; defaults to ``y``.
    :param title: Figure title.
    :param ax: Existing axis.
    :param theme: Existing theme.
    :return: Matplotlib figure; every run is drawn, none are averaged.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    figure = trajectories(frame, x=x, y=y, line=line, group=group, order=order, legend=legend,
                          xlabel=xlabel or x, ylabel=ylabel or y, title=title, ax=ax, theme=resolved)
    axis = figure.axes[0] if ax is None else ax
    if diagonal:
        axis.plot([0, 1], [0, 1], color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                  linewidth=resolved.reference_line_width, label="chance")
        if legend:
            axis.legend(fontsize=resolved.legend_font_size, loc=resolved.legend_location,
                        ncols=resolved.legend_columns, frameon=resolved.legend_frame)
    return figure


def scatter(frame: pd.DataFrame, *, x: str, y: str, group: str = "label",
            order: Sequence[str] | None = None, annotate: str | None = None, identity: bool = False,
            xlabel: str | None = None, ylabel: str | None = None, title: str = "", legend: bool = True,
            ax=None, theme=None):
    """Plot paired measurements with one point per supplied observation.

    :param frame: Paired records, including the grouping column.
    :param x: Horizontal measurement.
    :param y: Vertical measurement.
    :param group: Semantic grouping column controlling colour and legend.
    :param order: Explicit legend order.
    :param annotate: Optional column whose values label individual points.
    :param identity: Draw the ``y = x`` line, meaningful only for comparable axes.
    :param xlabel: Axis label; defaults to ``x``.
    :param ylabel: Axis label; defaults to ``y``.
    :param title: Figure title.
    :param ax: Existing axis.
    :param theme: Existing theme.
    :return: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for label in _group_order(frame, group, order):
        part = frame[frame[group] == label]
        ax.scatter(part[x], part[y], label=short_label(label), alpha=resolved.marker_alpha,
                   color=_color(label, resolved), s=12)
        if annotate:
            for _, row in part.iterrows():
                ax.annotate(short_label(row[annotate]), (row[x], row[y]), fontsize=6,
                            color=resolved.muted_text_color, xytext=(2, 2), textcoords="offset points")
    if identity and len(frame):
        limits = [min(frame[x].min(), frame[y].min()), max(frame[x].max(), frame[y].max())]
        ax.plot(limits, limits, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                linewidth=resolved.reference_line_width, label="y = x")
    _finish(ax, xlabel=xlabel or x, ylabel=ylabel or y, title=title, legend=legend, theme=resolved)
    figure.tight_layout()
    return figure


def histogram_overlay(frame: pd.DataFrame, *, group: str = "label", line: str = "model_id",
                      order: Sequence[str] | None = None, x_label: str | None = None,
                      normalize: bool = True, log_y: bool = False, title: str = "", legend: bool = True,
                      ax=None, theme=None):
    """Plot densities from stored bin counts without expanding the observations.

    The counts were produced from the complete population, so the drawn density uses
    every entry even though the underlying observations are never materialized in the
    notebook. Bin edges are fixed at precomputation time and therefore identical for
    every overlaid run.

    :param frame: Records with ``left_edge``, ``right_edge`` and ``count`` columns.
    :param group: Semantic grouping column controlling colour and legend.
    :param line: Identifier separating individual runs drawn as separate steps.
    :param order: Explicit legend order.
    :param x_label: Horizontal axis label naming the binned quantity.
    :param normalize: Divide counts by the total and the bin width to obtain a density.
    :param log_y: Use a logarithmic vertical axis, which is normally required when
        one evidence state holds orders of magnitude more entries than another.
    :param title: Figure title.
    :param ax: Existing axis.
    :param theme: Existing theme.
    :return: Matplotlib figure; each step outline is one run's complete distribution.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for label in _group_order(frame, group, order):
        part = frame[frame[group] == label]
        for number, (_, run) in enumerate(part.groupby(line, sort=True)):
            run = run.sort_values("left_edge")
            widths = (run.right_edge - run.left_edge).to_numpy()  # (H,)
            counts = run["count"].to_numpy(dtype=float)  # (H,)
            values = counts / (max(1.0, counts.sum()) * widths) if normalize else counts  # (H,)
            edges = np.r_[run.left_edge.to_numpy(), run.right_edge.iloc[-1]]  # (H+1,)
            ax.stairs(values, edges, color=_color(label, resolved),
                      alpha=resolved.overlapping_signal_alpha,
                      label=short_label(label) if number == 0 else None)
    if log_y:
        ax.set_yscale("log")
    _finish(ax, xlabel=x_label or "value", ylabel="density" if normalize else "entries",
            title=title, legend=legend, theme=resolved)
    figure.tight_layout()
    return figure


def facets(frame: pd.DataFrame, *, facet: str, plot=histogram_overlay, facet_order: Sequence[str] | None = None,
           columns: int = 3, panel_size: tuple[float, float] = (6.0, 4.0), title: str = "",
           theme=None, **kwargs):
    """Apply one axis-level plotting function to each value of a facet column.

    Quantities that are not on a common scale, such as the objective values of
    different loss families, must not share an axis. Faceting keeps them in one
    figure without implying that their vertical positions are comparable.

    :param frame: Records carrying the facet column.
    :param facet: Column whose distinct values become panels.
    :param plot: Any function in this module that accepts ``ax`` and ``theme``.
    :param facet_order: Explicit panel order.
    :param columns: Panels per row.
    :param panel_size: Width and height of one panel in inches.
    :param title: Figure title.
    :param theme: Existing theme.
    :param kwargs: Forwarded unchanged to ``plot``.
    :return: Matplotlib figure; unused panels are hidden rather than left blank.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    panels = _group_order(frame, facet, facet_order)
    rows = int(np.ceil(max(1, len(panels)) / max(1, columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(panel_size[0] * columns, panel_size[1] * rows),
                                dpi=resolved.figure_dpi, squeeze=False)
    # REMARK: Every panel shares the same groups, so repeating the legend in each of
    # them wastes the plotting area the data needs. It is drawn once, on the first panel.
    accepts_legend = "legend" in inspect.signature(plot).parameters
    for position, (axis, panel) in enumerate(zip(axes.ravel(), panels)):
        panel_options = {**kwargs, "legend": position == 0} if accepts_legend else kwargs
        plot(frame[frame[facet] == panel], title=str(panel), ax=axis, theme=resolved, **panel_options)
    for axis in axes.ravel()[len(panels):]:
        axis.set_visible(False)
    if title:
        figure.suptitle(title, y=1.001, fontsize=resolved.title_font_size)
    figure.tight_layout()
    return figure


def paired_intervals(frame: pd.DataFrame, *, label: str = "label", value: str = "mean_difference",
                     low: str = "ci_low", high: str = "ci_high", count: str | None = "pairs",
                     xlabel: str = "paired difference", title: str = "", ax=None, theme=None):
    """Show matched-pair effects and their intervals on one common zero axis.

    Each row is one comparison between two conditions evaluated on identical seeds
    and contracts. The interval is a descriptive Student-t interval across those
    pairs, not a corrected test: with many comparisons drawn together, an interval
    that merely excludes zero is weak evidence on its own.

    :param frame: One row per comparison, already restricted to one measurement.
    :param label: Column naming the comparison.
    :param value: Column holding the mean paired difference.
    :param low: Lower interval bound column.
    :param high: Upper interval bound column.
    :param count: Optional column with the number of pairs, appended to the tick label.
    :param xlabel: Horizontal axis label stating the sign convention.
    :param title: Figure title.
    :param ax: Existing axis.
    :param theme: Existing theme.
    :return: Matplotlib figure; the dashed vertical line marks no effect.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    ordered = frame.sort_values(value, ascending=True).reset_index(drop=True)
    if ax is None:
        figure, ax = plt.subplots(figsize=(resolved.figure_size[0], max(3.0, .38 * len(ordered) + 1.5)),
                                  dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    for position, row in ordered.iterrows():
        color = _color(row[label], resolved)
        if np.isfinite(row.get(low, np.nan)) and np.isfinite(row.get(high, np.nan)):
            ax.plot([row[low], row[high]], [position, position], color=color,
                    linewidth=resolved.line_width, alpha=resolved.primary_alpha)
        ax.scatter([row[value]], [position], color=color, s=28, zorder=3, alpha=resolved.primary_alpha)
    ax.axvline(0.0, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
               linewidth=resolved.reference_line_width)
    ticks = [f"{short_label(row[label])} (n={int(row[count])})" if count and np.isfinite(row.get(count, np.nan))
             else short_label(row[label]) for _, row in ordered.iterrows()]
    ax.set_yticks(range(len(ordered)), ticks, fontsize=7)
    _finish(ax, xlabel=xlabel, title=title, theme=resolved)
    ax.grid(resolved.grid_visible, axis="x", color=resolved.grid_color,
            alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
    figure.tight_layout()
    return figure


def heatmap(frame: pd.DataFrame, *, index: str, columns: str, values: str, aggregate: str = "mean",
            colormap: str | None = None, value_format: str = "{:.3f}", title: str = "",
            colorbar_label: str | None = None, theme=None):
    """Show one measurement across two categorical axes with the numbers printed.

    Use this only where both axes are genuinely categorical, for example condition
    against class stratum. The printed value keeps the figure auditable; the colour
    is a reading aid, never the only carrier of the result.

    :param frame: Long-form records.
    :param index: Column forming the rows.
    :param columns: Column forming the columns.
    :param values: Numeric column shown in each cell.
    :param aggregate: Pandas aggregation applied to repeated observations in a cell.
    :param colormap: Matplotlib colormap; defaults to the theme's probability map.
    :param value_format: Format string for the printed cell value.
    :param title: Figure title.
    :param colorbar_label: Label of the colour scale; defaults to ``values``.
    :param theme: Existing theme.
    :return: Matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    table = frame.pivot_table(index=index, columns=columns, values=values, aggfunc=aggregate)
    figure, ax = plt.subplots(figsize=(max(6.0, 1.3 * table.shape[1] + 4.0), max(3.0, .5 * table.shape[0] + 2.0)),
                              dpi=resolved.figure_dpi)
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto",
                      cmap=colormap or resolved.probability_colormap, interpolation="nearest")
    ax.set_xticks(range(table.shape[1]), [short_label(name) for name in table.columns],
                  rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(table.shape[0]), [short_label(name) for name in table.index], fontsize=7)
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            cell = table.to_numpy(dtype=float)[row, column]
            if np.isfinite(cell):
                ax.text(column, row, value_format.format(cell), ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=ax, label=colorbar_label or values)
    ax.grid(False)
    ax.set(title=title)
    figure.tight_layout()
    return figure


def class_comparison(frame: pd.DataFrame, *, value_left: str = "value_left", value_right: str = "value_right",
                     name: str = "class_name", left_label: str = "left", right_label: str = "right",
                     count: int = 20, metric_label: str = "average precision", title: str = "", theme=None):
    """Show the classes where two conditions disagree most, with both values drawn.

    A macro mean cannot say whether one condition is uniformly better or wins on a
    handful of ions. Each row is one class; the two markers are the two conditions'
    values and the connecting segment is the difference, so its length is the effect
    and its direction the sign.

    :param frame: Paired per-class records for exactly one condition pair.
    :param value_left: Column with the first condition's per-class value.
    :param value_right: Column with the second condition's per-class value.
    :param name: Column naming the class.
    :param left_label: Legend name of the first condition.
    :param right_label: Legend name of the second condition.
    :param count: Number of classes shown at each end of the sorted difference.
    :param metric_label: Horizontal axis label naming the compared metric.
    :param title: Figure title.
    :param theme: Existing theme.
    :return: Matplotlib figure with the largest gains at the top.
    :rtype: matplotlib.figure.Figure
    """
    resolved = resolve_theme(theme)
    ordered = frame.assign(_difference=frame[value_left] - frame[value_right]).dropna(subset=["_difference"])
    ordered = ordered.sort_values("_difference")
    selected = pd.concat([ordered.head(count), ordered.tail(count)]).drop_duplicates(subset=[name])
    figure, ax = plt.subplots(figsize=(resolved.figure_size[0], max(3.0, .3 * len(selected) + 1.5)),
                              dpi=resolved.figure_dpi)
    left_color, right_color = _color(left_label, resolved), _color(right_label, resolved)
    for position, (_, row) in enumerate(selected.iterrows()):
        ax.plot([row[value_right], row[value_left]], [position, position],
                color=resolved.muted_text_color, linewidth=1.0, alpha=resolved.secondary_alpha, zorder=1)
        ax.scatter([row[value_left]], [position], color=left_color, s=26, zorder=3,
                   label=short_label(left_label) if position == 0 else None)
        ax.scatter([row[value_right]], [position], color=right_color, s=26, zorder=3,
                   label=short_label(right_label) if position == 0 else None)
    ax.set_yticks(range(len(selected)), [str(value) for value in selected[name]], fontsize=6)
    _finish(ax, xlabel=metric_label, title=title, legend=True, theme=resolved)
    ax.grid(resolved.grid_visible, axis="x", color=resolved.grid_color,
            alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
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


def spectrum_comparison(frame: pd.DataFrame, *, row_position: int, group: str = "label",
                        order: Sequence[str] | None = None, window: tuple[float, float] | None = None,
                        residual: bool = True, title: str = "", theme=None):
    """Overlay several conditions' reconstructions of one identical input spectrum.

    Comparing conditions on their own worst cases compares different spectra. This
    draws one shared spectrum that every model saw, so any visible difference is a
    difference between decoders rather than between inputs.

    :param frame: ``spectrum_cases`` records restricted to one split, containing the
        requested ``row_position`` for every compared condition.
    :param row_position: Identity of the shared spectrum inside its split.
    :param group: Semantic grouping column controlling colour and legend.
    :param order: Explicit legend order.
    :param window: Optional ``(low, high)`` m/z limits for a zoomed view; the full
        range is drawn when omitted.
    :param residual: Draw a second panel with reconstruction minus input.
    :param title: Figure title.
    :param theme: Existing theme.
    :return: Matplotlib figure with the intensity panel and an optional residual panel.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If the requested spectrum is absent.
    """
    resolved = resolve_theme(theme)
    selected = frame[frame.row_position == row_position]
    if selected.empty:
        raise ValueError(f"Spectrum row {row_position} is not present in the supplied cases.")
    panels = 2 if residual else 1
    figure, axes = plt.subplots(panels, 1, figsize=(13.0, 4.0 * panels), dpi=resolved.figure_dpi,
                                squeeze=False, sharex=True)
    top = axes[0, 0]
    for label in _group_order(selected, group, order):
        part = selected[selected[group] == label]
        for number, (_, run) in enumerate(part.groupby("model_id", sort=True)):
            run = run.sort_values("mz")
            top.plot(run.mz, run.reconstruction, color=_color(label, resolved),
                     linewidth=resolved.reconstruction_line_width, alpha=resolved.overlapping_signal_alpha,
                     label=short_label(label) if number == 0 else None, zorder=resolved.reconstruction_zorder)
            if residual:
                axes[1, 0].plot(run.mz, run.reconstruction - run.input, color=_color(label, resolved),
                                linewidth=resolved.residual_line_width, alpha=resolved.overlapping_signal_alpha,
                                label=short_label(label) if number == 0 else None)
    ## REMARK: The input is identical for every condition, so it is drawn once - and last,
    ## above the reconstruction band, because a reference the comparison lines cover is useless.
    reference = selected[selected[group] == selected[group].iloc[0]].sort_values("mz")
    top.plot(reference.mz, reference.input, color=resolved.input_color, label="input",
             linewidth=resolved.input_line_width * 1.6, zorder=resolved.annotation_zorder)
    if residual:
        axes[1, 0].axhline(0.0, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                           linewidth=resolved.reference_line_width)
        _finish(axes[1, 0], xlabel="m/z", ylabel="reconstruction - input", theme=resolved)
    if window is not None:
        top.set_xlim(*window)
    _finish(top, xlabel=None if residual else "m/z", ylabel="TIC-normalized intensity",
            title=title, legend=True, theme=resolved)
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
    return histogram_overlay(frame, title=title, theme=theme,
                             x_label=frame.metric.iloc[0] if len(frame) else "distance")
