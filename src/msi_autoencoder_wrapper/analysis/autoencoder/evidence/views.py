"""Plots for the evidence-threshold analysis, all driven by the shared theme.

Colour conventions follow the project's analysis rules:

- a swept continuous parameter (the threshold, the dilation radius, an intensity
  stratum) is coloured from a sequential colormap indexed across the sweep, never
  from the categorical model palette, which wraps silently past six entries;
- non-orderable categories (annotated / unannotated / background, an acquisition)
  take the categorical palette;
- a second categorical dimension on one axes is encoded as linestyle, never as a
  second colour axis;
- overlaid distributions are drawn as step outlines so nothing is hidden.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from ....visualization import resolve_theme
from ....visualization.theme import VisualizationTheme
from ....utils.logger import get_custom_logger
from .precompute import EvidenceStatistics

logger = get_custom_logger(__name__)

# State colours are fixed across every figure in the campaign: an entry the
# annotations call positive, one they say nothing about, and the whole spectral axis
# as the background reference.
STATE_LABELS = {
    "annotated": "annotated (P)",
    "unannotated": "unannotated (N or U)",
    "background": "every bin (reference)",
}


def _theme(theme: VisualizationTheme | str | None) -> VisualizationTheme:
    """Resolve a theme argument the same way every plotting helper does."""
    return resolve_theme(theme)


def _state_color(theme: VisualizationTheme, state: str) -> str:
    """Return the fixed colour of one evidence state."""
    if state == "annotated":
        return theme.ground_truth_color
    if state == "unannotated":
        return theme.residual_color
    return theme.input_color


def _sequential_colors(theme: VisualizationTheme, count: int) -> list[Any]:
    """Sample the theme's sequential colormap across a sweep of known length."""
    colormap = plt.get_cmap(theme.image_colormap)
    if count <= 1:
        return [colormap(0.5)]
    return [colormap(index / (count - 1)) for index in range(count)]


def plot_evidence_distributions(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    candidate_thresholds: Sequence[float] = (),
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Densities of the relative evidence for annotated, unannotated and all bins.

    The x-axis is the relative evidence :math:`r` on a logarithmic scale; the y-axis
    is the share of entries per histogram bucket. Blue is the annotated population,
    red the unannotated one and grey every bin of every spectrum, the reference for
    what an arbitrary position on the axis looks like. Vertical dotted lines mark the
    candidate thresholds: everything to their left becomes an operational negative,
    everything to their right stays uncertain. Entries with exactly zero evidence are
    reported in the legend rather than plotted, since a logarithmic axis cannot show
    them.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param candidate_thresholds: Thresholds to mark.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges

    series = {
        "annotated": statistics.annotated_relative[radius].sum(axis=0).astype(np.float64),
        "unannotated": statistics.unannotated_relative[radius].sum(axis=0).astype(np.float64),
        "background": statistics.background_relative.astype(np.float64),
    }
    for state, counts in series.items():
        total = float(counts.sum())
        if total <= 0:
            continue
        ## Bucket 0 holds exactly-zero evidence and has no position on a log axis.
        zero_share = counts[0] / total
        axis.step(
            edges, counts[1 : edges.size + 1] / total, where="pre",
            color=_state_color(resolved, state),
            alpha=resolved.distribution_edge_alpha,
            linewidth=resolved.distribution_line_width,
            label=f"{STATE_LABELS[state]} (zero evidence: {zero_share:.1%})",
        )
    for threshold in candidate_thresholds:
        axis.axvline(
            float(threshold), color=resolved.baseline_color, linestyle=":",
            linewidth=resolved.reference_line_width,
        )
    axis.set(
        xscale="log", yscale="log",
        xlabel=r"relative evidence $r = s_{ic} / \max_b x_{ib}$",
        ylabel="share of entries per bucket",
        title=f"Relative evidence by annotation state (bin_radius={bin_radius})",
    )
    axis.legend(loc=resolved.legend_location, frameon=resolved.legend_frame)
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_threshold_sweep(
    records: Sequence[dict[str, Any]],
    *,
    candidate_thresholds: Sequence[float] = (),
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Negative yield and positive contradiction against the relative threshold.

    The x-axis is the relative threshold on a logarithmic scale. The solid line is
    the share of unannotated entries converted into operational negatives, which is
    the quantity the objective gains from raising the threshold. The dashed line is
    the share of annotated entries whose own evidence falls at or below the same
    threshold, which is what raising it costs in label credibility. Both are read on
    the same left axis because both are shares of their own population. Dotted
    vertical lines mark the candidate thresholds.

    :param records: Output of ``threshold_analysis.state_yield_records``.
    :param candidate_thresholds: Thresholds to mark.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    thresholds = np.asarray([record["relative_threshold"] for record in records], dtype=np.float64)
    negative = np.asarray([record["negative_fraction_of_unannotated"] for record in records], dtype=np.float64)
    contradiction = np.asarray([record["positive_contradiction_rate"] for record in records], dtype=np.float64)

    axis.plot(thresholds, negative, color=resolved.residual_color, linewidth=resolved.line_width,
              label="unannotated entries turned into N")
    axis.plot(thresholds, contradiction, color=resolved.ground_truth_color, linestyle="--",
              linewidth=resolved.line_width, label="annotated entries below the threshold")
    for threshold in candidate_thresholds:
        axis.axvline(float(threshold), color=resolved.baseline_color, linestyle=":",
                     linewidth=resolved.reference_line_width)
    axis.set(
        xscale="log", xlabel=r"relative threshold $\beta$", ylabel="share of its own population",
        title="Negative yield against contradiction of annotated entries",
    )
    axis.legend(loc=resolved.legend_location, frameon=resolved.legend_frame)
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_radius_sweeps(
    records_by_radius: dict[int, Sequence[dict[str, Any]]],
    *,
    metric: str = "negative_fraction_of_unannotated",
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """One sweep curve per dilation radius, coloured across the radius sweep.

    The x-axis is the relative threshold on a logarithmic scale, the y-axis the
    selected sweep quantity. Colour encodes the dilation radius, sampled across the
    sequential colormap, darker meaning wider. Curves are directly comparable: they
    are the same measurement on the same population, differing only in how many
    neighbouring bins the evidence is allowed to draw on.

    :param records_by_radius: Sweep records keyed by dilation radius.
    :param metric: Record field to plot.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    radii = sorted(records_by_radius)
    colors = _sequential_colors(resolved, len(radii))
    for color, radius in zip(colors, radii):
        records = records_by_radius[radius]
        axis.plot(
            [record["relative_threshold"] for record in records],
            [record[metric] for record in records],
            color=color, linewidth=resolved.line_width, label=f"bin_radius={radius}",
        )
    axis.set(xscale="log", xlabel=r"relative threshold $\beta$", ylabel=metric.replace("_", " "),
             title=f"{metric.replace('_', ' ')} across dilation radii")
    axis.legend(loc=resolved.legend_location, frameon=resolved.legend_frame)
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_background_exceedance(
    records: Sequence[dict[str, Any]],
    *,
    candidate_thresholds: Sequence[float] = (),
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """How much of an average spectrum sits above each relative level.

    The x-axis is the relative level on a logarithmic scale, the y-axis the expected
    number of bins of one spectrum above it, also logarithmic. The curve is measured
    over every bin of every pixel, so it describes the spectral axis itself rather
    than any ion. Dotted vertical lines mark the candidate thresholds; where the
    curve is still near the full bin count, the threshold sits inside the noise floor
    and separates almost nothing.

    :param records: Output of ``threshold_analysis.background_exceedance_records``.
    :param candidate_thresholds: Thresholds to mark.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    axis.plot(
        [record["relative_threshold"] for record in records],
        [record["bins_above_per_spectrum"] for record in records],
        color=resolved.input_color, linewidth=resolved.line_width,
    )
    for threshold in candidate_thresholds:
        axis.axvline(float(threshold), color=resolved.baseline_color, linestyle=":",
                     linewidth=resolved.reference_line_width)
    axis.set(xscale="log", yscale="log", xlabel="relative intensity level",
             ylabel="bins per spectrum above the level",
             title="Spectral background: bins above a relative level")
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_group_stability(
    records: Sequence[dict[str, Any]],
    *,
    metric: str = "negative_fraction_of_unannotated",
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """One curve per source acquisition, to expose threshold-dependent split effects.

    The x-axis is the relative threshold on a logarithmic scale and the y-axis the
    selected quantity, computed inside one acquisition. Every acquisition gets its
    own line at low opacity; the spread between lines at a candidate threshold is the
    quantity of interest, since the train/validation/test split assigns whole
    acquisitions and a wide spread means the splits are trained on different label
    semantics.

    :param records: Output of ``threshold_analysis.group_state_records``.
    :param metric: Record field to plot.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    for name, rows in groups.items():
        rows = sorted(rows, key=lambda row: row["relative_threshold"])
        axis.plot(
            [row["relative_threshold"] for row in rows],
            [row[metric] for row in rows],
            color=resolved.input_color, alpha=resolved.overlapping_signal_alpha,
            linewidth=resolved.distribution_line_width,
        )
    axis.set(xscale="log", xlabel=r"relative threshold $\beta$", ylabel=metric.replace("_", " "),
             title=f"{metric.replace('_', ' ')} per source acquisition ({len(groups)} lines)")
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_peak_displacement(
    records: Sequence[dict[str, Any]],
    *,
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Displacement of the strongest bin in the annotation window, with its null.

    The x-axis is the displacement from the mapped bin, in bins. Blue bars are the
    share of annotated entries whose strongest bin sits at that displacement; grey
    bars are the same share over unannotated entries, which is the null expected when
    no real peak is present. An excess of blue over grey at displacement zero
    supports the mapping; an excess at a nonzero displacement is a calibration offset
    that a dilation radius must cover.

    :param records: Output of ``alignment_analysis.peak_displacement_records``.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    offsets = np.asarray([record["offset_bins"] for record in records], dtype=np.float64)
    axis.bar(offsets - 0.18, [record["annotated_share"] for record in records], width=0.34,
             color=_state_color(resolved, "annotated"), alpha=resolved.primary_alpha,
             label=STATE_LABELS["annotated"])
    axis.bar(offsets + 0.18, [record["unannotated_share"] for record in records], width=0.34,
             color=_state_color(resolved, "unannotated"), alpha=resolved.secondary_alpha,
             label=STATE_LABELS["unannotated"])
    axis.set(xlabel="displacement of the strongest bin (bins)", ylabel="share of informative entries",
             title="Annotation-to-bin alignment against its unannotated null")
    axis.set_xticks(offsets)
    axis.legend(loc=resolved.legend_location, frameon=resolved.legend_frame)
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_class_regimes(
    records: Sequence[dict[str, Any]],
    *,
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Every ion placed by its prevalence against the negatives the rule gives it.

    The x-axis is annotation prevalence, the y-axis the share of that ion's
    unannotated entries the threshold turns into operational negatives; both are
    per ion, at one fixed threshold, and each point is one ion. Colour marks the
    regime assigned by ``class_regime_records``. The top-left corner holds ions that
    are both rarely annotated and almost entirely negative — classes a P/N objective
    sees as pure negatives.

    :param records: Output of ``class_population_analysis.class_regime_records``.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    regimes = sorted({record["regime"] for record in records})
    for index, regime in enumerate(regimes):
        selected = [record for record in records if record["regime"] == regime]
        axis.scatter(
            [record["prevalence"] for record in selected],
            [record["negative_fraction_of_unannotated"] for record in selected],
            s=resolved.marker_size ** 2 / 3, alpha=resolved.marker_alpha,
            color=resolved.color_for_model(regime, index),
            edgecolors="none", label=f"{regime} ({len(selected)})",
        )
    axis.set(xscale="log", xlabel="annotation prevalence", ylabel="unannotated entries turned into N",
             title="Ion regimes at the selected threshold")
    axis.legend(loc=resolved.legend_location, frameon=resolved.legend_frame)
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_mz_profile(
    records: Sequence[dict[str, Any]],
    *,
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Annotation density and negative yield across the configured m/z window.

    The x-axis is m/z over the configured 200-900 window. The blue step curve, on the
    left axis, is the mean annotation prevalence of the ions inside each window; the
    red step curve, on the right axis, is the share of their unannotated entries the
    threshold turns into negatives. Two axes are used because the quantities have
    unrelated scales; both are shares.

    :param records: Output of ``class_population_analysis.mz_window_records``.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The left axes.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    centres = [(record["mz_lower"] + record["mz_upper"]) / 2 for record in records]
    axis.step(centres, [record["mean_prevalence"] for record in records], where="mid",
              color=_state_color(resolved, "annotated"), linewidth=resolved.line_width,
              label="mean annotation prevalence")
    axis.set(xlabel="m/z", ylabel="mean annotation prevalence", title="Annotation and evidence across the m/z window")
    right = axis.twinx()
    right.step(centres, [record["negative_fraction_of_unannotated"] for record in records], where="mid",
               color=_state_color(resolved, "unannotated"), linestyle="--",
               linewidth=resolved.line_width, label="unannotated entries turned into N")
    right.set_ylabel("unannotated entries turned into N")
    right.grid(False)
    handles = axis.get_lines() + right.get_lines()
    axis.legend(handles, [line.get_label() for line in handles],
                loc=resolved.legend_location, frameon=resolved.legend_frame)
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_prevalence_curve(
    records: Sequence[dict[str, Any]],
    *,
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Every ion's annotation prevalence, ordered from most to least prevalent.

    The x-axis is the ion's rank by prevalence and the y-axis its prevalence on a
    logarithmic scale — the share of labelled pixels in which it is annotated. Each
    point is one ion. The shape of the curve, not any individual point, is the
    result: a long flat tail means most classes are annotated in a negligible share
    of pixels, so a pooled metric over classes is dominated by classes that are
    almost never positive.

    :param records: Output of ``class_population_analysis.class_prevalence_records``.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    prevalence = np.sort(
        np.asarray([record["prevalence"] for record in records], dtype=np.float64)
    )[::-1]  # (C,)
    axis.plot(np.arange(1, prevalence.size + 1), prevalence, color=resolved.ground_truth_color,
              marker="o", markersize=resolved.marker_size / 2, linestyle="-",
              linewidth=resolved.distribution_line_width, alpha=resolved.primary_alpha)
    axis.set(yscale="log", xlabel="ion rank by prevalence", ylabel="annotation prevalence",
             title=f"Annotation prevalence of {prevalence.size} ions")
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis


def plot_separation_summary(
    records: Sequence[dict[str, Any]],
    *,
    ax: Axes | None = None,
    theme: VisualizationTheme | str | None = None,
) -> Axes:
    """Per-ion evidence-annotation agreement against how often the ion is annotated.

    The x-axis is annotation prevalence on a logarithmic scale, the y-axis the rank
    statistic :math:`\\mathrm{AUC}_c` from ``evidence_separation_records``; each point
    is one ion. The horizontal reference line at ``0.5`` is the no-information level:
    an ion sitting on it shows no intensity difference at its bin between pixels
    where it is annotated and pixels where it is not, so no threshold can produce
    meaningful negatives for it. Points below the line indicate annotated pixels
    carrying *less* signal, which is a mapping or calibration symptom rather than a
    chemical one.

    :param records: Output of ``threshold_analysis.evidence_separation_records``.
    :param ax: Target axes; created when absent.
    :param theme: Theme override.
    :return: The axes drawn on.
    :rtype: matplotlib.axes.Axes
    """
    resolved = _theme(theme)
    axis = ax if ax is not None else plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)[1]
    prevalence = np.asarray(
        [
            record["positive_entries"] / max(record["positive_entries"] + record["unannotated_entries"], 1.0)
            for record in records
        ],
        dtype=np.float64,
    )  # (C,)
    auc = np.asarray([record["evidence_auc"] for record in records], dtype=np.float64)  # (C,)
    axis.scatter(prevalence, auc, s=resolved.marker_size ** 2 / 3, alpha=resolved.marker_alpha,
                 color=resolved.ground_truth_color, edgecolors="none")
    axis.axhline(0.5, color=resolved.baseline_color, linestyle=resolved.baseline_line_style,
                 linewidth=resolved.reference_line_width)
    axis.set(xscale="log", xlabel="annotation prevalence", ylabel=r"evidence $\mathrm{AUC}_c$",
             title="Agreement between annotation and local evidence, per ion")
    axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    return axis
