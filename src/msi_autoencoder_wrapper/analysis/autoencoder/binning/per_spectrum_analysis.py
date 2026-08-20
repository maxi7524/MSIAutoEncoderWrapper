"""Per-spectrum breakdown: does a metric's value track transformation quality, or just
spectrum complexity/scale?

Groups sampled spectra into quantile buckets by intrinsic raw-spectrum properties (TIC,
peak count, dominant-peak fraction, max intensity — computed once from raw data, not
from any comparison) and shows a chosen metric's distribution per bucket. A metric that
varies mostly with, say, TIC rather than with which method/parameter produced it is
telling you about spectrum scale, not about the transformation. Peak-collision rate is
deliberately not one of these grouping properties — it already exists as a metric in
``forward_sweep_records``/``region_dependency_analysis`` output (it depends on a chosen
Δm, so it is not an intrinsic raw-spectrum property); correlate against it directly via
that metric's own column instead. See ``methodology.md`` §6 step 5.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)

# Intrinsic raw-spectrum properties available for bucketing (see spectrum_properties).
PROPERTY_NAMES: tuple[str, ...] = ("tic", "peak_count", "max_intensity", "dominant_peak_fraction", "mz_span")


def spectrum_properties(precompute: BinningPrecompute) -> dict[int, dict[str, float]]:
    """Compute intrinsic properties of every sampled spectrum's raw data.

    :return: ``{spectrum_id: {"tic", "peak_count", "max_intensity",
        "dominant_peak_fraction", "mz_span"}}``. All-zero spectra get
        ``dominant_peak_fraction = 0.0`` (not NaN) so they sort into the lowest bucket
        rather than being dropped.
    """
    properties: dict[int, dict[str, float]] = {}
    for spectrum_id in precompute.spectrum_ids:
        mz, intensity = precompute.raw(spectrum_id)
        total = float(np.sum(intensity))
        maximum = float(np.max(intensity, initial=0.0))
        properties[int(spectrum_id)] = {
            "tic": total,
            "peak_count": int(np.count_nonzero(intensity > 0)),
            "max_intensity": maximum,
            "dominant_peak_fraction": (maximum / total) if total > 0 else 0.0,
            "mz_span": float(np.ptp(mz)) if mz.size else 0.0,
        }
    return properties


def bucket_by_property(properties: Mapping[int, Mapping[str, float]], property_name: str, n_buckets: int = 4) -> dict[int, str]:
    """Assign every spectrum to a quantile bucket of ``properties[...][property_name]``.

    :return: ``{spectrum_id: "Q1 (low)" | "Q2" | ... | "Qn (high)"}``. Bucket edges are
        computed from the property's own quantiles across the sampled spectra (not fixed
        thresholds), so bucket sizes stay roughly balanced regardless of the property's
        actual distribution shape.
    """
    ids = list(properties.keys())
    values = np.asarray([properties[spectrum_id][property_name] for spectrum_id in ids], dtype=np.float32)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_buckets + 1)))
    if edges.size < 2:
        return {spectrum_id: "Q1 (low)" for spectrum_id in ids}
    bucket_count = edges.size - 1
    indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bucket_count - 1)
    labels = [f"Q{index + 1}" + (" (low)" if index == 0 else " (high)" if index == bucket_count - 1 else "") for index in range(bucket_count)]
    return {spectrum_id: labels[index] for spectrum_id, index in zip(ids, indices)}


def plot_metric_by_property_bucket(
    records: Sequence[Mapping[str, Any]],
    properties: Mapping[int, Mapping[str, float]],
    property_name: str,
    metric: str,
    label: str,
    comparison: str,
    normalization: str = "raw",
    n_buckets: int = 4,
    ax=None,
    theme: VisualizationTheme | str | None = None,
):
    """Boxplot of ``metric`` (for one grid-point ``label``/``comparison``/
    ``normalization``) across quantile buckets of ``property_name``.

    Requires a single ``label`` (mixing several methods/parameter settings on one
    property-bucket plot would conflate "this method is worse" with "this bucket is
    harder," defeating the purpose). Run once per method/grid point you want to check.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    buckets = bucket_by_property(properties, property_name, n_buckets)
    value_by_spectrum = {
        record["spectrum_id"]: record["value"]
        for record in records
        if record["label"] == label and record["metric"] == metric and record["comparison"] == comparison and record["normalization"] == normalization
    }
    ordered_bucket_labels = sorted(set(buckets.values()), key=lambda text: int(text.split()[0][1:]))
    grouped = [
        [value_by_spectrum[spectrum_id] for spectrum_id, bucket in buckets.items() if bucket == bucket_label and spectrum_id in value_by_spectrum and np.isfinite(value_by_spectrum[spectrum_id])]
        for bucket_label in ordered_bucket_labels
    ]
    color = resolved.color_for_model(label)
    ax.boxplot(
        grouped,
        tick_labels=ordered_bucket_labels,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": resolved.line_width, "alpha": resolved.secondary_alpha},
        whiskerprops={"color": color, "linewidth": resolved.line_width, "alpha": resolved.primary_alpha},
        capprops={"color": color, "linewidth": resolved.line_width, "alpha": resolved.primary_alpha},
        medianprops={"color": resolved.text_color, "linewidth": resolved.line_width, "alpha": resolved.primary_alpha},
        meanprops={"markeredgecolor": resolved.panel_color, "markerfacecolor": color, "markersize": resolved.marker_size, "markeredgewidth": resolved.marker_edge_width, "alpha": resolved.marker_alpha},
        flierprops={"marker": "o", "markerfacecolor": color, "markeredgecolor": resolved.panel_color, "markersize": resolved.marker_size, "markeredgewidth": resolved.marker_edge_width, "alpha": resolved.marker_alpha},
    )
    ax.set(xlabel=f"{property_name} bucket", ylabel=metric)
    ax.set_title(f"{metric} by {property_name} bucket | {label} | {comparison}/{normalization}", loc=resolved.title_location)
    ax.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
    figure.tight_layout()
    return figure, ax


def plot_metric_by_property_bucket_multi(
    records: Sequence[Mapping[str, Any]],
    properties: Mapping[int, Mapping[str, float]],
    property_name: str,
    metric: str,
    labels: Sequence[str],
    comparison: str,
    normalization: str = "raw",
    n_buckets: int = 4,
    ax=None,
    theme: VisualizationTheme | str | None = None,
):
    """:func:`plot_metric_by_property_bucket` for several ``labels`` at once, grouped
    side by side within each bucket instead of one label per figure.

    Unlike the single-``label`` version, this does intentionally mix methods on one
    plot — that is the point here: comparing how *different* configurations degrade
    across the *same* buckets (color = label, same convention as every other multi-
    config plot in this package), not just one config's own bucket profile in
    isolation. Each label's box is colored by ``theme.color_for_model`` and offset
    within its bucket so boxes never overlap.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    buckets = bucket_by_property(properties, property_name, n_buckets)
    ordered_bucket_labels = sorted(set(buckets.values()), key=lambda text: int(text.split()[0][1:]))
    bucket_count = len(ordered_bucket_labels)
    label_count = len(labels)
    width = 0.8 / max(label_count, 1)
    for label_index, label in enumerate(labels):
        value_by_spectrum = {
            record["spectrum_id"]: record["value"]
            for record in records
            if record["label"] == label and record["metric"] == metric and record["comparison"] == comparison and record["normalization"] == normalization
        }
        grouped = [
            [value_by_spectrum[spectrum_id] for spectrum_id, bucket in buckets.items() if bucket == bucket_label and spectrum_id in value_by_spectrum and np.isfinite(value_by_spectrum[spectrum_id])]
            for bucket_label in ordered_bucket_labels
        ]
        offset = (label_index - (label_count - 1) / 2) * width
        positions = [bucket_index + 1 + offset for bucket_index in range(bucket_count)]
        color = resolved.color_for_model(label, label_index)
        boxes = ax.boxplot(
            grouped,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showmeans=True,
            boxprops={"edgecolor": color, "linewidth": resolved.line_width},
            whiskerprops={"color": color, "linewidth": resolved.line_width, "alpha": resolved.primary_alpha},
            capprops={"color": color, "linewidth": resolved.line_width, "alpha": resolved.primary_alpha},
            medianprops={"color": resolved.text_color, "linewidth": resolved.line_width, "alpha": resolved.primary_alpha},
            meanprops={"markeredgecolor": resolved.panel_color, "markerfacecolor": color, "markersize": resolved.marker_size, "markeredgewidth": resolved.marker_edge_width, "alpha": resolved.marker_alpha},
            flierprops={"marker": "o", "markerfacecolor": color, "markeredgecolor": resolved.panel_color, "markersize": resolved.marker_size, "markeredgewidth": resolved.marker_edge_width, "alpha": resolved.marker_alpha},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color); box.set_alpha(resolved.secondary_alpha)
        ax.plot([], [], color=color, linewidth=resolved.line_width, alpha=resolved.primary_alpha, label=label)  # legend proxy (boxplot artists don't auto-legend)
    ax.set_xticks(range(1, bucket_count + 1))
    ax.set_xticklabels(ordered_bucket_labels)
    ax.set(xlabel=f"{property_name} bucket", ylabel=metric)
    ax.set_title(f"{metric} by {property_name} bucket | {comparison}/{normalization}", loc=resolved.title_location)
    ax.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
    ax.legend(fontsize=resolved.legend_font_size, loc=resolved.legend_location, ncols=resolved.legend_columns, frameon=resolved.legend_frame)
    figure.tight_layout()
    return figure, ax
