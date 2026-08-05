"""Sub-range dependency: does the optimal Δm / inverse-binner parameter depend on
where in the m/z axis you are?

Repeats the forward sweep (:mod:`binner_forward_analysis`) and the inverse-binner sweep
(:mod:`inverse_binner_analysis`) on restricted, non-overlapping m/z windows, each a
genuinely separate, recalibrated binning run (``BinningPrecompute.forward``/``.inverse``
with explicit ``x_min``/``x_max``) — never the global run merely filtered to a range,
since that would keep the same forward/inverse-binner parameters as the whole-axis run
and hide any window-dependence. The spectrum *sample* itself is never redrawn per
window (same seeded sample throughout, per ``methodology.md`` §5) — only the m/z axis
each spectrum is binned/inverse-binned over changes. See ``methodology.md`` §6 step 3.

Global (whole-range) and local (per-window) results are always returned together in one
table/plot, never as a replacement for one another — the point is to compare them.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from .binner_forward_analysis import DEFAULT_NORMALIZATIONS, forward_sweep_records
from .inverse_binner_analysis import MAXIMIZE_METRICS, inverse_sweep_records
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)

GLOBAL_REGION_LABEL = "global"


def build_windows(mass_range: tuple[float, float], width: float = 100.0) -> list[tuple[float, float]]:
    """Build non-overlapping ``[start, end)`` windows of ``width`` spanning ``mass_range``.

    Windows are aligned to multiples of ``width`` (e.g. width=100 over (72, 1172) gives
    (0,100),(100,200),...,(1100,1200), clipped to the actual range at the two ends) so
    window boundaries are stable/predictable across different sample ranges, not just
    "however many fit starting from the exact minimum."

    Whether a boundary value itself (e.g. exactly 300.0 between the 200-300 and 300-400
    windows) is included in the lower or the upper window does not matter for any result
    computed from these windows: real m/z values landing exactly on an integer boundary
    have probability zero (a measure-zero set on a continuous axis), so the half-open
    ``[start, end)`` convention here is just a deterministic tie-break, not a modeling
    choice that could bias the analysis either way.
    """
    lo, hi = mass_range
    if width <= 0 or hi <= lo:
        return []
    start = float(np.floor(lo / width) * width)
    windows: list[tuple[float, float]] = []
    edge = start
    while edge < hi:
        windows.append((max(lo, edge), min(hi, edge + width)))
        edge += width
    return windows


def region_forward_sweep_records(
    precompute: BinningPrecompute,
    delta_m_grid: Sequence[float],
    window_width: float = 100.0,
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    normalizations: Sequence[str] = DEFAULT_NORMALIZATIONS,
    collision_min_relative_height: float = 0.0,
) -> list[dict[str, Any]]:
    """:func:`binner_forward_analysis.forward_sweep_records`, repeated per m/z window
    plus once for the whole (global) range, every record tagged with its ``region``.

    :return: Long-form records — same schema as ``forward_sweep_records`` plus
        ``region`` (``"global"`` or ``"{start:g}-{end:g}"``), ``region_x_min``,
        ``region_x_max`` (numeric, for plotting error/optimum vs region midpoint).
    """
    windows = build_windows(precompute.global_mass_range, window_width)
    regions = [(GLOBAL_REGION_LABEL, precompute.global_mass_range)] + [(f"{lo:g}-{hi:g}", (lo, hi)) for lo, hi in windows]
    logger.info("START region forward sweep: %s regions (1 global + %s windows of width %s).", len(regions), len(windows), window_width)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for region_label, (x_min, x_max) in regions:
        region_started = time.perf_counter()
        region_records = forward_sweep_records(precompute, delta_m_grid, x_min, x_max, tolerance, tolerance_unit, normalizations, collision_min_relative_height)
        for record in region_records:
            record["region"] = region_label
            record["region_x_min"], record["region_x_max"] = float(x_min), float(x_max)
        records.extend(region_records)
        logger.info("DONE region=%s forward sweep in %.2fs.", region_label, time.perf_counter() - region_started)
    logger.info("DONE region forward sweep: %s records in %.2fs.", len(records), time.perf_counter() - started)
    return records


def region_inverse_sweep_records(
    precompute: BinningPrecompute,
    delta_m: float,
    method_grid: Sequence[Mapping[str, Any]],
    window_width: float = 100.0,
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    normalizations: Sequence[str] = DEFAULT_NORMALIZATIONS,
) -> list[dict[str, Any]]:
    """:func:`inverse_binner_analysis.inverse_sweep_records`, repeated per m/z window
    plus once for the whole (global) range, every record tagged with its ``region``.

    Same ``region``/``region_x_min``/``region_x_max`` tagging as
    :func:`region_forward_sweep_records`.
    """
    windows = build_windows(precompute.global_mass_range, window_width)
    regions = [(GLOBAL_REGION_LABEL, precompute.global_mass_range)] + [(f"{lo:g}-{hi:g}", (lo, hi)) for lo, hi in windows]
    logger.info(
        "START region inverse sweep: %s regions x %s grid points at delta_m=%s.",
        len(regions), len(method_grid), delta_m,
    )
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for region_label, (x_min, x_max) in regions:
        region_started = time.perf_counter()
        region_records = inverse_sweep_records(precompute, delta_m, method_grid, x_min, x_max, tolerance, tolerance_unit, normalizations)
        for record in region_records:
            record["region"] = region_label
            record["region_x_min"], record["region_x_max"] = float(x_min), float(x_max)
        records.extend(region_records)
        logger.info("DONE region=%s inverse sweep in %.2fs.", region_label, time.perf_counter() - region_started)
    logger.info("DONE region inverse sweep: %s records in %.2fs.", len(records), time.perf_counter() - started)
    return records


def optimal_delta_m_by_region(
    region_forward_records: Sequence[Mapping[str, Any]],
    metric: str,
    normalization: str = "raw",
    statistic: str = "median",
) -> list[dict[str, Any]]:
    """For every region (including ``"global"``), find the ``delta_m`` that
    minimizes/maximizes ``metric`` (direction from :data:`MAXIMIZE_METRICS`).

    This is the direct answer to "does the optimal forward bin step depend on m/z
    location" — one row per region with the winning ``delta_m`` and its score, ready to
    plot against ``region_x_min``/``region_x_max`` midpoint.

    Concretely, with the default ``window_width=100`` windows built by
    :func:`build_windows` (e.g. 100-200, 200-300, 300-400, ...): for each window
    independently, take every ``delta_m`` in the grid that was swept for that window
    (via :func:`region_forward_sweep_records`), compute the chosen ``statistic`` (median
    by default — robust to a handful of degenerate/near-empty sampled spectra skewing a
    mean) of ``metric`` across the sampled spectra restricted to that window, and keep
    whichever ``delta_m`` gives the best score. Repeated per window, plus once for the
    whole spectrum as the ``"global"`` row.

    Groups per (region, delta_m) directly from the long-form records rather than
    reusing :func:`summarize_forward_sweep` — that helper groups by (delta_m,
    normalization, metric) only, which would merge different regions that happen to
    share a ``delta_m`` value.
    """
    region_midpoints = {record["region"]: (record["region_x_min"], record["region_x_max"]) for record in region_forward_records}
    grouped: dict[tuple[str, float], list[float]] = {}
    for record in region_forward_records:
        if record["metric"] != metric or record["normalization"] != normalization:
            continue
        grouped.setdefault((record["region"], record["delta_m"]), []).append(record["value"])
    per_region: dict[str, list[tuple[float, float]]] = {}
    for (region, delta_m), values in grouped.items():
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            continue
        score = float(np.median(finite)) if statistic == "median" else float(np.mean(finite))
        per_region.setdefault(region, []).append((delta_m, score))
    reverse = metric in MAXIMIZE_METRICS
    results: list[dict[str, Any]] = []
    for region, points in per_region.items():
        best_delta_m, best_score = sorted(points, key=lambda pair: pair[1], reverse=reverse)[0]
        x_min, x_max = region_midpoints[region]
        results.append({
            "region": region, "region_x_min": x_min, "region_x_max": x_max, "region_midpoint": (x_min + x_max) / 2,
            "optimal_delta_m": best_delta_m, "score": best_score, "metric": metric,
        })
    results.sort(key=lambda record: record["region_midpoint"])
    return results


def optimal_grid_point_by_region(
    region_inverse_records: Sequence[Mapping[str, Any]],
    method: str,
    metric: str,
    comparison: str,
    normalization: str = "raw",
    statistic: str = "median",
) -> list[dict[str, Any]]:
    """Like :func:`optimal_delta_m_by_region` but for one inverse-binner method's own
    parameter grid: for every region, which grid point (``label``) of ``method``
    minimizes/maximizes ``metric``.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    region_midpoints: dict[str, tuple[float, float]] = {}
    for record in region_inverse_records:
        if record["method"] != method or record["metric"] != metric or record["comparison"] != comparison or record["normalization"] != normalization:
            continue
        grouped.setdefault((record["region"], record["label"]), []).append(record["value"])
        region_midpoints[record["region"]] = (record["region_x_min"], record["region_x_max"])
    per_region: dict[str, list[tuple[str, float]]] = {}
    for (region, label), values in grouped.items():
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            continue
        score = float(np.median(finite)) if statistic == "median" else float(np.mean(finite))
        per_region.setdefault(region, []).append((label, score))
    reverse = metric in MAXIMIZE_METRICS
    results: list[dict[str, Any]] = []
    for region, points in per_region.items():
        best_label, best_score = sorted(points, key=lambda pair: pair[1], reverse=reverse)[0]
        x_min, x_max = region_midpoints[region]
        results.append({
            "region": region, "region_x_min": x_min, "region_x_max": x_max, "region_midpoint": (x_min + x_max) / 2,
            "method": method, "optimal_label": best_label, "score": best_score, "metric": metric,
        })
    results.sort(key=lambda record: record["region_midpoint"])
    return results


def plot_optimal_parameter_by_region(
    optimal_by_region: Sequence[Mapping[str, Any]],
    y: str = "optimal_delta_m",
    ax=None,
    theme: VisualizationTheme | str | None = None,
):
    """Plot the winning parameter value (``optimal_delta_m`` or, for
    :func:`optimal_grid_point_by_region` output, use the numeric parameter you swept —
    ``optimal_label`` is categorical and plotted as-is on a text y-axis) against region
    midpoint m/z. The ``"global"`` region (x_min/x_max spanning the whole sample) is
    excluded from the x-axis line and shown instead as a horizontal reference line, since
    it is not a point on the m/z axis.
    """
    import matplotlib.pyplot as plt

    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    local = [record for record in optimal_by_region if record["region"] != GLOBAL_REGION_LABEL]
    global_records = [record for record in optimal_by_region if record["region"] == GLOBAL_REGION_LABEL]
    local.sort(key=lambda record: record["region_midpoint"])
    xs = [record["region_midpoint"] for record in local]
    ys = [record[y] for record in local]
    if y == "optimal_label":
        ax.plot(xs, ys, marker="o", color=resolved.color_for_model("per-region"), alpha=resolved.primary_alpha, label="per-region optimum")
        ax.tick_params(axis="y", labelrotation=0)
    else:
        ax.plot(xs, ys, marker="o", color=resolved.color_for_model("per-region"), alpha=resolved.primary_alpha, label="per-region optimum")
    if global_records:
        global_value = global_records[0][y]
        if y != "optimal_label":
            ax.axhline(global_value, color=resolved.baseline_color, linestyle=resolved.baseline_line_style, linewidth=resolved.reference_line_width, label=f"global optimum ({global_value:g})")
    ax.set(xlabel="region midpoint m/z", ylabel=y)
    ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    ax.legend(fontsize=resolved.tick_font_size, loc=resolved.legend_location, frameon=resolved.legend_frame)
    figure.tight_layout()
    return figure, ax
