"""Forward-binning resolution sweep: B(X) vs X only, no inverse binner involved.

Answers a single question: as the bin step grows past the dataset's native
resolution, how does each metric's *distribution* across sampled spectra behave, and
does that depend on which m/z sub-range is being binned? This is decided once, first,
globally, because every downstream inverse-binner analysis runs on top of one chosen
bin step. See ``methodology.md`` §6 step 1.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
from tqdm.auto import tqdm

from ....metrics import match_spectral_points, peak_collision_rate, spectral_point_metrics
from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.metrics import plot_metric_distribution, plot_metric_tradeoff
from ..binning.metrics import normalize_intensity, summarize_metric
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)

DEFAULT_NORMALIZATIONS: tuple[str, ...] = ("raw", "tic", "max")


def sequential_colors(theme: VisualizationTheme, count: int) -> list[str]:
    """Sample ``count`` colors from the theme's sequential colormap, evenly spaced.

    For *continuous* swept parameters (e.g. ``delta_m``) — never use
    ``theme.model_palette``/``color_for_model`` for this: the palette has 6 fixed
    categorical entries and silently wraps around for any grid larger than 6, so two
    different swept values end up rendered in the same color. ``theme.model_palette`` is
    for categorical, non-orderable groups (e.g. different inverse-binner methods); a
    continuous colormap correctly encodes that swept values have an order.
    """
    if count <= 0:
        return []
    colormap = plt.get_cmap(theme.image_colormap)
    if count == 1:
        return [to_hex(colormap(0.5))]
    return [to_hex(colormap(index / (count - 1))) for index in range(count)]


def rescale_match(base_match, normalized_reference: np.ndarray, normalized_candidate: np.ndarray):
    """Re-sum matched intensities from an already-computed match under a new
    normalization, without re-running the O(n) point search.

    ``match_spectral_points(..., "one_to_one")`` decides *which* points pair up using
    only m/z positions and the tolerance window — never intensity — so the pairing
    (``matched_reference_indices``/``matched_candidate_indices``/``candidate_groups``/
    ``mz_errors_*``/``unmatched_*_indices``) is identical for any positive, order-
    preserving rescaling of the intensities (raw/tic/max all qualify). Only the summed
    intensities *at* the matched points depend on which scaling was used, so this
    function only recomputes those two fields on top of an already-computed match,
    instead of re-running the expensive search for every normalization.

    :param base_match: A :class:`SpectralPointMatch` computed on any (typically raw)
        intensities — its index/position fields are reused as-is.
    :param normalized_reference: Reference intensities under the *new* normalization,
        same length/order as the array ``base_match`` was computed from.
    :param normalized_candidate: Candidate intensities under the *new* normalization.
    :return: A new :class:`SpectralPointMatch` with the same pairing, resummed
        ``matched_reference_intensity``/``matched_candidate_intensity``.
    """
    matched_reference_intensity = (
        normalized_reference[base_match.matched_reference_indices] if base_match.matched_reference_indices.size else np.asarray([], dtype=np.float32)
    )
    matched_candidate_intensity = np.asarray(
        [float(np.sum(normalized_candidate[group])) for group in base_match.candidate_groups], dtype=np.float32
    ) if base_match.candidate_groups else np.asarray([], dtype=np.float32)
    return replace(base_match, matched_reference_intensity=matched_reference_intensity, matched_candidate_intensity=matched_candidate_intensity)


def crop_spectrum(mz: np.ndarray, intensity: np.ndarray, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a sparse (m/z, intensity) pair to ``[x_min, x_max]``.

    Required whenever a forward binner is built over a restricted window (region
    dependency analysis, see ``region_dependency_analysis.py``): the binner only
    aggregates raw points that fall inside its own ``[x_min, x_max]`` (points outside
    are silently dropped by ``scipy.stats.binned_statistic``), so comparing its output
    against an *uncropped* raw spectrum would match a partial-range candidate against a
    full-range reference — every raw peak outside the window would show up as
    "unmatched," corrupting every metric. Both sides of every comparison in this package
    must span the same window; this function is how the raw side gets there.
    """
    mask = (mz >= x_min) & (mz <= x_max)
    return mz[mask], intensity[mask]


def forward_sweep_records(
    precompute: BinningPrecompute,
    delta_m_grid: Sequence[float],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    normalizations: Sequence[str] = DEFAULT_NORMALIZATIONS,
    collision_min_relative_height: float = 0.0,
) -> list[dict[str, Any]]:
    """Long-form records of every metric, for every sampled spectrum, for every ``delta_m``.

    For each ``delta_m`` in ``delta_m_grid``: forward-bins every sampled spectrum via
    ``precompute.forward`` (cached), matches ``B(X)`` against the raw ``X`` with
    ``match_spectral_points(..., "one_to_one")``, and computes every field from
    ``metrics.spectral_points.spectral_point_metrics`` plus ``peak_collision_rate``.

    :return: One row per (spectrum, delta_m, normalization, metric) — long-form, ready
        to filter/group with plain list comprehensions or hand to ``pandas.DataFrame``.
        ``tic_relative_error`` is only meaningful under ``raw`` (elsewhere it is ~0 by
        construction of the normalization) and is reported as NaN for ``tic``/``max``.
        ``peak_collision_rate`` does not depend on normalization (it compares point
        positions, not scaled intensity) and is computed once per (spectrum, delta_m),
        not once per normalization.

    Performance note: matching is an O(points) Python loop per spectrum
    (``metrics.spectral_points`` is a reused global module, not reimplemented here). It
    is computed once per (spectrum, delta_m) — not once per normalization — via
    :func:`rescale_match`, a ~3x reduction versus naively re-matching per normalization.
    """
    logger.info(
        "START forward sweep: %s spectra x %s delta_m values x %s normalizations.",
        precompute.spectrum_ids.size, len(delta_m_grid), len(normalizations),
    )
    started = time.perf_counter()
    epsilon = np.finfo(float).eps
    records: list[dict[str, Any]] = []
    for delta_m in delta_m_grid:
        binner, forward_cache = precompute.forward(delta_m, x_min, x_max)
        grid_mz = np.asarray(binner.GetXAxis(), dtype=np.float32)
        dimension = binner.GetXAxisDepth()
        delta_m_started = time.perf_counter()
        for spectrum_id in tqdm(precompute.spectrum_ids, desc=f"Matching Δm={delta_m:g}", unit="spectrum"):
            raw_mz, raw_y = crop_spectrum(*precompute.raw(spectrum_id), binner.x_min, binner.x_max)
            grid_y = forward_cache[int(spectrum_id)]

            ## Collision rate: position-only, independent of normalization.
            collision_rate = (
                peak_collision_rate(raw_mz, raw_y, grid_mz, grid_y, tolerance, tolerance_unit, collision_min_relative_height)
                if float(np.sum(raw_y)) > 0 else 0.0
            )

            ## Match once (raw intensities), then re-sum per normalization (see rescale_match).
            base_match = match_spectral_points(raw_mz, raw_y, grid_mz, grid_y, tolerance, tolerance_unit, "one_to_one")
            for normalization in normalizations:
                normalized_raw = normalize_intensity(raw_y, normalization)
                normalized_grid = normalize_intensity(grid_y, normalization)
                match = rescale_match(base_match, normalized_raw, normalized_grid)
                values = spectral_point_metrics(raw_mz, normalized_raw, grid_mz, normalized_grid, match)
                values["tic_relative_error"] = (
                    abs(float(np.sum(raw_y)) - float(np.sum(grid_y))) / (abs(float(np.sum(raw_y))) + epsilon)
                    if normalization == "raw" else np.nan
                )
                values["peak_collision_rate"] = collision_rate
                for metric, value in values.items():
                    records.append({
                        "spectrum_id": int(spectrum_id), "delta_m": float(delta_m), "normalization": normalization,
                        "metric": metric, "value": float(value), "feature_dimension": int(dimension),
                        "x_min": float(binner.x_min), "x_max": float(binner.x_max),
                    })
        logger.info("DONE matching Δm=%s in %.2fs.", delta_m, time.perf_counter() - delta_m_started)
    logger.info("DONE forward sweep: %s records in %.2fs.", len(records), time.perf_counter() - started)
    return records


def summarize_forward_sweep(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group :func:`forward_sweep_records` output by (delta_m, normalization, metric).

    :return: One row per (delta_m, normalization, metric) with
        ``metrics.autoencoder.binning.metrics.summarize_metric``'s summary statistics
        (count/mean/std/min/q25/median/q75/q90/q95/q99/max) plus ``feature_dimension``
        (the forward binner's output size at that ``delta_m``, constant within a group).
    """
    grouped: dict[tuple[float, str, str], list[float]] = {}
    dimension_by_delta_m: dict[float, int] = {}
    for record in records:
        key = (record["delta_m"], record["normalization"], record["metric"])
        grouped.setdefault(key, []).append(record["value"])
        dimension_by_delta_m[record["delta_m"]] = record["feature_dimension"]
    return [
        {"delta_m": key[0], "feature_dimension": dimension_by_delta_m[key[0]], "normalization": key[1], "metric": key[2], **summarize_metric(values), "_values": [float(value) for value in values if np.isfinite(value)]}
        for key, values in grouped.items()
    ]


def plot_metric_by_delta_m(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    normalization: str = "raw",
    kind: str = "histogram",
    bins: int = 40,
    ax=None,
    theme: VisualizationTheme | str | None = None,
):
    """One plot for one metric: every ``delta_m`` overlaid as its own colored series.

    Answers "for this bin step, what's the shape of the error distribution across the
    sampled image" — not just a single summary number per ``delta_m`` (see
    :func:`plot_forward_tradeoff` for the summary-statistic-vs-delta_m curve instead).

    Color encodes ``delta_m`` via :func:`sequential_colors` (a sequential colormap
    sampled across the sweep, since ``delta_m`` is a continuous, ordered parameter —
    not the categorical model palette). Series combine translucent filled areas with
    stronger outlines so overlapping distributions remain distinguishable.

    Metrics are never shared across plots (different scales) and normalizations are
    never mixed on one plot — call once per (metric, normalization) pair, or use
    :func:`plot_all_metrics_by_delta_m` to do that for every metric at once.
    """
    resolved = resolve_theme(theme)
    figure = None
    delta_ms = sorted({record["delta_m"] for record in records if record["metric"] == metric and record["normalization"] == normalization})
    colors = sequential_colors(resolved, len(delta_ms))
    for delta_m, color in zip(delta_ms, colors):
        values = np.asarray([
            record["value"] for record in records
            if record["metric"] == metric and record["normalization"] == normalization and record["delta_m"] == delta_m
        ])
        figure, ax = plot_metric_distribution(values, metric, ax, bins, kind, label=f"Δm={delta_m:g}", theme=resolved, color=color, histtype="stepfilled")
    ax.set_title(f"{metric} ({normalization}) across Δm", loc=resolved.title_location)
    figure.tight_layout()
    return figure, ax


def plot_all_metrics_by_delta_m(
    records: Sequence[Mapping[str, Any]],
    normalization: str = "raw",
    metrics: Optional[Sequence[str]] = None,
    kind: str = "histogram",
    bins: int = 40,
    theme: VisualizationTheme | str | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Render :func:`plot_metric_by_delta_m` for every metric, one figure each.

    :param metrics: Which metrics to render; defaults to every metric present in
        ``records`` under ``normalization``.
    :return: ``{metric: (figure, ax)}`` — every returned figure/ax is still a normal
        matplotlib object and can be further modified (titles, limits, saving) after.
    """
    resolved_metrics = list(metrics) if metrics is not None else sorted({record["metric"] for record in records if record["normalization"] == normalization})
    return {metric: plot_metric_by_delta_m(records, metric, normalization, kind, bins, theme=theme) for metric in resolved_metrics}


def plot_forward_tradeoff(
    summary_records: Sequence[Mapping[str, Any]],
    metric: str,
    x: str = "delta_m",
    normalization: str = "raw",
    statistic: str = "median",
    ax=None,
    theme: VisualizationTheme | str | None = None,
    quantiles: Optional[tuple[float, float]] = (0.25, 0.75),
):
    """Plot one summary statistic of one metric against ``delta_m`` (or
    ``feature_dimension`` — pass ``x="feature_dimension"`` for the rate-distortion
    framing: output size vs error, instead of step size vs error).

    Takes :func:`summarize_forward_sweep` output (not the long-form records) — one
    point per ``delta_m``, connected in increasing-``x`` order. By default, vertical
    ranges and a translucent band show the per-spectrum interquartile interval.

    :param quantiles: Lower and upper per-spectrum quantiles drawn around the selected
        statistic, or ``None`` to draw only the central curve.
    :type quantiles: Optional[tuple[float, float]]
    """
    selected = [record for record in summary_records if record["metric"] == metric and record["normalization"] == normalization]
    selected.sort(key=lambda record: record[x])
    figure, ax = plot_metric_tradeoff([record[x] for record in selected], [record[statistic] for record in selected], x, metric, ax=ax, theme=theme)
    if quantiles is not None:
        lower_quantile, upper_quantile = (float(value) for value in quantiles)
        if not 0.0 <= lower_quantile <= upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1.")
        xs = np.asarray([record[x] for record in selected], dtype=np.float32)
        bounds = np.asarray([
            np.quantile(np.asarray(record["_values"], dtype=np.float32), (lower_quantile, upper_quantile)) if record["_values"] else (np.nan, np.nan)
            for record in selected
        ])
        color = ax.lines[-1].get_color()
        resolved = resolve_theme(theme)
        ax.fill_between(xs, bounds[:, 0], bounds[:, 1], color=color, alpha=resolved.distribution_fill_alpha, linewidth=0.0)
        ax.vlines(xs, bounds[:, 0], bounds[:, 1], color=color, linewidth=resolved.distribution_line_width, alpha=resolved.secondary_alpha)
    figure.tight_layout()
    return figure, ax
