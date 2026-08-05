"""Inverse-binner parameter sweep at a fixed, already-chosen forward bin step.

Answers: for one inverse-binner method, which parameter setting is a good trade-off,
and what does that trade-off actually look like across the sampled spectra? Nothing
here auto-selects a "best" parameter — grids and plots are produced so the choice can
be made by inspection (primarily via the Masserstein/W1 distance as the localization
proxy, per the methodology), then written into the manual parameter slot at the top of
the notebook. See ``methodology.md`` §6 step 2.

Two comparisons are always kept separate (never averaged together):
``inverse_binned`` (B(X) vs INB(X), the inverse binner's own compression error) and
``inverse_original`` (X vs INB(X), the full round trip).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ....binners.binners_manager import BinnerManager
from ....metrics import match_spectral_points, peak_collision_rate, spectral_point_metrics
from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.metrics import plot_metric_tradeoff
from ....visualization.spectra import plot_sparse_spectrum_match, plot_sparse_spectrum_multi_match
from .binner_forward_analysis import DEFAULT_NORMALIZATIONS, crop_spectrum, rescale_match
from ..binning.metrics import normalize_intensity, summarize_metric
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)

Comparison = str  # "inverse_binned" | "inverse_original"
COMPARISONS: tuple[Comparison, ...] = ("inverse_binned", "inverse_original")

# Linestyle encodes `comparison` on plots that overlay both at once (color is reserved
# for the method/group axis — see the binning-analysis-conventions skill).
LINESTYLE_BY_COMPARISON: dict[str, str] = {"inverse_binned": "-", "inverse_original": "--"}

# Metrics where a larger value is better (used to decide sort direction when ranking
# spectra "best" -> "worst"); everything else is treated as smaller-is-better.
MAXIMIZE_METRICS: frozenset[str] = frozenset({
    "cosine_similarity", "peak_precision", "peak_recall", "matched_intensity_fraction", "size_reduction",
})


def inverse_binner_factory(method: str, **params: Any) -> Callable[[Any], Any]:
    """Build a ``(binner) -> inverse_binner`` factory for :meth:`BinningPrecompute.inverse`.

    :param method: A name registered in ``BinnerManager.INVERSE_REGISTRY`` (e.g.
        ``"ThresholdInverseBinner"``).
    :param params: Constructor kwargs forwarded as-is (excluding ``binner``/``active_context``,
        which :meth:`BinningPrecompute.inverse` injects).
    """
    def build(binner: Any) -> Any:
        return BinnerManager.get_inverse_binner(method, binner=binner, **params)
    return build


def select_rank_positions(count: int, n_shown: int = 10, stride: Optional[int] = None) -> list[int]:
    """Pick up to ``n_shown`` representative 0-indexed positions spanning a ranked list.

    Used to sample a manageable, viewable subset of a long ranked spectrum list instead
    of only the literal best/worst extremes (which say nothing about the bulk of the
    distribution) or every single spectrum (which is not viewable).

    - If ``stride`` is given: positions ``0, stride, 2*stride, ...`` capped at
      ``n_shown`` entries, with the last rank always appended if not already included
      (e.g. ``stride=100`` -> "first, then every 100th, ..., plus the worst one").
    - Otherwise: ``n_shown`` positions evenly spaced across ``[0, count - 1]``
      (best -> worst inclusive), deduplicated — the default, order-agnostic behavior.

    :param count: Length of the ranked list (e.g. number of spectra with a finite value
        for the metric being ranked on).
    :return: Sorted, deduplicated list of positions, length <= ``n_shown``.
    """
    if count <= 0:
        return []
    if stride is not None and stride > 0:
        positions = list(range(0, count, stride))[:n_shown]
        if positions[-1] != count - 1:
            positions.append(count - 1)
        return positions
    shown = min(n_shown, count)
    if shown <= 1:
        return [0]
    return sorted({int(round(index * (count - 1) / (shown - 1))) for index in range(shown)})


def inverse_sweep_records(
    precompute: BinningPrecompute,
    delta_m: float,
    method_grid: Sequence[Mapping[str, Any]],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    normalizations: Sequence[str] = DEFAULT_NORMALIZATIONS,
    comparisons: Sequence[Comparison] = COMPARISONS,
) -> list[dict[str, Any]]:
    """Long-form records for every grid point, spectrum, comparison, normalization, metric.

    For every ``method_grid`` entry: resolves (and caches, via ``precompute.inverse``)
    that inverse binner's output for every sampled spectrum, then for each of
    ``comparisons`` matches the appropriate reference (``B(X)`` for ``inverse_binned``,
    raw ``X`` for ``inverse_original``) against ``INB(X)`` with
    ``match_spectral_points(..., "one_to_one")`` and computes every
    ``spectral_point_metrics`` field plus ``peak_collision_rate`` (position-only, one
    value per (spectrum, grid point, comparison), same across every normalization).
    Matching is computed once per (spectrum, grid point, comparison) and reused across
    normalizations via :func:`rescale_match` (see that function's docstring — matching
    is position-only, independent of scaling).

    :param method_grid: Sequence of ``{"label": str, "method": str, "params": dict}`` —
        ``method`` is a name registered in ``BinnerManager.INVERSE_REGISTRY``, ``params``
        its constructor kwargs. ``label`` identifies this grid point in every record and
        in plots; make it distinguishing (e.g. include the swept parameter's value, as
        in ``"threshold/q0.90"``). Grid points sharing the same ``method`` are treated as
        one comparable parameter family for grouping/plotting purposes.
    :return: One row per (grid point, spectrum, comparison, normalization, metric), plus
        every ``param_*`` field (display-stringified — not safe to reconstruct
        constructor kwargs from; use the original ``method_grid`` entry for that, see
        :func:`plot_ranked_spectra`) and any diagnostics the inverse binner produced
        (only if it was built with ``track_diagnostics=True``).
    """
    logger.info(
        "START inverse sweep: %s grid points x %s spectra x %s comparisons x %s normalizations at delta_m=%s.",
        len(method_grid), precompute.spectrum_ids.size, len(comparisons), len(normalizations), delta_m,
    )
    started = time.perf_counter()
    epsilon = np.finfo(float).eps
    records: list[dict[str, Any]] = []
    for point in method_grid:
        label, method, params = point["label"], point["method"], dict(point.get("params", {}))
        factory = inverse_binner_factory(method, **params)
        forward_binner, inverse_binner, inverse_cache = precompute.inverse(delta_m, factory, x_min, x_max, cache_key=label)
        _, forward_cache = precompute.forward(delta_m, x_min, x_max)
        grid_mz = np.asarray(forward_binner.GetXAxis(), dtype=np.float64)
        label_started = time.perf_counter()
        for spectrum_id in tqdm(precompute.spectrum_ids, desc=f"Matching {label}", unit="spectrum"):
            raw_mz, raw_y = crop_spectrum(*precompute.raw(spectrum_id), forward_binner.x_min, forward_binner.x_max)
            grid_y = forward_cache[int(spectrum_id)]
            result = inverse_cache[int(spectrum_id)]

            ## `inverse_binned` references B(X); `inverse_original` references raw X.
            reference_by_comparison = {"inverse_binned": (grid_mz, grid_y), "inverse_original": (raw_mz, raw_y)}
            for comparison in comparisons:
                reference_mz, reference_y = reference_by_comparison[comparison]
                base_match = match_spectral_points(reference_mz, reference_y, result.mz, result.intensity, tolerance, tolerance_unit, "one_to_one")
                ## Collision rate: position-only, independent of normalization (see binner_forward_analysis).
                collision_rate = (
                    peak_collision_rate(reference_mz, reference_y, result.mz, result.intensity, tolerance, tolerance_unit)
                    if float(np.sum(reference_y)) > 0 else 0.0
                )
                for normalization in normalizations:
                    normalized_reference = normalize_intensity(reference_y, normalization)
                    normalized_candidate = normalize_intensity(result.intensity, normalization)
                    match = rescale_match(base_match, normalized_reference, normalized_candidate)
                    values = spectral_point_metrics(reference_mz, normalized_reference, result.mz, normalized_candidate, match)
                    values["tic_relative_error"] = (
                        abs(float(np.sum(reference_y)) - float(np.sum(result.intensity))) / (abs(float(np.sum(reference_y))) + epsilon)
                        if normalization == "raw" else np.nan
                    )
                    values["peak_collision_rate"] = collision_rate
                    for metric, value in values.items():
                        records.append({
                            "spectrum_id": int(spectrum_id), "label": label, "method": method, "delta_m": float(delta_m),
                            "comparison": comparison, "normalization": normalization, "metric": metric, "value": float(value),
                            **{f"param_{key}": _stringify(val) for key, val in params.items()}, **result.diagnostics,
                        })
        logger.info("DONE matching %s in %.2fs.", label, time.perf_counter() - label_started)
    logger.info("DONE inverse sweep: %s records in %.2fs.", len(records), time.perf_counter() - started)
    return records


def _stringify(value: Any) -> Any:
    """Make a constructor parameter safe for a display/record field (not for reuse as a kwarg)."""
    return value if isinstance(value, (int, float, str, bool)) or value is None else repr(value)


def summarize_inverse_sweep(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group :func:`inverse_sweep_records` output by (label, comparison, normalization, metric).

    :return: One row per group with ``summarize_metric``'s summary statistics plus
        ``method`` (constant within a ``label`` group, carried through so plots can
        color/group by method — see :func:`plot_inverse_tradeoff`).
    """
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    method_by_label: dict[str, str] = {}
    for record in records:
        key = (record["label"], record["comparison"], record["normalization"], record["metric"])
        grouped.setdefault(key, []).append(record["value"])
        method_by_label[record["label"]] = record["method"]
    return [
        {"label": key[0], "method": method_by_label[key[0]], "comparison": key[1], "normalization": key[2], "metric": key[3], **summarize_metric(values), "_values": [float(value) for value in values if np.isfinite(value)]}
        for key, values in grouped.items()
    ]


def plot_inverse_tradeoff(
    summary_records: Sequence[Mapping[str, Any]],
    metric: str,
    comparisons: Sequence[Comparison] = COMPARISONS,
    normalization: str = "raw",
    statistic: str = "median",
    labels: Optional[Sequence[str]] = None,
    ax=None,
    theme: VisualizationTheme | str | None = None,
    quantiles: Optional[tuple[float, float]] = (0.25, 0.75),
):
    """One plot, every method and (by default) both comparisons overlaid at once —
    the primary plot for picking a manual parameter value.

    Visual encoding:

    - **Color** = ``method`` (the inverse-binner class, e.g. all ``ThresholdInverseBinner``
      grid points share a color, all ``CumulativeMassInverseBinner`` points share a
      different one) via ``theme.color_for_model`` — consistent across every call in a
      notebook run as long as the same method names keep appearing in the same order.
    - **Linestyle** = ``comparison`` (solid = ``inverse_binned``, dashed =
      ``inverse_original``, see :data:`LINESTYLE_BY_COMPARISON`).
    - **X axis** = one categorical tick per ``label`` (every grid point from every
      method, in ``labels`` order if given, else first-seen order). Points are
      connected with a line **only within the same method** — different methods have
      different, non-comparable parameter spaces, so connecting e.g. a threshold
      quantile point to a cumulative-mass fraction point across methods would imply a
      continuity that does not exist. This also means every grid point you swept is
      shown, not just one per method (multiple points of the same method, e.g.
      ``threshold/q0.50`` and ``threshold/q0.90``, both appear and are connected to
      each other, not collapsed).

    :param labels: Explicit x-axis order (typically ``METHOD_GRID_LABELS`` from the
        notebook); defaults to every label present in ``summary_records`` for this
        ``metric``/``normalization``, first-seen order.
    :param quantiles: Lower and upper per-spectrum quantiles drawn around every method
        and comparison curve, or ``None`` to draw only summary-statistic lines.
    :type quantiles: Optional[tuple[float, float]]
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure

    selected = [
        record for record in summary_records
        if record["metric"] == metric and record["normalization"] == normalization and record["comparison"] in comparisons
    ]
    ordered_labels = list(labels) if labels is not None else list(dict.fromkeys(record["label"] for record in selected))
    method_by_label = {record["label"]: record["method"] for record in selected}
    method_order = list(dict.fromkeys(method_by_label[label] for label in ordered_labels if label in method_by_label))
    by_key = {(record["label"], record["comparison"]): record for record in selected}
    x_position = {label: index for index, label in enumerate(ordered_labels)}
    if quantiles is not None:
        lower_quantile, upper_quantile = (float(value) for value in quantiles)
        if not 0.0 <= lower_quantile <= upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1.")

    for method_index, method in enumerate(method_order):
        color = resolved.color_for_model(method, method_index)
        for comparison in comparisons:
            xs = [x_position[label] for label in ordered_labels if method_by_label.get(label) == method and (label, comparison) in by_key]
            ys = [by_key[(label, comparison)][statistic] for label in ordered_labels if method_by_label.get(label) == method and (label, comparison) in by_key]
            if not xs:
                continue
            selected_records = [by_key[(label, comparison)] for label in ordered_labels if method_by_label.get(label) == method and (label, comparison) in by_key]
            ax.plot(
                xs,
                ys,
                marker="o",
                color=color,
                linestyle=LINESTYLE_BY_COMPARISON.get(comparison, "-."),
                linewidth=resolved.line_width,
                markersize=resolved.marker_size,
                markerfacecolor=color,
                markeredgecolor=resolved.panel_color,
                markeredgewidth=resolved.marker_edge_width,
                alpha=resolved.primary_alpha,
                label=f"{method} / {comparison}",
            )
            if quantiles is not None:
                bounds = np.asarray([
                    np.quantile(np.asarray(record["_values"], dtype=np.float64), (lower_quantile, upper_quantile)) if record["_values"] else (np.nan, np.nan)
                    for record in selected_records
                ])
                ax.fill_between(xs, bounds[:, 0], bounds[:, 1], color=color, alpha=resolved.distribution_fill_alpha, linewidth=0.0)
                ax.vlines(xs, bounds[:, 0], bounds[:, 1], color=color, linewidth=resolved.distribution_line_width, alpha=resolved.secondary_alpha)

    ax.set_xticks(list(x_position.values()))
    ax.set_xticklabels(list(x_position.keys()), rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} ({normalization}, {statistic})", loc=resolved.title_location)
    ax.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
    ax.legend(fontsize=resolved.legend_font_size, loc=resolved.legend_location, ncols=resolved.legend_columns, frameon=resolved.legend_frame)
    figure.tight_layout()
    return figure, ax


def plot_ranked_spectra(
    precompute: BinningPrecompute,
    records: Sequence[Mapping[str, Any]],
    method_grid_point: Mapping[str, Any],
    metric: str,
    comparison: Comparison,
    normalization: str = "raw",
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    n_best: int = 3,
    n_worst: int = 3,
    delta_m: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    theme: VisualizationTheme | str | None = None,
):
    """Mirror-plot the ``n_best`` and ``n_worst`` spectra for **one** grid point, to see
    which failure modes a metric actually weighs most/least — not for choosing a
    per-spectrum parameter (the parameter choice stays global). For comparing several
    methods' reconstructions of the *same* spectra side by side, use
    :func:`plot_ranked_spectra_across_methods` instead.

    Reuses ``visualization.spectra.plot_sparse_spectrum_match`` as-is: reference plotted
    positive, candidate mirrored negative, matched pairs connected by faint lines,
    unmatched points marked with an ``x`` (unmatched reference = a reference peak with
    no candidate within tolerance, i.e. lost signal; unmatched candidate = a candidate
    point with no reference peak within tolerance, i.e. spurious/extra signal), matched-
    intensity residual in the lower panel.

    :param method_grid_point: The exact ``{"label", "method", "params"}`` dict passed
        to :func:`inverse_sweep_records` for this label — reused verbatim to rebuild the
        inverse binner. Record fields are display-stringified (see ``_stringify``) and
        are not safe to parse back into constructor kwargs for nested options like
        ``threshold_options``, hence taking the original dict rather than reconstructing
        it from ``records``.
    """
    resolved = resolve_theme(theme)
    label, method, params = method_grid_point["label"], method_grid_point["method"], dict(method_grid_point.get("params", {}))
    selected = [
        record for record in records
        if record["label"] == label and record["metric"] == metric and record["comparison"] == comparison
        and record["normalization"] == normalization and np.isfinite(record["value"])
    ]
    ordered = sorted(selected, key=lambda record: record["value"], reverse=metric in MAXIMIZE_METRICS)
    chosen = ordered[:n_best] + (ordered[-n_worst:] if n_worst else [])
    if not chosen:
        logger.warning("No finite records for label=%s, metric=%s, comparison=%s, normalization=%s — skipping (metric not computed for this comparison, or all values are NaN).", label, metric, comparison, normalization)
        return None, None
    resolved_delta_m = delta_m if delta_m is not None else next(iter(records))["delta_m"]
    logger.info("Plotting %s best + %s worst spectra for label=%s, metric=%s, comparison=%s.", n_best, n_worst, label, metric, comparison)

    forward_binner, inverse_binner, inverse_cache = precompute.inverse(resolved_delta_m, inverse_binner_factory(method, **params), x_min, x_max, cache_key=label)
    _, forward_cache = precompute.forward(resolved_delta_m, x_min, x_max)
    grid_mz = np.asarray(forward_binner.GetXAxis(), dtype=np.float64)

    figure, axes = plt.subplots(2 * len(chosen), 1, figsize=(resolved.figure_size[0], 4 * len(chosen)), dpi=resolved.figure_dpi, squeeze=False)
    for row, record in enumerate(chosen):
        spectrum_id = record["spectrum_id"]
        raw_mz, raw_y = crop_spectrum(*precompute.raw(spectrum_id), forward_binner.x_min, forward_binner.x_max)
        grid_y = forward_cache[int(spectrum_id)]
        result = inverse_cache[int(spectrum_id)]
        reference_mz, reference_y = (grid_mz, grid_y) if comparison == "inverse_binned" else (raw_mz, raw_y)
        match = match_spectral_points(reference_mz, reference_y, result.mz, result.intensity, tolerance, tolerance_unit, "one_to_one")
        signal_axis, residual_axis = axes[2 * row, 0], axes[2 * row + 1, 0]
        plot_sparse_spectrum_match(reference_mz, reference_y, result.mz, result.intensity, match, axes=(signal_axis, residual_axis), candidate_label=label, theme=resolved)
        rank_label = "best" if row < n_best else "worst"
        signal_axis.set_title(f"{metric} {rank_label} | spectrum {spectrum_id} | {record['value']:.4g}", loc=resolved.title_location)
    figure.tight_layout()
    return figure, axes


def plot_ranked_spectra_across_methods(
    precompute: BinningPrecompute,
    records: Sequence[Mapping[str, Any]],
    method_grid: Sequence[Mapping[str, Any]],
    ranking_label: str,
    metric: str,
    comparison: Comparison,
    normalization: str = "raw",
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    n_shown: int = 10,
    stride: Optional[int] = None,
    delta_m: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    theme: VisualizationTheme | str | None = None,
):
    """For spectra ranked by one method's metric value, show *every* method in
    ``method_grid`` reconstructing that same spectrum, on one mirrored plot per spectrum.

    This is the cross-method complement to :func:`plot_ranked_spectra` (which only shows
    one method): the ranking (which spectra to look at, and in what order) is driven by
    ``ranking_label``'s ``metric``/``comparison``/``normalization``, but every row of the
    figure plots the **same reference spectrum** against **all** ``method_grid`` entries'
    reconstructions overlaid (via ``visualization.spectra.plot_sparse_spectrum_multi_match``:
    one reference (+), one mirrored (-) trace per method in that method's own consistent
    color, one residual trace per method). Answers "for the spectrum where method X does
    worst, how do the other methods handle it."

    Rank positions shown are picked by :func:`select_rank_positions` (``n_shown``/
    ``stride``) rather than only the literal extremes — the ranked list can span
    thousands of spectra, "best" and "worst" alone say nothing about the bulk of it, and
    showing every spectrum is not viewable, so this samples representative positions
    across the whole ranked list (default: ``n_shown`` evenly spaced positions
    including both endpoints; pass ``stride`` for "first, then every Nth" instead).

    :param method_grid: Same shape as :func:`inverse_sweep_records`'s ``method_grid`` —
        every entry's inverse binner is rebuilt (via the precompute cache, so no
        recomputation for grid points already swept) and shown on every plotted row.
    :param ranking_label: Which ``method_grid`` entry's ``label`` to rank spectra by.
    """
    resolved = resolve_theme(theme)
    selected = [
        record for record in records
        if record["label"] == ranking_label and record["metric"] == metric and record["comparison"] == comparison
        and record["normalization"] == normalization and np.isfinite(record["value"])
    ]
    ordered = sorted(selected, key=lambda record: record["value"], reverse=metric in MAXIMIZE_METRICS)
    if not ordered:
        logger.warning("No finite records for ranking_label=%s, metric=%s, comparison=%s, normalization=%s — skipping (metric not computed for this comparison, or all values are NaN).", ranking_label, metric, comparison, normalization)
        return None, None
    positions = select_rank_positions(len(ordered), n_shown, stride)
    chosen = [(position, ordered[position]) for position in positions]
    resolved_delta_m = delta_m if delta_m is not None else next(iter(records))["delta_m"]
    logger.info(
        "Plotting %s rank positions (out of %s) across %s methods, ranked by label=%s, metric=%s.",
        len(chosen), len(ordered), len(method_grid), ranking_label, metric,
    )

    # Resolve every method's inverse-binner cache once (precompute dedupes by cache_key,
    # so grid points already swept in inverse_sweep_records are not recomputed here).
    inverse_caches: dict[str, dict[int, Any]] = {}
    forward_binner = None
    for point in method_grid:
        point_label, point_method, point_params = point["label"], point["method"], dict(point.get("params", {}))
        forward_binner, _, inverse_caches[point_label] = precompute.inverse(resolved_delta_m, inverse_binner_factory(point_method, **point_params), x_min, x_max, cache_key=point_label)
    _, forward_cache = precompute.forward(resolved_delta_m, x_min, x_max)
    grid_mz = np.asarray(forward_binner.GetXAxis(), dtype=np.float64)

    figure, axes = plt.subplots(2 * len(chosen), 1, figsize=(resolved.figure_size[0], 4 * len(chosen)), dpi=resolved.figure_dpi, squeeze=False)
    for row, (position, record) in enumerate(chosen):
        spectrum_id = record["spectrum_id"]
        raw_mz, raw_y = crop_spectrum(*precompute.raw(spectrum_id), forward_binner.x_min, forward_binner.x_max)
        grid_y = forward_cache[int(spectrum_id)]
        reference_mz, reference_y = (grid_mz, grid_y) if comparison == "inverse_binned" else (raw_mz, raw_y)

        candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        matches: dict[str, Any] = {}
        for point in method_grid:
            point_label = point["label"]
            result = inverse_caches[point_label][int(spectrum_id)]
            candidates[point_label] = (result.mz, result.intensity)
            matches[point_label] = match_spectral_points(reference_mz, reference_y, result.mz, result.intensity, tolerance, tolerance_unit, "one_to_one")

        signal_axis, residual_axis = axes[2 * row, 0], axes[2 * row + 1, 0]
        plot_sparse_spectrum_multi_match(reference_mz, reference_y, candidates, matches, axes=(signal_axis, residual_axis), theme=resolved)
        signal_axis.set_title(
            f"rank {position + 1}/{len(ordered)} by {ranking_label} | {metric} | spectrum {spectrum_id} | {record['value']:.4g}",
            loc=resolved.title_location,
        )
    figure.tight_layout()
    return figure, axes


def plot_ranked_spectra_grid(
    precompute: BinningPrecompute,
    records: Sequence[Mapping[str, Any]],
    method_grid: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    comparison: Comparison,
    normalization: str = "raw",
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    n_best: int = 1,
    n_worst: int = 1,
    delta_m: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    theme: VisualizationTheme | str | None = None,
) -> dict[tuple[str, str], tuple[Any, Any]]:
    """Run :func:`plot_ranked_spectra` for every (grid point, metric) combination at
    once, instead of hand-editing ``method_grid_point``/``metric`` and re-running one
    cell repeatedly.

    :param metrics: Metrics to render for every grid point (typically ``KEY_METRICS``
        from the notebook config).
    :return: ``{(label, metric): (figure, axes)}`` — ``len(method_grid) * len(metrics)``
        figures; logs the total up front since this can be a lot of plots.
    """
    logger.info(
        "Rendering ranked-spectra grid: %s grid points x %s metrics = %s figures.",
        len(method_grid), len(metrics), len(method_grid) * len(metrics),
    )
    results: dict[tuple[str, str], tuple[Any, Any]] = {}
    for point in method_grid:
        for metric in metrics:
            results[(point["label"], metric)] = plot_ranked_spectra(
                precompute, records, point, metric, comparison, normalization,
                tolerance, tolerance_unit, n_best, n_worst, delta_m, x_min, x_max, theme,
            )
    return results


def plot_ranked_spectra_across_methods_grid(
    precompute: BinningPrecompute,
    records: Sequence[Mapping[str, Any]],
    method_grid: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    comparison: Comparison,
    normalization: str = "raw",
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    n_shown: int = 10,
    stride: Optional[int] = None,
    delta_m: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    theme: VisualizationTheme | str | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Run :func:`plot_ranked_spectra_across_methods` for every metric at once, ranking
    by every ``method_grid`` entry in turn (one figure per (ranking method, metric)).

    :return: ``{f"{ranking_label}|{metric}": (figure, axes)}``.
    """
    logger.info(
        "Rendering cross-method ranked-spectra grid: %s ranking methods x %s metrics = %s figures.",
        len(method_grid), len(metrics), len(method_grid) * len(metrics),
    )
    results: dict[str, tuple[Any, Any]] = {}
    for point in method_grid:
        for metric in metrics:
            results[f"{point['label']}|{metric}"] = plot_ranked_spectra_across_methods(
                precompute, records, method_grid, point["label"], metric, comparison, normalization,
                tolerance, tolerance_unit, n_shown, stride, delta_m, x_min, x_max, theme,
            )
    return results
