"""Part 2 — cross-method comparison at each method's own optimal settings.

Structurally this is the *same* computation as the Part 1 inverse-binner sweep
(:func:`inverse_binner_analysis.inverse_sweep_records`) — the only difference is
semantic: instead of a parameter *sweep within* one method, ``optimal_configs`` holds
exactly one, already-chosen grid point *per method* (the values you wrote into
``CHOSEN_INVERSE_CONFIG``-style decision cells after inspecting Part 1's tradeoff
plots). No new record-computation code is added here on purpose — reusing
``inverse_sweep_records``/``summarize_inverse_sweep``/``plot_inverse_tradeoff`` keeps
this comparison numerically identical in methodology to Part 1, and
``plot_inverse_tradeoff`` already groups by method/colors and overlays comparisons, so
it doubles as the Part 2 comparison plot unchanged. This module only adds a
method-vs-method ranking table, which Part 1's per-method sweep has no use for. See
``methodology.md`` §6 step 4.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ....utils.logger import get_custom_logger
from .inverse_binner_analysis import (
    Comparison,
    MAXIMIZE_METRICS,
    inverse_sweep_records,
    summarize_inverse_sweep,
)
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)


def compare_methods_at_optimal_settings(
    precompute: BinningPrecompute,
    delta_m: float,
    optimal_configs: Sequence[Mapping[str, Any]],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Thin, semantically-named wrapper over
    :func:`inverse_binner_analysis.inverse_sweep_records` for Part 2: ``optimal_configs``
    should hold exactly one grid point per method (its chosen optimum from Part 1), not
    a parameter sweep. Passing a full sweep grid here works too (nothing enforces the
    "one point per method" convention) but defeats the purpose — Part 1 already covers
    that.

    :param optimal_configs: Same shape as ``inverse_sweep_records``'s ``method_grid`` —
        one ``{"label", "method", "params"}`` entry per method being compared.
    """
    logger.info("Comparing %s methods at their own optimal settings, delta_m=%s.", len(optimal_configs), delta_m)
    return inverse_sweep_records(precompute, delta_m, optimal_configs, x_min, x_max, **kwargs)


def rank_methods_table(
    summary_records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    comparison: Comparison,
    normalization: str = "raw",
    statistic: str = "median",
) -> list[dict[str, Any]]:
    """Pivot :func:`compare_methods_at_optimal_settings` summary into one row per
    method with every requested metric as a column, plus a 1-indexed rank per metric
    (1 = best, direction from :data:`inverse_binner_analysis.MAXIMIZE_METRICS`) — the
    direct "who wins on what" table. No metrics are combined into one score; ranks are
    reported per metric, side by side, not averaged into an overall rank.

    :return: One row per ``label`` (``method_comparison_analysis`` calls its grid
        points "label" same as Part 1) with ``{metric: value}`` and
        ``{f"{metric}_rank": rank}`` for every metric in ``metrics``.
    """
    by_label_metric = {
        (record["label"], record["metric"]): record[statistic]
        for record in summary_records
        if record["comparison"] == comparison and record["normalization"] == normalization and record["metric"] in metrics
    }
    method_by_label = {record["label"]: record["method"] for record in summary_records}
    labels = list(dict.fromkeys(record["label"] for record in summary_records))
    table: dict[str, dict[str, Any]] = {label: {"label": label, "method": method_by_label.get(label, label)} for label in labels}
    for metric in metrics:
        values = {label: by_label_metric.get((label, metric)) for label in labels}
        finite_labels = [label for label, value in values.items() if value is not None]
        reverse = metric in MAXIMIZE_METRICS
        ranking = sorted(finite_labels, key=lambda label: values[label], reverse=reverse)
        rank_by_label = {label: rank + 1 for rank, label in enumerate(ranking)}
        for label in labels:
            table[label][metric] = values.get(label)
            table[label][f"{metric}_rank"] = rank_by_label.get(label)
    return [table[label] for label in labels]
