"""Per-ion annotation prevalence and the evidence regime each ion falls into.

A pooled negative count says nothing about *which* ions the negatives come from.
Two very different situations produce the same pooled number:

*rare but detectable* — the ion is annotated in few pixels, yet its bin still
carries intensity in most of the rest, so its unannotated entries stay ``U`` and the
class contributes almost nothing to a P/N objective;

*undetectable* — the ion's bin is essentially empty across the image, so nearly all
of its unannotated entries become ``N`` and the class is trained on negatives that
carry no chemical information, only the fact that nothing was measured there.

This module separates those regimes per ion and aggregates them along the m/z axis
of the configured 200-900 window.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....utils.logger import get_custom_logger
from .precompute import EvidenceStatistics, cumulative_below

logger = get_custom_logger(__name__)


def class_prevalence_records(statistics: EvidenceStatistics) -> list[dict[str, Any]]:
    """Annotation prevalence of every ion, independently of any threshold.

    Prevalence is the share of pixels with an available label in which the ion is
    annotated. It is the quantity that decides whether a class is *scoreable* at
    all, and it is fixed by the annotations alone — no evidence rule enters here.

    :param statistics: A completed population pass.
    :return: One record per ion.
    :rtype: list[dict[str, Any]]
    """
    # REMARK: Entry counts are identical across dilation radii — the radius changes
    # which bucket an entry lands in, never whether it exists — so any radius axis
    # position yields the same prevalence.
    radius = 0
    annotated = statistics.annotated_relative[radius].sum(axis=1).astype(np.float64)  # (C,)
    unannotated = statistics.unannotated_relative[radius].sum(axis=1).astype(np.float64)  # (C,)
    available = annotated + unannotated  # (C,)

    records: list[dict[str, Any]] = []
    for position, name in enumerate(statistics.class_names):
        records.append({
            "class_name": name,
            "mz": float(statistics.class_mz[position]),
            "bin_centre": float(statistics.class_bin_centre[position]),
            "mapped_bins": int(statistics.class_bin_counts[position]),
            "annotated_pixels": float(annotated[position]),
            "available_pixels": float(available[position]),
            "prevalence": float(annotated[position] / available[position]) if available[position] else float("nan"),
        })
    logger.info("Summarized annotation prevalence for %s ion(s).", len(records))
    return records


def class_regime_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    relative_threshold: float,
    undetectable_negative_fraction: float = 0.95,
    rare_prevalence: float = 0.01,
) -> list[dict[str, Any]]:
    """Assign every ion an evidence regime at one candidate threshold.

    The regime is a two-way split on quantities that are themselves reported, so a
    reader can move the cut-offs without recomputing anything:

    ``undetectable``
        the threshold converts at least ``undetectable_negative_fraction`` of the
        ion's unannotated entries into negatives — its bin is empty almost
        everywhere;
    ``rare``
        prevalence below ``rare_prevalence`` while the negative fraction stays below
        the undetectable cut-off — few annotations, but the signal is there;
    ``common``
        everything else.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param relative_threshold: The candidate threshold, a grid boundary.
    :param undetectable_negative_fraction: Negative-fraction cut-off for
        ``undetectable``.
    :param rare_prevalence: Prevalence cut-off for ``rare``.
    :return: One record per ion.
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges
    annotated = statistics.annotated_relative[radius]  # (C, K)
    unannotated = statistics.unannotated_relative[radius]  # (C, K)
    annotated_total = annotated.sum(axis=1).astype(np.float64)  # (C,)
    unannotated_total = unannotated.sum(axis=1).astype(np.float64)  # (C,)
    negative = cumulative_below(unannotated, edges, float(relative_threshold)).astype(np.float64)  # (C,)
    contradicted = cumulative_below(annotated, edges, float(relative_threshold)).astype(np.float64)  # (C,)
    available = annotated_total + unannotated_total  # (C,)

    records: list[dict[str, Any]] = []
    for position, name in enumerate(statistics.class_names):
        prevalence = annotated_total[position] / available[position] if available[position] else float("nan")
        negative_fraction = (
            negative[position] / unannotated_total[position] if unannotated_total[position] else float("nan")
        )
        if negative_fraction >= undetectable_negative_fraction:
            regime = "undetectable"
        elif prevalence < rare_prevalence:
            regime = "rare"
        else:
            regime = "common"
        records.append({
            "class_name": name,
            "mz": float(statistics.class_mz[position]),
            "bin_radius": int(bin_radius),
            "relative_threshold": float(relative_threshold),
            "prevalence": float(prevalence),
            "annotated_pixels": float(annotated_total[position]),
            "negative_entries": float(negative[position]),
            "uncertain_entries": float(unannotated_total[position] - negative[position]),
            "negative_fraction_of_unannotated": float(negative_fraction),
            "negative_to_positive_ratio": (
                float(negative[position] / annotated_total[position]) if annotated_total[position] else float("inf")
            ),
            "positive_contradiction_rate": (
                float(contradicted[position] / annotated_total[position]) if annotated_total[position] else float("nan")
            ),
            "regime": regime,
        })
    counts: dict[str, int] = {}
    for record in records:
        counts[record["regime"]] = counts.get(record["regime"], 0) + 1
    logger.info("Ion regimes at relative_threshold=%s: %s.", relative_threshold, counts)
    return records


def mz_window_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    relative_threshold: float,
    window_width: float = 50.0,
) -> list[dict[str, Any]]:
    """Aggregate prevalence and negative yield along the configured m/z axis.

    The binning window is fixed at 200-900 m/z, and neither annotation density nor
    spectral intensity is uniform across it. This aggregation shows where in the
    window the evidence rule is actually doing work and where it is mostly labelling
    empty axis.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param relative_threshold: The candidate threshold, a grid boundary.
    :param window_width: Width of one m/z aggregation window, in Da.
    :return: One record per m/z window that contains at least one ion.
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges
    annotated_total = statistics.annotated_relative[radius].sum(axis=1).astype(np.float64)  # (C,)
    unannotated_total = statistics.unannotated_relative[radius].sum(axis=1).astype(np.float64)  # (C,)
    negative = cumulative_below(
        statistics.unannotated_relative[radius], edges, float(relative_threshold)
    ).astype(np.float64)  # (C,)

    mz = np.where(np.isfinite(statistics.class_mz), statistics.class_mz, statistics.class_bin_centre)  # (C,)
    lower = np.floor(np.nanmin(mz) / window_width) * window_width
    upper = np.ceil(np.nanmax(mz) / window_width) * window_width
    boundaries = np.arange(lower, upper + window_width, window_width)

    records: list[dict[str, Any]] = []
    for position in range(boundaries.size - 1):
        selected = (mz >= boundaries[position]) & (mz < boundaries[position + 1])  # (C,)
        if not selected.any():
            continue
        annotated_sum = float(annotated_total[selected].sum())
        unannotated_sum = float(unannotated_total[selected].sum())
        negative_sum = float(negative[selected].sum())
        records.append({
            "mz_lower": float(boundaries[position]),
            "mz_upper": float(boundaries[position + 1]),
            "ions": int(selected.sum()),
            "bin_radius": int(bin_radius),
            "relative_threshold": float(relative_threshold),
            "annotated_entries": annotated_sum,
            "mean_prevalence": (
                annotated_sum / (annotated_sum + unannotated_sum) if annotated_sum + unannotated_sum else float("nan")
            ),
            "negative_fraction_of_unannotated": (
                negative_sum / unannotated_sum if unannotated_sum else float("nan")
            ),
        })
    return records
