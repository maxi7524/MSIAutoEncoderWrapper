"""How the P/N/U population responds to the evidence thresholds.

Every function here reads the histograms produced by
:mod:`.precompute` and returns long-form records, one row per swept value, so a
notebook can build a frame without reshaping anything. Nothing re-reads the image.

Two quantities carry the threshold decision:

*Negative yield* — the share of unannotated entries the rule converts into
operational negatives. It rises with :math:`\\beta` and is what the P/N and P/N/U
objectives actually train on.

*Positive contradiction* — the share of **annotated** entries whose own local
evidence falls at or below the same threshold. Those entries are protected by the
annotation (the rule assigns ``P`` before it looks at the signal), so they never
become negatives; they measure instead how often the rule disagrees with a label
that is known to be true. A threshold whose contradiction rate is high is calling
negative exactly the kind of entry that annotation shows can be positive, so the
negatives it produces cannot be trusted. This is the upper bound on :math:`\\beta`.

REMARK: Quantiles and rank statistics computed here have bucket resolution — they
are exact to the histogram boundary containing them, not to the underlying value.
Counts and the threshold sweep itself are exact, because every candidate threshold
is a boundary of the grid.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ....utils.logger import get_custom_logger
from .precompute import EvidenceStatistics, cumulative_below

logger = get_custom_logger(__name__)


def _sweep_thresholds(
    statistics: EvidenceStatistics,
    thresholds: Sequence[float] | None,
) -> np.ndarray:
    """Resolve the swept threshold values, defaulting to the whole grid."""
    if thresholds is None:
        return np.asarray(statistics.grid.relative_edges, dtype=np.float64)
    return np.asarray(thresholds, dtype=np.float64)


def _bucket_quantile(counts: np.ndarray, edges: np.ndarray, quantile: float) -> float:
    """Return the grid boundary at which a histogram reaches one cumulative share.

    :param counts: Bucket counts whose last axis matches ``edges.size + 1``.
    :param edges: The boundaries the histogram was built on.
    :param quantile: Cumulative share in ``[0, 1]``.
    :return: The smallest boundary whose cumulative count reaches the share, or
        ``nan`` for an empty histogram.
    :rtype: float
    """
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    cumulative = np.cumsum(counts) / total  # (K,)
    position = int(np.searchsorted(cumulative, quantile, side="left"))
    if position == 0:
        return 0.0
    if position > edges.size:
        return float(edges[-1])
    return float(edges[position - 1])


def state_yield_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Population-level P/N/U composition as a function of the relative threshold.

    One record per swept threshold, pooled over every ion and every pixel with an
    available label. ``negative`` and ``uncertain`` partition the unannotated
    entries; ``positive`` is fixed by the annotations and does not move with the
    threshold. ``positive_contradicted`` counts annotated entries whose evidence is
    at or below the threshold — see the module docstring.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read, one of the measured radii.
    :param thresholds: Threshold values, all of which must be grid boundaries.
        Defaults to the complete grid, giving the densest exact sweep available.
    :return: One record per threshold.
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges
    annotated = statistics.annotated_relative[radius].sum(axis=0)  # (K,)
    unannotated = statistics.unannotated_relative[radius].sum(axis=0)  # (K,)
    annotated_total = float(annotated.sum())
    unannotated_total = float(unannotated.sum())
    available_total = annotated_total + unannotated_total

    records: list[dict[str, Any]] = []
    for threshold in _sweep_thresholds(statistics, thresholds):
        negative = float(cumulative_below(unannotated, edges, float(threshold)))
        contradicted = float(cumulative_below(annotated, edges, float(threshold)))
        uncertain = unannotated_total - negative
        records.append({
            "bin_radius": int(bin_radius),
            "relative_threshold": float(threshold),
            "positive_entries": annotated_total,
            "negative_entries": negative,
            "uncertain_entries": uncertain,
            "negative_fraction_of_unannotated": negative / unannotated_total if unannotated_total else float("nan"),
            "uncertain_fraction_of_unannotated": uncertain / unannotated_total if unannotated_total else float("nan"),
            "negative_share_of_available": negative / available_total if available_total else float("nan"),
            "negative_to_positive_ratio": negative / annotated_total if annotated_total else float("nan"),
            "positive_contradicted_entries": contradicted,
            "positive_contradiction_rate": contradicted / annotated_total if annotated_total else float("nan"),
        })
    logger.info("Swept %s relative threshold(s) at bin_radius=%s.", len(records), bin_radius)
    return records


def class_state_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Per-ion P/N/U composition at a small set of candidate thresholds.

    Separates two failure modes a pooled count hides. An ion can be *rare* — few
    annotated pixels but a signal that stays above the threshold elsewhere, so most
    of its unannotated entries remain ``U``. Or it can be *undetectable* — its bin
    carries essentially no intensity anywhere, so nearly every unannotated entry
    becomes ``N`` and the class is trained almost entirely on negatives.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param thresholds: Candidate thresholds, all grid boundaries.
    :return: One record per (ion, threshold).
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges
    annotated = statistics.annotated_relative[radius]  # (C, K)
    unannotated = statistics.unannotated_relative[radius]  # (C, K)
    annotated_total = annotated.sum(axis=1).astype(np.float64)  # (C,)
    unannotated_total = unannotated.sum(axis=1).astype(np.float64)  # (C,)

    records: list[dict[str, Any]] = []
    for threshold in np.asarray(thresholds, dtype=np.float64):
        negative = cumulative_below(unannotated, edges, float(threshold)).astype(np.float64)  # (C,)
        contradicted = cumulative_below(annotated, edges, float(threshold)).astype(np.float64)  # (C,)
        for position, name in enumerate(statistics.class_names):
            available = annotated_total[position] + unannotated_total[position]
            records.append({
                "class_name": name,
                "mz": float(statistics.class_mz[position]),
                "bin_radius": int(bin_radius),
                "relative_threshold": float(threshold),
                "positive_entries": float(annotated_total[position]),
                "negative_entries": float(negative[position]),
                "uncertain_entries": float(unannotated_total[position] - negative[position]),
                "prevalence": annotated_total[position] / available if available else float("nan"),
                "negative_fraction_of_unannotated": (
                    negative[position] / unannotated_total[position] if unannotated_total[position] else float("nan")
                ),
                "negative_to_positive_ratio": (
                    negative[position] / annotated_total[position] if annotated_total[position] else float("inf")
                ),
                "positive_contradiction_rate": (
                    contradicted[position] / annotated_total[position] if annotated_total[position] else float("nan")
                ),
            })
    return records


def evidence_separation_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
) -> list[dict[str, Any]]:
    """Per-ion agreement between annotation and local spectral evidence.

    The evidence rule is only defensible if, within one ion, annotated pixels carry
    systematically more signal at that ion's bin than unannotated ones. The rank
    statistic

    .. math::

        \\mathrm{AUC}_c = \\Pr(r_{ic} > r_{jc}) + \\tfrac{1}{2}\\Pr(r_{ic} = r_{jc}),
        \\quad i \\in \\text{annotated}, \\; j \\in \\text{unannotated},

    measures exactly that, independently of any threshold. ``0.5`` means the signal
    at the ion's bin carries no information about whether the ion is annotated
    there, which invalidates *every* threshold for that ion, not just a badly chosen
    one. Values below ``0.5`` mean annotated pixels carry *less* signal, which points
    at a mapping or calibration problem rather than at chemistry.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :return: One record per ion.
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges
    annotated = statistics.annotated_relative[radius].astype(np.float64)  # (C, K)
    unannotated = statistics.unannotated_relative[radius].astype(np.float64)  # (C, K)

    # Rank statistic over shared buckets
    ## Entries in the same bucket are treated as ties, which is the bucket-resolution
    ## reading of the statistic and is conservative: it can only pull AUC toward 0.5.
    unannotated_below = np.cumsum(unannotated, axis=1) - unannotated  # (C, K)
    concordant = (annotated * (unannotated_below + 0.5 * unannotated)).sum(axis=1)  # (C,)
    annotated_total = annotated.sum(axis=1)  # (C,)
    unannotated_total = unannotated.sum(axis=1)  # (C,)
    pairs = annotated_total * unannotated_total  # (C,)
    auc = np.divide(concordant, pairs, out=np.full_like(concordant, np.nan), where=pairs > 0)  # (C,)

    records: list[dict[str, Any]] = []
    for position, name in enumerate(statistics.class_names):
        records.append({
            "class_name": name,
            "mz": float(statistics.class_mz[position]),
            "bin_centre": float(statistics.class_bin_centre[position]),
            "bin_radius": int(bin_radius),
            "positive_entries": float(annotated_total[position]),
            "unannotated_entries": float(unannotated_total[position]),
            "evidence_auc": float(auc[position]),
            "median_relative_annotated": _bucket_quantile(annotated[position], edges, 0.5),
            "median_relative_unannotated": _bucket_quantile(unannotated[position], edges, 0.5),
            "q05_relative_annotated": _bucket_quantile(annotated[position], edges, 0.05),
            "q95_relative_unannotated": _bucket_quantile(unannotated[position], edges, 0.95),
        })
    logger.info("Computed evidence separation for %s ion(s) at bin_radius=%s.", len(records), bin_radius)
    return records


def background_exceedance_records(statistics: EvidenceStatistics) -> list[dict[str, Any]]:
    """Reference curve: how many bins of an average spectrum exceed a relative level.

    Built over *every* bin of every pixel, not only the annotated ones. It answers
    the question a threshold has to be read against — at level :math:`\\beta`, how
    much of an ordinary spectrum is above it? A threshold below the noise floor
    leaves almost the whole axis above it, so essentially nothing becomes an
    operational negative; a threshold above the bulk of real peaks makes almost
    everything negative regardless of chemistry.

    :param statistics: A completed population pass.
    :return: One record per grid boundary.
    :rtype: list[dict[str, Any]]
    """
    edges = statistics.grid.relative_edges
    counts = statistics.background_relative.astype(np.float64)  # (K,)
    total = float(counts.sum())
    feature_count = float(statistics.metadata.get("feature_count", np.nan))

    records: list[dict[str, Any]] = []
    for threshold in edges:
        below = float(cumulative_below(counts, edges, float(threshold)))
        above_fraction = 1.0 - below / total if total else float("nan")
        records.append({
            "relative_threshold": float(threshold),
            "bins_at_or_below_fraction": below / total if total else float("nan"),
            "bins_above_fraction": above_fraction,
            "bins_above_per_spectrum": above_fraction * feature_count,
        })
    return records


def group_state_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Negative yield and positive contradiction per source acquisition.

    The train/validation/test split assigns whole acquisitions, so a threshold whose
    effect differs strongly between acquisitions produces splits that are trained and
    evaluated on different label semantics. This is the stability check on the
    selected value.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param thresholds: Candidate thresholds, all grid boundaries.
    :return: One record per (acquisition, threshold).
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges
    unannotated = statistics.group_relative[:, radius, 0]  # (G, K)
    annotated = statistics.group_relative[:, radius, 1]  # (G, K)
    unannotated_total = unannotated.sum(axis=1).astype(np.float64)  # (G,)
    annotated_total = annotated.sum(axis=1).astype(np.float64)  # (G,)

    records: list[dict[str, Any]] = []
    for threshold in np.asarray(thresholds, dtype=np.float64):
        negative = cumulative_below(unannotated, edges, float(threshold)).astype(np.float64)  # (G,)
        contradicted = cumulative_below(annotated, edges, float(threshold)).astype(np.float64)  # (G,)
        for position, name in enumerate(statistics.group_names):
            records.append({
                "group": name,
                "bin_radius": int(bin_radius),
                "relative_threshold": float(threshold),
                "pixels": int((statistics.spectrum_group_index == position).sum()),
                "positive_entries": float(annotated_total[position]),
                "negative_fraction_of_unannotated": (
                    negative[position] / unannotated_total[position] if unannotated_total[position] else float("nan")
                ),
                "positive_contradiction_rate": (
                    contradicted[position] / annotated_total[position] if annotated_total[position] else float("nan")
                ),
            })
    return records


def absolute_threshold_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """The same sweep for a pure absolute threshold, with the relative one disabled.

    Reports what the rule would do if it were driven by ``absolute_threshold`` alone
    (:math:`\\beta = 0`), on the TIC-normalized intensity scale the model sees. It is
    the alternative parameterization, and its yield curve is what makes the choice
    between a scale-free and a scale-fixed rule empirical rather than stylistic.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param thresholds: Absolute thresholds, all boundaries of the absolute grid.
    :return: One record per threshold.
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.absolute_edges
    annotated = statistics.annotated_absolute[radius].sum(axis=0)  # (K,)
    unannotated = statistics.unannotated_absolute[radius].sum(axis=0)  # (K,)
    annotated_total = float(annotated.sum())
    unannotated_total = float(unannotated.sum())

    records: list[dict[str, Any]] = []
    for threshold in (edges if thresholds is None else np.asarray(thresholds, dtype=np.float64)):
        negative = float(cumulative_below(unannotated, edges, float(threshold)))
        contradicted = float(cumulative_below(annotated, edges, float(threshold)))
        records.append({
            "bin_radius": int(bin_radius),
            "absolute_threshold": float(threshold),
            "negative_entries": negative,
            "negative_fraction_of_unannotated": negative / unannotated_total if unannotated_total else float("nan"),
            "positive_contradiction_rate": contradicted / annotated_total if annotated_total else float("nan"),
        })
    return records


def absolute_floor_records(
    statistics: EvidenceStatistics,
    *,
    relative_threshold: float,
    absolute_thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """How often an absolute floor would override the relative threshold.

    The rule applies :math:`\\tau_i = \\max(\\alpha, \\beta \\max_b x_{ib})`, so the
    absolute floor binds only in pixels whose maximum falls below
    :math:`\\alpha / \\beta`. That share is computed exactly from the retained
    per-pixel maxima, not from a histogram.

    :param statistics: A completed population pass.
    :param relative_threshold: The relative threshold :math:`\\beta` in force.
    :param absolute_thresholds: Candidate absolute floors :math:`\\alpha`.
    :return: One record per absolute floor.
    :rtype: list[dict[str, Any]]
    """
    maxima = statistics.spectrum_maximum.astype(np.float64)  # (N,)
    records: list[dict[str, Any]] = []
    for absolute in np.asarray(absolute_thresholds, dtype=np.float64):
        crossover = absolute / relative_threshold if relative_threshold > 0 else np.inf
        binding = float((maxima <= crossover).mean())
        records.append({
            "relative_threshold": float(relative_threshold),
            "absolute_threshold": float(absolute),
            "maximum_crossover": float(crossover),
            "pixels_with_binding_floor_fraction": binding,
            "pixels_with_binding_floor": int((maxima <= crossover).sum()),
        })
    return records


def relative_by_maximum_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Negative yield conditioned on the pixel's own dynamic range.

    The relative rule divides by the spectrum maximum, so it is only scale-free if
    the evidence ratio does not itself depend on that maximum. Pixels are stratified
    by their maximum and the negative yield recomputed inside each stratum; a strong
    trend means the same threshold behaves differently in weak and strong pixels,
    which is the argument for adding an absolute floor.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param thresholds: Candidate thresholds, all boundaries of the relative grid.
    :return: One record per (maximum stratum, threshold).
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    relative_edges = statistics.grid.relative_edges
    maximum_edges = statistics.grid.maximum_edges
    joint = statistics.joint_relative_maximum[radius].astype(np.float64)  # (K_r, K_m)

    records: list[dict[str, Any]] = []
    for stratum in range(joint.shape[1]):
        counts = joint[:, stratum]  # (K_r,)
        total = float(counts.sum())
        if total <= 0:
            continue
        lower = 0.0 if stratum == 0 else float(maximum_edges[stratum - 1])
        upper = float(maximum_edges[stratum]) if stratum < maximum_edges.size else float("inf")
        for threshold in np.asarray(thresholds, dtype=np.float64):
            negative = float(cumulative_below(counts, relative_edges, float(threshold)))
            records.append({
                "bin_radius": int(bin_radius),
                "relative_threshold": float(threshold),
                "maximum_lower": lower,
                "maximum_upper": upper,
                "unannotated_entries": total,
                "negative_fraction_of_unannotated": negative / total,
            })
    return records


def recommend_threshold(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    maximum_contradiction_rate: float = 0.01,
    candidates: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Select the largest threshold whose disagreement with annotations stays bounded.

    The selection rule is deliberately one-sided. Negative yield always increases
    with :math:`\\beta`, so there is no interior optimum to find; what bounds
    :math:`\\beta` from above is the rate at which the rule contradicts entries that
    annotation proves positive. The returned value is the largest swept threshold
    whose ``positive_contradiction_rate`` does not exceed the tolerance.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param maximum_contradiction_rate: Tolerated share of annotated entries falling
        at or below the threshold.
    :param candidates: Thresholds to choose among; defaults to the complete grid.
    :return: The selected threshold together with the yield it produces.
    :rtype: dict[str, Any]
    :raises ValueError: If no swept threshold satisfies the tolerance.
    """
    records = state_yield_records(statistics, bin_radius=bin_radius, thresholds=candidates)
    admissible = [
        record for record in records
        if record["positive_contradiction_rate"] <= maximum_contradiction_rate
    ]
    if not admissible:
        raise ValueError(
            f"No swept threshold keeps the positive contradiction rate at or below "
            f"{maximum_contradiction_rate} at bin_radius={bin_radius}."
        )
    selected = max(admissible, key=lambda record: record["relative_threshold"])
    logger.info(
        "Selected relative_threshold=%s at bin_radius=%s (contradiction=%.4f, negative yield=%.4f).",
        selected["relative_threshold"], bin_radius,
        selected["positive_contradiction_rate"], selected["negative_fraction_of_unannotated"],
    )
    return dict(selected) | {"maximum_contradiction_rate": float(maximum_contradiction_rate)}


def evidence_quantile_records(
    statistics: EvidenceStatistics,
    *,
    bin_radius: int = 1,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> list[dict[str, Any]]:
    """Bucket-resolution quantiles of the relative evidence, per annotation state.

    Reports the same three populations the distribution figure draws — annotated
    entries, unannotated entries and every bin of every spectrum — as a table, so the
    separation visible in the figure can be read as numbers. ``zero_share`` is the
    share of entries with exactly zero evidence, which no logarithmic axis can show
    and which is always below any threshold.

    :param statistics: A completed population pass.
    :param bin_radius: Dilation radius to read.
    :param quantiles: Cumulative shares to report.
    :return: One record per population.
    :rtype: list[dict[str, Any]]
    """
    radius = statistics.radius_index(bin_radius)
    edges = statistics.grid.relative_edges
    populations = {
        "annotated": statistics.annotated_relative[radius].sum(axis=0),
        "unannotated": statistics.unannotated_relative[radius].sum(axis=0),
        "background (all bins)": statistics.background_relative,
    }

    records: list[dict[str, Any]] = []
    for name, counts in populations.items():
        total = float(counts.sum())
        record = {
            "population": name,
            "bin_radius": int(bin_radius),
            "entries": total,
            "zero_share": float(counts[0] / total) if total else float("nan"),
        }
        for quantile in quantiles:
            record[f"q{int(round(quantile * 100)):02d}"] = _bucket_quantile(counts, edges, quantile)
        records.append(record)
    return records
