"""Threshold-free molecular ranking with explicit annotation/evidence populations.

Every quantity here is a sort followed by cumulative sums, so all classes are evaluated
simultaneously as tensor operations on one device rather than through a per-class
scikit-learn call. The per-class kernels are reused from :mod:`.batch_metrics`, whose
own tests assert that they reproduce scikit-learn exactly, including tie grouping and
undefined classes; this module adds the evaluation populations, the pooled aggregation
and the evidence-state diagnostics on top of them.

The reason is cost, not elegance. One scikit-learn call per class over 508 classes, two
populations, three splits and sixty-five checkpoints is roughly an hour of pure metric
time on a single core, while the model inference it accompanies takes seconds. The
reference implementation remains :mod:`.metrics` and :func:`positive_scores` keeps a
double-precision NumPy path for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from ....data.annotation_evidence import NEGATIVE, POSITIVE, UNAVAILABLE, UNLABELLED
from ....utils.logger import get_custom_logger
from .batch_metrics import _average_precision, _resolve_device, _roc_auc

logger = get_custom_logger(__name__)

# REMARK: Histogram edges must be identical for every model, otherwise the stored
# counts cannot be overlaid. Log odds are clipped into the outermost bins and the
# clipped entry counts are retained so a saturating head remains visible.
LOG_ODDS_RANGE = (-25.0, 25.0)
PROBABILITY_RANGE = (0.0, 1.0)
HISTOGRAM_BINS = 128
#: Fixed recall/false-positive-rate grid shared by every stored ranking curve.
CURVE_GRID = np.linspace(0.0, 1.0, 101)
STATE_LABELS = {POSITIVE: "P", NEGATIVE: "N", UNLABELLED: "U"}
#: Classes evaluated in one tensor pass. The per-class kernels allocate several
#: ``(N, chunk)`` intermediates, so this bounds peak device memory independently of
#: how many classes the catalogue holds.
CLASS_CHUNK = 128


def positive_scores(logits: np.ndarray) -> np.ndarray:
    """Return stable positive log odds for binary or N/P/U outputs.

    This is the double-precision reference path used by tests and by any caller that
    needs the scores themselves; the batched evaluation below builds the same
    quantity directly on the evaluation device in single precision.

    :param logits: Finite logits of shape ``(N, C)`` or ``(N, C, 3)``.
    :type logits: numpy.ndarray
    :return: Positive ranking scores, shape ``(N, C)``.
    :rtype: numpy.ndarray
    :raises ValueError: If dimensions or values are invalid.
    """
    values = np.asarray(logits, dtype=np.float64)
    _validate_logits(values)
    if values.ndim == 2:
        return values
    # REMARK: This is log(p_P / (1-p_P)), monotone in the existing positive
    # softmax probability. Ranking log odds avoids sigmoid saturation ties.
    return values[..., POSITIVE] - logsumexp(values[..., [NEGATIVE, UNLABELLED]], axis=-1)  # (N, C)


def _validate_logits(values: np.ndarray | torch.Tensor) -> None:
    """Reject non-finite values and shapes that are neither binary nor N/P/U."""
    finite = bool(torch.isfinite(values).all()) if torch.is_tensor(values) else bool(np.isfinite(values).all())
    if not finite:
        raise ValueError("Head logits must be finite.")
    if values.ndim == 2:
        return
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Expected binary (N, C) or N/P/U (N, C, 3) logits.")


def _score_tensor(logits: Any, device: torch.device) -> torch.Tensor:
    """Build the positive log odds on the evaluation device in single precision.

    :param logits: Binary ``(N, C)`` or N/P/U ``(N, C, 3)`` head outputs.
    :param device: Device every downstream reduction runs on.
    :return: Positive ranking scores, shape ``(N, C)``.
    :rtype: torch.Tensor
    :raises ValueError: If dimensions or values are invalid.
    """
    values = torch.as_tensor(np.asarray(logits), dtype=torch.float32, device=device)
    _validate_logits(values)
    if values.ndim == 2:
        return values
    return values[..., POSITIVE] - torch.logsumexp(values[..., [NEGATIVE, UNLABELLED]], dim=-1)  # (N, C)


def _to_unit_interval(scores: torch.Tensor) -> torch.Tensor:
    """Rescale each class column into ``[0, 1]`` without changing its ranking.

    The reused kernels push masked entries out of the way with the finite sentinels
    -1 and 2, which sort last only because a *probability* cannot leave ``[0, 1]``.
    The ranking scores here are unbounded log odds, so they must be mapped into that
    range first. The map is affine within a column, hence strictly increasing and
    tie-preserving: every per-class average precision and area under the curve is
    unchanged, while the sentinels regain their meaning. Scaling per column rather
    than globally keeps the full float32 resolution available to each class.

    :param scores: Unbounded ranking scores, shape ``(N, C)``.
    :type scores: torch.Tensor
    :return: Column-wise rescaled scores in ``[0, 1]``, shape ``(N, C)``. A constant
        column maps to a constant, which is already an all-tied ranking.
    :rtype: torch.Tensor
    """
    lowest = scores.amin(dim=0, keepdim=True)  # (1, C)
    span = scores.amax(dim=0, keepdim=True) - lowest  # (1, C)
    return torch.where(span > 0, (scores - lowest) / span.clamp(min=torch.finfo(scores.dtype).tiny),
                       torch.full_like(scores, 0.5))  # (N, C)


def _per_class_ranking(scores: torch.Tensor, targets: torch.Tensor, availability: torch.Tensor,
                       *, chunk: int = CLASS_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate every class in a few tensor passes, in bounded-memory class chunks.

    :param scores: Ranking scores, shape ``(N, C)``; any monotone scale is accepted.
    :param targets: Binary annotations as floats, shape ``(N, C)``.
    :param availability: Retained-entry indicator as floats, shape ``(N, C)``.
    :param chunk: Classes evaluated together.
    :return: Per-class average precision and ROC AUC, both shape ``(C,)``; undefined
        classes are ``nan``, exactly as the scikit-learn reference leaves them.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    precisions, areas = [], []
    for start in range(0, scores.shape[1], chunk):
        stop = start + chunk
        block_scores = _to_unit_interval(scores[:, start:stop])  # (N, chunk)
        block_targets = targets[:, start:stop]  # (N, chunk)
        block_available = availability[:, start:stop]  # (N, chunk)
        positive_counts = (block_targets * block_available).sum(dim=0)  # (chunk,)
        available_counts = block_available.sum(dim=0)  # (chunk,)
        precisions.append(_average_precision(block_scores, block_targets, block_available, positive_counts))
        areas.append(_roc_auc(block_scores, block_targets, block_available, positive_counts, available_counts))
    return torch.cat(precisions), torch.cat(areas)  # (C,), (C,)


def pooled_ranking(labels: Any, scores: Any, *, grid: np.ndarray = CURVE_GRID, device: Any = None) -> dict:
    """Rank every retained (pixel, class) entry jointly instead of per class.

    The macro convention averages one curve per class and therefore weights a class
    with three positives exactly like a class with three thousand. The pooled, or
    *micro*, convention concatenates all retained entries into a single ranking
    problem, so frequent classes dominate. Reporting both localizes an effect: a
    macro gain with a flat micro value is carried by rare classes, and the reverse
    pattern is carried by frequent ones.

    Average precision is :math:`\\sum_n (R_n - R_{n-1}) P_n` over successive distinct
    scores, and ROC AUC is the trapezoidal area under the (FPR, TPR) curve, which is
    the standard tie-aware convention. Both are computed from one descending sort;
    entries sharing a score form one threshold group, so ties neither inflate nor
    deflate either quantity.

    :param labels: Binary targets of the retained entries, shape ``(E,)``.
    :param scores: Ranking scores of the same entries, shape ``(E,)``.
    :param grid: Increasing recall/false-positive-rate coordinates for the curves.
    :param device: Evaluation device; CUDA is used when available and none is given.
    :return: ``average_precision``, ``roc_auc``, ``prevalence``, ``entries`` and the
        precision/true-positive-rate values interpolated onto ``grid``.
    :rtype: dict
    :raises ValueError: If the two arrays disagree or labels are not binary.
    """
    resolved = _resolve_device(device)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=resolved).reshape(-1)
    s = torch.as_tensor(np.asarray(scores), dtype=torch.float32, device=resolved).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("Pooled ranking needs one score per label.")
    if y.numel() and not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("Pooled ranking needs binary labels.")
    positives, entries = int(y.sum().item()), int(y.numel())
    if positives == 0 or positives == entries:
        return {"average_precision": np.nan, "roc_auc": np.nan,
                "prevalence": positives / entries if entries else np.nan, "entries": entries,
                "precision_at_recall": np.full(len(grid), np.nan),
                "tpr_at_fpr": np.full(len(grid), np.nan)}
    ## One descending sort; entries sharing a score become one threshold group
    order = torch.argsort(s, descending=True, stable=True)  # (E,)
    ordered_scores = s[order]  # (E,)
    ordered_labels = y[order]  # (E,)
    is_last = torch.cat([ordered_scores[:-1] != ordered_scores[1:],
                         torch.ones(1, dtype=torch.bool, device=resolved)])  # (E,)
    thresholds = torch.nonzero(is_last, as_tuple=False).reshape(-1)  # (T,)
    true_positives = torch.cumsum(ordered_labels, dim=0)[thresholds].double()  # (T,)
    ranks = (thresholds + 1).double()  # (T,)
    negatives = entries - positives
    recall = (true_positives / positives).cpu().numpy()  # (T,)
    precision = (true_positives / ranks).cpu().numpy()  # (T,)
    false_positive_rate = ((ranks - true_positives) / negatives).cpu().numpy()  # (T,)
    ## Summary statistics under the standard conventions
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    roc_auc = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, false_positive_rate]))
    ### REMARK: Both curves are step functions. Interpolating the running maximum of
    ### precision to the right of each recall level reproduces the achievable operating
    ### points; plain linear interpolation would invent points between two thresholds.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]  # (T,)
    precision_at_recall = np.interp(grid, np.r_[0.0, recall], np.r_[envelope[0], envelope])  # (G,)
    tpr_at_fpr = np.interp(grid, np.r_[0.0, false_positive_rate], np.r_[0.0, recall])  # (G,)
    return {"average_precision": average_precision, "roc_auc": roc_auc,
            "prevalence": positives / entries, "entries": entries,
            "precision_at_recall": precision_at_recall, "tpr_at_fpr": tpr_at_fpr}


def _validate_alignment(scores: torch.Tensor, targets: np.ndarray, states: np.ndarray,
                        train_counts: np.ndarray, class_names: tuple[str, ...]) -> None:
    """Reject misaligned catalogues, invalid codes and inconsistent evidence positives."""
    if (tuple(scores.shape) != targets.shape or states.shape != targets.shape
            or len(class_names) != scores.shape[1] or train_counts.shape != (scores.shape[1],)):
        raise ValueError("Scores, annotations, states and class catalogue must align.")
    if not np.isin(targets, [0, 1]).all() or not np.isin(states, [-1, 0, 1, 2]).all():
        raise ValueError("Invalid binary annotations or evidence-state codes.")
    available = states != UNAVAILABLE  # (N, C)
    if not np.array_equal((states == POSITIVE)[available], targets[available].astype(bool)):
        raise ValueError("Evidence positives must preserve available annotations exactly.")


def _frequency_labels(train_counts: np.ndarray) -> np.ndarray:
    """Group classes by training support with fixed, pre-declared boundaries."""
    return np.where(train_counts < 10, "rare", np.where(train_counts < 100, "medium", "frequent"))  # (C,)


def ranking_tables(
    logits: np.ndarray,
    targets: np.ndarray,
    states: np.ndarray,
    train_counts: np.ndarray,
    class_names: tuple[str, ...],
    *,
    device: Any = None,
) -> dict[str, pd.DataFrame]:
    """Measure ranking under two fixed evaluation populations, without a cutoff.

    :param logits: Active head outputs, ``(N, C)`` or ``(N, C, 3)``.
    :param targets: Original binary annotations, ``(N, C)``.
    :param states: Shared evidence states, ``(N, C)``; -1 excludes unavailable entries.
    :param train_counts: Observed training positives per class, ``(C,)``.
    :param class_names: Stable names in target-column order.
    :param device: Evaluation device; CUDA is used when available and none is given.
    :return: Aggregate, per-class, pooled-curve and per-state diagnostic DataFrames.
    :rtype: dict[str, pandas.DataFrame]
    :raises ValueError: If shapes, labels or evidence semantics disagree.
    """
    resolved = _resolve_device(device)
    scores = _score_tensor(logits, resolved)  # (N, C)
    targets, states = np.asarray(targets), np.asarray(states)
    train_counts = np.asarray(train_counts)
    _validate_alignment(scores, targets, states, train_counts, class_names)
    target_tensor = torch.as_tensor(targets != 0, dtype=torch.float32, device=resolved)  # (N, C)
    state_tensor = torch.as_tensor(states, dtype=torch.int8, device=resolved)  # (N, C)
    available = (state_tensor != UNAVAILABLE).to(torch.float32)  # (N, C)
    masks = {
        "annotation_retrieval": available,
        "operational_pn": ((state_tensor == POSITIVE) | (state_tensor == NEGATIVE)).to(torch.float32),
    }
    frequency = _frequency_labels(train_counts)  # (C,)
    rows, aggregate, curves = [], [], []
    # Per-class evaluation, retaining undefined values and their denominators
    for population, mask in masks.items():
        positives = (target_tensor * mask).sum(dim=0)  # (C,)
        retained = mask.sum(dim=0)  # (C,)
        precision, area = _per_class_ranking(scores, target_tensor, mask)
        positives_numpy = positives.cpu().numpy().astype(np.int64)  # (C,)
        retained_numpy = retained.cpu().numpy().astype(np.int64)  # (C,)
        negatives_numpy = retained_numpy - positives_numpy  # (C,)
        # REMARK: AP=1 for a positive-only class is algebraically valid but
        # carries no discrimination evidence; exclude it from macro ranking.
        eligible = (train_counts > 0) & (positives_numpy > 0) & (negatives_numpy > 0)  # (C,)
        prevalence = np.divide(positives_numpy, retained_numpy, out=np.full(len(class_names), np.nan),
                               where=retained_numpy > 0)  # (C,)
        frame = pd.DataFrame({
            "population": population, "class_index": np.arange(len(class_names)),
            "class_name": list(class_names), "train_positives": train_counts.astype(np.int64),
            "positives": positives_numpy, "negatives": negatives_numpy, "available": retained_numpy,
            "eligible": eligible, "frequency": frequency, "prevalence": prevalence,
            "average_precision": np.where(eligible, precision.cpu().numpy(), np.nan),
            "roc_auc": np.where(eligible, area.cpu().numpy(), np.nan),
        })
        frame["ap_above_prevalence"] = frame.average_precision - frame.prevalence
        rows.append(frame)
        ## Macro aggregation: one curve per class, every class weighted equally
        for scope in ("train_supported", "rare", "medium", "frequent"):
            selected = frame[frame.eligible & ((frame.frequency == scope) if scope != "train_supported" else True)]
            for metric in ("average_precision", "roc_auc", "ap_above_prevalence"):
                aggregate.append({"population": population, "scope": scope, "metric": metric,
                                  "value": selected[metric].mean(), "classes": len(selected),
                                  "entries": np.nan})
        ## Micro aggregation: one ranking over the pooled entries of eligible classes
        eligible_tensor = torch.as_tensor(eligible, device=resolved)  # (C,)
        pooled_mask = (mask > 0) & eligible_tensor.unsqueeze(0)  # (N, C)
        pooled = pooled_ranking(target_tensor[pooled_mask].cpu().numpy(),
                                scores[pooled_mask].cpu().numpy(), device=resolved)
        for metric in ("average_precision", "roc_auc", "prevalence"):
            aggregate.append({"population": population, "scope": "train_supported",
                              "metric": f"micro_{metric}", "value": pooled[metric],
                              "classes": int(eligible.sum()), "entries": pooled["entries"]})
        for name, y_values in (("precision_recall", pooled["precision_at_recall"]),
                               ("roc", pooled["tpr_at_fpr"])):
            curves.append(pd.DataFrame({"population": population, "curve": name, "x": CURVE_GRID,
                                        "y": y_values, "classes": int(eligible.sum()),
                                        "entries": pooled["entries"]}))
    # Score spread is a diagnostic, not evidence of molecular calibration
    probabilities = torch.sigmoid(scores)  # (N, C)
    quantiles = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=resolved)
    diagnostics = []
    for state, label in STATE_LABELS.items():
        selected = state_tensor == state  # (N, C)
        for quantity, array in (("positive_probability", probabilities), ("positive_log_odds", scores)):
            values = array[selected]  # (E_state,)
            # REMARK: torch.quantile caps its input size, so the reduction runs on the
            # sorted values directly and stays on the evaluation device.
            located = (torch.sort(values).values[
                (quantiles * (values.numel() - 1)).round().long()].cpu().numpy()
                if values.numel() else np.full(len(quantiles), np.nan))
            diagnostics.extend({"state": label, "quantity": quantity, "quantile": float(quantile),
                                "value": float(value), "entries": int(values.numel())}
                               for quantile, value in zip(quantiles.cpu().numpy(), located))
    logger.info("Evaluated %s classes in two ranking populations on %s.", scores.shape[1], resolved)
    return {"prediction": pd.DataFrame(aggregate), "per_class": pd.concat(rows, ignore_index=True),
            "ranking_curves": pd.concat(curves, ignore_index=True),
            "score_diagnostics": pd.DataFrame(diagnostics)}


def state_separation(
    logits: np.ndarray,
    states: np.ndarray,
    train_counts: np.ndarray,
    class_names: tuple[str, ...],
    *,
    device: Any = None,
) -> pd.DataFrame:
    """Measure where each head places evidence-negative and unlabelled entries.

    The two ranking populations both treat an entry as either a positive or a
    comparison entry. They therefore cannot say whether a head *distinguishes* the
    two kinds of comparison entry: an operational negative N, where the spectrum
    carries no signal at the ion's bins, and an unlabelled entry U, where the
    annotation is missing but a signal is present. This function reports the two
    remaining orderings directly, per class:

    - ``positive_vs_uncertain_auc``: probability that a random P outranks a random U;
    - ``negative_vs_uncertain_auc``: probability that a random N outranks a random U.

    A head that has genuinely learned the ion places P above U above N, so the first
    value exceeds 0.5 and the second falls below it. A value near 0.5 means the head
    orders that pair no better than chance; ``negative_vs_uncertain_auc`` above 0.5
    means N is scored *higher* than U, which contradicts the evidence rule.

    :param logits: Active head outputs, ``(N, C)`` or ``(N, C, 3)``.
    :type logits: numpy.ndarray
    :param states: Shared evidence states, ``(N, C)``.
    :type states: numpy.ndarray
    :param train_counts: Observed training positives per class, ``(C,)``.
    :type train_counts: numpy.ndarray
    :param class_names: Stable names in target-column order.
    :type class_names: tuple[str, ...]
    :param device: Evaluation device; CUDA is used when available and none is given.
    :type device: Any
    :return: One row per class with state counts, per-state score location and the
        two state-pair areas under the curve; undefined pairs remain NaN.
    :rtype: pandas.DataFrame
    :raises ValueError: If shapes or evidence-state codes are invalid.
    """
    resolved = _resolve_device(device)
    scores = _score_tensor(logits, resolved)  # (N, C)
    states = np.asarray(states)
    train_counts = np.asarray(train_counts)
    if tuple(scores.shape) != states.shape or len(class_names) != scores.shape[1] or train_counts.shape != (scores.shape[1],):
        raise ValueError("Scores, states and class catalogue must align.")
    if not np.isin(states, [-1, 0, 1, 2]).all():
        raise ValueError("Invalid evidence-state codes.")
    state_tensor = torch.as_tensor(states, dtype=torch.int8, device=resolved)  # (N, C)
    result = pd.DataFrame({"class_index": np.arange(len(class_names)), "class_name": list(class_names),
                           "train_positives": train_counts.astype(np.int64),
                           "frequency": _frequency_labels(train_counts)})
    ## Per-state occupancy and score location
    for state, label in STATE_LABELS.items():
        membership = (state_tensor == state).to(torch.float32)  # (N, C)
        counts = membership.sum(dim=0)  # (C,)
        masked = torch.where(membership > 0, scores, torch.full_like(scores, float("nan")))  # (N, C)
        result[f"{label}_entries"] = counts.cpu().numpy().astype(np.int64)
        result[f"{label}_score_median"] = torch.nanmedian(masked, dim=0).values.cpu().numpy()
        result[f"{label}_score_mean"] = (torch.nansum(masked, dim=0) / counts.clamp(min=1)).cpu().numpy()
        result.loc[result[f"{label}_entries"] == 0, [f"{label}_score_median", f"{label}_score_mean"]] = np.nan
    unannotated = result.N_entries + result.U_entries
    result["negative_share_of_unannotated"] = np.divide(
        result.N_entries, unannotated, out=np.full(len(result), np.nan), where=unannotated > 0)
    ## Ordering of the two comparison states relative to each other and to the positives
    for column, (high, low) in (("positive_vs_uncertain_auc", (POSITIVE, UNLABELLED)),
                                ("negative_vs_uncertain_auc", (NEGATIVE, UNLABELLED))):
        pair = ((state_tensor == high) | (state_tensor == low)).to(torch.float32)  # (N, C)
        higher = (state_tensor == high).to(torch.float32)  # (N, C)
        positive_counts = higher.sum(dim=0)  # (C,)
        area = torch.cat([_roc_auc(_to_unit_interval(scores[:, start:start + CLASS_CHUNK]),
                                   higher[:, start:start + CLASS_CHUNK], pair[:, start:start + CLASS_CHUNK],
                                   positive_counts[start:start + CLASS_CHUNK],
                                   pair[:, start:start + CLASS_CHUNK].sum(dim=0))
                          for start in range(0, scores.shape[1], CLASS_CHUNK)])  # (C,)
        result[column] = area.cpu().numpy()
    logger.info("Measured evidence-state separation for %s classes on %s.", len(class_names), resolved)
    return result


def score_histograms(
    logits: np.ndarray,
    states: np.ndarray,
    train_counts: np.ndarray,
    *,
    bins: int = HISTOGRAM_BINS,
    device: Any = None,
) -> pd.DataFrame:
    """Retain complete score distributions per evidence state on shared fixed edges.

    Every stored count uses the same edges for every model and split, so the stored
    counts can be overlaid without re-running inference. Log odds outside
    :data:`LOG_ODDS_RANGE` are counted in ``below_range``/``above_range`` rather than
    silently dropped, which keeps a saturating head visible.

    :param logits: Active head outputs, ``(N, C)`` or ``(N, C, 3)``.
    :type logits: numpy.ndarray
    :param states: Shared evidence states, ``(N, C)``.
    :type states: numpy.ndarray
    :param train_counts: Observed training positives per class, ``(C,)``.
    :type train_counts: numpy.ndarray
    :param bins: Number of equal-width bins per quantity.
    :type bins: int
    :param device: Evaluation device; CUDA is used when available and none is given.
    :type device: Any
    :return: Long-form counts indexed by quantity, evidence state and class scope.
    :rtype: pandas.DataFrame
    :raises ValueError: If shapes disagree or the bin count is not positive.
    """
    if bins < 1:
        raise ValueError("A positive bin count is required.")
    resolved = _resolve_device(device)
    scores = _score_tensor(logits, resolved)  # (N, C)
    states = np.asarray(states)
    train_counts = np.asarray(train_counts)
    if tuple(scores.shape) != states.shape or train_counts.shape != (scores.shape[1],):
        raise ValueError("Scores, states and training counts must align.")
    state_tensor = torch.as_tensor(states, dtype=torch.int8, device=resolved)  # (N, C)
    frequency = _frequency_labels(train_counts)  # (C,)
    scopes = {"all": np.ones_like(frequency, dtype=bool)}
    scopes.update({name: frequency == name for name in ("rare", "medium", "frequent")})
    quantities = {"positive_log_odds": (scores, LOG_ODDS_RANGE),
                  "positive_probability": (torch.sigmoid(scores), PROBABILITY_RANGE)}
    frames = []
    for quantity, (array, value_range) in quantities.items():
        edges = np.linspace(*value_range, bins + 1)  # (B+1,)
        low, high = float(value_range[0]), float(value_range[1])
        for state, label in STATE_LABELS.items():
            for scope, columns in scopes.items():
                selection = (state_tensor == state) & torch.as_tensor(columns, device=resolved).unsqueeze(0)
                values = array[selection]  # (E,)
                below = int((values < low).sum().item())
                above = int((values > high).sum().item())
                counts = (torch.histc(values.clamp(low, high), bins=bins, min=low, max=high)
                          if values.numel() else torch.zeros(bins, device=resolved))  # (B,)
                frames.append(pd.DataFrame({"quantity": quantity, "state": label, "scope": scope,
                                            "left_edge": edges[:-1], "right_edge": edges[1:],
                                            "count": counts.cpu().numpy().astype(np.int64),
                                            "entries": int(values.numel()),
                                            "below_range": below, "above_range": above}))
    logger.info("Stored score histograms over %s shared bins per quantity on %s.", bins, resolved)
    return pd.concat(frames, ignore_index=True)


def generalization_gaps(per_class: pd.DataFrame) -> pd.DataFrame:
    """Compare splits on identical eligible class support within each model.

    :param per_class: Per-class records with model, split and population identity.
    :type per_class: pandas.DataFrame
    :return: Train-minus-held-out gaps, matched on class before averaging.
    :rtype: pandas.DataFrame
    """
    rows = []
    for (model_id, population), group in per_class.groupby(["model_id", "population"]):
        for held_out in ("validation", "test"):
            left = group[(group.split == "train") & group.eligible]
            right = group[(group.split == held_out) & group.eligible]
            paired = left.merge(right, on="class_index", suffixes=("_train", "_held"), validate="one_to_one")
            for metric in ("average_precision", "roc_auc"):
                rows.append({"model_id": model_id, "population": population, "held_out": held_out,
                             "metric": metric, "classes": len(paired),
                             "train_value": paired[f"{metric}_train"].mean(),
                             "held_out_value": paired[f"{metric}_held"].mean(),
                             "value": (paired[f"{metric}_train"] - paired[f"{metric}_held"]).mean()})
    return pd.DataFrame(rows)


def class_agreement(per_class: pd.DataFrame, *, metric: str = "average_precision") -> pd.DataFrame:
    """Compare two conditions class by class on the classes both of them can rank.

    A macro mean hides whether two heads succeed on the *same* ions. This pairs the
    per-class values of every ordered condition pair on shared eligible classes, so
    a difference in the macro mean can be attributed either to a broad shift or to a
    small set of ions.

    :param per_class: Per-class records carrying ``label``, ``split`` and ``population``.
    :type per_class: pandas.DataFrame
    :param metric: Per-class column compared between the two conditions.
    :type metric: str
    :return: One row per (condition pair, split, population, class) with both values
        and their difference, plus the class properties needed for interpretation.
    :rtype: pandas.DataFrame
    :raises ValueError: If the requested metric column is absent.
    """
    if metric not in per_class:
        raise ValueError(f"Per-class records do not contain '{metric}'.")
    columns = ["class_index", "class_name", "train_positives", "frequency", "prevalence", "eligible", metric]
    ## Average repeated runs of one condition before comparing conditions
    averaged = (per_class[["label", "split", "population", *columns]]
                .groupby(["label", "split", "population", "class_index", "class_name",
                          "train_positives", "frequency"], as_index=False)
                .agg(prevalence=("prevalence", "mean"), eligible=("eligible", "all"), value=(metric, "mean")))
    rows = []
    labels = sorted(averaged.label.unique())
    for position, left in enumerate(labels):
        for right in labels[position + 1:]:
            a = averaged[averaged.label == left]
            b = averaged[averaged.label == right]
            paired = a.merge(b, on=["split", "population", "class_index", "class_name",
                                    "train_positives", "frequency"], suffixes=("_left", "_right"),
                             validate="one_to_one")
            paired = paired[paired.eligible_left & paired.eligible_right]
            rows.append(paired.assign(left=left, right=right, metric=metric,
                                      difference=paired.value_left - paired.value_right))
    if not rows:
        return pd.DataFrame(columns=["left", "right", "metric", "difference"])
    result = pd.concat(rows, ignore_index=True)
    logger.info("Paired %s class-level comparisons across %s conditions.", len(result), len(labels))
    return result.drop(columns=["eligible_left", "eligible_right"])
