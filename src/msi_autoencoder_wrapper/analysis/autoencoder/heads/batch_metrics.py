"""Vectorized multi-label head metrics, evaluated on one device in a few tensor passes.

:mod:`.metrics` evaluates a multi-label head by calling scikit-learn once per class,
which costs one Python-level call and one sort per class. Sweeping tens of models over
several hundred classes turns that into the dominant cost of an analysis whose actual
model inference is negligible: measured at 4.5 s for a single ``(1000, 500)``
evaluation, so 560 of them, the shape of a two-split two-scope sweep over 140 models,
is roughly 40 minutes of pure metric time on one core.

Every metric involved is a sort followed by cumulative sums, so all classes can be
evaluated simultaneously as tensor operations, on a GPU when one is available. This
module is that reimplementation, and it is a deliberate second implementation of
quantities :mod:`.metrics` already computes: it exists only for speed, its results are
required to match that module exactly, and ``tests/analysis/test_batch_metrics.py``
asserts that agreement against scikit-learn including the tie and empty-class edge
cases. Use :mod:`.metrics` as the reference; use this when the class count or the model
count makes the reference too slow.

Conventions are inherited verbatim from :func:`.metrics.evaluate_head`:

- macro precision, recall and F1 are plain means over every class, with a zero
  contribution from a class whose denominator is zero;
- ``average_precision`` and ``roc_auc`` are means that skip undefined classes, the
  former undefined without an available positive, the latter also without an available
  negative;
- ties in the score are grouped exactly as scikit-learn groups them, which matters
  because a saturating logistic produces exactly equal probabilities at both ends.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

# Sentinels that sort masked entries to the end without colliding with a probability,
# which always lies in [0, 1]. Using finite values keeps every downstream arithmetic
# operation finite, unlike an infinity.
_SORT_LAST_DESCENDING = -1.0
_SORT_LAST_ASCENDING = 2.0


def _resolve_device(device: Optional[Any]) -> torch.device:
    """Pick the evaluation device, preferring CUDA when the caller gave none."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tie_group_bounds(sorted_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate, for every sorted entry, the first and last index of its tie group.

    Entries sharing a score are contiguous once sorted, so a tie group is an index
    range. Both bounds are obtained with running maxima rather than a Python loop, so
    the cost stays one pass over the tensor.

    :param sorted_scores: Scores sorted along dimension 0, shape ``(N, C)``.
    :type sorted_scores: torch.Tensor
    :return: First and last index of each entry's group, both shape ``(N, C)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    sample_count = sorted_scores.shape[0]
    positions = torch.arange(sample_count, device=sorted_scores.device).unsqueeze(1)
    positions = positions.expand_as(sorted_scores)

    ## Group boundaries
    changes = sorted_scores[:-1] != sorted_scores[1:]  # (N-1, C)
    leading = torch.ones_like(sorted_scores[:1], dtype=torch.bool)
    is_first = torch.cat([leading, changes], dim=0)  # (N, C)
    is_last = torch.cat([changes, leading], dim=0)  # (N, C)

    ## Index of the most recent group start, at or before each entry
    group_first = torch.cummax(torch.where(is_first, positions, -1), dim=0).values

    ### REMARK: the group end needs the *nearest* boundary at or after each entry, so
    ### the running maximum has to be taken over reversed coordinates and mapped back.
    ### Taking it over original indices instead returns the furthest boundary, which is
    ### the last group of the column rather than the entry's own.
    reversed_is_last = torch.flip(is_last, dims=[0])
    reversed_last = torch.cummax(
        torch.where(reversed_is_last, positions, -1), dim=0
    ).values  # largest reversed coordinate at or before each reversed entry
    group_last = torch.flip(sample_count - 1 - reversed_last, dims=[0])
    return group_first, group_last


def _average_precision(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    availability: torch.Tensor,
    positive_counts: torch.Tensor,
) -> torch.Tensor:
    """Average precision per class, matching scikit-learn's tie grouping.

    Scikit-learn evaluates the precision-recall curve at one point per distinct score
    and sums $\\sum_n (R_n - R_{n-1}) P_n$. Within a tie group the recall gain is the
    group's positive count and the precision is the value at the group's end, so the
    sum rearranges into one term per positive entry, each weighted by the precision at
    the end of its own tie group. That form has no per-class loop.

    :param probabilities: Predicted probabilities, shape ``(N, C)``.
    :type probabilities: torch.Tensor
    :param targets: Binary ground truth, shape ``(N, C)``.
    :type targets: torch.Tensor
    :param availability: Availability mask, shape ``(N, C)``.
    :type availability: torch.Tensor
    :param positive_counts: Available positives per class, shape ``(C,)``.
    :type positive_counts: torch.Tensor
    :return: Average precision per class, ``nan`` where undefined, shape ``(C,)``.
    :rtype: torch.Tensor
    """
    ## Masked entries sort last as negatives, where they cannot alter any earlier point
    available = availability > 0
    scores = torch.where(available, probabilities, torch.full_like(probabilities, _SORT_LAST_DESCENDING))
    labels = torch.where(available, targets, torch.zeros_like(targets))

    order = torch.argsort(scores, dim=0, descending=True, stable=True)  # (N, C)
    sorted_scores = torch.gather(scores, 0, order)  # (N, C)
    sorted_labels = torch.gather(labels, 0, order)  # (N, C)

    ## Precision at every prefix, then at the end of each tie group
    true_positives = torch.cumsum(sorted_labels, dim=0)  # (N, C)
    ranks = torch.arange(
        1, sorted_scores.shape[0] + 1, device=scores.device, dtype=true_positives.dtype
    ).unsqueeze(1)
    precision = true_positives / ranks  # (N, C)
    _, group_last = _tie_group_bounds(sorted_scores)
    precision_at_group_end = torch.gather(precision, 0, group_last)  # (N, C)

    total = (sorted_labels * precision_at_group_end).sum(dim=0)  # (C,)
    average_precision = total / positive_counts.clamp(min=1)
    return torch.where(
        positive_counts > 0, average_precision, torch.full_like(average_precision, float("nan"))
    )


def _roc_auc(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    availability: torch.Tensor,
    positive_counts: torch.Tensor,
    available_counts: torch.Tensor,
) -> torch.Tensor:
    """ROC AUC per class as a rank statistic, averaging ranks within tie groups.

    The area equals the Mann-Whitney statistic
    $(\\sum_{i \\in +} r_i - P(P+1)/2) / (P \\cdot N)$ over available entries, where
    $r_i$ are ascending ranks and tied entries share the mean rank of their group,
    which is what reproduces scikit-learn's trapezoidal treatment of ties.

    :param probabilities: Predicted probabilities, shape ``(N, C)``.
    :type probabilities: torch.Tensor
    :param targets: Binary ground truth, shape ``(N, C)``.
    :type targets: torch.Tensor
    :param availability: Availability mask, shape ``(N, C)``.
    :type availability: torch.Tensor
    :param positive_counts: Available positives per class, shape ``(C,)``.
    :type positive_counts: torch.Tensor
    :param available_counts: Available entries per class, shape ``(C,)``.
    :type available_counts: torch.Tensor
    :return: ROC AUC per class, ``nan`` where undefined, shape ``(C,)``.
    :rtype: torch.Tensor
    """
    ## Masked entries sort last as negatives; they never enter an available rank
    available = availability > 0
    scores = torch.where(available, probabilities, torch.full_like(probabilities, _SORT_LAST_ASCENDING))
    labels = torch.where(available, targets, torch.zeros_like(targets))

    order = torch.argsort(scores, dim=0, descending=False, stable=True)
    sorted_scores = torch.gather(scores, 0, order)
    sorted_labels = torch.gather(labels, 0, order)

    group_first, group_last = _tie_group_bounds(sorted_scores)
    average_rank = (group_first + group_last).to(sorted_labels.dtype) / 2.0 + 1.0  # (N, C)

    positive_rank_sum = (sorted_labels * average_rank).sum(dim=0)  # (C,)
    negative_counts = available_counts - positive_counts
    pair_count = (positive_counts * negative_counts).clamp(min=1)
    area = (positive_rank_sum - positive_counts * (positive_counts + 1) / 2.0) / pair_count
    defined = (positive_counts > 0) & (negative_counts > 0)
    return torch.where(defined, area, torch.full_like(area, float("nan")))


def _threshold_counts(
    predicted: torch.Tensor, targets: torch.Tensor, availability: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class true positive, false positive and false negative counts."""
    positive_truth = targets * availability
    positive_prediction = predicted * availability
    true_positives = (positive_prediction * positive_truth).sum(dim=0)
    false_positives = (positive_prediction * (1.0 - positive_truth)).sum(dim=0)
    false_negatives = ((1.0 - positive_prediction) * positive_truth).sum(dim=0)
    return true_positives, false_positives, false_negatives


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Ratio with a zero result where the denominator vanishes, as ``zero_division=0``."""
    return torch.where(
        denominator > 0, numerator / denominator.clamp(min=1), torch.zeros_like(numerator)
    )


def _prepare(
    logits: Any,
    targets: Any,
    mask: Optional[Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move probabilities, targets and availability onto one device as ``(N, C)``."""
    logit_tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float32, device=device)
    probabilities = torch.sigmoid(logit_tensor)  # (N, C)
    target_tensor = torch.as_tensor(
        np.asarray(targets), dtype=torch.float32, device=device
    )
    target_tensor = (target_tensor != 0).to(torch.float32)  # (N, C)

    if mask is None:
        availability = torch.ones_like(target_tensor)
    else:
        mask_array = np.asarray(mask)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.float32, device=device)
        mask_tensor = (mask_tensor != 0).to(torch.float32)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(1).expand_as(target_tensor)
        availability = mask_tensor
    return probabilities, target_tensor, availability


def evaluate_head_batch(
    logits: Any,
    targets: Any,
    mask: Optional[Any] = None,
    threshold: float = 0.5,
    device: Optional[Any] = None,
) -> Dict[str, float]:
    """Evaluate one multi-label head, returning :func:`.metrics.evaluate_head`'s keys.

    :param logits: Unnormalized head outputs, shape ``(N, C)``.
    :type logits: numpy.ndarray | torch.Tensor
    :param targets: Binary multi-label targets, shape ``(N, C)``.
    :type targets: numpy.ndarray | torch.Tensor
    :param mask: Availability mask, one flag per sample ``(N,)`` or per (sample,
        class) pair ``(N, C)``; ``None`` treats every entry as available.
    :type mask: numpy.ndarray | torch.Tensor | None
    :param threshold: Probability threshold for a positive prediction.
    :type threshold: float
    :param device: Evaluation device; defaults to CUDA when available.
    :type device: str | torch.device | None
    :return: ``micro_f1``, ``macro_f1``, ``macro_precision``, ``macro_recall``,
        ``micro_precision``, ``micro_recall``, ``hamming_loss``,
        ``hamming_loss_baseline_positive_rate``, ``average_precision``, ``roc_auc``.
    :rtype: Dict[str, float]
    :raises ValueError: If no entry is available to score.
    """
    resolved_device = _resolve_device(device)
    probabilities, target_tensor, availability = _prepare(logits, targets, mask, resolved_device)
    if availability.sum() == 0:
        raise ValueError("No annotated samples are available.")

    predicted = (probabilities >= threshold).to(torch.float32)  # (N, C)
    positive_counts = (target_tensor * availability).sum(dim=0)  # (C,)
    available_counts = availability.sum(dim=0)  # (C,)

    # Per-class threshold metrics
    true_positives, false_positives, false_negatives = _threshold_counts(
        predicted, target_tensor, availability
    )
    precision = _safe_ratio(true_positives, true_positives + false_positives)  # (C,)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)  # (C,)
    f1_score = _safe_ratio(
        2.0 * true_positives, 2.0 * true_positives + false_positives + false_negatives
    )  # (C,)

    # Pooled (micro) threshold metrics over every available entry
    micro_true = true_positives.sum()
    micro_false_positive = false_positives.sum()
    micro_false_negative = false_negatives.sum()
    micro_precision = _safe_ratio(micro_true, micro_true + micro_false_positive)
    micro_recall = _safe_ratio(micro_true, micro_true + micro_false_negative)
    micro_f1 = _safe_ratio(
        2.0 * micro_true, 2.0 * micro_true + micro_false_positive + micro_false_negative
    )

    # Entry-level agreement over available entries only
    disagreement = ((target_tensor != predicted).to(torch.float32) * availability).sum()
    available_total = availability.sum()
    hamming = disagreement / available_total
    positive_rate = (target_tensor * availability).sum() / available_total

    # Ranking metrics
    average_precision = _average_precision(
        probabilities, target_tensor, availability, positive_counts
    )
    roc_auc = _roc_auc(
        probabilities, target_tensor, availability, positive_counts, available_counts
    )

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(f1_score.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "hamming_loss": float(hamming),
        "hamming_loss_baseline_positive_rate": float(positive_rate),
        "average_precision": float(_nanmean(average_precision)),
        "roc_auc": float(_nanmean(roc_auc)),
    }


def per_class_metrics_batch(
    logits: Any,
    targets: Any,
    mask: Optional[Any] = None,
    threshold: float = 0.5,
    device: Optional[Any] = None,
) -> list[Dict[str, float]]:
    """Per-class records with :func:`.metrics.per_class_metrics`' keys and conventions.

    :param logits: Unnormalized head outputs, shape ``(N, C)``. Note that the reference
        implementation takes probabilities here; this one takes logits, so that the
        logistic is applied once on the evaluation device.
    :type logits: numpy.ndarray | torch.Tensor
    :param targets: Binary multi-label targets, shape ``(N, C)``.
    :type targets: numpy.ndarray | torch.Tensor
    :param mask: Availability mask, as in :func:`evaluate_head_batch`.
    :type mask: numpy.ndarray | torch.Tensor | None
    :param threshold: Probability threshold for a positive prediction.
    :type threshold: float
    :param device: Evaluation device; defaults to CUDA when available.
    :type device: str | torch.device | None
    :return: One record per class with ``class_index``, ``positive_samples``,
        ``precision``, ``recall``, ``f1``, ``average_precision`` and ``roc_auc``.
    :rtype: list[Dict[str, float]]
    """
    resolved_device = _resolve_device(device)
    probabilities, target_tensor, availability = _prepare(logits, targets, mask, resolved_device)
    predicted = (probabilities >= threshold).to(torch.float32)

    positive_counts = (target_tensor * availability).sum(dim=0)
    available_counts = availability.sum(dim=0)
    true_positives, false_positives, false_negatives = _threshold_counts(
        predicted, target_tensor, availability
    )
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1_score = _safe_ratio(
        2.0 * true_positives, 2.0 * true_positives + false_positives + false_negatives
    )
    average_precision = _average_precision(
        probabilities, target_tensor, availability, positive_counts
    )
    roc_auc = _roc_auc(
        probabilities, target_tensor, availability, positive_counts, available_counts
    )

    columns = {
        "positive_samples": positive_counts.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1_score.cpu().numpy(),
        "average_precision": average_precision.cpu().numpy(),
        "roc_auc": roc_auc.cpu().numpy(),
    }
    return [
        {"class_index": float(index), **{name: float(values[index]) for name, values in columns.items()}}
        for index in range(target_tensor.shape[1])
    ]


def _nanmean(values: torch.Tensor) -> torch.Tensor:
    """Mean over the defined entries, ``nan`` when every entry is undefined."""
    defined = ~torch.isnan(values)
    count = defined.sum()
    if count == 0:
        return torch.tensor(float("nan"), device=values.device)
    return values[defined].sum() / count
