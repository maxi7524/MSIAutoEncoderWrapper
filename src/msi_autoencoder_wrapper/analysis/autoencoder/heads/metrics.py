"""Generic classification-head metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from ....utils.exceptions import raise_validation_error


def probabilities_from_logits(logits: np.ndarray, target_type: str) -> np.ndarray:
    """Convert logits using target-compatible activation semantics."""
    values = np.asarray(logits)
    if target_type == "multi_label":
        return 1.0 / (1.0 + np.exp(-values))
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _multi_label_availability(
    mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """Broadcast an optional target-availability mask to one boolean per (sample, class).

    Multi-label target availability can be recorded at two different granularities:
    one flag per sample (``shape == (N,)`` — e.g. ``unobserved_label_policy: masked``,
    where an entire sample's target vector is unusable), or one flag per (sample,
    class) pair (``shape == (N, C)`` — e.g. ``unobserved_label_policy: unlabelled``,
    where every class is independently either a confirmed label or an unlabelled
    entry; see ``PixelDataset``'s own comment on this distinction). Both are broadcast
    to the same ``(N, C)`` boolean matrix here so every metric below applies mask
    per class column, not per row — reshaping a per-class mask to one dimension and
    then row-filtering (the previous behavior) silently produces a shape mismatch or,
    worse, a wrong row selection whenever ``C != 1``.

    :param mask: ``None``, one flag per sample, or one flag per (sample, class) pair.
    :type mask: numpy.ndarray | None
    :param shape: Target/probability matrix shape, ``(N, C)``.
    :type shape: tuple[int, int]
    :return: Boolean availability matrix, shape ``(N, C)``.
    :rtype: numpy.ndarray
    """
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape == shape:
        return mask_array
    if mask_array.ndim == 1 and mask_array.shape[0] == shape[0]:
        return np.broadcast_to(mask_array[:, None], shape)
    raise_validation_error(
        "HeadAnalysis",
        f"mask shape {mask_array.shape} is compatible with neither one flag per "
        f"sample ({shape[0]},) nor one flag per (sample, class) pair {shape}.",
    )


def evaluate_head(
    logits: np.ndarray,
    targets: np.ndarray,
    target_type: str,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate one single-label or multi-label classification head.

    :param logits: Unnormalized head outputs, shape ``(N, C)``.
    :type logits: numpy.ndarray
    :param targets: Integer single-label targets ``(N,)`` or binary multi-label
        targets ``(N, C)``.
    :type targets: numpy.ndarray
    :param target_type: ``single_label`` or ``multi_label``.
    :type target_type: str
    :param mask: Optional target-availability mask. For ``single_label``, one flag
        per sample ``(N,)``. For ``multi_label``, one flag per sample ``(N,)`` *or*
        one flag per (sample, class) pair ``(N, C)`` — see
        :func:`_multi_label_availability` for why the two are not interchangeable.
    :type mask: numpy.ndarray | None
    :param threshold: Probability threshold for a positive multi-label prediction.
    :type threshold: float
    :return: Scalar metrics (and a confusion matrix for ``single_label``).
    :rtype: Dict[str, Any]
    :raises ValidationError: If the target type, threshold, or mask shape is invalid,
        or no samples/entries are available to score.
    """
    if not 0.0 <= threshold <= 1.0:
        raise_validation_error(
            "HeadAnalysis", "threshold must be between zero and one."
        )
    logits_array = np.asarray(logits)
    targets_array = np.asarray(targets)

    if target_type == "single_label":
        available = (
            np.ones(len(targets_array), dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool).reshape(-1)
        )
        selected_logits = logits_array[available]
        selected_targets = targets_array[available]
        if len(selected_targets) == 0:
            raise_validation_error(
                "HeadAnalysis", "No annotated samples are available."
            )
        predicted = np.argmax(selected_logits, axis=1)
        return {
            "accuracy": float(accuracy_score(selected_targets, predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(selected_targets, predicted)
            ),
            "macro_precision": float(
                precision_score(
                    selected_targets, predicted, average="macro", zero_division=0
                )
            ),
            "macro_recall": float(
                recall_score(
                    selected_targets, predicted, average="macro", zero_division=0
                )
            ),
            "macro_f1": float(
                f1_score(selected_targets, predicted, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(selected_targets, predicted),
        }
    if target_type != "multi_label":
        raise_validation_error(
            "HeadAnalysis", f"Unsupported target type '{target_type}'."
        )

    probabilities = probabilities_from_logits(logits_array, target_type)
    predicted = probabilities >= threshold
    binary_targets = targets_array.astype(bool)
    availability = _multi_label_availability(mask, binary_targets.shape)
    if not availability.any():
        raise_validation_error("HeadAnalysis", "No annotated samples are available.")
    per_class = _per_class_records(probabilities, binary_targets, availability, threshold)
    flat_true = binary_targets[availability]
    flat_pred = predicted[availability]
    return {
        "micro_f1": float(f1_score(flat_true, flat_pred, zero_division=0)),
        "macro_f1": float(np.mean([record["f1"] for record in per_class])),
        "micro_precision": float(
            precision_score(flat_true, flat_pred, zero_division=0)
        ),
        "micro_recall": float(recall_score(flat_true, flat_pred, zero_division=0)),
        "hamming_loss": float(np.mean(flat_true != flat_pred)),
        "average_precision": float(
            np.nanmean([record["average_precision"] for record in per_class])
        ),
    }


def _per_class_records(
    probabilities: np.ndarray,
    binary_targets: np.ndarray,
    availability: np.ndarray,
    threshold: float,
) -> list[Dict[str, float]]:
    """Compute one metrics record per class, masking availability per column.

    :param probabilities: Predicted probabilities, shape ``(N, C)``.
    :type probabilities: numpy.ndarray
    :param binary_targets: Boolean ground truth, shape ``(N, C)``.
    :type binary_targets: numpy.ndarray
    :param availability: Boolean mask, shape ``(N, C)`` (see
        :func:`_multi_label_availability`).
    :type availability: numpy.ndarray
    :param threshold: Probability threshold for a positive prediction.
    :type threshold: float
    :return: One record per class, in column order.
    :rtype: list[Dict[str, float]]
    """
    predicted = probabilities >= threshold
    records: list[Dict[str, float]] = []
    for class_index in range(binary_targets.shape[1]):
        column_available = availability[:, class_index]
        class_truth = binary_targets[column_available, class_index]
        class_prediction = predicted[column_available, class_index]
        class_probabilities = probabilities[column_available, class_index]
        has_positive = bool(class_truth.any())
        records.append(
            {
                "class_index": float(class_index),
                "positive_samples": float(np.sum(class_truth)),
                "precision": float(
                    precision_score(class_truth, class_prediction, zero_division=0)
                ),
                "recall": float(
                    recall_score(class_truth, class_prediction, zero_division=0)
                ),
                "f1": float(f1_score(class_truth, class_prediction, zero_division=0)),
                # average_precision_score is undefined without at least one positive;
                # nan (excluded from macro means via nanmean) rather than a
                # misleadingly concrete 0.0.
                "average_precision": (
                    float(average_precision_score(class_truth, class_probabilities))
                    if has_positive
                    else float("nan")
                ),
            }
        )
    return records


def per_class_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
) -> list[Dict[str, float]]:
    """Return multi-label metrics for every output class.

    :param probabilities: Predicted probabilities, shape ``(N, C)``.
    :type probabilities: numpy.ndarray
    :param targets: Binary ground truth, shape ``(N, C)``.
    :type targets: numpy.ndarray
    :param threshold: Probability threshold for a positive prediction.
    :type threshold: float
    :param mask: Optional availability mask, one flag per sample ``(N,)`` or one
        flag per (sample, class) pair ``(N, C)`` — see
        :func:`_multi_label_availability`.
    :type mask: numpy.ndarray | None
    :return: One record per class: ``class_index``, ``positive_samples``,
        ``precision``, ``recall``, ``f1``, ``average_precision`` (``nan`` for a class
        with zero available positives).
    :rtype: list[Dict[str, float]]
    """
    targets_array = np.asarray(targets)
    probabilities_array = np.asarray(probabilities)
    availability = _multi_label_availability(mask, targets_array.shape)
    return _per_class_records(
        probabilities_array, targets_array.astype(bool), availability, threshold
    )
