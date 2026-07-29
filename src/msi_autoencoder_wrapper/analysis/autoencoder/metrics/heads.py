"""Generic classification-head evaluation."""

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


def evaluate_head(
    logits: np.ndarray,
    targets: np.ndarray,
    target_type: str,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate one classification head using its dataset target semantics.

    :param logits: Unnormalized head outputs.
    :type logits: numpy.ndarray
    :param targets: Integer single-label or binary multi-label targets.
    :type targets: numpy.ndarray
    :param target_type: ``single_label`` or ``multi_label``.
    :type target_type: str
    :param mask: Optional per-sample target availability mask.
    :type mask: numpy.ndarray | None
    :param threshold: Probability threshold for multi-label predictions.
    :type threshold: float
    :return: Scalar metrics and a confusion matrix when applicable.
    :rtype: Dict[str, Any]
    :raises ValidationError: If the target type or threshold is invalid.
    """
    if not 0.0 <= threshold <= 1.0:
        raise_validation_error(
            "AutoencoderAnalysis", "Head threshold must be between zero and one."
        )
    available = (
        np.ones(len(targets), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    selected_logits = np.asarray(logits)[available]
    selected_targets = np.asarray(targets)[available]
    if len(selected_targets) == 0:
        raise_validation_error(
            "AutoencoderAnalysis", "No annotated samples are available for this head."
        )
    if target_type == "single_label":
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
            "AutoencoderAnalysis", f"Unsupported target type '{target_type}'."
        )
    probabilities = 1.0 / (1.0 + np.exp(-selected_logits))
    predicted = probabilities >= threshold
    binary_targets = selected_targets.astype(bool)
    return {
        "micro_f1": float(
            f1_score(binary_targets, predicted, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(binary_targets, predicted, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(binary_targets, predicted, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(binary_targets, predicted, average="micro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(binary_targets, predicted)),
        "average_precision": float(
            average_precision_score(binary_targets, probabilities, average="macro")
        ),
    }
