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


def evaluate_head(
    logits: np.ndarray,
    targets: np.ndarray,
    target_type: str,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate one single-label or multi-label classification head."""
    if not 0.0 <= threshold <= 1.0:
        raise_validation_error(
            "HeadAnalysis", "threshold must be between zero and one."
        )
    available = (
        np.ones(len(targets), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    selected_logits = np.asarray(logits)[available]
    selected_targets = np.asarray(targets)[available]
    if len(selected_targets) == 0:
        raise_validation_error("HeadAnalysis", "No annotated samples are available.")
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
            "HeadAnalysis", f"Unsupported target type '{target_type}'."
        )
    probabilities = probabilities_from_logits(selected_logits, target_type)
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


def per_class_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
) -> list[Dict[str, float]]:
    """Return multi-label metrics for every output class."""
    available = (
        np.ones(len(targets), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    truth = np.asarray(targets)[available].astype(bool)
    selected_probabilities = np.asarray(probabilities)[available]
    predicted = selected_probabilities >= threshold
    records: list[Dict[str, float]] = []
    for class_index in range(truth.shape[1]):
        class_truth = truth[:, class_index]
        class_prediction = predicted[:, class_index]
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
                "average_precision": float(
                    average_precision_score(
                        class_truth, selected_probabilities[:, class_index]
                    )
                ),
            }
        )
    return records
