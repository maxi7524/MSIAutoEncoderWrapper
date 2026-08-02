"""Streaming per-class metrics for multi-label predictions."""

from __future__ import annotations

import torch

from ..base import ClassMetric
from ...utils.exceptions import raise_validation_error


class PerClassClassification(ClassMetric):
    """Accumulate sufficient statistics and return metrics for every class."""

    direction = "maximize"

    def __init__(self, metric: str, threshold: float = 0.5) -> None:
        super().__init__()
        if metric not in {"precision", "recall", "f1"}:
            raise_validation_error(
                "PerClassClassification", "metric must be precision, recall, or f1."
            )
        if not 0.0 <= threshold <= 1.0:
            raise_validation_error(
                "PerClassClassification", "threshold must be between zero and one."
            )
        self.metric = metric
        self.threshold = float(threshold)
        self.register_buffer("true_positive", torch.empty(0), persistent=False)
        self.register_buffer("false_positive", torch.empty(0), persistent=False)
        self.register_buffer("false_negative", torch.empty(0), persistent=False)

    def reset(self) -> None:
        """Discard accumulated class counts."""
        self.true_positive = torch.empty(0, device=self.true_positive.device)
        self.false_positive = torch.empty(0, device=self.false_positive.device)
        self.false_negative = torch.empty(0, device=self.false_negative.device)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate TP, FP, and FN without retaining full model outputs."""
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise_validation_error(
                "PerClassClassification", "prediction and target must share [B, C]."
            )
        probabilities = torch.sigmoid(prediction) if prediction.is_floating_point() else prediction
        predicted = probabilities >= self.threshold
        truth = target.to(device=prediction.device, dtype=torch.bool)
        if mask is not None:
            available = mask.to(device=prediction.device, dtype=torch.bool)
            if available.ndim == 1:
                available = available.unsqueeze(1).expand_as(truth)
            if available.shape != truth.shape:
                raise_validation_error(
                    "PerClassClassification", "mask must have shape [B] or [B, C]."
                )
            predicted = predicted & available
            truth = truth & available
        tp = (predicted & truth).sum(dim=0, dtype=torch.float64)
        fp = (predicted & ~truth).sum(dim=0, dtype=torch.float64)
        fn = (~predicted & truth).sum(dim=0, dtype=torch.float64)
        if self.true_positive.numel() == 0:
            self.true_positive, self.false_positive, self.false_negative = tp, fp, fn
        else:
            self.true_positive += tp
            self.false_positive += fp
            self.false_negative += fn

    def compute(self) -> torch.Tensor:
        """Return precision, recall, or F1 for each accumulated class."""
        if self.true_positive.numel() == 0:
            raise_validation_error(
                "PerClassClassification", "No classification batches were accumulated."
            )
        if self.metric == "precision":
            denominator = self.true_positive + self.false_positive
            numerator = self.true_positive
        elif self.metric == "recall":
            denominator = self.true_positive + self.false_negative
            numerator = self.true_positive
        else:
            denominator = (
                2 * self.true_positive + self.false_positive + self.false_negative
            )
            numerator = 2 * self.true_positive
        return torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros_like(denominator),
        )
