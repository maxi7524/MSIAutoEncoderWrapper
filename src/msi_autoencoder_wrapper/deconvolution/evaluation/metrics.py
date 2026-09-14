"""Torch-native quantitative metrics for deconvolution experiments."""

from __future__ import annotations

import torch

from ..contracts import DeconvolutionResult


def deconvolution_metrics(
    result: DeconvolutionResult,
    *,
    true_abundances: torch.Tensor,
    true_presence: torch.Tensor | None = None,
    threshold: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute abundance, support, and reconstruction metrics in Torch.

    :param result: Estimated decomposition outputs.
    :param true_abundances: Exact non-negative abundance targets, shape ``(B, C)``.
    :param true_presence: Optional support targets, shape ``(B, C)``. When
        omitted, support is derived from positive abundance.
    :param threshold: Estimated abundance threshold defining predicted support.
    :type result: DeconvolutionResult
    :type true_abundances: torch.Tensor
    :type true_presence: torch.Tensor | None
    :type threshold: float
    :return: Scalar metric tensors.
    :rtype: dict[str, torch.Tensor]
    :raises ValueError: If target shapes or threshold are invalid.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if true_abundances.shape != result.abundances.shape:
        raise ValueError("true_abundances must have shape (B, C) matching estimates.")
    if true_presence is not None and true_presence.shape != result.abundances.shape:
        raise ValueError("true_presence must have shape (B, C) matching estimates.")
    if true_presence is None:
        true_presence = true_abundances > 0
    predicted_presence = result.abundances > threshold  # (B, C)
    true_presence = true_presence.bool()  # (B, C)
    true_positive = (predicted_presence & true_presence).sum().to(result.abundances.dtype)  # ()
    false_positive = (predicted_presence & ~true_presence).sum().to(result.abundances.dtype)  # ()
    false_negative = (~predicted_presence & true_presence).sum().to(result.abundances.dtype)  # ()
    eps = torch.finfo(result.abundances.dtype).eps
    precision = true_positive / (true_positive + false_positive).clamp_min(eps)  # ()
    recall = true_positive / (true_positive + false_negative).clamp_min(eps)  # ()
    support_f1 = 2 * precision * recall / (precision + recall).clamp_min(eps)  # ()
    return {
        "abundance_mae": (result.abundances - true_abundances).abs().mean(),  # ()
        "abundance_rmse": (result.abundances - true_abundances).square().mean().sqrt(),  # ()
        "reconstruction_mse": result.residual.square().mean(),  # ()
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": support_f1,
    }
