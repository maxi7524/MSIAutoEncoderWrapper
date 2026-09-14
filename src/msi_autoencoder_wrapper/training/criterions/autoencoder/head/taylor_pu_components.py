"""Reusable mathematical components of the Taylor-VPU objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def taylor_variational_risk(
    logits: torch.Tensor,
    positives: torch.Tensor,
    unlabelled: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Return Zhao et al.'s Taylor variational risk over eligible classes.

    :param logits: Binary positive-class logits, shape ``(B, C)``.
    :type logits: torch.Tensor
    :param positives: Observed-positive entries, shape ``(B, C)``.
    :type positives: torch.Tensor
    :param unlabelled: Unlabelled marginal entries, shape ``(B, C)``.
    :type unlabelled: torch.Tensor
    :param order: Positive Taylor-series truncation order.
    :type order: int
    :return: Mean risk over classes that contain both P and U, scalar.
    :rtype: torch.Tensor

    REMARK: ``N_sim`` is deliberately absent. The original Taylor-VPU core is
    defined on P/U, while simulated-negative supervision is a project-specific
    additive extension at the calling site.
    """
    positive_count = positives.sum(dim=0)  # (C,)
    unlabelled_count = unlabelled.sum(dim=0)  # (C,)
    active = (positive_count > 0) & (unlabelled_count > 0)  # (C,)
    if not bool(active.any()):
        return logits.sum() * 0.0  # ()
    active_logits = logits[:, active]  # (B, C_active)
    active_positive = positives[:, active]  # (B, C_active)
    active_unlabelled = unlabelled[:, active]  # (B, C_active)
    positive_loss = -(
        F.logsigmoid(active_logits) * active_positive
    ).sum(dim=0) / positive_count[active]  # (C_active,)
    unlabelled_mean = (
        active_logits.sigmoid() * active_unlabelled
    ).sum(dim=0) / unlabelled_count[active]  # (C_active,)
    remainder = 1.0 - unlabelled_mean  # (C_active,)
    taylor_log_mean = sum(
        -(remainder.pow(index) / index) for index in range(1, order + 1)
    )  # (C_active,)
    return (positive_loss + taylor_log_mean).mean()  # ()


def simulated_negative_bce(
    logits: torch.Tensor,
    simulated_negative: torch.Tensor,
) -> torch.Tensor:
    """Return mean negative BCE on explicitly declared ``N_sim`` entries.

    :param logits: Binary positive-class logits, shape ``(B, C)``.
    :type logits: torch.Tensor
    :param simulated_negative: Explicit reliable-negative mask, shape ``(B, C)``.
    :type simulated_negative: torch.Tensor
    :return: Negative BCE or graph-preserving zero, scalar.
    :rtype: torch.Tensor
    """
    if not bool(simulated_negative.any()):
        return logits.sum() * 0.0  # ()
    return F.softplus(logits[simulated_negative]).mean()  # ()
