"""Mathematical CoVPU components: planar flows, collaborator, and risk."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class CoVPUPlanarFlow(nn.Module):
    """Stack CoVPU planar layers ``v_next = v + u tanh(w^T v + a)``."""

    def __init__(self, layer_count: int) -> None:
        super().__init__()
        if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 1:
            raise ValueError("layer_count must be a positive integer.")
        self.u = nn.Parameter(torch.empty(layer_count, 2))
        self.w = nn.Parameter(torch.empty(layer_count, 2))
        self.a = nn.Parameter(torch.zeros(layer_count))
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.w, std=0.02)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformed values ``(..., 2)`` and log Jacobian ``(...)``."""
        if value.shape[-1] != 2:
            raise ValueError("CoVPU planar flow requires final dimension two.")
        transformed = value  # (..., 2)
        log_determinant = value.new_zeros(value.shape[:-1])  # (...,)
        for u, w, offset in zip(self.u, self.w, self.a, strict=True):
            activation = (transformed @ w + offset).tanh()  # (...,)
            transformed = transformed + activation.unsqueeze(-1) * u  # (..., 2)
            determinant = 1.0 + (1.0 - activation.square()) * (w * u).sum()  # (...,)
            log_determinant = log_determinant + determinant.abs().clamp_min(
                torch.finfo(value.dtype).tiny
            ).log()  # (...,)
        return transformed, log_determinant


@dataclass(frozen=True)
class CoVPUConsensus:
    """Collaborator masks needed by the CoVPU classifier objective."""

    pseudo_positive: torch.Tensor
    component_positive: torch.Tensor


def covpu_consensus(
    classifier_probability: torch.Tensor,
    component_assignment: torch.Tensor,
    unlabelled: torch.Tensor,
    query_count: int,
) -> CoVPUConsensus:
    """Implement Algorithm 1's top-J component/class positive consensus."""
    if classifier_probability.shape != component_assignment.shape or unlabelled.shape != classifier_probability.shape:
        raise ValueError("CoVPU consensus inputs must share shape (B, C).")
    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count < 1:
        raise ValueError("query_count must be a positive integer.")
    component_positive = torch.zeros_like(unlabelled)  # (B, C)
    pseudo_positive = torch.zeros_like(unlabelled)  # (B, C)
    hard_component = component_assignment.to(dtype=torch.long)  # (B, C)
    for column in range(classifier_probability.shape[1]):
        unlabelled_rows = unlabelled[:, column].nonzero(as_tuple=True)[0]  # (N_U,)
        predicted_positive = unlabelled_rows[classifier_probability[unlabelled_rows, column] >= 0.5]  # (N_pred,)
        if len(predicted_positive) == 0:
            continue
        high = predicted_positive[classifier_probability[predicted_positive, column].topk(
            min(query_count, len(predicted_positive))
        ).indices]  # (J,)
        components = hard_component[high, column]  # (J,)
        positive_component = int((components == 1).sum() >= (components == 0).sum())
        rows = unlabelled_rows[hard_component[unlabelled_rows, column] == positive_component]  # (N_component,)
        component_positive[rows, column] = True
        pseudo_positive[rows, column] = classifier_probability[rows, column] >= 0.5
    return CoVPUConsensus(pseudo_positive=pseudo_positive, component_positive=component_positive)


def covpu_rebalanced_variational_risk(
    classifier_logits: torch.Tensor,
    labelled_positive: torch.Tensor,
    pseudo_positive: torch.Tensor,
    remaining_negative: torch.Tensor,
    rebalanced_prior: float | torch.Tensor,
    labelled_positive_prior: float | torch.Tensor,
) -> torch.Tensor:
    """Return CoVPU equation (15), averaged over eligible classes."""
    if any(mask.shape != classifier_logits.shape for mask in (labelled_positive, pseudo_positive, remaining_negative)):
        raise ValueError("CoVPU masks must share classifier logits shape (B, C).")
    probability = classifier_logits.sigmoid()  # (B, C)
    pl_count = labelled_positive.sum(dim=0)  # (C,)
    pu_count = pseudo_positive.sum(dim=0)  # (C,)
    n_count = remaining_negative.sum(dim=0)  # (C,)
    active = (pl_count > 0) & (pu_count > 0) & (n_count > 0)  # (C,)
    if not bool(active.any()):
        return classifier_logits.sum() * 0.0  # ()
    pi_prime = torch.as_tensor(rebalanced_prior, dtype=classifier_logits.dtype, device=classifier_logits.device).expand_as(pl_count)[active]  # (C_active,)
    pi_labelled = torch.as_tensor(labelled_positive_prior, dtype=classifier_logits.dtype, device=classifier_logits.device).expand_as(pl_count)[active]  # (C_active,)
    if bool(((pi_prime <= 0) | (pi_prime >= 1) | (pi_labelled < 0) | (pi_labelled >= pi_prime)).any()):
        raise ValueError("CoVPU requires 0 <= pi_PL < pi_prime < 1.")
    value = probability[:, active]  # (B, C_active)
    pl_mean = (value * labelled_positive[:, active]).sum(dim=0) / pl_count[active]  # (C_active,)
    pu_mean = (value * pseudo_positive[:, active]).sum(dim=0) / pu_count[active]  # (C_active,)
    n_mean = (value * remaining_negative[:, active]).sum(dim=0) / n_count[active]  # (C_active,)
    mixture = (pi_prime + pi_labelled) * pl_mean / 2 + (pi_prime - pi_labelled) * pu_mean / 2 + (1 - pi_prime) * n_mean  # (C_active,)
    log_value = value.clamp_min(torch.finfo(value.dtype).tiny).log()  # (B, C_active)
    pl_log_mean = (log_value * labelled_positive[:, active]).sum(dim=0) / pl_count[active]  # (C_active,)
    pu_log_mean = (log_value * pseudo_positive[:, active]).sum(dim=0) / pu_count[active]  # (C_active,)
    risk = mixture.clamp_min(torch.finfo(value.dtype).tiny).log() - (pi_prime + pi_labelled) * pl_log_mean / (2 * pi_prime) - (pi_prime - pi_labelled) * pu_log_mean / (2 * pi_prime)  # (C_active,)
    return risk.mean()  # ()
