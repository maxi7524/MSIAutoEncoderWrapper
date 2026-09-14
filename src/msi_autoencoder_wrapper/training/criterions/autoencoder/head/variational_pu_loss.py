"""Prior-free multilabel variational PU risk and MixUp consistency.

The variational and consistency terms follow Chen et al., NeurIPS 2020,
equations (6), (8), and (9). The marginal contains the declared positive and
unlabelled entries; explicitly simulated negatives are excluded.
"""

from __future__ import annotations

import math
import weakref

import torch
import torch.nn.functional as F

from .evidence_losses import EvidenceHeadCriterion
from .supervision_masks import supervised_pu_masks
from ...criterions_manager import CriterionsManager
from .....data.annotation_evidence import NEGATIVE
from .....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@CriterionsManager.register_criterion("autoencoder", "head", "VariationalPULoss")
class VariationalPULoss(EvidenceHeadCriterion):
    """Train one variational PU score per ion without estimating class priors.

    :param head_id: Model head identifier.
    :param target_field: Binary annotation field.
    :param class_indices: Optional selected target columns.
    :param evidence: Signal evidence parameters, used only when negatives are enabled.
    :param alpha: Weight of the variational marginal term ``log E[p]``.
        Values above one are a project-specific precision-oriented modification.
    :type alpha: float
    :param beta: Weight of the positive term ``-E_P[log p]``.
    :type beta: float
    :param negative_weight: Additional mean BCE on signal negatives; zero disables it.
    :param simulated_negative_weight: Additional mean BCE on explicitly
        declared ``N_sim`` entries; zero disables it.
    :param consistency_weight: Weight of logarithmic MixUp consistency.
    :param mixup_alpha: Positive symmetric Beta distribution parameter.
    :param max_mixup_classes: Maximum randomly selected active classes per batch.
    :raises ValueError: If weights or MixUp settings are invalid.

    REMARK: Raw scores are not calibrated posteriors. ``normalize_scores``
    implements the paper's final scaling using maxima measured on training data.
    Signal-negative supervision is an experimental extension, not the original VPU.
    ``alpha != beta`` is an experimental asymmetric VPU variant; ``alpha=beta=1``
    reproduces the original variational risk.
    """

    def __init__(self, head_id: str, target_field: str, class_indices=None,
                 evidence=None, negative_weight: float = 0.0,
                 simulated_negative_weight: float = 0.0,
                 alpha: float = 1.0, beta: float = 1.0,
                 consistency_weight: float = 1.0, mixup_alpha: float = 0.3,
                 max_mixup_classes: int = 64) -> None:
        super().__init__(head_id, target_field, class_indices, evidence)
        if any(
            not math.isfinite(v) or v < 0
            for v in (
                negative_weight,
                simulated_negative_weight,
                alpha,
                beta,
                consistency_weight,
            )
        ):
            raise ValueError("VPU weights must be finite and nonnegative.")
        if alpha == 0 and beta == 0:
            raise ValueError("At least one of alpha or beta must be positive.")
        if not math.isfinite(mixup_alpha) or mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be finite and positive.")
        if isinstance(max_mixup_classes, bool) or not isinstance(max_mixup_classes, int) or max_mixup_classes < 1:
            raise ValueError("max_mixup_classes must be a positive integer.")
        self.negative_weight = negative_weight
        self.simulated_negative_weight = simulated_negative_weight
        self.alpha = alpha
        self.beta = beta
        self.consistency_weight = consistency_weight
        self.mixup_alpha = mixup_alpha
        self.max_mixup_classes = max_mixup_classes
        self._model_ref = None
        self._config.update(negative_weight=negative_weight,
                            simulated_negative_weight=simulated_negative_weight,
                            alpha=alpha, beta=beta,
                            consistency_weight=consistency_weight,
                            mixup_alpha=mixup_alpha, max_mixup_classes=max_mixup_classes)

    def on_phase_start(self, model, dataset, transient_cache) -> None:
        """Bind optional evidence and the model without registering it inside the loss."""
        if self.negative_weight > 0:
            super().on_phase_start(model, dataset, transient_cache)
        self._model_ref = weakref.ref(model)
        logger.info(
            "Starting variational PU: alpha=%s beta=%s consistency=%s signal_negative_weight=%s simulated_negative_weight=%s.",
            self.alpha,
            self.beta,
            self.consistency_weight,
            self.negative_weight,
            self.simulated_negative_weight,
        )

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return active-class mean variational risk plus configured regularization."""
        logits, targets, mask = self.multilabel_batch(model_outputs, batch_data)
        # Variational risk over the declared P/U marginal population
        positives, unlabelled, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            mask,
            self.class_indices,
        )
        marginal = positives | unlabelled  # (B, C)
        positive_count = positives.sum(dim=0)  # (C,)
        marginal_count = marginal.sum(dim=0)  # (C,)
        active = (positive_count > 0) & (marginal_count > 0)  # (C,)
        loss = logits.sum() * 0.0  # ()
        if bool(active.any()):
            log_probability = F.logsigmoid(logits[:, active])  # (B, C_active)
            log_mean = torch.logsumexp(
                log_probability.masked_fill(~marginal[:, active], -torch.inf), dim=0,
            ) - marginal_count[active].to(logits).log()  # (C_active,)
            positive_mean = (
                (log_probability * positives[:, active]).sum(dim=0) / positive_count[active]
            )  # (C_active,)
            # Project-specific asymmetric VPU
            ## alpha controls the marginal P/U pressure; beta retains explicit
            ## support for observed positives.
            loss = (self.alpha * log_mean - self.beta * positive_mean).mean()  # ()
            if self.consistency_weight > 0:
                loss = loss + self.consistency_weight * self._mixup_loss(
                    logits, positives, marginal, active, batch_data,
                )  # ()

        # Optional signal-negative supervision leaves the marginal term intact
        if self.negative_weight > 0:
            _, _, _, states = self.evidence_batch(model_outputs, batch_data)
            negatives = states == NEGATIVE  # (B, C)
            if bool(negatives.any()):
                loss = loss + self.negative_weight * F.softplus(logits[negatives]).mean()  # ()
        if self.simulated_negative_weight > 0 and bool(simulated_negative.any()):
            loss = loss + self.simulated_negative_weight * F.softplus(
                logits[simulated_negative]
            ).mean()  # ()
        return loss

    def _mixup_loss(self, logits, positives, mask, active, batch_data):
        """Mix one positive and one marginal example per sampled active class."""
        model = self._model_ref() if self._model_ref is not None else None
        if model is None:
            raise RuntimeError("VPU consistency requires on_phase_start(model, dataset, cache).")
        classes = active.nonzero(as_tuple=True)[0]  # (C_active,)
        classes = classes[torch.randperm(len(classes), device=classes.device)[:self.max_mixup_classes]]  # (K,)
        p_rows, u_rows = [], []
        for column in classes:
            p = positives[:, column].nonzero(as_tuple=True)[0]  # (N_p,)
            u = mask[:, column].nonzero(as_tuple=True)[0]  # (N_marginal,)
            p_rows.append(p[torch.randint(len(p), (), device=p.device)])
            u_rows.append(u[torch.randint(len(u), (), device=u.device)])
        p_rows, u_rows = torch.stack(p_rows), torch.stack(u_rows)  # (K,), (K,)
        alpha = logits.new_tensor(self.mixup_alpha)
        coefficient = torch.distributions.Beta(alpha, alpha).sample((len(classes),))  # (K,)
        spectra = batch_data[1].to(logits)  # (B, M)
        mixed = coefficient[:, None] * spectra[p_rows] + (1 - coefficient[:, None]) * spectra[u_rows]  # (K, M)
        guessed = coefficient + (1 - coefficient) * logits[u_rows, classes].sigmoid().detach()  # (K,)
        mixed_logits = model(mixed)[self.output_key]  # (K, C)
        selected = mixed_logits[torch.arange(len(classes), device=classes.device), classes]  # (K,)
        return (F.logsigmoid(selected) - guessed.clamp_min(torch.finfo(logits.dtype).tiny).log()).square().mean()  # ()


def normalize_scores(logits: torch.Tensor, training_maxima: torch.Tensor) -> torch.Tensor:
    """Apply VPU's final classwise normalization using training-only maxima.

    :param logits: Prediction logits, shape ``(B, C)``.
    :param training_maxima: Maximum sigmoid score per class on training data, ``(C,)``.
    :return: Normalized scores in [0, 1], shape ``(B, C)``.
    :raises ValueError: If the supplied maxima are invalid.
    """
    if training_maxima.shape != logits.shape[-1:] or not bool(torch.isfinite(training_maxima).all()):
        raise ValueError("training_maxima must be a finite vector aligned with logits.")
    if bool(((training_maxima <= 0) | (training_maxima > 1)).any()):
        raise ValueError("training_maxima must lie in (0, 1].")
    return (logits.sigmoid() / training_maxima.to(logits)).clamp_max(1)  # (B, C)
