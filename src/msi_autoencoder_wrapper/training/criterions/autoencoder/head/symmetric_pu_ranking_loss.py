"""Prior-free symmetric label ranking, an explicitly heuristic PU baseline."""

from __future__ import annotations

import math
import torch

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager


@CriterionsManager.register_criterion("autoencoder", "head", "SymmetricPURankingLoss")
class SymmetricPURankingLoss(MSIHeadCriterion):
    """Rank annotated labels above unlabelled labels with a bounded sigmoid loss.

    :param head_id: Model output head.
    :param target_field: Binary ion target.
    :param class_indices: Optional target-column selection.
    :param temperature: Positive scale of pairwise score differences.
    :param pairs_per_sample: Monte Carlo pair count per spectrum.
    :raises ValueError: If sampling settings are invalid.

    REMARK: The surrogate satisfies l(d) + l(-d) = 1. Uniform pair weights
    deliberately omit the prior-dependent correction of Kanehira and Harada
    (CVPR 2016); this is not their unbiased risk estimator.
    """

    def __init__(self, head_id: str, target_field: str, class_indices=None,
                 temperature: float = 1.0, pairs_per_sample: int = 64) -> None:
        super().__init__(head_id, target_field, class_indices)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if isinstance(pairs_per_sample, bool) or not isinstance(pairs_per_sample, int) or pairs_per_sample < 1:
            raise ValueError("pairs_per_sample must be a positive integer.")
        self.temperature = temperature
        self.pairs_per_sample = pairs_per_sample
        self._config = dict(head_id=head_id, target_field=target_field,
                            class_indices=self.class_indices, temperature=temperature,
                            pairs_per_sample=pairs_per_sample)

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return the mean sampled ranking loss over spectra with eligible pairs."""
        logits, targets, mask = self.multilabel_batch(model_outputs, batch_data)
        losses = []
        for scores, labels, available in zip(logits, targets, mask):
            positive = ((labels > 0.5) & available).nonzero(as_tuple=True)[0]  # (P,)
            unlabelled = ((labels <= 0.5) & available).nonzero(as_tuple=True)[0]  # (U,)
            if not len(positive) or not len(unlabelled):
                continue
            p = positive[torch.randint(len(positive), (self.pairs_per_sample,), device=logits.device)]  # (K,)
            u = unlabelled[torch.randint(len(unlabelled), (self.pairs_per_sample,), device=logits.device)]  # (K,)
            losses.append(torch.sigmoid((scores[u] - scores[p]) / self.temperature).mean())  # ()
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0  # ()
