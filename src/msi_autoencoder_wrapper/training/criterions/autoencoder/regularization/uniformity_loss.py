"""Uniformity regularization for canonicalized encoder latents."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

from ...autoencoder_base_criterions import MSIRegularizationCriterion
from ...criterions_manager import CriterionsManager
from .....utils.exceptions import (
    raise_incompatible_interface_error,
    raise_validation_error,
)
from .canonicalized_latent import CanonicalizedLatentMixin


@CriterionsManager.register_criterion("autoencoder", "regularization", "UniformityLoss")
class MSIUniformityLoss(CanonicalizedLatentMixin, MSIRegularizationCriterion):
    r"""Encourage canonicalized latents to occupy the LayerNorm sphere uniformly.

    The objective is ``log(sum_{i != j} exp(-t ||u_i-u_j||²) / B²)``. The
    diagonal is excluded from the numerator but the specified ``B²`` denominator
    is retained.

    :param temperature: Positive pairwise-distance scale ``t``.
    :type temperature: float
    :param latent_source: Model output key, or ``auto`` for ``latent_space``.
    :type latent_source: str
    """

    requires_input_grad = False

    def __init__(
        self,
        temperature: float = 2.0,
        latent_source: str = "auto",
    ) -> None:
        super().__init__()
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (float, int))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise_validation_error(
                "UniformityLoss",
                "temperature must be a finite positive number.",
            )
        self.temperature = float(temperature)
        self.latent_source = "latent_space" if latent_source == "auto" else latent_source
        self._config = {
            "temperature": temperature,
            "latent_source": latent_source,
        }

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the canonicalized latent uniformity penalty.

        :param model_outputs: Mapping returned by the active model.
        :type model_outputs: Dict[str, torch.Tensor]
        :param batch_data: Active training batch.
        :type batch_data: Tuple[torch.Tensor, ...]
        :param kwargs: Unused extension arguments.
        :type kwargs: Any
        :return: Scalar uniformity penalty.
        :rtype: torch.Tensor
        :raises IncompatibleInterfaceError: If the latent output is missing or
            the batch contains fewer than two spectra.
        """
        del batch_data, kwargs
        if self.latent_source not in model_outputs:
            raise_incompatible_interface_error(
                "UniformityLoss",
                f"Model outputs must contain '{self.latent_source}'.",
            )
        latent = model_outputs[self.latent_source]
        if latent.ndim != 2:
            raise_incompatible_interface_error(
                "UniformityLoss",
                "Latent representations must have shape (B, D).",
            )
        batch_size = latent.shape[0]
        if batch_size < 2:
            raise_incompatible_interface_error(
                "UniformityLoss",
                "UniformityLoss requires batches with at least two spectra.",
            )

        canonical_latent = self.canonicalize_latent(latent)  # (B, D)
        squared_norms = canonical_latent.square().sum(dim=1, keepdim=True)  # (B, 1)
        pairwise_squared_distances = (
            squared_norms
            + squared_norms.transpose(0, 1)
            - 2.0 * (canonical_latent @ canonical_latent.transpose(0, 1))
        ).clamp_min(0.0)  # (B, B)
        upper_indices = torch.triu_indices(
            batch_size,
            batch_size,
            offset=1,
            device=latent.device,
        )  # (2, B * (B - 1) / 2)
        upper_distances = pairwise_squared_distances[
            upper_indices[0], upper_indices[1]
        ]  # (B * (B - 1) / 2,)
        pair_scores = -self.temperature * upper_distances  # (B * (B - 1) / 2,)
        return (
            torch.logsumexp(pair_scores, dim=0)
            + math.log(2.0)
            - 2.0 * math.log(batch_size)
        )  # ()
