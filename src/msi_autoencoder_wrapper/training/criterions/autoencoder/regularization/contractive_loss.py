"""Contractive encoder regularization through local input Jacobians."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ...autoencoder_base_criterions import MSIRegularizationCriterion
from ...criterions_manager import CriterionsManager
from .....data import SpectrumBatch
from .....utils.exceptions import (
    raise_incompatible_interface_error,
    raise_validation_error,
)


SUPPORTED_CONTRACTIVE_METHODS = frozenset(
    {"exact_autograd_jacobian", "approximate_hutchinson_vjp"}
)


@CriterionsManager.register_criterion("autoencoder", "regularization", "ContractiveLoss")
class MSIContractiveLoss(MSIRegularizationCriterion):
    r"""Penalize the squared Frobenius norm of ``d enc(x) / d x``.

    ``exact_autograd_jacobian`` accumulates one reverse-mode derivative for
    every latent coordinate. ``approximate_hutchinson_vjp`` draws Rademacher
    directions in latent space and uses
    ``E[||J(x)^T v||²] = ||J(x)||²_F`` without materializing ``(B, D, M)``.
    Both methods evaluate local derivatives at each observed spectrum; they do
    not sample or interpolate between different spectra.

    :param calculation_method: Exact or approximate public strategy name.
    :type calculation_method: str
    :param num_probes: Rademacher directions used by the approximate method.
    :type num_probes: int
    :param probe_distribution: Currently ``rademacher``.
    :type probe_distribution: str
    :param latent_source: Model output key, or ``auto`` for ``latent_space``.
    :type latent_source: str
    """

    requires_input_grad = True

    def __init__(
        self,
        calculation_method: str = "approximate_hutchinson_vjp",
        num_probes: int = 1,
        probe_distribution: str = "rademacher",
        latent_source: str = "auto",
    ) -> None:
        super().__init__()
        if calculation_method not in SUPPORTED_CONTRACTIVE_METHODS:
            raise_validation_error(
                "ContractiveLoss",
                f"calculation_method must be one of {sorted(SUPPORTED_CONTRACTIVE_METHODS)}.",
            )
        if not isinstance(num_probes, int) or isinstance(num_probes, bool) or num_probes < 1:
            raise_validation_error("ContractiveLoss", "num_probes must be positive.")
        if probe_distribution != "rademacher":
            raise_validation_error(
                "ContractiveLoss", "probe_distribution must be 'rademacher'."
            )
        self.calculation_method = calculation_method
        self.num_probes = num_probes
        self.latent_source = "latent_space" if latent_source == "auto" else latent_source
        self._config = {
            "calculation_method": calculation_method,
            "num_probes": num_probes,
            "probe_distribution": probe_distribution,
            "latent_source": latent_source,
        }

    def on_batch_start(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        transient_cache: Dict[str, Any],
    ) -> Tuple[torch.Tensor, ...]:
        """Enable input derivatives before the shared model forward pass."""
        del transient_cache
        spectra = batch_data.spectra if isinstance(batch_data, SpectrumBatch) else batch_data[1]
        spectra.requires_grad_(True)
        return batch_data

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the batch mean local Jacobian squared norm."""
        del kwargs
        if self.latent_source not in model_outputs:
            raise_incompatible_interface_error(
                "ContractiveLoss",
                f"Model outputs must contain '{self.latent_source}'.",
            )
        inputs = batch_data.spectra if isinstance(batch_data, SpectrumBatch) else batch_data[1]
        latent = model_outputs[self.latent_source]
        batch_size = inputs.shape[0]
        latent = latent[:batch_size]  # (B, D)
        if not inputs.requires_grad or not latent.requires_grad:
            raise_incompatible_interface_error(
                "ContractiveLoss",
                "Encoder inputs and latent outputs must retain an autograd graph.",
            )
        if self.calculation_method == "exact_autograd_jacobian":
            squared_norm = torch.zeros(batch_size, device=inputs.device, dtype=inputs.dtype)  # (B,)
            for latent_index in range(latent.shape[1]):
                gradient = torch.autograd.grad(
                    latent[:, latent_index].sum(),
                    inputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]  # (B, M)
                squared_norm = squared_norm + gradient.square().sum(dim=1)  # (B,)
            return squared_norm.mean()

        estimate = torch.zeros(batch_size, device=inputs.device, dtype=inputs.dtype)  # (B,)
        for _ in range(self.num_probes):
            direction = torch.empty_like(latent).bernoulli_(0.5).mul_(2).sub_(1)  # (B, D)
            vjp = torch.autograd.grad(
                (latent * direction).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )[0]  # (B, M)
            estimate = estimate + vjp.square().sum(dim=1)  # (B,)
        return (estimate / self.num_probes).mean()
