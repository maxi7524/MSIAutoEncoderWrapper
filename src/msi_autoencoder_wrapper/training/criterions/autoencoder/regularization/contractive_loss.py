"""Contractive encoder regularization through local input Jacobians."""

from __future__ import annotations

import math
import weakref
from typing import Any, Callable, Dict, Tuple

import torch

from ...autoencoder_base_criterions import MSIRegularizationCriterion
from ...criterions_manager import CriterionsManager
from .....data import SpectrumBatch
from .....utils.exceptions import (
    raise_incompatible_interface_error,
    raise_validation_error,
)
from .canonicalized_latent import CanonicalizedLatentMixin


SUPPORTED_CONTRACTIVE_METHODS = frozenset(
    {"exact_autograd_jacobian", "approximate_hutchinson_vjp"}
)
SUPPORTED_PENALTY_METRICS = frozenset({"frobenius", "spectral", "hinged"})
SUPPORTED_PENALIZED_SPACES = frozenset({"z", "u"})
_SPECTRAL_POWER_ITERATION_STEPS = 3


@CriterionsManager.register_criterion("autoencoder", "regularization", "ContractiveLoss")
class MSIContractiveLoss(CanonicalizedLatentMixin, MSIRegularizationCriterion):
    r"""Penalize local input-Jacobian sensitivity of the encoder.

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
    :param penalty_metric: ``frobenius``, ``spectral``, or a Frobenius ``hinged``
        penalty.
    :type penalty_metric: str
    :param hinge_threshold: Threshold ``tau`` for ``hinged`` penalties.
    :type hinge_threshold: float | None
    :param penalized_space: Raw model latents (``z``) or LayerNorm-canonicalized
        latents (``u``).
    :type penalized_space: str
    """

    requires_input_grad = True

    def __init__(
        self,
        calculation_method: str = "approximate_hutchinson_vjp",
        num_probes: int = 1,
        probe_distribution: str = "rademacher",
        latent_source: str = "auto",
        penalty_metric: str = "frobenius",
        hinge_threshold: float | None = None,
        penalized_space: str = "z",
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
        if penalty_metric not in SUPPORTED_PENALTY_METRICS:
            raise_validation_error(
                "ContractiveLoss",
                f"penalty_metric must be one of {sorted(SUPPORTED_PENALTY_METRICS)}.",
            )
        if penalized_space not in SUPPORTED_PENALIZED_SPACES:
            raise_validation_error(
                "ContractiveLoss",
                f"penalized_space must be one of {sorted(SUPPORTED_PENALIZED_SPACES)}.",
            )
        if penalty_metric == "hinged":
            if (
                isinstance(hinge_threshold, bool)
                or not isinstance(hinge_threshold, (float, int))
                or not math.isfinite(float(hinge_threshold))
                or float(hinge_threshold) < 0
            ):
                raise_validation_error(
                    "ContractiveLoss",
                    "hinge_threshold must be a finite non-negative number for penalty_metric='hinged'.",
                )
        elif hinge_threshold is not None:
            raise_validation_error(
                "ContractiveLoss",
                "hinge_threshold is only valid for penalty_metric='hinged'.",
            )
        self.calculation_method = calculation_method
        self.num_probes = num_probes
        self.latent_source = "latent_space" if latent_source == "auto" else latent_source
        self.penalty_metric = penalty_metric
        self.hinge_threshold = None if hinge_threshold is None else float(hinge_threshold)
        self.penalized_space = penalized_space
        self._requires_canonicalized_latent = penalized_space == "u"
        self._model_reference: weakref.ReferenceType[torch.nn.Module] | None = None
        self._config = {
            "calculation_method": calculation_method,
            "num_probes": num_probes,
            "probe_distribution": probe_distribution,
            "latent_source": latent_source,
            "penalty_metric": penalty_metric,
            "hinge_threshold": hinge_threshold,
            "penalized_space": penalized_space,
        }

    def on_phase_start(
        self,
        model: torch.nn.Module,
        dataset: Any,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Capture dependencies required by canonical and spectral penalties."""
        super().on_phase_start(model, dataset, transient_cache)
        if self.penalty_metric == "spectral":
            self._model_reference = weakref.ref(model)

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
        penalized_latent = (
            self.canonicalize_latent(latent)
            if self.penalized_space == "u"
            else latent
        )  # (B, D)
        if self.penalty_metric == "spectral":
            return self._spectral_squared_norm(inputs, penalized_latent).mean()

        squared_norm = self._frobenius_squared_norm(inputs, penalized_latent)
        if self.penalty_metric == "hinged":
            assert self.hinge_threshold is not None
            threshold_squared = self.hinge_threshold**2
            squared_norm = torch.relu(squared_norm - threshold_squared)  # (B,)
        return squared_norm.mean()

    def _frobenius_squared_norm(
        self,
        inputs: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-spectrum squared Frobenius Jacobian norms.

        :param inputs: Encoder inputs with shape ``(B, M)``.
        :type inputs: torch.Tensor
        :param latent: Penalized latents with shape ``(B, D)``.
        :type latent: torch.Tensor
        :return: Squared norms with shape ``(B,)``.
        :rtype: torch.Tensor
        """
        batch_size = inputs.shape[0]
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
            return squared_norm

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
        return estimate / self.num_probes  # (B,)

    def _spectral_squared_norm(
        self,
        inputs: torch.Tensor,
        reference_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate per-spectrum squared spectral Jacobian norms by power iteration.

        :param inputs: Encoder inputs with shape ``(B, M)``.
        :type inputs: torch.Tensor
        :param reference_latent: Latents from the shared model forward pass with
            shape ``(B, D)``.
        :type reference_latent: torch.Tensor
        :return: Spectral-norm estimates with shape ``(B,)``.
        :rtype: torch.Tensor
        :raises IncompatibleInterfaceError: If the phase lifecycle hook did not
            supply the active model.
        """
        if self._model_reference is None or (model := self._model_reference()) is None:
            raise_incompatible_interface_error(
                "ContractiveLoss",
                "Spectral penalties require on_phase_start before evaluation.",
            )

        # Power iteration over J^T J, separately normalized for each spectrum.
        input_direction = self._normalize_per_sample(torch.randn_like(inputs))  # (B, M)
        output_function = self._spectral_output_function(model, inputs.shape[0])
        for _ in range(_SPECTRAL_POWER_ITERATION_STEPS):
            _, jvp = torch.func.jvp(
                output_function,
                (inputs,),
                (input_direction,),
            )  # (B, D)
            latent_direction = self._normalize_per_sample(jvp).detach()  # (B, D)
            vjp = torch.autograd.grad(
                (reference_latent * latent_direction).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )[0]  # (B, M)
            input_direction = self._normalize_per_sample(vjp).detach()  # (B, M)

        _, final_jvp = torch.func.jvp(
            output_function,
            (inputs,),
            (input_direction,),
        )  # (B, D)
        return final_jvp.square().sum(dim=1)  # (B,)

    def _spectral_output_function(
        self,
        model: torch.nn.Module,
        batch_size: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the differentiable input-to-penalized-latent map for JVPs."""

        def encode(spectra: torch.Tensor) -> torch.Tensor:
            # The standard latent representation depends only on the encoder.
            # Avoiding decoder, projector, and heads is critical because JVPs
            # rerun this mapping four times for three power iterations.
            if self.latent_source == "latent_space":
                encoded = model.encoder(spectra)
                latent = (
                    encoded["latent_space"]
                    if isinstance(encoded, dict)
                    else encoded
                )[:batch_size]  # (B, D)
            else:
                outputs = model(spectra)
                if self.latent_source not in outputs:
                    raise_incompatible_interface_error(
                        "ContractiveLoss",
                        f"Model outputs must contain '{self.latent_source}'.",
                    )
                latent = outputs[self.latent_source][:batch_size]  # (B, D)
            return (
                self.canonicalize_latent(latent)
                if self.penalized_space == "u"
                else latent
            )  # (B, D)

        return encode

    @staticmethod
    def _normalize_per_sample(values: torch.Tensor) -> torch.Tensor:
        """Normalize nonzero row vectors while retaining the derivative graph."""
        norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)  # (B, 1)
        return values / norms.clamp_min(torch.finfo(values.dtype).eps)  # (B, K)
