"""Differentiable Masserstein reconstruction loss for MSI spectra."""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import torch

from ...autoencoder_base_criterions  import MSIReconstructionCriterion
from ...criterions_manager import CriterionsManager
from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.exceptions import raise_validation_error
from .....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@CriterionsManager.register_criterion(
    "autoencoder",
    "reconstruction",
    "MassersteinLoss",
)
class MSIMassersteinLoss(MSIReconstructionCriterion):
    r"""Compare spectra by robust optimal transport with an auxiliary sink.

    This loss is a differentiable reconstruction adaptation of the Masserstein
    regression described by Ciach et al. (2021), DOI ``10.1002/rcm.8956``.
    The paper augments the mass axis with an auxiliary point :math:`\omega`.
    Signal may be transported to that point for a fixed cost :math:`\kappa`
    instead of being moved an implausibly long distance in the m/z domain.

    For autoencoder reconstruction, both the input and prediction may contain
    unmatched signal. The implementation therefore uses the symmetric extension
    with an auxiliary point on both sides. The real-to-real ground cost is the
    m/z distance, transport between a real bin and :math:`\omega` costs
    ``denoising_penalty``, and auxiliary-to-auxiliary transport is free. Thus,
    moving one unit through the auxiliary point costs :math:`2\kappa` and is
    preferred over direct transport only for sufficiently distant signal.

    The two spectra are divided by their joint maximum total ion current. This
    preserves their relative mass difference, unlike independent TIC
    normalization. Each measure is then augmented with the other measure's
    total mass, which balances the transport problem while retaining the amount
    of created or destroyed signal. A log-domain Sinkhorn solver computes a
    differentiable transport plan. The optional Sinkhorn divergence subtracts
    self-transport bias introduced by entropy regularization.

    ``axis_step`` must use the same unit as ``denoising_penalty``. When a
    dataset binner exposes ``GetXAxis()``, :meth:`on_phase_start` automatically
    uses that physical m/z grid instead of the fallback uniform axis.
    """

    def __init__(
        self,
        denoising_penalty: float = 0.5,
        entropy_regularization: float = 0.02,
        sinkhorn_iterations: int = 50,
        axis_step: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        debias: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        r"""Initialize the robust optimal-transport reconstruction objective.

        :param denoising_penalty: Cost :math:`\kappa` of moving one unit of
            signal between a real m/z bin and the auxiliary point.
        :type denoising_penalty: float
        :param entropy_regularization: Positive Sinkhorn smoothing coefficient.
        :type entropy_regularization: float
        :param sinkhorn_iterations: Number of log-domain scaling iterations.
        :type sinkhorn_iterations: int
        :param axis_step: Uniform fallback spacing between spectral bins.
        :type axis_step: float
        :param reduction: Batch reduction: ``mean``, ``sum``, or ``none``.
        :type reduction: Literal["mean", "sum", "none"]
        :param debias: Subtract entropy-induced self-transport costs.
        :type debias: bool
        :param epsilon: Numerical threshold used for empty spectra.
        :type epsilon: float
        :raises ValidationError: If a numerical or reduction parameter is invalid.
        """
        super().__init__()
        if denoising_penalty <= 0:
            raise_validation_error(
                context_name="MassersteinLoss",
                message="denoising_penalty must be greater than zero.",
            )
        if entropy_regularization <= 0:
            raise_validation_error(
                context_name="MassersteinLoss",
                message="entropy_regularization must be greater than zero.",
            )
        if sinkhorn_iterations < 1:
            raise_validation_error(
                context_name="MassersteinLoss",
                message="sinkhorn_iterations must be at least one.",
            )
        if axis_step <= 0 or epsilon <= 0:
            raise_validation_error(
                context_name="MassersteinLoss",
                message="axis_step and epsilon must be greater than zero.",
            )
        if reduction not in {"mean", "sum", "none"}:
            raise_validation_error(
                context_name="MassersteinLoss",
                message="reduction must be 'mean', 'sum', or 'none'.",
            )

        self.denoising_penalty = float(denoising_penalty)
        self.entropy_regularization = float(entropy_regularization)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.axis_step = float(axis_step)
        self.reduction = reduction
        self.debias = debias
        self.epsilon = float(epsilon)
        self.register_buffer("_mass_axis", torch.empty(0), persistent=False)
        self._config = {
            "denoising_penalty": self.denoising_penalty,
            "entropy_regularization": self.entropy_regularization,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "axis_step": self.axis_step,
            "reduction": reduction,
            "debias": debias,
            "epsilon": self.epsilon,
        }

    def on_phase_start(
        self,
        model: torch.nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Capture the physical binner axis when the dataset exposes one.

        :param model: Model trained in the current phase; not modified.
        :type model: torch.nn.Module
        :param dataset: Dataset whose active context may expose a binner axis.
        :type dataset: MSIBaseDataset
        :param transient_cache: Shared training cache; not modified.
        :type transient_cache: Dict[str, Any]
        """
        del model, transient_cache
        active_context = getattr(dataset, "active_context", None)
        binner = getattr(active_context, "binner", None)
        axis_getter = getattr(binner, "GetXAxis", None)
        if callable(axis_getter):
            self._mass_axis = torch.as_tensor(axis_getter(), dtype=torch.float64)
            logger.debug(
                "Masserstein loss captured a physical axis with %s bins.",
                self._mass_axis.numel(),
            )

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the robust transport cost for a reconstruction batch.

        :param model_outputs: Model outputs containing ``reconstruction``.
        :type model_outputs: Dict[str, torch.Tensor]
        :param batch_data: Batch containing the original spectra at index 1.
        :type batch_data: Tuple[torch.Tensor, ...]
        :param kwargs: Reserved criterion execution parameters.
        :return: Reduced differentiable Masserstein cost.
        :rtype: torch.Tensor
        :raises IncompatibleInterfaceError: If reconstruction tensors are absent.
        :raises ValidationError: If tensor shapes or the configured axis mismatch.
        """
        del kwargs
        reconstruction, original = self.reconstruction_pair(model_outputs, batch_data)
        if reconstruction.shape != original.shape or reconstruction.ndim != 2:
            raise_validation_error(
                context_name="MassersteinLoss",
                message=(
                    "Original and reconstructed spectra must have equal "
                    "two-dimensional [batch, bins] shapes."
                ),
            )

        axis = self._resolve_axis(
            reconstruction.shape[1],
            reconstruction.device,
            reconstruction.dtype,
        )
        original_mass = torch.clamp(original, min=0.0)
        reconstructed_mass = torch.clamp(reconstruction, min=0.0)
        costs = []
        for target_spectrum, predicted_spectrum in zip(
            original_mass,
            reconstructed_mass,
        ):
            cross_cost = self._transport_cost(
                target_spectrum,
                predicted_spectrum,
                axis,
            )
            if self.debias:
                target_bias = self._transport_cost(
                    target_spectrum,
                    target_spectrum,
                    axis,
                )
                prediction_bias = self._transport_cost(
                    predicted_spectrum,
                    predicted_spectrum,
                    axis,
                )
                cross_cost = torch.clamp(
                    cross_cost - 0.5 * target_bias - 0.5 * prediction_bias,
                    min=0.0,
                )
            costs.append(cross_cost)

        batch_costs = torch.stack(costs)
        if self.reduction == "sum":
            return batch_costs.sum()
        if self.reduction == "none":
            return batch_costs
        return batch_costs.mean()

    def _resolve_axis(
        self,
        bins: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a validated physical or uniform mass axis."""
        if self._mass_axis.numel() == 0:
            return torch.arange(bins, device=device, dtype=dtype) * self.axis_step
        if self._mass_axis.numel() != bins:
            raise_validation_error(
                context_name="MassersteinLoss",
                message=(
                    f"The binner axis has {self._mass_axis.numel()} bins, but "
                    f"the reconstruction has {bins}."
                ),
            )
        return self._mass_axis.to(device=device, dtype=dtype)

    def _transport_cost(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        axis: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one entropy-regularized auxiliary-point transport cost."""
        first_total = first.sum()
        second_total = second.sum()
        joint_scale = torch.maximum(first_total, second_total)
        if joint_scale.detach().item() <= self.epsilon:
            return (first.sum() + second.sum()) * 0.0

        first_scaled = first / joint_scale
        second_scaled = second / joint_scale
        source = torch.cat((first_scaled, second_scaled.sum().reshape(1)))
        target = torch.cat((second_scaled, first_scaled.sum().reshape(1)))
        shared_total = source.sum()
        source = source / shared_total
        target = target / shared_total

        real_cost = torch.abs(axis[:, None] - axis[None, :])
        bins = axis.numel()
        ground_cost = torch.full(
            (bins + 1, bins + 1),
            self.denoising_penalty,
            device=axis.device,
            dtype=axis.dtype,
        )
        ground_cost[:bins, :bins] = real_cost
        ground_cost[-1, -1] = 0.0

        with torch.no_grad():
            log_kernel = -ground_cost / self.entropy_regularization
            negative_infinity = torch.tensor(
                float("-inf"),
                device=axis.device,
                dtype=axis.dtype,
            )
            log_source = torch.where(
                source > self.epsilon,
                torch.log(torch.clamp(source, min=self.epsilon)),
                negative_infinity,
            )
            log_target = torch.where(
                target > self.epsilon,
                torch.log(torch.clamp(target, min=self.epsilon)),
                negative_infinity,
            )
            log_left = torch.zeros_like(source)
            log_right = torch.zeros_like(target)
            for _ in range(self.sinkhorn_iterations):
                log_left = log_source - torch.logsumexp(
                    log_kernel + log_right.unsqueeze(0),
                    dim=1,
                )
                log_right = log_target - torch.logsumexp(
                    log_kernel + log_left.unsqueeze(1),
                    dim=0,
                )

            transport_plan = torch.exp(
                log_left.unsqueeze(1) + log_kernel + log_right.unsqueeze(0)
            )
            numerical_cost = shared_total.detach() * torch.sum(
                transport_plan * ground_cost
            )

        source_dual = torch.where(
            source > self.epsilon,
            source * log_left,
            torch.zeros_like(source),
        ).sum()
        target_dual = torch.where(
            target > self.epsilon,
            target * log_right,
            torch.zeros_like(target),
        ).sum()
        gradient_surrogate = (
            shared_total
            * self.entropy_regularization
            * (source_dual + target_dual)
        )
        return numerical_cost + gradient_surrogate - gradient_surrogate.detach()
