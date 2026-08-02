"""Differentiable Masserstein reconstruction loss for MSI spectra."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from ...utils.exceptions import raise_validation_error
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class SpectrumMasserstein(torch.nn.Module):
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
    of created or destroyed signal. A stabilized Sinkhorn solver computes a
    differentiable transport plan. The optional Sinkhorn divergence subtracts
    self-transport bias introduced by entropy regularization.

    A dense Sinkhorn matrix is quadratic in the number of m/z bins and is not
    practical for training on MSI grids. For a regular axis this implementation
    applies the exponentially decaying transport kernel as a one-dimensional
    convolution and evaluates the whole batch together. Terms smaller than
    ``kernel_tolerance`` are omitted. The resulting complexity is
    :math:`O(BNRI)`, where :math:`B` is batch size, :math:`N` is the number of
    bins, :math:`R` is the retained kernel radius, and :math:`I` is the number
    of Sinkhorn iterations. An irregular axis uses a bounded-memory chunked
    fallback with quadratic runtime.

    ``axis_step`` must use the same unit as ``denoising_penalty``. Callers pass
    the physical m/z grid through ``mass_axis``. The uniform ``axis_step`` is a
    fallback for standalone calculations that do not provide an axis.
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
        kernel_tolerance: float = 1e-7,
        chunk_size: int = 512,
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
        :param kernel_tolerance: Smallest retained regular-axis kernel value.
        :type kernel_tolerance: float
        :param chunk_size: Row count used by the irregular-axis fallback.
        :type chunk_size: int
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
        if not 0 < kernel_tolerance < 1:
            raise_validation_error(
                context_name="MassersteinLoss",
                message="kernel_tolerance must be between zero and one.",
            )
        if chunk_size < 1:
            raise_validation_error(
                context_name="MassersteinLoss",
                message="chunk_size must be at least one.",
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
        self.kernel_tolerance = float(kernel_tolerance)
        self.chunk_size = int(chunk_size)
        self._irregular_axis_warning_emitted = False
        self.register_buffer("_mass_axis", torch.empty(0), persistent=False)
        self._config = {
            "denoising_penalty": self.denoising_penalty,
            "entropy_regularization": self.entropy_regularization,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "axis_step": self.axis_step,
            "reduction": reduction,
            "debias": debias,
            "epsilon": self.epsilon,
            "kernel_tolerance": self.kernel_tolerance,
            "chunk_size": self.chunk_size,
        }

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mass_axis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the robust transport cost for a reconstruction batch.

        :param prediction: Reconstructed spectra with shape ``[B, F]``.
        :type prediction: torch.Tensor
        :param target: Reference spectra with shape ``[B, F]``.
        :type target: torch.Tensor
        :param mass_axis: Optional physical axis with shape ``[F]``.
        :type mass_axis: torch.Tensor | None
        :return: Reduced differentiable Masserstein cost.
        :rtype: torch.Tensor
        :raises IncompatibleInterfaceError: If reconstruction tensors are absent.
        :raises ValidationError: If tensor shapes or the configured axis mismatch.
        """
        self._irregular_axis_warning_emitted = False
        reconstruction = prediction
        original = target.to(device=prediction.device, dtype=prediction.dtype)
        if reconstruction.shape != original.shape or reconstruction.ndim != 2:
            raise_validation_error(
                context_name="MassersteinLoss",
                message=(
                    "Original and reconstructed spectra must have equal "
                    "two-dimensional [batch, bins] shapes."
                ),
            )

        if mass_axis is None:
            axis = self._resolve_axis(
                reconstruction.shape[1],
                reconstruction.device,
                reconstruction.dtype,
            )
        else:
            axis = mass_axis.to(
                device=reconstruction.device,
                dtype=reconstruction.dtype,
            )
            if axis.ndim != 1 or axis.numel() != reconstruction.shape[1]:
                raise_validation_error(
                    "MassersteinLoss",
                    "mass_axis must contain one coordinate per spectrum feature.",
                )
        original_mass = torch.clamp(original, min=0.0)
        reconstructed_mass = torch.clamp(reconstruction, min=0.0)
        batch_costs = self._transport_cost_batch(
            original_mass,
            reconstructed_mass,
            axis,
        )
        if self.debias:
            target_bias = self._transport_cost_batch(
                original_mass,
                original_mass,
                axis,
            )
            prediction_bias = self._transport_cost_batch(
                reconstructed_mass,
                reconstructed_mass,
                axis,
            )
            batch_costs = torch.clamp(
                batch_costs - 0.5 * target_bias - 0.5 * prediction_bias,
                min=0.0,
            )
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

    def _transport_cost_batch(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        axis: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary-point transport costs for an entire batch."""
        first_total = first.sum(dim=1)
        second_total = second.sum(dim=1)
        joint_scale = torch.maximum(first_total, second_total)
        nonempty = joint_scale > self.epsilon
        safe_scale = torch.where(nonempty, joint_scale, torch.ones_like(joint_scale))

        first_scaled = first / safe_scale.unsqueeze(1)
        second_scaled = second / safe_scale.unsqueeze(1)
        source = torch.cat(
            (first_scaled, second_scaled.sum(dim=1, keepdim=True)),
            dim=1,
        )
        target = torch.cat(
            (second_scaled, first_scaled.sum(dim=1, keepdim=True)),
            dim=1,
        )
        shared_total = source.sum(dim=1)
        safe_total = torch.clamp(shared_total, min=self.epsilon)
        source = source / safe_total.unsqueeze(1)
        target = target / safe_total.unsqueeze(1)

        dummy_kernel = math.exp(
            -self.denoising_penalty / self.entropy_regularization
        )

        with torch.no_grad():
            work_dtype = (
                torch.float64
                if 2.0 * self.denoising_penalty / self.entropy_regularization > 80
                else axis.dtype
            )
            work_axis = axis.detach().to(dtype=work_dtype)
            detached_source = source.detach().to(dtype=work_dtype)
            detached_target = target.detach().to(dtype=work_dtype)
            minimum_positive = torch.finfo(work_dtype).tiny
            left = torch.ones_like(detached_source)
            right = torch.ones_like(detached_target)
            for _ in range(self.sinkhorn_iterations):
                kernel_right_real = self._real_kernel_matvec(
                    right[:, :-1],
                    work_axis,
                    include_cost=False,
                )
                kernel_right = torch.cat(
                    (
                        kernel_right_real + dummy_kernel * right[:, -1:],
                        dummy_kernel * right[:, :-1].sum(dim=1, keepdim=True)
                        + right[:, -1:],
                    ),
                    dim=1,
                )
                left = detached_source / torch.clamp(
                    kernel_right,
                    min=minimum_positive,
                )

                kernel_left_real = self._real_kernel_matvec(
                    left[:, :-1],
                    work_axis,
                    include_cost=False,
                )
                kernel_left = torch.cat(
                    (
                        kernel_left_real + dummy_kernel * left[:, -1:],
                        dummy_kernel * left[:, :-1].sum(dim=1, keepdim=True)
                        + left[:, -1:],
                    ),
                    dim=1,
                )
                right = detached_target / torch.clamp(
                    kernel_left,
                    min=minimum_positive,
                )

            real_transport_cost = (
                left[:, :-1]
                * self._real_kernel_matvec(
                    right[:, :-1],
                    work_axis,
                    include_cost=True,
                )
            ).sum(dim=1)
            dummy_transport_cost = self.denoising_penalty * dummy_kernel * (
                left[:, :-1].sum(dim=1) * right[:, -1]
                + left[:, -1] * right[:, :-1].sum(dim=1)
            )
            numerical_cost = shared_total.detach().to(dtype=work_dtype) * (
                real_transport_cost + dummy_transport_cost
            )

        left = left.to(device=source.device)
        right = right.to(device=target.device)
        source_dual = torch.where(
            source > self.epsilon,
            source * torch.log(torch.clamp(left, min=torch.finfo(left.dtype).tiny)),
            torch.zeros_like(source),
        ).sum(dim=1)
        target_dual = torch.where(
            target > self.epsilon,
            target * torch.log(torch.clamp(right, min=torch.finfo(right.dtype).tiny)),
            torch.zeros_like(target),
        ).sum(dim=1)
        gradient_surrogate = (
            shared_total
            * self.entropy_regularization
            * (source_dual + target_dual)
        )
        result = numerical_cost + gradient_surrogate - gradient_surrogate.detach()
        result = result.to(dtype=first.dtype)
        return torch.where(nonempty, result, (first + second).sum(dim=1) * 0.0)

    def _real_kernel_matvec(
        self,
        values: torch.Tensor,
        axis: torch.Tensor,
        include_cost: bool,
    ) -> torch.Tensor:
        """Multiply by the real-bin kernel without allocating a full batch matrix."""
        if axis.numel() < 2:
            return values * (0.0 if include_cost else 1.0)
        differences = torch.diff(axis)
        step = differences[0]
        regular_axis = bool(
            torch.allclose(
                differences,
                step.expand_as(differences),
                rtol=1e-2,
                atol=max(self.epsilon, abs(float(step)) * 1e-3),
            )
        )
        if regular_axis and float(step) > 0:
            cutoff = (
                2.0 * self.denoising_penalty
                - self.entropy_regularization * math.log(self.kernel_tolerance)
            )
            radius = min(
                axis.numel() - 1,
                max(1, math.ceil(cutoff / float(step))),
            )
            offsets = torch.arange(
                -radius,
                radius + 1,
                device=axis.device,
                dtype=axis.dtype,
            )
            distances = torch.abs(offsets * step)
            kernel = torch.exp(-distances / self.entropy_regularization)
            if include_cost:
                kernel = kernel * distances
            return F.conv1d(
                values.unsqueeze(1),
                kernel.view(1, 1, -1),
                padding=radius,
            ).squeeze(1)

        if not self._irregular_axis_warning_emitted:
            logger.warning(
                "Masserstein loss received an irregular axis; using the chunked "
                "quadratic-time kernel fallback."
            )
            self._irregular_axis_warning_emitted = True
        outputs = []
        for start in range(0, axis.numel(), self.chunk_size):
            chunk_axis = axis[start : start + self.chunk_size]
            distances = torch.abs(chunk_axis[:, None] - axis[None, :])
            kernel = torch.exp(-distances / self.entropy_regularization)
            if include_cost:
                kernel = kernel * distances
            outputs.append(values @ kernel.transpose(0, 1))
        return torch.cat(outputs, dim=1)
