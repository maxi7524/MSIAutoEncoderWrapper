"""Local sensitivity between the TIC simplex and canonical latent directions."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ....utils.logger import get_custom_logger
from ....training.criterions.autoencoder.regularization.contractive_loss import (
    MSIContractiveLoss,
)

logger = get_custom_logger(__name__)


class SensitivityDiagnostic(MSIContractiveLoss):
    """Adapt the training geometry and spectral estimator to a diagnostic map.

    :param output_map: Batch-separable differentiable mapping, ``(B, M) -> (B, D)``.
    :type output_map: Callable
    :param input_geometry: Training geometry, ``euclidean`` or ``fisher_rao``.
    :type input_geometry: str

    REMARK: This adapter deliberately reuses the criterion's protected geometry
    and power-iteration methods. Tests compare it with dense operators and the
    public criterion so changes to that internal interface cannot pass silently.
    It does not change the training loss or its output-space conventions.
    """

    def __init__(self, output_map: Callable, input_geometry: str = "fisher_rao"):
        super().__init__(penalty_metric="spectral", input_geometry=input_geometry)
        self.output_map = output_map

    def _spectral_output_function(self, model, batch_size):
        return self.output_map

    def estimate_spectral_squared(self, model, inputs):
        """Run the actual training estimator on the supplied diagnostic map.

        :param model: Encoder owner, retained weakly during estimation.
        :type model: torch.nn.Module
        :param inputs: Gradient-enabled spectra, shape ``(B, M)``.
        :type inputs: torch.Tensor
        :return: Detached per-spectrum estimates, shape ``(B,)``.
        :rtype: torch.Tensor
        """
        self.on_phase_start(model, None, {})
        outputs = self.output_map(inputs)  # (B, D)
        return self._spectral_squared_norm(inputs, outputs).detach()  # (B,)

    def transform_jacobian(self, jacobian, inputs):
        """Apply the training input-geometry adjoint to every Jacobian row.

        :param jacobian: Output derivatives, shape ``(B, D, M)``.
        :type jacobian: torch.Tensor
        :param inputs: TIC spectra, shape ``(B, M)``.
        :type inputs: torch.Tensor
        :return: Geometry-weighted operator, shape ``(B, D, M)``.
        :rtype: torch.Tensor
        """
        self._validate_tic_simplex_inputs(inputs)
        return torch.stack(
            [self._geometry_adjoint_direction(row, inputs)
             for row in jacobian.unbind(dim=1)], dim=1,
        )  # (B, D, M)


def canonical_direction(latent, gamma, beta):
    """Undo affine LayerNorm and normalize its centered direction.

    :param latent: Post-LayerNorm values, shape ``(B, D)``.
    :type latent: torch.Tensor
    :param gamma: Nonzero affine scales, shape ``(D,)``.
    :type gamma: torch.Tensor
    :param beta: Affine offsets, shape ``(D,)``.
    :type beta: torch.Tensor
    :return: Canonical values and unit directions, both ``(B, D)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: On nonfinite values, unstable scales or zero directions.

    The derivative of the direction includes both output tangent projections.
    Centering removes floating-point residuals in LayerNorm's zero-mean constraint.
    """
    if not all(torch.isfinite(value).all() for value in (latent, gamma, beta)):
        raise ValueError("Canonicalization requires finite values.")
    if torch.any(gamma.abs() < torch.finfo(latent.dtype).eps**0.5):
        raise ValueError("LayerNorm gamma is too close to zero.")
    canonical = (latent - beta) / gamma  # (B, D)
    centered = canonical - canonical.mean(dim=1, keepdim=True)  # (B, D)
    radius = torch.linalg.vector_norm(centered, dim=1, keepdim=True)  # (B, 1)
    if torch.any(radius <= torch.finfo(latent.dtype).eps):
        raise ValueError("Canonical direction is undefined at zero radius.")
    return canonical, centered / radius  # both (B, D)


def exact_output_jacobian(outputs, inputs):
    """Differentiate each coordinate of a batch-separable output map.

    :param outputs: Gradient-enabled map values, shape ``(B, D)``.
    :type outputs: torch.Tensor
    :param inputs: Gradient-enabled inputs, shape ``(B, M)``.
    :type inputs: torch.Tensor
    :return: Detached Jacobian, shape ``(B, D, M)``.
    :rtype: torch.Tensor

    Retains the graph for another output map from the same forward pass. Callers
    should release inputs and outputs after each batch. Batch-coupled models
    require a different differentiation strategy and must not use this function.
    """
    logger.debug("Computing exact Jacobian: inputs=%s outputs=%s", inputs.shape, outputs.shape)
    rows = [torch.autograd.grad(outputs[:, index].sum(), inputs, retain_graph=True)[0].detach()
            for index in range(outputs.shape[1])]
    return torch.stack(rows, dim=1)  # (B, D, M)


def metric_operators(jacobian, inputs, gaussian_kernel, bin_spacing):
    """Construct input-geometry operators without dense input metric matrices.

    :param jacobian: Derivatives in the selected output space, ``(B, D, M)``.
    :type jacobian: torch.Tensor
    :param inputs: Nonnegative unit-TIC spectra, ``(B, M)``.
    :type inputs: torch.Tensor
    :param gaussian_kernel: Symmetric, normalized, odd-length smoothing kernel.
    :type gaussian_kernel: torch.Tensor
    :param bin_spacing: Positive uniform m/z grid spacing.
    :type bin_spacing: float
    :return: Operators ``(B, D, K)``; ``K=M-1`` for Cramer, otherwise ``M``.
    :rtype: dict[str, torch.Tensor]
    :raises ValueError: On invalid shapes, kernel, spacing or nonfinite derivatives.

    Fisher uses the training adjoint ``P_x diag(sqrt(x))``. Cramer uses an
    incidence matrix divided by sqrt(spacing). Gaussian uses ``P K P`` with
    zero-padding and the Euclidean zero-sum projector P on both sides.
    At zero-intensity bins Fisher describes the fixed-support tangent limit.
    """
    if jacobian.ndim != 3 or inputs.shape != (jacobian.shape[0], jacobian.shape[2]):
        raise ValueError("Expected Jacobian (B, D, M) and inputs (B, M).")
    if not torch.isfinite(jacobian).all() or not 0 < bin_spacing < float("inf"):
        raise ValueError("Derivatives must be finite and bin spacing positive.")
    kernel = gaussian_kernel.to(jacobian)
    if (kernel.ndim != 1 or kernel.numel() % 2 != 1
            or not torch.isfinite(kernel).all() or torch.any(kernel < 0)
            or not torch.allclose(kernel, kernel.flip(0))
            or not torch.isclose(kernel.sum(), kernel.new_tensor(1.0))):
        raise ValueError("Expected a symmetric normalized odd-length Gaussian kernel.")
    fisher = SensitivityDiagnostic(lambda value: value)
    # Input tangent operators
    centered = jacobian - jacobian.mean(dim=-1, keepdim=True)  # (B, D, M)
    difference = jacobian[..., 1:] - jacobian[..., :-1]  # (B, D, M-1)
    flattened = centered.reshape(-1, 1, centered.shape[-1])  # (B*D, 1, M)
    smoothed = F.conv1d(flattened, kernel.view(1, 1, -1), padding=kernel.numel() // 2)
    smoothed = smoothed.reshape_as(centered)  # (B, D, M)
    return {
        "euclidean_ambient": jacobian,
        "euclidean_simplex": centered,
        "fisher_diagonal": jacobian * inputs.sqrt()[:, None, :],  # (B, D, M)
        "fisher_simplex": fisher.transform_jacobian(jacobian, inputs),
        "cramer_simplex": difference / bin_spacing**0.5,
        "gaussian_simplex": smoothed - smoothed.mean(dim=-1, keepdim=True),
    }


def operator_statistics(operator):
    """Compute exact trace and spectral statistics from a small output Gram matrix.

    :param operator: Finite weighted Jacobian, shape ``(B, D, K)``.
    :type operator: torch.Tensor
    :return: Gram matrices ``(B, D, D)``, Frobenius squared ``(B,)`` and
        spectral squared ``(B,)`` in float64.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    :raises ValueError: On nonfinite operators or materially indefinite Gram matrices.
    """
    if not torch.isfinite(operator).all():
        raise ValueError("Operator contains nonfinite values.")
    # REMARK: Accumulate in float64, particularly for weak tangent directions.
    precise = operator.double()
    gram = precise @ precise.transpose(-1, -2)  # (B, D, D)
    eigenvalues = torch.linalg.eigvalsh(gram)  # (B, D)
    tolerance = 100 * torch.finfo(gram.dtype).eps * gram.shape[-1] * gram.abs().amax(dim=(-2, -1))
    if torch.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError("Gram matrix is materially indefinite.")
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
    return gram, trace, eigenvalues[:, -1].clamp_min(0)  # (B,D,D), (B,), (B,)


def paired_direction_angles(left, right):
    """Compute stable angles between paired nonzero directions, in radians.

    :param left: First directions, shape ``(B, D)``.
    :type left: torch.Tensor
    :param right: Second directions of matching shape.
    :type right: torch.Tensor
    :return: Angles in ``[0, pi]``, shape ``(B,)``.
    :rtype: torch.Tensor
    :raises ValueError: On incompatible shapes, nonfinite or zero directions.
    """
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("Angles require matching (B, D) arrays.")
    normalized = []
    for value in (left, right):
        norms = torch.linalg.vector_norm(value.double(), dim=1, keepdim=True)  # (B, 1)
        if not torch.isfinite(value).all() or torch.any(norms == 0):
            raise ValueError("Angles require finite nonzero directions.")
        normalized.append(value.double() / norms)  # (B, D)
    a, b = normalized
    # REMARK: Half-angle formula avoids arccos cancellation close to zero.
    return 2 * torch.atan2(
        torch.linalg.vector_norm(a - b, dim=1),
        torch.linalg.vector_norm(a + b, dim=1),
    )  # (B,)
