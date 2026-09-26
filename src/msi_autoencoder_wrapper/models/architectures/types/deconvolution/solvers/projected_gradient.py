"""Torch-native non-negative sparse projected-gradient deconvolution."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..contracts import DeconvolutionResult
from ..data.dictionary import GlobalCandidateDictionary
from ......utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


class NonnegativeProjectedGradientSolver(nn.Module):
    """Solve a non-negative sparse least-squares decomposition in Torch.

    The solver is intentionally a deterministic baseline for later LISTA
    variants. It keeps the complete numerical path on the input device.

    :param iterations: Number of projected-gradient updates.
    :param step_size: Optional fixed update step. If omitted, a stable step is
        derived from the dictionary spectral norm per forward pass.
    :param l1_weight: Non-negative sparsity coefficient.
    :type iterations: int
    :type step_size: float | None
    :type l1_weight: float
    """

    def __init__(
        self,
        iterations: int = 32,
        step_size: float | None = None,
        l1_weight: float = 0.0,
    ) -> None:
        """Initialize the Torch-only solver."""
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive when provided.")
        if l1_weight < 0:
            raise ValueError("l1_weight must be non-negative.")
        self.iterations = int(iterations)
        self.step_size = step_size
        self.l1_weight = float(l1_weight)

    def forward(
        self,
        spectra: torch.Tensor,
        dictionary: GlobalCandidateDictionary | torch.Tensor,
    ) -> DeconvolutionResult:
        """Estimate non-negative abundances for one batch of spectra.

        :param spectra: Dense spectra with shape ``(B, M)``.
        :param dictionary: Global dictionary or matrix with shape ``(M, C)``.
        :type spectra: torch.Tensor
        :type dictionary: GlobalCandidateDictionary | torch.Tensor
        :return: Deconvolution estimates and reconstruction diagnostics.
        :rtype: DeconvolutionResult
        :raises ValueError: If the input dimensions or devices disagree.
        """
        matrix = dictionary.matrix if isinstance(dictionary, GlobalCandidateDictionary) else dictionary
        if spectra.ndim != 2 or matrix.ndim != 2:
            raise ValueError("spectra and dictionary must have shapes (B, M) and (M, C).")
        if spectra.shape[1] != matrix.shape[0]:
            raise ValueError("spectra and dictionary feature counts disagree.")
        if spectra.device != matrix.device:
            raise ValueError("spectra and dictionary must be on the same device.")
        if spectra.dtype != matrix.dtype:
            raise ValueError("spectra and dictionary must share a dtype.")

        # Projected non-negative sparse optimization
        ## The objective is 0.5 * ||A K^T - X||^2 + l1_weight * ||A||_1.
        step_size = self._resolve_step_size(matrix)
        logger.debug(
            "Projected-gradient deconvolution: batch=%s features=%s candidates=%s iterations=%s.",
            spectra.shape[0],
            spectra.shape[1],
            matrix.shape[1],
            self.iterations,
        )
        abundance = torch.zeros(
            (spectra.shape[0], matrix.shape[1]),
            device=spectra.device,
            dtype=spectra.dtype,
        )  # (B, C)
        for _ in range(self.iterations):
            reconstruction = abundance @ matrix.transpose(0, 1)  # (B, M)
            gradient = (reconstruction - spectra) @ matrix  # (B, C)
            abundance = torch.relu(abundance - step_size * (gradient + self.l1_weight))  # (B, C)

        reconstruction = abundance @ matrix.transpose(0, 1)  # (B, M)
        residual = spectra - reconstruction  # (B, M)
        objective = 0.5 * residual.square().sum(dim=1) + self.l1_weight * abundance.sum(dim=1)  # (B,)
        return DeconvolutionResult(
            abundances=abundance,
            reconstruction=reconstruction,
            residual=residual,
            objective=objective,
        )

    def _resolve_step_size(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return a stable Torch step size for the current global dictionary.

        :param matrix: Dictionary matrix with shape ``(M, C)``.
        :type matrix: torch.Tensor
        :return: Scalar update step.
        :rtype: torch.Tensor
        """
        if self.step_size is not None:
            return torch.as_tensor(self.step_size, dtype=matrix.dtype, device=matrix.device)
        largest_singular = torch.linalg.matrix_norm(matrix, ord=2)  # ()
        lipschitz_constant = largest_singular.square().clamp_min(torch.finfo(matrix.dtype).eps)  # ()
        return lipschitz_constant.reciprocal()  # ()
