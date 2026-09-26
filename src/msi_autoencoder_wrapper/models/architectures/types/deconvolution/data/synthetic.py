"""Deterministic Torch generation of sparse mixtures from global dictionaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts import DeconvolutionBatch
from .dictionary import GlobalCandidateDictionary
from ......utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class SyntheticDeconvolutionConfig:
    """Parameters controlling reproducible sparse synthetic mixtures.

    :param min_components: Minimum active candidates per synthetic spectrum.
    :param max_components: Maximum active candidates per synthetic spectrum.
    :param abundance_log_mean: Mean of log-normal abundance draws.
    :param abundance_log_std: Standard deviation of log-normal abundance draws.
    :param noise_std: Standard deviation of additive Gaussian noise.
    :param normalization: ``none``, ``tic``, or ``max``.
    :type min_components: int
    :type max_components: int
    :type abundance_log_mean: float
    :type abundance_log_std: float
    :type noise_std: float
    :type normalization: str
    """

    min_components: int = 1
    max_components: int = 4
    abundance_log_mean: float = 0.0
    abundance_log_std: float = 0.5
    noise_std: float = 0.0
    normalization: str = "none"

    def __post_init__(self) -> None:
        if self.min_components < 1 or self.max_components < self.min_components:
            raise ValueError("Component counts must satisfy 1 <= min <= max.")
        if self.abundance_log_std < 0 or self.noise_std < 0:
            raise ValueError("Standard deviations must be non-negative.")
        if self.normalization not in {"none", "tic", "max"}:
            raise ValueError("normalization must be 'none', 'tic', or 'max'.")


class SyntheticDeconvolutionGenerator:
    """Generate reproducible Torch batches from one global candidate dictionary."""

    def __init__(
        self,
        dictionary: GlobalCandidateDictionary,
        config: SyntheticDeconvolutionConfig = SyntheticDeconvolutionConfig(),
        *,
        seed: int = 42,
    ) -> None:
        """Initialize the deterministic mixture generator.

        :param dictionary: Global dictionary used for every generated batch.
        :param config: Mixture and noise configuration.
        :param seed: Base local random seed.
        :type dictionary: GlobalCandidateDictionary
        :type config: SyntheticDeconvolutionConfig
        :type seed: int
        """
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if config.max_components > dictionary.candidate_count:
            raise ValueError("max_components cannot exceed the global candidate count.")
        self.dictionary = dictionary
        self.config = config
        self.seed = int(seed)
        logger.info(
            "Initialized synthetic deconvolution generator with %s global candidates.",
            dictionary.candidate_count,
        )

    def generate(self, batch_size: int, *, epoch: int = 0) -> DeconvolutionBatch:
        """Generate one reproducible batch without touching global RNG state.

        :param batch_size: Number of spectra to generate.
        :param epoch: Deterministic stream offset.
        :type batch_size: int
        :type epoch: int
        :return: Exact synthetic spectra and their abundance/support targets.
        :rtype: DeconvolutionBatch
        """
        if batch_size < 1 or epoch < 0:
            raise ValueError("batch_size must be positive and epoch non-negative.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        candidate_count = self.dictionary.candidate_count
        logger.debug(
            "Generating deconvolution batch: batch_size=%s epoch=%s candidates=%s.",
            batch_size,
            epoch,
            candidate_count,
        )
        component_counts = torch.randint(
            self.config.min_components,
            self.config.max_components + 1,
            (batch_size,),
            generator=generator,
        )  # (B,)
        abundances = torch.zeros(batch_size, candidate_count, dtype=self.dictionary.matrix.dtype)  # (B, C)

        # Sparse candidate composition
        ## Every row samples an unordered subset from the same global dictionary.
        for row_index, component_count in enumerate(component_counts.tolist()):
            indices = torch.randperm(candidate_count, generator=generator)[:component_count]  # (S,)
            normal = torch.randn(component_count, generator=generator, dtype=abundances.dtype)  # (S,)
            abundances[row_index, indices] = torch.exp(
                self.config.abundance_log_mean + self.config.abundance_log_std * normal
            )  # (S,)

        presence = abundances > 0  # (B, C)
        spectra = abundances @ self.dictionary.matrix.transpose(0, 1)  # (B, M)
        if self.config.noise_std > 0:
            noise = torch.randn(spectra.shape, generator=generator, dtype=spectra.dtype)  # (B, M)
            spectra = torch.clamp_min(spectra + self.config.noise_std * noise, 0.0)  # (B, M)
        spectra, abundances = self._normalize(spectra, abundances)
        return DeconvolutionBatch(
            spectra=spectra,  # (B, M)
            dictionary=self.dictionary,
            abundances=abundances,  # (B, C)
            presence=presence,  # (B, C)
            metadata={"generator": "synthetic_deconvolution", "seed": self.seed, "epoch": epoch},
        )

    def _normalize(
        self,
        spectra: torch.Tensor,
        abundances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a shared spectrum/abundance scale transformation.

        :param spectra: Unnormalized spectra, shape ``(B, M)``.
        :param abundances: Unnormalized abundances, shape ``(B, C)``.
        :type spectra: torch.Tensor
        :type abundances: torch.Tensor
        :return: Normalized spectra and abundances.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if self.config.normalization == "none":
            return spectra, abundances
        denominator = (
            spectra.sum(dim=1, keepdim=True)
            if self.config.normalization == "tic"
            else spectra.amax(dim=1, keepdim=True)
        )  # (B, 1)
        denominator = denominator.clamp_min(torch.finfo(spectra.dtype).eps)  # (B, 1)
        return spectra / denominator, abundances / denominator  # (B, M), (B, C)
