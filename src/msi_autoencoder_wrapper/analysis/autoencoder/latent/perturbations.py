"""Chemically motivated finite perturbations of TIC-normalized spectra.

These are the perturbation families a mass-spectrometry encoder is expected to be
robust against, applied as finite transforms rather than as infinitesimal directions.
They complement the local sensitivity operators in :mod:`.sensitivity`: those describe
the Jacobian at a point, while these ask how far the code actually moves for a change
a real instrument or sample could produce.

Two of the four are deterministic given the spectrum (a discrete m/z shift and a peak
broadening) and two are stochastic (per-bin multiplicative and additive noise). The
distinction matters when interpreting a result: for the deterministic pair there is one
perturbation per spectrum and no averaging over draws is possible, so a spread across
spectra is entirely between-spectrum variation.

Every transform ends in TIC renormalization, matching the dataset's own preprocessing,
so the perturbed spectrum lies on the same simplex as the original and the comparison is
between two valid inputs rather than between an input and an off-manifold vector. The
effective perturbation is therefore always

.. math::

    \\delta = T(x) - x,

taken after renormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

PERTURBATION_NAMES = (
    "mz_shift",
    "width_jitter",
    "intensity_lognormal",
    "additive_noise",
)

#: Transforms that are a deterministic function of the spectrum. The remaining names in
#: :data:`PERTURBATION_NAMES` draw fresh noise per repeat.
DETERMINISTIC_PERTURBATIONS = frozenset({"mz_shift", "width_jitter"})


@dataclass(frozen=True)
class PerturbationSettings:
    """Amplitudes defining each perturbation family.

    :param shift_bins: Discrete m/z displacement of ``mz_shift``, in bins.
    :type shift_bins: int
    :param extra_width_sigma: Additional Gaussian width, in bins, convolved into the
        spectrum by ``width_jitter``. It adds in quadrature to the instrument's own
        empirical peak width.
    :type extra_width_sigma: float
    :param base_width_sigma: The instrument's empirical peak width in bins, obtained
        from a half-maximum measurement. ``width_jitter`` convolves with
        ``sqrt(base**2 + extra**2)``.
    :type base_width_sigma: float
    :param lognormal_sigma: Standard deviation of the per-bin log-multiplier of
        ``intensity_lognormal``.
    :type lognormal_sigma: float
    :param noise_fraction: Standard deviation of ``additive_noise``, as a fraction of
        the spectrum's own mean intensity.
    :type noise_fraction: float
    """

    shift_bins: int = 1
    extra_width_sigma: float = 1.0
    base_width_sigma: float = 0.42
    lognormal_sigma: float = 0.05
    noise_fraction: float = 0.02


@dataclass(frozen=True)
class PerturbationResult:
    """One applied perturbation and the diagnostics needed to trust it.

    :param spectra: Perturbed, TIC-renormalized spectra, shape ``(N, M)``.
    :type spectra: torch.Tensor
    :param boundary_mass_lost: Mass pushed off the m/z grid before renormalization,
        per spectrum, shape ``(N,)``. Non-zero only for transforms that move mass
        along the axis; a large value means the transform is partly an amputation
        rather than a displacement.
    :type boundary_mass_lost: torch.Tensor
    :param introduced_mass: Mass placed into bins that were empty in the original
        spectrum, per spectrum, shape ``(N,)``. A perturbation with large introduced
        mass leaves the support of the original and is outside the reach of any
        local, support-respecting sensitivity measure.
    :type introduced_mass: torch.Tensor
    :param input_displacement: Euclidean norm of ``delta`` per spectrum, shape
        ``(N,)``, giving the size of the perturbation in the ambient input metric.
    :type input_displacement: torch.Tensor
    """

    spectra: torch.Tensor
    boundary_mass_lost: torch.Tensor
    introduced_mass: torch.Tensor
    input_displacement: torch.Tensor


def gaussian_kernel(sigma_bins: float, device: Optional[torch.device] = None) -> torch.Tensor:
    """Build a normalized 1-D Gaussian convolution kernel.

    :param sigma_bins: Standard deviation in bins; must be positive.
    :type sigma_bins: float
    :param device: Device to place the kernel on.
    :type device: torch.device | None
    :return: Kernel of odd length covering three standard deviations, summing to one.
    :rtype: torch.Tensor
    :raises ValueError: If ``sigma_bins`` is not positive.
    """
    if not sigma_bins > 0:
        raise ValueError(f"sigma_bins must be positive, got {sigma_bins}.")
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    positions = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (positions / sigma_bins) ** 2)
    return kernel / kernel.sum()


def normalize_tic(spectra: torch.Tensor) -> torch.Tensor:
    """Restore unit total ion current without inventing mass in empty bins.

    :param spectra: Non-negative spectra, shape ``(N, M)``.
    :type spectra: torch.Tensor
    :return: Spectra whose rows sum to one, shape ``(N, M)``.
    :rtype: torch.Tensor
    :raises ValueError: If any spectrum has no remaining mass, which would make the
        normalization undefined rather than merely extreme.
    """
    totals = spectra.sum(dim=1, keepdim=True)  # (N, 1)
    if bool(torch.any(totals <= 0)):
        raise ValueError("Perturbation removed all mass from at least one spectrum.")
    return spectra / totals


def perturb_spectra(
    spectra: torch.Tensor,
    name: str,
    *,
    settings: Optional[PerturbationSettings] = None,
    generator: Optional[torch.Generator] = None,
) -> PerturbationResult:
    """Apply one named perturbation family to a batch of TIC-normalized spectra.

    The four families, with :math:`x` the input spectrum on the simplex:

    - ``mz_shift`` displaces the spectrum by ``shift_bins`` along the m/z axis without
      wraparound, so :math:`T(x)_m = x_{m-k}`. Mass leaving the last bin is reported as
      ``boundary_mass_lost`` rather than being folded back into the first, which a
      circular shift would do and which has no physical counterpart.
    - ``width_jitter`` convolves with a Gaussian of width
      :math:`\\sqrt{\\sigma_{\\mathrm{base}}^2+\\sigma_{\\mathrm{extra}}^2}`, modelling a
      loss of instrument resolution.
    - ``intensity_lognormal`` multiplies each bin by :math:`\\exp(\\epsilon_m)` with
      :math:`\\epsilon_m` independent normal draws. A single global factor would cancel
      under renormalization, so the multiplier is per bin, modelling local intensity
      variation such as ion suppression.
    - ``additive_noise`` adds independent normal noise scaled to the spectrum's own mean
      intensity and clips at zero, since intensities cannot be negative.

    :param spectra: TIC-normalized spectra, shape ``(N, M)``.
    :type spectra: torch.Tensor
    :param name: One of :data:`PERTURBATION_NAMES`.
    :type name: str
    :param settings: Perturbation amplitudes; defaults to :class:`PerturbationSettings`.
    :type settings: PerturbationSettings | None
    :param generator: Generator for the stochastic families. Required for a
        reproducible result on those; ignored by the deterministic ones.
    :type generator: torch.Generator | None
    :return: The perturbed spectra with their diagnostics.
    :rtype: PerturbationResult
    :raises ValueError: If ``name`` is not a known perturbation family.
    """
    if name not in PERTURBATION_NAMES:
        raise ValueError(f"Unknown perturbation '{name}'; expected one of {PERTURBATION_NAMES}.")
    settings = settings or PerturbationSettings()
    lost = torch.zeros(spectra.shape[0], device=spectra.device)

    # Perturbation families
    if name == "mz_shift":
        ## Non-circular displacement: mass leaving the grid is recorded, not wrapped
        shift = settings.shift_bins
        transformed = torch.zeros_like(spectra)
        transformed[:, shift:] = spectra[:, :-shift]
        lost = spectra[:, -shift:].sum(dim=1)  # (N,)
    elif name == "width_jitter":
        sigma = float(
            np.sqrt(settings.base_width_sigma**2 + settings.extra_width_sigma**2)
        )
        kernel = gaussian_kernel(sigma, device=spectra.device).to(spectra.dtype)
        transformed = F.conv1d(
            spectra[:, None, :], kernel[None, None, :], padding=kernel.numel() // 2
        ).squeeze(1)  # (N, M)
        lost = spectra.sum(dim=1) - transformed.sum(dim=1)  # (N,)
    else:
        ## Stochastic families: one fresh draw per call
        noise = torch.randn(spectra.shape, generator=generator).to(spectra)
        if name == "intensity_lognormal":
            transformed = spectra * torch.exp(settings.lognormal_sigma * noise)
        else:
            scale = settings.noise_fraction * spectra.mean(dim=1, keepdim=True)  # (N, 1)
            transformed = (spectra + scale * noise).clamp_min(0.0)

    normalized = normalize_tic(transformed)  # (N, M)
    empty_bins = spectra == 0
    return PerturbationResult(
        spectra=normalized,
        boundary_mass_lost=lost,
        introduced_mass=(normalized * empty_bins).sum(dim=1),  # (N,)
        input_displacement=torch.linalg.vector_norm(normalized - spectra, dim=1),  # (N,)
    )


def interpolate_perturbation(
    spectra: torch.Tensor, perturbed: torch.Tensor, amplitude: float
) -> torch.Tensor:
    """Scale a finite perturbation to a fraction of its full size.

    Interpolating toward the perturbed spectrum rather than re-applying the transform
    at a smaller setting keeps the perturbation *direction* fixed while the amplitude
    varies, which is what an amplitude sweep is meant to isolate. The result is
    renormalized because a convex combination of two simplex points is already on the
    simplex only up to floating-point error.

    :param spectra: Original spectra, shape ``(N, M)``.
    :type spectra: torch.Tensor
    :param perturbed: Fully perturbed spectra, shape ``(N, M)``.
    :type perturbed: torch.Tensor
    :param amplitude: Fraction of the full perturbation, typically in ``[0, 1]``.
    :type amplitude: float
    :return: Interpolated spectra, shape ``(N, M)``.
    :rtype: torch.Tensor
    """
    blended = spectra + amplitude * (perturbed - spectra)  # (N, M)
    return normalize_tic(blended.clamp_min(0.0))


def perturbation_diagnostics_frame(
    results: Dict[str, PerturbationResult]
) -> list[Dict[str, object]]:
    """Summarize how physically well-posed each applied perturbation is.

    :param results: Applied perturbations keyed by family name.
    :type results: Dict[str, PerturbationResult]
    :return: One record per family with the mean and maximum of the boundary mass
        lost, the mass introduced into originally empty bins, and the input-space
        displacement.
    :rtype: list[Dict[str, object]]
    """
    records = []
    for name, result in results.items():
        records.append(
            {
                "perturbation": name,
                "deterministic": name in DETERMINISTIC_PERTURBATIONS,
                "mean_boundary_mass_lost": float(result.boundary_mass_lost.mean()),
                "max_boundary_mass_lost": float(result.boundary_mass_lost.max()),
                "mean_introduced_mass": float(result.introduced_mass.mean()),
                "max_introduced_mass": float(result.introduced_mass.max()),
                "mean_input_displacement": float(result.input_displacement.mean()),
                "max_input_displacement": float(result.input_displacement.max()),
            }
        )
    logger.info("Summarized %s perturbation family diagnostics.", len(records))
    return records


def perturbation_norms(
    delta: torch.Tensor,
    spectra: torch.Tensor,
    *,
    gaussian_sigma_bins: float = 0.42,
    bin_spacing: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Measure one perturbation's length under each candidate input metric.

    These are *primal* lengths of the perturbation itself, :math:`\\lVert\\delta\\rVert_X`,
    not the dual operator norms :func:`..sensitivity.metric_operators` builds from a
    Jacobian. They answer a different question: how large a given input change is
    considered to be, before any model is involved. Two metrics can disagree by orders
    of magnitude on the same change, which is the whole reason the metric choice
    matters.

    The four metrics, with :math:`x` the spectrum on the simplex:

    - **euclidean**, :math:`\\lVert\\delta\\rVert_2`, treats every bin alike and is the
      convention the original penalty assumed;
    - **cramer**, :math:`\\lVert C\\delta\\rVert_2` with :math:`C` the cumulative sum
      scaled by the grid spacing, is the Hilbert-space relative of the transport
      distance and grows with displacement along the m/z axis;
    - **gaussian**, :math:`\\lVert K_\\sigma\\delta\\rVert_2`, keeps only the part of the
      change that survives smoothing at the instrument's resolution;
    - **fisher_rao**, :math:`\\sqrt{\\sum_m \\delta_m^2/x_m}`, weights a change by the
      inverse of the intensity already present. It is finite only where the
      perturbation stays on the spectrum's support: a bin with :math:`x_m=0` and
      :math:`\\delta_m\\neq0` contributes an infinite term. The value returned is
      computed over the support only and must be read together with
      ``off_support_fraction``, which reports how much of the perturbation's squared
      Euclidean length that truncation discards. A large off-support fraction means the
      Fisher length is not merely large but undefined, and no Fisher-based local bound
      constrains that direction.

    :param delta: Perturbations, shape ``(N, M)``.
    :type delta: torch.Tensor
    :param spectra: The unperturbed spectra the perturbations are taken at, ``(N, M)``.
    :type spectra: torch.Tensor
    :param gaussian_sigma_bins: Smoothing width for the Gaussian metric, in bins.
    :type gaussian_sigma_bins: float
    :param bin_spacing: Uniform m/z grid spacing, used to scale the Cramer integral.
    :type bin_spacing: float
    :return: Per-spectrum lengths under each metric, each shape ``(N,)``, plus
        ``off_support_fraction``.
    :rtype: Dict[str, torch.Tensor]
    :raises ValueError: On mismatched shapes or a non-positive bin spacing.
    """
    if delta.shape != spectra.shape or delta.ndim != 2:
        raise ValueError("delta and spectra must share the shape (N, M).")
    if not bin_spacing > 0:
        raise ValueError(f"bin_spacing must be positive, got {bin_spacing}.")

    euclidean = torch.linalg.vector_norm(delta, dim=1)  # (N,)

    ## Cramer: cumulative displacement along the mass axis
    cumulative = torch.cumsum(delta, dim=1) * bin_spacing  # (N, M)
    cramer = torch.linalg.vector_norm(cumulative, dim=1)  # (N,)

    ## Gaussian: the part of the change that survives instrument-resolution smoothing
    kernel = gaussian_kernel(gaussian_sigma_bins, device=delta.device).to(delta.dtype)
    blurred = F.conv1d(
        delta[:, None, :], kernel[None, None, :], padding=kernel.numel() // 2
    ).squeeze(1)  # (N, M)
    gaussian = torch.linalg.vector_norm(blurred, dim=1)  # (N,)

    ## Fisher-Rao: finite only on the support, so the truncation is quantified
    on_support = spectra > 0
    safe = torch.where(on_support, spectra, torch.ones_like(spectra))
    fisher_squared = torch.where(on_support, delta**2 / safe, torch.zeros_like(delta)).sum(dim=1)
    off_support_squared = torch.where(on_support, torch.zeros_like(delta), delta**2).sum(dim=1)
    total_squared = (delta**2).sum(dim=1)
    off_support_fraction = torch.where(
        total_squared > 0, off_support_squared / total_squared, torch.zeros_like(total_squared)
    )  # (N,)

    return {
        "euclidean": euclidean,
        "cramer": cramer,
        "gaussian": gaussian,
        "fisher_rao": torch.sqrt(fisher_squared),
        "off_support_fraction": off_support_fraction,
    }
