"""Known spatial partitions and spectral mixtures for clustering validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .partition_analysis import cluster_latents
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class SpectralPartition:
    """Retain a reference raster and its identity-aligned spectral observations.

    :param reference: Integer reference raster, -1 outside circular tissue.
    :param pixel_ids: Original row-major indices of observed pixels, shape ``(N,)``.
    :param xy: Source coordinates, shape ``(N, 2)``.
    :param spectra: Finite nonnegative unit-TIC spectra, shape ``(N, M)``.
    :param amplitudes: Realized component amplitudes, shape ``(N, P)``.
    :param prototypes: Frozen component mixtures, shape ``(C, P)``.
    :param mass_axis: Physical feature axis, shape ``(M,)``.
    """

    reference: np.ndarray
    pixel_ids: np.ndarray
    xy: np.ndarray
    spectra: np.ndarray
    amplitudes: np.ndarray
    prototypes: np.ndarray
    mass_axis: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        """Return reference classes in source-pixel order, shape ``(N,)``."""
        return self.reference.ravel()[self.pixel_ids]


def spatial_reference(layout: str, *, side: int, classes: int) -> np.ndarray:
    """Construct stripes or equal-width radial regions without random labels.

    ``two_stripes`` requires two classes. Circular tissue has radius 0.45*side;
    exterior pixels are unavailable, not an extra class. Radial regions have
    unequal areas, testing small-region sensitivity. Complexity O(side^2).

    :param layout: two_stripes, many_stripes or concentric_rings.
    :param side: Raster side length, at least twice the number of classes.
    :param classes: Prespecified number of anatomical fixture regions.
    :return: Integer reference raster, shape ``(side, side)``.
    :raises ValueError: For invalid settings or a missing region.
    """
    if not isinstance(classes, int) or classes < 2 or not isinstance(side, int) or side < 2 * classes:
        raise ValueError("Reference size must accommodate at least two nonempty classes.")
    if layout == "two_stripes" and classes != 2:
        raise ValueError("two_stripes requires two classes.")
    yy, xx = np.mgrid[:side, :side]  # (H, W), (H, W)
    if layout in {"two_stripes", "many_stripes"}:
        reference = np.minimum(xx * classes // side, classes - 1).astype(np.int32)  # (H, W)
    elif layout == "concentric_rings":
        radius = np.hypot(xx - (side - 1) / 2, yy - (side - 1) / 2)  # (H, W)
        tissue_radius = .45 * side
        reference = np.minimum(np.floor(radius * classes / tissue_radius), classes - 1).astype(np.int32)  # (H, W)
        reference[radius > tissue_radius] = -1
    else:
        raise ValueError("Unknown spatial reference layout.")
    if set(np.unique(reference[reference >= 0])) != set(range(classes)):
        raise ValueError("The reference contains an empty class; increase the raster size.")
    return reference


def spectral_partition(layout: str, *, side: int, classes: int, mass_axis: np.ndarray,
                       peak_positions: list[float], peak_width_mz: float,
                       amplitude_noise: float, seed: int, device: str = "cpu") -> SpectralPartition:
    """Assign mixture prototypes to regions and sample local spectral variability.

    Every region contains the same Gaussian peak components with cyclically
    permuted abundances. Independent multiplicative lognormal component noise
    changes relative abundances; no spatial coordinates enter the spectrum or
    encoder. This is an analytical Gaussian-mixture fixture, not chemical-isotope
    identification or regeneration of the actual pretraining corpus.
    Complexity O(M*P + N*P*M); storage O(N*M + N*P + C*P).

    :param layout: Reference geometry name.
    :param side: Raster side length.
    :param classes: Region/prototype count.
    :param mass_axis: Strictly increasing finite feature axis, shape ``(M,)``.
    :param peak_positions: At least classes distinct in-axis component centres.
    :param peak_width_mz: Positive component width in m/z.
    :param amplitude_noise: Nonnegative finite log-amplitude standard deviation.
    :param seed: Fixed generator seed; realized amplitudes are retained.
    :param device: Numerical generation backend; production comparisons require CUDA.
    :return: Complete known-input fixture with source identity.
    :raises ValueError: For invalid axis/components or nonfinite generated values.
    """
    axis, peaks = np.asarray(mass_axis, dtype=float), np.asarray(peak_positions, dtype=float)
    if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError("A finite strictly increasing mass axis is required.")
    if peaks.ndim != 1 or len(peaks) < classes or len(np.unique(peaks)) != len(peaks) or not np.isfinite(peaks).all() or np.any(peaks < axis.min()) or np.any(peaks > axis.max()):
        raise ValueError("Peak components must be distinct, in-axis and sufficient for all classes.")
    if not np.isfinite(peak_width_mz) or peak_width_mz <= 0 or not np.isfinite(amplitude_noise) or amplitude_noise < 0:
        raise ValueError("Component width and amplitude noise are invalid.")
    reference = spatial_reference(layout, side=side, classes=classes)  # (H, W)
    pixel_ids = np.flatnonzero(reference.ravel() >= 0)  # (N,)
    yy, xx = np.unravel_index(pixel_ids, reference.shape)  # (N,), (N,)
    labels = reference.ravel()[pixel_ids]  # (N,)
    base = np.linspace(.15, .45, len(peaks))  # (P,)
    base[0], base[1] = 3., 1.2
    prototypes = np.stack([np.roll(base, k) for k in range(classes)])  # (C, P)
    # Generate the spectral mixture on the selected numerical backend
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA spectral generation refuses CPU fallback.")
        generator = torch.Generator(device=device).manual_seed(seed)
        prototype_tensor = torch.as_tensor(prototypes, dtype=torch.float32, device=device)  # (C, P)
        label_tensor = torch.as_tensor(labels, device=device)  # (N,)
        noise = torch.randn((len(labels), len(peaks)), device=device, generator=generator)  # (N, P)
        amplitude_tensor = prototype_tensor[label_tensor] * (amplitude_noise * noise).exp()  # (N, P)
        axis_tensor = torch.as_tensor(axis, dtype=torch.float32, device=device)  # (M,)
        peaks_tensor = torch.as_tensor(peaks, dtype=torch.float32, device=device)  # (P,)
        components = (-.5 * ((axis_tensor[None, :] - peaks_tensor[:, None]) / peak_width_mz).square()).exp()  # (P, M)
        component_mass = components.sum(dim=1, keepdim=True)  # (P, 1)
        if not torch.isfinite(component_mass).all() or (component_mass <= 0).any():
            raise ValueError("The sampled axis cannot represent a peak component in float32.")
        components = components / component_mass  # (P, M)
        tensor = amplitude_tensor @ components  # (N, M)
        tensor = tensor / tensor.sum(dim=1, keepdim=True)  # (N, M)
        spectra, amplitudes = tensor.cpu().numpy(), amplitude_tensor.cpu().numpy()
    elif device == "cpu":
        rng = np.random.default_rng(seed)
        amplitudes = prototypes[labels] * np.exp(rng.normal(0, amplitude_noise, (len(labels), len(peaks))))  # (N, P)
        components = np.exp(-.5 * np.square((axis[None, :] - peaks[:, None]) / peak_width_mz))  # (P, M)
        components /= components.sum(axis=1, keepdims=True)  # (P, M)
        spectra = amplitudes @ components  # (N, M)
        spectra /= spectra.sum(axis=1, keepdims=True)  # (N, M)
    else:
        raise ValueError("Unsupported spectral generation device.")
    if not np.isfinite(spectra).all() or np.any(spectra < 0):
        raise ValueError("The synthetic spectrum is numerically invalid.")
    return SpectralPartition(reference, pixel_ids, np.column_stack([xx, yy]), spectra.astype(np.float32),
                             amplitudes, prototypes, axis)


def infer_frozen_encoder(model: Any, spectra: np.ndarray, *, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an existing model without changing training flags or weights.

    :param model: Loaded autoencoder with latent_space/reconstruction outputs.
    :param spectra: Unit-TIC nonnegative input matrix, shape ``(N, M)``.
    :param device: Torch device; CUDA availability is checked explicitly.
    :param batch_size: Maximum forward-pass batch size.
    :return: Actual encoder latents and reconstructions, shapes ``(N, D)`` and ``(N, M)``.
    :raises ValueError: For invalid inputs, outputs or batch settings.
    :raises RuntimeError: For unavailable CUDA.
    """
    x = np.asarray(spectra)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all() or np.any(x < 0) or batch_size < 1 or not np.allclose(x.sum(axis=1), 1, atol=1e-6, rtol=1e-6):
        raise ValueError("Inference requires finite nonnegative unit-TIC spectra and a positive batch size.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable.")
    modes = [(module, module.training) for module in model.modules()]
    latents, reconstructions = [], []
    model.to(device).eval()
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    try:
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                tensor = torch.as_tensor(np.array(x[start:start + batch_size], dtype=np.float32), device=device, dtype=dtype)  # (B, M)
                outputs = model(tensor)
                z, reconstructed = outputs["latent_space"], outputs["reconstruction"]  # (B, D), (B, M)
                if z.ndim != 2 or z.shape[0] != len(tensor) or reconstructed.shape != tensor.shape or not torch.isfinite(z).all() or not torch.isfinite(reconstructed).all():
                    raise ValueError("Nonfinite or incompatible autoencoder outputs.")
                latents.append(z.detach().cpu().numpy())
                reconstructions.append(reconstructed.detach().cpu().numpy())
    finally:
        for module, training in modes:
            module.training = training
    logger.debug("Encoded %s spectra on %s in batches of %s.", len(x), device, batch_size)
    return np.concatenate(latents), np.concatenate(reconstructions)


def cluster_partition_representation(values: np.ndarray, *, method: str, classes: int,
                                     seed: int, min_cluster_size: int, gmm_regularization: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Run fixed comparable methods, including explicit diagonal GMM.

    Diagonal covariance avoids O(M^2) covariance storage for full spectra. The
    covariance assumption and ridge are identical for raw and learned spaces;
    this is a distinct pipeline from the earlier full-covariance GMM demo.

    :param values: Raw TIC spectra or actual encoder latent matrix, shape ``(N, D)``.
    :param method: kmeans, hdbscan or gmm_diag.
    :param classes: Prespecified fixture class count for K-means/GMM.
    :param seed: Cluster initialization seed.
    :param min_cluster_size: Fixed HDBSCAN parameter.
    :param gmm_regularization: Positive diagonal covariance ridge.
    :return: Labels and estimator diagnostics.
    :raises ValueError: For invalid input/settings.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all() or classes < 2 or classes > len(x):
        raise ValueError("Clustering input or class count is invalid.")
    if method == "gmm_diag":
        if not np.isfinite(gmm_regularization) or gmm_regularization <= 0:
            raise ValueError("GMM covariance regularization must be positive.")
        estimator = GaussianMixture(n_components=classes, covariance_type="diag", reg_covar=gmm_regularization,
                                    n_init=3, random_state=seed)
        labels = estimator.fit_predict(x).astype(np.int64)  # (N,)
        return labels, {"converged": bool(estimator.converged_), "iterations": int(estimator.n_iter_),
                        "covariance_type": "diag", "gmm_regularization": gmm_regularization}
    labels = cluster_latents(x, method=method, seed=seed, n_clusters=classes, min_cluster_size=min_cluster_size)
    return labels, {"converged": None, "iterations": None, "covariance_type": None}
