"""Metrics defined between dense spectra."""

from __future__ import annotations

import torch

from ...utils.exceptions import raise_validation_error


def _validate_spectra(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise_validation_error(
            "SpectrumMetric", "prediction and target must have equal [B, F] shapes."
        )
    return prediction, target.to(device=prediction.device, dtype=prediction.dtype)


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean squared error independently for every spectrum."""
    prediction, target = _validate_spectra(prediction, target)
    return (prediction - target).square().mean(dim=1)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean absolute error independently for every spectrum."""
    prediction, target = _validate_spectra(prediction, target)
    return (prediction - target).abs().mean(dim=1)


def cosine_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return bounded cosine similarity independently for every spectrum."""
    prediction, target = _validate_spectra(prediction, target)
    numerator = (prediction * target).sum(dim=1)
    denominator = torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(
        target, dim=1
    )
    values = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return values.clamp(-1.0, 1.0)


def spectral_angle(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the angular distance between every pair of spectra."""
    return torch.acos(cosine_similarity(prediction, target))


def tic_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return reconstructed minus reference total ion current per spectrum."""
    prediction, target = _validate_spectra(prediction, target)
    return prediction.sum(dim=1) - target.sum(dim=1)


def feature_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return dataset-batch mean error profiles along the feature axis."""
    prediction, target = _validate_spectra(prediction, target)
    residual = prediction - target
    return {
        "feature_mse": residual.square().mean(dim=0),
        "feature_mae": residual.abs().mean(dim=0),
        "feature_bias": residual.mean(dim=0),
    }


def sobolev(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mass_axis: torch.Tensor | None = None,
    derivative_weight: float = 0.5,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return zero- and first-order spectral error per sample.

    When ``mass_axis`` is supplied the derivative is divided by physical m/z
    spacing. Omitting it retains the legacy unit-bin derivative definition.
    """
    prediction, target = _validate_spectra(prediction, target)
    if derivative_weight < 0 or epsilon <= 0:
        raise_validation_error(
            "SpectrumSobolev", "derivative_weight must be non-negative and epsilon positive."
        )
    weights = 1.0 / (target.mean(dim=1) + epsilon)
    zero_order = (prediction - target).square().mean(dim=1) * weights
    target_derivative = target[:, 1:] - target[:, :-1]
    prediction_derivative = prediction[:, 1:] - prediction[:, :-1]
    if mass_axis is not None:
        axis = mass_axis.to(device=prediction.device, dtype=prediction.dtype)
        if axis.ndim != 1 or axis.numel() != prediction.shape[1]:
            raise_validation_error(
                "SpectrumSobolev", "mass_axis must contain one coordinate per feature."
            )
        spacing = axis[1:] - axis[:-1]
        if not bool(torch.all(spacing > 0)):
            raise_validation_error(
                "SpectrumSobolev", "mass_axis must be strictly increasing."
            )
        target_derivative = target_derivative / spacing
        prediction_derivative = prediction_derivative / spacing
    first_order = (
        (prediction_derivative - target_derivative).square().mean(dim=1) * weights
    )
    return zero_order + derivative_weight * first_order
