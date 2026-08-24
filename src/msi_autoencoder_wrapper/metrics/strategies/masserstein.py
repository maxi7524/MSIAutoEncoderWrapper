"""Differentiable one-dimensional Wasserstein metric for MSI spectra."""

from __future__ import annotations

from typing import Literal

import torch

from ...utils.exceptions import raise_validation_error
from ...normalization import NormalizationTrace, ScalarNormalization
from ..base import MetricRequirements, SpectrumMetric
from ..compatibility import validate_metric_inputs


class SpectrumMasserstein(SpectrumMetric):
    r"""Compare spectra on a shared ordered m/z grid.

    For one-dimensional measures on a fixed ordered axis, the exact
    Wasserstein-1 distance is obtained from the cumulative mass difference.
    The implementation therefore requires neither a transport matrix nor an
    iterative optimal-transport solver. Its complexity is ``O(BN)``.

    """

    requirements = MetricRequirements(
        requires_nonnegative=True,
        requires_linear_intensity=True,
        accepts_samplewise_scalar=True,
    )

    def __init__(
        self,
        axis_step: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        epsilon: float = 1e-12,
        tic_atol: float = 1e-5,
        tic_rtol: float = 1e-4,
    ) -> None:
        """Initialize the exact one-dimensional Wasserstein metric."""
        super().__init__()
        if axis_step <= 0 or epsilon <= 0 or tic_atol < 0 or tic_rtol < 0:
            raise_validation_error(
                "MassersteinLoss",
                "axis_step and epsilon must be positive; TIC tolerances must be non-negative.",
            )
        if reduction not in {"mean", "sum", "none"}:
            raise_validation_error("MassersteinLoss", "reduction must be 'mean', 'sum', or 'none'.")

        self.axis_step = float(axis_step)
        self.reduction = reduction
        self.epsilon = float(epsilon)
        self.tic_atol = float(tic_atol)
        self.tic_rtol = float(tic_rtol)
        self._tic_normalization = ScalarNormalization(
            kind="tic",
            epsilon=self.epsilon,
        )
        self.register_buffer("_mass_axis", torch.empty(0), persistent=False)
        self.register_buffer("_bin_widths", torch.empty(0), persistent=False)
        self._config = {
            "axis_step": self.axis_step,
            "reduction": reduction,
            "epsilon": self.epsilon,
            "tic_atol": self.tic_atol,
            "tic_rtol": self.tic_rtol,
        }

    @property
    def has_mass_axis(self) -> bool:
        """Return whether the metric has a configured spectral axis."""
        return self._mass_axis.numel() > 0

    def set_mass_axis(self, mass_axis: torch.Tensor) -> None:
        """Cache the ordered m/z axis and adjacent bin widths once."""
        axis = torch.as_tensor(mass_axis).detach().to(dtype=torch.float32).clone()
        if axis.ndim != 1 or axis.numel() < 1:
            raise_validation_error("MassersteinLoss", "mass_axis must be a non-empty one-dimensional tensor.")
        if not bool(torch.isfinite(axis).all()) or (
            axis.numel() > 1 and not bool(torch.all(axis[1:] > axis[:-1]))
        ):
            raise_validation_error(
                "MassersteinLoss",
                "mass_axis must contain finite, strictly increasing coordinates.",
            )
        self._mass_axis = axis
        self._bin_widths = torch.diff(axis)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mass_axis: torch.Tensor | None = None,
        normalization_trace: NormalizationTrace | None = None,
        inputs_tic_normalized: bool = False,
    ) -> torch.Tensor:
        """Compute exact 1D Wasserstein costs in a shared TIC representation."""
        reconstruction = prediction
        original = target.to(device=prediction.device, dtype=prediction.dtype)
        if reconstruction.ndim != 2 or reconstruction.shape != original.shape:
            raise_validation_error(
                "MassersteinLoss",
                "Original and reconstructed spectra must have equal two-dimensional [batch, bins] shapes.",
            )
        validate_metric_inputs(
            self.requirements,
            reconstruction,
            original,
            trace=normalization_trace,
        )
        self._configure_axis_for_forward(mass_axis, reconstruction.shape[1])
        self._move_geometry_to(reconstruction.device, reconstruction.dtype)

        # Representation contract
        ## A declared TIC space must already be enforced by preprocessing and
        ## the decoder output normalization. The metric validates this contract
        ## instead of silently hiding a model configuration error.
        if inputs_tic_normalized:
            self._validate_tic_representation(reconstruction, "prediction")
            self._validate_tic_representation(original, "target")
            prediction_normalized = reconstruction  # (B, M)
            target_normalized = original  # (B, M)
        else:
            prediction_normalized, _ = self._tic_normalization.transform(
                reconstruction,
            )  # (B, M)
            target_normalized, _ = self._tic_normalization.transform(original)

        # One-dimensional transport cost
        ## Cumulative imbalance crossing each boundary is the transported mass.
        cumulative_difference = torch.cumsum(
            prediction_normalized - target_normalized,
            dim=1,
        )
        if reconstruction.shape[1] == 1:
            sample_costs = cumulative_difference[:, 0] * 0.0
        else:
            sample_costs = (
                cumulative_difference[:, :-1].abs()
                * self._bin_widths.unsqueeze(0)
            ).sum(dim=1)

        if self.reduction == "sum":
            return sample_costs.sum()
        if self.reduction == "none":
            return sample_costs
        return sample_costs.mean()

    def _validate_tic_representation(
        self,
        values: torch.Tensor,
        argument_name: str,
    ) -> None:
        """Reject tensors that falsely declare a TIC representation."""
        total_mass = values.sum(dim=1)  # (B,)
        unit_mass = torch.isclose(
            total_mass,
            torch.ones_like(total_mass),
            atol=self.tic_atol,
            rtol=self.tic_rtol,
        )  # (B,)
        zero_mass = total_mass <= self.epsilon  # (B,)
        valid = unit_mass | zero_mass  # (B,)
        if not bool(valid.all()):
            invalid_count = int((~valid).sum().item())
            raise_validation_error(
                "MassersteinLoss",
                (
                    f"The {argument_name} declares TIC normalization, but "
                    f"{invalid_count} spectrum/spectra have total mass neither "
                    "one nor zero. Normalize the model output and input before "
                    "evaluating this metric."
                ),
            )

    def _configure_axis_for_forward(
        self,
        mass_axis: torch.Tensor | None,
        bins: int,
    ) -> None:
        """Configure fallback axis once and reject incompatible spaces."""
        if not self.has_mass_axis:
            self.set_mass_axis(
                torch.arange(bins, dtype=torch.float32) * self.axis_step
                if mass_axis is None
                else mass_axis
            )
        elif mass_axis is not None:
            supplied_axis = torch.as_tensor(mass_axis).to(
                device=self._mass_axis.device,
                dtype=self._mass_axis.dtype,
            )
            if supplied_axis.shape != self._mass_axis.shape or not bool(
                torch.allclose(supplied_axis, self._mass_axis)
            ):
                raise_validation_error(
                    "MassersteinLoss",
                    "mass_axis differs from the configured static transport geometry.",
                )
        if self._mass_axis.numel() != bins:
            raise_validation_error(
                "MassersteinLoss",
                f"The configured mass axis has {self._mass_axis.numel()} bins, but the reconstruction has {bins}.",
            )

    def _move_geometry_to(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move cached axis widths to the active tensor device and dtype."""
        self._mass_axis = self._mass_axis.to(device=device, dtype=dtype)
        self._bin_widths = self._bin_widths.to(device=device, dtype=dtype)
