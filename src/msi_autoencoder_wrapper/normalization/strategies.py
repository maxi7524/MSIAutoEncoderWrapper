"""Built-in vectorized samplewise MSI normalizations."""

from __future__ import annotations

from typing import Literal, Mapping

import torch

from .base import BaseNormalization, NormalizationCapabilities


class ScalarNormalization(BaseNormalization):
    """Normalize each spectrum by one positive scalar."""

    capabilities = NormalizationCapabilities(
        invertible=True,
        preserves_nonnegativity=True,
        preserves_linear_intensity=True,
        samplewise_scalar=True,
        can_inverse_after_binning=True,
        can_inverse_after_inverse_binning=True,
    )

    def __init__(
        self,
        kind: Literal["tic", "max", "l2"] = "tic",
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__(epsilon=epsilon)
        if kind not in {"tic", "max", "l2"}:
            raise ValueError("kind must be 'tic', 'max', or 'l2'.")
        self.kind = kind

    def transform(
        self,
        values: torch.Tensor,
        *,
        sample_indices: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Apply a dense or packed segmented scalar normalization."""
        absolute = values.abs()
        if values.ndim == 2:
            if self.kind == "tic":
                scale = absolute.sum(dim=1, keepdim=True)
            elif self.kind == "max":
                scale = absolute.amax(dim=1, keepdim=True)
            else:
                scale = torch.linalg.vector_norm(values, dim=1, keepdim=True)
            safe = torch.where(scale > self.epsilon, scale, torch.ones_like(scale))
            return torch.where(scale > self.epsilon, values / safe, torch.zeros_like(values)), {"scale": scale}

        if values.ndim != 1 or sample_indices is None or batch_size is None:
            raise ValueError("Packed normalization requires [N] values, sample_indices, and batch_size.")
        scale = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
        if self.kind in {"tic", "l2"}:
            source = absolute if self.kind == "tic" else values.square()
            scale.scatter_add_(0, sample_indices, source)
            if self.kind == "l2":
                scale = scale.sqrt()
        else:
            scale.scatter_reduce_(0, sample_indices, absolute, reduce="amax", include_self=True)
        safe = torch.where(scale > self.epsilon, scale, torch.ones_like(scale))
        normalized = values / safe[sample_indices]
        normalized = torch.where(scale[sample_indices] > self.epsilon, normalized, torch.zeros_like(values))
        return normalized, {"scale": scale.unsqueeze(1)}

    def inverse(
        self,
        values: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        *,
        sample_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multiply values by the recorded per-sample scale."""
        scale = state["scale"]
        if values.ndim == 2:
            return values * scale
        if values.ndim == 1 and sample_indices is not None:
            return values * scale.reshape(-1)[sample_indices]
        raise ValueError("Packed inverse normalization requires sample_indices.")

    def get_config(self) -> dict[str, object]:
        """Return reproducible constructor parameters."""
        return {"kind": self.kind, "epsilon": self.epsilon}
