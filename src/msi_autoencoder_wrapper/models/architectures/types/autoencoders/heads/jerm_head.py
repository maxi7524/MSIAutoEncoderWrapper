"""Posterior and propensity head for instance-dependent PU learning."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from .base_head import MSIBaseHead


def _projection(input_dim: int, output_dim: int, hidden_dim: int | None, dropout: float) -> nn.Module:
    """Build one configurable prediction projection."""
    layers: list[nn.Module] = []
    if hidden_dim is not None:
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        input_dim = hidden_dim
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def _spectrum_statistics(input_spectrum: torch.Tensor) -> torch.Tensor:
    """Derive scale and sparsity descriptors from one model-space spectrum."""
    nonnegative = input_spectrum.clamp_min(0.0)  # (B, M)
    total = nonnegative.sum(dim=1, keepdim=True)  # (B, 1)
    normalized = nonnegative / total.clamp_min(torch.finfo(nonnegative.dtype).tiny)  # (B, M)
    entropy = -(normalized * normalized.clamp_min(torch.finfo(nonnegative.dtype).tiny).log()).sum(
        dim=1,
        keepdim=True,
    )  # (B, 1)
    return torch.cat(
        (
            total,
            nonnegative.amax(dim=1, keepdim=True),
            nonnegative.mean(dim=1, keepdim=True),
            nonnegative.std(dim=1, keepdim=True, unbiased=False),
            (nonnegative > 0).to(nonnegative.dtype).mean(dim=1, keepdim=True),
            entropy,
        ),
        dim=1,
    )  # (B, 6)


@ArchitecturesManager.register_component("autoencoder", "head", "JERMHead")
class JERMHead(MSIBaseHead):
    """Predict posterior and annotation propensity logits for every class.

    The final dimension stores ``(posterior_logit, propensity_logit)``. The
    propensity branch may use the latent vector, the original model-space
    spectrum, their concatenation, or six deterministic spectrum statistics
    (total intensity, base peak, mean, standard deviation, occupied fraction,
    and entropy). The statistic modes are a baseline for later integration of
    external METASPACE-style feature providers.

    :param latent_dim: Width of the encoder latent vector.
    :type latent_dim: int
    :param output_dim: Number of molecular classes.
    :type output_dim: int
    :param propensity_feature_mode: ``latent``, ``spectrum``,
        ``latent_spectrum``, ``spectrum_statistics``, or
        ``latent_spectrum_statistics``.
    :type propensity_feature_mode: str
    :param input_dim: Model-spectrum width required by modes using ``spectrum``.
    :type input_dim: int | None
    :param hidden_dim: Optional posterior projection width.
    :type hidden_dim: int | None
    :param propensity_hidden_dim: Optional propensity projection width.
    :type propensity_hidden_dim: int | None
    :param dropout: Projection dropout probability.
    :type dropout: float
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        propensity_feature_mode: str = "latent",
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        propensity_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        supported_modes = {
            "latent",
            "spectrum",
            "latent_spectrum",
            "spectrum_statistics",
            "latent_spectrum_statistics",
        }
        if propensity_feature_mode not in supported_modes:
            raise ValueError(f"propensity_feature_mode must be one of {sorted(supported_modes)}.")
        requires_spectrum = propensity_feature_mode != "latent"
        if propensity_feature_mode in {"spectrum", "latent_spectrum"} and (
            isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1
        ):
            raise ValueError("input_dim must be a positive integer for spectrum propensity features.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must belong to [0, 1).")
        if propensity_feature_mode == "latent":
            propensity_input_dim = latent_dim
        elif propensity_feature_mode == "spectrum":
            propensity_input_dim = input_dim
        elif propensity_feature_mode == "latent_spectrum":
            propensity_input_dim = latent_dim + int(input_dim)
        elif propensity_feature_mode == "spectrum_statistics":
            propensity_input_dim = 6
        else:
            propensity_input_dim = latent_dim + 6
        self.posterior = _projection(latent_dim, output_dim, hidden_dim, dropout)
        self.propensity = _projection(
            int(propensity_input_dim), output_dim, propensity_hidden_dim, dropout
        )
        self.output_dim = output_dim
        self.propensity_feature_mode = propensity_feature_mode
        self.requires_input_spectrum = requires_spectrum
        self._config = {
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "propensity_feature_mode": propensity_feature_mode,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "propensity_hidden_dim": propensity_hidden_dim,
            "dropout": dropout,
        }

    def forward(
        self,
        z: torch.Tensor,
        input_spectrum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return posterior and propensity logits with shape ``(B, C, 2)``."""
        if self.requires_input_spectrum and input_spectrum is None:
            raise ValueError("JERM spectrum propensity features require input_spectrum.")
        if self.propensity_feature_mode == "latent":
            propensity_features = z  # (B, D)
        elif self.propensity_feature_mode == "spectrum":
            propensity_features = input_spectrum  # (B, M)
        elif self.propensity_feature_mode == "latent_spectrum":
            propensity_features = torch.cat((z, input_spectrum), dim=1)  # (B, D + M)
        elif self.propensity_feature_mode == "spectrum_statistics":
            propensity_features = _spectrum_statistics(input_spectrum)  # (B, 6)
        else:
            propensity_features = torch.cat(
                (z, _spectrum_statistics(input_spectrum)),
                dim=1,
            )  # (B, D + 6)
        posterior_logits = self.posterior(z)  # (B, C)
        propensity_logits = self.propensity(propensity_features)  # (B, C)
        return torch.stack((posterior_logits, propensity_logits), dim=-1)  # (B, C, 2)
