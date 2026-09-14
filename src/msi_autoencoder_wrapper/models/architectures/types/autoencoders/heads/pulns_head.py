"""Classifier and negative-selector head required by the PULNS procedure."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from .base_head import MSIBaseHead
from .jerm_head import _projection


@ArchitecturesManager.register_component("autoencoder", "head", "PULNSHead")
class PULNSHead(MSIBaseHead):
    """Expose classifier representation, binary score, and PULNS selector.

    The regular head output is the classifier logit. ``representation`` and
    ``selector_logits`` are invoked by the episode controller, because the
    selector state depends on P/U membership and prior actions and therefore
    cannot be built in a label-agnostic model forward pass.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        representation_dim: int | None = None,
        classifier_hidden_dim: int | None = None,
        selector_hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if representation_dim is None:
            representation_dim = latent_dim
        if isinstance(representation_dim, bool) or representation_dim < 1:
            raise ValueError("representation_dim must be a positive integer.")
        if len(selector_hidden_dims) != 2 or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in selector_hidden_dims
        ):
            raise ValueError("selector_hidden_dims must contain two positive widths.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must belong to [0, 1).")
        self.representation_network = _projection(
            latent_dim, representation_dim, classifier_hidden_dim, dropout
        )
        self.classifier = nn.Linear(representation_dim, output_dim)
        self.selector = nn.Sequential(
            nn.Linear(3 * representation_dim, selector_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(selector_hidden_dims[0], selector_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(selector_hidden_dims[1], 1),
        )
        self.output_dim = output_dim
        self.representation_dim = representation_dim
        self._config = {
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "representation_dim": representation_dim,
            "classifier_hidden_dim": classifier_hidden_dim,
            "selector_hidden_dims": selector_hidden_dims,
            "dropout": dropout,
        }

    def representation(self, z: torch.Tensor) -> torch.Tensor:
        """Return the classifier representation ``x_prime``, shape ``(B, D)``."""
        return self.representation_network(z)  # (B, D)

    def selector_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Return one selection logit per PULNS state, shape ``(...,)``."""
        if state.shape[-1] != 3 * self.representation_dim:
            raise ValueError("PULNS selector state has an incompatible final dimension.")
        return self.selector(state).squeeze(-1)  # (...,)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return positive-class classifier logits with shape ``(B, C)``."""
        return self.classifier(self.representation(z))  # (B, C)
