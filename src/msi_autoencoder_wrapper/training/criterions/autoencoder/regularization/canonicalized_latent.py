"""Differentiable canonicalization for bottleneck LayerNorm representations."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.exceptions import raise_incompatible_interface_error


class CanonicalizedLatentMixin:
    """Provide differentiable canonicalization of an encoder latent space.

    The mixin relies on the CNN autoencoder contract in which the final module
    of ``model.encoder.bottleneck_layer`` is an affine :class:`torch.nn.LayerNorm`.
    Its parameters remain attached to the autograd graph, unlike the equivalent
    post-training NumPy utility used by latent-geometry analysis.
    """

    _requires_canonicalized_latent = True

    def on_phase_start(
        self,
        model: nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Capture the bottleneck LayerNorm needed for canonicalization.

        :param model: Model optimized during the current phase.
        :type model: torch.nn.Module
        :param dataset: Dataset used during the current phase.
        :type dataset: MSIBaseDataset
        :param transient_cache: Shared mutable training cache.
        :type transient_cache: Dict[str, Any]
        :raises IncompatibleInterfaceError: If the active model has no affine
            bottleneck LayerNorm compatible with canonicalization.
        """
        super().on_phase_start(model, dataset, transient_cache)
        self._validate_batch_separable_encoder(model)
        if not self._requires_canonicalized_latent:
            return

        bottleneck = getattr(getattr(model, "encoder", None), "bottleneck_layer", None)
        if not isinstance(bottleneck, nn.Sequential) or not bottleneck:
            raise_incompatible_interface_error(
                "CanonicalizedLatent",
                "Canonicalized regularization requires model.encoder.bottleneck_layer.",
            )
        layer_norm = bottleneck[-1]
        if (
            not isinstance(layer_norm, nn.LayerNorm)
            or not layer_norm.elementwise_affine
            or layer_norm.weight is None
            or layer_norm.bias is None
        ):
            raise_incompatible_interface_error(
                "CanonicalizedLatent",
                "The final encoder bottleneck module must be an affine torch.nn.LayerNorm.",
            )
        # REMARK: Keep a non-owning reference. Registering this already-owned
        # module below the criterion would duplicate its parameters in the
        # composite loss state dictionary.
        object.__setattr__(self, "_layer_norm", layer_norm)

    def canonicalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Undo the bottleneck LayerNorm affine transform without detaching it.

        :param latent: Post-LayerNorm latent representation with shape ``(B, D)``.
        :type latent: torch.Tensor
        :return: Canonicalized representation with shape ``(B, D)``.
        :rtype: torch.Tensor
        :raises IncompatibleInterfaceError: If the lifecycle hook did not run or
            a learned LayerNorm scale becomes zero.
        """
        layer_norm = getattr(self, "_layer_norm", None)
        if layer_norm is None:
            raise_incompatible_interface_error(
                "CanonicalizedLatent",
                "on_phase_start must run before canonicalized regularization is evaluated.",
            )
        gamma = layer_norm.weight  # (D,)
        beta = layer_norm.bias  # (D,)
        minimum_scale = torch.finfo(gamma.dtype).eps**0.5
        if torch.any(gamma.abs() < minimum_scale):
            raise_incompatible_interface_error(
                "CanonicalizedLatent",
                "LayerNorm gamma is too close to zero for stable canonicalization.",
            )
        return (latent - beta) / gamma  # (B, D)

    @staticmethod
    def _validate_batch_separable_encoder(model: nn.Module) -> None:
        """Reject encoders whose training outputs couple samples in a batch.

        :param model: Model containing the encoder used by the regularizer.
        :type model: torch.nn.Module
        :raises IncompatibleInterfaceError: If the encoder uses batch
            normalization, for which per-spectrum Jacobian penalties are not
            well-defined by the batched VJP implementation.
        """
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            raise_incompatible_interface_error(
                "CanonicalizedLatent",
                "Regularization requires a model.encoder module.",
            )
        if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in encoder.modules()):
            raise_incompatible_interface_error(
                "CanonicalizedLatent",
                "Regularization requires a batch-separable encoder; use LayerNorm instead of BatchNorm.",
            )
