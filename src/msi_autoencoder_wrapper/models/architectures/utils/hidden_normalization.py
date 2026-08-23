"""Hidden-feature normalization shared by dense autoencoder components."""

from __future__ import annotations

import torch.nn as nn

from ....utils.exceptions import raise_validation_error


SUPPORTED_HIDDEN_NORMALIZATIONS = frozenset({"layer", "batch", "none"})


def resolve_hidden_normalization(
    normalization: str | None,
    batch_normalization: bool | None,
    *,
    context_name: str,
) -> str:
    """Return one canonical hidden-normalization strategy.

    ``LayerNorm`` is the default because its output for one spectrum does not
    depend on the other spectra in the batch. The legacy
    ``batch_normalization`` flag remains loadable for persisted configurations;
    new configurations must use ``normalization`` explicitly.

    :param normalization: ``"layer"``, ``"batch"``, or ``"none"``.
    :type normalization: str | None
    :param batch_normalization: Legacy Boolean alias mapping to ``"batch"`` or
        ``"none"``.
    :type batch_normalization: bool | None
    :param context_name: Component name used in validation errors.
    :type context_name: str
    :return: Canonical normalization strategy.
    :rtype: str
    :raises ValidationError: If the strategy is unsupported or both interfaces
        are configured simultaneously.
    """
    if normalization is not None and batch_normalization is not None:
        raise_validation_error(
            context_name,
            "Use normalization or the legacy batch_normalization flag, not both.",
        )
    if batch_normalization is not None:
        if not isinstance(batch_normalization, bool):
            raise_validation_error(
                context_name,
                "batch_normalization must be Boolean when provided.",
            )
        return "batch" if batch_normalization else "none"
    resolved = "layer" if normalization is None else str(normalization)
    if resolved not in SUPPORTED_HIDDEN_NORMALIZATIONS:
        raise_validation_error(
            context_name,
            "normalization must be 'layer', 'batch', or 'none'.",
        )
    return resolved


def build_hidden_normalization(strategy: str, feature_dim: int) -> nn.Module | None:
    """Build normalization for one hidden feature vector.

    :param strategy: Canonical strategy returned by
        :func:`resolve_hidden_normalization`.
    :type strategy: str
    :param feature_dim: Hidden feature width.
    :type feature_dim: int
    :return: LayerNorm, BatchNorm1d, or ``None``.
    :rtype: torch.nn.Module | None
    """
    if strategy == "layer":
        return nn.LayerNorm(feature_dim)
    if strategy == "batch":
        return nn.BatchNorm1d(feature_dim)
    return None
