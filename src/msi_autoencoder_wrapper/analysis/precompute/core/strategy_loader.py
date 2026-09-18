"""Load explicitly supported precompute strategies by stable identifier."""

from __future__ import annotations

import importlib

from .contracts import PrecomputeStrategy


_PREFIX = "msi_autoencoder_wrapper.analysis.precompute.strategies."


def load_strategy(identifier: str) -> PrecomputeStrategy:
    """Import and construct a registered strategy.

    :param identifier: Dotted identifier below ``strategies``, for example
        ``autoencoder.heads_general_analysis``.
    :type identifier: str
    :return: The constructed complete strategy.
    :rtype: PrecomputeStrategy
    :raises ValueError: If the identifier is malformed or the module lacks
        ``build_strategy``.
    """
    if not identifier or identifier.startswith(".") or ".." in identifier:
        raise ValueError(f"Invalid precompute strategy identifier: {identifier!r}.")
    module = importlib.import_module(_PREFIX + identifier)
    factory = getattr(module, "build_strategy", None)
    if factory is None:
        raise ValueError(f"Strategy module '{identifier}' does not define build_strategy().")
    strategy = factory()
    if not isinstance(strategy, PrecomputeStrategy):
        raise TypeError(f"Strategy '{identifier}' did not return a PrecomputeStrategy.")
    return strategy
