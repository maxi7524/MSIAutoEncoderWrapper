"""Public package interface with lazy loading of model and training modules."""

from __future__ import annotations

from typing import Any

__all__ = ["MSIAutoEncoderWrapper"]


def __getattr__(name: str) -> Any:
    """Load the wrapper facade only when it is explicitly requested."""
    if name == "MSIAutoEncoderWrapper":
        from .core.wrapper import MSIAutoEncoderWrapper

        return MSIAutoEncoderWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
