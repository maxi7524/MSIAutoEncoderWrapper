"""Public package interface with lazy loading of model and training modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.wrapper import MSIAutoEncoderWrapper

__all__ = ["MSIAutoEncoderWrapper"]


def __getattr__(name: str) -> Any:
    if name == "MSIAutoEncoderWrapper":
        from .core.wrapper import MSIAutoEncoderWrapper

        return MSIAutoEncoderWrapper

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
