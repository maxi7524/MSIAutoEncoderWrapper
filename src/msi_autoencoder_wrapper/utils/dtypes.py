"""Floating-point dtype resolution shared by runtime components."""

from __future__ import annotations

from typing import Any

import torch

from .exceptions import raise_validation_error


_FLOATING_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def resolve_floating_dtype(value: torch.dtype | str | None) -> torch.dtype:
    """Resolve a supported floating-point Torch dtype.

    :param value: Torch dtype or canonical dtype name. ``None`` resolves to
        ``torch.float32``.
    :type value: torch.dtype | str | None
    :return: Resolved floating-point dtype.
    :rtype: torch.dtype
    :raises ValidationError: If the value is not a supported floating dtype.
    """
    if value is None:
        return torch.float32
    if isinstance(value, str):
        resolved = _FLOATING_DTYPES.get(value.removeprefix("torch.").lower())
    else:
        resolved = value
    if resolved not in _FLOATING_DTYPES.values():
        raise_validation_error(
            "DType",
            "dtype must be one of float16, float32, float64, or bfloat16.",
        )
    return resolved
