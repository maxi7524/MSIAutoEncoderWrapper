"""Compact serialization for merged annotation pixel indices."""

from __future__ import annotations

import struct
import zlib
from typing import Iterable, List

from ..utils.exceptions import raise_validation_error


def encode_pixel_indices(indices: Iterable[int]) -> bytes:
    """Encode sorted unique pixel indices as compressed unsigned deltas.

    :param indices: Zero-based merged spectrum indices.
    :type indices: Iterable[int]
    :return: Compressed little-endian ``uint32`` delta sequence.
    :rtype: bytes
    """
    ordered = sorted({int(value) for value in indices})
    if any(value < 0 or value > 0xFFFFFFFF for value in ordered):
        raise_validation_error(
            "AnnotationPixelBlob",
            "Pixel indices must fit in an unsigned 32-bit integer.",
        )
    previous = 0
    deltas = []
    for position, value in enumerate(ordered):
        deltas.append(value if position == 0 else value - previous)
        previous = value
    payload = struct.pack(f"<{len(deltas)}I", *deltas) if deltas else b""
    return zlib.compress(payload)


def decode_pixel_indices(payload: bytes) -> List[int]:
    """Decode a pixel-index blob produced by :func:`encode_pixel_indices`.

    :param payload: Compressed little-endian ``uint32`` delta sequence.
    :type payload: bytes
    :return: Sorted zero-based merged spectrum indices.
    :rtype: List[int]
    :raises ValidationError: If the payload is malformed.
    """
    try:
        raw = zlib.decompress(payload)
    except zlib.error as error:
        raise_validation_error("AnnotationPixelBlob", f"Invalid compressed data: {error}")
    if len(raw) % 4:
        raise_validation_error(
            "AnnotationPixelBlob",
            "Decoded pixel-index data is not aligned to uint32 values.",
        )
    deltas = struct.unpack(f"<{len(raw) // 4}I", raw) if raw else ()
    indices: List[int] = []
    current = 0
    for position, delta in enumerate(deltas):
        current = int(delta) if position == 0 else current + int(delta)
        indices.append(current)
    return indices
