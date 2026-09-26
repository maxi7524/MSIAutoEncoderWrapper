"""Assemble per-pixel values into two-dimensional images."""

from __future__ import annotations

import numpy as np

from ....utils.exceptions import raise_validation_error


def image_extent(x: np.ndarray, y: np.ndarray) -> tuple[int, int, int, int]:
    """Return the bounding box ``(x_min, y_min, width, height)`` of pixel coordinates.

    :param x: Column coordinates, shape ``(N,)``.
    :type x: numpy.ndarray
    :param y: Row coordinates, shape ``(N,)``.
    :type y: numpy.ndarray
    :return: Minimum coordinates and image size.
    :rtype: tuple[int, int, int, int]
    """
    x_min, y_min = int(np.min(x)), int(np.min(y))
    return x_min, y_min, int(np.max(x)) - x_min + 1, int(np.max(y)) - y_min + 1


def assemble_image(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                   extent: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Place pixel values on a regular grid; absent pixels are ``nan``.

    :param x: Column coordinates, shape ``(N,)``.
    :type x: numpy.ndarray
    :param y: Row coordinates, shape ``(N,)``.
    :type y: numpy.ndarray
    :param values: Pixel values, shape ``(N,)``.
    :type values: numpy.ndarray
    :param extent: Optional bounding box from :func:`image_extent`, used to align
        several images of the same acquisition.
    :type extent: tuple[int, int, int, int] | None
    :return: Image with rows indexed by ``y`` and columns by ``x``, shape ``(H, W)``.
    :rtype: numpy.ndarray
    :raises ValidationError: If inputs are misaligned or a coordinate repeats.
    """
    x, y, values = np.asarray(x), np.asarray(y), np.asarray(values, dtype=np.float64)
    if not (x.shape == y.shape == values.shape) or x.ndim != 1:
        raise_validation_error("SpatialImage", "x, y and values must be aligned one-dimensional arrays.")
    x_min, y_min, width, height = extent or image_extent(x, y)
    columns, rows = x - x_min, y - y_min
    if np.any(columns < 0) or np.any(rows < 0) or np.any(columns >= width) or np.any(rows >= height):
        raise_validation_error("SpatialImage", "Coordinates fall outside the requested extent.")
    flat = rows * width + columns
    if np.unique(flat).size != flat.size:
        raise_validation_error("SpatialImage", "Duplicate pixel coordinates.")
    image = np.full(height * width, np.nan)
    image[flat] = values
    return image.reshape(height, width)  # (H, W)
