"""Per-pixel spatial maps: where in the *image* (not just where in m/z) does error
concentrate — background, tissue edge, high-TIC regions, specific structures?

``reader.MapSpectrumValuesToImage`` requires one value per **every** spectrum in the
dataset, not just the precompute's sample — spectra outside the sample are filled with
NaN (rendered as gaps, not zeros). For genuinely full-image coverage, build the
``BinningPrecompute`` passed here with ``spectrum_ids=np.arange(reader.GetNumberOfSpectra())``
instead of a random sample (this module works with either; it only warns about
coverage). See ``methodology.md`` §6 step 7.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ....readers.base_reader import MSIBaseReader
from ....readers.spatial import SpatialImage
from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme
from ....visualization.spatial import plot_image_grid

logger = get_custom_logger(__name__)


def metric_spatial_values(
    reader: MSIBaseReader,
    records: Sequence[Mapping[str, Any]],
    label: str,
    metric: str,
    comparison: str,
    normalization: str = "raw",
) -> np.ndarray:
    """Full-length (one entry per every spectrum in the dataset) array of one
    (label, metric, comparison, normalization)'s per-spectrum value, ordered by
    ``spectrum_id``. NaN where a spectrum has no matching record (typically: it was not
    part of the precompute's sample).
    """
    total = reader.GetNumberOfSpectra()
    values = np.full(total, np.nan, dtype=np.float64)
    covered = 0
    for record in records:
        if record["label"] == label and record["metric"] == metric and record["comparison"] == comparison and record["normalization"] == normalization:
            values[record["spectrum_id"]] = record["value"]
            covered += 1
    if covered < total:
        logger.info("Spatial map for %s/%s covers %s of %s spectra (rest shown as gaps).", label, metric, covered, total)
    return values


def metric_spatial_image(
    reader: MSIBaseReader,
    records: Sequence[Mapping[str, Any]],
    label: str,
    metric: str,
    comparison: str,
    normalization: str = "raw",
) -> SpatialImage:
    """:func:`metric_spatial_values` mapped onto the reader's native ``(z, y, x)`` grid."""
    values = metric_spatial_values(reader, records, label, metric, comparison, normalization)
    return reader.MapSpectrumValuesToImage(values, fill_value=np.nan)


def plot_metric_spatial_maps(
    reader: MSIBaseReader,
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    metric: str,
    comparison: str,
    normalization: str = "raw",
    theme: VisualizationTheme | str | None = None,
):
    """One spatial map per ``label`` (e.g. per method), same shared color scale
    (``theme.shared_image_scale``, the default), arranged in a grid via
    ``visualization.spatial.plot_image_grid`` — so several methods' error maps for the
    same metric are directly visually comparable.
    """
    images = {label: metric_spatial_image(reader, records, label, metric, comparison, normalization).values for label in labels}
    figure, axes = plot_image_grid(images, theme=theme)
    figure.suptitle(f"{metric} | {comparison}/{normalization}")
    return figure, axes
