"""Three-way ion-image comparison: X vs B(X) vs INB(X) for one m/z window, so it is
visible *where in the image* signal spills at each transformation stage — not just
original vs round-trip. See ``methodology.md`` §6 step 8.

Reuses ``visualization.interactive.IonImageViewer`` for browsing across m/z (its
``.widget()`` gives a live ipywidgets slider in a running notebook; ``.plot(index)``
renders one m/z statically, used here for smoke-testing under ``nbconvert --execute``
where a widget cannot render).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ....readers.base_reader import MSIBaseReader
from ....readers.spatial import SpatialImage, aggregate_window
from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme
from ....visualization.interactive import ContinuousIonImageViewer, IonImageViewer
from ....visualization.spatial import plot_image_grid
from .inverse_binner_analysis import inverse_binner_factory
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)


def _aggregate_window_or_zero(values: np.ndarray, aggregation: str) -> float:
    """``aggregate_window`` on an empty selection is undefined; treat it as zero signal."""
    return aggregate_window(values, aggregation) if values.size else 0.0


def ion_image_from_raw(reader: MSIBaseReader, precompute: BinningPrecompute, mz: float, tolerance: float, aggregation: str = "sum") -> SpatialImage:
    """Ion image of ``X`` (raw spectra) for the sampled spectra; NaN elsewhere.

    Uses ``precompute.raw`` uncropped — unlike the matching-based comparisons
    elsewhere in this package, an ion image is a per-spectrum window aggregate, not a
    point-matching comparison, so it always reflects the true raw spectrum near ``mz``
    regardless of any ``x_min``/``x_max`` restriction used for a region's binner.
    """
    total = reader.GetNumberOfSpectra()
    values = np.full(total, np.nan, dtype=np.float64)
    for spectrum_id in precompute.spectrum_ids:
        raw_mz, raw_y = precompute.raw(spectrum_id)
        selected = np.abs(raw_mz - mz) <= tolerance
        values[int(spectrum_id)] = _aggregate_window_or_zero(raw_y[selected], aggregation)
    return reader.MapSpectrumValuesToImage(values, fill_value=np.nan)


def ion_image_from_forward(
    reader: MSIBaseReader, precompute: BinningPrecompute, delta_m: float, mz: float, tolerance: float,
    aggregation: str = "sum", x_min: Optional[float] = None, x_max: Optional[float] = None,
) -> SpatialImage:
    """Ion image of ``B(X)`` (forward-binned) for the sampled spectra; NaN elsewhere."""
    binner, forward_cache = precompute.forward(delta_m, x_min, x_max)
    grid_mz = np.asarray(binner.GetXAxis(), dtype=np.float64)
    selected = np.abs(grid_mz - mz) <= tolerance
    total = reader.GetNumberOfSpectra()
    values = np.full(total, np.nan, dtype=np.float64)
    for spectrum_id in precompute.spectrum_ids:
        values[int(spectrum_id)] = _aggregate_window_or_zero(forward_cache[int(spectrum_id)][selected], aggregation)
    return reader.MapSpectrumValuesToImage(values, fill_value=np.nan)


def ion_image_from_inverse(
    reader: MSIBaseReader, precompute: BinningPrecompute, delta_m: float, method_grid_point: dict[str, Any], mz: float, tolerance: float,
    aggregation: str = "sum", x_min: Optional[float] = None, x_max: Optional[float] = None,
) -> SpatialImage:
    """Ion image of ``INB(X)`` (inverse-binned) for the sampled spectra; NaN elsewhere."""
    label, method, params = method_grid_point["label"], method_grid_point["method"], dict(method_grid_point.get("params", {}))
    _, _, inverse_cache = precompute.inverse(delta_m, inverse_binner_factory(method, **params), x_min, x_max, cache_key=label)
    total = reader.GetNumberOfSpectra()
    values = np.full(total, np.nan, dtype=np.float64)
    for spectrum_id in precompute.spectrum_ids:
        result = inverse_cache[int(spectrum_id)]
        selected = np.abs(result.mz - mz) <= tolerance
        values[int(spectrum_id)] = _aggregate_window_or_zero(result.intensity[selected], aggregation)
    return reader.MapSpectrumValuesToImage(values, fill_value=np.nan)


def three_way_ion_images(
    reader: MSIBaseReader, precompute: BinningPrecompute, delta_m: float, method_grid_point: dict[str, Any], mz: float,
    tolerance: float = 0.01, aggregation: str = "sum", x_min: Optional[float] = None, x_max: Optional[float] = None,
) -> dict[str, SpatialImage]:
    """``{"X": ..., "B(X)": ..., "INB(X)": ...}`` ion images for one m/z window, all
    restricted to ``precompute``'s sampled spectra (NaN elsewhere in every image)."""
    return {
        "X": ion_image_from_raw(reader, precompute, mz, tolerance, aggregation),
        "B(X)": ion_image_from_forward(reader, precompute, delta_m, mz, tolerance, aggregation, x_min, x_max),
        "INB(X)": ion_image_from_inverse(reader, precompute, delta_m, method_grid_point, mz, tolerance, aggregation, x_min, x_max),
    }


def compare_ion_images(images: dict[str, SpatialImage]) -> dict[str, dict[str, float]]:
    """Pairwise relative L1 and spatial (Pearson) correlation between every pair of
    named images, computed only over positions finite in *both* images of the pair.

    :return: ``{"name_a vs name_b": {"relative_l1", "spatial_correlation"}}`` for every
        unordered pair. ``spatial_correlation`` is NaN when either image is constant
        over the shared valid region (correlation undefined).
    """
    epsilon = np.finfo(float).eps
    names = list(images)
    results: dict[str, dict[str, float]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first, second = images[names[i]].values.ravel(), images[names[j]].values.ravel()
            mask = np.isfinite(first) & np.isfinite(second)
            a, b = first[mask], second[mask]
            relative_l1 = float(np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + epsilon))
            correlation = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
            results[f"{names[i]} vs {names[j]}"] = {"relative_l1": relative_l1, "spatial_correlation": correlation}
    return results


def plot_three_way_ion_images(images: dict[str, SpatialImage], theme: VisualizationTheme | str | None = None):
    """Render :func:`three_way_ion_images` output as one row via ``plot_image_grid``."""
    return plot_image_grid({name: image.values for name, image in images.items()}, theme=theme)


def ion_image_browser(
    reader: MSIBaseReader, precompute: BinningPrecompute, delta_m: float, method_grid_point: dict[str, Any], mz_options: Sequence[float],
    tolerance: float = 0.01, aggregation: str = "sum", x_min: Optional[float] = None, x_max: Optional[float] = None,
    theme: VisualizationTheme | str | None = None,
) -> IonImageViewer:
    """Return an :class:`IonImageViewer` for browsing :func:`three_way_ion_images` across
    ``mz_options`` — call ``.widget()`` in a running notebook for a live ipywidgets
    slider, or ``.plot(index)`` for one static render (e.g. for automated smoke tests).
    """
    def provider(mz: float) -> dict[str, np.ndarray]:
        images = three_way_ion_images(reader, precompute, delta_m, method_grid_point, mz, tolerance, aggregation, x_min, x_max)
        return {name: image.values for name, image in images.items()}

    return IonImageViewer(mz_options, provider, theme=theme)


def ion_image_browser_range(
    reader: MSIBaseReader,
    precompute: BinningPrecompute,
    delta_m: float,
    method_grid_point: dict[str, Any],
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
    mz_step: Optional[float] = None,
    tolerance: float = 0.01,
    aggregation: str = "sum",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    theme: VisualizationTheme | str | None = None,
) -> ContinuousIonImageViewer:
    """Return a browser with a floating-point slider over a continuous m/z range.

    This is a separate interface from :func:`ion_image_browser`; the indexed browser
    and its methods remain unchanged. Missing slider boundaries are resolved from the
    sampled raw spectra, and ``mz_step`` defaults to ``delta_m``. The first render
    computes only the requested forward configuration and inverse method; subsequent
    slider updates reuse :class:`BinningPrecompute` caches.

    :param mz_min: Lower slider boundary. Defaults to the sampled global minimum.
    :type mz_min: Optional[float]
    :param mz_max: Upper slider boundary. Defaults to the sampled global maximum.
    :type mz_max: Optional[float]
    :param mz_step: Slider increment. Defaults to ``delta_m``.
    :type mz_step: Optional[float]
    :return: Viewer whose ``plot(mz)`` and ``widget(initial_mz)`` accept physical m/z.
    :rtype: ContinuousIonImageViewer
    """
    global_min, global_max = precompute.global_mass_range
    resolved_min = global_min if mz_min is None else float(mz_min)
    resolved_max = global_max if mz_max is None else float(mz_max)
    resolved_step = float(delta_m) if mz_step is None else float(mz_step)

    def provider(mz: float) -> dict[str, np.ndarray]:
        images = three_way_ion_images(
            reader,
            precompute,
            delta_m,
            method_grid_point,
            mz,
            tolerance,
            aggregation,
            x_min,
            x_max,
        )
        return {name: image.values for name, image in images.items()}

    return ContinuousIonImageViewer(
        resolved_min,
        resolved_max,
        resolved_step,
        provider,
        theme=theme,
    )
