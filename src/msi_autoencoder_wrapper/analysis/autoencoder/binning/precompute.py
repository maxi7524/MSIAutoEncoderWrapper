"""Shared, sample-once precompute for binning/inverse-binning analysis.

One :class:`BinningPrecompute` instance is built per notebook and reused by every
analysis-topic module (forward sweep, inverse-binner sweep, region-dependency,
method comparison, ...). It owns exactly one thing that must not be repeated per
analysis: the seeded spectrum sample and the raw (unbinned) spectra read from it.
Forward-binned and inverse-binned representations are derived on demand and cached,
keyed by their own parameters, so the same raw sample is reused everywhere but no
binning run is repeated for the same (bin_step, x_min, x_max) or inverse-binner config.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np
from tqdm.auto import tqdm

from ....binners.base_binner import MSIBaseBinner
from ....binners.base_inverse import MSIBaseInverseBinner
from ....binners.binners_strategies.linear_binner import LinearBinning
from ....readers.base_reader import MSIBaseReader
from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

InverseBinnerFactory = Callable[[MSIBaseBinner], MSIBaseInverseBinner]


@dataclass(frozen=True)
class InverseResult:
    """One spectrum's inverse-binned representation plus its optional diagnostics."""

    mz: np.ndarray
    intensity: np.ndarray
    diagnostics: Mapping[str, Any]


class BinningPrecompute:
    """Cache raw and derived spectra for a fixed, seeded spectrum sample.

    :param reader: Data reader already pointed at the target image.
    :param sample_size: Requested spectrum count. Capped to the dataset size — if the
        dataset has fewer spectra than requested, every spectrum is used and
        :attr:`capped` is ``True``.
    :param seed: Seed for the sample draw, reproducible across notebook re-runs.
    :param spectrum_ids: Explicit spectrum ids to use instead of a random draw
        (bypasses ``sample_size``/``seed``) — for full-image passes.
    """

    def __init__(
        self,
        reader: MSIBaseReader,
        sample_size: int = 10_000,
        seed: int = 0,
        spectrum_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.reader = reader
        self.seed = int(seed)
        self.sample_size = int(sample_size)
        total = int(reader.GetNumberOfSpectra())
        if total <= 0:
            raise_validation_error("BinningPrecompute", "Reader reports zero spectra.")

        # Resolve the sample: explicit ids bypass the seeded draw entirely (used for
        # full-image passes); otherwise draw without replacement, capping at `total`.
        if spectrum_ids is not None:
            self.spectrum_ids = np.unique(np.asarray(spectrum_ids, dtype=np.int64))
            self.capped = False
        else:
            size = min(self.sample_size, total)
            self.capped = size < self.sample_size
            rng = np.random.default_rng(self.seed)
            self.spectrum_ids = np.sort(rng.choice(total, size=size, replace=False))
        logger.info(
            "BinningPrecompute sample resolved: %s of %s spectra (seed=%s, capped=%s).",
            self.spectrum_ids.size, total, self.seed, self.capped,
        )

        self._raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._forward_binners: dict[tuple[float, float, float], MSIBaseBinner] = {}
        self._forward_cache: dict[tuple[float, float, float], dict[int, np.ndarray]] = {}
        self._inverse_cache: dict[tuple, dict[int, InverseResult]] = {}
        self._global_range: Optional[tuple[float, float]] = None

    # -- raw spectra ------------------------------------------------------------------

    def load_raw(self) -> "BinningPrecompute":
        """Eagerly read every sampled spectrum's native (m/z, intensity) once.

        This is the read-from-disk step, one ``reader.GetSpectrum`` call per sampled
        spectrum. Called automatically (lazily) by :meth:`raw`/:attr:`global_mass_range`
        the first time either is needed, so calling it explicitly is only useful to
        control when the read happens (e.g. to show its own progress bar up front,
        separate from a later ``forward``/``inverse`` call).
        """
        logger.info("START reading %s raw spectra from the reader.", self.spectrum_ids.size)
        started = time.perf_counter()
        for spectrum_id in tqdm(self.spectrum_ids, desc="Reading raw spectra", unit="spectrum"):
            mz, intensity = self.reader.GetSpectrum(int(spectrum_id))
            self._raw[int(spectrum_id)] = (np.asarray(mz, dtype=np.float32), np.asarray(intensity, dtype=np.float32))
        logger.info("DONE reading %s raw spectra in %.2fs.", len(self._raw), time.perf_counter() - started)
        return self

    def raw(self, spectrum_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the cached native (m/z, intensity) for one sampled spectrum."""
        if not self._raw:
            self.load_raw()
        return self._raw[int(spectrum_id)]

    @property
    def global_mass_range(self) -> tuple[float, float]:
        """True dataset-wide (min, max) m/z, scanned from the sample itself.

        Deliberately does not use ``reader.GetXMin()/GetXMax()`` — for
        ``PyImzMLReader`` those only inspect spectrum index 0, which is wrong for
        centroid/variable-length data where pixels can have different local extents
        (verified against ``C57BL6_Kidney_02-pos``: spectrum 0 there is an all-zero
        off-tissue background pixel).
        """
        if self._global_range is None:
            if not self._raw:
                self.load_raw()
            lows = [float(np.min(mz)) for mz, _ in self._raw.values() if mz.size]
            highs = [float(np.max(mz)) for mz, _ in self._raw.values() if mz.size]
            if not lows:
                raise_validation_error("BinningPrecompute", "No sampled spectrum has any finite m/z value.")
            self._global_range = (min(lows), max(highs))
            logger.info("Scanned true global m/z range from the sample: %s.", self._global_range)
        return self._global_range

    # -- forward binning ----------------------------------------------------------------

    def forward_binner(self, bin_step: float, x_min: Optional[float] = None, x_max: Optional[float] = None) -> MSIBaseBinner:
        """Return (and cache) a :class:`LinearBinning` for one (step, range) config.

        ``x_min``/``x_max`` default to :attr:`global_mass_range` (the whole sampled
        image). Passing an explicit narrower ``(x_min, x_max)`` is how a sub-range
        analysis gets a genuinely separate, recalibrated forward binner for that window,
        rather than filtering the full-range binner's output after the fact.
        """
        lo, hi = self.global_mass_range
        resolved_min = lo if x_min is None else float(x_min)
        resolved_max = hi if x_max is None else float(x_max)
        key = (round(float(bin_step), 10), round(resolved_min, 6), round(resolved_max, 6))
        if key not in self._forward_binners:
            logger.debug("Building LinearBinning(bin_step=%s, x_min=%s, x_max=%s).", bin_step, resolved_min, resolved_max)
            self._forward_binners[key] = LinearBinning(bin_step=bin_step, x_min=resolved_min, x_max=resolved_max)
        return self._forward_binners[key]

    def forward(self, bin_step: float, x_min: Optional[float] = None, x_max: Optional[float] = None) -> tuple[MSIBaseBinner, dict[int, np.ndarray]]:
        """Return the forward binner and every sampled spectrum's ``B(X)``, cached.

        Cache key is ``(bin_step, x_min, x_max)`` (rounded) — calling this again with
        the same arguments is a cache hit and does not re-run any binning.
        """
        binner = self.forward_binner(bin_step, x_min, x_max)
        key = (round(float(bin_step), 10), round(binner.x_min, 6), round(binner.x_max, 6))
        if key in self._forward_cache:
            logger.debug("Forward cache hit for bin_step=%s, range=(%s, %s).", bin_step, binner.x_min, binner.x_max)
            return binner, self._forward_cache[key]

        # Cache miss: run the forward binner once per sampled spectrum.
        logger.info(
            "START forward-binning %s spectra at bin_step=%s over range (%s, %s) — %s output bins.",
            self.spectrum_ids.size, bin_step, binner.x_min, binner.x_max, binner.GetXAxisDepth(),
        )
        started = time.perf_counter()
        cache: dict[int, np.ndarray] = {}
        for spectrum_id in tqdm(self.spectrum_ids, desc=f"Forward binning Δm={bin_step:g}", unit="spectrum"):
            mz, intensity = self.raw(spectrum_id)
            cache[int(spectrum_id)] = binner.transform_spectrum(mz, intensity).cpu().numpy().astype(np.float32, copy=False)
        self._forward_cache[key] = cache
        logger.info("DONE forward-binning at bin_step=%s in %.2fs.", bin_step, time.perf_counter() - started)
        return binner, cache

    # -- inverse binning ----------------------------------------------------------------

    def inverse(
        self,
        bin_step: float,
        inverse_binner_factory: InverseBinnerFactory,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        cache_key: Optional[str] = None,
    ) -> tuple[MSIBaseBinner, MSIBaseInverseBinner, dict[int, InverseResult]]:
        """Return the forward binner, one inverse-binner instance, and every sampled
        spectrum's ``INB(X)`` (with diagnostics if the instance opted in), cached.

        Builds the forward representation first (via :meth:`forward`, itself cached),
        then applies ``inverse_binner_factory(binner)`` to every sampled spectrum's
        ``B(X)``. Cache key is ``(bin_step, x_min, x_max, inverse binner class,
        cache_key)`` — two calls with the same key never re-run the inverse binner, so
        pick a ``cache_key`` that actually distinguishes every parameter point you sweep
        over (the default, ``repr(get_config())``, does this automatically as long as
        every constructor parameter that matters is recorded in ``_config``).

        :param inverse_binner_factory: Callable taking the resolved forward binner and
            returning a configured inverse-binner instance (e.g.
            ``lambda binner: BinnerManager.get_inverse_binner("QuantileInverseBinner", binner=binner, quantile=0.9)``).
        :param cache_key: Explicit cache key for this parameter point. Defaults to
            ``repr(inverse_binner.get_config())`` — sufficient whenever the config is
            a faithful, distinguishing summary of the strategy's behavior.
        """
        binner, forward_cache = self.forward(bin_step, x_min, x_max)
        inverse_binner = inverse_binner_factory(binner)
        resolved_key = cache_key if cache_key is not None else repr(inverse_binner.get_config())
        key = (round(float(bin_step), 10), round(binner.x_min, 6), round(binner.x_max, 6), type(inverse_binner).__name__, resolved_key)
        if key in self._inverse_cache:
            logger.debug("Inverse cache hit for %s / %s.", type(inverse_binner).__name__, resolved_key)
            return binner, inverse_binner, self._inverse_cache[key]

        # Cache miss: run the inverse binner once per sampled spectrum's B(X).
        logger.info(
            "START inverse-binning %s spectra with %s (%s).",
            self.spectrum_ids.size, type(inverse_binner).__name__, resolved_key,
        )
        logger.debug("Resolved inverse-binner config: %s", inverse_binner.get_config())
        started = time.perf_counter()
        cache: dict[int, InverseResult] = {}
        for spectrum_id in tqdm(self.spectrum_ids, desc=f"Inverse binning {resolved_key}", unit="spectrum"):
            grid_y = forward_cache[int(spectrum_id)]
            inverse_result = inverse_binner.transform_spectra(grid_y)
            end = int(inverse_result.offsets[1])
            mz = inverse_result.mass_values[:end].cpu().numpy()
            intensity = inverse_result.intensities[:end].cpu().numpy()
            diagnostics = dict(getattr(inverse_binner, "last_diagnostics", {}) or {})
            cache[int(spectrum_id)] = InverseResult(np.asarray(mz, dtype=np.float32), np.asarray(intensity, dtype=np.float32), diagnostics)
        self._inverse_cache[key] = cache
        logger.info("DONE inverse-binning %s (%s) in %.2fs.", type(inverse_binner).__name__, resolved_key, time.perf_counter() - started)
        return binner, inverse_binner, cache

    def clear_derived_cache(self) -> None:
        """Drop cached forward/inverse results while keeping the raw sample and range."""
        logger.info("Clearing derived (forward/inverse) cache; raw sample and global range are kept.")
        self._forward_binners.clear()
        self._forward_cache.clear()
        self._inverse_cache.clear()
