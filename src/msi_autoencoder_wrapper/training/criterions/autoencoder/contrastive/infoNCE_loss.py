"""Annotation-aware peak-permutation InfoNCE for MSI spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, peak_widths

from ...autoencoder_base_criterions import MSIContrastiveCriterion
from ...criterions_manager import CriterionsManager
from .....data import SpectrumBatch
from .....metrics import info_nce
from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.exceptions import (
    raise_incompatible_interface_error,
    raise_validation_error,
)
from .....utils.logger import get_custom_logger


logger = get_custom_logger(__name__)

SUPPORTED_PEAK_SELECTION_METHODS = frozenset(
    {"permutation_random", "permutation_label_invariant"}
)
SUPPORTED_NEGATIVE_WEIGHTING_METHODS = frozenset({"multilabel_jaccard"})


@dataclass(frozen=True)
class ProtectedIntervalIndex:
    """Store merged protected bin intervals in CSR form."""

    spectrum_ids: np.ndarray
    offsets: np.ndarray
    left: np.ndarray
    right: np.ndarray

    def intervals(self, spectrum_id: int) -> tuple[tuple[int, int], ...]:
        """Return half-open protected intervals for one spectrum."""
        position = int(np.searchsorted(self.spectrum_ids, int(spectrum_id)))
        if position >= self.spectrum_ids.size or int(self.spectrum_ids[position]) != int(
            spectrum_id
        ):
            return ()
        start = int(self.offsets[position])
        stop = int(self.offsets[position + 1])
        return tuple(
            (int(left), int(right))
            for left, right in zip(self.left[start:stop], self.right[start:stop])
        )


@CriterionsManager.register_criterion("autoencoder", "contrastive", "InfoNCELoss")
class MSIInfoNCELoss(MSIContrastiveCriterion):
    """Make representations invariant to controlled peak permutations.

    ``permutation_random`` selects non-overlapping envelopes from a train-derived
    global catalogue without consulting annotations. It is the control strategy
    and may alter annotated evidence.

    ``permutation_label_invariant`` resolves protected intervals separately for
    every spectrum through ``spectrum_id -> source dataset -> pixel labels ->
    source m/z -> bins``. A candidate is eligible only when it does not intersect
    that spectrum's protected intervals.

    Selected envelope shapes and masses are cyclically reassigned within one
    spectrum. If ``preserve_input_normalization`` is true, interpolation keeps
    every donor envelope's original mass, so the selected-region sum and the
    per-spectrum TIC remain invariant. Protected bins are unchanged. Original
    and augmented spectra form positive pairs.

    :param peak_selection_method: ``permutation_random`` or
        ``permutation_label_invariant``.
    :type peak_selection_method: str
    :param temperature: Positive InfoNCE temperature.
    :type temperature: float
    :param peak_sample_size: Maximum train spectra scanned once.
    :type peak_sample_size: int
    :param peak_sample_seed: Deterministic catalogue-sampling seed.
    :type peak_sample_seed: int
    :param peak_catalog_limit: Maximum unique envelopes retained.
    :type peak_catalog_limit: int
    :param permuted_peaks_per_view: Maximum envelopes permuted per view.
    :type permuted_peaks_per_view: int
    :param annotation_tolerance: Non-negative protected m/z tolerance.
    :type annotation_tolerance: float
    :param annotation_tolerance_unit: ``ppm`` or ``Da``.
    :type annotation_tolerance_unit: str
    :param preserve_input_normalization: Preserve mutable-region mass and TIC.
    :type preserve_input_normalization: bool
    :param negative_weighting_method: Optional ``multilabel_jaccard`` weighting.
    :type negative_weighting_method: str | None
    :param overlapping_label_negative_weight: Minimum negative weight for
        identical non-empty label sets.
    :type overlapping_label_negative_weight: float
    :param max_peaks_per_spectrum: Per-spectrum catalogue cap.
    :type max_peaks_per_spectrum: int
    :param noise_level: Deprecated compatibility parameter; ignored.
    :type noise_level: float | None
    :param max_noise_peaks: Deprecated alias for ``permuted_peaks_per_view``.
    :type max_noise_peaks: int | None
    """

    def __init__(
        self,
        peak_selection_method: str = "permutation_random",
        temperature: float = 0.07,
        peak_sample_size: int = 1000,
        peak_sample_seed: int = 0,
        peak_catalog_limit: int = 4096,
        permuted_peaks_per_view: int = 8,
        annotation_tolerance: float = 3.0,
        annotation_tolerance_unit: str = "ppm",
        preserve_input_normalization: bool = True,
        negative_weighting_method: str | None = None,
        overlapping_label_negative_weight: float = 0.25,
        max_peaks_per_spectrum: int = 32,
        noise_level: float | None = None,
        max_noise_peaks: int | None = None,
    ) -> None:
        super().__init__()
        del noise_level
        if peak_selection_method not in SUPPORTED_PEAK_SELECTION_METHODS:
            raise_validation_error(
                "InfoNCELoss",
                f"peak_selection_method must be one of {sorted(SUPPORTED_PEAK_SELECTION_METHODS)}.",
            )
        if temperature <= 0 or annotation_tolerance < 0:
            raise_validation_error(
                "InfoNCELoss", "temperature must be positive and tolerance non-negative."
            )
        if annotation_tolerance_unit not in {"ppm", "Da"}:
            raise_validation_error(
                "InfoNCELoss", "annotation_tolerance_unit must be 'ppm' or 'Da'."
            )
        integer_parameters = {
            "peak_sample_size": peak_sample_size,
            "peak_catalog_limit": peak_catalog_limit,
            "permuted_peaks_per_view": (
                max_noise_peaks if max_noise_peaks is not None else permuted_peaks_per_view
            ),
            "max_peaks_per_spectrum": max_peaks_per_spectrum,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integer_parameters.values()
        ):
            raise_validation_error(
                "InfoNCELoss", "Peak sampling and permutation limits must be positive integers."
            )
        if negative_weighting_method is not None and (
            negative_weighting_method not in SUPPORTED_NEGATIVE_WEIGHTING_METHODS
        ):
            raise_validation_error(
                "InfoNCELoss",
                f"negative_weighting_method must be one of {sorted(SUPPORTED_NEGATIVE_WEIGHTING_METHODS)}.",
            )
        if not 0 < overlapping_label_negative_weight <= 1:
            raise_validation_error(
                "InfoNCELoss", "overlapping_label_negative_weight must be in (0, 1]."
            )
        self.peak_selection_method = peak_selection_method
        self.temperature = float(temperature)
        self.peak_sample_size = peak_sample_size
        self.peak_sample_seed = int(peak_sample_seed)
        self.peak_catalog_limit = peak_catalog_limit
        self.permuted_peaks_per_view = integer_parameters["permuted_peaks_per_view"]
        self.annotation_tolerance = float(annotation_tolerance)
        self.annotation_tolerance_unit = annotation_tolerance_unit
        self.preserve_input_normalization = bool(preserve_input_normalization)
        self.negative_weighting_method = negative_weighting_method
        self.overlapping_label_negative_weight = float(overlapping_label_negative_weight)
        self.max_peaks_per_spectrum = max_peaks_per_spectrum
        self._cache_key = "peak_permutation::" + repr(
            (
                peak_selection_method,
                peak_sample_size,
                peak_sample_seed,
                peak_catalog_limit,
                max_peaks_per_spectrum,
                annotation_tolerance,
                annotation_tolerance_unit,
            )
        )
        self._config = {
            "peak_selection_method": peak_selection_method,
            "temperature": self.temperature,
            "peak_sample_size": peak_sample_size,
            "peak_sample_seed": peak_sample_seed,
            "peak_catalog_limit": peak_catalog_limit,
            "permuted_peaks_per_view": self.permuted_peaks_per_view,
            "annotation_tolerance": self.annotation_tolerance,
            "annotation_tolerance_unit": annotation_tolerance_unit,
            "preserve_input_normalization": self.preserve_input_normalization,
            "negative_weighting_method": negative_weighting_method,
            "overlapping_label_negative_weight": self.overlapping_label_negative_weight,
            "max_peaks_per_spectrum": max_peaks_per_spectrum,
        }

    def on_phase_start(
        self,
        model: torch.nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Precompute the train peak catalogue and per-spectrum protected bins."""
        del model
        if self._cache_key in transient_cache:
            transient_cache["chemical_peak_bank"] = transient_cache[self._cache_key][
                "catalogue"
            ]
            return
        train = (
            dataset.create_partitions().train
            if callable(getattr(dataset, "create_partitions", None))
            else dataset
        )
        sample_size = min(len(train), self.peak_sample_size)
        sample_indices = np.random.default_rng(self.peak_sample_seed).choice(
            len(train), size=sample_size, replace=False
        )
        candidates: dict[tuple[int, int], float] = {}
        for index in sample_indices:
            spectrum = train[int(index)][1].detach().cpu().numpy()
            peaks, properties = find_peaks(
                spectrum,
                prominence=max(float(np.mean(spectrum)), 0.0),
            )
            if peaks.size == 0:
                continue
            prominences = properties.get("prominences", np.ones_like(peaks))
            selected_order = np.argsort(prominences)[-self.max_peaks_per_spectrum :]
            selected = peaks[selected_order]
            widths = peak_widths(spectrum, selected, rel_height=0.8)
            for prominence_index, left, right in zip(
                selected_order, widths[2], widths[3]
            ):
                start = max(0, int(np.floor(left)))
                stop = min(len(spectrum), int(np.ceil(right)) + 1)
                if stop > start:
                    interval = (start, stop)
                    candidates[interval] = max(
                        candidates.get(interval, 0.0),
                        float(prominences[prominence_index]),
                    )
        catalogue = tuple(
            interval
            for interval, _ in sorted(
                candidates.items(), key=lambda item: item[1], reverse=True
            )[: self.peak_catalog_limit]
        )
        protected = (
            self._build_protected_intervals(dataset)
            if self.peak_selection_method == "permutation_label_invariant"
            else None
        )
        transient_cache[self._cache_key] = {
            "catalogue": catalogue,
            "protected": protected,
        }
        transient_cache["chemical_peak_bank"] = catalogue
        logger.info(
            "Precomputed %s peak envelopes using '%s'.",
            len(catalogue),
            self.peak_selection_method,
        )

    def _build_protected_intervals(
        self,
        dataset: MSIBaseDataset,
    ) -> ProtectedIntervalIndex:
        annotation_reader = getattr(dataset.active_context, "annotation_reader", None)
        bulk_getter = getattr(annotation_reader, "get_spectrum_annotation_index", None)
        if not callable(bulk_getter):
            raise_validation_error(
                "InfoNCELoss",
                "permutation_label_invariant requires a bulk annotation reader.",
            )
        annotation_index = bulk_getter(None)
        axis = np.asarray(dataset.active_context.binner.GetXAxis(), dtype=np.float64)
        offsets = np.zeros(annotation_index.spectrum_ids.size + 1, dtype=np.int64)
        merged_by_row: list[list[tuple[int, int]]] = []
        for row in range(annotation_index.spectrum_ids.size):
            start = int(annotation_index.spectrum_offsets[row])
            stop = int(annotation_index.spectrum_offsets[row + 1])
            intervals = []
            for mz in annotation_index.mz_values[start:stop]:
                if not np.isfinite(mz):
                    continue
                delta = (
                    float(mz) * self.annotation_tolerance * 1e-6
                    if self.annotation_tolerance_unit == "ppm"
                    else self.annotation_tolerance
                )
                left = int(np.searchsorted(axis, float(mz) - delta, side="left"))
                right = int(np.searchsorted(axis, float(mz) + delta, side="right"))
                if right > left:
                    intervals.append((max(0, left), min(axis.size, right)))
            merged = _merge_intervals(intervals)
            merged_by_row.append(merged)
            offsets[row + 1] = offsets[row] + len(merged)
        left_values = np.empty(int(offsets[-1]), dtype=np.int32)
        right_values = np.empty(int(offsets[-1]), dtype=np.int32)
        cursor = 0
        for intervals in merged_by_row:
            for left, right in intervals:
                left_values[cursor] = left
                right_values[cursor] = right
                cursor += 1
        return ProtectedIntervalIndex(
            spectrum_ids=annotation_index.spectrum_ids,
            offsets=offsets,
            left=left_values,
            right=right_values,
        )

    def on_batch_start(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        transient_cache: Dict[str, Any],
    ) -> Tuple[torch.Tensor, ...]:
        """Attach one TIC-preserving permuted view to the current batch."""
        spectra = batch_data.spectra if isinstance(batch_data, SpectrumBatch) else batch_data[1]
        sample_ids = batch_data.sample_ids if isinstance(batch_data, SpectrumBatch) else batch_data[0]
        cached = transient_cache.get(self._cache_key, {})
        catalogue = tuple(cached.get("catalogue", ()))
        protected_index = cached.get("protected")
        augmented = spectra.clone()  # (B, M)
        for row, spectrum_id in enumerate(sample_ids[: spectra.shape[0]].tolist()):
            protected = (
                protected_index.intervals(int(spectrum_id))
                if protected_index is not None
                else ()
            )
            eligible = [
                interval
                for interval in catalogue
                if not _intersects_any(interval, protected)
                and bool(
                    spectra[row, interval[0] : interval[1]].abs().sum()
                    > torch.finfo(spectra.dtype).eps
                )
            ]
            selected = _select_non_overlapping(
                eligible,
                self.permuted_peaks_per_view,
                device=spectra.device,
            )
            if len(selected) < 2:
                continue
            original_row = spectra[row]
            for destination_index, destination in enumerate(selected):
                source = selected[(destination_index + 1) % len(selected)]
                donor = original_row[source[0] : source[1]]  # (W_s,)
                resampled = F.interpolate(
                    donor.view(1, 1, -1),
                    size=destination[1] - destination[0],
                    mode="linear",
                    align_corners=False,
                ).view(-1)  # (W_d,)
                if self.preserve_input_normalization:
                    interpolated_mass = resampled.sum()
                    source_mass = donor.sum()
                    if interpolated_mass.abs() <= torch.finfo(resampled.dtype).eps:
                        resampled = original_row[destination[0] : destination[1]]
                    else:
                        resampled = resampled * (source_mass / interpolated_mass)
                augmented[row, destination[0] : destination[1]] = resampled

        if isinstance(batch_data, SpectrumBatch):
            return batch_data.with_view("contrastive", augmented)
        combined_spectra = torch.cat([spectra, augmented], dim=0)  # (2B, M)
        combined_indices = torch.cat([sample_ids, sample_ids], dim=0)  # (2B,)
        return (combined_indices, combined_spectra, *batch_data[2:])

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return ordinary or label-overlap-weighted symmetric InfoNCE."""
        del kwargs
        if "projection" not in model_outputs:
            raise_incompatible_interface_error(
                "InfoNCELoss", "Model outputs must contain a 'projection' tensor."
            )
        projection = model_outputs["projection"]
        batch_size = (
            batch_data.batch_size
            if isinstance(batch_data, SpectrumBatch)
            else projection.shape[0] // 2
        )
        if projection.shape[0] != 2 * batch_size:
            raise_incompatible_interface_error(
                "InfoNCELoss", "Projection must contain original and augmented halves."
            )
        original = projection[:batch_size]  # (B, D)
        augmented = projection[batch_size:]  # (B, D)
        if self.negative_weighting_method is None:
            return info_nce(original, augmented, temperature=self.temperature).mean()
        if not isinstance(batch_data, SpectrumBatch) or "molecule" not in batch_data.targets.values:
            raise_incompatible_interface_error(
                "InfoNCELoss",
                "multilabel_jaccard weighting requires SpectrumBatch molecule targets.",
            )
        labels = batch_data.targets.values["molecule"][:batch_size].to(
            projection.device
        )  # (B, C)
        return _weighted_info_nce(
            original,
            augmented,
            labels,
            temperature=self.temperature,
            minimum_negative_weight=self.overlapping_label_negative_weight,
        ).mean()


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def _intersects_any(
    interval: tuple[int, int], protected: tuple[tuple[int, int], ...]
) -> bool:
    return any(interval[0] < right and left < interval[1] for left, right in protected)


def _select_non_overlapping(
    intervals: list[tuple[int, int]],
    limit: int,
    *,
    device: torch.device,
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    order = torch.randperm(len(intervals), device=device).cpu().tolist()
    selected: list[tuple[int, int]] = []
    for index in order:
        candidate = intervals[index]
        if _intersects_any(candidate, tuple(selected)):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def _weighted_info_nce(
    original: torch.Tensor,
    augmented: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    minimum_negative_weight: float,
) -> torch.Tensor:
    batch_size = original.shape[0]
    representations = torch.cat(
        [F.normalize(original, dim=1), F.normalize(augmented, dim=1)], dim=0
    )  # (2B, D)
    similarities = representations @ representations.T / temperature  # (2B, 2B)
    binary_labels = (labels > 0.5).to(dtype=similarities.dtype)  # (B, C)
    intersection = binary_labels @ binary_labels.T  # (B, B)
    counts = binary_labels.sum(dim=1)  # (B,)
    union = counts[:, None] + counts[None, :] - intersection  # (B, B)
    jaccard = torch.where(union > 0, intersection / union.clamp_min(1.0), 0.0)  # (B, B)
    negative_weights = 1.0 - (1.0 - minimum_negative_weight) * jaccard  # (B, B)
    negative_weights = negative_weights.repeat(2, 2)  # (2B, 2B)
    rows = torch.arange(2 * batch_size, device=similarities.device)
    pairs = (rows + batch_size) % (2 * batch_size)
    candidate_mask = ~torch.eye(
        2 * batch_size, dtype=torch.bool, device=similarities.device
    )  # (2B, 2B)
    log_weights = negative_weights.clamp_min(
        torch.finfo(similarities.dtype).tiny
    ).log()
    weighted_logits = similarities + log_weights
    weighted_logits[rows, pairs] = similarities[rows, pairs]
    denominator = torch.logsumexp(
        weighted_logits.masked_fill(~candidate_mask, float("-inf")), dim=1
    )  # (2B,)
    return denominator - similarities[rows, pairs]  # (2B,)
