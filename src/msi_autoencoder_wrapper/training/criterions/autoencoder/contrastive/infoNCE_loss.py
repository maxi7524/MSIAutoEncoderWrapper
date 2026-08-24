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
from .....normalization import ScalarNormalization
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

    def dense_mask(
        self,
        spectrum_ids: np.ndarray,
        feature_count: int,
    ) -> np.ndarray:
        """Materialize protected bins for one batch from the sparse CSR index.

        :param spectrum_ids: One-dimensional source spectrum identifiers.
        :type spectrum_ids: numpy.ndarray
        :param feature_count: Number of bins in the model spectrum.
        :type feature_count: int
        :return: Batch protection mask with shape ``(B, M)``.
        :rtype: numpy.ndarray
        """
        requested = np.asarray(spectrum_ids, dtype=np.int64).reshape(-1)
        positions = np.searchsorted(self.spectrum_ids, requested)
        valid = positions < self.spectrum_ids.size
        valid[valid] &= self.spectrum_ids[positions[valid]] == requested[valid]
        difference = np.zeros(
            (requested.size, feature_count + 1),
            dtype=np.int32,
        )
        valid_rows = np.flatnonzero(valid)
        if valid_rows.size == 0:
            return difference[:, :-1].astype(bool)

        counts = self.offsets[positions[valid_rows] + 1] - self.offsets[
            positions[valid_rows]
        ]
        entry_indices = np.concatenate(
            [
                np.arange(
                    self.offsets[position],
                    self.offsets[position + 1],
                    dtype=np.int64,
                )
                for position in positions[valid_rows]
            ]
        )
        batch_rows = np.repeat(valid_rows, counts)
        np.add.at(difference, (batch_rows, self.left[entry_indices]), 1)
        np.add.at(difference, (batch_rows, self.right[entry_indices]), -1)
        return np.cumsum(difference[:, :-1], axis=1) > 0


@CriterionsManager.register_criterion("autoencoder", "contrastive", "InfoNCELoss")
class MSIInfoNCELoss(MSIContrastiveCriterion):
    """Make representations invariant to controlled peak permutations.

    ``permutation_random`` builds a train-derived global catalogue and a reusable
    bank of random, non-overlapping envelope groups. Every batch samples a small
    group pool tensorwise without scanning the catalogue. It is the control
    strategy and may alter annotated evidence.

    ``permutation_label_invariant`` resolves protected intervals separately for
    every spectrum through ``spectrum_id -> mapped dataset annotations ->
    binner bins``. The dataset owns the exact raw m/z-to-bin mapping, so an
    annotation protects the same bin as its binned peak. A candidate is eligible
    only when it does not intersect that spectrum's protected intervals.

    Selected envelope shapes and masses are cyclically reassigned within one
    spectrum using one batched gather/interpolate/scatter operation. If
    ``preserve_input_normalization`` is true, interpolation keeps every donor
    envelope's original mass, so the selected-region sum and the per-spectrum
    TIC remain invariant. Protected bins are unchanged. Original and augmented
    spectra form positive pairs.

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
    :param permutation_bank_size: Number of precomputed random, non-overlapping
        envelope groups retained for batchwise sampling.
    :type permutation_bank_size: int
    :param permuted_peaks_per_view: Maximum envelopes permuted per view.
    :type permuted_peaks_per_view: int
    :param permutation_selection_attempts: Number of precomputed groups sampled
        simultaneously per spectrum when searching for non-empty, unprotected
        envelopes.
    :type permutation_selection_attempts: int
    :param annotation_bin_radius: Optional number of adjacent binner bins to
        protect on either side of every mapped annotation coordinate.
    :type annotation_bin_radius: int
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
        permutation_bank_size: int = 7000,
        permuted_peaks_per_view: int = 3,
        permutation_selection_attempts: int = 64,
        annotation_bin_radius: int = 0,
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
        if temperature <= 0:
            raise_validation_error(
                "InfoNCELoss", "temperature must be positive."
            )
        if (
            isinstance(annotation_bin_radius, bool)
            or not isinstance(annotation_bin_radius, int)
            or annotation_bin_radius < 0
        ):
            raise_validation_error(
                "InfoNCELoss", "annotation_bin_radius must be a non-negative integer."
            )
        integer_parameters = {
            "peak_sample_size": peak_sample_size,
            "peak_catalog_limit": peak_catalog_limit,
            "permutation_bank_size": permutation_bank_size,
            "permuted_peaks_per_view": (
                max_noise_peaks if max_noise_peaks is not None else permuted_peaks_per_view
            ),
            "permutation_selection_attempts": permutation_selection_attempts,
            "max_peaks_per_spectrum": max_peaks_per_spectrum,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integer_parameters.values()
        ):
            raise_validation_error(
                "InfoNCELoss", "Peak sampling and permutation limits must be positive integers."
            )
        if integer_parameters["permuted_peaks_per_view"] < 2:
            raise_validation_error(
                "InfoNCELoss", "permuted_peaks_per_view must be at least two."
            )
        if negative_weighting_method is not None and (
            negative_weighting_method not in SUPPORTED_NEGATIVE_WEIGHTING_METHODS
        ):
            raise_validation_error(
                "InfoNCELoss",
                (
                    "negative_weighting_method must be one of "
                    f"{sorted(SUPPORTED_NEGATIVE_WEIGHTING_METHODS)}."
                ),
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
        self.permutation_bank_size = permutation_bank_size
        self.permuted_peaks_per_view = integer_parameters["permuted_peaks_per_view"]
        self.permutation_selection_attempts = integer_parameters[
            "permutation_selection_attempts"
        ]
        self.annotation_bin_radius = int(annotation_bin_radius)
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
                permutation_bank_size,
                self.permuted_peaks_per_view,
                max_peaks_per_spectrum,
                annotation_bin_radius,
            )
        )
        self._config = {
            "peak_selection_method": peak_selection_method,
            "temperature": self.temperature,
            "peak_sample_size": peak_sample_size,
            "peak_sample_seed": peak_sample_seed,
            "peak_catalog_limit": peak_catalog_limit,
            "permutation_bank_size": permutation_bank_size,
            "permuted_peaks_per_view": self.permuted_peaks_per_view,
            "permutation_selection_attempts": self.permutation_selection_attempts,
            "annotation_bin_radius": self.annotation_bin_radius,
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
        permutation_groups = _build_permutation_groups(
            catalogue,
            group_size=self.permuted_peaks_per_view,
            group_count=self.permutation_bank_size,
            seed=self.peak_sample_seed,
        )
        protected = (
            self._build_protected_intervals(dataset)
            if self.peak_selection_method == "permutation_label_invariant"
            else None
        )
        transient_cache[self._cache_key] = {
            "catalogue": catalogue,
            "permutation_groups": permutation_groups,
            "max_envelope_width": max(
                (right - left for left, right in catalogue),
                default=1,
            ),
            "device_group_cache": {},
            "protected": protected,
        }
        transient_cache["chemical_peak_bank"] = catalogue
        logger.info(
            "Precomputed %s peak envelopes and %s permutation groups using '%s'.",
            len(catalogue),
            permutation_groups.shape[0],
            self.peak_selection_method,
        )

    def _build_protected_intervals(
        self,
        dataset: MSIBaseDataset,
    ) -> ProtectedIntervalIndex:
        mapped_getter = getattr(dataset, "get_mapped_annotation_index", None)
        if not callable(mapped_getter):
            raise_validation_error(
                "InfoNCELoss",
                "permutation_label_invariant requires a dataset-level mapped annotation index.",
            )
        annotation_index = mapped_getter()
        if annotation_index.coordinate_system != "binner":
            raise_validation_error(
                "InfoNCELoss",
                "permutation_label_invariant requires annotation mapping to binner coordinates.",
            )
        axis_size = int(annotation_index.coordinate_axis.size)
        offsets = np.zeros(annotation_index.spectrum_ids.size + 1, dtype=np.int64)
        merged_by_row: list[list[tuple[int, int]]] = []
        for row in range(annotation_index.spectrum_ids.size):
            spectrum_id = int(annotation_index.spectrum_ids[row])
            coordinates = annotation_index.coordinates_for_spectrum(spectrum_id)
            intervals = [
                (
                    max(0, int(coordinate) - self.annotation_bin_radius),
                    min(axis_size, int(coordinate) + self.annotation_bin_radius + 1),
                )
                for coordinate in coordinates
            ]
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
        sample_ids = (
            batch_data.sample_ids
            if isinstance(batch_data, SpectrumBatch)
            else batch_data[0]
        )
        cached = transient_cache.get(self._cache_key, {})
        protected_index = cached.get("protected")
        # Contrastive view construction
        ## Resolve the precomputed global bank once per compute device.
        source_spectra = spectra.detach()  # (B, M)
        groups, max_envelope_width = self._device_permutation_groups(
            cached,
            spectra.device,
        )  # (G, S, 2), scalar
        protected_mask = self._batch_protected_mask(
            protected_index,
            sample_ids[: spectra.shape[0]],
            spectra.shape[1],
            spectra.device,
        )
        selected, active = _select_permutation_groups(
            source_spectra,
            groups,
            protected_mask=protected_mask,
            attempt_count=self.permutation_selection_attempts,
        )  # (B, S, 2), (B,)
        augmented = _permute_envelopes_batch(
            source_spectra,
            selected,
            active,
            max_envelope_width=max_envelope_width,
            preserve_mass=self.preserve_input_normalization,
        )  # (B, M)

        if self.preserve_input_normalization and isinstance(batch_data, SpectrumBatch):
            normalization_name = batch_data.space.normalization
            if normalization_name in {"tic", "max", "l2"}:
                # Post-augmentation input normalization
                ## Scalar samplewise normalizations are positively homogeneous,
                ## so N(P(N(x))) equals N(P(x)) without restoring source counts.
                normalizer = ScalarNormalization(kind=normalization_name)
                augmented, _ = normalizer.transform(augmented)  # (B, M)
            elif normalization_name != "none":
                raise_validation_error(
                    "InfoNCELoss",
                    (
                        "preserve_input_normalization supports typed batch spaces "
                        "'none', 'tic', 'max', and 'l2'."
                    ),
                )

        if isinstance(batch_data, SpectrumBatch):
            return batch_data.with_view("contrastive", augmented)
        combined_spectra = torch.cat([spectra, augmented], dim=0)  # (2B, M)
        combined_indices = torch.cat([sample_ids, sample_ids], dim=0)  # (2B,)
        return (combined_indices, combined_spectra, *batch_data[2:])

    def _device_permutation_groups(
        self,
        cached: Dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        """Return the immutable global permutation bank on one compute device."""
        groups = cached.get("permutation_groups")
        if groups is None:
            groups = _build_permutation_groups(
                tuple(cached.get("catalogue", ())),
                group_size=self.permuted_peaks_per_view,
                group_count=self.permutation_bank_size,
                seed=self.peak_sample_seed,
            )
            cached["permutation_groups"] = groups
            cached["max_envelope_width"] = max(
                (
                    right - left
                    for left, right in tuple(cached.get("catalogue", ()))
                ),
                default=1,
            )
        device_cache = cached.setdefault("device_group_cache", {})
        device_key = (device.type, device.index)
        if device_key not in device_cache:
            device_cache[device_key] = torch.as_tensor(
                groups,
                dtype=torch.long,
                device=device,
            )
        return device_cache[device_key], int(cached["max_envelope_width"])

    @staticmethod
    def _batch_protected_mask(
        protected_index: ProtectedIntervalIndex | None,
        sample_ids: torch.Tensor,
        feature_count: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Transfer one sparse-index batch mask with a single host synchronization."""
        if protected_index is None:
            return None
        spectrum_ids = sample_ids.detach().to(device="cpu").numpy()
        mask = protected_index.dense_mask(spectrum_ids, feature_count)
        return torch.as_tensor(mask, dtype=torch.bool, device=device)  # (B, M)

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


def _build_permutation_groups(
    catalogue: tuple[tuple[int, int], ...],
    *,
    group_size: int,
    group_count: int,
    seed: int,
) -> np.ndarray:
    """Precompute random groups of pairwise non-overlapping peak envelopes."""
    if len(catalogue) < group_size:
        return np.empty((0, group_size, 2), dtype=np.int32)
    intervals = np.asarray(catalogue, dtype=np.int32)
    generator = np.random.default_rng(seed)
    groups = np.empty((group_count, group_size, 2), dtype=np.int32)
    retained = 0
    for _ in range(group_count):
        selected: list[tuple[int, int]] = []
        for index in generator.permutation(len(catalogue)):
            candidate = tuple(int(value) for value in intervals[index])
            if _intersects_any(candidate, tuple(selected)):
                continue
            selected.append(candidate)
            if len(selected) == group_size:
                groups[retained] = np.asarray(selected, dtype=np.int32)
                retained += 1
                break
    return groups[:retained]


def _select_permutation_groups(
    spectra: torch.Tensor,
    groups: torch.Tensor,
    *,
    protected_mask: torch.Tensor | None,
    attempt_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select valid global-bank groups for a complete batch without scalar syncs."""
    batch_size, feature_count = spectra.shape
    group_size = groups.shape[1]
    if groups.shape[0] == 0:
        selected = torch.zeros(
            (batch_size, group_size, 2),
            dtype=torch.long,
            device=spectra.device,
        )
        selected[..., 1] = 1
        return selected, torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=spectra.device,
        )

    # Candidate selection
    ## Sample a small candidate pool from the precomputed bank in one operation.
    candidate_ids = torch.randint(
        groups.shape[0],
        (batch_size, attempt_count),
        device=spectra.device,
    )  # (B, A)
    candidates = groups[candidate_ids]  # (B, A, S, 2)
    left = candidates[..., 0]  # (B, A, S)
    right = candidates[..., 1]  # (B, A, S)
    prefix = F.pad(spectra.cumsum(dim=1), (1, 0))  # (B, M + 1)
    candidate_mass = torch.gather(
        prefix,
        1,
        right.reshape(batch_size, -1),
    ) - torch.gather(
        prefix,
        1,
        left.reshape(batch_size, -1),
    )  # (B, A * S)
    candidate_mass = candidate_mass.view(
        batch_size,
        attempt_count,
        group_size,
    )  # (B, A, S)
    valid = candidate_mass > torch.finfo(spectra.dtype).eps  # (B, A, S)

    if protected_mask is not None:
        protected_prefix = F.pad(
            protected_mask.to(dtype=torch.int32).cumsum(dim=1),
            (1, 0),
        )  # (B, M + 1)
        protected_count = torch.gather(
            protected_prefix,
            1,
            right.reshape(batch_size, -1),
        ) - torch.gather(
            protected_prefix,
            1,
            left.reshape(batch_size, -1),
        )  # (B, A * S)
        valid &= protected_count.view(
            batch_size,
            attempt_count,
            group_size,
        ) == 0

    # First valid group
    ## Invalid spectra retain their original view; no host-side branch is needed.
    valid_groups = valid.all(dim=2)  # (B, A)
    attempt_indices = torch.arange(
        attempt_count,
        device=spectra.device,
    ).unsqueeze(0)  # (1, A)
    first_valid = torch.where(
        valid_groups,
        attempt_indices,
        attempt_count,
    ).amin(dim=1)  # (B,)
    active = first_valid < attempt_count  # (B,)
    safe_indices = first_valid.clamp_max(attempt_count - 1)
    batch_indices = torch.arange(batch_size, device=spectra.device)
    selected = candidates[batch_indices, safe_indices]  # (B, S, 2)
    return selected, active


def _permute_envelopes_batch(
    spectra: torch.Tensor,
    selected: torch.Tensor,
    active: torch.Tensor,
    *,
    max_envelope_width: int,
    preserve_mass: bool,
) -> torch.Tensor:
    """Cyclically permute variable-width envelopes using batched gather/scatter."""
    batch_size, feature_count = spectra.shape
    destination_left = selected[..., 0]  # (B, S)
    destination_right = selected[..., 1]  # (B, S)
    destination_width = destination_right - destination_left  # (B, S)
    source_left = destination_left.roll(shifts=-1, dims=1)  # (B, S)
    source_right = destination_right.roll(shifts=-1, dims=1)  # (B, S)
    source_width = source_right - source_left  # (B, S)
    positions = torch.arange(
        max_envelope_width,
        dtype=spectra.dtype,
        device=spectra.device,
    ).view(1, 1, -1)  # (1, 1, W)
    valid_positions = positions < destination_width.unsqueeze(2)  # (B, S, W)

    # Batched linear interpolation
    ## Reproduce align_corners=False coordinates for every variable-width pair.
    source_coordinate = (
        (positions + 0.5)
        * source_width.unsqueeze(2).to(dtype=spectra.dtype)
        / destination_width.unsqueeze(2).clamp_min(1).to(dtype=spectra.dtype)
        - 0.5
    ).clamp_min(0.0)  # (B, S, W)
    lower = source_coordinate.floor().to(dtype=torch.long)
    maximum_source = (source_width - 1).clamp_min(0).unsqueeze(2)
    lower = torch.minimum(lower, maximum_source)
    upper = torch.minimum(lower + 1, maximum_source)
    interpolation_weight = source_coordinate - lower.to(dtype=spectra.dtype)
    row_offsets = (
        torch.arange(batch_size, device=spectra.device) * feature_count
    ).view(batch_size, 1, 1)  # (B, 1, 1)
    flat_spectra = spectra.reshape(-1)  # (B * M,)
    lower_values = flat_spectra[
        row_offsets + source_left.unsqueeze(2) + lower
    ]  # (B, S, W)
    upper_values = flat_spectra[
        row_offsets + source_left.unsqueeze(2) + upper
    ]  # (B, S, W)
    resampled = lower_values + (
        upper_values - lower_values
    ) * interpolation_weight  # (B, S, W)
    resampled = resampled * valid_positions

    if preserve_mass:
        # Envelope mass preservation
        ## A cyclic permutation retains the mutable-region TIC before the final
        ## samplewise normalization, leaving all unselected bins unchanged.
        prefix = F.pad(spectra.cumsum(dim=1), (1, 0))  # (B, M + 1)
        source_mass = torch.gather(prefix, 1, source_right) - torch.gather(
            prefix,
            1,
            source_left,
        )  # (B, S)
        interpolated_mass = resampled.sum(dim=2)  # (B, S)
        scale = source_mass / interpolated_mass.clamp_min(
            torch.finfo(spectra.dtype).eps
        )
        resampled = resampled * scale.unsqueeze(2)  # (B, S, W)

    # Batched destination write
    ## Precomputed groups are pairwise disjoint, so every destination index is unique.
    destination_indices = (
        row_offsets
        + destination_left.unsqueeze(2)
        + positions.to(dtype=torch.long)
    )  # (B, S, W)
    write_mask = valid_positions & active.view(batch_size, 1, 1)
    augmented = spectra.clone().reshape(-1)  # (B * M,)
    augmented.scatter_(
        0,
        destination_indices[write_mask],
        resampled[write_mask],
    )
    return augmented.view(batch_size, feature_count)  # (B, M)


def _intersects_any(
    interval: tuple[int, int], protected: tuple[tuple[int, int], ...]
) -> bool:
    return any(interval[0] < right and left < interval[1] for left, right in protected)


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
