"""Dataset-level projection of raw annotation indices onto model coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import AnnotationSettings, AnnotationTargetSettings
from .index import MappedSpectrumAnnotationIndex
from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


# Process-local mapped-index cache
## Workers commonly plan or train repeated model variants against the same reader
## and binner. The sparse projection is independent of neural-network weights.
_MAPPED_INDEX_CACHE: dict[tuple[Any, ...], MappedSpectrumAnnotationIndex] = {}


class DatasetAnnotationManager:
    """Build and retain the sparse annotation view required by one dataset.

    The annotation reader owns the raw molecular identity and m/z index. This
    manager owns its projection to the coordinate system used by the dataset,
    including removal of annotations outside that coordinate system.
    """

    def __init__(self, dataset: Any, settings: Mapping[str, Any] | None = None) -> None:
        """Initialize a manager without accessing the annotation reader.

        :param dataset: Dataset exposing an active context and, for binner
            mapping, an active binner.
        :type dataset: Any
        :param settings: Declarative annotation mapping configuration.
        :type settings: Mapping[str, Any] | None
        """
        self._dataset = dataset
        self._settings = AnnotationSettings.from_config(settings)
        self._mapped_index: MappedSpectrumAnnotationIndex | None = None
        self._selected_source_indices: np.ndarray | None = None

    def configure(self, settings: Mapping[str, Any] | None) -> None:
        """Replace settings and invalidate all derived dataset-level state.

        :param settings: Declarative annotation mapping configuration.
        :type settings: Mapping[str, Any] | None
        """
        self._settings = AnnotationSettings.from_config(settings)
        self._mapped_index = None
        self._selected_source_indices = None

    def get_settings(self) -> AnnotationSettings:
        """Return the validated immutable annotation settings.

        :return: Dataset annotation settings.
        :rtype: AnnotationSettings
        """
        return self._settings

    def get_target_settings(self, target_field: str) -> AnnotationTargetSettings:
        """Return target-specific annotation behaviour.

        :param target_field: Name of an annotation-derived target.
        :type target_field: str
        :return: Validated target settings.
        :rtype: AnnotationTargetSettings
        """
        return self._settings.target(target_field)

    def get_mapped_index(self) -> MappedSpectrumAnnotationIndex:
        """Return the sparse annotation index mapped to dataset coordinates.

        :return: Compact index containing only in-range annotation entries.
        :rtype: MappedSpectrumAnnotationIndex
        :raises ValidationError: If the active context cannot provide the raw
            annotation index or the requested coordinate mapping.
        """
        if self._mapped_index is None:
            self._mapped_index = self._build_mapped_index()
        return self._mapped_index

    def get_class_names(self) -> tuple[str, ...]:
        """Return canonical molecule class names after coordinate filtering.

        :return: Formula/adduct names in deterministic index order.
        :rtype: tuple[str, ...]
        """
        return tuple(
            f"{formula}|{adduct}"
            for formula, adduct in self.get_mapped_index().annotation_identities
        )

    def resolve_classes(self, class_names: list[str] | tuple[str, ...] | None) -> tuple[int, ...]:
        """Resolve requested canonical molecule names to target indices.

        :param class_names: Requested names, or ``None`` for every mapped class.
        :type class_names: list[str] | tuple[str, ...] | None
        :return: Target indices in requested order.
        :rtype: tuple[int, ...]
        :raises ValidationError: If a requested class is unavailable after
            coordinate filtering.
        """
        available = self.get_class_names()
        if class_names is None:
            return tuple(range(len(available)))
        lookup = {name: index for index, name in enumerate(available)}
        resolved: list[int] = []
        for name in class_names:
            index = lookup.get(str(name))
            if index is None:
                raise_validation_error(
                    "DatasetAnnotationManager",
                    f"Annotation class '{name}' is unavailable after coordinate mapping.",
                )
            resolved.append(index)
        return tuple(resolved)

    def get_selected_source_indices(self, source_length: int) -> np.ndarray:
        """Return selected raw source identifiers after annotation policies.

        :param source_length: Number of rows in the source spectrum reader.
        :type source_length: int
        :return: Sorted source identifiers visible through the dataset.
        :rtype: numpy.ndarray
        """
        if self._selected_source_indices is None:
            self._selected_source_indices = self._select_source_indices(source_length)
        return self._selected_source_indices

    # Sparse mapping
    ## The reader remains independent of bins, models, and training policies.
    def _build_mapped_index(self) -> MappedSpectrumAnnotationIndex:
        """Map the reader-owned CSR entries to the configured coordinate system."""
        annotation_reader = getattr(self._dataset.active_context, "annotation_reader", None)
        raw_getter = getattr(annotation_reader, "get_spectrum_annotation_index", None)
        if not callable(raw_getter):
            raise_validation_error(
                "DatasetAnnotationManager",
                "Annotation mapping requires get_spectrum_annotation_index() on the reader.",
            )
        raw_index = raw_getter(None)
        cache_key = self._mapped_index_cache_key(raw_index)
        cached = _MAPPED_INDEX_CACHE.get(cache_key)
        if cached is not None:
            logger.debug(
                "Reusing mapped annotation index: rows=%s entries=%s coordinate_system=%s.",
                cached.spectrum_ids.size,
                cached.annotation_indices.size,
                cached.coordinate_system,
            )
            return cached
        mapped_coordinates, coordinate_axis = self._map_coordinates(raw_index.mz_values)
        valid = mapped_coordinates >= 0  # (E,)

        ## Preserve CSR ordering while removing all entries outside the output axis.
        filtered_annotation_indices = raw_index.annotation_indices[valid].astype(np.int32, copy=False)
        filtered_coordinates = mapped_coordinates[valid].astype(np.int32, copy=False)
        cumulative_valid = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(valid, dtype=np.int64))
        )
        row_counts = (
            cumulative_valid[raw_index.spectrum_offsets[1:]]
            - cumulative_valid[raw_index.spectrum_offsets[:-1]]
            if raw_index.spectrum_ids.size
            else np.empty(0, dtype=np.int64)
        )
        retained_rows = row_counts > 0  # (R,)
        retained_spectrum_ids = raw_index.spectrum_ids[retained_rows].astype(np.int64, copy=True)
        retained_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(row_counts[retained_rows], dtype=np.int64))
        )

        ## Compact class identities to the subset that remains inside the mapped axis.
        used_annotation_indices = np.unique(filtered_annotation_indices)
        remapping = np.full(len(raw_index.annotation_identities), -1, dtype=np.int32)
        remapping[used_annotation_indices] = np.arange(used_annotation_indices.size, dtype=np.int32)
        remapped_annotation_indices = remapping[filtered_annotation_indices]
        identities = tuple(raw_index.annotation_identities[int(index)] for index in used_annotation_indices)

        for array in (
            retained_spectrum_ids,
            retained_offsets,
            remapped_annotation_indices,
            filtered_coordinates,
            coordinate_axis,
        ):
            array.setflags(write=False)
        mapped_index = MappedSpectrumAnnotationIndex(
            spectrum_ids=retained_spectrum_ids,
            spectrum_offsets=retained_offsets,
            annotation_indices=remapped_annotation_indices,
            coordinate_indices=filtered_coordinates,
            annotation_identities=identities,
            coordinate_axis=coordinate_axis,
            coordinate_system=self._settings.x_mapping,
        )
        _MAPPED_INDEX_CACHE[cache_key] = mapped_index
        logger.info(
            "Mapped annotation index: retained_rows=%s retained_entries=%s coordinate_system=%s.",
            mapped_index.spectrum_ids.size,
            mapped_index.annotation_indices.size,
            mapped_index.coordinate_system,
        )
        return mapped_index

    def _mapped_index_cache_key(self, raw_index: Any) -> tuple[Any, ...]:
        """Build a process-local key independent of model and run identity."""
        binner = getattr(self._dataset.active_context, "binner", None)
        binner_config = (
            _freeze_config(binner.get_config())
            if self._settings.x_mapping == "binner" and callable(getattr(binner, "get_config", None))
            else type(binner).__qualname__ if self._settings.x_mapping == "binner" else None
        )
        return (
            id(raw_index),
            self._settings.x_mapping,
            binner_config,
        )

    def _map_coordinates(self, mz_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map raw m/z values to the configured dataset coordinate system."""
        mz_array = np.asarray(mz_values, dtype=np.float64)
        if self._settings.x_mapping == "binner":
            binner = getattr(self._dataset.active_context, "binner", None)
            mapper = getattr(binner, "map_mass_values_to_bins", None)
            if not callable(mapper):
                raise_validation_error(
                    "DatasetAnnotationManager",
                    "mapping.x_mapping='binner' requires map_mass_values_to_bins() on the binner.",
                )
            coordinates = np.asarray(mapper(mz_array), dtype=np.int32)
            axis = np.asarray(binner.GetXAxis(), dtype=np.float64).copy()
            return coordinates, axis

        valid = np.isfinite(mz_array)
        axis = np.unique(mz_array[valid])
        coordinates = np.full(mz_array.shape, -1, dtype=np.int32)
        coordinates[valid] = np.searchsorted(axis, mz_array[valid]).astype(np.int32)
        return coordinates, axis

    # Dataset selection
    ## Annotation availability is evaluated after the coordinate mapping above.
    def _select_source_indices(self, source_length: int) -> np.ndarray:
        """Select source rows according to availability and fraction policies."""
        if source_length < 0:
            raise_validation_error("DatasetAnnotationManager", "source_length cannot be negative.")
        index = self.get_mapped_index()
        annotated = index.spectrum_ids[
            (index.spectrum_ids >= 0) & (index.spectrum_ids < source_length)
        ]
        annotated = np.unique(annotated.astype(np.int64, copy=False))
        all_indices = np.arange(source_length, dtype=np.int64)
        unannotated = np.setdiff1d(all_indices, annotated, assume_unique=True)
        target = self.get_target_settings("molecule")
        if target.empty_spectrum_policy == "exclude":
            selected = self._sample_indices(annotated, self._settings.annotated_fraction)
        else:
            selected_annotated = self._sample_indices(annotated, self._settings.annotated_fraction)
            selected_unannotated = self._sample_indices(
                unannotated,
                self._settings.unannotated_fraction,
                offset=1,
            )
            ratio = self._settings.max_unannotated_to_annotated_ratio
            if ratio is not None:
                limit = int(np.floor(ratio * selected_annotated.size))
                selected_unannotated = selected_unannotated[:limit]
            selected = np.sort(np.concatenate((selected_annotated, selected_unannotated)))
        selected.setflags(write=False)
        logger.info(
            "Selected %s/%s source spectra after annotation policies.",
            selected.size,
            source_length,
        )
        return selected

    def _sample_indices(self, indices: np.ndarray, fraction: float, offset: int = 0) -> np.ndarray:
        """Deterministically sample a fraction while retaining sorted source IDs."""
        count = int(np.floor(indices.size * fraction))
        if fraction == 1.0:
            return indices.copy()
        if count == 0:
            return np.empty(0, dtype=np.int64)
        generator = np.random.default_rng(self._settings.seed + offset)
        return np.sort(generator.choice(indices, size=count, replace=False).astype(np.int64))


def _freeze_config(value: Any) -> Any:
    """Convert nested component configuration values to an immutable cache key."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_config(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item) for item in value)
    return value
