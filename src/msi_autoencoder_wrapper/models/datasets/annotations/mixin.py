"""Reusable dataset interface for annotation-derived targets and selections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import AnnotationSettings, AnnotationTargetSettings
from .index import MappedSpectrumAnnotationIndex
from .manager import DatasetAnnotationManager


class AnnotationAwareDatasetMixin:
    """Provide shared dataset-level annotation mapping functionality.

    Concrete datasets call :meth:`_initialize_annotation_support` after their
    target definition is known. The mixin then applies one virtual source
    selection before normal subsetting and splitting, so all downstream
    consumers observe the same public index space.
    """

    def _initialize_annotation_support(
        self,
        annotation_settings: Mapping[str, Any] | None,
        enabled: bool,
    ) -> None:
        """Initialize the reusable manager after dataset-specific validation.

        :param annotation_settings: Dataset annotation mapping configuration.
        :type annotation_settings: Mapping[str, Any] | None
        :param enabled: Whether this dataset exposes annotation-derived targets.
        :type enabled: bool
        """
        self._annotation_manager = DatasetAnnotationManager(self, annotation_settings)
        self._annotation_support_enabled = bool(enabled)

    def configure_annotations(self, settings: Mapping[str, Any] | None) -> None:
        """Replace annotation settings and invalidate derived dataset state.

        :param settings: Dataset annotation mapping configuration.
        :type settings: Mapping[str, Any] | None
        """
        self._annotation_manager.configure(settings)
        self._selection = None
        self._partitions = None
        if hasattr(self, "_resolved_class_mappings"):
            self._resolved_class_mappings = None
        if hasattr(self, "_config"):
            self._config["annotation_settings"] = self.get_annotation_settings().get_config()

    def get_annotation_settings(self) -> AnnotationSettings:
        """Return validated dataset-level annotation settings.

        :return: Immutable annotation settings.
        :rtype: AnnotationSettings
        """
        return self._annotation_manager.get_settings()

    def get_annotation_target_settings(self, target_field: str) -> AnnotationTargetSettings:
        """Return annotation policies for one target field.

        :param target_field: Annotation-derived target name.
        :type target_field: str
        :return: Target policy configuration.
        :rtype: AnnotationTargetSettings
        """
        return self._annotation_manager.get_target_settings(target_field)

    def get_mapped_annotation_index(self) -> MappedSpectrumAnnotationIndex:
        """Return the sparse annotation index aligned to the dataset x-axis.

        :return: Coordinate-filtered CSR annotation index.
        :rtype: MappedSpectrumAnnotationIndex
        """
        return self._annotation_manager.get_mapped_index()

    def get_annotation_class_names(self) -> tuple[str, ...]:
        """Return canonical molecule names available after x-axis filtering."""
        return self._annotation_manager.get_class_names()

    def resolve_annotation_classes(
        self,
        class_names: list[str] | tuple[str, ...] | None,
    ) -> tuple[int, ...]:
        """Resolve requested molecular classes against mapped dataset targets."""
        return self._annotation_manager.resolve_classes(class_names)

    # Public index-space override
    ## Annotation exclusion is applied before optional user subsetting and splits.
    def __len__(self) -> int:
        """Return visible samples after annotation and explicit selections."""
        if not self._annotation_support_enabled:
            return super().__len__()
        visible = self._annotation_visible_source_indices()
        return int(visible.size) if self._selection is None else len(self._selection)

    def _source_index(self, index: int) -> int:
        """Resolve one public index through both virtual selection layers."""
        if not self._annotation_support_enabled:
            return super()._source_index(index)
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)
        if self._selection is not None:
            return int(self._selection[index])
        return int(self._annotation_visible_source_indices()[index])

    def subset(self, config: Mapping[str, Any] | None = None) -> Any:
        """Apply optional subsetting within the annotation-visible source rows."""
        if not self._annotation_support_enabled:
            return super().subset(config)
        from ..subsetting import DatasetSubsetter, IndexSelection

        self._subset_config = dict(config) if config is not None else None
        if config is None:
            self._selection = None
        else:
            visible = self._annotation_visible_source_indices()

            def group_provider(public_indices: range, **parameters: Any) -> list[Any]:
                source_indices = [int(visible[index]) for index in public_indices]
                return self._source_subset_groups(source_indices, **parameters)

            selected_positions = DatasetSubsetter.select_indices(
                source_length=int(visible.size),
                group_provider=group_provider,
                config=self._subset_config,
            )
            self._selection = IndexSelection(
                tuple(int(visible[position]) for position in selected_positions)
            )
        self._partitions = None
        return self

    def _annotation_visible_source_indices(self) -> np.ndarray:
        """Return source IDs remaining after dataset annotation selection."""
        return self._annotation_manager.get_selected_source_indices(self._source_length())
