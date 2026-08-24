"""Datasets that concatenate immutable cohort-member contexts."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

from ..base_dataset import RawMSIBaseDataset
from ..dataset_manager import DatasetManager
from ....core.mixins.cohort.context import CohortContext, CohortMember
from ....utils.exceptions import raise_validation_error
from .pixel_dataset import PixelDataset


@dataclass
class _MemberContext:
    """Expose the local context contract without mutating workspace state."""

    member: CohortMember

    @property
    def reader(self) -> Any:
        return self.member.reader

    @property
    def binner(self) -> Any:
        return self.member.binner

    @property
    def annotation_reader(self) -> Any:
        return self.member.annotation_reader

    def get_data_reader(self, source: str) -> Any:
        return self.member.get_reader(source)


class CohortDataset(RawMSIBaseDataset):
    """Concatenate member datasets while retaining stable image/sample identity."""

    source: Literal["image", "latent"]

    def __init__(
        self,
        cohort_context: CohortContext,
        *,
        source: Literal["image", "latent"],
        normalization: Optional[str] = None,
        normalization_epsilon: float = 1e-12,
        target_specs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        annotation_settings: Optional[Mapping[str, Any]] = None,
        split: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(active_context=None, split=split, **kwargs)
        if not cohort_context.members:
            raise_validation_error("CohortDataset", "The cohort has no members.")
        self.cohort_context = cohort_context
        self.source = source
        specs = dict(target_specs or {})
        self._datasets = tuple(
            PixelDataset(
                active_context=_MemberContext(member),
                source=source,
                normalization=normalization,
                normalization_epsilon=normalization_epsilon,
                target_specs=specs,
                annotation_settings=annotation_settings,
            )
            for member in cohort_context.members
        )
        lengths = [len(dataset) for dataset in self._datasets]
        self._offsets = tuple(accumulate(lengths))
        self._config = {
            "cohort": cohort_context.get_config(),
            "source": source,
            "normalization": normalization,
            "normalization_epsilon": normalization_epsilon,
            "target_specs": specs,
            "annotation_settings": dict(annotation_settings or {}),
            "split": self.get_split_config(),
        }

    def _source_length(self) -> int:
        """Return the total number of source pixels across cohort members."""
        return self._offsets[-1]

    def _get_source_item(self, idx: int) -> Tuple[Any, ...]:
        member_index, local_index = self._resolve_index(idx)
        sample = self._datasets[member_index][local_index]
        return ((self.cohort_context.members[member_index].image_key, sample[0]), *sample[1:])

    def _get_raw_source_item(self, idx: int) -> Any:
        member_index, local_index = self._resolve_index(idx)
        sample = self._datasets[member_index].get_raw_item(local_index)
        sample_id = (self.cohort_context.members[member_index].image_key, sample.sample_id)
        return type(sample)(
            sample_id=sample_id,
            mass_values=sample.mass_values,
            intensities=sample.intensities,
            targets=sample.targets,
        )

    def _get_source_sample_id(self, index: int) -> Any:
        member_index, local_index = self._resolve_index(index)
        return {
            "image_key": self.cohort_context.members[member_index].image_key,
            "spectrum_id": local_index,
        }

    def get_target_schemas(self) -> Dict[str, Any]:
        """Return schemas shared by all members."""
        schemas = self._datasets[0].get_target_schemas()
        for dataset in self._datasets[1:]:
            if dataset.get_target_schemas() != schemas:
                raise_validation_error(
                    "CohortDataset", "Cohort member target schemas are inconsistent."
                )
        return schemas

    def _get_source_split_target(self, idx: int, **parameters: Any) -> Any:
        member, local = self._resolve_index(idx)
        return self._datasets[member]._get_source_split_target(local, **parameters)

    def _get_source_split_mask(self, idx: int, **parameters: Any) -> Any:
        member, local = self._resolve_index(idx)
        return self._datasets[member]._get_source_split_mask(local, **parameters)

    def _get_source_split_group(self, idx: int, **parameters: Any) -> Any:
        member, local = self._resolve_index(idx)
        group_fields = parameters.get("group_fields")
        if group_fields == "image_key" or group_fields == ["image_key"]:
            return self.cohort_context.members[member].image_key
        return self._datasets[member]._get_source_split_group(local, **parameters)

    def _source_subset_groups(self, source_indices: range, **parameters: Any) -> list[Any]:
        """Stratify cohort sampling by source image unless overridden."""
        group_fields = parameters.get("group_fields")
        groups = []
        for source_index in source_indices:
            member, local = self._resolve_index(source_index)
            if group_fields == "image_key" or group_fields == ["image_key"]:
                groups.append(self.cohort_context.members[member].image_key)
            else:
                groups.append(
                    self._datasets[member]._get_source_split_group(
                        local,
                        **parameters,
                    )
                )
        return groups

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._source_length()
        if idx < 0 or idx >= self._source_length():
            raise IndexError(idx)
        member_index = bisect_right(self._offsets, idx)
        start = 0 if member_index == 0 else self._offsets[member_index - 1]
        return member_index, idx - start


@DatasetManager.register_dataset("CohortPixelDataset")
class CohortPixelDataset(CohortDataset):
    """Expose original spectra from every cohort member."""

    def __init__(self, cohort_context: CohortContext, **kwargs: Any) -> None:
        super().__init__(cohort_context, source="image", **kwargs)


@DatasetManager.register_dataset("CohortLatentDataset")
class CohortLatentDataset(CohortDataset):
    """Expose only previously materialized cohort-member latent spectra."""

    def __init__(self, cohort_context: CohortContext, **kwargs: Any) -> None:
        missing = [
            member.image_key
            for member in cohort_context.members
            if member.latent_reader is None
        ]
        if missing:
            raise_validation_error(
                "CohortLatentDataset",
                f"Materialized latent readers are required for: {missing}.",
            )
        super().__init__(cohort_context, source="latent", **kwargs)
