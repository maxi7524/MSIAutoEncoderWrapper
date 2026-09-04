"""
Concrete dataset strategy executing single-pixel spectra mapping sequences driven by an active context.
"""

from collections import defaultdict
from typing import Tuple, Any, Optional, Literal, Mapping, Dict, Sequence
import random
import torch
import numpy as np

from ..dataset_manager import DatasetManager
from ..base_dataset import RawMSIBaseDataset
from ..annotations import AnnotationAwareDatasetMixin
from ....utils.logger import get_custom_logger
from ....utils.exceptions import raise_validation_error
from ....core.mixins.active_context.active_context_mixin import ActiveContextProxy
from ..class_assignment import build_class_mapping, metadata_values, molecule_key
from ....data import (
    RawSpectrumCollator,
    RawSpectrumSample,
    SharedAxisRawBatch,
    TargetBatch,
    TargetSample,
    TargetSchema,
)

# Logger initialization
logger = get_custom_logger(__name__)


@DatasetManager.register_dataset("PixelDataset")
class PixelDataset(AnnotationAwareDatasetMixin, RawMSIBaseDataset):
    """
    Concrete single-pixel sampling strategy that pulls raw arrays and maps them onto uniform grids.
    """

    def __init__(
        self,
        active_context: Optional[ActiveContextProxy] = None,
        source: Literal["image", "latent"] = "image",
        normalization: Optional[Literal["none", "tic", "max", "l2"]] = None,
        normalization_epsilon: float = 1e-12,
        target_specs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        annotation_settings: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Constructs the independent pixel sampling dataset pipeline layer.

        :param active_context: Active execution session proxy tracking live datasets.
        :type active_context: Optional[ActiveContextProxy]
        :param source: Image or latent data source.
        :type source: Literal["image", "latent"]
        :param normalization: Spectrum scaling. Defaults to ``tic`` for images
            and ``none`` for latent data.
        :type normalization: Optional[Literal["none", "tic", "max", "l2"]]
        :param normalization_epsilon: Positive denominator safety threshold.
        :type normalization_epsilon: float
        :param target_specs: Target definitions keyed by annotation field. Each
            definition requires ``type`` (``single_label`` or ``multi_label``)
            and may provide ``class_mapping``.
        :type target_specs: Mapping[str, Mapping[str, Any]] | None
        :param annotation_settings: Mapping, selection, and target policies for
            reader-derived molecular annotations.
        :type annotation_settings: Mapping[str, Any] | None
        :raises ValidationError: If normalization settings are invalid.
        """
        super().__init__(active_context=active_context, **kwargs)
        resolved_normalization = normalization or (
            "none" if source == "latent" else "tic"
        )
        if resolved_normalization not in {"none", "tic", "max", "l2"}:
            raise_validation_error(
                context_name="PixelDataset",
                message=(
                    "normalization must be 'none', 'tic', 'max', or 'l2'."
                ),
            )
        if normalization_epsilon <= 0:
            raise_validation_error(
                context_name="PixelDataset",
                message="normalization_epsilon must be greater than zero.",
            )
        self.source = source
        self.dtype = getattr(getattr(active_context, "_wrapper", None), "dtype", torch.float32)
        self.normalization = resolved_normalization
        self.normalization_epsilon = float(normalization_epsilon)
        self.target_specs = self._validate_target_specs(target_specs or {})
        self._resolved_class_mappings: Optional[Dict[str, Dict[str, int]]] = None
        self._masked_training_positives: dict[int, frozenset[int]] = {}
        self._training_positive_mask_initialized = False
        self._initialize_annotation_support(
            annotation_settings,
            enabled="molecule" in self.target_specs,
        )
        self._config = {
            "source": source,
            "normalization": resolved_normalization,
            "normalization_epsilon": self.normalization_epsilon,
            "target_specs": self.target_specs,
            "annotation_settings": self.get_annotation_settings().get_config(),
            "split": self.get_split_config(),
        }

    def create_partitions(self) -> Any:
        """Create partitions and deterministically hide configured train positives.

        The masking map is derived only after splitting, therefore validation and
        test samples retain their original annotations. It is keyed by immutable
        source spectrum identifiers, so raw batch loading and dense loading use
        exactly the same masked targets.

        :return: Cached dataset partitions.
        :rtype: Any
        """
        partitions = super().create_partitions()
        if not self._training_positive_mask_initialized:
            self._configure_training_positive_mask(partitions.train)
            self._training_positive_mask_initialized = True
        return partitions

    def _get_source_split_target(self, idx: int, target_field: str, **_: Any) -> Any:
        """Return one available single-label target for stratified splitting."""
        spec = self.target_specs.get(target_field)
        if spec is None:
            raise_validation_error(
                "PixelDataset", f"Unknown split target field '{target_field}'."
            )
        if spec["type"] != "single_label":
            # TODO: Add iterative stratification across multiple or multi-label targets.
            raise_validation_error(
                "PixelDataset", "Only one single-label split target is supported."
            )
        sample = self._target_sample(idx)
        if not bool(sample.masks[target_field].item()):
            return None
        return int(sample.values[target_field].item())

    def _get_source_split_mask(self, idx: int, mask: str, **_: Any) -> Any:
        """Return one named target-availability or annotation mask value."""
        if mask in self.target_specs:
            sample = self._target_sample(idx)
            return bool(sample.masks[mask].item())
        if mask == "annotated" and "molecule" in self.target_specs:
            return self.get_mapped_annotation_index().has_annotations(idx)
        annotation_reader = getattr(self.active_context, "annotation_reader", None)
        getter = getattr(annotation_reader, "get_spectrum_mask", None)
        if callable(getter):
            return getter(idx, mask)
        raise_validation_error(
            "PixelDataset", f"The active annotation reader does not expose mask '{mask}'."
        )

    def _get_source_split_group(self, idx: int, group_fields: Any, **_: Any) -> Any:
        """Return a metadata group key, including merged-source provenance."""
        annotation_reader = getattr(self.active_context, "annotation_reader", None)
        if annotation_reader is None:
            raise_validation_error("PixelDataset", "Grouped splitting requires annotations.")
        metadata = annotation_reader.get_spectrum_metadata(idx)
        nested = metadata.get("metadata", {}) if isinstance(metadata, Mapping) else {}
        fields = [group_fields] if isinstance(group_fields, str) else list(group_fields)
        values = tuple(metadata.get(field, nested.get(field)) for field in fields)
        if any(value is None or value == "" for value in values):
            return ("__ungrouped__", self._get_source_sample_id(idx))
        return values

    def _source_subset_groups(self, indices: range, **parameters: Any) -> list[Any]:
        """Return subset strata through a reader-level bulk metadata API."""
        indices = list(indices)
        annotation_reader = getattr(self.active_context, "annotation_reader", None)
        bulk_getter = getattr(annotation_reader, "get_spectrum_groups", None)
        if callable(bulk_getter):
            return list(bulk_getter(indices, **parameters))
        metadata = (
            annotation_reader.get_dataset_metadata()
            if annotation_reader is not None
            else {}
        )
        nested = metadata.get("metadata", {}) if isinstance(metadata, Mapping) else {}
        fields = parameters.get("group_fields") or (
            "source_dataset_id",
            "dataset_id",
            "image_key",
        )
        fields = [fields] if isinstance(fields, str) else list(fields)
        values = tuple(metadata.get(field, nested.get(field)) for field in fields)
        values = tuple(value for value in values if value is not None and value != "")
        return [values or ("__all_samples__",) for _ in indices]

    def _source_subset_multilabel_data(
        self,
        source_indices: list[int],
        **parameters: Any,
    ) -> tuple[list[Any], list[frozenset[int]]]:
        """Return image groups and sparse positive molecules for source rows.

        :param source_indices: Original source spectrum identifiers.
        :type source_indices: list[int]
        :param parameters: Grouping parameters forwarded to the annotation
            reader.
        :type parameters: Any
        :return: Aligned image-group keys and positive molecule class sets.
        :rtype: tuple[list[Any], list[frozenset[int]]]
        """
        groups = self._source_subset_groups(source_indices, **parameters)
        mapped_index = self.get_mapped_annotation_index()
        positive_labels = [
            frozenset(
                int(label)
                for label in mapped_index.annotation_indices[
                    mapped_index.entry_slice(source_index)
                ]
            )
            for source_index in source_indices
        ]
        return groups, positive_labels

    def get_multilabel_split_data(
        self,
        indices: Sequence[int],
        **parameters: Any,
    ) -> tuple[list[Any], list[frozenset[int]]]:
        """Return split metadata in the current public dataset index space.

        :param indices: Public dataset positions being split.
        :type indices: Sequence[int]
        :param parameters: Image grouping parameters.
        :type parameters: Any
        :return: Aligned image groups and sparse positive molecule class sets.
        :rtype: tuple[list[Any], list[frozenset[int]]]
        """
        source_indices = [self._source_index(int(index)) for index in indices]
        return self._source_subset_multilabel_data(source_indices, **parameters)

    def _source_length(self) -> int:
        """
        Retrieves total pixel coordinates count exposed by the underlying storage loader.

        :return: Absolute size of the total spatial image layout.
        :rtype: int
        :raises ValueError: If the attached active context reader session is unassigned.
        """
        # Session state verification
        if not self.active_context:
            raise_validation_error(
                context_name="PixelDataset",
                message="The active image context has no reader instance.",
            )
            
        return self.active_context.get_data_reader(self.source).GetNumberOfSpectra()

    def _get_source_item(self, idx: int) -> Tuple[Any, ...]:
        """
        Extracts and resolves a singular experimental spectrum onto the active target grid.

        :param idx: Flat position tracking coordinate index targeting an explicit single tissue pixel.
        :type idx: int
        :return: Spectrum identifier and tensor, optionally followed by target
            and availability-mask dictionaries.
        :rtype: Tuple[Any, ...]
        """
        # Context extraction layer
        reader = self.active_context.get_data_reader(self.source)

        # Read raw variables arrays from data drivers
        xs, ys = reader.GetSpectrum(idx)
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            raise_validation_error(
                context_name="PixelDataset",
                message=f"Spectrum {idx} contains non-finite input values.",
            )

        if self.source == "latent":
            values = np.array(ys, dtype=np.float32, copy=True)
            normalized_values = self._normalize(values)
            return self._sample(idx, torch.from_numpy(normalized_values))

        binner = self.active_context.binner

        # Transformation execution pipeline block
        ## Map and normalize one spectrum without hiding invalid reader output
        mapped_values = binner.transform_spectrum(xs, ys).cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(mapped_values)):
            raise_validation_error(
                context_name="PixelDataset",
                message=f"Binned spectrum {idx} contains non-finite values.",
            )
        ## Attach configured annotation targets after spectrum transformation
        return self._sample(idx, torch.from_numpy(self._normalize(mapped_values)))

    def _get_raw_source_item(self, idx: int) -> RawSpectrumSample:
        """Return one unbinned spectrum for packed CPU or CUDA preprocessing.

        :param idx: Stable spectrum identifier.
        :type idx: int
        :return: Raw m/z positions, intensities, and annotation targets.
        :rtype: RawSpectrumSample
        :raises ValidationError: If source values are non-finite or latent data
            is requested because latent inputs have no raw mass coordinates.
        """
        if self.source != "image":
            raise_validation_error(
                "PixelDataset", "Raw spectral preprocessing requires source='image'."
            )
        reader = self.active_context.get_data_reader(self.source)
        mass_values, intensities = reader.GetSpectrum(idx)
        mass_array = np.array(mass_values, dtype=np.float32, copy=True)
        intensity_array = np.array(intensities, dtype=np.float32, copy=True)
        if (
            mass_array.ndim != 1
            or intensity_array.shape != mass_array.shape
            or not np.all(np.isfinite(mass_array))
            or not np.all(np.isfinite(intensity_array))
        ):
            raise_validation_error(
                "PixelDataset", f"Raw spectrum {idx} is invalid or non-finite."
            )
        return RawSpectrumSample(
            sample_id=idx,
            mass_values=torch.as_tensor(mass_array, dtype=self.dtype),
            intensities=torch.as_tensor(intensity_array, dtype=self.dtype),
            targets=self._target_sample(idx),
        )

    def _get_raw_source_batch(
        self,
        indices: list[int],
    ) -> Any:
        """Read one complete raw batch through the fastest reader capability.

        :param indices: Dataset spectrum identifiers selected by the DataLoader.
        :type indices: list[int]
        :return: Shared-axis native batch or packed variable-length fallback.
        :rtype: SharedAxisRawBatch | RawSpectrumBatch
        """
        if self.source != "image":
            return [self._get_source_item(index) for index in indices]
        reader_batch = self.active_context.get_data_reader(self.source).GetSpectrumBatch(
            indices
        )
        target_samples = [self._target_sample(int(index)) for index in reader_batch.sample_ids]
        if reader_batch.shared_mass_axis:
            targets = self._collate_targets(target_samples)
            return SharedAxisRawBatch(
                sample_ids=torch.from_numpy(reader_batch.sample_ids),
                mass_axis=torch.as_tensor(reader_batch.mass_values, dtype=self.dtype),
                intensities=torch.as_tensor(reader_batch.intensities, dtype=self.dtype),
                targets=targets,
            )
        samples = [
            RawSpectrumSample(
                sample_id=int(sample_id),
                mass_values=torch.tensor(axis, dtype=self.dtype),
                intensities=torch.tensor(values, dtype=self.dtype),
                targets=target,
            )
            for sample_id, axis, values, target in zip(
                reader_batch.sample_ids,
                reader_batch.mass_values,
                reader_batch.intensities,
                target_samples,
            )
        ]
        return RawSpectrumCollator(self.get_target_schemas())(samples)

    def _collate_targets(self, samples: list[TargetSample]) -> TargetBatch:
        """Stack target values and masks already resolved for native I/O."""
        if not samples or not samples[0].values:
            return TargetBatch.empty()
        return TargetBatch(
            values={
                name: torch.stack([sample.values[name] for sample in samples])
                for name in samples[0].values
            },
            masks={
                name: torch.stack([sample.masks[name] for sample in samples])
                for name in samples[0].masks
            },
            schemas=self.get_target_schemas(),
        )

    def get_target_schemas(self) -> Dict[str, TargetSchema]:
        """Return immutable target definitions shared by every batch."""
        mappings = self.get_class_mappings() if self.target_specs else {}
        return {
            name: TargetSchema(
                name=name,
                target_type=spec["type"],
                class_names=tuple(
                    value
                    for value, _ in sorted(
                        mappings[name].items(), key=lambda item: item[1]
                    )
                ),
            )
            for name, spec in self.target_specs.items()
        }

    def get_class_mappings(self) -> Dict[str, Dict[str, int]]:
        """Return deterministic target mappings for the active annotation store.

        :return: Mapping from target field to semantic value-to-index mapping.
        :rtype: Dict[str, Dict[str, int]]
        :raises ValidationError: If targets are requested without an annotation
            reader in the active context.
        """
        if self._resolved_class_mappings is not None:
            return self._resolved_class_mappings
        annotation_reader = getattr(self.active_context, "annotation_reader", None)
        if annotation_reader is None:
            raise_validation_error(
                "PixelDataset",
                "target_specs require an annotation reader in the active context.",
            )
        metadata = annotation_reader.get_dataset_metadata()
        all_annotations = annotation_reader.get_annotations()
        molecule_index = self.get_mapped_annotation_index() if "molecule" in self.target_specs else None
        mappings: Dict[str, Dict[str, int]] = {}
        for field, spec in self.target_specs.items():
            values = (
                [
                    f"{formula}|{adduct}"
                    for formula, adduct in molecule_index.annotation_identities
                ]
                if field == "molecule" and molecule_index is not None
                else [molecule_key(annotation) for annotation in all_annotations]
                if field == "molecule"
                else metadata_values(metadata, field)
            )
            mappings[field] = build_class_mapping(values, spec.get("class_mapping"))
        self._resolved_class_mappings = mappings
        return mappings

    def _sample(self, spectrum_id: int, spectrum: torch.Tensor) -> Tuple[Any, ...]:
        """Attach configured targets and per-target availability masks.

        :param spectrum_id: Stable spectrum identifier used by annotation readers.
        :type spectrum_id: int
        :param spectrum: Transformed spectrum tensor.
        :type spectrum: torch.Tensor
        :return: Sample tuple with optional target and mask dictionaries.
        :rtype: Tuple[Any, ...]
        """
        target_sample = self._target_sample(spectrum_id)
        if not target_sample.values:
            return spectrum_id, spectrum
        return spectrum_id, spectrum, target_sample.values, target_sample.masks

    def _target_sample(self, spectrum_id: int) -> TargetSample:
        """Build target tensors once for one raw or already-binned sample."""
        if not self.target_specs:
            return TargetSample.empty()
        annotation_reader = self.active_context.annotation_reader
        mappings = self.get_class_mappings()
        metadata = (
            annotation_reader.get_spectrum_metadata(spectrum_id)
            if any(field != "molecule" for field in self.target_specs)
            else {}
        )
        targets: Dict[str, torch.Tensor] = {}
        target_masks: Dict[str, torch.Tensor] = {}
        for field, spec in self.target_specs.items():
            mapping = mappings[field]
            if field == "molecule":
                target = torch.zeros(len(mapping), dtype=torch.float32)
                molecule_index = self.get_mapped_annotation_index()
                identities = (
                    molecule_index.identities_for_spectrum(spectrum_id)
                    if molecule_index is not None
                    else (
                        (
                            str(annotation.get("formula", "")),
                            str(annotation.get("adduct", "")),
                        )
                        for annotation in annotation_reader.get_spectrum_annotations(
                            spectrum_id
                        )
                    )
                )
                for formula, adduct in identities:
                    class_index = mapping.get(f"{formula}|{adduct}")
                    if class_index is not None:
                        target[class_index] = 1.0
                masked_classes = self._masked_training_positives.get(spectrum_id)
                if masked_classes:
                    target[list(masked_classes)] = 0.0
                targets[field] = target
                target_policy = self.get_annotation_target_settings(field)
                spectrum_has_annotation = molecule_index.has_annotations(spectrum_id)
                if (
                    target_policy.empty_spectrum_policy == "reconstruction_only"
                    and not spectrum_has_annotation
                ):
                    target_masks[field] = torch.zeros(len(mapping), dtype=torch.bool)
                elif target_policy.unobserved_label_policy == "masked":
                    target_masks[field] = target.bool()
                else:
                    # BCE interprets target zero as negative; PU interprets it as
                    # unlabelled. The dataset records availability, while the loss
                    # owns the statistical interpretation.
                    target_masks[field] = torch.ones(len(mapping), dtype=torch.bool)
                continue
            values = metadata_values(metadata, field)
            if spec["type"] == "multi_label":
                target = torch.zeros(len(mapping), dtype=torch.float32)
                for value in values:
                    class_index = mapping.get(value)
                    if class_index is not None:
                        target[class_index] = 1.0
                targets[field] = target
                target_masks[field] = torch.tensor(bool(values))
                continue
            if len(values) > 1:
                raise_validation_error(
                    "PixelDataset",
                    f"Target '{field}' is ambiguous for spectrum {spectrum_id}.",
                )
            class_index = mapping.get(values[0]) if values else None
            if not values:
                targets[field] = torch.tensor(0, dtype=torch.long)
                target_masks[field] = torch.tensor(False)
                continue
            if class_index is None:
                raise_validation_error(
                    "PixelDataset",
                    f"Target '{field}' value '{values[0]}' has no class mapping.",
                )
            targets[field] = torch.tensor(class_index, dtype=torch.long)
            target_masks[field] = torch.tensor(True)
        return TargetSample(values=targets, masks=target_masks)

    def _configure_training_positive_mask(self, train_partition: Any) -> None:
        """Materialize one reproducible positive-label mask for the train split.

        Each molecular class independently hides a fixed fraction of its observed
        train positives. Per-class sampling preserves a usable evaluation target
        for common and rare classes while the fixed seed makes paired BCE/nnPU
        campaigns reproduce the identical masking protocol.

        :param train_partition: Train dataset view produced by the splitter.
        :type train_partition: Any
        :return: None.
        :rtype: None
        """
        settings = self.get_annotation_target_settings("molecule")
        fraction = settings.train_positive_mask_fraction
        if fraction <= 0.0 or "molecule" not in self.target_specs:
            return

        indices = list(getattr(train_partition, "indices", range(len(train_partition))))
        candidates_by_class: dict[int, list[int]] = defaultdict(list)
        for public_index in indices:
            spectrum_id = self._source_index(int(public_index))
            sample = self._target_sample(spectrum_id)
            target = sample.values["molecule"]
            for class_index in torch.nonzero(target > 0.5, as_tuple=False).flatten().tolist():
                candidates_by_class[int(class_index)].append(spectrum_id)

        random_generator = random.Random(settings.train_positive_mask_seed)
        selected_by_spectrum: dict[int, set[int]] = defaultdict(set)
        for class_index, spectrum_ids in candidates_by_class.items():
            selected_count = int(round(len(spectrum_ids) * fraction))
            selected_count = min(selected_count, len(spectrum_ids))
            for spectrum_id in random_generator.sample(spectrum_ids, selected_count):
                selected_by_spectrum[spectrum_id].add(class_index)
        self._masked_training_positives = {
            spectrum_id: frozenset(class_indices)
            for spectrum_id, class_indices in selected_by_spectrum.items()
        }
        logger.info(
            "Applied train-only positive mask: fraction=%s, seed=%s, masked_entries=%s.",
            fraction,
            settings.train_positive_mask_seed,
            sum(len(class_indices) for class_indices in self._masked_training_positives.values()),
        )

    def get_target_batch(self, indices: Any) -> TargetBatch:
        """Resolve targets for many public indices without reading spectra.

        :param indices: Public dataset indices in the requested order.
        :type indices: Iterable[int]
        :return: Collated values, per-class masks, and schemas.
        :rtype: TargetBatch
        """
        samples = [
            self._target_sample(self._source_index(int(index)))
            for index in indices
        ]
        return self._collate_targets(samples)

    def normalize_batch(self, spectra: torch.Tensor) -> torch.Tensor:
        """Normalize dense spectra independently along their feature dimension."""
        if self.normalization == "none":
            return spectra
        absolute = spectra.abs()
        if self.normalization == "tic":
            denominator = absolute.sum(dim=1, keepdim=True)
        elif self.normalization == "max":
            denominator = absolute.amax(dim=1, keepdim=True)
        else:
            denominator = torch.linalg.vector_norm(spectra, dim=1, keepdim=True)
        valid = denominator > self.normalization_epsilon
        safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
        normalized = spectra / safe_denominator
        return torch.where(valid, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _validate_target_specs(
        target_specs: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Validate and copy configurable target definitions.

        :param target_specs: Target definitions keyed by annotation field.
        :type target_specs: Mapping[str, Mapping[str, Any]]
        :return: Independent normalized target definition mapping.
        :rtype: Dict[str, Dict[str, Any]]
        :raises ValidationError: If a target type or molecule definition is invalid.
        """
        validated: Dict[str, Dict[str, Any]] = {}
        for field, raw_spec in target_specs.items():
            spec = dict(raw_spec)
            target_type = spec.get("type")
            if target_type not in {"single_label", "multi_label"}:
                raise_validation_error(
                    "PixelDataset",
                    f"Target '{field}' type must be 'single_label' or 'multi_label'.",
                )
            if field == "molecule" and target_type != "multi_label":
                raise_validation_error(
                    "PixelDataset", "Target 'molecule' must be multi_label."
                )
            mapping = spec.get("class_mapping")
            if mapping is not None:
                spec["class_mapping"] = {
                    str(value): int(index) for value, index in mapping.items()
                }
                indices = sorted(spec["class_mapping"].values())
                if indices != list(range(len(indices))):
                    raise_validation_error(
                        "PixelDataset",
                        (
                            f"Target '{field}' class_mapping indices must be "
                            "unique and contiguous from zero."
                        ),
                    )
            validated[str(field)] = spec
        return validated

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Return one spectrum using the configured stable scale.

        :param values: Binned image intensities or latent components.
        :type values: numpy.ndarray
        :return: Float32 spectrum with the configured normalization.
        :rtype: numpy.ndarray
        """
        if self.normalization == "none":
            return values
        if self.normalization == "tic":
            denominator = float(np.sum(np.abs(values), dtype=np.float32))
        elif self.normalization == "max":
            denominator = float(np.max(np.abs(values), initial=0.0))
        else:
            denominator = float(np.linalg.norm(values))
        if denominator <= self.normalization_epsilon:
            return np.zeros_like(values)
        return values / denominator
